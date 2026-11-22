/**
 * OPFS Persistent Model Cache & Zero-OOM Streaming GPU Uploader.
 * Uses FileSystemSyncAccessHandle inside Dedicated Web Workers for synchronous,
 * low-heap disk streaming and O(1) externalized Embedding row lookups.
 */

const CACHE_ROOT = 'ignis-models';
const DEFAULT_STAGING_BYTES = 16 * 1024 * 1024; // 16 MiB reusable staging buffer

function alignUp(v, align = 4) {
  return Math.ceil(v / align) * align;
}

async function getModelDir(modelKey) {
  const root = await navigator.storage.getDirectory();
  const ignisDir = await root.getDirectoryHandle(CACHE_ROOT, { create: true });
  return ignisDir.getDirectoryHandle(modelKey.replace(/[^a-zA-Z0-9._-]/g, '_'), { create: true });
}

async function readJsonFromDir(dirHandle, filename) {
  try {
    const fh = await dirHandle.getFileHandle(filename);
    const file = await fh.getFile();
    return JSON.parse(await file.text());
  } catch {
    return null;
  }
}

async function writeBytesToOpfs(dirHandle, filename, uint8OrBuffer) {
  const fh = await dirHandle.getFileHandle(filename, { create: true });
  if (typeof fh.createSyncAccessHandle === 'function') {
    const access = await fh.createSyncAccessHandle();
    try {
      access.truncate(0);
      const data = uint8OrBuffer instanceof Uint8Array ? uint8OrBuffer : new Uint8Array(uint8OrBuffer);
      let written = 0;
      while (written < data.byteLength) {
        written += access.write(data.subarray(written), { at: written });
      }
      access.flush();
    } finally {
      access.close();
    }
  } else {
    const writable = await fh.createWritable();
    await writable.write(uint8OrBuffer);
    await writable.close();
  }
}

async function streamUrlToOpfs(url, dirHandle, filename, expectedBytes, onChunk) {
  const resp = await fetch(url);
  if (!resp.ok) {
    throw new Error(`Failed to fetch ${url}: ${resp.status}`);
  }
  const fh = await dirHandle.getFileHandle(filename, { create: true });
  const reader = resp.body.getReader();
  const totalBytes = expectedBytes ?? (Number(resp.headers.get('content-length')) || 0);
  let lastEmitAt = 0;
  const emit = (offset, force = false) => {
    if (!onChunk) return;
    const now = performance.now();
    if (force || now - lastEmitAt >= 60) {
      lastEmitAt = now;
      onChunk(offset, totalBytes);
    }
  };

  if (typeof fh.createSyncAccessHandle === 'function') {
    const access = await fh.createSyncAccessHandle();
    try {
      access.truncate(0);
      let offset = 0;
      while (true) {
        const { done, value } = await reader.read();
        if (done) break;
        let written = 0;
        while (written < value.byteLength) {
          written += access.write(value.subarray(written), { at: offset + written });
        }
        offset += value.byteLength;
        emit(offset);
      }
      emit(offset, true);
      access.flush();
      if (expectedBytes != null && offset !== expectedBytes) {
        throw new Error(`Size mismatch streaming ${filename}: got ${offset}, expected ${expectedBytes}`);
      }
    } finally {
      access.close();
    }
  } else {
    const writable = await fh.createWritable();
    let offset = 0;
    try {
      while (true) {
        const { done, value } = await reader.read();
        if (done) break;
        await writable.write(value);
        offset += value.byteLength;
        emit(offset);
      }
      emit(offset, true);
      await writable.close();
    } catch (e) {
      await writable.abort();
      throw e;
    }
  }
}

export async function ensureOpfsModel(modelBaseUrl, modelKey, { onProgress } = {}) {
  const dirHandle = await getModelDir(modelKey);
  if (onProgress) {
    onProgress({
      phase: 'manifest',
      file: 'manifest.json',
      loaded: 0,
      total: 0,
      overallLoaded: 0,
      overallTotal: 0,
      fileIndex: 0,
      fileCount: 0,
      cached: false,
    });
  }
  const manifestResp = await fetch(`${modelBaseUrl}/manifest.json`);
  if (!manifestResp.ok) throw new Error(`Failed to fetch manifest.json: ${manifestResp.status}`);
  const manifestText = await manifestResp.text();
  const manifest = JSON.parse(manifestText);

  const cachedState = (await readJsonFromDir(dirHandle, '.cache-state.json')) || { shards: {} };
  let cacheHits = 0;
  let cacheMisses = 0;

  const fileCount = manifest.shards.length;
  const overallTotal = manifest.shards.reduce((sum, s) => sum + (s.bytes || 0), 0);
  let completedBytes = 0;

  for (let idx = 0; idx < fileCount; idx++) {
    const shard = manifest.shards[idx];
    const recorded = cachedState.shards[shard.file];
    let hit = false;
    if (recorded && recorded.sha256 === shard.sha256 && recorded.bytes === shard.bytes) {
      try {
        const fh = await dirHandle.getFileHandle(shard.file);
        const f = await fh.getFile();
        if (f.size === shard.bytes) {
          hit = true;
        }
      } catch {
        hit = false;
      }
    }
    if (hit) {
      cacheHits += 1;
      if (onProgress) {
        onProgress({
          phase: 'cache',
          file: shard.file,
          loaded: shard.bytes,
          total: shard.bytes,
          overallLoaded: completedBytes + shard.bytes,
          overallTotal,
          fileIndex: idx + 1,
          fileCount,
          cached: true,
        });
      }
    } else {
      cacheMisses += 1;
      await streamUrlToOpfs(`${modelBaseUrl}/${shard.file}`, dirHandle, shard.file, shard.bytes, (loaded, total) => {
        if (onProgress) {
          onProgress({
            phase: 'download',
            file: shard.file,
            loaded,
            total,
            overallLoaded: completedBytes + loaded,
            overallTotal,
            fileIndex: idx + 1,
            fileCount,
            cached: false,
          });
        }
      });
      cachedState.shards[shard.file] = { sha256: shard.sha256, bytes: shard.bytes };
    }
    completedBytes += shard.bytes || 0;
  }

  // Cache tokenizer.json if available
  let hasTokenizer = false;
  try {
    const tokFh = await dirHandle.getFileHandle('tokenizer.json');
    const tokFile = await tokFh.getFile();
    if (tokFile.size > 0 && cachedState.hasTokenizer) {
      hasTokenizer = true;
    }
  } catch {
    hasTokenizer = false;
  }
  if (onProgress) {
    onProgress({
      phase: 'tokenizer',
      file: 'tokenizer.json',
      loaded: hasTokenizer ? 1 : 0,
      total: 1,
      overallLoaded: overallTotal,
      overallTotal,
      fileIndex: fileCount,
      fileCount,
      cached: hasTokenizer,
    });
  }
  if (!hasTokenizer) {
    const tokResp = await fetch(`${modelBaseUrl}/tokenizer.json`);
    if (tokResp.ok) {
      const tokBytes = new Uint8Array(await tokResp.arrayBuffer());
      await writeBytesToOpfs(dirHandle, 'tokenizer.json', tokBytes);
      cachedState.hasTokenizer = true;
    }
    if (onProgress) {
      onProgress({
        phase: 'tokenizer',
        file: 'tokenizer.json',
        loaded: 1,
        total: 1,
        overallLoaded: overallTotal,
        overallTotal,
        fileIndex: fileCount,
        fileCount,
        cached: false,
      });
    }
  }

  await writeBytesToOpfs(dirHandle, 'manifest.json', new TextEncoder().encode(manifestText));
  await writeBytesToOpfs(dirHandle, '.cache-state.json', new TextEncoder().encode(JSON.stringify(cachedState)));

  return { dirHandle, manifest, cacheHits, cacheMisses, hasTokenizer: !!cachedState.hasTokenizer };
}

export async function readOpfsFileBytes(dirHandle, filename) {
  const fh = await dirHandle.getFileHandle(filename);
  const file = await fh.getFile();
  return new Uint8Array(await file.arrayBuffer());
}

export async function uploadOpfsShardsToGpu(
  device,
  dirHandle,
  manifest,
  stagingBytes = DEFAULT_STAGING_BYTES,
  onProgress = null
) {
  const gpuShards = new Map();
  const gpuShardsInfo = manifest.shards.filter((s) => typeof s.id === 'number');
  const totalGpuBytes = gpuShardsInfo.reduce((sum, s) => sum + (s.bytes || 0), 0);
  const fileCount = gpuShardsInfo.length;
  let gpuDoneBytes = 0;
  let lastGpuEmitAt = 0;
  const emitGpuProgress = (file, loaded, total, fileIndex, force = false) => {
    if (!onProgress) return;
    const now = performance.now();
    if (force || now - lastGpuEmitAt >= 60) {
      lastGpuEmitAt = now;
      onProgress({
        phase: 'gpu_upload',
        file,
        loaded,
        total,
        overallLoaded: gpuDoneBytes + loaded,
        overallTotal: totalGpuBytes,
        fileIndex,
        fileCount,
        cached: true,
      });
    }
  };

  const staging = device.createBuffer({
    size: stagingBytes,
    usage: GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
  });

  try {
    for (let idx = 0; idx < fileCount; idx++) {
      const sInfo = gpuShardsInfo[idx];
      const fh = await dirHandle.getFileHandle(sInfo.file);
      const gpuSize = alignUp(sInfo.bytes, 4);
      const gpuBuf = device.createBuffer({
        size: gpuSize,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
        label: sInfo.file,
      });

      if (typeof fh.createSyncAccessHandle === 'function') {
        const access = await fh.createSyncAccessHandle();
        try {
          const chunk = new Uint8Array(Math.min(stagingBytes, sInfo.bytes));
          let offset = 0;
          while (offset < sInfo.bytes) {
            const toRead = Math.min(chunk.byteLength, sInfo.bytes - offset);
            const sliceView = chunk.subarray(0, toRead);
            let readTotal = 0;
            while (readTotal < toRead) {
              const n = access.read(sliceView.subarray(readTotal), { at: offset + readTotal });
              if (n === 0) break;
              readTotal += n;
            }
            const alignedCopy = alignUp(readTotal, 4);
            // If readTotal wasn't multiple of 4 (last chunk), write padded subarray
            if (alignedCopy !== readTotal) {
              const padded = new Uint8Array(alignedCopy);
              padded.set(sliceView.subarray(0, readTotal));
              device.queue.writeBuffer(staging, 0, padded);
            } else {
              device.queue.writeBuffer(staging, 0, sliceView.subarray(0, readTotal));
            }
            const enc = device.createCommandEncoder();
            enc.copyBufferToBuffer(staging, 0, gpuBuf, offset, alignedCopy);
            device.queue.submit([enc.finish()]);
            offset += readTotal;
            emitGpuProgress(sInfo.file, offset, sInfo.bytes, idx + 1);
          }
          emitGpuProgress(sInfo.file, sInfo.bytes, sInfo.bytes, idx + 1, true);
        } finally {
          access.close();
        }
      } else {
        const file = await fh.getFile();
        let offset = 0;
        while (offset < sInfo.bytes) {
          const toRead = Math.min(stagingBytes, sInfo.bytes - offset);
          const ab = await file.slice(offset, offset + toRead).arrayBuffer();
          const alignedCopy = alignUp(ab.byteLength, 4);
          if (alignedCopy !== ab.byteLength) {
            const padded = new Uint8Array(alignedCopy);
            padded.set(new Uint8Array(ab));
            device.queue.writeBuffer(staging, 0, padded);
          } else {
            device.queue.writeBuffer(staging, 0, ab);
          }
          const enc = device.createCommandEncoder();
          enc.copyBufferToBuffer(staging, 0, gpuBuf, offset, alignedCopy);
          device.queue.submit([enc.finish()]);
          offset += ab.byteLength;
          emitGpuProgress(sInfo.file, offset, sInfo.bytes, idx + 1);
        }
        emitGpuProgress(sInfo.file, sInfo.bytes, sInfo.bytes, idx + 1, true);
      }
      gpuDoneBytes += sInfo.bytes || 0;
      gpuShards.set(sInfo.id, gpuBuf);
    }
  } finally {
    staging.destroy();
  }
  return gpuShards;
}

export class OpfsEmbeddingTable {
  constructor(accessHandle, fileObj, baseOffset, rowStride, hiddenSize, maxCacheRows = 4096) {
    this.access = accessHandle; // SyncAccessHandle inside Worker, or null
    this.file = fileObj;
    this.baseOffset = baseOffset;
    this.rowStride = rowStride;
    this.hiddenSize = hiddenSize;
    this.maxCacheRows = maxCacheRows;
    this.lru = new Map();
    this.scratch = new Uint8Array(rowStride);
    this.scratchView = new DataView(this.scratch.buffer);
  }

  static async open(dirHandle, embdMeta, maxCacheRows = 4096, filename = 'embd.ignis') {
    const fh = await dirHandle.getFileHandle(filename);
    if (typeof fh.createSyncAccessHandle === 'function') {
      const access = await fh.createSyncAccessHandle();
      return new OpfsEmbeddingTable(access, null, embdMeta.offset, embdMeta.row_stride, embdMeta.shape[1], maxCacheRows);
    }
    const file = await fh.getFile();
    return new OpfsEmbeddingTable(null, file, embdMeta.offset, embdMeta.row_stride, embdMeta.shape[1], maxCacheRows);
  }

  async lookupBatch(tokenIds) {
    const D = this.hiddenSize;
    const out = new Float32Array(tokenIds.length * D);
    for (let t = 0; t < tokenIds.length; t++) {
      const id = tokenIds[t];
      let cached = this.lru.get(id);
      if (cached) {
        // Refresh LRU order
        this.lru.delete(id);
        this.lru.set(id, cached);
      } else {
        const at = this.baseOffset + id * this.rowStride;
        if (this.access) {
          let read = 0;
          while (read < this.rowStride) {
            const n = this.access.read(this.scratch.subarray(read), { at: at + read });
            if (n === 0) break;
            read += n;
          }
        } else {
          const ab = await this.file.slice(at, at + this.rowStride).arrayBuffer();
          this.scratch.set(new Uint8Array(ab));
        }
        const scale = this.scratchView.getFloat32(0, true);
        cached = new Float32Array(D);
        for (let d = 0; d < D; d++) {
          cached[d] = this.scratchView.getInt8(4 + d) * scale;
        }
        if (this.lru.size >= this.maxCacheRows) {
          const oldestKey = this.lru.keys().next().value;
          this.lru.delete(oldestKey);
        }
        this.lru.set(id, cached);
      }
      out.set(cached, t * D);
    }
    return out;
  }

  close() {
    if (this.access) {
      this.access.close();
      this.access = null;
    }
    this.lru.clear();
  }
}
