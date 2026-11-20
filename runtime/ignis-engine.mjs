/**
 * Ignis Core WebGPU Inference Engine (Phase 2 Reference Pipeline)
 */

const IGNIS_MAGIC = 'IGNIS001';

function alignUp(v, align = 256) {
  return Math.ceil(v / align) * align;
}

function parseShardHeader(buf) {
  const view = new DataView(buf);
  const magic = String.fromCharCode(...new Uint8Array(buf, 0, 8));
  if (magic !== IGNIS_MAGIC) {
    throw new Error(`Invalid Ignis shard magic: ${magic}`);
  }
  const version = view.getUint32(8, true);
  const shardIndex = view.getUint32(12, true);
  const payloadOffset = Number(view.getBigUint64(16, true));
  const payloadBytes = Number(view.getBigUint64(24, true));
  return { version, shardIndex, payloadOffset, payloadBytes };
}

export class IgnisEngine {
  constructor(device, manifest, gpuShards, embdBytes, pipelines, subgroupSizeGuess = 32, maxSeqLen = 512, maxBatchTokens = 64) {
    this.device = device;
    this.manifest = manifest;
    this.bp = manifest.blueprint;
    this.gpuShards = gpuShards;
    this.embdBytes = embdBytes;
    this.embdView = new DataView(embdBytes.buffer, embdBytes.byteOffset, embdBytes.byteLength);
    this.pipelines = pipelines;
    this.subgroupSizeGuess = subgroupSizeGuess;
    this.maxSeqLen = maxSeqLen;
    this.maxBatchTokens = maxBatchTokens;
    this.curPos = 0;

    this._allocateWorkBuffers();
  }

  static async create(modelBaseUrl, kernelBaseUrl, { maxSeqLen = 512, maxBatchTokens = 64 } = {}) {
    if (!('gpu' in navigator)) {
      throw new Error('WebGPU not supported');
    }
    const adapter = await navigator.gpu.requestAdapter({ powerPreference: 'high-performance' });
    if (!adapter) {
      throw new Error('Failed to request WebGPU adapter');
    }
    if (!adapter.features.has('shader-f16') || !adapter.features.has('subgroups')) {
      throw new Error(`Required WebGPU features missing (shader-f16=${adapter.features.has('shader-f16')}, subgroups=${adapter.features.has('subgroups')})`);
    }
    const subgroupSizeGuess = adapter.info?.subgroupMinSize || 32;
    const device = await adapter.requestDevice({
      requiredFeatures: ['shader-f16', 'subgroups'],
      requiredLimits: {
        maxStorageBufferBindingSize: adapter.limits.maxStorageBufferBindingSize,
        maxBufferSize: adapter.limits.maxBufferSize,
        maxComputeWorkgroupStorageSize: adapter.limits.maxComputeWorkgroupStorageSize,
      },
    });

    const manifestResp = await fetch(`${modelBaseUrl}/manifest.json`);
    if (!manifestResp.ok) throw new Error(`Failed to fetch manifest.json: ${manifestResp.status}`);
    const manifest = await manifestResp.json();

    // Load shards
    const gpuShards = new Map();
    let embdBytes = null;
    for (const shardInfo of manifest.shards) {
      const resp = await fetch(`${modelBaseUrl}/${shardInfo.file}`);
      if (!resp.ok) throw new Error(`Failed to fetch ${shardInfo.file}: ${resp.status}`);
      const ab = await resp.arrayBuffer();
      const hdr = parseShardHeader(ab);
      if (shardInfo.id === 'embd') {
        embdBytes = new Uint8Array(ab);
      } else {
        const paddedSize = alignUp(ab.byteLength, 4);
        const gpuBuf = device.createBuffer({
          size: paddedSize,
          usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
        });
        device.queue.writeBuffer(gpuBuf, 0, ab);
        gpuShards.set(shardInfo.id, gpuBuf);
      }
      if (hdr.payloadBytes !== shardInfo.payload_bytes) {
        throw new Error(`Shard payload size mismatch for ${shardInfo.file}`);
      }
    }

    // Compile shaders
    const shaderFiles = {
      rmsnorm: 'rmsnorm.wgsl',
      quant_matmul: 'quant_matmul.wgsl',
      linear_f32: 'linear_f32.wgsl',
      rope_kv: 'rope_kv.wgsl',
      attention: 'attention.wgsl',
      elementwise: 'elementwise.wgsl',
    };
    const modules = {};
    for (const [key, fname] of Object.entries(shaderFiles)) {
      const code = await (await fetch(`${kernelBaseUrl}/${fname}`)).text();
      const mod = device.createShaderModule({ code, label: fname });
      const info = await mod.getCompilationInfo();
      const errs = info.messages.filter((m) => m.type === 'error');
      if (errs.length > 0) {
        throw new Error(`WGSL compile error in ${fname}: ${errs.map((e) => `L${e.lineNum}:${e.message}`).join('; ')}`);
      }
      modules[key] = mod;
    }

    const pipelines = {
      rmsnorm: device.createComputePipeline({ layout: 'auto', compute: { module: modules.rmsnorm, entryPoint: 'main' } }),
      gemv: device.createComputePipeline({ layout: 'auto', compute: { module: modules.quant_matmul, entryPoint: 'gemv_q4_g64' } }),
      gemm4: device.createComputePipeline({ layout: 'auto', compute: { module: modules.quant_matmul, entryPoint: 'gemm4_q4_g64' } }),
      linearF32: device.createComputePipeline({ layout: 'auto', compute: { module: modules.linear_f32, entryPoint: 'linear_f32' } }),
      ropeKV: device.createComputePipeline({ layout: 'auto', compute: { module: modules.rope_kv, entryPoint: 'main' } }),
      attention: device.createComputePipeline({ layout: 'auto', compute: { module: modules.attention, entryPoint: 'main' } }),
      geglu: device.createComputePipeline({ layout: 'auto', compute: { module: modules.elementwise, entryPoint: 'geglu' } }),
      addResidual: device.createComputePipeline({ layout: 'auto', compute: { module: modules.elementwise, entryPoint: 'add_residual' } }),
    };

    return new IgnisEngine(device, manifest, gpuShards, embdBytes, pipelines, subgroupSizeGuess, maxSeqLen, maxBatchTokens);
  }

  _allocateWorkBuffers() {
    const bp = this.bp;
    const T = this.maxBatchTokens;
    const D = bp.hidden_size;
    const FF = bp.intermediate_size;
    const qDim = bp.num_attention_heads * bp.head_dim;
    const kvDim = bp.num_key_value_heads * bp.head_dim;
    const qkvDim = qDim + 2 * kvDim;

    const mkBuf = (bytes, usage = GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST) =>
      this.device.createBuffer({ size: alignUp(bytes, 16), usage });

    this.bufs = {
      x: mkBuf(T * D * 4),
      xNext: mkBuf(T * D * 4),
      norm: mkBuf(T * D * 4),
      qkv: mkBuf(T * qkvDim * 4),
      qRot: mkBuf(T * qDim * 4),
      attnGate: mkBuf(T * bp.num_attention_heads * 4),
      attnOut: mkBuf(T * qDim * 4),
      projOut: mkBuf(T * D * 4),
      ffnGate: mkBuf(T * FF * 4),
      ffnUp: mkBuf(T * FF * 4),
      ffnAct: mkBuf(T * FF * 4),
      logits: mkBuf(bp.vocab_size * 4), // last token logits
      // Dummy 16B gate buffer when has_attn_gate is false
      dummyGate: mkBuf(16),
    };

    // Hybrid KV Cache (f16 = 2 bytes per element)
    // SWA capacity = sliding_window + maxBatchTokens to prevent ring overwrite within a prefill batch
    this.kvCaches = [];
    for (let i = 0; i < bp.num_layers; i++) {
      const isSwa = bp.layer_types[i] === 'swa';
      const cap = isSwa ? (bp.sliding_window + this.maxBatchTokens) : this.maxSeqLen;
      const bytes = bp.num_key_value_heads * cap * bp.head_dim * 2;
      this.kvCaches.push({
        isSwa,
        capacity: cap,
        k: mkBuf(bytes, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST),
        v: mkBuf(bytes, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST),
      });
    }
  }

  reset() {
    this.curPos = 0;
  }

  _tensorBinding(name) {
    const t = this.manifest.tensors[name];
    if (!t) throw new Error(`Tensor not found in manifest: ${name}`);
    const buffer = this.gpuShards.get(t.shard);
    if (!buffer) throw new Error(`GPU shard not loaded: ${t.shard}`);
    return { buffer, offset: t.offset, size: alignUp(t.bytes, 4), meta: t };
  }

  lookupEmbeddings(tokenIds) {
    const em = this.manifest.tensors.token_embd;
    const D = this.bp.hidden_size;
    const stride = em.row_stride;
    const baseOff = em.offset;
    const out = new Float32Array(tokenIds.length * D);
    for (let t = 0; t < tokenIds.length; t++) {
      const id = tokenIds[t];
      const rowOff = baseOff + id * stride;
      const scale = this.embdView.getFloat32(rowOff, true);
      const dstBase = t * D;
      for (let d = 0; d < D; d++) {
        out[dstBase + d] = this.embdView.getInt8(rowOff + 4 + d) * scale;
      }
    }
    return out;
  }

  _makeUniform(u32OrF32Setup) {
    const ab = new ArrayBuffer(64);
    const u32 = new Uint32Array(ab);
    const f32 = new Float32Array(ab);
    u32OrF32Setup(u32, f32);
    const buf = this.device.createBuffer({
      size: 64,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });
    this.device.queue.writeBuffer(buf, 0, ab);
    return buf;
  }

  async _readBufferFloat32(buf, count) {
    const bytes = count * 4;
    const rb = this.device.createBuffer({
      size: bytes,
      usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
    });
    const enc = this.device.createCommandEncoder();
    enc.copyBufferToBuffer(buf, 0, rb, 0, bytes);
    this.device.queue.submit([enc.finish()]);
    await rb.mapAsync(GPUMapMode.READ);
    const res = new Float32Array(rb.getMappedRange().slice(0));
    rb.unmap();
    rb.destroy();
    return res;
  }

  _encodeMatmulQ4(enc, tempUniforms, weightBaseName, xBuf, xOffsetBytes, outBuf, T) {
    const q = this._tensorBinding(`${weightBaseName}.quants`);
    const s = this._tensorBinding(`${weightBaseName}.scales`);
    const [M, K] = q.meta.orig_shape;
    const [, K_pad] = q.meta.padded_shape;
    const uBuf = this._makeUniform((u32) => {
      u32[0] = M;
      u32[1] = K;
      u32[2] = T;
      u32[3] = K_pad;
    });
    tempUniforms.push(uBuf);

    const isGemv = T === 1;
    const pipe = isGemv ? this.pipelines.gemv : this.pipelines.gemm4;
    const bg = this.device.createBindGroup({
      layout: pipe.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: uBuf, size: 16 } },
        { binding: 1, resource: { buffer: q.buffer, offset: q.offset, size: q.size } },
        { binding: 2, resource: { buffer: s.buffer, offset: s.offset, size: s.size } },
        { binding: 3, resource: { buffer: xBuf, offset: xOffsetBytes } },
        { binding: 4, resource: { buffer: outBuf } },
      ],
    });
    const rowsPerWg = 4;
    const wgX = Math.ceil(M / rowsPerWg);
    const wgY = isGemv ? 1 : Math.ceil(T / 4);
    const pass = enc.beginComputePass();
    pass.setPipeline(pipe);
    pass.setBindGroup(0, bg);
    pass.dispatchWorkgroups(wgX, wgY, 1);
    pass.end();
  }

  _encodeRMSNorm(enc, tempUniforms, xBuf, normWeightName, outBuf, T) {
    const w = this._tensorBinding(normWeightName);
    const uBuf = this._makeUniform((u32, f32) => {
      u32[0] = this.bp.hidden_size;
      u32[1] = T;
      f32[2] = this.bp.rms_norm_eps;
      u32[3] = 0;
    });
    tempUniforms.push(uBuf);
    const bg = this.device.createBindGroup({
      layout: this.pipelines.rmsnorm.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: uBuf, size: 16 } },
        { binding: 1, resource: { buffer: xBuf } },
        { binding: 2, resource: { buffer: w.buffer, offset: w.offset, size: w.size } },
        { binding: 3, resource: { buffer: outBuf } },
      ],
    });
    const pass = enc.beginComputePass();
    pass.setPipeline(this.pipelines.rmsnorm);
    pass.setBindGroup(0, bg);
    pass.dispatchWorkgroups(T, 1, 1);
    pass.end();
  }

  async forward(tokenIds, { captureTraces = false } = {}) {
    const T = tokenIds.length;
    if (T > this.maxBatchTokens) {
      throw new Error(`Batch ${T} exceeds maxBatchTokens ${this.maxBatchTokens}`);
    }
    const startPos = this.curPos;
    const bp = this.bp;
    const D = bp.hidden_size;
    const FF = bp.intermediate_size;
    const qDim = bp.num_attention_heads * bp.head_dim;
    const kvDim = bp.num_key_value_heads * bp.head_dim;

    const emb = this.lookupEmbeddings(tokenIds);
    this.device.queue.writeBuffer(this.bufs.x, 0, emb);

    const traces = {};
    if (captureTraces) {
      traces.embed_last = Array.from(emb.slice((T - 1) * D, T * D));
    }

    const tempUniforms = [];
    for (let i = 0; i < bp.num_layers; i++) {
      const enc = this.device.createCommandEncoder();
      const kv = this.kvCaches[i];
      const isSwa = kv.isSwa;
      const ropeDim = isSwa ? bp.rope_dim_swa : bp.rope_dim_full;
      const ropeTheta = isSwa ? bp.rope_theta_swa : bp.rope_theta_full;

      // 1. attn_norm
      this._encodeRMSNorm(enc, tempUniforms, this.bufs.x, `blk.${i}.attn_norm`, this.bufs.norm, T);
      // 2. QKV proj
      this._encodeMatmulQ4(enc, tempUniforms, `blk.${i}.attn_qkv`, this.bufs.norm, 0, this.bufs.qkv, T);

      // 3. Optional attn_gate proj
      if (bp.has_attn_gate) {
        const gW = this._tensorBinding(`blk.${i}.attn_gate`);
        const uGate = this._makeUniform((u32) => {
          u32[0] = bp.num_attention_heads;
          u32[1] = D;
          u32[2] = T;
          u32[3] = 0;
        });
        tempUniforms.push(uGate);
        const bgGate = this.device.createBindGroup({
          layout: this.pipelines.linearF32.getBindGroupLayout(0),
          entries: [
            { binding: 0, resource: { buffer: uGate, size: 16 } },
            { binding: 1, resource: { buffer: gW.buffer, offset: gW.offset, size: gW.size } },
            { binding: 2, resource: { buffer: this.bufs.norm } },
            { binding: 3, resource: { buffer: this.bufs.attnGate } },
          ],
        });
        const passG = enc.beginComputePass();
        passG.setPipeline(this.pipelines.linearF32);
        passG.setBindGroup(0, bgGate);
        passG.dispatchWorkgroups(Math.ceil(bp.num_attention_heads / 64), T, 1);
        passG.end();
      }

      // 4. RoPE + KV Cache write
      const uRope = this._makeUniform((u32, f32) => {
        u32[0] = T;
        u32[1] = startPos;
        u32[2] = bp.num_attention_heads;
        u32[3] = bp.num_key_value_heads;
        u32[4] = bp.head_dim;
        u32[5] = ropeDim;
        u32[6] = kv.capacity;
        u32[7] = isSwa ? 1 : 0;
        f32[8] = ropeTheta;
      });
      tempUniforms.push(uRope);
      const bgRope = this.device.createBindGroup({
        layout: this.pipelines.ropeKV.getBindGroupLayout(0),
        entries: [
          { binding: 0, resource: { buffer: uRope, size: 48 } },
          { binding: 1, resource: { buffer: this.bufs.qkv } },
          { binding: 2, resource: { buffer: this.bufs.qRot } },
          { binding: 3, resource: { buffer: kv.k } },
          { binding: 4, resource: { buffer: kv.v } },
        ],
      });
      const passR = enc.beginComputePass();
      passR.setPipeline(this.pipelines.ropeKV);
      passR.setBindGroup(0, bgRope);
      passR.dispatchWorkgroups(Math.ceil(Math.max(qDim, kvDim) / 128), T, 1);
      passR.end();

      // 5. GQA Attention + Sigmoid Gate
      const uAttn = this._makeUniform((u32, f32) => {
        u32[0] = T;
        u32[1] = startPos;
        u32[2] = bp.num_attention_heads;
        u32[3] = bp.num_key_value_heads;
        u32[4] = bp.head_dim;
        u32[5] = kv.capacity;
        u32[6] = bp.sliding_window || 0;
        u32[7] = isSwa ? 1 : 0;
        u32[8] = bp.has_attn_gate ? 1 : 0;
        f32[9] = 1.0 / Math.sqrt(bp.head_dim);
      });
      tempUniforms.push(uAttn);
      const bgAttn = this.device.createBindGroup({
        layout: this.pipelines.attention.getBindGroupLayout(0),
        entries: [
          { binding: 0, resource: { buffer: uAttn, size: 48 } },
          { binding: 1, resource: { buffer: this.bufs.qRot } },
          { binding: 2, resource: { buffer: kv.k } },
          { binding: 3, resource: { buffer: kv.v } },
          { binding: 4, resource: { buffer: bp.has_attn_gate ? this.bufs.attnGate : this.bufs.dummyGate } },
          { binding: 5, resource: { buffer: this.bufs.attnOut } },
        ],
      });
      const passA = enc.beginComputePass();
      passA.setPipeline(this.pipelines.attention);
      passA.setBindGroup(0, bgAttn);
      passA.dispatchWorkgroups(bp.num_attention_heads, T, 1);
      passA.end();

      // 6. O proj + Residual
      this._encodeMatmulQ4(enc, tempUniforms, `blk.${i}.attn_output`, this.bufs.attnOut, 0, this.bufs.projOut, T);

      const uAdd1 = this._makeUniform((u32) => { u32[0] = T * D; });
      tempUniforms.push(uAdd1);
      const bgAdd1 = this.device.createBindGroup({
        layout: this.pipelines.addResidual.getBindGroupLayout(0),
        entries: [
          { binding: 0, resource: { buffer: uAdd1, size: 16 } },
          { binding: 1, resource: { buffer: this.bufs.x } },
          { binding: 2, resource: { buffer: this.bufs.projOut } },
          { binding: 3, resource: { buffer: this.bufs.xNext } },
        ],
      });
      const passAdd1 = enc.beginComputePass();
      passAdd1.setPipeline(this.pipelines.addResidual);
      passAdd1.setBindGroup(0, bgAdd1);
      passAdd1.dispatchWorkgroups(Math.ceil((T * D) / 256), 1, 1);
      passAdd1.end();

      // 7. FFN norm + GeGLU + Down + Residual
      this._encodeRMSNorm(enc, tempUniforms, this.bufs.xNext, `blk.${i}.ffn_norm`, this.bufs.norm, T);
      this._encodeMatmulQ4(enc, tempUniforms, `blk.${i}.ffn_gate`, this.bufs.norm, 0, this.bufs.ffnGate, T);
      this._encodeMatmulQ4(enc, tempUniforms, `blk.${i}.ffn_up`, this.bufs.norm, 0, this.bufs.ffnUp, T);

      const uGeglu = this._makeUniform((u32) => { u32[0] = T * FF; });
      tempUniforms.push(uGeglu);
      const bgGeglu = this.device.createBindGroup({
        layout: this.pipelines.geglu.getBindGroupLayout(0),
        entries: [
          { binding: 0, resource: { buffer: uGeglu, size: 16 } },
          { binding: 1, resource: { buffer: this.bufs.ffnGate } },
          { binding: 2, resource: { buffer: this.bufs.ffnUp } },
          { binding: 3, resource: { buffer: this.bufs.ffnAct } },
        ],
      });
      const passGeglu = enc.beginComputePass();
      passGeglu.setPipeline(this.pipelines.geglu);
      passGeglu.setBindGroup(0, bgGeglu);
      passGeglu.dispatchWorkgroups(Math.ceil((T * FF) / 256), 1, 1);
      passGeglu.end();

      this._encodeMatmulQ4(enc, tempUniforms, `blk.${i}.ffn_down`, this.bufs.ffnAct, 0, this.bufs.projOut, T);

      const uAdd2 = this._makeUniform((u32) => { u32[0] = T * D; });
      tempUniforms.push(uAdd2);
      const bgAdd2 = this.device.createBindGroup({
        layout: this.pipelines.addResidual.getBindGroupLayout(0),
        entries: [
          { binding: 0, resource: { buffer: uAdd2, size: 16 } },
          { binding: 1, resource: { buffer: this.bufs.xNext } },
          { binding: 2, resource: { buffer: this.bufs.projOut } },
          { binding: 3, resource: { buffer: this.bufs.x } },
        ],
      });
      const passAdd2 = enc.beginComputePass();
      passAdd2.setPipeline(this.pipelines.addResidual);
      passAdd2.setBindGroup(0, bgAdd2);
      passAdd2.dispatchWorkgroups(Math.ceil((T * D) / 256), 1, 1);
      passAdd2.end();

      this.device.queue.submit([enc.finish()]);

      if (captureTraces && (i === 0 || i === bp.num_layers - 1)) {
        const layerData = await this._readBufferFloat32(this.bufs.x, T * D);
        traces[`layer_${i}`] = Array.from(layerData.slice((T - 1) * D, T * D));
      }
    }

    // Final norm & LM Head for last token
    const encFinal = this.device.createCommandEncoder();
    this._encodeRMSNorm(encFinal, tempUniforms, this.bufs.x, 'output_norm', this.bufs.norm, T);
    // Notice output LM head is computed only on the last token (offset = (T-1) * D * 4)
    const lastOffset = alignUp((T - 1) * D * 4, 256);
    // Copy last token norm to offset 0 of xNext if unaligned (2560*4 = 10240 IS 256B aligned, but Tiny-Spark D=256*4=1024 is ALSO 256B aligned!).
    if ((T - 1) * D * 4 !== lastOffset) {
      encFinal.copyBufferToBuffer(this.bufs.norm, (T - 1) * D * 4, this.bufs.xNext, 0, D * 4);
      this._encodeMatmulQ4(encFinal, tempUniforms, 'output', this.bufs.xNext, 0, this.bufs.logits, 1);
    } else {
      this._encodeMatmulQ4(encFinal, tempUniforms, 'output', this.bufs.norm, (T - 1) * D * 4, this.bufs.logits, 1);
    }
    this.device.queue.submit([encFinal.finish()]);

    if (captureTraces) {
      const fnData = await this._readBufferFloat32(this.bufs.norm, T * D);
      traces.final_norm = Array.from(fnData.slice((T - 1) * D, T * D));
    }

    const logits = await this._readBufferFloat32(this.bufs.logits, bp.vocab_size);
    this.curPos += T;
    for (const ub of tempUniforms) ub.destroy();
    return { logits, traces };
  }
}
