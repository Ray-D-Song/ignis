/**
 * Ignis Core WebGPU Inference Engine (Phase 4: Fused Mega-Kernels & Static Single-Submit Decode)
 */

import {
  ensureOpfsModel,
  OpfsEmbeddingTable,
  readOpfsFileBytes,
  uploadOpfsShardsToGpu,
} from './opfs-loader.mjs';
import { buildFusedGluWgsl } from '../kernels/codegen.mjs';

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
  constructor(device, manifest, gpuShards, embdSource, pipelines, {
    subgroupSizeGuess = 32,
    maxSeqLen = 512,
    maxBatchTokens = 64,
    cacheHits = 0,
    cacheMisses = 0,
    hasTimestampQuery = false,
  } = {}) {
    this.device = device;
    this.manifest = manifest;
    this.bp = manifest.blueprint;
    this.gpuShards = gpuShards;
    this.embdSource = embdSource;
    this.embdView = embdSource instanceof Uint8Array
      ? new DataView(embdSource.buffer, embdSource.byteOffset, embdSource.byteLength)
      : null;
    this.pipelines = pipelines;
    this.subgroupSizeGuess = subgroupSizeGuess;
    this.maxSeqLen = maxSeqLen;
    this.maxBatchTokens = maxBatchTokens;
    this.cacheHits = cacheHits;
    this.cacheMisses = cacheMisses;
    this.hasTimestampQuery = hasTimestampQuery;
    this.curPos = 0;
    this.staticUniforms = [];

    this._allocateWorkBuffers();
    this._buildStaticDecodePlan();
  }

  static async _initDeviceAndPipelines(kernelBaseUrl, hiddenAct = 'gelu') {
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
    const hasTimestampQuery = adapter.features.has('timestamp-query');
    const requiredFeatures = ['shader-f16', 'subgroups'];
    if (hasTimestampQuery) requiredFeatures.push('timestamp-query');

    const subgroupSizeGuess = adapter.info?.subgroupMinSize || 32;
    const device = await adapter.requestDevice({
      requiredFeatures,
      requiredLimits: {
        maxStorageBufferBindingSize: adapter.limits.maxStorageBufferBindingSize,
        maxBufferSize: adapter.limits.maxBufferSize,
        maxComputeWorkgroupStorageSize: adapter.limits.maxComputeWorkgroupStorageSize,
      },
    });
    device.addEventListener('uncapturederror', (ev) => {
      console.error('[WebGPU uncapturederror]', ev.error?.message || ev.error);
    });

    const shaderFiles = {
      rmsnorm: 'rmsnorm.wgsl',
      quant_matmul: 'quant_matmul.wgsl',
      linear_f32: 'linear_f32.wgsl',
      rope_kv: 'rope_kv.wgsl',
      attention: 'attention.wgsl',
      attention_decode: 'attention_decode.wgsl',
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

    const gluCode = buildFusedGluWgsl(hiddenAct);
    const gluMod = device.createShaderModule({ code: gluCode, label: `codegen_glu_${hiddenAct}` });
    const gluInfo = await gluMod.getCompilationInfo();
    const gluErrs = gluInfo.messages.filter((m) => m.type === 'error');
    if (gluErrs.length > 0) {
      throw new Error(`WGSL codegen error (${hiddenAct}): ${gluErrs.map((e) => `L${e.lineNum}:${e.message}`).join('; ')}`);
    }
    modules.fused_glu_q4 = gluMod;

    const attnDecodeBgl = device.createBindGroupLayout({
      entries: [
        { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' } },
        { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
        { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
        { binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
        { binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
        { binding: 5, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
        { binding: 6, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
      ],
    });
    const attnDecodeLayout = device.createPipelineLayout({ bindGroupLayouts: [attnDecodeBgl] });

    const pipelines = {
      rmsnorm: device.createComputePipeline({ layout: 'auto', compute: { module: modules.rmsnorm, entryPoint: 'main' } }),
      gemv: device.createComputePipeline({ layout: 'auto', compute: { module: modules.quant_matmul, entryPoint: 'gemv_q4_g64' } }),
      gemvAdd: device.createComputePipeline({ layout: 'auto', compute: { module: modules.quant_matmul, entryPoint: 'gemv_q4_g64_add' } }),
      gemm4: device.createComputePipeline({ layout: 'auto', compute: { module: modules.quant_matmul, entryPoint: 'gemm4_q4_g64' } }),
      gemm4Add: device.createComputePipeline({ layout: 'auto', compute: { module: modules.quant_matmul, entryPoint: 'gemm4_q4_g64_add' } }),
      gegluGemv: device.createComputePipeline({ layout: 'auto', compute: { module: modules.fused_glu_q4, entryPoint: 'gemv_glu_q4_g64' } }),
      gegluGemm4: device.createComputePipeline({ layout: 'auto', compute: { module: modules.fused_glu_q4, entryPoint: 'gemm4_glu_q4_g64' } }),
      linearF32: device.createComputePipeline({ layout: 'auto', compute: { module: modules.linear_f32, entryPoint: 'linear_f32' } }),
      ropeKV: device.createComputePipeline({ layout: 'auto', compute: { module: modules.rope_kv, entryPoint: 'main' } }),
      attention: device.createComputePipeline({ layout: 'auto', compute: { module: modules.attention, entryPoint: 'main' } }),
      attnSplitDecode: device.createComputePipeline({ layout: attnDecodeLayout, compute: { module: modules.attention_decode, entryPoint: 'split_kv_gqa' } }),
      attnReduceDecode: device.createComputePipeline({ layout: attnDecodeLayout, compute: { module: modules.attention_decode, entryPoint: 'reduce_kv_gqa' } }),
      attnDecodeBgl,
    };

    return { device, pipelines, subgroupSizeGuess, hasTimestampQuery };
  }

  static async create(modelBaseUrl, kernelBaseUrl, { maxSeqLen = 512, maxBatchTokens = 64 } = {}) {
    const manifestResp = await fetch(`${modelBaseUrl}/manifest.json`);
    if (!manifestResp.ok) throw new Error(`Failed to fetch manifest.json: ${manifestResp.status}`);
    const manifest = await manifestResp.json();
    const { device, pipelines, subgroupSizeGuess, hasTimestampQuery } = await IgnisEngine._initDeviceAndPipelines(
      kernelBaseUrl,
      manifest.blueprint.hidden_act || 'gelu'
    );

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

    return new IgnisEngine(device, manifest, gpuShards, embdBytes, pipelines, {
      subgroupSizeGuess,
      maxSeqLen,
      maxBatchTokens,
      hasTimestampQuery,
    });
  }

  static async createWithOpfs(modelBaseUrl, modelKey, kernelBaseUrl, {
    maxSeqLen = 512,
    maxBatchTokens = 64,
    onProgress = null,
  } = {}) {
    const opfsRes = await ensureOpfsModel(modelBaseUrl, modelKey, { onProgress });
    const { dirHandle, manifest, cacheHits, cacheMisses, hasTokenizer } = opfsRes;
    const { device, pipelines, subgroupSizeGuess, hasTimestampQuery } = await IgnisEngine._initDeviceAndPipelines(
      kernelBaseUrl,
      manifest.blueprint.hidden_act || 'gelu'
    );
    const gpuShards = await uploadOpfsShardsToGpu(device, dirHandle, manifest);
    const embdTable = await OpfsEmbeddingTable.open(dirHandle, manifest.tensors.token_embd);
    const tokenizerBytes = hasTokenizer ? await readOpfsFileBytes(dirHandle, 'tokenizer.json') : null;

    const engine = new IgnisEngine(device, manifest, gpuShards, embdTable, pipelines, {
      subgroupSizeGuess,
      maxSeqLen,
      maxBatchTokens,
      cacheHits,
      cacheMisses,
      hasTimestampQuery,
    });
    return { engine, tokenizerBytes, dirHandle };
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
      norm: mkBuf(T * D * 4),
      qkv: mkBuf(T * qkvDim * 4),
      qRot: mkBuf(T * qDim * 4),
      attnGate: mkBuf(T * bp.num_attention_heads * 4),
      attnSplit: mkBuf(bp.num_attention_heads * 16 * (bp.head_dim + 2) * 4),
      attnOut: mkBuf(T * qDim * 4),
      ffnAct: mkBuf(T * FF * 4),
      logits: mkBuf(bp.vocab_size * 4),
      dummyGate: mkBuf(16),
    };

    if (this.hasTimestampQuery) {
      this.tsQuerySet = this.device.createQuerySet({ type: 'timestamp', count: 2 });
      this.tsResolveBuf = this.device.createBuffer({ size: 16, usage: GPUBufferUsage.QUERY_RESOLVE | GPUBufferUsage.COPY_SRC });
      this.tsReadBuf = this.device.createBuffer({ size: 16, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST });
    }

    // Hybrid KV Cache (f16)
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

  _makeStaticUniform(byteLen, setupFn) {
    const ab = new ArrayBuffer(byteLen);
    setupFn(new Uint32Array(ab), new Float32Array(ab));
    const buf = this.device.createBuffer({
      size: alignUp(byteLen, 16),
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });
    this.device.queue.writeBuffer(buf, 0, ab);
    this.staticUniforms.push(buf);
    return buf;
  }

  _tensorBinding(name) {
    const t = this.manifest.tensors[name];
    if (!t) throw new Error(`Tensor not found in manifest: ${name}`);
    const buffer = this.gpuShards.get(t.shard);
    if (!buffer) throw new Error(`GPU shard not loaded: ${t.shard}`);
    return { buffer, offset: t.offset, size: alignUp(t.bytes, 4), meta: t };
  }

  _buildStaticDecodePlan() {
    const bp = this.bp;
    const D = bp.hidden_size;
    const qDim = bp.num_attention_heads * bp.head_dim;
    const kvDim = bp.num_key_value_heads * bp.head_dim;
    const p = this.pipelines;

    // Shared immutable Uniforms for T=1 decode
    const uNorm = this._makeStaticUniform(16, (u32, f32) => {
      u32[0] = D;
      u32[1] = 1;
      f32[2] = bp.rms_norm_eps;
      u32[3] = 0;
    });

    const matmulUniforms = new Map();
    const getMatmulUniform = (M, K, K_pad) => {
      const key = `${M}_${K}_${K_pad}`;
      let u = matmulUniforms.get(key);
      if (!u) {
        u = this._makeStaticUniform(16, (u32) => {
          u32[0] = M; u32[1] = K; u32[2] = 1; u32[3] = K_pad;
        });
        matmulUniforms.set(key, u);
      }
      return u;
    };

    const uGate = bp.has_attn_gate
      ? this._makeStaticUniform(16, (u32) => { u32[0] = bp.num_attention_heads; u32[1] = D; u32[2] = 1; u32[3] = 0; })
      : null;

    const swaCap = (bp.sliding_window || 0) + this.maxBatchTokens;
    const makeRopeUniform = (isSwa) => this._makeStaticUniform(48, (u32, f32) => {
      u32[0] = 1;
      u32[1] = 0; // updated dynamically via writeBuffer(8 bytes)
      u32[2] = bp.num_attention_heads;
      u32[3] = bp.num_key_value_heads;
      u32[4] = bp.head_dim;
      u32[5] = isSwa ? bp.rope_dim_swa : bp.rope_dim_full;
      u32[6] = isSwa ? swaCap : this.maxSeqLen;
      u32[7] = isSwa ? 1 : 0;
      f32[8] = isSwa ? bp.rope_theta_swa : bp.rope_theta_full;
    });

    const makeAttnUniform = (isSwa) => this._makeStaticUniform(48, (u32, f32) => {
      u32[0] = 1;
      u32[1] = 0; // updated dynamically via writeBuffer(8 bytes)
      u32[2] = bp.num_attention_heads;
      u32[3] = bp.num_key_value_heads;
      u32[4] = bp.head_dim;
      u32[5] = isSwa ? swaCap : this.maxSeqLen;
      u32[6] = bp.sliding_window || 0;
      u32[7] = isSwa ? 1 : 0;
      u32[8] = bp.has_attn_gate ? 1 : 0;
      f32[9] = 1.0 / Math.sqrt(bp.head_dim);
    });

    this.dynUniforms = {
      hasSwa: bp.layer_types.includes('swa'),
      hasFull: bp.layer_types.includes('full'),
      uRopeSwa: makeRopeUniform(true),
      uRopeFull: makeRopeUniform(false),
      uAttnSwa: makeAttnUniform(true),
      uAttnFull: makeAttnUniform(false),
    };

    const mkQ4Step = (pipe, weightName, xBuf, outBuf) => {
      const q = this._tensorBinding(`${weightName}.quants`);
      const s = this._tensorBinding(`${weightName}.scales`);
      const [M, K] = q.meta.orig_shape;
      const [, K_pad] = q.meta.padded_shape;
      const u = getMatmulUniform(M, K, K_pad);
      const bg = this.device.createBindGroup({
        layout: pipe.getBindGroupLayout(0),
        entries: [
          { binding: 0, resource: { buffer: u, size: 16 } },
          { binding: 1, resource: { buffer: q.buffer, offset: q.offset, size: q.size } },
          { binding: 2, resource: { buffer: s.buffer, offset: s.offset, size: s.size } },
          { binding: 3, resource: { buffer: xBuf } },
          { binding: 4, resource: { buffer: outBuf } },
        ],
      });
      return { pipe, bg, wgX: Math.ceil(M / 4), wgY: 1 };
    };

    const mkNormStep = (weightName) => {
      const w = this._tensorBinding(weightName);
      const bg = this.device.createBindGroup({
        layout: p.rmsnorm.getBindGroupLayout(0),
        entries: [
          { binding: 0, resource: { buffer: uNorm, size: 16 } },
          { binding: 1, resource: { buffer: this.bufs.x } },
          { binding: 2, resource: { buffer: w.buffer, offset: w.offset, size: w.size } },
          { binding: 3, resource: { buffer: this.bufs.norm } },
        ],
      });
      return { pipe: p.rmsnorm, bg, wgX: 1, wgY: 1 };
    };

    this.decodeSteps = [];
    for (let i = 0; i < bp.num_layers; i++) {
      const kv = this.kvCaches[i];
      const isSwa = kv.isSwa;

      // 1. attn_norm
      this.decodeSteps.push(mkNormStep(`blk.${i}.attn_norm`));
      // 2. QKV
      this.decodeSteps.push(mkQ4Step(p.gemv, `blk.${i}.attn_qkv`, this.bufs.norm, this.bufs.qkv));
      // 3. Optional attn_gate
      if (bp.has_attn_gate) {
        const gW = this._tensorBinding(`blk.${i}.attn_gate`);
        const bgG = this.device.createBindGroup({
          layout: p.linearF32.getBindGroupLayout(0),
          entries: [
            { binding: 0, resource: { buffer: uGate, size: 16 } },
            { binding: 1, resource: { buffer: gW.buffer, offset: gW.offset, size: gW.size } },
            { binding: 2, resource: { buffer: this.bufs.norm } },
            { binding: 3, resource: { buffer: this.bufs.attnGate } },
          ],
        });
        this.decodeSteps.push({ pipe: p.linearF32, bg: bgG, wgX: Math.ceil(bp.num_attention_heads / 4), wgY: 1 });
      }
      // 4. RoPE + KV write
      const bgRope = this.device.createBindGroup({
        layout: p.ropeKV.getBindGroupLayout(0),
        entries: [
          { binding: 0, resource: { buffer: isSwa ? this.dynUniforms.uRopeSwa : this.dynUniforms.uRopeFull, size: 48 } },
          { binding: 1, resource: { buffer: this.bufs.qkv } },
          { binding: 2, resource: { buffer: this.bufs.qRot } },
          { binding: 3, resource: { buffer: kv.k } },
          { binding: 4, resource: { buffer: kv.v } },
        ],
      });
      this.decodeSteps.push({ pipe: p.ropeKV, bg: bgRope, wgX: Math.ceil(Math.max(qDim, kvDim) / 128), wgY: 1 });
      // 5. Attention + Gate (Split-KV FlashDecoding when GQA group_size <= 4)
      const gqaGroup = bp.num_attention_heads / bp.num_key_value_heads;
      if (gqaGroup <= 4) {
        const bgAttnDec = this.device.createBindGroup({
          layout: p.attnDecodeBgl,
          entries: [
            { binding: 0, resource: { buffer: isSwa ? this.dynUniforms.uAttnSwa : this.dynUniforms.uAttnFull, size: 48 } },
            { binding: 1, resource: { buffer: this.bufs.qRot } },
            { binding: 2, resource: { buffer: kv.k } },
            { binding: 3, resource: { buffer: kv.v } },
            { binding: 4, resource: { buffer: bp.has_attn_gate ? this.bufs.attnGate : this.bufs.dummyGate } },
            { binding: 5, resource: { buffer: this.bufs.attnSplit } },
            { binding: 6, resource: { buffer: this.bufs.attnOut } },
          ],
        });
        this.decodeSteps.push({ pipe: p.attnSplitDecode, bg: bgAttnDec, wgX: bp.num_key_value_heads * 16, wgY: 1 });
        this.decodeSteps.push({ pipe: p.attnReduceDecode, bg: bgAttnDec, wgX: bp.num_attention_heads, wgY: 1 });
      } else {
        const bgAttn = this.device.createBindGroup({
          layout: p.attention.getBindGroupLayout(0),
          entries: [
            { binding: 0, resource: { buffer: isSwa ? this.dynUniforms.uAttnSwa : this.dynUniforms.uAttnFull, size: 48 } },
            { binding: 1, resource: { buffer: this.bufs.qRot } },
            { binding: 2, resource: { buffer: kv.k } },
            { binding: 3, resource: { buffer: kv.v } },
            { binding: 4, resource: { buffer: bp.has_attn_gate ? this.bufs.attnGate : this.bufs.dummyGate } },
            { binding: 5, resource: { buffer: this.bufs.attnOut } },
          ],
        });
        this.decodeSteps.push({ pipe: p.attention, bg: bgAttn, wgX: bp.num_attention_heads, wgY: 1 });
      }
      // 6. O proj + in-place Residual Add into x
      this.decodeSteps.push(mkQ4Step(p.gemvAdd, `blk.${i}.attn_output`, this.bufs.attnOut, this.bufs.x));
      // 7. FFN norm
      this.decodeSteps.push(mkNormStep(`blk.${i}.ffn_norm`));
      // 8. Fused GeGLU (gate + up + gelu * up)
      const gq = this._tensorBinding(`blk.${i}.ffn_gate.quants`);
      const gs = this._tensorBinding(`blk.${i}.ffn_gate.scales`);
      const uq = this._tensorBinding(`blk.${i}.ffn_up.quants`);
      const us = this._tensorBinding(`blk.${i}.ffn_up.scales`);
      const [ffM, ffK] = gq.meta.orig_shape;
      const [, ffKpad] = gq.meta.padded_shape;
      const uFfn = getMatmulUniform(ffM, ffK, ffKpad);
      const bgGeglu = this.device.createBindGroup({
        layout: p.gegluGemv.getBindGroupLayout(0),
        entries: [
          { binding: 0, resource: { buffer: uFfn, size: 16 } },
          { binding: 1, resource: { buffer: gq.buffer, offset: gq.offset, size: gq.size } },
          { binding: 2, resource: { buffer: gs.buffer, offset: gs.offset, size: gs.size } },
          { binding: 3, resource: { buffer: uq.buffer, offset: uq.offset, size: uq.size } },
          { binding: 4, resource: { buffer: us.buffer, offset: us.offset, size: us.size } },
          { binding: 5, resource: { buffer: this.bufs.norm } },
          { binding: 6, resource: { buffer: this.bufs.ffnAct } },
        ],
      });
      this.decodeSteps.push({ pipe: p.gegluGemv, bg: bgGeglu, wgX: Math.ceil(ffM / 4), wgY: 1 });
      // 9. FFN down + in-place Residual Add into x
      this.decodeSteps.push(mkQ4Step(p.gemvAdd, `blk.${i}.ffn_down`, this.bufs.ffnAct, this.bufs.x));
    }

    // Final norm + LM head
    this.decodeSteps.push(mkNormStep('output_norm'));
    this.decodeSteps.push(mkQ4Step(p.gemv, 'output', this.bufs.norm, this.bufs.logits));
  }

  reset() {
    this.curPos = 0;
  }

  async lookupEmbeddings(tokenIds) {
    if (this.embdSource instanceof OpfsEmbeddingTable) {
      return this.embdSource.lookupBatch(tokenIds);
    }
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
    u32OrF32Setup(new Uint32Array(ab), new Float32Array(ab));
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

  async _forwardDecodeFast(tokenId, { profileGpu = false } = {}) {
    const emb = await this.lookupEmbeddings([tokenId]);
    this.device.queue.writeBuffer(this.bufs.x, 0, emb);

    const posHeader = new Uint32Array([1, this.curPos]);
    if (this.dynUniforms.hasSwa) {
      this.device.queue.writeBuffer(this.dynUniforms.uRopeSwa, 0, posHeader);
      this.device.queue.writeBuffer(this.dynUniforms.uAttnSwa, 0, posHeader);
    }
    if (this.dynUniforms.hasFull) {
      this.device.queue.writeBuffer(this.dynUniforms.uRopeFull, 0, posHeader);
      this.device.queue.writeBuffer(this.dynUniforms.uAttnFull, 0, posHeader);
    }

    const enc = this.device.createCommandEncoder();
    const useTs = profileGpu && this.hasTimestampQuery;
    const steps = this.decodeSteps;
    for (let s = 0; s < steps.length; s++) {
      const step = steps[s];
      const passDesc = {};
      if (useTs && (s === 0 || s === steps.length - 1)) {
        passDesc.timestampWrites = {
          querySet: this.tsQuerySet,
          ...(s === 0 ? { beginningOfPassWriteIndex: 0 } : {}),
          ...(s === steps.length - 1 ? { endOfPassWriteIndex: 1 } : {}),
        };
      }
      const pass = enc.beginComputePass(passDesc);
      pass.setPipeline(step.pipe);
      pass.setBindGroup(0, step.bg);
      pass.dispatchWorkgroups(step.wgX, step.wgY, 1);
      pass.end();
    }
    if (useTs) {
      enc.resolveQuerySet(this.tsQuerySet, 0, 2, this.tsResolveBuf, 0);
      enc.copyBufferToBuffer(this.tsResolveBuf, 0, this.tsReadBuf, 0, 16);
    }
    this.device.queue.submit([enc.finish()]);

    const logits = await this._readBufferFloat32(this.bufs.logits, this.bp.vocab_size);
    let gpuMs = null;
    if (useTs) {
      await this.tsReadBuf.mapAsync(GPUMapMode.READ);
      const ts = new BigInt64Array(this.tsReadBuf.getMappedRange().slice(0));
      this.tsReadBuf.unmap();
      gpuMs = Number(ts[1] - ts[0]) / 1e6;
    }
    this.curPos += 1;
    return { logits, traces: {}, gpuMs };
  }

  _encodeMatmulQ4(enc, tempUniforms, weightBaseName, xBuf, xOffsetBytes, outBuf, T, addResidual = false) {
    const q = this._tensorBinding(`${weightBaseName}.quants`);
    const s = this._tensorBinding(`${weightBaseName}.scales`);
    const [M, K] = q.meta.orig_shape;
    const [, K_pad] = q.meta.padded_shape;
    const uBuf = this._makeUniform((u32) => { u32[0] = M; u32[1] = K; u32[2] = T; u32[3] = K_pad; });
    tempUniforms.push(uBuf);

    const isGemv = T === 1;
    const pipe = isGemv
      ? (addResidual ? this.pipelines.gemvAdd : this.pipelines.gemv)
      : (addResidual ? this.pipelines.gemm4Add : this.pipelines.gemm4);
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
    const pass = enc.beginComputePass();
    pass.setPipeline(pipe);
    pass.setBindGroup(0, bg);
    pass.dispatchWorkgroups(Math.ceil(M / 4), isGemv ? 1 : Math.ceil(T / 4), 1);
    pass.end();
  }

  _encodeFusedGegluQ4(enc, tempUniforms, layerIdx, xBuf, outBuf, T) {
    const gq = this._tensorBinding(`blk.${layerIdx}.ffn_gate.quants`);
    const gs = this._tensorBinding(`blk.${layerIdx}.ffn_gate.scales`);
    const uq = this._tensorBinding(`blk.${layerIdx}.ffn_up.quants`);
    const us = this._tensorBinding(`blk.${layerIdx}.ffn_up.scales`);
    const [M, K] = gq.meta.orig_shape;
    const [, K_pad] = gq.meta.padded_shape;
    const uBuf = this._makeUniform((u32) => { u32[0] = M; u32[1] = K; u32[2] = T; u32[3] = K_pad; });
    tempUniforms.push(uBuf);

    const isGemv = T === 1;
    const pipe = isGemv ? this.pipelines.gegluGemv : this.pipelines.gegluGemm4;
    const bg = this.device.createBindGroup({
      layout: pipe.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: uBuf, size: 16 } },
        { binding: 1, resource: { buffer: gq.buffer, offset: gq.offset, size: gq.size } },
        { binding: 2, resource: { buffer: gs.buffer, offset: gs.offset, size: gs.size } },
        { binding: 3, resource: { buffer: uq.buffer, offset: uq.offset, size: uq.size } },
        { binding: 4, resource: { buffer: us.buffer, offset: us.offset, size: us.size } },
        { binding: 5, resource: { buffer: xBuf } },
        { binding: 6, resource: { buffer: outBuf } },
      ],
    });
    const pass = enc.beginComputePass();
    pass.setPipeline(pipe);
    pass.setBindGroup(0, bg);
    pass.dispatchWorkgroups(Math.ceil(M / 4), isGemv ? 1 : Math.ceil(T / 4), 1);
    pass.end();
  }

  _encodeRMSNorm(enc, tempUniforms, xBuf, normWeightName, outBuf, T) {
    const w = this._tensorBinding(normWeightName);
    const uBuf = this._makeUniform((u32, f32) => {
      u32[0] = this.bp.hidden_size; u32[1] = T; f32[2] = this.bp.rms_norm_eps; u32[3] = 0;
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

  async forward(tokenIds, { captureTraces = false, profileGpu = false } = {}) {
    const T = tokenIds.length;
    if (T > this.maxBatchTokens) {
      throw new Error(`Batch ${T} exceeds maxBatchTokens ${this.maxBatchTokens}`);
    }
    if (T === 1 && !captureTraces) {
      return this._forwardDecodeFast(tokenIds[0], { profileGpu });
    }

    const startPos = this.curPos;
    const bp = this.bp;
    const D = bp.hidden_size;
    const qDim = bp.num_attention_heads * bp.head_dim;
    const kvDim = bp.num_key_value_heads * bp.head_dim;

    const emb = await this.lookupEmbeddings(tokenIds);
    this.device.queue.writeBuffer(this.bufs.x, 0, emb);

    const traces = {};
    if (captureTraces) {
      traces.embed_last = Array.from(emb.slice((T - 1) * D, T * D));
    }

    const tempUniforms = [];
    let enc = this.device.createCommandEncoder();

    for (let i = 0; i < bp.num_layers; i++) {
      const kv = this.kvCaches[i];
      const isSwa = kv.isSwa;
      const ropeDim = isSwa ? bp.rope_dim_swa : bp.rope_dim_full;
      const ropeTheta = isSwa ? bp.rope_theta_swa : bp.rope_theta_full;

      this._encodeRMSNorm(enc, tempUniforms, this.bufs.x, `blk.${i}.attn_norm`, this.bufs.norm, T);
      this._encodeMatmulQ4(enc, tempUniforms, `blk.${i}.attn_qkv`, this.bufs.norm, 0, this.bufs.qkv, T, false);

      if (bp.has_attn_gate) {
        const gW = this._tensorBinding(`blk.${i}.attn_gate`);
        const uGate = this._makeUniform((u32) => { u32[0] = bp.num_attention_heads; u32[1] = D; u32[2] = T; u32[3] = 0; });
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
        passG.dispatchWorkgroups(Math.ceil(bp.num_attention_heads / 4), T, 1);
        passG.end();
      }

      const uRope = this._makeUniform((u32, f32) => {
        u32[0] = T; u32[1] = startPos; u32[2] = bp.num_attention_heads; u32[3] = bp.num_key_value_heads;
        u32[4] = bp.head_dim; u32[5] = ropeDim; u32[6] = kv.capacity; u32[7] = isSwa ? 1 : 0;
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

      const uAttn = this._makeUniform((u32, f32) => {
        u32[0] = T; u32[1] = startPos; u32[2] = bp.num_attention_heads; u32[3] = bp.num_key_value_heads;
        u32[4] = bp.head_dim; u32[5] = kv.capacity; u32[6] = bp.sliding_window || 0; u32[7] = isSwa ? 1 : 0;
        u32[8] = bp.has_attn_gate ? 1 : 0; f32[9] = 1.0 / Math.sqrt(bp.head_dim);
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

      // Fused O_proj + in-place Residual Add
      this._encodeMatmulQ4(enc, tempUniforms, `blk.${i}.attn_output`, this.bufs.attnOut, 0, this.bufs.x, T, true);

      // FFN norm + Fused GeGLU + Fused Down_proj + in-place Residual Add
      this._encodeRMSNorm(enc, tempUniforms, this.bufs.x, `blk.${i}.ffn_norm`, this.bufs.norm, T);
      this._encodeFusedGegluQ4(enc, tempUniforms, i, this.bufs.norm, this.bufs.ffnAct, T);
      this._encodeMatmulQ4(enc, tempUniforms, `blk.${i}.ffn_down`, this.bufs.ffnAct, 0, this.bufs.x, T, true);

      if (captureTraces && (i === 0 || i === bp.num_layers - 1)) {
        this.device.queue.submit([enc.finish()]);
        const layerData = await this._readBufferFloat32(this.bufs.x, T * D);
        traces[`layer_${i}`] = Array.from(layerData.slice((T - 1) * D, T * D));
        enc = this.device.createCommandEncoder();
      }
    }

    this._encodeRMSNorm(enc, tempUniforms, this.bufs.x, 'output_norm', this.bufs.norm, T);
    this._encodeMatmulQ4(enc, tempUniforms, 'output', this.bufs.norm, (T - 1) * D * 4, this.bufs.logits, 1, false);
    this.device.queue.submit([enc.finish()]);

    if (captureTraces) {
      const fnData = await this._readBufferFloat32(this.bufs.norm, T * D);
      traces.final_norm = Array.from(fnData.slice((T - 1) * D, T * D));
    }

    const logits = await this._readBufferFloat32(this.bufs.logits, bp.vocab_size);
    this.curPos += T;
    for (const ub of tempUniforms) ub.destroy();
    return { logits, traces, gpuMs: null };
  }

  destroy() {
    if (this.embdSource instanceof OpfsEmbeddingTable) {
      this.embdSource.close();
    }
    for (const buf of this.staticUniforms) buf.destroy();
    for (const buf of this.gpuShards.values()) buf.destroy();
    for (const buf of Object.values(this.bufs)) buf.destroy();
    for (const kv of this.kvCaches) {
      kv.k.destroy();
      kv.v.destroy();
    }
    if (this.tsQuerySet) this.tsQuerySet.destroy();
    if (this.tsResolveBuf) this.tsResolveBuf.destroy();
    if (this.tsReadBuf) this.tsReadBuf.destroy();
    this.device.destroy();
  }
}
