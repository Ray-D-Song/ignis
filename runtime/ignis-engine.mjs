/**
 * Ignis Core WebGPU Inference Engine (Phase 5 & 6: Universal Blueprint Engine)
 *
 * Supports Spark-2.5, Llama/Qwen/Phi, LFM3-Omni (ShortConv + SWA + Full), Gemma 4 (PLE + KV-Sharing + head_dim<=512),
 * INT3_G64 / INT4_G64 / INT8_G64 weights, W3C-baseline Subgroup & No-Subgroups fallback kernels,
 * multimodal embedding injection (forwardEmbeds), batched verification (forwardVerify), and state rollback.
 */

import {
  ensureOpfsModel,
  OpfsEmbeddingTable,
  readOpfsFileBytes,
  uploadOpfsShardsToGpu,
} from './opfs-loader.mjs';
import {
  buildAttentionDecodeWgsl,
  buildAttentionWgsl,
  buildFusedGluWgsl,
  buildLinearF32Wgsl,
  buildPleWgsl,
  buildQkNormWgsl,
  buildQuantMatmulWgsl,
  buildRmsNormWgsl,
  buildRopeKvWgsl,
  buildShortConvWgsl,
} from '../kernels/codegen.mjs';

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

async function compileShader(device, code, label) {
  const mod = device.createShaderModule({ code, label });
  const info = await mod.getCompilationInfo();
  const errs = info.messages.filter((m) => m.type === 'error');
  if (errs.length > 0) {
    throw new Error(`WGSL compile error in ${label}: ${errs.map((e) => `L${e.lineNum}:${e.message}`).join('; ')}`);
  }
  return mod;
}

export class IgnisEngine {
  constructor(device, manifest, gpuShards, embdSource, pipelines, {
    subgroupSizeGuess = 32,
    useSubgroups = true,
    maxSeqLen = 512,
    maxBatchTokens = 64,
    cacheHits = 0,
    cacheMisses = 0,
    hasTimestampQuery = false,
    pleEmbdSource = null,
    dirHandle = null,
  } = {}) {
    this.device = device;
    this.manifest = manifest;
    this.bp = manifest.blueprint;
    this.gpuShards = gpuShards;
    this.embdSource = embdSource;
    this.pleEmbdSource = pleEmbdSource;
    this.dirHandle = dirHandle;
    this.embdView = embdSource instanceof Uint8Array
      ? new DataView(embdSource.buffer, embdSource.byteOffset, embdSource.byteLength)
      : null;
    this.pipelines = pipelines;
    this.subgroupSizeGuess = subgroupSizeGuess;
    this.useSubgroups = useSubgroups;
    this.maxSeqLen = maxSeqLen;
    this.maxBatchTokens = maxBatchTokens;
    this.cacheHits = cacheHits;
    this.cacheMisses = cacheMisses;
    this.hasTimestampQuery = hasTimestampQuery;
    this.curPos = 0;
    this.staticUniforms = [];
    this.dynPosUniforms = [];
    this.verifyLogitsBuf = null;
    this.verifyLogitsCap = 0;

    this._allocateWorkBuffers();
    this._buildStaticDecodePlan();
  }

  static async _initDeviceAndPipelines(kernelBaseUrl, hiddenActOrOpts = 'gelu') {
    const opts = typeof hiddenActOrOpts === 'string'
      ? { hiddenAct: hiddenActOrOpts }
      : (hiddenActOrOpts || {});
    const hiddenAct = opts.hiddenAct || 'gelu';
    const weightBits = opts.weightBits || 4;
    const forceNoSubgroups = !!opts.forceNoSubgroups;

    if (!('gpu' in navigator)) {
      throw new Error('WebGPU not supported');
    }
    const adapter = await navigator.gpu.requestAdapter({ powerPreference: 'high-performance' });
    if (!adapter) {
      throw new Error('Failed to request WebGPU adapter');
    }
    if (!adapter.features.has('shader-f16')) {
      throw new Error('Required WebGPU feature shader-f16 missing');
    }
    const useSubgroups = adapter.features.has('subgroups') && !forceNoSubgroups;
    const hasTimestampQuery = adapter.features.has('timestamp-query');
    const requiredFeatures = ['shader-f16'];
    if (useSubgroups) requiredFeatures.push('subgroups');
    if (hasTimestampQuery) requiredFeatures.push('timestamp-query');

    const subgroupSizeGuess = useSubgroups ? (adapter.info?.subgroupMinSize || 32) : 0;
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

    const mods = {
      rmsnorm: await compileShader(device, buildRmsNormWgsl({ useSubgroups }), 'codegen_rmsnorm'),
      quant_matmul: await compileShader(device, buildQuantMatmulWgsl({ useSubgroups }), 'codegen_quant_matmul'),
      linear_f32: await compileShader(device, buildLinearF32Wgsl({ useSubgroups }), 'codegen_linear_f32'),
      qk_norm: await compileShader(device, buildQkNormWgsl({ useSubgroups }), 'codegen_qk_norm'),
      rope_kv: await compileShader(device, buildRopeKvWgsl(), 'codegen_rope_kv'),
      attention: await compileShader(device, buildAttentionWgsl({ useSubgroups }), 'codegen_attention'),
      attention_decode: await compileShader(device, buildAttentionDecodeWgsl({ useSubgroups }), 'codegen_attention_decode'),
      short_conv: await compileShader(device, buildShortConvWgsl(), 'codegen_short_conv'),
      ple_ops: await compileShader(device, buildPleWgsl({ useSubgroups }), 'codegen_ple_ops'),
      fused_glu: await compileShader(device, buildFusedGluWgsl(hiddenAct, { useSubgroups, weightBits }), `codegen_glu_${hiddenAct}_q${weightBits}`),
    };

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

    const gluSuffix = weightBits === 3 ? 'q3_g64' : 'q4_g64';
    const pipelines = {
      rmsnorm: device.createComputePipeline({ layout: 'auto', compute: { module: mods.rmsnorm, entryPoint: 'main' } }),
      postNormResidual: device.createComputePipeline({ layout: 'auto', compute: { module: mods.rmsnorm, entryPoint: 'post_norm_residual' } }),
      gemv: device.createComputePipeline({ layout: 'auto', compute: { module: mods.quant_matmul, entryPoint: 'gemv_q4_g64' } }),
      gemvAdd: device.createComputePipeline({ layout: 'auto', compute: { module: mods.quant_matmul, entryPoint: 'gemv_q4_g64_add' } }),
      gemm4: device.createComputePipeline({ layout: 'auto', compute: { module: mods.quant_matmul, entryPoint: 'gemm4_q4_g64' } }),
      gemm4Add: device.createComputePipeline({ layout: 'auto', compute: { module: mods.quant_matmul, entryPoint: 'gemm4_q4_g64_add' } }),
      gemvQ3: device.createComputePipeline({ layout: 'auto', compute: { module: mods.quant_matmul, entryPoint: 'gemv_q3_g64' } }),
      gemvQ3Add: device.createComputePipeline({ layout: 'auto', compute: { module: mods.quant_matmul, entryPoint: 'gemv_q3_g64_add' } }),
      gemm4Q3: device.createComputePipeline({ layout: 'auto', compute: { module: mods.quant_matmul, entryPoint: 'gemm4_q3_g64' } }),
      gemm4Q3Add: device.createComputePipeline({ layout: 'auto', compute: { module: mods.quant_matmul, entryPoint: 'gemm4_q3_g64_add' } }),
      gemvQ8: device.createComputePipeline({ layout: 'auto', compute: { module: mods.quant_matmul, entryPoint: 'gemv_q8_g64' } }),
      gemm4Q8: device.createComputePipeline({ layout: 'auto', compute: { module: mods.quant_matmul, entryPoint: 'gemm4_q8_g64' } }),
      gegluGemv: device.createComputePipeline({ layout: 'auto', compute: { module: mods.fused_glu, entryPoint: `gemv_glu_${gluSuffix}` } }),
      gegluGemm4: device.createComputePipeline({ layout: 'auto', compute: { module: mods.fused_glu, entryPoint: `gemm4_glu_${gluSuffix}` } }),
      linearF32: device.createComputePipeline({ layout: 'auto', compute: { module: mods.linear_f32, entryPoint: 'linear_f32' } }),
      qkNorm: device.createComputePipeline({ layout: 'auto', compute: { module: mods.qk_norm, entryPoint: 'main' } }),
      ropeKV: device.createComputePipeline({ layout: 'auto', compute: { module: mods.rope_kv, entryPoint: 'main' } }),
      attention: device.createComputePipeline({ layout: 'auto', compute: { module: mods.attention, entryPoint: 'main' } }),
      attnSplitDecode: device.createComputePipeline({ layout: attnDecodeLayout, compute: { module: mods.attention_decode, entryPoint: 'split_kv_gqa' } }),
      attnReduceDecode: device.createComputePipeline({ layout: attnDecodeLayout, compute: { module: mods.attention_decode, entryPoint: 'reduce_kv_gqa' } }),
      attnDecodeBgl,
      shortConv: device.createComputePipeline({ layout: 'auto', compute: { module: mods.short_conv, entryPoint: 'main' } }),
      pleCombine: device.createComputePipeline({ layout: 'auto', compute: { module: mods.ple_ops, entryPoint: 'ple_combine' } }),
      gateMulGeluTanh: device.createComputePipeline({ layout: 'auto', compute: { module: mods.ple_ops, entryPoint: 'gate_mul_gelu_tanh' } }),
      softcapLogits: device.createComputePipeline({ layout: 'auto', compute: { module: mods.ple_ops, entryPoint: 'softcap_logits' } }),
      addBias: device.createComputePipeline({ layout: 'auto', compute: { module: mods.ple_ops, entryPoint: 'add_bias' } }),
    };

    return { device, pipelines, subgroupSizeGuess, useSubgroups, hasTimestampQuery };
  }

  static async create(modelBaseUrl, kernelBaseUrl, { maxSeqLen = 512, maxBatchTokens = 64, forceNoSubgroups = false } = {}) {
    const manifestResp = await fetch(`${modelBaseUrl}/manifest.json`);
    if (!manifestResp.ok) throw new Error(`Failed to fetch manifest.json: ${manifestResp.status}`);
    const manifest = await manifestResp.json();
    const bp = manifest.blueprint;
    const { device, pipelines, subgroupSizeGuess, useSubgroups, hasTimestampQuery } = await IgnisEngine._initDeviceAndPipelines(
      kernelBaseUrl,
      { hiddenAct: bp.hidden_act || 'gelu', weightBits: bp.weight_bits || 4, forceNoSubgroups }
    );

    const gpuShards = new Map();
    let embdBytes = null;
    for (const shardInfo of manifest.shards) {
      if (shardInfo.id === 'ple_embd' || shardInfo.id === 'vision' || shardInfo.id === 'audio_enc' || shardInfo.id === 'audio_dec') {
        continue;
      }
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
      useSubgroups,
      maxSeqLen,
      maxBatchTokens,
      hasTimestampQuery,
    });
  }

  static async createWithOpfs(modelBaseUrl, modelKey, kernelBaseUrl, {
    maxSeqLen = 512,
    maxBatchTokens = 64,
    forceNoSubgroups = false,
    onProgress = null,
  } = {}) {
    const opfsRes = await ensureOpfsModel(modelBaseUrl, modelKey, { onProgress });
    const { dirHandle, manifest, cacheHits, cacheMisses, hasTokenizer } = opfsRes;
    const bp = manifest.blueprint;
    if (onProgress) {
      onProgress({
        phase: 'compile',
        file: 'WGSL Compute Pipelines',
        loaded: 0,
        total: 1,
        overallLoaded: 0,
        overallTotal: 1,
        cached: true,
      });
    }
    const { device, pipelines, subgroupSizeGuess, useSubgroups, hasTimestampQuery } = await IgnisEngine._initDeviceAndPipelines(
      kernelBaseUrl,
      { hiddenAct: bp.hidden_act || 'gelu', weightBits: bp.weight_bits || 4, forceNoSubgroups }
    );
    const gpuShards = await uploadOpfsShardsToGpu(device, dirHandle, manifest, undefined, onProgress);
    const embdTable = await OpfsEmbeddingTable.open(dirHandle, manifest.tensors.token_embd, 4096, 'embd.ignis');
    const pleEmbdTable = (bp.has_ple && manifest.tensors.token_embd_ple)
      ? await OpfsEmbeddingTable.open(dirHandle, manifest.tensors.token_embd_ple, 2048, 'ple_embd.ignis')
      : null;
    const tokenizerBytes = hasTokenizer ? await readOpfsFileBytes(dirHandle, 'tokenizer.json') : null;

    const engine = new IgnisEngine(device, manifest, gpuShards, embdTable, pipelines, {
      subgroupSizeGuess,
      useSubgroups,
      maxSeqLen,
      maxBatchTokens,
      cacheHits,
      cacheMisses,
      hasTimestampQuery,
      pleEmbdSource: pleEmbdTable,
      dirHandle,
    });
    return { engine, tokenizerBytes, dirHandle };
  }

  _layerSpec(i) {
    const bp = this.bp;
    const lt = bp.layer_types[i] || 'full';
    const isConv = lt === 'conv';
    const isSwa = lt === 'swa';
    const hd = bp.head_dims ? bp.head_dims[i] : bp.head_dim;
    const kvSrc = bp.kv_source_layers ? bp.kv_source_layers[i] : i;
    const isReuse = kvSrc !== i;
    const ropeDim = isSwa ? bp.rope_dim_swa : bp.rope_dim_full;
    const ropeTheta = isSwa ? bp.rope_theta_swa : bp.rope_theta_full;
    const ropeLayout = bp.rope_layout === 'proportional' ? 1 : 0;
    const attnScale = bp.attn_scale || (1.0 / Math.sqrt(hd));
    const layerScalar = (bp.has_layer_scalar && bp.layer_scalars) ? bp.layer_scalars[i] : 1.0;
    return { lt, isConv, isSwa, hd, kvSrc, isReuse, ropeDim, ropeTheta, ropeLayout, attnScale, layerScalar };
  }

  _allocateWorkBuffers() {
    const bp = this.bp;
    const T = this.maxBatchTokens;
    const D = bp.hidden_size;
    const FF = bp.intermediate_size;
    const qDimMax = bp.num_attention_heads * bp.head_dim;
    const kvDimMax = bp.num_key_value_heads * bp.head_dim;
    const qkvDimMax = Math.max(qDimMax + 2 * kvDimMax, 3 * D);

    const mkBuf = (bytes, usage = GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST) =>
      this.device.createBuffer({ size: Math.max(16, alignUp(bytes, 16)), usage });

    this.bufs = {
      x: mkBuf(T * D * 4),
      norm: mkBuf(T * D * 4),
      qkv: mkBuf(T * qkvDimMax * 4),
      qRot: mkBuf(T * qDimMax * 4),
      attnGate: mkBuf(T * bp.num_attention_heads * 4),
      attnSplit: mkBuf(bp.num_attention_heads * 16 * (bp.head_dim + 2) * 4),
      attnOut: mkBuf(T * Math.max(qDimMax, D) * 4),
      ffnAct: mkBuf(T * FF * 4),
      branchOut: mkBuf(T * D * 4),
      logits: mkBuf(bp.vocab_size * 4),
      dummyGate: mkBuf(16),
      dummyNormW: mkBuf(Math.max(bp.head_dim, D) * 4),
    };

    if (bp.has_ple) {
      const L = bp.num_layers;
      const P = bp.ple_dim;
      this.bufs.pleCtxProj = mkBuf(T * L * P * 4);
      this.bufs.pleTok = mkBuf(T * L * P * 4);
      this.bufs.pleCombined = mkBuf(L * T * P * 4);
      this.bufs.pleGate = mkBuf(T * P * 4);
      this.bufs.pleMul = mkBuf(T * P * 4);
    }

    if (this.hasTimestampQuery) {
      this.tsQuerySet = this.device.createQuerySet({ type: 'timestamp', count: 2 });
      this.tsResolveBuf = this.device.createBuffer({ size: 16, usage: GPUBufferUsage.QUERY_RESOLVE | GPUBufferUsage.COPY_SRC });
      this.tsReadBuf = this.device.createBuffer({ size: 16, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST });
    }

    const convStateLen = Math.max(0, (bp.conv_kernel_size || 0) - 1);
    this.convStates = [];
    this.convBackupBufs = [];
    this.kvCaches = [];
    for (let i = 0; i < bp.num_layers; i++) {
      const spec = this._layerSpec(i);
      if (spec.isConv) {
        this.kvCaches.push(null);
        const csBytes = D * Math.max(1, convStateLen) * 4;
        this.convStates.push(mkBuf(csBytes));
        this.convBackupBufs.push(mkBuf(csBytes));
      } else if (spec.isReuse) {
        this.convStates.push(null);
        this.convBackupBufs.push(null);
        this.kvCaches.push(this.kvCaches[spec.kvSrc]);
      } else {
        this.convStates.push(null);
        this.convBackupBufs.push(null);
        const cap = spec.isSwa ? ((bp.sliding_window || 512) + this.maxBatchTokens) : this.maxSeqLen;
        const bytes = bp.num_key_value_heads * cap * spec.hd * 2;
        this.kvCaches.push({
          isSwa: spec.isSwa,
          capacity: cap,
          headDim: spec.hd,
          k: mkBuf(bytes, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST),
          v: mkBuf(bytes, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST),
        });
      }
    }
    this.hasConvLayers = this.convStates.some(Boolean);
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

  _selectMatmulPipe(dtype, T, addResidual) {
    const isGemv = T === 1;
    if (dtype === 'u32_q8') return isGemv ? this.pipelines.gemvQ8 : this.pipelines.gemm4Q8;
    if (dtype === 'u32_q3') {
      return isGemv
        ? (addResidual ? this.pipelines.gemvQ3Add : this.pipelines.gemvQ3)
        : (addResidual ? this.pipelines.gemm4Q3Add : this.pipelines.gemm4Q3);
    }
    return isGemv
      ? (addResidual ? this.pipelines.gemvAdd : this.pipelines.gemv)
      : (addResidual ? this.pipelines.gemm4Add : this.pipelines.gemm4);
  }

  _buildStaticDecodePlan() {
    const bp = this.bp;
    const D = bp.hidden_size;
    const p = this.pipelines;

    const uNormMain = this._makeStaticUniform(16, (u32, f32) => {
      u32[0] = D; u32[1] = 1; f32[2] = bp.rms_norm_eps; f32[3] = 1.0;
    });
    const normResidualUniforms = new Map();
    const getPostNormUniform = (scalar = 1.0) => {
      const k = String(scalar);
      let u = normResidualUniforms.get(k);
      if (!u) {
        u = this._makeStaticUniform(16, (u32, f32) => {
          u32[0] = D; u32[1] = 1; f32[2] = bp.rms_norm_eps; f32[3] = scalar;
        });
        normResidualUniforms.set(k, u);
      }
      return u;
    };

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

    const dynUniformMap = new Map();
    this.dynPosUniforms = [];
    const getRopeUniform = (spec) => {
      const cap = spec.isSwa ? ((bp.sliding_window || 512) + this.maxBatchTokens) : this.maxSeqLen;
      const key = `rope_${spec.hd}_${spec.ropeDim}_${cap}_${spec.isSwa ? 1 : 0}_${spec.ropeTheta}_${spec.ropeLayout}_${spec.isReuse ? 1 : 0}`;
      let u = dynUniformMap.get(key);
      if (!u) {
        u = this._makeStaticUniform(48, (u32, f32) => {
          u32[0] = 1; u32[1] = 0;
          u32[2] = bp.num_attention_heads; u32[3] = bp.num_key_value_heads;
          u32[4] = spec.hd; u32[5] = spec.ropeDim;
          u32[6] = cap; u32[7] = spec.isSwa ? 1 : 0;
          f32[8] = spec.ropeTheta;
          u32[9] = spec.ropeLayout;
          u32[10] = spec.isReuse ? 1 : 0;
        });
        dynUniformMap.set(key, u);
        this.dynPosUniforms.push(u);
      }
      return u;
    };
    const getAttnUniform = (spec) => {
      const cap = spec.isSwa ? ((bp.sliding_window || 512) + this.maxBatchTokens) : this.maxSeqLen;
      const key = `attn_${spec.hd}_${cap}_${spec.isSwa ? 1 : 0}_${spec.attnScale}`;
      let u = dynUniformMap.get(key);
      if (!u) {
        u = this._makeStaticUniform(48, (u32, f32) => {
          u32[0] = 1; u32[1] = 0;
          u32[2] = bp.num_attention_heads; u32[3] = bp.num_key_value_heads;
          u32[4] = spec.hd; u32[5] = cap;
          u32[6] = bp.sliding_window || 0; u32[7] = spec.isSwa ? 1 : 0;
          u32[8] = bp.has_attn_gate ? 1 : 0; f32[9] = spec.attnScale;
        });
        dynUniformMap.set(key, u);
        this.dynPosUniforms.push(u);
      }
      return u;
    };

    const qkNormUniforms = new Map();
    const getQkUniform = (spec) => {
      const key = `qk_${spec.hd}_${spec.isReuse ? 1 : 0}`;
      let u = qkNormUniforms.get(key);
      if (!u) {
        u = this._makeStaticUniform(32, (u32, f32) => {
          u32[0] = bp.num_attention_heads; u32[1] = bp.num_key_value_heads;
          u32[2] = spec.hd; u32[3] = 1;
          f32[4] = bp.rms_norm_eps;
          u32[5] = bp.has_v_norm ? 1 : 0;
          u32[6] = spec.isReuse ? 1 : 0;
        });
        qkNormUniforms.set(key, u);
      }
      return u;
    };

    const mkMatmulStep = (weightName, xBuf, outBuf, addResidual = false) => {
      const q = this._tensorBinding(`${weightName}.quants`);
      const s = this._tensorBinding(`${weightName}.scales`);
      const pipe = this._selectMatmulPipe(q.meta.dtype, 1, addResidual);
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
      const totWgX = Math.ceil(M / 4);
      const wgZ = totWgX > 65535 ? Math.ceil(totWgX / 32768) : 1;
      const wgX = totWgX > 65535 ? 32768 : totWgX;
      return { pipe, bg, wgX, wgY: 1, wgZ };
    };

    const mkNormStep = (weightName, inBuf = this.bufs.x, outBuf = this.bufs.norm) => {
      const w = this._tensorBinding(weightName);
      const bg = this.device.createBindGroup({
        layout: p.rmsnorm.getBindGroupLayout(0),
        entries: [
          { binding: 0, resource: { buffer: uNormMain, size: 16 } },
          { binding: 1, resource: { buffer: inBuf } },
          { binding: 2, resource: { buffer: w.buffer, offset: w.offset, size: w.size } },
          { binding: 3, resource: { buffer: outBuf } },
        ],
      });
      return { pipe: p.rmsnorm, bg, wgX: 1, wgY: 1 };
    };

    const mkPostNormResStep = (weightName, branchBuf, scalar = 1.0) => {
      const w = this._tensorBinding(weightName);
      const bg = this.device.createBindGroup({
        layout: p.postNormResidual.getBindGroupLayout(0),
        entries: [
          { binding: 0, resource: { buffer: getPostNormUniform(scalar), size: 16 } },
          { binding: 1, resource: { buffer: branchBuf } },
          { binding: 2, resource: { buffer: w.buffer, offset: w.offset, size: w.size } },
          { binding: 3, resource: { buffer: this.bufs.x } },
        ],
      });
      return { pipe: p.postNormResidual, bg, wgX: 1, wgY: 1 };
    };

    this.decodeSteps = [];

    // Precompute Gemma 4 Per-Layer Embeddings [L, 1, P] inside static plan!
    if (bp.has_ple) {
      this.decodeSteps.push(mkMatmulStep('ple_model_proj', this.bufs.x, this.bufs.pleCtxProj, false));
      const uPleComb = this._makeStaticUniform(32, (u32, f32) => {
        u32[0] = bp.num_layers; u32[1] = bp.ple_dim; u32[2] = 1; u32[3] = 1; // max_batch=1 stride for decode
        f32[4] = 1.0 / Math.sqrt(D);
        f32[5] = Math.SQRT1_2;
        f32[6] = bp.rms_norm_eps;
      });
      const pleNormW = this._tensorBinding('ple_proj_norm');
      const bgPle = this.device.createBindGroup({
        layout: p.pleCombine.getBindGroupLayout(0),
        entries: [
          { binding: 0, resource: { buffer: uPleComb, size: 32 } },
          { binding: 1, resource: { buffer: this.bufs.pleCtxProj } },
          { binding: 2, resource: { buffer: this.bufs.pleTok } },
          { binding: 3, resource: { buffer: pleNormW.buffer, offset: pleNormW.offset, size: pleNormW.size } },
          { binding: 4, resource: { buffer: this.bufs.pleCombined } },
        ],
      });
      this.decodeSteps.push({ pipe: p.pleCombine, bg: bgPle, wgX: bp.num_layers, wgY: 1 });
    }

    const uGate = bp.has_attn_gate
      ? this._makeStaticUniform(16, (u32) => { u32[0] = bp.num_attention_heads; u32[1] = D; u32[2] = 1; u32[3] = 0; })
      : null;
    const uConv = bp.conv_kernel_size > 0
      ? this._makeStaticUniform(16, (u32) => { u32[0] = D; u32[1] = 1; u32[2] = bp.conv_kernel_size; u32[3] = bp.conv_kernel_size - 1; })
      : null;

    for (let i = 0; i < bp.num_layers; i++) {
      const spec = this._layerSpec(i);
      this.decodeSteps.push(mkNormStep(`blk.${i}.attn_norm`));

      if (spec.isConv) {
        this.decodeSteps.push(mkMatmulStep(`blk.${i}.conv_in_proj`, this.bufs.norm, this.bufs.qkv, false));
        const convW = this._tensorBinding(`blk.${i}.conv_kernel`);
        const bgConv = this.device.createBindGroup({
          layout: p.shortConv.getBindGroupLayout(0),
          entries: [
            { binding: 0, resource: { buffer: uConv, size: 16 } },
            { binding: 1, resource: { buffer: this.bufs.qkv } },
            { binding: 2, resource: { buffer: convW.buffer, offset: convW.offset, size: convW.size } },
            { binding: 3, resource: { buffer: this.convStates[i] } },
            { binding: 4, resource: { buffer: this.bufs.attnOut } },
          ],
        });
        this.decodeSteps.push({ pipe: p.shortConv, bg: bgConv, wgX: Math.ceil(D / 256), wgY: 1 });
        if (bp.has_post_norms) {
          this.decodeSteps.push(mkMatmulStep(`blk.${i}.conv_out_proj`, this.bufs.attnOut, this.bufs.branchOut, false));
          this.decodeSteps.push(mkPostNormResStep(`blk.${i}.post_attn_norm`, this.bufs.branchOut, 1.0));
        } else {
          this.decodeSteps.push(mkMatmulStep(`blk.${i}.conv_out_proj`, this.bufs.attnOut, this.bufs.x, true));
        }
      } else {
        const kv = this.kvCaches[i];
        const qDim = bp.num_attention_heads * spec.hd;
        const kvDim = bp.num_key_value_heads * spec.hd;

        this.decodeSteps.push(mkMatmulStep(`blk.${i}.attn_qkv`, this.bufs.norm, this.bufs.qkv, false));
        if (bp.has_qkv_bias && this.manifest.tensors[`blk.${i}.attn_qkv_bias`]) {
          const bT = this._tensorBinding(`blk.${i}.attn_qkv_bias`);
          const uBias = this._makeStaticUniform(32, (u32) => {
            u32[0] = bp.num_layers; u32[1] = bT.meta.shape[0]; u32[2] = 1; u32[3] = 1;
          });
          const bgB = this.device.createBindGroup({
            layout: p.addBias.getBindGroupLayout(0),
            entries: [
              { binding: 0, resource: { buffer: uBias, size: 32 } },
              { binding: 3, resource: { buffer: bT.buffer, offset: bT.offset, size: bT.size } },
              { binding: 4, resource: { buffer: this.bufs.qkv } },
            ],
          });
          this.decodeSteps.push({ pipe: p.addBias, bg: bgB, wgX: Math.ceil(bT.meta.shape[0] / 256), wgY: 1 });
        }

        if (bp.has_qk_norm) {
          const qn = this._tensorBinding(`blk.${i}.q_norm`);
          const kn = spec.isReuse
            ? { buffer: this.bufs.dummyNormW, offset: 0, size: spec.hd * 4 }
            : this._tensorBinding(`blk.${i}.k_norm`);
          const bgQk = this.device.createBindGroup({
            layout: p.qkNorm.getBindGroupLayout(0),
            entries: [
              { binding: 0, resource: { buffer: getQkUniform(spec), size: 32 } },
              { binding: 1, resource: { buffer: this.bufs.qkv } },
              { binding: 2, resource: { buffer: qn.buffer, offset: qn.offset, size: qn.size } },
              { binding: 3, resource: { buffer: kn.buffer, offset: kn.offset, size: kn.size } },
            ],
          });
          const totHeads = spec.isReuse
            ? bp.num_attention_heads
            : bp.num_attention_heads + bp.num_key_value_heads * (1 + (bp.has_v_norm ? 1 : 0));
          this.decodeSteps.push({ pipe: p.qkNorm, bg: bgQk, wgX: totHeads, wgY: 1 });
        }

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

        const bgRope = this.device.createBindGroup({
          layout: p.ropeKV.getBindGroupLayout(0),
          entries: [
            { binding: 0, resource: { buffer: getRopeUniform(spec), size: 48 } },
            { binding: 1, resource: { buffer: this.bufs.qkv } },
            { binding: 2, resource: { buffer: this.bufs.qRot } },
            { binding: 3, resource: { buffer: kv.k } },
            { binding: 4, resource: { buffer: kv.v } },
          ],
        });
        this.decodeSteps.push({ pipe: p.ropeKV, bg: bgRope, wgX: Math.ceil(Math.max(qDim, kvDim) / 128), wgY: 1 });

        const gqaGroup = bp.num_attention_heads / bp.num_key_value_heads;
        if (gqaGroup <= 4) {
          const bgAttnDec = this.device.createBindGroup({
            layout: p.attnDecodeBgl,
            entries: [
              { binding: 0, resource: { buffer: getAttnUniform(spec), size: 48 } },
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
              { binding: 0, resource: { buffer: getAttnUniform(spec), size: 48 } },
              { binding: 1, resource: { buffer: this.bufs.qRot } },
              { binding: 2, resource: { buffer: kv.k } },
              { binding: 3, resource: { buffer: kv.v } },
              { binding: 4, resource: { buffer: bp.has_attn_gate ? this.bufs.attnGate : this.bufs.dummyGate } },
              { binding: 5, resource: { buffer: this.bufs.attnOut } },
            ],
          });
          this.decodeSteps.push({ pipe: p.attention, bg: bgAttn, wgX: bp.num_attention_heads, wgY: 1 });
        }

        if (bp.has_post_norms) {
          this.decodeSteps.push(mkMatmulStep(`blk.${i}.attn_output`, this.bufs.attnOut, this.bufs.branchOut, false));
          this.decodeSteps.push(mkPostNormResStep(`blk.${i}.post_attn_norm`, this.bufs.branchOut, 1.0));
        } else {
          this.decodeSteps.push(mkMatmulStep(`blk.${i}.attn_output`, this.bufs.attnOut, this.bufs.x, true));
        }
      }

      // FFN block
      this.decodeSteps.push(mkNormStep(`blk.${i}.ffn_norm`));
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

      const ffnPostScalar = bp.has_ple ? 1.0 : spec.layerScalar;
      if (bp.has_post_norms) {
        this.decodeSteps.push(mkMatmulStep(`blk.${i}.ffn_down`, this.bufs.ffnAct, this.bufs.branchOut, false));
        this.decodeSteps.push(mkPostNormResStep(`blk.${i}.post_ffn_norm`, this.bufs.branchOut, ffnPostScalar));
      } else {
        this.decodeSteps.push(mkMatmulStep(`blk.${i}.ffn_down`, this.bufs.ffnAct, this.bufs.x, true));
      }

      // Gemma 4 Per-Layer Embedding Gated Injection
      if (bp.has_ple) {
        this.decodeSteps.push(mkMatmulStep(`blk.${i}.ple_gate`, this.bufs.x, this.bufs.pleGate, false));
        const uGateMul = this._makeStaticUniform(32, (u32) => {
          u32[0] = bp.num_layers; u32[1] = bp.ple_dim; u32[2] = 1; u32[3] = i * bp.ple_dim;
        });
        const bgGM = this.device.createBindGroup({
          layout: p.gateMulGeluTanh.getBindGroupLayout(0),
          entries: [
            { binding: 0, resource: { buffer: uGateMul, size: 32 } },
            { binding: 1, resource: { buffer: this.bufs.pleGate } },
            { binding: 2, resource: { buffer: this.bufs.pleCombined } },
            { binding: 4, resource: { buffer: this.bufs.pleMul } },
          ],
        });
        this.decodeSteps.push({ pipe: p.gateMulGeluTanh, bg: bgGM, wgX: Math.ceil(bp.ple_dim / 256), wgY: 1 });
        this.decodeSteps.push(mkMatmulStep(`blk.${i}.ple_proj`, this.bufs.pleMul, this.bufs.branchOut, false));
        this.decodeSteps.push(mkPostNormResStep(`blk.${i}.post_ple_norm`, this.bufs.branchOut, spec.layerScalar));
      }
    }

    this.decodeSteps.push(mkNormStep('output_norm'));
    this.decodeSteps.push(mkMatmulStep('output', this.bufs.norm, this.bufs.logits, false));

    if (bp.final_logit_softcapping && bp.final_logit_softcapping > 0) {
      const uCap = this._makeStaticUniform(32, (u32, f32) => {
        u32[0] = bp.num_layers; u32[1] = bp.vocab_size; u32[2] = 1; u32[3] = 1;
        f32[4] = bp.final_logit_softcapping;
      });
      const bgCap = this.device.createBindGroup({
        layout: p.softcapLogits.getBindGroupLayout(0),
        entries: [
          { binding: 0, resource: { buffer: uCap, size: 32 } },
          { binding: 4, resource: { buffer: this.bufs.logits } },
        ],
      });
      this.decodeSteps.push({ pipe: p.softcapLogits, bg: bgCap, wgX: Math.ceil(bp.vocab_size / 256), wgY: 1 });
    }
  }

  reset() {
    this.curPos = 0;
    for (const cs of this.convStates) {
      if (cs) this.device.queue.writeBuffer(cs, 0, new Uint8Array(cs.size));
    }
  }

  rollbackTo(pos) {
    this.curPos = pos;
  }

  async snapshotConvStates() {
    const snap = [];
    for (let i = 0; i < this.convStates.length; i++) {
      const cs = this.convStates[i];
      if (!cs) { snap.push(null); continue; }
      snap.push(await this._readBufferFloat32(cs, cs.size / 4));
    }
    return snap;
  }

  restoreConvStates(snap) {
    if (!snap) return;
    for (let i = 0; i < this.convStates.length; i++) {
      if (this.convStates[i] && snap[i]) {
        this.device.queue.writeBuffer(this.convStates[i], 0, snap[i]);
      }
    }
  }

  snapshotConvStatesGpu() {
    if (!this.hasConvLayers) return;
    const enc = this.device.createCommandEncoder();
    for (let i = 0; i < this.convStates.length; i++) {
      const src = this.convStates[i];
      const dst = this.convBackupBufs[i];
      if (src && dst) {
        enc.copyBufferToBuffer(src, 0, dst, 0, src.size);
      }
    }
    this.device.queue.submit([enc.finish()]);
  }

  restoreConvStatesGpu() {
    if (!this.hasConvLayers) return;
    const enc = this.device.createCommandEncoder();
    for (let i = 0; i < this.convStates.length; i++) {
      const dst = this.convStates[i];
      const src = this.convBackupBufs[i];
      if (src && dst) {
        enc.copyBufferToBuffer(src, 0, dst, 0, src.size);
      }
    }
    this.device.queue.submit([enc.finish()]);
  }

  async lookupEmbeddings(tokenIds) {
    let raw;
    if (this.embdSource instanceof OpfsEmbeddingTable) {
      raw = await this.embdSource.lookupBatch(tokenIds);
    } else {
      const em = this.manifest.tensors.token_embd;
      const D = this.bp.hidden_size;
      const stride = em.row_stride;
      const baseOff = em.offset;
      raw = new Float32Array(tokenIds.length * D);
      for (let t = 0; t < tokenIds.length; t++) {
        const id = tokenIds[t];
        const rowOff = baseOff + id * stride;
        const scale = this.embdView.getFloat32(rowOff, true);
        const dstBase = t * D;
        for (let d = 0; d < D; d++) {
          raw[dstBase + d] = this.embdView.getInt8(rowOff + 4 + d) * scale;
        }
      }
    }
    const embedScale = this.bp.embed_scale || 1.0;
    if (embedScale !== 1.0) {
      for (let i = 0; i < raw.length; i++) raw[i] *= embedScale;
    }
    return raw;
  }

  async lookupPleEmbeddings(tokenIds, isMultimodalMask = null) {
    if (!this.pleEmbdSource) return null;
    const padId = this.bp.pad_token_id || 0;
    const effectiveIds = tokenIds.map((id, i) => (isMultimodalMask && isMultimodalMask[i] ? padId : id));
    const raw = await this.pleEmbdSource.lookupBatch(effectiveIds);
    const scale = Math.sqrt(this.bp.ple_dim);
    for (let i = 0; i < raw.length; i++) raw[i] *= scale;
    return raw;
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

  async _readBufferFloat32(buf, count, offsetBytes = 0) {
    const bytes = count * 4;
    const rb = this.device.createBuffer({
      size: bytes,
      usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
    });
    const enc = this.device.createCommandEncoder();
    enc.copyBufferToBuffer(buf, offsetBytes, rb, 0, bytes);
    this.device.queue.submit([enc.finish()]);
    await rb.mapAsync(GPUMapMode.READ);
    const res = new Float32Array(rb.getMappedRange().slice(0));
    rb.unmap();
    rb.destroy();
    return res;
  }

  async _forwardDecodeFast(tokenId, { profileGpu = false, returnLastHidden = false } = {}) {
    const emb = await this.lookupEmbeddings([tokenId]);
    this.device.queue.writeBuffer(this.bufs.x, 0, emb);
    if (this.bp.has_ple && this.pleEmbdSource) {
      const pleTok = await this.lookupPleEmbeddings([tokenId]);
      this.device.queue.writeBuffer(this.bufs.pleTok, 0, pleTok);
    }

    const posHeader = new Uint32Array([1, this.curPos]);
    for (let i = 0; i < this.dynPosUniforms.length; i++) {
      this.device.queue.writeBuffer(this.dynPosUniforms[i], 0, posHeader);
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
      pass.dispatchWorkgroups(step.wgX, step.wgY, step.wgZ || 1);
      pass.end();
    }
    if (useTs) {
      enc.resolveQuerySet(this.tsQuerySet, 0, 2, this.tsResolveBuf, 0);
      enc.copyBufferToBuffer(this.tsResolveBuf, 0, this.tsReadBuf, 0, 16);
    }
    this.device.queue.submit([enc.finish()]);

    const logits = await this._readBufferFloat32(this.bufs.logits, this.bp.vocab_size);
    const lastHidden = returnLastHidden
      ? await this._readBufferFloat32(this.bufs.norm, this.bp.hidden_size)
      : null;
    let gpuMs = null;
    if (useTs) {
      await this.tsReadBuf.mapAsync(GPUMapMode.READ);
      const ts = new BigInt64Array(this.tsReadBuf.getMappedRange().slice(0));
      this.tsReadBuf.unmap();
      gpuMs = Number(ts[1] - ts[0]) / 1e6;
    }
    this.curPos += 1;
    return { logits, lastHidden, traces: {}, gpuMs };
  }

  _encodeMatmulQ(enc, tempUniforms, weightBaseName, xBuf, xOffsetBytes, outBuf, T, addResidual = false) {
    const q = this._tensorBinding(`${weightBaseName}.quants`);
    const s = this._tensorBinding(`${weightBaseName}.scales`);
    const [M, K] = q.meta.orig_shape;
    const [, K_pad] = q.meta.padded_shape;
    const uBuf = this._makeUniform((u32) => { u32[0] = M; u32[1] = K; u32[2] = T; u32[3] = K_pad; });
    tempUniforms.push(uBuf);

    const pipe = this._selectMatmulPipe(q.meta.dtype, T, addResidual);
    const isGemv = T === 1;
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
    const totWgX = Math.ceil(M / 4);
    const wgZ = totWgX > 65535 ? Math.ceil(totWgX / 32768) : 1;
    const wgX = totWgX > 65535 ? 32768 : totWgX;
    pass.dispatchWorkgroups(wgX, isGemv ? 1 : Math.ceil(T / 4), wgZ);
    pass.end();
  }

  _encodeFusedGegluQ(enc, tempUniforms, layerIdx, xBuf, outBuf, T) {
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
    const totWgX = Math.ceil(M / 4);
    const wgZ = totWgX > 65535 ? Math.ceil(totWgX / 32768) : 1;
    const wgX = totWgX > 65535 ? 32768 : totWgX;
    pass.dispatchWorkgroups(wgX, isGemv ? 1 : Math.ceil(T / 4), wgZ);
    pass.end();
  }

  _encodeRMSNorm(enc, tempUniforms, xBuf, normWeightName, outBuf, T) {
    const w = this._tensorBinding(normWeightName);
    const uBuf = this._makeUniform((u32, f32) => {
      u32[0] = this.bp.hidden_size; u32[1] = T; f32[2] = this.bp.rms_norm_eps; f32[3] = 1.0;
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

  _encodePostNormResidual(enc, tempUniforms, branchBuf, normWeightName, resBuf, T, scalar = 1.0) {
    const w = this._tensorBinding(normWeightName);
    const uBuf = this._makeUniform((u32, f32) => {
      u32[0] = this.bp.hidden_size; u32[1] = T; f32[2] = this.bp.rms_norm_eps; f32[3] = scalar;
    });
    tempUniforms.push(uBuf);
    const bg = this.device.createBindGroup({
      layout: this.pipelines.postNormResidual.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: uBuf, size: 16 } },
        { binding: 1, resource: { buffer: branchBuf } },
        { binding: 2, resource: { buffer: w.buffer, offset: w.offset, size: w.size } },
        { binding: 3, resource: { buffer: resBuf } },
      ],
    });
    const pass = enc.beginComputePass();
    pass.setPipeline(this.pipelines.postNormResidual);
    pass.setBindGroup(0, bg);
    pass.dispatchWorkgroups(T, 1, 1);
    pass.end();
  }

  async forward(tokenIds, options = {}) {
    const T = tokenIds.length;
    if (T > this.maxBatchTokens && (options.captureTraces || options.returnAllLogits)) {
      throw new Error(`Batch ${T} exceeds maxBatchTokens ${this.maxBatchTokens}`);
    }
    if (T === 1 && !options.captureTraces && !options.returnAllLogits && !options.skipLogits) {
      return this._forwardDecodeFast(tokenIds[0], options);
    }
    const emb = await this.lookupEmbeddings(tokenIds);
    return this.forwardEmbeds(emb, { tokenIds, ...options });
  }

  async forwardVerify(tokenIds, options = {}) {
    return this.forward(tokenIds, { ...options, returnAllLogits: true });
  }

  async forwardEmbeds(embFloat32, options = {}) {
    const {
      tokenIds = null,
      isMultimodalMask = null,
      bidirSpan = null,
      returnAllLogits = false,
      captureTraces = false,
      returnLastHidden = false,
      skipLogits = false,
    } = options;
    const bp = this.bp;
    const D = bp.hidden_size;
    const T = embFloat32.length / D;
    if (!Number.isInteger(T) || T <= 0) {
      throw new Error(`Invalid embed length ${embFloat32.length} for hidden_size=${D}`);
    }
    if (T > this.maxBatchTokens) {
      if (returnAllLogits || captureTraces) {
        throw new Error(`Batch ${T} exceeds maxBatchTokens ${this.maxBatchTokens} for returnAllLogits/captureTraces`);
      }
      let finalRes = null;
      for (let off = 0; off < T; off += this.maxBatchTokens) {
        const end = Math.min(T, off + this.maxBatchTokens);
        const isLast = end === T;
        finalRes = await this._forwardEmbedsChunk(embFloat32.subarray(off * D, end * D), {
          tokenIds: tokenIds ? tokenIds.slice(off, end) : null,
          isMultimodalMask: isMultimodalMask ? isMultimodalMask.slice(off, end) : null,
          bidirSpan,
          returnAllLogits: false,
          captureTraces: false,
          returnLastHidden: isLast ? returnLastHidden : false,
          skipLogits: isLast ? skipLogits : true,
        });
      }
      return finalRes;
    }
    return this._forwardEmbedsChunk(embFloat32, {
      tokenIds,
      isMultimodalMask,
      bidirSpan,
      returnAllLogits,
      captureTraces,
      returnLastHidden,
      skipLogits,
    });
  }

  async _forwardEmbedsChunk(embFloat32, {
    tokenIds = null,
    isMultimodalMask = null,
    bidirSpan = null,
    returnAllLogits = false,
    captureTraces = false,
    returnLastHidden = false,
    skipLogits = false,
  } = {}) {
    const bp = this.bp;
    const D = bp.hidden_size;
    const T = embFloat32.length / D;
    if (!Number.isInteger(T) || T <= 0 || T > this.maxBatchTokens) {
      throw new Error(`Invalid embed length ${embFloat32.length} for hidden_size=${D}, maxBatch=${this.maxBatchTokens}`);
    }

    const startPos = this.curPos;
    this.device.queue.writeBuffer(this.bufs.x, 0, embFloat32);

    if (bp.has_ple && this.pleEmbdSource) {
      const ids = tokenIds || new Array(T).fill(bp.pad_token_id || 0);
      const pleTok = await this.lookupPleEmbeddings(ids, isMultimodalMask);
      this.device.queue.writeBuffer(this.bufs.pleTok, 0, pleTok);
    }

    const traces = {};
    if (captureTraces) {
      traces.embed_last = Array.from(embFloat32.slice((T - 1) * D, T * D));
    }

    const tempUniforms = [];
    let enc = this.device.createCommandEncoder();

    if (bp.has_ple) {
      this._encodeMatmulQ(enc, tempUniforms, 'ple_model_proj', this.bufs.x, 0, this.bufs.pleCtxProj, T, false);
      const uPle = this._makeUniform((u32, f32) => {
        u32[0] = bp.num_layers; u32[1] = bp.ple_dim; u32[2] = T; u32[3] = this.maxBatchTokens;
        f32[4] = 1.0 / Math.sqrt(D);
        f32[5] = Math.SQRT1_2;
        f32[6] = bp.rms_norm_eps;
      });
      tempUniforms.push(uPle);
      const pleNormW = this._tensorBinding('ple_proj_norm');
      const bgPle = this.device.createBindGroup({
        layout: this.pipelines.pleCombine.getBindGroupLayout(0),
        entries: [
          { binding: 0, resource: { buffer: uPle, size: 32 } },
          { binding: 1, resource: { buffer: this.bufs.pleCtxProj } },
          { binding: 2, resource: { buffer: this.bufs.pleTok } },
          { binding: 3, resource: { buffer: pleNormW.buffer, offset: pleNormW.offset, size: pleNormW.size } },
          { binding: 4, resource: { buffer: this.bufs.pleCombined } },
        ],
      });
      const passPle = enc.beginComputePass();
      passPle.setPipeline(this.pipelines.pleCombine);
      passPle.setBindGroup(0, bgPle);
      passPle.dispatchWorkgroups(bp.num_layers, T, 1);
      passPle.end();
    }

    for (let i = 0; i < bp.num_layers; i++) {
      const spec = this._layerSpec(i);
      this._encodeRMSNorm(enc, tempUniforms, this.bufs.x, `blk.${i}.attn_norm`, this.bufs.norm, T);

      if (spec.isConv) {
        this._encodeMatmulQ(enc, tempUniforms, `blk.${i}.conv_in_proj`, this.bufs.norm, 0, this.bufs.qkv, T, false);
        const convW = this._tensorBinding(`blk.${i}.conv_kernel`);
        const uConv = this._makeUniform((u32) => {
          u32[0] = D; u32[1] = T; u32[2] = bp.conv_kernel_size; u32[3] = bp.conv_kernel_size - 1;
        });
        tempUniforms.push(uConv);
        const bgConv = this.device.createBindGroup({
          layout: this.pipelines.shortConv.getBindGroupLayout(0),
          entries: [
            { binding: 0, resource: { buffer: uConv, size: 16 } },
            { binding: 1, resource: { buffer: this.bufs.qkv } },
            { binding: 2, resource: { buffer: convW.buffer, offset: convW.offset, size: convW.size } },
            { binding: 3, resource: { buffer: this.convStates[i] } },
            { binding: 4, resource: { buffer: this.bufs.attnOut } },
          ],
        });
        const passC = enc.beginComputePass();
        passC.setPipeline(this.pipelines.shortConv);
        passC.setBindGroup(0, bgConv);
        passC.dispatchWorkgroups(Math.ceil(D / 256), 1, 1);
        passC.end();

        if (bp.has_post_norms) {
          this._encodeMatmulQ(enc, tempUniforms, `blk.${i}.conv_out_proj`, this.bufs.attnOut, 0, this.bufs.branchOut, T, false);
          this._encodePostNormResidual(enc, tempUniforms, this.bufs.branchOut, `blk.${i}.post_attn_norm`, this.bufs.x, T, 1.0);
        } else {
          this._encodeMatmulQ(enc, tempUniforms, `blk.${i}.conv_out_proj`, this.bufs.attnOut, 0, this.bufs.x, T, true);
        }
      } else {
        const kv = this.kvCaches[i];
        const qDim = bp.num_attention_heads * spec.hd;
        const kvDim = bp.num_key_value_heads * spec.hd;

        this._encodeMatmulQ(enc, tempUniforms, `blk.${i}.attn_qkv`, this.bufs.norm, 0, this.bufs.qkv, T, false);

        if (bp.has_qkv_bias && this.manifest.tensors[`blk.${i}.attn_qkv_bias`]) {
          const bT = this._tensorBinding(`blk.${i}.attn_qkv_bias`);
          const bLen = bT.meta.shape[0];
          const uB = this._makeUniform((u32) => { u32[0] = bp.num_layers; u32[1] = bLen; u32[2] = T; u32[3] = 1; });
          tempUniforms.push(uB);
          const bgB = this.device.createBindGroup({
            layout: this.pipelines.addBias.getBindGroupLayout(0),
            entries: [
              { binding: 0, resource: { buffer: uB, size: 32 } },
              { binding: 3, resource: { buffer: bT.buffer, offset: bT.offset, size: bT.size } },
              { binding: 4, resource: { buffer: this.bufs.qkv } },
            ],
          });
          const passB = enc.beginComputePass();
          passB.setPipeline(this.pipelines.addBias);
          passB.setBindGroup(0, bgB);
          passB.dispatchWorkgroups(Math.ceil((T * bLen) / 256), 1, 1);
          passB.end();
        }

        if (bp.has_qk_norm) {
          const qn = this._tensorBinding(`blk.${i}.q_norm`);
          const kn = spec.isReuse
            ? { buffer: this.bufs.dummyNormW, offset: 0, size: spec.hd * 4 }
            : this._tensorBinding(`blk.${i}.k_norm`);
          const uQk = this._makeUniform((u32, f32) => {
            u32[0] = bp.num_attention_heads; u32[1] = bp.num_key_value_heads;
            u32[2] = spec.hd; u32[3] = T; f32[4] = bp.rms_norm_eps;
            u32[5] = bp.has_v_norm ? 1 : 0; u32[6] = spec.isReuse ? 1 : 0;
          });
          tempUniforms.push(uQk);
          const bgQk = this.device.createBindGroup({
            layout: this.pipelines.qkNorm.getBindGroupLayout(0),
            entries: [
              { binding: 0, resource: { buffer: uQk, size: 32 } },
              { binding: 1, resource: { buffer: this.bufs.qkv } },
              { binding: 2, resource: { buffer: qn.buffer, offset: qn.offset, size: qn.size } },
              { binding: 3, resource: { buffer: kn.buffer, offset: kn.offset, size: kn.size } },
            ],
          });
          const totHeads = spec.isReuse
            ? bp.num_attention_heads
            : bp.num_attention_heads + bp.num_key_value_heads * (1 + (bp.has_v_norm ? 1 : 0));
          const passQk = enc.beginComputePass();
          passQk.setPipeline(this.pipelines.qkNorm);
          passQk.setBindGroup(0, bgQk);
          passQk.dispatchWorkgroups(totHeads, T, 1);
          passQk.end();
        }

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
          u32[4] = spec.hd; u32[5] = spec.ropeDim; u32[6] = kv.capacity; u32[7] = spec.isSwa ? 1 : 0;
          f32[8] = spec.ropeTheta; u32[9] = spec.ropeLayout; u32[10] = spec.isReuse ? 1 : 0;
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
          u32[4] = spec.hd; u32[5] = kv.capacity; u32[6] = bp.sliding_window || 0; u32[7] = spec.isSwa ? 1 : 0;
          u32[8] = bp.has_attn_gate ? 1 : 0; f32[9] = spec.attnScale;
          u32[10] = bidirSpan ? bidirSpan[0] : 0; u32[11] = bidirSpan ? bidirSpan[1] : 0;
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

        if (bp.has_post_norms) {
          this._encodeMatmulQ(enc, tempUniforms, `blk.${i}.attn_output`, this.bufs.attnOut, 0, this.bufs.branchOut, T, false);
          this._encodePostNormResidual(enc, tempUniforms, this.bufs.branchOut, `blk.${i}.post_attn_norm`, this.bufs.x, T, 1.0);
        } else {
          this._encodeMatmulQ(enc, tempUniforms, `blk.${i}.attn_output`, this.bufs.attnOut, 0, this.bufs.x, T, true);
        }
      }

      // FFN
      this._encodeRMSNorm(enc, tempUniforms, this.bufs.x, `blk.${i}.ffn_norm`, this.bufs.norm, T);
      this._encodeFusedGegluQ(enc, tempUniforms, i, this.bufs.norm, this.bufs.ffnAct, T);
      const ffnPostScalar = bp.has_ple ? 1.0 : spec.layerScalar;
      if (bp.has_post_norms) {
        this._encodeMatmulQ(enc, tempUniforms, `blk.${i}.ffn_down`, this.bufs.ffnAct, 0, this.bufs.branchOut, T, false);
        this._encodePostNormResidual(enc, tempUniforms, this.bufs.branchOut, `blk.${i}.post_ffn_norm`, this.bufs.x, T, ffnPostScalar);
      } else {
        this._encodeMatmulQ(enc, tempUniforms, `blk.${i}.ffn_down`, this.bufs.ffnAct, 0, this.bufs.x, T, true);
      }

      // Gemma 4 PLE injection
      if (bp.has_ple) {
        this._encodeMatmulQ(enc, tempUniforms, `blk.${i}.ple_gate`, this.bufs.x, 0, this.bufs.pleGate, T, false);
        const uGm = this._makeUniform((u32) => {
          u32[0] = bp.num_layers; u32[1] = bp.ple_dim; u32[2] = T; u32[3] = i * this.maxBatchTokens * bp.ple_dim;
        });
        tempUniforms.push(uGm);
        const bgGm = this.device.createBindGroup({
          layout: this.pipelines.gateMulGeluTanh.getBindGroupLayout(0),
          entries: [
            { binding: 0, resource: { buffer: uGm, size: 32 } },
            { binding: 1, resource: { buffer: this.bufs.pleGate } },
            { binding: 2, resource: { buffer: this.bufs.pleCombined } },
            { binding: 4, resource: { buffer: this.bufs.pleMul } },
          ],
        });
        const passGm = enc.beginComputePass();
        passGm.setPipeline(this.pipelines.gateMulGeluTanh);
        passGm.setBindGroup(0, bgGm);
        passGm.dispatchWorkgroups(Math.ceil((T * bp.ple_dim) / 256), 1, 1);
        passGm.end();

        this._encodeMatmulQ(enc, tempUniforms, `blk.${i}.ple_proj`, this.bufs.pleMul, 0, this.bufs.branchOut, T, false);
        this._encodePostNormResidual(enc, tempUniforms, this.bufs.branchOut, `blk.${i}.post_ple_norm`, this.bufs.x, T, spec.layerScalar);
      }

      if (captureTraces && (i === 0 || i === bp.num_layers - 1)) {
        this.device.queue.submit([enc.finish()]);
        const layerData = await this._readBufferFloat32(this.bufs.x, T * D);
        traces[`layer_${i}`] = Array.from(layerData.slice((T - 1) * D, T * D));
        enc = this.device.createCommandEncoder();
      }
    }

    this._encodeRMSNorm(enc, tempUniforms, this.bufs.x, 'output_norm', this.bufs.norm, T);

    let outLogitBuf = this.bufs.logits;
    let outLogitCount = bp.vocab_size;
    if (!skipLogits) {
      if (returnAllLogits) {
        outLogitCount = T * bp.vocab_size;
        if (!this.verifyLogitsBuf || this.verifyLogitsCap < outLogitCount) {
          if (this.verifyLogitsBuf) this.verifyLogitsBuf.destroy();
          this.verifyLogitsBuf = this.device.createBuffer({
            size: alignUp(outLogitCount * 4, 16),
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
          });
          this.verifyLogitsCap = outLogitCount;
        }
        outLogitBuf = this.verifyLogitsBuf;
        this._encodeMatmulQ(enc, tempUniforms, 'output', this.bufs.norm, 0, outLogitBuf, T, false);
      } else {
        this._encodeMatmulQ(enc, tempUniforms, 'output', this.bufs.norm, (T - 1) * D * 4, outLogitBuf, 1, false);
      }

      if (bp.final_logit_softcapping && bp.final_logit_softcapping > 0) {
        const uCap = this._makeUniform((u32, f32) => {
          u32[0] = bp.num_layers; u32[1] = outLogitCount; u32[2] = 1; u32[3] = 1;
          f32[4] = bp.final_logit_softcapping;
        });
        tempUniforms.push(uCap);
        const bgCap = this.device.createBindGroup({
          layout: this.pipelines.softcapLogits.getBindGroupLayout(0),
          entries: [
            { binding: 0, resource: { buffer: uCap, size: 32 } },
            { binding: 4, resource: { buffer: outLogitBuf } },
          ],
        });
        const passCap = enc.beginComputePass();
        passCap.setPipeline(this.pipelines.softcapLogits);
        passCap.setBindGroup(0, bgCap);
        passCap.dispatchWorkgroups(Math.ceil(outLogitCount / 256), 1, 1);
        passCap.end();
      }
    }

    this.device.queue.submit([enc.finish()]);

    if (captureTraces) {
      const fnData = await this._readBufferFloat32(this.bufs.norm, T * D);
      traces.final_norm = Array.from(fnData.slice((T - 1) * D, T * D));
    }

    const logits = skipLogits ? null : await this._readBufferFloat32(outLogitBuf, outLogitCount);
    const lastHidden = returnLastHidden
      ? await this._readBufferFloat32(this.bufs.norm, D, (T - 1) * D * 4)
      : null;
    this.curPos += T;
    for (const ub of tempUniforms) ub.destroy();
    return { logits, lastHidden, traces, gpuMs: null };
  }

  destroy() {
    if (this.embdSource instanceof OpfsEmbeddingTable) {
      this.embdSource.close();
    }
    if (this.pleEmbdSource instanceof OpfsEmbeddingTable) {
      this.pleEmbdSource.close();
    }
    for (const buf of this.staticUniforms) buf.destroy();
    for (const buf of this.gpuShards.values()) buf.destroy();
    for (const buf of Object.values(this.bufs)) buf.destroy();
    if (this.verifyLogitsBuf) this.verifyLogitsBuf.destroy();
    for (const cs of this.convStates) {
      if (cs) cs.destroy();
    }
    for (const cb of this.convBackupBufs || []) {
      if (cb) cb.destroy();
    }
    const freedKv = new Set();
    for (const kv of this.kvCaches) {
      if (!kv || freedKv.has(kv)) continue;
      freedKv.add(kv);
      kv.k.destroy();
      kv.v.destroy();
    }
    if (this.tsQuerySet) this.tsQuerySet.destroy();
    if (this.tsResolveBuf) this.tsResolveBuf.destroy();
    if (this.tsReadBuf) this.tsReadBuf.destroy();
    this.device.destroy();
  }
}
