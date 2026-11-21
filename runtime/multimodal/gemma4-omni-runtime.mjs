/**
 * Gemma-4-E4B-it Full Omni WebGPU Runtime (Stage 5C).
 *
 * Implements:
 * - Gemma4VisionModel (16L SigLIP-like ViT w/ Clipped Linears, 2D H-H-W-W RoPE, q/k/v_norm, 3x3 spatial pool) + EmbedVision -> vision.ignis
 * - Gemma4AudioModel (SubSample 2D Conv s=2 x2 + 12L Conformer w/ Clipped Linears, chunked rel-pos attn, causal DWConv1d) + EmbedAudio -> audio_enc.ignis
 * - Text + Vision + Audio multimodal embedding injection with PLE multimodal masking (`pad_token_id` in `lookupPleEmbeddings`)
 */

import { readOpfsFileBytes } from '../opfs-loader.mjs';
import { sampleToken } from '../sampler.mjs';
import {
  buildFlexAttentionWgsl,
  buildFlexRopeWgsl,
  buildGemma4AudioAttnWgsl,
  buildGemma4AudioConvWgsl,
  buildGemma4PatchPosAddWgsl,
  buildInterleaveQkvWgsl,
  buildMultimodalOpsWgsl,
} from '../../kernels/multimodal-codegen.mjs';
import { extractGemma4LogMel } from './audio-dsp.mjs';
import { preprocessGemma4Image } from './image-proc.mjs';

const IGNIS_MAGIC = 'IGNIS001';

function alignUp(v, align = 256) {
  return Math.ceil(v / align) * align;
}

function validateShardHeader(ab, expectedName) {
  const magic = String.fromCharCode(...new Uint8Array(ab, 0, 8));
  if (magic !== IGNIS_MAGIC) throw new Error(`Invalid magic in ${expectedName} shard: ${magic}`);
}

/**
 * Builds exact Gemma4AudioRelPositionalEncoding sinusoidal table:
 * shape [13, 1024] for dists [12, 11, ..., 0] with [sin..., cos...] concatenation.
 */
function buildGemma4AudioRelPosTable(hiddenSize = 1024, contextSize = 24) {
  const numPos = Math.floor(contextSize / 2) + 1; // 13: [12..0]
  const numTimescales = hiddenSize >> 1; // 512
  const logInc = Math.log(10000.0) / Math.max(numTimescales - 1, 1);
  const invTimescales = new Float64Array(numTimescales);
  for (let k = 0; k < numTimescales; k++) {
    invTimescales[k] = Math.exp(-k * logInc);
  }
  const table = new Float32Array(numPos * hiddenSize);
  for (let p = 0; p < numPos; p++) {
    const posId = Math.floor(contextSize / 2) - p; // 12 .. 0
    const row = p * hiddenSize;
    for (let k = 0; k < numTimescales; k++) {
      const ang = posId * invTimescales[k];
      table[row + k] = Math.sin(ang);
      table[row + numTimescales + k] = Math.cos(ang);
    }
  }
  return table;
}

export class Gemma4OmniRuntime {
  constructor(engine, { modelBaseUrl = null, exclusiveTower = true } = {}) {
    this.engine = engine;
    this.device = engine.device;
    this.manifest = engine.manifest;
    this.bp = engine.bp;
    this.mmBp = engine.bp.multimodal;
    this.mmMeta = engine.manifest.multimodal_tensors || {};
    this.modelBaseUrl = modelBaseUrl;
    this.exclusiveTower = exclusiveTower;

    this.mmShards = new Map();
    this.pipelines = null;
    this.mmOpsBgl = null;
    this.scratch = null;
    this.scratchCapTokens = 0;
    this.scratchMaxDim = 0;
    this.scratchOutTokens = 0;
    this.dsA = null;
    this.dsB = null;
    this.dsCapBytes = 0;
    this.audPosEmbBuf = null;
    this.peakGpuBytes = 0;
  }

  async _loadShardBytes(shardName) {
    const info = this.manifest.shards.find((s) => s.id === shardName);
    if (!info) throw new Error(`Multimodal shard '${shardName}' not found in manifest`);
    if (this.engine.dirHandle) {
      const u8 = await readOpfsFileBytes(this.engine.dirHandle, info.file);
      return u8.buffer.slice(u8.byteOffset, u8.byteOffset + u8.byteLength);
    }
    if (!this.modelBaseUrl) {
      throw new Error(`No dirHandle or modelBaseUrl to load shard '${info.file}'`);
    }
    const resp = await fetch(`${this.modelBaseUrl}/${info.file}`);
    if (!resp.ok) throw new Error(`Failed to fetch ${info.file}: ${resp.status}`);
    return resp.arrayBuffer();
  }

  _uploadShardToGpu(shardName, ab) {
    validateShardHeader(ab, shardName);
    const buf = this.device.createBuffer({
      size: alignUp(ab.byteLength, 4),
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
      label: `${shardName}.ignis`,
    });
    this.device.queue.writeBuffer(buf, 0, ab);
    this.mmShards.set(shardName, buf);
    return buf;
  }

  unloadShard(shardName) {
    const buf = this.mmShards.get(shardName);
    if (buf) {
      buf.destroy();
      this.mmShards.delete(shardName);
    }
  }

  async ensurePipelines() {
    if (this.pipelines) return this.pipelines;
    const dev = this.device;
    const useSubgroups = this.engine.useSubgroups;
    const compile = async (code, label) => {
      const mod = dev.createShaderModule({ code, label });
      const info = await mod.getCompilationInfo();
      const errs = info.messages.filter((m) => m.type === 'error');
      if (errs.length > 0) {
        throw new Error(`WGSL error in ${label}: ${errs.map((e) => `L${e.lineNum}:${e.message}`).join('; ')}`);
      }
      return mod;
    };

    const mods = {
      mmOps: await compile(buildMultimodalOpsWgsl({ useSubgroups }), 'g4_mm_ops'),
      flexRope: await compile(buildFlexRopeWgsl(), 'g4_flex_rope'),
      flexAttn: await compile(buildFlexAttentionWgsl({ useSubgroups }), 'g4_flex_attn'),
      interleaveQkv: await compile(buildInterleaveQkvWgsl(), 'g4_interleave_qkv'),
      patchPosAdd: await compile(buildGemma4PatchPosAddWgsl(), 'g4_patch_pos_add'),
      audConv: await compile(buildGemma4AudioConvWgsl(), 'g4_aud_conv'),
      audAttn: await compile(buildGemma4AudioAttnWgsl({ useSubgroups }), 'g4_aud_attn'),
    };

    const mmOpsBgl = dev.createBindGroupLayout({
      entries: [
        { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' } },
        { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
        { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
        { binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
        { binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
      ],
    });
    const mmOpsLayout = dev.createPipelineLayout({ bindGroupLayouts: [mmOpsBgl] });
    this.mmOpsBgl = mmOpsBgl;

    const mkPipe = (mod, entryPoint, layout = 'auto') => dev.createComputePipeline({ layout, compute: { module: mod, entryPoint } });
    this.pipelines = {
      rmsnorm: mkPipe(mods.mmOps, 'rmsnorm_f32', mmOpsLayout),
      rmsnormNoScale: mkPipe(mods.mmOps, 'rmsnorm_noscale_f32', mmOpsLayout),
      layernormNoBias: mkPipe(mods.mmOps, 'layernorm_nobias_f32', mmOpsLayout),
      clamp: mkPipe(mods.mmOps, 'clamp_f32', mmOpsLayout),
      copyClamp: mkPipe(mods.mmOps, 'copy_clamp_f32', mmOpsLayout),
      relu: mkPipe(mods.mmOps, 'relu_inplace', mmOpsLayout),
      silu: mkPipe(mods.mmOps, 'silu_inplace', mmOpsLayout),
      gluSigmoidHalf: mkPipe(mods.mmOps, 'glu_sigmoid_half', mmOpsLayout),
      gluGeluMul: mkPipe(mods.mmOps, 'glu_silu_mul', mmOpsLayout),
      addBias: mkPipe(mods.mmOps, 'add_bias_f32', mmOpsLayout),
      residualAdd: mkPipe(mods.mmOps, 'residual_add_f32', mmOpsLayout),
      visionPool3x3: mkPipe(mods.mmOps, 'g4_vision_pool_3x3', mmOpsLayout),
      flexRope: mkPipe(mods.flexRope, 'main'),
      flexAttn: mkPipe(mods.flexAttn, 'main'),
      interleaveQkv: mkPipe(mods.interleaveQkv, 'main'),
      patchPosAdd: mkPipe(mods.patchPosAdd, 'main'),
      audSubConv2d: mkPipe(mods.audConv, 'g4_sub_conv2d'),
      audCausalDwConv: mkPipe(mods.audConv, 'g4_causal_dw_conv1d'),
      audRelAttn: mkPipe(mods.audAttn, 'g4_chunked_rel_attn'),
    };
    return this.pipelines;
  }

  async ensureVisionLoaded() {
    await this.ensurePipelines();
    if (this.exclusiveTower) this.unloadShard('audio_enc');
    if (!this.mmShards.has('vision')) {
      const ab = await this._loadShardBytes('vision');
      this._uploadShardToGpu('vision', ab);
    }
    this._recordPeakGpuBytes();
  }

  async ensureAudioEncLoaded() {
    await this.ensurePipelines();
    if (this.exclusiveTower) this.unloadShard('vision');
    if (!this.mmShards.has('audio_enc')) {
      const ab = await this._loadShardBytes('audio_enc');
      this._uploadShardToGpu('audio_enc', ab);
    }
    if (!this.audPosEmbBuf) {
      const ac = this.mmBp.audio_config;
      const ctxSize = (ac.attention_chunk_size || 12) + (ac.attention_context_left || 13) - 1 + (ac.attention_context_right || 0);
      const tbl = buildGemma4AudioRelPosTable(ac.hidden_size || 1024, ctxSize);
      this.audPosEmbBuf = this.device.createBuffer({
        size: tbl.byteLength,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
      });
      this.device.queue.writeBuffer(this.audPosEmbBuf, 0, tbl);
    }
    this._recordPeakGpuBytes();
  }

  getTotalGpuMemoryBytes() {
    let total = 0;
    const e = this.engine;
    for (const buf of e.gpuShards.values()) total += buf.size;
    for (const buf of Object.values(e.bufs)) total += buf.size;
    const seenKv = new Set();
    for (const kv of e.kvCaches) {
      if (!kv || seenKv.has(kv)) continue;
      seenKv.add(kv);
      total += kv.k.size + kv.v.size;
    }
    for (const cs of e.convStates) {
      if (cs) total += cs.size;
    }
    for (const buf of this.mmShards.values()) total += buf.size;
    if (this.scratch) {
      for (const buf of Object.values(this.scratch)) total += buf.size;
    }
    if (this.dsA) total += this.dsA.size;
    if (this.dsB) total += this.dsB.size;
    if (this.audPosEmbBuf) total += this.audPosEmbBuf.size;
    return total;
  }

  _recordPeakGpuBytes() {
    const cur = this.getTotalGpuMemoryBytes();
    if (cur > this.peakGpuBytes) this.peakGpuBytes = cur;
    return cur;
  }

  _mmBinding(name) {
    const t = this.mmMeta[name];
    if (!t) throw new Error(`Multimodal tensor not found: ${name}`);
    const buffer = this.mmShards.get(t.shard);
    if (!buffer) throw new Error(`Multimodal shard '${t.shard}' not loaded`);
    return { buffer, offset: t.offset, size: alignUp(t.bytes, 4), meta: t };
  }

  _ensureScratch(maxTokens, maxDim = 3072, outTokens = maxTokens) {
    if (
      this.scratch &&
      this.scratchCapTokens >= maxTokens &&
      this.scratchMaxDim >= maxDim &&
      this.scratchOutTokens >= outTokens
    ) {
      return this.scratch;
    }
    if (this.scratch) {
      for (const b of Object.values(this.scratch)) b.destroy();
    }
    const T = Math.max(maxTokens, 512);
    const D = maxDim;
    const OT = Math.max(outTokens, 256);
    const mk = (bytes) => this.device.createBuffer({
      size: Math.max(16, alignUp(bytes, 16)),
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
    });
    this.scratch = {
      x: mk(T * 1024 * 4),
      norm: mk(Math.max(T * 1024, OT * 1536) * 4),
      clipTmp: mk(T * D * 4),
      qkv: mk(T * 3 * 1024 * 4),
      qRot: mk(T * 1024 * 4),
      kRot: mk(T * 1024 * 4),
      attnOut: mk(T * 1024 * 4),
      branchOut: mk(Math.max(T * 1024, OT * 1536) * 4),
      ffnGate: mk(T * D * 4),
      ffnUp: mk(T * D * 4),
      projHidden: mk(T * 1024 * 4),
      projOut: mk(OT * 2560 * 4),
      posHw: mk(T * 8),
      audRelK: mk(13 * 1024 * 4),
      dummy: mk(256),
    };
    this.scratchCapTokens = T;
    this.scratchMaxDim = D;
    this.scratchOutTokens = OT;
    this._recordPeakGpuBytes();
    return this.scratch;
  }

  _ensureDsScratch(bytesNeeded) {
    if (this.dsA && this.dsCapBytes >= bytesNeeded) return;
    if (this.dsA) { this.dsA.destroy(); this.dsB.destroy(); }
    const sz = Math.max(bytesNeeded, 1024 * 1024);
    this.dsA = this.device.createBuffer({
      size: alignUp(sz, 16),
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
    });
    this.dsB = this.device.createBuffer({
      size: alignUp(sz, 16),
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
    });
    this.dsCapBytes = sz;
  }

  _mkU(bytes, setupFn) {
    const ab = new ArrayBuffer(bytes);
    setupFn(new Uint32Array(ab), new Float32Array(ab), new Int32Array(ab));
    const buf = this.device.createBuffer({
      size: alignUp(bytes, 16),
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });
    this.device.queue.writeBuffer(buf, 0, ab);
    return buf;
  }

  _encMmOp(enc, temp, pipeName, { a = 0, b = 0, c = 0, d = 0, eps = 1e-6, f0 = 1.0, f1 = 1.0, w = null, bias = null, x = null, out, wgX }) {
    const sc = this.scratch;
    const dummy = { buffer: sc.dummy, offset: 0, size: 64 };
    const wb = w || dummy;
    const bb = bias || dummy;
    const xb = x || sc.dummy;
    const u = this._mkU(32, (u32, f32) => {
      u32[0] = a; u32[1] = b; u32[2] = c; u32[3] = d;
      f32[4] = eps; f32[5] = f0; f32[6] = f1;
    });
    temp.push(u);
    const pipe = this.pipelines[pipeName];
    const bg = this.device.createBindGroup({
      layout: this.mmOpsBgl,
      entries: [
        { binding: 0, resource: { buffer: u, size: 32 } },
        { binding: 1, resource: { buffer: wb.buffer, offset: wb.offset, size: wb.size } },
        { binding: 2, resource: { buffer: bb.buffer, offset: bb.offset, size: bb.size } },
        { binding: 3, resource: { buffer: xb } },
        { binding: 4, resource: { buffer: out } },
      ],
    });
    const pass = enc.beginComputePass();
    pass.setPipeline(pipe);
    pass.setBindGroup(0, bg);
    pass.dispatchWorkgroups(wgX, 1, 1);
    pass.end();
  }

  _encMmQ8(enc, temp, name, xBuf, outBuf, T) {
    const q = this._mmBinding(`${name}.quants`);
    const s = this._mmBinding(`${name}.scales`);
    const [M, K] = q.meta.orig_shape;
    const [, K_pad] = q.meta.padded_shape;
    const u = this._mkU(16, (u32) => { u32[0] = M; u32[1] = K; u32[2] = T; u32[3] = K_pad; });
    temp.push(u);
    const pipe = T === 1 ? this.engine.pipelines.gemvQ8 : this.engine.pipelines.gemm4Q8;
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
    const pass = enc.beginComputePass();
    pass.setPipeline(pipe);
    pass.setBindGroup(0, bg);
    pass.dispatchWorkgroups(Math.ceil(M / 4), T === 1 ? 1 : Math.ceil(T / 4), 1);
    pass.end();
  }

  _encMmQ8Clipped(enc, temp, name, xBuf, outBuf, T) {
    const qMeta = this.mmMeta[`${name}.quants`];
    const clip = qMeta?.clip;
    const [M, K] = qMeta.orig_shape;
    let srcBuf = xBuf;
    if (clip && Number.isFinite(clip[0]) && Number.isFinite(clip[1])) {
      const sc = this.scratch;
      const inCount = T * K;
      this._encMmOp(enc, temp, 'copyClamp', {
        a: inCount,
        f0: clip[0],
        f1: clip[1],
        x: xBuf,
        out: sc.clipTmp,
        wgX: Math.ceil(inCount / 256),
      });
      srcBuf = sc.clipTmp;
    }
    this._encMmQ8(enc, temp, name, srcBuf, outBuf, T);
    if (clip && Number.isFinite(clip[2]) && Number.isFinite(clip[3])) {
      const outCount = T * M;
      this._encMmOp(enc, temp, 'clamp', {
        a: outCount,
        f0: clip[2],
        f1: clip[3],
        out: outBuf,
        wgX: Math.ceil(outCount / 256),
      });
    }
  }

  _encRmsNorm(enc, temp, wName, xBuf, outBuf, D, T, eps = 1e-6) {
    const w = this._mmBinding(wName);
    this._encMmOp(enc, temp, 'rmsnorm', { a: D, b: T, eps, w, x: xBuf, out: outBuf, wgX: T });
  }

  _encRmsNormNoScale(enc, temp, xBuf, outBuf, D, T, eps = 1e-6) {
    this._encMmOp(enc, temp, 'rmsnormNoScale', { a: D, b: T, eps, x: xBuf, out: outBuf, wgX: T });
  }

  _encPostNormRes(enc, temp, wName, branchBuf, resBuf, D, T, eps = 1e-6, scalar = 1.0) {
    const w = this._mmBinding(wName);
    const u = this._mkU(16, (u32, f32) => { u32[0] = D; u32[1] = T; f32[2] = eps; f32[3] = scalar; });
    temp.push(u);
    const pipe = this.engine.pipelines.postNormResidual;
    const bg = this.device.createBindGroup({
      layout: pipe.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: u, size: 16 } },
        { binding: 1, resource: { buffer: branchBuf } },
        { binding: 2, resource: { buffer: w.buffer, offset: w.offset, size: w.size } },
        { binding: 3, resource: { buffer: resBuf } },
      ],
    });
    const pass = enc.beginComputePass();
    pass.setPipeline(pipe);
    pass.setBindGroup(0, bg);
    pass.dispatchWorkgroups(T, 1, 1);
    pass.end();
  }

  async encodeImage(imageOrRgba, optW = null, optH = null, opts = {}) {
    await this.ensureVisionLoaded();
    const vc = this.mmBp.vision_config;
    const imgObj = imageOrRgba instanceof Uint8Array || imageOrRgba instanceof Uint8ClampedArray
      ? {
          data: imageOrRgba,
          width: optW,
          height: optH,
          channels: optW && optH && imageOrRgba.length === optW * optH * 3 ? 3 : 4,
        }
      : imageOrRgba;

    const maxSoftTokens = opts.maxSoftTokens || vc.default_output_length || 280;
    const pre = preprocessGemma4Image(imgObj, {
      patchSize: vc.patch_size || 16,
      maxSoftTokens,
      poolingKernelSize: vc.pooling_kernel_size || 3,
    });
    const T = pre.numPatches;
    const outLen = pre.numVisionTokens;
    const D = vc.hidden_size || 768;
    const nh = vc.num_attention_heads || 12;
    const hd = vc.head_dim || 64;
    const eps = vc.rms_norm_eps || 1e-6;
    const theta = vc.rope_parameters?.rope_theta || 100.0;
    const outD = this.bp.hidden_size; // 2560

    const sc = this._ensureScratch(T, vc.intermediate_size || 3072, outLen);
    this.device.queue.writeBuffer(sc.projHidden, 0, pre.pixelValues);
    this.device.queue.writeBuffer(sc.posHw, 0, pre.positionIds);

    const temp = [];
    const enc = this.device.createCommandEncoder();

    // 1. Patch embedder: input_proj + 2D position table
    this._encMmQ8(enc, temp, 'vision.patch_proj', sc.projHidden, sc.x, T);
    const posTbl = this._mmBinding('vision.pos_table');
    const uPosAdd = this._mkU(16, (u32) => {
      u32[0] = T; u32[1] = D; u32[2] = vc.position_embedding_size || 10240; u32[3] = 0;
    });
    temp.push(uPosAdd);
    const bgPosAdd = this.device.createBindGroup({
      layout: this.pipelines.patchPosAdd.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: uPosAdd, size: 16 } },
        { binding: 1, resource: { buffer: posTbl.buffer, offset: posTbl.offset, size: posTbl.size } },
        { binding: 2, resource: { buffer: sc.posHw } },
        { binding: 3, resource: { buffer: sc.x } },
      ],
    });
    const passPA = enc.beginComputePass();
    passPA.setPipeline(this.pipelines.patchPosAdd);
    passPA.setBindGroup(0, bgPosAdd);
    passPA.dispatchWorkgroups(Math.ceil((T * D) / 256), 1, 1);
    passPA.end();

    // 2. 16 Vision Transformer layers
    const L = vc.num_hidden_layers || 16;
    for (let i = 0; i < L; i++) {
      const pfx = `vision.blk.${i}`;
      this._encRmsNorm(enc, temp, `${pfx}.input_layernorm`, sc.x, sc.norm, D, T, eps);

      this._encMmQ8Clipped(enc, temp, `${pfx}.q_proj`, sc.norm, sc.projHidden, T);
      this._encRmsNorm(enc, temp, `${pfx}.q_norm`, sc.projHidden, sc.qRot, hd, T * nh, eps);

      this._encMmQ8Clipped(enc, temp, `${pfx}.k_proj`, sc.norm, sc.projHidden, T);
      this._encRmsNorm(enc, temp, `${pfx}.k_norm`, sc.projHidden, sc.kRot, hd, T * nh, eps);

      this._encMmQ8Clipped(enc, temp, `${pfx}.v_proj`, sc.norm, sc.projHidden, T);
      this._encRmsNormNoScale(enc, temp, sc.projHidden, sc.attnOut, hd, T * nh, eps);

      // Pack [qRot, kRot, attnOut] -> sc.qkv
      const uPack = this._mkU(16, (u32) => { u32[0] = T; u32[1] = D; u32[2] = D; u32[3] = 0; });
      temp.push(uPack);
      const bgPack = this.device.createBindGroup({
        layout: this.pipelines.interleaveQkv.getBindGroupLayout(0),
        entries: [
          { binding: 0, resource: { buffer: uPack, size: 16 } },
          { binding: 1, resource: { buffer: sc.qRot } },
          { binding: 2, resource: { buffer: sc.kRot } },
          { binding: 3, resource: { buffer: sc.attnOut } },
          { binding: 4, resource: { buffer: sc.qkv } },
        ],
      });
      const passPk = enc.beginComputePass();
      passPk.setPipeline(this.pipelines.interleaveQkv);
      passPk.setBindGroup(0, bgPack);
      passPk.dispatchWorkgroups(Math.ceil((T * 3 * D) / 256), 1, 1);
      passPk.end();

      // 2D H-H-W-W RoPE (mode=3, theta=100.0)
      const uR = this._mkU(32, (u32, f32) => {
        u32[0] = T; u32[1] = nh; u32[2] = nh; u32[3] = hd;
        u32[4] = 3; f32[5] = theta;
      });
      temp.push(uR);
      const bgR = this.device.createBindGroup({
        layout: this.pipelines.flexRope.getBindGroupLayout(0),
        entries: [
          { binding: 0, resource: { buffer: uR, size: 32 } },
          { binding: 1, resource: { buffer: sc.qkv } },
          { binding: 2, resource: { buffer: sc.posHw } },
          { binding: 3, resource: { buffer: sc.qRot } },
          { binding: 4, resource: { buffer: sc.kRot } },
        ],
      });
      const passR = enc.beginComputePass();
      passR.setPipeline(this.pipelines.flexRope);
      passR.setBindGroup(0, bgR);
      passR.dispatchWorkgroups(Math.ceil(D / 128), T, 1);
      passR.end();

      // Full bidir attention with scale = 1.0
      const uA = this._mkU(32, (u32, f32) => {
        u32[0] = T; u32[1] = nh; u32[2] = nh; u32[3] = hd;
        u32[4] = 0; u32[5] = 0; u32[6] = 0; f32[7] = 1.0;
      });
      temp.push(uA);
      const bgA = this.device.createBindGroup({
        layout: this.pipelines.flexAttn.getBindGroupLayout(0),
        entries: [
          { binding: 0, resource: { buffer: uA, size: 32 } },
          { binding: 1, resource: { buffer: sc.qRot } },
          { binding: 2, resource: { buffer: sc.kRot } },
          { binding: 3, resource: { buffer: sc.qkv } },
          { binding: 4, resource: { buffer: sc.dummy } },
          { binding: 5, resource: { buffer: sc.attnOut } },
        ],
      });
      const passA = enc.beginComputePass();
      passA.setPipeline(this.pipelines.flexAttn);
      passA.setBindGroup(0, bgA);
      passA.dispatchWorkgroups(nh, T, 1);
      passA.end();

      this._encMmQ8Clipped(enc, temp, `${pfx}.o_proj`, sc.attnOut, sc.branchOut, T);
      this._encPostNormRes(enc, temp, `${pfx}.post_attention_layernorm`, sc.branchOut, sc.x, D, T, eps, 1.0);

      this._encRmsNorm(enc, temp, `${pfx}.pre_feedforward_layernorm`, sc.x, sc.norm, D, T, eps);
      this._encMmQ8Clipped(enc, temp, `${pfx}.gate_proj`, sc.norm, sc.ffnGate, T);
      this._encMmQ8Clipped(enc, temp, `${pfx}.up_proj`, sc.norm, sc.ffnUp, T);
      const actN = T * (vc.intermediate_size || 3072);
      this._encMmOp(enc, temp, 'gluGeluMul', {
        a: actN, c: 1, x: sc.ffnGate, out: sc.ffnUp, wgX: Math.ceil(actN / 256),
      });
      this._encMmQ8Clipped(enc, temp, `${pfx}.down_proj`, sc.ffnUp, sc.branchOut, T);
      this._encPostNormRes(enc, temp, `${pfx}.post_feedforward_layernorm`, sc.branchOut, sc.x, D, T, eps, 1.0);
    }

    // 3. 3x3 average pooling * sqrt(D) + embed_vision
    const kPool = vc.pooling_kernel_size || 3;
    this._encMmOp(enc, temp, 'visionPool3x3', {
      a: D,
      b: pre.patchW,
      c: kPool,
      d: outLen,
      f0: Math.sqrt(D),
      x: sc.x,
      out: sc.norm,
      wgX: Math.ceil((outLen * D) / 256),
    });

    this._encRmsNormNoScale(enc, temp, sc.norm, sc.projHidden, D, outLen, eps);
    this._encMmQ8(enc, temp, 'vision.embed_proj', sc.projHidden, sc.projOut, outLen);

    this.device.queue.submit([enc.finish()]);
    const embeds = await this.engine._readBufferFloat32(sc.projOut, outLen * outD);
    for (const ub of temp) ub.destroy();

    return {
      embeds,
      numTokens: outLen,
      patchH: pre.patchH,
      patchW: pre.patchW,
    };
  }

  async encodeAudio(pcm16k) {
    await this.ensureAudioEncLoaded();
    const ac = this.mmBp.audio_config;
    const mel = extractGemma4LogMel(pcm16k);
    const T0 = mel.numFrames;
    const F0 = mel.numMelBins; // 128
    const t1 = (T0 + 1) >> 1;
    const f1 = (F0 + 1) >> 1; // 64
    const c1 = ac.subsampling_conv_channels?.[0] || 128;
    const t2 = (t1 + 1) >> 1;
    const f2 = (f1 + 1) >> 1; // 32
    const c2 = ac.subsampling_conv_channels?.[1] || 32;
    const validT = mel.validAudioTokens;
    const D = ac.hidden_size || 1024;
    const nh = ac.num_attention_heads || 8;
    const hd = Math.floor(D / nh); // 128
    const eps = ac.rms_norm_eps || 1e-6;
    const leftCtx = (ac.attention_context_left || 13) - 1; // 12
    const outD = this.bp.hidden_size; // 2560

    const maxDsElems = Math.max(T0 * F0, t1 * f1 * c1, t2 * f2 * c2);
    this._ensureDsScratch(maxDsElems * 4);
    const sc = this._ensureScratch(Math.max(validT, 64), D * 4, validT);
    this._recordPeakGpuBytes();

    this.device.queue.writeBuffer(this.dsA, 0, mel.features);
    const temp = [];
    const enc = this.device.createCommandEncoder();

    const dispatchConv = (paramsArr, inBuf, wName, outBuf, totalElems) => {
      const w = this._mmBinding(wName);
      const u = this._mkU(32, (u32) => { u32.set(paramsArr); });
      temp.push(u);
      const bg = this.device.createBindGroup({
        layout: this.pipelines.audSubConv2d.getBindGroupLayout(0),
        entries: [
          { binding: 0, resource: { buffer: u, size: 32 } },
          { binding: 1, resource: { buffer: inBuf } },
          { binding: 2, resource: { buffer: w.buffer, offset: w.offset, size: w.size } },
          { binding: 3, resource: { buffer: outBuf } },
        ],
      });
      const pass = enc.beginComputePass();
      pass.setPipeline(this.pipelines.audSubConv2d);
      pass.setBindGroup(0, bg);
      pass.dispatchWorkgroups(Math.ceil(totalElems / 256), 1, 1);
      pass.end();
    };

    // SubSample Layer 0: [T0, 128, 1] -> [t1, 64, 128]
    dispatchConv([T0, F0, 1, t1, f1, c1, mel.validMelFrames, 0], this.dsA, 'audio.sub.layer0.conv', this.dsB, t1 * f1 * c1);
    const wLn0 = this._mmBinding('audio.sub.layer0.norm');
    this._encMmOp(enc, temp, 'layernormNoBias', { a: c1, b: t1 * f1, eps, w: wLn0, x: this.dsB, out: this.dsA, wgX: t1 * f1 });
    this._encMmOp(enc, temp, 'relu', { a: t1 * f1 * c1, out: this.dsA, wgX: Math.ceil((t1 * f1 * c1) / 256) });

    // SubSample Layer 1: [t1, 64, 128] -> [t2, 32, 32]
    dispatchConv([t1, f1, c1, t2, f2, c2, mel.validSub1, 0], this.dsA, 'audio.sub.layer1.conv', this.dsB, t2 * f2 * c2);
    const wLn1 = this._mmBinding('audio.sub.layer1.norm');
    this._encMmOp(enc, temp, 'layernormNoBias', { a: c2, b: t2 * f2, eps, w: wLn1, x: this.dsB, out: this.dsA, wgX: t2 * f2 });
    this._encMmOp(enc, temp, 'relu', { a: t2 * f2 * c2, out: this.dsA, wgX: Math.ceil((t2 * f2 * c2) / 256) });

    // Trim validT rows -> input_proj_linear [validT, 1024] -> sc.x
    this._encMmQ8(enc, temp, 'audio.sub.input_proj', this.dsA, sc.x, validT);

    // 12 Conformer layers
    const L = ac.num_hidden_layers || 12;
    const qScale = (hd ** -0.5) / Math.LN2;
    const kScale = Math.log(1.0 + Math.E) / Math.LN2;
    const resW = ac.residual_weight ?? 0.5;
    const numRelPos = leftCtx + 1;

    const runFfn = (pfx) => {
      this._encRmsNorm(enc, temp, `${pfx}.pre_norm`, sc.x, sc.norm, D, validT, eps);
      this._encMmQ8Clipped(enc, temp, `${pfx}.w1`, sc.norm, sc.ffnGate, validT);
      const nAct = validT * D * 4;
      this._encMmOp(enc, temp, 'silu', { a: nAct, out: sc.ffnGate, wgX: Math.ceil(nAct / 256) });
      this._encMmQ8Clipped(enc, temp, `${pfx}.w2`, sc.ffnGate, sc.branchOut, validT);
      this._encRmsNorm(enc, temp, `${pfx}.post_norm`, sc.branchOut, sc.norm, D, validT, eps);
      this._encMmOp(enc, temp, 'residualAdd', {
        a: validT * D, f0: resW, x: sc.norm, out: sc.x, wgX: Math.ceil((validT * D) / 256),
      });
    };

    for (let i = 0; i < L; i++) {
      const pfx = `audio.blk.${i}`;
      runFfn(`${pfx}.ff1`);

      // Chunked relative attention
      this._encRmsNorm(enc, temp, `${pfx}.norm_pre_attn`, sc.x, sc.norm, D, validT, eps);
      this._encMmQ8Clipped(enc, temp, `${pfx}.q_proj`, sc.norm, sc.qRot, validT);
      this._encMmQ8Clipped(enc, temp, `${pfx}.k_proj`, sc.norm, sc.kRot, validT);
      this._encMmQ8Clipped(enc, temp, `${pfx}.v_proj`, sc.norm, sc.qkv, validT);
      this._encMmQ8(enc, temp, `${pfx}.rel_k_proj`, this.audPosEmbBuf, sc.audRelK, numRelPos);

      const perDim = this._mmBinding(`${pfx}.per_dim_scale`);
      const uAtt = this._mkU(32, (u32, f32) => {
        u32[0] = validT; u32[1] = nh; u32[2] = hd; u32[3] = leftCtx;
        f32[4] = qScale; f32[5] = kScale;
      });
      temp.push(uAtt);
      const bgAtt = this.device.createBindGroup({
        layout: this.pipelines.audRelAttn.getBindGroupLayout(0),
        entries: [
          { binding: 0, resource: { buffer: uAtt, size: 32 } },
          { binding: 1, resource: { buffer: sc.qRot } },
          { binding: 2, resource: { buffer: sc.kRot } },
          { binding: 3, resource: { buffer: sc.qkv } },
          { binding: 4, resource: { buffer: sc.audRelK } },
          { binding: 5, resource: { buffer: perDim.buffer, offset: perDim.offset, size: perDim.size } },
          { binding: 6, resource: { buffer: sc.attnOut } },
        ],
      });
      const passAt = enc.beginComputePass();
      passAt.setPipeline(this.pipelines.audRelAttn);
      passAt.setBindGroup(0, bgAtt);
      passAt.dispatchWorkgroups(nh, validT, 1);
      passAt.end();

      this._encMmQ8Clipped(enc, temp, `${pfx}.post_proj`, sc.attnOut, sc.branchOut, validT);
      this._encPostNormRes(enc, temp, `${pfx}.norm_post_attn`, sc.branchOut, sc.x, D, validT, eps, 1.0);

      // LConv1D: norm -> lin_start -> glu(sigmoid) -> causal_dw_conv(5) -> conv_norm -> silu -> lin_end + residual
      this._encRmsNorm(enc, temp, `${pfx}.lconv.pre_norm`, sc.x, sc.norm, D, validT, eps);
      this._encMmQ8Clipped(enc, temp, `${pfx}.lconv.lin_start`, sc.norm, sc.ffnGate, validT);
      this._encMmOp(enc, temp, 'gluSigmoidHalf', {
        a: D, b: validT, x: sc.ffnGate, out: sc.norm, wgX: Math.ceil((validT * D) / 256),
      });

      const dwW = this._mmBinding(`${pfx}.lconv.dw_conv`);
      const uDw = this._mkU(32, (u32) => { u32[0] = validT; u32[1] = D; u32[2] = ac.conv_kernel_size || 5; });
      temp.push(uDw);
      const bgDw = this.device.createBindGroup({
        layout: this.pipelines.audCausalDwConv.getBindGroupLayout(0),
        entries: [
          { binding: 0, resource: { buffer: uDw, size: 32 } },
          { binding: 1, resource: { buffer: sc.norm } },
          { binding: 2, resource: { buffer: dwW.buffer, offset: dwW.offset, size: dwW.size } },
          { binding: 3, resource: { buffer: sc.attnOut } },
        ],
      });
      const passDw = enc.beginComputePass();
      passDw.setPipeline(this.pipelines.audCausalDwConv);
      passDw.setBindGroup(0, bgDw);
      passDw.dispatchWorkgroups(Math.ceil((validT * D) / 256), 1, 1);
      passDw.end();

      this._encRmsNorm(enc, temp, `${pfx}.lconv.conv_norm`, sc.attnOut, sc.norm, D, validT, eps);
      this._encMmOp(enc, temp, 'silu', { a: validT * D, out: sc.norm, wgX: Math.ceil((validT * D) / 256) });
      this._encMmQ8Clipped(enc, temp, `${pfx}.lconv.lin_end`, sc.norm, sc.branchOut, validT);
      this._encMmOp(enc, temp, 'residualAdd', {
        a: validT * D, f0: 1.0, x: sc.branchOut, out: sc.x, wgX: Math.ceil((validT * D) / 256),
      });

      runFfn(`${pfx}.ff2`);

      this._encRmsNorm(enc, temp, `${pfx}.norm_out`, sc.x, sc.norm, D, validT, eps);
      enc.copyBufferToBuffer(sc.norm, 0, sc.x, 0, validT * D * 4);
    }

    // output_proj + bias -> embed_audio
    const outDims = ac.output_proj_dims || 1536;
    this._encMmQ8(enc, temp, 'audio.output_proj', sc.x, sc.branchOut, validT);
    const outBias = this._mmBinding('audio.output_proj_b');
    this._encMmOp(enc, temp, 'addBias', {
      a: outDims, b: validT, bias: outBias, out: sc.branchOut, wgX: Math.ceil((validT * outDims) / 256),
    });
    this._encRmsNormNoScale(enc, temp, sc.branchOut, sc.norm, outDims, validT, eps);
    this._encMmQ8(enc, temp, 'audio.embed_proj', sc.norm, sc.projOut, validT);

    this.device.queue.submit([enc.finish()]);
    const embeds = await this.engine._readBufferFloat32(sc.projOut, validT * outD);
    for (const ub of temp) ub.destroy();

    return { embeds, numTokens: validT };
  }

  /**
   * Expands single placeholder occurrences (`<|image|>`, `<|audio|>`) or uses pre-expanded placeholder IDs,
   * replaces multimodal slots with soft token embeddings (without double-scaling by sqrt(D)),
   * and returns `{ tokenIds, embeds, isMultimodalMask }`.
   */
  async buildOmniPromptEmbeddings(promptIds, { imageEmbeds = [], audioEmbeds = [] } = {}) {
    const mm = this.mmBp;
    const D = this.bp.hidden_size;
    const imgTok = mm.image_token_id;
    const audTok = mm.audio_token_id;
    const boiTok = mm.boi_token_id;
    const eoiTok = mm.eoi_token_id;
    const boaTok = mm.boa_token_id;
    const eoaTok = mm.eoa_token_id;
    const padId = this.bp.pad_token_id ?? 0;

    let rawImgSlots = 0;
    let rawAudSlots = 0;
    for (const id of promptIds) {
      if (id === imgTok) rawImgSlots++;
      else if (id === audTok) rawAudSlots++;
    }
    const preExpandedImg = rawImgSlots > 0 && rawImgSlots === imageEmbeds.reduce((s, e) => s + e.numTokens, 0);
    const preExpandedAud = rawAudSlots > 0 && rawAudSlots === audioEmbeds.reduce((s, e) => s + e.numTokens, 0);

    const expandedIds = [];
    const lookupIds = [];
    const isMultimodalMask = [];
    let imgIdx = 0;
    let audIdx = 0;

    for (let i = 0; i < promptIds.length; i++) {
      const id = promptIds[i];
      if (!preExpandedImg && id === boiTok && promptIds[i + 1] !== imgTok && imgIdx < imageEmbeds.length) {
        expandedIds.push(boiTok);
        lookupIds.push(boiTok);
        isMultimodalMask.push(0);
        const cnt = imageEmbeds[imgIdx++].numTokens;
        for (let k = 0; k < cnt; k++) {
          expandedIds.push(imgTok);
          lookupIds.push(padId);
          isMultimodalMask.push(1);
        }
        if (promptIds[i + 1] !== eoiTok) {
          expandedIds.push(eoiTok);
          lookupIds.push(eoiTok);
          isMultimodalMask.push(0);
        }
      } else if (!preExpandedImg && id === imgTok && imgIdx < imageEmbeds.length) {
        if (expandedIds[expandedIds.length - 1] !== boiTok) {
          expandedIds.push(boiTok);
          lookupIds.push(boiTok);
          isMultimodalMask.push(0);
        }
        const cnt = imageEmbeds[imgIdx++].numTokens;
        for (let k = 0; k < cnt; k++) {
          expandedIds.push(imgTok);
          lookupIds.push(padId);
          isMultimodalMask.push(1);
        }
        if (promptIds[i + 1] !== eoiTok) {
          expandedIds.push(eoiTok);
          lookupIds.push(eoiTok);
          isMultimodalMask.push(0);
        }
      } else if (!preExpandedAud && id === boaTok && promptIds[i + 1] !== audTok && audIdx < audioEmbeds.length) {
        expandedIds.push(boaTok);
        lookupIds.push(boaTok);
        isMultimodalMask.push(0);
        const cnt = audioEmbeds[audIdx++].numTokens;
        for (let k = 0; k < cnt; k++) {
          expandedIds.push(audTok);
          lookupIds.push(padId);
          isMultimodalMask.push(1);
        }
        if (promptIds[i + 1] !== eoaTok) {
          expandedIds.push(eoaTok);
          lookupIds.push(eoaTok);
          isMultimodalMask.push(0);
        }
      } else if (!preExpandedAud && id === audTok && audIdx < audioEmbeds.length) {
        if (expandedIds[expandedIds.length - 1] !== boaTok) {
          expandedIds.push(boaTok);
          lookupIds.push(boaTok);
          isMultimodalMask.push(0);
        }
        const cnt = audioEmbeds[audIdx++].numTokens;
        for (let k = 0; k < cnt; k++) {
          expandedIds.push(audTok);
          lookupIds.push(padId);
          isMultimodalMask.push(1);
        }
        if (promptIds[i + 1] !== eoaTok) {
          expandedIds.push(eoaTok);
          lookupIds.push(eoaTok);
          isMultimodalMask.push(0);
        }
      } else {
        const isMm = id === imgTok || id === audTok;
        expandedIds.push(id);
        lookupIds.push(isMm ? padId : id);
        isMultimodalMask.push(isMm ? 1 : 0);
      }
    }

    const emb = await this.engine.lookupEmbeddings(lookupIds);
    const imgFlat = imageEmbeds.length > 0
      ? imageEmbeds.reduce((acc, item) => { const n = new Float32Array(acc.length + item.embeds.length); n.set(acc); n.set(item.embeds, acc.length); return n; }, new Float32Array(0))
      : null;
    const audFlat = audioEmbeds.length > 0
      ? audioEmbeds.reduce((acc, item) => { const n = new Float32Array(acc.length + item.embeds.length); n.set(acc); n.set(item.embeds, acc.length); return n; }, new Float32Array(0))
      : null;

    let iOff = 0;
    let aOff = 0;
    for (let t = 0; t < expandedIds.length; t++) {
      if (expandedIds[t] === imgTok && imgFlat) {
        emb.set(imgFlat.subarray(iOff * D, (iOff + 1) * D), t * D);
        iOff++;
      } else if (expandedIds[t] === audTok && audFlat) {
        emb.set(audFlat.subarray(aOff * D, (aOff + 1) * D), t * D);
        aOff++;
      }
    }

    return { tokenIds: expandedIds, embeds: emb, isMultimodalMask };
  }

  async generateOmni(promptIds, {
    images = [],
    audios = [],
    maxNewTokens = 128,
    temperature = 0.0,
    topK = 40,
    repetitionPenalty = 1.0,
    onTextToken = null,
    shouldAbort = null,
    releaseVisionAfterEncode = false,
    releaseAudioAfterEncode = false,
    rng = Math.random,
  } = {}) {
    const imgEmbeds = [];
    for (const im of images) {
      imgEmbeds.push(await this.encodeImage(im));
    }
    if (releaseVisionAfterEncode && images.length > 0) this.unloadShard('vision');

    const audEmbeds = [];
    for (const au of audios) {
      audEmbeds.push(await this.encodeAudio(au));
    }
    if (releaseAudioAfterEncode && audios.length > 0) this.unloadShard('audio_enc');

    const { tokenIds, embeds, isMultimodalMask } = await this.buildOmniPromptEmbeddings(promptIds, {
      imageEmbeds: imgEmbeds,
      audioEmbeds: audEmbeds,
    });

    this.engine.reset();
    let fwd = await this.engine.forwardEmbeds(embeds, {
      tokenIds,
      isMultimodalMask,
    });

    const eosSet = new Set(this.bp.eos_token_ids || [this.bp.eos_token_id]);
    const suppress = new Set([
      this.mmBp.image_token_id,
      this.mmBp.audio_token_id,
      this.mmBp.boi_token_id,
      this.mmBp.eoi_token_id,
      this.mmBp.boa_token_id,
      this.mmBp.eoa_token_id,
    ]);

    const outIds = [];
    for (let step = 0; step < maxNewTokens; step++) {
      if (shouldAbort && shouldAbort()) break;
      const logits = new Float32Array(fwd.logits);
      for (const sid of suppress) {
        if (sid != null && sid >= 0 && sid < logits.length) logits[sid] = -Infinity;
      }
      const tok = sampleToken(logits, outIds.slice(-32), {
        temperature,
        topK,
        repetitionPenalty,
        rng,
      });
      if (eosSet.has(tok)) break;
      outIds.push(tok);
      if (onTextToken) onTextToken(tok, step);
      if (step === maxNewTokens - 1) break;
      fwd = await this.engine.forward([tok]);
    }
    return { textTokenIds: outIds };
  }

  destroy() {
    if (this.audPosEmbBuf) this.audPosEmbBuf.destroy();
    if (this.dsA) this.dsA.destroy();
    if (this.dsB) this.dsB.destroy();
    if (this.scratch) {
      for (const b of Object.values(this.scratch)) b.destroy();
    }
    for (const buf of this.mmShards.values()) buf.destroy();
    this.mmShards.clear();
  }
}
