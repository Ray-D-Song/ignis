/**
 * LFM3-Omni-2B Full Multimodal WebGPU Runtime (Stage 5B).
 *
 * Implements:
 * - Lfm3VlVisionModel (14L ShortConv 3x3 + Bidir GQA w/ 2D Axial RoPE + Projector) -> vision.ignis
 * - Lfm3OmniDownsampler + 17L Lfm3OmniEncoder + AudioProjector -> audio_enc.ignis
 * - CodebookEmbedding + Lfm3OmniDepthDecoder (MaskGIT [4,4,4]) + LiquidCodecDecoder (24 kHz iSTFT) -> audio_dec.ignis
 * - Unified text + image + audio-in + speech/interleaved-out generation via IgnisEngine
 */

import { readOpfsFileBytes } from '../opfs-loader.mjs';
import { sampleToken } from '../sampler.mjs';
import {
  buildAudioConv2dWgsl,
  buildFlexAttentionWgsl,
  buildFlexRopeWgsl,
  buildLinearF16Wgsl,
  buildMultimodalOpsWgsl,
  buildVisionConv2dWgsl,
} from '../../kernels/multimodal-codegen.mjs';
import { extractLfm3LogMel, liquidCodecIstft, encodeWavPcm16 } from './audio-dsp.mjs';
import { computeVisionPositionIds, preprocessLfm3Image } from './image-proc.mjs';

const IGNIS_MAGIC = 'IGNIS001';

function alignUp(v, align = 256) {
  return Math.ceil(v / align) * align;
}

function f16ToF32(h) {
  const s = (h & 0x8000) >> 15;
  const e = (h & 0x7c00) >> 10;
  const f = h & 0x03ff;
  if (e === 0) return (s ? -1 : 1) * Math.pow(2, -14) * (f / 1024);
  if (e === 0x1f) return f ? NaN : (s ? -Infinity : Infinity);
  return (s ? -1 : 1) * Math.pow(2, e - 15) * (1 + f / 1024);
}

function validateShardHeader(ab, expectedName) {
  const magic = String.fromCharCode(...new Uint8Array(ab, 0, 8));
  if (magic !== IGNIS_MAGIC) throw new Error(`Invalid magic in ${expectedName} shard: ${magic}`);
}

export class Lfm3OmniRuntime {
  constructor(engine, { modelBaseUrl = null } = {}) {
    this.engine = engine;
    this.device = engine.device;
    this.manifest = engine.manifest;
    this.bp = engine.bp;
    this.mmBp = engine.bp.multimodal;
    this.mmMeta = engine.manifest.multimodal_tensors || {};
    this.modelBaseUrl = modelBaseUrl;

    this.mmShards = new Map(); // shardName -> GPUBuffer
    this.pipelines = null;
    this.staticUniforms = [];
    this.scratch = null;
    this.scratchCapTokens = 0;
    this.scratchMaxDim = 0;

    // CPU tables extracted from audio_dec.ignis
    this.codebookEmbdBytes = null;
    this.depthCbF16 = null; // Array(12) of Uint16Array(2017 * 1024)
    this.depthMaskEmb = null;
    this.codecCpu = null;
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
      mmOps: await compile(buildMultimodalOpsWgsl({ useSubgroups }), 'mm_ops'),
      linF16: await compile(buildLinearF16Wgsl({ useSubgroups }), 'mm_linear_f16'),
      visConv: await compile(buildVisionConv2dWgsl(), 'mm_vis_conv2d'),
      audConv: await compile(buildAudioConv2dWgsl(), 'mm_aud_conv2d'),
      flexRope: await compile(buildFlexRopeWgsl(), 'mm_flex_rope'),
      flexAttn: await compile(buildFlexAttentionWgsl({ useSubgroups }), 'mm_flex_attn'),
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
      layernorm: mkPipe(mods.mmOps, 'layernorm_f32', mmOpsLayout),
      geluErf: mkPipe(mods.mmOps, 'gelu_erf_inplace', mmOpsLayout),
      gluSiluMul: mkPipe(mods.mmOps, 'glu_silu_mul', mmOpsLayout),
      addBias: mkPipe(mods.mmOps, 'add_bias_f32', mmOpsLayout),
      residualAdd: mkPipe(mods.mmOps, 'residual_add_f32', mmOpsLayout),
      linearF16: mkPipe(mods.linF16, 'main'),
      visConv2d: mkPipe(mods.visConv, 'main'),
      conv2dIn: mkPipe(mods.audConv, 'conv2d_in_s2'),
      conv2dDw: mkPipe(mods.audConv, 'conv2d_dw_s2'),
      conv2dPw: mkPipe(mods.audConv, 'conv2d_pw_1x1'),
      flexRope: mkPipe(mods.flexRope, 'main'),
      flexAttn: mkPipe(mods.flexAttn, 'main'),
    };
    return this.pipelines;
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

  async ensureVisionLoaded() {
    await this.ensurePipelines();
    if (!this.mmShards.has('vision')) {
      const ab = await this._loadShardBytes('vision');
      this._uploadShardToGpu('vision', ab);
    }
  }

  async ensureAudioEncLoaded() {
    await this.ensurePipelines();
    if (!this.mmShards.has('audio_enc')) {
      const ab = await this._loadShardBytes('audio_enc');
      this._uploadShardToGpu('audio_enc', ab);
    }
  }

  async ensureAudioDecLoaded() {
    await this.ensurePipelines();
    if (!this.mmShards.has('audio_dec')) {
      const ab = await this._loadShardBytes('audio_dec');
      this._uploadShardToGpu('audio_dec', ab);

      // Extract CPU-resident lookup tables & tiny codec linear projections
      const readF32 = (name) => {
        const m = this.mmMeta[name];
        return new Float32Array(ab.slice(m.offset, m.offset + m.bytes));
      };
      const cbMeta = this.mmMeta['adec.codebook_embd'];
      this.codebookEmbdBytes = new Uint8Array(ab.slice(cbMeta.offset, cbMeta.offset + cbMeta.bytes));
      this.codebookEmbdView = new DataView(this.codebookEmbdBytes.buffer);
      this.depthMaskEmb = readF32('adec.depth.mask_emb');

      const nQ = this.mmBp.num_codebooks || 12;
      this.depthCbF16 = [];
      for (let k = 0; k < nQ; k++) {
        const m = this.mmMeta[`adec.depth.cb.${k}.emb`];
        this.depthCbF16.push(new Uint16Array(ab.slice(m.offset, m.offset + m.bytes)));
      }

      this.codecCpu = {
        projOutW: readF32('adec.codec.project_out_w'),
        projOutB: readF32('adec.codec.project_out_b'),
        fromQW: readF32('adec.codec.from_q_w'),
        fromQB: readF32('adec.codec.from_q_b'),
        istftWindow: readF32('adec.codec.istft_window'),
      };
    }
  }

  _mmBinding(name) {
    const t = this.mmMeta[name];
    if (!t) throw new Error(`Multimodal tensor not found: ${name}`);
    const buffer = this.mmShards.get(t.shard);
    if (!buffer) throw new Error(`Multimodal shard '${t.shard}' not loaded`);
    return { buffer, offset: t.offset, size: alignUp(t.bytes, 4), meta: t };
  }

  _ensureScratch(maxTokens, maxDim = 4096) {
    if (this.scratch && this.scratchCapTokens >= maxTokens && this.scratchMaxDim >= maxDim) {
      return this.scratch;
    }
    if (this.scratch) {
      for (const b of Object.values(this.scratch)) b.destroy();
    }
    const T = Math.max(maxTokens, 512);
    const D = Math.max(maxDim, 4096);
    const mk = (bytes) => this.device.createBuffer({
      size: Math.max(16, alignUp(bytes, 16)),
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
    });
    this.scratch = {
      x: mk(T * D * 4),
      norm: mk(T * D * 4),
      qkv: mk(T * D * 3 * 4),
      qRot: mk(T * D * 4),
      kRot: mk(T * D * 4),
      attnGate: mk(T * 64 * 4),
      attnOut: mk(T * D * 4),
      branchOut: mk(T * D * 4),
      ffnGate: mk(T * 4096 * 4),
      ffnUp: mk(T * 4096 * 4),
      projHidden: mk(T * 4096 * 4),
      projOut: mk(T * 2048 * 4),
      convState: mk(D * 8 * 4),
      posHw: mk(T * 8),
      dummy: mk(256),
      dsA: mk(Math.max(Math.ceil(T / 2) * 64 * 256, T * 256) * 4),
      dsB: mk(Math.max(Math.ceil(T / 2) * 64 * 256, T * 256) * 4),
      logitsCb: mk(2048 * 4),
    };
    this.scratchCapTokens = T;
    this.scratchMaxDim = D;
    return this.scratch;
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

  _encMmLinear(enc, temp, name, xBuf, outBuf, T) {
    if (this.mmMeta[name]?.dtype === 'u32_f16x2') {
      return this._encMmF16(enc, temp, name, xBuf, outBuf, T);
    }
    return this._encMmQ8(enc, temp, name, xBuf, outBuf, T);
  }

  _encMmF16(enc, temp, name, xBuf, outBuf, T) {
    const w = this._mmBinding(name);
    const [M, K] = w.meta.orig_shape;
    const [, K_pad] = w.meta.padded_shape;
    const u = this._mkU(16, (u32) => { u32[0] = M; u32[1] = K; u32[2] = T; u32[3] = K_pad; });
    temp.push(u);
    const pipe = this.pipelines.linearF16;
    const bg = this.device.createBindGroup({
      layout: pipe.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: u, size: 16 } },
        { binding: 1, resource: { buffer: w.buffer, offset: w.offset, size: w.size } },
        { binding: 2, resource: { buffer: xBuf } },
        { binding: 3, resource: { buffer: outBuf } },
      ],
    });
    const pass = enc.beginComputePass();
    pass.setPipeline(pipe);
    pass.setBindGroup(0, bg);
    pass.dispatchWorkgroups(Math.ceil(M / 4), T, 1);
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

  _encMmF32(enc, temp, name, xBuf, outBuf, T) {
    const w = this._mmBinding(name);
    const [M, K] = w.meta.shape;
    const u = this._mkU(16, (u32) => { u32[0] = M; u32[1] = K; u32[2] = T; u32[3] = 0; });
    temp.push(u);
    const pipe = this.engine.pipelines.linearF32;
    const bg = this.device.createBindGroup({
      layout: pipe.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: u, size: 16 } },
        { binding: 1, resource: { buffer: w.buffer, offset: w.offset, size: w.size } },
        { binding: 2, resource: { buffer: xBuf } },
        { binding: 3, resource: { buffer: outBuf } },
      ],
    });
    const pass = enc.beginComputePass();
    pass.setPipeline(pipe);
    pass.setBindGroup(0, bg);
    pass.dispatchWorkgroups(Math.ceil(M / 4), T, 1);
    pass.end();
  }

  _encRmsNorm(enc, temp, wName, xBuf, outBuf, D, T, eps) {
    const w = this._mmBinding(wName);
    const u = this._mkU(16, (u32, f32) => { u32[0] = D; u32[1] = T; f32[2] = eps; f32[3] = 1.0; });
    temp.push(u);
    const pipe = this.engine.pipelines.rmsnorm;
    const bg = this.device.createBindGroup({
      layout: pipe.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: u, size: 16 } },
        { binding: 1, resource: { buffer: xBuf } },
        { binding: 2, resource: { buffer: w.buffer, offset: w.offset, size: w.size } },
        { binding: 3, resource: { buffer: outBuf } },
      ],
    });
    const pass = enc.beginComputePass();
    pass.setPipeline(pipe);
    pass.setBindGroup(0, bg);
    pass.dispatchWorkgroups(T, 1, 1);
    pass.end();
  }

  _encPostNormRes(enc, temp, wName, branchBuf, resBuf, D, T, eps, scalar = 1.0) {
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

  _encMmOp(enc, temp, pipeName, { a = 0, b = 0, c = 0, eps = 1e-5, f0 = 1.0, w = null, bias = null, x = null, out, wgX }) {
    const sc = this.scratch;
    const dummy = { buffer: sc.dummy, offset: 0, size: 64 };
    const wb = w || dummy;
    const bb = bias || dummy;
    const xb = x || sc.dummy;
    const u = this._mkU(32, (u32, f32) => {
      u32[0] = a; u32[1] = b; u32[2] = c; u32[3] = 0;
      f32[4] = eps; f32[5] = f0;
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

  _encAttentionBlock(enc, temp, prefix, {
    T, D, qHeads, kvHeads, hd, eps, ropeMode, theta, isCausal, slidingWindow, hasGate, usePosHw = false,
  }) {
    const sc = this.scratch;
    const qDim = qHeads * hd;
    const kvDim = kvHeads * hd;

    this._encMmLinear(enc, temp, `${prefix}.attn_qkv`, sc.norm, sc.qkv, T);

    // QK RMSNorm (on Q and K only)
    const qn = this._mmBinding(`${prefix}.q_norm`);
    const kn = this._mmBinding(`${prefix}.k_norm`);
    const uQk = this._mkU(32, (u32, f32) => {
      u32[0] = qHeads; u32[1] = kvHeads; u32[2] = hd; u32[3] = T;
      f32[4] = eps; u32[5] = 0; u32[6] = 0;
    });
    temp.push(uQk);
    const bgQk = this.device.createBindGroup({
      layout: this.engine.pipelines.qkNorm.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: uQk, size: 32 } },
        { binding: 1, resource: { buffer: sc.qkv } },
        { binding: 2, resource: { buffer: qn.buffer, offset: qn.offset, size: qn.size } },
        { binding: 3, resource: { buffer: kn.buffer, offset: kn.offset, size: kn.size } },
      ],
    });
    const pQk = enc.beginComputePass();
    pQk.setPipeline(this.engine.pipelines.qkNorm);
    pQk.setBindGroup(0, bgQk);
    pQk.dispatchWorkgroups(qHeads + kvHeads, T, 1);
    pQk.end();

    if (hasGate) {
      this._encMmF32(enc, temp, `${prefix}.attn_gate`, sc.norm, sc.attnGate, T);
    }

    const uR = this._mkU(32, (u32, f32) => {
      u32[0] = T; u32[1] = qHeads; u32[2] = kvHeads; u32[3] = hd;
      u32[4] = ropeMode; f32[5] = theta;
    });
    temp.push(uR);
    const bgR = this.device.createBindGroup({
      layout: this.pipelines.flexRope.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: uR, size: 32 } },
        { binding: 1, resource: { buffer: sc.qkv } },
        { binding: 2, resource: { buffer: usePosHw ? sc.posHw : sc.dummy } },
        { binding: 3, resource: { buffer: sc.qRot } },
        { binding: 4, resource: { buffer: sc.kRot } },
      ],
    });
    const pR = enc.beginComputePass();
    pR.setPipeline(this.pipelines.flexRope);
    pR.setBindGroup(0, bgR);
    pR.dispatchWorkgroups(Math.ceil(Math.max(qDim, kvDim) / 128), T, 1);
    pR.end();

    const uA = this._mkU(32, (u32, f32) => {
      u32[0] = T; u32[1] = qHeads; u32[2] = kvHeads; u32[3] = hd;
      u32[4] = isCausal ? 1 : 0; u32[5] = slidingWindow || 0; u32[6] = hasGate ? 1 : 0;
      f32[7] = 1.0 / Math.sqrt(hd);
    });
    temp.push(uA);
    const bgA = this.device.createBindGroup({
      layout: this.pipelines.flexAttn.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: uA, size: 32 } },
        { binding: 1, resource: { buffer: sc.qRot } },
        { binding: 2, resource: { buffer: sc.kRot } },
        { binding: 3, resource: { buffer: sc.qkv } },
        { binding: 4, resource: { buffer: hasGate ? sc.attnGate : sc.dummy } },
        { binding: 5, resource: { buffer: sc.attnOut } },
      ],
    });
    const pA = enc.beginComputePass();
    pA.setPipeline(this.pipelines.flexAttn);
    pA.setBindGroup(0, bgA);
    pA.dispatchWorkgroups(qHeads, T, 1);
    pA.end();

    this._encMmLinear(enc, temp, `${prefix}.attn_out`, sc.attnOut, sc.branchOut, T);
  }

  _encMlpBlock(enc, temp, prefix, { T, D, FF, eps, sandwichNormName = null }) {
    const sc = this.scratch;
    this._encRmsNorm(enc, temp, `${prefix}.ffn_norm`, sc.x, sc.norm, D, T, eps);
    this._encMmLinear(enc, temp, `${prefix}.ffn_w1`, sc.norm, sc.ffnGate, T);
    this._encMmLinear(enc, temp, `${prefix}.ffn_w3`, sc.norm, sc.ffnUp, T);
    this._encMmOp(enc, temp, 'gluSiluMul', {
      a: T * FF, c: 0, x: sc.ffnGate, out: sc.ffnUp, wgX: Math.ceil((T * FF) / 256),
    });
    this._encMmLinear(enc, temp, `${prefix}.ffn_w2`, sc.ffnUp, sc.branchOut, T);
    if (sandwichNormName) {
      this._encPostNormRes(enc, temp, sandwichNormName, sc.branchOut, sc.x, D, T, eps, 1.0);
    } else {
      this._encMmOp(enc, temp, 'residualAdd', {
        a: T * D, f0: 1.0, x: sc.branchOut, out: sc.x, wgX: Math.ceil((T * D) / 256),
      });
    }
  }

  async encodeImage(pixelValues, gridThw) {
    await this.ensureVisionLoaded();
    const visCfg = this.mmBp.vision_config;
    const [, gridH, gridW] = gridThw;
    const T = gridH * gridW;
    const D = visCfg.hidden_size; // 1024
    const FF = visCfg.intermediate_size; // 2560
    const qHeads = visCfg.num_attention_heads; // 16
    const kvHeads = visCfg.num_key_value_heads; // 8
    const hd = D / qHeads; // 64
    const eps = visCfg.norm_eps || 1e-5;
    const theta = visCfg.rope_parameters?.rope_theta || 1000000.0;
    const merge = visCfg.spatial_merge_size || 2;
    const sc = this._ensureScratch(T, 4096);

    const posHw = computeVisionPositionIds(gridH, gridW, merge);
    this.device.queue.writeBuffer(sc.ffnGate, 0, pixelValues); // reuse ffnGate (size >= T*4096*4 >= T*1536*4)
    this.device.queue.writeBuffer(sc.posHw, 0, posHw);

    const temp = [];
    const enc = this.device.createCommandEncoder();
    this._encMmQ8(enc, temp, 'vis.patchifier', sc.ffnGate, sc.x, T);
    this._encRmsNorm(enc, temp, 'vis.embedding_norm', sc.x, sc.norm, D, T, eps);
    enc.copyBufferToBuffer(sc.norm, 0, sc.x, 0, T * D * 4);

    for (let i = 0; i < visCfg.layer_types.length; i++) {
      const lt = visCfg.layer_types[i];
      const prefix = `vis.blk.${i}`;
      this._encRmsNorm(enc, temp, `${prefix}.operator_norm`, sc.x, sc.norm, D, T, eps);
      if (lt === 'conv') {
        this._encMmQ8(enc, temp, `${prefix}.conv.in_proj`, sc.norm, sc.qkv, T);
        const convW = this._mmBinding(`${prefix}.conv.spatial_conv`);
        const uC = this._mkU(16, (u32) => { u32[0] = D; u32[1] = gridH; u32[2] = gridW; u32[3] = merge; });
        temp.push(uC);
        const bgC = this.device.createBindGroup({
          layout: this.pipelines.visConv2d.getBindGroupLayout(0),
          entries: [
            { binding: 0, resource: { buffer: uC, size: 16 } },
            { binding: 1, resource: { buffer: sc.qkv } },
            { binding: 2, resource: { buffer: convW.buffer, offset: convW.offset, size: convW.size } },
            { binding: 3, resource: { buffer: sc.attnOut } },
          ],
        });
        const pC = enc.beginComputePass();
        pC.setPipeline(this.pipelines.visConv2d);
        pC.setBindGroup(0, bgC);
        pC.dispatchWorkgroups(Math.ceil(D / 256), T, 1);
        pC.end();
        this._encMmQ8(enc, temp, `${prefix}.conv.out_proj`, sc.attnOut, sc.branchOut, T);
      } else {
        this._encAttentionBlock(enc, temp, prefix, {
          T, D, qHeads, kvHeads, hd, eps, ropeMode: 2, theta, isCausal: false, slidingWindow: 0, hasGate: false, usePosHw: true,
        });
      }
      this._encMmOp(enc, temp, 'residualAdd', {
        a: T * D, f0: 1.0, x: sc.branchOut, out: sc.x, wgX: Math.ceil((T * D) / 256),
      });
      this._encMlpBlock(enc, temp, prefix, { T, D, FF, eps });
    }

    // MultiModalProjector: LayerNorm(x) [T, 1024] -> view as [T_tok, 4096] -> Linear1 -> GELU -> Linear2
    const projLnW = this._mmBinding('vis.proj.ln_w');
    const projLnB = this._mmBinding('vis.proj.ln_b');
    this._encMmOp(enc, temp, 'layernorm', {
      a: D, b: T, eps: this.mmBp.projector_layer_norm_eps || 1e-6,
      w: projLnW, bias: projLnB, x: sc.x, out: sc.norm, wgX: T,
    });
    const T_tok = T / (merge * merge);
    this._encMmQ8(enc, temp, 'vis.proj.l1', sc.norm, sc.projHidden, T_tok);
    this._encMmOp(enc, temp, 'addBias', {
      a: 4096, b: T_tok, bias: this._mmBinding('vis.proj.l1_b'), out: sc.projHidden, wgX: Math.ceil((T_tok * 4096) / 256),
    });
    this._encMmOp(enc, temp, 'geluErf', {
      a: T_tok * 4096, out: sc.projHidden, wgX: Math.ceil((T_tok * 4096) / 256),
    });
    this._encMmQ8(enc, temp, 'vis.proj.l2', sc.projHidden, sc.projOut, T_tok);
    this._encMmOp(enc, temp, 'addBias', {
      a: this.bp.hidden_size, b: T_tok, bias: this._mmBinding('vis.proj.l2_b'), out: sc.projOut,
      wgX: Math.ceil((T_tok * this.bp.hidden_size) / 256),
    });

    this.device.queue.submit([enc.finish()]);
    const feats = await this.engine._readBufferFloat32(sc.projOut, T_tok * this.bp.hidden_size);
    for (const u of temp) u.destroy();
    return feats;
  }

  async preprocessAndEncodeImage(image) {
    const prep = preprocessLfm3Image(image);
    const embeddings = await this.encodeImage(prep.pixelValues, prep.gridThw);
    return { embeddings, numTokens: prep.numVisionTokens, gridThw: prep.gridThw };
  }

  async encodeAudio(melFeatures, numFrames) {
    await this.ensureAudioEncLoaded();
    const encCfg = this.mmBp.audio_encoder_config;
    const tCfg = encCfg.text_config;
    const T0 = numFrames;
    const T1 = (T0 + 1) >> 1;
    const T2 = (T1 + 1) >> 1;
    const T3 = (T2 + 1) >> 1;
    const T = T3;
    const D = tCfg.hidden_size; // 768
    const FF = this.mmBp.audio_encoder_intermediate_size || tCfg.intermediate_size; // 1792
    const qHeads = tCfg.num_attention_heads; // 8
    const kvHeads = tCfg.num_key_value_heads; // 4
    const hd = D / qHeads; // 96
    const eps = tCfg.norm_eps || 1e-5;
    const theta = tCfg.rope_parameters?.rope_theta || 1000000.0;
    const K_conv = tCfg.conv_L_cache || 3;

    const sc = this._ensureScratch(Math.max(T0, 256), 4096);
    this.device.queue.writeBuffer(sc.dsA, 0, melFeatures);

    const temp = [];
    const enc = this.device.createCommandEncoder();

    const runC2d = (pipeName, wName, bName, inBuf, outBuf, pArr) => {
      const w = this._mmBinding(wName);
      const b = this._mmBinding(bName);
      const u = this._mkU(32, (u32) => { for (let j = 0; j < 8; j++) u32[j] = pArr[j]; });
      temp.push(u);
      const pipe = this.pipelines[pipeName];
      const bg = this.device.createBindGroup({
        layout: pipe.getBindGroupLayout(0),
        entries: [
          { binding: 0, resource: { buffer: u, size: 32 } },
          { binding: 1, resource: { buffer: inBuf } },
          { binding: 2, resource: { buffer: w.buffer, offset: w.offset, size: w.size } },
          { binding: 3, resource: { buffer: b.buffer, offset: b.offset, size: b.size } },
          { binding: 4, resource: { buffer: outBuf } },
        ],
      });
      const pass = enc.beginComputePass();
      pass.setPipeline(pipe);
      pass.setBindGroup(0, bg);
      pass.dispatchWorkgroups(Math.ceil((pArr[0] * pArr[3] * pArr[4]) / 256), 1, 1);
      pass.end();
    };

    runC2d('conv2dIn', 'aenc.ds.conv_in_w', 'aenc.ds.conv_in_b', sc.dsA, sc.dsB, [256, 128, T0, 64, T1, T1, 1, 0]);
    runC2d('conv2dDw', 'aenc.ds.conv_mid_w', 'aenc.ds.conv_mid_b', sc.dsB, sc.dsA, [256, 64, T1, 32, T2, T2, 0, 0]);
    runC2d('conv2dPw', 'aenc.ds.lin_mid_w', 'aenc.ds.lin_mid_b', sc.dsA, sc.dsB, [256, 256, T2, 32, T2, T2, 1, 0]);
    runC2d('conv2dDw', 'aenc.ds.conv_end_w', 'aenc.ds.conv_end_b', sc.dsB, sc.dsA, [256, 32, T2, 16, T3, T3, 0, 0]);
    runC2d('conv2dPw', 'aenc.ds.lin_end_w', 'aenc.ds.lin_end_b', sc.dsA, sc.dsB, [256, 256, T3, 16, T3, T3, 1, 1]);

    // lin_out: [T, 4096] -> [T, 768] + bias
    this._encMmQ8(enc, temp, 'aenc.ds.lin_out', sc.dsB, sc.x, T);
    this._encMmOp(enc, temp, 'addBias', {
      a: D, b: T, bias: this._mmBinding('aenc.ds.lin_out_b'), out: sc.x, wgX: Math.ceil((T * D) / 256),
    });

    for (let i = 0; i < tCfg.layer_types.length; i++) {
      const lt = tCfg.layer_types[i];
      const prefix = `aenc.blk.${i}`;
      this._encRmsNorm(enc, temp, `${prefix}.operator_norm`, sc.x, sc.norm, D, T, eps);
      if (lt === 'conv') {
        this._encMmQ8(enc, temp, `${prefix}.conv.in_proj`, sc.norm, sc.qkv, T);
        enc.clearBuffer(sc.convState, 0, D * (K_conv - 1) * 4);
        const convW = this._mmBinding(`${prefix}.conv.kernel`);
        const uConv = this._mkU(16, (u32) => { u32[0] = D; u32[1] = T; u32[2] = K_conv; u32[3] = K_conv - 1; });
        temp.push(uConv);
        const bgConv = this.device.createBindGroup({
          layout: this.engine.pipelines.shortConv.getBindGroupLayout(0),
          entries: [
            { binding: 0, resource: { buffer: uConv, size: 16 } },
            { binding: 1, resource: { buffer: sc.qkv } },
            { binding: 2, resource: { buffer: convW.buffer, offset: convW.offset, size: convW.size } },
            { binding: 3, resource: { buffer: sc.convState } },
            { binding: 4, resource: { buffer: sc.attnOut } },
          ],
        });
        const passC = enc.beginComputePass();
        passC.setPipeline(this.engine.pipelines.shortConv);
        passC.setBindGroup(0, bgConv);
        passC.dispatchWorkgroups(Math.ceil(D / 256), 1, 1);
        passC.end();
        this._encMmQ8(enc, temp, `${prefix}.conv.out_proj`, sc.attnOut, sc.branchOut, T);
      } else {
        this._encAttentionBlock(enc, temp, prefix, {
          T, D, qHeads, kvHeads, hd, eps, ropeMode: 1, theta,
          isCausal: true, slidingWindow: encCfg.sliding_window || 32, hasGate: false,
        });
      }
      this._encMmOp(enc, temp, 'residualAdd', {
        a: T * D, f0: 1.0, x: sc.branchOut, out: sc.x, wgX: Math.ceil((T * D) / 256),
      });
      this._encMlpBlock(enc, temp, prefix, { T, D, FF, eps });
    }

    this._encRmsNorm(enc, temp, 'aenc.embedding_norm', sc.x, sc.norm, D, T, eps);
    this._encMmOp(enc, temp, 'layernorm', {
      a: D, b: T, eps: 1e-5, w: this._mmBinding('aenc.proj.ln_w'), bias: this._mmBinding('aenc.proj.ln_b'),
      x: sc.norm, out: sc.x, wgX: T,
    });
    const textD = this.bp.hidden_size; // 2048
    this._encMmQ8(enc, temp, 'aenc.proj.l1', sc.x, sc.projHidden, T);
    this._encMmOp(enc, temp, 'addBias', {
      a: textD, b: T, bias: this._mmBinding('aenc.proj.l1_b'), out: sc.projHidden, wgX: Math.ceil((T * textD) / 256),
    });
    this._encMmOp(enc, temp, 'geluErf', { a: T * textD, out: sc.projHidden, wgX: Math.ceil((T * textD) / 256) });
    this._encMmQ8(enc, temp, 'aenc.proj.l2', sc.projHidden, sc.projOut, T);
    this._encMmOp(enc, temp, 'addBias', {
      a: textD, b: T, bias: this._mmBinding('aenc.proj.l2_b'), out: sc.projOut, wgX: Math.ceil((T * textD) / 256),
    });

    this.device.queue.submit([enc.finish()]);
    const feats = await this.engine._readBufferFloat32(sc.projOut, T * textD);
    for (const u of temp) u.destroy();
    return feats;
  }

  async preprocessAndEncodeAudio(pcm16k) {
    const mel = extractLfm3LogMel(pcm16k);
    const embeddings = await this.encodeAudio(mel.features, mel.numFrames);
    return { embeddings, numTokens: mel.numEncoderTokens };
  }

  async lookupCodebookEmbed(codes12) {
    await this.ensureAudioDecLoaded();
    const D = this.bp.hidden_size; // 2048
    const nQ = this.mmBp.num_codebooks || 12;
    const cbSize = this.mmBp.codebook_size || 2017;
    const meta = this.mmMeta['adec.codebook_embd'];
    const stride = meta.row_stride;
    const out = new Float32Array(D);
    for (let k = 0; k < nQ; k++) {
      const idx = k * cbSize + codes12[k];
      const rowOff = idx * stride;
      const scale = this.codebookEmbdView.getFloat32(rowOff, true);
      for (let d = 0; d < D; d++) {
        out[d] += this.codebookEmbdView.getInt8(rowOff + 4 + d) * scale;
      }
    }
    return out;
  }

  async _depthBaseProj(hidden2048) {
    const sc = this._ensureScratch(64, 4096);
    this.device.queue.writeBuffer(sc.projOut, 0, hidden2048);
    const temp = [];
    const enc = this.device.createCommandEncoder();
    const nQ = this.mmBp.num_codebooks || 12;
    const dpD = this.mmBp.depth_config.hidden_size; // 1024
    const M = nQ * dpD;
    this._encMmQ8(enc, temp, 'adec.depth.linear', sc.projOut, sc.ffnGate, 1);
    this._encMmOp(enc, temp, 'addBias', {
      a: M, b: 1, bias: this._mmBinding('adec.depth.linear_b'), out: sc.ffnGate, wgX: Math.ceil(M / 256),
    });
    this.device.queue.submit([enc.finish()]);
    const depthBase = await this.engine._readBufferFloat32(sc.ffnGate, M);
    for (const u of temp) u.destroy();
    return depthBase;
  }

  async runDepthDecoderPass(depthBase, codes, masked, pickIndices = null) {
    const dpCfg = this.mmBp.depth_config;
    const nQ = this.mmBp.num_codebooks || 12;
    const dpD = dpCfg.hidden_size; // 1024
    const FF = dpCfg.intermediate_size; // 2816
    const qHeads = dpCfg.num_attention_heads; // 32
    const kvHeads = dpCfg.num_key_value_heads; // 8
    const hd = dpCfg.head_dim || 32;
    const eps = dpCfg.norm_eps || 1e-5;
    const cbSize = this.mmBp.codebook_size || 2017;

    const depthIn = new Float32Array(depthBase);
    for (let k = 0; k < nQ; k++) {
      const off = k * dpD;
      if (masked[k]) {
        for (let d = 0; d < dpD; d++) depthIn[off + d] += this.depthMaskEmb[d];
      } else {
        const tbl = this.depthCbF16[k];
        const rowOff = codes[k] * dpD;
        for (let d = 0; d < dpD; d++) depthIn[off + d] += f16ToF32(tbl[rowOff + d]);
      }
    }

    const sc = this._ensureScratch(64, 4096);
    this.device.queue.writeBuffer(sc.x, 0, depthIn);

    const temp = [];
    const enc = this.device.createCommandEncoder();
    for (let i = 0; i < dpCfg.num_hidden_layers; i++) {
      const prefix = `adec.depth.blk.${i}`;
      this._encRmsNorm(enc, temp, `${prefix}.operator_norm`, sc.x, sc.norm, dpD, nQ, eps);
      this._encAttentionBlock(enc, temp, prefix, {
        T: nQ, D: dpD, qHeads, kvHeads, hd, eps, ropeMode: 1, theta: 1000000.0,
        isCausal: false, slidingWindow: 0, hasGate: true,
      });
      this._encMmOp(enc, temp, 'residualAdd', {
        a: nQ * dpD, f0: 1.0, x: sc.branchOut, out: sc.x, wgX: Math.ceil((nQ * dpD) / 256),
      });
      this._encMlpBlock(enc, temp, prefix, { T: nQ, D: dpD, FF, eps });
    }
    this._encRmsNorm(enc, temp, 'adec.depth.out_norm', sc.x, sc.norm, dpD, nQ, eps);
    this.device.queue.submit([enc.finish()]);
    const depthOut = await this.engine._readBufferFloat32(sc.norm, nQ * dpD);
    for (const u of temp) u.destroy();

    const targets = pickIndices || Array.from({ length: nQ }, (_, i) => i);
    const logitsBySlot = new Map();
    for (const k of targets) {
      this.device.queue.writeBuffer(sc.x, 0, depthOut.subarray(k * dpD, (k + 1) * dpD));
      const t2 = [];
      const e2 = this.device.createCommandEncoder();
      this._encRmsNorm(e2, t2, `adec.depth.cb.${k}.norm`, sc.x, sc.norm, dpD, 1, eps);
      const embW = this._mmBinding(`adec.depth.cb.${k}.emb`);
      const [M, K_pad] = embW.meta.padded_shape;
      const uL = this._mkU(16, (u32) => { u32[0] = cbSize; u32[1] = dpD; u32[2] = 1; u32[3] = K_pad; });
      t2.push(uL);
      const bgL = this.device.createBindGroup({
        layout: this.pipelines.linearF16.getBindGroupLayout(0),
        entries: [
          { binding: 0, resource: { buffer: uL, size: 16 } },
          { binding: 1, resource: { buffer: embW.buffer, offset: embW.offset, size: embW.size } },
          { binding: 2, resource: { buffer: sc.norm } },
          { binding: 3, resource: { buffer: sc.logitsCb } },
        ],
      });
      const pL = e2.beginComputePass();
      pL.setPipeline(this.pipelines.linearF16);
      pL.setBindGroup(0, bgL);
      pL.dispatchWorkgroups(Math.ceil(cbSize / 4), 1, 1);
      pL.end();
      this.device.queue.submit([e2.finish()]);
      logitsBySlot.set(k, await this.engine._readBufferFloat32(sc.logitsCb, cbSize));
      for (const u of t2) u.destroy();
    }
    return logitsBySlot;
  }

  async sampleAudioFrame(hidden2048, {
    temperature = 0.4,
    topK = 0,
    repetitionPenalty = 1.2,
    history = [],
    order = null,
    rng = Math.random,
  } = {}) {
    await this.ensureAudioDecLoaded();
    const nQ = this.mmBp.num_codebooks || 12;
    const schedule = this.mmBp.maskgit_schedule || [4, 4, 4];
    const codes = new Int32Array(nQ);
    const masked = new Array(nQ).fill(true);

    let perm = order ? Array.from(order) : Array.from({ length: nQ }, (_, i) => i);
    if (!order) {
      for (let i = perm.length - 1; i > 0; i--) {
        const j = Math.floor(rng() * (i + 1));
        [perm[i], perm[j]] = [perm[j], perm[i]];
      }
    }

    const depthBase = await this._depthBaseProj(hidden2048);
    let offset = 0;
    for (const count of schedule) {
      const pick = perm.slice(offset, offset + count);
      offset += count;
      const logitsMap = await this.runDepthDecoderPass(depthBase, codes, masked, pick);
      for (const idx of pick) {
        const cbHist = history.map((f) => f[idx]);
        codes[idx] = sampleToken(logitsMap.get(idx), {
          temperature: temperature || 0,
          topK: topK || 0,
          topP: 1.0,
          minP: 0.0,
          repetitionPenalty: repetitionPenalty || 1.0,
          recentTokens: cbHist,
          rng,
        });
        masked[idx] = false;
      }
    }
    return codes;
  }

  async decodeAudioCodes(frames) {
    await this.ensureAudioDecLoaded();
    const eosCode = this.mmBp.audio_eos_code ?? 2016;
    const validFrames = frames.filter((f) => !f.some((c) => c === eosCode || c < 0));
    if (validFrames.length === 0) return new Float32Array(0);

    const cfg = this.mmBp.codec_config;
    const levels = cfg.fsq_levels || [8, 7, 6, 6];
    const basis = [1];
    for (let i = 0; i < levels.length - 1; i++) basis.push(basis[i] * levels[i]);
    const halfW = levels.map((lv) => Math.floor(lv / 2));

    const nQ = cfg.n_q || 12;
    const cbDim = levels.length; // 4
    const qDim = nQ * cbDim; // 48
    const nFrames = validFrames.length;
    const { projOutW, projOutB, fromQW, fromQB, istftWindow } = this.codecCpu;

    const fsqProj = new Float32Array(nFrames * 256);
    const lat = new Float32Array(qDim);
    for (let f = 0; f < nFrames; f++) {
      const codes = validFrames[f];
      for (let q = 0; q < nQ; q++) {
        const c = codes[q];
        for (let j = 0; j < cbDim; j++) {
          lat[q * cbDim + j] = ((Math.floor(c / basis[j]) % levels[j]) - halfW[j]) / halfW[j];
        }
      }
      for (let m = 0; m < 256; m++) {
        let sum = projOutB[m];
        const row = m * qDim;
        for (let k = 0; k < qDim; k++) sum += projOutW[row + k] * lat[k];
        fsqProj[f * 256 + m] = sum;
      }
    }

    const frameStack = cfg.stft_frame_stack || 6;
    const T_dec = nFrames * frameStack;
    const D = cfg.encoder_dim || 512;
    const upsampled = new Float32Array(T_dec * D);
    const embed512 = new Float32Array(D);
    for (let f = 0; f < nFrames; f++) {
      for (let m = 0; m < D; m++) {
        let sum = fromQB[m];
        const row = m * 256;
        for (let k = 0; k < 256; k++) sum += fromQW[row + k] * fsqProj[f * 256 + k];
        embed512[m] = sum;
      }
      for (let r = 0; r < frameStack; r++) {
        upsampled.set(embed512, (f * frameStack + r) * D);
      }
    }

    const sc = this._ensureScratch(T_dec, 4096);
    this.device.queue.writeBuffer(sc.x, 0, upsampled);

    const layerTypes = this.mmBp.codec_layer_types || ['conv', 'conv', 'swa', 'conv', 'swa', 'conv', 'swa', 'conv'];
    const eps = 1e-5;
    const FF = 2304;
    const qHeads = 8;
    const kvHeads = 2;
    const hd = 64;
    const W = cfg.decoder_window || 30;
    const K_conv = 7;

    const temp = [];
    const enc = this.device.createCommandEncoder();
    for (let i = 0; i < layerTypes.length; i++) {
      const lt = layerTypes[i];
      const prefix = `adec.codec.blk.${i}`;
      this._encRmsNorm(enc, temp, `${prefix}.operator_norm`, sc.x, sc.norm, D, T_dec, eps);
      if (lt === 'conv') {
        this._encMmLinear(enc, temp, `${prefix}.conv.in_proj`, sc.norm, sc.qkv, T_dec);
        enc.clearBuffer(sc.convState, 0, D * (K_conv - 1) * 4);
        const convW = this._mmBinding(`${prefix}.conv.kernel`);
        const uConv = this._mkU(16, (u32) => { u32[0] = D; u32[1] = T_dec; u32[2] = K_conv; u32[3] = K_conv - 1; });
        temp.push(uConv);
        const bgConv = this.device.createBindGroup({
          layout: this.engine.pipelines.shortConv.getBindGroupLayout(0),
          entries: [
            { binding: 0, resource: { buffer: uConv, size: 16 } },
            { binding: 1, resource: { buffer: sc.qkv } },
            { binding: 2, resource: { buffer: convW.buffer, offset: convW.offset, size: convW.size } },
            { binding: 3, resource: { buffer: sc.convState } },
            { binding: 4, resource: { buffer: sc.attnOut } },
          ],
        });
        const passC = enc.beginComputePass();
        passC.setPipeline(this.engine.pipelines.shortConv);
        passC.setBindGroup(0, bgConv);
        passC.dispatchWorkgroups(Math.ceil(D / 256), 1, 1);
        passC.end();
        this._encMmLinear(enc, temp, `${prefix}.conv.out_proj`, sc.attnOut, sc.branchOut, T_dec);
      } else {
        this._encAttentionBlock(enc, temp, prefix, {
          T: T_dec, D, qHeads, kvHeads, hd, eps, ropeMode: 1, theta: 1000000.0,
          isCausal: true, slidingWindow: W, hasGate: true,
        });
      }
      this._encPostNormRes(enc, temp, `${prefix}.post_operator_norm`, sc.branchOut, sc.x, D, T_dec, eps, 1.0);
      this._encMlpBlock(enc, temp, prefix, {
        T: T_dec, D, FF, eps, sandwichNormName: `${prefix}.post_ffn_norm`,
      });
    }

    this._encRmsNorm(enc, temp, 'adec.codec.out_norm', sc.x, sc.norm, D, T_dec, eps);
    const nSpecBins = (cfg.stft_n_fft || 1280) + 2; // 1282
    this._encMmLinear(enc, temp, 'adec.codec.lin', sc.norm, sc.projOut, T_dec);
    this._encMmOp(enc, temp, 'addBias', {
      a: nSpecBins, b: T_dec, bias: this._mmBinding('adec.codec.lin_b'), out: sc.projOut,
      wgX: Math.ceil((T_dec * nSpecBins) / 256),
    });

    this.device.queue.submit([enc.finish()]);
    const linOut = await this.engine._readBufferFloat32(sc.projOut, T_dec * nSpecBins);
    for (const u of temp) u.destroy();

    return liquidCodecIstft(linOut, T_dec, istftWindow, {
      nFft: cfg.stft_n_fft || 1280,
      hopLength: cfg.stft_hop_length || 320,
      winLength: cfg.stft_win_length || 1280,
    });
  }

  /**
   * End-to-end multimodal prefill + text/audio/interleaved decode loop.
   */
  async generateOmni({
    inputIds,
    images = null,          // Array<{ pixelValues, gridThw }> or raw { data, width, height, channels }
    audioMels = null,       // Array<{ features, numFrames }> or Float32Array PCM 16k
    generationMode = 'text', // 'text' | 'audio' | 'interleaved'
    maxNewTokens = 128,
    temperature = 0.0,
    topK = 1,
    repetitionPenalty = 1.0,
    audioTemperature = 0.4,
    audioTopK = 0,
    audioRepetitionPenalty = 1.2,
    onTextToken = null,
    rng = Math.random,
  } = {}) {
    const mm = this.mmBp;
    const D = this.bp.hidden_size;

    const imageFeats = [];
    if (images && images.length > 0) {
      for (const img of images) {
        const res = img.pixelValues
          ? await this.encodeImage(img.pixelValues, img.gridThw)
          : (await this.preprocessAndEncodeImage(img)).embeddings;
        imageFeats.push(res);
      }
    }
    const audioFeats = [];
    if (audioMels && audioMels.length > 0) {
      for (const aud of audioMels) {
        const res = aud.features
          ? await this.encodeAudio(aud.features, aud.numFrames)
          : (await this.preprocessAndEncodeAudio(aud)).embeddings;
        audioFeats.push(res);
      }
    }

    // Build expanded inputIds replacing each <|image|> or <|audio|> placeholder with N tokens
    const imgTok = mm.image_token_id;
    const audTok = mm.audio_token_id;
    const vidTok = this.manifest.blueprint.video_token_id ?? 125029;
    const audOutTok = mm.audio_out_token_id;

    let imgCursor = 0;
    let audCursor = 0;
    const expandedIds = [];
    const imgFlat = imageFeats.length ? new Float32Array(imageFeats.reduce((a, b) => a + b.length, 0)) : null;
    if (imgFlat) {
      let off = 0;
      for (const f of imageFeats) { imgFlat.set(f, off); off += f.length; }
    }
    const audFlat = audioFeats.length ? new Float32Array(audioFeats.reduce((a, b) => a + b.length, 0)) : null;
    if (audFlat) {
      let off = 0;
      for (const f of audioFeats) { audFlat.set(f, off); off += f.length; }
    }

    // If caller didn't already repeat placeholder tokens, expand 1->N automatically
    const hasImgPlaceholder = imgFlat && inputIds.filter((id) => id === imgTok).length === imageFeats.length && imageFeats.every((f) => f.length / D > 1);
    const hasAudPlaceholder = audFlat && inputIds.filter((id) => id === audTok).length === audioFeats.length && audioFeats.every((f) => f.length / D > 1);

    for (const id of inputIds) {
      if (hasImgPlaceholder && id === imgTok) {
        const n = imageFeats[imgCursor++].length / D;
        for (let i = 0; i < n; i++) expandedIds.push(imgTok);
      } else if (hasAudPlaceholder && id === audTok) {
        const n = audioFeats[audCursor++].length / D;
        for (let i = 0; i < n; i++) expandedIds.push(audTok);
      } else {
        expandedIds.push(id);
      }
    }

    let mode = generationMode;
    if (mode === 'audio' && expandedIds[expandedIds.length - 1] !== mm.audio_start_token_id && expandedIds[expandedIds.length - 1] !== audOutTok) {
      expandedIds.push(mm.audio_start_token_id);
    } else if (mode === 'interleaved' && expandedIds[expandedIds.length - 1] !== mm.mixed_start_token_id) {
      expandedIds.push(mm.mixed_start_token_id);
    }

    const emb = await this.engine.lookupEmbeddings(expandedIds);
    let imgTokPos = 0;
    let audTokPos = 0;
    for (let t = 0; t < expandedIds.length; t++) {
      if (expandedIds[t] === imgTok && imgFlat) {
        emb.set(imgFlat.subarray(imgTokPos * D, (imgTokPos + 1) * D), t * D);
        imgTokPos++;
      } else if (expandedIds[t] === audTok && audFlat) {
        emb.set(audFlat.subarray(audTokPos * D, (audTokPos + 1) * D), t * D);
        audTokPos++;
      }
    }

    this.engine.reset();
    const eosSet = new Set(this.bp.eos_token_ids || [this.bp.eos_token_id]);
    const suppress = new Set([imgTok, audTok, audOutTok, vidTok]);
    if (mode === 'text') {
      suppress.add(mm.mixed_start_token_id);
      suppress.add(mm.audio_start_token_id);
    }

    let fwd = await this.engine.forwardEmbeds(emb, {
      tokenIds: expandedIds,
      returnLastHidden: mode !== 'text',
    });

    const textOutIds = [];
    const audioFrames = [];
    const textHistory = [];
    const audioHistory = [];
    const nText = mm.interleaved_n_text || 6;
    const nAudio = mm.interleaved_n_audio || 12;
    const eosCode = mm.audio_eos_code ?? 2016;
    let slotsLeft = nText;
    let speaking = mode === 'audio';
    let textDone = false;

    for (let step = 0; step < maxNewTokens; step++) {
      const isInterleaved = mode === 'interleaved';
      const doAudio = mode === 'audio' || (isInterleaved && speaking);

      let nextEmbed = null;
      let nextTokenId = null;

      if (doAudio) {
        const codes = await this.sampleAudioFrame(fwd.lastHidden, {
          temperature: audioTemperature,
          topK: audioTopK,
          repetitionPenalty: audioRepetitionPenalty,
          history: audioHistory.slice(-(mm.audio_repetition_window || 64)),
          rng,
        });
        const isAudEos = codes.some((c) => c === eosCode);
        if (isAudEos) {
          codes.fill(eosCode);
          mode = 'text';
          speaking = false;
        } else {
          audioFrames.push(codes);
          audioHistory.push(codes);
        }
        nextEmbed = await this.lookupCodebookEmbed(codes);
      } else {
        const logits = new Float32Array(fwd.logits);
        for (const sid of suppress) {
          if (sid != null && sid >= 0 && sid < logits.length) logits[sid] = -Infinity;
        }
        const tok = sampleToken(logits, {
          temperature,
          topK,
          repetitionPenalty,
          recentTokens: textHistory.slice(-32),
          rng,
        });
        textHistory.push(tok);
        textOutIds.push(tok);
        if (onTextToken) onTextToken(tok, step);
        nextTokenId = tok;

        if (tok === mm.text_end_token_id) {
          if (isInterleaved) {
            textDone = true;
            speaking = true;
          } else {
            break;
          }
        } else if (eosSet.has(tok)) {
          break;
        } else if (!isInterleaved && tok === mm.audio_start_token_id) {
          mode = 'audio';
          speaking = true;
        } else if (!isInterleaved && tok === mm.mixed_start_token_id) {
          mode = 'interleaved';
          slotsLeft = nText;
          speaking = false;
          textDone = false;
        }
      }

      if (mode === 'interleaved' && !textDone && isInterleaved) {
        slotsLeft -= 1;
        if (slotsLeft === 0) {
          speaking = !speaking;
          slotsLeft = speaking ? nAudio : nText;
        }
      }

      if (step === maxNewTokens - 1) break;

      if (nextEmbed) {
        fwd = await this.engine.forwardEmbeds(nextEmbed, {
          returnLastHidden: true,
          skipLogits: mode === 'audio' || (mode === 'interleaved' && speaking),
        });
      } else {
        fwd = await this.engine.forward([nextTokenId], {
          returnLastHidden: mode !== 'text',
        });
      }
    }

    const pcm24k = audioFrames.length > 0 ? await this.decodeAudioCodes(audioFrames) : new Float32Array(0);
    const wavBytes = pcm24k.length > 0 ? encodeWavPcm16(pcm24k, 24000) : null;
    return {
      textTokenIds: textOutIds,
      audioCodes: audioFrames,
      pcm24k,
      wavBytes,
    };
  }

  destroy() {
    for (const u of this.staticUniforms) u.destroy();
    for (const buf of this.mmShards.values()) buf.destroy();
    if (this.scratch) {
      for (const buf of Object.values(this.scratch)) buf.destroy();
    }
    this.mmShards.clear();
  }
}
