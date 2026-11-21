/**
 * @ignis/core OpenAI-compatible multimodal streaming client (`IgnisOpenAI`).
 * Zero-backend WebGPU inference supporting Spark2.5 / Qwen3, LFM3-Omni-2B, and Gemma-4-E4B-it.
 */

import { IgnisClient } from '../../runtime/ignis-client.mjs';
import { IgnisEngine } from '../../runtime/ignis-engine.mjs';
import { IgnisTokenizer, resolveResponseSchema } from '../../runtime/wasm-tokenizer.mjs';
import { NgramLookupSpeculator } from '../../runtime/speculative.mjs';
import { Lfm3OmniRuntime } from '../../runtime/multimodal/lfm3-omni-runtime.mjs';
import { Gemma4OmniRuntime } from '../../runtime/multimodal/gemma4-omni-runtime.mjs';
import { decodeWav, resampleLinear } from '../../runtime/multimodal/audio-dsp.mjs';

function makeCompletionId() {
  return `ignis-${Date.now().toString(36)}-${Math.random().toString(36).slice(2, 8)}`;
}

async function decodeImageSource(imgSpec) {
  const src = typeof imgSpec === 'string' ? { url: imgSpec } : imgSpec;
  if (src?.data) {
    return {
      data: src.data instanceof Uint8Array || src.data instanceof Uint8ClampedArray
        ? src.data
        : new Uint8Array(src.data),
      width: src.width,
      height: src.height,
      channels: src.channels || 4,
    };
  }
  if (!src?.url) {
    throw new Error('image_url part requires either url or raw { data, width, height }');
  }
  const resp = await fetch(src.url);
  if (!resp.ok) throw new Error(`Failed to fetch image_url: ${resp.status}`);
  const blob = await resp.blob();
  if (typeof createImageBitmap !== 'function' || typeof OffscreenCanvas === 'undefined') {
    throw new Error('Browser createImageBitmap & OffscreenCanvas required to decode image_url blobs');
  }
  const bmp = await createImageBitmap(blob);
  try {
    const canvas = new OffscreenCanvas(bmp.width, bmp.height);
    const ctx = canvas.getContext('2d');
    ctx.drawImage(bmp, 0, 0);
    const imgData = ctx.getImageData(0, 0, bmp.width, bmp.height);
    return { data: new Uint8Array(imgData.data.buffer), width: bmp.width, height: bmp.height, channels: 4 };
  } finally {
    bmp.close();
  }
}

async function decodeAudioSource(audSpec, targetSr = 16000) {
  if (!audSpec) throw new Error('input_audio requires data');
  const d = audSpec.data ?? audSpec;
  let samples = null;
  let sr = audSpec.sampleRate || targetSr;

  if (typeof d === 'string') {
    if (d.startsWith('data:') || d.startsWith('http://') || d.startsWith('https://') || d.startsWith('/')) {
      const ab = await (await fetch(d)).arrayBuffer();
      const dec = decodeWav(ab);
      samples = dec.samples;
      sr = dec.sampleRate;
    } else {
      const bin = atob(d);
      const u8 = new Uint8Array(bin.length);
      for (let i = 0; i < bin.length; i++) u8[i] = bin.charCodeAt(i);
      if (audSpec.format === 'pcm_f32') {
        samples = new Float32Array(u8.buffer);
      } else {
        const dec = decodeWav(u8.buffer);
        samples = dec.samples;
        sr = dec.sampleRate;
      }
    }
  } else if (d instanceof ArrayBuffer) {
    if (audSpec.format === 'pcm_f32') {
      samples = new Float32Array(d);
    } else {
      const dec = decodeWav(d);
      samples = dec.samples;
      sr = dec.sampleRate;
    }
  } else if (d instanceof Uint8Array) {
    if (audSpec.format === 'pcm_f32') {
      samples = new Float32Array(d.buffer, d.byteOffset, Math.floor(d.byteLength / 4));
    } else {
      const dec = decodeWav(d.buffer.slice(d.byteOffset, d.byteOffset + d.byteLength));
      samples = dec.samples;
      sr = dec.sampleRate;
    }
  } else if (d instanceof Float32Array) {
    samples = d;
  } else if (Array.isArray(d)) {
    samples = Float32Array.from(d);
  } else {
    throw new Error('Unsupported input_audio payload');
  }

  return sr === targetSr ? samples : resampleLinear(samples, sr, targetSr);
}

/**
 * Formats OpenAI `messages` + multimodal attachments into the exact chat template for Spark, LFM3-Omni, or Gemma-4.
 */
export async function buildIgnisChatPrompt(messages, blueprint, {
  enableThinking = false,
  thinkingEffort = 'low',
} = {}) {
  const topType = blueprint?.top_model_type || blueprint?.architecture || 'spark2_5';
  const isGemma4 = topType === 'gemma4' || blueprint?.architecture === 'gemma4_text';
  const isLfm3 = topType === 'lfm3_omni' || topType === 'lfm3_vl' || blueprint?.architecture === 'lfm3';

  const images = [];
  const audios = [];

  const renderParts = async (content, imgTag, audTag) => {
    if (typeof content === 'string') return content;
    if (!Array.isArray(content)) return String(content ?? '');
    let s = '';
    for (const part of content) {
      if (part.type === 'text') {
        s += part.text || '';
      } else if (part.type === 'image_url') {
        images.push(await decodeImageSource(part.image_url));
        s += imgTag;
      } else if (part.type === 'input_audio') {
        audios.push(await decodeAudioSource(part.input_audio, 16000));
        s += audTag;
      }
    }
    return s;
  };

  if (isGemma4) {
    let prompt = '<bos>';
    let systemHeader = '';
    for (const m of messages) {
      const role = m.role === 'assistant' ? 'model' : m.role;
      const body = await renderParts(m.content, '<|image|>', '<|audio|>');
      if (role === 'system') {
        systemHeader += body + '\n';
        continue;
      }
      const txt = systemHeader ? `${systemHeader}${body}` : body;
      systemHeader = '';
      prompt += `<|turn>${role}\n${txt}<turn|>\n`;
    }
    prompt += enableThinking ? '<|turn>model\n<|thought|>\n' : '<|turn>model\n';
    return { prompt, images, audios };
  }

  if (isLfm3) {
    let prompt = '<|startoftext|>';
    for (const m of messages) {
      const body = await renderParts(
        m.content,
        '<|vision_start|><|image|><|vision_end|>',
        '<|vision_start|><|audio|><|vision_end|>'
      );
      prompt += `<|im_start|>${m.role}\n${body}<|im_end|>\n`;
    }
    prompt += enableThinking ? '<|im_start|>assistant\n<think>' : '<|im_start|>assistant\n<think></think>';
    return { prompt, images, audios };
  }

  // Spark2.5 / Qwen2 / Qwen3
  const hasExplicitSys = messages.some((m) => m.role === 'system');
  let prompt = '';
  if (!hasExplicitSys) {
    const reasoning = enableThinking
      ? `\n\`thinking_effort\` guides on how much to think in your thinking channel (not including the response channel), supported values include \`low\` and \`high\`.\nNow the system is invoked with \`thinking_effort=${thinkingEffort}\`.`
      : '';
    prompt += `<｜start▁of▁sentence｜><|System|>\nyou are a helpful assistant.${reasoning}<｜end▁of▁sentence｜>`;
  }
  for (const m of messages) {
    const body = await renderParts(m.content, '', '');
    if (m.role === 'system') {
      prompt += `<｜start▁of▁sentence｜><|System|>\n${body}<｜end▁of▁sentence｜>`;
    } else if (m.role === 'user') {
      prompt += `<｜start▁of▁sentence｜><|User|>${body}<｜end▁of▁sentence｜>`;
    } else {
      prompt += `<｜start▁of▁sentence｜><|Bot|></think>${body}<｜end▁of▁sentence｜>`;
    }
  }
  prompt += `<｜start▁of▁sentence｜><|Bot|>${enableThinking ? '<think>' : '</think>'}`;
  return { prompt, images, audios };
}

class AsyncQueue {
  constructor() {
    this.items = [];
    this.waiters = [];
    this.closed = false;
    this.error = null;
  }
  push(v) {
    if (this.closed) return;
    if (this.waiters.length > 0) {
      this.waiters.shift().resolve({ value: v, done: false });
    } else {
      this.items.push(v);
    }
  }
  close() {
    this.closed = true;
    while (this.waiters.length > 0) {
      this.waiters.shift().resolve({ value: undefined, done: true });
    }
  }
  fail(err) {
    this.error = err;
    this.closed = true;
    while (this.waiters.length > 0) {
      this.waiters.shift().reject(err);
    }
  }
  async next() {
    if (this.items.length > 0) return { value: this.items.shift(), done: false };
    if (this.error) throw this.error;
    if (this.closed) return { value: undefined, done: true };
    return new Promise((resolve, reject) => this.waiters.push({ resolve, reject }));
  }
  [Symbol.asyncIterator]() { return this; }
}

export class IgnisOpenAI {
  constructor(options = {}) {
    this.options = options;
    this.workerUrl = options.workerUrl || null;
    this.models = options.models || {};
    this.onProgress = options.onProgress || null;
    this.currentModelKey = null;
    this.blueprint = null;

    // Worker mode
    this._client = null;
    // Direct (non-worker) mode
    this._engine = null;
    this._tokenizer = null;
    this._omni = null;
    this._aborted = false;

    this.chat = {
      completions: {
        create: (params) => this._createChatCompletion(params),
      },
    };
  }

  _resolveModelConfig(modelOrCfg) {
    if (typeof modelOrCfg === 'object' && modelOrCfg !== null) {
      return {
        modelKey: modelOrCfg.modelKey || modelOrCfg.modelUrl,
        ...this.options,
        ...modelOrCfg,
      };
    }
    if (typeof modelOrCfg === 'string' && this.models[modelOrCfg]) {
      return {
        modelKey: this.models[modelOrCfg].modelKey || `ignis-${modelOrCfg}`,
        ...this.options,
        ...this.models[modelOrCfg],
      };
    }
    return {
      modelUrl: this.options.modelUrl || modelOrCfg,
      modelKey: this.options.modelKey || (typeof modelOrCfg === 'string' ? `ignis-${modelOrCfg}` : 'ignis-default'),
      kernelBaseUrl: this.options.kernelBaseUrl || '/kernels',
      wasmUrl: this.options.wasmUrl || '/runtime/wasm/ignis_tokenizer.wasm',
      maxSeqLen: this.options.maxSeqLen || 4096,
      maxBatchTokens: this.options.maxBatchTokens || 64,
      forceNoSubgroups: !!this.options.forceNoSubgroups,
      opfs: this.options.opfs !== false,
    };
  }

  async loadModel(modelOrCfg) {
    const cfg = this._resolveModelConfig(modelOrCfg);
    if (!cfg.modelUrl) throw new Error('IgnisOpenAI: modelUrl is required');

    if (this.workerUrl) {
      await this.destroy();
      this._client = new IgnisClient(this.workerUrl);
      if (this.onProgress) this._client.onProgress = this.onProgress;
      const res = await this._client.init({
        modelBaseUrl: cfg.modelUrl,
        modelKey: cfg.modelKey,
        kernelBaseUrl: cfg.kernelBaseUrl || '/kernels',
        wasmUrl: cfg.wasmUrl || '/runtime/wasm/ignis_tokenizer.wasm',
        maxSeqLen: cfg.maxSeqLen || 4096,
        maxBatchTokens: cfg.maxBatchTokens || 64,
        forceNoSubgroups: !!cfg.forceNoSubgroups,
      });
      this.blueprint = res.blueprint;
      this.currentModelKey = typeof modelOrCfg === 'string' ? modelOrCfg : cfg.modelKey;
      return res;
    }

    // Direct in-context engine (for unit tests or Dedicated Workers)
    await this.destroy();
    const t0 = performance.now();
    let engine;
    let tokBytes = null;
    if (cfg.opfs && typeof navigator !== 'undefined' && navigator.storage?.getDirectory) {
      const opfsRes = await IgnisEngine.createWithOpfs(
        cfg.modelUrl,
        cfg.modelKey,
        cfg.kernelBaseUrl || '/kernels',
        {
          maxSeqLen: cfg.maxSeqLen || 4096,
          maxBatchTokens: cfg.maxBatchTokens || 64,
          forceNoSubgroups: !!cfg.forceNoSubgroups,
          onProgress: this.onProgress,
        }
      );
      engine = opfsRes.engine;
      tokBytes = opfsRes.tokenizerBytes;
    } else {
      engine = await IgnisEngine.create(cfg.modelUrl, cfg.kernelBaseUrl || '/kernels', {
        maxSeqLen: cfg.maxSeqLen || 4096,
        maxBatchTokens: cfg.maxBatchTokens || 64,
        forceNoSubgroups: !!cfg.forceNoSubgroups,
      });
    }
    this._engine = engine;
    this.blueprint = engine.bp;

    const wasmBytes = await (await fetch(cfg.wasmUrl || '/runtime/wasm/ignis_tokenizer.wasm')).arrayBuffer();
    if (!tokBytes) {
      const tr = await fetch(`${cfg.modelUrl}/tokenizer.json`);
      if (tr.ok) tokBytes = new Uint8Array(await tr.arrayBuffer());
    }
    if (tokBytes) {
      this._tokenizer = await IgnisTokenizer.fromBuffers(wasmBytes, tokBytes);
    }

    const mmKind = engine.bp.multimodal?.kind;
    if (mmKind === 'lfm3_omni') {
      this._omni = new Lfm3OmniRuntime(engine, { modelBaseUrl: cfg.modelUrl });
    } else if (mmKind === 'gemma4') {
      this._omni = new Gemma4OmniRuntime(engine, { modelBaseUrl: cfg.modelUrl });
    }
    this.currentModelKey = typeof modelOrCfg === 'string' ? modelOrCfg : cfg.modelKey;
    return {
      initMs: performance.now() - t0,
      cacheHits: engine.cacheHits || 0,
      cacheMisses: engine.cacheMisses || 0,
      hasTokenizer: !!this._tokenizer,
      blueprint: engine.bp,
    };
  }

  _createChatCompletion(params) {
    if (params?.stream) {
      return this._streamChatCompletion(params);
    }
    return this._runNonStreaming(params);
  }

  async _ensureLoadedForCall(model) {
    if (!this.blueprint || (model && this.currentModelKey !== model && (this.models[model] || model.startsWith('/') || model.startsWith('http')))) {
      await this.loadModel(model || this.options.modelUrl);
    }
    if (!this.blueprint) throw new Error('IgnisOpenAI: call loadModel() or provide modelUrl first');
  }

  async _runNonStreaming(params) {
    let fullContent = '';
    let finishReason = 'stop';
    let finalUsage = { prompt_tokens: 0, completion_tokens: 0, total_tokens: 0 };
    let ignisMeta = null;
    let audioOut = undefined;
    const id = makeCompletionId();
    const created = Math.floor(Date.now() / 1000);
    const modelName = params.model || this.currentModelKey || this.blueprint?.architecture || 'ignis';

    for await (const chunk of this._streamChatCompletion({ ...params, _id: id, _created: created, _captureStats: (s) => { finalUsage = s.usage; ignisMeta = s.ignis; audioOut = s.audio; } })) {
      const c0 = chunk.choices?.[0];
      if (c0?.delta?.content) fullContent += c0.delta.content;
      if (c0?.finish_reason) finishReason = c0.finish_reason;
    }

    return {
      id,
      object: 'chat.completion',
      created,
      model: modelName,
      choices: [{
        index: 0,
        message: {
          role: 'assistant',
          content: fullContent,
          ...(audioOut ? { audio: audioOut } : {}),
        },
        finish_reason: finishReason,
      }],
      usage: finalUsage,
      ...(ignisMeta ? { ignis: ignisMeta } : {}),
    };
  }

  _streamChatCompletion(params) {
    const queue = new AsyncQueue();
    const id = params._id || makeCompletionId();
    const created = params._created || Math.floor(Date.now() / 1000);

    (async () => {
      try {
        await this._ensureLoadedForCall(params.model);
        const modelName = params.model || this.currentModelKey || this.blueprint?.architecture || 'ignis';
        const fsmSchema = resolveResponseSchema(params.response_format);
        const { prompt, images, audios } = await buildIgnisChatPrompt(params.messages || [], this.blueprint, {
          enableThinking: fsmSchema ? false : !!params.enable_thinking,
          thinkingEffort: params.thinking_effort || 'low',
        });

        queue.push({
          id,
          object: 'chat.completion.chunk',
          created,
          model: modelName,
          choices: [{ index: 0, delta: { role: 'assistant' }, finish_reason: null }],
        });

        const maxTokens = params.max_tokens ?? 256;
        const sampling = {
          temperature: params.temperature ?? 0.0,
          topP: params.top_p ?? 1.0,
          topK: params.top_k ?? 40,
          repetitionPenalty: params.repetition_penalty ?? 1.0,
        };

        if (this._client) {
          this._client.onToken = (ev) => {
            if (ev.text) {
              queue.push({
                id,
                object: 'chat.completion.chunk',
                created,
                model: modelName,
                choices: [{ index: 0, delta: { content: ev.text }, finish_reason: null }],
              });
            }
          };
          await this._client.reset();
          const enc = await this._client.encode(prompt);
          const res = await this._client.generate({
            promptIds: enc.ids,
            images,
            audios,
            maxNewTokens: maxTokens,
            maxThinkingTokens: params.thinking_budget || 0,
            generationMode: params.generation_mode || 'text',
            sampling,
            responseFormat: params.response_format || null,
            speculative: params.speculative || null,
            stopOnEos: true,
          });
          const finishReason = res.aborted ? 'abort' : (res.generatedIds?.length >= maxTokens ? 'length' : 'stop');
          const audioOut = res.wavBytes || res.pcm24k
            ? { wavBytes: res.wavBytes, pcm24k: res.pcm24k, sampleRate: 24000 }
            : undefined;
          if (params._captureStats) {
            params._captureStats({
              usage: {
                prompt_tokens: res.promptLen || enc.ids.length,
                completion_tokens: res.generatedIds?.length || 0,
                total_tokens: (res.promptLen || enc.ids.length) + (res.generatedIds?.length || 0),
              },
              ignis: {
                prefillMs: res.prefillMs || 0,
                decodeMs: res.decodeMs || 0,
                decodeTokPerSec: res.decodeTokPerSec || 0,
                speculativeAccepted: res.speculativeAccepted,
                speculativeProposed: res.speculativeProposed,
                ...(res.fsmMaskMsSamples ? { fsmMaskMsSamples: res.fsmMaskMsSamples } : {}),
              },
              audio: audioOut,
            });
          }
          queue.push({
            id,
            object: 'chat.completion.chunk',
            created,
            model: modelName,
            choices: [{ index: 0, delta: audioOut ? { audio: audioOut } : {}, finish_reason: finishReason }],
          });
          queue.close();
          return;
        }

        // Direct mode (runs in worker or main thread directly)
        this._aborted = false;
        const promptIds = this._tokenizer.encode(prompt);
        const generatedIds = [];
        let committedText = '';
        const t0 = performance.now();

        const emitToken = (tokId) => {
          generatedIds.push(tokId);
          const decoded = this._tokenizer.decode(generatedIds, true);
          const clean = decoded.endsWith('\uFFFD') ? decoded.replace(/\uFFFD+$/, '') : decoded;
          if (clean.length > committedText.length) {
            const delta = clean.slice(committedText.length);
            committedText = clean;
            queue.push({
              id,
              object: 'chat.completion.chunk',
              created,
              model: modelName,
              choices: [{ index: 0, delta: { content: delta }, finish_reason: null }],
            });
          }
        };

        let audioOut = undefined;
        let speculativeStats = null;
        const mmKind = this.blueprint.multimodal?.kind;
        if (this._omni && (images.length > 0 || audios.length > 0 || (mmKind === 'lfm3_omni' && (params.generation_mode || 'text') !== 'text'))) {
          const genRes = await this._omni.generateOmni(
            mmKind === 'lfm3_omni'
              ? {
                  inputIds: promptIds,
                  images,
                  audioMels: audios,
                  generationMode: params.generation_mode || 'text',
                  maxNewTokens: maxTokens,
                  temperature: sampling.temperature,
                  topK: sampling.topK,
                  repetitionPenalty: sampling.repetitionPenalty,
                  onTextToken: (tokId) => emitToken(tokId),
                  shouldAbort: () => this._aborted,
                }
              : promptIds,
            mmKind === 'gemma4'
              ? {
                  images,
                  audios,
                  maxNewTokens: maxTokens,
                  temperature: sampling.temperature,
                  topK: sampling.topK,
                  repetitionPenalty: sampling.repetitionPenalty,
                  onTextToken: (tokId) => emitToken(tokId),
                  shouldAbort: () => this._aborted,
                  releaseVisionAfterEncode: true,
                  releaseAudioAfterEncode: true,
                }
              : undefined
          );
          if (genRes.wavBytes || genRes.pcm24k?.length) {
            audioOut = { wavBytes: genRes.wavBytes, pcm24k: genRes.pcm24k, sampleRate: 24000 };
          }
        } else if (params.speculative?.mode === 'ngram' && !fsmSchema) {
          const ngramSpec = new NgramLookupSpeculator({
            k: params.speculative.k || 4,
            maxNgram: params.speculative.ngramN || 4,
          });
          const sRes = await ngramSpec.generate(this._engine, promptIds, {
            maxNewTokens: maxTokens,
            sampling,
            onToken: (tokId) => emitToken(tokId),
            shouldAbort: () => this._aborted,
          });
          speculativeStats = {
            speculativeAccepted: sRes.speculativeAccepted,
            speculativeProposed: sRes.speculativeProposed,
          };
        } else {
          this._engine.reset();
          const { sampleToken } = await import('../../runtime/sampler.mjs');
          const fwd = await this._engine.forward(promptIds);
          let logits = fwd.logits;
          const eosSet = new Set(
            Array.isArray(this.blueprint.eos_token_ids) && this.blueprint.eos_token_ids.length
              ? this.blueprint.eos_token_ids
              : [this.blueprint.eos_token_id]
          );
          const fsm = fsmSchema ? this._tokenizer.createFsm(fsmSchema, [...eosSet]) : null;
          const hist = [...promptIds];
          try {
            for (let step = 0; step < maxTokens; step++) {
              if (this._aborted) break;
              if (fsm) fsm.maskLogits(logits);
              const tok = sampleToken(logits, hist, sampling);
              if (eosSet.has(tok)) break;
              if (fsm && !fsm.advance(tok)) break;
              hist.push(tok);
              emitToken(tok);
              if (step + 1 < maxTokens && !this._aborted) {
                logits = (await this._engine.forward([tok])).logits;
              }
            }
          } finally {
            if (fsm) fsm.free();
          }
        }

        const totalMs = performance.now() - t0;
        const finishReason = this._aborted ? 'abort' : (generatedIds.length >= maxTokens ? 'length' : 'stop');
        if (params._captureStats) {
          params._captureStats({
            usage: {
              prompt_tokens: promptIds.length,
              completion_tokens: generatedIds.length,
              total_tokens: promptIds.length + generatedIds.length,
            },
            ignis: {
              prefillMs: 0,
              decodeMs: totalMs,
              decodeTokPerSec: generatedIds.length / Math.max(0.001, totalMs / 1000),
              ...(speculativeStats || {}),
            },
            audio: audioOut,
          });
        }
        queue.push({
          id,
          object: 'chat.completion.chunk',
          created,
          model: modelName,
          choices: [{ index: 0, delta: audioOut ? { audio: audioOut } : {}, finish_reason: finishReason }],
        });
        queue.close();
      } catch (err) {
        queue.fail(err);
      }
    })();

    return queue;
  }

  async abort() {
    this._aborted = true;
    if (this._client) await this._client.abort();
  }

  async reset() {
    if (this._client) await this._client.reset();
    if (this._engine) this._engine.reset();
  }

  async destroy() {
    if (this._client) {
      await this._client.destroy();
      this._client = null;
    }
    if (this._omni) {
      this._omni.destroy();
      this._omni = null;
    }
    if (this._engine) {
      this._engine.destroy();
      this._engine = null;
    }
    if (this._tokenizer) {
      this._tokenizer.free();
      this._tokenizer = null;
    }
    this.blueprint = null;
    this.currentModelKey = null;
  }
}
