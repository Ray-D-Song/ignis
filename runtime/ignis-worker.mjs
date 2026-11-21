import { IgnisEngine } from './ignis-engine.mjs';
import { IgnisTokenizer, resolveResponseSchema } from './wasm-tokenizer.mjs';
import { sampleToken } from './sampler.mjs';
import { Lfm3OmniRuntime } from './multimodal/lfm3-omni-runtime.mjs';
import { Gemma4OmniRuntime } from './multimodal/gemma4-omni-runtime.mjs';
import { NgramLookupSpeculator, DraftTargetSpeculator } from './speculative.mjs';

let engine = null;
let draftEngine = null;
let draftModelUrlCache = null;
let omni = null;
let tokenizer = null;
let modelBaseUrlCache = null;
let initOptionsCache = null;
let deviceLostReason = null;
let abortRequested = false;

function emitTokenEvent(requestId, tokInstance, tokId, step, generated, streamState, tDec0) {
  let fullText = streamState.committedText;
  let deltaText = '';
  if (tokInstance) {
    const decoded = tokInstance.decode(generated, true);
    const clean = decoded.endsWith('\uFFFD') ? decoded.replace(/\uFFFD+$/, '') : decoded;
    if (clean.length >= streamState.committedText.length) {
      deltaText = clean.slice(streamState.committedText.length);
      streamState.committedText = clean;
    }
    fullText = streamState.committedText;
  }
  const elapsedSec = Math.max(0.001, (performance.now() - tDec0) / 1000);
  self.postMessage({
    type: 'token',
    requestId,
    tokenId: tokId,
    text: deltaText,
    fullText,
    step,
    tokPerSec: generated.length / elapsedSec,
  });
}

self.onmessage = async (ev) => {
  const { id, cmd, payload } = ev.data;
  const reply = (ok, data, error) => self.postMessage({ id, ok, data, error });

  try {
    if (cmd === 'abort') {
      abortRequested = true;
      reply(true, { aborted: true });
      return;
    }
    if (cmd === 'init') {
      if (omni) {
        omni.destroy();
        omni = null;
      }
      if (draftEngine) {
        draftEngine.destroy();
        draftEngine = null;
        draftModelUrlCache = null;
      }
      if (engine) {
        engine.destroy();
        engine = null;
      }
      if (tokenizer) {
        tokenizer.free();
        tokenizer = null;
      }
      deviceLostReason = null;
      modelBaseUrlCache = null;
      initOptionsCache = null;

      const t0 = performance.now();
      const {
        modelBaseUrl,
        modelKey = 'ignis-model',
        kernelBaseUrl = '/kernels',
        wasmUrl = '/runtime/wasm/ignis_tokenizer.wasm',
        tokenizerJsonUrl = null,
        maxSeqLen = 8192,
        maxBatchTokens = 128,
        forceNoSubgroups = false,
      } = payload;

      const { engine: eng, tokenizerBytes } = await IgnisEngine.createWithOpfs(
        modelBaseUrl,
        modelKey,
        kernelBaseUrl,
        {
          maxSeqLen,
          maxBatchTokens,
          forceNoSubgroups: !!forceNoSubgroups,
          onProgress: (p) => self.postMessage({ type: 'progress', ...p }),
        }
      );
      engine = eng;
      modelBaseUrlCache = modelBaseUrl;
      initOptionsCache = {
        kernelBaseUrl,
        maxSeqLen,
        maxBatchTokens,
        forceNoSubgroups: !!forceNoSubgroups,
      };
      engine.device.lost.then((info) => {
        deviceLostReason = `${info.reason}: ${info.message}`;
        self.postMessage({ type: 'device_lost', reason: deviceLostReason });
      });

      let tokBytes = tokenizerBytes;
      if (!tokBytes && tokenizerJsonUrl) {
        const tr = await fetch(tokenizerJsonUrl);
        if (tr.ok) tokBytes = new Uint8Array(await tr.arrayBuffer());
      }
      if (tokBytes) {
        const wasmBytes = await (await fetch(wasmUrl)).arrayBuffer();
        tokenizer = await IgnisTokenizer.fromBuffers(wasmBytes, tokBytes);
      }

      reply(true, {
        initMs: performance.now() - t0,
        cacheHits: engine.cacheHits,
        cacheMisses: engine.cacheMisses,
        hasTokenizer: !!tokenizer,
        blueprint: engine.bp,
      });
    } else if (cmd === 'encode') {
      if (!tokenizer) throw new Error('Tokenizer not loaded');
      reply(true, { ids: tokenizer.encode(payload.text) });
    } else if (cmd === 'decode') {
      if (!tokenizer) throw new Error('Tokenizer not loaded');
      reply(true, { text: tokenizer.decode(payload.ids, !!payload.skipSpecial) });
    } else if (cmd === 'reset') {
      if (!engine) throw new Error('Engine not initialized');
      engine.reset();
      if (draftEngine) draftEngine.reset();
      reply(true, { curPos: engine.curPos });
    } else if (cmd === 'generate') {
      if (!engine) throw new Error('Engine not initialized');
      const {
        promptIds = null,
        promptText = null,
        images = [],
        audios = [],
        generationMode = 'text',
        maxNewTokens = 32,
        maxThinkingTokens = 0,
        endThinkTokenId = 4,
        chunkSize = engine.maxBatchTokens,
        sampling = { temperature: 0.0 },
        responseFormat = null,
        speculative = null,
        stopOnEos = true,
      } = payload;

      let ids = promptIds;
      if (!ids) {
        if (promptText == null || !tokenizer) {
          throw new Error('Must provide promptIds or (promptText with loaded tokenizer)');
        }
        ids = tokenizer.encode(promptText);
      }
      if (!ids.length) throw new Error('Empty prompt');

      abortRequested = false;
      const eosSet = new Set(
        Array.isArray(engine.bp.eos_token_ids) && engine.bp.eos_token_ids.length
          ? engine.bp.eos_token_ids
          : [engine.bp.eos_token_id]
      );

      const mmKind = engine.bp.multimodal?.kind;
      if (mmKind && (images.length > 0 || audios.length > 0 || (mmKind === 'lfm3_omni' && generationMode !== 'text'))) {
        if (!omni) {
          if (mmKind === 'lfm3_omni') {
            omni = new Lfm3OmniRuntime(engine, { modelBaseUrl: modelBaseUrlCache });
          } else if (mmKind === 'gemma4') {
            omni = new Gemma4OmniRuntime(engine, { modelBaseUrl: modelBaseUrlCache, exclusiveTower: true });
          } else {
            throw new Error(`Unsupported multimodal kind: ${mmKind}`);
          }
        }
        const generated = [];
        const streamState = { committedText: '' };
        const tDec0 = performance.now();
        const onTextToken = (tokId, step) => {
          if (stopOnEos && eosSet.has(tokId)) return;
          generated.push(tokId);
          emitTokenEvent(id, tokenizer, tokId, step, generated, streamState, tDec0);
        };
        const genRes = mmKind === 'lfm3_omni'
          ? await omni.generateOmni({
              inputIds: ids,
              images,
              audioMels: audios,
              generationMode,
              maxNewTokens,
              temperature: sampling.temperature ?? 0.0,
              topK: sampling.topK ?? 40,
              repetitionPenalty: sampling.repetitionPenalty ?? 1.0,
              onTextToken,
              shouldAbort: () => abortRequested,
            })
          : await omni.generateOmni(ids, {
              images,
              audios,
              maxNewTokens,
              temperature: sampling.temperature ?? 0.0,
              topK: sampling.topK ?? 40,
              repetitionPenalty: sampling.repetitionPenalty ?? 1.0,
              onTextToken,
              shouldAbort: () => abortRequested,
              releaseVisionAfterEncode: true,
              releaseAudioAfterEncode: true,
            });
        const decodeMs = performance.now() - tDec0;
        const finalIds = genRes.textTokenIds || generated;
        reply(true, {
          promptLen: ids.length,
          generatedIds: finalIds,
          generatedText: tokenizer ? tokenizer.decode(finalIds, true) : null,
          wavBytes: genRes.wavBytes || null,
          pcm24k: genRes.pcm24k?.length ? genRes.pcm24k : null,
          aborted: abortRequested,
          prefillMs: 0,
          decodeMs,
          decodeTokPerSec: finalIds.length > 0 ? (finalIds.length / (decodeMs / 1000)) : 0,
          curPos: engine.curPos,
          deviceLostReason,
        });
        return;
      }

      const fsmSchema = resolveResponseSchema(responseFormat);
      if (fsmSchema && !tokenizer) {
        throw new Error('Tokenizer required for responseFormat JSON Schema FSM');
      }

      if (speculative && !fsmSchema && speculative.mode === 'draft' && speculative.draftModelUrl) {
        if (!draftEngine || draftModelUrlCache !== speculative.draftModelUrl) {
          if (draftEngine) {
            draftEngine.destroy();
            draftEngine = null;
          }
          const dRes = await IgnisEngine.createWithOpfs(
            speculative.draftModelUrl,
            speculative.draftModelKey || 'ignis-draft',
            initOptionsCache?.kernelBaseUrl || '/kernels',
            {
              maxSeqLen: initOptionsCache?.maxSeqLen || 4096,
              maxBatchTokens: initOptionsCache?.maxBatchTokens || 64,
              forceNoSubgroups: !!initOptionsCache?.forceNoSubgroups,
            }
          );
          draftEngine = dRes.engine;
          draftModelUrlCache = speculative.draftModelUrl;
        }
        const generated = [];
        const streamState = { committedText: '' };
        const tDec0 = performance.now();
        const spec = new DraftTargetSpeculator(engine, draftEngine, { k: speculative.k || 4 });
        const sRes = await spec.generate(ids, {
          maxNewTokens,
          sampling,
          eosSet: stopOnEos ? eosSet : new Set(),
          onToken: (tokId, step) => {
            generated.push(tokId);
            emitTokenEvent(id, tokenizer, tokId, step, generated, streamState, tDec0);
          },
          shouldAbort: () => abortRequested,
        });
        reply(true, {
          promptLen: ids.length,
          generatedIds: sRes.generatedIds,
          generatedText: tokenizer ? tokenizer.decode(sRes.generatedIds, true) : null,
          aborted: abortRequested,
          prefillMs: sRes.prefillMs,
          decodeMs: sRes.decodeMs,
          decodeTokPerSec: sRes.decodeTokPerSec,
          speculativeAccepted: sRes.speculativeAccepted,
          speculativeProposed: sRes.speculativeProposed,
          curPos: engine.curPos,
          deviceLostReason,
        });
        return;
      }

      const tPrefill0 = performance.now();
      let lastLogits = null;
      const stepSize = Math.min(chunkSize, engine.maxBatchTokens);
      for (let off = 0; off < ids.length; off += stepSize) {
        if (abortRequested) break;
        const chunk = ids.slice(off, off + stepSize);
        const res = await engine.forward(chunk);
        lastLogits = res.logits;
        self.postMessage({
          type: 'prefill_progress',
          requestId: id,
          processed: Math.min(ids.length, off + chunk.length),
          total: ids.length,
        });
        // Yield briefly between prefill chunks so GPU compositor / main page stay responsive
        if (off + stepSize < ids.length) {
          await new Promise((r) => setTimeout(r, 0));
        }
      }
      const prefillMs = performance.now() - tPrefill0;

      if (speculative && !fsmSchema && speculative.mode === 'ngram' && !abortRequested && lastLogits) {
        const generated = [];
        const streamState = { committedText: '' };
        const tDec0 = performance.now();
        const spec = new NgramLookupSpeculator({
          k: speculative.k || 4,
          maxNgram: speculative.ngramN || 4,
        });
        const sRes = await spec.generate(engine, ids, {
          maxNewTokens,
          sampling,
          eosSet: stopOnEos ? eosSet : new Set(),
          onToken: (tokId, step) => {
            generated.push(tokId);
            emitTokenEvent(id, tokenizer, tokId, step, generated, streamState, tDec0);
          },
          shouldAbort: () => abortRequested,
          skipPrefill: true,
          initialLogits: lastLogits,
        });
        reply(true, {
          promptLen: ids.length,
          generatedIds: sRes.generatedIds,
          generatedText: tokenizer ? tokenizer.decode(sRes.generatedIds, true) : null,
          aborted: abortRequested,
          prefillMs,
          decodeMs: sRes.decodeMs,
          decodeTokPerSec: sRes.decodeTokPerSec,
          speculativeAccepted: sRes.speculativeAccepted,
          speculativeProposed: sRes.speculativeProposed,
          curPos: engine.curPos,
          deviceLostReason,
        });
        return;
      }

      const generated = [];
      const history = [...ids];
      const tDec0 = performance.now();
      const streamState = { committedText: '' };

      const fsm = fsmSchema ? tokenizer.createFsm(fsmSchema, [...eosSet]) : null;
      const fsmMaskMsSamples = [];

      try {
        if (!abortRequested && lastLogits) {
          let thinkClosed = !(maxThinkingTokens > 0);
          for (let step = 0; step < maxNewTokens; step++) {
            if (abortRequested) break;
            let nextTok;
            if (fsm) {
              const tMask0 = performance.now();
              fsm.maskLogits(lastLogits);
              fsmMaskMsSamples.push(performance.now() - tMask0);
              nextTok = sampleToken(lastLogits, history, sampling);
            } else if (!thinkClosed && step >= maxThinkingTokens) {
              // Budget forcing: close <think> channel with </think> so remaining tokens go to final answer
              nextTok = endThinkTokenId;
              thinkClosed = true;
            } else {
              nextTok = sampleToken(lastLogits, history, sampling);
              if (nextTok === endThinkTokenId) {
                thinkClosed = true;
              }
            }
            if (stopOnEos && eosSet.has(nextTok)) {
              break;
            }
            if (fsm && !fsm.advance(nextTok)) {
              break;
            }
            generated.push(nextTok);
            history.push(nextTok);
            emitTokenEvent(id, tokenizer, nextTok, step, generated, streamState, tDec0);

            if (step + 1 < maxNewTokens && !abortRequested) {
              const res = await engine.forward([nextTok]);
              lastLogits = res.logits;
            }
          }
        }
      } finally {
        if (fsm) fsm.free();
      }
      const decodeMs = performance.now() - tDec0;

      reply(true, {
        promptLen: ids.length,
        generatedIds: generated,
        generatedText: tokenizer ? tokenizer.decode(generated, true) : null,
        aborted: abortRequested,
        prefillMs,
        decodeMs,
        decodeTokPerSec: generated.length > 0 ? (generated.length / (decodeMs / 1000)) : 0,
        fsmMaskMsSamples: fsmMaskMsSamples.length > 0 ? fsmMaskMsSamples : undefined,
        curPos: engine.curPos,
        deviceLostReason,
      });
    } else if (cmd === 'destroy') {
      if (omni) {
        omni.destroy();
        omni = null;
      }
      if (draftEngine) {
        draftEngine.destroy();
        draftEngine = null;
        draftModelUrlCache = null;
      }
      if (engine) {
        engine.destroy();
        engine = null;
      }
      if (tokenizer) {
        tokenizer.free();
        tokenizer = null;
      }
      reply(true, {});
    } else {
      throw new Error(`Unknown command: ${cmd}`);
    }
  } catch (err) {
    reply(false, null, err instanceof Error ? err.stack || err.message : String(err));
  }
};
