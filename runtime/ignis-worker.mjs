import { IgnisEngine } from './ignis-engine.mjs';
import { IgnisTokenizer } from './wasm-tokenizer.mjs';
import { sampleToken } from './sampler.mjs';

let engine = null;
let tokenizer = null;
let deviceLostReason = null;

self.onmessage = async (ev) => {
  const { id, cmd, payload } = ev.data;
  const reply = (ok, data, error) => self.postMessage({ id, ok, data, error });

  try {
    if (cmd === 'init') {
      if (engine) {
        engine.destroy();
        engine = null;
      }
      if (tokenizer) {
        tokenizer.free();
        tokenizer = null;
      }
      deviceLostReason = null;

      const t0 = performance.now();
      const {
        modelBaseUrl,
        modelKey = 'ignis-model',
        kernelBaseUrl = '/kernels',
        wasmUrl = '/runtime/wasm/ignis_tokenizer.wasm',
        tokenizerJsonUrl = null,
        maxSeqLen = 8192,
        maxBatchTokens = 128,
      } = payload;

      const { engine: eng, tokenizerBytes } = await IgnisEngine.createWithOpfs(
        modelBaseUrl,
        modelKey,
        kernelBaseUrl,
        {
          maxSeqLen,
          maxBatchTokens,
          onProgress: (p) => self.postMessage({ type: 'progress', ...p }),
        }
      );
      engine = eng;
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
      reply(true, { curPos: engine.curPos });
    } else if (cmd === 'generate') {
      if (!engine) throw new Error('Engine not initialized');
      const {
        promptIds = null,
        promptText = null,
        maxNewTokens = 32,
        chunkSize = engine.maxBatchTokens,
        sampling = { temperature: 0.0 },
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

      const tPrefill0 = performance.now();
      let lastLogits = null;
      const stepSize = Math.min(chunkSize, engine.maxBatchTokens);
      for (let off = 0; off < ids.length; off += stepSize) {
        const chunk = ids.slice(off, off + stepSize);
        const res = await engine.forward(chunk);
        lastLogits = res.logits;
        // Yield briefly between prefill chunks so GPU compositor / main page stay responsive
        if (off + stepSize < ids.length) {
          await new Promise((r) => setTimeout(r, 0));
        }
      }
      const prefillMs = performance.now() - tPrefill0;

      const generated = [];
      const history = [...ids];
      const eosId = engine.bp.eos_token_id;
      const tDec0 = performance.now();

      for (let step = 0; step < maxNewTokens; step++) {
        const nextTok = sampleToken(lastLogits, history, sampling);
        generated.push(nextTok);
        history.push(nextTok);

        const textChunk = tokenizer ? tokenizer.decode([nextTok], true) : '';
        self.postMessage({
          type: 'token',
          requestId: id,
          tokenId: nextTok,
          text: textChunk,
          step,
        });

        if (stopOnEos && nextTok === eosId) {
          break;
        }
        if (step + 1 < maxNewTokens) {
          const res = await engine.forward([nextTok]);
          lastLogits = res.logits;
        }
      }
      const decodeMs = performance.now() - tDec0;

      reply(true, {
        promptLen: ids.length,
        generatedIds: generated,
        generatedText: tokenizer ? tokenizer.decode(generated, true) : null,
        prefillMs,
        decodeMs,
        decodeTokPerSec: generated.length > 0 ? (generated.length / (decodeMs / 1000)) : 0,
        curPos: engine.curPos,
        deviceLostReason,
      });
    } else if (cmd === 'destroy') {
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
