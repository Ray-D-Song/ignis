import http from 'node:http';
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import { chromium } from 'playwright-core';
import { resolveBrowserExecutable } from './browser-env.mjs';

const ROOT_DIR = path.resolve(import.meta.dirname, '..');

const WORKER_JS = `
import { IgnisEngine } from '/runtime/ignis-engine.mjs';
import { IgnisTokenizer } from '/runtime/wasm-tokenizer.mjs';
import { sampleToken } from '/runtime/sampler.mjs';
import { NgramLookupSpeculator, DraftTargetSpeculator } from '/runtime/speculative.mjs';

function makeSeededRng(seed = 123456789) {
  let s = seed >>> 0;
  return () => {
    s = (Math.imul(1664525, s) + 1013904223) >>> 0;
    return s / 4294967296;
  };
}

async function runAutoregressiveGreedy(engine, promptIds, maxNewTokens) {
  const eosSet = new Set(Array.isArray(engine.bp.eos_token_ids) && engine.bp.eos_token_ids.length ? engine.bp.eos_token_ids : [engine.bp.eos_token_id]);
  engine.reset();
  let fwd = await engine.forward(promptIds);
  const generatedIds = [];
  const t0 = performance.now();
  for (let step = 0; step < maxNewTokens; step++) {
    const tok = sampleToken(fwd.logits, [], { temperature: 0.0 });
    if (eosSet.has(tok)) break;
    generatedIds.push(tok);
    if (step + 1 < maxNewTokens) {
      fwd = await engine.forward([tok]);
    }
  }
  const decodeMs = performance.now() - t0;
  return {
    generatedIds,
    decodeMs,
    decodeTokPerSec: generatedIds.length > 0 ? (generatedIds.length / (decodeMs / 1000)) : 0,
  };
}

self.onmessage = async (ev) => {
  if (ev.data?.type !== 'run') return;
  const log = (msg) => self.postMessage({ type: 'log', msg });
  try {
    log('worker started; fetching ignis_tokenizer.wasm');
    const wasmBytes = new Uint8Array(await (await fetch('/runtime/wasm/ignis_tokenizer.wasm')).arrayBuffer());
    const out = {};

    // Test 1: Zero-VRAM N-gram Lookup on Spark-X2.6-4B (code/JSON repetition copy-edit)
    {
      const modelUrl = '/tests/fixtures/spark-4b-ignis-q8head';
      log('loading spark-4b-ignis-q8head');
      const { engine, tokenizerBytes } = await IgnisEngine.createWithOpfs(
        modelUrl, 'spec-spark-4b', '/kernels', { maxSeqLen: 1024, maxBatchTokens: 64 }
      );
      engine.device.lost.then((info) => log('spark device.lost: ' + info.reason + ' ' + info.message));
      log('spark loaded; initializing tokenizer');
      const tokBytes = tokenizerBytes || new Uint8Array(await (await fetch(modelUrl + '/tokenizer.json')).arrayBuffer());
      const tok = await IgnisTokenizer.fromBuffers(wasmBytes, tokBytes);
      try {
        const prompt = '<｜start▁of▁sentence｜><|System|>\\nyou are a helpful assistant.<｜end▁of▁sentence｜><｜start▁of▁sentence｜><|User|>Copy this JSON verbatim replacing only "localhost" with "prod.example.com":\\n\`\`\`json\\n{\\n  "serviceName": "ignis-webgpu-gateway",\\n  "region": "us-west-2",\\n  "host": "localhost",\\n  "port": 8443,\\n  "protocol": "https",\\n  "maxRetries": 5,\\n  "timeoutMs": 30000,\\n  "enableCompression": true\\n}\\n\`\`\`<｜end▁of▁sentence｜><｜start▁of▁sentence｜><|Bot|></think>';
        const promptIds = tok.encode(prompt);
        log('running spark baseline (' + promptIds.length + ' prompt tokens)');
        const baseline = await runAutoregressiveGreedy(engine, promptIds, 64);
        log('spark baseline done (' + baseline.generatedIds.length + ' tok, ' + baseline.decodeMs.toFixed(0) + 'ms); running N-gram');
        const spec = new NgramLookupSpeculator({ k: 5, minNgram: 2, maxNgram: 4 });
        const specRes = await spec.generate(engine, promptIds, { maxNewTokens: 64, sampling: { temperature: 0.0 } });
        log('spark N-gram done (' + specRes.generatedIds.length + ' tok, ' + specRes.decodeMs.toFixed(0) + 'ms, acc=' + specRes.speculativeAccepted + '/' + specRes.speculativeProposed + ')');

        out.ngram = {
          promptTokens: promptIds.length,
          baselineLen: baseline.generatedIds.length,
          specLen: specRes.generatedIds.length,
          exactMatch: JSON.stringify(baseline.generatedIds) === JSON.stringify(specRes.generatedIds),
          baselineDecodeMs: baseline.decodeMs,
          specDecodeMs: specRes.decodeMs,
          baselineTps: baseline.decodeTokPerSec,
          specTps: specRes.decodeTokPerSec,
          speedup: baseline.decodeMs / Math.max(0.1, specRes.decodeMs),
          proposed: specRes.speculativeProposed,
          accepted: specRes.speculativeAccepted,
          acceptRate: specRes.speculativeProposed > 0 ? specRes.speculativeAccepted / specRes.speculativeProposed : 0,
          rounds: specRes.rounds,
          text: tok.decode(specRes.generatedIds, true),
        };
      } finally {
        tok.free();
        engine.destroy();
      }
      log('spark engine destroyed');
    }

    // Test 2 & 3: Draft-Target (LFM3 INT3_G64 -> INT4_G64+INT8_G64) Greedy Bit-Exact + Stochastic
    {
      const tgtUrl = '/tests/fixtures/lfm3-omni-2b-ignis';
      const drfUrl = '/tests/fixtures/lfm3-omni-2b-ignis-q3';
      log('loading lfm3 target');
      const { engine: targetEngine, tokenizerBytes } = await IgnisEngine.createWithOpfs(
        tgtUrl, 'spec-lfm3-tgt', '/kernels', { maxSeqLen: 1024, maxBatchTokens: 64 }
      );
      targetEngine.device.lost.then((info) => log('lfm3 target device.lost: ' + info.reason + ' ' + info.message));
      log('lfm3 target loaded; loading lfm3 Q3 draft');
      const { engine: draftEngine } = await IgnisEngine.createWithOpfs(
        drfUrl, 'spec-lfm3-drf', '/kernels', { maxSeqLen: 1024, maxBatchTokens: 64 }
      );
      draftEngine.device.lost.then((info) => log('lfm3 draft device.lost: ' + info.reason + ' ' + info.message));
      log('lfm3 draft loaded; initializing tokenizer');
      const tokBytes = tokenizerBytes || new Uint8Array(await (await fetch(tgtUrl + '/tokenizer.json')).arrayBuffer());
      const tok = await IgnisTokenizer.fromBuffers(wasmBytes, tokBytes);
      try {
        const prompt = '<|startoftext|><|im_start|>user\\nList the stages of the water cycle clearly and concisely.<|im_end|>\\n<|im_start|>assistant\\n<think></think>';
        const promptIds = tok.encode(prompt);
        log('running lfm3 greedy baseline (' + promptIds.length + ' prompt tokens)');
        const baseline = await runAutoregressiveGreedy(targetEngine, promptIds, 48);
        log('lfm3 baseline done (' + baseline.generatedIds.length + ' tok, ' + baseline.decodeMs.toFixed(0) + 'ms); running Draft-Target');

        const dtSpec = new DraftTargetSpeculator(targetEngine, draftEngine, { k: 4 });
        const specRes = await dtSpec.generate(promptIds, { maxNewTokens: 48, sampling: { temperature: 0.0 } });
        log('lfm3 Draft-Target greedy done (' + specRes.generatedIds.length + ' tok, acc=' + specRes.speculativeAccepted + '/' + specRes.speculativeProposed + ')');

        out.draftTarget = {
          promptTokens: promptIds.length,
          baselineLen: baseline.generatedIds.length,
          specLen: specRes.generatedIds.length,
          exactMatch: JSON.stringify(baseline.generatedIds) === JSON.stringify(specRes.generatedIds),
          baselineDecodeMs: baseline.decodeMs,
          specDecodeMs: specRes.decodeMs,
          baselineTps: baseline.decodeTokPerSec,
          specTps: specRes.decodeTokPerSec,
          speedup: baseline.decodeMs / Math.max(0.1, specRes.decodeMs),
          proposed: specRes.speculativeProposed,
          accepted: specRes.speculativeAccepted,
          acceptRate: specRes.speculativeProposed > 0 ? specRes.speculativeAccepted / specRes.speculativeProposed : 0,
          avgTokensPerRound: specRes.rounds > 0 ? specRes.generatedIds.length / specRes.rounds : 0,
          rounds: specRes.rounds,
          text: tok.decode(specRes.generatedIds, true),
        };

        log('running lfm3 Draft-Target stochastic (T=0.7)');
        const stochRes = await dtSpec.generate(promptIds, {
          maxNewTokens: 24,
          sampling: { temperature: 0.7, topK: 40, topP: 0.95 },
          rng: makeSeededRng(20261121),
        });
        log('lfm3 stochastic done (' + stochRes.generatedIds.length + ' tok)');
        out.stochastic = {
          len: stochRes.generatedIds.length,
          allFiniteIds: stochRes.generatedIds.every((x) => Number.isInteger(x) && x >= 0),
          proposed: stochRes.speculativeProposed,
          accepted: stochRes.speculativeAccepted,
          rounds: stochRes.rounds,
          text: tok.decode(stochRes.generatedIds, true),
        };
      } finally {
        tok.free();
        draftEngine.destroy();
        targetEngine.destroy();
      }
    }

    self.postMessage({ ok: true, result: out });
  } catch (err) {
    self.postMessage({ ok: false, error: err?.stack || String(err) });
  }
};
`;

const HARNESS_HTML = `<!doctype html><html><body><script type="module">
  window.__RUN_SPEC_SUITE__ = () => new Promise((resolve, reject) => {
    const w = new Worker('/__phase6_spec_worker__.mjs', { type: 'module' });
    const timer = setTimeout(() => {
      w.terminate();
      reject(new Error('Speculative worker timed out after 180s'));
    }, 180000);
    w.onmessage = (e) => {
      if (e.data?.type === 'log') {
        console.log('[spec-worker]', e.data.msg);
        return;
      }
      clearTimeout(timer);
      w.terminate();
      if (e.data.ok) resolve(e.data.result);
      else reject(new Error(e.data.error));
    };
    w.onerror = (e) => {
      clearTimeout(timer);
      w.terminate();
      reject(new Error(e.message || 'Worker error'));
    };
    w.postMessage({ type: 'run' });
  });
</script></body></html>`;

function startStaticServer() {
  const mime = {
    '.html': 'text/html',
    '.mjs': 'application/javascript',
    '.js': 'application/javascript',
    '.json': 'application/json',
    '.wasm': 'application/wasm',
    '.wgsl': 'text/plain',
    '.ignis': 'application/octet-stream',
  };
  return new Promise((resolve) => {
    const server = http.createServer((req, res) => {
      const cleanUrl = (req.url || '/').split('?')[0];
      if (cleanUrl === '/favicon.ico') { res.writeHead(204); res.end(); return; }
      const hdr = (ct) => ({
        'Content-Type': ct,
        'Cross-Origin-Opener-Policy': 'same-origin',
        'Cross-Origin-Embedder-Policy': 'require-corp',
      });
      if (cleanUrl === '/__phase6_spec__.html') {
        res.writeHead(200, hdr('text/html'));
        res.end(HARNESS_HTML);
        return;
      }
      if (cleanUrl === '/__phase6_spec_worker__.mjs') {
        res.writeHead(200, hdr('application/javascript'));
        res.end(WORKER_JS);
        return;
      }
      const rel = cleanUrl === '/' ? 'index.html' : cleanUrl.replace(/^\/+/, '');
      const fullPath = path.join(ROOT_DIR, rel);
      if (!fullPath.startsWith(ROOT_DIR) || !fs.existsSync(fullPath) || fs.statSync(fullPath).isDirectory()) {
        res.writeHead(404);
        res.end('Not found');
        return;
      }
      const ext = path.extname(fullPath);
      res.writeHead(200, hdr(mime[ext] || 'application/octet-stream'));
      fs.createReadStream(fullPath).pipe(res);
    });
    server.listen(0, '127.0.0.1', () => resolve({ server, port: server.address().port }));
  });
}

async function main() {
  const { server, port } = await startStaticServer();
  const exePath = resolveBrowserExecutable();
  const userDataDir = fs.mkdtempSync(path.join(os.tmpdir(), 'ignis-spec-'));
  console.log(`[test-phase6-speculative] Launching ${exePath ? path.basename(exePath) : 'playwright-chromium'} against http://127.0.0.1:${port}`);

  const errors = [];
  let context;
  try {
    context = await chromium.launchPersistentContext(userDataDir, {
      ...(exePath ? { executablePath: exePath } : {}),
      headless: true,
      args: ['--headless=new'],
    });
    const page = context.pages()[0] || await context.newPage();
    page.on('console', (m) => {
      if (m.type() === 'error') {
        console.error('[browser error]', m.text());
        errors.push(m.text());
      } else if (m.type() === 'warning' || m.text().startsWith('[spec-worker]')) {
        console.log(m.text());
      }
    });
    page.on('pageerror', (e) => {
      console.error('[pageerror]', String(e));
      errors.push(String(e));
    });

    await page.goto(`http://127.0.0.1:${port}/__phase6_spec__.html`);
    const res = await page.evaluate(() => window.__RUN_SPEC_SUITE__());
    console.log('\n[test-phase6-speculative RESULTS]', JSON.stringify(res, null, 2));

    const checks = [
      ['N-gram greedy 100% bit-exact equivalence', res.ngram.exactMatch && res.ngram.specLen > 10, `${res.ngram.specLen} tok`],
      ['N-gram acceptance >= 50% & > 0 accepted', res.ngram.accepted > 0 && res.ngram.acceptRate >= 0.50, `${(res.ngram.acceptRate * 100).toFixed(1)}% (${res.ngram.accepted}/${res.ngram.proposed})`],
      ['N-gram wall-clock speedup > 1.3x', res.ngram.speedup > 1.3, `${res.ngram.speedup.toFixed(2)}x (${res.ngram.baselineTps.toFixed(1)} -> ${res.ngram.specTps.toFixed(1)} tok/s)`],
      ['Draft-Target ShortConv greedy 100% bit-exact equivalence', res.draftTarget.exactMatch && res.draftTarget.specLen >= 20, `${res.draftTarget.specLen} tok`],
      ['Draft-Target acceptance >= 50% & avgTok/round >= 2.0', res.draftTarget.acceptRate >= 0.50 && res.draftTarget.avgTokensPerRound >= 2.0, `accept=${(res.draftTarget.acceptRate * 100).toFixed(1)}%, avg=${res.draftTarget.avgTokensPerRound.toFixed(2)} tok/round (${res.draftTarget.baselineTps.toFixed(1)} -> ${res.draftTarget.specTps.toFixed(1)} tok/s)`],
      ['Stochastic rejection sampling (T=0.7) valid & proposed > 0', res.stochastic.len > 0 && res.stochastic.allFiniteIds && res.stochastic.proposed > 0, `len=${res.stochastic.len}, acc=${res.stochastic.accepted}/${res.stochastic.proposed}`],
      ['Zero browser console/page errors', errors.length === 0, errors],
    ];

    let failed = 0;
    for (const [label, ok, val] of checks) {
      console.log(`  [${ok ? 'PASS' : 'FAIL'}] ${label} (${JSON.stringify(val)})`);
      if (!ok) failed++;
    }
    if (failed > 0) throw new Error(`${failed} Stage 6C speculative check(s) failed`);
  } finally {
    if (context) await context.close();
    server.close();
    fs.rmSync(userDataDir, { recursive: true, force: true });
  }
}

main().catch((err) => {
  console.error(err);
  process.exit(1);
});
