import http from 'node:http';
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import { chromium } from 'playwright-core';
import { resolveBrowserExecutable } from './browser-env.mjs';

const ROOT_DIR = path.resolve(import.meta.dirname, '..');
const Q3_DIR = path.join(ROOT_DIR, 'tests', 'fixtures', 'lfm3-omni-2b-ignis-q3');

if (!fs.existsSync(path.join(Q3_DIR, 'manifest.json'))) {
  throw new Error(`Missing packed INT3_G64 model at ${Q3_DIR}`);
}

function startStaticServer(harnessHtml) {
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
      if (cleanUrl === '/__phase6_deg__.html') {
        res.writeHead(200, {
          'Content-Type': 'text/html',
          'Cross-Origin-Opener-Policy': 'same-origin',
          'Cross-Origin-Embedder-Policy': 'require-corp',
        });
        res.end(harnessHtml);
        return;
      }
      if (cleanUrl === '/__phase6_deg_worker__.mjs') {
        res.writeHead(200, {
          'Content-Type': 'application/javascript',
          'Cross-Origin-Opener-Policy': 'same-origin',
          'Cross-Origin-Embedder-Policy': 'require-corp',
        });
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
      res.writeHead(200, {
        'Content-Type': mime[ext] || 'application/octet-stream',
        'Cross-Origin-Opener-Policy': 'same-origin',
        'Cross-Origin-Embedder-Policy': 'require-corp',
      });
      fs.createReadStream(fullPath).pipe(res);
    });
    server.listen(0, '127.0.0.1', () => resolve({ server, port: server.address().port }));
  });
}

const WORKER_JS = `
import { IgnisEngine } from '/runtime/ignis-engine.mjs';
import { IgnisTokenizer } from '/runtime/wasm-tokenizer.mjs';
import { detectDeviceProfile } from '/runtime/device-profile.mjs';
import { sampleToken } from '/runtime/sampler.mjs';

const cosSim = (a, b) => {
  let dot = 0, na = 0, nb = 0;
  const n = Math.min(a.length, b.length);
  for (let i = 0; i < n; i++) {
    dot += a[i] * b[i];
    na += a[i] * a[i];
    nb += b[i] * b[i];
  }
  return dot / (Math.sqrt(na) * Math.sqrt(nb) + 1e-12);
};

async function runGreedyText(modelUrl, modelKey, forceNoSubgroups, promptText, maxSteps = 16) {
  const { engine, tokenizerBytes } = await IgnisEngine.createWithOpfs(
    modelUrl,
    modelKey,
    '/kernels',
    { maxSeqLen: 1024, maxBatchTokens: 64, forceNoSubgroups }
  );
  try {
    const wasmBytes = new Uint8Array(await (await fetch('/runtime/wasm/ignis_tokenizer.wasm')).arrayBuffer());
    const tokBytes = tokenizerBytes || new Uint8Array(await (await fetch(modelUrl + '/tokenizer.json')).arrayBuffer());
    const tok = await IgnisTokenizer.fromBuffers(wasmBytes, tokBytes);
    try {
      const ids = tok.encode(promptText);
      const eosSet = new Set(engine.bp.eos_token_ids || [engine.bp.eos_token_id]);
      const gpuBodyBytes = (engine.manifest.shards || [])
        .filter((s) => typeof s.id === 'number')
        .reduce((acc, s) => acc + s.bytes, 0);

      engine.reset();
      let fwd = await engine.forward(ids);
      const firstLogits = new Float32Array(fwd.logits);
      const genIds = [];
      for (let s = 0; s < maxSteps; s++) {
        const nextTok = sampleToken(fwd.logits, [], { temperature: 0.0 });
        if (eosSet.has(nextTok)) break;
        genIds.push(nextTok);
        if (s + 1 < maxSteps) fwd = await engine.forward([nextTok]);
      }
      return {
        bpWeightBits: engine.bp.weight_bits,
        useSubgroups: engine.useSubgroups,
        gpuBodyMiB: gpuBodyBytes / (1024 * 1024),
        firstLogits,
        genIds,
        reply: tok.decode(genIds, true),
      };
    } finally {
      tok.free();
    }
  } finally {
    engine.destroy();
  }
}

self.onmessage = async (ev) => {
  if (ev.data?.type !== 'run') return;
  try {
    const liveProfile = await detectDeviceProfile();
    const mockUnsupported = await detectDeviceProfile(null);
    const mockNoSg = await detectDeviceProfile({
      features: new Set(['shader-f16']),
      limits: { maxBufferSize: 4 * 1024 ** 3, maxStorageBufferBindingSize: 512 * 1024 ** 2, maxComputeWorkgroupStorageSize: 32768, maxComputeInvocationsPerWorkgroup: 256 },
    });
    const mockLowMem = await detectDeviceProfile({
      features: new Set(['shader-f16', 'subgroups']),
      limits: { maxBufferSize: 512 * 1024 ** 2, maxStorageBufferBindingSize: 128 * 1024 ** 2, maxComputeWorkgroupStorageSize: 32768, maxComputeInvocationsPerWorkgroup: 256 },
    });

    const prompt = '<|startoftext|><|im_start|>user\\nWhat is the capital of France? Answer in one short sentence.<|im_end|>\\n<|im_start|>assistant\\n<think></think>';
    const rSg = await runGreedyText('/tests/fixtures/lfm3-omni-2b-ignis', 'deg-lfm3-q4', false, prompt, 16);
    const rNoSg = await runGreedyText('/tests/fixtures/lfm3-omni-2b-ignis', 'deg-lfm3-q4', true, prompt, 16);
    const logitsCos = cosSim(rSg.firstLogits, rNoSg.firstLogits);
    const exactIdMatch = JSON.stringify(rSg.genIds) === JSON.stringify(rNoSg.genIds);

    const rQ3 = await runGreedyText('/tests/fixtures/lfm3-omni-2b-ignis-q3', 'deg-lfm3-q3', false, prompt, 24);

    self.postMessage({
      ok: true,
      result: {
        liveProfile,
        mockUnsupported,
        mockNoSg,
        mockLowMem,
        sgVsNoSg: {
          useSubgroupsTrue: rSg.useSubgroups,
          useSubgroupsFalse: rNoSg.useSubgroups,
          logitsCos,
          exactIdMatch,
          genIdsSg: rSg.genIds,
          genIdsNoSg: rNoSg.genIds,
          reply: rSg.reply,
        },
        q3: {
          weightBits: rQ3.bpWeightBits,
          gpuBodyMiB: rQ3.gpuBodyMiB,
          genIds: rQ3.genIds,
          reply: rQ3.reply,
        },
      },
    });
  } catch (err) {
    self.postMessage({ ok: false, error: err?.stack || String(err) });
  }
};
`;

const HARNESS_HTML = `<!doctype html><html><body><script type="module">
  window.__RUN_DEG_TEST__ = () => new Promise((resolve, reject) => {
    const w = new Worker('/__phase6_deg_worker__.mjs', { type: 'module' });
    w.onmessage = (e) => { w.terminate(); if (e.data.ok) resolve(e.data.result); else reject(new Error(e.data.error)); };
    w.onerror = (e) => { w.terminate(); reject(new Error(e.message || 'Worker error')); };
    w.postMessage({ type: 'run' });
  });
</script></body></html>`;

async function main() {
  const { server, port } = await startStaticServer(HARNESS_HTML);
  const exePath = resolveBrowserExecutable();
  const userDataDir = fs.mkdtempSync(path.join(os.tmpdir(), 'ignis-deg-'));
  console.log(`[test-phase6-degradation] Launching ${exePath ? path.basename(exePath) : 'playwright-chromium'} against http://127.0.0.1:${port}`);

  const errors = [];
  let context;
  try {
    context = await chromium.launchPersistentContext(userDataDir, {
      ...(exePath ? { executablePath: exePath } : {}),
      headless: true,
      args: ['--headless=new'],
    });
    const page = context.pages()[0] || await context.newPage();
    page.on('console', (m) => { if (m.type() === 'error') errors.push(m.text()); });
    page.on('pageerror', (e) => errors.push(String(e)));

    await page.goto(`http://127.0.0.1:${port}/__phase6_deg__.html`);
    const res = await page.evaluate(() => window.__RUN_DEG_TEST__());
    console.log('\n[test-phase6-degradation RESULTS]', JSON.stringify(res, null, 2));

    const checks = [
      ['Live device profile is tier1_full', res.liveProfile.tier === 'tier1_full' && res.liveProfile.hasF16 && res.liveProfile.hasSubgroups && res.liveProfile.recommendedWeightBits === 4, res.liveProfile.tier],
      ['Null adapter profile is unsupported', res.mockUnsupported.supported === false && res.mockUnsupported.tier === 'unsupported', res.mockUnsupported.tier],
      ['Synthetic no-subgroups mock -> tier2_no_subgroups', res.mockNoSg.tier === 'tier2_no_subgroups' && res.mockNoSg.recommendedUseSubgroups === false, res.mockNoSg.tier],
      ['Synthetic low-VRAM mock -> recommendedWeightBits === 3', res.mockLowMem.recommendedWeightBits === 3 && res.mockLowMem.estimatedVramClass === 'low', res.mockLowMem],
      ['No-subgroups logits cos >= 0.99999 vs subgroups', res.sgVsNoSg.logitsCos >= 0.99999 && res.sgVsNoSg.useSubgroupsTrue && !res.sgVsNoSg.useSubgroupsFalse, res.sgVsNoSg.logitsCos],
      ['No-subgroups 16-step greedy tokens 100% bit-exact match', res.sgVsNoSg.exactIdMatch, res.sgVsNoSg.reply],
      ['INT3_G64 weight_bits === 3 & GPU shards < 800 MiB', res.q3.weightBits === 3 && res.q3.gpuBodyMiB < 800, `${res.q3.gpuBodyMiB.toFixed(2)} MiB`],
      ['INT3_G64 greedy generation mentions Paris', /paris/i.test(res.q3.reply), res.q3.reply],
      ['Zero browser console/page errors', errors.length === 0, errors],
    ];

    let failed = 0;
    for (const [label, ok, val] of checks) {
      console.log(`  [${ok ? 'PASS' : 'FAIL'}] ${label} (${typeof val === 'number' ? val.toFixed(7) : JSON.stringify(val)})`);
      if (!ok) failed++;
    }
    if (failed > 0) throw new Error(`${failed} Stage 6A check(s) failed`);
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
