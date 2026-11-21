import http from 'node:http';
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import { execFileSync } from 'node:child_process';
import { chromium } from 'playwright-core';
import { resolveBrowserExecutable, resolvePythonExecutable } from './browser-env.mjs';

const ROOT_DIR = path.resolve(import.meta.dirname, '..');
const GOLDEN_DIR = path.join(ROOT_DIR, 'tests', 'fixtures', 'gemma4-golden');
const MODEL_DIR = path.join(ROOT_DIR, 'tests', 'fixtures', 'gemma-4-e4b-ignis');

if (!fs.existsSync(path.join(MODEL_DIR, 'manifest.json'))) {
  throw new Error(`Missing packed Gemma-4-E4B-it model at ${MODEL_DIR}`);
}
if (!fs.existsSync(path.join(GOLDEN_DIR, 'golden.json'))) {
  console.log('[test-gemma4-e4b] Generating PyTorch golden reference...');
  execFileSync(resolvePythonExecutable(ROOT_DIR), [path.join(ROOT_DIR, 'tests', 'gen_gemma4_golden.py')], {
    cwd: ROOT_DIR,
    stdio: 'inherit',
  });
}

const WORKER_JS = `
import { IgnisEngine } from '/runtime/ignis-engine.mjs';
import { IgnisTokenizer } from '/runtime/wasm-tokenizer.mjs';
import { Gemma4OmniRuntime } from '/runtime/multimodal/gemma4-omni-runtime.mjs';
import { preprocessGemma4Image } from '/runtime/multimodal/image-proc.mjs';
import { extractGemma4LogMel } from '/runtime/multimodal/audio-dsp.mjs';

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

const argmax = (arr) => {
  let best = 0;
  for (let i = 1; i < arr.length; i++) if (arr[i] > arr[best]) best = i;
  return best;
};

self.onmessage = async (ev) => {
  if (ev.data?.type !== 'run') return;
  try {
    const modelUrl = '/tests/fixtures/gemma-4-e4b-ignis';
    const goldenUrl = '/tests/fixtures/gemma4-golden';
    const fetchJson = async (u) => (await fetch(u)).json();
    const fetchF32 = async (u) => new Float32Array(await (await fetch(u)).arrayBuffer());
    const fetchU8 = async (u) => new Uint8Array(await (await fetch(u)).arrayBuffer());

    const golden = await fetchJson(goldenUrl + '/golden.json');
    const { engine, tokenizerBytes } = await IgnisEngine.createWithOpfs(
      modelUrl,
      'gemma-4-e4b-test',
      '/kernels',
      { maxSeqLen: 1024, maxBatchTokens: 64 }
    );
    const wasmBytes = await fetchU8('/runtime/wasm/ignis_tokenizer.wasm');
    const tokBytes = tokenizerBytes || await fetchU8(modelUrl + '/tokenizer.json');
    const tokenizer = await IgnisTokenizer.fromBuffers(wasmBytes, tokBytes);
    const omni = new Gemma4OmniRuntime(engine, { modelBaseUrl: modelUrl, exclusiveTower: true });
    const baseGpuBytes = omni._recordPeakGpuBytes();

    const results = { baseGpuMiB: baseGpuBytes / (1024 * 1024) };

    // 1. Text Forward & Greedy Generation
    engine.reset();
    const textFwd = await engine.forward(golden.text.input_ids);
    const expTextLogits = await fetchF32(goldenUrl + '/text_logits_last.bin');
    const textCos = cosSim(textFwd.logits, expTextLogits);
    const textTop1 = argmax(textFwd.logits);
    const expTextTop1 = argmax(expTextLogits);

    const textGen = await omni.generateOmni(golden.text.input_ids, {
      maxNewTokens: 16,
      temperature: 0.0,
    });
    const textReply = tokenizer.decode(textGen.textTokenIds, true);
    results.text = {
      textCos,
      textTop1,
      expTextTop1,
      top1Match: textTop1 === expTextTop1,
      textReply,
      expReply: golden.text.reply,
    };

    // 2. Vision Preprocessing, Encoder Features, VQA Forward & Reply
    const imgU8 = await fetchU8(goldenUrl + '/image_rgb_96.bin');
    const prep = preprocessGemma4Image({ data: imgU8, width: 96, height: 96, channels: 3 });
    const expPix = await fetchF32(goldenUrl + '/pixel_values_real_hf.bin');
    const pixCos = cosSim(prep.pixelValues, expPix);

    const visRes = await omni.encodeImage(imgU8, 96, 96);
    const expVisFeats = await fetchF32(goldenUrl + '/image_features_hf.bin');
    const visCos = cosSim(visRes.embeds, expVisFeats);
    const visionPeakMiB = omni.peakGpuBytes / (1024 * 1024);

    omni.unloadShard('vision');
    const vqaBuilt = await omni.buildOmniPromptEmbeddings(golden.vision.expanded_prompt_ids, {
      imageEmbeds: [visRes],
    });
    engine.reset();
    const vqaFwd = await engine.forwardEmbeds(vqaBuilt.embeds, {
      tokenIds: vqaBuilt.tokenIds,
      isMultimodalMask: vqaBuilt.isMultimodalMask,
    });
    const expVqaLogits = await fetchF32(goldenUrl + '/vqa_logits_last.bin');
    const vqaCos = cosSim(vqaFwd.logits, expVqaLogits);

    const vqaGen = await omni.generateOmni(golden.vision.raw_prompt_ids, {
      images: [{ data: imgU8, width: 96, height: 96, channels: 3 }],
      maxNewTokens: 20,
      temperature: 0.0,
      releaseVisionAfterEncode: true,
    });
    const vqaReply = tokenizer.decode(vqaGen.textTokenIds, true);
    results.vision = {
      pixCos,
      visCos,
      vqaCos,
      vqaTop1Match: argmax(vqaFwd.logits) === argmax(expVqaLogits),
      vqaReply,
      expVqaReply: golden.vision.reply,
      numTokens: visRes.numTokens,
      visionPeakMiB,
    };

    // 3. Audio Mel + Audio Encoder + Audio-In Logits
    const pcm16k = await fetchF32(goldenUrl + '/audio_16k.bin');
    const mel = extractGemma4LogMel(pcm16k);
    const expMel = await fetchF32(goldenUrl + '/mel_hf.bin');
    const melCos = cosSim(mel.features, expMel);

    const audRes = await omni.encodeAudio(pcm16k);
    const expAudFeats = await fetchF32(goldenUrl + '/audio_features_hf.bin');
    const audCos = cosSim(audRes.embeds, expAudFeats);

    omni.unloadShard('audio_enc');
    const audBuilt = await omni.buildOmniPromptEmbeddings(golden.audio_in.expanded_prompt_ids, {
      audioEmbeds: [audRes],
    });
    engine.reset();
    const audFwd = await engine.forwardEmbeds(audBuilt.embeds, {
      tokenIds: audBuilt.tokenIds,
      isMultimodalMask: audBuilt.isMultimodalMask,
    });
    const expAudLogits = await fetchF32(goldenUrl + '/audio_in_logits_last.bin');
    const audLogitsCos = cosSim(audFwd.logits, expAudLogits);

    results.audioIn = {
      melCos,
      audCos,
      audLogitsCos,
      top1Match: argmax(audFwd.logits) === golden.audio_in.top5_logits[0],
      numTokens: audRes.numTokens,
      melFrames: mel.numFrames,
    };
    results.peakGpuBytes = omni.peakGpuBytes;
    results.peakGpuMiB = omni.peakGpuBytes / (1024 * 1024);

    omni.destroy();
    engine.destroy();
    self.postMessage({ type: 'done', results });
  } catch (err) {
    self.postMessage({ type: 'error', message: err?.stack || err?.message || String(err) });
  }
};
`;

const HARNESS_HTML = `<!doctype html><html><body><script type="module">
  window.__RUN_GEMMA4_SUITE__ = () => new Promise((resolve, reject) => {
    const worker = new Worker('/__gemma4_worker__.mjs', { type: 'module' });
    worker.onmessage = (ev) => {
      if (ev.data?.type === 'done') {
        worker.terminate();
        resolve(ev.data.results);
      } else if (ev.data?.type === 'error') {
        worker.terminate();
        reject(new Error(ev.data.message));
      }
    };
    worker.onerror = (err) => {
      worker.terminate();
      reject(new Error(err.message || 'Worker error'));
    };
    worker.postMessage({ type: 'run' });
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
    '.bin': 'application/octet-stream',
  };
  return new Promise((resolve) => {
    const server = http.createServer((req, res) => {
      const cleanUrl = (req.url || '/').split('?')[0];
      if (cleanUrl === '/favicon.ico') { res.writeHead(204); res.end(); return; }
      const headers = (ct) => ({
        'Content-Type': ct,
        'Cross-Origin-Opener-Policy': 'same-origin',
        'Cross-Origin-Embedder-Policy': 'require-corp',
      });
      if (cleanUrl === '/__gemma4_harness__.html') {
        res.writeHead(200, headers('text/html'));
        res.end(HARNESS_HTML);
        return;
      }
      if (cleanUrl === '/__gemma4_worker__.mjs') {
        res.writeHead(200, headers('application/javascript'));
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
      res.writeHead(200, headers(mime[ext] || 'application/octet-stream'));
      fs.createReadStream(fullPath).pipe(res);
    });
    server.listen(0, '127.0.0.1', () => resolve({ server, port: server.address().port }));
  });
}

async function runBrowserSuite() {
  const { server, port } = await startStaticServer();
  const exePath = resolveBrowserExecutable();
  const tmpProfileDir = fs.mkdtempSync(path.join(os.tmpdir(), 'ignis-gemma4-opfs-'));
  console.log(`[test-gemma4-e4b] Launching ${exePath ? path.basename(exePath) : 'playwright-chromium'} (persistent OPFS profile) against http://127.0.0.1:${port}`);

  const context = await chromium.launchPersistentContext(tmpProfileDir, {
    ...(exePath ? { executablePath: exePath } : {}),
    headless: true,
    args: ['--headless=new'],
  });

  try {
    const page = context.pages()[0] || await context.newPage();
    page.on('console', (m) => {
      if (m.type() === 'error') console.error('[browser error]', m.text());
    });
    page.on('pageerror', (e) => console.error('[pageerror]', e.message));

    await page.goto(`http://127.0.0.1:${port}/__gemma4_harness__.html`);
    const res = await page.evaluate(() => window.__RUN_GEMMA4_SUITE__());
    console.log('\n[test-gemma4-e4b RESULTS]', JSON.stringify(res, null, 2));

    const MAX_VRAM = 3.2 * 1024 * 1024 * 1024;
    const checks = [
      ['Text logits cos >= 0.995', res.text.textCos >= 0.995 && res.text.top1Match, res.text.textCos],
      ['Text reply mentions Paris', /paris/i.test(res.text.textReply), res.text.textReply],
      ['Vision pix cos >= 0.999', res.vision.pixCos >= 0.999, res.vision.pixCos],
      ['Vision features cos >= 0.999', res.vision.visCos >= 0.999, res.vision.visCos],
      ['VQA logits cos >= 0.99 & top1Match', res.vision.vqaCos >= 0.99 && res.vision.vqaTop1Match, res.vision.vqaCos],
      ['VQA reply describes red/blue', /red/i.test(res.vision.vqaReply) && /blue/i.test(res.vision.vqaReply), res.vision.vqaReply],
      ['Audio mel cos >= 0.9999', res.audioIn.melCos >= 0.9999, res.audioIn.melCos],
      ['Audio features cos >= 0.999', res.audioIn.audCos >= 0.999, res.audioIn.audCos],
      ['Audio-in logits cos >= 0.94 & top1Match', res.audioIn.audLogitsCos >= 0.94 && res.audioIn.top1Match, res.audioIn.audLogitsCos],
      ['Peak GPU VRAM < 3.2 GiB', res.peakGpuBytes < MAX_VRAM, `${res.peakGpuMiB.toFixed(1)} MiB`],
    ];

    let failed = 0;
    for (const [label, ok, val] of checks) {
      console.log(`  [${ok ? 'PASS' : 'FAIL'}] ${label} (${typeof val === 'number' ? val.toFixed(6) : JSON.stringify(val)})`);
      if (!ok) failed++;
    }
    if (failed > 0) throw new Error(`${failed} Gemma-4-E4B-it check(s) failed`);
  } finally {
    await context.close();
    server.close();
    fs.rmSync(tmpProfileDir, { recursive: true, force: true });
  }
}

runBrowserSuite().catch((err) => {
  console.error(err);
  process.exit(1);
});
