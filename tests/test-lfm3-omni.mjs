import http from 'node:http';
import fs from 'node:fs';
import path from 'node:path';
import { execFileSync } from 'node:child_process';
import { chromium } from 'playwright-core';

const ROOT_DIR = path.resolve(import.meta.dirname, '..');
const THORIUM_PATH = 'C:\\Users\\ray\\AppData\\Local\\Thorium\\Application\\thorium.exe';
const EDGE_PATH = 'C:\\Program Files (x86)\\Microsoft\\Edge\\Application\\msedge.exe';
const GOLDEN_DIR = path.join(ROOT_DIR, 'tests', 'fixtures', 'lfm3-omni-golden');
const MODEL_DIR = path.join(ROOT_DIR, 'tests', 'fixtures', 'lfm3-omni-2b-ignis');

if (!fs.existsSync(path.join(MODEL_DIR, 'manifest.json'))) {
  throw new Error(`Missing packed LFM3-Omni model at ${MODEL_DIR}`);
}
if (!fs.existsSync(path.join(GOLDEN_DIR, 'golden.json'))) {
  console.log('[test-lfm3-omni] Generating PyTorch golden reference...');
  execFileSync(path.join(ROOT_DIR, 'packer', '.venv', 'Scripts', 'python.exe'), [path.join(ROOT_DIR, 'tests', 'gen_lfm3_golden.py')], {
    cwd: ROOT_DIR,
    stdio: 'inherit',
  });
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
    '.bin': 'application/octet-stream',
  };
  return new Promise((resolve) => {
    const server = http.createServer((req, res) => {
      const cleanUrl = (req.url || '/').split('?')[0];
      if (cleanUrl === '/favicon.ico') { res.writeHead(204); res.end(); return; }
      if (cleanUrl === '/__lfm3_harness__.html') {
        res.writeHead(200, {
          'Content-Type': 'text/html',
          'Cross-Origin-Opener-Policy': 'same-origin',
          'Cross-Origin-Embedder-Policy': 'require-corp',
        });
        res.end(harnessHtml);
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

const HARNESS_HTML = `<!doctype html><html><body><script type="module">
  import { IgnisEngine } from '/runtime/ignis-engine.mjs';
  import { IgnisTokenizer } from '/runtime/wasm-tokenizer.mjs';
  import { Lfm3OmniRuntime } from '/runtime/multimodal/lfm3-omni-runtime.mjs';
  import { preprocessLfm3Image } from '/runtime/multimodal/image-proc.mjs';
  import { extractLfm3LogMel } from '/runtime/multimodal/audio-dsp.mjs';

  window.__RUN_LFM3_SUITE__ = async () => {
    const modelUrl = '/tests/fixtures/lfm3-omni-2b-ignis';
    const goldenUrl = '/tests/fixtures/lfm3-omni-golden';

    const fetchJson = async (u) => (await fetch(u)).json();
    const fetchF32 = async (u) => new Float32Array(await (await fetch(u)).arrayBuffer());
    const fetchU8 = async (u) => new Uint8Array(await (await fetch(u)).arrayBuffer());

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
    const seededRng = (seed = 12345) => {
      let s = seed >>> 0;
      return () => {
        s = (s + 0x6d2b79f5) >>> 0;
        let t = Math.imul(s ^ (s >>> 15), 1 | s);
        t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
        return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
      };
    };

    const golden = await fetchJson(goldenUrl + '/golden.json');
    const engine = await IgnisEngine.create(modelUrl, '/kernels', { maxSeqLen: 1024, maxBatchTokens: 64 });
    const [wasmBytes, tokBytes] = await Promise.all([
      fetchU8('/runtime/wasm/ignis_tokenizer.wasm'),
      fetchU8(modelUrl + '/tokenizer.json'),
    ]);
    const tokenizer = await IgnisTokenizer.fromBuffers(wasmBytes, tokBytes);
    const omni = new Lfm3OmniRuntime(engine, { modelBaseUrl: modelUrl });

    const results = {};

    // 1. Text Forward & Greedy Generation
    engine.reset();
    const textFwd = await engine.forward(golden.text.input_ids);
    const expTextLogits = await fetchF32(goldenUrl + '/text_logits_last.bin');
    const textCos = cosSim(textFwd.logits, expTextLogits);
    const textTop1 = argmax(textFwd.logits);
    const expTextTop1 = argmax(expTextLogits);

    const textGen = await omni.generateOmni({
      inputIds: golden.text.input_ids,
      generationMode: 'text',
      maxNewTokens: 16,
      temperature: 0.0,
    });
    const textReply = tokenizer.decode(textGen.textTokenIds, true);
    results.text = { textCos, textTop1, expTextTop1, top1Match: textTop1 === expTextTop1, textReply, expReply: golden.text.reply };

    // 2. Vision Preprocessing, Encoder Features, VQA Forward & Reply
    const imgU8 = await fetchU8(goldenUrl + '/image_rgb_256.bin');
    const prep = preprocessLfm3Image({ data: imgU8, width: 256, height: 256, channels: 3 });
    const expPix = await fetchF32(goldenUrl + '/pixel_values_hf.bin');
    const pixCos = cosSim(prep.pixelValues, expPix);

    const visFeats = await omni.encodeImage(prep.pixelValues, prep.gridThw);
    const expVisFeats = await fetchF32(goldenUrl + '/image_features_hf.bin');
    const visCos = cosSim(visFeats, expVisFeats);

    engine.reset();
    const D = engine.bp.hidden_size;
    const visEmb = await engine.lookupEmbeddings(golden.vision.expanded_prompt_ids);
    const imgTok = engine.bp.multimodal.image_token_id;
    let vPtr = 0;
    for (let t = 0; t < golden.vision.expanded_prompt_ids.length; t++) {
      if (golden.vision.expanded_prompt_ids[t] === imgTok) {
        visEmb.set(visFeats.subarray(vPtr * D, (vPtr + 1) * D), t * D);
        vPtr++;
      }
    }
    const vqaFwd = await engine.forwardEmbeds(visEmb);
    const expVqaLogits = await fetchF32(goldenUrl + '/vqa_logits_last.bin');
    const vqaCos = cosSim(vqaFwd.logits, expVqaLogits);

    const vqaGen = await omni.generateOmni({
      inputIds: golden.vision.raw_prompt_ids,
      images: [{ data: imgU8, width: 256, height: 256, channels: 3 }],
      generationMode: 'text',
      maxNewTokens: 24,
      temperature: 0.0,
    });
    const vqaReply = tokenizer.decode(vqaGen.textTokenIds, true);
    results.vision = { pixCos, visCos, vqaCos, vqaTop1Match: argmax(vqaFwd.logits) === argmax(expVqaLogits), vqaReply, expVqaReply: golden.vision.reply };

    // 3. Audio Mel + Audio Encoder + Audio-In Logits
    const pcm16k = await fetchF32(goldenUrl + '/audio_16k.bin');
    const mel = extractLfm3LogMel(pcm16k);
    const expMel = await fetchF32(goldenUrl + '/mel_hf.bin');
    const melCos = cosSim(mel.features, expMel);

    const audFeats = await omni.encodeAudio(mel.features, mel.numFrames);
    const expAudFeats = await fetchF32(goldenUrl + '/audio_features_hf.bin');
    const audCos = cosSim(audFeats, expAudFeats);

    engine.reset();
    const audEmb = await engine.lookupEmbeddings(golden.audio_in.expanded_prompt_ids);
    const audTok = engine.bp.multimodal.audio_token_id;
    let aPtr = 0;
    for (let t = 0; t < golden.audio_in.expanded_prompt_ids.length; t++) {
      if (golden.audio_in.expanded_prompt_ids[t] === audTok) {
        audEmb.set(audFeats.subarray(aPtr * D, (aPtr + 1) * D), t * D);
        aPtr++;
      }
    }
    const audFwd = await engine.forwardEmbeds(audEmb);
    const expAudLogits = await fetchF32(goldenUrl + '/audio_in_logits_last.bin');
    const audLogitsCos = cosSim(audFwd.logits, expAudLogits);
    results.audioIn = { melCos, audCos, audLogitsCos, top1Match: argmax(audFwd.logits) === golden.audio_in.top5_logits[0], numTokens: audFeats.length / D };

    // 4. Audio-Out: Codebook Emb + DepthDecoder + LiquidCodec 24 kHz Decode
    await omni.ensureAudioDecLoaded();
    const cbEmb = await omni.lookupCodebookEmbed(golden.speech_out.greedy_frame0);
    const expCbEmb = await fetchF32(goldenUrl + '/codebook_emb_frame0.bin');
    const cbCos = cosSim(cbEmb, expCbEmb);

    const hLast = await fetchF32(goldenUrl + '/speech_hidden_last.bin');
    const depthBase = await omni._depthBaseProj(hLast);
    const logitsPass0 = await omni.runDepthDecoderPass(
      depthBase,
      new Int32Array(12),
      new Array(12).fill(true),
      [0]
    );
    const expDepthLogits0 = await fetchF32(goldenUrl + '/depth_pass0_cb0_logits.bin');
    const depthLogitsCos = cosSim(logitsPass0.get(0), expDepthLogits0);

    const greedyCodes = await omni.sampleAudioFrame(hLast, {
      temperature: 0,
      repetitionPenalty: 1.0,
      order: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
    });
    const expCodes = golden.speech_out.greedy_frame0;
    let codeMatches = 0;
    for (let k = 0; k < 12; k++) if (greedyCodes[k] === expCodes[k]) codeMatches++;

    const synWav = await omni.decodeAudioCodes(golden.speech_out.syn_codes_frame_major);
    const expWav = await fetchF32(goldenUrl + '/codec_wav_24k.bin');
    const codecWavCos = cosSim(synWav, expWav);
    results.audioOut = { cbCos, depthLogitsCos, codeMatches, greedyCodes: Array.from(greedyCodes), expCodes, codecWavCos, wavLenMatch: synWav.length === expWav.length };

    // 5. E2E Speech-Out Generation (audio + interleaved modes)
    const e2eAudio = await omni.generateOmni({
      inputIds: golden.speech_out.prompt_ids.slice(0, -1),
      generationMode: 'audio',
      maxNewTokens: 20,
      audioTemperature: 0.4,
      audioRepetitionPenalty: 1.2,
      rng: seededRng(2026),
    });
    const audioFinite = e2eAudio.pcm24k.every((v) => Number.isFinite(v));
    const wavHdr = e2eAudio.wavBytes ? String.fromCharCode(...e2eAudio.wavBytes.slice(0, 4)) : '';

    const e2eMixed = await omni.generateOmni({
      inputIds: golden.speech_out.prompt_ids.slice(0, -1),
      generationMode: 'interleaved',
      maxNewTokens: 42,
      temperature: 0.0,
      audioTemperature: 0.4,
      audioRepetitionPenalty: 1.2,
      rng: seededRng(777),
    });
    let rmsMixed = 0;
    let mixedFinite = true;
    for (let i = 0; i < e2eMixed.pcm24k.length; i++) {
      const v = e2eMixed.pcm24k[i];
      if (!Number.isFinite(v)) mixedFinite = false;
      rmsMixed += v * v;
    }
    rmsMixed = e2eMixed.pcm24k.length ? Math.sqrt(rmsMixed / e2eMixed.pcm24k.length) : 0;
    const mixedWavHdr = e2eMixed.wavBytes ? String.fromCharCode(...e2eMixed.wavBytes.slice(0, 4)) : '';
    const mixedText = tokenizer.decode(e2eMixed.textTokenIds, true);
    results.e2eSpeech = {
      numFrames: e2eAudio.audioCodes.length,
      pcmSamples: e2eAudio.pcm24k.length,
      audioFinite,
      wavHdr,
      mixedTextTokens: e2eMixed.textTokenIds.length,
      mixedAudioFrames: e2eMixed.audioCodes.length,
      mixedSamples: e2eMixed.pcm24k.length,
      mixedFinite,
      rmsMixed,
      mixedWavHdr,
      mixedText,
    };

    omni.destroy();
    engine.destroy();
    return results;
  };
</script></body></html>`;

async function runBrowserSuite() {
  const { server, port } = await startStaticServer(HARNESS_HTML);
  const exePath = fs.existsSync(THORIUM_PATH) ? THORIUM_PATH : EDGE_PATH;
  console.log(`[test-lfm3-omni] Launching ${path.basename(exePath)} against http://127.0.0.1:${port}`);

  const browser = await chromium.launch({
    executablePath: exePath,
    headless: true,
    args: ['--headless=new'],
  });

  try {
    const context = await browser.newContext();
    const page = await context.newPage();
    page.on('console', (m) => {
      if (m.type() === 'error') console.error('[browser error]', m.text());
    });
    page.on('pageerror', (e) => console.error('[pageerror]', e.message));

    await page.goto(`http://127.0.0.1:${port}/__lfm3_harness__.html`);

    const res = await page.evaluate(() => window.__RUN_LFM3_SUITE__());
    console.log('\n[test-lfm3-omni RESULTS]', JSON.stringify(res, null, 2));

    const checks = [
      ['Text logits cos >= 0.995', res.text.textCos >= 0.995, res.text.textCos],
      ['Text reply mentions Paris', /paris/i.test(res.text.textReply), res.text.textReply],
      ['Vision pix cos >= 0.999', res.vision.pixCos >= 0.999, res.vision.pixCos],
      ['Vision features cos >= 0.99', res.vision.visCos >= 0.99, res.vision.visCos],
      ['VQA logits cos >= 0.99', res.vision.vqaCos >= 0.99, res.vision.vqaCos],
      ['VQA reply describes red/blue', /red/i.test(res.vision.vqaReply) && /blue/i.test(res.vision.vqaReply), res.vision.vqaReply],
      ['Audio mel cos >= 0.999', res.audioIn.melCos >= 0.999, res.audioIn.melCos],
      ['Audio features cos >= 0.99', res.audioIn.audCos >= 0.99, res.audioIn.audCos],
      ['Audio-in logits cos >= 0.99', res.audioIn.audLogitsCos >= 0.99, res.audioIn.audLogitsCos],
      ['Codebook emb cos >= 0.995', res.audioOut.cbCos >= 0.995, res.audioOut.cbCos],
      ['DepthDecoder pass0 logits cos >= 0.999', res.audioOut.depthLogitsCos >= 0.999, res.audioOut.depthLogitsCos],
      ['DepthDecoder greedy match >= 8/12 (0.0016 near-tie FSQ)', res.audioOut.codeMatches >= 8, `${res.audioOut.codeMatches}/12`],
      ['LiquidCodec 24kHz wav cos >= 0.99', res.audioOut.codecWavCos >= 0.99 && res.audioOut.wavLenMatch, res.audioOut.codecWavCos],
      ['E2E audio mode generates finite 24kHz WAV', res.e2eSpeech.numFrames >= 4 && res.e2eSpeech.audioFinite && res.e2eSpeech.wavHdr === 'RIFF', res.e2eSpeech.numFrames],
      ['E2E interleaved emits audible speech WAV + text', res.e2eSpeech.mixedTextTokens >= 1 && res.e2eSpeech.mixedAudioFrames >= 12 && res.e2eSpeech.mixedFinite && res.e2eSpeech.rmsMixed > 0.01 && res.e2eSpeech.mixedWavHdr === 'RIFF', res.e2eSpeech],
    ];

    let failed = 0;
    for (const [label, ok, val] of checks) {
      console.log(`  [${ok ? 'PASS' : 'FAIL'}] ${label} (${typeof val === 'number' ? val.toFixed(6) : JSON.stringify(val)})`);
      if (!ok) failed++;
    }
    if (failed > 0) throw new Error(`${failed} LFM3-Omni check(s) failed`);
  } finally {
    await browser.close();
    server.close();
  }
}

runBrowserSuite().catch((err) => {
  console.error(err);
  process.exit(1);
});
