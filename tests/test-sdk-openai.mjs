import http from 'node:http';
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import { chromium } from 'playwright-core';
import { resolveBrowserExecutable } from './browser-env.mjs';

const ROOT_DIR = path.resolve(import.meta.dirname, '..');
const MODEL_DIR = path.join(ROOT_DIR, 'tests', 'fixtures', 'lfm3-omni-2b-ignis');

if (!fs.existsSync(path.join(MODEL_DIR, 'manifest.json'))) {
  throw new Error(`Missing packed LFM3-Omni model at ${MODEL_DIR}`);
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
      if (cleanUrl === '/__sdk_harness__.html') {
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
  import { IgnisOpenAI } from '/packages/core/index.mjs';

  function makeRedOnBlueRgb(w = 128, h = 128) {
    const data = new Uint8Array(w * h * 3);
    for (let y = 0; y < h; y++) {
      for (let x = 0; x < w; x++) {
        const off = (y * w + x) * 3;
        const inBox = x >= (w >> 2) && x < (w - (w >> 2)) && y >= (h >> 2) && y < (h - (h >> 2));
        data[off] = inBox ? 240 : 0;
        data[off + 1] = inBox ? 20 : 0;
        data[off + 2] = inBox ? 20 : 240;
      }
    }
    return { data, width: w, height: h, channels: 3 };
  }

  window.__RUN_SDK_SUITE__ = async () => {
    const out = {};
    const workerClient = new IgnisOpenAI({
      workerUrl: '/runtime/ignis-worker.mjs',
      models: {
        lfm3: {
          modelUrl: '/tests/fixtures/lfm3-omni-2b-ignis',
          modelKey: 'sdk-test-lfm3',
          maxSeqLen: 1024,
          maxBatchTokens: 64,
        },
      },
    });

    try {
      await workerClient.loadModel('lfm3');

      const nonStream = await workerClient.chat.completions.create({
        model: 'lfm3',
        messages: [{ role: 'user', content: 'What is the capital of France? Answer concisely.' }],
        max_tokens: 16,
        temperature: 0.0,
      });
      out.nonStream = {
        object: nonStream.object,
        content: nonStream.choices?.[0]?.message?.content,
        finishReason: nonStream.choices?.[0]?.finish_reason,
        usage: nonStream.usage,
      };

      const stream = workerClient.chat.completions.create({
        model: 'lfm3',
        stream: true,
        messages: [{ role: 'user', content: 'What is the capital of France? Answer concisely.' }],
        max_tokens: 16,
        temperature: 0.0,
      });
      const chunks = [];
      let streamedText = '';
      for await (const c of stream) {
        chunks.push(c);
        const delta = c.choices?.[0]?.delta?.content;
        if (delta) streamedText += delta;
      }
      out.stream = {
        chunkCount: chunks.length,
        firstRole: chunks[0]?.choices?.[0]?.delta?.role,
        lastFinishReason: chunks[chunks.length - 1]?.choices?.[0]?.finish_reason,
        content: streamedText,
      };

      const rgb = makeRedOnBlueRgb(128, 128);
      const vqa = await workerClient.chat.completions.create({
        model: 'lfm3',
        messages: [{
          role: 'user',
          content: [
            { type: 'image_url', image_url: rgb },
            { type: 'text', text: 'What colors do you see in this image? Reply in one short sentence.' },
          ],
        }],
        max_tokens: 24,
        temperature: 0.0,
      });
      out.vqa = {
        content: vqa.choices?.[0]?.message?.content,
        usage: vqa.usage,
      };
    } finally {
      await workerClient.destroy();
    }

    const directClient = new IgnisOpenAI({
      modelUrl: '/tests/fixtures/lfm3-omni-2b-ignis',
      modelKey: 'sdk-test-lfm3-direct',
      maxSeqLen: 1024,
      maxBatchTokens: 64,
    });
    try {
      await directClient.loadModel('lfm3-direct');
      const directRes = await directClient.chat.completions.create({
        messages: [{ role: 'user', content: 'What is the capital of France? Answer concisely.' }],
        max_tokens: 16,
        temperature: 0.0,
      });
      out.direct = {
        object: directRes.object,
        content: directRes.choices?.[0]?.message?.content,
        usage: directRes.usage,
      };
    } finally {
      await directClient.destroy();
    }

    return out;
  };
</script></body></html>`;

async function main() {
  const { server, port } = await startStaticServer(HARNESS_HTML);
  const exePath = resolveBrowserExecutable();
  const userDataDir = fs.mkdtempSync(path.join(os.tmpdir(), 'ignis-sdk-e2e-'));
  console.log(`[test-sdk-openai] Launching ${exePath ? path.basename(exePath) : 'playwright-chromium'} against http://127.0.0.1:${port}`);

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

    await page.goto(`http://127.0.0.1:${port}/__sdk_harness__.html`, { waitUntil: 'networkidle' });
    const res = await page.evaluate(() => window.__RUN_SDK_SUITE__());
    console.log('\n[test-sdk-openai RESULTS]', JSON.stringify(res, null, 2));

    const checks = [
      ['Worker non-stream returns chat.completion + Paris', res.nonStream.object === 'chat.completion' && /paris/i.test(res.nonStream.content) && res.nonStream.usage.completion_tokens > 0, res.nonStream.content],
      ['Worker streaming >= 2 chunks + assistant role + Paris', res.stream.chunkCount >= 2 && res.stream.firstRole === 'assistant' && !!res.stream.lastFinishReason && /paris/i.test(res.stream.content), res.stream],
      ['Worker VQA image_url mentions red & blue', /red/i.test(res.vqa.content) && /blue/i.test(res.vqa.content), res.vqa.content],
      ['Direct mode non-stream returns Paris', res.direct.object === 'chat.completion' && /paris/i.test(res.direct.content), res.direct.content],
      ['Zero browser console/page errors', errors.length === 0, errors],
    ];

    let failed = 0;
    for (const [label, ok, val] of checks) {
      console.log(`  [${ok ? 'PASS' : 'FAIL'}] ${label} (${JSON.stringify(val)})`);
      if (!ok) failed++;
    }
    if (failed > 0) throw new Error(`${failed} SDK check(s) failed`);
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
