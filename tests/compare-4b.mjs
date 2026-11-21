import http from 'node:http';
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import { chromium } from 'playwright-core';
import { resolveBrowserExecutable } from './browser-env.mjs';

const EDGE_PATH = resolveBrowserExecutable();
const ROOT_DIR = path.resolve(import.meta.dirname, '..');

function analyzeLogits(bf16Path, ignisBuf) {
  const bRaw = fs.readFileSync(bf16Path);
  const bf16 = new Float32Array(bRaw.buffer, bRaw.byteOffset, bRaw.byteLength / 4);
  const ig = new Float32Array(ignisBuf.buffer, ignisBuf.byteOffset, ignisBuf.byteLength / 4);
  if (bf16.length !== ig.length) throw new Error(`Logits length mismatch ${bf16.length} vs ${ig.length}`);

  let dot = 0, na = 0, nb = 0, maxDiff = 0;
  for (let i = 0; i < bf16.length; i++) {
    const a = bf16[i], b = ig[i];
    dot += a * b; na += a * a; nb += b * b;
    const d = Math.abs(a - b);
    if (d > maxDiff) maxDiff = d;
  }
  const cos = dot / (Math.sqrt(na) * Math.sqrt(nb) + 1e-12);

  const topK = (arr, k) => {
    const idxs = Array.from({ length: arr.length }, (_, i) => i);
    idxs.sort((x, y) => arr[y] - arr[x]);
    return idxs.slice(0, k);
  };
  const bTop5 = topK(bf16, 5);
  const iTop5 = topK(ig, 5);
  const top1Match = bTop5[0] === iTop5[0];
  const bf16Top1InTop5 = iTop5.includes(bTop5[0]);
  const top5Overlap = bTop5.filter((x) => iTop5.includes(x)).length;

  // Exact KL(P_bf16 || P_ignis)
  const softmax = (arr) => {
    let mx = -Infinity;
    for (let i = 0; i < arr.length; i++) if (arr[i] > mx) mx = arr[i];
    const exps = new Float64Array(arr.length);
    let sum = 0;
    for (let i = 0; i < arr.length; i++) {
      const e = Math.exp(arr[i] - mx);
      exps[i] = e; sum += e;
    }
    for (let i = 0; i < arr.length; i++) exps[i] /= sum;
    return exps;
  };
  const p = softmax(bf16);
  const q = softmax(ig);
  let kl = 0;
  for (let i = 0; i < p.length; i++) {
    if (p[i] > 1e-12) {
      kl += p[i] * Math.log(p[i] / Math.max(q[i], 1e-12));
    }
  }
  return {
    cosineSim: +cos.toFixed(6),
    klDivergence: +kl.toFixed(5),
    maxLogitDiff: +maxDiff.toFixed(4),
    top1Match,
    bf16Top1InTop5,
    top5Overlap,
    bf16Top5: bTop5,
    ignisTop5: iTop5,
  };
}

function startServer(uploadedLogits) {
  return new Promise((resolve) => {
    const server = http.createServer((req, res) => {
      const urlObj = new URL(req.url || '/', 'http://127.0.0.1');
      if (urlObj.pathname === '/favicon.ico') { res.writeHead(204); res.end(); return; }
      if (urlObj.pathname === '/upload-logits' && req.method === 'POST') {
        const name = urlObj.searchParams.get('name');
        const chunks = [];
        req.on('data', (c) => chunks.push(c));
        req.on('end', () => {
          uploadedLogits.set(name, Buffer.concat(chunks));
          res.writeHead(200);
          res.end('ok');
        });
        return;
      }
      const filePath = path.join(ROOT_DIR, urlObj.pathname.replace(/^\/+/, ''));
      if (!filePath.startsWith(ROOT_DIR) || !fs.existsSync(filePath) || fs.statSync(filePath).isDirectory()) {
        res.writeHead(404); res.end('Not Found'); return;
      }
      const ext = path.extname(filePath);
      const mime = ext === '.html' ? 'text/html; charset=utf-8'
        : (ext === '.js' || ext === '.mjs') ? 'application/javascript; charset=utf-8'
        : ext === '.json' ? 'application/json; charset=utf-8'
        : ext === '.wasm' ? 'application/wasm'
        : (ext === '.wgsl' || ext === '.txt') ? 'text/plain; charset=utf-8'
        : 'application/octet-stream';
      res.writeHead(200, {
        'Content-Type': mime,
        'Cross-Origin-Opener-Policy': 'same-origin',
        'Cross-Origin-Embedder-Policy': 'require-corp',
      });
      fs.createReadStream(filePath).pipe(res);
    });
    server.listen(0, '127.0.0.1', () => resolve({ server, port: server.address().port }));
  });
}

async function main() {
  const uploadedLogits = new Map();
  const { server, port } = await startServer(uploadedLogits);
  const golden = JSON.parse(fs.readFileSync(path.join(ROOT_DIR, 'tests/fixtures/spark-4b-bf16-golden.json'), 'utf-8'));
  const prompts = golden.map((g) => ({ name: g.name, ids: g.ids }));
  const userDataDir = fs.mkdtempSync(path.join(os.tmpdir(), 'ignis-4b-eval-'));

  let context;
  try {
    context = await chromium.launchPersistentContext(userDataDir, {
      ...(EDGE_PATH ? { executablePath: EDGE_PATH } : {}),
      headless: true,
      args: [
        '--headless=new',
        '--enable-unsafe-webgpu',
        '--use-angle=d3d11',
        '--enable-features=WebGPUDeveloperFeatures,Vulkan',
        '--ignore-gpu-blocklist',
      ],
    });
    const page = context.pages()[0] || await context.newPage();
    page.on('console', (m) => { if (m.type() === 'error') console.error('[console]', m.text()); });
    page.on('pageerror', (e) => console.error('[pageerror]', e));

    await page.goto(`http://127.0.0.1:${port}/tests/compare-4b-runner.html`, { waitUntil: 'domcontentloaded' });

    const variants = [
      { label: 'INT4_G64_LMHead', url: '/tests/fixtures/spark-4b-ignis-q4' },
      { label: 'INT8_G64_LMHead', url: '/tests/fixtures/spark-4b-ignis-q8head' },
    ];

    const summary = {};
    for (const v of variants) {
      uploadedLogits.clear();
      const res = await page.evaluate(
        ({ url, prompts }) => window.__eval4BVariant({ modelBaseUrl: url, prompts }),
        { url: v.url, prompts }
      );
      for (const pr of res.prompts) {
        const gItem = golden.find((g) => g.name === pr.name);
        const logitsBuf = uploadedLogits.get(pr.name);
        pr.qualityVsBf16 = analyzeLogits(path.join(ROOT_DIR, gItem.logits_file), logitsBuf);
        pr.bf16GreedyIds = gItem.greedy_ids;
        let prefixMatch = 0;
        for (let k = 0; k < Math.min(pr.greedyIds.length, gItem.greedy_ids.length); k++) {
          if (pr.greedyIds[k] === gItem.greedy_ids[k]) prefixMatch++;
          else break;
        }
        pr.greedyPrefixMatch = prefixMatch;
      }
      summary[v.label] = res;
    }

    // Also run Worker + OPFS cold/warm test on INT4_G64_LMHead
    const coldWorker = await page.evaluate(
      ({ ids }) => window.__eval4BWorkerOpfs({
        modelBaseUrl: '/tests/fixtures/spark-4b-ignis-q4',
        modelKey: 'spark-4b-q4',
        promptIds: ids,
      }),
      { ids: prompts[0].ids }
    );
    await page.reload({ waitUntil: 'domcontentloaded' });
    const warmWorker = await page.evaluate(
      ({ ids }) => window.__eval4BWorkerOpfs({
        modelBaseUrl: '/tests/fixtures/spark-4b-ignis-q4',
        modelKey: 'spark-4b-q4',
        promptIds: ids,
      }),
      { ids: prompts[0].ids }
    );

    const out = { summary, workerOpfs: { cold: coldWorker, warm: warmWorker } };
    fs.writeFileSync(path.join(ROOT_DIR, 'tests/fixtures/spark-4b-comparison.json'), JSON.stringify(out, null, 2));
    console.log(JSON.stringify(out, null, 2));
  } finally {
    if (context) await context.close();
    server.close();
    fs.rmSync(userDataDir, { recursive: true, force: true });
  }
}

main().catch((err) => { console.error(err); process.exit(1); });
