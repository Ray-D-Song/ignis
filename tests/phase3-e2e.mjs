import http from 'node:http';
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import { chromium } from 'playwright-core';
import { resolveBrowserExecutable } from './browser-env.mjs';

const EDGE_PATH = resolveBrowserExecutable();
const ROOT_DIR = path.resolve(import.meta.dirname, '..');

function startServer() {
  return new Promise((resolve) => {
    const server = http.createServer((req, res) => {
      const reqPath = (req.url || '/').split('?')[0];
      if (reqPath === '/favicon.ico') { res.writeHead(204); res.end(); return; }
      const filePath = path.join(ROOT_DIR, reqPath.replace(/^\/+/, ''));
      if (!filePath.startsWith(ROOT_DIR) || !fs.existsSync(filePath) || fs.statSync(filePath).isDirectory()) {
        res.writeHead(404);
        res.end('Not Found');
        return;
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

function assertRun(res, label) {
  if (res.deviceLost) throw new Error(`[${label}] GPUDevice lost: ${res.deviceLost}`);
  if (!res.tokRoundTripOk) throw new Error(`[${label}] Tokenizer round-trip failed`);
  const expectedTok = [26240, 25, 3179, 14, 220, 16491, 3078, 220, 30, 31, 32, 33, 34, 1627, 52399, 5211, 2407, 948, 1812, 1336, 220, 30];
  if (JSON.stringify(res.tokIds) !== JSON.stringify(expectedTok)) {
    throw new Error(`[${label}] Worker tokenizer IDs mismatch: ${JSON.stringify(res.tokIds)}`);
  }
  if (JSON.stringify(res.actualGreedy1) !== JSON.stringify(res.expectedGreedy)) {
    throw new Error(`[${label}] Run 1 greedy mismatch: ${res.actualGreedy1} vs ${res.expectedGreedy}`);
  }
  if (JSON.stringify(res.actualGreedy2) !== JSON.stringify(res.expectedGreedy)) {
    throw new Error(`[${label}] Run 2 (after reset) greedy mismatch: ${res.actualGreedy2} vs ${res.expectedGreedy}`);
  }
  if (!res.streamedMatches) {
    throw new Error(`[${label}] Streamed token events mismatch`);
  }
}

async function main() {
  const { server, port } = await startServer();
  const userDataDir = fs.mkdtempSync(path.join(os.tmpdir(), 'ignis-opfs-'));
  let context;
  try {
    context = await chromium.launchPersistentContext(userDataDir, {
      ...(EDGE_PATH ? { executablePath: EDGE_PATH } : {}),
      headless: true,
      args: [
        '--headless=new',
      ],
    });
    const page = context.pages()[0] || await context.newPage();
    const errors = [];
    page.on('console', (msg) => {
      if (msg.type() === 'error') errors.push(msg.text());
    });
    page.on('pageerror', (err) => errors.push(String(err)));

    await page.goto(`http://127.0.0.1:${port}/tests/phase3-runner.html`, { waitUntil: 'domcontentloaded' });
    const cold = await page.evaluate(() => window.__runPhase3E2E());
    console.log('=== Cold Start (OPFS Populate) ===');
    console.log(JSON.stringify(cold, null, 2));
    assertRun(cold, 'cold');
    if (cold.initInfo.cacheMisses !== 3 || cold.initInfo.cacheHits !== 0) {
      throw new Error(`Expected 3 cold cache misses, got hits=${cold.initInfo.cacheHits} misses=${cold.initInfo.cacheMisses}`);
    }

    // Reload page and verify warm OPFS cache hit
    await page.reload({ waitUntil: 'domcontentloaded' });
    const warm = await page.evaluate(() => window.__runPhase3E2E());
    console.log('=== Warm Reload (OPFS Cache Hit) ===');
    console.log(JSON.stringify(warm, null, 2));
    assertRun(warm, 'warm');
    if (warm.initInfo.cacheHits !== 3 || warm.initInfo.cacheMisses !== 0) {
      throw new Error(`Expected 3 warm cache hits, got hits=${warm.initInfo.cacheHits} misses=${warm.initInfo.cacheMisses}`);
    }

    if (errors.length > 0) throw new Error(`Browser console errors: ${errors.join(' | ')}`);
    console.log('[PASS] Phase 3 E2E passed: Worker WebGPU + OPFS cold/warm + WASM 131K Tokenizer + Chunked Prefill + Reset!');
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
