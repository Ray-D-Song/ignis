import http from 'node:http';
import fs from 'node:fs';
import path from 'node:path';
import { chromium } from 'playwright-core';

const EDGE_PATH = 'C:\\Program Files (x86)\\Microsoft\\Edge\\Application\\msedge.exe';
const ROOT_DIR = path.resolve(import.meta.dirname, '..');

function startServer() {
  return new Promise((resolve) => {
    const server = http.createServer((req, res) => {
      const reqPath = (req.url || '/').split('?')[0];
      if (reqPath === '/favicon.ico') {
        res.writeHead(204);
        res.end();
        return;
      }
      const rel = reqPath.replace(/^\/+/, '');
      const filePath = path.join(ROOT_DIR, rel);
      if (!filePath.startsWith(ROOT_DIR) || !fs.existsSync(filePath) || fs.statSync(filePath).isDirectory()) {
        res.writeHead(404);
        res.end('Not Found');
        return;
      }
      const ext = path.extname(filePath);
      const mime = ext === '.html' ? 'text/html; charset=utf-8'
        : (ext === '.js' || ext === '.mjs') ? 'application/javascript; charset=utf-8'
        : ext === '.json' ? 'application/json; charset=utf-8'
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
  const { server, port } = await startServer();
  let browser;
  try {
    browser = await chromium.launch({
      executablePath: EDGE_PATH,
      headless: true,
      args: [
        '--headless=new',
        '--enable-unsafe-webgpu',
        '--use-angle=d3d11',
        '--enable-features=WebGPUDeveloperFeatures,Vulkan',
        '--ignore-gpu-blocklist',
      ],
    });
    const page = await browser.newPage();
    const errors = [];
    page.on('console', (msg) => {
      if (msg.type() === 'error') errors.push(msg.text());
    });
    page.on('pageerror', (err) => errors.push(String(err)));

    await page.goto(`http://127.0.0.1:${port}/tests/phase2-runner.html`, { waitUntil: 'domcontentloaded' });
    const result = await page.evaluate(() => window.__runPhase2Test());
    console.log(JSON.stringify(result, null, 2));

    if (errors.length > 0) throw new Error(`Browser errors: ${errors.join(' | ')}`);
    const minCos = Math.min(
      result.prefillCos.embed,
      result.prefillCos.layer0,
      result.prefillCos.layer3,
      result.prefillCos.finalNorm,
      ...result.decodeCos.flatMap((d) => [d.layer0, d.layer3, d.finalNorm])
    );
    if (minCos < 0.999) {
      throw new Error(`Numerical divergence detected! minCos=${minCos}`);
    }
    if (!result.greedyExactMatch) {
      throw new Error(`Greedy tokens mismatch! expected=${result.expectedGreedy} actual=${result.actualGreedy}`);
    }
    console.log(`[PASS] Phase 2 numerical alignment verified: minCos=${minCos.toFixed(6)}, greedy exact match!`);
  } finally {
    if (browser) await browser.close();
    server.close();
  }
}

main().catch((err) => {
  console.error(err);
  process.exit(1);
});
