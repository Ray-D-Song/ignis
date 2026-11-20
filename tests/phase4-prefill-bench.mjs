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
      if (reqPath === '/favicon.ico') { res.writeHead(204); res.end(); return; }
      const filePath = path.join(ROOT_DIR, reqPath.replace(/^\/+/, ''));
      if (!filePath.startsWith(ROOT_DIR) || !fs.existsSync(filePath) || fs.statSync(filePath).isDirectory()) {
        res.writeHead(404); res.end('Not Found'); return;
      }
      const ext = path.extname(filePath);
      const mime = ext === '.html' ? 'text/html' : ext === '.js' || ext === '.mjs' ? 'application/javascript' : 'text/plain';
      res.writeHead(200, { 'Content-Type': mime });
      fs.createReadStream(filePath).pipe(res);
    });
    server.listen(0, '127.0.0.1', () => resolve({ server, port: server.address().port }));
  });
}

async function main() {
  const { server, port } = await startServer();
  const browser = await chromium.launch({
    executablePath: EDGE_PATH,
    headless: true,
    args: ['--headless=new', '--enable-unsafe-webgpu', '--use-angle=d3d11', '--enable-features=WebGPUDeveloperFeatures,Vulkan', '--ignore-gpu-blocklist'],
  });
  try {
    const page = await browser.newPage();
    page.on('console', (m) => { if (m.type() === 'error') console.error('[console]', m.text()); });
    page.on('pageerror', (e) => console.error('[pageerror]', e));
    await page.goto(`http://127.0.0.1:${port}/tests/phase4-prefill-bench.html`);
    const res = await page.evaluate(() => window.__runPhase4Prefill());
    console.log(JSON.stringify(res, null, 2));
    if (res.mainThreadMaxFrameMs > 50) {
      throw new Error(`Main thread frame drop detected: ${res.mainThreadMaxFrameMs} ms`);
    }
    console.log(`[PASS] 1024-token 36-layer Chunked Prefill: ${res.prefillTokPerSec.toFixed(1)} tok/s, maxFrameMs=${res.mainThreadMaxFrameMs}ms`);
  } finally {
    await browser.close();
    server.close();
  }
}

main().catch((err) => { console.error(err); process.exit(1); });
