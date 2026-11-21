import http from 'node:http';
import fs from 'node:fs';
import path from 'node:path';
import { chromium } from 'playwright-core';
import { resolveBrowserExecutable } from './browser-env.mjs';

const EDGE_PATH = resolveBrowserExecutable();
const TESTS_DIR = path.resolve(import.meta.dirname);

function startServer() {
  return new Promise((resolve) => {
    const server = http.createServer((req, res) => {
      const reqPath = (req.url || '/').split('?')[0];
      if (reqPath === '/favicon.ico') {
        res.writeHead(204);
        res.end();
        return;
      }
      const rel = reqPath === '/' ? 'webgpu-probe.html' : reqPath.replace(/^\/+/, '');
      const filePath = path.join(TESTS_DIR, rel);
      if (!filePath.startsWith(TESTS_DIR) || !fs.existsSync(filePath)) {
        res.writeHead(404);
        res.end('Not Found');
        return;
      }
      const ext = path.extname(filePath);
      const mime = ext === '.html' ? 'text/html; charset=utf-8'
        : ext === '.js' || ext === '.mjs' ? 'application/javascript; charset=utf-8'
        : 'application/octet-stream';
      res.writeHead(200, {
        'Content-Type': mime,
        'Cross-Origin-Opener-Policy': 'same-origin',
        'Cross-Origin-Embedder-Policy': 'require-corp',
      });
      fs.createReadStream(filePath).pipe(res);
    });
    server.listen(0, '127.0.0.1', () => {
      const addr = server.address();
      resolve({ server, port: addr.port });
    });
  });
}

async function main() {
  const { server, port } = await startServer();
  let browser;
  try {
    browser = await chromium.launch({
      ...(EDGE_PATH ? { executablePath: EDGE_PATH } : {}),
      headless: true,
      args: [
        '--headless=new',
      ],
    });
    const context = await browser.newContext();
    const page = await context.newPage();
    page.on('console', (msg) => {
      if (msg.type() === 'error') console.error('[browser console error]', msg.text());
    });
    page.on('pageerror', (err) => console.error('[browser pageerror]', err));

    await page.goto(`http://127.0.0.1:${port}/webgpu-probe.html`, { waitUntil: 'domcontentloaded' });
    const result = await page.evaluate(() => window.__runIgnisProbe());
    console.log(JSON.stringify(result, null, 2));

    if (!result.gpu?.supported) {
      throw new Error(`WebGPU not supported: ${result.gpu?.reason}`);
    }
    if (result.gpu?.shaderSmoke?.shaderError) {
      throw new Error(`Shader smoke failed: ${result.gpu.shaderSmoke.shaderError}`);
    }
    if (!result.sandbox?.opfs?.ok) {
      throw new Error(`OPFS failed: ${result.sandbox?.opfs?.error}`);
    }
    if (!result.sandbox?.workerWebGPU?.hasAdapter) {
      throw new Error(`Worker WebGPU failed: ${JSON.stringify(result.sandbox?.workerWebGPU)}`);
    }
  } finally {
    if (browser) await browser.close();
    server.close();
  }
}

main().catch((err) => {
  console.error(err);
  process.exit(1);
});
