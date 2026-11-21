import { execFileSync } from 'node:child_process';
import fs from 'node:fs';
import http from 'node:http';
import os from 'node:os';
import path from 'node:path';
import { chromium } from 'playwright-core';
import { resolveBrowserExecutable } from './browser-env.mjs';

const ROOT_DIR = path.resolve(import.meta.dirname, '..');
const FIXTURES_DIR = path.join(ROOT_DIR, 'tests', 'fixtures');
const APP_DIR = path.join(ROOT_DIR, 'examples', 'chat-vinext');
const OUT_DIR = path.join(APP_DIR, 'out');
const BROWSER_PATH = resolveBrowserExecutable();

const MIME_TYPES = {
  '.html': 'text/html; charset=utf-8',
  '.js': 'application/javascript; charset=utf-8',
  '.mjs': 'application/javascript; charset=utf-8',
  '.css': 'text/css; charset=utf-8',
  '.json': 'application/json; charset=utf-8',
  '.wasm': 'application/wasm',
  '.png': 'image/png',
  '.wav': 'audio/wav',
  '.svg': 'image/svg+xml',
  '.txt': 'text/plain; charset=utf-8',
  '.ignis': 'application/octet-stream',
};

function parseCloudflareHeaders(headersFilePath) {
  const rules = [];
  if (!fs.existsSync(headersFilePath)) return rules;
  const lines = fs.readFileSync(headersFilePath, 'utf-8').split(/\r?\n/);
  let current = null;
  for (const line of lines) {
    if (!line.trim() || line.trim().startsWith('#')) continue;
    if (!/^\s/.test(line)) {
      current = { pattern: line.trim(), headers: [] };
      rules.push(current);
    } else if (current) {
      const idx = line.indexOf(':');
      if (idx !== -1) {
        current.headers.push([line.slice(0, idx).trim(), line.slice(idx + 1).trim()]);
      }
    }
  }
  return rules;
}

function matchHeaders(rules, urlPath) {
  const out = {};
  for (const r of rules) {
    let matched = false;
    if (r.pattern === '/*') {
      matched = true;
    } else if (r.pattern.endsWith('/*')) {
      const prefix = r.pattern.slice(0, -1); // e.g. '/ignis/'
      matched = urlPath.startsWith(prefix);
    } else {
      matched = urlPath === r.pattern;
    }
    if (matched) {
      for (const [k, v] of r.headers) {
        out[k] = v;
      }
    }
  }
  return out;
}

function serveFileWithRange(req, res, filePath, extraHeaders) {
  const ext = path.extname(filePath).toLowerCase();
  const contentType = MIME_TYPES[ext] || 'application/octet-stream';
  const stat = fs.statSync(filePath);
  const total = stat.size;

  const baseHeaders = {
    'Content-Type': contentType,
    'Accept-Ranges': 'bytes',
    ...extraHeaders,
  };

  const rangeHeader = req.headers.range;
  if (rangeHeader && rangeHeader.startsWith('bytes=')) {
    const [sStr, eStr] = rangeHeader.slice(6).split('-');
    const start = sStr === '' ? Math.max(0, total - Number(eStr)) : Number(sStr);
    const end = eStr === '' || sStr === '' ? total - 1 : Math.min(total - 1, Number(eStr));
    if (!Number.isFinite(start) || !Number.isFinite(end) || start > end || start < 0) {
      res.writeHead(416, { ...baseHeaders, 'Content-Range': `bytes */${total}` });
      res.end();
      return;
    }
    const len = end - start + 1;
    res.writeHead(206, {
      ...baseHeaders,
      'Content-Range': `bytes ${start}-${end}/${total}`,
      'Content-Length': String(len),
    });
    if (req.method === 'HEAD') {
      res.end();
      return;
    }
    fs.createReadStream(filePath, { start, end }).pipe(res);
    return;
  }

  res.writeHead(200, {
    ...baseHeaders,
    'Content-Length': String(total),
  });
  if (req.method === 'HEAD') {
    res.end();
    return;
  }
  fs.createReadStream(filePath).pipe(res);
}

function startStaticServer(rootDir, getExtraHeaders) {
  return new Promise((resolve, reject) => {
    const server = http.createServer((req, res) => {
      const urlPath = decodeURIComponent((req.url || '/').split('?')[0]);
      const extraHeaders = getExtraHeaders(urlPath);

      if (req.method === 'OPTIONS') {
        res.writeHead(204, extraHeaders);
        res.end();
        return;
      }

      let relPath = urlPath.replace(/^\/+/, '');
      if (!relPath || urlPath.endsWith('/')) {
        relPath = path.posix.join(relPath, 'index.html');
      }
      let fullPath = path.resolve(rootDir, relPath);
      if (!fullPath.startsWith(rootDir + path.sep) && fullPath !== rootDir) {
        res.writeHead(403, extraHeaders);
        res.end('Forbidden');
        return;
      }
      if (!fs.existsSync(fullPath) || fs.statSync(fullPath).isDirectory()) {
        res.writeHead(404, extraHeaders);
        res.end('Not found');
        return;
      }
      serveFileWithRange(req, res, fullPath, extraHeaders);
    });

    server.on('error', reject);
    server.listen(0, '127.0.0.1', () => {
      const addr = server.address();
      resolve({ server, port: addr.port, origin: `http://127.0.0.1:${addr.port}` });
    });
  });
}

async function main() {
  // 1. Start Server B (Cross-Origin Cloudflare R2 Simulator) serving tests/fixtures/
  const r2Headers = {
    'Access-Control-Allow-Origin': '*',
    'Access-Control-Allow-Methods': 'GET, HEAD, OPTIONS',
    'Access-Control-Allow-Headers': 'Range, If-None-Match, Content-Type',
    'Access-Control-Expose-Headers': 'Content-Length, Content-Range, ETag, Accept-Ranges',
    'Cross-Origin-Resource-Policy': 'cross-origin',
  };
  const r2Srv = await startStaticServer(FIXTURES_DIR, () => r2Headers);

  // 2. Build Cloudflare Pages static export pointing NEXT_PUBLIC_IGNIS_CDN_BASE to Server B
  console.log(`[test-demo-static] Building static export with CDN=${r2Srv.origin}`);
  execFileSync(process.execPath, [path.join(APP_DIR, 'scripts', 'build-pages.mjs')], {
    cwd: APP_DIR,
    stdio: 'inherit',
    env: {
      ...process.env,
      IGNIS_STATIC_EXPORT: '1',
      NEXT_PUBLIC_IGNIS_CDN_BASE: r2Srv.origin,
    },
  });

  const headerRules = parseCloudflareHeaders(path.join(OUT_DIR, '_headers'));
  if (headerRules.length === 0) {
    throw new Error('Parsed zero rules from out/_headers');
  }

  // 3. Start Server A (Cloudflare Pages Simulator) serving out/ with exact _headers rules
  const pagesSrv = await startStaticServer(OUT_DIR, (urlPath) => matchHeaders(headerRules, urlPath));

  const userDataDir = fs.mkdtempSync(path.join(os.tmpdir(), 'ignis-static-e2e-'));
  let context;
  const errors = [];

  try {
    console.log(`[test-demo-static] Pages=${pagesSrv.origin} | R2=${r2Srv.origin} | Browser=${BROWSER_PATH || 'playwright-chromium'}`);
    context = await chromium.launchPersistentContext(userDataDir, {
      ...(BROWSER_PATH ? { executablePath: BROWSER_PATH } : {}),
      headless: true,
      args: ['--headless=new'],
    });

    const page = context.pages()[0] || await context.newPage();
    page.on('console', (m) => {
      if (m.type() === 'error') errors.push(`${m.text()} (${m.location()?.url || ''})`);
    });
    page.on('pageerror', (e) => errors.push(String(e)));

    await page.goto(pagesSrv.origin, { waitUntil: 'networkidle' });

    // Assert crossOriginIsolated + COOP/COEP + Hardware Readiness card + Show HN Hero
    const isolated = await page.evaluate(() => window.crossOriginIsolated === true);
    if (!isolated) {
      throw new Error('Expected window.crossOriginIsolated === true under COOP: same-origin + COEP: credentialless');
    }

    const heroVisible = await page.locator('#hero-banner').isVisible();
    const hwText = await page.locator('#hardware-card').innerText();
    if (!heroVisible || !hwText.includes('tier') || hwText.includes('WebGPU Unavailable')) {
      throw new Error(`Hero/Hardware check failed: hero=${heroVisible}, hwText="${hwText}"`);
    }

    const selectedModel = await page.locator('select').first().inputValue();
    if (selectedModel !== 'lfm3-omni-2b') {
      throw new Error(`Expected default CDN preset 'lfm3-omni-2b', got '${selectedModel}'`);
    }

    // 4. Load LFM3-Omni-2B cross-origin from Server B into OPFS + WebGPU
    await page.click('#btn-load-model');
    await page.waitForFunction(
      () => document.getElementById('status-text')?.textContent?.includes('Ready · INT8_G64 LM Head · Full Omni'),
      null,
      { timeout: 120000 },
    );
    console.log('[status]', await page.locator('#status-text').innerText());

    // 5. Showcase 1: 1-Click Omni VQA (demo-shapes.png)
    await page.click('#showcase-vqa');
    await page.waitForFunction(
      () =>
        document.querySelectorAll('.msg.assistant').length >= 1 &&
        document.getElementById('status-text')?.textContent?.includes('Ready · INT8_G64 LM Head · Full Omni'),
      null,
      { timeout: 90000 },
    );
    const vqaAnswer = await page.locator('.msg.assistant .answer-body').last().innerText();
    console.log('[showcase-vqa]', vqaAnswer.trim());
    if (!/red|blue|square/i.test(vqaAnswer)) {
      throw new Error(`Expected Omni VQA answer to describe red/blue/square, got: "${vqaAnswer}"`);
    }

    // 6. Showcase 2: 1-Click JSON Schema FSM
    await page.click('#showcase-fsm');
    await page.waitForFunction(
      () =>
        document.querySelectorAll('.msg.assistant').length >= 2 &&
        document.getElementById('status-text')?.textContent?.includes('Ready · INT8_G64 LM Head · Full Omni'),
      null,
      { timeout: 90000 },
    );
    const fsmAnswer = await page.locator('.msg.assistant .answer-body').last().innerText();
    const fsmBadge = await page.locator('[data-testid="fsm-badge"]').last().innerText();
    console.log('[showcase-fsm]', fsmAnswer.trim(), '|', fsmBadge);
    const parsedJson = JSON.parse(fsmAnswer);
    if (typeof parsedJson.engine !== 'string' || typeof parsedJson.storage !== 'string' || typeof parsedJson.inBrowser !== 'boolean') {
      throw new Error(`FSM JSON missing expected typed properties: ${fsmAnswer}`);
    }
    if (!fsmBadge.includes('JSON.parse ✓') || !fsmBadge.includes('ms/tok')) {
      throw new Error(`Unexpected FSM badge text: "${fsmBadge}"`);
    }

    // 7. Showcase 3: 1-Click N-gram Speculative
    await page.click('#showcase-ngram');
    await page.waitForFunction(
      () =>
        document.querySelectorAll('.msg.assistant').length >= 3 &&
        document.getElementById('status-text')?.textContent?.includes('Ready · INT8_G64 LM Head · Full Omni'),
      null,
      { timeout: 90000 },
    );
    const ngramAnswer = await page.locator('.msg.assistant .answer-body').last().innerText();
    const ngramBadge = await page.locator('[data-testid="ngram-badge"]').last().innerText();
    console.log('[showcase-ngram]', ngramAnswer.trim().slice(0, 120), '|', ngramBadge);
    if (!ngramBadge.includes('accepted')) {
      throw new Error(`Unexpected N-gram badge text: "${ngramBadge}"`);
    }

    // 8. Clear OPFS Cache
    await page.click('#btn-clear-opfs');
    await page.waitForFunction(
      () => document.getElementById('status-text')?.textContent?.includes('OPFS cache cleared'),
      null,
      { timeout: 15000 },
    );

    if (errors.length > 0) {
      throw new Error(`Console/page errors detected: ${errors.join(' | ')}`);
    }

    console.log('[PASS] Static Cloudflare Pages + Cross-Origin R2 Showcase E2E verified (COEP credentialless + LFM3 VQA + JSON FSM + N-gram + Clear OPFS)!');
  } finally {
    if (context) await context.close();
    pagesSrv.server.closeAllConnections?.();
    pagesSrv.server.close();
    r2Srv.server.closeAllConnections?.();
    r2Srv.server.close();
    fs.rmSync(userDataDir, { recursive: true, force: true });
  }
}

main().catch((err) => {
  console.error(err);
  process.exit(1);
});
