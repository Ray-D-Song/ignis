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
  '.woff2': 'font/woff2',
  '.woff': 'font/woff',
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
      viewport: { width: 1440, height: 900 },
      args: ['--headless=new'],
    });

    const page = context.pages()[0] || await context.newPage();
    page.on('console', (m) => {
      if (m.type() === 'error') errors.push(`${m.text()} (${m.location()?.url || ''})`);
    });
    page.on('pageerror', (e) => errors.push(String(e)));

    await page.goto(pagesSrv.origin, { waitUntil: 'domcontentloaded' });
    await page.waitForSelector('#model-select', { timeout: 20000 });

    // Assert crossOriginIsolated + COOP/COEP + clean Agent Hero + Hardware Readiness card
    const isolated = await page.evaluate(() => window.crossOriginIsolated === true);
    if (!isolated) {
      throw new Error('Expected window.crossOriginIsolated === true under COOP: same-origin + COEP: credentialless');
    }

    const heroVisible = await page.locator('#hero-banner').isVisible();
    const heroText = heroVisible ? await page.locator('#hero-banner').innerText() : '';
    if (!heroVisible || /256B|ONNX|SyncAccessHandle/i.test(heroText)) {
      throw new Error(`Hero check failed: visible=${heroVisible}, text="${heroText}"`);
    }

    if (!(await page.locator('#workspace-pane').isVisible())) {
      await page.click('#btn-toggle-workspace');
    }
    await page.click('#tab-btn-runtime');
    await page.waitForFunction(
      () => {
        const txt = document.getElementById('hardware-card')?.textContent || '';
        return txt.includes('tier') && txt.includes('COOP/COEP');
      },
      null,
      { timeout: 20000 },
    );
    const hwText = await page.locator('#hardware-card').innerText();
    if (!hwText.includes('tier') || !hwText.includes('f16:✓') || !hwText.includes('sg:✓') || hwText.includes('WebGPU Unavailable')) {
      throw new Error(`Hardware check failed: hwText="${hwText}"`);
    }
    await page.click('#tab-btn-preview');

    const selectedModel = await page.locator('#model-select').inputValue();
    if (selectedModel !== 'spark-4b-q8head') {
      throw new Error(`Expected default CDN preset 'spark-4b-q8head', got '${selectedModel}'`);
    }

    // 4. Load default Spark-X2.6 4B cross-origin from Server B into OPFS + WebGPU
    await page.click('#btn-load-model');
    await page.waitForFunction(
      () => document.getElementById('status-text')?.textContent?.includes('Ready · Spark'),
      null,
      { timeout: 120000 },
    );
    console.log('[status]', await page.locator('#status-text').innerText());

    // Wait for WebContainer PulseBoard full-stack server to boot & render live preview
    await page.waitForSelector('#preview-iframe', { timeout: 120000 });
    const previewFrame = page.frameLocator('#preview-iframe');
    await previewFrame.locator('header h1').waitFor({ timeout: 30000 });
    const preFixTitle = await previewFrame.locator('header h1').innerText();
    const preFixRate = await previewFrame.locator('.kpi-card strong.accent').innerText();
    console.log('[webcontainer-preview before fix]', preFixTitle, 'completionRate=', preFixRate);
    if (!preFixTitle.includes('PulseBoard')) {
      throw new Error(`Expected PulseBoard inside #preview-iframe, got "${preFixTitle}"`);
    }

    // 5. Ignis Code Agent Harness E2E (FSM tool calls + WebContainer npm test + multi-file bugfix)
    await page.locator('.starter-card').first().click();
    await page.waitForFunction(
      () =>
        document.querySelectorAll('.msg.assistant').length >= 1 &&
        document.getElementById('status-text')?.textContent?.includes('Ready · Spark'),
      null,
      { timeout: 240000 },
    );

    const toolCount = await page.locator('.tool-call-card').count();
    const toolSummaries = await page.locator('.tool-call-summary').allInnerTexts();
    const agentAnswer = await page.locator('.msg.assistant .answer-body').last().innerText();
    const verifierBadge = await page.locator('[data-testid="test-verifier-badge"]').last().innerText();
    console.log('[agent-harness] tools=', toolCount, 'badge=', verifierBadge, toolSummaries.map((s) => s.replace(/\s+/g, ' ')));
    console.log('[agent-answer]', agentAnswer.trim().slice(0, 220));
    if (toolCount < 3) {
      throw new Error(`Expected at least 3 real FSM tool calls in Agent bugfix run, got ${toolCount}`);
    }
    if (!verifierBadge.includes('6/6 PASS')) {
      throw new Error(`Expected 6/6 PASS from real WebContainer npm test verifier, got "${verifierBadge}"`);
    }
    if (agentAnswer.trim().length < 15) {
      throw new Error(`Expected substantive Agent answer, got "${agentAnswer}"`);
    }

    // Assert live PulseBoard preview reflects the fixed stats calculation (50% -> 25%)
    await page.waitForFunction(
      () => {
        const ifr = document.getElementById('preview-iframe');
        const txt = ifr?.contentDocument?.querySelector('.kpi-card strong.accent')?.textContent || '';
        return txt.includes('25%');
      },
      null,
      { timeout: 20000 },
    ).catch(async () => {
      await previewFrame.locator('.kpi-card strong.accent', { hasText: '25%' }).waitFor({ timeout: 20000 });
    });
    const postFixRate = await previewFrame.locator('.kpi-card strong.accent').innerText();
    console.log('[webcontainer-preview after fix] completionRate=', postFixRate);
    if (!postFixRate.includes('25%')) {
      throw new Error(`Expected updated PulseBoard KPI 25% inside #preview-iframe, got "${postFixRate}"`);
    }

    await page.click('#tab-btn-trace');
    const traceCount = await page.locator('[data-testid="trace-entry"]').count();
    console.log('[trace] entries=', traceCount);
    if (traceCount < 4) {
      throw new Error(`Expected >= 4 trace entries after Agent bugfix run, got ${traceCount}`);
    }

    // 6. Showcase 1: Direct JSON Schema FSM on Spark-4B
    await page.click('#tab-btn-runtime');
    await page.click('#showcase-fsm');
    await page.waitForFunction(
      () =>
        document.querySelectorAll('.msg.assistant').length >= 2 &&
        document.getElementById('status-text')?.textContent?.includes('Ready · Spark'),
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

    // 7. Showcase 2: Direct N-gram Speculative on Spark-4B
    await page.click('#showcase-ngram');
    await page.waitForFunction(
      () =>
        document.querySelectorAll('.msg.assistant').length >= 3 &&
        document.getElementById('status-text')?.textContent?.includes('Ready · Spark'),
      null,
      { timeout: 90000 },
    );
    const ngramAnswer = await page.locator('.msg.assistant .answer-body').last().innerText();
    const ngramBadge = await page.locator('[data-testid="ngram-badge"]').last().innerText();
    console.log('[showcase-ngram]', ngramAnswer.trim().slice(0, 120), '|', ngramBadge);
    if (!ngramBadge.includes('accepted')) {
      throw new Error(`Unexpected N-gram badge text: "${ngramBadge}"`);
    }

    // 8. Switch to LFM3-Omni 2B for Multimodal Vision VQA (demo-shapes.png)
    await page.selectOption('#model-select', 'lfm3-omni-2b');
    await page.click('#btn-load-model');
    await page.waitForFunction(
      () => document.getElementById('status-text')?.textContent?.includes('Ready · LFM3'),
      null,
      { timeout: 120000 },
    );
    await page.click('#showcase-vqa');
    await page.waitForFunction(
      () =>
        document.querySelectorAll('.msg.assistant').length >= 4 &&
        document.getElementById('status-text')?.textContent?.includes('Ready · LFM3'),
      null,
      { timeout: 90000 },
    );
    const vqaAnswer = await page.locator('.msg.assistant .answer-body').last().innerText();
    console.log('[showcase-vqa]', vqaAnswer.trim());
    if (!/red|blue|square/i.test(vqaAnswer)) {
      throw new Error(`Expected Omni VQA answer to describe red/blue/square, got: "${vqaAnswer}"`);
    }

    // 9. Clear OPFS Cache
    await page.click('#btn-clear-opfs');
    await page.waitForFunction(
      () => document.getElementById('status-text')?.textContent?.includes('OPFS cache cleared'),
      null,
      { timeout: 15000 },
    );

    if (errors.length > 0) {
      throw new Error(`Console/page errors detected: ${errors.join(' | ')}`);
    }

    console.log('[PASS] Static Cloudflare Pages + Cross-Origin R2 Ignis Code E2E verified (COEP credentialless + WebContainer PulseBoard + Spark-4B 6/6 Bugfix + JSON FSM + N-gram + LFM3 VQA + Clear OPFS)!');
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
