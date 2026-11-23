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
      if (m.type() === 'error') {
        const text = m.text();
        const url = m.location()?.url || '';
        const isWebContainerHmrNoise =
          url.includes('.webcontainer-api.io') &&
          (text.includes('504 (Outdated Request)') ||
            text.includes('failed to connect to websocket') ||
            text.includes('Error during WebSocket handshake'));
        if (!isWebContainerHmrNoise) {
          errors.push(`${text} (${url})`);
        }
      }
    });
    page.on('pageerror', (e) => errors.push(String(e)));

    await page.goto(pagesSrv.origin, { waitUntil: 'domcontentloaded' });
    await page.waitForSelector('#model-select', { timeout: 20000 });

    // Assert crossOriginIsolated + COOP/COEP + clean Agent Hero + Hardware Readiness card
    const isolated = await page.evaluate(() => window.crossOriginIsolated === true);
    if (!isolated) {
      throw new Error('Expected window.crossOriginIsolated === true under COOP: same-origin + COEP: credentialless');
    }

    const brandWordmark = (await page.locator('.brand-wordmark').innerText()).trim();
    if (brandWordmark !== 'Ignis') {
      throw new Error(`Expected brand wordmark 'Ignis', got '${brandWordmark}'`);
    }

    const heroVisible = await page.locator('#hero-banner').isVisible();
    const heroText = heroVisible ? await page.locator('#hero-banner').innerText() : '';
    if (
      !heroVisible ||
      !heroText.includes('Ignis: Browser-Native LLM Inference Engine') ||
      /256B|ONNX|SyncAccessHandle|Claude Code/i.test(heroText)
    ) {
      throw new Error(`Hero check failed: visible=${heroVisible}, text="${heroText}"`);
    }
    if ((await page.locator('.empty-sub').count()) > 0) {
      throw new Error('Expected empty heroSubtitle to omit .empty-sub');
    }

    // Verify Dark <-> Light mode toggle works cleanly
    await page.click('#btn-theme-toggle');
    const lightThemeAttr = await page.evaluate(() => document.documentElement.getAttribute('data-theme'));
    if (lightThemeAttr !== 'light') {
      throw new Error(`Expected light theme after click, got '${lightThemeAttr}'`);
    }
    await page.click('#btn-theme-toggle');
    const darkThemeAttr = await page.evaluate(() => document.documentElement.getAttribute('data-theme'));
    if (darkThemeAttr !== 'dark') {
      throw new Error(`Expected dark theme restored after toggle, got '${darkThemeAttr}'`);
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
    const cdnOptions = await page.locator('#model-select option').evaluateAll((opts) =>
      opts.map((o) => o.value),
    );
    if (!cdnOptions.includes('spark-1.7b-q8head')) {
      throw new Error(`Expected CDN model list to include 'spark-1.7b-q8head', got ${JSON.stringify(cdnOptions)}`);
    }
    if (cdnOptions.includes('gemma-4-e4b')) {
      throw new Error(`Expected 'gemma-4-e4b' to be removed from CDN model selector, got ${JSON.stringify(cdnOptions)}`);
    }

    // 4. Load default Spark-X2.6 4B cross-origin from Server B into OPFS + WebGPU
    await page.click('#btn-load-model');
    await page.waitForFunction(
      () => document.getElementById('status-text')?.textContent?.includes('Ready · Spark'),
      null,
      { timeout: 120000 },
    );
    console.log('[status]', await page.locator('#status-text').innerText());

    // Wait for WebContainer Vite + React default starter server to boot & render live preview (diagnostic logging every 15s)
    const previewDeadline = Date.now() + 240000;
    let lastOverlayLog = 0;
    while (true) {
      if (await page.locator('#preview-iframe').isVisible().catch(() => false)) break;
      const errBadge = await page.locator('#sandbox-boot-overlay .feature-badge.err').innerText().catch(() => '');
      if (errBadge) {
        throw new Error(`WebContainer boot error: ${errBadge}`);
      }
      if (Date.now() - lastOverlayLog >= 15000) {
        const overlayTxt = await page.locator('#sandbox-boot-overlay').innerText().catch(() => '');
        console.log('[webcontainer-booting]', overlayTxt.replace(/\s+/g, ' ').trim());
        lastOverlayLog = Date.now();
      }
      if (Date.now() >= previewDeadline) {
        const overlayTxt = await page.locator('#sandbox-boot-overlay').innerText().catch(() => '');
        throw new Error(`Timed out after 240000ms waiting for #preview-iframe (overlay="${overlayTxt.replace(/\s+/g, ' ').trim()}")`);
      }
      await page.waitForTimeout(1000);
    }
    const previewFrame = page.frameLocator('#preview-iframe');
    await previewFrame.locator('h1').waitFor({ timeout: 30000 });
    const starterTitle = (await previewFrame.locator('h1').innerText()).trim();
    console.log('[webcontainer-preview starter]', starterTitle);
    if (!starterTitle.includes('Vite + React')) {
      throw new Error(`Expected Vite + React inside #preview-iframe, got "${starterTitle}"`);
    }

    // 5. Ignis Agent Harness E2E (FSM tool calls + Vite React Kanban app building + HMR live preview)
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
    console.log('[agent-harness] tools=', toolCount, toolSummaries.map((s) => s.replace(/\s+/g, ' ')));
    console.log('[agent-answer]', agentAnswer.trim().slice(0, 220));
    if (toolCount < 2) {
      throw new Error(`Expected at least 2 real FSM tool calls in Agent app-building run, got ${toolCount}`);
    }
    if (agentAnswer.trim().length < 15) {
      throw new Error(`Expected substantive Agent answer, got "${agentAnswer}"`);
    }

    // Assert live preview reflects updated app content with zero Vite error overlays
    const previewDeadline2 = Date.now() + 25000;
    let builtPreviewBody = '';
    while (Date.now() < previewDeadline2) {
      builtPreviewBody = (await previewFrame.locator('body').innerText().catch(() => '')).trim();
      if (builtPreviewBody.length > 0 && !builtPreviewBody.includes('Edit src/App.jsx and save to test HMR')) {
        break;
      }
      await page.waitForTimeout(400);
    }
    const viteErrs = await previewFrame.locator('vite-error-overlay').count();
    console.log('[webcontainer-preview after build]', builtPreviewBody.replace(/\s+/g, ' ').slice(0, 140), 'viteErrs=', viteErrs);
    if (viteErrs > 0 || !builtPreviewBody || builtPreviewBody.includes('Edit src/App.jsx and save to test HMR')) {
      const overlayText = await previewFrame.locator('vite-error-overlay').innerText().catch(() => '');
      const toolPreTexts = await page.locator('.tool-call-pre').allInnerTexts().catch(() => []);
      console.error('[diagnostic vite-error-overlay]', overlayText);
      console.error('[diagnostic tool outputs]', toolPreTexts.map((x) => x.slice(0, 500)));
    }
    if (!builtPreviewBody || builtPreviewBody.includes('Edit src/App.jsx and save to test HMR')) {
      throw new Error(`Expected Agent to replace default Vite starter inside #preview-iframe, got "${builtPreviewBody}"`);
    }
    if (viteErrs > 0) {
      throw new Error('Detected Vite error overlay inside #preview-iframe after Agent build');
    }

    await page.click('#tab-btn-trace');
    const traceCount = await page.locator('[data-testid="trace-entry"]').count();
    console.log('[trace] entries=', traceCount);
    if (traceCount < 3) {
      throw new Error(`Expected >= 3 trace entries after Agent run, got ${traceCount}`);
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

    // 8. Switch to Spark-X2.6 1.7B, reset workspace, and verify Pi Agent Markdown Notes build
    await page.selectOption('#model-select', 'spark-1.7b-q8head');
    await page.click('#btn-load-model');
    await page.waitForFunction(
      () => document.getElementById('status-text')?.textContent?.includes('Ready · Spark-X2.6 1.7B'),
      null,
      { timeout: 120000 },
    );
    await page.click('#btn-reset-project');
    await page.click('#tab-btn-preview');
    await page.waitForTimeout(300);
    await page.click('#btn-mode-agent');
    await page.fill('#chat-input', 'Build a Markdown notes app in src/App.jsx with a note list, textarea editor, and live preview.');
    await page.click('#btn-send');
    await page.waitForFunction(
      () =>
        document.querySelectorAll('.msg.assistant').length >= 4 &&
        document.getElementById('status-text')?.textContent?.includes('Ready · Spark-X2.6 1.7B'),
      null,
      { timeout: 180000 },
    );
    const spark17Tools = await page.locator('.msg.assistant').last().locator('.tool-call-summary').allInnerTexts();
    const spark17Out = await page.locator('.msg.assistant .answer-body').last().innerText();
    console.log('[spark-1.7b agent] tools=', spark17Tools.map((s) => s.replace(/\s+/g, ' ')), '| answer=', spark17Out.trim().slice(0, 140));
    if (spark17Tools.length < 1 || !spark17Tools.some((s) => /Write|Edit/i.test(s))) {
      await page.click('#tab-btn-trace').catch(() => {});
      const spark17Traces = await page.locator('[data-testid="trace-entry"]').allInnerTexts().catch(() => []);
      console.error('[spark-1.7b trace]', spark17Traces.map((s) => s.replace(/\s+/g, ' ')));
      throw new Error(`Expected Spark-1.7B Pi Agent to execute Write/Edit tool call, got ${JSON.stringify(spark17Tools)}`);
    }
    if (spark17Out.trim().length < 5 || spark17Out.includes('\uFFFD')) {
      throw new Error(`Unexpected Spark-1.7B Agent output: "${spark17Out}"`);
    }
    const mdDeadline = Date.now() + 20000;
    let mdPreviewBody = '';
    while (Date.now() < mdDeadline) {
      mdPreviewBody = (await previewFrame.locator('body').innerText().catch(() => '')).trim();
      if (mdPreviewBody.length > 0 && !mdPreviewBody.includes('Edit src/App.jsx and save to test HMR')) {
        break;
      }
      await page.waitForTimeout(400);
    }
    const mdViteErrs = await previewFrame.locator('vite-error-overlay').count();
    console.log('[spark-1.7b markdown preview]', mdPreviewBody.replace(/\s+/g, ' ').slice(0, 140), 'viteErrs=', mdViteErrs);
    if (!mdPreviewBody || mdPreviewBody.includes('Edit src/App.jsx and save to test HMR') || mdViteErrs > 0) {
      throw new Error(`Expected Spark-1.7B Markdown notes preview without Vite errors, got "${mdPreviewBody}" (viteErrs=${mdViteErrs})`);
    }

    // 9. Switch to LFM3-Omni 2B for Multimodal Vision VQA (demo-shapes.png)
    await page.click('#tab-btn-runtime');
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
        document.querySelectorAll('.msg.assistant').length >= 5 &&
        document.getElementById('status-text')?.textContent?.includes('Ready · LFM3'),
      null,
      { timeout: 90000 },
    );
    const vqaAnswer = await page.locator('.msg.assistant .answer-body').last().innerText();
    console.log('[showcase-vqa]', vqaAnswer.trim());
    if (!/red|blue|square/i.test(vqaAnswer)) {
      throw new Error(`Expected Omni VQA answer to describe red/blue/square, got: "${vqaAnswer}"`);
    }

    // 10. Clear OPFS Cache
    await page.click('#btn-clear-opfs');
    await page.waitForFunction(
      () => document.getElementById('status-text')?.textContent?.includes('OPFS cache cleared'),
      null,
      { timeout: 15000 },
    );

    if (errors.length > 0) {
      throw new Error(`Console/page errors detected: ${errors.join(' | ')}`);
    }

    console.log('[PASS] Static Cloudflare Pages + Cross-Origin R2 Ignis E2E verified (COEP credentialless + Vite Starter + Light Theme + Spark-4B Kanban Build + JSON FSM + N-gram + Spark-1.7B + LFM3 VQA + Clear OPFS)!');
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
