import { spawn } from 'node:child_process';
import fs from 'node:fs';
import net from 'node:net';
import os from 'node:os';
import path from 'node:path';
import { chromium } from 'playwright-core';

const THORIUM_PATH = 'C:\\Users\\ray\\AppData\\Local\\Thorium\\Application\\thorium.exe';
const EDGE_PATH = 'C:\\Program Files (x86)\\Microsoft\\Edge\\Application\\msedge.exe';
const BROWSER_PATH = fs.existsSync(THORIUM_PATH) ? THORIUM_PATH : EDGE_PATH;
const ROOT_DIR = path.resolve(import.meta.dirname, '..');
const APP_DIR = path.join(ROOT_DIR, 'examples', 'chat-vinext');

function getFreePort() {
  return new Promise((resolve, reject) => {
    const srv = net.createServer();
    srv.listen(0, '127.0.0.1', () => {
      const port = srv.address().port;
      srv.close(() => resolve(port));
    });
    srv.on('error', reject);
  });
}

async function waitForServer(url, timeoutMs = 30000) {
  const start = Date.now();
  while (Date.now() - start < timeoutMs) {
    try {
      const r = await fetch(url);
      if (r.ok) return;
    } catch {
      // retry
    }
    await new Promise((r) => setTimeout(r, 300));
  }
  throw new Error(`vinext dev server did not become ready within ${timeoutMs}ms`);
}

async function main() {
  // Sync ignis runtime/kernels first
  const vinextCli = path.join(APP_DIR, 'node_modules', 'vinext', 'dist', 'cli.js');
  const syncBin = path.join(APP_DIR, 'scripts', 'sync-ignis.mjs');
  await new Promise((resolve, reject) => {
    const p = spawn('node', [syncBin], { cwd: APP_DIR, stdio: 'inherit' });
    p.on('exit', (c) => (c === 0 ? resolve() : reject(new Error(`sync failed: ${c}`))));
  });

  // Reuse an already-running vinext dev server if present (e.g. user running npm run demo:chat)
  let baseUrl = null;
  let serverProc = null;
  for (const cand of ['http://localhost:5173', 'http://127.0.0.1:5173']) {
    try {
      const r = await fetch(cand);
      if (r.ok) { baseUrl = cand; break; }
    } catch { /* not running */ }
  }

  let serverLogs = '';
  if (!baseUrl) {
    const PORT = await getFreePort();
    baseUrl = `http://127.0.0.1:${PORT}`;
    serverProc = spawn('node', [vinextCli, 'dev', '--host', '127.0.0.1', '--port', String(PORT)], {
      cwd: APP_DIR,
      stdio: ['ignore', 'pipe', 'pipe'],
    });
    serverProc.stdout.on('data', (d) => { serverLogs += d.toString(); });
    serverProc.stderr.on('data', (d) => { serverLogs += d.toString(); });
  }

  const userDataDir = fs.mkdtempSync(path.join(os.tmpdir(), 'ignis-vinext-e2e-'));
  let context;
  const errors = [];
  let failed = false;

  try {
    await waitForServer(baseUrl);
    console.log(`[vinext] testing against ${baseUrl} using ${BROWSER_PATH}`);

    context = await chromium.launchPersistentContext(userDataDir, {
      executablePath: BROWSER_PATH,
      headless: true,
      args: [
        '--headless=new',
      ],
    });
    const page = context.pages()[0] || await context.newPage();
    page.on('console', (m) => { if (m.type() === 'error') errors.push(`${m.text()} (${m.location()?.url || ''})`); });
    page.on('pageerror', (e) => errors.push(String(e)));

    await page.goto(baseUrl, { waitUntil: 'networkidle' });

    // Verify COOP/COEP + WebGPU capabilities detected
    const capsText = await page.locator('.topbar').innerText();
    console.log('[topbar]', capsText.replace(/\s+/g, ' '));
    if (!capsText.includes('f16:✓') || !capsText.includes('subgroups:✓') || !capsText.includes('COOP/COEP')) {
      throw new Error(`Missing expected WebGPU/COOP/COEP caps in topbar: ${capsText}`);
    }

    // Verify default selected model is INT8_G64 LM Head
    const selectedVal = await page.locator('select').first().inputValue();
    if (selectedVal !== 'spark-4b-q8head') {
      throw new Error(`Expected default model spark-4b-q8head, got ${selectedVal}`);
    }

    // Click Load Model and wait until ready
    await page.click('#btn-load-model');
    await page.waitForFunction(
      () => document.getElementById('status-text')?.textContent?.includes('就绪 · INT8_G64 LM Head'),
      null,
      { timeout: 60000 }
    );
    console.log('[status]', await page.locator('#status-text').innerText());

    // Turn 1: send Chinese reasoning prompt
    await page.fill('#chat-input', '请用一句话解释什么是第一性原理。');
    await page.click('#btn-send');
    await page.waitForFunction(
      () => document.getElementById('status-text')?.textContent?.includes('就绪 · INT8_G64 LM Head'),
      null,
      { timeout: 60000 }
    );

    const turn1Text = await page.locator('.msg.assistant .bubble').last().innerText();
    const decode1 = await page.locator('#stat-decode').innerText();
    console.log('=== Turn 1 Output ===\n', turn1Text, '\n[Decode Speed]:', decode1);
    if (turn1Text.includes('\uFFFD')) throw new Error('Detected UTF-8 replacement mojibake in Turn 1');
    if (turn1Text.trim().length < 10) throw new Error('Turn 1 output unexpectedly short');

    // Turn 2: incremental multi-turn follow-up
    await page.fill('#chat-input', '能举一个生活中的简单例子吗？');
    await page.click('#btn-send');
    await page.waitForFunction(
      () => document.getElementById('status-text')?.textContent?.includes('就绪 · INT8_G64 LM Head'),
      null,
      { timeout: 60000 }
    );

    const turn2Text = await page.locator('.msg.assistant .bubble').last().innerText();
    const decode2 = await page.locator('#stat-decode').innerText();
    console.log('=== Turn 2 (Incremental KV) Output ===\n', turn2Text, '\n[Decode Speed]:', decode2);
    if (turn2Text.includes('\uFFFD')) throw new Error('Detected UTF-8 replacement mojibake in Turn 2');
    if (turn2Text.trim().length < 10) throw new Error('Turn 2 output unexpectedly short');

    const tps1 = parseFloat(decode1);
    const tps2 = parseFloat(decode2);
    if (!(tps1 >= 40) || !(tps2 >= 40)) {
      throw new Error(`Expected >= 40 tok/s with O(V) sampler, got tps1=${tps1}, tps2=${tps2}`);
    }

    if (errors.length > 0) {
      throw new Error(`Browser console/page errors: ${errors.join(' | ')}`);
    }
    console.log('[PASS] vinext Chat Demo (Spark-X2.6-4B INT8_G64 LM Head) multi-turn WebGPU E2E verified!');
  } catch (err) {
    failed = true;
    throw err;
  } finally {
    if (context) await context.close();
    if (serverProc && !serverProc.killed) {
      spawn('taskkill', ['/pid', String(serverProc.pid), '/T', '/F'], { stdio: 'ignore' });
    }
    fs.rmSync(userDataDir, { recursive: true, force: true });
    if (serverLogs && (process.env.DEBUG_VINEXT || failed || errors.length > 0)) console.log(serverLogs);
  }
}

main().catch((err) => {
  console.error(err);
  process.exit(1);
});
