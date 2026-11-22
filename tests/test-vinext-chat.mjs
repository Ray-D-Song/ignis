import { spawn } from 'node:child_process';
import fs from 'node:fs';
import net from 'node:net';
import os from 'node:os';
import path from 'node:path';
import { chromium } from 'playwright-core';
import { resolveBrowserExecutable } from './browser-env.mjs';

const BROWSER_PATH = resolveBrowserExecutable();
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

  // Always spawn a fresh vinext dev server unless IGNIS_REUSE_DEV=1 is explicitly set
  let baseUrl = null;
  let serverProc = null;
  if (process.env.IGNIS_REUSE_DEV === '1') {
    for (const cand of ['http://localhost:5173', 'http://127.0.0.1:5173']) {
      try {
        const r = await fetch(cand);
        if (r.ok) { baseUrl = cand; break; }
      } catch { /* not running */ }
    }
  }

  let serverLogs = '';
  if (!baseUrl) {
    const PORT = await getFreePort();
    baseUrl = `http://127.0.0.1:${PORT}`;
    serverProc = spawn('node', [vinextCli, 'dev', '--host', '127.0.0.1', '--port', String(PORT)], {
      cwd: APP_DIR,
      stdio: ['ignore', 'pipe', 'pipe'],
      env: { ...process.env, VINEXT_NO_DEV_LOCK: '1' },
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
    console.log(`[vinext] testing against ${baseUrl} using ${BROWSER_PATH || 'playwright-chromium'}`);

    context = await chromium.launchPersistentContext(userDataDir, {
      ...(BROWSER_PATH ? { executablePath: BROWSER_PATH } : {}),
      headless: true,
      viewport: { width: 1440, height: 900 },
      args: [
        '--headless=new',
      ],
    });
    const page = context.pages()[0] || await context.newPage();
    page.on('console', (m) => { if (m.type() === 'error') errors.push(`${m.text()} (${m.location()?.url || ''})`); });
    page.on('pageerror', (e) => errors.push(String(e)));

    await page.goto(baseUrl, { waitUntil: 'domcontentloaded' });
    await page.waitForSelector('#model-select', { timeout: 20000 });

    const isolated = await page.evaluate(() => window.crossOriginIsolated === true);
    if (!isolated) {
      throw new Error('Expected window.crossOriginIsolated === true under vinext dev COOP/COEP headers');
    }

    // Verify COOP/COEP + WebGPU capabilities detected inside Runtime tab (retry click until React hydrates)
    if (!(await page.locator('#workspace-pane').isVisible())) {
      await page.click('#btn-toggle-workspace');
    }
    const hydrateDeadline = Date.now() + 30000;
    while (Date.now() < hydrateDeadline) {
      await page.click('#tab-btn-runtime');
      if (await page.locator('#hardware-card').isVisible().catch(() => false)) {
        break;
      }
      await page.waitForTimeout(250);
    }
    await page.waitForFunction(
      () => {
        const txt = document.getElementById('hardware-card')?.textContent || '';
        return txt.includes('f16:') && txt.includes('COOP/COEP');
      },
      null,
      { timeout: 20000 },
    );
    const capsText = await page.locator('#hardware-card').innerText();
    console.log('[hardware-card]', capsText.replace(/\s+/g, ' '));
    if (!capsText.includes('f16:✓') || !capsText.includes('sg:✓') || !capsText.includes('COOP/COEP')) {
      throw new Error(`Missing expected WebGPU/COOP/COEP caps in Runtime card: ${capsText}`);
    }
    await page.click('#tab-btn-preview');

    // Verify default selected model is Spark-4B Q8Head
    const selectedVal = await page.locator('#model-select').inputValue();
    if (selectedVal !== 'spark-4b-q8head') {
      throw new Error(`Expected default model spark-4b-q8head, got ${selectedVal}`);
    }

    // Click Load Model and wait until ready
    await page.click('#btn-load-model');
    await page.waitForFunction(
      () => document.getElementById('status-text')?.textContent?.includes('就绪 · Spark'),
      null,
      { timeout: 120000 }
    );
    console.log('[status]', await page.locator('#status-text').innerText());

    // Wait for WebContainer PulseBoard live preview
    await page.waitForSelector('#preview-iframe', { timeout: 120000 });
    await page.frameLocator('#preview-iframe').locator('header h1').waitFor({ timeout: 30000 });

    // Enable HITL Approval gate & run Coding Starter 1 on Spark-X2.6 4B
    await page.click('#btn-mode-agent');
    await page.click('#btn-approval-policy');
    const policyLabel = await page.locator('#btn-approval-policy').innerText();
    if (!policyLabel.includes('HITL')) {
      throw new Error(`Expected HITL policy label after toggle, got "${policyLabel}"`);
    }

    await page.locator('.starter-card').first().click();

    const shotDir = process.env.IGNIS_SCREENSHOT_DIR || '';
    if (shotDir) fs.mkdirSync(shotDir, { recursive: true });

    let approvalCount = 0;
    const deadline = Date.now() + 240000;
    while (Date.now() < deadline) {
      const approvalVisible = await page.locator('#approval-card').isVisible().catch(() => false);
      if (approvalVisible) {
        approvalCount++;
        if (shotDir && approvalCount === 1) {
          await page.screenshot({ path: path.join(shotDir, '01-hitl-approval-desktop.png') });
        }
        const btnId = approvalCount === 1 ? '#btn-approve-once' : '#btn-approve-all';
        console.log(`[hitl] approval #${approvalCount} -> clicking ${btnId}`);
        await page.click(btnId);
        await page.waitForTimeout(250);
        continue;
      }
      const done = await page.evaluate(() => {
        const st = document.getElementById('status-text')?.textContent || '';
        return document.querySelectorAll('.msg.assistant').length >= 1 && st.includes('就绪 · Spark');
      }).catch(() => false);
      if (done) break;
      await page.waitForTimeout(400);
    }
    if (approvalCount < 1) {
      throw new Error('Expected at least 1 HITL #approval-card prompt during Spark Agent turn');
    }

    const sparkTools = await page.locator('.tool-call-card').count();
    const toolSummaries = await page.locator('.tool-call-summary').allInnerTexts();
    const verifierBadge = await page.locator('[data-testid="test-verifier-badge"]').last().innerText();
    const agentOut = await page.locator('.msg.assistant .answer-body').last().innerText();
    console.log('[spark-agent] tools=', sparkTools, 'badge=', verifierBadge, toolSummaries.map((s) => s.replace(/\s+/g, ' ')));
    console.log('[spark-agent-answer]', agentOut.trim().slice(0, 180));
    if (sparkTools < 3 || !verifierBadge.includes('6/6 PASS')) {
      await page.click('#tab-btn-trace').catch(() => {});
      const traces = await page.locator('[data-testid="trace-entry"]').allInnerTexts().catch(() => []);
      console.log('[spark-agent-traces]', traces.map((x) => x.replace(/\s+/g, ' ')));
    }
    if (sparkTools < 3) {
      throw new Error(`Expected >= 3 tool calls on Spark Agent bugfix run, got ${sparkTools}`);
    }
    if (!verifierBadge.includes('6/6 PASS')) {
      throw new Error(`Expected 6/6 PASS from real WebContainer test verifier, got "${verifierBadge}"`);
    }

    // Verify CodeMirror 6 Diff View & Revert File button
    if (await page.locator('#btn-editor-diff').isVisible()) {
      await page.click('#btn-editor-diff');
    } else {
      throw new Error('Expected #btn-editor-diff to be visible after agent code edits');
    }
    await page.locator('#code-editor-pane .cm-editor .cm-changedLine').first().waitFor({ timeout: 15000 });
    if (shotDir) {
      await page.screenshot({ path: path.join(shotDir, '02-bugfix-6of6-diff-desktop.png') });
    }
    await page.click('#btn-revert-file');
    await page.locator('#btn-revert-file').waitFor({ state: 'detached', timeout: 10000 });
    console.log('[editor] CodeMirror unifiedMergeView diff + revert verified');

    // Reset session & switch to Direct Chat mode to verify incremental KV throughput
    await page.click('#btn-new-session');
    await page.click('#btn-mode-chat');
    if (!(await page.locator('#workspace-pane').isVisible())) {
      await page.click('#btn-toggle-workspace');
    }
    await page.click('#tab-btn-runtime');

    // Turn 1: send Chinese reasoning prompt
    await page.fill('#chat-input', '请用一句话解释什么是第一性原理。');
    await page.click('#btn-send');
    await page.waitForFunction(
      () =>
        document.querySelectorAll('.msg.assistant').length >= 1 &&
        document.getElementById('status-text')?.textContent?.includes('就绪 · Spark'),
      null,
      { timeout: 60000 }
    );

    const turn1Text = await page.locator('.msg.assistant .answer-body').last().innerText();
    const decode1 = await page.locator('#stat-decode').innerText();
    console.log('=== Turn 1 Answer ===\n', turn1Text, '\n[Decode Speed]:', decode1);
    if (turn1Text.includes('\uFFFD')) throw new Error('Detected UTF-8 replacement mojibake in Turn 1');
    if (turn1Text.trim().length < 10 || turn1Text.trim() === '...') {
      throw new Error(`Turn 1 final answer outside <think> missing or too short: "${turn1Text}"`);
    }

    // Turn 2: incremental multi-turn follow-up
    await page.fill('#chat-input', '比较 WebGPU 与 WebAssembly 在纯浏览器端运行大语言模型时的本质瓶颈差异。');
    await page.click('#btn-send');
    await page.waitForFunction(
      () =>
        document.querySelectorAll('.msg.assistant').length >= 2 &&
        document.getElementById('status-text')?.textContent?.includes('就绪 · Spark'),
      null,
      { timeout: 90000 }
    );

    const turn2Text = await page.locator('.msg.assistant .answer-body').last().innerText();
    const decode2 = await page.locator('#stat-decode').innerText();
    console.log('=== Turn 2 (Incremental KV) Answer ===\n', turn2Text, '\n[Decode Speed]:', decode2);
    if (turn2Text.includes('\uFFFD')) throw new Error('Detected UTF-8 replacement mojibake in Turn 2');
    if (turn2Text.trim().length < 20 || turn2Text.trim() === '...') {
      throw new Error(`Turn 2 final answer outside <think> missing or too short: "${turn2Text}"`);
    }

    const tps1 = parseFloat(decode1);
    const tps2 = parseFloat(decode2);
    if (!(tps1 >= 40) || !(tps2 >= 40)) {
      throw new Error(`Expected >= 40 tok/s with O(V) sampler, got tps1=${tps1}, tps2=${tps2}`);
    }

    if (shotDir) {
      await page.setViewportSize({ width: 390, height: 844 });
      await page.waitForTimeout(250);
      await page.screenshot({ path: path.join(shotDir, '03-mobile-390x844.png') });
    }

    if (errors.length > 0) {
      throw new Error(`Browser console/page errors: ${errors.join(' | ')}`);
    }
    console.log('[PASS] Ignis Code vinext IDE (Spark-X2.6-4B HITL 6/6 Bugfix + Diff/Revert + Multi-Turn Direct Chat) verified!');
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
