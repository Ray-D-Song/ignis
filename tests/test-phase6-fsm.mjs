import http from 'node:http';
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import { chromium } from 'playwright-core';

const ROOT_DIR = path.resolve(import.meta.dirname, '..');
const THORIUM_PATH = 'C:\\Users\\ray\\AppData\\Local\\Thorium\\Application\\thorium.exe';
const EDGE_PATH = 'C:\\Program Files (x86)\\Microsoft\\Edge\\Application\\msedge.exe';

function startStaticServer(harnessHtml) {
  const mime = {
    '.html': 'text/html',
    '.mjs': 'application/javascript',
    '.js': 'application/javascript',
    '.json': 'application/json',
    '.wasm': 'application/wasm',
    '.wgsl': 'text/plain',
    '.ignis': 'application/octet-stream',
  };
  return new Promise((resolve) => {
    const server = http.createServer((req, res) => {
      const cleanUrl = (req.url || '/').split('?')[0];
      if (cleanUrl === '/favicon.ico') { res.writeHead(204); res.end(); return; }
      if (cleanUrl === '/__phase6_fsm__.html') {
        res.writeHead(200, {
          'Content-Type': 'text/html',
          'Cross-Origin-Opener-Policy': 'same-origin',
          'Cross-Origin-Embedder-Policy': 'require-corp',
        });
        res.end(harnessHtml);
        return;
      }
      const rel = cleanUrl === '/' ? 'index.html' : cleanUrl.replace(/^\/+/, '');
      const fullPath = path.join(ROOT_DIR, rel);
      if (!fullPath.startsWith(ROOT_DIR) || !fs.existsSync(fullPath) || fs.statSync(fullPath).isDirectory()) {
        res.writeHead(404);
        res.end('Not found');
        return;
      }
      const ext = path.extname(fullPath);
      res.writeHead(200, {
        'Content-Type': mime[ext] || 'application/octet-stream',
        'Cross-Origin-Opener-Policy': 'same-origin',
        'Cross-Origin-Embedder-Policy': 'require-corp',
      });
      fs.createReadStream(fullPath).pipe(res);
    });
    server.listen(0, '127.0.0.1', () => resolve({ server, port: server.address().port }));
  });
}

const HARNESS_HTML = `<!doctype html><html><body><script type="module">
  import { IgnisOpenAI } from '/packages/core/index.mjs';

  const SCHEMAS = [
    {
      name: 'flat_city_country',
      prompt: 'Output a JSON object for the capital of France with fields city and country.',
      schema: {
        type: 'object',
        properties: {
          city: { type: 'string' },
          country: { type: 'string' },
        },
        required: ['city', 'country'],
      },
      validate: (o) => typeof o.city === 'string' && o.city.length > 0 && typeof o.country === 'string' && o.country.length > 0,
    },
    {
      name: 'enum_int_bool',
      prompt: 'Analyze the review "This engine is super fast!" into JSON with sentiment, score (integer 1-100), and verified (boolean true).',
      schema: {
        type: 'object',
        properties: {
          sentiment: { enum: ['positive', 'negative', 'neutral'] },
          score: { type: 'integer' },
          verified: { type: 'boolean' },
        },
        required: ['sentiment', 'score', 'verified'],
      },
      validate: (o) => ['positive', 'negative', 'neutral'].includes(o.sentiment) && Number.isInteger(o.score) && typeof o.verified === 'boolean',
    },
    {
      name: 'nested_object_array',
      prompt: 'Return JSON for user Alice aged 28 with tags rust and webgpu.',
      schema: {
        type: 'object',
        properties: {
          user: {
            type: 'object',
            properties: {
              name: { type: 'string' },
              age: { type: 'integer' },
            },
            required: ['name', 'age'],
          },
          tags: {
            type: 'array',
            items: { type: 'string' },
            minItems: 1,
          },
        },
        required: ['user', 'tags'],
      },
      validate: (o) => (
        o.user && typeof o.user === 'object' &&
        typeof o.user.name === 'string' && o.user.name.length > 0 &&
        Number.isInteger(o.user.age) &&
        Array.isArray(o.tags) && o.tags.length >= 1 && o.tags.every((t) => typeof t === 'string')
      ),
    },
  ];

  const pctile = (arr, p) => {
    if (!arr.length) return 0;
    const sorted = [...arr].sort((a, b) => a - b);
    return sorted[Math.min(sorted.length - 1, Math.floor((sorted.length - 1) * p))];
  };

  window.__RUN_FSM_SUITE__ = async () => {
    const models = [
      {
        id: 'spark4b',
        modelUrl: '/tests/fixtures/spark-4b-ignis-q8head',
        modelKey: 'fsm-spark-4b',
      },
      {
        id: 'lfm3',
        modelUrl: '/tests/fixtures/lfm3-omni-2b-ignis',
        modelKey: 'fsm-lfm3',
      },
    ];

    const runs = [];
    const allMaskSamples = [];

    for (const m of models) {
      const client = new IgnisOpenAI({
        workerUrl: '/runtime/ignis-worker.mjs',
        models: {
          [m.id]: {
            modelUrl: m.modelUrl,
            modelKey: m.modelKey,
            maxSeqLen: 1024,
            maxBatchTokens: 64,
          },
        },
      });
      try {
        await client.loadModel(m.id);
        for (const sc of SCHEMAS) {
          const resp = await client.chat.completions.create({
            model: m.id,
            messages: [{ role: 'user', content: sc.prompt }],
            max_tokens: 96,
            temperature: 0.0,
            response_format: {
              type: 'json_schema',
              json_schema: { name: sc.name, schema: sc.schema, strict: true },
            },
          });
          const raw = resp.choices?.[0]?.message?.content || '';
          let parsed = null;
          let jsonOk = false;
          let schemaOk = false;
          try {
            parsed = JSON.parse(raw);
            jsonOk = true;
            schemaOk = sc.validate(parsed);
          } catch {
            jsonOk = false;
          }
          const samples = resp.ignis?.fsmMaskMsSamples || [];
          allMaskSamples.push(...samples);
          runs.push({
            model: m.id,
            schema: sc.name,
            raw,
            jsonOk,
            schemaOk,
            tokens: resp.usage?.completion_tokens || 0,
            medianMaskMs: pctile(samples, 0.5),
            p95MaskMs: pctile(samples, 0.95),
          });
        }
      } finally {
        await client.destroy();
      }
    }

    return {
      runs,
      overallMedianMaskMs: pctile(allMaskSamples, 0.5),
      overallP95MaskMs: pctile(allMaskSamples, 0.95),
      sampleCount: allMaskSamples.length,
    };
  };
</script></body></html>`;

async function main() {
  const { server, port } = await startStaticServer(HARNESS_HTML);
  const exePath = fs.existsSync(THORIUM_PATH) ? THORIUM_PATH : EDGE_PATH;
  const userDataDir = fs.mkdtempSync(path.join(os.tmpdir(), 'ignis-fsm-'));
  console.log(`[test-phase6-fsm] Launching ${path.basename(exePath)} against http://127.0.0.1:${port}`);

  const errors = [];
  let context;
  try {
    context = await chromium.launchPersistentContext(userDataDir, {
      executablePath: exePath,
      headless: true,
      args: ['--headless=new'],
    });
    const page = context.pages()[0] || await context.newPage();
    page.on('console', (m) => { if (m.type() === 'error') errors.push(m.text()); });
    page.on('pageerror', (e) => errors.push(String(e)));

    await page.goto(`http://127.0.0.1:${port}/__phase6_fsm__.html`, { waitUntil: 'networkidle' });
    const res = await page.evaluate(() => window.__RUN_FSM_SUITE__());
    console.log('\n[test-phase6-fsm RESULTS]', JSON.stringify(res, null, 2));

    let failed = 0;
    for (const r of res.runs) {
      const ok = r.jsonOk && r.schemaOk;
      console.log(`  [${ok ? 'PASS' : 'FAIL'}] ${r.model} / ${r.schema}: ${r.raw} (med=${r.medianMaskMs.toFixed(3)}ms, p95=${r.p95MaskMs.toFixed(3)}ms)`);
      if (!ok) failed++;
    }
    const latOk = res.sampleCount > 0 && res.overallMedianMaskMs < 1.0;
    console.log(`  [${latOk ? 'PASS' : 'FAIL'}] Overall median maskLogits < 1.0 ms/tok (median=${res.overallMedianMaskMs.toFixed(3)}ms, p95=${res.overallP95MaskMs.toFixed(3)}ms, n=${res.sampleCount})`);
    if (!latOk) failed++;
    const errOk = errors.length === 0;
    console.log(`  [${errOk ? 'PASS' : 'FAIL'}] Zero browser console/page errors (${JSON.stringify(errors)})`);
    if (!errOk) failed++;

    if (failed > 0) throw new Error(`${failed} Stage 6B FSM check(s) failed`);
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
