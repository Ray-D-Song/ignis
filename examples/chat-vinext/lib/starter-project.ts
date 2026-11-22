export interface StarterFileSpec {
  path: string;
  content: string;
}

const PACKAGE_JSON = `{
  "name": "pulseboard-fullstack",
  "version": "1.0.0",
  "private": true,
  "type": "module",
  "scripts": {
    "dev": "node server.mjs",
    "test": "node --test test/api.test.mjs"
  },
  "dependencies": {
    "react": "^18.3.1",
    "react-dom": "^18.3.1"
  },
  "devDependencies": {
    "@vitejs/plugin-react": "^4.5.2",
    "vite": "^6.3.5"
  }
}
`;

const VITE_CONFIG_JS = `import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';

export default defineConfig({
  plugins: [react()],
  server: {
    host: true,
  },
});
`;

const SERVER_MJS = `import http from 'node:http';
import { createServer as createViteServer } from 'vite';

const PORT = Number(process.env.PORT || 3000);
let viteRef = null;

const server = http.createServer(async (req, res) => {
  const url = req.url || '/';
  if (url.startsWith('/api/')) {
    const started = Date.now();
    try {
      const { handleApi } = await viteRef.ssrLoadModule('/server/api.mjs');
      await handleApi(req, res);
    } catch (err) {
      if (!res.headersSent) {
        res.statusCode = 500;
        res.setHeader('Content-Type', 'application/json; charset=utf-8');
        res.end(JSON.stringify({ error: err instanceof Error ? err.message : String(err) }));
      }
    }
    console.log(\`\${req.method || 'GET'} \${url} \${res.statusCode} \${Date.now() - started}ms\`);
    return;
  }

  viteRef.middlewares(req, res);
});

viteRef = await createViteServer({
  server: {
    middlewareMode: true,
    hmr: { server },
  },
  appType: 'spa',
});

server.listen(PORT, () => {
  console.log(\`PulseBoard full-stack server listening on http://localhost:\${PORT}\`);
});
`;

const STORE_MJS = `import fs from 'node:fs/promises';
import path from 'node:path';

export const DEFAULT_DATA_FILE = path.resolve(process.cwd(), 'data', 'issues.json');
export const VALID_STATUSES = ['open', 'in_progress', 'done'];
export const VALID_PRIORITIES = ['low', 'medium', 'high'];

export async function readIssues(dataFile = DEFAULT_DATA_FILE) {
  const raw = await fs.readFile(dataFile, 'utf-8');
  return JSON.parse(raw);
}

export async function writeIssues(issues, dataFile = DEFAULT_DATA_FILE) {
  await fs.mkdir(path.dirname(dataFile), { recursive: true });
  await fs.writeFile(dataFile, JSON.stringify(issues, null, 2) + '\\n', 'utf-8');
}

export async function listIssues(filters = {}, dataFile = DEFAULT_DATA_FILE) {
  let items = await readIssues(dataFile);
  if (filters.status && filters.status !== 'all') {
    items = items.filter((item) => item.status === filters.status);
  }
  if (filters.priority && filters.priority !== 'all') {
    items = items.filter((item) => item.priority === filters.priority);
  }
  if (filters.q) {
    const q = String(filters.q).trim().toLowerCase();
    items = items.filter((item) => item.title.includes(q));
  }
  return items;
}

export async function createIssue(input, dataFile = DEFAULT_DATA_FILE) {
  const title = String(input?.title || '').trim();
  if (!title) {
    throw new Error('Title is required');
  }
  const status = input?.status || 'open';
  if (!VALID_STATUSES.includes(status)) {
    throw new Error('Invalid status');
  }
  const priority = input?.priority || 'medium';
  if (!VALID_PRIORITIES.includes(priority)) {
    throw new Error('Invalid priority');
  }

  const issues = await readIssues(dataFile);
  const nextId = issues.reduce((max, x) => Math.max(max, Number(x.id) || 0), 0) + 1;
  const created = {
    id: nextId,
    title,
    status,
    priority,
    assignee: String(input?.assignee || 'unassigned').trim() || 'unassigned',
    createdAt: new Date().toISOString().slice(0, 10),
  };
  issues.unshift(created);
  await writeIssues(issues, dataFile);
  return created;
}

export async function updateIssue(id, patch, dataFile = DEFAULT_DATA_FILE) {
  const numId = Number(id);
  const issues = await readIssues(dataFile);
  const idx = issues.findIndex((item) => item.id === numId);
  if (idx === -1) return null;

  if (patch?.status !== undefined && !VALID_STATUSES.includes(patch.status)) {
    throw new Error('Invalid status');
  }
  if (patch?.priority !== undefined && !VALID_PRIORITIES.includes(patch.priority)) {
    throw new Error('Invalid priority');
  }

  const current = issues[idx];
  const updated = {
    ...current,
    ...(patch?.title !== undefined ? { title: String(patch.title).trim() || current.title } : {}),
    ...(patch?.status !== undefined ? { status: patch.status } : {}),
    ...(patch?.priority !== undefined ? { priority: patch.priority } : {}),
    ...(patch?.assignee !== undefined ? { assignee: String(patch.assignee).trim() || 'unassigned' } : {}),
  };
  issues[idx] = updated;
  await writeIssues(issues, dataFile);
  return updated;
}

export async function deleteIssue(id, dataFile = DEFAULT_DATA_FILE) {
  const numId = Number(id);
  const issues = await readIssues(dataFile);
  const next = issues.filter((item) => item.id !== numId);
  if (next.length === issues.length) return false;
  await writeIssues(next, dataFile);
  return true;
}
`;

const STATS_MJS = `export function getStats(issues) {
  const total = issues.length;
  const open = issues.filter((i) => i.status === 'open').length;
  const inProgress = issues.filter((i) => i.status === 'in_progress').length;
  const done = issues.filter((i) => i.status !== 'open').length;
  const completionRate = total === 0 ? 0 : Math.round((done / total) * 100);
  return { total, open, inProgress, done, completionRate };
}
`;

const API_MJS = `import {
  DEFAULT_DATA_FILE,
  createIssue,
  deleteIssue,
  listIssues,
  readIssues,
  updateIssue,
} from './store.mjs';
import { getStats } from './stats.mjs';

function sendJson(res, statusCode, payload) {
  res.statusCode = statusCode;
  res.setHeader('Content-Type', 'application/json; charset=utf-8');
  res.end(JSON.stringify(payload));
}

async function readJsonBody(req) {
  let raw = '';
  for await (const chunk of req) {
    raw += chunk;
  }
  if (!raw.trim()) return {};
  return JSON.parse(raw);
}

export function createApiHandler(dataFile = DEFAULT_DATA_FILE) {
  return async function handleApi(req, res) {
    const parsedUrl = new URL(req.url || '/', 'http://localhost');
    const pathname = parsedUrl.pathname;
    const method = (req.method || 'GET').toUpperCase();

    try {
      if (method === 'GET' && pathname === '/api/health') {
        return sendJson(res, 200, { ok: true, service: 'pulseboard-api' });
      }

      if (method === 'GET' && pathname === '/api/stats') {
        const issues = await readIssues(dataFile);
        return sendJson(res, 200, getStats(issues));
      }

      if (method === 'GET' && pathname === '/api/issues') {
        const issues = await listIssues(
          {
            status: parsedUrl.searchParams.get('status') || undefined,
            priority: parsedUrl.searchParams.get('priority') || undefined,
            q: parsedUrl.searchParams.get('q') || undefined,
          },
          dataFile,
        );
        return sendJson(res, 200, { issues });
      }

      if (method === 'POST' && pathname === '/api/issues') {
        const body = await readJsonBody(req);
        const created = await createIssue(body, dataFile);
        return sendJson(res, 201, { issue: created });
      }

      const idMatch = pathname.match(/^\\/api\\/issues\\/(\\d+)$/);
      if (idMatch) {
        const id = Number(idMatch[1]);
        if (method === 'PATCH') {
          const body = await readJsonBody(req);
          const updated = await updateIssue(id, body, dataFile);
          if (!updated) return sendJson(res, 404, { error: 'Issue not found' });
          return sendJson(res, 200, { issue: updated });
        }
        if (method === 'DELETE') {
          const deleted = await deleteIssue(id, dataFile);
          if (!deleted) return sendJson(res, 404, { error: 'Issue not found' });
          return sendJson(res, 200, { deleted: true, id });
        }
      }

      return sendJson(res, 404, { error: 'Not found' });
    } catch (err) {
      const msg = err instanceof Error ? err.message : String(err);
      const isBadRequest =
        msg === 'Title is required' || msg === 'Invalid status' || msg === 'Invalid priority';
      return sendJson(res, isBadRequest ? 400 : 500, { error: msg });
    }
  };
}

export const handleApi = createApiHandler();
`;

const ISSUES_JSON = `[
  {
    "id": 104,
    "title": "Validate WebGPU storage buffer alignment on Intel Arc",
    "status": "open",
    "priority": "high",
    "assignee": "lin",
    "createdAt": "2026-11-18"
  },
  {
    "id": 103,
    "title": "Stream OPFS shard CRC verification without blocking UI",
    "status": "in_progress",
    "priority": "high",
    "assignee": "maya",
    "createdAt": "2026-11-17"
  },
  {
    "id": 102,
    "title": "Clamp sliding-window KV cache offsets after reset",
    "status": "open",
    "priority": "medium",
    "assignee": "alex",
    "createdAt": "2026-11-15"
  },
  {
    "id": 101,
    "title": "Wire COEP credentialless headers in preview static build",
    "status": "done",
    "priority": "low",
    "assignee": "lin",
    "createdAt": "2026-11-12"
  }
]
`;

const INDEX_HTML = `<!DOCTYPE html>
<html lang="en">
  <head>
    <meta charset="UTF-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1.0" />
    <title>PulseBoard — Full-Stack Issue Tracker</title>
  </head>
  <body>
    <div id="root"></div>
    <script type="module" src="/src/main.jsx"></script>
  </body>
</html>
`;

const MAIN_JSX = `import React from 'react';
import { createRoot } from 'react-dom/client';
import App from './App.jsx';
import './styles.css';

createRoot(document.getElementById('root')).render(
  <React.StrictMode>
    <App />
  </React.StrictMode>,
);
`;

const APP_JSX = `import React, { useCallback, useEffect, useState } from 'react';

const NEXT_STATUS = {
  open: 'in_progress',
  in_progress: 'done',
  done: 'open',
};

const STATUS_LABEL = {
  open: 'Open',
  in_progress: 'In Progress',
  done: 'Done',
};

export default function App() {
  const [issues, setIssues] = useState([]);
  const [stats, setStats] = useState({ total: 0, open: 0, inProgress: 0, done: 0, completionRate: 0 });
  const [statusFilter, setStatusFilter] = useState('all');
  const [title, setTitle] = useState('');
  const [priority, setPriority] = useState('medium');
  const [assignee, setAssignee] = useState('lin');
  const [error, setError] = useState('');

  const loadData = useCallback(async () => {
    try {
      const qs = statusFilter !== 'all' ? \`?status=\${encodeURIComponent(statusFilter)}\` : '';
      const [issuesRes, statsRes] = await Promise.all([
        fetch(\`/api/issues\${qs}\`),
        fetch('/api/stats'),
      ]);
      const issuesJson = await issuesRes.json();
      const statsJson = await statsRes.json();
      setIssues(issuesJson.issues || []);
      setStats(statsJson);
      setError('');
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to load issues');
    }
  }, [statusFilter]);

  useEffect(() => {
    loadData();
  }, [loadData]);

  const handleCreate = async (e) => {
    e.preventDefault();
    if (!title.trim()) return;
    const res = await fetch('/api/issues', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ title, priority, assignee }),
    });
    if (!res.ok) {
      const data = await res.json();
      setError(data.error || 'Create failed');
      return;
    }
    setTitle('');
    await loadData();
  };

  const cycleStatus = async (issue) => {
    const next = NEXT_STATUS[issue.status] || 'open';
    const res = await fetch(\`/api/issues/\${issue.id}\`, {
      method: 'PATCH',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ status: next }),
    });
    if (res.ok) await loadData();
  };

  const handleDelete = async (id) => {
    const res = await fetch(\`/api/issues/\${id}\`, { method: 'DELETE' });
    if (res.ok) await loadData();
  };

  return (
    <div className="shell">
      <header className="topbar">
        <div>
          <h1>PulseBoard</h1>
          <p>Vite 6 + React 18 + Node REST API running inside WebContainer</p>
        </div>
        <span className="pill">/api/stats live</span>
      </header>

      <section className="kpi-grid">
        <div className="kpi-card">
          <span>Total</span>
          <strong>{stats.total}</strong>
        </div>
        <div className="kpi-card">
          <span>Open</span>
          <strong>{stats.open}</strong>
        </div>
        <div className="kpi-card">
          <span>Done</span>
          <strong>{stats.done}</strong>
        </div>
        <div className="kpi-card">
          <span>Completion</span>
          <strong className="accent">{stats.completionRate}%</strong>
        </div>
      </section>

      <form className="create-row" onSubmit={handleCreate}>
        <input
          value={title}
          onChange={(e) => setTitle(e.target.value)}
          placeholder="Add a new engineering issue..."
        />
        <select value={priority} onChange={(e) => setPriority(e.target.value)}>
          <option value="low">low</option>
          <option value="medium">medium</option>
          <option value="high">high</option>
        </select>
        <input
          className="assignee"
          value={assignee}
          onChange={(e) => setAssignee(e.target.value)}
          placeholder="owner"
        />
        <button type="submit">Create</button>
      </form>

      {error && <div className="error-box">{error}</div>}

      <div className="filter-row">
        {['all', 'open', 'in_progress', 'done'].map((st) => (
          <button
            key={st}
            type="button"
            className={statusFilter === st ? 'active' : ''}
            onClick={() => setStatusFilter(st)}
          >
            {st === 'all' ? 'All' : STATUS_LABEL[st]}
          </button>
        ))}
      </div>

      <ul className="issue-list">
        {issues.map((issue) => (
          <li key={issue.id} className="issue-item">
            <div className="issue-main">
              <span className="issue-id">#{issue.id}</span>
              <span className="issue-title">{issue.title}</span>
              <span className={\`tag priority-\${issue.priority}\`}>{issue.priority}</span>
              <span className="assignee-tag">@{issue.assignee}</span>
            </div>
            <div className="issue-actions">
              <button type="button" className={\`status-btn \${issue.status}\`} onClick={() => cycleStatus(issue)}>
                {STATUS_LABEL[issue.status] || issue.status}
              </button>
              <button type="button" className="del-btn" onClick={() => handleDelete(issue.id)}>
                Delete
              </button>
            </div>
          </li>
        ))}
      </ul>
    </div>
  );
}
`;

const STYLES_CSS = `:root {
  color-scheme: dark;
  font-family: Inter, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
}
* { box-sizing: border-box; }
body {
  margin: 0;
  background: #09090b;
  color: #fafafa;
  font-size: 13px;
}
.shell { max-width: 720px; margin: 0 auto; padding: 18px 16px 28px; }
.topbar {
  display: flex; align-items: center; justify-content: space-between;
  margin-bottom: 14px; gap: 10px;
}
.topbar h1 { margin: 0; font-size: 18px; letter-spacing: -0.02em; }
.topbar p { margin: 2px 0 0; color: #a1a1aa; font-size: 12px; }
.pill {
  padding: 3px 9px; border-radius: 999px; font-size: 11px;
  background: rgba(34, 197, 94, 0.12); color: #86efac;
  border: 1px solid rgba(34, 197, 94, 0.3);
}
.kpi-grid {
  display: grid; grid-template-columns: repeat(4, 1fr); gap: 8px; margin-bottom: 14px;
}
.kpi-card {
  background: #111113; border: 1px solid rgba(255,255,255,0.08);
  border-radius: 10px; padding: 10px 12px;
}
.kpi-card span { display: block; color: #71717a; font-size: 11px; }
.kpi-card strong { font-size: 20px; font-weight: 650; }
.kpi-card strong.accent { color: #f97316; }
.create-row {
  display: flex; flex-wrap: wrap; gap: 6px; margin-bottom: 12px;
}
.create-row input, .create-row select {
  background: #111113; color: #fafafa;
  border: 1px solid rgba(255,255,255,0.12); border-radius: 8px;
  padding: 7px 10px; font-size: 12.5px; min-width: 0;
}
.create-row input:first-child { flex: 1 1 180px; }
.create-row .assignee { width: 80px; }
.create-row button {
  background: #fafafa; color: #09090b; border: none; border-radius: 8px;
  padding: 7px 12px; font-weight: 600; cursor: pointer;
}
.filter-row { display: flex; gap: 6px; margin-bottom: 10px; flex-wrap: wrap; }
.filter-row button {
  background: #111113; color: #a1a1aa; border: 1px solid rgba(255,255,255,0.08);
  border-radius: 999px; padding: 4px 10px; font-size: 12px; cursor: pointer;
}
.filter-row button.active {
  background: #18181b; color: #fafafa; border-color: rgba(255,255,255,0.22);
}
.issue-list { list-style: none; padding: 0; margin: 0; display: flex; flex-direction: column; gap: 7px; }
.issue-item {
  display: flex; align-items: center; justify-content: space-between; gap: 10px;
  padding: 10px 12px; background: #111113; border: 1px solid rgba(255,255,255,0.08);
  border-radius: 10px;
}
.issue-main { display: flex; align-items: center; gap: 8px; flex-wrap: wrap; min-width: 0; }
.issue-id { color: #71717a; font-family: monospace; }
.issue-title { font-weight: 500; }
.tag {
  font-size: 10.5px; padding: 1px 6px; border-radius: 5px;
  border: 1px solid rgba(255,255,255,0.12); text-transform: uppercase;
}
.priority-high { color: #fca5a5; background: rgba(239,68,68,0.12); }
.priority-medium { color: #fde68a; background: rgba(234,179,8,0.12); }
.priority-low { color: #93c5fd; background: rgba(59,130,246,0.12); }
.assignee-tag { color: #71717a; font-size: 11.5px; }
.issue-actions { display: flex; gap: 6px; flex-shrink: 0; }
.status-btn, .del-btn {
  border-radius: 7px; padding: 4px 9px; font-size: 11.5px; cursor: pointer;
  border: 1px solid rgba(255,255,255,0.12); background: #18181b; color: #fafafa;
}
.status-btn.done { border-color: rgba(34,197,94,0.35); color: #86efac; }
.status-btn.in_progress { border-color: rgba(249,115,22,0.4); color: #fdba74; }
.del-btn { color: #a1a1aa; }
.error-box {
  margin-bottom: 10px; padding: 8px 10px; border-radius: 8px;
  background: rgba(239,68,68,0.12); border: 1px solid rgba(239,68,68,0.35); color: #fca5a5;
}
`;

const API_TEST_MJS = `import test, { after, before } from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs/promises';
import http from 'node:http';
import os from 'node:os';
import path from 'node:path';
import { createApiHandler } from '../server/api.mjs';

let server;
let baseUrl;
let tmpDir;
let dataFile;

before(async () => {
  tmpDir = await fs.mkdtemp(path.join(os.tmpdir(), 'pulseboard-test-'));
  dataFile = path.join(tmpDir, 'issues.json');
  const seed = await fs.readFile(path.resolve(process.cwd(), 'data', 'issues.json'), 'utf-8');
  await fs.writeFile(dataFile, seed, 'utf-8');

  const handler = createApiHandler(dataFile);
  server = http.createServer(handler);
  await new Promise((resolve) => server.listen(0, '127.0.0.1', resolve));
  const addr = server.address();
  baseUrl = \`http://127.0.0.1:\${addr.port}\`;
});

after(async () => {
  if (server) await new Promise((resolve) => server.close(resolve));
  if (tmpDir) await fs.rm(tmpDir, { recursive: true, force: true });
});

test('GET /api/health returns ok', async () => {
  const res = await fetch(\`\${baseUrl}/api/health\`);
  assert.equal(res.status, 200);
  const body = await res.json();
  assert.equal(body.ok, true);
});

test('GET /api/issues returns seeded issues', async () => {
  const res = await fetch(\`\${baseUrl}/api/issues\`);
  assert.equal(res.status, 200);
  const body = await res.json();
  assert.equal(body.issues.length, 4);
});

test('GET /api/stats counts only done issues in completionRate', async () => {
  const res = await fetch(\`\${baseUrl}/api/stats\`);
  assert.equal(res.status, 200);
  const stats = await res.json();
  // Seeded: 2 open, 1 in_progress, 1 done => done=1, completionRate=25
  assert.equal(stats.done, 1, \`Expected stats.done=1 (got \${stats.done})\`);
  assert.equal(stats.completionRate, 25, \`Expected completionRate=25 (got \${stats.completionRate})\`);
});

test('POST /api/issues creates an issue and rejects empty title', async () => {
  const bad = await fetch(\`\${baseUrl}/api/issues\`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ title: '   ' }),
  });
  assert.equal(bad.status, 400);

  const ok = await fetch(\`\${baseUrl}/api/issues\`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ title: 'Add keyboard shortcut hint', priority: 'low', assignee: 'maya' }),
  });
  assert.equal(ok.status, 201);
  const created = await ok.json();
  assert.equal(created.issue.title, 'Add keyboard shortcut hint');
});

test('PATCH /api/issues/:id updates issue status to done', async () => {
  const res = await fetch(\`\${baseUrl}/api/issues/104\`, {
    method: 'PATCH',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ status: 'done' }),
  });
  assert.equal(res.status, 200);
  const data = await res.json();
  assert.equal(data.issue.status, 'done');
});

test('GET /api/issues?q=webgpu matches title case-insensitively', async () => {
  const res = await fetch(\`\${baseUrl}/api/issues?q=webgpu\`);
  assert.equal(res.status, 200);
  const body = await res.json();
  assert.equal(
    body.issues.length,
    1,
    \`Expected 1 issue matching "webgpu" case-insensitively (got \${body.issues.length})\`,
  );
  assert.equal(body.issues[0].id, 104);
});
`;

export const STARTER_PROJECT_FILES: StarterFileSpec[] = [
  { path: 'package.json', content: PACKAGE_JSON },
  { path: 'vite.config.js', content: VITE_CONFIG_JS },
  { path: 'server.mjs', content: SERVER_MJS },
  { path: 'server/stats.mjs', content: STATS_MJS },
  { path: 'server/store.mjs', content: STORE_MJS },
  { path: 'server/api.mjs', content: API_MJS },
  { path: 'data/issues.json', content: ISSUES_JSON },
  { path: 'index.html', content: INDEX_HTML },
  { path: 'src/main.jsx', content: MAIN_JSX },
  { path: 'src/App.jsx', content: APP_JSX },
  { path: 'src/styles.css', content: STYLES_CSS },
  { path: 'test/api.test.mjs', content: API_TEST_MJS },
];

export type WebContainerTree = Record<string, unknown>;

export function toWebContainerTree(files: Array<{ path: string; content: string }>): WebContainerTree {
  const root: Record<string, unknown> = {};
  for (const f of files) {
    const parts = f.path.replace(/^\/+/, '').split('/').filter(Boolean);
    let cursor = root;
    for (let i = 0; i < parts.length; i++) {
      const seg = parts[i];
      if (i === parts.length - 1) {
        cursor[seg] = { file: { contents: f.content } };
      } else {
        if (!cursor[seg] || typeof (cursor[seg] as { directory?: unknown }).directory !== 'object') {
          cursor[seg] = { directory: {} };
        }
        cursor = (cursor[seg] as { directory: Record<string, unknown> }).directory;
      }
    }
  }
  return root;
}
