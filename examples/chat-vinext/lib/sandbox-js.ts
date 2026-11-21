import type { WorkspaceFile } from './workspace-fs';

export interface SandboxExecutionResult {
  ok: boolean;
  logs: string[];
  result: string;
  writtenFiles: Record<string, string>;
  durationMs: number;
  error?: string;
}

const WORKER_SCRIPT = `
self.onmessage = async (ev) => {
  const { code, initialFiles } = ev.data;
  const files = { ...initialFiles };
  const writtenFiles = {};
  const logs = [];

  const fmt = (val) => {
    if (typeof val === 'string') return val;
    if (val === undefined) return 'undefined';
    try {
      return JSON.stringify(val, null, 2);
    } catch {
      return String(val);
    }
  };

  const sandboxConsole = {
    log: (...args) => logs.push(args.map(fmt).join(' ')),
    info: (...args) => logs.push(args.map(fmt).join(' ')),
    warn: (...args) => logs.push('[warn] ' + args.map(fmt).join(' ')),
    error: (...args) => logs.push('[error] ' + args.map(fmt).join(' ')),
  };

  const workspace = {
    list: () => Object.keys(files),
    read: (path) => {
      const key = String(path || '').trim().replace(/^\\/+/, '').replace(/^workspace\\//, '');
      return Object.prototype.hasOwnProperty.call(files, key) ? files[key] : null;
    },
    parseCsv: (path) => {
      const raw = workspace.read(path);
      if (raw == null) return null;
      const lines = raw.trim().split(/\\r?\\n/).filter((l) => l.trim().length > 0);
      if (!lines.length) return { headers: [], rows: [], records: [] };
      const splitLine = (line) => line.split(',').map((s) => s.trim());
      const headers = splitLine(lines[0]);
      const rows = lines.slice(1).map(splitLine);
      const records = rows.map((r) => {
        const obj = {};
        headers.forEach((h, i) => {
          const v = r[i] ?? '';
          const n = Number(v);
          obj[h] = v !== '' && Number.isFinite(n) ? n : v;
        });
        return obj;
      });
      return { headers, rows, records };
    },
    write: (path, content) => {
      const key = String(path || '').trim().replace(/^\\/+/, '').replace(/^workspace\\//, '') || 'output.txt';
      const str = typeof content === 'string' ? content : JSON.stringify(content, null, 2);
      files[key] = str;
      writtenFiles[key] = str;
      return { path: key, bytes: str.length };
    },
  };

  try {
    const AsyncFn = Object.getPrototypeOf(async function () {}).constructor;
    let fn = null;
    const trimmed = String(code || '').trim();

    if (trimmed.length > 0 && !/\\breturn\\b/.test(trimmed)) {
      if (!/[;\\n]/.test(trimmed)) {
        try {
          fn = new AsyncFn('workspace', 'console', 'fetch', 'XMLHttpRequest', 'WebSocket', 'return (' + trimmed + ');');
        } catch {
          fn = null;
        }
      } else {
        const parts = trimmed.split(';').map((s) => s.trim()).filter(Boolean);
        const lastStmt = parts[parts.length - 1];
        if (
          lastStmt &&
          !/^(const|let|var|for|while|if|else|switch|try|catch|function|class|throw)\\b/.test(lastStmt)
        ) {
          const head = parts.slice(0, -1).join(';\\n');
          try {
            fn = new AsyncFn(
              'workspace',
              'console',
              'fetch',
              'XMLHttpRequest',
              'WebSocket',
              (head ? head + ';\\n' : '') + 'return (' + lastStmt + ');'
            );
          } catch {
            fn = null;
          }
        }
      }
    }

    if (!fn) {
      try {
        fn = new AsyncFn('workspace', 'console', 'fetch', 'XMLHttpRequest', 'WebSocket', trimmed);
      } catch (err) {
        self.postMessage({ ok: false, logs, result: '', writtenFiles, error: err && err.message ? err.message : String(err) });
        return;
      }
    }

    const ret = await fn(workspace, sandboxConsole, undefined, undefined, undefined);
    self.postMessage({
      ok: true,
      logs,
      result: ret === undefined ? '' : fmt(ret),
      writtenFiles,
    });
  } catch (err) {
    self.postMessage({
      ok: false,
      logs,
      result: '',
      writtenFiles,
      error: err && err.stack ? err.stack.split('\\n').slice(0, 3).join(' | ') : String(err),
    });
  }
};
`;

export function executeSandboxedJs(
  code: string,
  workspaceFiles: WorkspaceFile[],
  timeoutMs = 2500,
): Promise<SandboxExecutionResult> {
  const t0 = performance.now();
  const initialFiles: Record<string, string> = {};
  for (const f of workspaceFiles) {
    initialFiles[f.path] = f.content;
  }

  return new Promise((resolve) => {
    let settled = false;
    let worker: Worker | null = null;
    let blobUrl = '';

    const cleanup = () => {
      if (worker) {
        worker.terminate();
        worker = null;
      }
      if (blobUrl) {
        URL.revokeObjectURL(blobUrl);
        blobUrl = '';
      }
    };

    const finish = (res: SandboxExecutionResult) => {
      if (settled) return;
      settled = true;
      clearTimeout(timer);
      cleanup();
      resolve(res);
    };

    const timer = setTimeout(() => {
      finish({
        ok: false,
        logs: [],
        result: '',
        writtenFiles: {},
        durationMs: performance.now() - t0,
        error: `Execution timed out after ${timeoutMs}ms`,
      });
    }, timeoutMs);

    try {
      const blob = new Blob([WORKER_SCRIPT], { type: 'application/javascript' });
      blobUrl = URL.createObjectURL(blob);
      worker = new Worker(blobUrl, { type: 'module' });

      worker.onmessage = (ev) => {
        const data = ev.data || {};
        finish({
          ok: Boolean(data.ok),
          logs: Array.isArray(data.logs) ? data.logs : [],
          result: typeof data.result === 'string' ? data.result : '',
          writtenFiles: data.writtenFiles && typeof data.writtenFiles === 'object' ? data.writtenFiles : {},
          durationMs: performance.now() - t0,
          ...(data.error ? { error: String(data.error) } : {}),
        });
      };

      worker.onerror = (ev) => {
        finish({
          ok: false,
          logs: [],
          result: '',
          writtenFiles: {},
          durationMs: performance.now() - t0,
          error: ev.message || 'Sandbox worker error',
        });
      };

      worker.postMessage({ code, initialFiles });
    } catch (err) {
      finish({
        ok: false,
        logs: [],
        result: '',
        writtenFiles: {},
        durationMs: performance.now() - t0,
        error: err instanceof Error ? err.message : String(err),
      });
    }
  });
}
