import type { WebContainer, WebContainerProcess } from '@webcontainer/api';
import { STARTER_PROJECT_FILES, toWebContainerTree } from './starter-project';
import { inferFileKind, normalizeWorkspacePath, type WorkspaceFile } from './workspace-fs';

export type SandboxBootStatus =
  | 'idle'
  | 'booting'
  | 'installing'
  | 'starting'
  | 'ready'
  | 'error';

export interface NodeTestCaseResult {
  name: string;
  ok: boolean;
  error?: string;
}

export interface NodeTestSummary {
  total: number;
  passed: number;
  failed: number;
  cases: NodeTestCaseResult[];
}

export interface CommandExecResult {
  command: string;
  exitCode: number;
  output: string;
  durationMs: number;
  timedOut: boolean;
  testSummary: NodeTestSummary | null;
}

export interface InteractiveShellHandle {
  write: (data: string) => Promise<void>;
  resize: (cols: number, rows: number) => void;
  kill: () => void;
}

export interface BootCallbacks {
  onStatus?: (status: SandboxBootStatus, detail?: string) => void;
  onTerminalData?: (chunk: string) => void;
  onServerLog?: (chunk: string) => void;
  onServerReady?: (url: string, port: number) => void;
  onError?: (message: string) => void;
}

interface SingletonState {
  instance: WebContainer | null;
  bootPromise: Promise<WebContainer> | null;
  readyPromise: Promise<void> | null;
  devProc: WebContainerProcess | null;
  status: SandboxBootStatus;
  statusDetail: string;
  serverUrl: string | null;
  serverPort: number | null;
  serverLogUnsub: (() => void) | null;
}

const GLOBAL_KEY = '__ignisWebContainerSingleton';

function getSingleton(): SingletonState {
  const g = globalThis as unknown as Record<string, SingletonState | undefined>;
  if (!g[GLOBAL_KEY]) {
    g[GLOBAL_KEY] = {
      instance: null,
      bootPromise: null,
      readyPromise: null,
      devProc: null,
      status: 'idle',
      statusDetail: '',
      serverUrl: null,
      serverPort: null,
      serverLogUnsub: null,
    };
  }
  return g[GLOBAL_KEY]!;
}

export function stripAnsi(raw: string): string {
  return raw
    .replace(/\x1b\[[0-9;?]*[ -/]*[@-~]/g, '')
    .replace(/\x1b\][^\x07]*(\x07|\x1b\\)/g, '')
    .replace(/\r/g, '');
}

export function parseNodeTestOutput(rawOutput: string): NodeTestSummary | null {
  const clean = stripAnsi(rawOutput);
  const cases: NodeTestCaseResult[] = [];
  for (const line of clean.split('\n')) {
    const passMatch = line.match(/^\s*(?:✔|ok \d+ -)\s+(.+?)(?:\s+\([\d.]+ms\))?$/);
    if (passMatch) {
      cases.push({ name: passMatch[1].trim(), ok: true });
      continue;
    }
    const failMatch = line.match(/^\s*(?:✖|not ok \d+ -)\s+(.+?)(?:\s+\([\d.]+ms\))?$/);
    if (failMatch) {
      cases.push({ name: failMatch[1].trim(), ok: false });
    }
  }

  const testsMatch = clean.match(/ℹ\s*tests\s+(\d+)/i) || clean.match(/#\s*tests\s+(\d+)/i);
  const passCountMatch = clean.match(/ℹ\s*pass\s+(\d+)/i) || clean.match(/#\s*pass\s+(\d+)/i);
  const failCountMatch = clean.match(/ℹ\s*fail\s+(\d+)/i) || clean.match(/#\s*fail\s+(\d+)/i);

  if (!testsMatch && cases.length === 0) return null;

  const total = testsMatch ? Number(testsMatch[1]) : cases.length;
  const passed = passCountMatch
    ? Number(passCountMatch[1])
    : cases.filter((c) => c.ok).length;
  const failed = failCountMatch
    ? Number(failCountMatch[1])
    : Math.max(0, total - passed);

  // Deduplicate summary failure repeats printed under "failing tests:"
  const uniqueCases: NodeTestCaseResult[] = [];
  const seen = new Set<string>();
  for (const c of cases) {
    if (seen.has(c.name)) continue;
    seen.add(c.name);
    uniqueCases.push(c);
  }

  return { total, passed, failed, cases: uniqueCases };
}

function parentDir(relPath: string): string {
  const idx = relPath.lastIndexOf('/');
  return idx === -1 ? '' : relPath.slice(0, idx);
}

const SKIP_DIRS = new Set(['node_modules', '.git', '.vite', 'dist', '.next']);

async function ensureContainerBooted(
  callbacks: BootCallbacks = {},
): Promise<WebContainer> {
  if (typeof window === 'undefined') {
    throw new Error('WebContainer can only boot in a browser environment');
  }
  const s = getSingleton();
  if (s.instance) return s.instance;
  if (s.bootPromise) return s.bootPromise;

  s.status = 'booting';
  s.statusDetail = 'Booting WebContainer Wasm Node runtime';
  callbacks.onStatus?.('booting', s.statusDetail);

  s.bootPromise = (async () => {
    const { WebContainer } = await import('@webcontainer/api');
    const wc = await WebContainer.boot({
      coep: 'credentialless',
      workdirName: 'pulseboard',
    });
    s.instance = wc;

    const tree = toWebContainerTree(STARTER_PROJECT_FILES);
    await wc.mount(tree as Parameters<WebContainer['mount']>[0]);
    return wc;
  })().catch((err) => {
    s.bootPromise = null;
    s.status = 'error';
    const msg = err instanceof Error ? err.message : String(err);
    s.statusDetail = msg;
    callbacks.onStatus?.('error', msg);
    callbacks.onError?.(msg);
    throw err;
  });

  return s.bootPromise;
}

export async function bootAndStartFullStack(
  callbacks: BootCallbacks = {},
): Promise<{ status: SandboxBootStatus; serverUrl: string | null }> {
  const s = getSingleton();

  if (s.status === 'ready' && s.instance) {
    callbacks.onStatus?.('ready', s.statusDetail);
    if (s.serverUrl && s.serverPort !== null) {
      callbacks.onServerReady?.(s.serverUrl, s.serverPort);
    }
    return { status: 'ready', serverUrl: s.serverUrl };
  }

  if (s.readyPromise) {
    await s.readyPromise;
    callbacks.onStatus?.(s.status, s.statusDetail);
    if (s.serverUrl && s.serverPort !== null) {
      callbacks.onServerReady?.(s.serverUrl, s.serverPort);
    }
    return { status: s.status, serverUrl: s.serverUrl };
  }

  s.readyPromise = (async () => {
    const wc = await ensureContainerBooted(callbacks);

    let hasVite = false;
    try {
      await wc.fs.readFile('node_modules/vite/package.json', 'utf8');
      hasVite = true;
    } catch {
      hasVite = false;
    }

    if (!hasVite) {
      s.status = 'installing';
      s.statusDetail = 'npm install (vite + react)...';
      callbacks.onStatus?.('installing', s.statusDetail);
      callbacks.onTerminalData?.('\r\n$ npm install --no-audit --no-fund\r\n');

      const installProc = await wc.spawn('npm', ['install', '--no-audit', '--no-fund']);
      installProc.output.pipeTo(
        new WritableStream({
          write(data) {
            callbacks.onTerminalData?.(data);
          },
        }),
      ).catch(() => {});

      const exitCode = await installProc.exit;
      if (exitCode !== 0) {
        const msg = `npm install failed with exit code ${exitCode}`;
        s.status = 'error';
        s.statusDetail = msg;
        callbacks.onStatus?.('error', msg);
        callbacks.onError?.(msg);
        throw new Error(msg);
      }
    }

    await startOrRestartDevServer(callbacks);
  })().catch((err) => {
    s.readyPromise = null;
    if (s.status !== 'error') {
      const msg = err instanceof Error ? err.message : String(err);
      s.status = 'error';
      s.statusDetail = msg;
      callbacks.onStatus?.('error', msg);
      callbacks.onError?.(msg);
    }
    throw err;
  });

  await s.readyPromise;
  return { status: s.status, serverUrl: s.serverUrl };
}

export async function startOrRestartDevServer(callbacks: BootCallbacks = {}): Promise<void> {
  const s = getSingleton();
  const wc = await ensureContainerBooted(callbacks);

  if (s.devProc) {
    try {
      s.devProc.kill();
    } catch {
      // ignore
    }
    s.devProc = null;
  }

  s.status = 'starting';
  s.statusDetail = 'Starting Vite 6 + Node API server...';
  callbacks.onStatus?.('starting', s.statusDetail);
  callbacks.onServerLog?.('\r\n$ npm run dev\r\n');

  const unsubServer = wc.on('server-ready', (port, url) => {
    // Ignore Vite auxiliary HMR websocket ports (e.g. 24678) if any extra port opens
    if (port === 24678) return;
    s.serverUrl = url;
    s.serverPort = port;
    s.status = 'ready';
    s.statusDetail = `http://localhost:${port}`;
    callbacks.onStatus?.('ready', s.statusDetail);
    callbacks.onServerReady?.(url, port);
    unsubServer();
  });

  const devProc = await wc.spawn('npm', ['run', 'dev']);
  s.devProc = devProc;

  devProc.output.pipeTo(
    new WritableStream({
      write(data) {
        callbacks.onServerLog?.(data);
      },
    }),
  ).catch(() => {});

  devProc.exit.then((code) => {
    if (s.devProc === devProc && code !== 0 && code !== 143) {
      callbacks.onServerLog?.(`\r\n[dev server exited with code ${code}]\r\n`);
    }
  }).catch(() => {});
}

export function getWebContainerStatus(): {
  status: SandboxBootStatus;
  detail: string;
  serverUrl: string | null;
} {
  const s = getSingleton();
  return { status: s.status, detail: s.statusDetail, serverUrl: s.serverUrl };
}

export async function wcWriteFile(rawPath: string, content: string): Promise<void> {
  const wc = await ensureContainerBooted();
  const rel = normalizeWorkspacePath(rawPath);
  const dir = parentDir(rel);
  if (dir) {
    await wc.fs.mkdir(dir, { recursive: true });
  }
  await wc.fs.writeFile(rel, content);
}

export async function wcReadFile(rawPath: string): Promise<string | null> {
  const wc = await ensureContainerBooted();
  const rel = normalizeWorkspacePath(rawPath);
  try {
    return await wc.fs.readFile(rel, 'utf8');
  } catch {
    return null;
  }
}

export async function wcDeleteFile(rawPath: string): Promise<void> {
  const wc = await ensureContainerBooted();
  const rel = normalizeWorkspacePath(rawPath);
  await wc.fs.rm(rel, { force: true });
}

export async function wcListProjectFiles(): Promise<WorkspaceFile[]> {
  const wc = await ensureContainerBooted();
  const out: WorkspaceFile[] = [];
  const now = Date.now();

  async function walk(dirPath: string) {
    const entries = await wc.fs.readdir(dirPath, { withFileTypes: true });
    for (const ent of entries) {
      if (SKIP_DIRS.has(ent.name)) continue;
      const rel = dirPath === '.' ? ent.name : `${dirPath}/${ent.name}`;
      if (ent.isDirectory()) {
        await walk(rel);
      } else if (ent.isFile()) {
        if (ent.name === 'package-lock.json') continue;
        try {
          const content = await wc.fs.readFile(rel, 'utf8');
          out.push({
            path: rel,
            kind: inferFileKind(rel),
            content,
            updatedAt: now,
          });
        } catch {
          // ignore unreadable/binary files
        }
      }
    }
  }

  await walk('.');
  return out.sort((a, b) => a.path.localeCompare(b.path));
}

export async function wcRunCommand(
  command: string,
  opts?: { timeoutMs?: number; onOutput?: (chunk: string) => void },
): Promise<CommandExecResult> {
  const wc = await ensureContainerBooted();
  const timeoutMs = opts?.timeoutMs ?? 30_000;
  const t0 = performance.now();
  const proc = await wc.spawn('jsh', ['-c', command]);

  let rawOutput = '';
  let timedOut = false;
  let killed = false;

  const timer = setTimeout(() => {
    timedOut = true;
    if (!killed) {
      killed = true;
      try {
        proc.kill();
      } catch {
        // ignore
      }
    }
  }, timeoutMs);

  const pipePromise = proc.output
    .pipeTo(
      new WritableStream({
        write(data) {
          rawOutput += data;
          opts?.onOutput?.(data);
        },
      }),
    )
    .catch(() => {});

  const exitCode = await proc.exit;
  clearTimeout(timer);
  await Promise.race([pipePromise, new Promise((r) => setTimeout(r, 120))]);

  const clean = stripAnsi(rawOutput).trim();
  const testSummary = parseNodeTestOutput(clean);

  return {
    command,
    exitCode: timedOut ? 124 : exitCode,
    output: timedOut ? `${clean}\n\n[Timed out after ${timeoutMs}ms]` : clean,
    durationMs: performance.now() - t0,
    timedOut,
    testSummary,
  };
}

export async function wcAttachInteractiveShell(
  cols: number,
  rows: number,
  onData: (chunk: string) => void,
): Promise<InteractiveShellHandle> {
  const wc = await ensureContainerBooted();
  const shellProc = await wc.spawn('jsh', {
    terminal: {
      cols: Math.max(20, cols),
      rows: Math.max(8, rows),
    },
  });

  shellProc.output
    .pipeTo(
      new WritableStream({
        write(data) {
          onData(data);
        },
      }),
    )
    .catch(() => {});

  const writer = shellProc.input.getWriter();

  return {
    async write(data: string) {
      await writer.write(data);
    },
    resize(nextCols: number, nextRows: number) {
      try {
        shellProc.resize({
          cols: Math.max(20, nextCols),
          rows: Math.max(8, nextRows),
        });
      } catch {
        // ignore closed shell
      }
    },
    kill() {
      try {
        writer.releaseLock();
      } catch {
        // ignore
      }
      try {
        shellProc.kill();
      } catch {
        // ignore
      }
    },
  };
}

export async function wcResetStarterProject(): Promise<WorkspaceFile[]> {
  const wc = await ensureContainerBooted();
  for (const spec of STARTER_PROJECT_FILES) {
    const dir = parentDir(spec.path);
    if (dir) await wc.fs.mkdir(dir, { recursive: true });
    await wc.fs.writeFile(spec.path, spec.content);
  }
  const now = Date.now();
  return STARTER_PROJECT_FILES.map((f) => ({
    path: f.path,
    kind: inferFileKind(f.path),
    content: f.content,
    updatedAt: now,
  }));
}
