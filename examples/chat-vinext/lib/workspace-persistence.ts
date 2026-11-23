import { inferFileKind, normalizeWorkspacePath, type WorkspaceFile } from './workspace-fs';

const DB_NAME = 'ignis-workspace-db';
const DB_VERSION = 1;
const STORE_WORKSPACE = 'workspace';
const STORE_NPM_CACHE = 'npm-cache';
const WORKSPACE_KEY = 'current-project';
const MAX_NPM_CACHE_ENTRIES = 2;

interface PersistedFileItem {
  path: string;
  content: string;
  modifiedByAgent?: boolean;
  addedLines?: number;
  removedLines?: number;
}

interface PersistedWorkspaceRecord {
  updatedAt: number;
  files: PersistedFileItem[];
}

interface NpmCacheRecord {
  key: string;
  updatedAt: number;
  snapshot: Uint8Array;
}

function openWorkspaceDb(): Promise<IDBDatabase | null> {
  if (typeof window === 'undefined' || typeof indexedDB === 'undefined') {
    return Promise.resolve(null);
  }
  return new Promise((resolve) => {
    try {
      const req = indexedDB.open(DB_NAME, DB_VERSION);
      req.onupgradeneeded = () => {
        const db = req.result;
        if (!db.objectStoreNames.contains(STORE_WORKSPACE)) {
          db.createObjectStore(STORE_WORKSPACE);
        }
        if (!db.objectStoreNames.contains(STORE_NPM_CACHE)) {
          db.createObjectStore(STORE_NPM_CACHE, { keyPath: 'key' });
        }
      };
      req.onsuccess = () => resolve(req.result);
      req.onerror = () => resolve(null);
      req.onblocked = () => resolve(null);
    } catch {
      resolve(null);
    }
  });
}

function sortRecordEntries(rec: unknown): Record<string, string> {
  if (!rec || typeof rec !== 'object' || Array.isArray(rec)) return {};
  const entries = Object.entries(rec as Record<string, unknown>)
    .filter(([, v]) => typeof v === 'string')
    .sort(([a], [b]) => a.localeCompare(b));
  return Object.fromEntries(entries) as Record<string, string>;
}

/** Build a deterministic cache key from package.json dependencies & devDependencies. */
export function computePackageDepsKey(packageJsonText: string): string {
  try {
    const parsed = JSON.parse(packageJsonText) as {
      dependencies?: unknown;
      devDependencies?: unknown;
    };
    return JSON.stringify({
      dependencies: sortRecordEntries(parsed.dependencies),
      devDependencies: sortRecordEntries(parsed.devDependencies),
    });
  } catch {
    return packageJsonText.trim();
  }
}

export async function loadPersistedWorkspace(): Promise<WorkspaceFile[] | null> {
  const db = await openWorkspaceDb();
  if (!db) return null;
  return new Promise((resolve) => {
    try {
      const tx = db.transaction(STORE_WORKSPACE, 'readonly');
      const store = tx.objectStore(STORE_WORKSPACE);
      const req = store.get(WORKSPACE_KEY);
      req.onsuccess = () => {
        const record = req.result as PersistedWorkspaceRecord | undefined;
        db.close();
        if (!record || !Array.isArray(record.files) || record.files.length === 0) {
          resolve(null);
          return;
        }
        const now = record.updatedAt || Date.now();
        const validFiles: WorkspaceFile[] = [];
        for (const f of record.files) {
          if (!f || typeof f.path !== 'string' || typeof f.content !== 'string') continue;
          const cleanPath = normalizeWorkspacePath(f.path);
          if (!cleanPath) continue;
          validFiles.push({
            path: cleanPath,
            kind: inferFileKind(cleanPath),
            content: f.content,
            updatedAt: now,
            modifiedByAgent: Boolean(f.modifiedByAgent),
            addedLines: f.addedLines,
            removedLines: f.removedLines,
          });
        }
        resolve(validFiles.length > 0 ? validFiles : null);
      };
      req.onerror = () => {
        db.close();
        resolve(null);
      };
    } catch {
      db.close();
      resolve(null);
    }
  });
}

export async function savePersistedWorkspace(files: WorkspaceFile[]): Promise<void> {
  if (!files || files.length === 0) return;
  const db = await openWorkspaceDb();
  if (!db) return;
  const payload: PersistedWorkspaceRecord = {
    updatedAt: Date.now(),
    files: files.map(({ path, content, modifiedByAgent, addedLines, removedLines }) => ({
      path,
      content,
      modifiedByAgent,
      addedLines,
      removedLines,
    })),
  };
  await new Promise<void>((resolve) => {
    try {
      const tx = db.transaction(STORE_WORKSPACE, 'readwrite');
      tx.objectStore(STORE_WORKSPACE).put(payload, WORKSPACE_KEY);
      tx.oncomplete = () => {
        db.close();
        resolve();
      };
      tx.onerror = () => {
        db.close();
        resolve();
      };
      tx.onabort = () => {
        db.close();
        resolve();
      };
    } catch {
      db.close();
      resolve();
    }
  });
}

export async function clearPersistedWorkspace(): Promise<void> {
  const db = await openWorkspaceDb();
  if (!db) return;
  await new Promise<void>((resolve) => {
    try {
      const tx = db.transaction(STORE_WORKSPACE, 'readwrite');
      tx.objectStore(STORE_WORKSPACE).delete(WORKSPACE_KEY);
      tx.oncomplete = () => {
        db.close();
        resolve();
      };
      tx.onerror = () => {
        db.close();
        resolve();
      };
      tx.onabort = () => {
        db.close();
        resolve();
      };
    } catch {
      db.close();
      resolve();
    }
  });
}

export async function loadNpmCacheSnapshot(depsKey: string): Promise<Uint8Array | null> {
  if (!depsKey) return null;
  const db = await openWorkspaceDb();
  if (!db) return null;
  return new Promise((resolve) => {
    try {
      const tx = db.transaction(STORE_NPM_CACHE, 'readonly');
      const req = tx.objectStore(STORE_NPM_CACHE).get(depsKey);
      req.onsuccess = () => {
        const rec = req.result as NpmCacheRecord | undefined;
        db.close();
        resolve(rec && rec.snapshot instanceof Uint8Array ? rec.snapshot : null);
      };
      req.onerror = () => {
        db.close();
        resolve(null);
      };
    } catch {
      db.close();
      resolve(null);
    }
  });
}

export async function saveNpmCacheSnapshot(depsKey: string, snapshot: Uint8Array): Promise<void> {
  if (!depsKey || !(snapshot instanceof Uint8Array) || snapshot.byteLength === 0) return;
  const db = await openWorkspaceDb();
  if (!db) return;
  const record: NpmCacheRecord = {
    key: depsKey,
    updatedAt: Date.now(),
    snapshot,
  };
  await new Promise<void>((resolve) => {
    try {
      const tx = db.transaction(STORE_NPM_CACHE, 'readwrite');
      const store = tx.objectStore(STORE_NPM_CACHE);
      store.put(record);
      const getAllReq = store.getAll();
      getAllReq.onsuccess = () => {
        const all = (getAllReq.result as NpmCacheRecord[] | undefined) || [];
        if (all.length > MAX_NPM_CACHE_ENTRIES) {
          all.sort((a, b) => (b.updatedAt || 0) - (a.updatedAt || 0));
          for (const stale of all.slice(MAX_NPM_CACHE_ENTRIES)) {
            if (stale?.key) store.delete(stale.key);
          }
        }
      };
      tx.oncomplete = () => {
        db.close();
        resolve();
      };
      tx.onerror = () => {
        db.close();
        resolve();
      };
      tx.onabort = () => {
        db.close();
        resolve();
      };
    } catch {
      db.close();
      resolve();
    }
  });
}
