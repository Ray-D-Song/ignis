import { computeLineDiff, formatUnifiedDiff } from './diff';
import { STARTER_PROJECT_FILES } from './starter-project';

export type WorkspaceFileKind =
  | 'js'
  | 'jsx'
  | 'ts'
  | 'tsx'
  | 'json'
  | 'css'
  | 'html'
  | 'markdown'
  | 'text';

export interface WorkspaceFile {
  path: string;
  kind: WorkspaceFileKind;
  content: string;
  modifiedByAgent?: boolean;
  addedLines?: number;
  removedLines?: number;
  updatedAt: number;
}

export interface WorkspaceCheckpoint {
  id: string;
  step: number;
  tool: 'edit' | 'write' | 'edit_file' | 'write_file' | 'manual';
  path: string;
  existedBefore: boolean;
  beforeContent: string;
  afterContent: string;
  added: number;
  removed: number;
  diffText: string;
  createdAt: number;
}

export function inferFileKind(filePath: string): WorkspaceFileKind {
  const lower = filePath.toLowerCase();
  if (lower.endsWith('.jsx')) return 'jsx';
  if (lower.endsWith('.tsx')) return 'tsx';
  if (lower.endsWith('.ts') || lower.endsWith('.mts') || lower.endsWith('.cts')) return 'ts';
  if (lower.endsWith('.js') || lower.endsWith('.mjs') || lower.endsWith('.cjs')) return 'js';
  if (lower.endsWith('.json')) return 'json';
  if (lower.endsWith('.css')) return 'css';
  if (lower.endsWith('.html') || lower.endsWith('.htm') || lower.endsWith('.svg')) return 'html';
  if (lower.endsWith('.md') || lower.endsWith('.markdown')) return 'markdown';
  return 'text';
}

export function normalizeWorkspacePath(raw: string): string {
  return raw
    .trim()
    .replace(/\\/g, '/')
    .replace(/^\.\//, '')
    .replace(/^\/+/, '')
    .replace(/^workspace\//, '');
}

export function createInitialWorkspace(): WorkspaceFile[] {
  const now = Date.now();
  return STARTER_PROJECT_FILES.map((f) => ({
    path: f.path,
    kind: inferFileKind(f.path),
    content: f.content,
    updatedAt: now,
  }));
}

export function formatWorkspaceTree(files: WorkspaceFile[]): string {
  return files
    .map((f) => `- ${f.path} (${f.content.split(/\r?\n/).length} lines)`)
    .join('\n');
}

export function readWorkspaceFile(files: WorkspaceFile[], rawPath: string): WorkspaceFile | undefined {
  const target = normalizeWorkspacePath(rawPath);
  return files.find((f) => normalizeWorkspacePath(f.path) === target);
}

export function upsertWorkspaceFile(
  files: WorkspaceFile[],
  rawPath: string,
  content: string,
  modifiedByAgent = false,
  diffStats?: { addedLines?: number; removedLines?: number },
): WorkspaceFile[] {
  const cleanPath = normalizeWorkspacePath(rawPath) || 'untitled.txt';
  const kind = inferFileKind(cleanPath);
  const nextEntry: WorkspaceFile = {
    path: cleanPath,
    kind,
    content,
    modifiedByAgent,
    addedLines: diffStats?.addedLines,
    removedLines: diffStats?.removedLines,
    updatedAt: Date.now(),
  };
  const idx = files.findIndex((f) => normalizeWorkspacePath(f.path) === cleanPath);
  if (idx === -1) {
    return [...files, nextEntry].sort((a, b) => a.path.localeCompare(b.path));
  }
  const copy = [...files];
  copy[idx] = { ...copy[idx], ...nextEntry };
  return copy;
}

export function formatNumberedContent(
  content: string,
  startLine = 1,
  endLine?: number,
  maxLines = 140,
): { text: string; totalLines: number; shownFrom: number; shownTo: number; truncated: boolean } {
  const lines = content.replace(/\r\n/g, '\n').split('\n');
  const totalLines = lines.length;
  const from = Math.max(1, startLine);
  const desiredEnd = endLine ? Math.min(totalLines, Math.max(from, endLine)) : Math.min(totalLines, from + maxLines - 1);
  const clampedEnd = Math.min(desiredEnd, from + maxLines - 1);
  const slice = lines.slice(from - 1, clampedEnd);
  const width = String(clampedEnd).length;
  const numbered = slice
    .map((line, i) => `${String(from + i).padStart(width, ' ')} | ${line}`)
    .join('\n');
  return {
    text: numbered,
    totalLines,
    shownFrom: from,
    shownTo: clampedEnd,
    truncated: clampedEnd < totalLines,
  };
}

export function grepWorkspaceFiles(
  files: WorkspaceFile[],
  pattern: string,
  pathPrefix?: string,
  maxMatches = 30,
): string[] {
  const q = pattern.trim();
  if (!q) return [];
  const prefix = pathPrefix ? normalizeWorkspacePath(pathPrefix) : '';

  let re: RegExp | null = null;
  try {
    re = new RegExp(q, 'i');
  } catch {
    re = null;
  }
  const lowerQ = q.toLowerCase();
  const hits: string[] = [];

  for (const f of files) {
    if (prefix && !normalizeWorkspacePath(f.path).startsWith(prefix)) continue;
    const lines = f.content.split(/\r?\n/);
    for (let i = 0; i < lines.length; i++) {
      const line = lines[i];
      const matched = re ? re.test(line) : line.toLowerCase().includes(lowerQ);
      if (matched) {
        hits.push(`${f.path}:${i + 1}: ${line.trimEnd().slice(0, 140)}`);
        if (hits.length >= maxMatches) return hits;
      }
    }
  }
  return hits;
}

export type EditPatchResult =
  | {
      ok: true;
      updatedContent: string;
      added: number;
      removed: number;
      diffText: string;
    }
  | {
      ok: false;
      error: string;
    };

export function applyEditPatch(
  filePath: string,
  original: string,
  oldText: string,
  newText: string,
): EditPatchResult {
  const normOrig = original.replace(/\r\n/g, '\n');
  const normOld = oldText.replace(/\r\n/g, '\n');
  const normNew = newText.replace(/\r\n/g, '\n');

  if (!normOld) {
    return { ok: false, error: 'old_text cannot be empty' };
  }
  if (normOld === normNew) {
    return {
      ok: false,
      error: 'No-op edit rejected: new_text is identical to old_text. Provide the fixed replacement code.',
    };
  }

  // 1. Exact substring match (must be unique)
  const firstIdx = normOrig.indexOf(normOld);
  if (firstIdx !== -1) {
    const secondIdx = normOrig.indexOf(normOld, firstIdx + normOld.length);
    if (secondIdx !== -1) {
      return {
        ok: false,
        error: 'old_text matched multiple locations; include 1-2 surrounding lines to make it unique.',
      };
    }
    const updatedContent =
      normOrig.slice(0, firstIdx) + normNew + normOrig.slice(firstIdx + normOld.length);
    const { added, removed } = computeLineDiff(normOrig, updatedContent);
    const diffText = formatUnifiedDiff(filePath, normOrig, updatedContent);
    return { ok: true, updatedContent, added, removed, diffText };
  }

  // 2. Unambiguous trimmed-line fallback for minor indentation drift from small models
  const origLines = normOrig.split('\n');
  const oldLines = normOld.split('\n');
  while (oldLines.length > 1 && oldLines[oldLines.length - 1] === '') oldLines.pop();

  const matches: number[] = [];
  for (let i = 0; i <= origLines.length - oldLines.length; i++) {
    let allMatch = true;
    for (let k = 0; k < oldLines.length; k++) {
      if (origLines[i + k].trim() !== oldLines[k].trim()) {
        allMatch = false;
        break;
      }
    }
    if (allMatch) matches.push(i);
  }

  if (matches.length === 1) {
    const startLine = matches[0];
    const nextLines = [
      ...origLines.slice(0, startLine),
      ...normNew.split('\n'),
      ...origLines.slice(startLine + oldLines.length),
    ];
    const updatedContent = nextLines.join('\n');
    const { added, removed } = computeLineDiff(normOrig, updatedContent);
    const diffText = formatUnifiedDiff(filePath, normOrig, updatedContent);
    return { ok: true, updatedContent, added, removed, diffText };
  }

  // 3. Provide helpful line-numbered hint around closest anchor line
  const anchor = oldLines.map((l) => l.trim()).find((l) => l.length >= 6) || '';
  if (anchor) {
    const hintIdx = origLines.findIndex((l) => l.includes(anchor));
    if (hintIdx !== -1) {
      const from = Math.max(0, hintIdx - 4);
      const to = Math.min(origLines.length, hintIdx + 8);
      const snippet = origLines
        .slice(from, to)
        .map((l, k) => `${from + k + 1} | ${l}`)
        .join('\n');
      return {
        ok: false,
        error: `old_text did not match verbatim in ${filePath}. Closest lines:\n${snippet}`,
      };
    }
  }

  return {
    ok: false,
    error: `old_text not found in ${filePath}. Re-read ${filePath} and copy exact lines.`,
  };
}
