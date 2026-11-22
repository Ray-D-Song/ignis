export interface DiffLine {
  type: 'context' | 'add' | 'del';
  oldLine: number | null;
  newLine: number | null;
  text: string;
}

export interface LineDiffResult {
  lines: DiffLine[];
  added: number;
  removed: number;
  changed: boolean;
}

function splitLines(text: string): string[] {
  if (!text) return [];
  const norm = text.replace(/\r\n/g, '\n');
  const lines = norm.split('\n');
  if (lines.length > 1 && lines[lines.length - 1] === '') {
    lines.pop();
  }
  return lines;
}

export function computeLineDiff(before: string, after: string): LineDiffResult {
  const a = splitLines(before);
  const b = splitLines(after);
  const n = a.length;
  const m = b.length;

  if (n === 0 && m === 0) {
    return { lines: [], added: 0, removed: 0, changed: false };
  }

  // Fallback for very large files to keep O(N+M) memory bounded
  if (n * m > 400_000) {
    let prefix = 0;
    while (prefix < n && prefix < m && a[prefix] === b[prefix]) prefix++;
    let suffix = 0;
    while (
      suffix < n - prefix &&
      suffix < m - prefix &&
      a[n - 1 - suffix] === b[m - 1 - suffix]
    ) {
      suffix++;
    }
    const lines: DiffLine[] = [];
    for (let i = 0; i < prefix; i++) {
      lines.push({ type: 'context', oldLine: i + 1, newLine: i + 1, text: a[i] });
    }
    for (let i = prefix; i < n - suffix; i++) {
      lines.push({ type: 'del', oldLine: i + 1, newLine: null, text: a[i] });
    }
    for (let j = prefix; j < m - suffix; j++) {
      lines.push({ type: 'add', oldLine: null, newLine: j + 1, text: b[j] });
    }
    for (let k = 0; k < suffix; k++) {
      lines.push({
        type: 'context',
        oldLine: n - suffix + k + 1,
        newLine: m - suffix + k + 1,
        text: a[n - suffix + k],
      });
    }
    const added = Math.max(0, m - prefix - suffix);
    const removed = Math.max(0, n - prefix - suffix);
    return { lines, added, removed, changed: added > 0 || removed > 0 };
  }

  // Standard LCS DP table
  const dp = new Int32Array((n + 1) * (m + 1));
  const cols = m + 1;
  for (let i = n - 1; i >= 0; i--) {
    const rowOff = i * cols;
    const nextOff = (i + 1) * cols;
    for (let j = m - 1; j >= 0; j--) {
      if (a[i] === b[j]) {
        dp[rowOff + j] = dp[nextOff + j + 1] + 1;
      } else {
        const down = dp[nextOff + j];
        const right = dp[rowOff + j + 1];
        dp[rowOff + j] = down >= right ? down : right;
      }
    }
  }

  const lines: DiffLine[] = [];
  let i = 0;
  let j = 0;
  let added = 0;
  let removed = 0;

  while (i < n && j < m) {
    if (a[i] === b[j]) {
      lines.push({ type: 'context', oldLine: i + 1, newLine: j + 1, text: a[i] });
      i++;
      j++;
    } else if (dp[(i + 1) * cols + j] >= dp[i * cols + j + 1]) {
      lines.push({ type: 'del', oldLine: i + 1, newLine: null, text: a[i] });
      removed++;
      i++;
    } else {
      lines.push({ type: 'add', oldLine: null, newLine: j + 1, text: b[j] });
      added++;
      j++;
    }
  }
  while (i < n) {
    lines.push({ type: 'del', oldLine: i + 1, newLine: null, text: a[i] });
    removed++;
    i++;
  }
  while (j < m) {
    lines.push({ type: 'add', oldLine: null, newLine: j + 1, text: b[j] });
    added++;
    j++;
  }

  return { lines, added, removed, changed: added > 0 || removed > 0 };
}

export function formatUnifiedDiff(
  filePath: string,
  before: string,
  after: string,
  contextLines = 2,
): string {
  const { lines, changed } = computeLineDiff(before, after);
  if (!changed) return `No changes in ${filePath}`;

  const include = new Uint8Array(lines.length);
  for (let i = 0; i < lines.length; i++) {
    if (lines[i].type !== 'context') {
      const start = Math.max(0, i - contextLines);
      const end = Math.min(lines.length - 1, i + contextLines);
      for (let k = start; k <= end; k++) include[k] = 1;
    }
  }

  const out: string[] = [`--- a/${filePath}`, `+++ b/${filePath}`];
  let idx = 0;
  while (idx < lines.length) {
    while (idx < lines.length && !include[idx]) idx++;
    if (idx >= lines.length) break;
    const hunkStart = idx;
    while (idx < lines.length && include[idx]) idx++;
    const hunk = lines.slice(hunkStart, idx);

    let oldStart = 0;
    let newStart = 0;
    let oldCount = 0;
    let newCount = 0;
    for (const ln of hunk) {
      if (oldStart === 0 && ln.oldLine !== null) oldStart = ln.oldLine;
      if (newStart === 0 && ln.newLine !== null) newStart = ln.newLine;
      if (ln.type !== 'add') oldCount++;
      if (ln.type !== 'del') newCount++;
    }
    out.push(`@@ -${oldStart},${oldCount} +${newStart},${newCount} @@`);
    for (const ln of hunk) {
      const prefix = ln.type === 'add' ? '+' : ln.type === 'del' ? '-' : ' ';
      out.push(`${prefix}${ln.text}`);
    }
  }
  return out.join('\n');
}
