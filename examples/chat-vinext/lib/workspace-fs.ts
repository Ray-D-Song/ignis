export type WorkspaceFileKind = 'csv' | 'markdown' | 'json' | 'html' | 'text' | 'image' | 'audio';

export interface WorkspaceFile {
  path: string;
  kind: WorkspaceFileKind;
  content: string;
  assetUrl?: string;
  modifiedByAgent?: boolean;
  updatedAt: number;
}

const INITIAL_CSV = `month,region,revenue_usd,active_users,churn_pct
2026-10,North America,48200,1920,2.1
2026-10,Europe,29400,1180,1.9
2026-11,North America,53600,2190,1.8
2026-11,Europe,33100,1340,1.7
2026-12,North America,61900,2540,1.5
2026-12,Europe,38800,1590,1.6
`;

const INITIAL_SPEC = `# Pulse Edge Analytics — Q4 Release Brief

## Product Goal
Ship an offline-capable telemetry and cohort workbench running directly inside customer browsers with zero data egress.

## Target Milestones
1. **Browser Sandbox Core**: Execute CSV & JSON aggregations inside an isolated Web Worker within 100 ms.
2. **Interactive Artifacts**: Render responsive charts, KPI grids, and interactive estimators with dark-theme parity.
3. **Structured Output Lock**: Guarantee valid tool invocation schemas on edge devices with sub-millisecond logit masking.
4. **Multimodal Briefings**: Inspect uploaded screenshots locally and synthesize spoken summaries on supported hardware.

## Success Metrics
- Server dependency: **0 API roundtrips**
- Interactive responsiveness: **< 25 ms per frame**
- Q4 revenue benchmark target: **$250,000+**
`;

export function inferFileKind(filePath: string): WorkspaceFileKind {
  const lower = filePath.toLowerCase();
  if (lower.endsWith('.csv')) return 'csv';
  if (lower.endsWith('.md') || lower.endsWith('.markdown')) return 'markdown';
  if (lower.endsWith('.json')) return 'json';
  if (lower.endsWith('.html') || lower.endsWith('.htm') || lower.endsWith('.svg')) return 'html';
  if (lower.endsWith('.png') || lower.endsWith('.jpg') || lower.endsWith('.jpeg') || lower.endsWith('.webp')) {
    return 'image';
  }
  if (lower.endsWith('.wav') || lower.endsWith('.mp3')) return 'audio';
  return 'text';
}

export function normalizeWorkspacePath(raw: string): string {
  return raw.trim().replace(/^\/+/, '').replace(/^workspace\//, '');
}

export function createInitialWorkspace(): WorkspaceFile[] {
  const now = Date.now();
  return [
    {
      path: 'revenue_q4.csv',
      kind: 'csv',
      content: INITIAL_CSV,
      updatedAt: now,
    },
    {
      path: 'product_spec.md',
      kind: 'markdown',
      content: INITIAL_SPEC,
      updatedAt: now,
    },
    {
      path: 'demo-shapes.png',
      kind: 'image',
      content: '[Binary PNG sample: red and blue geometric shapes]',
      assetUrl: '/samples/demo-shapes.png',
      updatedAt: now,
    },
    {
      path: 'demo-tone.wav',
      kind: 'audio',
      content: '[Binary WAV audio sample: 16 kHz calibration tone]',
      assetUrl: '/samples/demo-tone.wav',
      updatedAt: now,
    },
  ];
}

export function formatWorkspaceTree(files: WorkspaceFile[]): string {
  return files
    .map((f) => {
      if (f.assetUrl) return `- ${f.path} (${f.kind} media asset)`;
      return `- ${f.path} (${f.kind}, ${f.content.length} chars)`;
    })
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
): WorkspaceFile[] {
  const cleanPath = normalizeWorkspacePath(rawPath) || 'untitled.txt';
  const kind = inferFileKind(cleanPath);
  const nextEntry: WorkspaceFile = {
    path: cleanPath,
    kind,
    content,
    modifiedByAgent,
    updatedAt: Date.now(),
  };
  const idx = files.findIndex((f) => normalizeWorkspacePath(f.path) === cleanPath);
  if (idx === -1) {
    return [...files, nextEntry];
  }
  const copy = [...files];
  copy[idx] = { ...copy[idx], ...nextEntry };
  return copy;
}

export function parseCsvRows(content: string): string[][] {
  return content
    .trim()
    .split(/\r?\n/)
    .filter((line) => line.trim().length > 0)
    .map((line) => {
      const cells: string[] = [];
      let cur = '';
      let inQuotes = false;
      for (let i = 0; i < line.length; i++) {
        const ch = line[i];
        if (ch === '"') {
          if (inQuotes && line[i + 1] === '"') {
            cur += '"';
            i++;
          } else {
            inQuotes = !inQuotes;
          }
        } else if (ch === ',' && !inQuotes) {
          cells.push(cur.trim());
          cur = '';
        } else {
          cur += ch;
        }
      }
      cells.push(cur.trim());
      return cells;
    });
}
