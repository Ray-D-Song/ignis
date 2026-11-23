import {
  javascriptLanguage,
  jsxLanguage,
  tsxLanguage,
  typescriptLanguage,
} from '@codemirror/lang-javascript';
import type { IgnisClient } from '../../../runtime/ignis-client.mjs';
import { computeLineDiff, formatUnifiedDiff } from './diff';
import {
  buildPiSystemPrompt,
  PiAgent,
  type PiAgentContext,
  type PiAgentMessage,
  type PiAgentTool,
  type PiAssistantMessage,
  type PiToolCallContent,
  type PiToolResultMessage,
  type PiUserMessage,
} from './pi-agent-core';
import { formatSingleTurnPrompt, splitThinking, type ArchKind, type OmniGenMode } from './prompt-templates';
import {
  parseNodeTestOutput,
  stripAnsi,
  type CommandExecResult,
  type NodeTestSummary,
} from './webcontainer-manager';
import {
  applyEditPatch,
  formatNumberedContent,
  formatWorkspaceTree,
  normalizeWorkspacePath,
  readWorkspaceFile,
  upsertWorkspaceFile,
  type WorkspaceCheckpoint,
  type WorkspaceFile,
} from './workspace-fs';

export type AgentToolName =
  | 'read'
  | 'write'
  | 'edit'
  | 'bash'
  | 'list_files'
  | 'read_file'
  | 'grep_files'
  | 'edit_file'
  | 'write_file'
  | 'run_command'
  | 'finish';

export type ApprovalPolicy = 'auto' | 'ask';

export interface ApprovalRequest {
  id: string;
  step: number;
  tool: 'write' | 'edit' | 'bash' | 'edit_file' | 'write_file' | 'run_command';
  thought: string;
  target: string;
  preview: string;
  added?: number;
  removed?: number;
}

export type ApprovalDecision =
  | { approved: true; approveAll?: boolean }
  | { approved: false; reason?: string };

export interface AgentToolCallRecord {
  id: string;
  step: number;
  tool: AgentToolName;
  thought: string;
  args: Record<string, unknown>;
  output: string;
  ok: boolean;
  durationMs: number;
  fsmMedianMs: number | null;
  decodeTokPerSec?: number | null;
  diffText?: string;
  added?: number;
  removed?: number;
  exitCode?: number;
  testSummary?: NodeTestSummary | null;
  checkpointId?: string;
}

export interface AgentTraceEntry {
  id: string;
  phase: 'perception' | 'decision' | 'tool' | 'approval' | 'synthesis';
  step: number;
  title: string;
  detail: string;
  durationMs: number;
  fsmMedianMs?: number | null;
  decodeTokPerSec?: number | null;
  promptTokens?: number;
  ok: boolean;
}

export interface PendingImageInput {
  data: Uint8Array;
  width: number;
  height: number;
  channels: number;
  previewUrl: string;
  name: string;
}

export interface PendingAudioInput {
  pcm16k: Float32Array;
  durationSec: number;
  name: string;
}

export interface CodingSandboxAdapter {
  writeFile: (path: string, content: string) => Promise<void>;
  readFile?: (path: string) => Promise<string | null>;
  listFiles?: () => Promise<WorkspaceFile[]>;
  runCommand: (command: string) => Promise<CommandExecResult>;
}

export interface AgentTurnCallbacks {
  onStatusText?: (text: string) => void;
  onStepThought?: (thought: string, liveTps: number) => void;
  onToolStepStart?: (step: number, maxSteps: number) => void;
  onToolStepComplete?: (record: AgentToolCallRecord) => void;
  onTraceEntry?: (entry: AgentTraceEntry) => void;
  onWorkspaceUpdate?: (files: WorkspaceFile[]) => void;
  onCheckpoint?: (checkpoint: WorkspaceCheckpoint) => void;
  onApprovalRequest?: (req: ApprovalRequest) => Promise<ApprovalDecision>;
  onOpenFile?: (path: string, mode?: 'code' | 'diff') => void;
  onFinalToken?: (fullRenderedText: string, liveTps: number) => void;
}

export interface AgentTurnResult {
  finalText: string;
  replyThought?: string;
  toolCalls: AgentToolCallRecord[];
  traceEntries: AgentTraceEntry[];
  workspaceFiles: WorkspaceFile[];
  checkpoints: WorkspaceCheckpoint[];
  transcriptMessages: PiAgentMessage[];
  testSummary: NodeTestSummary | null;
  audioUrl?: string;
  totalPromptTokens: number;
  totalGeneratedTokens: number;
  peakPromptTokens: number;
  maxSeqLen: number;
  decodeTokPerSec: number;
  prefillMs: number;
  curPos: number;
  aborted: boolean;
}

/** Strict WASM JSON Schema FSM covering Pi's 4 core tools + reply. */
export function buildAgentActionSchema(options?: { allowReply?: boolean; allowRead?: boolean }): Record<string, unknown> {
  const branches: Record<string, unknown>[] = [];
  if (options?.allowRead !== false) {
    branches.push({
      type: 'object',
      properties: {
        thought: { type: 'string' },
        tool: { const: 'read' },
        path: { type: 'string' },
      },
      required: ['thought', 'tool', 'path'],
      additionalProperties: false,
    });
  }
  branches.push(
    {
      type: 'object',
      properties: {
        thought: { type: 'string' },
        tool: { const: 'write' },
        path: { type: 'string' },
        content: { type: 'string' },
      },
      required: ['thought', 'tool', 'path', 'content'],
      additionalProperties: false,
    },
    {
      type: 'object',
      properties: {
        thought: { type: 'string' },
        tool: { const: 'edit' },
        path: { type: 'string' },
        oldText: { type: 'string' },
        newText: { type: 'string' },
      },
      required: ['thought', 'tool', 'path', 'oldText', 'newText'],
      additionalProperties: false,
    },
    {
      type: 'object',
      properties: {
        thought: { type: 'string' },
        tool: { const: 'bash' },
        command: { type: 'string' },
      },
      required: ['thought', 'tool', 'command'],
      additionalProperties: false,
    },
  );
  if (options?.allowReply !== false) {
    branches.push({
      type: 'object',
      properties: {
        thought: { type: 'string' },
        tool: { const: 'reply' },
        text: { type: 'string' },
      },
      required: ['thought', 'tool', 'text'],
      additionalProperties: false,
    });
  }
  return { oneOf: branches };
}

function median(values?: number[]): number | null {
  if (!values || values.length === 0) return null;
  const s = [...values].sort((a, b) => a - b);
  return s[Math.floor(s.length / 2)];
}

function normalizeToLf(s: string): string {
  return s.replace(/\r\n/g, '\n');
}

function summarizeArgs(args: Record<string, unknown>): Record<string, unknown> {
  const out: Record<string, unknown> = {};
  for (const [k, v] of Object.entries(args)) {
    if (k === 'thought' || k === 'tool') continue;
    out[k] = v;
  }
  return out;
}

function compactCommandOutput(raw: string): string {
  const lines = raw.split('\n');
  const kept: string[] = [];
  let skipStackBlock = false;
  for (const line of lines) {
    if (/^\s*at\s+(?:async\s+)?(?:Test\.|TestContext|process\.|node:internal)/.test(line)) {
      continue;
    }
    if (/^\s*\{\s*$/.test(line) && kept.length > 0 && kept[kept.length - 1].includes('at ')) {
      skipStackBlock = true;
      continue;
    }
    if (skipStackBlock) {
      if (/^\s*\}\s*$/.test(line)) skipStackBlock = false;
      else if (/^\s*(actual|expected):/.test(line)) kept.push(line);
      continue;
    }
    kept.push(line);
  }
  const joined = kept.join('\n').trim();
  return joined.length > 1800 ? `${joined.slice(0, 1800)}\n...[truncated]` : joined;
}

/** Collapse common LLM React wrapper duplication (e.g. redundant `export default function App() { return <App />; }`). */
function normalizeReactModuleContent(targetPath: string, content: string): string {
  if (!/\.(?:[cm]?jsx?|[cm]?tsx?)$/i.test(targetPath)) return content;
  const appDecls =
    content.match(
      /(?:^|\n)\s*(?:export\s+(?:default\s+)?)?(?:async\s+)?function\s+App\b|(?:^|\n)\s*(?:export\s+)?(?:const|let|var|class)\s+App\b/g,
    ) || [];
  if (appDecls.length < 2) return content;

  const wrapperRe =
    /(\r?\n)\s*export\s+default\s+function\s+App\s*\(\s*\)\s*\{\s*return\s*<\s*App\s*\/?\s*>\s*;?\s*\}\s*;?\s*$/;
  if (!wrapperRe.test(content)) return content;

  const withoutWrapper = content.replace(wrapperRe, '');
  if (/\bexport\s+default\b/.test(withoutWrapper)) {
    return `${withoutWrapper.trimEnd()}\n`;
  }
  return `${withoutWrapper.trimEnd()}\n\nexport default App;\n`;
}

/** Validate JS/TS/JSON syntax, scope duplicates, JSX tags, and imports locally via Lezer without emitting WebContainer stderr. */
function validateCodeSyntax(
  targetPath: string,
  content: string,
  workspace?: WorkspaceFile[],
): string | null {
  const lower = targetPath.toLowerCase();
  if (lower.endsWith('.json')) {
    try {
      JSON.parse(content);
      return null;
    } catch (e) {
      return `${targetPath}: JSON SyntaxError: ${e instanceof Error ? e.message : String(e)}`;
    }
  }

  let lang: typeof javascriptLanguage | null = null;
  if (lower.endsWith('.tsx')) lang = tsxLanguage;
  else if (lower.endsWith('.ts') || lower.endsWith('.mts') || lower.endsWith('.cts')) lang = typescriptLanguage;
  else if (lower.endsWith('.jsx')) lang = jsxLanguage;
  else if (lower.endsWith('.js') || lower.endsWith('.mjs') || lower.endsWith('.cjs')) lang = javascriptLanguage;
  if (!lang) return null;

  const getTagName = (tagNode: {
    firstChild: { name: string; from: number; to: number; nextSibling: unknown } | null;
  }): string => {
    for (
      let c = tagNode.firstChild as { name: string; from: number; to: number; nextSibling: unknown } | null;
      c;
      c = c.nextSibling as { name: string; from: number; to: number; nextSibling: unknown } | null
    ) {
      if (
        c.name === 'JSXBuiltin' ||
        c.name === 'JSXIdentifier' ||
        c.name === 'JSXNamespacedName' ||
        c.name === 'JSXMemberExpression'
      ) {
        return content.slice(c.from, c.to).trim();
      }
    }
    return '';
  };

  const tree = lang.parser.parse(content);
  let errFrom: number | null = null;
  let customErr: string | null = null;
  const scopeStack: Array<Set<string>> = [];

  tree.iterate({
    enter(node) {
      if (errFrom !== null) return false;
      if (node.type.isError) {
        errFrom = node.from;
        return false;
      }
      if (node.name === 'Script' || node.name === 'Block') {
        scopeStack.push(new Set<string>());
      }
      if (node.name === 'VariableDefinition' && scopeStack.length > 0) {
        let cur = node.node.parent;
        let inParam = false;
        let isDecl = false;
        while (cur && cur.name !== 'Block' && cur.name !== 'Script') {
          if (cur.name === 'ParamList') {
            inParam = true;
            break;
          }
          if (
            cur.name === 'VariableDeclaration' ||
            cur.name === 'FunctionDeclaration' ||
            cur.name === 'ClassDeclaration'
          ) {
            isDecl = true;
          }
          cur = cur.parent;
        }
        if (!inParam && isDecl) {
          const ident = content.slice(node.from, node.to).trim();
          const currentScope = scopeStack[scopeStack.length - 1];
          if (ident && currentScope) {
            if (currentScope.has(ident)) {
              errFrom = node.from;
              customErr = `Identifier '${ident}' has already been declared in the same scope.`;
              return false;
            }
            currentScope.add(ident);
          }
        }
      }
      if (node.name === 'JSXElement') {
        let openTag: { firstChild: never; from: number } | null = null;
        let closeTag: { firstChild: never; from: number } | null = null;
        for (let c = node.node.firstChild; c; c = c.nextSibling) {
          if (c.name === 'JSXOpenTag') openTag = c as unknown as { firstChild: never; from: number };
          else if (c.name === 'JSXCloseTag') closeTag = c as unknown as { firstChild: never; from: number };
        }
        if (openTag) {
          const openName = getTagName(openTag);
          const closeName = closeTag ? getTagName(closeTag) : '';
          if (!closeTag || openName !== closeName) {
            errFrom = closeTag ? closeTag.from : openTag.from;
            customErr = `Mismatched JSX closing tag: <${openName}> closed by ${closeTag ? `</${closeName}>` : 'EOF'}`;
            return false;
          }
        }
      }
    },
    leave(node) {
      if (node.name === 'Script' || node.name === 'Block') {
        scopeStack.pop();
      }
    },
  });

  if (errFrom !== null) {
    const linesBefore = content.slice(0, errFrom).split('\n');
    const errLine = linesBefore.length;
    const errCol = linesBefore[linesBefore.length - 1].length + 1;
    const snippet = content.split('\n')[errLine - 1] ?? '';
    const pointer = `${' '.repeat(Math.max(0, errCol - 1))}^`;
    return `${targetPath}:${errLine}:${errCol}\n${snippet}\n${pointer}\nSyntaxError: ${customErr || `Unexpected token or unbalanced syntax at line ${errLine}, column ${errCol}`}`;
  }

  const topDeclRe =
    /^(?:export\s+(?:default\s+)?)?(?:async\s+)?function\s+([A-Za-z_$][\w$]*)|^(?:export\s+)?(?:const|let|var|class)\s+([A-Za-z_$][\w$]*)/gm;
  const seenTopNames = new Set<string>();
  let declMatch: RegExpExecArray | null;
  while ((declMatch = topDeclRe.exec(content)) !== null) {
    const ident = declMatch[1] || declMatch[2];
    if (!ident) continue;
    if (seenTopNames.has(ident)) {
      return `${targetPath}: SyntaxError: Identifier '${ident}' has already been declared at top level. Define '${ident}' only once and export it with "export default ${ident};".`;
    }
    seenTopNames.add(ident);
  }

  const normalizedLower = targetPath.replace(/\\/g, '/').toLowerCase();
  if (
    (normalizedLower === 'src/app.jsx' ||
      normalizedLower === 'src/app.tsx' ||
      normalizedLower.endsWith('/app.jsx') ||
      normalizedLower.endsWith('/app.tsx')) &&
    !/\bexport\s+default\b/.test(content)
  ) {
    return `${targetPath}: SyntaxError: ${targetPath} must include a default export (e.g. "export default function App()" or "export default App;").`;
  }

  if (workspace && workspace.length > 0) {
    const pkgFile = workspace.find((f) => normalizeWorkspacePath(f.path) === 'package.json');
    const installedPkgs = new Set<string>(['react', 'react-dom', 'vite', '@vitejs/plugin-react']);
    if (pkgFile) {
      try {
        const parsedPkg = JSON.parse(pkgFile.content) as {
          dependencies?: Record<string, string>;
          devDependencies?: Record<string, string>;
        };
        for (const k of Object.keys(parsedPkg.dependencies || {})) installedPkgs.add(k);
        for (const k of Object.keys(parsedPkg.devDependencies || {})) installedPkgs.add(k);
      } catch {
        // ignore invalid package.json parse
      }
    }

    const fileSet = new Set(workspace.map((f) => normalizeWorkspacePath(f.path)));
    const targetDirParts = normalizeWorkspacePath(targetPath).split('/').slice(0, -1);
    const importRe = /^\s*import\s+(?:[\s\S]*?\s+from\s+)?['"]([^'"]+)['"]/gm;
    let impMatch: RegExpExecArray | null;
    while ((impMatch = importRe.exec(content)) !== null) {
      const spec = impMatch[1].trim();
      if (!spec || spec.startsWith('node:')) continue;
      if (spec.startsWith('.')) {
        const parts = [...targetDirParts];
        for (const seg of spec.split('/')) {
          if (!seg || seg === '.') continue;
          if (seg === '..') parts.pop();
          else parts.push(seg);
        }
        const base = parts.join('/');
        const candidates = [
          base,
          `${base}.js`,
          `${base}.jsx`,
          `${base}.ts`,
          `${base}.tsx`,
          `${base}.mjs`,
          `${base}.json`,
          `${base}.css`,
          `${base}/index.js`,
          `${base}/index.jsx`,
        ];
        if (!candidates.some((c) => fileSet.has(c))) {
          return `${targetPath}: SyntaxError: Cannot resolve relative import '${spec}'. Existing workspace files: ${workspace.map((f) => f.path).join(', ')}`;
        }
      } else {
        const matched = [...installedPkgs].some((pkg) => spec === pkg || spec.startsWith(`${pkg}/`));
        if (!matched) {
          const pkgName = spec.startsWith('@') ? spec.split('/').slice(0, 2).join('/') : spec.split('/')[0];
          return `${targetPath}: Package '${spec}' is not installed in package.json yet. Run bash ("npm install ${pkgName}") first to install it, or implement without external packages.`;
        }
      }
    }
  }

  return null;
}

interface ToolResultDetails {
  filePath?: string;
  compactOut?: string;
  diffText?: string;
  added?: number;
  removed?: number;
  exitCode?: number;
  testSummary?: NodeTestSummary | null;
  checkpointId?: string;
}

interface PreparedWritePayload {
  kind: 'write';
  targetPath: string;
  content: string;
  before: string;
  existedBefore: boolean;
  added: number;
  removed: number;
  diffText: string;
}

interface PreparedEditPayload {
  kind: 'edit';
  targetPath: string;
  updatedContent: string;
  before: string;
  added: number;
  removed: number;
  diffText: string;
}

type PreparedMutation = PreparedWritePayload | PreparedEditPayload;

/** Render stateful Pi transcript messages compactly within model context limits. */
function serializePiContextMessages(
  messages: PiAgentMessage[],
  maxSeqLen: number,
  extraGuidance = '',
): string {
  const latestFileMsgIdx = new Set<number>();
  const seenFiles = new Set<string>();
  let lastBashIdx = -1;

  for (let i = messages.length - 1; i >= 0; i--) {
    const m = messages[i];
    if (m.role !== 'toolResult') continue;
    const det = m.details as ToolResultDetails | undefined;
    if (lastBashIdx === -1 && m.toolName === 'bash') {
      lastBashIdx = i;
    }
    if (seenFiles.size < 4 && det?.filePath && !m.isError) {
      if (!seenFiles.has(det.filePath)) {
        seenFiles.add(det.filePath);
        latestFileMsgIdx.add(i);
      }
    }
  }

  const keepFullSet = new Set<number>(latestFileMsgIdx);
  if (lastBashIdx !== -1) keepFullSet.add(lastBashIdx);

  const lines = messages.map((m, idx) => {
    const isRecent = idx >= messages.length - 4;
    if (m.role === 'user') {
      const txt = typeof m.content === 'string' ? m.content : m.content.map((c) => c.text).join('\n');
      return `[User]: ${txt}`;
    }
    if (m.role === 'assistant') {
      const parts: string[] = [];
      for (const blk of m.content) {
        if (blk.type === 'toolCall') {
          const argPreview =
            blk.name === 'write'
              ? JSON.stringify({
                  path: blk.arguments.path,
                  contentChars: String(blk.arguments.content || '').length,
                })
              : blk.name === 'edit'
                ? JSON.stringify({
                    path: blk.arguments.path,
                    oldTextLen: String(blk.arguments.oldText || '').length,
                    newTextLen: String(blk.arguments.newText || '').length,
                  })
                : JSON.stringify(blk.arguments);
          parts.push(`[Assistant Action -> ${blk.name}]: ${argPreview}`);
        } else if (blk.type === 'text' && blk.text.trim()) {
          parts.push(`[Assistant]: ${isRecent ? blk.text.trim() : blk.text.trim().slice(0, 320)}`);
        }
      }
      return parts.join('\n');
    }

    const rawText = m.content.map((c) => c.text).join('\n');
    const det = m.details as ToolResultDetails | undefined;
    if (isRecent || keepFullSet.has(idx)) {
      const cap = det?.filePath ? 5800 : 2400;
      return `[Tool Result (${m.toolName} -> ${m.isError ? 'ERROR' : 'OK'})]:\n${rawText.slice(0, cap)}`;
    }
    const shortSummary = det?.compactOut || rawText.split('\n')[0].slice(0, 180);
    return `[Tool Result (${m.toolName} -> ${m.isError ? 'ERROR' : 'OK'})]: ${shortSummary}`;
  });

  const charBudget = maxSeqLen <= 4096 ? 6800 : 16500;
  let joined = lines.filter(Boolean).join('\n\n');
  let shrinkCursor = 0;
  while (joined.length > charBudget && shrinkCursor < lines.length - 2) {
    const m = messages[shrinkCursor];
    if (m && m.role === 'toolResult' && !latestFileMsgIdx.has(shrinkCursor)) {
      const det = m.details as ToolResultDetails | undefined;
      const shortSummary = det?.compactOut || m.content[0]?.text?.slice(0, 140) || '';
      lines[shrinkCursor] = `[Tool Result (${m.toolName} -> ${m.isError ? 'ERROR' : 'OK'})]: ${shortSummary}`;
      joined = lines.filter(Boolean).join('\n\n');
    }
    shrinkCursor++;
  }

  return extraGuidance ? `${joined}\n\n${extraGuidance}` : joined;
}

export async function runAgentTurn(params: {
  client: IgnisClient;
  archKind: ArchKind;
  maxSeqLen: number;
  capabilities: { vision: boolean; audioIn: boolean; audioOut: boolean; thinking: boolean };
  userPrompt: string;
  historyText?: string;
  transcriptMessages?: PiAgentMessage[];
  workspaceFiles: WorkspaceFile[];
  approvalPolicy: ApprovalPolicy;
  attachedImage?: PendingImageInput | null;
  attachedAudio?: PendingAudioInput | null;
  generationMode: OmniGenMode;
  maxNewTokens: number;
  maxSteps?: number;
  lang: 'en' | 'zh';
  enableThinking: boolean;
  effort: 'low' | 'high';
  sandbox: CodingSandboxAdapter;
  callbacks: AgentTurnCallbacks;
}): Promise<AgentTurnResult> {
  const {
    client,
    archKind,
    maxSeqLen,
    capabilities,
    userPrompt,
    historyText = '',
    transcriptMessages = [],
    workspaceFiles: initialFiles,
    approvalPolicy,
    attachedImage,
    attachedAudio,
    generationMode,
    maxNewTokens,
    maxSteps = 12,
    lang,
    enableThinking,
    effort,
    sandbox,
    callbacks,
  } = params;

  let workspace = initialFiles.map((f) => ({ ...f }));
  const traceEntries: AgentTraceEntry[] = [];
  const checkpoints: WorkspaceCheckpoint[] = [];
  const toolCalls: AgentToolCallRecord[] = [];
  const preparedMutations = new Map<string, PreparedMutation>();
  const freshReadPaths = new Set<string>();
  const freshBashCommands = new Set<string>();
  let verifiedBuildSinceEdit = false;

  let totalPromptTokens = 0;
  let totalGeneratedTokens = 0;
  let peakPromptTokens = 0;
  let totalDecodeMs = 0;
  let totalPrefillMs = 0;
  let lastCurPos = 0;
  let wasAborted = false;
  const lastTestSummaryRef: { current: NodeTestSummary | null } = { current: null };
  let lastToolError = false;
  let approveAllThisRun = false;
  let stepCounter = 0;
  let retriedReplyGate = false;
  let consecutiveRepeatedReadBlocks = 0;

  const hasPriorCheckpointsInTranscript = transcriptMessages.some(
    (m) => m.role === 'toolResult' && (m.toolName === 'write' || m.toolName === 'edit') && !m.isError,
  );
  const isReadOnlyQuestion =
    /^\s*(?:what|why|how|explain|describe|summarize|list|show|review|audit|tell me|什么|为什么|解释|介绍|总结|列出|查看|分析)/i.test(
      userPrompt.trim(),
    );

  // Build seed transcript if continuing from text history without prior Pi messages
  let baseMessages: PiAgentMessage[] = [...transcriptMessages];
  if (baseMessages.length === 0 && historyText.trim()) {
    baseMessages = [{ role: 'user', content: `Prior context:\n${historyText.trim()}`, timestamp: Date.now() }];
  }

  let effectiveTaskPrompt = userPrompt;

  // Step 0: Multimodal Perception when user attached UI screenshot or voice memo
  if ((attachedImage && capabilities.vision) || (attachedAudio && capabilities.audioIn)) {
    const tPerc0 = performance.now();
    await client.reset();
    const percPrompt = formatSingleTurnPrompt(archKind, {
      systemPrompt:
        'Describe the attached software screenshot or voice instruction concisely so a coding agent can inspect files and implement or fix it.',
      userPrompt,
      enableThinking: false,
      includeImageTag: Boolean(attachedImage && capabilities.vision),
      includeAudioTag: Boolean(attachedAudio && capabilities.audioIn),
    });
    const enc0 = await client.encode(percPrompt);
    peakPromptTokens = Math.max(peakPromptTokens, enc0.ids.length);
    const percRes = await client.generate({
      promptIds: enc0.ids,
      images:
        attachedImage && capabilities.vision
          ? [{ data: attachedImage.data, width: attachedImage.width, height: attachedImage.height, channels: attachedImage.channels }]
          : [],
      audios:
        attachedAudio && capabilities.audioIn
          ? [{ pcm16k: attachedAudio.pcm16k }]
          : [],
      generationMode: 'text',
      maxNewTokens: 256,
      chunkSize: 64,
      sampling: { temperature: 0.2, topK: 20, topP: 0.9, repetitionPenalty: 1.0 },
      stopOnEos: true,
    });
    totalPromptTokens += percRes.promptLen || enc0.ids.length;
    totalGeneratedTokens += percRes.generatedIds.length;
    totalDecodeMs += percRes.decodeMs || 0;
    totalPrefillMs += percRes.prefillMs || 0;
    lastCurPos = percRes.curPos || 0;
    wasAborted = Boolean(percRes.aborted);

    const perceptionText = splitThinking(percRes.generatedText || '').answer.trim();
    if (perceptionText) {
      effectiveTaskPrompt = `${userPrompt}\n\n[Multimodal Inspection Note]:\n${perceptionText}`;
    }
    const percEntry: AgentTraceEntry = {
      id: `trace-perc-${Date.now()}`,
      phase: 'perception',
      step: 0,
      title: 'Multimodal Analysis',
      detail: perceptionText || '(no multimodal observation)',
      durationMs: performance.now() - tPerc0,
      promptTokens: percRes.promptLen || enc0.ids.length,
      ok: Boolean(perceptionText),
    };
    traceEntries.push(percEntry);
    callbacks.onTraceEntry?.(percEntry);
  }

  const prepareWrite = async (targetPathRaw: string, contentRaw: string): Promise<{ ok: true; payload: PreparedWritePayload } | { ok: false; error: string }> => {
    const targetPath = normalizeWorkspacePath(targetPathRaw || 'src/App.jsx');
    const content = normalizeReactModuleContent(targetPath, contentRaw);
    const existing = readWorkspaceFile(workspace, targetPath);
    const before = existing ? existing.content : '';

    if (existing && normalizeToLf(before).trim() === normalizeToLf(content).trim()) {
      const alreadyModifiedThisTurn = checkpoints.some((c) => normalizeWorkspacePath(c.path) === targetPath);
      return {
        ok: false,
        error: alreadyModifiedThisTurn
          ? `[Pi Harness]: ${targetPath} already has this exact updated content from your earlier edit. Do NOT call "write" on ${targetPath} again — call "reply" now.`
          : `write rejected: content is identical to existing ${targetPath}. Write new application code implementing the user's request.`,
      };
    }

    const syntaxErr = validateCodeSyntax(targetPath, content, workspace);
    if (syntaxErr) {
      return {
        ok: false,
        error: `write to ${targetPath} failed validation (file was NOT modified on disk):\n${syntaxErr}\n\nFix the error above and call 'write' on ${targetPath} again with the complete corrected file content.`,
      };
    }

    const diffCounts = computeLineDiff(before, content);
    const diffText = formatUnifiedDiff(targetPath, before, content);
    return {
      ok: true,
      payload: {
        kind: 'write',
        targetPath,
        content,
        before,
        existedBefore: Boolean(existing),
        added: diffCounts.added,
        removed: diffCounts.removed,
        diffText,
      },
    };
  };

  const prepareEdit = async (
    targetPathRaw: string,
    oldText: string,
    newText: string,
  ): Promise<{ ok: true; payload: PreparedEditPayload } | { ok: false; error: string }> => {
    const targetPath = normalizeWorkspacePath(targetPathRaw || '');
    if (!targetPath) {
      return { ok: false, error: 'edit requires a valid file path.' };
    }

    if (normalizeToLf(oldText).trim() === normalizeToLf(newText).trim()) {
      return {
        ok: false,
        error: `edit rejected: newText is identical to oldText (no changes made). To build or replace ${targetPath}, call 'write' with the complete new file content instead of 'edit'.`,
      };
    }

    let existing = readWorkspaceFile(workspace, targetPath);
    if (!existing && sandbox.readFile) {
      const disk = await sandbox.readFile(targetPath);
      if (disk !== null) {
        workspace = upsertWorkspaceFile(workspace, targetPath, disk, false);
        callbacks.onWorkspaceUpdate?.(workspace);
        existing = readWorkspaceFile(workspace, targetPath);
      }
    }
    if (!existing) {
      return {
        ok: false,
        error: `File "${targetPath}" does not exist. Use 'write' to create it or 'read' "." to inspect project files.`,
      };
    }

    const patchRes = applyEditPatch(targetPath, existing.content, oldText, newText);
    if (!patchRes.ok) {
      return {
        ok: false,
        error: `${patchRes.error}\n\nTip: Use 'write' instead of 'edit' when replacing most or all of ${targetPath}.`,
      };
    }

    const normalizedUpdated = normalizeReactModuleContent(targetPath, patchRes.updatedContent);
    const syntaxErr = validateCodeSyntax(targetPath, normalizedUpdated, workspace);
    if (syntaxErr) {
      return {
        ok: false,
        error: `edit to ${targetPath} failed validation and was NOT written to disk:\n${syntaxErr}\n\nFix the issue and retry 'edit' or use 'write' on ${targetPath}.`,
      };
    }

    return {
      ok: true,
      payload: {
        kind: 'edit',
        targetPath,
        updatedContent: normalizedUpdated,
        before: existing.content,
        added: patchRes.added,
        removed: patchRes.removed,
        diffText: patchRes.diffText,
      },
    };
  };

  // Core 4 Pi tools wired to StackBlitz WebContainer + CodeMirror Diff Checkpoints
  const piTools: PiAgentTool[] = [
    {
      name: 'read',
      label: 'Read',
      description: 'Read a file with line numbers (or pass "." to list project files).',
      execute: async (_toolCallId, args) => {
        const rawPath = String(args.path ?? '.').trim();
        const targetPath = normalizeWorkspacePath(rawPath);

        if (!rawPath || rawPath === '.' || rawPath === './' || targetPath === '') {
          if (sandbox.listFiles) {
            workspace = await sandbox.listFiles();
            callbacks.onWorkspaceUpdate?.(workspace);
          }
          const tree = formatWorkspaceTree(workspace);
          const out = `Listed ${workspace.length} files:\n${tree}`;
          return {
            content: [{ type: 'text', text: out }],
            details: { compactOut: `Listed ${workspace.length} files` } satisfies ToolResultDetails,
          };
        }

        let f = readWorkspaceFile(workspace, targetPath);
        if (!f && sandbox.readFile) {
          const disk = await sandbox.readFile(targetPath);
          if (disk !== null) {
            workspace = upsertWorkspaceFile(workspace, targetPath, disk, false);
            callbacks.onWorkspaceUpdate?.(workspace);
            f = readWorkspaceFile(workspace, targetPath);
          }
        }

        if (!f) {
          // Support listing a subdirectory prefix (e.g. "src") cleanly without erroring
          const dirPrefix = targetPath.replace(/\/+$/, '');
          const matched = workspace.filter((wf) => wf.path.startsWith(`${dirPrefix}/`));
          if (matched.length > 0) {
            const out = `Listed ${matched.length} files under ${dirPrefix}/:\n${formatWorkspaceTree(matched)}`;
            return {
              content: [{ type: 'text', text: out }],
              details: { compactOut: `Listed ${matched.length} files in ${dirPrefix}/` } satisfies ToolResultDetails,
            };
          }
          return {
            content: [
              {
                type: 'text',
                text: `File "${targetPath}" not found. Existing files:\n${formatWorkspaceTree(workspace)}`,
              },
            ],
            isError: true,
            details: { compactOut: `read "${targetPath}" not found` } satisfies ToolResultDetails,
          };
        }

        const startLine = typeof args.offset === 'number' && args.offset > 0 ? Math.floor(args.offset) : 1;
        const maxLines = typeof args.limit === 'number' && args.limit > 0 ? Math.floor(args.limit) : 160;
        const numbered = formatNumberedContent(f.content, startLine, undefined, maxLines);
        const out = `${f.path} (lines ${numbered.shownFrom}-${numbered.shownTo} of ${numbered.totalLines}):\n${numbered.text}`;
        freshReadPaths.add(normalizeWorkspacePath(f.path));
        if (checkpoints.length === 0) {
          callbacks.onOpenFile?.(f.path, 'code');
        }
        return {
          content: [{ type: 'text', text: out }],
          details: {
            filePath: f.path,
            compactOut: `Read ${f.path} (${numbered.totalLines} lines)`,
          } satisfies ToolResultDetails,
        };
      },
    },
    {
      name: 'write',
      label: 'Write',
      description: 'Create a new file or overwrite an existing file completely.',
      execute: async (toolCallId, args) => {
        let prepared = preparedMutations.get(toolCallId);
        if (!prepared || prepared.kind !== 'write') {
          const res = await prepareWrite(String(args.path ?? ''), String(args.content ?? ''));
          if (!res.ok) {
            return {
              content: [{ type: 'text', text: res.error }],
              isError: true,
              details: { compactOut: `write ${normalizeWorkspacePath(String(args.path ?? ''))} rejected` } satisfies ToolResultDetails,
            };
          }
          prepared = res.payload;
        }
        preparedMutations.delete(toolCallId);

        const { targetPath, content, before, existedBefore, added, removed, diffText } = prepared;
        await sandbox.writeFile(targetPath, content);
        freshReadPaths.add(normalizeWorkspacePath(targetPath));
        freshBashCommands.clear();
        verifiedBuildSinceEdit = false;
        consecutiveRepeatedReadBlocks = 0;
        workspace = upsertWorkspaceFile(workspace, targetPath, content, true, {
          addedLines: added,
          removedLines: removed,
        });
        callbacks.onWorkspaceUpdate?.(workspace);

        const checkpointId = `ckpt-${stepCounter}-${Date.now()}`;
        const ckpt: WorkspaceCheckpoint = {
          id: checkpointId,
          step: stepCounter,
          tool: 'write',
          path: targetPath,
          existedBefore,
          beforeContent: before,
          afterContent: content,
          added,
          removed,
          diffText,
          createdAt: Date.now(),
        };
        checkpoints.push(ckpt);
        callbacks.onCheckpoint?.(ckpt);
        callbacks.onOpenFile?.(targetPath, existedBefore ? 'diff' : 'code');

        const numbered = formatNumberedContent(content, 1, undefined, 160);
        const out = `Wrote ${content.length} chars to ${targetPath} (${numbered.totalLines} lines):\n${numbered.text}`;
        return {
          content: [{ type: 'text', text: out }],
          details: {
            filePath: targetPath,
            checkpointId,
            diffText,
            added,
            removed,
            compactOut: `Wrote ${targetPath} (${content.length} chars)`,
          } satisfies ToolResultDetails,
        };
      },
    },
    {
      name: 'edit',
      label: 'Edit',
      description: 'Surgical exact text replacement inside an existing file.',
      execute: async (toolCallId, args) => {
        let prepared = preparedMutations.get(toolCallId);
        if (!prepared || prepared.kind !== 'edit') {
          const res = await prepareEdit(
            String(args.path ?? ''),
            String(args.oldText ?? ''),
            String(args.newText ?? ''),
          );
          if (!res.ok) {
            return {
              content: [{ type: 'text', text: res.error }],
              isError: true,
              details: { compactOut: `edit ${normalizeWorkspacePath(String(args.path ?? ''))} failed` } satisfies ToolResultDetails,
            };
          }
          prepared = res.payload;
        }
        preparedMutations.delete(toolCallId);

        const { targetPath, updatedContent, before, added, removed, diffText } = prepared;
        const existing = readWorkspaceFile(workspace, targetPath);
        await sandbox.writeFile(targetPath, updatedContent);
        freshReadPaths.add(normalizeWorkspacePath(targetPath));
        freshBashCommands.clear();
        verifiedBuildSinceEdit = false;
        consecutiveRepeatedReadBlocks = 0;
        workspace = upsertWorkspaceFile(workspace, targetPath, updatedContent, true, {
          addedLines: (existing?.addedLines || 0) + added,
          removedLines: (existing?.removedLines || 0) + removed,
        });
        callbacks.onWorkspaceUpdate?.(workspace);

        const checkpointId = `ckpt-${stepCounter}-${Date.now()}`;
        const ckpt: WorkspaceCheckpoint = {
          id: checkpointId,
          step: stepCounter,
          tool: 'edit',
          path: targetPath,
          existedBefore: true,
          beforeContent: before,
          afterContent: updatedContent,
          added,
          removed,
          diffText,
          createdAt: Date.now(),
        };
        checkpoints.push(ckpt);
        callbacks.onCheckpoint?.(ckpt);
        callbacks.onOpenFile?.(targetPath, 'diff');

        const numbered = formatNumberedContent(updatedContent, 1, undefined, 160);
        const out = `Edited ${targetPath} (+${added}/-${removed}):\n${diffText}\n\nCurrent ${targetPath} (${numbered.totalLines} lines):\n${numbered.text}`;
        return {
          content: [{ type: 'text', text: out }],
          details: {
            filePath: targetPath,
            checkpointId,
            diffText,
            added,
            removed,
            compactOut: `Edited ${targetPath} (+${added}/-${removed})`,
          } satisfies ToolResultDetails,
        };
      },
    },
    {
      name: 'bash',
      label: 'Bash',
      description: 'Execute a shell command inside WebContainer.',
      execute: async (_toolCallId, args) => {
        const cmd = String(args.command ?? '').trim();
        if (!cmd || ['read', 'write', 'edit', 'read_file', 'write_file', 'edit_file'].includes(cmd)) {
          return {
            content: [
              {
                type: 'text',
                text: `"${cmd || '(empty)'}" is a tool name, not a shell command. Call the "${cmd || 'read'}" tool directly instead of bash.`,
              },
            ],
            isError: true,
            details: { compactOut: `Invalid bash command: ${cmd}` } satisfies ToolResultDetails,
          };
        }

        const cmdRes = await sandbox.runCommand(cmd);
        consecutiveRepeatedReadBlocks = 0;
        const normalizedCmd = cmd.trim();
        const isPkgInstallCmd = /\b(?:npm|pnpm|yarn)\s+(?:i|install|add|remove|uninstall|rm)\b/i.test(normalizedCmd);
        if (cmdRes.exitCode === 0 && normalizedCmd) {
          freshBashCommands.add(normalizedCmd);
          if (isPkgInstallCmd && sandbox.readFile) {
            try {
              const latestPkg = await sandbox.readFile('package.json');
              if (typeof latestPkg === 'string' && latestPkg.trim()) {
                workspace = upsertWorkspaceFile(workspace, 'package.json', latestPkg, false);
                freshReadPaths.delete('package.json');
                callbacks.onWorkspaceUpdate?.(workspace);
              }
            } catch {
              // ignore if package.json cannot be refreshed
            }
          } else if (checkpoints.length > 0) {
            verifiedBuildSinceEdit = true;
          }
        }
        const cleanOut = stripAnsi(cmdRes.output);
        const testSummary = cmdRes.testSummary || parseNodeTestOutput(cleanOut);
        if (testSummary) lastTestSummaryRef.current = testSummary;

        const compactOutText = compactCommandOutput(cleanOut);
        const out = `$ ${cmd} (exit ${cmdRes.exitCode}, ${Math.round(cmdRes.durationMs)}ms)\n${compactOutText || '(no output)'}`;
        return {
          content: [{ type: 'text', text: out }],
          isError: cmdRes.exitCode !== 0,
          details: {
            exitCode: cmdRes.exitCode,
            testSummary,
            compactOut: testSummary
              ? `$ ${cmd} -> exit ${cmdRes.exitCode} (${testSummary.passed}/${testSummary.total} passed)`
              : `$ ${cmd} -> exit ${cmdRes.exitCode}`,
          } satisfies ToolResultDetails,
        };
      },
    },
  ];

  const schema = buildAgentActionSchema();
  const actionOnlySchema = buildAgentActionSchema({ allowReply: false, allowRead: false });
  const agent = new PiAgent({
    systemPrompt: '',
    messages: baseMessages,
    tools: piTools,
  });

  const actionStartTimeMap = new Map<string, number>();
  const actionCallMap = new Map<string, PiToolCallContent>();

  agent.subscribe((ev) => {
    if (ev.type === 'tool_execution_start') {
      actionStartTimeMap.set(ev.toolCallId, performance.now());
    } else if (ev.type === 'tool_execution_end') {
      const tc = actionCallMap.get(ev.toolCallId);
      const det = (ev.result.details || {}) as ToolResultDetails;
      const outText = ev.result.content.map((c) => c.text).join('\n');
      const dur = performance.now() - (actionStartTimeMap.get(ev.toolCallId) ?? performance.now());
      const isRepeatedGuard =
        (ev.toolName === 'read' || ev.toolName === 'bash' || ev.toolName === 'write') &&
        ev.isError &&
        outText.startsWith('[Pi Harness]:');
      lastToolError = ev.isError && !isRepeatedGuard;

      if (!isRepeatedGuard) {
        const record: AgentToolCallRecord = {
          id: ev.toolCallId,
          step: stepCounter,
          tool: ev.toolName as AgentToolName,
          thought: tc?.thought || '',
          args: tc?.arguments || {},
          output: outText,
          ok: !ev.isError,
          durationMs: dur,
          fsmMedianMs: tc?.fsmMedianMs ?? null,
          decodeTokPerSec: tc?.decodeTokPerSec ?? null,
          diffText: det.diffText,
          added: det.added,
          removed: det.removed,
          exitCode: det.exitCode,
          testSummary: det.testSummary,
          checkpointId: det.checkpointId,
        };
        toolCalls.push(record);
        callbacks.onToolStepComplete?.(record);
      }

      const toolTrace: AgentTraceEntry = {
        id: `trace-tool-${stepCounter}-${Date.now()}`,
        phase: 'tool',
        step: stepCounter,
        title: `${ev.toolName} -> ${ev.isError ? 'err' : 'ok'}`,
        detail: det.compactOut || outText.slice(0, 220),
        durationMs: dur,
        fsmMedianMs: tc?.fsmMedianMs ?? null,
        decodeTokPerSec: tc?.decodeTokPerSec ?? null,
        promptTokens: tc?.promptTokens,
        ok: !ev.isError,
      };
      traceEntries.push(toolTrace);
      callbacks.onTraceEntry?.(toolTrace);
    }
  });

  const runDecisionStep = async (
    context: PiAgentContext,
    turnIndex: number,
    extraHint = '',
    overrideSchema?: Record<string, unknown>,
    allowJsonRetry = true,
  ): Promise<{
    kind: 'toolCall';
    tc: PiToolCallContent;
  } | {
    kind: 'reply';
    text: string;
    thought: string;
    decodeTokPerSec: number;
  } | {
    kind: 'stop';
    reason: 'aborted' | 'error';
    message: string;
  }> => {
    const sysPrompt = buildPiSystemPrompt({
      workspaceTree: formatWorkspaceTree(workspace),
      lang,
    });
    const reserveGen = maxSeqLen <= 4096 ? 1200 : 2048;

    let workingMessages = [...context.messages];
    let serialized = serializePiContextMessages(workingMessages, maxSeqLen, extraHint);
    const statusHint =
      freshReadPaths.size > 0 && checkpoints.length === 0
        ? `\n[Status]: Already read [${[...freshReadPaths].join(', ')}]. Do NOT read them again; call "write" on src/App.jsx (keep code concise, under 85 lines) or read another unread file.`
        : verifiedBuildSinceEdit && checkpoints.length > 0 && !lastToolError
          ? `\n[Status]: Modified [${[...new Set(checkpoints.map((c) => c.path))].join(', ')}] and verified command succeeded (exit 0). Do NOT call "bash" or "read" again — call "reply" now.`
          : checkpoints.length > 0 && !lastToolError
            ? `\n[Status]: Modified [${[...new Set(checkpoints.map((c) => c.path))].join(', ')}]. Call "reply" now if the task is complete (or edit/write another file such as src/App.css).`
            : '';
    const buildPrompt = (ctxText: string) =>
      formatSingleTurnPrompt(archKind, {
        systemPrompt: sysPrompt,
        userPrompt: `${ctxText}${statusHint}\n\nChoose the next single JSON action:`,
        enableThinking: false,
      });

    let enc = await client.encode(buildPrompt(serialized));
    while (enc.ids.length > maxSeqLen - reserveGen && workingMessages.length > 2) {
      workingMessages = workingMessages.slice(1);
      serialized = serializePiContextMessages(workingMessages, maxSeqLen, extraHint);
      enc = await client.encode(buildPrompt(serialized));
    }

    peakPromptTokens = Math.max(peakPromptTokens, enc.ids.length);
    await client.reset();
    const tDec0 = performance.now();
    const effectiveSchema =
      overrideSchema ??
      (consecutiveRepeatedReadBlocks >= 1 &&
      checkpoints.length === 0 &&
      !hasPriorCheckpointsInTranscript &&
      !isReadOnlyQuestion
        ? actionOnlySchema
        : schema);
    let streamedDec = '';
    let lastStreamedThought = '';
    client.onToken = (ev: { text: string; fullText?: string; tokPerSec?: number }) => {
      const full = ev.fullText !== undefined ? ev.fullText : (streamedDec += ev.text || '');
      const m = full.match(/"thought"\s*:\s*"((?:[^"\\]|\\.)*)/);
      if (m) {
        const thoughtText = m[1]
          .replace(/\\$/, '')
          .replace(/\\(["\\/nrt])/g, (_s, ch) => {
            if (ch === 'n') return '\n';
            if (ch === 'r') return '';
            if (ch === 't') return '\t';
            return ch;
          })
          .trim();
        if (thoughtText && thoughtText !== lastStreamedThought) {
          lastStreamedThought = thoughtText;
          callbacks.onStepThought?.(thoughtText, ev.tokPerSec || 0);
        }
      }
    };
    let stepRes;
    try {
      stepRes = await client.generate({
        promptIds: enc.ids,
        images: [],
        audios: [],
        generationMode: 'text',
        maxNewTokens: Math.min(reserveGen - 32, Math.max(256, maxSeqLen - enc.ids.length - 32)),
        chunkSize: 64,
        sampling: { temperature: 0, topK: 1, topP: 1.0, repetitionPenalty: 1.0 },
        responseFormat: {
          type: 'json_schema',
          json_schema: { name: 'pi_coding_action', schema: effectiveSchema, strict: true },
        },
        stopOnEos: true,
      });
    } finally {
      client.onToken = null;
    }

    const promptLen = stepRes.promptLen || enc.ids.length;
    totalPromptTokens += promptLen;
    peakPromptTokens = Math.max(peakPromptTokens, promptLen);
    totalGeneratedTokens += stepRes.generatedIds.length;
    totalDecodeMs += stepRes.decodeMs || performance.now() - tDec0;
    totalPrefillMs += stepRes.prefillMs || 0;
    lastCurPos = stepRes.curPos || lastCurPos;
    wasAborted = wasAborted || Boolean(stepRes.aborted);

    if (stepRes.aborted) {
      return { kind: 'stop', reason: 'aborted', message: 'Stopped by user.' };
    }

    const fsmMs = median(stepRes.fsmMaskMsSamples);
    const stepTokPerSec = stepRes.decodeTokPerSec || 0;
    const rawJson = splitThinking(stepRes.generatedText || '').answer.trim();

    let action: Record<string, unknown> | null = null;
    try {
      action = JSON.parse(rawJson) as Record<string, unknown>;
    } catch {
      const traceFail: AgentTraceEntry = {
        id: `trace-dec-err-${turnIndex}-${Date.now()}`,
        phase: 'decision',
        step: stepCounter,
        title: `Turn ${turnIndex} JSON Truncated`,
        detail: rawJson.slice(0, 320),
        durationMs: stepRes.decodeMs || 0,
        fsmMedianMs: fsmMs,
        decodeTokPerSec: stepTokPerSec,
        promptTokens: promptLen,
        ok: false,
      };
      traceEntries.push(traceFail);
      callbacks.onTraceEntry?.(traceFail);
      if (allowJsonRetry) {
        return runDecisionStep(
          context,
          turnIndex,
          '[Retry]: Your previous JSON action exceeded step length and was truncated. Keep code concise (under 65 lines, no comments) inside write/edit and try again.',
          effectiveSchema,
          false,
        );
      }
      return { kind: 'stop', reason: 'error', message: 'Model output exceeded single-step JSON budget.' };
    }

    const toolName = String(action.tool || 'reply');
    const thought = String(action.thought || '').trim();
    const cleanArgs = summarizeArgs(action);

    const decisionTrace: AgentTraceEntry = {
      id: `trace-dec-${turnIndex}-${Date.now()}`,
      phase: 'decision',
      step: stepCounter,
      title: `Pi FSM -> ${toolName}`,
      detail: thought ? `${thought}\n${JSON.stringify(cleanArgs).slice(0, 260)}` : JSON.stringify(cleanArgs).slice(0, 260),
      durationMs: stepRes.decodeMs || 0,
      fsmMedianMs: fsmMs,
      decodeTokPerSec: stepTokPerSec,
      promptTokens: promptLen,
      ok: true,
    };
    traceEntries.push(decisionTrace);
    callbacks.onTraceEntry?.(decisionTrace);

    if (toolName === 'reply') {
      if (
        checkpoints.length === 0 &&
        !hasPriorCheckpointsInTranscript &&
        !isReadOnlyQuestion &&
        !retriedReplyGate &&
        !extraHint
      ) {
        retriedReplyGate = true;
        return runDecisionStep(
          context,
          turnIndex,
          '[Retry]: No workspace files were modified yet. Call "write" on src/App.jsx with concise, complete application code (under 75 lines, no comments) or run "bash" to install a needed package now.',
          actionOnlySchema,
          true,
        );
      }
      return {
        kind: 'reply',
        thought,
        text: String(action.text || '').trim(),
        decodeTokPerSec: stepTokPerSec,
      };
    }

    const callId = `tc-${turnIndex}-${Date.now()}`;
    const tc: PiToolCallContent = {
      type: 'toolCall',
      id: callId,
      name: toolName,
      arguments: cleanArgs,
      thought,
      fsmMedianMs: fsmMs,
      decodeTokPerSec: stepTokPerSec,
      promptTokens: promptLen,
      durationMs: stepRes.decodeMs || 0,
    };
    actionCallMap.set(callId, tc);
    return { kind: 'toolCall', tc };
  };

  let directReplyText = '';
  let directReplyThought = '';

  const updatedTranscript = await agent.prompt(effectiveTaskPrompt, {
    maxTurns: maxSteps,
    streamFn: async (context, turnIndex) => {
      stepCounter = turnIndex;
      callbacks.onToolStepStart?.(turnIndex, maxSteps);
      const stepOutcome = await runDecisionStep(context, turnIndex);

      if (stepOutcome.kind === 'stop') {
        const stoppedMsg: PiAssistantMessage = {
          role: 'assistant',
          content: [{ type: 'text', text: stepOutcome.message }],
          stopReason: stepOutcome.reason,
          timestamp: Date.now(),
        };
        return stoppedMsg;
      }

      if (stepOutcome.kind === 'reply') {
        directReplyText = stepOutcome.text;
        directReplyThought = stepOutcome.thought;
        if (stepOutcome.thought) {
          callbacks.onStepThought?.(stepOutcome.thought, stepOutcome.decodeTokPerSec);
        }
        return {
          role: 'assistant',
          content: [{ type: 'text', text: stepOutcome.text }],
          stopReason: 'stop',
          timestamp: Date.now(),
        };
      }

      return {
        role: 'assistant',
        content: [stepOutcome.tc],
        stopReason: 'toolUse',
        timestamp: Date.now(),
      };
    },
    beforeToolCall: async ({ toolCall, args }) => {
      const name = toolCall.name;
      if (name === 'read') {
        const targetPath = normalizeWorkspacePath(String(args.path ?? ''));
        if (targetPath && freshReadPaths.has(targetPath)) {
          consecutiveRepeatedReadBlocks++;
          return {
            block: true,
            reason: `[Pi Harness]: ${targetPath} is already shown in full in the tool results above and has not changed. Do NOT call 'read' on ${targetPath} again — call 'write' on src/App.jsx with the complete new application code (or read a different unread file such as src/App.css).`,
          };
        }
        consecutiveRepeatedReadBlocks = 0;
        return;
      }
      if (name === 'bash') {
        const cmd = String(args.command ?? '').trim();
        if (cmd && freshBashCommands.has(cmd)) {
          return {
            block: true,
            reason: `[Pi Harness]: Command "${cmd}" already succeeded (exit 0) and no files changed since. Do NOT run bash again — call "reply" now.`,
          };
        }
      }
      if (name !== 'write' && name !== 'edit' && name !== 'bash') {
        consecutiveRepeatedReadBlocks = 0;
        return;
      }

      let preview = '';
      let target = '';
      let added: number | undefined;
      let removed: number | undefined;

      if (name === 'write') {
        const prep = await prepareWrite(String(args.path ?? ''), String(args.content ?? ''));
        if (!prep.ok) {
          return { block: true, reason: prep.error };
        }
        preparedMutations.set(toolCall.id, prep.payload);
        target = prep.payload.targetPath;
        preview = prep.payload.diffText;
        added = prep.payload.added;
        removed = prep.payload.removed;
      } else if (name === 'edit') {
        const prep = await prepareEdit(
          String(args.path ?? ''),
          String(args.oldText ?? ''),
          String(args.newText ?? ''),
        );
        if (!prep.ok) {
          return { block: true, reason: prep.error };
        }
        preparedMutations.set(toolCall.id, prep.payload);
        target = prep.payload.targetPath;
        preview = prep.payload.diffText;
        added = prep.payload.added;
        removed = prep.payload.removed;
      } else {
        target = 'jsh';
        preview = `$ ${String(args.command ?? '')}`;
      }

      if (approvalPolicy === 'ask' && !approveAllThisRun && callbacks.onApprovalRequest) {
        const tAppr0 = performance.now();
        const decision = await callbacks.onApprovalRequest({
          id: `appr-${stepCounter}-${Date.now()}`,
          step: stepCounter,
          tool: name,
          thought: toolCall.thought || '',
          target,
          preview,
          added,
          removed,
        });
        const apprDur = performance.now() - tAppr0;

        if (decision.approved) {
          if (decision.approveAll) approveAllThisRun = true;
          const apprTrace: AgentTraceEntry = {
            id: `trace-appr-${stepCounter}-${Date.now()}`,
            phase: 'approval',
            step: stepCounter,
            title: `Approved -> ${name} (${target})`,
            detail: approveAllThisRun ? 'Approved (all remaining for turn)' : 'Approved by user',
            durationMs: apprDur,
            ok: true,
          };
          traceEntries.push(apprTrace);
          callbacks.onTraceEntry?.(apprTrace);
        } else {
          preparedMutations.delete(toolCall.id);
          const denyReason = decision.reason || 'User denied this action.';
          const denyTrace: AgentTraceEntry = {
            id: `trace-deny-${stepCounter}-${Date.now()}`,
            phase: 'approval',
            step: stepCounter,
            title: `Denied -> ${name} (${target})`,
            detail: denyReason,
            durationMs: apprDur,
            ok: false,
          };
          traceEntries.push(denyTrace);
          callbacks.onTraceEntry?.(denyTrace);
          return { block: true, reason: `[User Denied ${name}]: ${denyReason}.` };
        }
      }
      consecutiveRepeatedReadBlocks = 0;
    },
    shouldStopAfterTurn: ({ toolResults }) =>
      checkpoints.length > 0 &&
      !lastToolError &&
      toolResults.some(
        (tr) =>
          (tr.toolName === 'bash' &&
            tr.isError &&
            Boolean(tr.content[0]?.text?.startsWith('[Pi Harness]: Command'))) ||
          (tr.toolName === 'write' &&
            tr.isError &&
            Boolean(tr.content[0]?.text?.startsWith('[Pi Harness]:'))),
      ),
    getSteeringMessages: ({ lastAssistantMessage }) => {
      if (lastAssistantMessage.stopReason !== 'stop') return [];
      const needFixAfterError = lastToolError && stepCounter < maxSteps - 1;
      const needCodeFirst = checkpoints.length === 0 && !hasPriorCheckpointsInTranscript && stepCounter < maxSteps - 1;
      if (!retriedReplyGate && (needFixAfterError || needCodeFirst)) {
        retriedReplyGate = true;
        directReplyText = '';
        const msg = needFixAfterError
          ? '[Pi Steering]: The previous tool call failed. Recover by installing any missing package via "bash" or calling "write" on src/App.jsx with valid code before replying.'
          : '[Pi Steering]: No code files were modified yet. Call "write" on src/App.jsx now to implement the application before replying.';
        return [{ role: 'user', content: msg, timestamp: Date.now() }];
      }
      return [];
    },
  });

  // Final Grounded Response (either direct rich Pi reply or grounded Markdown synthesis / 24 kHz voice)
  let finalText = '';
  let audioUrl: string | undefined;

  const needVoiceSynth = Boolean(capabilities.audioOut && generationMode !== 'text');
  const shouldUseDirectReply = !needVoiceSynth && directReplyText.length >= 80 && checkpoints.length > 0;

  if (wasAborted) {
    finalText = directReplyText || 'Stopped.';
    callbacks.onFinalToken?.(finalText, 0);
  } else if (shouldUseDirectReply) {
    finalText = directReplyText;
    callbacks.onFinalToken?.(finalText, 0);
  } else {
    callbacks.onStatusText?.(lang === 'zh' ? '正在汇总修改结果...' : 'Synthesizing engineering summary...');
    const tSynth0 = performance.now();
    await client.reset();

    const evidenceSummary = serializePiContextMessages(updatedTranscript.slice(-8), maxSeqLen);
    const lastTest = lastTestSummaryRef.current;
    const verifiedLine = lastTest
      ? `Observed test summary: ${lastTest.passed}/${lastTest.total} passed (${lastTest.failed} failed).`
      : checkpoints.length > 0
        ? `Updated files: ${[...new Set(checkpoints.map((c) => c.path))].join(', ')}.`
        : 'No workspace files were modified in this turn.';
    const synthUserPrompt = `Task:\n${userPrompt}\n\nExecuted Activity:\n${evidenceSummary}\n\n${verifiedLine}\n\nWrite a concise Markdown summary stating exact files modified and observed results.`;

    const useThinking = Boolean(capabilities.thinking && enableThinking && !needVoiceSynth);
    const synthPrompt = formatSingleTurnPrompt(archKind, {
      systemPrompt: 'You are Ignis Assistant. Summarize file changes and observed results accurately in Markdown.',
      userPrompt: synthUserPrompt,
      enableThinking: useThinking,
      effort,
    });
    const thinkPrefix = useThinking ? (archKind === 'gemma4' ? '<|thought|>' : '<think>') : '';

    let streamedSynth = '';
    client.onToken = (ev: { text: string; fullText?: string; tokPerSec?: number }) => {
      const body = ev.fullText !== undefined ? ev.fullText : (streamedSynth += ev.text || '');
      callbacks.onFinalToken?.(thinkPrefix ? `${thinkPrefix}${body}` : body, ev.tokPerSec || 0);
    };

    let synthRes;
    let synthPromptLen = 0;
    try {
      const enc = await client.encode(synthPrompt);
      synthPromptLen = enc.ids.length;
      peakPromptTokens = Math.max(peakPromptTokens, synthPromptLen);
      synthRes = await client.generate({
        promptIds: enc.ids,
        images: [],
        audios: [],
        generationMode: needVoiceSynth ? generationMode : 'text',
        maxNewTokens: Math.min(maxNewTokens, 640),
        maxThinkingTokens: archKind === 'spark' && useThinking ? (effort === 'low' ? 192 : 384) : 0,
        chunkSize: 64,
        sampling: {
          temperature: 0.3,
          topK: 40,
          topP: 0.95,
          repetitionPenalty: archKind === 'spark' ? 1.05 : 1.0,
        },
        stopOnEos: true,
      });
    } finally {
      client.onToken = null;
    }

    if (synthRes.wavBytes && synthRes.wavBytes.byteLength > 44) {
      audioUrl = URL.createObjectURL(new Blob([new Uint8Array(synthRes.wavBytes)], { type: 'audio/wav' }));
    }
    finalText = thinkPrefix
      ? `${thinkPrefix}${synthRes.generatedText || ''}`
      : synthRes.generatedText || directReplyText || 'Completed.';
    totalPromptTokens += synthRes.promptLen || synthPromptLen;
    totalGeneratedTokens += synthRes.generatedIds.length;
    totalDecodeMs += synthRes.decodeMs || performance.now() - tSynth0;
    totalPrefillMs += synthRes.prefillMs || 0;
    lastCurPos = synthRes.curPos || lastCurPos;
    wasAborted = wasAborted || Boolean(synthRes.aborted);

    const synthTrace: AgentTraceEntry = {
      id: `trace-synth-${Date.now()}`,
      phase: 'synthesis',
      step: toolCalls.length + 1,
      title: needVoiceSynth ? 'Final Synthesis + 24 kHz Voice' : 'Final Engineering Summary',
      detail: `${synthRes.generatedIds.length} tokens @ ${synthRes.decodeTokPerSec.toFixed(1)} tok/s`,
      durationMs: performance.now() - tSynth0,
      promptTokens: synthPromptLen,
      ok: true,
    };
    traceEntries.push(synthTrace);
    callbacks.onTraceEntry?.(synthTrace);
  }

  return {
    finalText,
    replyThought: directReplyThought || undefined,
    toolCalls,
    traceEntries,
    workspaceFiles: workspace,
    checkpoints,
    transcriptMessages: updatedTranscript,
    testSummary: lastTestSummaryRef.current,
    audioUrl,
    totalPromptTokens,
    totalGeneratedTokens,
    peakPromptTokens,
    maxSeqLen,
    decodeTokPerSec: totalDecodeMs > 0 ? totalGeneratedTokens / (totalDecodeMs / 1000) : 0,
    prefillMs: totalPrefillMs,
    curPos: lastCurPos,
    aborted: wasAborted,
  };
}
