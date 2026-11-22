import {
  javascriptLanguage,
  jsxLanguage,
  tsxLanguage,
  typescriptLanguage,
} from '@codemirror/lang-javascript';
import type { IgnisClient } from '../../../runtime/ignis-client.mjs';
import { computeLineDiff, formatUnifiedDiff } from './diff';
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
  grepWorkspaceFiles,
  normalizeWorkspacePath,
  readWorkspaceFile,
  upsertWorkspaceFile,
  type WorkspaceCheckpoint,
  type WorkspaceFile,
} from './workspace-fs';

export type AgentToolName =
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
  tool: 'edit_file' | 'write_file' | 'run_command';
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
  toolCalls: AgentToolCallRecord[];
  traceEntries: AgentTraceEntry[];
  workspaceFiles: WorkspaceFile[];
  checkpoints: WorkspaceCheckpoint[];
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

export function buildAgentActionSchema(): Record<string, unknown> {
  return {
    oneOf: [
      {
        type: 'object',
        properties: {
          thought: { type: 'string' },
          tool: { const: 'list_files' },
          dir: { type: 'string' },
        },
        required: ['thought', 'tool', 'dir'],
        additionalProperties: false,
      },
      {
        type: 'object',
        properties: {
          thought: { type: 'string' },
          tool: { const: 'read_file' },
          path: { type: 'string' },
        },
        required: ['thought', 'tool', 'path'],
        additionalProperties: false,
      },
      {
        type: 'object',
        properties: {
          thought: { type: 'string' },
          tool: { const: 'grep_files' },
          pattern: { type: 'string' },
        },
        required: ['thought', 'tool', 'pattern'],
        additionalProperties: false,
      },
      {
        type: 'object',
        properties: {
          thought: { type: 'string' },
          tool: { const: 'edit_file' },
          path: { type: 'string' },
          old_text: { type: 'string' },
          new_text: { type: 'string' },
        },
        required: ['thought', 'tool', 'path', 'old_text', 'new_text'],
        additionalProperties: false,
      },
      {
        type: 'object',
        properties: {
          thought: { type: 'string' },
          tool: { const: 'write_file' },
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
          tool: { const: 'run_command' },
          command: { type: 'string' },
        },
        required: ['thought', 'tool', 'command'],
        additionalProperties: false,
      },
      {
        type: 'object',
        properties: {
          thought: { type: 'string' },
          tool: { const: 'finish' },
          summary: { type: 'string' },
        },
        required: ['thought', 'tool', 'summary'],
        additionalProperties: false,
      },
    ],
  };
}

function median(values?: number[]): number | null {
  if (!values || values.length === 0) return null;
  const s = [...values].sort((a, b) => a - b);
  return s[Math.floor(s.length / 2)];
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

/** Validate JS/TS/JSON syntax locally in-browser via Lezer / JSON.parse without emitting WebContainer stderr. */
function validateCodeSyntax(targetPath: string, content: string): string | null {
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

  const tree = lang.parser.parse(content);
  let errFrom: number | null = null;
  tree.iterate({
    enter(node) {
      if (node.type.isError) {
        errFrom = node.from;
        return false;
      }
    },
  });
  if (errFrom === null) return null;

  const linesBefore = content.slice(0, errFrom).split('\n');
  const errLine = linesBefore.length;
  const errCol = linesBefore[linesBefore.length - 1].length + 1;
  const snippet = content.split('\n')[errLine - 1] ?? '';
  const pointer = `${' '.repeat(Math.max(0, errCol - 1))}^`;
  return `${targetPath}:${errLine}:${errCol}\n${snippet}\n${pointer}\nSyntaxError: Unexpected token or unbalanced syntax at line ${errLine}, column ${errCol}`;
}

function buildAgentSystemPrompt(workspace: WorkspaceFile[], lang: 'en' | 'zh'): string {
  const langHint =
    lang === 'zh'
      ? 'Write explanations in Chinese when the user writes in Chinese; keep file paths, code, and commands exact.'
      : 'Keep responses concise and grounded in observed tool outputs.';

  return `You are Ignis Code, a disciplined full-stack software engineering agent running locally inside a WebContainer sandbox.
Output ONE valid JSON action per turn matching the schema.

Tools:
- run_command {"thought":"...","tool":"run_command","command":"..."}: Run real Node/npm/shell commands (e.g. "npm test", "node --test test/api.test.mjs", "ls -la").
- read_file {"thought":"...","tool":"read_file","path":"..."}: Read a project file with line numbers.
- grep_files {"thought":"...","tool":"grep_files","pattern":"..."}: Search regex/substring across project files.
- list_files {"thought":"...","tool":"list_files","dir":"."}: List files in workspace.
- edit_file {"thought":"...","tool":"edit_file","path":"...","old_text":"...","new_text":"..."}: Replace an exact existing snippet cleanly (prefer over write_file for edits). Copy exact lines from read_file without line-number prefixes.
- write_file {"thought":"...","tool":"write_file","path":"...","content":"..."}: Create a new file or write complete updated file content.
- finish {"thought":"...","tool":"finish","summary":"..."}: Finish after verifying tests pass.

Examples:
- {"thought":"Run the test suite to reproduce failures","tool":"run_command","command":"npm test"}
- {"thought":"Inspect module implementation","tool":"read_file","path":"server/store.mjs"}
- {"thought":"Fix comparison boundary check","tool":"edit_file","path":"lib/helper.mjs","old_text":"if (count > limit) return false;","new_text":"if (count >= limit) return false;"}
- {"thought":"Verified all tests pass after edits","tool":"finish","summary":"Patched bug and confirmed npm test passes."}

Project files:
${formatWorkspaceTree(workspace)}

Strict engineering rules:
1. Reproduce issues or inspect relevant files first using run_command / read_file. Never guess file contents.
2. When editing existing code, use edit_file with exact verbatim old_text (without line-number prefixes like "12 | ").
3. After ANY code edit, re-run the test suite with run_command (e.g. "npm test") before calling finish.
4. Never fabricate test passes or command outputs. ${langHint}`;
}

interface ObservationItem {
  step: number;
  tool: string;
  filePath?: string;
  isAutoRead?: boolean;
  ok: boolean;
  full: string;
  compact: string;
}

function buildCompactObservationsBlock(
  observations: ObservationItem[],
  maxRecentDetail = 5,
  charBudget = 7200,
): string {
  if (observations.length === 0) return '';
  // Keep latest read_file for up to 5 distinct files plus latest run_command test output
  const keptReadIndices: number[] = [];
  const seenReadPaths = new Set<string>();
  let lastCmdIdx = -1;
  for (let i = observations.length - 1; i >= 0; i--) {
    const obs = observations[i];
    if (lastCmdIdx === -1 && obs.tool === 'run_command') {
      lastCmdIdx = i;
    }
    if (keptReadIndices.length < 5 && ((obs.tool === 'read_file' && obs.ok) || obs.isAutoRead)) {
      const key = obs.filePath || String(i);
      if (!seenReadPaths.has(key)) {
        seenReadPaths.add(key);
        keptReadIndices.push(i);
      }
    }
  }
  const keptReadSet = new Set(keptReadIndices);
  if (lastCmdIdx !== -1) {
    keptReadSet.add(lastCmdIdx);
  }

  const selected = observations.map((obs, i) => {
    const isRecent = i >= observations.length - maxRecentDetail || keptReadSet.has(i);
    return isRecent ? obs.full : obs.compact;
  });
  let joined = selected.join('\n\n');
  let shrinkIdx = 0;
  while (joined.length > charBudget && shrinkIdx < selected.length - 1) {
    if (!keptReadSet.has(shrinkIdx)) {
      selected[shrinkIdx] = observations[shrinkIdx].compact;
      joined = selected.join('\n\n');
    }
    shrinkIdx++;
  }
  for (let k = keptReadIndices.length - 1; k >= 0 && joined.length > charBudget; k--) {
    const idx = keptReadIndices[k];
    if (idx < observations.length - 2) {
      selected[idx] = observations[idx].compact;
      joined = selected.join('\n\n');
    }
  }
  return joined;
}

export async function runAgentTurn(opts: {
  client: IgnisClient;
  archKind: ArchKind;
  maxSeqLen: number;
  capabilities: { vision: boolean; audioIn: boolean; audioOut: boolean; thinking: boolean };
  userPrompt: string;
  historyText?: string;
  workspaceFiles: WorkspaceFile[];
  sandbox: CodingSandboxAdapter;
  approvalPolicy?: ApprovalPolicy;
  attachedImage?: PendingImageInput | null;
  attachedAudio?: PendingAudioInput | null;
  generationMode: OmniGenMode;
  maxNewTokens: number;
  maxSteps?: number;
  lang: 'en' | 'zh';
  enableThinking?: boolean;
  effort?: 'low' | 'high';
  callbacks?: AgentTurnCallbacks;
}): Promise<AgentTurnResult> {
  const {
    client,
    archKind,
    maxSeqLen,
    capabilities,
    userPrompt,
    historyText = '',
    sandbox,
    approvalPolicy = 'auto',
    attachedImage = null,
    attachedAudio = null,
    generationMode,
    maxNewTokens,
    maxSteps = 8,
    lang,
    enableThinking = false,
    effort = 'low',
    callbacks = {},
  } = opts;

  let workspace = [...opts.workspaceFiles];
  const toolCalls: AgentToolCallRecord[] = [];
  const traceEntries: AgentTraceEntry[] = [];
  const checkpoints: WorkspaceCheckpoint[] = [];
  const seenSignatures = new Set<string>();
  const readPaths = new Set<string>();
  const observations: ObservationItem[] = [];

  let totalPromptTokens = 0;
  let totalGeneratedTokens = 0;
  let peakPromptTokens = 0;
  let totalDecodeMs = 0;
  let totalPrefillMs = 0;
  let lastCurPos = 0;
  let wasAborted = false;
  let lastTestSummary: NodeTestSummary | null = null;
  let editedSinceLastTest = false;
  let approveAllThisRun = false;

  const schema = buildAgentActionSchema();

  // Step 0: Multimodal Perception when user attached UI screenshot or voice memo
  if ((attachedImage && capabilities.vision) || (attachedAudio && capabilities.audioIn)) {
    const tPerc0 = performance.now();
    await client.reset();
    const percPrompt = formatSingleTurnPrompt(archKind, {
      systemPrompt:
        'Describe the attached software screenshot or voice instruction concisely so a coding agent can inspect files and fix or implement it.',
      userPrompt,
      includeImageTag: Boolean(attachedImage && capabilities.vision),
      includeAudioTag: Boolean(attachedAudio && capabilities.audioIn),
    });
    const enc = await client.encode(percPrompt);
    const percRes = await client.generate({
      promptIds: enc.ids,
      images:
        attachedImage && capabilities.vision
          ? [{ data: attachedImage.data, width: attachedImage.width, height: attachedImage.height, channels: attachedImage.channels }]
          : [],
      audios: attachedAudio && capabilities.audioIn ? [attachedAudio.pcm16k] : [],
      generationMode: 'text',
      maxNewTokens: 160,
      chunkSize: 64,
      sampling: { temperature: 0, topK: 1, topP: 1.0, repetitionPenalty: 1.0 },
      stopOnEos: true,
    });
    const dur = performance.now() - tPerc0;
    const obsText = (percRes.generatedText || '').trim();
    const percTokens = percRes.promptLen || enc.ids.length;
    totalPromptTokens += percTokens;
    peakPromptTokens = Math.max(peakPromptTokens, percTokens);
    totalGeneratedTokens += percRes.generatedIds.length;
    totalDecodeMs += percRes.decodeMs || 0;
    totalPrefillMs += percRes.prefillMs || 0;
    lastCurPos = percRes.curPos || lastCurPos;
    wasAborted = Boolean(percRes.aborted);

    observations.push({
      step: 0,
      tool: 'perception',
      ok: Boolean(obsText),
      full: `[Attached Media Perception]: ${obsText}`,
      compact: `[Attached Media Perception]: ${obsText.slice(0, 220)}`,
    });
    const entry: AgentTraceEntry = {
      id: `trace-perc-${Date.now()}`,
      phase: 'perception',
      step: 0,
      title: 'Multimodal Perception',
      detail: obsText || '(No observation produced)',
      durationMs: dur,
      promptTokens: percTokens,
      ok: Boolean(obsText),
    };
    traceEntries.push(entry);
    callbacks.onTraceEntry?.(entry);
  }

  let finishSummary = '';

  // Steps 1..maxSteps: Schema-Locked Tool Selection & WebContainer Execution
  for (let step = 1; step <= maxSteps && !wasAborted; step++) {
    callbacks.onToolStepStart?.(step, maxSteps);
    const sysPrompt = buildAgentSystemPrompt(workspace, lang);
    const historyBlock = historyText ? `Prior context:\n${historyText}\n\n` : '';

    let nudge = '';
    if (toolCalls.length > 0) {
      const seq = toolCalls.map((c) => `${c.tool}${c.ok ? '' : '(err)'}`).join(' -> ');
      nudge = `\nExecuted so far: ${seq}.`;
      const failingNames = lastTestSummary
        ? lastTestSummary.cases.filter((c) => !c.ok).map((c) => c.name)
        : [];
      if (editedSinceLastTest) {
        nudge += ' File edited — run "npm test" via run_command next to verify.';
      } else if (lastTestSummary && lastTestSummary.failed > 0) {
        const failDetail = failingNames.length > 0 ? `: ${failingNames.join('; ')}` : '';
        nudge += ` Still failing (${lastTestSummary.failed} of ${lastTestSummary.total}${failDetail}) — check the exact endpoint/function tested by the failure and patch it cleanly with edit_file.`;
        if (readPaths.size >= 2) {
          nudge += ` Already read: [${[...readPaths].join(', ')}] — do NOT call read_file on these again; call edit_file now to patch the bug.`;
        }
      }
    }

    // Fit observations strictly within model maxSeqLen budget
    let obsBudget = maxSeqLen <= 4096 ? 7200 : 15500;
    let obsTextBlock = buildCompactObservationsBlock(observations, maxSeqLen <= 4096 ? 4 : 6, obsBudget);
    const buildStepPrompt = (obsStr: string) => {
      const obsSection = obsStr
        ? `\n\nObservations:\n${obsStr}${nudge}\n\nChoose the next JSON tool action (or call "finish" once verified):`
        : '\n\nChoose your first JSON tool action:';
      return formatSingleTurnPrompt(archKind, {
        systemPrompt: sysPrompt,
        userPrompt: `${historyBlock}Task:\n${userPrompt}${obsSection}`,
        enableThinking: false,
      });
    };

    let fullPrompt = buildStepPrompt(obsTextBlock);
    let enc = await client.encode(fullPrompt);
    while (enc.ids.length > maxSeqLen - 850 && obsBudget > 1000) {
      obsBudget = Math.floor(obsBudget * 0.6);
      obsTextBlock = buildCompactObservationsBlock(observations, 1, obsBudget);
      fullPrompt = buildStepPrompt(obsTextBlock);
      enc = await client.encode(fullPrompt);
    }

    peakPromptTokens = Math.max(peakPromptTokens, enc.ids.length);
    await client.reset();
    const tDecStart = performance.now();
    const stepRes = await client.generate({
      promptIds: enc.ids,
      images: [],
      audios: [],
      generationMode: 'text',
      maxNewTokens: Math.min(820, Math.max(256, maxSeqLen - enc.ids.length - 32)),
      chunkSize: 64,
      sampling: { temperature: 0, topK: 1, topP: 1.0, repetitionPenalty: 1.0 },
      responseFormat: {
        type: 'json_schema',
        json_schema: { name: 'ignis_coding_action', schema, strict: true },
      },
      stopOnEos: true,
    });

    const promptLen = stepRes.promptLen || enc.ids.length;
    totalPromptTokens += promptLen;
    peakPromptTokens = Math.max(peakPromptTokens, promptLen);
    totalGeneratedTokens += stepRes.generatedIds.length;
    totalDecodeMs += stepRes.decodeMs || performance.now() - tDecStart;
    totalPrefillMs += stepRes.prefillMs || 0;
    lastCurPos = stepRes.curPos || lastCurPos;
    wasAborted = Boolean(stepRes.aborted);

    const fsmMs = median(stepRes.fsmMaskMsSamples);
    const rawJson = splitThinking(stepRes.generatedText || '').answer.trim();

    let action: Record<string, unknown> | null = null;
    try {
      action = JSON.parse(rawJson) as Record<string, unknown>;
    } catch {
      const traceFail: AgentTraceEntry = {
        id: `trace-dec-err-${step}-${Date.now()}`,
        phase: 'decision',
        step,
        title: `Step ${step} JSON Decode Fallback`,
        detail: rawJson.slice(0, 320),
        durationMs: stepRes.decodeMs || 0,
        fsmMedianMs: fsmMs,
        promptTokens: promptLen,
        ok: false,
      };
      traceEntries.push(traceFail);
      callbacks.onTraceEntry?.(traceFail);
      break;
    }

    const toolName = String(action.tool || 'finish') as AgentToolName;
    const thought = String(action.thought || '').trim();
    const cleanArgs = summarizeArgs(action);
    const sig = `${toolName}:${JSON.stringify(cleanArgs)}`;

    const decisionTrace: AgentTraceEntry = {
      id: `trace-dec-${step}-${Date.now()}`,
      phase: 'decision',
      step,
      title: `FSM -> ${toolName}`,
      detail: thought ? `${thought}\n${JSON.stringify(cleanArgs)}` : JSON.stringify(cleanArgs),
      durationMs: stepRes.decodeMs || 0,
      fsmMedianMs: fsmMs,
      promptTokens: promptLen,
      ok: true,
    };
    traceEntries.push(decisionTrace);
    callbacks.onTraceEntry?.(decisionTrace);

    if (toolName === 'finish') {
      const stillFailing = Boolean(lastTestSummary && lastTestSummary.failed > 0);
      if ((editedSinceLastTest || stillFailing) && step < maxSteps && !seenSignatures.has('finish-gate')) {
        seenSignatures.add('finish-gate');
        observations.push({
          step,
          tool: 'finish',
          ok: false,
          full: `[Harness Verification Gate]: Cannot finish yet (${
            stillFailing
              ? `${lastTestSummary!.passed}/${lastTestSummary!.total} tests passed, ${lastTestSummary!.failed} STILL FAILING`
              : 'edits were not verified with npm test'
          }). Fix the remaining bug(s) using edit_file and run "npm test" via run_command.`,
          compact: '[Gate]: Tests not all passing; continue fixing & testing.',
        });
        continue;
      }
      finishSummary = String(action.summary || '').trim();
      break;
    }
    seenSignatures.delete('finish-gate');

    if (seenSignatures.has(sig)) {
      if (toolName === 'read_file') {
        const targetPath = normalizeWorkspacePath(String(action.path || ''));
        const f = readWorkspaceFile(workspace, targetPath);
        if (f) {
          const numbered = formatNumberedContent(f.content, 1, undefined, 130);
          observations.push({
            step,
            tool: toolName,
            filePath: targetPath,
            isAutoRead: true,
            ok: false,
            full: `[Harness Guard]: ${targetPath} was already read (content shown below). Do NOT call read_file again — call edit_file now with exact old_text and new_text:\n${numbered.text}`,
            compact: `[Guard]: ${targetPath} already read — call edit_file now.`,
          });
          if (step >= maxSteps - 1) break;
          continue;
        }
      }
      observations.push({
        step,
        tool: toolName,
        ok: false,
        full: `[Harness Guard]: Tool ${toolName} with identical arguments was already executed. Choose a different file, command, or edit.`,
        compact: `[Guard]: Repeated ${toolName} skipped.`,
      });
      if (step >= maxSteps - 1) break;
      continue;
    }
    seenSignatures.add(sig);

    // Pre-check HITL Approval Gate for side-effecting tools
    // Skip prompting when Guardrail A will intercept an unread edit_file into an auto-read
    const targetEditPath = toolName === 'edit_file' ? normalizeWorkspacePath(String(action.path || '')) : '';
    const unreadBeforeEdit = toolName === 'edit_file' && !readPaths.has(targetEditPath);
    let deniedReason: string | null = null;
    if (
      approvalPolicy === 'ask' &&
      !approveAllThisRun &&
      callbacks.onApprovalRequest &&
      ((toolName === 'edit_file' && !unreadBeforeEdit) ||
        toolName === 'write_file' ||
        toolName === 'run_command')
    ) {
      let target = '';
      let preview = '';
      let added = 0;
      let removed = 0;

      if (toolName === 'edit_file') {
        target = normalizeWorkspacePath(String(action.path || ''));
        const f = readWorkspaceFile(workspace, target);
        if (f) {
          const sim = applyEditPatch(
            target,
            f.content,
            String(action.old_text || ''),
            String(action.new_text || ''),
          );
          if (sim.ok) {
            preview = sim.diffText;
            added = sim.added;
            removed = sim.removed;
          } else {
            preview = `old_text:\n${String(action.old_text || '')}\n\nnew_text:\n${String(action.new_text || '')}`;
          }
        } else {
          preview = `Missing file: ${target}`;
        }
      } else if (toolName === 'write_file') {
        target = normalizeWorkspacePath(String(action.path || ''));
        const existing = readWorkspaceFile(workspace, target);
        const nextContent = String(action.content || '');
        preview = existing
          ? formatUnifiedDiff(target, existing.content, nextContent)
          : nextContent.slice(0, 1200);
      } else {
        target = 'jsh';
        preview = `$ ${String(action.command || '')}`;
      }

      const tAppr0 = performance.now();
      const decision = await callbacks.onApprovalRequest({
        id: `appr-${step}-${Date.now()}`,
        step,
        tool: toolName,
        thought,
        target,
        preview,
        added,
        removed,
      });
      const apprDur = performance.now() - tAppr0;

      if (decision.approved) {
        if (decision.approveAll) approveAllThisRun = true;
        const apprTrace: AgentTraceEntry = {
          id: `trace-appr-${step}-${Date.now()}`,
          phase: 'approval',
          step,
          title: `Approved -> ${toolName} (${target})`,
          detail: approveAllThisRun ? 'Approved (all remaining for turn)' : 'Approved by user',
          durationMs: apprDur,
          ok: true,
        };
        traceEntries.push(apprTrace);
        callbacks.onTraceEntry?.(apprTrace);
      } else {
        deniedReason = decision.reason || 'User denied this action.';
        const denyTrace: AgentTraceEntry = {
          id: `trace-deny-${step}-${Date.now()}`,
          phase: 'approval',
          step,
          title: `Denied -> ${toolName} (${target})`,
          detail: deniedReason,
          durationMs: apprDur,
          ok: false,
        };
        traceEntries.push(denyTrace);
        callbacks.onTraceEntry?.(denyTrace);
      }
    }

    // Execute Tool
    const tTool0 = performance.now();
    let ok = true;
    let output = '';
    let compactOut = '';
    let diffText: string | undefined;
    let added: number | undefined;
    let removed: number | undefined;
    let exitCode: number | undefined;
    let testSummary: NodeTestSummary | null = null;
    let checkpointId: string | undefined;
    let isAutoRead = false;

    if (deniedReason !== null) {
      ok = false;
      output = `[User Denied ${toolName}]: ${deniedReason}. Adjust your approach or call finish.`;
      compactOut = `[Denied ${toolName}]: ${deniedReason}`;
    } else if (toolName === 'list_files') {
      if (sandbox.listFiles) {
        workspace = await sandbox.listFiles();
        callbacks.onWorkspaceUpdate?.(workspace);
      }
      const dirPrefix = normalizeWorkspacePath(String(action.dir || ''));
      const matched = workspace.filter((f) => !dirPrefix || dirPrefix === '.' || f.path.startsWith(dirPrefix));
      output = formatWorkspaceTree(matched) || '(empty directory)';
      compactOut = `Listed ${matched.length} files`;
    } else if (toolName === 'read_file') {
      const targetPath = normalizeWorkspacePath(String(action.path || ''));
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
        ok = false;
        output = `File "${targetPath}" not found. Project files:\n${formatWorkspaceTree(workspace)}`;
        compactOut = `read_file "${targetPath}" missing`;
      } else {
        readPaths.add(normalizeWorkspacePath(f.path));
        const numbered = formatNumberedContent(f.content, 1, undefined, 130);
        output = `${f.path} (lines ${numbered.shownFrom}-${numbered.shownTo} of ${numbered.totalLines}):\n${numbered.text}`;
        compactOut = `Read ${f.path} (${numbered.totalLines} lines)`;
        callbacks.onOpenFile?.(f.path, 'code');
      }
    } else if (toolName === 'grep_files') {
      const pat = String(action.pattern || '');
      const hits = grepWorkspaceFiles(workspace, pat);
      output = hits.length > 0 ? hits.join('\n') : `No matches found for "${pat}".`;
      compactOut = `grep "${pat}": ${hits.length} hit(s)`;
    } else if (toolName === 'edit_file') {
      const targetPath = normalizeWorkspacePath(String(action.path || ''));
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
        ok = false;
        output = `Cannot edit "${targetPath}": file not found.`;
        compactOut = `edit_file "${targetPath}" missing`;
      } else if (!readPaths.has(targetPath)) {
        // Guardrail A: force reading unread files before accepting blind patches
        readPaths.add(targetPath);
        seenSignatures.delete(sig);
        isAutoRead = true;
        const numbered = formatNumberedContent(existing.content, 1, undefined, 200);
        ok = false;
        output = `[Harness Guard]: You must read ${targetPath} before editing it. Here is the current content of ${targetPath} (lines ${numbered.shownFrom}-${numbered.shownTo} of ${numbered.totalLines}):\n${numbered.text}\nNow call edit_file with exact old_text.`;
        compactOut = `Auto-read ${targetPath} before edit`;
        callbacks.onOpenFile?.(targetPath, 'code');
      } else {
        const patchRes = applyEditPatch(
          targetPath,
          existing.content,
          String(action.old_text || ''),
          String(action.new_text || ''),
        );
        if (!patchRes.ok) {
          ok = false;
          output = patchRes.error;
          compactOut = `edit_file "${targetPath}" failed to match`;
        } else {
          const syntaxErr = validateCodeSyntax(targetPath, patchRes.updatedContent);

          if (syntaxErr !== null) {
            // Guardrail B: reject syntax-breaking edits and keep original file intact
            const currentNumbered = formatNumberedContent(existing.content, 1, undefined, 130);
            ok = false;
            output = `SyntaxError introduced in ${targetPath} — edit was rolled back automatically:\n${syntaxErr}\n\nCurrent ${targetPath} content:\n${currentNumbered.text}\n\nFix the syntax in your next edit_file call.`;
            compactOut = `Syntax error in ${targetPath} (auto-reverted)`;
          } else {
            await sandbox.writeFile(targetPath, patchRes.updatedContent);
            workspace = upsertWorkspaceFile(workspace, targetPath, patchRes.updatedContent, true, {
              addedLines: (existing.addedLines || 0) + patchRes.added,
              removedLines: (existing.removedLines || 0) + patchRes.removed,
            });
            callbacks.onWorkspaceUpdate?.(workspace);

            checkpointId = `ckpt-${step}-${Date.now()}`;
            const ckpt: WorkspaceCheckpoint = {
              id: checkpointId,
              step,
              tool: 'edit_file',
              path: targetPath,
              existedBefore: true,
              beforeContent: existing.content,
              afterContent: patchRes.updatedContent,
              added: patchRes.added,
              removed: patchRes.removed,
              diffText: patchRes.diffText,
              createdAt: Date.now(),
            };
            checkpoints.push(ckpt);
            callbacks.onCheckpoint?.(ckpt);
            callbacks.onOpenFile?.(targetPath, 'diff');

            diffText = patchRes.diffText;
            added = patchRes.added;
            removed = patchRes.removed;
            editedSinceLastTest = true;

            // Clear stale read/command cache signatures since source changed
            for (const s of [...seenSignatures]) {
              if (s.startsWith('run_command:') || s.startsWith('read_file:')) {
                seenSignatures.delete(s);
              }
            }

            // Refresh any cached read_file observation of this path so the model sees current lines
            const updatedNumbered = formatNumberedContent(patchRes.updatedContent, 1, undefined, 130);
            for (const prevObs of observations) {
              if ((prevObs.tool === 'read_file' || prevObs.isAutoRead) && prevObs.filePath === targetPath) {
                prevObs.full = `[Current ${targetPath} after Step ${step} edit (${updatedNumbered.totalLines} lines)]:\n${updatedNumbered.text}`;
              }
            }

            output = `Patched ${targetPath} (+${patchRes.added}/-${patchRes.removed}):\n${patchRes.diffText}`;
            compactOut = `Patched ${targetPath} (+${patchRes.added}/-${patchRes.removed})`;
          }
        }
      }
    } else if (toolName === 'write_file') {
      const targetPath = normalizeWorkspacePath(String(action.path || 'untitled.txt'));
      const content = String(action.content || '');
      const existing = readWorkspaceFile(workspace, targetPath);
      const before = existing ? existing.content : '';
      const diffCounts = computeLineDiff(before, content);
      const computedDiff = formatUnifiedDiff(targetPath, before, content);

      await sandbox.writeFile(targetPath, content);
      readPaths.add(targetPath);
      workspace = upsertWorkspaceFile(workspace, targetPath, content, true, {
        addedLines: diffCounts.added,
        removedLines: diffCounts.removed,
      });
      callbacks.onWorkspaceUpdate?.(workspace);

      checkpointId = `ckpt-${step}-${Date.now()}`;
      const ckpt: WorkspaceCheckpoint = {
        id: checkpointId,
        step,
        tool: 'write_file',
        path: targetPath,
        existedBefore: Boolean(existing),
        beforeContent: before,
        afterContent: content,
        added: diffCounts.added,
        removed: diffCounts.removed,
        diffText: computedDiff,
        createdAt: Date.now(),
      };
      checkpoints.push(ckpt);
      callbacks.onCheckpoint?.(ckpt);
      callbacks.onOpenFile?.(targetPath, existing ? 'diff' : 'code');

      diffText = computedDiff;
      added = diffCounts.added;
      removed = diffCounts.removed;
      editedSinceLastTest = true;
      for (const s of [...seenSignatures]) {
        if (s.startsWith('run_command:') || s.startsWith('read_file:')) {
          seenSignatures.delete(s);
        }
      }

      output = `Wrote ${content.length} chars to ${targetPath}`;
      compactOut = `Wrote ${targetPath} (${content.length} chars)`;
    } else if (toolName === 'run_command') {
      const cmd = String(action.command || '').trim() || 'npm test';
      const cmdRes = await sandbox.runCommand(cmd);
      exitCode = cmdRes.exitCode;
      ok = cmdRes.exitCode === 0;
      const cleanOut = stripAnsi(cmdRes.output);
      testSummary = cmdRes.testSummary || parseNodeTestOutput(cleanOut);
      if (testSummary) {
        lastTestSummary = testSummary;
        editedSinceLastTest = false;
      }

      const compactOutText = compactCommandOutput(cleanOut);
      output = `$ ${cmd} (exit ${cmdRes.exitCode}, ${Math.round(cmdRes.durationMs)}ms)\n${compactOutText || '(no output)'}`;
      compactOut = testSummary
        ? `$ ${cmd} -> exit ${cmdRes.exitCode} (${testSummary.passed}/${testSummary.total} passed)`
        : `$ ${cmd} -> exit ${cmdRes.exitCode}`;
    }

    const toolDur = performance.now() - tTool0;
    const record: AgentToolCallRecord = {
      id: `tc-${step}-${Date.now()}`,
      step,
      tool: toolName,
      thought,
      args: cleanArgs,
      output,
      ok,
      durationMs: toolDur,
      fsmMedianMs: fsmMs,
      diffText,
      added,
      removed,
      exitCode,
      testSummary,
      checkpointId,
    };
    toolCalls.push(record);
    callbacks.onToolStepComplete?.(record);

    const toolTrace: AgentTraceEntry = {
      id: `trace-tool-${step}-${Date.now()}`,
      phase: 'tool',
      step,
      title: `${toolName} (${ok ? 'ok' : 'err'})`,
      detail: output.slice(0, 700),
      durationMs: toolDur,
      fsmMedianMs: fsmMs,
      promptTokens: promptLen,
      ok,
    };
    traceEntries.push(toolTrace);
    callbacks.onTraceEntry?.(toolTrace);

    const obsCap = toolName === 'read_file' || isAutoRead ? 5600 : 2400;
    const obsFilePath =
      toolName === 'read_file' || toolName === 'edit_file' || toolName === 'write_file'
        ? normalizeWorkspacePath(String(action.path || ''))
        : undefined;
    observations.push({
      step,
      tool: toolName,
      filePath: obsFilePath,
      isAutoRead,
      ok,
      full: `[Step ${step} ${toolName} -> ${ok ? 'OK' : 'ERROR'}]:\n${output.slice(0, obsCap)}`,
      compact: `[Step ${step} ${toolName} -> ${ok ? 'OK' : 'ERROR'}]: ${compactOut}`,
    });
  }

  // Final Synthesis Stage
  let finalText = '';
  let audioUrl: string | undefined;

  const needVoiceSynth = Boolean(capabilities.audioOut && generationMode !== 'text');
  const shouldSynthesize =
    !wasAborted && (toolCalls.length > 0 || needVoiceSynth || finishSummary.length < 140);

  if (shouldSynthesize) {
    callbacks.onStatusText?.(lang === 'zh' ? '正在汇总修改与测试结果...' : 'Synthesizing engineering summary...');
    const tSynth0 = performance.now();
    await client.reset();

    const evidenceBlock =
      observations.length > 0
        ? `\n\nObserved Tool Execution Results:\n${buildCompactObservationsBlock(observations, 4, maxSeqLen <= 4096 ? 4800 : 8500)}`
        : '';
    const verifiedLine = lastTestSummary
      ? `Verified latest test result: ${lastTestSummary.passed}/${lastTestSummary.total} tests passed (${lastTestSummary.failed} failed). CRITICAL: NEVER claim tests passed if failed > 0.`
      : 'Verified latest test result: no test suite summary observed.';
    const finishNote = finishSummary ? `\n\nAgent Draft Summary:\n${finishSummary}` : '';
    const synthUserPrompt = `${historyText ? `Prior context:\n${historyText}\n\n` : ''}Original task:\n${userPrompt}${evidenceBlock}\n\n${verifiedLine}${finishNote}\n\nWrite a concise Markdown engineering summary stating exact files changed and exact observed test results.`;

    const useThinking = Boolean(capabilities.thinking && enableThinking && !needVoiceSynth);
    const synthPrompt = formatSingleTurnPrompt(archKind, {
      systemPrompt:
        'You are Ignis Code Assistant. Summarize code changes and observed test/command results accurately in Markdown.',
      userPrompt: synthUserPrompt,
      enableThinking: useThinking,
      effort,
    });
    const thinkPrefix = useThinking ? (archKind === 'gemma4' ? '<|thought|>' : '<think>') : '';

    client.onToken = (ev: { text: string; fullText?: string; tokPerSec?: number }) => {
      const body = ev.fullText !== undefined ? ev.fullText : ev.text;
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
        maxNewTokens: Math.min(maxNewTokens, 768),
        maxThinkingTokens: archKind === 'spark' && useThinking ? (effort === 'low' ? 256 : 512) : 0,
        chunkSize: 64,
        sampling: {
          temperature: 0.35,
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
      : synthRes.generatedText || finishSummary;
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
  } else {
    finalText = finishSummary || 'Task completed.';
    callbacks.onFinalToken?.(finalText, 0);
  }

  return {
    finalText,
    toolCalls,
    traceEntries,
    workspaceFiles: workspace,
    checkpoints,
    testSummary: lastTestSummary,
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
