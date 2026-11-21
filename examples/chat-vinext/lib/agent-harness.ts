import type { IgnisClient } from '../../../runtime/ignis-client.mjs';
import { formatSingleTurnPrompt, splitThinking, type ArchKind, type OmniGenMode } from './prompt-templates';
import { executeSandboxedJs } from './sandbox-js';
import {
  formatWorkspaceTree,
  normalizeWorkspacePath,
  readWorkspaceFile,
  upsertWorkspaceFile,
  type WorkspaceFile,
} from './workspace-fs';

export type AgentToolName =
  | 'read_file'
  | 'write_file'
  | 'run_js'
  | 'render_artifact'
  | 'inspect_image'
  | 'finish';

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
}

export interface AgentArtifact {
  id: string;
  title: string;
  html: string;
  createdAt: number;
}

export interface AgentTraceEntry {
  id: string;
  phase: 'perception' | 'decision' | 'tool' | 'synthesis';
  step: number;
  title: string;
  detail: string;
  durationMs: number;
  fsmMedianMs?: number | null;
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

export interface AgentTurnCallbacks {
  onStatusText?: (text: string) => void;
  onToolStepStart?: (step: number, maxSteps: number) => void;
  onToolStepComplete?: (record: AgentToolCallRecord) => void;
  onTraceEntry?: (entry: AgentTraceEntry) => void;
  onWorkspaceUpdate?: (files: WorkspaceFile[]) => void;
  onArtifact?: (artifact: AgentArtifact) => void;
  onFinalToken?: (fullRenderedText: string, liveTps: number) => void;
  loadWorkspaceImage?: (assetUrl: string, name: string) => Promise<PendingImageInput>;
}

export interface AgentTurnResult {
  finalText: string;
  toolCalls: AgentToolCallRecord[];
  traceEntries: AgentTraceEntry[];
  workspaceFiles: WorkspaceFile[];
  artifact: AgentArtifact | null;
  audioUrl?: string;
  totalPromptTokens: number;
  totalGeneratedTokens: number;
  decodeTokPerSec: number;
  prefillMs: number;
  curPos: number;
  aborted: boolean;
}

export function buildAgentActionSchema(capabilities: { vision: boolean }): Record<string, unknown> {
  const branches: Array<Record<string, unknown>> = [
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
        tool: { const: 'run_js' },
        code: { type: 'string' },
      },
      required: ['thought', 'tool', 'code'],
      additionalProperties: false,
    },
    {
      type: 'object',
      properties: {
        thought: { type: 'string' },
        tool: { const: 'render_artifact' },
        title: { type: 'string' },
        html: { type: 'string' },
      },
      required: ['thought', 'tool', 'title', 'html'],
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
  ];

  if (capabilities.vision) {
    branches.splice(4, 0, {
      type: 'object',
      properties: {
        thought: { type: 'string' },
        tool: { const: 'inspect_image' },
        path: { type: 'string' },
        question: { type: 'string' },
      },
      required: ['thought', 'tool', 'path', 'question'],
      additionalProperties: false,
    });
  }

  return { oneOf: branches };
}

export function buildArtifactSrcDoc(rawHtml: string, title = 'Artifact'): string {
  const trimmed = rawHtml.trim();
  const baseStyle = `
<style>
  *, *::before, *::after { box-sizing: border-box; }
  :root {
    color-scheme: dark;
    --bg: #09090b;
    --panel: #111113;
    --surface: #18181b;
    --border: rgba(255,255,255,0.1);
    --text: #fafafa;
    --muted: #a1a1aa;
    --accent: #f97316;
  }
  html, body {
    margin: 0;
    padding: 0;
    background: var(--bg);
    color: var(--text);
    font-family: -apple-system, BlinkMacSystemFont, "Inter", "Segoe UI", Roboto, sans-serif;
    font-size: 14px;
    line-height: 1.55;
    -webkit-font-smoothing: antialiased;
  }
  body { padding: 20px; }
  h1, h2, h3 { margin: 0 0 10px; letter-spacing: -0.02em; font-weight: 600; }
  p { margin: 0 0 10px; color: var(--muted); }
  .card, section, article {
    background: var(--panel);
    border: 1px solid var(--border);
    border-radius: 10px;
    padding: 14px 16px;
  }
  .grid { display: grid; gap: 12px; grid-template-columns: repeat(auto-fit, minmax(150px, 1fr)); margin-bottom: 14px; }
  table { width: 100%; border-collapse: collapse; margin: 10px 0; font-size: 13px; }
  th, td { padding: 8px 10px; text-align: left; border-bottom: 1px solid var(--border); }
  th { color: var(--muted); font-weight: 500; font-size: 12px; }
  input[type="range"] { accent-color: var(--accent); width: 100%; }
  button {
    background: #fafafa; color: #09090b; border: none; border-radius: 7px;
    padding: 7px 12px; font-weight: 600; cursor: pointer; font-size: 13px;
  }
  svg { max-width: 100%; height: auto; display: block; }
</style>`;

  if (/<html[\s>]/i.test(trimmed) || /^<!doctype/i.test(trimmed)) {
    if (/<head[\s>]/i.test(trimmed)) {
      return trimmed.replace(/<head([\s>])/i, `<head$1${baseStyle}`);
    }
    return trimmed;
  }

  return `<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8" />
<meta name="viewport" content="width=device-width, initial-scale=1.0" />
<title>${title.replace(/</g, '&lt;')}</title>
${baseStyle}
</head>
<body>
${trimmed}
</body>
</html>`;
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

function buildAgentSystemPrompt(
  workspace: WorkspaceFile[],
  capabilities: { vision: boolean },
  lang: 'en' | 'zh',
): string {
  const inspectLine = capabilities.vision
    ? `\n- inspect_image {"thought":"...","tool":"inspect_image","path":"<image_file>","question":"..."}: Run local vision understanding on an image file.`
    : '';
  const langHint = lang === 'zh'
    ? 'Respond concisely and accurately in Chinese when the user asks in Chinese.'
    : 'Respond concisely and accurately.';

  return `You are Ignis Agent running inside an isolated browser WebGPU workspace.
Output ONE valid JSON action per turn matching the schema.

Tools:
- read_file {"thought":"...","tool":"read_file","path":"<filename>"}
- write_file {"thought":"...","tool":"write_file","path":"<filename>","content":"..."}
- run_js {"thought":"...","tool":"run_js","code":"..."}: Browser Web Worker sandbox (NO require, NO fs, NO import). Pre-injected global: \`workspace.read(path)\`, \`workspace.parseCsv(path)\` -> \`{ headers, rows, records }\`, \`workspace.write(path, text)\`, \`console.log(...)\`. Keep code short (< 8 lines) and ONLY compute numbers/data; never put SVG/HTML inside run_js.
  Example run_js: {"thought":"Sum revenue and average churn from CSV","tool":"run_js","code":"const { records } = workspace.parseCsv('revenue_q4.csv'); const total = records.reduce((s,r)=>s+r.revenue_usd,0); const byRegion = {}; records.forEach(r=>byRegion[r.region]=(byRegion[r.region]||0)+r.revenue_usd); const avgChurn = records.reduce((s,r)=>s+r.churn_pct,0)/records.length; return { rowCount: records.length, total, byRegion, avgChurn: Number(avgChurn.toFixed(2)) };"}
- render_artifact {"thought":"...","tool":"render_artifact","title":"...","html":"..."}: Render a live dark-mode HTML/SVG UI preview (host already styles .grid, .card, table, th, td, button, h1-h3, svg on #09090b dark background). Keep HTML compact (< 1200 chars), zero external URLs or CDNs.
  Example render_artifact: {"thought":"Render KPI cards and bar chart with exact computed figures","tool":"render_artifact","title":"Q4 Revenue Dashboard","html":"<div class='grid'><div class='card'><small>Total Q4</small><h2>$265,000</h2></div><div class='card'><small>Avg Churn</small><h2>1.77%</h2></div></div><svg viewBox='0 0 360 120'><rect x='20' y='20' width='180' height='26' rx='4' fill='#f97316'/><text x='208' y='38' fill='#fafafa' font-size='12'>North America $163,700</text><rect x='20' y='64' width='112' height='26' rx='4' fill='#52525b'/><text x='140' y='82' fill='#fafafa' font-size='12'>Europe $101,300</text></svg>"}${inspectLine}
- finish {"thought":"...","tool":"finish","summary":"..."}: End tool execution once gathered results are ready.

Workspace files:
${formatWorkspaceTree(workspace)}

Strict rules:
1. Read target workspace files first if not yet in observations.
2. Never use Node.js \`require('fs')\` or invent numbers. Use \`workspace.parseCsv()\` or \`workspace.read()\` inside \`run_js\` for exact arithmetic, then use those exact numbers in \`render_artifact\` and \`finish\`.
3. Never repeat an identical tool call twice. Keep each tool step concise.
4. ${langHint}`;
}

export async function runAgentTurn(opts: {
  client: IgnisClient;
  archKind: ArchKind;
  capabilities: { vision: boolean; audioIn: boolean; audioOut: boolean; thinking: boolean };
  userPrompt: string;
  historyText?: string;
  workspaceFiles: WorkspaceFile[];
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
    capabilities,
    userPrompt,
    historyText = '',
    attachedImage = null,
    attachedAudio = null,
    generationMode,
    maxNewTokens,
    maxSteps = 5,
    lang,
    enableThinking = false,
    effort = 'low',
    callbacks = {},
  } = opts;

  let workspace = [...opts.workspaceFiles];
  const toolCalls: AgentToolCallRecord[] = [];
  const traceEntries: AgentTraceEntry[] = [];
  let latestArtifact: AgentArtifact | null = null;
  const seenSignatures = new Set<string>();
  const observations: string[] = [];

  let totalPromptTokens = 0;
  let totalGeneratedTokens = 0;
  let totalDecodeMs = 0;
  let totalPrefillMs = 0;
  let lastCurPos = 0;
  let wasAborted = false;

  const schema = buildAgentActionSchema({ vision: capabilities.vision });

  // Step 0: Multimodal Perception when user directly attached media
  if ((attachedImage && capabilities.vision) || (attachedAudio && capabilities.audioIn)) {
    const tPerc0 = performance.now();
    await client.reset();
    const percPrompt = formatSingleTurnPrompt(archKind, {
      systemPrompt: 'Describe the attached media concisely and accurately so a tool-using agent can act on it.',
      userPrompt,
      includeImageTag: Boolean(attachedImage && capabilities.vision),
      includeAudioTag: Boolean(attachedAudio && capabilities.audioIn),
    });
    const enc = await client.encode(percPrompt);
    const percRes = await client.generate({
      promptIds: enc.ids,
      images: attachedImage && capabilities.vision
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
    totalPromptTokens += percRes.promptLen || enc.ids.length;
    totalGeneratedTokens += percRes.generatedIds.length;
    totalDecodeMs += percRes.decodeMs || 0;
    totalPrefillMs += percRes.prefillMs || 0;
    lastCurPos = percRes.curPos || lastCurPos;
    wasAborted = Boolean(percRes.aborted);

    observations.push(`[Attached Media Perception]: ${obsText}`);
    const entry: AgentTraceEntry = {
      id: `trace-perc-${Date.now()}`,
      phase: 'perception',
      step: 0,
      title: 'Multimodal Perception',
      detail: obsText || '(No observation produced)',
      durationMs: dur,
      ok: Boolean(obsText),
    };
    traceEntries.push(entry);
    callbacks.onTraceEntry?.(entry);
  }

  let finishSummary = '';

  // Steps 1..maxSteps: Schema-Locked Tool Selection & Execution
  for (let step = 1; step <= maxSteps && !wasAborted; step++) {
    callbacks.onToolStepStart?.(step, maxSteps);
    const sysPrompt = buildAgentSystemPrompt(workspace, capabilities, lang);
    const historyBlock = historyText ? `Prior conversation context:\n${historyText}\n\n` : '';
    const toolProgressHint = toolCalls.length > 0
      ? `\nTools already executed: ${toolCalls.map((c) => c.tool).join(' -> ')}. Do NOT call ${toolCalls[toolCalls.length - 1].tool} again unless fixing an error; move to the next required tool (such as render_artifact, write_file, or finish).`
      : '';
    const obsBlock = observations.length > 0
      ? `\n\nCompleted observations so far:\n${observations.join('\n\n')}${toolProgressHint}\n\nPick the next required tool action or call "finish" if ready:`
      : '\n\nChoose the best first tool action (JSON only):';

    const stepUserPrompt = `${historyBlock}User task:\n${userPrompt}${obsBlock}`;
    const fullPrompt = formatSingleTurnPrompt(archKind, {
      systemPrompt: sysPrompt,
      userPrompt: stepUserPrompt,
      enableThinking: false,
    });

    await client.reset();
    const enc = await client.encode(fullPrompt);
    const tDecStart = performance.now();
    const stepRes = await client.generate({
      promptIds: enc.ids,
      images: [],
      audios: [],
      generationMode: 'text',
      maxNewTokens: 1024,
      chunkSize: 64,
      sampling: { temperature: 0, topK: 1, topP: 1.0, repetitionPenalty: 1.0 },
      responseFormat: {
        type: 'json_schema',
        json_schema: { name: 'ignis_agent_action', schema, strict: true },
      },
      stopOnEos: true,
    });

    totalPromptTokens += stepRes.promptLen || enc.ids.length;
    totalGeneratedTokens += stepRes.generatedIds.length;
    totalDecodeMs += stepRes.decodeMs || (performance.now() - tDecStart);
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
      title: `FSM Decision -> ${toolName}`,
      detail: thought ? `${thought}\n${JSON.stringify(cleanArgs)}` : JSON.stringify(cleanArgs),
      durationMs: stepRes.decodeMs || 0,
      fsmMedianMs: fsmMs,
      ok: true,
    };
    traceEntries.push(decisionTrace);
    callbacks.onTraceEntry?.(decisionTrace);

    if (toolName === 'finish') {
      finishSummary = String(action.summary || '').trim();
      break;
    }

    if (seenSignatures.has(sig)) {
      observations.push(`[Harness Guard] Tool ${toolName} with identical arguments was already executed. Moving to final answer.`);
      break;
    }
    seenSignatures.add(sig);

    // Execute Tool
    const tTool0 = performance.now();
    let ok = true;
    let output = '';

    if (toolName === 'read_file') {
      const targetPath = String(action.path || '');
      const f = readWorkspaceFile(workspace, targetPath);
      if (!f) {
        ok = false;
        output = `File "${targetPath}" not found. Available: ${workspace.map((x) => x.path).join(', ')}`;
      } else if (f.assetUrl) {
        output = `${f.path} is a ${f.kind} media file (${f.assetUrl}).${
          f.kind === 'image' && capabilities.vision ? ' Use inspect_image to analyze it visually.' : ''
        }`;
      } else {
        output = f.content;
      }
    } else if (toolName === 'write_file') {
      const targetPath = normalizeWorkspacePath(String(action.path || 'output.txt'));
      const content = String(action.content || '');
      workspace = upsertWorkspaceFile(workspace, targetPath, content, true);
      callbacks.onWorkspaceUpdate?.(workspace);
      output = `Wrote ${content.length} chars to ${targetPath}`;
    } else if (toolName === 'run_js') {
      const code = String(action.code || '');
      const execRes = await executeSandboxedJs(code, workspace);
      ok = execRes.ok;
      for (const [wPath, wContent] of Object.entries(execRes.writtenFiles)) {
        workspace = upsertWorkspaceFile(workspace, wPath, wContent, true);
      }
      if (Object.keys(execRes.writtenFiles).length > 0) {
        callbacks.onWorkspaceUpdate?.(workspace);
      }
      const parts: string[] = [];
      if (execRes.logs.length > 0) parts.push(`Console:\n${execRes.logs.join('\n')}`);
      if (execRes.result) parts.push(`Return: ${execRes.result}`);
      if (Object.keys(execRes.writtenFiles).length > 0) {
        parts.push(`Modified files: ${Object.keys(execRes.writtenFiles).join(', ')}`);
      }
      if (!execRes.ok && execRes.error) parts.push(`Error: ${execRes.error}`);
      output = parts.join('\n\n') || '(Executed cleanly with no output)';
    } else if (toolName === 'render_artifact') {
      const artTitle = String(action.title || 'Interactive Artifact').trim();
      const rawHtml = String(action.html || '').trim();
      const safeName = `${artTitle.toLowerCase().replace(/[^a-z0-9]+/g, '-').replace(/^-|-$/g, '') || 'artifact'}.html`;
      workspace = upsertWorkspaceFile(workspace, safeName, rawHtml, true);
      callbacks.onWorkspaceUpdate?.(workspace);

      const artifact: AgentArtifact = {
        id: `art-${Date.now()}-${step}`,
        title: artTitle,
        html: rawHtml,
        createdAt: Date.now(),
      };
      latestArtifact = artifact;
      callbacks.onArtifact?.(artifact);
      output = `Rendered live artifact "${artTitle}" (${rawHtml.length} chars) and saved snapshot to ${safeName}`;
    } else if (toolName === 'inspect_image') {
      const targetPath = String(action.path || '');
      const question = String(action.question || 'Describe all details in this image concisely.');
      const f = readWorkspaceFile(workspace, targetPath);
      if (!capabilities.vision) {
        ok = false;
        output = 'Active model does not support local vision.';
      } else if (!f || f.kind !== 'image' || !f.assetUrl) {
        ok = false;
        output = `Image file "${targetPath}" not found in workspace. Available: ${workspace
          .filter((x) => x.kind === 'image')
          .map((x) => x.path)
          .join(', ') || 'none'}`;
      } else if (!callbacks.loadWorkspaceImage) {
        ok = false;
        output = 'Workspace image decoder unavailable.';
      } else {
        const decodedImg = await callbacks.loadWorkspaceImage(f.assetUrl, f.path);
        await client.reset();
        const vPrompt = formatSingleTurnPrompt(archKind, {
          userPrompt: question,
          includeImageTag: true,
        });
        const vEnc = await client.encode(vPrompt);
        const vRes = await client.generate({
          promptIds: vEnc.ids,
          images: [{ data: decodedImg.data, width: decodedImg.width, height: decodedImg.height, channels: decodedImg.channels }],
          audios: [],
          generationMode: 'text',
          maxNewTokens: 160,
          chunkSize: 64,
          sampling: { temperature: 0, topK: 1, topP: 1.0, repetitionPenalty: 1.0 },
          stopOnEos: true,
        });
        totalPromptTokens += vRes.promptLen || vEnc.ids.length;
        totalGeneratedTokens += vRes.generatedIds.length;
        totalDecodeMs += vRes.decodeMs || 0;
        totalPrefillMs += vRes.prefillMs || 0;
        lastCurPos = vRes.curPos || lastCurPos;
        output = (vRes.generatedText || '').trim() || '(No visual description generated)';
      }
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
    };
    toolCalls.push(record);
    callbacks.onToolStepComplete?.(record);

    const toolTrace: AgentTraceEntry = {
      id: `trace-tool-${step}-${Date.now()}`,
      phase: 'tool',
      step,
      title: `${toolName} (${ok ? 'ok' : 'err'})`,
      detail: output.slice(0, 600),
      durationMs: toolDur,
      fsmMedianMs: fsmMs,
      ok,
    };
    traceEntries.push(toolTrace);
    callbacks.onTraceEntry?.(toolTrace);

    observations.push(`[Step ${step} ${toolName} -> ${ok ? 'OK' : 'ERROR'}]:\n${output.slice(0, 1200)}`);
  }

  // Final Synthesis Stage
  let finalText = '';
  let audioUrl: string | undefined;

  const needVoiceSynth = Boolean(capabilities.audioOut && generationMode !== 'text');
  const shouldSynthesize = !wasAborted && (toolCalls.length > 0 || needVoiceSynth || finishSummary.length < 140);

  if (shouldSynthesize) {
    callbacks.onStatusText?.(lang === 'zh' ? '正在汇总最终答复...' : 'Synthesizing final response...');
    const tSynth0 = performance.now();
    await client.reset();

    const evidenceBlock = observations.length > 0
      ? `\n\nTool Execution Results:\n${observations.join('\n\n')}`
      : '';
    const finishNote = finishSummary ? `\n\nAgent Draft Summary:\n${finishSummary}` : '';
    const synthUserPrompt = `${historyText ? `Prior conversation:\n${historyText}\n\n` : ''}Original user request:\n${userPrompt}${evidenceBlock}${finishNote}\n\nWrite a clear, helpful Markdown final answer grounded strictly in the observed tool results above.`;

    const useThinking = Boolean(capabilities.thinking && enableThinking && !needVoiceSynth);
    const synthPrompt = formatSingleTurnPrompt(archKind, {
      systemPrompt: 'You are Ignis Assistant. Present crisp, accurate Markdown answers grounded in verified tool outputs.',
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
      synthRes = await client.generate({
        promptIds: enc.ids,
        images: [],
        audios: [],
        generationMode: needVoiceSynth ? generationMode : 'text',
        maxNewTokens: Math.min(maxNewTokens, 768),
        maxThinkingTokens: archKind === 'spark' && useThinking ? (effort === 'low' ? 256 : 512) : 0,
        chunkSize: 64,
        sampling: {
          temperature: 0.4,
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
    finalText = thinkPrefix ? `${thinkPrefix}${synthRes.generatedText || ''}` : synthRes.generatedText || finishSummary;
    totalPromptTokens += synthRes.promptLen || synthPromptLen;
    totalGeneratedTokens += synthRes.generatedIds.length;
    totalDecodeMs += synthRes.decodeMs || (performance.now() - tSynth0);
    totalPrefillMs += synthRes.prefillMs || 0;
    lastCurPos = synthRes.curPos || lastCurPos;
    wasAborted = wasAborted || Boolean(synthRes.aborted);

    const synthTrace: AgentTraceEntry = {
      id: `trace-synth-${Date.now()}`,
      phase: 'synthesis',
      step: toolCalls.length + 1,
      title: needVoiceSynth ? 'Final Synthesis + 24 kHz Voice' : 'Final Response Synthesis',
      detail: `${synthRes.generatedIds.length} tokens @ ${synthRes.decodeTokPerSec.toFixed(1)} tok/s`,
      durationMs: performance.now() - tSynth0,
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
    artifact: latestArtifact,
    audioUrl,
    totalPromptTokens,
    totalGeneratedTokens,
    decodeTokPerSec: totalDecodeMs > 0 ? totalGeneratedTokens / (totalDecodeMs / 1000) : 0,
    prefillMs: totalPrefillMs,
    curPos: lastCurPos,
    aborted: wasAborted,
  };
}
