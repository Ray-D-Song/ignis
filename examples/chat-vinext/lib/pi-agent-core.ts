/**
 * Browser-native stateful Agent loop & Pi system prompt adapted from
 * `@mariozechner/pi-agent-core` and `@mariozechner/pi-coding-agent`.
 */

export interface PiTextContent {
  type: 'text';
  text: string;
}

export interface PiThinkingContent {
  type: 'thinking';
  thinking: string;
}

export interface PiToolCallContent {
  type: 'toolCall';
  id: string;
  name: string;
  arguments: Record<string, unknown>;
  thought?: string;
  fsmMedianMs?: number | null;
  decodeTokPerSec?: number | null;
  promptTokens?: number;
  durationMs?: number;
}

export type PiAssistantBlock = PiTextContent | PiThinkingContent | PiToolCallContent;

export interface PiUserMessage {
  role: 'user';
  content: string | PiTextContent[];
  timestamp: number;
}

export interface PiAssistantMessage {
  role: 'assistant';
  content: PiAssistantBlock[];
  stopReason: 'stop' | 'toolUse' | 'aborted' | 'error';
  errorMessage?: string;
  timestamp: number;
}

export interface PiToolResultMessage<TDetails = unknown> {
  role: 'toolResult';
  toolCallId: string;
  toolName: string;
  content: PiTextContent[];
  details?: TDetails;
  isError: boolean;
  timestamp: number;
}

export type PiAgentMessage = PiUserMessage | PiAssistantMessage | PiToolResultMessage;

export interface PiAgentToolResult<TDetails = unknown> {
  content: PiTextContent[];
  details?: TDetails;
  isError?: boolean;
  terminate?: boolean;
}

export interface PiAgentTool<
  TParams extends Record<string, unknown> = Record<string, unknown>,
  TDetails = unknown,
> {
  name: string;
  label: string;
  description: string;
  parameters?: Record<string, unknown>;
  execute: (
    toolCallId: string,
    args: TParams,
    signal?: AbortSignal,
    onUpdate?: (partial: PiAgentToolResult<TDetails>) => void,
  ) => Promise<PiAgentToolResult<TDetails>>;
}

export interface PiAgentContext {
  systemPrompt: string;
  messages: PiAgentMessage[];
  tools: PiAgentTool[];
}

export interface PiBeforeToolCallContext {
  assistantMessage: PiAssistantMessage;
  toolCall: PiToolCallContent;
  args: Record<string, unknown>;
  context: PiAgentContext;
}

export interface PiBeforeToolCallResult {
  block?: boolean;
  reason?: string;
}

export interface PiAfterToolCallContext {
  assistantMessage: PiAssistantMessage;
  toolCall: PiToolCallContent;
  args: Record<string, unknown>;
  result: PiAgentToolResult;
  isError: boolean;
  context: PiAgentContext;
}

export interface PiAfterToolCallResult {
  content?: PiTextContent[];
  details?: unknown;
  isError?: boolean;
  terminate?: boolean;
}

export interface PiShouldStopContext {
  message: PiAssistantMessage;
  toolResults: PiToolResultMessage[];
  context: PiAgentContext;
  newMessages: PiAgentMessage[];
}

export type PiAgentEvent =
  | { type: 'agent_start' }
  | { type: 'agent_end'; messages: PiAgentMessage[] }
  | { type: 'turn_start'; turnIndex: number }
  | { type: 'turn_end'; message: PiAssistantMessage; toolResults: PiToolResultMessage[] }
  | { type: 'message_start'; message: PiAgentMessage }
  | { type: 'message_end'; message: PiAgentMessage }
  | { type: 'tool_execution_start'; toolCallId: string; toolName: string; args: Record<string, unknown> }
  | { type: 'tool_execution_update'; toolCallId: string; toolName: string; args: Record<string, unknown>; partialResult: PiAgentToolResult }
  | { type: 'tool_execution_end'; toolCallId: string; toolName: string; result: PiAgentToolResult; isError: boolean };

export type PiStreamFn = (
  context: PiAgentContext,
  turnIndex: number,
  signal?: AbortSignal,
) => Promise<PiAssistantMessage>;

export interface PiAgentLoopConfig {
  maxTurns?: number;
  streamFn: PiStreamFn;
  beforeToolCall?: (
    ctx: PiBeforeToolCallContext,
    signal?: AbortSignal,
  ) => Promise<PiBeforeToolCallResult | void> | PiBeforeToolCallResult | void;
  afterToolCall?: (
    ctx: PiAfterToolCallContext,
    signal?: AbortSignal,
  ) => Promise<PiAfterToolCallResult | void> | PiAfterToolCallResult | void;
  shouldStopAfterTurn?: (ctx: PiShouldStopContext) => Promise<boolean> | boolean;
  getSteeringMessages?: (
    ctx: { lastAssistantMessage: PiAssistantMessage; context: PiAgentContext },
  ) => Promise<PiUserMessage[]> | PiUserMessage[];
  getFollowUpMessages?: () => Promise<PiUserMessage[]> | PiUserMessage[];
}

function createErrorToolResult(message: string): PiAgentToolResult {
  return {
    content: [{ type: 'text', text: message }],
    details: {},
    isError: true,
  };
}

export async function runPiAgentLoop(
  prompts: PiUserMessage[],
  context: PiAgentContext,
  config: PiAgentLoopConfig,
  emit: (event: PiAgentEvent) => Promise<void> | void,
  signal?: AbortSignal,
): Promise<PiAgentMessage[]> {
  const maxTurns = config.maxTurns ?? 12;
  const newMessages: PiAgentMessage[] = [...prompts];
  const currentContext: PiAgentContext = {
    systemPrompt: context.systemPrompt,
    messages: [...context.messages, ...prompts],
    tools: [...context.tools],
  };

  await emit({ type: 'agent_start' });
  let turnIndex = 0;

  for (const prompt of prompts) {
    await emit({ type: 'message_start', message: prompt });
    await emit({ type: 'message_end', message: prompt });
  }

  let pendingMessages: PiUserMessage[] = [];

  while (true) {
    let hasMoreToolCalls = true;

    while ((hasMoreToolCalls || pendingMessages.length > 0) && turnIndex < maxTurns) {
      if (signal?.aborted) {
        await emit({ type: 'agent_end', messages: newMessages });
        return currentContext.messages;
      }

      turnIndex++;
      await emit({ type: 'turn_start', turnIndex });

      if (pendingMessages.length > 0) {
        for (const msg of pendingMessages) {
          await emit({ type: 'message_start', message: msg });
          await emit({ type: 'message_end', message: msg });
          currentContext.messages.push(msg);
          newMessages.push(msg);
        }
        pendingMessages = [];
      }

      const assistantMsg = await config.streamFn(currentContext, turnIndex, signal);
      currentContext.messages.push(assistantMsg);
      newMessages.push(assistantMsg);
      await emit({ type: 'message_start', message: assistantMsg });
      await emit({ type: 'message_end', message: assistantMsg });

      if (assistantMsg.stopReason === 'error' || assistantMsg.stopReason === 'aborted') {
        await emit({ type: 'turn_end', message: assistantMsg, toolResults: [] });
        await emit({ type: 'agent_end', messages: newMessages });
        return currentContext.messages;
      }

      const toolCalls = assistantMsg.content.filter(
        (b): b is PiToolCallContent => b.type === 'toolCall',
      );
      const toolResults: PiToolResultMessage[] = [];
      let terminateBatch = false;

      if (toolCalls.length > 0) {
        for (const tc of toolCalls) {
          await emit({
            type: 'tool_execution_start',
            toolCallId: tc.id,
            toolName: tc.name,
            args: tc.arguments,
          });

          const tool = currentContext.tools.find((t) => t.name === tc.name);
          let result: PiAgentToolResult;
          let isError = false;

          if (!tool) {
            result = createErrorToolResult(`Tool "${tc.name}" not found.`);
            isError = true;
          } else {
            const before = config.beforeToolCall
              ? await config.beforeToolCall(
                  {
                    assistantMessage: assistantMsg,
                    toolCall: tc,
                    args: tc.arguments,
                    context: currentContext,
                  },
                  signal,
                )
              : undefined;

            if (before?.block) {
              result = createErrorToolResult(before.reason || 'Tool execution was blocked.');
              isError = true;
            } else {
              try {
                result = await tool.execute(tc.id, tc.arguments, signal, (partial) => {
                  void emit({
                    type: 'tool_execution_update',
                    toolCallId: tc.id,
                    toolName: tc.name,
                    args: tc.arguments,
                    partialResult: partial,
                  });
                });
                isError = Boolean(result.isError);
              } catch (err) {
                result = createErrorToolResult(err instanceof Error ? err.message : String(err));
                isError = true;
              }

              if (config.afterToolCall) {
                const after = await config.afterToolCall(
                  {
                    assistantMessage: assistantMsg,
                    toolCall: tc,
                    args: tc.arguments,
                    result,
                    isError,
                    context: currentContext,
                  },
                  signal,
                );
                if (after) {
                  result = {
                    content: after.content ?? result.content,
                    details: after.details ?? result.details,
                    terminate: after.terminate ?? result.terminate,
                  };
                  isError = after.isError ?? isError;
                }
              }
            }
          }

          if (result.terminate) terminateBatch = true;

          await emit({
            type: 'tool_execution_end',
            toolCallId: tc.id,
            toolName: tc.name,
            result,
            isError,
          });

          const toolResultMsg: PiToolResultMessage = {
            role: 'toolResult',
            toolCallId: tc.id,
            toolName: tc.name,
            content: result.content,
            details: result.details,
            isError,
            timestamp: Date.now(),
          };
          toolResults.push(toolResultMsg);
          currentContext.messages.push(toolResultMsg);
          newMessages.push(toolResultMsg);
          await emit({ type: 'message_start', message: toolResultMsg });
          await emit({ type: 'message_end', message: toolResultMsg });
        }
      }

      hasMoreToolCalls = toolCalls.length > 0 && !terminateBatch;
      await emit({ type: 'turn_end', message: assistantMsg, toolResults });

      if (
        await config.shouldStopAfterTurn?.({
          message: assistantMsg,
          toolResults,
          context: currentContext,
          newMessages,
        })
      ) {
        await emit({ type: 'agent_end', messages: newMessages });
        return currentContext.messages;
      }

      const steering =
        (await config.getSteeringMessages?.({
          lastAssistantMessage: assistantMsg,
          context: currentContext,
        })) || [];
      if (steering.length > 0) {
        pendingMessages = steering;
      }
    }

    const followUp = (await config.getFollowUpMessages?.()) || [];
    if (followUp.length > 0 && turnIndex < maxTurns) {
      pendingMessages = followUp;
      continue;
    }

    break;
  }

  await emit({ type: 'agent_end', messages: newMessages });
  return currentContext.messages;
}

export class PiAgent {
  private _systemPrompt: string;
  private _messages: PiAgentMessage[];
  private _tools: PiAgentTool[];
  private _isStreaming = false;
  private listeners = new Set<(ev: PiAgentEvent) => void>();

  constructor(init?: { systemPrompt?: string; messages?: PiAgentMessage[]; tools?: PiAgentTool[] }) {
    this._systemPrompt = init?.systemPrompt ?? '';
    this._messages = init?.messages ? [...init.messages] : [];
    this._tools = init?.tools ? [...init.tools] : [];
  }

  get state() {
    return {
      systemPrompt: this._systemPrompt,
      messages: this._messages,
      tools: this._tools,
      isStreaming: this._isStreaming,
    };
  }

  setSystemPrompt(prompt: string) {
    this._systemPrompt = prompt;
  }

  setTools(tools: PiAgentTool[]) {
    this._tools = [...tools];
  }

  replaceMessages(messages: PiAgentMessage[]) {
    this._messages = [...messages];
  }

  subscribe(listener: (ev: PiAgentEvent) => void): () => void {
    this.listeners.add(listener);
    return () => this.listeners.delete(listener);
  }

  async prompt(
    input: string | PiUserMessage,
    config: PiAgentLoopConfig,
    signal?: AbortSignal,
  ): Promise<PiAgentMessage[]> {
    const userMsg: PiUserMessage =
      typeof input === 'string'
        ? { role: 'user', content: input, timestamp: Date.now() }
        : input;
    this._isStreaming = true;
    try {
      const updated = await runPiAgentLoop(
        [userMsg],
        {
          systemPrompt: this._systemPrompt,
          messages: this._messages,
          tools: this._tools,
        },
        config,
        (ev) => {
          for (const fn of this.listeners) fn(ev);
        },
        signal,
      );
      this._messages = updated;
      return updated;
    } finally {
      this._isStreaming = false;
    }
  }
}

export function buildPiSystemPrompt(options: {
  workspaceTree: string;
  lang: 'en' | 'zh';
  cwd?: string;
}): string {
  const { workspaceTree, lang, cwd = '~/ignis-app' } = options;
  const langRule =
    lang === 'zh'
      ? 'Respond to the user in Chinese when they write in Chinese; keep code, identifiers, paths, and commands exact.'
      : 'Keep your final response concise and grounded strictly in observed files and tool results.';

  return `You are an expert coding assistant operating inside pi, a minimal coding agent harness running in a browser WebContainer sandbox (${cwd}).
Output ONE valid JSON action matching the schema per turn.

Available tools (output ONE JSON object per turn):
- read {"thought":"...","tool":"read","path":"src/App.jsx"}: Read file contents with line numbers (or pass "." to list files).
- write {"thought":"...","tool":"write","path":"src/App.jsx","content":"..."}: Create a new file or completely overwrite an existing file (always prefer write when building or replacing an app in src/App.jsx or src/App.css).
- edit {"thought":"...","tool":"edit","path":"src/App.css","oldText":"...","newText":"..."}: Replace an exact small snippet (1-15 lines) in an existing file. Never rewrite an entire file with edit.
- bash {"thought":"...","tool":"bash","command":"npm run build"}: Execute a shell command inside WebContainer.
- reply {"thought":"...","tool":"reply","text":"..."}: Send your final Markdown answer to the user once code changes are complete.

Examples:
- {"thought":"Inspect App entry point first","tool":"read","path":"src/App.jsx"}
- {"thought":"Inspect stylesheet","tool":"read","path":"src/App.css"}
- {"thought":"Write complete interactive React app","tool":"write","path":"src/App.jsx","content":"import { useState } from 'react';\\nimport './App.css';\\n\\nexport default function App() {\\n  return <div className=\\"app\\">Hello</div>;\\n}\\n"}
- {"thought":"Verify production bundle compiles","tool":"bash","command":"npm run build"}
- {"thought":"Application built and verified","tool":"reply","text":"Built the requested React application in src/App.jsx and src/App.css."}

Project files:
${workspaceTree}

Guidelines:
- Check unfamiliar files once with read before editing; use write to build or replace src/App.jsx completely and keep code concise (under ~85 lines, minimal comments).
- You may install npm packages with bash (e.g. "npm install <pkg>") before importing them, or write pure React, SVG, and CSS directly when no external package is needed.
- If a tool call returns an error, inspect the error and recover cleanly (prefer write if edit fails to match).
- Only report file changes and command outputs you actually observed.
- ${langRule}`;
}
