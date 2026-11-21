'use client';

import React from 'react';
import {
  ArrowRight,
  BarChart3,
  CheckCircle2,
  ChevronRight,
  Code2,
  Eye,
  FileCode2,
  FileText,
  Layout,
  Sparkles,
  Volume2,
  XCircle,
} from 'lucide-react';
import { COPY, type UiLang } from '@/lib/i18n';
import type { ModelPreset } from '@/lib/model-paths';
import { splitThinking } from '@/lib/prompt-templates';
import type {
  AgentArtifact,
  AgentToolCallRecord,
  PendingAudioInput,
  PendingImageInput,
} from '@/lib/agent-harness';
import { MarkdownContent } from './MarkdownContent';
import { SparkMark } from './TopBar';

export interface StudioMessage {
  id: string;
  role: 'user' | 'assistant';
  content: string;
  modelLabel?: string;
  image?: PendingImageInput | null;
  audio?: PendingAudioInput | null;
  audioUrl?: string;
  toolCalls?: AgentToolCallRecord[];
  artifact?: AgentArtifact | null;
  meta?: string;
  fsmValid?: boolean;
  fsmMedianMs?: number | null;
  specAccepted?: number;
  specProposed?: number;
}

interface ThreadViewProps {
  lang: UiLang;
  messages: StudioMessage[];
  status: 'idle' | 'loading' | 'ready' | 'generating' | 'error';
  activeModel: ModelPreset;
  termsBlocked: boolean;
  onRunStarter: (prompt: string, opts?: { generationMode?: 'text' | 'interleaved' }) => void;
  onOpenArtifact: (artifact: AgentArtifact) => void;
  onPreviewHtml: (title: string, html: string) => void;
  scrollEndRef: React.RefObject<HTMLDivElement | null>;
}

function summarizeToolCall(tc: AgentToolCallRecord, lang: UiLang): string {
  const path = typeof tc.args.path === 'string' ? tc.args.path : '';
  const title = typeof tc.args.title === 'string' ? tc.args.title : 'Artifact';
  switch (tc.tool) {
    case 'read_file':
      return lang === 'zh' ? `读取文件 ${path}` : `Read ${path}`;
    case 'write_file':
      return lang === 'zh' ? `写入文件 ${path}` : `Wrote ${path}`;
    case 'run_js':
      return lang === 'zh' ? `沙盒执行 JavaScript` : `Executed JavaScript`;
    case 'render_artifact':
      return lang === 'zh' ? `渲染产物：${title}` : `Rendered Artifact: ${title}`;
    case 'inspect_image':
      return lang === 'zh' ? `本地视觉识图 ${path}` : `Inspected ${path}`;
    default:
      return tc.tool;
  }
}

function ToolIcon({ tool }: { tool: AgentToolCallRecord['tool'] }) {
  switch (tool) {
    case 'read_file':
      return <FileText size={13} style={{ color: '#60a5fa', flexShrink: 0 }} />;
    case 'write_file':
      return <FileCode2 size={13} style={{ color: '#4ade80', flexShrink: 0 }} />;
    case 'run_js':
      return <Code2 size={13} style={{ color: '#fb923c', flexShrink: 0 }} />;
    case 'render_artifact':
      return <Layout size={13} style={{ color: '#c084fc', flexShrink: 0 }} />;
    case 'inspect_image':
      return <Eye size={13} style={{ color: '#38bdf8', flexShrink: 0 }} />;
    default:
      return <Sparkles size={13} style={{ color: '#a1a1aa', flexShrink: 0 }} />;
  }
}

export function ThreadView(props: ThreadViewProps) {
  const {
    lang,
    messages,
    status,
    activeModel,
    termsBlocked,
    onRunStarter,
    onOpenArtifact,
    onPreviewHtml,
    scrollEndRef,
  } = props;

  const t = COPY[lang];

  if (messages.length === 0) {
    const busy = status === 'loading' || status === 'generating';
    const starters = t.starters;

    return (
      <div className="thread-scroll">
        <div className="empty-canvas" id="hero-banner">
          <div className="eyebrow-pill">
            <span className={`status-dot ${status === 'ready' ? 'ready' : 'busy'}`} style={{ width: 6, height: 6 }} />
            <span>{t.eyebrow}</span>
          </div>

          <h1 className="empty-headline">{t.heroTitle}</h1>
          <p className="empty-sub">{t.heroSubtitle}</p>

          <div className="starter-grid">
            {starters.map((s) => {
              if (s.id === 'voice_or_spec') {
                const useVoice = Boolean(activeModel.capabilities.audioOut);
                const title = useVoice ? s.titleVoice : s.titleText;
                const subtitle = useVoice ? s.subtitleVoice : s.subtitleText;
                const prompt = useVoice ? s.promptVoice : s.promptText;
                return (
                  <button
                    key={s.id}
                    type="button"
                    className="starter-card"
                    disabled={busy || termsBlocked}
                    onClick={() => onRunStarter(prompt, { generationMode: useVoice ? 'interleaved' : 'text' })}
                  >
                    <div className="starter-card-head">
                      {useVoice ? (
                        <Volume2 size={15} style={{ color: 'var(--accent)' }} />
                      ) : (
                        <FileText size={15} style={{ color: 'var(--accent)' }} />
                      )}
                      <span>{title}</span>
                    </div>
                    <div className="starter-card-desc">{subtitle}</div>
                  </button>
                );
              }

              const disabledVision = Boolean(s.requiresVision && !activeModel.capabilities.vision);
              const Icon =
                s.id === 'revenue'
                  ? BarChart3
                  : s.id === 'vision'
                    ? Eye
                    : Layout;

              return (
                <button
                  key={s.id}
                  type="button"
                  className="starter-card"
                  disabled={busy || termsBlocked || disabledVision}
                  title={disabledVision ? 'Switch to LFM3-Omni or Gemma-4 for local vision' : undefined}
                  onClick={() => onRunStarter(s.prompt)}
                >
                  <div className="starter-card-head">
                    <Icon size={15} style={{ color: 'var(--accent)' }} />
                    <span>{s.title}</span>
                  </div>
                  <div className="starter-card-desc">{s.subtitle}</div>
                </button>
              );
            })}
          </div>
        </div>
        <div ref={scrollEndRef} />
      </div>
    );
  }

  return (
    <div className="thread-scroll">
      <div className="thread-list">
        {messages.map((m, idx) => {
          if (m.role === 'user') {
            return (
              <div key={m.id} className="turn-user">
                {(m.image || m.audio) && (
                  <div className="attach-strip" style={{ justifyContent: 'flex-end', marginBottom: 0 }}>
                    {m.image && (
                      <span className="attach-pill">
                        <img src={m.image.previewUrl} alt={m.image.name} />
                        <span>{m.image.name}</span>
                      </span>
                    )}
                    {m.audio && (
                      <span className="attach-pill">
                        <Volume2 size={13} />
                        <span>
                          {m.audio.name} ({m.audio.durationSec.toFixed(1)}s)
                        </span>
                      </span>
                    )}
                  </div>
                )}
                <div className="user-pill">{m.content}</div>
              </div>
            );
          }

          const parsed = splitThinking(m.content);
          const isLastGenerating = status === 'generating' && idx === messages.length - 1;
          const isActivelyThinking = Boolean(parsed.isThinking && isLastGenerating);

          return (
            <div key={m.id} className="turn-assistant msg assistant">
              <div className="assistant-avatar">
                <SparkMark size={14} />
              </div>

              <div className="assistant-body">
                <div className="assistant-header">
                  <span>{m.modelLabel || 'Ignis'}</span>
                  {isLastGenerating && !parsed.answer && (!m.toolCalls || m.toolCalls.length === 0) && (
                    <span style={{ color: 'var(--accent)', fontSize: 11.5 }}>●</span>
                  )}
                </div>

                {m.toolCalls && m.toolCalls.length > 0 && (
                  <div className="tool-stack">
                    {m.toolCalls.map((tc) => (
                      <details key={tc.id} className="tool-call-card">
                        <summary className="tool-call-summary">
                          <span className="tool-call-left">
                            <ChevronRight size={13} style={{ color: 'var(--text-muted)', flexShrink: 0 }} />
                            <ToolIcon tool={tc.tool} />
                            <span style={{ fontWeight: 500, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>
                              {summarizeToolCall(tc, lang)}
                            </span>
                          </span>
                          <span className="tool-call-meta">
                            {tc.fsmMedianMs !== null && tc.fsmMedianMs !== undefined && (
                              <span>FSM {tc.fsmMedianMs.toFixed(2)}ms/t</span>
                            )}
                            <span>{Math.max(1, Math.round(tc.durationMs))}ms</span>
                            {tc.ok ? (
                              <CheckCircle2 size={12} style={{ color: 'var(--ok)' }} />
                            ) : (
                              <XCircle size={12} style={{ color: 'var(--danger)' }} />
                            )}
                          </span>
                        </summary>
                        <div className="tool-call-body">
                          {tc.thought && <div className="tool-call-thought">{tc.thought}</div>}
                          <pre className="tool-call-pre">{JSON.stringify(tc.args, null, 2)}</pre>
                          <pre className="tool-call-pre">{tc.output}</pre>
                        </div>
                      </details>
                    ))}
                  </div>
                )}

                {parsed.think && (
                  <details className="think-block" open={isActivelyThinking}>
                    <summary>
                      {isActivelyThinking
                        ? t.thinkOpen
                        : parsed.isThinking
                          ? t.thinkAborted
                          : t.thinkView}
                    </summary>
                    <pre>{parsed.think}</pre>
                  </details>
                )}

                {parsed.answer ? (
                  <MarkdownContent content={parsed.answer} onPreviewHtml={onPreviewHtml} />
                ) : m.audioUrl ? (
                  <div className="answer-body">{t.audioGenerated}</div>
                ) : (
                  <div className="answer-body" style={{ color: 'var(--text-muted)' }}>
                    {isActivelyThinking ? '' : isLastGenerating ? '…' : ''}
                  </div>
                )}

                {m.artifact && (
                  <button
                    type="button"
                    className="artifact-callout"
                    onClick={() => onOpenArtifact(m.artifact!)}
                  >
                    <span style={{ display: 'flex', alignItems: 'center', gap: 9 }}>
                      <Layout size={15} style={{ color: 'var(--accent)' }} />
                      <span>
                        <strong style={{ fontSize: 13, display: 'block' }}>{m.artifact.title}</strong>
                        <span style={{ fontSize: 11.5, color: 'var(--text-secondary)' }}>
                          {lang === 'zh' ? '点击在右侧面板预览实时产物' : 'Open live interactive preview'}
                        </span>
                      </span>
                    </span>
                    <ArrowRight size={14} style={{ color: 'var(--text-secondary)' }} />
                  </button>
                )}

                {m.audioUrl && (
                  <audio controls src={m.audioUrl} style={{ width: '100%', maxWidth: 460, height: 36 }} />
                )}

                {(m.meta || m.fsmValid !== undefined || m.specProposed !== undefined) && (
                  <div className="turn-footer">
                    {m.meta && <span>{m.meta}</span>}
                    {m.fsmValid !== undefined && (
                      <span
                        className={`feature-badge ${m.fsmValid ? 'ok' : 'err'}`}
                        data-testid="fsm-badge"
                      >
                        FSM: {m.fsmValid ? 'JSON.parse ✓' : 'JSON Invalid ✗'}
                        {m.fsmMedianMs !== null && m.fsmMedianMs !== undefined
                          ? ` · ${m.fsmMedianMs.toFixed(3)} ms/tok`
                          : ''}
                      </span>
                    )}
                    {m.specProposed !== undefined && (
                      <span className="feature-badge info" data-testid="ngram-badge">
                        N-gram: {m.specAccepted ?? 0}/{m.specProposed} accepted (
                        {m.specProposed > 0 ? Math.round(((m.specAccepted ?? 0) / m.specProposed) * 100) : 0}%)
                      </span>
                    )}
                  </div>
                )}
              </div>
            </div>
          );
        })}
        <div ref={scrollEndRef} />
      </div>
    </div>
  );
}
