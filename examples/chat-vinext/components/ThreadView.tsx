'use client';

import React from 'react';
import {
  ArrowRight,
  BarChart3,
  Check,
  CheckCircle2,
  ChevronRight,
  Circle,
  Code2,
  Download,
  Eye,
  FileCode2,
  FileText,
  Layout,
  Loader2,
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

export type LoadPhase =
  | 'manifest'
  | 'download'
  | 'cache'
  | 'tokenizer'
  | 'compile'
  | 'gpu_upload'
  | 'tokenizer_init';

export interface LoadProgressState {
  phase: LoadPhase;
  file: string;
  loaded: number;
  total: number;
  overallLoaded: number;
  overallTotal: number;
  fileIndex: number;
  fileCount: number;
  cached: boolean;
  overallPct: number;
  speedMBps: number | null;
  queuedPrompt: string | null;
}

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
  statusText: string;
  loadProgress: LoadProgressState | null;
  activeModel: ModelPreset;
  termsBlocked: boolean;
  onLoadModel: () => void;
  onRunStarter: (prompt: string, opts?: { generationMode?: 'text' | 'interleaved' }) => void;
  onOpenArtifact: (artifact: AgentArtifact) => void;
  onPreviewHtml: (title: string, html: string) => void;
  scrollEndRef: React.RefObject<HTMLDivElement | null>;
}

function formatBytes(bytes: number): string {
  if (!bytes || bytes <= 0) return '0 B';
  if (bytes >= 1024 * 1024 * 1024) return `${(bytes / (1024 * 1024 * 1024)).toFixed(2)} GB`;
  if (bytes >= 1024 * 1024) return `${(bytes / (1024 * 1024)).toFixed(0)} MB`;
  return `${(bytes / 1024).toFixed(0)} KB`;
}

function LoadingStateCard({
  lang,
  activeModel,
  loadProgress,
}: {
  lang: UiLang;
  activeModel: ModelPreset;
  loadProgress: LoadProgressState;
}) {
  const t = COPY[lang];
  const displayName = activeModel.shortName || activeModel.label.split('·')[0].trim();
  const phase = loadProgress.phase;
  const stageOrder =
    phase === 'compile' ? 1 : phase === 'gpu_upload' || phase === 'tokenizer_init' ? 2 : 0;

  const steps = [
    { key: 'download', label: t.loadingPhaseDownload },
    { key: 'compile', label: t.loadingPhaseCompile },
    { key: 'gpu_upload', label: t.loadingPhaseGpu },
  ];

  return (
    <div className="loading-stage-card" id="model-loading-card">
      <div className="loading-stage-head">
        <div style={{ display: 'flex', alignItems: 'center', gap: 10, minWidth: 0 }}>
          <span className="loading-icon-badge">
            <Loader2 size={16} className="spin-icon" style={{ color: 'var(--accent)' }} />
          </span>
          <div style={{ minWidth: 0 }}>
            <div className="loading-stage-title">{t.loadingTitle(displayName)}</div>
            <div className="loading-stage-meta">
              {loadProgress.file}
              {loadProgress.fileCount > 0 ? ` · ${loadProgress.fileIndex}/${loadProgress.fileCount}` : ''}
              {loadProgress.overallTotal > 0
                ? ` · ${formatBytes(loadProgress.overallLoaded)} / ${formatBytes(loadProgress.overallTotal)}`
                : ''}
            </div>
          </div>
        </div>
        <div className="loading-stage-pct">{loadProgress.overallPct}%</div>
      </div>

      {loadProgress.queuedPrompt && (
        <div className="queued-task-pill">
          <span style={{ color: 'var(--accent)', fontWeight: 600, whiteSpace: 'nowrap', flexShrink: 0 }}>
            {t.loadingQueued}:
          </span>
          <span style={{ overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>
            {loadProgress.queuedPrompt}
          </span>
        </div>
      )}

      <div className="loading-bar-track" role="progressbar" aria-valuenow={loadProgress.overallPct} aria-valuemin={0} aria-valuemax={100}>
        <div className="loading-bar-fill" style={{ width: `${Math.max(2, loadProgress.overallPct)}%` }} />
      </div>

      <div className="loading-telemetry-row">
        <span>
          {phase === 'download' && loadProgress.speedMBps !== null
            ? `${loadProgress.speedMBps.toFixed(1)} MB/s`
            : loadProgress.cached
              ? t.loadingCacheHit
              : loadProgress.file}
        </span>
        {activeModel.sizeLabel && <span>{activeModel.sizeLabel}</span>}
      </div>

      <div className="loading-steps">
        {steps.map((s, idx) => {
          const done = stageOrder > idx;
          const active = stageOrder === idx;
          return (
            <div key={s.key} className={`loading-step ${done ? 'done' : active ? 'active' : ''}`}>
              {done ? (
                <Check size={13} style={{ color: 'var(--ok)', flexShrink: 0 }} />
              ) : active ? (
                <Loader2 size={13} className="spin-icon" style={{ color: 'var(--accent)', flexShrink: 0 }} />
              ) : (
                <Circle size={12} style={{ color: 'var(--text-muted)', flexShrink: 0 }} />
              )}
              <span>{s.label}</span>
            </div>
          );
        })}
      </div>

      <div className="loading-note">{t.loadingNote}</div>
    </div>
  );
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
    statusText,
    loadProgress,
    activeModel,
    termsBlocked,
    onLoadModel,
    onRunStarter,
    onOpenArtifact,
    onPreviewHtml,
    scrollEndRef,
  } = props;

  const t = COPY[lang];
  const displayName = activeModel.shortName || activeModel.label.split('·')[0].trim();

  if (messages.length === 0) {
    const busy = status === 'loading' || status === 'generating';
    const starters = t.starters;

    return (
      <div className="thread-scroll">
        <div className="empty-canvas" id="hero-banner">
          {status === 'loading' && loadProgress ? (
            <LoadingStateCard lang={lang} activeModel={activeModel} loadProgress={loadProgress} />
          ) : (
            <>
              <div className="eyebrow-pill">
                <span
                  className={`status-dot ${status === 'ready' ? 'ready' : status === 'error' ? 'error' : 'idle'}`}
                  style={{ width: 6, height: 6 }}
                />
                <span>{t.eyebrow}</span>
              </div>

              <h1 className="empty-headline">{t.heroTitle}</h1>
              <p className="empty-sub">{t.heroSubtitle}</p>

              {(status === 'idle' || status === 'error') && (
                <div className={`idle-readiness-banner ${status === 'error' ? 'is-error' : ''}`} id="model-idle-banner">
                  <div className="idle-readiness-copy">
                    <div className="idle-readiness-title">
                      <span className={`status-dot ${status === 'error' ? 'error' : 'idle'}`} />
                      <span>{status === 'error' ? statusText : t.idleBannerTitle}</span>
                    </div>
                    <div className="idle-readiness-desc">
                      {termsBlocked
                        ? t.idleBannerTermsHint
                        : t.idleBannerDesc(displayName, activeModel.sizeLabel)}
                    </div>
                  </div>
                  <button
                    id="btn-hero-load-model"
                    type="button"
                    className="load-btn primary"
                    disabled={termsBlocked}
                    onClick={onLoadModel}
                  >
                    <Download size={13} />
                    <span>
                      {t.idleBannerCta}
                      {activeModel.sizeLabel ? ` (${activeModel.sizeLabel})` : ''}
                    </span>
                  </button>
                </div>
              )}

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
            </>
          )}
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
        {status === 'loading' && loadProgress && (
          <LoadingStateCard lang={lang} activeModel={activeModel} loadProgress={loadProgress} />
        )}
        <div ref={scrollEndRef} />
      </div>
    </div>
  );
}
