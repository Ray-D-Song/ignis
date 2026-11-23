'use client';

import React, { useState } from 'react';
import {
  BarChart3,
  Check,
  CheckCircle2,
  ChevronRight,
  Circle,
  Clock,
  Download,
  FileCode2,
  FileText,
  FolderTree,
  GitCompare,
  Layers,
  Loader2,
  Search,
  ShieldAlert,
  Sparkles,
  Terminal,
  Undo2,
  Volume2,
  XCircle,
} from 'lucide-react';
import type {
  AgentToolCallRecord,
  ApprovalDecision,
  ApprovalRequest,
  PendingAudioInput,
  PendingImageInput,
} from '@/lib/agent-harness';
import { COPY, type UiLang } from '@/lib/i18n';
import type { ModelPreset } from '@/lib/model-paths';
import { splitThinking } from '@/lib/prompt-templates';
import type { NodeTestSummary } from '@/lib/webcontainer-manager';
import type { WorkspaceCheckpoint } from '@/lib/workspace-fs';
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
  checkpoints?: WorkspaceCheckpoint[];
  testSummary?: NodeTestSummary | null;
  meta?: string;
  fsmValid?: boolean;
  fsmMedianMs?: number | null;
  decodeTokPerSec?: number | null;
  streamingThought?: string;
  replyThought?: string;
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
  pendingApproval: ApprovalRequest | null;
  onResolveApproval: (decision: ApprovalDecision) => void;
  onLoadModel: () => void;
  onRunStarter: (prompt: string) => void;
  onInspectDiff: (path: string) => void;
  onRollbackCheckpoint: (checkpointId: string) => void;
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

      <div
        className="loading-bar-track"
        role="progressbar"
        aria-valuenow={loadProgress.overallPct}
        aria-valuemin={0}
        aria-valuemax={100}
      >
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

function ColorizedDiffBlock({ diffText }: { diffText: string }) {
  const lines = diffText.split('\n');
  return (
    <pre className="diff-pre">
      {lines.map((line, i) => {
        const cls = line.startsWith('+') && !line.startsWith('+++')
          ? 'diff-add'
          : line.startsWith('-') && !line.startsWith('---')
            ? 'diff-del'
            : line.startsWith('@@')
              ? 'diff-hunk'
              : 'diff-ctx';
        return (
          <span key={i} className={`diff-line ${cls}`}>
            {line}
            {'\n'}
          </span>
        );
      })}
    </pre>
  );
}

function summarizeToolCall(tc: AgentToolCallRecord, lang: UiLang): string {
  const path = typeof tc.args.path === 'string' ? tc.args.path : '';
  const dir = typeof tc.args.dir === 'string' ? tc.args.dir : '.';
  const pattern = typeof tc.args.pattern === 'string' ? tc.args.pattern : '';
  const command = typeof tc.args.command === 'string' ? tc.args.command : '';

  switch (tc.tool) {
    case 'list_files':
      return lang === 'zh' ? `列出目录 ${dir}` : `List ${dir}`;
    case 'read':
    case 'read_file':
      return lang === 'zh' ? `读取 ${path || '.'}` : `Read ${path || '.'}`;
    case 'grep_files':
      return lang === 'zh' ? `搜索 "${pattern}"` : `Grep "${pattern}"`;
    case 'edit':
    case 'edit_file':
      return lang === 'zh' ? `编辑 ${path}` : `Edit ${path}`;
    case 'write':
    case 'write_file':
      return lang === 'zh' ? `写入 ${path}` : `Write ${path}`;
    case 'bash':
    case 'run_command':
      return `$ ${command}`;
    default:
      return tc.tool;
  }
}

function ToolIcon({ tool }: { tool: AgentToolCallRecord['tool'] }) {
  switch (tool) {
    case 'list_files':
      return <FolderTree size={13} style={{ color: '#a1a1aa', flexShrink: 0 }} />;
    case 'read':
    case 'read_file':
      return <FileText size={13} style={{ color: '#60a5fa', flexShrink: 0 }} />;
    case 'grep_files':
      return <Search size={13} style={{ color: '#38bdf8', flexShrink: 0 }} />;
    case 'edit':
    case 'edit_file':
    case 'write':
    case 'write_file':
      return <FileCode2 size={13} style={{ color: '#4ade80', flexShrink: 0 }} />;
    case 'bash':
    case 'run_command':
      return <Terminal size={13} style={{ color: '#fb923c', flexShrink: 0 }} />;
    default:
      return <Sparkles size={13} style={{ color: '#a1a1aa', flexShrink: 0 }} />;
  }
}

function InlineApprovalCard({
  lang,
  req,
  onResolve,
}: {
  lang: UiLang;
  req: ApprovalRequest;
  onResolve: (d: ApprovalDecision) => void;
}) {
  const t = COPY[lang];
  const [denyReason, setDenyReason] = useState('');

  return (
    <div className="approval-card" id="approval-card">
      <div className="approval-head">
        <span style={{ display: 'inline-flex', alignItems: 'center', gap: 7 }}>
          <ShieldAlert size={14} style={{ color: 'var(--accent)' }} />
          <strong>{t.approvalTitle}</strong>
        </span>
        <span className="feature-badge info" style={{ fontSize: 10.5 }}>
          {req.tool} · {req.target}
        </span>
      </div>

      {req.thought && <div className="tool-call-thought">{req.thought}</div>}

      {req.preview.includes('@@') || req.preview.startsWith('--- ') ? (
        <ColorizedDiffBlock diffText={req.preview} />
      ) : (
        <pre className="tool-call-pre">{req.preview}</pre>
      )}

      <div className="approval-actions">
        <button
          id="btn-approve-once"
          type="button"
          className="load-btn primary"
          onClick={() => onResolve({ approved: true })}
        >
          <Check size={13} />
          <span>{t.approvalApprove}</span>
        </button>
        <button
          id="btn-approve-all"
          type="button"
          className="load-btn secondary"
          onClick={() => onResolve({ approved: true, approveAll: true })}
        >
          <span>{t.approvalApproveAll}</span>
        </button>
        <input
          type="text"
          className="ui-input"
          style={{ flex: 1, minWidth: 110, height: 30, fontSize: 12 }}
          value={denyReason}
          onChange={(e) => setDenyReason(e.target.value)}
          placeholder={t.approvalDenyPlaceholder}
        />
        <button
          id="btn-deny-action"
          type="button"
          className="stop-cta"
          style={{ borderRadius: 7 }}
          onClick={() => onResolve({ approved: false, reason: denyReason.trim() || undefined })}
        >
          <XCircle size={13} />
          <span>{t.approvalDeny}</span>
        </button>
      </div>
    </div>
  );
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
    pendingApproval,
    onResolveApproval,
    onLoadModel,
    onRunStarter,
    onInspectDiff,
    onRollbackCheckpoint,
    onPreviewHtml,
    scrollEndRef,
  } = props;

  const t = COPY[lang];
  const displayName = activeModel.shortName || activeModel.label.split('·')[0].trim();

  if (messages.length === 0) {
    const busy = status === 'loading' || status === 'generating';

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
              {t.heroSubtitle ? <p className="empty-sub">{t.heroSubtitle}</p> : null}

              {(status === 'idle' || status === 'error') && (
                <div
                  className={`idle-readiness-banner ${status === 'error' ? 'is-error' : ''}`}
                  id="model-idle-banner"
                >
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
                {t.starters.map((s) => (
                  <button
                    key={s.id}
                    type="button"
                    className="starter-card"
                    disabled={busy || termsBlocked}
                    onClick={() => onRunStarter(s.prompt)}
                  >
                    <div className="starter-card-head">
                      {s.id === 'build_kanban' ? (
                        <Layers size={15} style={{ color: 'var(--accent)', flexShrink: 0 }} />
                      ) : s.id === 'build_pomodoro' ? (
                        <Clock size={15} style={{ color: 'var(--accent)', flexShrink: 0 }} />
                      ) : s.id === 'build_expense' ? (
                        <BarChart3 size={15} style={{ color: 'var(--accent)', flexShrink: 0 }} />
                      ) : (
                        <Sparkles size={15} style={{ color: 'var(--accent)', flexShrink: 0 }} />
                      )}
                      <span>{s.title}</span>
                    </div>
                    <div className="starter-card-desc">{s.subtitle}</div>
                  </button>
                ))}
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
          const lastCkpt = m.checkpoints && m.checkpoints.length > 0 ? m.checkpoints[0] : null;

          return (
            <div key={m.id} className="turn-assistant msg assistant">
              <div className="assistant-avatar">
                <SparkMark size={20} />
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
                    {m.toolCalls.map((tc) => {
                      const filePath = typeof tc.args.path === 'string' ? tc.args.path : '';
                      return (
                        <div key={tc.id} className="tool-step-group">
                          {tc.thought && <div className="tool-step-thought">{tc.thought}</div>}
                          <details className="tool-call-card">
                            <summary className="tool-call-summary">
                              <span className="tool-call-left">
                                <ChevronRight size={13} style={{ color: 'var(--text-muted)', flexShrink: 0 }} />
                                <ToolIcon tool={tc.tool} />
                                <span
                                  style={{
                                    fontWeight: 500,
                                    fontFamily: tc.tool === 'run_command' ? 'var(--font-mono)' : undefined,
                                    overflow: 'hidden',
                                    textOverflow: 'ellipsis',
                                    whiteSpace: 'nowrap',
                                  }}
                                >
                                  {summarizeToolCall(tc, lang)}
                                </span>
                              </span>
                              <span className="tool-call-meta">
                                {(tc.added !== undefined || tc.removed !== undefined) && (
                                  <span style={{ color: '#86efac' }}>
                                    +{tc.added ?? 0}/-{tc.removed ?? 0}
                                  </span>
                                )}
                                {tc.testSummary && (
                                  <span
                                    style={{
                                      color: tc.testSummary.failed === 0 ? '#86efac' : '#fca5a5',
                                    }}
                                  >
                                    {tc.testSummary.passed}/{tc.testSummary.total}
                                  </span>
                                )}
                                {tc.decodeTokPerSec !== null &&
                                  tc.decodeTokPerSec !== undefined &&
                                  tc.decodeTokPerSec > 0 && <span>{tc.decodeTokPerSec.toFixed(1)} tok/s</span>}
                                <span>{Math.max(1, Math.round(tc.durationMs))}ms</span>
                                {tc.ok ? (
                                  <CheckCircle2 size={12} style={{ color: 'var(--ok)' }} />
                                ) : (
                                  <XCircle size={12} style={{ color: 'var(--danger)' }} />
                                )}
                              </span>
                            </summary>
                            <div className="tool-call-body">
                              {tc.diffText ? (
                                <>
                                  <ColorizedDiffBlock diffText={tc.diffText} />
                                  {filePath && (
                                    <div>
                                      <button
                                        type="button"
                                        className="icon-btn"
                                        style={{ height: 24, fontSize: 11.5 }}
                                        onClick={() => onInspectDiff(filePath)}
                                      >
                                        <GitCompare size={12} />
                                        <span>{t.diffTab} · {filePath}</span>
                                      </button>
                                    </div>
                                  )}
                                </>
                              ) : (
                                <>
                                  <pre className="tool-call-pre">{JSON.stringify(tc.args, null, 2)}</pre>
                                  <pre className="tool-call-pre">{tc.output}</pre>
                                </>
                              )}
                            </div>
                          </details>
                        </div>
                      );
                    })}
                  </div>
                )}

                {isLastGenerating && m.streamingThought && (
                  <div className="tool-step-thought is-streaming">{m.streamingThought}</div>
                )}

                {isLastGenerating && pendingApproval && (
                  <InlineApprovalCard lang={lang} req={pendingApproval} onResolve={onResolveApproval} />
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

                {m.replyThought && !parsed.think && (
                  <div className="tool-step-thought">{m.replyThought}</div>
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

                {m.audioUrl && (
                  <audio controls src={m.audioUrl} style={{ width: '100%', maxWidth: 460, height: 36 }} />
                )}

                {(m.meta ||
                  m.testSummary ||
                  lastCkpt ||
                  m.fsmValid !== undefined ||
                  m.specProposed !== undefined) && (
                  <div className="turn-footer">
                    {m.meta && <span>{m.meta}</span>}
                    {m.testSummary && (
                      <span
                        className={`feature-badge ${m.testSummary.failed === 0 ? 'ok' : 'err'}`}
                        data-testid="test-verifier-badge"
                      >
                        Tests: {m.testSummary.passed}/{m.testSummary.total}{' '}
                        {m.testSummary.failed === 0 ? 'PASS ✓' : 'FAIL'}
                      </span>
                    )}
                    {lastCkpt && (
                      <button
                        type="button"
                        className="feature-badge"
                        style={{ cursor: 'pointer' }}
                        onClick={() => onRollbackCheckpoint(lastCkpt.id)}
                        title={`Rollback edits in ${lastCkpt.path}`}
                      >
                        <Undo2 size={11} />
                        <span>{t.checkpointRollback}</span>
                      </button>
                    )}
                    {m.fsmValid !== undefined && (
                      <span
                        className={`feature-badge ${m.fsmValid ? 'ok' : 'err'}`}
                        data-testid="fsm-badge"
                      >
                        FSM: {m.fsmValid ? 'JSON.parse ✓' : 'JSON Invalid ✗'}
                        {m.decodeTokPerSec !== null && m.decodeTokPerSec !== undefined && m.decodeTokPerSec > 0
                          ? ` · ${m.decodeTokPerSec.toFixed(1)} tok/s`
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
