'use client';

import React, { useState } from 'react';
import {
  Activity,
  Cpu,
  ExternalLink,
  Globe,
  Loader2,
  RefreshCw,
  Trash2,
  Undo2,
  X,
} from 'lucide-react';
import type { detectDeviceProfile } from '../../../runtime/device-profile.mjs';
import type { AgentTraceEntry } from '@/lib/agent-harness';
import { COPY, type UiLang } from '@/lib/i18n';
import type { ModelPreset } from '@/lib/model-paths';
import type { ArchKind, OmniGenMode } from '@/lib/prompt-templates';
import type { TerminalChannel } from '@/lib/terminal-channel';
import type { InteractiveShellHandle, SandboxBootStatus } from '@/lib/webcontainer-manager';
import type { WorkspaceCheckpoint } from '@/lib/workspace-fs';
import { TerminalPane } from './TerminalPane';

type IgnisDeviceProfile = Awaited<ReturnType<typeof detectDeviceProfile>>;

export type WorkspaceTab = 'preview' | 'trace' | 'runtime';

export interface FsmPreset {
  id: string;
  label: string;
  schema: Record<string, unknown>;
}

interface WorkspacePaneProps {
  lang: UiLang;
  theme?: 'dark' | 'light';
  activeTab: WorkspaceTab;
  onTabChange: (tab: WorkspaceTab) => void;
  onClose: () => void;

  sandboxStatus: SandboxBootStatus;
  sandboxDetail: string;
  previewUrl: string | null;
  previewRefreshSeq?: number;
  shellChannel: TerminalChannel;
  serverChannel: TerminalChannel;
  onAttachShell: (
    cols: number,
    rows: number,
    onData: (chunk: string) => void,
  ) => Promise<InteractiveShellHandle>;
  onRestartServer: () => void;

  traceEntries: AgentTraceEntry[];
  checkpoints: WorkspaceCheckpoint[];
  onRollbackCheckpoint: (checkpointId: string) => void;
  contextBudget: { peak: number; max: number };

  deviceProfile: IgnisDeviceProfile | null;
  gpuCaps: string;
  stats: {
    initMs: string;
    cacheMode: string;
    prefillSpeed: string;
    decodeSpeed: string;
    ctxPos: number;
    maxCtx: number;
    heapMB: string;
  };
  status: 'idle' | 'loading' | 'ready' | 'generating' | 'error';
  activeModel: ModelPreset;
  archKind: ArchKind;
  enableThinking: boolean;
  onThinkingChange: (v: boolean) => void;
  effort: 'low' | 'high';
  onEffortChange: (v: 'low' | 'high') => void;
  generationMode: OmniGenMode;
  onGenerationModeChange: (v: OmniGenMode) => void;
  enableFsm: boolean;
  onFsmToggle: (v: boolean) => void;
  fsmPresets: FsmPreset[];
  fsmPresetId: string;
  onFsmPresetChange: (id: string) => void;
  fsmSchemaText: string;
  onFsmSchemaTextChange: (txt: string) => void;
  enableNgram: boolean;
  onNgramToggle: (v: boolean) => void;
  temperature: number;
  onTemperatureChange: (v: number) => void;
  maxNewTokens: number;
  onMaxNewTokensChange: (v: number) => void;
  onRunShowcase: (kind: 'vqa' | 'voice' | 'fsm' | 'ngram') => void;
  onClearOpfs: () => void;
}

export function WorkspacePane(props: WorkspacePaneProps) {
  const {
    lang,
    theme = 'dark',
    activeTab,
    onTabChange,
    onClose,
    sandboxStatus,
    sandboxDetail,
    previewUrl,
    previewRefreshSeq = 0,
    shellChannel,
    serverChannel,
    onAttachShell,
    onRestartServer,
    traceEntries,
    checkpoints,
    onRollbackCheckpoint,
    contextBudget,
    deviceProfile,
    gpuCaps,
    stats,
    status,
    activeModel,
    archKind,
    enableThinking,
    onThinkingChange,
    effort,
    onEffortChange,
    generationMode,
    onGenerationModeChange,
    enableFsm,
    onFsmToggle,
    fsmPresets,
    fsmPresetId,
    onFsmPresetChange,
    fsmSchemaText,
    onFsmSchemaTextChange,
    enableNgram,
    onNgramToggle,
    temperature,
    onTemperatureChange,
    maxNewTokens,
    onMaxNewTokensChange,
    onRunShowcase,
    onClearOpfs,
  } = props;

  const t = COPY[lang];
  const [previewKey, setPreviewKey] = useState(0);

  const budgetPct =
    contextBudget.max > 0 ? Math.min(100, Math.round((contextBudget.peak / contextBudget.max) * 100)) : 0;

  return (
    <aside className="workspace-pane" id="workspace-pane">
      <div className="pane-tabs">
        <div className="pane-tab-group" role="tablist">
          <button
            id="tab-btn-preview"
            type="button"
            role="tab"
            aria-selected={activeTab === 'preview'}
            className={`pane-tab ${activeTab === 'preview' ? 'active' : ''}`}
            onClick={() => onTabChange('preview')}
          >
            <Globe size={13} />
            <span>{t.tabs.preview}</span>
          </button>
          <button
            id="tab-btn-trace"
            type="button"
            role="tab"
            aria-selected={activeTab === 'trace'}
            className={`pane-tab ${activeTab === 'trace' ? 'active' : ''}`}
            onClick={() => onTabChange('trace')}
          >
            <Activity size={13} />
            <span>{t.tabs.trace}</span>
          </button>
          <button
            id="tab-btn-runtime"
            type="button"
            role="tab"
            aria-selected={activeTab === 'runtime'}
            className={`pane-tab ${activeTab === 'runtime' ? 'active' : ''}`}
            onClick={() => onTabChange('runtime')}
          >
            <Cpu size={13} />
            <span>{t.tabs.runtime}</span>
          </button>
        </div>

        <button
          id="btn-close-workspace"
          type="button"
          className="icon-btn"
          style={{ height: 26, padding: '0 6px' }}
          onClick={onClose}
          title="Close pane"
          aria-label="Close workspace pane"
        >
          <X size={14} />
        </button>
      </div>

      <div className="pane-body">
        {activeTab === 'preview' && (
          <div className="preview-split-col">
            <div className="preview-frame-box">
              <div className="preview-url-bar">
                <div className="preview-url-pill">
                  <span
                    className={`status-dot ${
                      previewUrl ? 'ready' : sandboxStatus === 'error' ? 'error' : 'busy'
                    }`}
                  />
                  <span>{previewUrl || 'http://localhost:3000'}</span>
                </div>
                <div style={{ display: 'flex', gap: 4 }}>
                  <button
                    id="btn-refresh-preview"
                    type="button"
                    className="icon-btn"
                    style={{ height: 24, padding: '0 6px' }}
                    disabled={!previewUrl}
                    onClick={() => setPreviewKey((k) => k + 1)}
                    title={t.refreshPreview}
                  >
                    <RefreshCw size={12} />
                  </button>
                  {previewUrl && (
                    <a
                      href={previewUrl}
                      target="_blank"
                      rel="noreferrer"
                      className="icon-btn"
                      style={{ height: 24, padding: '0 6px' }}
                      title={t.openPreviewExternal}
                    >
                      <ExternalLink size={12} />
                    </a>
                  )}
                </div>
              </div>

              {previewUrl ? (
                <iframe
                  id="preview-iframe"
                  key={`${previewUrl}-${previewKey}-${previewRefreshSeq}`}
                  className="preview-iframe"
                  src={previewUrl}
                  allow="cross-origin-isolated"
                  title="Vite + React WebContainer Preview"
                />
              ) : (
                <div className="preview-boot-overlay" id="sandbox-boot-overlay">
                  {sandboxStatus === 'error' ? (
                    <span className="feature-badge err">{t.sandboxError}: {sandboxDetail}</span>
                  ) : (
                    <>
                      <Loader2 size={20} className="spin-icon" style={{ color: 'var(--accent)' }} />
                      <div style={{ fontWeight: 600, fontSize: 13 }}>
                        {sandboxStatus === 'installing'
                          ? t.sandboxInstalling
                          : sandboxStatus === 'starting'
                            ? t.sandboxStarting
                            : t.sandboxBooting}
                      </div>
                      <div style={{ fontSize: 12, color: 'var(--text-muted)' }}>
                        {sandboxDetail || 'Vite 6 + React 18'}
                      </div>
                    </>
                  )}
                </div>
              )}
            </div>

            <TerminalPane
              lang={lang}
              theme={theme}
              sandboxStatus={sandboxStatus}
              sandboxDetail={sandboxDetail}
              shellChannel={shellChannel}
              serverChannel={serverChannel}
              onAttachShell={onAttachShell}
              onRestartServer={onRestartServer}
            />
          </div>
        )}

        {activeTab === 'trace' && (
          <div className="trace-timeline" id="trace-timeline">
            <div className="inspector-card">
              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                <span className="inspector-title">{t.contextBudget}</span>
                <span style={{ fontFamily: 'var(--font-mono)', fontSize: 11.5, color: 'var(--text-secondary)' }}>
                  {contextBudget.peak.toLocaleString()} / {contextBudget.max.toLocaleString()} tok ({budgetPct}%)
                </span>
              </div>
              <div className="loading-bar-track" style={{ height: 6 }}>
                <div
                  className="loading-bar-fill"
                  style={{
                    width: `${budgetPct}%`,
                    animation: 'none',
                    background: budgetPct > 85 ? 'var(--danger)' : 'var(--accent)',
                  }}
                />
              </div>
            </div>

            {checkpoints.length > 0 && (
              <div className="inspector-card">
                <div className="inspector-title">Checkpoints ({checkpoints.length})</div>
                {checkpoints.map((cp) => (
                  <div key={cp.id} className="checkpoint-row">
                    <div style={{ minWidth: 0 }}>
                      <div style={{ fontFamily: 'var(--font-mono)', fontSize: 12, color: 'var(--text)' }}>
                        #{cp.step} · {cp.path}
                      </div>
                      <div style={{ fontSize: 11, color: 'var(--text-muted)' }}>
                        +{cp.added} / -{cp.removed} lines
                      </div>
                    </div>
                    <button
                      type="button"
                      className="icon-btn"
                      style={{ height: 24, fontSize: 11.5 }}
                      onClick={() => onRollbackCheckpoint(cp.id)}
                    >
                      <Undo2 size={11} />
                      <span>{t.checkpointRollback}</span>
                    </button>
                  </div>
                ))}
              </div>
            )}

            {traceEntries.length === 0 ? (
              <div style={{ margin: '36px auto', textAlign: 'center', maxWidth: 320, color: 'var(--text-secondary)', fontSize: 12.5, lineHeight: 1.55 }}>
                {t.traceEmpty}
              </div>
            ) : (
              traceEntries.map((entry) => (
                <div key={entry.id} className="trace-card" data-testid="trace-entry">
                  <div className="trace-card-head">
                    <span style={{ display: 'inline-flex', alignItems: 'center', gap: 6 }}>
                      <span
                        className={`feature-badge ${entry.ok ? 'ok' : 'err'}`}
                        style={{ textTransform: 'uppercase', fontSize: 10 }}
                      >
                        {entry.phase}
                      </span>
                      <span>{entry.title}</span>
                    </span>
                    <span style={{ fontFamily: 'var(--font-mono)', fontSize: 11, color: 'var(--text-muted)' }}>
                      {entry.fsmMedianMs !== undefined && entry.fsmMedianMs !== null
                        ? `${entry.fsmMedianMs.toFixed(2)}ms/t · `
                        : ''}
                      {Math.max(1, Math.round(entry.durationMs))} ms
                    </span>
                  </div>
                  <pre className="trace-card-detail">{entry.detail}</pre>
                </div>
              ))
            )}
          </div>
        )}

        {activeTab === 'runtime' && (
          <div className="runtime-scroll">
            <div className="inspector-card" id="hardware-card">
              <div className="inspector-title">{t.runtimeHardware}</div>
              {!deviceProfile ? (
                <div style={{ fontSize: 12, color: 'var(--text-secondary)' }}>Probing WebGPU adapter...</div>
              ) : !deviceProfile.supported ? (
                <>
                  <span className="hw-pill err">WebGPU Unavailable</span>
                  <div style={{ fontSize: 12, color: 'var(--text-secondary)' }}>
                    Desktop Chromium 138+ or Edge 157+ with hardware WebGPU enabled is required.
                  </div>
                </>
              ) : (
                <>
                  <span className={`hw-pill ${deviceProfile.tier === 'tier1_full' ? 'ok' : 'warn'}`}>
                    {deviceProfile.tier} · {deviceProfile.estimatedVramClass.toUpperCase()} VRAM
                  </span>
                  <div style={{ fontSize: 12, color: 'var(--text-secondary)' }}>
                    {deviceProfile.adapterInfo?.description ||
                      deviceProfile.adapterInfo?.architecture ||
                      deviceProfile.adapterInfo?.vendor ||
                      'WebGPU Adapter'}
                  </div>
                  <div style={{ fontFamily: 'var(--font-mono)', fontSize: 11.5, color: 'var(--text-muted)' }}>
                    {gpuCaps} · MaxBuf {(deviceProfile.maxBufferSize / (1024 * 1024)).toFixed(0)} MiB
                  </div>
                </>
              )}
            </div>

            <div className="inspector-card">
              <div className="inspector-title">{t.runtimeTelemetry}</div>
              <div className="stat-grid">
                <div className="stat-box">
                  <div className="k">Decode</div>
                  <div className="v" id="stat-decode">
                    {stats.decodeSpeed}
                  </div>
                </div>
                <div className="stat-box">
                  <div className="k">Prefill</div>
                  <div className="v">{stats.prefillSpeed}</div>
                </div>
                <div className="stat-box">
                  <div className="k">Init</div>
                  <div className="v">{stats.initMs}</div>
                </div>
                <div className="stat-box">
                  <div className="k">KV Ctx</div>
                  <div className="v">
                    {stats.ctxPos} / {stats.maxCtx}
                  </div>
                </div>
                <div className="stat-box">
                  <div className="k">Heap</div>
                  <div className="v">{stats.heapMB}</div>
                </div>
                <div className="stat-box">
                  <div className="k">OPFS</div>
                  <div className="v" style={{ fontSize: 11 }}>
                    {stats.cacheMode}
                  </div>
                </div>
              </div>
            </div>

            <div className="inspector-card">
              <div className="inspector-title">{t.runtimeControls}</div>

              {activeModel?.capabilities.thinking && (
                <div className="toggle-line">
                  <span>{t.thinking}</span>
                  <input
                    type="checkbox"
                    checked={enableThinking && !enableFsm}
                    disabled={status === 'generating' || enableFsm}
                    onChange={(e) => onThinkingChange(e.target.checked)}
                  />
                </div>
              )}

              {archKind === 'spark' && enableThinking && !enableFsm && (
                <label className="field-label">
                  {t.effort}
                  <select
                    className="ui-select"
                    value={effort}
                    disabled={status === 'generating'}
                    onChange={(e) => onEffortChange(e.target.value as 'low' | 'high')}
                  >
                    <option value="low">low (&le;384 tok)</option>
                    <option value="high">high (&le;65% budget)</option>
                  </select>
                </label>
              )}

              {activeModel?.capabilities.audioOut && (
                <label className="field-label">
                  {t.outputModality}
                  <select
                    className="ui-select"
                    value={generationMode}
                    disabled={status === 'generating' || enableFsm}
                    onChange={(e) => onGenerationModeChange(e.target.value as OmniGenMode)}
                  >
                    <option value="text">Text only</option>
                    <option value="interleaved">Text + 24 kHz Voice</option>
                    <option value="audio">24 kHz WAV only</option>
                  </select>
                </label>
              )}

              <div className="toggle-line">
                <span>{t.fsmToggle}</span>
                <input
                  id="toggle-fsm"
                  type="checkbox"
                  checked={enableFsm}
                  disabled={status === 'generating'}
                  onChange={(e) => onFsmToggle(e.target.checked)}
                />
              </div>

              {enableFsm && (
                <>
                  <label className="field-label">
                    {t.fsmPreset}
                    <select
                      className="ui-select"
                      value={fsmPresetId}
                      disabled={status === 'generating'}
                      onChange={(e) => onFsmPresetChange(e.target.value)}
                    >
                      {fsmPresets.map((p) => (
                        <option key={p.id} value={p.id}>
                          {p.label}
                        </option>
                      ))}
                    </select>
                  </label>
                  <label className="field-label">
                    {t.fsmCustomJson}
                    <textarea
                      id="fsm-schema-input"
                      className="ui-textarea"
                      rows={5}
                      value={fsmSchemaText}
                      disabled={status === 'generating'}
                      onChange={(e) => onFsmSchemaTextChange(e.target.value)}
                    />
                  </label>
                </>
              )}

              <div className="toggle-line">
                <span>{t.ngramToggle}</span>
                <input
                  id="toggle-ngram"
                  type="checkbox"
                  checked={enableNgram && !enableFsm}
                  disabled={status === 'generating' || enableFsm}
                  onChange={(e) => onNgramToggle(e.target.checked)}
                />
              </div>

              <label className="field-label">
                Temperature: {temperature.toFixed(2)}
                <input
                  type="range"
                  min="0"
                  max="1.2"
                  step="0.05"
                  value={temperature}
                  onChange={(e) => onTemperatureChange(parseFloat(e.target.value))}
                />
              </label>

              <label className="field-label">
                Max New Tokens: {maxNewTokens}
                <input
                  type="range"
                  min="128"
                  max="4096"
                  step="128"
                  value={maxNewTokens}
                  onChange={(e) => onMaxNewTokensChange(parseInt(e.target.value, 10))}
                />
              </label>
            </div>

            <div className="inspector-card">
              <div className="inspector-title">{t.runtimeVerify}</div>
              <div className="verify-grid">
                <button
                  id="showcase-vqa"
                  type="button"
                  className="verify-btn"
                  disabled={status !== 'ready' || !activeModel?.capabilities.vision}
                  onClick={() => onRunShowcase('vqa')}
                >
                  <span>Omni VQA</span>
                  <small>demo-shapes.png</small>
                </button>
                <button
                  id="showcase-voice"
                  type="button"
                  className="verify-btn"
                  disabled={status !== 'ready' || !activeModel?.capabilities.audioOut}
                  onClick={() => onRunShowcase('voice')}
                >
                  <span>24 kHz Voice</span>
                  <small>LiquidCodec WAV</small>
                </button>
                <button
                  id="showcase-fsm"
                  type="button"
                  className="verify-btn"
                  disabled={status !== 'ready'}
                  onClick={() => onRunShowcase('fsm')}
                >
                  <span>JSON Schema FSM</span>
                  <small>0.3 ms/tok DFA</small>
                </button>
                <button
                  id="showcase-ngram"
                  type="button"
                  className="verify-btn"
                  disabled={status !== 'ready'}
                  onClick={() => onRunShowcase('ngram')}
                >
                  <span>N-gram Speculative</span>
                  <small>Lossless draft</small>
                </button>
              </div>
            </div>

            <div className="inspector-card">
              <div className="inspector-title">{t.runtimeStorage}</div>
              <button
                id="btn-clear-opfs"
                type="button"
                className="load-btn secondary"
                disabled={status === 'loading' || status === 'generating'}
                onClick={onClearOpfs}
                style={{ justifyContent: 'center' }}
              >
                <Trash2 size={13} />
                <span>{t.clearOpfs}</span>
              </button>
            </div>
          </div>
        )}
      </div>
    </aside>
  );
}
