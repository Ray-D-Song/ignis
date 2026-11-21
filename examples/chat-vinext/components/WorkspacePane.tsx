'use client';

import React, { useEffect, useState } from 'react';
import {
  Activity,
  Code2,
  Cpu,
  Download,
  FileCode2,
  Layout,
  RefreshCw,
  RotateCcw,
  Save,
  Trash2,
  X,
} from 'lucide-react';
import type { detectDeviceProfile } from '../../../runtime/device-profile.mjs';

type IgnisDeviceProfile = Awaited<ReturnType<typeof detectDeviceProfile>>;
import {
  buildArtifactSrcDoc,
  type AgentArtifact,
  type AgentTraceEntry,
} from '@/lib/agent-harness';
import { COPY, type UiLang } from '@/lib/i18n';
import type { ModelPreset } from '@/lib/model-paths';
import type { ArchKind, OmniGenMode } from '@/lib/prompt-templates';
import { parseCsvRows, type WorkspaceFile } from '@/lib/workspace-fs';
import { MarkdownContent } from './MarkdownContent';

export type WorkspaceTab = 'preview' | 'files' | 'trace' | 'runtime';

export interface FsmPreset {
  id: string;
  label: string;
  schema: Record<string, unknown>;
}

interface WorkspacePaneProps {
  lang: UiLang;
  activeTab: WorkspaceTab;
  onTabChange: (tab: WorkspaceTab) => void;
  onClose: () => void;

  artifact: AgentArtifact | null;
  onClearArtifact: () => void;

  workspaceFiles: WorkspaceFile[];
  selectedFilePath: string;
  onSelectFilePath: (path: string) => void;
  onSaveFileContent: (path: string, content: string) => void;
  onResetWorkspace: () => void;

  traceEntries: AgentTraceEntry[];

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
    activeTab,
    onTabChange,
    onClose,
    artifact,
    onClearArtifact,
    workspaceFiles,
    selectedFilePath,
    onSelectFilePath,
    onSaveFileContent,
    onResetWorkspace,
    traceEntries,
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
  const [artifactSubTab, setArtifactSubTab] = useState<'render' | 'source'>('render');
  const [iframeRefreshKey, setIframeRefreshKey] = useState(0);

  const selectedFile =
    workspaceFiles.find((f) => f.path === selectedFilePath) || workspaceFiles[0] || null;
  const [draftContent, setDraftContent] = useState(selectedFile?.content || '');

  useEffect(() => {
    setDraftContent(selectedFile?.content || '');
  }, [selectedFile?.path, selectedFile?.content]);

  const downloadHtml = () => {
    if (!artifact) return;
    const html = buildArtifactSrcDoc(artifact.html, artifact.title);
    const url = URL.createObjectURL(new Blob([html], { type: 'text/html;charset=utf-8' }));
    const a = document.createElement('a');
    a.href = url;
    a.download = `${artifact.title.toLowerCase().replace(/[^a-z0-9]+/g, '-') || 'artifact'}.html`;
    a.click();
    URL.revokeObjectURL(url);
  };

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
            <Layout size={13} />
            <span>{t.tabs.preview}</span>
          </button>
          <button
            id="tab-btn-files"
            type="button"
            role="tab"
            aria-selected={activeTab === 'files'}
            className={`pane-tab ${activeTab === 'files' ? 'active' : ''}`}
            onClick={() => onTabChange('files')}
          >
            <FileCode2 size={13} />
            <span>{t.tabs.files}</span>
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
          <>
            {artifact ? (
              <>
                <div className="artifact-header">
                  <div style={{ minWidth: 0 }}>
                    <div
                      style={{
                        fontSize: 13,
                        fontWeight: 600,
                        overflow: 'hidden',
                        textOverflow: 'ellipsis',
                        whiteSpace: 'nowrap',
                      }}
                    >
                      {artifact.title}
                    </div>
                  </div>
                  <div style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
                    <div className="segmented" style={{ width: 'auto' }}>
                      <button
                        type="button"
                        className={artifactSubTab === 'render' ? 'active' : ''}
                        style={{ padding: '0 8px', height: 24, fontSize: 11.5 }}
                        onClick={() => setArtifactSubTab('render')}
                      >
                        {t.previewTabRender}
                      </button>
                      <button
                        type="button"
                        className={artifactSubTab === 'source' ? 'active' : ''}
                        style={{ padding: '0 8px', height: 24, fontSize: 11.5 }}
                        onClick={() => setArtifactSubTab('source')}
                      >
                        <Code2 size={11} />
                        {t.previewTabCode}
                      </button>
                    </div>
                    <button
                      type="button"
                      className="icon-btn"
                      style={{ height: 26, padding: '0 6px' }}
                      onClick={() => setIframeRefreshKey((k) => k + 1)}
                      title={t.refreshArtifact}
                    >
                      <RefreshCw size={12} />
                    </button>
                    <button
                      type="button"
                      className="icon-btn"
                      style={{ height: 26, padding: '0 6px' }}
                      onClick={downloadHtml}
                      title={t.downloadArtifact}
                    >
                      <Download size={12} />
                    </button>
                    <button
                      type="button"
                      className="icon-btn"
                      style={{ height: 26, padding: '0 6px' }}
                      onClick={onClearArtifact}
                      title="Show workspace file preview"
                    >
                      <X size={12} />
                    </button>
                  </div>
                </div>
                {artifactSubTab === 'render' ? (
                  <iframe
                    key={`${artifact.id}-${iframeRefreshKey}`}
                    className="artifact-iframe"
                    sandbox="allow-scripts"
                    title={artifact.title}
                    srcDoc={buildArtifactSrcDoc(artifact.html, artifact.title)}
                  />
                ) : (
                  <div style={{ padding: 14, overflow: 'auto', flex: 1 }}>
                    <pre className="tool-call-pre" style={{ maxHeight: 'none' }}>
                      {artifact.html}
                    </pre>
                  </div>
                )}
              </>
            ) : selectedFile ? (
              <div style={{ padding: 14, display: 'flex', flexDirection: 'column', gap: 10, flex: 1 }}>
                <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
                  <span style={{ fontFamily: 'var(--font-mono)', fontSize: 12.5, fontWeight: 600 }}>
                    {selectedFile.path}
                  </span>
                  <button
                    type="button"
                    className="icon-btn"
                    style={{ height: 26, fontSize: 11.5 }}
                    onClick={() => onTabChange('files')}
                  >
                    Edit
                  </button>
                </div>

                {selectedFile.kind === 'csv' && (
                  <div style={{ overflowX: 'auto' }}>
                    {(() => {
                      const rows = parseCsvRows(selectedFile.content);
                      if (rows.length === 0) return null;
                      const [header, ...body] = rows;
                      return (
                        <div className="answer-body">
                          <table>
                            <thead>
                              <tr>
                                {header.map((h, i) => (
                                  <th key={i}>{h}</th>
                                ))}
                              </tr>
                            </thead>
                            <tbody>
                              {body.map((r, ri) => (
                                <tr key={ri}>
                                  {r.map((c, ci) => (
                                    <td key={ci}>{c}</td>
                                  ))}
                                </tr>
                              ))}
                            </tbody>
                          </table>
                        </div>
                      );
                    })()}
                  </div>
                )}

                {selectedFile.kind === 'markdown' && (
                  <MarkdownContent content={selectedFile.content} />
                )}

                {selectedFile.kind === 'image' && selectedFile.assetUrl && (
                  <div style={{ display: 'flex', justifyContent: 'center', padding: 12, background: 'var(--bg)', borderRadius: 10, border: '1px solid var(--border)' }}>
                    <img
                      src={selectedFile.assetUrl}
                      alt={selectedFile.path}
                      style={{ maxWidth: '100%', maxHeight: 340, borderRadius: 6 }}
                    />
                  </div>
                )}

                {selectedFile.kind === 'audio' && selectedFile.assetUrl && (
                  <div style={{ padding: 14, background: 'var(--bg)', borderRadius: 10, border: '1px solid var(--border)' }}>
                    <audio controls src={selectedFile.assetUrl} style={{ width: '100%' }} />
                  </div>
                )}

                {selectedFile.kind === 'html' && (
                  <iframe
                    className="artifact-iframe"
                    sandbox="allow-scripts"
                    title={selectedFile.path}
                    srcDoc={buildArtifactSrcDoc(selectedFile.content, selectedFile.path)}
                  />
                )}

                {(selectedFile.kind === 'json' || selectedFile.kind === 'text') && (
                  <pre className="tool-call-pre" style={{ maxHeight: 'none', flex: 1 }}>
                    {selectedFile.content}
                  </pre>
                )}
              </div>
            ) : (
              <div style={{ margin: 'auto', padding: 28, textAlign: 'center', maxWidth: 320 }}>
                <div style={{ fontWeight: 600, marginBottom: 6 }}>{t.previewEmptyTitle}</div>
                <div style={{ fontSize: 12.5, color: 'var(--text-secondary)', lineHeight: 1.5 }}>
                  {t.previewEmptyBody}
                </div>
              </div>
            )}
          </>
        )}

        {activeTab === 'files' && (
          <div className="file-editor-wrap">
            <div style={{ display: 'flex', gap: 8, alignItems: 'center' }}>
              <select
                className="ui-select"
                value={selectedFile?.path || ''}
                onChange={(e) => onSelectFilePath(e.target.value)}
              >
                {workspaceFiles.map((f) => (
                  <option key={f.path} value={f.path}>
                    {f.path}
                    {f.modifiedByAgent ? ' ●' : ''}
                  </option>
                ))}
              </select>
              <button
                type="button"
                className="icon-btn"
                onClick={onResetWorkspace}
                title={t.filesReset}
              >
                <RotateCcw size={13} />
              </button>
            </div>

            {selectedFile?.assetUrl ? (
              <div className="inspector-card">
                <div className="inspector-title">{selectedFile.kind.toUpperCase()} ASSET</div>
                {selectedFile.kind === 'image' && (
                  <img
                    src={selectedFile.assetUrl}
                    alt={selectedFile.path}
                    style={{ maxWidth: '100%', maxHeight: 300, borderRadius: 6 }}
                  />
                )}
                {selectedFile.kind === 'audio' && (
                  <audio controls src={selectedFile.assetUrl} style={{ width: '100%' }} />
                )}
                <div style={{ fontSize: 12, color: 'var(--text-secondary)' }}>{selectedFile.content}</div>
              </div>
            ) : (
              <>
                <textarea
                  className="file-editor-area"
                  value={draftContent}
                  onChange={(e) => setDraftContent(e.target.value)}
                />
                <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                  <span style={{ fontSize: 11.5, color: 'var(--text-muted)', fontFamily: 'var(--font-mono)' }}>
                    {draftContent.length} chars
                  </span>
                  <button
                    type="button"
                    className="load-btn primary"
                    disabled={!selectedFile || draftContent === selectedFile.content}
                    onClick={() => selectedFile && onSaveFileContent(selectedFile.path, draftContent)}
                  >
                    <Save size={13} />
                    <span>{t.filesSave}</span>
                  </button>
                </div>
              </>
            )}
          </div>
        )}

        {activeTab === 'trace' && (
          <div className="trace-timeline" id="trace-timeline">
            {traceEntries.length === 0 ? (
              <div style={{ margin: '48px auto', textAlign: 'center', maxWidth: 320, color: 'var(--text-secondary)', fontSize: 12.5, lineHeight: 1.55 }}>
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
