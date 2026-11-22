'use client';

import React, { useEffect, useRef, useState } from 'react';
import {
  Check,
  ChevronDown,
  Code2,
  Cpu,
  Download,
  Eye,
  Loader2,
  Mic,
  Moon,
  PanelRight,
  RefreshCw,
  ShieldAlert,
  ShieldCheck,
  Sun,
  Volume2,
} from 'lucide-react';
import type { ApprovalPolicy } from '@/lib/agent-harness';
import { COPY, type UiLang } from '@/lib/i18n';
import type { ModelPreset } from '@/lib/model-paths';

export const CUSTOM_MODEL_ID = '__custom__';

interface TopBarProps {
  lang: UiLang;
  onToggleLang: () => void;
  theme: 'dark' | 'light';
  onToggleTheme: () => void;
  editorOpen: boolean;
  onToggleEditor: () => void;
  rightPaneOpen: boolean;
  onToggleRightPane: () => void;
  approvalPolicy: ApprovalPolicy;
  onToggleApprovalPolicy: () => void;
  models: ModelPreset[];
  selectedModelId: string;
  activeModel: ModelPreset;
  customUrl: string;
  onCustomUrlChange: (url: string) => void;
  onSelectModel: (id: string) => void;
  termsAcked: boolean;
  onTermsAckChange: (acked: boolean) => void;
  status: 'idle' | 'loading' | 'ready' | 'generating' | 'error';
  statusText: string;
  loadPct: number;
  onLoadModel: () => void;
}

export function SparkMark({ size = 16 }: { size?: number }) {
  return (
    <svg width={size} height={size} viewBox="0 0 24 24" fill="none" aria-hidden="true">
      <path
        d="M13.2 2.5L5.5 13.4H11.5L10.4 21.5L18.5 10.6H12.4L13.2 2.5Z"
        fill="#f97316"
        stroke="#fb923c"
        strokeWidth="1.1"
        strokeLinejoin="round"
      />
    </svg>
  );
}

export function TopBar(props: TopBarProps) {
  const {
    lang,
    onToggleLang,
    theme,
    onToggleTheme,
    editorOpen,
    onToggleEditor,
    rightPaneOpen,
    onToggleRightPane,
    approvalPolicy,
    onToggleApprovalPolicy,
    models,
    selectedModelId,
    activeModel,
    customUrl,
    onCustomUrlChange,
    onSelectModel,
    termsAcked,
    onTermsAckChange,
    status,
    statusText,
    loadPct,
    onLoadModel,
  } = props;

  const t = COPY[lang];
  const [popoverOpen, setPopoverOpen] = useState(false);
  const pickerRef = useRef<HTMLDivElement | null>(null);

  useEffect(() => {
    const handleOutside = (ev: MouseEvent) => {
      if (pickerRef.current && !pickerRef.current.contains(ev.target as Node)) {
        setPopoverOpen(false);
      }
    };
    document.addEventListener('mousedown', handleOutside);
    return () => document.removeEventListener('mousedown', handleOutside);
  }, []);

  const dotClass =
    status === 'ready'
      ? 'ready'
      : status === 'idle'
        ? 'idle'
        : status === 'loading' || status === 'generating'
          ? 'busy'
          : status === 'error'
            ? 'error'
            : '';

  const termsBlocked = Boolean(activeModel?.requiresTermsAck && !termsAcked);
  const displayName = activeModel.isCustom
    ? t.customOption
    : activeModel.shortName || activeModel.label.split('·')[0].trim();

  return (
    <header className="topbar">
      <div className="topbar-left">
        <div className="brand-lockup">
          <SparkMark size={17} />
          <span className="brand-wordmark">{t.brandName}</span>
        </div>

        <div className="topbar-divider hide-mobile" />

        <div className="model-picker-wrap" ref={pickerRef}>
          {/* Keep accessible native <select> synced as FIRST select in DOM */}
          <select
            id="model-select"
            aria-label={t.selectModel}
            className="sr-select"
            value={selectedModelId}
            disabled={status === 'loading' || status === 'generating'}
            onChange={(e) => onSelectModel(e.target.value)}
          >
            {models.map((m) => (
              <option key={m.id} value={m.id}>
                {m.label}
              </option>
            ))}
            <option value={CUSTOM_MODEL_ID}>{t.customOption}</option>
          </select>

          <button
            id="btn-model-trigger"
            type="button"
            className="model-trigger"
            aria-expanded={popoverOpen}
            onClick={() => setPopoverOpen((v) => !v)}
          >
            <span style={{ overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>
              {displayName}
            </span>
            {activeModel.sizeLabel && !activeModel.isCustom && (
              <span className="model-size">{activeModel.sizeLabel}</span>
            )}
            <ChevronDown size={14} style={{ color: 'var(--text-muted)', flexShrink: 0 }} />
          </button>

          {popoverOpen && (
            <div className="model-popover" id="model-popover" role="dialog" aria-label={t.selectModel}>
              {models.map((m) => {
                const isSel = m.id === selectedModelId;
                return (
                  <button
                    key={m.id}
                    type="button"
                    className={`model-option ${isSel ? 'selected' : ''}`}
                    disabled={status === 'loading' || status === 'generating'}
                    onClick={() => {
                      onSelectModel(m.id);
                      if (!m.requiresTermsAck) setPopoverOpen(false);
                    }}
                  >
                    <div className="model-option-head">
                      <span style={{ display: 'inline-flex', alignItems: 'center', gap: 6 }}>
                        {m.shortName || m.label.split('·')[0].trim()}
                        {isSel && <Check size={13} style={{ color: 'var(--accent)' }} />}
                      </span>
                      <span style={{ fontFamily: 'var(--font-mono)', fontSize: 11, color: 'var(--text-muted)' }}>
                        {m.sizeLabel || `${Math.round((m.gpuMiB + m.embdMiB + m.pleMiB) / 100) / 10} GB`}
                      </span>
                    </div>
                    <div className="model-option-desc">{m.description}</div>
                    <div className="cap-row">
                      {m.capabilities.vision && (
                        <span className="cap-tag">
                          <Eye size={10} style={{ verticalAlign: '-1px', marginRight: 3 }} />
                          Vision
                        </span>
                      )}
                      {m.capabilities.audioIn && (
                        <span className="cap-tag">
                          <Mic size={10} style={{ verticalAlign: '-1px', marginRight: 3 }} />
                          Audio
                        </span>
                      )}
                      {m.capabilities.audioOut && (
                        <span className="cap-tag">
                          <Volume2 size={10} style={{ verticalAlign: '-1px', marginRight: 3 }} />
                          Voice
                        </span>
                      )}
                      {m.capabilities.thinking && (
                        <span className="cap-tag">
                          <Cpu size={10} style={{ verticalAlign: '-1px', marginRight: 3 }} />
                          Reasoning
                        </span>
                      )}
                    </div>
                  </button>
                );
              })}

              <button
                type="button"
                className={`model-option ${selectedModelId === CUSTOM_MODEL_ID ? 'selected' : ''}`}
                disabled={status === 'loading' || status === 'generating'}
                onClick={() => onSelectModel(CUSTOM_MODEL_ID)}
              >
                <div className="model-option-head">
                  <span>{t.customOption}</span>
                  {selectedModelId === CUSTOM_MODEL_ID && <Check size={13} style={{ color: 'var(--accent)' }} />}
                </div>
                <div className="model-option-desc">{t.customUrlLabel}</div>
              </button>

              {selectedModelId === CUSTOM_MODEL_ID && (
                <div style={{ padding: '4px 6px 6px' }}>
                  <input
                    id="custom-model-url"
                    type="url"
                    className="ui-input"
                    value={customUrl}
                    disabled={status === 'loading' || status === 'generating'}
                    placeholder={t.customUrlPlaceholder}
                    onChange={(e) => onCustomUrlChange(e.target.value)}
                  />
                </div>
              )}

              {activeModel?.requiresTermsAck && (
                <div className="terms-banner">
                  <div>{t.termsNotice}</div>
                  <label>
                    <input
                      id="gemma-terms-checkbox"
                      type="checkbox"
                      checked={termsAcked}
                      onChange={(e) => onTermsAckChange(e.target.checked)}
                    />
                    <span>
                      {t.termsAck}{' '}
                      <a href="https://ai.google.dev/gemma/terms" target="_blank" rel="noreferrer">
                        ↗
                      </a>
                    </span>
                  </label>
                </div>
              )}
            </div>
          )}
        </div>

        <button
          id="btn-load-model"
          type="button"
          className={`load-btn ${status === 'ready' ? 'secondary' : 'primary'}`}
          disabled={!activeModel || termsBlocked || status === 'loading' || status === 'generating'}
          onClick={() => {
            if (termsBlocked) {
              setPopoverOpen(true);
              return;
            }
            onLoadModel();
          }}
        >
          {status === 'ready' ? (
            <>
              <RefreshCw size={12} />
              <span>{t.reloadModel}</span>
            </>
          ) : status === 'loading' ? (
            <>
              <Loader2 size={13} className="spin-icon" />
              <span>{t.loadingPct(loadPct)}</span>
            </>
          ) : (
            <>
              <Download size={12} />
              <span>
                {t.loadModel}
                {activeModel.sizeLabel ? ` (${activeModel.sizeLabel})` : ''}
              </span>
            </>
          )}
        </button>

        <button
          id="btn-approval-policy"
          type="button"
          className={`icon-btn hide-mobile ${approvalPolicy === 'ask' ? 'active' : ''}`}
          style={{
            borderColor: approvalPolicy === 'ask' ? 'var(--accent-border)' : undefined,
            color: approvalPolicy === 'ask' ? '#fdba74' : undefined,
          }}
          onClick={onToggleApprovalPolicy}
          title={approvalPolicy === 'ask' ? t.policyAskTitle : t.policyAutoTitle}
        >
          {approvalPolicy === 'ask' ? <ShieldAlert size={13} /> : <ShieldCheck size={13} />}
          <span>{approvalPolicy === 'ask' ? t.policyAsk : t.policyAuto}</span>
        </button>
      </div>

      <div className="topbar-right">
        <div className={`status-pill ${status}`} title={statusText}>
          <span className={`status-dot ${dotClass}`} />
          <span id="status-text">{statusText}</span>
        </div>

        <button
          id="btn-lang-toggle"
          type="button"
          className="icon-btn"
          onClick={onToggleLang}
          title="Switch language"
        >
          {lang === 'en' ? '中文' : 'EN'}
        </button>

        <button
          id="btn-theme-toggle"
          type="button"
          className="icon-btn"
          onClick={onToggleTheme}
          title={theme === 'dark' ? t.themeLightTitle : t.themeDarkTitle}
          aria-label={theme === 'dark' ? t.themeLightTitle : t.themeDarkTitle}
          aria-pressed={theme === 'light'}
        >
          {theme === 'dark' ? <Sun size={15} /> : <Moon size={15} />}
        </button>

        <button
          id="btn-toggle-code"
          type="button"
          className={`icon-btn hide-mobile ${editorOpen ? 'active' : ''}`}
          onClick={onToggleEditor}
          title="Toggle CodeMirror Editor"
          aria-label="Toggle Code Editor"
        >
          <Code2 size={15} />
        </button>

        <button
          id="btn-toggle-workspace"
          type="button"
          className={`icon-btn ${rightPaneOpen ? 'active' : ''}`}
          onClick={onToggleRightPane}
          title="Toggle Full-Stack Preview, Trace & Runtime"
          aria-label="Toggle Workspace Pane"
        >
          <PanelRight size={16} />
        </button>

        <a
          href="https://github.com/Ray-D-Song/Ignis"
          target="_blank"
          rel="noreferrer"
          className="icon-btn hide-mobile"
          title="Ignis on GitHub"
          aria-label="Ignis GitHub"
        >
          <svg width="15" height="15" viewBox="0 0 24 24" fill="currentColor" aria-hidden="true">
            <path d="M12 .5C5.65.5.5 5.65.5 12a11.5 11.5 0 0 0 7.86 10.91c.58.11.79-.25.79-.55v-2.16c-3.2.7-3.87-1.36-3.87-1.36-.52-1.33-1.28-1.68-1.28-1.68-1.05-.72.08-.7.08-.7 1.16.08 1.77 1.19 1.77 1.19 1.03 1.76 2.7 1.25 3.36.96.1-.75.4-1.25.73-1.54-2.55-.29-5.23-1.28-5.23-5.68 0-1.26.45-2.28 1.18-3.09-.12-.29-.51-1.46.11-3.04 0 0 .97-.31 3.17 1.18a10.96 10.96 0 0 1 5.76 0c2.2-1.49 3.16-1.18 3.16-1.18.63 1.58.24 2.75.12 3.04.74.81 1.18 1.83 1.18 3.09 0 4.41-2.69 5.38-5.25 5.67.41.36.78 1.07.78 2.15v3.19c0 .3.21.67.8.55A11.5 11.5 0 0 0 23.5 12C23.5 5.65 18.35.5 12 .5Z" />
          </svg>
        </a>
      </div>
    </header>
  );
}
