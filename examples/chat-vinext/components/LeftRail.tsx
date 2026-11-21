'use client';

import React, { useState } from 'react';
import {
  Bot,
  Code2,
  Eye,
  FileCode2,
  FileSpreadsheet,
  FileText,
  Layout,
  MessageSquare,
  Plus,
  Volume2,
} from 'lucide-react';
import { COPY, type UiLang } from '@/lib/i18n';
import type { WorkspaceFile } from '@/lib/workspace-fs';

interface LeftRailProps {
  lang: UiLang;
  mode: 'agent' | 'chat';
  onModeChange: (mode: 'agent' | 'chat') => void;
  onNewSession: () => void;
  canReset: boolean;
  busy: boolean;
  workspaceFiles: WorkspaceFile[];
  selectedFilePath: string;
  onSelectFile: (path: string) => void;
  onCreateFile: (path: string) => void;
  supportsVision: boolean;
}

function FileKindIcon({ kind }: { kind: WorkspaceFile['kind'] }) {
  switch (kind) {
    case 'csv':
      return <FileSpreadsheet size={14} style={{ color: '#4ade80', flexShrink: 0 }} />;
    case 'markdown':
    case 'text':
      return <FileText size={14} style={{ color: '#a1a1aa', flexShrink: 0 }} />;
    case 'json':
    case 'html':
      return <FileCode2 size={14} style={{ color: '#fb923c', flexShrink: 0 }} />;
    case 'image':
      return <Eye size={14} style={{ color: '#60a5fa', flexShrink: 0 }} />;
    case 'audio':
      return <Volume2 size={14} style={{ color: '#c084fc', flexShrink: 0 }} />;
    default:
      return <FileText size={14} style={{ color: '#a1a1aa', flexShrink: 0 }} />;
  }
}

export function LeftRail(props: LeftRailProps) {
  const {
    lang,
    mode,
    onModeChange,
    onNewSession,
    canReset,
    busy,
    workspaceFiles,
    selectedFilePath,
    onSelectFile,
    onCreateFile,
    supportsVision,
  } = props;

  const t = COPY[lang];
  const [creating, setCreating] = useState(false);
  const [newName, setNewName] = useState('');

  const handleCreate = (e: React.FormEvent) => {
    e.preventDefault();
    const cleaned = newName.trim();
    if (!cleaned) return;
    onCreateFile(cleaned);
    setNewName('');
    setCreating(false);
  };

  return (
    <aside className="left-rail" id="left-rail">
      <button
        id="btn-new-session"
        type="button"
        className="new-session-btn"
        disabled={busy || !canReset}
        onClick={onNewSession}
      >
        <Plus size={14} />
        <span>{t.newSession}</span>
      </button>

      <div className="segmented" role="tablist" aria-label="Execution mode">
        <button
          id="btn-mode-agent"
          type="button"
          role="tab"
          aria-selected={mode === 'agent'}
          className={mode === 'agent' ? 'active' : ''}
          disabled={busy}
          onClick={() => onModeChange('agent')}
        >
          <Bot size={13} />
          <span>{t.modeAgent}</span>
        </button>
        <button
          id="btn-mode-chat"
          type="button"
          role="tab"
          aria-selected={mode === 'chat'}
          className={mode === 'chat' ? 'active' : ''}
          disabled={busy}
          onClick={() => onModeChange('chat')}
        >
          <MessageSquare size={13} />
          <span>{t.modeChat}</span>
        </button>
      </div>

      <div className="rail-section">
        <div className="rail-head">
          <span>{t.workspaceHeading}</span>
          <button
            type="button"
            className="icon-btn"
            style={{ height: 22, padding: '0 4px' }}
            onClick={() => setCreating((v) => !v)}
            title={t.newFile}
            aria-label={t.newFile}
          >
            <Plus size={13} />
          </button>
        </div>

        {creating && (
          <form onSubmit={handleCreate} style={{ display: 'flex', gap: 4, padding: '2px 4px 6px' }}>
            <input
              type="text"
              className="ui-input"
              placeholder="notes.md"
              value={newName}
              onChange={(e) => setNewName(e.target.value)}
              autoFocus
            />
          </form>
        )}

        {workspaceFiles.map((f) => {
          const isActive = f.path === selectedFilePath;
          return (
            <button
              key={f.path}
              type="button"
              className={`rail-file ${isActive ? 'active' : ''}`}
              onClick={() => onSelectFile(f.path)}
            >
              <span className="rail-file-left">
                <FileKindIcon kind={f.kind} />
                <span className="rail-file-name">{f.path}</span>
              </span>
              {f.modifiedByAgent && (
                <span
                  title={t.agentModified}
                  style={{
                    width: 6,
                    height: 6,
                    borderRadius: '50%',
                    background: 'var(--accent)',
                    flexShrink: 0,
                  }}
                />
              )}
            </button>
          );
        })}
      </div>

      {mode === 'agent' && (
        <div className="rail-section" style={{ marginTop: 4 }}>
          <div className="rail-head">
            <span>{t.toolsHeading}</span>
          </div>
          <div className="tool-item">
            <FileText size={13} style={{ color: 'var(--text-secondary)', marginTop: 2, flexShrink: 0 }} />
            <div>
              <div className="tool-item-name">read_file / write_file</div>
              <div className="tool-item-desc">{t.toolDescRead} · {t.toolDescWrite}</div>
            </div>
          </div>
          <div className="tool-item">
            <Code2 size={13} style={{ color: 'var(--text-secondary)', marginTop: 2, flexShrink: 0 }} />
            <div>
              <div className="tool-item-name">run_js</div>
              <div className="tool-item-desc">{t.toolDescJs}</div>
            </div>
          </div>
          <div className="tool-item">
            <Layout size={13} style={{ color: 'var(--text-secondary)', marginTop: 2, flexShrink: 0 }} />
            <div>
              <div className="tool-item-name">render_artifact</div>
              <div className="tool-item-desc">{t.toolDescArtifact}</div>
            </div>
          </div>
          {supportsVision && (
            <div className="tool-item">
              <Eye size={13} style={{ color: 'var(--text-secondary)', marginTop: 2, flexShrink: 0 }} />
              <div>
                <div className="tool-item-name">inspect_image</div>
                <div className="tool-item-desc">{t.toolDescVision}</div>
              </div>
            </div>
          )}
        </div>
      )}
    </aside>
  );
}
