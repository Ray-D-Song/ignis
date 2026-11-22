'use client';

import React, { useEffect, useRef, useState } from 'react';
import {
  Download,
  FileCode2,
  FileJson,
  FileText,
  FolderTree,
  GitCompare,
  Plus,
  RotateCcw,
  Save,
  Undo2,
  Upload,
} from 'lucide-react';
import { COPY, type UiLang } from '@/lib/i18n';
import type { WorkspaceFile } from '@/lib/workspace-fs';

interface CodeEditorPaneProps {
  lang: UiLang;
  theme?: 'dark' | 'light';
  files: WorkspaceFile[];
  selectedPath: string;
  editorMode: 'code' | 'diff';
  diffOriginals: Record<string, string>;
  onSelectFile: (path: string) => void;
  onModeChange: (mode: 'code' | 'diff') => void;
  onSaveFile: (path: string, content: string) => void;
  onRevertFile: (path: string) => void;
  onCreateFile: (path: string) => void;
  onUploadFiles: (files: FileList) => void;
  onResetProject: () => void;
  busy?: boolean;
}

interface CmModules {
  EditorState: typeof import('@codemirror/state').EditorState;
  EditorView: typeof import('@codemirror/view').EditorView;
  lineNumbers: typeof import('@codemirror/view').lineNumbers;
  highlightActiveLine: typeof import('@codemirror/view').highlightActiveLine;
  highlightActiveLineGutter: typeof import('@codemirror/view').highlightActiveLineGutter;
  drawSelection: typeof import('@codemirror/view').drawSelection;
  keymap: typeof import('@codemirror/view').keymap;
  defaultKeymap: typeof import('@codemirror/commands').defaultKeymap;
  history: typeof import('@codemirror/commands').history;
  historyKeymap: typeof import('@codemirror/commands').historyKeymap;
  indentWithTab: typeof import('@codemirror/commands').indentWithTab;
  bracketMatching: typeof import('@codemirror/language').bracketMatching;
  indentOnInput: typeof import('@codemirror/language').indentOnInput;
  HighlightStyle: typeof import('@codemirror/language').HighlightStyle;
  syntaxHighlighting: typeof import('@codemirror/language').syntaxHighlighting;
  highlightSelectionMatches: typeof import('@codemirror/search').highlightSelectionMatches;
  searchKeymap: typeof import('@codemirror/search').searchKeymap;
  closeBrackets: typeof import('@codemirror/autocomplete').closeBrackets;
  unifiedMergeView: typeof import('@codemirror/merge').unifiedMergeView;
  javascript: typeof import('@codemirror/lang-javascript').javascript;
  html: typeof import('@codemirror/lang-html').html;
  css: typeof import('@codemirror/lang-css').css;
  json: typeof import('@codemirror/lang-json').json;
  markdown: typeof import('@codemirror/lang-markdown').markdown;
  tags: typeof import('@lezer/highlight').tags;
}

let cmModulesPromise: Promise<CmModules> | null = null;

function loadCmModules(): Promise<CmModules> {
  if (!cmModulesPromise) {
    cmModulesPromise = Promise.all([
      import('@codemirror/state'),
      import('@codemirror/view'),
      import('@codemirror/commands'),
      import('@codemirror/language'),
      import('@codemirror/search'),
      import('@codemirror/autocomplete'),
      import('@codemirror/merge'),
      import('@codemirror/lang-javascript'),
      import('@codemirror/lang-html'),
      import('@codemirror/lang-css'),
      import('@codemirror/lang-json'),
      import('@codemirror/lang-markdown'),
      import('@lezer/highlight'),
    ]).then(([st, vw, cmd, lang, srch, ac, mrg, js, htm, cs, jsn, md, hl]) => ({
      EditorState: st.EditorState,
      EditorView: vw.EditorView,
      lineNumbers: vw.lineNumbers,
      highlightActiveLine: vw.highlightActiveLine,
      highlightActiveLineGutter: vw.highlightActiveLineGutter,
      drawSelection: vw.drawSelection,
      keymap: vw.keymap,
      defaultKeymap: cmd.defaultKeymap,
      history: cmd.history,
      historyKeymap: cmd.historyKeymap,
      indentWithTab: cmd.indentWithTab,
      bracketMatching: lang.bracketMatching,
      indentOnInput: lang.indentOnInput,
      HighlightStyle: lang.HighlightStyle,
      syntaxHighlighting: lang.syntaxHighlighting,
      highlightSelectionMatches: srch.highlightSelectionMatches,
      searchKeymap: srch.searchKeymap,
      closeBrackets: ac.closeBrackets,
      unifiedMergeView: mrg.unifiedMergeView,
      javascript: js.javascript,
      html: htm.html,
      css: cs.css,
      json: jsn.json,
      markdown: md.markdown,
      tags: hl.tags,
    }));
  }
  return cmModulesPromise;
}

function FileIcon({ kind }: { kind: WorkspaceFile['kind'] }) {
  switch (kind) {
    case 'json':
      return <FileJson size={13} style={{ color: '#fbbf24', flexShrink: 0 }} />;
    case 'jsx':
    case 'tsx':
      return <FileCode2 size={13} style={{ color: '#38bdf8', flexShrink: 0 }} />;
    case 'js':
    case 'ts':
      return <FileCode2 size={13} style={{ color: '#fb923c', flexShrink: 0 }} />;
    case 'css':
    case 'html':
      return <FileCode2 size={13} style={{ color: '#c084fc', flexShrink: 0 }} />;
    default:
      return <FileText size={13} style={{ color: '#a1a1aa', flexShrink: 0 }} />;
  }
}

export function CodeEditorPane(props: CodeEditorPaneProps) {
  const {
    lang,
    theme = 'dark',
    files,
    selectedPath,
    editorMode,
    diffOriginals,
    onSelectFile,
    onModeChange,
    onSaveFile,
    onRevertFile,
    onCreateFile,
    onUploadFiles,
    onResetProject,
    busy = false,
  } = props;

  const t = COPY[lang];
  const activeFile = files.find((f) => f.path === selectedPath) || files[0] || null;
  const activePath = activeFile?.path || '';
  const activeContent = activeFile?.content || '';
  const diffBase = activePath ? diffOriginals[activePath] : undefined;
  const hasDiff = diffBase !== undefined && diffBase !== activeContent;
  const effectiveMode: 'code' | 'diff' = editorMode === 'diff' && hasDiff ? 'diff' : 'code';

  const [treeOpen, setTreeOpen] = useState(true);
  const [creating, setCreating] = useState(false);
  const [newName, setNewName] = useState('');
  const [draftText, setDraftText] = useState(activeContent);
  const [dirty, setDirty] = useState(false);
  const [dragOver, setDragOver] = useState(false);

  const hostRef = useRef<HTMLDivElement | null>(null);
  const viewRef = useRef<import('@codemirror/view').EditorView | null>(null);
  const uploadInputRef = useRef<HTMLInputElement | null>(null);
  const onSaveRef = useRef(onSaveFile);
  onSaveRef.current = onSaveFile;
  const activePathRef = useRef(activePath);
  activePathRef.current = activePath;

  useEffect(() => {
    setDraftText(activeContent);
    setDirty(false);
  }, [activePath, activeContent]);

  useEffect(() => {
    let cancelled = false;
    const host = hostRef.current;
    if (!host || !activeFile) return;

    (async () => {
      const cm = await loadCmModules();
      if (cancelled || !hostRef.current) return;

      if (viewRef.current) {
        viewRef.current.destroy();
        viewRef.current = null;
      }

      let langExt;
      switch (activeFile.kind) {
        case 'jsx':
        case 'tsx':
          langExt = cm.javascript({ jsx: true, typescript: activeFile.kind === 'tsx' });
          break;
        case 'js':
        case 'ts':
          langExt = cm.javascript({ jsx: false, typescript: activeFile.kind === 'ts' });
          break;
        case 'html':
          langExt = cm.html();
          break;
        case 'css':
          langExt = cm.css();
          break;
        case 'json':
          langExt = cm.json();
          break;
        case 'markdown':
          langExt = cm.markdown();
          break;
        default:
          langExt = cm.javascript({ jsx: false });
          break;
      }

      const darkTheme = cm.EditorView.theme(
        {
          '&': {
            backgroundColor: '#09090b',
            color: '#e4e4e7',
            height: '100%',
            fontSize: '12px',
            fontFamily: "var(--font-mono)",
          },
          '.cm-scroller': {
            overflow: 'auto',
            fontFamily: "var(--font-mono)",
            lineHeight: '1.52',
          },
          '.cm-content': {
            padding: '10px 0',
            caretColor: '#f97316',
          },
          '.cm-gutters': {
            backgroundColor: '#0d0d10',
            color: '#52525b',
            borderRight: '1px solid rgba(255,255,255,0.06)',
            minWidth: '38px',
          },
          '.cm-activeLineGutter': {
            backgroundColor: '#18181b',
            color: '#a1a1aa',
          },
          '.cm-activeLine': {
            backgroundColor: 'rgba(255, 255, 255, 0.03)',
          },
          '&.cm-focused .cm-selectionBackground, .cm-selectionBackground': {
            backgroundColor: 'rgba(249, 115, 22, 0.22)',
          },
          '.cm-matchingBracket': {
            backgroundColor: 'rgba(249, 115, 22, 0.28)',
            outline: '1px solid rgba(249, 115, 22, 0.5)',
          },
          '.cm-changedLine': {
            backgroundColor: 'rgba(34, 197, 94, 0.08)',
          },
          '.cm-deletedChunk': {
            backgroundColor: 'rgba(239, 68, 68, 0.1)',
          },
        },
        { dark: true },
      );

      const darkHighlight = cm.HighlightStyle.define([
        { tag: cm.tags.keyword, color: '#fb923c' },
        { tag: [cm.tags.name, cm.tags.deleted, cm.tags.character, cm.tags.macroName], color: '#e4e4e7' },
        { tag: [cm.tags.propertyName, cm.tags.attributeName], color: '#93c5fd' },
        { tag: [cm.tags.function(cm.tags.variableName), cm.tags.labelName], color: '#60a5fa' },
        { tag: [cm.tags.color, cm.tags.constant(cm.tags.name), cm.tags.standard(cm.tags.name)], color: '#facc15' },
        { tag: [cm.tags.definition(cm.tags.name), cm.tags.separator], color: '#fafafa' },
        { tag: [cm.tags.typeName, cm.tags.className, cm.tags.number, cm.tags.changed, cm.tags.annotation, cm.tags.modifier], color: '#fdba74' },
        { tag: [cm.tags.operator, cm.tags.operatorKeyword], color: '#a1a1aa' },
        { tag: [cm.tags.string, cm.tags.special(cm.tags.string)], color: '#86efac' },
        { tag: [cm.tags.meta, cm.tags.comment], color: '#71717a', fontStyle: 'italic' },
        { tag: cm.tags.bool, color: '#f472b6' },
      ]);

      const lightTheme = cm.EditorView.theme(
        {
          '&': {
            backgroundColor: '#ffffff',
            color: '#0f172a',
            height: '100%',
            fontSize: '12px',
            fontFamily: "var(--font-mono)",
          },
          '.cm-scroller': {
            overflow: 'auto',
            fontFamily: "var(--font-mono)",
            lineHeight: '1.52',
          },
          '.cm-content': {
            padding: '10px 0',
            caretColor: '#ea580c',
          },
          '.cm-gutters': {
            backgroundColor: '#f8fafc',
            color: '#94a3b8',
            borderRight: '1px solid rgba(15, 23, 42, 0.08)',
            minWidth: '38px',
          },
          '.cm-activeLineGutter': {
            backgroundColor: '#f1f5f9',
            color: '#334155',
          },
          '.cm-activeLine': {
            backgroundColor: 'rgba(15, 23, 42, 0.03)',
          },
          '&.cm-focused .cm-selectionBackground, .cm-selectionBackground': {
            backgroundColor: 'rgba(234, 88, 12, 0.16)',
          },
          '.cm-matchingBracket': {
            backgroundColor: 'rgba(234, 88, 12, 0.2)',
            outline: '1px solid rgba(234, 88, 12, 0.45)',
          },
          '.cm-changedLine': {
            backgroundColor: 'rgba(22, 163, 74, 0.08)',
          },
          '.cm-deletedChunk': {
            backgroundColor: 'rgba(220, 38, 38, 0.08)',
          },
        },
        { dark: false },
      );

      const lightHighlight = cm.HighlightStyle.define([
        { tag: cm.tags.keyword, color: '#c2410c' },
        { tag: [cm.tags.name, cm.tags.deleted, cm.tags.character, cm.tags.macroName], color: '#0f172a' },
        { tag: [cm.tags.propertyName, cm.tags.attributeName], color: '#0369a1' },
        { tag: [cm.tags.function(cm.tags.variableName), cm.tags.labelName], color: '#1d4ed8' },
        { tag: [cm.tags.color, cm.tags.constant(cm.tags.name), cm.tags.standard(cm.tags.name)], color: '#b45309' },
        { tag: [cm.tags.definition(cm.tags.name), cm.tags.separator], color: '#0f172a' },
        { tag: [cm.tags.typeName, cm.tags.className, cm.tags.number, cm.tags.changed, cm.tags.annotation, cm.tags.modifier], color: '#9a3412' },
        { tag: [cm.tags.operator, cm.tags.operatorKeyword], color: '#475569' },
        { tag: [cm.tags.string, cm.tags.special(cm.tags.string)], color: '#15803d' },
        { tag: [cm.tags.meta, cm.tags.comment], color: '#64748b', fontStyle: 'italic' },
        { tag: cm.tags.bool, color: '#be185d' },
      ]);

      const activeCmTheme = theme === 'light' ? lightTheme : darkTheme;
      const activeCmHighlight = theme === 'light' ? lightHighlight : darkHighlight;

      const saveShortcut = cm.keymap.of([
        {
          key: 'Mod-s',
          run: (view) => {
            onSaveRef.current(activePathRef.current, view.state.doc.toString());
            setDirty(false);
            return true;
          },
        },
      ]);

      const extensions = [
        cm.lineNumbers(),
        cm.highlightActiveLineGutter(),
        cm.highlightActiveLine(),
        cm.drawSelection(),
        cm.history(),
        cm.indentOnInput(),
        cm.bracketMatching(),
        cm.closeBrackets(),
        cm.highlightSelectionMatches(),
        saveShortcut,
        cm.keymap.of([
          ...cm.defaultKeymap,
          ...cm.historyKeymap,
          ...cm.searchKeymap,
          cm.indentWithTab,
        ]),
        langExt,
        activeCmTheme,
        cm.syntaxHighlighting(activeCmHighlight),
        cm.EditorView.updateListener.of((upd) => {
          if (upd.docChanged) {
            const val = upd.state.doc.toString();
            setDraftText(val);
            setDirty(val !== activeContent);
          }
        }),
      ];

      if (effectiveMode === 'diff' && diffBase !== undefined) {
        extensions.push(
          cm.unifiedMergeView({
            original: diffBase,
            highlightChanges: true,
            gutter: true,
            mergeControls: false,
          }),
        );
      }

      const state = cm.EditorState.create({
        doc: activeContent,
        extensions,
      });

      viewRef.current = new cm.EditorView({
        state,
        parent: hostRef.current,
      });
    })();

    return () => {
      cancelled = true;
      if (viewRef.current) {
        viewRef.current.destroy();
        viewRef.current = null;
      }
    };
  }, [activePath, activeContent, effectiveMode, diffBase, theme]);

  const handleDownload = () => {
    if (!activeFile) return;
    const blob = new Blob([draftText], { type: 'text/plain;charset=utf-8' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = activeFile.path.split('/').pop() || 'file.txt';
    a.click();
    URL.revokeObjectURL(url);
  };

  return (
    <section
      className={`editor-pane ${dragOver ? 'drag-over' : ''}`}
      id="code-editor-pane"
      onDragOver={(e) => {
        e.preventDefault();
        setDragOver(true);
      }}
      onDragLeave={() => setDragOver(false)}
      onDrop={(e) => {
        e.preventDefault();
        setDragOver(false);
        if (e.dataTransfer?.files?.length) {
          onUploadFiles(e.dataTransfer.files);
        }
      }}
    >
      <div className="editor-toolbar">
        <div className="editor-toolbar-left">
          <button
            type="button"
            className={`icon-btn ${treeOpen ? 'active' : ''}`}
            style={{ height: 26, padding: '0 6px' }}
            onClick={() => setTreeOpen((v) => !v)}
            title={t.explorerHeading}
          >
            <FolderTree size={13} />
          </button>
          <span className="editor-filepath" id="editor-active-path" title={activePath}>
            {activePath || '—'}
          </span>
          {dirty && <span className="editor-dirty-dot" title="Unsaved changes" />}
          {activeFile?.modifiedByAgent && (
            <span
              className="feature-badge ok"
              style={{ padding: '1px 6px', fontSize: 10, whiteSpace: 'nowrap' }}
              title={t.agentModified}
            >
              +{activeFile.addedLines ?? 0}/-{activeFile.removedLines ?? 0}
            </span>
          )}
        </div>

        <div className="editor-toolbar-right">
          {hasDiff && (
            <div className="segmented" style={{ width: 'auto', padding: 2 }}>
              <button
                id="btn-editor-code"
                type="button"
                className={effectiveMode === 'code' ? 'active' : ''}
                style={{ height: 22, padding: '0 7px', fontSize: 11 }}
                onClick={() => onModeChange('code')}
              >
                {t.codeTab}
              </button>
              <button
                id="btn-editor-diff"
                type="button"
                className={effectiveMode === 'diff' ? 'active' : ''}
                style={{ height: 22, padding: '0 7px', fontSize: 11 }}
                onClick={() => onModeChange('diff')}
              >
                <GitCompare size={11} />
                {t.diffTab}
              </button>
            </div>
          )}

          {hasDiff && (
            <button
              id="btn-revert-file"
              type="button"
              className="icon-btn"
              style={{ height: 26, padding: '0 6px' }}
              disabled={busy}
              onClick={() => activePath && onRevertFile(activePath)}
              title={t.revertFile}
              aria-label={t.revertFile}
            >
              <Undo2 size={12} />
            </button>
          )}

          <button
            id="btn-save-file"
            type="button"
            className="load-btn secondary"
            style={{ height: 26, padding: '0 8px', fontSize: 11.5 }}
            disabled={!activeFile || !dirty || busy}
            onClick={() => activeFile && onSaveFile(activeFile.path, draftText)}
            title={t.saveFile}
          >
            <Save size={12} />
            <span className="hide-mobile">{t.saveFile}</span>
          </button>

          <button
            type="button"
            className="icon-btn"
            style={{ height: 26, padding: '0 6px' }}
            disabled={!activeFile}
            onClick={handleDownload}
            title={t.downloadFile}
          >
            <Download size={12} />
          </button>
        </div>
      </div>

      <div className="editor-body-row">
        {treeOpen && (
          <aside className="file-tree-col" id="file-explorer">
            <div className="file-tree-head">
              <span>{t.explorerHeading}</span>
              <div style={{ display: 'flex', gap: 2 }}>
                <input
                  ref={uploadInputRef}
                  type="file"
                  multiple
                  style={{ display: 'none' }}
                  onChange={(e) => {
                    if (e.target.files?.length) onUploadFiles(e.target.files);
                    e.target.value = '';
                  }}
                />
                <button
                  type="button"
                  className="icon-btn"
                  style={{ height: 20, padding: '0 4px' }}
                  onClick={() => uploadInputRef.current?.click()}
                  title={t.uploadFile}
                >
                  <Upload size={11} />
                </button>
                <button
                  type="button"
                  className="icon-btn"
                  style={{ height: 20, padding: '0 4px' }}
                  onClick={() => setCreating((v) => !v)}
                  title={t.newFile}
                >
                  <Plus size={12} />
                </button>
                <button
                  id="btn-reset-project"
                  type="button"
                  className="icon-btn"
                  style={{ height: 20, padding: '0 4px' }}
                  disabled={busy}
                  onClick={onResetProject}
                  title={t.resetProject}
                >
                  <RotateCcw size={11} />
                </button>
              </div>
            </div>

            {creating && (
              <form
                onSubmit={(e) => {
                  e.preventDefault();
                  const clean = newName.trim();
                  if (!clean) return;
                  onCreateFile(clean);
                  setNewName('');
                  setCreating(false);
                }}
                style={{ padding: '0 6px 6px' }}
              >
                <input
                  type="text"
                  className="ui-input"
                  style={{ padding: '4px 7px', fontSize: 11.5 }}
                  placeholder="server/routes.mjs"
                  value={newName}
                  onChange={(e) => setNewName(e.target.value)}
                  autoFocus
                />
              </form>
            )}

            <div className="file-tree-list">
              {files.map((f) => {
                const isActive = f.path === activePath;
                return (
                  <button
                    key={f.path}
                    type="button"
                    className={`tree-item ${isActive ? 'active' : ''}`}
                    onClick={() => onSelectFile(f.path)}
                    title={f.path}
                  >
                    <span className="tree-item-left">
                      <FileIcon kind={f.kind} />
                      <span className="tree-item-name">{f.path}</span>
                    </span>
                    {f.modifiedByAgent && <span className="tree-mod-dot" title={t.agentModified} />}
                  </button>
                );
              })}
            </div>
          </aside>
        )}

        <div className="codemirror-host" ref={hostRef} />
      </div>
    </section>
  );
}
