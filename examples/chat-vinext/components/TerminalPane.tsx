'use client';

import React, { useEffect, useRef, useState } from 'react';
import { RefreshCw, Terminal as TerminalIcon } from 'lucide-react';
import { COPY, type UiLang } from '@/lib/i18n';
import type { TerminalChannel } from '@/lib/terminal-channel';
import type { InteractiveShellHandle, SandboxBootStatus } from '@/lib/webcontainer-manager';

interface TerminalPaneProps {
  lang: UiLang;
  theme?: 'dark' | 'light';
  sandboxStatus: SandboxBootStatus;
  sandboxDetail: string;
  shellChannel: TerminalChannel;
  serverChannel: TerminalChannel;
  onAttachShell: (
    cols: number,
    rows: number,
    onData: (chunk: string) => void,
  ) => Promise<InteractiveShellHandle>;
  onRestartServer: () => void;
}

const DARK_TERM_THEME = {
  background: '#09090b',
  foreground: '#e4e4e7',
  cursor: '#f97316',
  selectionBackground: 'rgba(249, 115, 22, 0.25)',
  black: '#18181b',
  green: '#4ade80',
  yellow: '#facc15',
  red: '#f87171',
  blue: '#60a5fa',
  cyan: '#38bdf8',
};

const LIGHT_TERM_THEME = {
  background: '#f8fafc',
  foreground: '#0f172a',
  cursor: '#ea580c',
  selectionBackground: 'rgba(234, 88, 12, 0.22)',
  black: '#e2e8f0',
  green: '#16a34a',
  yellow: '#ca8a04',
  red: '#dc2626',
  blue: '#2563eb',
  cyan: '#0284c7',
};

export function TerminalPane(props: TerminalPaneProps) {
  const {
    lang,
    theme = 'dark',
    sandboxStatus,
    sandboxDetail,
    shellChannel,
    serverChannel,
    onAttachShell,
    onRestartServer,
  } = props;

  const t = COPY[lang];
  const [activeTab, setActiveTab] = useState<'shell' | 'server'>('shell');
  const shellElRef = useRef<HTMLDivElement | null>(null);
  const serverElRef = useRef<HTMLDivElement | null>(null);
  const themeRef = useRef(theme);
  themeRef.current = theme;

  const shellTermRef = useRef<import('@xterm/xterm').Terminal | null>(null);
  const serverTermRef = useRef<import('@xterm/xterm').Terminal | null>(null);
  const shellFitRef = useRef<import('@xterm/addon-fit').FitAddon | null>(null);
  const serverFitRef = useRef<import('@xterm/addon-fit').FitAddon | null>(null);
  const shellHandleRef = useRef<InteractiveShellHandle | null>(null);

  useEffect(() => {
    let disposed = false;
    let resizeObs: ResizeObserver | null = null;
    let cleanupSubs = () => {};

    (async () => {
      const [{ Terminal }, { FitAddon }] = await Promise.all([
        import('@xterm/xterm'),
        import('@xterm/addon-fit'),
      ]);
      if (disposed || !shellElRef.current || !serverElRef.current) return;

      const initialTheme = themeRef.current === 'light' ? LIGHT_TERM_THEME : DARK_TERM_THEME;

      const shellTerm = new Terminal({
        convertEol: true,
        fontFamily: "'JetBrains Mono Variable', 'JetBrains Mono', monospace",
        fontSize: 12,
        lineHeight: 1.35,
        cursorBlink: true,
        theme: initialTheme,
      });
      const shellFit = new FitAddon();
      shellTerm.loadAddon(shellFit);
      shellTerm.open(shellElRef.current);
      try {
        shellFit.fit();
      } catch {
        // ignore if hidden
      }

      const serverTerm = new Terminal({
        convertEol: true,
        disableStdin: true,
        fontFamily: "'JetBrains Mono Variable', 'JetBrains Mono', monospace",
        fontSize: 12,
        lineHeight: 1.35,
        theme: initialTheme,
      });
      const serverFit = new FitAddon();
      serverTerm.loadAddon(serverFit);
      serverTerm.open(serverElRef.current);
      try {
        serverFit.fit();
      } catch {
        // ignore if hidden
      }

      shellTermRef.current = shellTerm;
      serverTermRef.current = serverTerm;
      shellFitRef.current = shellFit;
      serverFitRef.current = serverFit;

      let unsubShell = () => {};
      let unsubServer = () => {};
      if (!disposed) {
        unsubShell = shellChannel.subscribe((chunk) => shellTerm.write(chunk), true);
        unsubServer = serverChannel.subscribe((chunk) => serverTerm.write(chunk), true);
      }

      shellTerm.onData((data) => {
        shellHandleRef.current?.write(data);
      });

      const doFit = () => {
        try {
          shellFitRef.current?.fit();
          serverFitRef.current?.fit();
          if (shellTermRef.current && shellHandleRef.current) {
            shellHandleRef.current.resize(shellTermRef.current.cols, shellTermRef.current.rows);
          }
        } catch {
          // ignore unmounted or 0-size box
        }
      };

      resizeObs = new ResizeObserver(doFit);
      if (shellElRef.current) resizeObs.observe(shellElRef.current);
      if (serverElRef.current) resizeObs.observe(serverElRef.current);

      cleanupSubs = () => {
        unsubShell();
        unsubServer();
      };
    })();

    return () => {
      disposed = true;
      cleanupSubs();
      resizeObs?.disconnect();
      shellHandleRef.current?.kill();
      shellHandleRef.current = null;
      shellTermRef.current?.dispose();
      serverTermRef.current?.dispose();
      shellTermRef.current = null;
      serverTermRef.current = null;
    };
  }, [shellChannel, serverChannel]);

  // Attach interactive jsh shell once sandbox reaches 'ready'
  useEffect(() => {
    if (sandboxStatus !== 'ready' || shellHandleRef.current) return;
    let cancelled = false;
    (async () => {
      const cols = shellTermRef.current?.cols || 76;
      const rows = shellTermRef.current?.rows || 12;
      const handle = await onAttachShell(cols, rows, (chunk) => {
        shellTermRef.current?.write(chunk);
      });
      if (cancelled) {
        handle.kill();
        return;
      }
      shellHandleRef.current = handle;
    })().catch(() => {});

    return () => {
      cancelled = true;
    };
  }, [sandboxStatus, onAttachShell]);

  useEffect(() => {
    const nextTheme = theme === 'light' ? LIGHT_TERM_THEME : DARK_TERM_THEME;
    try {
      if (shellTermRef.current) {
        shellTermRef.current.options.theme = nextTheme;
        shellTermRef.current.refresh(0, Math.max(0, shellTermRef.current.rows - 1));
      }
      if (serverTermRef.current) {
        serverTermRef.current.options.theme = nextTheme;
        serverTermRef.current.refresh(0, Math.max(0, serverTermRef.current.rows - 1));
      }
    } catch {
      // ignore unmounted terminal repaint errors
    }
  }, [theme]);

  useEffect(() => {
    try {
      shellFitRef.current?.fit();
      serverFitRef.current?.fit();
    } catch {
      // ignore
    }
  }, [activeTab]);

  const badgeText =
    sandboxStatus === 'ready'
      ? sandboxDetail || t.sandboxReady
      : sandboxStatus === 'installing'
        ? 'npm install...'
        : sandboxStatus === 'starting'
          ? 'Vite + Node...'
          : sandboxStatus === 'error'
            ? 'Error'
            : 'Booting';

  return (
    <div className="terminal-dock" id="terminal-pane">
      <div className="terminal-bar">
        <div className="terminal-bar-left">
          <TerminalIcon size={12} style={{ color: 'var(--text-muted)' }} />
          <button
            type="button"
            className={`term-tab ${activeTab === 'shell' ? 'active' : ''}`}
            onClick={() => setActiveTab('shell')}
          >
            {t.terminalTabShell}
          </button>
          <button
            type="button"
            className={`term-tab ${activeTab === 'server' ? 'active' : ''}`}
            onClick={() => setActiveTab('server')}
          >
            {t.terminalTabServer}
          </button>
        </div>

        <div className="terminal-bar-right">
          <span
            id="sandbox-status-pill"
            className={`feature-badge ${
              sandboxStatus === 'ready' ? 'ok' : sandboxStatus === 'error' ? 'err' : 'info'
            }`}
            style={{ padding: '1px 7px', fontSize: 10.5 }}
          >
            {badgeText}
          </span>
          <button
            type="button"
            className="icon-btn"
            style={{ height: 22, padding: '0 5px' }}
            disabled={sandboxStatus === 'booting' || sandboxStatus === 'installing'}
            onClick={onRestartServer}
            title={t.restartServer}
          >
            <RefreshCw size={11} />
          </button>
        </div>
      </div>

      <div className="terminal-stage">
        <div
          ref={shellElRef}
          className="xterm-mount"
          style={{ display: activeTab === 'shell' ? 'block' : 'none' }}
        />
        <div
          ref={serverElRef}
          className="xterm-mount"
          style={{ display: activeTab === 'server' ? 'block' : 'none' }}
        />
      </div>
    </div>
  );
}
