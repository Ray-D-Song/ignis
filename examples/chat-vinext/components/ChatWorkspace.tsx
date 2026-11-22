'use client';

import React, { useCallback, useEffect, useRef, useState } from 'react';
import { Bot, MessageSquare, Plus } from 'lucide-react';
import type { ModelPreset } from '@/lib/model-paths';
import { IgnisClient } from '../../../runtime/ignis-client.mjs';
import { decodeWav, resampleLinear } from '../../../runtime/multimodal/audio-dsp.mjs';
import { detectDeviceProfile } from '../../../runtime/device-profile.mjs';
import {
  runAgentTurn,
  type AgentToolCallRecord,
  type AgentTraceEntry,
  type ApprovalDecision,
  type ApprovalRequest,
  type ApprovalPolicy,
  type PendingAudioInput,
  type PendingImageInput,
} from '@/lib/agent-harness';
import { COPY, type UiLang } from '@/lib/i18n';
import {
  buildSparkSystemPrefix,
  getArchKind,
  splitThinking,
  type OmniGenMode,
} from '@/lib/prompt-templates';
import { TerminalChannel } from '@/lib/terminal-channel';
import {
  bootAndStartFullStack,
  startOrRestartDevServer,
  wcAttachInteractiveShell,
  wcDeleteFile,
  wcListProjectFiles,
  wcReadFile,
  wcResetStarterProject,
  wcRunCommand,
  wcWriteFile,
  type SandboxBootStatus,
} from '@/lib/webcontainer-manager';
import {
  createInitialWorkspace,
  inferFileKind,
  normalizeWorkspacePath,
  readWorkspaceFile,
  upsertWorkspaceFile,
  type WorkspaceCheckpoint,
  type WorkspaceFile,
} from '@/lib/workspace-fs';
import { CodeEditorPane } from './CodeEditorPane';
import { Composer } from './Composer';
import { ThreadView, type LoadProgressState, type StudioMessage } from './ThreadView';
import { TopBar } from './TopBar';
import { WorkspacePane, type FsmPreset, type WorkspaceTab } from './WorkspacePane';

type IgnisDeviceProfile = Awaited<ReturnType<typeof detectDeviceProfile>>;

interface IgnisProgressEvent {
  phase?: LoadProgressState['phase'];
  file: string;
  loaded: number;
  total: number;
  overallLoaded?: number;
  overallTotal?: number;
  fileIndex?: number;
  fileCount?: number;
  cached: boolean;
}

interface IgnisTokenEvent {
  tokenId: number;
  text: string;
  fullText?: string;
  step: number;
  tokPerSec?: number;
}

const CUSTOM_MODEL_ID = '__custom__';

const FSM_PRESETS: FsmPreset[] = [
  {
    id: 'ignis_summary',
    label: 'Engine Summary Object',
    schema: {
      type: 'object',
      properties: {
        engine: { type: 'string' },
        storage: { type: 'string' },
        inBrowser: { type: 'boolean' },
      },
      required: ['engine', 'storage', 'inBrowser'],
      additionalProperties: false,
    },
  },
  {
    id: 'entity_status',
    label: 'Enum + Integer Metadata',
    schema: {
      type: 'object',
      properties: {
        topic: { type: 'string' },
        status: { type: 'string', enum: ['ready', 'beta', 'experimental'] },
        quantBits: { type: 'integer' },
      },
      required: ['topic', 'status', 'quantBits'],
      additionalProperties: false,
    },
  },
];

interface DirectOverrides {
  image?: PendingImageInput | null;
  audio?: PendingAudioInput | null;
  generationMode?: OmniGenMode;
  enableFsm?: boolean;
  schemaJson?: string;
  enableNgram?: boolean;
  maxNewTokens?: number;
  freshContext?: boolean;
}

function medianNumber(values?: number[]): number | null {
  if (!values || values.length === 0) return null;
  const sorted = [...values].sort((a, b) => a - b);
  return sorted[Math.floor(sorted.length / 2)];
}

async function bitmapBlobToRgba(
  blob: Blob,
  name: string,
  previewUrl: string,
): Promise<PendingImageInput> {
  const bmp = await createImageBitmap(blob);
  try {
    const maxEdge = 1024;
    const scale = Math.min(1, maxEdge / Math.max(bmp.width, bmp.height));
    const w = Math.max(1, Math.round(bmp.width * scale));
    const h = Math.max(1, Math.round(bmp.height * scale));
    const canvas = document.createElement('canvas');
    canvas.width = w;
    canvas.height = h;
    const ctx = canvas.getContext('2d');
    if (!ctx) throw new Error('2D canvas unavailable');
    ctx.drawImage(bmp, 0, 0, w, h);
    const rgba = new Uint8Array(ctx.getImageData(0, 0, w, h).data.buffer);
    return { data: rgba, width: w, height: h, channels: 4, previewUrl, name };
  } finally {
    bmp.close();
  }
}

function getReadyLabel(model: ModelPreset | undefined, lang: UiLang): string {
  const name = model?.shortName || model?.label.split('·')[0].trim() || 'Model';
  return lang === 'zh' ? `就绪 · ${name}` : `Ready · ${name}`;
}

export function ChatWorkspace({
  models,
  isPublicCdn = false,
}: {
  models: ModelPreset[];
  isPublicCdn?: boolean;
}) {
  const initialLang: UiLang = isPublicCdn ? 'en' : 'zh';
  const [lang, setLang] = useState<UiLang>(initialLang);
  const [theme, setTheme] = useState<'dark' | 'light'>('dark');
  const themeReadyRef = useRef(false);
  const t = COPY[lang];

  useEffect(() => {
    try {
      const saved = window.localStorage.getItem('ignis-theme');
      const next: 'dark' | 'light' = saved === 'light' || saved === 'dark' ? saved : 'dark';
      setTheme(next);
      document.documentElement.setAttribute('data-theme', next);
    } catch {
      document.documentElement.setAttribute('data-theme', 'dark');
    } finally {
      themeReadyRef.current = true;
    }
  }, []);

  useEffect(() => {
    document.documentElement.setAttribute('data-theme', theme);
    if (!themeReadyRef.current) return;
    try {
      window.localStorage.setItem('ignis-theme', theme);
    } catch {
      // ignore storage errors
    }
  }, [theme]);

  const defaultModelId = models[0]?.id || CUSTOM_MODEL_ID;
  const [selectedModelId, setSelectedModelId] = useState(defaultModelId);
  const [customUrl, setCustomUrl] = useState('');
  const [customPresetOverride, setCustomPresetOverride] = useState<Partial<ModelPreset> | null>(null);
  const [termsAcked, setTermsAcked] = useState(false);

  const [mode, setMode] = useState<'agent' | 'chat'>('agent');
  const [approvalPolicy, setApprovalPolicy] = useState<ApprovalPolicy>('auto');
  const [editorOpen, setEditorOpen] = useState(true);
  const [rightPaneOpen, setRightPaneOpen] = useState(true);
  const [workspaceTab, setWorkspaceTab] = useState<WorkspaceTab>('preview');

  const [status, setStatus] = useState<'idle' | 'loading' | 'ready' | 'generating' | 'error'>('idle');
  const [statusText, setStatusText] = useState<string>(COPY[initialLang].statusIdleShort);
  const [loadPct, setLoadPct] = useState(0);
  const [loadProgress, setLoadProgress] = useState<LoadProgressState | null>(null);

  const [enableThinking, setEnableThinking] = useState(() => getArchKind(models[0]) === 'spark');
  const [effort, setEffort] = useState<'low' | 'high'>('low');
  const [generationMode, setGenerationMode] = useState<OmniGenMode>('text');
  const [temperature, setTemperature] = useState(0.6);
  const [maxNewTokens, setMaxNewTokens] = useState(1536);

  const [enableFsm, setEnableFsm] = useState(false);
  const [fsmPresetId, setFsmPresetId] = useState(FSM_PRESETS[0].id);
  const [fsmSchemaText, setFsmSchemaText] = useState(JSON.stringify(FSM_PRESETS[0].schema, null, 2));
  const [enableNgram, setEnableNgram] = useState(false);

  // Coding project state
  const [workspaceFiles, setWorkspaceFiles] = useState<WorkspaceFile[]>(() => createInitialWorkspace());
  const [selectedFilePath, setSelectedFilePath] = useState<string>('src/App.jsx');
  const [editorMode, setEditorMode] = useState<'code' | 'diff'>('code');
  const [diffOriginals, setDiffOriginals] = useState<Record<string, string>>({});
  const [checkpoints, setCheckpoints] = useState<WorkspaceCheckpoint[]>([]);
  const [traceEntries, setTraceEntries] = useState<AgentTraceEntry[]>([]);
  const [contextBudget, setContextBudget] = useState({ peak: 0, max: 4096 });
  const [previewRefreshSeq, setPreviewRefreshSeq] = useState(0);

  // WebContainer runtime state
  const [sandboxStatus, setSandboxStatus] = useState<SandboxBootStatus>('idle');
  const [sandboxDetail, setSandboxDetail] = useState('');
  const [previewUrl, setPreviewUrl] = useState<string | null>(null);
  const [shellChannel] = useState(() => new TerminalChannel());
  const [serverChannel] = useState(() => new TerminalChannel());

  // Thread + composer + HITL state
  const [messages, setMessages] = useState<StudioMessage[]>([]);
  const [input, setInput] = useState('');
  const [pendingImage, setPendingImage] = useState<PendingImageInput | null>(null);
  const [pendingAudio, setPendingAudio] = useState<PendingAudioInput | null>(null);
  const [pendingApproval, setPendingApproval] = useState<ApprovalRequest | null>(null);

  const [gpuCaps, setGpuCaps] = useState('Detecting WebGPU...');
  const [deviceProfile, setDeviceProfile] = useState<IgnisDeviceProfile | null>(null);
  const [stats, setStats] = useState({
    initMs: '-',
    cacheMode: '-',
    prefillSpeed: '-',
    decodeSpeed: '-',
    ctxPos: 0,
    maxCtx: 8192,
    heapMB: '-',
  });

  const clientRef = useRef<IgnisClient | null>(null);
  const loadedModelIdRef = useRef<string | null>(null);
  const statusRef = useRef(status);
  statusRef.current = status;
  const activePresetRef = useRef<ModelPreset | null>(null);
  const kvTurnsRef = useRef<number>(0);
  const kvThinkingKeyRef = useRef<string>('');
  const objectUrlsRef = useRef<string[]>([]);
  const scrollEndRef = useRef<HTMLDivElement | null>(null);
  const workspaceRef = useRef(workspaceFiles);
  workspaceRef.current = workspaceFiles;
  const approvalResolverRef = useRef<((d: ApprovalDecision) => void) | null>(null);

  const customModelPreset: ModelPreset = {
    id: CUSTOM_MODEL_ID,
    shortName: 'Custom Model',
    label: t.customOption,
    description: customUrl ? `Remote .ignis base: ${customUrl}` : t.customUrlLabel,
    dirPath: customUrl,
    modelBaseUrl: customUrl,
    lmHeadDtype: customPresetOverride?.lmHeadDtype || 'u32_q8',
    architecture: customPresetOverride?.architecture || 'spark2_5',
    topModelType: customPresetOverride?.topModelType || 'spark2_5',
    gpuMiB: 0,
    embdMiB: 0,
    pleMiB: 0,
    isCustom: true,
    capabilities: customPresetOverride?.capabilities || {
      vision: false,
      audioIn: false,
      audioOut: false,
      thinking: false,
    },
  };

  const activeModel =
    selectedModelId === CUSTOM_MODEL_ID
      ? customModelPreset
      : models.find((m) => m.id === selectedModelId) || models[0] || customModelPreset;
  activePresetRef.current = activeModel;
  const archKind = getArchKind(activeModel);
  const termsBlocked = Boolean(activeModel?.requiresTermsAck && !termsAcked);

  const pushShellOutput = useCallback(
    (chunk: string) => {
      shellChannel.push(chunk);
    },
    [shellChannel],
  );

  const pushServerOutput = useCallback(
    (chunk: string) => {
      serverChannel.push(chunk);
    },
    [serverChannel],
  );

  const revokeAllObjectUrls = () => {
    for (const u of objectUrlsRef.current) URL.revokeObjectURL(u);
    objectUrlsRef.current = [];
  };

  useEffect(() => {
    if (typeof window !== 'undefined' && window.innerWidth < 1200) {
      setEditorOpen(false);
      setRightPaneOpen(false);
    }

    (async () => {
      try {
        const profile = await detectDeviceProfile();
        setDeviceProfile(profile);
        if (!profile.supported) {
          setGpuCaps('WebGPU Unsupported');
        } else {
          const iso = window.crossOriginIsolated ? 'COOP/COEP' : 'No-COI';
          setGpuCaps(`f16:${profile.hasF16 ? '✓' : '✗'} · sg:${profile.hasSubgroups ? '✓' : '✗'} · ${iso}`);
        }
      } catch {
        setGpuCaps('WebGPU Probe Failed');
      }
    })();

    // Boot StackBlitz WebContainer full-stack sandbox in parallel
    bootAndStartFullStack({
      onStatus: (st, detail) => {
        setSandboxStatus(st);
        if (detail) setSandboxDetail(detail);
      },
      onTerminalData: (chunk) => pushShellOutput(chunk),
      onServerLog: (chunk) => pushServerOutput(chunk),
      onServerReady: (url) => {
        setPreviewUrl(url);
      },
      onError: (msg) => {
        setSandboxStatus('error');
        setSandboxDetail(msg);
      },
    }).catch(() => {
      // status + detail already propagated via callbacks
    });

    return () => {
      revokeAllObjectUrls();
      if (approvalResolverRef.current) {
        approvalResolverRef.current({ approved: false, reason: 'Unmounted' });
        approvalResolverRef.current = null;
      }
      if (clientRef.current) {
        clientRef.current.destroy();
        clientRef.current = null;
      }
    };
  }, [pushServerOutput, pushShellOutput]);

  useEffect(() => {
    if (messages.length === 0 && !pendingApproval) return;
    scrollEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages, pendingApproval]);

  const handleModelChange = (nextId: string) => {
    setSelectedModelId(nextId);
    setTermsAcked(false);
    setCustomPresetOverride(null);
    const nextModel = models.find((m) => m.id === nextId);
    setEnableThinking(getArchKind(nextModel) === 'spark');
    setGenerationMode('text');
    setPendingImage(null);
    setPendingAudio(null);
    kvTurnsRef.current = 0;
    kvThinkingKeyRef.current = '';
    if (loadedModelIdRef.current && loadedModelIdRef.current !== nextId) {
      setStatus('idle');
      statusRef.current = 'idle';
      setLoadPct(0);
      setLoadProgress(null);
      setStatusText(t.statusSwitched);
    } else if (status === 'error') {
      setStatus('idle');
      statusRef.current = 'idle';
      setLoadPct(0);
      setLoadProgress(null);
      setStatusText(t.statusIdleShort);
    }
  };

  const handleImageFile = async (file: File) => {
    const previewUrl = URL.createObjectURL(file);
    objectUrlsRef.current.push(previewUrl);
    const img = await bitmapBlobToRgba(file, file.name, previewUrl);
    setPendingImage(img);
  };

  const handleAudioFile = async (file: File) => {
    const ab = await file.arrayBuffer();
    const { samples, sampleRate } = decodeWav(ab);
    const pcm16k = sampleRate === 16000 ? samples : resampleLinear(samples, sampleRate, 16000);
    setPendingAudio({ pcm16k, durationSec: pcm16k.length / 16000, name: file.name });
  };

  const clearOpfsCache = async () => {
    try {
      if (clientRef.current) {
        await clientRef.current.destroy();
        clientRef.current = null;
        await new Promise((r) => setTimeout(r, 40));
      }
      loadedModelIdRef.current = null;
      kvTurnsRef.current = 0;

      if (navigator.storage?.getDirectory) {
        const root = await navigator.storage.getDirectory();
        const names: string[] = [];
        for await (const [name] of (root as unknown as { entries(): AsyncIterable<[string, unknown]> }).entries()) {
          names.push(name);
        }
        for (const name of names) {
          for (let attempt = 0; attempt < 3; attempt++) {
            try {
              await root.removeEntry(name, { recursive: true });
              break;
            } catch (e) {
              if (attempt === 2) throw e;
              await new Promise((r) => setTimeout(r, 60));
            }
          }
        }
      }

      setStats((s) => ({ ...s, cacheMode: t.opfsCleared, ctxPos: 0 }));
      setStatus('idle');
      setLoadPct(0);
      setLoadProgress(null);
      setStatusText(t.opfsCleared);
    } catch (err) {
      setStatus('error');
      setStatusText(`OPFS clear failed: ${err instanceof Error ? err.message : String(err)}`);
    }
  };

  const loadModel = async (
    queuedPrompt: string | null = null,
  ): Promise<{ client: IgnisClient; resolvedModel: ModelPreset } | null> => {
    const current = activePresetRef.current || activeModel;
    if (!current) return null;
    const baseUrl = (
      current.isCustom ? customUrl.trim() : current.modelBaseUrl || `/api/models/${current.id}`
    ).replace(/\/+$/, '');
    if (!baseUrl) {
      setStatus('error');
      setStatusText('Enter a valid custom .ignis base URL first');
      return null;
    }

    const displayName = current.shortName || current.label.split('·')[0].trim();
    setStatus('loading');
    statusRef.current = 'loading';
    setLoadPct(2);
    setLoadProgress({
      phase: 'manifest',
      file: 'manifest.json',
      loaded: 0,
      total: 0,
      overallLoaded: 0,
      overallTotal: 0,
      fileIndex: 0,
      fileCount: 0,
      cached: false,
      overallPct: 2,
      speedMBps: null,
      queuedPrompt,
    });
    setStatusText(lang === 'zh' ? `加载 ${displayName}...` : `Loading ${displayName}...`);
    try {
      if (clientRef.current) {
        await clientRef.current.destroy();
        clientRef.current = null;
      }
      const client = new IgnisClient('/ignis/runtime/ignis-worker.mjs');
      clientRef.current = client;

      let highestPct = 2;
      let lastSpeedAt = performance.now();
      let lastSpeedBytes = 0;
      let emaSpeedMBps: number | null = null;
      let lastDownloadLoaded = 0;
      let lastDownloadTotal = 0;

      client.onProgress = (p: IgnisProgressEvent) => {
        const phase = p.phase || (p.cached ? 'cache' : 'download');
        let mappedPct = highestPct;
        if (phase === 'manifest') {
          mappedPct = 2;
        } else if (phase === 'download' || phase === 'cache') {
          const oLoaded = p.overallLoaded ?? p.loaded;
          const oTotal = p.overallTotal ?? p.total;
          lastDownloadLoaded = oLoaded;
          lastDownloadTotal = oTotal;
          mappedPct = oTotal > 0 ? 4 + Math.round((oLoaded / oTotal) * 78) : highestPct;
          if (phase === 'download') {
            const now = performance.now();
            const dt = now - lastSpeedAt;
            if (dt >= 180 && oLoaded >= lastSpeedBytes) {
              const inst = ((oLoaded - lastSpeedBytes) / (1024 * 1024)) / (dt / 1000);
              emaSpeedMBps = emaSpeedMBps === null ? inst : emaSpeedMBps * 0.65 + inst * 0.35;
              lastSpeedAt = now;
              lastSpeedBytes = oLoaded;
            }
          }
        } else if (phase === 'tokenizer') {
          mappedPct = 84;
        } else if (phase === 'compile') {
          mappedPct = 88;
        } else if (phase === 'gpu_upload') {
          const gTotal = p.overallTotal || p.total || 1;
          const gLoaded = p.overallLoaded ?? p.loaded ?? 0;
          mappedPct = 89 + Math.round((gLoaded / gTotal) * 9);
        } else if (phase === 'tokenizer_init') {
          mappedPct = 99;
        }

        highestPct = Math.max(highestPct, Math.min(99, mappedPct));
        const shownLoaded =
          phase === 'download' || phase === 'cache'
            ? p.overallLoaded ?? p.loaded
            : lastDownloadLoaded || p.overallLoaded || 0;
        const shownTotal =
          phase === 'download' || phase === 'cache'
            ? p.overallTotal ?? p.total
            : lastDownloadTotal || p.overallTotal || 0;

        setLoadPct(highestPct);
        setLoadProgress({
          phase,
          file: p.file,
          loaded: p.loaded,
          total: p.total,
          overallLoaded: shownLoaded,
          overallTotal: shownTotal,
          fileIndex: p.fileIndex ?? 0,
          fileCount: p.fileCount ?? 0,
          cached: p.cached,
          overallPct: highestPct,
          speedMBps: phase === 'download' ? emaSpeedMBps : null,
          queuedPrompt,
        });

        if (phase === 'download') {
          const speedStr = emaSpeedMBps !== null ? ` · ${emaSpeedMBps.toFixed(1)} MB/s` : '';
          setStatusText(`${lang === 'zh' ? '加载中' : 'Loading'} ${highestPct}%${speedStr}`);
        } else if (phase === 'compile') {
          setStatusText(
            lang === 'zh' ? `加载中 ${highestPct}% · 编译 WebGPU` : `Loading ${highestPct}% · WebGPU Shaders`,
          );
        } else if (phase === 'gpu_upload') {
          setStatusText(
            lang === 'zh' ? `加载中 ${highestPct}% · 装载显存` : `Loading ${highestPct}% · GPU VRAM`,
          );
        } else {
          setStatusText(`${lang === 'zh' ? '加载中' : 'Loading'} ${highestPct}%`);
        }
      };
      client.onDeviceLost = (reason: string) => {
        setStatus('error');
        statusRef.current = 'error';
        setStatusText(`GPUDevice Lost: ${reason}`);
      };

      const safeKey = current.isCustom
        ? `custom-${baseUrl.replace(/[^a-zA-Z0-9_-]/g, '_').slice(-48)}`
        : `${isPublicCdn ? 'cdn' : 'vinext'}-${current.id}`;

      const preKind = getArchKind(current);
      const maxSeq = preKind === 'spark' ? 8192 : 4096;
      const info = await client.init({
        modelBaseUrl: baseUrl,
        modelKey: safeKey,
        kernelBaseUrl: '/ignis/kernels',
        wasmUrl: '/ignis/runtime/wasm/ignis_tokenizer.wasm',
        maxSeqLen: maxSeq,
        maxBatchTokens: 64,
      });

      let resolvedPreset = current;
      if (current.isCustom && info.blueprint) {
        const bp = info.blueprint;
        const mmKind = bp.multimodal?.kind;
        const inferred: Partial<ModelPreset> = {
          architecture: bp.architecture || 'spark2_5',
          topModelType: bp.top_model_type || bp.architecture || 'spark2_5',
          capabilities: {
            vision: mmKind === 'lfm3_omni' || mmKind === 'gemma4',
            audioIn: mmKind === 'lfm3_omni' || mmKind === 'gemma4',
            audioOut: mmKind === 'lfm3_omni',
            thinking: bp.architecture === 'spark2_5' || bp.top_model_type === 'gemma4',
          },
        };
        setCustomPresetOverride(inferred);
        resolvedPreset = { ...current, ...inferred };
      }

      loadedModelIdRef.current = current.id;
      kvTurnsRef.current = 0;
      kvThinkingKeyRef.current = '';
      setContextBudget({ peak: 0, max: maxSeq });
      const perfMem = (performance as Performance & { memory?: { usedJSHeapSize: number } }).memory;
      setStats((s) => ({
        ...s,
        initMs: `${(info.initMs / 1000).toFixed(2)}s`,
        cacheMode:
          info.cacheMisses === 0 ? `OPFS Hit (${info.cacheHits})` : `Cached (${info.cacheMisses})`,
        ctxPos: 0,
        maxCtx: maxSeq,
        heapMB: perfMem ? `${(perfMem.usedJSHeapSize / (1024 * 1024)).toFixed(1)} MB` : '-',
      }));
      setLoadPct(100);
      setLoadProgress(null);
      setStatus('ready');
      statusRef.current = 'ready';
      setStatusText(getReadyLabel(resolvedPreset, lang));
      return { client, resolvedModel: resolvedPreset };
    } catch (err) {
      setLoadProgress(null);
      setStatus('error');
      statusRef.current = 'error';
      setStatusText(`Load failed: ${err instanceof Error ? err.message : String(err)}`);
      return null;
    }
  };

  const resetChat = async () => {
    if (clientRef.current && status === 'ready') {
      await clientRef.current.reset();
    }
    if (approvalResolverRef.current) {
      approvalResolverRef.current({ approved: false, reason: 'Session reset' });
      approvalResolverRef.current = null;
    }
    setPendingApproval(null);
    revokeAllObjectUrls();
    kvTurnsRef.current = 0;
    kvThinkingKeyRef.current = '';
    setPendingImage(null);
    setPendingAudio(null);
    setMessages([]);
    setTraceEntries([]);
    setStats((s) => ({ ...s, ctxPos: 0 }));
  };

  const stopGeneration = async () => {
    if (approvalResolverRef.current) {
      approvalResolverRef.current({ approved: false, reason: 'Stopped by user' });
      approvalResolverRef.current = null;
      setPendingApproval(null);
    }
    if (clientRef.current && status === 'generating') {
      await clientRef.current.abort();
    }
  };

  const handleResolveApproval = (decision: ApprovalDecision) => {
    setPendingApproval(null);
    const res = approvalResolverRef.current;
    approvalResolverRef.current = null;
    res?.(decision);
  };

  const handleSaveFile = async (rawPath: string, content: string) => {
    const clean = normalizeWorkspacePath(rawPath);
    try {
      await wcWriteFile(clean, content);
    } catch {
      // still update virtual buffer if container not yet ready
    }
    setWorkspaceFiles((prev) => upsertWorkspaceFile(prev, clean, content, false));
    setPreviewRefreshSeq((s) => s + 1);
  };

  const handleRevertFile = async (rawPath: string) => {
    const clean = normalizeWorkspacePath(rawPath);
    const orig = diffOriginals[clean];
    if (orig === undefined) return;
    try {
      await wcWriteFile(clean, orig);
    } catch {
      // ignore if container restarting
    }
    setWorkspaceFiles((prev) =>
      upsertWorkspaceFile(prev, clean, orig, false, { addedLines: 0, removedLines: 0 }),
    );
    setDiffOriginals((prev) => {
      const next = { ...prev };
      delete next[clean];
      return next;
    });
    setEditorMode('code');
    setPreviewRefreshSeq((s) => s + 1);
  };

  const handleRollbackCheckpoint = async (checkpointId: string) => {
    const cp = checkpoints.find((x) => x.id === checkpointId);
    if (!cp) return;
    if (cp.existedBefore) {
      try {
        await wcWriteFile(cp.path, cp.beforeContent);
      } catch {
        // ignore
      }
      setWorkspaceFiles((prev) =>
        upsertWorkspaceFile(prev, cp.path, cp.beforeContent, false, { addedLines: 0, removedLines: 0 }),
      );
      setSelectedFilePath(cp.path);
    } else {
      try {
        await wcDeleteFile(cp.path);
      } catch {
        // ignore
      }
      setWorkspaceFiles((prev) => prev.filter((f) => normalizeWorkspacePath(f.path) !== cp.path));
    }
    setDiffOriginals((prev) => {
      const next = { ...prev };
      delete next[cp.path];
      return next;
    });
    setCheckpoints((prev) => prev.filter((x) => x.id !== checkpointId));
    setEditorMode('code');
    setPreviewRefreshSeq((s) => s + 1);
    pushShellOutput(`\r\n[checkpoint] ${t.checkpointRestored}: ${cp.path}\r\n`);
  };

  const handleCreateFile = async (rawPath: string) => {
    const clean = normalizeWorkspacePath(rawPath);
    if (!clean) return;
    try {
      await wcWriteFile(clean, '');
    } catch {
      // ignore
    }
    setWorkspaceFiles((prev) => upsertWorkspaceFile(prev, clean, '', false));
    setSelectedFilePath(clean);
    setEditorMode('code');
    setEditorOpen(true);
  };

  const handleUploadFiles = async (fileList: FileList) => {
    let lastPath = '';
    for (const file of Array.from(fileList)) {
      const rel = normalizeWorkspacePath(file.name);
      const text = await file.text();
      try {
        await wcWriteFile(rel, text);
      } catch {
        // ignore
      }
      setWorkspaceFiles((prev) => upsertWorkspaceFile(prev, rel, text, false));
      lastPath = rel;
    }
    if (lastPath) {
      setSelectedFilePath(lastPath);
      setEditorMode('code');
      setEditorOpen(true);
    }
  };

  const handleResetProject = async () => {
    try {
      const resetFiles = await wcResetStarterProject();
      setWorkspaceFiles(resetFiles);
    } catch {
      setWorkspaceFiles(createInitialWorkspace());
    }
    setDiffOriginals({});
    setCheckpoints([]);
    setSelectedFilePath('src/App.jsx');
    setEditorMode('code');
    setPreviewRefreshSeq((s) => s + 1);
    pushShellOutput('\r\n[workspace] Starter Vite + React files restored.\r\n');
  };

  const sendDirectMessage = async (
    client: IgnisClient,
    modelForTurn: ModelPreset,
    rawText: string,
    overrideOpts?: DirectOverrides,
  ) => {
    const imgAttach = overrideOpts?.image !== undefined ? overrideOpts.image : pendingImage;
    const audAttach = overrideOpts?.audio !== undefined ? overrideOpts.audio : pendingAudio;
    const text = rawText.trim() || (imgAttach || audAttach ? t.analyzeMm : '');
    if (!text) return;

    const useFsm = overrideOpts?.enableFsm ?? enableFsm;
    const rawSchemaText = overrideOpts?.schemaJson ?? fsmSchemaText;
    let parsedFsmSchema: Record<string, unknown> | null = null;
    if (useFsm) {
      try {
        parsedFsmSchema = JSON.parse(rawSchemaText) as Record<string, unknown>;
      } catch (e) {
        setStatusText(`Invalid JSON Schema: ${e instanceof Error ? e.message : String(e)}`);
        return;
      }
    }

    const curArch = getArchKind(modelForTurn);
    const activeMode: OmniGenMode = modelForTurn.capabilities.audioOut
      ? overrideOpts?.generationMode ?? generationMode
      : 'text';
    const hasMm = Boolean(imgAttach || audAttach);
    const useNgram =
      !useFsm && !hasMm && activeMode === 'text' && (overrideOpts?.enableNgram ?? enableNgram);
    const effectiveThinking = enableThinking && !useFsm && Boolean(modelForTurn.capabilities.thinking);

    if (!overrideOpts) {
      setInput('');
      setPendingImage(null);
      setPendingAudio(null);
    }

    const userMsg: StudioMessage = {
      id: `u-${Date.now()}`,
      role: 'user',
      content: text,
      image: imgAttach,
      audio: audAttach,
    };
    const assistantId = `a-${Date.now()}`;
    const shortModelName = modelForTurn.shortName || modelForTurn.label.split('·')[0].trim();
    const thinkPrefix = effectiveThinking ? (curArch === 'gemma4' ? '<|thought|>' : '<think>') : '';
    const assistantMsg: StudioMessage = {
      id: assistantId,
      role: 'assistant',
      content: thinkPrefix,
      modelLabel: shortModelName,
    };

    const useFresh = Boolean(overrideOpts?.freshContext);
    const nextMessages = [...messages, userMsg];
    const promptMessages = useFresh ? [userMsg] : nextMessages;
    setMessages([...nextMessages, assistantMsg]);
    setStatus('generating');
    setStatusText(lang === 'zh' ? '处理上下文...' : 'Prefilling context...');

    const thinkingKey = `${curArch}:${effectiveThinking}:${effort}`;
    let promptDeltaText = '';
    const reqImages: Array<{ data: Uint8Array; width: number; height: number; channels: number }> = [];
    const reqAudios: Float32Array[] = [];
    const includeMm = !useFsm && !useNgram;

    if (curArch === 'spark') {
      const canIncremental =
        !useFresh &&
        kvTurnsRef.current === messages.length &&
        kvThinkingKeyRef.current === thinkingKey &&
        kvTurnsRef.current > 0 &&
        !useFsm &&
        !useNgram;
      if (canIncremental) {
        promptDeltaText = `<｜end▁of▁sentence｜><｜start▁of▁sentence｜><|User|>${text}<｜end▁of▁sentence｜><｜start▁of▁sentence｜><|Bot|>${
          effectiveThinking ? '<think>' : '</think>'
        }`;
      } else {
        await client.reset();
        let fullPrompt = buildSparkSystemPrefix(effectiveThinking, effort);
        for (const m of promptMessages) {
          if (m.role === 'user') {
            fullPrompt += `<｜start▁of▁sentence｜><|User|>${m.content}<｜end▁of▁sentence｜>`;
          } else {
            const parsed = splitThinking(m.content);
            fullPrompt += `<｜start▁of▁sentence｜><|Bot|></think>${parsed.answer}<｜end▁of▁sentence｜>`;
          }
        }
        fullPrompt += `<｜start▁of▁sentence｜><|Bot|>${effectiveThinking ? '<think>' : '</think>'}`;
        promptDeltaText = fullPrompt;
      }
    } else if (curArch === 'lfm3') {
      await client.reset();
      let fullPrompt = '<|startoftext|>';
      for (const m of promptMessages) {
        if (m.role === 'user') {
          let mmTags = '';
          if (includeMm && m.image) {
            reqImages.push({
              data: m.image.data,
              width: m.image.width,
              height: m.image.height,
              channels: m.image.channels,
            });
            mmTags += '<|vision_start|><|image|><|vision_end|>';
          }
          if (includeMm && m.audio) {
            reqAudios.push(m.audio.pcm16k);
            mmTags += '<|vision_start|><|audio|><|vision_end|>';
          }
          fullPrompt += `<|im_start|>user\n${mmTags}${m.content}<|im_end|>\n`;
        } else {
          const parsed = splitThinking(m.content);
          fullPrompt += `<|im_start|>assistant\n<think></think>${parsed.answer}<|im_end|>\n`;
        }
      }
      fullPrompt += '<|im_start|>assistant\n<think></think>';
      promptDeltaText = fullPrompt;
    } else {
      await client.reset();
      let fullPrompt = '<bos>';
      for (const m of promptMessages) {
        if (m.role === 'user') {
          let mmTags = '';
          if (includeMm && m.image) {
            reqImages.push({
              data: m.image.data,
              width: m.image.width,
              height: m.image.height,
              channels: m.image.channels,
            });
            mmTags += '<|image|>';
          }
          if (includeMm && m.audio) {
            reqAudios.push(m.audio.pcm16k);
            mmTags += '<|audio|>';
          }
          fullPrompt += `<|turn>user\n${mmTags}${m.content}<turn|>\n`;
        } else {
          const parsed = splitThinking(m.content);
          fullPrompt += `<|turn>model\n${parsed.answer}<turn|>\n`;
        }
      }
      fullPrompt += effectiveThinking ? '<|turn>model\n<|thought|>\n' : '<|turn>model\n';
      promptDeltaText = fullPrompt;
    }

    client.onPrefillProgress = (p: { processed: number; total: number }) => {
      setStatusText(`Prefill ${p.processed}/${p.total} tokens...`);
    };

    let liveTps = 0;
    client.onToken = (ev: IgnisTokenEvent) => {
      if (ev.tokPerSec) liveTps = ev.tokPerSec;
      const body = ev.fullText !== undefined ? ev.fullText : ev.text;
      const rendered = thinkPrefix ? `${thinkPrefix}${body}` : body;
      setMessages((prev) => prev.map((m) => (m.id === assistantId ? { ...m, content: rendered } : m)));
      setStatusText(`${lang === 'zh' ? '生成中' : 'Generating'} · ${liveTps.toFixed(1)} tok/s`);
    };

    try {
      const enc = await client.encode(promptDeltaText);
      const maxThinkingTokens =
        curArch === 'spark' && effectiveThinking
          ? effort === 'low'
            ? Math.min(384, Math.floor(maxNewTokens * 0.45))
            : Math.floor(maxNewTokens * 0.65)
          : 0;

      const genBudget = overrideOpts?.maxNewTokens ?? maxNewTokens;
      const res = await client.generate({
        promptIds: enc.ids,
        images: reqImages,
        audios: reqAudios,
        generationMode: activeMode,
        maxNewTokens: genBudget,
        maxThinkingTokens,
        chunkSize: 64,
        sampling:
          useFsm || useNgram
            ? { temperature: 0, topK: 1, topP: 1.0, repetitionPenalty: 1.0 }
            : {
                temperature,
                topK: 40,
                topP: 0.95,
                repetitionPenalty: curArch === 'spark' ? 1.05 : 1.0,
              },
        ...(parsedFsmSchema
          ? {
              responseFormat: {
                type: 'json_schema',
                json_schema: { name: 'ignis_schema', schema: parsedFsmSchema, strict: true },
              },
            }
          : {}),
        ...(useNgram ? { speculative: { mode: 'ngram', k: 5, ngramN: 4 } } : {}),
        stopOnEos: true,
      });

      let audioUrl: string | undefined;
      if (res.wavBytes && res.wavBytes.byteLength > 44) {
        audioUrl = URL.createObjectURL(new Blob([new Uint8Array(res.wavBytes)], { type: 'audio/wav' }));
        objectUrlsRef.current.push(audioUrl);
      }

      const finalRaw = thinkPrefix ? `${thinkPrefix}${res.generatedText || ''}` : res.generatedText || '';
      const parsedOut = splitThinking(finalRaw);
      let fsmValid: boolean | undefined;
      if (parsedFsmSchema) {
        try {
          JSON.parse(parsedOut.answer);
          fsmValid = true;
        } catch {
          fsmValid = false;
        }
      }
      const fsmMedianMs = medianNumber(res.fsmMaskMsSamples);
      const specAccepted = typeof res.speculativeAccepted === 'number' ? res.speculativeAccepted : undefined;
      const specProposed = typeof res.speculativeProposed === 'number' ? res.speculativeProposed : undefined;

      const meta = `${res.generatedIds.length} tok · ${res.decodeTokPerSec.toFixed(1)} tok/s · Prefill ${res.prefillMs.toFixed(0)}ms${audioUrl ? ' · 24kHz WAV' : ''}${res.aborted ? ' · Stopped' : ''}`;
      setMessages((prev) =>
        prev.map((m) =>
          m.id === assistantId
            ? {
                ...m,
                content: finalRaw,
                audioUrl,
                meta,
                fsmValid,
                fsmMedianMs,
                specAccepted,
                specProposed,
              }
            : m,
        ),
      );

      if (curArch === 'spark' && !res.aborted && !useFresh && !useFsm && !useNgram) {
        kvTurnsRef.current = nextMessages.length + 1;
        kvThinkingKeyRef.current = thinkingKey;
      } else {
        kvTurnsRef.current = -1;
      }

      const perfMem = (performance as Performance & { memory?: { usedJSHeapSize: number } }).memory;
      setStats((s) => ({
        ...s,
        prefillSpeed: `${(res.promptLen / (Math.max(1, res.prefillMs) / 1000)).toFixed(1)} tok/s`,
        decodeSpeed: `${res.decodeTokPerSec.toFixed(1)} tok/s`,
        ctxPos: res.curPos,
        heapMB: perfMem ? `${(perfMem.usedJSHeapSize / (1024 * 1024)).toFixed(1)} MB` : s.heapMB,
      }));
      setStatus('ready');
      setStatusText(getReadyLabel(modelForTurn, lang));
    } catch (err) {
      setStatus('error');
      setStatusText(`Generation error: ${err instanceof Error ? err.message : String(err)}`);
    } finally {
      client.onToken = null;
      client.onPrefillProgress = null;
    }
  };

  const runAgentTask = async (
    client: IgnisClient,
    modelForTurn: ModelPreset,
    rawText: string,
    clearComposer = true,
  ) => {
    const imgAttach = clearComposer ? pendingImage : null;
    const audAttach = clearComposer ? pendingAudio : null;
    const promptText = rawText.trim() || (imgAttach || audAttach ? t.analyzeMm : '');
    if (!promptText) return;

    if (clearComposer) {
      setInput('');
      setPendingImage(null);
      setPendingAudio(null);
    }

    const curArch = getArchKind(modelForTurn);
    const maxSeqLen = curArch === 'spark' ? 8192 : 4096;

    const userMsg: StudioMessage = {
      id: `u-${Date.now()}`,
      role: 'user',
      content: promptText,
      image: imgAttach,
      audio: audAttach,
    };
    const assistantId = `a-${Date.now()}`;
    const shortModelName = modelForTurn.shortName || modelForTurn.label.split('·')[0].trim();
    const assistantMsg: StudioMessage = {
      id: assistantId,
      role: 'assistant',
      content: '',
      modelLabel: shortModelName,
      toolCalls: [],
      checkpoints: [],
    };

    const historyText = messages
      .slice(-4)
      .map((m) => `${m.role === 'user' ? 'User' : 'Assistant'}: ${splitThinking(m.content).answer.slice(0, 260)}`)
      .join('\n');

    setMessages((prev) => [...prev, userMsg, assistantMsg]);
    setTraceEntries([]);
    setContextBudget({ peak: 0, max: maxSeqLen });
    setStatus('generating');
    setStatusText(t.loadingStep(1, 12, 'FSM Tools'));
    kvTurnsRef.current = -1;

    try {
      const result = await runAgentTurn({
        client,
        archKind: curArch,
        maxSeqLen,
        capabilities: modelForTurn.capabilities,
        userPrompt: promptText,
        historyText,
        workspaceFiles: workspaceRef.current,
        approvalPolicy,
        attachedImage: imgAttach,
        attachedAudio: audAttach,
        generationMode: modelForTurn.capabilities.audioOut ? generationMode : 'text',
        maxNewTokens,
        maxSteps: 12,
        lang,
        enableThinking,
        effort,
        sandbox: {
          writeFile: wcWriteFile,
          readFile: wcReadFile,
          listFiles: async () => {
            const existingMap = new Map(workspaceRef.current.map((f) => [f.path, f]));
            const listed = await wcListProjectFiles();
            return listed.map((f) => {
              const prev = existingMap.get(f.path);
              return prev
                ? {
                    ...f,
                    kind: inferFileKind(f.path),
                    modifiedByAgent: prev.modifiedByAgent,
                    addedLines: prev.addedLines,
                    removedLines: prev.removedLines,
                  }
                : f;
            });
          },
          runCommand: async (command: string) => {
            pushShellOutput(`\r\n$ ${command}\r\n`);
            const res = await wcRunCommand(command, {
              onOutput: (chunk) => pushShellOutput(chunk),
            });
            return res;
          },
        },
        callbacks: {
          onStatusText: (txt) => setStatusText(txt),
          onToolStepStart: (step, max) => {
            setStatusText(t.loadingStep(step, max, 'FSM Tool'));
          },
          onToolStepComplete: (record: AgentToolCallRecord) => {
            setMessages((prev) =>
              prev.map((m) =>
                m.id === assistantId ? { ...m, toolCalls: [...(m.toolCalls || []), record] } : m,
              ),
            );
            setStatusText(t.loadingStep(record.step, 12, record.tool));
          },
          onTraceEntry: (entry) => {
            setTraceEntries((prev) => [...prev, entry]);
            if (entry.promptTokens) {
              setContextBudget((b) => ({ ...b, peak: Math.max(b.peak, entry.promptTokens || 0) }));
            }
          },
          onWorkspaceUpdate: (nextFiles) => {
            setWorkspaceFiles(nextFiles);
            workspaceRef.current = nextFiles;
          },
          onCheckpoint: (ckpt) => {
            setCheckpoints((prev) => [ckpt, ...prev]);
            if (ckpt.existedBefore) {
              setDiffOriginals((prev) =>
                prev[ckpt.path] !== undefined ? prev : { ...prev, [ckpt.path]: ckpt.beforeContent },
              );
            }
          },
          onOpenFile: (filePath, nextMode = 'code') => {
            setSelectedFilePath(filePath);
            setEditorMode(nextMode);
          },
          onApprovalRequest: (req) =>
            new Promise<ApprovalDecision>((resolve) => {
              approvalResolverRef.current = resolve;
              setPendingApproval(req);
            }),
          onFinalToken: (rendered, liveTps) => {
            setMessages((prev) =>
              prev.map((m) => (m.id === assistantId ? { ...m, content: rendered } : m)),
            );
            if (liveTps > 0) {
              setStatusText(`${lang === 'zh' ? '生成中' : 'Synthesizing'} · ${liveTps.toFixed(1)} tok/s`);
            }
          },
        },
      });

      if (result.audioUrl) {
        objectUrlsRef.current.push(result.audioUrl);
      }

      const meta = `${result.totalGeneratedTokens} tok · ${result.decodeTokPerSec.toFixed(1)} tok/s · ${result.toolCalls.length} tool${result.toolCalls.length === 1 ? '' : 's'}${result.aborted ? ' · Stopped' : ''}`;
      setMessages((prev) =>
        prev.map((m) =>
          m.id === assistantId
            ? {
                ...m,
                content: result.finalText,
                audioUrl: result.audioUrl,
                toolCalls: result.toolCalls,
                checkpoints: result.checkpoints,
                testSummary: result.testSummary,
                meta,
              }
            : m,
        ),
      );
      setContextBudget({ peak: result.peakPromptTokens, max: result.maxSeqLen });
      if (result.checkpoints.length > 0) {
        const lastEdited = result.checkpoints[result.checkpoints.length - 1];
        setSelectedFilePath(lastEdited.path);
        setEditorMode(lastEdited.existedBefore ? 'diff' : 'code');
        setPreviewRefreshSeq((s) => s + 1);
      }

      const perfMem = (performance as Performance & { memory?: { usedJSHeapSize: number } }).memory;
      setStats((s) => ({
        ...s,
        prefillSpeed:
          result.prefillMs > 0 && result.totalPromptTokens > 0
            ? `${(result.totalPromptTokens / (result.prefillMs / 1000)).toFixed(1)} tok/s`
            : s.prefillSpeed,
        decodeSpeed: `${result.decodeTokPerSec.toFixed(1)} tok/s`,
        ctxPos: result.curPos,
        heapMB: perfMem ? `${(perfMem.usedJSHeapSize / (1024 * 1024)).toFixed(1)} MB` : s.heapMB,
      }));
      setStatus('ready');
      setStatusText(getReadyLabel(modelForTurn, lang));
    } catch (err) {
      setStatus('error');
      setStatusText(`Agent error: ${err instanceof Error ? err.message : String(err)}`);
    }
  };

  const handleSubmit = async () => {
    if (termsBlocked || status === 'loading' || status === 'generating') return;
    let client = clientRef.current;
    let resolved = activeModel;

    if (statusRef.current !== 'ready' || !client) {
      const loaded = await loadModel(input.trim() || null);
      if (!loaded) return;
      client = loaded.client;
      resolved = loaded.resolvedModel;
    }

    if (mode === 'agent') {
      await runAgentTask(client, resolved, input, true);
    } else {
      await sendDirectMessage(client, resolved, input);
    }
  };

  const handleRunStarter = async (prompt: string) => {
    if (termsBlocked || status === 'loading' || status === 'generating') return;
    setMode('agent');
    let client = clientRef.current;
    let resolved = activeModel;

    if (statusRef.current !== 'ready' || !client) {
      const loaded = await loadModel(prompt);
      if (!loaded) return;
      client = loaded.client;
      resolved = loaded.resolvedModel;
    }

    await runAgentTask(client, resolved, prompt, false);
  };

  const runDirectShowcase = async (kind: 'vqa' | 'voice' | 'fsm' | 'ngram') => {
    if (status !== 'ready' || !clientRef.current) return;
    const client = clientRef.current;
    setMode('chat');

    if (kind === 'vqa') {
      const resp = await fetch('/samples/demo-shapes.png');
      const blob = await resp.blob();
      const previewUrl = URL.createObjectURL(blob);
      objectUrlsRef.current.push(previewUrl);
      const img = await bitmapBlobToRgba(blob, 'demo-shapes.png', previewUrl);
      setEnableFsm(false);
      setEnableNgram(false);
      setGenerationMode('text');
      await sendDirectMessage(client, activeModel, 'Describe the main shape and colors visible in this image.', {
        image: img,
        audio: null,
        generationMode: 'text',
        enableFsm: false,
        enableNgram: false,
        maxNewTokens: 96,
        freshContext: true,
      });
    } else if (kind === 'voice') {
      setEnableFsm(false);
      setEnableNgram(false);
      setGenerationMode('interleaved');
      await sendDirectMessage(
        client,
        activeModel,
        'Say a concise two-sentence greeting demonstrating browser 24 kHz speech output.',
        {
          image: null,
          audio: null,
          generationMode: 'interleaved',
          enableFsm: false,
          enableNgram: false,
          maxNewTokens: 160,
          freshContext: true,
        },
      );
    } else if (kind === 'fsm') {
      const schemaStr = JSON.stringify(FSM_PRESETS[0].schema, null, 2);
      setEnableNgram(false);
      setEnableFsm(true);
      setFsmPresetId(FSM_PRESETS[0].id);
      setFsmSchemaText(schemaStr);
      setGenerationMode('text');
      await sendDirectMessage(
        client,
        activeModel,
        'Return JSON for the Ignis WebGPU inference engine using OPFS storage running strictly in browser.',
        {
          image: null,
          audio: null,
          generationMode: 'text',
          enableFsm: true,
          schemaJson: schemaStr,
          enableNgram: false,
          maxNewTokens: 96,
          freshContext: true,
        },
      );
    } else if (kind === 'ngram') {
      setEnableFsm(false);
      setEnableNgram(true);
      setGenerationMode('text');
      await sendDirectMessage(
        client,
        activeModel,
        'Repeat this exact line three times word for word:\nIgnis streams WebGPU kernels from OPFS with zero server backend.',
        {
          image: null,
          audio: null,
          generationMode: 'text',
          enableFsm: false,
          enableNgram: true,
          maxNewTokens: 96,
          freshContext: true,
        },
      );
    }
  };

  return (
    <div className="studio-shell">
      <TopBar
        lang={lang}
        onToggleLang={() => {
          const nextLang: UiLang = lang === 'en' ? 'zh' : 'en';
          setLang(nextLang);
          if (status === 'idle') {
            setStatusText(COPY[nextLang].statusIdleShort);
          } else if (status === 'ready') {
            setStatusText(getReadyLabel(activeModel, nextLang));
          }
        }}
        theme={theme}
        onToggleTheme={() => setTheme((prev) => (prev === 'dark' ? 'light' : 'dark'))}
        editorOpen={editorOpen}
        onToggleEditor={() => setEditorOpen((v) => !v)}
        rightPaneOpen={rightPaneOpen}
        onToggleRightPane={() => setRightPaneOpen((v) => !v)}
        approvalPolicy={approvalPolicy}
        onToggleApprovalPolicy={() => setApprovalPolicy((p) => (p === 'auto' ? 'ask' : 'auto'))}
        models={models}
        selectedModelId={selectedModelId}
        activeModel={activeModel}
        customUrl={customUrl}
        onCustomUrlChange={setCustomUrl}
        onSelectModel={handleModelChange}
        termsAcked={termsAcked}
        onTermsAckChange={setTermsAcked}
        status={status}
        statusText={statusText}
        loadPct={loadPct}
        onLoadModel={() => {
          loadModel();
        }}
      />

      <div className="studio-body">
        <main className="agent-pane">
          <div className="agent-subbar">
            <div className="segmented" role="tablist" aria-label="Execution mode" style={{ width: 160 }}>
              <button
                id="btn-mode-agent"
                type="button"
                role="tab"
                aria-selected={mode === 'agent'}
                className={mode === 'agent' ? 'active' : ''}
                disabled={status === 'loading' || status === 'generating'}
                style={{ height: 24, fontSize: 12 }}
                onClick={() => setMode('agent')}
              >
                <Bot size={12} />
                <span>{t.modeAgent}</span>
              </button>
              <button
                id="btn-mode-chat"
                type="button"
                role="tab"
                aria-selected={mode === 'chat'}
                className={mode === 'chat' ? 'active' : ''}
                disabled={status === 'loading' || status === 'generating'}
                style={{ height: 24, fontSize: 12 }}
                onClick={() => setMode('chat')}
              >
                <MessageSquare size={12} />
                <span>{t.modeChat}</span>
              </button>
            </div>

            <button
              id="btn-new-session"
              type="button"
              className="icon-btn"
              style={{ height: 26, fontSize: 12 }}
              disabled={status === 'loading' || status === 'generating' || (messages.length === 0 && status !== 'ready')}
              onClick={resetChat}
            >
              <Plus size={12} />
              <span>{t.newSession}</span>
            </button>
          </div>

          {status === 'loading' && (
            <div className="load-progress-strip">
              <div className="load-progress-fill" style={{ width: `${loadPct}%` }} />
            </div>
          )}

          <ThreadView
            lang={lang}
            messages={messages}
            status={status}
            statusText={statusText}
            loadProgress={loadProgress}
            activeModel={activeModel}
            termsBlocked={termsBlocked}
            pendingApproval={pendingApproval}
            onResolveApproval={handleResolveApproval}
            onLoadModel={() => {
              loadModel();
            }}
            onRunStarter={handleRunStarter}
            onInspectDiff={(path) => {
              const target = readWorkspaceFile(workspaceRef.current, path)?.path || path;
              setSelectedFilePath(target);
              setEditorMode('diff');
              setEditorOpen(true);
            }}
            onRollbackCheckpoint={handleRollbackCheckpoint}
            onPreviewHtml={(_, html) => {
              handleSaveFile('preview-snippet.html', html);
            }}
            scrollEndRef={scrollEndRef}
          />

          <Composer
            lang={lang}
            mode={mode}
            input={input}
            onInputChange={setInput}
            status={status}
            termsBlocked={termsBlocked}
            supportsVision={activeModel.capabilities.vision}
            supportsAudioIn={activeModel.capabilities.audioIn}
            pendingImage={pendingImage}
            pendingAudio={pendingAudio}
            onRemoveImage={() => setPendingImage(null)}
            onRemoveAudio={() => setPendingAudio(null)}
            onImageFile={handleImageFile}
            onAudioFile={handleAudioFile}
            onSubmit={handleSubmit}
            onStop={stopGeneration}
          />
        </main>

        {editorOpen && (
          <CodeEditorPane
            lang={lang}
            theme={theme}
            files={workspaceFiles}
            selectedPath={selectedFilePath}
            editorMode={editorMode}
            diffOriginals={diffOriginals}
            onSelectFile={(p) => {
              setSelectedFilePath(p);
              setEditorMode('code');
            }}
            onModeChange={setEditorMode}
            onSaveFile={handleSaveFile}
            onRevertFile={handleRevertFile}
            onCreateFile={handleCreateFile}
            onUploadFiles={handleUploadFiles}
            onResetProject={handleResetProject}
            busy={status === 'generating'}
          />
        )}

        {rightPaneOpen && (
          <WorkspacePane
            lang={lang}
            theme={theme}
            activeTab={workspaceTab}
            onTabChange={setWorkspaceTab}
            onClose={() => setRightPaneOpen(false)}
            sandboxStatus={sandboxStatus}
            sandboxDetail={sandboxDetail}
            previewUrl={previewUrl}
            previewRefreshSeq={previewRefreshSeq}
            shellChannel={shellChannel}
            serverChannel={serverChannel}
            onAttachShell={wcAttachInteractiveShell}
            onRestartServer={() => {
              startOrRestartDevServer({
                onStatus: (st, detail) => {
                  setSandboxStatus(st);
                  if (detail) setSandboxDetail(detail);
                },
                onServerLog: (chunk) => pushServerOutput(chunk),
                onServerReady: (url) => setPreviewUrl(url),
              }).catch(() => {});
            }}
            traceEntries={traceEntries}
            checkpoints={checkpoints}
            onRollbackCheckpoint={handleRollbackCheckpoint}
            contextBudget={contextBudget}
            deviceProfile={deviceProfile}
            gpuCaps={gpuCaps}
            stats={stats}
            status={status}
            activeModel={activeModel}
            archKind={archKind}
            enableThinking={enableThinking}
            onThinkingChange={setEnableThinking}
            effort={effort}
            onEffortChange={setEffort}
            generationMode={generationMode}
            onGenerationModeChange={setGenerationMode}
            enableFsm={enableFsm}
            onFsmToggle={(next) => {
              setEnableFsm(next);
              if (next) {
                setEnableNgram(false);
                setGenerationMode('text');
              }
            }}
            fsmPresets={FSM_PRESETS}
            fsmPresetId={fsmPresetId}
            onFsmPresetChange={(id) => {
              setFsmPresetId(id);
              const found = FSM_PRESETS.find((x) => x.id === id);
              if (found) setFsmSchemaText(JSON.stringify(found.schema, null, 2));
            }}
            fsmSchemaText={fsmSchemaText}
            onFsmSchemaTextChange={setFsmSchemaText}
            enableNgram={enableNgram}
            onNgramToggle={setEnableNgram}
            temperature={temperature}
            onTemperatureChange={setTemperature}
            maxNewTokens={maxNewTokens}
            onMaxNewTokensChange={setMaxNewTokens}
            onRunShowcase={runDirectShowcase}
            onClearOpfs={clearOpfsCache}
          />
        )}
      </div>
    </div>
  );
}
