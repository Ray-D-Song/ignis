'use client';

import React, { useEffect, useRef, useState } from 'react';
import type { ModelPreset } from '@/lib/model-paths';
import { IgnisClient } from '../../../runtime/ignis-client.mjs';
import { decodeWav, resampleLinear } from '../../../runtime/multimodal/audio-dsp.mjs';
import { detectDeviceProfile } from '../../../runtime/device-profile.mjs';

type IgnisDeviceProfile = Awaited<ReturnType<typeof detectDeviceProfile>>;

interface PendingImage {
  data: Uint8Array;
  width: number;
  height: number;
  channels: number;
  previewUrl: string;
  name: string;
}

interface PendingAudio {
  pcm16k: Float32Array;
  durationSec: number;
  name: string;
}

interface ChatMessage {
  id: string;
  role: 'user' | 'assistant';
  content: string;
  modelLabel?: string;
  image?: PendingImage | null;
  audio?: PendingAudio | null;
  audioUrl?: string;
  meta?: string;
  fsmValid?: boolean;
  fsmMedianMs?: number | null;
  specAccepted?: number;
  specProposed?: number;
}

interface IgnisProgressEvent {
  file: string;
  loaded: number;
  total: number;
  cached: boolean;
}

interface IgnisTokenEvent {
  tokenId: number;
  text: string;
  fullText?: string;
  step: number;
  tokPerSec?: number;
}

interface SendOverrides {
  image?: PendingImage | null;
  audio?: PendingAudio | null;
  generationMode?: OmniGenMode;
  enableFsm?: boolean;
  schemaJson?: string;
  enableNgram?: boolean;
  maxNewTokens?: number;
  freshContext?: boolean;
}

type ArchKind = 'spark' | 'lfm3' | 'gemma4';
type OmniGenMode = 'text' | 'interleaved' | 'audio';
type UiLang = 'en' | 'zh';

const CUSTOM_MODEL_ID = '__custom__';

const FSM_PRESETS: Array<{ id: string; label: string; schema: Record<string, unknown> }> = [
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

const DEFAULT_FSM_SCHEMA_STR = JSON.stringify(FSM_PRESETS[0].schema, null, 2);

const COPY = {
  en: {
    subtitle: 'Browser-Native WebGPU · Zero Dependencies',
    heroTitle: 'Ignis · Browser-Native WebGPU LLM & Full-Omni Engine',
    heroDesc:
      'Built from first principles without ONNX, TVM, or llama.cpp: 256B-aligned .ignis shards, OPFS SyncAccessHandle row streaming, W3C baseline WGSL mega-kernels, Rust/WASM JSON Schema FSM, and Dual-Mode Speculative Decoding.',
    hideHero: 'Hide Overview',
    showHero: 'Show Overview',
    modelSection: 'Model Configuration',
    modelSelect: 'Select Model Preset or Custom URL',
    customUrlLabel: 'Custom .ignis Base URL (must serve manifest.json + shards)',
    customOption: 'Custom .ignis Model URL...',
    loadModel: 'Load Model to WebGPU',
    reloadModel: 'Reload Model',
    loadingPct: (p: number) => `Loading ${p}%`,
    clearOpfs: 'Clear OPFS Cache',
    hardwareSection: 'Hardware Readiness',
    inferenceSection: 'Inference & Acceleration',
    thinking: 'Thinking Channel (<think>)',
    effort: 'Thinking Effort',
    outputModality: 'LFM3 Output Modality',
    fsmToggle: 'JSON Schema FSM (WASM)',
    fsmPreset: 'Schema Preset',
    fsmCustomJson: 'Schema JSON',
    ngramToggle: 'N-gram Speculative (0 MiB VRAM)',
    showcasesSection: '1-Click Showcases',
    telemetrySection: 'Live Telemetry',
    resetChat: 'Reset Chat',
    idleStatus: 'Ready to load model (first run streams cleanly into browser OPFS)',
    switchedStatus: 'Model switched — click "Load Model to WebGPU" on the left',
    send: 'Send',
    stop: 'Stop',
    inputReady: 'Type a prompt (Enter to send, Shift+Enter for newline)...',
    inputWait: 'Load a model on the left sidebar to begin...',
    analyzeMm: 'Describe and analyze the attached media.',
    starterTitle: 'Pure In-Browser WebGPU · Multi-Arch & Full-Omni',
    starterBody:
      'Streams weights and multi-gigabyte embeddings through OPFS directly into WebGPU inside a Dedicated Worker. Load LFM3-Omni-2B (Vision + Audio + 24 kHz Voice), Spark-X2.6-4B (85–134 tok/s), or Gemma-4-E4B-it, or click any Showcase button on the left.',
    termsNotice: 'Gemma-4-E4B-it is governed by the Gemma Terms of Use.',
    termsAck: 'I accept the Gemma Terms of Use (ai.google.dev/gemma/terms)',
    opfsCleared: 'OPFS cache cleared',
    thinkOpen: 'Thinking...',
    thinkAborted: 'Thinking (stopped or budget reached)',
    thinkView: 'View reasoning process',
    audioGenerated: '(24 kHz WAV speech response generated)',
    starterPrompts: [
      'Explain first-principles engineering cleanly in three bullet points.',
      'Write a concise Python function for longest palindromic substring in O(1) extra space.',
      'Compare WebGPU vs WebAssembly bottlenecks for running LLMs directly inside a browser.',
      'Summarize why streaming embeddings from OPFS saves gigabytes of GPU VRAM.',
    ],
  },
  zh: {
    subtitle: 'vinext · WebGPU · 零外部运行时',
    heroTitle: 'Ignis · 第一性原理纯浏览器 WebGPU 全模态引擎',
    heroDesc:
      '零 ONNX / TVM / llama.cpp 依赖：256B 对齐 .ignis 分片、OPFS SyncAccessHandle 外置词表流式查表、W3C 基线 WGSL 融合算子、Rust/WASM JSON Schema FSM 与无损投机解码。',
    hideHero: '收起概览',
    showHero: '展开概览',
    modelSection: '模型配置',
    modelSelect: '选择已打包权重或自定义 URL',
    customUrlLabel: '自定义 .ignis 模型目录 URL（需含 manifest.json）',
    customOption: '自定义 .ignis URL...',
    loadModel: '加载模型到 WebGPU',
    reloadModel: '重新加载模型',
    loadingPct: (p: number) => `加载中 ${p}%`,
    clearOpfs: '清理 OPFS 缓存',
    hardwareSection: 'WebGPU 硬件就绪探测',
    inferenceSection: '推理与加速参数',
    thinking: '深度思考',
    effort: '思考预算 (thinking_effort)',
    outputModality: 'LFM3 输出模态',
    fsmToggle: 'JSON Schema FSM 约束解码',
    fsmPreset: 'Schema 预设',
    fsmCustomJson: 'Schema JSON 定义',
    ngramToggle: 'N-gram 投机解码 (零额外显存)',
    showcasesSection: '一键能力演示',
    telemetrySection: '实时硬件遥测',
    resetChat: '清空对话',
    idleStatus: '待加载模型（首次将流式缓存至浏览器 OPFS）',
    switchedStatus: '已切换模型，请点击左侧「加载模型到 WebGPU」',
    send: '发送',
    stop: '停止',
    inputReady: '输入消息（Enter 发送，Shift+Enter 换行）...',
    inputWait: '请先在左侧点击「加载模型到 WebGPU」',
    analyzeMm: '请分析并回答相关内容。',
    starterTitle: 'Ignis WebGPU · 纯浏览器多架构全模态推理',
    starterBody:
      '零后端、零外部运行时：支持 LFM3-Omni-2B（视觉/音频理解 + 24 kHz 语音合成）、Spark-X2.6-4B（85–134 tok/s）与 Gemma-4-E4B-it（PLE + 262K 词表），加载后可直接运行左侧一键 Showcase。',
    termsNotice: '使用 Gemma-4-E4B-it 需同意 Google Gemma Terms of Use。',
    termsAck: '我已阅读并同意 Gemma 使用条款 (ai.google.dev/gemma/terms)',
    opfsCleared: 'OPFS 模型缓存已清空',
    thinkOpen: '正在思考...',
    thinkAborted: '思考过程（已中止或达上限）',
    thinkView: '查看思考过程',
    audioGenerated: '（已生成 24 kHz 语音回复）',
    starterPrompts: [
      '请用通俗且严谨的语言解释什么是第一性原理。',
      'Write a Python function to find the longest palindromic substring with O(n^2) time and O(1) space.',
      '一辆高铁以 280 km/h 行驶 45 分钟，然后以 160 km/h 行驶 30 分钟，总路程和平均速度各是多少？',
      '比较 WebGPU 与 WebAssembly 在纯浏览器端运行大语言模型时的本质瓶颈差异。',
    ],
  },
} as const;

function getArchKind(model?: ModelPreset): ArchKind {
  if (!model) return 'spark';
  if (model.topModelType === 'lfm3_omni' || model.architecture === 'lfm3') return 'lfm3';
  if (model.topModelType === 'gemma4' || model.architecture === 'gemma4_text') return 'gemma4';
  return 'spark';
}

function getReadyLabel(model: ModelPreset | undefined, lang: UiLang): string {
  const headLabel = model?.lmHeadDtype === 'u32_q8' ? 'INT8_G64 LM Head' : 'INT4_G64 LM Head';
  const isOmni = getArchKind(model) !== 'spark';
  if (lang === 'zh') {
    return isOmni ? `就绪 · ${headLabel} · Full Omni` : `就绪 · ${headLabel}`;
  }
  return isOmni ? `Ready · ${headLabel} · Full Omni` : `Ready · ${headLabel}`;
}

function splitThinking(raw: string): { think: string; answer: string; isThinking: boolean } {
  const pairs: Array<[string, string]> = [
    ['<think>', '</think>'],
    ['<|thought|>', '<thought|>'],
  ];
  for (const [startTag, endTag] of pairs) {
    const sIdx = raw.indexOf(startTag);
    const eIdx = raw.indexOf(endTag);
    if (sIdx !== -1 && eIdx !== -1 && eIdx >= sIdx) {
      return {
        think: raw.slice(sIdx + startTag.length, eIdx).trim(),
        answer: raw.slice(eIdx + endTag.length).trimStart(),
        isThinking: false,
      };
    }
    if (sIdx !== -1 && eIdx === -1) {
      return {
        think: raw.slice(sIdx + startTag.length).trimStart(),
        answer: '',
        isThinking: true,
      };
    }
    if (eIdx !== -1) {
      return {
        think: raw.slice(0, eIdx).trim(),
        answer: raw.slice(eIdx + endTag.length).trimStart(),
        isThinking: false,
      };
    }
  }
  return { think: '', answer: raw, isThinking: false };
}

function buildSparkSystemPrefix(enableThinking: boolean, effort: 'low' | 'high'): string {
  const reasoning = enableThinking
    ? `\n\`thinking_effort\` guides on how much to think in your thinking channel (not including the response channel), supported values include \`low\` and \`high\`.\nNow the system is invoked with \`thinking_effort=${effort}\`.`
    : '';
  return `<｜start▁of▁sentence｜><|System|>\nyou are a helpful assistant.${reasoning}<｜end▁of▁sentence｜>`;
}

function medianNumber(values?: number[]): number | null {
  if (!values || values.length === 0) return null;
  const sorted = [...values].sort((a, b) => a - b);
  return sorted[Math.floor(sorted.length / 2)];
}

async function bitmapBlobToRgba(blob: Blob, name: string, previewUrl: string): Promise<PendingImage> {
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

export function ChatWorkspace({
  models,
  isPublicCdn = false,
}: {
  models: ModelPreset[];
  isPublicCdn?: boolean;
}) {
  const initialLang: UiLang = isPublicCdn ? 'en' : 'zh';
  const [lang, setLang] = useState<UiLang>(initialLang);
  const t = COPY[lang];

  const defaultModel = models[0]?.id || CUSTOM_MODEL_ID;
  const [selectedModelId, setSelectedModelId] = useState(defaultModel);
  const [customUrl, setCustomUrl] = useState('');
  const [customPresetOverride, setCustomPresetOverride] = useState<Partial<ModelPreset> | null>(null);
  const [termsAcked, setTermsAcked] = useState(false);

  const [status, setStatus] = useState<'idle' | 'loading' | 'ready' | 'generating' | 'error'>('idle');
  const [statusText, setStatusText] = useState<string>(COPY[initialLang].idleStatus);
  const [loadPct, setLoadPct] = useState(0);
  const [showHero, setShowHero] = useState(true);

  const [enableThinking, setEnableThinking] = useState(() => getArchKind(models[0]) === 'spark');
  const [effort, setEffort] = useState<'low' | 'high'>('low');
  const [generationMode, setGenerationMode] = useState<OmniGenMode>('text');
  const [temperature, setTemperature] = useState(0.6);
  const [maxNewTokens, setMaxNewTokens] = useState(1536);

  const [enableFsm, setEnableFsm] = useState(false);
  const [fsmPresetId, setFsmPresetId] = useState(FSM_PRESETS[0].id);
  const [fsmSchemaText, setFsmSchemaText] = useState(DEFAULT_FSM_SCHEMA_STR);
  const [enableNgram, setEnableNgram] = useState(false);

  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [input, setInput] = useState('');
  const [pendingImage, setPendingImage] = useState<PendingImage | null>(null);
  const [pendingAudio, setPendingAudio] = useState<PendingAudio | null>(null);

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
  const kvTurnsRef = useRef<number>(0);
  const kvThinkingKeyRef = useRef<string>('');
  const objectUrlsRef = useRef<string[]>([]);
  const imgInputRef = useRef<HTMLInputElement | null>(null);
  const audInputRef = useRef<HTMLInputElement | null>(null);
  const scrollEndRef = useRef<HTMLDivElement | null>(null);

  const customModelPreset: ModelPreset = {
    id: CUSTOM_MODEL_ID,
    label: t.customOption,
    description: customUrl ? `Remote .ignis base: ${customUrl}` : 'Enter any HTTPS directory containing manifest.json & .ignis shards',
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
  const archKind = getArchKind(activeModel);

  const revokeAllObjectUrls = () => {
    for (const u of objectUrlsRef.current) URL.revokeObjectURL(u);
    objectUrlsRef.current = [];
  };

  useEffect(() => {
    (async () => {
      try {
        const profile = await detectDeviceProfile();
        setDeviceProfile(profile);
        if (!profile.supported) {
          setGpuCaps('WebGPU Unsupported');
          return;
        }
        const iso = window.crossOriginIsolated ? 'COOP/COEP' : 'No-COI';
        setGpuCaps(
          `WebGPU · f16:${profile.hasF16 ? '✓' : '✗'} · subgroups:${profile.hasSubgroups ? '✓' : '✗'} · ${iso}`,
        );
      } catch {
        setGpuCaps('WebGPU Probe Failed');
      }
    })();

    return () => {
      revokeAllObjectUrls();
      if (clientRef.current) {
        clientRef.current.destroy();
        clientRef.current = null;
      }
    };
  }, []);

  useEffect(() => {
    scrollEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages]);

  const handleModelChange = (nextId: string) => {
    setSelectedModelId(nextId);
    setTermsAcked(false);
    setCustomPresetOverride(null);
    const nextModel = models.find((m) => m.id === nextId);
    const nextKind = getArchKind(nextModel);
    setEnableThinking(nextKind === 'spark');
    setGenerationMode('text');
    setPendingImage(null);
    setPendingAudio(null);
    kvTurnsRef.current = 0;
    kvThinkingKeyRef.current = '';
    if (loadedModelIdRef.current && loadedModelIdRef.current !== nextId && status === 'ready') {
      setStatus('idle');
      setStatusText(t.switchedStatus);
    }
  };

  const handleImageUpload = async (ev: React.ChangeEvent<HTMLInputElement>) => {
    const file = ev.target.files?.[0];
    ev.target.value = '';
    if (!file) return;
    const previewUrl = URL.createObjectURL(file);
    objectUrlsRef.current.push(previewUrl);
    const img = await bitmapBlobToRgba(file, file.name, previewUrl);
    setPendingImage(img);
  };

  const handleAudioUpload = async (ev: React.ChangeEvent<HTMLInputElement>) => {
    const file = ev.target.files?.[0];
    ev.target.value = '';
    if (!file) return;
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
      setStatusText(t.opfsCleared);
    } catch (err) {
      setStatus('error');
      setStatusText(`OPFS clear failed: ${err instanceof Error ? err.message : String(err)}`);
    }
  };

  const loadModel = async () => {
    if (!activeModel) return;
    const baseUrl = (activeModel.isCustom ? customUrl.trim() : activeModel.modelBaseUrl || `/api/models/${activeModel.id}`).replace(/\/+$/, '');
    if (!baseUrl) {
      setStatus('error');
      setStatusText('Enter a valid custom .ignis base URL first');
      return;
    }

    setStatus('loading');
    setLoadPct(0);
    setStatusText(`Initializing ${activeModel.label}...`);
    try {
      if (clientRef.current) {
        await clientRef.current.destroy();
        clientRef.current = null;
      }
      const client = new IgnisClient('/ignis/runtime/ignis-worker.mjs');
      clientRef.current = client;

      client.onProgress = (p: IgnisProgressEvent) => {
        const pct = p.total ? Math.round((p.loaded / p.total) * 100) : 0;
        setLoadPct(pct);
        setStatusText(`${p.cached ? 'OPFS Hit' : 'Streaming to OPFS'}: ${p.file} (${pct}%)`);
      };
      client.onDeviceLost = (reason: string) => {
        setStatus('error');
        setStatusText(`GPUDevice Lost: ${reason}`);
      };

      const safeKey = activeModel.isCustom
        ? `custom-${baseUrl.replace(/[^a-zA-Z0-9_-]/g, '_').slice(-48)}`
        : `${isPublicCdn ? 'cdn' : 'vinext'}-${activeModel.id}`;

      const preKind = getArchKind(activeModel);
      const maxSeq = preKind === 'spark' ? 8192 : 4096;
      const info = await client.init({
        modelBaseUrl: baseUrl,
        modelKey: safeKey,
        kernelBaseUrl: '/ignis/kernels',
        wasmUrl: '/ignis/runtime/wasm/ignis_tokenizer.wasm',
        maxSeqLen: maxSeq,
        maxBatchTokens: 64,
      });

      let resolvedPreset = activeModel;
      if (activeModel.isCustom && info.blueprint) {
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
        resolvedPreset = { ...activeModel, ...inferred };
      }

      loadedModelIdRef.current = activeModel.id;
      kvTurnsRef.current = 0;
      kvThinkingKeyRef.current = '';
      const perfMem = (performance as Performance & { memory?: { usedJSHeapSize: number } }).memory;
      setStats((s) => ({
        ...s,
        initMs: `${(info.initMs / 1000).toFixed(2)}s`,
        cacheMode:
          lang === 'zh'
            ? (info.cacheMisses === 0 ? `OPFS 热命中 (${info.cacheHits})` : `冷写入 (${info.cacheMisses}片)`)
            : (info.cacheMisses === 0 ? `OPFS Hit (${info.cacheHits})` : `Cold Write (${info.cacheMisses})`),
        ctxPos: 0,
        maxCtx: maxSeq,
        heapMB: perfMem ? `${(perfMem.usedJSHeapSize / (1024 * 1024)).toFixed(1)} MB` : '-',
      }));
      setLoadPct(100);
      setStatus('ready');
      setStatusText(getReadyLabel(resolvedPreset, lang));
    } catch (err) {
      setStatus('error');
      setStatusText(`Load failed: ${err instanceof Error ? err.message : String(err)}`);
    }
  };

  const resetChat = async () => {
    if (clientRef.current && status === 'ready') {
      await clientRef.current.reset();
    }
    revokeAllObjectUrls();
    kvTurnsRef.current = 0;
    kvThinkingKeyRef.current = '';
    setPendingImage(null);
    setPendingAudio(null);
    setMessages([]);
    setStats((s) => ({ ...s, ctxPos: 0 }));
  };

  const stopGeneration = async () => {
    if (clientRef.current && status === 'generating') {
      await clientRef.current.abort();
    }
  };

  const sendMessage = async (overrideText?: string, overrideOpts?: SendOverrides) => {
    const rawText = (overrideText ?? input).trim();
    const imgAttach = overrideOpts?.image !== undefined ? overrideOpts.image : (overrideText ? null : pendingImage);
    const audAttach = overrideOpts?.audio !== undefined ? overrideOpts.audio : (overrideText ? null : pendingAudio);
    const text = rawText || (imgAttach || audAttach ? t.analyzeMm : '');
    if (!text || !clientRef.current || status !== 'ready') return;

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

    const activeMode: OmniGenMode = activeModel?.capabilities.audioOut
      ? (overrideOpts?.generationMode ?? generationMode)
      : 'text';
    const hasMm = Boolean(imgAttach || audAttach);
    const useNgram = !useFsm && !hasMm && activeMode === 'text' && (overrideOpts?.enableNgram ?? enableNgram);
    const effectiveThinking = enableThinking && !useFsm;

    if (!overrideText) {
      setInput('');
      setPendingImage(null);
      setPendingAudio(null);
    }

    const userMsg: ChatMessage = {
      id: `u-${Date.now()}`,
      role: 'user',
      content: text,
      image: imgAttach,
      audio: audAttach,
    };
    const assistantId = `a-${Date.now()}`;
    const shortModelName = `${(activeModel?.label || 'Ignis').split('·')[0].trim()} (Ignis)`;
    const thinkPrefix = effectiveThinking ? (archKind === 'gemma4' ? '<|thought|>' : '<think>') : '';
    const assistantMsg: ChatMessage = {
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
    setStatusText(lang === 'zh' ? 'Prefill 上下文中...' : 'Prefilling context...');

    const client = clientRef.current;
    const thinkingKey = `${archKind}:${effectiveThinking}:${effort}`;
    let promptDeltaText = '';
    const reqImages: Array<{ data: Uint8Array; width: number; height: number; channels: number }> = [];
    const reqAudios: Float32Array[] = [];
    const includeMm = !useFsm && !useNgram;

    if (archKind === 'spark') {
      const canIncremental =
        !useFresh &&
        kvTurnsRef.current === messages.length &&
        kvThinkingKeyRef.current === thinkingKey &&
        kvTurnsRef.current > 0 &&
        !useFsm &&
        !useNgram;
      if (canIncremental) {
        promptDeltaText = `<｜end▁of▁sentence｜><｜start▁of▁sentence｜><|User|>${text}<｜end▁of▁sentence｜><｜start▁of▁sentence｜><|Bot|>${effectiveThinking ? '<think>' : '</think>'}`;
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
    } else if (archKind === 'lfm3') {
      await client.reset();
      let fullPrompt = '<|startoftext|>';
      for (const m of promptMessages) {
        if (m.role === 'user') {
          let mmTags = '';
          if (includeMm && m.image) {
            reqImages.push({ data: m.image.data, width: m.image.width, height: m.image.height, channels: m.image.channels });
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
            reqImages.push({ data: m.image.data, width: m.image.width, height: m.image.height, channels: m.image.channels });
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
        archKind === 'spark' && effectiveThinking
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
        sampling: useFsm || useNgram
          ? { temperature: 0, topK: 1, topP: 1.0, repetitionPenalty: 1.0 }
          : {
              temperature,
              topK: 40,
              topP: 0.95,
              repetitionPenalty: archKind === 'spark' ? 1.05 : 1.0,
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

      if (archKind === 'spark' && !res.aborted && !useFresh && !useFsm && !useNgram) {
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
      setStatusText(getReadyLabel(activeModel, lang));
    } catch (err) {
      setStatus('error');
      setStatusText(`Generation error: ${err instanceof Error ? err.message : String(err)}`);
    }
  };

  const runShowcase = async (kind: 'vqa' | 'voice' | 'fsm' | 'ngram') => {
    if (status !== 'ready') return;
    if (kind === 'vqa') {
      const resp = await fetch('/samples/demo-shapes.png');
      const blob = await resp.blob();
      const previewUrl = URL.createObjectURL(blob);
      objectUrlsRef.current.push(previewUrl);
      const img = await bitmapBlobToRgba(blob, 'demo-shapes.png', previewUrl);
      setEnableFsm(false);
      setEnableNgram(false);
      setGenerationMode('text');
      await sendMessage('Describe the main shape and colors visible in this image.', {
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
      await sendMessage('Say a concise two-sentence greeting demonstrating browser 24 kHz speech output.', {
        image: null,
        audio: null,
        generationMode: 'interleaved',
        enableFsm: false,
        enableNgram: false,
        maxNewTokens: 160,
        freshContext: true,
      });
    } else if (kind === 'fsm') {
      const schemaStr = JSON.stringify(FSM_PRESETS[0].schema, null, 2);
      setEnableNgram(false);
      setEnableFsm(true);
      setFsmPresetId(FSM_PRESETS[0].id);
      setFsmSchemaText(schemaStr);
      setGenerationMode('text');
      await sendMessage(
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
      await sendMessage(
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

  const dotClass =
    status === 'ready' ? 'ok' : status === 'loading' || status === 'generating' ? 'busy' : status === 'error' ? 'err' : '';

  const termsBlocked = Boolean(activeModel?.requiresTermsAck && !termsAcked);

  return (
    <div className="app-shell">
      <aside className="sidebar">
        <div className="brand">
          <div className="brand-badge">I</div>
          <div>
            <h1>Ignis</h1>
            <span>{t.subtitle}</span>
          </div>
        </div>

        <div className="card">
          <div className="card-title">{t.modelSection}</div>
          <label className="field">
            {t.modelSelect}
            {/* Model selector MUST remain the first <select> in DOM */}
            <select
              value={selectedModelId}
              disabled={status === 'loading' || status === 'generating'}
              onChange={(e) => handleModelChange(e.target.value)}
            >
              {models.map((m) => (
                <option key={m.id} value={m.id}>{m.label}</option>
              ))}
              <option value={CUSTOM_MODEL_ID}>{t.customOption}</option>
            </select>
          </label>
          {selectedModelId === CUSTOM_MODEL_ID && (
            <label className="field">
              {t.customUrlLabel}
              <input
                id="custom-model-url"
                type="url"
                value={customUrl}
                disabled={status === 'loading' || status === 'generating'}
                placeholder="https://models.example.com/my-model-ignis/"
                onChange={(e) => setCustomUrl(e.target.value)}
              />
            </label>
          )}
          {activeModel && !activeModel.isCustom && (
            <div style={{ fontSize: '0.76rem', color: 'var(--muted)', lineHeight: 1.45 }}>
              {activeModel.description}<br />
              VRAM: <b>{activeModel.gpuMiB} MiB</b> · Embd: <b>{activeModel.embdMiB} MiB</b>
              {activeModel.pleMiB > 0 && <> · PLE: <b>{(activeModel.pleMiB / 1024).toFixed(2)} GiB</b></>}
              {activeModel.licenseName && <> · {activeModel.licenseName}</>}
            </div>
          )}
          {activeModel?.requiresTermsAck && (
            <div className="terms-box">
              <div>{t.termsNotice}</div>
              <label>
                <input
                  id="gemma-terms-checkbox"
                  type="checkbox"
                  checked={termsAcked}
                  onChange={(e) => setTermsAcked(e.target.checked)}
                />
                <span>
                  {t.termsAck}{' '}
                  <a href="https://ai.google.dev/gemma/terms" target="_blank" rel="noreferrer">↗</a>
                </span>
              </label>
            </div>
          )}
          {status === 'loading' && (
            <div className="progress-bar">
              <div className="progress-fill" style={{ width: `${loadPct}%` }} />
            </div>
          )}
          <button
            id="btn-load-model"
            className="btn btn-primary"
            disabled={!activeModel || termsBlocked || status === 'loading' || status === 'generating'}
            onClick={loadModel}
          >
            {status === 'ready' ? t.reloadModel : status === 'loading' ? t.loadingPct(loadPct) : t.loadModel}
          </button>
          <button
            id="btn-clear-opfs"
            type="button"
            className="btn btn-secondary"
            disabled={status === 'loading' || status === 'generating'}
            onClick={clearOpfsCache}
            style={{ fontSize: '0.78rem', padding: '6px 10px' }}
          >
            {t.clearOpfs}
          </button>
        </div>

        <div className="card hw-card" id="hardware-card">
          <div className="card-title">{t.hardwareSection}</div>
          {!deviceProfile ? (
            <span>Probing GPU adapter & WebGPU limits...</span>
          ) : !deviceProfile.supported ? (
            <>
              <span className="hw-badge err">WebGPU Unavailable</span>
              <span>
                Desktop Chromium 138+ or Edge 157+ with hardware WebGPU enabled is required to run quantized WGSL kernels locally.
              </span>
            </>
          ) : (
            <>
              <span className={`hw-badge ${deviceProfile.tier === 'tier1_full' ? 'ok' : 'warn'}`}>
                {deviceProfile.tier} · {deviceProfile.estimatedVramClass.toUpperCase()} VRAM
              </span>
              <div>
                {(deviceProfile.adapterInfo?.description || deviceProfile.adapterInfo?.architecture || deviceProfile.adapterInfo?.vendor || 'WebGPU Adapter')}
              </div>
              <div>
                Recommended: <b>INT{deviceProfile.recommendedWeightBits}_G64</b> · Subgroups:{' '}
                <b>{deviceProfile.recommendedUseSubgroups ? 'on' : 'shmem-tree'}</b> · MaxBuf:{' '}
                <b>{(deviceProfile.maxBufferSize / (1024 * 1024)).toFixed(0)} MiB</b>
              </div>
            </>
          )}
        </div>

        <div className="card">
          <div className="card-title">{t.inferenceSection}</div>
          {activeModel?.capabilities.thinking && (
            <div className="toggle-row">
              <span>{t.thinking}</span>
              <input
                type="checkbox"
                checked={enableThinking && !enableFsm}
                disabled={status === 'generating' || enableFsm}
                onChange={(e) => setEnableThinking(e.target.checked)}
              />
            </div>
          )}
          {archKind === 'spark' && enableThinking && !enableFsm && (
            <label className="field">
              {t.effort}
              <select
                value={effort}
                disabled={status === 'generating'}
                onChange={(e) => setEffort(e.target.value as 'low' | 'high')}
              >
                <option value="low">low (&le;384 tok)</option>
                <option value="high">high (&le;65% budget)</option>
              </select>
            </label>
          )}
          {activeModel?.capabilities.audioOut && (
            <label className="field">
              {t.outputModality}
              <select
                value={generationMode}
                disabled={status === 'generating' || enableFsm}
                onChange={(e) => setGenerationMode(e.target.value as OmniGenMode)}
              >
                <option value="text">Text Only (text)</option>
                <option value="interleaved">Text + 24 kHz Voice (interleaved)</option>
                <option value="audio">24 kHz WAV Only (audio)</option>
              </select>
            </label>
          )}
          <div className="toggle-row">
            <span>{t.fsmToggle}</span>
            <input
              id="toggle-fsm"
              type="checkbox"
              checked={enableFsm}
              disabled={status === 'generating'}
              onChange={(e) => {
                const next = e.target.checked;
                setEnableFsm(next);
                if (next) {
                  setEnableNgram(false);
                  setGenerationMode('text');
                }
              }}
            />
          </div>
          {enableFsm && (
            <>
              <label className="field">
                {t.fsmPreset}
                <select
                  value={fsmPresetId}
                  disabled={status === 'generating'}
                  onChange={(e) => {
                    const p = FSM_PRESETS.find((x) => x.id === e.target.value);
                    setFsmPresetId(e.target.value);
                    if (p) setFsmSchemaText(JSON.stringify(p.schema, null, 2));
                  }}
                >
                  {FSM_PRESETS.map((p) => (
                    <option key={p.id} value={p.id}>{p.label}</option>
                  ))}
                </select>
              </label>
              <label className="field">
                {t.fsmCustomJson}
                <textarea
                  id="fsm-schema-input"
                  className="schema-box"
                  value={fsmSchemaText}
                  disabled={status === 'generating'}
                  onChange={(e) => setFsmSchemaText(e.target.value)}
                />
              </label>
            </>
          )}
          <div className="toggle-row">
            <span>{t.ngramToggle}</span>
            <input
              id="toggle-ngram"
              type="checkbox"
              checked={enableNgram && !enableFsm}
              disabled={status === 'generating' || enableFsm}
              onChange={(e) => setEnableNgram(e.target.checked)}
            />
          </div>
          <label className="field">
            Temperature: {temperature.toFixed(2)}
            <input
              type="range"
              min="0"
              max="1.2"
              step="0.05"
              value={temperature}
              onChange={(e) => setTemperature(parseFloat(e.target.value))}
            />
          </label>
          <label className="field">
            Max New Tokens: {maxNewTokens}
            <input
              type="range"
              min="128"
              max="4096"
              step="128"
              value={maxNewTokens}
              onChange={(e) => setMaxNewTokens(parseInt(e.target.value, 10))}
            />
          </label>
        </div>

        <div className="card">
          <div className="card-title">{t.showcasesSection}</div>
          <div className="showcase-grid">
            <button
              id="showcase-vqa"
              type="button"
              className="btn-showcase"
              disabled={status !== 'ready' || !activeModel?.capabilities.vision}
              onClick={() => runShowcase('vqa')}
            >
              Omni VQA
              <small>demo-shapes.png</small>
            </button>
            <button
              id="showcase-voice"
              type="button"
              className="btn-showcase"
              disabled={status !== 'ready' || !activeModel?.capabilities.audioOut}
              onClick={() => runShowcase('voice')}
            >
              24 kHz Voice
              <small>LiquidCodec WAV</small>
            </button>
            <button
              id="showcase-fsm"
              type="button"
              className="btn-showcase"
              disabled={status !== 'ready'}
              onClick={() => runShowcase('fsm')}
            >
              JSON Schema FSM
              <small>0.3 ms/tok DFA</small>
            </button>
            <button
              id="showcase-ngram"
              type="button"
              className="btn-showcase"
              disabled={status !== 'ready'}
              onClick={() => runShowcase('ngram')}
            >
              N-gram Speculative
              <small>0 MiB VRAM draft</small>
            </button>
          </div>
        </div>

        <div className="card">
          <div className="card-title">{t.telemetrySection}</div>
          <div className="stat-grid">
            <div className="stat-box"><div className="k">Decode</div><div className="v" id="stat-decode">{stats.decodeSpeed}</div></div>
            <div className="stat-box"><div className="k">Prefill</div><div className="v">{stats.prefillSpeed}</div></div>
            <div className="stat-box"><div className="k">Init</div><div className="v">{stats.initMs}</div></div>
            <div className="stat-box"><div className="k">Main Heap</div><div className="v">{stats.heapMB}</div></div>
            <div className="stat-box"><div className="k">KV Ctx</div><div className="v">{stats.ctxPos} / {stats.maxCtx}</div></div>
            <div className="stat-box"><div className="k">OPFS</div><div className="v" style={{ fontSize: '0.73rem' }}>{stats.cacheMode}</div></div>
          </div>
        </div>
      </aside>

      <main className="main-pane">
        <header className="topbar">
          <div className="pill">
            <span className={`dot ${dotClass}`} />
            <span id="status-text">{statusText}</span>
          </div>
          <div className="topbar-actions">
            <span className="pill">{gpuCaps}</span>
            <button
              id="btn-lang-toggle"
              type="button"
              className="btn btn-secondary btn-xs"
              onClick={() => setLang((prev) => (prev === 'en' ? 'zh' : 'en'))}
            >
              {lang === 'en' ? '中文' : 'EN'}
            </button>
            <button
              id="btn-toggle-hero"
              type="button"
              className="btn btn-secondary btn-xs"
              onClick={() => setShowHero((prev) => !prev)}
            >
              {showHero ? t.hideHero : t.showHero}
            </button>
            <button
              className="btn btn-secondary"
              disabled={status !== 'ready' || messages.length === 0}
              onClick={resetChat}
            >
              {t.resetChat}
            </button>
          </div>
        </header>

        <div className="chat-scroll">
          {showHero && (
            <section className="hero-banner" id="hero-banner">
              <div className="hero-head">
                <div>
                  <h2>{t.heroTitle}</h2>
                  <p className="hero-sub">{t.heroDesc}</p>
                  <div className="hero-links">
                    <a href="https://github.com/Ray-D-Song/Ignis" target="_blank" rel="noreferrer">GitHub (Apache-2.0) ↗</a>
                    <a href="https://www.npmjs.com/package/ignis-webgpu" target="_blank" rel="noreferrer">npm: ignis-webgpu ↗</a>
                    <a href="https://www.npmjs.com/package/ignis-webgpu-packer" target="_blank" rel="noreferrer">npm: ignis-webgpu-packer ↗</a>
                  </div>
                </div>
              </div>
              <div className="bench-strip">
                <div className="bench-item">
                  <div className="bk">Spark-X2.6-4B (RX 6800 XT)</div>
                  <div className="bv">85.4 tok/s · 134 tok/s N-gram</div>
                </div>
                <div className="bench-item">
                  <div className="bk">LFM3-Omni-2B Full Omni</div>
                  <div className="bv">1.05 GiB · 15/15 PyTorch Parity</div>
                </div>
                <div className="bench-item">
                  <div className="bk">Gemma-4-E4B (262K + PLE)</div>
                  <div className="bv">2.69 GiB PLE Streamed via OPFS</div>
                </div>
                <div className="bench-item">
                  <div className="bk">WASM JSON Schema FSM</div>
                  <div className="bv">0.285 ms/tok · 6/6 Strict Match</div>
                </div>
              </div>
            </section>
          )}

          {messages.length === 0 ? (
            <div className="empty-state">
              <h2>{t.starterTitle}</h2>
              <p>{t.starterBody}</p>
              <div className="chips">
                {t.starterPrompts.map((p) => (
                  <button
                    key={p}
                    className="chip"
                    disabled={status !== 'ready'}
                    onClick={() => sendMessage(p)}
                  >
                    {p}
                  </button>
                ))}
              </div>
            </div>
          ) : (
            messages.map((m, idx) => {
              const parsed = m.role === 'assistant' ? splitThinking(m.content) : null;
              const isActivelyThinking = Boolean(
                parsed?.isThinking && status === 'generating' && idx === messages.length - 1,
              );
              return (
                <div key={m.id} className={`msg ${m.role}`}>
                  <div className="msg-role">
                    <span>{m.role === 'user' ? 'You' : m.modelLabel || 'Ignis'}</span>
                    {m.meta && <span>· {m.meta}</span>}
                  </div>
                  <div className="bubble">
                    {m.role === 'user' ? (
                      <>
                        {(m.image || m.audio) && (
                          <div className="msg-media">
                            {m.image && <img src={m.image.previewUrl} alt={m.image.name} />}
                            {m.audio && (
                              <span className="attach-chip">
                                Audio: {m.audio.name} ({m.audio.durationSec.toFixed(1)}s)
                              </span>
                            )}
                          </div>
                        )}
                        {m.content}
                      </>
                    ) : (
                      <>
                        {parsed?.think && (
                          <details className="think-box" open={isActivelyThinking}>
                            <summary>
                              {isActivelyThinking ? t.thinkOpen : parsed?.isThinking ? t.thinkAborted : t.thinkView}
                            </summary>
                            <pre>{parsed.think}</pre>
                          </details>
                        )}
                        <div className="answer-body">
                          {parsed?.answer || (m.audioUrl ? t.audioGenerated : isActivelyThinking ? '' : '...')}
                        </div>
                        {(m.fsmValid !== undefined || m.specProposed !== undefined) && (
                          <div className="msg-badges">
                            {m.fsmValid !== undefined && (
                              <span className={`feature-badge ${m.fsmValid ? 'ok' : 'err'}`} data-testid="fsm-badge">
                                FSM: {m.fsmValid ? 'JSON.parse ✓' : 'JSON Invalid ✗'}
                                {m.fsmMedianMs !== null && m.fsmMedianMs !== undefined
                                  ? ` · ${m.fsmMedianMs.toFixed(3)} ms/tok`
                                  : ''}
                              </span>
                            )}
                            {m.specProposed !== undefined && (
                              <span className="feature-badge info" data-testid="ngram-badge">
                                N-gram speculative: {m.specAccepted ?? 0}/{m.specProposed} accepted (
                                {m.specProposed > 0 ? Math.round(((m.specAccepted ?? 0) / m.specProposed) * 100) : 0}%)
                              </span>
                            )}
                          </div>
                        )}
                        {m.audioUrl && (
                          <audio controls src={m.audioUrl} style={{ marginTop: 10, width: '100%' }} />
                        )}
                      </>
                    )}
                  </div>
                </div>
              );
            })
          )}
          <div ref={scrollEndRef} />
        </div>

        <div className="composer-wrap">
          {(pendingImage || pendingAudio) && (
            <div className="attach-row">
              {pendingImage && (
                <span className="attach-chip">
                  <img src={pendingImage.previewUrl} alt={pendingImage.name} />
                  <span>{pendingImage.name} ({pendingImage.width}x{pendingImage.height})</span>
                  <button type="button" onClick={() => setPendingImage(null)} aria-label="Remove image">✕</button>
                </span>
              )}
              {pendingAudio && (
                <span className="attach-chip">
                  <span>WAV: {pendingAudio.name} ({pendingAudio.durationSec.toFixed(1)}s)</span>
                  <button type="button" onClick={() => setPendingAudio(null)} aria-label="Remove audio">✕</button>
                </span>
              )}
            </div>
          )}
          <div className="composer">
            {activeModel?.capabilities.vision && (
              <>
                <input
                  ref={imgInputRef}
                  type="file"
                  accept="image/*"
                  style={{ display: 'none' }}
                  onChange={handleImageUpload}
                />
                <button
                  type="button"
                  className="btn-icon"
                  disabled={status !== 'ready'}
                  onClick={() => imgInputRef.current?.click()}
                >
                  {lang === 'zh' ? '图片' : 'Image'}
                </button>
              </>
            )}
            {activeModel?.capabilities.audioIn && (
              <>
                <input
                  ref={audInputRef}
                  type="file"
                  accept="audio/wav,.wav"
                  style={{ display: 'none' }}
                  onChange={handleAudioUpload}
                />
                <button
                  type="button"
                  className="btn-icon"
                  disabled={status !== 'ready'}
                  onClick={() => audInputRef.current?.click()}
                >
                  WAV
                </button>
              </>
            )}
            <textarea
              id="chat-input"
              value={input}
              placeholder={status === 'ready' ? t.inputReady : t.inputWait}
              disabled={status !== 'ready' && status !== 'generating'}
              onChange={(e) => setInput(e.target.value)}
              onKeyDown={(e) => {
                if (e.key === 'Enter' && !e.shiftKey) {
                  e.preventDefault();
                  if (status === 'ready') sendMessage();
                }
              }}
            />
            {status === 'generating' ? (
              <button id="btn-stop" className="btn btn-danger" onClick={stopGeneration}>{t.stop}</button>
            ) : (
              <button
                id="btn-send"
                className="btn btn-primary"
                disabled={status !== 'ready' || (!input.trim() && !pendingImage && !pendingAudio)}
                onClick={() => sendMessage()}
              >
                {t.send}
              </button>
            )}
          </div>
        </div>
      </main>
    </div>
  );
}
