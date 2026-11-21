'use client';

import React, { useEffect, useRef, useState } from 'react';
import type { ModelPreset } from '@/lib/model-paths';
import { IgnisClient } from '../../../runtime/ignis-client.mjs';
import { decodeWav, resampleLinear } from '../../../runtime/multimodal/audio-dsp.mjs';

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

type ArchKind = 'spark' | 'lfm3' | 'gemma4';
type OmniGenMode = 'text' | 'interleaved' | 'audio';

const STARTER_PROMPTS = [
  '请用通俗且严谨的语言解释什么是第一性原理。',
  'Write a Python function to find the longest palindromic substring with O(n^2) time and O(1) space.',
  '一辆高铁以 280 km/h 行驶 45 分钟，然后以 160 km/h 行驶 30 分钟，总路程和平均速度各是多少？',
  '比较 WebGPU 与 WebAssembly 在纯浏览器端运行大语言模型时的本质瓶颈差异。',
];

function getArchKind(model?: ModelPreset): ArchKind {
  if (!model) return 'spark';
  if (model.topModelType === 'lfm3_omni' || model.architecture === 'lfm3') return 'lfm3';
  if (model.topModelType === 'gemma4' || model.architecture === 'gemma4_text') return 'gemma4';
  return 'spark';
}

function getReadyLabel(model?: ModelPreset): string {
  const headLabel = model?.lmHeadDtype === 'u32_q8' ? 'INT8_G64 LM Head' : 'INT4_G64 LM Head';
  return getArchKind(model) === 'spark' ? `就绪 · ${headLabel}` : `就绪 · ${headLabel} · Full Omni`;
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

export function ChatWorkspace({ models }: { models: ModelPreset[] }) {
  const defaultModel = models[0]?.id || 'spark-4b-q8head';
  const [selectedModelId, setSelectedModelId] = useState(defaultModel);
  const [status, setStatus] = useState<'idle' | 'loading' | 'ready' | 'generating' | 'error'>('idle');
  const [statusText, setStatusText] = useState('待加载模型（首次将流式缓存至浏览器 OPFS）');
  const [loadPct, setLoadPct] = useState(0);

  const [enableThinking, setEnableThinking] = useState(true);
  const [effort, setEffort] = useState<'low' | 'high'>('low');
  const [generationMode, setGenerationMode] = useState<OmniGenMode>('text');
  const [temperature, setTemperature] = useState(0.6);
  const [maxNewTokens, setMaxNewTokens] = useState(1536);

  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [input, setInput] = useState('');
  const [pendingImage, setPendingImage] = useState<PendingImage | null>(null);
  const [pendingAudio, setPendingAudio] = useState<PendingAudio | null>(null);

  const [gpuCaps, setGpuCaps] = useState('检测中...');
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

  const activeModel = models.find((m) => m.id === selectedModelId) || models[0];
  const archKind = getArchKind(activeModel);

  const revokeAllObjectUrls = () => {
    for (const u of objectUrlsRef.current) URL.revokeObjectURL(u);
    objectUrlsRef.current = [];
  };

  useEffect(() => {
    (async () => {
      if (!('gpu' in navigator)) {
        setGpuCaps('当前浏览器不支持 WebGPU');
        return;
      }
      const adapter = await navigator.gpu.requestAdapter({ powerPreference: 'high-performance' });
      if (!adapter) {
        setGpuCaps('未检测到可用 WebGPU Adapter');
        return;
      }
      const f16 = adapter.features.has('shader-f16');
      const sg = adapter.features.has('subgroups');
      const iso = window.crossOriginIsolated ? 'COOP/COEP' : 'No-COI';
      setGpuCaps(`WebGPU · f16:${f16 ? '✓' : '✗'} · subgroups:${sg ? '✓' : '✗'} · ${iso}`);
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
      setStatusText('已切换模型，请点击左侧「加载模型到 WebGPU」');
    }
  };

  const handleImageUpload = async (ev: React.ChangeEvent<HTMLInputElement>) => {
    const file = ev.target.files?.[0];
    ev.target.value = '';
    if (!file) return;
    const bmp = await createImageBitmap(file);
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
      const previewUrl = URL.createObjectURL(file);
      objectUrlsRef.current.push(previewUrl);
      setPendingImage({ data: rgba, width: w, height: h, channels: 4, previewUrl, name: file.name });
    } finally {
      bmp.close();
    }
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

  const loadModel = async () => {
    if (!activeModel) return;
    setStatus('loading');
    setLoadPct(0);
    setStatusText(`正在初始化 ${activeModel.label}...`);
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
        setStatusText(`${p.cached ? 'OPFS 命中' : '流式写入 OPFS'}: ${p.file} (${pct}%)`);
      };
      client.onDeviceLost = (reason: string) => {
        setStatus('error');
        setStatusText(`GPUDevice 丢失: ${reason}`);
      };

      const maxSeq = archKind === 'spark' ? 8192 : 4096;
      const info = await client.init({
        modelBaseUrl: `/api/models/${activeModel.id}`,
        modelKey: `vinext-${activeModel.id}`,
        kernelBaseUrl: '/ignis/kernels',
        wasmUrl: '/ignis/runtime/wasm/ignis_tokenizer.wasm',
        maxSeqLen: maxSeq,
        maxBatchTokens: 64,
      });

      loadedModelIdRef.current = activeModel.id;
      kvTurnsRef.current = 0;
      kvThinkingKeyRef.current = '';
      const perfMem = (performance as Performance & { memory?: { usedJSHeapSize: number } }).memory;
      setStats((s) => ({
        ...s,
        initMs: `${(info.initMs / 1000).toFixed(2)}s`,
        cacheMode: info.cacheMisses === 0 ? `OPFS 热命中 (${info.cacheHits})` : `冷写入 (${info.cacheMisses}片)`,
        ctxPos: 0,
        maxCtx: maxSeq,
        heapMB: perfMem ? `${(perfMem.usedJSHeapSize / (1024 * 1024)).toFixed(1)} MB` : '-',
      }));
      setLoadPct(100);
      setStatus('ready');
      setStatusText(getReadyLabel(activeModel));
    } catch (err) {
      setStatus('error');
      setStatusText(`加载失败: ${err instanceof Error ? err.message : String(err)}`);
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

  const sendMessage = async (overrideText?: string) => {
    const rawText = (overrideText ?? input).trim();
    const imgAttach = overrideText ? null : pendingImage;
    const audAttach = overrideText ? null : pendingAudio;
    const text = rawText || (imgAttach || audAttach ? '请分析并回答相关内容。' : '');
    if (!text || !clientRef.current || status !== 'ready') return;

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
    const thinkPrefix = enableThinking ? (archKind === 'gemma4' ? '<|thought|>' : '<think>') : '';
    const assistantMsg: ChatMessage = {
      id: assistantId,
      role: 'assistant',
      content: thinkPrefix,
      modelLabel: shortModelName,
    };

    const nextMessages = [...messages, userMsg];
    setMessages([...nextMessages, assistantMsg]);
    setStatus('generating');
    setStatusText('Prefill 上下文中...');

    const client = clientRef.current;
    const thinkingKey = `${archKind}:${enableThinking}:${effort}`;
    let promptDeltaText = '';
    const reqImages: Array<{ data: Uint8Array; width: number; height: number; channels: number }> = [];
    const reqAudios: Float32Array[] = [];

    if (archKind === 'spark') {
      const canIncremental = kvTurnsRef.current === messages.length && kvThinkingKeyRef.current === thinkingKey && kvTurnsRef.current > 0;
      if (canIncremental) {
        promptDeltaText = `<｜end▁of▁sentence｜><｜start▁of▁sentence｜><|User|>${text}<｜end▁of▁sentence｜><｜start▁of▁sentence｜><|Bot|>${enableThinking ? '<think>' : '</think>'}`;
      } else {
        await client.reset();
        let fullPrompt = buildSparkSystemPrefix(enableThinking, effort);
        for (const m of nextMessages) {
          if (m.role === 'user') {
            fullPrompt += `<｜start▁of▁sentence｜><|User|>${m.content}<｜end▁of▁sentence｜>`;
          } else {
            const parsed = splitThinking(m.content);
            fullPrompt += `<｜start▁of▁sentence｜><|Bot|></think>${parsed.answer}<｜end▁of▁sentence｜>`;
          }
        }
        fullPrompt += `<｜start▁of▁sentence｜><|Bot|>${enableThinking ? '<think>' : '</think>'}`;
        promptDeltaText = fullPrompt;
      }
    } else if (archKind === 'lfm3') {
      await client.reset();
      let fullPrompt = '<|startoftext|>';
      for (const m of nextMessages) {
        if (m.role === 'user') {
          let mmTags = '';
          if (m.image) {
            reqImages.push({ data: m.image.data, width: m.image.width, height: m.image.height, channels: m.image.channels });
            mmTags += '<|vision_start|><|image|><|vision_end|>';
          }
          if (m.audio) {
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
      // gemma4
      await client.reset();
      let fullPrompt = '<bos>';
      for (const m of nextMessages) {
        if (m.role === 'user') {
          let mmTags = '';
          if (m.image) {
            reqImages.push({ data: m.image.data, width: m.image.width, height: m.image.height, channels: m.image.channels });
            mmTags += '<|image|>';
          }
          if (m.audio) {
            reqAudios.push(m.audio.pcm16k);
            mmTags += '<|audio|>';
          }
          fullPrompt += `<|turn>user\n${mmTags}${m.content}<turn|>\n`;
        } else {
          const parsed = splitThinking(m.content);
          fullPrompt += `<|turn>model\n${parsed.answer}<turn|>\n`;
        }
      }
      fullPrompt += enableThinking ? '<|turn>model\n<|thought|>\n' : '<|turn>model\n';
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
      setStatusText(`生成中 · ${liveTps.toFixed(1)} tok/s`);
    };

    try {
      const enc = await client.encode(promptDeltaText);
      const maxThinkingTokens = archKind === 'spark' && enableThinking
        ? (effort === 'low' ? Math.min(384, Math.floor(maxNewTokens * 0.45)) : Math.floor(maxNewTokens * 0.65))
        : 0;
      const activeMode: OmniGenMode = activeModel?.capabilities.audioOut ? generationMode : 'text';
      const res = await client.generate({
        promptIds: enc.ids,
        images: reqImages,
        audios: reqAudios,
        generationMode: activeMode,
        maxNewTokens,
        maxThinkingTokens,
        chunkSize: 64,
        sampling: {
          temperature,
          topK: 40,
          topP: 0.95,
          repetitionPenalty: archKind === 'spark' ? 1.05 : 1.0,
        },
        stopOnEos: true,
      });

      let audioUrl: string | undefined;
      if (res.wavBytes && res.wavBytes.byteLength > 44) {
        audioUrl = URL.createObjectURL(new Blob([new Uint8Array(res.wavBytes)], { type: 'audio/wav' }));
        objectUrlsRef.current.push(audioUrl);
      }

      const finalRaw = thinkPrefix ? `${thinkPrefix}${res.generatedText || ''}` : (res.generatedText || '');
      const meta = `${res.generatedIds.length} tok · ${res.decodeTokPerSec.toFixed(1)} tok/s · Prefill ${res.prefillMs.toFixed(0)}ms${audioUrl ? ' · 24kHz WAV' : ''}${res.aborted ? ' · 已中止' : ''}`;
      setMessages((prev) => prev.map((m) => (m.id === assistantId ? { ...m, content: finalRaw, audioUrl, meta } : m)));

      if (archKind === 'spark' && !res.aborted) {
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
      setStatusText(getReadyLabel(activeModel));
    } catch (err) {
      setStatus('error');
      setStatusText(`生成异常: ${err instanceof Error ? err.message : String(err)}`);
    }
  };

  const dotClass = status === 'ready' ? 'ok' : (status === 'loading' || status === 'generating') ? 'busy' : status === 'error' ? 'err' : '';

  return (
    <div className="app-shell">
      <aside className="sidebar">
        <div className="brand">
          <div className="brand-badge">I</div>
          <div>
            <h1>Ignis Chat</h1>
            <span>vinext · WebGPU · Full Omni</span>
          </div>
        </div>

        <div className="card">
          <div className="card-title">模型配置</div>
          <label className="field">
            选择已打包权重
            <select
              value={selectedModelId}
              disabled={status === 'loading' || status === 'generating'}
              onChange={(e) => handleModelChange(e.target.value)}
            >
              {models.map((m) => (
                <option key={m.id} value={m.id}>{m.label}</option>
              ))}
            </select>
          </label>
          {activeModel && (
            <div style={{ fontSize: '0.77rem', color: 'var(--muted)', lineHeight: 1.45 }}>
              {activeModel.description}<br />
              VRAM: <b>{activeModel.gpuMiB} MiB</b> · Embd: <b>{activeModel.embdMiB} MiB</b>
              {activeModel.pleMiB > 0 && <> · PLE: <b>{(activeModel.pleMiB / 1024).toFixed(2)} GiB</b></>}
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
            disabled={!activeModel || status === 'loading' || status === 'generating'}
            onClick={loadModel}
          >
            {status === 'ready' ? '重新加载模型' : status === 'loading' ? `加载中 ${loadPct}%` : '加载模型到 WebGPU'}
          </button>
        </div>

        <div className="card">
          <div className="card-title">推理参数</div>
          {activeModel?.capabilities.thinking && (
            <div className="toggle-row">
              <span>深度思考</span>
              <input
                type="checkbox"
                checked={enableThinking}
                disabled={status === 'generating'}
                onChange={(e) => setEnableThinking(e.target.checked)}
              />
            </div>
          )}
          {archKind === 'spark' && enableThinking && (
            <label className="field">
              思考预算 (thinking_effort)
              <select
                value={effort}
                disabled={status === 'generating'}
                onChange={(e) => setEffort(e.target.value as 'low' | 'high')}
              >
                <option value="low">low（快速推理 · 思考 ≤384 tok）</option>
                <option value="high">high（深度思考 · 思考 ≤65% 预算）</option>
              </select>
            </label>
          )}
          {activeModel?.capabilities.audioOut && (
            <label className="field">
              LFM3 输出模态
              <select
                value={generationMode}
                disabled={status === 'generating'}
                onChange={(e) => setGenerationMode(e.target.value as OmniGenMode)}
              >
                <option value="text">纯文本 (text)</option>
                <option value="interleaved">图文音混合 + 24kHz 语音 (interleaved)</option>
                <option value="audio">纯语音输出 24kHz WAV (audio)</option>
              </select>
            </label>
          )}
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
          <div className="card-title">实时硬件遥测</div>
          <div className="stat-grid">
            <div className="stat-box"><div className="k">Decode 速度</div><div className="v" id="stat-decode">{stats.decodeSpeed}</div></div>
            <div className="stat-box"><div className="k">Prefill 速度</div><div className="v">{stats.prefillSpeed}</div></div>
            <div className="stat-box"><div className="k">初始化 / 缓存</div><div className="v">{stats.initMs}</div></div>
            <div className="stat-box"><div className="k">主线程 Heap</div><div className="v">{stats.heapMB}</div></div>
            <div className="stat-box"><div className="k">KV 上下文</div><div className="v">{stats.ctxPos} / {stats.maxCtx}</div></div>
            <div className="stat-box"><div className="k">OPFS 状态</div><div className="v" style={{ fontSize: '0.75rem' }}>{stats.cacheMode}</div></div>
          </div>
        </div>
      </aside>

      <main className="main-pane">
        <header className="topbar">
          <div className="pill">
            <span className={`dot ${dotClass}`} />
            <span id="status-text">{statusText}</span>
          </div>
          <div style={{ display: 'flex', gap: 8, alignItems: 'center' }}>
            <span className="pill">{gpuCaps}</span>
            <button
              className="btn btn-secondary"
              disabled={status !== 'ready' || messages.length === 0}
              onClick={resetChat}
            >
              清空对话
            </button>
          </div>
        </header>

        <div className="chat-scroll">
          {messages.length === 0 ? (
            <div className="empty-state">
              <h2>Ignis WebGPU · 纯浏览器多架构全模态推理</h2>
              <p>
                零后端、零外部运行时：支持 <b>Spark-X2.6-4B</b>（<code>INT8_G64</code> 输出头 + Ring SWA）、
                <b>LFM3-Omni-2B</b>（ShortConv 混合架构 + 视觉/音频理解 + 24 kHz 语音合成）与
                <b>Gemma-4-E4B-it</b>（PLE + KV-Sharing + 262K 词表 + SigLIP2/Conformer）。
                点击左侧「加载模型到 WebGPU」即可体验本地毫秒级推理。
              </p>
              <div className="chips">
                {STARTER_PROMPTS.map((p) => (
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
                parsed?.isThinking && status === 'generating' && idx === messages.length - 1
              );
              return (
                <div key={m.id} className={`msg ${m.role}`}>
                  <div className="msg-role">
                    <span>{m.role === 'user' ? 'You' : (m.modelLabel || 'Ignis')}</span>
                    {m.meta && <span>· {m.meta}</span>}
                  </div>
                  <div className="bubble">
                    {m.role === 'user' ? (
                      <>
                        {(m.image || m.audio) && (
                          <div className="msg-media">
                            {m.image && <img src={m.image.previewUrl} alt={m.image.name} />}
                            {m.audio && <span className="attach-chip">音频: {m.audio.name} ({m.audio.durationSec.toFixed(1)}s)</span>}
                          </div>
                        )}
                        {m.content}
                      </>
                    ) : (
                      <>
                        {parsed?.think && (
                          <details className="think-box" open={isActivelyThinking}>
                            <summary>
                              {isActivelyThinking
                                ? '正在思考...'
                                : parsed?.isThinking
                                  ? '思考过程（已中止或达上限）'
                                  : '查看思考过程'}
                            </summary>
                            <pre>{parsed.think}</pre>
                          </details>
                        )}
                        <div className="answer-body">
                          {parsed?.answer || (m.audioUrl ? '（已生成 24 kHz 语音回复）' : (isActivelyThinking ? '' : '...'))}
                        </div>
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
                  <button type="button" onClick={() => setPendingImage(null)} aria-label="移除图片">✕</button>
                </span>
              )}
              {pendingAudio && (
                <span className="attach-chip">
                  <span>WAV: {pendingAudio.name} ({pendingAudio.durationSec.toFixed(1)}s)</span>
                  <button type="button" onClick={() => setPendingAudio(null)} aria-label="移除音频">✕</button>
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
                  图片
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
              placeholder={status === 'ready' ? '输入消息（Enter 发送，Shift+Enter 换行）...' : '请先在左侧点击「加载模型到 WebGPU」'}
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
              <button id="btn-stop" className="btn btn-danger" onClick={stopGeneration}>停止</button>
            ) : (
              <button
                id="btn-send"
                className="btn btn-primary"
                disabled={status !== 'ready' || (!input.trim() && !pendingImage && !pendingAudio)}
                onClick={() => sendMessage()}
              >
                发送
              </button>
            )}
          </div>
        </div>
      </main>
    </div>
  );
}
