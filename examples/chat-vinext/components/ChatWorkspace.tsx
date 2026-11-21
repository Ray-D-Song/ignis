'use client';

import React, { useEffect, useRef, useState } from 'react';
import type { ModelPreset } from '@/lib/model-paths';
import { IgnisClient } from '../../../runtime/ignis-client.mjs';

interface ChatMessage {
  id: string;
  role: 'user' | 'assistant';
  content: string;
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

const STARTER_PROMPTS = [
  '请用通俗且严谨的语言解释什么是第一性原理。',
  'Write a Python function to find the longest palindromic substring with O(n^2) time and O(1) space.',
  '一辆高铁以 280 km/h 行驶 45 分钟，然后以 160 km/h 行驶 30 分钟，总路程和平均速度各是多少？',
  '比较 WebGPU 与 WebAssembly 在纯浏览器端运行大语言模型时的本质瓶颈差异。',
];

function splitThinking(raw: string): { think: string; answer: string; isThinking: boolean } {
  const startTag = '<think>';
  const endTag = '</think>';
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
  return { think: '', answer: raw, isThinking: false };
}

function buildSystemPrefix(enableThinking: boolean, effort: 'low' | 'high'): string {
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
  const [temperature, setTemperature] = useState(0.6);
  const [maxNewTokens, setMaxNewTokens] = useState(1024);

  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [input, setInput] = useState('');
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
  const kvTurnsRef = useRef<number>(0);
  const kvThinkingKeyRef = useRef<string>('');
  const scrollEndRef = useRef<HTMLDivElement | null>(null);

  const activeModel = models.find((m) => m.id === selectedModelId) || models[0];

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
      if (clientRef.current) {
        clientRef.current.destroy();
        clientRef.current = null;
      }
    };
  }, []);

  useEffect(() => {
    scrollEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages]);

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

      const info = await client.init({
        modelBaseUrl: `/api/models/${activeModel.id}`,
        modelKey: `vinext-${activeModel.id}`,
        kernelBaseUrl: '/ignis/kernels',
        wasmUrl: '/ignis/runtime/wasm/ignis_tokenizer.wasm',
        maxSeqLen: 8192,
        maxBatchTokens: 64,
      });

      kvTurnsRef.current = 0;
      kvThinkingKeyRef.current = '';
      const perfMem = (performance as Performance & { memory?: { usedJSHeapSize: number } }).memory;
      setStats((s) => ({
        ...s,
        initMs: `${(info.initMs / 1000).toFixed(2)}s`,
        cacheMode: info.cacheMisses === 0 ? `OPFS 热命中 (${info.cacheHits})` : `冷写入 (${info.cacheMisses}片)`,
        ctxPos: 0,
        heapMB: perfMem ? `${(perfMem.usedJSHeapSize / (1024 * 1024)).toFixed(1)} MB` : '-',
      }));
      setLoadPct(100);
      setStatus('ready');
      setStatusText(`就绪 · ${activeModel.lmHeadDtype === 'u32_q8' ? 'INT8_G64 LM Head' : 'INT4_G64 LM Head'}`);
    } catch (err) {
      setStatus('error');
      setStatusText(`加载失败: ${err instanceof Error ? err.message : String(err)}`);
    }
  };

  const resetChat = async () => {
    if (clientRef.current && status === 'ready') {
      await clientRef.current.reset();
    }
    kvTurnsRef.current = 0;
    kvThinkingKeyRef.current = '';
    setMessages([]);
    setStats((s) => ({ ...s, ctxPos: 0 }));
  };

  const stopGeneration = async () => {
    if (clientRef.current && status === 'generating') {
      await clientRef.current.abort();
    }
  };

  const sendMessage = async (overrideText?: string) => {
    const text = (overrideText ?? input).trim();
    if (!text || !clientRef.current || status !== 'ready') return;
    if (!overrideText) setInput('');

    const userMsg: ChatMessage = { id: `u-${Date.now()}`, role: 'user', content: text };
    const assistantId = `a-${Date.now()}`;
    const assistantMsg: ChatMessage = { id: assistantId, role: 'assistant', content: enableThinking ? '<think>' : '' };

    const nextMessages = [...messages, userMsg];
    setMessages([...nextMessages, assistantMsg]);
    setStatus('generating');
    setStatusText('Prefill 上下文中...');

    const client = clientRef.current;
    const thinkingKey = `${enableThinking}:${effort}`;
    const canIncremental = kvTurnsRef.current === messages.length && kvThinkingKeyRef.current === thinkingKey && kvTurnsRef.current > 0;

    let promptDeltaText = '';
    if (canIncremental) {
      // Incremental KV reuse: continue right after previous turn's assistant output
      promptDeltaText = `<｜end▁of▁sentence｜><｜start▁of▁sentence｜><|User|>${text}<｜end▁of▁sentence｜><｜start▁of▁sentence｜><|Bot|>${enableThinking ? '<think>' : '</think>'}`;
    } else {
      await client.reset();
      let fullPrompt = buildSystemPrefix(enableThinking, effort);
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

    client.onPrefillProgress = (p: { processed: number; total: number }) => {
      setStatusText(`Prefill ${p.processed}/${p.total} tokens...`);
    };

    let liveTps = 0;
    client.onToken = (ev: IgnisTokenEvent) => {
      if (ev.tokPerSec) liveTps = ev.tokPerSec;
      const body = ev.fullText !== undefined ? ev.fullText : ev.text;
      const rendered = enableThinking ? `<think>${body}` : body;
      setMessages((prev) => prev.map((m) => (m.id === assistantId ? { ...m, content: rendered } : m)));
      setStatusText(`生成中 · ${liveTps.toFixed(1)} tok/s`);
    };

    try {
      const enc = await client.encode(promptDeltaText);
      const res = await client.generate({
        promptIds: enc.ids,
        maxNewTokens,
        chunkSize: 64,
        sampling: { temperature, topK: 40, topP: 0.95, repetitionPenalty: 1.05 },
        stopOnEos: true,
      });

      const finalRaw = enableThinking ? `<think>${res.generatedText || ''}` : (res.generatedText || '');
      const meta = `${res.generatedIds.length} tok · ${res.decodeTokPerSec.toFixed(1)} tok/s · Prefill ${res.prefillMs.toFixed(0)}ms${res.aborted ? ' · 已中止' : ''}`;
      setMessages((prev) => prev.map((m) => (m.id === assistantId ? { ...m, content: finalRaw, meta } : m)));

      if (res.aborted) {
        kvTurnsRef.current = -1; // force clean re-prefill on next turn after abort
      } else {
        kvTurnsRef.current = nextMessages.length + 1;
        kvThinkingKeyRef.current = thinkingKey;
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
      setStatusText(`就绪 · ${activeModel?.lmHeadDtype === 'u32_q8' ? 'INT8_G64 LM Head' : 'INT4_G64 LM Head'}`);
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
            <span>vinext · WebGPU · 零后端</span>
          </div>
        </div>

        <div className="card">
          <div className="card-title">模型配置</div>
          <label className="field">
            选择已打包权重
            <select
              value={selectedModelId}
              disabled={status === 'loading' || status === 'generating'}
              onChange={(e) => setSelectedModelId(e.target.value)}
            >
              {models.map((m) => (
                <option key={m.id} value={m.id}>{m.label}</option>
              ))}
            </select>
          </label>
          {activeModel && (
            <div style={{ fontSize: '0.77rem', color: 'var(--muted)', lineHeight: 1.45 }}>
              {activeModel.description}<br />
              VRAM: <b>{activeModel.gpuMiB} MiB</b> · OPFS Embedding: <b>{activeModel.embdMiB} MiB</b>
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
          <div className="toggle-row">
            <span>深度思考 (&lt;think&gt;)</span>
            <input
              type="checkbox"
              checked={enableThinking}
              disabled={status === 'generating'}
              onChange={(e) => setEnableThinking(e.target.checked)}
            />
          </div>
          {enableThinking && (
            <label className="field">
              思考预算 (thinking_effort)
              <select
                value={effort}
                disabled={status === 'generating'}
                onChange={(e) => setEffort(e.target.value as 'low' | 'high')}
              >
                <option value="low">low（快速推理）</option>
                <option value="high">high（深度思考）</option>
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
              min="64"
              max="2048"
              step="64"
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
              <h2>Spark-X2.6-4B · 纯浏览器实时推理</h2>
              <p>
                本 Demo 运行在 <b>vinext</b> + <b>Ignis WebGPU 引擎</b> 上：36 层权重使用 <code>INT4_G64</code>，
                131K 输出头使用 <code>INT8_G64</code>，词嵌入常驻 OPFS 外置查表，27 层滑动窗口使用固定环形 KV-Cache。
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
            messages.map((m) => {
              const parsed = m.role === 'assistant' ? splitThinking(m.content) : null;
              return (
                <div key={m.id} className={`msg ${m.role}`}>
                  <div className="msg-role">
                    <span>{m.role === 'user' ? 'You' : 'Spark-X2.6-4B (Ignis)'}</span>
                    {m.meta && <span>· {m.meta}</span>}
                  </div>
                  <div className="bubble">
                    {m.role === 'user' ? (
                      m.content
                    ) : (
                      <>
                        {parsed?.think && (
                          <details className="think-box" open={parsed.isThinking}>
                            <summary>{parsed.isThinking ? '正在思考...' : '查看思考过程'}</summary>
                            <pre>{parsed.think}</pre>
                          </details>
                        )}
                        <div>{parsed?.answer || (parsed?.isThinking ? '' : '...')}</div>
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
          <div className="composer">
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
                disabled={status !== 'ready' || !input.trim()}
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
