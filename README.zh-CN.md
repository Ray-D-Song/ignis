> [English](README.md) | **简体中文**

<p align="center">
  <img src="assets/logo.svg" alt="Ignis Logo" width="72" height="72" />
</p>

<h1 align="center">Ignis</h1>

<p align="center">
  零依赖的 <strong>Browser-Native WebGPU LLM &amp; Full-Omni 多模态推理引擎</strong>，不依赖 GGUF / ONNX / TVM。
</p>

- **仓库**：https://github.com/Ray-D-Song/Ignis
- **在线演示**：https://ignis.ray-d-song.com
- **npm 包**：[`ignis-webgpu`](packages/core/package.json) · [`ignis-webgpu-packer`](packages/packer/package.json)

## 真机实测指标摘要（Thorium 138 · AMD RX 6800 XT）

| 指标（Qwen3-1.7B · 4-bit · `ctx=4096`） | **Ignis** (`qwen3-1.7b-ignis-q4`) | **WebLLM** (`Qwen3-1.7B-q4f16_1-MLC`) | 对比 |
| :--- | :--- | :--- | :--- |
| **GPU 权重 / 显存** | `923.40 MiB`（`+299.07 MiB` OPFS 词表） | `923.16 MiB`（预估 `2036.66 MB` VRAM） | 词表外置 OPFS |
| **本地缓存占用** | `1293.38 MB`（OPFS） | `986.19 MB`（Cache API） | +307.2 MB 换显存 |
| **冷 / 热启动初始化** | `6432.4 ms` / `2865.9 ms` | `6416.8 ms` / `2865.6 ms` | 持平 |
| **首次热身（TTFT / Decode）** | **`79.6 ms`** / **`90.81 tok/s`** | `16061.0 ms` / `30.51 tok/s` | 无首跑编译卡顿 |
| **短 Prompt Decode**（`~128 tok`，3 次均值） | **`90.30 tok/s`** | `53.94 tok/s` | **1.67×** |
| **短 Prompt Prefill**（3 次均值） | **`841.0 tok/s`** | `481.7 tok/s` | **1.75×** |
| **中等上下文 Prefill / TTFT**（`~450 tok`） | **`918.0 tok/s`** / **`483.7 ms`** | `593.1 tok/s` / `767.2 ms` | **1.55×** / **-37.0% TTFT** |
| **中等上下文 Decode**（2 次均值） | **`92.11 tok/s`** | `51.38 tok/s` | **1.79×** |
| **较长文本 Decode**（`~256 tok`，2 次均值） | **`96.84 tok/s`** | `52.83 tok/s` | **1.83×** |
| **高重复文本**（N-gram `k=5` vs 常规） | **`180.82 tok/s`**（`94/125` 命中） | `53.67 tok/s` | **3.37×** |

## 为什么选择 Ignis？

- **一套引擎跑全模态与混合架构**：声明式蓝图统一支持 `Spark`、`Llama`、`Qwen2/3`、`Phi3`、`LFM3-Omni`（含 24 kHz 语音合成）与 `Gemma-4-E4B-it`（视觉 + 音频 + PLE）。
- **低显存外置存储与硬件降级**：OPFS `SyncAccessHandle` 按行流式读取词表与 PLE（`~0.1 ms`），支持 `INT4` / `INT8` / `INT3`（LFM3-2B 压至 `755 MiB`）；全 WGSL 严格遵循 W3C 基线并内置无 Subgroup 回退。
- **自研 Rust/WASM 分词与结构化输出**：单个 `1.3 MiB` WASM 支持 BPE / SentencePiece 与 JSON Schema FSM 约束解码（中位屏蔽开销 `0.285 ms/tok`）。
- **内置无损投机解码**：支持零额外显存 **N-gram Lookup** 与 **Draft-Target**（`INT3 → INT4/INT8`），与贪婪解码比特级一致。
- **开箱即用 SDK 与工作台**：提供 OpenAI 兼容客户端 `ignis-webgpu`、纯 Python 打包工具 `ignis-webgpu-packer` 与 `examples/chat-vinext` 全模态工作台。

## Ignis 与 WebLLM 对比

| 维度 | **Ignis** | **WebLLM（MLC-LLM / TVM）** |
| :--- | :--- | :--- |
| **构建与接入** | 纯 JS + WGSL + 单 Rust/WASM；Python CLI 分钟级打包 Safetensors | 依赖 Apache TVM + Emscripten 编译模型与独立 `.wasm` 算子库 |
| **显存与存储** | OPFS 外置词表与 Gemma 4 `2.82 GB` PLE，显著降低 VRAM | 权重与词表经 Cache API / IndexedDB 加载至 GPU 显存 |
| **全模态能力** | 原生支持图像、音频输入与 24 kHz 神经网络语音合成 | 主打文本与常见 VLM，不含原生语音合成管线 |
| **约束与投机解码** | 内置 Rust/WASM JSON Schema FSM + N-gram / Draft-Target 投机解码 | 集成 XGrammar 结构化输出；聚焦常规自回归解码 |
| **驱动兼容性** | 禁用 `@builtin(subgroup_id)`，提供共享内存精确回退与 `INT3` 降级 | 丰富 `q4f16_1` 等 TVM 预设，Chrome 插件 / SW 生态成熟 |
| **生态定位** | 轻量化自有打包与前沿混合架构（`Spark`、`LFM3-Omni`、`Gemma 4`、`Qwen`） | 拥有广泛的 MLC Hub 预编译模型仓库与社区沉淀 |

### 什么时候选择 Ignis？

- **显存吃紧或超大词表**：需要 OPFS 外置词表/PLE 与 `INT3` 降级在消费级设备运行大模型。
- **全模态语音交互**：需要在单 Worker 内同时完成文本、图像、音频理解与实时语音播报。
- **浏览器 Agent / 代码与 JSON 生成**：需要低延迟 JSON Schema 约束解码 + N-gram 投机加速。
- **快速接入自定义模型**：不想维护 TVM/C++ 编译链，希望纯 Python 打包、零运行时依赖部署。

> 若只需直接复用 MLC Hub 现成预编译纯文本/常见 VLM 模型，或依赖成熟 Chrome 扩展模板，选 **WebLLM** 更省心。

## 快速开始

```bash
npm install && npm run build:wasm

# 打包 HuggingFace 模型为 .ignis
npx ignis-webgpu-packer --input <hf-model-dir> --output <ignis-out-dir> --lm-head-dtype int8

# 审计与回归测试
npm run audit:wgsl && npm run pack:check && npm test

# 启动全模态 Chat Demo
npm run demo:chat
```

详见 [CONTRIBUTING.md](CONTRIBUTING.md) · [SECURITY.md](SECURITY.md) · [LICENSE](LICENSE)。
