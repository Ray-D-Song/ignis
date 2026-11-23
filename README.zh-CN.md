> [English](README.md) | **简体中文**

<p align="center">
  <img src="assets/logo.svg" alt="Ignis Logo" width="72" height="72" />
</p>

<h1 align="center">Ignis</h1>

<p align="center">
  <strong>Ignis</strong> 是一个零依赖的 <strong>Browser-Native WebGPU LLM & Full-Omni 多模态推理引擎与通用框架</strong>。
</p>

它不依赖 GGUF/ONNX/TVM 等外部推理运行时，从 `.ignis` 分片存储格式、W3C 基线 WGSL 算子代码生成、OPFS 流式底座到 Rust/WASM 分词与 JSON Schema FSM 约束解码全部面向浏览器沙箱垂直设计。

- **仓库**：https://github.com/Ray-D-Song/Ignis
- **在线演示**：https://ignis.ray-d-song.com
- **npm 包**：[`ignis-webgpu`](packages/core/package.json) · [`ignis-webgpu-packer`](packages/packer/package.json)

## 真机实测指标摘要（Thorium 138 · AMD RX 6800 XT）

| 指标（Qwen3-1.7B · 4-bit · `ctx=4096`） | **Ignis** (`qwen3-1.7b-ignis-q4`) | **WebLLM** (`Qwen3-1.7B-q4f16_1-MLC`) | 对比 |
| :--- | :--- | :--- | :--- |
| **GPU 权重分片 / 显存形态** | `923.40 MiB` GPU（`+299.07 MiB` OPFS 外置 Embedding） | `923.16 MiB` `.bin`（预估 `2036.66 MB` VRAM） | GPU 权重持平，Ignis 词表常驻 OPFS |
| **本地缓存占用**（`navigator.storage`） | `1293.38 MB`（OPFS） | `986.19 MB`（Cache API） | +307.2 MB 换取更低显存占用 |
| **冷启动 / 热启动初始化** | `6432.4 ms` / `2865.9 ms` | `6416.8 ms` / `2865.6 ms` | 基本持平 |
| **首次热身推理（TTFT / Decode）** | **`79.6 ms`** / **`90.81 tok/s`** | `16061.0 ms` / `30.51 tok/s` | 无首跑着色器卡顿 |
| **短 Prompt Decode**（`~128 tok`，3 次均值） | **`90.30 tok/s`** | `53.94 tok/s` | **1.67×** |
| **短 Prompt Prefill**（3 次均值） | **`841.0 tok/s`** | `481.7 tok/s` | **1.75×** |
| **中等上下文 Prefill / TTFT**（`~450 tok`，2 次均值） | **`918.0 tok/s`** / **`483.7 ms`** | `593.1 tok/s` / `767.2 ms` | **1.55×** / **TTFT -37.0%** |
| **中等上下文 Decode**（2 次均值） | **`92.11 tok/s`** | `51.38 tok/s` | **1.79×** |
| **较长文本持续生成**（`~256 tok`，2 次均值） | **`96.84 tok/s`** | `52.83 tok/s` | **1.83×** |
| **高重复文本生成**（N-gram `k=5` vs 常规解码） | **`180.82 tok/s`**（`94/125` 命中） | `53.67 tok/s`（常规自回归） | **3.37×** |

## 为什么选择 Ignis？

* **多架构统一 Declarative Blueprint 引擎**：
  * 原生一套引擎支持 **Spark-X2.6-4B**（Ring SWA + Full + Sigmoid Gate）、**Llama / Qwen2 / Qwen3 / Phi3**、**LiquidAI/LFM3-Omni-2B**（ShortConv `K=7` + SWA + Full + 图像/音频理解 + DepthDecoder + 24 kHz LiquidCodec 语音合成）与 **google/gemma-4-E4B-it**（`262K` 词表 Z-Tiling + 42 层 KV-Sharing + 外置 `INT8_ROW` PLE + SigLIP2 + Chunked Conformer）。
* **WebGPU 原生对齐格式与低显存分级降级**：
  * 支持 `INT4_G64_ASYM`、`INT8_G64_SYM`（LM Head / PLE 投影）及 **`INT3_G64_ASYM`**（32×3-bit 打入 3×`u32`，LFM3-Omni 显存压缩至 `755 MiB`）；超大词表 `token_embd` 与 Gemma 4 `ple_embd` 常驻 OPFS 外置查表（节省数 GiB VRAM）。
  * 严格遵循 **W3C WebGPU 基线**：所有 `kernels/` 禁用实验性 `@builtin(subgroup_id)` / `@builtin(num_subgroups)`，内置 `useSubgroups=false` 共享内存归约降级（Bit-Exact 对齐）与 `detectDeviceProfile()` 硬件分级适配。
* **Rust/WASM 分词与结构化 JSON Schema FSM 约束解码**：
  * 自研零外部 JS 依赖 Rust/WASM 引擎（`tokenizer/`），同时支持 Byte-Level BPE 与 SentencePiece `ByteFallback`（131K / 128K / 262K 词表 **100% 精确一致**）。
  * 内置 JSON Schema → Thompson NFA → 等价类 DFA → 状态级 packed `u32` Bitset **FSM 约束解码**（`response_format`），零拷贝屏蔽 131K+ 词表仅 **`0.285 ms/tok`**（`p95 = 2.05 ms`）。
* **双模式无损投机解码（Speculative Decoding）**：
  * **N-gram Lookup**（零额外显存后缀匹配，`Spark-4B` 实测 `84.5%` 接受率、`134.0 tok/s` / `1.30x` 加速）。
  * **Draft-Target**（`INT3_G64` 草稿 + `INT4/INT8` 目标模型，支持 ShortConv 零读取 GPU 状态快照回滚与 Leviathan 无损随机拒绝采样，与自回归基线 **100% 比特级一致**）。
* **开发者 SDK 与全栈示例**：
  * `ignis-webgpu`（`IgnisOpenAI` OpenAI 兼容文本/流式/全模态客户端）与 `ignis-webgpu-packer`（`ignis-pack` 离线打包 CLI）。
  * `examples/chat-vinext/`：基于 `vinext` + React 19 RSC 的多模型全模态浏览器工作台（OPFS 模型缓存、`<think>` 折叠、图像/WAV 上传、JSON Schema FSM、N-gram 投机解码与 24 kHz 语音播放，支持一键切换编码 Agent 沙盒）。

## Ignis 与 WebLLM 对比

| 对比维度 | **Ignis** | **WebLLM（MLC-LLM / TVM）** |
| :--- | :--- | :--- |
| **运行时与构建链路** | 零外部运行时依赖的 JS + 纯 W3C WGSL Mega-Kernels + 单个 `1.3 MiB` Rust/WASM；纯 Python CLI（`ignis-pack`）分钟级打包 HuggingFace Safetensors，无需编译 TVM/C++ | 通过 Apache TVM 编译模型计算图与 Emscripten C++/WASM 运行时（新增自定义模型需完整 `mlc_llm` 编译链） |
| **新增模型架构难度** | 声明式 JSON 蓝图 + WGSL 代码生成；ShortConv、KV 共享、PLE 等混合架构共用同一套引擎运行 | 需编写 TVM Relax IR 定义并按架构/上下文规格预编译独立 `.wasm` 算子库 |
| **超大词表与显存占用** | 通过 Worker OPFS `SyncAccessHandle` 外置读取 `embd.ignis` 与 Gemma 4 的 `2.82 GB` `ple_embd.ignis`（`~0.1 ms` 行读取），节省数 GiB 显存 | 模型权重与 Embedding 表通常通过 Cache API / IndexedDB 加载进 WebGPU 显存缓冲区 |
| **Full-Omni 全模态能力** | 原生统一支持视觉、音频输入及 24 kHz `LiquidCodec` 神经网络语音合成（`LFM3-Omni-2B`、`Gemma-4-E4B-it`） | 主打文本与图像理解（`Phi-3.5-vision`、`Qwen2-VL`、`Llama-3.2-Vision`），不含原生端到端语音合成管线 |
| **结构化 JSON Schema 约束** | 自研 Rust/WASM Thompson NFA → DFA → packed `u32` Bitset FSM，131K+ 词表中位屏蔽仅 `0.285 ms/tok` | 集成 WASM XGrammar 引擎实现 CFG / JSON Schema 结构化输出 |
| **投机解码（Speculative）** | 原生内置 **N-gram Lookup**（`0 MiB` 额外显存，`Spark-4B` `1.30x`）与 GPU **Draft-Target**（`INT3` → `INT4`/`INT8`） | 主要聚焦标准自回归解码 |
| **量化格式与驱动兼容性** | `INT4_G64` + `INT8_G64` LM Head + `INT3_G64` 硬件降级（LFM3-2B 仅 `755 MiB`）；全内核禁用 `@builtin(subgroup_id)` 并提供共享内存无误差回退 | 提供丰富的 `q4f16_1` / `q4f32_1` / `q0f16` 量化预设，Chrome 插件与 Service Worker 生态非常成熟 |
| **生态与预构建模型库** | 聚焦高效参考架构（`Spark`、`LFM3-Omni`、`Gemma 4`、`Llama`/`Qwen`/`Phi`）与轻量化自有打包工具 | 拥有广泛的 MLC Hub 预编译模型仓库（`Llama`、`Qwen`、`Mistral`、`Phi`、`SmolLM`、`Gemma 2`）与长期社区沉淀 |

### Ignis 的核心优势与什么时候应该采用 Ignis

当你的浏览器端应用命中以下任意场景时，推荐采用 **Ignis**：

- **显存预算紧张或超大词表模型部署**：利用 OPFS Worker 同步行读取外置 Embedding / PLE 表（如 Gemma 4 高达 `2.82 GB` 的 `ple_embd.ignis`）并配合 `INT3_G64` 量化降级（将 `LFM3-Omni-2B` GPU 权重压至 `755.1 MiB`），在消费级笔记本与浏览器显存受限环境下也能跑起多模态大模型。
- **需要原生 Full-Omni 视觉、语音输入与语音播报**：同一运行时内置图像理解、音频理解与 `24 kHz` `LiquidCodec` 神经网络语音合成，适合实时语音助手与多模态交互。
- **构建浏览器原生 Agent、结构化工具调用与代码/JSON 编辑**：自研 Rust/WASM JSON Schema FSM 约束解码（中位 `0.285 ms/tok`）可 100% 保证合法 JSON 工具输出，叠加零额外显存的 **N-gram**（`Spark-4B` `134.0 tok/s` / `1.30x`）与 **Draft-Target** 无损投机解码显著提升重复代码或结构化文本生成速度。
- **不想维护 TVM/C++ 编译链、需要快速接入自有或新架构模型**：纯 Python CLI（`ignis-webgpu-packer`）分钟级打包 HuggingFace Safetensors，一套声明式引擎直接支持 Ring SWA、ShortConv、KV 共享、PLE 等混合架构，打包静态前端资源零外部运行时依赖。
- **需要更高的跨驱动 / 跨浏览器 WebGPU 稳定性**：严格遵循 W3C WebGPU 基线（禁用高风险的 `@builtin(subgroup_id)` 假设），并提供比特级精确的 `useSubgroups = false` 共享内存回退路径。

> 如果你的核心诉求是直接复用 MLC Hub 现成的海量预编译纯文本/常见 VLM 模型，或依赖成熟 Chrome 扩展 / Service Worker 模板，WebLLM 会是更省心的选择。

## 快速开始与自动化测试

```bash
# 安装测试依赖（Playwright）与构建 Rust/WASM 引擎
npm install
npm run build:wasm

# 离线打包 HuggingFace 模型为 .ignis（支持 Spark / Qwen / Llama / LFM3-Omni / Gemma-4）
npx ignis-webgpu-packer --input <hf-model-dir> --output <ignis-out-dir> --lm-head-dtype int8

# 运行 CI 无头 WGSL/Tokenizer 审计与全套回归测试
npm run audit:wgsl
npm run pack:check
npm test
npm run test:lfm3
npm run test:gemma4
npm run test:sdk
npm run test:phase6

# 启动 vinext 全模态 Chat Demo
npm run demo:chat
npm run test:vinext
```

贡献指南见 [CONTRIBUTING.md](CONTRIBUTING.md)，安全政策见 [SECURITY.md](SECURITY.md)，开源许可见 [LICENSE](LICENSE)。
