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

| 模型 / 能力 | 显存 / 耗时 | 精度 / 速度实测结果 |
| :--- | :--- | :--- |
| **Spark-X2.6-4B** (`INT8_G64` LM Head) | `2.32 GiB` VRAM | Wall **`85.4 tok/s`**（`vinext` 多轮 `86.0 / 81.8 tok/s`），vs BF16 `cos=0.9990~0.9997` |
| **LFM3-Omni-2B** (Full Omni) | `1054.86 MiB` GPU | 15/15 全模态对拍全过；Text `cos=0.9981` / VQA `cos=0.9981` / Audio `cos=0.9990` / 24kHz WAV `cos=0.9990` |
| **Gemma-4-E4B-it** (Full Omni + Q8 PLE) | `2588.7 MiB` Base / `3075.8 MiB` Peak | 10/10 全模态对拍全过；Text `cos=0.9988` / VQA `cos=0.9934` / Audio `cos=0.9590`（`Top-1` 全中） |
| **`INT3_G64` 硬件降级 (`LFM3-2B`)** | **`755.10 MiB`** GPU | `useSubgroups=false` 余弦 `0.99999999996`；Q3 Greedy Token IDs 与 Q4 100% 一致 |
| **Rust/WASM JSON Schema FSM** | **`0.285 ms/tok`** 中位掩码开销 | Spark-4B + LFM3 × 3 类复杂 Schema **6/6 100% 合法严格匹配** |
| **双模式投机解码 (`N-gram` + `Draft-Target`)** | Zero-VRAM / Dual-Engine | Greedy **100% 比特级等价**；Spark N-gram **`134.0 tok/s`（`1.30x`）**；LFM3 ShortConv 接受率 **`90.2%`** |

## 核心能力全景

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
