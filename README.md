# Ignis

**Ignis** 是一个基于第一性原理从零构建的 **Browser-Native WebGPU LLM 推理引擎与通用框架**。  
它不依赖 GGUF/ONNX/TVM 等外部推理运行时，从存储格式、内存重排、WGSL 融合算子到 OPFS 流式底座全部面向浏览器沙箱和 WebGPU 物理特性垂直设计。

## 核心特性

* **WebGPU 原生对齐格式（`.ignis` v1）**：
  * 线性层采用 `INT4_G64_ASYM`（8×4bit 打入 `u32` + `f16x2` scale/min 分离存储），LM Head 支持 `INT4_G64` / `INT8_G64` 可选配置，零分支解包。
  * **大词表 Embedding 外置**：`token_embd` 转成 `INT8_ROW` 存放于 `embd.ignis`，通过 Worker OPFS `SyncAccessHandle` + LRU 按行查表，不占 GPU 显存（Spark-X2.6-4B 节省 ~670 MiB）。
* **混合注意力 KV-Cache 分治**：
  * 原生支持 `Sliding Window Attention`（固定环形 Ring Buffer，窗口外不增长）与 `Full Attention` 交替架构，以及双基频部分旋转 RoPE、逐头 `sigmoid(attn_gate)` 门控。
* **高性能 WGSL 计算管线**：
  * 强制启用 `shader-f16` 与 `subgroups`（`subgroupAdd` 归约兼容 Wave32/Wave64）。
  * Split-KV GQA FlashDecoding（`attention_decode.wgsl`）、原地残差融合、轻量 WGSL 模板代码生成（`kernels/codegen.mjs` 支持 `GeGLU` / `SwiGLU`）与静态预绑定单次提交 Decode。
* **浏览器零卡顿工程底座**：
  * 全链路 Dedicated Web Worker 运行，OPFS 持久化缓存 + 16 MiB 环形 `StagingBuffer` 流式入显存（热启动主线程 JS Heap ~1.1 MB），`Chunked Prefill` 保障主页面 60fps 零掉帧。
* **极简 Rust → WASM Tokenizer**：
  * 自研零 JS 依赖 Byte-Level BPE 分词器（`tokenizer/`），在 131K 词表上与 `llama-tokenize` 100% 对齐。

## 真实 Spark-X2.6-4B 实测摘要

在 Windows 11 / Edge `157.0` / AMD RX 6800 XT 上对 `XHToken/Spark-X2.6-4B`（4.45B 参数）实测（详见 [docs/benchmark-spark-4b.md](docs/benchmark-spark-4b.md)）：
* **显存占用**：`2.16 GiB`（`INT4` LM Head） / `2.32 GiB`（`INT8` LM Head）
* **Decode 速度**：**`85.4 tok/s`（Wall-Clock） / `127.2 tok/s`（GPU `timestamp-query`）**
* **OPFS 热启动**：`2.14 s` 就绪，主线程 JS Heap `1.11 MB`，`maxFrameMs = 18.01 ms`
* **精度对齐**：Logits 余弦相似度 vs BF16 达 **`0.9990 ~ 0.9997`**，Top-1 命中率 **100%**

## 快速开始与自动化测试

```bash
# 安装测试依赖（Playwright + Edge 驱动）
npm install

# 离线打包 HuggingFace Safetensors 模型为 .ignis v1
packer/.venv/Scripts/python.exe packer/ignis_pack.py \
  --input <hf-model-dir> --output <ignis-out-dir> --lm-head-dtype int8

# 运行全套自动化验证（硬件探针 + WGSL 算子单测 + 131K 分词器 + 逐层数值对齐 + OPFS Worker E2E + 多架构 SwiGLU）
npm test

# 运行 36 层 4B 尺寸吞吐与 60fps Chunked Prefill 基准压测
npm run test:phase4
```

详细路线图见 [ROADMAP.md](ROADMAP.md)，二进制格式规范见 [docs/format-spec.md](docs/format-spec.md)。
