> **English** | [简体中文](README.zh-CN.md)

<p align="center">
  <img src="assets/logo.svg" alt="Ignis Logo" width="72" height="72" />
</p>

<h1 align="center">Ignis</h1>

<p align="center">
  <strong>A zero-dependency, browser-native WebGPU LLM and Full-Omni multimodal inference engine.</strong>
</p>

<p align="center">
  <a href="https://ignis.ray-d-song.com"><strong>Live Demo</strong></a> ·
  <a href="packages/core/README.md"><strong>ignis-webgpu</strong></a> ·
  <a href="packages/packer/README.md"><strong>ignis-webgpu-packer</strong></a> ·
  <a href="CONTRIBUTING.md"><strong>Contributing</strong></a>
</p>

Ignis does not wrap ONNX Runtime, llama.cpp/WASM, or Apache TVM. Every layer—from its 256-byte-aligned `.ignis` binary shards and OPFS `SyncAccessHandle` streaming loader, to strict W3C baseline WGSL kernels, Rust/WASM tokenizers, JSON Schema FSM logit masking, and GPU-resident speculative decoding—is engineered specifically around browser sandbox constraints (`ArrayBuffer` caps, COOP/COEP isolation, WGSL subgroup divergence, and VRAM limits).

## Verified Benchmarks (Thorium 138 · AMD Radeon RX 6800 XT)

Every number below is measured on real desktop hardware in Thorium 138 (`shader-f16` + `subgroups`) against PyTorch BF16 reference outputs:

| Model / Capability | GPU VRAM (Loaded) | Throughput / Accuracy |
| :--- | :--- | :--- |
| **Spark-X2.6-4B** (`INT4_G64` body, `INT8_G64` LM Head) | `2,371.34 MiB` (`+322 MiB` OPFS) | **`85.44 tok/s`** raw / **`86.0 -> 81.8 tok/s`** multi-turn chat (`cos = 0.9990–0.9997` vs BF16) |
| **LiquidAI/LFM3-Omni-2B** (Text + Vision + Audio + 24 kHz Voice) | `1,054.86 MiB` (`+251.95 MiB` OPFS) | **15/15** PyTorch parity checks (`Text cos=0.9981`, `VQA 0.9981`, `Audio 0.9990`, `24 kHz WAV 0.9990`) |
| **google/gemma-4-E4B-it** (`262K` Vocab + Q8 PLE + SigLIP2 + Conformer) | `2,588.7 MiB` base / `3,075.8 MiB` peak | **10/10** PyTorch parity checks (`Text cos=0.9988`, `VQA 0.9934`, `Audio 0.9590`, Top-1 match) |
| **`INT3_G64` Degradation** (`LFM3-Omni-2B` body) | **`755.10 MiB`** | `useSubgroups=false` cosine `0.99999999996`; Q3 greedy token IDs match Q4 100% |
| **JSON Schema FSM** (`131K / 128K` vocabs × 3 schemas) | `+0 MiB` VRAM | **6/6** strictly valid JSON; **`0.285 ms/tok`** median mask time (`n = 162`) |
| **Speculative Decode** (`N-gram k=5` / `INT3->INT4` Draft-Target) | `0 MiB` (N-gram) | Spark N-gram **`134.0 tok/s` (`1.30x`)**; LFM3 Draft-Target `90.2%` accept (`4.36 tok/round`) |

## Why Ignis?

- **OPFS Externalized Embeddings & 2.8 GiB PLE Streaming**: Instead of pinning massive `vocab_size × hidden_dim` embedding matrices in VRAM, `embd.ignis` and Gemma 4's `2.82 GB` `ple_embd.ignis` live in Origin Private File System and stream row-by-row via Worker `createSyncAccessHandle()` (`~0.1 ms` per token lookup, saving gigabytes of GPU memory).
- **One Engine, Full-Omni Native Pipelines**: Declarative architecture blueprints (`packer/blueprints.py` + `kernels/codegen.mjs` + `runtime/ignis-engine.mjs`) run standard Transformers (`Spark-X2.6-4B`, `Llama`, `Qwen2/3`, `Phi3`), Hybrid ShortConv+SWA Audio/Vision Omni (`LiquidAI/LFM3-Omni-2B` with 24 kHz `LiquidCodec` speech synthesis), and `262K`-vocab Per-Layer-Embedding Omni (`google/gemma-4-E4B-it` with SigLIP2 Vision + Chunked Conformer Audio) without per-model JS rewrites.
- **Strict W3C WebGPU Baseline + Verified Tiered Degradation**: Zero experimental `@builtin(subgroup_id)` or `@builtin(num_subgroups)` assumptions (which break Chrome/Edge even when `subgroups` is advertised). Includes bit-accurate `useSubgroups = false` shared-memory tree reduction (`cos = 0.99999999996`) and **`INT3_G64_ASYM`** quantization (packing `32 × 3-bit` weights into `3 × u32`, shrinking `LFM3-Omni-2B` GPU weights to **`755.1 MiB`**).
- **Sub-Millisecond Rust/WASM Tokenizer & JSON Schema FSM**: Single zero-JS-dependency `ignis_tokenizer.wasm` supports Byte-Level BPE (`131K` Spark) and SentencePiece `ByteFallback` (`128K` LFM3, `262K` Gemma-4) with 100% Oracle parity, plus a compiled NFA -> DFA -> packed `u32` bitset FSM (`response_format`) masking `131K+` logits in **`0.285 ms/tok` median** (`p95 = 2.05 ms/tok`).
- **Dual-Mode Lossless Speculative Decoding**:
  - **N-gram Lookup** (0 MiB extra VRAM): `84.5%` acceptance and **`134.0 tok/s` (`1.30x`)** on Spark-X2.6-4B.
  - **Draft-Target** (`INT3_G64` draft -> `INT4/INT8` target): Zero-readback GPU `ShortConv` state snapshot/rollback + Leviathan lossless rejection sampling, bit-identical to greedy autoregressive decoding (`90.2%` acceptance on LFM3-Omni-2B).

## Ignis vs. WebLLM

| Dimension | **Ignis** | **WebLLM (MLC-LLM / TVM)** |
| :--- | :--- | :--- |
| **Runtime & Build Pipeline** | Zero-dependency JS + pure W3C WGSL mega-kernels + single Rust/WASM (`1.3 MiB`); pack Safetensors in minutes via pure Python (`ignis-pack`) without compiling TVM/C++ | Compiles models through Apache TVM + C++/Emscripten WASM runtime (`mlc_llm` toolchain required per compiled model library) |
| **Adding Architectures** | Declarative JSON blueprint + WGSL codegen; hybrid ShortConv, KV-sharing, and PLE architectures run on one shared engine | Requires TVM Relax IR graph compilation & compiled `.wasm` shader libraries per architecture/context configuration |
| **OPFS Externalized Embeddings & VRAM** | Externalizes `embd.ignis` & Gemma 4's `2.82 GB` `ple_embd.ignis` to OPFS (`SyncAccessHandle` row reads, `~0.1 ms`), saving GiBs of VRAM on large-vocab models | Weights and embedding tables are loaded into WebGPU GPU buffers via Cache API / IndexedDB chunks |
| **Full-Omni Multimodal & Audio Output** | Native vision + audio-in + 24 kHz `LiquidCodec` neural audio synthesis (`LFM3-Omni-2B`, `Gemma-4-E4B-it`) | Primarily text & vision-language (`Phi-3.5-vision`, `Qwen2-VL`, `Llama-3.2-Vision`); no native neural speech synthesis |
| **Structured JSON Schema FSM** | Built-in Rust/WASM Thompson NFA -> DFA -> packed `u32` bitset masking across `131K+` tokens (`0.285 ms/tok` median) | Employs WASM XGrammar engine for CFG / JSON schema constrained decoding |
| **Speculative Decoding** | Built-in **N-gram Lookup** (`0 MiB` extra VRAM, `1.30x` on Spark-4B) + GPU **Draft-Target** (`INT3` -> `INT4`/`INT8`) | Focused on standard autoregressive decoding |
| **Quantization & WGSL Portability** | `INT4_G64` + `INT8_G64` LM Head + `INT3_G64` degradation (`755 MiB` LFM3-2B); zero `@builtin(subgroup_id)` with verified shared-memory fallback | Broad `q4f16_1` / `q4f32_1` / `q0f16` TVM quantization presets; mature Chrome Extension & Service Worker adapters |
| **Ecosystem & Model Catalog** | Focused reference blueprints (`Spark`, `LFM3-Omni`, `Gemma 4`, `Llama`/`Qwen`/`Phi`) & customizable `ignis-webgpu-packer` | Wide pre-built MLC Hub catalog (`Llama`, `Qwen`, `Mistral`, `Phi`, `SmolLM`, `Gemma 2`) and longstanding community adoption |

### When to Choose Ignis

Choose **Ignis** when your browser application needs one or more of the following advantages:

- **Tight GPU VRAM budgets or huge vocabularies**: OPFS-externalized embedding & PLE tables (`2.82 GB` `ple_embd.ignis` for Gemma 4) plus `INT3_G64` degradation (`755.1 MiB` GPU weights for `LFM3-Omni-2B`) run multi-billion-parameter models in browser GPUs where pinning full tables in VRAM is tight or fails.
- **Full-Omni speech & multimodal interaction**: Native single-worker support for image inputs, audio inputs, and `24 kHz` neural voice output (`LiquidCodec`) alongside text.
- **Browser-native Agents, structured outputs, or code/JSON editing**: Built-in Rust/WASM JSON Schema FSM decoding (`0.285 ms/tok` median) guarantees schema-valid tool calls, while zero-VRAM **N-gram** (`134.0 tok/s` / `1.30x` on Spark-4B) and **Draft-Target** speculative decoding accelerate repetitive edits without sacrificing lossless greedy parity.
- **Fast iteration on custom HuggingFace checkpoints without TVM C++ builds**: Pack models to 256B-aligned `.ignis` shards in minutes with pure Python (`ignis-webgpu-packer`) and run hybrid architectures (Ring SWA, ShortConv, KV-Sharing, PLE) via declarative blueprints.
- **Broad WebGPU driver portability**: Strict W3C WGSL baseline avoids experimental `@builtin(subgroup_id)` assumptions and includes a bit-accurate shared-memory fallback (`useSubgroups = false`).

> If you only need standard text/vision chat from MLC's existing pre-compiled Hub catalog or mature Chrome Extension service-worker templates, **WebLLM** is a strong, battle-tested choice.

## Architecture

## Quickstart

### 1. Run the OpenAI-Compatible WebGPU SDK (`ignis-webgpu`)

```bash
npm install ignis-webgpu
```

```js
import { IgnisOpenAI } from 'ignis-webgpu';

// Spawns a Dedicated Worker automatically with OPFS shard caching
const client = await IgnisOpenAI.create({
  modelBaseUrl: 'https://your-cdn.example.com/lfm3-omni-2b-ignis/',
  modelId: 'lfm3-omni-2b',
});

const stream = await client.chat.completions.create({
  model: 'lfm3-omni-2b',
  stream: true,
  response_format: {
    type: 'json_schema',
    json_schema: {
      name: 'summary',
      strict: true,
      schema: {
        type: 'object',
        properties: { answer: { type: 'string' }, confidence: { type: 'number' } },
        required: ['answer', 'confidence'],
        additionalProperties: false,
      },
    },
  },
  messages: [{ role: 'user', content: 'Summarize WebGPU OPFS streaming.' }],
});

for await (const chunk of stream) {
  process.stdout.write(chunk.choices[0]?.delta?.content ?? '');
}
```

### 2. Pack Any Supported HuggingFace Model (`ignis-webgpu-packer`)

```bash
# Install Python dependencies once (numpy, safetensors, torch, transformers)
npx ignis-webgpu-packer --help

# Quantize to 256B-aligned .ignis shards + OPFS embd.ignis (INT8 LM Head preserves coding/CJK quality)
npx ignis-webgpu-packer \
  --input ./path/to/hf-model \
  --output ./path/to/model-ignis \
  --lm-head-dtype int8 \
  --weight-bits 4
```

### 3. Run Locally from Source & Verify

```bash
git clone https://github.com/Ray-D-Song/Ignis.git
cd Ignis
npm install
npm run audit:wgsl     # Zero-GPU W3C WGSL baseline + Rust/WASM FSM audit
npm run pack:check     # Isolated npm tarball + CLI smoke check
npm run demo:chat      # Start examples/chat-vinext locally at http://localhost:5173
```

## Honest Caveats & Browser Limits

- **COOP / COEP Headers Required for Peak Mode**: High-throughput OPFS `SyncAccessHandle` works inside Dedicated Workers regardless, and cross-origin model buckets on Cloudflare R2 require `Access-Control-Allow-Origin: *` plus `Cross-Origin-Resource-Policy: cross-origin` when the app sets `Cross-Origin-Embedder-Policy: credentialless`.
- **Incognito OPFS Quota**: Chromium caps OPFS storage at `2.00 GB` in Incognito/private windows. Models whose quantized weights + `embd.ignis` + `ple_embd.ignis` exceed 2 GB (such as Gemma-4-E4B-it at `~6.5 GB` total shards) require a normal browser profile (and Playwright tests must use `launchPersistentContext`).
- **Desktop WebGPU Hardware**: Verified primarily on Chromium 138+ / Edge 157+ desktop GPUs (`shader-f16`). Devices without `shader-f16` or with `< 2 GiB` `maxBufferSize` should load smaller checkpoints (`LFM3-Omni-2B INT3_G64` at `755 MiB` GPU weights) or check `detectDeviceProfile()`.
- **Pre-Compiled Binary Not Checked In Without Source**: All Rust/WASM tokenizer sources live in `tokenizer/`, with reproducible builds via `npm run build:wasm`.

## License & Attribution

- Ignis engine, kernels, packer, tokenizer, SDK, and demo UI are licensed under the **Apache License 2.0** ([LICENSE](LICENSE)).
- Contribution guidelines and upstream attribution notes live in [CONTRIBUTING.md](CONTRIBUTING.md).
- Pre-quantized demo weights remain governed by their upstream model licenses (`Apache-2.0` for `XHToken/Spark-X2.6-4B`, `LFM Open License v1.0` for `LiquidAI/LFM3-Omni-2B`, and `Gemma Terms of Use` for `google/gemma-4-E4B-it`).
