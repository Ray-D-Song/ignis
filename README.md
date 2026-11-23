> **English** | [简体中文](README.zh-CN.md)

<p align="center">
  <img src="assets/logo.svg" alt="Ignis Logo" width="72" height="72" />
</p>

<h1 align="center">Ignis</h1>

<p align="center">
  <strong>Zero-dependency, browser-native WebGPU LLM &amp; Full-Omni multimodal inference engine</strong>—no ONNX, llama.cpp/WASM, or TVM.
</p>

<p align="center">
  <a href="https://ignis.ray-d-song.com"><strong>Live Demo</strong></a> ·
  <a href="packages/core/README.md"><strong>ignis-webgpu</strong></a> ·
  <a href="packages/packer/README.md"><strong>ignis-webgpu-packer</strong></a> ·
  <a href="CONTRIBUTING.md"><strong>Contributing</strong></a>
</p>

## Verified Benchmarks (Thorium 138 · AMD Radeon RX 6800 XT)

| Metric (Qwen3-1.7B · 4-bit · `ctx=4096`) | **Ignis** (`qwen3-1.7b-ignis-q4`) | **WebLLM** (`Qwen3-1.7B-q4f16_1-MLC`) | Delta |
| :--- | :--- | :--- | :--- |
| **GPU Weights / Memory** | `923.40 MiB` (`+299.07 MiB` OPFS embd) | `923.16 MiB` (`2036.66 MB` est. VRAM) | Embd kept on OPFS |
| **Local Storage Usage** | `1293.38 MB` (OPFS) | `986.19 MB` (Cache API) | +307.2 MB disk for lower VRAM |
| **Cold / Warm Init** | `6432.4 ms` / `2865.9 ms` | `6416.8 ms` / `2865.6 ms` | Parity |
| **First-Run Warmup (TTFT / Decode)** | **`79.6 ms`** / **`90.81 tok/s`** | `16061.0 ms` / `30.51 tok/s` | No first-query shader stall |
| **Short Prompt Decode** (`~128 tok`, 3-run avg) | **`90.30 tok/s`** | `53.94 tok/s` | **1.67×** |
| **Short Prompt Prefill** (3-run avg) | **`841.0 tok/s`** | `481.7 tok/s` | **1.75×** |
| **Medium Context Prefill / TTFT** (`~450 tok`) | **`918.0 tok/s`** / **`483.7 ms`** | `593.1 tok/s` / `767.2 ms` | **1.55×** / **-37.0% TTFT** |
| **Medium Context Decode** (2-run avg) | **`92.11 tok/s`** | `51.38 tok/s` | **1.79×** |
| **Sustained Decode** (`~256 tok`, 2-run avg) | **`96.84 tok/s`** | `52.83 tok/s` | **1.83×** |
| **Repetitive Text** (N-gram `k=5` vs standard) | **`180.82 tok/s`** (`94/125` accepted) | `53.67 tok/s` | **3.37×** |

## Why Ignis?

- **One Engine, Full-Omni & Hybrid Architectures**: Declarative blueprints run `Spark`, `Llama`, `Qwen2/3`, `Phi3`, `LFM3-Omni` (vision + audio + 24 kHz speech synthesis), and `Gemma-4-E4B-it` (`262K` vocab + PLE + SigLIP2 + Conformer) with no per-model JS rewrites.
- **OPFS Externalized Tables & Tiered Degradation**: Streams `embd.ignis` and Gemma 4's `2.82 GB` `ple_embd.ignis` row-by-row via Worker `SyncAccessHandle` (`~0.1 ms`), supports `INT4` / `INT8` / `INT3` (`755 MiB` for `LFM3-Omni-2B`), and sticks to strict W3C WGSL with bit-accurate non-subgroup fallback.
- **Sub-Millisecond Rust/WASM Tokenizer & JSON Schema FSM**: Single `1.3 MiB` WASM handles BPE / SentencePiece and `response_format` JSON Schema logit masking (`0.285 ms/tok` median).
- **Lossless Speculative Decoding**: Built-in zero-VRAM **N-gram Lookup** and GPU **Draft-Target** (`INT3` → `INT4/INT8`), bit-identical to greedy autoregressive decoding.
- **Batteries-Included SDK & Demo**: Ships OpenAI-compatible `ignis-webgpu`, pure-Python `ignis-webgpu-packer`, and the `examples/chat-vinext` Full-Omni workbench.

## Ignis vs. WebLLM

| Dimension | **Ignis** | **WebLLM (MLC-LLM / TVM)** |
| :--- | :--- | :--- |
| **Runtime & Build** | Zero-dep JS + W3C WGSL + single Rust/WASM; pack Safetensors in minutes via Python CLI | Apache TVM + Emscripten WASM toolchain (`mlc_llm` required per compiled model lib) |
| **VRAM & Storage** | Externalizes embeddings & Gemma 4 `2.82 GB` PLE to OPFS, cutting GPU VRAM | Loads weights and embeddings into WebGPU buffers via Cache API / IndexedDB |
| **Multimodal & Audio** | Native vision + audio-in + 24 kHz neural speech synthesis | Text & VLM focus; no native neural speech synthesis |
| **FSM & Speculative** | Built-in Rust/WASM JSON Schema FSM + N-gram / Draft-Target speculative decoding | WASM XGrammar structured output; standard autoregressive focus |
| **Portability** | Zero `@builtin(subgroup_id)` assumptions, shared-memory fallback, `INT3` tier | Broad TVM quantization presets (`q4f16_1` etc.), mature Chrome Extension / SW adapters |
| **Ecosystem** | Fast custom-checkpoint packing & hybrid architectures (`Spark`, `LFM3-Omni`, `Gemma 4`, `Qwen`) | Large pre-built MLC Hub model catalog & community ecosystem |

### When to Choose Ignis

- **Tight VRAM or huge vocabularies**: OPFS-externalized tables and `INT3` degradation fit multi-billion-parameter models on consumer GPUs.
- **Full-Omni voice & multimodal UX**: Single-worker image, audio-in, and 24 kHz speech output.
- **Browser Agents & structured/code edits**: Sub-ms JSON Schema FSM + N-gram speculative acceleration.
- **Custom HuggingFace checkpoints without TVM builds**: Pure-Python packing to static `.ignis` shards with zero external runtime dependencies.

> If you mainly want pre-compiled text/VLM models from MLC Hub or Chrome Extension service-worker templates, **WebLLM** is a strong, battle-tested choice.

## Quickstart

### 1. OpenAI-Compatible SDK (`ignis-webgpu`)

```bash
npm install ignis-webgpu
```

```js
import { IgnisOpenAI } from 'ignis-webgpu';

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

### 2. Pack HuggingFace Checkpoints (`ignis-webgpu-packer`)

```bash
npx ignis-webgpu-packer \
  --input ./path/to/hf-model \
  --output ./path/to/model-ignis \
  --lm-head-dtype int8 \
  --weight-bits 4
```

### 3. Build & Verify Locally

```bash
git clone https://github.com/Ray-D-Song/Ignis.git && cd Ignis
npm install
npm run audit:wgsl     # Zero-GPU WGSL + Rust/WASM FSM audit
npm run pack:check     # npm tarball + CLI smoke check
npm run demo:chat      # Start examples/chat-vinext at http://localhost:5173
```

## Browser Limits & Notes

- **CORS / CORP**: Cross-origin model buckets under `COEP: credentialless` require `Access-Control-Allow-Origin: *` and `Cross-Origin-Resource-Policy: cross-origin`.
- **Incognito OPFS Cap**: Chromium limits Incognito OPFS to `2 GB`; larger models (e.g., Gemma-4-E4B-it at `~6.5 GB`) require a normal profile (`launchPersistentContext` in Playwright).
- **WebGPU Tiering**: Verified on Chromium 138+ / Edge 157+ desktop GPUs (`shader-f16`). Lower-VRAM devices can use `detectDeviceProfile()` or `INT3_G64` checkpoints (`755 MiB` for `LFM3-Omni-2B`).
- **Reproducible WASM**: Tokenizer sources live in `tokenizer/` and build via `npm run build:wasm`.

## License & Attribution

- Apache License 2.0 ([LICENSE](LICENSE)); contribution & upstream notes in [CONTRIBUTING.md](CONTRIBUTING.md).
- Pre-quantized demo weights follow upstream licenses (`Apache-2.0` for `Spark-X2.6-4B`, `LFM Open License v1.0` for `LFM3-Omni-2B`, `Gemma Terms of Use` for `Gemma-4-E4B-it`).
