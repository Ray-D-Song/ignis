> **English** | [简体中文](README.zh-CN.md)

# Ignis

**A zero-dependency, browser-native WebGPU LLM and Full-Omni multimodal inference engine built from first principles.**

[**Live Demo**](https://ignis.ray-d-song.com) · [**ignis-webgpu**](packages/core/README.md) · [**ignis-webgpu-packer**](packages/packer/README.md) · [**Contributing**](CONTRIBUTING.md)

Ignis does not wrap ONNX Runtime, llama.cpp/WASM, or Apache TVM. Every layer—from its 256-byte-aligned `.ignis` binary shards and OPFS `SyncAccessHandle` streaming loader, to strict W3C baseline WGSL kernels, Rust/WASM tokenizers, JSON Schema FSM logit masking, and GPU-resident speculative decoding—is engineered specifically around browser sandbox constraints (`ArrayBuffer` caps, COOP/COEP isolation, WGSL subgroup divergence, and VRAM limits).

## Why Ignis?

- **OPFS Externalized Embeddings & 2.8 GiB PLE Streaming**: Instead of pinning massive `vocab_size × hidden_dim` embedding matrices in VRAM, `embd.ignis` and Gemma 4's `2.82 GB` `ple_embd.ignis` live in Origin Private File System and stream row-by-row via Worker `createSyncAccessHandle()` (`~0.1 ms` per token lookup, saving gigabytes of GPU memory).
- **One Engine, Full-Omni Native Pipelines**: Declarative architecture blueprints (`packer/blueprints.py` + `kernels/codegen.mjs` + `runtime/ignis-engine.mjs`) run standard Transformers (`Spark-X2.6-4B`, `Llama`, `Qwen2/3`, `Phi3`), Hybrid ShortConv+SWA Audio/Vision Omni (`LiquidAI/LFM3-Omni-2B` with 24 kHz `LiquidCodec` speech synthesis), and `262K`-vocab Per-Layer-Embedding Omni (`google/gemma-4-E4B-it` with SigLIP2 Vision + Chunked Conformer Audio) without per-model JS rewrites.
- **Strict W3C WebGPU Baseline + Verified Tiered Degradation**: Zero experimental `@builtin(subgroup_id)` or `@builtin(num_subgroups)` assumptions (which break Chrome/Edge even when `subgroups` is advertised). Includes bit-accurate `useSubgroups = false` shared-memory tree reduction (`cos = 0.99999999996`) and **`INT3_G64_ASYM`** quantization (packing `32 × 3-bit` weights into `3 × u32`, shrinking `LFM3-Omni-2B` GPU weights to **`755.1 MiB`**).
- **Sub-Millisecond Rust/WASM Tokenizer & JSON Schema FSM**: Single zero-JS-dependency `ignis_tokenizer.wasm` supports Byte-Level BPE (`131K` Spark) and SentencePiece `ByteFallback` (`128K` LFM3, `262K` Gemma-4) with 100% Oracle parity, plus a compiled NFA -> DFA -> packed `u32` bitset FSM (`response_format`) masking `131K+` logits in **`0.285 ms/tok` median** (`p95 = 2.05 ms/tok`).
- **Dual-Mode Lossless Speculative Decoding**:
  - **N-gram Lookup** (0 MiB extra VRAM): `84.5%` acceptance and **`134.0 tok/s` (`1.30x`)** on Spark-X2.6-4B.
  - **Draft-Target** (`INT3_G64` draft -> `INT4/INT8` target): Zero-readback GPU `ShortConv` state snapshot/rollback + Leviathan lossless rejection sampling, bit-identical to greedy autoregressive decoding (`90.2%` acceptance on LFM3-Omni-2B).

## Architecture

```mermaid
flowchart LR
  subgraph Storage["Browser Storage & Network"]
    CDN["HTTPS Shards / Local Model"] -->|"Range stream (no full-RAM blob)"| OPFS["OPFS Cache (.ignis)"]
    OPFS -->|"SyncAccessHandle row reads"| Embd["Host Row Lookup (embd.ignis / ple_embd.ignis)"]
  end

  subgraph Worker["Dedicated Web Worker (Isolated Heap ~1-73 MB)"]
    Tok["Rust/WASM Tokenizer & JSON Schema DFA FSM"] --> Engine["IgnisEngine (Declarative Blueprint)"]
    MM["Lfm3OmniRuntime / Gemma4OmniRuntime (Vision + Audio + 24 kHz LiquidCodec)"] --> Engine
    Spec["Speculative Coordinator (N-gram / Draft-Target)"] --> Engine
  end

  subgraph WebGPU["Strict W3C WebGPU Baseline"]
    Embd -->|"16 MiB StagingBuffer"| Shards["VRAM Quants (INT3_G64 / INT4_G64 / INT8_G64)"]
    Engine -->|"Single-submit Decode Graph"| Kernels["WGSL Mega-Kernels & Z-Tiled Output"]
    Kernels -->|"Optional Shared-Memory Fallback"| Shards
  end
```

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
