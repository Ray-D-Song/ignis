# `ignis-webgpu`

Zero-dependency browser-native **WebGPU LLM & Full-Omni multimodal inference SDK** (`IgnisOpenAI`, `IgnisEngine`, `IgnisTokenizer`, `IgnisFsm`, `detectDeviceProfile`, `NgramLookupSpeculator`, `DraftTargetSpeculator`).

Powered by native `.ignis` quantized shards (`INT4_G64` / `INT8_G64` / `INT3_G64`), externalized OPFS embedding & PLE tables, W3C-baseline WGSL kernels, and a Rust/WASM tokenizer + JSON Schema FSM engine.

## Install

```bash
npm install ignis-webgpu
```

## Quick Start

```js
import { IgnisOpenAI, getIgnisAssetUrls, detectDeviceProfile } from 'ignis-webgpu';

const profile = await detectDeviceProfile();
console.log(profile.tier, profile.recommendedWeightBits);

const assets = getIgnisAssetUrls();
const ignis = new IgnisOpenAI({
  workerUrl: assets.workerUrl,
  wasmUrl: assets.wasmUrl,
  kernelBaseUrl: assets.kernelBaseUrl,
  modelUrl: 'https://your-r2-cdn.example.com/lfm3-omni-2b-ignis',
  modelKey: 'lfm3-omni-2b',
});

// 1. Streaming chat completion (OpenAI-compatible)
const stream = ignis.chat.completions.create({
  stream: true,
  messages: [{ role: 'user', content: 'Explain WebGPU subgroup reductions concisely.' }],
  max_tokens: 256,
});
for await (const chunk of stream) {
  process.stdout?.write(chunk.choices[0]?.delta?.content || '');
}

// 2. JSON Schema Constrained Decoding (~0.25 ms/tok via Rust/WASM FSM)
const jsonResp = await ignis.chat.completions.create({
  messages: [{ role: 'user', content: 'Return JSON for Tokyo, Japan.' }],
  response_format: {
    type: 'json_schema',
    json_schema: {
      strict: true,
      schema: {
        type: 'object',
        properties: { city: { type: 'string' }, country: { type: 'string' } },
        required: ['city', 'country'],
      },
    },
  },
});

// 3. Zero-VRAM N-gram Speculative Decoding
const fastResp = await ignis.chat.completions.create({
  messages: [{ role: 'user', content: 'Refactor this JSON config...' }],
  speculative: { mode: 'ngram', k: 5 },
});
```

> Serve your host page with `Cross-Origin-Opener-Policy: same-origin` and `Cross-Origin-Embedder-Policy: credentialless` (or `require-corp`), and ensure remote model CDNs set `Access-Control-Allow-Origin: *` and `Cross-Origin-Resource-Policy: cross-origin`.

Full documentation & benchmarks: [github.com/Ray-D-Song/Ignis](https://github.com/Ray-D-Song/Ignis)
