# `ignis-webgpu-packer`

Offline HuggingFace Safetensors -> `.ignis` quantized shard packer CLI (`ignis-pack`).

Supports `Spark-2.5`, `Llama`, `Qwen2/Qwen3`, `Phi3`, `LiquidAI/LFM3-Omni-2B` (Vision + Audio-In + 24 kHz Speech Codec), and `google/gemma-4-E4B-it` (262K vocab + external `ple_embd.ignis` + Q8 PLE + Vision + Audio).

## Install & Usage

```bash
# Run via npx (requires Python 3.10+ with numpy, safetensors, torch, transformers)
npx ignis-webgpu-packer \
  --input ./hf-model-dir \
  --output ./out-ignis \
  --weight-bits 4 \
  --lm-head-dtype int8

# Ultra-low VRAM (3-bit body weights + 4-bit LM head, skips multimodal towers)
npx ignis-webgpu-packer \
  --input ./lfm3-omni-2b-hf \
  --output ./lfm3-omni-2b-q3 \
  --weight-bits 3 \
  --lm-head-dtype int4 \
  --no-multimodal
```

## CLI Options

- `--input <dir>`: HuggingFace model directory containing `config.json`, `tokenizer.json`, and `*.safetensors`
- `--output <dir>`: Output directory for `.ignis` shards + `manifest.json`
- `--weight-bits {4,3}`: Transformer body quantization (`INT4_G64_ASYM` or `INT3_G64_ASYM`)
- `--lm-head-dtype {int4,int8,int3}`: Output projection quantization
- `--max-shard-mb <int>`: GPU shard size cap in MiB (default: `512`)
- `--no-multimodal`: Pack text backbone only
- `--no-verify`: Skip per-tensor cosine verification during packing

Full docs & format spec: [github.com/Ray-D-Song/Ignis](https://github.com/Ray-D-Song/Ignis)
