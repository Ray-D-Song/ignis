# Contributing to Ignis

Thank you for your interest in **Ignis**! We welcome contributions across WebGPU kernels, model blueprints, multimodal preprocessors, Rust/WASM tokenizer/FSM enhancements, and cross-platform browser verification.

## Development Setup

```bash
git clone https://github.com/Ray-D-Song/Ignis.git
cd Ignis
npm install
npm --prefix examples/chat-vinext install
```

### Building the Rust/WASM Tokenizer & JSON Schema FSM

Requires Rust (`rustup target add wasm32-unknown-unknown`):

```bash
cargo test --manifest-path tokenizer/Cargo.toml
npm run build:wasm
```

## Verification Gates

1. **W3C WebGPU Baseline Compliance**:
   Every WGSL kernel in `kernels/` must strictly adhere to W3C WebGPU baseline compute builtins (`workgroup_id`, `local_invocation_id`, `local_invocation_index`, `global_invocation_id`, `num_workgroups`, `subgroup_invocation_id`, `subgroup_size`). Experimental builtins like `@builtin(subgroup_id)` or `@builtin(num_subgroups)` are strictly forbidden.
   Run:
   ```bash
   npm run audit:wgsl
   ```
2. **Self-Contained Package Smoke Check**:
   ```bash
   npm run pack:check
   ```
3. **Browser Hardware Tests** (requires a WebGPU-capable Chromium/Edge/Thorium executable; override via `IGNIS_BROWSER=/path/to/browser`):
   ```bash
   npm run test:kernels
   npm run test:demo:static
   ```

## Adding a New Model Architecture

1. Add a normalized blueprint mapping in `packer/blueprints.py` and tensor key rules in `packer/ignis_pack.py`.
2. Reuse WGSL kernel templates in `kernels/codegen.mjs` (or add new multimodal kernels in `kernels/multimodal-codegen.mjs`).
3. Verify numerical parity (`cosine >= 0.995` vs PyTorch BF16 logits and Top-1 token match) using a golden reference script in `tests/`.
