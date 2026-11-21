# Security Policy

## Scope

Ignis executes quantized neural network inference and compiles JSON Schema FSMs strictly inside the user's browser sandbox (`WebGPU` + `WebAssembly` + `OPFS` + `Dedicated Worker`), with zero telemetry or server-side inference backend.

## Reporting a Vulnerability

If you discover a security issue (for example in WASM memory handling inside `tokenizer/src/`, untrusted `manifest.json` shard parsing, or HTTP header isolation templates), please open a private security advisory on GitHub or email the repository maintainer directly rather than filing a public issue with exploit details.

Please include:
- Ignis version / commit SHA
- Browser name, version, OS, and GPU adapter
- Minimal reproduction steps
