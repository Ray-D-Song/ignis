import fs from 'node:fs';
import path from 'node:path';
import { IgnisTokenizer } from '../runtime/wasm-tokenizer.mjs';

const ROOT_DIR = path.resolve(import.meta.dirname, '..');

const ALLOWED_COMPUTE_BUILTINS = new Set([
  'workgroup_id',
  'local_invocation_id',
  'local_invocation_index',
  'global_invocation_id',
  'num_workgroups',
  'subgroup_invocation_id',
  'subgroup_size',
]);

async function main() {
  // 1. Static WGSL W3C-Baseline Specification Compliance Check
  const kernelsDir = path.join(ROOT_DIR, 'kernels');
  let scannedFiles = 0;
  for (const fname of fs.readdirSync(kernelsDir)) {
    if (!fname.endsWith('.wgsl') && !fname.endsWith('.mjs')) continue;
    scannedFiles++;
    const content = fs.readFileSync(path.join(kernelsDir, fname), 'utf-8');
    for (const m of content.matchAll(/@builtin\(\s*([a-zA-Z0-9_]+)\s*\)/g)) {
      const builtinName = m[1];
      if (!ALLOWED_COMPUTE_BUILTINS.has(builtinName)) {
        throw new Error(
          `Non-baseline WGSL @builtin(${builtinName}) in kernels/${fname}. ` +
          `Allowed W3C baseline compute builtins: ${[...ALLOWED_COMPUTE_BUILTINS].join(', ')}`
        );
      }
    }
  }
  console.log(`[PASS] W3C WebGPU baseline WGSL audit (${scannedFiles} kernel modules checked, zero forbidden builtins)`);

  // 2. WASM Binary + Mini-BPE Roundtrip + JSON Schema FSM Check (no GPU/large models required)
  const wasmPath = path.join(ROOT_DIR, 'runtime/wasm/ignis_tokenizer.wasm');
  const miniTokPath = path.join(ROOT_DIR, 'tests/ci-fixtures/mini-bpe-tokenizer.json');
  const wasmBytes = fs.readFileSync(wasmPath);
  const tokBytes = fs.readFileSync(miniTokPath);

  const tok = await IgnisTokenizer.fromBuffers(wasmBytes, tokBytes);
  try {
    const sample = '{"city": "Paris", "country": "France"} hello 世界 🚀';
    const ids = tok.encode(sample);
    const decoded = tok.decode(ids, false);
    if (decoded !== sample || ids.length === 0) {
      throw new Error(`Mini-BPE roundtrip failed: expected=${sample}, got=${decoded}`);
    }

    const eosId = tok.getVocabSize() - 1;
    const fsm = tok.createFsm(
      {
        type: 'object',
        properties: {
          city: { type: 'string' },
          score: { type: 'integer' },
        },
        required: ['city', 'score'],
      },
      [eosId]
    );
    try {
      const V = tok.getVocabSize();
      const logits = new Float32Array(V);
      const allowed0 = fsm.maskLogits(logits);
      if (allowed0 <= 0) throw new Error('FSM initial state allowed 0 tokens');
      if (Number.isFinite(logits[eosId])) throw new Error('FSM should disallow EOS before JSON object starts');

      const jsonIds = tok.encode('{"city": "Paris", "score": 42}');
      for (const tid of jsonIds) {
        const stepLogits = new Float32Array(V);
        fsm.maskLogits(stepLogits);
        if (!Number.isFinite(stepLogits[tid])) {
          throw new Error(`FSM masked valid token ${tid} (${tok.decode([tid], false)})`);
        }
        if (!fsm.advance(tid)) {
          throw new Error(`FSM failed to advance valid token ${tid}`);
        }
      }
      if (!fsm.isDone()) throw new Error('FSM did not reach accepting state after valid JSON');
      const endLogits = new Float32Array(V);
      fsm.maskLogits(endLogits);
      if (!Number.isFinite(endLogits[eosId])) throw new Error('FSM failed to allow EOS after closing brace');
    } finally {
      fsm.free();
    }
    console.log('[PASS] Rust/WASM Tokenizer & JSON Schema FSM verified offline');
  } finally {
    tok.free();
  }
}

main().catch((err) => {
  console.error(err);
  process.exit(1);
});
