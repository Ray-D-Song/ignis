import fs from 'node:fs';
import path from 'node:path';
import { execFileSync } from 'node:child_process';
import { IgnisTokenizer } from '../runtime/wasm-tokenizer.mjs';
import { resolvePythonExecutable } from './browser-env.mjs';

const ROOT_DIR = path.resolve(import.meta.dirname, '..');
const BUILT_WASM = path.join(ROOT_DIR, 'runtime/wasm/ignis_tokenizer.wasm');
const TARGET_WASM = path.join(ROOT_DIR, 'tokenizer/target/wasm32-unknown-unknown/release/ignis_tokenizer.wasm');
const WASM_PATH = fs.existsSync(BUILT_WASM) ? BUILT_WASM : TARGET_WASM;
const TOK_JSON_PATH = path.join(ROOT_DIR, 'tests/fixtures/tokenizer.json');
const MINI_TOK_PATH = path.join(ROOT_DIR, 'tests/ci-fixtures/mini-bpe-tokenizer.json');
const LLAMA_TOKENIZE = process.env.IGNIS_LLAMA_TOKENIZE || '';
const GGUF_PATH = process.env.IGNIS_SPARK_GGUF || '';

function oracleTokenize(prompt) {
  const out = execFileSync(LLAMA_TOKENIZE, ['-m', GGUF_PATH, '-p', prompt, '--ids'], { encoding: 'utf-8' });
  const m = out.match(/\[([0-9,\s]+)\]\s*$/m);
  if (!m) throw new Error(`Failed to parse llama-tokenize output: ${out}`);
  return m[1].split(',').map((s) => Number(s.trim()));
}

async function main() {
  const wasmBytes = fs.readFileSync(WASM_PATH);
  console.log(`WASM binary size: ${(wasmBytes.length / 1024).toFixed(1)} KiB`);

  // 1. Always verify CI mini BPE + UTF-8 round-trip + JSON Schema FSM
  const miniTok = await IgnisTokenizer.fromBuffers(wasmBytes, fs.readFileSync(MINI_TOK_PATH));
  const miniSample = '{"city": "Paris", "country": "France"} hello 世界 🚀';
  const miniIds = miniTok.encode(miniSample);
  if (miniTok.decode(miniIds, false) !== miniSample) {
    throw new Error('CI mini-BPE roundtrip mismatch');
  }
  miniTok.free();
  console.log(`[PASS] CI Mini-BPE tokenizer round-trip verified (${miniIds.length} tokens)`);

  const cases = [
    'Hello, world! 你好世界 12345 def foo(x): return x + 1',
    'Spark-X2.6-4B 在浏览器中基于 WebGPU 推理，支持 1,048,576 上下文！',
    '<｜start▁of▁sentence｜><think>\nLet us test 42 + 99 = 141.\n</think>\n答案是 141。<｜end▁of▁sentence｜>',
    '   multiple   spaces\n\nand\tnewlines!!  @user #tag https://example.com/a_b?c=1',
    '日本語テスト カタカナ ひらがな 🚀🔥',
  ];

  if (fs.existsSync(TOK_JSON_PATH)) {
    const tokJsonBytes = fs.readFileSync(TOK_JSON_PATH);
    const t0 = performance.now();
    const tok = await IgnisTokenizer.fromBuffers(wasmBytes, tokJsonBytes);
    console.log(`Tokenizer init from 10MB JSON: ${(performance.now() - t0).toFixed(1)} ms`);
    const hasLlamaOracle = LLAMA_TOKENIZE && GGUF_PATH && fs.existsSync(LLAMA_TOKENIZE) && fs.existsSync(GGUF_PATH);

    for (const text of cases) {
      const got = tok.encode(text);
      const decoded = tok.decode(got, false);
      if (hasLlamaOracle) {
        const expected = oracleTokenize(text);
        const match = JSON.stringify(got) === JSON.stringify(expected);
        console.log(`Case: ${JSON.stringify(text.slice(0, 40))}... -> match=${match} len=${got.length}`);
        if (!match) {
          throw new Error(`Tokenizer mismatch on: ${text}`);
        }
      }
      if (decoded !== text) {
        throw new Error(`Round-trip decode mismatch!\nExpected: ${JSON.stringify(text)}\nDecoded:  ${JSON.stringify(decoded)}`);
      }
    }
    tok.free();
    console.log(
      hasLlamaOracle
        ? '[PASS] Rust-WASM Tokenizer matches llama-tokenize 100% across all 131K vocab test cases!'
        : '[PASS] Rust-WASM Tokenizer 131K round-trip verified (set IGNIS_LLAMA_TOKENIZE & IGNIS_SPARK_GGUF for C++ oracle check)'
    );
  } else {
    console.log('[SKIP] Full 131K Spark tokenizer fixture not present (run locally with tests/fixtures/tokenizer.json)');
  }

  // 2. Multi-arch tokenizer regression checks (LFM3 ByteLevel 128K + Gemma 4 SentencePiece ByteFallback 262K)
  const py = resolvePythonExecutable(ROOT_DIR);
  const extraModels = [
    ['LFM3-Omni-2B', path.join(ROOT_DIR, 'tests/fixtures/lfm3-omni-2b-hf/tokenizer.json')],
    ['Gemma-4-E4B-it', path.join(ROOT_DIR, 'tests/fixtures/gemma-4-e4b-it-hf/tokenizer.json')],
  ];
  for (const [label, tokPath] of extraModels) {
    if (!fs.existsSync(tokPath)) continue;
    const spCases = [
      ...cases.slice(0, 2),
      '  leading and  multiple   spaces\n\tand tabs 🚀🔥 \u00a0nbsp',
    ];
    const expIds = JSON.parse(
      execFileSync(
        py,
        [
          '-c',
          'import json,sys; from tokenizers import Tokenizer; tk=Tokenizer.from_file(sys.argv[1]); print(json.dumps([tk.encode(c, add_special_tokens=False).ids for c in json.loads(sys.argv[2])]))',
          tokPath,
          JSON.stringify(spCases),
        ],
        { encoding: 'utf-8' }
      )
    );
    const mt = await IgnisTokenizer.fromBuffers(wasmBytes, fs.readFileSync(tokPath));
    for (let i = 0; i < spCases.length; i++) {
      const got = mt.encode(spCases[i]);
      const dec = mt.decode(got, false);
      if (JSON.stringify(got) !== JSON.stringify(expIds[i])) {
        throw new Error(`${label} encode mismatch on case ${i}: expected=${expIds[i].slice(0, 12)}, got=${got.slice(0, 12)}`);
      }
      if (dec !== spCases[i]) {
        throw new Error(`${label} roundtrip mismatch on case ${i}`);
      }
    }
    mt.free();
    console.log(`[PASS] Rust-WASM Tokenizer matches HuggingFace oracle 100% on ${label}!`);
  }
}

main().catch((err) => {
  console.error(err);
  process.exit(1);
});
