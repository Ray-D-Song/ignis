import http from 'node:http';
import fs from 'node:fs';
import path from 'node:path';
import { execFileSync } from 'node:child_process';
import { chromium } from 'playwright-core';

const EDGE_PATH = 'C:\\Program Files (x86)\\Microsoft\\Edge\\Application\\msedge.exe';
const ROOT_DIR = path.resolve(import.meta.dirname, '..');
const PY = path.join(ROOT_DIR, 'packer/.venv/Scripts/python.exe');

// Generate synthetic Llama/Qwen-style SwiGLU + separate q/k/v + untied lm_head fixture
const genLlamaPy = `
import json
from pathlib import Path
import torch, torch.nn as nn, torch.nn.functional as F
from safetensors.torch import save_file

torch.manual_seed(123)
out_dir = Path('tests/fixtures/tiny-llama-hf')
out_dir.mkdir(parents=True, exist_ok=True)
cfg = {
    "architectures": ["LlamaForCausalLM"],
    "model_type": "llama",
    "vocab_size": 256,
    "hidden_size": 256,
    "intermediate_size": 512,
    "num_hidden_layers": 2,
    "num_attention_heads": 4,
    "num_key_value_heads": 2,
    "head_dim": 64,
    "hidden_act": "silu",
    "rms_norm_eps": 1e-6,
    "rope_theta": 10000.0,
    "headwise_attn_output_gate": False,
    "tie_word_embeddings": False,
}
(out_dir / 'config.json').write_text(json.dumps(cfg, indent=2), encoding='utf-8')
tensors = {
    "model.embed_tokens.weight": torch.randn(256, 256) * 0.04,
    "lm_head.weight": torch.randn(256, 256) * 0.04,
    "model.norm.weight": 1.0 + torch.randn(256) * 0.05,
}
for i in range(2):
    p = f"model.layers.{i}"
    tensors[f"{p}.input_layernorm.weight"] = 1.0 + torch.randn(256) * 0.05
    tensors[f"{p}.post_attention_layernorm.weight"] = 1.0 + torch.randn(256) * 0.05
    tensors[f"{p}.self_attn.q_proj.weight"] = torch.randn(256, 256) * 0.04
    tensors[f"{p}.self_attn.k_proj.weight"] = torch.randn(128, 256) * 0.04
    tensors[f"{p}.self_attn.v_proj.weight"] = torch.randn(128, 256) * 0.04
    tensors[f"{p}.self_attn.o_proj.weight"] = torch.randn(256, 256) * 0.04
    tensors[f"{p}.mlp.gate_proj.weight"] = torch.randn(512, 256) * 0.04
    tensors[f"{p}.mlp.up_proj.weight"] = torch.randn(512, 256) * 0.04
    tensors[f"{p}.mlp.down_proj.weight"] = torch.randn(256, 512) * 0.04
save_file({k: v.half() for k, v in tensors.items()}, str(out_dir / 'model.safetensors'))
golden = {"prompt_ids": [5, 17, 42, 99, 120, 7], "prefill_last": {"final_norm": [0]*256}, "generated_greedy_ids": [0]*6, "decode_steps": [{"final_norm": [0]*256} for _ in range(5)]}
Path('tests/fixtures/tiny-llama-golden.json').write_text(json.dumps(golden), encoding='utf-8')
`;

execFileSync(PY, ['-c', genLlamaPy], { cwd: ROOT_DIR, stdio: 'inherit' });
execFileSync(PY, ['packer/ignis_pack.py', '--input', 'tests/fixtures/tiny-llama-hf', '--output', 'tests/fixtures/tiny-llama-ignis'], { cwd: ROOT_DIR, stdio: 'inherit' });
execFileSync(PY, ['packer/verify_pack.py', '--pkg', 'tests/fixtures/tiny-llama-ignis', '--golden', 'tests/fixtures/tiny-llama-golden.json', '--out-quant-golden', 'tests/fixtures/tiny-llama-quant-golden.json'], { cwd: ROOT_DIR, stdio: 'inherit' });

const server = http.createServer((req, res) => {
  const reqPath = (req.url || '/').split('?')[0];
  if (reqPath === '/favicon.ico') { res.writeHead(204); res.end(); return; }
  const filePath = path.join(ROOT_DIR, reqPath.replace(/^\/+/, ''));
  if (!filePath.startsWith(ROOT_DIR) || !fs.existsSync(filePath)) { res.writeHead(404); res.end(); return; }
  const ext = path.extname(filePath);
  const mime = ext === '.html' ? 'text/html' : (ext === '.js' || ext === '.mjs') ? 'application/javascript' : ext === '.json' ? 'application/json' : 'text/plain';
  res.writeHead(200, { 'Content-Type': mime });
  fs.createReadStream(filePath).pipe(res);
});
await new Promise((r) => server.listen(0, '127.0.0.1', r));
const port = server.address().port;

const browser = await chromium.launch({
  executablePath: EDGE_PATH,
  headless: true,
  args: ['--headless=new'],
});
try {
  const page = await browser.newPage();
  const errs = [];
  page.on('console', (m) => { if (m.type() === 'error') errs.push(m.text()); });
  page.on('pageerror', (e) => errs.push(String(e)));
  await page.goto(`http://127.0.0.1:${port}/tests/phase5-runner.html`);
  const res = await page.evaluate(() => window.__runPhase5MultiArch());
  console.log(JSON.stringify(res, null, 2));
  if (errs.length) throw new Error(`Browser errors: ${errs.join(' | ')}`);
  const minCos = Math.min(res.prefillFinalNormCos, ...res.decodeFinalNormCos);
  if (minCos < 0.999 || !res.greedyMatch) {
    throw new Error(`Multi-arch SwiGLU check failed! minCos=${minCos}, greedyMatch=${res.greedyMatch}`);
  }
  console.log(`[PASS] Phase 5 Multi-Arch SwiGLU verified: minCos=${minCos.toFixed(6)}, greedyMatch=true`);
} finally {
  await browser.close();
  server.close();
}
