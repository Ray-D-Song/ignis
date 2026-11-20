import http from 'node:http';
import fs from 'node:fs';
import path from 'node:path';
import { execFileSync } from 'node:child_process';
import { chromium } from 'playwright-core';

const EDGE_PATH = 'C:\\Program Files (x86)\\Microsoft\\Edge\\Application\\msedge.exe';
const ROOT_DIR = path.resolve(import.meta.dirname, '..');

function maxAbsErr(a, b) {
  let m = 0;
  for (let i = 0; i < a.length; i++) {
    const d = Math.abs(a[i] - b[i]);
    if (d > m) m = d;
  }
  return m;
}

const pyCode = `
import json, sys, numpy as np, torch, torch.nn.functional as F
sys.path.insert(0, 'packer')
from quant import quantize_int4_g64_asym, dequantize_int4_g64_asym, quantize_int8_g64_sym, dequantize_int8_g64_sym

rng = np.random.default_rng(7)
# RMSNorm
x_rms = rng.normal(size=(4, 256)).astype(np.float32)
w_rms = rng.normal(1.0, 0.1, size=(256,)).astype(np.float32)
t_x = torch.from_numpy(x_rms)
t_w = torch.from_numpy(w_rms)
y_rms = (t_w * t_x * torch.rsqrt(t_x.pow(2).mean(-1, keepdim=True) + 1e-6)).numpy().reshape(-1).tolist()

# Q4_G64 GEMV & GEMM
M, K, T = 64, 256, 6
W = rng.normal(0.0, 0.1, size=(M, K)).astype(np.float32)
q, s, (Mp, Kp) = quantize_int4_g64_asym(W)
W_deq = dequantize_int4_g64_asym(q, s, K)
X = rng.normal(0.0, 1.0, size=(T, K)).astype(np.float32)
Y = (X @ W_deq.T).astype(np.float32)

q8, s8, _ = quantize_int8_g64_sym(W)
W_deq8 = dequantize_int8_g64_sym(q8, s8, K)
Y8 = (X[0] @ W_deq8.T).astype(np.float32)

# GeGLU
ga = rng.normal(size=(512,)).astype(np.float32)
gb = rng.normal(size=(512,)).astype(np.float32)
gy = (F.gelu(torch.from_numpy(ga)) * torch.from_numpy(gb)).numpy().tolist()

# RoPE + SWA Attention + Gate
from gen_tiny_fixture import compute_rope_cos_sin, apply_rotary_pos_emb
T_a, q_h, kv_h, hd, rdim, win = 12, 8, 2, 64, 64, 8
q_dim, kv_dim = q_h * hd, kv_h * hd
qkv_np = rng.normal(0.0, 0.2, size=(T_a, q_dim + 2 * kv_dim)).astype(np.float32)
gate_np = rng.normal(0.0, 0.5, size=(T_a, q_h)).astype(np.float32)
qt = torch.from_numpy(qkv_np[:, :q_dim]).view(1, T_a, q_h, hd).transpose(1, 2)
kt = torch.from_numpy(qkv_np[:, q_dim:q_dim+kv_dim]).view(1, T_a, kv_h, hd).transpose(1, 2)
vt = torch.from_numpy(qkv_np[:, q_dim+kv_dim:]).view(1, T_a, kv_h, hd).transpose(1, 2)
cos, sin = compute_rope_cos_sin(torch.arange(T_a), hd, 10000.0, rdim / hd)
qr = apply_rotary_pos_emb(qt, cos, sin)
kr = apply_rotary_pos_emb(kt, cos, sin)
qrot_expected = qr.transpose(1, 2).contiguous().reshape(-1).numpy().tolist()
ng = q_h // kv_h
kr_rep = kr[:, :, None, :, :].expand(1, kv_h, ng, T_a, hd).reshape(1, q_h, T_a, hd)
vr_rep = vt[:, :, None, :, :].expand(1, kv_h, ng, T_a, hd).reshape(1, q_h, T_a, hd)
scores = torch.matmul(qr, kr_rep.transpose(2, 3)) / np.sqrt(hd)
qp = torch.arange(T_a)[:, None]
kp = torch.arange(T_a)[None, :]
mask = (kp <= qp) & ((qp - kp) < win)
scores = scores.masked_fill(~mask[None, None, :, :], float('-inf'))
probs = F.softmax(scores, dim=-1)
ao = torch.matmul(probs, vr_rep) * torch.sigmoid(torch.from_numpy(gate_np).view(1, T_a, q_h, 1).transpose(1, 2))
attn_expected = ao.transpose(1, 2).contiguous().reshape(-1).numpy().tolist()

print(json.dumps({
  "rms": {"dim": 256, "tokens": 4, "eps": 1e-6, "x": x_rms.reshape(-1).tolist(), "w": w_rms.tolist(), "expected": y_rms},
  "qmm": {"M": M, "K": K, "T": T, "K_pad": Kp, "quants": q.reshape(-1).tolist(), "scales": s.reshape(-1).tolist(), "x": X.reshape(-1).tolist(), "expected_gemv": Y[0].tolist(), "expected_gemm": Y.reshape(-1).tolist()},
  "q8": {"quants": q8.reshape(-1).tolist(), "scales": s8.reshape(-1).tolist(), "expected": Y8.tolist()},
  "geglu": {"a": ga.tolist(), "b": gb.tolist(), "expected": gy},
  "attn": {"T": T_a, "start_pos": 0, "q_heads": q_h, "kv_heads": kv_h, "head_dim": hd, "rope_dim": rdim, "kv_capacity": win + T_a, "sliding_window": win, "is_swa": 1, "has_gate": 1, "rope_theta": 10000.0, "qkv": qkv_np.reshape(-1).tolist(), "gate": gate_np.reshape(-1).tolist(), "expected_qrot": qrot_expected, "expected_attn": attn_expected}
}))
`;

const golden = JSON.parse(
  execFileSync(path.join(ROOT_DIR, 'packer/.venv/Scripts/python.exe'), ['-c', pyCode], { cwd: ROOT_DIR, encoding: 'utf-8', maxBuffer: 50 * 1024 * 1024 })
);

const server = http.createServer((req, res) => {
  const reqPath = (req.url || '/').split('?')[0];
  if (reqPath === '/favicon.ico') { res.writeHead(204); res.end(); return; }
  const filePath = path.join(ROOT_DIR, reqPath.replace(/^\/+/, ''));
  if (!fs.existsSync(filePath)) { res.writeHead(404); res.end(); return; }
  const ext = path.extname(filePath);
  const mime = ext === '.html' ? 'text/html' : ext === '.js' || ext === '.mjs' ? 'application/javascript' : 'text/plain';
  res.writeHead(200, { 'Content-Type': mime });
  fs.createReadStream(filePath).pipe(res);
});

await new Promise((r) => server.listen(0, '127.0.0.1', r));
const port = server.address().port;
const browser = await chromium.launch({
  executablePath: EDGE_PATH,
  headless: true,
  args: ['--headless=new', '--enable-unsafe-webgpu', '--use-angle=d3d11', '--enable-features=WebGPUDeveloperFeatures,Vulkan', '--ignore-gpu-blocklist'],
});
try {
  const page = await browser.newPage();
  await page.goto(`http://127.0.0.1:${port}/tests/test-kernels.html`);
  const out = await page.evaluate((g) => window.__runKernelTests(g), golden);
  console.log('RMSNorm maxAbsErr:', maxAbsErr(out.rmsOut, golden.rms.expected));
  console.log('GEMV    maxAbsErr:', maxAbsErr(out.gemvOut, golden.qmm.expected_gemv));
  console.log('GEMV_Q8 maxAbsErr:', maxAbsErr(out.gemvQ8Out, golden.q8.expected));
  console.log('GEMM4   maxAbsErr:', maxAbsErr(out.gemmOut, golden.qmm.expected_gemm));
  console.log('GeGLU   maxAbsErr:', maxAbsErr(out.gegluOut, golden.geglu.expected));
  console.log('QRot    maxAbsErr:', maxAbsErr(out.qRotOut, golden.attn.expected_qrot));
  console.log('Attn    maxAbsErr:', maxAbsErr(out.attnOut, golden.attn.expected_attn));
} finally {
  await browser.close();
  server.close();
}
