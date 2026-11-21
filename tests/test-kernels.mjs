import http from 'node:http';
import fs from 'node:fs';
import path from 'node:path';
import { execFileSync } from 'node:child_process';
import { chromium } from 'playwright-core';

const EDGE_PATH = 'C:\\Program Files (x86)\\Microsoft\\Edge\\Application\\msedge.exe';
const THORIUM_PATH = 'C:\\Users\\ray\\AppData\\Local\\Thorium\\Application\\thorium.exe';
const ROOT_DIR = path.resolve(import.meta.dirname, '..');

// Static WGSL W3C-Baseline Specification Compliance Check:
// Ensures no kernel uses non-baseline subgroup builtins (such as subgroup_id or num_subgroups)
// that fail on Chromium 134..14x without experimental flags.
const ALLOWED_COMPUTE_BUILTINS = new Set([
  'workgroup_id',
  'local_invocation_id',
  'local_invocation_index',
  'global_invocation_id',
  'num_workgroups',
  'subgroup_invocation_id',
  'subgroup_size',
]);
const kernelsDir = path.join(ROOT_DIR, 'kernels');
for (const fname of fs.readdirSync(kernelsDir)) {
  if (!fname.endsWith('.wgsl') && !fname.endsWith('.mjs')) continue;
  const content = fs.readFileSync(path.join(kernelsDir, fname), 'utf-8');
  for (const m of content.matchAll(/@builtin\(\s*([a-zA-Z0-9_]+)\s*\)/g)) {
    const builtinName = m[1];
    if (!ALLOWED_COMPUTE_BUILTINS.has(builtinName)) {
      throw new Error(
        `Non-baseline WGSL @builtin(${builtinName}) in kernels/${fname}. ` +
        `Only W3C baseline compute/subgroups builtins are allowed: ${[...ALLOWED_COMPUTE_BUILTINS].join(', ')}`
      );
    }
  }
}

function maxAbsErr(a, b) {
  let m = 0;
  for (let i = 0; i < a.length; i++) {
    const d = Math.abs(a[i] - b[i]);
    if (d > m) m = d;
  }
  return m;
}

function cosineSim(a, b) {
  let dot = 0, na = 0, nb = 0;
  for (let i = 0; i < a.length; i++) {
    dot += a[i] * b[i];
    na += a[i] * a[i];
    nb += b[i] * b[i];
  }
  return dot / (Math.sqrt(na) * Math.sqrt(nb) + 1e-12);
}

const pyCode = `
import json, sys, numpy as np, torch, torch.nn.functional as F
sys.path.insert(0, 'packer')
from quant import quantize_int4_g64_asym, dequantize_int4_g64_asym, quantize_int8_g64_sym, dequantize_int8_g64_sym, quantize_int3_g64_asym, dequantize_int3_g64_asym

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

q3, s3, _ = quantize_int3_g64_asym(W)
W_deq3 = dequantize_int3_g64_asym(q3, s3, K)
Y3 = (X @ W_deq3.T).astype(np.float32)

# ShortConv K=7 (LFM3)
D_sc = 128
bcx_pre = rng.normal(size=(4, 3 * D_sc)).astype(np.float32)
bcx_dec = rng.normal(size=(1, 3 * D_sc)).astype(np.float32)
w_sc = rng.normal(size=(D_sc, 7)).astype(np.float32)
def ref_sconv(bcx_all, start, length):
  b, c, xv = torch.from_numpy(bcx_all[:, :D_sc]), torch.from_numpy(bcx_all[:, D_sc:2*D_sc]), torch.from_numpy(bcx_all[:, 2*D_sc:])
  bx = (b * xv).T.unsqueeze(0)
  out_full = F.conv1d(bx, torch.from_numpy(w_sc).unsqueeze(1), padding=6, groups=D_sc)[:, :, :bx.shape[2]]
  y = (c * out_full[0].T).numpy()
  return y[start:start+length].reshape(-1).tolist()
bcx_cat = np.concatenate([bcx_pre, bcx_dec], axis=0)
sc_pre_exp = ref_sconv(bcx_cat, 0, 4)
sc_dec_exp = ref_sconv(bcx_cat, 4, 1)

# Gemma 4 head_dim=512 + QK+V Norm + Proportional RoPE + Attn(scale=1.0)
T5, qh5, kvh5, hd5, rdim5 = 3, 4, 2, 512, 128
qkv5 = rng.normal(0.0, 0.25, size=(T5, (qh5 + 2 * kvh5) * hd5)).astype(np.float32)
qn5 = (1.0 + rng.normal(0.0, 0.05, size=(hd5,))).astype(np.float32)
kn5 = (1.0 + rng.normal(0.0, 0.05, size=(hd5,))).astype(np.float32)
t_qkv5 = torch.from_numpy(qkv5)
qt5 = t_qkv5[:, :qh5*hd5].view(T5, qh5, hd5)
kt5 = t_qkv5[:, qh5*hd5:(qh5+kvh5)*hd5].view(T5, kvh5, hd5)
vt5 = t_qkv5[:, (qh5+kvh5)*hd5:].view(T5, kvh5, hd5)
def rms_u(z, w=None):
  r = z * torch.rsqrt(z.pow(2).mean(-1, keepdim=True) + 1e-6)
  return r * torch.from_numpy(w) if w is not None else r
def rotate_half_t(x):
  return torch.cat((-x[..., x.shape[-1] // 2:], x[..., :x.shape[-1] // 2]), dim=-1)
qt5 = rms_u(qt5, qn5).transpose(0, 1).unsqueeze(0)
kt5 = rms_u(kt5, kn5).transpose(0, 1).unsqueeze(0)
vt5 = rms_u(vt5, None).transpose(0, 1).unsqueeze(0)
pos5 = torch.arange(T5, dtype=torch.float32)[:, None]
ang5 = rdim5 // 2
inv5 = 1.0 / (1000000.0 ** (torch.arange(0, 2 * ang5, 2, dtype=torch.float32) / hd5))
inv5 = torch.cat([inv5, torch.zeros(hd5 // 2 - ang5, dtype=torch.float32)])
fr5 = pos5 * inv5[None, :]
em5 = torch.cat([fr5, fr5], dim=-1)
cos5, sin5 = em5.cos()[None, None], em5.sin()[None, None]
qt5 = qt5 * cos5 + rotate_half_t(qt5) * sin5
kt5 = kt5 * cos5 + rotate_half_t(kt5) * sin5
kt5_rep = kt5[:, :, None].expand(1, kvh5, qh5 // kvh5, T5, hd5).reshape(1, qh5, T5, hd5)
vt5_rep = vt5[:, :, None].expand(1, kvh5, qh5 // kvh5, T5, hd5).reshape(1, qh5, T5, hd5)
sc5 = torch.matmul(qt5, kt5_rep.transpose(2, 3)) * 1.0
mask5 = torch.arange(T5)[None, :] <= torch.arange(T5)[:, None]
sc5 = sc5.masked_fill(~mask5[None, None], float('-inf'))
ao5 = torch.matmul(F.softmax(sc5, dim=-1), vt5_rep).squeeze(0).transpose(0, 1).contiguous().reshape(-1).numpy().tolist()

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
  "q3": {"quants": q3.reshape(-1).tolist(), "scales": s3.reshape(-1).tolist(), "expected_gemv": Y3[0].tolist(), "expected_gemm": Y3.reshape(-1).tolist()},
  "sconv": {"dim": D_sc, "bcx_prefill": bcx_pre.reshape(-1).tolist(), "bcx_decode": bcx_dec.reshape(-1).tolist(), "w": w_sc.reshape(-1).tolist(), "expected_pre": sc_pre_exp, "expected_dec": sc_dec_exp},
  "hd512": {"T": T5, "q_heads": qh5, "kv_heads": kvh5, "qkv": qkv5.reshape(-1).tolist(), "q_norm": qn5.tolist(), "k_norm": kn5.tolist(), "expected": ao5},
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
const candidateBrowsers = [THORIUM_PATH, EDGE_PATH].filter((p) => fs.existsSync(p));
try {
  for (const executablePath of candidateBrowsers) {
    const browser = await chromium.launch({
      executablePath,
      headless: true,
      args: ['--headless=new'],
    });
    try {
      const page = await browser.newPage();
      await page.goto(`http://127.0.0.1:${port}/tests/test-kernels.html`);
      const ua = await page.evaluate(() => navigator.userAgent);
      console.log(`[test-kernels] Browser: ${path.basename(executablePath)} (${ua})`);
      const out = await page.evaluate((g) => window.__runKernelTests(g), golden);
      console.log('RMSNorm maxAbsErr:', maxAbsErr(out.rmsOut, golden.rms.expected));
      console.log('GEMV    maxAbsErr:', maxAbsErr(out.gemvOut, golden.qmm.expected_gemv));
      console.log('GEMV_Q8 maxAbsErr:', maxAbsErr(out.gemvQ8Out, golden.q8.expected));
      console.log('GEMM4   maxAbsErr:', maxAbsErr(out.gemmOut, golden.qmm.expected_gemm));
      console.log('GeGLU   maxAbsErr:', maxAbsErr(out.gegluOut, golden.geglu.expected));
      console.log('QRot    maxAbsErr:', maxAbsErr(out.qRotOut, golden.attn.expected_qrot));
      console.log('Attn    maxAbsErr:', maxAbsErr(out.attnOut, golden.attn.expected_attn));
      for (const [label, cgOut] of [['Subgroup=true', out.cgSg], ['Subgroup=false', out.cgNoSg]]) {
        const eQ3v = maxAbsErr(cgOut.q3vOut, golden.q3.expected_gemv);
        const eQ3m = maxAbsErr(cgOut.q3mOut, golden.q3.expected_gemm);
        const eScPre = maxAbsErr(cgOut.sconvPreOut, golden.sconv.expected_pre);
        const eScDec = maxAbsErr(cgOut.sconvDecOut, golden.sconv.expected_dec);
        const eHd512 = maxAbsErr(cgOut.hd512Out, golden.hd512.expected);
        const cHd512 = cosineSim(cgOut.hd512Out, golden.hd512.expected);
        console.log(`  [Codegen ${label}] Q3v=${eQ3v.toExponential(2)} Q3m=${eQ3m.toExponential(2)} SConvPre=${eScPre.toExponential(2)} SConvDec=${eScDec.toExponential(2)} HD512=${eHd512.toExponential(2)} (cos=${cHd512.toFixed(6)})`);
        if (Math.max(eQ3v, eQ3m, eScPre, eScDec) > 1e-4 || cHd512 < 0.9999 || eHd512 > 1e-2) {
          throw new Error(`Codegen error threshold exceeded (${label})`);
        }
      }
    } finally {
      await browser.close();
    }
  }
} finally {
  server.close();
}
