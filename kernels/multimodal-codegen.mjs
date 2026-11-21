/**
 * W3C-Baseline WGSL Kernels for Ignis Multimodal Towers (Vision, Audio Encoder, Depth Decoder, Audio Codec).
 *
 * Strictly avoids non-baseline subgroup builtins (no `subgroup_id` or `num_subgroups` attributes);
 * supports both standard WebGPU `subgroups` and pure shared-memory fallback (`useSubgroups: false`).
 */

function wgslHeader({ useSubgroups = true } = {}) {
  return useSubgroups ? 'enable subgroups;\n\n' : '';
}

function sgParams({ useSubgroups = true } = {}) {
  return useSubgroups
    ? `  @builtin(workgroup_id) wid: vec3<u32>,
  @builtin(local_invocation_id) lid: vec3<u32>,
  @builtin(subgroup_invocation_id) sg_lid: u32,
  @builtin(subgroup_size) sg_size: u32,`
    : `  @builtin(workgroup_id) wid: vec3<u32>,
  @builtin(local_invocation_id) lid: vec3<u32>,`;
}

export function buildMultimodalOpsWgsl({ useSubgroups = true } = {}) {
  const smem = useSubgroups
    ? `var<workgroup> sg_partial: array<f32, 64>;\nvar<workgroup> s_mean: f32;\nvar<workgroup> s_inv_std: f32;`
    : `var<workgroup> wg_red: array<f32, 256>;\nvar<workgroup> s_mean: f32;\nvar<workgroup> s_inv_std: f32;`;

  const reduceSum = (varName, outStmt) => useSubgroups
    ? `  {
    let sg_id = lid.x / sg_size;
    let num_sg = (256u + sg_size - 1u) / sg_size;
    let sg_s = subgroupAdd(${varName});
    if (sg_lid == 0u) { sg_partial[sg_id] = sg_s; }
    workgroupBarrier();
    if (lid.x == 0u) {
      var tot: f32 = 0.0;
      for (var s: u32 = 0u; s < num_sg; s += 1u) { tot += sg_partial[s]; }
      ${outStmt}
    }
  }
  workgroupBarrier();`
    : `  {
    wg_red[lid.x] = ${varName};
    workgroupBarrier();
    for (var stride: u32 = 128u; stride > 0u; stride >>= 1u) {
      if (lid.x < stride) { wg_red[lid.x] += wg_red[lid.x + stride]; }
      workgroupBarrier();
    }
    if (lid.x == 0u) {
      let tot = wg_red[0];
      ${outStmt}
    }
  }
  workgroupBarrier();`;

  return `${wgslHeader({ useSubgroups })}struct MmOpsParams {
  a: u32,     // dim / M / count
  b: u32,     // tokens / K
  c: u32,     // T / mode
  d: u32,     // unused
  eps: f32,
  f0: f32,    // scalar
  f1: f32,
  f2: f32,
};

@group(0) @binding(0) var<uniform> params: MmOpsParams;
@group(0) @binding(1) var<storage, read> weight: array<f32>;
@group(0) @binding(2) var<storage, read> bias: array<f32>;
@group(0) @binding(3) var<storage, read> x: array<f32>;
@group(0) @binding(4) var<storage, read_write> out: array<f32>;

${smem}

@compute @workgroup_size(256)
fn layernorm_f32(${sgParams({ useSubgroups })}) {
  let tok = wid.x;
  if (tok >= params.b) { return; }
  let dim = params.a;
  let base = tok * dim;

  var sum_x: f32 = 0.0;
  for (var i: u32 = lid.x; i < dim; i += 256u) {
    sum_x += x[base + i];
  }
${reduceSum('sum_x', 's_mean = tot / f32(dim);')}

  let mean = s_mean;
  var sum_sq: f32 = 0.0;
  for (var i: u32 = lid.x; i < dim; i += 256u) {
    let df = x[base + i] - mean;
    sum_sq += df * df;
  }
${reduceSum('sum_sq', 's_inv_std = inverseSqrt(tot / f32(dim) + params.eps);')}

  let inv_std = s_inv_std;
  for (var i: u32 = lid.x; i < dim; i += 256u) {
    out[base + i] = ((x[base + i] - mean) * inv_std) * weight[i] + bias[i];
  }
}

@compute @workgroup_size(256)
fn gelu_erf_inplace(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i = gid.x;
  if (i >= params.a) { return; }
  let v = out[i];
  let ax = abs(v) * 0.7071067811865476;
  let t = 1.0 / (1.0 + 0.3275911 * ax);
  let poly = (((((1.061405429 * t - 1.453152027) * t) + 1.421413741) * t - 0.284496736) * t + 0.254829592) * t;
  let erf_ax = 1.0 - poly * exp(-min(ax * ax, 40.0));
  let erf_x = select(-erf_ax, erf_ax, v >= 0.0);
  out[i] = 0.5 * v * (1.0 + erf_x);
}

@compute @workgroup_size(256)
fn glu_silu_mul(@builtin(global_invocation_id) gid: vec3<u32>) {
  // out[i] = silu(x[i]) * out[i] (or gelu_tanh if params.c != 0)
  let i = gid.x;
  if (i >= params.a) { return; }
  let v = x[i];
  var act: f32;
  if (params.c != 0u) {
    let u = clamp(0.7978845608028654 * (v + 0.044715 * v * v * v), -15.0, 15.0);
    act = 0.5 * v * (1.0 + tanh(u));
  } else {
    act = v / (1.0 + exp(-clamp(v, -30.0, 30.0)));
  }
  out[i] = act * out[i];
}

@compute @workgroup_size(256)
fn add_bias_f32(@builtin(global_invocation_id) gid: vec3<u32>) {
  // out[t * M + m] += bias[m]; a = M, b = T
  let idx = gid.x;
  let tot = params.a * params.b;
  if (idx >= tot) { return; }
  out[idx] += bias[idx % params.a];
}

@compute @workgroup_size(256)
fn residual_add_f32(@builtin(global_invocation_id) gid: vec3<u32>) {
  // out[i] += x[i] * params.f0; a = count
  let i = gid.x;
  if (i >= params.a) { return; }
  out[i] += x[i] * params.f0;
}
`;
}

export function buildLinearF16Wgsl({ useSubgroups = true } = {}) {
  const smem = useSubgroups
    ? `var<workgroup> sg_partial: array<f32, 32>;`
    : `var<workgroup> row_smem: array<f32, 256>;`;

  const red = useSubgroups
    ? `  let sg_id = lid.x / sg_size;
  let sg_sum = subgroupAdd(acc);
  if (sg_lid == 0u) { sg_partial[sg_id] = sg_sum; }
  workgroupBarrier();
  if (lane == 0u && row < params.M) {
    let sg_per_row = max(1u, THREADS_PER_ROW / sg_size);
    let sg_start = local_row * sg_per_row;
    var tot: f32 = 0.0;
    for (var s: u32 = 0u; s < sg_per_row; s += 1u) { tot += sg_partial[sg_start + s]; }
    out[tok * params.M + row] = tot;
  }`
    : `  row_smem[lid.x] = acc;
  workgroupBarrier();
  for (var stride: u32 = 32u; stride > 0u; stride >>= 1u) {
    if (lane < stride) { row_smem[lid.x] += row_smem[lid.x + stride]; }
    workgroupBarrier();
  }
  if (lane == 0u && row < params.M) {
    out[tok * params.M + row] = row_smem[local_row * THREADS_PER_ROW];
  }`;

  return `${wgslHeader({ useSubgroups })}struct Lin16Params {
  M: u32,
  K: u32,
  T: u32,
  K_pad: u32,
};

@group(0) @binding(0) var<uniform> params: Lin16Params;
@group(0) @binding(1) var<storage, read> weight_f16: array<u32>;
@group(0) @binding(2) var<storage, read> x: array<f32>;
@group(0) @binding(3) var<storage, read_write> out: array<f32>;

const ROWS_PER_WG: u32 = 4u;
const THREADS_PER_ROW: u32 = 64u;
${smem}

@compute @workgroup_size(256)
fn main(${sgParams({ useSubgroups })}) {
  let tok = wid.y;
  if (tok >= params.T) { return; }
  let local_row = lid.x / THREADS_PER_ROW;
  let lane = lid.x % THREADS_PER_ROW;
  let row = wid.x * ROWS_PER_WG + local_row;
  let safe_row = min(row, params.M - 1u);

  let k_words = params.K_pad >> 1u;
  let w_off = safe_row * k_words;
  let x_off = tok * params.K;

  var acc: f32 = 0.0;
  for (var wi: u32 = lane; wi < k_words; wi += THREADS_PER_ROW) {
    let f2 = unpack2x16float(weight_f16[w_off + wi]);
    let k0 = wi << 1u;
    if (k0 < params.K) { acc += f2.x * x[x_off + k0]; }
    if (k0 + 1u < params.K) { acc += f2.y * x[x_off + k0 + 1u]; }
  }
${red}
}
`;
}

export function buildVisionConv2dWgsl() {
  return `struct VisConvParams {
  dim: u32,
  grid_h: u32,
  grid_w: u32,
  merge: u32,
};

@group(0) @binding(0) var<uniform> params: VisConvParams;
@group(0) @binding(1) var<storage, read> bcx: array<f32>;
@group(0) @binding(2) var<storage, read> conv_w: array<f32>;
@group(0) @binding(3) var<storage, read_write> out: array<f32>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let d = gid.x;
  let t = gid.y;
  let D = params.dim;
  let N = params.grid_h * params.grid_w;
  if (d >= D || t >= N) { return; }

  let m = params.merge;
  let m2 = m * m;
  let bw = params.grid_w / m;
  let blk = t / m2;
  let rem = t % m2;
  let by = blk / bw;
  let bx = blk % bw;
  let my = rem / m;
  let mx = rem % m;
  let y = i32(by * m + my);
  let x = i32(bx * m + mx);

  var acc: f32 = 0.0;
  for (var ky: i32 = -1; ky <= 1; ky += 1) {
    let ny = y + ky;
    if (ny < 0 || ny >= i32(params.grid_h)) { continue; }
    for (var kx: i32 = -1; kx <= 1; kx += 1) {
      let nx = x + kx;
      if (nx < 0 || nx >= i32(params.grid_w)) { continue; }
      let uy = u32(ny);
      let ux = u32(nx);
      let nt = (((uy / m) * bw + (ux / m)) * m + (uy % m)) * m + (ux % m);
      let n_base = nt * 3u * D;
      let gated = bcx[n_base + d] * bcx[n_base + 2u * D + d];
      let w = conv_w[d * 9u + u32((ky + 1) * 3 + (kx + 1))];
      acc += gated * w;
    }
  }
  let C = bcx[t * 3u * D + D + d];
  out[t * D + d] = C * acc;
}
`;
}

export function buildAudioConv2dWgsl() {
  return `struct Conv2dParams {
  c_out: u32,
  f_in: u32,
  t_in: u32,
  f_out: u32,
  t_out: u32,
  valid_t_out: u32,
  do_silu: u32,
  transpose_out: u32,
};

@group(0) @binding(0) var<uniform> params: Conv2dParams;
@group(0) @binding(1) var<storage, read> input: array<f32>;
@group(0) @binding(2) var<storage, read> weight: array<f32>;
@group(0) @binding(3) var<storage, read> bias: array<f32>;
@group(0) @binding(4) var<storage, read_write> out: array<f32>;

fn silu(v: f32) -> f32 {
  return v / (1.0 + exp(-clamp(v, -30.0, 30.0)));
}

@compute @workgroup_size(256)
fn conv2d_in_s2(@builtin(global_invocation_id) gid: vec3<u32>) {
  // input: [1, f_in, t_in], weight: [c_out, 9], out: [c_out, f_out, t_out]
  let idx = gid.x;
  let total = params.c_out * params.f_out * params.t_out;
  if (idx >= total) { return; }
  let ot = idx % params.t_out;
  let rem = idx / params.t_out;
  let out_f = rem % params.f_out;
  let oc = rem / params.f_out;

  if (ot >= params.valid_t_out) {
    out[idx] = 0.0;
    return;
  }

  var acc: f32 = bias[oc];
  let w_base = oc * 9u;
  for (var ky: i32 = 0; ky < 3; ky += 1) {
    let iy = i32(out_f * 2u) + ky - 1;
    if (iy < 0 || iy >= i32(params.f_in)) { continue; }
    for (var kx: i32 = 0; kx < 3; kx += 1) {
      let ix = i32(ot * 2u) + kx - 1;
      if (ix < 0 || ix >= i32(params.t_in)) { continue; }
      acc += input[u32(iy) * params.t_in + u32(ix)] * weight[w_base + u32(ky * 3 + kx)];
    }
  }
  out[idx] = select(acc, silu(acc), params.do_silu != 0u);
}

@compute @workgroup_size(256)
fn conv2d_dw_s2(@builtin(global_invocation_id) gid: vec3<u32>) {
  // depthwise Conv2d k=3, s=2, pad=1: [C, f_in, t_in] -> [C, f_out, t_out]
  let idx = gid.x;
  let total = params.c_out * params.f_out * params.t_out;
  if (idx >= total) { return; }
  let ot = idx % params.t_out;
  let rem = idx / params.t_out;
  let out_f = rem % params.f_out;
  let oc = rem / params.f_out;

  var acc: f32 = bias[oc];
  let in_base = oc * params.f_in * params.t_in;
  let w_base = oc * 9u;
  for (var ky: i32 = 0; ky < 3; ky += 1) {
    let iy = i32(out_f * 2u) + ky - 1;
    if (iy < 0 || iy >= i32(params.f_in)) { continue; }
    for (var kx: i32 = 0; kx < 3; kx += 1) {
      let ix = i32(ot * 2u) + kx - 1;
      if (ix < 0 || ix >= i32(params.t_in)) { continue; }
      acc += input[in_base + u32(iy) * params.t_in + u32(ix)] * weight[w_base + u32(ky * 3 + kx)];
    }
  }
  out[idx] = select(acc, silu(acc), params.do_silu != 0u);
}

@compute @workgroup_size(256)
fn conv2d_pw_1x1(@builtin(global_invocation_id) gid: vec3<u32>) {
  // 1x1 Conv2d [c_in(=f_in), f_out(=F), t_out(=T)] -> [c_out, F, T] (or transposed [T, c_out * F])
  let idx = gid.x;
  let total = params.c_out * params.f_out * params.t_out;
  if (idx >= total) { return; }
  let ot = idx % params.t_out;
  let rem = idx / params.t_out;
  let out_f = rem % params.f_out;
  let oc = rem / params.f_out;

  var val: f32 = 0.0;
  if (ot < params.valid_t_out) {
    let c_in = params.f_in;
    let ft = params.f_out * params.t_out;
    let px_off = out_f * params.t_out + ot;
    let w_base = oc * c_in;
    var acc: f32 = bias[oc];
    for (var ic: u32 = 0u; ic < c_in; ic += 1u) {
      acc += input[ic * ft + px_off] * weight[w_base + ic];
    }
    val = select(acc, silu(acc), params.do_silu != 0u);
  }

  if (params.transpose_out != 0u) {
    let flat_c = params.c_out * params.f_out;
    out[ot * flat_c + oc * params.f_out + out_f] = val;
  } else {
    out[idx] = val;
  }
}
`;
}

export function buildFlexRopeWgsl() {
  return `struct FlexRopeParams {
  T: u32,
  q_heads: u32,
  kv_heads: u32,
  head_dim: u32,
  rope_mode: u32, // 0=none, 1=1D, 2=2D axial (spatial_dim=head_dim/2)
  theta: f32,
  _pad0: u32,
  _pad1: u32,
};

@group(0) @binding(0) var<uniform> params: FlexRopeParams;
@group(0) @binding(1) var<storage, read> qkv: array<f32>;
@group(0) @binding(2) var<storage, read> pos_hw: array<i32>;
@group(0) @binding(3) var<storage, read_write> q_rot: array<f32>;
@group(0) @binding(4) var<storage, read_write> k_rot: array<f32>;

fn rot_angle(tok: u32, d: u32) -> vec2<f32> {
  // Returns (cos, sin) and mate_sign implicitly via caller
  let hd = params.head_dim;
  let half_hd = hd >> 1u;
  if (params.rope_mode == 1u) {
    let pair = d % half_hd;
    let inv_freq = pow(params.theta, -f32(pair * 2u) / f32(hd));
    let ang = f32(tok) * inv_freq;
    return vec2<f32>(cos(ang), sin(ang));
  }
  // 2D axial: spatial_dim = half_hd; freq_h[0..qhd-1], freq_w[0..qhd-1]
  let q_hd = half_hd >> 1u;
  let within = d % half_hd;
  let is_w = within >= q_hd;
  let coord = f32(pos_hw[tok * 2u + select(0u, 1u, is_w)]);
  let pair = within % q_hd;
  let inv_freq = pow(params.theta, -f32(pair * 2u) / f32(half_hd));
  let ang = coord * inv_freq;
  return vec2<f32>(cos(ang), sin(ang));
}

@compute @workgroup_size(128)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let q_dim = params.q_heads * params.head_dim;
  let kv_dim = params.kv_heads * params.head_dim;
  let total_qkv = q_dim + 2u * kv_dim;
  let tok = gid.y;
  let idx = gid.x;
  if (tok >= params.T) { return; }

  let half_hd = params.head_dim >> 1u;
  let qkv_base = tok * total_qkv;

  if (idx < q_dim) {
    let h = idx / params.head_dim;
    let d = idx % params.head_dim;
    var val = qkv[qkv_base + idx];
    if (params.rope_mode != 0u) {
      let cs = rot_angle(tok, d);
      let is_hi = d >= half_hd;
      let mate_d = select(d + half_hd, d - half_hd, is_hi);
      let mate = qkv[qkv_base + h * params.head_dim + mate_d];
      val = select(val * cs.x - mate * cs.y, val * cs.x + mate * cs.y, is_hi);
    }
    q_rot[tok * q_dim + idx] = val;
  }

  if (idx < kv_dim) {
    let h = idx / params.head_dim;
    let d = idx % params.head_dim;
    var val = qkv[qkv_base + q_dim + idx];
    if (params.rope_mode != 0u) {
      let cs = rot_angle(tok, d);
      let is_hi = d >= half_hd;
      let mate_d = select(d + half_hd, d - half_hd, is_hi);
      let mate = qkv[qkv_base + q_dim + h * params.head_dim + mate_d];
      val = select(val * cs.x - mate * cs.y, val * cs.x + mate * cs.y, is_hi);
    }
    k_rot[tok * kv_dim + idx] = val;
  }
}
`;
}

export function buildFlexAttentionWgsl({ useSubgroups = true } = {}) {
  const smem = useSubgroups
    ? `var<workgroup> sg_sum_buf: array<f32, 64>;\nvar<workgroup> shared_score: f32;`
    : `var<workgroup> wg_red: array<f32, 256>;\nvar<workgroup> shared_score: f32;`;

  const redScore = useSubgroups
    ? `    let sg_id = lid.x / sg_size;
    let num_sg = (256u + sg_size - 1u) / sg_size;
    let sg_dot = subgroupAdd(prod);
    if (sg_lid == 0u) { sg_sum_buf[sg_id] = sg_dot; }
    workgroupBarrier();
    if (lid.x == 0u) {
      var dot_sum: f32 = 0.0;
      for (var s: u32 = 0u; s < num_sg; s += 1u) { dot_sum += sg_sum_buf[s]; }
      shared_score = dot_sum * params.scale;
    }
    workgroupBarrier();`
    : `    wg_red[lid.x] = prod;
    workgroupBarrier();
    for (var stride: u32 = 128u; stride > 0u; stride >>= 1u) {
      if (lid.x < stride) { wg_red[lid.x] += wg_red[lid.x + stride]; }
      workgroupBarrier();
    }
    if (lid.x == 0u) { shared_score = wg_red[0] * params.scale; }
    workgroupBarrier();`;

  return `${wgslHeader({ useSubgroups })}struct FlexAttnParams {
  T: u32,
  q_heads: u32,
  kv_heads: u32,
  head_dim: u32,
  is_causal: u32,
  sliding_window: u32,
  has_gate: u32,
  scale: f32,
};

@group(0) @binding(0) var<uniform> params: FlexAttnParams;
@group(0) @binding(1) var<storage, read> q_rot: array<f32>;
@group(0) @binding(2) var<storage, read> k_rot: array<f32>;
@group(0) @binding(3) var<storage, read> qkv: array<f32>; // V read directly from q_dim + kv_dim offset
@group(0) @binding(4) var<storage, read> attn_gate: array<f32>;
@group(0) @binding(5) var<storage, read_write> out: array<f32>;

${smem}

@compute @workgroup_size(256)
fn main(${sgParams({ useSubgroups })}) {
  let tok = wid.y;
  let qh = wid.x;
  if (tok >= params.T || qh >= params.q_heads) { return; }

  let d0 = lid.x;
  let act0 = d0 < params.head_dim;
  let q_dim = params.q_heads * params.head_dim;
  let kv_dim = params.kv_heads * params.head_dim;
  let qkv_stride = q_dim + 2u * kv_dim;
  let kv_h = qh / (params.q_heads / params.kv_heads);

  let q_off = tok * q_dim + qh * params.head_dim;
  let q0 = select(0.0, q_rot[q_off + d0], act0);

  var kv_start: u32 = 0u;
  var kv_end: u32 = params.T;
  if (params.is_causal != 0u) {
    kv_end = tok + 1u;
    if (params.sliding_window > 0u && kv_end > params.sliding_window) {
      kv_start = kv_end - params.sliding_window;
    }
  } else if (params.sliding_window > 0u) {
    // bidirectional sliding window if needed
    kv_start = select(0u, tok - min(tok, params.sliding_window), tok > params.sliding_window);
    kv_end = min(params.T, tok + params.sliding_window + 1u);
  }

  var m_i: f32 = -3.402823e38;
  var l_i: f32 = 0.0;
  var acc0: f32 = 0.0;

  for (var p: u32 = kv_start; p < kv_end; p += 1u) {
    let k_off = p * kv_dim + kv_h * params.head_dim;
    let v_off = p * qkv_stride + q_dim + kv_dim + kv_h * params.head_dim;
    var prod: f32 = 0.0;
    if (act0) { prod = q0 * k_rot[k_off + d0]; }

${redScore}

    let score = shared_score;
    let m_new = max(m_i, score);
    let alpha = exp(m_i - m_new);
    let beta = exp(score - m_new);
    l_i = l_i * alpha + beta;
    if (act0) { acc0 = acc0 * alpha + beta * qkv[v_off + d0]; }
    m_i = m_new;
  }

  var g_scale: f32 = 1.0;
  if (params.has_gate != 0u) {
    let g = attn_gate[tok * params.q_heads + qh];
    g_scale = 1.0 / (1.0 + exp(-g));
  }
  if (act0) {
    out[q_off + d0] = (acc0 / l_i) * g_scale;
  }
}
`;
}
