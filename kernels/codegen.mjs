/**
 * Unified W3C-Baseline WGSL Kernel Codegen for Ignis Framework (Phase 5 & 6).
 *
 * Strictly avoids non-baseline subgroup builtins (no `subgroup_id` or `num_subgroups` attributes);
 * supports both standard WebGPU `subgroups` and pure shared-memory fallback (`useSubgroups: false`).
 */

function wgslHeader({ f16 = false, useSubgroups = true } = {}) {
  const lines = [];
  if (f16) lines.push('enable f16;');
  if (useSubgroups) lines.push('enable subgroups;');
  return lines.join('\n') + (lines.length ? '\n\n' : '');
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

export function buildRmsNormWgsl({ useSubgroups = true } = {}) {
  const smemDecl = useSubgroups
    ? `var<workgroup> sg_partial: array<f32, 64>;\nvar<workgroup> inv_rms_shared: f32;`
    : `var<workgroup> wg_red: array<f32, 256>;\nvar<workgroup> inv_rms_shared: f32;`;

  const reduceSum = (varName, dimExpr, addResidualScalar) => useSubgroups
    ? `  let sg_id = lid.x / sg_size;
  let num_sg = (WG_SIZE + sg_size - 1u) / sg_size;
  let sg_sum = subgroupAdd(${varName});
  if (sg_lid == 0u) { sg_partial[sg_id] = sg_sum; }
  workgroupBarrier();
  if (lid.x == 0u) {
    var total: f32 = 0.0;
    for (var s: u32 = 0u; s < num_sg; s += 1u) { total += sg_partial[s]; }
    inv_rms_shared = inverseSqrt(total / f32(${dimExpr}) + params.eps);
  }
  workgroupBarrier();`
    : `  wg_red[lid.x] = ${varName};
  workgroupBarrier();
  for (var stride: u32 = 128u; stride > 0u; stride >>= 1u) {
    if (lid.x < stride) { wg_red[lid.x] += wg_red[lid.x + stride]; }
    workgroupBarrier();
  }
  if (lid.x == 0u) {
    inv_rms_shared = inverseSqrt(wg_red[0] / f32(${dimExpr}) + params.eps);
  }
  workgroupBarrier();`;

  return `${wgslHeader({ useSubgroups })}struct Params {
  dim: u32,
  tokens: u32,
  eps: f32,
  scalar: f32,
};

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> x: array<f32>;
@group(0) @binding(2) var<storage, read> weight: array<f32>;
@group(0) @binding(3) var<storage, read_write> out: array<f32>;

const WG_SIZE: u32 = 256u;
${smemDecl}

@compute @workgroup_size(256)
fn main(
${sgParams({ useSubgroups })}
) {
  let tok = wid.x;
  if (tok >= params.tokens) { return; }
  let base = tok * params.dim;

  var sum_sq: f32 = 0.0;
  for (var i: u32 = lid.x; i < params.dim; i += WG_SIZE) {
    let v = x[base + i];
    sum_sq += v * v;
  }

${reduceSum('sum_sq', 'params.dim', false)}

  let inv_rms = inv_rms_shared;
  for (var i: u32 = lid.x; i < params.dim; i += WG_SIZE) {
    out[base + i] = x[base + i] * inv_rms * weight[i];
  }
}

@compute @workgroup_size(256)
fn post_norm_residual(
${sgParams({ useSubgroups })}
) {
  let tok = wid.x;
  if (tok >= params.tokens) { return; }
  let base = tok * params.dim;

  var sum_sq: f32 = 0.0;
  for (var i: u32 = lid.x; i < params.dim; i += WG_SIZE) {
    let v = x[base + i];
    sum_sq += v * v;
  }

${reduceSum('sum_sq', 'params.dim', true)}

  let inv_rms = inv_rms_shared;
  for (var i: u32 = lid.x; i < params.dim; i += WG_SIZE) {
    out[base + i] = (out[base + i] + x[base + i] * inv_rms * weight[i]) * params.scalar;
  }
}

@compute @workgroup_size(256)
fn post_norm_gate_ple_residual(
${sgParams({ useSubgroups })}
) {
  // For Gemma 4 PLE step: residual += norm(x_ple_proj) and optional layer_scalar multiply; params.scalar = layer_scalar
  let tok = wid.x;
  if (tok >= params.tokens) { return; }
  let base = tok * params.dim;

  var sum_sq: f32 = 0.0;
  for (var i: u32 = lid.x; i < params.dim; i += WG_SIZE) {
    let v = x[base + i];
    sum_sq += v * v;
  }

${reduceSum('sum_sq', 'params.dim', true)}

  let inv_rms = inv_rms_shared;
  for (var i: u32 = lid.x; i < params.dim; i += WG_SIZE) {
    out[base + i] = (out[base + i] + x[base + i] * inv_rms * weight[i]) * params.scalar;
  }
}
`;
}

export function buildLinearF32Wgsl({ useSubgroups = true } = {}) {
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
    var total: f32 = 0.0;
    for (var s: u32 = 0u; s < sg_per_row; s += 1u) { total += sg_partial[sg_start + s]; }
    out[tok * params.M + row] = total;
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

  return `${wgslHeader({ useSubgroups })}struct LinearParams {
  M: u32,
  K: u32,
  T: u32,
  _pad: u32,
};

@group(0) @binding(0) var<uniform> params: LinearParams;
@group(0) @binding(1) var<storage, read> weight: array<f32>;
@group(0) @binding(2) var<storage, read> x: array<f32>;
@group(0) @binding(3) var<storage, read_write> out: array<f32>;

const ROWS_PER_WG: u32 = 4u;
const THREADS_PER_ROW: u32 = 64u;
${smem}

@compute @workgroup_size(256)
fn linear_f32(
${sgParams({ useSubgroups })}
) {
  let tok = wid.y;
  if (tok >= params.T) { return; }
  let local_row = lid.x / THREADS_PER_ROW;
  let lane = lid.x % THREADS_PER_ROW;
  let row = wid.x * ROWS_PER_WG + local_row;
  let safe_row = min(row, params.M - 1u);

  let w_off = safe_row * params.K;
  let x_off = tok * params.K;

  var acc: f32 = 0.0;
  let k_vec4 = params.K / 4u;
  for (var idx4: u32 = lane; idx4 < k_vec4; idx4 += THREADS_PER_ROW) {
    let k = idx4 * 4u;
    let wv = vec4<f32>(weight[w_off + k], weight[w_off + k + 1u], weight[w_off + k + 2u], weight[w_off + k + 3u]);
    let xv = vec4<f32>(x[x_off + k], x[x_off + k + 1u], x[x_off + k + 2u], x[x_off + k + 3u]);
    acc += dot(wv, xv);
  }
  let tail_start = k_vec4 * 4u;
  for (var k: u32 = tail_start + lane; k < params.K; k += THREADS_PER_ROW) {
    acc += weight[w_off + k] * x[x_off + k];
  }

${red}
}
`;
}

function rowReduceAssign({ useSubgroups, vecType, varName, destExpr, addAssign }) {
  const op = addAssign ? '+=' : '=';
  if (useSubgroups) {
    const arr = vecType === 'f32' ? 'sg_partial_f32' : 'sg_partial_v4';
    return `  let sg_id = lid.x / sg_size;
  let sg_sum = subgroupAdd(${varName});
  if (sg_lid == 0u) { ${arr}[sg_id] = sg_sum; }
  workgroupBarrier();
  if (lane == 0u && row < params.M) {
    let sg_per_row = max(1u, THREADS_PER_ROW / sg_size);
    let sg_start = local_row * sg_per_row;
    var tot = ${vecType === 'f32' ? '0.0' : 'vec4<f32>(0.0)'};
    for (var s: u32 = 0u; s < sg_per_row; s += 1u) { tot += ${arr}[sg_start + s]; }
    ${destExpr}
  }`;
  }
  const arr = vecType === 'f32' ? 'wg_row_f32' : 'wg_row_v4';
  return `  ${arr}[lid.x] = ${varName};
  workgroupBarrier();
  for (var stride: u32 = 32u; stride > 0u; stride >>= 1u) {
    if (lane < stride) { ${arr}[lid.x] += ${arr}[lid.x + stride]; }
    workgroupBarrier();
  }
  if (lane == 0u && row < params.M) {
    let tot = ${arr}[local_row * THREADS_PER_ROW];
    ${destExpr}
  }`;
}

export function buildQuantMatmulWgsl({ useSubgroups = true } = {}) {
  const smem = useSubgroups
    ? `var<workgroup> sg_partial_f32: array<f32, 32>;\nvar<workgroup> sg_partial_v4: array<vec4<f32>, 32>;`
    : `var<workgroup> wg_row_f32: array<f32, 256>;\nvar<workgroup> wg_row_v4: array<vec4<f32>, 256>;`;

  const red1 = (add) => rowReduceAssign({
    useSubgroups,
    vecType: 'f32',
    varName: 'acc',
    destExpr: add ? 'out[tok * params.M + row] += tot;' : 'out[tok * params.M + row] = tot;',
    addAssign: add,
  });

  const red4 = (add) => rowReduceAssign({
    useSubgroups,
    vecType: 'vec4<f32>',
    varName: 'acc',
    destExpr: add
      ? `out[t0 * params.M + row] += tot.x;
    if (t0 + 1u < params.T) { out[(t0 + 1u) * params.M + row] += tot.y; }
    if (t0 + 2u < params.T) { out[(t0 + 2u) * params.M + row] += tot.z; }
    if (t0 + 3u < params.T) { out[(t0 + 3u) * params.M + row] += tot.w; }`
      : `out[t0 * params.M + row] = tot.x;
    if (t0 + 1u < params.T) { out[(t0 + 1u) * params.M + row] = tot.y; }
    if (t0 + 2u < params.T) { out[(t0 + 2u) * params.M + row] = tot.z; }
    if (t0 + 3u < params.T) { out[(t0 + 3u) * params.M + row] = tot.w; }`,
    addAssign: add,
  });

  return `${wgslHeader({ useSubgroups })}struct Params {
  M: u32,
  K: u32,
  T: u32,
  K_pad: u32,
};

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> quants: array<u32>;
@group(0) @binding(2) var<storage, read> scales: array<u32>;
@group(0) @binding(3) var<storage, read> x: array<f32>;
@group(0) @binding(4) var<storage, read_write> out: array<f32>;

const ROWS_PER_WG: u32 = 4u;
const THREADS_PER_ROW: u32 = 64u;
${smem}

fn q4_gemv_dot(row: u32, lane: u32, tok: u32) -> f32 {
  let safe_row = min(row, params.M - 1u);
  let k_words = params.K_pad >> 3u;
  let k_groups = params.K_pad >> 6u;
  let row_q_off = safe_row * k_words;
  let row_s_off = safe_row * k_groups;
  let x_off = tok * params.K;

  var acc: f32 = 0.0;
  for (var w_idx: u32 = lane; w_idx < k_words; w_idx += THREADS_PER_ROW) {
    let word = quants[row_q_off + w_idx];
    let sm = unpack2x16float(scales[row_s_off + (w_idx >> 3u)]);
    let k_base = w_idx << 3u;

    var xv0 = vec4<f32>(0.0);
    var xv1 = vec4<f32>(0.0);
    if (k_base + 7u < params.K) {
      xv0 = vec4<f32>(x[x_off + k_base], x[x_off + k_base + 1u], x[x_off + k_base + 2u], x[x_off + k_base + 3u]);
      xv1 = vec4<f32>(x[x_off + k_base + 4u], x[x_off + k_base + 5u], x[x_off + k_base + 6u], x[x_off + k_base + 7u]);
    } else {
      for (var j: u32 = 0u; j < 4u; j += 1u) {
        if (k_base + j < params.K) { xv0[j] = x[x_off + k_base + j]; }
        if (k_base + 4u + j < params.K) { xv1[j] = x[x_off + k_base + 4u + j]; }
      }
    }
    let q0 = vec4<f32>(f32(word & 0xFu), f32((word >> 4u) & 0xFu), f32((word >> 8u) & 0xFu), f32((word >> 12u) & 0xFu));
    let q1 = vec4<f32>(f32((word >> 16u) & 0xFu), f32((word >> 20u) & 0xFu), f32((word >> 24u) & 0xFu), f32((word >> 28u) & 0xFu));
    acc += dot(q0 * sm.x + sm.y, xv0) + dot(q1 * sm.x + sm.y, xv1);
  }
  return acc;
}

fn q4_gemm4_dot(row: u32, lane: u32, t0: u32) -> vec4<f32> {
  let safe_row = min(row, params.M - 1u);
  let k_words = params.K_pad >> 3u;
  let k_groups = params.K_pad >> 6u;
  let row_q_off = safe_row * k_words;
  let row_s_off = safe_row * k_groups;

  let t1 = min(t0 + 1u, params.T - 1u);
  let t2 = min(t0 + 2u, params.T - 1u);
  let t3 = min(t0 + 3u, params.T - 1u);

  var acc = vec4<f32>(0.0);
  for (var w_idx: u32 = lane; w_idx < k_words; w_idx += THREADS_PER_ROW) {
    let word = quants[row_q_off + w_idx];
    let sm = unpack2x16float(scales[row_s_off + (w_idx >> 3u)]);
    let k_base = w_idx << 3u;

    let w0 = vec4<f32>(f32(word & 0xFu), f32((word >> 4u) & 0xFu), f32((word >> 8u) & 0xFu), f32((word >> 12u) & 0xFu)) * sm.x + sm.y;
    let w1 = vec4<f32>(f32((word >> 16u) & 0xFu), f32((word >> 20u) & 0xFu), f32((word >> 24u) & 0xFu), f32((word >> 28u) & 0xFu)) * sm.x + sm.y;

    if (k_base + 7u < params.K) {
      let b0 = t0 * params.K + k_base;
      let b1 = t1 * params.K + k_base;
      let b2 = t2 * params.K + k_base;
      let b3 = t3 * params.K + k_base;
      acc.x += dot(w0, vec4<f32>(x[b0], x[b0+1u], x[b0+2u], x[b0+3u])) + dot(w1, vec4<f32>(x[b0+4u], x[b0+5u], x[b0+6u], x[b0+7u]));
      acc.y += dot(w0, vec4<f32>(x[b1], x[b1+1u], x[b1+2u], x[b1+3u])) + dot(w1, vec4<f32>(x[b1+4u], x[b1+5u], x[b1+6u], x[b1+7u]));
      acc.z += dot(w0, vec4<f32>(x[b2], x[b2+1u], x[b2+2u], x[b2+3u])) + dot(w1, vec4<f32>(x[b2+4u], x[b2+5u], x[b2+6u], x[b2+7u]));
      acc.w += dot(w0, vec4<f32>(x[b3], x[b3+1u], x[b3+2u], x[b3+3u])) + dot(w1, vec4<f32>(x[b3+4u], x[b3+5u], x[b3+6u], x[b3+7u]));
    } else {
      for (var j: u32 = 0u; j < 4u; j += 1u) {
        if (k_base + j < params.K) {
          acc.x += w0[j] * x[t0 * params.K + k_base + j];
          acc.y += w0[j] * x[t1 * params.K + k_base + j];
          acc.z += w0[j] * x[t2 * params.K + k_base + j];
          acc.w += w0[j] * x[t3 * params.K + k_base + j];
        }
        if (k_base + 4u + j < params.K) {
          acc.x += w1[j] * x[t0 * params.K + k_base + 4u + j];
          acc.y += w1[j] * x[t1 * params.K + k_base + 4u + j];
          acc.z += w1[j] * x[t2 * params.K + k_base + 4u + j];
          acc.w += w1[j] * x[t3 * params.K + k_base + 4u + j];
        }
      }
    }
  }
  return acc;
}

@compute @workgroup_size(256)
fn gemv_q4_g64(${sgParams({ useSubgroups })}) {
  let tok = wid.y;
  if (tok >= params.T) { return; }
  let local_row = lid.x / THREADS_PER_ROW;
  let lane = lid.x % THREADS_PER_ROW;
  let row = wid.x * ROWS_PER_WG + local_row;
  let acc = q4_gemv_dot(row, lane, tok);
${red1(false)}
}

@compute @workgroup_size(256)
fn gemv_q4_g64_add(${sgParams({ useSubgroups })}) {
  let tok = wid.y;
  if (tok >= params.T) { return; }
  let local_row = lid.x / THREADS_PER_ROW;
  let lane = lid.x % THREADS_PER_ROW;
  let row = wid.x * ROWS_PER_WG + local_row;
  let acc = q4_gemv_dot(row, lane, tok);
${red1(true)}
}

@compute @workgroup_size(256)
fn gemm4_q4_g64(${sgParams({ useSubgroups })}) {
  let t0 = wid.y * 4u;
  if (t0 >= params.T) { return; }
  let local_row = lid.x / THREADS_PER_ROW;
  let lane = lid.x % THREADS_PER_ROW;
  let row = wid.x * ROWS_PER_WG + local_row;
  let acc = q4_gemm4_dot(row, lane, t0);
${red4(false)}
}

@compute @workgroup_size(256)
fn gemm4_q4_g64_add(${sgParams({ useSubgroups })}) {
  let t0 = wid.y * 4u;
  if (t0 >= params.T) { return; }
  let local_row = lid.x / THREADS_PER_ROW;
  let lane = lid.x % THREADS_PER_ROW;
  let row = wid.x * ROWS_PER_WG + local_row;
  let acc = q4_gemm4_dot(row, lane, t0);
${red4(true)}
}

fn q3_unpack_chunk(w0: u32, w1: u32, w2: u32, sm: vec2<f32>, out_w: ptr<function, array<vec4<f32>, 8>>) {
  (*out_w)[0] = vec4<f32>(f32(w0 & 7u), f32((w0 >> 3u) & 7u), f32((w0 >> 6u) & 7u), f32((w0 >> 9u) & 7u)) * sm.x + sm.y;
  (*out_w)[1] = vec4<f32>(f32((w0 >> 12u) & 7u), f32((w0 >> 15u) & 7u), f32((w0 >> 18u) & 7u), f32((w0 >> 21u) & 7u)) * sm.x + sm.y;
  (*out_w)[2] = vec4<f32>(f32((w0 >> 24u) & 7u), f32((w0 >> 27u) & 7u), f32(((w0 >> 30u) & 3u) | ((w1 & 1u) << 2u)), f32((w1 >> 1u) & 7u)) * sm.x + sm.y;
  (*out_w)[3] = vec4<f32>(f32((w1 >> 4u) & 7u), f32((w1 >> 7u) & 7u), f32((w1 >> 10u) & 7u), f32((w1 >> 13u) & 7u)) * sm.x + sm.y;
  (*out_w)[4] = vec4<f32>(f32((w1 >> 16u) & 7u), f32((w1 >> 19u) & 7u), f32((w1 >> 22u) & 7u), f32((w1 >> 25u) & 7u)) * sm.x + sm.y;
  (*out_w)[5] = vec4<f32>(f32((w1 >> 28u) & 7u), f32(((w1 >> 31u) & 1u) | ((w2 & 3u) << 1u)), f32((w2 >> 2u) & 7u), f32((w2 >> 5u) & 7u)) * sm.x + sm.y;
  (*out_w)[6] = vec4<f32>(f32((w2 >> 8u) & 7u), f32((w2 >> 11u) & 7u), f32((w2 >> 14u) & 7u), f32((w2 >> 17u) & 7u)) * sm.x + sm.y;
  (*out_w)[7] = vec4<f32>(f32((w2 >> 20u) & 7u), f32((w2 >> 23u) & 7u), f32((w2 >> 26u) & 7u), f32((w2 >> 29u) & 7u)) * sm.x + sm.y;
}

fn q3_gemv_dot(row: u32, lane: u32, tok: u32) -> f32 {
  let safe_row = min(row, params.M - 1u);
  let n_chunks = params.K_pad >> 5u;
  let k_words = n_chunks * 3u;
  let k_groups = params.K_pad >> 6u;
  let row_q_off = safe_row * k_words;
  let row_s_off = safe_row * k_groups;
  let x_off = tok * params.K;

  var acc: f32 = 0.0;
  var uw: array<vec4<f32>, 8>;
  for (var c: u32 = lane; c < n_chunks; c += THREADS_PER_ROW) {
    let q_off = row_q_off + c * 3u;
    let sm = unpack2x16float(scales[row_s_off + (c >> 1u)]);
    q3_unpack_chunk(quants[q_off], quants[q_off + 1u], quants[q_off + 2u], sm, &uw);
    let k_base = c << 5u;
    if (k_base + 31u < params.K) {
      for (var g: u32 = 0u; g < 8u; g += 1u) {
        let kb = x_off + k_base + g * 4u;
        acc += dot(uw[g], vec4<f32>(x[kb], x[kb + 1u], x[kb + 2u], x[kb + 3u]));
      }
    } else {
      for (var g: u32 = 0u; g < 8u; g += 1u) {
        let g_vec = uw[g];
        let kb = k_base + g * 4u;
        for (var j: u32 = 0u; j < 4u; j += 1u) {
          if (kb + j < params.K) { acc += g_vec[j] * x[x_off + kb + j]; }
        }
      }
    }
  }
  return acc;
}

fn q3_gemm4_dot(row: u32, lane: u32, t0: u32) -> vec4<f32> {
  let safe_row = min(row, params.M - 1u);
  let n_chunks = params.K_pad >> 5u;
  let k_words = n_chunks * 3u;
  let k_groups = params.K_pad >> 6u;
  let row_q_off = safe_row * k_words;
  let row_s_off = safe_row * k_groups;

  let t1 = min(t0 + 1u, params.T - 1u);
  let t2 = min(t0 + 2u, params.T - 1u);
  let t3 = min(t0 + 3u, params.T - 1u);

  var acc = vec4<f32>(0.0);
  var uw: array<vec4<f32>, 8>;
  for (var c: u32 = lane; c < n_chunks; c += THREADS_PER_ROW) {
    let q_off = row_q_off + c * 3u;
    let sm = unpack2x16float(scales[row_s_off + (c >> 1u)]);
    q3_unpack_chunk(quants[q_off], quants[q_off + 1u], quants[q_off + 2u], sm, &uw);
    let k_base = c << 5u;
    if (k_base + 31u < params.K) {
      for (var g: u32 = 0u; g < 8u; g += 1u) {
        let g_vec = uw[g];
        let kb = k_base + g * 4u;
        let b0 = t0 * params.K + kb;
        let b1 = t1 * params.K + kb;
        let b2 = t2 * params.K + kb;
        let b3 = t3 * params.K + kb;
        acc.x += dot(g_vec, vec4<f32>(x[b0], x[b0+1u], x[b0+2u], x[b0+3u]));
        acc.y += dot(g_vec, vec4<f32>(x[b1], x[b1+1u], x[b1+2u], x[b1+3u]));
        acc.z += dot(g_vec, vec4<f32>(x[b2], x[b2+1u], x[b2+2u], x[b2+3u]));
        acc.w += dot(g_vec, vec4<f32>(x[b3], x[b3+1u], x[b3+2u], x[b3+3u]));
      }
    } else {
      for (var g: u32 = 0u; g < 8u; g += 1u) {
        let g_vec = uw[g];
        let kb = k_base + g * 4u;
        for (var j: u32 = 0u; j < 4u; j += 1u) {
          if (kb + j < params.K) {
            acc.x += g_vec[j] * x[t0 * params.K + kb + j];
            acc.y += g_vec[j] * x[t1 * params.K + kb + j];
            acc.z += g_vec[j] * x[t2 * params.K + kb + j];
            acc.w += g_vec[j] * x[t3 * params.K + kb + j];
          }
        }
      }
    }
  }
  return acc;
}

@compute @workgroup_size(256)
fn gemv_q3_g64(${sgParams({ useSubgroups })}) {
  let tok = wid.y;
  if (tok >= params.T) { return; }
  let local_row = lid.x / THREADS_PER_ROW;
  let lane = lid.x % THREADS_PER_ROW;
  let row = wid.x * ROWS_PER_WG + local_row;
  let acc = q3_gemv_dot(row, lane, tok);
${red1(false)}
}

@compute @workgroup_size(256)
fn gemv_q3_g64_add(${sgParams({ useSubgroups })}) {
  let tok = wid.y;
  if (tok >= params.T) { return; }
  let local_row = lid.x / THREADS_PER_ROW;
  let lane = lid.x % THREADS_PER_ROW;
  let row = wid.x * ROWS_PER_WG + local_row;
  let acc = q3_gemv_dot(row, lane, tok);
${red1(true)}
}

@compute @workgroup_size(256)
fn gemm4_q3_g64(${sgParams({ useSubgroups })}) {
  let t0 = wid.y * 4u;
  if (t0 >= params.T) { return; }
  let local_row = lid.x / THREADS_PER_ROW;
  let lane = lid.x % THREADS_PER_ROW;
  let row = wid.x * ROWS_PER_WG + local_row;
  let acc = q3_gemm4_dot(row, lane, t0);
${red4(false)}
}

@compute @workgroup_size(256)
fn gemm4_q3_g64_add(${sgParams({ useSubgroups })}) {
  let t0 = wid.y * 4u;
  if (t0 >= params.T) { return; }
  let local_row = lid.x / THREADS_PER_ROW;
  let lane = lid.x % THREADS_PER_ROW;
  let row = wid.x * ROWS_PER_WG + local_row;
  let acc = q3_gemm4_dot(row, lane, t0);
${red4(true)}
}

@compute @workgroup_size(256)
fn gemv_q8_g64(${sgParams({ useSubgroups })}) {
  let tok = wid.y;
  if (tok >= params.T) { return; }
  let local_row = lid.x / THREADS_PER_ROW;
  let lane = lid.x % THREADS_PER_ROW;
  let row = wid.x * ROWS_PER_WG + local_row;
  let safe_row = min(row, params.M - 1u);

  let k_words = params.K_pad >> 2u;
  let k_groups = params.K_pad >> 6u;
  let row_q_off = safe_row * k_words;
  let row_s_off = safe_row * k_groups;
  let x_off = tok * params.K;

  var acc: f32 = 0.0;
  for (var w_idx: u32 = lane; w_idx < k_words; w_idx += THREADS_PER_ROW) {
    let word = i32(quants[row_q_off + w_idx]);
    let sm = unpack2x16float(scales[row_s_off + (w_idx >> 4u)]);
    let k_base = w_idx << 2u;

    var xv = vec4<f32>(0.0);
    if (k_base + 3u < params.K) {
      xv = vec4<f32>(x[x_off + k_base], x[x_off + k_base + 1u], x[x_off + k_base + 2u], x[x_off + k_base + 3u]);
    } else {
      for (var j: u32 = 0u; j < 4u; j += 1u) {
        if (k_base + j < params.K) { xv[j] = x[x_off + k_base + j]; }
      }
    }
    let qv = vec4<f32>(
      f32(extractBits(word, 0u, 8u)),
      f32(extractBits(word, 8u, 8u)),
      f32(extractBits(word, 16u, 8u)),
      f32(extractBits(word, 24u, 8u)),
    );
    acc += dot(qv * sm.x, xv);
  }
${red1(false)}
}

@compute @workgroup_size(256)
fn gemm4_q8_g64(${sgParams({ useSubgroups })}) {
  let t0 = wid.y * 4u;
  if (t0 >= params.T) { return; }
  let local_row = lid.x / THREADS_PER_ROW;
  let lane = lid.x % THREADS_PER_ROW;
  let row = wid.x * ROWS_PER_WG + local_row;
  let safe_row = min(row, params.M - 1u);

  let k_words = params.K_pad >> 2u;
  let k_groups = params.K_pad >> 6u;
  let row_q_off = safe_row * k_words;
  let row_s_off = safe_row * k_groups;

  let t1 = min(t0 + 1u, params.T - 1u);
  let t2 = min(t0 + 2u, params.T - 1u);
  let t3 = min(t0 + 3u, params.T - 1u);

  var acc = vec4<f32>(0.0);
  for (var w_idx: u32 = lane; w_idx < k_words; w_idx += THREADS_PER_ROW) {
    let word = i32(quants[row_q_off + w_idx]);
    let sm = unpack2x16float(scales[row_s_off + (w_idx >> 4u)]);
    let qv = vec4<f32>(
      f32(extractBits(word, 0u, 8u)),
      f32(extractBits(word, 8u, 8u)),
      f32(extractBits(word, 16u, 8u)),
      f32(extractBits(word, 24u, 8u)),
    ) * sm.x;
    let k_base = w_idx << 2u;
    if (k_base + 3u < params.K) {
      let b0 = t0 * params.K + k_base;
      let b1 = t1 * params.K + k_base;
      let b2 = t2 * params.K + k_base;
      let b3 = t3 * params.K + k_base;
      acc.x += dot(qv, vec4<f32>(x[b0], x[b0+1u], x[b0+2u], x[b0+3u]));
      acc.y += dot(qv, vec4<f32>(x[b1], x[b1+1u], x[b1+2u], x[b1+3u]));
      acc.z += dot(qv, vec4<f32>(x[b2], x[b2+1u], x[b2+2u], x[b2+3u]));
      acc.w += dot(qv, vec4<f32>(x[b3], x[b3+1u], x[b3+2u], x[b3+3u]));
    } else {
      for (var j: u32 = 0u; j < 4u; j += 1u) {
        if (k_base + j < params.K) {
          acc.x += qv[j] * x[t0 * params.K + k_base + j];
          acc.y += qv[j] * x[t1 * params.K + k_base + j];
          acc.z += qv[j] * x[t2 * params.K + k_base + j];
          acc.w += qv[j] * x[t3 * params.K + k_base + j];
        }
      }
    }
  }
${red4(false)}
}
`;
}

export function buildFusedGluWgsl(hiddenAct = 'gelu', { useSubgroups = true, weightBits = 4 } = {}) {
  let actHelper;
  if (hiddenAct === 'silu') {
    actHelper = `
fn glu_act(v: f32) -> f32 {
  return v / (1.0 + exp(-v));
}
`;
  } else if (hiddenAct === 'gelu_pytorch_tanh') {
    actHelper = `
fn glu_act(v: f32) -> f32 {
  let u = clamp(0.7978845608028654 * (v + 0.044715 * v * v * v), -15.0, 15.0);
  return 0.5 * v * (1.0 + tanh(u));
}
`;
  } else if (hiddenAct === 'gelu') {
    actHelper = `
fn erf_approx(v: f32) -> f32 {
  let sign = select(-1.0, 1.0, v >= 0.0);
  let ax = abs(v);
  let t = 1.0 / (1.0 + 0.3275911 * ax);
  let y = 1.0 - (((((1.061405429 * t - 1.453152027) * t) + 1.421413741) * t - 0.284496736) * t + 0.254829592) * t * exp(-ax * ax);
  return sign * y;
}

fn glu_act(v: f32) -> f32 {
  return 0.5 * v * (1.0 + erf_approx(v * 0.7071067811865475));
}
`;
  } else {
    throw new Error(`Unsupported hidden_act for Ignis codegen: ${hiddenAct}`);
  }

  const smem = useSubgroups
    ? `var<workgroup> sg_partial: array<vec2<f32>, 32>;
var<workgroup> sg_partial_g: array<vec4<f32>, 32>;
var<workgroup> sg_partial_u: array<vec4<f32>, 32>;`
    : `var<workgroup> wg_v2: array<vec2<f32>, 256>;
var<workgroup> wg_g4: array<vec4<f32>, 256>;
var<workgroup> wg_u4: array<vec4<f32>, 256>;`;

  const redV2 = useSubgroups
    ? `  let sg_id = lid.x / sg_size;
  let sg_sum = subgroupAdd(acc);
  if (sg_lid == 0u) { sg_partial[sg_id] = sg_sum; }
  workgroupBarrier();
  if (lane == 0u && row < params.M) {
    let sg_per_row = max(1u, THREADS_PER_ROW / sg_size);
    let sg_start = local_row * sg_per_row;
    var tot = vec2<f32>(0.0);
    for (var s: u32 = 0u; s < sg_per_row; s += 1u) { tot += sg_partial[sg_start + s]; }
    out[tok * params.M + row] = glu_act(tot.x) * tot.y;
  }`
    : `  wg_v2[lid.x] = acc;
  workgroupBarrier();
  for (var stride: u32 = 32u; stride > 0u; stride >>= 1u) {
    if (lane < stride) { wg_v2[lid.x] += wg_v2[lid.x + stride]; }
    workgroupBarrier();
  }
  if (lane == 0u && row < params.M) {
    let tot = wg_v2[local_row * THREADS_PER_ROW];
    out[tok * params.M + row] = glu_act(tot.x) * tot.y;
  }`;

  const redDualV4 = useSubgroups
    ? `  let sg_id = lid.x / sg_size;
  let sg_g = subgroupAdd(acc_g);
  let sg_u = subgroupAdd(acc_u);
  if (sg_lid == 0u) {
    sg_partial_g[sg_id] = sg_g;
    sg_partial_u[sg_id] = sg_u;
  }
  workgroupBarrier();
  if (lane == 0u && row < params.M) {
    let sg_per_row = max(1u, THREADS_PER_ROW / sg_size);
    let sg_start = local_row * sg_per_row;
    var tg = vec4<f32>(0.0);
    var tu = vec4<f32>(0.0);
    for (var s: u32 = 0u; s < sg_per_row; s += 1u) {
      tg += sg_partial_g[sg_start + s];
      tu += sg_partial_u[sg_start + s];
    }
    out[t0 * params.M + row] = glu_act(tg.x) * tu.x;
    if (t0 + 1u < params.T) { out[(t0 + 1u) * params.M + row] = glu_act(tg.y) * tu.y; }
    if (t0 + 2u < params.T) { out[(t0 + 2u) * params.M + row] = glu_act(tg.z) * tu.z; }
    if (t0 + 3u < params.T) { out[(t0 + 3u) * params.M + row] = glu_act(tg.w) * tu.w; }
  }`
    : `  wg_g4[lid.x] = acc_g;
  wg_u4[lid.x] = acc_u;
  workgroupBarrier();
  for (var stride: u32 = 32u; stride > 0u; stride >>= 1u) {
    if (lane < stride) {
      wg_g4[lid.x] += wg_g4[lid.x + stride];
      wg_u4[lid.x] += wg_u4[lid.x + stride];
    }
    workgroupBarrier();
  }
  if (lane == 0u && row < params.M) {
    let tg = wg_g4[local_row * THREADS_PER_ROW];
    let tu = wg_u4[local_row * THREADS_PER_ROW];
    out[t0 * params.M + row] = glu_act(tg.x) * tu.x;
    if (t0 + 1u < params.T) { out[(t0 + 1u) * params.M + row] = glu_act(tg.y) * tu.y; }
    if (t0 + 2u < params.T) { out[(t0 + 2u) * params.M + row] = glu_act(tg.z) * tu.z; }
    if (t0 + 3u < params.T) { out[(t0 + 3u) * params.M + row] = glu_act(tg.w) * tu.w; }
  }`;

  if (weightBits === 3) {
    return `${wgslHeader({ useSubgroups })}struct Params {
  M: u32,
  K: u32,
  T: u32,
  K_pad: u32,
};

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> gate_quants: array<u32>;
@group(0) @binding(2) var<storage, read> gate_scales: array<u32>;
@group(0) @binding(3) var<storage, read> up_quants: array<u32>;
@group(0) @binding(4) var<storage, read> up_scales: array<u32>;
@group(0) @binding(5) var<storage, read> x: array<f32>;
@group(0) @binding(6) var<storage, read_write> out: array<f32>;

const ROWS_PER_WG: u32 = 4u;
const THREADS_PER_ROW: u32 = 64u;
${smem}
${actHelper}
fn q3_unpack_chunk(w0: u32, w1: u32, w2: u32, sm: vec2<f32>, out_w: ptr<function, array<vec4<f32>, 8>>) {
  (*out_w)[0] = vec4<f32>(f32(w0 & 7u), f32((w0 >> 3u) & 7u), f32((w0 >> 6u) & 7u), f32((w0 >> 9u) & 7u)) * sm.x + sm.y;
  (*out_w)[1] = vec4<f32>(f32((w0 >> 12u) & 7u), f32((w0 >> 15u) & 7u), f32((w0 >> 18u) & 7u), f32((w0 >> 21u) & 7u)) * sm.x + sm.y;
  (*out_w)[2] = vec4<f32>(f32((w0 >> 24u) & 7u), f32((w0 >> 27u) & 7u), f32(((w0 >> 30u) & 3u) | ((w1 & 1u) << 2u)), f32((w1 >> 1u) & 7u)) * sm.x + sm.y;
  (*out_w)[3] = vec4<f32>(f32((w1 >> 4u) & 7u), f32((w1 >> 7u) & 7u), f32((w1 >> 10u) & 7u), f32((w1 >> 13u) & 7u)) * sm.x + sm.y;
  (*out_w)[4] = vec4<f32>(f32((w1 >> 16u) & 7u), f32((w1 >> 19u) & 7u), f32((w1 >> 22u) & 7u), f32((w1 >> 25u) & 7u)) * sm.x + sm.y;
  (*out_w)[5] = vec4<f32>(f32((w1 >> 28u) & 7u), f32(((w1 >> 31u) & 1u) | ((w2 & 3u) << 1u)), f32((w2 >> 2u) & 7u), f32((w2 >> 5u) & 7u)) * sm.x + sm.y;
  (*out_w)[6] = vec4<f32>(f32((w2 >> 8u) & 7u), f32((w2 >> 11u) & 7u), f32((w2 >> 14u) & 7u), f32((w2 >> 17u) & 7u)) * sm.x + sm.y;
  (*out_w)[7] = vec4<f32>(f32((w2 >> 20u) & 7u), f32((w2 >> 23u) & 7u), f32((w2 >> 26u) & 7u), f32((w2 >> 29u) & 7u)) * sm.x + sm.y;
}

@compute @workgroup_size(256)
fn gemv_glu_q3_g64(${sgParams({ useSubgroups })}) {
  let tok = wid.y;
  if (tok >= params.T) { return; }
  let local_row = lid.x / THREADS_PER_ROW;
  let lane = lid.x % THREADS_PER_ROW;
  let row = wid.x * ROWS_PER_WG + local_row;
  let safe_row = min(row, params.M - 1u);

  let n_chunks = params.K_pad >> 5u;
  let k_words = n_chunks * 3u;
  let k_groups = params.K_pad >> 6u;
  let row_q_off = safe_row * k_words;
  let row_s_off = safe_row * k_groups;
  let x_off = tok * params.K;

  var acc = vec2<f32>(0.0);
  var wg: array<vec4<f32>, 8>;
  var wu: array<vec4<f32>, 8>;
  for (var c: u32 = lane; c < n_chunks; c += THREADS_PER_ROW) {
    let q_off = row_q_off + c * 3u;
    let g_sm = unpack2x16float(gate_scales[row_s_off + (c >> 1u)]);
    let u_sm = unpack2x16float(up_scales[row_s_off + (c >> 1u)]);
    q3_unpack_chunk(gate_quants[q_off], gate_quants[q_off + 1u], gate_quants[q_off + 2u], g_sm, &wg);
    q3_unpack_chunk(up_quants[q_off], up_quants[q_off + 1u], up_quants[q_off + 2u], u_sm, &wu);
    let k_base = c << 5u;
    if (k_base + 31u < params.K) {
      for (var g: u32 = 0u; g < 8u; g += 1u) {
        let kb = x_off + k_base + g * 4u;
        let xv = vec4<f32>(x[kb], x[kb + 1u], x[kb + 2u], x[kb + 3u]);
        acc.x += dot(wg[g], xv);
        acc.y += dot(wu[g], xv);
      }
    } else {
      for (var g: u32 = 0u; g < 8u; g += 1u) {
        let kb = k_base + g * 4u;
        for (var j: u32 = 0u; j < 4u; j += 1u) {
          if (kb + j < params.K) {
            let xv = x[x_off + kb + j];
            acc.x += wg[g][j] * xv;
            acc.y += wu[g][j] * xv;
          }
        }
      }
    }
  }
${redV2}
}

@compute @workgroup_size(256)
fn gemm4_glu_q3_g64(${sgParams({ useSubgroups })}) {
  let t0 = wid.y * 4u;
  if (t0 >= params.T) { return; }
  let local_row = lid.x / THREADS_PER_ROW;
  let lane = lid.x % THREADS_PER_ROW;
  let row = wid.x * ROWS_PER_WG + local_row;
  let safe_row = min(row, params.M - 1u);

  let n_chunks = params.K_pad >> 5u;
  let k_words = n_chunks * 3u;
  let k_groups = params.K_pad >> 6u;
  let row_q_off = safe_row * k_words;
  let row_s_off = safe_row * k_groups;

  let t1 = min(t0 + 1u, params.T - 1u);
  let t2 = min(t0 + 2u, params.T - 1u);
  let t3 = min(t0 + 3u, params.T - 1u);

  var acc_g = vec4<f32>(0.0);
  var acc_u = vec4<f32>(0.0);
  var wg: array<vec4<f32>, 8>;
  var wu: array<vec4<f32>, 8>;
  for (var c: u32 = lane; c < n_chunks; c += THREADS_PER_ROW) {
    let q_off = row_q_off + c * 3u;
    let g_sm = unpack2x16float(gate_scales[row_s_off + (c >> 1u)]);
    let u_sm = unpack2x16float(up_scales[row_s_off + (c >> 1u)]);
    q3_unpack_chunk(gate_quants[q_off], gate_quants[q_off + 1u], gate_quants[q_off + 2u], g_sm, &wg);
    q3_unpack_chunk(up_quants[q_off], up_quants[q_off + 1u], up_quants[q_off + 2u], u_sm, &wu);
    let k_base = c << 5u;
    if (k_base + 31u < params.K) {
      for (var g: u32 = 0u; g < 8u; g += 1u) {
        let kb = k_base + g * 4u;
        let b0 = t0 * params.K + kb;
        let b1 = t1 * params.K + kb;
        let b2 = t2 * params.K + kb;
        let b3 = t3 * params.K + kb;
        let x0 = vec4<f32>(x[b0], x[b0+1u], x[b0+2u], x[b0+3u]);
        let x1 = vec4<f32>(x[b1], x[b1+1u], x[b1+2u], x[b1+3u]);
        let x2 = vec4<f32>(x[b2], x[b2+1u], x[b2+2u], x[b2+3u]);
        let x3 = vec4<f32>(x[b3], x[b3+1u], x[b3+2u], x[b3+3u]);
        acc_g.x += dot(wg[g], x0); acc_u.x += dot(wu[g], x0);
        acc_g.y += dot(wg[g], x1); acc_u.y += dot(wu[g], x1);
        acc_g.z += dot(wg[g], x2); acc_u.z += dot(wu[g], x2);
        acc_g.w += dot(wg[g], x3); acc_u.w += dot(wu[g], x3);
      }
    } else {
      for (var g: u32 = 0u; g < 8u; g += 1u) {
        let kb = k_base + g * 4u;
        for (var j: u32 = 0u; j < 4u; j += 1u) {
          if (kb + j < params.K) {
            let x0 = x[t0 * params.K + kb + j];
            let x1 = x[t1 * params.K + kb + j];
            let x2 = x[t2 * params.K + kb + j];
            let x3 = x[t3 * params.K + kb + j];
            acc_g.x += wg[g][j] * x0; acc_u.x += wu[g][j] * x0;
            acc_g.y += wg[g][j] * x1; acc_u.y += wu[g][j] * x1;
            acc_g.z += wg[g][j] * x2; acc_u.z += wu[g][j] * x2;
            acc_g.w += wg[g][j] * x3; acc_u.w += wu[g][j] * x3;
          }
        }
      }
    }
  }
${redDualV4}
}
`;
  }

  return `${wgslHeader({ useSubgroups })}struct Params {
  M: u32,
  K: u32,
  T: u32,
  K_pad: u32,
};

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> gate_quants: array<u32>;
@group(0) @binding(2) var<storage, read> gate_scales: array<u32>;
@group(0) @binding(3) var<storage, read> up_quants: array<u32>;
@group(0) @binding(4) var<storage, read> up_scales: array<u32>;
@group(0) @binding(5) var<storage, read> x: array<f32>;
@group(0) @binding(6) var<storage, read_write> out: array<f32>;

const ROWS_PER_WG: u32 = 4u;
const THREADS_PER_ROW: u32 = 64u;
${smem}
${actHelper}
@compute @workgroup_size(256)
fn gemv_glu_q4_g64(${sgParams({ useSubgroups })}) {
  let tok = wid.y;
  if (tok >= params.T) { return; }
  let local_row = lid.x / THREADS_PER_ROW;
  let lane = lid.x % THREADS_PER_ROW;
  let row = wid.x * ROWS_PER_WG + local_row;
  let safe_row = min(row, params.M - 1u);

  let k_words = params.K_pad >> 3u;
  let k_groups = params.K_pad >> 6u;
  let row_q_off = safe_row * k_words;
  let row_s_off = safe_row * k_groups;
  let x_off = tok * params.K;

  var acc = vec2<f32>(0.0);
  for (var w_idx: u32 = lane; w_idx < k_words; w_idx += THREADS_PER_ROW) {
    let gw = gate_quants[row_q_off + w_idx];
    let uw = up_quants[row_q_off + w_idx];
    let g_sm = unpack2x16float(gate_scales[row_s_off + (w_idx >> 3u)]);
    let u_sm = unpack2x16float(up_scales[row_s_off + (w_idx >> 3u)]);
    let k_base = w_idx << 3u;

    var xv0 = vec4<f32>(0.0);
    var xv1 = vec4<f32>(0.0);
    if (k_base + 7u < params.K) {
      xv0 = vec4<f32>(x[x_off + k_base], x[x_off + k_base + 1u], x[x_off + k_base + 2u], x[x_off + k_base + 3u]);
      xv1 = vec4<f32>(x[x_off + k_base + 4u], x[x_off + k_base + 5u], x[x_off + k_base + 6u], x[x_off + k_base + 7u]);
    } else {
      for (var j: u32 = 0u; j < 4u; j += 1u) {
        if (k_base + j < params.K) { xv0[j] = x[x_off + k_base + j]; }
        if (k_base + 4u + j < params.K) { xv1[j] = x[x_off + k_base + 4u + j]; }
      }
    }

    let gq0 = vec4<f32>(f32(gw & 0xFu), f32((gw >> 4u) & 0xFu), f32((gw >> 8u) & 0xFu), f32((gw >> 12u) & 0xFu));
    let gq1 = vec4<f32>(f32((gw >> 16u) & 0xFu), f32((gw >> 20u) & 0xFu), f32((gw >> 24u) & 0xFu), f32((gw >> 28u) & 0xFu));
    let uq0 = vec4<f32>(f32(uw & 0xFu), f32((uw >> 4u) & 0xFu), f32((uw >> 8u) & 0xFu), f32((uw >> 12u) & 0xFu));
    let uq1 = vec4<f32>(f32((uw >> 16u) & 0xFu), f32((uw >> 20u) & 0xFu), f32((uw >> 24u) & 0xFu), f32((uw >> 28u) & 0xFu));

    acc.x += dot(gq0 * g_sm.x + g_sm.y, xv0) + dot(gq1 * g_sm.x + g_sm.y, xv1);
    acc.y += dot(uq0 * u_sm.x + u_sm.y, xv0) + dot(uq1 * u_sm.x + u_sm.y, xv1);
  }
${redV2}
}

@compute @workgroup_size(256)
fn gemm4_glu_q4_g64(${sgParams({ useSubgroups })}) {
  let t0 = wid.y * 4u;
  if (t0 >= params.T) { return; }
  let local_row = lid.x / THREADS_PER_ROW;
  let lane = lid.x % THREADS_PER_ROW;
  let row = wid.x * ROWS_PER_WG + local_row;
  let safe_row = min(row, params.M - 1u);

  let k_words = params.K_pad >> 3u;
  let k_groups = params.K_pad >> 6u;
  let row_q_off = safe_row * k_words;
  let row_s_off = safe_row * k_groups;

  let t1 = min(t0 + 1u, params.T - 1u);
  let t2 = min(t0 + 2u, params.T - 1u);
  let t3 = min(t0 + 3u, params.T - 1u);

  var acc_g = vec4<f32>(0.0);
  var acc_u = vec4<f32>(0.0);

  for (var w_idx: u32 = lane; w_idx < k_words; w_idx += THREADS_PER_ROW) {
    let gw = gate_quants[row_q_off + w_idx];
    let uw = up_quants[row_q_off + w_idx];
    let g_sm = unpack2x16float(gate_scales[row_s_off + (w_idx >> 3u)]);
    let u_sm = unpack2x16float(up_scales[row_s_off + (w_idx >> 3u)]);
    let k_base = w_idx << 3u;

    let wg0 = vec4<f32>(f32(gw & 0xFu), f32((gw >> 4u) & 0xFu), f32((gw >> 8u) & 0xFu), f32((gw >> 12u) & 0xFu)) * g_sm.x + g_sm.y;
    let wg1 = vec4<f32>(f32((gw >> 16u) & 0xFu), f32((gw >> 20u) & 0xFu), f32((gw >> 24u) & 0xFu), f32((gw >> 28u) & 0xFu)) * g_sm.x + g_sm.y;
    let wu0 = vec4<f32>(f32(uw & 0xFu), f32((uw >> 4u) & 0xFu), f32((uw >> 8u) & 0xFu), f32((uw >> 12u) & 0xFu)) * u_sm.x + u_sm.y;
    let wu1 = vec4<f32>(f32((uw >> 16u) & 0xFu), f32((uw >> 20u) & 0xFu), f32((uw >> 24u) & 0xFu), f32((uw >> 28u) & 0xFu)) * u_sm.x + u_sm.y;

    if (k_base + 7u < params.K) {
      let b0 = t0 * params.K + k_base;
      let b1 = t1 * params.K + k_base;
      let b2 = t2 * params.K + k_base;
      let b3 = t3 * params.K + k_base;

      let x0a = vec4<f32>(x[b0], x[b0+1u], x[b0+2u], x[b0+3u]);
      let x0b = vec4<f32>(x[b0+4u], x[b0+5u], x[b0+6u], x[b0+7u]);
      let x1a = vec4<f32>(x[b1], x[b1+1u], x[b1+2u], x[b1+3u]);
      let x1b = vec4<f32>(x[b1+4u], x[b1+5u], x[b1+6u], x[b1+7u]);
      let x2a = vec4<f32>(x[b2], x[b2+1u], x[b2+2u], x[b2+3u]);
      let x2b = vec4<f32>(x[b2+4u], x[b2+5u], x[b2+6u], x[b2+7u]);
      let x3a = vec4<f32>(x[b3], x[b3+1u], x[b3+2u], x[b3+3u]);
      let x3b = vec4<f32>(x[b3+4u], x[b3+5u], x[b3+6u], x[b3+7u]);

      acc_g.x += dot(wg0, x0a) + dot(wg1, x0b);
      acc_u.x += dot(wu0, x0a) + dot(wu1, x0b);
      acc_g.y += dot(wg0, x1a) + dot(wg1, x1b);
      acc_u.y += dot(wu0, x1a) + dot(wu1, x1b);
      acc_g.z += dot(wg0, x2a) + dot(wg1, x2b);
      acc_u.z += dot(wu0, x2a) + dot(wu1, x2b);
      acc_g.w += dot(wg0, x3a) + dot(wg1, x3b);
      acc_u.w += dot(wu0, x3a) + dot(wu1, x3b);
    } else {
      for (var j: u32 = 0u; j < 4u; j += 1u) {
        if (k_base + j < params.K) {
          let xv0 = x[t0 * params.K + k_base + j];
          let xv1 = x[t1 * params.K + k_base + j];
          let xv2 = x[t2 * params.K + k_base + j];
          let xv3 = x[t3 * params.K + k_base + j];
          acc_g.x += wg0[j] * xv0; acc_u.x += wu0[j] * xv0;
          acc_g.y += wg0[j] * xv1; acc_u.y += wu0[j] * xv1;
          acc_g.z += wg0[j] * xv2; acc_u.z += wu0[j] * xv2;
          acc_g.w += wg0[j] * xv3; acc_u.w += wu0[j] * xv3;
        }
        if (k_base + 4u + j < params.K) {
          let xv0 = x[t0 * params.K + k_base + 4u + j];
          let xv1 = x[t1 * params.K + k_base + 4u + j];
          let xv2 = x[t2 * params.K + k_base + 4u + j];
          let xv3 = x[t3 * params.K + k_base + 4u + j];
          acc_g.x += wg1[j] * xv0; acc_u.x += wu1[j] * xv0;
          acc_g.y += wg1[j] * xv1; acc_u.y += wu1[j] * xv1;
          acc_g.z += wg1[j] * xv2; acc_u.z += wu1[j] * xv2;
          acc_g.w += wg1[j] * xv3; acc_u.w += wu1[j] * xv3;
        }
      }
    }
  }
${redDualV4}
}
`;
}

export function buildQkNormWgsl({ useSubgroups = true } = {}) {
  const smem = useSubgroups
    ? `var<workgroup> sg_partial: array<f32, 64>;\nvar<workgroup> inv_rms_shared: f32;`
    : `var<workgroup> wg_red: array<f32, 256>;\nvar<workgroup> inv_rms_shared: f32;`;

  const red = useSubgroups
    ? `  let sg_id = lid.x / sg_size;
  let num_sg = (256u + sg_size - 1u) / sg_size;
  let sg_sum = subgroupAdd(sum_sq);
  if (sg_lid == 0u) { sg_partial[sg_id] = sg_sum; }
  workgroupBarrier();
  if (lid.x == 0u) {
    var total: f32 = 0.0;
    for (var s: u32 = 0u; s < num_sg; s += 1u) { total += sg_partial[s]; }
    inv_rms_shared = inverseSqrt(total / f32(params.head_dim) + params.eps);
  }
  workgroupBarrier();`
    : `  wg_red[lid.x] = sum_sq;
  workgroupBarrier();
  for (var stride: u32 = 128u; stride > 0u; stride >>= 1u) {
    if (lid.x < stride) { wg_red[lid.x] += wg_red[lid.x + stride]; }
    workgroupBarrier();
  }
  if (lid.x == 0u) {
    inv_rms_shared = inverseSqrt(wg_red[0] / f32(params.head_dim) + params.eps);
  }
  workgroupBarrier();`;

  return `${wgslHeader({ useSubgroups })}struct QkParams {
  q_heads: u32,
  kv_heads: u32,
  head_dim: u32,
  tokens: u32,
  eps: f32,
  norm_v: u32,
  skip_kv: u32,
  _pad: u32,
};

@group(0) @binding(0) var<uniform> params: QkParams;
@group(0) @binding(1) var<storage, read_write> qkv: array<f32>;
@group(0) @binding(2) var<storage, read> q_weight: array<f32>;
@group(0) @binding(3) var<storage, read> k_weight: array<f32>;

${smem}

@compute @workgroup_size(256)
fn main(${sgParams({ useSubgroups })}) {
  let tok = wid.y;
  let h = wid.x;
  let total_heads = select(params.q_heads + params.kv_heads * (1u + params.norm_v), params.q_heads, params.skip_kv != 0u);
  if (tok >= params.tokens || h >= total_heads) { return; }

  let q_dim = params.q_heads * params.head_dim;
  let kv_dim = params.kv_heads * params.head_dim;
  let stride = select(q_dim + 2u * kv_dim, q_dim, params.skip_kv != 0u);

  let is_q = h < params.q_heads;
  let is_k = !is_q && h < (params.q_heads + params.kv_heads);
  let head_base = tok * stride + h * params.head_dim;

  var sum_sq: f32 = 0.0;
  for (var d: u32 = lid.x; d < params.head_dim; d += 256u) {
    let v = qkv[head_base + d];
    sum_sq += v * v;
  }

${red}

  let inv_rms = inv_rms_shared;
  for (var d: u32 = lid.x; d < params.head_dim; d += 256u) {
    var w: f32 = 1.0;
    if (is_q) { w = q_weight[d]; }
    else if (is_k) { w = k_weight[d]; }
    qkv[head_base + d] *= inv_rms * w;
  }
}
`;
}

export function buildRopeKvWgsl() {
  return `${wgslHeader({ f16: true, useSubgroups: false })}struct RopeKVParams {
  T: u32,
  start_pos: u32,
  q_heads: u32,
  kv_heads: u32,
  head_dim: u32,
  rope_dim: u32,
  kv_capacity: u32,
  is_swa: u32,
  rope_theta: f32,
  rope_layout: u32,   // 0 = truncated (denom=rope_dim), 1 = proportional (denom=head_dim)
  skip_kv_write: u32, // 1 = Q-only rotation without overwriting shared KV cache
  _pad: u32,
};

@group(0) @binding(0) var<uniform> params: RopeKVParams;
@group(0) @binding(1) var<storage, read> qkv: array<f32>;
@group(0) @binding(2) var<storage, read_write> q_rot: array<f32>;
@group(0) @binding(3) var<storage, read_write> k_cache: array<f16>;
@group(0) @binding(4) var<storage, read_write> v_cache: array<f16>;

@compute @workgroup_size(128)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let q_dim = params.q_heads * params.head_dim;
  let kv_dim = params.kv_heads * params.head_dim;
  let total_qkv = select(q_dim + 2u * kv_dim, q_dim, params.skip_kv_write != 0u);
  let denom = f32(select(params.rope_dim, params.head_dim, params.rope_layout != 0u));

  // Wait: half_rope is half of head_dim when proportional, or half of rope_dim when truncated?
  // CHECK BOTH LFM3 AND GEMMA4 AND SPARK2.5:
  // In all 3: rotate_half splits head_dim // 2, with inv_freq padded with 0s for tails ->
  // cos = 1, sin = 0 cleanly outside rotated region! Using pair_stride = head_dim / 2
  // OR rope_dim / 2: wait: Spark2.5 modeling_spark_ref divides q into [q_rot(:rope_dim), q_pass(rope_dim:)]
  // and applies rotate_half inside q_rot (mate offset = rope_dim / 2)!
  // LFM3 & Gemma4 have inv_freq = [inv_freq[:rope_dim/2], zeros(head_dim/2 - rope_dim/2)] concatenated with itself:
  // cos/sin[:rope_dim/2] matches mate at +head_dim/2, and cos/sin[rope_dim/2 : head_dim/2] == 1, 0 (identity)!
  // So truncated (layout=0) rotates [0..rope_dim) with mate offset rope_dim/2;
  // proportional (layout=1) rotates [0..rope_dim/2) AND [head_dim/2 .. head_dim/2 + rope_dim/2) with mate offset head_dim/2!
  let half_rope = params.rope_dim >> 1u;
  let half_head = params.head_dim >> 1u;
  let mate_stride = select(half_rope, half_head, params.rope_layout != 0u);

  let tok = gid.y;
  if (tok >= params.T) { return; }
  let pos = f32(params.start_pos + tok);
  let slot = select(params.start_pos + tok, (params.start_pos + tok) % params.kv_capacity, params.is_swa != 0u);

  let idx = gid.x;
  let qkv_base = tok * total_qkv;

  if (idx < q_dim) {
    let h = idx / params.head_dim;
    let d = idx % params.head_dim;
    var val = qkv[qkv_base + idx];
    var pair: u32 = 0u;
    var is_hi: bool = false;
    var do_rot: bool = false;
    if (params.rope_layout == 0u) {
      if (d < params.rope_dim) {
        is_hi = d >= half_rope;
        pair = select(d, d - half_rope, is_hi);
        do_rot = true;
      }
    } else {
      if (d < half_rope) {
        pair = d;
        is_hi = false;
        do_rot = true;
      } else if (d >= half_head && d < half_head + half_rope) {
        pair = d - half_head;
        is_hi = true;
        do_rot = true;
      }
    }
    if (do_rot) {
      let inv_freq = pow(params.rope_theta, -f32(pair * 2u) / denom);
      let angle = pos * inv_freq;
      let c = cos(angle);
      let s = sin(angle);
      let mate_d = select(d + mate_stride, d - mate_stride, is_hi);
      let mate_val = qkv[qkv_base + h * params.head_dim + mate_d];
      val = select(val * c - mate_val * s, val * c + mate_val * s, is_hi);
    }
    q_rot[tok * q_dim + idx] = val;
  }

  if (params.skip_kv_write == 0u && idx < kv_dim) {
    let h = idx / params.head_dim;
    let d = idx % params.head_dim;
    let k_in_idx = qkv_base + q_dim + idx;
    let v_in_idx = qkv_base + q_dim + kv_dim + idx;
    var k_val = qkv[k_in_idx];
    var pair: u32 = 0u;
    var is_hi: bool = false;
    var do_rot: bool = false;
    if (params.rope_layout == 0u) {
      if (d < params.rope_dim) {
        is_hi = d >= half_rope;
        pair = select(d, d - half_rope, is_hi);
        do_rot = true;
      }
    } else {
      if (d < half_rope) {
        pair = d;
        is_hi = false;
        do_rot = true;
      } else if (d >= half_head && d < half_head + half_rope) {
        pair = d - half_head;
        is_hi = true;
        do_rot = true;
      }
    }
    if (do_rot) {
      let inv_freq = pow(params.rope_theta, -f32(pair * 2u) / denom);
      let angle = pos * inv_freq;
      let c = cos(angle);
      let s = sin(angle);
      let mate_d = select(d + mate_stride, d - mate_stride, is_hi);
      let mate_val = qkv[qkv_base + q_dim + h * params.head_dim + mate_d];
      k_val = select(k_val * c - mate_val * s, k_val * c + mate_val * s, is_hi);
    }
    let cache_idx = (h * params.kv_capacity + slot) * params.head_dim + d;
    k_cache[cache_idx] = f16(k_val);
    v_cache[cache_idx] = f16(qkv[v_in_idx]);
  }
}
`;
}

export function buildAttentionWgsl({ useSubgroups = true } = {}) {
  const smem = useSubgroups
    ? `var<workgroup> sg_sum_buf: array<f32, 64>;\nvar<workgroup> shared_score: f32;`
    : `var<workgroup> wg_red: array<f32, 256>;\nvar<workgroup> shared_score: f32;`;

  const redScore = useSubgroups
    ? `    let sg_id = lid.x / sg_size;
    let num_sg = (256u + sg_size - 1u) / sg_size;
    let sg_dot = subgroupAdd(prod);
    if (sg_lid == 0u) { sg_sum_buf[sg_id] = sg_dot; }
    workgroupBarrier();
    if (d0 == 0u) {
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
    if (d0 == 0u) { shared_score = wg_red[0] * params.scale; }
    workgroupBarrier();`;

  return `${wgslHeader({ f16: true, useSubgroups })}struct AttnParams {
  T: u32,
  start_pos: u32,
  q_heads: u32,
  kv_heads: u32,
  head_dim: u32,
  kv_capacity: u32,
  sliding_window: u32,
  is_swa: u32,
  has_gate: u32,
  scale: f32,
  _pad0: u32,
  _pad1: u32,
};

@group(0) @binding(0) var<uniform> params: AttnParams;
@group(0) @binding(1) var<storage, read> q_rot: array<f32>;
@group(0) @binding(2) var<storage, read> k_cache: array<f16>;
@group(0) @binding(3) var<storage, read> v_cache: array<f16>;
@group(0) @binding(4) var<storage, read> attn_gate: array<f32>;
@group(0) @binding(5) var<storage, read_write> out: array<f32>;

${smem}

@compute @workgroup_size(256)
fn main(${sgParams({ useSubgroups })}) {
  let tok = wid.y;
  let qh = wid.x;
  if (tok >= params.T || qh >= params.q_heads) { return; }

  let d0 = lid.x;
  let d1 = lid.x + 256u;
  let act0 = d0 < params.head_dim;
  let act1 = d1 < params.head_dim;

  let q_pos = params.start_pos + tok;
  let kv_h = qh / (params.q_heads / params.kv_heads);
  let q_off = (tok * params.q_heads + qh) * params.head_dim;
  let q0 = select(0.0, q_rot[q_off + d0], act0);
  let q1 = select(0.0, q_rot[q_off + d1], act1);

  var kv_start: u32 = 0u;
  if (params.is_swa != 0u && q_pos + 1u > params.sliding_window) {
    kv_start = q_pos + 1u - params.sliding_window;
  }
  let kv_end = q_pos + 1u;

  var m_i: f32 = -3.402823e38;
  var l_i: f32 = 0.0;
  var acc0: f32 = 0.0;
  var acc1: f32 = 0.0;

  let kv_head_base = kv_h * params.kv_capacity * params.head_dim;

  for (var p: u32 = kv_start; p < kv_end; p += 1u) {
    let slot = select(p, p % params.kv_capacity, params.is_swa != 0u);
    let k_base = kv_head_base + slot * params.head_dim;
    var prod: f32 = 0.0;
    if (act0) { prod += q0 * f32(k_cache[k_base + d0]); }
    if (act1) { prod += q1 * f32(k_cache[k_base + d1]); }

${redScore}

    let score = shared_score;
    let m_new = max(m_i, score);
    let alpha = exp(m_i - m_new);
    let beta = exp(score - m_new);
    l_i = l_i * alpha + beta;

    if (act0) { acc0 = acc0 * alpha + beta * f32(v_cache[k_base + d0]); }
    if (act1) { acc1 = acc1 * alpha + beta * f32(v_cache[k_base + d1]); }
    m_i = m_new;
  }

  var g_scale: f32 = 1.0;
  if (params.has_gate != 0u) {
    let g = attn_gate[tok * params.q_heads + qh];
    g_scale = 1.0 / (1.0 + exp(-g));
  }
  if (act0) { out[q_off + d0] = (acc0 / l_i) * g_scale; }
  if (act1) { out[q_off + d1] = (acc1 / l_i) * g_scale; }
}
`;
}

export function buildAttentionDecodeWgsl({ useSubgroups = true } = {}) {
  const smem = useSubgroups
    ? `var<workgroup> sg_dots: array<vec4<f32>, 64>;\nvar<workgroup> shared_scores: vec4<f32>;`
    : `var<workgroup> wg_dots: array<vec4<f32>, 256>;\nvar<workgroup> shared_scores: vec4<f32>;`;

  const redDot = useSubgroups
    ? `    let sg_id = lid.x / sg_size;
    let num_sg = (256u + sg_size - 1u) / sg_size;
    let sg_dot = subgroupAdd(prod);
    if (sg_lid == 0u) { sg_dots[sg_id] = sg_dot; }
    workgroupBarrier();
    if (d0 == 0u) {
      var dot_sum = vec4<f32>(0.0);
      for (var s: u32 = 0u; s < num_sg; s += 1u) { dot_sum += sg_dots[s]; }
      shared_scores = dot_sum * params.scale;
    }
    workgroupBarrier();`
    : `    wg_dots[lid.x] = prod;
    workgroupBarrier();
    for (var stride: u32 = 128u; stride > 0u; stride >>= 1u) {
      if (lid.x < stride) { wg_dots[lid.x] += wg_dots[lid.x + stride]; }
      workgroupBarrier();
    }
    if (d0 == 0u) { shared_scores = wg_dots[0] * params.scale; }
    workgroupBarrier();`;

  return `${wgslHeader({ f16: true, useSubgroups })}struct AttnParams {
  T: u32,
  start_pos: u32,
  q_heads: u32,
  kv_heads: u32,
  head_dim: u32,
  kv_capacity: u32,
  sliding_window: u32,
  is_swa: u32,
  has_gate: u32,
  scale: f32,
  _pad0: u32,
  _pad1: u32,
};

const K_SPLITS: u32 = 16u;

@group(0) @binding(0) var<uniform> params: AttnParams;
@group(0) @binding(1) var<storage, read> q_rot: array<f32>;
@group(0) @binding(2) var<storage, read> k_cache: array<f16>;
@group(0) @binding(3) var<storage, read> v_cache: array<f16>;
@group(0) @binding(4) var<storage, read> attn_gate: array<f32>;
@group(0) @binding(5) var<storage, read_write> split_buf: array<f32>;
@group(0) @binding(6) var<storage, read_write> out: array<f32>;

${smem}

@compute @workgroup_size(256)
fn split_kv_gqa(${sgParams({ useSubgroups })}) {
  let kv_h = wid.x / K_SPLITS;
  let split_id = wid.x % K_SPLITS;
  if (kv_h >= params.kv_heads) { return; }

  let d0 = lid.x;
  let d1 = lid.x + 256u;
  let act0 = d0 < params.head_dim;
  let act1 = d1 < params.head_dim;
  let group_size = params.q_heads / params.kv_heads;
  let q_pos = params.start_pos;

  var kv_start: u32 = 0u;
  if (params.is_swa != 0u && q_pos + 1u > params.sliding_window) {
    kv_start = q_pos + 1u - params.sliding_window;
  }
  let kv_end = q_pos + 1u;
  let total_kv = kv_end - kv_start;
  let chunk = (total_kv + K_SPLITS - 1u) / K_SPLITS;
  let s_begin = kv_start + split_id * chunk;
  let s_end = min(kv_end, s_begin + chunk);
  let stride = params.head_dim + 2u;

  if (s_begin >= s_end) {
    if (d0 == 0u) {
      for (var g: u32 = 0u; g < group_size; g += 1u) {
        let qh = kv_h * group_size + g;
        let base = (qh * K_SPLITS + split_id) * stride;
        split_buf[base] = -3.402823e38;
        split_buf[base + 1u] = 0.0;
      }
    }
    return;
  }

  var qv0 = vec4<f32>(0.0);
  var qv1 = vec4<f32>(0.0);
  for (var g: u32 = 0u; g < group_size; g += 1u) {
    let qh = kv_h * group_size + g;
    if (act0) { qv0[g] = q_rot[qh * params.head_dim + d0]; }
    if (act1) { qv1[g] = q_rot[qh * params.head_dim + d1]; }
  }

  var m = vec4<f32>(-3.402823e38);
  var l = vec4<f32>(0.0);
  var acc0 = vec4<f32>(0.0);
  var acc1 = vec4<f32>(0.0);
  let kv_head_base = kv_h * params.kv_capacity * params.head_dim;

  for (var p: u32 = s_begin; p < s_end; p += 1u) {
    let slot = select(p, p % params.kv_capacity, params.is_swa != 0u);
    let k_base = kv_head_base + slot * params.head_dim;
    var prod = vec4<f32>(0.0);
    if (act0) { prod += qv0 * f32(k_cache[k_base + d0]); }
    if (act1) { prod += qv1 * f32(k_cache[k_base + d1]); }

${redDot}

    let scores = shared_scores;
    let m_new = max(m, scores);
    let alpha = exp(m - m_new);
    let beta = exp(scores - m_new);
    l = l * alpha + beta;

    if (act0) { acc0 = acc0 * alpha + beta * f32(v_cache[k_base + d0]); }
    if (act1) { acc1 = acc1 * alpha + beta * f32(v_cache[k_base + d1]); }
    m = m_new;
  }

  for (var g: u32 = 0u; g < group_size; g += 1u) {
    let qh = kv_h * group_size + g;
    let base = (qh * K_SPLITS + split_id) * stride;
    if (d0 == 0u) {
      split_buf[base] = m[g];
      split_buf[base + 1u] = l[g];
    }
    if (act0) { split_buf[base + 2u + d0] = acc0[g]; }
    if (act1) { split_buf[base + 2u + d1] = acc1[g]; }
  }
}

@compute @workgroup_size(256)
fn reduce_kv_gqa(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
  let qh = wid.x;
  if (qh >= params.q_heads) { return; }
  let stride = params.head_dim + 2u;
  var g_scale: f32 = 1.0;
  if (params.has_gate != 0u) {
    let g = attn_gate[qh];
    g_scale = 1.0 / (1.0 + exp(-g));
  }

  for (var d: u32 = lid.x; d < params.head_dim; d += 256u) {
    var m_tot: f32 = -3.402823e38;
    var l_tot: f32 = 0.0;
    var acc_tot: f32 = 0.0;
    for (var s: u32 = 0u; s < K_SPLITS; s += 1u) {
      let base = (qh * K_SPLITS + s) * stride;
      let l_s = split_buf[base + 1u];
      if (l_s <= 0.0) { continue; }
      let m_s = split_buf[base];
      let v_s = split_buf[base + 2u + d];
      let m_new = max(m_tot, m_s);
      let alpha = select(0.0, exp(m_tot - m_new), l_tot > 0.0);
      let beta = exp(m_s - m_new);
      acc_tot = acc_tot * alpha + beta * v_s;
      l_tot = l_tot * alpha + beta * l_s;
      m_tot = m_new;
    }
    out[qh * params.head_dim + d] = (acc_tot / l_tot) * g_scale;
  }
}
`;
}

export function buildShortConvWgsl() {
  return `${wgslHeader({ useSubgroups: false })}struct ConvParams {
  dim: u32,
  tokens: u32,
  kernel_size: u32,
  state_len: u32,
};

@group(0) @binding(0) var<uniform> params: ConvParams;
@group(0) @binding(1) var<storage, read> bcx: array<f32>;
@group(0) @binding(2) var<storage, read> conv_w: array<f32>;
@group(0) @binding(3) var<storage, read_write> conv_state: array<f32>;
@group(0) @binding(4) var<storage, read_write> out: array<f32>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let d = gid.x;
  if (d >= params.dim) { return; }

  let D = params.dim;
  let S = params.state_len;
  let K = params.kernel_size;

  var st: array<f32, 8>;
  for (var j: u32 = 0u; j < S; j += 1u) {
    st[j] = conv_state[d * S + j];
  }
  var w: array<f32, 8>;
  for (var k: u32 = 0u; k < K; k += 1u) {
    w[k] = conv_w[d * K + k];
  }

  for (var t: u32 = 0u; t < params.tokens; t += 1u) {
    let base = t * 3u * D;
    let B = bcx[base + d];
    let C = bcx[base + D + d];
    let x = bcx[base + 2u * D + d];
    let bx = B * x;

    var acc: f32 = bx * w[K - 1u];
    for (var j: u32 = 0u; j < S; j += 1u) {
      acc += st[j] * w[j];
    }
    out[t * D + d] = C * acc;

    if (S > 0u) {
      for (var j: u32 = 0u; j + 1u < S; j += 1u) {
        st[j] = st[j + 1u];
      }
      st[S - 1u] = bx;
    }
  }

  for (var j: u32 = 0u; j < S; j += 1u) {
    conv_state[d * S + j] = st[j];
  }
}
`;
}

export function buildPleWgsl({ useSubgroups = true } = {}) {
  const smem = useSubgroups
    ? `var<workgroup> sg_partial: array<f32, 64>;\nvar<workgroup> inv_rms_shared: f32;`
    : `var<workgroup> wg_red: array<f32, 256>;\nvar<workgroup> inv_rms_shared: f32;`;

  const red = useSubgroups
    ? `  let sg_id = lid.x / sg_size;
  let num_sg = (256u + sg_size - 1u) / sg_size;
  let sg_sum = subgroupAdd(sum_sq);
  if (sg_lid == 0u) { sg_partial[sg_id] = sg_sum; }
  workgroupBarrier();
  if (lid.x == 0u) {
    var total: f32 = 0.0;
    for (var s: u32 = 0u; s < num_sg; s += 1u) { total += sg_partial[s]; }
    inv_rms_shared = inverseSqrt(total / f32(params.ple_dim) + params.eps);
  }
  workgroupBarrier();`
    : `  wg_red[lid.x] = sum_sq;
  workgroupBarrier();
  for (var stride: u32 = 128u; stride > 0u; stride >>= 1u) {
    if (lid.x < stride) { wg_red[lid.x] += wg_red[lid.x + stride]; }
    workgroupBarrier();
  }
  if (lid.x == 0u) {
    inv_rms_shared = inverseSqrt(wg_red[0] / f32(params.ple_dim) + params.eps);
  }
  workgroupBarrier();`;

  return `${wgslHeader({ useSubgroups })}struct PleParams {
  num_layers: u32,
  ple_dim: u32,
  tokens: u32,
  max_batch: u32,
  proj_scale: f32, // hidden_size ** -0.5
  out_scale: f32,  // 2.0 ** -0.5
  eps: f32,
  extra_f32: f32,
};

@group(0) @binding(0) var<uniform> params: PleParams;
@group(0) @binding(1) var<storage, read> ctx_proj: array<f32>;
@group(0) @binding(2) var<storage, read> tok_ple: array<f32>;
@group(0) @binding(3) var<storage, read> norm_w: array<f32>;
@group(0) @binding(4) var<storage, read_write> out: array<f32>;

${smem}

@compute @workgroup_size(256)
fn ple_combine(${sgParams({ useSubgroups })}) {
  // wid.x = layer (0..L-1), wid.y = tok (0..T-1), outputs [L, T, ple_dim] for contiguous per-layer bind ranges!
  let l = wid.x;
  let tok = wid.y;
  if (l >= params.num_layers || tok >= params.tokens) { return; }

  let idx_tl = (tok * params.num_layers + l) * params.ple_dim;
  var sum_sq: f32 = 0.0;
  for (var p: u32 = lid.x; p < params.ple_dim; p += 256u) {
    let v = ctx_proj[idx_tl + p] * params.proj_scale;
    sum_sq += v * v;
  }

${red}

  let inv_rms = inv_rms_shared;
  let out_base = (l * params.max_batch + tok) * params.ple_dim;
  for (var p: u32 = lid.x; p < params.ple_dim; p += 256u) {
    let c = (ctx_proj[idx_tl + p] * params.proj_scale) * inv_rms * norm_w[p];
    let t = tok_ple[idx_tl + p];
    out[out_base + p] = (c + t) * params.out_scale;
  }
}

@compute @workgroup_size(256)
fn gate_mul_gelu_tanh(@builtin(global_invocation_id) gid: vec3<u32>) {
  // out[i] = gelu_pytorch_tanh(out[i]) * ctx_proj[tok_ple_layer_slice + i]
  // Note bindings reused cleanly via ple_ops BGL: ctx_proj=gate_tmp(rw? no, out has gate, tok_ple has layer ple!)
  let idx = gid.x;
  let total = params.tokens * params.ple_dim;
  if (idx >= total) { return; }
  let v = ctx_proj[idx];
  let u = clamp(0.7978845608028654 * (v + 0.044715 * v * v * v), -15.0, 15.0);
  let act = 0.5 * v * (1.0 + tanh(u));
  out[idx] = act * tok_ple[params.max_batch + idx];
}

@compute @workgroup_size(256)
fn softcap_logits(@builtin(global_invocation_id) gid: vec3<u32>) {
  // params.ple_dim = total floats, params.proj_scale = softcap
  let i = gid.x;
  if (i >= params.ple_dim) { return; }
  let cap = params.proj_scale;
  let v = out[i];
  out[i] = cap * tanh(clamp(v / cap, -15.0, 15.0));
}

@compute @workgroup_size(256)
fn add_bias(@builtin(global_invocation_id) gid: vec3<u32>) {
  // out[tok * M + m] += norm_w[m]; params.ple_dim = M, params.tokens = T
  let idx = gid.x;
  let total = params.tokens * params.ple_dim;
  if (idx >= total) { return; }
  let m = idx % params.ple_dim;
  out[idx] += norm_w[m];
}
`;
}
