enable subgroups;

struct Params {
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
var<workgroup> sg_partial: array<vec2<f32>, 32>;
var<workgroup> sg_partial_g: array<vec4<f32>, 32>;
var<workgroup> sg_partial_u: array<vec4<f32>, 32>;

fn erf_approx(v: f32) -> f32 {
  let sign = select(-1.0, 1.0, v >= 0.0);
  let ax = abs(v);
  let t = 1.0 / (1.0 + 0.3275911 * ax);
  let y = 1.0 - (((((1.061405429 * t - 1.453152027) * t) + 1.421413741) * t - 0.284496736) * t + 0.254829592) * t * exp(-ax * ax);
  return sign * y;
}

fn gelu_exact(v: f32) -> f32 {
  return 0.5 * v * (1.0 + erf_approx(v * 0.7071067811865475));
}

@compute @workgroup_size(256)
fn gemv_geglu_q4_g64(
  @builtin(workgroup_id) wid: vec3<u32>,
  @builtin(local_invocation_id) lid: vec3<u32>,
  @builtin(subgroup_invocation_id) sg_lid: u32,
  @builtin(subgroup_id) sg_id: u32,
  @builtin(subgroup_size) sg_size: u32,
) {
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

  let sg_sum = subgroupAdd(acc);
  if (sg_lid == 0u) { sg_partial[sg_id] = sg_sum; }
  workgroupBarrier();

  if (lane == 0u && row < params.M) {
    let sg_per_row = max(1u, THREADS_PER_ROW / sg_size);
    let sg_start = local_row * sg_per_row;
    var tot = vec2<f32>(0.0);
    for (var s: u32 = 0u; s < sg_per_row; s += 1u) { tot += sg_partial[sg_start + s]; }
    out[tok * params.M + row] = gelu_exact(tot.x) * tot.y;
  }
}

@compute @workgroup_size(256)
fn gemm4_geglu_q4_g64(
  @builtin(workgroup_id) wid: vec3<u32>,
  @builtin(local_invocation_id) lid: vec3<u32>,
  @builtin(subgroup_invocation_id) sg_lid: u32,
  @builtin(subgroup_id) sg_id: u32,
  @builtin(subgroup_size) sg_size: u32,
) {
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
    out[t0 * params.M + row] = gelu_exact(tg.x) * tu.x;
    if (t0 + 1u < params.T) { out[(t0 + 1u) * params.M + row] = gelu_exact(tg.y) * tu.y; }
    if (t0 + 2u < params.T) { out[(t0 + 2u) * params.M + row] = gelu_exact(tg.z) * tu.z; }
    if (t0 + 3u < params.T) { out[(t0 + 3u) * params.M + row] = gelu_exact(tg.w) * tu.w; }
  }
}
