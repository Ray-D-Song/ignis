enable subgroups;

struct Params {
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
var<workgroup> sg_partial: array<f32, 32>;
var<workgroup> sg_partial_gemm: array<vec4<f32>, 32>;

fn gemv_row_dot(row: u32, lane: u32, tok: u32) -> f32 {
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
    let scale = sm.x;
    let min_v = sm.y;
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

    let q0 = vec4<f32>(
      f32(word & 0xFu),
      f32((word >> 4u) & 0xFu),
      f32((word >> 8u) & 0xFu),
      f32((word >> 12u) & 0xFu),
    );
    let q1 = vec4<f32>(
      f32((word >> 16u) & 0xFu),
      f32((word >> 20u) & 0xFu),
      f32((word >> 24u) & 0xFu),
      f32((word >> 28u) & 0xFu),
    );

    acc += dot(q0 * scale + min_v, xv0) + dot(q1 * scale + min_v, xv1);
  }
  return acc;
}

fn gemm4_row_dot(row: u32, lane: u32, t0: u32) -> vec4<f32> {
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
    let scale = sm.x;
    let min_v = sm.y;
    let k_base = w_idx << 3u;

    let q0 = vec4<f32>(
      f32(word & 0xFu),
      f32((word >> 4u) & 0xFu),
      f32((word >> 8u) & 0xFu),
      f32((word >> 12u) & 0xFu),
    );
    let q1 = vec4<f32>(
      f32((word >> 16u) & 0xFu),
      f32((word >> 20u) & 0xFu),
      f32((word >> 24u) & 0xFu),
      f32((word >> 28u) & 0xFu),
    );
    let w0 = q0 * scale + min_v;
    let w1 = q1 * scale + min_v;

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
fn gemv_q4_g64(
  @builtin(workgroup_id) wid: vec3<u32>,
  @builtin(local_invocation_id) lid: vec3<u32>,
  @builtin(subgroup_invocation_id) sg_lid: u32,
  @builtin(subgroup_size) sg_size: u32,
) {
  let sg_id = lid.x / sg_size;
  let tok = wid.y;
  if (tok >= params.T) { return; }
  let local_row = lid.x / THREADS_PER_ROW;
  let lane = lid.x % THREADS_PER_ROW;
  let row = wid.x * ROWS_PER_WG + local_row;

  let sg_sum = subgroupAdd(gemv_row_dot(row, lane, tok));
  if (sg_lid == 0u) { sg_partial[sg_id] = sg_sum; }
  workgroupBarrier();

  if (lane == 0u && row < params.M) {
    let sg_per_row = max(1u, THREADS_PER_ROW / sg_size);
    let sg_start = local_row * sg_per_row;
    var total: f32 = 0.0;
    for (var s: u32 = 0u; s < sg_per_row; s += 1u) { total += sg_partial[sg_start + s]; }
    out[tok * params.M + row] = total;
  }
}

@compute @workgroup_size(256)
fn gemv_q4_g64_add(
  @builtin(workgroup_id) wid: vec3<u32>,
  @builtin(local_invocation_id) lid: vec3<u32>,
  @builtin(subgroup_invocation_id) sg_lid: u32,
  @builtin(subgroup_size) sg_size: u32,
) {
  let sg_id = lid.x / sg_size;
  let tok = wid.y;
  if (tok >= params.T) { return; }
  let local_row = lid.x / THREADS_PER_ROW;
  let lane = lid.x % THREADS_PER_ROW;
  let row = wid.x * ROWS_PER_WG + local_row;

  let sg_sum = subgroupAdd(gemv_row_dot(row, lane, tok));
  if (sg_lid == 0u) { sg_partial[sg_id] = sg_sum; }
  workgroupBarrier();

  if (lane == 0u && row < params.M) {
    let sg_per_row = max(1u, THREADS_PER_ROW / sg_size);
    let sg_start = local_row * sg_per_row;
    var total: f32 = 0.0;
    for (var s: u32 = 0u; s < sg_per_row; s += 1u) { total += sg_partial[sg_start + s]; }
    out[tok * params.M + row] += total;
  }
}

@compute @workgroup_size(256)
fn gemm4_q4_g64(
  @builtin(workgroup_id) wid: vec3<u32>,
  @builtin(local_invocation_id) lid: vec3<u32>,
  @builtin(subgroup_invocation_id) sg_lid: u32,
  @builtin(subgroup_size) sg_size: u32,
) {
  let sg_id = lid.x / sg_size;
  let t0 = wid.y * 4u;
  if (t0 >= params.T) { return; }
  let local_row = lid.x / THREADS_PER_ROW;
  let lane = lid.x % THREADS_PER_ROW;
  let row = wid.x * ROWS_PER_WG + local_row;

  let sg_sum = subgroupAdd(gemm4_row_dot(row, lane, t0));
  if (sg_lid == 0u) { sg_partial_gemm[sg_id] = sg_sum; }
  workgroupBarrier();

  if (lane == 0u && row < params.M) {
    let sg_per_row = max(1u, THREADS_PER_ROW / sg_size);
    let sg_start = local_row * sg_per_row;
    var total = vec4<f32>(0.0);
    for (var s: u32 = 0u; s < sg_per_row; s += 1u) { total += sg_partial_gemm[sg_start + s]; }
    out[t0 * params.M + row] = total.x;
    if (t0 + 1u < params.T) { out[(t0 + 1u) * params.M + row] = total.y; }
    if (t0 + 2u < params.T) { out[(t0 + 2u) * params.M + row] = total.z; }
    if (t0 + 3u < params.T) { out[(t0 + 3u) * params.M + row] = total.w; }
  }
}

@compute @workgroup_size(256)
fn gemm4_q4_g64_add(
  @builtin(workgroup_id) wid: vec3<u32>,
  @builtin(local_invocation_id) lid: vec3<u32>,
  @builtin(subgroup_invocation_id) sg_lid: u32,
  @builtin(subgroup_size) sg_size: u32,
) {
  let sg_id = lid.x / sg_size;
  let t0 = wid.y * 4u;
  if (t0 >= params.T) { return; }
  let local_row = lid.x / THREADS_PER_ROW;
  let lane = lid.x % THREADS_PER_ROW;
  let row = wid.x * ROWS_PER_WG + local_row;

  let sg_sum = subgroupAdd(gemm4_row_dot(row, lane, t0));
  if (sg_lid == 0u) { sg_partial_gemm[sg_id] = sg_sum; }
  workgroupBarrier();

  if (lane == 0u && row < params.M) {
    let sg_per_row = max(1u, THREADS_PER_ROW / sg_size);
    let sg_start = local_row * sg_per_row;
    var total = vec4<f32>(0.0);
    for (var s: u32 = 0u; s < sg_per_row; s += 1u) { total += sg_partial_gemm[sg_start + s]; }
    out[t0 * params.M + row] += total.x;
    if (t0 + 1u < params.T) { out[(t0 + 1u) * params.M + row] += total.y; }
    if (t0 + 2u < params.T) { out[(t0 + 2u) * params.M + row] += total.z; }
    if (t0 + 3u < params.T) { out[(t0 + 3u) * params.M + row] += total.w; }
  }
}

// INT8_G64_SYM GEMV (4x signed int8 per u32, group_size=64 -> 16 u32 words per scale)
@compute @workgroup_size(256)
fn gemv_q8_g64(
  @builtin(workgroup_id) wid: vec3<u32>,
  @builtin(local_invocation_id) lid: vec3<u32>,
  @builtin(subgroup_invocation_id) sg_lid: u32,
  @builtin(subgroup_size) sg_size: u32,
) {
  let sg_id = lid.x / sg_size;
  let tok = wid.y;
  if (tok >= params.T) { return; }
  let local_row = lid.x / THREADS_PER_ROW;
  let lane = lid.x % THREADS_PER_ROW;
  let row = wid.x * ROWS_PER_WG + local_row;
  let safe_row = min(row, params.M - 1u);

  let k_words = params.K_pad >> 2u;   // 4 i8 per u32
  let k_groups = params.K_pad >> 6u;  // 64 weights per group
  let row_q_off = safe_row * k_words;
  let row_s_off = safe_row * k_groups;
  let x_off = tok * params.K;

  var acc: f32 = 0.0;
  for (var w_idx: u32 = lane; w_idx < k_words; w_idx += THREADS_PER_ROW) {
    let word = i32(quants[row_q_off + w_idx]);
    let sm = unpack2x16float(scales[row_s_off + (w_idx >> 4u)]);
    let scale = sm.x;
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
    acc += dot(qv * scale, xv);
  }

  let sg_sum = subgroupAdd(acc);
  if (sg_lid == 0u) { sg_partial[sg_id] = sg_sum; }
  workgroupBarrier();

  if (lane == 0u && row < params.M) {
    let sg_per_row = max(1u, THREADS_PER_ROW / sg_size);
    let sg_start = local_row * sg_per_row;
    var total: f32 = 0.0;
    for (var s: u32 = 0u; s < sg_per_row; s += 1u) { total += sg_partial[sg_start + s]; }
    out[tok * params.M + row] = total;
  }
}
