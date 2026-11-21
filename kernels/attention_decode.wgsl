enable f16;
enable subgroups;

// Split-KV GQA FlashDecoding for T=1 decode:
// Processes all Q-heads sharing a KV-head together (group_size <= 4) and splits KV range across K_SPLITS=16 workgroups.
struct AttnParams {
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
@group(0) @binding(5) var<storage, read_write> split_buf: array<f32>; // [q_heads * K_SPLITS * (head_dim + 2)]
@group(0) @binding(6) var<storage, read_write> out: array<f32>;

var<workgroup> sg_dots: array<vec4<f32>, 64>;
var<workgroup> shared_scores: vec4<f32>;

@compute @workgroup_size(256)
fn split_kv_gqa(
  @builtin(workgroup_id) wid: vec3<u32>,
  @builtin(local_invocation_id) lid: vec3<u32>,
  @builtin(subgroup_invocation_id) sg_lid: u32,
  @builtin(subgroup_id) sg_id: u32,
  @builtin(subgroup_size) sg_size: u32,
) {
  let num_sg = (256u + sg_size - 1u) / sg_size;
  let kv_h = wid.x / K_SPLITS;
  let split_id = wid.x % K_SPLITS;
  if (kv_h >= params.kv_heads) {
    return;
  }

  let d = lid.x;
  let is_active = d < params.head_dim;
  let group_size = params.q_heads / params.kv_heads; // supports 1..4
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
    // Empty split: write l = 0.0 marker
    if (d == 0u) {
      for (var g: u32 = 0u; g < group_size; g += 1u) {
        let qh = kv_h * group_size + g;
        let base = (qh * K_SPLITS + split_id) * stride;
        split_buf[base] = -3.402823e38;
        split_buf[base + 1u] = 0.0;
      }
    }
    return;
  }

  var qv = vec4<f32>(0.0);
  if (is_active) {
    for (var g: u32 = 0u; g < group_size; g += 1u) {
      let qh = kv_h * group_size + g;
      qv[g] = q_rot[qh * params.head_dim + d];
    }
  }

  var m = vec4<f32>(-3.402823e38);
  var l = vec4<f32>(0.0);
  var acc = vec4<f32>(0.0);

  let kv_head_base = kv_h * params.kv_capacity * params.head_dim;

  for (var p: u32 = s_begin; p < s_end; p += 1u) {
    let slot = select(p, p % params.kv_capacity, params.is_swa != 0u);
    let k_idx = kv_head_base + slot * params.head_dim + d;
    let k_val = select(0.0, f32(k_cache[k_idx]), is_active);
    let prod = qv * k_val;

    let sg_dot = subgroupAdd(prod);
    if (sg_lid == 0u) {
      sg_dots[sg_id] = sg_dot;
    }
    workgroupBarrier();

    if (d == 0u) {
      var dot_sum = vec4<f32>(0.0);
      for (var s: u32 = 0u; s < num_sg; s += 1u) {
        dot_sum += sg_dots[s];
      }
      shared_scores = dot_sum * params.scale;
    }
    workgroupBarrier();

    let scores = shared_scores;
    let m_new = max(m, scores);
    let alpha = exp(m - m_new);
    let beta = exp(scores - m_new);
    l = l * alpha + beta;

    let v_val = select(0.0, f32(v_cache[k_idx]), is_active);
    acc = acc * alpha + beta * v_val;
    m = m_new;
  }

  for (var g: u32 = 0u; g < group_size; g += 1u) {
    let qh = kv_h * group_size + g;
    let base = (qh * K_SPLITS + split_id) * stride;
    if (d == 0u) {
      split_buf[base] = m[g];
      split_buf[base + 1u] = l[g];
    }
    if (is_active) {
      split_buf[base + 2u + d] = acc[g];
    }
  }
}

@compute @workgroup_size(256)
fn reduce_kv_gqa(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
  let qh = wid.x;
  let d = lid.x;
  if (qh >= params.q_heads || d >= params.head_dim) {
    return;
  }
  let stride = params.head_dim + 2u;

  var m_tot: f32 = -3.402823e38;
  var l_tot: f32 = 0.0;
  var acc_tot: f32 = 0.0;

  for (var s: u32 = 0u; s < K_SPLITS; s += 1u) {
    let base = (qh * K_SPLITS + s) * stride;
    let l_s = split_buf[base + 1u];
    if (l_s <= 0.0) {
      continue;
    }
    let m_s = split_buf[base];
    let v_s = split_buf[base + 2u + d];

    let m_new = max(m_tot, m_s);
    let alpha = select(0.0, exp(m_tot - m_new), l_tot > 0.0);
    let beta = exp(m_s - m_new);

    acc_tot = acc_tot * alpha + beta * v_s;
    l_tot = l_tot * alpha + beta * l_s;
    m_tot = m_new;
  }

  var val = acc_tot / l_tot;
  if (params.has_gate != 0u) {
    let g = attn_gate[qh];
    val = val * (1.0 / (1.0 + exp(-g)));
  }
  out[qh * params.head_dim + d] = val;
}
