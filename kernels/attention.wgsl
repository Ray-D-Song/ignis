enable f16;
enable subgroups;

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

@group(0) @binding(0) var<uniform> params: AttnParams;
@group(0) @binding(1) var<storage, read> q_rot: array<f32>;
@group(0) @binding(2) var<storage, read> k_cache: array<f16>;
@group(0) @binding(3) var<storage, read> v_cache: array<f16>;
@group(0) @binding(4) var<storage, read> attn_gate: array<f32>;
@group(0) @binding(5) var<storage, read_write> out: array<f32>;

// Workgroup handles one (token, q_head) pair; head_dim <= 256, workgroup_size = 256.
// Each thread i (0 .. head_dim-1) maintains its own scalar q[i] and accumulator acc[i] in registers!
var<workgroup> sg_max_buf: array<f32, 64>;
var<workgroup> sg_sum_buf: array<f32, 64>;
var<workgroup> shared_score: f32;

@compute @workgroup_size(256)
fn main(
  @builtin(workgroup_id) wid: vec3<u32>,
  @builtin(local_invocation_id) lid: vec3<u32>,
  @builtin(subgroup_invocation_id) sg_lid: u32,
  @builtin(subgroup_id) sg_id: u32,
  @builtin(num_subgroups) num_sg: u32,
) {
  let tok = wid.y;
  let qh = wid.x;
  if (tok >= params.T || qh >= params.q_heads) {
    return;
  }

  let d = lid.x;
  let is_active = d < params.head_dim;
  let q_pos = params.start_pos + tok;
  let kv_h = qh / (params.q_heads / params.kv_heads);
  let q_off = (tok * params.q_heads + qh) * params.head_dim;
  let q_val = select(0.0, q_rot[q_off + d], is_active);

  var kv_start: u32 = 0u;
  if (params.is_swa != 0u && q_pos + 1u > params.sliding_window) {
    kv_start = q_pos + 1u - params.sliding_window;
  }
  let kv_end = q_pos + 1u;

  var m_i: f32 = -3.402823e38;
  var l_i: f32 = 0.0;
  var acc: f32 = 0.0;

  let kv_head_base = kv_h * params.kv_capacity * params.head_dim;

  for (var p: u32 = kv_start; p < kv_end; p += 1u) {
    let slot = select(p, p % params.kv_capacity, params.is_swa != 0u);
    let k_idx = kv_head_base + slot * params.head_dim + d;
    let prod = select(0.0, q_val * f32(k_cache[k_idx]), is_active);

    let sg_dot = subgroupAdd(prod);
    if (sg_lid == 0u) {
      sg_sum_buf[sg_id] = sg_dot;
    }
    workgroupBarrier();

    if (d == 0u) {
      var dot_sum: f32 = 0.0;
      for (var s: u32 = 0u; s < num_sg; s += 1u) {
        dot_sum += sg_sum_buf[s];
      }
      shared_score = dot_sum * params.scale;
    }
    workgroupBarrier();

    let score = shared_score;
    let m_new = max(m_i, score);
    let alpha = exp(m_i - m_new);
    let beta = exp(score - m_new);
    l_i = l_i * alpha + beta;

    let v_val = select(0.0, f32(v_cache[k_idx]), is_active);
    acc = acc * alpha + beta * v_val;
    m_i = m_new;
  }

  if (is_active) {
    var val = acc / l_i;
    if (params.has_gate != 0u) {
      let g = attn_gate[tok * params.q_heads + qh];
      let sig = 1.0 / (1.0 + exp(-g));
      val = val * sig;
    }
    out[q_off + d] = val;
  }
}
