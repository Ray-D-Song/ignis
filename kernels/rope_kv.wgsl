enable f16;

struct RopeKVParams {
  T: u32,
  start_pos: u32,
  q_heads: u32,
  kv_heads: u32,
  head_dim: u32,
  rope_dim: u32,
  kv_capacity: u32,  // sliding_window for SWA, max_seq_len for Full
  is_swa: u32,       // 1 = ring buffer (pos % kv_capacity), 0 = linear
  rope_theta: f32,
  _pad0: u32,
  _pad1: u32,
  _pad2: u32,
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
  let total_qkv = q_dim + 2u * kv_dim;
  let half_rope = params.rope_dim >> 1u;

  let tok = gid.y;
  if (tok >= params.T) {
    return;
  }
  let pos = f32(params.start_pos + tok);
  let slot = select(params.start_pos + tok, (params.start_pos + tok) % params.kv_capacity, params.is_swa != 0u);

  let idx = gid.x;
  let qkv_base = tok * total_qkv;

  // Dispatch covers max(q_heads, kv_heads) * head_dim threads along X;
  // handle Q elements and K/V elements in parallel per thread.
  if (idx < q_dim) {
    let h = idx / params.head_dim;
    let d = idx % params.head_dim;
    let q_in_idx = qkv_base + idx;
    var val = qkv[q_in_idx];
    if (d < params.rope_dim) {
      let pair = select(d, d - half_rope, d >= half_rope);
      let inv_freq = pow(params.rope_theta, -f32(pair * 2u) / f32(params.rope_dim));
      let angle = pos * inv_freq;
      let c = cos(angle);
      let s = sin(angle);
      let mate_d = select(d + half_rope, d - half_rope, d >= half_rope);
      let mate_val = qkv[qkv_base + h * params.head_dim + mate_d];
      val = select(val * c - mate_val * s, val * c + mate_val * s, d >= half_rope);
    }
    q_rot[tok * q_dim + idx] = val;
  }

  if (idx < kv_dim) {
    let h = idx / params.head_dim;
    let d = idx % params.head_dim;
    let k_in_idx = qkv_base + q_dim + idx;
    let v_in_idx = qkv_base + q_dim + kv_dim + idx;
    var k_val = qkv[k_in_idx];
    if (d < params.rope_dim) {
      let pair = select(d, d - half_rope, d >= half_rope);
      let inv_freq = pow(params.rope_theta, -f32(pair * 2u) / f32(params.rope_dim));
      let angle = pos * inv_freq;
      let c = cos(angle);
      let s = sin(angle);
      let mate_d = select(d + half_rope, d - half_rope, d >= half_rope);
      let mate_val = qkv[qkv_base + q_dim + h * params.head_dim + mate_d];
      k_val = select(k_val * c - mate_val * s, k_val * c + mate_val * s, d >= half_rope);
    }
    // KV cache layout: [kv_heads, kv_capacity, head_dim]
    let cache_idx = (h * params.kv_capacity + slot) * params.head_dim + d;
    k_cache[cache_idx] = f16(k_val);
    v_cache[cache_idx] = f16(qkv[v_in_idx]);
  }
}
