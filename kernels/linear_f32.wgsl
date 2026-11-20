// F32 dense linear projection (e.g. attn_gate [num_heads, hidden_size]) + GeGLU + Residual Add

struct LinearParams {
  M: u32,
  K: u32,
  T: u32,
  _pad: u32,
};

@group(0) @binding(0) var<uniform> params: LinearParams;
@group(0) @binding(1) var<storage, read> weight: array<f32>;
@group(0) @binding(2) var<storage, read> x: array<f32>;
@group(0) @binding(3) var<storage, read_write> out: array<f32>;

@compute @workgroup_size(64)
fn linear_f32(
  @builtin(global_invocation_id) gid: vec3<u32>,
) {
  let row = gid.x;
  let tok = gid.y;
  if (row >= params.M || tok >= params.T) {
    return;
  }
  let w_off = row * params.K;
  let x_off = tok * params.K;
  var acc: f32 = 0.0;
  for (var k: u32 = 0u; k < params.K; k += 1u) {
    acc += weight[w_off + k] * x[x_off + k];
  }
  out[tok * params.M + row] = acc;
}
