enable subgroups;

struct Params {
  dim: u32,
  tokens: u32,
  eps: f32,
  _pad: u32,
};

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> x: array<f32>;
@group(0) @binding(2) var<storage, read> weight: array<f32>;
@group(0) @binding(3) var<storage, read_write> out: array<f32>;

const WG_SIZE: u32 = 256u;
var<workgroup> sg_partial: array<f32, 64>;
var<workgroup> inv_rms_shared: f32;

@compute @workgroup_size(256)
fn main(
  @builtin(workgroup_id) wid: vec3<u32>,
  @builtin(local_invocation_id) lid: vec3<u32>,
  @builtin(subgroup_invocation_id) sg_lid: u32,
  @builtin(subgroup_size) sg_size: u32,
) {
  let sg_id = lid.x / sg_size;
  let num_sg = (WG_SIZE + sg_size - 1u) / sg_size;
  let tok = wid.x;
  if (tok >= params.tokens) {
    return;
  }
  let base = tok * params.dim;

  var sum_sq: f32 = 0.0;
  for (var i: u32 = lid.x; i < params.dim; i += WG_SIZE) {
    let v = x[base + i];
    sum_sq += v * v;
  }

  let sg_sum = subgroupAdd(sum_sq);
  if (sg_lid == 0u) {
    sg_partial[sg_id] = sg_sum;
  }
  workgroupBarrier();

  if (lid.x == 0u) {
    var total: f32 = 0.0;
    for (var s: u32 = 0u; s < num_sg; s += 1u) {
      total += sg_partial[s];
    }
    inv_rms_shared = inverseSqrt(total / f32(params.dim) + params.eps);
  }
  workgroupBarrier();

  let inv_rms = inv_rms_shared;
  for (var i: u32 = lid.x; i < params.dim; i += WG_SIZE) {
    out[base + i] = x[base + i] * inv_rms * weight[i];
  }
}
