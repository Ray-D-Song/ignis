enable subgroups;

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

const ROWS_PER_WG: u32 = 4u;
const THREADS_PER_ROW: u32 = 64u;
var<workgroup> sg_partial: array<f32, 32>;

@compute @workgroup_size(256)
fn linear_f32(
  @builtin(workgroup_id) wid: vec3<u32>,
  @builtin(local_invocation_id) lid: vec3<u32>,
  @builtin(subgroup_invocation_id) sg_lid: u32,
  @builtin(subgroup_size) sg_size: u32,
) {
  let sg_id = lid.x / sg_size;
  let tok = wid.y;
  if (tok >= params.T) {
    return;
  }
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

  let sg_sum = subgroupAdd(acc);
  if (sg_lid == 0u) {
    sg_partial[sg_id] = sg_sum;
  }
  workgroupBarrier();

  if (lane == 0u && row < params.M) {
    let sg_per_row = max(1u, THREADS_PER_ROW / sg_size);
    let sg_start = local_row * sg_per_row;
    var total: f32 = 0.0;
    for (var s: u32 = 0u; s < sg_per_row; s += 1u) {
      total += sg_partial[sg_start + s];
    }
    out[tok * params.M + row] = total;
  }
}
