struct ElemParams {
  count: u32,
  _p0: u32,
  _p1: u32,
  _p2: u32,
};

@group(0) @binding(0) var<uniform> params: ElemParams;
@group(0) @binding(1) var<storage, read> a: array<f32>;
@group(0) @binding(2) var<storage, read> b: array<f32>;
@group(0) @binding(3) var<storage, read_write> out: array<f32>;

// Abramowitz & Stegun 7.1.26 erf(x) approximation (|err| < 1.5e-7)
fn erf_approx(x: f32) -> f32 {
  let sign = select(-1.0, 1.0, x >= 0.0);
  let ax = abs(x);
  let t = 1.0 / (1.0 + 0.3275911 * ax);
  let y = 1.0 - (((((1.061405429 * t - 1.453152027) * t) + 1.421413741) * t - 0.284496736) * t + 0.254829592) * t * exp(-ax * ax);
  return sign * y;
}

@compute @workgroup_size(256)
fn geglu(@builtin(global_invocation_id) gid: vec3<u32>) {
  let idx = gid.x;
  if (idx >= params.count) {
    return;
  }
  let g = a[idx];
  let u = b[idx];
  let gelu_g = 0.5 * g * (1.0 + erf_approx(g * 0.7071067811865475));
  out[idx] = gelu_g * u;
}

@compute @workgroup_size(256)
fn add_residual(@builtin(global_invocation_id) gid: vec3<u32>) {
  let idx = gid.x;
  if (idx >= params.count) {
    return;
  }
  out[idx] = a[idx] + b[idx];
}
