/**
 * WebGPU Hardware Capability Probe & Graceful Degradation Profile Selector (Stage 6A).
 */

const MIB = 1024 * 1024;
const GIB = 1024 * MIB;

function hasFeature(features, name) {
  if (!features) return false;
  if (typeof features.has === 'function') return features.has(name);
  if (Array.isArray(features)) return features.includes(name);
  return Boolean(features[name]);
}

export async function detectDeviceProfile(adapterOrNull = undefined) {
  let adapter = adapterOrNull;
  if (adapter === undefined && typeof navigator !== 'undefined' && navigator.gpu) {
    adapter = await navigator.gpu.requestAdapter({ powerPreference: 'high-performance' });
  }
  if (!adapter) {
    return {
      supported: false,
      tier: 'unsupported',
      hasF16: false,
      hasSubgroups: false,
      hasTimestampQuery: false,
      maxBufferSize: 0,
      maxStorageBufferBindingSize: 0,
      maxComputeWorkgroupStorageSize: 0,
      maxComputeInvocationsPerWorkgroup: 0,
      recommendedWeightBits: 3,
      recommendedUseSubgroups: false,
      recommendedMaxShardBytes: 0,
      estimatedVramClass: 'low',
      adapterInfo: null,
    };
  }

  const features = adapter.features;
  const limits = adapter.limits || {};
  const hasF16 = hasFeature(features, 'shader-f16');
  const hasSubgroups = hasFeature(features, 'subgroups');
  const hasTimestampQuery = hasFeature(features, 'timestamp-query');

  const maxBufferSize = Number(limits.maxBufferSize || 0);
  const maxStorageBufferBindingSize = Number(limits.maxStorageBufferBindingSize || 0);
  const maxComputeWorkgroupStorageSize = Number(limits.maxComputeWorkgroupStorageSize || 0);
  const maxComputeInvocationsPerWorkgroup = Number(limits.maxComputeInvocationsPerWorkgroup || 0);

  const estimatedVramClass = maxBufferSize >= 2 * GIB
    ? 'high'
    : maxBufferSize >= GIB
      ? 'medium'
      : 'low';

  let tier = 'tier3_constrained';
  if (hasF16 && hasSubgroups && maxBufferSize >= GIB) {
    tier = 'tier1_full';
  } else if (hasF16 && !hasSubgroups) {
    tier = 'tier2_no_subgroups';
  }

  const recommendedWeightBits = maxBufferSize < GIB ? 3 : 4;
  const recommendedUseSubgroups = hasSubgroups;
  const recommendedMaxShardBytes = Math.min(
    512 * MIB,
    Math.floor(maxStorageBufferBindingSize / 256) * 256
  );

  const rawInfo = adapter.info || null;
  const adapterInfo = rawInfo
    ? {
        vendor: rawInfo.vendor || '',
        architecture: rawInfo.architecture || '',
        device: rawInfo.device || '',
        description: rawInfo.description || '',
      }
    : null;

  return {
    supported: true,
    tier,
    hasF16,
    hasSubgroups,
    hasTimestampQuery,
    maxBufferSize,
    maxStorageBufferBindingSize,
    maxComputeWorkgroupStorageSize,
    maxComputeInvocationsPerWorkgroup,
    recommendedWeightBits,
    recommendedUseSubgroups,
    recommendedMaxShardBytes,
    estimatedVramClass,
    adapterInfo,
  };
}
