import fs from 'node:fs';
import path from 'node:path';
import { getPublicCdnModels, type ModelPreset } from './model-catalog';

export type { ModelPreset };
export { getPublicCdnModels };

function findFixturesRoot(): string {
  const candidates = [
    path.resolve(process.cwd(), '../../tests/fixtures'),
    path.resolve(process.cwd(), 'tests/fixtures'),
  ];
  for (const c of candidates) {
    if (fs.existsSync(c)) return c;
  }
  return candidates[0];
}

export function discoverIgnisModels(): ModelPreset[] {
  const fixturesRoot = findFixturesRoot();
  const defs = [
    {
      id: 'spark-4b-q8head',
      folder: 'spark-4b-ignis-q8head',
      shortName: 'Spark-X2.6 4B',
      label: 'Spark-X2.6 4B · Fast Reasoning',
      description: 'Fast local reasoning & tool use',
      sizeLabel: '2.6 GB',
      licenseName: 'Apache-2.0',
    },
    {
      id: 'spark-4b-q4',
      folder: 'spark-4b-ignis-q4',
      shortName: 'Spark-X2.6 4B (Compact)',
      label: 'Spark-X2.6 4B · Compact 4-bit',
      description: 'Compact VRAM footprint for local reasoning',
      sizeLabel: '2.2 GB',
      licenseName: 'Apache-2.0',
    },
    {
      id: 'lfm3-omni-2b',
      folder: 'lfm3-omni-2b-ignis',
      shortName: 'LFM3-Omni 2B',
      label: 'LFM3-Omni 2B · Vision, Audio & Voice',
      description: 'Multimodal · Vision, Audio & 24 kHz Voice',
      sizeLabel: '1.8 GB',
      licenseName: 'LFM Open License v1.0',
    },
    {
      id: 'gemma-4-e4b',
      folder: 'gemma-4-e4b-ignis',
      shortName: 'Gemma-4 E4B',
      label: 'Gemma-4 E4B · High-Capacity Omni',
      description: 'High-capacity multimodal & reasoning',
      sizeLabel: '6.2 GB',
      licenseName: 'Gemma Terms of Use',
    },
  ];

  const available: ModelPreset[] = [];
  for (const d of defs) {
    const dirPath = path.join(fixturesRoot, d.folder);
    const manifestPath = path.join(dirPath, 'manifest.json');
    if (!fs.existsSync(manifestPath)) continue;
    const manifest = JSON.parse(fs.readFileSync(manifestPath, 'utf-8'));
    const bp = manifest.blueprint || {};
    const shards: Array<{ id: string | number; bytes: number }> = manifest.shards || [];
    const gpuBytes = shards
      .filter((s) => typeof s.id === 'number')
      .reduce((acc, s) => acc + s.bytes, 0);
    const embdBytes = shards.find((s) => s.id === 'embd')?.bytes || 0;
    const pleBytes = shards.find((s) => s.id === 'ple_embd')?.bytes || 0;
    const lmHeadDtype = manifest.tensors?.['output.quants']?.dtype || 'u32_q4';
    const architecture = bp.architecture || 'spark2_5';
    const topModelType = bp.top_model_type || architecture;
    const mmKind = bp.multimodal?.kind;
    available.push({
      id: d.id,
      shortName: d.shortName,
      label: d.label,
      description: d.description,
      sizeLabel: d.sizeLabel,
      dirPath,
      modelBaseUrl: `/api/models/${d.id}`,
      lmHeadDtype,
      architecture,
      topModelType,
      gpuMiB: +(gpuBytes / (1024 * 1024)).toFixed(1),
      embdMiB: +(embdBytes / (1024 * 1024)).toFixed(1),
      pleMiB: +(pleBytes / (1024 * 1024)).toFixed(1),
      licenseName: d.licenseName,
      capabilities: {
        vision: mmKind === 'lfm3_omni' || mmKind === 'gemma4',
        audioIn: mmKind === 'lfm3_omni' || mmKind === 'gemma4',
        audioOut: mmKind === 'lfm3_omni',
        thinking: architecture === 'spark2_5' || topModelType === 'gemma4',
      },
    });
  }
  return available;
}

export function isPublicCdnMode(): boolean {
  return Boolean((process.env.NEXT_PUBLIC_IGNIS_CDN_BASE || '').trim());
}

export function getConfiguredModels(): ModelPreset[] {
  const cdnBase = (process.env.NEXT_PUBLIC_IGNIS_CDN_BASE || '').trim();
  if (cdnBase) {
    return getPublicCdnModels(cdnBase);
  }
  return discoverIgnisModels();
}
