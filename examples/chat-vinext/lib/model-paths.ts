import fs from 'node:fs';
import path from 'node:path';

export interface ModelPreset {
  id: string;
  label: string;
  description: string;
  dirPath: string;
  lmHeadDtype: string;
  architecture: string;
  topModelType: string;
  gpuMiB: number;
  embdMiB: number;
  pleMiB: number;
  capabilities: {
    vision: boolean;
    audioIn: boolean;
    audioOut: boolean;
    thinking: boolean;
  };
}

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
      label: 'Spark-X2.6-4B · INT8_G64 LM Head（推荐）',
      description: '36层 Body INT4_G64 + LM Head INT8_G64 · 更高中文/代码保真度',
    },
    {
      id: 'spark-4b-q4',
      folder: 'spark-4b-ignis-q4',
      label: 'Spark-X2.6-4B · INT4_G64 LM Head（省显存）',
      description: '36层 Body + LM Head 全 INT4_G64 · 极致显存压缩',
    },
    {
      id: 'lfm3-omni-2b',
      folder: 'lfm3-omni-2b-ignis',
      label: 'LFM3-Omni-2B · Full Omni（图/音输入 + 24kHz 语音输出）',
      description: 'ShortConv+SWA 24层主干 + 视觉/音频双编码 + LiquidCodec 24kHz 语音合成',
    },
    {
      id: 'gemma-4-e4b',
      folder: 'gemma-4-e4b-ignis',
      label: 'Gemma-4-E4B-it · Full Omni（PLE + 图像/音频理解）',
      description: '42层 KV-Sharing + INT8_G64 PLE + 262K 词表 + SigLIP2/Conformer 全模态',
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
      label: d.label,
      description: d.description,
      dirPath,
      lmHeadDtype,
      architecture,
      topModelType,
      gpuMiB: +(gpuBytes / (1024 * 1024)).toFixed(1),
      embdMiB: +(embdBytes / (1024 * 1024)).toFixed(1),
      pleMiB: +(pleBytes / (1024 * 1024)).toFixed(1),
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
