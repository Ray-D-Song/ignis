import fs from 'node:fs';
import path from 'node:path';

export interface ModelPreset {
  id: string;
  label: string;
  description: string;
  dirPath: string;
  lmHeadDtype: string;
  gpuMiB: number;
  embdMiB: number;
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
  ];

  const available: ModelPreset[] = [];
  for (const d of defs) {
    const dirPath = path.join(fixturesRoot, d.folder);
    const manifestPath = path.join(dirPath, 'manifest.json');
    if (!fs.existsSync(manifestPath)) continue;
    const manifest = JSON.parse(fs.readFileSync(manifestPath, 'utf-8'));
    const gpuBytes = manifest.shards
      .filter((s: { id: string | number }) => s.id !== 'embd')
      .reduce((acc: number, s: { bytes: number }) => acc + s.bytes, 0);
    const embdBytes = manifest.shards.find((s: { id: string | number }) => s.id === 'embd')?.bytes || 0;
    const lmHeadDtype = manifest.tensors['output.quants']?.dtype || 'u32_q4';
    available.push({
      id: d.id,
      label: d.label,
      description: d.description,
      dirPath,
      lmHeadDtype,
      gpuMiB: +(gpuBytes / (1024 * 1024)).toFixed(1),
      embdMiB: +(embdBytes / (1024 * 1024)).toFixed(1),
    });
  }
  return available;
}
