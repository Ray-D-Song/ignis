export interface ModelPreset {
  id: string;
  shortName?: string;
  label: string;
  description: string;
  sizeLabel?: string;
  dirPath: string;
  modelBaseUrl?: string;
  lmHeadDtype: string;
  architecture: string;
  topModelType: string;
  gpuMiB: number;
  embdMiB: number;
  pleMiB: number;
  licenseName?: string;
  requiresTermsAck?: boolean;
  isCustom?: boolean;
  capabilities: {
    vision: boolean;
    audioIn: boolean;
    audioOut: boolean;
    thinking: boolean;
  };
}

export function getPublicCdnModels(cdnBaseRaw: string): ModelPreset[] {
  const base = cdnBaseRaw.replace(/\/+$/, '');
  return [
    {
      id: 'spark-4b-q8head',
      shortName: 'Spark-X2.6 4B',
      label: 'Spark-X2.6 4B · Fast Reasoning',
      description: 'Fast local reasoning & tool use',
      sizeLabel: '2.6 GB',
      dirPath: `${base}/spark-4b-ignis-q8head`,
      modelBaseUrl: `${base}/spark-4b-ignis-q8head`,
      lmHeadDtype: 'u32_q8',
      architecture: 'spark2_5',
      topModelType: 'spark2_5',
      gpuMiB: 2371.3,
      embdMiB: 322.0,
      pleMiB: 0,
      licenseName: 'Apache-2.0',
      capabilities: {
        vision: false,
        audioIn: false,
        audioOut: false,
        thinking: true,
      },
    },
    {
      id: 'spark-1.7b-q8head',
      shortName: 'Spark-X2.6 1.7B',
      label: 'Spark-X2.6 1.7B · Ultra-Fast Reasoning',
      description: 'Lightweight 1.7B local reasoning & coding',
      sizeLabel: '1.3 GB',
      dirPath: `${base}/spark-1.7b-ignis-q8head`,
      modelBaseUrl: `${base}/spark-1.7b-ignis-q8head`,
      lmHeadDtype: 'u32_q8',
      architecture: 'spark2_5',
      topModelType: 'spark2_5',
      gpuMiB: 1046.0,
      embdMiB: 258.0,
      pleMiB: 0,
      licenseName: 'Apache-2.0',
      capabilities: {
        vision: false,
        audioIn: false,
        audioOut: false,
        thinking: true,
      },
    },
    {
      id: 'lfm3-omni-2b',
      shortName: 'LFM3-Omni 2B',
      label: 'LFM3-Omni 2B · Vision, Audio & Voice',
      description: 'Multimodal · Vision, Audio & 24 kHz Voice',
      sizeLabel: '1.8 GB',
      dirPath: `${base}/lfm3-omni-2b-ignis`,
      modelBaseUrl: `${base}/lfm3-omni-2b-ignis`,
      lmHeadDtype: 'u32_q8',
      architecture: 'lfm3',
      topModelType: 'lfm3_omni',
      gpuMiB: 1054.9,
      embdMiB: 252.0,
      pleMiB: 0,
      licenseName: 'LFM Open License v1.0',
      capabilities: {
        vision: true,
        audioIn: true,
        audioOut: true,
        thinking: false,
      },
    },
    {
      id: 'gemma-4-e4b',
      shortName: 'Gemma-4 E4B',
      label: 'Gemma-4 E4B · High-Capacity Omni',
      description: 'High-capacity multimodal & reasoning',
      sizeLabel: '6.2 GB',
      dirPath: `${base}/gemma-4-e4b-ignis`,
      modelBaseUrl: `${base}/gemma-4-e4b-ignis`,
      lmHeadDtype: 'u32_q8',
      architecture: 'gemma4_text',
      topModelType: 'gemma4',
      gpuMiB: 2532.7,
      embdMiB: 644.0,
      pleMiB: 2692.0,
      licenseName: 'Gemma Terms of Use',
      requiresTermsAck: true,
      capabilities: {
        vision: true,
        audioIn: true,
        audioOut: false,
        thinking: true,
      },
    },
  ];
}
