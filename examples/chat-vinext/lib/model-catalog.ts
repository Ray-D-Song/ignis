export interface ModelPreset {
  id: string;
  label: string;
  description: string;
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
      id: 'lfm3-omni-2b',
      label: 'LFM3-Omni-2B · Full Omni (Vision + Audio + 24 kHz Voice)',
      description: 'ShortConv+SWA 24-layer backbone · INT8_G64 LM Head · Vision/Audio in + 24 kHz LiquidCodec out',
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
      id: 'spark-4b-q8head',
      label: 'Spark-X2.6-4B · INT8_G64 LM Head (Fast Reasoning)',
      description: '36-layer Ring SWA+Full INT4_G64 body + INT8_G64 LM Head · 85-134 tok/s on desktop WebGPU',
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
      id: 'gemma-4-e4b',
      label: 'Gemma-4-E4B-it · Full Omni (262K Vocab + Q8 PLE)',
      description: '42-layer KV-Sharing + OPFS 2.69 GiB PLE streaming + SigLIP2 Vision + Chunked Conformer',
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
