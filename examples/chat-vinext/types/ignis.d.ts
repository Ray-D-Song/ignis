declare module '../../../runtime/ignis-client.mjs' {
  export interface IgnisProgressEvent {
    phase?: 'manifest' | 'download' | 'cache' | 'tokenizer' | 'compile' | 'gpu_upload' | 'tokenizer_init';
    file: string;
    loaded: number;
    total: number;
    overallLoaded?: number;
    overallTotal?: number;
    fileIndex?: number;
    fileCount?: number;
    cached: boolean;
  }

  export interface IgnisTokenEvent {
    tokenId: number;
    text: string;
    fullText?: string;
    step: number;
    tokPerSec?: number;
  }

  export class IgnisClient {
    onProgress: ((p: IgnisProgressEvent) => void) | null;
    onPrefillProgress: ((p: { processed: number; total: number }) => void) | null;
    onToken: ((ev: IgnisTokenEvent) => void) | null;
    onDeviceLost: ((reason: string) => void) | null;

    constructor(workerUrl?: string);
    init(options: Record<string, unknown>): Promise<{
      initMs: number;
      cacheHits: number;
      cacheMisses: number;
      hasTokenizer: boolean;
      blueprint: {
        architecture: string;
        top_model_type?: string;
        num_layers: number;
        eos_token_id: number;
        eos_token_ids?: number[];
        multimodal?: { kind: string };
      };
    }>;
    encode(text: string): Promise<{ ids: number[] }>;
    decode(ids: number[], skipSpecial?: boolean): Promise<{ text: string }>;
    reset(): Promise<{ curPos: number }>;
    generate(options: {
      promptIds: number[];
      images?: Array<{ data: Uint8Array; width: number; height: number; channels: number }>;
      audios?: Float32Array[];
      generationMode?: 'text' | 'interleaved' | 'audio';
      maxNewTokens?: number;
      maxThinkingTokens?: number;
      chunkSize?: number;
      sampling?: Record<string, unknown>;
      responseFormat?: {
        type: 'json_schema';
        json_schema: {
          name?: string;
          schema: Record<string, unknown>;
          strict?: boolean;
        };
      };
      speculative?: {
        mode: 'ngram' | 'draft';
        k?: number;
        ngramN?: number;
        draftModelUrl?: string;
        draftModelKey?: string;
      };
      stopOnEos?: boolean;
    }): Promise<{
      promptLen: number;
      generatedIds: number[];
      generatedText: string | null;
      wavBytes?: Uint8Array | null;
      pcm24k?: Float32Array | null;
      aborted?: boolean;
      prefillMs: number;
      decodeMs: number;
      decodeTokPerSec: number;
      fsmMaskMsSamples?: number[];
      speculativeAccepted?: number;
      speculativeProposed?: number;
      curPos: number;
    }>;
    abort(): Promise<{ aborted: boolean }>;
    destroy(): Promise<void>;
  }
}

declare module '../../../runtime/multimodal/audio-dsp.mjs' {
  export function decodeWav(arrayBuffer: ArrayBuffer): { samples: Float32Array; sampleRate: number };
  export function resampleLinear(samples: Float32Array, fromSr: number, toSr: number): Float32Array;
}

declare module '../../../runtime/device-profile.mjs' {
  export interface IgnisDeviceProfile {
    supported: boolean;
    tier: 'tier1_full' | 'tier2_no_subgroups' | 'tier3_constrained' | 'unsupported';
    hasF16: boolean;
    hasSubgroups: boolean;
    hasTimestampQuery: boolean;
    maxBufferSize: number;
    maxStorageBufferBindingSize: number;
    maxComputeWorkgroupStorageSize: number;
    maxComputeInvocationsPerWorkgroup: number;
    recommendedWeightBits: number;
    recommendedUseSubgroups: boolean;
    recommendedMaxShardBytes: number;
    estimatedVramClass: 'high' | 'medium' | 'low';
    adapterInfo: {
      vendor: string;
      architecture: string;
      device: string;
      description: string;
    } | null;
  }

  export function detectDeviceProfile(): Promise<IgnisDeviceProfile>;
}
