export type IgnisThinkingEffort = 'low' | 'high';

export type ChatContentPart =
  | { type: 'text'; text: string }
  | {
      type: 'image_url';
      image_url:
        | string
        | {
            url?: string;
            data?: Uint8Array | Uint8ClampedArray;
            width?: number;
            height?: number;
            channels?: number;
          };
    }
  | {
      type: 'input_audio';
      input_audio: {
        data: Float32Array | number[] | Uint8Array | ArrayBuffer | string;
        sampleRate?: number;
        format?: 'pcm_f32' | 'wav';
      };
    };

export interface ChatMessage {
  role: 'system' | 'user' | 'assistant';
  content: string | ChatContentPart[];
}

export interface JsonSchemaResponseFormat {
  type: 'text' | 'json_object' | 'json_schema';
  json_schema?: {
    name?: string;
    schema: Record<string, unknown>;
    strict?: boolean;
  };
}

export interface SpeculativeOptions {
  mode: 'ngram' | 'draft';
  draftModelUrl?: string;
  draftModelKey?: string;
  k?: number;
  ngramN?: number;
}

export interface ChatCompletionCreateParams {
  model?: string;
  messages: ChatMessage[];
  stream?: boolean;
  max_tokens?: number;
  temperature?: number;
  top_p?: number;
  top_k?: number;
  repetition_penalty?: number;
  enable_thinking?: boolean;
  thinking_effort?: IgnisThinkingEffort;
  thinking_budget?: number;
  response_format?: JsonSchemaResponseFormat;
  speculative?: SpeculativeOptions;
  generation_mode?: 'text' | 'audio' | 'interleaved';
  stop?: string | string[];
}

export interface AudioOutput {
  pcm24k?: Float32Array;
  wavBytes?: Uint8Array;
  sampleRate: number;
}

export interface ChatCompletion {
  id: string;
  object: 'chat.completion';
  created: number;
  model: string;
  choices: Array<{
    index: number;
    message: {
      role: 'assistant';
      content: string;
      reasoning_content?: string;
      audio?: AudioOutput;
    };
    finish_reason: 'stop' | 'length' | 'abort';
  }>;
  usage: {
    prompt_tokens: number;
    completion_tokens: number;
    total_tokens: number;
  };
  ignis?: {
    prefillMs: number;
    decodeMs: number;
    decodeTokPerSec: number;
    speculativeAccepted?: number;
    speculativeProposed?: number;
    peakGpuMiB?: number;
  };
}

export interface ChatCompletionChunk {
  id: string;
  object: 'chat.completion.chunk';
  created: number;
  model: string;
  choices: Array<{
    index: number;
    delta: {
      role?: 'assistant';
      content?: string;
      reasoning_content?: string;
      audio?: AudioOutput;
    };
    finish_reason: 'stop' | 'length' | 'abort' | null;
  }>;
}

export interface IgnisModelConfig {
  modelUrl: string;
  modelKey?: string;
  kernelBaseUrl?: string;
  wasmUrl?: string;
  maxSeqLen?: number;
  maxBatchTokens?: number;
  forceNoSubgroups?: boolean;
  opfs?: boolean;
}

export interface IgnisOpenAIOptions extends Partial<IgnisModelConfig> {
  workerUrl?: string;
  models?: Record<string, IgnisModelConfig>;
  onProgress?: (event: { file: string; loaded: number; total: number; cached: boolean }) => void;
}

export class IgnisOpenAI {
  readonly chat: {
    completions: {
      create(params: ChatCompletionCreateParams & { stream: true }): AsyncIterable<ChatCompletionChunk>;
      create(params: ChatCompletionCreateParams & { stream?: false }): Promise<ChatCompletion>;
      create(params: ChatCompletionCreateParams): Promise<ChatCompletion> | AsyncIterable<ChatCompletionChunk>;
    };
  };

  constructor(options?: IgnisOpenAIOptions);
  loadModel(modelNameOrConfig: string | IgnisModelConfig): Promise<{ blueprint: Record<string, unknown>; cacheHits: number; cacheMisses: number; initMs: number }>;
  abort(): Promise<void>;
  reset(): Promise<void>;
  destroy(): Promise<void>;
}

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
  recommendedWeightBits: 3 | 4;
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

export function detectDeviceProfile(adapterOrNull?: GPUAdapter | null): Promise<IgnisDeviceProfile>;

export class IgnisFsm {
  maskLogits(logitsFloat32: Float32Array): number;
  advance(tokenId: number): boolean;
  isDone(): boolean;
  reset(): void;
  free(): void;
}

export function resolveResponseSchema(responseFormat?: JsonSchemaResponseFormat | null): Record<string, unknown> | null;

export interface SpeculativeGenerateResult {
  generatedIds: number[];
  prefillMs: number;
  decodeMs: number;
  decodeTokPerSec: number;
  speculativeProposed: number;
  speculativeAccepted: number;
  rounds: number;
}

export class NgramLookupSpeculator {
  constructor(options?: { k?: number; minNgram?: number; maxNgram?: number });
  propose(history: number[], maxCount: number, eosSet?: Set<number>): number[];
  generate(
    engine: unknown,
    promptIds: number[],
    options?: Record<string, unknown>
  ): Promise<SpeculativeGenerateResult>;
}

export class DraftTargetSpeculator {
  constructor(targetEngine: unknown, draftEngine: unknown, options?: { k?: number });
  generate(
    promptIds: number[],
    options?: Record<string, unknown>
  ): Promise<SpeculativeGenerateResult>;
}

export function buildIgnisChatPrompt(
  messages: ChatMessage[],
  blueprint: Record<string, unknown>,
  options?: { enableThinking?: boolean; thinkingEffort?: IgnisThinkingEffort }
): Promise<{ prompt: string; images: unknown[]; audios: Float32Array[] }>;

export function getIgnisAssetUrls(): {
  workerUrl: string;
  wasmUrl: string;
  kernelBaseUrl: string;
};
