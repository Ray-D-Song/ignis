declare module '../../../runtime/ignis-client.mjs' {
  export interface IgnisProgressEvent {
    file: string;
    loaded: number;
    total: number;
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
      blueprint: { architecture: string; num_layers: number; eos_token_id: number };
    }>;
    encode(text: string): Promise<{ ids: number[] }>;
    decode(ids: number[], skipSpecial?: boolean): Promise<{ text: string }>;
    reset(): Promise<{ curPos: number }>;
    generate(options: Record<string, unknown>): Promise<{
      promptLen: number;
      generatedIds: number[];
      generatedText: string | null;
      aborted?: boolean;
      prefillMs: number;
      decodeMs: number;
      decodeTokPerSec: number;
      curPos: number;
    }>;
    abort(): Promise<{ aborted: boolean }>;
    destroy(): Promise<void>;
  }
}
