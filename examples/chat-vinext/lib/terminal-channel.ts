export class TerminalChannel {
  private buffer: string[] = [];
  private bufferChars = 0;
  private maxBufferChars: number;
  private listeners = new Set<(chunk: string) => void>();

  constructor(maxBufferChars = 32_000) {
    this.maxBufferChars = maxBufferChars;
  }

  push(chunk: string): void {
    if (!chunk) return;
    this.buffer.push(chunk);
    this.bufferChars += chunk.length;
    while (this.bufferChars > this.maxBufferChars && this.buffer.length > 1) {
      const removed = this.buffer.shift()!;
      this.bufferChars -= removed.length;
    }
    for (const fn of this.listeners) {
      try {
        fn(chunk);
      } catch {
        // ignore listener failure
      }
    }
  }

  subscribe(listener: (chunk: string) => void, replay = true): () => void {
    if (replay && this.buffer.length > 0) {
      listener(this.buffer.join(''));
    }
    this.listeners.add(listener);
    return () => {
      this.listeners.delete(listener);
    };
  }
}
