/**
 * Minimal zero-dependency JS wrapper for Ignis Rust-WASM Tokenizer (works in Node & Browser/Worker).
 */

export class IgnisTokenizer {
  constructor(instance, handle) {
    this.inst = instance;
    this.ex = instance.exports;
    this.handle = handle;
    this.enc = new TextEncoder();
    this.dec = new TextDecoder();
  }

  static async fromBuffers(wasmBytes, tokenizerJsonBytes) {
    const { instance } = await WebAssembly.instantiate(wasmBytes, {});
    const ex = instance.exports;
    const jsonBytes = tokenizerJsonBytes instanceof Uint8Array
      ? tokenizerJsonBytes
      : new Uint8Array(tokenizerJsonBytes);

    const ptr = ex.ignis_alloc(jsonBytes.length);
    new Uint8Array(ex.memory.buffer, ptr, jsonBytes.length).set(jsonBytes);
    const handle = ex.ignis_tok_create(ptr, jsonBytes.length);
    ex.ignis_dealloc(ptr, jsonBytes.length);
    if (!handle) {
      throw new Error('Failed to initialize Ignis WASM tokenizer from tokenizer.json');
    }
    return new IgnisTokenizer(instance, handle);
  }

  encode(text) {
    const bytes = this.enc.encode(text);
    const inPtr = bytes.length > 0 ? this.ex.ignis_alloc(bytes.length) : 0;
    if (bytes.length > 0) {
      new Uint8Array(this.ex.memory.buffer, inPtr, bytes.length).set(bytes);
    }
    const lenPtr = this.ex.ignis_alloc(4);
    const outPtr = this.ex.ignis_tok_encode(this.handle, inPtr, bytes.length, lenPtr);
    const outLen = new Uint32Array(this.ex.memory.buffer, lenPtr, 1)[0];
    const result = Array.from(new Uint32Array(this.ex.memory.buffer, outPtr, outLen));
    if (bytes.length > 0) this.ex.ignis_dealloc(inPtr, bytes.length);
    this.ex.ignis_dealloc(lenPtr, 4);
    this.ex.ignis_tok_free_u32(outPtr, outLen);
    return result;
  }

  decode(ids, skipSpecial = false) {
    if (!ids.length) return '';
    const byteLen = ids.length * 4;
    const inPtr = this.ex.ignis_alloc(byteLen);
    new Uint32Array(this.ex.memory.buffer, inPtr, ids.length).set(ids);
    const lenPtr = this.ex.ignis_alloc(4);
    const outPtr = this.ex.ignis_tok_decode(this.handle, inPtr, ids.length, skipSpecial ? 1 : 0, lenPtr);
    const outLen = new Uint32Array(this.ex.memory.buffer, lenPtr, 1)[0];
    const outBytes = new Uint8Array(this.ex.memory.buffer, outPtr, outLen).slice(0);
    this.ex.ignis_dealloc(inPtr, byteLen);
    this.ex.ignis_dealloc(lenPtr, 4);
    this.ex.ignis_tok_free_bytes(outPtr, outLen);
    return this.dec.decode(outBytes);
  }

  free() {
    if (this.handle) {
      this.ex.ignis_tok_free(this.handle);
      this.handle = 0;
    }
  }
}
