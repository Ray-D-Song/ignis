/**
 * Minimal zero-dependency JS wrapper for Ignis Rust-WASM Tokenizer (works in Node & Browser/Worker).
 */

/**
 * Normalize OpenAI `response_format` into a JSON schema object, or null when unconstrained.
 */
export function resolveResponseSchema(responseFormat) {
  if (!responseFormat || responseFormat.type === 'text') return null;
  if (responseFormat.type === 'json_object') return { type: 'object' };
  if (responseFormat.type === 'json_schema') {
    return responseFormat.json_schema?.schema || responseFormat.schema || { type: 'object' };
  }
  return null;
}

export class IgnisFsm {
  constructor(instance, handle, vocabSize) {
    this.inst = instance;
    this.ex = instance.exports;
    this.handle = handle;
    this.vocabSize = vocabSize;
    this.wordsLenPtr = this.ex.ignis_alloc(4);
    this.allowedCntPtr = this.ex.ignis_alloc(4);
  }

  /**
   * Masks disallowed vocabulary logits in-place to -Infinity using the WASM-cached bitset.
   * Zero 512 KB logits copy: only reads the ~4-8 KB packed bitset directly from WASM memory.
   */
  maskLogits(logitsFloat32) {
    if (!this.handle) throw new Error('IgnisFsm already freed');
    const wordsPtr = this.ex.ignis_fsm_allowed_bitset(this.handle, this.wordsLenPtr, this.allowedCntPtr);
    const dv = new DataView(this.ex.memory.buffer);
    const wordsLen = dv.getUint32(this.wordsLenPtr, true);
    const allowedCount = dv.getUint32(this.allowedCntPtr, true);
    const words = new Uint32Array(this.ex.memory.buffer, wordsPtr, wordsLen);

    const V = logitsFloat32.length;
    const tokV = Math.min(V, this.vocabSize);
    const numW = Math.min(wordsLen, Math.ceil(tokV / 32));
    for (let wi = 0; wi < numW; wi++) {
      const w = words[wi] >>> 0;
      if (w === 0xffffffff) continue;
      const base = wi << 5;
      const end = Math.min(base + 32, tokV);
      if (w === 0) {
        logitsFloat32.fill(-Infinity, base, end);
        continue;
      }
      for (let bit = 0; bit < end - base; bit++) {
        if (((w >>> bit) & 1) === 0) {
          logitsFloat32[base + bit] = -Infinity;
        }
      }
    }
    if (V > this.vocabSize) {
      logitsFloat32.fill(-Infinity, this.vocabSize, V);
    }
    return allowedCount;
  }

  advance(tokenId) {
    if (!this.handle) throw new Error('IgnisFsm already freed');
    return this.ex.ignis_fsm_advance(this.handle, tokenId >>> 0) !== 0;
  }

  isDone() {
    if (!this.handle) return false;
    return this.ex.ignis_fsm_is_done(this.handle) !== 0;
  }

  reset() {
    if (!this.handle) throw new Error('IgnisFsm already freed');
    this.ex.ignis_fsm_reset(this.handle);
  }

  free() {
    if (this.handle) {
      this.ex.ignis_fsm_free(this.handle);
      this.handle = 0;
    }
    if (this.wordsLenPtr) {
      this.ex.ignis_dealloc(this.wordsLenPtr, 4);
      this.wordsLenPtr = 0;
    }
    if (this.allowedCntPtr) {
      this.ex.ignis_dealloc(this.allowedCntPtr, 4);
      this.allowedCntPtr = 0;
    }
  }
}

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

  getVocabSize() {
    if (!this.handle) throw new Error('IgnisTokenizer already freed');
    return this.ex.ignis_tok_vocab_size(this.handle);
  }

  createFsm(schemaOrObj, eosIds = []) {
    if (!this.handle) throw new Error('IgnisTokenizer already freed');
    const schemaStr = typeof schemaOrObj === 'string'
      ? schemaOrObj
      : JSON.stringify(resolveResponseSchema(schemaOrObj) || schemaOrObj || { type: 'object' });
    const schemaBytes = this.enc.encode(schemaStr);
    const schemaPtr = schemaBytes.length > 0 ? this.ex.ignis_alloc(schemaBytes.length) : 0;
    if (schemaBytes.length > 0) {
      new Uint8Array(this.ex.memory.buffer, schemaPtr, schemaBytes.length).set(schemaBytes);
    }
    const eosArr = Uint32Array.from(eosIds);
    const eosByteLen = eosArr.length * 4;
    const eosPtr = eosByteLen > 0 ? this.ex.ignis_alloc(eosByteLen) : 0;
    if (eosByteLen > 0) {
      new Uint32Array(this.ex.memory.buffer, eosPtr, eosArr.length).set(eosArr);
    }
    const fsmHandle = this.ex.ignis_fsm_create(
      this.handle,
      schemaPtr,
      schemaBytes.length,
      eosPtr,
      eosArr.length
    );
    if (schemaPtr) this.ex.ignis_dealloc(schemaPtr, schemaBytes.length);
    if (eosPtr) this.ex.ignis_dealloc(eosPtr, eosByteLen);
    if (!fsmHandle) {
      throw new Error(`Failed to compile JSON Schema FSM: ${schemaStr}`);
    }
    return new IgnisFsm(this.inst, fsmHandle, this.getVocabSize());
  }

  free() {
    if (this.handle) {
      this.ex.ignis_tok_free(this.handle);
      this.handle = 0;
    }
  }
}
