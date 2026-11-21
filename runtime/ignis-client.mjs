/**
 * Main-thread client handle for Ignis Dedicated Web Worker.
 */

export class IgnisClient {
  constructor(workerUrl = '/runtime/ignis-worker.mjs') {
    this.worker = new Worker(workerUrl, { type: 'module' });
    this.nextId = 1;
    this.pending = new Map();
    this.onProgress = null;
    this.onPrefillProgress = null;
    this.onToken = null;
    this.onDeviceLost = null;

    this.worker.onmessage = (ev) => {
      const msg = ev.data;
      if (msg.type === 'progress') {
        if (this.onProgress) this.onProgress(msg);
        return;
      }
      if (msg.type === 'prefill_progress') {
        if (this.onPrefillProgress) this.onPrefillProgress(msg);
        return;
      }
      if (msg.type === 'token') {
        if (this.onToken) this.onToken(msg);
        return;
      }
      if (msg.type === 'device_lost') {
        if (this.onDeviceLost) this.onDeviceLost(msg.reason);
        return;
      }
      const p = this.pending.get(msg.id);
      if (!p) return;
      this.pending.delete(msg.id);
      if (msg.ok) p.resolve(msg.data);
      else p.reject(new Error(msg.error || 'Worker error'));
    };
    this.worker.onerror = (ev) => {
      for (const p of this.pending.values()) {
        p.reject(new Error(ev.message || 'Worker crashed'));
      }
      this.pending.clear();
    };
  }

  _call(cmd, payload = {}) {
    const id = this.nextId++;
    return new Promise((resolve, reject) => {
      this.pending.set(id, { resolve, reject });
      this.worker.postMessage({ id, cmd, payload });
    });
  }

  init(options) {
    return this._call('init', options);
  }

  encode(text) {
    return this._call('encode', { text });
  }

  decode(ids, skipSpecial = false) {
    return this._call('decode', { ids, skipSpecial });
  }

  reset() {
    return this._call('reset');
  }

  generate(options) {
    return this._call('generate', options);
  }

  abort() {
    return this._call('abort');
  }

  async destroy() {
    try {
      await this._call('destroy');
    } finally {
      this.worker.terminate();
    }
  }
}
