/**
 * Dual-Mode Lossless Speculative Decoding for Ignis WebGPU Runtime (Stage 6C):
 * 1. NgramLookupSpeculator: Zero-VRAM suffix n-gram lookup over prompt + output history
 * 2. DraftTargetSpeculator: Dual-model Draft-Target speculative verification (e.g. INT3_G64 -> INT4/INT8)
 * Both modes share exact greedy/stochastic rejection verification, KV rollback, and GPU ShortConv state restore.
 */

import { computeSamplingDistribution, sampleToken } from './sampler.mjs';

function buildProbMap(dist) {
  const m = new Map();
  for (let i = 0; i < dist.cutoff; i++) {
    m.set(dist.indices[i], dist.probs[i]);
  }
  return m;
}

function sampleFromDist(dist, rng) {
  let r = rng();
  for (let i = 0; i < dist.cutoff; i++) {
    r -= dist.probs[i];
    if (r <= 0) return dist.indices[i];
  }
  return dist.indices[dist.cutoff - 1];
}

function sampleResidual(pDist, qDraft, rng) {
  const res = new Float32Array(pDist.cutoff);
  let sum = 0;
  if (qDraft.single !== undefined) {
    for (let i = 0; i < pDist.cutoff; i++) {
      const p = pDist.indices[i] === qDraft.single ? 0.0 : pDist.probs[i];
      res[i] = p;
      sum += p;
    }
  } else {
    for (let i = 0; i < pDist.cutoff; i++) {
      const id = pDist.indices[i];
      const d = Math.max(0.0, pDist.probs[i] - (qDraft.probMap.get(id) || 0.0));
      res[i] = d;
      sum += d;
    }
  }
  if (sum <= 1e-12) {
    return sampleFromDist(pDist, rng);
  }
  let r = rng() * sum;
  for (let i = 0; i < pDist.cutoff; i++) {
    r -= res[i];
    if (r <= 0) return pDist.indices[i];
  }
  return pDist.indices[pDist.cutoff - 1];
}

async function rollbackEngine(engine, startPos, acceptedWithT0, totalVerified) {
  if (acceptedWithT0.length === totalVerified) return;
  if (engine.hasConvLayers) {
    engine.restoreConvStatesGpu();
    engine.rollbackTo(startPos);
    await engine.forward(acceptedWithT0, { skipLogits: true });
  } else {
    engine.rollbackTo(startPos + acceptedWithT0.length);
  }
}

async function verifyDraftBatch(
  targetEngine,
  curToken,
  draftTokens,
  draftDists,
  historyWithCur,
  sampling,
  eosSet,
  rng
) {
  const V = targetEngine.bp.vocab_size;
  const m = draftTokens.length;
  const startPos = targetEngine.curPos;
  if (targetEngine.hasConvLayers) {
    targetEngine.snapshotConvStatesGpu();
  }

  const ver = await targetEngine.forwardVerify([curToken, ...draftTokens]);
  const acceptedDrafts = [];
  const curHist = [...historyWithCur];
  const isGreedy = (sampling.temperature ?? 0.0) <= 0;
  let nextTok = null;
  let hitEos = false;

  for (let j = 0; j < m; j++) {
    const rowLogits = ver.logits.subarray(j * V, (j + 1) * V);
    const cand = draftTokens[j];
    if (isGreedy) {
      const targetTok = sampleToken(rowLogits, curHist, sampling);
      if (targetTok === cand) {
        if (eosSet.has(cand)) {
          hitEos = true;
          break;
        }
        acceptedDrafts.push(cand);
        curHist.push(cand);
      } else {
        nextTok = targetTok;
        break;
      }
    } else {
      const pTarget = computeSamplingDistribution(rowLogits, curHist, sampling);
      const pMap = buildProbMap(pTarget);
      const pProb = pMap.get(cand) || 0.0;
      const qProb = draftDists ? (draftDists[j].probMap.get(cand) || 1e-12) : 1.0;
      if (rng() <= Math.min(1.0, pProb / Math.max(1e-12, qProb))) {
        if (eosSet.has(cand)) {
          hitEos = true;
          break;
        }
        acceptedDrafts.push(cand);
        curHist.push(cand);
      } else {
        const qObj = draftDists ? draftDists[j] : { single: cand };
        nextTok = sampleResidual(pTarget, qObj, rng);
        break;
      }
    }
  }

  if (!hitEos && acceptedDrafts.length === m) {
    const bonusLogits = ver.logits.subarray(m * V, (m + 1) * V);
    nextTok = sampleToken(bonusLogits, curHist, { ...sampling, rng });
  }

  if (acceptedDrafts.length < m || hitEos) {
    await rollbackEngine(targetEngine, startPos, [curToken, ...acceptedDrafts], 1 + m);
  }

  return { acceptedDrafts, nextTok, hitEos };
}

function defaultEosSet(bp) {
  return new Set(
    Array.isArray(bp.eos_token_ids) && bp.eos_token_ids.length
      ? bp.eos_token_ids
      : [bp.eos_token_id]
  );
}

export class NgramLookupSpeculator {
  constructor({ k = 4, minNgram = 2, maxNgram = 4 } = {}) {
    this.k = k;
    this.minNgram = minNgram;
    this.maxNgram = maxNgram;
  }

  propose(history, maxCount, eosSet = new Set()) {
    const len = history.length;
    const cap = Math.min(this.k, maxCount);
    if (cap <= 0 || len < this.minNgram + 1) return [];

    const maxN = Math.min(this.maxNgram, len - 1);
    for (let n = maxN; n >= this.minNgram; n--) {
      const patStart = len - n;
      for (let i = patStart - 1; i >= 0; i--) {
        let match = true;
        for (let j = 0; j < n; j++) {
          if (history[i + j] !== history[patStart + j]) {
            match = false;
            break;
          }
        }
        if (match) {
          const out = [];
          const srcStart = i + n;
          for (let d = 0; d < cap && srcStart + d < len; d++) {
            const tok = history[srcStart + d];
            if (eosSet.has(tok)) break;
            out.push(tok);
          }
          if (out.length > 0) return out;
        }
      }
    }
    return [];
  }

  async generate(engine, promptIds, {
    maxNewTokens = 64,
    sampling = { temperature: 0.0 },
    eosSet = null,
    onToken = null,
    shouldAbort = null,
    rng = Math.random,
    skipPrefill = false,
    initialLogits = null,
  } = {}) {
    const eos = eosSet || defaultEosSet(engine.bp);
    const tPrefill0 = performance.now();
    let baseLogits = initialLogits;
    if (!skipPrefill) {
      engine.reset();
      const fwd0 = await engine.forward(promptIds);
      baseLogits = fwd0.logits;
    }
    const prefillMs = skipPrefill ? 0 : (performance.now() - tPrefill0);

    const generatedIds = [];
    const history = [...promptIds];
    let curToken = sampleToken(baseLogits, history, { ...sampling, rng });
    let speculativeProposed = 0;
    let speculativeAccepted = 0;
    let rounds = 0;
    const tDec0 = performance.now();

    while (generatedIds.length < maxNewTokens) {
      if (shouldAbort && shouldAbort()) break;
      if (eos.has(curToken)) break;

      rounds++;
      generatedIds.push(curToken);
      history.push(curToken);
      if (onToken) onToken(curToken, generatedIds.length - 1);

      const remaining = maxNewTokens - generatedIds.length;
      if (remaining <= 0) break;

      const drafts = this.propose(history, remaining, eos);
      if (drafts.length === 0) {
        const fwd = await engine.forward([curToken]);
        curToken = sampleToken(fwd.logits, history, { ...sampling, rng });
        continue;
      }

      speculativeProposed += drafts.length;
      const res = await verifyDraftBatch(
        engine,
        curToken,
        drafts,
        null,
        history,
        sampling,
        eos,
        rng
      );
      speculativeAccepted += res.acceptedDrafts.length;
      for (const accTok of res.acceptedDrafts) {
        generatedIds.push(accTok);
        history.push(accTok);
        if (onToken) onToken(accTok, generatedIds.length - 1);
      }
      if (res.hitEos || generatedIds.length >= maxNewTokens) break;
      curToken = res.nextTok;
    }

    const decodeMs = performance.now() - tDec0;
    return {
      generatedIds,
      prefillMs,
      decodeMs,
      decodeTokPerSec: generatedIds.length > 0 ? (generatedIds.length / (decodeMs / 1000)) : 0,
      speculativeProposed,
      speculativeAccepted,
      rounds,
    };
  }
}

export class DraftTargetSpeculator {
  constructor(targetEngine, draftEngine, { k = 4 } = {}) {
    this.targetEngine = targetEngine;
    this.draftEngine = draftEngine;
    this.k = k;
  }

  async generate(promptIds, {
    maxNewTokens = 48,
    sampling = { temperature: 0.0 },
    eosSet = null,
    onToken = null,
    shouldAbort = null,
    rng = Math.random,
  } = {}) {
    const eos = eosSet || defaultEosSet(this.targetEngine.bp);
    const isGreedy = (sampling.temperature ?? 0.0) <= 0;

    const tPrefill0 = performance.now();
    this.targetEngine.reset();
    this.draftEngine.reset();
    const tgtPre = await this.targetEngine.forward(promptIds);
    await this.draftEngine.forward(promptIds, { skipLogits: true });
    const prefillMs = performance.now() - tPrefill0;

    const generatedIds = [];
    const history = [...promptIds];
    let curToken = sampleToken(tgtPre.logits, history, { ...sampling, rng });
    let speculativeProposed = 0;
    let speculativeAccepted = 0;
    let rounds = 0;
    const tDec0 = performance.now();

    while (generatedIds.length < maxNewTokens) {
      if (shouldAbort && shouldAbort()) break;
      if (eos.has(curToken)) break;

      rounds++;
      generatedIds.push(curToken);
      history.push(curToken);
      if (onToken) onToken(curToken, generatedIds.length - 1);

      const remaining = maxNewTokens - generatedIds.length;
      if (remaining <= 0) break;

      const draftCount = Math.min(this.k, remaining);
      const startPos = this.draftEngine.curPos;
      if (this.draftEngine.hasConvLayers) {
        this.draftEngine.snapshotConvStatesGpu();
      }

      const draftTokens = [];
      const draftDists = isGreedy ? null : [];
      const draftHist = [...history];
      let dFwd = await this.draftEngine.forward([curToken]);

      for (let s = 0; s < draftCount; s++) {
        let dTok;
        if (isGreedy) {
          dTok = sampleToken(dFwd.logits, draftHist, sampling);
        } else {
          const dist = computeSamplingDistribution(dFwd.logits, draftHist, sampling);
          draftDists.push({ dist, probMap: buildProbMap(dist) });
          dTok = sampleFromDist(dist, rng);
        }
        if (eos.has(dTok)) break;
        draftTokens.push(dTok);
        draftHist.push(dTok);
        if (s + 1 < draftCount) {
          dFwd = await this.draftEngine.forward([dTok]);
        }
      }

      if (draftTokens.length === 0) {
        const tFwd = await this.targetEngine.forward([curToken]);
        curToken = sampleToken(tFwd.logits, history, { ...sampling, rng });
        continue;
      }

      speculativeProposed += draftTokens.length;
      const res = await verifyDraftBatch(
        this.targetEngine,
        curToken,
        draftTokens,
        draftDists,
        history,
        sampling,
        eos,
        rng
      );
      speculativeAccepted += res.acceptedDrafts.length;

      for (const accTok of res.acceptedDrafts) {
        generatedIds.push(accTok);
        history.push(accTok);
        if (onToken) onToken(accTok, generatedIds.length - 1);
      }

      if (res.hitEos || generatedIds.length >= maxNewTokens) break;

      // Keep draftEngine synced to targetEngine.curPos (startPos + 1 + acceptedDrafts.length)
      const targetPos = this.targetEngine.curPos;
      if (this.draftEngine.curPos < targetPos) {
        await this.draftEngine.forward([res.acceptedDrafts[res.acceptedDrafts.length - 1]], { skipLogits: true });
      } else if (this.draftEngine.curPos > targetPos) {
        if (this.draftEngine.hasConvLayers) {
          this.draftEngine.restoreConvStatesGpu();
          this.draftEngine.rollbackTo(startPos);
          await this.draftEngine.forward([curToken, ...res.acceptedDrafts], { skipLogits: true });
        } else {
          this.draftEngine.rollbackTo(targetPos);
        }
      }

      curToken = res.nextTok;
    }

    const decodeMs = performance.now() - tDec0;
    return {
      generatedIds,
      prefillMs,
      decodeMs,
      decodeTokPerSec: generatedIds.length > 0 ? (generatedIds.length / (decodeMs / 1000)) : 0,
      speculativeProposed,
      speculativeAccepted,
      rounds,
    };
  }
}
