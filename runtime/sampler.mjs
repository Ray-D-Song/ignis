/**
 * Token Sampler: Greedy, Temperature, Top-K, Top-P, Min-P, Repetition Penalty.
 */

export function sampleToken(
  logits,
  historyIds = [],
  { temperature = 0.0, topK = 40, topP = 0.95, minP = 0.0, repetitionPenalty = 1.0 } = {}
) {
  const V = logits.length;
  // Work on a copy only if penalty/sampling modifies logits
  if (temperature <= 0 && repetitionPenalty === 1.0) {
    let bestIdx = 0;
    let bestVal = logits[0];
    for (let i = 1; i < V; i++) {
      if (logits[i] > bestVal) {
        bestVal = logits[i];
        bestIdx = i;
      }
    }
    return bestIdx;
  }

  const work = new Float32Array(logits);
  if (repetitionPenalty !== 1.0 && historyIds.length > 0) {
    const seen = new Set(historyIds);
    for (const id of seen) {
      if (id >= 0 && id < V) {
        work[id] = work[id] > 0 ? work[id] / repetitionPenalty : work[id] * repetitionPenalty;
      }
    }
  }

  if (temperature <= 0) {
    let bestIdx = 0;
    let bestVal = work[0];
    for (let i = 1; i < V; i++) {
      if (work[i] > bestVal) {
        bestVal = work[i];
        bestIdx = i;
      }
    }
    return bestIdx;
  }

  // Top-K filtering via partial scan
  const k = topK > 0 && topK < V ? topK : V;
  const candidates = [];
  for (let i = 0; i < V; i++) {
    candidates.push(i);
  }
  candidates.sort((a, b) => work[b] - work[a]);
  const topSlice = candidates.slice(0, k);

  const maxLogit = work[topSlice[0]];
  let sumExp = 0;
  const probs = new Float32Array(topSlice.length);
  for (let i = 0; i < topSlice.length; i++) {
    const p = Math.exp((work[topSlice[i]] - maxLogit) / temperature);
    probs[i] = p;
    sumExp += p;
  }
  for (let i = 0; i < probs.length; i++) {
    probs[i] /= sumExp;
  }

  // Min-P & Top-P truncation
  const maxProb = probs[0];
  let cum = 0;
  let cutoff = probs.length;
  for (let i = 0; i < probs.length; i++) {
    if (minP > 0 && probs[i] < maxProb * minP) {
      cutoff = Math.max(1, i);
      break;
    }
    cum += probs[i];
    if (cum >= topP) {
      cutoff = i + 1;
      break;
    }
  }

  let finalSum = 0;
  for (let i = 0; i < cutoff; i++) finalSum += probs[i];
  let r = Math.random() * finalSum;
  for (let i = 0; i < cutoff; i++) {
    r -= probs[i];
    if (r <= 0) return topSlice[i];
  }
  return topSlice[cutoff - 1];
}
