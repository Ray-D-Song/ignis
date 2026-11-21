/**
 * Token Sampler: Greedy, Temperature, Top-K (O(V) min-heap), Top-P, Min-P, Repetition Penalty.
 */

function argmax(arr, len) {
  let bestIdx = 0;
  let bestVal = arr[0];
  for (let i = 1; i < len; i++) {
    if (arr[i] > bestVal) {
      bestVal = arr[i];
      bestIdx = i;
    }
  }
  return bestIdx;
}

/**
 * O(V log K) min-heap Top-K selection using typed arrays (zero GC churn over 131K vocab).
 */
function selectTopK(logits, V, k) {
  const heapIdx = new Int32Array(k);
  const heapVal = new Float32Array(k);

  for (let i = 0; i < k; i++) {
    heapIdx[i] = i;
    heapVal[i] = logits[i];
  }
  // Heapify min-heap
  for (let i = (k >> 1) - 1; i >= 0; i--) {
    siftDown(heapIdx, heapVal, k, i);
  }

  let minVal = heapVal[0];
  for (let i = k; i < V; i++) {
    const v = logits[i];
    if (v > minVal) {
      heapIdx[0] = i;
      heapVal[0] = v;
      siftDown(heapIdx, heapVal, k, 0);
      minVal = heapVal[0];
    }
  }

  // Sort the small k-element array descending
  const pairs = new Array(k);
  for (let i = 0; i < k; i++) {
    pairs[i] = { idx: heapIdx[i], val: heapVal[i] };
  }
  pairs.sort((a, b) => b.val - a.val);
  return pairs;
}

function siftDown(hIdx, hVal, size, root) {
  const idx0 = hIdx[root];
  const val0 = hVal[root];
  let i = root;
  while (true) {
    const left = (i << 1) + 1;
    if (left >= size) break;
    let smallest = left;
    const right = left + 1;
    if (right < size && hVal[right] < hVal[left]) {
      smallest = right;
    }
    if (hVal[smallest] >= val0) break;
    hIdx[i] = hIdx[smallest];
    hVal[i] = hVal[smallest];
    i = smallest;
  }
  hIdx[i] = idx0;
  hVal[i] = val0;
}

export function sampleToken(
  logits,
  historyIds = [],
  { temperature = 0.0, topK = 40, topP = 0.95, minP = 0.0, repetitionPenalty = 1.0, rng = Math.random } = {}
) {
  const V = logits.length;
  if (temperature <= 0 && repetitionPenalty === 1.0) {
    return argmax(logits, V);
  }

  // Apply repetition penalty in-place on a copy only when needed
  let work = logits;
  if (repetitionPenalty !== 1.0 && historyIds.length > 0) {
    work = new Float32Array(logits);
    const seen = new Set(historyIds);
    for (const id of seen) {
      if (id >= 0 && id < V) {
        work[id] = work[id] > 0 ? work[id] / repetitionPenalty : work[id] * repetitionPenalty;
      }
    }
  }

  if (temperature <= 0) {
    return argmax(work, V);
  }

  const k = topK > 0 && topK < V ? topK : Math.min(V, 256);
  const topPairs = selectTopK(work, V, k);

  const maxLogit = topPairs[0].val;
  let sumExp = 0;
  const probs = new Float32Array(k);
  for (let i = 0; i < k; i++) {
    const p = Math.exp((topPairs[i].val - maxLogit) / temperature);
    probs[i] = p;
    sumExp += p;
  }
  for (let i = 0; i < k; i++) {
    probs[i] /= sumExp;
  }

  const maxProb = probs[0];
  let cum = 0;
  let cutoff = k;
  for (let i = 0; i < k; i++) {
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
  let r = rng() * finalSum;
  for (let i = 0; i < cutoff; i++) {
    r -= probs[i];
    if (r <= 0) return topPairs[i].idx;
  }
  return topPairs[cutoff - 1].idx;
}
