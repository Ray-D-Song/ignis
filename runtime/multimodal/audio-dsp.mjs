/**
 * Pure-JS Audio DSP Primitives for Ignis Multimodal Models (Stage 5B & 5C):
 * - Mixed-radix (2/3/5) Complex FFT & Hermitian iRFFT (Float64 exact)
 * - Slaney & Kaldi/HTK Mel Filter Banks
 * - LFM3-Omni 128-bin Log-Mel Extractor (16 kHz)
 * - LiquidCodec 1280-pt 24 kHz iSTFT Synthesis
 * - 16-bit PCM WAV Encoder / Decoder & Resampler
 */

const FFT_CACHE = new Map();

function getTwiddles(n) {
  let entry = FFT_CACHE.get(n);
  if (entry) return entry;
  // Factor n into factors of 2, 3, 5, else fallback to DFT
  const factors = [];
  let rem = n;
  for (const r of [4, 2, 5, 3]) {
    while (rem % r === 0) {
      factors.push(r);
      rem /= r;
    }
  }
  entry = { n, factors, rem };
  FFT_CACHE.set(n, entry);
  return entry;
}

/**
 * Out-of-place recursive Stockham/Cooley-Tukey radix-{2,3,4,5} FFT in float64.
 * inverse=false -> exp(-i 2pi k n / N) (unnormalized)
 * inverse=true  -> exp(+i 2pi k n / N) (unnormalized; caller divides by N if backward)
 */
export function fftComplex(reIn, imIn, reOut, imOut, n, inverse = false) {
  if (n === 1) {
    reOut[0] = reIn[0];
    imOut[0] = imIn[0];
    return;
  }
  const { factors, rem } = getTwiddles(n);
  if (rem !== 1) {
    const sign = inverse ? 1 : -1;
    for (let k = 0; k < n; k++) {
      let sr = 0;
      let si = 0;
      const step = (sign * 2 * Math.PI * k) / n;
      for (let j = 0; j < n; j++) {
        const ang = step * j;
        const c = Math.cos(ang);
        const s = Math.sin(ang);
        sr += reIn[j] * c - imIn[j] * s;
        si += reIn[j] * s + imIn[j] * c;
      }
      reOut[k] = sr;
      imOut[k] = si;
    }
    return;
  }

  const r = factors[0];
  const m = n / r;
  const subReIn = new Float64Array(m);
  const subImIn = new Float64Array(m);
  const subReOut = new Float64Array(m);
  const subImOut = new Float64Array(m);
  const yRe = new Float64Array(r * m);
  const yIm = new Float64Array(r * m);

  for (let q = 0; q < r; q++) {
    for (let j = 0; j < m; j++) {
      subReIn[j] = reIn[j * r + q];
      subImIn[j] = imIn[j * r + q];
    }
    fftComplex(subReIn, subImIn, subReOut, subImOut, m, inverse);
    yRe.set(subReOut, q * m);
    yIm.set(subImOut, q * m);
  }

  const sign = inverse ? 1 : -1;
  const twoPiN = (sign * 2 * Math.PI) / n;
  const twoPiR = (sign * 2 * Math.PI) / r;

  // Precompute r x r radix DFT twiddles
  const wrCos = new Float64Array(r * r);
  const wrSin = new Float64Array(r * r);
  for (let p = 0; p < r; p++) {
    for (let q = 0; q < r; q++) {
      const ang = twoPiR * p * q;
      wrCos[p * r + q] = Math.cos(ang);
      wrSin[p * r + q] = Math.sin(ang);
    }
  }

  for (let k = 0; k < m; k++) {
    const tRe = new Float64Array(r);
    const tIm = new Float64Array(r);
    for (let q = 0; q < r; q++) {
      const yr = yRe[q * m + k];
      const yi = yIm[q * m + k];
      if (q === 0) {
        tRe[0] = yr;
        tIm[0] = yi;
      } else {
        const ang = twoPiN * q * k;
        const c = Math.cos(ang);
        const s = Math.sin(ang);
        tRe[q] = yr * c - yi * s;
        tIm[q] = yr * s + yi * c;
      }
    }
    for (let p = 0; p < r; p++) {
      let sr = 0;
      let si = 0;
      const base = p * r;
      for (let q = 0; q < r; q++) {
        const c = wrCos[base + q];
        const s = wrSin[base + q];
        sr += tRe[q] * c - tIm[q] * s;
        si += tRe[q] * s + tIm[q] * c;
      }
      reOut[p * m + k] = sr;
      imOut[p * m + k] = si;
    }
  }
}

export function rfft(realFrame, nFft) {
  const reIn = new Float64Array(nFft);
  const imIn = new Float64Array(nFft);
  reIn.set(realFrame);
  const reOut = new Float64Array(nFft);
  const imOut = new Float64Array(nFft);
  fftComplex(reIn, imIn, reOut, imOut, nFft, false);
  const bins = (nFft >> 1) + 1;
  return { re: reOut.subarray(0, bins), im: imOut.subarray(0, bins) };
}

export function irfftBackward(specRe, specIm, nFft) {
  const half = nFft >> 1;
  const reIn = new Float64Array(nFft);
  const imIn = new Float64Array(nFft);
  reIn[0] = specRe[0];
  imIn[0] = 0;
  for (let k = 1; k < half; k++) {
    reIn[k] = specRe[k];
    imIn[k] = specIm[k];
    reIn[nFft - k] = specRe[k];
    imIn[nFft - k] = -specIm[k];
  }
  if (nFft % 2 === 0) {
    reIn[half] = specRe[half];
    imIn[half] = 0;
  }
  const reOut = new Float64Array(nFft);
  const imOut = new Float64Array(nFft);
  fftComplex(reIn, imIn, reOut, imOut, nFft, true);
  const out = new Float32Array(nFft);
  const invN = 1.0 / nFft;
  for (let i = 0; i < nFft; i++) {
    out[i] = reOut[i] * invN;
  }
  return out;
}

export function hannWindowPeriodic(winLength) {
  const w = new Float64Array(winLength);
  for (let i = 0; i < winLength; i++) {
    w[i] = 0.5 * (1.0 - Math.cos((2.0 * Math.PI * i) / winLength));
  }
  return w;
}

export function hertzToMel(freq, melScale = 'htk') {
  if (melScale === 'htk') return 2595.0 * Math.log10(1.0 + freq / 700.0);
  if (melScale === 'kaldi') return 1127.0 * Math.log(1.0 + freq / 700.0);
  const minLogHz = 1000.0;
  const minLogMel = 15.0;
  const logStep = 27.0 / Math.log(6.4);
  return freq >= minLogHz ? minLogMel + Math.log(freq / minLogHz) * logStep : (3.0 * freq) / 200.0;
}

export function melToHertz(mel, melScale = 'htk') {
  if (melScale === 'htk') return 700.0 * (Math.pow(10.0, mel / 2595.0) - 1.0);
  if (melScale === 'kaldi') return 700.0 * (Math.exp(mel / 1127.0) - 1.0);
  const minLogHz = 1000.0;
  const minLogMel = 15.0;
  const logStep = Math.log(6.4) / 27.0;
  return mel >= minLogMel ? minLogHz * Math.exp(logStep * (mel - minLogMel)) : (200.0 * mel) / 3.0;
}

/**
 * Builds mel filter bank matrix [numMelFilters, numFreqBins] (transposed vs HF for fast bin dot).
 */
export function buildMelFilterBank({
  numFreqBins,
  numMelFilters,
  minFreq = 0.0,
  maxFreq = 8000.0,
  samplingRate = 16000,
  norm = 'slaney',
  melScale = 'slaney',
} = {}) {
  const melMin = hertzToMel(minFreq, melScale);
  const melMax = hertzToMel(maxFreq, melScale);
  const nPts = numMelFilters + 2;
  const filterFreqs = new Float64Array(nPts);
  for (let i = 0; i < nPts; i++) {
    const m = melMin + ((melMax - melMin) * i) / (nPts - 1);
    filterFreqs[i] = melToHertz(m, melScale);
  }
  const fftFreqs = new Float64Array(numFreqBins);
  const halfSr = Math.floor(samplingRate / 2);
  for (let b = 0; b < numFreqBins; b++) {
    fftFreqs[b] = (halfSr * b) / (numFreqBins - 1);
  }

  // filters[mel, bin]
  const bank = new Float32Array(numMelFilters * numFreqBins);
  for (let m = 0; m < numMelFilters; m++) {
    const fLow = filterFreqs[m];
    const fCtr = filterFreqs[m + 1];
    const fHi = filterFreqs[m + 2];
    const dLow = fCtr - fLow;
    const dHi = fHi - fCtr;
    const enorm = norm === 'slaney' ? 2.0 / (fHi - fLow) : 1.0;
    const rowOff = m * numFreqBins;
    for (let b = 0; b < numFreqBins; b++) {
      const f = fftFreqs[b];
      const down = (f - fLow) / dLow;
      const up = (fHi - f) / dHi;
      const w = Math.max(0.0, Math.min(down, up)) * enorm;
      bank[rowOff + b] = w;
    }
  }
  return bank;
}

let LFM3_MEL_BANK = null;
let LFM3_HANN = null;

/**
 * Exact Lfm3OmniFeatureExtractor log-mel frontend (16 kHz mono -> [128, numFrames] row-major).
 */
export function extractLfm3LogMel(pcm16k, {
  featureSize = 128,
  samplingRate = 16000,
  nFft = 400,
  hopLength = 160,
  winLength = 400,
  logFloor = -12.0,
  normOffset = 4.0,
  normDivisor = 4.0,
} = {}) {
  if (!LFM3_MEL_BANK || nFft !== 400 || featureSize !== 128) {
    LFM3_MEL_BANK = buildMelFilterBank({
      numFreqBins: (nFft >> 1) + 1,
      numMelFilters: featureSize,
      minFreq: 0.0,
      maxFreq: samplingRate / 2,
      samplingRate,
      norm: 'slaney',
      melScale: 'slaney',
    });
  }
  if (!LFM3_HANN || LFM3_HANN.length !== winLength) {
    LFM3_HANN = hannWindowPeriodic(winLength);
  }

  const N = pcm16k.length;
  const padLen = N < nFft ? nFft : N + ((-(N - nFft) % hopLength + hopLength) % hopLength);
  const padded = new Float64Array(padLen);
  for (let i = 0; i < N; i++) padded[i] = pcm16k[i];

  const numFrames = 1 + Math.floor((padLen - nFft) / hopLength);
  const numBins = (nFft >> 1) + 1;
  const features = new Float32Array(featureSize * numFrames);
  const frameBuf = new Float64Array(nFft);
  const powerBuf = new Float64Array(numBins);

  for (let t = 0; t < numFrames; t++) {
    const start = t * hopLength;
    for (let j = 0; j < winLength; j++) {
      frameBuf[j] = padded[start + j] * LFM3_HANN[j];
    }
    const { re, im } = rfft(frameBuf, nFft);
    for (let b = 0; b < numBins; b++) {
      powerBuf[b] = re[b] * re[b] + im[b] * im[b];
    }
    for (let m = 0; m < featureSize; m++) {
      const mOff = m * numBins;
      let acc = 0.0;
      for (let b = 0; b < numBins; b++) {
        acc += LFM3_MEL_BANK[mOff + b] * powerBuf[b];
      }
      const lm = Math.max(logFloor, Math.log10(Math.max(acc, 1e-38)));
      features[m * numFrames + t] = (lm + normOffset) / normDivisor;
    }
  }

  return {
    features,
    numMelBins: featureSize,
    numFrames,
    numEncoderTokens: Math.ceil(numFrames / 8),
  };
}

let GEMMA4_MEL_BANK = null;
let GEMMA4_HANN = null;

/**
 * Exact Gemma4AudioFeatureExtractor log-mel frontend:
 * - 16 kHz mono, padded to multiple of 128 samples, semicausal left-pad 160 samples
 * - 320-sample (20ms) periodic Hann window, 160-sample hop (unfold size=321 -> drop last sample)
 * - 512-pt RFFT magnitude |stft|, 128-bin HTK mel (norm=None, 0..8000 Hz), ln(mel + 0.001)
 * - Masked by sampleMask[t*hop + 320] === 1; returns row-major [numFrames, 128] + subsample valid counts
 */
export function extractGemma4LogMel(pcm16k, {
  featureSize = 128,
  samplingRate = 16000,
  frameLength = 320,
  hopLength = 160,
  fftLength = 512,
  melFloor = 0.001,
  padToMultipleOf = 128,
  maxLength = 480000,
} = {}) {
  if (!GEMMA4_MEL_BANK || fftLength !== 512 || featureSize !== 128) {
    GEMMA4_MEL_BANK = buildMelFilterBank({
      numFreqBins: (fftLength >> 1) + 1,
      numMelFilters: featureSize,
      minFreq: 0.0,
      maxFreq: samplingRate / 2,
      samplingRate,
      norm: null,
      melScale: 'htk',
    });
  }
  if (!GEMMA4_HANN || GEMMA4_HANN.length !== frameLength) {
    GEMMA4_HANN = hannWindowPeriodic(frameLength);
  }

  const rawLen = Math.min(pcm16k.length, maxLength);
  const paddedLen = Math.max(padToMultipleOf, Math.ceil(rawLen / padToMultipleOf) * padToMultipleOf);
  const padLeft = frameLength >> 1; // 160
  const totalWaveLen = padLeft + paddedLen;
  const wave = new Float64Array(totalWaveLen);
  const sampleMask = new Uint8Array(totalWaveLen);
  for (let i = 0; i < rawLen; i++) {
    wave[padLeft + i] = pcm16k[i];
    sampleMask[padLeft + i] = 1;
  }

  const unfoldSize = frameLength + 1; // 321
  const numFrames = totalWaveLen < unfoldSize ? 0 : Math.floor((totalWaveLen - unfoldSize) / hopLength) + 1;
  const numBins = (fftLength >> 1) + 1;
  const features = new Float32Array(numFrames * featureSize);
  const frameMask = new Uint8Array(numFrames);
  const frameBuf = new Float64Array(fftLength);
  const magBuf = new Float64Array(numBins);

  let validMelFrames = 0;
  let validSub1 = 0;
  let validAudioTokens = 0;

  for (let t = 0; t < numFrames; t++) {
    const start = t * hopLength;
    const isValid = sampleMask[start + unfoldSize - 1] === 1;
    frameMask[t] = isValid ? 1 : 0;
    if (!isValid) continue;
    validMelFrames++;
    if (t % 2 === 0) validSub1++;
    if (t % 4 === 0) validAudioTokens++;

    for (let j = 0; j < frameLength; j++) {
      frameBuf[j] = wave[start + j] * GEMMA4_HANN[j];
    }
    for (let j = frameLength; j < fftLength; j++) frameBuf[j] = 0;

    const { re, im } = rfft(frameBuf, fftLength);
    for (let b = 0; b < numBins; b++) {
      magBuf[b] = Math.hypot(re[b], im[b]);
    }
    const rowOff = t * featureSize;
    for (let m = 0; m < featureSize; m++) {
      const mOff = m * numBins;
      let acc = 0.0;
      for (let b = 0; b < numBins; b++) {
        acc += GEMMA4_MEL_BANK[mOff + b] * magBuf[b];
      }
      features[rowOff + m] = Math.log(acc + melFloor);
    }
  }

  return {
    features,
    frameMask,
    numMelBins: featureSize,
    numFrames,
    validMelFrames,
    validSub1,
    validAudioTokens,
  };
}

/**
 * LiquidCodecISTFT: converts decoder output [numStftFrames, 2 * 641] -> 24 kHz float32 PCM.
 */
export function liquidCodecIstft(linOut, numStftFrames, window, {
  nFft = 1280,
  hopLength = 320,
  winLength = 1280,
} = {}) {
  if (numStftFrames === 0) return new Float32Array(0);
  const bins = (nFft >> 1) + 1; // 641
  const stride = 2 * bins;       // 1282
  const outSize = (numStftFrames - 1) * hopLength + winLength;
  const audio = new Float64Array(outSize);
  const envelope = new Float64Array(outSize);
  const specRe = new Float64Array(bins);
  const specIm = new Float64Array(bins);

  for (let t = 0; t < numStftFrames; t++) {
    const base = t * stride;
    for (let b = 0; b < bins; b++) {
      const mag = Math.exp(linOut[base + b]);
      const ang = linOut[base + bins + b];
      specRe[b] = mag * Math.cos(ang);
      specIm[b] = mag * Math.sin(ang);
    }
    const td = irfftBackward(specRe, specIm, nFft);
    const dstStart = t * hopLength;
    for (let i = 0; i < winLength; i++) {
      const w = window[i];
      audio[dstStart + i] += td[i] * w;
      envelope[dstStart + i] += w * w;
    }
  }

  const trim = (winLength - hopLength) >> 1; // 480
  const trimmedLen = outSize - 2 * trim;
  const result = new Float32Array(trimmedLen);
  for (let i = 0; i < trimmedLen; i++) {
    const env = envelope[trim + i];
    result[i] = env > 1e-12 ? audio[trim + i] / env : 0.0;
  }
  return result;
}

/**
 * Simple linear resampler (sufficient & fast for 44.1/48kHz -> 16kHz speech UI uploads).
 */
export function resampleLinear(samples, srcRate, dstRate) {
  if (srcRate === dstRate) return samples instanceof Float32Array ? samples : Float32Array.from(samples);
  const ratio = srcRate / dstRate;
  const outLen = Math.max(1, Math.round(samples.length / ratio));
  const out = new Float32Array(outLen);
  for (let i = 0; i < outLen; i++) {
    const pos = i * ratio;
    const i0 = Math.floor(pos);
    const i1 = Math.min(samples.length - 1, i0 + 1);
    const frac = pos - i0;
    out[i] = samples[i0] * (1.0 - frac) + samples[i1] * frac;
  }
  return out;
}

/**
 * Encode mono float32 PCM [-1, 1] as 16-bit signed PCM WAV bytes.
 */
export function encodeWavPcm16(samples, sampleRate = 24000) {
  const numSamples = samples.length;
  const buf = new ArrayBuffer(44 + numSamples * 2);
  const view = new DataView(buf);
  const writeStr = (off, s) => {
    for (let i = 0; i < s.length; i++) view.setUint8(off + i, s.charCodeAt(i));
  };
  writeStr(0, 'RIFF');
  view.setUint32(4, 36 + numSamples * 2, true);
  writeStr(8, 'WAVE');
  writeStr(12, 'fmt ');
  view.setUint32(16, 16, true);
  view.setUint16(20, 1, true); // PCM
  view.setUint16(22, 1, true); // mono
  view.setUint32(24, sampleRate, true);
  view.setUint32(28, sampleRate * 2, true);
  view.setUint16(32, 2, true);
  view.setUint16(34, 16, true);
  writeStr(36, 'data');
  view.setUint32(40, numSamples * 2, true);
  for (let i = 0; i < numSamples; i++) {
    const clamped = Math.max(-1.0, Math.min(1.0, samples[i]));
    view.setInt16(44 + i * 2, clamped < 0 ? Math.round(clamped * 32768) : Math.round(clamped * 32767), true);
  }
  return new Uint8Array(buf);
}

/**
 * Decode 16-bit PCM or 32-bit float WAV buffer to mono Float32Array + sampleRate.
 */
export function decodeWav(arrayBuffer) {
  const view = new DataView(arrayBuffer);
  const tag = (off) => String.fromCharCode(
    view.getUint8(off), view.getUint8(off + 1), view.getUint8(off + 2), view.getUint8(off + 3)
  );
  if (tag(0) !== 'RIFF' || tag(8) !== 'WAVE') throw new Error('Invalid WAV header');

  let pos = 12;
  let audioFormat = 1;
  let channels = 1;
  let sampleRate = 16000;
  let bitsPerSample = 16;
  let dataOffset = 0;
  let dataBytes = 0;

  while (pos + 8 <= arrayBuffer.byteLength) {
    const id = tag(pos);
    const size = view.getUint32(pos + 4, true);
    if (id === 'fmt ') {
      audioFormat = view.getUint16(pos + 8, true);
      channels = view.getUint16(pos + 10, true);
      sampleRate = view.getUint32(pos + 12, true);
      bitsPerSample = view.getUint16(pos + 22, true);
    } else if (id === 'data') {
      dataOffset = pos + 8;
      dataBytes = size;
      break;
    }
    pos += 8 + size + (size & 1);
  }

  if (!dataBytes) throw new Error('WAV missing data chunk');
  const bytesPerSample = bitsPerSample >> 3;
  const totalFrames = Math.floor(dataBytes / (bytesPerSample * channels));
  const mono = new Float32Array(totalFrames);

  for (let i = 0; i < totalFrames; i++) {
    let sum = 0;
    for (let c = 0; c < channels; c++) {
      const off = dataOffset + (i * channels + c) * bytesPerSample;
      if (audioFormat === 3 && bitsPerSample === 32) {
        sum += view.getFloat32(off, true);
      } else if (bitsPerSample === 16) {
        sum += view.getInt16(off, true) / 32768.0;
      } else if (bitsPerSample === 32) {
        sum += view.getInt32(off, true) / 2147483648.0;
      } else if (bitsPerSample === 8) {
        sum += (view.getUint8(off) - 128) / 128.0;
      } else {
        throw new Error(`Unsupported WAV bitsPerSample=${bitsPerSample}`);
      }
    }
    mono[i] = sum / channels;
  }

  return { samples: mono, sampleRate };
}
