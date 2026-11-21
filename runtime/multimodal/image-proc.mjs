/**
 * Pure-JS Image Preprocessors & Patchifiers for Ignis Multimodal Models (Stage 5B & 5C).
 * - LFM3-VL dynamic smartResize + Bicubic resampling + merge-block patchifier
 * - 2D Axial Vision Position IDs matching transformers.vision_utils.get_vision_position_ids
 */

/**
 * Cubic Keys convolution filter (a = -0.5, matching PIL / torchvision bicubic).
 */
function cubicKeys(x) {
  const ax = Math.abs(x);
  if (ax < 1.0) return 1.5 * ax * ax * ax - 2.5 * ax * ax + 1.0;
  if (ax < 2.0) return -0.5 * ax * ax * ax + 2.5 * ax * ax - 4.0 * ax + 2.0;
  return 0.0;
}

function computeResampleWeights(inSize, outSize) {
  const scale = inSize / outSize;
  const support = scale > 1.0 ? 2.0 * scale : 2.0; // antialias when downsampling
  const kScale = scale > 1.0 ? 1.0 / scale : 1.0;
  const rows = new Array(outSize);

  for (let i = 0; i < outSize; i++) {
    const center = (i + 0.5) * scale - 0.5;
    const minIdx = Math.max(0, Math.floor(center - support + 1));
    const maxIdx = Math.min(inSize - 1, Math.ceil(center + support - 1));
    const indices = [];
    const weights = [];
    let wSum = 0.0;
    for (let j = minIdx; j <= maxIdx; j++) {
      const w = cubicKeys((j - center) * kScale);
      if (w !== 0.0) {
        indices.push(j);
        weights.push(w);
        wSum += w;
      }
    }
    if (wSum === 0.0) {
      const clampIdx = Math.max(0, Math.min(inSize - 1, Math.round(center)));
      indices.push(clampIdx);
      weights.push(1.0);
    } else {
      for (let k = 0; k < weights.length; k++) weights[k] /= wSum;
    }
    rows[i] = { indices, weights };
  }
  return rows;
}

/**
 * Separable float32 bicubic resize on RGB or RGBA uint8 data -> float32 RGB in [0, 255].
 */
export function resizeRgbBicubic(pixels, srcW, srcH, dstW, dstH, channels = 3) {
  const out = new Float32Array(dstW * dstH * 3);
  if (srcW === dstW && srcH === dstH) {
    for (let i = 0, n = dstW * dstH; i < n; i++) {
      out[i * 3] = pixels[i * channels];
      out[i * 3 + 1] = pixels[i * channels + 1];
      out[i * 3 + 2] = pixels[i * channels + 2];
    }
    return out;
  }

  const xWeights = computeResampleWeights(srcW, dstW);
  const yWeights = computeResampleWeights(srcH, dstH);
  const horiz = new Float32Array(srcH * dstW * 3);

  for (let y = 0; y < srcH; y++) {
    const srcRow = y * srcW * channels;
    const dstRow = y * dstW * 3;
    for (let x = 0; x < dstW; x++) {
      const { indices, weights } = xWeights[x];
      let r = 0, g = 0, b = 0;
      for (let k = 0; k < indices.length; k++) {
        const sOff = srcRow + indices[k] * channels;
        const w = weights[k];
        r += pixels[sOff] * w;
        g += pixels[sOff + 1] * w;
        b += pixels[sOff + 2] * w;
      }
      const dOff = dstRow + x * 3;
      horiz[dOff] = r;
      horiz[dOff + 1] = g;
      horiz[dOff + 2] = b;
    }
  }

  for (let y = 0; y < dstH; y++) {
    const { indices, weights } = yWeights[y];
    const dstRow = y * dstW * 3;
    for (let x = 0; x < dstW; x++) {
      let r = 0, g = 0, b = 0;
      for (let k = 0; k < indices.length; k++) {
        const hOff = (indices[k] * dstW + x) * 3;
        const w = weights[k];
        r += horiz[hOff] * w;
        g += horiz[hOff + 1] * w;
        b += horiz[hOff + 2] * w;
      }
      const dOff = dstRow + x * 3;
      out[dOff] = Math.max(0.0, Math.min(255.0, r));
      out[dOff + 1] = Math.max(0.0, Math.min(255.0, g));
      out[dOff + 2] = Math.max(0.0, Math.min(255.0, b));
    }
  }
  return out;
}

export function smartResizeLfm3(height, width, {
  factor = 32,
  minPixels = 65536,
  maxPixels = 2097152,
} = {}) {
  if (Math.max(height, width) / Math.min(height, width) > 200) {
    throw new Error(`Image aspect ratio exceeds 200: ${height}x${width}`);
  }
  let hBar = Math.round(height / factor) * factor;
  let wBar = Math.round(width / factor) * factor;
  if (hBar * wBar > maxPixels) {
    const beta = Math.sqrt((height * width) / maxPixels);
    hBar = Math.max(factor, Math.floor(height / beta / factor) * factor);
    wBar = Math.max(factor, Math.floor(width / beta / factor) * factor);
  } else if (hBar * wBar < minPixels) {
    const beta = Math.sqrt(minPixels / (height * width));
    hBar = Math.ceil((height * beta) / factor) * factor;
    wBar = Math.ceil((width * beta) / factor) * factor;
  }
  return { height: hBar, width: wBar };
}

/**
 * Exact port of Lfm3VlImageProcessor:
 * Outputs flattened merge-block ordered patches `[numPatches, 3 * T_patch * P * P]`.
 */
export function preprocessLfm3Image(image, {
  patchSize = 16,
  temporalPatchSize = 2,
  mergeSize = 2,
  minPixels = 65536,
  maxPixels = 2097152,
  mean = [0.5, 0.5, 0.5],
  std = [0.5, 0.5, 0.5],
} = {}) {
  const { data, width, height, channels = 3 } = image;
  const factor = patchSize * mergeSize; // 32
  const resized = smartResizeLfm3(height, width, { factor, minPixels, maxPixels });
  const rgb255 = resizeRgbBicubic(data, width, height, resized.width, resized.height, channels);

  // Normalize (rgb / 255 - mean) / std into CHW float32
  const W = resized.width;
  const H = resized.height;
  const hw = H * W;
  const chw = new Float32Array(3 * hw);
  const rescale = 1.0 / 255.0;
  for (let i = 0; i < hw; i++) {
    const s = i * 3;
    chw[i] = (rgb255[s] * rescale - mean[0]) / std[0];
    chw[hw + i] = (rgb255[s + 1] * rescale - mean[1]) / std[1];
    chw[2 * hw + i] = (rgb255[s + 2] * rescale - mean[2]) / std[2];
  }

  const gridH = H / patchSize;
  const gridW = W / patchSize;
  const numPatches = gridH * gridW;
  const p2 = patchSize * patchSize;
  const patchDim = 3 * temporalPatchSize * p2; // 1536
  const pixelValues = new Float32Array(numPatches * patchDim);

  const blocksW = gridW / mergeSize;
  for (let by = 0; by < gridH / mergeSize; by++) {
    for (let bx = 0; bx < blocksW; bx++) {
      for (let my = 0; my < mergeSize; my++) {
        for (let mx = 0; mx < mergeSize; mx++) {
          const patchIdx = ((by * blocksW + bx) * mergeSize + my) * mergeSize + mx;
          const outBase = patchIdx * patchDim;
          const pyStart = (by * mergeSize + my) * patchSize;
          const pxStart = (bx * mergeSize + mx) * patchSize;
          for (let c = 0; c < 3; c++) {
            const cOff = c * hw;
            const outCOff = outBase + c * temporalPatchSize * p2;
            for (let py = 0; py < patchSize; py++) {
              const rowOff = cOff + (pyStart + py) * W + pxStart;
              for (let px = 0; px < patchSize; px++) {
                const v = chw[rowOff + px];
                const cell = py * patchSize + px;
                for (let tp = 0; tp < temporalPatchSize; tp++) {
                  pixelValues[outCOff + tp * p2 + cell] = v;
                }
              }
            }
          }
        }
      }
    }
  }

  return {
    pixelValues,
    gridThw: [1, gridH, gridW],
    resizedHeight: H,
    resizedWidth: W,
    numVisionTokens: numPatches / (mergeSize * mergeSize),
  };
}

/**
 * Exact port of transformers.vision_utils.get_vision_position_ids (include_temporal=False).
 */
export function computeVisionPositionIds(gridH, gridW, mergeSize = 2) {
  const numPatches = gridH * gridW;
  const pos = new Int32Array(numPatches * 2);
  const bw = gridW / mergeSize;
  let idx = 0;
  for (let by = 0; by < gridH / mergeSize; by++) {
    for (let bx = 0; bx < bw; bx++) {
      for (let my = 0; my < mergeSize; my++) {
        const y = by * mergeSize + my;
        for (let mx = 0; mx < mergeSize; mx++) {
          const x = bx * mergeSize + mx;
          pos[idx * 2] = y;
          pos[idx * 2 + 1] = x;
          idx++;
        }
      }
    }
  }
  return pos;
}

export function smartResizeGemma4(height, width, {
  patchSize = 16,
  maxSoftTokens = 280,
  poolingKernelSize = 3,
} = {}) {
  const maxPatches = maxSoftTokens * poolingKernelSize * poolingKernelSize;
  const totalPx = height * width;
  const targetPx = maxPatches * (patchSize * patchSize);
  const factor = Math.sqrt(targetPx / totalPx);
  const sideMult = poolingKernelSize * patchSize;

  let targetHeight = Math.floor((factor * height) / sideMult) * sideMult;
  let targetWidth = Math.floor((factor * width) / sideMult) * sideMult;
  if (targetHeight === 0 && targetWidth === 0) {
    throw new Error(`Cannot resize ${height}x${width} for Gemma 4 sideMult=${sideMult}`);
  }
  const maxSideLength = Math.floor(maxPatches / (poolingKernelSize * poolingKernelSize)) * sideMult;
  if (targetHeight === 0) {
    targetHeight = sideMult;
    targetWidth = Math.min(Math.floor(width / height) * sideMult, maxSideLength);
  } else if (targetWidth === 0) {
    targetWidth = sideMult;
    targetHeight = Math.min(Math.floor(height / width) * sideMult, maxSideLength);
  }
  return { height: targetHeight, width: targetWidth };
}

/**
 * Exact port of Gemma4ImageProcessor + Gemma4VisionPatchEmbedder pixel transform `2 * (x/255 - 0.5)`:
 * Outputs unpadded real patches `[H_p * W_p, 3 * patchSize^2]` in `(py, px, c)` layout and `positionIds` `[x, y]`.
 */
export function preprocessGemma4Image(image, {
  patchSize = 16,
  maxSoftTokens = 280,
  poolingKernelSize = 3,
} = {}) {
  const { data, width, height, channels = 3 } = image;
  const resized = smartResizeGemma4(height, width, { patchSize, maxSoftTokens, poolingKernelSize });
  const rgb255 = resizeRgbBicubic(data, width, height, resized.width, resized.height, channels);

  const W = resized.width;
  const H = resized.height;
  const gridH = H / patchSize;
  const gridW = W / patchSize;
  const numPatches = gridH * gridW;
  const patchDim = 3 * patchSize * patchSize;
  const pixelValues = new Float32Array(numPatches * patchDim);
  const positionIds = new Int32Array(numPatches * 2);
  const rescale = 1.0 / 255.0;

  for (let by = 0; by < gridH; by++) {
    for (let bx = 0; bx < gridW; bx++) {
      const pIdx = by * gridW + bx;
      positionIds[pIdx * 2] = bx;
      positionIds[pIdx * 2 + 1] = by;
      let dst = pIdx * patchDim;
      for (let py = 0; py < patchSize; py++) {
        const y = by * patchSize + py;
        for (let px = 0; px < patchSize; px++) {
          const x = bx * patchSize + px;
          const src = (y * W + x) * 3;
          pixelValues[dst++] = 2.0 * (rgb255[src] * rescale - 0.5);
          pixelValues[dst++] = 2.0 * (rgb255[src + 1] * rescale - 0.5);
          pixelValues[dst++] = 2.0 * (rgb255[src + 2] * rescale - 0.5);
        }
      }
    }
  }

  return {
    pixelValues,
    positionIds,
    patchH: gridH,
    patchW: gridW,
    numPatches,
    numVisionTokens: numPatches / (poolingKernelSize * poolingKernelSize),
    resizedHeight: H,
    resizedWidth: W,
  };
}
