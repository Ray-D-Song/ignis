"""Ignis quantization & packing primitives for WebGPU-aligned .ignis v1 format."""

from __future__ import annotations
import struct
import numpy as np

IGNIS_MAGIC = b"IGNIS001"
IGNIS_VERSION = 1
PAYLOAD_OFFSET = 256
TENSOR_ALIGN = 256


def align_up(value: int, align: int = TENSOR_ALIGN) -> int:
    return ((value + align - 1) // align) * align


def make_shard_header(shard_index: int, payload_bytes: int) -> bytes:
    """Create the 256-byte shard header (64-byte struct + zero padding to 256B)."""
    hdr = struct.pack(
        "<8sIIQQ32s",
        IGNIS_MAGIC,
        IGNIS_VERSION,
        shard_index & 0xFFFFFFFF,
        PAYLOAD_OFFSET,
        payload_bytes,
        b"\x00" * 32,
    )
    return hdr + b"\x00" * (PAYLOAD_OFFSET - len(hdr))


def quantize_int4_g64_asym(weight: np.ndarray) -> tuple[np.ndarray, np.ndarray, tuple[int, int]]:
    """
    Quantize a 2D float weight [M, K] into INT4_G64_ASYM.
    Returns:
      quants_u32: uint32 array of shape [M, K_pad // 8] (8x 4-bit unsigned ints packed LE)
      scales_u32: uint32 array of shape [M, K_pad // 64] (f16 scale low16, f16 min high16)
      padded_shape: (M, K_pad)
    """
    if weight.ndim != 2:
        raise ValueError(f"Expected 2D weight, got shape {weight.shape}")
    w = np.asarray(weight, dtype=np.float32)
    m, k = w.shape
    k_pad = align_up(k, 64)
    if k_pad != k:
        w = np.pad(w, ((0, 0), (0, k_pad - k)), mode="constant", constant_values=0.0)

    groups = k_pad // 64
    wg = w.reshape(m, groups, 64)

    w_min = wg.min(axis=-1)
    w_max = wg.max(axis=-1)
    span = w_max - w_min

    scale_f16 = np.where(span > 0.0, span / 15.0, 0.0).astype(np.float16)
    min_f16 = w_min.astype(np.float16)

    scale_f32 = scale_f16.astype(np.float32)[..., None]
    min_f32 = min_f16.astype(np.float32)[..., None]

    inv_scale = np.where(scale_f32 > 0.0, 1.0 / scale_f32, 0.0)
    q = np.clip(np.round((wg - min_f32) * inv_scale), 0, 15).astype(np.uint32)

    # Pack 8 x 4-bit values into 1 uint32 along K
    q_8 = q.reshape(m, k_pad // 8, 8)
    shifts = (np.arange(8, dtype=np.uint32) * 4)[None, None, :]
    quants_u32 = np.bitwise_or.reduce(q_8 << shifts, axis=-1).astype(np.uint32)

    # Pack (scale_f16 -> low16, min_f16 -> high16) into uint32 matching WGSL unpack2x16float
    scale_bits = scale_f16.view(np.uint16).astype(np.uint32)
    min_bits = min_f16.view(np.uint16).astype(np.uint32)
    scales_u32 = (scale_bits | (min_bits << 16)).astype(np.uint32)

    return np.ascontiguousarray(quants_u32), np.ascontiguousarray(scales_u32), (m, k_pad)


def dequantize_int4_g64_asym(quants_u32: np.ndarray, scales_u32: np.ndarray, orig_k: int | None = None) -> np.ndarray:
    """Dequantize INT4_G64_ASYM back to float32 for verification."""
    m, k_words = quants_u32.shape
    k_pad = k_words * 8
    groups = k_pad // 64

    shifts = (np.arange(8, dtype=np.uint32) * 4)[None, None, :]
    q = ((quants_u32[:, :, None] >> shifts) & 0xF).reshape(m, groups, 64).astype(np.float32)

    scale_bits = (scales_u32 & 0xFFFF).astype(np.uint16)
    min_bits = ((scales_u32 >> 16) & 0xFFFF).astype(np.uint16)
    scale = scale_bits.view(np.float16).astype(np.float32)[..., None]
    min_val = min_bits.view(np.float16).astype(np.float32)[..., None]

    w = (q * scale + min_val).reshape(m, k_pad)
    if orig_k is not None and orig_k < k_pad:
        w = w[:, :orig_k]
    return w


def quantize_int8_row(weight: np.ndarray) -> tuple[bytes, int]:
    """
    Row-wise symmetric INT8 quantization for CPU/OPFS embedding lookup.
    Row layout (16-byte aligned stride): [f32 scale (4B)] [i8[D]] [zero padding].
    Returns (raw_bytes, row_stride).
    """
    if weight.ndim != 2:
        raise ValueError(f"Expected 2D embedding weight, got shape {weight.shape}")
    w = np.asarray(weight, dtype=np.float32)
    vocab, dim = w.shape
    row_stride = align_up(4 + dim, 16)

    max_abs = np.max(np.abs(w), axis=-1)
    scales = np.where(max_abs > 0.0, max_abs / 127.0, 0.0).astype(np.float32)
    inv_scales = np.where(scales > 0.0, 1.0 / scales, 0.0)[:, None]
    q = np.clip(np.round(w * inv_scales), -127, 127).astype(np.int8)

    buf = np.zeros((vocab, row_stride), dtype=np.uint8)
    buf[:, :4] = scales.view(np.uint8).reshape(vocab, 4)
    buf[:, 4 : 4 + dim] = q.view(np.uint8)
    return buf.tobytes(), row_stride


def dequantize_int8_row(raw: bytes, vocab: int, dim: int, row_stride: int) -> np.ndarray:
    buf = np.frombuffer(raw, dtype=np.uint8).reshape(vocab, row_stride)
    scales = np.ascontiguousarray(buf[:, :4]).view(np.float32)
    q = buf[:, 4 : 4 + dim].view(np.int8).astype(np.float32)
    return q * scales
