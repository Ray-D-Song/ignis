"""Ignis offline packer CLI: converts HuggingFace Safetensors to .ignis v1 shards."""

from __future__ import annotations
import argparse
import hashlib
import json
import os
import shutil
from pathlib import Path
from typing import Any

import numpy as np
from safetensors import safe_open

from blueprints import build_blueprint
from pack_multimodal import pack_multimodal_shards
from quant import (
    PAYLOAD_OFFSET,
    TENSOR_ALIGN,
    align_up,
    dequantize_int3_g64_asym,
    dequantize_int4_g64_asym,
    dequantize_int8_g64_sym,
    make_shard_header,
    quantize_int3_g64_asym,
    quantize_int4_g64_asym,
    quantize_int8_g64_sym,
    stream_write_int8_row_shard,
)


class ShardWriter:
    """Packs 256-byte aligned GPU tensor blobs into bounded .ignis shards."""

    def __init__(self, out_dir: Path, max_shard_bytes: int = 512 * 1024 * 1024) -> None:
        self.out_dir = out_dir
        self.max_shard_bytes = max_shard_bytes
        self.current_idx = 0
        self.current_buf = bytearray()
        self.shards_meta: list[dict[str, Any]] = []

    def _flush_current(self) -> None:
        if not self.current_buf:
            return
        payload_bytes = len(self.current_buf)
        header = make_shard_header(self.current_idx, payload_bytes)
        filename = f"shard-{self.current_idx:03d}.ignis"
        out_path = self.out_dir / filename
        data = header + self.current_buf
        out_path.write_bytes(data)
        sha = hashlib.sha256(data).hexdigest()
        self.shards_meta.append({
            "id": self.current_idx,
            "file": filename,
            "bytes": len(data),
            "payload_bytes": payload_bytes,
            "sha256": sha,
        })
        self.current_idx += 1
        self.current_buf = bytearray()

    def add_tensor(self, raw: bytes) -> tuple[int, int, int]:
        """Returns (shard_id, absolute_file_offset, byte_length)."""
        aligned_len = align_up(len(raw), TENSOR_ALIGN)
        if self.current_buf and (PAYLOAD_OFFSET + len(self.current_buf) + aligned_len > self.max_shard_bytes):
            self._flush_current()
        rel_off = len(self.current_buf)
        abs_off = PAYLOAD_OFFSET + rel_off
        self.current_buf.extend(raw)
        pad = aligned_len - len(raw)
        if pad:
            self.current_buf.extend(b"\x00" * pad)
        return self.current_idx, abs_off, len(raw)

    def finalize(self) -> list[dict[str, Any]]:
        self._flush_current()
        return self.shards_meta


def load_hf_tensor_reader(model_dir: Path):
    """Returns a lazy tensor accessor supporting BF16/FP16/FP32 safetensors via PyTorch."""
    index_file = model_dir / "model.safetensors.index.json"
    if index_file.exists():
        index = json.loads(index_file.read_text(encoding="utf-8"))
        weight_map = index["weight_map"]
        files = sorted(set(weight_map.values()))
    else:
        files = sorted(p.name for p in model_dir.glob("*.safetensors"))
        weight_map = None
    if not files:
        raise FileNotFoundError(f"No .safetensors files found in {model_dir}")

    handles = {fname: safe_open(str(model_dir / fname), framework="pt", device="cpu") for fname in files}
    if weight_map is None:
        weight_map = {}
        for fname, h in handles.items():
            for k in h.keys():
                weight_map[k] = fname

    class Reader:
        def __contains__(self, key: str) -> bool:
            return key in weight_map

        def get_shape(self, key: str) -> list[int]:
            fname = weight_map[key]
            return list(handles[fname].get_slice(key).get_shape())

        def get_slice(self, key: str):
            fname = weight_map[key]
            return handles[fname].get_slice(key)

        def get(self, key: str) -> np.ndarray:
            fname = weight_map[key]
            return handles[fname].get_tensor(key).float().numpy()

    return Reader()


def _quantize_chunked_2d(tensor_slice, shape: list[int], fn, chunk_rows: int = 16384):
    """Quantize large 2D tensors chunk-by-chunk to avoid full float32 RAM allocation."""
    m, k = shape
    if m <= chunk_rows:
        arr = tensor_slice[:]
        np_arr = arr.float().numpy() if hasattr(arr, "float") else np.asarray(arr, dtype=np.float32)
        return fn(np_arr)
    q_parts, s_parts = [], []
    padded = None
    for s in range(0, m, chunk_rows):
        e = min(m, s + chunk_rows)
        arr = tensor_slice[s:e]
        np_arr = arr.float().numpy() if hasattr(arr, "float") else np.asarray(arr, dtype=np.float32)
        q_c, s_c, ps = fn(np_arr)
        q_parts.append(q_c)
        s_parts.append(s_c)
        padded = (m, ps[1])
    return np.ascontiguousarray(np.concatenate(q_parts, axis=0)), np.ascontiguousarray(np.concatenate(s_parts, axis=0)), padded


def pack_model(
    model_dir: Path,
    out_dir: Path,
    max_shard_mb: int = 512,
    verify: bool = True,
    lm_head_dtype: str = "int4",
    weight_bits: int = 4,
    pack_multimodal: bool = True,
) -> dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    cfg = json.loads((model_dir / "config.json").read_text(encoding="utf-8"))
    blueprint = build_blueprint(cfg)
    blueprint["weight_bits"] = weight_bits
    hf_tensors = load_hf_tensor_reader(model_dir)

    writer = ShardWriter(out_dir, max_shard_bytes=max_shard_mb * 1024 * 1024)
    tensors_meta: dict[str, Any] = {}
    quant_errors: list[tuple[str, float]] = []

    def register_f32(ignis_name: str, arr: np.ndarray) -> None:
        f32 = np.ascontiguousarray(arr.astype(np.float32))
        shard_id, off, nbytes = writer.add_tensor(f32.tobytes())
        tensors_meta[ignis_name] = {
            "shard": shard_id,
            "offset": off,
            "bytes": nbytes,
            "shape": list(f32.shape),
            "dtype": "f32",
            "layout": "row_major",
        }

    def _cosine_verify(ignis_name: str, q_u32: np.ndarray, s_u32: np.ndarray, deq_fn, src_slice_or_arr, orig_shape: list[int]) -> None:
        if not verify:
            return
        rows = min(orig_shape[0], 4096)
        raw_sub = src_slice_or_arr[:rows] if not isinstance(src_slice_or_arr, np.ndarray) else src_slice_or_arr[:rows]
        orig_f32 = raw_sub.float().numpy() if hasattr(raw_sub, "float") else np.asarray(raw_sub, dtype=np.float32)
        deq = deq_fn(q_u32[:rows], s_u32[:rows], orig_shape[1])
        cos = float(np.sum(orig_f32 * deq) / (np.linalg.norm(orig_f32) * np.linalg.norm(deq) + 1e-12))
        quant_errors.append((ignis_name, cos))

    def register_q3(ignis_name: str, src, shape: list[int] | None = None) -> None:
        if isinstance(src, np.ndarray):
            quants_u32, scales_u32, padded_shape = quantize_int3_g64_asym(src)
            orig_shape = list(src.shape)
        else:
            orig_shape = shape
            quants_u32, scales_u32, padded_shape = _quantize_chunked_2d(src, orig_shape, quantize_int3_g64_asym)
        q_shard, q_off, q_bytes = writer.add_tensor(quants_u32.tobytes())
        s_shard, s_off, s_bytes = writer.add_tensor(scales_u32.tobytes())
        tensors_meta[f"{ignis_name}.quants"] = {
            "shard": q_shard,
            "offset": q_off,
            "bytes": q_bytes,
            "shape": list(quants_u32.shape),
            "orig_shape": orig_shape,
            "padded_shape": list(padded_shape),
            "dtype": "u32_q3",
            "layout": "row_major",
        }
        tensors_meta[f"{ignis_name}.scales"] = {
            "shard": s_shard,
            "offset": s_off,
            "bytes": s_bytes,
            "shape": list(scales_u32.shape),
            "dtype": "u32_f16x2",
            "group_size": 64,
            "layout": "row_major",
        }
        _cosine_verify(ignis_name, quants_u32, scales_u32, dequantize_int3_g64_asym, src, orig_shape)

    def register_q4(ignis_name: str, src, shape: list[int] | None = None) -> None:
        if isinstance(src, np.ndarray):
            quants_u32, scales_u32, padded_shape = quantize_int4_g64_asym(src)
            orig_shape = list(src.shape)
        else:
            orig_shape = shape
            quants_u32, scales_u32, padded_shape = _quantize_chunked_2d(src, orig_shape, quantize_int4_g64_asym)
        q_shard, q_off, q_bytes = writer.add_tensor(quants_u32.tobytes())
        s_shard, s_off, s_bytes = writer.add_tensor(scales_u32.tobytes())
        tensors_meta[f"{ignis_name}.quants"] = {
            "shard": q_shard,
            "offset": q_off,
            "bytes": q_bytes,
            "shape": list(quants_u32.shape),
            "orig_shape": orig_shape,
            "padded_shape": list(padded_shape),
            "dtype": "u32_q4",
            "layout": "row_major",
        }
        tensors_meta[f"{ignis_name}.scales"] = {
            "shard": s_shard,
            "offset": s_off,
            "bytes": s_bytes,
            "shape": list(scales_u32.shape),
            "dtype": "u32_f16x2",
            "group_size": 64,
            "layout": "row_major",
        }
        _cosine_verify(ignis_name, quants_u32, scales_u32, dequantize_int4_g64_asym, src, orig_shape)

    def register_q8(ignis_name: str, src, shape: list[int] | None = None) -> None:
        if isinstance(src, np.ndarray):
            quants_u32, scales_u32, padded_shape = quantize_int8_g64_sym(src)
            orig_shape = list(src.shape)
        else:
            orig_shape = shape
            quants_u32, scales_u32, padded_shape = _quantize_chunked_2d(src, orig_shape, quantize_int8_g64_sym)
        q_shard, q_off, q_bytes = writer.add_tensor(quants_u32.tobytes())
        s_shard, s_off, s_bytes = writer.add_tensor(scales_u32.tobytes())
        tensors_meta[f"{ignis_name}.quants"] = {
            "shard": q_shard,
            "offset": q_off,
            "bytes": q_bytes,
            "shape": list(quants_u32.shape),
            "orig_shape": orig_shape,
            "padded_shape": list(padded_shape),
            "dtype": "u32_q8",
            "layout": "row_major",
        }
        tensors_meta[f"{ignis_name}.scales"] = {
            "shard": s_shard,
            "offset": s_off,
            "bytes": s_bytes,
            "shape": list(scales_u32.shape),
            "dtype": "u32_f16x2",
            "group_size": 64,
            "layout": "row_major",
        }
        _cosine_verify(ignis_name, quants_u32, scales_u32, dequantize_int8_g64_sym, src, orig_shape)

    register_body = register_q3 if weight_bits == 3 else register_q4

    # Detect HF weight key prefix & family
    arch = blueprint["architecture"]
    if "model.language_model.embed_tokens.weight" in hf_tensors:
        tp = "model.language_model."
        embd_key = f"{tp}embed_tokens.weight"
    elif "model.embedding.weight" in hf_tensors:
        tp = "model."
        embd_key = "model.embedding.weight"
    else:
        tp = "model."
        embd_key = "model.embed_tokens.weight"

    # 1. Externalized Embedding -> embd.ignis (streamed cleanly with low RAM)
    embd_shape = hf_tensors.get_shape(embd_key)
    embd_path = out_dir / "embd.ignis"
    row_stride, embd_payload_bytes, embd_sha, cos_e = stream_write_int8_row_shard(
        hf_tensors.get_slice(embd_key), embd_shape[0], embd_shape[1], embd_path, shard_id=0xFFFFFFFF, verify=verify
    )
    if verify:
        quant_errors.append(("token_embd", cos_e))

    tensors_meta["token_embd"] = {
        "shard": "embd",
        "offset": PAYLOAD_OFFSET,
        "bytes": embd_payload_bytes,
        "shape": embd_shape,
        "row_stride": row_stride,
        "dtype": "i8_row",
    }
    extra_opfs_shards = [
        {
            "id": "embd",
            "file": "embd.ignis",
            "bytes": PAYLOAD_OFFSET + embd_payload_bytes,
            "payload_bytes": embd_payload_bytes,
            "sha256": embd_sha,
        }
    ]

    # Optional Gemma 4 Per-Layer Embeddings (PLE) -> ple_embd.ignis (0 MiB VRAM!)
    if blueprint["has_ple"]:
        ple_key = f"{tp}embed_tokens_per_layer.weight"
        ple_shape = hf_tensors.get_shape(ple_key)
        ple_path = out_dir / "ple_embd.ignis"
        ple_stride, ple_payload_bytes, ple_sha, cos_ple = stream_write_int8_row_shard(
            hf_tensors.get_slice(ple_key), ple_shape[0], ple_shape[1], ple_path, shard_id=0xFFFFFFFE, verify=verify
        )
        if verify:
            quant_errors.append(("token_embd_ple", cos_ple))
        tensors_meta["token_embd_ple"] = {
            "shard": "ple_embd",
            "offset": PAYLOAD_OFFSET,
            "bytes": ple_payload_bytes,
            "shape": ple_shape,
            "row_stride": ple_stride,
            "dtype": "i8_row",
        }
        extra_opfs_shards.append({
            "id": "ple_embd",
            "file": "ple_embd.ignis",
            "bytes": PAYLOAD_OFFSET + ple_payload_bytes,
            "payload_bytes": ple_payload_bytes,
            "sha256": ple_sha,
        })
        register_body("ple_model_proj", hf_tensors.get(f"{tp}per_layer_model_projection.weight"))
        register_f32("ple_proj_norm", hf_tensors.get(f"{tp}per_layer_projection_norm.weight"))

    # 2. LM Head (either explicit or tied to embedding)
    lm_head_key = embd_key if blueprint["tie_word_embeddings"] or "lm_head.weight" not in hf_tensors else "lm_head.weight"
    lm_head_shape = hf_tensors.get_shape(lm_head_key)
    lm_head_slice = hf_tensors.get_slice(lm_head_key)
    if lm_head_dtype == "int8":
        register_q8("output", lm_head_slice, lm_head_shape)
    elif lm_head_dtype == "int4":
        register_q4("output", lm_head_slice, lm_head_shape)
    elif lm_head_dtype == "int3":
        register_q3("output", lm_head_slice, lm_head_shape)
    else:
        raise ValueError(f"Unsupported lm_head_dtype: {lm_head_dtype}")

    final_norm_key = f"{tp}embedding_norm.weight" if f"{tp}embedding_norm.weight" in hf_tensors else f"{tp}norm.weight"
    register_f32("output_norm", hf_tensors.get(final_norm_key))

    # 3. Transformer / Hybrid Layers
    is_lfm = arch in ("lfm3", "lfm2") or blueprint["top_model_type"] == "lfm3_omni"
    is_gemma4 = arch in ("gemma4", "gemma4_text") or blueprint["top_model_type"] == "gemma4"

    for i in range(blueprint["num_layers"]):
        pfx = f"{tp}layers.{i}"
        opfx = f"blk.{i}"
        lt = blueprint["layer_types"][i]

        if is_lfm:
            register_f32(f"{opfx}.attn_norm", hf_tensors.get(f"{pfx}.operator_norm.weight"))
            register_f32(f"{opfx}.ffn_norm", hf_tensors.get(f"{pfx}.ffn_norm.weight"))
            if blueprint["use_sandwich_norm"]:
                register_f32(f"{opfx}.post_attn_norm", hf_tensors.get(f"{pfx}.post_operator_norm.weight"))
                register_f32(f"{opfx}.post_ffn_norm", hf_tensors.get(f"{pfx}.post_ffn_norm.weight"))

            if lt == "conv":
                register_body(f"{opfx}.conv_in_proj", hf_tensors.get(f"{pfx}.conv.in_proj.weight"))
                conv_w = hf_tensors.get(f"{pfx}.conv.conv.weight")
                register_f32(f"{opfx}.conv_kernel", conv_w.reshape(conv_w.shape[0], -1))
                register_body(f"{opfx}.conv_out_proj", hf_tensors.get(f"{pfx}.conv.out_proj.weight"))
            else:
                qkv_w = np.concatenate([
                    hf_tensors.get(f"{pfx}.self_attn.q_proj.weight"),
                    hf_tensors.get(f"{pfx}.self_attn.k_proj.weight"),
                    hf_tensors.get(f"{pfx}.self_attn.v_proj.weight"),
                ], axis=0)
                register_body(f"{opfx}.attn_qkv", qkv_w)
                register_body(f"{opfx}.attn_output", hf_tensors.get(f"{pfx}.self_attn.o_proj.weight"))
                register_f32(f"{opfx}.q_norm", hf_tensors.get(f"{pfx}.self_attn.q_norm.weight"))
                register_f32(f"{opfx}.k_norm", hf_tensors.get(f"{pfx}.self_attn.k_norm.weight"))
                register_f32(f"{opfx}.attn_gate", hf_tensors.get(f"{pfx}.self_attn.gate_proj.weight"))

            register_body(f"{opfx}.ffn_gate", hf_tensors.get(f"{pfx}.feed_forward.w1.weight"))
            register_body(f"{opfx}.ffn_up", hf_tensors.get(f"{pfx}.feed_forward.w3.weight"))
            register_body(f"{opfx}.ffn_down", hf_tensors.get(f"{pfx}.feed_forward.w2.weight"))

        elif is_gemma4:
            if f"{pfx}.layer_scalar" in hf_tensors:
                blueprint["layer_scalars"][i] = float(hf_tensors.get(f"{pfx}.layer_scalar").reshape(-1)[0])
            register_f32(f"{opfx}.attn_norm", hf_tensors.get(f"{pfx}.input_layernorm.weight"))
            register_f32(f"{opfx}.post_attn_norm", hf_tensors.get(f"{pfx}.post_attention_layernorm.weight"))
            register_f32(f"{opfx}.ffn_norm", hf_tensors.get(f"{pfx}.pre_feedforward_layernorm.weight"))
            register_f32(f"{opfx}.post_ffn_norm", hf_tensors.get(f"{pfx}.post_feedforward_layernorm.weight"))

            is_reuse = blueprint["kv_source_layers"][i] != i
            q_w = hf_tensors.get(f"{pfx}.self_attn.q_proj.weight")
            register_f32(f"{opfx}.q_norm", hf_tensors.get(f"{pfx}.self_attn.q_norm.weight"))
            if is_reuse:
                register_body(f"{opfx}.attn_qkv", q_w)
            else:
                k_w = hf_tensors.get(f"{pfx}.self_attn.k_proj.weight")
                v_w = hf_tensors.get(f"{pfx}.self_attn.v_proj.weight") if f"{pfx}.self_attn.v_proj.weight" in hf_tensors else k_w
                register_body(f"{opfx}.attn_qkv", np.concatenate([q_w, k_w, v_w], axis=0))
                register_f32(f"{opfx}.k_norm", hf_tensors.get(f"{pfx}.self_attn.k_norm.weight"))
            register_body(f"{opfx}.attn_output", hf_tensors.get(f"{pfx}.self_attn.o_proj.weight"))

            register_body(f"{opfx}.ffn_gate", hf_tensors.get(f"{pfx}.mlp.gate_proj.weight"))
            register_body(f"{opfx}.ffn_up", hf_tensors.get(f"{pfx}.mlp.up_proj.weight"))
            register_body(f"{opfx}.ffn_down", hf_tensors.get(f"{pfx}.mlp.down_proj.weight"))

            if blueprint["has_ple"]:
                register_body(f"{opfx}.ple_gate", hf_tensors.get(f"{pfx}.per_layer_input_gate.weight"))
                register_body(f"{opfx}.ple_proj", hf_tensors.get(f"{pfx}.per_layer_projection.weight"))
                register_f32(f"{opfx}.post_ple_norm", hf_tensors.get(f"{pfx}.post_per_layer_input_norm.weight"))

        else:
            # Spark2_5 / Llama / Qwen2 / Qwen3 / Phi3
            register_f32(f"{opfx}.attn_norm", hf_tensors.get(f"{pfx}.input_layernorm.weight"))
            register_f32(f"{opfx}.ffn_norm", hf_tensors.get(f"{pfx}.post_attention_layernorm.weight"))

            if f"{pfx}.self_attn.q_k_v_proj.weight" in hf_tensors:
                qkv_w = hf_tensors.get(f"{pfx}.self_attn.q_k_v_proj.weight")
            elif f"{pfx}.self_attn.qkv_proj.weight" in hf_tensors:
                qkv_w = hf_tensors.get(f"{pfx}.self_attn.qkv_proj.weight")
            else:
                qkv_w = np.concatenate([
                    hf_tensors.get(f"{pfx}.self_attn.q_proj.weight"),
                    hf_tensors.get(f"{pfx}.self_attn.k_proj.weight"),
                    hf_tensors.get(f"{pfx}.self_attn.v_proj.weight"),
                ], axis=0)
            register_body(f"{opfx}.attn_qkv", qkv_w)

            if f"{pfx}.self_attn.q_proj.bias" in hf_tensors:
                blueprint["has_qkv_bias"] = True
                qkv_b = np.concatenate([
                    hf_tensors.get(f"{pfx}.self_attn.q_proj.bias"),
                    hf_tensors.get(f"{pfx}.self_attn.k_proj.bias"),
                    hf_tensors.get(f"{pfx}.self_attn.v_proj.bias"),
                ], axis=0)
                register_f32(f"{opfx}.attn_qkv_bias", qkv_b)

            if blueprint["has_qk_norm"] or f"{pfx}.self_attn.q_norm.weight" in hf_tensors:
                blueprint["has_qk_norm"] = True
                register_f32(f"{opfx}.q_norm", hf_tensors.get(f"{pfx}.self_attn.q_norm.weight"))
                register_f32(f"{opfx}.k_norm", hf_tensors.get(f"{pfx}.self_attn.k_norm.weight"))

            o_key = f"{pfx}.self_attn.out_proj.weight" if f"{pfx}.self_attn.out_proj.weight" in hf_tensors else f"{pfx}.self_attn.o_proj.weight"
            register_body(f"{opfx}.attn_output", hf_tensors.get(o_key))

            if blueprint["has_attn_gate"]:
                register_f32(f"{opfx}.attn_gate", hf_tensors.get(f"{pfx}.self_attn.g_proj.weight"))

            if f"{pfx}.mlp.gate_up_proj.weight" in hf_tensors:
                gu = hf_tensors.get(f"{pfx}.mlp.gate_up_proj.weight")
                half = gu.shape[0] // 2
                register_body(f"{opfx}.ffn_gate", gu[:half])
                register_body(f"{opfx}.ffn_up", gu[half:])
            else:
                register_body(f"{opfx}.ffn_gate", hf_tensors.get(f"{pfx}.mlp.gate_proj.weight"))
                register_body(f"{opfx}.ffn_up", hf_tensors.get(f"{pfx}.mlp.up_proj.weight"))
            register_body(f"{opfx}.ffn_down", hf_tensors.get(f"{pfx}.mlp.down_proj.weight"))

    gpu_shards = writer.finalize()
    mm_shards, mm_tensors = ([], {})
    if pack_multimodal and blueprint.get("multimodal"):
        mm_shards, mm_tensors = pack_multimodal_shards(
            model_dir, out_dir, cfg, blueprint, hf_tensors, quant_errors, verify=verify
        )
    shards_meta = [*extra_opfs_shards, *gpu_shards, *mm_shards]

    # Copy tokenizer.json if present
    tok_src = model_dir / "tokenizer.json"
    if tok_src.exists():
        shutil.copy2(tok_src, out_dir / "tokenizer.json")

    manifest = {
        "format": "ignis-v1",
        "model_id": model_dir.name,
        "blueprint": blueprint,
        "shards": shards_meta,
        "tensors": tensors_meta,
        **({"multimodal_tensors": mm_tensors} if mm_tensors else {}),
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    if verify and quant_errors:
        min_name, min_cos = min(quant_errors, key=lambda x: x[1])
        avg_cos = sum(x[1] for x in quant_errors) / len(quant_errors)
        print(f"[packer verify] checked={len(quant_errors)} tensors | avg_cos={avg_cos:.6f} | min_cos={min_cos:.6f} ({min_name})")

    total_gpu_bytes = sum(s["bytes"] for s in gpu_shards)
    print(f"[packer] wrote {len(gpu_shards)} GPU shard(s) ({total_gpu_bytes / (1024**2):.2f} MiB) + embd.ignis ({(PAYLOAD_OFFSET + embd_payload_bytes) / (1024**2):.2f} MiB) -> {out_dir}")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="Ignis offline model packer (.safetensors -> .ignis v1)")
    parser.add_argument("--input", type=Path, required=True, help="HF model directory containing config.json & *.safetensors")
    parser.add_argument("--output", type=Path, required=True, help="Output directory for .ignis package")
    parser.add_argument("--max-shard-mb", type=int, default=512, help="Max GPU shard size in MiB")
    parser.add_argument("--lm-head-dtype", choices=["int4", "int8", "int3"], default="int4", help="Quantization dtype for LM head")
    parser.add_argument("--weight-bits", type=int, choices=[4, 3], default=4, help="Quantization bit-width for transformer body weights")
    parser.add_argument("--no-multimodal", action="store_true", help="Skip packing vision/audio components")
    parser.add_argument("--no-verify", action="store_true", help="Skip dequantization cosine verification")
    args = parser.parse_args()
    pack_model(
        args.input,
        args.output,
        max_shard_mb=args.max_shard_mb,
        verify=not args.no_verify,
        lm_head_dtype=args.lm_head_dtype,
        weight_bits=args.weight_bits,
        pack_multimodal=not args.no_multimodal,
    )


if __name__ == "__main__":
    main()
