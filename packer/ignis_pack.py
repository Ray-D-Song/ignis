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

from quant import (
    PAYLOAD_OFFSET,
    TENSOR_ALIGN,
    align_up,
    dequantize_int4_g64_asym,
    dequantize_int8_row,
    make_shard_header,
    quantize_int4_g64_asym,
    quantize_int8_row,
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


def build_blueprint(cfg: dict[str, Any]) -> dict[str, Any]:
    layer_types_raw = cfg.get("layer_types") or ["full_attention"] * cfg["num_hidden_layers"]
    layer_types = ["swa" if lt == "sliding_attention" else "full" for lt in layer_types_raw]
    rope_params = cfg.get("rope_parameters") or {}
    full_rope = rope_params.get("full_attention", {})
    swa_rope = rope_params.get("sliding_attention", {})
    head_dim = cfg.get("head_dim") or (cfg["hidden_size"] // cfg["num_attention_heads"])
    return {
        "architecture": cfg.get("model_type", "spark2_5"),
        "num_layers": cfg["num_hidden_layers"],
        "hidden_size": cfg["hidden_size"],
        "intermediate_size": cfg["intermediate_size"],
        "vocab_size": cfg["vocab_size"],
        "num_attention_heads": cfg["num_attention_heads"],
        "num_key_value_heads": cfg["num_key_value_heads"],
        "head_dim": head_dim,
        "rms_norm_eps": cfg.get("rms_norm_eps", 1e-6),
        "hidden_act": cfg.get("hidden_act", "gelu"),
        "sliding_window": cfg.get("sliding_window"),
        "rope_theta_full": float(full_rope.get("rope_theta", cfg.get("rope_theta", 5000000.0))),
        "rope_dim_full": int(round(head_dim * float(full_rope.get("partial_rotary_factor", 1.0)))),
        "rope_theta_swa": float(swa_rope.get("rope_theta", 10000.0)),
        "rope_dim_swa": int(round(head_dim * float(swa_rope.get("partial_rotary_factor", 1.0)))),
        "layer_types": layer_types,
        "has_attn_gate": bool(cfg.get("headwise_attn_output_gate", False)),
        "gate_attn_act_mode": cfg.get("gate_attn_act_mode", "sigmoid"),
        "tie_word_embeddings": bool(cfg.get("tie_word_embeddings", False)),
        "bos_token_id": cfg.get("bos_token_id", 0),
        "eos_token_id": cfg.get("eos_token_id", 1),
        "pad_token_id": cfg.get("pad_token_id", 2),
    }


def load_hf_tensors(model_dir: Path) -> dict[str, np.ndarray]:
    index_file = model_dir / "model.safetensors.index.json"
    if index_file.exists():
        index = json.loads(index_file.read_text(encoding="utf-8"))
        files = sorted(set(index["weight_map"].values()))
    else:
        files = sorted(p.name for p in model_dir.glob("*.safetensors"))
    if not files:
        raise FileNotFoundError(f"No .safetensors files found in {model_dir}")

    tensors: dict[str, np.ndarray] = {}
    for fname in files:
        with safe_open(str(model_dir / fname), framework="numpy") as f:
            for key in f.keys():
                tensors[key] = f.get_tensor(key)
    return tensors


def pack_model(
    model_dir: Path,
    out_dir: Path,
    max_shard_mb: int = 512,
    verify: bool = True,
) -> dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    cfg = json.loads((model_dir / "config.json").read_text(encoding="utf-8"))
    blueprint = build_blueprint(cfg)
    hf_tensors = load_hf_tensors(model_dir)

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

    def register_q4(ignis_name: str, arr: np.ndarray) -> None:
        quants_u32, scales_u32, padded_shape = quantize_int4_g64_asym(arr)
        q_shard, q_off, q_bytes = writer.add_tensor(quants_u32.tobytes())
        s_shard, s_off, s_bytes = writer.add_tensor(scales_u32.tobytes())
        tensors_meta[f"{ignis_name}.quants"] = {
            "shard": q_shard,
            "offset": q_off,
            "bytes": q_bytes,
            "shape": list(quants_u32.shape),
            "orig_shape": list(arr.shape),
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
        if verify:
            orig_f32 = arr.astype(np.float32)
            deq = dequantize_int4_g64_asym(quants_u32, scales_u32, arr.shape[1])
            cos = float(np.sum(orig_f32 * deq) / (np.linalg.norm(orig_f32) * np.linalg.norm(deq) + 1e-12))
            quant_errors.append((ignis_name, cos))

    # 1. Externalized Embedding -> embd.ignis
    embd_key = "model.embedding.weight" if "model.embedding.weight" in hf_tensors else "model.embed_tokens.weight"
    embd_w = hf_tensors[embd_key]
    embd_raw, row_stride = quantize_int8_row(embd_w)
    embd_header = make_shard_header(0xFFFFFFFF, len(embd_raw))
    embd_bytes = embd_header + embd_raw
    embd_path = out_dir / "embd.ignis"
    embd_path.write_bytes(embd_bytes)
    embd_sha = hashlib.sha256(embd_bytes).hexdigest()

    if verify:
        deq_embd = dequantize_int8_row(embd_raw, embd_w.shape[0], embd_w.shape[1], row_stride)
        orig_e = embd_w.astype(np.float32)
        cos_e = float(np.sum(orig_e * deq_embd) / (np.linalg.norm(orig_e) * np.linalg.norm(deq_embd) + 1e-12))
        quant_errors.append(("token_embd", cos_e))

    tensors_meta["token_embd"] = {
        "shard": "embd",
        "offset": PAYLOAD_OFFSET,
        "bytes": len(embd_raw),
        "shape": list(embd_w.shape),
        "row_stride": row_stride,
        "dtype": "i8_row",
    }

    # 2. LM Head (either explicit or tied to embedding)
    lm_head_w = embd_w if blueprint["tie_word_embeddings"] or "lm_head.weight" not in hf_tensors else hf_tensors["lm_head.weight"]
    register_q4("output", lm_head_w)
    register_f32("output_norm", hf_tensors["model.norm.weight"])

    # 3. Transformer layers
    for i in range(blueprint["num_layers"]):
        pfx = f"model.layers.{i}"
        opfx = f"blk.{i}"
        register_f32(f"{opfx}.attn_norm", hf_tensors[f"{pfx}.input_layernorm.weight"])
        register_f32(f"{opfx}.ffn_norm", hf_tensors[f"{pfx}.post_attention_layernorm.weight"])

        if f"{pfx}.self_attn.q_k_v_proj.weight" in hf_tensors:
            qkv_w = hf_tensors[f"{pfx}.self_attn.q_k_v_proj.weight"]
        else:
            qkv_w = np.concatenate([
                hf_tensors[f"{pfx}.self_attn.q_proj.weight"],
                hf_tensors[f"{pfx}.self_attn.k_proj.weight"],
                hf_tensors[f"{pfx}.self_attn.v_proj.weight"],
            ], axis=0)
        register_q4(f"{opfx}.attn_qkv", qkv_w)

        o_key = f"{pfx}.self_attn.out_proj.weight" if f"{pfx}.self_attn.out_proj.weight" in hf_tensors else f"{pfx}.self_attn.o_proj.weight"
        register_q4(f"{opfx}.attn_output", hf_tensors[o_key])

        if blueprint["has_attn_gate"]:
            register_f32(f"{opfx}.attn_gate", hf_tensors[f"{pfx}.self_attn.g_proj.weight"])

        register_q4(f"{opfx}.ffn_gate", hf_tensors[f"{pfx}.mlp.gate_proj.weight"])
        register_q4(f"{opfx}.ffn_up", hf_tensors[f"{pfx}.mlp.up_proj.weight"])
        register_q4(f"{opfx}.ffn_down", hf_tensors[f"{pfx}.mlp.down_proj.weight"])

    gpu_shards = writer.finalize()
    shards_meta = [
        {
            "id": "embd",
            "file": "embd.ignis",
            "bytes": len(embd_bytes),
            "payload_bytes": len(embd_raw),
            "sha256": embd_sha,
        },
        *gpu_shards,
    ]

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
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    if verify and quant_errors:
        min_name, min_cos = min(quant_errors, key=lambda x: x[1])
        avg_cos = sum(x[1] for x in quant_errors) / len(quant_errors)
        print(f"[packer verify] checked={len(quant_errors)} tensors | avg_cos={avg_cos:.6f} | min_cos={min_cos:.6f} ({min_name})")

    total_gpu_bytes = sum(s["bytes"] for s in gpu_shards)
    print(f"[packer] wrote {len(gpu_shards)} GPU shard(s) ({total_gpu_bytes / (1024**2):.2f} MiB) + embd.ignis ({len(embd_bytes) / (1024**2):.2f} MiB) -> {out_dir}")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="Ignis offline model packer (.safetensors -> .ignis v1)")
    parser.add_argument("--input", type=Path, required=True, help="HF model directory containing config.json & *.safetensors")
    parser.add_argument("--output", type=Path, required=True, help="Output directory for .ignis package")
    parser.add_argument("--max-shard-mb", type=int, default=512, help="Max GPU shard size in MiB")
    parser.add_argument("--no-verify", action="store_true", help="Skip dequantization cosine verification")
    args = parser.parse_args()
    pack_model(args.input, args.output, max_shard_mb=args.max_shard_mb, verify=not args.no_verify)


if __name__ == "__main__":
    main()
