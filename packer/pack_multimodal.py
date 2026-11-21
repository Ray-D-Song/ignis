"""Ignis multimodal weight packer for LFM3-Omni-2B and Gemma-4-E4B-it."""

from __future__ import annotations
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
from safetensors import safe_open

from quant import (
    PAYLOAD_OFFSET,
    TENSOR_ALIGN,
    align_up,
    dequantize_int8_g64_sym,
    make_shard_header,
    quantize_int8_g64_sym,
    quantize_int8_row,
)


class SingleShardWriter:
    """Packs 256-byte aligned blobs into one named .ignis shard file (e.g. vision.ignis)."""

    def __init__(self, out_dir: Path, shard_name: str, filename: str, shard_id_u32: int) -> None:
        self.out_dir = out_dir
        self.shard_name = shard_name
        self.filename = filename
        self.shard_id_u32 = shard_id_u32
        self.buf = bytearray()

    def add(self, raw: bytes) -> tuple[int, int]:
        aligned_len = align_up(len(raw), TENSOR_ALIGN)
        abs_off = PAYLOAD_OFFSET + len(self.buf)
        self.buf.extend(raw)
        pad = aligned_len - len(raw)
        if pad:
            self.buf.extend(b"\x00" * pad)
        return abs_off, len(raw)

    def add_f32(self, meta: dict[str, Any], name: str, arr: np.ndarray) -> None:
        f32 = np.ascontiguousarray(arr.astype(np.float32))
        off, nbytes = self.add(f32.tobytes())
        meta[name] = {
            "shard": self.shard_name,
            "offset": off,
            "bytes": nbytes,
            "shape": list(f32.shape),
            "dtype": "f32",
            "layout": "row_major",
        }

    def add_f16_packed(self, meta: dict[str, Any], name: str, arr: np.ndarray) -> None:
        w = np.asarray(arr, dtype=np.float32)
        m, k = w.shape
        k_pad = align_up(k, 2)
        if k_pad != k:
            w = np.pad(w, ((0, 0), (0, k_pad - k)), mode="constant", constant_values=0.0)
        f16_u32 = np.ascontiguousarray(w.astype(np.float16)).view(np.uint32).reshape(m, k_pad // 2)
        off, nbytes = self.add(f16_u32.tobytes())
        meta[name] = {
            "shard": self.shard_name,
            "offset": off,
            "bytes": nbytes,
            "shape": [m, k_pad // 2],
            "orig_shape": [m, k],
            "padded_shape": [m, k_pad],
            "dtype": "u32_f16x2",
            "layout": "row_major",
        }

    def add_q8(
        self,
        meta: dict[str, Any],
        name: str,
        arr: np.ndarray,
        verify_list: list[tuple[str, float]] | None = None,
        clip: list[float] | None = None,
    ) -> None:
        w = np.asarray(arr, dtype=np.float32)
        q_u32, s_u32, padded_shape = quantize_int8_g64_sym(w)
        q_off, q_bytes = self.add(q_u32.tobytes())
        s_off, s_bytes = self.add(s_u32.tobytes())
        q_entry: dict[str, Any] = {
            "shard": self.shard_name,
            "offset": q_off,
            "bytes": q_bytes,
            "shape": list(q_u32.shape),
            "orig_shape": list(w.shape),
            "padded_shape": list(padded_shape),
            "dtype": "u32_q8",
            "layout": "row_major",
        }
        if clip is not None:
            q_entry["clip"] = clip
        meta[f"{name}.quants"] = q_entry
        meta[f"{name}.scales"] = {
            "shard": self.shard_name,
            "offset": s_off,
            "bytes": s_bytes,
            "shape": list(s_u32.shape),
            "dtype": "u32_f16x2",
            "group_size": 64,
            "layout": "row_major",
        }
        if verify_list is not None:
            deq = dequantize_int8_g64_sym(q_u32, s_u32, w.shape[1])
            cos = float(np.sum(w * deq) / (np.linalg.norm(w) * np.linalg.norm(deq) + 1e-12))
            verify_list.append((name, cos))

    def add_i8_row(self, meta: dict[str, Any], name: str, arr: np.ndarray) -> None:
        raw, row_stride = quantize_int8_row(arr)
        off, nbytes = self.add(raw)
        meta[name] = {
            "shard": self.shard_name,
            "offset": off,
            "bytes": nbytes,
            "shape": list(arr.shape),
            "row_stride": row_stride,
            "dtype": "i8_row",
        }

    def finalize(self) -> dict[str, Any]:
        payload_bytes = len(self.buf)
        header = make_shard_header(self.shard_id_u32, payload_bytes)
        data = header + self.buf
        out_path = self.out_dir / self.filename
        out_path.write_bytes(data)
        sha = hashlib.sha256(data).hexdigest()
        return {
            "id": self.shard_name,
            "file": self.filename,
            "bytes": len(data),
            "payload_bytes": payload_bytes,
            "sha256": sha,
        }


def _pack_lfm3_omni(
    model_dir: Path,
    out_dir: Path,
    cfg: dict[str, Any],
    blueprint: dict[str, Any],
    hf_tensors,
    quant_errors: list[tuple[str, float]],
    verify: bool = True,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    mm_meta: dict[str, Any] = {}
    mm_shards: list[dict[str, Any]] = []
    vlist = quant_errors if verify else None

    # 1. Vision Tower + MultiModal Projector -> vision.ignis
    vw = SingleShardWriter(out_dir, "vision", "vision.ignis", 0xFFFFFF01)
    vis_cfg = cfg["vision_config"]
    vw.add_q8(mm_meta, "vis.patchifier", hf_tensors.get("model.vision_tower.patchifier.weight"), vlist)
    vw.add_f32(mm_meta, "vis.embedding_norm", hf_tensors.get("model.vision_tower.embedding_norm.weight"))

    for i, lt in enumerate(vis_cfg["layer_types"]):
        pfx = f"model.vision_tower.layers.{i}"
        opfx = f"vis.blk.{i}"
        vw.add_f32(mm_meta, f"{opfx}.operator_norm", hf_tensors.get(f"{pfx}.operator_norm.weight"))
        vw.add_f32(mm_meta, f"{opfx}.ffn_norm", hf_tensors.get(f"{pfx}.ffn_norm.weight"))
        if lt == "conv":
            vw.add_q8(mm_meta, f"{opfx}.conv.in_proj", hf_tensors.get(f"{pfx}.conv.in_proj.weight"), vlist)
            sc = hf_tensors.get(f"{pfx}.conv.spatial_conv.weight")  # [D, 1, 3, 3]
            vw.add_f32(mm_meta, f"{opfx}.conv.spatial_conv", sc.reshape(sc.shape[0], -1))
            vw.add_q8(mm_meta, f"{opfx}.conv.out_proj", hf_tensors.get(f"{pfx}.conv.out_proj.weight"), vlist)
        else:
            qkv = np.concatenate([
                hf_tensors.get(f"{pfx}.self_attn.q_proj.weight"),
                hf_tensors.get(f"{pfx}.self_attn.k_proj.weight"),
                hf_tensors.get(f"{pfx}.self_attn.v_proj.weight"),
            ], axis=0)
            vw.add_q8(mm_meta, f"{opfx}.attn_qkv", qkv, vlist)
            vw.add_f32(mm_meta, f"{opfx}.q_norm", hf_tensors.get(f"{pfx}.self_attn.q_layernorm.weight"))
            vw.add_f32(mm_meta, f"{opfx}.k_norm", hf_tensors.get(f"{pfx}.self_attn.k_layernorm.weight"))
            vw.add_q8(mm_meta, f"{opfx}.attn_out", hf_tensors.get(f"{pfx}.self_attn.out_proj.weight"), vlist)
        vw.add_q8(mm_meta, f"{opfx}.ffn_w1", hf_tensors.get(f"{pfx}.feed_forward.w1.weight"), vlist)
        vw.add_q8(mm_meta, f"{opfx}.ffn_w3", hf_tensors.get(f"{pfx}.feed_forward.w3.weight"), vlist)
        vw.add_q8(mm_meta, f"{opfx}.ffn_w2", hf_tensors.get(f"{pfx}.feed_forward.w2.weight"), vlist)

    vw.add_f32(mm_meta, "vis.proj.ln_w", hf_tensors.get("model.multi_modal_projector.layer_norm.weight"))
    vw.add_f32(mm_meta, "vis.proj.ln_b", hf_tensors.get("model.multi_modal_projector.layer_norm.bias"))
    vw.add_q8(mm_meta, "vis.proj.l1", hf_tensors.get("model.multi_modal_projector.linear_1.weight"), vlist)
    vw.add_f32(mm_meta, "vis.proj.l1_b", hf_tensors.get("model.multi_modal_projector.linear_1.bias"))
    vw.add_q8(mm_meta, "vis.proj.l2", hf_tensors.get("model.multi_modal_projector.linear_2.weight"), vlist)
    vw.add_f32(mm_meta, "vis.proj.l2_b", hf_tensors.get("model.multi_modal_projector.linear_2.bias"))
    mm_shards.append(vw.finalize())

    # 2. Audio Encoder (Downsampler + 17L LFM2 + Audio Projector) -> audio_enc.ignis
    aew = SingleShardWriter(out_dir, "audio_enc", "audio_enc.ignis", 0xFFFFFF02)
    aenc_cfg = cfg["audio_encoder_config"]["text_config"]
    dpfx = "model.audio_encoder.downsample"
    conv_in_w = hf_tensors.get(f"{dpfx}.conv_in.weight")  # [256, 1, 3, 3]
    aew.add_f32(mm_meta, "aenc.ds.conv_in_w", conv_in_w.reshape(conv_in_w.shape[0], -1))
    aew.add_f32(mm_meta, "aenc.ds.conv_in_b", hf_tensors.get(f"{dpfx}.conv_in.bias"))
    conv_mid_w = hf_tensors.get(f"{dpfx}.conv_mid.weight")  # [256, 1, 3, 3]
    aew.add_f32(mm_meta, "aenc.ds.conv_mid_w", conv_mid_w.reshape(conv_mid_w.shape[0], -1))
    aew.add_f32(mm_meta, "aenc.ds.conv_mid_b", hf_tensors.get(f"{dpfx}.conv_mid.bias"))
    lin_mid_w = hf_tensors.get(f"{dpfx}.lin_mid.weight")  # [256, 256, 1, 1]
    aew.add_f32(mm_meta, "aenc.ds.lin_mid_w", lin_mid_w.reshape(lin_mid_w.shape[0], lin_mid_w.shape[1]))
    aew.add_f32(mm_meta, "aenc.ds.lin_mid_b", hf_tensors.get(f"{dpfx}.lin_mid.bias"))
    conv_end_w = hf_tensors.get(f"{dpfx}.conv_end.weight")  # [256, 1, 3, 3]
    aew.add_f32(mm_meta, "aenc.ds.conv_end_w", conv_end_w.reshape(conv_end_w.shape[0], -1))
    aew.add_f32(mm_meta, "aenc.ds.conv_end_b", hf_tensors.get(f"{dpfx}.conv_end.bias"))
    lin_end_w = hf_tensors.get(f"{dpfx}.lin_end.weight")  # [256, 256, 1, 1]
    aew.add_f32(mm_meta, "aenc.ds.lin_end_w", lin_end_w.reshape(lin_end_w.shape[0], lin_end_w.shape[1]))
    aew.add_f32(mm_meta, "aenc.ds.lin_end_b", hf_tensors.get(f"{dpfx}.lin_end.bias"))
    lin_out_w = hf_tensors.get(f"{dpfx}.lin_out.weight")  # [768, 4096, 1]
    aew.add_q8(mm_meta, "aenc.ds.lin_out", lin_out_w.reshape(lin_out_w.shape[0], lin_out_w.shape[1]), vlist)
    aew.add_f32(mm_meta, "aenc.ds.lin_out_b", hf_tensors.get(f"{dpfx}.lin_out.bias"))

    # Discover real intermediate size from audio encoder checkpoint tensors
    blueprint["multimodal"]["audio_encoder_intermediate_size"] = int(
        hf_tensors.get_shape("model.audio_encoder.lfm.layers.0.feed_forward.w1.weight")[0]
    )

    for i, lt in enumerate(aenc_cfg["layer_types"]):
        pfx = f"model.audio_encoder.lfm.layers.{i}"
        opfx = f"aenc.blk.{i}"
        aew.add_f32(mm_meta, f"{opfx}.operator_norm", hf_tensors.get(f"{pfx}.operator_norm.weight"))
        aew.add_f32(mm_meta, f"{opfx}.ffn_norm", hf_tensors.get(f"{pfx}.ffn_norm.weight"))
        if lt == "conv":
            aew.add_q8(mm_meta, f"{opfx}.conv.in_proj", hf_tensors.get(f"{pfx}.conv.in_proj.weight"), vlist)
            cw = hf_tensors.get(f"{pfx}.conv.conv.weight")  # [768, 1, 3]
            aew.add_f32(mm_meta, f"{opfx}.conv.kernel", cw.reshape(cw.shape[0], -1))
            aew.add_q8(mm_meta, f"{opfx}.conv.out_proj", hf_tensors.get(f"{pfx}.conv.out_proj.weight"), vlist)
        else:
            qkv = np.concatenate([
                hf_tensors.get(f"{pfx}.self_attn.q_proj.weight"),
                hf_tensors.get(f"{pfx}.self_attn.k_proj.weight"),
                hf_tensors.get(f"{pfx}.self_attn.v_proj.weight"),
            ], axis=0)
            aew.add_q8(mm_meta, f"{opfx}.attn_qkv", qkv, vlist)
            aew.add_f32(mm_meta, f"{opfx}.q_norm", hf_tensors.get(f"{pfx}.self_attn.q_layernorm.weight"))
            aew.add_f32(mm_meta, f"{opfx}.k_norm", hf_tensors.get(f"{pfx}.self_attn.k_layernorm.weight"))
            aew.add_q8(mm_meta, f"{opfx}.attn_out", hf_tensors.get(f"{pfx}.self_attn.out_proj.weight"), vlist)
        aew.add_q8(mm_meta, f"{opfx}.ffn_w1", hf_tensors.get(f"{pfx}.feed_forward.w1.weight"), vlist)
        aew.add_q8(mm_meta, f"{opfx}.ffn_w3", hf_tensors.get(f"{pfx}.feed_forward.w3.weight"), vlist)
        aew.add_q8(mm_meta, f"{opfx}.ffn_w2", hf_tensors.get(f"{pfx}.feed_forward.w2.weight"), vlist)

    aew.add_f32(mm_meta, "aenc.embedding_norm", hf_tensors.get("model.audio_encoder.lfm.embedding_norm.weight"))
    aew.add_f32(mm_meta, "aenc.proj.ln_w", hf_tensors.get("model.audio_projector.layer_norm.weight"))
    aew.add_f32(mm_meta, "aenc.proj.ln_b", hf_tensors.get("model.audio_projector.layer_norm.bias"))
    aew.add_q8(mm_meta, "aenc.proj.l1", hf_tensors.get("model.audio_projector.linear_1.weight"), vlist)
    aew.add_f32(mm_meta, "aenc.proj.l1_b", hf_tensors.get("model.audio_projector.linear_1.bias"))
    aew.add_q8(mm_meta, "aenc.proj.l2", hf_tensors.get("model.audio_projector.linear_2.weight"), vlist)
    aew.add_f32(mm_meta, "aenc.proj.l2_b", hf_tensors.get("model.audio_projector.linear_2.bias"))
    mm_shards.append(aew.finalize())

    # 3. Audio Output (Codebook Embedding + DepthDecoder + LiquidCodec Decoder) -> audio_dec.ignis
    adw = SingleShardWriter(out_dir, "audio_dec", "audio_dec.ignis", 0xFFFFFF03)
    adw.add_i8_row(mm_meta, "adec.codebook_embd", hf_tensors.get("model.codebook_embedding.weight"))

    dp_cfg = cfg["depth_config"]
    adw.add_q8(mm_meta, "adec.depth.linear", hf_tensors.get("model.depth_decoder.depth_linear.weight"), vlist)
    adw.add_f32(mm_meta, "adec.depth.linear_b", hf_tensors.get("model.depth_decoder.depth_linear.bias"))
    adw.add_f32(mm_meta, "adec.depth.mask_emb", hf_tensors.get("model.depth_decoder.mask_embedding"))
    adw.add_f32(mm_meta, "adec.depth.out_norm", hf_tensors.get("model.depth_decoder.model.embedding_norm.weight"))

    for i in range(dp_cfg["num_hidden_layers"]):
        pfx = f"model.depth_decoder.model.layers.{i}"
        opfx = f"adec.depth.blk.{i}"
        adw.add_f32(mm_meta, f"{opfx}.operator_norm", hf_tensors.get(f"{pfx}.operator_norm.weight"))
        adw.add_f32(mm_meta, f"{opfx}.ffn_norm", hf_tensors.get(f"{pfx}.ffn_norm.weight"))
        qkv = np.concatenate([
            hf_tensors.get(f"{pfx}.self_attn.q_proj.weight"),
            hf_tensors.get(f"{pfx}.self_attn.k_proj.weight"),
            hf_tensors.get(f"{pfx}.self_attn.v_proj.weight"),
        ], axis=0)
        adw.add_q8(mm_meta, f"{opfx}.attn_qkv", qkv, vlist)
        adw.add_f32(mm_meta, f"{opfx}.q_norm", hf_tensors.get(f"{pfx}.self_attn.q_norm.weight"))
        adw.add_f32(mm_meta, f"{opfx}.k_norm", hf_tensors.get(f"{pfx}.self_attn.k_norm.weight"))
        adw.add_f32(mm_meta, f"{opfx}.attn_gate", hf_tensors.get(f"{pfx}.self_attn.gate_proj.weight"))
        adw.add_q8(mm_meta, f"{opfx}.attn_out", hf_tensors.get(f"{pfx}.self_attn.o_proj.weight"), vlist)
        adw.add_q8(mm_meta, f"{opfx}.ffn_w1", hf_tensors.get(f"{pfx}.feed_forward.w1.weight"), vlist)
        adw.add_q8(mm_meta, f"{opfx}.ffn_w3", hf_tensors.get(f"{pfx}.feed_forward.w3.weight"), vlist)
        adw.add_q8(mm_meta, f"{opfx}.ffn_w2", hf_tensors.get(f"{pfx}.feed_forward.w2.weight"), vlist)

    for k in range(cfg["num_codebooks"]):
        pfx = f"model.depth_decoder.depth_embeddings.{k}"
        adw.add_f16_packed(mm_meta, f"adec.depth.cb.{k}.emb", hf_tensors.get(f"{pfx}.embedding.weight"))
        adw.add_f32(mm_meta, f"adec.depth.cb.{k}.norm", hf_tensors.get(f"{pfx}.embedding_norm.weight"))

    codec_dir = model_dir / "audio_tokenizer"
    codec_cfg = json.loads((codec_dir / "config.json").read_text(encoding="utf-8"))
    blueprint["multimodal"]["codec_config"] = codec_cfg
    with safe_open(str(codec_dir / "model.safetensors"), framework="pt", device="cpu") as cf:
        def cg(key: str) -> np.ndarray:
            return cf.get_tensor(key).float().numpy()

        adw.add_f32(mm_meta, "adec.codec.project_out_w", cg("quantizer.project_out.weight"))
        adw.add_f32(mm_meta, "adec.codec.project_out_b", cg("quantizer.project_out.bias"))
        adw.add_f32(mm_meta, "adec.codec.from_q_w", cg("from_quantizer.weight"))
        adw.add_f32(mm_meta, "adec.codec.from_q_b", cg("from_quantizer.bias"))

        codec_layer_types = ["swa" if i in (2, 4, 6) else "conv" for i in range(8)]
        blueprint["multimodal"]["codec_layer_types"] = codec_layer_types
        for i, lt in enumerate(codec_layer_types):
            pfx = f"decoder.lfm.layers.{i}"
            opfx = f"adec.codec.blk.{i}"
            adw.add_f32(mm_meta, f"{opfx}.operator_norm", cg(f"{pfx}.operator_norm.weight"))
            adw.add_f32(mm_meta, f"{opfx}.post_operator_norm", cg(f"{pfx}.post_operator_norm.weight"))
            adw.add_f32(mm_meta, f"{opfx}.ffn_norm", cg(f"{pfx}.ffn_norm.weight"))
            adw.add_f32(mm_meta, f"{opfx}.post_ffn_norm", cg(f"{pfx}.post_ffn_norm.weight"))
            if lt == "conv":
                adw.add_f16_packed(mm_meta, f"{opfx}.conv.in_proj", cg(f"{pfx}.conv.in_proj.weight"))
                cw = cg(f"{pfx}.conv.conv.weight")  # [512, 1, 7]
                adw.add_f32(mm_meta, f"{opfx}.conv.kernel", cw.reshape(cw.shape[0], -1))
                adw.add_f16_packed(mm_meta, f"{opfx}.conv.out_proj", cg(f"{pfx}.conv.out_proj.weight"))
            else:
                qkv = np.concatenate([
                    cg(f"{pfx}.self_attn.q_proj.weight"),
                    cg(f"{pfx}.self_attn.k_proj.weight"),
                    cg(f"{pfx}.self_attn.v_proj.weight"),
                ], axis=0)
                adw.add_f16_packed(mm_meta, f"{opfx}.attn_qkv", qkv)
                adw.add_f32(mm_meta, f"{opfx}.q_norm", cg(f"{pfx}.self_attn.q_norm.weight"))
                adw.add_f32(mm_meta, f"{opfx}.k_norm", cg(f"{pfx}.self_attn.k_norm.weight"))
                adw.add_f32(mm_meta, f"{opfx}.attn_gate", cg(f"{pfx}.self_attn.gate_proj.weight"))
                adw.add_f16_packed(mm_meta, f"{opfx}.attn_out", cg(f"{pfx}.self_attn.o_proj.weight"))
            adw.add_f16_packed(mm_meta, f"{opfx}.ffn_w1", cg(f"{pfx}.feed_forward.w1.weight"))
            adw.add_f16_packed(mm_meta, f"{opfx}.ffn_w3", cg(f"{pfx}.feed_forward.w3.weight"))
            adw.add_f16_packed(mm_meta, f"{opfx}.ffn_w2", cg(f"{pfx}.feed_forward.w2.weight"))

        adw.add_f32(mm_meta, "adec.codec.out_norm", cg("decoder.lfm.embedding_norm.weight"))
        adw.add_f16_packed(mm_meta, "adec.codec.lin", cg("decoder.lin.weight"))
        adw.add_f32(mm_meta, "adec.codec.lin_b", cg("decoder.lin.bias"))
        adw.add_f32(mm_meta, "adec.codec.istft_window", cg("decoder.istft.window"))
    mm_shards.append(adw.finalize())

    return mm_shards, mm_meta


def _pack_gemma4_omni(
    out_dir: Path,
    cfg: dict[str, Any],
    hf_tensors,
    quant_errors: list[tuple[str, float]],
    verify: bool = True,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    mm_shards: list[dict[str, Any]] = []
    mm_meta: dict[str, Any] = {}
    vlist = quant_errors if verify else None

    def add_clipped_q8(writer: SingleShardWriter, name: str, hf_pfx: str) -> None:
        clip = None
        if f"{hf_pfx}.input_min" in hf_tensors:
            clip = [
                float(hf_tensors.get(f"{hf_pfx}.input_min").reshape(-1)[0]),
                float(hf_tensors.get(f"{hf_pfx}.input_max").reshape(-1)[0]),
                float(hf_tensors.get(f"{hf_pfx}.output_min").reshape(-1)[0]),
                float(hf_tensors.get(f"{hf_pfx}.output_max").reshape(-1)[0]),
            ]
        writer.add_q8(mm_meta, name, hf_tensors.get(f"{hf_pfx}.linear.weight"), vlist, clip=clip)

    # 1. Vision Tower + EmbedVision -> vision.ignis
    vc = cfg["vision_config"]
    vw = SingleShardWriter(out_dir, "vision", "vision.ignis", 0xFFFFFFF0)
    vw.add_q8(mm_meta, "vision.patch_proj", hf_tensors.get("model.vision_tower.patch_embedder.input_proj.weight"), vlist)
    pos_tbl = hf_tensors.get("model.vision_tower.patch_embedder.position_embedding_table")  # [2, 10240, 768]
    vw.add_f16_packed(mm_meta, "vision.pos_table", pos_tbl.reshape(-1, pos_tbl.shape[-1]))

    for i in range(vc["num_hidden_layers"]):
        pfx = f"model.vision_tower.encoder.layers.{i}"
        opfx = f"vision.blk.{i}"
        vw.add_f32(mm_meta, f"{opfx}.input_layernorm", hf_tensors.get(f"{pfx}.input_layernorm.weight"))
        vw.add_f32(mm_meta, f"{opfx}.post_attention_layernorm", hf_tensors.get(f"{pfx}.post_attention_layernorm.weight"))
        vw.add_f32(mm_meta, f"{opfx}.pre_feedforward_layernorm", hf_tensors.get(f"{pfx}.pre_feedforward_layernorm.weight"))
        vw.add_f32(mm_meta, f"{opfx}.post_feedforward_layernorm", hf_tensors.get(f"{pfx}.post_feedforward_layernorm.weight"))
        vw.add_f32(mm_meta, f"{opfx}.q_norm", hf_tensors.get(f"{pfx}.self_attn.q_norm.weight"))
        vw.add_f32(mm_meta, f"{opfx}.k_norm", hf_tensors.get(f"{pfx}.self_attn.k_norm.weight"))
        add_clipped_q8(vw, f"{opfx}.q_proj", f"{pfx}.self_attn.q_proj")
        add_clipped_q8(vw, f"{opfx}.k_proj", f"{pfx}.self_attn.k_proj")
        add_clipped_q8(vw, f"{opfx}.v_proj", f"{pfx}.self_attn.v_proj")
        add_clipped_q8(vw, f"{opfx}.o_proj", f"{pfx}.self_attn.o_proj")
        add_clipped_q8(vw, f"{opfx}.gate_proj", f"{pfx}.mlp.gate_proj")
        add_clipped_q8(vw, f"{opfx}.up_proj", f"{pfx}.mlp.up_proj")
        add_clipped_q8(vw, f"{opfx}.down_proj", f"{pfx}.mlp.down_proj")

    if vc.get("standardize") and "model.vision_tower.std_bias" in hf_tensors:
        vw.add_f32(mm_meta, "vision.std_bias", hf_tensors.get("model.vision_tower.std_bias"))
        vw.add_f32(mm_meta, "vision.std_scale", hf_tensors.get("model.vision_tower.std_scale"))
    vw.add_q8(mm_meta, "vision.embed_proj", hf_tensors.get("model.embed_vision.embedding_projection.weight"), vlist)
    mm_shards.append(vw.finalize())

    # 2. Audio Tower (SubSample + 12L Conformer) + EmbedAudio -> audio_enc.ignis
    ac = cfg["audio_config"]
    aw = SingleShardWriter(out_dir, "audio_enc", "audio_enc.ignis", 0xFFFFFFF1)
    aw.add_f32(mm_meta, "audio.sub.layer0.conv", hf_tensors.get("model.audio_tower.subsample_conv_projection.layer0.conv.weight"))
    aw.add_f32(mm_meta, "audio.sub.layer0.norm", hf_tensors.get("model.audio_tower.subsample_conv_projection.layer0.norm.weight"))
    aw.add_f32(mm_meta, "audio.sub.layer1.conv", hf_tensors.get("model.audio_tower.subsample_conv_projection.layer1.conv.weight"))
    aw.add_f32(mm_meta, "audio.sub.layer1.norm", hf_tensors.get("model.audio_tower.subsample_conv_projection.layer1.norm.weight"))
    aw.add_q8(mm_meta, "audio.sub.input_proj", hf_tensors.get("model.audio_tower.subsample_conv_projection.input_proj_linear.weight"), vlist)

    for i in range(ac["num_hidden_layers"]):
        pfx = f"model.audio_tower.layers.{i}"
        opfx = f"audio.blk.{i}"
        aw.add_f32(mm_meta, f"{opfx}.ff1.pre_norm", hf_tensors.get(f"{pfx}.feed_forward1.pre_layer_norm.weight"))
        aw.add_f32(mm_meta, f"{opfx}.ff1.post_norm", hf_tensors.get(f"{pfx}.feed_forward1.post_layer_norm.weight"))
        add_clipped_q8(aw, f"{opfx}.ff1.w1", f"{pfx}.feed_forward1.ffw_layer_1")
        add_clipped_q8(aw, f"{opfx}.ff1.w2", f"{pfx}.feed_forward1.ffw_layer_2")

        aw.add_f32(mm_meta, f"{opfx}.norm_pre_attn", hf_tensors.get(f"{pfx}.norm_pre_attn.weight"))
        aw.add_f32(mm_meta, f"{opfx}.norm_post_attn", hf_tensors.get(f"{pfx}.norm_post_attn.weight"))
        add_clipped_q8(aw, f"{opfx}.q_proj", f"{pfx}.self_attn.q_proj")
        add_clipped_q8(aw, f"{opfx}.k_proj", f"{pfx}.self_attn.k_proj")
        add_clipped_q8(aw, f"{opfx}.v_proj", f"{pfx}.self_attn.v_proj")
        add_clipped_q8(aw, f"{opfx}.post_proj", f"{pfx}.self_attn.post")
        aw.add_q8(mm_meta, f"{opfx}.rel_k_proj", hf_tensors.get(f"{pfx}.self_attn.relative_k_proj.weight"), vlist)
        aw.add_f32(mm_meta, f"{opfx}.per_dim_scale", hf_tensors.get(f"{pfx}.self_attn.per_dim_scale"))

        aw.add_f32(mm_meta, f"{opfx}.lconv.pre_norm", hf_tensors.get(f"{pfx}.lconv1d.pre_layer_norm.weight"))
        add_clipped_q8(aw, f"{opfx}.lconv.lin_start", f"{pfx}.lconv1d.linear_start")
        dw_w = hf_tensors.get(f"{pfx}.lconv1d.depthwise_conv1d.weight")
        aw.add_f32(mm_meta, f"{opfx}.lconv.dw_conv", dw_w.reshape(dw_w.shape[0], -1))
        aw.add_f32(mm_meta, f"{opfx}.lconv.conv_norm", hf_tensors.get(f"{pfx}.lconv1d.conv_norm.weight"))
        add_clipped_q8(aw, f"{opfx}.lconv.lin_end", f"{pfx}.lconv1d.linear_end")

        aw.add_f32(mm_meta, f"{opfx}.ff2.pre_norm", hf_tensors.get(f"{pfx}.feed_forward2.pre_layer_norm.weight"))
        aw.add_f32(mm_meta, f"{opfx}.ff2.post_norm", hf_tensors.get(f"{pfx}.feed_forward2.post_layer_norm.weight"))
        add_clipped_q8(aw, f"{opfx}.ff2.w1", f"{pfx}.feed_forward2.ffw_layer_1")
        add_clipped_q8(aw, f"{opfx}.ff2.w2", f"{pfx}.feed_forward2.ffw_layer_2")

        aw.add_f32(mm_meta, f"{opfx}.norm_out", hf_tensors.get(f"{pfx}.norm_out.weight"))

    aw.add_q8(mm_meta, "audio.output_proj", hf_tensors.get("model.audio_tower.output_proj.weight"), vlist)
    aw.add_f32(mm_meta, "audio.output_proj_b", hf_tensors.get("model.audio_tower.output_proj.bias"))
    aw.add_q8(mm_meta, "audio.embed_proj", hf_tensors.get("model.embed_audio.embedding_projection.weight"), vlist)
    mm_shards.append(aw.finalize())

    return mm_shards, mm_meta


def pack_multimodal_shards(
    model_dir: Path,
    out_dir: Path,
    cfg: dict[str, Any],
    blueprint: dict[str, Any],
    hf_tensors,
    quant_errors: list[tuple[str, float]],
    verify: bool = True,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    mm_bp = blueprint.get("multimodal")
    if not mm_bp:
        return [], {}
    kind = mm_bp.get("kind")
    if kind == "lfm3_omni":
        return _pack_lfm3_omni(model_dir, out_dir, cfg, blueprint, hf_tensors, quant_errors, verify=verify)
    if kind == "gemma4":
        return _pack_gemma4_omni(out_dir, cfg, hf_tensors, quant_errors, verify=verify)
    raise ValueError(f"Unsupported multimodal kind: {kind}")
