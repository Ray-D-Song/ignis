"""Declarative Model Blueprint & Tensor Mapping Registry for Ignis Framework.

Each supported architecture family defines its normalized blueprint parameters and HF->Ignis
weight mapping in < 100 lines, eliminating per-model runtime/shader rewrites.
"""

from __future__ import annotations
from typing import Any


def _normalize_layer_types(raw_types: list[str] | None, num_layers: int) -> list[str]:
    if not raw_types:
        return ["full"] * num_layers
    out = []
    for lt in raw_types:
        if lt in ("sliding_attention", "swa"):
            out.append("swa")
        elif lt == "conv":
            out.append("conv")
        else:
            out.append("full")
    return out


def build_blueprint(cfg: dict[str, Any]) -> dict[str, Any]:
    """Convert any supported HuggingFace config.json into an Ignis Model Blueprint."""
    top_type = cfg.get("model_type", "spark2_5")
    text_cfg = cfg.get("text_config", cfg) if top_type in ("lfm3_omni", "lfm3_vl", "gemma4") else cfg
    arch = text_cfg.get("model_type", top_type)

    num_layers = text_cfg["num_hidden_layers"]
    hidden_size = text_cfg["hidden_size"]
    num_heads = text_cfg["num_attention_heads"]
    num_kv_heads = text_cfg["num_key_value_heads"]
    head_dim = text_cfg.get("head_dim") or (hidden_size // num_heads)
    layer_types = _normalize_layer_types(text_cfg.get("layer_types"), num_layers)

    rope_params = text_cfg.get("rope_parameters") or {}
    full_rope = rope_params.get("full_attention", rope_params)
    swa_rope = rope_params.get("sliding_attention", {})

    base: dict[str, Any] = {
        "architecture": arch,
        "top_model_type": top_type,
        "num_layers": num_layers,
        "hidden_size": hidden_size,
        "intermediate_size": text_cfg["intermediate_size"],
        "vocab_size": text_cfg["vocab_size"],
        "num_attention_heads": num_heads,
        "num_key_value_heads": num_kv_heads,
        "head_dim": head_dim,
        "head_dims": [head_dim] * num_layers,
        "rms_norm_eps": text_cfg.get("rms_norm_eps", text_cfg.get("norm_eps", 1e-6)),
        "rms_norm_offset": 0.0,
        "hidden_act": text_cfg.get("hidden_activation", text_cfg.get("hidden_act", "gelu")),
        "sliding_window": text_cfg.get("sliding_window"),
        "rope_layout": "truncated",
        "rope_theta_full": float(full_rope.get("rope_theta", text_cfg.get("rope_theta", 5000000.0))),
        "rope_dim_full": int(
            round(
                text_cfg.get("global_head_dim", head_dim)
                * float(full_rope.get("partial_rotary_factor", text_cfg.get("partial_rotary_factor", 1.0)))
            )
        ),
        "rope_theta_swa": float(swa_rope.get("rope_theta", 10000.0)),
        "rope_dim_swa": int(round(head_dim * float(swa_rope.get("partial_rotary_factor", 1.0)))),
        "layer_types": layer_types,
        "attn_scale": float(head_dim) ** -0.5,
        "has_attn_gate": bool(text_cfg.get("headwise_attn_output_gate", False)),
        "gate_attn_act_mode": text_cfg.get("gate_attn_act_mode", "sigmoid"),
        "has_qk_norm": False,
        "has_v_norm": False,
        "has_qkv_bias": False,
        "has_post_norms": False,
        "use_sandwich_norm": False,
        "conv_kernel_size": 0,
        "embed_scale": 1.0,
        "has_ple": False,
        "ple_dim": 0,
        "kv_source_layers": list(range(num_layers)),
        "has_layer_scalar": False,
        "layer_scalars": [1.0] * num_layers,
        "final_logit_softcapping": text_cfg.get("final_logit_softcapping"),
        "tie_word_embeddings": bool(text_cfg.get("tie_word_embeddings", cfg.get("tie_word_embeddings", False))),
        "bos_token_id": text_cfg.get("bos_token_id", cfg.get("bos_token_id", 0)),
        "eos_token_id": (cfg.get("eos_token_id", text_cfg.get("eos_token_id", 1))[0]
                         if isinstance(cfg.get("eos_token_id", text_cfg.get("eos_token_id", 1)), list)
                         else cfg.get("eos_token_id", text_cfg.get("eos_token_id", 1))),
        "eos_token_ids": (cfg.get("eos_token_id", text_cfg.get("eos_token_id", 1))
                          if isinstance(cfg.get("eos_token_id", text_cfg.get("eos_token_id", 1)), list)
                          else [cfg.get("eos_token_id", text_cfg.get("eos_token_id", 1))]),
        "pad_token_id": text_cfg.get("pad_token_id", cfg.get("pad_token_id", 2)),
    }

    if arch in ("lfm3", "lfm2") or top_type == "lfm3_omni":
        base["rope_layout"] = "proportional"
        base["hidden_act"] = "silu"
        base["conv_kernel_size"] = int(text_cfg.get("conv_L_cache", 7))
        base["use_sandwich_norm"] = bool(text_cfg.get("use_sandwich_norm", True))
        base["has_post_norms"] = base["use_sandwich_norm"]
        base["has_qk_norm"] = True
        base["has_attn_gate"] = True
        base["embed_scale"] = 1.0
        base["rms_norm_eps"] = float(text_cfg.get("norm_eps", 1e-5))
        if top_type == "lfm3_omni":
            base["multimodal"] = {
                "kind": "lfm3_omni",
                "image_token_id": cfg.get("image_token_id", 125030),
                "audio_token_id": cfg.get("audio_token_id", 125031),
                "audio_out_token_id": cfg.get("audio_out_token_id", 125032),
                "audio_start_token_id": cfg.get("audio_start_token_id", 125012),
                "mixed_start_token_id": cfg.get("mixed_start_token_id", 125015),
                "text_end_token_id": cfg.get("text_end_token_id", 125014),
                "downsample_factor": cfg.get("downsample_factor", 2),
                "projector_hidden_act": cfg.get("projector_hidden_act", "gelu"),
                "projector_layer_norm_eps": cfg.get("projector_layer_norm_eps", 1e-6),
                "num_codebooks": cfg.get("num_codebooks", 12),
                "codebook_size": cfg.get("codebook_size", 2017),
                "audio_eos_code": cfg.get("audio_eos_code", 2016),
                "interleaved_n_audio": cfg.get("interleaved_n_audio", 12),
                "interleaved_n_text": cfg.get("interleaved_n_text", 6),
                "maskgit_schedule": cfg.get("maskgit_schedule", [4, 4, 4]),
                "vision_config": cfg.get("vision_config"),
                "audio_encoder_config": cfg.get("audio_encoder_config"),
                "depth_config": cfg.get("depth_config"),
            }
        return base

    if arch in ("gemma4", "gemma4_text") or top_type == "gemma4":
        global_hd = int(text_cfg.get("global_head_dim", 512))
        base["head_dims"] = [global_hd if lt == "full" else head_dim for lt in layer_types]
        base["head_dim"] = max(base["head_dims"])
        base["rope_layout"] = "proportional"
        base["hidden_act"] = text_cfg.get("hidden_activation", "gelu_pytorch_tanh")
        base["attn_scale"] = 1.0
        base["has_qk_norm"] = True
        base["has_v_norm"] = True
        base["has_post_norms"] = True
        base["has_layer_scalar"] = True
        base["embed_scale"] = float(hidden_size ** 0.5)

        ple_dim = int(text_cfg.get("hidden_size_per_layer_input", 0) or 0)
        if ple_dim > 0:
            base["has_ple"] = True
            base["ple_dim"] = ple_dim
            base["vocab_size_per_layer_input"] = int(text_cfg.get("vocab_size_per_layer_input", text_cfg["vocab_size"]))

        n_shared = int(text_cfg.get("num_kv_shared_layers", 0) or 0)
        if n_shared > 0:
            first_shared = num_layers - n_shared
            producers: dict[str, int] = {}
            for idx in range(first_shared):
                producers[layer_types[idx]] = idx
            src_layers = []
            for idx in range(num_layers):
                src_layers.append(producers[layer_types[idx]] if idx >= first_shared else idx)
            base["kv_source_layers"] = src_layers

        if top_type == "gemma4":
            base["multimodal"] = {
                "kind": "gemma4",
                "image_token_id": cfg.get("image_token_id", 258880),
                "audio_token_id": cfg.get("audio_token_id", 258881),
                "boi_token_id": cfg.get("boi_token_id", 255999),
                "eoi_token_id": cfg.get("eoi_token_id", 258882),
                "boa_token_id": cfg.get("boa_token_id", 256000),
                "eoa_token_id": cfg.get("eoa_token_id", 258883),
                "vision_soft_tokens_per_image": cfg.get("vision_soft_tokens_per_image", 280),
                "vision_config": cfg.get("vision_config"),
                "audio_config": cfg.get("audio_config"),
            }
        return base

    if arch in ("qwen2", "qwen3", "llama", "phi3"):
        base["rope_layout"] = "truncated"
        base["hidden_act"] = "silu"
        if arch == "qwen3":
            base["has_qk_norm"] = True
        return base

    return base
