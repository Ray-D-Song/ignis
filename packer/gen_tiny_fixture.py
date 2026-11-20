"""
Standalone PyTorch reference for Spark2_5ForCausalLM (matching docs/modeling_spark_ref.py)
and isomorphic Tiny-Spark fixture + golden reference generator.
"""

from __future__ import annotations
import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from safetensors.torch import save_file


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def compute_rope_cos_sin(positions: torch.Tensor, head_dim: int, rope_theta: float, partial_rotary_factor: float):
    rope_head_dim = int(round(head_dim * partial_rotary_factor))
    inv_freq = 1.0 / (rope_theta ** (torch.arange(0, rope_head_dim, 2, dtype=torch.float32) / rope_head_dim))
    freqs = torch.outer(positions.float(), inv_freq)
    freqs = torch.cat([freqs, freqs], dim=-1)
    return freqs.cos(), freqs.sin()


def apply_rotary_pos_emb(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    rope_head_dim = cos.shape[-1]
    c = cos.unsqueeze(0).unsqueeze(0)
    s = sin.unsqueeze(0).unsqueeze(0)
    if x.shape[-1] > rope_head_dim:
        x_rot = x[..., :rope_head_dim]
        x_pass = x[..., rope_head_dim:]
        x_rot = x_rot * c + rotate_half(x_rot) * s
        return torch.cat([x_rot, x_pass], dim=-1)
    return x * c + rotate_half(x) * s


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        xf = x.float()
        var = xf.pow(2).mean(-1, keepdim=True)
        return (self.weight.float() * xf * torch.rsqrt(var + self.eps)).to(x.dtype)


class SparkMLP(nn.Module):
    def __init__(self, hidden_size: int, intermediate_size: int) -> None:
        super().__init__()
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(F.gelu(self.gate_proj(x)) * self.up_proj(x))


class SparkAttention(nn.Module):
    def __init__(self, cfg: dict[str, Any], layer_idx: int) -> None:
        super().__init__()
        self.layer_idx = layer_idx
        self.hidden_size = cfg["hidden_size"]
        self.num_heads = cfg["num_attention_heads"]
        self.num_kv_heads = cfg["num_key_value_heads"]
        self.head_dim = cfg["head_dim"]
        self.num_groups = self.num_heads // self.num_kv_heads
        self.scaling = 1.0 / math.sqrt(self.head_dim)
        self.q_dim = self.num_heads * self.head_dim
        self.kv_dim = self.num_kv_heads * self.head_dim

        self.q_k_v_proj = nn.Linear(self.hidden_size, self.q_dim + 2 * self.kv_dim, bias=False)
        self.g_proj = nn.Linear(self.hidden_size, self.num_heads, bias=False)
        self.out_proj = nn.Linear(self.q_dim, self.hidden_size, bias=False)

        self.layer_type = cfg["layer_types"][layer_idx]
        self.sliding_window = cfg["sliding_window"] if self.layer_type == "sliding_attention" else None
        rp = cfg["rope_parameters"][self.layer_type]
        self.rope_theta = float(rp["rope_theta"])
        self.partial_rotary_factor = float(rp["partial_rotary_factor"])

    def forward(
        self,
        hidden_states: torch.Tensor,
        positions: torch.Tensor,
        kv_cache: dict[int, tuple[torch.Tensor, torch.Tensor]],
    ) -> torch.Tensor:
        bsz, seq_len, _ = hidden_states.shape
        qkv = self.q_k_v_proj(hidden_states)
        q = qkv[..., : self.q_dim]
        k = qkv[..., self.q_dim : self.q_dim + self.kv_dim]
        v = qkv[..., self.q_dim + self.kv_dim :]
        gate_score = self.g_proj(hidden_states)

        q = q.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(bsz, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = v.view(bsz, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2)
        gate_score = gate_score.view(bsz, seq_len, self.num_heads, 1).transpose(1, 2)

        cos, sin = compute_rope_cos_sin(positions, self.head_dim, self.rope_theta, self.partial_rotary_factor)
        q = apply_rotary_pos_emb(q, cos, sin)
        k = apply_rotary_pos_emb(k, cos, sin)

        if self.layer_idx in kv_cache:
            pk, pv = kv_cache[self.layer_idx]
            k = torch.cat([pk, k], dim=2)
            v = torch.cat([pv, v], dim=2)
        kv_cache[self.layer_idx] = (k, v)

        total_kv = k.shape[2]
        k_rep = k[:, :, None, :, :].expand(bsz, self.num_kv_heads, self.num_groups, total_kv, self.head_dim).reshape(bsz, self.num_heads, total_kv, self.head_dim)
        v_rep = v[:, :, None, :, :].expand(bsz, self.num_kv_heads, self.num_groups, total_kv, self.head_dim).reshape(bsz, self.num_heads, total_kv, self.head_dim)

        scores = torch.matmul(q, k_rep.transpose(2, 3)) * self.scaling
        # Causal + optional sliding window mask
        q_pos = positions[:, None]
        k_pos = torch.arange(total_kv, dtype=torch.long)[None, :]
        mask = k_pos <= q_pos
        if self.sliding_window is not None:
            mask = mask & ((q_pos - k_pos) < self.sliding_window)
        scores = scores.masked_fill(~mask[None, None, :, :], float("-inf"))

        probs = F.softmax(scores, dim=-1, dtype=torch.float32)
        attn_out = torch.matmul(probs, v_rep)
        attn_out = attn_out * torch.sigmoid(gate_score.float())
        attn_out = attn_out.transpose(1, 2).contiguous().view(bsz, seq_len, self.q_dim)
        return self.out_proj(attn_out)


class SparkDecoderLayer(nn.Module):
    def __init__(self, cfg: dict[str, Any], layer_idx: int) -> None:
        super().__init__()
        self.input_layernorm = RMSNorm(cfg["hidden_size"], cfg["rms_norm_eps"])
        self.self_attn = SparkAttention(cfg, layer_idx)
        self.post_attention_layernorm = RMSNorm(cfg["hidden_size"], cfg["rms_norm_eps"])
        self.mlp = SparkMLP(cfg["hidden_size"], cfg["intermediate_size"])

    def forward(self, x: torch.Tensor, positions: torch.Tensor, kv_cache: dict[int, tuple[torch.Tensor, torch.Tensor]]) -> torch.Tensor:
        x = x + self.self_attn(self.input_layernorm(x), positions, kv_cache)
        x = x + self.mlp(self.post_attention_layernorm(x))
        return x


class TinySparkModel(nn.Module):
    def __init__(self, cfg: dict[str, Any]) -> None:
        super().__init__()
        self.cfg = cfg
        self.embedding = nn.Embedding(cfg["vocab_size"], cfg["hidden_size"])
        self.layers = nn.ModuleList([SparkDecoderLayer(cfg, i) for i in range(cfg["num_hidden_layers"])])
        self.norm = RMSNorm(cfg["hidden_size"], cfg["rms_norm_eps"])

    def forward_with_traces(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        kv_cache: dict[int, tuple[torch.Tensor, torch.Tensor]],
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        traces: dict[str, Any] = {}
        x = self.embedding(input_ids).float()
        traces["embed"] = x.detach().clone()
        for idx, layer in enumerate(self.layers):
            x = layer(x, positions, kv_cache)
            traces[f"layer_{idx}"] = x.detach().clone()
        x = self.norm(x)
        traces["final_norm"] = x.detach().clone()
        logits = F.linear(x, self.embedding.weight.float())
        return logits, traces


def make_tiny_config() -> dict[str, Any]:
    return {
        "architectures": ["Spark2_5ForCausalLM"],
        "model_type": "spark2_5",
        "vocab_size": 256,
        "hidden_size": 256,
        "intermediate_size": 512,
        "num_hidden_layers": 4,
        "num_attention_heads": 8,
        "num_key_value_heads": 2,
        "head_dim": 64,
        "hidden_act": "gelu",
        "rms_norm_eps": 1e-6,
        "sliding_window": 8,
        "layer_types": ["sliding_attention", "sliding_attention", "sliding_attention", "full_attention"],
        "rope_parameters": {
            "sliding_attention": {"rope_theta": 10000, "partial_rotary_factor": 1.0},
            "full_attention": {"rope_theta": 5000000, "partial_rotary_factor": 0.25},
        },
        "headwise_attn_output_gate": True,
        "gate_attn_act_mode": "sigmoid",
        "tie_word_embeddings": True,
        "bos_token_id": 0,
        "eos_token_id": 1,
        "pad_token_id": 2,
    }


def generate_tiny_fixture(out_hf_dir: Path, out_golden_path: Path) -> None:
    torch.manual_seed(42)
    cfg = make_tiny_config()
    out_hf_dir.mkdir(parents=True, exist_ok=True)
    out_golden_path.parent.mkdir(parents=True, exist_ok=True)

    model = TinySparkModel(cfg)
    with torch.no_grad():
        for name, p in model.named_parameters():
            if "norm" in name:
                p.normal_(mean=1.0, std=0.05)
            else:
                p.normal_(mean=0.0, std=0.04)

    # Save HF Safetensors + config
    state = {f"model.{k}": v.half() for k, v in model.state_dict().items()}
    save_file(state, str(out_hf_dir / "model.safetensors"))
    (out_hf_dir / "config.json").write_text(json.dumps(cfg, indent=2), encoding="utf-8")

    # Convert reference model to float32 matching stored fp16 weights for exact baseline
    with torch.no_grad():
        for k, v in model.state_dict().items():
            v.copy_(v.half().float())

    # Prompt of 12 tokens -> forces SWA (window=8) to evict first 4 tokens!
    prompt_ids = [10, 24, 55, 7, 91, 128, 3, 200, 44, 19, 88, 12]
    kv_cache: dict[int, tuple[torch.Tensor, torch.Tensor]] = {}
    with torch.no_grad():
        inp = torch.tensor([prompt_ids], dtype=torch.long)
        pos = torch.arange(len(prompt_ids), dtype=torch.long)
        logits, prefill_traces = model.forward_with_traces(inp, pos, kv_cache)

        last_logits = logits[0, -1, :]
        next_tok = int(torch.argmax(last_logits).item())
        generated = [next_tok]
        decode_traces = []
        for step in range(5):
            cur_pos = torch.tensor([len(prompt_ids) + step], dtype=torch.long)
            cur_in = torch.tensor([[generated[-1]]], dtype=torch.long)
            d_logits, d_tr = model.forward_with_traces(cur_in, cur_pos, kv_cache)
            decode_traces.append({
                "pos": int(cur_pos.item()),
                "token_in": generated[-1],
                "layer_0": d_tr["layer_0"][0, 0].tolist(),
                "layer_3": d_tr["layer_3"][0, 0].tolist(),
                "final_norm": d_tr["final_norm"][0, 0].tolist(),
                "logits_top8_idx": torch.topk(d_logits[0, 0], 8).indices.tolist(),
                "logits_top8_val": torch.topk(d_logits[0, 0], 8).values.tolist(),
            })
            generated.append(int(torch.argmax(d_logits[0, 0]).item()))

    golden = {
        "prompt_ids": prompt_ids,
        "prefill_last": {
            "embed": prefill_traces["embed"][0, -1].tolist(),
            "layer_0": prefill_traces["layer_0"][0, -1].tolist(),
            "layer_3": prefill_traces["layer_3"][0, -1].tolist(),
            "final_norm": prefill_traces["final_norm"][0, -1].tolist(),
            "logits_top8_idx": torch.topk(last_logits, 8).indices.tolist(),
            "logits_top8_val": torch.topk(last_logits, 8).values.tolist(),
        },
        "generated_greedy_ids": generated,
        "decode_steps": decode_traces,
    }
    out_golden_path.write_text(json.dumps(golden, indent=2), encoding="utf-8")
    print(f"[tiny_fixture] saved HF -> {out_hf_dir}, golden -> {out_golden_path}, greedy={generated}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--hf-out", type=Path, required=True)
    parser.add_argument("--golden-out", type=Path, required=True)
    args = parser.parse_args()
    generate_tiny_fixture(args.hf_out, args.golden_out)


if __name__ == "__main__":
    main()
