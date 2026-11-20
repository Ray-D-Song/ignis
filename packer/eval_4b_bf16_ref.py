"""
Run exact BF16 PyTorch forward pass on Spark-X2.6-4B (using tests/fixtures/spark-4b-hf)
to produce a ground-truth BF16 golden baseline (full logits & greedy tokens) for comparing
INT4_G64 LM Head vs INT8_G64 LM Head in Ignis WebGPU.
"""

from __future__ import annotations
import json
import math
import time
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from safetensors import safe_open

from gen_tiny_fixture import apply_rotary_pos_emb, compute_rope_cos_sin


def load_4b_weights(model_dir: Path) -> tuple[dict, dict[str, torch.Tensor]]:
    cfg = json.loads((model_dir / "config.json").read_text(encoding="utf-8"))
    idx = json.loads((model_dir / "model.safetensors.index.json").read_text(encoding="utf-8"))
    files = sorted(set(idx["weight_map"].values()))
    tw: dict[str, torch.Tensor] = {}
    for fname in files:
        with safe_open(str(model_dir / fname), framework="pt", device="cpu") as f:
            for k in f.keys():
                tw[k] = f.get_tensor(k)
    return cfg, tw


def rms_norm(x: torch.Tensor, w: torch.Tensor, eps: float) -> torch.Tensor:
    xf = x.float()
    return xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + eps) * w.float()


@torch.no_grad()
def forward_bf16_step(
    cfg: dict,
    tw: dict[str, torch.Tensor],
    input_ids: torch.Tensor,
    positions: torch.Tensor,
    kv_cache: dict[int, tuple[torch.Tensor, torch.Tensor]],
) -> torch.Tensor:
    x = tw["model.embedding.weight"][input_ids].float()  # [B, T, D]
    bsz, seq_len, _ = x.shape
    nh = cfg["num_attention_heads"]
    nkv = cfg["num_key_value_heads"]
    hd = cfg["head_dim"]
    ng = nh // nkv
    q_dim = nh * hd
    kv_dim = nkv * hd
    eps = cfg["rms_norm_eps"]
    scaling = 1.0 / math.sqrt(hd)

    rope_cache = {}
    for lt in ("sliding_attention", "full_attention"):
        rp = cfg["rope_parameters"][lt]
        rope_cache[lt] = compute_rope_cos_sin(positions, hd, float(rp["rope_theta"]), float(rp["partial_rotary_factor"]))

    for i in range(cfg["num_hidden_layers"]):
        pfx = f"model.layers.{i}"
        lt = cfg["layer_types"][i]
        win = cfg["sliding_window"] if lt == "sliding_attention" else None
        cos, sin = rope_cache[lt]

        h = rms_norm(x, tw[f"{pfx}.input_layernorm.weight"], eps).bfloat16()
        qkv = F.linear(h, tw[f"{pfx}.self_attn.q_k_v_proj.weight"]).float()
        q = qkv[..., :q_dim].view(bsz, seq_len, nh, hd).transpose(1, 2)
        k = qkv[..., q_dim : q_dim + kv_dim].view(bsz, seq_len, nkv, hd).transpose(1, 2)
        v = qkv[..., q_dim + kv_dim :].view(bsz, seq_len, nkv, hd).transpose(1, 2)

        q = apply_rotary_pos_emb(q, cos, sin)
        k = apply_rotary_pos_emb(k, cos, sin)

        if i in kv_cache:
            pk, pv = kv_cache[i]
            k = torch.cat([pk, k], dim=2)
            v = torch.cat([pv, v], dim=2)
        kv_cache[i] = (k, v)

        total_kv = k.shape[2]
        k_rep = k[:, :, None, :, :].expand(bsz, nkv, ng, total_kv, hd).reshape(bsz, nh, total_kv, hd)
        v_rep = v[:, :, None, :, :].expand(bsz, nkv, ng, total_kv, hd).reshape(bsz, nh, total_kv, hd)

        scores = torch.matmul(q, k_rep.transpose(2, 3)) * scaling
        q_pos = positions[:, None]
        k_pos = torch.arange(total_kv, dtype=torch.long)[None, :]
        mask = k_pos <= q_pos
        if win is not None:
            mask = mask & ((q_pos - k_pos) < win)
        scores = scores.masked_fill(~mask[None, None, :, :], float("-inf"))
        probs = F.softmax(scores, dim=-1, dtype=torch.float32)
        attn_out = torch.matmul(probs, v_rep)

        gate = torch.sigmoid(F.linear(h, tw[f"{pfx}.self_attn.g_proj.weight"]).float())
        gate = gate.view(bsz, seq_len, nh, 1).transpose(1, 2)
        attn_out = (attn_out * gate).transpose(1, 2).contiguous().view(bsz, seq_len, q_dim).bfloat16()
        x = x + F.linear(attn_out, tw[f"{pfx}.self_attn.out_proj.weight"]).float()

        hf = rms_norm(x, tw[f"{pfx}.post_attention_layernorm.weight"], eps).bfloat16()
        g_out = F.gelu(F.linear(hf, tw[f"{pfx}.mlp.gate_proj.weight"]).float())
        u_out = F.linear(hf, tw[f"{pfx}.mlp.up_proj.weight"]).float()
        x = x + F.linear((g_out * u_out).bfloat16(), tw[f"{pfx}.mlp.down_proj.weight"]).float()

    x = rms_norm(x[:, -1:, :], tw["model.norm.weight"], eps).bfloat16()
    logits = F.linear(x, tw["model.embedding.weight"]).float()
    return logits[0, 0]


def main() -> None:
    root = Path(__file__).resolve().parent.parent
    model_dir = root / "tests/fixtures/spark-4b-hf"
    t0 = time.time()
    cfg, tw = load_4b_weights(model_dir)
    print(f"Loaded BF16 4B weights in {time.time() - t0:.1f}s")

    # Token IDs will be passed in via JSON or hardcoded from llama-tokenize; we support prompt_ids JSON file
    prompts = json.loads((root / "tests/fixtures/spark-4b-prompts.json").read_text(encoding="utf-8"))
    results = []
    for item in prompts:
        ids = item["ids"]
        kv_cache: dict[int, tuple[torch.Tensor, torch.Tensor]] = {}
        t_p = time.time()
        logits = forward_bf16_step(cfg, tw, torch.tensor([ids], dtype=torch.long), torch.arange(len(ids), dtype=torch.long), kv_cache)
        prefill_s = time.time() - t_p

        # Save full prefill logits as float32 binary for exact cosine / KL comparison, plus greedy 24 tokens
        logits_bin_path = root / f"tests/fixtures/{item['name']}_bf16_logits.f32"
        logits.numpy().astype(np.float32).tofile(logits_bin_path)

        gen = [int(torch.argmax(logits).item())]
        cur_pos = len(ids)
        for _ in range(23):
            dl = forward_bf16_step(cfg, tw, torch.tensor([[gen[-1]]], dtype=torch.long), torch.tensor([cur_pos], dtype=torch.long), kv_cache)
            gen.append(int(torch.argmax(dl).item()))
            cur_pos += 1
            if gen[-1] == cfg["eos_token_id"]:
                break

        results.append({
            "name": item["name"],
            "ids": ids,
            "prefill_s": round(prefill_s, 2),
            "top10_idx": torch.topk(logits, 10).indices.tolist(),
            "top10_val": [round(float(v), 4) for v in torch.topk(logits, 10).values],
            "greedy_ids": gen,
            "logits_file": str(logits_bin_path.relative_to(root)),
        })
        print(f"[BF16] {item['name']}: prefill={prefill_s:.1f}s | top1={gen[0]} | greedy={gen}")

    (root / "tests/fixtures/spark-4b-bf16-golden.json").write_text(json.dumps(results, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
