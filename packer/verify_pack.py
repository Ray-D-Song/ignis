"""
Verify .ignis package integrity (SHA-256, headers, shapes) and run a pure CPU reference
forward pass using the exact dequantized .ignis weights to emit tiny-spark-quant-golden.json.
"""

from __future__ import annotations
import argparse
import hashlib
import json
import math
import struct
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from quant import IGNIS_MAGIC, PAYLOAD_OFFSET, dequantize_int4_g64_asym, dequantize_int8_row
from gen_tiny_fixture import apply_rotary_pos_emb, compute_rope_cos_sin


def read_ignis_package(pkg_dir: Path) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    manifest = json.loads((pkg_dir / "manifest.json").read_text(encoding="utf-8"))
    shard_bytes: dict[Any, bytes] = {}
    for s in manifest["shards"]:
        p = pkg_dir / s["file"]
        raw = p.read_bytes()
        assert len(raw) == s["bytes"], f"Size mismatch for {s['file']}"
        sha = hashlib.sha256(raw).hexdigest()
        assert sha == s["sha256"], f"SHA-256 mismatch for {s['file']}"
        magic, ver, sidx, poff, pbytes = struct.unpack("<8sIIQQ", raw[:32])
        assert magic == IGNIS_MAGIC, f"Bad magic in {s['file']}: {magic!r}"
        assert ver == 1 and poff == PAYLOAD_OFFSET and pbytes == s["payload_bytes"]
        shard_bytes[s["id"]] = raw

    tensors: dict[str, np.ndarray] = {}
    tmeta = manifest["tensors"]

    # Decode externalized embedding
    em = tmeta["token_embd"]
    ebuf = shard_bytes["embd"][em["offset"] : em["offset"] + em["bytes"]]
    tensors["token_embd"] = dequantize_int8_row(ebuf, em["shape"][0], em["shape"][1], em["row_stride"])

    # Decode GPU tensors
    seen_q4 = set()
    for name, info in tmeta.items():
        if name == "token_embd":
            continue
        if name.endswith(".quants"):
            base = name[: -len(".quants")]
            if base in seen_q4:
                continue
            seen_q4.add(base)
            s_info = tmeta[f"{base}.scales"]
            q_raw = shard_bytes[info["shard"]][info["offset"] : info["offset"] + info["bytes"]]
            s_raw = shard_bytes[s_info["shard"]][s_info["offset"] : s_info["offset"] + s_info["bytes"]]
            q_arr = np.frombuffer(q_raw, dtype=np.uint32).reshape(info["shape"])
            s_arr = np.frombuffer(s_raw, dtype=np.uint32).reshape(s_info["shape"])
            tensors[base] = dequantize_int4_g64_asym(q_arr, s_arr, info["orig_shape"][1])
        elif name.endswith(".scales"):
            continue
        else:
            raw = shard_bytes[info["shard"]][info["offset"] : info["offset"] + info["bytes"]]
            tensors[name] = np.frombuffer(raw, dtype=np.float32).reshape(info["shape"]).copy()

    return manifest, tensors


def rms_norm(x: torch.Tensor, w: torch.Tensor, eps: float) -> torch.Tensor:
    var = x.pow(2).mean(-1, keepdim=True)
    return w * x * torch.rsqrt(var + eps)


def forward_ignis_ref(
    bp: dict[str, Any],
    tw: dict[str, torch.Tensor],
    input_ids: torch.Tensor,
    positions: torch.Tensor,
    kv_cache: dict[int, tuple[torch.Tensor, torch.Tensor]],
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    traces: dict[str, torch.Tensor] = {}
    x = tw["token_embd"][input_ids]  # [B, T, D]
    traces["embed"] = x.clone()

    bsz, seq_len, _ = x.shape
    nh = bp["num_attention_heads"]
    nkv = bp["num_key_value_heads"]
    hd = bp["head_dim"]
    ng = nh // nkv
    q_dim = nh * hd
    kv_dim = nkv * hd
    eps = bp["rms_norm_eps"]
    scaling = 1.0 / math.sqrt(hd)

    for i in range(bp["num_layers"]):
        lt = bp["layer_types"][i]
        theta = bp["rope_theta_swa"] if lt == "swa" else bp["rope_theta_full"]
        rdim = bp["rope_dim_swa"] if lt == "swa" else bp["rope_dim_full"]
        prf = rdim / hd
        win = bp["sliding_window"] if lt == "swa" else None

        h = rms_norm(x, tw[f"blk.{i}.attn_norm"], eps)
        qkv = F.linear(h, tw[f"blk.{i}.attn_qkv"])
        q = qkv[..., :q_dim].view(bsz, seq_len, nh, hd).transpose(1, 2)
        k = qkv[..., q_dim : q_dim + kv_dim].view(bsz, seq_len, nkv, hd).transpose(1, 2)
        v = qkv[..., q_dim + kv_dim :].view(bsz, seq_len, nkv, hd).transpose(1, 2)

        cos, sin = compute_rope_cos_sin(positions, hd, theta, prf)
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

        if bp["has_attn_gate"]:
            gate = torch.sigmoid(F.linear(h, tw[f"blk.{i}.attn_gate"]))
            gate = gate.view(bsz, seq_len, nh, 1).transpose(1, 2)
            attn_out = attn_out * gate

        attn_out = attn_out.transpose(1, 2).contiguous().view(bsz, seq_len, q_dim)
        x = x + F.linear(attn_out, tw[f"blk.{i}.attn_output"])

        hf = rms_norm(x, tw[f"blk.{i}.ffn_norm"], eps)
        mlp_h = F.gelu(F.linear(hf, tw[f"blk.{i}.ffn_gate"])) * F.linear(hf, tw[f"blk.{i}.ffn_up"])
        x = x + F.linear(mlp_h, tw[f"blk.{i}.ffn_down"])
        traces[f"layer_{i}"] = x.clone()

    x = rms_norm(x, tw["output_norm"], eps)
    traces["final_norm"] = x.clone()
    logits = F.linear(x, tw["output"])
    return logits, traces


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pkg", type=Path, required=True)
    parser.add_argument("--golden", type=Path, required=True)
    parser.add_argument("--out-quant-golden", type=Path, required=True)
    args = parser.parse_args()

    manifest, np_tensors = read_ignis_package(args.pkg)
    tw = {k: torch.from_numpy(v).float() for k, v in np_tensors.items()}
    bp = manifest["blueprint"]
    fp_golden = json.loads(args.golden.read_text(encoding="utf-8"))

    prompt_ids = fp_golden["prompt_ids"]
    kv_cache: dict[int, tuple[torch.Tensor, torch.Tensor]] = {}
    with torch.no_grad():
        logits, pre_tr = forward_ignis_ref(
            bp, tw, torch.tensor([prompt_ids], dtype=torch.long), torch.arange(len(prompt_ids), dtype=torch.long), kv_cache
        )
        last_logits = logits[0, -1, :]
        next_tok = int(torch.argmax(last_logits).item())
        generated = [next_tok]
        decode_steps = []
        for step in range(5):
            pos = torch.tensor([len(prompt_ids) + step], dtype=torch.long)
            cin = torch.tensor([[generated[-1]]], dtype=torch.long)
            d_logits, d_tr = forward_ignis_ref(bp, tw, cin, pos, kv_cache)
            decode_steps.append({
                "pos": int(pos.item()),
                "token_in": generated[-1],
                "layer_0": d_tr["layer_0"][0, 0].tolist(),
                "layer_3": d_tr["layer_3"][0, 0].tolist(),
                "final_norm": d_tr["final_norm"][0, 0].tolist(),
                "logits_top8_idx": torch.topk(d_logits[0, 0], 8).indices.tolist(),
                "logits_top8_val": torch.topk(d_logits[0, 0], 8).values.tolist(),
            })
            generated.append(int(torch.argmax(d_logits[0, 0]).item()))

    def cos_sim(a: list[float], b: list[float]) -> float:
        va = np.array(a, dtype=np.float32)
        vb = np.array(b, dtype=np.float32)
        return float(np.dot(va, vb) / (np.linalg.norm(va) * np.linalg.norm(vb) + 1e-12))

    cos_pre = cos_sim(pre_tr["final_norm"][0, -1].tolist(), fp_golden["prefill_last"]["final_norm"])
    cos_dec0 = cos_sim(decode_steps[0]["final_norm"], fp_golden["decode_steps"][0]["final_norm"])
    print(f"[verify_pack] SHA-256 & headers OK | prefill final_norm cos={cos_pre:.6f} | decode0 final_norm cos={cos_dec0:.6f}")
    print(f"[verify_pack] FP16 greedy={fp_golden['generated_greedy_ids']} | Q4 .ignis greedy={generated}")

    quant_golden = {
        "prompt_ids": prompt_ids,
        "prefill_last": {
            "embed": pre_tr["embed"][0, -1].tolist(),
            "layer_0": pre_tr["layer_0"][0, -1].tolist(),
            "layer_3": pre_tr["layer_3"][0, -1].tolist(),
            "final_norm": pre_tr["final_norm"][0, -1].tolist(),
            "logits_top8_idx": torch.topk(last_logits, 8).indices.tolist(),
            "logits_top8_val": torch.topk(last_logits, 8).values.tolist(),
        },
        "generated_greedy_ids": generated,
        "decode_steps": decode_steps,
    }
    args.out_quant_golden.write_text(json.dumps(quant_golden, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
