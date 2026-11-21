"""Generates PyTorch BF16 reference golden outputs for google/gemma-4-E4B-it (Text, Vision, Audio-In)."""

from __future__ import annotations
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from transformers import AutoModelForMultimodalLM, AutoProcessor

ROOT = Path(__file__).resolve().parent.parent
HF_DIR = ROOT / "tests" / "fixtures" / "gemma-4-e4b-it-hf"
OUT_DIR = ROOT / "tests" / "fixtures" / "gemma4-golden"


def save_f32(name: str, t: torch.Tensor | np.ndarray) -> None:
    arr = t.detach().cpu().float().numpy() if torch.is_tensor(t) else np.asarray(t, dtype=np.float32)
    (OUT_DIR / name).write_bytes(np.ascontiguousarray(arr).tobytes())


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"[golden-gemma4] Loading {HF_DIR} ...")
    proc = AutoProcessor.from_pretrained(str(HF_DIR))
    model = AutoModelForMultimodalLM.from_pretrained(
        str(HF_DIR),
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
    )
    model.eval()

    meta: dict = {}

    # 1. Text Golden
    messages_txt = [{"role": "user", "content": "What is the capital of France? Answer in one word."}]
    prompt_txt = proc.apply_chat_template(
        messages_txt, tokenize=False, add_generation_prompt=True, enable_thinking=False
    )
    txt_inputs = proc(text=prompt_txt, return_tensors="pt")
    txt_ids = txt_inputs["input_ids"][0].tolist()
    with torch.no_grad():
        out_txt = model(input_ids=txt_inputs["input_ids"], use_cache=False)
        gen_txt = model.generate(
            input_ids=txt_inputs["input_ids"],
            max_new_tokens=16,
            do_sample=False,
        )
    gen_ids_txt = gen_txt[0, len(txt_ids):].tolist()
    save_f32("text_logits_last.bin", out_txt.logits[0, -1])
    meta["text"] = {
        "prompt": prompt_txt,
        "input_ids": txt_ids,
        "greedy_ids": gen_ids_txt,
        "reply": proc.tokenizer.decode(gen_ids_txt, skip_special_tokens=True),
    }
    print("[golden-gemma4] Text reply:", repr(meta["text"]["reply"]))

    # 2. Vision Golden (96x96 bright red square on blue background -> resized to 768x768, 256 tokens)
    img_arr = np.zeros((96, 96, 3), dtype=np.uint8)
    img_arr[:, :, 2] = 240  # Blue background
    img_arr[24:72, 24:72, :] = [240, 20, 20]  # Red center square
    (OUT_DIR / "image_rgb_96.bin").write_bytes(img_arr.tobytes())
    pil_img = Image.fromarray(img_arr, mode="RGB")

    messages_vis = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": pil_img},
                {"type": "text", "text": "What colors do you see in this image? Reply in one short sentence."},
            ],
        }
    ]
    prompt_vis = proc.apply_chat_template(
        messages_vis, tokenize=False, add_generation_prompt=True, enable_thinking=False
    )
    vis_inputs = proc(text=prompt_vis, images=[pil_img], return_tensors="pt")
    non_pad = (vis_inputs["image_position_ids"][0] != -1).all(dim=-1)
    num_real_patches = int(non_pad.sum().item())
    with torch.no_grad():
        vis_feats = model.model.get_image_features(
            vis_inputs["pixel_values"].to(torch.bfloat16),
            vis_inputs["image_position_ids"],
        )
        img_soft = vis_feats.pooler_output[0]  # [256, 2560]
        out_vis = model(
            input_ids=vis_inputs["input_ids"],
            pixel_values=vis_inputs["pixel_values"].to(torch.bfloat16),
            image_position_ids=vis_inputs["image_position_ids"],
            use_cache=False,
        )
        gen_vis = model.generate(
            input_ids=vis_inputs["input_ids"],
            pixel_values=vis_inputs["pixel_values"].to(torch.bfloat16),
            image_position_ids=vis_inputs["image_position_ids"],
            max_new_tokens=20,
            do_sample=False,
        )
    vis_prompt_raw = proc.tokenizer(prompt_vis, add_special_tokens=False)["input_ids"]
    gen_ids_vis = gen_vis[0, vis_inputs["input_ids"].shape[1]:].tolist()
    save_f32("pixel_values_real_hf.bin", 2.0 * (vis_inputs["pixel_values"][0, :num_real_patches].float() - 0.5))
    save_f32("image_features_hf.bin", img_soft)
    save_f32("vqa_logits_last.bin", out_vis.logits[0, -1])
    meta["vision"] = {
        "width": 96,
        "height": 96,
        "num_real_patches": num_real_patches,
        "num_soft_tokens": int(img_soft.shape[0]),
        "raw_prompt": prompt_vis,
        "raw_prompt_ids": vis_prompt_raw,
        "expanded_prompt_ids": vis_inputs["input_ids"][0].tolist(),
        "greedy_ids": gen_ids_vis,
        "reply": proc.tokenizer.decode(gen_ids_vis, skip_special_tokens=True),
    }
    print("[golden-gemma4] VQA reply:", repr(meta["vision"]["reply"]))

    # 3. Audio-In Golden (1.0s 16 kHz tone)
    sr = 16000
    t = np.arange(sr, dtype=np.float32) / sr
    env = np.sin(np.pi * t) * (0.6 + 0.4 * np.sin(2.0 * np.pi * 4.0 * t))
    pcm = (0.35 * env * (np.sin(2.0 * np.pi * 220.0 * t) + 0.5 * np.sin(2.0 * np.pi * 440.0 * t))).astype(np.float32)
    save_f32("audio_16k.bin", pcm)

    messages_aud = [
        {
            "role": "user",
            "content": [
                {"type": "audio", "audio": pcm},
                {"type": "text", "text": "Briefly describe this sound."},
            ],
        }
    ]
    prompt_aud = proc.apply_chat_template(
        messages_aud, tokenize=False, add_generation_prompt=True, enable_thinking=False
    )
    aud_inputs = proc(text=prompt_aud, audio=[pcm], return_tensors="pt")
    with torch.no_grad():
        aud_out = model.model.get_audio_features(
            aud_inputs["input_features"].to(torch.bfloat16),
            aud_inputs["input_features_mask"],
        )
        # aud_out.pooler_output is [1, valid_or_t2, 2560]; attention_mask is True = valid
        valid_mask = aud_out.attention_mask[0]
        aud_feats_valid = aud_out.pooler_output[0][valid_mask]
        out_aud = model(
            input_ids=aud_inputs["input_ids"],
            input_features=aud_inputs["input_features"].to(torch.bfloat16),
            input_features_mask=aud_inputs["input_features_mask"],
            use_cache=False,
        )
    aud_prompt_raw = proc.tokenizer(prompt_aud, add_special_tokens=False)["input_ids"]
    save_f32("mel_hf.bin", aud_inputs["input_features"][0])
    save_f32("audio_features_hf.bin", aud_feats_valid)
    save_f32("audio_in_logits_last.bin", out_aud.logits[0, -1])
    meta["audio_in"] = {
        "num_samples": int(len(pcm)),
        "mel_shape": list(aud_inputs["input_features"][0].shape),
        "valid_mel_frames": int(aud_inputs["input_features_mask"][0].sum().item()),
        "valid_audio_tokens": int(aud_feats_valid.shape[0]),
        "raw_prompt": prompt_aud,
        "raw_prompt_ids": aud_prompt_raw,
        "expanded_prompt_ids": aud_inputs["input_ids"][0].tolist(),
        "top5_logits": torch.topk(out_aud.logits[0, -1].float(), 5).indices.tolist(),
    }
    print("[golden-gemma4] Audio feats:", list(aud_feats_valid.shape), "top5:", meta["audio_in"]["top5_logits"])

    (OUT_DIR / "golden.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"[golden-gemma4] Wrote {OUT_DIR / 'golden.json'}")


if __name__ == "__main__":
    main()
