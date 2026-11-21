"""Generates PyTorch BF16 reference golden outputs for LFM3-Omni-2B (Text, Vision, Audio-In, Speech-Out)."""

from __future__ import annotations
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from transformers import AutoModelForMultimodalLM, AutoProcessor

ROOT = Path(__file__).resolve().parent.parent
HF_DIR = ROOT / "tests" / "fixtures" / "lfm3-omni-2b-hf"
OUT_DIR = ROOT / "tests" / "fixtures" / "lfm3-omni-golden"


def save_f32(name: str, t: torch.Tensor | np.ndarray) -> None:
    arr = t.detach().cpu().float().numpy() if torch.is_tensor(t) else np.asarray(t, dtype=np.float32)
    (OUT_DIR / name).write_bytes(np.ascontiguousarray(arr).tobytes())


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"[golden] Loading {HF_DIR} ...")
    proc = AutoProcessor.from_pretrained(str(HF_DIR), trust_remote_code=True)
    model = AutoModelForMultimodalLM.from_pretrained(str(HF_DIR), torch_dtype=torch.float32, trust_remote_code=True)
    model.eval()
    codec = proc.audio_tokenizer
    codec.eval()

    meta: dict = {}

    # 1. Text Golden
    messages_txt = [{"role": "user", "content": "What is the capital of France? Answer concisely."}]
    prompt_txt = proc.apply_chat_template(messages_txt, tokenize=False, add_generation_prompt=True)
    txt_ids = proc.tokenizer(prompt_txt, add_special_tokens=False)["input_ids"]
    with torch.no_grad():
        out_txt = model(input_ids=torch.tensor([txt_ids], dtype=torch.long), use_cache=False)
        gen_txt = model.generate(
            torch.tensor([txt_ids], dtype=torch.long),
            generation_mode="text",
            max_new_tokens=16,
            do_sample=False,
        )
    gen_ids_txt = gen_txt.sequences[0, len(txt_ids):].tolist()
    save_f32("text_logits_last.bin", out_txt.logits[0, -1])
    meta["text"] = {
        "prompt": prompt_txt,
        "input_ids": txt_ids,
        "greedy_ids": gen_ids_txt,
        "reply": proc.tokenizer.decode(gen_ids_txt, skip_special_tokens=True),
    }
    print("[golden] Text reply:", repr(meta["text"]["reply"]))

    # 2. Vision Golden (256x256 bright red square on solid blue background)
    img_arr = np.zeros((256, 256, 3), dtype=np.uint8)
    img_arr[:, :, 2] = 240  # Blue background
    img_arr[64:192, 64:192, :] = [240, 20, 20]  # Red center square
    (OUT_DIR / "image_rgb_256.bin").write_bytes(img_arr.tobytes())
    pil_img = Image.fromarray(img_arr, mode="RGB")

    messages_vis = [
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": "What colors do you see in this image? Reply in one short sentence."},
            ],
        }
    ]
    prompt_vis = proc.apply_chat_template(messages_vis, tokenize=False, add_generation_prompt=True)
    vis_inputs = proc(text=prompt_vis, images=pil_img, return_tensors="pt")
    with torch.no_grad():
        vis_feats = model.get_image_features(vis_inputs["pixel_values"], vis_inputs["image_grid_thw"])
        out_vis = model(
            input_ids=vis_inputs["input_ids"],
            pixel_values=vis_inputs["pixel_values"],
            image_grid_thw=vis_inputs["image_grid_thw"],
            use_cache=False,
        )
        gen_vis = model.generate(
            vis_inputs["input_ids"],
            pixel_values=vis_inputs["pixel_values"],
            image_grid_thw=vis_inputs["image_grid_thw"],
            generation_mode="text",
            max_new_tokens=20,
            do_sample=False,
        )
    vis_prompt_raw = proc.tokenizer(prompt_vis, add_special_tokens=False)["input_ids"]
    gen_ids_vis = gen_vis.sequences[0, vis_inputs["input_ids"].shape[1]:].tolist()
    save_f32("pixel_values_hf.bin", vis_inputs["pixel_values"])
    save_f32("image_features_hf.bin", vis_feats)
    save_f32("vqa_logits_last.bin", out_vis.logits[0, -1])
    meta["vision"] = {
        "width": 256,
        "height": 256,
        "grid_thw": vis_inputs["image_grid_thw"][0].tolist(),
        "raw_prompt_ids": vis_prompt_raw,
        "expanded_prompt_ids": vis_inputs["input_ids"][0].tolist(),
        "greedy_ids": gen_ids_vis,
        "reply": proc.tokenizer.decode(gen_ids_vis, skip_special_tokens=True),
    }
    print("[golden] VQA reply:", repr(meta["vision"]["reply"]))

    # 3. Audio-In Golden (1.5s 16 kHz synthetic speech-like chord / tone)
    sr = 16000
    t = np.arange(int(1.5 * sr), dtype=np.float32) / sr
    env = np.sin(np.pi * t / 1.5) * (0.6 + 0.4 * np.sin(2.0 * np.pi * 4.0 * t))
    pcm = (0.35 * env * (np.sin(2.0 * np.pi * 220.0 * t) + 0.5 * np.sin(2.0 * np.pi * 440.0 * t) + 0.25 * np.sin(2.0 * np.pi * 880.0 * t))).astype(np.float32)
    save_f32("audio_16k.bin", pcm)

    messages_aud = [
        {
            "role": "user",
            "content": [
                {"type": "audio"},
                {"type": "text", "text": "Briefly describe this audio clip."},
            ],
        }
    ]
    prompt_aud = proc.apply_chat_template(messages_aud, tokenize=False, add_generation_prompt=True)
    aud_inputs = proc(text=prompt_aud, audio=pcm, sampling_rate=sr, return_tensors="pt")
    with torch.no_grad():
        aud_feats = model.get_audio_features(aud_inputs["input_features"], aud_inputs["feature_attention_mask"])
        out_aud = model(
            input_ids=aud_inputs["input_ids"],
            input_features=aud_inputs["input_features"],
            feature_attention_mask=aud_inputs["feature_attention_mask"],
            use_cache=False,
        )
    aud_prompt_raw = proc.tokenizer(prompt_aud, add_special_tokens=False)["input_ids"]
    save_f32("mel_hf.bin", aud_inputs["input_features"][0])
    save_f32("audio_features_hf.bin", aud_feats)
    save_f32("audio_in_logits_last.bin", out_aud.logits[0, -1])
    meta["audio_in"] = {
        "num_samples": int(len(pcm)),
        "mel_shape": list(aud_inputs["input_features"][0].shape),
        "raw_prompt_ids": aud_prompt_raw,
        "expanded_prompt_ids": aud_inputs["input_ids"][0].tolist(),
        "top5_logits": torch.topk(out_aud.logits[0, -1], 5).indices.tolist(),
    }
    print("[golden] Audio feats:", list(aud_feats.shape), "top5:", meta["audio_in"]["top5_logits"])

    # 4. Speech-Out Golden (Codebook Embeds + DepthDecoder + LiquidCodec)
    messages_sp = [{"role": "user", "content": "Say hello."}]
    prompt_sp = proc.apply_chat_template(messages_sp, tokenize=False, add_generation_prompt=True)
    sp_ids = proc.tokenizer(prompt_sp, add_special_tokens=False)["input_ids"] + [model.config.audio_start_token_id]
    with torch.no_grad():
        out_sp = model(input_ids=torch.tensor([sp_ids], dtype=torch.long), output_hidden_states=False, use_cache=False)
        # model.model(...) returns last_hidden_state directly; model() returns logits!
        m_out = model.model(input_ids=torch.tensor([sp_ids], dtype=torch.long), use_cache=False)
        h_last = m_out.last_hidden_state[:, -1]  # [1, 2048]
        # Deterministic pass-0 MaskGIT over all 12 slots masked
        codes_zero = torch.zeros((1, 12), dtype=torch.long)
        masked_all = torch.ones((1, 12), dtype=torch.bool)
        depth_logits_0 = model.depth_decoder(h_last, codes_zero, masked_all)  # [1, 12, 2017]
        # Test greedy step-by-step MaskGIT [4,4,4] with fixed permutation [0..11]
        g_codes = torch.zeros((1, 12), dtype=torch.long)
        g_masked = torch.ones((1, 12), dtype=torch.bool)
        for chunk in [[0, 1, 2, 3], [4, 5, 6, 7], [8, 9, 10, 11]]:
            dl = model.depth_decoder(h_last, g_codes, g_masked)
            for idx in chunk:
                g_codes[0, idx] = dl[0, idx].argmax(-1)
                g_masked[0, idx] = False
        cb_emb = model.model.get_audio_out_embeds(g_codes)  # [1, 2048]

        # Test LiquidCodec decode on 8 synthetic non-EOS speech frames
        syn_codes = torch.tensor([[(f * 97 + q * 131) % 1000 for f in range(8)] for q in range(12)], dtype=torch.long).unsqueeze(0)
        wav_24k = codec.decode(syn_codes, return_dict=True).audio_values[0]

    save_f32("speech_hidden_last.bin", h_last[0])
    save_f32("depth_pass0_cb0_logits.bin", depth_logits_0[0, 0])
    save_f32("codebook_emb_frame0.bin", cb_emb[0])
    save_f32("codec_wav_24k.bin", wav_24k)
    meta["speech_out"] = {
        "prompt_ids": sp_ids,
        "greedy_frame0": g_codes[0].tolist(),
        "syn_codes_frame_major": syn_codes[0].T.tolist(),
        "wav_len": int(wav_24k.shape[0]),
    }
    print("[golden] Greedy depth frame0:", meta["speech_out"]["greedy_frame0"], "wav len:", meta["speech_out"]["wav_len"])

    (OUT_DIR / "golden.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"[golden] Saved all golden assets to {OUT_DIR}")


if __name__ == "__main__":
    main()
