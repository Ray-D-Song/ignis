"""Resumable downloader for real HuggingFace multimodal checkpoints used in Ignis Phase 5+6."""

from __future__ import annotations
import argparse
import os
from pathlib import Path
from huggingface_hub import snapshot_download

ROOT = Path(__file__).resolve().parent.parent

MODELS = {
    "lfm3": {
        "repo_id": "LiquidAI/LFM3-Omni-2B",
        "local_dir": ROOT / "tests" / "fixtures" / "lfm3-omni-2b-hf",
        "ignore_patterns": ["assets/*", "*.png", "*.jpg", "*.md"],
    },
    "gemma4": {
        "repo_id": "google/gemma-4-E4B-it",
        "local_dir": ROOT / "tests" / "fixtures" / "gemma-4-e4b-it-hf",
        "ignore_patterns": ["*.md"],
    },
}


def download_one(key: str) -> None:
    cfg = MODELS[key]
    out_dir: Path = cfg["local_dir"]
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[download_hf] Starting/resuming {cfg['repo_id']} -> {out_dir}", flush=True)
    path = snapshot_download(
        repo_id=cfg["repo_id"],
        local_dir=str(out_dir),
        ignore_patterns=cfg["ignore_patterns"],
        max_workers=4,
    )
    total = sum(p.stat().st_size for p in Path(path).rglob("*") if p.is_file())
    print(f"[download_hf] Done {cfg['repo_id']}: {total / (1024**3):.2f} GiB", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--targets", nargs="+", choices=list(MODELS.keys()), default=["lfm3", "gemma4"])
    args = parser.parse_args()
    os.environ["HF_HUB_DISABLE_XET"] = "1"
    os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "0")
    for t in args.targets:
        download_one(t)


if __name__ == "__main__":
    main()
