#!/usr/bin/env python3
"""Upload a trained checkpoint to the HuggingFace Hub with a model card.

Uploads the model weights + tokenizer + config + README (model card), skipping training-only
artifacts (intermediate checkpoints, optimizer/trainer state). Run on a LUMI login node (the
compute nodes have no internet) after authenticating: set HF_TOKEN or `huggingface-cli login`.

Usage:
  python scripts/upload_model_hf.py \
    --ckpt /scratch/.../output/qwen35-9b-sft \
    --repo birgermoell/Qwen3.5-9B-EU-SFT \
    --card model_cards/qwen35-9b-eu-sft-README.md [--private]
"""
from __future__ import annotations

import argparse
import shutil
import tempfile
import pathlib

from huggingface_hub import HfApi, create_repo, upload_folder

# weights/tokenizer/config to ship; everything else (checkpoint-*/, *.bin, optimizer) is skipped
KEEP_SUFFIXES = (".safetensors", ".json", ".jinja", ".model", ".txt")
KEEP_NAMES = {"tokenizer.json", "tokenizer_config.json", "special_tokens_map.json",
              "config.json", "generation_config.json", "chat_template.jinja",
              "vocab.json", "merges.txt", "model.safetensors.index.json", "preprocessor_config.json"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--repo", required=True, help="e.g. birgermoell/Qwen3.5-9B-EU-SFT")
    ap.add_argument("--card", required=True, help="model card markdown -> uploaded as README.md")
    ap.add_argument("--private", action="store_true")
    a = ap.parse_args()

    ckpt = pathlib.Path(a.ckpt)
    with tempfile.TemporaryDirectory() as tmp:
        staging = pathlib.Path(tmp)
        # stage model files (top-level only; skip checkpoint-*/ and trainer state)
        for f in ckpt.iterdir():
            if f.is_dir():
                continue
            if f.name in KEEP_NAMES or (f.suffix in KEEP_SUFFIXES and not f.name.endswith(".bin")):
                shutil.copy2(f, staging / f.name)
        # model card as README.md
        shutil.copy2(a.card, staging / "README.md")

        staged = sorted(p.name for p in staging.iterdir())
        print(f"[upload] staging {len(staged)} files: {staged}")

        create_repo(a.repo, repo_type="model", private=a.private, exist_ok=True)
        upload_folder(folder_path=str(staging), repo_id=a.repo, repo_type="model",
                      commit_message="Upload Qwen3.5-9B EU-SFT (weights + tokenizer + model card)")
    print(f"[upload] done -> https://huggingface.co/{a.repo}")


if __name__ == "__main__":
    main()
