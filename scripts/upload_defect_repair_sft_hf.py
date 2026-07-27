#!/usr/bin/env python3
"""Upload oellm-eu-defect-repair-sft-v1 to Hugging Face."""

from __future__ import annotations

import argparse
from pathlib import Path

from huggingface_hub import HfApi

DEFAULT_DATASET_DIR = Path("data/defect_repair/oellm-eu-defect-repair-sft-v1")
DEFAULT_REPO_ID = "birgermoell/oellm-eu-defect-repair-sft-v1"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-dir", type=Path, default=DEFAULT_DATASET_DIR)
    parser.add_argument("--repo-id", default=DEFAULT_REPO_ID)
    parser.add_argument("--private", action="store_true")
    args = parser.parse_args()

    api = HfApi()
    api.create_repo(repo_id=args.repo_id, repo_type="dataset", private=args.private, exist_ok=True)
    api.upload_folder(
        repo_id=args.repo_id,
        repo_type="dataset",
        folder_path=str(args.dataset_dir),
        commit_message="Upload European defect repair SFT dataset v0.1",
    )
    print(f"Uploaded dataset to https://huggingface.co/datasets/{args.repo_id}")


if __name__ == "__main__":
    main()
