#!/usr/bin/env python3
"""Stage pinned multilingual Instruct candidates on Leonardo; no training admission."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from huggingface_hub import HfApi, snapshot_download


SOURCES = (
    ("openeurollm/Dolci-Instruct-SFT-translated", "63202c34b23a4f155658a9e157e988902fb62279", "dolci-instruct-translated"),
    ("openeurollm/EU-Instruct-Synthetic", "c13be5d71144feb1007708aae2a8c2d823f8e59a", "eu-instruct-synthetic"),
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    args.output_root.mkdir(parents=True, exist_ok=True)
    api = HfApi()
    for repo, revision, name in SOURCES:
        info = api.dataset_info(repo, revision=revision, files_metadata=True)
        if info.sha != revision:
            raise RuntimeError(f"Unexpected revision for {repo}: {info.sha}")
        files = sorted((s.rfilename, s.size) for s in info.siblings if s.rfilename.endswith(".parquet"))
        if not files:
            raise RuntimeError(f"No Parquet files for {repo}")
        dest = args.output_root / name
        print(f"Staging {repo}@{revision}: {len(files)} Parquet files", flush=True)
        snapshot_download(repo_id=repo, repo_type="dataset", revision=revision,
                          allow_patterns=["*.parquet", "README.md"],
                          local_dir=dest, max_workers=4)
        missing = [path for path, size in files if not (dest / path).is_file() or (dest / path).stat().st_size != size]
        if missing:
            raise RuntimeError(f"Missing or wrong-size files for {repo}: {missing[:10]}")
        (dest / "staging_manifest.json").write_text(json.dumps({
            "repo": repo, "revision": revision, "kind": "staged_candidate_not_admitted",
            "files": [{"path": path, "bytes": size} for path, size in files],
        }, indent=2) + "\n", encoding="utf-8")
        print(f"Verified {len(files)} files at {dest}", flush=True)


if __name__ == "__main__":
    main()
