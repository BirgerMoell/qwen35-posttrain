#!/usr/bin/env python3
"""Stage only the pinned Excellent-SFT Parquet corpus and verify its metadata."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


REPO = "openeurollm/OpenEuroLLM-Excellent-SFT"
REVISION = "c4ee55804f256c5f8cc4a6ef3fdecbe4a4ea95e0"
EXPECTED_SHARDS = 62
EXPECTED_ROWS = 6_187_716
EXPECTED_BYTES = 27_609_764_587


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    import pyarrow.parquet as pq
    from huggingface_hub import HfApi, snapshot_download

    info = HfApi().dataset_info(REPO, revision=REVISION)
    if info.sha != REVISION:
        raise ValueError(f"dataset revision changed: {info.sha}")
    folder = Path(snapshot_download(
        repo_id=REPO, repo_type="dataset", revision=REVISION,
        local_dir=args.output, allow_patterns=["README.md", "data/*.parquet"],
        max_workers=8,
    ))
    shards = sorted((folder / "data").glob("train-*.parquet"))
    if len(shards) != EXPECTED_SHARDS:
        raise ValueError(f"expected {EXPECTED_SHARDS} shards; got {len(shards)}")
    total_bytes = sum(path.stat().st_size for path in shards)
    if total_bytes != EXPECTED_BYTES:
        raise ValueError(f"Parquet byte count {total_bytes} != {EXPECTED_BYTES}")
    total_rows = sum(pq.ParquetFile(path).metadata.num_rows for path in shards)
    if total_rows != EXPECTED_ROWS:
        raise ValueError(f"Parquet row count {total_rows} != {EXPECTED_ROWS}")
    print(json.dumps({"dataset": REPO, "revision": REVISION,
                      "directory": str(folder), "shards": len(shards),
                      "bytes": total_bytes, "rows": total_rows}, indent=2))


if __name__ == "__main__":
    main()
