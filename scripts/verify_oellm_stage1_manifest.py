#!/usr/bin/env python3
"""Fail-closed gate before a Stage-1 GPU job reads the full data build."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--template", type=Path, required=True)
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--verify-file-hashes", action="store_true")
    args = parser.parse_args()
    report = json.loads(args.manifest.read_text(encoding="utf-8"))
    if report.get("kind") != "full_stage1_direct_audited":
        raise ValueError("Not a completed full Stage-1 build")
    config = report["config"]
    if config["repo"] != "openeurollm/OpenEuroLLM-Excellent-SFT":
        raise ValueError("Unexpected source repository")
    if config["revision"] != "c4ee55804f256c5f8cc4a6ef3fdecbe4a4ea95e0":
        raise ValueError("Unexpected source revision")
    if config["max_length"] != 4096:
        raise ValueError("Dataset length cap does not match first training stage")
    if config["template_sha256"] != sha256(args.template):
        raise ValueError("Training template differs from audited template")
    if config["tokenizer_json_sha256"] != sha256(args.tokenizer / "tokenizer.json"):
        raise ValueError("Training tokenizer differs from audited tokenizer")
    if report["counts"].get("source_rows") != 6_187_716 or report["counts"].get("direct") != 2_649_634:
        raise ValueError("Source/direct row counts differ from frozen corpus")
    if "zero <think>/</think> token IDs" not in report.get("invariant", ""):
        raise ValueError("Missing full-corpus no-think invariant")
    totals = {}
    for kind in ("train", "validation", "warmup"):
        files = report["files"].get(kind)
        if not files or len(files) != 62:
            raise ValueError(f"Missing {kind} files for 62 shards")
        totals[kind] = 0
        for info in files:
            path = Path(info["path"])
            if not path.is_file() or path.stat().st_size != info["bytes"]:
                raise ValueError(f"Missing or wrong-size {kind} file: {path}")
            if args.verify_file_hashes and sha256(path) != info["sha256"]:
                raise ValueError(f"SHA-256 mismatch for {path}")
            totals[kind] += info["rows"]
    if totals["train"] < 1_000_000 or totals["validation"] < 1000 or totals["warmup"] < 10_000:
        raise ValueError(f"Implausibly small admitted mixture: {totals}")
    if totals["train"] + totals["validation"] + report["counts"].get("overlength", 0) + report["counts"].get("render_rejected", 0) != report["counts"]["direct"]:
        raise ValueError("Direct row accounting does not balance")
    print(json.dumps({"status": "passed", "source_revision": config["revision"],
                      "max_length": config["max_length"], "files_hashed": args.verify_file_hashes,
                      "rows": totals}, indent=2), flush=True)


if __name__ == "__main__":
    main()
