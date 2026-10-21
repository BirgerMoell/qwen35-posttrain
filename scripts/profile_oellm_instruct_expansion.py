#!/usr/bin/env python3
"""Structural profile and reproducible review samples for large SFT additions.

This is an admission input, not a data builder or a quality certificate.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import gc
import hashlib
import heapq
import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from audit_two_stage_sft import source_kind


def sample_key(source: str, language: str, identifier: str) -> int:
    return int.from_bytes(hashlib.blake2b(
        f"{source}:{language}:{identifier}".encode(), digest_size=8).digest(), "big")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--samples-per-language", type=int, default=12)
    args = parser.parse_args()
    sources = {"dolci-instruct-translated": "openeurollm/Dolci-Instruct-SFT-translated",
               "eu-instruct-synthetic": "openeurollm/EU-Instruct-Synthetic"}
    results = {}
    for directory, repo in sources.items():
        root = args.source_root / directory
        manifest = json.loads((root / "staging_manifest.json").read_text(encoding="utf-8"))
        paths = [root / item["path"] for item in manifest["files"]]
        if not all(path.is_file() for path in paths):
            raise FileNotFoundError(f"Missing Parquet shard under {root}")
        counts = Counter()
        by_language = defaultdict(Counter)
        samples = defaultdict(list)
        for path in paths:
            language_from_path = path.relative_to(root).parts[0]
            parquet = pq.ParquetFile(path)
            columns = [column for column in ("id", "messages", "language") if column in parquet.schema_arrow.names]
            if "messages" not in columns:
                raise ValueError(f"No messages column in {path}")
            for batch in parquet.iter_batches(batch_size=512, columns=columns):
                for row in batch.to_pylist():
                    language = row.get("language") or language_from_path
                    if language != language_from_path:
                        counts["language_path_disagreement"] += 1
                    bucket = by_language[language]
                    counts["rows"] += 1
                    bucket["rows"] += 1
                    messages = row.get("messages")
                    try:
                        kind = source_kind(messages)
                    except (KeyError, TypeError, ValueError):
                        kind = "malformed_or_unsupported"
                    counts[kind] += 1
                    bucket[kind] += 1
                    if kind != "direct":
                        continue
                    identifier = str(row.get("id") or f"{path.name}:{counts['rows']}")
                    score = sample_key(repo, language, identifier)
                    snippet = {"id": identifier, "language": language,
                               "messages": messages}
                    heap = samples[language]
                    entry = (-score, identifier, counts["rows"], snippet)
                    if len(heap) < args.samples_per_language:
                        heapq.heappush(heap, entry)
                    elif score < -heap[0][0]:
                        heapq.heapreplace(heap, entry)
            del batch, parquet
            gc.collect()
            pa.default_memory_pool().release_unused()
        results[repo] = {
            "revision": manifest["revision"], "parquet_files": len(paths),
            "counts": dict(counts),
            "by_language": {lang: dict(counter) for lang, counter in sorted(by_language.items())},
            "review_samples": {lang: [item for _, _, _, item in sorted(heap, key=lambda value: (-value[0], value[1]))]
                               for lang, heap in sorted(samples.items())},
        }
        print(f"{repo}: {dict(counts)}", flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(results, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
