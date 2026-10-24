#!/usr/bin/env python3
"""Stage-4 DPO v2 data: a stratified subset of the pre-tokenized v1 pairs plus the format-compliance pairs.

The full v1 set (1,441,722 pairs) costs ~19K local-h at the measured 25.3 s/step, so v2 trains one epoch on ~250K
pairs. Selection is deterministic (sha256 of the pair id with a salt, lowest first), shard by shard, so memory stays
at one shard:
- every format-compliance pair (`--format`, pre-tokenized by build_oellm_format_pairs + the main pretok guards);
- every HelpSteer3 pair (human preferences, incl. code and many languages);
- `--ultrafeedback` UltraFeedback pairs;
- `--dolci-per-language` Dolci-Instruct-DPO-translated pairs per language (each language's prompts are different
  items, not translations of the same row, so no cross-language duplicates arise).
Both inputs must carry the same template / tokenizer / max_length manifest; the output manifest keeps them so
dpo_train_pretok.check_manifest accepts it. Validation = the v1 validation shard plus the format validation shard.
Runs on a login node with the container's python (pyarrow).
"""
import argparse
import collections
import glob
import hashlib
import json
from pathlib import Path

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

DOLCI = "openeurollm/Dolci-Instruct-DPO-translated"
HELPSTEER = "nvidia/HelpSteer3"
ULTRAFEEDBACK = "argilla/ultrafeedback-binarized-preferences-cleaned"
SALT = "#dpo-v2-250k"
COMPAT_KEYS = ("kind", "max_length", "template_sha256", "tokenizer_json_sha256", "dropped_tag_ids")


def rank(pair_id: str) -> str:
    return hashlib.sha256((pair_id + SALT).encode()).hexdigest()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--main", type=Path, required=True, help="pretok-v1 directory")
    ap.add_argument("--format", type=Path, required=True, help="format-v1/pretok directory")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--ultrafeedback", type=int, default=30000)
    ap.add_argument("--dolci-per-language", type=int, default=14691)
    ap.add_argument("--rows-per-shard", type=int, default=50000)
    a = ap.parse_args()

    main_manifest = json.loads((a.main / "manifest.json").read_text())
    format_manifest = json.loads((a.format / "manifest.json").read_text())
    for key in COMPAT_KEYS:
        if main_manifest[key] != format_manifest[key]:
            raise SystemExit(f"manifests differ on {key}: {main_manifest[key]!r} vs {format_manifest[key]!r}")

    # Pass 1: choose ids from the metadata columns only.
    shards = sorted(glob.glob(str(a.main / "train-*.parquet")))
    candidates = collections.defaultdict(list)
    for shard in shards:
        meta = pq.read_table(shard, columns=["id", "source", "language"]).to_pydict()
        for pair_id, source, language in zip(meta["id"], meta["source"], meta["language"]):
            if source == DOLCI:
                candidates[(source, language)].append(pair_id)
            elif source in (HELPSTEER, ULTRAFEEDBACK):
                candidates[(source, "*")].append(pair_id)
            else:
                raise SystemExit(f"unexpected source {source}")
    quota = {}
    for key, ids in candidates.items():
        source = key[0]
        want = len(ids) if source == HELPSTEER else a.ultrafeedback if source == ULTRAFEEDBACK else a.dolci_per_language
        if want > len(ids):
            raise SystemExit(f"{key}: want {want}, only {len(ids)} available")
        quota[key] = want
    chosen = set()
    for key, ids in candidates.items():
        chosen.update(sorted(ids, key=rank)[: quota[key]])

    # Pass 2: copy the chosen rows, shard by shard, then the format pairs.
    a.out.mkdir(parents=True, exist_ok=True)
    counts = collections.Counter()
    buffer, written = [], 0

    def flush(final: bool = False) -> None:
        """Write full shards from the buffer (and the remainder when final)."""
        nonlocal buffer, written
        if not buffer:
            return
        table = pa.concat_tables(buffer).combine_chunks()
        offset = 0
        while table.num_rows - offset >= a.rows_per_shard or (final and offset < table.num_rows):
            pq.write_table(table.slice(offset, a.rows_per_shard), a.out / f"train-{written:04d}.parquet")
            written += 1
            offset += a.rows_per_shard
        buffer = [table.slice(offset)] if offset < table.num_rows else []

    schema = None
    for shard in shards:
        table = pq.read_table(shard)
        schema = schema or table.schema
        keep = table.filter(pc.is_in(table["id"], value_set=pa.array(sorted(chosen.intersection(table["id"].to_pylist())))))
        for source, language in zip(keep["source"].to_pylist(), keep["language"].to_pylist()):
            counts[f"{source}:{language}"] += 1
        buffer.append(keep)
        flush()
    for shard in sorted(glob.glob(str(a.format / "train-*.parquet"))):
        table = pq.read_table(shard).select(schema.names).cast(schema)
        for source in table["source"].to_pylist():
            counts[f"format:{source}"] += 1
        buffer.append(table)
    flush(final=True)

    validation = []
    for directory in (a.main, a.format):
        for shard in sorted(glob.glob(str(directory / "validation-*.parquet"))):
            validation.append(pq.read_table(shard).select(schema.names).cast(schema))
    pq.write_table(pa.concat_tables(validation).combine_chunks(), a.out / "validation-0000.parquet")

    total = sum(counts.values())
    manifest = {key: main_manifest[key] for key in main_manifest if key != "splits"}
    manifest["subset"] = {
        "builder": "scripts/build_oellm_dpo_subset.py", "salt": SALT, "main": str(a.main), "format": str(a.format),
        "quota": {f"{s}:{l}": q for (s, l), q in sorted(quota.items())}, "train_pairs": total,
        "train_shards": written, "validation_pairs": sum(t.num_rows for t in validation),
        "counts": dict(sorted(counts.items())),
    }
    (a.out / "manifest.json").write_text(json.dumps(manifest, indent=1) + "\n")
    print(json.dumps(manifest["subset"], indent=1))


if __name__ == "__main__":
    main()
