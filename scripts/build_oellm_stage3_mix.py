#!/usr/bin/env python3
"""Build the pre-tokenized Stage-3 (tool-use) SFT mix: tool pool + Stage-2 replay.

Tool rows come from stage3-tool-pool-v1 (rendered with template v2.1), selected per source up to a
rendered-token budget; 0.5 % of tool rows (by id hash) are held out as validation. Replay rows come
from the Stage-2 mix (template v2): tool-free conversations render with identical token IDs and masks
under v2 and v2.1 (tests/test_chat_template_v2_1.py and a 2,843-row real-data check), so their tokens
are reused unchanged. Identity and safety rows are always kept. Same columns as earlier mixes.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

from build_oellm_stage1_mix import SCHEMA

TOOL_BUDGET = {  # rendered tokens; None = keep all
    "toucan": 70e6, "nemotron_tool_calling": 50e6, "toolmind_graphsyn": 25e6,
}
REPLAY_BUDGET = {"reasoning": 85e6, "direct": 80e6}
ALWAYS_KEEP = ("replay:identity", "dolci32b:WildGuardMix", "dolci32b:WildJailbreak", "dolci32b:CoCoNot")
# Stage-2 eval showed long, non-converging thinking; replay only short, decisive traces in Stage 3.
MAX_REPLAY_REASONING_TOKENS = 8192


def h(x: str) -> float:
    return int(hashlib.sha256(x.encode()).hexdigest()[:16], 16) / 2**64


def lengths(t: pa.Table) -> list[int]:
    return pc.list_value_length(t["input_ids"]).to_pylist()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tool-pool", type=Path, required=True)
    ap.add_argument("--stage2-mix", type=Path, required=True)
    ap.add_argument("--template", type=Path, required=True, help="template v2.1 used for training")
    ap.add_argument("--tokenizer", type=Path, required=True)
    ap.add_argument("--output-dir", type=Path, required=True)
    a = ap.parse_args()
    a.output_dir.mkdir(parents=True, exist_ok=True)
    stats: dict[str, Counter] = {}

    # --- tool pool: per-source keep probability from measured token totals
    tool_files = sorted(a.tool_pool.glob("train-*.parquet"))
    totals = Counter()
    for f in tool_files:
        t = pq.read_table(f, columns=["source", "input_ids"])
        for s, n in zip(t["source"].to_pylist(), lengths(t)):
            totals[s.replace("+translated", "")] += n
    keep_p = {s: min(1.0, TOOL_BUDGET[s] / totals[s]) if s in TOOL_BUDGET else 1.0 for s in totals}
    train_parts, val_parts = [], []
    for f in tool_files:
        t = pq.read_table(f)
        ids, srcs, lens = t["id"].to_pylist(), t["source"].to_pylist(), lengths(t)
        is_val = [h(i + "#val") < 0.005 for i in ids]
        keep = [(not v) and h(i) < keep_p[s.replace("+translated", "")] for i, s, v in zip(ids, srcs, is_val)]
        for split, mask in (("train", keep), ("validation", is_val)):
            part = t.filter(pa.array(mask)).select(SCHEMA.names).cast(SCHEMA)
            (train_parts if split == "train" else val_parts).append(part)
            c = stats.setdefault(f"{split}:tool", Counter())
            c["rows"] += part.num_rows
            c["tokens"] += sum(n for n, m in zip(lens, mask) if m)

    # --- Stage-2 replay: reasoning vs direct groups, identity + safety always kept
    s2_files = sorted(a.stage2_mix.glob("train-*.parquet"))
    group_tokens = Counter()
    for f in s2_files:
        t = pq.read_table(f, columns=["source", "input_ids"])
        for s, n in zip(t["source"].to_pylist(), lengths(t)):
            if s not in ALWAYS_KEEP:
                g = "direct" if s.startswith("replay:") or s.endswith(":direct") else "reasoning"
                if g == "direct" or n <= MAX_REPLAY_REASONING_TOKENS:
                    group_tokens[g] += n
    replay_p = {g: min(1.0, REPLAY_BUDGET[g] / group_tokens[g]) for g in REPLAY_BUDGET}
    for f in s2_files:
        t = pq.read_table(f)
        srcs, ids, lens = t["source"].to_pylist(), t["id"].to_pylist(), lengths(t)
        groups = ["always" if s in ALWAYS_KEEP else "direct" if s.startswith("replay:") or s.endswith(":direct") else "reasoning"
                  for s in srcs]
        keep = [g == "always" or ((g == "direct" or n <= MAX_REPLAY_REASONING_TOKENS) and h(i + "#s3") < replay_p[g])
                for i, g, n in zip(ids, groups, lens)]
        part = t.filter(pa.array(keep)).select(SCHEMA.names).cast(SCHEMA)
        train_parts.append(part)
        for g, n, k in zip(groups, lens, keep):
            if k:
                c = stats.setdefault(f"train:replay_{g}", Counter())
                c["rows"] += 1
                c["tokens"] += n
    for f in sorted(a.stage2_mix.glob("validation-*.parquet")):
        val_parts.append(pq.read_table(f).select(SCHEMA.names).cast(SCHEMA))

    # --- write shards (~200 MB each) and manifest
    files = []
    for split, parts in (("train", train_parts), ("validation", val_parts)):
        table = pa.concat_tables([p for p in parts if p.num_rows])
        ids = table["id"].to_pylist()
        table = table.take(sorted(range(table.num_rows), key=lambda i: h(ids[i] + "#order")))  # deterministic shuffle
        n_shards = max(1, table.num_rows // 20000)
        for k in range(n_shards):
            shard = table.slice(k * table.num_rows // n_shards, (k + 1) * table.num_rows // n_shards - k * table.num_rows // n_shards)
            path = a.output_dir / f"{split}-{k:04d}.parquet"
            pq.write_table(shard, path, compression="zstd")
            files.append({"path": str(path), "rows": shard.num_rows})
    sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
    manifest = {"kind": "stage3_tool_mix_pretokenized", "version": "v1",
                "config": {"template": str(a.template), "template_sha256": sha(a.template),
                           "tokenizer_json_sha256": sha(a.tokenizer / "tokenizer.json"),
                           "tool_budget": TOOL_BUDGET, "tool_keep_probability": keep_p,
                           "replay_budget": REPLAY_BUDGET, "replay_keep_probability": replay_p, "always_keep": ALWAYS_KEEP,
                           "max_replay_reasoning_tokens": MAX_REPLAY_REASONING_TOKENS,
                           "note": "replay tokens from the v2 template are identical under v2.1 for tool-free rows"},
                "totals": {k: dict(v) for k, v in sorted(stats.items())}, "files": files}
    (a.output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest["totals"], indent=1))


if __name__ == "__main__":
    main()
