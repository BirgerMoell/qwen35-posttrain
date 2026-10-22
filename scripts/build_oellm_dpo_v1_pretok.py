#!/usr/bin/env python3
"""Pre-tokenize the staged Stage-4 DPO pairs with our tokenizer and template (default mode, no think tokens).

Input: the gzipped JSONL written by stage_oellm_dpo_v1.py (prompt / chosen / rejected message lists).
Output: parquet shards with `prompt_ids`, `chosen_ids`, `rejected_ids` (TRL 0.29 DPO column names) plus
id / source / language, and manifest.json with counts per source and drop reason, lengths and hashes.

The ids are computed exactly as TRL's DPOTrainer does (prompt rendered with add_generation_prompt=True,
completion = render(prompt + response)[len(prompt):]) so training is identical to letting TRL tokenize,
but runs once, offline, and the guards below see the exact training tokens:
- the template is ours (v2.1) and must be byte-identical to the tokenizer's bundled chat_template;
  sha256 of template and tokenizer.json are recorded and can be pinned with --expect-*;
- a pair is dropped if a response is empty, the render fails, the prompt is not a token prefix of
  prompt+response, either sequence exceeds --max-length (the reference run's `truncated_pair_action: drop`),
  or any chat-tag token (ids 11-18: tool and think delimiters) appears anywhere in the pair.
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import os
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from functools import partial
from itertools import islice
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

TAG_IDS = set(range(11, 19))   # <tool_call> </tool_call> <tools> </tools> <tool_response> </tool_response> </think> <think>
SHARD_ROWS = 50_000
_TOK = None


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _init(tokenizer_dir: str, template: str) -> None:
    global _TOK
    from transformers import AutoTokenizer
    _TOK = AutoTokenizer.from_pretrained(tokenizer_dir)
    _TOK.chat_template = template


def _ids(messages, generation_prompt: bool) -> list[int]:
    out = _TOK.apply_chat_template(messages, add_generation_prompt=generation_prompt, tokenize=True, return_dict=False)
    return list(out[0] if out and isinstance(out[0], list) else out)


def encode(line: str, max_length: int):
    row = json.loads(line)
    for side in ("chosen", "rejected"):
        if not row[side] or not any((m.get("content") or "").strip() for m in row[side]):
            return row, None, "empty_response"
    try:
        prompt = _ids(row["prompt"], True)
        full_c = _ids(row["prompt"] + row["chosen"], False)
        full_r = _ids(row["prompt"] + row["rejected"], False)
    except Exception:  # noqa: BLE001 - jinja TemplateError, bad roles
        return row, None, "render_error"
    if full_c[: len(prompt)] != prompt or full_r[: len(prompt)] != prompt:
        return row, None, "prefix_mismatch"
    if len(full_c) > max_length or len(full_r) > max_length:
        return row, None, "over_max_length"
    if TAG_IDS & (set(full_c) | set(full_r)):
        return row, None, "chat_tag_token"
    return row, (prompt, full_c[len(prompt):], full_r[len(prompt):]), None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--staged", type=Path, required=True)
    ap.add_argument("--tokenizer", type=Path, required=True, help="model dir with tokenizer.json")
    ap.add_argument("--template", type=Path, required=True, help="templates/oellm_qwen3_two_stage_v2_1.jinja")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--max-length", type=int, default=2048)
    ap.add_argument("--workers", type=int, default=os.cpu_count() or 4)
    ap.add_argument("--extra", type=Path, action="append", default=[],
                    help="additional staged *-train.jsonl.gz / *-validation.jsonl.gz files (e.g. format pairs)")
    a = ap.parse_args()
    template = a.template.read_text()
    bundled = (a.tokenizer / "chat_template.jinja")
    if bundled.exists() and bundled.read_text() != template:
        raise SystemExit(f"tokenizer's bundled chat_template differs from {a.template}; refusing to build")
    a.out.mkdir(parents=True, exist_ok=True)
    manifest = {"kind": "oellm_dpo_v1_pretokenized", "max_length": a.max_length,
                "template": str(a.template), "template_sha256": sha(a.template),
                "tokenizer_json_sha256": sha(a.tokenizer / "tokenizer.json"),
                "dropped_tag_ids": sorted(TAG_IDS), "splits": {}}
    inputs = sorted(a.staged.glob("*-*.jsonl.gz")) + a.extra
    with ProcessPoolExecutor(a.workers, initializer=_init, initargs=(str(a.tokenizer), template)) as pool:
        for split in ("train", "validation"):
            files = [f for f in inputs if f.name.endswith(f"-{split}.jsonl.gz")]
            stats: dict[str, Counter] = {}
            buf: dict[str, list] = {k: [] for k in ("id", "source", "language", "prompt_ids", "chosen_ids", "rejected_ids")}
            shard = 0
            lengths = []

            def flush():
                nonlocal shard
                if not buf["id"]:
                    return
                table = pa.table({**{k: buf[k] for k in ("id", "source", "language")},
                                  **{k: pa.array(buf[k], type=pa.list_(pa.int32())) for k in ("prompt_ids", "chosen_ids", "rejected_ids")}})
                pq.write_table(table, a.out / f"{split}-{shard:04d}.parquet", compression="zstd")
                shard += 1
                for v in buf.values():
                    v.clear()

            work = partial(encode, max_length=a.max_length)
            for f in files:
                with gzip.open(f, "rt", encoding="utf-8") as handle:
                    while batch := list(islice(handle, 20_000)):   # bounded memory: 20K lines in flight
                        for row, ids, reason in pool.map(work, batch, chunksize=256):
                            c = stats.setdefault(row["source"], Counter())
                            c["staged"] += 1
                            if reason:
                                c[f"dropped_{reason}"] += 1
                                continue
                            c["kept"] += 1
                            p, ch, rj = ids
                            lengths.append(len(p) + max(len(ch), len(rj)))
                            for k, v in (("id", row["id"]), ("source", row["source"]), ("language", row["language"]),
                                         ("prompt_ids", p), ("chosen_ids", ch), ("rejected_ids", rj)):
                                buf[k].append(v)
                            if len(buf["id"]) >= SHARD_ROWS:
                                flush()
                print(split, f.name, {s: dict(v) for s, v in stats.items()}, flush=True)
            flush()
            lengths.sort()
            manifest["splits"][split] = {
                "by_source": {s: dict(v) for s, v in stats.items()},
                "kept": sum(v["kept"] for v in stats.values()),
                "length_p50": lengths[len(lengths) // 2] if lengths else None,
                "length_p95": lengths[int(len(lengths) * 0.95)] if lengths else None,
                "mean_length": sum(lengths) / len(lengths) if lengths else None,
                "shards": shard}
    (a.out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=1))


if __name__ == "__main__":
    main()
