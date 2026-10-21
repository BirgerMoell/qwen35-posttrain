#!/usr/bin/env python3
"""Build the pre-tokenized Stage-2 (reasoning) SFT mixture with the v2 chat template.

reasoning  rows from stage2-reasoning-pool-v1, selected per source up to a rendered-token
           budget in deterministic id-hash order (already audited: one prompt-side <think>,
           one supervised </think>, nonempty trace and answer, <=32K tokens)
replay     direct-answer rows from the cleaned openeurollm-instruct-sft-v1 dataset, sampled by
           id hash; all identity rows (v1 + v2) kept at a fixed exposure count; answers wrapped
           whole in <Answer>/<AnswerGeneration>/<response> tags are unwrapped. Rendered and
           audited exactly like Stage 1 (no supervised think token, answer preserved).

Same columns as the Stage-1 mix, so the trainer consumes both by glob.
"""
from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
import os
import re
from pathlib import Path

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
from jinja2 import TemplateError

from audit_two_stage_sft import source_kind
from build_oellm_excellent_stage1 import THINK_IDS, file_sha256, get_tokenizer
from build_oellm_stage1_mix import SCHEMA, SplitWriter, h64

# Rendered-token budgets for the reasoning pool, by source (pool totals in comments).
REASONING_BUDGET = {
    "nemotron/multilingual_de": 150e6, "nemotron/multilingual_es": 150e6,   # ~443M / ~397M
    "nemotron/multilingual_fr": 150e6, "nemotron/multilingual_it": 150e6,   # ~426M / ~422M
    "smoltalk2/OpenThoughts3_1.2M_think": 300e6,                            # ~1.11B
    "nemotron/chat": 1e12, "smoltalk2/smoltalk_multilingual8_Qwen3_32B_think": 1e12,
    "smoltalk2/LongAlign_64k_Qwen3_32B_yarn_131k_think": 1e12,
    "smoltalk2/multi_turn_reasoning_if_think": 1e12, "smoltalk2/s1k_1.1_think": 1e12,
    "smoltalk2/aya_dataset_Qwen3_32B_think": 1e12, "smoltalk2/table_gpt_Qwen3_32B_think": 1e12,
    "smoltalk2/smoltalk_everyday_convs_reasoning_Qwen3_32B_think": 1e12,
}
WRAPPED = re.compile(r"^\s*<(AnswerGeneration|Answer|answer|response|Response)>\s*(.*?)\s*</\1>\s*$", re.S)


def select_reasoning(pool_dir: Path) -> dict[str, set[int]]:
    """Return {file: selected row indices} honouring the per-source token budgets."""
    rows = []
    for path in sorted(pool_dir.glob("train-*.parquet")):
        t = pq.read_table(path, columns=["id", "source", "input_ids"])
        lengths = pc.list_value_length(t["input_ids"]).to_pylist()
        for i, (rid, src, n) in enumerate(zip(t["id"].to_pylist(), t["source"].to_pylist(), lengths)):
            rows.append((h64(rid), src, n, str(path), i))
    rows.sort()
    used: Counter[str] = Counter()
    chosen: dict[str, set[int]] = {}
    for _, src, n, path, i in rows:
        if used[src] + n > REASONING_BUDGET.get(src, 0):
            continue
        used[src] += n
        chosen.setdefault(path, set()).add(i)
    print("reasoning tokens by source:", {k: f"{v/1e6:.0f}M" for k, v in sorted(used.items())}, flush=True)
    return chosen


def copy_reasoning(path: str, keep: list[int], out: str, split: str) -> dict:
    t = pq.read_table(path).take(sorted(keep)).select(SCHEMA.names).cast(SCHEMA)
    dest = Path(out) / f"{split}-reasoning-{Path(path).stem.split('-', 1)[1]}.parquet"
    pq.write_table(t, dest, compression="zstd")
    n = pc.sum(pc.list_value_length(t["input_ids"])).as_py()
    return {"path": str(dest), "rows": t.num_rows, "rendered_tokens": n, "kind": "reasoning"}


def replay_unit(path: str, out: str, config: dict) -> dict:
    tokenizer = get_tokenizer(config["tokenizer"], config["template"])
    split = "validation" if Path(path).name.startswith("validation-") else "train"
    writer = SplitWriter(Path(out), split, "replay-" + Path(path).stem.split("-", 1)[1])
    c: Counter[str] = Counter()
    for batch in pq.ParquetFile(path).iter_batches(batch_size=1024):
        for row in batch.to_pylist():
            source, identity = row["source"], row["source"] in ("identity", "identity_v2")
            if not identity and h64(row["id"]) / 2**64 >= config["replay_fraction"]:
                continue
            messages = []
            for m in row["messages"]:
                content = m["content"]
                if m["role"] == "assistant" and (w := WRAPPED.match(content)):
                    content = w.group(2)
                    c["unwrapped_answer_tag"] += 1
                messages.append({"role": m["role"], "content": content})
            try:
                if source_kind(messages) != "direct":
                    raise ValueError("not direct")
                r = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=False,
                                                  return_dict=True, return_assistant_tokens_mask=True)
                ids, mask = list(r["input_ids"]), r.get("assistant_masks")
                if mask is None or len(ids) != len(mask) or not 0 < sum(mask) < len(mask) or mask[0]:
                    raise ValueError("invalid mask")
                target = [t for t, a in zip(ids, mask) if a]
                if any(t in THINK_IDS for t in target):
                    raise RuntimeError(f"FATAL: replay row {row['id']} supervises a think token")
                if len(ids) > config["max_length"]:
                    c["overlength"] += 1
                    continue
            except RuntimeError:
                raise
            except (KeyError, TypeError, ValueError, TemplateError):
                c["rejected"] += 1
                continue
            for _ in range(config["identity_exposures"] if identity and split == "train" else 1):
                writer.add({"input_ids": ids, "assistant_masks": [int(v) for v in mask], "source": source,
                            "language": row["language"], "id": row["id"], "supervised_tokens": len(target)})
                c["rows"] += 1
                c["rendered_tokens"] += len(ids)
    info = writer.finish() or {}
    return {**info, **c, "kind": "replay", "split": split}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pool-dir", type=Path, required=True)
    ap.add_argument("--instruct-dir", type=Path, required=True)
    ap.add_argument("--tokenizer", type=Path, required=True)
    ap.add_argument("--template", type=Path, required=True)
    ap.add_argument("--output-dir", type=Path, required=True)
    ap.add_argument("--replay-fraction", type=float, default=0.19)
    ap.add_argument("--identity-exposures", type=int, default=2)
    ap.add_argument("--max-length", type=int, default=32768)
    ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args()
    pool_manifest = json.loads((a.pool_dir / "manifest.json").read_text())
    template_sha = file_sha256(a.template)
    if pool_manifest["config"]["template_sha256"] != template_sha:
        raise ValueError("reasoning pool was built with a different template")
    get_tokenizer(str(a.tokenizer), str(a.template))
    a.output_dir.mkdir(parents=True, exist_ok=True)
    config = {"tokenizer": str(a.tokenizer), "template": str(a.template), "template_sha256": template_sha,
              "tokenizer_json_sha256": file_sha256(a.tokenizer / "tokenizer.json"),
              "replay_fraction": a.replay_fraction, "identity_exposures": a.identity_exposures,
              "max_length": a.max_length, "reasoning_budget": REASONING_BUDGET}
    chosen = select_reasoning(a.pool_dir)
    files = []
    with ProcessPoolExecutor(max_workers=a.workers) as pool:
        jobs = [pool.submit(copy_reasoning, p, sorted(k), str(a.output_dir), "train") for p, k in chosen.items()]
        jobs += [pool.submit(copy_reasoning, str(p), list(range(pq.ParquetFile(p).metadata.num_rows)), str(a.output_dir), "validation")
                 for p in sorted(a.pool_dir.glob("validation-*.parquet"))]
        jobs += [pool.submit(replay_unit, str(p), str(a.output_dir), config)
                 for p in sorted((a.instruct_dir / "data").glob("*.parquet"))]
        for done, f in enumerate(as_completed(jobs), 1):
            files.append(f.result())
            if done % 100 == 0:
                print(f"{done}/{len(jobs)}", flush=True)
    totals: dict[str, Counter] = {}
    for f in files:
        split = f.get("split") or ("validation" if "/validation-" in f["path"] else "train")
        k = totals.setdefault(f"{split}_{f['kind']}", Counter())
        k["rows"] += f.get("rows", 0)
        k["rendered_tokens"] += f.get("rendered_tokens", 0)
        for key in ("unwrapped_answer_tag", "rejected", "overlength"):
            k[key] += f.get(key, 0)
    manifest = {"kind": "stage2_reasoning_mix_pretokenized", "version": "v1", "config": config,
                "totals": {k: dict(v) for k, v in sorted(totals.items())},
                "files": [f for f in files if f.get("path")],
                "status": "candidate_internal_training_data_not_release_clearance"}
    (a.output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest["totals"], indent=1))


if __name__ == "__main__":
    main()
