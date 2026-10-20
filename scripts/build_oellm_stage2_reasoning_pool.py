#!/usr/bin/env python3
"""Build the pre-tokenized Stage-2 reasoning *candidate pool* with the v2 chat template.

Reads the reasoning rows of pinned Excellent-SFT, normalizes inline <think> spans into
`reasoning_content`, and renders with the v2 template, which puts the opening <think>
prompt-side. Output rows have the same columns as the Stage-1 mix (plus the normalized
messages), so a Stage-2 run mixes this pool with Stage-1 direct replay by file glob.

Normalization: only the final assistant turn keeps its reasoning (earlier turns drop it,
matching inference, where clients do not send reasoning back). Every admitted row must
have a nonempty trace, a nonempty final answer, exactly one prompt-side <think> and one
supervised </think>, and no supervised <think>. Overlength rows are rejected, never
truncated. Per-source sampling is deterministic by prompt hash; this is a pool to select
from, not the final Stage-2 mixture.
"""

from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
import os
from pathlib import Path

from jinja2 import TemplateError
import pyarrow as pa
import pyarrow.parquet as pq

from audit_two_stage_sft import source_kind
from build_oellm_excellent_stage1 import canonical_prompt, file_sha256, get_tokenizer
from build_oellm_stage1_mix import SplitWriter, h64

THINK_OPEN, THINK_CLOSE = 18, 17
# Source -> (keep fraction, language of the final answer). Traces are English throughout.
SOURCES = {
    "nemotron/multilingual_de": (0.10, "de"),
    "nemotron/multilingual_es": (0.10, "es"),
    "nemotron/multilingual_fr": (0.10, "fr"),
    "nemotron/multilingual_it": (0.10, "it"),
    "nemotron/chat": (0.60, "en"),
    "smoltalk2/OpenThoughts3_1.2M_think": (0.30, "en"),
    "smoltalk2/smoltalk_multilingual8_Qwen3_32B_think": (1.0, "multi"),
    "smoltalk2/multi_turn_reasoning_if_think": (1.0, "en"),
    "smoltalk2/s1k_1.1_think": (1.0, "en"),
    "smoltalk2/aya_dataset_Qwen3_32B_think": (1.0, "multi"),
    "smoltalk2/smoltalk_everyday_convs_reasoning_Qwen3_32B_think": (1.0, "en"),
    "smoltalk2/table_gpt_Qwen3_32B_think": (1.0, "en"),
    "smoltalk2/LongAlign_64k_Qwen3_32B_yarn_131k_think": (1.0, "en"),
}
SCHEMA = pa.schema([
    ("input_ids", pa.list_(pa.int32())), ("assistant_masks", pa.list_(pa.int8())),
    ("source", pa.string()), ("language", pa.string()), ("id", pa.string()),
    ("supervised_tokens", pa.int32()), ("reasoning_tokens", pa.int32()),
    ("messages_json", pa.string()),
])


def normalize(messages: list[dict]) -> list[dict]:
    """Inline <think> -> reasoning_content on the final turn; drop it from earlier turns."""
    out = []
    last = max(i for i, m in enumerate(messages) if m["role"] == "assistant")
    for i, message in enumerate(messages):
        content = message["content"]
        if message["role"] != "assistant":
            out.append({"role": message["role"], "content": content})
            continue
        reasoning = message.get("reasoning_content") or ""
        if "<think>" in content:
            reasoning, _, content = content.partition("<think>")[2].partition("</think>")
        reasoning, content = reasoning.strip(), content.strip()
        if i == last:
            if not reasoning or not content:
                raise ValueError("final turn needs a nonempty trace and answer")
            out.append({"role": "assistant", "reasoning_content": reasoning, "content": content})
        else:
            out.append({"role": "assistant", "content": content})
    return out


def process_shard(index: int, shard_path: str, output_dir: str, config: dict) -> dict:
    out = Path(output_dir)
    name = f"excellent-{index:05d}"
    report_path = out / f"{name}.report.json"
    if report_path.exists():
        previous = json.loads(report_path.read_text(encoding="utf-8"))
        if previous.get("config") == config and previous.get("shard_path") == shard_path:
            return previous
    tokenizer = get_tokenizer(config["tokenizer"], config["template"])
    writers = {split: SplitWriter(out, split, name, batch_rows=500, schema=SCHEMA)
               for split in ("train", "validation")}
    counts: Counter[str] = Counter()
    by_source: dict[str, Counter[str]] = {}
    for batch in pq.ParquetFile(shard_path).iter_batches(batch_size=256, columns=["id", "messages", "source", "tools"]):
        for row in batch.to_pylist():
            source = str(row.get("source"))
            if source not in SOURCES or row.get("tools"):
                continue
            fraction, language = SOURCES[source]
            group = by_source.setdefault(source, Counter())
            messages = row["messages"]
            try:
                prompt_hash = canonical_prompt(messages)
                if int(prompt_hash[:16], 16) / 2**64 >= fraction:
                    continue
                if source_kind(messages) != "reasoning":
                    continue
                group["sampled_reasoning"] += 1
                if any("<image>" in m["content"] for m in messages if m["role"] == "user"):
                    raise ValueError("prompt references a missing image")
                normalized = normalize(messages)
                rendered = tokenizer.apply_chat_template(
                    normalized, tokenize=True, add_generation_prompt=False,
                    return_dict=True, return_assistant_tokens_mask=True,
                )
                ids = list(rendered["input_ids"])
                mask = rendered.get("assistant_masks")
                if mask is None:
                    mask = rendered.get("assistant_tokens_mask")
                if mask is None or len(ids) != len(mask) or not 0 < sum(mask) < len(mask) or mask[0]:
                    raise ValueError("invalid assistant-only loss mask")
                target = [token for token, active in zip(ids, mask) if active]
                if THINK_OPEN in target:
                    raise RuntimeError(f"FATAL: {row.get('id')} supervises <think>")
                if ids.count(THINK_OPEN) != 1 or target.count(THINK_CLOSE) != 1 or ids.count(THINK_CLOSE) != 1:
                    raise ValueError("expected one prompt-side <think> and one supervised </think>")
                if len(ids) > config["max_length"]:
                    group["overlength"] += 1
                    continue
                reasoning_tokens = ids.index(THINK_CLOSE) - ids.index(THINK_OPEN) - 1
            except RuntimeError:
                raise
            except (KeyError, TypeError, ValueError, TemplateError) as exc:
                group["rejected"] += 1
                counts["rejected"] += 1
                if counts["rejected"] <= 3:
                    print(f"{name} rejected {row.get('id')}: {exc}", flush=True)
                continue
            split = "validation" if int(prompt_hash[16:32], 16) % 1000 == 0 else "train"
            writers[split].add({
                "input_ids": ids, "assistant_masks": [int(v) for v in mask], "source": source,
                "language": language, "id": str(row.get("id")), "supervised_tokens": len(target),
                "reasoning_tokens": reasoning_tokens,
                "messages_json": json.dumps(normalized, ensure_ascii=False),
            })
            group[f"{split}_rows"] += 1
            group[f"{split}_rendered_tokens"] += len(ids)
            group[f"{split}_supervised_tokens"] += len(target)
            group[f"{split}_reasoning_tokens"] += reasoning_tokens
            bucket = next(b for b in (4096, 8192, 16384, 32768, 1 << 30) if len(ids) <= b)
            group[f"len_le_{bucket}"] += 1
    files = {split: info for split, writer in writers.items() if (info := writer.finish())}
    report = {"shard": index, "shard_path": shard_path, "config": config, "counts": dict(counts),
              "by_source": {k: dict(v) for k, v in sorted(by_source.items())}, "files": files}
    partial = report_path.with_name(report_path.name + ".partial")
    partial.write_text(json.dumps(report, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(partial, report_path)
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--excellent-dir", type=Path, required=True)
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--template", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-length", type=int, default=32768)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--max-shards", type=int, default=0)
    args = parser.parse_args()
    shards = sorted((args.excellent_dir / "data").glob("train-*.parquet"))
    if len(shards) != 62:
        raise ValueError(f"expected 62 Excellent shards, got {len(shards)}")
    tokenizer = get_tokenizer(str(args.tokenizer), str(args.template))
    probe = tokenizer.apply_chat_template([{"role": "user", "content": "x"}], tokenize=False,
                                          add_generation_prompt=True, enable_thinking=True)
    if not probe.endswith("<|im_start|>assistant\n<think>\n"):
        raise ValueError("template is not the v2 think-switch template")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    config = {
        "tokenizer": str(args.tokenizer), "template": str(args.template),
        "template_sha256": file_sha256(args.template),
        "tokenizer_json_sha256": file_sha256(args.tokenizer / "tokenizer.json"),
        "max_length": args.max_length, "sources": {k: list(v) for k, v in SOURCES.items()},
    }
    selected = shards[:args.max_shards] if args.max_shards else shards
    reports = []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        jobs = [pool.submit(process_shard, i, str(p), str(args.output_dir), config) for i, p in enumerate(selected)]
        for future in as_completed(jobs):
            report = future.result()
            reports.append(report)
            print(f"shard {report['shard']} done: {sum(v.get('train_rows', 0) for v in report['by_source'].values())} train rows", flush=True)
    by_source: dict[str, Counter[str]] = {}
    files = {"train": [], "validation": []}
    for report in sorted(reports, key=lambda r: r["shard"]):
        for source, values in report["by_source"].items():
            by_source.setdefault(source, Counter()).update(values)
        for split, info in report["files"].items():
            files[split].append(info)
    totals = {split: {key: sum(v.get(f"{split}_{key}", 0) for v in by_source.values())
                      for key in ("rows", "rendered_tokens", "supervised_tokens", "reasoning_tokens")}
              for split in ("train", "validation")}
    manifest = {
        "kind": "stage2_reasoning_pool_pretokenized" if not args.max_shards else "engineering_smoke_not_production",
        "version": "v1", "config": config, "totals": totals,
        "by_source": {k: dict(v) for k, v in sorted(by_source.items())}, "files": files,
        "invariant": "One prompt-side <think>, one supervised </think>, nonempty trace and answer, no truncation",
        "status": "candidate_pool_not_final_mixture",
    }
    path = args.output_dir / "manifest.json"
    partial = path.with_name(path.name + ".partial")
    partial.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(partial, path)
    print(json.dumps({"totals": totals, "by_source": manifest["by_source"]}, indent=2), flush=True)


if __name__ == "__main__":
    main()
