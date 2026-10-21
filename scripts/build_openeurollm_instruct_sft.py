#!/usr/bin/env python3
"""Build the OpenEuroLLM Instruct SFT dataset (messages format) from the Stage-1 selection.

Starts from exactly the rows selected for stage1-mix-v1 (same units, hashing and caps; identity
v1 stored once) and applies two fixes:

1. Foreign identity claims. English "As an AI developed by OpenAI, ..." phrasing is rewritten to
   "As an AI, ..."; any row whose assistant text still names another assistant or lab (OpenAI,
   ChatGPT, GPT-n, Anthropic, Qwen, DeepSeek, ...) while no user/system turn mentions it is dropped.
   This catches translated self-claims without language-specific rules.
2. Identity v2: short, direct identity Q&A in 36 languages (`resources/identity_v2_core.json`):
   plain, under a generic system prompt, and appended after a real task conversation.

Output: Parquet shards `data/{train,validation}-*.parquet` with id, source, language, messages,
modification; plus manifest.json with per-source counts of rewritten and dropped rows.
"""
from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
import json
import os
import re
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from build_oellm_stage1_mix import iter_unit, plan_units

OPENAI_REWRITE = re.compile(
    r"\b((?:as|I am|I'm)\s+an?\s+[^,.\n]{0,40}?)\s*,?\s+(?:developed|created|trained|made|built|designed)\s+by\s+OpenAI\b",
    re.I)
OPENAI_SELF = re.compile(
    r"\b(?:as an?|I am an?|I'm an?|I was)\b[^.\n]{0,50}?\b(?:developed|created|trained|made|built|designed) by OpenAI\b"
    r"|\bI(?:'m| am) (?:ChatGPT|GPT-?[345])\b|\bas ChatGPT\b", re.I)
FOREIGN = re.compile(
    r"\b(?:OpenAI|ChatGPT|GPT-?(?:3(?:\.5)?|4(?:o|\.\d)?|5)|Anthropic|Qwen|Tongyi|DeepSeek|Mistral AI|Meta AI|SmolLM)\b")
SCHEMA = pa.schema([
    ("id", pa.string()), ("source", pa.string()), ("language", pa.string()),
    ("messages", pa.list_(pa.struct([("role", pa.string()), ("content", pa.string())]))),
    ("modification", pa.string()),
])
SYSTEM_PROMPT = "You are a helpful assistant."


def h64(text: str) -> int:
    return int(hashlib.sha256(text.encode("utf-8")).hexdigest()[:16], 16)


def clean(messages: list[dict]) -> tuple[list[dict] | None, str]:
    """Return (messages, modification) or (None, reason) when the row is dropped."""
    out, modification = [], ""
    for m in messages:
        content = m["content"]
        if m["role"] == "assistant" and OPENAI_REWRITE.search(content):
            content = OPENAI_REWRITE.sub(r"\1", content)
            modification = "rewrote_openai_self_attribution"
        out.append({"role": m["role"], "content": content})
    answer = "\n".join(m["content"] for m in out if m["role"] == "assistant")
    context = "\n".join(m["content"] for m in out if m["role"] != "assistant")
    if OPENAI_SELF.search(answer):
        return None, "drop_openai_self_claim"
    if FOREIGN.search(answer) and not FOREIGN.search(context):
        return None, "drop_unprompted_foreign_model_mention"
    return out, modification


def process(unit: dict, out_dir: str, config: dict) -> dict:
    name = unit["name"]
    rows = {"train": [], "validation": []}
    counts: dict[str, Counter] = {}
    examples: list[dict] = []
    contexts: list[tuple[int, str, list[dict]]] = []  # candidates for mid-conversation identity
    for split, source, language, row_id, messages in iter_unit(unit, config):
        c = counts.setdefault(source, Counter())
        c[f"{split}_in"] += 1
        try:
            messages = [{"role": m["role"], "content": m["content"]} for m in messages]
            if not all(isinstance(m["content"], str) for m in messages):
                raise TypeError
        except (KeyError, TypeError):
            c["drop_malformed"] += 1
            continue
        cleaned, mod = clean(messages)
        if cleaned is None:
            c[mod] += 1
            if len(examples) < 3:
                examples.append({"id": row_id, "source": source, "reason": mod,
                                 "answer": "\n".join(m["content"] for m in messages if m["role"] == "assistant")[:300]})
            continue
        if mod:
            c[mod] += 1
        rows[split].append({"id": row_id, "source": source, "language": language,
                            "messages": cleaned, "modification": mod})
        c[f"{split}_out"] += 1
        if (split == "train" and len(cleaned) == 2 and cleaned[0]["role"] == "user"
                and len(cleaned[1]["content"]) < 1500 and source in config["context_sources"]):
            lang = language if source == "dolci_tr" else "en"
            contexts.append((h64(row_id), lang, cleaned))
    files = {}
    for split, items in rows.items():
        if items:
            path = Path(out_dir) / "data" / f"{split}-{name}.parquet"
            pq.write_table(pa.Table.from_pylist(items, schema=SCHEMA), path, compression="zstd")
            files[split] = {"path": str(path), "rows": len(items)}
    contexts.sort(key=lambda x: x[0])
    keep: dict[str, list] = {}
    for _, lang, msgs in contexts:
        if len(keep.setdefault(lang, [])) < 6:
            keep[lang].append(msgs)
    return {"unit": name, "counts": {k: dict(v) for k, v in counts.items()}, "files": files,
            "examples": examples, "contexts": keep}


def identity_v2(spec: dict, contexts: dict[str, list]) -> list[dict]:
    rows = []
    for lang, block in spec["languages"].items():
        for qkey, akey in spec["answer_map"].items():
            q, a = block["q"][qkey], block["a"][akey]
            base = f"oellm-id-v2-{lang}-{qkey}"
            rows.append({"id": f"{base}-plain", "messages": [{"role": "user", "content": q}, {"role": "assistant", "content": a}]})
            rows.append({"id": f"{base}-system", "messages": [{"role": "system", "content": SYSTEM_PROMPT},
                                                             {"role": "user", "content": q}, {"role": "assistant", "content": a}]})
        for i, (qkey, akey) in enumerate(spec["answer_map"].items()):
            ctx = contexts.get(lang, [])
            if i < len(ctx):
                rows.append({"id": f"oellm-id-v2-{lang}-{qkey}-after-task", "messages": ctx[i] + [
                    {"role": "user", "content": block["q"][qkey]},
                    {"role": "assistant", "content": block["a"][akey]}]})
        for r in rows:
            r.setdefault("language", lang)
    for r in rows:
        r.update({"source": "identity_v2", "modification": "new_identity_v2"})
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, required=True)
    ap.add_argument("--identity-v2", type=Path, required=True)
    ap.add_argument("--output-dir", type=Path, required=True)
    ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args()
    manifest_in = json.loads((a.root / "data/stage1-mix-v1/manifest.json").read_text())
    config = dict(manifest_in["config"])
    config["identity_exposures"] = 1  # store unique identity rows; exposure is a training choice
    config["context_sources"] = ["dolci_tr", "openeurollm/OpenEuroLLM-Excellent-SFT:dolci/train"]
    args = argparse.Namespace(
        excellent_dir=a.root / "data/excellent-stage1-full-v1",
        dolci_dir=a.root / "source-datasets/dolci-instruct-translated",
        aya_dir=a.root / "aya-dataset", identity_dir=a.root / "data/model-identity-v1")
    (a.output_dir / "data").mkdir(parents=True, exist_ok=True)
    units = plan_units(args)
    counts: dict[str, Counter] = {}
    files = {"train": [], "validation": []}
    examples, contexts = [], {}
    with ProcessPoolExecutor(max_workers=a.workers) as pool:
        for future in as_completed([pool.submit(process, u, str(a.output_dir), config) for u in units]):
            r = future.result()
            for source, c in r["counts"].items():
                counts.setdefault(source, Counter()).update(c)
            for split, info in r["files"].items():
                files[split].append(info)
            examples.extend(r["examples"][: max(0, 30 - len(examples))])
            for lang, msgs in r["contexts"].items():
                contexts.setdefault(lang, []).extend(msgs)
    for lang in contexts:  # deterministic choice independent of completion order
        contexts[lang] = sorted(contexts[lang], key=lambda m: h64(json.dumps(m, ensure_ascii=False)))[:6]
    spec = json.loads(a.identity_v2.read_text(encoding="utf-8"))
    v2 = identity_v2(spec, contexts)
    path = a.output_dir / "data" / "train-identity_v2.parquet"
    pq.write_table(pa.Table.from_pylist(v2, schema=SCHEMA), path, compression="zstd")
    files["train"].append({"path": str(path), "rows": len(v2)})
    totals = {split: sum(f["rows"] for f in fs) for split, fs in files.items()}
    grand = Counter()
    for c in counts.values():
        grand.update(c)
    manifest = {
        "name": "openeurollm-instruct-sft", "version": "1.0.0", "totals": totals,
        "identity_v2_rows": len(v2), "identity_v2_spec": spec["version"],
        "cleaning": dict(grand), "by_source": {k: dict(v) for k, v in sorted(counts.items())},
        "drop_examples": examples, "files": files,
        "selection": "stage1-mix-v1 selection (manifest config sha-pinned), identity v1 stored once",
    }
    (a.output_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"totals": totals, "identity_v2_rows": len(v2), "cleaning": {k: v for k, v in grand.items() if not k.endswith(("_in", "_out"))}}, indent=1))


if __name__ == "__main__":
    main()
