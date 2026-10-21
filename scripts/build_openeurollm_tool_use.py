#!/usr/bin/env python3
"""Build OpenEuroLLM Tool-Use v1: canonical tool-use conversations (public dataset) plus a
pre-tokenized pool for the Stage-3 mix, rendered with chat template v2.1.

Per row: source adapter -> validation (declared tools, schema-valid arguments, well-formed
trajectory) -> identity hygiene -> mode/category labels -> render + token audit (no supervised
<think>, <tool_response> or <tools>; every call supervised exactly once; <= max length).
Sampling is deterministic (hash of the raw record vs a per-source keep probability derived from
a token budget); duplicates (same first user prompt + tool set) are removed across sources in
priority order; rows overlapping BFCL / When2Call test prompts (13-gram) are removed.
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
from jinja2 import TemplateError
from transformers import AutoTokenizer

from build_oellm_stage1_mix import SCHEMA as TOKEN_SCHEMA
from tooluse_adapters import ADAPTERS, assign_mode, categorize, check_special_text, trim_trailing_user, validate

CHARS_PER_TOKEN = 3.3
FOREIGN = re.compile(r"\b(?:OpenAI|ChatGPT|GPT-?(?:3(?:\.5)?|4(?:o|\.\d)?|5)|Anthropic|Claude|Qwen|Tongyi|DeepSeek|"
                     r"Mistral AI|Meta AI|SmolLM|OLMo|Olmo|Ai2|AllenAI|Kimi|Moonshot|GLM|Zhipu|Gemini|Nemotron)\b")
WORD = re.compile(r"\w+", re.U)
FORBIDDEN_LABEL_IDS = (18, 15, 13)  # <think>, <tool_response>, <tools>
TOOL_CALL_ID = 11

# source key -> (adapter, upstream repo@rev, license, token budget or "all", priority for dedup)
SOURCES = {
    "eu_tooluse": ("eu_tooluse", "birgermoell/oellm-eu-tooluse-v1@b131d571", "Apache-2.0", "all", 0),
    "toolace": ("toolace", "Team-ACE/ToolACE@6bda777c", "Apache-2.0", "all", 1),
    "when2call": ("when2call", "nvidia/When2Call@0582f774", "CC-BY-4.0", "all", 2),
    "nemotron_tool_calling": ("nemotron_agentic", "nvidia/Nemotron-SFT-Agentic-v2@7c804833", "CC-BY-4.0", 80e6, 3),
    "nemotron_search": ("nemotron_agentic", "nvidia/Nemotron-SFT-Agentic-v2@7c804833", "CC-BY-4.0", 15e6, 4),
    "nemotron_interactive": ("nemotron_agentic", "nvidia/Nemotron-SFT-Agentic-v2@7c804833", "CC-BY-4.0", 30e6, 5),
    "toucan": ("toucan", "Agent-Ark/Toucan-1.5M@0df3cf37", "Apache-2.0", 60e6, 6),
    "toolmind_graphsyn": ("toolmind", "Nanbeige/ToolMind@8020ed1c", "Apache-2.0", 50e6, 7),
    "toolmind_toolace": ("toolmind", "Nanbeige/ToolMind@8020ed1c", "Apache-2.0 (ToolMind; prompts from ToolACE, Apache-2.0)", 15e6, 8),
    "toolmind_when2call": ("toolmind", "Nanbeige/ToolMind@8020ed1c", "Apache-2.0 (ToolMind; prompts from When2Call, CC-BY-4.0)", 15e6, 9),
    "toolmind_glaive": ("toolmind", "Nanbeige/ToolMind@8020ed1c", "Apache-2.0 (ToolMind; prompts from glaive-function-calling-v2, Apache-2.0)", 15e6, 10),
}
FILES = {
    "eu_tooluse": ["oellm-eu-tooluse-v1/train.parquet"],
    "toolace": ["ToolACE/data.json"],
    "when2call": ["When2Call/train/when2call_train_sft.jsonl"],
    "nemotron_tool_calling": ["Nemotron-SFT-Agentic-v2/data/tool_calling.jsonl"],
    "nemotron_search": ["Nemotron-SFT-Agentic-v2/data/search.jsonl"],
    "nemotron_interactive": ["Nemotron-SFT-Agentic-v2/data/interactive_agent.jsonl"],
    "toucan": ["Toucan-1.5M/SFT/*.parquet"],
    "toolmind_graphsyn": ["ToolMind/graph_syn_datasets/graphsyn.jsonl"],
    "toolmind_toolace": ["ToolMind/open_datasets/ToolACE-query.jsonl"],
    "toolmind_when2call": ["ToolMind/open_datasets/When2Call-query.jsonl"],
    "toolmind_glaive": ["ToolMind/open_datasets/glaive-function-calling-v2-query.jsonl"],
}
CHUNK_BYTES = 256 * 1024 * 1024

MSG = pa.list_(pa.struct([
    ("role", pa.string()), ("content", pa.string()), ("reasoning_content", pa.string()),
    ("tool_calls", pa.list_(pa.struct([("type", pa.string()), ("function", pa.struct([
        ("name", pa.string()), ("arguments", pa.string())]))]))),
]))
DATA_SCHEMA = pa.schema([("id", pa.string()), ("source", pa.string()), ("source_dataset", pa.string()),
                         ("license", pa.string()), ("language", pa.string()), ("category", pa.string()),
                         ("mode", pa.string()), ("tools", pa.string()), ("messages", MSG), ("dedup_key", pa.string())])
TOK_SCHEMA = TOKEN_SCHEMA.append(pa.field("category", pa.string())).append(pa.field("mode", pa.string())) \
                         .append(pa.field("dedup_key", pa.string()))


def sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def ngrams(text: str, n: int = 13) -> set[str]:
    words = WORD.findall(text.lower())
    return {" ".join(words[i:i + n]) for i in range(len(words) - n + 1)}


# ---------------------------------------------------------------- record readers

def read_records(unit: dict):
    """Yield (record_hash, record) for one unit; jsonl units are byte ranges."""
    path = unit["path"]
    if path.endswith(".parquet"):
        for b in pq.ParquetFile(path).iter_batches(batch_size=512):
            for r in b.to_pylist():
                yield sha(json.dumps(r, sort_keys=True, ensure_ascii=False, default=str)), r
    elif path.endswith(".json"):
        for r in json.load(open(path, encoding="utf-8")):
            yield sha(json.dumps(r, sort_keys=True, ensure_ascii=False)), r
    else:
        with open(path, "rb") as f:
            start, end = unit["start"], unit["end"]
            f.seek(start)
            if start:
                f.readline()  # skip the partial line; it belongs to the previous chunk
            while f.tell() <= end:
                line = f.readline()
                if not line:
                    break
                if line.strip():
                    yield hashlib.sha256(line).hexdigest(), line


# ---------------------------------------------------------------- per-unit build

_TOK = None


def tokenizer(cfg):
    global _TOK
    if _TOK is None:
        _TOK = AutoTokenizer.from_pretrained(cfg["tokenizer"], local_files_only=True)
        _TOK.chat_template = Path(cfg["template"]).read_text(encoding="utf-8")
    return _TOK


def to_storage(messages):
    out = []
    for m in messages:
        calls = [{"type": "function", "function": {"name": c["function"]["name"],
                  "arguments": json.dumps(c["function"]["arguments"], ensure_ascii=False)}} for c in m.get("tool_calls") or []]
        out.append({"role": m["role"], "content": m.get("content") or "", "reasoning_content": m.get("reasoning_content"),
                    "tool_calls": calls or None})
    return out


def build_unit(unit: dict, cfg: dict, keep_prob: dict, contam: set) -> dict:
    src = unit["source"]
    adapter_key, source_dataset, license_, _, _ = SOURCES[src]
    adapter = ADAPTERS[adapter_key]
    tok = tokenizer(cfg)
    p = keep_prob[src]
    data_rows, tok_rows = [], []
    c: Counter[str] = Counter()
    examples: dict[str, str] = {}
    for rec_hash, rec in read_records(unit):
        if int(rec_hash[:16], 16) / 2**64 >= p:
            continue
        c["sampled"] += 1
        raw = None
        try:
            raw = json.loads(rec) if isinstance(rec, (bytes, str)) else rec
            row = trim_trailing_user(adapter(raw))
            validate(row)
            check_special_text(row)
            users = [m["content"] for m in row["messages"] if m["role"] == "user"]
            if not users or any("<image>" in u for u in users):
                raise ValueError("no user turn / image placeholder")
            context = "\n".join(m["content"] for m in row["messages"] if m["role"] != "assistant") + json.dumps(row["tools"])
            answer = "\n".join((m.get("content") or "") + (m.get("reasoning_content") or "")
                               for m in row["messages"] if m["role"] == "assistant")
            if FOREIGN.search(answer) and not FOREIGN.search(context):
                raise LookupError("foreign assistant identity")
            if any(ngrams(u) & contam for u in users):
                raise LookupError("benchmark overlap")
            category = categorize(row)
            mode = assign_mode(row)
            rendered = tok.apply_chat_template(row["messages"], tools=row["tools"] or None, tokenize=True,
                                               return_dict=True, return_assistant_tokens_mask=True)
            ids, mask = list(rendered["input_ids"]), list(rendered["assistant_masks"])
            labels = [t for t, a in zip(ids, mask) if a]
            n_calls = sum(len(m.get("tool_calls") or []) for m in row["messages"])
            if any(t in labels for t in FORBIDDEN_LABEL_IDS) or labels.count(TOOL_CALL_ID) != n_calls:
                raise RuntimeError(f"FATAL: label audit failed for {src}")
            if mode == "reasoning" and 17 not in labels:
                raise RuntimeError(f"FATAL: reasoning row without supervised </think> in {src}")
            if len(ids) > cfg["max_length"]:
                raise OverflowError("overlength")
        except RuntimeError:
            raise
        except OverflowError:
            c["drop_overlength"] += 1
            continue
        except LookupError as exc:
            c[f"drop_{str(exc).replace(' ', '_')}"] += 1
            continue
        except (ValueError, KeyError, TypeError, AttributeError, json.JSONDecodeError, TemplateError, SyntaxError) as exc:
            reason = re.sub(r"'[^']*'", "'…'", str(exc))[:60] or type(exc).__name__
            c["drop_invalid"] += 1
            c[f"invalid:{reason}"] += 1
            examples.setdefault(reason, json.dumps(raw, ensure_ascii=False, default=str)[:300])
            continue
        tool_names = sorted(t["function"]["name"] for t in row["tools"])
        dedup_key = sha(" ".join(WORD.findall(users[0].lower())) + "|" + ",".join(tool_names))
        rid = f"{src}:{rec_hash[:20]}"
        language = raw.get("lang", "en") if isinstance(raw, dict) else "en"
        data_rows.append({"id": rid, "source": src, "source_dataset": source_dataset, "license": license_,
                          "language": language, "category": category, "mode": mode,
                          "tools": json.dumps(row["tools"], ensure_ascii=False), "messages": to_storage(row["messages"]),
                          "dedup_key": dedup_key})
        tok_rows.append({"input_ids": ids, "assistant_masks": mask, "source": src, "language": language, "id": rid,
                         "supervised_tokens": len(labels), "category": category, "mode": mode, "dedup_key": dedup_key})
        c["kept"] += 1
        c[f"cat:{category}"] += 1
        c[f"mode:{mode}"] += 1
        c["tokens"] += len(ids)
    out = Path(cfg["work_dir"])
    name = unit["name"]
    if data_rows:
        pq.write_table(pa.Table.from_pylist(data_rows, schema=DATA_SCHEMA), out / f"data-{name}.parquet", compression="zstd")
        pq.write_table(pa.Table.from_pylist(tok_rows, schema=TOK_SCHEMA), out / f"tok-{name}.parquet", compression="zstd")
    return {"unit": name, "source": src, "counts": dict(c), "examples": examples}


# ---------------------------------------------------------------- planning

def plan_units(staging: Path) -> list[dict]:
    units = []
    for src, patterns in FILES.items():
        for pattern in patterns:
            for path in sorted(staging.glob(pattern)):
                size = path.stat().st_size
                if path.suffix == ".jsonl" and size > CHUNK_BYTES:
                    for i, start in enumerate(range(0, size, CHUNK_BYTES)):
                        units.append({"source": src, "path": str(path), "start": start,
                                      "end": min(start + CHUNK_BYTES, size), "bytes": min(CHUNK_BYTES, size - start),
                                      "name": f"{src}-{path.stem}-{i:03d}"})
                else:
                    units.append({"source": src, "path": str(path), "start": 0, "end": size, "bytes": size,
                                  "name": f"{src}-{path.stem}"})
    return units


def contamination_set(staging: Path) -> set[str]:
    grams: set[str] = set()

    def walk(x):
        if isinstance(x, str):
            grams.update(ngrams(x))
        elif isinstance(x, dict):
            for k, v in x.items():
                if k in ("content", "question", "query", "prompt") or isinstance(v, (list, dict)):
                    walk(v)
        elif isinstance(x, list):
            for v in x:
                walk(v)

    files = list((staging / "Berkeley-Function-Calling-Leaderboard").glob("*.json*")) + \
        list((staging / "When2Call" / "test").glob("*.jsonl"))
    for path in files:
        for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
            line = line.strip()
            if line.startswith("{"):
                try:
                    walk(json.loads(line))
                except json.JSONDecodeError:
                    pass
    return grams


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--staging", type=Path, required=True)
    ap.add_argument("--tokenizer", type=Path, required=True)
    ap.add_argument("--template", type=Path, required=True)
    ap.add_argument("--dataset-dir", type=Path, required=True)
    ap.add_argument("--pool-dir", type=Path, required=True)
    ap.add_argument("--work-dir", type=Path, required=True)
    ap.add_argument("--max-length", type=int, default=32768)
    ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args()
    for d in (a.dataset_dir / "data", a.dataset_dir / "samples", a.pool_dir, a.work_dir):
        d.mkdir(parents=True, exist_ok=True)
    cfg = {"tokenizer": str(a.tokenizer), "template": str(a.template), "work_dir": str(a.work_dir), "max_length": a.max_length}
    units = plan_units(a.staging)
    # Budgets -> keep probabilities from on-disk sizes (jsonl bytes ~ characters; parquet/json kept whole).
    size_by_src = Counter()
    for u in units:
        size_by_src[u["source"]] += u["bytes"]
    keep_prob = {}
    for src, (_, _, _, budget, _) in SOURCES.items():
        est_tokens = size_by_src[src] / CHARS_PER_TOKEN / 1.4  # JSON quoting/keys inflate bytes ~1.4x
        keep_prob[src] = 1.0 if budget == "all" or not est_tokens else min(1.0, budget / est_tokens)
    print("keep probabilities:", json.dumps({k: round(v, 4) for k, v in keep_prob.items()}), flush=True)
    contam = contamination_set(a.staging)
    print(f"contamination 13-grams: {len(contam)}", flush=True)
    results = []
    with ProcessPoolExecutor(a.workers) as pool:
        jobs = [pool.submit(build_unit, u, cfg, keep_prob, contam) for u in units]
        for i, f in enumerate(as_completed(jobs), 1):
            results.append(f.result())
            print(f"{i}/{len(jobs)} {results[-1]['unit']}: kept {results[-1]['counts'].get('kept', 0)}", flush=True)
    # Cross-source dedup in priority order, then write final shards.
    priority = {s: v[4] for s, v in SOURCES.items()}
    owner: dict[str, tuple] = {}
    for r in sorted(results, key=lambda r: (priority[r["source"]], r["unit"])):
        path = a.work_dir / f"data-{r['unit']}.parquet"
        if not path.exists():
            continue
        for i, key in enumerate(pq.read_table(path, columns=["dedup_key"])["dedup_key"].to_pylist()):
            owner.setdefault(key, (r["unit"], i))
    counts: dict[str, Counter] = {}
    samples: dict[str, list] = {}
    for r in results:
        src_c = counts.setdefault(r["source"], Counter())
        src_c.update({k: v for k, v in r["counts"].items() if not k.startswith(("cat:", "mode:", "tokens", "kept"))})
        path = a.work_dir / f"data-{r['unit']}.parquet"
        if not path.exists():
            continue
        data, toks = pq.read_table(path), pq.read_table(a.work_dir / f"tok-{r['unit']}.parquet")
        keep = [owner[k] == (r["unit"], i) for i, k in enumerate(data["dedup_key"].to_pylist())]
        src_c["drop_duplicate"] += keep.count(False)
        data, toks = data.filter(pa.array(keep)), toks.filter(pa.array(keep))
        if data.num_rows:
            pq.write_table(data.drop(["dedup_key"]), a.dataset_dir / "data" / f"train-{r['unit']}.parquet", compression="zstd")
            pq.write_table(toks.drop(["dedup_key"]), a.pool_dir / f"train-{r['unit']}.parquet", compression="zstd")
        lengths = [len(x) for x in toks["input_ids"].to_pylist()]
        src_c["kept"] += data.num_rows
        src_c["tokens"] += sum(lengths)
        for cat, mode, row in zip(data["category"].to_pylist(), data["mode"].to_pylist(), data.to_pylist()):
            src_c[f"cat:{cat}"] += 1
            src_c[f"mode:{mode}"] += 1
            bucket = samples.setdefault(cat, [])
            if len(bucket) < 50 and int(sha(row["id"])[:2], 16) < 64:
                row.pop("dedup_key", None)
                bucket.append(row)
    for cat, rows in samples.items():
        with open(a.dataset_dir / "samples" / f"{cat}.jsonl", "w", encoding="utf-8") as f:
            for row in rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
    totals = Counter()
    for c in counts.values():
        totals.update({k: v for k, v in c.items() if k in ("kept", "tokens") or k.startswith(("cat:", "mode:"))})
    tok = AutoTokenizer.from_pretrained(str(a.tokenizer), local_files_only=True)
    manifest = {"name": "openeurollm-tool-use", "version": "1.0.0", "template": str(a.template),
                "template_sha256": sha(Path(a.template).read_text(encoding="utf-8")),
                "tokenizer_vocab_size": len(tok), "keep_probability": keep_prob, "totals": dict(totals),
                "by_source": {k: dict(v) for k, v in sorted(counts.items())},
                "invalid_examples": {r["source"]: r["examples"] for r in results if r["examples"]}}
    (a.dataset_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n")
    (a.pool_dir / "manifest.json").write_text(json.dumps(dict(manifest, kind="stage3_tool_pool_pretokenized"), indent=2) + "\n")
    print(json.dumps({"totals": dict(totals)}, indent=1))
    for s, c in sorted(counts.items()):
        print(f"  {s:<24} kept={c.get('kept', 0):>7} tokens={c.get('tokens', 0)/1e6:7.1f}M "
              + " ".join(f"{k}={v}" for k, v in sorted(c.items()) if k.startswith(("drop_", "cat:", "mode:"))))


if __name__ == "__main__":
    main()
