#!/usr/bin/env python3
"""Multilingual slice for openeurollm-tool-use: translate user/system turns and assistant text of
selected direct-mode rows; never tool names, schemas, call arguments or tool outputs.

Each source row is translated into exactly one target language (hash-assigned). Strings and
numbers that appear in tool-call arguments and in the original text are protected: the
translation must keep them verbatim, otherwise the call would no longer match the request.
Rows are validated (JSON shape, protected strings, language ID, length ratio), then rendered with
template v2.1 and token-audited like the main build. Resumable: finished rows are appended to
per-language JSONL files and skipped on restart.
"""
from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
import random
import re
import threading
import urllib.request
from pathlib import Path

import langid
import pyarrow as pa
import pyarrow.parquet as pq
from transformers import AutoTokenizer

LANGS = {"cs": "Czech", "de": "German", "el": "Greek", "es": "Spanish", "fi": "Finnish", "fr": "French", "it": "Italian",
         "nl": "Dutch", "pl": "Polish", "ro": "Romanian", "sv": "Swedish", "uk": "Ukrainian", "da": "Danish",
         "no": "Norwegian (Bokmål)", "is": "Icelandic", "pt": "Portuguese (European)"}
LANGID_OK = {"no": {"no", "nb", "nn", "da"}, "da": {"da", "no", "nb"}}  # close pairs are confused by langid
CATEGORY_WEIGHTS = {"single_call": 0.30, "parallel_call": 0.10, "multi_turn": 0.15, "no_call_direct": 0.15,
                    "ask_for_info": 0.15, "cannot_answer": 0.15}
FORBIDDEN_LABEL_IDS = (18, 15, 13)
SYSTEM = ("You are a professional translator. Translate every value of the JSON object from English into {lang}. "
          "Keep the JSON keys unchanged and return only the JSON object. Translate naturally and idiomatically. "
          "Preserve Markdown, code, numbers, URLs, e-mail addresses and placeholders. Do not translate function or "
          "tool names. Keep these exact strings verbatim wherever they occur: {protected}")


def sha(x: str) -> str:
    return hashlib.sha256(x.encode("utf-8")).hexdigest()


def protected_strings(row) -> list[str]:
    values = []

    def walk(v):
        if isinstance(v, dict):
            for x in v.values():
                walk(x)
        elif isinstance(v, list):
            for x in v:
                walk(x)
        elif isinstance(v, bool) or v is None:
            return
        elif isinstance(v, (int, float, str)):
            text = str(v).strip()
            if text and (len(text) >= 2 or text.isdigit()):
                values.append(text)

    for m in row["messages"]:
        for c in m.get("tool_calls") or []:
            walk(json.loads(c["function"]["arguments"]))
    return sorted(set(values), key=len, reverse=True)


def segments(row) -> dict[str, str]:
    return {f"s{i}": m["content"] for i, m in enumerate(row["messages"])
            if m["role"] in ("system", "user", "assistant") and (m.get("content") or "").strip()}


def select(rows, per_lang: int, seed: int) -> list[tuple[dict, str]]:
    langs = sorted(LANGS)
    pools: dict[tuple[str, str], list] = {}
    for r in rows:
        if r["mode"] != "direct" or r["category"] not in CATEGORY_WEIGHTS:
            continue
        seg = segments(r)
        if not seg or sum(len(v) for v in seg.values()) > 2000 or sum(m["role"] == "user" for m in r["messages"]) > 3:
            continue
        lang = langs[int(sha(r["id"])[:8], 16) % len(langs)]  # one language per source row
        pools.setdefault((lang, r["category"]), []).append(r)
    rng = random.Random(seed)
    chosen = []
    for lang in langs:
        for cat, w in CATEGORY_WEIGHTS.items():
            pool = sorted(pools.get((lang, cat), []), key=lambda r: r["id"])
            rng.shuffle(pool)
            chosen += [(r, lang) for r in pool[: round(per_lang * w)]]
    return chosen


def call_ollama(endpoint: str, model: str, lang: str, protected: list[str], seg: dict[str, str]) -> dict:
    body = {"model": model, "stream": False, "format": "json", "think": False, "options": {"temperature": 0, "num_ctx": 8192},
            "messages": [{"role": "system", "content": SYSTEM.format(lang=LANGS[lang], protected=json.dumps(protected, ensure_ascii=False))},
                         {"role": "user", "content": json.dumps(seg, ensure_ascii=False)}]}
    req = urllib.request.Request(f"{endpoint}/api/chat", data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=600) as resp:
        return json.loads(json.loads(resp.read())["message"]["content"])


def validate(lang: str, seg: dict, out: dict, protected: list[str]) -> str | None:
    if not isinstance(out, dict) or set(out) != set(seg) or not all(isinstance(v, str) and v.strip() for v in out.values()):
        return "shape"
    for k, src in seg.items():
        for p in protected:
            if p in src and p not in out[k]:
                return "protected_string_lost"
        ratio = len(out[k]) / max(1, len(src))
        if len(src) > 30 and not 0.5 <= ratio <= 2.5:
            return "length_ratio"
        letters = re.sub(r"`[^`]*`|```.*?```|https?://\S+|[\d\W_]+", " ", out[k], flags=re.S)
        if len(letters.strip()) >= 40:
            code = langid.classify(letters)[0]
            if code not in LANGID_OK.get(lang, {lang}):
                return f"wrong_language:{code}"
    return None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset-dir", type=Path, required=True)
    ap.add_argument("--pool-dir", type=Path, required=True)
    ap.add_argument("--work-dir", type=Path, required=True)
    ap.add_argument("--tokenizer", type=Path, required=True)
    ap.add_argument("--template", type=Path, required=True)
    ap.add_argument("--endpoints", required=True, help="comma-separated Ollama base URLs")
    ap.add_argument("--model", default="gemma4:31b")
    ap.add_argument("--model-revision", default="")
    ap.add_argument("--per-lang", type=int, default=900)
    ap.add_argument("--threads", type=int, default=32)
    ap.add_argument("--limit", type=int, default=0, help="smoke: translate only this many rows")
    ap.add_argument("--finalize", action="store_true", help="write parquet + pool and add translation columns")
    a = ap.parse_args()
    a.work_dir.mkdir(parents=True, exist_ok=True)
    files = sorted((a.dataset_dir / "data").glob("train-*.parquet"))
    rows = [r for f in files if "translated" not in f.name for r in pq.read_table(f).to_pylist()]
    chosen = select(rows, a.per_lang, seed=20261021)
    if a.limit:
        chosen = chosen[:: max(1, len(chosen) // a.limit)][: a.limit]
    done = set()
    for p in a.work_dir.glob("*.jsonl"):
        done |= {json.loads(l)["source_id"] for l in p.read_text().splitlines() if l.strip()}
    todo = [(r, lang) for r, lang in chosen if r["id"] not in done]
    print(f"selected {len(chosen)}; already done {len(chosen) - len(todo)}; to translate {len(todo)}", flush=True)
    endpoints = a.endpoints.split(",")
    lock = threading.Lock()
    stats = Counter()

    def work(i, row, lang):
        seg, prot = segments(row), protected_strings(row)
        try:
            out = call_ollama(endpoints[i % len(endpoints)], a.model, lang, prot, seg)
            reason = validate(lang, seg, out, prot)
        except Exception as exc:  # network / JSON errors are recorded, not fatal
            out, reason = None, f"error:{type(exc).__name__}"
        rec = {"source_id": row["id"], "lang": lang, "ok": reason is None, "reason": reason, "segments": out if reason is None else None}
        with lock:
            stats["ok" if reason is None else reason.split(":")[0]] += 1
            with open(a.work_dir / f"{lang}.jsonl", "a", encoding="utf-8") as f:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            n = sum(stats.values())
            if n % 200 == 0:
                print(f"{n}/{len(todo)} {dict(stats)}", flush=True)

    with ThreadPoolExecutor(a.threads) as pool:
        for f in as_completed([pool.submit(work, i, r, lang) for i, (r, lang) in enumerate(todo)]):
            f.result()
    print("translation pass:", dict(stats), flush=True)
    if not a.finalize:
        return

    # ---- finalize: build translated rows, render + audit, write dataset and pool shards
    tok = AutoTokenizer.from_pretrained(str(a.tokenizer), local_files_only=True)
    tok.chat_template = a.template.read_text(encoding="utf-8")
    by_id = {r["id"]: r for r in rows}
    data_schema = pq.read_schema(files[0]).append(pa.field("translation_model", pa.string()))
    pool_schema = pq.read_schema(next(a.pool_dir.glob("train-*.parquet")))
    out_rows, tok_rows, audit = [], [], Counter()
    stats = Counter()  # recompute over all passes (the run may have been resumed)
    for p in a.work_dir.glob("*.jsonl"):
        for line in p.read_text().splitlines():
            rec = json.loads(line)
            stats["ok" if rec["ok"] else rec["reason"].split(":")[0]] += 1
    model_tag = f"{a.model}@{a.model_revision}" if a.model_revision else a.model
    for p in sorted(a.work_dir.glob("*.jsonl")):
        for line in p.read_text().splitlines():
            rec = json.loads(line)
            if not rec["ok"]:
                continue
            src = by_id[rec["source_id"]]
            msgs = [dict(m) for m in src["messages"]]
            for k, text in rec["segments"].items():
                msgs[int(k[1:])]["content"] = text
            render_msgs = []
            for m in msgs:
                mm = {"role": m["role"], "content": m["content"] or ""}
                if m.get("tool_calls"):
                    mm["tool_calls"] = [{"type": "function", "function": {"name": c["function"]["name"],
                                         "arguments": json.loads(c["function"]["arguments"])}} for c in m["tool_calls"]]
                render_msgs.append(mm)
            tools = json.loads(src["tools"])
            enc = tok.apply_chat_template(render_msgs, tools=tools or None, tokenize=True, return_dict=True,
                                          return_assistant_tokens_mask=True)
            ids, mask = list(enc["input_ids"]), list(enc["assistant_masks"])
            labels = [t for t, m in zip(ids, mask) if m]
            n_calls = sum(len(m.get("tool_calls") or []) for m in render_msgs)
            if any(t in labels for t in FORBIDDEN_LABEL_IDS) or labels.count(11) != n_calls or len(ids) > 32768:
                audit["drop_audit"] += 1
                continue
            rid = f"{src['id']}:tr-{rec['lang']}"
            out_rows.append({**src, "id": rid, "source": f"{src['source']}+translated", "language": rec["lang"],
                             "license": f"{src['license']}; translated text: {a.model} output",
                             "messages": msgs, "translation_model": model_tag})
            tok_rows.append({"input_ids": ids, "assistant_masks": mask, "source": f"{src['source']}+translated",
                             "language": rec["lang"], "id": rid, "supervised_tokens": len(labels),
                             "category": src["category"], "mode": src["mode"]})
            audit[f"lang:{rec['lang']}"] += 1
    pq.write_table(pa.Table.from_pylist(out_rows, schema=data_schema), a.dataset_dir / "data" / "train-translated.parquet", compression="zstd")
    pq.write_table(pa.Table.from_pylist(tok_rows, schema=pool_schema), a.pool_dir / "train-translated.parquet", compression="zstd")
    for f in files:  # add the (null) translation column to original shards so all files share one schema
        t = pq.read_table(f)
        if "translation_model" not in t.column_names:
            pq.write_table(t.append_column("translation_model", pa.nulls(t.num_rows, pa.string())), f, compression="zstd")
    for d in (a.dataset_dir, a.pool_dir):
        m = json.loads((d / "manifest.json").read_text())
        m["translation"] = {"model": model_tag, "rows": len(out_rows), "requested": len(chosen),
                            "pass_stats": dict(stats), "audit": dict(audit), "languages": sorted(LANGS)}
        m["totals"]["kept"] = m["totals"].get("kept", 0) + len(out_rows)
        (d / "manifest.json").write_text(json.dumps(m, indent=2, ensure_ascii=False) + "\n")
    print("finalized:", len(out_rows), dict(audit), flush=True)


if __name__ == "__main__":
    main()
