#!/usr/bin/env python3
"""Multilingual slice of the A4b tool-RL pool (W2): translate the user turns of sampled rows into one European
language per row (hash-assigned), keeping the system turn (tools, policy), earlier assistant turns and tool results
verbatim. The gold action does not change: a call is scored on tool name and arguments, a reply on not calling.

Protected strings: every string or number in the gold call's arguments that occurs verbatim in the original user text
must also occur verbatim in the translation, otherwise the row is dropped (the request would no longer determine the
call). Each translation must be identified as the target language (langid, close pairs accepted). Resumable via a
JSONL work file. Translator: a local Ollama model (default gemma4:31b, as for the Stage-3 slice), think off.
"""
from __future__ import annotations

import argparse
import collections
import hashlib
import json
import random
import threading
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import langid
import pyarrow as pa
import pyarrow.parquet as pq

LANGS = {"cs": "Czech", "de": "German", "el": "Greek", "es": "Spanish", "fi": "Finnish", "fr": "French", "it": "Italian",
         "nl": "Dutch", "pl": "Polish", "ro": "Romanian", "sv": "Swedish", "uk": "Ukrainian", "da": "Danish",
         "no": "Norwegian (Bokmål)", "is": "Icelandic", "pt": "Portuguese (European)"}
LANGID_OK = {"no": {"no", "nb", "nn", "da"}, "da": {"da", "no", "nb"}}
SYSTEM = ("You are a professional translator. Translate every value of the JSON object from English into {lang}. "
          "Keep the JSON keys unchanged and return only the JSON object. Translate naturally and idiomatically, as a "
          "native speaker would write to a customer-service or assistant chat. Preserve numbers, dates, IDs, codes, "
          "URLs, e-mail addresses and names of products, people and places. Keep these exact strings verbatim: {protected}")


def language_for(row_id: str) -> str:
    keys = sorted(LANGS)
    return keys[int(hashlib.sha256((row_id + "#rl-tool-lang").encode()).hexdigest()[:8], 16) % len(keys)]


def protected_strings(gold: list[dict], user_text: str) -> list[str]:
    found = set()

    def walk(value):
        if isinstance(value, dict):
            for v in value.values():
                walk(v)
        elif isinstance(value, list):
            for v in value:
                walk(v)
        elif isinstance(value, bool) or value is None:
            return
        else:
            text = str(value)
            if len(text) >= 2 and text in user_text:
                found.add(text)

    for call in gold:
        walk(call.get("arguments") or {})
    return sorted(found, key=len, reverse=True)


class Translator:
    def __init__(self, endpoints: list[str], model: str):
        self.endpoints, self.model, self.lock, self.i = endpoints, model, threading.Lock(), 0

    def __call__(self, payload: dict, lang: str, protected: list[str]) -> dict:
        with self.lock:
            endpoint = self.endpoints[self.i % len(self.endpoints)]
            self.i += 1
        body = {"model": self.model, "stream": False, "think": False, "format": "json",
                "options": {"temperature": 0.2, "seed": 0, "num_ctx": 8192},
                "messages": [{"role": "system", "content": SYSTEM.format(lang=LANGS[lang], protected=json.dumps(protected, ensure_ascii=False))},
                             {"role": "user", "content": json.dumps(payload, ensure_ascii=False)}]}
        req = urllib.request.Request(endpoint + "/api/chat", data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=600) as resp:
            return json.loads(json.loads(resp.read())["message"]["content"])


def translate_row(row: dict, translate: Translator) -> tuple[dict | None, str]:
    lang = language_for(row["id"])
    gold = json.loads(row["ground_truth"])["calls"]
    users = [k for k, m in enumerate(row["messages"]) if m["role"] == "user"]
    if not users:
        return None, "no_user_turn"
    payload = {f"u{k}": row["messages"][k]["content"] for k in users}
    user_text = "\n".join(payload.values())
    protected = protected_strings(gold, user_text)
    try:
        out = translate(payload, lang, protected)
    except Exception as error:  # network / JSON
        return None, f"translator_error:{type(error).__name__}"
    if set(out) != set(payload) or not all(isinstance(v, str) and v.strip() for v in out.values()):
        return None, "bad_json_shape"
    joined = "\n".join(out[k] for k in payload)
    if any(p not in joined for p in protected):
        return None, "protected_string_lost"
    ratio = len(joined) / max(1, len(user_text))
    if not 0.4 <= ratio <= 2.5:
        return None, "length_ratio"
    detected, _ = langid.classify(joined)
    if detected not in LANGID_OK.get(lang, {lang}):
        return None, f"langid:{detected}"
    messages = [dict(m) for m in row["messages"]]
    for k in users:
        messages[k]["content"] = out[f"u{k}"]
    return {**row, "id": row["id"] + "-" + lang, "messages": messages, "language": lang}, "ok"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pool", type=Path, required=True, help="A4b train.parquet")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--endpoints", required=True)
    ap.add_argument("--model", default="gemma4:31b")
    ap.add_argument("--model-revision", default="")
    ap.add_argument("--rows", type=int, default=640)
    ap.add_argument("--threads", type=int, default=32)
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    table = pq.read_table(a.pool)
    rows = table.to_pylist()
    rng = random.Random(20261026)
    rows = sorted(rows, key=lambda r: r["id"])
    rng.shuffle(rows)
    rows = rows[: a.rows]
    work = a.out / "work.jsonl"
    done = {json.loads(l)["source_id"]: json.loads(l) for l in work.open()} if work.exists() else {}
    translate = Translator(a.endpoints.split(","), a.model)
    todo = [r for r in rows if r["id"] not in done]
    with ThreadPoolExecutor(a.threads) as pool, work.open("a") as sink:
        futures = {pool.submit(translate_row, r, translate): r for r in todo}
        for future in as_completed(futures):
            row = futures[future]
            result, status = future.result()
            record = {"source_id": row["id"], "status": status, "row": result}
            sink.write(json.dumps(record, ensure_ascii=False) + "\n"); sink.flush()
            done[row["id"]] = record
    kept = [rec["row"] for rec in done.values() if rec["status"] == "ok"]
    status = collections.Counter(rec["status"] for rec in done.values())
    schema = table.schema
    pq.write_table(pa.Table.from_pylist([{k: r[k] for k in schema.names} for r in kept], schema=schema), a.out / "train.parquet")
    sha = hashlib.sha256((a.out / "train.parquet").read_bytes()).hexdigest()
    manifest = {"kind": "rl_tool_pool_translated", "source_pool": str(a.pool), "rows_sampled": len(rows), "rows_kept": len(kept),
                "status": dict(status), "languages": dict(collections.Counter(r["language"] for r in kept)),
                "translator": {"model": a.model, "revision": a.model_revision, "think": False, "temperature": 0.2},
                "sha256": sha}
    (a.out / "manifest.json").write_text(json.dumps(manifest, indent=1, ensure_ascii=False) + "\n")
    print(json.dumps(manifest, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
