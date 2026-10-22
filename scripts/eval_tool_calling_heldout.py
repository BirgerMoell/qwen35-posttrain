#!/usr/bin/env python3
"""Held-out tool-calling check for a Stage-3 checkpoint (default mode, greedy).

Rows of openeurollm-tool-use-v1 whose id falls in the Stage-3 validation hash bucket (sha256(id + "#val")
< 0.005, see build_oellm_stage3_mix.py) were never trained on. For each sampled row the prompt is the
conversation up to the FIRST assistant turn, so the model makes the row's first decision: call (which
tools, with what arguments) or answer without calling (ask for missing info, refuse, answer directly).

Per category and overall: decision accuracy (call vs no call), tool-name set match, JSON validity, and
argument validity (object, all required keys, no undeclared keys). `run --rank R --world W` shards over
GPUs; `score` merges.
"""
from __future__ import annotations

import argparse
import glob
import hashlib
import json
import random
import re
from collections import defaultdict
from pathlib import Path

CALL = re.compile(r"<tool_call>\s*(.*?)\s*</tool_call>", re.S)


def h(x: str) -> float:
    return int(hashlib.sha256(x.encode()).hexdigest()[:16], 16) / 2**64


def prepare(a) -> None:
    import pyarrow.parquet as pq
    rows = []
    for f in sorted(glob.glob(f"{a.dataset}/data/*.parquet")):
        t = pq.read_table(f, columns=["id", "category", "tools", "messages", "language", "mode"]).to_pylist()
        rows += [r for r in t if h(r["id"] + "#val") < 0.005]
    by = defaultdict(list)
    for r in rows:
        first = next((i for i, m in enumerate(r["messages"]) if m["role"] == "assistant"), None)
        if first is None:
            continue
        gold = r["messages"][first]
        by[r["category"]].append({"id": r["id"], "category": r["category"], "language": r["language"],
                                  "tools": r["tools"], "messages": r["messages"][:first],
                                  "gold_calls": [c["function"]["name"] for c in (gold.get("tool_calls") or [])]})
    rng = random.Random(1234)
    out = []
    for cat in sorted(by):
        out += rng.sample(by[cat], min(a.per_category, len(by[cat])))
    a.out.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in out))
    print(f"held-out rows {len(rows)}; wrote {len(out)}:", {c: min(a.per_category, len(v)) for c, v in sorted(by.items())})


def run(a) -> None:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    rows = [json.loads(l) for l in a.data.read_text().splitlines()][a.rank::a.world]
    tok = AutoTokenizer.from_pretrained(a.model)
    tok.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(a.model, dtype=torch.bfloat16, attn_implementation="flash_attention_2",
                                                 device_map="cuda").eval()
    end = tok.convert_tokens_to_ids("<|im_end|>")
    texts = [tok.apply_chat_template(r["messages"], tools=json.loads(r["tools"]) if r["tools"] else None, tokenize=False,
                                     add_generation_prompt=True, enable_thinking=False) for r in rows]
    order = sorted(range(len(rows)), key=lambda i: len(texts[i]))
    out = []
    for b in range(0, len(order), a.batch):
        idx = order[b:b + a.batch]
        enc = tok([texts[i] for i in idx], return_tensors="pt", padding=True, add_special_tokens=False).to("cuda")
        with torch.no_grad():
            gen = model.generate(**enc, max_new_tokens=1024, do_sample=False, eos_token_id=end, pad_token_id=tok.pad_token_id)
        for i, seq in zip(idx, gen[:, enc["input_ids"].shape[1]:].tolist()):
            n = seq.index(end) + 1 if end in seq else len(seq)
            out.append({**rows[i], "raw": tok.decode(seq[:n], skip_special_tokens=False), "terminated": end in seq})
        print(f"rank {a.rank}: {len(out)}/{len(rows)}", flush=True)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in out))


def judge(r: dict) -> dict:
    tools = {t["function"]["name"]: t["function"].get("parameters") or {} for t in json.loads(r["tools"] or "[]")}
    blocks = CALL.findall(r["raw"])
    calls, json_ok, args_ok = [], True, True
    for b in blocks:
        try:
            c = json.loads(b)
            name, args = c.get("name"), c.get("arguments", {})
            calls.append(name)
            spec = tools.get(name)
            if spec is None or not isinstance(args, dict):
                args_ok = False
                continue
            props, req = spec.get("properties") or {}, spec.get("required") or []
            if any(k not in args for k in req) or (props and any(k not in props for k in args)):
                args_ok = False
        except (json.JSONDecodeError, AttributeError):
            json_ok = args_ok = False
    should_call = bool(r["gold_calls"])
    return {"decision": bool(blocks) == should_call,
            "names": sorted(calls) == sorted(r["gold_calls"]) if should_call else not blocks,
            "json_ok": json_ok if blocks else None, "args_ok": args_ok if blocks else None,
            "think_leak": "<think>" in r["raw"] or "</think>" in r["raw"], "terminated": r["terminated"]}


def score(a) -> None:
    rows = [json.loads(l) for p in sorted(a.results.glob("*.jsonl")) for l in p.read_text().splitlines()]
    agg = defaultdict(list)
    for r in rows:
        j = judge(r)
        for key in ("ALL", r["category"], "lang:en" if r["language"] == "en" else "lang:translated"):
            agg[key].append(j)
    report = {}
    print(f"{'slice':<20}{'n':>5}{'decision':>10}{'names':>8}{'json':>7}{'args':>7}{'think':>7}{'term':>7}")
    for key in sorted(agg, key=lambda k: (k != "ALL", k.startswith("lang:"), k)):
        v = agg[key]
        rate = lambda f: (sum(x[f] for x in v if x[f] is not None) / max(1, sum(x[f] is not None for x in v)))
        m = {"n": len(v), **{f: rate(f) for f in ("decision", "names", "json_ok", "args_ok", "think_leak", "terminated")}}
        report[key] = m
        print(f"{key:<20}{m['n']:>5}{m['decision']:>10.3f}{m['names']:>8.3f}{m['json_ok']:>7.3f}{m['args_ok']:>7.3f}"
              f"{m['think_leak']:>7.3f}{m['terminated']:>7.3f}")
    a.out.write_text(json.dumps(report, indent=2) + "\n")


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("prepare")
    p.add_argument("--dataset", type=Path, required=True); p.add_argument("--out", type=Path, required=True)
    p.add_argument("--per-category", type=int, default=50)
    r = sub.add_parser("run")
    for k, t in (("--model", str), ("--data", Path), ("--out", Path)):
        r.add_argument(k, type=t, required=True)
    r.add_argument("--rank", type=int, default=0); r.add_argument("--world", type=int, default=1)
    r.add_argument("--batch", type=int, default=8)
    s = sub.add_parser("score"); s.add_argument("--results", type=Path, required=True); s.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    {"prepare": lambda: prepare(a), "run": lambda: run(a), "score": lambda: score(a)}[a.cmd]()


if __name__ == "__main__":
    main()
