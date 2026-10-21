#!/usr/bin/env python3
"""Count refusal answers and tag-wrapped answers per source in the Instruct dataset and the
Excellent reasoning rows (which the earlier reference SFT trained on, and Stage 1 excluded)."""
import json, re, sys
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import pyarrow.parquet as pq

REFUSAL = re.compile(r"\bI(?:'m| am) sorry,? but I (?:can't|cannot|won't)|\bI (?:can't|cannot|won't) (?:help|assist|provide|comply|create|write|support) (?:with )?(?:that|this|you)", re.I)
TAG = re.compile(r"^\s*<([A-Za-z_-]+)>")

def scan_instruct(path):
    c = {}
    for b in pq.ParquetFile(path).iter_batches(batch_size=4096, columns=["source", "messages"]):
        for r in b.to_pylist():
            a = [m["content"] for m in r["messages"] if m["role"] == "assistant"]
            k = c.setdefault(r["source"], Counter()); k["rows"] += 1
            k["refusal"] += any(REFUSAL.search(x[:400]) for x in a)
            m = TAG.match(a[-1]) if a else None
            if m: k["tag_" + m.group(1).lower()] += 1; k["tag_any"] += 1
    return c

def scan_excellent(path):
    c = {}
    for b in pq.ParquetFile(path).iter_batches(batch_size=1024, columns=["source", "messages"]):
        for r in b.to_pylist():
            last = r["messages"][-1]["content"] or ""
            if "</think>" not in last: continue
            ans = last.split("</think>", 1)[1]
            k = c.setdefault("reasoning:" + r["source"], Counter()); k["rows"] += 1
            k["refusal"] += bool(REFUSAL.search(ans[:400]))
    return c

def merge(parts):
    t = {}
    for p in parts:
        for s, k in p.items(): t.setdefault(s, Counter()).update(k)
    return t

if __name__ == "__main__":
    root = Path(sys.argv[1])
    inst = sorted((root / "data/openeurollm-instruct-sft-v1/data").glob("train-*.parquet"))
    exc = sorted((root / "excellent-sft-dataset/data").glob("train-*.parquet"))
    with ProcessPoolExecutor(8) as pool:
        a = merge(pool.map(scan_instruct, map(str, inst)))
        b = merge(pool.map(scan_excellent, map(str, exc)))
    out = {"instruct": {k: dict(v) for k, v in a.items()}, "excellent_reasoning": {k: dict(v) for k, v in b.items()}}
    Path(sys.argv[2]).write_text(json.dumps(out, indent=1))
    for name, d in out.items():
        rows = sum(v["rows"] for v in d.values()); ref = sum(v.get("refusal", 0) for v in d.values()); tags = sum(v.get("tag_any", 0) for v in d.values())
        print(f"== {name}: rows={rows} refusal={ref} tag_wrapped={tags}")
        for s, v in sorted(d.items(), key=lambda x: -(x[1].get("refusal", 0) + x[1].get("tag_any", 0)))[:14]:
            tg = sorted(((k[4:], n) for k, n in v.items() if k.startswith("tag_") and k != "tag_any"), key=lambda x: -x[1])[:4]
            print(f"   {s.replace('openeurollm/OpenEuroLLM-Excellent-SFT:', '')[:60]:<60} rows={v['rows']:>7} refusal={v.get('refusal', 0):>5} tags={v.get('tag_any', 0):>5} {tg}")
