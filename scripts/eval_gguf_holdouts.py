#!/usr/bin/env python3
"""Evaluate a QUANTIZED GGUF on the EU holdouts via a llama-server OpenAI endpoint.

Measures quantization quality retention: same rows + same deterministic scorers as
scripts/eval_eu_holdouts.py, but answers come from a running `llama-server` serving the GGUF
(so it exercises the actual on-device weights AND the embedded chat template — no-think default).

Start the server first (in the container), then point this at it:
  llama-server -m model-Q4_K_M.gguf -c 4096 --port 8080 &
  python scripts/eval_gguf_holdouts.py --base-url http://127.0.0.1:8080 \
    --data data/eval_holdouts/oellm-eu-eval-holdouts-v1/data/dev.jsonl --out eval/results/eu-holdouts/<tag>-q4.json
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys
import urllib.request

# reuse the exact scorers from the full-precision evaluator
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from eval_eu_holdouts import score_row  # noqa: E402


def user_text(r):
    return (r.get("context", "") + "\n\n" + r["prompt"]) if r.get("context") else r["prompt"]


def chat(base_url, content, max_tokens, timeout=300):
    body = json.dumps({
        "messages": [{"role": "user", "content": content}],
        "max_tokens": max_tokens, "temperature": 0.0, "stream": False,
    }).encode()
    req = urllib.request.Request(base_url.rstrip("/") + "/v1/chat/completions",
                                 data=body, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        msg = json.loads(r.read())["choices"][0]["message"]
    # llama-server splits <think> output into reasoning_content (reasoning_format=deepseek);
    # concatenate so scoring sees the full answer whether the model thinks or not.
    return ((msg.get("reasoning_content") or "") + "\n" + (msg.get("content") or "")).strip()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", required=True, help="llama-server root, e.g. http://127.0.0.1:8080")
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--tag", default="gguf-q4")
    ap.add_argument("--max-new-tokens", type=int, default=512)
    ap.add_argument("--limit-per-bucket", type=int, default=None)
    ap.add_argument("--exclude-buckets", default="long_context_retrieval")
    ap.add_argument("--buckets", default=None)
    a = ap.parse_args()

    excl = {b.strip() for b in (a.exclude_buckets or "").split(",") if b.strip()}
    incl = {b.strip() for b in a.buckets.split(",")} if a.buckets else None
    rows = [json.loads(l) for l in open(a.data) if l.strip()]
    rows = [r for r in rows if r["bucket"] not in excl and (incl is None or r["bucket"] in incl)]
    if a.limit_per_bucket:
        seen, kept = {}, []
        for r in rows:
            b = r["bucket"]
            if seen.get(b, 0) < a.limit_per_bucket:
                kept.append(r); seen[b] = seen.get(b, 0) + 1
        rows = kept
    print(f"[gguf-holdouts] {len(rows)} rows via {a.base_url}", flush=True)

    by_bucket, by_lang = {}, {}
    for n, r in enumerate(rows, 1):
        try:
            ans = chat(a.base_url, user_text(r), a.max_new_tokens)
        except Exception as e:
            ans = ""
            print(f"[gguf-holdouts] req {n} failed: {e}", flush=True)
        ok = score_row(ans, r)
        by_bucket.setdefault(r["bucket"], []).append(ok)
        by_lang.setdefault(r.get("language", "?"), []).append(ok)
        if n % 50 == 0:
            print(f"[gguf-holdouts] {n}/{len(rows)}", flush=True)

    def acc(d):
        return {k: round(100 * sum(v) / len(v), 1) for k, v in sorted(d.items())}

    overall = sum(sum(v) for v in by_bucket.values())
    total = sum(len(v) for v in by_bucket.values())
    report = {
        "model": a.tag, "quantized": True, "n": total,
        "overall_acc": round(100 * overall / total, 1) if total else 0,
        "by_bucket": acc(by_bucket), "by_language": acc(by_lang),
    }
    pathlib.Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    json.dump(report, open(a.out, "w"), indent=2)
    print(json.dumps({k: report[k] for k in ("n", "overall_acc", "by_bucket")}, indent=2))
    print(f"[gguf-holdouts] overall {report['overall_acc']}%  -> {a.out}")


if __name__ == "__main__":
    main()
