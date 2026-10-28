#!/usr/bin/env python3
"""Prompts for on-policy identity DPO (behaviour "identity" in scripts/judge_onpolicy_dpo.py).

Input: a local copy of the OpenEuroLLM model-identity dataset (v2: data/train.jsonl + identity_facts.json). Every
distinct first user turn of the TRAIN split (per language) becomes one prompt; its first assistant turn is the verified
reference answer (used as `chosen` when none of the policy's own answers is judged correct). The validation split is
never used, so it stays an independent identity test. The judge receives every fact statement of the profile.

    python scripts/build_identity_dpo_prompts.py --identity ../openeurollm-model-identity --out <work>/prompts.jsonl
"""
from __future__ import annotations

import argparse
import collections
import hashlib
import json
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--identity", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    facts = json.loads((a.identity / "identity_facts.json").read_text())
    reference = [f["statement"] + (f" (as of {f['as_of']})" if f.get("as_of") else "") for f in facts["facts"]]
    seen, rows = set(), []
    for line in (a.identity / "data" / "train.jsonl").open(encoding="utf-8"):
        r = json.loads(line)
        prompt, answer = r["messages"][0]["content"], r["messages"][1]["content"]
        key = (r["language"], prompt)
        if key in seen:
            continue
        seen.add(key)
        pid = hashlib.sha256(f"{r['language']}|{prompt}".encode()).hexdigest()[:16]
        rows.append({"id": f"id-{r['language']}-{pid}", "prompt": prompt, "behaviour": "identity", "category": r["category"],
                     "language": r["language"], "reference": reference, "reference_answer": answer,
                     "source": f"openeurollm-model-identity@{facts['version']}:{r['category']}"})
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")
    print(json.dumps({"prompts": len(rows), "by_language": dict(collections.Counter(r["language"] for r in rows)),
                      "by_category": dict(collections.Counter(r["category"] for r in rows))}, indent=1))


if __name__ == "__main__":
    main()
