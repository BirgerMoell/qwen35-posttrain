#!/usr/bin/env python3
"""Generate answers to ArenaHard-EU prompts for a model — step 1 of the EU win-rate eval.

Reads ArenaHard prompt files ({question_id, turns:[{content}]}), applies the model's chat
template, generates an answer per prompt, and writes {question_id, prompt, answer} jsonl.
Run this for the candidate (SFT/DPO) and the baseline; then arena_judge.py compares them.

Usage:
  python scripts/arena_generate.py \
    --model /path/to/checkpoint \
    --prompts /scratch/.../ArenaHard/eng/train.jsonl,/scratch/.../ArenaHard/fin/train.jsonl \
    --out eval/results/arena/<tag>.jsonl [--max-new-tokens 1024] [--limit N]
"""
from __future__ import annotations

import argparse
import json
import pathlib

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def read_prompts(paths: list[str], limit: int | None):
    rows = []
    for p in paths:
        for line in open(p):
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            turns = d.get("turns") or []
            if not turns:
                continue
            rows.append({
                "question_id": d.get("question_id"),
                "lang": pathlib.Path(p).parent.name,
                "prompt": turns[0]["content"],
            })
            if limit and len(rows) >= limit:
                return rows
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--prompts", required=True, help="comma-separated jsonl prompt files")
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-new-tokens", type=int, default=1024)
    ap.add_argument("--limit", type=int, default=None)
    a = ap.parse_args()

    rows = read_prompts([p.strip() for p in a.prompts.split(",") if p.strip()], a.limit)
    print(f"[arena] {len(rows)} prompts; loading {a.model}")

    tok = AutoTokenizer.from_pretrained(a.model, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        a.model, dtype=torch.bfloat16, device_map="auto", trust_remote_code=True
    )
    model.eval()

    out_path = pathlib.Path(a.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        for i, r in enumerate(rows):
            messages = [{"role": "user", "content": r["prompt"]}]
            inputs = tok.apply_chat_template(
                messages, add_generation_prompt=True, return_tensors="pt", return_dict=True,
            ).to(model.device)
            with torch.no_grad():
                out = model.generate(
                    **inputs, max_new_tokens=a.max_new_tokens, do_sample=False,
                    pad_token_id=tok.pad_token_id or tok.eos_token_id,
                )
            answer = tok.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
            f.write(json.dumps({**r, "answer": answer}, ensure_ascii=False) + "\n")
            if (i + 1) % 25 == 0:
                print(f"[arena] {i+1}/{len(rows)}")
    print(f"[arena] wrote {out_path}")


if __name__ == "__main__":
    main()
