#!/usr/bin/env python3
"""Pairwise LLM-judge for ArenaHard-EU: candidate vs baseline -> win-rate.

Reads two answer files (from arena_generate.py) keyed by question_id, and for each prompt
asks a judge model which answer is better. Judges both orderings (A/B and B/A) to cancel
position bias. Reports win / tie / loss counts and the candidate win-rate, overall and
per language.

Judge model runs locally (no internet on LUMI compute nodes). Default judge is a strong
local model; pass --judge to override. Self-judging (judge == candidate base) is a weak
but workable start — swap in a larger neutral judge when available.

Usage:
  python scripts/arena_judge.py \
    --candidate eval/results/arena/sft.jsonl \
    --baseline  eval/results/arena/base.jsonl \
    --judge /scratch/.../models/Qwen3.5-9B \
    --out eval/results/arena/sft_vs_base.json
"""
from __future__ import annotations

import argparse
import json
import pathlib
import re

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

JUDGE_TEMPLATE = """You are an impartial judge evaluating two AI assistant answers to a user question. \
Pick the answer that is more helpful, correct, and well-written. If the question is in a \
non-English language, the answer should be in that same language and fluent.

[User question]
{prompt}

[Answer A]
{a}

[Answer B]
{b}

Respond with EXACTLY one token: "A" if Answer A is better, "B" if Answer B is better, or \
"tie" if they are equal. Verdict:"""


def load(path):
    by_id = {}
    for line in open(path):
        line = line.strip()
        if line:
            d = json.loads(line)
            by_id[d["question_id"]] = d
    return by_id


def verdict(judge_model, tok, prompt, a_ans, b_ans) -> str:
    text = JUDGE_TEMPLATE.format(prompt=prompt[:4000], a=a_ans[:4000], b=b_ans[:4000])
    ids = tok.apply_chat_template(
        [{"role": "user", "content": text}], add_generation_prompt=True, return_tensors="pt"
    ).to(judge_model.device)
    with torch.no_grad():
        out = judge_model.generate(ids, max_new_tokens=4, do_sample=False,
                                   pad_token_id=tok.pad_token_id or tok.eos_token_id)
    resp = tok.decode(out[0][ids.shape[1]:], skip_special_tokens=True).strip().lower()
    if resp.startswith("a"):
        return "A"
    if resp.startswith("b"):
        return "B"
    return "tie"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--candidate", required=True)
    ap.add_argument("--baseline", required=True)
    ap.add_argument("--judge", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--limit", type=int, default=None)
    a = ap.parse_args()

    cand, base = load(a.candidate), load(a.baseline)
    ids = [qid for qid in cand if qid in base]
    if a.limit:
        ids = ids[: a.limit]
    print(f"[judge] {len(ids)} paired prompts; loading judge {a.judge}")

    tok = AutoTokenizer.from_pretrained(a.judge, trust_remote_code=True)
    judge = AutoModelForCausalLM.from_pretrained(
        a.judge, dtype=torch.bfloat16, device_map="auto", trust_remote_code=True
    ).eval()

    stats = {}  # lang -> {win, tie, loss}
    for i, qid in enumerate(ids):
        c, b = cand[qid], base[qid]
        lang = c.get("lang", "all")
        # Order 1: A=candidate, B=baseline. Order 2: swapped. Average to cancel position bias.
        v1 = verdict(judge, tok, c["prompt"], c["answer"], b["answer"])
        v2 = verdict(judge, tok, c["prompt"], b["answer"], c["answer"])
        # candidate wins this prompt if it's preferred in both orderings; tie if mixed.
        cand_pref = (v1 == "A") + (v2 == "B")
        base_pref = (v1 == "B") + (v2 == "A")
        s = stats.setdefault(lang, {"win": 0, "tie": 0, "loss": 0})
        if cand_pref > base_pref:
            s["win"] += 1
        elif base_pref > cand_pref:
            s["loss"] += 1
        else:
            s["tie"] += 1
        if (i + 1) % 25 == 0:
            print(f"[judge] {i+1}/{len(ids)}")

    # Aggregate
    report = {"per_lang": {}, "overall": {"win": 0, "tie": 0, "loss": 0}}
    for lang, s in stats.items():
        n = sum(s.values())
        wr = 100 * (s["win"] + 0.5 * s["tie"]) / n if n else 0
        report["per_lang"][lang] = {**s, "win_rate": round(wr, 1)}
        for k in ("win", "tie", "loss"):
            report["overall"][k] += s[k]
    n = sum(report["overall"].values())
    report["overall"]["win_rate"] = round(
        100 * (report["overall"]["win"] + 0.5 * report["overall"]["tie"]) / n, 1
    ) if n else 0

    pathlib.Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    json.dump(report, open(a.out, "w"), indent=2)
    print(json.dumps(report, indent=2))
    print(f"\n[judge] candidate win-rate vs baseline: {report['overall']['win_rate']}%  -> {a.out}")


if __name__ == "__main__":
    main()
