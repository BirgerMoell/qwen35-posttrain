#!/usr/bin/env python3
"""Medical-domain eval — verifiable Swedish medical MCQ accuracy.

Separate track from the general EU holdouts: scores a model on the held-out medical MCQ test
(birgermoell/oellm-eu-medical-posttrain-v1, grpo/test split — Swedish medical licensing/specialist
exam questions). Each row carries a self-contained `prompt` ("answer with one letter A-E") and a
gold `answer_letter`, so scoring is deterministic letter-exact (no judge needed).

Usage:
  python scripts/eval_medical_mcq.py --model <ckpt> --data <medical_grpo_test.jsonl> \
    --out eval/results/medical/<tag>.json [--batch-size 32]
"""
from __future__ import annotations

import argparse
import json
import pathlib
import re

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

LETTER = re.compile(r"\b([A-E])\b")


def extract_letter(text: str):
    # the answer is a single letter; take the first standalone A-E (strip any think block first)
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    m = LETTER.search(text.strip().upper())
    return m.group(1) if m else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--data", required=True, help="medical grpo/test jsonl (prompt + answer_letter)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--max-new-tokens", type=int, default=8)
    ap.add_argument("--max-input-len", type=int, default=2048)
    a = ap.parse_args()

    rows = [json.loads(l) for l in open(a.data) if l.strip()]
    rows = [r for r in rows if r.get("prompt") and r.get("answer_letter")]
    print(f"[medical-mcq] {len(rows)} rows; loading {a.model}", flush=True)

    tok = AutoTokenizer.from_pretrained(a.model, trust_remote_code=True)
    tok.padding_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        a.model, dtype=torch.bfloat16, device_map="auto", trust_remote_code=True).eval()

    # Force NO-THINK: the prompt asks for a single letter with no explanation, and Qwen3.5
    # reasoning models otherwise emit a long <think> block that never reaches the answer within
    # a small budget (scoring ~random). enable_thinking=False makes every model answer directly.
    def render(r):
        try:
            return tok.apply_chat_template([{"role": "user", "content": r["prompt"]}],
                                           add_generation_prompt=True, tokenize=False,
                                           enable_thinking=False)
        except TypeError:
            return tok.apply_chat_template([{"role": "user", "content": r["prompt"]}],
                                           add_generation_prompt=True, tokenize=False)
    rendered = [render(r) for r in rows]
    order = sorted(range(len(rows)), key=lambda i: len(rendered[i]))

    correct = total = 0
    by_domain = {}
    bs = a.batch_size
    for s in range(0, len(order), bs):
        idx = order[s:s + bs]
        inputs = tok([rendered[i] for i in idx], return_tensors="pt", padding=True,
                     truncation=True, max_length=a.max_input_len).to(model.device)
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=a.max_new_tokens, do_sample=False,
                                 pad_token_id=tok.pad_token_id)
        gen = out[:, inputs["input_ids"].shape[1]:]
        for k, i in enumerate(idx):
            ans = tok.decode(gen[k], skip_special_tokens=True)
            ok = extract_letter(ans) == str(rows[i]["answer_letter"]).strip().upper()
            correct += ok; total += 1
            d = rows[i].get("medical_domain", "?")
            by_domain.setdefault(d, []).append(ok)
        print(f"[medical-mcq] {min(s + bs, len(order))}/{len(order)}", flush=True)

    report = {
        "model": a.model, "n": total,
        "accuracy": round(100 * correct / total, 1) if total else 0,
        "by_domain": {k: round(100 * sum(v) / len(v), 1) for k, v in sorted(by_domain.items())},
    }
    pathlib.Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    json.dump(report, open(a.out, "w"), indent=2, ensure_ascii=False)
    print(json.dumps(report, indent=2))
    print(f"[medical-mcq] accuracy {report['accuracy']}%  -> {a.out}")


if __name__ == "__main__":
    main()
