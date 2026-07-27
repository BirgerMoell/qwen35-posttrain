#!/usr/bin/env python3
"""Open-ended GENERATION eval with a rubric — complements the deterministic holdout benchmark.

The holdout benchmark scores exact/verifiable answers; it can't see fluency, depth, or
language-purity in free-form text. This runs a fixed open-ended probe (default: "What is the
meaning of life?" translated into each target language, data/meaning_of_life_prompts.jsonl),
generates the model's answer per language, and scores it two ways:

  1. AUTOMATIC (always): langdetect on the answer — is it actually in the target language, and with
     what confidence? (catches English-leak / wrong-language replies) + word count / non-empty.
  2. RUBRIC (optional LLM judge, OpenAI-compatible): 1-5 on language_quality, coherence, depth,
     helpfulness. Point --judge-api-base/--judge-model at GLM-5.2 / any OpenAI-compatible endpoint.

Saves every generation + scores to JSON so you can EYEBALL the raw text, not just a number.

Usage:
  python scripts/eval_generation_rubric.py --model <ckpt> --prompts data/meaning_of_life_prompts.jsonl \
    --out eval/results/rubric/<tag>.json [--judge-api-base https://api.z.ai/api/paas/v4 \
    --judge-model glm-5.2 --judge-key-env GLM_API_KEY]
"""
from __future__ import annotations

import argparse
import json
import os
import pathlib
import re

RUBRIC = (
    "You are grading an AI answer to the question \"{question}\" which was asked in {name}. "
    "The answer should be written entirely in {name}. Score 1-5 (integers) on each dimension and "
    "return ONLY JSON: {{\"language_quality\": n, \"coherence\": n, \"depth\": n, "
    "\"helpfulness\": n, \"note\": \"<=15 words\"}}. "
    "language_quality: fully in {name}, fluent, grammatical, no foreign-language leakage. "
    "coherence: logically consistent and on-topic. depth: thoughtful and substantive vs generic "
    "filler. helpfulness: genuinely engages the question.\n\nANSWER:\n{answer}"
)


def gen_local(model_path, rows, max_new_tokens, think):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_path, dtype=torch.bfloat16, device_map="auto", trust_remote_code=True).eval()
    outs = []
    for r in rows:
        msgs = [{"role": "user", "content": r["prompt"]}]
        # transformers 5.x apply_chat_template returns a BatchEncoding (dict) -> use return_dict + **
        inputs = tok.apply_chat_template(msgs, add_generation_prompt=True, return_tensors="pt",
                                         return_dict=True, enable_thinking=think).to(model.device)
        plen = inputs["input_ids"].shape[1]
        with torch.no_grad():
            o = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=True,
                               temperature=0.7, top_p=0.95, pad_token_id=tok.eos_token_id)
        outs.append(tok.decode(o[0, plen:], skip_special_tokens=True).strip())
        print(f"[rubric] generated {r['lang']} ({len(outs)}/{len(rows)})", flush=True)
    return outs


def strip_think(t):
    return re.sub(r"<think>.*?</think>", "", t, flags=re.DOTALL).strip()


def lang_score(answer, lang):
    from langdetect import detect_langs, DetectorFactory
    DetectorFactory.seed = 0
    try:
        for c in detect_langs(answer[:2000]):
            if c.lang == lang:
                return round(c.prob, 3)
    except Exception:
        return 0.0
    return 0.0


def judge(api_base, model, key, name, question, answer, timeout=300):
    import urllib.request
    body = json.dumps({"model": model, "temperature": 0.0, "max_tokens": 200,
                       "messages": [{"role": "user", "content": RUBRIC.format(
                           name=name, question=question, answer=answer[:3000])}]}).encode()
    req = urllib.request.Request(api_base.rstrip("/") + "/chat/completions", data=body,
                                 headers={"Content-Type": "application/json",
                                          "Authorization": f"Bearer {key}"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        txt = json.loads(r.read())["choices"][0]["message"]["content"]
    m = re.search(r"\{.*\}", txt, re.DOTALL)
    return json.loads(m.group(0)) if m else {}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--prompts", default="data/meaning_of_life_prompts.jsonl")
    ap.add_argument("--out", required=True)
    ap.add_argument("--tag", default="rubric")
    ap.add_argument("--max-new-tokens", type=int, default=300)
    ap.add_argument("--think", action="store_true")
    ap.add_argument("--judge-api-base", default=None)
    ap.add_argument("--judge-model", default="glm-5.2")
    ap.add_argument("--judge-key-env", default="GLM_API_KEY")
    a = ap.parse_args()

    rows = [json.loads(l) for l in open(a.prompts) if l.strip()]
    print(f"[rubric] {len(rows)} languages; model={a.model}", flush=True)
    gens = gen_local(a.model, rows, a.max_new_tokens, a.think)

    key = os.environ.get(a.judge_key_env, "").strip() if a.judge_api_base else ""
    results, agg = [], {"language": [], "language_quality": [], "coherence": [], "depth": [], "helpfulness": []}
    for r, g in zip(rows, gens):
        ans = strip_think(g)
        ls = lang_score(ans, r["lang"])
        rec = {"lang": r["lang"], "name": r["name"], "prompt": r["prompt"],
               "answer": ans, "lang_id_prob": ls, "words": len(ans.split())}
        agg["language"].append(ls)
        if a.judge_api_base and key:
            try:
                jr = judge(a.judge_api_base, a.judge_model, key, r["name"], r["prompt"], ans)
                rec["rubric"] = jr
                for k in ("language_quality", "coherence", "depth", "helpfulness"):
                    if isinstance(jr.get(k), (int, float)):
                        agg[k].append(jr[k])
            except Exception as e:
                rec["rubric_error"] = str(e)[:120]
        results.append(rec)
        print(f"[rubric] {r['lang']}: lang_id={ls} | {ans[:80]!r}", flush=True)

    summary = {k: round(sum(v) / len(v), 3) for k, v in agg.items() if v}
    report = {"model": a.tag, "n": len(results), "summary": summary, "results": results}
    pathlib.Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    json.dump(report, open(a.out, "w"), indent=2, ensure_ascii=False)
    print("[rubric] SUMMARY:", json.dumps(summary), flush=True)
    print(f"[rubric] -> {a.out}")


if __name__ == "__main__":
    main()
