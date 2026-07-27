#!/usr/bin/env python3
"""Self-distillation: the strong 9B-v2 teaches the 2B on its weak buckets.

The 2B is capacity-bound (adding bulk data overfits it — see the scale finding), so the lever is
DATA QUALITY, not volume: take a targeted prompt set covering the 2B's weak areas (summarization
with key points, abstention/no-answer, tight reasoning, instruction-following) and distill the
9B-v2's answers into a small, clean SFT slice.

On-prem teacher (no API key): load the 9B-v2 with transformers (vLLM doesn't support qwen3_5 yet).
Generate with the same anti-loop decoding we shipped to the 2B, so the teacher data itself is
loop-free. HARD FILTER before keeping a row:
  - language purity: the answer must be in the target language (langdetect) for non-English prompts
  - no degeneration: reject answers with a sentence repeated >=3x
  - length sanity: within [min_words, max_words]
  - abstention prompts: the answer MUST contain a refusal marker (teach the 2B to say "I don't know")

Emits {"messages": [{user},{assistant}], "lang"} jsonl for scripts/sft_train.py.

Usage (LUMI compute node, offline):
  python scripts/distill_from_9b.py --prompts data/distill_seed_prompts.jsonl \
    --teacher /scratch/.../output/qwen35-9b-eu-grpo-v2full-bf16 --out data/distill/9b_to_2b.jsonl
"""
from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path

REFUSAL_MARKERS = ["cannot", "can't", "won't", "not able", "don't have", "no information",
                   "inte kan", "vet inte", "kan inte", "en voi", "no puedo", "ne peux",
                   "kann nicht", "not appropriate", "insufficient", "inte tillräckligt"]


def _repeated_sentence(text: str, k=3) -> bool:
    sents = [s.strip().lower() for s in re.split(r"[.!?\n]+", text) if len(s.strip()) > 12]
    from collections import Counter
    return any(c >= k for c in Counter(sents).values())


def _purity_ok(text: str, lang: str) -> bool:
    if lang == "en":
        return True
    try:
        from langdetect import detect
        return detect(text) == lang
    except Exception:
        return True  # langdetect can't do is/ga/mt — don't reject on those


def _api_generate(prompt, think, api_base, api_model, key, max_new_tokens):
    """OpenAI-compatible chat completion (e.g. GLM-5.2 via z.ai / bigmodel.cn). No GPU — the teacher
    runs remotely, so this path runs on any host with internet (LUMI LOGIN node or the L4 box; NOT
    compute nodes, which have no egress)."""
    import urllib.request
    body = json.dumps({"model": api_model,
                       "messages": [{"role": "user", "content": prompt}],
                       "temperature": 0.3, "max_tokens": max_new_tokens,
                       "thinking": {"type": "enabled" if think else "disabled"}}).encode()
    req = urllib.request.Request(api_base.rstrip("/") + "/chat/completions", data=body,
                                 headers={"Authorization": f"Bearer {key}",
                                          "Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=180) as r:
        data = json.loads(r.read())
    return data["choices"][0]["message"]["content"].strip()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompts", default="data/distill_seed_prompts.jsonl")
    ap.add_argument("--teacher", help="local HF model dir (transformers backend)")
    ap.add_argument("--out", default="data/distill/9b_to_2b.jsonl")
    ap.add_argument("--max-new-tokens", type=int, default=400)
    ap.add_argument("--min-words", type=int, default=8)
    ap.add_argument("--max-words", type=int, default=320)
    # API teacher (e.g. GLM-5.2) — set --api-base to use it instead of a local --teacher
    ap.add_argument("--api-base", default=None, help="e.g. https://api.z.ai/api/paas/v4")
    ap.add_argument("--api-model", default="glm-5.2")
    ap.add_argument("--api-key-env", default="GLM_API_KEY")
    a = ap.parse_args()

    rows = [json.loads(l) for l in open(a.prompts) if l.strip()]
    use_api = bool(a.api_base)
    if use_api:
        key = os.environ.get(a.api_key_env, "").strip()
        if not key:
            raise SystemExit(f"set the teacher API key in ${a.api_key_env}")
        print(f"[distill] API teacher: {a.api_model} @ {a.api_base}", flush=True)
        tok = model = None
    else:
        if not a.teacher:
            raise SystemExit("pass --teacher (local) or --api-base (API teacher)")
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        tok = AutoTokenizer.from_pretrained(a.teacher)
        model = AutoModelForCausalLM.from_pretrained(a.teacher, dtype=torch.bfloat16,
                                                     device_map="auto").eval()

    kept, dropped = [], {"purity": 0, "loop": 0, "length": 0, "no_refusal": 0}
    for r in rows:
        prompt, lang, cat = r["prompt"], r.get("lang", "en"), r.get("category", "")
        think = cat == "reasoning"
        if use_api:
            try:
                ans = _api_generate(prompt, think, a.api_base, a.api_model, key, a.max_new_tokens)
            except Exception as e:
                print(f"[distill] api error ({type(e).__name__}) — skipping", flush=True)
                continue
        else:
            import torch
            enc = tok.apply_chat_template([{"role": "user", "content": prompt}],
                                          add_generation_prompt=True, return_tensors="pt",
                                          return_dict=True, enable_thinking=think).to(model.device)
            with torch.no_grad():
                out = model.generate(**enc, max_new_tokens=a.max_new_tokens, do_sample=False,
                                      repetition_penalty=1.15, no_repeat_ngram_size=4,
                                      pad_token_id=tok.pad_token_id or tok.eos_token_id)
            ans = tok.decode(out[0, enc["input_ids"].shape[1]:], skip_special_tokens=True).strip()

        wc = len(ans.split())
        if not (a.min_words <= wc <= a.max_words):
            dropped["length"] += 1; continue
        if _repeated_sentence(ans):
            dropped["loop"] += 1; continue
        if not _purity_ok(ans, lang):
            dropped["purity"] += 1; continue
        if cat == "abstention" and not any(m in ans.lower() for m in REFUSAL_MARKERS):
            dropped["no_refusal"] += 1; continue  # teacher didn't abstain -> not a clean example
        kept.append({"messages": [{"role": "user", "content": prompt},
                                  {"role": "assistant", "content": ans}], "lang": lang})
        print(f"[distill] KEEP {cat}/{lang} ({wc}w): {ans[:70]!r}", flush=True)

    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    with open(a.out, "w") as f:
        for k in kept:
            f.write(json.dumps(k, ensure_ascii=False) + "\n")
    print(f"\n[distill] kept {len(kept)}/{len(rows)}  dropped={dropped}")
    print(f"[distill] -> {a.out}")


if __name__ == "__main__":
    main()
