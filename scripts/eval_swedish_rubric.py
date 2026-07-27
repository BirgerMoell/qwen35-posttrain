#!/usr/bin/env python3
"""Swedish capability rubric eval — open-ended generation scored by a Swedish-language rubric.

A practical, vLLM-free alternative to EuroEval for Swedish (Qwen3.5 isn't yet supported by the
fast inference runtimes EuroEval needs). Generates the model's answer to a categorized set of
Swedish prompts (data/swedish_rubric_prompts.jsonl — spanning sentiment, Swedish knowledge,
reasoning, summarization, linguistic correctness, instruction-following, common sense, creative),
then scores each on a Swedish rubric:

  - språkkvalitet : correct, fluent, idiomatic Swedish; right å/ä/ö; no foreign-language leakage
  - korrekthet    : factually / logically correct, no hallucination
  - instruktion   : follows the prompt (format, constraints, length)
  - hjälpsamhet   : genuinely useful, appropriate depth

Two scores per answer: AUTOMATIC Swedish-purity via langdetect (always), and the 4-dim LLM-judge
rubric (optional — point --judge-api-base / --judge-model at GLM-5.2 / any OpenAI-compatible API).
Reports per-category and overall, and saves every generation for eyeballing.

Usage:
  python scripts/eval_swedish_rubric.py --model <ckpt> --out eval/results/swedish-rubric/<tag>.json \
    [--judge-api-base https://api.z.ai/api/paas/v4 --judge-model glm-5.2 --judge-key-env GLM_API_KEY]
"""
from __future__ import annotations

import argparse
import json
import os
import pathlib
import re
import statistics

DIMS = ["sprakkvalitet", "korrekthet", "instruktion", "hjalpsamhet"]

RUBRIK = (
    "Du är en noggrann bedömare av svenska AI-svar. Frågan (kategori: {category}) löd:\n"
    "{question}\n\nSVAR ATT BEDÖMA:\n{answer}\n\n"
    "Betygsätt svaret 1-5 (heltal) på varje dimension och returnera ENBART JSON på formen "
    "{{\"sprakkvalitet\": n, \"korrekthet\": n, \"instruktion\": n, \"hjalpsamhet\": n, "
    "\"motivering\": \"<=20 ord\"}}.\n"
    "- sprakkvalitet: korrekt, flytande och idiomatisk svenska; rätt å/ä/ö; ingen engelska eller "
    "annat språk som läcker in; inga upprepningar.\n"
    "- korrekthet: faktamässigt och logiskt korrekt, inga påhitt.\n"
    "- instruktion: följer det frågan ber om (format, begränsningar, längd).\n"
    "- hjalpsamhet: genuint användbart svar med lagom djup."
)


def strip_think(t):
    return re.sub(r"<think>.*?</think>", "", t, flags=re.DOTALL).strip()


def gen_local(model_path, rows, max_new_tokens, think):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_path, dtype=torch.bfloat16, device_map="auto", trust_remote_code=True).eval()
    outs = []
    for r in rows:
        inputs = tok.apply_chat_template([{"role": "user", "content": r["prompt"]}],
                                         add_generation_prompt=True, return_tensors="pt",
                                         return_dict=True, enable_thinking=think).to(model.device)
        plen = inputs["input_ids"].shape[1]
        with torch.no_grad():
            o = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=True,
                               temperature=0.7, top_p=0.95, pad_token_id=tok.eos_token_id)
        outs.append(strip_think(tok.decode(o[0, plen:], skip_special_tokens=True).strip()))
        print(f"[sv-rubric] generated {len(outs)}/{len(rows)} ({r['category']})", flush=True)
    return outs


def sv_purity(answer):
    from langdetect import detect_langs, DetectorFactory
    DetectorFactory.seed = 0
    try:
        for c in detect_langs(answer[:2000]):
            if c.lang == "sv":
                return round(c.prob, 3)
    except Exception:
        return 0.0
    return 0.0


def judge(api_base, model, key, category, question, answer, timeout=300):
    import urllib.request
    body = json.dumps({"model": model, "temperature": 0.0, "max_tokens": 250,
                       "messages": [{"role": "user", "content": RUBRIK.format(
                           category=category, question=question, answer=answer[:3500])}]}).encode()
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
    ap.add_argument("--prompts", default="data/swedish_rubric_prompts.jsonl")
    ap.add_argument("--out", required=True)
    ap.add_argument("--tag", default="sv-rubric")
    ap.add_argument("--max-new-tokens", type=int, default=400)
    ap.add_argument("--think", action="store_true")
    ap.add_argument("--judge-api-base", default=None)
    ap.add_argument("--judge-model", default="glm-5.2")
    ap.add_argument("--judge-key-env", default="GLM_API_KEY")
    a = ap.parse_args()

    rows = [json.loads(l) for l in open(a.prompts) if l.strip()]
    print(f"[sv-rubric] {len(rows)} Swedish prompts; model={a.model}", flush=True)
    gens = gen_local(a.model, rows, a.max_new_tokens, a.think)

    key = os.environ.get(a.judge_key_env, "").strip() if a.judge_api_base else ""
    results = []
    for r, ans in zip(rows, gens):
        rec = {"category": r["category"], "prompt": r["prompt"], "answer": ans,
               "sv_purity": sv_purity(ans), "words": len(ans.split())}
        if a.judge_api_base and key:
            try:
                rec["rubric"] = judge(a.judge_api_base, a.judge_model, key, r["category"],
                                      r["prompt"], ans)
            except Exception as e:
                rec["rubric_error"] = str(e)[:120]
        results.append(rec)
        print(f"[sv-rubric] {r['category']}: sv={rec['sv_purity']} | {ans[:70]!r}", flush=True)

    # aggregate: overall + per-category, automatic purity + each rubric dim (if judged)
    def avg(xs):
        xs = [x for x in xs if isinstance(x, (int, float))]
        return round(statistics.mean(xs), 2) if xs else None

    cats = sorted({r["category"] for r in results})
    summary = {"sv_purity": avg([r["sv_purity"] for r in results])}
    for d in DIMS:
        summary[d] = avg([r.get("rubric", {}).get(d) for r in results])
    by_category = {}
    for c in cats:
        cr = [r for r in results if r["category"] == c]
        by_category[c] = {"n": len(cr), "sv_purity": avg([r["sv_purity"] for r in cr]),
                          **{d: avg([r.get("rubric", {}).get(d) for r in cr]) for d in DIMS}}

    report = {"model": a.tag, "n": len(results), "summary": summary,
              "by_category": by_category, "results": results}
    pathlib.Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    json.dump(report, open(a.out, "w"), indent=2, ensure_ascii=False)
    print("[sv-rubric] SUMMARY:", json.dumps(summary, ensure_ascii=False), flush=True)
    print(f"[sv-rubric] -> {a.out}")


if __name__ == "__main__":
    main()
