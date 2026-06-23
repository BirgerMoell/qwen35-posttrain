#!/usr/bin/env python3
"""Generate multilingual reasoning traces from a teacher — reasoning IN the target language.

The #1 data gap (docs/LANGUAGE_GAPS.md) is reasoning data: we only have it in English+Finnish.
Strong teachers (GLM-5.2, etc.) reason well but **mix English into the chain-of-thought** even
when answering in another language. This script:

  1. Prompts the teacher to think AND answer ENTIRELY in the target language (system + few-shot
     style instruction reinforcing both the reasoning trace and the final answer).
  2. Generates N samples per problem.
  3. FILTERS hard:
       - language consistency: language-ID the reasoning trace; reject if it isn't the target
         language (this is what kills the English-leak problem).
       - answer correctness (when a gold answer is available): verifiable math/exact match.
  4. Emits clean SFT data `{messages:[user, assistant], lang, ...}` for scripts/sft_train.py.

Teacher-agnostic (vLLM): point --teacher at GLM-5.2 (served multi-node, see
slurm/distill_teacher_lumi.sbatch) or any model. Output feeds the reasoning-SFT stage.
"""
from __future__ import annotations

import argparse
import json
import os
import pathlib
import re

# code -> (English name, native name) — reinforce the instruction in both.
LANGS = {
    "sv": ("Swedish", "svenska"), "fi": ("Finnish", "suomi"), "da": ("Danish", "dansk"),
    "no": ("Norwegian", "norsk"), "is": ("Icelandic", "íslenska"), "de": ("German", "Deutsch"),
    "fr": ("French", "français"), "es": ("Spanish", "español"), "it": ("Italian", "italiano"),
    "pt": ("Portuguese", "português"), "nl": ("Dutch", "Nederlands"), "pl": ("Polish", "polski"),
    "cs": ("Czech", "čeština"), "sk": ("Slovak", "slovenčina"), "sl": ("Slovenian", "slovenščina"),
    "hr": ("Croatian", "hrvatski"), "sr": ("Serbian", "srpski"), "bs": ("Bosnian", "bosanski"),
    "bg": ("Bulgarian", "български"), "ro": ("Romanian", "română"), "hu": ("Hungarian", "magyar"),
    "el": ("Greek", "ελληνικά"), "et": ("Estonian", "eesti"), "lv": ("Latvian", "latviešu"),
    "lt": ("Lithuanian", "lietuvių"), "mt": ("Maltese", "Malti"), "ga": ("Irish", "Gaeilge"),
    "cy": ("Welsh", "Cymraeg"), "eu": ("Basque", "euskara"), "gl": ("Galician", "galego"),
    "ca": ("Catalan", "català"), "mk": ("Macedonian", "македонски"), "sq": ("Albanian", "shqip"),
    "lb": ("Luxembourgish", "Lëtzebuergesch"), "uk": ("Ukrainian", "українська"),
    "ru": ("Russian", "русский"), "tr": ("Turkish", "Türkçe"),
}

SYSTEM_TMPL = (
    "You are an expert tutor who reasons entirely in {name} ({native}). "
    "Solve the problem by thinking step by step. CRITICAL: your ENTIRE reasoning — every "
    "thought, not just the final answer — must be written ONLY in {name}. Do NOT use English "
    "or any other language at any point in your reasoning. Put the final answer in \\boxed{{}}."
)


def extract_think(text: str) -> str:
    m = re.search(r"<think>(.*?)</think>", text, re.DOTALL)
    return m.group(1) if m else text


def boxed(s: str):
    m = re.findall(r"\\boxed\{([^}]*)\}", s)
    return m[-1].strip() if m else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--teacher", required=True)
    ap.add_argument("--problems", required=True, help="jsonl with {problem|question|prompt}[, answer]")
    ap.add_argument("--lang", required=True, help="target language code, e.g. sv")
    ap.add_argument("--out", required=True)
    ap.add_argument("--tp", type=int, default=8)
    ap.add_argument("--n", type=int, default=4)
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--max-tokens", type=int, default=8192)
    ap.add_argument("--max-model-len", type=int, default=16384)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--verify", choices=["none", "math"], default="math")
    ap.add_argument("--lang-min-conf", type=float, default=0.85,
                    help="reject a trace if langdetect's target-language probability is below this")
    ap.add_argument("--model-impl", default="auto")
    a = ap.parse_args()

    name, native = LANGS.get(a.lang, (a.lang, a.lang))
    rows = []
    for line in open(a.problems):
        line = line.strip()
        if not line:
            continue
        d = json.loads(line)
        p = d.get("problem") or d.get("question") or d.get("prompt")
        if p:
            rows.append({"problem": p, "answer": d.get("answer") or d.get("expected_answer")})
        if a.limit and len(rows) >= a.limit:
            break
    print(f"[reason] {len(rows)} problems x {a.n}; lang={a.lang} ({name}); teacher={a.teacher}", flush=True)

    from vllm import LLM, SamplingParams
    from transformers import AutoTokenizer
    from langdetect import detect_langs, DetectorFactory
    DetectorFactory.seed = 0

    tok = AutoTokenizer.from_pretrained(a.teacher, trust_remote_code=True)
    sysmsg = SYSTEM_TMPL.format(name=name, native=native)
    prompts = [tok.apply_chat_template(
        [{"role": "system", "content": sysmsg}, {"role": "user", "content": r["problem"]}],
        add_generation_prompt=True, tokenize=False) for r in rows]

    llm = LLM(model=a.teacher, tensor_parallel_size=a.tp, dtype="bfloat16",
              gpu_memory_utilization=0.9, max_model_len=a.max_model_len, trust_remote_code=True,
              model_impl=a.model_impl, distributed_executor_backend="ray" if a.tp > 8 else "mp")
    sp = SamplingParams(n=a.n, temperature=a.temperature, top_p=0.95, max_tokens=a.max_tokens)
    outs = llm.generate(prompts, sp)

    def lang_ok(text):
        think = extract_think(text)[:3000]
        try:
            for cand in detect_langs(think):
                if cand.lang == a.lang and cand.prob >= a.lang_min_conf:
                    return True
        except Exception:
            return False
        return False

    pathlib.Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    kept = total = lang_rej = ans_rej = 0
    with open(a.out, "w") as f:
        for r, o in zip(rows, outs):
            for cand in o.outputs:
                total += 1
                trace = cand.text
                if not lang_ok(trace):            # the English-leak filter
                    lang_rej += 1
                    continue
                if a.verify == "math" and r["answer"] is not None:
                    got = boxed(trace)
                    if got is None or str(got).strip() != str(r["answer"]).strip():
                        ans_rej += 1
                        continue
                kept += 1
                f.write(json.dumps({
                    "messages": [{"role": "user", "content": r["problem"]},
                                 {"role": "assistant", "content": trace}],
                    "lang": a.lang, "verified": a.verify != "none",
                }, ensure_ascii=False) + "\n")
    print(f"[reason] kept {kept}/{total}  (lang-rejected {lang_rej}, answer-rejected {ans_rej}) "
          f"-> {a.out}", flush=True)


if __name__ == "__main__":
    main()
