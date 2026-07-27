#!/usr/bin/env python3
"""Generate multilingual reasoning traces from a teacher — reasoning IN the target language.

The #1 data gap (docs/LANGUAGE_GAPS.md) is reasoning data: we only have it in English+Finnish.
Strong teachers (GLM-5.2, etc.) reason well but **mix English into the chain-of-thought** even
when answering in another language. This script:

  1. Prompts the teacher to think AND answer ENTIRELY in the target language (language-forcing
     system prompt reinforcing both the reasoning trace and the final answer).
  2. Generates N samples per problem.
  3. FILTERS hard:
       - language consistency: language-ID the reasoning trace; reject if it isn't the target
         language (this is what kills the English-leak problem).
       - answer correctness (when a gold answer is available): verifiable math/exact match.
  4. Emits clean SFT data `{messages:[user, assistant], lang, ...}` for scripts/sft_train.py.

Two teacher backends:
  --backend api   : an OpenAI-compatible API (GLM-5.2 via z.ai / bigmodel.cn). No GPUs needed.
                    `--teacher glm-5.2 --api-base https://api.z.ai/api/paas/v4` + GLM_API_KEY env.
  --backend vllm  : self-host the teacher with vLLM (see slurm/distill_teacher_lumi.sbatch).
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


def load_rows(a):
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
    return rows


def gen_api(rows, a, sysmsg):
    """Concurrent OpenAI-compatible chat-completions (GLM-5.2). Returns per-row list of texts,
    each text reconstructed as `<think>{reasoning}</think>{answer}` so the downstream filters
    (language-ID on the think block, boxed-answer check) work identically to the vLLM path."""
    import concurrent.futures
    import time
    import urllib.request

    key = os.environ.get(a.api_key_env, "").strip()
    if not key:
        raise SystemExit(f"[reason] set the API key in env ${a.api_key_env}")
    url = a.api_base.rstrip("/") + "/chat/completions"

    def one(problem):
        body = {
            "model": a.teacher,
            "messages": [{"role": "system", "content": sysmsg},
                         {"role": "user", "content": problem}],
            "temperature": a.temperature, "top_p": 0.95, "max_tokens": a.max_tokens,
        }
        if a.thinking:
            body["thinking"] = {"type": "enabled"}   # GLM thinking mode (ignored if unsupported)
        data = json.dumps(body).encode()
        req = urllib.request.Request(url, data=data, headers={
            "Content-Type": "application/json", "Authorization": f"Bearer {key}"})
        for attempt in range(5):
            try:
                with urllib.request.urlopen(req, timeout=a.timeout) as r:
                    msg = json.loads(r.read())["choices"][0]["message"]
                reasoning = (msg.get("reasoning_content") or "").strip()
                content = (msg.get("content") or "").strip()
                if reasoning and "<think>" not in content:
                    return f"<think>\n{reasoning}\n</think>\n\n{content}"
                return content
            except Exception as e:                       # noqa: BLE001 — retry w/ backoff
                if attempt == 4:
                    print(f"[reason] api error (giving up): {str(e)[:120]}", flush=True)
                time.sleep(2 * (attempt + 1))
        return ""

    results = [[] for _ in rows]
    total = len(rows) * a.n
    with concurrent.futures.ThreadPoolExecutor(max_workers=a.concurrency) as ex:
        fut = {}
        for i, r in enumerate(rows):
            for _ in range(a.n):
                fut[ex.submit(one, r["problem"])] = i
        done = 0
        for f in concurrent.futures.as_completed(fut):
            results[fut[f]].append(f.result())
            done += 1
            if done % 25 == 0 or done == total:
                print(f"[reason] api {done}/{total}", flush=True)
    return results


def gen_vllm(rows, a, sysmsg):
    from vllm import LLM, SamplingParams
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(a.teacher, trust_remote_code=True)
    prompts = [tok.apply_chat_template(
        [{"role": "system", "content": sysmsg}, {"role": "user", "content": r["problem"]}],
        add_generation_prompt=True, tokenize=False) for r in rows]
    llm = LLM(model=a.teacher, tensor_parallel_size=a.tp, dtype="bfloat16",
              gpu_memory_utilization=0.9, max_model_len=a.max_model_len, trust_remote_code=True,
              model_impl=a.model_impl, distributed_executor_backend="ray" if a.tp > 8 else "mp")
    sp = SamplingParams(n=a.n, temperature=a.temperature, top_p=0.95, max_tokens=a.max_tokens)
    outs = llm.generate(prompts, sp)
    return [[c.text for c in o.outputs] for o in outs]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", choices=["api", "vllm"], default="api")
    ap.add_argument("--teacher", required=True, help="API model id (e.g. glm-5.2) or local model path")
    ap.add_argument("--problems", required=True, help="jsonl with {problem|question|prompt}[, answer]")
    ap.add_argument("--lang", required=True, help="target language code, e.g. sv")
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=4)
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--max-tokens", type=int, default=8192)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--verify", choices=["none", "math"], default="math")
    ap.add_argument("--lang-min-conf", type=float, default=0.85,
                    help="reject a trace if langdetect's target-language probability is below this")
    # API backend
    ap.add_argument("--api-base", default="https://api.z.ai/api/paas/v4",
                    help="OpenAI-compatible base (z.ai default; bigmodel: https://open.bigmodel.cn/api/paas/v4)")
    ap.add_argument("--api-key-env", default="GLM_API_KEY")
    ap.add_argument("--concurrency", type=int, default=8)
    ap.add_argument("--timeout", type=int, default=600)
    ap.add_argument("--thinking", action="store_true", default=True)
    ap.add_argument("--no-thinking", dest="thinking", action="store_false")
    # vLLM backend
    ap.add_argument("--tp", type=int, default=8)
    ap.add_argument("--max-model-len", type=int, default=16384)
    ap.add_argument("--model-impl", default="auto")
    a = ap.parse_args()

    name, native = LANGS.get(a.lang, (a.lang, a.lang))
    sysmsg = SYSTEM_TMPL.format(name=name, native=native)
    rows = load_rows(a)
    print(f"[reason] {len(rows)} problems x {a.n}; lang={a.lang} ({name}); "
          f"backend={a.backend}; teacher={a.teacher}", flush=True)

    from langdetect import detect_langs, DetectorFactory
    DetectorFactory.seed = 0

    per_row_texts = gen_api(rows, a, sysmsg) if a.backend == "api" else gen_vllm(rows, a, sysmsg)

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
    kept = total = lang_rej = ans_rej = empty = 0
    with open(a.out, "w") as f:
        for r, texts in zip(rows, per_row_texts):
            for trace in texts:
                total += 1
                if not trace:
                    empty += 1
                    continue
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
    print(f"[reason] kept {kept}/{total}  (lang-rejected {lang_rej}, answer-rejected {ans_rej}, "
          f"empty {empty}) -> {a.out}", flush=True)


if __name__ == "__main__":
    main()
