#!/usr/bin/env python3
"""Distillation data generation — serve a teacher across many GPUs (vLLM) and harvest traces.

Reads a prompt bank, generates N samples per prompt from the teacher at scale, optionally
verifies (math: boxed-answer match) and keeps only correct traces, writes a distillation SFT
file `{messages:[user, assistant]}` ready for scripts/sft_train.py.

Multi-node: vLLM uses the Ray cluster started by the sbatch (distributed_executor_backend=ray)
with tensor_parallel_size = total GPUs. Throughput scales with nodes — the whole point.

Usage (driver runs on the Ray head; see slurm/distill_teacher_lumi.sbatch):
  python scripts/distill_generate.py \
    --teacher /scratch/.../models/Qwen3.5-32B \
    --prompts data/prompt_bank.jsonl \
    --out /scratch/.../posttrain-data/distill/traces.jsonl \
    --tp 16 --n 4 --temperature 0.8 --max-tokens 4096 --verify math
"""
from __future__ import annotations

import argparse
import json
import os
import pathlib
import re


def load_prompts(path, limit):
    rows = []
    for line in open(path):
        line = line.strip()
        if not line:
            continue
        d = json.loads(line)
        # accept {prompt} or {problem}/{question}; carry an optional gold answer for verification
        p = d.get("prompt") or d.get("problem") or d.get("question")
        if not p:
            continue
        rows.append({"prompt": p, "answer": d.get("answer") or d.get("expected_answer"),
                     "lang": d.get("lang", "en")})
        if limit and len(rows) >= limit:
            break
    return rows


def boxed(s: str):
    m = re.findall(r"\\boxed\{([^}]*)\}", s)
    return m[-1].strip() if m else None


def verify_math(trace: str, gold) -> bool:
    if gold is None:
        return True  # no gold -> can't verify, keep (caller decides)
    got = boxed(trace)
    if got is None:
        nums = re.findall(r"-?\d+(?:\.\d+)?", trace.replace(",", ""))
        got = nums[-1] if nums else None
    return got is not None and str(got).strip() == str(gold).strip()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--teacher", required=True)
    ap.add_argument("--prompts", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--tp", type=int, default=8, help="tensor_parallel_size = total GPUs")
    ap.add_argument("--n", type=int, default=4, help="samples per prompt")
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--max-tokens", type=int, default=4096)
    ap.add_argument("--max-model-len", type=int, default=8192)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--verify", choices=["none", "math"], default="none")
    ap.add_argument("--model-impl", default="auto", help="auto | transformers (fallback for unsupported arch)")
    a = ap.parse_args()

    from vllm import LLM, SamplingParams
    from transformers import AutoTokenizer

    rows = load_prompts(a.prompts, a.limit)
    print(f"[distill] {len(rows)} prompts x {a.n} samples; teacher={a.teacher} tp={a.tp}", flush=True)

    tok = AutoTokenizer.from_pretrained(a.teacher, trust_remote_code=True)
    prompts = [tok.apply_chat_template([{"role": "user", "content": r["prompt"]}],
                                       add_generation_prompt=True, tokenize=False) for r in rows]

    llm = LLM(model=a.teacher, tensor_parallel_size=a.tp, dtype="bfloat16",
              gpu_memory_utilization=0.9, max_model_len=a.max_model_len,
              trust_remote_code=True, model_impl=a.model_impl,
              distributed_executor_backend="ray" if a.tp > 8 else "mp")
    sp = SamplingParams(n=a.n, temperature=a.temperature, top_p=0.95, max_tokens=a.max_tokens)

    outs = llm.generate(prompts, sp)

    pathlib.Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    kept = total = 0
    with open(a.out, "w") as f:
        for r, o in zip(rows, outs):
            for cand in o.outputs:
                total += 1
                trace = cand.text
                if a.verify == "math" and not verify_math(trace, r["answer"]):
                    continue
                kept += 1
                f.write(json.dumps({
                    "messages": [{"role": "user", "content": r["prompt"]},
                                 {"role": "assistant", "content": trace}],
                    "lang": r["lang"], "verified": a.verify != "none",
                }, ensure_ascii=False) + "\n")
    print(f"[distill] kept {kept}/{total} traces -> {a.out}", flush=True)


if __name__ == "__main__":
    main()
