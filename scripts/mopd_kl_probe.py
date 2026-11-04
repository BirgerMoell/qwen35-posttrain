#!/usr/bin/env python3
"""Same-origin check before multi-teacher on-policy distillation (RL program §10.7, MOPD).

The student samples one completion per prompt (temperature 1, the rollout distribution); every candidate teacher
then scores exactly those tokens (vLLM prompt_logprobs). The mean of log p_student(y_t) - log p_teacher(y_t) over
completion tokens is a Monte Carlo estimate of the per-token reverse KL(student || teacher), the quantity MOPD
minimises. MOPD reports ~0.04 for same-origin teachers and ~0.19 for an external teacher that destabilised training.

  gen   --student DIR --prompts rows.jsonl --out DIR      (rows: messages, mode default|thinking, domain)
  score --teacher NAME=DIR --out DIR                       (one call per teacher; appends to kl.json)
  report --out DIR
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path


def gen(a) -> None:
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams
    rows = [json.loads(l) for l in Path(a.prompts).read_text(encoding="utf-8").split("\n") if l.strip()]
    tok = AutoTokenizer.from_pretrained(a.student)
    llm = LLM(a.student, dtype="bfloat16", max_model_len=a.max_model_len, gpu_memory_utilization=0.88, seed=0,
              enforce_eager=True)
    prompts = [tok.apply_chat_template(r["messages"], tokenize=False, add_generation_prompt=True,
                                       enable_thinking=r.get("mode") == "thinking") for r in rows]
    params = SamplingParams(temperature=1.0, max_tokens=a.max_tokens, logprobs=0, seed=0)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "student.jsonl", "w", encoding="utf-8") as f:
        for r, p, res in zip(rows, prompts, llm.generate(prompts, params)):
            o = res.outputs[0]
            lps = [next(iter(d.values())).logprob for d in o.logprobs]
            f.write(json.dumps({"domain": r["domain"], "mode": r.get("mode", "default"),
                                "prompt_ids": list(res.prompt_token_ids), "ids": list(o.token_ids),
                                "student_lp": lps}) + "\n")
    print(f"[gen] {len(rows)} completions", flush=True)


def score(a) -> None:
    from vllm import LLM, SamplingParams
    from vllm.inputs import TokensPrompt
    name, path = a.teacher.split("=", 1)
    rows = [json.loads(l) for l in (Path(a.out) / "student.jsonl").read_text().split("\n") if l.strip()]
    llm = LLM(path, dtype="bfloat16", max_model_len=a.max_model_len, gpu_memory_utilization=0.88, seed=0,
              enforce_eager=True)
    params = SamplingParams(max_tokens=1, prompt_logprobs=0, temperature=1.0)
    inputs = [TokensPrompt(prompt_token_ids=r["prompt_ids"] + r["ids"]) for r in rows]
    per_domain = defaultdict(lambda: [0.0, 0, 0.0])   # sum log-ratio, tokens, sum |log-ratio| > 2 count
    for r, res in zip(rows, llm.generate(inputs, params)):
        plp = res.prompt_logprobs
        start = len(r["prompt_ids"])
        for i, tok_id in enumerate(r["ids"]):
            entry = plp[start + i]
            if entry is None or tok_id not in entry:
                continue
            diff = r["student_lp"][i] - entry[tok_id].logprob
            acc = per_domain[(r["domain"], r["mode"])]
            acc[0] += diff
            acc[1] += 1
            acc[2] += abs(diff) > 2
    path_out = Path(a.out) / "kl.json"
    data = json.loads(path_out.read_text()) if path_out.exists() else {}
    data[name] = {f"{d}/{m}": {"kl": round(s / n, 4), "tokens": n, "far_tokens": round(far / n, 4)}
                  for (d, m), (s, n, far) in sorted(per_domain.items()) if n}
    path_out.write_text(json.dumps(data, indent=1) + "\n")
    print(f"[score] {name}", json.dumps(data[name]), flush=True)


def report(a) -> None:
    data = json.loads((Path(a.out) / "kl.json").read_text())
    keys = sorted({k for v in data.values() for k in v})
    print("domain/mode".ljust(26) + "".join(n[:16].rjust(17) for n in data))
    for k in keys:
        print(k.ljust(26) + "".join((f"{data[n][k]['kl']:.3f}" if k in data[n] else "-").rjust(17) for n in data))


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    g = sub.add_parser("gen")
    g.add_argument("--student", required=True)
    g.add_argument("--prompts", required=True)
    g.add_argument("--out", required=True)
    g.add_argument("--max-tokens", type=int, default=1024)
    g.add_argument("--max-model-len", type=int, default=8192)
    s = sub.add_parser("score")
    s.add_argument("--teacher", required=True)
    s.add_argument("--out", required=True)
    s.add_argument("--max-model-len", type=int, default=8192)
    r = sub.add_parser("report")
    r.add_argument("--out", required=True)
    a = ap.parse_args()
    {"gen": gen, "score": score, "report": report}[a.cmd](a)


if __name__ == "__main__":
    main()
