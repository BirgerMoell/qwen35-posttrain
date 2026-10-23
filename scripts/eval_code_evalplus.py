#!/usr/bin/env python3
"""Code baseline: HumanEval+ and MBPP+ (EvalPlus) in default and thinking mode.

Two subcommands, run in different environments on purpose:
- `gen` (GPU, vLLM venv): renders each problem with the model's chat template (`enable_thinking` set explicitly, so
  the RL thinking-default template behaves like v2.1), generates greedily, keeps only the text after a closed trace
  (an unclosed trace yields an empty solution), extracts the code with EvalPlus' sanitizer and writes
  `<out>/<dataset>-<mode>.jsonl` with `task_id`, `solution`, `raw`, `closed`, `new_tokens`.
- `score` (CPU, EvalPlus venv): runs `evalplus.evaluate` on each samples file. The launcher runs this step in an
  isolated container (no home, no project directories, only the samples directory writable), because it executes
  model-written code.

Prompt: the EvalPlus instruction prompt for chat models ("Please provide a self-contained Python script ...").
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

INSTRUCTION = ("Please provide a self-contained Python script that solves the following problem in a markdown "
               "code block:\n```\n{prompt}\n```")


def load_problems(dataset: str) -> dict:
    from evalplus.data import get_human_eval_plus, get_mbpp_plus
    return get_human_eval_plus() if dataset == "humaneval" else get_mbpp_plus()


def answer_part(text: str, thinking: bool) -> tuple[str, bool]:
    if not thinking:
        return text, "</think>" not in text and "<think>" not in text
    if text.count("</think>") != 1:
        return "", False
    return text.split("</think>", 1)[1], True


def gen(a) -> None:
    from evalplus.sanitize import sanitize
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams

    tok = AutoTokenizer.from_pretrained(a.model)
    llm = LLM(a.model, dtype="bfloat16", max_model_len=a.max_model_len, gpu_memory_utilization=0.85, seed=0,
              enforce_eager=True)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    for dataset in a.datasets:
        problems = load_problems(dataset)
        for mode in a.modes:
            thinking = mode == "thinking"
            ids = sorted(problems)
            prompts = [tok.apply_chat_template([{"role": "user", "content": INSTRUCTION.format(
                prompt=problems[t]["prompt"].strip())}], tokenize=False, add_generation_prompt=True,
                enable_thinking=thinking) for t in ids]
            params = SamplingParams(temperature=0.0, max_tokens=a.thinking_tokens if thinking else a.default_tokens)
            results = llm.generate(prompts, params)
            with open(out / f"{dataset}-{mode}.jsonl", "w") as handle:
                for t, r in zip(ids, results):
                    raw = r.outputs[0].text
                    answer, closed = answer_part(raw, thinking)
                    code = sanitize(answer, problems[t]["entry_point"]) if answer.strip() else ""
                    handle.write(json.dumps({"task_id": t, "solution": code, "raw": raw, "closed": closed,
                                             "new_tokens": len(r.outputs[0].token_ids)}) + "\n")
            print(f"[gen] {dataset} {mode}: {len(ids)} problems", flush=True)


def score(a) -> None:
    from evalplus.evaluate import evaluate
    report = {}
    for samples in sorted(Path(a.out).glob("*-*.jsonl")):
        dataset, mode = samples.stem.split("-", 1)
        evaluate(dataset=dataset, samples=str(samples), parallel=a.parallel, i_just_wanna_run=True)
        result = json.loads(Path(str(samples).replace(".jsonl", "_eval_results.json")).read_text())
        rows = [json.loads(line) for line in samples.open()]
        base = plus = 0
        for task_id, entries in result["eval"].items():
            base += entries[0]["base_status"] == "pass"
            plus += entries[0]["base_status"] == "pass" and entries[0]["plus_status"] == "pass"
        n = len(result["eval"])
        report[samples.stem] = {"n": n, "pass@1_base": base / n, "pass@1_plus": plus / n,
                                "closed": sum(r["closed"] for r in rows) / len(rows),
                                "mean_new_tokens": sum(r["new_tokens"] for r in rows) / len(rows)}
        print(samples.stem, report[samples.stem], flush=True)
    (Path(a.out) / "report.json").write_text(json.dumps(report, indent=1) + "\n")


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    g = sub.add_parser("gen")
    g.add_argument("--model", required=True)
    g.add_argument("--out", required=True)
    g.add_argument("--datasets", nargs="+", default=["humaneval", "mbpp"])
    g.add_argument("--modes", nargs="+", default=["default", "thinking"])
    g.add_argument("--default-tokens", type=int, default=2048)
    g.add_argument("--thinking-tokens", type=int, default=8192)
    g.add_argument("--max-model-len", type=int, default=10240)
    s = sub.add_parser("score")
    s.add_argument("--out", required=True)
    s.add_argument("--parallel", type=int, default=os.cpu_count() or 8)
    a = ap.parse_args()
    gen(a) if a.cmd == "gen" else score(a)


if __name__ == "__main__":
    main()
