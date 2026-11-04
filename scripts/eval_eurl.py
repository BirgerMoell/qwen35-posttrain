#!/usr/bin/env python3
"""L1 baselines for european-rl typed environments (github.com/BirgerMoell/european-rl), default and thinking mode.

Rows come from `python -m eurl.cli export ENV --split eval ...` (chat `messages`, JSON `ground_truth` with the answer
spec). `gen` (GPU, vLLM venv) renders each prompt with the model's chat template (`enable_thinking` set explicitly)
and generates greedily; `score` uses the same function as the RL verifier (oellm-rlvr `compat.eurl_score`: the
environment's own scorer plus the mode rules), reporting mean reward and the share of replies without an ANSWER line
per environment, language, locale and level. A useful training environment sits at roughly 10–90 % for the model.
  OELLM_EUROPEAN_RL=<checkout> eval_eurl.py gen --model M --data rows.jsonl --out DIR
  OELLM_EUROPEAN_RL=<checkout> eval_eurl.py score --out DIR
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from collections import defaultdict
from pathlib import Path


def gen(a) -> None:
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams
    rows = [json.loads(l) for l in Path(a.data).read_text(encoding="utf-8").split("\n") if l.strip()]
    tok = AutoTokenizer.from_pretrained(a.model)
    llm = LLM(a.model, dtype="bfloat16", max_model_len=a.max_model_len, gpu_memory_utilization=0.88, seed=0,
              enforce_eager=True)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    for mode in a.modes:
        thinking = mode == "thinking"
        prompts = [tok.apply_chat_template(r["messages"], tokenize=False, add_generation_prompt=True,
                                           enable_thinking=thinking) for r in rows]
        params = SamplingParams(temperature=0.0, max_tokens=a.thinking_tokens if thinking else a.default_tokens)
        with open(out / f"{mode}.jsonl", "w", encoding="utf-8") as f:
            for r, res in zip(rows, llm.generate(prompts, params)):
                o = res.outputs[0]
                f.write(json.dumps({**r, "mode": mode, "raw": o.text, "new_tokens": len(o.token_ids),
                                    "finished": o.finish_reason == "stop"}, ensure_ascii=False) + "\n")
        print(f"[gen] {mode}: {len(rows)} prompts", flush=True)


def eurl_score(prediction: str, contract: dict) -> float:
    """Same rule as oellm-rlvr compat.eurl_score (kept identical; the RL verifier is the reference)."""
    from eurl.answers import score_answer
    text = (prediction or "").replace("<|im_end|>", "")
    mode = contract.get("mode", "default")
    if mode == "default" and ("<think>" in text or "</think>" in text):
        return 0.0
    if mode == "thinking" and (text.count("</think>") != 1 or "<think>" in text.split("</think>", 1)[1]):
        return 0.0
    return float(score_answer(contract["spec"], text)["reward"])


def score(a) -> None:
    from eurl.answers import final_answer, strip_reasoning
    report = {}
    for path in sorted(Path(a.out).glob("*.jsonl")):
        mode = path.stem
        groups = defaultdict(list)
        for line in path.read_text(encoding="utf-8").split("\n"):
            if not line.strip():
                continue
            r = json.loads(line)
            contract = {**json.loads(r["ground_truth"]), "mode": mode}
            reward = eurl_score(r["raw"], contract)
            no_answer = final_answer(strip_reasoning(r["raw"])) is None
            env = contract["env"]
            for key in (f"{env}/ALL", f"{env}/lang:{r['language']}", f"{env}/locale:{r['locale']}",
                        f"{env}/level:{r.get('level')}", f"{env}/abstain_task:{contract['spec']['type'] == 'abstain'}"):
                groups[key].append((reward, no_answer, r["new_tokens"]))
        report[mode] = {k: {"n": len(v), "reward": round(sum(x[0] for x in v) / len(v), 3),
                            "no_answer": round(sum(x[1] for x in v) / len(v), 3),
                            "mean_new_tokens": round(sum(x[2] for x in v) / len(v), 1)}
                        for k, v in sorted(groups.items())}
        print(f"\n== {mode}")
        for k, v in report[mode].items():
            print(f"{k:<52} n={v['n']:<5} reward {v['reward']:.3f}  no_answer {v['no_answer']:.3f}  tokens {v['mean_new_tokens']}")
    (Path(a.out) / "report.json").write_text(json.dumps(report, indent=1) + "\n")


def main() -> None:
    root = os.environ.get("OELLM_EUROPEAN_RL")
    if root:
        sys.path.insert(0, root)
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    g = sub.add_parser("gen")
    g.add_argument("--model", required=True)
    g.add_argument("--data", required=True)
    g.add_argument("--out", required=True)
    g.add_argument("--modes", nargs="+", default=["default", "thinking"])
    g.add_argument("--default-tokens", type=int, default=1536)
    g.add_argument("--thinking-tokens", type=int, default=6144)
    g.add_argument("--max-model-len", type=int, default=12288)
    s = sub.add_parser("score")
    s.add_argument("--out", required=True)
    a = ap.parse_args()
    {"gen": gen, "score": score}[a.cmd](a)


if __name__ == "__main__":
    main()
