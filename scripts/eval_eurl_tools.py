#!/usr/bin/env python3
"""L1 baselines for european-rl tool episodes (e.g. meeting_reschedule), default and thinking mode.

Each exported row (`eurl.cli export ENV`) names the environment, seed, split and language; the harness resets that
environment, shows the model the task with the environment's tool schemas through the chat template, executes
Hermes `<tool_call>` blocks against the environment one at a time, feeds results back as tool messages, and
scores the end state when the model answers without a tool call or the step limit is reached. Tasks advance in
lockstep batches through vLLM.
  OELLM_EUROPEAN_RL=<checkout> eval_eurl_tools.py gen --model M --data rows.jsonl --out DIR
  OELLM_EUROPEAN_RL=<checkout> eval_eurl_tools.py score --out DIR
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from collections import defaultdict
from pathlib import Path

CALL = re.compile(r"<tool_call>\s*(.*?)\s*</tool_call>", re.S)


def gen(a) -> None:
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams
    from eurl.core import load
    rows = [json.loads(l) for l in Path(a.data).read_text(encoding="utf-8").split("\n") if l.strip()]
    tok = AutoTokenizer.from_pretrained(a.model)
    llm = LLM(a.model, dtype="bfloat16", max_model_len=a.max_model_len, gpu_memory_utilization=0.88, seed=0,
              enforce_eager=True)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    for mode in a.modes:
        thinking = mode == "thinking"
        params = SamplingParams(temperature=0.0, max_tokens=a.thinking_tokens if thinking else a.default_tokens)
        envs, convs, done, results, steps = [], [], [], [None] * len(rows), [0] * len(rows)
        for r in rows:
            gt = json.loads(r["ground_truth"])
            env = load(gt["env"])
            obs = env.reset(gt["seed"], gt["split"], gt["language"])
            envs.append((env, obs["tools"]))
            convs.append([{"role": "user", "content": obs["instruction"]}])
            done.append(False)
        for _ in range(a.max_rounds):
            active = [i for i in range(len(rows)) if not done[i]]
            if not active:
                break
            prompts = [tok.apply_chat_template(convs[i], tools=envs[i][1], tokenize=False, add_generation_prompt=True,
                                               enable_thinking=thinking) for i in active]
            for i, res in zip(active, llm.generate(prompts, params)):
                text = res.outputs[0].text
                env = envs[i][0]
                visible = text.rsplit("</think>", 1)[1] if "</think>" in text else text
                calls = CALL.findall(visible)
                if not calls or res.outputs[0].finish_reason != "stop":
                    step = env.step({"final": visible.strip()})
                    results[i], done[i] = step, True
                    continue
                content = CALL.sub("", visible).strip().replace("<think>", "").replace("</think>", "")
                parsed = []
                for raw in calls[: a.calls_per_turn]:
                    try:
                        obj = json.loads(raw)
                        parsed.append({"name": obj.get("name"), "arguments": obj.get("arguments") or {}})
                    except json.JSONDecodeError:
                        parsed.append({"name": None, "arguments": {}})
                convs[i].append({"role": "assistant", "content": content,
                                 "tool_calls": [{"type": "function", "function": {"name": c["name"] or "invalid",
                                                                                  "arguments": c["arguments"]}}
                                                for c in parsed]})
                for c in parsed:
                    step = env.step({"tool": c["name"], "arguments": c["arguments"]} if c["name"] else {"x": 1})
                    steps[i] += 1
                    convs[i].append({"role": "tool", "content": json.dumps(step["observation"], ensure_ascii=False)[:4000]})
                    if step["terminated"] or step["truncated"]:
                        results[i], done[i] = step, True
                        break
        for i in range(len(rows)):
            if not done[i]:
                results[i] = envs[i][0].step({"final": ""})
        trace_env = [e for e, _ in envs]
        rows_index = {id(r): i for i, r in enumerate(rows)}
        with open(out / f"{mode}.jsonl", "w", encoding="utf-8") as f:
            for r, res, n in zip(rows, results, steps):
                f.write(json.dumps({**r, "mode": mode, "reward": res["reward"], "tool_steps": n,
                                    "info": {k: v for k, v in res["info"].items() if k != "meta"},
                                    "trace": [t.get("action") for t in getattr(trace_env[rows_index[id(r)]], "trace", [])][:40]},
                                   ensure_ascii=False, default=str) + "\n")
        print(f"[gen] {mode}: {len(rows)} episodes", flush=True)


def score(a) -> None:
    report = {}
    for path in sorted(Path(a.out).glob("*.jsonl")):
        groups = defaultdict(list)
        for line in path.read_text(encoding="utf-8").split("\n"):
            if line.strip():
                r = json.loads(line)
                env = json.loads(r["ground_truth"])["env"]
                for key in (f"{env}/ALL", f"{env}/lang:{r['language']}", f"{env}/level:{r.get('level')}"):
                    groups[key].append((r["reward"], r["info"].get("score", 0.0), r["tool_steps"]))
        report[path.stem] = {k: {"n": len(v), "reward": round(sum(x[0] for x in v) / len(v), 3),
                                 "score": round(sum(x[1] for x in v) / len(v), 3),
                                 "tool_steps": round(sum(x[2] for x in v) / len(v), 1)} for k, v in sorted(groups.items())}
        print(f"\n== {path.stem}")
        for k, v in report[path.stem].items():
            print(f"{k:<48} n={v['n']:<5} reward {v['reward']:.3f}  score {v['score']:.3f}  steps {v['tool_steps']}")
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
    g.add_argument("--thinking-tokens", type=int, default=4096)
    g.add_argument("--max-model-len", type=int, default=24576)
    g.add_argument("--max-rounds", type=int, default=24)
    g.add_argument("--calls-per-turn", type=int, default=4)
    s = sub.add_parser("score")
    s.add_argument("--out", required=True)
    a = ap.parse_args()
    {"gen": gen, "score": score}[a.cmd](a)


if __name__ == "__main__":
    main()
