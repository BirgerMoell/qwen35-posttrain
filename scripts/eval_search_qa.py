#!/usr/bin/env python3
"""Search-augmented factual QA eval (hallucination plan E7): eu_qa / hallu-v1 questions with an offline Wikipedia
search tool (oellm-rlvr oellm_rlvr.wiki_search; server started by slurm/eval_search_qa_leonardo.sbatch).

The model gets the `wiki_search` tool through the chat template's `tools` (hermes <tool_call> format). Each turn
is generated greedily; a reply with tool calls gets the search results as `tool` messages, up to --max-calls
rounds. The final reply is graded with eu_tasks.grade_qa (the RL reward code): correct / abstain / wrong.
Reported per condition and mode: the three rates, hallucination rate, how often the model searched, and accuracy
for questions with and without a search. Without search the same questions are in eval_eu_tasks.py (report.json
of the hallu-v1 run), so the two runs compare directly.
  python eval_search_qa.py --model M --data hallu-v1/eval.jsonl --out OUT --endpoint http://127.0.0.1:18090/search
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import urllib.request
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from oellm_rlvr import eu_tasks as E
from oellm_rlvr.wiki_search import LANGUAGES, TOOL_DESCRIPTION, TOOL_PARAMETERS, parse_tool_calls

TOOL = {"type": "function", "function": {"name": "wiki_search", "description": TOOL_DESCRIPTION,
                                          "parameters": TOOL_PARAMETERS}}
CALL_BLOCK = re.compile(r"<tool_call>.*?</tool_call>", re.DOTALL)


def search(endpoint: str, args: dict) -> str:
    query = str(args.get("query") or "").strip()
    if not query:
        return "Empty query."
    language = args.get("language") if args.get("language") in LANGUAGES else "en"
    body = json.dumps({"query": query, "language": language, "k": 5}).encode()
    req = urllib.request.Request(endpoint, data=body, headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=30) as r:
            return json.loads(r.read())["text"]
    except Exception as e:  # reported to the model like a tool error
        return f"Search failed: {e}"


def run_mode(llm, tok, rows, mode: str, a) -> list[dict]:
    from vllm import SamplingParams

    thinking = mode == "thinking"
    params = SamplingParams(temperature=0.0, max_tokens=a.thinking_tokens if thinking else a.default_tokens)
    convs = [[{"role": "user", "content": r["prompt"]}] for r in rows]
    trace = [[] for _ in rows]
    final: list[str | None] = [None] * len(rows)
    active = list(range(len(rows)))
    pool = ThreadPoolExecutor(32)
    for round_ in range(a.max_calls + 1):
        if not active:
            break
        prompts = [tok.apply_chat_template(convs[i], tools=[TOOL], tokenize=False, add_generation_prompt=True,
                                           enable_thinking=thinking) for i in active]
        outs = llm.generate(prompts, params)
        still = []
        jobs = []
        for i, out in zip(active, outs):
            text = out.outputs[0].text
            calls = [c for c in parse_tool_calls(text) if c["name"] == "wiki_search"]
            if calls and round_ < a.max_calls and out.outputs[0].finish_reason == "stop":
                calls = calls[: a.calls_per_turn]
                content = CALL_BLOCK.sub("", text).strip()
                if "</think>" in content:
                    content = content.rsplit("</think>", 1)[1].strip()
                # an unclosed or stray think tag (either mode) would make the template reject the history
                content = content.replace("<think>", "").replace("</think>", "").strip()
                convs[i].append({"role": "assistant", "content": content,
                                 "tool_calls": [{"type": "function", "function": {"name": c["name"], "arguments": c["arguments"]}}
                                                for c in calls]})
                for c in calls:
                    jobs.append((i, c, pool.submit(search, a.endpoint, c["arguments"])))
                still.append(i)
            else:
                final[i] = text
                trace[i].append({"final": text, "finished": out.outputs[0].finish_reason == "stop"})
        for i, c, fut in jobs:
            result = fut.result()
            convs[i].append({"role": "tool", "content": result})
            trace[i].append({"call": c["arguments"], "result": result[:600]})
        active = still
    results = []
    for r, f, t in zip(rows, final, trace):
        calls = [s["call"] for s in t if "call" in s]
        results.append({**{k: r[k] for k in ("id", "condition", "band", "subset", "language", "label")},
                        "mode": mode, "final": f or "", "calls": calls, "trace": t})
    return results


def score(results: list[dict]) -> dict:
    rep: dict = defaultdict(lambda: Counter())
    for r in results:
        g = E.grade_qa(r["final"], {**r["label"], "mode": r["mode"]})
        g = "wrong" if g == "no_answer" else g
        searched = "searched" if r["calls"] else "no_search"
        for key in (f"{r['mode']}/{r['condition']}/ALL", f"{r['mode']}/{r['condition']}/band:{r['band']}",
                    f"{r['mode']}/{r['condition']}/lang:{r['language']}", f"{r['mode']}/{r['condition']}/{searched}"):
            c = rep[key]
            c[g] += 1
            c["n"] += 1
            c["searched"] += bool(r["calls"])
            c["calls"] += len(r["calls"])
    out = {}
    for key, c in sorted(rep.items()):
        n = c["n"]
        out[key] = {"n": n, "accuracy": c["correct"] / n, "abstain": c["abstain"] / n, "wrong": c["wrong"] / n,
                    "hallucination_rate": c["wrong"] / max(1, n - c["correct"]), "search_rate": c["searched"] / n,
                    "mean_calls": c["calls"] / n, "index": 100 * (c["correct"] - c["wrong"]) / n}
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--endpoint", default="http://127.0.0.1:18090/search")
    ap.add_argument("--conditions", nargs="+", default=["standard", "fabricated"])
    ap.add_argument("--modes", nargs="+", default=["default", "thinking"])
    ap.add_argument("--max-calls", type=int, default=3, help="search rounds before the answer is forced")
    ap.add_argument("--calls-per-turn", type=int, default=2)
    ap.add_argument("--default-tokens", type=int, default=1024)
    ap.add_argument("--thinking-tokens", type=int, default=6144)
    ap.add_argument("--max-model-len", type=int, default=16384)
    a = ap.parse_args()
    from transformers import AutoTokenizer
    from vllm import LLM

    rows = [r for r in map(json.loads, Path(a.data).read_text().splitlines())
            if r["suite"] == "qa" and r.get("condition", "standard") in a.conditions]
    for r in rows:
        r.setdefault("condition", "standard")
    tok = AutoTokenizer.from_pretrained(a.model)
    llm = LLM(a.model, dtype="bfloat16", max_model_len=a.max_model_len, gpu_memory_utilization=0.85, seed=0,
              enforce_eager=True)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    allres = []
    for mode in a.modes:
        res = run_mode(llm, tok, rows, mode, a)
        with open(out / f"{mode}.jsonl", "w") as f:
            for r in res:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        allres += res
        print(f"[gen] {mode}: {len(res)} questions", flush=True)
    report = score(allres)
    (out / "report.json").write_text(json.dumps(report, indent=1) + "\n")
    print(f"{'key':<40}{'n':>5}{'acc':>7}{'abst':>7}{'wrong':>7}{'halluc':>8}{'search':>8}{'calls':>7}")
    for key, m in report.items():
        if "/lang:" not in key:
            print(f"{key:<40}{m['n']:>5}{m['accuracy']:>7.3f}{m['abstain']:>7.3f}{m['wrong']:>7.3f}"
                  f"{m['hallucination_rate']:>8.3f}{m['search_rate']:>8.3f}{m['mean_calls']:>7.2f}")


if __name__ == "__main__":
    sys.exit(main())
