#!/usr/bin/env python3
"""EU task suites eval: calibrated factual QA (eu_qa) and document tasks (eu_doc), default and thinking mode.

The eval set and the scoring come from oellm-rlvr (scripts/build_eu_task_suites.py, src/oellm_rlvr/eu_tasks.py);
eu_tasks.py is copied next to this script on the cluster, so the eval scores exactly what the RL verifiers reward.

`gen` (GPU, vLLM venv): renders each prompt with the model's chat template (`enable_thinking` set explicitly),
generates greedily and writes <out>/<mode>.jsonl. `score` reports, per mode:
- eu_qa: accuracy, abstention, wrong (incl. no ANSWER line), hallucination rate = wrong / (n - correct) and the
  knowledge index 100 * (correct - wrong) / n (AA-Omniscience style; 0 = always abstain), by popularity band, type
  and language;
- eu_doc: mean score (JSON: field fraction), exact rate, by family, variant (held-out variants marked), level and
  language; plus the rate of replies without a final answer.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))


def gen(a) -> None:
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams
    rows = [json.loads(l) for l in Path(a.data).read_text().splitlines()]
    tok = AutoTokenizer.from_pretrained(a.model)
    llm = LLM(a.model, dtype="bfloat16", max_model_len=a.max_model_len, gpu_memory_utilization=0.88, seed=0, enforce_eager=True)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    for mode in a.modes:
        thinking = mode == "thinking"
        prompts = [tok.apply_chat_template([{"role": "user", "content": r["prompt"]}], tokenize=False, add_generation_prompt=True,
                                           enable_thinking=thinking) for r in rows]
        params = SamplingParams(temperature=0.0, max_tokens=a.thinking_tokens if thinking else a.default_tokens)
        results = llm.generate(prompts, params)
        with open(out / f"{mode}.jsonl", "w") as f:
            for r, res in zip(rows, results):
                o = res.outputs[0]
                # Templates that pre-fill "<think>" in the generation prompt: the reply then holds only "...</think>".
                f.write(json.dumps({**r, "mode": mode, "raw": o.text, "new_tokens": len(o.token_ids),
                                    "finished": o.finish_reason == "stop"}, ensure_ascii=False) + "\n")
        print(f"[gen] {mode}: {len(rows)} prompts", flush=True)


def score(a) -> None:
    import eu_tasks as E
    report = {}
    for path in sorted(Path(a.out).glob("*.jsonl")):
        mode = path.stem
        rows = [json.loads(l) for l in path.read_text().splitlines()]
        qa = defaultdict(lambda: defaultdict(int))
        doc = defaultdict(list)
        for r in rows:
            label = {**r["label"], "mode": mode}
            if r["suite"] == "qa":
                g = E.grade_qa(r["raw"], label)
                g = "wrong" if g == "no_answer" else g
                for key in ("ALL", f"band:{r['band']}", f"type:{r['subset']}", f"lang:{r['language']}"):
                    qa[key][g] += 1
                    qa[key]["no_answer"] += E.final_answer(E.answer_part(r["raw"], mode) or "") is None
            else:
                s = E.score_doc(r["raw"], label)
                text = E.answer_part(r["raw"], mode) or ""
                missing = (E.last_json_object(text) if r["label"]["kind"] == "json" else E.final_answer(text)) is None
                held = "held_out" if r.get("held_out_variant") else "seen"
                for key in ("ALL", f"variants:{held}", f"family:{r['subset']}", f"variant:{r['subset']}/{r['variant']}{'*' if held == 'held_out' else ''}",
                            f"level:{r['level']}", f"lang:{r['language']}"):
                    doc[key].append((s, missing, r["new_tokens"]))
        qa_rep = {}
        for key, c in qa.items():
            n = c["correct"] + c["abstain"] + c["wrong"]
            qa_rep[key] = {"n": n, "accuracy": c["correct"] / n, "abstain": c["abstain"] / n, "wrong": c["wrong"] / n,
                           "no_answer_line": c["no_answer"] / n,
                           "hallucination_rate": c["wrong"] / max(1, n - c["correct"]), "index": 100 * (c["correct"] - c["wrong"]) / n}
        doc_rep = {key: {"n": len(v), "score": sum(x[0] for x in v) / len(v), "exact": sum(x[0] == 1.0 for x in v) / len(v),
                         "no_answer": sum(x[1] for x in v) / len(v), "mean_new_tokens": sum(x[2] for x in v) / len(v)} for key, v in doc.items()}
        report[mode] = {"qa": dict(sorted(qa_rep.items())), "doc": dict(sorted(doc_rep.items())),
                        "finished": sum(r["finished"] for r in rows) / len(rows)}
        print(f"== {mode}  (finished {report[mode]['finished']:.3f})")
        print(f"  {'eu_qa':<28}{'n':>5}{'acc':>7}{'abst':>7}{'wrong':>7}{'halluc':>8}{'index':>8}")
        for key, m in report[mode]["qa"].items():
            if key == "ALL" or key.startswith(("band:", "type:")):
                print(f"  {key:<28}{m['n']:>5}{m['accuracy']:>7.3f}{m['abstain']:>7.3f}{m['wrong']:>7.3f}{m['hallucination_rate']:>8.3f}{m['index']:>8.1f}")
        print(f"  {'eu_doc':<28}{'n':>5}{'score':>7}{'exact':>7}{'noans':>7}{'tokens':>8}")
        for key, m in report[mode]["doc"].items():
            if not key.startswith("lang:"):
                print(f"  {key:<28}{m['n']:>5}{m['score']:>7.3f}{m['exact']:>7.3f}{m['no_answer']:>7.3f}{m['mean_new_tokens']:>8.0f}")
    (Path(a.out) / "report.json").write_text(json.dumps(report, indent=1) + "\n")


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    g = sub.add_parser("gen")
    g.add_argument("--model", required=True)
    g.add_argument("--data", required=True)
    g.add_argument("--out", required=True)
    g.add_argument("--modes", nargs="+", default=["default", "thinking"])
    g.add_argument("--default-tokens", type=int, default=2048)
    g.add_argument("--thinking-tokens", type=int, default=8192)
    g.add_argument("--max-model-len", type=int, default=12288)
    s = sub.add_parser("score")
    s.add_argument("--out", required=True)
    a = ap.parse_args()
    gen(a) if a.cmd == "gen" else score(a)


if __name__ == "__main__":
    main()
