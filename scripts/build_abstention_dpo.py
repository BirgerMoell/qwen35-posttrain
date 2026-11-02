#!/usr/bin/env python3
"""On-policy abstention DPO pairs from self-consistency (hallucination plan, OELLM_HALLUCINATION_PLAN.md, E4).

Default mode only (the Stage-4 DPO pipeline renders without think tokens). Phases, all in --work:
  prompts  eu_qa train questions (oellm-rlvr data/eu-tasks/rl_qa.parquet; disjoint from the eval by subject) plus
           fabricated-entity questions (a seed and subjects disjoint from hallu-v1) -> prompts.jsonl
  sample   k replies per prompt from the policy at temperature 1 (vLLM, enable_thinking=False) -> samples.jsonl
  pairs    grade every reply with eu_tasks.grade_qa; agreement = share of the k replies giving the same answer
           (oellm_rlvr.hallucination.answer_key), then:
             unknown    agreement of the majority answer <= 2/8 and the majority answer wrong:
                        chosen = a sampled abstention, else "ANSWER: UNKNOWN"; rejected = a sampled wrong reply
             fabricated chosen = a sampled abstention, else "ANSWER: UNKNOWN"; rejected = an invented answer
             known      majority answer correct with agreement >= 6/8 (the over-refusal guard):
                        chosen = a correct reply; rejected = a sampled abstention, else "ANSWER: UNKNOWN"
           known pairs are capped/filled to --known-share of the total; writes staged/abstention-{train,validation}
           .jsonl.gz in the stage_oellm_dpo_v1 schema and pairs-manifest.json.
  python build_abstention_dpo.py prompts --work W --rl-qa rl_qa.parquet --eval hallu-v1/eval.jsonl --qa 2400 --fabricated 600
  python build_abstention_dpo.py sample  --work W --policy MODEL_DIR --k 8
  python build_abstention_dpo.py pairs   --work W
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import random
from collections import Counter, defaultdict
from pathlib import Path

from oellm_rlvr import eu_tasks as E
from oellm_rlvr.hallucination import ABSTAIN_KEY, NO_ANSWER_KEY, answer_key, fabricated_qa

UNKNOWN_REPLY = "ANSWER: UNKNOWN"


def h(x: str) -> float:
    return int(hashlib.sha256(x.encode()).hexdigest()[:8], 16) / 0xFFFFFFFF


def phase_prompts(a) -> None:
    import pyarrow.parquet as pq

    rng = random.Random(a.seed)
    eval_rows = [json.loads(line) for line in Path(a.eval).read_text().splitlines()]
    eval_subjects = {r.get("subject") for r in eval_rows if r.get("subject")}
    eval_ids = {r["id"].rsplit("-", 1)[0] for r in eval_rows}
    qa = [r for r in pq.read_table(a.rl_qa).to_pylist() if r["dataset"] == "eu_qa"]
    qa = [r for r in qa if r["id"].rsplit("-", 1)[0] not in eval_ids]
    rng.shuffle(qa)
    rows = []
    for r in qa[: a.qa]:
        label = json.loads(r["ground_truth"])
        rows.append({"id": r["id"], "kind": "qa", "language": r["language"], "prompt": r["messages"][-1]["content"],
                     "label": {**label, "mode": "default"}})
    per_lang = -(-a.fabricated // len(E.LANGS))
    fab = [r for r in fabricated_qa(2 * per_lang, seed=a.seed) if r["subject"] not in eval_subjects]
    by_lang = Counter()
    for r in fab:
        if by_lang[r["language"]] < per_lang:
            by_lang[r["language"]] += 1
            rows.append({"id": r["id"] + "-train", "kind": "fabricated", "language": r["language"],
                         "prompt": r["prompt"], "label": {**r["label"], "mode": "default"}})
    a.work.mkdir(parents=True, exist_ok=True)
    with (a.work / "prompts.jsonl").open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(json.dumps({"prompts": len(rows), "by_kind": dict(Counter(r["kind"] for r in rows)),
                      "by_language": dict(Counter(r["language"] for r in rows))}))


def phase_sample(a) -> None:
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams

    rows = [json.loads(line) for line in (a.work / "prompts.jsonl").read_text().splitlines()]
    tok = AutoTokenizer.from_pretrained(a.policy)
    prompts = [tok.apply_chat_template([{"role": "user", "content": r["prompt"]}], tokenize=False,
                                       add_generation_prompt=True, enable_thinking=False) for r in rows]
    llm = LLM(a.policy, dtype="bfloat16", max_model_len=4096, gpu_memory_utilization=0.88, seed=a.seed, enforce_eager=True)
    params = SamplingParams(n=a.k, temperature=1.0, top_p=1.0, max_tokens=a.max_tokens, seed=a.seed)
    with (a.work / "samples.jsonl").open("w", encoding="utf-8") as f:
        for r, res in zip(rows, llm.generate(prompts, params)):
            f.write(json.dumps({"id": r["id"], "replies": [o.text for o in res.outputs],
                                "finished": [o.finish_reason == "stop" for o in res.outputs]}, ensure_ascii=False) + "\n")
    (a.work / "sample-manifest.json").write_text(json.dumps({"policy": a.policy, "k": a.k, "temperature": 1.0,
                                                             "max_tokens": a.max_tokens, "seed": a.seed}, indent=1))
    print(f"sampled {len(rows)} prompts x {a.k}")


def phase_pairs(a) -> None:
    rng = random.Random(a.seed)
    prompts = {r["id"]: r for r in map(json.loads, (a.work / "prompts.jsonl").read_text().splitlines())}
    groups = [json.loads(line) for line in (a.work / "samples.jsonl").read_text().splitlines()]
    pairs = {"unknown": [], "fabricated": [], "known": []}
    stats = Counter()
    for g in groups:
        p = prompts[g["id"]]
        label = p["label"]
        replies = [t.strip() for t, done in zip(g["replies"], g["finished"]) if done and t.strip()]
        if len(replies) < len(g["replies"]) // 2:
            stats["too_many_unfinished"] += 1
            continue
        graded = [(t, E.grade_qa(t, label), answer_key(t, label)) for t in replies]
        abstains = [t for t, grade, _ in graded if grade == "abstain"]
        answered = Counter(k for _, _, k in graded if k not in (ABSTAIN_KEY, NO_ANSWER_KEY))
        top_key, top_n = answered.most_common(1)[0] if answered else (None, 0)
        agreement = top_n / len(g["replies"])
        top_grade = next((grade for _, grade, k in graded if k == top_key), None)
        abstain_reply = rng.choice(abstains) if abstains else UNKNOWN_REPLY
        wrong = [t for t, grade, k in graded if grade == "wrong" and k != NO_ANSWER_KEY]
        correct = [t for t, grade, _ in graded if grade == "correct"]
        if p["kind"] == "fabricated":
            case = "fabricated" if wrong else "fabricated_already_abstains"
            if wrong:
                pairs["fabricated"].append((p, abstain_reply, rng.choice(wrong)))
        elif top_grade == "wrong" and agreement <= 2 / 8 and wrong:
            case = "unknown"
            pairs["unknown"].append((p, abstain_reply, rng.choice(wrong)))
        elif top_grade == "correct" and agreement >= 6 / 8 and correct:
            case = "known"
            pairs["known"].append((p, rng.choice(correct), abstain_reply))
        else:
            case = "no_pair"
        stats[f"{p['kind']}|{case}"] += 1
        stats[f"{p['kind']}|abstain_sampled_{bool(abstains)}"] += 1
    # Known pairs: about known_share of all pairs (fewer if there are not enough known prompts).
    abstain_pairs = pairs["unknown"] + pairs["fabricated"]
    want_known = round(a.known_share / (1 - a.known_share) * len(abstain_pairs))
    rng.shuffle(pairs["known"])
    known = pairs["known"][:want_known]
    rows = []
    for kind, items in (("unknown", pairs["unknown"]), ("fabricated", pairs["fabricated"]), ("known", known)):
        for p, chosen, rejected in items:
            rows.append({"id": f"abstention-{kind}-{p['id']}", "prompt": [{"role": "user", "content": p["prompt"]}],
                         "chosen": [{"role": "assistant", "content": chosen}],
                         "rejected": [{"role": "assistant", "content": rejected}],
                         "source": f"oellm-abstention-v1:{kind}", "language": p["language"]})
    rng.shuffle(rows)
    staged = a.work / "staged"
    staged.mkdir(parents=True, exist_ok=True)
    split = {"train": [r for r in rows if h(r["id"]) >= 0.05], "validation": [r for r in rows if h(r["id"]) < 0.05]}
    for name, part in split.items():
        with gzip.open(staged / f"abstention-{name}.jsonl.gz", "wt", encoding="utf-8") as f:
            for r in part:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
    manifest = {"kind": "oellm_abstention_onpolicy_pairs_v1", "work": str(a.work),
                "sample": json.loads((a.work / "sample-manifest.json").read_text()),
                "pairs": len(rows), "by_kind": dict(Counter(r["source"].split(":")[1] for r in rows)),
                "known_available": len(pairs["known"]), "known_share_target": a.known_share,
                "splits": {k: len(v) for k, v in split.items()},
                "chosen_constructed_unknown_reply": sum(1 for r in rows if UNKNOWN_REPLY == r["chosen"][0]["content"]),
                "by_language": dict(Counter(r["language"] for r in rows)), "stats": dict(sorted(stats.items()))}
    (a.work / "pairs-manifest.json").write_text(json.dumps(manifest, indent=1) + "\n")
    print(json.dumps({k: v for k, v in manifest.items() if k != "stats"}, indent=1))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("phase", choices=["prompts", "sample", "pairs"])
    ap.add_argument("--work", type=Path, required=True)
    ap.add_argument("--rl-qa", type=Path)
    ap.add_argument("--eval", type=Path, help="hallu-v1 eval.jsonl (its questions and subjects are excluded)")
    ap.add_argument("--qa", type=int, default=2400)
    ap.add_argument("--fabricated", type=int, default=600)
    ap.add_argument("--policy", default="")
    ap.add_argument("--k", type=int, default=8)
    ap.add_argument("--max-tokens", type=int, default=384)
    ap.add_argument("--known-share", type=float, default=1 / 3)
    ap.add_argument("--seed", type=int, default=20261103)
    a = ap.parse_args()
    {"prompts": phase_prompts, "sample": phase_sample, "pairs": phase_pairs}[a.phase](a)


if __name__ == "__main__":
    main()
