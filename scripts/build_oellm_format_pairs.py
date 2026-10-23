#!/usr/bin/env python3
"""Stage-4 format-compliance preference pairs (default mode), on-policy from Stage-3 v1.1.

Why: EUROPA Eval shows the Stage-3 model ignores strict answer formats ("answer only with the letter/number"); the
instruct data taught it to (playbook §8.3). These pairs prefer the bare, verified answer over the model's own verbose
or wrong reply to prompts that ask for exactly that.

Sources (instructions are native in each source, so nothing is machine translated):
- letter-only MCQ: `birgermoell/oellm-eu-exam-mcq-v1` GRPO train rows from `exams_qa` (CC-BY-SA-4.0) and
  `slovak_financial_exam` (CC-BY-SA-4.0) only. Excluded on purpose: belebele, global_mmlu, mmmlu, xcopa (they are
  evaluation benchmarks we report), and every source whose license is unknown, non-commercial or under review.
  Prompts end with a native "answer only with the letter" instruction; chosen = the gold letter.
- final-answer-only math: `birgermoell/oellm-math-rlvr` train (Apache-2.0), 24 EU languages, prompts end with a native
  "give only the final answer in \\boxed{}" instruction; chosen = `\\boxed{<answer>}`.

Phases:
  select   (CPU, login node for the downloads): balanced prompt sample per language -> prompts.jsonl
  generate (1 GPU, vLLM venv): Stage-3 v1.1 greedy replies in default mode (enable_thinking=False, 512 tokens); a pair
           is kept only when the reply finished (EOS) and is not already the bare correct answer. Writes
           format-train.jsonl.gz / format-validation.jsonl.gz in the staged-DPO schema (id, prompt, chosen, rejected,
           source, language) plus manifest.json with compliance rates per source and language.
"""
from __future__ import annotations

import argparse
import collections
import gzip
import hashlib
import json
import random
import re
from pathlib import Path

MCQ_REPO, MCQ_REV = "birgermoell/oellm-eu-exam-mcq-v1", "12a22a620a"
MATH_REPO, MATH_REV = "birgermoell/oellm-math-rlvr", "0ffc9d6dc82717c25733b3172f4dbd63e48bab68"
MCQ_SOURCES = {"exams_qa", "slovak_financial_exam"}
MCQ_LICENSES = {"CC-BY-SA-4.0"}


def bucket(key: str, salt: str) -> float:
    return int(hashlib.sha256((key + salt).encode()).hexdigest()[:12], 16) / 16 ** 12


def select(a) -> None:
    from huggingface_hub import hf_hub_download
    import pyarrow.parquet as pq

    rng = random.Random(a.seed)
    out = []
    mcq_path = hf_hub_download(MCQ_REPO, "grpo/train.jsonl", repo_type="dataset", revision=MCQ_REV)
    by_lang = collections.defaultdict(list)
    with open(mcq_path, encoding="utf-8") as handle:
        for line in handle:
            r = json.loads(line)
            if r["source_id"] in MCQ_SOURCES and r["source_license"] in MCQ_LICENSES and re.fullmatch(r"[A-J]", r["answer"]):
                by_lang[r["language"]].append(r)
    for lang, rows in sorted(by_lang.items()):
        rows.sort(key=lambda r: r["id"])
        rng.shuffle(rows)
        for r in rows[: a.per_language_mcq]:
            out.append({"id": f"fmt-mcq-{r['id']}", "kind": "mcq_letter", "language": lang,
                        "source": f"{MCQ_REPO}:{r['source_id']}", "license": r["source_license"],
                        "prompt": r["prompt"], "answer": r["answer"]})
    math_path = hf_hub_download(MATH_REPO, "data/train-00000-of-00001.parquet", repo_type="dataset", revision=MATH_REV)
    table = pq.read_table(math_path, columns=["id", "messages", "ground_truth", "language", "difficulty"])
    by_lang = collections.defaultdict(list)
    for r in table.to_pylist():
        if r["language"] != "en" and r["difficulty"] <= 3:
            by_lang[r["language"]].append(r)
    for lang, rows in sorted(by_lang.items()):
        rows.sort(key=lambda r: r["id"])
        rng.shuffle(rows)
        for r in rows[: a.per_language_math]:
            content = r["messages"][0]["content"] if isinstance(r["messages"], list) else json.loads(r["messages"])[0]["content"]
            out.append({"id": f"fmt-math-{r['id']}", "kind": "math_final_only", "language": lang,
                        "source": MATH_REPO, "license": "Apache-2.0", "prompt": content,
                        "answer": str(r["ground_truth"])})
    Path(a.out).mkdir(parents=True, exist_ok=True)
    with open(Path(a.out) / "prompts.jsonl", "w", encoding="utf-8") as handle:
        for r in out:
            handle.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(json.dumps(collections.Counter((r["kind"], r["language"]) for r in out).most_common(), ensure_ascii=False))
    print("prompts:", len(out))


def chosen_text(r: dict) -> str:
    return r["answer"] if r["kind"] == "mcq_letter" else "\\boxed{%s}" % r["answer"]


def is_bare_correct(reply: str, r: dict) -> bool:
    text = reply.strip().rstrip(".").strip()
    if r["kind"] == "mcq_letter":
        return text.upper() in {r["answer"], f"{r['answer']})", f"({r['answer']})"}
    return re.sub(r"\s+", "", text) == re.sub(r"\s+", "", chosen_text(r))


def generate(a) -> None:
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams

    rows = [json.loads(line) for line in open(Path(a.out) / "prompts.jsonl", encoding="utf-8")]
    tok = AutoTokenizer.from_pretrained(a.model)
    template = Path(a.template).read_text(encoding="utf-8")
    bundled = Path(a.model) / "chat_template.jinja"
    if bundled.exists() and bundled.read_text(encoding="utf-8") != template:
        raise SystemExit("model's bundled chat template differs from --template")
    tok.chat_template = template
    prompts = [tok.apply_chat_template([{"role": "user", "content": r["prompt"]}], tokenize=False,
                                       add_generation_prompt=True, enable_thinking=False) for r in rows]
    llm = LLM(a.model, dtype="bfloat16", max_model_len=4096, gpu_memory_utilization=0.85, seed=0, enforce_eager=True)
    results = llm.generate(prompts, SamplingParams(temperature=0.0, max_tokens=a.max_tokens))
    stats = collections.defaultdict(collections.Counter)
    out_dir = Path(a.out)
    handles = {split: gzip.open(out_dir / f"format-{split}.jsonl.gz", "wt", encoding="utf-8")
               for split in ("train", "validation")}
    for r, res in zip(rows, results):
        o = res.outputs[0]
        reply = o.text
        key = f"{r['kind']}|{r['language']}"
        stats[key]["prompts"] += 1
        if o.finish_reason != "stop":
            stats[key]["dropped_unfinished"] += 1
            continue
        if "<think>" in reply or "</think>" in reply:
            stats[key]["dropped_think_tag"] += 1
            continue
        if is_bare_correct(reply, r):
            stats[key]["already_compliant"] += 1
            continue
        stats[key]["pairs"] += 1
        split = "validation" if bucket(r["id"], "#fmtval") < 0.02 else "train"
        handles[split].write(json.dumps({
            "id": r["id"], "prompt": [{"role": "user", "content": r["prompt"]}],
            "chosen": [{"role": "assistant", "content": chosen_text(r)}],
            "rejected": [{"role": "assistant", "content": reply.strip()}],
            "source": r["source"], "language": r["language"]}, ensure_ascii=False) + "\n")
    for h in handles.values():
        h.close()
    totals = collections.Counter()
    for v in stats.values():
        totals.update(v)
    manifest = {"kind": "oellm_format_pairs_v1", "model": a.model, "template": a.template,
                "template_sha256": hashlib.sha256(template.encode()).hexdigest(),
                "generation": {"mode": "default", "temperature": 0.0, "max_tokens": a.max_tokens},
                "sources": {"mcq": f"{MCQ_REPO}@{MCQ_REV} sources={sorted(MCQ_SOURCES)}", "math": f"{MATH_REPO}@{MATH_REV}"},
                "totals": dict(totals), "by_kind_language": {k: dict(v) for k, v in sorted(stats.items())}}
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=1, ensure_ascii=False) + "\n")
    print(json.dumps(manifest["totals"]))


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("select")
    s.add_argument("--out", required=True)
    s.add_argument("--per-language-mcq", type=int, default=800)
    s.add_argument("--per-language-math", type=int, default=500)
    s.add_argument("--seed", type=int, default=20261023)
    g = sub.add_parser("generate")
    g.add_argument("--out", required=True)
    g.add_argument("--model", required=True)
    g.add_argument("--template", required=True)
    g.add_argument("--max-tokens", type=int, default=512)
    a = ap.parse_args()
    select(a) if a.cmd == "select" else generate(a)


if __name__ == "__main__":
    main()
