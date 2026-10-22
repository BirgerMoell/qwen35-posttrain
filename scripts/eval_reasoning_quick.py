#!/usr/bin/env python3
"""Quick reasoning eval in default vs thinking mode: GSM8K, MGSM (en/de/es/fr), MATH-500 subset and
oellm-math-rlvr test prompts in 12 EU languages.

`prepare` writes a fixed JSONL eval set. `run --rank R --world W` greedy-decodes its shard in one mode
on one GPU (HF transformers, left-padded batches). `score` merges shards and reports accuracy,
unfinished-trace rate (thinking mode: no `</think>` before the token limit = loop/runaway), and
lengths, per subset and mode. Answers are read only after `</think>`; the last `\\boxed{}` wins,
else the last number. MATH-500 uses normalized-string / numeric comparison (approximate).
"""
from __future__ import annotations

import argparse
import json
import random
import re
from collections import defaultdict
from pathlib import Path

INSTRUCTION = "\n\nPut your final answer in \\boxed{}."
EU = ["sv", "fi", "da", "pl", "cs", "nl", "it", "pt", "ro", "el", "hu", "bg"]


def prepare(out: Path) -> None:
    import pyarrow.parquet as pq
    from huggingface_hub import hf_hub_download
    rng = random.Random(1234)
    rows = []
    t = pq.read_table(hf_hub_download("openai/gsm8k", "main/test-00000-of-00001.parquet", repo_type="dataset", revision="740312ad")).to_pylist()
    for i, r in enumerate(rng.sample(t, 150)):
        rows.append({"id": f"gsm8k-{i}", "subset": "gsm8k", "language": "en", "prompt": r["question"] + INSTRUCTION,
                     "gold": r["answer"].split("####")[-1].strip().replace(",", "")})
    for lang in ["en", "de", "es", "fr"]:
        t = pq.read_table(hf_hub_download("juletxara/mgsm", f"{lang}/test-00000-of-00001.parquet", repo_type="dataset", revision="b2f13d42")).to_pylist()
        for i, r in enumerate(rng.sample(t, 50)):
            rows.append({"id": f"mgsm-{lang}-{i}", "subset": "mgsm", "language": lang, "prompt": r["question"] + INSTRUCTION,
                         "gold": str(r["answer_number"]).rstrip("0").rstrip(".") if "." in str(r["answer_number"]) else str(r["answer_number"])})
    lines = open(hf_hub_download("HuggingFaceH4/MATH-500", "test.jsonl", repo_type="dataset", revision="6e4ed1a2")).read().splitlines()
    for i, r in enumerate(rng.sample([json.loads(l) for l in lines], 100)):
        rows.append({"id": f"math500-{i}", "subset": "math500", "language": "en", "prompt": r["problem"] + INSTRUCTION, "gold": r["answer"]})
    t = pq.read_table(hf_hub_download("birgermoell/oellm-math-rlvr", "data/test-00000-of-00001.parquet", repo_type="dataset", revision="0ffc9d6d")).to_pylist()
    for lang in EU:
        pool = [r for r in t if r["language"] == lang and r["difficulty"] >= 3]
        for i, r in enumerate(rng.sample(pool, 10)):
            rows.append({"id": f"oellm-{lang}-{i}", "subset": "oellm_math_eu", "language": lang,
                         "prompt": r["messages"][-1]["content"], "gold": r["ground_truth"][0]})
    out.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))
    print(f"wrote {len(rows)} prompts to {out}")


def run(a) -> None:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    rows = [json.loads(l) for l in a.data.read_text().splitlines()][a.rank::a.world]
    tok = AutoTokenizer.from_pretrained(a.model)
    tok.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(a.model, dtype=torch.bfloat16, attn_implementation="flash_attention_2",
                                                 device_map="cuda").eval()
    end = tok.convert_tokens_to_ids("<|im_end|>")
    thinking = a.mode == "thinking"
    max_new = a.max_new_tokens or (6144 if thinking else 1024)
    rows.sort(key=lambda r: len(r["prompt"]))
    out = []
    for b in range(0, len(rows), a.batch):
        batch = rows[b:b + a.batch]
        texts = [tok.apply_chat_template([{"role": "user", "content": r["prompt"]}], tokenize=False,
                                         add_generation_prompt=True, enable_thinking=thinking) for r in batch]
        enc = tok(texts, return_tensors="pt", padding=True, add_special_tokens=False).to("cuda")
        with torch.no_grad():
            gen = model.generate(**enc, max_new_tokens=max_new, do_sample=False, eos_token_id=end, pad_token_id=tok.pad_token_id)
        for r, seq in zip(batch, gen[:, enc["input_ids"].shape[1]:].tolist()):
            n = seq.index(end) + 1 if end in seq else len(seq)
            raw = tok.decode(seq[:n], skip_special_tokens=False)
            out.append({**r, "mode": a.mode, "raw": raw, "new_tokens": n, "terminated": end in seq})
        print(f"rank {a.rank} {a.mode}: {len(out)}/{len(rows)}", flush=True)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in out))


def last_boxed(text: str) -> str | None:
    i = text.rfind("\\boxed{")
    if i < 0:
        return None
    j, depth = i + len("\\boxed{"), 1
    while j < len(text) and depth:
        depth += {"{": 1, "}": -1}.get(text[j], 0)
        j += 1
    return text[i + len("\\boxed{"):j - 1] if depth == 0 else None


def norm(s: str) -> str:
    s = s.strip().strip("$").replace("\\left", "").replace("\\right", "").replace("\\dfrac", "\\frac").replace("\\tfrac", "\\frac")
    s = re.sub(r"\\text\{([^}]*)\}", r"\1", s).replace("\\!", "").replace("\\,", "").replace(" ", "").rstrip(".")
    s = re.sub(r"^\\?\(|\\?\)$", "", s) if s.count("(") != s.count(")") else s
    return s.replace("{,}", "").replace(",", "") if re.fullmatch(r"-?[\d,{}]+(\.\d+)?", s) else s


def to_num(s: str):
    try:
        m = re.fullmatch(r"(-?\d+)/(\d+)|\\frac\{(-?\d+)\}\{(\d+)\}", s)
        if m:
            p, q = (m.group(1), m.group(2)) if m.group(1) else (m.group(3), m.group(4))
            return int(p) / int(q)
        return float(s)
    except (ValueError, ZeroDivisionError):
        return None


def correct(r: dict) -> bool:
    text = r["raw"].replace("<|im_end|>", "")
    if r["mode"] == "thinking":
        if "</think>" not in text:
            return False
        text = text.split("</think>", 1)[1]
    pred = last_boxed(text)
    if pred is None:
        nums = re.findall(r"-?\d[\d,]*(?:\.\d+)?", text)
        pred = nums[-1] if nums else ""
    p, g = norm(pred), norm(r["gold"])
    if p == g:
        return True
    pn, gn = to_num(p), to_num(g)
    return pn is not None and gn is not None and abs(pn - gn) < 1e-6 * max(1, abs(gn))


def score(a) -> None:
    rows = [json.loads(l) for p in sorted(a.results.glob("*.jsonl")) for l in p.read_text().splitlines()]
    agg = defaultdict(lambda: defaultdict(list))
    for r in rows:
        ok = correct(r)
        unfinished = r["mode"] == "thinking" and "</think>" not in r["raw"]
        for key in (r["subset"], "ALL") + ((f"lang:{r['language']}",) if r["subset"] in ("mgsm", "oellm_math_eu") else ()):
            agg[key][r["mode"]].append((ok, unfinished, r["new_tokens"], r["terminated"]))
    report = {}
    print(f"{'subset':<16}{'mode':<10}{'n':>5}{'acc':>8}{'unfinished':>12}{'not_term':>10}{'mean_tok':>10}")
    for key in sorted(agg, key=lambda k: (k.startswith("lang:"), k)):
        for mode in ("default", "thinking"):
            v = agg[key].get(mode, [])
            if not v:
                continue
            n = len(v)
            m = {"n": n, "accuracy": sum(x[0] for x in v) / n, "unfinished_rate": sum(x[1] for x in v) / n,
                 "not_terminated_rate": sum(not x[3] for x in v) / n, "mean_new_tokens": sum(x[2] for x in v) / n}
            report.setdefault(key, {})[mode] = m
            print(f"{key:<16}{mode:<10}{n:>5}{m['accuracy']:>8.3f}{m['unfinished_rate']:>12.3f}{m['not_terminated_rate']:>10.3f}{m['mean_new_tokens']:>10.0f}")
    a.out.write_text(json.dumps(report, indent=2) + "\n")


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("prepare"); p.add_argument("--out", type=Path, required=True)
    r = sub.add_parser("run")
    for k, t in (("--model", str), ("--data", Path), ("--out", Path), ("--mode", str)):
        r.add_argument(k, type=t, required=True)
    r.add_argument("--rank", type=int, default=0); r.add_argument("--world", type=int, default=1)
    r.add_argument("--batch", type=int, default=16); r.add_argument("--max-new-tokens", type=int, default=0)
    s = sub.add_parser("score"); s.add_argument("--results", type=Path, required=True); s.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    {"prepare": lambda: prepare(a.out), "run": lambda: run(a), "score": lambda: score(a)}[a.cmd]()


if __name__ == "__main__":
    main()
