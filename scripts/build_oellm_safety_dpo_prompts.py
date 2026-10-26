#!/usr/bin/env python3
"""Prompts for the judge-labelled on-policy DPO pass (refusal calibration and grounding), Stage 4b.

EUROPA (v1 of the A4b/W1a line) regressed on grounding & safety (0.70 -> 0.58) and refusal calibration (0.44 -> 0.35)
after DPO + merge. This builds prompts with an expected behaviour; the policy's own answers are later scored by the
judge (scripts/judge_onpolicy_dpo.py) and turned into best-vs-worst pairs.

Behaviours and sources (open licenses only, pinned revisions):
  refuse   - harmful requests: nvidia/Aegis-AI-Content-Safety-Dataset-2.0 @ d86bb8be (CC-BY-4.0) prompts labelled
             unsafe (not only "Needs Caution"), LibrAI/do-not-answer @ 74e74f2e (Apache-2.0),
             bench-llm/or-bench @ e36d8b80 `or-bench-toxic` (CC-BY-4.0);
  comply   - legitimate requests that look sensitive: or-bench `or-bench-80k` minus its `or-bench-hard-1k` evaluation
             subset, plus Aegis prompts labelled safe;
  grounded - rajpurkar/squad_v2 @ 3ffb306f train (CC-BY-SA-4.0): a passage + question, answerable (reference answers)
             or unanswerable (the passage does not contain the answer), with varied "answer from the text" phrasings.
Every candidate is decontaminated against the EUROPA Eval and Alignment items (8-gram overlap >= 0.3 is dropped) and
deduplicated by normalized text. Output: prompts.jsonl (id, behaviour, source, license, language "en", prompt,
category, reference) and manifest.json.
"""
from __future__ import annotations

import argparse
import collections
import csv
import glob
import hashlib
import json
import random
import re
from pathlib import Path

import pyarrow.parquet as pq

TEMPLATES = [
    "{context}\n\nBased only on the text above, answer the question: {question}",
    "Read the passage and answer using only the information in it.\n\nPassage:\n{context}\n\nQuestion: {question}",
    "Context:\n{context}\n\nUsing the context, {question_lower}",
    "Here is a text:\n\n{context}\n\nAccording to this text, {question_lower}",
]


def norm(text: str) -> str:
    return re.sub(r"\s+", " ", text.lower()).strip()


def grams(text: str, n: int = 8) -> set[tuple[str, ...]]:
    words = re.findall(r"\w+", text.lower())
    return {tuple(words[i:i + n]) for i in range(max(0, len(words) - n + 1))}


def europa_grams(roots: list[str]) -> set[tuple[str, ...]]:
    out: set[tuple[str, ...]] = set()

    def walk(value):
        if isinstance(value, str):
            if len(value) >= 40:
                out.update(grams(value))
        elif isinstance(value, dict):
            for v in value.values():
                walk(v)
        elif isinstance(value, list):
            for v in value:
                walk(v)

    for root in roots:
        for path in glob.glob(f"{root}/**/*.jsonl", recursive=True):
            if "/results/" in path:
                continue
            for line in open(path, encoding="utf-8"):
                try:
                    walk(json.loads(line))
                except json.JSONDecodeError:
                    continue
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sources", type=Path, required=True)
    ap.add_argument("--europa", nargs="+", required=True, help="EUROPA repos (item jsonl files are read)")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--seed", type=int, default=20261026)
    a = ap.parse_args()
    rng = random.Random(a.seed)
    europa = europa_grams(a.europa)
    seen: set[str] = set()
    dropped = collections.Counter()
    picked: list[dict] = []

    def add(rows: list[dict], n: int, tag: str) -> None:
        rows = sorted(rows, key=lambda r: r["id"])
        rng.shuffle(rows)
        k = 0
        for r in rows:
            key = norm(r["prompt"])
            if key in seen:
                dropped[f"{tag}:duplicate"] += 1
                continue
            g = grams(r["prompt"])
            if g and len(g & europa) / len(g) >= 0.3:
                dropped[f"{tag}:europa_overlap"] += 1
                continue
            seen.add(key)
            picked.append(r)
            k += 1
            if k >= n:
                break

    s = a.sources
    aegis = json.load(open(s / "Aegis-AI-Content-Safety-Dataset-2.0/train.json", encoding="utf-8"))
    def aegis_ok(r):
        p = (r.get("prompt") or "").strip()
        return 15 <= len(p) <= 2000 and "REDACTED" not in p and not r.get("reconstruction_id_if_redacted")
    unsafe = [{"id": f"aegis-{r['id']}", "behaviour": "refuse", "source": "nvidia/Aegis-AI-Content-Safety-Dataset-2.0@d86bb8be",
               "license": "CC-BY-4.0", "prompt": r["prompt"].strip(), "category": r.get("violated_categories") or "", "reference": None}
              for r in aegis if aegis_ok(r) and r.get("prompt_label") == "unsafe"
              and (r.get("violated_categories") or "").strip() not in ("", "Needs Caution")]
    safe = [{"id": f"aegis-{r['id']}", "behaviour": "comply", "source": "nvidia/Aegis-AI-Content-Safety-Dataset-2.0@d86bb8be",
             "license": "CC-BY-4.0", "prompt": r["prompt"].strip(), "category": "safe", "reference": None}
            for r in aegis if aegis_ok(r) and r.get("prompt_label") == "safe"]
    dna = pq.read_table(glob.glob(str(s / "do-not-answer/data/*.parquet"))[0], columns=["id", "question", "risk_area"]).to_pylist()
    dna = [{"id": f"dna-{r['id']}", "behaviour": "refuse", "source": "LibrAI/do-not-answer@74e74f2e", "license": "Apache-2.0",
            "prompt": r["question"].strip(), "category": r["risk_area"], "reference": None} for r in dna]
    hard = {norm(r["prompt"]) for r in csv.DictReader(open(s / "or-bench/or-bench-hard-1k.csv", encoding="utf-8"))}
    orb = [{"id": "orb80k-" + hashlib.sha256(r["prompt"].encode()).hexdigest()[:12], "behaviour": "comply",
            "source": "bench-llm/or-bench@e36d8b80:or-bench-80k", "license": "CC-BY-4.0", "prompt": r["prompt"].strip(),
            "category": r["category"], "reference": None}
           for r in csv.DictReader(open(s / "or-bench/or-bench-80k.csv", encoding="utf-8")) if norm(r["prompt"]) not in hard]
    dropped["orbench:hard1k_excluded"] = 80359 - len(orb)
    toxic = [{"id": "orbtox-" + hashlib.sha256(r["prompt"].encode()).hexdigest()[:12], "behaviour": "refuse",
              "source": "bench-llm/or-bench@e36d8b80:or-bench-toxic", "license": "CC-BY-4.0", "prompt": r["prompt"].strip(),
              "category": r["category"], "reference": None}
             for r in csv.DictReader(open(s / "or-bench/or-bench-toxic.csv", encoding="utf-8"))]
    squad = pq.read_table(s / "squad_v2/squad_v2/train-00000-of-00001.parquet").to_pylist()
    rng.shuffle(squad)
    answerable, unanswerable, contexts = [], [], set()
    for r in squad:
        if len(r["context"]) > 2500 or r["context"] in contexts:
            continue
        q = r["question"].strip()
        t = TEMPLATES[int(hashlib.sha256(r["id"].encode()).hexdigest()[:4], 16) % len(TEMPLATES)]
        prompt = t.format(context=r["context"].strip(), question=q, question_lower=q[:1].lower() + q[1:])
        texts = list(dict.fromkeys(r["answers"]["text"]))
        row = {"id": f"squad2-{r['id']}", "behaviour": "grounded", "source": "rajpurkar/squad_v2@3ffb306f", "license": "CC-BY-SA-4.0",
               "prompt": prompt, "category": "answerable" if texts else "unanswerable", "reference": texts or None}
        (answerable if texts else unanswerable).append(row)
        contexts.add(r["context"])
        if len(answerable) >= 3000 and len(unanswerable) >= 3000:
            break
    add(unsafe, 800, "aegis_unsafe"); add(dna, 400, "dna"); add(toxic, 200, "orb_toxic")
    add(orb, 1000, "orb80k"); add(safe, 600, "aegis_safe")
    add(answerable, 600, "squad_answerable"); add(unanswerable, 600, "squad_unanswerable")
    for r in picked:
        r["language"] = "en"
    a.out.mkdir(parents=True, exist_ok=True)
    with open(a.out / "prompts.jsonl", "w", encoding="utf-8") as handle:
        for r in picked:
            handle.write(json.dumps(r, ensure_ascii=False) + "\n")
    manifest = {"kind": "oellm_safety_dpo_prompts_v1", "rows": len(picked),
                "by_behaviour": dict(collections.Counter(r["behaviour"] for r in picked)),
                "by_source": dict(collections.Counter(r["source"] for r in picked)),
                "grounded": dict(collections.Counter(r["category"] for r in picked if r["behaviour"] == "grounded")),
                "dropped": dict(dropped), "europa_ngrams": len(europa), "seed": a.seed}
    (a.out / "manifest.json").write_text(json.dumps(manifest, indent=1) + "\n")
    print(json.dumps(manifest, indent=1))


if __name__ == "__main__":
    main()
