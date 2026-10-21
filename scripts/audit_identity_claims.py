#!/usr/bin/env python3
"""Count model-identity claims in the exact rows selected for the Stage-1 mix.

Replays the mix builder's selection (same units, same hashing, same caps) and reports,
per source: assistant turns that name another model/organisation, first-person English
self-identification claims ("I am ChatGPT", "I was developed by OpenAI"), and rows whose
user prompt asks who the model is, with the organisation named in the answer.
"""
from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
import re
from pathlib import Path

from build_oellm_stage1_mix import iter_unit, plan_units

ORGS = {
    "openai": r"\bOpen ?AI\b", "chatgpt": r"\bChat ?GPT\b", "gpt": r"\bGPT-?[345](?:\.\d)?o?\b",
    "qwen": r"\bQwen\b|\bTongyi\b", "alibaba": r"\bAlibaba\b", "claude": r"\bClaude\b",
    "anthropic": r"\bAnthropic\b", "llama_meta": r"\bLLa?MA\b|\bMeta AI\b", "gemini": r"\bGemini\b|\bBard\b",
    "mistral": r"\bMistral\b", "deepseek": r"\bDeepSeek\b", "microsoft": r"\bMicrosoft\b|\bPhi-\d",
    "nvidia": r"\bNVIDIA\b|\bNemotron\b", "smollm_hf": r"\bSmolLM\b|\bHugging ?Face\b",
    "ai2": r"\bOLMo\b|\bT[uü]lu\b|\bAllen Institute\b|\bAI2\b", "openeurollm": r"\bOpen ?Euro ?LLM\b",
}
ORG_RE = {k: re.compile(v, re.I) for k, v in ORGS.items()}
SELF_CLAIM = re.compile(
    r"\b(?:I am|I'm|I was|my name is|as)\b[^.\n]{0,60}?\b(?:"
    r"ChatGPT|GPT-?[345]|Qwen|Claude|Llama|Gemini|Bard|Mistral|DeepSeek|SmolLM|OLMo|T[uü]lu|Nemotron"
    r"|(?:developed|created|trained|built|made) by (?:OpenAI|Alibaba|Anthropic|Meta|Google|Microsoft|Mistral|DeepSeek|NVIDIA|Hugging ?Face|AI2|the Allen))",
    re.I)
# Strict: the assistant itself claims to be ChatGPT / made by OpenAI (what the model then imitates).
OPENAI_SELF = re.compile(
    r"\b(?:as an?|I am an?|I'm an?|I was)\b[^.\n]{0,50}?\b(?:developed|created|trained|made|built|designed) by OpenAI\b"
    r"|\bI(?:'m| am) (?:ChatGPT|GPT-?[345])\b|\bas ChatGPT\b", re.I)
ASKS_IDENTITY = re.compile(
    r"\b(?:who (?:are|made|created|built|trained|developed) you|what(?:'s| is) your name|what (?:model|AI|LLM|language model) are you"
    r"|which (?:model|AI|LLM) are you|are you (?:ChatGPT|GPT|an AI|a bot|human))\b"
    r"|vem är du|wer bist du|qui es-tu|qui êtes-vous|quién eres|chi sei|kim jesteś|kdo jsi|wie ben je", re.I)


def scan(unit: dict, config: dict) -> dict:
    counts: dict[str, Counter] = {}
    examples: dict[str, list] = {}
    for split, source, language, row_id, messages in iter_unit(unit, config):
        if split != "train":
            continue
        c = counts.setdefault(source, Counter())
        c["rows"] += 1
        answer = "\n".join(m["content"] for m in messages if m["role"] == "assistant" and isinstance(m.get("content"), str))
        prompt = "\n".join(m["content"] for m in messages if m["role"] == "user" and isinstance(m.get("content"), str))
        named = [k for k, rx in ORG_RE.items() if rx.search(answer)]
        for k in named:
            c[f"mentions_{k}"] += 1
        if named:
            c["rows_naming_any_model_or_org"] += 1
        strict = OPENAI_SELF.search(answer)
        if strict:
            c["openai_self_claim"] += 1
            ex = examples.setdefault(f"{source}|openai_self_claim", [])
            if len(ex) < 4:
                ex.append({"id": row_id, "lang": language, "prompt": prompt[:160], "match": answer[max(0, strict.start() - 40):strict.end() + 80]})
        claim = SELF_CLAIM.search(answer)
        if claim:
            c["self_identity_claim_en"] += 1
            ex = examples.setdefault(f"{source}|self_claim", [])
            if len(ex) < 6:
                ex.append({"id": row_id, "lang": language, "prompt": prompt[:200], "match": answer[max(0, claim.start() - 80):claim.end() + 80]})
        if ASKS_IDENTITY.search(prompt):
            c["prompt_asks_identity"] += 1
            for k in named:
                c[f"identity_answer_names_{k}"] += 1
            ex = examples.setdefault(f"{source}|asks_identity", [])
            if len(ex) < 6:
                ex.append({"id": row_id, "lang": language, "prompt": prompt[:200], "answer": answer[:300]})
    return {"counts": {k: dict(v) for k, v in counts.items()}, "examples": examples}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args()
    manifest = json.loads((a.root / "data/stage1-mix-v1/manifest.json").read_text())
    config = dict(manifest["config"])
    args = argparse.Namespace(
        excellent_dir=a.root / "data/excellent-stage1-full-v1",
        dolci_dir=a.root / "source-datasets/dolci-instruct-translated",
        aya_dir=a.root / "aya-dataset", identity_dir=a.root / "data/model-identity-v1")
    units = plan_units(args)
    total: dict[str, Counter] = {}
    examples: dict[str, list] = {}
    with ProcessPoolExecutor(max_workers=a.workers) as pool:
        for future in as_completed([pool.submit(scan, u, config) for u in units]):
            r = future.result()
            for source, c in r["counts"].items():
                total.setdefault(source, Counter()).update(c)
            for key, ex in r["examples"].items():
                examples.setdefault(key, []).extend(ex[: max(0, 6 - len(examples.get(key, [])))])
    out = {"counts": {k: dict(v) for k, v in sorted(total.items())}, "examples": examples}
    a.out.write_text(json.dumps(out, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    grand = Counter()
    for c in total.values():
        grand.update(c)
    print(json.dumps({k: v for k, v in grand.items() if not k.startswith("identity_answer") or v}, indent=1))


if __name__ == "__main__":
    main()
