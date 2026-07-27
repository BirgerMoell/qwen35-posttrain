#!/usr/bin/env python3
"""Self-contained function-calling eval (BFCL-style) — no vLLM needed.

BFCL's official harness serves models via vLLM, which doesn't yet support Qwen3.5 (the same block
that stopped EuroEval). So we grade the way we do the Swedish rubric: load with transformers
(device_map="auto"), generate no-think, parse the first <tool_call>, and score AST-style.

The eval set is a HELD-OUT slice of Glaive (rows past the training cap) reusing the exact training
normalizer, so it matches the format the model was taught. We report, over callable prompts:
  format_valid  - emitted a well-formed <tool_call> with valid JSON
  name_acc      - correct function name
  args_exact    - arguments dict matches gold exactly (AST simple/parallel equivalent)
and over refusal prompts (gold = no call): abstain_acc - correctly did NOT call (relevance).

Usage:
  python scripts/eval_tool_calling.py --model <path-or-id> [--n 300] [--out results/tool_eval.json]
"""
from __future__ import annotations

import argparse
import itertools
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))  # so build_tooluse_sft resolves

import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

# reuse the exact training-time normalizer so eval format == train format
from build_tooluse_sft import (  # noqa: E402  (scripts/ on sys.path)
    _tools_system, _glaive_call, _balanced_json_objects, _all_tool_calls,
)


def held_out_glaive(n, skip):
    """Build a held-out eval set from Glaive rows AFTER the training cap (skip=50000)."""
    ds = load_dataset("glaiveai/glaive-function-calling-v2", split="train")
    import re
    marker = re.compile(r"(USER:|ASSISTANT:|FUNCTION RESPONSE:)")
    out = []
    for row in itertools.islice(ds, skip, skip + n * 4):
        if len(out) >= n:
            break
        sys_txt, chat = row.get("system", "") or "", row.get("chat", "") or ""
        tools = _balanced_json_objects(sys_txt.split("required -", 1)[-1])
        if not tools or "USER:" not in chat:
            continue
        parts = marker.split(chat)
        it = iter(parts[1:])
        first_user, gold = None, None
        for tag, body in zip(it, it):
            body = body.replace("<|endoftext|>", "").strip()
            if tag == "USER:" and first_user is None:
                first_user = body
            elif tag == "ASSISTANT:" and first_user is not None:
                if "<functioncall>" in body:
                    c = _glaive_call(body)
                    gold = [c] if c and c.get("name") else None
                else:
                    gold = []          # refusal / no-call turn (relevance case)
                break
        if first_user is None or gold is None:
            continue
        out.append({"tools": tools, "user": first_user, "gold": gold})
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--skip", type=int, default=50000, help="past the 50k training cap = held out")
    ap.add_argument("--out", default="results/tool_eval.json")
    args = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16,
                                                 device_map="auto")
    model.eval()

    evalset = held_out_glaive(args.n, args.skip)
    callable_n = sum(1 for e in evalset if e["gold"])
    refusal_n = len(evalset) - callable_n
    print(f"eval set: {len(evalset)}  (callable={callable_n}, refusal={refusal_n})")

    fmt_ok = name_ok = args_ok = abstain_ok = 0
    for e in evalset:
        msgs = [{"role": "system", "content": _tools_system(e["tools"])},
                {"role": "user", "content": e["user"]}]
        # return_dict=True -> a BatchEncoding we can splat into generate(); passing it positionally
        # would make generate() call .shape on the dict (KeyError 'shape').
        enc = tok.apply_chat_template(msgs, add_generation_prompt=True, return_tensors="pt",
                                      return_dict=True, enable_thinking=False).to(model.device)
        in_len = enc["input_ids"].shape[1]
        with torch.no_grad():
            out = model.generate(**enc, max_new_tokens=256, do_sample=False,
                                 pad_token_id=tok.pad_token_id or tok.eos_token_id)
        text = tok.decode(out[0, in_len:], skip_special_tokens=True)
        calls = _all_tool_calls(text)

        if not e["gold"]:                       # refusal case: correct iff it did NOT call
            if not calls:
                abstain_ok += 1
            continue
        gold = e["gold"][0]
        if calls:
            fmt_ok += 1
            pred = calls[0]
            if str(pred.get("name")) == str(gold.get("name")):
                name_ok += 1
                if pred.get("arguments", {}) == gold.get("arguments", {}):
                    args_ok += 1

    res = {
        "model": args.model, "n": len(evalset), "callable_n": callable_n, "refusal_n": refusal_n,
        "format_valid": round(fmt_ok / callable_n, 3) if callable_n else None,
        "name_acc": round(name_ok / callable_n, 3) if callable_n else None,
        "args_exact": round(args_ok / callable_n, 3) if callable_n else None,
        "abstain_acc": round(abstain_ok / refusal_n, 3) if refusal_n else None,
    }
    print(json.dumps(res, indent=2))
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(res, indent=2))
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
