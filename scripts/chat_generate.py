#!/usr/bin/env python3
"""Quick interactive-style generation from a checkpoint — eyeball the model's outputs.

Usage:
  python scripts/chat_generate.py --model /path/to/ckpt \
    --prompts "Vad är meningen med livet?" "Skriv en dikt om havet." [--max-new-tokens 400]
"""
from __future__ import annotations

import argparse
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--prompts", nargs="+", required=True)
    ap.add_argument("--max-new-tokens", type=int, default=400)
    ap.add_argument("--temperature", type=float, default=0.7)
    a = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(a.model, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        a.model, dtype=torch.bfloat16, device_map="auto", trust_remote_code=True
    ).eval()

    for p in a.prompts:
        inputs = tok.apply_chat_template(
            [{"role": "user", "content": p}], add_generation_prompt=True,
            return_tensors="pt", return_dict=True,
        ).to(model.device)
        with torch.no_grad():
            out = model.generate(
                **inputs, max_new_tokens=a.max_new_tokens,
                do_sample=a.temperature > 0, temperature=a.temperature, top_p=0.95,
                pad_token_id=tok.pad_token_id or tok.eos_token_id,
            )
        ans = tok.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        print("\n" + "=" * 70)
        print("PROMPT:", p)
        print("-" * 70)
        print(ans)
    print("=" * 70)


if __name__ == "__main__":
    main()
