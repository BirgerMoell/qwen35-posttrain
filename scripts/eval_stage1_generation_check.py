#!/usr/bin/env python3
"""Post-training generation check for a Stage-1 (direct-answer) checkpoint.

Greedy-decodes a fixed multilingual prompt set in default mode and records, per prompt,
whether the reply contains a think delimiter (must never happen in Stage 1) and whether
it terminated with <|im_end|> (vs running into the token limit). Writes all outputs to
JSON for human review; exits nonzero if any default-mode reply contains a think tag.
"""
from __future__ import annotations

import argparse
import json
import sys

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

PROMPTS = [
    ("en", "Explain in three sentences why the sky is blue."),
    ("en", "Write a haiku about autumn. Output only the haiku."),
    ("en", "List three European capitals as a JSON array of strings and nothing else."),
    ("sv", "Vad är skillnaden mellan väder och klimat? Svara kort."),
    ("fi", "Kerro lyhyesti, mikä on fotosynteesi."),
    ("de", "Schreibe eine kurze, höfliche E-Mail, in der du einen Termin absagst."),
    ("fr", "Donne trois conseils pour mieux dormir."),
    ("es", "¿Cuál es la capital de Australia? Responde en una frase."),
    ("it", "Riassumi in due frasi la trama di Cappuccetto Rosso."),
    ("nl", "Wat zijn de voordelen van fietsen naar je werk?"),
    ("pl", "Wyjaśnij krótko, czym jest inflacja."),
    ("cs", "Napiš krátkou báseň o moři."),
    ("ro", "Care sunt cele mai mari trei orașe din România?"),
    ("el", "Τι είναι η δημοκρατία; Απάντησε σε δύο προτάσεις."),
    ("uk", "Порадь, як вивчити нову мову швидше."),
    ("da", "Hvad er hovedstaden i Danmark, og hvad er den kendt for?"),
    ("pt", "Explique o que é energia renovável em poucas palavras."),
    ("hu", "Mi Magyarország fővárosa?"),
    ("ga", "Cad é príomhchathair na hÉireann?"),
    ("en", "Who are you and who made you?"),
    ("de", "Wer bist du?"),
    ("sv", "Vilken modell är du?"),
    ("en", "What is 17 * 23? Give only the number."),
    ("en", "A bat and a ball cost $1.10 in total. The bat costs $1.00 more than the ball. How much does the ball cost?"),
]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-new-tokens", type=int, default=768)
    args = ap.parse_args()
    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16, device_map="cuda").eval()
    end_id = tok.convert_tokens_to_ids("<|im_end|>")
    results = []
    for language, prompt in PROMPTS:
        inputs = tok.apply_chat_template([{"role": "user", "content": prompt}], add_generation_prompt=True,
                                         return_tensors="pt", return_dict=True).to(model.device)
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=args.max_new_tokens, do_sample=False,
                                 eos_token_id=end_id, pad_token_id=end_id)
        new = out[0][inputs["input_ids"].shape[1]:].tolist()
        raw = tok.decode(new, skip_special_tokens=False)
        results.append({
            "language": language, "prompt": prompt,
            "reply": tok.decode(new, skip_special_tokens=True).strip(),
            "think_tag": "<think>" in raw or "</think>" in raw,
            "terminated": end_id in new, "new_tokens": len(new),
        })
        print(f"[{language}] {prompt}\n{results[-1]['reply'][:600]}\n", flush=True)
    summary = {
        "model": args.model, "prompts": len(results),
        "think_tag_replies": sum(r["think_tag"] for r in results),
        "terminated_replies": sum(r["terminated"] for r in results),
    }
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump({"summary": summary, "results": results}, f, ensure_ascii=False, indent=2)
    print(json.dumps(summary, indent=2))
    return 1 if summary["think_tag_replies"] else 0


if __name__ == "__main__":
    sys.exit(main())
