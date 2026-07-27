#!/usr/bin/env python3
"""Quick A/B: does a repetition penalty kill the 2B's degeneration loops?

The published 2B's generation_config has no repetition control, and it degenerates on constrained
prompts (city list -> 'Kronoberg' x60, story -> paragraph loop). Regenerate the known loopers under
(A) current defaults vs (B) repetition_penalty=1.15 + no_repeat_ngram_size=4, and eyeball.
"""
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MID = "/scratch/project_465002530/users/bmoell/models/Qwen3.5-2B-EU-published"
PROMPTS = [
    "Lista fem svenska städer i bokstavsordning, en per rad, utan någon annan text.",
    "Hitta på en kort berättelse (cirka 100 ord) om en räv som hittar en försvunnen vante i skogen.",
    "En vän säger att hen känner sig stressad inför ett jobbintervju imorgon. Vad kan du säga för att hjälpa och lugna?",
    "Förklara skillnaden mellan \"de\" och \"dem\" och ge ett exempel på korrekt användning av vardera.",
]
CONFIGS = {
    "A_default": dict(do_sample=False),
    "B_reppenalty": dict(do_sample=False, repetition_penalty=1.15, no_repeat_ngram_size=4),
}

tok = AutoTokenizer.from_pretrained(MID)
model = AutoModelForCausalLM.from_pretrained(MID, dtype=torch.bfloat16, device_map="auto").eval()

for p in PROMPTS:
    print("\n" + "=" * 90 + f"\nPROMPT: {p}")
    enc = tok.apply_chat_template([{"role": "user", "content": p}], add_generation_prompt=True,
                                  return_tensors="pt", return_dict=True,
                                  enable_thinking=False).to(model.device)
    for name, kw in CONFIGS.items():
        with torch.no_grad():
            out = model.generate(**enc, max_new_tokens=220,
                                 pad_token_id=tok.pad_token_id or tok.eos_token_id, **kw)
        txt = tok.decode(out[0, enc["input_ids"].shape[1]:], skip_special_tokens=True)
        print(f"\n--- {name} ({len(txt.split())} words) ---\n{txt}")
