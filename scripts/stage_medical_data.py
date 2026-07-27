#!/usr/bin/env python3
"""Stage birgermoell/oellm-eu-medical-posttrain-v1 onto LUMI: eval test jsonl + SFT/DPO parquet
+ GRPO train jsonl, for the separate medical-domain track."""
import json
import os

from datasets import Dataset
from huggingface_hub import hf_hub_download

R = "birgermoell/oellm-eu-medical-posttrain-v1"
B = "/scratch/project_465002530/users/bmoell"


def load(t):
    return [json.loads(l) for l in open(hf_hub_download(R, t, repo_type="dataset")) if l.strip()]


os.makedirs(f"{B}/qwen35-posttrain/data/medical", exist_ok=True)
os.makedirs(f"{B}/posttrain-data/medical-sft-parquet", exist_ok=True)
os.makedirs(f"{B}/posttrain-data/medical-dpo-parquet", exist_ok=True)

# eval test + grpo train (raw jsonl for the verifiable track)
for t, name in [("grpo/test.jsonl", "grpo_test.jsonl"), ("grpo/train.jsonl", "grpo_train.jsonl")]:
    rows = load(t)
    with open(f"{B}/qwen35-posttrain/data/medical/{name}", "w") as f:
        f.writelines(json.dumps(r, ensure_ascii=False) + "\n" for r in rows)
    print("staged", name, len(rows))

# SFT parquet: open-answer (clinical knowledge) + answer-only MCQ
sft = load("sft/swedish_open_answer_draft_train.jsonl") + load("sft/train.jsonl")
Dataset.from_list([{"messages": r["messages"], "lang": r.get("language", "sv")} for r in sft]) \
    .to_parquet(f"{B}/posttrain-data/medical-sft-parquet/train.parquet")
print("medical SFT parquet:", len(sft))

# DPO parquet: prompt / chosen / rejected as text
dpo = load("dpo/train.jsonl")


def txt(r, k):
    return str(r.get(k + "_text") or r.get(k))


Dataset.from_list([{"prompt": r["prompt"], "chosen": txt(r, "chosen"), "rejected": txt(r, "rejected")}
                   for r in dpo]).to_parquet(f"{B}/posttrain-data/medical-dpo-parquet/train.parquet")
print("medical DPO parquet:", len(dpo))
print("DONE")
