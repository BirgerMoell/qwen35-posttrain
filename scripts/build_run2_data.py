#!/usr/bin/env python3
"""Build the run #2 data: reasoning-SFT mix + multilingual DPO mix, as Parquet.

Run #1 used EuroBlocks synthetic SFT + English-only DPO. Run #2 adds the two biggest gaps,
using distilled data already on LUMI (no generation needed):

  reasoning-SFT : Nemotron-v2 math (EN reasoning) + Finnish DeepSeek math (EU reasoning),
                  both `{input: [msgs], output: str}` -> TRL `{messages}` (input + assistant turn).
  multilingual-DPO : per-language Dolci/HelpSteer/UltraFeedback DPO ({prompt,chosen,rejected})
                  concatenated across fin/swe/dan/nor/isl + the multilingual set.

Output Parquet (memory-mapped -> no per-rank JSON OOM):
  .../posttrain-data/qwen35-9b-reasoning-sft-parquet/train.parquet
  .../posttrain-data/qwen35-9b-multiling-dpo-parquet/train.parquet

Run inside the container on a login node (binds /scratch project dirs).
"""
from __future__ import annotations

import os
from datasets import load_dataset, concatenate_datasets, Dataset

SRC = "/scratch/project_462000963/datasets/posttraining_data"
OUT = "/scratch/project_465002530/users/bmoell/posttrain-data"

# ---- reasoning SFT: {input, output} -> {messages} ----
REASONING_FILES = [
    f"{SRC}/Nemotron-Post-Training-Dataset-v2/math.jsonl",                 # EN reasoning
    f"{SRC}/Llama-Nemotron-Post-Training-Dataset/SFT-math-sample-100k.jsonl",  # incl. Finnish DeepSeek
]

def to_messages(ex):
    msgs = list(ex.get("input") or [])
    out = ex.get("output")
    if out is not None:
        msgs = msgs + [{"role": "assistant", "content": out}]
    return {"messages": msgs}

def build_reasoning():
    parts = []
    for f in REASONING_FILES:
        if not os.path.exists(f):
            print(f"  skip (missing): {f}"); continue
        ds = load_dataset("json", data_files={"train": f}, split="train")
        ds = ds.map(to_messages, remove_columns=[c for c in ds.column_names if c != "messages"])
        ds = ds.filter(lambda e: len(e["messages"]) >= 2)
        print(f"  {f.split('/')[-1]}: {len(ds)} examples")
        parts.append(ds)
    combined = concatenate_datasets(parts)
    dst = f"{OUT}/qwen35-9b-reasoning-sft-parquet"
    os.makedirs(dst, exist_ok=True)
    combined.select_columns(["messages"]).to_parquet(f"{dst}/train.parquet")
    print(f"reasoning-SFT: {len(combined)} -> {dst}/train.parquet")

# ---- multilingual DPO: concat per-language {prompt,chosen,rejected} ----
DPO_LANG_DIRS = ["fin", "swe", "dan", "nor", "isl", "multiling"]

def build_dpo():
    import glob
    parts = []
    for lang in DPO_LANG_DIRS:
        for f in glob.glob(f"{SRC}/DPOTrainer_format/{lang}/**/train.jsonl", recursive=True):
            try:
                ds = load_dataset("json", data_files={"train": f}, split="train")
                cols = set(ds.column_names)
                if {"prompt", "chosen", "rejected"} <= cols:
                    ds = ds.select_columns(["prompt", "chosen", "rejected"])
                    parts.append(ds)
                    print(f"  {lang}/{f.split('/')[-2]}: {len(ds)}")
            except Exception as e:
                print(f"  skip {f}: {e}")
    combined = concatenate_datasets(parts)
    dst = f"{OUT}/qwen35-9b-multiling-dpo-parquet"
    os.makedirs(dst, exist_ok=True)
    combined.to_parquet(f"{dst}/train.parquet")
    print(f"multilingual-DPO: {len(combined)} -> {dst}/train.parquet")

if __name__ == "__main__":
    import sys
    which = sys.argv[1] if len(sys.argv) > 1 else "all"
    if which in ("all", "reasoning"):
        build_reasoning()
    if which in ("all", "dpo"):
        build_dpo()
