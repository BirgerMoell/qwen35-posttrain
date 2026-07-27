#!/usr/bin/env python3
"""GRPO / RLVR entrypoint with a verifiable TOOL-CALL reward — for the tool-use fold-in.

Reward = `tool_call_exact`: a function call is structurally checkable, so we grade it directly
(no reward model, beta=0), analogous to `mcq_letter_exact`. Partial, shaped credit:
  0.25  emitted a well-formed <tool_call> with valid JSON
  +0.35 correct function name
  +0.40 arguments match the gold (exact key/value match)
Gold comes from the `answer` field (json list of {name, arguments}); we grade against the first
call (the GRPO jsonl is built from single-call xLAM rows).

Data: posttrain-data/eu-tooluse-parquet/grpo/train.jsonl  ({prompt:[msgs], answer, language}).
"""
from __future__ import annotations

import json
import re

import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

# --- same TRL vllm-ascend workaround as grpo_train.py (must precede GRPOTrainer import) ---
import trl.import_utils as _trl_iu  # noqa: E402
_trl_iu.is_vllm_ascend_available = lambda *a, **k: False

from trl import GRPOConfig, GRPOTrainer, ModelConfig, ScriptArguments, TrlParser, get_peft_config  # noqa: E402

_CALL = re.compile(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", re.DOTALL)


def _text(completion):
    if isinstance(completion, list):
        return completion[-1].get("content", "") if completion else ""
    return completion or ""


def _first_call(txt: str):
    m = _CALL.search(txt)
    if not m:
        return None
    try:
        return json.loads(m.group(1))
    except Exception:
        return None


def tool_call_exact(completions, answer, **kwargs):
    """Shaped, verifiable reward with an ABSTENTION component (fixes the reward-hack where a
    call-only reward collapsed relevance to 0). Gold is a JSON list of {name, arguments}:
      - empty list  -> the query needs NO tool: reward 1.0 for NOT calling, 0.0 for a spurious call.
      - one call     -> 0.25 well-formed call + 0.35 correct name + 0.40 exact args."""
    rewards = []
    for comp, gold_json in zip(completions, answer):
        pred = _first_call(_text(comp))
        try:
            gold_list = json.loads(gold_json)
        except Exception:
            gold_list = None
        # Refusal / relevance case: gold is an empty list -> the model should NOT call.
        if isinstance(gold_list, list) and len(gold_list) == 0:
            rewards.append(1.0 if pred is None else 0.0)
            continue
        gold = gold_list[0] if gold_list else None
        r = 0.0
        if pred is not None and isinstance(pred, dict) and "name" in pred:
            r += 0.25
            if gold is not None:
                if str(pred.get("name")) == str(gold.get("name")):
                    r += 0.35
                if pred.get("arguments", {}) == gold.get("arguments", {}):
                    r += 0.40
        rewards.append(r)
    return rewards


def main(script_args, training_args, model_args):
    dtype_str = getattr(model_args, "dtype", None) or getattr(model_args, "torch_dtype", None)
    dtype = dtype_str if dtype_str in (None, "auto") else getattr(torch, dtype_str)

    model = AutoModelForCausalLM.from_pretrained(
        model_args.model_name_or_path, dtype=dtype,
        attn_implementation=model_args.attn_implementation,
        trust_remote_code=model_args.trust_remote_code, low_cpu_mem_usage=True,
    )
    names = getattr(model, "_no_split_modules", None)
    if names:
        present = {type(m).__name__ for m in model.modules()}
        model._no_split_modules = [c for c in names if c in present]

    tokenizer = AutoTokenizer.from_pretrained(
        model_args.model_name_or_path, trust_remote_code=model_args.trust_remote_code)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    dataset = load_dataset("json", data_files={"train": script_args.dataset_name}, split="train")
    keep = [c for c in dataset.column_names if c in ("prompt", "answer", "language")]
    dataset = dataset.select_columns(keep)

    trainer = GRPOTrainer(
        model=model,
        reward_funcs=[tool_call_exact],
        args=training_args,
        train_dataset=dataset,
        processing_class=tokenizer,
        peft_config=get_peft_config(model_args),
    )
    trainer.train()
    trainer.accelerator.print("✅ Training completed.")
    trainer.save_model(training_args.output_dir)
    trainer.accelerator.print(f"💾 Model saved to {training_args.output_dir}.")


if __name__ == "__main__":
    parser = TrlParser((ScriptArguments, GRPOConfig, ModelConfig))
    script_args, training_args, model_args, _ = parser.parse_args_and_config(
        return_remaining_strings=True)
    main(script_args, training_args, model_args)
