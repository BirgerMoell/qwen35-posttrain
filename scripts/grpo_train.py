#!/usr/bin/env python3
"""GRPO / RLVR entrypoint with a verifiable MCQ reward — for the EU exam dataset.

Reward = `mcq_letter_exact`: the model must answer with a single option letter; reward 1.0 if
it matches the gold `answer`, else 0.0. No reward model, no preference pairs — pure verifiable
RL, which sidesteps the translationese problem of translated preference data and optimizes
EU-language *correctness* directly.

Rollouts: TRL GRPOTrainer generates with the model itself by default (slow but no extra
dependency) or vLLM if `use_vllm: true` in the config (fast — once vLLM-ROCm serves qwen3_5).

Data: data/exam_mcq/oellm-eu-exam-mcq-v1/grpo/train.jsonl  ({prompt, answer, language, ...}).
"""
from __future__ import annotations

import re

import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

# --- TRL bug workaround (must run BEFORE importing GRPOTrainer) ---
# trl.import_utils.is_vllm_ascend_available() returns the (exists, version) TUPLE from
# _is_package_available instead of just the bool. A non-empty tuple is truthy, so TRL's
# `if is_vllm_ascend_available():` always fires and imports vllm_ascend (a Huawei-NPU package,
# absent on ROCm) -> ImportError at GRPOTrainer import. Force it False first.
import trl.import_utils as _trl_iu  # noqa: E402
_trl_iu.is_vllm_ascend_available = lambda *a, **k: False

from trl import GRPOConfig, GRPOTrainer, ModelConfig, ScriptArguments, TrlParser, get_peft_config  # noqa: E402

LETTER = re.compile(r"\b([A-Da-d])\b")


def _text(completion):
    # GRPO passes completions as message lists (conversational) or strings.
    if isinstance(completion, list):
        return completion[-1].get("content", "") if completion else ""
    return completion or ""


def mcq_letter_exact(completions, answer, **kwargs):
    """1.0 if the model's chosen letter matches the gold letter, else 0.0."""
    rewards = []
    for comp, gold in zip(completions, answer):
        txt = _text(comp).strip()
        # prefer the first standalone A–D; fall back to first such char anywhere
        m = LETTER.search(txt) or re.search(r"[A-Da-d]", txt)
        pred = m.group(1).upper() if m and m.lastindex else (m.group(0).upper() if m else None)
        rewards.append(1.0 if pred == str(gold).strip().upper() else 0.0)
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
    # keep `prompt` (GRPO generates from it) + `answer` (reward func reads it via kwargs)
    keep = [c for c in dataset.column_names if c in ("prompt", "answer", "language")]
    dataset = dataset.select_columns(keep)

    trainer = GRPOTrainer(
        model=model,
        reward_funcs=[mcq_letter_exact],
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
