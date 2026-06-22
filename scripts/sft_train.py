#!/usr/bin/env python3
"""Text-only SFT entrypoint (TRL) — loads AutoModelForCausalLM + explicit tokenizer.

Why this exists: the stock `trl.scripts.sft` loads 2026 multimodal-origin models
(Gemma 4, Qwen 3.5) via AutoModelForImageTextToText. That loader:
  - pulls in the vision/audio towers we don't need for text SFT, and
  - ignores FSDP `cpu_ram_efficient_loading`, so every rank loads a full model copy
    to CPU (8 x 19 GB ~= 152 GB on one node) and OOMs at 9B+.

This wrapper loads the text-only AutoModelForCausalLM with low_cpu_mem_usage, which
respects the FSDP rank-0-only loading path, and passes an explicit tokenizer. Reuses
TRL's config dataclasses + dataset mixture loader, so the same YAML configs work.
"""
from __future__ import annotations

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import (
    DatasetMixtureConfig,
    ModelConfig,
    ScriptArguments,
    SFTConfig,
    SFTTrainer,
    TrlParser,
    get_dataset,
    get_peft_config,
)


def _prune_no_split_modules(m):
    """Drop phantom vision/audio block classes that don't exist in the text-only model
    (otherwise FSDP TRANSFORMER_BASED_WRAP fails to find them)."""
    names = getattr(m, "_no_split_modules", None)
    if names:
        present = {type(sub).__name__ for sub in m.modules()}
        m._no_split_modules = [c for c in names if c in present]


def main(script_args, training_args, model_args, dataset_args):
    dtype_str = getattr(model_args, "dtype", None) or getattr(model_args, "torch_dtype", None)
    dtype = dtype_str if dtype_str in (None, "auto") else getattr(torch, dtype_str)

    model = AutoModelForCausalLM.from_pretrained(
        model_args.model_name_or_path,
        revision=model_args.model_revision,
        attn_implementation=model_args.attn_implementation,
        dtype=dtype,
        trust_remote_code=model_args.trust_remote_code,
        low_cpu_mem_usage=True,
    )
    _prune_no_split_modules(model)

    tokenizer = AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        trust_remote_code=model_args.trust_remote_code,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    dataset = get_dataset(dataset_args)

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset[script_args.dataset_train_split],
        eval_dataset=(
            dataset[script_args.dataset_test_split]
            if training_args.eval_strategy != "no"
            else None
        ),
        processing_class=tokenizer,
        peft_config=get_peft_config(model_args),
    )

    trainer.train()
    trainer.accelerator.print("✅ Training completed.")
    trainer.save_model(training_args.output_dir)
    trainer.accelerator.print(f"💾 Model saved to {training_args.output_dir}.")


if __name__ == "__main__":
    parser = TrlParser((ScriptArguments, SFTConfig, ModelConfig, DatasetMixtureConfig))
    script_args, training_args, model_args, dataset_args, _ = parser.parse_args_and_config(
        return_remaining_strings=True
    )
    main(script_args, training_args, model_args, dataset_args)
