#!/usr/bin/env python3
"""Reference-free preference optimization (SimPO) entrypoint — the practical replacement for DPO.

Why: DPO loads TWO 9B models (policy + frozen reference). On one LUMI node the reference copy
repeatedly OOMs CPU during load (FSDP rank-0-only loading wouldn't engage). SimPO (and CPO/ORPO)
are **reference-free** — a single model — so the memory problem vanishes. SimPO also matches or
beats DPO on AlpacaEval-2 / Arena-Hard, so this is an upgrade, not a workaround.

Implemented via TRL's CPOTrainer with loss_type="simpo" (CPO + SimPO gamma). Same flat YAML
schema (ModelConfig + DatasetMixtureConfig + CPOConfig), same {prompt,chosen,rejected} data.
"""
from __future__ import annotations

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import (
    CPOConfig,
    CPOTrainer,
    DatasetMixtureConfig,
    ModelConfig,
    ScriptArguments,
    TrlParser,
    get_dataset,
    get_peft_config,
)


def main(script_args, training_args, model_args, dataset_args):
    dtype_str = getattr(model_args, "dtype", None) or getattr(model_args, "torch_dtype", None)
    dtype = dtype_str if dtype_str in (None, "auto") else getattr(torch, dtype_str)

    # ONE model only — no reference. This is the whole point.
    model = AutoModelForCausalLM.from_pretrained(
        model_args.model_name_or_path,
        revision=model_args.model_revision,
        attn_implementation=model_args.attn_implementation,
        dtype=dtype,
        trust_remote_code=model_args.trust_remote_code,
        low_cpu_mem_usage=True,
    )

    # Drop phantom vision/audio classes so FSDP TRANSFORMER_BASED_WRAP can wrap the text model.
    names = getattr(model, "_no_split_modules", None)
    if names:
        present = {type(m).__name__ for m in model.modules()}
        model._no_split_modules = [c for c in names if c in present]

    tokenizer = AutoTokenizer.from_pretrained(
        model_args.model_name_or_path, trust_remote_code=model_args.trust_remote_code
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    dataset = get_dataset(dataset_args)

    trainer = CPOTrainer(
        model,
        args=training_args,
        train_dataset=dataset[script_args.dataset_train_split],
        eval_dataset=(
            dataset[script_args.dataset_test_split]
            if training_args.eval_strategy != "no" else None
        ),
        processing_class=tokenizer,
        peft_config=get_peft_config(model_args),
    )

    trainer.train()
    trainer.accelerator.print("✅ Training completed.")
    trainer.save_model(training_args.output_dir)
    trainer.accelerator.print(f"💾 Model saved to {training_args.output_dir}.")


if __name__ == "__main__":
    parser = TrlParser((ScriptArguments, CPOConfig, ModelConfig, DatasetMixtureConfig))
    script_args, training_args, model_args, dataset_args, _ = parser.parse_args_and_config(
        return_remaining_strings=True
    )
    main(script_args, training_args, model_args, dataset_args)
