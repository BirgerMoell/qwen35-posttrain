#!/usr/bin/env python3
"""Text-only DPO entrypoint (TRL) — passes an explicit tokenizer as processing_class.

Why this exists: the stock `trl.scripts.dpo` lets DPOTrainer auto-load a processing
class. For 2026 multimodal-origin models (Gemma 4, Qwen 3.5) that resolves to a
multimodal `AutoProcessor` which expects image/feature-extractor configs and breaks
text-only preference tuning (KeyError 'images' / "Can't load feature extractor").

This wrapper mirrors trl.scripts.dpo but:
  - loads the model/ref as AutoModelForCausalLM (text-only),
  - loads an AutoTokenizer (not AutoProcessor),
  - passes it to DPOTrainer as processing_class.

Reuses TRL's own config dataclasses + dataset mixture loader, so the same YAML
configs work unchanged.
"""
from __future__ import annotations

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import (
    DatasetMixtureConfig,
    DPOConfig,
    DPOTrainer,
    ModelConfig,
    ScriptArguments,
    TrlParser,
    get_dataset,
    get_peft_config,
)


def main(script_args, training_args, model_args, dataset_args):
    # TRL renamed ModelConfig.torch_dtype -> dtype; support both.
    dtype_str = getattr(model_args, "dtype", None) or getattr(model_args, "torch_dtype", None)
    dtype = dtype_str if dtype_str in (None, "auto") else getattr(torch, dtype_str)
    model_kwargs = dict(
        revision=model_args.model_revision,
        attn_implementation=model_args.attn_implementation,
        dtype=dtype,
        # DPO loads TWO 9B models (policy + reference). Without this, all 8 ranks load both
        # full copies to CPU (8 x 38 GB -> node OOM). low_cpu_mem_usage engages FSDP
        # rank-0-only loading.
        low_cpu_mem_usage=True,
    )

    model = AutoModelForCausalLM.from_pretrained(
        model_args.model_name_or_path,
        trust_remote_code=model_args.trust_remote_code,
        **model_kwargs,
    )
    peft_config = get_peft_config(model_args)
    ref_model = None
    if peft_config is None:
        ref_model = AutoModelForCausalLM.from_pretrained(
            model_args.model_name_or_path,
            trust_remote_code=model_args.trust_remote_code,
            **model_kwargs,
        )

    # 2026 multimodal-origin models (Gemma 4, Qwen 3.5) declare vision/audio block classes
    # in `_no_split_modules`. When loaded text-only via AutoModelForCausalLM those classes
    # don't exist, so FSDP TRANSFORMER_BASED_WRAP fails ("Could not find the transformer
    # layer class ...VisionBlock"). Filter to classes actually present in the model.
    def _prune_no_split_modules(m):
        names = getattr(m, "_no_split_modules", None)
        if names:
            present = {type(sub).__name__ for sub in m.modules()}
            m._no_split_modules = [c for c in names if c in present]
    _prune_no_split_modules(model)
    if ref_model is not None:
        _prune_no_split_modules(ref_model)

    # Explicit text tokenizer — the whole point of this wrapper.
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        trust_remote_code=model_args.trust_remote_code,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    if dataset_args.datasets and script_args.dataset_name:
        dataset = get_dataset(dataset_args)
    elif dataset_args.datasets:
        dataset = get_dataset(dataset_args)
    elif script_args.dataset_name:
        from datasets import load_dataset

        dataset = load_dataset(script_args.dataset_name, name=script_args.dataset_config)
    else:
        raise ValueError("Either `datasets` or `dataset_name` must be provided.")

    trainer = DPOTrainer(
        model,
        ref_model,
        args=training_args,
        train_dataset=dataset[script_args.dataset_train_split],
        eval_dataset=(
            dataset[script_args.dataset_test_split]
            if training_args.eval_strategy != "no"
            else None
        ),
        processing_class=tokenizer,
        peft_config=peft_config,
    )

    trainer.train()
    trainer.accelerator.print("✅ Training completed.")
    trainer.save_model(training_args.output_dir)
    trainer.accelerator.print(f"💾 Model saved to {training_args.output_dir}.")


if __name__ == "__main__":
    parser = TrlParser((ScriptArguments, DPOConfig, ModelConfig, DatasetMixtureConfig))
    script_args, training_args, model_args, dataset_args, _ = parser.parse_args_and_config(
        return_remaining_strings=True
    )
    main(script_args, training_args, model_args, dataset_args)
