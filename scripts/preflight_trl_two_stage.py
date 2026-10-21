#!/usr/bin/env python3
"""No-GPU TRL collator smoke for the no-empty-tag SFT template.

Creates a one-layer random Qwen3 model only to exercise SFTTrainer's real
dataset preparation and label masking. It never trains or loads a checkpoint.
"""

from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--template", type=Path, required=True)
    parser.add_argument("--instruct-data", type=Path)
    parser.add_argument("--reasoning-data", type=Path)
    args = parser.parse_args()
    if bool(args.instruct_data) != bool(args.reasoning_data):
        raise ValueError("supply both real-data paths or neither")

    from datasets import Dataset
    from transformers import AutoModelForCausalLM, AutoTokenizer, Qwen3Config
    from trl import SFTConfig, SFTTrainer

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, local_files_only=True)
    tokenizer.chat_template = args.template.read_text(encoding="utf-8")
    config = Qwen3Config(
        vocab_size=len(tokenizer), hidden_size=64, intermediate_size=128,
        num_hidden_layers=1, num_attention_heads=4, num_key_value_heads=2,
        max_position_embeddings=256, eos_token_id=4, pad_token_id=tokenizer.pad_token_id,
    )
    direct_messages = [
        {"role": "user", "content": "USER_SENTINEL_DIRECT"},
        {"role": "assistant", "content": "DIRECT_SENTINEL_ANSWER"},
    ]
    reasoning_messages = [
        {"role": "user", "content": "USER_SENTINEL_REASON"},
        {"role": "assistant", "reasoning_content": "REASON_SENTINEL_STEP is correct.",
         "content": "FINAL_SENTINEL_ANSWER"},
    ]
    real_data = None
    if args.instruct_data:
        real_data = {
            "instruct": json.loads(args.instruct_data.read_text(encoding="utf-8").splitlines()[0])["messages"],
            "reasoning": json.loads(args.reasoning_data.read_text(encoding="utf-8").splitlines()[0])["messages"],
        }
    results = {}
    with tempfile.TemporaryDirectory(prefix="oellm-trl-preflight-") as output_root:
        samples = [("instruct", direct_messages), ("reasoning", reasoning_messages)]
        if real_data:
            samples += [("instruct_real", real_data["instruct"]),
                        ("reasoning_real", real_data["reasoning"])]
        for stage, messages in samples:
            model = AutoModelForCausalLM.from_config(config)
            training_args = SFTConfig(
                output_dir=str(Path(output_root) / stage),
                assistant_only_loss=True, packing=False,
                max_length=8192 if stage.endswith("_real") else 128,
                per_device_train_batch_size=1, report_to="none", use_cpu=True,
                dataset_num_proc=1,
            )
            trainer = SFTTrainer(
                model=model, args=training_args,
                train_dataset=Dataset.from_list([{"messages": messages}]),
                processing_class=tokenizer,
            )
            batch = trainer.data_collator([trainer.train_dataset[0]])
            ids = batch["input_ids"][0].tolist()
            labels = batch["labels"][0].tolist()
            supervised = [token for token, label in zip(ids, labels) if label != -100]
            decoded = tokenizer.decode(supervised, skip_special_tokens=False)
            if "USER_SENTINEL" in decoded or "<think>\n\n</think>" in decoded:
                raise ValueError(f"{stage}: bad supervised target: {decoded!r}")
            if stage == "instruct":
                if "DIRECT_SENTINEL_ANSWER" not in decoded or "<think>" in decoded:
                    raise ValueError(f"Instruct labels are not direct-only: {decoded!r}")
            elif stage == "reasoning":
                if not (
                    "<think>\nREASON_SENTINEL_STEP is correct.\n</think>" in decoded
                    and "FINAL_SENTINEL_ANSWER" in decoded
                ):
                    raise ValueError(f"Reasoning labels lost the verified trace: {decoded!r}")
            elif messages[-1]["content"].strip() not in decoded:
                raise ValueError(f"{stage}: final assistant content was lost")
            if stage.endswith("_real") and ("<think>\n\n</think>" in decoded):
                raise ValueError(f"{stage}: empty think span in real assistant labels")
            results[stage] = {
                "total_tokens": len(ids),
                "supervised_tokens": len(supervised),
                "supervised_text": decoded if not stage.endswith("_real") else f"<{len(decoded)} characters verified>",
            }
            del trainer, model
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
