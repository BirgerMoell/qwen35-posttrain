#!/usr/bin/env python3
"""Create a publishable BF16 Hugging Face export of an OELLM SFT checkpoint.

The LUMI FSDP training stack consolidates the final checkpoint in FP32.  This
script loads it on CPU in BF16, saves bounded-size safetensors shards, preserves
the Gemma-style chat template, and validates the long-context metadata before
atomically installing the export directory.
"""
from __future__ import annotations

import argparse
import json
import shutil
import tempfile
from pathlib import Path

import torch
from safetensors import safe_open
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer, GenerationConfig


EXPECTED_CONTEXT = 262_144
EXPECTED_THETA = 64_000_000
EXPECTED_EOT_ID = 4
EXPECTED_PAD_ID = 262_144
EXPECTED_VOCAB_SIZE = 263_168


def rope_theta(config) -> int | None:
    params = getattr(config, "rope_parameters", None)
    if isinstance(params, dict):
        return params.get("rope_theta")
    return getattr(config, "rope_theta", None)


def validate_metadata(src: Path) -> tuple[object, object]:
    config = AutoConfig.from_pretrained(src, trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained(src, trust_remote_code=True)

    assert config.model_type == "qwen3", config.model_type
    assert config.max_position_embeddings == EXPECTED_CONTEXT, config.max_position_embeddings
    assert rope_theta(config) == EXPECTED_THETA, rope_theta(config)
    assert config.vocab_size == EXPECTED_VOCAB_SIZE, config.vocab_size
    assert tokenizer.eos_token_id == EXPECTED_EOT_ID, tokenizer.eos_token_id
    assert tokenizer.pad_token_id == EXPECTED_PAD_ID, tokenizer.pad_token_id
    assert tokenizer.chat_template, "Missing chat template"
    assert "<start_of_turn>" in tokenizer.chat_template
    assert "<end_of_turn>" in tokenizer.chat_template
    assert "<|im_start|>" not in tokenizer.chat_template

    rendered = tokenizer.apply_chat_template(
        [{"role": "user", "content": "Svara kort på svenska."}],
        tokenize=False,
        add_generation_prompt=True,
    )
    assert "<start_of_turn>user" in rendered, rendered
    assert rendered.endswith("<start_of_turn>model\n"), rendered
    return config, tokenizer


def validate_saved_dtypes(dst: Path) -> dict[str, int]:
    shards = sorted(dst.glob("*.safetensors"))
    assert shards, "No safetensors shards were written"
    tensor_count = 0
    bf16_count = 0
    total_bytes = 0
    for shard in shards:
        total_bytes += shard.stat().st_size
        with safe_open(shard, framework="pt", device="cpu") as handle:
            for name in handle.keys():
                tensor = handle.get_tensor(name)
                tensor_count += 1
                if tensor.is_floating_point():
                    assert tensor.dtype == torch.bfloat16, f"{name}: {tensor.dtype}"
                    bf16_count += 1
    return {
        "shards": len(shards),
        "tensors": tensor_count,
        "bf16_tensors": bf16_count,
        "weight_bytes": total_bytes,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--src", type=Path, required=True)
    parser.add_argument("--dst", type=Path, required=True)
    parser.add_argument("--card", type=Path, required=True)
    parser.add_argument("--max-shard-size", default="5GB")
    args = parser.parse_args()

    src = args.src.resolve()
    dst = args.dst.resolve()
    card = args.card.resolve()
    if not (src / "model.safetensors").is_file():
        raise SystemExit(f"Missing consolidated model: {src / 'model.safetensors'}")
    if not card.is_file():
        raise SystemExit(f"Missing model card: {card}")
    if dst.exists():
        raise SystemExit(f"Destination already exists; refusing to overwrite: {dst}")

    _, tokenizer = validate_metadata(src)
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = Path(tempfile.mkdtemp(prefix=f".{dst.name}.tmp-", dir=dst.parent))
    try:
        print(f"[export] loading {src} as BF16 on CPU", flush=True)
        model = AutoModelForCausalLM.from_pretrained(
            src,
            dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
            trust_remote_code=True,
        )
        model.config.use_cache = True
        model.config.eos_token_id = EXPECTED_EOT_ID
        model.config.pad_token_id = EXPECTED_PAD_ID
        model.save_pretrained(
            tmp,
            safe_serialization=True,
            max_shard_size=args.max_shard_size,
        )
        tokenizer.save_pretrained(tmp)

        generation = GenerationConfig.from_pretrained(src)
        generation.eos_token_id = EXPECTED_EOT_ID
        generation.pad_token_id = EXPECTED_PAD_ID
        generation.use_cache = True
        generation.save_pretrained(tmp)
        shutil.copy2(card, tmp / "README.md")

        saved_config, saved_tokenizer = validate_metadata(tmp)
        assert str(saved_config.dtype) == "torch.bfloat16", saved_config.dtype
        assert saved_config.use_cache is True
        assert saved_tokenizer.eos_token_id == EXPECTED_EOT_ID
        stats = validate_saved_dtypes(tmp)

        manifest = {
            "source_checkpoint": src.name,
            "base_model": "openeurollm/oellm-9b-256k-theta64m-prelude",
            "dtype": "bfloat16",
            "max_position_embeddings": saved_config.max_position_embeddings,
            "rope_theta": rope_theta(saved_config),
            "eos_token": "<end_of_turn>",
            "eos_token_id": EXPECTED_EOT_ID,
            "pad_token_id": EXPECTED_PAD_ID,
            **stats,
        }
        (tmp / "export_validation.json").write_text(
            json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
        )
        tmp.rename(dst)
        print(json.dumps(manifest, indent=2), flush=True)
        print(f"[export] ready: {dst}", flush=True)
    except BaseException:
        shutil.rmtree(tmp, ignore_errors=True)
        raise


if __name__ == "__main__":
    main()
