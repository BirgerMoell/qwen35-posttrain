#!/usr/bin/env python3
"""Load a local Hugging Face model and run deterministic inference smoke tests."""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer


PROMPTS = [
    "Svara på svenska med exakt en kort mening: Vad är Sveriges huvudstad?",
    "Reply with only the result: 17 + 25 = ?",
    "Antworte auf Deutsch in einem kurzen Satz: Was ist Photosynthese?",
]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--max-new-tokens", type=int, default=96)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("GPU is required for this inference validation")

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    config = AutoConfig.from_pretrained(args.model, trust_remote_code=True)
    assert config.model_type == "qwen3"
    assert config.max_position_embeddings == 262_144
    assert config.rope_parameters["rope_theta"] == 64_000_000
    assert str(config.dtype) == "torch.bfloat16", config.dtype
    assert tokenizer.eos_token_id == 4
    assert tokenizer.pad_token_id == 262_144
    assert tokenizer.chat_template

    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    ).eval()

    results = []
    for index, prompt in enumerate(PROMPTS):
        rendered = tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False,
            add_generation_prompt=True,
        )
        assert "<start_of_turn>user" in rendered
        assert rendered.endswith("<start_of_turn>model\n")
        assert "<|im_start|>" not in rendered

        inputs = tokenizer(
            rendered,
            return_tensors="pt",
            add_special_tokens=False,
        ).to(model.device)
        if index == 0:
            with torch.inference_mode():
                logits = model(**inputs, use_cache=False).logits[:, -1, :]
            assert torch.isfinite(logits).all(), "Non-finite logits"

        with torch.inference_mode():
            output = model.generate(
                **inputs,
                max_new_tokens=args.max_new_tokens,
                do_sample=False,
                eos_token_id=tokenizer.eos_token_id,
                pad_token_id=tokenizer.pad_token_id,
                use_cache=True,
            )
        generated = output[0, inputs["input_ids"].shape[1] :]
        text = tokenizer.decode(generated, skip_special_tokens=True).strip()
        raw = tokenizer.decode(generated, skip_special_tokens=False)
        assert text, f"Empty generation for prompt {index}"
        assert "<|im_start|>" not in raw
        assert "<start_of_turn>" not in raw
        record = {
            "prompt": prompt,
            "response": text,
            "generated_tokens": int(generated.numel()),
            "ended_with_eot": bool(generated[-1].item() == tokenizer.eos_token_id),
        }
        results.append(record)
        print(json.dumps(record, ensure_ascii=False), flush=True)

    report = {
        "status": "pass",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "model": str(args.model.resolve()),
        "device": str(model.device),
        "dtype": str(model.dtype),
        "max_position_embeddings": config.max_position_embeddings,
        "rope_theta": config.rope_parameters["rope_theta"],
        "chat_template": "gemma_start_end_of_turn",
        "generations": results,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"[inference] PASS -> {args.out}", flush=True)


if __name__ == "__main__":
    main()
