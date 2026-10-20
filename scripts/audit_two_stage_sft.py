#!/usr/bin/env python3
"""Fail-closed, no-GPU label audit for two-stage Qwen3 text SFT.

Run this with the exact tokenizer and chat template intended for training.
It checks actual assistant-mask token IDs, not just the source message text.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Iterator


SPECIAL_IDS = {"<|im_start|>": 3, "<|im_end|>": 4, "<think>": 18, "</think>": 17}


def source_kind(messages: list[dict]) -> str:
    """Classify assistant source text; reject malformed or duplicated traces."""
    if not messages or messages[-1]["role"] != "assistant":
        raise ValueError("the final message must be assistant")
    kinds = set()
    for message in messages:
        if message["role"] not in {"system", "user", "assistant"}:
            raise ValueError("tool and other nontext roles are outside this SFT stage")
        if message["role"] != "assistant":
            continue
        content = message.get("content")
        if not isinstance(content, str) or not content.strip():
            raise ValueError("assistant content must be nonempty text")
        reasoning = message.get("reasoning_content") or ""
        if not isinstance(reasoning, str):
            raise ValueError("reasoning_content must be text when supplied")
        opening, closing = content.count("<think>"), content.count("</think>")
        if opening != closing or opening > 1:
            raise ValueError("unbalanced or repeated think delimiters in source content")
        if reasoning.strip() and opening:
            raise ValueError("reasoning duplicated in content and reasoning_content")
        if opening:
            if not content.lstrip().startswith("<think>"):
                raise ValueError("inline reasoning must begin the assistant turn")
            inside = content.split("<think>", 1)[1].split("</think>", 1)[0]
            if not inside.strip():
                raise ValueError("empty source think span")
            if not content.split("</think>", 1)[1].strip():
                raise ValueError("inline reasoning has no final answer")
            kinds.add("reasoning")
        elif reasoning.strip():
            kinds.add("reasoning")
        else:
            kinds.add("direct")
    return "reasoning" if "reasoning" in kinds else "direct"


def verify_render(tokenizer, messages: list[dict], stage: str, max_length: int) -> dict:
    kind = source_kind(messages)
    if stage == "instruct" and kind != "direct":
        raise ValueError("Instruct stage contains a reasoning-tagged source row")
    prompt_text = tokenizer.apply_chat_template(
        messages[:-1], tokenize=False, add_generation_prompt=True,
    )
    full_text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=False,
    )
    if not full_text.startswith(prompt_text):
        raise ValueError("assistant prompt text is not a prefix of full exchange")
    rendered = tokenizer.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=False,
        return_dict=True, return_assistant_tokens_mask=True,
    )
    ids = rendered["input_ids"]
    mask = rendered.get("assistant_masks")
    if mask is None:
        mask = rendered.get("assistant_tokens_mask")
    if mask is None or len(mask) != len(ids) or not 0 < sum(mask) < len(mask):
        raise ValueError("missing, empty, or all-token assistant loss mask")
    # BPE may merge bytes across the assistant-header/content boundary, so
    # separately tokenized prompt IDs need not be an exact prefix of the full
    # sequence. Check the rendered text boundary and the tokenizer-provided
    # assistant mask instead.
    if len(ids) > max_length:
        raise ValueError(f"rendered row is {len(ids)} tokens, above cap {max_length}")
    target = tokenizer.decode([ids[i] for i, value in enumerate(mask) if value], skip_special_tokens=False)
    if not target.strip():
        raise ValueError("no decoded assistant target")
    if mask[0] or "<|im_start|>user" in target:
        raise ValueError("non-assistant content received assistant loss")
    if messages[-1]["content"].strip() not in target:
        raise ValueError("final assistant content was lost from supervised labels")
    final_reasoning = messages[-1].get("reasoning_content") or ""
    if final_reasoning.strip() and final_reasoning.strip() not in target:
        raise ValueError("final reasoning_content was lost from supervised labels")
    opening, closing = target.count("<think>"), target.count("</think>")
    if opening != closing or opening > 1:
        raise ValueError("malformed rendered think delimiters in assistant labels")
    if opening:
        inside = target.split("<think>", 1)[1].split("</think>", 1)[0]
        if not inside.strip():
            raise ValueError("empty rendered think span in assistant labels")
    if (kind == "reasoning") != bool(opening):
        raise ValueError("reasoning status changed between source and assistant labels")
    if stage == "instruct" and opening:
        raise ValueError("Instruct assistant labels contain think tags")
    return {"kind": kind, "tokens": len(ids), "supervised_tokens": int(sum(mask))}


def iter_rows(path: Path, limit: int) -> Iterator[dict]:
    if path.suffix == ".jsonl":
        with path.open(encoding="utf-8") as stream:
            for index, line in enumerate(stream):
                if index >= limit:
                    break
                yield json.loads(line)
    elif path.suffix == ".parquet":
        import pyarrow.parquet as pq

        seen = 0
        for batch in pq.ParquetFile(path).iter_batches(batch_size=min(limit, 128), columns=["messages"]):
            for row in batch.to_pylist():
                yield row
                seen += 1
                if seen >= limit:
                    return
    else:
        raise ValueError("data must be .jsonl or .parquet")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--template", type=Path, required=True)
    parser.add_argument("--stage", choices=["instruct", "reasoning"], required=True)
    parser.add_argument("--data", type=Path)
    parser.add_argument("--max-rows", type=int, default=128)
    parser.add_argument("--max-length", type=int, default=8192)
    args = parser.parse_args()
    if args.max_rows < 1 or args.max_length < 1:
        raise ValueError("max-rows and max-length must be positive")

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, local_files_only=True)
    tokenizer.chat_template = args.template.read_text(encoding="utf-8")
    for token, expected in SPECIAL_IDS.items():
        actual = tokenizer.encode(token, add_special_tokens=False)
        if actual != [expected]:
            raise ValueError(f"wrong {token} token ID: {actual} != {[expected]}")

    synthetic = [
        [{"role": "user", "content": "USER_SENTINEL_A"},
         {"role": "assistant", "content": "DIRECT_SENTINEL_A"}],
        [{"role": "user", "content": "USER_SENTINEL_B"},
         {"role": "assistant", "reasoning_content": "REASON_SENTINEL_B is verified.",
          "content": "FINAL_SENTINEL_B"}],
    ]
    direct = verify_render(tokenizer, synthetic[0], "instruct", args.max_length)
    reasoning = verify_render(tokenizer, synthetic[1], "reasoning", args.max_length)
    if direct["kind"] != "direct" or reasoning["kind"] != "reasoning":
        raise ValueError("synthetic render classification failed")
    prompt = tokenizer.apply_chat_template(synthetic[0][:-1], tokenize=False, add_generation_prompt=True)
    if "<think>" in prompt or "</think>" in prompt:
        raise ValueError("generation prompt injects think tags")

    counts = {"direct": 0, "reasoning": 0}
    max_tokens = 0
    if args.data:
        for index, row in enumerate(iter_rows(args.data, args.max_rows)):
            try:
                checked = verify_render(tokenizer, row["messages"], args.stage, args.max_length)
            except Exception as exc:
                raise ValueError(f"row {index} failed render audit: {exc}") from exc
            counts[checked["kind"]] += 1
            max_tokens = max(max_tokens, checked["tokens"])
        if not sum(counts.values()):
            raise ValueError("no data rows audited")
        if args.stage == "reasoning" and counts["reasoning"] == 0:
            raise ValueError("Reasoning stage sample contains no nonempty reasoning rows")
    print(json.dumps({
        "stage": args.stage,
        "tokenizer": str(args.tokenizer),
        "template_sha256": hashlib.sha256(args.template.read_bytes()).hexdigest(),
        "synthetic_direct": direct,
        "synthetic_reasoning": reasoning,
        "data_rows_checked": counts,
        "max_rendered_tokens": max_tokens,
    }, indent=2))


if __name__ == "__main__":
    main()
