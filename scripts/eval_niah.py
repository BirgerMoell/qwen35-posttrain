#!/usr/bin/env python3
"""Needle-in-a-haystack retrieval grid for the OELLM-9B 256K chat checkpoint.

The evaluator inserts one deterministic passkey at controlled depths in a long
background document, asks for only that passkey, and reports strict exact match.
It validates the 64M RoPE theta and Gemma-style turn markers before loading the
model. Results are written after every cell so interrupted 128K/256K runs resume.

Example (inside the LUMI training container):
  python scripts/eval_niah.py \
    --model output/oellm9b-256k-simpo \
    --out eval/results/oellm9b_256k_niah.json
"""
from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
import re
import time

import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer


FILLER_BLOCK = (
    "The archive describes ordinary administrative work. Staff catalogue correspondence, "
    "review meeting notes, preserve maps, and organize reports for later research. None of "
    "these routine paragraphs contains a passkey or an answer to the final question. "
    "Researchers compare old filing practices, repair damaged bindings, and update descriptive "
    "labels while preserving the original order of the material.\n"
)
INTRO = (
    "Read the following archive carefully. One important record is hidden among routine "
    "background material. Remember that record and answer the question at the end.\n\n"
)


def csv_ints(value: str) -> list[int]:
    return [int(part.strip()) for part in value.split(",") if part.strip()]


def extract_rope_theta(config) -> float | None:
    theta = getattr(config, "rope_theta", None)
    if theta is not None:
        return float(theta)
    for name in ("rope_parameters", "rope_scaling"):
        value = getattr(config, name, None)
        if isinstance(value, dict) and value.get("rope_theta") is not None:
            return float(value["rope_theta"])
    return None


def render_ids(tokenizer, content: str) -> list[int]:
    ids = tokenizer.apply_chat_template(
        [{"role": "user", "content": content}],
        tokenize=True,
        add_generation_prompt=True,
    )
    if isinstance(ids, torch.Tensor):
        ids = ids.squeeze(0).tolist()
    if ids and isinstance(ids[0], list):
        ids = ids[0]
    return ids


def make_secret(model: str, requested_length: int, depth: int) -> tuple[str, str]:
    digest = hashlib.sha256(f"{model}|{requested_length}|{depth}".encode()).hexdigest().upper()
    return f"record-{digest[:8].lower()}", f"NIAH-{digest[8:20]}"


def build_case(tokenizer, model_name: str, requested_length: int, target_length: int, depth: int):
    record_id, secret = make_secret(model_name, requested_length, depth)
    needle = f"IMPORTANT RECORD: The passkey for {record_id} is {secret}.\n"
    question = (
        f"\nQuestion: What is the passkey for {record_id}? "
        "Answer with only the exact passkey and no other words."
    )

    def content_for(repeats: int) -> str:
        before = round(repeats * depth / 100)
        return INTRO + FILLER_BLOCK * before + needle + FILLER_BLOCK * (repeats - before) + question

    base_length = len(render_ids(tokenizer, content_for(0)))
    block_length = max(1, len(tokenizer(FILLER_BLOCK, add_special_tokens=False)["input_ids"]))
    if target_length < base_length:
        raise ValueError(f"Target {target_length} is smaller than prompt overhead {base_length}")

    repeats = max(0, (target_length - base_length) // block_length)
    ids = []
    content = ""
    # Token boundaries at repeated-block joins can shift by a few tokens. Two or three
    # corrections normally place the input within one filler block of the target.
    for _ in range(5):
        content = content_for(repeats)
        ids = render_ids(tokenizer, content)
        delta = target_length - len(ids)
        if 0 <= delta < block_length:
            break
        adjustment = int(delta / block_length)
        if adjustment == 0:
            adjustment = -1 if delta < 0 else 1
        repeats = max(0, repeats + adjustment)

    while len(ids) > target_length and repeats > 0:
        repeats -= 1
        content = content_for(repeats)
        ids = render_ids(tokenizer, content)

    secret_candidates = [
        tokenizer(" " + secret, add_special_tokens=False)["input_ids"],
        tokenizer(secret, add_special_tokens=False)["input_ids"],
    ]
    needle_index = None
    for candidate in secret_candidates:
        for index in range(len(ids) - len(candidate) + 1):
            if ids[index:index + len(candidate)] == candidate:
                needle_index = index
                break
        if needle_index is not None:
            break

    return {
        "input_ids": ids,
        "content": content,
        "record_id": record_id,
        "secret": secret,
        "filler_blocks": repeats,
        "needle_token_index": needle_index,
        "needle_prompt_fraction": (
            round(needle_index / len(ids), 6) if needle_index is not None and ids else None
        ),
    }


def normalize_answer(value: str) -> str:
    value = re.sub(r"\s+", "", value or "")
    return value.strip("\"'`.,:;!?()[]{} ").upper()


def write_report(path: pathlib.Path, report: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    temporary.replace(path)


def update_grid(report: dict):
    grid = {}
    for result in report["results"]:
        length = str(result["requested_length"])
        depth = str(result["depth_percent"])
        grid.setdefault(length, {})[depth] = result.get("strict_exact_match")
    report["strict_exact_match_grid"] = grid


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--lengths", default="8192,32768,131072,262144")
    parser.add_argument("--depths", default="0,25,50,75,100")
    parser.add_argument("--max-new-tokens", type=int, default=24)
    parser.add_argument("--attn-implementation", default="flash_attention_2")
    parser.add_argument("--device-map", default="auto")
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--allow-nonstandard-theta", action="store_true")
    args = parser.parse_args()

    lengths = csv_ints(args.lengths)
    depths = csv_ints(args.depths)
    if not lengths or any(length <= 0 for length in lengths):
        raise SystemExit("--lengths must contain positive integers")
    if not depths or any(depth < 0 or depth > 100 for depth in depths):
        raise SystemExit("--depths must be between 0 and 100")

    config = AutoConfig.from_pretrained(args.model, trust_remote_code=True)
    theta = extract_rope_theta(config)
    if not args.allow_nonstandard_theta and (theta is None or abs(theta - 64_000_000) > 1):
        raise SystemExit(f"Expected rope_theta=64000000, found {theta!r}")
    model_limit = int(getattr(config, "max_position_embeddings", 0) or 0)
    if model_limit < max(lengths):
        raise SystemExit(f"Model limit {model_limit} is below requested grid maximum {max(lengths)}")

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    template = tokenizer.chat_template or ""
    if "<start_of_turn>" not in template or "<end_of_turn>" not in template:
        raise SystemExit("Expected the staged Gemma-style <start_of_turn>/<end_of_turn> chat template")
    eot_id = tokenizer.convert_tokens_to_ids("<end_of_turn>")
    if eot_id == tokenizer.unk_token_id:
        raise SystemExit("<end_of_turn> maps to the unknown token")
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    output = pathlib.Path(args.out)
    if args.resume and output.exists():
        report = json.loads(output.read_text())
        if report.get("model") != args.model:
            raise SystemExit(f"Refusing to resume {output}: model differs")
    else:
        report = {
            "model": args.model,
            "rope_theta": theta,
            "max_position_embeddings": model_limit,
            "requested_lengths": lengths,
            "depths_percent": depths,
            "max_new_tokens": args.max_new_tokens,
            "results": [],
        }
    completed = {(r["requested_length"], r["depth_percent"]) for r in report["results"]}

    print(f"[niah] loading {args.model}; theta={theta:g}; limit={model_limit}", flush=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        dtype=torch.bfloat16,
        device_map=args.device_map,
        attn_implementation=args.attn_implementation,
        trust_remote_code=True,
        low_cpu_mem_usage=True,
    ).eval()
    input_device = model.get_input_embeddings().weight.device
    eos_ids = list(dict.fromkeys([eot_id, tokenizer.eos_token_id]))

    for requested_length in lengths:
        target_length = min(requested_length, model_limit - args.max_new_tokens)
        for depth in depths:
            key = (requested_length, depth)
            if key in completed:
                print(f"[niah] skip completed length={requested_length} depth={depth}%", flush=True)
                continue

            started = time.time()
            try:
                case = build_case(tokenizer, args.model, requested_length, target_length, depth)
                input_ids = torch.tensor([case.pop("input_ids")], dtype=torch.long, device=input_device)
                attention_mask = torch.ones_like(input_ids)
                prompt_tokens = input_ids.shape[1]
                with torch.inference_mode():
                    generated = model.generate(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        max_new_tokens=args.max_new_tokens,
                        do_sample=False,
                        eos_token_id=eos_ids,
                        pad_token_id=tokenizer.pad_token_id,
                    )
                answer = tokenizer.decode(generated[0, prompt_tokens:], skip_special_tokens=True).strip()
                secret = case["secret"]
                result = {
                    "requested_length": requested_length,
                    "target_input_length": target_length,
                    "actual_input_tokens": prompt_tokens,
                    "depth_percent": depth,
                    **{key: value for key, value in case.items() if key != "content"},
                    "answer": answer,
                    "strict_exact_match": normalize_answer(answer) == normalize_answer(secret),
                    "contains_match": secret.upper() in answer.upper(),
                    "elapsed_seconds": round(time.time() - started, 3),
                }
                print(
                    f"[niah] length={requested_length} actual={prompt_tokens} depth={depth}% "
                    f"exact={result['strict_exact_match']} answer={answer!r}",
                    flush=True,
                )
            except Exception as error:
                result = {
                    "requested_length": requested_length,
                    "target_input_length": target_length,
                    "depth_percent": depth,
                    "strict_exact_match": False,
                    "contains_match": False,
                    "error": f"{type(error).__name__}: {error}",
                    "elapsed_seconds": round(time.time() - started, 3),
                }
                print(f"[niah] ERROR length={requested_length} depth={depth}%: {error}", flush=True)
            report["results"].append(result)
            update_grid(report)
            write_report(output, report)
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    passed = sum(bool(result.get("strict_exact_match")) for result in report["results"])
    print(f"[niah] strict exact: {passed}/{len(report['results'])}; wrote {output}", flush=True)


if __name__ == "__main__":
    main()
