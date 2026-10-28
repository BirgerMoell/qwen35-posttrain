#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.10"
# dependencies = ["ollama>=0.6"]
# ///
"""Chat with Agora with a thinking budget (budget forcing), through Ollama's raw generate API.

    uv run local/agora_chat.py "Ett tåg går 14:20 och kommer fram 16:05. Hur lång är resan?"            # direct mode
    uv run local/agora_chat.py --think "..."                     # thinking, no limit (the model closes the trace itself)
    uv run local/agora_chat.py --think --budget 300 "..."        # at most 300 thinking tokens, then answer

With a budget, the reasoning is generated until it closes (</think>) or hits the budget; on the budget, the script
closes the trace itself ("\n</think>\n\n") and lets the model write the answer from what it has. The prompt is the
model's own template (ChatML; thinking opens "<think>\n" after the assistant header). Training used up to 6-7K
thinking tokens; most traces after RL are 0.5-2K, and very small budgets cost accuracy on hard problems.
"""
import argparse
import sys
import time

import ollama


def prompt_for(question: str, system: str | None, think: bool) -> str:
    p = f"<|im_start|>system\n{system}<|im_end|>\n" if system else ""
    p += f"<|im_start|>user\n{question}<|im_end|>\n<|im_start|>assistant\n"
    return p + ("<think>\n" if think else "")


def generate(model: str, prompt: str, stop: list[str], limit: int | None) -> tuple[str, int]:
    options = {"stop": stop, "temperature": 0.6, "top_p": 0.95, "num_ctx": 32768}
    if limit:
        options["num_predict"] = limit
    r = ollama.generate(model=model, prompt=prompt, raw=True, options=options)
    return r.response, r.eval_count or 0


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("question")
    ap.add_argument("--model", default="agora")
    ap.add_argument("--think", action="store_true")
    ap.add_argument("--budget", type=int, default=0, help="max thinking tokens (0 = no limit)")
    ap.add_argument("--system")
    a = ap.parse_args()
    t0 = time.time()
    prompt = prompt_for(a.question, a.system, a.think)
    if a.think:
        thinking, n = generate(a.model, prompt, ["</think>", "<|im_end|>"], a.budget or 8192)
        cut = bool(a.budget) and n >= a.budget
        print(f"[thinking: {n} tokens{', budget reached, trace closed by the script' if cut else ''}]\n{thinking.strip()}\n",
              file=sys.stderr)
        prompt += thinking.rstrip() + "\n</think>\n\n"
    answer, m = generate(a.model, prompt, ["<|im_end|>"], 4096)
    print(answer.strip())
    print(f"\n[answer: {m} tokens, {time.time() - t0:.1f} s]", file=sys.stderr)


if __name__ == "__main__":
    main()
