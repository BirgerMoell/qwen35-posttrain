#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.10"
# dependencies = ["ollama>=0.6"]
# ///
"""Agora with Ollama's hosted web search and web fetch tools (a small agent loop).

    export OLLAMA_API_KEY=...            # create one at https://ollama.com/settings/keys
    uv run local/agora_web.py "Vilka organisationer ingår i OpenEuroLLM?"
    uv run local/agora_web.py --think "..."          # reason before each step
    uv run local/agora_web.py --model birgermoell/agora "..."

The model decides when to call web_search(query) and web_fetch(url); results are passed back as tool messages until it
answers. Search queries and fetched URLs are sent to ollama.com.
"""
import argparse
import json
import sys

from ollama import chat, web_fetch, web_search

SYSTEM = ("You are Agora, a multilingual assistant with web access. For any question about facts (organisations, people, "
          "projects, dates, numbers, current events), first call web_search; never answer such questions from memory. Call "
          "web_fetch to read a page when the search snippets are not enough. Then answer in the user's language, base every "
          "factual claim on what you found, and list the source URLs. If the sources do not say, say that you do not know.")
TOOLS = {"web_search": web_search, "web_fetch": web_fetch}


def run_tool(name: str, args: dict) -> str:
    try:
        result = TOOLS[name](**args)
    except Exception as error:  # surface tool errors to the model instead of crashing the loop
        return f"error: {error}"
    text = str(result)
    return text[:8000] + ("\n[truncated]" if len(text) > 8000 else "")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("question")
    ap.add_argument("--model", default="agora")
    ap.add_argument("--think", action="store_true")
    ap.add_argument("--max-steps", type=int, default=6)
    a = ap.parse_args()
    messages = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": a.question}]
    for step in range(a.max_steps):
        response = chat(model=a.model, messages=messages, tools=[web_search, web_fetch], think=a.think)
        message = response.message
        messages.append(message)
        if a.think and message.thinking:
            print(f"\n[thinking] {message.thinking.strip()[:600]}", file=sys.stderr)
        if not message.tool_calls:
            print("\n" + (message.content or "").strip())
            return
        for call in message.tool_calls:
            name, args = call.function.name, dict(call.function.arguments)
            print(f"[tool] {name}({json.dumps(args, ensure_ascii=False)})", file=sys.stderr)
            messages.append({"role": "tool", "content": run_tool(name, args), "tool_name": name})
    print("\n[stopped after max steps]")


if __name__ == "__main__":
    main()
