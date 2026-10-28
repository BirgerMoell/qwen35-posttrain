#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.10"
# dependencies = ["ollama>=0.6", "ddgs>=9", "httpx", "beautifulsoup4"]
# ///
"""Agora with web search and web fetch tools (a small agent loop).

    uv run local/agora_web.py "Vilka organisationer ingår i OpenEuroLLM?"        # free: DuckDuckGo + direct fetch
    OLLAMA_API_KEY=... uv run local/agora_web.py --backend ollama "..."          # Ollama's hosted tools (needs credits)
    uv run local/agora_web.py --think "..."          # reason before each step
    uv run local/agora_web.py --model birgermoell/agora "..."

The model decides when to call web_search(query) and web_fetch(url); results are passed back as tool messages until it
answers. With --backend ollama, queries and URLs go to ollama.com (web search needs usage credits on the account); the
default backend sends queries to DuckDuckGo and fetches pages directly.
"""
import argparse
import json
import sys

from ollama import chat


def ddg_search(query: str, max_results: int = 5) -> str:
    """Search the web and return the top results (title, URL, snippet).

    Args:
        query: the search query
        max_results: number of results (1-10)
    """
    from ddgs import DDGS
    hits = DDGS().text(query, max_results=max(1, min(int(max_results), 10)))
    return "\n\n".join(f"{h.get('title')}\n{h.get('href')}\n{h.get('body')}" for h in hits) or "no results"


def fetch_page(url: str) -> str:
    """Fetch a web page and return its main text.

    Args:
        url: the page URL
    """
    import httpx
    from bs4 import BeautifulSoup
    r = httpx.get(url, follow_redirects=True, timeout=20, headers={"User-Agent": "Mozilla/5.0 (agora-local)"})
    soup = BeautifulSoup(r.text, "html.parser")
    for tag in soup(["script", "style", "nav", "footer", "header", "noscript"]):
        tag.decompose()
    text = " ".join(soup.get_text(" ").split())
    return f"{soup.title.string.strip() if soup.title and soup.title.string else url}\n{text}"

SYSTEM = ("You are Agora, a multilingual assistant with web access. For any question about facts (organisations, people, "
          "projects, dates, numbers, current events), first call web_search; never answer such questions from memory. Call "
          "web_fetch to read a page when the search snippets are not enough. Then answer in the user's language, base every "
          "factual claim on what you found, and list the source URLs. If the sources do not say, say that you do not know.")
TOOLS = {}


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
    ap.add_argument("--backend", choices=["ddg", "ollama"], default="ddg")
    a = ap.parse_args()
    if a.backend == "ollama":
        from ollama import web_fetch, web_search
    else:
        def web_search(query: str, max_results: int = 5) -> str:
            """Search the web for a query and return the top results.

            Args:
                query: the search query
                max_results: number of results (1-10)
            """
            return ddg_search(query, max_results)

        def web_fetch(url: str) -> str:
            """Fetch a web page by URL and return its text.

            Args:
                url: the page URL
            """
            return fetch_page(url)
    TOOLS.update({"web_search": web_search, "web_fetch": web_fetch})
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
