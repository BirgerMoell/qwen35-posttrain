#!/usr/bin/env python3
"""Build tool-use / agentic SFT (+ GRPO) data, normalized to Qwen3.5's native tool-call format.

We fold function-calling into the v-next mobile mix. The base Qwen3.5 already ships a Hermes-style
tool template (`<tools>...</tools>` in the system turn, `<tool_call>{json}</tool_call>` from the
assistant), so we teach a capability the arch already scaffolds — low risk.

Output schema matches the rest of our SFT parquets exactly so it drops straight into the mix:
    {"messages": [{"role","content"}...], "lang": "en", "source": "...", "kind": "tool_use"}
The tool definitions are baked into the SYSTEM message using Qwen3.5's exact wrapper, so training
matches inference (where the tools are injected via apply_chat_template(tools=...)).

Also emits a GRPO jsonl for the verifiable stage (single-call rows only, where the gold is a clean
checkable target):
    {"prompt": [system, user], "answer": "<json list of {name, arguments}>", "language": "en"}

Sources (all openly licensed; xLAM is gated -> pass --hf-token or it's skipped):
  - Salesforce/xlam-function-calling-60k  (APIGen; verifiable single/parallel/multi calls)  [gated]
  - NousResearch/hermes-function-calling-v1  (already Qwen/Hermes-formatted, multi-turn)
  - Team-ACE/ToolACE  (multi-turn, dependency chains, 26k APIs)

Run inside the LUMI container (has datasets/pyarrow) on a login node (needs internet):
  python scripts/build_tooluse_sft.py --out posttrain-data/eu-tooluse-parquet [--hf-token hf_...]
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

from datasets import load_dataset

# Qwen3.5 native tool system block (reproduces what apply_chat_template(tools=...) renders).
_SYS_PREAMBLE = (
    "You are a helpful assistant with access to tools.\n\n"
    "# Tools\n\n"
    "You may call one or more functions to assist with the user query.\n\n"
    "You are provided with function signatures within <tools></tools> XML tags:\n<tools>"
)
_SYS_SUFFIX = (
    "\n</tools>\n\n"
    "For each function call, return a json object with function name and arguments within "
    "<tool_call></tool_call> XML tags:\n"
    '<tool_call>\n{"name": <function-name>, "arguments": <args-json-object>}\n</tool_call>'
)


def _tools_system(tools: list) -> str:
    lines = "\n".join(json.dumps(t, ensure_ascii=False) for t in tools)
    return f"{_SYS_PREAMBLE}\n{lines}{_SYS_SUFFIX}"


def _tool_call_text(calls: list) -> str:
    """One <tool_call> block per call, newline-separated — Qwen3.5's assistant format."""
    return "\n".join(
        f'<tool_call>\n{{"name": "{c["name"]}", "arguments": '
        f'{json.dumps(c.get("arguments", {}), ensure_ascii=False)}}}\n</tool_call>'
        for c in calls
    )


def _loads(x):
    return json.loads(x) if isinstance(x, str) else x


# ---------------------------------------------------------------------------- xLAM (APIGen)
def gen_xlam(token, limit):
    ds = load_dataset("Salesforce/xlam-function-calling-60k", split="train", token=token)
    for row in ds.select(range(min(limit, len(ds)))) if limit else ds:
        try:
            tools = _loads(row["tools"])
            calls = _loads(row["answers"])
            if not tools or not calls:
                continue
            messages = [
                {"role": "system", "content": _tools_system(tools)},
                {"role": "user", "content": row["query"]},
                {"role": "assistant", "content": _tool_call_text(calls)},
            ]
            yield {"messages": messages, "lang": "en", "source": "xlam-60k", "kind": "tool_use",
                   "n_calls": len(calls)}
        except Exception:
            continue


# ---------------------------------------------------------------------------- ToolACE
def _extract_json_array(text: str):
    """Pull the first balanced [...] JSON array out of ToolACE's system prompt (the tool list)."""
    i = text.find("[")
    while i != -1:
        depth, j = 0, i
        while j < len(text):
            if text[j] == "[":
                depth += 1
            elif text[j] == "]":
                depth -= 1
                if depth == 0:
                    try:
                        return json.loads(text[i:j + 1])
                    except Exception:
                        break
            j += 1
        i = text.find("[", i + 1)
    return None


def _parse_pythonic_calls(text: str):
    """Parse ToolACE's pythonic assistant turn `[fn(a=1, b='x'), ...]` -> [{name, arguments}].

    These are (almost) valid Python expressions, so we parse with `ast` and read keyword args via
    literal_eval — robust to nested lists/dicts/quotes, unlike a regex."""
    import ast
    text = (text or "").strip()
    if not (text.startswith("[") and text.endswith("]")):
        return None
    try:
        tree = ast.parse(text, mode="eval")
    except Exception:
        return None
    calls = []
    nodes = tree.body.elts if isinstance(tree.body, ast.List) else [tree.body]
    for node in nodes:
        if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Name):
            return None  # not a clean call list -> skip the whole turn
        args = {}
        for kw in node.keywords:
            try:
                args[kw.arg] = ast.literal_eval(kw.value)
            except Exception:
                return None
        calls.append({"name": node.func.id, "arguments": args})
    return calls or None


def gen_toolace(limit):
    ds = load_dataset("Team-ACE/ToolACE", split="train")
    role_map = {"user": "user", "human": "user", "tool": "tool", "observation": "tool"}
    n = 0
    for row in ds:
        if limit and n >= limit:
            break
        tools = _extract_json_array(row.get("system", ""))
        convs = row.get("conversations", [])
        if not tools or not convs:
            continue
        messages = [{"role": "system", "content": _tools_system(tools)}]
        ok = True
        for turn in convs:
            frm = str(turn.get("from", "")).lower()
            val = turn.get("value", "")
            if frm in ("assistant", "gpt"):
                calls = _parse_pythonic_calls(val)
                if calls:                       # tool-call turn -> re-render as Qwen native JSON
                    messages.append({"role": "assistant", "content": _tool_call_text(calls)})
                else:                           # plain-text assistant turn -> keep as-is
                    messages.append({"role": "assistant", "content": val})
            else:
                r = role_map.get(frm)
                if r is None:
                    ok = False
                    break
                messages.append({"role": r, "content": val})
        if not ok or len(messages) < 3 or not any(m["role"] == "assistant" for m in messages):
            continue
        # require at least one real tool call in the conversation
        if not any("<tool_call>" in m["content"] for m in messages if m["role"] == "assistant"):
            continue
        yield {"messages": messages, "lang": "en", "source": "toolace", "kind": "tool_use"}
        n += 1


# ---------------------------------------------------------------------------- Glaive FC v2
def _balanced_json_objects(text: str):
    """Yield every balanced top-level {...} JSON object in a blob (Glaive lists tool defs this way)."""
    objs, i = [], text.find("{")
    while i != -1:
        depth, j, instr, esc = 0, i, False, False
        while j < len(text):
            c = text[j]
            if instr:
                if esc:
                    esc = False
                elif c == "\\":
                    esc = True
                elif c == '"':
                    instr = False
            else:
                if c == '"':
                    instr = True
                elif c == "{":
                    depth += 1
                elif c == "}":
                    depth -= 1
                    if depth == 0:
                        try:
                            objs.append(json.loads(text[i:j + 1]))
                        except Exception:
                            pass
                        break
            j += 1
        i = text.find("{", j + 1)
    return objs


def _glaive_call(seg: str):
    """Parse Glaive's `<functioncall> {"name":..,"arguments":'{...}'}` -> {name, arguments dict}.
    Glaive wraps the inner arguments JSON in SINGLE quotes, so the outer object isn't valid JSON;
    extract the name by regex and the inner {...} (double-quoted, valid) separately."""
    k = seg.find("<functioncall>")
    if k == -1:
        return None
    blob = seg[k + len("<functioncall>"):].split("<|endoftext|>")[0].strip()
    m = _re.search(r'"name"\s*:\s*"([^"]+)"', blob)
    if not m:
        return None
    args = {}
    a = blob.find('"arguments"')
    if a != -1:
        b = blob.find("{", a)
        if b != -1:
            objs = _balanced_json_objects(blob[b:])
            if objs:
                args = objs[0]
    return {"name": m.group(1), "arguments": args}


def gen_glaive(limit):
    ds = load_dataset("glaiveai/glaive-function-calling-v2", split="train")
    marker = _re.compile(r"(USER:|ASSISTANT:|FUNCTION RESPONSE:)")
    n = 0
    for row in ds:
        if limit and n >= limit:
            break
        sys_txt = row.get("system", "") or ""
        chat = row.get("chat", "") or ""
        tools = _balanced_json_objects(sys_txt.split("required -", 1)[-1])
        if not tools or "USER:" not in chat:
            continue
        messages = [{"role": "system", "content": _tools_system(tools)}]
        parts = marker.split(chat)
        # parts = ['', 'USER:', ' ...', 'ASSISTANT:', ' ...', ...] -> pair marker+body
        it = iter(parts[1:])
        ok = True
        for tag, body in zip(it, it):
            body = body.replace("<|endoftext|>", "").strip()
            if tag == "USER:":
                messages.append({"role": "user", "content": body})
            elif tag == "FUNCTION RESPONSE:":
                messages.append({"role": "tool", "content": body})
            elif tag == "ASSISTANT:":
                if "<functioncall>" in body:
                    call = _glaive_call(body)
                    if not call or not call.get("name"):
                        ok = False
                        break
                    messages.append({"role": "assistant", "content": _tool_call_text([call])})
                else:
                    messages.append({"role": "assistant", "content": body})
        if not ok or not any(m["role"] == "assistant" and "<tool_call>" in m["content"]
                             for m in messages):
            continue
        yield {"messages": messages, "lang": "en", "source": "glaive-v2", "kind": "tool_use"}
        n += 1


def gen_glaive_refusals(limit):
    """Harvest Glaive's 'don't call' turns (tools present, assistant declines) as GRPO ABSTENTION
    examples: prompt=[system(tools), user], gold=[] (empty). Balances the reward so it doesn't
    reward-hack into always-calling."""
    ds = load_dataset("glaiveai/glaive-function-calling-v2", split="train")
    marker = _re.compile(r"(USER:|ASSISTANT:|FUNCTION RESPONSE:)")
    n = 0
    for row in ds:
        if limit and n >= limit:
            break
        sys_txt, chat = row.get("system", "") or "", row.get("chat", "") or ""
        tools = _balanced_json_objects(sys_txt.split("required -", 1)[-1])
        if not tools or "USER:" not in chat:
            continue
        parts = marker.split(chat)
        it = iter(parts[1:])
        first_user = None
        for tag, body in zip(it, it):
            body = body.replace("<|endoftext|>", "").strip()
            if tag == "USER:" and first_user is None:
                first_user = body
            elif tag == "ASSISTANT:" and first_user is not None:
                if "<functioncall>" not in body:   # the model declined -> a relevance example
                    yield {"messages": [{"role": "system", "content": _tools_system(tools)},
                                        {"role": "user", "content": first_user}],
                           "lang": "en", "source": "glaive-refusal", "kind": "refusal"}
                    n += 1
                break


# ---------------------------------------------------------------------------- Hermes FC v1
def gen_hermes(limit):
    # config with clean single-turn function calls; already Hermes/Qwen formatted
    ds = load_dataset("NousResearch/hermes-function-calling-v1", "func_calling_singleturn",
                      split="train")
    role_map = {"system": "system", "human": "user", "user": "user", "gpt": "assistant",
                "assistant": "assistant", "tool": "tool"}
    n = 0
    for row in ds:
        if limit and n >= limit:
            break
        convs = row.get("conversations") or []
        messages = []
        for turn in convs:
            r = role_map.get(str(turn.get("from", "")).lower())
            if r is None:
                continue
            messages.append({"role": r, "content": turn.get("value", "")})
        if len(messages) < 2 or not any(m["role"] == "assistant" for m in messages):
            continue
        yield {"messages": messages, "lang": "en", "source": "hermes-fc-v1", "kind": "tool_use"}
        n += 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="posttrain-data/eu-tooluse-parquet")
    ap.add_argument("--hf-token", default=None, help="needed for the gated xLAM download")
    ap.add_argument("--xlam-limit", type=int, default=40000)
    ap.add_argument("--glaive-limit", type=int, default=50000)  # non-gated JSON-native backbone
    ap.add_argument("--toolace-limit", type=int, default=15000)
    ap.add_argument("--hermes-limit", type=int, default=8000)
    ap.add_argument("--grpo-refusal-limit", type=int, default=20000,
                    help="abstention/relevance rows (empty gold) for the balanced GRPO reward")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    rows = []
    for name, gen in [
        ("xlam", lambda: gen_xlam(args.hf_token, args.xlam_limit)),
        ("glaive", lambda: gen_glaive(args.glaive_limit)),
        ("toolace", lambda: gen_toolace(args.toolace_limit)),
        ("hermes", lambda: gen_hermes(args.hermes_limit)),
    ]:
        try:
            got = list(gen())
            rows.extend(got)
            print(f"  {name}: {len(got)} rows")
        except Exception as e:
            print(f"  {name}: SKIPPED ({type(e).__name__}: {e})")

    random.Random(args.seed).shuffle(rows)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    # SFT parquet (messages schema, drops into the v-next mix)
    from datasets import Dataset
    sft = Dataset.from_list([{"messages": r["messages"], "lang": r["lang"]} for r in rows])
    sft.to_parquet(str(out / "train.parquet"))

    # GRPO jsonl: verifiable single-call rows from ANY source. We scan each conversation for the
    # first assistant turn that emits exactly ONE <tool_call>, and use everything before it as the
    # prompt and that call as the gold. Works for xLAM (single-call), ToolACE and Hermes alike.
    grpo_dir = out / "grpo"
    grpo_dir.mkdir(exist_ok=True)
    n_grpo = 0
    with open(grpo_dir / "train.jsonl", "w") as f:
        for r in rows:
            m = r["messages"]
            for i, msg in enumerate(m):
                if msg["role"] != "assistant" or i == 0:
                    continue
                calls = _all_tool_calls(msg["content"])
                # exactly one call, and the preceding turn is a user turn -> clean checkable gold
                if len(calls) != 1 or m[i - 1]["role"] != "user":
                    continue
                gold = [{"name": calls[0].get("name"), "arguments": calls[0].get("arguments", {})}]
                f.write(json.dumps({"prompt": m[:i], "answer": json.dumps(gold, ensure_ascii=False),
                                    "language": r["lang"]}, ensure_ascii=False) + "\n")
                n_grpo += 1
                break  # one GRPO example per conversation

        # ABSTENTION / relevance examples (empty gold) — balances the reward against always-calling
        n_refusal = 0
        try:
            for rr in gen_glaive_refusals(args.grpo_refusal_limit):
                f.write(json.dumps({"prompt": rr["messages"], "answer": "[]",
                                    "language": "en"}, ensure_ascii=False) + "\n")
                n_refusal += 1
        except Exception as e:
            print(f"  refusals: SKIPPED ({type(e).__name__}: {e})")
        print(f"  + {n_refusal} refusal (abstention) GRPO rows")

    print(f"\nWrote {len(rows)} SFT rows -> {out/'train.parquet'}")
    print(f"Wrote {n_grpo} verifiable GRPO rows -> {grpo_dir/'train.jsonl'}")


import re as _re
_TC = _re.compile(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", _re.DOTALL)


def _all_tool_calls(assistant_text: str):
    """All well-formed <tool_call> JSON objects in an assistant turn (any source's format)."""
    out = []
    for m in _TC.finditer(assistant_text or ""):
        try:
            obj = json.loads(m.group(1))
            if isinstance(obj, dict) and "name" in obj:
                out.append(obj)
        except Exception:
            continue
    return out


if __name__ == "__main__":
    main()
