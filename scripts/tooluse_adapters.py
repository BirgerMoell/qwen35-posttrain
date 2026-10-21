#!/usr/bin/env python3
"""Adapters from tool-use source datasets to one canonical schema, plus validation.

Canonical row: {"tools": [{"type": "function", "function": {name, description, parameters}}],
                "messages": [...]}
with roles system/user/assistant/tool. Assistant turns may carry `tool_calls`
([{"type": "function", "function": {"name", "arguments": dict}}]) and `reasoning_content`;
tool turns carry the tool output as text. This is the HF/OpenAI structure that template v2.1
renders. Every adapter raises ValueError for rows it cannot convert faithfully.
"""
from __future__ import annotations

import ast
import json
import re

# ---------------------------------------------------------------- tool schemas

_TYPE_MAP = {"str": "string", "string": "string", "int": "integer", "integer": "integer", "float": "number",
             "number": "number", "bool": "boolean", "boolean": "boolean", "dict": "object", "object": "object",
             "list": "array", "array": "array", "tuple": "array", "set": "array", "null": "null", "none": "null"}


def _norm_type(raw) -> tuple[dict, bool]:
    """Map loose type strings ("str, optional", "List[int]", "dict") to JSON Schema; return (schema, optional)."""
    if isinstance(raw, list):
        return {"type": [(_norm_type(t)[0].get("type") or "string") for t in raw]}, False
    if not isinstance(raw, str):
        return {}, False
    text = raw.strip()
    optional = bool(re.search(r",\s*optional\s*$", text, re.I)) or text.lower().startswith("optional[")
    text = re.sub(r",\s*optional\s*$", "", text, flags=re.I)
    text = re.sub(r"^optional\[(.*)\]$", r"\1", text, flags=re.I)
    base = re.split(r"[\[\s|]", text, maxsplit=1)[0].lower()
    if base in ("list", "array", "tuple", "set", "sequence"):
        inner = re.search(r"\[(.*)\]", text)
        item = _norm_type(inner.group(1))[0] if inner else {}
        return ({"type": "array", "items": item} if item else {"type": "array"}), optional
    if base in ("any", ""):
        return {}, optional
    return {"type": _TYPE_MAP.get(base, "string")}, optional


def normalize_parameters(params) -> dict:
    if not isinstance(params, dict):
        return {"type": "object", "properties": {}}
    props_in = params.get("properties") or {}
    props, optional_names = {}, set()
    for name, spec in props_in.items():
        spec = dict(spec) if isinstance(spec, dict) else {}
        schema, optional = _norm_type(spec.get("type"))
        out = {k: v for k, v in spec.items() if k not in ("type",)}
        out.update(schema)
        if out.get("type") == "object" and "properties" in out:
            out = {**out, **normalize_parameters(out)}
        if optional:
            optional_names.add(name)
        props[name] = out
    required = params.get("required")
    required = [r for r in required if r in props and r not in optional_names] if isinstance(required, list) else \
        [n for n, s in props_in.items() if n not in optional_names and isinstance(s, dict) and "default" not in s]
    out = {"type": "object", "properties": props}
    if required:
        out["required"] = required
    return out


def normalize_tool(tool) -> dict:
    if isinstance(tool, str):
        tool = json.loads(tool)
    fn = tool.get("function", tool) if isinstance(tool, dict) else None
    if not isinstance(fn, dict) or not isinstance(fn.get("name"), str) or not fn["name"].strip():
        raise ValueError("tool without a name")
    params = fn.get("parameters")
    if params is None and isinstance(fn.get("arguments"), dict):
        params = fn["arguments"]  # some sources (ToolMind/When2Call) use "arguments" for the schema
    if isinstance(params, dict) and "properties" not in params and "type" not in params and \
            all(isinstance(v, dict) for v in params.values()):
        params = {"type": "object", "properties": params}  # flat property map
    out = {"name": fn["name"].strip(), "description": str(fn.get("description") or "").strip(),
           "parameters": normalize_parameters(params)}
    return {"type": "function", "function": out}


def parse_arguments(args) -> dict:
    if isinstance(args, dict):
        return args
    if isinstance(args, str):
        args = args.strip() or "{}"
        value = json.loads(args)
        if isinstance(value, dict):
            return value
    raise ValueError("tool-call arguments are not a JSON object")


def make_call(name: str, arguments) -> dict:
    return {"type": "function", "function": {"name": name, "arguments": parse_arguments(arguments)}}


def split_inline_think(content: str) -> tuple[str, str]:
    """'<think>trace</think>answer' -> (trace, answer); content without tags -> ('', content)."""
    if "<think>" not in content and "</think>" not in content:
        return "", content
    m = re.fullmatch(r"\s*<think>(.*?)</think>(.*)", content, re.S)
    if not m or "<think>" in m.group(2) or "</think>" in m.group(2):
        raise ValueError("malformed inline think block")
    return m.group(1).strip(), m.group(2).strip()


def _assistant(content: str = "", calls=None, reasoning: str = "") -> dict:
    msg = {"role": "assistant", "content": content.strip()}
    if calls:
        msg["tool_calls"] = calls
    if reasoning.strip():
        msg["reasoning_content"] = reasoning.strip()
    return msg


# ---------------------------------------------------------------- adapters

_TOOLS_BLOCK = re.compile(r"<tools>\s*(.*?)\s*</tools>", re.S)
_CALL_BLOCK = re.compile(r"<tool_call>\s*(.*?)\s*</tool_call>", re.S)


def adapt_eu_tooluse(row: dict) -> dict:
    """birgermoell/oellm-eu-tooluse-v1: Hermes text (tools in system text, <tool_call> blocks in content)."""
    msgs = row["messages"]
    if not msgs or msgs[0]["role"] != "system":
        raise ValueError("expected a system turn with tools")
    system = msgs[0]["content"]
    # The instruction text itself mentions "<tools></tools>"; the signatures are the last non-empty block.
    blocks = [b for b in _TOOLS_BLOCK.findall(system) if b.strip()]
    if not blocks:
        raise ValueError("no <tools> block")
    tools = []
    for line in blocks[-1].splitlines():
        if line.strip():
            value = json.loads(line)  # one tool per line, or the whole list on one line
            tools.extend(normalize_tool(t) for t in (value if isinstance(value, list) else [value]))
    preamble = system.split("# Tools", 1)[0].strip()
    out = [{"role": "system", "content": preamble}] if preamble else []
    for m in msgs[1:]:
        if m["role"] == "assistant":
            reasoning, text = split_inline_think(m["content"])
            calls = []
            for raw in _CALL_BLOCK.findall(text):
                obj = json.loads(raw)
                calls.append(make_call(obj["name"], obj.get("arguments", {})))
            text = _CALL_BLOCK.sub("", text).strip()
            out.append(_assistant(text, calls, reasoning))
        elif m["role"] in ("user", "tool"):
            out.append({"role": m["role"], "content": m["content"]})
        else:
            raise ValueError(f"unexpected role {m['role']}")
    return {"tools": tools, "messages": out}


def adapt_when2call(row: dict) -> dict:
    """nvidia/When2Call SFT: tools as JSON strings; final assistant turn is a non-call text answer."""
    tools = [normalize_tool(t) for t in row.get("tools") or []]
    msgs = [{"role": m["role"], "content": m["content"]} for m in row["messages"]]
    if not msgs or msgs[-1]["role"] != "assistant":
        raise ValueError("expected a final assistant turn")
    return {"tools": tools, "messages": msgs}


def _openai_messages(msgs: list[dict]) -> list[dict]:
    out = []
    for m in msgs:
        role = m["role"]
        if role == "system":
            if (m.get("content") or "").strip():
                out.append({"role": "system", "content": m["content"]})
        elif role == "user":
            out.append({"role": "user", "content": m["content"]})
        elif role == "assistant":
            reasoning = m.get("reasoning_content") or ""
            inline, text = split_inline_think(m.get("content") or "")
            if inline and reasoning:
                raise ValueError("reasoning in both fields")
            calls = [make_call(c["function"]["name"], c["function"].get("arguments", {}))
                     for c in (m.get("tool_calls") or [])]
            out.append(_assistant(text, calls, reasoning or inline))
        elif role == "tool":
            content = m.get("content")
            out.append({"role": "tool", "content": content if isinstance(content, str) else json.dumps(content, ensure_ascii=False)})
        else:
            raise ValueError(f"unexpected role {role}")
    return out


def adapt_nemotron_agentic(row: dict) -> dict:
    """nvidia/Nemotron-SFT-Agentic-v2: OpenAI tool_calls (string arguments) + reasoning_content per step."""
    if row.get("filter_reason"):
        raise ValueError(f"source-filtered: {row['filter_reason']}")
    return {"tools": [normalize_tool(t) for t in row["tools"]], "messages": _openai_messages(row["messages"])}


def adapt_toolmind(row: dict) -> dict:
    """Nanbeige/ToolMind: `conversations` with tool_calls (dict arguments), inline <think> in content."""
    return {"tools": [normalize_tool(t) for t in row["tools"]], "messages": _openai_messages(row["conversations"])}


def adapt_toucan(row: dict) -> dict:
    """Agent-Ark/Toucan-1.5M: separate `tool_call` (Python-repr dict) and `tool_response` messages."""
    tools = [normalize_tool(t) for t in json.loads(row["tools"])]
    msgs = row["messages"] if isinstance(row["messages"], list) else json.loads(row["messages"])
    out: list[dict] = []
    for m in msgs:
        role = m["role"]
        if role == "tool_call":
            try:
                obj = ast.literal_eval(m["content"]) if m["content"].lstrip().startswith("{'") else json.loads(m["content"])
            except (ValueError, SyntaxError) as exc:
                raise ValueError("unparseable tool_call") from exc
            call = make_call(obj["name"], obj.get("arguments", {}))
            if out and out[-1]["role"] == "assistant":  # calls belong to the preceding assistant turn
                out[-1].setdefault("tool_calls", []).append(call)
            else:
                out.append(_assistant("", [call]))
        elif role == "tool_response":
            out.append({"role": "tool", "content": m["content"]})
        elif role == "assistant":
            if out and out[-1]["role"] == "assistant" and not out[-1].get("tool_calls"):
                raise ValueError("consecutive assistant turns")
            reasoning, text = split_inline_think(m.get("content") or "")
            out.append(_assistant(text, None, reasoning))
        elif role in ("user", "system"):
            if role == "system" and not (m.get("content") or "").strip():
                continue
            out.append({"role": role, "content": m["content"]})
        else:
            raise ValueError(f"unexpected role {role}")
    return {"tools": tools, "messages": out}


_TOOLACE_TOOLS = re.compile(r"Here is a list of functions in JSON format that you can invoke:\s*(\[.*\])\.\s*\n", re.S)


def _parse_pythonic_calls(text: str) -> list[dict]:
    """'[Func A(x="1", y=2), other(z=[1])]' -> calls. Tool names may contain spaces."""
    text = text.strip()
    if not (text.startswith("[") and text.endswith("]")):
        raise ValueError("not a pythonic call list")
    calls, body, i = [], text[1:-1], 0
    while i < len(body):
        m = re.match(r"\s*,?\s*([^(\[\]]+?)\s*\(", body[i:])
        if not m:
            if body[i:].strip(" ,"):
                raise ValueError("trailing text in call list")
            break
        name, start = m.group(1).strip(), i + m.end()
        depth, j, quote = 1, start, None
        while j < len(body) and depth:
            ch = body[j]
            if quote:
                if ch == "\\":
                    j += 1
                elif ch == quote:
                    quote = None
            elif ch in "\"'":
                quote = ch
            elif ch == "(":
                depth += 1
            elif ch == ")":
                depth -= 1
            j += 1
        if depth:
            raise ValueError("unbalanced call")
        args_src = body[start:j - 1]
        node = ast.parse(f"f({args_src})", mode="eval").body
        args = {kw.arg: ast.literal_eval(kw.value) for kw in node.keywords}
        if node.args:
            raise ValueError("positional arguments in call")
        calls.append(make_call(name, args))
        i = j
    if not calls:
        raise ValueError("empty call list")
    return calls


def adapt_toolace(row: dict) -> dict:
    """Team-ACE/ToolACE: tools JSON inside the system prose; pythonic calls; from/value turns."""
    m = _TOOLACE_TOOLS.search(row["system"])
    if not m:
        raise ValueError("no tool list in system prompt")
    tools = [normalize_tool(t) for t in json.loads(m.group(1))]
    out = []
    for turn in row["conversations"]:
        role, value = turn["from"], turn["value"]
        if role == "user":
            out.append({"role": "user", "content": value})
        elif role == "assistant":
            stripped = value.strip()
            if stripped.startswith("[") and stripped.endswith("]") and "(" in stripped:
                out.append(_assistant("", _parse_pythonic_calls(stripped)))
            else:
                out.append(_assistant(value))
        elif role == "tool":
            out.append({"role": "tool", "content": value})
        else:
            raise ValueError(f"unexpected role {role}")
    return {"tools": tools, "messages": out}


ADAPTERS = {"eu_tooluse": adapt_eu_tooluse, "when2call": adapt_when2call, "nemotron_agentic": adapt_nemotron_agentic,
            "toolmind": adapt_toolmind, "toucan": adapt_toucan, "toolace": adapt_toolace}

SPECIAL_TEXT = ("<tool_call>", "</tool_call>", "<tools>", "</tools>", "<tool_response>", "</tool_response>",
                "<think>", "</think>", "<|im_start|>", "<|im_end|>")


def trim_trailing_user(row: dict) -> dict:
    """Drop trailing user turns (e.g. a closing "thanks") so the row ends on an assistant turn.
    Trailing tool results are never trimmed: that would orphan the calls they answer."""
    msgs = row["messages"]
    while msgs and msgs[-1]["role"] == "user" and any(m["role"] == "assistant" for m in msgs[:-1]):
        msgs.pop()
    return row


def check_special_text(row: dict) -> None:
    """Literal tag text would be tokenized as our special tag tokens and corrupt the structure."""
    texts = [json.dumps(row["tools"], ensure_ascii=False)]
    for m in row["messages"]:
        texts += [m.get("content") or "", m.get("reasoning_content") or ""]
        texts += [json.dumps(c["function"]["arguments"], ensure_ascii=False) for c in m.get("tool_calls") or []]
    for text in texts:
        for tag in SPECIAL_TEXT:
            if tag in text:
                raise ValueError(f"literal {tag} in content")


# ---------------------------------------------------------------- validation

_JSON_TYPES = {"string": str, "integer": int, "number": (int, float), "boolean": bool, "array": list, "object": dict}


def _type_ok(value, schema: dict) -> bool:
    t = schema.get("type")
    if t is None or isinstance(t, list):
        return True
    py = _JSON_TYPES.get(t)
    if py is None:
        return True
    if t in ("integer", "number") and isinstance(value, bool):
        return False
    if t == "number" and isinstance(value, str):
        return False
    return isinstance(value, py) or (t == "integer" and isinstance(value, float) and value.is_integer())


def validate(row: dict) -> None:
    """Raise ValueError unless every call targets a declared tool with schema-valid arguments and
    the trajectory is well-formed: k calls -> k tool results, and the row ends with an assistant turn
    (either an answer, or calls whose results are not part of the row — the model learns to stop there)."""
    tools = {t["function"]["name"]: t["function"] for t in row["tools"]}
    msgs = row["messages"]
    if not msgs or msgs[-1]["role"] != "assistant":
        raise ValueError("must end with an assistant turn")
    pending = 0
    for i, m in enumerate(msgs):
        role = m["role"]
        if role == "tool":
            if pending <= 0:
                raise ValueError("tool result without a call")
            pending -= 1
            continue
        if pending:
            raise ValueError("missing tool results")
        if role == "assistant":
            calls = m.get("tool_calls") or []
            if not calls and not m["content"].strip():
                raise ValueError("empty assistant turn")
            for c in calls:
                fn = c["function"]
                spec = tools.get(fn["name"])
                if spec is None:
                    raise ValueError(f"call to undeclared tool {fn['name']!r}")
                params = spec["parameters"]
                for req in params.get("required", []):
                    if req not in fn["arguments"]:
                        raise ValueError(f"missing required argument {req!r}")
                for key, value in fn["arguments"].items():
                    prop = params["properties"].get(key)
                    if prop is None:
                        raise ValueError(f"unknown argument {key!r}")
                    if not _type_ok(value, prop):
                        raise ValueError(f"argument {key!r} has wrong type")
            pending = len(calls)
            if i == 0:
                raise ValueError("conversation starts with the assistant")
    if pending and not msgs[-1].get("tool_calls"):
        raise ValueError("trailing calls without results")


# ---------------------------------------------------------------- labels

_ASK = re.compile(r"\?\s*$|\b(could you (please )?(provide|tell|specify|share)|please (provide|specify|let me know)|"
                  r"you need to provide)\b|^\s*(in )?(which|what|who|where|when|how)\b[^.\n]*\?", re.I)
# Missing-parameter explanations ("The query is missing `x`", "I'll need the username") are requests for information.
_MISSING = re.compile(r"\b(is missing|are missing|missing (the|a|an|required)|(not|n't) (been )?provided?|"
                      r"(does not|doesn't|didn't|did not) (provide|contain|include|specify) (the |a |an |any )?(required|necessary|needed)|"
                      r"lacks? the|(I|we)('ll| will| would)? need (the|your|to know)|requires? (both|the|a|an|you to)|"
                      r"insufficient information|not (contain )?sufficient information)\b", re.I)
_CANNOT = re.compile(r"\b(unable to|can(no|')t|cannot|not able to|don't have (access|the ability|the capability)|"
                     r"(don't|do not) have (a|any|the) [^.\n]{0,40}(tool|function|capabilit)|"
                     r"none of the (provided |available |given )?functions|"
                     r"functions? (provided |available |given |listed )?(do|does) not (include|provide|support|cover)|"
                     r"(do|does) not include (a |any )?function|no (available |suitable |relevant )?(tool|function)s?\b|"
                     r"not (relevant|applicable) to (the|any) (available|provided|given) functions)\b", re.I)


_META = re.compile(r"\[No Need to Ask\]|(provides?|provided|have provided) sufficient information to call|"
                   r"Required Arguments:|Provided Arguments:|the query provides sufficient information", re.I)


def check_not_meta(row: dict) -> None:
    """Reject answers that describe a call ("sufficient information to call X") instead of making it."""
    for m in row["messages"]:
        if m["role"] == "assistant" and not m.get("tool_calls") and _META.search(m.get("content") or ""):
            raise ValueError("describes a call instead of making it")


def categorize(row: dict) -> str:
    msgs = row["messages"]
    call_turns = [m for m in msgs if m["role"] == "assistant" and m.get("tool_calls")]
    users_with_calls = 0
    seen_call = False
    for m in msgs:
        if m["role"] == "user":
            seen_call = False
        elif m["role"] == "assistant" and m.get("tool_calls") and not seen_call:
            seen_call = True
            users_with_calls += 1
    if not call_turns:
        final = msgs[-1]["content"]
        if not row["tools"]:
            return "no_tools_direct"
        if _ASK.search(final[-300:]) or _MISSING.search(final[:400]):
            return "ask_for_info"
        if _CANNOT.search(final[:300]):
            return "cannot_answer"
        return "no_call_direct"
    if users_with_calls > 1:
        return "multi_turn"
    if len(call_turns) > 1:
        return "multi_step"
    return "parallel_call" if len(call_turns[0]["tool_calls"]) > 1 else "single_call"


def assign_mode(row: dict) -> str:
    """'reasoning' iff every assistant turn after the last user turn has a trace; else strip all traces."""
    msgs = row["messages"]
    last_user = max(i for i, m in enumerate(msgs) if m["role"] == "user")
    current = [m for m in msgs[last_user + 1:] if m["role"] == "assistant"]
    if current and all((m.get("reasoning_content") or "").strip() for m in current):
        for m in msgs[:last_user]:
            m.pop("reasoning_content", None)
        return "reasoning"
    for m in msgs:
        m.pop("reasoning_content", None)
    return "direct"
