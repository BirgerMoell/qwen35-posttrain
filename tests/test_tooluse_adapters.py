"""Tool-use adapters: unit cases plus real fixture rows rendered end-to-end with template v2.1."""

import json
import re
import sys
import unittest
from pathlib import Path

from transformers.utils.chat_template_utils import render_jinja_template

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from tooluse_adapters import (ADAPTERS, _parse_pythonic_calls, assign_mode, categorize,  # noqa: E402
                              normalize_tool, split_inline_think, validate)

V21 = (ROOT / "templates" / "oellm_qwen3_two_stage_v2_1.jinja").read_text(encoding="utf-8")
FIXTURES = ROOT / "tests" / "fixtures" / "tooluse"
SOURCES = {"eu_tooluse": "eu_tooluse", "when2call_sft": "when2call", "nemotron_agentic": "nemotron_agentic",
           "toucan": "toucan", "toolmind": "toolmind", "toolace": "toolace"}


def tool(name, props, required=()):
    return {"type": "function", "function": {"name": name, "description": "",
            "parameters": {"type": "object", "properties": props, "required": list(required)}}}


def call(name, **args):
    return {"type": "function", "function": {"name": name, "arguments": args}}


class Units(unittest.TestCase):
    def test_loose_types_normalized(self):
        t = normalize_tool({"name": "f", "parameters": {"type": "dict", "properties": {
            "a": {"type": "str"}, "b": {"type": "int, optional"}, "c": {"type": "List[float]"}}, "required": ["a", "b"]}})
        p = t["function"]["parameters"]
        self.assertEqual(p["type"], "object")
        self.assertEqual([p["properties"][k]["type"] for k in "abc"], ["string", "integer", "array"])
        self.assertEqual(p["properties"]["c"]["items"], {"type": "number"})
        self.assertEqual(p["required"], ["a"])  # "optional" wins over the source's required list

    def test_flat_schema_under_arguments(self):
        t = normalize_tool({"type": "function", "function": {"name": "g", "arguments": {
            "category": {"type": "str", "default": ""}, "limit": {"type": "int", "default": ""}}}})
        p = t["function"]["parameters"]
        self.assertEqual(sorted(p["properties"]), ["category", "limit"])
        self.assertNotIn("required", p)  # defaults make them optional

    def test_pythonic_calls(self):
        calls = _parse_pythonic_calls('[Market Trends API(trend_type="MARKET_INDEXES", country="us"), f(x=[1, 2], y="a)b")]')
        self.assertEqual([c["function"]["name"] for c in calls], ["Market Trends API", "f"])
        self.assertEqual(calls[1]["function"]["arguments"], {"x": [1, 2], "y": "a)b"})

    def test_inline_think(self):
        self.assertEqual(split_inline_think("<think>\nplan\n</think>\n\nanswer"), ("plan", "answer"))
        self.assertEqual(split_inline_think("plain"), ("", "plain"))
        with self.assertRaises(ValueError):
            split_inline_think("<think>unclosed")

    def test_validate_rejections(self):
        tools = [tool("get_weather", {"city": {"type": "string"}}, ["city"])]
        user = {"role": "user", "content": "Weather?"}
        ok = {"tools": tools, "messages": [user, {"role": "assistant", "content": "", "tool_calls": [call("get_weather", city="Rome")]},
                                          {"role": "tool", "content": "21"}, {"role": "assistant", "content": "21 °C"}]}
        validate(ok)
        validate({"tools": tools, "messages": ok["messages"][:2]})  # ends on a call: allowed
        bad = [
            [user, {"role": "assistant", "content": "", "tool_calls": [call("other", city="Rome")]}],
            [user, {"role": "assistant", "content": "", "tool_calls": [call("get_weather")]}],
            [user, {"role": "assistant", "content": "", "tool_calls": [call("get_weather", city=3)]}],
            [user, {"role": "tool", "content": "21"}, {"role": "assistant", "content": "x"}],
            [user, {"role": "assistant", "content": "", "tool_calls": [call("get_weather", city="Rome")]},
             {"role": "assistant", "content": "x"}],
        ]
        for messages in bad:
            with self.assertRaises(ValueError):
                validate({"tools": tools, "messages": messages})

    def test_missing_parameter_is_ask_for_info(self):
        tools = [tool("get_intervals", {"interval_type": {"type": "string"}}, ["interval_type"])]
        for answer in ["The query is missing the `interval_type` required to call the function.",
                       "To proceed, I'll need the CNPJ number for the establishment."]:
            row = {"tools": tools, "messages": [{"role": "user", "content": "Find intervals."},
                                                {"role": "assistant", "content": answer}]}
            self.assertEqual(categorize(row), "ask_for_info")
        row = {"tools": tools, "messages": [{"role": "user", "content": "Order a pizza."},
                                            {"role": "assistant", "content": "I'm unable to order food with these tools."}]}
        self.assertEqual(categorize(row), "cannot_answer")

    def test_mode_assignment(self):
        row = {"messages": [{"role": "user", "content": "a"}, {"role": "assistant", "content": "b", "reasoning_content": "old"},
                            {"role": "user", "content": "c"}, {"role": "assistant", "content": "", "reasoning_content": "t1",
                                                               "tool_calls": [call("f")]},
                            {"role": "tool", "content": "r"}, {"role": "assistant", "content": "d"}]}
        self.assertEqual(assign_mode(row), "direct")  # last step has no trace -> strip all
        self.assertFalse(any("reasoning_content" in m for m in row["messages"]))


def supervised(messages, tools):
    rendered, idx = render_jinja_template(conversations=[messages], tools=tools or None, chat_template=V21,
                                          return_assistant_tokens_mask=True)
    text = rendered[0]
    return text, "".join(text[a:b] for a, b in idx[0])


class Fixtures(unittest.TestCase):
    def test_sources_end_to_end(self):
        for fname, key in SOURCES.items():
            rows = [json.loads(l) for l in (FIXTURES / f"{fname}.jsonl").read_text().splitlines()]
            valid = 0
            for raw in rows:
                try:
                    row = ADAPTERS[key](raw)
                    validate(row)
                except ValueError:
                    continue
                valid += 1
                categorize(row)
                mode = assign_mode(row)
                text, labels = supervised(row["messages"], row["tools"])
                stored = [c["function"] for m in row["messages"] if m["role"] == "assistant" for c in m.get("tool_calls", [])]
                parsed = [json.loads(b) for b in re.findall(r"<tool_call>\n(.*?)\n</tool_call>", labels, re.S)]
                with self.subTest(source=fname, row=valid):
                    self.assertEqual(parsed, [{"name": s["name"], "arguments": s["arguments"]} for s in stored])
                    self.assertNotIn("<tool_response>", labels)
                    self.assertNotIn("<think>", labels)
                    self.assertNotIn("<tools>", labels)
                    if mode == "reasoning":
                        self.assertIn("</think>", labels)
            with self.subTest(source=fname):
                self.assertGreaterEqual(valid / len(rows), 0.8, f"{fname}: only {valid}/{len(rows)} valid")


if __name__ == "__main__":
    unittest.main()
