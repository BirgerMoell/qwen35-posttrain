"""Checks for template v2.1: tool-free rendering equals v2; tools follow the Qwen3/Hermes format."""

import json
import unittest
from pathlib import Path

from transformers.utils.chat_template_utils import render_jinja_template

TEMPLATES = Path(__file__).resolve().parents[1] / "templates"
V2 = (TEMPLATES / "oellm_qwen3_two_stage_v2.jinja").read_text(encoding="utf-8")
V21 = (TEMPLATES / "oellm_qwen3_two_stage_v2_1.jinja").read_text(encoding="utf-8")

USER = {"role": "user", "content": "What is two plus two?"}
TOOL_FREE = [
    [USER, {"role": "assistant", "content": "4"}],
    [{"role": "system", "content": "Svara kort."}, USER, {"role": "assistant", "content": "Fyra."},
     {"role": "user", "content": "Och tre plus tre?"}, {"role": "assistant", "content": "Sex."}],
    [USER, {"role": "assistant", "reasoning_content": "2 + 2 = 4", "content": "4"}],
    [{"role": "system", "content": "Be brief."}, USER, {"role": "assistant", "content": "4"},
     {"role": "user", "content": "And 3+3?"}, {"role": "assistant", "reasoning_content": "3 + 3 = 6", "content": "6"}],
]
WEATHER = {"type": "function", "function": {"name": "get_weather", "description": "Current weather for a city.",
           "parameters": {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]}}}
CALL = {"type": "function", "function": {"name": "get_weather", "arguments": {"city": "Stockholm"}}}
LOOP = [
    {"role": "user", "content": "Hur varmt är det i Stockholm och Oslo?"},
    {"role": "assistant", "reasoning_content": "Need both cities.", "content": "",
     "tool_calls": [CALL, {"type": "function", "function": {"name": "get_weather", "arguments": {"city": "Oslo"}}}]},
    {"role": "tool", "content": "{\"temp_c\": 12}"},
    {"role": "tool", "content": "{\"temp_c\": 9}"},
    {"role": "assistant", "reasoning_content": "Stockholm 12, Oslo 9.", "content": "Stockholm: 12 °C, Oslo: 9 °C."},
]


def render(template, messages, **kwargs):
    rendered, indices = render_jinja_template(
        conversations=[messages], chat_template=template, return_assistant_tokens_mask=True, **kwargs)
    text = rendered[0]
    return text, "".join(text[a:b] for a, b in indices[0])


class ToolFreeIdentity(unittest.TestCase):
    def test_rows_identical_to_v2(self):
        for messages in TOOL_FREE:
            self.assertEqual(render(V2, messages), render(V21, messages))

    def test_generation_prompts_identical_to_v2(self):
        for kwargs in ({}, {"enable_thinking": False}, {"enable_thinking": True}):
            self.assertEqual(render(V2, [USER], add_generation_prompt=True, **kwargs),
                             render(V21, [USER], add_generation_prompt=True, **kwargs))

    def test_inline_think_still_rejected(self):
        with self.assertRaisesRegex(Exception, "reasoning_content"):
            render(V21, [USER, {"role": "assistant", "content": "<think>x</think>4"}])


class ToolRendering(unittest.TestCase):
    def test_system_tools_block_is_qwen3_hermes(self):
        text, labels = render(V21, [USER, {"role": "assistant", "content": "4"}], tools=[WEATHER])
        self.assertTrue(text.startswith("<|im_start|>system\n# Tools\n\nYou may call one or more functions"))
        self.assertIn("<tools>\n" + json.dumps(WEATHER, ensure_ascii=False) + "\n</tools>", text)
        self.assertIn("<tool_call>\n{\"name\": <function-name>, \"arguments\": <args-json-object>}\n</tool_call><|im_end|>\n", text)
        self.assertNotIn("<tools>", labels)

    def test_system_prompt_merged_before_tools(self):
        text, _ = render(V21, [{"role": "system", "content": "Du är hjälpsam."}, USER,
                               {"role": "assistant", "content": "4"}], tools=[WEATHER])
        self.assertTrue(text.startswith("<|im_start|>system\nDu är hjälpsam.\n\n# Tools\n"))
        self.assertEqual(text.count("<|im_start|>system"), 1)

    def test_tool_loop_masks_and_format(self):
        text, labels = render(V21, LOOP, tools=[WEATHER])
        first = ('<tool_call>\n{"name": "get_weather", "arguments": {"city": "Stockholm"}}\n</tool_call>\n'
                 '<tool_call>\n{"name": "get_weather", "arguments": {"city": "Oslo"}}\n</tool_call><|im_end|>\n')
        self.assertIn("<|im_start|>assistant\n<think>\nNeed both cities.\n</think>\n\n" + first, text)
        self.assertIn("<|im_start|>user\n<tool_response>\n{\"temp_c\": 12}\n</tool_response>\n"
                      "<tool_response>\n{\"temp_c\": 9}\n</tool_response><|im_end|>\n", text)
        # Supervised: both reasoning spans, both calls, the final answer. Never: <think>, tool responses, tools.
        self.assertIn("Need both cities.\n</think>\n\n" + first, labels)
        self.assertIn("Stockholm 12, Oslo 9.\n</think>\n\nStockholm: 12 °C, Oslo: 9 °C.<|im_end|>\n", labels)
        for forbidden in ("<think>", "<tool_response>", "temp_c", "<tools>"):
            self.assertNotIn(forbidden, labels)

    def test_reasoning_before_last_user_dropped(self):
        messages = [USER, {"role": "assistant", "reasoning_content": "old trace", "content": "4"},
                    {"role": "user", "content": "Thanks!"}, {"role": "assistant", "content": "You're welcome."}]
        text, _ = render(V21, messages)
        self.assertNotIn("old trace", text)

    def test_string_arguments_and_content_before_call(self):
        msg = {"role": "assistant", "content": "Let me check.",
               "tool_calls": [{"type": "function", "function": {"name": "get_weather", "arguments": "{\"city\": \"Rome\"}"}}]}
        text, labels = render(V21, [USER, msg], tools=[WEATHER])
        self.assertIn('Let me check.\n<tool_call>\n{"name": "get_weather", "arguments": {"city": "Rome"}}\n</tool_call><|im_end|>', labels)

    def test_null_content_with_tool_calls(self):
        text, labels = render(V21, [USER, {"role": "assistant", "content": None, "tool_calls": [CALL]}], tools=[WEATHER])
        self.assertTrue(labels.startswith("<tool_call>\n"))

    def test_thinking_generation_prompt_after_tool_result(self):
        text, _ = render(V21, LOOP[:4], tools=[WEATHER], add_generation_prompt=True, enable_thinking=True)
        self.assertTrue(text.endswith("</tool_response><|im_end|>\n<|im_start|>assistant\n<think>\n"))

    def test_unknown_role_rejected(self):
        with self.assertRaisesRegex(Exception, "Unsupported role"):
            render(V21, [USER, {"role": "function", "content": "x"}])


if __name__ == "__main__":
    unittest.main()
