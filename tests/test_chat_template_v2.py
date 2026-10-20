"""Checks for the v2 think-mode switch: Stage-1 rendering must equal v1 exactly."""

import unittest
from pathlib import Path

from transformers.utils.chat_template_utils import render_jinja_template

TEMPLATES = Path(__file__).resolve().parents[1] / "templates"
V1 = (TEMPLATES / "oellm_qwen3_no_empty_assistant_mask.jinja").read_text(encoding="utf-8")
V2 = (TEMPLATES / "oellm_qwen3_two_stage_v2.jinja").read_text(encoding="utf-8")

USER = {"role": "user", "content": "What is two plus two?"}
DIRECT_ROWS = [
    [USER, {"role": "assistant", "content": "4"}],
    [{"role": "system", "content": "Svara kort."}, USER, {"role": "assistant", "content": "Fyra."},
     {"role": "user", "content": "Och tre plus tre?"}, {"role": "assistant", "content": "Sex."}],
]
REASONING_ROW = [USER, {"role": "assistant", "reasoning_content": "2 + 2 = 4", "content": "4"}]


def render(template, messages, **kwargs):
    rendered, indices = render_jinja_template(
        conversations=[messages], chat_template=template, return_assistant_tokens_mask=True, **kwargs,
    )
    text = rendered[0]
    return text, "".join(text[start:end] for start, end in indices[0])


class TemplateV2Tests(unittest.TestCase):
    def test_direct_rows_identical_to_v1(self):
        for messages in DIRECT_ROWS:
            self.assertEqual(render(V1, messages), render(V2, messages))

    def test_default_generation_prompt_identical_to_v1(self):
        for kwargs in ({}, {"enable_thinking": False}):
            text, _ = render(V2, [USER], add_generation_prompt=True, **kwargs)
            self.assertEqual(text, render(V1, [USER], add_generation_prompt=True)[0])
            self.assertNotIn("<think>", text)

    def test_thinking_generation_prompt_prefills_open_tag(self):
        text, _ = render(V2, [USER], add_generation_prompt=True, enable_thinking=True)
        self.assertTrue(text.endswith("<|im_start|>assistant\n<think>\n"))

    def test_reasoning_row_open_tag_is_prompt_side(self):
        text, labels = render(V2, REASONING_ROW)
        self.assertIn("<|im_start|>assistant\n<think>\n2 + 2 = 4\n</think>\n\n4<|im_end|>", text)
        self.assertEqual(labels, "2 + 2 = 4\n</think>\n\n4<|im_end|>\n")
        self.assertNotIn("<think>", labels)

    def test_inline_think_tags_rejected(self):
        with self.assertRaisesRegex(Exception, "reasoning_content"):
            render(V2, [USER, {"role": "assistant", "content": "<think>\nx\n</think>\n\n4"}])

    def test_empty_reasoning_renders_as_direct(self):
        messages = [USER, {"role": "assistant", "reasoning_content": "  ", "content": "4"}]
        self.assertEqual(render(V2, messages), render(V2, DIRECT_ROWS[0]))


if __name__ == "__main__":
    unittest.main()
