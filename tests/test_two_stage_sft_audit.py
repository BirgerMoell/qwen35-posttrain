"""Unit checks for the fail-closed two-stage SFT source classifier."""

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from audit_two_stage_sft import source_kind  # noqa: E402


USER = {"role": "user", "content": "What is two plus two?"}


class SourceKindTests(unittest.TestCase):
    def test_direct(self):
        self.assertEqual(source_kind([USER, {"role": "assistant", "content": "4"}]), "direct")

    def test_separate_reasoning_field(self):
        self.assertEqual(source_kind([USER, {
            "role": "assistant", "reasoning_content": "2 + 2 = 4", "content": "4",
        }]), "reasoning")

    def test_inline_reasoning(self):
        self.assertEqual(source_kind([USER, {
            "role": "assistant", "content": "<think>\n2 + 2 = 4\n</think>\n\n4",
        }]), "reasoning")

    def test_empty_reasoning_rejected(self):
        with self.assertRaisesRegex(ValueError, "empty source think span"):
            source_kind([USER, {"role": "assistant", "content": "<think>\n\n</think>\n\n4"}])

    def test_unclosed_reasoning_rejected(self):
        with self.assertRaisesRegex(ValueError, "unbalanced"):
            source_kind([USER, {"role": "assistant", "content": "<think>2 + 2 = 4"}])

    def test_duplicate_reasoning_rejected(self):
        with self.assertRaisesRegex(ValueError, "duplicated"):
            source_kind([USER, {
                "role": "assistant", "reasoning_content": "2 + 2 = 4",
                "content": "<think>2 + 2 = 4</think>4",
            }])

    def test_missing_final_answer_rejected(self):
        with self.assertRaisesRegex(ValueError, "no final answer"):
            source_kind([USER, {"role": "assistant", "content": "<think>2 + 2 = 4</think>"}])

    def test_tool_role_rejected(self):
        with self.assertRaisesRegex(ValueError, "nontext roles"):
            source_kind([USER, {"role": "tool", "content": "4"},
                         {"role": "assistant", "content": "4"}])


if __name__ == "__main__":
    unittest.main()
