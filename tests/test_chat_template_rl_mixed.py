"""RL mixed-mode template: thinking by default, default mode for rows marked rl_mode == "default"."""
from pathlib import Path

from transformers.utils.chat_template_utils import render_jinja_template

ROOT = Path(__file__).resolve().parents[1] / "templates"


def render(name, messages, **kwargs):
    rendered, _ = render_jinja_template(conversations=[messages], chat_template=(ROOT / name).read_text(encoding="utf-8"),
                                        add_generation_prompt=True, **kwargs)
    return rendered[0]


def test_thinking_by_default_and_default_mode_per_row():
    user = [{"role": "user", "content": "2+2?"}]
    assert render("oellm_qwen3_two_stage_v2_1_rl_mixed.jinja", user).endswith("<|im_start|>assistant\n<think>\n")
    marked = [{"role": "user", "content": "2+2?", "rl_mode": "default"}]
    assert render("oellm_qwen3_two_stage_v2_1_rl_mixed.jinja", marked).endswith("<|im_start|>assistant\n")
    assert "rl_mode" not in render("oellm_qwen3_two_stage_v2_1_rl_mixed.jinja", marked)
    none_marked = [{"role": "user", "content": "2+2?", "rl_mode": None}]   # parquet fills missing struct fields
    assert render("oellm_qwen3_two_stage_v2_1_rl_mixed.jinja", none_marked).endswith("<think>\n")
    assert render("oellm_qwen3_two_stage_v2_1_rl_mixed.jinja", user, enable_thinking=False).endswith("assistant\n")


def test_prompt_rendering_matches_rl_think_variant_for_unmarked_rows():
    convo = [{"role": "system", "content": "Be brief."}, {"role": "user", "content": "Hej"}]
    assert render("oellm_qwen3_two_stage_v2_1_rl_mixed.jinja", convo) == render(
        "oellm_qwen3_two_stage_v2_1_rl_think.jinja", convo)
