"""Guards of the pre-tokenized DPO entrypoint (no TRL needed; torch only for the collator guard)."""
import importlib.util
import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

spec = importlib.util.spec_from_file_location(
    "dpo_train_pretok", Path(__file__).resolve().parents[1] / "scripts" / "dpo_train_pretok.py")
dpo = importlib.util.module_from_spec(spec)
spec.loader.exec_module(dpo)

EOS = 4
FORBIDDEN = dpo.parse_ids(dpo.DEFAULT_FORBIDDEN)


def shard(tmp_path, rows, name="train-0000.parquet"):
    table = pa.table({k: pa.array([r[i] for r in rows], type=pa.list_(pa.int32()))
                      for i, k in enumerate(dpo.COLUMNS)})
    path = tmp_path / name
    pq.write_table(table, path)
    return str(path)


GOOD = [([3, 100, 101, 4, 3, 200], [300, 301, 4, 10], [302, 4, 10]),
        ([3, 100, 4, 3, 200], [303, 4, 10], [304, 305, 306, 4, 10])]


def test_parse_ids():
    assert FORBIDDEN == list(range(11, 19))
    assert dpo.parse_ids("18:15, 13") == [13, 15, 18]


def test_scan_accepts_clean_shard(tmp_path):
    assert dpo.scan_shard(shard(tmp_path, GOOD), FORBIDDEN, EOS, 2048) == 2


@pytest.mark.parametrize("bad, message", [
    (([3, 100, 4], [300, 12, 4], [302, 4]), "forbidden"),       # </tool_call> in chosen
    (([3, 17, 4], [300, 4], [302, 4]), "forbidden"),            # </think> in the prompt
    (([3, 100, 4], [], [302, 4]), "empty"),
    (([3, 100, 4], [300, 301], [302, 4]), "without <|im_end|>"),
    (([3, 100, 4], [300, 4], [302, 303]), "without <|im_end|>"),
])
def test_scan_rejects(tmp_path, bad, message):
    with pytest.raises(ValueError, match=message.replace("|", r"\|")):
        dpo.scan_shard(shard(tmp_path, GOOD + [bad]), FORBIDDEN, EOS, 2048)


def test_scan_rejects_over_length(tmp_path):
    with pytest.raises(ValueError, match="longer than 8"):
        dpo.scan_shard(shard(tmp_path, GOOD), FORBIDDEN, EOS, 8)


def test_manifest_checks(tmp_path):
    template = tmp_path / "t.jinja"
    template.write_text("{{ messages }}")
    tok = tmp_path / "tokenizer.json"
    tok.write_text("{}")
    manifest = {"kind": "oellm_dpo_v1_pretokenized", "max_length": 2048,
                "template_sha256": dpo.sha256(template), "tokenizer_json_sha256": dpo.sha256(tok)}
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(manifest))
    assert dpo.check_manifest(str(path), str(template), str(tok), 2048)["kind"] == manifest["kind"]
    with pytest.raises(ValueError, match="max_length"):
        dpo.check_manifest(str(path), str(template), str(tok), 1024)
    template.write_text("changed")
    with pytest.raises(ValueError, match="template"):
        dpo.check_manifest(str(path), str(template), str(tok), 2048)
    path.write_text(json.dumps({**manifest, "kind": "something_else"}))
    with pytest.raises(ValueError, match="kind"):
        dpo.check_manifest(str(path), str(template), str(tok), 2048)


def test_collator_guard():
    torch = pytest.importorskip("torch")

    def inner(features):
        ids = torch.tensor(features)
        return {"input_ids": ids, "attention_mask": (ids != 0).long()}

    guard = dpo.ForbiddenIdGuard(inner, FORBIDDEN)
    assert guard([[3, 5, 4], [3, 6, 0]])["input_ids"].shape == (2, 3)
    with pytest.raises(RuntimeError, match="forbidden"):
        guard([[3, 17, 4]])
    guard_pad = dpo.ForbiddenIdGuard(inner, [0])   # padded positions are ignored
    guard_pad([[3, 5, 0]])
