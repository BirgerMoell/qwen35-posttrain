#!/usr/bin/env python3
"""Wrapper to run EuroEval in a container that ships flash_attn.

EuroEval hard-refuses to start if `flash_attn` is installed ("uninstall it..."), but we can't
uninstall it from the read-only container. This patches `importlib.metadata` + `find_spec` to
hide flash_attn *before* EuroEval loads, then hands off to EuroEval's CLI. EuroEval uses the
transformers backend with sdpa/eager attention, so hiding flash_attn is harmless for eval.

Usage: python scripts/run_euroeval.py --model <path> --language sv --model-impl transformers ...
"""
import importlib.metadata as _md
import importlib.util as _iu
import sys

_orig_version = _md.version
_orig_find_spec = _iu.find_spec


def _version(name, *a, **k):
    if name == "flash_attn":
        raise _md.PackageNotFoundError(name)
    return _orig_version(name, *a, **k)


def _find_spec(name, *a, **k):
    if name == "flash_attn" or name.startswith("flash_attn."):
        return None
    return _orig_find_spec(name, *a, **k)


_md.version = _version
_iu.find_spec = _find_spec
# also block a direct import, in case EuroEval try/excepts it
sys.modules["flash_attn"] = None  # type: ignore

# hand off to EuroEval's CLI (console-script entry point euroeval.cli:benchmark)
sys.argv = ["euroeval"] + sys.argv[1:]
from euroeval.cli import benchmark  # noqa: E402
benchmark()
