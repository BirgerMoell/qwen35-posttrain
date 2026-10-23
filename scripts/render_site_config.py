#!/usr/bin/env python3
"""Print a YAML config with every ${NAME} replaced by the environment value (site settings, leonardo.env).

Tracked configs keep site-specific paths as ${OELLM_WORK} / ${OELLM_SCRATCH} references; the Slurm launchers render
them per job. An unset variable is an error. Python 3.6 compatible (Leonardo login/compute host python).
"""
import os
import re
import sys

REFERENCE = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)\}")


def render(text):
    missing = sorted({name for name in REFERENCE.findall(text) if name not in os.environ})
    if missing:
        raise SystemExit("unset site variables %s (source leonardo.env)" % ", ".join(missing))
    return REFERENCE.sub(lambda match: os.environ[match.group(1)], text)


if __name__ == "__main__":
    with open(sys.argv[1], encoding="utf-8") as handle:
        sys.stdout.write(render(handle.read()))
