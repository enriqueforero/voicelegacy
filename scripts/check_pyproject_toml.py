#!/usr/bin/env python
"""Pre-commit hook: validate that pyproject.toml parses.

Exists because the original A01 audit caught `where = [".]` (unclosed string)
shipping to main; tomllib only reports such breakage at the line/column level,
which is invisible by eye. Running this on every commit guarantees the file
will not ship invalid again.
"""

from __future__ import annotations

import sys

import tomllib


def main(argv: list[str]) -> int:
    if len(argv) < 2:
        print("usage: check_pyproject_toml.py <path-to-pyproject.toml> [...]", file=sys.stderr)
        return 2
    rc = 0
    for path in argv[1:]:
        try:
            with open(path, "rb") as f:
                tomllib.load(f)
            print(f"OK {path}")
        except (OSError, tomllib.TOMLDecodeError) as exc:
            print(f"FAIL {path}: {exc}", file=sys.stderr)
            rc = 1
    return rc


if __name__ == "__main__":
    sys.exit(main(sys.argv))
