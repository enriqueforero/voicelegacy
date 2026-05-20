#!/usr/bin/env python
"""Pre-commit hook: validate Jupyter notebooks against the nbformat schema.

A notebook with broken JSON or invalid schema fails silently on `git pull` and
only shows up when someone opens it. This runs nbformat.validate() on every
commit, which is exactly the check nbformat uses internally.
"""

from __future__ import annotations

import sys

import nbformat


def main(argv: list[str]) -> int:
    if len(argv) < 2:
        print("usage: check_notebook_schema.py <notebook.ipynb> [...]", file=sys.stderr)
        return 2
    rc = 0
    for path in argv[1:]:
        try:
            nb = nbformat.read(path, as_version=4)
            nbformat.validate(nb)
            print(f"OK {path}  ({len(nb.cells)} cells)")
        except Exception as exc:
            print(f"FAIL {path}: {exc}", file=sys.stderr)
            rc = 1
    return rc


if __name__ == "__main__":
    sys.exit(main(sys.argv))
