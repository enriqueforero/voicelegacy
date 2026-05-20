#!/usr/bin/env python
"""Pre-commit hook: refuse notebooks containing executable `runtime.unassign()`.

The original Cell 36 of the legacy notebook had this line UNCOMMENTED, so
every `Run all` killed the Colab session. The fix is to keep the line as a
commented hint only. This hook makes sure it never sneaks back in as live code.
"""

from __future__ import annotations

import json
import sys


def _is_bomb(line: str) -> bool:
    if "runtime.unassign" not in line:
        return False
    return not line.lstrip().startswith("#")


def main(argv: list[str]) -> int:
    if len(argv) < 2:
        print("usage: check_no_runtime_unassign.py <notebook.ipynb> [...]", file=sys.stderr)
        return 2
    rc = 0
    for path in argv[1:]:
        try:
            with open(path, encoding="utf-8") as f:
                nb = json.load(f)
        except (OSError, json.JSONDecodeError) as exc:
            print(f"FAIL {path}: cannot read notebook: {exc}", file=sys.stderr)
            rc = 1
            continue

        bombs: list[tuple[int, str]] = []
        for i, cell in enumerate(nb.get("cells", [])):
            if cell.get("cell_type") != "code":
                continue
            src = cell.get("source", "")
            if isinstance(src, list):
                src = "".join(src)
            for line in src.splitlines():
                if _is_bomb(line):
                    bombs.append((i, line.strip()))
        if bombs:
            for cell_idx, line in bombs:
                print(
                    f"FAIL {path}: BOMB in cell {cell_idx}: live runtime.unassign() — {line!r}",
                    file=sys.stderr,
                )
            rc = 1
        else:
            print(f"OK {path}")
    return rc


if __name__ == "__main__":
    sys.exit(main(sys.argv))
