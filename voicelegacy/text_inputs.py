"""Load synthesis texts from CLI-friendly files."""

from __future__ import annotations

import csv
from pathlib import Path


def load_texts_from_file(path: Path, *, column: str = "text") -> list[str]:
    """Load synthesis texts from .txt or .csv.

    .txt: one non-empty line per utterance. Blank lines and lines starting with
    # are ignored.

    .csv: uses a column named ``text`` by default. If that column is absent and
    the CSV has exactly one column, that column is used. This keeps the command
    forgiving without guessing in multi-column files.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Text file not found: {path}")
    suffix = path.suffix.lower()
    if suffix == ".txt":
        return [
            line.strip()
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip() and not line.lstrip().startswith("#")
        ]
    if suffix == ".csv":
        with path.open("r", encoding="utf-8-sig", newline="") as f:
            reader = csv.DictReader(f)
            if reader.fieldnames is None:
                return []
            fieldnames = [name.strip() for name in reader.fieldnames]
            selected = column if column in fieldnames else None
            if selected is None and len(fieldnames) == 1:
                selected = fieldnames[0]
            if selected is None:
                raise ValueError(
                    f"CSV must contain a '{column}' column or exactly one column; found {fieldnames}"
                )
            texts = []
            for row in reader:
                value = (row.get(selected) or "").strip()
                if value:
                    texts.append(value)
            return texts
    raise ValueError("Unsupported text file type. Use .txt or .csv")


def resolve_text_inputs(text: str | None, text_file: Path | None) -> list[str]:
    """Resolve CLI text options into a non-empty list."""
    texts: list[str] = []
    if text is not None and text.strip():
        texts.append(text.strip())
    if text_file is not None:
        texts.extend(load_texts_from_file(text_file))
    if not texts:
        raise ValueError("Provide --text, --text-file, or both.")
    return texts
