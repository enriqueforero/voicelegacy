"""Tests for CLI text-file input loading."""

from __future__ import annotations

from pathlib import Path

import pytest

from voicelegacy.text_inputs import load_texts_from_file, resolve_text_inputs


def test_load_txt_ignores_blank_and_comments(tmp_path: Path) -> None:
    p = tmp_path / "texts.txt"
    p.write_text("hola\n\n# comentario\nadios\n", encoding="utf-8")
    assert load_texts_from_file(p) == ["hola", "adios"]


def test_load_csv_text_column(tmp_path: Path) -> None:
    p = tmp_path / "texts.csv"
    p.write_text("id,text\n1,Hola mundo\n2,Adios\n", encoding="utf-8")
    assert load_texts_from_file(p) == ["Hola mundo", "Adios"]


def test_load_single_column_csv(tmp_path: Path) -> None:
    p = tmp_path / "texts.csv"
    p.write_text("frase\nUno\nDos\n", encoding="utf-8")
    assert load_texts_from_file(p) == ["Uno", "Dos"]


def test_rejects_ambiguous_csv(tmp_path: Path) -> None:
    p = tmp_path / "texts.csv"
    p.write_text("id,frase\n1,Uno\n", encoding="utf-8")
    with pytest.raises(ValueError, match="text"):
        load_texts_from_file(p)


def test_resolve_text_inputs_combines_text_and_file(tmp_path: Path) -> None:
    p = tmp_path / "texts.txt"
    p.write_text("dos\n", encoding="utf-8")
    assert resolve_text_inputs("uno", p) == ["uno", "dos"]


def test_resolve_text_inputs_requires_something() -> None:
    with pytest.raises(ValueError, match="Provide"):
        resolve_text_inputs(None, None)
