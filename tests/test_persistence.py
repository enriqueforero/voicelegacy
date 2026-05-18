"""Tests for the persistence module.

Covers the SQLite-backed idempotency cache: hashing, schema creation,
record round-trip, and the cache-miss / cache-hit branches.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from voicelegacy.persistence import (
    compute_run_hash,
    get_synthesis_record,
    hash_file,
    hash_reference_set,
    hash_text,
    open_db,
    save_synthesis_record,
)


class TestHashing:
    def test_hash_text_is_deterministic(self) -> None:
        a = hash_text("hola mundo")
        b = hash_text("hola mundo")
        assert a == b
        assert len(a) == 64  # SHA-256 hex

    def test_hash_text_changes_with_input(self) -> None:
        assert hash_text("a") != hash_text("b")

    def test_hash_file_streams_and_matches_content(self, tmp_path: Path) -> None:
        p = tmp_path / "x.bin"
        p.write_bytes(b"voicelegacy is the project")
        h1 = hash_file(p)
        h2 = hash_file(p)
        assert h1 == h2
        # Different content → different hash
        p.write_bytes(b"different content")
        assert hash_file(p) != h1

    def test_hash_reference_set_is_order_independent(self, tmp_path: Path) -> None:
        a = tmp_path / "a.wav"
        b = tmp_path / "b.wav"
        a.write_bytes(b"AAA")
        b.write_bytes(b"BBB")
        h_ab = hash_reference_set([a, b])
        h_ba = hash_reference_set([b, a])
        assert h_ab == h_ba

    def test_hash_reference_set_changes_with_content(self, tmp_path: Path) -> None:
        a = tmp_path / "a.wav"
        b = tmp_path / "b.wav"
        a.write_bytes(b"AAA")
        b.write_bytes(b"BBB")
        h1 = hash_reference_set([a, b])
        b.write_bytes(b"CCC")
        h2 = hash_reference_set([a, b])
        assert h1 != h2

    def test_run_hash_combines_inputs(self) -> None:
        h1 = compute_run_hash("text1", "refset_hash", '{"a":1}')
        h2 = compute_run_hash("text2", "refset_hash", '{"a":1}')
        h3 = compute_run_hash("text1", "different_refs", '{"a":1}')
        h4 = compute_run_hash("text1", "refset_hash", '{"a":2}')
        # All four should be distinct: each input affects the hash
        assert len({h1, h2, h3, h4}) == 4


class TestDBLifecycle:
    def test_open_db_creates_file_and_schema(self, tmp_path: Path) -> None:
        db_path = tmp_path / "subdir" / "runs.db"
        with open_db(db_path) as conn:
            # Schema must include both tables
            rows = conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()
        names = {r["name"] for r in rows}
        assert "synthesis_runs" in names
        assert "reference_builds" in names
        assert db_path.exists()

    def test_open_db_is_idempotent(self, tmp_path: Path) -> None:
        db_path = tmp_path / "runs.db"
        with open_db(db_path):
            pass
        # Re-opening should not raise
        with open_db(db_path) as conn:
            rows = conn.execute("SELECT name FROM sqlite_master").fetchall()
        assert len(rows) > 0


class TestSynthesisRecords:
    def test_save_then_get_round_trip(self, tmp_path: Path) -> None:
        db_path = tmp_path / "runs.db"
        out_path = tmp_path / "out.wav"
        out_path.touch()

        save_synthesis_record(
            db_path=db_path,
            run_hash="abc123",
            text="Hola mundo",
            reference_set="refhash",
            output_path=out_path,
            config={"language": "es", "temperature": 0.7},
        )

        rec = get_synthesis_record(db_path, "abc123")
        assert rec is not None
        assert rec.text == "Hola mundo"
        assert rec.reference_set == "refhash"
        assert rec.output_path == out_path
        # config_json is stored sorted+canonical
        assert "language" in rec.config_json
        assert "temperature" in rec.config_json

    def test_get_missing_returns_none(self, tmp_path: Path) -> None:
        db_path = tmp_path / "runs.db"
        assert get_synthesis_record(db_path, "does_not_exist") is None

    def test_save_is_idempotent_on_same_hash(self, tmp_path: Path) -> None:
        db_path = tmp_path / "runs.db"
        out = tmp_path / "x.wav"
        out.touch()
        for i in range(3):
            save_synthesis_record(
                db_path=db_path,
                run_hash="same_hash",
                text=f"text {i}",
                reference_set="r",
                output_path=out,
                config={"i": i},
            )
        # Only one row in the table
        with open_db(db_path) as conn:
            n = conn.execute("SELECT COUNT(*) AS c FROM synthesis_runs").fetchone()["c"]
        assert n == 1
        # And the LAST save won
        rec = get_synthesis_record(db_path, "same_hash")
        assert rec is not None
        assert rec.text == "text 2"


@pytest.mark.parametrize(
    "text,expected_change", [("", True), ("a", True), ("muy largo " * 100, True)]
)
def test_hash_text_is_unique_across_inputs(text: str, expected_change: bool) -> None:
    baseline = hash_text("baseline")
    if expected_change:
        assert hash_text(text) != baseline
