"""voicelegacy — Persistence layer for idempotent runs.

Uses a tiny SQLite database to track:
- Which reference WAVs have been built (content-hashed → skip if unchanged).
- Which (text, reference_set) pairs have already been synthesized.

This lets the notebook be re-run cheaply: only new inputs trigger work.
"""

from __future__ import annotations

import hashlib
import importlib.metadata as metadata
import json
import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from voicelegacy.logging_config import get_logger

logger = get_logger()


SCHEMA = """
CREATE TABLE IF NOT EXISTS reference_builds (
    content_hash TEXT PRIMARY KEY,
    source_json  TEXT NOT NULL,
    output_path  TEXT NOT NULL,
    duration_s   REAL,
    snr_db       REAL,
    score        REAL,
    created_at   TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS synthesis_runs (
    run_hash      TEXT PRIMARY KEY,
    text          TEXT NOT NULL,
    reference_set TEXT NOT NULL,
    output_path   TEXT NOT NULL,
    config_json   TEXT NOT NULL,
    created_at    TEXT NOT NULL
);
"""


@dataclass(frozen=True)
class SynthesisRecord:
    """One row in the synthesis_runs table."""

    run_hash: str
    text: str
    reference_set: str
    output_path: Path
    config_json: str
    created_at: str


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def hash_file(path: Path, chunk_size: int = 65536) -> str:
    """Compute SHA-256 of a file, streaming to keep memory bounded."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(chunk_size):
            h.update(chunk)
    return h.hexdigest()


def hash_text(text: str) -> str:
    """SHA-256 of a UTF-8 encoded string."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def hash_reference_set(reference_paths: list[Path]) -> str:
    """Stable hash of a set of reference files (order-independent, content-aware)."""
    digests = sorted(hash_file(p) for p in reference_paths)
    joined = "|".join(digests)
    return hashlib.sha256(joined.encode("utf-8")).hexdigest()


# ─── DB lifecycle ──────────────────────────────────────────────────
@contextmanager
def open_db(db_path: Path) -> Iterator[sqlite3.Connection]:
    """Context manager for a SQLite connection with WAL + foreign keys."""
    db_path = Path(db_path)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    try:
        conn.executescript(SCHEMA)
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA foreign_keys=ON")
        yield conn
        conn.commit()
    finally:
        conn.close()


# ─── Synthesis records ─────────────────────────────────────────────
def package_version() -> str:
    """Return the installed voicelegacy version for cache invalidation."""
    try:
        return metadata.version("voicelegacy")
    except metadata.PackageNotFoundError:
        return "0.1.0"


def compute_run_hash(text: str, reference_set_hash: str, config_json: str) -> str:
    """Deterministic identifier for a (text, refs, config, package version) tuple.

    Including the package version prevents stale synthesis outputs from being
    reused after the algorithm changes. That is deliberate cache invalidation.
    """
    h = hashlib.sha256()
    h.update(package_version().encode("utf-8"))
    h.update(b"\x00")
    h.update(text.encode("utf-8"))
    h.update(b"\x00")
    h.update(reference_set_hash.encode("utf-8"))
    h.update(b"\x00")
    h.update(config_json.encode("utf-8"))
    return h.hexdigest()


def get_synthesis_record(db_path: Path, run_hash: str) -> SynthesisRecord | None:
    """Fetch an existing synthesis record by hash, or None."""
    with open_db(db_path) as conn:
        row = conn.execute(
            "SELECT * FROM synthesis_runs WHERE run_hash = ?", (run_hash,)
        ).fetchone()
    if row is None:
        return None
    return SynthesisRecord(
        run_hash=row["run_hash"],
        text=row["text"],
        reference_set=row["reference_set"],
        output_path=Path(row["output_path"]),
        config_json=row["config_json"],
        created_at=row["created_at"],
    )


def save_synthesis_record(
    db_path: Path,
    run_hash: str,
    text: str,
    reference_set: str,
    output_path: Path,
    config: dict[str, object],
) -> None:
    """Insert or replace a synthesis record."""
    with open_db(db_path) as conn:
        conn.execute(
            """
            INSERT OR REPLACE INTO synthesis_runs
                (run_hash, text, reference_set, output_path, config_json, created_at)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                run_hash,
                text,
                reference_set,
                str(output_path),
                json.dumps(config, ensure_ascii=False, sort_keys=True),
                _utc_now_iso(),
            ),
        )
    logger.info("Persisted synthesis record {} → {}", run_hash[:12], output_path.name)
