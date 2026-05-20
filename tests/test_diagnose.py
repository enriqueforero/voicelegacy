"""Tests for operational diagnostics."""

from __future__ import annotations

import json
from pathlib import Path

from typer.testing import CliRunner

from voicelegacy.cli import app
from voicelegacy.config import WorkspacePaths
from voicelegacy.diagnose import diagnose_workspace

runner = CliRunner()


def _workspace(tmp_path: Path) -> WorkspacePaths:
    paths = WorkspacePaths(workspace=tmp_path)
    paths.mkdirs()
    return paths


def test_diagnose_reports_missing_workspace_dirs(tmp_path: Path) -> None:
    report = diagnose_workspace(tmp_path / "missing")
    assert report.ready is False
    assert any(c.name == "dir:workspace" and c.status == "fail" for c in report.checks)


def test_diagnose_validates_speakerscribe_json(tmp_path: Path) -> None:
    paths = _workspace(tmp_path)
    (paths.speakerscribe_out / "bad.json").write_text(
        json.dumps({"source_audio": "x.wav", "segments": [{"start": "bad", "end": 2.0}]}),
        encoding="utf-8",
    )
    report = diagnose_workspace(tmp_path)
    check = next(c for c in report.checks if c.name == "speakerscribe_json")
    assert check.status == "fail"
    assert "invalid" in check.detail


def test_diagnose_cli_json_output(tmp_path: Path) -> None:
    paths = _workspace(tmp_path)
    (paths.speakerscribe_out / "ok.json").write_text(
        json.dumps({"source_audio": "x.wav", "segments": [{"start": 0.0, "end": 2.0}]}),
        encoding="utf-8",
    )
    result = runner.invoke(app, ["diagnose", "--workspace", str(tmp_path), "--json"])
    assert result.exit_code == 0
    assert "speakerscribe_json" in result.output


def test_diagnose_cli_human_output(tmp_path: Path) -> None:
    _workspace(tmp_path)
    result = runner.invoke(app, ["diagnose", "--workspace", str(tmp_path)])
    assert result.exit_code == 0
    assert "voicelegacy diagnose" in result.output
    assert "ready=" in result.output
