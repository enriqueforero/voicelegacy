"""Tests for the voicelegacy CLI (P0-8 additions).

Uses typer.testing.CliRunner so no subprocess overhead. The CLI was 0%
covered before these tests — referenced in the plan §3.9 as deferred to
v0.1.1, but we close it now because the new convert-audio and
list-speakers commands replace hand-edited notebook cells.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf
from typer.testing import CliRunner

from voicelegacy.audio import _ffmpeg_available
from voicelegacy.cli import app

runner = CliRunner()


def _make_workspace(tmp_path: Path) -> Path:
    """Build a minimal workspace tree (matches WorkspacePaths convention)."""
    for sub in (
        "interviews_raw",
        "speakerscribe_out",
        "reference_corpus",
        "synthesis_out",
        "reports",
    ):
        (tmp_path / sub).mkdir(parents=True, exist_ok=True)
    return tmp_path


def _write_pcm_wav(path: Path, sr: int, duration_s: float = 1.0) -> Path:
    n = int(sr * duration_s)
    t = np.arange(n) / sr
    y = (0.3 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)
    sf.write(str(path), y, sr, subtype="PCM_16")
    return path


# ─── convert-audio ─────────────────────────────────────────────────
class TestConvertAudioCommand:
    @pytest.mark.skipif(not _ffmpeg_available(), reason="ffmpeg not on PATH")
    def test_runs_against_empty_interviews(self, tmp_path: Path) -> None:
        ws = _make_workspace(tmp_path)
        result = runner.invoke(app, ["convert-audio", "--workspace", str(ws)])
        assert result.exit_code == 0, result.output
        assert "0 WAV(s) ready" in result.output

    def test_fails_when_interviews_raw_missing(self, tmp_path: Path) -> None:
        # Workspace without interviews_raw subdir
        result = runner.invoke(app, ["convert-audio", "--workspace", str(tmp_path)])
        assert result.exit_code != 0
        assert "interviews_raw" in result.output


# ─── list-speakers ─────────────────────────────────────────────────
class TestListSpeakersCommand:
    def test_fails_when_no_jsons(self, tmp_path: Path) -> None:
        ws = _make_workspace(tmp_path)
        result = runner.invoke(app, ["list-speakers", "--workspace", str(ws)])
        assert result.exit_code != 0
        assert "No speakerscribe JSONs" in result.output

    def test_lists_speakers_from_json(self, tmp_path: Path) -> None:
        ws = _make_workspace(tmp_path)
        payload = {
            "source_audio": "interview.wav",
            "language_detected": "es",
            "segments": [
                {"speaker": "SPEAKER_00", "start": 0.0, "end": 5.0, "text": "hola"},
                {"speaker": "SPEAKER_00", "start": 5.0, "end": 12.0, "text": "como estas"},
                {"speaker": "SPEAKER_01", "start": 12.0, "end": 14.5, "text": "bien"},
            ],
        }
        (ws / "speakerscribe_out" / "interview.json").write_text(
            json.dumps(payload, ensure_ascii=False), encoding="utf-8"
        )
        result = runner.invoke(app, ["list-speakers", "--workspace", str(ws), "--no-show-files"])
        assert result.exit_code == 0, result.output
        # Both speakers must show in the rendered table
        assert "SPEAKER_00" in result.output
        assert "SPEAKER_01" in result.output

    def test_handles_malformed_json_gracefully(self, tmp_path: Path) -> None:
        ws = _make_workspace(tmp_path)
        (ws / "speakerscribe_out" / "broken.json").write_text("{not valid", encoding="utf-8")
        result = runner.invoke(app, ["list-speakers", "--workspace", str(ws), "--no-show-files"])
        # Command must not crash — it should report the malformed file and exit.
        # exit_code may be 0 (graceful) or 1 (signaled) — both are acceptable
        # as long as the output names the broken file.
        assert "broken.json" in result.output


# ─── build-corpus / synthesize: smoke-only ─────────────────────────
# We can't run the real synthesis path without coqui-tts + a GPU, but we
# can at least confirm the CLI surface accepts the documented flags and
# fails predictably when prerequisites are missing.
class TestBuildCorpusSurface:
    def test_help_works(self) -> None:
        result = runner.invoke(app, ["build-corpus", "--help"])
        assert result.exit_code == 0
        assert "--workspace" in result.output
        assert "--speaker" in result.output
        assert "--accept-tos" in result.output

    def test_missing_workspace_fails(self) -> None:
        # No --workspace flag → typer should reject
        result = runner.invoke(app, ["build-corpus"])
        assert result.exit_code != 0


class TestSynthesizeSurface:
    def test_help_works(self) -> None:
        result = runner.invoke(app, ["synthesize", "--help"])
        assert result.exit_code == 0
        assert "--text" in result.output
        assert "--accept-tos" in result.output

    def test_fails_without_reference_corpus(self, tmp_path: Path) -> None:
        ws = _make_workspace(tmp_path)
        # reference_corpus/ is empty → command must refuse and exit non-zero
        result = runner.invoke(
            app,
            [
                "synthesize",
                "--workspace",
                str(ws),
                "--text",
                "hola",
                "--accept-tos",
            ],
        )
        assert result.exit_code == 2
        assert "No reference WAVs" in result.output


# ─── Root callback / verbose flag ──────────────────────────────────
class TestRootCallback:
    def test_verbose_flag_accepted(self) -> None:
        result = runner.invoke(app, ["--verbose", "build-corpus", "--help"])
        assert result.exit_code == 0


class TestBuildCorpusCommandExecution:
    def test_build_corpus_prints_summary_with_mocked_pipeline(
        self, tmp_path: Path, monkeypatch
    ) -> None:
        import voicelegacy.cli as cli
        from voicelegacy.audio import AudioStats
        from voicelegacy.pipeline import CorpusBuildResult
        from voicelegacy.quality import QualityReport

        ws = _make_workspace(tmp_path)
        wav = ws / "reference_corpus" / "ref.wav"
        wav.write_bytes(b"wav")
        report = QualityReport(
            path=wav,
            stats=AudioStats(8.0, 22050, 1, -20.0, -6.0, 25.0),
            score=0.8,
            passed=True,
            reasons=(),
        )

        monkeypatch.setattr(
            cli,
            "run_reference_phase",
            lambda paths, config: CorpusBuildResult([wav], [wav], [report]),
        )

        result = runner.invoke(
            app,
            ["build-corpus", "--workspace", str(ws), "--accept-tos", "--speaker", "SPEAKER_00"],
        )

        assert result.exit_code == 0, result.output
        assert "Reference corpus summary" in result.output
        assert "Top selected" in result.output


class TestSynthesizeCommandExecution:
    def test_synthesize_uses_text_file_and_prints_sidecar(
        self, tmp_path: Path, monkeypatch
    ) -> None:
        import voicelegacy.cli as cli
        from voicelegacy.pipeline import SynthesisResult

        ws = _make_workspace(tmp_path)
        for i in range(3):
            (ws / "reference_corpus" / f"ref_{i}.wav").write_bytes(b"ref")
        text_file = ws / "texts.txt"
        text_file.write_text("hola\nadios\n", encoding="utf-8")
        wav = ws / "synthesis_out" / "hola.wav"
        sidecar = ws / "synthesis_out" / "hola.json"

        def _fake_batch(texts, reference_wavs, paths, config):
            assert texts == ["hola", "adios"]
            assert len(reference_wavs) == 3
            return [
                SynthesisResult(
                    output_path=wav,
                    text="hola",
                    reference_set_hash="hash",
                    cached=False,
                    metadata_path=sidecar,
                )
            ]

        monkeypatch.setattr(cli, "run_batch_synthesis", _fake_batch)

        result = runner.invoke(
            app,
            ["synthesize", "--workspace", str(ws), "--text-file", str(text_file), "--accept-tos"],
        )

        assert result.exit_code == 0, result.output
        assert "metadata=" in result.output
        assert "hola" in result.output and ".wav" in result.output

    def test_diagnose_json_success_path(self, tmp_path: Path, monkeypatch) -> None:
        import voicelegacy.cli as cli
        from voicelegacy.diagnose import DiagnosticCheck, DiagnosticReport

        ws = _make_workspace(tmp_path)
        report = DiagnosticReport(workspace=ws, checks=[DiagnosticCheck("python", "ok", "ready")])
        monkeypatch.setattr(cli, "diagnose_workspace", lambda workspace, require_gpu=False: report)

        result = runner.invoke(app, ["diagnose", "--workspace", str(ws), "--json"])

        assert result.exit_code == 0, result.output
        assert '"ready": true' in result.output
