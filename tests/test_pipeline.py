"""Tests for the pipeline module.

These tests cover the guard rails and the cache path. The actual XTTS-v2
inference is mocked: we never invoke the real model in CI (no GPU, no
weights download, no CPML acceptance). The mock asserts that the
synthesis function is called with the right arguments.
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import pytest

from voicelegacy.config import (
    MIN_USABLE_REFERENCE_SEGMENTS,
    PipelineConfig,
    ReferenceConfig,
    SynthesisConfig,
    WorkspacePaths,
)
from voicelegacy.pipeline import (
    CorpusBuildResult,
    SynthesisResult,
    run_reference_phase,
    run_synthesis,
)


def _make_config(**kwargs) -> PipelineConfig:
    """Build a PipelineConfig with TOS pre-accepted (tests only)."""
    return PipelineConfig(accept_coqui_tos=True, **kwargs)


class TestReferencePhaseEmptyInputs:
    def test_returns_empty_when_no_jsons(self, tmp_path: Path) -> None:
        paths = WorkspacePaths(workspace=tmp_path)
        config = _make_config()
        result = run_reference_phase(paths, config)
        assert isinstance(result, CorpusBuildResult)
        assert result.all_wavs == []
        assert result.top_wavs == []
        assert result.reports == []


class TestSynthesisGuards:
    def test_empty_reference_list_raises(self, tmp_path: Path) -> None:
        paths = WorkspacePaths(workspace=tmp_path)
        paths.mkdirs()
        config = _make_config()
        with pytest.raises(ValueError, match="No reference WAVs"):
            run_synthesis("Hola", [], paths, config)

    def test_too_few_references_raises(self, tmp_path: Path) -> None:
        """Below MIN_USABLE_REFERENCE_SEGMENTS we abort to prevent shipping a bad clone."""
        paths = WorkspacePaths(workspace=tmp_path)
        paths.mkdirs()
        config = _make_config()
        # Provide one less than the floor
        too_few = [tmp_path / f"ref_{i}.wav" for i in range(MIN_USABLE_REFERENCE_SEGMENTS - 1)]
        for p in too_few:
            p.touch()
        with pytest.raises(ValueError, match=str(MIN_USABLE_REFERENCE_SEGMENTS)):
            run_synthesis("Hola", too_few, paths, config)


class TestSynthesisCachePath:
    """Idempotency: identical (text, refs, config) → cache hit, no inference."""

    @patch("voicelegacy.pipeline.synthesize_to_file")
    @patch("voicelegacy.pipeline.load_xtts_model")
    def test_cache_hit_skips_inference(
        self,
        mock_load_model,
        mock_synthesize,
        tmp_path: Path,
    ) -> None:
        # Arrange
        paths = WorkspacePaths(workspace=tmp_path)
        paths.mkdirs()
        # Build a reference set with real content so hashing works
        refs = []
        for i in range(MIN_USABLE_REFERENCE_SEGMENTS):
            p = paths.reference_corpus / f"ref_{i}.wav"
            p.write_bytes(f"fake-audio-{i}".encode())
            refs.append(p)

        text = "Frase de prueba para clonar"
        config = _make_config(synthesis=SynthesisConfig(language="es"))

        # 1st call: cache miss → synthesize is called, output is materialized
        def _fake_synthesize(tts, text, speaker_wav, output_path, config):
            Path(output_path).write_bytes(b"fake-wav")
            return output_path

        mock_synthesize.side_effect = _fake_synthesize

        first = run_synthesis(text, refs, paths, config)
        assert first.cached is False
        assert mock_synthesize.call_count == 1
        assert first.output_path.exists()
        assert first.metadata_path is not None
        assert first.metadata_path.exists()
        metadata = json.loads(first.metadata_path.read_text(encoding="utf-8"))
        assert metadata["synthesis_config"]["seed"] == 42
        assert metadata["similarity"]["status"] == "skipped"
        assert metadata["source_quality"]["degraded_mode"] is False

        # 2nd call with identical inputs: should be a cache hit
        second = run_synthesis(text, refs, paths, config)
        assert second.cached is True
        assert second.output_path == first.output_path
        assert second.metadata_path == first.metadata_path
        # Synthesize NOT called again
        assert mock_synthesize.call_count == 1

    @patch("voicelegacy.pipeline.synthesize_to_file")
    @patch("voicelegacy.pipeline.load_xtts_model")
    def test_force_resynthesize_bypasses_cache(
        self,
        mock_load_model,
        mock_synthesize,
        tmp_path: Path,
    ) -> None:
        paths = WorkspacePaths(workspace=tmp_path)
        paths.mkdirs()
        refs = []
        for i in range(MIN_USABLE_REFERENCE_SEGMENTS):
            p = paths.reference_corpus / f"ref_{i}.wav"
            p.write_bytes(f"fake-{i}".encode())
            refs.append(p)

        def _fake(tts, text, speaker_wav, output_path, config):
            Path(output_path).write_bytes(b"x")
            return output_path

        mock_synthesize.side_effect = _fake

        config1 = _make_config()
        run_synthesis("texto", refs, paths, config1)
        assert mock_synthesize.call_count == 1

        config2 = _make_config(force_resynthesize=True)
        run_synthesis("texto", refs, paths, config2)
        # Forced re-run → synthesize called a second time
        assert mock_synthesize.call_count == 2

    @patch("voicelegacy.pipeline.synthesize_to_file")
    @patch("voicelegacy.pipeline.load_xtts_model")
    def test_different_text_yields_different_output(
        self,
        mock_load_model,
        mock_synthesize,
        tmp_path: Path,
    ) -> None:
        paths = WorkspacePaths(workspace=tmp_path)
        paths.mkdirs()
        refs = []
        for i in range(MIN_USABLE_REFERENCE_SEGMENTS):
            p = paths.reference_corpus / f"r_{i}.wav"
            p.write_bytes(f"r-{i}".encode())
            refs.append(p)

        def _fake(tts, text, speaker_wav, output_path, config):
            Path(output_path).write_bytes(b"x")
            return output_path

        mock_synthesize.side_effect = _fake

        config = _make_config()
        r1 = run_synthesis("primer texto", refs, paths, config)
        r2 = run_synthesis("segundo texto", refs, paths, config)
        assert r1.output_path != r2.output_path
        assert r1.cached is False
        assert r2.cached is False

    @patch("voicelegacy.pipeline.synthesize_to_file")
    @patch("voicelegacy.pipeline.load_xtts_model")
    def test_sidecar_marks_degraded_mode_when_threshold_is_low(
        self,
        mock_load_model,
        mock_synthesize,
        tmp_path: Path,
    ) -> None:
        paths = WorkspacePaths(workspace=tmp_path)
        paths.mkdirs()
        refs = []
        for i in range(MIN_USABLE_REFERENCE_SEGMENTS):
            p = paths.reference_corpus / f"degraded_{i}.wav"
            p.write_bytes(f"fake-{i}".encode())
            refs.append(p)

        def _fake(tts, text, speaker_wav, output_path, config):
            Path(output_path).write_bytes(b"x")
            return output_path

        mock_synthesize.side_effect = _fake
        config = _make_config(
            reference=ReferenceConfig(min_snr_db=3.0),
            synthesis=SynthesisConfig(compute_similarity=False),
        )
        result = run_synthesis("texto degradado", refs, paths, config)

        assert result.metadata_path is not None
        metadata = json.loads(result.metadata_path.read_text(encoding="utf-8"))
        assert metadata["source_quality"]["degraded_mode"] is True
        assert metadata["similarity"]["status"] == "disabled"


class TestSynthesisResultDataclass:
    """SynthesisResult should be frozen and round-trip cleanly."""

    def test_synthesis_result_is_frozen(self, tmp_path: Path) -> None:
        from dataclasses import FrozenInstanceError

        r = SynthesisResult(
            output_path=tmp_path / "x.wav",
            text="hola",
            reference_set_hash="abc",
            cached=False,
        )
        with pytest.raises(FrozenInstanceError):
            r.text = "no"  # type: ignore[misc]


class TestReferenceRebuildBackup:
    def test_force_rebuild_moves_existing_references_to_backup(self, tmp_path: Path) -> None:
        from voicelegacy.pipeline import _backup_reference_corpus

        paths = WorkspacePaths(workspace=tmp_path)
        paths.mkdirs()
        existing = paths.reference_corpus / "old.wav"
        existing.write_bytes(b"old")

        backup = _backup_reference_corpus(paths)

        assert backup is not None
        assert not existing.exists()
        assert (backup / "old.wav").read_bytes() == b"old"

    def test_sidecar_records_voicelegacy_version(
        self,
        tmp_path: Path,
    ) -> None:
        from unittest.mock import patch

        paths = WorkspacePaths(workspace=tmp_path)
        paths.mkdirs()
        refs = []
        for i in range(MIN_USABLE_REFERENCE_SEGMENTS):
            p = paths.reference_corpus / f"v_{i}.wav"
            p.write_bytes(f"v-{i}".encode())
            refs.append(p)

        def _fake(tts, text, speaker_wav, output_path, config):
            Path(output_path).write_bytes(b"x")
            return output_path

        with (
            patch("voicelegacy.pipeline.load_xtts_model"),
            patch("voicelegacy.pipeline.synthesize_to_file", side_effect=_fake),
        ):
            result = run_synthesis(
                "version metadata",
                refs,
                paths,
                _make_config(synthesis=SynthesisConfig(compute_similarity=False)),
            )

        assert result.metadata_path is not None
        payload = json.loads(result.metadata_path.read_text(encoding="utf-8"))
        assert payload["voicelegacy_version"]


class TestOptionalSimilarityBranches:
    def test_similarity_disabled_returns_disabled_payload(self, tmp_path: Path) -> None:
        from voicelegacy.pipeline import _compute_optional_similarity

        score, payload = _compute_optional_similarity(tmp_path / "out.wav", [], enabled=False)

        assert score is None
        assert payload == {"status": "disabled"}

    def test_similarity_success_payload_includes_quality_band(
        self, tmp_path: Path, monkeypatch
    ) -> None:
        import voicelegacy.similarity as similarity
        from voicelegacy.pipeline import _compute_optional_similarity

        out = tmp_path / "out.wav"
        out.write_bytes(b"x")
        refs = [tmp_path / "ref.wav"]
        refs[0].write_bytes(b"r")

        class FakeReport:
            score = 0.82

            def to_dict(self):
                return {"speaker_similarity_score": 0.82, "encoder": "fake"}

            def quality_band(self):
                return "high"

        monkeypatch.setattr(
            similarity, "compute_similarity", lambda output, references: FakeReport()
        )

        score, payload = _compute_optional_similarity(out, refs, enabled=True)

        assert score == 0.82
        assert payload["status"] == "ok"
        assert payload["quality_band"] == "high"

    def test_similarity_runtime_failure_is_non_fatal(self, tmp_path: Path, monkeypatch) -> None:
        import voicelegacy.similarity as similarity
        from voicelegacy.pipeline import _compute_optional_similarity

        def _boom(output, references):
            raise RuntimeError("encoder failed")

        monkeypatch.setattr(similarity, "compute_similarity", _boom)

        score, payload = _compute_optional_similarity(tmp_path / "out.wav", [], enabled=True)

        assert score is None
        assert payload["status"] == "failed"
        assert "encoder failed" in str(payload["reason"])


class TestQualitySummaryAndBatchSynthesis:
    def test_reference_quality_summary_handles_evaluation_failures(
        self, tmp_path: Path, monkeypatch
    ) -> None:
        from voicelegacy.pipeline import _reference_quality_summary

        ref = tmp_path / "missing.wav"
        cfg = _make_config(reference=ReferenceConfig(min_snr_db=3.0))

        summary = _reference_quality_summary([ref], cfg)

        assert summary["reference_count"] == 1
        assert summary["degraded_mode"] is True
        assert summary["quality_reports"] == []

    def test_run_batch_synthesis_empty_returns_empty(self, tmp_path: Path) -> None:
        from voicelegacy.pipeline import run_batch_synthesis

        paths = WorkspacePaths(workspace=tmp_path)
        paths.mkdirs()

        assert run_batch_synthesis([], [], paths, _make_config()) == []

    def test_run_batch_synthesis_preloads_once_and_runs_each_text(
        self, tmp_path: Path, monkeypatch
    ) -> None:
        from voicelegacy import pipeline

        paths = WorkspacePaths(workspace=tmp_path)
        paths.mkdirs()
        refs = []
        for i in range(MIN_USABLE_REFERENCE_SEGMENTS):
            p = paths.reference_corpus / f"ref_{i}.wav"
            p.write_bytes(f"ref-{i}".encode())
            refs.append(p)

        preload_calls = []
        monkeypatch.setattr(
            pipeline, "load_xtts_model", lambda config, accept_tos: preload_calls.append(True)
        )
        monkeypatch.setattr(
            pipeline,
            "run_synthesis",
            lambda text, reference_wavs, paths, config: SynthesisResult(
                output_path=paths.synthesis_out / f"{text}.wav",
                text=text,
                reference_set_hash="hash",
                cached=False,
            ),
        )

        results = pipeline.run_batch_synthesis(["uno", "dos"], refs, paths, _make_config())

        assert len(preload_calls) == 1
        assert [r.text for r in results] == ["uno", "dos"]
