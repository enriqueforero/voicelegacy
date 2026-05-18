"""Tests for the pipeline module.

These tests cover the guard rails and the cache path. The actual XTTS-v2
inference is mocked: we never invoke the real model in CI (no GPU, no
weights download, no CPML acceptance). The mock asserts that the
synthesis function is called with the right arguments.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest

from voicelegacy.config import (
    MIN_USABLE_REFERENCE_SEGMENTS,
    PipelineConfig,
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

        # 2nd call with identical inputs: should be a cache hit
        second = run_synthesis(text, refs, paths, config)
        assert second.cached is True
        assert second.output_path == first.output_path
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
