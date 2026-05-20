"""Tests for optional speaker similarity scoring."""

from __future__ import annotations

import sys
import types
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from voicelegacy.similarity import (
    SimilarityReport,
    _cosine_similarity,
    _load_encoder,
    compute_similarity,
    compute_similarity_batch,
    is_available,
    release_encoder,
)


# ─── Fakes ──────────────────────────────────────────────────────────
class FakeEncoder:
    """Deterministic fake encoder for tests without Resemblyzer.

    The embedding scheme is intentionally simple:
    - paths containing "out"   → [1, 0, 0]
    - paths containing "ref_a" → [1, 0, 0]    (perfect match)
    - everything else          → [0, 1, 0]    (orthogonal)

    This makes cosine results easy to reason about in assertions.
    """

    def embed_utterance(self, wav: np.ndarray | Path) -> np.ndarray:
        s = str(wav)
        if "out" in s:
            return np.array([1.0, 0.0, 0.0], dtype=np.float32)
        if "ref_a" in s:
            return np.array([1.0, 0.0, 0.0], dtype=np.float32)
        return np.array([0.0, 1.0, 0.0], dtype=np.float32)


@pytest.fixture(autouse=True)
def _clear_encoder_cache() -> None:
    """Each test starts with an empty encoder cache."""
    from voicelegacy.similarity import _ENCODER_CACHE

    _ENCODER_CACHE.clear()


# ─── _cosine_similarity ─────────────────────────────────────────────
class TestCosineSimilarity:
    def test_zero_vector_returns_zero(self) -> None:
        assert _cosine_similarity(np.zeros(3), np.ones(3)) == 0.0

    def test_zero_vector_either_side(self) -> None:
        # Both branches of the zero-norm guard
        assert _cosine_similarity(np.ones(3), np.zeros(3)) == 0.0

    def test_identical_vectors_return_one(self) -> None:
        v = np.array([1.0, 2.0, 3.0])
        assert _cosine_similarity(v, v) == pytest.approx(1.0)

    def test_orthogonal_vectors_return_zero(self) -> None:
        a = np.array([1.0, 0.0, 0.0])
        b = np.array([0.0, 1.0, 0.0])
        assert _cosine_similarity(a, b) == pytest.approx(0.0)

    def test_opposite_vectors_return_minus_one(self) -> None:
        a = np.array([1.0, 0.0, 0.0])
        b = np.array([-1.0, 0.0, 0.0])
        assert _cosine_similarity(a, b) == pytest.approx(-1.0)


# ─── SimilarityReport.quality_band — all bands ─────────────────────
class TestQualityBands:
    """Each band must be reachable and have explicit thresholds."""

    @pytest.mark.parametrize(
        ("score", "expected_band"),
        [
            (0.99, "very_high"),
            (0.85, "very_high"),  # boundary inclusive
            (0.84, "high"),
            (0.75, "high"),  # boundary inclusive
            (0.74, "marginal"),
            (0.60, "marginal"),  # boundary inclusive
            (0.59, "low"),
            (0.0, "low"),
        ],
    )
    def test_band_for_score(self, tmp_path: Path, score: float, expected_band: str) -> None:
        r = SimilarityReport(
            output_path=tmp_path / "x.wav",
            score=score,
            n_references=1,
            per_reference_scores=(score,),
            encoder_name="fake",
        )
        assert r.quality_band() == expected_band

    def test_to_dict_serializes_all_fields(self, tmp_path: Path) -> None:
        r = SimilarityReport(
            output_path=tmp_path / "out.wav",
            score=0.7634,
            n_references=3,
            per_reference_scores=(0.8123, 0.7045, 0.7734),
            encoder_name="resemblyzer_v0",
        )
        d = r.to_dict()
        assert d["speaker_similarity_score"] == 0.7634
        assert d["n_references"] == 3
        assert d["per_reference_scores"] == [0.8123, 0.7045, 0.7734]
        assert d["encoder"] == "resemblyzer_v0"
        assert str(tmp_path) in d["output_path"]


# ─── _load_encoder — ImportError path (resemblyzer absent) ─────────
class TestLoadEncoderWithoutResemblyzer:
    """When resemblyzer is not installed, callers get an explicit ImportError
    with installation hint. This is intentional: the rest of voicelegacy
    must work without forcing a 50MB+ optional dep on every user.
    """

    def test_import_error_when_resemblyzer_missing(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # Force resemblyzer import to fail even if installed somewhere
        monkeypatch.setitem(sys.modules, "resemblyzer", None)
        with pytest.raises(ImportError, match="resemblyzer is not installed"):
            _load_encoder()

    def test_is_available_false_when_resemblyzer_missing(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setitem(sys.modules, "resemblyzer", None)
        assert is_available() is False


# ─── _load_encoder — successful path with injected fake module ────
class TestLoadEncoderWithFakeResemblyzer:
    """We don't ship resemblyzer in CI, but we still want to cover the
    successful loading path. Injecting a fake resemblyzer module into
    sys.modules is the standard way to do this without monkeypatching
    the internals of voicelegacy.similarity.
    """

    @pytest.fixture
    def fake_resemblyzer(self, monkeypatch: pytest.MonkeyPatch) -> types.ModuleType:
        """Install a minimal fake `resemblyzer` module."""
        module = types.ModuleType("resemblyzer")

        class FakeVoiceEncoder:
            """Stand-in that records initialization for assertions."""

            instantiated = 0

            def __init__(self) -> None:
                FakeVoiceEncoder.instantiated += 1

            def embed_utterance(self, wav: Any) -> np.ndarray:
                return np.array([1.0, 0.0, 0.0], dtype=np.float32)

        module.VoiceEncoder = FakeVoiceEncoder  # type: ignore[attr-defined]
        module.preprocess_wav = lambda p: np.zeros(16000, dtype=np.float32)  # type: ignore[attr-defined]
        monkeypatch.setitem(sys.modules, "resemblyzer", module)
        return module

    def test_loads_and_returns_voice_encoder(self, fake_resemblyzer: types.ModuleType) -> None:
        encoder = _load_encoder()
        assert encoder is not None
        # Type comes from the fake module
        assert encoder.__class__.__name__ == "FakeVoiceEncoder"

    def test_second_call_returns_cached_encoder(self, fake_resemblyzer: types.ModuleType) -> None:
        fake_resemblyzer.VoiceEncoder.instantiated = 0
        e1 = _load_encoder()
        e2 = _load_encoder()
        # Same instance both times, only one VoiceEncoder() call
        assert e1 is e2
        assert fake_resemblyzer.VoiceEncoder.instantiated == 1

    def test_is_available_true_when_resemblyzer_present(
        self, fake_resemblyzer: types.ModuleType
    ) -> None:
        assert is_available() is True


# ─── release_encoder ───────────────────────────────────────────────
class TestReleaseEncoder:
    """release_encoder must:
    - clear the module cache so the next call reloads
    - invoke gc.collect (always safe)
    - call torch.cuda.empty_cache when torch + CUDA available (best effort)
    - silently no-op when torch absent (the common test environment case)
    """

    def test_clears_cache(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from voicelegacy.similarity import _ENCODER_CACHE

        _ENCODER_CACHE["resemblyzer"] = "sentinel"
        release_encoder()
        assert _ENCODER_CACHE == {}

    def test_no_op_when_torch_absent(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # Block torch import to exercise the ImportError branch.
        monkeypatch.setitem(sys.modules, "torch", None)
        # Must not raise
        release_encoder()

    def test_calls_empty_cache_when_cuda_available(self, monkeypatch: pytest.MonkeyPatch) -> None:
        empty_calls: list[int] = []

        # Build a fake torch.cuda surface.
        fake_cuda = types.SimpleNamespace(
            is_available=lambda: True,
            empty_cache=lambda: empty_calls.append(1),
        )
        fake_torch = types.ModuleType("torch")
        fake_torch.cuda = fake_cuda  # type: ignore[attr-defined]
        monkeypatch.setitem(sys.modules, "torch", fake_torch)

        release_encoder()
        assert empty_calls == [1]

    def test_skips_empty_cache_when_cuda_unavailable(self, monkeypatch: pytest.MonkeyPatch) -> None:
        empty_calls: list[int] = []

        fake_cuda = types.SimpleNamespace(
            is_available=lambda: False,
            empty_cache=lambda: empty_calls.append(1),
        )
        fake_torch = types.ModuleType("torch")
        fake_torch.cuda = fake_cuda  # type: ignore[attr-defined]
        monkeypatch.setitem(sys.modules, "torch", fake_torch)

        release_encoder()
        assert empty_calls == []


# ─── compute_similarity — happy path + validation ─────────────────
class TestComputeSimilarity:
    def test_happy_path_with_injected_encoder(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        out = tmp_path / "out.wav"
        ref_a = tmp_path / "ref_a.wav"
        ref_b = tmp_path / "ref_b.wav"
        for p in (out, ref_a, ref_b):
            p.write_bytes(b"fake")

        def fake_embed(_encoder: Any, wav_path: Path) -> np.ndarray:
            return FakeEncoder().embed_utterance(wav_path)

        monkeypatch.setattr("voicelegacy.similarity._embed_one", fake_embed)
        report = compute_similarity(out, [ref_a, ref_b], encoder=FakeEncoder())

        assert report.n_references == 2
        # centroid of [1,0,0] and [0,1,0] = [0.5, 0.5, 0]; out is [1,0,0]
        # cosine = 0.5 / sqrt(0.5) ≈ 0.707
        assert 0.70 < report.score < 0.72
        assert report.per_reference_scores == (1.0, 0.0)
        assert report.encoder_name == "resemblyzer_v0"
        assert report.output_path == out

    def test_negative_cosine_is_clamped_to_zero(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        out = tmp_path / "out.wav"
        ref = tmp_path / "ref_b.wav"  # orthogonal-then-opposite to "out"
        for p in (out, ref):
            p.write_bytes(b"fake")

        def opposite_embed(_encoder: Any, wav_path: Path) -> np.ndarray:
            if "out" in str(wav_path):
                return np.array([1.0, 0.0, 0.0], dtype=np.float32)
            return np.array([-1.0, 0.0, 0.0], dtype=np.float32)

        monkeypatch.setattr("voicelegacy.similarity._embed_one", opposite_embed)
        report = compute_similarity(out, [ref], encoder=FakeEncoder())
        # raw cosine = -1, clamped
        assert report.score == 0.0
        assert report.quality_band() == "low"

    def test_requires_references(self, tmp_path: Path) -> None:
        out = tmp_path / "out.wav"
        out.write_bytes(b"fake")
        with pytest.raises(ValueError, match="At least one"):
            compute_similarity(out, [], encoder=FakeEncoder())

    def test_raises_when_output_missing(self, tmp_path: Path) -> None:
        ref = tmp_path / "ref.wav"
        ref.write_bytes(b"fake")
        with pytest.raises(FileNotFoundError, match="Output WAV not found"):
            compute_similarity(tmp_path / "ghost.wav", [ref], encoder=FakeEncoder())

    def test_raises_when_any_reference_missing(self, tmp_path: Path) -> None:
        out = tmp_path / "out.wav"
        ref_ok = tmp_path / "ref.wav"
        out.write_bytes(b"fake")
        ref_ok.write_bytes(b"fake")
        with pytest.raises(FileNotFoundError, match="Reference WAV not found"):
            compute_similarity(out, [ref_ok, tmp_path / "missing.wav"], encoder=FakeEncoder())

    def test_uses_module_default_encoder_when_none_passed(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """If no encoder is injected, _load_encoder is used. We mock both
        the loader and the per-file embed so this runs with no real model."""
        out = tmp_path / "out.wav"
        ref = tmp_path / "ref_a.wav"
        for p in (out, ref):
            p.write_bytes(b"fake")

        load_calls: list[int] = []

        def fake_load() -> object:
            load_calls.append(1)
            return object()

        def fake_embed(_encoder: Any, wav_path: Path) -> np.ndarray:
            return FakeEncoder().embed_utterance(wav_path)

        monkeypatch.setattr("voicelegacy.similarity._load_encoder", fake_load)
        monkeypatch.setattr("voicelegacy.similarity._embed_one", fake_embed)

        report = compute_similarity(out, [ref])  # no encoder arg
        assert load_calls == [1]
        assert report.score == pytest.approx(1.0)  # both [1,0,0]


# ─── compute_similarity_batch ──────────────────────────────────────
class TestComputeSimilarityBatch:
    def test_empty_input_returns_empty_list(self) -> None:
        assert compute_similarity_batch([]) == []

    def test_scores_multiple_outputs_with_shared_encoder(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        # Three (output, refs) pairs
        out1 = tmp_path / "out1.wav"
        out2 = tmp_path / "out2.wav"
        ref_a = tmp_path / "ref_a.wav"
        ref_b = tmp_path / "ref_b.wav"
        for p in (out1, out2, ref_a, ref_b):
            p.write_bytes(b"fake")

        load_calls: list[int] = []

        def fake_load() -> object:
            load_calls.append(1)
            return object()

        def fake_embed(_encoder: Any, wav_path: Path) -> np.ndarray:
            return FakeEncoder().embed_utterance(wav_path)

        monkeypatch.setattr("voicelegacy.similarity._load_encoder", fake_load)
        monkeypatch.setattr("voicelegacy.similarity._embed_one", fake_embed)

        reports = compute_similarity_batch([(out1, [ref_a, ref_b]), (out2, [ref_a])])

        assert len(reports) == 2
        # Encoder loaded once and shared between calls
        assert load_calls == [1]
        # out2 vs single ref_a (both [1,0,0]) → cosine = 1.0
        assert reports[1].score == pytest.approx(1.0)
        assert reports[1].quality_band() == "very_high"

    def test_skips_failed_pairs_and_continues(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        good_out = tmp_path / "out.wav"
        good_ref = tmp_path / "ref_a.wav"
        for p in (good_out, good_ref):
            p.write_bytes(b"fake")

        def fake_load() -> object:
            return object()

        def fake_embed(_encoder: Any, wav_path: Path) -> np.ndarray:
            return FakeEncoder().embed_utterance(wav_path)

        monkeypatch.setattr("voicelegacy.similarity._load_encoder", fake_load)
        monkeypatch.setattr("voicelegacy.similarity._embed_one", fake_embed)

        reports = compute_similarity_batch(
            [
                (tmp_path / "ghost_out.wav", [good_ref]),  # output missing → skipped
                (good_out, []),  # empty refs → skipped
                (good_out, [good_ref]),  # valid
            ]
        )
        assert len(reports) == 1
        assert reports[0].score == pytest.approx(1.0)
