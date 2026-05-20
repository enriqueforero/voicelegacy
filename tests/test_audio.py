"""Tests for the audio module."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from voicelegacy.audio import (
    compute_stats,
    load_audio_mono,
    loudness_normalize,
    preprocess_full,
    save_wav,
    slice_segment,
    trim_silence,
)
from voicelegacy.config import XTTS_INPUT_SR


class TestLoad:
    def test_loads_synthetic_wav(self, synthetic_speech_wav: Path) -> None:
        y, original_sr = load_audio_mono(synthetic_speech_wav)
        assert y.dtype == np.float32
        assert y.ndim == 1
        assert np.max(np.abs(y)) <= 1.0
        assert len(y) > XTTS_INPUT_SR  # at least 1s
        # Fixture writes at XTTS_INPUT_SR, so original_sr must echo it.
        assert original_sr == XTTS_INPUT_SR

    def test_missing_file_raises(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError):
            load_audio_mono(tmp_path / "nonexistent.wav")


class TestStats:
    def test_stats_for_known_signal(self, synthetic_speech_wav: Path) -> None:
        y, _ = load_audio_mono(synthetic_speech_wav)
        stats = compute_stats(y, XTTS_INPUT_SR)
        assert stats.sample_rate == XTTS_INPUT_SR
        assert 9.0 < stats.duration_s < 11.0
        assert stats.snr_db > 0  # signal stronger than noise


class TestSlice:
    def test_slice_in_bounds(self) -> None:
        sr = XTTS_INPUT_SR
        y = np.linspace(-1.0, 1.0, sr * 10, dtype=np.float32)  # 10s
        out = slice_segment(y, sr, 2.0, 5.0)
        assert len(out) == sr * 3

    def test_invalid_bounds_raises(self) -> None:
        sr = XTTS_INPUT_SR
        y = np.zeros(sr, dtype=np.float32)
        with pytest.raises(ValueError):
            slice_segment(y, sr, 0.5, 0.5)


class TestNormalize:
    def test_loudness_normalize_does_not_clip(self, synthetic_speech_wav: Path) -> None:
        y, _ = load_audio_mono(synthetic_speech_wav)
        y_norm = loudness_normalize(y, XTTS_INPUT_SR, target_lufs=-20.0)
        assert np.max(np.abs(y_norm)) <= 1.0


class TestTrim:
    def test_trim_silence_removes_leading_padding(self) -> None:
        sr = XTTS_INPUT_SR
        # 1s of silence, 1s of tone, 1s of silence
        silence = np.zeros(sr, dtype=np.float32)
        tone = 0.5 * np.sin(2 * np.pi * 440 * np.arange(sr) / sr).astype(np.float32)
        y = np.concatenate([silence, tone, silence])
        y_trim = trim_silence(y, top_db=20.0)
        assert len(y_trim) < len(y)
        assert len(y_trim) >= int(sr * 0.5)  # tone should remain


class TestSaveWav:
    def test_roundtrip(self, tmp_path: Path) -> None:
        sr = XTTS_INPUT_SR
        y = 0.1 * np.sin(2 * np.pi * 440 * np.arange(sr) / sr).astype(np.float32)
        out = tmp_path / "roundtrip.wav"
        save_wav(out, y, sr)
        assert out.exists()
        y2, _ = load_audio_mono(out)
        # PCM_16 quantization is fine; just confirm length and roughly same content
        assert len(y2) == len(y)
        assert np.corrcoef(y, y2)[0, 1] > 0.99


class TestPreprocessFull:
    def test_end_to_end(self, synthetic_speech_wav: Path) -> None:
        y, stats = preprocess_full(synthetic_speech_wav, apply_denoise=False)
        assert stats.duration_s > 0
        assert stats.sample_rate == XTTS_INPUT_SR
        assert np.max(np.abs(y)) <= 1.0


class TestLoudnessNoClipping:
    """Regression tests for the circular-clipping bug fixed in P0-4.

    Before the fix, pyloudnorm could push transient peaks above 1.0 when the
    target_lufs was loud relative to the source, and the subsequent
    np.clip(-1, 1) hard-clipped them to exactly 1.0. The quality gate then
    flagged the file for "clipping risk: peak 0.0 dBFS" — failure introduced
    by the pipeline itself.

    These tests assert that, regardless of input level, the output never
    sits at the ceiling exact: peak must be strictly below 1.0 with
    measurable headroom.
    """

    @staticmethod
    def _generate_aggressive_signal(sr: int, duration_s: float, peak_target: float) -> np.ndarray:
        """Build a speech-like signal with peaks near `peak_target` (in [-1,1])."""
        n = int(sr * duration_s)
        t = np.arange(n) / sr
        signal = (
            0.6 * np.sin(2 * np.pi * 220 * t)
            + 0.3 * np.sin(2 * np.pi * 440 * t)
            + 0.15 * np.sin(2 * np.pi * 880 * t)
        ).astype(np.float32)
        # Scale so max abs == peak_target
        signal = signal * (peak_target / (float(np.max(np.abs(signal))) + 1e-9))
        return signal

    def test_aggressive_input_does_not_clip(self) -> None:
        sr = XTTS_INPUT_SR
        y = self._generate_aggressive_signal(sr, duration_s=5.0, peak_target=0.95)
        y_norm = loudness_normalize(y, sr, target_lufs=-16.0, peak_ceiling_dbfs=-3.0)
        peak = float(np.max(np.abs(y_norm)))
        # -3 dBFS = 0.7079. Allow a tiny epsilon for float32 rounding.
        assert peak < 0.71, f"peak {peak:.4f} above -3 dBFS ceiling — limiter failed"
        # Hard-clip signature: peak landed exactly at 1.0. Must NEVER happen.
        assert peak < 0.999, "Hard clipping detected (peak ≈ 1.0)"

    def test_peak_ceiling_dbfs_validation(self) -> None:
        sr = XTTS_INPUT_SR
        y = self._generate_aggressive_signal(sr, duration_s=3.0, peak_target=0.5)
        with pytest.raises(ValueError, match="peak_ceiling_dbfs"):
            loudness_normalize(y, sr, target_lufs=-23.0, peak_ceiling_dbfs=0.0)

    def test_quiet_input_preserved(self) -> None:
        """When source LUFS == target, the limiter must not engage."""
        sr = XTTS_INPUT_SR
        # Quiet input — already around -23 LUFS, peaks well below ceiling
        y = self._generate_aggressive_signal(sr, duration_s=5.0, peak_target=0.10)
        y_norm = loudness_normalize(y, sr, target_lufs=-23.0, peak_ceiling_dbfs=-3.0)
        peak = float(np.max(np.abs(y_norm)))
        assert peak < 0.71  # below ceiling
        # Result is finite, audible, not silent
        assert peak > 1e-3


class TestP1AudioCleanup:
    """P1 cleanup filters for sub-optimal archival audio."""

    def test_preemphasis_changes_signal_and_preserves_shape(self) -> None:
        sr = XTTS_INPUT_SR
        y = np.sin(2 * np.pi * 220 * np.arange(sr) / sr).astype(np.float32)
        out = __import__("voicelegacy.audio", fromlist=["apply_preemphasis"]).apply_preemphasis(y)
        assert out.shape == y.shape
        assert out.dtype == np.float32
        assert not np.allclose(out, y)

    def test_bandpass_preserves_shape(self) -> None:
        from voicelegacy.audio import apply_bandpass

        sr = XTTS_INPUT_SR
        y = np.random.default_rng(42).standard_normal(sr).astype(np.float32) * 0.01
        out = apply_bandpass(y, sr)
        assert out.shape == y.shape
        assert out.dtype == np.float32

    def test_dynamic_range_alias_matches_estimator(self) -> None:
        from voicelegacy.audio import _estimate_dynamic_range_db, _estimate_snr_db

        y = np.concatenate(
            [
                np.zeros(2048, dtype=np.float32),
                np.ones(4096, dtype=np.float32) * 0.2,
            ]
        )
        assert _estimate_snr_db(y) == _estimate_dynamic_range_db(y)
