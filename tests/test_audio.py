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
        y = load_audio_mono(synthetic_speech_wav)
        assert y.dtype == np.float32
        assert y.ndim == 1
        assert np.max(np.abs(y)) <= 1.0
        assert len(y) > XTTS_INPUT_SR  # at least 1s

    def test_missing_file_raises(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError):
            load_audio_mono(tmp_path / "nonexistent.wav")


class TestStats:
    def test_stats_for_known_signal(self, synthetic_speech_wav: Path) -> None:
        y = load_audio_mono(synthetic_speech_wav)
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
        y = load_audio_mono(synthetic_speech_wav)
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
        y2 = load_audio_mono(out)
        # PCM_16 quantization is fine; just confirm length and roughly same content
        assert len(y2) == len(y)
        assert np.corrcoef(y, y2)[0, 1] > 0.99


class TestPreprocessFull:
    def test_end_to_end(self, synthetic_speech_wav: Path) -> None:
        y, stats = preprocess_full(synthetic_speech_wav, apply_denoise=False)
        assert stats.duration_s > 0
        assert stats.sample_rate == XTTS_INPUT_SR
        assert np.max(np.abs(y)) <= 1.0
