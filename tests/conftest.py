"""Pytest fixtures shared across the test suite."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from voicelegacy.config import XTTS_INPUT_SR


@pytest.fixture
def tmp_workspace(tmp_path: Path) -> Path:
    """Create an empty workspace tree on a tmp path."""
    for sub in (
        "interviews_raw",
        "speakerscribe_out",
        "reference_corpus",
        "synthesis_out",
        "reports",
    ):
        (tmp_path / sub).mkdir()
    return tmp_path


@pytest.fixture
def synthetic_speech_wav(tmp_path: Path) -> Path:
    """Synthesize a 10-second pseudo-speech signal (sum of sines + low noise).

    Not real speech, but has speech-like spectral structure and dynamic range.
    Sufficient to exercise the audio + quality modules without external assets.
    """
    sr = XTTS_INPUT_SR
    duration_s = 10.0
    n = int(sr * duration_s)
    t = np.arange(n) / sr

    # Mix of harmonics to fake voiced segments
    signal = (
        0.30 * np.sin(2 * np.pi * 220 * t)
        + 0.15 * np.sin(2 * np.pi * 440 * t)
        + 0.07 * np.sin(2 * np.pi * 880 * t)
    )
    # Envelope that mimics syllables
    env = 0.5 + 0.5 * np.sin(2 * np.pi * 3 * t)
    signal = signal * env

    # Add a small amount of noise
    rng = np.random.default_rng(seed=42)
    signal += 0.005 * rng.standard_normal(n)

    path = tmp_path / "synthetic.wav"
    sf.write(str(path), signal.astype(np.float32), sr, subtype="PCM_16")
    return path


@pytest.fixture
def speakerscribe_json_factory(tmp_path: Path):
    """Factory to create speakerscribe-shaped JSONs for tests."""

    def _make(filename: str, segments: list[dict], audio_name: str = "source.wav") -> Path:
        payload = {
            "source_audio": audio_name,
            "language_detected": "es",
            "segments": segments,
        }
        p = tmp_path / filename
        p.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
        return p

    return _make
