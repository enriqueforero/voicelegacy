"""Tests for the quality module."""

from __future__ import annotations

from pathlib import Path

from voicelegacy.audio import AudioStats
from voicelegacy.quality import evaluate_file, rank_candidates, score_segment


class TestScoreSegment:
    def test_perfect_segment_passes(self) -> None:
        stats = AudioStats(
            duration_s=10.0,
            sample_rate=22050,
            n_channels=1,
            rms_db=-20.0,
            peak_db=-6.0,
            snr_db=35.0,
        )
        score, passed, reasons = score_segment(stats)
        assert passed is True
        assert reasons == ()
        assert score > 0

    def test_phone_audio_rejected(self) -> None:
        stats = AudioStats(
            duration_s=10.0,
            sample_rate=8000,
            n_channels=1,
            rms_db=-20.0,
            peak_db=-6.0,
            snr_db=30.0,
        )
        _, passed, reasons = score_segment(stats)
        assert passed is False
        assert any("phone-codec" in r or "sample_rate" in r for r in reasons)

    def test_too_short_rejected(self) -> None:
        stats = AudioStats(
            duration_s=2.0,
            sample_rate=22050,
            n_channels=1,
            rms_db=-20.0,
            peak_db=-6.0,
            snr_db=30.0,
        )
        _, passed, reasons = score_segment(stats)
        assert passed is False
        assert any("duration" in r for r in reasons)

    def test_noisy_rejected(self) -> None:
        stats = AudioStats(
            duration_s=10.0,
            sample_rate=22050,
            n_channels=1,
            rms_db=-20.0,
            peak_db=-6.0,
            snr_db=5.0,
        )
        _, passed, reasons = score_segment(stats)
        assert passed is False
        assert any("snr" in r.lower() for r in reasons)

    def test_clipping_rejected(self) -> None:
        stats = AudioStats(
            duration_s=10.0,
            sample_rate=22050,
            n_channels=1,
            rms_db=-15.0,
            peak_db=-0.5,
            snr_db=30.0,
        )
        _, passed, reasons = score_segment(stats)
        assert passed is False
        assert any("clip" in r.lower() for r in reasons)


class TestRanking:
    def test_top_n_filters_failures(self) -> None:
        from voicelegacy.quality import QualityReport

        good = QualityReport(
            path=Path("/tmp/a.wav"),
            stats=AudioStats(10.0, 22050, 1, -20, -6, 30),
            score=0.9,
            passed=True,
            reasons=(),
        )
        bad = QualityReport(
            path=Path("/tmp/b.wav"),
            stats=AudioStats(2.0, 22050, 1, -20, -6, 30),
            score=0.1,
            passed=False,
            reasons=("too short",),
        )
        top = rank_candidates([good, bad], top_n=5)
        assert len(top) == 1
        assert top[0].path.name == "a.wav"

    def test_empty_when_nothing_passes(self) -> None:
        from voicelegacy.quality import QualityReport

        bad = QualityReport(
            path=Path("/tmp/x.wav"),
            stats=AudioStats(2.0, 22050, 1, -20, -6, 30),
            score=0.0,
            passed=False,
            reasons=("bad",),
        )
        assert rank_candidates([bad]) == []


class TestEvaluateFile:
    def test_returns_report(self, synthetic_speech_wav: Path) -> None:
        report = evaluate_file(synthetic_speech_wav, min_snr_db=0.0)
        assert report.path == synthetic_speech_wav
        assert report.stats.duration_s > 0
        # Synthetic signal has high SNR, ~10s long → should pass gates
        assert report.passed is True or len(report.reasons) <= 1
