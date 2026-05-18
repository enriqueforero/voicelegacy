"""voicelegacy — Quality scoring for reference-audio candidates.

The single most important driver of clone quality is the reference audio.
This module scores candidate segments so we can keep the top N and discard
the rest, instead of feeding XTTS-v2 noisy or short fragments.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from voicelegacy.audio import AudioStats, compute_stats, load_audio_mono
from voicelegacy.config import (
    MAX_REF_DURATION_S,
    MIN_REF_DURATION_S,
    MIN_SAMPLING_RATE_HZ,
    XTTS_INPUT_SR,
)
from voicelegacy.logging_config import get_logger

logger = get_logger()


@dataclass(frozen=True)
class QualityReport:
    """Result of evaluating a candidate reference audio segment."""

    path: Path
    stats: AudioStats
    score: float
    passed: bool
    reasons: tuple[str, ...]

    def to_dict(self) -> dict[str, object]:
        """Serialize to a JSON-friendly dict for reporting."""
        return {
            "path": str(self.path),
            "duration_s": round(self.stats.duration_s, 3),
            "sample_rate": self.stats.sample_rate,
            "rms_db": round(self.stats.rms_db, 2),
            "peak_db": round(self.stats.peak_db, 2),
            "snr_db": round(self.stats.snr_db, 2),
            "score": round(self.score, 4),
            "passed": self.passed,
            "reasons": list(self.reasons),
        }


def score_segment(
    stats: AudioStats,
    min_duration_s: float = MIN_REF_DURATION_S,
    max_duration_s: float = MAX_REF_DURATION_S,
    min_snr_db: float = 15.0,
    min_sr_hz: int = MIN_SAMPLING_RATE_HZ,
) -> tuple[float, bool, tuple[str, ...]]:
    """Score an audio segment against quality gates.

    Scoring rubric (higher = better):
        score = snr_normalized * duration_factor

    Where:
        - snr_normalized in [0, 1] from min_snr_db (0) to 40 dB (1).
        - duration_factor in [0, 1] peaks at the midpoint of the allowed range
          and falls off symmetrically outside it.

    A segment is considered passing if it clears all hard gates:
        - sample_rate >= min_sr_hz
        - min_duration_s <= duration <= max_duration_s
        - snr_db >= min_snr_db
        - peak_db not clipping (> -1 dB FS)

    Args:
        stats: AudioStats from compute_stats.
        min_duration_s: Hard floor.
        max_duration_s: Hard ceiling.
        min_snr_db: Hard floor for cleanliness.
        min_sr_hz: Hard floor — reject phone-codec audio.

    Returns:
        Tuple (score, passed, list_of_failure_reasons).
    """
    reasons: list[str] = []

    # Hard gates
    if stats.sample_rate < min_sr_hz:
        reasons.append(f"sample_rate {stats.sample_rate}Hz < {min_sr_hz}Hz (phone-codec audio)")
    if stats.duration_s < min_duration_s:
        reasons.append(f"duration {stats.duration_s:.2f}s < {min_duration_s}s")
    if stats.duration_s > max_duration_s:
        reasons.append(f"duration {stats.duration_s:.2f}s > {max_duration_s}s")
    if stats.snr_db < min_snr_db:
        reasons.append(f"snr {stats.snr_db:.1f}dB < {min_snr_db}dB")
    if stats.peak_db > -1.0:
        reasons.append(f"clipping risk: peak {stats.peak_db:.1f}dBFS")

    passed = len(reasons) == 0

    # Soft score (always computed for ranking, even when failing)
    snr_norm = float(np.clip((stats.snr_db - min_snr_db) / (40.0 - min_snr_db), 0.0, 1.0))

    # Duration factor: triangle function peaking at the middle of the range
    mid = (min_duration_s + max_duration_s) / 2
    span = (max_duration_s - min_duration_s) / 2
    if span > 0:
        dur_factor = max(0.0, 1.0 - abs(stats.duration_s - mid) / span)
    else:
        dur_factor = 1.0

    score = snr_norm * dur_factor
    return float(score), passed, tuple(reasons)


def evaluate_file(
    path: Path,
    min_duration_s: float = MIN_REF_DURATION_S,
    max_duration_s: float = MAX_REF_DURATION_S,
    min_snr_db: float = 15.0,
    min_sr_hz: int = MIN_SAMPLING_RATE_HZ,
    target_sr: int = XTTS_INPUT_SR,
) -> QualityReport:
    """Load an audio file and produce a full quality report.

    Args:
        path: Path to audio file.
        min_duration_s: Min duration in seconds.
        max_duration_s: Max duration in seconds.
        min_snr_db: Min SNR in dB.
        min_sr_hz: Min sample rate in Hz (rejects phone audio).
        target_sr: Sample rate to load at (analysis happens at this rate).

    Returns:
        QualityReport with score, passed flag, and reasons.
    """
    path = Path(path)
    y = load_audio_mono(path, target_sr=target_sr)
    stats = compute_stats(y, target_sr)
    score, passed, reasons = score_segment(
        stats,
        min_duration_s=min_duration_s,
        max_duration_s=max_duration_s,
        min_snr_db=min_snr_db,
        min_sr_hz=min_sr_hz,
    )
    return QualityReport(path=path, stats=stats, score=score, passed=passed, reasons=reasons)


def rank_candidates(reports: list[QualityReport], top_n: int = 10) -> list[QualityReport]:
    """Return the top-N reports that passed, sorted by score descending.

    Args:
        reports: List of QualityReport objects from evaluate_file.
        top_n: Maximum number to return.

    Returns:
        Sorted list, length <= top_n, only including passing reports.
    """
    passing = [r for r in reports if r.passed]
    if not passing:
        logger.warning("No candidates passed quality gates. Returning empty list.")
        return []

    passing.sort(key=lambda r: r.score, reverse=True)
    return passing[:top_n]
