"""Reproducible denoise comparison harness.

DeepFilterNet is intentionally optional. The correct production stance is to
compare it against the existing noisereduce path on real source audio before
making it a default dependency.
"""

from __future__ import annotations

import json
import shutil
import subprocess
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import soundfile as sf

from voicelegacy.audio import (
    AudioStats,
    compute_stats,
    denoise,
    load_audio_mono,
    loudness_normalize,
)
from voicelegacy.config import XTTS_INPUT_SR
from voicelegacy.logging_config import get_logger
from voicelegacy.telemetry import timed_step

logger = get_logger()


@dataclass(frozen=True)
class DenoiseCandidate:
    """One denoise method output for one source file."""

    source_path: Path
    method: str
    status: str
    output_path: Path | None
    stats: AudioStats | None
    reason: str | None = None

    def to_dict(self) -> dict[str, object]:
        """Serialize to JSON-friendly dict."""
        payload: dict[str, object] = {
            "source_path": str(self.source_path),
            "method": self.method,
            "status": self.status,
            "output_path": str(self.output_path) if self.output_path else None,
            "reason": self.reason,
        }
        if self.stats is not None:
            payload["stats"] = {
                "duration_s": self.stats.duration_s,
                "sample_rate": self.stats.sample_rate,
                "n_channels": self.stats.n_channels,
                "rms_db": self.stats.rms_db,
                "peak_db": self.stats.peak_db,
                "snr_db": self.stats.snr_db,
                "dynamic_range_db": self.stats.snr_db,
            }
        return payload


def _safe_stem(path: Path) -> str:
    return path.stem.replace(" ", "_")


def _write_noisereduce_candidate(source: Path, output_dir: Path) -> DenoiseCandidate:
    """Generate a noisereduce candidate using voicelegacy's current path."""
    output_path = output_dir / f"{_safe_stem(source)}__noisereduce.wav"
    with timed_step(f"denoise:noisereduce:{source.name}"):
        y, original_sr = load_audio_mono(source, target_sr=XTTS_INPUT_SR)
        y_clean = denoise(y, XTTS_INPUT_SR, stationary=False)
        y_clean = loudness_normalize(y_clean, XTTS_INPUT_SR, target_lufs=-23.0)
        sf.write(output_path, y_clean, XTTS_INPUT_SR, subtype="PCM_16")
        stats = compute_stats(y_clean, XTTS_INPUT_SR, original_sr=original_sr)
    return DenoiseCandidate(source, "noisereduce_nonstationary", "ok", output_path, stats)


def _find_deepfilter_output(source: Path, output_dir: Path) -> Path | None:
    """Find DeepFilterNet's output without depending on one filename convention."""
    candidates = sorted(output_dir.glob(f"*{source.stem}*.wav"))
    if candidates:
        return candidates[-1]
    direct = output_dir / source.name
    return direct if direct.exists() else None


def _run_deepfilter_candidate(source: Path, output_dir: Path) -> DenoiseCandidate:
    """Run DeepFilterNet CLI if installed; otherwise return a skipped candidate."""
    exe = shutil.which("deepFilter")
    if exe is None:
        return DenoiseCandidate(
            source,
            "deepfilternet",
            "skipped",
            None,
            None,
            reason="deepFilter CLI not installed; run pip install deepfilternet first",
        )

    method_dir = output_dir / "deepfilternet"
    method_dir.mkdir(parents=True, exist_ok=True)
    cmd = [exe, "--output-dir", str(method_dir), str(source)]
    with timed_step(f"denoise:deepfilternet:{source.name}"):
        proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
    if proc.returncode != 0:
        return DenoiseCandidate(
            source,
            "deepfilternet",
            "failed",
            None,
            None,
            reason=(proc.stderr or proc.stdout or "deepFilter failed").strip(),
        )
    enhanced = _find_deepfilter_output(source, method_dir)
    if enhanced is None:
        return DenoiseCandidate(
            source,
            "deepfilternet",
            "failed",
            None,
            None,
            reason="deepFilter completed but no WAV output was found",
        )
    y, original_sr = load_audio_mono(enhanced, target_sr=XTTS_INPUT_SR)
    stats = compute_stats(y, XTTS_INPUT_SR, original_sr=original_sr)
    return DenoiseCandidate(source, "deepfilternet", "ok", enhanced, stats)


def evaluate_denoise_methods(
    audio_files: list[Path],
    output_dir: Path,
    *,
    include_deepfilter: bool = False,
) -> dict[str, object]:
    """Generate comparable denoise candidates and a JSON-ready report.

    Args:
        audio_files: Real sample files to evaluate. Use 3-5 representative files:
            clean, moderate noise, heavy noise, phone-codec, and long interview.
        output_dir: Directory for enhanced WAVs and report JSON.
        include_deepfilter: If True, also attempts DeepFilterNet via the
            ``deepFilter`` CLI. If not installed, records a skipped row.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    candidates: list[DenoiseCandidate] = []
    for raw in audio_files:
        source = Path(raw)
        if not source.exists():
            candidates.append(
                DenoiseCandidate(source, "input", "failed", None, None, "source file not found")
            )
            continue
        try:
            candidates.append(_write_noisereduce_candidate(source, output_dir))
        except Exception as exc:  # pragma: no cover - defensive path for real media edge cases
            candidates.append(
                DenoiseCandidate(
                    source, "noisereduce_nonstationary", "failed", None, None, str(exc)
                )
            )
        if include_deepfilter:
            candidates.append(_run_deepfilter_candidate(source, output_dir))

    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "recommended_sample_set": [
            "clean_recording",
            "moderate_noise",
            "heavy_noise_or_crosstalk",
            "phone_codec_or_low_sample_rate",
            "long_interview_excerpt",
        ],
        "decision_rule": (
            "DeepFilterNet should become default only if it improves listening quality "
            "and downstream speaker_similarity_score without adding speech artifacts."
        ),
        "candidates": [c.to_dict() for c in candidates],
    }
    report_path = (
        output_dir / f"denoise_evaluation_{datetime.now(timezone.utc):%Y%m%dT%H%M%SZ}.json"
    )
    report_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    logger.info("Denoise evaluation report → {}", report_path)
    payload["report_path"] = str(report_path)
    return payload
