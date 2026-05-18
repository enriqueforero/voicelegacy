"""voicelegacy — Build a reference corpus from speakerscribe outputs.

speakerscribe emits a structured JSON with diarized segments. This module
walks the JSON, extracts segments belonging to the target speaker, slices
them from the original audio, cleans them up, and writes them to disk as
WAVs ready for XTTS-v2 conditioning.

Expected speakerscribe JSON shape (the relevant subset):
    {
      "source_audio": "interview_01.mp3",
      "language_detected": "es",
      "segments": [
        {"start": 12.34, "end": 18.91, "speaker": "SPEAKER_00", "text": "..."},
        ...
      ]
    }

Anything else in the JSON is ignored.
"""

from __future__ import annotations

import json
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from voicelegacy.audio import (
    denoise,
    load_audio_mono,
    loudness_normalize,
    save_wav,
    slice_segment,
    trim_silence,
)
from voicelegacy.config import XTTS_INPUT_SR, ReferenceConfig, WorkspacePaths
from voicelegacy.logging_config import get_logger

logger = get_logger()


@dataclass(frozen=True)
class SegmentRef:
    """One diarized segment from speakerscribe, after light validation."""

    source_audio: Path
    start_s: float
    end_s: float
    speaker: str
    text: str

    @property
    def duration_s(self) -> float:
        return self.end_s - self.start_s


# ─── JSON parsing ──────────────────────────────────────────────────
def load_speakerscribe_json(json_path: Path, audio_root: Path | None = None) -> list[SegmentRef]:
    """Parse a speakerscribe .json and return diarized segments.

    Args:
        json_path: Path to a `.json` file produced by speakerscribe.
        audio_root: Directory where the source audio lives. If None, defaults
            to the same directory as json_path's parent's parent (matching
            the speakerscribe workspace convention).

    Returns:
        List of SegmentRef objects. May be empty if no segments parse.

    Raises:
        FileNotFoundError: If json_path or referenced source audio is missing.
        ValueError: If JSON is malformed.
    """
    json_path = Path(json_path)
    if not json_path.exists():
        raise FileNotFoundError(f"speakerscribe JSON not found: {json_path}")

    try:
        data = json.loads(json_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"Malformed JSON in {json_path.name}: {exc}") from exc

    source_name = data.get("source_audio") or data.get("audio_file") or json_path.stem
    if audio_root is None:
        # speakerscribe convention: transcripts/<file>.json → data/<file>
        audio_root = json_path.parent.parent / "data"

    source_audio = Path(audio_root) / source_name
    # If extension was stripped, try common ones
    if not source_audio.exists():
        for ext in (".wav", ".mp3", ".m4a", ".flac", ".ogg", ".mp4"):
            candidate = source_audio.with_suffix(ext)
            if candidate.exists():
                source_audio = candidate
                break

    raw_segments: Iterable[dict] = data.get("segments") or []
    out: list[SegmentRef] = []
    for s in raw_segments:
        try:
            seg = SegmentRef(
                source_audio=source_audio,
                start_s=float(s["start"]),
                end_s=float(s["end"]),
                speaker=str(s.get("speaker", "UNKNOWN")),
                text=str(s.get("text", "")).strip(),
            )
        except (KeyError, ValueError, TypeError) as exc:
            logger.warning("Skipping malformed segment in {}: {}", json_path.name, exc)
            continue
        out.append(seg)

    logger.info("Parsed {} segments from {}", len(out), json_path.name)
    return out


# ─── Filtering ─────────────────────────────────────────────────────
def filter_segments(
    segments: list[SegmentRef],
    target_speaker: str,
    min_duration_s: float,
    max_duration_s: float,
) -> list[SegmentRef]:
    """Filter to segments of the target speaker within the duration window.

    Args:
        segments: All segments from one or more speakerscribe JSONs.
        target_speaker: Speaker label to keep (e.g. 'SPEAKER_00').
        min_duration_s: Minimum segment length in seconds.
        max_duration_s: Maximum segment length in seconds.

    Returns:
        Filtered list, sorted by source then by start time.
    """
    out = [
        s
        for s in segments
        if s.speaker == target_speaker and min_duration_s <= s.duration_s <= max_duration_s
    ]
    out.sort(key=lambda s: (str(s.source_audio), s.start_s))
    logger.info(
        "Kept {}/{} segments for speaker '{}' (duration {}-{}s)",
        len(out),
        len(segments),
        target_speaker,
        min_duration_s,
        max_duration_s,
    )
    return out


# ─── Extraction ────────────────────────────────────────────────────
def extract_segments_to_wav(
    segments: list[SegmentRef],
    out_dir: Path,
    config: ReferenceConfig,
    target_sr: int = XTTS_INPUT_SR,
) -> list[Path]:
    """Slice each segment from its source audio and write a clean WAV.

    Caches the loaded source audio across segments from the same file —
    interviews are typically long single-file recordings, so this avoids
    redundant decoding.

    Args:
        segments: Filtered list of SegmentRef objects.
        out_dir: Directory to write extracted WAVs to.
        config: ReferenceConfig controlling cleanup parameters.
        target_sr: Sample rate to load and save at.

    Returns:
        List of paths to written WAV files. Same order as input segments.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    cache: dict[Path, np.ndarray] = {}
    written: list[Path] = []

    for idx, seg in enumerate(segments):
        # Load (and cache) the source audio
        if seg.source_audio not in cache:
            try:
                cache[seg.source_audio] = load_audio_mono(seg.source_audio, target_sr=target_sr)
                logger.info(
                    "Loaded source: {} ({:,} samples)",
                    seg.source_audio.name,
                    len(cache[seg.source_audio]),
                )
            except FileNotFoundError:
                logger.warning("Source audio missing, skipping segment: {}", seg.source_audio)
                continue
            except Exception as exc:
                logger.error("Failed to load {}: {}", seg.source_audio.name, exc)
                continue

        full_y = cache[seg.source_audio]
        try:
            y_seg = slice_segment(full_y, target_sr, seg.start_s, seg.end_s, pad_s=0.05)
        except ValueError as exc:
            logger.warning("Bad segment bounds, skipping: {}", exc)
            continue

        if config.apply_denoise:
            y_seg = denoise(y_seg, target_sr)

        y_seg = trim_silence(y_seg)
        if y_seg.size < int(target_sr * 1.0):  # less than 1s after trim — drop
            logger.warning("Segment too short after trim, skipping (idx={})", idx)
            continue

        y_seg = loudness_normalize(y_seg, target_sr, target_lufs=config.target_loudness_lufs)

        out_path = out_dir / f"{seg.source_audio.stem}_{idx:04d}_{seg.start_s:08.2f}.wav"
        save_wav(out_path, y_seg, target_sr)
        written.append(out_path)

    logger.info("Wrote {} reference WAVs to {}", len(written), out_dir)
    return written


# ─── End-to-end ────────────────────────────────────────────────────
def build_reference_corpus(
    paths: WorkspacePaths,
    config: ReferenceConfig,
) -> list[Path]:
    """End-to-end: walk speakerscribe JSONs → write reference WAVs.

    Args:
        paths: Workspace paths object.
        config: ReferenceConfig.

    Returns:
        List of paths to all written reference WAVs.
    """
    paths.mkdirs()
    json_files = sorted(paths.speakerscribe_out.glob("*.json"))
    if not json_files:
        logger.warning(
            "No speakerscribe JSONs found in {}. Run speakerscribe first.",
            paths.speakerscribe_out,
        )
        return []

    all_segments: list[SegmentRef] = []
    for jf in json_files:
        all_segments.extend(load_speakerscribe_json(jf, audio_root=paths.interviews_raw))

    filtered = filter_segments(
        all_segments,
        target_speaker=config.target_speaker_label,
        min_duration_s=config.min_segment_duration_s,
        max_duration_s=config.max_segment_duration_s,
    )

    return extract_segments_to_wav(
        filtered,
        out_dir=paths.reference_corpus,
        config=config,
    )
