"""voicelegacy — Top-level pipeline orchestration.

Coordinates the end-to-end flow:
    speakerscribe JSON
        → reference corpus build (extract + clean target-speaker segments)
        → quality gating (rank, keep top N)
        → XTTS-v2 inference (text → cloned voice)
        → output + idempotency cache

Designed to be called from a notebook with a single function call per phase.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from voicelegacy.config import (
    MIN_USABLE_REFERENCE_SEGMENTS,
    PipelineConfig,
    WorkspacePaths,
)
from voicelegacy.corpus import build_reference_corpus
from voicelegacy.logging_config import get_logger
from voicelegacy.persistence import (
    compute_run_hash,
    get_synthesis_record,
    hash_reference_set,
    save_synthesis_record,
)
from voicelegacy.quality import QualityReport, evaluate_file, rank_candidates
from voicelegacy.synthesis import load_xtts_model, synthesize_to_file

logger = get_logger()


@dataclass(frozen=True)
class CorpusBuildResult:
    """Outcome of the corpus-build phase."""

    all_wavs: list[Path]
    top_wavs: list[Path]
    reports: list[QualityReport]


@dataclass(frozen=True)
class SynthesisResult:
    """Outcome of a single text → audio synthesis."""

    output_path: Path
    text: str
    reference_set_hash: str
    cached: bool


# ─── Phase 1: build reference corpus ───────────────────────────────
def run_reference_phase(
    paths: WorkspacePaths,
    config: PipelineConfig,
) -> CorpusBuildResult:
    """Build the curated reference corpus from speakerscribe JSONs.

    Steps:
        1. Walk all .json in speakerscribe_out/
        2. Filter to target_speaker segments within duration window
        3. Extract slices, denoise, normalize, write WAVs
        4. Score each WAV via quality module
        5. Rank top_n_segments by score
        6. Write quality report to reports/

    Args:
        paths: Workspace paths.
        config: Top-level pipeline config.

    Returns:
        CorpusBuildResult with all wavs, top-N, and per-segment reports.
    """
    paths.mkdirs()

    if config.force_rebuild_reference:
        logger.warning("force_rebuild_reference=True — wiping reference_corpus/")
        for p in paths.reference_corpus.glob("*.wav"):
            p.unlink()

    # Build (or reuse) WAV segments from JSONs.
    existing = sorted(paths.reference_corpus.glob("*.wav"))
    if existing and not config.force_rebuild_reference:
        logger.info("Found {} existing reference WAVs — skipping build.", len(existing))
        all_wavs = existing
    else:
        all_wavs = build_reference_corpus(paths, config.reference)

    if not all_wavs:
        logger.error("Reference corpus is empty. Cannot proceed.")
        return CorpusBuildResult(all_wavs=[], top_wavs=[], reports=[])

    # Score each WAV.
    # IMPORTANT: the duration bounds used here MUST match the bounds applied
    # at corpus-build time (corpus.filter_segments). Otherwise a segment can
    # pass the corpus filter and silently fail the quality gate, wasting work
    # and producing misleading "reasons" in the report.
    logger.info("Scoring {} candidate segments...", len(all_wavs))
    reports = [
        evaluate_file(
            wav,
            min_duration_s=config.reference.min_segment_duration_s,
            max_duration_s=config.reference.max_segment_duration_s,
            min_snr_db=config.reference.min_snr_db,
        )
        for wav in all_wavs
    ]
    top_reports = rank_candidates(reports, top_n=config.reference.top_n_segments)
    top_wavs = [r.path for r in top_reports]

    # Write report
    _write_quality_report(paths, reports, top_reports)

    logger.info(
        "Reference phase complete: {} total / {} top / {} passed",
        len(all_wavs),
        len(top_wavs),
        sum(1 for r in reports if r.passed),
    )
    return CorpusBuildResult(all_wavs=all_wavs, top_wavs=top_wavs, reports=reports)


def _write_quality_report(
    paths: WorkspacePaths,
    all_reports: list[QualityReport],
    top_reports: list[QualityReport],
) -> None:
    """Persist a JSON quality report to reports/."""
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out = paths.reports / f"reference_quality_{ts}.json"
    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "summary": {
            "total_candidates": len(all_reports),
            "passing": sum(1 for r in all_reports if r.passed),
            "top_selected": len(top_reports),
        },
        "top_segments": [r.to_dict() for r in top_reports],
        "all_segments": [r.to_dict() for r in all_reports],
    }
    out.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    logger.info("Quality report → {}", out)


# ─── Phase 2: synthesize ───────────────────────────────────────────
def _slugify(text: str, max_len: int = 60) -> str:
    """Make a filesystem-safe slug from text (first ~max_len chars)."""
    slug = re.sub(r"[^\w\s-]", "", text, flags=re.UNICODE).strip().lower()
    slug = re.sub(r"[\s-]+", "-", slug)
    return slug[:max_len] or "untitled"


def run_synthesis(
    text: str,
    reference_wavs: list[Path],
    paths: WorkspacePaths,
    config: PipelineConfig,
    output_name: str | None = None,
) -> SynthesisResult:
    """Synthesize one piece of text using the given reference set.

    Idempotent: if an identical (text, references, synthesis_config) combo
    has been generated before and force_resynthesize is False, the cached
    output path is returned without running inference.

    Args:
        text: Text to vocalize. Will be auto-split at sentence boundaries.
        reference_wavs: List of clean reference WAVs (typically top-N from phase 1).
        paths: Workspace paths.
        config: Pipeline config.
        output_name: Optional filename stem. If None, derived from text.

    Returns:
        SynthesisResult with output path and cache info.
    """
    if not reference_wavs:
        raise ValueError("No reference WAVs provided — run phase 1 first.")

    if len(reference_wavs) < MIN_USABLE_REFERENCE_SEGMENTS:
        raise ValueError(
            f"Only {len(reference_wavs)} reference segment(s) provided, but "
            f"voicelegacy requires at least {MIN_USABLE_REFERENCE_SEGMENTS} "
            "for stable zero-shot synthesis. Collect more / better source "
            "audio and re-run phase 1. Override only if you understand the "
            "quality penalty: pass the reference list directly to "
            "synthesis.synthesize_to_file()."
        )

    ref_hash = hash_reference_set(reference_wavs)
    config_json = config.synthesis.model_dump_json()
    run_hash = compute_run_hash(text, ref_hash, config_json)

    # Idempotency check
    if not config.force_resynthesize:
        existing = get_synthesis_record(paths.db_path, run_hash)
        if existing is not None and existing.output_path.exists():
            logger.info("♻️  Cached output → {}", existing.output_path.name)
            return SynthesisResult(
                output_path=existing.output_path,
                text=text,
                reference_set_hash=ref_hash,
                cached=True,
            )

    # Output path
    stem = output_name or f"{_slugify(text)}_{run_hash[:8]}"
    out_path = paths.synthesis_out / f"{stem}.wav"

    # Load model and synthesize
    tts = load_xtts_model(config.synthesis, accept_tos=config.accept_coqui_tos)
    synthesize_to_file(
        tts=tts,
        text=text,
        speaker_wav=reference_wavs,
        output_path=out_path,
        config=config.synthesis,
    )

    save_synthesis_record(
        paths.db_path,
        run_hash=run_hash,
        text=text,
        reference_set=ref_hash,
        output_path=out_path,
        config=json.loads(config_json),
    )
    return SynthesisResult(
        output_path=out_path,
        text=text,
        reference_set_hash=ref_hash,
        cached=False,
    )


def run_batch_synthesis(
    texts: list[str],
    reference_wavs: list[Path],
    paths: WorkspacePaths,
    config: PipelineConfig,
) -> list[SynthesisResult]:
    """Synthesize a list of texts. Loads the model once and reuses it."""
    if not texts:
        logger.warning("Empty texts list — nothing to do.")
        return []
    # Force model load before the loop so first-call latency is observable.
    load_xtts_model(config.synthesis, accept_tos=config.accept_coqui_tos)
    results: list[SynthesisResult] = []
    for i, text in enumerate(texts, 1):
        logger.info("[{}/{}] {!r}...", i, len(texts), text[:60])
        res = run_synthesis(text, reference_wavs, paths, config)
        results.append(res)
    return results
