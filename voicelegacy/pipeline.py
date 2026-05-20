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
import shutil
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from voicelegacy.audio import _ffmpeg_available, convert_directory_to_wav
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
    package_version,
    save_synthesis_record,
)
from voicelegacy.quality import QualityReport, evaluate_file, rank_candidates
from voicelegacy.synthesis import load_xtts_model, synthesize_to_file
from voicelegacy.telemetry import timed_step
from voicelegacy.text_strategy import plan_text_synthesis

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
    metadata_path: Path | None = None
    similarity_score: float | None = None


def _backup_reference_corpus(paths: WorkspacePaths) -> Path | None:
    """Move existing reference WAVs aside before a forced rebuild.

    Destructive rebuilds are unacceptable in a legacy-audio workflow. If the
    corpus contains WAVs, move them to reference_corpus_backup_<UTC>/ before
    rebuilding so manual cleanup work is not lost.
    """
    existing = sorted(paths.reference_corpus.glob("*.wav"))
    if not existing:
        logger.info("force_rebuild_reference=True — no existing reference WAVs to back up.")
        return None

    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    backup_dir = paths.workspace / f"reference_corpus_backup_{ts}"
    backup_dir.mkdir(parents=True, exist_ok=False)
    for wav in existing:
        shutil.move(str(wav), str(backup_dir / wav.name))
    logger.warning(
        "force_rebuild_reference=True — moved {} reference WAV(s) to {}",
        len(existing),
        backup_dir,
    )
    return backup_dir


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

    # Pre-flight: auto-convert any non-WAV media (mp4, m4a, mkv, webm, ...) in
    # interviews_raw/ to mono WAV at XTTS-v2's expected rate. This replaces
    # the hand-edited Cell 17 of the legacy notebook. corpus.load_speakerscribe_json
    # cannot decode container formats, only WAV. ffmpeg is required.
    if _ffmpeg_available():
        converted = convert_directory_to_wav(paths.interviews_raw)
        if converted:
            logger.info("Auto-converted {} container file(s) to WAV", len(converted))
    else:
        logger.warning(
            "ffmpeg not on PATH — container files in interviews_raw/ will be "
            "skipped. Install ffmpeg to enable mp4/m4a/mkv/webm/aac inputs."
        )

    if config.force_rebuild_reference:
        _backup_reference_corpus(paths)

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
    reports = [evaluate_file(wav, config=config.reference) for wav in all_wavs]
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
            "rejection_reason_counts": _reason_counts([r for r in all_reports if not r.passed]),
            "warning_reason_counts": _reason_counts(
                [r for r in all_reports if r.passed and r.reasons]
            ),
        },
        "top_segments": [r.to_dict() for r in top_reports],
        "all_segments": [r.to_dict() for r in all_reports],
    }
    out.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    logger.info("Quality report → {}", out)


def _reason_counts(reports: list[QualityReport]) -> dict[str, int]:
    """Count rejection/warning reasons across quality reports."""
    counts: dict[str, int] = {}
    for report in reports:
        for reason in report.reasons:
            counts[reason] = counts.get(reason, 0) + 1
    return dict(sorted(counts.items(), key=lambda kv: (-kv[1], kv[0])))


def _reference_quality_summary(
    reference_wavs: list[Path],
    config: PipelineConfig,
) -> dict[str, object]:
    """Evaluate reference WAVs and return sidecar-ready source-quality metadata."""
    reports: list[QualityReport] = []
    for wav in reference_wavs:
        try:
            reports.append(evaluate_file(wav, config=config.reference))
        except Exception as exc:  # pragma: no cover - defensive; evaluate_file already degrades
            logger.warning("Could not evaluate reference {} for sidecar: {}", wav.name, exc)

    snrs = [r.stats.snr_db for r in reports if r.stats is not None]
    sample_rates = [r.stats.sample_rate for r in reports if r.stats is not None]
    mean_snr = round(sum(snrs) / len(snrs), 2) if snrs else None
    min_sample_rate = min(sample_rates) if sample_rates else None
    degraded = bool(
        config.reference.min_snr_db < 10.0
        or (mean_snr is not None and mean_snr < 10.0)
        or (min_sample_rate is not None and min_sample_rate < 16000)
    )
    return {
        "reference_count": len(reference_wavs),
        "mean_dynamic_range_db": mean_snr,
        "min_sample_rate_hz": min_sample_rate,
        "configured_min_snr_db": config.reference.min_snr_db,
        "degraded_mode": degraded,
        "degraded_reason": ("source_quality_below_recommended_threshold" if degraded else None),
        "quality_reports": [r.to_dict() for r in reports],
    }


def _compute_optional_similarity(
    output_path: Path,
    reference_wavs: list[Path],
    enabled: bool,
) -> tuple[float | None, dict[str, object]]:
    """Run optional speaker-similarity scoring without making synthesis fragile."""
    if not enabled:
        return None, {"status": "disabled"}
    try:
        from voicelegacy.similarity import compute_similarity

        report = compute_similarity(output_path, reference_wavs)
        return report.score, {
            "status": "ok",
            **report.to_dict(),
            "quality_band": report.quality_band(),
        }
    except ImportError as exc:
        return None, {"status": "skipped", "reason": str(exc)}
    except Exception as exc:  # pragma: no cover - defensive for optional dependency failures
        logger.warning("Similarity scoring failed for {}: {}", output_path.name, exc)
        return None, {"status": "failed", "reason": str(exc)}


def _write_synthesis_sidecar(
    output_path: Path,
    *,
    text: str,
    reference_wavs: list[Path],
    reference_set_hash: str,
    run_hash: str,
    config: PipelineConfig,
    cached: bool,
    similarity_score: float | None,
    similarity_payload: dict[str, object],
) -> Path:
    """Write JSON metadata next to a synthesized WAV.

    This sidecar is the audit trail: it records the exact synthesis config, seed,
    reference set, source-quality summary, and whether the output was produced
    from degraded source material. A legacy audio file without this is not
    reproducible enough for production use.
    """
    metadata_path = output_path.with_suffix(".json")
    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "output_path": str(output_path),
        "text": text,
        "text_plan": plan_text_synthesis(text, config.synthesis).to_dict(),
        "cached": cached,
        "run_hash": run_hash,
        "voicelegacy_version": package_version(),
        "reference_set_hash": reference_set_hash,
        "reference_wavs": [str(p) for p in reference_wavs],
        "synthesis_config": config.synthesis.model_dump(mode="json"),
        "reference_config": config.reference.model_dump(mode="json"),
        "source_quality": _reference_quality_summary(reference_wavs, config),
        "similarity": similarity_payload,
        "speaker_similarity_score": similarity_score,
    }
    metadata_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    logger.info("Synthesis metadata → {}", metadata_path)
    return metadata_path


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
            metadata_path = existing.output_path.with_suffix(".json")
            similarity_score = None
            if metadata_path.exists():
                try:
                    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
                    similarity_score = metadata.get("speaker_similarity_score")
                except json.JSONDecodeError:
                    similarity_score = None
            return SynthesisResult(
                output_path=existing.output_path,
                text=text,
                reference_set_hash=ref_hash,
                cached=True,
                metadata_path=metadata_path if metadata_path.exists() else None,
                similarity_score=similarity_score if isinstance(similarity_score, float) else None,
            )

    # Output path
    stem = output_name or f"{_slugify(text)}_{run_hash[:8]}"
    out_path = paths.synthesis_out / f"{stem}.wav"

    # Load model and synthesize with timing/VRAM telemetry.
    with timed_step("load_xtts_model"):
        tts = load_xtts_model(config.synthesis, accept_tos=config.accept_coqui_tos)
    with timed_step(f"synthesize:{out_path.name}"):
        synthesize_to_file(
            tts=tts,
            text=text,
            speaker_wav=reference_wavs,
            output_path=out_path,
            config=config.synthesis,
        )

    with timed_step(f"similarity:{out_path.name}"):
        similarity_score, similarity_payload = _compute_optional_similarity(
            out_path, reference_wavs, enabled=config.synthesis.compute_similarity
        )
    metadata_path = _write_synthesis_sidecar(
        out_path,
        text=text,
        reference_wavs=reference_wavs,
        reference_set_hash=ref_hash,
        run_hash=run_hash,
        config=config,
        cached=False,
        similarity_score=similarity_score,
        similarity_payload=similarity_payload,
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
        metadata_path=metadata_path,
        similarity_score=similarity_score,
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
    with timed_step("load_xtts_model:batch_preload"):
        load_xtts_model(config.synthesis, accept_tos=config.accept_coqui_tos)
    results: list[SynthesisResult] = []
    for i, text in enumerate(texts, 1):
        logger.info("[{}/{}] {!r}...", i, len(texts), text[:60])
        res = run_synthesis(text, reference_wavs, paths, config)
        results.append(res)
    return results
