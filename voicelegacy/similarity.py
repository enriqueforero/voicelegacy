"""voicelegacy — Speaker similarity scoring for synthetic outputs.

The single most expensive blindspot of the original pipeline: every
quality metric measured the *input* (reference corpus), but nothing
measured the *output* (synthetic clone). "Does it sound like her?"
was decided by ear, every time, by a human listener. That doesn't
scale, isn't reproducible, and depends on whose ear validates.

This module fixes that. It computes a numeric similarity score in
[0, 1] between the synthetic output and the reference corpus, using
a pre-trained speaker encoder (Resemblyzer, GE2E loss).

Why Resemblyzer specifically:
    - Apache-2.0, small (~50 MB model file).
    - No coqui-tts / speechbrain heavyweight dependencies.
    - GE2E-trained: cosine similarity is the canonical comparison.
    - Empirically validated for cross-clip speaker verification at
      utterance-level (Wan et al. 2017).

Reference: https://github.com/resemble-ai/Resemblyzer

Typical score ranges (informational, not enforced):
    > 0.85   → very confident same-speaker. Rare for zero-shot clones.
    0.75–0.85 → confident same-speaker. Good legacy-quality clone.
    0.60–0.75 → marginal. The clone has the speaker's character but
                noticeable drift or contamination.
    < 0.60   → likely different speaker / failed clone. Investigate.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from voicelegacy.logging_config import get_logger

logger = get_logger()

# Module-level cache of the loaded encoder. Loading is ~3-5 s and downloads
# weights on first use; we don't want to pay that per evaluation.
_ENCODER_CACHE: dict[str, Any] = {}


@dataclass(frozen=True)
class SimilarityReport:
    """Result of comparing a synthetic output against a reference set.

    Attributes:
        output_path: The synthesized WAV that was scored.
        score: Cosine similarity ∈ [0, 1] between the output embedding
            and the centroid of the reference embeddings. Higher = more
            similar speaker.
        n_references: How many reference clips contributed to the centroid.
        per_reference_scores: Cosine similarity of the output against each
            individual reference. Useful to detect contaminated references
            (one outlier drags the centroid).
        encoder_name: Identifier of the speaker encoder used (for audit).
    """

    output_path: Path
    score: float
    n_references: int
    per_reference_scores: tuple[float, ...]
    encoder_name: str

    def to_dict(self) -> dict[str, object]:
        """Serialize to a JSON-friendly dict."""
        return {
            "output_path": str(self.output_path),
            "speaker_similarity_score": round(self.score, 4),
            "n_references": self.n_references,
            "per_reference_scores": [round(s, 4) for s in self.per_reference_scores],
            "encoder": self.encoder_name,
        }

    def quality_band(self) -> str:
        """Map the numeric score to a human-readable band.

        Bands are documented in the module docstring. Not a substitute for
        listening, but a quick triage signal.
        """
        if self.score >= 0.85:
            return "very_high"
        if self.score >= 0.75:
            return "high"
        if self.score >= 0.60:
            return "marginal"
        return "low"


# ─── Encoder loading ───────────────────────────────────────────────
def _load_encoder() -> Any:
    """Load (or fetch from cache) a Resemblyzer VoiceEncoder.

    Raises:
        ImportError: If resemblyzer is not installed. The caller decides
            whether to skip similarity scoring or fail hard. We don't
            force-install on every machine because the model download is
            ~50 MB.
    """
    if "resemblyzer" in _ENCODER_CACHE:
        return _ENCODER_CACHE["resemblyzer"]

    try:
        from resemblyzer import VoiceEncoder
    except ImportError as exc:
        raise ImportError(
            "resemblyzer is not installed. To enable speaker similarity "
            "scoring, run: pip install resemblyzer. See "
            "https://github.com/resemble-ai/Resemblyzer for details. "
            "This is an OPTIONAL dependency — the rest of voicelegacy "
            "works without it."
        ) from exc

    logger.info("Loading Resemblyzer VoiceEncoder (first call may download weights)...")
    encoder = VoiceEncoder()
    _ENCODER_CACHE["resemblyzer"] = encoder
    logger.info("Resemblyzer encoder ready.")
    return encoder


def release_encoder() -> None:
    """Evict the cached encoder. Useful before loading another large model."""
    _ENCODER_CACHE.clear()
    import gc

    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        pass
    logger.info("Similarity encoder released.")


# ─── Core computation ──────────────────────────────────────────────
def _embed_one(encoder: Any, wav_path: Path) -> np.ndarray:
    """Embed a single WAV file with the given encoder.

    Returns:
        1-D float numpy array (the speaker embedding vector).
    """
    from resemblyzer import preprocess_wav

    # preprocess_wav handles resampling to 16 kHz and VAD-trimming internally.
    wav = preprocess_wav(Path(wav_path))
    return encoder.embed_utterance(wav)


def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity in [-1, 1], guarded against zero-norm vectors."""
    na = float(np.linalg.norm(a))
    nb = float(np.linalg.norm(b))
    if na < 1e-12 or nb < 1e-12:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def compute_similarity(
    output_wav: Path,
    reference_wavs: list[Path],
    encoder: Any | None = None,
) -> SimilarityReport:
    """Score how similar `output_wav` sounds to the reference set.

    The reference centroid is the mean of per-reference embeddings. Comparing
    the output against the centroid (instead of any single reference) makes
    the score robust to small clean variations across references — it
    captures the speaker's "average" voice.

    Args:
        output_wav: Path to the synthesized WAV (e.g. from synthesize_to_file).
        reference_wavs: Paths to the reference corpus WAVs (typically the
            top-N from the quality phase). At least 1 required.
        encoder: Optional pre-loaded encoder. If None, loads/caches the
            module-level default.

    Returns:
        SimilarityReport with score, per-reference breakdown, and metadata.

    Raises:
        ValueError: If reference_wavs is empty or output_wav does not exist.
        ImportError: If resemblyzer is not installed.
    """
    output_wav = Path(output_wav)
    if not output_wav.exists():
        raise FileNotFoundError(f"Output WAV not found: {output_wav}")
    if not reference_wavs:
        raise ValueError("At least one reference WAV is required.")
    for r in reference_wavs:
        if not Path(r).exists():
            raise FileNotFoundError(f"Reference WAV not found: {r}")

    enc = encoder if encoder is not None else _load_encoder()

    # Embed each reference. We could batch this, but the reference set is
    # typically small (3-10 files) so the overhead is not worth the code.
    ref_embeddings = [_embed_one(enc, r) for r in reference_wavs]
    centroid = np.mean(np.stack(ref_embeddings, axis=0), axis=0)

    out_embedding = _embed_one(enc, output_wav)

    # Resemblyzer embeddings come L2-normalized; cosine reduces to dot.
    # But we don't trust that assumption blindly — use the explicit form.
    score = _cosine_similarity(out_embedding, centroid)
    per_ref = tuple(_cosine_similarity(out_embedding, e) for e in ref_embeddings)

    # Resemblyzer scores are in [0, 1] for same-modality utterances (the
    # encoder is trained so within-speaker cosine is positive). Clamp
    # defensively to remove the [-1, 0) range that no real speaker pair hits.
    score = max(0.0, score)

    return SimilarityReport(
        output_path=output_wav,
        score=score,
        n_references=len(reference_wavs),
        per_reference_scores=per_ref,
        encoder_name="resemblyzer_v0",
    )


def compute_similarity_batch(
    outputs_and_refs: list[tuple[Path, list[Path]]],
) -> list[SimilarityReport]:
    """Score multiple outputs in one pass, reusing the loaded encoder.

    Args:
        outputs_and_refs: List of (output_wav, reference_wavs) pairs.

    Returns:
        List of SimilarityReport, same order as input.
    """
    if not outputs_and_refs:
        return []

    encoder = _load_encoder()
    results: list[SimilarityReport] = []
    for out_path, refs in outputs_and_refs:
        try:
            report = compute_similarity(out_path, refs, encoder=encoder)
            logger.info(
                "Similarity {} = {:.3f} ({}, {} refs)",
                out_path.name,
                report.score,
                report.quality_band(),
                report.n_references,
            )
            results.append(report)
        except (FileNotFoundError, ValueError) as exc:
            logger.warning("Similarity skipped for {}: {}", out_path.name, exc)
    return results


def is_available() -> bool:
    """Return True if resemblyzer is importable. Useful for graceful skip."""
    try:
        import resemblyzer  # noqa: F401

        return True
    except ImportError:
        return False
