"""voicelegacy — Fine-tuning dataset assembly + corpus coherence validation.

Two responsibilities, both born from a real audit:

1. ``build_finetune_dataset``: turn a ``reference_corpus/`` (clean WAVs + their
   ``.txt`` transcript sidecars, written by ``corpus.extract_segments_to_wav``)
   into the LJSpeech-like layout XTTS-v2 fine-tuning expects (``wavs/`` +
   ``metadata_train.csv`` + ``metadata_eval.csv``).

   This replaces the previous fragile approach where notebooks tried to
   reconstruct the WAV<->text pairing by parsing filenames with a pattern that
   did NOT match what ``extract_segments_to_wav`` actually wrote — producing an
   empty dataset and a confusing ``RuntimeError``. Pairing now reads the sidecar
   adjacent to each WAV: robust by construction, no filename parsing.

2. ``validate_corpus_coherence``: when the target speaker was identified
   per-interview by hand (the bridge workflow), one mislabeled interview
   silently contaminates the corpus with the wrong voice. This function embeds
   every WAV with Resemblyzer and flags clips whose embedding is far from the
   corpus centroid — catching the contamination BEFORE hours of fine-tuning.
"""

from __future__ import annotations

import csv
import random
import shutil
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from voicelegacy.logging_config import get_logger

logger = get_logger()


# ─── Fine-tuning dataset assembly ──────────────────────────────────────────
@dataclass(frozen=True)
class DatasetBuildResult:
    """Outcome of assembling an LJSpeech-like fine-tuning dataset.

    Attributes:
        dataset_dir: Root directory of the built dataset.
        n_train: Number of training rows.
        n_eval: Number of eval rows.
        n_skipped_no_text: WAVs skipped because no transcript sidecar existed.
        total_wavs_seen: Total WAVs found in the corpus.
    """

    dataset_dir: Path
    n_train: int
    n_eval: int
    n_skipped_no_text: int
    total_wavs_seen: int

    @property
    def n_total(self) -> int:
        """Total paired rows (train + eval)."""
        return self.n_train + self.n_eval


def build_finetune_dataset(
    reference_corpus: Path,
    dataset_dir: Path,
    speaker_name: str = "target_speaker",
    eval_fraction: float = 0.1,
    min_text_chars: int = 10,
    seed: int = 42,
) -> DatasetBuildResult:
    """Assemble an LJSpeech-like dataset from a reference corpus + sidecars.

    Reads each ``*.wav`` in ``reference_corpus`` and its adjacent ``*.txt``
    transcript sidecar (written by ``extract_segments_to_wav``). Copies the
    WAVs into ``dataset_dir/wavs/`` and writes ``metadata_train.csv`` and
    ``metadata_eval.csv`` with the ``audio_file|text|speaker_name`` format.

    Args:
        reference_corpus: Directory of clean WAVs + ``.txt`` sidecars.
        dataset_dir: Destination root for the dataset.
        speaker_name: Speaker tag written in the CSV third column.
        eval_fraction: Fraction held out for eval. Default 0.1 (90/10 split).
        min_text_chars: Skip clips whose transcript is shorter than this.
        seed: RNG seed for the deterministic shuffle/split.

    Returns:
        DatasetBuildResult with row counts and skip diagnostics.

    Raises:
        FileNotFoundError: If ``reference_corpus`` does not exist.
        ValueError: If no WAV has a usable transcript sidecar (empty dataset).
    """
    reference_corpus = Path(reference_corpus)
    if not reference_corpus.is_dir():
        raise FileNotFoundError(f"reference_corpus not found: {reference_corpus}")
    if not 0.0 < eval_fraction < 1.0:
        raise ValueError(f"eval_fraction must be in (0, 1), got {eval_fraction}")

    dataset_dir = Path(dataset_dir)
    wavs_dir = dataset_dir / "wavs"
    wavs_dir.mkdir(parents=True, exist_ok=True)

    wavs = sorted(reference_corpus.glob("*.wav"))
    rows: list[tuple[str, str, str]] = []
    skipped = 0
    for wav in wavs:
        sidecar = wav.with_suffix(".txt")
        if not sidecar.exists():
            skipped += 1
            continue
        text = sidecar.read_text(encoding="utf-8").strip()
        if len(text) < min_text_chars:
            skipped += 1
            continue
        dest = wavs_dir / wav.name
        if not dest.exists():
            shutil.copy(str(wav), str(dest))
        rows.append((f"wavs/{wav.name}", text, speaker_name))

    if not rows:
        raise ValueError(
            f"No WAV in {reference_corpus} had a usable transcript sidecar "
            f"({len(wavs)} WAVs seen, {skipped} skipped). Did you build the "
            "corpus with voicelegacy >= 0.3.4? Older corpora lack .txt sidecars; "
            "re-run extract_segments_to_wav to generate them."
        )

    rng = random.Random(seed)
    rng.shuffle(rows)
    cut = max(1, int(len(rows) * (1.0 - eval_fraction)))
    train_rows, eval_rows = rows[:cut], rows[cut:]
    # Guarantee at least one eval row when there are >= 2 rows.
    if not eval_rows and len(rows) >= 2:
        eval_rows = [train_rows.pop()]

    _write_metadata_csv(dataset_dir / "metadata_train.csv", train_rows)
    _write_metadata_csv(dataset_dir / "metadata_eval.csv", eval_rows)

    logger.info(
        "Fine-tune dataset: {} train + {} eval rows ({} WAVs skipped, no text)",
        len(train_rows),
        len(eval_rows),
        skipped,
    )
    return DatasetBuildResult(
        dataset_dir=dataset_dir,
        n_train=len(train_rows),
        n_eval=len(eval_rows),
        n_skipped_no_text=skipped,
        total_wavs_seen=len(wavs),
    )


def _write_metadata_csv(path: Path, rows: list[tuple[str, str, str]]) -> None:
    """Write an LJSpeech-like metadata CSV with a header."""
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f, delimiter="|")
        writer.writerow(["audio_file", "text", "speaker_name"])
        writer.writerows(rows)


# ─── Corpus coherence validation (Resemblyzer) ─────────────────────────────
@dataclass(frozen=True)
class CoherenceResult:
    """Result of checking that all corpus WAVs are the same speaker.

    Attributes:
        n_wavs: Number of WAVs embedded.
        mean_similarity: Mean cosine similarity of each clip to the centroid.
        min_similarity: Lowest clip-to-centroid similarity.
        outliers: List of (wav_path, similarity) below the threshold.
        threshold: The similarity threshold used.
        is_coherent: True if no clip fell below the threshold.
    """

    n_wavs: int
    mean_similarity: float
    min_similarity: float
    outliers: list[tuple[Path, float]]
    threshold: float
    is_coherent: bool

    def to_dict(self) -> dict[str, object]:
        """JSON-friendly summary for audit reports."""
        return {
            "n_wavs": self.n_wavs,
            "mean_similarity": round(self.mean_similarity, 4),
            "min_similarity": round(self.min_similarity, 4),
            "threshold": self.threshold,
            "is_coherent": self.is_coherent,
            "outliers": [{"wav": str(p), "similarity": round(s, 4)} for p, s in self.outliers],
        }


def validate_corpus_coherence(
    reference_corpus: Path,
    threshold: float = 0.70,
    max_wavs: int = 200,
    encoder: object | None = None,
) -> CoherenceResult:
    """Check that all corpus WAVs plausibly belong to the same speaker.

    Embeds each WAV with Resemblyzer, computes the centroid, and flags clips
    whose cosine similarity to the centroid is below ``threshold``. A low
    outlier almost always means a mislabeled segment leaked in from another
    speaker — the silent-contamination failure mode of the manual per-interview
    labeling in the bridge workflow.

    Args:
        reference_corpus: Directory of clean WAVs to validate.
        threshold: Minimum clip-to-centroid cosine similarity. Default 0.70.
        max_wavs: Cap on WAVs embedded (cost control on large corpora).
        encoder: Optional pre-loaded Resemblyzer VoiceEncoder (for reuse/testing).

    Returns:
        CoherenceResult with per-outlier detail and an ``is_coherent`` verdict.

    Raises:
        FileNotFoundError: If ``reference_corpus`` has no WAVs.
        ImportError: If resemblyzer is not installed.
    """
    reference_corpus = Path(reference_corpus)
    wavs = sorted(reference_corpus.glob("*.wav"))[:max_wavs]
    if not wavs:
        raise FileNotFoundError(f"No WAVs in {reference_corpus}")

    enc = encoder if encoder is not None else _load_resemblyzer_encoder()
    embeddings = _embed_wavs(enc, wavs)

    stacked = np.stack(embeddings, axis=0)
    centroid = stacked.mean(axis=0)
    sims = [_cosine(e, centroid) for e in embeddings]

    outliers = [(w, s) for w, s in zip(wavs, sims, strict=True) if s < threshold]
    mean_sim = float(np.mean(sims))
    min_sim = float(np.min(sims))
    is_coherent = len(outliers) == 0

    if is_coherent:
        logger.info(
            "Corpus coherent: {} WAVs, mean sim {:.3f}, min {:.3f} (>= {:.2f})",
            len(wavs),
            mean_sim,
            min_sim,
            threshold,
        )
    else:
        logger.warning(
            "Corpus has {} outlier(s) below {:.2f} — possible contamination from "
            "a mislabeled speaker. Review before fine-tuning.",
            len(outliers),
            threshold,
        )

    return CoherenceResult(
        n_wavs=len(wavs),
        mean_similarity=mean_sim,
        min_similarity=min_sim,
        outliers=outliers,
        threshold=threshold,
        is_coherent=is_coherent,
    )


def _load_resemblyzer_encoder() -> object:
    """Load a Resemblyzer VoiceEncoder, or raise a clear ImportError."""
    try:
        from resemblyzer import VoiceEncoder
    except ImportError as exc:
        raise ImportError(
            "resemblyzer is not installed. Install with: "
            "pip install voicelegacy[similarity]  (or pip install resemblyzer). "
            "Corpus coherence validation needs it."
        ) from exc
    logger.info("Loading Resemblyzer VoiceEncoder for coherence check...")
    return VoiceEncoder()


def _embed_wavs(encoder: object, wavs: list[Path]) -> list[np.ndarray]:
    """Embed each WAV with Resemblyzer's preprocess + embed_utterance."""
    from resemblyzer import preprocess_wav

    embeddings: list[np.ndarray] = []
    for w in wavs:
        wav = preprocess_wav(w)
        embeddings.append(encoder.embed_utterance(wav))  # type: ignore[attr-defined]
    return embeddings


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity guarded against zero-norm vectors."""
    na = float(np.linalg.norm(a))
    nb = float(np.linalg.norm(b))
    if na < 1e-12 or nb < 1e-12:
        return 0.0
    return float(np.dot(a, b) / (na * nb))
