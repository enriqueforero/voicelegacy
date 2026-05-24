"""Tests for voicelegacy.finetune_dataset.

Includes the integration test the audit said was missing: build a corpus with
extract_segments_to_wav, then build the fine-tune dataset from it, and assert
the dataset is NON-EMPTY. This would have caught the criterion-6 bug where the
WAV<->text pairing silently produced zero rows.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from types import ModuleType

import numpy as np
import pytest
import soundfile as sf

from voicelegacy.config import ReferenceConfig
from voicelegacy.corpus import extract_segments_to_wav, filter_segments, load_speakerscribe_json
from voicelegacy.finetune_dataset import (
    CoherenceResult,
    DatasetBuildResult,
    _cosine,
    build_finetune_dataset,
    validate_corpus_coherence,
)


# ─── Fixtures ──────────────────────────────────────────────────────────────
def _write_corpus_with_sidecars(corpus_dir: Path, n: int = 4) -> None:
    """Create n WAVs each with a .txt transcript sidecar."""
    corpus_dir.mkdir(parents=True, exist_ok=True)
    sr = 22050
    for i in range(n):
        y = (0.2 * np.sin(2 * np.pi * 200 * np.arange(sr * 3) / sr)).astype(np.float32)
        wav = corpus_dir / f"src_{i:04d}_{i * 10:08.2f}.wav"
        sf.write(str(wav), y, sr, subtype="PCM_16")
        wav.with_suffix(".txt").write_text(
            f"Esta es la transcripcion numero {i} con suficiente longitud.",
            encoding="utf-8",
        )


# ─── build_finetune_dataset ────────────────────────────────────────────────
class TestBuildFinetuneDataset:
    def test_happy_path(self, tmp_path: Path):
        corpus = tmp_path / "reference_corpus"
        _write_corpus_with_sidecars(corpus, n=10)
        result = build_finetune_dataset(corpus, tmp_path / "dataset")
        assert isinstance(result, DatasetBuildResult)
        assert result.n_total == 10
        assert result.n_train >= 1
        assert result.n_eval >= 1
        assert result.n_skipped_no_text == 0
        # CSVs exist and have a header + rows
        train_csv = (tmp_path / "dataset" / "metadata_train.csv").read_text(encoding="utf-8")
        assert "audio_file|text|speaker_name" in train_csv
        assert "wavs/" in train_csv

    def test_wavs_copied_into_dataset(self, tmp_path: Path):
        corpus = tmp_path / "reference_corpus"
        _write_corpus_with_sidecars(corpus, n=5)
        build_finetune_dataset(corpus, tmp_path / "dataset")
        copied = list((tmp_path / "dataset" / "wavs").glob("*.wav"))
        assert len(copied) == 5

    def test_skips_wav_without_sidecar(self, tmp_path: Path):
        corpus = tmp_path / "reference_corpus"
        _write_corpus_with_sidecars(corpus, n=3)
        # Add a WAV with NO sidecar
        sr = 22050
        orphan = corpus / "orphan_0099_00099.00.wav"
        sf.write(str(orphan), np.zeros(sr, dtype=np.float32), sr)
        result = build_finetune_dataset(corpus, tmp_path / "dataset")
        assert result.n_skipped_no_text == 1
        assert result.n_total == 3

    def test_skips_short_text(self, tmp_path: Path):
        corpus = tmp_path / "reference_corpus"
        corpus.mkdir()
        sr = 22050
        wav = corpus / "src_0000_00000.00.wav"
        sf.write(str(wav), np.zeros(sr, dtype=np.float32), sr)
        wav.with_suffix(".txt").write_text("corto", encoding="utf-8")  # < 10 chars
        with pytest.raises(ValueError, match=r"No WAV.*usable transcript"):
            build_finetune_dataset(corpus, tmp_path / "dataset")

    def test_empty_corpus_raises(self, tmp_path: Path):
        corpus = tmp_path / "reference_corpus"
        corpus.mkdir()
        with pytest.raises(ValueError, match="No WAV"):
            build_finetune_dataset(corpus, tmp_path / "dataset")

    def test_missing_corpus_dir_raises(self, tmp_path: Path):
        with pytest.raises(FileNotFoundError, match="reference_corpus not found"):
            build_finetune_dataset(tmp_path / "ghost", tmp_path / "dataset")

    def test_invalid_eval_fraction(self, tmp_path: Path):
        corpus = tmp_path / "reference_corpus"
        _write_corpus_with_sidecars(corpus, n=3)
        with pytest.raises(ValueError, match="eval_fraction"):
            build_finetune_dataset(corpus, tmp_path / "dataset", eval_fraction=1.5)

    def test_deterministic_split(self, tmp_path: Path):
        corpus = tmp_path / "reference_corpus"
        _write_corpus_with_sidecars(corpus, n=20)
        r1 = build_finetune_dataset(corpus, tmp_path / "ds1", seed=42)
        r2 = build_finetune_dataset(corpus, tmp_path / "ds2", seed=42)
        assert r1.n_train == r2.n_train
        assert r1.n_eval == r2.n_eval
        csv1 = (tmp_path / "ds1" / "metadata_train.csv").read_text()
        csv2 = (tmp_path / "ds2" / "metadata_train.csv").read_text()
        assert csv1 == csv2  # same seed => identical split

    def test_two_rows_guarantees_eval(self, tmp_path: Path):
        corpus = tmp_path / "reference_corpus"
        _write_corpus_with_sidecars(corpus, n=2)
        result = build_finetune_dataset(corpus, tmp_path / "dataset")
        assert result.n_eval >= 1
        assert result.n_train >= 1


# ─── INTEGRATION: corpus -> finetune dataset (the missing audit test) ───────
class TestCorpusToFinetuneDatasetIntegration:
    """The audit's criterion-6 regression test.

    Build a corpus with the REAL extract_segments_to_wav, then build the
    fine-tune dataset from it. Asserts the dataset is NON-EMPTY. Before the fix,
    the notebook's filename-parsing pairing produced zero rows here.
    """

    def test_corpus_extraction_then_dataset_prep_is_nonempty(self, tmp_path: Path):
        # 1. Create a synthetic source recording (60s, 22.05 kHz)
        sr = 22050
        audios = tmp_path / "originales"
        audios.mkdir()
        t = np.arange(sr * 60) / sr
        y = (0.3 * np.sin(2 * np.pi * 180 * t)).astype(np.float32) * (
            0.5 + 0.5 * np.sin(2 * np.pi * 3 * t)
        )
        sf.write(str(audios / "entrevista.wav"), y, sr, subtype="PCM_16")

        # 2. speakerscribe-style JSON: target speaker with 3 long segments
        ss = {
            "audio_file": "entrevista.wav",
            "language_detected": "es",
            "segments": [
                {
                    "id": 0,
                    "start": 2.0,
                    "end": 12.0,
                    "text": "Nacimos en un pueblo pequeno rodeado de montanas verdes y rios.",
                    "speaker": "SPEAKER_00",
                },
                {
                    "id": 1,
                    "start": 14.0,
                    "end": 17.0,
                    "text": "Pregunta breve.",
                    "speaker": "SPEAKER_01",
                },
                {
                    "id": 2,
                    "start": 20.0,
                    "end": 32.0,
                    "text": "Mi madre cocinaba temprano cada manana sobre una estufa de lena.",
                    "speaker": "SPEAKER_00",
                },
                {
                    "id": 3,
                    "start": 34.0,
                    "end": 47.0,
                    "text": "Mi padre salia al campo antes del amanecer durante toda su vida.",
                    "speaker": "SPEAKER_00",
                },
            ],
        }
        jf = tmp_path / "entrevista.json"
        jf.write_text(json.dumps(ss, ensure_ascii=False), encoding="utf-8")

        # 3. Extract target-speaker corpus with sidecars (real pipeline)
        config = ReferenceConfig(
            min_segment_duration_s=6.0,
            max_segment_duration_s=15.0,
            enable_f0_outlier_filter=False,
        )
        segs = load_speakerscribe_json(jf, audio_root=audios)
        segs_ok = filter_segments(
            segs, target_speaker="SPEAKER_00", min_duration_s=6.0, max_duration_s=15.0
        )
        corpus = tmp_path / "reference_corpus"
        written = extract_segments_to_wav(segs_ok, corpus, config)
        assert len(written) >= 2, "Extraction should produce >= 2 WAVs"

        # 3b. Sidecars must exist next to the WAVs
        sidecars = list(corpus.glob("*.txt"))
        assert len(sidecars) == len(written), "Every WAV must have a .txt sidecar"

        # 4. Build the fine-tune dataset — MUST be non-empty (the regression)
        result = build_finetune_dataset(corpus, tmp_path / "dataset")
        assert result.n_total >= 2, (
            "REGRESSION (criterion 6): corpus->dataset produced an empty dataset"
        )
        assert result.n_skipped_no_text == 0

        # 5. The text in the CSV must match the original transcriptions
        train_csv = (tmp_path / "dataset" / "metadata_train.csv").read_text(encoding="utf-8")
        eval_csv = (tmp_path / "dataset" / "metadata_eval.csv").read_text(encoding="utf-8")
        all_text = train_csv + eval_csv
        assert (
            "pueblo pequeno" in all_text
            or "estufa de lena" in all_text
            or "antes del amanecer" in all_text
        )
        # The interviewer's line must NOT be present
        assert "Pregunta breve" not in all_text


# ─── validate_corpus_coherence ──────────────────────────────────────────────
def _fake_resemblyzer(monkeypatch, embed_fn):
    """Inject a fake resemblyzer module with preprocess_wav + VoiceEncoder."""
    mod = ModuleType("resemblyzer")
    mod.preprocess_wav = lambda p: np.array([0.0])  # type: ignore[attr-defined]

    class FakeEncoder:
        def embed_utterance(self, wav):
            return embed_fn()

    mod.VoiceEncoder = FakeEncoder  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "resemblyzer", mod)
    return FakeEncoder()


class TestValidateCorpusCoherence:
    def test_coherent_corpus(self, tmp_path: Path, monkeypatch):
        corpus = tmp_path / "reference_corpus"
        _write_corpus_with_sidecars(corpus, n=5)
        # All embeddings identical => perfectly coherent
        base = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        enc = _fake_resemblyzer(monkeypatch, lambda: base.copy())
        result = validate_corpus_coherence(corpus, threshold=0.70, encoder=enc)
        assert isinstance(result, CoherenceResult)
        assert result.is_coherent
        assert result.n_wavs == 5
        assert not result.outliers
        assert result.mean_similarity > 0.99

    def test_detects_contamination(self, tmp_path: Path, monkeypatch):
        corpus = tmp_path / "reference_corpus"
        _write_corpus_with_sidecars(corpus, n=4)
        # First 3 clips speaker A, last clip an orthogonal "other speaker"
        calls = {"i": 0}
        vectors = [
            np.array([1.0, 0.0, 0.0]),
            np.array([1.0, 0.05, 0.0]),
            np.array([1.0, 0.0, 0.05]),
            np.array([0.0, 1.0, 0.0]),  # contaminant — orthogonal
        ]

        def _embed():
            v = vectors[calls["i"] % len(vectors)]
            calls["i"] += 1
            return v.astype(np.float32)

        enc = _fake_resemblyzer(monkeypatch, _embed)
        result = validate_corpus_coherence(corpus, threshold=0.70, encoder=enc)
        assert not result.is_coherent
        assert len(result.outliers) >= 1
        assert result.min_similarity < 0.70

    def test_no_wavs_raises(self, tmp_path: Path):
        corpus = tmp_path / "reference_corpus"
        corpus.mkdir()
        with pytest.raises(FileNotFoundError, match="No WAVs"):
            validate_corpus_coherence(corpus)

    def test_missing_resemblyzer_raises(self, tmp_path: Path, monkeypatch):
        corpus = tmp_path / "reference_corpus"
        _write_corpus_with_sidecars(corpus, n=2)
        monkeypatch.setitem(sys.modules, "resemblyzer", None)
        with pytest.raises(ImportError, match="resemblyzer is not installed"):
            validate_corpus_coherence(corpus)

    def test_to_dict_serializable(self, tmp_path: Path, monkeypatch):
        corpus = tmp_path / "reference_corpus"
        _write_corpus_with_sidecars(corpus, n=3)
        base = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        enc = _fake_resemblyzer(monkeypatch, lambda: base.copy())
        result = validate_corpus_coherence(corpus, encoder=enc)
        d = result.to_dict()
        json.dumps(d)  # must not raise
        assert d["is_coherent"] is True
        assert "outliers" in d


class TestCosine:
    def test_identical(self):
        v = np.array([1.0, 2.0, 3.0])
        assert _cosine(v, v) == pytest.approx(1.0)

    def test_orthogonal(self):
        assert _cosine(np.array([1.0, 0.0]), np.array([0.0, 1.0])) == pytest.approx(0.0)

    def test_zero_norm_guarded(self):
        assert _cosine(np.array([0.0, 0.0]), np.array([1.0, 1.0])) == 0.0
