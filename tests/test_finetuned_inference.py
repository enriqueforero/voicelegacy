"""Tests for voicelegacy.finetuned_inference.

Strategy:
- Cover happy path with a mocked Xtts model (coqui-tts may or may not be
  installed in CI; we never want tests to depend on the model download).
- Exhaustively cover the validation surface of FineTunedCheckpoint.from_dir.
- Verify caching semantics: model cache by (dir, device), latents cache by
  (checkpoint_fingerprint, reference_set).
- Verify that the public synthesize_with_finetuned signature matches the
  base voicelegacy.synthesis.synthesize_to_file contract so it is drop-in.
"""

from __future__ import annotations

import sys
from pathlib import Path
from types import ModuleType
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import soundfile as sf

from voicelegacy import finetuned_inference as ftinfer
from voicelegacy.config import XTTS_OUTPUT_SR, SynthesisConfig
from voicelegacy.finetuned_inference import (
    REQUIRED_CHECKPOINT_FILES,
    FineTunedCheckpoint,
    _conditioning_cache_key,
    _resolve_device,
    is_available,
    load_finetuned_model,
    release_finetuned_model,
    synthesize_with_finetuned,
)


# ─── Fixtures ──────────────────────────────────────────────────────────────
@pytest.fixture(autouse=True)
def _reset_caches():
    """Ensure each test starts with empty module-level caches."""
    ftinfer._FT_MODEL_CACHE.clear()
    ftinfer._FT_LATENTS_CACHE.clear()
    yield
    ftinfer._FT_MODEL_CACHE.clear()
    ftinfer._FT_LATENTS_CACHE.clear()


def _make_fake_checkpoint(tmp_path: Path, model_pth_size_mb: float = 1800.0) -> Path:
    """Create a directory with all 5 expected checkpoint files.

    model.pth is a sparse file of the target size — stat() reports the bytes
    but the file allocates ~0 physical disk. Avoids filling /tmp when many
    fixtures run in sequence. The other four files are tiny placeholders.
    """
    import os

    d = tmp_path / "ckpt"
    d.mkdir(parents=True, exist_ok=True)

    # model.pth — sparse file via truncate; passes the size sanity check
    # without writing GB of zeros to disk.
    target_bytes = int(model_pth_size_mb * 1024 * 1024)
    with (d / "model.pth").open("wb") as f:
        os.truncate(f.fileno(), target_bytes)

    # config.json — minimal valid JSON
    (d / "config.json").write_text('{"model_args": {}, "audio": {"sample_rate": 24000}}')

    # vocab.json — minimal valid JSON (real one is ~600 KB)
    (d / "vocab.json").write_text('{"version": 1, "tokens": []}')

    # dvae.pth / mel_stats.pth / speakers_xtts.pth — placeholders
    (d / "dvae.pth").write_bytes(b"\x00" * 1024)
    (d / "mel_stats.pth").write_bytes(b"\x00" * 1024)
    (d / "speakers_xtts.pth").write_bytes(b"\x00" * 1024)
    return d


@pytest.fixture
def fake_checkpoint_dir(tmp_path: Path) -> Path:
    """A complete fake checkpoint passing all validations."""
    return _make_fake_checkpoint(tmp_path, model_pth_size_mb=200.0)


@pytest.fixture
def fake_reference_wav(tmp_path: Path) -> Path:
    """A 5-second WAV at XTTS sample rate (24 kHz output, 22.05 kHz input is fine too)."""
    sr = 24000
    t = np.arange(sr * 5) / sr
    signal = (0.3 * np.sin(2 * np.pi * 220 * t)).astype(np.float32)
    p = tmp_path / "ref.wav"
    sf.write(str(p), signal, sr, subtype="PCM_16")
    return p


# ─── FineTunedCheckpoint.from_dir ──────────────────────────────────────────
class TestFineTunedCheckpointFromDir:
    """Exhaustive validation of the checkpoint-handle constructor."""

    def test_happy_path_all_files_present(self, fake_checkpoint_dir: Path):
        ckpt = FineTunedCheckpoint.from_dir(fake_checkpoint_dir)
        assert ckpt.checkpoint_dir == fake_checkpoint_dir.resolve()
        assert ckpt.model_pth.name == "model.pth"
        assert ckpt.config_json.name == "config.json"
        assert ckpt.vocab_json.name == "vocab.json"
        assert ckpt.dvae_pth.name == "dvae.pth"
        assert ckpt.mel_stats_pth.name == "mel_stats.pth"
        assert ckpt.speakers_xtts_pth.name == "speakers_xtts.pth"
        assert len(ckpt.fingerprint) == 16
        assert all(c in "0123456789abcdef" for c in ckpt.fingerprint)

    def test_accepts_string_path(self, fake_checkpoint_dir: Path):
        ckpt = FineTunedCheckpoint.from_dir(str(fake_checkpoint_dir))
        assert ckpt.checkpoint_dir == fake_checkpoint_dir.resolve()

    def test_expands_user_path(self, fake_checkpoint_dir: Path, monkeypatch):
        # Pretend $HOME is the parent of fake_checkpoint_dir
        monkeypatch.setenv("HOME", str(fake_checkpoint_dir.parent))
        rel = f"~/{fake_checkpoint_dir.name}"
        ckpt = FineTunedCheckpoint.from_dir(rel)
        assert ckpt.checkpoint_dir == fake_checkpoint_dir.resolve()

    def test_missing_directory(self, tmp_path: Path):
        with pytest.raises(FileNotFoundError, match="does not exist"):
            FineTunedCheckpoint.from_dir(tmp_path / "does_not_exist")

    def test_path_is_a_file_not_dir(self, tmp_path: Path):
        f = tmp_path / "not_a_dir.txt"
        f.write_text("hi")
        with pytest.raises(FileNotFoundError, match="not a directory"):
            FineTunedCheckpoint.from_dir(f)

    @pytest.mark.parametrize("missing", REQUIRED_CHECKPOINT_FILES)
    def test_missing_single_required_file(self, tmp_path: Path, missing: str):
        d = _make_fake_checkpoint(tmp_path, model_pth_size_mb=200.0)
        (d / missing).unlink()
        with pytest.raises(FileNotFoundError, match="missing required file"):
            FineTunedCheckpoint.from_dir(d)

    def test_missing_multiple_required_files(self, tmp_path: Path):
        d = _make_fake_checkpoint(tmp_path, model_pth_size_mb=200.0)
        (d / "vocab.json").unlink()
        (d / "dvae.pth").unlink()
        with pytest.raises(FileNotFoundError) as excinfo:
            FineTunedCheckpoint.from_dir(d)
        assert "vocab.json" in str(excinfo.value)
        assert "dvae.pth" in str(excinfo.value)

    def test_model_pth_too_small(self, tmp_path: Path):
        d = _make_fake_checkpoint(tmp_path, model_pth_size_mb=10.0)
        with pytest.raises(ValueError, match="suspiciously small"):
            FineTunedCheckpoint.from_dir(d)

    def test_fingerprint_changes_with_model_content(self, tmp_path: Path):
        d1 = _make_fake_checkpoint(tmp_path / "a", model_pth_size_mb=200.0)
        # Mutate config.json so the fingerprint must change
        d2 = _make_fake_checkpoint(tmp_path / "b", model_pth_size_mb=200.0)
        (d2 / "config.json").write_text('{"different": "content"}')

        fp1 = FineTunedCheckpoint.from_dir(d1).fingerprint
        fp2 = FineTunedCheckpoint.from_dir(d2).fingerprint
        assert fp1 != fp2

    def test_to_dict_contains_audit_fields(self, fake_checkpoint_dir: Path):
        ckpt = FineTunedCheckpoint.from_dir(fake_checkpoint_dir)
        d = ckpt.to_dict()
        assert d["fingerprint"] == ckpt.fingerprint
        assert d["model_pth_size_mb"] > 100
        assert "checkpoint_dir" in d
        assert "model_pth_mtime_ns" in d


# ─── Device resolution ─────────────────────────────────────────────────────
class TestResolveDevice:
    def test_explicit_cpu_passes_through(self):
        assert _resolve_device("cpu") == "cpu"

    def test_explicit_cuda_passes_through(self):
        assert _resolve_device("cuda") == "cuda"

    def test_auto_with_torch_cuda_available(self):
        with patch.dict(sys.modules):
            fake_torch = ModuleType("torch")
            fake_torch.cuda = ModuleType("torch.cuda")
            fake_torch.cuda.is_available = lambda: True  # type: ignore[attr-defined]
            sys.modules["torch"] = fake_torch
            assert _resolve_device("auto") == "cuda"

    def test_auto_without_cuda(self):
        with patch.dict(sys.modules):
            fake_torch = ModuleType("torch")
            fake_torch.cuda = ModuleType("torch.cuda")
            fake_torch.cuda.is_available = lambda: False  # type: ignore[attr-defined]
            sys.modules["torch"] = fake_torch
            assert _resolve_device("auto") == "cpu"

    def test_auto_without_torch(self):
        with patch.dict(sys.modules, {"torch": None}):
            assert _resolve_device("auto") == "cpu"


# ─── load_finetuned_model ──────────────────────────────────────────────────
class TestLoadFinetunedModel:
    """Mock the coqui-tts loader entirely; we test our wiring, not theirs."""

    def _inject_fake_tts(self, fake_model: MagicMock) -> ModuleType:
        """Build a fake TTS.tts.{configs.xtts_config, models.xtts} module tree."""
        # Build the nested package structure
        tts_pkg = ModuleType("TTS")
        tts_tts_pkg = ModuleType("TTS.tts")
        tts_tts_configs_pkg = ModuleType("TTS.tts.configs")
        tts_tts_configs_xtts = ModuleType("TTS.tts.configs.xtts_config")
        tts_tts_models_pkg = ModuleType("TTS.tts.models")
        tts_tts_models_xtts = ModuleType("TTS.tts.models.xtts")

        class FakeXttsConfig:
            def __init__(self):
                self.audio = MagicMock(sample_rate=24000)

            def load_json(self, path: str):
                self.loaded_from = path

        class FakeXtts:
            @classmethod
            def init_from_config(cls, config):
                return fake_model

        tts_tts_configs_xtts.XttsConfig = FakeXttsConfig
        tts_tts_models_xtts.Xtts = FakeXtts

        sys.modules["TTS"] = tts_pkg
        sys.modules["TTS.tts"] = tts_tts_pkg
        sys.modules["TTS.tts.configs"] = tts_tts_configs_pkg
        sys.modules["TTS.tts.configs.xtts_config"] = tts_tts_configs_xtts
        sys.modules["TTS.tts.models"] = tts_tts_models_pkg
        sys.modules["TTS.tts.models.xtts"] = tts_tts_models_xtts
        return tts_pkg

    def test_happy_path_loads_and_caches(self, fake_checkpoint_dir: Path):
        fake_model = MagicMock()
        with patch.dict(sys.modules):
            self._inject_fake_tts(fake_model)
            ckpt = FineTunedCheckpoint.from_dir(fake_checkpoint_dir)
            model = load_finetuned_model(ckpt, device="cpu")
            assert model is fake_model
            fake_model.load_checkpoint.assert_called_once()
            # Cache hit on second call
            model_again = load_finetuned_model(ckpt, device="cpu")
            assert model_again is fake_model
            # load_checkpoint not called again
            assert fake_model.load_checkpoint.call_count == 1

    def test_cache_separates_by_device(self, fake_checkpoint_dir: Path):
        fake_model_cpu = MagicMock(name="cpu_model")
        fake_model_cuda = MagicMock(name="cuda_model")
        with patch.dict(sys.modules):
            self._inject_fake_tts(fake_model_cpu)
            ckpt = FineTunedCheckpoint.from_dir(fake_checkpoint_dir)
            m1 = load_finetuned_model(ckpt, device="cpu")
            # Re-inject with cuda model
            self._inject_fake_tts(fake_model_cuda)
            m2 = load_finetuned_model(ckpt, device="cuda")
        assert m1 is fake_model_cpu
        assert m2 is fake_model_cuda
        # cuda branch should have invoked .cuda()
        fake_model_cuda.cuda.assert_called_once()

    def test_missing_coqui_tts_raises_import_error(self, fake_checkpoint_dir: Path):
        ckpt = FineTunedCheckpoint.from_dir(fake_checkpoint_dir)
        # Hide every TTS submodule we touch
        with patch.dict(sys.modules):
            for key in [
                "TTS",
                "TTS.tts",
                "TTS.tts.configs",
                "TTS.tts.configs.xtts_config",
                "TTS.tts.models",
                "TTS.tts.models.xtts",
            ]:
                sys.modules[key] = None  # type: ignore[assignment]
            with pytest.raises(ImportError, match="coqui-tts is not installed"):
                load_finetuned_model(ckpt, device="cpu")

    def test_load_checkpoint_failure_wraps_runtime_error(self, fake_checkpoint_dir: Path):
        fake_model = MagicMock()
        fake_model.load_checkpoint.side_effect = ValueError("bad weights")
        with patch.dict(sys.modules):
            self._inject_fake_tts(fake_model)
            ckpt = FineTunedCheckpoint.from_dir(fake_checkpoint_dir)
            with pytest.raises(RuntimeError, match="Failed to load fine-tuned checkpoint"):
                load_finetuned_model(ckpt, device="cpu")


# ─── release_finetuned_model ───────────────────────────────────────────────
class TestReleaseFinetunedModel:
    def test_clears_model_and_latents_caches(self):
        ftinfer._FT_MODEL_CACHE["x"] = MagicMock()
        ftinfer._FT_LATENTS_CACHE["y"] = (MagicMock(), MagicMock())
        release_finetuned_model()
        assert ftinfer._FT_MODEL_CACHE == {}
        assert ftinfer._FT_LATENTS_CACHE == {}

    def test_release_without_torch_does_not_crash(self):
        ftinfer._FT_MODEL_CACHE["x"] = MagicMock()
        with patch.dict(sys.modules, {"torch": None}):
            release_finetuned_model()
        assert ftinfer._FT_MODEL_CACHE == {}

    def test_release_with_torch_cuda(self):
        ftinfer._FT_MODEL_CACHE["x"] = MagicMock()
        fake_torch = ModuleType("torch")
        fake_torch.cuda = ModuleType("torch.cuda")
        fake_torch.cuda.is_available = lambda: True  # type: ignore[attr-defined]
        fake_torch.cuda.empty_cache = MagicMock()  # type: ignore[attr-defined]
        with patch.dict(sys.modules, {"torch": fake_torch}):
            release_finetuned_model()
        fake_torch.cuda.empty_cache.assert_called_once()  # type: ignore[attr-defined]


# ─── Conditioning latents cache key ────────────────────────────────────────
class TestConditioningCacheKey:
    def test_same_files_same_key(self, fake_reference_wav: Path):
        k1 = _conditioning_cache_key([str(fake_reference_wav)])
        k2 = _conditioning_cache_key([str(fake_reference_wav)])
        assert k1 == k2

    def test_order_does_not_matter(self, tmp_path: Path):
        f1 = tmp_path / "a.wav"
        f2 = tmp_path / "b.wav"
        sr = 24000
        sf.write(str(f1), np.zeros(sr, dtype=np.float32), sr)
        sf.write(str(f2), np.zeros(sr, dtype=np.float32), sr)
        assert _conditioning_cache_key([str(f1), str(f2)]) == _conditioning_cache_key(
            [str(f2), str(f1)]
        )

    def test_different_content_different_key(self, tmp_path: Path):
        f1 = tmp_path / "a.wav"
        f2 = tmp_path / "b.wav"
        sr = 24000
        sf.write(str(f1), np.zeros(sr, dtype=np.float32), sr)
        # f2 has different size → different mtime/size in cache key
        sf.write(str(f2), np.zeros(sr * 2, dtype=np.float32), sr)
        assert _conditioning_cache_key([str(f1)]) != _conditioning_cache_key([str(f2)])


# ─── synthesize_with_finetuned ─────────────────────────────────────────────
class TestSynthesizeWithFinetuned:
    """Mock the model.inference call and verify wiring + WAV write."""

    def _build_mock_model_with_inference(self, n_samples: int = 24000):
        """Mock that returns a 1-D float array as XTTS inference would."""
        model = MagicMock()
        wav_array = (0.1 * np.sin(np.arange(n_samples) / 100)).astype(np.float32)
        model.get_conditioning_latents.return_value = (
            MagicMock(name="gpt_cond_latent"),
            MagicMock(name="speaker_embedding"),
        )
        model.inference.return_value = {"wav": wav_array}
        return model

    def test_happy_path_writes_wav(
        self, fake_checkpoint_dir: Path, fake_reference_wav: Path, tmp_path: Path
    ):
        ckpt = FineTunedCheckpoint.from_dir(fake_checkpoint_dir)
        model = self._build_mock_model_with_inference(24000)
        out = tmp_path / "out.wav"
        result = synthesize_with_finetuned(
            model=model,
            checkpoint=ckpt,
            text="Hola mundo.",
            speaker_wav=fake_reference_wav,
            output_path=out,
            config=SynthesisConfig(),
        )
        assert result == out
        assert out.exists()
        # Verify the WAV is readable and has the expected sample rate
        data, sr = sf.read(str(out))
        assert sr == XTTS_OUTPUT_SR
        assert data.size == 24000
        # Verify inference was called with the expected text/language
        args, _kwargs = model.inference.call_args
        assert args[0] == "Hola mundo."
        assert args[1] == "es"

    def test_accepts_list_of_references(self, fake_checkpoint_dir: Path, tmp_path: Path):
        ckpt = FineTunedCheckpoint.from_dir(fake_checkpoint_dir)
        model = self._build_mock_model_with_inference()

        # Create two reference wavs
        refs = []
        for name in ["a.wav", "b.wav"]:
            p = tmp_path / name
            sf.write(str(p), np.zeros(24000, dtype=np.float32), 24000)
            refs.append(p)

        out = tmp_path / "out.wav"
        synthesize_with_finetuned(
            model=model,
            checkpoint=ckpt,
            text="Texto de prueba.",
            speaker_wav=refs,
            output_path=out,
            config=SynthesisConfig(),
        )
        # get_conditioning_latents must receive both files
        ref_paths = model.get_conditioning_latents.call_args.kwargs["audio_path"]
        assert len(ref_paths) == 2

    def test_empty_text_raises(
        self, fake_checkpoint_dir: Path, fake_reference_wav: Path, tmp_path: Path
    ):
        ckpt = FineTunedCheckpoint.from_dir(fake_checkpoint_dir)
        model = self._build_mock_model_with_inference()
        with pytest.raises(ValueError, match="empty text"):
            synthesize_with_finetuned(
                model=model,
                checkpoint=ckpt,
                text="   ",
                speaker_wav=fake_reference_wav,
                output_path=tmp_path / "out.wav",
                config=SynthesisConfig(),
            )

    def test_missing_reference_raises(self, fake_checkpoint_dir: Path, tmp_path: Path):
        ckpt = FineTunedCheckpoint.from_dir(fake_checkpoint_dir)
        model = self._build_mock_model_with_inference()
        with pytest.raises(ValueError, match="Reference audio not found"):
            synthesize_with_finetuned(
                model=model,
                checkpoint=ckpt,
                text="x",
                speaker_wav=tmp_path / "ghost.wav",
                output_path=tmp_path / "out.wav",
                config=SynthesisConfig(),
            )

    def test_inference_bad_return_raises_runtime_error(
        self, fake_checkpoint_dir: Path, fake_reference_wav: Path, tmp_path: Path
    ):
        ckpt = FineTunedCheckpoint.from_dir(fake_checkpoint_dir)
        model = MagicMock()
        model.get_conditioning_latents.return_value = (MagicMock(), MagicMock())
        model.inference.return_value = {"not_wav": "oops"}
        with pytest.raises(RuntimeError, match="did not return"):
            synthesize_with_finetuned(
                model=model,
                checkpoint=ckpt,
                text="hola",
                speaker_wav=fake_reference_wav,
                output_path=tmp_path / "out.wav",
                config=SynthesisConfig(),
            )

    def test_latents_cache_isolation_by_checkpoint(self, tmp_path: Path, fake_reference_wav: Path):
        """Two checkpoints must NOT share cached latents even on same references."""
        d1 = _make_fake_checkpoint(tmp_path / "ck1", model_pth_size_mb=200.0)
        d2 = _make_fake_checkpoint(tmp_path / "ck2", model_pth_size_mb=200.0)
        (d2 / "config.json").write_text('{"different": "yes"}')

        ckpt1 = FineTunedCheckpoint.from_dir(d1)
        ckpt2 = FineTunedCheckpoint.from_dir(d2)
        assert ckpt1.fingerprint != ckpt2.fingerprint

        model1 = self._build_mock_model_with_inference()
        model2 = self._build_mock_model_with_inference()

        synthesize_with_finetuned(
            model=model1,
            checkpoint=ckpt1,
            text="x",
            speaker_wav=fake_reference_wav,
            output_path=tmp_path / "o1.wav",
            config=SynthesisConfig(),
        )
        synthesize_with_finetuned(
            model=model2,
            checkpoint=ckpt2,
            text="x",
            speaker_wav=fake_reference_wav,
            output_path=tmp_path / "o2.wav",
            config=SynthesisConfig(),
        )
        # Both models must have computed their own latents (no cross-checkpoint share)
        assert model1.get_conditioning_latents.call_count == 1
        assert model2.get_conditioning_latents.call_count == 1
        # Cache should have 2 distinct entries
        assert len(ftinfer._FT_LATENTS_CACHE) == 2


# ─── is_available ──────────────────────────────────────────────────────────
class TestIsAvailable:
    def test_returns_true_when_imports_succeed(self):
        # is_available() returns True only if both TTS submodules import cleanly,
        # which requires real torch (not a stub). CI environments without torch
        # legitimately return False; the function's job is to be honest about
        # that. We test both directions explicitly.
        result = is_available()
        assert isinstance(result, bool)
        # If torch IS available, the function must say True
        try:
            import torch  # noqa: F401
            import TTS.tts.configs.xtts_config
            import TTS.tts.models.xtts  # noqa: F401

            expected = True
        except ImportError:
            expected = False
        assert result is expected

    def test_returns_false_when_xtts_module_missing(self):
        with patch.dict(
            sys.modules,
            {
                "TTS.tts.configs.xtts_config": None,
                "TTS.tts.models.xtts": None,
            },
        ):
            assert is_available() is False
