"""Tests for synthesis wrapper behavior that does not require XTTS weights."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from voicelegacy.config import SynthesisConfig
from voicelegacy.synthesis import _apply_seed, synthesize_to_file


class FakeTTS:
    """Minimal TTS API stub."""

    def __init__(self) -> None:
        self.kwargs = None

    def tts_to_file(self, **kwargs) -> None:
        self.kwargs = kwargs
        Path(kwargs["file_path"]).write_bytes(b"wav")


def test_apply_seed_makes_numpy_repeatable() -> None:
    _apply_seed(123)
    a = np.random.random(4)
    _apply_seed(123)
    b = np.random.random(4)
    assert np.allclose(a, b)


def test_synthesize_to_file_passes_seeded_config(tmp_path: Path) -> None:
    ref = tmp_path / "ref.wav"
    ref.write_bytes(b"fake")
    out = tmp_path / "out.wav"
    tts = FakeTTS()

    result = synthesize_to_file(
        tts=tts,
        text="hola",
        speaker_wav=ref,
        output_path=out,
        config=SynthesisConfig(seed=7, compute_similarity=False),
    )

    assert result == out
    assert out.exists()
    assert tts.kwargs["temperature"] == 0.7
    assert tts.kwargs["language"] == "es"


class FakeLowerLevelModel:
    def __init__(self) -> None:
        self.latent_calls = 0
        self.inference_calls = 0

    def get_conditioning_latents(self, audio_path):
        self.latent_calls += 1
        return "gpt_latent", "speaker_embedding"

    def inference(self, text, language, gpt_cond_latent, speaker_embedding, **kwargs):
        self.inference_calls += 1
        assert gpt_cond_latent == "gpt_latent"
        assert speaker_embedding == "speaker_embedding"
        assert kwargs["enable_text_splitting"] is False
        return {"wav": np.zeros(2400, dtype=np.float32)}


class FakeSynthesizer:
    def __init__(self) -> None:
        self.tts_model = FakeLowerLevelModel()


class FakeTTSWithModel:
    def __init__(self) -> None:
        self.synthesizer = FakeSynthesizer()
        self.fallback_calls = 0

    def tts_to_file(self, **kwargs) -> None:
        self.fallback_calls += 1
        Path(kwargs["file_path"]).write_bytes(b"fallback")


def test_synthesize_uses_cached_conditioning_latents_when_model_api_exists(tmp_path: Path) -> None:
    from voicelegacy.synthesis import release_conditioning_latents

    release_conditioning_latents()
    ref = tmp_path / "ref.wav"
    ref.write_bytes(b"fake-ref")
    out1 = tmp_path / "out1.wav"
    out2 = tmp_path / "out2.wav"
    tts = FakeTTSWithModel()
    config = SynthesisConfig(seed=7, compute_similarity=False, cache_conditioning_latents=True)

    synthesize_to_file(tts, "hola", ref, out1, config)
    synthesize_to_file(tts, "hola de nuevo", ref, out2, config)

    assert out1.exists()
    assert out2.exists()
    assert tts.synthesizer.tts_model.latent_calls == 1
    assert tts.synthesizer.tts_model.inference_calls == 2
    assert tts.fallback_calls == 0


def test_synthesize_falls_back_when_latent_cache_disabled(tmp_path: Path) -> None:
    ref = tmp_path / "ref.wav"
    ref.write_bytes(b"fake-ref")
    out = tmp_path / "out.wav"
    tts = FakeTTSWithModel()

    synthesize_to_file(
        tts,
        "hola",
        ref,
        out,
        SynthesisConfig(seed=7, compute_similarity=False, cache_conditioning_latents=False),
    )

    assert tts.synthesizer.tts_model.latent_calls == 0
    assert tts.fallback_calls == 1


class TestModelLoadingWithoutWeights:
    def test_load_xtts_model_requires_tos(self) -> None:
        from voicelegacy.synthesis import load_xtts_model

        try:
            load_xtts_model(SynthesisConfig(), accept_tos=False)
        except RuntimeError as exc:
            assert "CPML" in str(exc)
        else:  # pragma: no cover - defensive
            raise AssertionError("load_xtts_model must require CPML acceptance")

    def test_load_xtts_model_returns_cached_model_without_importing_tts(self) -> None:
        import voicelegacy.synthesis as synthesis

        synthesis.release_model()
        cfg = SynthesisConfig(device="cpu")
        key = f"{cfg.model_name}|{cfg.device}"
        sentinel = object()
        synthesis._MODEL_CACHE[key] = sentinel

        try:
            assert synthesis.load_xtts_model(cfg, accept_tos=True) is sentinel
        finally:
            synthesis.release_model()

    def test_load_xtts_model_raises_clear_import_error_when_coqui_missing(self) -> None:
        import voicelegacy.synthesis as synthesis

        synthesis.release_model()
        try:
            synthesis.load_xtts_model(SynthesisConfig(device="cpu"), accept_tos=True)
        except ImportError as exc:
            assert "coqui-tts is not installed" in str(exc)
        else:  # pragma: no cover - would mean the test environment changed
            synthesis.release_model()
            raise AssertionError("Expected ImportError in this lightweight CI environment")


class TestLowerLevelInferenceFallbacks:
    def test_extract_xtts_model_returns_none_without_expected_api(self) -> None:
        from voicelegacy.synthesis import _extract_xtts_model

        class Wrapper:
            synthesizer = object()

        assert _extract_xtts_model(Wrapper()) is None
        assert _extract_xtts_model(object()) is None

    def test_tensor_to_numpy_rejects_empty_audio(self) -> None:
        from voicelegacy.synthesis import _tensor_to_numpy_1d

        try:
            _tensor_to_numpy_1d(np.asarray([], dtype=np.float32))
        except RuntimeError as exc:
            assert "empty/non-1D" in str(exc)
        else:  # pragma: no cover - defensive
            raise AssertionError("empty XTTS output must fail")

    def test_manual_inference_falls_back_when_output_shape_is_invalid(self, tmp_path: Path) -> None:
        class BadModel(FakeLowerLevelModel):
            def inference(self, *args, **kwargs):
                return {"wav": np.zeros((2, 2, 2), dtype=np.float32)}

        class BadSynthesizer:
            def __init__(self) -> None:
                self.tts_model = BadModel()

        class BadTTS(FakeTTSWithModel):
            def __init__(self) -> None:
                self.synthesizer = BadSynthesizer()
                self.fallback_calls = 0

        from voicelegacy.synthesis import release_conditioning_latents

        release_conditioning_latents()
        ref = tmp_path / "ref.wav"
        ref.write_bytes(b"fake-ref")
        out = tmp_path / "out.wav"
        tts = BadTTS()

        synthesize_to_file(tts, "hola", ref, out, SynthesisConfig(compute_similarity=False))

        assert out.read_bytes() == b"fallback"
        assert tts.fallback_calls == 1
