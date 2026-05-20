"""voicelegacy — Inference with a locally fine-tuned XTTS-v2 checkpoint.

This module is the bridge between the upstream Coqui XTTS-v2 fine-tuning
recipe and the rest of the voicelegacy pipeline. After running the
fine-tuning notebook (notebooks/notebook_voicelegacy_finetune.ipynb), you
get a checkpoint folder with five files:

    checkpoint_dir/
    ├── model.pth          ← fine-tuned GPT encoder weights
    ├── config.json        ← model architecture + tokenizer hyperparams
    ├── vocab.json         ← BPE vocab
    ├── dvae.pth           ← discrete VAE for audio tokens (NOT fine-tuned)
    ├── mel_stats.pth      ← mel-spectrogram normalization stats (NOT fine-tuned)
    └── speakers_xtts.pth  ← speaker embeddings table

Only ``model.pth`` was actually updated during fine-tuning; the other four
files are copied verbatim from the base XTTS-v2 release. The notebook
handles that copy step automatically.

This module deliberately does NOT depend on the high-level ``TTS.api.TTS``
wrapper used by ``voicelegacy.synthesis``. The high-level wrapper assumes
a Coqui-hub model name and downloads weights; we are loading a local,
custom checkpoint and need direct access to the lower-level ``Xtts``
class. This is the same code path that Coqui's fine-tuning gradio demo
uses internally.

Reference for the loading procedure:
- https://docs.coqui.ai/en/latest/models/xtts.html#training
- https://github.com/idiap/coqui-ai-TTS/blob/main/TTS/tts/models/xtts.py

Cache strategy mirrors ``voicelegacy.synthesis``:
- ``_FT_MODEL_CACHE``: model object keyed by checkpoint_dir
- ``_FT_LATENTS_CACHE``: conditioning latents keyed by reference set
"""

from __future__ import annotations

import gc
import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf

from voicelegacy.config import XTTS_OUTPUT_SR, SynthesisConfig
from voicelegacy.logging_config import get_logger
from voicelegacy.text_strategy import plan_text_synthesis

logger = get_logger()

# ─── Module-level caches ──────────────────────────────────────────────────
# Keyed by absolute path of the checkpoint directory so multiple checkpoints
# can coexist in memory if the user is comparing fine-tuned variants.
_FT_MODEL_CACHE: dict[str, Any] = {}
_FT_LATENTS_CACHE: dict[str, tuple[Any, Any]] = {}

# Files expected in a fine-tuned checkpoint directory. The fine-tuning notebook
# materializes all five before the user attempts inference. If any is missing,
# we fail early with a clear message instead of letting coqui-tts emit a
# stack-trace from deep inside its loader.
REQUIRED_CHECKPOINT_FILES: tuple[str, ...] = (
    "model.pth",
    "config.json",
    "vocab.json",
    "dvae.pth",
    "mel_stats.pth",
    "speakers_xtts.pth",
)


@dataclass(frozen=True)
class FineTunedCheckpoint:
    """Validated handle to a fine-tuned XTTS-v2 checkpoint on disk.

    Construct via ``FineTunedCheckpoint.from_dir(path)``; that classmethod
    runs the file-existence + size sanity checks. A constructed instance is
    safe to pass to ``load_finetuned_model``.
    """

    checkpoint_dir: Path
    model_pth: Path
    config_json: Path
    vocab_json: Path
    dvae_pth: Path
    mel_stats_pth: Path
    speakers_xtts_pth: Path
    fingerprint: str  # hash of (model.pth size, mtime, config.json content)

    @classmethod
    def from_dir(cls, checkpoint_dir: Path | str) -> FineTunedCheckpoint:
        """Validate the checkpoint directory and produce a handle.

        Raises:
            FileNotFoundError: If the directory or any required file is missing.
            ValueError: If model.pth is suspiciously small (< 100 MB ≈ corrupt).
        """
        d = Path(checkpoint_dir).expanduser().resolve()
        if not d.is_dir():
            raise FileNotFoundError(
                f"Checkpoint directory does not exist or is not a directory: {d}"
            )

        files = {name: d / name for name in REQUIRED_CHECKPOINT_FILES}
        missing = [name for name, p in files.items() if not p.is_file()]
        if missing:
            raise FileNotFoundError(
                f"Checkpoint at {d} is missing required file(s): {sorted(missing)}. "
                f"All of {REQUIRED_CHECKPOINT_FILES} must be present."
            )

        # Sanity: the GPT encoder weights weigh ~1.8 GB. Anything under 100 MB
        # is either corrupt, a placeholder, or the wrong file.
        size_mb = files["model.pth"].stat().st_size / (1024 * 1024)
        if size_mb < 100:
            raise ValueError(
                f"model.pth is suspiciously small ({size_mb:.1f} MB). "
                f"Expected ~1800 MB. Check that fine-tuning actually completed "
                f"and the file is not a placeholder."
            )

        # Fingerprint for cache keys + audit trail in sidecars
        fp = hashlib.sha256()
        fp.update(str(files["model.pth"].stat().st_size).encode("ascii"))
        fp.update(str(files["model.pth"].stat().st_mtime_ns).encode("ascii"))
        fp.update(files["config.json"].read_bytes())
        fingerprint = fp.hexdigest()[:16]

        return cls(
            checkpoint_dir=d,
            model_pth=files["model.pth"],
            config_json=files["config.json"],
            vocab_json=files["vocab.json"],
            dvae_pth=files["dvae.pth"],
            mel_stats_pth=files["mel_stats.pth"],
            speakers_xtts_pth=files["speakers_xtts.pth"],
            fingerprint=fingerprint,
        )

    def to_dict(self) -> dict[str, object]:
        """Audit-trail dict for sidecars."""
        return {
            "checkpoint_dir": str(self.checkpoint_dir),
            "fingerprint": self.fingerprint,
            "model_pth_size_mb": round(self.model_pth.stat().st_size / (1024 * 1024), 1),
            "model_pth_mtime_ns": self.model_pth.stat().st_mtime_ns,
        }


# ─── Model loading ────────────────────────────────────────────────────────
def _resolve_device(requested: str) -> str:
    """Resolve 'auto' to 'cuda' if torch CUDA is present, else 'cpu'.

    Mirrors ``voicelegacy.synthesis._resolve_device`` deliberately to keep
    behavior consistent across the two inference paths.
    """
    if requested != "auto":
        return requested
    try:
        import torch

        return "cuda" if torch.cuda.is_available() else "cpu"
    except ImportError:
        return "cpu"


def load_finetuned_model(
    checkpoint: FineTunedCheckpoint,
    device: str = "auto",
) -> Any:
    """Load (or fetch from cache) a fine-tuned XTTS-v2 model.

    Args:
        checkpoint: A validated FineTunedCheckpoint handle.
        device: 'auto', 'cuda', or 'cpu'. Defaults to 'auto'.

    Returns:
        Initialized Xtts model object ready for inference. NOT the high-level
        TTS API wrapper — this is the lower-level model that exposes
        ``inference()`` and ``get_conditioning_latents()`` directly.

    Raises:
        ImportError: If coqui-tts is not installed.
        RuntimeError: If the checkpoint cannot be loaded (mismatched config,
            corrupted weights, etc.).
    """
    cache_key = f"{checkpoint.checkpoint_dir}|{device}"
    if cache_key in _FT_MODEL_CACHE:
        logger.info(
            "Using cached fine-tuned model fp={} dir={}",
            checkpoint.fingerprint,
            checkpoint.checkpoint_dir.name,
        )
        return _FT_MODEL_CACHE[cache_key]

    try:
        from TTS.tts.configs.xtts_config import XttsConfig
        from TTS.tts.models.xtts import Xtts
    except ImportError as exc:
        raise ImportError(
            "coqui-tts is not installed. Install with: pip install coqui-tts"
        ) from exc

    resolved_device = _resolve_device(device)
    logger.info(
        "Loading fine-tuned XTTS-v2 from {} on device '{}' (fp={})",
        checkpoint.checkpoint_dir,
        resolved_device,
        checkpoint.fingerprint,
    )

    try:
        config = XttsConfig()
        config.load_json(str(checkpoint.config_json))
        model = Xtts.init_from_config(config)
        # use_deepspeed=False because Colab Free T4 lacks DeepSpeed dependencies
        # and the speedup is irrelevant for one-off legacy clips. Setting it
        # explicitly avoids a noisy auto-detection branch in coqui-tts.
        model.load_checkpoint(
            config,
            checkpoint_dir=str(checkpoint.checkpoint_dir),
            use_deepspeed=False,
        )
    except Exception as exc:
        raise RuntimeError(
            f"Failed to load fine-tuned checkpoint from {checkpoint.checkpoint_dir}: {exc}. "
            "Common causes: (1) config.json from a different XTTS version than the .pth files, "
            "(2) corrupted model.pth, (3) coqui-tts upgraded its API since the checkpoint was "
            "produced. Try re-running the fine-tuning notebook with the same coqui-tts version."
        ) from exc

    if resolved_device == "cuda":
        model.cuda()

    _FT_MODEL_CACHE[cache_key] = model
    logger.info("Fine-tuned model loaded and cached.")
    return model


def release_finetuned_model() -> None:
    """Evict cached fine-tuned model(s) and conditioning latents, free VRAM.

    Useful before loading another large model (e.g., another checkpoint, or
    switching back to the base XTTS-v2 in ``voicelegacy.synthesis``).
    """
    global _FT_MODEL_CACHE, _FT_LATENTS_CACHE
    _FT_MODEL_CACHE.clear()
    _FT_LATENTS_CACHE.clear()
    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        pass
    logger.info("Fine-tuned model + latent caches released.")


# ─── Conditioning latents (replicates the synthesis.py contract) ──────────
def _conditioning_cache_key(reference_wavs: list[str]) -> str:
    """SHA256 of the reference set: identifies the speaker conditioning."""
    h = hashlib.sha256()
    for raw in sorted(reference_wavs):
        p = Path(raw)
        stat = p.stat()
        h.update(str(p.resolve()).encode("utf-8"))
        h.update(str(stat.st_size).encode("ascii"))
        h.update(str(stat.st_mtime_ns).encode("ascii"))
    return h.hexdigest()


def _get_conditioning_latents_cached(
    model: Any,
    reference_wavs: list[str],
    checkpoint_fingerprint: str,
) -> tuple[Any, Any]:
    """Get or compute XTTS conditioning latents for a reference set.

    The cache key includes the checkpoint fingerprint so latents computed
    against checkpoint A are never returned when the user is inferring with
    checkpoint B (they would still be numerically valid but semantically
    wrong).
    """
    key = f"{checkpoint_fingerprint}|{_conditioning_cache_key(reference_wavs)}"
    if key in _FT_LATENTS_CACHE:
        logger.info("Using cached fine-tuned conditioning latents key={}", key[:24])
        return _FT_LATENTS_CACHE[key]

    logger.info(
        "Computing fine-tuned XTTS conditioning latents for {} reference file(s)...",
        len(reference_wavs),
    )
    latents = model.get_conditioning_latents(audio_path=reference_wavs)
    if not isinstance(latents, tuple) or len(latents) != 2:
        raise RuntimeError("Fine-tuned Xtts.get_conditioning_latents returned an unexpected value")
    _FT_LATENTS_CACHE[key] = latents
    return latents


def _tensor_to_numpy_1d(wav: Any) -> np.ndarray:
    """Convert XTTS output audio to a 1-D float32 numpy array."""
    try:
        import torch

        if isinstance(wav, torch.Tensor):
            wav = wav.detach().cpu().numpy()
    except ImportError:
        pass
    arr = np.asarray(wav, dtype=np.float32).squeeze()
    if arr.ndim != 1 or arr.size == 0:
        raise RuntimeError("Fine-tuned XTTS inference produced empty/non-1D audio")
    return arr


def _apply_seed(seed: int | None) -> None:
    """Apply a deterministic seed across the libraries XTTS uses.

    Replicates ``voicelegacy.synthesis._apply_seed`` to keep reproducibility
    semantics identical between base and fine-tuned inference paths.
    """
    if seed is None:
        return
    import random

    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


# ─── Public synthesis entry point ─────────────────────────────────────────
def synthesize_with_finetuned(
    model: Any,
    checkpoint: FineTunedCheckpoint,
    text: str,
    speaker_wav: Path | list[Path],
    output_path: Path,
    config: SynthesisConfig,
) -> Path:
    """Run inference with a fine-tuned XTTS-v2 model and save to WAV.

    This function intentionally mirrors the public signature of
    ``voicelegacy.synthesis.synthesize_to_file`` so it is a drop-in
    replacement at the call site — except that ``tts`` is replaced by the
    pair ``(model, checkpoint)`` because the high-level wrapper is not used.

    Args:
        model: Loaded Xtts model (from ``load_finetuned_model``).
        checkpoint: The FineTunedCheckpoint that produced ``model`` — used
            for the conditioning-latents cache key + sidecar audit.
        text: Text to synthesize. Cannot be empty.
        speaker_wav: Path or list of paths to reference audio (the same
            reference corpus used with the base model is fine).
        output_path: Destination WAV path.
        config: SynthesisConfig (the same one voicelegacy.synthesis uses).

    Returns:
        Path to the written file (same as ``output_path``).

    Raises:
        ValueError: If text is empty or any reference file is missing.
        RuntimeError: If inference returns an unexpected shape.
    """
    if not text.strip():
        raise ValueError("Cannot synthesize empty text.")

    if isinstance(speaker_wav, str | Path):
        speaker_wav_list = [str(speaker_wav)]
    else:
        speaker_wav_list = [str(p) for p in speaker_wav]

    for p in speaker_wav_list:
        if not Path(p).exists():
            raise ValueError(f"Reference audio not found: {p}")

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    _apply_seed(config.seed)

    text_plan = plan_text_synthesis(text, config)
    if text_plan.warning:
        logger.warning("Text synthesis warning: {}", text_plan.warning)

    logger.info(
        "Synthesizing {} char(s) to {} with fine-tuned checkpoint fp={} "
        "({} ref file(s), seed={}, split_sentences={})",
        len(text),
        output_path.name,
        checkpoint.fingerprint,
        len(speaker_wav_list),
        config.seed,
        text_plan.xtts_split_sentences,
    )

    gpt_cond_latent, speaker_embedding = _get_conditioning_latents_cached(
        model=model,
        reference_wavs=speaker_wav_list,
        checkpoint_fingerprint=checkpoint.fingerprint,
    )

    out = model.inference(
        text,
        config.language,
        gpt_cond_latent,
        speaker_embedding,
        temperature=config.temperature,
        length_penalty=config.length_penalty,
        repetition_penalty=config.repetition_penalty,
        top_k=config.top_k,
        top_p=config.top_p,
        speed=config.speed,
        enable_text_splitting=text_plan.xtts_split_sentences,
    )
    if not isinstance(out, dict) or "wav" not in out:
        raise RuntimeError(
            "Fine-tuned Xtts.inference did not return {'wav': ...}. "
            "API likely changed in this coqui-tts version."
        )
    wav = _tensor_to_numpy_1d(out["wav"])
    sf.write(str(output_path), wav, XTTS_OUTPUT_SR, subtype="PCM_16")
    logger.info(
        "Wrote {} ({} Hz) using fine-tuned checkpoint fp={}",
        output_path,
        XTTS_OUTPUT_SR,
        checkpoint.fingerprint,
    )
    return output_path


def is_available() -> bool:
    """Return True if coqui-tts is importable for fine-tuned inference.

    The fine-tuning notebook installs coqui-tts via pip, so this should
    almost always be True in a Colab session. Used by the pipeline to
    decide whether to surface the ``--use-finetuned`` flag.
    """
    try:
        import TTS.tts.configs.xtts_config
        import TTS.tts.models.xtts  # noqa: F401

        return True
    except ImportError:
        return False
