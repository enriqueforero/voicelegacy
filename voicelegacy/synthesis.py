"""voicelegacy — XTTS-v2 inference wrapper.

Thin wrapper around `coqui-tts` (the Idiap fork) that:
- Validates that CUDA / VRAM are available before model load.
- Sets the COQUI_TOS_AGREED env var explicitly (user must accept CPML).
- Caches the loaded model on the module level to avoid re-loading on every call.
- Frees VRAM on demand.

Why the env var: XTTS-v2 weights are released under the Coqui Public Model
License (CPML), which requires explicit acceptance. The library checks
COQUI_TOS_AGREED=1 before loading. See: https://coqui.ai/cpml
"""

from __future__ import annotations

import gc
import os
from pathlib import Path
from typing import Any

from voicelegacy.config import XTTS_OUTPUT_SR, SynthesisConfig
from voicelegacy.logging_config import get_logger

logger = get_logger()

# Module-level cache. The model is large (~2 GB on disk, 2-3 GB VRAM) and
# expensive to instantiate. Reload only when explicitly requested.
_MODEL_CACHE: dict[str, Any] = {}


def _resolve_device(requested: str) -> str:
    """Resolve 'auto' to 'cuda' if available, else 'cpu'."""
    if requested != "auto":
        return requested
    try:
        import torch  # local import — avoid forcing torch on test rigs without it

        return "cuda" if torch.cuda.is_available() else "cpu"
    except ImportError:
        return "cpu"


def load_xtts_model(config: SynthesisConfig, accept_tos: bool) -> Any:
    """Load (or fetch from cache) an XTTS-v2 model.

    Args:
        config: SynthesisConfig with model_name and device.
        accept_tos: Must be True. Setting it sets COQUI_TOS_AGREED=1.

    Returns:
        Initialized TTS object ready for inference.

    Raises:
        RuntimeError: If accept_tos is False.
        ImportError: If coqui-tts is not installed.
    """
    if not accept_tos:
        raise RuntimeError(
            "You must accept the Coqui Public Model License (CPML) before loading XTTS-v2. "
            "Read it at https://coqui.ai/cpml then pass accept_tos=True."
        )

    os.environ["COQUI_TOS_AGREED"] = "1"

    cache_key = f"{config.model_name}|{config.device}"
    if cache_key in _MODEL_CACHE:
        logger.info("Using cached model: {}", cache_key)
        return _MODEL_CACHE[cache_key]

    try:
        from TTS.api import TTS
    except ImportError as exc:
        raise ImportError(
            "coqui-tts is not installed. Install with: pip install coqui-tts"
        ) from exc

    device = _resolve_device(config.device)
    logger.info("Loading XTTS model '{}' on device '{}'", config.model_name, device)

    tts = TTS(config.model_name)
    tts = tts.to(device)

    _MODEL_CACHE[cache_key] = tts
    logger.info("Model loaded and cached.")
    return tts


def release_model() -> None:
    """Evict cached model(s) and free VRAM.

    Call before switching to another large model or between very long
    notebook sessions where Colab is reclaiming idle resources.
    """
    global _MODEL_CACHE
    _MODEL_CACHE.clear()
    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        logger.info("Model released, VRAM freed.")
    except ImportError:
        pass


def synthesize_to_file(
    tts: Any,
    text: str,
    speaker_wav: Path | list[Path],
    output_path: Path,
    config: SynthesisConfig,
) -> Path:
    """Run XTTS-v2 inference and save the result to a WAV file.

    Args:
        tts: Loaded TTS object (from load_xtts_model).
        text: Text to synthesize. Will be auto-split at sentence boundaries
            if config.enable_text_splitting is True.
        speaker_wav: Path (or list of paths) to reference audio for voice
            conditioning. Passing a list averages the speaker latents — useful
            when you have multiple short clean segments.
        output_path: Where to write the output WAV.
        config: SynthesisConfig with language and decoder hyperparameters.

    Returns:
        Path to the written file.

    Raises:
        ValueError: If text is empty or reference files don't exist.
    """
    if not text.strip():
        raise ValueError("Cannot synthesize empty text.")

    # Normalize speaker_wav to a list of str (TTS API accepts either)
    if isinstance(speaker_wav, str | Path):
        speaker_wav_list = [str(speaker_wav)]
    else:
        speaker_wav_list = [str(p) for p in speaker_wav]

    for p in speaker_wav_list:
        if not Path(p).exists():
            raise ValueError(f"Reference audio not found: {p}")

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    logger.info(
        "Synthesizing {} char(s) to {} with {} ref file(s)",
        len(text),
        output_path.name,
        len(speaker_wav_list),
    )

    tts.tts_to_file(
        text=text,
        file_path=str(output_path),
        speaker_wav=speaker_wav_list if len(speaker_wav_list) > 1 else speaker_wav_list[0],
        language=config.language,
        split_sentences=config.enable_text_splitting,
        # XTTS-v2 supports these decoder kwargs via the API:
        temperature=config.temperature,
        length_penalty=config.length_penalty,
        repetition_penalty=config.repetition_penalty,
        top_k=config.top_k,
        top_p=config.top_p,
        speed=config.speed,
    )

    logger.info("Wrote {} ({} Hz)", output_path, XTTS_OUTPUT_SR)
    return output_path
