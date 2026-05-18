"""voicelegacy — Audio preprocessing utilities.

Functions to load, clean, normalize, resample, and segment audio files so
they meet XTTS-v2 input expectations (mono, 22050 Hz, clean, properly
normalized).

Why these specific steps:
- Resample to 22050 Hz: XTTS-v2 conditioning encoder was trained at this rate.
- Mono: XTTS-v2 ignores stereo info; converting saves memory.
- Loudness normalize to -23 LUFS: EBU R128 broadcast standard; avoids
  the conditioning network seeing wildly varying signal levels.
- Spectral denoise: XTTS-v2 will faithfully clone background noise if it's
  in the reference. Better to remove it.
- Trim silences: pad-silence in the reference makes the model think pauses
  are part of the speaker's style.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from pathlib import Path

import librosa
import noisereduce as nr
import numpy as np
import pyloudnorm as pyln
import soundfile as sf

from voicelegacy.config import XTTS_INPUT_SR
from voicelegacy.logging_config import get_logger

logger = get_logger()

# Suppress librosa/soundfile deprecation noise during batch processing
warnings.filterwarnings("ignore", category=UserWarning, module="librosa")


@dataclass(frozen=True)
class AudioStats:
    """Lightweight container for audio analysis results."""

    duration_s: float
    sample_rate: int
    n_channels: int
    rms_db: float
    peak_db: float
    snr_db: float


# ─── Loading ───────────────────────────────────────────────────────
def load_audio_mono(path: Path, target_sr: int = XTTS_INPUT_SR) -> np.ndarray:
    """Load audio as mono float32, resampled to target_sr.

    Args:
        path: Path to input audio file (any format soundfile/librosa supports).
        target_sr: Target sampling rate in Hz. Defaults to XTTS-v2's 22050.

    Returns:
        1-D float32 numpy array, values in [-1, 1].

    Raises:
        FileNotFoundError: If `path` does not exist.
        RuntimeError: If the file cannot be decoded.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Audio file not found: {path}")

    try:
        # librosa handles resampling and mono conversion robustly.
        y, _sr = librosa.load(str(path), sr=target_sr, mono=True, dtype=np.float32)
    except Exception as exc:
        raise RuntimeError(f"Failed to decode {path.name}: {exc}") from exc

    if y.size == 0:
        raise RuntimeError(f"Empty audio after load: {path}")

    return y


def compute_stats(y: np.ndarray, sr: int) -> AudioStats:
    """Compute basic acoustic statistics for a mono audio array.

    Args:
        y: Mono audio, float32 in [-1, 1].
        sr: Sample rate in Hz.

    Returns:
        AudioStats with duration, levels, and SNR estimate.
    """
    duration_s = float(len(y) / sr)
    rms = float(np.sqrt(np.mean(y**2)) + 1e-12)
    peak = float(np.max(np.abs(y)) + 1e-12)
    rms_db = 20.0 * np.log10(rms)
    peak_db = 20.0 * np.log10(peak)
    snr_db = _estimate_snr_db(y)

    return AudioStats(
        duration_s=duration_s,
        sample_rate=sr,
        n_channels=1,
        rms_db=rms_db,
        peak_db=peak_db,
        snr_db=snr_db,
    )


def _estimate_snr_db(y: np.ndarray, frame_length: int = 2048, hop_length: int = 512) -> float:
    """Estimate SNR by comparing top-decile vs bottom-decile frame energies.

    This is a crude but useful heuristic: clean speech has a wide dynamic range
    between voiced frames and silent frames, while noisy audio has a narrow one.

    Args:
        y: Mono audio array.
        frame_length: STFT-equivalent frame size.
        hop_length: Hop between frames.

    Returns:
        Estimated SNR in dB. Higher = cleaner.
    """
    if len(y) < frame_length:
        return 0.0

    # Frame-wise RMS energy
    frames = librosa.util.frame(y, frame_length=frame_length, hop_length=hop_length, axis=0)
    rms = np.sqrt(np.mean(frames**2, axis=0) + 1e-12)
    rms_sorted = np.sort(rms)

    n = len(rms_sorted)
    if n < 10:
        return 0.0

    # Top 10% = speech, bottom 10% = noise
    noise_rms = float(np.mean(rms_sorted[: max(1, n // 10)]))
    signal_rms = float(np.mean(rms_sorted[-max(1, n // 10) :]))

    if noise_rms <= 1e-9:
        return 60.0  # essentially noiseless

    return float(20.0 * np.log10(signal_rms / noise_rms))


# ─── Cleaning ──────────────────────────────────────────────────────
def denoise(y: np.ndarray, sr: int, prop_decrease: float = 0.85) -> np.ndarray:
    """Apply spectral-gating noise reduction.

    Uses the noisereduce library (stationary noise model). Removes background
    hiss, fan noise, and broadband floor without distorting speech if
    prop_decrease is kept moderate.

    Args:
        y: Mono audio array.
        sr: Sample rate.
        prop_decrease: How aggressively to attenuate noise (0-1).
            0.85 is a good balance for speech.

    Returns:
        Denoised audio, same length as input.
    """
    try:
        y_clean = nr.reduce_noise(y=y, sr=sr, prop_decrease=prop_decrease, stationary=True)
    except Exception as exc:
        logger.warning("Denoise failed ({}), returning original.", exc)
        return y
    return np.asarray(y_clean, dtype=np.float32)


def loudness_normalize(y: np.ndarray, sr: int, target_lufs: float = -23.0) -> np.ndarray:
    """Normalize audio to target loudness (EBU R128 / LUFS).

    Args:
        y: Mono audio.
        sr: Sample rate.
        target_lufs: Target integrated loudness. -23 LUFS = EBU broadcast standard.

    Returns:
        Loudness-normalized audio, clipped to [-1, 1].
    """
    meter = pyln.Meter(sr)  # BS.1770 meter
    try:
        loudness = meter.integrated_loudness(y)
    except ValueError:
        # Happens when signal is too short for gated measurement
        logger.warning("Loudness measurement failed (signal too short), using peak normalization.")
        peak = np.max(np.abs(y)) + 1e-9
        return np.clip(y * (0.9 / peak), -1.0, 1.0).astype(np.float32)

    y_norm = pyln.normalize.loudness(y, loudness, target_lufs)
    # Safety clip — loudness normalization can produce values > 1.0
    return np.clip(y_norm, -1.0, 1.0).astype(np.float32)


def trim_silence(
    y: np.ndarray, top_db: float = 35.0, frame_length: int = 2048, hop_length: int = 512
) -> np.ndarray:
    """Trim leading and trailing silence using librosa.effects.trim.

    Args:
        y: Mono audio.
        top_db: Silence threshold below the peak in dB.
        frame_length: Analysis frame size.
        hop_length: Frame hop.

    Returns:
        Trimmed audio (may be empty if everything was below threshold).
    """
    y_trim, _ = librosa.effects.trim(
        y, top_db=top_db, frame_length=frame_length, hop_length=hop_length
    )
    return np.asarray(y_trim, dtype=np.float32)


# ─── Persistence ───────────────────────────────────────────────────
def save_wav(path: Path, y: np.ndarray, sr: int, subtype: str = "PCM_16") -> None:
    """Save mono audio array to a WAV file.

    Args:
        path: Output path. Parent directory is created if missing.
        y: Mono audio in [-1, 1].
        sr: Sample rate.
        subtype: soundfile subtype. PCM_16 is universal; PCM_24 for higher fidelity.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(path), y, sr, subtype=subtype)


# ─── Slicing ───────────────────────────────────────────────────────
def slice_segment(
    y: np.ndarray, sr: int, start_s: float, end_s: float, pad_s: float = 0.0
) -> np.ndarray:
    """Extract a segment by time, with optional symmetric padding.

    Args:
        y: Full audio array.
        sr: Sample rate.
        start_s: Segment start in seconds.
        end_s: Segment end in seconds.
        pad_s: Extra context to include on each side (clamped to bounds).

    Returns:
        Sliced audio array.
    """
    if end_s <= start_s:
        raise ValueError(f"end_s ({end_s}) must be > start_s ({start_s})")

    n = len(y)
    i0 = max(0, int((start_s - pad_s) * sr))
    i1 = min(n, int((end_s + pad_s) * sr))
    return y[i0:i1]


# ─── Pipeline shortcut ─────────────────────────────────────────────
def preprocess_full(
    path: Path,
    target_sr: int = XTTS_INPUT_SR,
    apply_denoise: bool = True,
    target_lufs: float = -23.0,
) -> tuple[np.ndarray, AudioStats]:
    """One-shot preprocessing for a single file.

    Loads → denoises → trims silence → loudness-normalizes → returns clean audio
    plus before/after stats. This is what you call when you have a single clean
    reference recording and just want it cleaned up.

    Args:
        path: Path to input file.
        target_sr: Target sample rate.
        apply_denoise: Whether to apply spectral denoising.
        target_lufs: Target loudness.

    Returns:
        Tuple of (clean_audio, stats_after_cleanup).
    """
    y = load_audio_mono(path, target_sr=target_sr)
    if apply_denoise:
        y = denoise(y, target_sr)
    y = trim_silence(y)
    y = loudness_normalize(y, target_sr, target_lufs=target_lufs)
    stats = compute_stats(y, target_sr)
    return y, stats
