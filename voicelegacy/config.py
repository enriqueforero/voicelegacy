"""voicelegacy — Centralized configuration with Pydantic v2 validation.

Single source of truth for the voice-cloning pipeline. Zero magic numbers in
business logic — every tunable lives here.

Context: Google Colab Free Tier (T4 GPU, ~12 GB system RAM, 15 GB VRAM).
Model: XTTS-v2 (coqui-tts fork) for Spanish zero-shot voice cloning.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

# ─── Public constants ──────────────────────────────────────────────
# XTTS-v2 native sampling rates
XTTS_INPUT_SR: int = 22050
"""Sampling rate XTTS-v2 expects for reference audio."""

XTTS_OUTPUT_SR: int = 24000
"""Sampling rate XTTS-v2 generates output at."""

# Reference audio constraints — empirical from XTTS-v2 papers and field tests
MIN_REF_DURATION_S: float = 6.0
"""Below this, the speaker conditioning is too weak — voice drift kicks in."""

MAX_REF_DURATION_S: float = 30.0
"""Above this, no quality gain and Perceiver context window saturates."""

# Quality gating thresholds
MIN_SNR_DB: float = 15.0
"""Below this SNR, the reference is too noisy — clone will sound muffled."""

MIN_SAMPLING_RATE_HZ: int = 16000
"""Anything below 16kHz means phone-codec audio. Reject — see README §Why."""

MIN_USABLE_REFERENCE_SEGMENTS: int = 3
"""Hard floor on top-N references before we'll attempt synthesis.

Below 3 clean reference segments, XTTS-v2 zero-shot drifts noticeably and
the result is usually unsuitable for legacy use. Better to fail loudly and
let the user collect more material than to ship a bad clone."""

SupportedLanguage = Literal[
    "es",
    "en",
    "pt",
    "fr",
    "it",
    "de",
    "pl",
    "tr",
    "ru",
    "nl",
    "cs",
    "ar",
    "zh-cn",
    "ja",
    "hu",
    "ko",
]
"""Languages supported by XTTS-v2."""

DeviceType = Literal["auto", "cuda", "cpu"]


# ─── ReferenceConfig ───────────────────────────────────────────────
class ReferenceConfig(BaseModel):
    """Configuration for building the reference-audio corpus.

    The reference corpus is the set of clean speech segments from the target
    speaker that XTTS-v2 will use to condition voice generation. Quality of
    this corpus dominates the final result — garbage in, garbage out.
    """

    model_config = ConfigDict(frozen=False, validate_assignment=True)

    target_speaker_label: str = Field(
        default="SPEAKER_00",
        description="Speaker label in the speakerscribe JSON to extract (e.g. 'SPEAKER_00').",
    )
    min_segment_duration_s: float = Field(
        default=4.0,
        ge=1.0,
        le=60.0,
        description="Drop segments shorter than this (too short = weak signal).",
    )
    max_segment_duration_s: float = Field(
        default=15.0,
        ge=1.0,
        le=60.0,
        description="Drop segments longer than this (likely contains pauses or overlap).",
    )
    target_loudness_lufs: float = Field(
        default=-23.0,
        ge=-30.0,
        le=-12.0,
        description="EBU R128 target loudness for normalization.",
    )
    apply_denoise: bool = Field(
        default=True,
        description="Apply spectral-gating noise reduction (noisereduce).",
    )
    top_n_segments: int = Field(
        default=10,
        ge=1,
        le=100,
        description="Keep only the N best-quality segments (by SNR + duration).",
    )
    min_snr_db: float = Field(
        default=MIN_SNR_DB,
        ge=0.0,
        le=60.0,
        description="Reject segments below this SNR.",
    )

    @model_validator(mode="after")
    def _check_duration_range(self) -> ReferenceConfig:
        if self.max_segment_duration_s <= self.min_segment_duration_s:
            raise ValueError(
                f"max_segment_duration_s ({self.max_segment_duration_s}) must be "
                f"> min_segment_duration_s ({self.min_segment_duration_s})"
            )
        return self


# ─── SynthesisConfig ───────────────────────────────────────────────
class SynthesisConfig(BaseModel):
    """Configuration for the XTTS-v2 inference step."""

    model_config = ConfigDict(frozen=False, validate_assignment=True)

    model_name: str = Field(
        default="tts_models/multilingual/multi-dataset/xtts_v2",
        description="Coqui TTS model identifier. Don't change unless you know what you're doing.",
    )
    language: SupportedLanguage = Field(
        default="es",
        description="ISO code for target language. 'es' = Spanish.",
    )
    device: DeviceType = Field(
        default="auto",
        description="'auto' detects CUDA, falls back to CPU.",
    )
    temperature: float = Field(
        default=0.7,
        ge=0.1,
        le=1.5,
        description="XTTS sampling temperature. Lower = more stable, higher = more expressive.",
    )
    length_penalty: float = Field(
        default=1.0,
        ge=0.5,
        le=2.0,
        description="Penalty on output length. >1 = shorter outputs.",
    )
    repetition_penalty: float = Field(
        default=5.0,
        ge=1.0,
        le=20.0,
        description="Discourages repeating tokens. 5.0 is XTTS-v2 default.",
    )
    top_k: int = Field(default=50, ge=0, le=200)
    top_p: float = Field(default=0.85, ge=0.0, le=1.0)
    speed: float = Field(
        default=1.0,
        ge=0.5,
        le=2.0,
        description="Speech rate multiplier. 1.0 = natural pace.",
    )
    enable_text_splitting: bool = Field(
        default=True,
        description="Auto-split long texts at sentence boundaries (recommended).",
    )


# ─── WorkspacePaths ────────────────────────────────────────────────
class WorkspacePaths(BaseModel):
    """Filesystem layout for a voicelegacy workspace.

    Convention (mirrors speakerscribe):
        workspace/
        ├── interviews_raw/      ← raw audio (mp3, wav, m4a) input
        ├── speakerscribe_out/   ← .json outputs from speakerscribe (diarization)
        ├── reference_corpus/    ← curated clean reference segments (built by us)
        ├── synthesis_out/       ← generated audio files
        ├── reports/             ← quality reports, metadata
        └── runs.db              ← idempotency cache (content-hash → output)
    """

    model_config = ConfigDict(frozen=False, validate_assignment=True)

    workspace: Path = Field(description="Root workspace folder (typically on Drive).")

    @field_validator("workspace", mode="before")
    @classmethod
    def _coerce_to_path(cls, v: object) -> Path:
        return Path(v) if not isinstance(v, Path) else v

    @property
    def interviews_raw(self) -> Path:
        return self.workspace / "interviews_raw"

    @property
    def speakerscribe_out(self) -> Path:
        return self.workspace / "speakerscribe_out"

    @property
    def reference_corpus(self) -> Path:
        return self.workspace / "reference_corpus"

    @property
    def synthesis_out(self) -> Path:
        return self.workspace / "synthesis_out"

    @property
    def reports(self) -> Path:
        return self.workspace / "reports"

    @property
    def db_path(self) -> Path:
        return self.workspace / "runs.db"

    def mkdirs(self) -> None:
        """Create all expected subdirectories. Idempotent."""
        for p in (
            self.interviews_raw,
            self.speakerscribe_out,
            self.reference_corpus,
            self.synthesis_out,
            self.reports,
        ):
            p.mkdir(parents=True, exist_ok=True)


# ─── PipelineConfig (top-level) ────────────────────────────────────
class PipelineConfig(BaseModel):
    """Top-level pipeline configuration. Bundles all sub-configs."""

    model_config = ConfigDict(frozen=False, validate_assignment=True)

    reference: ReferenceConfig = Field(default_factory=ReferenceConfig)
    synthesis: SynthesisConfig = Field(default_factory=SynthesisConfig)
    force_rebuild_reference: bool = Field(
        default=False,
        description="If True, rebuild reference corpus from scratch (ignores cache).",
    )
    force_resynthesize: bool = Field(
        default=False,
        description="If True, re-run synthesis even if output already exists.",
    )
    accept_coqui_tos: bool = Field(
        default=False,
        description=(
            "Must be explicitly set to True. Setting this means you have read and "
            "accept the Coqui Public Model License (CPML): https://coqui.ai/cpml"
        ),
    )

    @model_validator(mode="after")
    def _enforce_tos(self) -> PipelineConfig:
        if not self.accept_coqui_tos:
            raise ValueError(
                "You must explicitly accept the CPML license by setting "
                "accept_coqui_tos=True. Read it first: https://coqui.ai/cpml"
            )
        return self
