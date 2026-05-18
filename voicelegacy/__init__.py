"""voicelegacy — Voice cloning pipeline for family legacy.

Builds curated reference corpora from diarized interviews (via speakerscribe)
and synthesizes new speech in the target speaker's voice using XTTS-v2.

Public API:
    >>> from voicelegacy import (
    ...     PipelineConfig, ReferenceConfig, SynthesisConfig, WorkspacePaths,
    ...     run_reference_phase, run_synthesis, run_batch_synthesis,
    ... )
    >>> paths = WorkspacePaths(workspace="/content/drive/MyDrive/Legado")
    >>> config = PipelineConfig(
    ...     reference=ReferenceConfig(target_speaker_label="SPEAKER_00"),
    ...     synthesis=SynthesisConfig(language="es"),
    ...     accept_coqui_tos=True,
    ... )
    >>> corpus = run_reference_phase(paths, config)
    >>> result = run_synthesis("Hola mi nieto.", corpus.top_wavs, paths, config)
"""

from voicelegacy.config import (
    MAX_REF_DURATION_S,
    MIN_REF_DURATION_S,
    MIN_SAMPLING_RATE_HZ,
    MIN_SNR_DB,
    MIN_USABLE_REFERENCE_SEGMENTS,
    XTTS_INPUT_SR,
    XTTS_OUTPUT_SR,
    PipelineConfig,
    ReferenceConfig,
    SynthesisConfig,
    WorkspacePaths,
)
from voicelegacy.logging_config import configure_logging, get_logger
from voicelegacy.pipeline import (
    CorpusBuildResult,
    SynthesisResult,
    run_batch_synthesis,
    run_reference_phase,
    run_synthesis,
)
from voicelegacy.quality import QualityReport, evaluate_file, rank_candidates
from voicelegacy.synthesis import load_xtts_model, release_model

__version__ = "0.1.0"

__all__ = [
    "MAX_REF_DURATION_S",
    # Constants
    "MIN_REF_DURATION_S",
    "MIN_SAMPLING_RATE_HZ",
    "MIN_SNR_DB",
    "MIN_USABLE_REFERENCE_SEGMENTS",
    "XTTS_INPUT_SR",
    "XTTS_OUTPUT_SR",
    # Pipeline
    "CorpusBuildResult",
    # Config
    "PipelineConfig",
    # Quality
    "QualityReport",
    "ReferenceConfig",
    "SynthesisConfig",
    "SynthesisResult",
    "WorkspacePaths",
    # Logging
    "configure_logging",
    "evaluate_file",
    "get_logger",
    # Synthesis
    "load_xtts_model",
    "rank_candidates",
    "release_model",
    "run_batch_synthesis",
    "run_reference_phase",
    "run_synthesis",
]
