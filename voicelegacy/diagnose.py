"""Operational diagnostics for a voicelegacy workspace."""

from __future__ import annotations

import importlib.metadata as metadata
import os
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from voicelegacy.audio import EXTENSIONS_CONVERTIBLE_TO_WAV
from voicelegacy.config import MIN_USABLE_REFERENCE_SEGMENTS, SynthesisConfig, WorkspacePaths
from voicelegacy.speakerscribe_schema import load_and_validate_speakerscribe_document
from voicelegacy.telemetry import runtime_snapshot

Status = Literal["ok", "warn", "fail"]


@dataclass(frozen=True)
class DiagnosticCheck:
    """One diagnostic check result."""

    name: str
    status: Status
    detail: str
    remediation: str | None = None

    def to_dict(self) -> dict[str, object]:
        return {
            "name": self.name,
            "status": self.status,
            "detail": self.detail,
            "remediation": self.remediation,
        }


@dataclass(frozen=True)
class DiagnosticReport:
    """Complete diagnosis for a workspace and local runtime."""

    workspace: Path
    checks: list[DiagnosticCheck]

    @property
    def ready(self) -> bool:
        """True when no hard failures were detected."""
        return all(c.status != "fail" for c in self.checks)

    @property
    def failed(self) -> int:
        return sum(1 for c in self.checks if c.status == "fail")

    @property
    def warnings(self) -> int:
        return sum(1 for c in self.checks if c.status == "warn")

    def to_dict(self) -> dict[str, object]:
        return {
            "workspace": str(self.workspace),
            "ready": self.ready,
            "failed": self.failed,
            "warnings": self.warnings,
            "checks": [c.to_dict() for c in self.checks],
        }


def _package_version(name: str) -> str | None:
    if name == "voicelegacy":
        try:
            import voicelegacy

            return voicelegacy.__version__
        except Exception:
            pass
    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError:
        return None


def _count_files(path: Path, patterns: tuple[str, ...]) -> int:
    if not path.exists():
        return 0
    return sum(1 for pattern in patterns for _ in path.glob(pattern))


def _validate_speakerscribe_dir(path: Path) -> tuple[int, int, str]:
    json_files = sorted(path.glob("*.json")) if path.exists() else []
    invalid: list[str] = []
    for jf in json_files:
        try:
            load_and_validate_speakerscribe_document(jf)
        except Exception as exc:
            invalid.append(f"{jf.name}: {exc}")
    detail = f"{len(json_files)} JSON file(s); {len(invalid)} invalid"
    if invalid:
        detail += " | " + "; ".join(invalid[:3])
        if len(invalid) > 3:
            detail += f"; +{len(invalid) - 3} more"
    return len(json_files), len(invalid), detail


def diagnose_workspace(
    workspace: Path,
    *,
    require_gpu: bool = False,
    synthesis_config: SynthesisConfig | None = None,
) -> DiagnosticReport:
    """Run operational diagnostics without loading XTTS weights.

    The goal is to tell a user whether the workspace is structurally ready and
    whether the local runtime has the minimum prerequisites. We intentionally do
    not load the model weights here; that can take minutes and should remain an
    explicit synthesis action.
    """
    paths = WorkspacePaths(workspace=workspace)
    cfg = synthesis_config or SynthesisConfig()
    checks: list[DiagnosticCheck] = []

    py_ok = sys.version_info >= (3, 10)
    checks.append(
        DiagnosticCheck(
            "python",
            "ok" if py_ok else "fail",
            f"{sys.version.split()[0]} ({sys.executable})",
            None if py_ok else "Use Python >=3.10.",
        )
    )

    for pkg in (
        "voicelegacy",
        "librosa",
        "soundfile",
        "noisereduce",
        "pyloudnorm",
        "pydantic",
        "typer",
    ):
        version = _package_version(pkg)
        checks.append(
            DiagnosticCheck(
                f"package:{pkg}",
                "ok" if version else "fail",
                version or "not installed",
                None if version else f"Install package dependency: {pkg}",
            )
        )

    coqui_version = _package_version("coqui-tts") or _package_version("TTS")
    checks.append(
        DiagnosticCheck(
            "package:coqui-tts",
            "ok" if coqui_version else "warn",
            coqui_version or "not installed/importable by package metadata",
            None if coqui_version else "Install coqui-tts before synthesis in Colab/local runtime.",
        )
    )

    ffmpeg_path = shutil.which("ffmpeg")
    checks.append(
        DiagnosticCheck(
            "ffmpeg",
            "ok" if ffmpeg_path else "warn",
            ffmpeg_path or "not found on PATH",
            None if ffmpeg_path else "Install ffmpeg to convert mp4/m4a/mkv/webm/aac inputs.",
        )
    )

    snap = runtime_snapshot()
    checks.append(
        DiagnosticCheck(
            "cuda",
            "ok" if snap.cuda_available else ("fail" if require_gpu else "warn"),
            str(snap.to_dict()),
            None
            if snap.cuda_available
            else "Use a CUDA runtime for practical XTTS-v2 synthesis, or omit --require-gpu.",
        )
    )

    cpml_ok = os.getenv("COQUI_TOS_AGREED") == "1"
    checks.append(
        DiagnosticCheck(
            "coqui_cpml_acceptance",
            "ok" if cpml_ok else "warn",
            "COQUI_TOS_AGREED=1" if cpml_ok else "COQUI_TOS_AGREED is not set to 1",
            None
            if cpml_ok
            else "Pass --accept-tos for synthesis only after reading https://coqui.ai/cpml.",
        )
    )

    checks.append(
        DiagnosticCheck(
            "xtts_model_config",
            "ok",
            f"{cfg.model_name} | language={cfg.language} | device={cfg.device}",
        )
    )

    for name, path in (
        ("workspace", paths.workspace),
        ("interviews_raw", paths.interviews_raw),
        ("speakerscribe_out", paths.speakerscribe_out),
        ("reference_corpus", paths.reference_corpus),
        ("synthesis_out", paths.synthesis_out),
        ("reports", paths.reports),
    ):
        exists = path.exists()
        checks.append(
            DiagnosticCheck(
                f"dir:{name}",
                "ok" if exists else "fail",
                str(path) if exists else f"missing: {path}",
                None if exists else f"Create {path} or run WorkspacePaths(...).mkdirs().",
            )
        )

    raw_patterns = (
        "*.wav",
        "*.mp3",
        "*.m4a",
        "*.flac",
        "*.ogg",
        *(f"*{ext}" for ext in EXTENSIONS_CONVERTIBLE_TO_WAV),
    )
    raw_count = _count_files(paths.interviews_raw, raw_patterns)
    checks.append(
        DiagnosticCheck(
            "raw_audio",
            "ok" if raw_count else "warn",
            f"{raw_count} audio/container file(s) found",
            None if raw_count else "Place interviews in interviews_raw/.",
        )
    )

    json_count, invalid_count, json_detail = _validate_speakerscribe_dir(paths.speakerscribe_out)
    checks.append(
        DiagnosticCheck(
            "speakerscribe_json",
            "ok" if json_count and not invalid_count else ("fail" if invalid_count else "warn"),
            json_detail,
            None
            if json_count and not invalid_count
            else "Run speakerscribe and fix JSON schema errors before building corpus.",
        )
    )

    ref_count = _count_files(paths.reference_corpus, ("*.wav",))
    checks.append(
        DiagnosticCheck(
            "reference_corpus",
            "ok" if ref_count >= MIN_USABLE_REFERENCE_SEGMENTS else "warn",
            f"{ref_count} reference WAV(s); minimum recommended={MIN_USABLE_REFERENCE_SEGMENTS}",
            None
            if ref_count >= MIN_USABLE_REFERENCE_SEGMENTS
            else "Run build-corpus or collect more clean source audio.",
        )
    )

    report_count = _count_files(paths.reports, ("*.json",))
    checks.append(
        DiagnosticCheck(
            "reports",
            "ok" if report_count else "warn",
            f"{report_count} report JSON file(s) found",
            None if report_count else "Run build-corpus/synthesis to generate audit reports.",
        )
    )

    synth_wavs = _count_files(paths.synthesis_out, ("*.wav",))
    sidecars = _count_files(paths.synthesis_out, ("*.json",))
    checks.append(
        DiagnosticCheck(
            "synthesis_outputs",
            "ok" if synth_wavs and sidecars >= synth_wavs else "warn",
            f"{synth_wavs} WAV(s), {sidecars} sidecar JSON(s)",
            None
            if synth_wavs and sidecars >= synth_wavs
            else "Run synthesize and verify sidecar JSON metadata.",
        )
    )

    checks.append(
        DiagnosticCheck(
            "runs_db",
            "ok" if paths.db_path.exists() else "warn",
            str(paths.db_path) if paths.db_path.exists() else "runs.db not created yet",
            None if paths.db_path.exists() else "runs.db is created after the first synthesis run.",
        )
    )

    return DiagnosticReport(workspace=Path(workspace), checks=checks)
