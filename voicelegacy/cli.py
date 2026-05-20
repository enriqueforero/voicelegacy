"""voicelegacy — Command-line interface.

Provides a minimal CLI for running the pipeline outside of a notebook:

    voicelegacy build-corpus  --workspace /path/to/ws --speaker SPEAKER_00
    voicelegacy synthesize    --workspace /path/to/ws --text "Hola mi nieto."
"""

from __future__ import annotations

from contextlib import suppress
from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table

from voicelegacy.audio import _ffmpeg_available, convert_directory_to_wav
from voicelegacy.config import (
    PipelineConfig,
    ReferenceConfig,
    SynthesisConfig,
    WorkspacePaths,
)
from voicelegacy.denoise_eval import evaluate_denoise_methods
from voicelegacy.diagnose import diagnose_workspace
from voicelegacy.logging_config import configure_logging
from voicelegacy.pipeline import (
    run_batch_synthesis,
    run_reference_phase,
)
from voicelegacy.text_inputs import resolve_text_inputs

app = typer.Typer(help="Voice cloning pipeline for family legacy — XTTS-v2 + speakerscribe.")
console = Console()


@app.callback()
def _root(verbose: bool = typer.Option(False, "--verbose", "-v", help="Verbose logging.")) -> None:
    configure_logging(level="DEBUG" if verbose else "INFO")


@app.command("build-corpus")
def build_corpus(
    workspace: Path = typer.Option(..., help="Workspace root containing speakerscribe_out/."),
    speaker: str = typer.Option("SPEAKER_00", help="Target speaker label."),
    top_n: int = typer.Option(10, help="Keep top N best-scoring segments."),
    min_dur: float = typer.Option(4.0, help="Min segment duration (s)."),
    max_dur: float = typer.Option(15.0, help="Max segment duration (s)."),
    min_snr: float = typer.Option(15.0, help="Min SNR in dB."),
    no_denoise: bool = typer.Option(False, "--no-denoise", help="Skip noise reduction."),
    force: bool = typer.Option(False, "--force", help="Rebuild even if outputs exist."),
    accept_tos: bool = typer.Option(False, "--accept-tos", help="Accept Coqui CPML license."),
) -> None:
    """Build the reference corpus from speakerscribe outputs."""
    paths = WorkspacePaths(workspace=workspace)
    config = PipelineConfig(
        reference=ReferenceConfig(
            target_speaker_label=speaker,
            top_n_segments=top_n,
            min_segment_duration_s=min_dur,
            max_segment_duration_s=max_dur,
            min_snr_db=min_snr,
            apply_denoise=not no_denoise,
        ),
        force_rebuild_reference=force,
        accept_coqui_tos=accept_tos,
    )

    result = run_reference_phase(paths, config)

    table = Table(title="Reference corpus summary")
    table.add_column("Metric")
    table.add_column("Value", justify="right")
    table.add_row("Total candidates", str(len(result.all_wavs)))
    table.add_row("Passing", str(sum(1 for r in result.reports if r.passed)))
    table.add_row("Top selected", str(len(result.top_wavs)))
    console.print(table)


@app.command("synthesize")
def synthesize(
    workspace: Path = typer.Option(...),
    text: str | None = typer.Option(None, help="Text to vocalize."),
    text_file: Path | None = typer.Option(
        None, "--text-file", help=".txt one utterance per line, or .csv with a text column."
    ),
    language: str = typer.Option("es", help="Language ISO code."),
    accept_tos: bool = typer.Option(False, "--accept-tos", help="Accept Coqui CPML license."),
    force: bool = typer.Option(False, "--force", help="Re-run even if cached."),
) -> None:
    """Synthesize one or more texts using the existing reference corpus."""
    paths = WorkspacePaths(workspace=workspace)
    top_wavs = sorted(paths.reference_corpus.glob("*.wav"))
    if not top_wavs:
        console.print("[red]No reference WAVs found. Run 'build-corpus' first.[/red]")
        raise typer.Exit(code=2)

    try:
        texts = resolve_text_inputs(text, text_file)
    except Exception as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(code=2) from exc

    config = PipelineConfig(
        synthesis=SynthesisConfig(language=language),  # type: ignore[arg-type]
        force_resynthesize=force,
        accept_coqui_tos=accept_tos,
    )

    results = run_batch_synthesis(texts, top_wavs, paths, config)
    for r in results:
        marker = "♻️ " if r.cached else "✨"
        sidecar = f"  metadata={r.metadata_path}" if r.metadata_path else ""
        # soft_wrap=True keeps file paths on a single logical line so output
        # snapshots and CLI test assertions don't break when the rendering
        # width is narrow (e.g. typer.testing.CliRunner uses 80 cols).
        console.print(f"{marker} {r.output_path}{sidecar}", soft_wrap=True)
        if r.similarity_score is not None:
            console.print(
                f"   speaker_similarity_score={r.similarity_score:.3f}",
                soft_wrap=True,
            )


@app.command("diagnose")
def diagnose(
    workspace: Path = typer.Option(..., help="Workspace root to inspect."),
    require_gpu: bool = typer.Option(
        False, "--require-gpu", help="Fail the CUDA check if no GPU is available."
    ),
    json_output: bool = typer.Option(False, "--json", help="Print machine-readable JSON."),
) -> None:
    """Diagnose local installation and workspace readiness without guessing."""
    import json

    report = diagnose_workspace(workspace, require_gpu=require_gpu)
    if json_output:
        console.print_json(json.dumps(report.to_dict(), ensure_ascii=False))
    else:
        table = Table(title=f"voicelegacy diagnose — {workspace}")
        table.add_column("Check")
        table.add_column("Status")
        table.add_column("Detail", overflow="fold")
        table.add_column("Remediation", overflow="fold")
        for check in report.checks:
            style = {"ok": "green", "warn": "yellow", "fail": "red"}[check.status]
            table.add_row(
                check.name,
                f"[{style}]{check.status.upper()}[/{style}]",
                check.detail,
                check.remediation or "",
            )
        console.print(table)
        console.print(
            f"ready={report.ready} | failures={report.failed} | warnings={report.warnings}"
        )
    if not report.ready:
        raise typer.Exit(code=2)


@app.command("convert-audio")
def convert_audio(
    workspace: Path = typer.Option(
        ..., help="Workspace root. Containers in workspace/interviews_raw/ get converted."
    ),
    overwrite: bool = typer.Option(False, "--overwrite", help="Re-encode even if .wav exists."),
    sample_rate: int = typer.Option(22050, help="Target sample rate for the output WAV."),
) -> None:
    """Convert mp4/m4a/mkv/webm/aac/mov/mp3/ogg/flac in interviews_raw/ to WAV.

    Reemplaza la Cell 17 hand-edited del notebook. Idempotent: skips files
    whose .wav twin already exists unless --overwrite is set.
    """
    paths = WorkspacePaths(workspace=workspace)
    if not paths.interviews_raw.is_dir():
        console.print(f"[red]interviews_raw/ not found at {paths.interviews_raw}[/red]")
        raise typer.Exit(code=2)

    if not _ffmpeg_available():
        console.print(
            "[red]ffmpeg is not on PATH. Install it before running this command "
            "(apt/brew/choco install ffmpeg).[/red]"
        )
        raise typer.Exit(code=2)

    produced = convert_directory_to_wav(
        paths.interviews_raw, target_sr=sample_rate, overwrite=overwrite
    )
    table = Table(title="convert-audio summary")
    table.add_column("File", overflow="fold")
    table.add_column("Size (MB)", justify="right")
    for wav in produced:
        size_mb = wav.stat().st_size / 1e6
        table.add_row(wav.name, f"{size_mb:.1f}")
    table.add_row("─" * 40, "─" * 8)
    table.add_row(f"{len(produced)} WAV(s) ready", "")
    console.print(table)


@app.command("list-speakers")
def list_speakers(
    workspace: Path = typer.Option(..., help="Workspace root containing speakerscribe_out/*.json."),
    show_files: bool = typer.Option(
        True, "--show-files/--no-show-files", help="Also list audio files in interviews_raw/."
    ),
) -> None:
    """List speakers detected by speakerscribe + segment counts and durations.

    Reemplaza la Cell 19 hand-edited del notebook. Útil para verificar el
    label que debes pasar a build-corpus --speaker antes de extraer.
    """
    import json
    from collections import defaultdict

    paths = WorkspacePaths(workspace=workspace)
    json_files = sorted(paths.speakerscribe_out.glob("*.json"))
    if not json_files:
        console.print(f"[red]No speakerscribe JSONs found in {paths.speakerscribe_out}[/red]")
        raise typer.Exit(code=2)

    for jf in json_files:
        try:
            data = json.loads(jf.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            console.print(f"[red]{jf.name}: malformed JSON ({exc})[/red]")
            continue

        source = data.get("source_audio") or data.get("audio_file") or "(unknown)"
        segments = data.get("segments") or []

        counts: dict[str, int] = defaultdict(int)
        duration: dict[str, float] = defaultdict(float)
        for s in segments:
            lbl = str(s.get("speaker", "?"))
            counts[lbl] += 1
            with suppress(KeyError, ValueError, TypeError):
                duration[lbl] += float(s["end"]) - float(s["start"])

        table = Table(title=f"{jf.name}  (source: {source})")
        table.add_column("Speaker label")
        table.add_column("Segments", justify="right")
        table.add_column("Total duration (s)", justify="right")
        for lbl in sorted(counts):
            table.add_row(lbl, str(counts[lbl]), f"{duration[lbl]:.1f}")
        console.print(table)

    if show_files and paths.interviews_raw.is_dir():
        files = sorted(paths.interviews_raw.iterdir())
        table = Table(title="interviews_raw/")
        table.add_column("File", overflow="fold")
        for f in files:
            table.add_row(f.name)
        console.print(table)


@app.command("evaluate-denoise")
def evaluate_denoise(
    workspace: Path = typer.Option(..., help="Workspace root with interviews_raw/ and reports/."),
    audio: list[Path] | None = typer.Option(
        None,
        "--audio",
        help="Specific audio file(s) to evaluate. Repeat option for multiple files.",
    ),
    include_deepfilter: bool = typer.Option(
        False,
        "--deepfilter",
        help="Also evaluate DeepFilterNet via the optional deepFilter CLI if installed.",
    ),
) -> None:
    """Compare noisereduce against optional DeepFilterNet on real samples.

    Use this with 3-5 representative files before changing production denoise
    defaults. DeepFilterNet is intentionally not enabled blindly.
    """
    paths = WorkspacePaths(workspace=workspace)
    if audio:
        files = [Path(a) for a in audio]
    else:
        files = sorted(
            p
            for p in paths.interviews_raw.iterdir()
            if p.is_file() and p.suffix.lower() in {".wav", ".mp3", ".m4a", ".flac", ".ogg"}
        )[:5]
    if not files:
        console.print("[red]No audio files found. Pass --audio or populate interviews_raw/.[/red]")
        raise typer.Exit(code=2)

    report = evaluate_denoise_methods(
        files,
        paths.reports / "denoise_eval",
        include_deepfilter=include_deepfilter,
    )
    table = Table(title="Denoise evaluation")
    table.add_column("Source", overflow="fold")
    table.add_column("Method")
    table.add_column("Status")
    table.add_column("Dynamic range", justify="right")
    table.add_column("Output", overflow="fold")
    for row in report["candidates"]:
        stats = row.get("stats") or {}
        table.add_row(
            Path(str(row["source_path"])).name,
            str(row["method"]),
            str(row["status"]),
            str(stats.get("snr_db", "")),
            str(row.get("output_path") or row.get("reason") or ""),
        )
    console.print(table)
    console.print(f"report={report['report_path']}")


if __name__ == "__main__":
    app()
