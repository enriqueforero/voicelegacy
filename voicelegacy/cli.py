"""voicelegacy — Command-line interface.

Provides a minimal CLI for running the pipeline outside of a notebook:

    voicelegacy build-corpus  --workspace /path/to/ws --speaker SPEAKER_00
    voicelegacy synthesize    --workspace /path/to/ws --text "Hola mi nieto."
"""

from __future__ import annotations

from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table

from voicelegacy.config import (
    PipelineConfig,
    ReferenceConfig,
    SynthesisConfig,
    WorkspacePaths,
)
from voicelegacy.logging_config import configure_logging
from voicelegacy.pipeline import (
    run_batch_synthesis,
    run_reference_phase,
)

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
    text: str = typer.Option(..., help="Text to vocalize."),
    language: str = typer.Option("es", help="Language ISO code."),
    accept_tos: bool = typer.Option(False, "--accept-tos", help="Accept Coqui CPML license."),
    force: bool = typer.Option(False, "--force", help="Re-run even if cached."),
) -> None:
    """Synthesize a single text using the existing reference corpus."""
    paths = WorkspacePaths(workspace=workspace)
    top_wavs = sorted(paths.reference_corpus.glob("*.wav"))
    if not top_wavs:
        console.print("[red]No reference WAVs found. Run 'build-corpus' first.[/red]")
        raise typer.Exit(code=2)

    config = PipelineConfig(
        synthesis=SynthesisConfig(language=language),  # type: ignore[arg-type]
        force_resynthesize=force,
        accept_coqui_tos=accept_tos,
    )

    results = run_batch_synthesis([text], top_wavs, paths, config)
    for r in results:
        marker = "♻️ " if r.cached else "✨"
        console.print(f"{marker} {r.output_path}")


if __name__ == "__main__":
    app()
