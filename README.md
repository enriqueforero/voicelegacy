# voicelegacy

`voicelegacy` is a Python pipeline for building a curated XTTS-v2 reference corpus from diarized interview audio and synthesizing new speech in the target speaker's voice.

It is designed for family-archive / legacy use cases: preserve a voice from interviews, clean and rank usable segments, generate new audio, and keep an audit trail next to every synthesized WAV.

This is not a magic voice-restoration product. If the source audio is noisy, phone-codec, clipped, badly diarized, or too short, the system should make that visible instead of pretending the clone is reliable.

## What the pipeline does

1. Reads `speakerscribe` JSON diarization output.
2. Finds the target speaker, usually `SPEAKER_00`.
3. Extracts candidate reference clips from the original interview audio.
4. Applies conservative cleanup: band-pass, optional pre-emphasis, denoise, trim, loudness normalization with headroom.
5. Rejects bad clips through quality gates: duration, source sample rate, dynamic range, clipping risk.
6. Detects pitch/F0 outliers to reduce diarization contamination.
7. Selects the top reference clips.
8. Runs XTTS-v2 synthesis with reproducible seed and optional conditioning-latent cache.
9. Writes a WAV plus a JSON sidecar containing version, config, references, source-quality metadata and optional speaker-similarity score.
10. Provides `voicelegacy diagnose` so a new user can check whether their workspace/runtime is ready.

## Hard limitations

Zero-shot XTTS-v2 is highly sensitive to reference quality. Better code cannot fully rescue bad material.

Typical failure modes:

- 8 kHz / phone-codec audio: rejected by default.
- Clipped or aggressively compressed input: may be rejected or generate unstable voice.
- Background noise, cross-talk, music, echo or multiple speakers: may reduce similarity.
- Wrong speaker label in diarization: pollutes the reference corpus.
- Too few clean reference segments: synthesis is blocked by default.
- Long texts with automatic splitting: possible voice drift between sentences.

The output must still be listened to by a human. `speaker_similarity_score` is a triage metric, not a legal/biometric guarantee.

## License and model-use warning

This repository code is MIT-licensed. XTTS-v2 model weights are not MIT-licensed. They are governed by the Coqui Public Model License (CPML): https://coqui.ai/cpml

`voicelegacy` deliberately requires explicit acceptance before loading XTTS-v2:

```python
PipelineConfig(accept_coqui_tos=True)
```

or from CLI:

```bash
voicelegacy synthesize --workspace /path/to/workspace --text "Hola" --accept-tos
```

Do not use this software to impersonate, deceive, bypass consent, or create synthetic speech for a person who has not authorized that use.

## Installation

### Local development

```bash
git clone https://github.com/EnriqueForero/voicelegacy.git
cd voicelegacy
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
python -m pip install --upgrade pip
pip install -e ".[dev]"
pre-commit install
```

Optional speaker-similarity scoring:

```bash
pip install -e ".[dev,similarity]"
```

`similarity` installs Resemblyzer. The rest of the pipeline works without it; sidecars will record similarity as `skipped`.

### Google Colab

Recommended when running XTTS-v2 because it needs practical GPU access.

```bash
!apt-get update -qq && apt-get install -y -qq ffmpeg
!pip install -e ".[dev,similarity]"
```

Then use `notebooks/notebook_voicelegacy.ipynb`, which is generated from `notebooks/build_notebook.py`. Do not hand-edit the notebook as the source of truth.

## Workspace structure

Create one workspace per person/project:

```text
workspace/
├── interviews_raw/      # raw audio/video: wav, mp3, m4a, mp4, mkv, webm, aac, mov
├── speakerscribe_out/   # diarization JSONs from speakerscribe
├── reference_corpus/    # generated clean reference clips
├── synthesis_out/       # generated WAVs + sidecar JSON files
├── reports/             # quality/F0 reports
└── runs.db              # idempotency cache
```

## First command to run

```bash
voicelegacy diagnose --workspace /path/to/workspace
```

Machine-readable output:

```bash
voicelegacy diagnose --workspace /path/to/workspace --json
```

`diagnose` checks Python, packages, FFmpeg, CUDA, CPML environment flag, model config, required workspace folders, raw media, speakerscribe JSON schema validity, reference corpus, reports, outputs and `runs.db`.

## CLI usage

### Convert raw media to WAV

```bash
voicelegacy convert-audio --workspace /path/to/workspace
```

### Inspect speaker labels

```bash
voicelegacy list-speakers --workspace /path/to/workspace
```

Use this before choosing the `--speaker` value.

### Build reference corpus

```bash
voicelegacy build-corpus \
  --workspace /path/to/workspace \
  --speaker SPEAKER_00 \
  --top-n 10 \
  --min-dur 4 \
  --max-dur 15 \
  --min-snr 15 \
  --accept-tos
```

Force rebuild without losing manual cleanup work:

```bash
voicelegacy build-corpus --workspace /path/to/workspace --force --accept-tos
```

Existing WAVs are moved to `reference_corpus_backup_YYYYMMDDTHHMMSSZ/` before rebuilding.

### Synthesize one text

```bash
voicelegacy synthesize \
  --workspace /path/to/workspace \
  --text "Hola, este es un mensaje de prueba." \
  --accept-tos
```

### Synthesize from TXT

One utterance per non-empty line:

```bash
voicelegacy synthesize \
  --workspace /path/to/workspace \
  --text-file textos.txt \
  --accept-tos
```

### Synthesize from CSV

Use a column named `text`, or a single-column CSV:

```bash
voicelegacy synthesize \
  --workspace /path/to/workspace \
  --text-file textos.csv \
  --accept-tos
```

## Python usage

```python
from pathlib import Path

from voicelegacy.config import PipelineConfig, ReferenceConfig, SynthesisConfig, WorkspacePaths
from voicelegacy.pipeline import run_reference_phase, run_synthesis

paths = WorkspacePaths(workspace=Path("/content/drive/MyDrive/voicelegacy_ws"))
config = PipelineConfig(
    accept_coqui_tos=True,
    reference=ReferenceConfig(target_speaker_label="SPEAKER_00"),
    synthesis=SynthesisConfig(language="es", seed=42),
)

corpus = run_reference_phase(paths, config)
result = run_synthesis("Hola, este es un mensaje de prueba.", corpus.top_wavs, paths, config)
print(result.output_path)
print(result.metadata_path)
```

## Interpreting `speaker_similarity_score`

If `voicelegacy[similarity]` is installed, each synthesized WAV sidecar may include a similarity report using Resemblyzer: https://github.com/resemble-ai/Resemblyzer

Informal bands:

| Score | Band | Interpretation |
|---:|---|---|
| `>= 0.85` | very_high | Strong same-speaker signal; still listen manually. |
| `0.75–0.85` | high | Usually acceptable for zero-shot legacy use. |
| `0.60–0.75` | marginal | Review carefully; possible drift/noise/contamination. |
| `< 0.60` | low | Treat as failed clone until proven otherwise. |

Low score actions:

1. Review `reports/reference_quality_*.json`.
2. Review `reports/f0_outliers_*.json`.
3. Run `voicelegacy list-speakers` to confirm target label.
4. Collect cleaner recordings or adjust diarization.
5. Consider denoising/fine-tuning only after source quality is understood.

## Quality metadata in sidecars

Each synthesized WAV has a sibling `.json` sidecar with:

- `voicelegacy_version`
- `run_hash`
- `reference_set_hash`
- `reference_wavs`
- `synthesis_config`
- `reference_config`
- `source_quality`
- `similarity`
- `speaker_similarity_score`

If `source_quality.degraded_mode` is `true`, do not treat the WAV as production-quality without manual review.

## Tests, lint and format

```bash
python -m ruff format --check .
python -m ruff check .
python -m pytest -q
python notebooks/build_notebook.py
```

The release-candidate bar is:

- ruff format passes.
- ruff lint passes.
- pytest passes.
- coverage floor: 75%.
- notebook regenerates and validates.
- `pyproject.toml` parses.

## GitHub Actions

CI runs on pushes and pull requests:

1. Install package with dev dependencies.
2. Validate `pyproject.toml`.
3. Regenerate/validate notebook.
4. Run `ruff format --check`.
5. Run `ruff check`.
6. Run `pytest` with coverage.

The CI intentionally does not download XTTS-v2 weights or run GPU inference. XTTS calls are mocked in tests.

## Troubleshooting

### `No reference WAVs found`

Run:

```bash
voicelegacy build-corpus --workspace /path/to/workspace --accept-tos
```

Then check `reference_corpus/` and `reports/reference_quality_*.json`.

### `Only 1 reference segment provided`

The pipeline requires at least three usable reference segments by default. Add more source material or improve diarization/cleanup.

### `COQUI_TOS_AGREED is not set`

Read CPML first: https://coqui.ai/cpml

Then pass `--accept-tos` or set `accept_coqui_tos=True`.

### `ffmpeg not on PATH`

Install FFmpeg:

```bash
sudo apt-get install ffmpeg       # Debian/Ubuntu/Colab
brew install ffmpeg               # macOS
choco install ffmpeg              # Windows Chocolatey
```

### Similarity is skipped

Install the optional dependency:

```bash
pip install -e ".[similarity]"
```

### Similarity is low

Do not tune blindly. First inspect whether the reference set is contaminated, noisy, clipped, too short or diarized under the wrong speaker label.

## Release status

Current status: release candidate for GitHub testing. It is suitable for technical users who understand CPML, consent requirements and zero-shot cloning limitations. It is not yet a packaged PyPI release.

## P3 product hardening

### Denoise evaluation before changing defaults

DeepFilterNet is not enabled by default. It is an optional evaluation path because aggressive enhancement can improve perceived noise while damaging speaker identity.

Install optional support:

```bash
pip install -e ".[deepfilter]"
```

Run the evaluation harness on real samples:

```bash
voicelegacy evaluate-denoise \
  --workspace /path/to/workspace \
  --audio clean.wav \
  --audio moderate_noise.wav \
  --audio heavy_noise.wav \
  --audio phone_codec.wav \
  --audio long_interview_excerpt.wav \
  --deepfilter
```

The command writes enhanced candidates and `reports/denoise_eval/denoise_evaluation_*.json`. DeepFilterNet should become a production default only if it improves human listening quality and downstream `speaker_similarity_score` without adding artifacts.

### Long text strategy

`SynthesisConfig` exposes an explicit long-text policy:

```python
SynthesisConfig(
    long_text_strategy="auto",      # "auto", "single_pass", or "coqui_split"
    max_single_pass_chars=240,
    long_text_warning_chars=600,
)
```

The default `auto` policy sends short text as one XTTS pass to reduce sentence-to-sentence drift, and delegates splitting to XTTS only for longer prose. Every output sidecar includes `text_plan` so this decision is auditable.

### Ethics and release documentation

See:

- `docs/ETHICS.md`
- `docs/P3_EVALUATION.md`
- `docs/RELEASE.md`

The short version: do not clone a person's voice without consent, do not use outputs to deceive or impersonate, and do not separate generated WAVs from their JSON sidecars.

## Packaging and PyPI

The release workflow builds artifacts on tags and creates a draft GitHub release. It does **not** publish to PyPI automatically. PyPI publication requires manual workflow dispatch with `publish_pypi=true` and a configured PyPI Trusted Publisher.

Relevant official references:

- Python packaging guide: https://packaging.python.org/tutorials/packaging-projects/
- PyPI Trusted Publishing: https://docs.pypi.org/trusted-publishers/
- DeepFilterNet: https://github.com/Rikorose/DeepFilterNet
- Coqui Public Model License: https://coqui.ai/cpml
