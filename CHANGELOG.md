# Changelog

All notable changes to this project will be documented in this file.
Format: [Keep a Changelog](https://keepachangelog.com/en/1.0.0/)
Versioning: [Semantic Versioning](https://semver.org/)

## [Unreleased]

- P3-27 (tabla SNR fuente → similarity esperada): experimento empírico con audio real pendiente. Bloqueante para v1.0; no bloqueante para v0.3.0 que aporta valor de ingeniería independiente.
- P3-31 (Polars/DuckDB): no aplica al volumen actual (1-3 entrevistas).

## [0.3.0] — 2026-05-19 — Turno 10 · fine-tuning XTTS-v2

Salto MINOR (0.2.0 → 0.3.0) por feature pública nueva backward-compatible: módulo `voicelegacy.finetuned_inference` para cargar y usar checkpoints XTTS-v2 fine-tuneados. El paquete sigue funcionando idénticamente sin él — la nueva ruta es opt-in.

### Added

- **Módulo `voicelegacy.finetuned_inference`** (411 LOC, 88.7% cobertura). Tres entradas públicas:
  - `FineTunedCheckpoint.from_dir(path)` — valida los 6 archivos requeridos del checkpoint (`model.pth`, `config.json`, `vocab.json`, `dvae.pth`, `mel_stats.pth`, `speakers_xtts.pth`), produce un handle inmutable con `fingerprint` (16 hex chars) para audit trail.
  - `load_finetuned_model(checkpoint, device="auto")` — carga el modelo con `Xtts.init_from_config + load_checkpoint`, no usa la API alta `TTS.api.TTS` porque el checkpoint es local (no hub). Cacheado por `(checkpoint_dir, device)`.
  - `synthesize_with_finetuned(model, checkpoint, text, speaker_wav, output_path, config)` — drop-in del `synthesize_to_file` de `synthesis.py`. Conditioning latents cacheados por `(fingerprint, reference_set)` para evitar mezcla cruzada entre checkpoints.
- **Notebook `notebooks/notebook_voicelegacy_finetune.ipynb`** (27 celdas, regenerable desde `build_finetune_notebook.py`) — flujo completo Colab Free T4: instalación, descarga base XTTS-v2 (cacheada en Drive), preparación dataset LJSpeech-like desde `reference_corpus/` + transcripciones de speakerscribe, configuración del GPTTrainer (sólo GPT, vocoder congelado), entrenamiento con checkpoints intermedios (resumible si Colab corta sesión), materialización de checkpoint reutilizable, validación end-to-end, **comparación A/B base vs fine-tuned con `speaker_similarity_score`**.
- **38 tests nuevos** en `tests/test_finetuned_inference.py` (220 total ahora, era 182). Cubren: validación de los 6 archivos requeridos (paramétrico), fingerprint estable y discriminante, cache hit/miss por device, ImportError graceful cuando coqui-tts no está, fallo de `load_checkpoint` envuelto en `RuntimeError` con mensaje accionable, aislamiento de latents cache entre checkpoints distintos sobre las mismas referencias, contrato drop-in con `synthesis.py`.

### Changed

- **`__version__`** sincronizado a `"0.3.0"` en `voicelegacy/__init__.py` y `pyproject.toml`.
- **`voicelegacy/__init__.py`** exporta los 4 símbolos nuevos: `FineTunedCheckpoint`, `load_finetuned_model`, `release_finetuned_model`, `synthesize_with_finetuned`. `__all__` ordenado alfabéticamente como antes.

### Coverage by module

| Módulo | 0.2.0 | 0.3.0 | Cambio |
|---|---|---|---|
| `finetuned_inference.py` (nuevo) | — | **88.7%** | nuevo |
| **Total** | 85.58% | **85.89%** | +0.31 pp |
| **Tests** | 182 | **220** | +38 |

### Verified

- `python -c "import tomllib; tomllib.load(open('pyproject.toml','rb'))"` → OK
- `python -c "import voicelegacy; print(voicelegacy.__version__)"` → `0.3.0`
- `pyproject.toml::version == __init__.py::__version__` → `0.3.0`
- `ruff check --config pyproject.toml voicelegacy/ tests/ scripts/` → All checks passed
- `ruff format --check --config pyproject.toml voicelegacy/ tests/ scripts/` → 38 files already formatted
- `pytest tests/` → **220 passed in ~10s**
- `pytest tests/ --cov=voicelegacy --cov-fail-under=80` → **85.89%**, piso 80% alcanzado
- `python notebooks/build_notebook.py` → 37 celdas (inferencia)
- `python notebooks/build_finetune_notebook.py` → 27 celdas (fine-tuning)
- `python -m build` → produce `voicelegacy-0.3.0.tar.gz` + `voicelegacy-0.3.0-py3-none-any.whl` sin warnings
- `twine check dist/*` → PASSED en ambos artefactos
- Smoke install desde wheel en venv aislado: `voicelegacy --help` muestra 6 subcomandos, `voicelegacy.__version__` retorna `"0.3.0"`, `FineTunedCheckpoint` importable.

### Not in this release (documented limitations)

- **Inferencia fine-tuned no expuesta vía CLI** todavía. Por diseño: el checkpoint vive en Drive, no en `interviews_raw/`, y el flujo natural es notebook. Si la demanda aparece, agregar `voicelegacy synthesize --finetuned-dir PATH` en 0.4.0.
- **Sin pipeline `run_synthesis_finetuned`** equivalente al `run_synthesis` de `pipeline.py`. El notebook llama directo a `synthesize_with_finetuned`. Razón: el sidecar/runs.db actual no contempla el campo `checkpoint_fingerprint`; añadirlo es scope de 0.4.0.
- **P3-27 (audio real benchmark + decisión zero-shot vs fine-tuned)**: sigue pendiente. v0.3.0 da la herramienta para fine-tunear; falta correrla con material real y comparar A/B.
- **Sin test E2E con coqui-tts real**: imposible en CI sin GPU y sin 2 GB de pesos descargados. Mitigación: el smoke del notebook (celda 9) lo verifica en cada uso.

## [0.2.0] — 2026-05-19 — Turno 9 · ingeniería impecable para PyPI

Salto MINOR (0.1.6 → 0.2.0) por acumulación de cambios significativos desde 0.1.0, incluyendo un breaking change documentado en 0.1.x (P2-24: speakerscribe JSON malformado ahora falla en vez de saltarse segmentos silenciosamente). SemVer 0.x.y permite cualquier cambio en patch, pero documentamos explícitamente para usuarios pre-existentes.

### Added

- **Tests de `similarity.py` ampliados de 4 a 32 casos** (incluye `quality_band` en sus cuatro umbrales, encoder cache hit, `release_encoder`, `compute_similarity_batch` con error parcial recuperable, validación de paths inexistentes, `is_available` con ambas ramas).
- **Tests de `corpus.py` ampliados de 9 a 29 casos** (incluye end-to-end de `extract_segments_to_wav` con audio sintético, `estimate_median_f0_hz` con voz/silencio/duración corta, `analyze_f0_outliers` con detección real de outlier, `filter_f0_outliers`, `build_reference_corpus` completo, branch de búsqueda de extensión alternativa en `load_speakerscribe_json`).
- **`pyproject.toml` con metadatos completos para PyPI**: `Development Status :: 4 - Beta` (subido desde Alpha), `Environment :: GPU :: NVIDIA CUDA`, classifiers ampliados (Conversion, Libraries :: Python Modules, Spanish/English natural language), `license-files = ["LICENSE"]` (PEP 639), `maintainers`, `Issues` URL, keyword extras (`text-to-speech`, `speaker-similarity`, `xtts-v2`, `coqui-tts`, `speakerscribe`, `audio-processing`, `denoising`, `diarization`, `spanish`).
- **Meta-extra `all`** en optional-dependencies: `pip install voicelegacy[all]` instala similarity + deepfilter + notebook + nbformat/ipykernel en un solo comando.
- **Extra `notebook`** para usuarios que quieren `nbformat + ipykernel` sin pip-installing herramientas dev.
- **`PUBLISHING_CHECKLIST.md`**: gate ordenado para tag GitHub y publicación PyPI, con paso explícito de `twine check`, TestPyPI primero, validación de paridad `pyproject.toml::version == __init__.py::__version__`.

### Changed

- **`__version__`** sincronizado a `"0.2.0"` en `voicelegacy/__init__.py` (antes desincronizado: `pyproject.toml=0.2.0` vs `__init__.py=0.1.6`; el bug habría hecho `pip show` y `voicelegacy.__version__` reportar versiones diferentes).
- **`--cov-fail-under`** subido de 75 a 80. Margen restante: 5.58 puntos sobre el piso (85.58% actual).
- **PROGRESS.md tracking honesto**:
  - P1-WADA reclasificado de "✅ cerrado por decisión técnica" a "🛑 Deferred — ver P3-5". Cerrar por no-hacer es contabilidad mañosa; deferred es la categoría correcta.
  - P3 con tabla de cross-reference original (P3-27..P3-31) ↔ renombrado (P3-1..P3-4) para no romper trazabilidad con el plan A01.
  - P3-27 marcado como **prioridad alta — bloqueante para v1.0** (no para v0.2.0).
  - Resumen ejecutivo corregido: P1 8/8 (no 9/9), P3 3 de 5 originales + 1 extra (no 4/5).

### Coverage by module

| Módulo | 0.1.6 | 0.2.0 | Cambio |
|---|---|---|---|
| `similarity.py` | 53.4% | **97.5%** | +44.1 pp |
| `corpus.py` | 42.9% | **95.5%** | +52.6 pp |
| **Total** | 76.21% | **85.58%** | +9.37 pp |
| **Tests** | 134 | **182** | +48 |

### Verified

- `python -c "import tomllib; tomllib.load(open('pyproject.toml','rb'))"` → OK
- `python -c "import voicelegacy; print(voicelegacy.__version__)"` → `0.2.0`
- `grep -E '^version = ' pyproject.toml` y `__version__` sincronizados
- `ruff check --config pyproject.toml voicelegacy/ tests/ scripts/` → All checks passed
- `ruff format --check --config pyproject.toml voicelegacy/ tests/ scripts/` → 36 files already formatted
- `pytest tests/` → **182 passed in ~51s**
- `pytest tests/ --cov=voicelegacy --cov-fail-under=80` → **85.58%**, piso 80% alcanzado
- `python notebooks/build_notebook.py` → 37 celdas, regenerable e idempotente
- `python -m build` → genera `voicelegacy-0.2.0.tar.gz` + `voicelegacy-0.2.0-py3-none-any.whl` sin warnings
- `twine check dist/*` → PASSED en ambos artefactos

### Not in this release (documented limitations)

- **P3-27 (audio real benchmark)**: el experimento que valida si el `speaker_similarity_score` con material real sub-óptimo cae en banda `high` (≥0.75) NO está hecho. v0.2.0 es "ingeniería impecable, validación empírica pendiente". v1.0 requiere ese experimento + bench documentado.
- **Tests con coqui-tts real**: todos los tests usan mocks. Un breaking change en `coqui-tts<1.0` upstream se detectaría en producción, no en CI. Mitigación: `release_conditioning_latents()` + ruta de fallback `tts_to_file` ya implementadas.

## [0.1.6] — 2026-05-18 — Turno 8 · pre-commit hardening

### Fixed

- **Critical**: `.pre-commit-config.yaml` did not parse because three local hooks used `entry: python -c "..."` with inline commands containing `:`, which YAML interpreted as mapping-value separators. The file aborted with `InvalidConfigError` before any hook ran, silently disabling all custom protections (TOML validity, notebook schema, no-live-`runtime.unassign()`). Anyone installing the hooks lost guard against the exact bugs P0-1, P0-2, and notebook-generator divergence.
- The same hook block declared `additional_dependencies: [nbformat]` with `language: system`, a combination pre-commit refuses. The schema validator never installed `nbformat`.
- `notebooks/build_notebook.py` wrote the `.ipynb` without a trailing newline, so `end-of-file-fixer` modified the file after every regeneration, creating a loop where every `python build_notebook.py` made the working tree dirty.

### Added

- `scripts/check_pyproject_toml.py`, `scripts/check_notebook_schema.py`, `scripts/check_no_runtime_unassign.py`: the three local hooks extracted as standalone, testable Python files with docstrings. No more YAML-escaped one-liners.

### Changed

- Bumped `pre-commit/pre-commit-hooks` from `v4.5.0` to `v5.0.0` (the previous version emitted deprecation warnings about stage names).
- `language: system` → `language: python` for the notebook-schema hook so `additional_dependencies` actually installs `nbformat`.
- `files: ^notebooks/.*\.ipynb$` → `files: ^notebooks/[^/]+\.ipynb$` for the notebook hooks, so `notebooks/_archive/` (preserved historical bugs) does not block commits.
- `build_notebook.py` now writes with a trailing newline, keeping `end-of-file-fixer` idempotent across regenerations.

### Verified

- `pre-commit run --all-files` on a freshly initialized repo, no cache: 10 hooks passed (ruff, ruff-format, trailing-whitespace, end-of-file-fixer, check-yaml, check-toml, check-added-large-files, validate-pyproject-toml, validate-notebook-schema, no-live-runtime-unassign).
- `scripts/check_no_runtime_unassign.py notebooks/_archive/notebook_voicelegacy_*_handedited_37cells.ipynb` correctly fails (positive control: the archived broken notebook is still detected, the hook is real).
- 134 tests passing, coverage 76.31%, ruff clean, TOML valid, notebook regenerable.

## [0.1.5] — 2026-05-18 — Turno 7 · cierre P2-20 + notebook sync

### Added

- `evaluate_file(path, config=ReferenceConfig)` accepts a `ReferenceConfig` so the gating thresholds (`min_segment_duration_s`, `max_segment_duration_s`, `min_snr_db`) have one source of truth. Explicit keyword overrides still take precedence to keep ad-hoc scripts working.
- Tests in `tests/test_quality.py` covering the new `config=` signature and the override precedence rule.
- Notebook now surfaces six features that previously lived only in the code:
  - `voicelegacy diagnose` as the first sanity check after install/import.
  - Sidecar JSON inspection (`speaker_similarity_score`, `quality_band`, `degraded_mode`, `text_plan`, `run_hash`).
  - Optional `evaluate-denoise` cell for comparing `noisereduce` vs DeepFilterNet on real samples.
  - Tunable variables in the config cell for `DENOISE_STATIONARY`, `APPLY_BANDPASS_FILTER`, `APPLY_PREEMPHASIS_FILTER`, `ENABLE_F0_OUTLIER_FILTER`.
  - `LONG_TEXT_STRATEGY`, `MAX_SINGLE_PASS_CHARS`, `LONG_TEXT_WARNING_CHARS`.
  - Troubleshooting table expanded with the matching CLI commands.

### Changed

- `quality.score_segment()` default `min_snr_db` now points to the package constant `MIN_SNR_DB` instead of the literal `15.0` (eliminates the three-source-of-truth bug — closes P2-20 from the original audit §3.4).
- `pipeline.py` uses the simplified `evaluate_file(wav, config=config.reference)` signature in both `run_reference_phase` and `_reference_quality_summary`.
- CLI `synthesize` output uses `console.print(..., soft_wrap=True)` to avoid Rich breaking file paths at the 80-column boundary that `typer.testing.CliRunner` defaults to. The previous behavior made `tests/test_cli.py::test_synthesize_uses_text_file_and_prints_sidecar` fail intermittently in CI on narrow terminals.
- CLI `synthesize` now also prints `speaker_similarity_score` per result when available.

### Fixed

- Notebook cell 6 (`PipelineConfig` assembly) previously referenced `DENOISE_STATIONARY`, `APPLY_BANDPASS_FILTER`, `APPLY_PREEMPHASIS_FILTER` without those names being defined in the user-edited config cell. Any actual notebook run crashed with `NameError`. Variables are now defined in the config cell and consistently propagated.

### Verified

- `python -m ruff format --check .` → 33 files already formatted.
- `python -m ruff check .` → All checks passed!
- `python -m pytest -q` → **134 tests**, **76.31%** coverage, 75% floor reached.
- `python notebooks/build_notebook.py` → validates against nbformat v4.5, writes **37 cells**.
- `pyproject.toml` parses with `tomllib`.

## [0.1.4] — 2026-05-18 — GitHub release candidate

### Added

- GitHub Actions CI gate for install, TOML validation, notebook validation, Ruff format, Ruff lint and pytest coverage.
- Additional mocked XTTS-v2 tests covering CPML acceptance, missing `coqui-tts`, model cache, lower-level XTTS API, conditioning-latent cache and fallback paths.
- CLI execution tests for `build-corpus`, `synthesize --text-file`, and `diagnose --json` with mocked pipeline functions.
- Pipeline tests for optional similarity success/failure branches and batch synthesis preload behavior.

### Changed

- Coverage floor raised from 65% to **75%**.
- README rewritten as an operational guide for local use, Colab use, workspace structure, CLI commands, sidecar interpretation, troubleshooting and zero-shot limits.
- CI explicitly avoids real XTTS weight downloads / GPU inference; those paths remain mocked.

### Verified

- `python -m ruff format --check .` passes.
- `python -m ruff check .` passes.
- `python -m pytest -q` passes: **125 tests**, **77.61%** coverage, 75% floor reached.
- `python notebooks/build_notebook.py` validates and writes the notebook.
- `pyproject.toml` parses with `tomllib`.

## [0.1.3] — 2026-05-18 — P2 core robustness

### Added

- `voicelegacy diagnose --workspace ...` operational readiness command with human and JSON output.
- `voicelegacy synthesize --text-file` for `.txt` and `.csv` batch utterances.
- Pydantic speakerscribe schema validation (`SpeakerscribeDocument`, `SpeakerscribeSegment`).
- Runtime telemetry helpers for elapsed time and CUDA/VRAM snapshots.

### Changed

- `compute_run_hash()` now includes the installed `voicelegacy` version so cache entries are invalidated after algorithm/package upgrades.
- Synthesis sidecar metadata now records `voicelegacy_version`.
- Malformed speakerscribe segments now fail the document explicitly instead of being skipped silently.

### Fixed

- `force_rebuild_reference=True` now backs up existing reference WAVs to `reference_corpus_backup_<UTC>/` instead of deleting them.

## [0.1.2] — 2026-05-18 — P1 quality pass

### Added

- Reproducible synthesis via `SynthesisConfig.seed` and deterministic seeding in `synthesis.py`.
- Optional Resemblyzer speaker-similarity scoring for synthetic outputs.
- Synthesis sidecars with run hash, seed, reference set, source-quality summary, degraded-mode flag and optional speaker-similarity score.
- Adaptive cleanup controls: non-stationary denoise, conservative band-pass filtering and optional pre-emphasis.
- F0 outlier guard for target-speaker contamination detection.
- XTTS conditioning-latent cache with fallback to `tts_to_file()`.
- Grouped quality reasons in reference reports.

### Changed

- `ReferenceConfig.target_loudness_lufs` upper bound tightened to `-16` LUFS.
- `SynthesisConfig.temperature` upper bound tightened to `0.9`.
- The internal SNR heuristic is documented as dynamic-range estimation; `_estimate_snr_db` remains as a backward-compatible alias.
- `load_audio_mono` prefers `soundfile` + `scipy.signal.resample_poly` for WAV-like formats.

### Fixed

- `trim_silence` no longer depends on `librosa.effects.trim`, avoiding slow numba initialization.
- `audio.denoise()` now honors the non-stationary denoise configuration instead of forcing `stationary=True`.

## [0.1.0] — 2026-05-16

### Added

- Pydantic configuration models.
- Audio preprocessing utilities.
- Reference-segment scoring and quality gates.
- Corpus builder for speakerscribe JSON outputs.
- XTTS-v2 wrapper with explicit CPML handling.
- SQLite idempotency cache.
- Two-phase pipeline orchestration.
- Typer CLI.
- Generated Colab notebook.
- Initial test suite.

### Notes

- XTTS-v2 model weights are governed by CPML: https://coqui.ai/cpml
- The project consumes speakerscribe outputs; it does not perform diarization itself.

## 0.1.0-rc.P3 — Turno 6

### Added

- P3 denoise evaluation harness: `voicelegacy evaluate-denoise` compares the current noisereduce path against optional DeepFilterNet on real user-selected samples.
- Optional extras: `deepfilter` and `publish`.
- Explicit long-text policy via `SynthesisConfig.long_text_strategy`, `max_single_pass_chars`, and `long_text_warning_chars`.
- `text_plan` metadata in synthesis sidecars.
- Documentation: `docs/P3_EVALUATION.md`, `docs/ETHICS.md`, and `docs/RELEASE.md`.

### Changed

- XTTS sentence splitting is no longer a blind boolean for all text. The default `auto` policy avoids splitting short utterances and enables splitting for longer prose.
- Release workflow now builds artifacts and draft GitHub releases on tags, but PyPI publishing requires manual workflow dispatch and Trusted Publisher setup.

### Not changed deliberately

- DeepFilterNet is not a production default. It must win on real audio and downstream similarity before replacing noisereduce.
