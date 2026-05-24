# Changelog

All notable changes to this project will be documented in this file.
Format: [Keep a Changelog](https://keepachangelog.com/en/1.0.0/)
Versioning: [Semantic Versioning](https://semver.org/)

## [Unreleased]

- P3-27 (tabla SNR fuente → similarity esperada): experimento empírico con audio real pendiente. Bloqueante para v1.0; no bloqueante para v0.3.0 que aporta valor de ingeniería independiente.
- P3-31 (Polars/DuckDB): no aplica al volumen actual (1-3 entrevistas).

## [0.4.0] — 2026-05-24 — Turno 14 · correcciones de auditoría (bug crítico, coherencia, defaults)

Salto MINOR (0.3.3 → 0.4.0): API pública nueva + cambios de comportamiento en defaults de limpieza. Atiende los hallazgos críticos de la auditoría externa de v0.3.3.

### Fixed

- **Bug crítico criterio 6 (WAV↔texto roto).** El notebook `notebook_voicelegacy_finetune.ipynb` emparejaba WAVs con transcripciones usando el patrón `{source_stem}_seg{idx:04d}`, que NO coincide con el patrón real que escribe `extract_segments_to_wav` (`{stem}_{idx:04d}_{start:08.2f}`). Resultado: `text_index.get(wav.stem)` devolvía `None` siempre → dataset vacío → `RuntimeError`. **Solución de raíz:** `extract_segments_to_wav` ahora escribe un sidecar `.txt` con la transcripción junto a cada WAV, y la nueva función `build_finetune_dataset` empareja leyendo el sidecar adyacente (no parsea nombres). Reproducido y verificado con un test de integración.

### Added

- **Módulo `voicelegacy.finetune_dataset`** (109 LOC, 96.2% cobertura):
  - `build_finetune_dataset(reference_corpus, dataset_dir, ...)` → arma el dataset LJSpeech-like leyendo los sidecars `.txt`. Robusto por construcción. Devuelve `DatasetBuildResult` con conteos y diagnóstico de descartes.
  - `validate_corpus_coherence(reference_corpus, threshold=0.70, ...)` → **criterio 19**: embebe cada WAV con Resemblyzer y marca clips cuya similitud al centroide cae bajo el umbral. Detecta contaminación silenciosa cuando una entrevista se etiquetó al hablante equivocado en el flujo manual del bridge. Devuelve `CoherenceResult` con outliers y veredicto.
- **Sidecar de transcripción** (`.txt`) escrito junto a cada WAV por `extract_segments_to_wav`.
- **Celda 6.bis en el notebook bridge**: valida coherencia del corpus con Resemblyzer ANTES del dataset, con guía de qué hacer si hay outliers. Escribe `reports/coherence_report.json`.
- **20 tests nuevos** (`test_finetune_dataset.py`), incluido el **test de integración corpus→dataset** (criterio 9) que reproduce y previene la regresión del criterio 6: extrae un corpus real con `extract_segments_to_wav`, arma el dataset, y afirma que NO está vacío + que el interlocutor fue excluido. Total: 238 tests (era 220).

### Changed (cambios de comportamiento en defaults — criterios 1 y 3)

- **`bandpass high_hz` 7600 → 10000 Hz**: 7600 cortaba armónicos altos clave para el timbre; XTTS-v2 emite a 24 kHz.
- **Orden de limpieza en `extract_segments_to_wav`**: ahora `denoise → bandpass → preemphasis` (antes `bandpass → preemphasis → denoise`). noisereduce estima el perfil de ruido del espectro completo; filtrar antes lo privaba de información.
- **`target_loudness_lufs` -23 → -20**: -23 (broadcast EBU) está por debajo del rango de entrenamiento de XTTS-v2 (~-23 a -18); -20 queda dentro.
- **`min_segment_duration_s` 4.0 → 6.0**: consistencia con `MIN_REF_DURATION_S=6.0` y el mínimo recomendado de XTTS-v2 para contexto prosódico.
- **`top_n_segments` 10 → 5**: pocas referencias excelentes superan a muchas mediocres.
- **`notebook_voicelegacy_finetune.ipynb` y bridge** ahora usan `build_finetune_dataset` (función del paquete) en vez de emparejamiento frágil por nombre.

### CI (criterio 11)

- El workflow ahora regenera y valida los **4 notebooks** (antes solo el principal): loop sobre todos los `build_*.py` + schema + hook anti-bomba sobre cada `.ipynb`.

### Verified

- Bug criterio 6 reproducido (patrón viejo nunca coincide) y corregido (test de integración pasa, dataset no vacío, interlocutor excluido).
- `validate_corpus_coherence` detecta contaminación: test con 3 clips de un hablante + 1 ortogonal → marca el outlier, `is_coherent=False`.
- 238 tests verdes, cobertura 86.51% (subió desde 85.90%), `finetune_dataset.py` 96.2%.
- ruff check + format limpios; los 4 notebooks: 0 errores de sintaxis (IPython transformer), schema + anti-bomba OK.
- `python -m build` + `twine check` → PASSED para 0.4.0.

### Not in this release (límites honestos que siguen)

- **Criterio 10 (validación empírica con audio real):** sigue pendiente. Los tests usan audio sintético; nadie ha medido aún `speaker_similarity_score` baseline vs cleanup vs finetune con material real. Es trabajo de Colab con GPU + material del usuario, no de CI.
- Criterios 2 (presets de hiperparámetros legacy vs expresivo) y 4 (renombrar `snr_db`→`dynamic_range_db` en API pública) quedan como mejoras futuras documentadas.

## [0.3.3] — 2026-05-23 — Turno 13 · notebooks conformes a las skills de Colab

Salto PATCH (0.3.2 → 0.3.3): reescritura del notebook puente y mejoras de gestión de memoria en el standalone para cumplir las skills `colab-notebook-dev` y `python-data-library-dev`. Sin cambios en la API del paquete.

### Changed

- **`notebook_voicelegacy_bridge.ipynb` reescrito** siguiendo las skills:
  - **Estructura EXTRAS/EJECUTAR** (skill §4): una celda EXTRAS con imports + `@dataclass BridgeConfig` + todas las funciones; celdas EJECUTAR de pocas líneas con solo variables de usuario + una llamada.
  - **Disciplina de RAM (skill §8, lo más crítico):** el notebook anterior cargaba la entrevista COMPLETA a RAM (`sf.read(str(audio_path))`) solo para reproducir 12 s — con 10+ horas de entrevistas esto reventaba los 12 GB de Colab Free. Ahora usa **lectura parcial desde disco** (`leer_fragmento` con `soundfile` seek start/stop): una entrevista de 1 hora cuesta la misma RAM que una de 12 s. Verificado: lee solo el 18% del archivo para una muestra.
  - **Checkpointing por entrevista (skill §6):** `bridge_manifest.json` registra qué entrevistas ya se extrajeron; re-correr salta las hechas. Sobrevive a sesión Colab cortada. Verificado idempotente.
  - **Liberación de RAM entre entrevistas:** `gc.collect()` tras cada una; monitoreo con `psutil` y aviso al 85%.
  - **Config centralizada `@dataclass` (skill §1):** `BridgeConfig` con `__post_init__` validation, cero magic numbers (umbrales de minutos, SNR, preview, RAM warn todos en la config).
  - **Observabilidad (skill §5):** progreso `[i/total]`, resumen de composición por entrevista, monitoreo de RAM.
  - **Metadata de reproducibilidad (skill §12):** `bridge_metadata.json` con config + stats + python_version.
- **`notebook_voicelegacy_finetune_standalone.ipynb`:** añadida liberación explícita del audio completo (`del audio_full; gc.collect()`) tras la segmentación, progreso `🔄 segmento i/total`, y comentario que justifica por qué para UN archivo con segmentación densa cargar una vez es defendible (vs el bridge que lee dispersamente de muchos archivos largos y por eso usa lectura parcial).

### Verified

- 4 tests end-to-end del flujo del bridge ejecutados con datos sintéticos: (1) lectura parcial lee solo el fragmento pedido (18% del archivo, no 100%); (2) extracción filtra al interlocutor (`Kept N/M segments for SPEAKER_00`); (3) checkpointing idempotente (re-correr salta las hechas); (4) metadata de reproducibilidad guardada.
- La celda EXTRAS ejecuta como módulo real; `BridgeConfig.__post_init__` rechaza valores inválidos.
- 220 tests verdes, cobertura 85.90%, ruff check + format limpios, los 4 notebooks pasan schema + anti-bomba.
- `python -m build` + `twine check` → PASSED para 0.3.3.

### Documented

- `docs/PROCESO_COMPLETO_ENTREVISTAS.md` actualizado con la sección de gestión de memoria (por qué no se carga el audio completo, checkpointing, reanudación).

## [0.3.2] — 2026-05-23 — Turno 12 · notebook puente speakerscribe → voicelegacy

Salto PATCH (0.3.1 → 0.3.2): nuevo notebook de enlace para el caso real de muchas entrevistas con varios hablantes. Sin cambios en la API del paquete.

### Added

- **Notebook `notebooks/notebook_voicelegacy_bridge.ipynb`** (19 celdas, regenerable desde `build_bridge_notebook.py`). Conecta la salida de speakerscribe (entrevistas diarizadas con varias personas) con voicelegacy (corpus de un solo hablante). Resuelve el problema de que las etiquetas `SPEAKER_xx` de speakerscribe NO son consistentes entre archivos (la diarización es por-archivo): incluye una celda de identificación asistida por audio (reproduce una muestra de cada hablante por entrevista) y un mapa `TARGET_SPEAKER_MAP` por entrevista. Recorta + limpia solo los segmentos del hablante objetivo de todas las entrevistas, consolida en un `reference_corpus/` único, y opcionalmente construye el dataset LJSpeech-like para fine-tuning emparejando cada WAV con su transcripción.

### Verified

- Compatibilidad speakerscribe → voicelegacy confirmada ejecutando código real: el schema Pydantic de voicelegacy (`extra="allow"`) parsea el JSON de speakerscribe con todas sus claves (`audio_file`, `language_detected`, `segments[].start/end/text/speaker`, más claves extra `id`, `speaker_overlap_s`, `words`).
- Cadena de extracción probada end-to-end con JSON + WAV sintéticos: `filter_segments(target_speaker="SPEAKER_00")` descartó al interlocutor (`SPEAKER_01`) y extrajo solo los segmentos del objetivo. Log: `Kept 3/4 segments for speaker 'SPEAKER_00'`.
- Las 19 celdas de código validadas con el `TransformerManager` de IPython (cero errores de sintaxis).
- 220 tests verdes, cobertura 85.90%, ruff check + format limpios, los 4 notebooks pasan schema + anti-bomba.

### Documented

- `docs/PROCESO_COMPLETO_ENTREVISTAS.md`: explicación desde cero del flujo de 2 librerías (speakerscribe diariza, voicelegacy clona), por qué el audio debe estar aislado, por qué las etiquetas varían entre archivos, y el paso a paso completo.

## [0.3.1] — 2026-05-19 — Turno 11 · fine-tuning standalone desde grabación cruda

Salto PATCH (0.3.0 → 0.3.1): nuevo notebook autónomo + extra opcional `finetune`. Sin cambios en la API del paquete; el código Python es idéntico a 0.3.0.

### Added

- **Notebook `notebooks/notebook_voicelegacy_finetune_standalone.ipynb`** (27 celdas, regenerable desde `build_finetune_standalone_notebook.py`). A diferencia de `notebook_voicelegacy_finetune.ipynb` (que requiere `reference_corpus/` + transcripciones de speakerscribe), este parte de **UNA grabación cruda** (mp3/m4a/wav/mp4) y hace todo: conversión a 22.05 kHz mono, transcripción con faster-whisper (word timestamps + VAD), segmentación inteligente en clips de 2-11 s por pausas/oraciones, limpieza con `voicelegacy.audio` (denoise + bandpass + loudness), construcción del dataset LJSpeech-like, fine-tuning del GPT encoder, materialización del checkpoint y A/B contra el base.
- **Extra opcional `finetune`** en `pyproject.toml`: `pip install voicelegacy[finetune]` añade `faster-whisper>=1.0,<2.0`. Incluido también en el meta-extra `all`.
- **Hyperparams calibrados para dataset pequeño** (15-25 min de voz neta, lo que rinde una grabación de 30 min tras quitar silencios): `NUM_EPOCHS=10` (vs 6), `LEARNING_RATE=3e-6` (vs 5e-6, anti catastrophic-forgetting), `WEIGHT_DECAY=5e-2` (vs 1e-2, regularización fuerte anti-overfit).

### Verified

- 220 tests verdes (sin cambios — el notebook no toca código del paquete)
- Cobertura 85.90%, piso 80%
- ruff check + format limpios sobre voicelegacy/ tests/ scripts/ notebooks/
- Las 27 celdas de código del notebook standalone validadas con el `TransformerManager` de IPython (cero errores de sintaxis incluyendo magics `!` y `=!`)
- Schema nbformat v4.5 + hook anti-bomba pasan en los 3 notebooks
- `python -m build` + `twine check` → PASSED para 0.3.1

### Documented

- `docs/PLAYBOOK_FINETUNING_30MIN.md`: guía paso a paso completa para fine-tunear desde una grabación de 30 min, con la advertencia honesta de que 30 min es el límite inferior (no el ideal de 2-5 h).

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
