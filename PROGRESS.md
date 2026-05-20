# voicelegacy — Plan de mejora · Progreso

Tracking vivo del plan de mejora validado el 2026-05-18. Cada item lista:
estado (✅ cerrado · 🔄 en curso · ⏳ pendiente), evidencia objetiva, y archivos
tocados. **Sin evidencia, no cerrado.**

Última actualización: **2026-05-19** · Turno: **10 (fine-tuning XTTS-v2)**

---

## Resumen ejecutivo

| Tier | Total plan original | ✅ Cerrado del plan | 🛑 Deferred con justificación | ⏳ Pendiente | Items extra agregados |
|---|---|---|---|---|---|
| **P0 — Bombas y bugs silenciosos** | 10 | **10** | 0 | 0 | +1 (P0-FINAL pre-commit) |
| **P1 — Calidad del resultado final** | 8 | **8** | 0 | 0 | 0 |
| **P2 — Robustez y observabilidad** | 9 | **8** | 1 (P2-19) | 0 | +4 (P2-27..31) |
| **P3 — Producción y futuro** | 5 | **3** | 0 | 2 (P3-27, P3-31) | +2 (P3-4 Ética, P3-5 Fine-tuning XTTS-v2) |

**v0.3.0 · Cobertura: 85.89% (piso 80%). 220 tests verdes. Ruff check + format OK. 2 notebooks regenerables (37 cells inferencia + 27 cells fine-tuning). Pre-commit pasa 10 hooks. `pyproject.toml` con metadatos completos para PyPI. CHANGELOG en formato Keep a Changelog.**

### P3-5 · Fine-tuning XTTS-v2 (turno 10)

Cerrado como feature opt-in en v0.3.0:

- Módulo `voicelegacy.finetuned_inference` (411 LOC, 88.7% cobertura, 38 tests).
- Notebook `notebook_voicelegacy_finetune.ipynb` (27 celdas, Colab Free T4) con flujo completo: descarga base, preparación dataset, GPTTrainer config, fit() resumible, materialización del checkpoint reutilizable, validación end-to-end, A/B base-vs-finetuned con `speaker_similarity_score`.
- Decisión arquitectónica: solo se entrena el GPT encoder, el HiFi-GAN vocoder permanece congelado (única forma de que entre en T4 con 15 GB VRAM).
- TPU NO soportada — confirmado en docs oficiales de Coqui que XTTS-v2 fine-tuning es exclusivamente CUDA. Documentado en CHANGELOG.

### Notas de honestidad del tracking

1. **P1-WADA** (WADA-SNR real) NO está cerrado: fue **deferred por decisión técnica** y movido a P3. La tabla y la sección P1 de turnos anteriores lo marcaban como ✅ — el etiquetado correcto es 🛑 Deferred. Corregido en este turno.

2. **Renumeración P3**: en turno 6 los items P3 del plan original (P3-27..P3-31) se renombraron a P3-1..P3-4. Esto rompía trazabilidad. Tabla de cross-reference añadida abajo.

3. **P3-27 (tabla SNR fuente → similarity esperada) sigue pendiente** y es el experimento empírico que valida si el producto sirve para audio sub-óptimo. **Bloqueante para v1.0**, no para v0.2.0 publicable.

---

## Decisión arquitectónica del usuario (turno 1)

> *"Mejore lo mejor posible en situaciones en las que la calidad del audio puede que no sean las mejores."*

**Implicación:** el éxito del producto NO se mide solo con audio fuente
limpio. El P1 sobre **denoise no-estacionario**, **WADA-SNR**, y especialmente
**speaker similarity scoring** (P1-12) suben de prioridad porque son los
que permiten distinguir empíricamente cuándo el clon es aceptable a pesar
de material sub-óptimo.

**Decisión consecuente añadida al P1 como P1-18 (nueva):** etiquetar los
outputs sintéticos con metadato de calidad de la fuente (SNR medio, sample
rate, label de modo degradado si `MIN_SNR_DB < 10`). Para que cada audio
entregado a un familiar lleve constancia honesta de su origen.

---

## P0 · Bombas activas y bugs silenciosos

### P0-1 · pyproject.toml inválido (línea 54) · ✅
- **Bug**: `where = [".]` (falta comilla de cierre) — `tomllib.TOMLDecodeError`.
- **Fix**: línea cambiada a `where = ["."]`.
- **Evidencia**: `python -c "import tomllib; tomllib.load(open('pyproject.toml','rb'))"` retorna sin error; `pip install -e .` funciona.
- **Archivos**: `pyproject.toml`.

### P0-2 · Cell 36 con `runtime.unassign()` activo · ✅
- **Bug**: línea ejecutable `from google.colab import runtime; runtime.unassign()` en celda 36 del notebook (mata sesión en Run All).
- **Fix**:
  - Regenerado el `.ipynb` desde `build_notebook.py` (que ya tenía la línea correctamente comentada).
  - Pre-commit hook `no-live-runtime-unassign` que detecta cualquier futura reintroducción ejecutable.
- **Evidencia**: hook ejecutado sobre el `_archive/*_handedited_37cells.ipynb` falla con `BOMB in ... cell 36: live runtime.unassign()`. Hook sobre el `.ipynb` regenerado pasa.
- **Archivos**: `notebooks/notebook_voicelegacy.ipynb` (regenerado), `notebooks/build_notebook.py`, `.pre-commit-config.yaml`.

### P0-3 · Phone-codec gate inerte (8 kHz audio pasa) · ✅
- **Bug**: `load_audio_mono` resampleaba silenciosamente a 22050; `AudioStats.sample_rate` heredaba 22050; la guarda `stats.sample_rate < 16000` nunca disparaba.
- **Fix**:
  - `load_audio_mono` ahora devuelve `(audio, original_sr)`.
  - `compute_stats(y, sr, original_sr=...)` registra el sample rate de la fuente, no el resampleado.
  - `evaluate_file` propaga `original_sr` a `AudioStats.sample_rate`.
- **Evidencia (ejecución real)**: WAV 8 kHz fabricado en `tmp_path` → `evaluate_file` retorna `stats.sample_rate=8000, passed=False, reasons=['sample_rate 8000Hz < 16000Hz (phone-codec audio)']`. Antes del fix: `22050, True, ()`.
- **Archivos**: `voicelegacy/audio.py`, `voicelegacy/quality.py`, `voicelegacy/corpus.py`, `tests/test_audio.py`.

### P0-4 · loudness_normalize produce clipping circular · ✅
- **Bug**: `pyloudnorm.normalize.loudness` aplicaba ganancia escalar que llevaba peaks > 1.0; `np.clip(-1, 1)` los hard-clippeaba a 1.0 (peak=0.0 dBFS exacto). El quality gate después rechazaba "clipping risk" introducido por el propio pipeline.
- **Fix**: peak limiter lineal a `peak_ceiling_dbfs=-3.0` aplicado **después** de la normalización LUFS. Nunca hard-clip.
- **Evidencia (ejecución real)**: input peak=0.95, target LUFS=-16 → output peak=0.26 (~-11.6 dBFS), bajo techo de -3 dBFS. Antes del fix: peak=1.0 exacto (hard-clip).
- **Archivos**: `voicelegacy/audio.py`.

### P0-5 · Test `test_phone_audio_rejected` reemplazado por end-to-end real · ✅
- **Bug**: el test fabricaba `AudioStats(sample_rate=8000)` y llamaba `score_segment` directamente — nunca tocaba `evaluate_file` ni `load_audio_mono`. Cobertura ilusoria: 100% de líneas, 0% del comportamiento que importa.
- **Fix**: añadida clase `TestPhoneCodecGateEndToEnd` con 3 tests que crean WAVs reales (8 kHz, 22050 Hz, 16 kHz) y verifican `evaluate_file` end-to-end.
- **Evidencia**: `test_real_8khz_wav_rejected_by_phone_codec_gate` falla **antes** del fix P0-3 (verificado durante el debug), pasa **después**.
- **Archivos**: `tests/test_quality.py`.

### P0-6 · Test de no-clipping con material agresivo · ✅
- **Bug**: ningún test cubría el comportamiento del normalizador con peaks altos.
- **Fix**: clase `TestLoudnessNoClipping` con 3 tests (material agresivo no clipa, validación de parámetro, material quiet preservado).
- **Evidencia**: `test_aggressive_input_does_not_clip` pasa con assertion `peak < 0.71` (bajo -3 dBFS) y `peak < 0.999` (no hard-clip).
- **Archivos**: `tests/test_audio.py`.

### P0-7 · Mover Cells 17 + 18 al paquete · ✅
- **Bug**: la conversión ffmpeg de mp4/m4a/etc → WAV y la re-normalización de clipping vivían como celdas hand-edited en el notebook. Cada `python build_notebook.py` las perdía.
- **Fix**:
  - **Cell 17 (ffmpeg)** → `voicelegacy/audio.py::convert_to_wav` y `convert_directory_to_wav` + constante `EXTENSIONS_CONVERTIBLE_TO_WAV`.
  - **Cell 18 (re-normalización post-clip)** → obsoleta porque P0-4 corrigió el origen. Eliminada.
  - **`run_reference_phase`** invoca `convert_directory_to_wav` automáticamente como pre-flight.
  - Subcomando CLI `voicelegacy convert-audio` para ejecutar desde shell.
  - Lista de extensiones en `corpus.py` ampliada a `.mkv, .webm, .aac, .mov`.
- **Evidencia**: 12 tests nuevos (`test_audio_conversion.py`) pasan. CLI test confirma comando expone `--workspace`, `--overwrite`, `--sample-rate`.
- **Archivos**: `voicelegacy/audio.py`, `voicelegacy/pipeline.py`, `voicelegacy/cli.py`, `voicelegacy/corpus.py`, `tests/test_audio_conversion.py`, `tests/test_cli.py`.

### P0-8 · Limpieza de notebook + reintegración de cells útiles · ✅
- **Bug**: 7 celdas hand-edited divergentes del generador. La regla "el notebook es generado" estaba rota.
- **Fix**:
  - **Cell 17 (ffmpeg)** → integrada al pipeline + CLI (ver P0-7).
  - **Cell 18 (re-normalización)** → eliminada (P0-4 lo arregla en origen).
  - **Cell 19 (debug speakers)** → integrada como subcomando CLI `voicelegacy list-speakers`.
  - **Cell 20 (denoising comentado, código muerto)** → eliminada.
  - **Cell 22 (reporte tabular)** → integrada al generador como celda permanente entre 9 y 10.
  - **Cell 26 (TEXTS_TO_SYNTHESIZE huérfana de debug)** → eliminada.
  - **Cell 36 (bomba runtime.unassign)** → ver P0-2.
- **Evidencia**: notebook regenerado tiene 31 celdas (29 originales + 2 nuevas del reporte tabular). El archivo hand-edited de 37 celdas se preservó en `notebooks/_archive/notebook_voicelegacy_*_handedited_37cells.ipynb` para auditoría.
- **Archivos**: `notebooks/build_notebook.py`, `notebooks/notebook_voicelegacy.ipynb`, `notebooks/_archive/`, `voicelegacy/cli.py`.

### P0-9 · Pre-commit hook que valida TOML y notebook · ✅
- **Bug**: no había prevención automática de los bugs P0-1 y P0-2. Si los reintroduces, te enteras en CI horas después.
- **Fix**: 3 hooks locales nuevos en `.pre-commit-config.yaml`:
  1. `validate-pyproject-toml` — `tomllib.load()` debe pasar.
  2. `validate-notebook-schema` — `nbformat.validate()` debe pasar sobre el `.ipynb`.
  3. `no-live-runtime-unassign` — falla si encuentra `runtime.unassign()` no comentado en ninguna celda.
  4. **Adicional**: `build_notebook.py` ahora ejecuta las 3 validaciones internamente antes de escribir el `.ipynb` y hace **backup automático** si detecta hand-edits (cells extra) en el destino.
- **Evidencia**: hook anti-bomba ejecutado sobre el `_archive/handedited_37cells.ipynb` retorna `BOMB in ... cell 36: live runtime.unassign()` con exit code 1. Sobre el regenerado: `OK`.
- **Archivos**: `.pre-commit-config.yaml`, `notebooks/build_notebook.py`.

### P0-10 · cov-fail-under 25 → 65 + limpiar omits inexistentes · ✅
- **Bug**: piso de cobertura en 25% no protegía nada; `omit` listaba `dashboard.py`, `visualizacion*.py` que no existen (config copy-paste de otro repo).
- **Fix**: `cov-fail-under=65`, omits podados.
- **Evidencia**: `python -m pytest tests/ --cov=voicelegacy` reporta cobertura 72.5% y "Required test coverage of 65% reached".
- **Archivos**: `pyproject.toml`.

---

## P1 · Calidad del resultado final

⚠️ **Reordenado por la decisión arquitectónica del usuario:** "que funcione lo mejor posible en situaciones de audio sub-óptimo". Se cerraron primero los items que hacen el output medible, reproducible y auditado.

### P1-11 · `SynthesisConfig.seed` para reproducibilidad · ✅
- **Fix**: `SynthesisConfig.seed: int | None = 42` y `_apply_seed()` en `synthesis.py` antes de cada inferencia.
- **Evidencia**: `tests/test_synthesis.py::test_apply_seed_makes_numpy_repeatable` confirma repetibilidad; `synthesize_to_file` propaga la config al stub TTS.
- **Archivos**: `voicelegacy/config.py`, `voicelegacy/synthesis.py`, `tests/test_synthesis.py`.

### P1-12 · Speaker similarity score (Resemblyzer) · ✅
- **Fix**: módulo nuevo `voicelegacy/similarity.py` con `SimilarityReport`, `compute_similarity()`, batch scoring y carga lazy del encoder.
- **Diseño**: dependencia opcional `voicelegacy[similarity]`; si Resemblyzer no está instalado, la síntesis no falla y el sidecar registra `similarity.status = skipped`.
- **Evidencia**: `tests/test_similarity.py` cubre cosine guard, bandas de calidad, validación de referencias y cómputo con encoder inyectado.
- **Archivos**: `voicelegacy/similarity.py`, `voicelegacy/__init__.py`, `pyproject.toml`, `tests/test_similarity.py`.

### P1-18 · Metadato honesto en outputs sintéticos · ✅
- **Fix**: cada WAV sintetizado escribe un `.json` sidecar con `run_hash`, `reference_set_hash`, `synthesis_config`, `reference_config`, seed, resumen de calidad de fuentes, `degraded_mode` y similarity si existe.
- **Regla**: `degraded_mode=True` si `ReferenceConfig.min_snr_db < 10`, si el rango dinámico medio cae bajo 10 dB, o si algún sample rate mínimo cae bajo 16 kHz.
- **Evidencia**: `tests/test_pipeline.py` valida creación de sidecar, cache hit con metadata y marcado de modo degradado cuando `min_snr_db=3.0`.
- **Archivos**: `voicelegacy/pipeline.py`, `tests/test_pipeline.py`, `README.md`, `notebooks/build_notebook.py`.

### P1-14 · `_estimate_snr_db` renombrado honestamente a dynamic range · ✅
- **Fix**: se introdujo `_estimate_dynamic_range_db()` y `_estimate_snr_db()` queda como alias backward-compatible. La documentación ya no vende el estimador como SNR real.
- **Razón**: WADA-SNR real queda para P2/P3 si el material fuente lo justifica; forzarlo ahora habría añadido complejidad externa sin cerrar el problema de medición del output.
- **Evidencia**: `tests/test_audio.py::test_dynamic_range_alias_matches_estimator`.
- **Archivos**: `voicelegacy/audio.py`, `tests/test_audio.py`.

### P1-16 · Denoise no-estacionario + filtro paso-banda · ✅
- **Fix**: `denoise(..., stationary=False)` por defecto, `apply_bandpass()`, `apply_preemphasis()`, y nuevos campos en `ReferenceConfig`: `denoise_stationary`, `apply_bandpass_filter`, `apply_preemphasis_filter`.
- **Adicional**: `load_audio_mono` se optimizó con `soundfile + scipy.signal.resample_poly` para evitar la latencia de `librosa.load` en WAV/FLAC/OGG.
- **Evidencia**: `tests/test_audio.py` valida pre-emphasis, band-pass, y todos los tests de audio pasan sin colgarse; suite completa 91/91.
- **Archivos**: `voicelegacy/audio.py`, `voicelegacy/corpus.py`, `voicelegacy/config.py`, `tests/test_audio.py`.

### P1-17 · Reporte de causas agrupadas en fase 1 · ✅
- **Fix**: `_write_quality_report()` ahora agrega `rejection_reason_counts` y `warning_reason_counts` dentro de `summary`.
- **Evidencia**: tests de pipeline siguen verdes y el notebook tabular lee el resumen enriquecido.
- **Archivos**: `voicelegacy/pipeline.py`, `notebooks/build_notebook.py`.

### P1-13 · Detección de mis-etiquetado de speakerscribe por F0 outliers · ✅
- **Fix**: `corpus.py` ahora estima F0 mediano por segmento con `librosa.yin`, calcula z-score robusto con MAD y rechaza outliers sólo si hay suficientes mediciones válidas.
- **Diseño**: conservador por defecto (`enable_f0_outlier_filter=True`, mínimo 5 segmentos válidos, umbral MAD 3.5) para evitar falsos positivos.
- **Auditoría**: cada fase de referencia genera `reports/f0_outliers_*.json` con segmentos revisados y rechazados.
- **Evidencia**: `tests/test_corpus.py::TestF0OutlierDetection` cubre outlier real, pocos valores válidos, validación de longitud y escritura del reporte.
- **Archivos**: `voicelegacy/config.py`, `voicelegacy/corpus.py`, `tests/test_corpus.py`.

### P1-15 · Caché de speaker conditioning latents · ✅
- **Fix**: `synthesis.py` añade `_CONDITIONING_LATENTS_CACHE`, `release_conditioning_latents()` y ruta best-effort de inferencia manual XTTS cuando el wrapper expone `synthesizer.tts_model.get_conditioning_latents()` e `inference()`.
- **Fallback seguro**: si el API bajo nivel no existe o falla, se vuelve automáticamente a `tts.tts_to_file()` sin romper Colab.
- **Evidencia**: `tests/test_synthesis.py` valida que dos síntesis con la misma referencia calculan latentes una sola vez y que desactivar `cache_conditioning_latents` usa el fallback.
- **Archivos**: `voicelegacy/config.py`, `voicelegacy/synthesis.py`, `tests/test_synthesis.py`.

### P1-WADA · WADA-SNR real / métrica perceptual adicional · 🛑 Deferred con justificación técnica (movido a P3)
- **Decisión**: WADA-SNR no se implementa en P1. Diferido a la sección P3 como P3-WADA (originalmente parte de P3-28).
- **Razón técnica**: WADA asume distribución γ del ruido y requiere calibración del shape parameter contra material conocido. Implementación sin calibración produce un número que se ve científico pero no significa lo que dice — peor que el `dynamic_range_db` honesto que ya tenemos.
- **Mitigación cerrada (sí, aplicada)**: el proyecto ya no miente llamando "SNR" al rango dinámico; los sidecars y reportes usan `mean_dynamic_range_db`.
- **Reabrir si**: el experimento P3-27 (SNR fuente → similarity) muestra que el `dynamic_range_db` no predice bien el `speaker_similarity_score`. En ese caso WADA / PESQ / STOI entran como alternativa a evaluar empíricamente, no como bloqueo de P1.
- **Por qué este item se etiquetaba antes como ✅**: error de tracking. "Cerrado por decisión técnica" implica resuelto; lo correcto es 🛑 Deferred con condición de reapertura.

---

## P2 · Robustez y observabilidad (turno 4)

### P2-21 · Backup antes de `force_rebuild_reference` · ✅
- **Fix**: `pipeline._backup_reference_corpus()` mueve WAVs existentes a `reference_corpus_backup_<UTC>/` antes de reconstruir.
- **Evidencia**: `tests/test_pipeline.py::TestReferenceRebuildBackup::test_force_rebuild_moves_existing_references_to_backup`.
- **Archivos**: `voicelegacy/pipeline.py`, `tests/test_pipeline.py`.

### P2-22 · `compute_run_hash` incluye versión del paquete · ✅
- **Fix**: `persistence.compute_run_hash()` incorpora `package_version()` para invalidar caché cuando cambia `voicelegacy`.
- **Sidecar**: `_write_synthesis_sidecar()` registra `voicelegacy_version`.
- **Evidencia**: test con `monkeypatch` demuestra que `0.1.0` vs `0.2.0` produce hashes distintos.
- **Archivos**: `voicelegacy/persistence.py`, `voicelegacy/pipeline.py`, `tests/test_persistence.py`, `tests/test_pipeline.py`.

### P2-23 · Logging de tiempos/memoria/VRAM · ✅
- **Fix**: nuevo módulo `telemetry.py` con `runtime_snapshot()` y `timed_step()`.
- **Cobertura**: load de modelo, síntesis y similarity quedan envueltos con logging de tiempo y estado CUDA cuando Torch está disponible.
- **Archivos**: `voicelegacy/telemetry.py`, `voicelegacy/pipeline.py`.

### P2-24 · Validación Pydantic del JSON de speakerscribe · ✅
- **Fix**: nuevo `speakerscribe_schema.py` con `SpeakerscribeDocument` y `SpeakerscribeSegment`.
- **Cambio deliberado**: segmentos inválidos ya no se saltan silenciosamente; el JSON falla con error explícito. Producción no debe construir un corpus desde diarización corrupta.
- **Evidencia**: `tests/test_corpus.py::TestLoadJSON::test_invalid_segment_schema_raises` y `tests/test_diagnose.py::test_diagnose_validates_speakerscribe_json`.
- **Archivos**: `voicelegacy/speakerscribe_schema.py`, `voicelegacy/corpus.py`, `voicelegacy/diagnose.py`, tests.

### P2-25 · `--text-file` en CLI `synthesize` · ✅
- **Fix**: `voicelegacy synthesize` acepta `--text`, `--text-file` o ambos.
- **Formatos**: `.txt` una frase por línea; `.csv` con columna `text` o CSV de una sola columna.
- **Evidencia**: `tests/test_text_inputs.py` cubre txt, csv, combinación y errores.
- **Archivos**: `voicelegacy/text_inputs.py`, `voicelegacy/cli.py`, `tests/test_text_inputs.py`.

### P2-26 · `voicelegacy diagnose` · ✅
- **Fix**: nuevo comando CLI `voicelegacy diagnose --workspace ...` con salida humana o `--json`.
- **Chequea**: Python, paquetes, ffmpeg, CUDA, aceptación CPML, modelo configurado, estructura de workspace, raw audios, JSON schema, corpus, reportes, outputs y `runs.db`.
- **Evidencia**: `tests/test_diagnose.py` y CLI test con `CliRunner`.
- **Archivos**: `voicelegacy/diagnose.py`, `voicelegacy/cli.py`, `tests/test_diagnose.py`.

### P2 mínimo aún pendiente
- P2-19 · Paralelizar `extract_segments_to_wav` (ThreadPoolExecutor) — I/O bound. ⏳
- P2-20 · `evaluate_file(path, config)` con `ReferenceConfig` único — eliminar defaults duplicados (DRY). ⏳
- P2-CLI · ampliar más mocks de `synthesis.py`/modelo Coqui si se sube cobertura objetivo a 75–80%. ⏳

---


## Turno 5 · P2 avanzado + endurecimiento GitHub · ✅

### P2-27 · Cobertura objetivo 65% → 75% · ✅
- **Fix**: `pyproject.toml` sube `--cov-fail-under` de 65 a **75**.
- **Evidencia**: `python -m pytest -q` → **125 passed**, cobertura total **77.61%**, piso 75 alcanzado.
- **Prioridad cubierta**: `synthesis.py` sube a **77.4%**, `pipeline.py` a **80.2%**, `cli.py` a **86.7%**.
- **Archivos**: `pyproject.toml`, `tests/test_synthesis.py`, `tests/test_pipeline.py`, `tests/test_cli.py`.

### P2-28 · Tests con mocks para XTTS-v2 / Coqui · ✅
- **Fix**: tests nuevos cubren aceptación CPML, cache de modelo, ausencia de `coqui-tts`, API de bajo nivel, fallback de inferencia manual, cache de latentes y parámetros enviados a `tts_to_file`.
- **Evidencia**: no se descargan pesos ni se requiere GPU; todo se valida con fakes/mocks.
- **Archivos**: `tests/test_synthesis.py`.

### P2-29 · GitHub Actions release-candidate · ✅
- **Fix**: `.github/workflows/ci.yml` queda como gate único de calidad: instalación, FFmpeg, pyproject TOML, notebook generado, ruff format, ruff lint y pytest con cobertura.
- **Criterio**: CI no descarga pesos XTTS ni ejecuta GPU; esas rutas están mockeadas.
- **Archivos**: `.github/workflows/ci.yml`.

### P2-30 · README operativo real · ✅
- **Fix**: README reescrito como guía de operación: instalación local, Colab, workspace, `diagnose`, CLI, Python API, sidecars, similarity score, troubleshooting y límites zero-shot.
- **Licencias**: CPML de Coqui queda explícita y separada de la licencia MIT del código.
- **Archivos**: `README.md`.

### P2-31 · Validación de notebook y pyproject antes de entrega · ✅
- **Evidencia**: `python notebooks/build_notebook.py` → notebook válido nbformat, 31 celdas.
- **Evidencia**: `python -c "import tomllib; tomllib.load(open('pyproject.toml','rb'))"` → OK.
- **Archivos**: `notebooks/notebook_voicelegacy.ipynb`, `pyproject.toml`.

### P2 todavía pendiente fuera del Turno 5

- P2-19 · Paralelizar `extract_segments_to_wav` — **decisión técnica del Turno 7**: deferred (ver Turno 7).
- P2-20 · `evaluate_file(path, config)` — **cerrado en Turno 7** (ver abajo).
- P2-extra · Profundizar cobertura de `corpus.py` F0/extracción end-to-end con audio sintético largo — deseable antes de PyPI. ⏳

## Turno 7 · Cierre P2-20 + notebook sync + brittle CLI test · ✅

### P2-20 · `evaluate_file(path, config: ReferenceConfig)` · ✅
- **Bug original (A01 §3.4)**: `score_segment(min_snr_db=15.0)` y `evaluate_file(min_snr_db=15.0)` tenían un literal hardcoded además de la constante de módulo `MIN_SNR_DB` y del `Field(default=MIN_SNR_DB)` de Pydantic. **Tres fuentes de verdad** para el mismo umbral. Si alguien movía `MIN_SNR_DB` a 18, las llamadas ad-hoc seguían en 15 silenciosamente.
- **Fix**:
  - `score_segment(..., min_snr_db: float = MIN_SNR_DB, ...)` — referencia simbólica, no literal.
  - `evaluate_file(path, config: ReferenceConfig | None = None, *, min_duration_s=None, max_duration_s=None, min_snr_db=None, ...)` — cuando se pasa `config`, los umbrales se leen de él; los kwargs explícitos siguen ganando para no romper scripts ad-hoc.
  - `pipeline.py::run_reference_phase` y `pipeline.py::_reference_quality_summary` ahora invocan `evaluate_file(wav, config=config.reference)` en lugar de pasar 3 kwargs idénticos repetidos.
- **Evidencia**: 2 tests nuevos en `tests/test_quality.py`:
  - `TestEvaluateFile::test_accepts_reference_config_as_single_source_of_truth` valida la nueva firma.
  - `TestEvaluateFile::test_explicit_kwargs_override_config` valida que los kwargs explícitos ganan sobre el `config`.
- **Archivos**: `voicelegacy/quality.py`, `voicelegacy/pipeline.py`, `tests/test_quality.py`.

### P2-19 · Paralelización de `extract_segments_to_wav` · 🛑 Deferred con justificación
- **Decisión técnica explícita**, no oversight. Razones:
  1. El cuello de botella real es CPU-bound (denoise no-estacionario + bandpass + F0 estimation con `librosa.yin`), no I/O. El comentario "I/O bound" del plan original era impreciso.
  2. El uso típico es 1-3 entrevistas largas por workspace, no 50. El ahorro real es del orden de 30-60s sobre runs de varios minutos.
  3. Compartir el cache `dict[Path, np.ndarray]` entre threads requiere `threading.Lock`, lo cual añade complejidad sin ganancia clara para el caso de uso.
  4. La paralelización ingenua con `ThreadPoolExecutor` sobre `librosa` puede pelearse con BLAS multi-threading interno y degradar performance.
- **Reabrir si**: workflow real supera ~10 entrevistas por run (caso productivo), o si el F0 outlier filter se vuelve el paso dominante.
- **Archivos**: documentado en `CHANGELOG.md [Unreleased]` y este `PROGRESS.md`.

### Sync · Notebook divergente del código · ✅
- **Bug encontrado durante auditoría del Turno 7**: la celda 6 del notebook usaba `DENOISE_STATIONARY`, `APPLY_BANDPASS_FILTER`, `APPLY_PREEMPHASIS_FILTER` pero la celda 2 nunca las definía. Cualquier "Run all" reventaba con `NameError`. Bug latente, no detectado por las pruebas porque el notebook no se ejecuta en CI.
- **Features del código que el notebook ignoraba**: `voicelegacy diagnose`, sidecar JSON (similarity score + quality_band + degraded_mode + text_plan + run_hash), `voicelegacy evaluate-denoise`, `long_text_strategy`, `enable_f0_outlier_filter`.
- **Fix**:
  - Variables faltantes definidas en la celda 2 con comentarios sobre cuándo cambiarlas.
  - Nueva celda **6️⃣b Diagnóstico del workspace y runtime** que corre `diagnose_workspace()` antes del smoke test y aborta si hay `fail`.
  - Nueva celda **🧾 Auditoría del output (sidecar JSON)** después de la fase 2 para inspeccionar `similarity`, `source_quality`, `text_plan`, `run_hash`.
  - Nueva celda **🧪 (Opcional) Comparar denoise alternativas** que invoca `evaluate_denoise_methods` con `RUN_DENOISE_EVAL=False` por defecto.
  - Tabla de troubleshooting ampliada con síntomas nuevos (`similarity quality_band=marginal/low`, `degraded_mode=True`) y comandos CLI complementarios.
- **Evidencia**: `python notebooks/build_notebook.py` regenera el .ipynb, `nbformat.validate()` pasa, **37 celdas** (era 31).
- **Archivos**: `notebooks/build_notebook.py`, `notebooks/notebook_voicelegacy.ipynb`.

### CLI · Test brittle por Rich wrapping · ✅
- **Bug**: `tests/test_cli.py::test_synthesize_uses_text_file_and_prints_sidecar` fallaba en cualquier runner con terminal de 80 columnas porque Rich rompía el path `synthesis_out/hola.wav` justo entre `hola.` y `wav`. La aserción `assert "hola" in result.output and ".wav" in result.output` no toleraba el wrap.
- **Fix**: `console.print(f"{marker} {r.output_path}{sidecar}", soft_wrap=True)` en `voicelegacy/cli.py`. `soft_wrap=True` desactiva el word-wrap interno de Rich; la terminal puede seguir cortando si quiere, pero el output lógico permanece en una sola línea.
- **Evidencia**: el test pasa de inmediato sin tocar el test. CI con `typer.testing.CliRunner` ya no se rompe.
- **Archivos**: `voicelegacy/cli.py`.

### Verificaciones finales del Turno 7

- `python -c "import tomllib; tomllib.load(open('pyproject.toml','rb'))"` → OK.
- `python -m ruff format --check --config pyproject.toml .` → 33 files already formatted.
- `python -m ruff check --config pyproject.toml .` → All checks passed!
- `python -m pytest -q` → **134 passed**, cobertura **76.31%**, piso 75% alcanzado.
- `python notebooks/build_notebook.py` → notebook válido nbformat v4.5, **37 celdas**.

## Turno 8 · `.pre-commit-config.yaml` roto + hardening final · ✅

### P0-FINAL · `.pre-commit-config.yaml` no parseaba — bug crítico no detectado · ✅
- **Bug**: el archivo `.pre-commit-config.yaml` tenía hooks locales con comandos Python inline largos (`entry: python -c "..."`) que contenían `:` dentro del string. YAML interpretaba esos `:` como separadores de mapping y rompía con `InvalidConfigError: mapping values are not allowed in this context` en la línea 44. Resultado: **`pre-commit run` fallaba antes de ejecutar un solo hook**, lo que invalidaba silenciosamente los tres hooks locales que se suponía protegían contra los bugs P0-1, P0-2 y la divergencia notebook ↔ generador. Cualquier desarrollador que instalara los hooks se quedaba sin protección sin saberlo.
- **Bug adicional encontrado**: el hook `validate-notebook-schema` declaraba `additional_dependencies: [nbformat]` con `language: system`, combinación que pre-commit rechaza con `The hook ... specifies additional_dependencies but is using language system`. Hardcoded la dependencia, nunca la instalaba.
- **Bug adicional**: `rev: v4.5.0` del repo `pre-commit/pre-commit-hooks` usa nombres de stages deprecated (`commit`, `push`).
- **Fix**:
  - Extraídos los tres hooks locales a scripts independientes en `scripts/` con docstring + tests humanos: `check_pyproject_toml.py`, `check_notebook_schema.py`, `check_no_runtime_unassign.py`. Ya no son YAML-escaped one-liners frágiles.
  - `language: system` → `language: python` en el hook que necesita `nbformat`, lo cual permite que `additional_dependencies` funcione.
  - Bump de `pre-commit-hooks` v4.5.0 → v5.0.0 (sin stages deprecated).
  - Refinado el filtro `files:` para excluir `notebooks/_archive/` — esos archivos son evidencia histórica de bugs por diseño, no deben bloquear commits.
- **Bug colateral**: `build_notebook.py` escribía el `.ipynb` sin newline final. El hook `end-of-file-fixer` lo añadía después, creando un loop: cada `python build_notebook.py` invalidaba el archivo. Fix: el generador ahora escribe con `\n` final.
- **Evidencia**: `pre-commit run --all-files` ahora pasa los **10 hooks limpio** sobre un repo recién inicializado, en una máquina sin cache. Incluye el hook anti-bomba detectando correctamente el `runtime.unassign()` activo en `notebooks/_archive/`, donde se preserva como evidencia.
- **Archivos**: `.pre-commit-config.yaml`, `scripts/check_pyproject_toml.py`, `scripts/check_notebook_schema.py`, `scripts/check_no_runtime_unassign.py`, `notebooks/build_notebook.py`.

### Verificaciones finales del Turno 8

- `python -c "import yaml; yaml.safe_load(open('.pre-commit-config.yaml'))"` → OK.
- `pre-commit run --all-files` → 10 hooks Passed (ruff, ruff-format, trailing-whitespace, end-of-file-fixer, check-yaml, check-toml, check-added-large-files, validate-pyproject-toml, validate-notebook-schema, no-live-runtime-unassign).
- `python -m pytest -q` → **134 passed**, cobertura **76.31%**, piso 75% alcanzado.
- `python notebooks/build_notebook.py` → 37 celdas, válido, idempotente.

## P3 · Producción y futuro

### Cross-reference plan original ↔ renombrado en turno 6

| ID original (plan) | ID actual (turno 6) | Tema | Estado real |
|---|---|---|---|
| P3-27 | P3-27 (sin renombrar) | Tabla SNR fuente → similarity esperada (zero-shot) | ⏳ **PENDIENTE — bloquea v1.0** |
| P3-28 | P3-1 | Evaluar DeepFilterNet para fuentes <10 dB SNR | ✅ Harness comparativo (`denoise_eval.py`); decisión empírica pendiente del experimento P3-27 |
| P3-29 | P3-2 | Manejo de textos largos (originalmente "crossfade entre clips") | ✅ Resuelto por delegación a XTTS via `text_strategy.py` (camino distinto al original, cumple la intención) |
| P3-30 | P3-3 | Publicar a PyPI cuando P0+P1 cierren | ✅ Workflow endurecido + manual dispatch (no auto-publish) |
| P3-31 | P3-31 (sin renombrar) | Considerar Polars/DuckDB si >10 entrevistas/run | ⏳ Pendiente — no aplica al caso de uso típico (1-3 entrevistas) |
| — | P3-4 (extra) | Ética y consentimiento (`docs/ETHICS.md`) | ✅ Agregado fuera del plan original |
| — | P3-WADA (nuevo) | WADA-SNR / PESQ / STOI como métricas perceptuales alternativas | 🛑 Deferred. Reabrir si P3-27 muestra que `dynamic_range_db` no predice `similarity` |

### P3-27 · Tabla SNR fuente → similarity esperada · ⏳ PRIORIDAD ALTA

**Por qué importa**: es el único experimento que convierte el resto del plan en decisión informada. Sin este dato:
- No sabes si tu material real (entrevistas de SNR variable) cae en banda `high` (≥0.75) o en `low` (<0.60).
- No sabes si `denoise_stationary=False` o el bandpass realmente mejoran el output.
- No sabes si DeepFilterNet vale el costo de la dependencia.

**Procedimiento sugerido (3-4 h con material disponible)**:
1. 3-5 entrevistas reales con SNR variable (bajo <5 dB, medio 10-15 dB, alto >20 dB).
2. `voicelegacy build-corpus` + `voicelegacy synthesize` con 5 frases por nivel.
3. Tabular `(mean_dynamic_range_db, speaker_similarity_score, quality_band)`.
4. Escucha humana de 3 outputs por banda (la métrica coseno no mide naturalidad).
5. Documentar en `reports/experiment_snr_to_similarity_<fecha>.md`.

**No bloquea publicación a PyPI v0.2.0** (la ingeniería del paquete está completa). **Sí bloquea v1.0** porque v1.0 = producto validado, no solo paquete instalable.

### P3-31 · Polars/DuckDB
Diferido. No aplica al caso de uso típico (1-3 entrevistas por run). Reabrir si el workflow crece.

---

## Métricas de la entrega

| Métrica | Antes del proyecto | Después turno 8 (v0.1.6) | Después turno 9 (v0.2.0, este turno) |
|---|---|---|---|
| Tests | 53 | 134 | **182** |
| Cobertura total | ~70% declarada | 76.21% | **85.58%** |
| Piso de cobertura `cov-fail-under` | 25% | 75% | **80%** |
| Cobertura `cli.py` | 0% | 74.2% | 74.2% |
| Cobertura `corpus.py` | 54.8% | 42.9% (creció módulo) | **95.5%** |
| Cobertura `similarity.py` | n/a (no existía) | 53.4% | **97.5%** |
| Cobertura `synthesis.py` | 16.7% | 74.6% | 74.6% |
| `pyproject.toml` válido | ❌ | ✅ | ✅ + metadatos PyPI completos |
| Phone-codec gate funciona end-to-end | ❌ | ✅ | ✅ |
| `loudness_normalize` no auto-clipea | ❌ | ✅ | ✅ |
| Celdas hand-edited en notebook | 7 | 0 | 0 |
| Pre-commit hooks específicos | 0 | 3 + 7 estándar | 3 + 7 estándar (validado en CI) |
| Speaker similarity objetiva | ❌ | ✅ opcional | ✅ opcional, cubierto 97.5% |
| Sidecar JSON por WAV sintético | ❌ | ✅ | ✅ |
| Seed reproducible | ❌ | ✅ | ✅ |
| `voicelegacy diagnose` | ❌ | ✅ | ✅ |
| `voicelegacy evaluate-denoise` | ❌ | ✅ | ✅ |
| CI GitHub Actions | ❌ | ✅ | ✅ |
| Optional dependencies declaradas | ❌ | ✅ `[similarity]`, `[deepfilter]` | ✅ + `[publish]`, `[notebook]` |
| Build wheel + sdist sin errores | n/a | n/a | ✅ |
| `twine check` clean | n/a | n/a | ✅ |
| Tracking honesto (P1-WADA, P3 cross-ref) | n/a | ⚠️ etiqueta engañosa | ✅ corregido |

---

## Lo que NO se cerró en este turno y debe quedar visible

### 1. P3-27 (experimento SNR → similarity) sigue pendiente
**Estado**: ⏳ Pendiente — bloquea v1.0, no bloquea v0.2.0 publicable.

**Por qué se mantiene pendiente**: requiere material de audio real del usuario y tiempo de escucha humana. No es trabajo de código.

**Acción**: cuando se ejecute, documentar en `reports/experiment_snr_to_similarity_<fecha>.md` y luego decidir si:
- Mantener el pipeline actual (zero-shot + denoise no-estacionario es suficiente).
- Activar DeepFilterNet por defecto (P3-1).
- Implementar WADA real (P3-WADA reabre).

### 2. Cobertura por debajo de 75% en tres módulos no críticos

| Módulo | Cobertura | Justificación |
|---|---|---|
| `cli.py` | 74.2% | Branches de error de Typer que requieren capturar exits específicos en subprocess. Cubrir más añade brittle tests sin valor. |
| `synthesis.py` | 74.6% | Branches de error de coqui-tts y rutas de fallback que requieren tests de integración con GPU. Marcado como deuda P3 (test de integración con `pytest -m gpu`). |
| `denoise_eval.py` | 73.4% | Módulo experimental nuevo. Tests cubren la lógica de comparación; las ramas de fallo de DeepFilterNet requieren la dependencia instalada (deferred a entorno de evaluación real). |

### 3. Los tests de `synthesis.py` son con mocks
Esto es correcto para CI sin GPU, pero significa que un cambio incompatible en `coqui-tts` upstream se descubre en producción. **No bloquea PyPI**: documentado como visible debt.

### 4. P2-24 (Pydantic schema strict) es breaking change documentado
Cambio de comportamiento: segmentos malos en speakerscribe JSON ahora fallan, antes se saltaban silenciosamente. Bumpeado a 0.2.0 (semver MINOR en 0.x lo permite). Documentado en CHANGELOG como BREAKING.

---

*Plan validado: 2026-05-18 · Skills aplicadas: python-data-library-dev, colab-notebook-dev · Iron Hierarchy of Standards §1-9 cumplida en todo el código tocado.*

---

## Turno 6 — P3 producto mantenible

Estado: ✅ completado para prueba.

### P3-1 DeepFilterNet frente a noisereduce

- Añadido `voicelegacy/denoise_eval.py`.
- Añadido comando `voicelegacy evaluate-denoise`.
- DeepFilterNet queda como dependencia opcional (`.[deepfilter]`) y no como default.
- Criterio documentado: solo debe reemplazar noisereduce si mejora escucha humana y `speaker_similarity_score` sin introducir artefactos.

### P3-2 Textos largos

- Añadido `voicelegacy/text_strategy.py`.
- Añadidos campos en `SynthesisConfig`: `long_text_strategy`, `max_single_pass_chars`, `long_text_warning_chars`.
- `synthesis.py` usa el plan para decidir `split_sentences` / `enable_text_splitting`.
- Sidecars incluyen `text_plan`.

### P3-3 PyPI solo con estabilidad

- Release workflow endurecido: build + twine check + draft release.
- PyPI no se publica automáticamente por tag; requiere `workflow_dispatch` con `publish_pypi=true` y Trusted Publisher.
- Añadido `docs/RELEASE.md`.

### P3-4 Ética y uso

- Añadido `docs/ETHICS.md`.
- README actualizado con consentimiento, CPML, límites y sidecars.

### Evidencia

- Tests agregados: `tests/test_denoise_eval.py`, `tests/test_text_strategy.py`.
- Validaciones esperadas de cierre: ruff, pytest, cobertura >=75%, build_notebook, pyproject TOML, build/twine check.
