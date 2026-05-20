"""Build the voicelegacy companion notebook programmatically.

Generating .ipynb JSON by hand is fragile (one missed comma kills the file).
This script constructs it with the nbformat-compatible dict and writes it,
guaranteed-valid.
"""

from __future__ import annotations

import json
import uuid
from pathlib import Path


def _cell_id() -> str:
    """Return a short unique cell id (nbformat ≥4.5 requires it)."""
    return uuid.uuid4().hex[:12]


def md_cell(source: str) -> dict:
    return {
        "cell_type": "markdown",
        "id": _cell_id(),
        "metadata": {},
        "source": source.splitlines(keepends=True),
    }


def code_cell(source: str) -> dict:
    return {
        "cell_type": "code",
        "id": _cell_id(),
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": source.splitlines(keepends=True),
    }


CELLS: list[dict] = []

# ─── Cell 0: title + checklist ──────────────────────────────────────
CELLS.append(
    md_cell("""# 🎙️ voicelegacy — Notebook de producción

**Clonación de voz por zero-shot con XTTS-v2, optimizado para Google Colab Free Tier (T4 GPU).**

Pipeline integrado con [speakerscribe](https://github.com/EnriqueForero/speakerscribe): consume sus JSON diarizados, construye un corpus curado de la voz objetivo, y sintetiza nuevo audio.

Powered by [coqui-tts (fork de Idiap)](https://github.com/idiap/coqui-ai-TTS) + XTTS-v2.

---

## ✅ Antes de empezar — checklist obligatorio

1. **Activa la GPU**: `Runtime → Change runtime type → T4 GPU`. Sin GPU, la inferencia es ~30× más lenta.

2. **Acepta la licencia CPML del modelo** antes de tocar nada: https://coqui.ai/cpml. Es uso no comercial / personal. Para legado familiar está dentro del scope; para una app comercial **léela completa**.

3. **Procesa primero tus entrevistas con `speakerscribe`**. Este notebook *consume* sus salidas, no transcribe. Si no tienes los `.json` diarizados, no puede continuar.

4. **Layout esperado del workspace en Drive**:
   ```
   <WORKSPACE>/
   ├── interviews_raw/        ← audio original (mp3/wav/m4a)
   ├── speakerscribe_out/     ← .json producidos por speakerscribe
   ├── reference_corpus/      ← (lo crea este notebook)
   ├── synthesis_out/         ← (lo crea este notebook)
   └── reports/               ← (lo crea este notebook)
   ```

5. **Reglas duras de calidad** (se aplican automáticamente):
   - Audio de llamadas (≤16 kHz) → **rechazado**. La compresión de codec contamina el clon.
   - Segmentos < 4 s o > 15 s → descartados.
   - SNR < 15 dB → descartado.
   - Solo se usan los top-N segmentos por score (SNR × duración).
""")
)

# ─── Cell 1: mount Drive ────────────────────────────────────────────
CELLS.append(
    md_cell("""---

## 1️⃣ Montar Google Drive
""")
)

CELLS.append(
    code_cell("""from google.colab import drive
drive.mount('/content/drive')
""")
)

# ─── Cell 2: Configuration ─────────────────────────────────────────
CELLS.append(
    md_cell("""---

## 2️⃣ Configuración del run

**Esta es la única celda que debes editar.** Todo lo demás se autoconfigura.
""")
)

CELLS.append(
    code_cell("""# ══════════════════════════════════════════════════════════════════
#  ✏️  EDITAR AQUÍ — única celda que requiere cambios
# ══════════════════════════════════════════════════════════════════

# ── Workspace (carpeta raíz en Drive) ─────────────────────────────
WORKSPACE = '/content/drive/MyDrive/Legado/Abuela'

# ── Ruta a la librería voicelegacy en Drive (modo desarrollo) ────
# Sube la carpeta voicelegacy (con su pyproject.toml) a Drive y apunta aquí.
# Si vas a usar PyPI (futuro), deja esto en None y descomenta la celda B.
PACKAGE_DIR_IN_DRIVE = '/content/drive/MyDrive/Repos/voicelegacy'

# ── Speaker objetivo en los JSON de speakerscribe ────────────────
# Abre uno de los .transcript.md de speakerscribe y verifica qué etiqueta
# corresponde a la abuela. Suele ser 'SPEAKER_00' o 'SPEAKER_01'.
# Tip: desde shell `voicelegacy list-speakers --workspace <ws>` te muestra
# todas las etiquetas detectadas y cuántos segmentos tienen.
TARGET_SPEAKER = 'SPEAKER_00'

# ── Selección de corpus ──────────────────────────────────────────
TOP_N_SEGMENTS = 10              # Top N segmentos limpios para usar como referencia
MIN_SEGMENT_DURATION_S = 4.0     # Descartar segmentos más cortos
MAX_SEGMENT_DURATION_S = 15.0    # Descartar segmentos más largos
MIN_SNR_DB = 15.0                # SNR mínimo aceptable
APPLY_DENOISE = True             # Reducción de ruido espectral (noisereduce)

# ── Limpieza avanzada (audio sub-óptimo) ─────────────────────────
# El default es no-estacionario porque se comporta mejor con ruido real
# de entrevistas (toses, carros, voces lejanas). Pasa True solo si tu
# fuente tiene ruido constante (zumbido eléctrico, AC continuo).
DENOISE_STATIONARY = False
APPLY_BANDPASS_FILTER = True     # 80-7600 Hz; limpia rumble y siseo
APPLY_PREEMPHASIS_FILTER = False # Útil sólo para fuentes muy apagadas / archivos antiguos
ENABLE_F0_OUTLIER_FILTER = True  # Detecta segmentos mal-etiquetados por F0

# ── Síntesis ──────────────────────────────────────────────────────
LANGUAGE = 'es'                  # 'es', 'en', 'pt', 'it', 'fr', ...
TEMPERATURE = 0.7                # 0.5-0.9 — estabilidad vs expresividad
SPEED = 1.0                      # 0.8 = más lento, 1.2 = más rápido
SEED = 42                        # Reproducibilidad del WAV generado
COMPUTE_SIMILARITY = True        # Requiere: pip install voicelegacy[similarity] o resemblyzer

# ── Estrategia para textos largos ────────────────────────────────
# 'auto' (recomendado): no divide si el texto cabe en max_single_pass_chars,
#                       divide cuando es más largo. Mejor balance.
# 'single_pass'        : nunca divide. Mejor para frases cortas; falla en textos largos.
# 'coqui_split'        : siempre delega a XTTS. Útil para textos muy largos (>600 chars).
LONG_TEXT_STRATEGY = 'auto'
MAX_SINGLE_PASS_CHARS = 240      # Bajo este umbral, NO se divide (reduce voice drift)
LONG_TEXT_WARNING_CHARS = 600    # Sobre este umbral, el sidecar emite warning

# ── Textos a sintetizar (lista) ──────────────────────────────────
TEXTS_TO_SYNTHESIZE = [
    'Mi querido nieto, hoy quiero contarte algo importante: '
    'la paciencia es la virtud que más vale la pena cultivar.',
    'Cuando tengas dudas, escucha a las personas que te aman, '
    'pero al final decide tú, con la cabeza fría.',
]

# ── Aceptación de licencia del modelo (OBLIGATORIO) ──────────────
# Al poner True confirmas que leíste https://coqui.ai/cpml
ACCEPT_CPML = False  # ← cambia a True después de leer la licencia

# ── Re-procesar (overrides de caché) ─────────────────────────────
FORCE_REBUILD_REFERENCE = False  # True = reconstruir corpus aunque exista
FORCE_RESYNTHESIZE = False       # True = re-sintetizar aunque exista cache

# ── Smoke test ───────────────────────────────────────────────────
RUN_SMOKE_TEST = True            # Probar carga del modelo antes del run real
""")
)

# ─── Cell 3: pre-flight ─────────────────────────────────────────────
CELLS.append(
    md_cell("""---

## 3️⃣ Pre-flight: verificar entorno (GPU, RAM, versiones)
""")
)

CELLS.append(
    code_cell("""import sys, platform, subprocess
from pathlib import Path

# Asegurar que el workspace existe
ws_root = Path(WORKSPACE)
if not ws_root.exists():
    print(f'⚠️  El workspace {WORKSPACE} no existe. Lo crearé.')
    ws_root.mkdir(parents=True, exist_ok=True)

# Python
print(f'🐍 Python {platform.python_version()}')
if sys.version_info >= (3, 13):
    print('   ⚠️  Python 3.13+ puede tener incompatibilidades con coqui-tts.')

# GPU
try:
    nvidia_out = subprocess.check_output(
        ['nvidia-smi', '--query-gpu=name,memory.total', '--format=csv,noheader'],
        text=True,
    )
    print(f'🎮 GPU detectada: {nvidia_out.strip()}')
except (subprocess.CalledProcessError, FileNotFoundError):
    print('❌ No se detectó GPU. Ve a Runtime → Change runtime type → T4 GPU.')
    raise SystemExit('GPU requerida para velocidad razonable.')

# RAM
try:
    import psutil
    ram_gb = psutil.virtual_memory().total / (1024**3)
    print(f'💾 RAM total: {ram_gb:.1f} GB')
    if ram_gb < 11:
        print('   ⚠️  Menos de 11 GB. Posible OOM con archivos largos.')
except ImportError:
    pass

# CPML
if not ACCEPT_CPML:
    print()
    print('⛔ ACCEPT_CPML está en False.')
    print('   Lee https://coqui.ai/cpml — si aceptas, cambia ACCEPT_CPML=True arriba.')
    raise SystemExit('Aceptación de licencia pendiente.')

print('\\n✅ Pre-flight OK')
""")
)

# ─── Cell 4: install package ────────────────────────────────────────
CELLS.append(
    md_cell("""---

## 4️⃣ Instalar `voicelegacy` y dependencias

**OPCIÓN A — desarrollo**: instala desde Drive en modo editable.
**OPCIÓN B — producción**: instala desde PyPI (cuando esté publicado).

Después de instalar, **reinicia el runtime** (`Runtime → Restart runtime`) la primera vez.
""")
)

CELLS.append(
    code_cell("""%%time
# ── Opción A: instalar desde Drive (DEVELOPMENT) ────────────────
import subprocess, sys
from pathlib import Path

if PACKAGE_DIR_IN_DRIVE and (Path(PACKAGE_DIR_IN_DRIVE) / 'pyproject.toml').exists():
    print(f'📦 Instalando desde {PACKAGE_DIR_IN_DRIVE} (editable)...')
    result = subprocess.run(
        [sys.executable, '-m', 'pip', 'install', '-e', PACKAGE_DIR_IN_DRIVE, '--quiet'],
        capture_output=True, text=True,
    )
    if result.returncode == 0:
        print('✅ voicelegacy instalado en modo editable')
        print('⚠️  Si es la PRIMERA instalación: Runtime → Restart runtime,')
        print('   luego re-corre desde la celda 1.')
    else:
        print('❌ Falló la instalación local:')
        print(result.stderr[-1500:])
        print('\\n   Intenta la Opción B (PyPI) en la siguiente celda.')
else:
    print(f'⚠️  pyproject.toml no encontrado en {PACKAGE_DIR_IN_DRIVE}')
    print('   1) Sube la carpeta voicelegacy a Drive en esa ruta, O')
    print('   2) Usa la siguiente celda (PyPI), O')
    print('   3) Cambia PACKAGE_DIR_IN_DRIVE en la celda 2.')
""")
)

CELLS.append(
    code_cell("""# ── Opción B: instalar desde PyPI (PRODUCCIÓN) ──────────────────
# Descomenta SI no tienes el paquete en Drive. Después: Runtime → Restart runtime.

# %pip install -U voicelegacy --quiet
# print('✅ voicelegacy instalado/actualizado desde PyPI. Reinicia el runtime ahora.')
""")
)

# ─── Cell 5: imports + version check ────────────────────────────────
CELLS.append(
    md_cell("""---

## 5️⃣ Validar import y versiones
""")
)

CELLS.append(
    code_cell("""import voicelegacy
from voicelegacy import (
    PipelineConfig, ReferenceConfig, SynthesisConfig, WorkspacePaths,
    run_reference_phase, run_synthesis, run_batch_synthesis,
    configure_logging, release_model,
)

configure_logging(level='INFO')
print(f'✅ voicelegacy v{voicelegacy.__version__}')

# Validar coqui-tts
try:
    import TTS
    print(f'✅ coqui-tts importado (TTS v{getattr(TTS, \"__version__\", \"?\")})')
except ImportError:
    print('❌ coqui-tts NO está disponible. Re-corre la celda 4 e intenta de nuevo.')
    raise

# Validar torch + GPU
import torch
print(f'✅ PyTorch {torch.__version__}, CUDA disponible: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'   Device: {torch.cuda.get_device_name(0)}')
    print(f'   VRAM  : {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB')
""")
)

# ─── Cell 6: build config + paths ──────────────────────────────────
CELLS.append(
    md_cell("""---

## 6️⃣ Armar el `PipelineConfig` y crear directorios
""")
)

CELLS.append(
    code_cell("""paths = WorkspacePaths(workspace=WORKSPACE)
paths.mkdirs()

config = PipelineConfig(
    reference=ReferenceConfig(
        target_speaker_label=TARGET_SPEAKER,
        top_n_segments=TOP_N_SEGMENTS,
        min_segment_duration_s=MIN_SEGMENT_DURATION_S,
        max_segment_duration_s=MAX_SEGMENT_DURATION_S,
        min_snr_db=MIN_SNR_DB,
        apply_denoise=APPLY_DENOISE,
        denoise_stationary=DENOISE_STATIONARY,
        apply_bandpass_filter=APPLY_BANDPASS_FILTER,
        apply_preemphasis_filter=APPLY_PREEMPHASIS_FILTER,
        enable_f0_outlier_filter=ENABLE_F0_OUTLIER_FILTER,
    ),
    synthesis=SynthesisConfig(
        language=LANGUAGE,
        temperature=TEMPERATURE,
        speed=SPEED,
        seed=SEED,
        compute_similarity=COMPUTE_SIMILARITY,
        long_text_strategy=LONG_TEXT_STRATEGY,
        max_single_pass_chars=MAX_SINGLE_PASS_CHARS,
        long_text_warning_chars=LONG_TEXT_WARNING_CHARS,
    ),
    accept_coqui_tos=ACCEPT_CPML,
    force_rebuild_reference=FORCE_REBUILD_REFERENCE,
    force_resynthesize=FORCE_RESYNTHESIZE,
)

print(f'📁 Workspace      : {paths.workspace}')
print(f'   Entrevistas    : {paths.interviews_raw}')
print(f'   speakerscribe  : {paths.speakerscribe_out}')
print(f'   ref_corpus     : {paths.reference_corpus}')
print(f'   synth_out      : {paths.synthesis_out}')
print(f'   reports        : {paths.reports}')
print(f'   runs.db        : {paths.db_path}')
print()
print(f'🎯 Speaker objetivo: {config.reference.target_speaker_label}')
print(f'🔢 Top N           : {config.reference.top_n_segments}')
print(f'🌐 Idioma          : {config.synthesis.language}')
print(f'🎲 Seed            : {config.synthesis.seed}')
print(f'🧪 Similarity      : {config.synthesis.compute_similarity}')
print(f'🧹 Denoise modo    : {"stationary" if config.reference.denoise_stationary else "non-stationary"}')
print(f'🎚️  Bandpass       : {config.reference.apply_bandpass_filter}')
print(f'📏 Long text       : strategy={config.synthesis.long_text_strategy}  '
      f'max_single_pass={config.synthesis.max_single_pass_chars}')

# Sanity: chequear que haya JSONs de speakerscribe
json_files = list(paths.speakerscribe_out.glob('*.json'))
if not json_files:
    print()
    print(f'⚠️  No hay archivos .json en {paths.speakerscribe_out}.')
    print('   Procesa las entrevistas con speakerscribe y copia los .json ahí.')
else:
    print(f'\\n✅ {len(json_files)} archivo(s) .json encontrados:')
    for jf in json_files[:5]:
        print(f'   • {jf.name}')
    if len(json_files) > 5:
        print(f'   ... y {len(json_files) - 5} más')
""")
)

# ─── Cell 6b: diagnose workspace (P2-26 surfaced in notebook) ─────────
CELLS.append(
    md_cell("""---

## 6️⃣b Diagnóstico del workspace y runtime

Antes del smoke test, valida que el workspace está bien armado y que el runtime tiene lo mínimo (Python ≥3.10, ffmpeg, CUDA, paquetes de Python, JSONs de speakerscribe válidos, etc.). Si esto sale con un solo `FAIL`, no continúes — corrige primero.

Equivalente CLI: `voicelegacy diagnose --workspace <ws> --json`.
""")
)

CELLS.append(
    code_cell("""from voicelegacy.diagnose import diagnose_workspace

report = diagnose_workspace(paths.workspace, synthesis_config=config.synthesis)
print(f'{"Check":<28} {"Status":<6}  Detail')
print('─' * 100)
for c in report.checks:
    icon = {'ok': '✅', 'warn': '⚠️ ', 'fail': '❌'}[c.status]
    detail = c.detail if len(c.detail) <= 60 else c.detail[:57] + '...'
    print(f'{c.name:<28} {icon} {c.status:<4}  {detail}')
    if c.remediation:
        print(f'{"":<28}        → {c.remediation}')

print()
print(f'ready={report.ready}  failures={report.failed}  warnings={report.warnings}')
if not report.ready:
    raise SystemExit('Workspace o runtime no están listos. Mira los FAIL arriba.')
""")
)

# ─── Cell 7: smoke test ─────────────────────────────────────────────
CELLS.append(
    md_cell("""---

## 7️⃣ Smoke test — verificar que XTTS-v2 carga y genera

Antes del run real, sintetiza una frase corta con un archivo de referencia mínimo. Confirma que:
- coqui-tts descarga los pesos (la primera vez baja ~2 GB).
- La GPU funciona.
- La librería completa el ciclo sin errores.

Si esto falla, **no pases a la fase 1**.
""")
)

CELLS.append(
    code_cell("""%%time
import numpy as np, soundfile as sf
from voicelegacy.synthesis import load_xtts_model, synthesize_to_file

if RUN_SMOKE_TEST:
    # Audio sintético de 8s — solo para probar el pipeline mecánicamente.
    smoke_dir = paths.reports / '_smoke'
    smoke_dir.mkdir(exist_ok=True)
    smoke_ref = smoke_dir / 'fake_ref.wav'

    sr = 22050
    t = np.arange(int(sr * 8)) / sr
    y = 0.30 * np.sin(2 * np.pi * 220 * t) + 0.15 * np.sin(2 * np.pi * 440 * t)
    y *= 0.5 + 0.5 * np.sin(2 * np.pi * 3 * t)  # envelope
    sf.write(str(smoke_ref), y.astype(np.float32), sr)

    print('⏬ Cargando XTTS-v2 (primera vez descarga ~2 GB)...')
    tts = load_xtts_model(config.synthesis, accept_tos=config.accept_coqui_tos)

    smoke_out = smoke_dir / 'smoke_test.wav'
    synthesize_to_file(
        tts=tts,
        text='Esto es una prueba de funcionamiento del pipeline.',
        speaker_wav=smoke_ref,
        output_path=smoke_out,
        config=config.synthesis,
    )
    print(f'\\n✅ Smoke test OK — output: {smoke_out}')
    print('   (Sonará raro: la referencia es sintética. Era solo prueba mecánica.)')
else:
    print('⏭️  Smoke test omitido (RUN_SMOKE_TEST=False)')
""")
)

# ─── Cell 8: phase 1 — build corpus ────────────────────────────────
CELLS.append(
    md_cell("""---

## 8️⃣ Fase 1: Construir el corpus de referencia

Lee los `.json` de `speakerscribe_out/`, extrae los segmentos del speaker objetivo, los limpia (denoise → trim → normalize), los scorea y guarda los mejores en `reference_corpus/`.

Idempotente: si ya hay WAVs en `reference_corpus/`, los reusa (salvo `FORCE_REBUILD_REFERENCE=True`).

**Conversión automática mp4/m4a/mkv/webm/...**: `run_reference_phase` ahora invoca ffmpeg internamente sobre `interviews_raw/` antes de buildear. No necesitas hacerlo aparte. Si prefieres correrlo manualmente desde la terminal, equivalente CLI: `voicelegacy convert-audio --workspace <ws>`.

**Verificar etiquetas de speaker antes de ejecutar**: si no sabes qué label usar en `TARGET_SPEAKER`, desde shell: `voicelegacy list-speakers --workspace <ws>`. Te muestra todos los SPEAKER_xx detectados, sus conteos de segmentos y duraciones acumuladas.
""")
)

CELLS.append(
    code_cell("""%%time
corpus_result = run_reference_phase(paths, config)

print()
print('📊 Resumen fase 1:')
print(f'   Candidatos totales : {len(corpus_result.all_wavs)}')
print(f'   Aprobados (pass)   : {sum(1 for r in corpus_result.reports if r.passed)}')
print(f'   Top seleccionados  : {len(corpus_result.top_wavs)}')

if corpus_result.top_wavs:
    print()
    print('🏆 Top segmentos por score:')
    top_reports = sorted(
        [r for r in corpus_result.reports if r.passed],
        key=lambda r: r.score,
        reverse=True,
    )
    for r in top_reports[:TOP_N_SEGMENTS]:
        print(f'   • score={r.score:.3f}  dur={r.stats.duration_s:5.2f}s  '
              f'snr={r.stats.snr_db:5.1f}dB  {r.path.name}')
else:
    print()
    print('❌ No se construyó corpus. Causas más probables:')
    print('   1. ¿Hay .json en speakerscribe_out/? (re-corre la celda 6)')
    print('   2. ¿El TARGET_SPEAKER coincide con la etiqueta real en los JSON?')
    print('      Abre uno y verifica.')
    print('   3. ¿El audio fuente es de llamada (≤16 kHz)? Está rechazado por diseño.')
    print('   4. ¿Las grabaciones son muy cortas o muy ruidosas? Relaja MIN_SNR_DB o')
    print('      MIN_SEGMENT_DURATION_S en la celda 2 — entendiendo el costo de calidad.')
    raise SystemExit('Fase 1 sin output. No continúo a fase 2.')
""")
)

# ─── Cell 9: phase 1 — listen to references ─────────────────────────
CELLS.append(
    md_cell("""---

## 9️⃣ Escuchar las referencias top (control de calidad humano)

Antes de sintetizar, **escucha** los segmentos elegidos. Si alguno suena raro (eco, otra voz, palabras cortadas), elimínalo manualmente o ajusta los filtros y re-corre la fase 1.
""")
)

CELLS.append(
    code_cell("""from IPython.display import Audio, display
import json

# Mostrar metadata + reproductor por cada top WAV
for i, wav_path in enumerate(corpus_result.top_wavs, 1):
    print(f'\\n[{i}/{len(corpus_result.top_wavs)}] {wav_path.name}')
    display(Audio(str(wav_path)))

# Imprimir ruta al reporte más reciente
reports = sorted(paths.reports.glob('reference_quality_*.json'), reverse=True)
if reports:
    print(f'\\n📋 Reporte JSON detallado: {reports[0]}')
    with open(reports[0]) as f:
        summary = json.load(f).get('summary', {})
    print(f'   Resumen: {summary}')
""")
)

# ─── Cell 9b: per-segment tabular report (replaces hand-edited Cell 22) ─
CELLS.append(
    md_cell("""---

### 🔍 Reporte por segmento (qué pasó la calidad y qué no)

Útil para entender por qué un segmento fue rechazado: SNR insuficiente, duración fuera de rango, sample rate de teléfono, etc. Si la mayoría falla por la misma razón, es un problema sistémico de la fuente, no de la curaduría.
""")
)

CELLS.append(
    code_cell("""import json

reportes = sorted(paths.reports.glob('reference_quality_*.json'), reverse=True)
if not reportes:
    print('⚠️  No hay reportes aún — corre la Fase 1 primero.')
else:
    reporte = json.loads(reportes[0].read_text(encoding='utf-8'))
    print(f'Reporte: {reportes[0].name}')
    print(f'Resumen: {reporte[\"summary\"]}')
    print()
    print(f'{\"#\":<4} {\"score\":>6} {\"dur(s)\":>7} {\"snr(dB)\":>8} {\"sr(Hz)\":>7} {\"pass\":>5}  razón de fallo')
    print('─' * 90)
    for i, seg in enumerate(reporte['all_segments'], 1):
        print(
            f'{i:<4} {seg[\"score\"]:>6.3f} {seg[\"duration_s\"]:>7.2f} '
            f'{seg[\"snr_db\"]:>8.1f} {seg[\"sample_rate\"]:>7} '
            f'{\"✅\" if seg[\"passed\"] else \"❌\":>5}  '
            f'{\", \".join(seg[\"reasons\"][:2]) if seg[\"reasons\"] else \"\"}'
        )
""")
)

# ─── Cell 10: phase 2 — synthesize ─────────────────────────────────
CELLS.append(
    md_cell("""---

## 🔟 Fase 2: Sintetizar los textos

Para cada texto en `TEXTS_TO_SYNTHESIZE`, genera un WAV con la voz clonada usando los top segmentos como condicionamiento.

Idempotente: si ya se sintetizó la misma combinación (texto + referencias + config), se reusa el archivo.
""")
)

CELLS.append(
    code_cell("""%%time
results = run_batch_synthesis(
    texts=TEXTS_TO_SYNTHESIZE,
    reference_wavs=corpus_result.top_wavs,
    paths=paths,
    config=config,
)

print()
print(f'📦 {len(results)} archivo(s) sintetizado(s):')
for r in results:
    marker = '♻️ ' if r.cached else '✨'
    sim = f' similarity={r.similarity_score:.3f}' if r.similarity_score is not None else ''
    meta = f' metadata={r.metadata_path.name}' if r.metadata_path else ''
    print(f'  {marker} {r.output_path}{sim}{meta}')
""")
)

# ─── Cell 11: listen to outputs ────────────────────────────────────
CELLS.append(
    md_cell("""---

## 1️⃣1️⃣ Escuchar resultados
""")
)

CELLS.append(
    code_cell("""from IPython.display import Audio, display

for r in results:
    print(f'\\n📝 {r.text[:100]}{\"...\" if len(r.text) > 100 else \"\"}')
    print(f'   {r.output_path.name} (cached={r.cached}, similarity={r.similarity_score})')
    if r.metadata_path and r.metadata_path.exists():
        print(f'   metadata: {r.metadata_path}')
    display(Audio(str(r.output_path)))
""")
)

# ─── Cell 11b: sidecar JSON detail (P1-18 surfaced in notebook) ────
CELLS.append(
    md_cell("""---

### 🧾 Auditoría del output (sidecar JSON)

Cada WAV sintetizado lleva su `.json` al lado, con:

- `speaker_similarity_score` y `similarity.quality_band` — qué tan parecido al objetivo (very_high / high / marginal / low).
- `source_quality` — calidad del corpus de referencia y si el output se generó en **modo degradado** (`degraded_mode=True` cuando `min_snr_db<10`, dinámica baja o sample rate <16 kHz).
- `text_plan` — si XTTS dividió el texto en frases o lo sintetizó de un solo paso. Útil para diagnosticar voice drift.
- `synthesis_config` + `reference_set_hash` + `run_hash` + `voicelegacy_version` — para regenerar el WAV idéntico si se pierde.

Si `quality_band` cae a `marginal` o `low` de forma sistemática, no es problema de UI: el material fuente probablemente no permite zero-shot. Considera `evaluate-denoise` (próxima celda) o fine-tuning.
""")
)

CELLS.append(
    code_cell("""import json

for r in results:
    if not (r.metadata_path and r.metadata_path.exists()):
        print(f'⚠️  Sin sidecar para {r.output_path.name}')
        continue
    meta = json.loads(r.metadata_path.read_text(encoding='utf-8'))
    print(f'\\n🎵 {r.output_path.name}')
    sim = meta.get('similarity', {})
    print(f'   similarity        : score={meta.get(\"speaker_similarity_score\")}  '
          f'band={sim.get(\"quality_band\")}  status={sim.get(\"status\")}')
    src = meta.get('source_quality', {})
    print(f'   source_quality    : n_ref={src.get(\"reference_count\")}  '
          f'mean_dyn_range_db={src.get(\"mean_dynamic_range_db\")}  '
          f'min_sr_hz={src.get(\"min_sample_rate_hz\")}')
    if src.get('degraded_mode'):
        print(f'   ⚠️  degraded_mode  : {src.get(\"degraded_reason\")}')
    plan = meta.get('text_plan', {})
    print(f'   text_plan         : strategy={plan.get(\"strategy\")}  '
          f'xtts_split_sentences={plan.get(\"xtts_split_sentences\")}  '
          f'len_chars={plan.get(\"length_chars\")}')
    if plan.get('warning'):
        print(f'   ⚠️  text warning   : {plan.get(\"warning\")}')
    print(f'   run_hash          : {meta.get(\"run_hash\", \"?\")[:16]}…  '
          f'voicelegacy_version={meta.get(\"voicelegacy_version\")}')
""")
)

# ─── Cell 11c: optional denoise evaluation (P3-1 surfaced in notebook) ─
CELLS.append(
    md_cell("""---

### 🧪 (Opcional) Comparar denoise alternativas — `evaluate-denoise`

Si el clon resulta mediocre por **fuente sub-óptima** (ruido, móvil, archivo antiguo), antes de cambiar defaults usa esta evaluación reproducible. Procesa 3-5 muestras representativas (limpio / ruido moderado / ruido pesado / phone-codec / entrevista larga) con `noisereduce` y opcionalmente con DeepFilterNet si lo instalaste (`pip install voicelegacy[deepfilter]`).

DeepFilterNet sólo debe convertirse en default si mejora la escucha humana **y** el `speaker_similarity_score` sin introducir artefactos. La celda guarda el reporte y los WAVs procesados en `reports/denoise_eval/`.
""")
)

CELLS.append(
    code_cell("""# Cambia RUN_DENOISE_EVAL a True si quieres ejecutar esta evaluación.
RUN_DENOISE_EVAL = False
INCLUDE_DEEPFILTER = False  # Requiere `pip install voicelegacy[deepfilter]` + deepFilter CLI

if RUN_DENOISE_EVAL:
    from voicelegacy.denoise_eval import evaluate_denoise_methods

    raw_files = sorted(p for p in paths.interviews_raw.iterdir()
                       if p.is_file() and p.suffix.lower() in {'.wav', '.mp3', '.m4a', '.flac', '.ogg'})[:5]
    if not raw_files:
        print('⚠️  No hay audio en interviews_raw/. Pon 3-5 muestras representativas primero.')
    else:
        report = evaluate_denoise_methods(
            raw_files,
            paths.reports / 'denoise_eval',
            include_deepfilter=INCLUDE_DEEPFILTER,
        )
        print(f'\\n📋 Reporte: {report[\"report_path\"]}')
        print(f'{\"source\":<35} {\"method\":<28} {\"status\":<8}  output')
        print('─' * 100)
        for row in report['candidates']:
            src = Path(str(row['source_path'])).name
            out = row.get('output_path') or row.get('reason') or ''
            out_str = str(out)[:40]
            print(f'{src:<35} {row[\"method\"]:<28} {row[\"status\"]:<8}  {out_str}')
else:
    print('Saltando evaluate-denoise. Cambia RUN_DENOISE_EVAL=True para activarlo.')
""")
)

# ─── Cell 12: copy deliverables ─────────────────────────────────────
CELLS.append(
    md_cell("""---

## 1️⃣2️⃣ Empaquetar entregables (opcional)

Copia los WAVs sintetizados + reportes de calidad a una carpeta `entregables/` lista para compartir.
""")
)

CELLS.append(
    code_cell("""import shutil
from datetime import datetime

ts = datetime.now().strftime('%Y%m%d_%H%M%S')
deliverables = paths.workspace / f'entregables_{ts}'
deliverables.mkdir(parents=True, exist_ok=True)

# Copiar audios
for r in results:
    if r.output_path.exists():
        shutil.copy2(r.output_path, deliverables / r.output_path.name)
    if r.metadata_path and r.metadata_path.exists():
        shutil.copy2(r.metadata_path, deliverables / r.metadata_path.name)

# Copiar reporte más reciente
latest_report = sorted(paths.reports.glob('reference_quality_*.json'), reverse=True)
if latest_report:
    shutil.copy2(latest_report[0], deliverables / latest_report[0].name)

# Metadata del run
import json
import sys

meta = {
    'timestamp': datetime.now().isoformat(),
    'voicelegacy_version': voicelegacy.__version__,
    'python_version': sys.version,
    'config': config.model_dump(mode='json'),
    'n_reference_segments': len(corpus_result.top_wavs),
    'n_synthesis_outputs': len(results),
    'texts': TEXTS_TO_SYNTHESIZE,
}
(deliverables / 'run_metadata.json').write_text(
    json.dumps(meta, indent=2, ensure_ascii=False),
    encoding='utf-8',
)

print(f'📦 Entregables en: {deliverables}')
for p in sorted(deliverables.iterdir()):
    size_kb = p.stat().st_size / 1024
    print(f'   • {p.name} ({size_kb:.1f} KB)')
""")
)

# ─── Cell 13: cleanup ──────────────────────────────────────────────
CELLS.append(
    md_cell("""---

## 1️⃣3️⃣ Liberar VRAM y limpiar estado
""")
)

CELLS.append(
    code_cell("""import gc

release_model()
gc.collect()
print('🧹 VRAM liberada, estado limpio.')

# ─────────────────────────────────────────────────────────────────────
# CIERRE DE SESIÓN (OPCIONAL)
#
# La línea de abajo libera el T4 al pool compartido. NO la descomentes
# durante uso normal — termina la sesión y mata cualquier trabajo no
# guardado en /content. Solo úsala cuando ya descargaste todos los
# artefactos a Drive y quieres ser cortés con la cuota.
#
#   from google.colab import runtime
#   runtime.unassign()
# ─────────────────────────────────────────────────────────────────────
""")
)

# ─── Cell 14: troubleshooting ──────────────────────────────────────
CELLS.append(
    md_cell("""---

## 🔧 Troubleshooting

| Síntoma | Causa probable | Acción |
|---|---|---|
| `❌ No se detectó GPU` | Runtime sin GPU | Runtime → Change runtime type → T4 GPU |
| `RuntimeError: You must accept the Coqui Public Model License` | `ACCEPT_CPML=False` | Lee https://coqui.ai/cpml, cambia a `True` en celda 2 |
| `ModuleNotFoundError: No module named 'voicelegacy'` | Paquete no instalado | Re-corre la celda 4 |
| `Reference corpus is empty` | No hay JSONs / mal speaker / audio rechazado | Ver mensajes de la celda 8; usa `voicelegacy list-speakers` para confirmar etiquetas |
| `diagnose` reporta `fail` en `dir:*` | Workspace incompleto | Crea la carpeta indicada o llama `paths.mkdirs()` |
| `diagnose` reporta `warn:cuda` | CPU-only (10× más lento) | Cambia a runtime con GPU si es posible |
| `diagnose` reporta `warn:raw_audio` o `warn:speakerscribe_json` | Sin material en interviews_raw/ o sin .json | Pon entrevistas y procesa con speakerscribe |
| `CUDA out of memory` | Otro modelo cargado | Reinicia runtime o llama `release_model()` |
| El clon suena "como en teléfono" | Audio fuente comprimido (≤16 kHz) | El gate rechaza ≤16 kHz. Re-graba con micrófono decente |
| El clon ignora la entonación | Referencias muy cortas | Aumenta `MAX_SEGMENT_DURATION_S` a 20, baja `MIN_SEGMENT_DURATION_S` a 6 |
| Voice drift entre oraciones | `LONG_TEXT_STRATEGY='coqui_split'` o texto > `MAX_SINGLE_PASS_CHARS` | Usa `'auto'` (default) y baja `MAX_SINGLE_PASS_CHARS`; o parte el texto manualmente |
| `similarity quality_band=marginal/low` sistemático | Fuente irrecuperable para zero-shot | Corre celda 11c (`evaluate-denoise`); evalúa DeepFilterNet o considera fine-tuning |
| Sidecar muestra `degraded_mode=True` | `min_snr_db<10` o sample rate `<16 kHz` | El output va con etiqueta honesta de modo degradado. Mejor fuente > mejor ajuste |

### Si quieres calidad superior a zero-shot

Necesitas **fine-tuning**, que requiere:
- ~1 hora de audio limpio + transcripciones en formato LJSpeech.
- 12–24 horas de GPU continua (RTX 4090 o equivalente).
- **No es viable en Colab Free**. Opciones: Colab Pro+ ($49.99/mes), RunPod ($0.34–0.79/h en RTX 4090), Lambda Labs.

### Comandos CLI complementarios

| Comando | Para qué |
|---|---|
| `voicelegacy diagnose --workspace <ws>` | Validar workspace y runtime |
| `voicelegacy convert-audio --workspace <ws>` | Convertir mp4/m4a/mkv/webm a WAV (`run_reference_phase` ya lo hace internamente) |
| `voicelegacy list-speakers --workspace <ws>` | Listar etiquetas SPEAKER_xx en los JSON de speakerscribe |
| `voicelegacy build-corpus --workspace <ws> --speaker SPEAKER_00 --accept-tos` | Fase 1 desde shell |
| `voicelegacy synthesize --workspace <ws> --text-file textos.txt --accept-tos` | Fase 2 batch desde shell |
| `voicelegacy evaluate-denoise --workspace <ws> [--deepfilter]` | Comparar denoise alternativas |
""")
)


def build_notebook() -> dict:
    return {
        "cells": CELLS,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {
                "name": "python",
                "version": "3.11",
            },
            "colab": {
                "provenance": [],
                "gpuType": "T4",
            },
            "accelerator": "GPU",
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


def _validate_pyproject_toml(root: Path) -> None:
    """Fail fast if pyproject.toml is unparseable.

    Even though build_notebook doesn't generate pyproject.toml today, running
    this check at build time gives early feedback in the same workflow:
    `python build_notebook.py && pip install -e .` won't surprise the user
    with a TOML error mid-install. See SKILL §18.5 for the underlying lesson.
    """
    try:
        # Python 3.11+
        import tomllib
    except ImportError:  # pragma: no cover — only on 3.10
        import tomli as tomllib  # type: ignore[no-redef]

    pyproject = root.parent / "pyproject.toml"
    if not pyproject.exists():
        print(f"⚠️  pyproject.toml not found at {pyproject} — skipping validation.")
        return
    try:
        with open(pyproject, "rb") as fh:
            tomllib.load(fh)
        print(f"✅ pyproject.toml is valid TOML ({pyproject.name})")
    except tomllib.TOMLDecodeError as exc:
        raise RuntimeError(
            f"pyproject.toml is invalid: {exc}. Fix it before regenerating "
            "the notebook — otherwise CI and `pip install -e .` will fail."
        ) from exc


def _validate_notebook(nb: dict) -> None:
    """Validate the in-memory notebook dict against the nbformat schema.

    Catches malformed cells, missing ids, wrong types — anything that would
    make Jupyter refuse to open the file later.
    """
    try:
        import nbformat
    except ImportError:
        print("⚠️  nbformat not installed — skipping schema validation.")
        print("   pip install nbformat to enable it.")
        return
    # nbformat.from_dict + validate raises ValidationError on schema breakage
    nbformat.validate(nbformat.from_dict(nb))
    print(f"✅ Notebook validates against nbformat v{nb['nbformat']}.{nb['nbformat_minor']}")


def _backup_if_hand_edited(out_path: Path, expected_cells: int) -> None:
    """If the destination notebook exists and has more cells than the generator
    produces, snapshot it to _archive/ before overwriting.

    Rationale: the regression in the original repo was that the canonical
    notebook had 37 cells while build_notebook.py only generated 29 — 7
    cells were hand-edited additions, some containing real fixes. Silently
    overwriting them is destructive. This guard preserves the editorial
    history so any hand-edit can be re-integrated into the generator on
    purpose, not lost on accident.
    """
    if not out_path.exists():
        return
    try:
        existing = json.loads(out_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        # Corrupt destination — overwrite is the safer outcome.
        return
    existing_cells = len(existing.get("cells", []))
    if existing_cells <= expected_cells:
        return

    archive_dir = out_path.parent / "_archive"
    archive_dir.mkdir(exist_ok=True)
    from datetime import datetime, timezone

    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    backup = archive_dir / f"{out_path.stem}_{ts}_handedited_{existing_cells}cells.ipynb"
    backup.write_text(json.dumps(existing, indent=1, ensure_ascii=False), encoding="utf-8")
    print(
        f"⚠️  Destination notebook has {existing_cells} cells, generator produces "
        f"{expected_cells}. Saved snapshot of hand-edited version to {backup}"
    )


if __name__ == "__main__":
    root = Path(__file__).parent
    out = root / "notebook_voicelegacy.ipynb"

    nb = build_notebook()
    _validate_pyproject_toml(root)
    _validate_notebook(nb)
    _backup_if_hand_edited(out, expected_cells=len(nb["cells"]))

    # Trailing newline keeps `end-of-file-fixer` happy across regenerations;
    # without it, every `python build_notebook.py` invalidates the file and
    # the pre-commit fixer rewrites it, producing churn in git history.
    out.write_text(json.dumps(nb, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"✅ Notebook written: {out}")
    print(f"   Cells: {len(nb['cells'])}")
