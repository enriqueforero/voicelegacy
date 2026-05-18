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
CELLS.append(md_cell("""# 🎙️ voicelegacy — Notebook de producción

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
"""))

# ─── Cell 1: mount Drive ────────────────────────────────────────────
CELLS.append(md_cell("""---

## 1️⃣ Montar Google Drive
"""))

CELLS.append(code_cell("""from google.colab import drive
drive.mount('/content/drive')
"""))

# ─── Cell 2: Configuration ─────────────────────────────────────────
CELLS.append(md_cell("""---

## 2️⃣ Configuración del run

**Esta es la única celda que debes editar.** Todo lo demás se autoconfigura.
"""))

CELLS.append(code_cell("""# ══════════════════════════════════════════════════════════════════
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
TARGET_SPEAKER = 'SPEAKER_00'

# ── Selección de corpus ──────────────────────────────────────────
TOP_N_SEGMENTS = 10            # Top N segmentos limpios para usar como referencia
MIN_SEGMENT_DURATION_S = 4.0   # Descartar segmentos más cortos
MAX_SEGMENT_DURATION_S = 15.0  # Descartar segmentos más largos
MIN_SNR_DB = 15.0              # SNR mínimo aceptable
APPLY_DENOISE = True           # Reducción de ruido espectral

# ── Síntesis ──────────────────────────────────────────────────────
LANGUAGE = 'es'                # 'es', 'en', 'pt', 'it', 'fr', ...
TEMPERATURE = 0.7              # 0.5-0.9 — estabilidad vs expresividad
SPEED = 1.0                    # 0.8 = más lento, 1.2 = más rápido

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
"""))

# ─── Cell 3: pre-flight ─────────────────────────────────────────────
CELLS.append(md_cell("""---

## 3️⃣ Pre-flight: verificar entorno (GPU, RAM, versiones)
"""))

CELLS.append(code_cell("""import sys, platform, subprocess
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
"""))

# ─── Cell 4: install package ────────────────────────────────────────
CELLS.append(md_cell("""---

## 4️⃣ Instalar `voicelegacy` y dependencias

**OPCIÓN A — desarrollo**: instala desde Drive en modo editable.
**OPCIÓN B — producción**: instala desde PyPI (cuando esté publicado).

Después de instalar, **reinicia el runtime** (`Runtime → Restart runtime`) la primera vez.
"""))

CELLS.append(code_cell("""%%time
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
"""))

CELLS.append(code_cell("""# ── Opción B: instalar desde PyPI (PRODUCCIÓN) ──────────────────
# Descomenta SI no tienes el paquete en Drive. Después: Runtime → Restart runtime.

# %pip install -U voicelegacy --quiet
# print('✅ voicelegacy instalado/actualizado desde PyPI. Reinicia el runtime ahora.')
"""))

# ─── Cell 5: imports + version check ────────────────────────────────
CELLS.append(md_cell("""---

## 5️⃣ Validar import y versiones
"""))

CELLS.append(code_cell("""import voicelegacy
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
"""))

# ─── Cell 6: build config + paths ──────────────────────────────────
CELLS.append(md_cell("""---

## 6️⃣ Armar el `PipelineConfig` y crear directorios
"""))

CELLS.append(code_cell("""paths = WorkspacePaths(workspace=WORKSPACE)
paths.mkdirs()

config = PipelineConfig(
    reference=ReferenceConfig(
        target_speaker_label=TARGET_SPEAKER,
        top_n_segments=TOP_N_SEGMENTS,
        min_segment_duration_s=MIN_SEGMENT_DURATION_S,
        max_segment_duration_s=MAX_SEGMENT_DURATION_S,
        min_snr_db=MIN_SNR_DB,
        apply_denoise=APPLY_DENOISE,
    ),
    synthesis=SynthesisConfig(
        language=LANGUAGE,
        temperature=TEMPERATURE,
        speed=SPEED,
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
"""))

# ─── Cell 7: smoke test ─────────────────────────────────────────────
CELLS.append(md_cell("""---

## 7️⃣ Smoke test — verificar que XTTS-v2 carga y genera

Antes del run real, sintetiza una frase corta con un archivo de referencia mínimo. Confirma que:
- coqui-tts descarga los pesos (la primera vez baja ~2 GB).
- La GPU funciona.
- La librería completa el ciclo sin errores.

Si esto falla, **no pases a la fase 1**.
"""))

CELLS.append(code_cell("""%%time
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
"""))

# ─── Cell 8: phase 1 — build corpus ────────────────────────────────
CELLS.append(md_cell("""---

## 8️⃣ Fase 1: Construir el corpus de referencia

Lee los `.json` de `speakerscribe_out/`, extrae los segmentos del speaker objetivo, los limpia (denoise → trim → normalize), los scorea y guarda los mejores en `reference_corpus/`.

Idempotente: si ya hay WAVs en `reference_corpus/`, los reusa (salvo `FORCE_REBUILD_REFERENCE=True`).
"""))

CELLS.append(code_cell("""%%time
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
"""))

# ─── Cell 9: phase 1 — listen to references ─────────────────────────
CELLS.append(md_cell("""---

## 9️⃣ Escuchar las referencias top (control de calidad humano)

Antes de sintetizar, **escucha** los segmentos elegidos. Si alguno suena raro (eco, otra voz, palabras cortadas), elimínalo manualmente o ajusta los filtros y re-corre la fase 1.
"""))

CELLS.append(code_cell("""from IPython.display import Audio, display
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
"""))

# ─── Cell 10: phase 2 — synthesize ─────────────────────────────────
CELLS.append(md_cell("""---

## 🔟 Fase 2: Sintetizar los textos

Para cada texto en `TEXTS_TO_SYNTHESIZE`, genera un WAV con la voz clonada usando los top segmentos como condicionamiento.

Idempotente: si ya se sintetizó la misma combinación (texto + referencias + config), se reusa el archivo.
"""))

CELLS.append(code_cell("""%%time
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
    print(f'  {marker} {r.output_path}')
"""))

# ─── Cell 11: listen to outputs ────────────────────────────────────
CELLS.append(md_cell("""---

## 1️⃣1️⃣ Escuchar resultados
"""))

CELLS.append(code_cell("""from IPython.display import Audio, display

for r in results:
    print(f'\\n📝 {r.text[:100]}{\"...\" if len(r.text) > 100 else \"\"}')
    print(f'   {r.output_path.name} (cached={r.cached})')
    display(Audio(str(r.output_path)))
"""))

# ─── Cell 12: copy deliverables ─────────────────────────────────────
CELLS.append(md_cell("""---

## 1️⃣2️⃣ Empaquetar entregables (opcional)

Copia los WAVs sintetizados + reportes de calidad a una carpeta `entregables/` lista para compartir.
"""))

CELLS.append(code_cell("""import shutil
from datetime import datetime

ts = datetime.now().strftime('%Y%m%d_%H%M%S')
deliverables = paths.workspace / f'entregables_{ts}'
deliverables.mkdir(parents=True, exist_ok=True)

# Copiar audios
for r in results:
    if r.output_path.exists():
        shutil.copy2(r.output_path, deliverables / r.output_path.name)

# Copiar reporte más reciente
latest_report = sorted(paths.reports.glob('reference_quality_*.json'), reverse=True)
if latest_report:
    shutil.copy2(latest_report[0], deliverables / latest_report[0].name)

# Metadata del run
import json, sys
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
"""))

# ─── Cell 13: cleanup ──────────────────────────────────────────────
CELLS.append(md_cell("""---

## 1️⃣3️⃣ Liberar VRAM y limpiar estado
"""))

CELLS.append(code_cell("""import gc

release_model()
gc.collect()
print('🧹 VRAM liberada, estado limpio.')

# Para terminar la sesión de Colab (libera GPU del pool compartido):
# from google.colab import runtime
# runtime.unassign()
"""))

# ─── Cell 14: troubleshooting ──────────────────────────────────────
CELLS.append(md_cell("""---

## 🔧 Troubleshooting

| Síntoma | Causa probable | Acción |
|---|---|---|
| `❌ No se detectó GPU` | Runtime sin GPU | Runtime → Change runtime type → T4 GPU |
| `RuntimeError: You must accept the Coqui Public Model License` | `ACCEPT_CPML=False` | Lee https://coqui.ai/cpml, cambia a `True` en celda 2 |
| `ModuleNotFoundError: No module named 'voicelegacy'` | Paquete no instalado | Re-corre la celda 4 |
| `Reference corpus is empty` | No hay JSONs / mal speaker / audio rechazado | Ver mensajes de la celda 8 |
| `CUDA out of memory` | Otro modelo cargado | Reinicia runtime |
| El clon suena "como en teléfono" | Audio fuente comprimido | Re-graba con micrófono decente; las llamadas no sirven |
| El clon ignora la entonación | Referencias muy cortas | Aumenta `MAX_SEGMENT_DURATION_S` a 20, baja `MIN_SEGMENT_DURATION_S` a 6 |
| Voice drift en textos largos | Texto > 250 caracteres sin pausas | `enable_text_splitting=True` (default); o divide el texto en oraciones cortas |

### Si quieres calidad superior a zero-shot

Necesitas **fine-tuning**, que requiere:
- ~1 hora de audio limpio + transcripciones en formato LJSpeech.
- 12–24 horas de GPU continua (RTX 4090 o equivalente).
- **No es viable en Colab Free**. Opciones: Colab Pro+ ($49.99/mes), RunPod ($0.34–0.79/h en RTX 4090), Lambda Labs.
"""))


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


if __name__ == "__main__":
    out = Path(__file__).parent / "notebook_voicelegacy.ipynb"
    nb = build_notebook()
    out.write_text(json.dumps(nb, indent=1, ensure_ascii=False), encoding="utf-8")
    print(f"✅ Notebook written: {out}")
    print(f"   Cells: {len(nb['cells'])}")
