# PLAYBOOK · Clonación de voz con fine-tuning desde una grabación de 30 min

Guía completa, paso a paso, para entrenar XTTS-v2 con una sola grabación larga
cuando el zero-shot te falló. Diseñado para **Google Colab Free Tier (T4 GPU)**.

---

## 0 · Antes de empezar — la verdad que debes aceptar

**30 minutos es el límite inferior, no el ideal.**

- El sweet spot del fine-tuning XTTS-v2 es **2-5 horas** de voz neta.
- Una grabación de 30 min cruda ≈ **15-22 min de voz neta** tras quitar silencios, respiraciones y ruido.
- Resultado realista: **"reconocible y aceptable para legado familiar"**, NO "indistinguible del original".

**Por qué el zero-shot te falló con clips de 5-15 s:** XTTS-v2 zero-shot clona de
6 segundos de referencia, pero el encoder necesita *contexto prosódico*. Clips
cortos y aislados dan poco contexto; el modelo captura el timbre pero pierde la
entonación. Una grabación continua de 30 min da muchísimo más contexto —
por eso el fine-tuning con tu grabación larga **probablemente mejore**, aunque
no llegue a perfección.

**Regla de cierre:** si al terminar (celda 10 del notebook) el modelo fine-tuned
NO supera al base por **≥ +0.05** en `speaker_similarity_score`, tu material es
insuficiente. Ningún ajuste de hyperparams lo arregla. Necesitas más audio.

---

## 1 · Qué notebook usar

Tienes **tres** notebooks en `notebooks/`. Para tu caso (UNA grabación de 30 min,
sin transcripción previa) usa el **standalone**:

| Notebook | Cuándo | Tu caso |
|---|---|---|
| `notebook_voicelegacy.ipynb` | Inferencia zero-shot con corpus ya construido | Ya lo probaste, salió mal |
| `notebook_voicelegacy_finetune.ipynb` | Fine-tune con `reference_corpus/` + speakerscribe JSON | ❌ No tienes speakerscribe |
| **`notebook_voicelegacy_finetune_standalone.ipynb`** | **Fine-tune desde UNA grabación cruda** | ✅ **ESTE** |

El standalone hace TODO: transcribe (faster-whisper), segmenta, limpia, arma
dataset, entrena, valida. No necesita nada más que tu archivo de audio.

---

## 2 · Preparación del material (antes de abrir Colab)

### 2.1 La grabación

- Formato: cualquiera (mp3, m4a, wav, mp4, ogg, flac). El notebook convierte.
- Duración: 30 min está bien. Más es mejor.
- **Calidad importa más que cantidad.** Una grabación de 20 min limpia supera a 40 min con ruido de fondo constante.

### 2.2 ¿Hay una o varias voces?

**Crítico.** faster-whisper transcribe pero NO separa hablantes.

- **Si es monólogo** (solo la persona objetivo habla): perfecto, deja `ES_MONOLOGO = True`.
- **Si es entrevista** (entrevistador + entrevistado): pon `ES_MONOLOGO = False`. Al terminar la transcripción (celda 5), tendrás que **editar manualmente** el CSV para borrar las filas del entrevistador. El notebook te avisa cuándo.

### 2.3 Subir a Drive

Sube la grabación a tu Google Drive, por ejemplo a:
```
MyDrive/grabacion_abuela_30min.m4a
```
Anota la ruta exacta — la necesitas en la celda 2.

---

## 3 · Configuración de Colab

1. Abrir https://colab.research.google.com → File → Upload notebook → subir `notebook_voicelegacy_finetune_standalone.ipynb`.
2. **Runtime → Change runtime type → Hardware accelerator: T4 GPU → Save.** (NO TPU — XTTS-v2 no soporta TPU.)
3. Confirmar GPU: la celda 1 verifica que haya T4. Si dice "NINGUNA", el runtime no tiene GPU.

---

## 4 · Ejecución paso a paso

### Celda 1 — Aceptar CPML + Drive + GPU
- Cambia `ACCEPT_CPML = False` → `True` (tras leer https://coqui.ai/cpml).
- Monta Drive (autoriza el popup).
- Verifica T4.

### Celda 2 — Configuración
Edita SOLO estas líneas:
```python
RAW_RECORDING = Path("/content/drive/MyDrive/grabacion_abuela_30min.m4a")  # tu archivo
WORKSPACE_FT  = Path("/content/drive/MyDrive/voicelegacy_finetune_standalone")
RUN_NAME      = "abuela_es_v1"
LANGUAGE      = "es"
ES_MONOLOGO   = True   # False si hay varias voces
```
**No toques los hyperparams** — ya están calibrados para dataset pequeño:
`NUM_EPOCHS=10`, `LEARNING_RATE=3e-6`, `WEIGHT_DECAY=5e-2`.

### Celda 3 — Instalar dependencias
- Tarda 2-3 min. Ignora los errores rojos de `pip dependency resolver`.
- Verifica que los 5 imports muestren ✅.

### Celda 4 — Convertir a 22.05 kHz mono
- ffmpeg convierte tu archivo. ~30 s.
- Verifica que la duración reportada sea la esperada (~30 min).

### Celda 5 — Transcribir + segmentar (LA MÁS IMPORTANTE)
- faster-whisper `large-v3` transcribe. **5-10 min para 30 min de audio.**
- Segmenta en clips de 2-11 s cortando en pausas naturales.
- Limpia cada clip (denoise + bandpass + loudness).
- **Lee la salida:**
  - `Segmentos guardados: N` — quieres N ≥ 150. Si N < 50, hay un warning.
  - `Voz neta total: X min` — quieres X ≥ 12 min. Si X < 12, warning crítico.

**Si `ES_MONOLOGO = False`:** ahora abre el explorador de archivos de Colab
(ícono de carpeta a la izquierda), navega a
`WORKSPACE_FT/dataset/metadata_train.csv`, descárgalo, borra las filas del
hablante que NO quieres (las que tengan texto del entrevistador), súbelo de
vuelta. Repite con `metadata_eval.csv`.

### Celda 6 — Descargar base XTTS-v2
- ~2 GB, 5-10 min. Solo la primera vez (queda cacheado en Drive).

### Celda 7 — Configurar trainer
- Instantáneo. Muestra el resumen de epochs/batch.

### Celda 8 — ENTRENAR (la larga)
- **2-4 horas.** Aquí es donde Colab puede cortarte.
- Verás el loss bajando cada 50 pasos.
- Si ves `CUDA out of memory`: baja `BATCH_SIZE` a 2 en celda 2, sube `GRAD_ACUMM_STEPS` a 126, reinicia runtime (Runtime → Restart), re-corre desde celda 1.

### Celda 8.bis — Reanudar (SOLO si Colab cortó)
- Si la sesión murió a mitad de entrenamiento: vuelve, monta Drive (celda 1), salta DIRECTO a 8.bis.
- Reanuda desde el último checkpoint intermedio (guardado cada 500 pasos).
- **NO pierdes la transcripción ni el dataset** — están en Drive.

### Celda 9 — Materializar checkpoint
- Copia `best_model.pth` + los 5 archivos base a `finetuned/`.
- Instantáneo.

### Celda 10 — Validar + A/B (EL VEREDICTO)
- Sintetiza la misma frase con BASE y FINE-TUNED.
- Muestra `ΔSim`:
  - `≥ +0.05` → ✅ **Fine-tuning aportó. Usa el checkpoint.**
  - `+0.00 a +0.05` → ⚠ Marginal. Escucha A/B. Probablemente necesitas más audio.
  - `< 0.00` → ❌ **Empeoró. 30 min fue insuficiente.** Consigue más material.
- **Escucha ambos audios.** La métrica coseno no mide naturalidad; tu oído sí.

### Celda 11 — Usar en producción
- Código copy-paste para generar cualquier frase nueva con el checkpoint.

---

## 5 · Qué se guarda (sobrevive a sesión muerta)

Todo en Drive bajo `WORKSPACE_FT/`:

| Archivo/carpeta | Contenido | Se reprocesa si... |
|---|---|---|
| `source_22k_mono.wav` | grabación convertida | borras el archivo |
| `transcript_words.json` | transcripción con timestamps | borras el archivo |
| `dataset/wavs/` | segmentos limpios | borras la carpeta |
| `dataset/metadata_*.csv` | pares (wav, texto) | re-corres celda 5 |
| `base_xtts/` | modelo base (~2 GB) | borras la carpeta |
| `runs/<run>/checkpoint_*.pth` | checkpoints intermedios | nunca (para reanudar) |
| `finetuned/` | checkpoint final (~2 GB) | re-corres celda 9 |

**Si Colab te corta:** vuelve, celda 1 (Drive), salta a 8.bis. La transcripción
(10 min) y la descarga del base (10 min) NO se repiten.

---

## 6 · Solución de problemas

| Síntoma | Causa | Solución |
|---|---|---|
| Celda 1: "Sin GPU" | Runtime sin T4 | Runtime → Change runtime type → T4 GPU |
| Celda 5: `Voz neta < 12 min` | Grabación con mucho silencio/ruido, o demasiado corta | Consigue más audio. 30 min con 50% silencio = solo 15 min útil |
| Celda 5: pocos segmentos | VAD agresivo cortó demasiado | Baja `min_silence_duration_ms` a 300 en la celda 5 |
| Celda 5: transcripción con errores | Audio ruidoso confunde a Whisper | Limpia el audio fuente antes (Audacity), o acepta WER alto |
| Celda 8: `CUDA out of memory` | Batch muy grande para T4 | `BATCH_SIZE=2`, `GRAD_ACUMM_STEPS=126`, reinicia |
| Celda 8: sesión cortada | Quota Colab Free | Reabre, celda 1, salta a 8.bis |
| Celda 10: `ΔSim < 0` | Material insuficiente / transcripción mala | Más audio (objetivo: 1+ hora). Verifica que las transcripciones del CSV sean correctas |
| Celda 10: suena robótico | Overfit (muy pocos datos, muchos epochs) | Baja `NUM_EPOCHS` a 6, re-entrena |
| Celda 10: suena como otra persona | `ES_MONOLOGO=False` y no limpiaste el CSV | Borra filas del otro hablante en los CSV, re-corre desde celda 7 |

---

## 7 · Cómo mejorar si 30 min no alcanzó

Si la celda 10 dio `ΔSim < +0.05`, en orden de impacto:

1. **Consigue más audio del mismo hablante.** Es lo único que de verdad mueve la aguja. Objetivo: 1-2 horas. Puedes combinar varias grabaciones — sube todas, conviértelas, y concaténalas antes de la celda 5.
2. **Verifica las transcripciones.** Abre `metadata_train.csv` y lee 10 filas al azar. Si Whisper transcribió mal (WER alto), el modelo aprende texto-audio desalineado. Re-transcribe con audio más limpio.
3. **Limpia el audio fuente.** Si hay ruido de fondo constante (ventilador, tráfico), pásalo por un denoiser dedicado (Audacity → Noise Reduction) ANTES de subirlo.
4. **Reduce epochs si sobre-ajusta.** Si suena exagerado/caricaturesco, es overfit: baja `NUM_EPOCHS` a 6.

**NO pierdas tiempo** ajustando learning rate o batch size si el problema es
cantidad de datos. 15 min de voz neta tiene un techo de calidad que los
hyperparams no rompen.

---

## 8 · Checklist de cierre del proyecto

Cuando el fine-tuning funcione (ΔSim ≥ +0.05):

- [ ] Checkpoint en `finetuned/` validado con la celda 9.
- [ ] A/B escuchado: el fine-tuned suena mejor que el base a tu oído.
- [ ] `speaker_similarity_score` documentado (anótalo).
- [ ] Generaste 3-5 frases de prueba con la celda 11 y suenan aceptables.
- [ ] Backup del `finetuned/` a otra carpeta de Drive (es ~2 GB, no lo pierdas).

Con esto, la librería está **probada y cerrada para tu caso de uso**: tienes un
checkpoint reutilizable que produce la voz de la persona en calidad de legado.

---

## URLs de referencia

- Notebook oficial Coqui (base del enfoque): https://colab.research.google.com/drive/1GiI4_X724M8q2W-zZ-jXo7cWTV7RfaH-
- Docs XTTS-v2 training: https://docs.coqui.ai/en/latest/models/xtts.html#training
- faster-whisper: https://github.com/SYSTRAN/faster-whisper
- coqui-tts (fork mantenido): https://pypi.org/project/coqui-tts/
- CPML: https://coqui.ai/cpml
