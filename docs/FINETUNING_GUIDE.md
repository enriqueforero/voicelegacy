# Guía de fine-tuning · voicelegacy 0.3.0+

Cuándo y cómo usar `voicelegacy.finetuned_inference` y el notebook
`notebook_voicelegacy_finetune.ipynb`.

---

## ¿Cuándo conviene fine-tunear?

**Fine-tunea sólo si TODAS estas condiciones se cumplen.** Si alguna falla,
quédate con zero-shot — XTTS-v2 base ya es bueno.

| Condición | Por qué importa |
|---|---|
| Tienes **2-5 horas** de audio limpio del speaker objetivo | < 30 min ≈ overfit garantizado; > 5 h ≈ rendimientos decrecientes |
| Las transcripciones de speakerscribe están correctas (WER < 5%) | XTTS aprende texto + audio juntos; transcripciones malas envenenan el modelo |
| Ya probaste zero-shot y `speaker_similarity_score` cae en banda `marginal` (0.60-0.75) sistemáticamente | Si ya estás en `high` (≥0.75), fine-tuning es over-engineering |
| Aceptas que el checkpoint pesa ~2 GB en Drive (15 GB Free) | Cada experimento consume espacio |
| Tienes 3-5 h disponibles para una sesión Colab | El training toma 3-5 h en T4 para 2-5 h de audio × 6 epochs |
| Aceptas la [CPML](https://coqui.ai/cpml) — el checkpoint hereda esa licencia | XTTS-v2 weights NO son Apache; tu fine-tune tampoco lo es |

**Si tu material es < 30 min, fine-tuning EMPEORA la voz**: el GPT encoder
sobre-aprende muletillas y entonación de los pocos segmentos disponibles.
Va a sonar exagerado, casi caricaturesco.

---

## Decisión rápida con datos

Después de procesar 3-5 entrevistas con el notebook principal:

```
| speaker_similarity_score (zero-shot) | Acción                              |
|--------------------------------------|-------------------------------------|
| ≥ 0.85 (very_high)                   | No fine-tunees. Ya está.            |
| 0.75-0.85 (high)                     | Fine-tune sólo si quieres pulir.    |
| 0.60-0.75 (marginal)                 | Fine-tuning probablemente ayuda.    |
| < 0.60 (low)                         | Material insuficiente. Otro modelo. |
```

El último caso es importante: si la similarity zero-shot es < 0.60,
**fine-tuning no te va a sacar del hoyo**. El GPT necesita un punto de
partida razonable; si el speaker es irreconocible para el encoder base,
el fine-tuning sólo amplifica el ruido.

---

## Flujo operativo (resumido)

1. **Pre-requisito**: ya corriste `notebook_voicelegacy.ipynb` (el principal) y tienes `reference_corpus/` poblado con ≥ 30 WAVs limpios.
2. Abrir `notebook_voicelegacy_finetune.ipynb` en Colab → **Runtime → Change runtime type → T4 GPU**.
3. Aceptar la CPML en la celda 1 (`ACCEPT_CPML = True`).
4. Editar las 3 rutas + `SPEAKER_LABEL` en la celda 2.
5. Correr celdas 3-7 en orden. La celda 7 (`trainer.fit()`) es la que tarda 3-5 h.
6. Si Colab te corta a mitad: regresar, montar Drive, **saltar a celda 7.bis** (reanuda desde el último checkpoint intermedio).
7. Celdas 8-10: materializar checkpoint + validar + A/B contra base.

---

## Qué se guarda en Drive (sobrevive a sesión muerta)

```
WORKSPACE_FT/
├── base_xtts/         ← 6 archivos del modelo base XTTS-v2 (~2 GB, descarga una vez)
├── dataset/           ← wavs + metadata_train.csv + metadata_eval.csv
├── runs/<run_name>/   ← checkpoints intermedios cada SAVE_EVERY_N_STEPS
└── finetuned/         ← checkpoint final, 6 archivos, listo para inferencia
```

Lo único que se pierde si la sesión muere: el modelo cargado en VRAM y
los conditioning latents calculados. **Cualquier WAV ya producido,
cualquier checkpoint guardado, cualquier dataset preparado sobrevive.**

---

## Uso del checkpoint en código

Idéntico al pipeline base, sustituyendo dos llamadas:

```python
from voicelegacy.config import SynthesisConfig
from voicelegacy.finetuned_inference import (
    FineTunedCheckpoint,
    load_finetuned_model,
    synthesize_with_finetuned,
    release_finetuned_model,
)

# 1. Validar el directorio (los 6 archivos requeridos)
ckpt = FineTunedCheckpoint.from_dir("/content/drive/MyDrive/voicelegacy_finetune/finetuned")
print("Fingerprint:", ckpt.fingerprint)

# 2. Cargar
model = load_finetuned_model(ckpt, device="auto")

# 3. Sintetizar
synthesize_with_finetuned(
    model=model,
    checkpoint=ckpt,
    text="Hola, este es el modelo fine-tuneado.",
    speaker_wav=[
        "/content/drive/MyDrive/voicelegacy_workspace/reference_corpus/ref_001.wav",
        "/content/drive/MyDrive/voicelegacy_workspace/reference_corpus/ref_002.wav",
    ],
    output_path="/content/drive/MyDrive/output.wav",
    config=SynthesisConfig(language="es", seed=42),
)

# 4. Liberar VRAM cuando termines
release_finetuned_model()
```

---

## Limitaciones conocidas (v0.3.0)

1. **Sin CLI dedicado.** Por diseño en esta versión — el checkpoint vive en Drive y el flujo natural es notebook. Si quieres `voicelegacy synthesize --finetuned-dir PATH`, abre issue para 0.4.0.
2. **Sin integración con `runs.db`.** Las síntesis fine-tuned NO entran al cache idempotente del paquete base. Si necesitas cache, vuelve a abrir el sidecar a mano. Scope de 0.4.0.
3. **Sin TPU.** Confirmado por docs oficiales de Coqui: XTTS-v2 fine-tuning es exclusivamente CUDA. T4 es la única opción gratuita en Colab.
4. **El test E2E con coqui-tts real NO está en CI** (requiere GPU + 2 GB de pesos). El notebook lo valida cada vez que lo ejecutas (celda 9).
5. **Pin estricto recomendado** de `coqui-tts==0.27.5` antes de fine-tunear, para que el checkpoint sea reproducible. Si actualizas el paquete entre experimentos, la API interna de `Xtts.load_checkpoint` puede cambiar y romper el checkpoint viejo.

---

## Cómo decidir si el fine-tune SIRVIÓ

La celda 10 del notebook automatiza esto. Lee el resultado:

| Δ Similarity (FT − BASE) | Interpretación |
|---|---|
| `≥ +0.05` | Fine-tune APORTA. Usa el checkpoint. |
| `+0.00 a +0.05` | Marginal. Escucha A/B antes de decidir. |
| `< 0.00` | Fine-tune EMPEORA. Causas: overfit, datos insuficientes, transcripciones desalineadas. **NO uses el checkpoint.** |

**No confíes solo en el número.** La métrica coseno no mide naturalidad
ni prosodia. Escucha 3-5 outputs A/B antes de declarar éxito.

---

## URLs oficiales

- Notebook oficial Coqui (fuente del enfoque): https://colab.research.google.com/drive/1GiI4_X724M8q2W-zZ-jXo7cWTV7RfaH-
- Docs XTTS-v2 training: https://docs.coqui.ai/en/latest/models/xtts.html#training
- Recipe LJSpeech (que adapté para corpus libre): https://github.com/idiap/coqui-ai-TTS/blob/main/recipes/ljspeech/xtts_v2/train_gpt_xtts.py
- coqui-tts (fork mantenido): https://pypi.org/project/coqui-tts/
- CPML: https://coqui.ai/cpml
- Resemblyzer (la métrica de similarity): https://github.com/resemble-ai/Resemblyzer
