# Proceso completo: clonar una voz desde entrevistas con varias personas

Explicado desde cero. Si nunca has hecho esto, este documento te da claridad total.

---

## El problema en palabras simples

Tienes grabaciones de entrevistas. En cada una hablan **dos o más personas**:
la persona cuya voz quieres recrear (digamos, tu abuela) y otra (el entrevistador).
El audio es una sola pista donde las voces se **alternan en el tiempo**.

Para que una IA aprenda a hablar como tu abuela, necesita escuchar **solo a tu
abuela**. Si le das audio donde también habla el entrevistador, aprende una
mezcla y el resultado suena a nadie en particular.

Entonces el reto es: **separar lo que dijo tu abuela de lo que dijeron los demás.**

---

## Las dos librerías y qué hace cada una

Este proyecto usa DOS herramientas en cadena. Cada una hace una cosa.

### speakerscribe — "¿quién habló y cuándo?"

Toma una entrevista con varias personas y produce un archivo `.json` que dice,
para cada tramo de tiempo, **quién habló y qué dijo**:

```
segundo 0 a 12   → SPEAKER_01 (entrevistador): "Cuéntame de tu infancia"
segundo 12 a 28  → SPEAKER_00 (tu abuela): "Nací en un pueblo en los años cuarenta"
segundo 28 a 33  → SPEAKER_01: "¿Y cómo era?"
segundo 33 a 50  → SPEAKER_00: "Era muy tranquilo, había..."
```

Esto se llama **diarización** (identificar "quién habla cuándo"). speakerscribe
usa pyannote.audio para esto, y faster-whisper para transcribir el texto.

### voicelegacy — "aprende y recrea la voz"

Toma esos tramos etiquetados, **recorta solo los de tu abuela**, los limpia
(quita ruido, normaliza volumen), y entrena el modelo XTTS-v2 para clonar su voz.

---

## ¿El audio debe estar totalmente aislado de los demás? SÍ

Para entrenar, el audio debe contener **solo la voz de tu abuela**. Ni una
palabra del entrevistador.

Pero "aislado" no significa que edites a mano. Significa que, después de la
diarización, el sistema **recorta y junta solo los tramos de tu abuela** y
descarta el resto automáticamente. Tú solo le dices cuál etiqueta
(`SPEAKER_00`, `SPEAKER_01`...) es ella.

Los tramos donde **se pisan las voces** (hablan dos a la vez) son problemáticos
y el sistema los descarta.

---

## El detalle crítico: las etiquetas NO son consistentes entre archivos

Esto es lo más importante de entender, y la razón de que el proceso tenga un
paso manual.

speakerscribe diariza **cada entrevista por separado**. No tiene memoria entre
archivos. Entonces:

- En la entrevista A, tu abuela puede ser `SPEAKER_00`.
- En la entrevista B, puede ser `SPEAKER_01`.
- En la entrevista C, `SPEAKER_00` otra vez, pero por casualidad.

**El sistema no sabe que es la misma persona entre archivos.** Solo separa voces
dentro de cada uno y les pone nombres genéricos.

Por eso, en el notebook puente, tienes que **identificar a tu abuela una vez por
entrevista**: escuchas una muestra de cada etiqueta y anotas cuál es ella. Con
10 entrevistas son ~20 minutos de tu tiempo. Es tedioso pero es la única forma
confiable. Si te equivocas de etiqueta, contaminas el corpus con la voz
equivocada y arruinas horas de entrenamiento.

---

## El flujo completo, paso a paso

```
PASO 0  Tienes: 10+ horas de entrevistas (varias personas por entrevista)
           │
           ▼
PASO 1  speakerscribe  →  notebook_speakerscribe.ipynb
        Diariza + transcribe cada entrevista.
        Produce: un .json por entrevista (quién dijo qué y cuándo).
        Necesitas: token de HuggingFace (gratis) para pyannote. YA LO TIENES.
           │
           ▼
PASO 2  Notebook puente  →  notebook_voicelegacy_bridge.ipynb   ← TU SIGUIENTE PASO
        a) Lee los .json de speakerscribe.
        b) Te muestra cuántos hablantes hay por entrevista y cuánto habla cada uno.
        c) Reproduce una muestra de cada hablante — TÚ identificas a la abuela.
        d) Anotas el mapa: {entrevista_01.json: SPEAKER_00, entrevista_02.json: SPEAKER_01, ...}
        e) Recorta + limpia SOLO los segmentos de la abuela de TODAS las entrevistas.
        f) Consolida en un reference_corpus/ único.
        g) Te dice cuántos minutos netos de voz juntaste.
           │
           ▼
PASO 3  Fine-tuning  →  notebook_voicelegacy_finetune.ipynb
        Entrena XTTS-v2 con la voz limpia de la abuela.
        Produce: un checkpoint reutilizable.
        Con 2-5 h de voz neta → calidad de legado.
           │
           ▼
PASO 4  Generar voz nueva  →  con el checkpoint, cualquier texto en su voz.
```

---

## Por qué tu caso (5 horas) es el ideal

Antes te advertí que 30 minutos era el límite inferior. **5 horas de voz neta de
tu abuela es exactamente el escenario ideal** del fine-tuning (2-5 horas).

De 10+ horas de entrevistas crudas, extraer 5 horas netas de tu abuela es
realista si ella habla más o menos la mitad del tiempo. Con eso:

- El fine-tuning funcionará bien, no marginal.
- Esperarías un clon de calidad de legado real.
- Tienes margen para descartar tramos de mala calidad y aún quedar con material
  de sobra.

**Este es el caso que vale la pena hacer bien.**

---

## Qué notebook usar para qué

Tienes cuatro notebooks. Para tu caso (muchas entrevistas, varias personas):

| Notebook | Para qué | ¿Tu caso? |
|---|---|---|
| `notebook_speakerscribe.ipynb` (otra librería) | Diarizar + transcribir | ✅ PASO 1 |
| **`notebook_voicelegacy_bridge.ipynb`** | Enlazar: extraer la voz de la abuela | ✅ **PASO 2** |
| `notebook_voicelegacy_finetune.ipynb` | Entrenar XTTS-v2 con el corpus | ✅ PASO 3 |
| `notebook_voicelegacy_finetune_standalone.ipynb` | Todo-en-uno desde UNA grabación de monólogo | ❌ (es para una sola voz) |
| `notebook_voicelegacy.ipynb` | Inferencia zero-shot (sin entrenar) | ❌ (ya viste que sale mal) |

---

## Respuestas directas a tus preguntas

**¿Lo que me compartió sirve, o debo diarizar con speakerscribe?**
Debes diarizar con speakerscribe primero. Lo que te entregué (voicelegacy) NO
diariza — recibe el resultado ya diarizado. Son complementarios, en cadena.

**¿El audio debe estar totalmente aislado de los demás?**
Sí, para entrenar. Pero no lo aíslas a mano: el notebook puente recorta solo los
tramos de tu abuela usando las etiquetas de speakerscribe.

**¿Su notebook hace eso?**
El notebook puente (nuevo, en esta entrega) hace exactamente eso: identifica a la
abuela por entrevista, recorta y limpia solo su voz, consolida todo. El notebook
standalone (de 30 min) NO sirve para tu caso porque no separa hablantes.

**¿Me apalanco en la otra librería o lo que entregó es suficiente?**
Te apalancas en ambas, en cadena: speakerscribe (diariza) → voicelegacy (clona).
Ninguna sola es suficiente para tu caso.

---

## Gestión de memoria (por qué esto corre en Colab Free sin reventar)

Colab gratuito da ~12 GB de RAM. Con 10+ horas de entrevistas, cargar los audios
enteros a memoria revienta la sesión. El notebook puente está diseñado para evitarlo:

- **Lectura parcial desde disco.** Cuando reproduce una muestra de un hablante (para
  que la identifiques), NO carga la entrevista completa: lee solo ese fragmento de
  ~12 segundos directamente del disco. Una entrevista de 1 hora cuesta la misma RAM
  que una de 12 segundos.
- **Una entrevista a la vez.** El recorte procesa una entrevista, escribe sus WAVs,
  libera la RAM (`gc.collect()`), y recién entonces pasa a la siguiente. Nunca hay
  dos entrevistas completas en memoria al mismo tiempo.
- **Checkpointing.** Tras cada entrevista, guarda un `bridge_manifest.json` que
  registra lo ya hecho. Si Colab te corta la sesión a media extracción, vuelves a
  correr la celda 6 y **salta las entrevistas ya procesadas** — no reprocesa nada.
- **Monitoreo de RAM.** Avisa si la RAM cruza el 85%, para que sepas antes de que
  reviente.

En la práctica: puedes procesar las 10+ horas en una sola sesión, y si se corta,
reanudas sin perder trabajo.

## Requisitos antes de empezar el PASO 2

- [ ] Ya corriste speakerscribe sobre tus entrevistas (tienes los `.json`).
- [ ] Tienes los audios originales accesibles en una carpeta de Drive.
- [ ] Sabes (o vas a identificar escuchando) cuál hablante es tu abuela en cada entrevista.
- [ ] Colab con runtime GPU (el filtro de tono usa GPU; sin ella funciona pero más lento).

---

## Solución de problemas

| Síntoma | Causa | Solución |
|---|---|---|
| "No hay .json" | No corriste speakerscribe, o ruta mal | Corre el PASO 1 primero |
| "No encuentro el audio de X" | Los audios originales no están donde el JSON dice | Pon los audios originales en `ORIGINAL_AUDIOS` con el nombre que declara el JSON |
| "Ninguno — ¿etiqueta correcta?" | Mapeaste la etiqueta equivocada | Re-escucha con la celda 4, corrige el mapa |
| El clon suena a 2 personas | Corpus contaminado (etiqueta mal) | Revisa el mapa por entrevista, vuelve a extraer |
| Pocos minutos netos | La abuela habla poco, o pocas entrevistas | Procesa más entrevistas con speakerscribe |

---

## URLs de referencia

- speakerscribe (diarización): tu repo / https://github.com/EnriqueForero/speakerscribe
- pyannote.audio (motor de diarización): https://github.com/pyannote/pyannote-audio
- faster-whisper (transcripción): https://github.com/SYSTRAN/faster-whisper
- XTTS-v2 (clonación): https://docs.coqui.ai/en/latest/models/xtts.html
- HuggingFace tokens: https://huggingface.co/settings/tokens
