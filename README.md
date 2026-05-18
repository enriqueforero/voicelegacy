# voicelegacy

> **Voice cloning pipeline for family legacy** — XTTS-v2 + speakerscribe integration, optimized for Google Colab Free Tier (T4 GPU).

[![Python](https://img.shields.io/badge/python-3.10%2B-blue)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Model License: CPML](https://img.shields.io/badge/Model_License-CPML-orange.svg)](https://coqui.ai/cpml)

---

## What it does

Takes diarized interview transcripts produced by [speakerscribe](https://github.com/EnriqueForero/speakerscribe), extracts the target speaker's clean speech segments, scores them, builds a curated reference corpus, and uses [XTTS-v2 (coqui-tts fork)](https://github.com/idiap/coqui-ai-TTS) to synthesize new audio in that voice.

```
interviews_raw/*.mp3
     │
     ▼ (speakerscribe — separate project)
speakerscribe_out/*.json
     │
     ▼  build_reference_corpus
reference_corpus/*.wav  (cleaned + ranked top N segments)
     │
     ▼  run_synthesis
synthesis_out/*.wav  (new audio in the target's voice)
```

---

## ⚠️ Read this before you build anything

### 1. The reference audio dominates everything
XTTS-v2 is zero-shot voice cloning. The model is frozen. **The only thing you control is the reference audio quality.** Garbage in → garbage out, and "garbage" includes:

- Phone-call recordings (8 kHz Opus/AMR codec): rejected by default. The model will faithfully clone the *codec compression*, not the voice.
- Audio with background TV / chatter / overlapping speakers.
- Clipped audio (peak above −1 dBFS).
- Anything below 16 kHz sample rate.

### 2. Colab Free is for **inference only**, not training
| Mode | Colab Free | Why |
|---|---|---|
| Zero-shot inference (this notebook) | ✅ Works | XTTS-v2 needs ~2–3 GB VRAM; T4 has 15 GB |
| Fine-tuning XTTS-v2 | ❌ Forget it | Requires 12–24 h continuous GPU; Colab disconnects at 12 h and on idle |

If zero-shot quality is insufficient, do fine-tuning on **RunPod, Lambda Labs, or Colab Pro+** — not Free.

### 3. License obligations (CPML)
The XTTS-v2 model weights are released under the **Coqui Public Model License**. You must explicitly accept it (`accept_coqui_tos=True` in `PipelineConfig`). For personal / family use this is fine. For commercial use, **read the license**: https://coqui.ai/cpml

### 4. Ethical considerations
A child interacting with the cloned voice of a deceased relative is non-trivial psychologically. Speak to a child psychologist before deployment. The strongest legacy is often the **archive of real recordings**, not synthetic speech that puts new words in the speaker's mouth.

This tool does not police usage. The author assumes consent of the cloned subject has been obtained.

---

## Quick start (Google Colab)

Open `notebooks/notebook_voicelegacy.ipynb` and follow the cells. The full flow is:

```python
from voicelegacy import (
    PipelineConfig, ReferenceConfig, SynthesisConfig, WorkspacePaths,
    run_reference_phase, run_synthesis,
)

paths = WorkspacePaths(workspace="/content/drive/MyDrive/Legado")
paths.mkdirs()

config = PipelineConfig(
    reference=ReferenceConfig(target_speaker_label="SPEAKER_00", top_n_segments=10),
    synthesis=SynthesisConfig(language="es"),
    accept_coqui_tos=True,  # you've read https://coqui.ai/cpml
)

# Phase 1: build clean reference set from speakerscribe outputs
corpus = run_reference_phase(paths, config)

# Phase 2: synthesize
result = run_synthesis(
    text="Mi querido nieto, quiero contarte que la paciencia es la virtud más importante.",
    reference_wavs=corpus.top_wavs,
    paths=paths,
    config=config,
)
print(result.output_path)
```

---

## Workspace layout

```
workspace/                          ← root (Drive)
├── interviews_raw/                 ← put raw audio here (mp3/wav/m4a/...)
├── speakerscribe_out/              ← speakerscribe writes .json here
├── reference_corpus/               ← voicelegacy writes clean WAVs here
├── synthesis_out/                  ← cloned-voice outputs land here
├── reports/                        ← per-run quality JSONs
└── runs.db                         ← SQLite idempotency cache
```

---

## Installation (local development)

```bash
git clone https://github.com/EnriqueForero/voicelegacy
cd voicelegacy
pip install -e ".[dev]"
pre-commit install
```

PyTorch is not pinned — it ships pre-installed on Colab. For local installs:

```bash
# CUDA 12.1 example; check pytorch.org for your hardware
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu121
```

---

## CLI

```bash
voicelegacy build-corpus \
  --workspace ~/Drive/Legado \
  --speaker SPEAKER_00 \
  --top-n 10 \
  --accept-tos

voicelegacy synthesize \
  --workspace ~/Drive/Legado \
  --text "Hola mi nieto, hoy te quiero contar..." \
  --accept-tos
```

---

## Architecture

```
voicelegacy/
├── config.py       ← Pydantic v2 models (single source of truth)
├── audio.py        ← load / normalize / denoise / trim / slice
├── quality.py      ← per-segment SNR/duration scoring + ranking
├── corpus.py       ← parse speakerscribe JSON → extract → write WAVs
├── synthesis.py    ← XTTS-v2 wrapper (TOS handling, VRAM mgmt)
├── persistence.py  ← SQLite cache for idempotent re-runs
├── pipeline.py     ← orchestrates phase 1 (corpus) + phase 2 (synthesis)
└── cli.py          ← Typer CLI
```

---

## Testing

```bash
pytest                # full suite
pytest -m "not gpu"   # skip GPU-required tests
pytest --cov          # with coverage report
```

---

## License

- **Code (this repository)**: MIT.
- **Model weights (XTTS-v2)**: Coqui Public Model License — https://coqui.ai/cpml — accepted at runtime by the user.
