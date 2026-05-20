# Ethics and permitted use

`voicelegacy` is intended for consent-based family archive and legacy-preservation workflows.

Required rules:

1. Obtain explicit consent from the person whose voice is cloned, or from the legally authorized representative when the person cannot consent.
2. Do not use generated audio to impersonate, deceive, pressure, scam, manipulate, or bypass identity checks.
3. Do not publish or distribute generated voice audio without consent from the voice owner or authorized representative.
4. Keep the JSON sidecar with every generated WAV. A voice file without provenance is not acceptable for this project.
5. Label degraded outputs honestly. If `source_quality.degraded_mode` is true or similarity is low, do not present the output as reliable.

The code in this repository may be MIT-licensed, but XTTS-v2 model weights are governed by the Coqui Public Model License. Read and accept that license before loading the model.
