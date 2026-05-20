# P3 evaluation protocol

P3 is product improvement, not a prerequisite for basic operation. Do not make a new denoise model or text-splitting policy the default without evidence from real source material.

## DeepFilterNet versus noisereduce

DeepFilterNet is optional. The package extra is:

```bash
pip install -e ".[deepfilter]"
```

The evaluation command is:

```bash
voicelegacy evaluate-denoise \
  --workspace /path/to/workspace \
  --audio /path/to/clean.wav \
  --audio /path/to/moderate_noise.wav \
  --audio /path/to/heavy_noise.wav \
  --audio /path/to/phone_codec.wav \
  --audio /path/to/long_interview_excerpt.wav \
  --deepfilter
```

Minimum sample set:

1. Clean recording.
2. Moderate noise.
3. Heavy noise or cross-talk.
4. Phone-codec / low sample rate.
5. Long interview excerpt.

Decision rule: DeepFilterNet may become the default only if it improves human listening quality and downstream `speaker_similarity_score` without adding speech artifacts. Better noise suppression with lower speaker similarity is not a win for this project.

## Long text policy

`SynthesisConfig.long_text_strategy` controls how text is sent to XTTS-v2:

- `auto`: default. Short utterances are sent as a single pass; long prose delegates sentence splitting to XTTS/Coqui.
- `single_pass`: never split. Useful for controlled short phrases only.
- `coqui_split`: always delegate sentence splitting to XTTS/Coqui.

Sidecar JSON includes `text_plan`, so long-text decisions are auditable.

## Manual listening protocol

For each generated output, review:

1. The WAV itself.
2. Its sidecar JSON.
3. `speaker_similarity_score` and quality band.
4. `source_quality.degraded_mode`.
5. Reference and F0 reports.

A synthetic output is not accepted merely because tests pass.
