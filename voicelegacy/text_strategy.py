"""Text-length policy for XTTS-v2 synthesis.

XTTS-v2 can split long text internally. That is convenient, but it may also
introduce voice drift between sentences because each segment is synthesized with
less shared context. This module makes the policy explicit and auditable.
"""

from __future__ import annotations

from dataclasses import dataclass

from voicelegacy.config import SynthesisConfig


@dataclass(frozen=True)
class TextSynthesisPlan:
    """Decision record for how one text will be sent to XTTS-v2."""

    char_count: int
    strategy: str
    xtts_split_sentences: bool
    warning: str | None = None

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-serializable representation."""
        return {
            "char_count": self.char_count,
            "strategy": self.strategy,
            "xtts_split_sentences": self.xtts_split_sentences,
            "warning": self.warning,
        }


def plan_text_synthesis(text: str, config: SynthesisConfig) -> TextSynthesisPlan:
    """Decide whether XTTS-v2 should split the text internally.

    Policy:
        - ``single_pass``: never split; best for short controlled phrases.
        - ``coqui_split``: always let XTTS/Coqui split; best for long prose.
        - ``auto``: avoid splitting short text to reduce drift; enable splitting
          only once the text exceeds ``max_single_pass_chars``.

    The legacy boolean ``enable_text_splitting`` remains as a hard off-switch:
    if it is False, no strategy may re-enable splitting.
    """
    stripped = text.strip()
    if not stripped:
        return TextSynthesisPlan(
            char_count=0,
            strategy=config.long_text_strategy,
            xtts_split_sentences=False,
            warning="empty_text",
        )

    n_chars = len(stripped)
    warning = None
    if n_chars >= config.long_text_warning_chars:
        warning = (
            "long_text_may_drift; prefer shorter utterances or review output sidecars/listening"
        )

    if not config.enable_text_splitting:
        return TextSynthesisPlan(
            char_count=n_chars,
            strategy="legacy_split_disabled",
            xtts_split_sentences=False,
            warning=warning,
        )

    if config.long_text_strategy == "single_pass":
        split = False
    elif config.long_text_strategy == "coqui_split":
        split = True
    else:
        split = n_chars > config.max_single_pass_chars

    return TextSynthesisPlan(
        char_count=n_chars,
        strategy=config.long_text_strategy,
        xtts_split_sentences=split,
        warning=warning,
    )
