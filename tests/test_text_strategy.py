from voicelegacy.config import SynthesisConfig
from voicelegacy.text_strategy import plan_text_synthesis


def test_auto_strategy_does_not_split_short_text() -> None:
    plan = plan_text_synthesis("Hola mundo.", SynthesisConfig())
    assert plan.xtts_split_sentences is False
    assert plan.strategy == "auto"


def test_auto_strategy_splits_long_text() -> None:
    text = "a" * 300
    plan = plan_text_synthesis(text, SynthesisConfig(max_single_pass_chars=240))
    assert plan.xtts_split_sentences is True


def test_legacy_split_disabled_overrides_strategy() -> None:
    cfg = SynthesisConfig(enable_text_splitting=False, long_text_strategy="coqui_split")
    plan = plan_text_synthesis("a" * 1000, cfg)
    assert plan.xtts_split_sentences is False
    assert plan.strategy == "legacy_split_disabled"


def test_long_text_warning_is_recorded() -> None:
    plan = plan_text_synthesis("a" * 700, SynthesisConfig(long_text_warning_chars=600))
    assert plan.warning is not None
    assert "long_text_may_drift" in plan.warning
