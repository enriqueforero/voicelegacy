from pathlib import Path

import numpy as np
import soundfile as sf

from voicelegacy.denoise_eval import evaluate_denoise_methods


def _write_wav(path: Path, sr: int = 22050) -> None:
    t = np.linspace(0, 1.2, int(sr * 1.2), endpoint=False)
    y = 0.1 * np.sin(2 * np.pi * 220 * t).astype(np.float32)
    sf.write(path, y, sr)


def test_evaluate_denoise_methods_writes_report_and_noisereduce_candidate(tmp_path: Path) -> None:
    src = tmp_path / "sample.wav"
    _write_wav(src)

    report = evaluate_denoise_methods([src], tmp_path / "out", include_deepfilter=False)

    assert Path(str(report["report_path"])).exists()
    assert len(report["candidates"]) == 1
    row = report["candidates"][0]
    assert row["method"] == "noisereduce_nonstationary"
    assert row["status"] == "ok"
    assert Path(str(row["output_path"])).exists()


def test_evaluate_denoise_methods_records_missing_file(tmp_path: Path) -> None:
    report = evaluate_denoise_methods([tmp_path / "missing.wav"], tmp_path / "out")
    assert report["candidates"][0]["status"] == "failed"
    assert "not found" in report["candidates"][0]["reason"]


def test_deepfilter_candidate_skips_when_cli_missing(tmp_path: Path, monkeypatch) -> None:
    src = tmp_path / "sample.wav"
    _write_wav(src)
    monkeypatch.setattr("voicelegacy.denoise_eval.shutil.which", lambda name: None)

    report = evaluate_denoise_methods([src], tmp_path / "out", include_deepfilter=True)

    methods = [r["method"] for r in report["candidates"]]
    assert methods == ["noisereduce_nonstationary", "deepfilternet"]
    assert report["candidates"][1]["status"] == "skipped"
