"""Tests for audio container-conversion utilities (P0-7 additions).

These tests exercise convert_to_wav and convert_directory_to_wav, which
replaced the hand-edited Cell 17 of the legacy notebook.

ffmpeg is required. Tests are skipped if it is not on PATH so the suite
stays green on environments without it (e.g. minimal CI runners).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from voicelegacy.audio import (
    EXTENSIONS_CONVERTIBLE_TO_WAV,
    _ffmpeg_available,
    convert_directory_to_wav,
    convert_to_wav,
)

pytestmark = pytest.mark.skipif(
    not _ffmpeg_available(),
    reason="ffmpeg not on PATH",
)


def _write_pcm_wav(path: Path, sr: int, duration_s: float = 2.0) -> Path:
    """Write a small mono WAV file for tests to convert."""
    n = int(sr * duration_s)
    t = np.arange(n) / sr
    y = (0.3 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)
    sf.write(str(path), y, sr, subtype="PCM_16")
    return path


class TestConvertToWav:
    def test_converts_mp3_to_wav(self, tmp_path: Path) -> None:
        """Use a WAV source and request a different extension as the bridge —
        avoids needing an MP3 encoder in the test environment. The transcode
        itself is what we want to validate, not the input format detection.
        """
        src = tmp_path / "source.wav"
        _write_pcm_wav(src, sr=16000, duration_s=1.0)
        dst = tmp_path / "out.wav"
        produced = convert_to_wav(src, dst=dst, target_sr=22050)

        assert produced == dst
        assert dst.exists()
        # The output must really be at 22050 Hz (ffmpeg resampled it)
        info = sf.info(str(dst))
        assert info.samplerate == 22050
        assert info.channels == 1

    def test_skips_when_dst_exists_and_no_overwrite(self, tmp_path: Path) -> None:
        src = _write_pcm_wav(tmp_path / "s.wav", sr=22050)
        dst = _write_pcm_wav(tmp_path / "dst.wav", sr=16000, duration_s=0.5)
        original_size = dst.stat().st_size

        produced = convert_to_wav(src, dst=dst, overwrite=False)
        assert produced == dst
        # File untouched
        assert dst.stat().st_size == original_size

    def test_overwrites_when_flag_set(self, tmp_path: Path) -> None:
        src = _write_pcm_wav(tmp_path / "s.wav", sr=22050, duration_s=3.0)
        dst = _write_pcm_wav(tmp_path / "dst.wav", sr=16000, duration_s=0.5)
        original_size = dst.stat().st_size

        convert_to_wav(src, dst=dst, overwrite=True)
        # File was re-encoded from a longer source → must differ in size
        assert dst.stat().st_size != original_size

    def test_missing_source_raises(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError, match="Source audio not found"):
            convert_to_wav(tmp_path / "nope.wav", dst=tmp_path / "x.wav")

    def test_default_dst_replaces_suffix(self, tmp_path: Path) -> None:
        """When dst=None, the default is src.with_suffix('.wav'). The caller's
        responsibility: src must NOT already be a .wav, otherwise the implicit
        dst collides with src and ffmpeg refuses in-place edits.
        """
        # Build an .ogg via ffmpeg first
        import subprocess

        src_wav_tmp = _write_pcm_wav(tmp_path / "tmp.wav", sr=22050, duration_s=0.5)
        src_ogg = tmp_path / "track.ogg"
        result = subprocess.run(
            ["ffmpeg", "-i", str(src_wav_tmp), "-y", str(src_ogg)],
            capture_output=True,
        )
        if result.returncode != 0 or not src_ogg.exists():
            pytest.skip("ffmpeg in this environment cannot encode ogg")
        src_wav_tmp.unlink()

        produced = convert_to_wav(src_ogg, dst=None, target_sr=22050, overwrite=True)
        assert produced.suffix == ".wav"
        assert produced == src_ogg.with_suffix(".wav")
        info = sf.info(str(produced))
        assert info.samplerate == 22050


class TestConvertDirectoryToWav:
    def test_empty_dir_returns_empty_list(self, tmp_path: Path) -> None:
        produced = convert_directory_to_wav(tmp_path)
        assert produced == []

    def test_skips_files_with_unsupported_extensions(self, tmp_path: Path) -> None:
        # A .txt is not in EXTENSIONS_CONVERTIBLE_TO_WAV
        (tmp_path / "notes.txt").write_text("hi")
        # A WAV is also skipped (it's the OUTPUT format, not a candidate input)
        _write_pcm_wav(tmp_path / "already.wav", sr=22050)
        produced = convert_directory_to_wav(tmp_path)
        # Nothing to convert (.wav is not in the candidate set)
        assert produced == []

    def test_not_a_directory_raises(self, tmp_path: Path) -> None:
        bogus = tmp_path / "does_not_exist"
        with pytest.raises(NotADirectoryError):
            convert_directory_to_wav(bogus)

    def test_idempotent_on_second_run(self, tmp_path: Path) -> None:
        # Write an .ogg source — ffmpeg has built-in vorbis decode
        src_ogg = tmp_path / "voice.ogg"
        _write_pcm_wav(tmp_path / "voice_tmp.wav", sr=22050, duration_s=1.0)
        # Use ffmpeg via convert_to_wav with explicit dst to create the .ogg
        import subprocess

        result = subprocess.run(
            ["ffmpeg", "-i", str(tmp_path / "voice_tmp.wav"), "-y", str(src_ogg)],
            capture_output=True,
        )
        if result.returncode != 0 or not src_ogg.exists():
            pytest.skip("ffmpeg in this environment cannot encode ogg")
        (tmp_path / "voice_tmp.wav").unlink()

        # First run: converts
        produced_1 = convert_directory_to_wav(tmp_path)
        assert len(produced_1) == 1
        assert produced_1[0].name == "voice.wav"
        size_after_first = produced_1[0].stat().st_size

        # Second run: skips (file exists)
        produced_2 = convert_directory_to_wav(tmp_path, overwrite=False)
        assert len(produced_2) == 1
        # Size unchanged → no re-encode happened
        assert produced_2[0].stat().st_size == size_after_first


class TestExtensionsSet:
    def test_extensions_set_is_frozen(self) -> None:
        """EXTENSIONS_CONVERTIBLE_TO_WAV is the single source of truth shared
        across audio.py, corpus.py, and the CLI. It must be immutable to
        prevent accidental mutation at runtime.
        """
        assert isinstance(EXTENSIONS_CONVERTIBLE_TO_WAV, frozenset)
        assert ".mp4" in EXTENSIONS_CONVERTIBLE_TO_WAV
        assert ".mkv" in EXTENSIONS_CONVERTIBLE_TO_WAV
        assert ".webm" in EXTENSIONS_CONVERTIBLE_TO_WAV
        # .wav is the OUTPUT format; must NOT be in the input candidate set
        assert ".wav" not in EXTENSIONS_CONVERTIBLE_TO_WAV
