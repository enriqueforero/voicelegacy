"""Tests for the corpus module."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from voicelegacy.corpus import filter_segments, load_speakerscribe_json


class TestLoadJSON:
    def test_parses_well_formed(self, speakerscribe_json_factory, tmp_path: Path) -> None:
        # Create the referenced source audio so the path resolves
        audio_dir = tmp_path.parent / "data"
        audio_dir.mkdir(exist_ok=True)
        (audio_dir / "source.wav").touch()

        jp = speakerscribe_json_factory(
            "interview01.json",
            segments=[
                {"start": 0.0, "end": 5.0, "speaker": "SPEAKER_00", "text": "Hola"},
                {"start": 5.0, "end": 9.0, "speaker": "SPEAKER_01", "text": "Adios"},
            ],
        )
        segs = load_speakerscribe_json(jp, audio_root=tmp_path)
        assert len(segs) == 2
        assert segs[0].speaker == "SPEAKER_00"
        assert segs[0].duration_s == pytest.approx(5.0)

    def test_missing_file_raises(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError):
            load_speakerscribe_json(tmp_path / "ghost.json")

    def test_malformed_json_raises(self, tmp_path: Path) -> None:
        bad = tmp_path / "bad.json"
        bad.write_text("{not really json", encoding="utf-8")
        with pytest.raises(ValueError):
            load_speakerscribe_json(bad)

    def test_skips_malformed_segments(self, tmp_path: Path) -> None:
        p = tmp_path / "partial.json"
        payload = {
            "source_audio": "x.wav",
            "segments": [
                {"start": 0.0, "end": 5.0, "speaker": "SPEAKER_00", "text": "ok"},
                {"start": "not-a-number", "end": 9.0, "speaker": "SPEAKER_00"},  # bad
            ],
        }
        p.write_text(json.dumps(payload), encoding="utf-8")
        segs = load_speakerscribe_json(p, audio_root=tmp_path)
        assert len(segs) == 1  # malformed one skipped


class TestFilter:
    def test_filters_by_speaker_and_duration(self, tmp_path: Path) -> None:
        from voicelegacy.corpus import SegmentRef

        audio = tmp_path / "x.wav"
        segs = [
            SegmentRef(audio, 0.0, 5.0, "SPEAKER_00", "a"),  # keep
            SegmentRef(audio, 5.0, 5.5, "SPEAKER_00", "b"),  # too short
            SegmentRef(audio, 10.0, 30.0, "SPEAKER_00", "c"),  # too long
            SegmentRef(audio, 30.0, 35.0, "SPEAKER_01", "d"),  # wrong speaker
            SegmentRef(audio, 40.0, 47.0, "SPEAKER_00", "e"),  # keep
        ]
        out = filter_segments(
            segs,
            target_speaker="SPEAKER_00",
            min_duration_s=4.0,
            max_duration_s=15.0,
        )
        kept_texts = [s.text for s in out]
        assert kept_texts == ["a", "e"]
