"""Pydantic contracts for speakerscribe JSON files.

The old loader was deliberately permissive: it skipped malformed segments and
kept going. That is convenient in a notebook, but fragile in production because
bad diarization JSON can silently produce a bad reference corpus. These models
make the expected contract explicit while still allowing unrelated metadata.
"""

from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator, model_validator


class SpeakerscribeSegment(BaseModel):
    """One diarized speech segment from speakerscribe."""

    model_config = ConfigDict(extra="allow")

    start: float = Field(ge=0.0, description="Segment start time in seconds.")
    end: float = Field(gt=0.0, description="Segment end time in seconds.")
    speaker: str = Field(default="UNKNOWN", min_length=1)
    text: str = Field(default="")

    @field_validator("speaker", mode="before")
    @classmethod
    def _coerce_speaker(cls, value: object) -> str:
        text = str(value).strip() if value is not None else "UNKNOWN"
        return text or "UNKNOWN"

    @field_validator("text", mode="before")
    @classmethod
    def _coerce_text(cls, value: object) -> str:
        return "" if value is None else str(value).strip()

    @model_validator(mode="after")
    def _end_after_start(self) -> SpeakerscribeSegment:
        if self.end <= self.start:
            raise ValueError(f"segment end ({self.end}) must be > start ({self.start})")
        return self


class SpeakerscribeDocument(BaseModel):
    """Relevant subset of a speakerscribe JSON document."""

    model_config = ConfigDict(extra="allow")

    source_audio: str | None = None
    audio_file: str | None = None
    language_detected: str | None = None
    segments: list[SpeakerscribeSegment] = Field(default_factory=list)

    @model_validator(mode="after")
    def _has_source_or_stem_fallback(self) -> SpeakerscribeDocument:
        # We allow source_audio/audio_file to be absent because legacy exports
        # sometimes rely on the JSON stem. The loader owns that fallback.
        return self

    @property
    def source_name(self) -> str | None:
        """Return the declared source audio name, if any."""
        return self.source_audio or self.audio_file


def load_and_validate_speakerscribe_document(json_path: Path) -> SpeakerscribeDocument:
    """Read and validate a speakerscribe JSON file.

    Raises:
        FileNotFoundError: If the file is missing.
        ValueError: If the JSON is malformed or violates the contract.
    """
    json_path = Path(json_path)
    if not json_path.exists():
        raise FileNotFoundError(f"speakerscribe JSON not found: {json_path}")
    raw = json_path.read_text(encoding="utf-8")
    try:
        return SpeakerscribeDocument.model_validate_json(raw)
    except ValidationError as exc:
        raise ValueError(f"Invalid speakerscribe JSON schema in {json_path.name}: {exc}") from exc
