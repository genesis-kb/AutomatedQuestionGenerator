"""S1 contracts: a talk and its chunks."""

from __future__ import annotations

from datetime import date, datetime

from pydantic import BaseModel, Field


class ConferenceInfo(BaseModel):
    name: str
    slug: str
    year: int | None = None
    source: str = "fallback"  # taxonomy | content_source | cli | fallback


class Candidate(BaseModel):
    """Cheap metadata used for triage (no full text)."""

    content_item_id: str
    transcript_id: str
    transcript_version: int = 1
    title: str
    description: str | None = None
    url: str | None = None
    published_at: datetime | None = None
    event_date: date | None = None
    technical_score: int | None = None
    conference_name: str | None = None
    source_name: str | None = None
    source_slug: str | None = None
    speakers: list[str] = Field(default_factory=list)
    summary: str | None = None
    word_count: int | None = None
    stt_model: str | None = None
    keyword_hits: list[str] = Field(default_factory=list)
    triage_score: float | None = None


class TranscriptDoc(BaseModel):
    candidate: Candidate
    text: str
    text_field: str  # corrected_text | raw_text | file
    conference: ConferenceInfo


class Chunk(BaseModel):
    chunk_id: int
    text: str
    char_start: int
    char_end: int
