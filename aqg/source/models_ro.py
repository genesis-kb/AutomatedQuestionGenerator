"""Slim read-only SQLAlchemy models for the transcription_engine schema.

Deliberately independent of transcription_engine code. Only the columns we read are mapped.
Never call ``create_all`` on this metadata.
"""

from __future__ import annotations

from datetime import date, datetime

from sqlalchemy import Boolean, Date, DateTime, ForeignKey, Integer, Text
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import DeclarativeBase, Mapped, deferred, mapped_column


class ROBase(DeclarativeBase):
    pass


class Taxonomy(ROBase):
    __tablename__ = "taxonomies"

    id: Mapped[str] = mapped_column(UUID(as_uuid=False), primary_key=True)
    type: Mapped[str] = mapped_column(Text)
    name: Mapped[str] = mapped_column(Text)
    slug: Mapped[str] = mapped_column(Text)
    meta: Mapped[dict | None] = mapped_column("metadata", JSONB)


class ContentSource(ROBase):
    __tablename__ = "content_sources"

    id: Mapped[str] = mapped_column(UUID(as_uuid=False), primary_key=True)
    name: Mapped[str] = mapped_column(Text)
    slug: Mapped[str] = mapped_column(Text)
    source_type: Mapped[str | None] = mapped_column(Text)
    config: Mapped[dict | None] = mapped_column(JSONB)


class ContentItem(ROBase):
    __tablename__ = "content_items"

    id: Mapped[str] = mapped_column(UUID(as_uuid=False), primary_key=True)
    source_id: Mapped[str | None] = mapped_column(UUID(as_uuid=False), ForeignKey("content_sources.id"))
    external_id: Mapped[str | None] = mapped_column(Text)
    title: Mapped[str] = mapped_column(Text)
    description: Mapped[str | None] = mapped_column(Text)
    content_type: Mapped[str | None] = mapped_column(Text)
    url: Mapped[str | None] = mapped_column(Text)
    published_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    event_date: Mapped[date | None] = mapped_column(Date)
    event_id: Mapped[str | None] = mapped_column(UUID(as_uuid=False), ForeignKey("taxonomies.id"))
    status: Mapped[str | None] = mapped_column(Text)
    technical_score: Mapped[int | None] = mapped_column(Integer)
    source_metadata: Mapped[dict | None] = mapped_column(JSONB)


class Transcript(ROBase):
    __tablename__ = "transcripts"

    id: Mapped[str] = mapped_column(UUID(as_uuid=False), primary_key=True)
    content_item_id: Mapped[str] = mapped_column(UUID(as_uuid=False), ForeignKey("content_items.id"))
    is_current: Mapped[bool | None] = mapped_column(Boolean)
    version: Mapped[int | None] = mapped_column(Integer)
    raw_text: Mapped[str | None] = deferred(mapped_column(Text))
    corrected_text: Mapped[str | None] = deferred(mapped_column(Text))
    stt_model: Mapped[str | None] = mapped_column(Text)
    correction_model: Mapped[str | None] = mapped_column(Text)
    duration_seconds: Mapped[int | None] = mapped_column(Integer)
    word_count: Mapped[int | None] = mapped_column(Integer)
    chapters: Mapped[list | None] = deferred(mapped_column(JSONB))


class Summary(ROBase):
    __tablename__ = "summaries"

    id: Mapped[str] = mapped_column(UUID(as_uuid=False), primary_key=True)
    transcript_id: Mapped[str] = mapped_column(UUID(as_uuid=False), ForeignKey("transcripts.id"))
    summary_type: Mapped[str] = mapped_column(Text)
    content: Mapped[str] = mapped_column(Text)


class Speaker(ROBase):
    __tablename__ = "speakers"

    id: Mapped[str] = mapped_column(UUID(as_uuid=False), primary_key=True)
    name: Mapped[str] = mapped_column(Text)
    slug: Mapped[str] = mapped_column(Text)


class ContentItemSpeaker(ROBase):
    __tablename__ = "content_item_speakers"

    content_item_id: Mapped[str] = mapped_column(UUID(as_uuid=False), ForeignKey("content_items.id"),
                                                 primary_key=True)
    speaker_id: Mapped[str] = mapped_column(UUID(as_uuid=False), ForeignKey("speakers.id"), primary_key=True)
    role: Mapped[str | None] = mapped_column(Text)
