"""Read-only access to the transcription_engine Postgres.

Three layers keep this strictly read-only:
  1. ``postgresql_readonly`` execution option (driver-level read-only transactions),
  2. ``SET SESSION CHARACTERISTICS AS TRANSACTION READ ONLY`` on every new connection,
  3. a ``before_cursor_execute`` guard that refuses anything but SELECT / WITH / SET / SHOW.
"""

from __future__ import annotations

import re
from typing import Any

from sqlalchemy import String, and_, cast, create_engine, event, func, or_, select, text
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session, aliased, sessionmaker

from aqg.config import ConferenceRef
from aqg.logging_setup import get_logger, step
from aqg.schemas.transcript import Candidate, ConferenceInfo, TranscriptDoc
from aqg.source.cleaner import clean_text
from aqg.source.models_ro import ContentItem, ContentItemSpeaker, ContentSource, Speaker, Summary, Taxonomy, Transcript
from aqg.textutil import slugify

log = get_logger("source")

ALLOWED_KEYWORDS = ("SELECT", "WITH", "SET", "SHOW")
GENERIC_SOURCE_SLUGS = ("manual-imports", "misc")
SUMMARY_PREFERENCE = ("technical", "tldr")
_SLUG_RE = re.compile(r"^[a-z0-9_-]+$")
_LEADING_NOISE = re.compile(r"^(?:\s+|--[^\n]*\n?|/\*.*?\*/|\()+", re.DOTALL)


class ReadOnlyViolation(RuntimeError):
    """A statement that could write was about to be sent to the source database."""


class NoTextError(ValueError):
    """The transcript has neither corrected_text nor raw_text (or the file is empty)."""


def _first_keyword(statement: str) -> str:
    stripped = _LEADING_NOISE.sub("", statement)
    match = re.match(r"[A-Za-z]+", stripped)
    return match.group(0).upper() if match else ""


def _normalize_url(url: str) -> str:
    if url.startswith("postgresql://"):
        return "postgresql+psycopg://" + url[len("postgresql://"):]
    if url.startswith("postgres://"):
        return "postgresql+psycopg://" + url[len("postgres://"):]
    return url


def _install_guards(engine: Engine) -> None:
    @event.listens_for(engine, "connect")
    def _set_session_read_only(dbapi_connection: Any, connection_record: Any) -> None:
        cursor = dbapi_connection.cursor()
        try:
            cursor.execute("SET SESSION CHARACTERISTICS AS TRANSACTION READ ONLY")
        finally:
            cursor.close()
        dbapi_connection.commit()

    @event.listens_for(engine, "before_cursor_execute")
    def _refuse_writes(conn: Any, cursor: Any, statement: str, parameters: Any, context: Any,
                       executemany: bool) -> None:
        keyword = _first_keyword(statement)
        if keyword not in ALLOWED_KEYWORDS:
            log.critical("write attempt on source DB blocked",
                         extra={"keyword": keyword, "statement": statement[:200]})
            raise ReadOnlyViolation(f"Refusing non-read statement on source DB: {keyword or statement[:40]!r}")


def resolve_conference(candidate: Candidate, fallback: ConferenceRef,
                       override: ConferenceInfo | None = None) -> ConferenceInfo:
    """override -> taxonomy conference -> content source (unless generic) -> fallback."""
    if override is not None:
        return override
    year = None
    if candidate.event_date is not None:
        year = candidate.event_date.year
    elif candidate.published_at is not None:
        year = candidate.published_at.year

    def build(name: str, base_slug: str, source: str) -> ConferenceInfo | None:
        slug = base_slug
        if year is not None and str(year) not in slug:
            slug = f"{slug}-{year}"
        if not _SLUG_RE.match(slug) or len(slug) < 3:
            log.debug("conference slug rejected", extra={"conference_name": name, "slug": slug, "via": source})
            return None
        return ConferenceInfo(name=name, slug=slug, year=year, source=source)

    options: list[tuple[str, str, str]] = []
    if candidate.conference_name and candidate.conference_name.strip():
        name = candidate.conference_name.strip()
        options.append((name, slugify(name), "taxonomy"))
    if (candidate.source_name and candidate.source_name.strip()
            and (candidate.source_slug or "") not in GENERIC_SOURCE_SLUGS):
        name = candidate.source_name.strip()
        options.append((name, slugify(name), "content_source"))
    options.append((fallback.name, slugify(fallback.slug) or slugify(fallback.name), "fallback"))
    for name, slug, source in options:
        info = build(name, slug, source)
        if info is not None:
            log.debug("conference resolved", extra={"conference": info.slug, "via": source})
            return info
    return ConferenceInfo(name=fallback.name, slug=fallback.slug, year=year, source="fallback")


class TranscriptSource:
    """Read-only reader for candidate talks and their transcript text."""

    def __init__(self, url: str):
        url = _normalize_url(url)
        try:
            self.engine = create_engine(url, pool_pre_ping=True,
                                        execution_options={"postgresql_readonly": True})
        except Exception:  # dialect without postgresql_readonly support
            log.warning("postgresql_readonly unsupported; relying on session/guard read-only layers")
            self.engine = create_engine(url, pool_pre_ping=True)
        _install_guards(self.engine)
        self._sessions = sessionmaker(self.engine, autoflush=False, expire_on_commit=False)

    def _session(self) -> Session:
        return self._sessions()

    def ping(self) -> None:
        try:
            with self.engine.connect() as conn:
                conn.execute(text("SELECT 1")).scalar_one()
        except Exception as exc:
            log.error("source database unreachable", extra={"error": str(exc)})
            raise
        log.debug("source database reachable")

    # ── candidates ───────────────────────────────────────────────────────────

    def _candidate_query(self):
        conf = aliased(Taxonomy)
        best_text = func.coalesce(func.nullif(func.btrim(Transcript.corrected_text), ""), Transcript.raw_text)
        computed_wc = func.array_length(func.regexp_split_to_array(func.btrim(best_text), r"\s+"), 1)
        has_text = or_(func.length(func.btrim(func.coalesce(Transcript.corrected_text, ""))) > 0,
                       func.length(func.btrim(func.coalesce(Transcript.raw_text, ""))) > 0)
        return (
            select(
                ContentItem.id, ContentItem.title, ContentItem.description, ContentItem.url,
                ContentItem.published_at, ContentItem.event_date, ContentItem.technical_score,
                Transcript.id.label("transcript_id"), Transcript.version, Transcript.stt_model,
                func.coalesce(Transcript.word_count, computed_wc).label("word_count"),
                conf.name.label("conference_name"),
                ContentSource.name.label("source_name"), ContentSource.slug.label("source_slug"),
            )
            .join(Transcript, and_(Transcript.content_item_id == ContentItem.id, Transcript.is_current.is_(True)))
            .outerjoin(conf, and_(conf.id == ContentItem.event_id, conf.type == "conference"))
            .outerjoin(ContentSource, ContentSource.id == ContentItem.source_id)
            .where(has_text)
        )

    def _hydrate(self, session: Session, rows: list[Any]) -> list[Candidate]:
        if not rows:
            return []
        item_ids = [r.id for r in rows]
        transcript_ids = [r.transcript_id for r in rows]

        speakers: dict[str, list[str]] = {}
        for item_id, name in session.execute(
            select(ContentItemSpeaker.content_item_id, Speaker.name)
            .join(Speaker, Speaker.id == ContentItemSpeaker.speaker_id)
            .where(ContentItemSpeaker.content_item_id.in_(item_ids))
            .order_by(ContentItemSpeaker.content_item_id, Speaker.name)
        ):
            speakers.setdefault(str(item_id), []).append(name)

        summaries: dict[str, dict[str, str]] = {}
        for transcript_id, summary_type, content in session.execute(
            select(Summary.transcript_id, Summary.summary_type, Summary.content)
            .where(Summary.transcript_id.in_(transcript_ids))
            .order_by(Summary.transcript_id, Summary.summary_type)
        ):
            summaries.setdefault(str(transcript_id), {}).setdefault(summary_type, content)

        candidates = []
        for r in rows:
            by_type = summaries.get(str(r.transcript_id), {})
            summary = next((by_type[t] for t in SUMMARY_PREFERENCE if by_type.get(t)), None)
            if summary is None and by_type:
                summary = next(iter(by_type.values()))
            candidates.append(Candidate(
                content_item_id=str(r.id), transcript_id=str(r.transcript_id),
                transcript_version=r.version or 1, title=r.title, description=r.description, url=r.url,
                published_at=r.published_at, event_date=r.event_date, technical_score=r.technical_score,
                conference_name=r.conference_name, source_name=r.source_name, source_slug=r.source_slug,
                speakers=speakers.get(str(r.id), []), summary=summary, word_count=r.word_count,
                stt_model=r.stt_model,
            ))
        return candidates

    def list_candidates(self, exclude_ids: set[str], limit: int) -> list[Candidate]:
        with step("list_candidates", limit=limit, excluded=len(exclude_ids)):
            query = self._candidate_query().where(ContentItem.status == "transcribed")
            if exclude_ids:
                query = query.where(cast(ContentItem.id, String).not_in(sorted(exclude_ids)))
            query = query.order_by(ContentItem.technical_score.desc().nulls_last(),
                                   ContentItem.published_at.desc().nulls_last(),
                                   ContentItem.id).limit(limit)
            with self._session() as session:
                rows = list(session.execute(query))
                candidates = self._hydrate(session, rows)
        log.info("candidates listed", extra={"count": len(candidates), "excluded": len(exclude_ids), "limit": limit})
        return candidates

    def get_candidate(self, content_item_id: str) -> Candidate | None:
        query = self._candidate_query().where(cast(ContentItem.id, String) == content_item_id)
        with self._session() as session:
            rows = list(session.execute(query))
            candidates = self._hydrate(session, rows)
        if not candidates:
            log.warning("candidate not found or has no current transcript text",
                        extra={"content_item_id": content_item_id})
            return None
        log.info("candidate loaded", extra={"content_item_id": content_item_id})
        return candidates[0]

    # ── documents ────────────────────────────────────────────────────────────

    def load_document(self, candidate: Candidate, fallback: ConferenceRef,
                      conference_override: ConferenceInfo | None = None) -> TranscriptDoc:
        with step("load_document", content_item_id=candidate.content_item_id):
            with self._session() as session:
                row = session.execute(
                    select(Transcript.corrected_text, Transcript.raw_text)
                    .where(cast(Transcript.id, String) == candidate.transcript_id)
                ).first()
        corrected = (row.corrected_text if row else None) or ""
        raw = (row.raw_text if row else None) or ""
        if corrected.strip():
            text_value, field = corrected, "corrected_text"
        elif raw.strip():
            text_value, field = raw, "raw_text"
            log.warning("fell back to raw_text", extra={"content_item_id": candidate.content_item_id,
                                                         "transcript_id": candidate.transcript_id})
        else:
            log.error("transcript has no text", extra={"content_item_id": candidate.content_item_id})
            raise NoTextError(f"Transcript {candidate.transcript_id} has no corrected_text or raw_text")
        cleaned = clean_text(text_value)
        if not cleaned:
            raise NoTextError(f"Transcript {candidate.transcript_id} is empty after cleaning")
        conference = resolve_conference(candidate, fallback, conference_override)
        log.info("document loaded", extra={"content_item_id": candidate.content_item_id, "text_field": field,
                                           "chars": len(cleaned), "raw_chars": len(text_value),
                                           "conference": conference.slug})
        return TranscriptDoc(candidate=candidate, text=cleaned, text_field=field, conference=conference)

    def dispose(self) -> None:
        self.engine.dispose()
