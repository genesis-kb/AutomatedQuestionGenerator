"""Load a transcript from a local file instead of Postgres.

Supported inputs:
  * fixture JSON (keys content_item / transcript / source / speakers) — real ids are kept,
  * transcription_engine whisper JSON ("text" and/or "segments"), with a sibling metadata_*.json,
  * Markdown (optional YAML front matter), plain text.
"""

from __future__ import annotations

import hashlib
import json
import re
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any

import yaml

from aqg.config import ConferenceRef
from aqg.logging_setup import get_logger
from aqg.schemas.transcript import Candidate, ConferenceInfo, TranscriptDoc
from aqg.source.cleaner import clean_text
from aqg.source.queries import NoTextError, resolve_conference

log = get_logger("source.file")

_FRONT_MATTER = re.compile(r"\A﻿?---[ \t]*\r?\n(.*?)\r?\n---[ \t]*(?:\r?\n|\Z)", re.DOTALL)


def _speaker_names(raw: Any) -> list[str]:
    names = []
    for s in raw or []:
        name = s.get("name") if isinstance(s, dict) else s
        if isinstance(name, str) and name.strip():
            names.append(name.strip())
    return names


def _parse_date(value: Any) -> datetime | None:
    if isinstance(value, datetime):
        return value if value.tzinfo else value.replace(tzinfo=timezone.utc)
    if isinstance(value, date):
        return datetime(value.year, value.month, value.day, tzinfo=timezone.utc)
    if isinstance(value, str) and value.strip():
        try:
            parsed = datetime.fromisoformat(value.strip().replace("Z", "+00:00"))
        except ValueError:
            return None
        return parsed if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)
    return None


def _title_from_filename(path: Path) -> str:
    return re.sub(r"[-_]+", " ", path.stem).strip() or path.name


def _file_ids(data: bytes) -> str:
    return "file-" + hashlib.sha256(data).hexdigest()[:16]


def _finish(path: Path, candidate: Candidate, text_value: str, field: str, fallback: ConferenceRef,
            override: ConferenceInfo | None) -> TranscriptDoc:
    cleaned = clean_text(text_value or "")
    if not cleaned:
        log.error("file has no transcript text", extra={"path": str(path)})
        raise NoTextError(f"No transcript text in {path}")
    conference = resolve_conference(candidate, fallback, override)
    log.info("document loaded from file", extra={"path": str(path), "content_item_id": candidate.content_item_id,
                                                 "text_field": field, "chars": len(cleaned),
                                                 "conference": conference.slug})
    return TranscriptDoc(candidate=candidate, text=cleaned, text_field=field, conference=conference)


def _from_fixture(path: Path, data: dict, fallback: ConferenceRef, override: ConferenceInfo | None) -> TranscriptDoc:
    item = data.get("content_item") or {}
    transcript = data.get("transcript") or {}
    source = data.get("source") or {}
    corrected = transcript.get("corrected_text") or ""
    raw = transcript.get("raw_text") or ""
    if corrected.strip():
        text_value, field = corrected, "corrected_text"
    else:
        text_value, field = raw, "raw_text"
        if raw.strip():
            log.warning("fell back to raw_text", extra={"path": str(path)})
    candidate = Candidate(
        content_item_id=str(item["id"]), transcript_id=str(transcript.get("id") or item["id"]),
        transcript_version=transcript.get("version") or 1, title=item.get("title") or _title_from_filename(path),
        description=item.get("description"), url=item.get("url"), published_at=item.get("published_at"),
        event_date=item.get("event_date"), technical_score=item.get("technical_score"),
        conference_name=(data.get("conference") or {}).get("name"),
        source_name=source.get("name"), source_slug=source.get("slug"),
        speakers=_speaker_names(data.get("speakers")), word_count=transcript.get("word_count"),
        stt_model=transcript.get("stt_model"),
    )
    return _finish(path, candidate, text_value, field, fallback, override)


def _sibling_metadata(path: Path) -> dict:
    for meta_path in sorted(path.parent.glob("metadata_*.json"), reverse=True):
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            log.warning("unreadable metadata file", extra={"path": str(meta_path), "error": str(exc)})
            continue
        if isinstance(meta, dict):
            log.debug("using sibling metadata", extra={"path": str(meta_path)})
            return meta
    return {}


def _from_whisper(path: Path, raw_bytes: bytes, data: dict, fallback: ConferenceRef,
                  override: ConferenceInfo | None) -> TranscriptDoc:
    text_value = data.get("text") or ""
    if not text_value.strip() and isinstance(data.get("segments"), list):
        text_value = " ".join(str(s.get("text", "")).strip() for s in data["segments"] if isinstance(s, dict))
    meta = _sibling_metadata(path)
    file_id = _file_ids(raw_bytes)
    candidate = Candidate(
        content_item_id=file_id, transcript_id=file_id, transcript_version=1,
        title=meta.get("title") or _title_from_filename(path),
        description=meta.get("description") if isinstance(meta.get("description"), str) else None,
        url=next((u for u in (meta.get("media"), meta.get("source_file")) if isinstance(u, str) and u), None),
        published_at=_parse_date(meta.get("date")), speakers=_speaker_names(meta.get("speakers")),
    )
    return _finish(path, candidate, text_value, "file", fallback, override)


def _from_text(path: Path, raw_bytes: bytes, fallback: ConferenceRef, override: ConferenceInfo | None) -> TranscriptDoc:
    content = raw_bytes.decode("utf-8", errors="replace")
    front: dict = {}
    if path.suffix.lower() in (".md", ".markdown"):
        match = _FRONT_MATTER.match(content)
        if match:
            try:
                parsed = yaml.safe_load(match.group(1))
                front = parsed if isinstance(parsed, dict) else {}
            except yaml.YAMLError as exc:
                log.warning("unparseable front matter", extra={"path": str(path), "error": str(exc)})
            content = content[match.end():]
    file_id = _file_ids(raw_bytes)
    title = front.get("title")
    candidate = Candidate(
        content_item_id=file_id, transcript_id=file_id, transcript_version=1,
        title=str(title).strip() if title else _title_from_filename(path),
        description=front.get("description") if isinstance(front.get("description"), str) else None,
        url=front.get("url") if isinstance(front.get("url"), str) else None,
        published_at=_parse_date(front.get("date")), speakers=_speaker_names(front.get("speakers")),
        conference_name=front.get("conference") if isinstance(front.get("conference"), str) else None,
    )
    return _finish(path, candidate, content, "file", fallback, override)


def load_from_file(path: Path, fallback: ConferenceRef, override: ConferenceInfo | None = None) -> TranscriptDoc:
    path = Path(path)
    raw_bytes = path.read_bytes()
    if path.suffix.lower() == ".json":
        data = json.loads(raw_bytes.decode("utf-8"))
        if isinstance(data, dict) and "content_item" in data and "transcript" in data:
            return _from_fixture(path, data, fallback, override)
        if isinstance(data, dict) and ("text" in data or "segments" in data):
            return _from_whisper(path, raw_bytes, data, fallback, override)
        raise ValueError(f"Unrecognised JSON transcript format: {path}")
    return _from_text(path, raw_bytes, fallback, override)
