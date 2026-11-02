import os

import pytest
from sqlalchemy import text

from aqg.config import ConferenceRef
from aqg.source.queries import ReadOnlyViolation, TranscriptSource, resolve_conference

pytestmark = pytest.mark.postgres

URL = os.getenv("TE_DATABASE_URL") or "postgresql://bitcoin:bitcoin@127.0.0.1:5434/transcription_engine"
FORGE_ID = "cd2ea32f-5eb8-4861-b4fc-859bc89eb874"
FALLBACK = ConferenceRef(name="Bitcoin Talks", slug="bitcoin-talks")


@pytest.fixture(scope="module")
def source():
    src = TranscriptSource(URL)
    try:
        src.ping()
    except Exception as exc:
        pytest.skip(f"transcription_engine Postgres unreachable: {exc}")
    yield src
    src.dispose()


@pytest.fixture(scope="module")
def forge(source):
    candidate = next((c for c in source.list_candidates(set(), 50) if c.content_item_id == FORGE_ID), None)
    if candidate is None:
        pytest.skip("Forge talk not present in the live DB")
    return candidate


def test_list_candidates(source, forge):
    assert forge.title.startswith("Raw Bitcoin Transaction Capstone")
    assert forge.source_slug == "manual-imports"
    assert len(forge.speakers) == 2
    assert forge.word_count and forge.word_count > 100
    assert FORGE_ID not in {c.content_item_id for c in source.list_candidates({FORGE_ID}, 50)}
    assert source.get_candidate(FORGE_ID).transcript_id == forge.transcript_id
    assert source.get_candidate("00000000-0000-0000-0000-000000000000") is None


def test_load_document(source, forge):
    doc = source.load_document(forge, FALLBACK)
    assert doc.text_field == "raw_text"
    assert len(doc.text) > 1000


def test_resolve_conference_fallback(forge):
    conf = resolve_conference(forge, FALLBACK)
    assert conf.source == "fallback" and conf.name == "Bitcoin Talks" and conf.slug == "bitcoin-talks"


@pytest.mark.parametrize("statement", [
    "UPDATE content_items SET title = 'x' WHERE false",
    "INSERT INTO speakers (name, slug) VALUES ('x', 'x')",
    "/* sneaky */ DELETE FROM summaries WHERE false",
])
def test_writes_refused(source, statement):
    with pytest.raises(ReadOnlyViolation):
        with source.engine.begin() as conn:
            conn.execute(text(statement))


def test_session_is_read_only(source):
    with source.engine.connect() as conn:
        assert conn.execute(text("SHOW transaction_read_only")).scalar() == "on"
