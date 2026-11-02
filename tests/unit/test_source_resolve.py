from datetime import date, datetime, timezone

from aqg.config import ConferenceRef
from aqg.schemas.transcript import Candidate, ConferenceInfo
from aqg.source.queries import _first_keyword, resolve_conference

FALLBACK = ConferenceRef(name="Bitcoin Talks", slug="bitcoin-talks")


def _cand(**kw):
    return Candidate(content_item_id="c", transcript_id="t", title="T", **kw)


def test_precedence_and_year():
    override = ConferenceInfo(name="X Conf", slug="x-conf", source="cli")
    assert resolve_conference(_cand(conference_name="TABConf"), FALLBACK, override) is override
    conf = resolve_conference(_cand(conference_name="TABConf", event_date=date(2023, 10, 1),
                                    source_name="YT", source_slug="yt"), FALLBACK)
    assert (conf.slug, conf.year, conf.source) == ("tabconf-2023", 2023, "taxonomy")
    conf = resolve_conference(_cand(conference_name="Bitcoin 2024", event_date=date(2024, 5, 1)), FALLBACK)
    assert conf.slug == "bitcoin-2024"
    conf = resolve_conference(_cand(source_name="Chaincode Labs", source_slug="chaincode",
                                    published_at=datetime(2022, 1, 1, tzinfo=timezone.utc)), FALLBACK)
    assert (conf.slug, conf.source) == ("chaincode-labs-2022", "content_source")
    conf = resolve_conference(_cand(source_name="Manual Imports", source_slug="manual-imports"), FALLBACK)
    assert (conf.slug, conf.source, conf.year) == ("bitcoin-talks", "fallback", None)
    conf = resolve_conference(_cand(conference_name="!!", source_name="Misc", source_slug="misc"), FALLBACK)
    assert conf.source == "fallback"


def test_first_keyword():
    assert _first_keyword("  -- c\n select 1") == "SELECT"
    assert _first_keyword("(SELECT 1)") == "SELECT"
    assert _first_keyword("/* x */ UPDATE t") == "UPDATE"
