import json
from pathlib import Path

import pytest

from aqg.config import ConferenceRef
from aqg.schemas.transcript import ConferenceInfo
from aqg.source.file_source import load_from_file
from aqg.source.queries import NoTextError

ROOT = Path(__file__).resolve().parents[2]
FIXTURE = ROOT / "tests" / "fixtures" / "transcripts" / "forge_raw_tx_capstone.json"
WHISPER_DIR = ROOT.parent / "transcription_engine" / "metadata" / "forge-college" / \
    "raw-bitcoin-transaction-capstone-forge-college"
FALLBACK = ConferenceRef(name="Bitcoin Talks", slug="bitcoin-talks")


def test_fixture_json():
    doc = load_from_file(FIXTURE, FALLBACK)
    data = json.loads(FIXTURE.read_text())
    assert doc.candidate.content_item_id == data["content_item"]["id"]
    assert doc.candidate.transcript_id == data["transcript"]["id"]
    assert doc.candidate.speakers == ["Name1", "Name2"]
    assert doc.text_field == "raw_text"
    assert len(doc.text) > 1000
    assert doc.conference.source == "fallback" and doc.conference.slug == "bitcoin-talks"


def test_fixture_json_with_override():
    override = ConferenceInfo(name="Forge College", slug="forge-college", source="cli")
    assert load_from_file(FIXTURE, FALLBACK, override).conference == override


def test_markdown_front_matter(tmp_path):
    md = tmp_path / "my-talk.md"
    md.write_text("---\ntitle: Taproot Deep Dive\ndate: 2023-09-01\n---\n\n[Music] Um, today we cover Taproot.\n\n"
                  "Second paragraph about Schnorr.\n")
    doc = load_from_file(md, FALLBACK)
    assert doc.candidate.title == "Taproot Deep Dive"
    assert doc.candidate.content_item_id.startswith("file-") and len(doc.candidate.content_item_id) == 21
    assert doc.candidate.transcript_id == doc.candidate.content_item_id
    assert doc.text == "today we cover Taproot.\n\nSecond paragraph about Schnorr."
    assert doc.text_field == "file"
    assert doc.conference.slug == "bitcoin-talks-2023" and doc.conference.year == 2023


def test_txt_and_empty(tmp_path):
    txt = tmp_path / "segwit_talk.txt"
    txt.write_text("Segwit moves witness data.")
    doc = load_from_file(txt, FALLBACK)
    assert doc.candidate.title == "segwit talk"
    empty = tmp_path / "empty.txt"
    empty.write_text("  [music]  ")
    with pytest.raises(NoTextError):
        load_from_file(empty, FALLBACK)


def test_real_whisper_json():
    files = sorted(WHISPER_DIR.glob("whisper_*.json")) if WHISPER_DIR.exists() else []
    if not files:
        pytest.skip("transcription_engine whisper output not present")
    doc = load_from_file(files[-1], FALLBACK)
    assert doc.candidate.title == "Raw Bitcoin Transaction Capstone — Forge College"
    assert doc.candidate.url and "youtube" in doc.candidate.url
    assert doc.text_field == "file"
    assert len(doc.text) > 1000
