import json
from pathlib import Path

from aqg.source.chunker import chunk_text

FIXTURE = Path(__file__).resolve().parents[1] / "fixtures" / "transcripts" / "forge_raw_tx_capstone.json"


def _check_invariants(text, chunks):
    assert [c.chunk_id for c in chunks] == list(range(len(chunks)))
    for c in chunks:
        assert text[c.char_start:c.char_end] == c.text
        assert c.char_end > c.char_start
    assert chunks[0].char_start == 0
    assert chunks[-1].char_end == len(text)
    for prev, nxt in zip(chunks, chunks[1:]):
        assert nxt.char_start <= prev.char_end  # contiguous coverage
        assert nxt.char_start > prev.char_start


def _synthetic():
    parts = []
    for i in range(40):
        speaker = "Speaker 1" if i % 2 == 0 else "ALICE"
        sentences = " ".join(f"Sentence {i}.{j} talks about varints and little-endian bytes." for j in range(8))
        parts.append(f"{speaker}: {sentences}")
    return "\n\n".join(parts)


def test_empty_and_short():
    assert chunk_text("") == []
    chunks = chunk_text("Just a short talk.", max_tokens=100)
    assert len(chunks) == 1 and chunks[0].text == "Just a short talk."


def test_synthetic_long_text_invariants_and_overlap():
    text = _synthetic()
    chunks = chunk_text(text, max_tokens=500, overlap_tokens=50)
    assert len(chunks) > 3
    _check_invariants(text, chunks)
    for prev, nxt in zip(chunks, chunks[1:]):
        assert nxt.char_start < prev.char_end  # overlap present
        assert len(prev.text) <= 500 * 4
    # Breaks prefer speaker turns: every non-final chunk ends right before a speaker line.
    for c in chunks[:-1]:
        following = text[c.char_end:c.char_end + 12]
        assert following.startswith(("Speaker 1:", "ALICE:"))


def test_hard_cut_without_structure():
    text = "x" * 10_000
    chunks = chunk_text(text, max_tokens=100, overlap_tokens=10)
    _check_invariants(text, chunks)
    assert all(len(c.text) <= 400 for c in chunks)


def test_no_overlap():
    text = _synthetic()
    chunks = chunk_text(text, max_tokens=400, overlap_tokens=0)
    _check_invariants(text, chunks)
    assert "".join(c.text for c in chunks) == text


def test_fixture_raw_text():
    raw = json.loads(FIXTURE.read_text())["transcript"]["raw_text"]
    assert len(chunk_text(raw)) == 1
    chunks = chunk_text(raw, max_tokens=200, overlap_tokens=30)
    assert len(chunks) > 3
    _check_invariants(raw, chunks)
    for prev, nxt in zip(chunks, chunks[1:]):
        assert nxt.char_start < prev.char_end
