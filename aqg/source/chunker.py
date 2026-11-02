"""Structure-aware chunking with overlap (tokens ~= chars / 4).

Invariant: for every chunk, ``text[chunk.char_start:chunk.char_end] == chunk.text``; chunks are
numbered 0..n-1 and together cover the whole text.
"""

from __future__ import annotations

import re

from aqg.schemas.transcript import Chunk

CHARS_PER_TOKEN = 4

# A speaker turn starts a line: "Speaker 1:", "SPEAKER_00:", "Alice:", "Pieter Wuille:".
_SPEAKER_TURN = re.compile(r"(?m)^(?=[ \t]*(?:(?i:speaker)[ _]?\w{1,3}|[A-Z][A-Za-z.'\-]*(?:[ ][A-Z][A-Za-z.'\-]*){0,3})[ \t]*:)")
_PARAGRAPH = re.compile(r"\n[ \t]*\n\s*")
_SENTENCE_END = re.compile(r"[.!?…][\"'”’)\]]*\s+")
_WHITESPACE = re.compile(r"\s+")


def _last_break(pattern: re.Pattern[str], text: str, lo: int, hi: int, *, use_start: bool) -> int | None:
    """Last break position in (lo, hi]; a break is match.start() or match.end()."""
    best = None
    for m in pattern.finditer(text, lo, hi):
        pos = m.start() if use_start else m.end()
        if lo < pos <= hi:
            best = pos
    return best


def _find_end(text: str, start: int, max_chars: int) -> int:
    hard = start + max_chars
    lo = start + max_chars // 2
    for pattern, use_start in ((_SPEAKER_TURN, True), (_PARAGRAPH, False), (_SENTENCE_END, False),
                               (_WHITESPACE, False)):
        pos = _last_break(pattern, text, lo, hard, use_start=use_start)
        if pos is not None:
            return pos
    return hard


def _snap_start(text: str, candidate: int, end: int) -> int:
    """Move an overlap start forward to the next sentence start, else the next word start."""
    for pattern in (_SENTENCE_END, _WHITESPACE):
        m = pattern.search(text, candidate, end)
        if m and m.end() < end:
            return m.end()
    return candidate


def chunk_text(text: str, max_tokens: int = 4000, overlap_tokens: int = 300) -> list[Chunk]:
    if not text:
        return []
    n = len(text)
    max_chars = max(1, max_tokens * CHARS_PER_TOKEN)
    overlap_chars = max(0, overlap_tokens * CHARS_PER_TOKEN)
    if n <= max_chars:
        return [Chunk(chunk_id=0, text=text, char_start=0, char_end=n)]

    chunks: list[Chunk] = []
    start = 0
    while True:
        end = n if start + max_chars >= n else _find_end(text, start, max_chars)
        chunks.append(Chunk(chunk_id=len(chunks), text=text[start:end], char_start=start, char_end=end))
        if end >= n:
            break
        if overlap_chars:
            candidate = max(start + 1, end - min(overlap_chars, (end - start) // 2))
            next_start = _snap_start(text, candidate, end)
        else:
            next_start = end
        start = max(next_start, start + 1)
    return chunks
