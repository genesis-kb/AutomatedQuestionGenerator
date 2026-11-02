"""Deterministic transcript cleanup: whitespace, non-speech markers, timestamps, filler words."""

from __future__ import annotations

import re

_UNICODE_SPACES = re.compile(r"[\u00a0\u1680\u2000-\u200a\u202f\u205f\u3000\t\f\v]")
_ZERO_WIDTH = re.compile(r"[\u200b\u200c\u200d\u2060\ufeff]")
_MARKER_WORDS = (
    r"music(?:\s+playing)?|applause|laughter|laughs|laughing|inaudible|crosstalk|silence|noise|"
    r"background\s+noise|cheering|coughs?|coughing|indistinct|unintelligible|blank_audio|pause"
)
_MARKERS = re.compile(rf"[\[(]\s*(?:{_MARKER_WORDS})\s*[\])]", re.IGNORECASE)
_TS = r"\d{1,2}:\d{2}(?::\d{2})?(?:[.,]\d{1,3})?"
_BRACKETED_TS = re.compile(rf"[\[(]\s*{_TS}(?:\s*(?:-->|-)\s*{_TS})?\s*[\])]")
_LINE_TS = re.compile(rf"(?m)^[ ]*{_TS}(?:\s*-->\s*{_TS})?(?=\s|$)")
_FILLERS = re.compile(r"(?i)\b(?:uh|um|uhm|erm)\b(?!['’\-])[ ]*,?[ ]*")
_SPACE_BEFORE_PUNCT = re.compile(r"[ ]+([,.!?;:])")
_DOUBLE_COMMA = re.compile(r",(?:[ ]*,)+")
_LEADING_PUNCT = re.compile(r"(?m)^[ ]*[,;:][ ]*")
_PUNCT_AFTER_STOP = re.compile(r"([.!?])[ ]*,")
_SPACES = re.compile(r"[ ]{2,}")
_BLANK_LINES = re.compile(r"\n{3,}")


def clean_text(text: str) -> str:
    """Return cleaned text; paragraph breaks (blank lines) are preserved."""
    if not text:
        return ""
    text = text.replace("\r\n", "\n").replace("\r", "\n").replace("\u2028", "\n").replace("\u2029", "\n\n")
    text = _ZERO_WIDTH.sub("", text)
    text = _UNICODE_SPACES.sub(" ", text)
    text = _MARKERS.sub(" ", text)
    text = _BRACKETED_TS.sub(" ", text)
    text = _LINE_TS.sub(" ", text)
    text = _FILLERS.sub(" ", text)
    text = _DOUBLE_COMMA.sub(",", text)
    text = _SPACE_BEFORE_PUNCT.sub(r"\1", text)
    text = _PUNCT_AFTER_STOP.sub(r"\1", text)
    text = _LEADING_PUNCT.sub("", text)
    text = _SPACES.sub(" ", text)
    text = "\n".join(line.strip() for line in text.split("\n"))
    text = _BLANK_LINES.sub("\n\n", text)
    return text.strip()
