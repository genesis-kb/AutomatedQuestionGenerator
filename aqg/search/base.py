"""Search client protocol, BM25 passage selection, and untrusted-data wrapping for prompts."""

from __future__ import annotations

import re
from typing import Protocol

from rank_bm25 import BM25Okapi

from aqg.schemas.grounding import Reference

UNTRUSTED_OPEN = "<<<UNTRUSTED_DATA>>>"
UNTRUSTED_CLOSE = "<<<END_UNTRUSTED_DATA>>>"

_TOKEN_RE = re.compile(r"[a-z0-9]+")
_PARA_RE = re.compile(r"\n\s*\n")
_SENT_RE = re.compile(r"(?<=[.!?])\s+")


class SearchClient(Protocol):
    def search(self, query: str, max_results: int) -> list[Reference]: ...


def tokenize(text: str) -> list[str]:
    return _TOKEN_RE.findall(text.lower())


def _units(text: str, chars: int) -> list[str]:
    """Paragraphs; paragraphs longer than `chars` are split into sentences, then hard-split."""
    units: list[str] = []
    for para in _PARA_RE.split(text):
        para = para.strip()
        if not para:
            continue
        if len(para) <= chars:
            units.append(para)
            continue
        for sent in _SENT_RE.split(para):
            sent = sent.strip()
            while len(sent) > chars:
                units.append(sent[:chars])
                sent = sent[chars:].strip()
            if sent:
                units.append(sent)
    return units


def split_windows(text: str, chars: int = 800) -> list[str]:
    """Greedily pack paragraph/sentence units into windows of at most ~`chars` characters."""
    chars = max(1, chars)
    windows: list[str] = []
    current: list[str] = []
    size = 0
    for unit in _units(text, chars):
        if current and size + len(unit) + 2 > chars:
            windows.append("\n\n".join(current))
            current, size = [], 0
        current.append(unit)
        size += len(unit) + 2
    if current:
        windows.append("\n\n".join(current))
    return [w for w in windows if w.strip()]


def select_passages(text: str, query: str, chars: int = 800, n: int = 2) -> list[str]:
    """Top-n windows of `text` by BM25 score against `query`, best first."""
    windows = split_windows(text or "", chars)
    if not windows or n <= 0:
        return []
    query_tokens = tokenize(query)
    if not query_tokens:
        return windows[:n]
    bm25 = BM25Okapi([tokenize(w) or [""] for w in windows])
    scores = bm25.get_scores(query_tokens)
    order = sorted(range(len(windows)), key=lambda i: (-scores[i], i))
    return [windows[i] for i in order[:n]]


def _strip_delimiters(text: str) -> str:
    previous = None
    while previous != text:  # repeat so nested/overlapping fragments cannot reassemble a delimiter
        previous = text
        text = text.replace(UNTRUSTED_CLOSE, "").replace(UNTRUSTED_OPEN, "")
    return text


def wrap_untrusted(refs: list[Reference]) -> str:
    """Render references as numbered, delimited data blocks for LLM prompts (data, never instructions)."""
    blocks = []
    for i, ref in enumerate(refs):
        title = _strip_delimiters(ref.title or "").replace("\n", " ").strip()
        passage = _strip_delimiters(ref.passage)
        blocks.append(f"[{i}] {title} — {ref.url}\n{UNTRUSTED_OPEN}\n{passage}\n{UNTRUSTED_CLOSE}")
    return "\n\n".join(blocks)
