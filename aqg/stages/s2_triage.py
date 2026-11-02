"""S2: talk-level triage and chunk-level concept extraction (map/reduce)."""

from __future__ import annotations

import re

from rapidfuzz import fuzz

from aqg import prompts
from aqg.logging_setup import get_logger, step
from aqg.schemas import Candidate, Chunk, ChunkConcepts, Concept, TranscriptDoc, TriageScore
from aqg.stages.context import RunContext
from aqg.textutil import sha256_hex, slugify

log = get_logger("s2")

CODEABLE_KEYWORDS = (
    "serializ", "deserializ", "varint", "compactsize", "compact size", "little endian", "little-endian", "big endian",
    "byte order", "hash", "sha256", "sha-256", "ripemd", "hash160", "double sha", "merkle", "script", "opcode",
    "op_", "sighash", "signature hash", "taproot", "tapscript", "schnorr", "segwit", "witness", "bech32", "base58",
    "checksum", "address", "transaction id", "txid", "wtxid", "outpoint", "utxo", "input", "output", "locktime",
    "sequence", "fee", "feerate", "weight", "vbyte", "virtual size", "psbt", "descriptor", "derivation", "bip32",
    "bip 32", "hd wallet", "xpub", "block header", "nonce", "difficulty", "target", "bits", "coinbase", "halving",
    "subsidy", "timelock", "htlc", "encoding", "hex", "bip",
)


def keyword_hits(*texts: str | None) -> list[str]:
    haystack = " ".join(t for t in texts if t).lower()
    return sorted({kw for kw in CODEABLE_KEYWORDS if kw in haystack})


def triage(ctx: RunContext, candidate: Candidate, excerpt: str) -> TriageScore:
    """LLM score 0-5 for 'contains a codeable mechanism'."""
    candidate.keyword_hits = keyword_hits(candidate.title, candidate.description, candidate.summary, excerpt)
    prompt = prompts.render(
        "s2_triage", title=candidate.title, description=candidate.description or "(none)",
        summary=candidate.summary or "(none)", excerpt=excerpt[:3000],
    )
    score = ctx.llm.generate("s2_triage", prompts.system_prompt(), prompt, TriageScore)
    candidate.triage_score = float(score.score)
    log.info("talk triaged", extra={"content_item_id": candidate.content_item_id, "score": score.score,
                                    "reason": score.reason, "keyword_hits": len(candidate.keyword_hits)})
    return score


def _verify_excerpt(excerpt: str, chunk: Chunk, doc_text: str, threshold: int) -> tuple[bool, float, int | None]:
    if not excerpt.strip():
        return False, 0.0, None
    align = fuzz.partial_ratio_alignment(excerpt.lower(), chunk.text.lower())
    if align is None:
        return False, 0.0, None
    score = float(align.score)
    start = chunk.char_start + align.dest_start
    return score >= threshold, score, start


def extract_concepts(ctx: RunContext, doc: TranscriptDoc, chunks: list[Chunk]) -> list[Concept]:
    """Map over chunks, verify excerpts against the source text, dedupe and rank."""
    cfg = ctx.settings.pipeline
    found: list[Concept] = []
    for chunk in chunks:
        with step("extract_chunk", chunk_id=chunk.chunk_id, chars=len(chunk.text)):
            prompt = prompts.render("s2_extract", title=doc.candidate.title, chunk=chunk.text)
            result = ctx.llm.generate("s2_extract", prompts.system_prompt(), prompt, ChunkConcepts)
        for draft in result.concepts:
            ok, score, start = _verify_excerpt(draft.excerpt, chunk, doc.text, cfg.excerpt_match_threshold)
            concept = Concept(
                **draft.model_dump(), chunk_id=chunk.chunk_id, char_start=start, excerpt_verified=ok,
                excerpt_score=score,
                concept_hash=sha256_hex(f"{doc.candidate.content_item_id}:{slugify(draft.concept)}")[:16],
            )
            if not ok:
                log.warning("excerpt not found verbatim in chunk", extra={
                    "concept": draft.concept, "excerpt_score": round(score, 1), "chunk_id": chunk.chunk_id})
            found.append(concept)
    log.info("concepts extracted", extra={"raw_concepts": len(found), "chunks": len(chunks)})
    return rank_concepts(found, cfg.min_codeability)


def rank_concepts(concepts: list[Concept], min_codeability: int) -> list[Concept]:
    kept: list[Concept] = []
    for c in sorted(concepts, key=lambda c: (c.codeability, c.excerpt_score), reverse=True):
        reasons = []
        if c.codeability < min_codeability:
            reasons.append(f"codeability {c.codeability} < {min_codeability}")
        if c.requires_unsupported:
            reasons.append("needs non-standard libraries / big-int curve math")
        if c.excerpt_score < 60:
            reasons.append(f"excerpt unsupported by transcript (score {c.excerpt_score:.0f})")
        if reasons:
            log.info("concept dropped", extra={"concept": c.concept, "reasons": reasons})
            continue
        name = re.sub(r"\s+", " ", c.concept.lower()).strip()
        if any(fuzz.token_set_ratio(name, k.concept.lower()) >= 85 for k in kept):
            log.debug("concept deduplicated", extra={"concept": c.concept})
            continue
        kept.append(c)
    log.info("concepts ranked", extra={"kept": [c.concept for c in kept]})
    return kept
