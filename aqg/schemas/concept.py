"""S2 contracts: triage scores and codeable concepts."""

from __future__ import annotations

from pydantic import BaseModel, Field


class TriageScore(BaseModel):
    score: int = Field(ge=0, le=5, description="0 = nothing codeable, 5 = clearly contains a deterministic algorithm")
    reason: str


class ConceptDraft(BaseModel):
    """What the LLM returns per chunk."""

    concept: str = Field(description="Short name of the deterministic mechanism, e.g. 'CompactSize integer encoding'")
    excerpt: str = Field(description="Verbatim quote (<=300 chars) from the chunk that mentions it")
    why_codeable: str
    codeability: int = Field(ge=0, le=5)
    requires_unsupported: bool = Field(
        default=False,
        description="true ONLY if solving it needs elliptic-curve math, integers above 64 bits, network access or "
                    "external libraries; false for serialization, parsing, hashing, encoding and arithmetic")


class ChunkConcepts(BaseModel):
    concepts: list[ConceptDraft] = Field(default_factory=list)


class Concept(ConceptDraft):
    chunk_id: int
    char_start: int | None = None
    excerpt_verified: bool = False
    excerpt_score: float = 0.0
    concept_hash: str = ""


class ConceptList(BaseModel):
    concepts: list[Concept] = Field(default_factory=list)
