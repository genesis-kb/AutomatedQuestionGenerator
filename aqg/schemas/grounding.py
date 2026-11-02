"""S4 contracts: references, fact checks, test vectors."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Literal

from pydantic import BaseModel, Field


class Reference(BaseModel):
    url: str
    title: str = ""
    kind: Literal["spec", "test_vector", "explainer"] = "explainer"
    passage: str
    provider: str = "tavily"
    retrieved_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class FactCheck(BaseModel):
    fact: str
    status: Literal["confirmed", "corrected", "unverified"]
    corrected_value: str = ""
    reference_index: int = Field(default=-1, description="Index into the numbered passages, -1 if none")


class VectorDraft(BaseModel):
    args: list[str] = Field(description="One string per function parameter, in parameter order, exactly as it would appear on its stdin line")
    expected: str = Field(description="Expected output line(s) exactly as the function result would be printed")
    reference_index: int = -1


class ReconcileResult(BaseModel):
    facts: list[FactCheck] = Field(default_factory=list)
    test_vectors: list[VectorDraft] = Field(default_factory=list)


class TestVector(BaseModel):
    args: list[str]
    expected: str
    source_url: str


class Correction(BaseModel):
    fact: str
    to: str
    source: str


class Grounding(BaseModel):
    references: list[Reference] = Field(default_factory=list)
    facts: list[FactCheck] = Field(default_factory=list)
    corrections: list[Correction] = Field(default_factory=list)
    test_vectors: list[TestVector] = Field(default_factory=list)
    grounding: Literal["spec_backed", "explainer_backed", "transcript_only"] = "transcript_only"
