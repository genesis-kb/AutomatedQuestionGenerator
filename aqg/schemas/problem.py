"""S5-S7 contracts: drafts from the LLM, executed tests, validation, final export."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Literal

from pydantic import BaseModel, Field

# ── LLM outputs ───────────────────────────────────────────────────────────────


class ProblemDraft(BaseModel):
    title: str = Field(max_length=120, description="Short, specific, unique problem title")
    background: str = Field(description="Markdown: 1-2 paragraphs of context from the talk")
    task: str = Field(description="Markdown: exactly what the function must compute, including byte order / encoding rules")
    notes: list[str] = Field(default_factory=list, description="Extra rules, hints or edge cases worth stating")
    tags: list[str] = Field(default_factory=list, max_length=6)
    difficulty: Literal["easy", "medium", "hard"] = "medium"


class ReferenceCode(BaseModel):
    code: str = Field(description="Complete function definition (plus any private helper functions). No I/O, no main.")


class TestPlan(BaseModel):
    samples: list[dict[str, str]] = Field(
        description="1-3 small illustrative inputs; each maps EVERY parameter name to its argument string")
    edge_cases: list[dict[str, str]] = Field(
        description="8-15 boundary / tricky inputs; each maps EVERY parameter name to its argument string")
    generator_code: str = Field(
        description="Python: def generate(rng) -> list returning one random valid argument list (Python values, "
                    "one per parameter). Use only the `random.Random` instance passed in; standard library only.")


# ── executed / validated ──────────────────────────────────────────────────────


class TestCase(BaseModel):
    input: str
    expected_output: str
    is_sample: bool = False
    points: int = 1
    order_index: int = 0
    origin: Literal["sample", "edge", "random", "vector"] = "edge"


class CheckResult(BaseModel):
    name: str
    passed: bool
    detail: str = ""
    language: str | None = None


class ValidationReport(BaseModel):
    passed: bool
    checks: list[CheckResult] = Field(default_factory=list)
    repair_loops: int = 0
    per_language: dict[str, dict] = Field(default_factory=dict)

    def failures(self) -> list[CheckResult]:
        return [c for c in self.checks if not c.passed]


# ── export (maps to hello-bitcoin) ────────────────────────────────────────────


class ExportProblem(BaseModel):
    title: str = Field(max_length=200)
    expected_id: str = Field(max_length=100)
    source_type: Literal["conference"] = "conference"
    order_index: int = 0
    description: str
    starter_code: dict[str, str]
    wrapper_code: dict[str, str]
    checker_code: str = ""
    time_limit: float = 2.0
    memory_limit: int = 256
    is_published: Literal[False] = False


class ExportConference(BaseModel):
    name: str = Field(max_length=200)
    slug: str = Field(min_length=3, max_length=200, pattern=r"^[a-z0-9_-]+$")
    year: int | None = None


class ExportTestCase(BaseModel):
    input: str = Field(max_length=1_048_576)
    expected_output: str = Field(max_length=1_048_576)
    is_sample: bool
    points: int = Field(ge=0)
    order_index: int = Field(ge=0)


class Authoring(BaseModel):
    difficulty: str
    tags: list[str]
    reference_solution: dict[str, str]
    helpers_used: list[str] = Field(default_factory=list)
    spec: dict = Field(default_factory=dict)


class Provenance(BaseModel):
    content_item_id: str
    transcript_id: str
    transcript_version: int
    title: str
    url: str | None = None
    stt_model: str | None = None
    text_field: str
    concept: str
    excerpts: list[dict] = Field(default_factory=list)
    references: list[dict] = Field(default_factory=list)
    corrections: list[dict] = Field(default_factory=list)
    grounding: str
    models: dict[str, str] = Field(default_factory=dict)
    prompt_hashes: dict[str, str] = Field(default_factory=dict)
    run_id: str
    generated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class ProblemExport(BaseModel):
    schema_version: Literal["2.0"] = "2.0"
    problem: ExportProblem
    conference: ExportConference
    test_cases: list[ExportTestCase]
    authoring: Authoring
    provenance: Provenance
    validation: ValidationReport
