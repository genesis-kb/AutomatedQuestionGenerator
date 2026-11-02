"""S3: turn a concept into a language-neutral CodingSpec."""

from __future__ import annotations

from aqg import prompts
from aqg.logging_setup import get_logger
from aqg.schemas import CodingSpec, Concept, TranscriptDoc
from aqg.stages.context import RunContext

log = get_logger("s3")
CONTEXT_CHARS = 2500


def concept_context(doc: TranscriptDoc, concept: Concept) -> str:
    center = concept.char_start if concept.char_start is not None else 0
    start = max(0, center - CONTEXT_CHARS)
    end = min(len(doc.text), center + CONTEXT_CHARS)
    return doc.text[start:end]


def build_spec(ctx: RunContext, doc: TranscriptDoc, concept: Concept) -> CodingSpec:
    prompt = prompts.render(
        "s3_spec", concept=concept.concept, why=concept.why_codeable, excerpt=concept.excerpt,
        context=concept_context(doc, concept),
    )
    spec = ctx.llm.generate("s3_spec", prompts.system_prompt(), prompt, CodingSpec)
    if not spec.search_queries:
        spec.search_queries = [spec.title_hint]
        log.warning("spec had no search queries; using the title hint", extra={"title_hint": spec.title_hint})
    log.info("spec built", extra={
        "function": spec.function_name, "params": [f"{p.name}:{p.type}" for p in spec.params],
        "returns": spec.return_type, "uncertain_facts": len(spec.uncertain_facts), "helpers": spec.needs_helpers,
    })
    return spec
