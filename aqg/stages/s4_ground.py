"""S4: ground the spec in web references (Tavily, allowlisted domains) and fact-check it against them."""

from __future__ import annotations

from aqg import prompts
from aqg.langs import typemap
from aqg.logging_setup import get_logger, step
from aqg.schemas import CodingSpec, Correction, Grounding, ReconcileResult, Reference, TestVector
from aqg.search.base import wrap_untrusted
from aqg.stages.context import RunContext

log = get_logger("s4")


def collect_references(ctx: RunContext, spec: CodingSpec) -> list[Reference]:
    cfg = ctx.settings.search
    queries = spec.search_queries[:4] or [spec.title_hint]
    refs: list[Reference] = []
    for query in queries:
        if ctx.web is not None:
            with step("web_search", query=query):
                refs.extend(ctx.web.search(query, cfg.max_web_results))
    seen: set[tuple[str, str]] = set()
    unique: list[Reference] = []
    for ref in sorted(refs, key=lambda r: {"test_vector": 0, "spec": 1, "explainer": 2}[r.kind]):
        key = (ref.url, ref.passage[:120])
        if key not in seen:
            seen.add(key)
            unique.append(ref)
    chosen = unique[: cfg.max_passages]
    log.info("references collected", extra={"queries": queries, "found": len(refs), "kept": len(chosen),
                                           "urls": [r.url for r in chosen]})
    return chosen


def _validate_vectors(spec: CodingSpec, result: ReconcileResult, refs: list[Reference]) -> list[TestVector]:
    vectors: list[TestVector] = []
    for v in result.test_vectors:
        if not 0 <= v.reference_index < len(refs):
            log.warning("test vector without a valid reference dropped", extra={"vector_args": v.args})
            continue
        try:
            stdin = typemap.normalize_args(spec, v.args)
            expected = typemap.format_value(spec.return_type, typemap.parse_line(spec.return_type, v.expected.strip()))
        except ValueError as exc:
            log.warning("test vector does not fit the signature, dropped", extra={"error": str(exc), "vector_args": v.args})
            continue
        source = refs[v.reference_index]
        if v.expected.strip().lower() not in source.passage.lower():
            log.warning("test vector output not literally in its passage, dropped", extra={"expected": v.expected})
            continue
        vectors.append(TestVector(args=stdin.rstrip("\n").split("\n"), expected=expected, source_url=source.url))
    return vectors


def ground(ctx: RunContext, spec: CodingSpec) -> Grounding:
    refs = collect_references(ctx, spec)
    if not refs:
        log.warning("no references found; grounding = transcript_only (stricter validation applies)")
        return Grounding(grounding="transcript_only")

    prompt = prompts.render(
        "s4_reconcile", spec=spec.model_dump_json(indent=2),
        facts="\n".join(f"- {f}" for f in spec.uncertain_facts) or "- (none listed)",
        passages=wrap_untrusted(refs), signature=typemap.signature(spec, "python3"),
    )
    result = ctx.llm.generate("s4_reconcile", prompts.system_prompt(), prompt, ReconcileResult)

    corrections: list[Correction] = []
    backed_by: set[str] = set()
    for fact in result.facts:
        ref = refs[fact.reference_index] if 0 <= fact.reference_index < len(refs) else None
        if fact.status in ("confirmed", "corrected") and ref is None:
            log.warning("fact marked verified without a reference; treated as unverified", extra={"fact": fact.fact})
            fact.status, fact.corrected_value = "unverified", ""
            continue
        if ref is not None and fact.status != "unverified":
            backed_by.add(ref.kind)
        if fact.status == "corrected" and ref is not None:
            corrections.append(Correction(fact=fact.fact, to=fact.corrected_value, source=ref.url))
            log.warning("transcript fact corrected by reference", extra={
                "fact": fact.fact, "to": fact.corrected_value, "source": ref.url})

    vectors = _validate_vectors(spec, result, refs)
    if vectors:
        backed_by.add("test_vector")
    level = ("spec_backed" if backed_by & {"spec", "test_vector"}
             else "explainer_backed" if "explainer" in backed_by else "transcript_only")
    grounding = Grounding(references=refs, facts=result.facts, corrections=corrections,
                          test_vectors=vectors, grounding=level)
    log.info("grounding done", extra={
        "grounding": level, "confirmed": sum(f.status == "confirmed" for f in result.facts),
        "corrected": len(corrections), "unverified": sum(f.status == "unverified" for f in result.facts),
        "test_vectors": len(vectors)})
    if level == "transcript_only":
        log.warning("no fact could be verified; grounding = transcript_only")
    return grounding


def facts_text(grounding: Grounding) -> str:
    lines = []
    for f in grounding.facts:
        if f.status == "corrected":
            lines.append(f"- CORRECTED: {f.fact} -> {f.corrected_value}")
        elif f.status == "confirmed":
            lines.append(f"- confirmed: {f.fact}")
    return "\n".join(lines) or "- (no verified facts; rely on the spec)"
