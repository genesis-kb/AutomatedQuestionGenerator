"""S7: assemble the hello-bitcoin import JSON and write it atomically."""

from __future__ import annotations

from pathlib import Path

from aqg.logging_setup import get_logger
from aqg.schemas import (
    Authoring, CodingSpec, Concept, ExportConference, ExportProblem, ExportTestCase, Grounding, ProblemExport,
    Provenance, TranscriptDoc, ValidationReport,
)
from aqg.stages.context import RunContext, save_json, save_model
from aqg.stages.s5_generate import ProblemBundle
from aqg.textutil import slugify

log = get_logger("s7")


def build_export(ctx: RunContext, doc: TranscriptDoc, concept: Concept, spec: CodingSpec, grounding: Grounding,
                 bundle: ProblemBundle, report: ValidationReport, prompt_hashes: dict[str, str]) -> ProblemExport:
    cfg = ctx.settings.pipeline
    cand = doc.candidate
    return ProblemExport(
        problem=ExportProblem(
            title=bundle.draft.title,
            expected_id=slugify(bundle.draft.title),
            description=bundle.description,
            starter_code={lang: bundle.starters[lang] for lang in bundle.languages},
            wrapper_code={lang: bundle.wrappers[lang] for lang in bundle.languages},
            time_limit=cfg.time_limit_s,
            memory_limit=cfg.memory_limit_mb,
        ),
        conference=ExportConference(name=doc.conference.name, slug=doc.conference.slug, year=doc.conference.year),
        test_cases=[ExportTestCase(input=t.input, expected_output=t.expected_output, is_sample=t.is_sample,
                                   points=t.points, order_index=t.order_index) for t in bundle.tests],
        authoring=Authoring(
            difficulty=bundle.draft.difficulty, tags=bundle.draft.tags,
            reference_solution={lang: bundle.references[lang] for lang in bundle.languages},
            helpers_used=sorted({h for hs in bundle.helpers.values() for h in hs}),
            spec=spec.model_dump(),
        ),
        provenance=Provenance(
            content_item_id=cand.content_item_id, transcript_id=cand.transcript_id,
            transcript_version=cand.transcript_version, title=cand.title, url=cand.url, stt_model=cand.stt_model,
            text_field=doc.text_field, concept=concept.concept,
            excerpts=[{"text": concept.excerpt, "char_start": concept.char_start,
                       "verified": concept.excerpt_verified, "score": concept.excerpt_score}],
            references=[r.model_dump(mode="json") for r in grounding.references],
            corrections=[c.model_dump() for c in grounding.corrections],
            grounding=grounding.grounding, models=dict(ctx.llm.models_used), prompt_hashes=prompt_hashes,
            run_id=ctx.run_id,
        ),
        validation=report,
    )


def write_accepted(ctx: RunContext, export: ProblemExport) -> Path:
    path = ctx.settings.out_dir / "problems" / f"{export.problem.expected_id}.json"
    save_model(path, export)
    log.info("problem exported", extra={"path": str(path), "title": export.problem.title,
                                        "tests": len(export.test_cases)})
    return path


def write_rejected(ctx: RunContext, slug: str, payload: dict) -> Path:
    path = ctx.settings.out_dir / "rejected" / f"{ctx.run_id}-{slug or 'unnamed'}.json"
    save_json(path, payload)
    log.warning("problem rejected", extra={"path": str(path)})
    return path


def write_manifest(ctx: RunContext, manifest: dict) -> Path:
    path = ctx.settings.out_dir / "manifests" / f"{ctx.run_id}.json"
    save_json(path, manifest)
    log.info("manifest written", extra={"path": str(path)})
    return path
