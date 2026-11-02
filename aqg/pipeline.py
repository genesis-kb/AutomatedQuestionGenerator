"""Pipeline orchestrator: S1 -> S7 with checkpoints, repair loops, ledger and repo guards."""

from __future__ import annotations

import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from pydantic import BaseModel

from aqg import prompts, protect
from aqg.config import Settings
from aqg.langs import typemap
from aqg.ledger import Ledger
from aqg.logging_setup import content_item, get_logger, run_log, stage
from aqg.sandbox.judge_protocol import helper_code
from aqg.schemas import (
    Chunk, CodingSpec, Concept, ConceptList, ConferenceInfo, Grounding, ProblemDraft, TestPlan, TranscriptDoc,
    ValidationReport,
)
from aqg.source.chunker import chunk_text
from aqg.stages import s2_triage, s3_spec, s4_ground, s5_generate, s6_validate, s7_export
from aqg.stages.context import RunContext, checkpoint, save_json, save_model
from aqg.textutil import slugify

log = get_logger("pipeline")


class ChunkList(BaseModel):
    chunks: list[Chunk]


def new_run_id() -> str:
    return f"{datetime.now(timezone.utc):%Y%m%dT%H%M%SZ}-{uuid.uuid4().hex[:6]}"


# ── setup ─────────────────────────────────────────────────────────────────────


def _init_search(ctx: RunContext, use_web: bool) -> None:
    from aqg.search.cache import JsonCache
    from aqg.search.tavily import TavilySearch

    s = ctx.settings
    if use_web:
        ctx.web = TavilySearch(s.secrets.tavily_api_key, s.search.allow_domains,
                               JsonCache(s.data_dir / "cache" / "tavily"), s.search.passage_chars)
    else:
        log.info("web search disabled for this run")


def _select_documents(ctx: RunContext, *, content_item_id: str | None, from_file: Path | None, limit: int,
                      conference_override: ConferenceInfo | None) -> list[TranscriptDoc]:
    s = ctx.settings
    fallback = s.pipeline.fallback_conference
    if from_file:
        from aqg.source.file_source import load_from_file

        return [load_from_file(from_file, fallback, conference_override)]

    from aqg.source.queries import NoTextError, TranscriptSource

    if not s.secrets.te_database_url:
        raise RuntimeError("TE_DATABASE_URL is not set (see .env.example)")
    ctx.source = TranscriptSource(s.secrets.te_database_url)
    ctx.source.ping()
    if content_item_id:
        cand = ctx.source.get_candidate(content_item_id)
        if cand is None:
            raise RuntimeError(f"content item {content_item_id} not found or not transcribed")
        return [ctx.source.load_document(cand, fallback, conference_override)]

    used = ctx.ledger.used_content_item_ids()
    candidates = ctx.source.list_candidates(exclude_ids=used, limit=s.pipeline.candidate_batch)
    log.info("candidates listed", extra={"candidates": len(candidates), "excluded_already_used": len(used)})
    scored: list[tuple[float, TranscriptDoc]] = []
    with stage("S2a triage", candidates=len(candidates)):
        for cand in candidates:
            with content_item(cand.content_item_id):
                try:
                    doc = ctx.source.load_document(cand, fallback, conference_override)
                except NoTextError:
                    log.warning("talk has no text; skipped")
                    ctx.ledger.mark_source(cand.content_item_id, cand.transcript_version, "", "skipped:no_text",
                                           ctx.run_id)
                    continue
                score = s2_triage.triage(ctx, cand, doc.text)
                if score.score >= s.pipeline.min_codeability:
                    scored.append((score.score + 0.1 * len(cand.keyword_hits), doc))
                else:
                    log.info("talk not codeable at triage", extra={"score": score.score})
                    ctx.ledger.mark_source(cand.content_item_id, cand.transcript_version, "",
                                           "skipped:not_codeable", ctx.run_id)
    scored.sort(key=lambda x: x[0], reverse=True)
    return [doc for _, doc in scored[:limit]]


# ── one concept → one problem ─────────────────────────────────────────────────


def _produce_problem(ctx: RunContext, doc: TranscriptDoc, concept: Concept, cdir: Path
                     ) -> tuple[CodingSpec, Grounding, s5_generate.ProblemBundle | None, ValidationReport]:
    cfg = ctx.settings.pipeline
    with stage("S3 spec", concept=concept.concept):
        spec = checkpoint(cdir / "s3_spec.json", CodingSpec, lambda: s3_spec.build_spec(ctx, doc, concept),
                          force=ctx.force)
    with stage("S4 ground"):
        grounding = checkpoint(cdir / "s4_grounding.json", Grounding, lambda: s4_ground.ground(ctx, spec),
                               force=ctx.force)

    langs = ctx.languages
    helpers = {lang: s5_generate.effective_helpers(spec, lang) for lang in langs}
    wrappers = {lang: typemap.wrapper(spec, lang) for lang in langs}
    starters = {lang: typemap.starter(spec, lang, helper_code(lang, helpers[lang]) if helpers[lang] else "")
                for lang in langs}

    with stage("S5a statement"):
        draft = checkpoint(cdir / "s5_statement.json", ProblemDraft,
                           lambda: s5_generate.write_statement(ctx, doc, spec, grounding), force=ctx.force)
    with stage("S5d test plan"):
        plan = checkpoint(cdir / "s5_test_plan.json", TestPlan, lambda: s5_generate.plan_tests(ctx, spec),
                          force=ctx.force)

    refs: dict[str, str] = {}
    feedback: dict[str, str] = {}
    bundle: s5_generate.ProblemBundle | None = None
    report = ValidationReport(passed=False)
    for loop in range(cfg.max_repair_loops + 1):
        with stage("S5 generate", loop=loop, feedback=sorted(feedback)):
            if "statement" in feedback:
                draft = s5_generate.write_statement(ctx, doc, spec, grounding, feedback["statement"])
                save_model(cdir / "s5_statement.json", draft)
            if "tests" in feedback:
                plan = s5_generate.plan_tests(ctx, spec, feedback["tests"])
                save_model(cdir / "s5_test_plan.json", plan)
            if "python3" not in refs or "python3" in feedback:
                refs["python3"] = s5_generate.write_reference(ctx, spec, grounding, "python3",
                                                              feedback.get("python3", ""), refs.get("python3", ""))
            static = s5_generate.static_problems(spec, "python3", refs["python3"])
            if static:
                feedback = {"python3": "Static check failures:\n" + "\n".join(static)}
                log.warning("python reference violates static rules; repairing", extra={"problems": static})
                continue
            tests, issues = s5_generate.materialize_tests(ctx, spec, plan, grounding, refs["python3"])
            ref_issues = [i for i in issues if i.startswith("python reference")]
            if ref_issues and len(tests) < cfg.min_hidden_tests + 1:
                feedback = {"python3": "\n".join(ref_issues[:8])}
                continue
            for lang in langs[1:]:
                if lang not in refs or lang in feedback:
                    refs[lang] = s5_generate.write_reference(ctx, spec, grounding, lang, feedback.get(lang, ""),
                                                             refs.get(lang, ""))
            samples = [t for t in tests if t.is_sample]
            description = s5_generate.render_description(spec, draft, samples, langs, cfg.time_limit_s,
                                                         cfg.memory_limit_mb)
            bundle = s5_generate.ProblemBundle(
                draft=draft, description=description, languages=langs, helpers=helpers, references=dict(refs),
                starters=starters, wrappers=wrappers, tests=tests, test_issues=issues)
            save_model(cdir / "s5_bundle.json", bundle)

        with stage("S6 validate", loop=loop):
            report = s6_validate.validate(ctx, spec, bundle, grounding)
            report.repair_loops = loop
            save_model(cdir / "s6_report.json", report)
        if report.passed:
            break
        feedback = s6_validate.feedback_for(report)
        if "generator_bug" in feedback:
            log.error("starter/wrapper generated by aqg failed validation (aqg bug, not repairable by the LLM)",
                      extra={"detail": feedback["generator_bug"][:1500]})
            break
        test_issues = [t for t in bundle.test_issues if not t.startswith("python reference")]
        if test_issues:
            feedback["tests"] = (feedback.get("tests", "") + "\nConcrete problems:\n" + "\n".join(test_issues[:8])).strip()
        if not (set(feedback) & {"statement", "tests", *langs}):
            log.error("validation failed for reasons the repair loop cannot fix", extra={"feedback": feedback})
            break
        log.warning("repair loop", extra={"loop": loop + 1, "targets": sorted(feedback)})
    if bundle is None:
        # every loop ended before validation (reference never ran cleanly): say why instead of "no bundle"
        from aqg.schemas import CheckResult

        detail = "\n".join(f"{k}: {v}" for k, v in feedback.items())[:2000]
        report = ValidationReport(passed=False, repair_loops=cfg.max_repair_loops, checks=[
            CheckResult(name="python_reference_runs", passed=False, language="python3", detail=detail)])
        log.error("python reference never ran cleanly; concept rejected", extra={"detail": detail[:1000]})
    return spec, grounding, bundle, report


# ── run ───────────────────────────────────────────────────────────────────────


def run(settings: Settings, *, content_item_id: str | None = None, from_file: Path | None = None, limit: int = 1,
        languages: list[str] | None = None, conference_override: ConferenceInfo | None = None,
        force: bool = False, run_id: str | None = None, use_web: bool = True, llm_overrides: dict | None = None,
        max_problems: int | None = None) -> dict[str, Any]:
    from aqg.llm import LLMRouter

    run_id = run_id or new_run_id()
    run_dir = settings.runs_dir / run_id
    langs = settings.normalized_languages(languages)
    params = {"content_item_id": content_item_id, "from_file": str(from_file) if from_file else None,
              "limit": limit, "languages": langs, "force": force, "use_web": use_web, "max_problems": max_problems,
              "conference_override": conference_override.model_dump() if conference_override else None}
    started = time.monotonic()
    summary: dict[str, Any] = {"run_id": run_id, "run_dir": str(run_dir), "languages": langs, "talks": [],
                               "accepted": [], "rejected": [], "status": "failed"}
    ledger = Ledger(settings.data_dir / "ledger.sqlite")
    with run_log(run_dir, run_id) as run_log_path:
        summary["run_log"] = str(run_log_path)
        log.info("run started", extra={"params": params})
        before = protect.snapshot(settings.protected_repos)
        save_json(run_dir / "params.json", params)
        ledger.start_run(run_id, params)
        ctx = RunContext(settings=settings, run_id=run_id, run_dir=run_dir, ledger=ledger,
                         llm=LLMRouter(settings, ledger=ledger, run_id=run_id, overrides=llm_overrides),
                         languages=langs, force=force)
        try:
            with stage("S1 retrieve"):
                docs = _select_documents(ctx, content_item_id=content_item_id, from_file=from_file, limit=limit,
                                         conference_override=conference_override)
            log.info("talks selected", extra={"talks": len(docs)})
            if docs:
                _init_search(ctx, use_web)
            for doc in docs:
                summary["talks"].append(_process_talk(ctx, doc, summary, max_problems))
            summary["status"] = "completed"
        except Exception as exc:
            log.error("run failed", exc_info=True)
            summary["error"] = f"{type(exc).__name__}: {exc}"
        finally:
            ctx.close()
            after = protect.snapshot(settings.protected_repos)
            diffs = protect.compare(before, after)
            summary["protected_repos_unchanged"] = not diffs
            if diffs:
                log.critical("protected repository changed during the run", extra={"differences": diffs})
                summary["protected_repo_changes"] = diffs
            summary["duration_s"] = round(time.monotonic() - started, 1)
            summary["llm"] = ledger.llm_totals(run_id)
            summary["models"] = dict(ctx.llm.models_used)
            manifest = {**summary, "params": params, "prompt_hashes": prompts.all_hashes(),
                        "finished_at": datetime.now(timezone.utc).isoformat()}
            summary["manifest"] = str(s7_export.write_manifest(ctx, manifest))
            ledger.finish_run(run_id, summary["status"], summary)
            log.info("run finished", extra={"status": summary["status"], "accepted": len(summary["accepted"]),
                                            "rejected": len(summary["rejected"]), "duration_s": summary["duration_s"]})
    return summary


def _process_talk(ctx: RunContext, doc: TranscriptDoc, summary: dict, max_problems: int | None) -> dict:
    cfg = ctx.settings.pipeline
    cand = doc.candidate
    talk_dir = ctx.run_dir / slugify(cand.content_item_id)[:40]
    talk = {"content_item_id": cand.content_item_id, "title": cand.title, "conference": doc.conference.slug,
            "text_field": doc.text_field, "concepts": [], "accepted": [], "rejected": [], "status": "processing"}
    wanted = max_problems or cfg.max_problems_per_talk
    with content_item(cand.content_item_id):
        save_model(talk_dir / "s1_document.json", doc)
        chunks = chunk_text(doc.text, cfg.chunk_tokens, cfg.chunk_overlap_tokens)
        save_model(talk_dir / "s1_chunks.json", ChunkList(chunks=chunks))
        log.info("talk loaded", extra={"title": cand.title, "chars": len(doc.text), "chunks": len(chunks),
                                       "conference": doc.conference.slug, "text_field": doc.text_field})
        with stage("S2b extract concepts", chunks=len(chunks)):
            concepts = checkpoint(talk_dir / "s2_concepts.json", ConceptList,
                                  lambda: ConceptList(concepts=s2_triage.extract_concepts(ctx, doc, chunks)),
                                  force=ctx.force).concepts
        talk["concepts"] = [c.concept for c in concepts]
        if not concepts:
            log.info("no codeable concept in talk")
            ctx.ledger.mark_source(cand.content_item_id, cand.transcript_version, "", "skipped:not_codeable",
                                   ctx.run_id)
            talk["status"] = "skipped:not_codeable"
            return talk

        # Each concept costs >= 5 LLM calls, so try only the best-ranked ones: one spare beyond what we want.
        for idx, concept in enumerate(concepts[: wanted + 1]):
            if len(talk["accepted"]) >= wanted:
                break
            if ctx.ledger.concept_used(cand.content_item_id, concept.concept_hash) and not ctx.force:
                log.info("concept already used in an earlier run; skipped", extra={"concept": concept.concept})
                continue
            cdir = talk_dir / f"c{idx}-{slugify(concept.concept)[:40]}"
            try:
                spec, grounding, bundle, report = _produce_problem(ctx, doc, concept, cdir)
            except Exception as exc:
                log.error("concept failed", exc_info=True, extra={"concept": concept.concept})
                path = s7_export.write_rejected(ctx, slugify(concept.concept), {
                    "concept": concept.model_dump(), "error": f"{type(exc).__name__}: {exc}"})
                # "failed" (not "rejected"): an error such as a rate limit or network drop says nothing about the
                # concept, so a later run may try it again.
                ctx.ledger.mark_source(cand.content_item_id, cand.transcript_version, concept.concept_hash,
                                       "failed", ctx.run_id)
                entry = {"concept": concept.concept, "reason": f"{type(exc).__name__}: {exc}"[:300], "path": str(path)}
                talk["rejected"].append(entry)
                summary["rejected"].append(entry)
                continue

            if bundle is not None and report.passed:
                with stage("S7 export"):
                    export = s7_export.build_export(ctx, doc, concept, spec, grounding, bundle, report,
                                                    prompts.all_hashes())
                    path = s7_export.write_accepted(ctx, export)
                ctx.ledger.mark_source(cand.content_item_id, cand.transcript_version, concept.concept_hash,
                                       "accepted", ctx.run_id, problem_title=bundle.draft.title)
                entry = {"title": bundle.draft.title, "path": str(path), "tests": len(bundle.tests),
                         "grounding": grounding.grounding, "repair_loops": report.repair_loops}
                talk["accepted"].append(entry)
                summary["accepted"].append(entry)
            else:
                reasons = [f"{c.name}" + (f"[{c.language}]" if c.language else "") for c in report.failures()]
                path = s7_export.write_rejected(ctx, slugify(bundle.draft.title if bundle else concept.concept), {
                    "concept": concept.model_dump(), "spec": spec.model_dump(), "grounding": grounding.model_dump(mode="json"),
                    "bundle": bundle.model_dump() if bundle else None, "validation": report.model_dump()})
                ctx.ledger.mark_source(cand.content_item_id, cand.transcript_version, concept.concept_hash,
                                       "rejected", ctx.run_id, problem_title=bundle.draft.title if bundle else None)
                entry = {"concept": concept.concept, "reason": ", ".join(reasons) or "no bundle", "path": str(path)}
                talk["rejected"].append(entry)
                summary["rejected"].append(entry)
        talk["status"] = "accepted" if talk["accepted"] else "rejected"
        ctx.ledger.record_stage(ctx.run_id, "talk", talk["status"], f"{len(talk['accepted'])} accepted",
                                cand.content_item_id)
    return talk
