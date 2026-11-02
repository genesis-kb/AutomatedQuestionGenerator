"""S6: the validation gate. A problem is exported only if every check passes."""

from __future__ import annotations

from aqg.logging_setup import get_logger, step
from aqg.sandbox.judge_protocol import JudgeRequest
from aqg.schemas import CheckResult, CodingSpec, Grounding, TestCase, ValidationReport
from aqg.stages.context import RunContext
from aqg.stages.s5_generate import ProblemBundle, program, static_problems
from aqg.textutil import slugify

log = get_logger("s6")
MAX_FIELD = 1_048_576


def _run_all(ctx: RunContext, language: str, source: str, tests: list[TestCase]):
    cfg = ctx.settings.pipeline
    reqs = [JudgeRequest(language=language, source=source, stdin=t.input, time_limit=cfg.time_limit_s,
                         memory_limit=cfg.memory_limit_mb, expected_output=t.expected_output) for t in tests]
    return ctx.get_sandbox().run_many(reqs)


def _first_failures(tests: list[TestCase], results, limit: int = 3) -> str:
    lines = []
    for t, r in zip(tests, results):
        if r.verdict != "AC":
            got = r.stdout.strip()[:200]
            lines.append(f"input {t.input.strip()!r}: verdict {r.verdict}; expected {t.expected_output[:200]!r}; "
                         f"got {got!r}; {r.checker_msg[:200]} {r.stderr[:600]}".strip())
            if len(lines) >= limit:
                break
    return "\n".join(lines)


def validate(ctx: RunContext, spec: CodingSpec, bundle: ProblemBundle, grounding: Grounding) -> ValidationReport:
    cfg = ctx.settings.pipeline
    checks: list[CheckResult] = []
    per_language: dict[str, dict] = {}
    tests = bundle.tests

    def add(name: str, passed: bool, detail: str = "", language: str | None = None) -> None:
        checks.append(CheckResult(name=name, passed=passed, detail=detail, language=language))
        if not passed:
            log.warning("validation check failed", extra={"check": name, "language": language, "detail": detail[:1000]})

    # 1. title / identity
    slug = slugify(bundle.draft.title)
    title_ok = 3 <= len(slug) <= 100 and len(bundle.draft.title) <= 200
    dup = ctx.ledger.title_used(bundle.draft.title)
    add("title_valid_and_unique", title_ok and not dup,
        f"slug {slug!r}" + (" was already generated in an earlier run" if dup else "")
        + ("" if title_ok else " has invalid length"))

    # 6. test distribution (cheap, before sandbox work)
    samples = [t for t in tests if t.is_sample]
    hidden = len(tests) - len(samples)
    contiguous = [t.order_index for t in tests] == list(range(len(tests)))
    sizes_ok = all(len(t.input) < MAX_FIELD and len(t.expected_output) < MAX_FIELD for t in tests)
    add("test_distribution",
        1 <= len(samples) <= cfg.max_samples and hidden >= cfg.min_hidden_tests and contiguous and sizes_ok,
        f"{len(samples)} samples, {hidden} hidden (need >= {cfg.min_hidden_tests}), contiguous={contiguous}, "
        f"sizes_ok={sizes_ok}")

    # 7. spec test vectors included
    vectors_in = sum(t.origin == "vector" for t in tests)
    add("spec_vectors_included", vectors_in == len(grounding.test_vectors),
        f"{vectors_in}/{len(grounding.test_vectors)} vectors present")

    # 8. examples in the description equal the sample tests
    missing = [t for t in samples if t.input.rstrip() not in bundle.description or t.expected_output not in bundle.description]
    add("examples_match_samples", not missing, f"{len(missing)} sample(s) not rendered in the description")

    # 10. transcript-only problems may not hard-code protocol constants
    if grounding.grounding == "transcript_only":
        risky = [c.name for c in spec.constants if c.source == "transcript"]
        add("transcript_only_no_constants", not risky, f"unverified constants: {risky}" if risky else "")

    if not tests:
        add("has_tests", False, "no tests could be generated")
        return ValidationReport(passed=False, checks=checks, per_language=per_language)

    time_budget_ms = cfg.time_limit_s * 1000 * 0.5
    mem_budget_kb = cfg.memory_limit_mb * 1024 * 0.5
    for lang in bundle.languages:
        stats: dict = {}
        helpers = bundle.helpers.get(lang, [])
        ref = bundle.references.get(lang, "")

        # 9. static rules (std only, no I/O, no main)
        problems = static_problems(spec, lang, ref)
        add("static_rules", not problems, "; ".join(problems), lang)

        # 2 + 3. reference passes every test twice, identically (for js/rust this is the cross-language check)
        with step("reference_runs", language=lang):
            source = program(lang, helpers, ref, bundle.wrappers[lang])
            first = _run_all(ctx, lang, source, tests)
            second = _run_all(ctx, lang, source, tests)
        passed_first = sum(r.verdict == "AC" for r in first)
        deterministic = all(a.stdout == b.stdout for a, b in zip(first, second))
        stats.update(reference_ac=f"{passed_first}/{len(tests)}", deterministic=deterministic)
        add("reference_passes_all_tests" if lang == "python3" else "cross_language_agreement",
            passed_first == len(tests) and all(r.verdict == "AC" for r in second),
            _first_failures(tests, first), lang)
        add("deterministic", deterministic, "" if deterministic else "two runs produced different output", lang)

        # 5. resource headroom
        max_ms = max((r.time_ms for r in first), default=0)
        max_kb = max((r.memory_kb for r in first), default=0)
        stats.update(max_time_ms=max_ms, max_memory_kb=max_kb)
        # hello-bitcoin's run.py reports getrusage(RUSAGE_CHILDREN), which for Rust includes the rustc
        # compile (~170 MB). The real limit (RLIMIT_AS) only applies to the executed binary, so the
        # memory figure is not meaningful for Rust; time is still checked.
        mem_ok = lang == "rust" or max_kb < mem_budget_kb
        add("resource_headroom", max_ms < time_budget_ms and mem_ok,
            f"max {max_ms} ms (budget {time_budget_ms:.0f}), {max_kb} KB (budget {mem_budget_kb:.0f}"
            + (", not enforced for rust: includes rustc" if lang == "rust" else "") + ")", lang)

        # 4. starter compiles but does not solve the problem
        with step("starter_runs", language=lang):
            # starters already contain their helpers, so none are prepended here
            starter_source = program(lang, [], bundle.starters[lang], bundle.wrappers[lang])
            starter_results = _run_all(ctx, lang, starter_source, tests)
        ce = [r for r in starter_results if r.verdict == "CE"]
        all_ac = all(r.verdict == "AC" for r in starter_results)
        stats.update(starter_verdicts=sorted({r.verdict for r in starter_results}))
        add("starter_compiles", not ce, ce[0].stderr[:800] if ce else "", lang)
        add("starter_fails", not all_ac, "starter already passes every test" if all_ac else "", lang)
        per_language[lang] = stats

    report = ValidationReport(passed=all(c.passed for c in checks), checks=checks, per_language=per_language)
    log.info("validation finished", extra={"passed": report.passed, "failed_checks": [
        f"{c.name}[{c.language}]" if c.language else c.name for c in report.failures()]})
    return report


def feedback_for(report: ValidationReport) -> dict[str, str]:
    """Route failures to the sub-step that can fix them: 'statement', 'tests', or a language."""
    routed: dict[str, list[str]] = {}
    for c in report.failures():
        if c.name == "title_valid_and_unique":
            routed.setdefault("statement", []).append(c.detail)
        elif c.name in ("test_distribution", "has_tests"):
            routed.setdefault("tests", []).append(c.detail)
        elif c.language and c.name in ("static_rules", "reference_passes_all_tests", "cross_language_agreement",
                                       "deterministic", "resource_headroom"):
            routed.setdefault(c.language, []).append(f"{c.name}: {c.detail}")
        elif c.name in ("starter_compiles", "starter_fails"):
            routed.setdefault("generator_bug", []).append(f"{c.name}[{c.language}]: {c.detail}")
        else:
            routed.setdefault("other", []).append(f"{c.name}: {c.detail}")
    return {k: "\n".join(v) for k, v in routed.items()}

