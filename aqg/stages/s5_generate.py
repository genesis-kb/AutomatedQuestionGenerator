"""S5: generate statement, reference solutions and tests; compute expected outputs by execution.

The LLM writes only: the statement prose, the reference function body per language,
and test *inputs* (plus a random-input generator). Wrappers and starters are generated
deterministically from the spec (aqg/langs/typemap.py) and expected outputs come from
running the Python reference inside hello-bitcoin's runner image.
"""

from __future__ import annotations

import json

from pydantic import BaseModel, Field

from aqg import prompts
from aqg.langs import typemap
from aqg.langs.static_checks import check_solution, strip_code_fences
from aqg.logging_setup import get_logger, step
from aqg.sandbox.judge_protocol import JudgeRequest, full_source, helper_code
from aqg.schemas import CodingSpec, Grounding, ProblemDraft, ReferenceCode, TestCase, TestPlan, TranscriptDoc
from aqg.stages.context import RunContext
from aqg.stages.s4_ground import facts_text

log = get_logger("s5")

LANGUAGE_NAMES = {"python3": "Python 3", "javascript": "JavaScript (Node.js 20)", "rust": "Rust (2021 edition)"}
LANGUAGE_RULES = {
    "python3": "Python 3.11, standard library only. Keep the exact signature including type hints.",
    "javascript": ("Node.js 20, CommonJS. Use a plain `function name(...) { ... }` declaration. u64 parameters arrive "
                   "as BigInt (return BigInt or Number for u64). Only `require('crypto')` is allowed; no `import`, "
                   "no `fs`, no console output."),
    "rust": ("Rust 2021 compiled with plain `rustc` - NO crates, NO Cargo, std only. Parameters are owned values "
             "exactly as in the signature. No `fn main`. Avoid integer overflow (use checked/wrapping ops or wider types)."),
}
HELPER_API = {
    "python3": {"sha256": "sha256(data: bytes) -> bytes", "ripemd160": "ripemd160(data: bytes) -> bytes",
                "hex": "hex_decode(s: str) -> bytes, hex_encode(data: bytes) -> str"},
    "javascript": {"sha256": "sha256(buf) -> Buffer", "ripemd160": "ripemd160(buf) -> Buffer",
                   "hex": "hexDecode(s) -> Buffer, hexEncode(buf) -> string"},
    "rust": {"sha256": "fn sha256(data: &[u8]) -> Vec<u8>", "ripemd160": "fn ripemd160(data: &[u8]) -> Vec<u8>",
             "hex": "fn hex_decode(s: &str) -> Vec<u8>, fn hex_encode(data: &[u8]) -> String"},
}
HELPER_NAMES = {
    "python3": {"sha256": ["sha256"], "ripemd160": ["ripemd160"], "hex": ["hex_decode", "hex_encode"]},
    "javascript": {"sha256": ["sha256"], "ripemd160": ["ripemd160"], "hex": ["hexDecode", "hexEncode"]},
    "rust": {"sha256": ["sha256"], "ripemd160": ["ripemd160"], "hex": ["hex_decode", "hex_encode"]},
}


class ProblemBundle(BaseModel):
    draft: ProblemDraft
    description: str = ""
    languages: list[str]
    helpers: dict[str, list[str]] = Field(default_factory=dict)
    references: dict[str, str] = Field(default_factory=dict)
    starters: dict[str, str] = Field(default_factory=dict)
    wrappers: dict[str, str] = Field(default_factory=dict)
    tests: list[TestCase] = Field(default_factory=list)
    test_issues: list[str] = Field(default_factory=list)


def effective_helpers(spec: CodingSpec, language: str) -> list[str]:
    helpers = set(spec.needs_helpers)
    uses_str = any("str" in t for t in [spec.return_type, *(p.type for p in spec.params)])
    if language == "rust" and (helpers & {"sha256", "ripemd160"} or uses_str):
        helpers.add("hex")  # Rust std has no hex codec; harmless if unused
    return sorted(helpers)


def program(language: str, helpers: list[str], solution: str, wrapper: str) -> str:
    """helpers + solution + hidden wrapper, exactly as the judge will assemble it."""
    user = solution if not helpers else f"{helper_code(language, helpers)}\n\n\n{solution}"
    return full_source(user, wrapper, language)


# ── statement ─────────────────────────────────────────────────────────────────


def write_statement(ctx: RunContext, doc: TranscriptDoc, spec: CodingSpec, grounding: Grounding,
                    feedback: str = "") -> ProblemDraft:
    prompt = prompts.render(
        "s5_statement", spec=spec.model_dump_json(indent=2), facts=facts_text(grounding),
        talk_title=doc.candidate.title, conference=doc.conference.name,
    )
    if feedback:
        prompt += f"\n\nYour previous title/statement was rejected: {feedback}\nWrite a different, more specific title."
    draft = ctx.llm.generate("s5_statement", prompts.system_prompt(), prompt, ProblemDraft)
    log.info("statement written", extra={"title": draft.title, "difficulty": draft.difficulty})
    return draft


def render_description(spec: CodingSpec, draft: ProblemDraft, samples: list[TestCase], languages: list[str],
                       time_limit: float, memory_limit: int) -> str:
    inp, out = typemap.io_format_markdown(spec)
    parts = [draft.background.strip(), "## Task", draft.task.strip(), "## Function to implement"]
    for lang in languages:
        fence = {"python3": "python", "javascript": "javascript", "rust": "rust"}[lang]
        parts.append(f"**{LANGUAGE_NAMES[lang]}**\n\n```{fence}\n{typemap.signature(spec, lang)}\n```")
    if any(effective_helpers(spec, lang) for lang in languages):
        parts.append("Helper functions provided in the starter code (do not remove them): "
                     + "; ".join(f"{LANGUAGE_NAMES[lang]}: `" + "`, `".join(
                         HELPER_API[lang][h] for h in effective_helpers(spec, lang)) + "`"
                                 for lang in languages if effective_helpers(spec, lang)))
    parts += ["## Input", inp, "## Output", out]
    if draft.notes:
        parts += ["## Notes", "\n".join(f"- {n}" for n in draft.notes)]
    if samples:
        parts.append("## Examples")
        for i, tc in enumerate(samples, 1):
            parts.append(f"**Example {i}**\n\nInput:\n```\n{tc.input.rstrip()}\n```\nOutput:\n```\n{tc.expected_output}\n```")
    parts += ["## Constraints", f"- Time limit: {time_limit:g} s per test\n- Memory limit: {memory_limit} MB"]
    if "rust" in languages:
        parts.append("Rust submissions are compiled with plain `rustc` (standard library only). "
                     "Do not define `fn main` - the judge provides it.")
    return "\n\n".join(p for p in parts if p)


# ── reference solutions ───────────────────────────────────────────────────────


def write_reference(ctx: RunContext, spec: CodingSpec, grounding: Grounding, language: str,
                    feedback: str = "", previous: str = "") -> str:
    helpers = effective_helpers(spec, language)
    helpers_note = ("No helper functions are provided." if not helpers else
                    "These helper functions are ALREADY defined before your code - call them, do NOT redefine them:\n"
                    + "\n".join(f"- {HELPER_API[language][h]}" for h in helpers))
    previous_attempt = ""
    if feedback:
        previous_attempt = (f"\nYour previous attempt FAILED. Fix it.\nPrevious code:\n```\n{previous[:6000]}\n```\n"
                            f"Problems found:\n{feedback[:4000]}")
    prompt = prompts.render(
        "s5_reference", language_name=LANGUAGE_NAMES[language], spec=spec.model_dump_json(indent=2),
        facts=facts_text(grounding), signature=typemap.signature(spec, language), helpers_note=helpers_note,
        language_rules=LANGUAGE_RULES[language], previous_attempt=previous_attempt,
    )
    code = strip_code_fences(ctx.llm.generate("s5_reference", prompts.system_prompt(), prompt, ReferenceCode).code)
    log.info("reference written", extra={"language": language, "chars": len(code), "repair": bool(feedback)})
    return code


def static_problems(spec: CodingSpec, language: str, code: str) -> list[str]:
    problems = check_solution(language, code, typemap.function_name(spec, language))
    for helper in effective_helpers(spec, language):
        for name in HELPER_NAMES[language][helper]:
            if f"def {name}(" in code or f"function {name}(" in code or f"fn {name}(" in code:
                problems.append(f"helper `{name}` is already provided; do not redefine it")
    return problems


# ── tests ─────────────────────────────────────────────────────────────────────


def plan_tests(ctx: RunContext, spec: CodingSpec, feedback: str = "") -> TestPlan:
    params = "\n".join(f"{i}. {p.name}: {p.type} - {p.description}" + (f" ({p.constraints})" if p.constraints else "")
                       for i, p in enumerate(spec.params, 1))
    previous_attempt = f"\nYour previous test plan had problems - fix them:\n{feedback[:4000]}" if feedback else ""
    example = {p.name: {"int": "5", "u64": "253", "bool": "true", "str": "00ff"}.get(p.type, "1 2 3")
               for p in spec.params}
    prompt = prompts.render("s5_tests", spec=spec.model_dump_json(indent=2),
                            signature=typemap.signature(spec, "python3"), params=params,
                            param_names=", ".join(p.name for p in spec.params),
                            example=json.dumps(example), previous_attempt=previous_attempt)
    plan = ctx.llm.generate("s5_tests", prompts.system_prompt(), prompt, TestPlan)
    log.info("test plan written", extra={"samples": len(plan.samples), "edge_cases": len(plan.edge_cases)})
    return plan


def run_generator(ctx: RunContext, spec: CodingSpec, plan: TestPlan) -> tuple[list[str], list[str]]:
    """Execute the LLM-written generator in the sandbox; return (stdin texts, issues)."""
    n = ctx.settings.pipeline.random_tests
    if n <= 0 or "def generate" not in plan.generator_code:
        return [], ["no usable generator_code"] if n > 0 else []
    harness = (strip_code_fences(plan.generator_code) + "\n\nimport json as __json, random as __random\n"
               f"__out = [generate(__random.Random(7919 + __i)) for __i in range({n})]\n"
               "print(__json.dumps(__out))\n")
    res = ctx.get_sandbox().run(JudgeRequest(language="python3", source=harness, stdin="", time_limit=10,
                                             memory_limit=256, expected_output=""))
    if res.verdict not in ("AC", "WA"):
        return [], [f"generator failed ({res.verdict}): {res.stderr[:800]}"]
    try:
        values = json.loads(res.stdout.strip().splitlines()[-1])
    except (json.JSONDecodeError, IndexError) as exc:
        return [], [f"generator output is not JSON: {exc}"]
    inputs, issues = [], []
    for value in values:
        try:
            inputs.append(typemap.values_to_stdin(spec, value))
        except ValueError as exc:
            issues.append(f"generated input invalid: {exc}")
    return inputs, issues


def materialize_tests(ctx: RunContext, spec: CodingSpec, plan: TestPlan, grounding: Grounding,
                      py_reference: str) -> tuple[list[TestCase], list[str]]:
    """Validate inputs, execute the Python reference to get expected outputs, merge test vectors."""
    cfg = ctx.settings.pipeline
    issues: list[str] = []
    staged: list[tuple[str, str]] = []  # (origin, stdin)
    seen: set[str] = set()

    def add(origin: str, stdin: str) -> None:
        if stdin not in seen:
            seen.add(stdin)
            staged.append((origin, stdin))

    for origin, cases in (("sample", plan.samples[: cfg.max_samples]), ("edge", plan.edge_cases)):
        for args in cases:
            try:
                add(origin, typemap.normalize_args(spec, args))
            except ValueError as exc:
                issues.append(f"{origin} input {args!r} invalid: {exc}")
    with step("run_generator"):
        generated, gen_issues = run_generator(ctx, spec, plan)
    issues += gen_issues
    for stdin in generated:
        add("random", stdin)

    helpers = effective_helpers(spec, "python3")
    source = program("python3", helpers, py_reference, typemap.wrapper(spec, "python3"))
    reqs = [JudgeRequest(language="python3", source=source, stdin=stdin, time_limit=cfg.time_limit_s * 2,
                         memory_limit=cfg.memory_limit_mb) for _, stdin in staged]
    results = ctx.get_sandbox().run_many(reqs)

    tests: list[TestCase] = []
    for (origin, stdin), res in zip(staged, results):
        if res.verdict in ("CE",):
            issues.append(f"python reference does not compile: {res.stderr[:800]}")
            return [], issues
        if res.verdict not in ("AC", "WA"):  # WA only means "differs from empty expected"
            issues.append(f"python reference {res.verdict} on {origin} input {stdin!r}: {res.stderr[:300]}")
            continue
        out = typemap.canonical_output(res.stdout)
        try:
            typemap.parse_line(spec.return_type, out)
        except ValueError as exc:
            issues.append(f"python reference output {out!r} does not match return type: {exc}")
            continue
        tests.append(TestCase(input=stdin, expected_output=out, origin=origin))  # type: ignore[arg-type]

    for vec in grounding.test_vectors:
        stdin = "\n".join(vec.args) + "\n"
        if stdin in seen:
            tests = [t for t in tests if t.input != stdin]  # the spec's vector wins over the reference
        tests.append(TestCase(input=stdin, expected_output=vec.expected, origin="vector"))

    order = {"sample": 0, "edge": 1, "vector": 2, "random": 3}
    tests.sort(key=lambda t: order[t.origin])
    samples = [t for t in tests if t.origin == "sample"][: cfg.max_samples]
    if not samples and tests:
        samples = [tests[0]]
    for i, t in enumerate(tests):
        t.order_index = i
        t.points = cfg.points_per_test
        t.is_sample = t in samples
    log.info("tests materialized", extra={"tests": len(tests), "samples": len(samples),
                                         "vectors": sum(t.origin == "vector" for t in tests), "issues": len(issues)})
    for issue in issues:
        log.warning("test issue", extra={"issue": issue[:500]})
    return tests, issues
