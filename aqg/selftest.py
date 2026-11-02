"""Self-test: push a known problem through wrapper/starter generation and the S6 gate.

No LLM involved. Uses hand-written references for Bitcoin's CompactSize encoding in all
three judge languages, so a pass proves that aqg's generated wrappers and starters work
inside hello-bitcoin's runner image exactly as the judge will run them.
"""

from __future__ import annotations

from aqg.langs import typemap
from aqg.logging_setup import get_logger
from aqg.sandbox.judge_protocol import helper_code
from aqg.schemas import CodingSpec, Grounding, ProblemDraft, TestCase
from aqg.stages.context import RunContext
from aqg.stages.s5_generate import ProblemBundle, effective_helpers, program, render_description
from aqg.stages.s6_validate import validate

log = get_logger("selftest")

SPEC = CodingSpec(
    title_hint="Encode a CompactSize integer",
    problem_kind="encode",
    function_name="compact_size_encode",
    params=[{"name": "n", "type": "u64", "description": "the integer to encode", "constraints": "0 <= n < 2^64"}],
    return_type="str",
    return_description="the CompactSize encoding as lowercase hex",
    algorithm_outline=["n < 0xfd: 1 byte", "n <= 0xffff: 0xfd + 2 bytes LE", "n <= 0xffffffff: 0xfe + 4 bytes LE",
                       "else 0xff + 8 bytes LE"],
    search_queries=["CompactSize unsigned integer encoding"],
    difficulty="easy",
)

REFERENCES = {
    "python3": (
        "def compact_size_encode(n: int) -> str:\n"
        "    if n < 0xfd:\n        return n.to_bytes(1, 'little').hex()\n"
        "    if n <= 0xffff:\n        return 'fd' + n.to_bytes(2, 'little').hex()\n"
        "    if n <= 0xffffffff:\n        return 'fe' + n.to_bytes(4, 'little').hex()\n"
        "    return 'ff' + n.to_bytes(8, 'little').hex()\n"
    ),
    "javascript": (
        "function compactSizeEncode(n) {\n"
        "    n = BigInt(n);\n"
        "    const le = (v, bytes) => { let s = ''; for (let i = 0; i < bytes; i++) { "
        "s += Number((v >> BigInt(8 * i)) & 0xffn).toString(16).padStart(2, '0'); } return s; };\n"
        "    if (n < 0xfdn) return le(n, 1);\n"
        "    if (n <= 0xffffn) return 'fd' + le(n, 2);\n"
        "    if (n <= 0xffffffffn) return 'fe' + le(n, 4);\n"
        "    return 'ff' + le(n, 8);\n"
        "}\n"
    ),
    "rust": (
        "fn compact_size_encode(n: u64) -> String {\n"
        "    let bytes = n.to_le_bytes();\n"
        "    let (prefix, len) = if n < 0xfd { (\"\", 1) } else if n <= 0xffff { (\"fd\", 2) }\n"
        "        else if n <= 0xffff_ffff { (\"fe\", 4) } else { (\"ff\", 8) };\n"
        "    format!(\"{}{}\", prefix, hex_encode(&bytes[..len]))\n"
        "}\n"
    ),
}

INPUTS = ["0", "1", "252", "253", "254", "255", "256", "65535", "65536", "4294967295", "4294967296",
          "18446744073709551615", "1000", "100000", "123456789", "515"]


def run_selftest(ctx: RunContext) -> dict:
    """Returns {'passed': bool, 'failed_checks': [...], 'per_language': {...}}."""
    langs = ctx.languages
    helpers = {lang: effective_helpers(SPEC, lang) for lang in langs}
    wrappers = {lang: typemap.wrapper(SPEC, lang) for lang in langs}
    starters = {lang: typemap.starter(SPEC, lang, helper_code(lang, helpers[lang]) if helpers[lang] else "")
                for lang in langs}

    # expected outputs from the python reference, exactly like S5
    sandbox = ctx.get_sandbox()
    from aqg.sandbox.judge_protocol import JudgeRequest

    py_source = program("python3", helpers["python3"], REFERENCES["python3"], wrappers["python3"])
    results = sandbox.run_many([JudgeRequest(language="python3", source=py_source, stdin=f"{i}\n") for i in INPUTS])
    tests = [TestCase(input=f"{i}\n", expected_output=typemap.canonical_output(r.stdout), is_sample=k < 2,
                      order_index=k, origin="sample" if k < 2 else "edge")
             for k, (i, r) in enumerate(zip(INPUTS, results))]
    draft = ProblemDraft(title="Selftest Compact Size Encoding", background="Self-test.", task="Encode n.",
                         tags=["selftest"], difficulty="easy")
    bundle = ProblemBundle(
        draft=draft, languages=langs, helpers=helpers, references={lang: REFERENCES[lang] for lang in langs},
        starters=starters, wrappers=wrappers, tests=tests,
        description=render_description(SPEC, draft, [t for t in tests if t.is_sample], langs, 2.0, 256),
    )
    report = validate(ctx, SPEC, bundle, Grounding(grounding="spec_backed"))
    failed = [f"{c.name}[{c.language}]: {c.detail}" if c.language else f"{c.name}: {c.detail}"
              for c in report.failures() if c.name != "title_valid_and_unique"]
    log.info("selftest finished", extra={"passed": not failed, "failed": failed})
    return {"passed": not failed, "failed_checks": failed, "per_language": report.per_language,
            "expected_sample": {t.input.strip(): t.expected_output for t in tests[:6]}}
