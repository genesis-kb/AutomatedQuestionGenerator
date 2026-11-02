"""Mirror of hello-bitcoin's judging protocol (read from its source, not imported).

hello-bitcoin/backend/services/judging.py builds:
    full_source = user_source + "\\n\\n" + separator + "\\n" + wrapper
and hello-bitcoin/backend/judge/runner.py sends this request JSON to
/judge/run.py inside the runner container.
"""

from __future__ import annotations

from importlib import resources

from pydantic import BaseModel

_SEPARATORS = {
    "python3": "# ─── JUDGE HARNESS (hidden) ───",
    "javascript": "// ─── JUDGE HARNESS (hidden) ───",
    "rust": "// ─── JUDGE HARNESS (hidden) ───",
}
_HELPER_EXT = {"python3": "py", "javascript": "js", "rust": "rs"}


def full_source(user_source: str, wrapper: str, language: str) -> str:
    if wrapper and wrapper.strip():
        return user_source + f"\n\n{_SEPARATORS[language]}\n" + wrapper
    return user_source


def helper_code(language: str, helpers: list[str]) -> str:
    parts = []
    for name in sorted(set(helpers)):
        parts.append(resources.files("aqg.sandbox.helpers").joinpath(f"{name}.{_HELPER_EXT[language]}").read_text())
    return "\n\n".join(p.rstrip() for p in parts)


class JudgeRequest(BaseModel):
    language: str
    source: str
    stdin: str
    time_limit: float = 5.0
    memory_limit: int = 256
    expected_output: str = ""
    checker_code: str = ""


class JudgeResult(BaseModel):
    verdict: str
    stdout: str = ""
    stderr: str = ""
    time_ms: int = 0
    exit_code: int = 0
    memory_kb: int = 0
    checker_msg: str = ""
