"""End-to-end: S1 (file) -> S7 export with a scripted LLM and the real judge sandbox."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from aqg import pipeline
from aqg.config import load_settings
from aqg.llm import FakeLLMClient
from aqg.sandbox.runner import docker_available, image_exists
from aqg.schemas import ProblemExport
from aqg.selftest import REFERENCES, SPEC

FIXTURE = Path(__file__).resolve().parents[1] / "fixtures" / "transcripts" / "forge_raw_tx_capstone.json"

pytestmark = pytest.mark.docker


def _docker_ready() -> bool:
    ok, _ = docker_available()
    return ok and image_exists("hello-bitcoin-runner")


def _script(excerpt: str):
    def reply(stage: str, system: str, prompt: str):
        if stage == "s2_extract":
            return {"concepts": [{"concept": "CompactSize integer encoding", "excerpt": excerpt,
                                  "why_codeable": "deterministic byte encoding", "codeability": 5,
                                  "requires_unsupported": False}]}
        if stage == "s3_spec":
            return SPEC.model_dump()
        if stage == "s5_statement":
            return {"title": "Encode a CompactSize Integer (fake llm test)", "background": "Varints prefix counts.",
                    "task": "Return the CompactSize encoding of `n` as lowercase hex.", "notes": [],
                    "tags": ["serialization"], "difficulty": "easy"}
        if stage == "s5_reference":
            for lang, name in (("Rust", "rust"), ("JavaScript", "javascript"), ("Python", "python3")):
                if f"in {lang}" in prompt:
                    return {"code": REFERENCES[name]}
        if stage == "s5_tests":
            values = ["0", "252", "253", "65535", "65536", "4294967295", "4294967296", "18446744073709551615",
                      "1", "300", "70000", "5000000000"]
            return {"samples": [{"n": "1"}, {"n": "253"}], "edge_cases": [{"n": v} for v in values],
                    "generator_code": "def generate(rng):\n    return [rng.randrange(0, 2**64)]\n"}
        raise AssertionError(f"unexpected stage {stage}")
    return reply


@pytest.mark.skipif(not _docker_ready(), reason="needs Docker and the hello-bitcoin-runner image")
def test_pipeline_produces_validated_problem_in_three_languages(tmp_path, monkeypatch):
    settings = load_settings()
    settings.paths.data_dir = str(tmp_path / "data")
    settings.paths.runs_dir = str(tmp_path / "runs")
    settings.paths.out_dir = str(tmp_path / "out")
    settings.paths.protected_repos = []
    monkeypatch.setattr(pipeline, "_init_search", lambda ctx, use_web: None)  # offline: transcript_only grounding

    text = json.loads(FIXTURE.read_text())["transcript"]["raw_text"]
    excerpt = text.strip()[40:160]
    fake = FakeLLMClient(_script(excerpt))
    summary = pipeline.run(settings, from_file=FIXTURE, languages=["python3", "javascript", "rust"],
                           use_web=False, llm_overrides={"*": fake}, max_problems=1)

    assert summary["status"] == "completed", summary
    assert len(summary["accepted"]) == 1, summary
    export = ProblemExport.model_validate_json(Path(summary["accepted"][0]["path"]).read_text())
    assert export.problem.expected_id == "encode-a-compactsize-integer-fake-llm-test"
    assert set(export.problem.starter_code) == {"python3", "javascript", "rust"}
    assert export.problem.is_published is False
    assert export.validation.passed
    outputs = {t.input.strip(): t.expected_output for t in export.test_cases}
    assert outputs["253"] == "fdfd00" and outputs["18446744073709551615"] == "ffffffffffffffffff"
    assert sum(t.is_sample for t in export.test_cases) == 2
    assert "## Examples" in export.problem.description
