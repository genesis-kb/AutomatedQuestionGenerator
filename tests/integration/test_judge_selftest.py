"""Judge selftest: a known problem (CompactSize encoding) through aqg's wrappers, starters and the S6 gate.

No LLM. Hand-written references in all three languages run inside hello-bitcoin's runner image, so a pass
proves that the generated wrappers and starters work exactly the way the judge will run them.
"""

from __future__ import annotations

import pytest

from aqg.config import SUPPORTED_LANGUAGES, load_settings
from aqg.ledger import Ledger
from aqg.llm import FakeLLMClient, LLMRouter
from aqg.sandbox.runner import docker_available, image_exists
from aqg.selftest import run_selftest
from aqg.stages.context import RunContext

pytestmark = pytest.mark.docker


def _docker_ready() -> bool:
    ok, _ = docker_available()
    return ok and image_exists("hello-bitcoin-runner")


@pytest.mark.skipif(not _docker_ready(), reason="needs Docker and the hello-bitcoin-runner image")
def test_wrappers_and_starters_pass_in_judge(tmp_path):
    settings = load_settings()
    ledger = Ledger(tmp_path / "ledger.sqlite")
    ctx = RunContext(settings=settings, run_id="selftest", run_dir=tmp_path, ledger=ledger,
                     llm=LLMRouter(settings, ledger=ledger, overrides={"*": FakeLLMClient()}),
                     languages=list(SUPPORTED_LANGUAGES))
    try:
        result = run_selftest(ctx)
    finally:
        ctx.close()
    assert result["passed"], result["failed_checks"]
    assert set(result["per_language"]) == set(SUPPORTED_LANGUAGES)
