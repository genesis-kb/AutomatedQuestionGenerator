"""Deterministic in-process LLM client for tests and dry runs.

Stage awareness: `generate_structured` calls `complete_json_for_stage(stage, ...)`
when a client defines it, so FakeLLMClient always knows which stage is asking.
(`complete_json` alone uses `self.stage`, default "*".)

responses:
  * dict stage -> list of replies, popped in order; key "*" is the fallback list
    for stages without their own entry;
  * callable(stage, system, prompt) -> reply;
  * None -> every call raises LLMError.
A reply may be a str (returned verbatim), a dict/list (json.dumps), a BaseModel
(model_dump_json) or an Exception instance (raised, e.g. to simulate a transient error).
"""

from __future__ import annotations

from typing import Any, Callable

from pydantic import BaseModel

from aqg.llm.base import LLMError, LLMResponse, to_json_text

Reply = str | dict | list | BaseModel | Exception


class FakeLLMClient:
    provider = "fake"
    model = "fake"

    def __init__(self, responses: dict[str, list[Reply]] | Callable[[str, str, str], Reply] | None = None):
        if isinstance(responses, dict):
            self.responses: Any = {k: list(v) for k, v in responses.items()}
        else:
            self.responses = responses
        self.calls: list[dict[str, Any]] = []
        self.stage = "*"

    def complete_json_for_stage(self, stage: str, system: str, prompt: str, schema_model: type[BaseModel],
                                temperature: float) -> LLMResponse:
        self.calls.append({"stage": stage, "system": system, "prompt": prompt, "schema": schema_model,
                           "temperature": temperature})
        reply = self._next(stage, system, prompt)
        if isinstance(reply, BaseException):
            raise reply
        text = to_json_text(reply)
        return LLMResponse(text=text, input_tokens=len(system + prompt) // 4, output_tokens=len(text) // 4)

    def complete_json(self, system: str, prompt: str, schema_model: type[BaseModel],
                      temperature: float) -> LLMResponse:
        return self.complete_json_for_stage(self.stage, system, prompt, schema_model, temperature)

    def _next(self, stage: str, system: str, prompt: str) -> Reply:
        if self.responses is None:
            raise LLMError(f"FakeLLMClient has no responses configured (stage {stage})")
        if callable(self.responses):
            return self.responses(stage, system, prompt)
        queue = self.responses.get(stage)
        if not queue:
            queue = self.responses.get("*")
        if not queue:
            raise LLMError(f"FakeLLMClient has no response left for stage {stage!r}")
        return queue.pop(0)

    def remaining(self, stage: str) -> int:
        if isinstance(self.responses, dict):
            return len(self.responses.get(stage, []))
        return 0
