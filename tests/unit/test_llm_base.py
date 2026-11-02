import json

import httpx
import pytest
from pydantic import BaseModel

from aqg.llm import FakeLLMClient, LLMError, TransientLLMError, generate_structured, strip_json
from aqg.llm import base as llm_base


class Answer(BaseModel):
    answer: int
    note: str = ""


class FakeLedger:
    def __init__(self):
        self.rows = []

    def record_llm_call(self, **fields):
        self.rows.append(fields)


@pytest.fixture(autouse=True)
def no_backoff(monkeypatch):
    monkeypatch.setattr(llm_base, "BACKOFF_BASE_S", 0.0)


# ── strip_json ───────────────────────────────────────────────────────────

@pytest.mark.parametrize("raw, expected", [
    ('{"a": 1}', {"a": 1}),
    ('```json\n{"a": 1}\n```', {"a": 1}),
    ('```\n{"a": 1}\n```', {"a": 1}),
    ('Sure! Here is the JSON:\n{"a": {"b": [1, 2]}}\nHope that helps.', {"a": {"b": [1, 2]}}),
    ('{"s": "brace } inside \\" string {"}', {"s": 'brace } inside " string {'}),
    ('prefix {"a": 1} {"b": 2}', {"a": 1}),
    ('Text before ```json\n{"x": "y"}\n``` text after', {"x": "y"}),
    ('[1, 2, 3]', [1, 2, 3]),
])
def test_strip_json(raw, expected):
    assert json.loads(strip_json(raw)) == expected


def test_strip_json_unparseable_passthrough():
    assert strip_json("no json here") == "no json here"
    assert strip_json("") == ""


# ── generate_structured ──────────────────────────────────────────────────

def test_repair_path_records_every_attempt():
    fake = FakeLLMClient({"s3_spec": [
        "this is not json",
        {"answer": "five"},          # wrong type
        "```json\n{\"answer\": 5}\n```",
    ]})
    ledger = FakeLedger()
    result = generate_structured(fake, stage="s3_spec", system="sys", prompt="what is 2+3?",
                                 schema_model=Answer, temperature=0.0, max_retries=2,
                                 ledger=ledger, run_id="r1")
    assert result == Answer(answer=5)
    assert len(fake.calls) == 3
    assert all(c["stage"] == "s3_spec" for c in fake.calls)
    # The repair prompt carries the previous output and the validation error.
    assert "this is not json" in fake.calls[1]["prompt"]
    assert "answer" in fake.calls[2]["prompt"] and "five" in fake.calls[2]["prompt"]
    assert fake.calls[2]["prompt"].startswith("what is 2+3?")
    assert [r["ok"] for r in ledger.rows] == [False, False, True]
    row = ledger.rows[-1]
    assert row["run_id"] == "r1" and row["stage"] == "s3_spec" and row["provider"] == "fake"
    assert row["prompt_hash"] == llm_base.prompt_hash("sys", "what is 2+3?")
    assert len(row["prompt_hash"]) == 16
    assert row["error"] is None and ledger.rows[0]["error"].startswith("schema:")
    expected_keys = {"run_id", "stage", "provider", "model", "prompt_hash", "input_chars", "output_chars",
                     "input_tokens", "output_tokens", "latency_ms", "ok", "error"}
    assert all(set(r) == expected_keys for r in ledger.rows)


def test_final_failure_raises():
    fake = FakeLLMClient({"s2_triage": ["nope", "{}", '{"answer": null}']})
    ledger = FakeLedger()
    with pytest.raises(LLMError, match="Answer"):
        generate_structured(fake, stage="s2_triage", system="s", prompt="p", schema_model=Answer,
                            temperature=0.0, max_retries=2, ledger=ledger)
    assert len(fake.calls) == 3
    assert [r["ok"] for r in ledger.rows] == [False, False, False]


def test_transient_errors_are_retried():
    fake = FakeLLMClient({"*": [TransientLLMError("overloaded"), httpx.ConnectError("boom"), {"answer": 1}]})
    ledger = FakeLedger()
    out = generate_structured(fake, stage="x", system="s", prompt="p", schema_model=Answer,
                              temperature=0.0, max_retries=0, ledger=ledger)
    assert out.answer == 1
    assert [r["ok"] for r in ledger.rows] == [False, False, True]


def test_transient_errors_give_up_after_three():
    fake = FakeLLMClient({"*": [TransientLLMError("a"), TransientLLMError("b"), TransientLLMError("c"),
                                {"answer": 1}]})
    with pytest.raises(LLMError):
        generate_structured(fake, stage="x", system="s", prompt="p", schema_model=Answer,
                            temperature=0.0, max_retries=2)
    assert len(fake.calls) == 3


def test_fatal_error_not_retried():
    fake = FakeLLMClient({"*": [LLMError("bad key"), {"answer": 1}]})
    with pytest.raises(LLMError, match="bad key"):
        generate_structured(fake, stage="x", system="s", prompt="p", schema_model=Answer, temperature=0.0)
    assert len(fake.calls) == 1


def test_fake_callable_and_model_replies():
    fake = FakeLLMClient(lambda stage, system, prompt: Answer(answer=len(stage)))
    out = generate_structured(fake, stage="abc", system="s", prompt="p", schema_model=Answer, temperature=0.0)
    assert out.answer == 3


def test_fake_without_responses_raises():
    with pytest.raises(LLMError):
        generate_structured(FakeLLMClient(), stage="x", system="s", prompt="p", schema_model=Answer,
                            temperature=0.0)


def test_parse_retry_after_header_and_groq_message():
    from aqg.llm.base import parse_retry_after

    assert parse_retry_after({"retry-after": "7"}, "") == 7.0
    msg = "Rate limit reached ... Please try again in 12.7275s. Need more tokens?"
    assert parse_retry_after({}, msg) == 12.7275
    assert parse_retry_after({}, "Please try again in 1m2.5s.") == 62.5
    assert parse_retry_after({}, "Please try again in 450ms.") == 0.45
    assert parse_retry_after({}, "Please try again in 2h3m4s.") == 7384.0
    assert parse_retry_after({}, "Please try again in 5m.") == 300.0
    assert parse_retry_after({}, "quota exceeded") is None


def test_rate_limit_is_waited_out_and_not_counted(monkeypatch):
    import aqg.llm.base as base
    from aqg.llm.base import LLMResponse, RateLimitError, generate_structured

    slept = []
    monkeypatch.setattr(base.time, "sleep", slept.append)

    class Out(BaseModel):
        value: str

    class Limited:
        provider, model = "groq", "m"

        def __init__(self):
            self.n = 0

        def complete_json(self, system, prompt, schema_model, temperature):
            self.n += 1
            if self.n <= 4:  # more 429s than TRANSIENT_ATTEMPTS: must still succeed
                raise RateLimitError("groq HTTP 429", 9.0)
            return LLMResponse(text='{"value": "ok"}', input_tokens=3, output_tokens=2)

    class Ledger:
        rows = []

        def record_llm_call(self, **f):
            self.rows.append(f)

    ledger = Ledger()
    assert generate_structured(Limited(), stage="s", system="", prompt="p", schema_model=Out,
                               temperature=0, ledger=ledger).value == "ok"
    assert slept == [9.5] * 4
    assert [r["ok"] for r in ledger.rows] == [True]  # waits cost nothing and are not "calls"


def test_daily_quota_style_wait_fails_fast(monkeypatch):
    import aqg.llm.base as base
    from aqg.llm.base import LLMError, RateLimitError, generate_structured

    monkeypatch.setattr(base.time, "sleep", lambda s: (_ for _ in ()).throw(AssertionError("slept")))

    class Out(BaseModel):
        value: str

    class Quota:
        provider, model = "groq", "m"

        def complete_json(self, *a):
            raise RateLimitError("groq HTTP 429", 3600.0)

    with pytest.raises(LLMError, match="daily quota"):
        generate_structured(Quota(), stage="s", system="", prompt="p", schema_model=Out, temperature=0)
