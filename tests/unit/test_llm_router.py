import pytest
from pydantic import BaseModel

from aqg.config import LLMStageConfig, Secrets, Settings
from aqg.llm import FakeLLMClient, LLMError, LLMRouter, make_client
from aqg.llm.anthropic_client import AnthropicClient
from aqg.llm.base import prompt_hash
from aqg.llm.ollama import OllamaClient
from aqg.llm.openai_compat import OpenAICompatClient


class Out(BaseModel):
    value: str


def _settings(**llm):
    base = {
        "default": {"provider": "ollama", "model": "gemma3:4b", "temperature": 0.2},
        "s2_triage": {"provider": "ollama", "model": "gemma3:4b", "temperature": 0.0},
        "s3_spec": {"provider": "anthropic", "model": "claude-opus-5-5", "temperature": 0.1},
        "max_schema_retries": 1,
        "request_timeout_s": 30,
    }
    base.update(llm)
    return Settings(llm=base, secrets=Secrets(anthropic_api_key="sk-test"))


def test_router_stage_override_and_models_used():
    triage = FakeLLMClient({"s2_triage": [{"value": "t"}]})
    rest = FakeLLMClient({"*": [{"value": "r"}]})
    router = LLMRouter(_settings(), overrides={"s2_triage": triage, "*": rest}, run_id="run1")
    assert router.generate("s2_triage", "sys", "p1", Out).value == "t"
    assert router.generate("s3_spec", "sys", "p2", Out).value == "r"
    assert len(triage.calls) == 1 and triage.calls[0]["temperature"] == 0.0
    assert rest.calls[0]["stage"] == "s3_spec" and rest.calls[0]["temperature"] == 0.1
    assert router.models_used == {"s2_triage": "fake:fake", "s3_spec": "fake:fake"}
    assert router.prompt_hashes["s3_spec"] == prompt_hash("sys", "p2")


def test_router_uses_max_schema_retries():
    fake = FakeLLMClient({"*": ["bad", "bad", {"value": "ok"}]})
    router = LLMRouter(_settings(), overrides={"*": fake})
    with pytest.raises(LLMError):
        router.generate("s2_triage", "s", "p", Out)
    assert len(fake.calls) == 2  # 1 + max_schema_retries(1)


def test_router_caches_real_clients():
    router = LLMRouter(_settings(s2_extract={"provider": "ollama", "model": "gemma3:4b", "temperature": 0.5}))
    a = router.client_for("s2_triage")
    b = router.client_for("s2_extract")
    assert a is b and isinstance(a, OllamaClient)
    assert isinstance(router.client_for("s3_spec"), AnthropicClient)
    assert router.client_for("unknown_stage") is a  # falls back to default


def test_make_client_errors_and_providers():
    secrets = Secrets()
    with pytest.raises(LLMError, match="unknown"):
        make_client(LLMStageConfig(provider="nope", model="m"), secrets, 10)
    with pytest.raises(LLMError, match="GOOGLE_API_KEY"):
        make_client(LLMStageConfig(provider="gemini", model="gemini-3-flash-preview"), secrets, 10)
    with pytest.raises(LLMError, match="OPENAI_API_KEY"):
        make_client(LLMStageConfig(provider="openai_compat", model="gpt"), secrets, 10)
    local = make_client(LLMStageConfig(provider="openai_compat", model="qwen"),
                        Secrets(openai_base_url="http://localhost:8000/v1"), 10)
    assert isinstance(local, OpenAICompatClient)
    gem = make_client(LLMStageConfig(provider="gemini", model="gemini-3-flash-preview"),
                      Secrets(google_api_key="g-test"), 10)
    assert gem.provider == "gemini" and gem.model == "gemini-3-flash-preview"
