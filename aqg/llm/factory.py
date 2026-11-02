"""Build provider clients from config and route each pipeline stage to its model."""

from __future__ import annotations

import threading
from typing import Any, TypeVar

from pydantic import BaseModel

from aqg.config import LLMStageConfig, Secrets, Settings
from aqg.llm.base import LLMClient, LLMError, generate_structured, prompt_hash

T = TypeVar("T", bound=BaseModel)

PROVIDERS = ("ollama", "anthropic", "bedrock", "gemini", "openai_compat", "deepseek", "groq")


def _openai_key_required(base_url: str) -> bool:
    return "api.openai.com" in base_url


def make_client(cfg: LLMStageConfig, secrets: Secrets, timeout: float) -> LLMClient:
    provider = cfg.provider.strip().lower()
    if provider == "ollama":
        from aqg.llm.ollama import OllamaClient
        return OllamaClient(host=secrets.ollama_host, model=cfg.model, timeout=timeout)
    if provider == "anthropic":
        from aqg.llm.anthropic_client import AnthropicClient
        # A missing key is allowed: the SDK resolves credentials itself (env / profile).
        return AnthropicClient(model=cfg.model, api_key=secrets.anthropic_api_key, timeout=timeout,
                               effort=cfg.reasoning_effort)
    if provider == "bedrock" and "claude-" not in cfg.model:
        from aqg.llm.anthropic_client import resolve_aws_region
        from aqg.llm.openai_compat import OpenAICompatClient, aws_sigv4_signer
        # Non-Claude Bedrock models (e.g. Kimi K3 "global.moonshotai.kimi-k3") speak OpenAI Chat Completions on
        # bedrock-runtime, signed with SigV4 from the default AWS credential chain.
        region = resolve_aws_region(cfg.aws_region)
        return OpenAICompatClient(base_url=f"https://bedrock-runtime.{region}.amazonaws.com/openai/v1", api_key=None,
                                  model=cfg.model, timeout=timeout, provider="bedrock",
                                  reasoning_effort=cfg.reasoning_effort, sign=aws_sigv4_signer(region))
    if provider == "bedrock":
        from aqg.llm.anthropic_client import AnthropicClient
        # Claude in Amazon Bedrock. No key in .env: the default AWS credential chain is used
        # (EC2 instance role, AWS_PROFILE, or AWS_ACCESS_KEY_ID/AWS_SECRET_ACCESS_KEY).
        return AnthropicClient(model=cfg.model, timeout=timeout, bedrock=True, aws_region=cfg.aws_region,
                               effort=cfg.reasoning_effort, bedrock_endpoint=cfg.bedrock_endpoint)
    if provider == "gemini":
        if not secrets.google_api_key:
            raise LLMError("provider 'gemini' needs GOOGLE_API_KEY in .env")
        from aqg.llm.gemini import GeminiClient
        return GeminiClient(model=cfg.model, api_key=secrets.google_api_key, timeout=timeout)
    if provider == "openai_compat":
        if not secrets.openai_api_key and _openai_key_required(secrets.openai_base_url):
            raise LLMError("provider 'openai_compat' with api.openai.com needs OPENAI_API_KEY in .env")
        from aqg.llm.openai_compat import OpenAICompatClient
        return OpenAICompatClient(base_url=secrets.openai_base_url, api_key=secrets.openai_api_key,
                                  model=cfg.model, timeout=timeout, reasoning_effort=cfg.reasoning_effort)
    if provider == "deepseek":
        if not secrets.deepseek_api_key:
            raise LLMError("provider 'deepseek' needs DEEPSEEK_API_KEY in .env")
        from aqg.llm.openai_compat import OpenAICompatClient
        # DeepSeek's API is OpenAI-compatible but supports only json_object, not json_schema.
        return OpenAICompatClient(base_url=secrets.deepseek_base_url, api_key=secrets.deepseek_api_key,
                                  model=cfg.model, timeout=timeout, provider="deepseek", json_mode="json_object")
    if provider == "groq":
        if not secrets.groq_api_key:
            raise LLMError("provider 'groq' needs GROQ_API_KEY in .env")
        from aqg.llm.openai_compat import OpenAICompatClient
        # Groq is OpenAI-compatible; models without json_schema support fall back to json_object on HTTP 400.
        return OpenAICompatClient(base_url=secrets.groq_base_url, api_key=secrets.groq_api_key,
                                  model=cfg.model, timeout=timeout, provider="groq",
                                  reasoning_effort=cfg.reasoning_effort)
    raise LLMError(f"unknown LLM provider {cfg.provider!r}; choose from {PROVIDERS}")


class LLMRouter:
    """Stage-aware entry point: `router.generate("s3_spec", system, prompt, CodingSpec)`.

    `overrides` maps a stage name (or "*" for every stage) to a ready client, e.g. a
    FakeLLMClient in tests. Real clients are cached per (provider, model).
    """

    def __init__(self, settings: Settings, ledger: Any | None = None, run_id: str | None = None,
                 overrides: dict[str, LLMClient] | None = None):
        self.settings = settings
        self.ledger = ledger
        self.run_id = run_id
        self.overrides = dict(overrides or {})
        self.models_used: dict[str, str] = {}
        self.prompt_hashes: dict[str, str] = {}
        self._clients: dict[tuple[str, str, str | None, str | None, str | None], LLMClient] = {}
        self._lock = threading.Lock()

    def client_for(self, stage: str) -> LLMClient:
        if stage in self.overrides:
            return self.overrides[stage]
        if "*" in self.overrides:
            return self.overrides["*"]
        cfg = self.settings.llm_stage(stage)
        key = (cfg.provider, cfg.model, cfg.reasoning_effort, cfg.aws_region, cfg.bedrock_endpoint)
        with self._lock:
            if key not in self._clients:
                self._clients[key] = make_client(cfg, self.settings.secrets, self.settings.llm_timeout_s)
            return self._clients[key]

    def generate(self, stage: str, system: str, prompt: str, schema_model: type[T]) -> T:
        cfg = self.settings.llm_stage(stage)
        client = self.client_for(stage)
        self.models_used[stage] = f"{client.provider}:{client.model}"
        self.prompt_hashes[stage] = prompt_hash(system, prompt)
        return generate_structured(client, stage=stage, system=system, prompt=prompt, schema_model=schema_model,
                                   temperature=cfg.temperature, max_retries=self.settings.llm_max_schema_retries,
                                   ledger=self.ledger, run_id=self.run_id)
