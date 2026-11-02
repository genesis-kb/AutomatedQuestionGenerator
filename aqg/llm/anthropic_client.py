"""Anthropic Claude client using the official SDK (verified against anthropic 1.13.0).

Structured output goes through `client.messages.parse(output_format=<pydantic model>)`.
If the API rejects the schema (or parse is unavailable), we fall back to
`client.messages.create` with the JSON schema embedded in the system prompt and let
`generate_structured` validate / repair the text.

Sampling parameters: the 1.x SDK no longer exposes `temperature` as a named
argument, and the newest models (claude-opus-5-5, claude-sonnet-5-5, claude-fable-*,
claude-opus-4-8, claude-opus-4-7) reject it with a 400. For older models temperature is passed via
`extra_body`; if the API still complains about it we drop it for the rest of the
client's life. `thinking` is sent only to Opus 4.6/4.7/4.8 and Sonnet 4.6 (`adaptive`), which otherwise skip
thinking; newer models think by default. Depth is set with
`output_config.effort`, per stage via config.yaml `reasoning_effort` (low|medium|high|xhigh|max, default medium).
MAX_TOKENS includes thinking, so it is generous; every request passes `request_timeout_s` explicitly, which keeps
the SDK from requiring streaming for that size.

The SDK's own retries are disabled (max_retries=0): `generate_structured` owns
retry/backoff so every attempt is visible in the ledger.

Amazon Bedrock (provider "bedrock") reuses this client with
`anthropic.AnthropicBedrockMantle`, the SDK's client for Claude in Amazon Bedrock
(`https://bedrock-mantle.{region}.api.aws/anthropic`, same Messages API shape; needs the
`anthropic[bedrock]` extra for boto3). Model IDs carry an `anthropic.` prefix
(`anthropic.claude-sonnet-5-5`). Credentials come from the default AWS chain (EC2 instance
role, AWS_PROFILE, AWS_ACCESS_KEY_ID/...), so no key is passed. Anthropic's Bedrock page
lists structured outputs as unsupported there, so when `messages.parse` is rejected the
client switches to `messages.create` + schema-in-prompt for the rest of its life.

Some regions (e.g. ap-south-2, Hyderabad) have no mantle endpoint yet. There, `bedrock_endpoint: runtime` uses
`anthropic.AnthropicBedrock`, which sends the same Messages body to `bedrock-runtime.<region>.amazonaws.com`
InvokeModel; model IDs there are usually inference profiles (`global.anthropic.claude-opus-5-5`).
"""

from __future__ import annotations

import json
import os
from typing import Any

from pydantic import BaseModel, ValidationError

from aqg.llm.base import LLMError, LLMResponse, RateLimitError, TransientLLMError, parse_retry_after
from aqg.logging_setup import get_logger

log = get_logger("llm")

MAX_TOKENS = 64000
EFFORT = "medium"
EFFORTS = ("low", "medium", "high", "xhigh", "max")
NO_SAMPLING_PREFIXES = ("claude-opus-5", "claude-sonnet-5", "claude-haiku-5", "claude-fable", "claude-opus-4-8",
                        "claude-opus-4-7")
# These models run without thinking unless asked (newer ones think by default). Thinking does not mix with a
# custom temperature, so they get no temperature either.
ADAPTIVE_THINKING_PREFIXES = ("claude-opus-4-8", "claude-opus-4-7", "claude-opus-4-6", "claude-sonnet-4-6")
EFFORT_PREFIXES = ("claude-opus-5", "claude-sonnet-5", "claude-haiku-5", "claude-fable", "claude-opus-4")
_SCHEMA_HINTS = ("schema", "output_format", "output_config", "json")
# Bedrock error names that mean "retry later" whatever HTTP status they arrive with.
_BEDROCK_TRANSIENT = ("throttlingexception", "serviceunavailable", "modelnotready", "too many requests")
DEFAULT_AWS_REGION = "us-east-1"
BEDROCK_ENDPOINTS = ("mantle", "runtime")


def base_model(model: str) -> str:
    """'anthropic.claude-sonnet-5-5' (Bedrock) -> 'claude-sonnet-5-5'; first-party IDs unchanged."""
    return model[model.find("claude-"):] if "claude-" in model else model


def supports_temperature(model: str) -> bool:
    return not base_model(model).startswith(NO_SAMPLING_PREFIXES + ADAPTIVE_THINKING_PREFIXES)


def supports_effort(model: str) -> bool:
    return base_model(model).startswith(EFFORT_PREFIXES)


def resolve_aws_region(configured: str | None = None) -> str:
    """config.yaml `aws_region`, else AWS_REGION / AWS_DEFAULT_REGION, else us-east-1."""
    for value in (configured, os.getenv("AWS_REGION"), os.getenv("AWS_DEFAULT_REGION")):
        if value and value.strip():
            return value.strip()
    return DEFAULT_AWS_REGION


def _aws_credential_errors() -> tuple[type[BaseException], ...]:
    try:
        from botocore import exceptions as be
    except ImportError:  # pragma: no cover - bedrock extra not installed
        return ()
    return (be.NoCredentialsError, be.PartialCredentialsError, be.CredentialRetrievalError, be.NoRegionError,
            be.BotoCoreError)


class AnthropicClient:
    provider = "anthropic"

    def __init__(self, model: str = "claude-opus-5-5", api_key: str | None = None, timeout: float = 600,
                 client: Any | None = None, *, bedrock: bool = False, aws_region: str | None = None,
                 effort: str | None = None, bedrock_endpoint: str | None = None):
        try:
            import anthropic
        except ImportError as exc:  # pragma: no cover - optional dependency
            extra = "bedrock" if bedrock else "anthropic"
            raise LLMError(f"anthropic SDK is not installed (pip install -e '.[{extra}]')") from exc
        self._sdk = anthropic
        self.model = model
        self.timeout = timeout
        self.effort = (effort or EFFORT).strip().lower()
        if self.effort not in EFFORTS:
            raise LLMError(f"reasoning_effort {effort!r} is not one of {', '.join(EFFORTS)}")
        self._send_temperature = supports_temperature(model)
        self._use_parse = True  # False once the API rejects structured output
        self.aws_region: str | None = None
        self.bedrock_endpoint: str | None = None
        self._errors: tuple[type[BaseException], ...] = (anthropic.APIError,)
        if bedrock:
            self.provider = "bedrock"
            self.aws_region = resolve_aws_region(aws_region)
            self.bedrock_endpoint = (bedrock_endpoint or "mantle").strip().lower()
            if self.bedrock_endpoint not in BEDROCK_ENDPOINTS:
                raise LLMError(f"bedrock_endpoint {bedrock_endpoint!r} is not one of {', '.join(BEDROCK_ENDPOINTS)}")
            # Missing AWS credentials surface at signing time as botocore errors or a RuntimeError.
            self._errors = (anthropic.APIError, RuntimeError, *_aws_credential_errors())
        if client is not None:  # tests inject a fake
            self._client = client
        elif bedrock:
            try:
                import boto3  # noqa: F401  (the SDK imports it lazily when signing; fail early instead)
            except ImportError as exc:
                raise LLMError("provider 'bedrock' needs boto3 (pip install -e '.[bedrock]')") from exc
            sdk_class = anthropic.AnthropicBedrock if self.bedrock_endpoint == "runtime" else anthropic.AnthropicBedrockMantle
            self._client = sdk_class(aws_region=self.aws_region, timeout=timeout, max_retries=0)
        else:
            kwargs: dict[str, Any] = {"timeout": timeout, "max_retries": 0}
            if api_key:
                kwargs["api_key"] = api_key  # otherwise the SDK resolves credentials itself
            self._client = anthropic.Anthropic(**kwargs)

    # ── helpers ──────────────────────────────────────────────────────────
    def _common(self, temperature: float) -> dict[str, Any]:
        # An explicit per-request timeout also stops the SDK demanding streaming for a large max_tokens.
        kwargs: dict[str, Any] = {"model": self.model, "max_tokens": MAX_TOKENS, "timeout": self.timeout}
        if supports_effort(self.model):
            kwargs["output_config"] = {"effort": self.effort}
        if base_model(self.model).startswith(ADAPTIVE_THINKING_PREFIXES):
            kwargs["thinking"] = {"type": "adaptive"}
        if self._send_temperature:
            kwargs["extra_body"] = {"temperature": temperature}
        return kwargs

    def _translate(self, exc: BaseException) -> BaseException:
        sdk, p = self._sdk, self.provider
        if isinstance(exc, LLMError):
            return exc
        if not isinstance(exc, sdk.APIError):  # bedrock: AWS credential / signing failures
            text = f"{type(exc).__name__}: {exc}"
            if isinstance(exc, RuntimeError) and "credential" not in str(exc).lower():
                return exc
            return LLMError(f"bedrock: no usable AWS credentials ({text}). On EC2 attach an IAM role to the "
                            "instance; locally set AWS_PROFILE or run `aws configure`.")
        if isinstance(exc, sdk.RateLimitError) or (
                isinstance(exc, sdk.APIStatusError) and any(h in str(exc).lower() for h in _BEDROCK_TRANSIENT)):
            response = getattr(exc, "response", None)
            wait = parse_retry_after(getattr(response, "headers", None), str(exc))
            if wait is not None:
                return RateLimitError(f"{p} rate limited: {exc}", retry_after=wait)
            return TransientLLMError(f"{p} transient: {type(exc).__name__}: {exc}")
        if isinstance(exc, sdk.APIConnectionError):  # includes APITimeoutError
            return TransientLLMError(f"{p} transient: {type(exc).__name__}: {exc}")
        if isinstance(exc, sdk.PermissionDeniedError) and p == "bedrock":
            return LLMError(f"bedrock: access denied for {self.model} in {self.aws_region}. Check that the IAM "
                            "policy allows bedrock-mantle:CreateInference / bedrock:InvokeModel and that model access is enabled in the "
                            f"Bedrock console: {exc}")
        if isinstance(exc, (sdk.BadRequestError, sdk.AuthenticationError, sdk.PermissionDeniedError,
                            sdk.NotFoundError)):
            return LLMError(f"{p}: {type(exc).__name__}: {exc}")
        if isinstance(exc, sdk.APIStatusError):
            if exc.status_code == 429 or exc.status_code >= 500:  # overloaded (529), 5xx
                return TransientLLMError(f"{p} HTTP {exc.status_code}: {exc}")
            return LLMError(f"{p} HTTP {exc.status_code}: {exc}")
        return exc

    def _raise_translated(self, exc: BaseException) -> None:
        translated = self._translate(exc)
        if translated is exc:
            raise exc
        raise translated from exc

    @staticmethod
    def _check_refusal(response: Any) -> None:
        if getattr(response, "stop_reason", None) == "refusal":
            details = getattr(response, "stop_details", None)
            category = getattr(details, "category", None) if details is not None else None
            raise LLMError(f"anthropic refused the request (category={category})")

    @staticmethod
    def _usage(response: Any) -> tuple[int | None, int | None]:
        usage = getattr(response, "usage", None)
        if usage is None:
            return None, None
        return getattr(usage, "input_tokens", None), getattr(usage, "output_tokens", None)

    def _call(self, fn: Any, temperature: float, **kwargs: Any) -> Any:
        """Call fn(**common, **kwargs); drop temperature once if the API rejects it."""
        try:
            return fn(**self._common(temperature), **kwargs)
        except self._sdk.BadRequestError as exc:
            if self._send_temperature and "temperature" in str(exc).lower():
                log.warning("anthropic rejected temperature; retrying without it", extra={"model": self.model})
                self._send_temperature = False
                return fn(**self._common(temperature), **kwargs)
            raise

    # ── API ──────────────────────────────────────────────────────────────
    def complete_json(self, system: str, prompt: str, schema_model: type[BaseModel],
                      temperature: float) -> LLMResponse:
        messages = [{"role": "user", "content": prompt}]
        if not self._use_parse:
            return self._create_fallback(system, messages, schema_model, temperature)
        try:
            try:
                response = self._call(self._client.messages.parse, temperature, system=system,
                                      messages=messages, output_format=schema_model)
            except (AttributeError, TypeError, ValidationError) as exc:
                log.warning(f"{self.provider} messages.parse unusable; falling back to create",
                            extra={"model": self.model, "error": f"{type(exc).__name__}: {exc}"[:500]})
                return self._create_fallback(system, messages, schema_model, temperature)
            except self._sdk.BadRequestError as exc:
                # Bedrock does not document structured outputs, so any 400 on parse gets one create attempt.
                if self.provider == "bedrock" or any(h in str(exc).lower() for h in _SCHEMA_HINTS):
                    log.warning(f"{self.provider} rejected structured output; using create from now on",
                                extra={"model": self.model, "error": str(exc)[:500]})
                    self._use_parse = False
                    return self._create_fallback(system, messages, schema_model, temperature)
                raise
        except self._errors as exc:
            self._raise_translated(exc)

        self._check_refusal(response)
        input_tokens, output_tokens = self._usage(response)
        parsed = getattr(response, "parsed_output", None)
        if parsed is not None:
            return LLMResponse(text=parsed.model_dump_json(), input_tokens=input_tokens,
                               output_tokens=output_tokens)
        # Not parsed (e.g. truncated at max_tokens): hand the raw text to the repair loop.
        return LLMResponse(text=self._text_of(response), input_tokens=input_tokens, output_tokens=output_tokens)

    def _create_fallback(self, system: str, messages: list[dict], schema_model: type[BaseModel],
                         temperature: float) -> LLMResponse:
        schema = json.dumps(schema_model.model_json_schema(), indent=1)
        fallback_system = (
            f"{system}\n\nRespond with a single JSON object only (no prose, no markdown fences) that conforms "
            f"to this JSON Schema:\n{schema}"
        )
        try:
            response = self._call(self._client.messages.create, temperature, system=fallback_system,
                                  messages=messages)
        except self._errors as exc:
            self._raise_translated(exc)
        self._check_refusal(response)
        input_tokens, output_tokens = self._usage(response)
        return LLMResponse(text=self._text_of(response), input_tokens=input_tokens, output_tokens=output_tokens)

    @staticmethod
    def _text_of(response: Any) -> str:
        return "".join(getattr(block, "text", "") or "" for block in (getattr(response, "content", None) or [])
                       if getattr(block, "type", None) == "text")
