"""OpenAI-compatible /chat/completions client (OpenAI, DeepSeek, Groq, vLLM, LM Studio, llama.cpp server...)."""

from __future__ import annotations

import json
import threading
import time
from collections.abc import Callable

import httpx
from pydantic import BaseModel

from aqg.llm.base import (LLMError, LLMResponse, RateLimitError, TransientLLMError, parse_duration,
                          parse_retry_after)
from aqg.logging_setup import get_logger

log = get_logger("llm")

CHARS_PER_TOKEN = 3.5
OUTPUT_RESERVE_TOKENS = {None: 1500, "low": 1000, "medium": 2500, "high": 5000}


class TokenBudget:
    """Per-minute token window of one (server, model), learned from x-ratelimit-* response headers.

    Before a request, `wait_for(n)` sleeps until the window resets if `n` tokens would not fit, so requests
    are paced under the limit instead of bouncing off HTTP 429. Servers without these headers are never paced.
    """

    def __init__(self):
        self.lock = threading.Lock()
        self.limit: int | None = None
        self.remaining: int | None = None
        self.reset_at = 0.0

    def update(self, headers: httpx.Headers) -> None:
        try:
            limit = headers.get("x-ratelimit-limit-tokens")
            remaining = headers.get("x-ratelimit-remaining-tokens")
            reset = parse_duration(headers.get("x-ratelimit-reset-tokens"))
        except Exception:
            return
        if remaining is None or reset is None:
            return
        with self.lock:
            self.limit = int(limit) if limit and limit.isdigit() else self.limit
            self.remaining = int(remaining) if remaining.isdigit() else self.remaining
            self.reset_at = time.monotonic() + reset

    def wait_for(self, tokens: int, model: str) -> None:
        with self.lock:
            if self.remaining is None:
                return
            now = time.monotonic()
            if now >= self.reset_at:  # window has refilled
                self.remaining = self.limit or self.remaining
                return
            if tokens <= self.remaining:
                self.remaining -= tokens  # reserve, so back-to-back calls also pace correctly
                return
            delay = self.reset_at - now
            if self.limit:  # the window refills continuously (limit per minute): wait only for the shortfall
                delay = min(delay, (tokens - self.remaining) * 60 / self.limit)
            delay += 0.5
            self.remaining = 0
        log.info("pacing to stay under the token rate limit",
                 extra={"model": model, "wait_s": round(delay, 1), "need_tokens": tokens,
                        "remaining_tokens": self.remaining, "limit_tokens": self.limit})
        time.sleep(delay)


_budgets: dict[tuple[str, str], TokenBudget] = {}
_budgets_lock = threading.Lock()


def _budget(base_url: str, model: str) -> TokenBudget:
    with _budgets_lock:
        return _budgets.setdefault((base_url, model), TokenBudget())


def aws_sigv4_signer(region: str) -> Callable[[str, bytes], dict[str, str]]:
    """SigV4 auth headers for Bedrock's OpenAI-compatible endpoint, from boto3's default credential chain
    (EC2 instance role, AWS_PROFILE or AWS_* keys). No Bedrock API key needed."""
    try:
        import boto3
        from botocore.auth import SigV4Auth
        from botocore.awsrequest import AWSRequest
    except ImportError as exc:
        raise LLMError("provider 'bedrock' needs boto3 (pip install -e '.[bedrock]')") from exc
    session = boto3.Session()

    def sign(url: str, body: bytes) -> dict[str, str]:
        creds = session.get_credentials()
        if creds is None:
            raise LLMError("no AWS credentials found (EC2 instance role, AWS_PROFILE or AWS_ACCESS_KEY_ID/...)")
        request = AWSRequest(method="POST", url=url, data=body, headers={"Content-Type": "application/json"})
        SigV4Auth(creds.get_frozen_credentials(), "bedrock", region).add_auth(request)
        return dict(request.headers.items())

    return sign


def _failed_generation(resp: httpx.Response) -> str | None:
    """The model's rejected output when the server refused it for not matching the schema, else None."""
    if resp.status_code != 400:
        return None
    try:
        err = resp.json().get("error") or {}
    except (ValueError, AttributeError):
        return None
    if not isinstance(err, dict):
        return None
    if err.get("code") == "json_validate_failed" or "Failed to generate JSON" in str(err.get("message", "")):
        return str(err.get("failed_generation") or "")
    return None


class OpenAICompatClient:
    """`json_mode` is "json_schema" (strict schema, falls back to json_object on HTTP 400) or
    "json_object" for servers that only support plain JSON mode (DeepSeek)."""

    def __init__(self, base_url: str, api_key: str | None, model: str, timeout: float = 600,
                 transport: httpx.BaseTransport | None = None, provider: str = "openai_compat",
                 json_mode: str = "json_schema", reasoning_effort: str | None = None,
                 sign: Callable[[str, bytes], dict[str, str]] | None = None):
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.model = model
        self.timeout = timeout
        self.provider = provider
        self.json_mode = json_mode
        self.reasoning_effort = reasoning_effort
        self.sign = sign  # returns auth headers for (url, body), e.g. AWS SigV4 for Bedrock
        self._transport = transport
        self.budget = _budget(self.base_url, model)  # shared by every client of this server + model

    def _post(self, payload: dict) -> httpx.Response:
        url = f"{self.base_url}/chat/completions"
        body = json.dumps(payload).encode()
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        if self.sign:
            headers.update(self.sign(url, body))  # signed per request so refreshed role credentials are used
        chars = sum(len(m["content"]) for m in payload["messages"])
        self.budget.wait_for(int(chars / CHARS_PER_TOKEN) + OUTPUT_RESERVE_TOKENS.get(self.reasoning_effort, 1500),
                             self.model)
        with httpx.Client(timeout=self.timeout, transport=self._transport) as client:
            resp = client.post(url, content=body, headers=headers)
        self.budget.update(resp.headers)
        return resp

    def complete_json(self, system: str, prompt: str, schema_model: type[BaseModel],
                      temperature: float) -> LLMResponse:
        schema = schema_model.model_json_schema()

        def payload(mode: str) -> dict:
            user = prompt
            if mode == "json_object":
                # Plain JSON mode does not carry the schema, so put it in the prompt.
                user = f"{prompt}\n\nReturn JSON matching this JSON Schema:\n{json.dumps(schema)}"
                response_format = {"type": "json_object"}
            else:
                response_format = {"type": "json_schema",
                                   "json_schema": {"name": schema_model.__name__, "schema": schema, "strict": False}}
            body = {"model": self.model, "temperature": temperature, "response_format": response_format,
                    "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}]}
            if self.reasoning_effort:
                body["reasoning_effort"] = self.reasoning_effort
            return body

        resp = self._post(payload(self.json_mode))
        failed = _failed_generation(resp)
        if failed is not None:
            # The server checked the model's JSON and it did not fit the schema (Groq: json_validate_failed).
            # That is a bad answer, not a bad request: hand it to the schema-repair loop like any invalid JSON.
            log.warning("server rejected the model's JSON output; handing it to the repair loop",
                        extra={"model": self.model, "body": resp.text[:300]})
            return LLMResponse(text=failed)
        if resp.status_code == 400 and self.json_mode == "json_schema":
            log.warning("json_schema response_format rejected; using json_object from now on",
                        extra={"model": self.model, "body": resp.text[:500]})
            self.json_mode = "json_object"  # don't pay for a rejected request on every later call
            resp = self._post(payload("json_object"))
        if resp.status_code == 429:
            wait = parse_retry_after(resp.headers, resp.text)
            if wait is not None:
                raise RateLimitError(f"{self.provider} HTTP 429: {resp.text[:500]}", wait)
        if resp.status_code == 429 or resp.status_code >= 500:
            raise TransientLLMError(f"{self.provider} HTTP {resp.status_code}: {resp.text[:500]}")
        if resp.status_code >= 400:
            raise LLMError(f"{self.provider} HTTP {resp.status_code}: {resp.text[:500]}")
        try:
            data = resp.json()
            text = data["choices"][0]["message"].get("content") or ""
        except (ValueError, KeyError, IndexError, TypeError) as exc:
            raise LLMError(f"{self.provider}: unexpected response body: {resp.text[:500]}") from exc
        usage = data.get("usage") or {}
        return LLMResponse(text=text, input_tokens=usage.get("prompt_tokens"),
                           output_tokens=usage.get("completion_tokens"))
