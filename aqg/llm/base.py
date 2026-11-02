"""Provider-neutral LLM plumbing: the client protocol, JSON extraction and the
schema-validated generate-with-repair loop every stage goes through.

Retry policy of `generate_structured`:
  * rate limits that say how long to wait (`RateLimitError.retry_after`, e.g. Groq's
    per-minute token limit) are waited out exactly, up to `RATE_LIMIT_WAITS` times per
    logical call. They cost no tokens, so they are not recorded as calls in the ledger;
  * other transient errors (httpx transport errors, HTTP 429/5xx without a wait hint)
    are retried with exponential backoff, up to `TRANSIENT_ATTEMPTS` attempts per call;
  * output that is not JSON or does not validate against the schema is repaired
    by re-prompting with the previous output and the exact validation errors, up
    to `max_retries` extra calls.
Every attempt (successful or not) is recorded in the ledger when one is given.

Stage awareness: a client may implement the optional method
`complete_json_for_stage(stage, system, prompt, schema_model, temperature)`;
`generate_structured` prefers it over `complete_json` (FakeLLMClient uses this to
serve per-stage canned responses).
"""

from __future__ import annotations

import hashlib
import json
import re
import time
from dataclasses import dataclass
from typing import Any, Protocol, TypeVar, runtime_checkable

import httpx
from pydantic import BaseModel, ValidationError

from aqg.logging_setup import get_logger

log = get_logger("llm")

T = TypeVar("T", bound=BaseModel)

TRANSIENT_ATTEMPTS = 3
BACKOFF_BASE_S = 2.0
RATE_LIMIT_WAITS = 8
MAX_RATE_LIMIT_WAIT_S = 90.0  # longer waits mean a daily quota: fail instead of hanging
RAW_LOG_CHARS = 4000
REPAIR_ECHO_CHARS = 6000


class LLMError(RuntimeError):
    pass


class TransientLLMError(LLMError):
    """A retryable failure (rate limit, overload, connection drop, 5xx)."""


class RateLimitError(TransientLLMError):
    """HTTP 429 with a server-provided wait (seconds) before the request can succeed."""

    def __init__(self, message: str, retry_after: float):
        super().__init__(message)
        self.retry_after = retry_after


_WAIT_RE = re.compile(r"try again in\s+(?:(\d+)h)?(?:(\d+)m(?!s))?(\d+(?:\.\d+)?)?(ms|s)?", re.IGNORECASE)


def parse_retry_after(headers: Any, body: str) -> float | None:
    """Seconds to wait from a `retry-after` header or a 'try again in 1m2.5s' message, else None."""
    value = headers.get("retry-after") if headers is not None else None
    if value:
        try:
            return float(value)
        except ValueError:
            pass
    m = _WAIT_RE.search(body or "")
    if not m or not any(m.group(i) for i in (1, 2, 3)):
        return None
    return _seconds(m)


def parse_duration(text: str | None) -> float | None:
    """'3.157s', '1m26.4s', '450ms', '2h3m4s' -> seconds (Groq's x-ratelimit-reset-* headers)."""
    m = _WAIT_RE.match(f"try again in {(text or '').strip()}")
    if not m or not any(m.group(i) for i in (1, 2, 3)):
        return None
    return _seconds(m)


def _seconds(m: re.Match) -> float:
    seconds = float(m.group(3) or 0)
    if m.group(4) == "ms":
        seconds /= 1000
    return 3600 * int(m.group(1) or 0) + 60 * int(m.group(2) or 0) + seconds


@dataclass
class LLMResponse:
    text: str
    input_tokens: int | None = None
    output_tokens: int | None = None


@runtime_checkable
class LLMClient(Protocol):
    provider: str
    model: str

    def complete_json(self, system: str, prompt: str, schema_model: type[BaseModel],
                      temperature: float) -> LLMResponse:
        """Return raw JSON text that (ideally) conforms to schema_model."""
        ...


def prompt_hash(system: str, prompt: str) -> str:
    return hashlib.sha256((system + "\n" + prompt).encode("utf-8")).hexdigest()[:16]


def _transient_types() -> tuple[type[BaseException], ...]:
    types: list[type[BaseException]] = [TransientLLMError, httpx.TransportError]
    try:  # the anthropic / google-genai SDKs are built on httpx2
        import httpx2  # type: ignore[import-not-found]
        types.append(httpx2.TransportError)
    except Exception:  # pragma: no cover - optional
        pass
    return tuple(types)


_TRANSIENT = _transient_types()


def is_transient(exc: BaseException) -> bool:
    if isinstance(exc, _TRANSIENT):
        return True
    if isinstance(exc, httpx.HTTPStatusError):
        code = exc.response.status_code
        return code == 429 or code >= 500
    return False


# ── JSON extraction ───────────────────────────────────────────────────────


def _balanced_end(text: str, start: int) -> int | None:
    """Index just past the bracket matching text[start], honouring JSON strings."""
    opener = text[start]
    closer = "}" if opener == "{" else "]"
    depth = 0
    in_string = False
    escaped = False
    for i in range(start, len(text)):
        ch = text[i]
        if in_string:
            if escaped:
                escaped = False
            elif ch == "\\":
                escaped = True
            elif ch == '"':
                in_string = False
            continue
        if ch == '"':
            in_string = True
        elif ch in "{[":
            depth += 1
        elif ch in "}]":
            depth -= 1
            if depth == 0:
                return i + 1 if ch == closer else None
    return None


def strip_json(text: str) -> str:
    """Remove ``` fences and surrounding prose; return the outermost JSON object.

    Falls back to the first '[' array if there is no object, and to the stripped
    text when nothing balanced can be found (so the validator reports the error).
    """
    if not text:
        return ""
    s = text.strip()
    if s.startswith("﻿"):
        s = s[1:]
    # Prefer the content of a fenced block when present.
    if "```" in s:
        parts = s.split("```")
        for block in parts[1::2]:
            body = block
            first_nl = body.find("\n")
            if first_nl != -1 and body[:first_nl].strip().isalpha():
                body = body[first_nl + 1:]  # drop the language tag (```json)
            elif body.lower().startswith("json"):
                body = body[4:]
            if "{" in body or "[" in body:
                s = body.strip()
                break
    for opener in "{[":
        start = s.find(opener)
        while start != -1:
            end = _balanced_end(s, start)
            if end is not None:
                return s[start:end]
            start = s.find(opener, start + 1)
    # Unbalanced: best effort first '{' .. last '}'.
    first, last = s.find("{"), s.rfind("}")
    if first != -1 and last > first:
        return s[first:last + 1]
    return s


# ── generate with repair ──────────────────────────────────────────────────


def _record(ledger: Any | None, **fields: Any) -> None:
    if ledger is None:
        return
    try:
        ledger.record_llm_call(**fields)
    except Exception:
        log.warning("ledger.record_llm_call failed", exc_info=True)


def _format_errors(exc: Exception) -> str:
    if isinstance(exc, ValidationError):
        lines = []
        for err in exc.errors(include_url=False):
            loc = ".".join(str(p) for p in err.get("loc", ())) or "<root>"
            lines.append(f"- {loc}: {err.get('msg')} ({err.get('type')})")
        return "\n".join(lines[:50])
    return f"- {type(exc).__name__}: {exc}"


def _repair_prompt(prompt: str, previous: str, errors: str) -> str:
    shown = previous if len(previous) <= REPAIR_ECHO_CHARS else previous[:REPAIR_ECHO_CHARS] + "\n...[truncated]"
    return (
        f"{prompt}\n\n"
        "----\n"
        "Your previous reply could not be accepted. Previous reply:\n"
        f"```\n{shown}\n```\n"
        "Validation errors:\n"
        f"{errors}\n\n"
        "Reply again with ONLY the corrected JSON object that conforms to the schema. "
        "No prose, no markdown fences."
    )


def _call_once(client: Any, stage: str, system: str, prompt: str, schema_model: type[BaseModel],
               temperature: float) -> LLMResponse:
    for_stage = getattr(client, "complete_json_for_stage", None)
    if callable(for_stage):
        return for_stage(stage, system, prompt, schema_model, temperature)
    return client.complete_json(system, prompt, schema_model, temperature)


def generate_structured(client: Any, *, stage: str, system: str, prompt: str, schema_model: type[T],
                        temperature: float, max_retries: int = 2, ledger: Any | None = None,
                        run_id: str | None = None) -> T:
    """Call `client` and return a validated `schema_model` instance, or raise LLMError."""
    provider = getattr(client, "provider", "?")
    model = getattr(client, "model", "?")
    phash = prompt_hash(system, prompt)
    base = dict(run_id=run_id, stage=stage, provider=provider, model=model, prompt_hash=phash)
    current_prompt = prompt
    last_error = "no attempt made"

    for schema_attempt in range(max_retries + 1):
        input_chars = len(system) + len(current_prompt)
        log.debug("llm request", extra={"llm_stage": stage, "provider": provider, "model": model,
                                        "prompt_hash": phash, "input_chars": input_chars,
                                        "schema_attempt": schema_attempt})
        response: LLMResponse | None = None
        transport_attempt = 0
        rate_limit_waits = 0
        while transport_attempt < TRANSIENT_ATTEMPTS:
            started = time.monotonic()
            try:
                response = _call_once(client, stage, system, current_prompt, schema_model, temperature)
            except RateLimitError as exc:
                if rate_limit_waits < RATE_LIMIT_WAITS and exc.retry_after <= MAX_RATE_LIMIT_WAIT_S:
                    rate_limit_waits += 1
                    delay = exc.retry_after + 0.5
                    log.warning("llm rate limited; waiting as instructed",
                                extra={"llm_stage": stage, "provider": provider, "model": model,
                                       "wait_s": round(delay, 1), "wait": rate_limit_waits})
                    time.sleep(delay)
                    continue
                error = f"{type(exc).__name__}: {exc}"[:1000]
                _record(ledger, **base, input_chars=input_chars, output_chars=0, input_tokens=None,
                        output_tokens=None, latency_ms=0, ok=False, error=error)
                log.error("llm rate limit not recoverable", extra={"llm_stage": stage, "provider": provider,
                                                                   "model": model, "error": error})
                raise LLMError(f"[{stage}] {provider}:{model} rate limited "
                               f"(asked to wait {exc.retry_after:.0f}s; daily quota?): {error}") from exc
            except Exception as exc:
                transport_attempt += 1
                latency_ms = int((time.monotonic() - started) * 1000)
                error = f"{type(exc).__name__}: {exc}"[:1000]
                _record(ledger, **base, input_chars=input_chars, output_chars=0, input_tokens=None,
                        output_tokens=None, latency_ms=latency_ms, ok=False, error=error)
                if is_transient(exc) and transport_attempt < TRANSIENT_ATTEMPTS:
                    delay = BACKOFF_BASE_S * (2 ** (transport_attempt - 1))
                    log.warning("llm transient error; retrying",
                                extra={"llm_stage": stage, "provider": provider, "model": model,
                                       "attempt": transport_attempt, "delay_s": delay, "error": error})
                    time.sleep(delay)
                    continue
                log.error("llm call failed", extra={"llm_stage": stage, "provider": provider, "model": model,
                                                    "prompt_hash": phash, "error": error})
                if isinstance(exc, LLMError) and not isinstance(exc, TransientLLMError):
                    raise
                raise LLMError(f"[{stage}] {provider}:{model} failed: {error}") from exc
            latency_ms = int((time.monotonic() - started) * 1000)
            break
        assert response is not None

        text = response.text or ""
        log.debug("llm raw output", extra={"llm_stage": stage, "prompt_hash": phash,
                                           "output_chars": len(text), "raw": text[:RAW_LOG_CHARS]})
        try:
            result = schema_model.model_validate_json(strip_json(text))
        except (ValidationError, ValueError) as exc:
            last_error = _format_errors(exc)
            _record(ledger, **base, input_chars=input_chars, output_chars=len(text),
                    input_tokens=response.input_tokens, output_tokens=response.output_tokens,
                    latency_ms=latency_ms, ok=False, error=("schema: " + last_error)[:1000])
            if schema_attempt < max_retries:
                log.warning("llm output failed schema validation; re-prompting",
                            extra={"llm_stage": stage, "provider": provider, "model": model,
                                   "schema": schema_model.__name__, "attempt": schema_attempt + 1,
                                   "errors": last_error[:2000]})
                current_prompt = _repair_prompt(prompt, text, last_error)
            continue

        _record(ledger, **base, input_chars=input_chars, output_chars=len(text),
                input_tokens=response.input_tokens, output_tokens=response.output_tokens,
                latency_ms=latency_ms, ok=True, error=None)
        log.info(f"llm {stage} ok", extra={"model": f"{provider}:{model}", "latency_ms": latency_ms,
                                           "tokens_in_out": f"{response.input_tokens}/{response.output_tokens}",
                                           **({"schema_retry": schema_attempt} if schema_attempt else {})})
        return result

    log.error("llm output never validated", extra={"llm_stage": stage, "provider": provider, "model": model,
                                                   "schema": schema_model.__name__, "prompt_hash": phash,
                                                   "errors": last_error[:2000]})
    raise LLMError(f"[{stage}] {provider}:{model} produced invalid {schema_model.__name__} after "
                   f"{max_retries + 1} attempt(s):\n{last_error}")


def to_json_text(value: Any) -> str:
    """str -> as is; BaseModel -> model_dump_json; anything else -> json.dumps."""
    if isinstance(value, str):
        return value
    if isinstance(value, BaseModel):
        return value.model_dump_json()
    return json.dumps(value)
