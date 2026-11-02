"""Ollama /api/chat client with JSON-schema constrained output."""

from __future__ import annotations

import httpx
from pydantic import BaseModel

from aqg.llm.base import LLMError, LLMResponse, TransientLLMError

NUM_CTX = 16384


def _model_names_match(wanted: str, available: str) -> bool:
    def norm(name: str) -> str:
        return name[: -len(":latest")] if name.endswith(":latest") else name
    return norm(wanted) == norm(available)


class OllamaClient:
    provider = "ollama"

    def __init__(self, host: str, model: str, timeout: float = 600,
                 transport: httpx.BaseTransport | None = None):
        self.host = host.rstrip("/")
        self.model = model
        self.timeout = timeout
        self._transport = transport  # tests inject httpx.MockTransport

    def _client(self, timeout: float | None = None) -> httpx.Client:
        return httpx.Client(timeout=timeout or self.timeout, transport=self._transport)

    def complete_json(self, system: str, prompt: str, schema_model: type[BaseModel],
                      temperature: float) -> LLMResponse:
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": prompt},
            ],
            "stream": False,
            "format": schema_model.model_json_schema(),
            "options": {"temperature": temperature, "num_ctx": NUM_CTX},
        }
        with self._client() as client:
            resp = client.post(f"{self.host}/api/chat", json=payload)
        if resp.status_code == 429 or resp.status_code >= 500:
            raise TransientLLMError(f"ollama HTTP {resp.status_code}: {resp.text[:500]}")
        if resp.status_code >= 400:
            raise LLMError(f"ollama HTTP {resp.status_code}: {resp.text[:500]}")
        try:
            data = resp.json()
        except ValueError as exc:
            raise TransientLLMError(f"ollama returned non-JSON body: {resp.text[:300]}") from exc
        if data.get("error"):
            raise LLMError(f"ollama error: {data['error']}")
        text = (data.get("message") or {}).get("content") or ""
        return LLMResponse(text=text, input_tokens=data.get("prompt_eval_count"),
                           output_tokens=data.get("eval_count"))

    def ping(self) -> tuple[bool, str]:
        try:
            with self._client(timeout=10) as client:
                resp = client.get(f"{self.host}/api/tags")
            resp.raise_for_status()
            names = [m.get("name") or m.get("model") or "" for m in resp.json().get("models", [])]
        except (httpx.HTTPError, ValueError) as exc:
            return False, f"ollama unreachable at {self.host}: {exc}"
        if any(_model_names_match(self.model, n) for n in names):
            return True, f"ollama at {self.host} has {self.model}"
        return False, f"model {self.model} not pulled on {self.host} (have: {', '.join(names[:10]) or 'none'})"
