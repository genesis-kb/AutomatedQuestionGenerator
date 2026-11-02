"""Google Gemini client via google-genai (verified against google-genai 2.29)."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel

from aqg.llm.base import LLMError, LLMResponse, TransientLLMError


class GeminiClient:
    provider = "gemini"

    def __init__(self, model: str = "gemini-3-flash-preview", api_key: str | None = None, timeout: float = 600,
                 client: Any | None = None):
        self.model = model
        self.timeout = timeout
        if client is not None:  # tests inject a fake genai.Client
            self._client = client
            return
        try:
            from google import genai
            from google.genai import types
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise LLMError("google-genai is not installed (pip install 'aqg[gemini]')") from exc
        self._client = genai.Client(api_key=api_key,
                                    http_options=types.HttpOptions(timeout=int(timeout * 1000)))

    def complete_json(self, system: str, prompt: str, schema_model: type[BaseModel],
                      temperature: float) -> LLMResponse:
        from google.genai import errors, types

        config = types.GenerateContentConfig(
            system_instruction=system,
            temperature=temperature,
            response_mime_type="application/json",
            response_json_schema=schema_model.model_json_schema(),
        )
        try:
            response = self._client.models.generate_content(model=self.model, contents=prompt, config=config)
        except errors.ServerError as exc:
            raise TransientLLMError(f"gemini {exc.code}: {exc}") from exc
        except errors.ClientError as exc:
            if exc.code == 429:
                raise TransientLLMError(f"gemini rate limited: {exc}") from exc
            raise LLMError(f"gemini {exc.code}: {exc}") from exc

        text = response.text
        if not text:
            feedback = getattr(response, "prompt_feedback", None)
            reason = None
            if response.candidates:
                reason = getattr(response.candidates[0], "finish_reason", None)
            raise LLMError(f"gemini returned no text (finish_reason={reason}, prompt_feedback={feedback})")
        usage = response.usage_metadata
        return LLMResponse(
            text=text,
            input_tokens=getattr(usage, "prompt_token_count", None) if usage else None,
            output_tokens=getattr(usage, "candidates_token_count", None) if usage else None,
        )
