import json

import httpx
import pytest
from pydantic import BaseModel

from aqg.llm import LLMError, TransientLLMError, generate_structured
from aqg.llm.ollama import OllamaClient


class Sum(BaseModel):
    answer: int


def test_request_shape_and_usage():
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["url"] = str(request.url)
        seen["body"] = json.loads(request.content)
        return httpx.Response(200, json={"message": {"role": "assistant", "content": '{"answer": 5}'},
                                         "prompt_eval_count": 12, "eval_count": 4, "done": True})

    client = OllamaClient("http://ollama:11434/", "gemma3:4b", timeout=5, transport=httpx.MockTransport(handler))
    resp = client.complete_json("be terse", "2+3?", Sum, 0.1)
    assert resp.text == '{"answer": 5}' and resp.input_tokens == 12 and resp.output_tokens == 4
    body = seen["body"]
    assert seen["url"] == "http://ollama:11434/api/chat"
    assert body["stream"] is False
    assert body["model"] == "gemma3:4b"
    assert body["format"] == Sum.model_json_schema()
    assert body["messages"] == [{"role": "system", "content": "be terse"}, {"role": "user", "content": "2+3?"}]
    assert body["options"] == {"temperature": 0.1, "num_ctx": 16384}


def test_http_errors():
    def handler(request):
        return httpx.Response(503 if "busy" in request.url.host else 404, json={"error": "model not found"})

    with pytest.raises(TransientLLMError):
        OllamaClient("http://busy", "m", transport=httpx.MockTransport(handler)).complete_json("s", "p", Sum, 0)
    with pytest.raises(LLMError, match="404"):
        OllamaClient("http://x", "m", transport=httpx.MockTransport(handler)).complete_json("s", "p", Sum, 0)


def test_ping():
    def handler(request):
        assert request.url.path == "/api/tags"
        return httpx.Response(200, json={"models": [{"name": "gemma3:4b"}, {"name": "llama3:latest"}]})

    t = httpx.MockTransport(handler)
    assert OllamaClient("http://h", "gemma3:4b", transport=t).ping()[0] is True
    assert OllamaClient("http://h", "llama3", transport=t).ping()[0] is True
    ok, msg = OllamaClient("http://h", "qwen3:8b", transport=t).ping()
    assert ok is False and "not pulled" in msg

    def down(request):
        raise httpx.ConnectError("refused")

    assert OllamaClient("http://h", "gemma3:4b", transport=httpx.MockTransport(down)).ping()[0] is False


def _ollama_ready() -> bool:
    ok, _ = OllamaClient("http://127.0.0.1:11434", "gemma3:4b", timeout=3).ping()
    return ok


@pytest.mark.ollama
@pytest.mark.skipif(not _ollama_ready(), reason="local Ollama with gemma3:4b not available")
def test_live_gemma():
    client = OllamaClient("http://127.0.0.1:11434", "gemma3:4b", timeout=120)
    out = generate_structured(client, stage="live", system="You are a calculator. Reply with JSON only.",
                              prompt="What is 2+3? Put the number in field 'answer'.", schema_model=Sum,
                              temperature=0.0)
    assert out.answer == 5
