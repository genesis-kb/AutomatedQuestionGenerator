"""Provider clients against mocked transports — no network, no API keys."""

import json
from types import SimpleNamespace

import httpx
import pytest
from pydantic import BaseModel

from aqg.llm import LLMError, TransientLLMError
from aqg.llm.base import RateLimitError
from aqg.llm.openai_compat import OpenAICompatClient


class Out(BaseModel):
    answer: int


# ── openai_compat ────────────────────────────────────────────────────────

def test_openai_compat_json_schema_then_json_object_fallback():
    bodies = []

    def handler(request):
        body = json.loads(request.content)
        bodies.append((request.headers.get("authorization"), body))
        if body["response_format"]["type"] == "json_schema":
            return httpx.Response(400, json={"error": "json_schema unsupported"})
        return httpx.Response(200, json={"choices": [{"message": {"content": '{"answer": 7}'}}],
                                         "usage": {"prompt_tokens": 9, "completion_tokens": 3}})

    client = OpenAICompatClient("http://vllm:8000/v1/", "k-1", "qwen", 5, transport=httpx.MockTransport(handler))
    resp = client.complete_json("s", "p", Out, 0.2)
    assert resp.text == '{"answer": 7}' and resp.input_tokens == 9 and resp.output_tokens == 3
    first = bodies[0][1]
    assert bodies[0][0] == "Bearer k-1"
    assert first["response_format"]["json_schema"] == {"name": "Out", "schema": Out.model_json_schema(),
                                                       "strict": False}
    assert first["temperature"] == 0.2
    assert bodies[1][1]["response_format"] == {"type": "json_object"}


def test_deepseek_uses_json_object_with_schema_in_prompt():
    from aqg.config import LLMStageConfig, Secrets
    from aqg.llm import make_client

    bodies = []

    def handler(request):
        bodies.append((str(request.url), request.headers.get("authorization"), json.loads(request.content)))
        return httpx.Response(200, json={"choices": [{"message": {"content": '{"answer": 1}'}}]})

    with pytest.raises(LLMError, match="DEEPSEEK_API_KEY"):
        make_client(LLMStageConfig(provider="deepseek", model="deepseek-chat"), Secrets(), 5)
    client = make_client(LLMStageConfig(provider="deepseek", model="deepseek-chat"),
                         Secrets(deepseek_api_key="ds-key"), 5)
    assert client.provider == "deepseek" and client.base_url == "https://api.deepseek.com"
    client._transport = httpx.MockTransport(handler)
    assert client.complete_json("s", "p", Out, 0.1).text == '{"answer": 1}'
    assert len(bodies) == 1  # no json_schema attempt first
    url, auth, body = bodies[0]
    assert url == "https://api.deepseek.com/chat/completions" and auth == "Bearer ds-key"
    assert body["model"] == "deepseek-chat" and body["response_format"] == {"type": "json_object"}
    assert json.dumps(Out.model_json_schema()) in body["messages"][1]["content"]


def test_groq_client_uses_groq_endpoint_and_json_schema():
    from aqg.config import LLMStageConfig, Secrets
    from aqg.llm import make_client

    with pytest.raises(LLMError, match="GROQ_API_KEY"):
        make_client(LLMStageConfig(provider="groq", model="openai/gpt-oss-120b"), Secrets(), 5)
    bodies = []

    def handler(request):
        bodies.append((str(request.url), request.headers.get("authorization"), json.loads(request.content)))
        return httpx.Response(200, json={"choices": [{"message": {"content": '{"answer": 2}'}}]})

    client = make_client(LLMStageConfig(provider="groq", model="openai/gpt-oss-120b"),
                         Secrets(groq_api_key="gsk-key"), 5)
    client._transport = httpx.MockTransport(handler)
    assert client.provider == "groq" and client.complete_json("s", "p", Out, 0).text == '{"answer": 2}'
    url, auth, body = bodies[0]
    assert url == "https://api.groq.com/openai/v1/chat/completions" and auth == "Bearer gsk-key"
    assert body["response_format"]["type"] == "json_schema"


def test_openai_compat_errors():
    t = httpx.MockTransport(lambda r: httpx.Response(429, text="slow down"))
    with pytest.raises(TransientLLMError):
        OpenAICompatClient("http://x", None, "m", 5, transport=t).complete_json("s", "p", Out, 0)
    t = httpx.MockTransport(lambda r: httpx.Response(401, text="bad key"))
    with pytest.raises(LLMError, match="401"):
        OpenAICompatClient("http://x", None, "m", 5, transport=t).complete_json("s", "p", Out, 0)


# ── anthropic (real SDK, mocked httpx2 transport) ────────────────────────

anthropic = pytest.importorskip("anthropic")
httpx2 = pytest.importorskip("httpx2")


def _message(text, stop_reason="end_turn", stop_details=None):
    return {"id": "msg_1", "type": "message", "role": "assistant", "model": "claude-opus-5-5",
            "content": [{"type": "text", "text": text}], "stop_reason": stop_reason,
            "stop_details": stop_details, "stop_sequence": None,
            "usage": {"input_tokens": 11, "output_tokens": 5}}


def _anthropic_client(handler, model="claude-opus-5-5"):
    from aqg.llm.anthropic_client import AnthropicClient
    sdk = anthropic.Anthropic(api_key="sk-test", max_retries=0,
                              http_client=httpx2.Client(transport=httpx2.MockTransport(handler)))
    return AnthropicClient(model=model, client=sdk)


def test_anthropic_parse_request_shape():
    seen = []

    def handler(request):
        seen.append(json.loads(request.content))
        return httpx2.Response(200, json=_message('{"answer": 5}'))

    resp = _anthropic_client(handler).complete_json("sys", "2+3?", Out, 0.3)
    assert json.loads(resp.text) == {"answer": 5}
    assert (resp.input_tokens, resp.output_tokens) == (11, 5)
    body = seen[0]
    assert "temperature" not in body and "thinking" not in body
    assert body["system"] == "sys" and body["max_tokens"] == 64000
    assert body["messages"] == [{"role": "user", "content": "2+3?"}]
    assert body["output_config"]["effort"] == "medium"
    assert body["output_config"]["format"]["type"] == "json_schema"


def test_anthropic_opus_4_8_gets_adaptive_thinking_and_no_temperature():
    seen = []

    def handler(request):
        seen.append(json.loads(request.content))
        return httpx2.Response(200, json=_message('{"answer": 3}'))

    _anthropic_client(handler, model="global.anthropic.claude-opus-4-8").complete_json("sys", "1+2?", Out, 0.3)
    assert seen[0]["thinking"] == {"type": "adaptive"} and "temperature" not in seen[0]
    assert seen[0]["output_config"]["effort"] == "medium"
    seen.clear()
    _anthropic_client(handler, model="global.anthropic.claude-opus-4-6-v1").complete_json("sys", "1+2?", Out, 0.3)
    assert seen[0]["thinking"] == {"type": "adaptive"} and "temperature" not in seen[0]


def test_anthropic_older_model_sends_temperature():
    seen = []

    def handler(request):
        seen.append(json.loads(request.content))
        return httpx2.Response(200, json=_message('{"answer": 5}'))

    _anthropic_client(handler, model="claude-haiku-4-5").complete_json("s", "p", Out, 0.3)
    assert seen[0]["temperature"] == 0.3


def test_anthropic_refusal():
    def handler(request):
        return httpx2.Response(200, json=_message("", stop_reason="refusal",
                                                  stop_details={"type": "refusal", "category": "cyber"}))

    with pytest.raises(LLMError, match="cyber"):
        _anthropic_client(handler).complete_json("s", "p", Out, 0)


def test_anthropic_schema_rejected_falls_back_to_create():
    seen = []

    def handler(request):
        body = json.loads(request.content)
        seen.append(body)
        if "format" in (body.get("output_config") or {}):
            return httpx2.Response(400, json={"type": "error", "error": {
                "type": "invalid_request_error", "message": "output_config.format: schema not supported"}})
        return httpx2.Response(200, json=_message('```json\n{"answer": 5}\n```'))

    resp = _anthropic_client(handler).complete_json("sys", "p", Out, 0)
    assert "answer" in resp.text
    assert len(seen) == 2 and "JSON Schema" in seen[1]["system"]


@pytest.mark.parametrize("status, exc", [(429, TransientLLMError), (529, TransientLLMError),
                                         (500, TransientLLMError), (401, LLMError), (404, LLMError)])
def test_anthropic_error_mapping(status, exc):
    def handler(request):
        return httpx2.Response(status, json={"type": "error", "error": {"type": "x", "message": "nope"}})

    with pytest.raises(exc) as info:
        _anthropic_client(handler).complete_json("s", "p", Out, 0)
    if exc is LLMError:
        assert not isinstance(info.value, TransientLLMError)


# ── bedrock (real AnthropicBedrockMantle, fake AWS keys, mocked httpx2 transport) ──

BEDROCK_MODEL = "anthropic.claude-sonnet-5-5"


@pytest.fixture
def aws_env(monkeypatch):
    for name in ("AWS_REGION", "AWS_DEFAULT_REGION", "AWS_PROFILE", "AWS_BEARER_TOKEN_BEDROCK",
                 "ANTHROPIC_AWS_API_KEY", "ANTHROPIC_BEDROCK_MANTLE_BASE_URL"):
        monkeypatch.delenv(name, raising=False)
    return monkeypatch


def _bedrock_client(handler, model=BEDROCK_MODEL, region="us-east-1"):
    pytest.importorskip("boto3")
    from aqg.llm.anthropic_client import AnthropicClient
    sdk = anthropic.AnthropicBedrockMantle(aws_region=region, aws_access_key="AKIDTEST", aws_secret_key="secret",
                                           max_retries=0,
                                           http_client=httpx2.Client(transport=httpx2.MockTransport(handler)))
    return AnthropicClient(model=model, client=sdk, bedrock=True, aws_region=region)


@pytest.mark.parametrize("configured, env, expected", [
    ("eu-west-1", {"AWS_REGION": "us-west-2"}, "eu-west-1"),
    (None, {"AWS_REGION": "us-west-2", "AWS_DEFAULT_REGION": "eu-central-1"}, "us-west-2"),
    (None, {"AWS_DEFAULT_REGION": "eu-central-1"}, "eu-central-1"),
    (None, {}, "us-east-1"),
])
def test_bedrock_region_resolution(aws_env, configured, env, expected):
    from aqg.llm.anthropic_client import resolve_aws_region
    for k, v in env.items():
        aws_env.setenv(k, v)
    assert resolve_aws_region(configured) == expected


def test_factory_builds_bedrock_mantle_client(aws_env):
    pytest.importorskip("boto3")
    from aqg.config import LLMStageConfig, Secrets
    from aqg.llm import make_client

    aws_env.setenv("AWS_REGION", "us-west-2")
    client = make_client(LLMStageConfig(provider="bedrock", model=BEDROCK_MODEL), Secrets(), 42)
    assert client.provider == "bedrock" and client.model == BEDROCK_MODEL and client.aws_region == "us-west-2"
    assert isinstance(client._client, anthropic.AnthropicBedrockMantle)
    assert str(client._client.base_url).rstrip("/") == "https://bedrock-mantle.us-west-2.api.aws/anthropic"
    assert client._client.max_retries == 0 and client._client.timeout == 42
    pinned = make_client(LLMStageConfig(provider="bedrock", model=BEDROCK_MODEL, aws_region="eu-west-1"),
                         Secrets(), 5)
    assert pinned.aws_region == "eu-west-1" and "eu-west-1" in str(pinned._client.base_url)


def test_bedrock_stage_effort_reaches_request(aws_env):
    pytest.importorskip("boto3")
    from aqg.config import LLMStageConfig, Secrets
    from aqg.llm import make_client

    seen = []

    def handler(request):
        seen.append(json.loads(request.content))
        return httpx2.Response(200, json=_message('{"answer": 1}'))

    client = make_client(LLMStageConfig(provider="bedrock", model="anthropic.claude-fable-5-1",
                                        reasoning_effort="xhigh"), Secrets(), 5)
    assert client.effort == "xhigh"
    client._client = anthropic.AnthropicBedrockMantle(
        aws_region="us-east-1", aws_access_key="AKIDTEST", aws_secret_key="secret", max_retries=0,
        http_client=httpx2.Client(transport=httpx2.MockTransport(handler)))
    client.complete_json("sys", "1?", Out, 0.0)
    assert seen[0]["output_config"]["effort"] == "xhigh" and "temperature" not in seen[0]
    with pytest.raises(LLMError, match="reasoning_effort"):
        make_client(LLMStageConfig(provider="bedrock", model="anthropic.claude-fable-5-1",
                                   reasoning_effort="extreme"), Secrets(), 5)


def test_bedrock_runtime_endpoint_invokes_model(aws_env):
    pytest.importorskip("boto3")
    from aqg.config import LLMStageConfig, Secrets
    from aqg.llm import make_client

    seen = []

    def handler(request):
        seen.append((request, json.loads(request.content)))
        return httpx2.Response(200, json=_message('{"answer": 2}'))

    cfg = LLMStageConfig(provider="bedrock", model="global.anthropic.claude-opus-5-5", aws_region="ap-south-2",
                         bedrock_endpoint="runtime", reasoning_effort="high")
    client = make_client(cfg, Secrets(), 7)
    assert isinstance(client._client, anthropic.AnthropicBedrock)
    assert str(client._client.base_url).rstrip("/") == "https://bedrock-runtime.ap-south-2.amazonaws.com"
    client._client = anthropic.AnthropicBedrock(
        aws_region="ap-south-2", aws_access_key="AKIDTEST", aws_secret_key="secret", max_retries=0,
        http_client=httpx2.Client(transport=httpx2.MockTransport(handler)))
    assert json.loads(client.complete_json("sys", "1+1?", Out, 0.0).text) == {"answer": 2}
    request, body = seen[-1]
    assert request.url.path == "/model/global.anthropic.claude-opus-5-5/invoke"
    assert request.headers["authorization"].startswith("AWS4-HMAC-SHA256")
    assert body["output_config"]["effort"] == "high" and "temperature" not in body
    with pytest.raises(LLMError, match="bedrock_endpoint"):
        make_client(LLMStageConfig(provider="bedrock", model=BEDROCK_MODEL, bedrock_endpoint="other"), Secrets(), 7)


def test_bedrock_signed_request_and_sticky_create_fallback(aws_env):
    seen = []

    def handler(request):
        body = json.loads(request.content)
        seen.append((request.url.path, request.headers.get("authorization", ""), body))
        if "format" in (body.get("output_config") or {}):
            return httpx2.Response(400, json={"type": "error", "error": {
                "type": "invalid_request_error", "message": "Extra inputs are not permitted"}})
        return httpx2.Response(200, json=_message('{"answer": 5}'))

    client = _bedrock_client(handler)
    assert json.loads(client.complete_json("sys", "p", Out, 0.3).text) == {"answer": 5}
    path, auth, body = seen[0]
    assert path == "/anthropic/v1/messages" and auth.startswith("AWS4-HMAC-SHA256")
    assert body["model"] == BEDROCK_MODEL and "temperature" not in body
    assert body["output_config"]["effort"] == "medium"
    assert len(seen) == 2 and "JSON Schema" in seen[1][2]["system"]
    client.complete_json("sys", "p2", Out, 0.3)
    assert len(seen) == 3  # parse is not retried once Bedrock rejected it


@pytest.mark.parametrize("status, payload, headers, exc", [
    (429, {"message": "ThrottlingException: Too many tokens"}, {"retry-after": "7"}, RateLimitError),
    (429, {"message": "ThrottlingException: Too many tokens"}, {}, TransientLLMError),
    (400, {"message": "ThrottlingException: rate exceeded"}, {}, TransientLLMError),
    (503, {"message": "ServiceUnavailableException"}, {}, TransientLLMError),
    (403, {"message": "AccessDeniedException: not authorized to perform bedrock-mantle:CreateInference"}, {},
     LLMError),
    (404, {"message": "model not found"}, {}, LLMError),
])
def test_bedrock_error_mapping(aws_env, status, payload, headers, exc):
    def handler(request):
        return httpx2.Response(status, json=payload, headers=headers)

    with pytest.raises(exc) as info:
        _bedrock_client(handler).complete_json("s", "p", Out, 0)
    if exc is LLMError:
        assert not isinstance(info.value, TransientLLMError)
        assert "bedrock" in str(info.value)
    if exc is RateLimitError:
        assert info.value.retry_after == 7
    if status == 403:
        assert "bedrock-mantle:CreateInference" in str(info.value) and "us-east-1" in str(info.value)


def test_bedrock_non_claude_model_uses_signed_chat_completions(aws_env):
    pytest.importorskip("boto3")
    from aqg.config import LLMStageConfig, Secrets
    from aqg.llm import make_client

    aws_env.setenv("AWS_ACCESS_KEY_ID", "AKIDTEST")
    aws_env.setenv("AWS_SECRET_ACCESS_KEY", "secret")
    cfg = LLMStageConfig(provider="bedrock", model="global.moonshotai.kimi-k3", aws_region="ap-south-2",
                         reasoning_effort="high")
    client = make_client(cfg, Secrets(), 5)
    assert isinstance(client, OpenAICompatClient) and client.provider == "bedrock"
    assert client.base_url == "https://bedrock-runtime.ap-south-2.amazonaws.com/openai/v1"
    seen = []

    def handler(request):
        seen.append((request.url.path, request.headers["authorization"], json.loads(request.content)))
        return httpx.Response(200, json={"choices": [{"message": {"content": '{"answer": 3}'}}]})

    client._transport = httpx.MockTransport(handler)
    assert client.complete_json("s", "p", Out, 1.0).text == '{"answer": 3}'
    path, auth, body = seen[0]
    assert path == "/openai/v1/chat/completions"
    assert auth.startswith("AWS4-HMAC-SHA256") and "/ap-south-2/bedrock/aws4_request" in auth
    assert body["model"] == "global.moonshotai.kimi-k3" and body["reasoning_effort"] == "high"


def test_bedrock_missing_credentials_is_fatal_not_transient(aws_env):
    botocore_exceptions = pytest.importorskip("botocore.exceptions")
    import anthropic.lib.bedrock._mantle as mantle

    def no_creds(**_):
        raise botocore_exceptions.NoCredentialsError()

    aws_env.setattr(mantle, "get_auth_headers", no_creds)
    client = _bedrock_client(lambda r: httpx2.Response(200, json=_message('{"answer": 1}')))
    with pytest.raises(LLMError, match="no usable AWS credentials") as info:
        client.complete_json("s", "p", Out, 0)
    assert not isinstance(info.value, TransientLLMError)


# ── gemini (fake genai client) ───────────────────────────────────────────

def test_gemini_config_and_usage():
    pytest.importorskip("google.genai")
    from aqg.llm.gemini import GeminiClient
    seen = {}

    def generate_content(model, contents, config):
        seen.update(model=model, contents=contents, config=config)
        return SimpleNamespace(text='{"answer": 5}', candidates=[], prompt_feedback=None,
                               usage_metadata=SimpleNamespace(prompt_token_count=8, candidates_token_count=2))

    fake = SimpleNamespace(models=SimpleNamespace(generate_content=generate_content))
    resp = GeminiClient("gemini-3-flash-preview", client=fake).complete_json("sys", "p", Out, 0.4)
    assert (resp.text, resp.input_tokens, resp.output_tokens) == ('{"answer": 5}', 8, 2)
    cfg = seen["config"]
    assert seen["model"] == "gemini-3-flash-preview" and seen["contents"] == "p"
    assert cfg.system_instruction == "sys" and cfg.temperature == 0.4
    assert cfg.response_mime_type == "application/json"
    assert cfg.response_json_schema == Out.model_json_schema()


def test_gemini_error_mapping():
    pytest.importorskip("google.genai")
    from google.genai import errors

    from aqg.llm.gemini import GeminiClient

    def raiser(code):
        def generate_content(**_):
            cls = errors.ServerError if code >= 500 else errors.ClientError
            raise cls(code, {"error": {"code": code, "message": "x", "status": "S"}})
        return SimpleNamespace(models=SimpleNamespace(generate_content=generate_content))

    with pytest.raises(TransientLLMError):
        GeminiClient(client=raiser(429)).complete_json("s", "p", Out, 0)
    with pytest.raises(TransientLLMError):
        GeminiClient(client=raiser(503)).complete_json("s", "p", Out, 0)
    with pytest.raises(LLMError):
        GeminiClient(client=raiser(400)).complete_json("s", "p", Out, 0)


def test_parse_duration_formats():
    from aqg.llm.base import parse_duration

    assert parse_duration("3.157s") == 3.157
    assert parse_duration("1m26.4s") == 86.4
    assert parse_duration("450ms") == 0.45
    assert parse_duration("") is None and parse_duration(None) is None


def test_token_budget_paces_before_exceeding_limit(monkeypatch):
    import aqg.llm.openai_compat as oc

    slept = []
    clock = [100.0]
    monkeypatch.setattr(oc.time, "monotonic", lambda: clock[0])
    monkeypatch.setattr(oc.time, "sleep", slept.append)
    budget = oc.TokenBudget()
    budget.wait_for(5000, "m")  # nothing learned yet: never waits
    budget.update(httpx.Headers({"x-ratelimit-limit-tokens": "8000", "x-ratelimit-remaining-tokens": "3000",
                                 "x-ratelimit-reset-tokens": "12s"}))
    budget.wait_for(2000, "m")  # fits: reserved, no wait
    assert slept == [] and budget.remaining == 1000
    budget.wait_for(2000, "m")  # 1000 short: wait for the shortfall to refill (8000/min), not the full reset
    assert slept == [1000 * 60 / 8000 + 0.5]
    budget.wait_for(9000, "m")  # bigger than the shortfall allows: capped at the full reset
    assert slept[-1] == 12.5
    clock[0] += 13
    budget.wait_for(7000, "m")  # window refilled to the limit
    assert len(slept) == 2 and budget.remaining == 8000


def test_groq_clients_of_same_model_share_budget():
    from aqg.llm.openai_compat import OpenAICompatClient

    a = OpenAICompatClient("https://api.groq.com/openai/v1", "k", "openai/gpt-oss-120b", reasoning_effort="low")
    b = OpenAICompatClient("https://api.groq.com/openai/v1/", "k", "openai/gpt-oss-120b", reasoning_effort="medium")
    c = OpenAICompatClient("https://api.groq.com/openai/v1", "k", "openai/gpt-oss-20b")
    assert a.budget is b.budget and a.budget is not c.budget


def test_groq_json_validate_failed_goes_to_repair_loop_not_fatal():
    from aqg.llm import generate_structured

    bodies = []

    def handler(request):
        bodies.append(json.loads(request.content))
        if len(bodies) == 1:
            return httpx.Response(400, json={"error": {"message": "Failed to validate JSON.", "code": "json_validate_failed",
                                                       "failed_generation": '{"answer": "x"}'}})
        return httpx.Response(200, json={"choices": [{"message": {"content": '{"answer": 5}'}}]})

    client = OpenAICompatClient("https://api.groq.com/openai/v1", "k", "m", 5, provider="groq",
                                transport=httpx.MockTransport(handler))
    out = generate_structured(client, stage="s5_tests", system="s", prompt="p", schema_model=Out, temperature=0)
    assert out.answer == 5 and len(bodies) == 2
    assert client.json_mode == "json_schema"  # a bad answer must not switch the response mode
    assert bodies[1]["response_format"]["type"] == "json_schema"
    assert '{"answer": "x"}' in bodies[1]["messages"][1]["content"]  # the repair prompt shows the bad answer
