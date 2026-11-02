"""Configuration: non-secret settings from config.yaml, secrets from .env / environment."""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

import yaml
from dotenv import load_dotenv
from pydantic import BaseModel, Field

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SUPPORTED_LANGUAGES = ("python3", "javascript", "rust")


class LLMStageConfig(BaseModel):
    provider: str = "ollama"
    model: str = "gemma3:4b"
    temperature: float = 0.2
    # gpt-oss / o-series on OpenAI-compatible APIs (groq, openai_compat): low | medium | high.
    # Reasoning tokens count against rate limits, so keep it low where the task is simple.
    # anthropic / bedrock (Claude): sent as output_config.effort: low | medium | high | xhigh | max (default medium).
    reasoning_effort: str | None = None
    # provider bedrock only: AWS region of the Bedrock endpoint. Unset -> AWS_REGION / AWS_DEFAULT_REGION -> us-east-1.
    aws_region: str | None = None
    # provider bedrock only: "mantle" (default; bedrock-mantle.<region>.api.aws, the Messages-API endpoint) or
    # "runtime" (bedrock-runtime.<region>.amazonaws.com InvokeModel), for regions without a mantle endpoint
    # such as ap-south-2. Runtime model IDs are usually inference profiles, e.g. "global.anthropic.claude-opus-5-5".
    bedrock_endpoint: str | None = None


class PathsConfig(BaseModel):
    data_dir: str = "data"
    runs_dir: str = "runs"
    out_dir: str = "out"
    logs_dir: str = "logs"
    protected_repos: list[str] = Field(default_factory=lambda: ["../transcription_engine", "../hello-bitcoin"])


class ConferenceRef(BaseModel):
    name: str
    slug: str


class PipelineConfig(BaseModel):
    max_problems_per_talk: int = 2
    min_codeability: int = 3
    chunk_tokens: int = 4000
    chunk_overlap_tokens: int = 300
    excerpt_match_threshold: int = 90
    max_repair_loops: int = 3
    min_hidden_tests: int = 10
    max_samples: int = 3
    random_tests: int = 8
    time_limit_s: float = 2.0
    memory_limit_mb: int = 256
    points_per_test: int = 1
    candidate_batch: int = 20
    fallback_conference: ConferenceRef = ConferenceRef(name="Bitcoin Talks", slug="bitcoin-talks")


class SandboxConfig(BaseModel):
    image: str = "hello-bitcoin-runner"
    # Built automatically from hello-bitcoin's own Dockerfile.runner when the image is missing.
    build_dir: str = "../hello-bitcoin/docker"
    container_memory_mb: int = 1024
    pids_limit: int = 64
    cpus: float = 0.5
    workers: int = 4
    exec_deadline_s: int = 120


class SearchConfig(BaseModel):
    max_web_results: int = 5
    max_passages: int = 6
    passage_chars: int = 800
    allow_domains: list[str] = Field(default_factory=list)


class LoggingConfig(BaseModel):
    file_level: str = "DEBUG"
    rotate_mb: int = 10
    rotate_backups: int = 5


class Secrets(BaseModel):
    te_database_url: str | None = None
    tavily_api_key: str | None = None
    ollama_host: str = "http://127.0.0.1:11434"
    google_api_key: str | None = None
    anthropic_api_key: str | None = None
    openai_api_key: str | None = None
    openai_base_url: str = "https://api.openai.com/v1"
    deepseek_api_key: str | None = None
    deepseek_base_url: str = "https://api.deepseek.com"
    groq_api_key: str | None = None
    groq_base_url: str = "https://api.groq.com/openai/v1"

    @classmethod
    def from_env(cls) -> "Secrets":
        def env(name: str) -> str | None:
            value = os.getenv(name)
            return value.strip() if value and value.strip() else None

        return cls(
            te_database_url=env("TE_DATABASE_URL"),
            tavily_api_key=env("TAVILY_API_KEY"),
            ollama_host=env("OLLAMA_HOST") or "http://127.0.0.1:11434",
            google_api_key=env("GOOGLE_API_KEY"),
            anthropic_api_key=env("ANTHROPIC_API_KEY"),
            openai_api_key=env("OPENAI_API_KEY"),
            openai_base_url=env("OPENAI_BASE_URL") or "https://api.openai.com/v1",
            deepseek_api_key=env("DEEPSEEK_API_KEY"),
            deepseek_base_url=env("DEEPSEEK_BASE_URL") or "https://api.deepseek.com",
            groq_api_key=env("GROQ_API_KEY"),
            groq_base_url=env("GROQ_BASE_URL") or "https://api.groq.com/openai/v1",
        )

    def secret_values(self) -> list[str]:
        """Values the logging filter must mask."""
        names = ("tavily_api_key", "google_api_key", "anthropic_api_key", "openai_api_key", "deepseek_api_key",
                 "groq_api_key")
        values = [getattr(self, n) for n in names]
        # Local Bedrock use may put AWS keys in .env; they are read by boto3, not by aqg, but must not be logged.
        values += [os.getenv(n) for n in ("AWS_SECRET_ACCESS_KEY", "AWS_SESSION_TOKEN", "AWS_BEARER_TOKEN_BEDROCK")]
        values.append(self.te_database_url)  # whole URL; embedded passwords are masked by pattern in logging
        return [v for v in values if v and len(v) >= 4]


class Settings(BaseModel):
    root: Path = PROJECT_ROOT
    languages: list[str] = Field(default_factory=lambda: ["python3"])
    paths: PathsConfig = PathsConfig()
    llm: dict = Field(default_factory=dict)
    pipeline: PipelineConfig = PipelineConfig()
    sandbox: SandboxConfig = SandboxConfig()
    search: SearchConfig = SearchConfig()
    logging: LoggingConfig = LoggingConfig()
    secrets: Secrets = Secrets()

    # ── derived paths ──────────────────────────────────────────────────────
    def path(self, rel: str) -> Path:
        p = Path(rel)
        return p if p.is_absolute() else (self.root / p).resolve()

    @property
    def data_dir(self) -> Path:
        return self.path(self.paths.data_dir)

    @property
    def runs_dir(self) -> Path:
        return self.path(self.paths.runs_dir)

    @property
    def out_dir(self) -> Path:
        return self.path(self.paths.out_dir)

    @property
    def logs_dir(self) -> Path:
        return self.path(self.paths.logs_dir)

    @property
    def protected_repos(self) -> list[Path]:
        return [self.path(p) for p in self.paths.protected_repos]

    def llm_stage(self, stage: str) -> LLMStageConfig:
        raw = self.llm.get(stage) or self.llm.get("default") or {}
        return LLMStageConfig(**raw)

    @property
    def llm_timeout_s(self) -> float:
        return float(self.llm.get("request_timeout_s", 600))

    @property
    def llm_max_schema_retries(self) -> int:
        return int(self.llm.get("max_schema_retries", 2))

    def normalized_languages(self, override: list[str] | None = None) -> list[str]:
        """python3 first (it computes expected outputs), then the requested others, deduplicated."""
        requested = override if override else self.languages
        unknown = [lang for lang in requested if lang not in SUPPORTED_LANGUAGES]
        if unknown:
            raise ValueError(f"Unsupported language(s) {unknown}; choose from {SUPPORTED_LANGUAGES}")
        return ["python3"] + [lang for lang in SUPPORTED_LANGUAGES[1:] if lang in requested]


def load_settings(config_path: Path | None = None) -> Settings:
    root = PROJECT_ROOT
    config_path = config_path or root / "config.yaml"
    load_dotenv(root / ".env", override=False)
    raw: dict = {}
    if config_path.exists():
        raw = yaml.safe_load(config_path.read_text()) or {}
    settings = Settings(root=config_path.parent.resolve(), **raw)
    settings.secrets = Secrets.from_env()
    return settings


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return load_settings()
