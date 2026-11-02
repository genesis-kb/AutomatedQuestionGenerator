"""Shared state for one pipeline run, plus JSON checkpoint helpers."""

from __future__ import annotations

import json
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, TypeVar

from pydantic import BaseModel

from aqg.config import Settings
from aqg.logging_setup import get_logger

log = get_logger("checkpoint")
M = TypeVar("M", bound=BaseModel)


@dataclass
class RunContext:
    settings: Settings
    run_id: str
    run_dir: Path
    ledger: Any                      # aqg.ledger.Ledger
    llm: Any                         # aqg.llm.LLMRouter
    languages: list[str]
    force: bool = False
    source: Any = None               # aqg.source.queries.TranscriptSource (DB mode)
    sandbox: Any = None              # aqg.sandbox.runner.DockerSandbox (started lazily)
    web: Any = None                  # aqg.search.tavily.TavilySearch

    def get_sandbox(self):
        if self.sandbox is None:
            from aqg.sandbox.runner import DockerSandbox

            self.sandbox = DockerSandbox(self.settings.sandbox,
                                         build_dir=self.settings.path(self.settings.sandbox.build_dir))
            self.sandbox.start()
        return self.sandbox

    def close(self) -> None:
        if self.sandbox is not None:
            self.sandbox.stop()
            self.sandbox = None


def atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            fh.write(text)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp, path)
    except BaseException:
        if os.path.exists(tmp):
            os.unlink(tmp)
        raise


def save_model(path: Path, model: BaseModel) -> None:
    atomic_write_text(path, model.model_dump_json(indent=2))


def save_json(path: Path, data: Any) -> None:
    atomic_write_text(path, json.dumps(data, indent=2, default=str, ensure_ascii=False))


def checkpoint(path: Path, model_cls: type[M], build: Callable[[], M], *, force: bool = False) -> M:
    """Load `path` if it exists (resume), otherwise build, save and return."""
    if path.exists() and not force:
        try:
            model = model_cls.model_validate_json(path.read_text(encoding="utf-8"))
            log.info("checkpoint reused", extra={"checkpoint": path.name})
            return model
        except Exception:
            log.warning("checkpoint unreadable, rebuilding", exc_info=True, extra={"checkpoint": str(path)})
    model = build()
    save_model(path, model)
    log.debug("checkpoint saved", extra={"checkpoint": path.name})
    return model
