"""File-only, human-readable logging.

Nothing is written to the terminal. Sinks (plain text, one event per line):
  runs/<run_id>/run.log   INFO+ for one run: the readable story of what happened (long values shortened)
  logs/aqg.log            all levels (DEBUG+) across runs, full detail: prompts and raw LLM output are
                          printed in full, indented under their line; rotating
  logs/errors.log         ERROR and CRITICAL only, with tracebacks

Line format:  2026-10-16 21:14:41 INFO  [S3 spec] llm call ok | stage=s3_spec tokens=1834/412 latency=2.1s

Every record carries run_id / stage / step from context variables, and every
known secret value is masked before it is written.
"""

from __future__ import annotations

import contextvars
import json
import logging
import re
import time
import traceback
from contextlib import contextmanager
from datetime import datetime
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Any, Iterator

LOGGER_NAME = "aqg"

_run_id: contextvars.ContextVar[str | None] = contextvars.ContextVar("aqg_run_id", default=None)
_stage: contextvars.ContextVar[str | None] = contextvars.ContextVar("aqg_stage", default=None)
_step: contextvars.ContextVar[str | None] = contextvars.ContextVar("aqg_step", default=None)
_item: contextvars.ContextVar[str | None] = contextvars.ContextVar("aqg_item", default=None)

_STANDARD_ATTRS = set(vars(logging.makeLogRecord({}))) | {"message", "asctime"}

# Never reach the terminal: even before setup_logging() runs, 'aqg' records must not
# propagate to the root logger (whose last-resort handler prints WARNING+ to stderr).
logging.getLogger(LOGGER_NAME).addHandler(logging.NullHandler())
logging.getLogger(LOGGER_NAME).propagate = False


class ContextFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        record.run_id = getattr(record, "run_id", None) or _run_id.get()
        record.stage = getattr(record, "stage", None) or _stage.get()
        record.step = getattr(record, "step", None) or _step.get()
        record.content_item_id = getattr(record, "content_item_id", None) or _item.get()
        return True


class SecretMaskingFilter(logging.Filter):
    """Replaces every registered secret value with '***' in message and extras."""

    def __init__(self, secrets: list[str] | None = None):
        super().__init__()
        self.secrets = sorted({s for s in (secrets or []) if s}, key=len, reverse=True)

    _URL_CREDENTIALS = re.compile(r"(\w+://[^:/@\s]+:)[^@\s]+@")

    def mask(self, text: str) -> str:
        for secret in self.secrets:
            text = text.replace(secret, "***")
        return self._URL_CREDENTIALS.sub(r"\1***@", text)

    def filter(self, record: logging.LogRecord) -> bool:
        record.msg = self.mask(record.getMessage())
        record.args = ()
        for key, value in list(vars(record).items()):
            if key not in _STANDARD_ATTRS and isinstance(value, str):
                setattr(record, key, self.mask(value))
        return True


_CONTEXT_ATTRS = {"run_id", "stage", "step", "content_item_id"}


def _short(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:g}"
    if isinstance(value, (dict, list, tuple)):
        return json.dumps(value, default=str, ensure_ascii=False)
    return str(value)


class TextFormatter(logging.Formatter):
    """`<local time> <LEVEL> [<stage> / <step>] <message> | key=value ...`

    `max_value_chars` shortens long values (run.log); None keeps them whole and prints multi-line
    values (prompts, raw LLM output) indented under the line (aqg.log). `show_run` adds the run id,
    for logs shared across runs.
    """

    def __init__(self, max_value_chars: int | None = 160, show_run: bool = False):
        super().__init__()
        self.max_value_chars = max_value_chars
        self.show_run = show_run

    def format(self, record: logging.LogRecord) -> str:
        ts = datetime.fromtimestamp(record.created).strftime("%Y-%m-%d %H:%M:%S")
        where = " / ".join(x for x in (getattr(record, "stage", None), getattr(record, "step", None)) if x)
        head = f"{ts} {record.levelname:<5}"
        if self.show_run and getattr(record, "run_id", None):
            head += f" ({record.run_id})"
        line = f"{head} {f'[{where}] ' if where else ''}{record.getMessage()}"

        inline: list[str] = []
        blocks: list[str] = []
        for key, value in vars(record).items():
            if key in _STANDARD_ATTRS or key in _CONTEXT_ATTRS or value is None or value == "" or value == {}:
                continue
            if key == "duration_ms" and isinstance(value, (int, float)):
                key, text = "took", f"{value / 1000:.1f}s"
            elif key == "latency_ms" and isinstance(value, (int, float)):
                key, text = "latency", f"{value / 1000:.1f}s"
            else:
                text = _short(value)
            if self.max_value_chars is None and ("\n" in text or len(text) > 300):
                blocks.append(f"    {key}:\n" + "\n".join("      " + ln for ln in text.splitlines()))
                continue
            if self.max_value_chars is not None:
                text = text.replace("\n", " ")
                if len(text) > self.max_value_chars:
                    text = text[: self.max_value_chars] + "…"
            inline.append(f"{key}={text}")
        if inline:
            line += " | " + " ".join(inline)
        if blocks:
            line += "\n" + "\n".join(blocks)
        if record.exc_info:
            line += "\n" + "".join(traceback.format_exception(*record.exc_info)).rstrip()
        return line


_masking_filter = SecretMaskingFilter()
_configured = False


def setup_logging(logs_dir: Path, *, level: str = "DEBUG", rotate_mb: int = 10, backups: int = 5,
                  secrets: list[str] | None = None) -> logging.Logger:
    """Configure the 'aqg' logger with file sinks only. Safe to call more than once."""
    global _configured
    logger = logging.getLogger(LOGGER_NAME)
    _masking_filter.secrets = sorted({s for s in (secrets or []) if s}, key=len, reverse=True)
    if _configured:
        return logger

    logs_dir.mkdir(parents=True, exist_ok=True)
    logger.setLevel(logging.DEBUG)
    logger.propagate = False  # never bubble up to a root/console handler

    main = RotatingFileHandler(logs_dir / "aqg.log", maxBytes=rotate_mb * 1024 * 1024,
                               backupCount=backups, encoding="utf-8")
    main.setLevel(getattr(logging, level.upper(), logging.DEBUG))
    errors = RotatingFileHandler(logs_dir / "errors.log", maxBytes=rotate_mb * 1024 * 1024,
                                 backupCount=backups, encoding="utf-8")
    errors.setLevel(logging.ERROR)
    main.setFormatter(TextFormatter(max_value_chars=None, show_run=True))
    errors.setFormatter(TextFormatter(max_value_chars=2000, show_run=True))
    for handler in (main, errors):
        handler.addFilter(ContextFilter())
        handler.addFilter(_masking_filter)
        logger.addHandler(handler)

    # Third-party libraries log into our files too (WARNING+), never to the console.
    for name in ("httpx", "httpcore", "urllib3", "sqlalchemy", "trafilatura", "anthropic", "tavily", "botocore", "boto3"):
        lib = logging.getLogger(name)
        lib.setLevel(logging.WARNING)
        lib.propagate = False
        lib.handlers = [main, errors]

    _configured = True
    return logger


def get_logger(name: str | None = None) -> logging.Logger:
    return logging.getLogger(f"{LOGGER_NAME}.{name}" if name else LOGGER_NAME)


@contextmanager
def run_log(run_dir: Path, run_id: str) -> Iterator[Path]:
    """Attach a per-run log file and set the run_id context for the duration."""
    run_dir.mkdir(parents=True, exist_ok=True)
    path = run_dir / "run.log"
    handler = logging.FileHandler(path, encoding="utf-8")
    handler.setLevel(logging.INFO)  # the readable story; DEBUG detail (prompts, raw output) is in logs/aqg.log
    handler.setFormatter(TextFormatter(max_value_chars=160))
    handler.addFilter(ContextFilter())
    handler.addFilter(_masking_filter)
    # Only records belonging to this run go into run.log.
    handler.addFilter(lambda record: getattr(record, "run_id", None) == run_id)
    logger = logging.getLogger(LOGGER_NAME)
    logger.addHandler(handler)
    token = _run_id.set(run_id)
    try:
        yield path
    finally:
        _run_id.reset(token)
        logger.removeHandler(handler)
        handler.close()


@contextmanager
def content_item(content_item_id: str | None) -> Iterator[None]:
    token = _item.set(content_item_id)
    try:
        yield
    finally:
        _item.reset(token)


@contextmanager
def stage(name: str, **fields: Any) -> Iterator[None]:
    """Log stage start/end/duration; failures are logged at ERROR with traceback and re-raised."""
    log = get_logger("pipeline")
    token = _stage.set(name)
    started = time.monotonic()
    log.info("▶ started", extra=fields)
    try:
        yield
    except Exception:
        log.error("✗ stage failed", exc_info=True,
                  extra={"duration_ms": int((time.monotonic() - started) * 1000), **fields})
        raise
    else:
        log.info("✓ finished", extra={"duration_ms": int((time.monotonic() - started) * 1000), **fields})
    finally:
        _stage.reset(token)


@contextmanager
def step(name: str, **fields: Any) -> Iterator[None]:
    """Like stage() but one level finer, logged at DEBUG unless it fails."""
    log = get_logger("pipeline")
    token = _step.set(name)
    started = time.monotonic()
    log.debug("step started", extra=fields)
    try:
        yield
    except Exception:
        log.error("step failed", exc_info=True,
                  extra={"duration_ms": int((time.monotonic() - started) * 1000), **fields})
        raise
    else:
        log.debug("step finished", extra={"duration_ms": int((time.monotonic() - started) * 1000), **fields})
    finally:
        _step.reset(token)
