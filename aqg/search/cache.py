"""Tiny on-disk JSON cache (one file per key, atomic writes)."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any

from aqg.logging_setup import get_logger

log = get_logger("search.cache")


class JsonCache:
    def __init__(self, directory: Path):
        self.directory = Path(directory)

    @staticmethod
    def key(*parts: str) -> str:
        return hashlib.sha256("\x1f".join(str(p) for p in parts).encode("utf-8")).hexdigest()

    def _path(self, key: str) -> Path:
        return self.directory / f"{key}.json"

    def get(self, key: str) -> Any | None:
        path = self._path(key)
        if not path.exists():
            return None
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            log.warning("cache entry unreadable; ignoring", extra={"key": key, "error": str(exc)})
            return None

    def set(self, key: str, value: Any) -> None:
        self.directory.mkdir(parents=True, exist_ok=True)
        fd, tmp = tempfile.mkstemp(dir=self.directory, prefix=f".{key}.", suffix=".tmp")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as fh:
                json.dump(value, fh, ensure_ascii=False, default=str)
            os.replace(tmp, self._path(key))
        except BaseException:
            try:
                os.unlink(tmp)
            except OSError:
                pass
            raise
        log.debug("cache set", extra={"key": key})
