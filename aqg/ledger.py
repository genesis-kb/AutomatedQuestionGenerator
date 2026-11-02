"""SQLite run ledger: runs, stage events, used sources (dedup) and LLM call accounting."""

from __future__ import annotations

import json
import sqlite3
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from aqg.logging_setup import get_logger
from aqg.textutil import slugify

log = get_logger("ledger")

_SCHEMA = """
CREATE TABLE IF NOT EXISTS runs (
    run_id TEXT PRIMARY KEY, started_at TEXT, finished_at TEXT, status TEXT, params TEXT, summary TEXT
);
CREATE TABLE IF NOT EXISTS stage_events (
    id INTEGER PRIMARY KEY AUTOINCREMENT, run_id TEXT, content_item_id TEXT, stage TEXT, status TEXT,
    detail TEXT, ts TEXT
);
CREATE INDEX IF NOT EXISTS idx_stage_events_run ON stage_events(run_id);
CREATE TABLE IF NOT EXISTS used_sources (
    content_item_id TEXT NOT NULL, transcript_version INTEGER, concept_hash TEXT NOT NULL DEFAULT '',
    problem_title TEXT, status TEXT, run_id TEXT, ts TEXT,
    UNIQUE (content_item_id, concept_hash)
);
CREATE TABLE IF NOT EXISTS llm_calls (
    id INTEGER PRIMARY KEY AUTOINCREMENT, run_id TEXT, stage TEXT, provider TEXT, model TEXT, prompt_hash TEXT,
    input_chars INTEGER, output_chars INTEGER, input_tokens INTEGER, output_tokens INTEGER, latency_ms INTEGER,
    ok INTEGER, error TEXT, ts TEXT
);
CREATE INDEX IF NOT EXISTS idx_llm_calls_run ON llm_calls(run_id);
"""

_LLM_COLUMNS = ("run_id", "stage", "provider", "model", "prompt_hash", "input_chars", "output_chars",
                "input_tokens", "output_tokens", "latency_ms", "ok", "error")
DEFAULT_USED_STATUSES = ("accepted", "skipped:not_codeable", "skipped:no_text")


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


class Ledger:
    def __init__(self, path: Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(str(self.path), check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        with self._lock:
            self._conn.execute("PRAGMA journal_mode=WAL")
            self._conn.executescript(_SCHEMA)
            self._conn.commit()
        log.debug("ledger opened", extra={"path": str(self.path)})

    def _write(self, sql: str, params: tuple | dict) -> None:
        with self._lock:
            self._conn.execute(sql, params)
            self._conn.commit()

    def _read(self, sql: str, params: tuple | dict = ()) -> list[sqlite3.Row]:
        with self._lock:
            return self._conn.execute(sql, params).fetchall()

    def close(self) -> None:
        with self._lock:
            self._conn.close()

    # ── runs ────────────────────────────────────────────────────────────────

    def start_run(self, run_id: str, params: dict) -> None:
        self._write("INSERT OR REPLACE INTO runs (run_id, started_at, status, params) VALUES (?, ?, 'running', ?)",
                    (run_id, _now(), json.dumps(params, default=str)))
        log.info("run started in ledger", extra={"run_id": run_id})

    def finish_run(self, run_id: str, status: str, summary: dict) -> None:
        self._write("UPDATE runs SET finished_at = ?, status = ?, summary = ? WHERE run_id = ?",
                    (_now(), status, json.dumps(summary, default=str), run_id))
        log.info("run finished in ledger", extra={"run_id": run_id, "status": status})

    def get_run(self, run_id: str) -> dict | None:
        rows = self._read("SELECT * FROM runs WHERE run_id = ?", (run_id,))
        if not rows:
            return None
        run = dict(rows[0])
        for key in ("params", "summary"):
            run[key] = json.loads(run[key]) if run[key] else None
        return run

    def record_stage(self, run_id: str, stage: str, status: str, detail: str = "",
                     content_item_id: str | None = None) -> None:
        self._write("INSERT INTO stage_events (run_id, content_item_id, stage, status, detail, ts) "
                    "VALUES (?, ?, ?, ?, ?, ?)", (run_id, content_item_id, stage, status, detail, _now()))
        log.debug("stage event recorded", extra={"stage_name": stage, "status": status})

    # ── used sources ────────────────────────────────────────────────────────

    def mark_source(self, content_item_id: str, transcript_version: int, concept_hash: str, status: str,
                    run_id: str, problem_title: str | None = None) -> None:
        self._write(
            "INSERT INTO used_sources (content_item_id, transcript_version, concept_hash, problem_title, status, "
            "run_id, ts) VALUES (?, ?, ?, ?, ?, ?, ?) "
            "ON CONFLICT (content_item_id, concept_hash) DO UPDATE SET transcript_version = excluded.transcript_version,"
            " problem_title = excluded.problem_title, status = excluded.status, run_id = excluded.run_id,"
            " ts = excluded.ts",
            (content_item_id, transcript_version, concept_hash or "", problem_title, status, run_id, _now()),
        )
        log.info("source marked", extra={"content_item_id": content_item_id, "status": status,
                                         "concept_hash": concept_hash})

    def used_content_item_ids(self, statuses: tuple[str, ...] = DEFAULT_USED_STATUSES) -> set[str]:
        if not statuses:
            return set()
        marks = ",".join("?" for _ in statuses)
        rows = self._read(f"SELECT DISTINCT content_item_id FROM used_sources WHERE status IN ({marks})",
                          tuple(statuses))
        return {r["content_item_id"] for r in rows}

    def concept_used(self, content_item_id: str, concept_hash: str) -> bool:
        rows = self._read("SELECT 1 FROM used_sources WHERE content_item_id = ? AND concept_hash = ? "
                          "AND status IN ('accepted', 'rejected') LIMIT 1", (content_item_id, concept_hash))
        return bool(rows)

    def title_used(self, title: str) -> bool:
        wanted = slugify(title)
        if not wanted:
            return False
        rows = self._read("SELECT problem_title FROM used_sources WHERE status = 'accepted' "
                          "AND problem_title IS NOT NULL")
        return any(slugify(r["problem_title"]) == wanted for r in rows)

    # ── LLM accounting ──────────────────────────────────────────────────────

    def record_llm_call(self, **fields: Any) -> None:
        values = {k: fields.get(k) for k in _LLM_COLUMNS}
        if isinstance(values["ok"], bool):
            values["ok"] = int(values["ok"])
        values["ts"] = _now()
        cols = ", ".join(values)
        marks = ", ".join(f":{k}" for k in values)
        self._write(f"INSERT INTO llm_calls ({cols}) VALUES ({marks})", values)

    def llm_totals(self, run_id: str) -> dict:
        row = self._read(
            "SELECT COUNT(*) AS calls, COALESCE(SUM(ok), 0) AS ok_calls, "
            "COALESCE(SUM(input_tokens), 0) AS input_tokens, "
            "COALESCE(SUM(output_tokens), 0) AS output_tokens, COALESCE(SUM(latency_ms), 0) AS latency_ms "
            "FROM llm_calls WHERE run_id = ?", (run_id,))[0]
        return dict(row)
