"""Logging must never reach the terminal, must mask secrets, and must not crash on extras."""

import logging
import pathlib
import re

from aqg.logging_setup import SecretMaskingFilter, get_logger, run_log, setup_logging

AQG = pathlib.Path(__file__).resolve().parents[2] / "aqg"


def test_no_reserved_logrecord_keys_in_extra():
    reserved = set(vars(logging.makeLogRecord({}))) | {"message", "asctime"}
    offenders = []
    for path in AQG.rglob("*.py"):
        for no, line in enumerate(path.read_text().splitlines(), 1):
            for block in re.findall(r"extra=\{[^}]*", line):
                offenders += [f"{path.name}:{no}:{k}" for k in re.findall(r'"(\w+)":', block) if k in reserved]
    assert offenders == []


def test_masking_filter_masks_values_and_url_credentials():
    f = SecretMaskingFilter(["sk-SECRET-123"])
    assert f.mask("key sk-SECRET-123 used") == "key *** used"
    assert f.mask("postgresql+psycopg://bitcoin:hunter2@127.0.0.1/db") == "postgresql+psycopg://bitcoin:***@127.0.0.1/db"
    assert f.mask("hello-bitcoin bitcoin-talks") == "hello-bitcoin bitcoin-talks"


def test_logs_go_to_files_not_terminal(tmp_path, capsys):
    setup_logging(tmp_path / "logs", secrets=["tok-abcdef"])
    log = get_logger("test")
    with run_log(tmp_path / "run", "r1"):
        log.warning("hello tok-abcdef", extra={"detail": "x"})
        log.error("boom")
    captured = capsys.readouterr()
    assert captured.out == "" and captured.err == ""
    run_lines = (tmp_path / "run" / "run.log").read_text().splitlines()
    assert "WARNING hello *** | detail=x" in run_lines[0] and "ERROR boom" in run_lines[1]
    assert "tok-abcdef" not in (tmp_path / "logs" / "aqg.log").read_text()
