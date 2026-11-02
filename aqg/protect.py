"""Guards proving the protected repos (transcription_engine, hello-bitcoin) are never modified."""

from __future__ import annotations

import subprocess
from pathlib import Path

from aqg.logging_setup import get_logger

log = get_logger("protect")


def _git(path: Path, *args: str) -> str:
    result = subprocess.run(["git", "-C", str(path), *args], capture_output=True, text=True, timeout=60)
    if result.returncode != 0:
        raise RuntimeError(result.stderr.strip() or f"git {' '.join(args)} failed with code {result.returncode}")
    return result.stdout


def repo_state(path: Path) -> dict:
    """{"path", "head", "changes"} from `git status --porcelain`, or {"path", "error"}."""
    path = Path(path)
    try:
        head = _git(path, "rev-parse", "HEAD").strip()
        changes = [line for line in _git(path, "status", "--porcelain").splitlines() if line.strip()]
    except Exception as exc:  # missing repo, git absent, timeout
        log.warning("could not read repo state", extra={"path": str(path), "error": str(exc)})
        return {"path": str(path), "error": str(exc)}
    return {"path": str(path), "head": head, "changes": changes}


def snapshot(repos: list[Path]) -> dict[str, dict]:
    return {str(Path(p)): repo_state(Path(p)) for p in repos}


def compare(before: dict, after: dict) -> list[str]:
    """Human-readable differences between two snapshots."""
    diffs: list[str] = []
    for key in sorted(set(before) | set(after)):
        b, a = before.get(key), after.get(key)
        if b is None or a is None:
            diffs.append(f"{key}: present only in {'after' if b is None else 'before'} snapshot")
            continue
        if b.get("error") or a.get("error"):
            if b.get("error") != a.get("error"):
                diffs.append(f"{key}: error changed {b.get('error')!r} -> {a.get('error')!r}")
            continue
        if b.get("head") != a.get("head"):
            diffs.append(f"{key}: HEAD moved {b.get('head')} -> {a.get('head')}")
        old, new = set(b.get("changes", [])), set(a.get("changes", []))
        diffs.extend(f"{key}: new change {line!r}" for line in sorted(new - old))
        diffs.extend(f"{key}: change gone {line!r}" for line in sorted(old - new))
    return diffs


def assert_clean(repos: list[Path]) -> list[str]:
    """Problems for each protected repo that is dirty or unreadable (empty list = all clean)."""
    problems: list[str] = []
    for path in repos:
        state = repo_state(Path(path))
        if state.get("error"):
            problem = f"{path}: cannot verify ({state['error']})"
            log.critical("protected repo cannot be verified", extra={"path": str(path), "error": state["error"]})
            problems.append(problem)
        elif state["changes"]:
            log.critical("protected repo is dirty", extra={"path": str(path), "changes": state["changes"][:50]})
            problems.append(f"{path}: {len(state['changes'])} uncommitted change(s): "
                            + "; ".join(state["changes"][:10]))
        else:
            log.info("protected repo clean", extra={"path": str(path), "head": state["head"]})
    return problems
