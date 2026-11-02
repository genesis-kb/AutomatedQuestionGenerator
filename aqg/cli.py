"""`aqg` command line: two commands.

    aqg generate   turn transcripts into validated coding problems (JSON in out/)
    aqg test       run the test suite (unit + judge integration tests)

Everything else (LLM per stage, languages, limits, paths, keys) comes from config.yaml and .env.
All logging goes to files (logs/aqg.log, logs/errors.log, runs/<run_id>/run.log).
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import Optional

import typer

from aqg.config import PROJECT_ROOT, SUPPORTED_LANGUAGES, Settings, get_settings
from aqg.logging_setup import get_logger, setup_logging

app = typer.Typer(add_completion=False, no_args_is_help=True, help="AutomatedQuestionGenerator: talks -> judge problems")
log = get_logger("cli")


def _settings() -> Settings:
    settings = get_settings()
    setup_logging(settings.logs_dir, level=settings.logging.file_level, rotate_mb=settings.logging.rotate_mb,
                  backups=settings.logging.rotate_backups, secrets=settings.secrets.secret_values())
    return settings


def _langs(value: Optional[str]) -> list[str] | None:
    if not value:
        return None
    langs = [v.strip() for v in value.split(",") if v.strip()]
    if langs == ["all"]:
        return list(SUPPORTED_LANGUAGES)
    return langs


@app.command()
def generate(
    content_item: Optional[str] = typer.Option(None, "--content-item", help="one transcription_engine content_items.id"),
    from_file: Optional[Path] = typer.Option(None, "--from-file", exists=True, help="a transcript file instead of the DB"),
    limit: int = typer.Option(1, help="how many talks to process when picking from the DB"),
    langs: Optional[str] = typer.Option(None, "--langs", help="python3,javascript,rust or 'all' (default: config.yaml)"),
    resume: Optional[str] = typer.Option(None, "--resume", metavar="RUN_ID",
                                         help="continue an earlier run: same talk, every finished step reused"),
) -> None:
    """Generate problems. With no option, picks the most codeable unused talk from the transcription DB.

    --resume RUN_ID re-runs that run in place. Each step whose JSON exists in runs/RUN_ID/ is reused, so only the
    failed or missing steps call the LLM again. Delete a step's file (e.g. c0-*/s4_grounding.json) to redo it.
    """
    settings = _settings()
    from aqg import pipeline

    run_id = None
    if resume:
        params_file = settings.runs_dir / resume / "params.json"
        if not params_file.exists():
            typer.echo(f"No run {resume} (missing {params_file})")
            raise typer.Exit(2)
        params = json.loads(params_file.read_text())
        run_id, content_item = resume, params.get("content_item_id")
        from_file = Path(params["from_file"]) if params.get("from_file") else None
        limit = params.get("limit", limit)
        langs = langs or ",".join(params.get("languages") or []) or None
        typer.echo(f"Resuming {resume} (reusing finished steps)")

    typer.echo("Running... (progress is written to the log files, not the terminal)")
    summary = pipeline.run(settings, content_item_id=content_item, from_file=from_file, limit=limit,
                           languages=_langs(langs), run_id=run_id)
    _print_summary(summary)
    raise typer.Exit(0 if summary.get("status") == "completed" else 1)


@app.command(context_settings={"allow_extra_args": True, "ignore_unknown_options": True})
def test(ctx: typer.Context) -> None:
    """Run pytest. Builds the judge image first if Docker is up, so the judge tests run instead of skipping.

    Extra arguments go straight to pytest, e.g. `aqg test -k deepseek` or `aqg test tests/unit`.
    """
    settings = _settings()
    from aqg.sandbox.runner import SandboxError, docker_available, ensure_image

    ok, detail = docker_available()
    if ok:
        try:
            ensure_image(settings.sandbox.image, settings.path(settings.sandbox.build_dir))
        except SandboxError as exc:
            typer.echo(f"warning: {exc}; judge tests will be skipped")
    else:
        typer.echo(f"warning: Docker unavailable ({detail[:80]}); judge tests will be skipped")
    args = ctx.args or ["tests"]
    raise typer.Exit(subprocess.call([sys.executable, "-m", "pytest", "-q", *args], cwd=PROJECT_ROOT))


def _print_summary(summary: dict) -> None:
    typer.echo(f"\nRun {summary['run_id']}: {summary.get('status')} in {summary.get('duration_s')}s "
               f"(languages: {', '.join(summary.get('languages', []))})")
    if summary.get("error"):
        typer.echo(f"  error: {summary['error']}")
    for talk in summary.get("talks", []):
        typer.echo(f"  talk: {talk['title']} [{talk['status']}] concepts={len(talk['concepts'])}")
    for a in summary.get("accepted", []):
        typer.echo(f"  ACCEPTED  {a['title']}  ({a['tests']} tests, {a['grounding']}, "
                   f"{a['repair_loops']} repair loop(s))\n            {a['path']}")
    for r in summary.get("rejected", []):
        typer.echo(f"  REJECTED  {r['concept']}: {r['reason'][:160]}\n            {r['path']}")
    llm = summary.get("llm") or {}
    failed = llm.get("calls", 0) - llm.get("ok_calls", 0)
    typer.echo(f"  LLM calls: {llm.get('ok_calls', 0)} ok" + (f", {failed} failed/retried" if failed else "")
               + f", tokens in/out: {llm.get('input_tokens', 0)}/{llm.get('output_tokens', 0)}")
    if summary.get("protected_repos_unchanged") is False:
        typer.echo("  !! A protected repository changed during the run - see logs/errors.log")
    typer.echo(f"  Run log:  {summary.get('run_log')}\n  Manifest: {summary.get('manifest')}")


if __name__ == "__main__":
    app()
