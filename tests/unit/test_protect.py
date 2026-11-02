import shutil
import subprocess

import pytest

from aqg.protect import assert_clean, compare, repo_state, snapshot

pytestmark = pytest.mark.skipif(shutil.which("git") is None, reason="git not installed")


def _git(repo, *args):
    subprocess.run(["git", "-C", str(repo), *args], check=True, capture_output=True)


@pytest.fixture
def repo(tmp_path):
    path = tmp_path / "repo"
    path.mkdir()
    _git(path, "init", "-q")
    _git(path, "config", "user.email", "t@example.com")
    _git(path, "config", "user.name", "t")
    (path / ".gitignore").write_text("ignored/\n")
    (path / "a.txt").write_text("a\n")
    _git(path, "add", ".")
    _git(path, "commit", "-q", "-m", "init")
    return path


def test_clean_then_dirty(repo):
    state = repo_state(repo)
    assert state["changes"] == [] and len(state["head"]) == 40
    assert assert_clean([repo]) == []
    before = snapshot([repo])

    (repo / "ignored").mkdir()
    (repo / "ignored" / "x.log").write_text("fine")
    assert assert_clean([repo]) == []

    (repo / "a.txt").write_text("changed\n")
    (repo / "new.txt").write_text("new\n")
    problems = assert_clean([repo])
    assert len(problems) == 1 and "2 uncommitted" in problems[0]
    diffs = compare(before, snapshot([repo]))
    assert any("a.txt" in d for d in diffs) and any("?? new.txt" in d for d in diffs)

    _git(repo, "add", ".")
    _git(repo, "commit", "-q", "-m", "second")
    diffs = compare(before, snapshot([repo]))
    assert any("HEAD moved" in d for d in diffs)


def test_not_a_repo(tmp_path):
    assert "error" in repo_state(tmp_path / "nope")
    assert len(assert_clean([tmp_path / "nope"])) == 1
