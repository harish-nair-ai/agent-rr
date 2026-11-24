"""Shared fixtures: hermetic git identity/config and real temporary git repositories."""

from __future__ import annotations

import os
import subprocess
import sys
import tempfile
from collections.abc import Iterator
from pathlib import Path

import pytest
from upstream import Upstream

_SCRUBBED_ENV = (
    "AGENT_RR_TAPE_PATH",
    "AGENT_RR_CHAOS",
    "OPENAI_BASE_URL",
    "ANTHROPIC_BASE_URL",
    "GIT_DIR",
    "GIT_WORK_TREE",
    "GIT_INDEX_FILE",
)


@pytest.fixture(autouse=True)
def hermetic_env(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in _SCRUBBED_ENV:
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("GIT_CONFIG_GLOBAL", os.devnull)
    monkeypatch.setenv("GIT_CONFIG_NOSYSTEM", "1")
    for role in ("AUTHOR", "COMMITTER"):
        monkeypatch.setenv(f"GIT_{role}_NAME", "Test Bot")
        monkeypatch.setenv(f"GIT_{role}_EMAIL", "bot@example.com")


def git(repo: Path, *args: str) -> str:
    return subprocess.run(
        ["git", *args], cwd=repo, check=True, capture_output=True, text=True
    ).stdout.strip()


def commit_all(repo: Path, message: str) -> str:
    git(repo, "add", "-A")
    git(repo, "commit", "-q", "-m", message)
    return git(repo, "rev-parse", "HEAD")


@pytest.fixture
def repo(tmp_path: Path) -> Path:
    root = tmp_path / "repo"
    root.mkdir()
    git(root, "init", "-q", "-b", "main")
    (root / "app.py").write_text("def add(a, b):\n    return a + b\n")
    (root / "README.md").write_text("# demo\n")
    commit_all(root, "initial")
    return root


@pytest.fixture(autouse=True)
def no_leaked_worktree_dirs() -> Iterator[None]:
    tmp = Path(tempfile.gettempdir())
    before = set(tmp.glob("agent-rr-*"))
    yield
    leaked = set(tmp.glob("agent-rr-*")) - before
    assert not leaked, f"leaked temp dirs: {sorted(leaked)}"


@pytest.fixture
def upstream() -> Iterator[Upstream]:
    server = Upstream()
    server.start()
    yield server
    server.stop()


def run_cli(
    *args: str, cwd: Path, env: dict[str, str] | None = None, timeout: float = 120
) -> subprocess.CompletedProcess[str]:
    """Run the real `agent-rr` CLI in a subprocess."""
    return subprocess.run(
        [sys.executable, "-m", "agent_rr", *args],
        cwd=cwd,
        env={**os.environ, **(env or {})},
        capture_output=True,
        text=True,
        timeout=timeout,
    )
