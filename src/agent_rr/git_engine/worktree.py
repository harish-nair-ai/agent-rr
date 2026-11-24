"""Ephemeral `git worktree` lifecycle with guaranteed cleanup (contracts §6.3)."""

import shutil
import signal
import subprocess
import tempfile
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from types import FrameType

from agent_rr.git_engine.snapshot import run_git


class WorktreeContext:
    """Detached worktree at `commit` whose files are reset to `tree_sha` (if given)."""

    def __init__(self, repo_root: Path, commit: str, tree_sha: str | None = None) -> None:
        self.repo_root = repo_root
        self.commit = commit
        self.tree_sha = tree_sha
        self.path = Path()

    def __enter__(self) -> Path:
        self.path = Path(tempfile.mkdtemp(prefix="agent-rr-"))
        try:
            run_git(
                ["worktree", "add", "-q", "--detach", str(self.path), self.commit], self.repo_root
            )
            if self.tree_sha:
                run_git(["read-tree", "--reset", "-u", self.tree_sha], self.path)
                run_git(["clean", "-fdq"], self.path)
        except BaseException:
            self.remove()
            raise
        return self.path

    def __exit__(self, *exc: object) -> None:
        self.remove()

    def remove(self) -> None:
        subprocess.run(
            ["git", "worktree", "remove", "--force", str(self.path)],
            cwd=self.repo_root,
            capture_output=True,
        )
        shutil.rmtree(self.path, ignore_errors=True)
        subprocess.run(["git", "worktree", "prune"], cwd=self.repo_root, capture_output=True)


def _raise_exit(signum: int, _frame: FrameType | None) -> None:
    raise SystemExit(128 + signum)


@contextmanager
def signal_cleanup() -> Iterator[None]:
    """Turn SIGTERM/SIGHUP into SystemExit so every `finally` cleanup runs."""
    previous = {sig: signal.signal(sig, _raise_exit) for sig in (signal.SIGTERM, signal.SIGHUP)}
    try:
        yield
    finally:
        for sig, handler in previous.items():
            signal.signal(sig, handler)
