"""Non-destructive tree snapshots via an isolated GIT_INDEX_FILE (contracts §6.1)."""

import os
import subprocess
import tempfile
from collections.abc import Mapping
from pathlib import Path

from agent_rr.errors import AgentRRError

NO_COMMITS = "repository has no commits; make an initial commit first"


def run_git(args: list[str], cwd: Path, env: Mapping[str, str] | None = None) -> str:
    proc = subprocess.run(
        ["git", *args],
        cwd=cwd,
        env=None if env is None else dict(env),
        capture_output=True,
        text=True,
    )
    if proc.returncode != 0:
        raise AgentRRError(f"git {' '.join(args)} failed: {proc.stderr.strip()}")
    return proc.stdout


def repo_toplevel(path: Path) -> Path:
    proc = subprocess.run(
        ["git", "rev-parse", "--show-toplevel"], cwd=path, capture_output=True, text=True
    )
    if proc.returncode != 0:
        raise AgentRRError(f"not a git repository: {path}")
    return Path(proc.stdout.strip())


def head_sha(repo_root: Path) -> str:
    repo_toplevel(repo_root)
    proc = subprocess.run(
        ["git", "rev-parse", "--verify", "-q", "HEAD^{commit}"],
        cwd=repo_root,
        capture_output=True,
        text=True,
    )
    if proc.returncode != 0:
        raise AgentRRError(NO_COMMITS)
    return proc.stdout.strip()


def capture_tree_sha(repo_root: Path) -> str:
    """Write the working tree (minus ignored and agent runtime dirs) as a git tree object."""
    head_sha(repo_root)
    fd, tmp_idx = tempfile.mkstemp(dir=tempfile.gettempdir())
    os.close(fd)
    os.unlink(tmp_idx)  # git initialises a fresh index at this path
    env = {**os.environ, "GIT_INDEX_FILE": tmp_idx}
    try:
        run_git(["read-tree", "HEAD"], repo_root, env)
        run_git(["add", "-A", "--", ".", ":!.agent-rr", ":!.flywire"], repo_root, env)
        return run_git(["write-tree"], repo_root, env).strip()
    finally:
        for leftover in (tmp_idx, tmp_idx + ".lock"):
            if os.path.exists(leftover):
                os.unlink(leftover)


def diff_trees(repo_root: Path, pre: str, post: str) -> str:
    if pre == post:
        return ""
    return run_git(["diff-tree", "-p", "--no-color", pre, post], repo_root)
