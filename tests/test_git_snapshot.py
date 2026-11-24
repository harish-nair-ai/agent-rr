import os
import re
import signal
import subprocess
import sys
import tempfile
import textwrap
from pathlib import Path

import pytest
from conftest import commit_all, git

from agent_rr.errors import AgentRRError
from agent_rr.git_engine.snapshot import capture_tree_sha, diff_trees, head_sha
from agent_rr.git_engine.worktree import WorktreeContext


def _host_state(repo: Path) -> dict[str, object]:
    return {
        "index": (repo / ".git" / "index").read_bytes(),
        "status": git(repo, "status", "--porcelain=v1", "--untracked-files=all"),
        "cached": git(repo, "diff", "--cached"),
        "head": git(repo, "rev-parse", "HEAD"),
        "symbolic": git(repo, "symbolic-ref", "HEAD"),
        "refs": git(repo, "show-ref"),
        "worktrees": git(repo, "worktree", "list", "--porcelain"),
    }


def _dirty(repo: Path) -> None:
    (repo / "app.py").write_text("def add(a, b):\n    return a + b + 0\n")
    git(repo, "add", "app.py")  # staged change
    (repo / "README.md").write_text("# demo\nunstaged line\n")  # unstaged change
    (repo / "new.txt").write_text("untracked\n")
    (repo / ".gitignore").write_text("ignored.log\n")
    (repo / "ignored.log").write_text("noise\n")
    for runtime in (".agent-rr", ".flywire"):
        (repo / runtime).mkdir()
        (repo / runtime / "state.db").write_text("x")


def _tree_files(repo: Path, tree: str) -> set[str]:
    return set(git(repo, "ls-tree", "-r", "--name-only", tree).splitlines())


def test_capture_is_valid_tree_and_leaves_index_untouched(repo: Path) -> None:
    _dirty(repo)
    before = _host_state(repo)
    sha = capture_tree_sha(repo)
    assert re.fullmatch(r"[0-9a-f]{40}", sha)
    assert git(repo, "cat-file", "-t", sha) == "tree"
    assert _host_state(repo) == before  # index bytes, staged changes, HEAD, refs identical
    files = _tree_files(repo, sha)
    assert {"app.py", "README.md", "new.txt", ".gitignore"} <= files
    assert "ignored.log" not in files
    assert not any(f.startswith((".agent-rr/", ".flywire/")) for f in files)
    assert git(repo, "show", f"{sha}:README.md") == "# demo\nunstaged line"


def test_capture_is_deterministic_and_cleans_temp_index(
    repo: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    scratch = tmp_path / "scratch"
    scratch.mkdir()
    monkeypatch.setattr(tempfile, "tempdir", str(scratch))
    assert capture_tree_sha(repo) == capture_tree_sha(repo) == git(repo, "rev-parse", "HEAD^{tree}")
    assert list(scratch.iterdir()) == []


def test_diff_trees_is_git_unified_diff(repo: Path) -> None:
    pre = capture_tree_sha(repo)
    (repo / "app.py").write_text("def add(a, b):\n    return b + a\n")
    (repo / "added.py").write_text("X = 1\n")
    post = capture_tree_sha(repo)
    diff = diff_trees(repo, pre, post)
    assert "diff --git a/app.py b/app.py" in diff
    assert "--- /dev/null\n+++ b/added.py" in diff
    assert "-    return a + b\n+    return b + a" in diff
    assert re.search(r"^@@ ", diff, re.MULTILINE)
    assert diff_trees(repo, pre, pre) == ""


def test_repo_without_commits_is_actionable_error(tmp_path: Path) -> None:
    empty = tmp_path / "empty"
    empty.mkdir()
    git(empty, "init", "-q")
    for fn in (capture_tree_sha, head_sha):
        with pytest.raises(AgentRRError) as exc:
            fn(empty)
        assert str(exc.value) == "repository has no commits; make an initial commit first"
        assert exc.value.exit_code == 2


def test_not_a_repo_is_actionable_error(tmp_path: Path) -> None:
    with pytest.raises(AgentRRError, match="not a git repository"):
        capture_tree_sha(tmp_path)


def test_worktree_restores_exact_tree_and_is_removed(repo: Path) -> None:
    base = head_sha(repo)
    _dirty(repo)
    (repo / "README.md").unlink()
    tree = capture_tree_sha(repo)
    before = _host_state(repo)
    with WorktreeContext(repo, base, tree) as wt:
        assert wt.name.startswith("agent-rr-")
        assert wt.resolve().parent == Path(tempfile.gettempdir()).resolve()
        assert (wt / "app.py").read_text() == "def add(a, b):\n    return a + b + 0\n"
        assert (wt / "new.txt").read_text() == "untracked\n"
        assert not (wt / "README.md").exists()
        assert not (wt / "ignored.log").exists()
        assert capture_tree_sha(wt) == tree
        assert len(git(repo, "worktree", "list").splitlines()) == 2
    assert not wt.exists()
    assert _host_state(repo) == before
    assert len(git(repo, "worktree", "list").splitlines()) == 1


def test_worktree_without_tree_is_commit_checkout_and_cleans_on_error(repo: Path) -> None:
    first = head_sha(repo)
    (repo / "app.py").write_text("changed\n")
    commit_all(repo, "second")
    with pytest.raises(RuntimeError), WorktreeContext(repo, first) as wt:
        assert (wt / "app.py").read_text() == "def add(a, b):\n    return a + b\n"
        raise RuntimeError("boom")
    assert not wt.exists()
    assert git(repo, "worktree", "list", "--porcelain").count("worktree ") == 1


def test_bad_tree_sha_cleans_up(repo: Path) -> None:
    with pytest.raises(AgentRRError), WorktreeContext(repo, head_sha(repo), "f" * 40):
        pass
    assert git(repo, "worktree", "list", "--porcelain").count("worktree ") == 1


def test_sigterm_removes_worktree(repo: Path) -> None:
    script = textwrap.dedent(
        f"""
        import sys, time
        from pathlib import Path
        from agent_rr.git_engine.worktree import WorktreeContext, signal_cleanup
        from agent_rr.git_engine.snapshot import head_sha
        repo = Path({str(repo)!r})
        with signal_cleanup(), WorktreeContext(repo, head_sha(repo)) as wt:
            print(wt, flush=True)
            time.sleep(60)
        """
    )
    proc = subprocess.Popen(
        [sys.executable, "-c", script], stdout=subprocess.PIPE, text=True, env=os.environ.copy()
    )
    assert proc.stdout is not None
    wt = Path(proc.stdout.readline().strip())
    assert wt.is_dir()
    proc.send_signal(signal.SIGTERM)
    assert proc.wait(30) == 128 + signal.SIGTERM
    assert not wt.exists()
    assert git(repo, "worktree", "list", "--porcelain").count("worktree ") == 1
