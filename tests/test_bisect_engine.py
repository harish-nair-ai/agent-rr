import re
import sys
import textwrap
from pathlib import Path

import pytest
from agent_rr.bisect.engine import majority_threshold, overlay_tree
from conftest import commit_all, git, run_cli

from agent_rr.git_engine.snapshot import capture_tree_sha

AGENT = textwrap.dedent(
    """
    import os
    from pathlib import Path
    from agent_rr import tape

    @tape.record
    def write_result(text: str) -> str:
        Path("result.txt").write_text(text)
        return "ok"

    log = os.environ.get("EVAL_LOG")
    run = 0
    if log:
        with open(log, "a") as f:
            f.write(Path("notes.txt").read_text().strip() + "\\n")
        run = len(Path(log).read_text().splitlines())
    rules = Path("AGENTS.md").read_text()
    flaky = os.environ.get("FLAKY") == "1" and run % 3 == 0  # 1 of every 3 runs misbehaves
    write_result("BROKEN" if "BROKEN" in rules or flaky else "ok")
    """
)
CHECK = (
    f'{sys.executable} -c "import pathlib, sys; '
    f"sys.exit(pathlib.Path('result.txt').read_text() != 'ok')\""
)


@pytest.fixture
def history(tmp_path: Path) -> tuple[Path, list[str], Path, Path]:
    """8 commits; commit #5 breaks AGENTS.md. Returns (repo, shas, tape, agent)."""
    repo = tmp_path / "harness"
    repo.mkdir()
    git(repo, "init", "-q", "-b", "main")
    shas = []
    for i in range(1, 9):
        (repo / "notes.txt").write_text(f"{i}\n")
        if i == 1:
            (repo / "AGENTS.md").write_text("Always write ok.\n")
        if i == 5:
            (repo / "AGENTS.md").write_text("Always write BROKEN.\n")
        shas.append(commit_all(repo, f"commit {i}" + (": tweak AGENTS.md" if i == 5 else "")))
    agent = tmp_path / "agent.py"
    agent.write_text(AGENT)
    tape = tmp_path / "session.tape"
    proc = run_cli("record", "--out", str(tape), "--", sys.executable, str(agent), cwd=repo)
    assert proc.returncode == 0, proc.stderr
    (repo / "result.txt").unlink()
    assert git(repo, "status", "--porcelain") == ""
    return repo, shas, tape, agent


def _bisect(
    repo: Path, good: str, bad: str, tape: Path, agent: Path, log: Path, *extra: str, **env: str
) -> tuple[int, str, str]:
    proc = run_cli(
        "bisect",
        "--good",
        good,
        "--bad",
        bad,
        "--tape",
        str(tape),
        "--at-step",
        "0",
        "--check",
        CHECK,
        "--trials",
        "3",
        *extra,
        "--",
        sys.executable,
        str(agent),
        cwd=repo,
        env={"EVAL_LOG": str(log), **env},
        timeout=300,
    )
    return proc.returncode, proc.stdout, proc.stderr


def _host(repo: Path) -> tuple[bytes, str, str, str]:
    return (
        (repo / ".git" / "index").read_bytes(),
        git(repo, "status", "--porcelain", "--untracked-files=all"),
        git(repo, "rev-parse", "HEAD"),
        git(repo, "show-ref"),
    )


def test_majority_threshold_is_ceil_two_thirds() -> None:
    assert [majority_threshold(k) for k in (1, 2, 3, 4, 5, 6)] == [1, 2, 2, 3, 4, 4]


def test_bisect_pinpoints_commit_5_within_4_evaluations(
    history: tuple[Path, list[str], Path, Path], tmp_path: Path
) -> None:
    repo, shas, tape, agent = history
    before = _host(repo)
    log = tmp_path / "evals.log"
    code, out, err = _bisect(repo, shas[0], shas[-1], tape, agent, log, "--skip-endpoint-check")
    assert code == 0, err
    assert f"first bad commit: {shas[4]}" in out
    assert "Author: Test Bot <bot@example.com>" in out
    assert "commit 5: tweak AGENTS.md" in out
    assert "-Always write ok.\n+Always write BROKEN." in out
    found = re.search(r"evaluations: (\d+)", out)
    assert found is not None
    evaluations = int(found.group(1))
    assert evaluations <= 4
    runs = log.read_text().splitlines()
    assert len(runs) == 3 * evaluations  # K=3 independent trials per evaluation, no endpoints
    # Workspace state came from the tape (commit 8's notes.txt), harness from commit C.
    assert set(runs) == {"8"}
    assert _host(repo) == before
    assert git(repo, "worktree", "list", "--porcelain").count("worktree ") == 1
    assert "Traceback" not in err


def test_bisect_majority_vote_tolerates_one_flaky_trial(
    history: tuple[Path, list[str], Path, Path], tmp_path: Path
) -> None:
    repo, shas, tape, agent = history
    log = tmp_path / "evals.log"
    code, out, err = _bisect(repo, shas[0], shas[-1], tape, agent, log, FLAKY="1")
    assert code == 0, err
    assert f"first bad commit: {shas[4]}" in out
    assert "2/3 trials passed" in err  # good commits survive one flaky failure (D15)
    endpoint_runs = 6
    assert (len(log.read_text().splitlines()) - endpoint_runs) // 3 <= 4


def test_bisect_inconclusive_when_bad_endpoint_passes(
    history: tuple[Path, list[str], Path, Path], tmp_path: Path
) -> None:
    repo, shas, tape, agent = history
    code, _, err = _bisect(repo, shas[0], shas[2], tape, agent, tmp_path / "log")
    assert code == 3
    assert "inconclusive" in err and "--bad" in err
    assert git(repo, "worktree", "list", "--porcelain").count("worktree ") == 1


def test_bisect_bad_args(history: tuple[Path, list[str], Path, Path], tmp_path: Path) -> None:
    repo, shas, tape, agent = history
    code, _, err = _bisect(repo, "no-such-rev", shas[-1], tape, agent, tmp_path / "log")
    assert code == 2 and err.startswith("error: ")
    code, _, err = _bisect(
        repo, shas[0], shas[-1], tmp_path / "missing.tape", agent, tmp_path / "l"
    )
    assert code == 2 and "tape not found" in err


def test_overlay_keeps_harness_paths_from_commit(
    history: tuple[Path, list[str], Path, Path],
) -> None:
    repo, shas, _, _ = history
    (repo / "prompts").mkdir()
    (repo / "prompts" / "sys.txt").write_text("workspace prompt\n")
    (repo / "extra.py").write_text("x = 1\n")
    workspace = capture_tree_sha(repo)
    tree = overlay_tree(repo, shas[0], workspace, ["AGENTS.md", "prompts/**"])
    files = git(repo, "ls-tree", "-r", "--name-only", tree).splitlines()
    assert sorted(files) == ["AGENTS.md", "extra.py", "notes.txt"]  # prompts/ absent at commit 1
    assert git(repo, "show", f"{tree}:AGENTS.md") == "Always write ok."
    assert git(repo, "show", f"{tree}:notes.txt") == "8"
