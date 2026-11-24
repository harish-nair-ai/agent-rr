import json
import os
import signal
import sqlite3
import subprocess
import sys
import textwrap
import time
from pathlib import Path

import pytest
from conftest import git, run_cli
from typer.testing import CliRunner

from agent_rr import cli
from agent_rr.tape.store import TapeReader


def test_version() -> None:
    proc = run_cli("--version", cwd=Path.cwd())
    assert proc.returncode == 0
    assert proc.stdout == "agent-rr 0.1.0\n"


@pytest.mark.parametrize("command", ["record", "inspect", "fork", "diff", "freeze", "bisect"])
def test_help(command: str) -> None:
    proc = run_cli(command, "--help", cwd=Path.cwd())
    assert proc.returncode == 0 and "Usage" in proc.stdout


def test_record_injects_env_returns_child_code_and_writes_default_tape(repo: Path) -> None:
    child = textwrap.dedent(
        """
        import json, os, sys
        keys = ["OPENAI_BASE_URL", "ANTHROPIC_BASE_URL", "AGENT_RR_TAPE_PATH"]
        print(json.dumps({k: os.environ[k] for k in keys}))
        sys.exit(5)
        """
    )
    proc = run_cli("record", "--", sys.executable, "-c", child, cwd=repo)
    assert proc.returncode == 5, proc.stderr
    env = json.loads(proc.stdout)
    port = env["OPENAI_BASE_URL"].split(":")[2].split("/")[0]
    assert env["OPENAI_BASE_URL"] == f"http://127.0.0.1:{port}/openai/v1"
    assert env["ANTHROPIC_BASE_URL"] == f"http://127.0.0.1:{port}/anthropic"
    tape = Path(env["AGENT_RR_TAPE_PATH"])
    assert tape.parent == (repo / ".agent-rr" / "tapes").resolve()
    with TapeReader(tape) as r:
        assert r.meta.command_argv == [sys.executable, "-c", child]
        assert r.meta.base_commit_sha == git(repo, "rev-parse", "HEAD")
        assert r.meta.chaos_seed is None and r.meta.metadata == {"producer": "agent-rr"}
    assert f"tape written to {tape.relative_to(repo.resolve())}" in proc.stderr or str(tape) in (
        proc.stderr
    )
    assert git(repo, "status", "--porcelain") == "?? .agent-rr/"


def test_record_outside_git_repo(tmp_path: Path) -> None:
    proc = run_cli("record", "--", "true", cwd=tmp_path)
    assert proc.returncode == 2
    assert proc.stderr.startswith("error: not a git repository")


def test_record_repo_without_commits(tmp_path: Path) -> None:
    git(tmp_path, "init", "-q")
    proc = run_cli("record", "--", "true", cwd=tmp_path)
    assert proc.returncode == 2
    assert proc.stderr == "error: repository has no commits; make an initial commit first\n"


@pytest.mark.parametrize(
    ("args", "message"),
    [
        (["record", "--", "definitely-not-a-command-xyz"], "error: cannot run"),
        (["record", "--chaos", "meteor", "--", "true"], "error: unknown chaos preset"),
        (["record", "--out", "README.md", "--", "true"], "error: tape already exists"),
    ],
)
def test_record_bad_args_exit_2(repo: Path, args: list[str], message: str) -> None:
    proc = run_cli(*args, cwd=repo)
    assert proc.returncode == 2
    assert proc.stderr.startswith(message), proc.stderr
    assert "Traceback" not in proc.stderr


def test_missing_command_is_usage_error(repo: Path) -> None:
    proc = run_cli("record", cwd=repo)
    assert proc.returncode == 2 and "Traceback" not in proc.stderr


def test_corrupt_tape_is_exit_2(tmp_path: Path) -> None:
    tape = tmp_path / "bad.tape"
    conn = sqlite3.connect(tape)
    conn.execute("CREATE TABLE tape_meta (x)")
    conn.execute("INSERT INTO tape_meta VALUES (1)")
    conn.commit()
    conn.close()
    proc = run_cli("inspect", str(tape), cwd=tmp_path)
    assert proc.returncode == 2 and proc.stderr.startswith("error: ")


def test_unexpected_error_is_reported_without_traceback(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    def boom(_: Path) -> None:
        raise RuntimeError("kaboom")

    monkeypatch.setattr(cli, "read_steps", boom)
    result = CliRunner().invoke(cli.app, ["inspect", str(tmp_path / "x.tape")])
    assert result.exit_code == 1
    assert "error: internal: RuntimeError: kaboom" in result.output
    assert "Traceback" not in result.output


def test_sigterm_during_fork_cleans_worktree_and_child(repo: Path, tmp_path: Path) -> None:
    tape = tmp_path / "t.tape"
    tool = textwrap.dedent(
        """
        from agent_rr import tape
        @tape.record
        def noop() -> str:
            return "ok"
        noop()
        """
    )
    rec = run_cli("record", "--out", str(tape), "--", sys.executable, "-c", tool, cwd=repo)
    assert rec.returncode == 0, rec.stderr
    pid_file = tmp_path / "child.pid"
    sleeper = (
        f"import os, time; open({str(pid_file)!r}, 'w').write(str(os.getpid())); time.sleep(60)"
    )
    proc = subprocess.Popen(
        [sys.executable, "-m", "agent_rr", "fork", str(tape), "--at-step", "0", "--out",
         str(tmp_path / "c.tape"), "--", sys.executable, "-c", sleeper],
        cwd=tmp_path,
        env=os.environ.copy(),
        stderr=subprocess.PIPE,
        text=True,
    )  # fmt: skip
    deadline = time.monotonic() + 30
    while not pid_file.exists() or not pid_file.read_text():
        assert time.monotonic() < deadline, "fork child never started"
        time.sleep(0.05)
    child_pid = int(pid_file.read_text())
    assert git(repo, "worktree", "list", "--porcelain").count("worktree ") == 2
    proc.send_signal(signal.SIGTERM)
    assert proc.wait(30) == 128 + signal.SIGTERM
    assert proc.stderr is not None and "Traceback" not in proc.stderr.read()
    assert git(repo, "worktree", "list", "--porcelain").count("worktree ") == 1
    with pytest.raises(ProcessLookupError):
        os.kill(child_pid, 0)
