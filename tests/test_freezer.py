import ast
import os
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest
from conftest import run_cli

from agent_rr.models import StepData, ToolCall
from agent_rr.tape.store import TapeReader, TapeWriter

AGENT = textwrap.dedent(
    """
    from pathlib import Path
    from agent_rr import tape

    @tape.record
    def edit_file(path: str, text: str) -> str:
        Path(path).write_text(text)
        return "ok"

    @tape.record
    def run_tests() -> str:
        return "3 passed"

    @tape.record
    def commit(message: str) -> str:
        return "committed"

    edit_file("app.py", "def add(a, b):\\n    return a + b  # fixed\\n")
    run_tests()
    commit("fix add")
    """
)

RUFF = Path(sys.executable).with_name("ruff")


@pytest.fixture
def fixture_tape(repo: Path, tmp_path: Path) -> Path:
    agent = tmp_path / "agent.py"
    agent.write_text(AGENT)
    tape = tmp_path / "fixture.tape"
    proc = run_cli("record", "--out", str(tape), "--", sys.executable, str(agent), cwd=repo)
    assert proc.returncode == 0, proc.stderr
    return tape


def _pytest(test_file: Path, env_tape: Path | None = None) -> subprocess.CompletedProcess[str]:
    env = {k: v for k, v in os.environ.items() if k != "AGENT_RR_TAPE_PATH"}
    if env_tape is not None:
        env["AGENT_RR_TAPE_PATH"] = str(env_tape)
    return subprocess.run(
        [sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider", str(test_file)],
        cwd=test_file.parent,
        env=env,
        capture_output=True,
        text=True,
        timeout=120,
    )


def test_freeze_generates_passing_ruff_clean_pytest_module(
    fixture_tape: Path, tmp_path: Path
) -> None:
    out = tmp_path / "evals" / "test_frozen.py"
    proc = run_cli("freeze", str(fixture_tape), "--out", str(out), cwd=tmp_path)
    assert proc.returncode == 0, proc.stderr
    source = out.read_text()

    # Metadata header.
    with TapeReader(fixture_tape) as r:
        meta = r.meta
    assert f"Source tape_id: {meta.tape_id}" in source
    assert f"Base commit: {meta.base_commit_sha}" in source
    assert "Total steps: 3" in source
    assert "Token usage:" in source
    assert "AGENT_RR_TAPE_PATH" in source

    # All five helpers, imported only from the allowed modules.
    for helper in (
        "assert_max_steps",
        "assert_no_edits_matching",
        "assert_tool_called_before",
        "assert_no_repeated_error_loop",
        "assert_final_diff_clean",
    ):
        assert f"testing.{helper}(" in source
    imports = set()
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Import):
            imports |= {a.name for a in node.names}
        elif isinstance(node, ast.ImportFrom):
            imports |= {f"{node.module}.{a.name}" for a in node.names}
    assert imports == {"os", "pathlib", "pytest", "agent_rr.testing"}
    assert '"edit_file",\n        "commit",' in source

    # D17: the reference tape is copied next to the test, checkpointed into one file.
    copy = out.with_suffix(".tape")
    assert copy.is_file() and not Path(f"{copy}-wal").exists()
    with TapeReader(copy) as r:
        assert r.meta == meta and len(r.steps()) == 3

    # Runnable, zero-API-cost, and lint/format clean.
    run = _pytest(out)
    assert run.returncode == 0, run.stdout + run.stderr
    assert "5 passed" in run.stdout
    rules = ["--isolated", "--line-length", "100", "--target-version", "py311"]
    lint = subprocess.run(
        [str(RUFF), "check", *rules, "--select", "E,F,W,I,B,UP,SIM,RUF", str(out)],
        capture_output=True,
        text=True,
    )
    assert lint.returncode == 0, lint.stdout
    fmt = subprocess.run(
        [str(RUFF), "format", *rules, "--check", str(out)], capture_output=True, text=True
    )
    assert fmt.returncode == 0, fmt.stdout + fmt.stderr


def test_frozen_test_evaluates_the_run_under_test(fixture_tape: Path, tmp_path: Path) -> None:
    out = tmp_path / "test_regress.py"
    assert run_cli("freeze", str(fixture_tape), "--out", str(out), cwd=tmp_path).returncode == 0
    with TapeReader(fixture_tape) as r:
        meta, steps = r.meta, r.steps()
    sha = steps[0].pre_tree_sha
    bad = tmp_path / "bad_run.tape"
    deletion = (
        "diff --git a/tests/test_app.py b/tests/test_app.py\n--- a/tests/test_app.py\n"
        "+++ b/tests/test_app.py\n@@ -1,2 +1,1 @@\n def test_add():\n-    assert add(1, 2) == 3\n"
    )
    with TapeWriter(bad) as w:
        w.ensure_meta(meta)
        for i in range(10):
            w.append_step(
                StepData(
                    ts_start=0,
                    ts_end=0,
                    kind="tool_call",
                    pre_tree_sha=sha,
                    post_tree_sha=sha,
                    diff_text=deletion if i == 0 else "",
                    tool=ToolCall(
                        tool_name="commit", arguments_json="{}", result_text="E", is_error=True
                    ),
                )
            )
    run = _pytest(out, env_tape=bad)
    assert run.returncode == 1
    for failing in (
        "test_step_budget",
        "test_tool_order",
        "test_no_repeated_error_loop",
        "test_final_diff_clean",
    ):
        assert failing in run.stdout, run.stdout


def test_freeze_missing_tape_is_exit_2(tmp_path: Path) -> None:
    proc = run_cli(
        "freeze", str(tmp_path / "x.tape"), "--out", str(tmp_path / "t.py"), cwd=tmp_path
    )
    assert proc.returncode == 2
    assert proc.stderr.startswith("error: tape not found")
    assert not (tmp_path / "t.py").exists()
