from pathlib import Path

import pytest

from agent_rr import testing
from agent_rr.models import StepData, TapeMeta, ToolCall
from agent_rr.tape.store import TapeWriter

SHA = "a" * 40


def _diff(path: str, removed: list[str], added: list[str]) -> str:
    body = "".join(f"-{line}\n" for line in removed) + "".join(f"+{line}\n" for line in added)
    return (
        f"diff --git a/{path} b/{path}\nindex 1..2 100644\n--- a/{path}\n+++ b/{path}\n"
        f"@@ -1,{len(removed)} +1,{len(added)} @@\n{body}"
    )


def _tape(tmp_path: Path, steps: list[StepData]) -> Path:
    path = tmp_path / "t.tape"
    with TapeWriter(path) as w:
        w.ensure_meta(
            TapeMeta(
                tape_id="e" * 32,
                created_at=0,
                command_argv=[],
                repo_root="/r",
                base_commit_sha="c" * 40,
            )
        )
        for step in steps:
            w.append_step(step)
    return path


def _tool(name: str, diff: str = "", error: str | None = None, status: str = "ok") -> StepData:
    return StepData.model_validate(
        {
            "ts_start": 0,
            "ts_end": 0,
            "kind": "tool_call",
            "status": status,
            "pre_tree_sha": SHA,
            "post_tree_sha": SHA,
            "diff_text": diff,
            "tool": ToolCall(
                tool_name=name,
                arguments_json="{}",
                result_text=error or "ok",
                is_error=error is not None,
            ),
        }
    )


def test_assert_max_steps(tmp_path: Path) -> None:
    tape = _tape(tmp_path, [_tool("a")] * 3)
    testing.assert_max_steps(tape, limit=3)
    with pytest.raises(AssertionError, match="3 steps > limit 2"):
        testing.assert_max_steps(str(tape), limit=2)


def test_assert_no_edits_matching(tmp_path: Path) -> None:
    tape = _tape(
        tmp_path,
        [
            _tool("edit", _diff("src/app.py", ["a"], ["b"])),
            _tool("edit", _diff("migrations/001.sql", ["x"], ["y"]), status="steered"),
        ],
    )
    # The steered migration edit was rejected, never applied.
    testing.assert_no_edits_matching(tape, globs=["migrations/**", "*.lock"])
    with pytest.raises(AssertionError, match=r"src/app\.py"):
        testing.assert_no_edits_matching(tape, globs=["src/*"])


def test_assert_tool_called_before(tmp_path: Path) -> None:
    tape = _tape(tmp_path, [_tool("read"), _tool("pytest"), _tool("commit"), _tool("pytest")])
    testing.assert_tool_called_before(tape, "pytest", "commit")
    with pytest.raises(AssertionError, match="'read' was called before 'commit'"):
        testing.assert_tool_called_before(tape, "commit", "read")
    with pytest.raises(AssertionError, match="never called"):
        testing.assert_tool_called_before(tape, "pytest", "deploy")


def test_assert_no_repeated_error_loop(tmp_path: Path) -> None:
    same = [_tool("pytest", error="E: boom")] * 3
    tape = _tape(tmp_path, [*same, _tool("pytest", error="E: other")])
    testing.assert_no_repeated_error_loop(tape, max_repeats=3)
    with pytest.raises(AssertionError, match="repeated 3 times"):
        testing.assert_no_repeated_error_loop(tape, max_repeats=2)


def test_assert_final_diff_clean(tmp_path: Path) -> None:
    clean = _tape(
        tmp_path,
        [
            _tool("e", _diff("tests/test_a.py", ["    assert f() == 1"], ["    assert f() == 2"])),
            _tool("e", _diff("src/a.py", ["    assert x"], [])),  # not a test file
        ],
    )
    testing.assert_final_diff_clean(clean)
    clean.unlink()
    deleted = _tape(tmp_path, [_tool("e", _diff("tests/test_a.py", ["    assert f() == 1"], []))])
    with pytest.raises(AssertionError, match="deleted 1 assertion"):
        testing.assert_final_diff_clean(deleted)
    deleted.unlink()
    ignore = _tape(tmp_path, [_tool("e", _diff("src/a.py", [], ["x = f()  # type: ignore"]))])
    with pytest.raises(AssertionError, match="type: ignore"):
        testing.assert_final_diff_clean(ignore)
