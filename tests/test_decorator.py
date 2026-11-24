import json
import time
from pathlib import Path

import pytest
from conftest import git

import agent_rr
from agent_rr import tape
from agent_rr.git_engine.snapshot import capture_tree_sha, head_sha
from agent_rr.models import Step, StepData, TapeMeta, ToolCall
from agent_rr.tape.redactor import REDACTED_SECRET
from agent_rr.tape.store import TapeReader, TapeWriter

SECRET = "ghp_" + "q" * 36


@pytest.fixture
def recording(repo: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    path = tmp_path / "rec.tape"
    monkeypatch.setenv("AGENT_RR_TAPE_PATH", str(path))
    monkeypatch.chdir(repo)
    return path


def _steps(path: Path) -> list[Step]:
    with TapeReader(path) as r:
        return r.steps()


@tape.record
def write_file(path: str, content: str) -> str:
    Path(path).write_text(content)
    return f"wrote {len(content)} bytes"


@tape.record(name="async_bash")
async def run_async(cmd: str) -> str:
    return f"ran {cmd}"


@agent_rr.record
def explode(token: str) -> str:
    raise ValueError(f"bad token {token}")


def test_public_api_exports() -> None:
    assert tape.record is agent_rr.record
    assert agent_rr.TapeReader is TapeReader and agent_rr.TapeWriter is TapeWriter


def test_passthrough_without_tape_env(repo: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(repo)
    assert write_file("x.txt", "hi") == "wrote 2 bytes"
    assert not list(repo.glob("*.tape"))


def test_sync_tool_call_is_recorded_with_diff(recording: Path, repo: Path) -> None:
    pre = capture_tree_sha(repo)
    assert write_file(path="notes.txt", content=f"token {SECRET}\n") == "wrote 47 bytes"
    [step] = _steps(recording)
    assert step.kind == "tool_call" and step.status == "ok"
    assert step.pre_tree_sha == pre
    assert step.post_tree_sha == capture_tree_sha(repo) != pre
    assert "diff --git a/notes.txt b/notes.txt" in step.diff_text
    assert SECRET not in step.diff_text and REDACTED_SECRET in step.diff_text
    assert step.tool.tool_name == "write_file"
    assert json.loads(step.tool.arguments_json) == {
        "path": "notes.txt",
        "content": f"token {REDACTED_SECRET}\n",
    }
    assert step.tool.result_text == "wrote 47 bytes"
    assert not step.tool.is_error
    with TapeReader(recording) as r:  # first writer created tape_meta
        assert r.meta.base_commit_sha == head_sha(repo)
        assert r.meta.metadata == {"producer": "agent-rr"}
        assert r.meta.repo_root == str(repo.resolve())
    assert SECRET.encode() not in recording.read_bytes()


async def test_async_tool_call_is_recorded(recording: Path) -> None:
    assert await run_async("ls -la") == "ran ls -la"
    [step] = _steps(recording)
    assert step.tool.tool_name == "async_bash"
    assert json.loads(step.tool.arguments_json) == {"cmd": "ls -la"}


def test_exception_is_recorded_and_reraised(recording: Path) -> None:
    with pytest.raises(ValueError, match="bad token"):
        explode(SECRET)
    [step] = _steps(recording)
    assert step.status == "error" and step.tool.is_error
    assert step.tool.result_text == f"ValueError: bad token {REDACTED_SECRET}"


def test_fork_mode_replays_tool_calls_before_fork_step(
    repo: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    child = tmp_path / "child.tape"
    sha = capture_tree_sha(repo)
    with TapeWriter(child) as w:
        w.ensure_meta(
            TapeMeta(
                tape_id="c" * 32,
                created_at=time.time(),
                command_argv=["agent"],
                repo_root=str(repo),
                base_commit_sha=head_sha(repo),
                parent_tape_id="p" * 32,
                fork_step_idx=1,
            )
        )
        w.append_step(
            StepData(
                ts_start=0,
                ts_end=0,
                kind="tool_call",
                pre_tree_sha=sha,
                post_tree_sha=sha,
                tool=ToolCall(tool_name="write_file", arguments_json="{}", result_text="recorded!"),
            )
        )
    monkeypatch.setenv("AGENT_RR_TAPE_PATH", str(child))
    monkeypatch.chdir(repo)
    assert write_file("mocked.txt", "never written") == "recorded!"
    assert not (repo / "mocked.txt").exists()
    assert write_file("live.txt", "abc") == "wrote 3 bytes"
    steps = _steps(child)
    assert [s.step_idx for s in steps] == [0, 1]
    assert "live.txt" in steps[1].diff_text
    assert git(repo, "status", "--porcelain") == "?? live.txt"
