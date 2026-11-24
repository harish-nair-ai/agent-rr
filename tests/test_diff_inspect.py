import json
import shutil
import time
from pathlib import Path

import pytest
from agent_rr.evals.diff_viewer import first_divergence
from conftest import run_cli

from agent_rr.git_engine.snapshot import capture_tree_sha, diff_trees, head_sha
from agent_rr.models import ChaosEvent, LlmCall, StepData, TapeMeta, ToolCall
from agent_rr.tape.store import TapeWriter, read_steps

WIDE = {"COLUMNS": "200"}


def _llm(model: str, reply: str, tokens: tuple[int, int], tree: str) -> StepData:
    return StepData(
        ts_start=0,
        ts_end=0.25,
        kind="llm_call",
        pre_tree_sha=tree,
        post_tree_sha=tree,
        llm=LlmCall(
            provider="openai",
            endpoint="/v1/chat/completions",
            model=model,
            request_json=json.dumps({"model": model, "messages": []}),
            response_json=json.dumps(
                {"id": f"x{time.time()}", "choices": [{"message": {"content": reply}}]}
            ),
            http_status=200,
            input_tokens=tokens[0],
            output_tokens=tokens[1],
            latency_ms=250.0,
        ),
    )


def _edit(repo: Path, name: str, content: str, tool: str = "edit_file") -> StepData:
    pre = capture_tree_sha(repo)
    (repo / name).write_text(content)
    post = capture_tree_sha(repo)
    return StepData(
        ts_start=0,
        ts_end=0.5,
        kind="tool_call",
        pre_tree_sha=pre,
        post_tree_sha=post,
        diff_text=diff_trees(repo, pre, post),
        tool=ToolCall(tool_name=tool, arguments_json=json.dumps({"path": name}), result_text="ok"),
    )


def _write(path: Path, repo: Path, steps: list[StepData], tape_id: str) -> Path:
    with TapeWriter(path) as w:
        w.ensure_meta(
            TapeMeta(
                tape_id=tape_id * 32,
                created_at=0,
                command_argv=["agent"],
                repo_root=str(repo),
                base_commit_sha=head_sha(repo),
            )
        )
        for s in steps:
            w.append_step(s)
    return path


@pytest.fixture
def tapes(repo: Path, tmp_path: Path) -> tuple[Path, Path, Path]:
    base = capture_tree_sha(repo)
    s0 = _llm("gpt-4o", "edit app.py", (10, 5), base)
    s1 = _edit(repo, "app.py", "def add(a, b):\n    return a - b\n")
    a = _write(tmp_path / "a.tape", repo, [s0, s1], "a")
    (repo / "app.py").write_text("def add(a, b):\n    return a + b\n")
    b_steps = [
        s0,
        _edit(repo, "README.md", "# changed\n", tool="write_docs"),
        _llm("gpt-4o", "done", (20, 7), capture_tree_sha(repo)),
    ]
    b = _write(tmp_path / "b.tape", repo, b_steps, "b")
    same = tmp_path / "a_copy.tape"
    shutil.copy(a, same)
    return a, b, same


def test_diff_identical_tapes_exit_0(tapes: tuple[Path, Path, Path], tmp_path: Path) -> None:
    a, _, same = tapes
    proc = run_cli("diff", str(a), str(same), cwd=tmp_path, env=WIDE)
    assert proc.returncode == 0, proc.stderr
    assert "identical" in proc.stdout


def test_diff_diverged_tapes_exit_1(tapes: tuple[Path, Path, Path], tmp_path: Path) -> None:
    a, b, _ = tapes
    proc = run_cli("diff", str(a), str(b), cwd=tmp_path, env=WIDE)
    assert proc.returncode == 1, proc.stderr
    out = proc.stdout
    assert "diverged at step 1" in out
    assert "edit_file" in out and "write_docs" in out  # tool-sequence delta
    assert "tokens in: 10 -> 30 (+20)" in out
    assert "tokens out: 5 -> 12 (+7)" in out
    assert "diff --git a/README.md b/README.md" in out  # final trees
    assert "-    return a - b" in out and "+    return a + b" in out


def test_first_divergence_ignores_volatile_response_ids(tapes: tuple[Path, Path, Path]) -> None:
    a, _, _ = tapes
    _, steps = read_steps(a)
    llm = steps[0].llm
    assert llm is not None
    rerun = json.loads(llm.response_json) | {"id": "another-id", "created": 99}
    replay = [
        steps[0].model_copy(
            update={"llm": llm.model_copy(update={"response_json": json.dumps(rerun)})}
        ),
        *steps[1:],
    ]
    assert first_divergence(steps, replay) is None
    assert first_divergence(steps, steps[:1]) == 1


def test_inspect_timeline(tapes: tuple[Path, Path, Path], tmp_path: Path) -> None:
    _, b, _ = tapes
    proc = run_cli("inspect", str(b), cwd=tmp_path, env=WIDE)
    assert proc.returncode == 0, proc.stderr
    out = proc.stdout
    assert "b" * 32 in out
    for text in ("llm_call", "tool_call", "gpt-4o", "write_docs", "10/5", "+1 -1", "250"):
        assert text in out, text


def test_inspect_step_detail(tapes: tuple[Path, Path, Path], tmp_path: Path) -> None:
    a, _, _ = tapes
    out = run_cli("inspect", str(a), "--step", "1", cwd=tmp_path, env=WIDE).stdout
    assert "edit_file" in out and '"path": "app.py"' in out
    assert "+    return a - b" in out
    out = run_cli("inspect", str(a), "--step", "0", cwd=tmp_path, env=WIDE).stdout
    assert "edit app.py" in out and "/v1/chat/completions" in out


def test_inspect_shows_chaos_events(repo: Path, tmp_path: Path) -> None:
    sha = capture_tree_sha(repo)
    fault = StepData(
        ts_start=0,
        ts_end=0,
        kind="chaos_fault",
        status="injected_fault",
        pre_tree_sha=sha,
        post_tree_sha=sha,
        chaos=ChaosEvent(
            preset="rate-limit-429", seed=42, target="llm", detail_json='{"llm_idx":0}'
        ),
    )
    tape = _write(tmp_path / "c.tape", repo, [fault], "c")
    out = run_cli("inspect", str(tape), "--step", "0", cwd=tmp_path, env=WIDE).stdout
    assert "rate-limit-429" in out and "seed 42" in out and '"llm_idx": 0' in out


@pytest.mark.parametrize("args", [["inspect", "{a}", "--step", "5"], ["inspect", "{missing}"]])
def test_inspect_errors_exit_2(
    tapes: tuple[Path, Path, Path], tmp_path: Path, args: list[str]
) -> None:
    a, _, _ = tapes
    argv = [x.format(a=a, missing=tmp_path / "nope.tape") for x in args]
    proc = run_cli(*argv, cwd=tmp_path)
    assert proc.returncode == 2
    assert proc.stderr.startswith("error: ") and "Traceback" not in proc.stderr


def test_diff_missing_tape_exit_2(tapes: tuple[Path, Path, Path], tmp_path: Path) -> None:
    a, _, _ = tapes
    proc = run_cli("diff", str(a), str(tmp_path / "nope.tape"), cwd=tmp_path)
    assert proc.returncode == 2 and proc.stderr.startswith("error: tape not found")
