import hashlib
import json
import sys
import tempfile
import textwrap
from pathlib import Path

import pytest
from conftest import git, run_cli
from upstream import Upstream

from agent_rr.tape.store import TapeReader

AGENT = textwrap.dedent(
    """
    import json, os, sys, urllib.request
    from pathlib import Path

    files = sorted(p.name for p in Path(".").glob("step*.txt"))
    print(json.dumps({"cwd": os.getcwd(), "files": files}))
    base = os.environ["OPENAI_BASE_URL"]
    messages = [{"role": "system", "content": "You are a careful agent."}]
    for i in range(4):
        messages.append({"role": "user", "content": f"turn {i}"})
        body = {"model": "gpt-4o", "messages": messages, "stream": i == 1}
        req = urllib.request.Request(
            f"{base}/chat/completions",
            data=json.dumps(body).encode(),
            headers={"Content-Type": "application/json", "Authorization": "Bearer sk-" + "k" * 30},
        )
        raw = urllib.request.urlopen(req).read().decode()
        if i == 1:
            reply = "".join(
                json.loads(line[6:])["choices"][0]["delta"].get("content") or ""
                for line in raw.splitlines()
                if line.startswith("data: {")
            )
        else:
            reply = json.loads(raw)["choices"][0]["message"]["content"]
        messages.append({"role": "assistant", "content": reply})
        Path(f"step{i}.txt").write_text(reply)
        print(json.dumps({"turn": i, "reply": reply}))
    sys.exit(int(os.environ.get("AGENT_EXIT", "0")))
    """
)


@pytest.fixture
def recorded(repo: Path, tmp_path: Path, upstream: Upstream) -> tuple[Path, Path, list[str]]:
    agent = tmp_path / "agent.py"
    agent.write_text(AGENT)
    tape = tmp_path / "parent.tape"
    proc = run_cli(
        "record",
        "--out",
        str(tape),
        "--",
        sys.executable,
        str(agent),
        cwd=repo,
        env={"OPENAI_BASE_URL": f"{upstream.url}/v1"},
    )
    assert proc.returncode == 0, proc.stderr
    assert len(upstream.requests) == 4
    upstream.requests.clear()
    return tape, agent, proc.stdout.splitlines()


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_fork_mocks_prefix_and_goes_live_with_overrides(
    repo: Path, tmp_path: Path, upstream: Upstream, recorded: tuple[Path, Path, list[str]]
) -> None:
    parent, agent, parent_out = recorded
    parent_digest = _sha(parent)
    host_status = git(repo, "status", "--porcelain")
    prompt = tmp_path / "system.txt"
    prompt.write_text("You are a reckless agent.")
    child = tmp_path / "child.tape"

    proc = run_cli(
        "fork",
        str(parent),
        "--at-step",
        "2",
        "--model",
        "gpt-override",
        "--system-prompt-file",
        str(prompt),
        "--out",
        str(child),
        "--",
        sys.executable,
        str(agent),
        cwd=tmp_path,
        env={"OPENAI_BASE_URL": f"{upstream.url}/v1"},
    )
    assert proc.returncode == 0, proc.stderr
    assert "diverged" not in proc.stderr  # D13: identical prefix requests

    # Steps 0-1 were served from the parent tape: zero upstream calls for them.
    assert len(upstream.requests) == 2
    live = upstream.requests[0].body
    assert live["model"] == "gpt-override"
    assert live["messages"][0] == {"role": "system", "content": "You are a reckless agent."}
    assert [m["content"] for m in live["messages"] if m["role"] == "user"] == [
        "turn 0",
        "turn 1",
        "turn 2",
    ]

    out = [json.loads(line) for line in proc.stdout.splitlines()]
    header = out[0]
    assert Path(header["cwd"]).name.startswith("agent-rr-")
    assert Path(header["cwd"]).resolve().parent == Path(tempfile.gettempdir()).resolve()
    assert header["files"] == ["step0.txt", "step1.txt"]  # worktree = pre-state of step 2
    parent_replies = [json.loads(line) for line in parent_out[1:]]
    assert out[1:3] == parent_replies[:2]  # mocked replies identical to the recording
    assert out[3]["reply"] == "echo:6:gpt-override"

    with TapeReader(parent) as p, TapeReader(child) as c:
        pm, ps, cm, cs = p.meta, p.steps(), c.meta, c.steps()
    assert cm.parent_tape_id == pm.tape_id and cm.fork_step_idx == 2
    assert cm.base_commit_sha == pm.base_commit_sha and cm.tape_id != pm.tape_id
    assert [s.step_idx for s in cs] == [0, 1, 2, 3]
    assert cs[:2] == ps[:2]
    assert cs[2].llm is not None and json.loads(cs[2].llm.request_json)["model"] == "gpt-override"
    assert cs[2].pre_tree_sha == ps[2].pre_tree_sha  # restored state is exactly step 2's
    assert b"sk-kkkk" not in child.read_bytes()

    assert _sha(parent) == parent_digest  # parent tape is immutable
    assert git(repo, "status", "--porcelain") == host_status
    assert git(repo, "worktree", "list", "--porcelain").count("worktree ") == 1


def test_fork_returns_child_exit_code(
    tmp_path: Path, upstream: Upstream, recorded: tuple[Path, Path, list[str]]
) -> None:
    parent, agent, _ = recorded
    proc = run_cli(
        "fork",
        str(parent),
        "--at-step",
        "0",
        "--out",
        str(tmp_path / "c.tape"),
        "--",
        sys.executable,
        str(agent),
        cwd=tmp_path,
        env={"OPENAI_BASE_URL": f"{upstream.url}/v1", "AGENT_EXIT": "7"},
    )
    assert proc.returncode == 7, proc.stderr


@pytest.mark.parametrize("step", ["4", "-1", "99"])
def test_fork_invalid_step_is_exit_2(
    tmp_path: Path, recorded: tuple[Path, Path, list[str]], step: str
) -> None:
    parent, _, _ = recorded
    proc = run_cli("fork", str(parent), "--at-step", step, "--", "true", cwd=tmp_path)
    assert proc.returncode == 2
    assert proc.stderr.startswith("error: invalid step")
    assert "Traceback" not in proc.stderr


def test_fork_refuses_to_overwrite_and_missing_inputs(
    tmp_path: Path, recorded: tuple[Path, Path, list[str]]
) -> None:
    parent, _, _ = recorded
    cases = [
        ["fork", str(parent), "--at-step", "1", "--out", str(parent), "--", "true"],
        ["fork", str(tmp_path / "missing.tape"), "--at-step", "0", "--", "true"],
        ["fork", str(parent), "--at-step", "0", "--system-prompt-file", "nope.txt", "--", "true"],
    ]
    for args in cases:
        proc = run_cli(*args, cwd=tmp_path)
        assert proc.returncode == 2, args
        assert proc.stderr.startswith("error: ") and "Traceback" not in proc.stderr
