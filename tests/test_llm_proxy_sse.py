import json
import threading
import time
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

import httpx
import pytest
from conftest import git
from upstream import SSE_CHUNK_COUNT, Upstream, openai_sse_events

from agent_rr.git_engine.snapshot import head_sha
from agent_rr.models import LlmCall, Step, TapeMeta
from agent_rr.proxy.llm_proxy import ProxyState, build_app, proxy_env, serve
from agent_rr.proxy.replay_router import ReplayRouter
from agent_rr.tape.redactor import REDACTED_SECRET
from agent_rr.tape.store import TapeReader, TapeWriter

SECRET = "sk-live_" + "Z" * 40


@contextmanager
def running_proxy(
    repo: Path, tape: Path, upstream: Upstream, router: ReplayRouter | None = None
) -> Iterator[str]:
    writer = TapeWriter(tape)
    writer.ensure_meta(
        TapeMeta(
            tape_id="f" * 32,
            created_at=time.time(),
            command_argv=["test"],
            repo_root=str(repo),
            base_commit_sha=head_sha(repo),
        )
    )
    state = ProxyState(
        writer=writer,
        snapshot_root=repo,
        openai_upstream=f"{upstream.url}/v1",
        anthropic_upstream=upstream.url,
        router=router or ReplayRouter([], 0),
    )
    try:
        with serve(build_app(state)) as port:
            yield f"http://127.0.0.1:{port}"
    finally:
        writer.close()


def _steps(tape: Path) -> list[Step]:
    with TapeReader(tape) as r:
        return r.steps()


def test_openai_sse_is_streamed_chunk_by_chunk_and_recorded(
    repo: Path, tmp_path: Path, upstream: Upstream
) -> None:
    tape = tmp_path / "s.tape"
    upstream.gate = threading.Event()
    received: list[bytes] = []
    body = {
        "model": "gpt-4o",
        "stream": True,
        "messages": [{"role": "user", "content": f"my key is {SECRET}"}],
    }
    with (
        running_proxy(repo, tape, upstream) as base,
        httpx.Client(timeout=30) as client,
        client.stream(
            "POST",
            f"{base}/openai/v1/chat/completions",
            json=body,
            headers={"Authorization": f"Bearer {SECRET}"},
        ) as resp,
    ):
        assert resp.status_code == 200
        assert resp.headers["content-type"].startswith("text/event-stream")
        for chunk in resp.iter_raw():
            received.append(chunk)
            upstream.gate.set()  # upstream only continues once we saw the first bytes
    # The first chunk reached the client while upstream was still blocked: true streaming.
    assert upstream.gate_released_in_time is True
    assert b"".join(received) == b"".join(upstream.sent)
    events = [e for e in b"".join(received).split(b"\n\n") if e]
    assert len(events) == SSE_CHUNK_COUNT

    # Upstream got the real credential and the unredacted body.
    req = upstream.requests[0]
    assert req.path == "/v1/chat/completions"
    assert req.headers["authorization"] == f"Bearer {SECRET}"
    assert req.body == body

    [step] = _steps(tape)
    assert step.kind == "llm_call" and step.status == "ok"
    assert step.pre_tree_sha == step.post_tree_sha == git(repo, "rev-parse", "HEAD^{tree}")
    llm = step.llm
    assert llm is not None
    assert llm.provider == "openai"
    assert llm.endpoint == "/v1/chat/completions"
    assert llm.model == "gpt-4o"
    assert llm.http_status == 200
    assert llm.raw_sse_chunks is not None and len(llm.raw_sse_chunks) == SSE_CHUNK_COUNT
    assert "".join(llm.raw_sse_chunks).encode() == b"".join(upstream.sent)
    resp_obj = json.loads(llm.response_json)
    assert resp_obj["choices"][0]["message"] == {
        "role": "assistant",
        "content": "w0 w1 w2 w3 w4 w5 w6 ",
    }
    assert resp_obj["choices"][0]["finish_reason"] == "stop"
    assert (llm.input_tokens, llm.output_tokens) == (11, 7)
    assert json.loads(llm.request_json)["messages"][0]["content"] == f"my key is {REDACTED_SECRET}"
    assert llm.latency_ms > 0
    assert SECRET.encode() not in tape.read_bytes()


def test_anthropic_sse_reassembles_text_and_tool_use(
    repo: Path, tmp_path: Path, upstream: Upstream
) -> None:
    tape = tmp_path / "s.tape"
    with running_proxy(repo, tape, upstream) as base:
        resp = httpx.post(
            f"{base}/anthropic/v1/messages",
            json={"model": "claude-x", "stream": True, "max_tokens": 5, "messages": []},
            headers={"x-api-key": SECRET},
            timeout=30,
        )
    assert resp.status_code == 200
    assert upstream.requests[0].path == "/v1/messages"
    assert upstream.requests[0].headers["x-api-key"] == SECRET
    [step] = _steps(tape)
    llm = step.llm
    assert llm is not None and llm.provider == "anthropic" and llm.endpoint == "/v1/messages"
    msg = json.loads(llm.response_json)
    assert msg["content"] == [
        {"type": "text", "text": "Hello"},
        {"type": "tool_use", "id": "tu_1", "name": "bash", "input": {"cmd": "ls"}},
    ]
    assert msg["stop_reason"] == "tool_use"
    assert (llm.input_tokens, llm.output_tokens) == (21, 9)
    assert SECRET.encode() not in tape.read_bytes()


def test_non_streaming_json_roundtrip(repo: Path, tmp_path: Path, upstream: Upstream) -> None:
    tape = tmp_path / "s.tape"
    with running_proxy(repo, tape, upstream) as base:
        resp = httpx.post(
            f"{base}/openai/v1/chat/completions?api-version=1",
            json={"model": "gpt-4o", "messages": [{"role": "user", "content": "hi"}]},
            timeout=30,
        )
    assert resp.json()["choices"][0]["message"]["content"] == "echo:1:gpt-4o"
    assert upstream.requests[0].path == "/v1/chat/completions?api-version=1"
    [step] = _steps(tape)
    assert step.llm is not None and step.llm.raw_sse_chunks is None
    assert json.loads(step.llm.response_json) == resp.json()
    assert (step.llm.input_tokens, step.llm.output_tokens) == (3, 2)


def test_upstream_error_status_is_passed_through_and_recorded(
    repo: Path, tmp_path: Path, upstream: Upstream
) -> None:
    tape = tmp_path / "s.tape"
    upstream.fail_status = 500
    with running_proxy(repo, tape, upstream) as base:
        resp = httpx.post(f"{base}/openai/v1/chat/completions", json={"model": "m"}, timeout=30)
    assert resp.status_code == 500
    [step] = _steps(tape)
    assert step.status == "error" and step.llm is not None and step.llm.http_status == 500


def test_unreachable_upstream_returns_502_and_records_error(repo: Path, tmp_path: Path) -> None:
    dead = Upstream()
    dead.url = "http://127.0.0.1:9"  # discard port: connection refused
    tape = tmp_path / "s.tape"
    with running_proxy(repo, tape, dead) as base:
        resp = httpx.post(f"{base}/openai/v1/chat/completions", json={"model": "m"}, timeout=30)
    assert resp.status_code == 502
    [step] = _steps(tape)
    assert step.status == "error" and step.llm is not None and step.llm.http_status == 502


@pytest.mark.parametrize("payload", [b"not json", b"[1, 2]"])
def test_non_object_body_is_rejected_without_forwarding(
    repo: Path, tmp_path: Path, upstream: Upstream, payload: bytes
) -> None:
    tape = tmp_path / "s.tape"
    with running_proxy(repo, tape, upstream) as base:
        resp = httpx.post(f"{base}/openai/v1/chat/completions", content=payload, timeout=30)
    assert resp.status_code == 400
    assert upstream.requests == []
    assert _steps(tape) == []


def test_mock_phase_replays_without_upstream_and_warns_on_divergence(
    repo: Path, tmp_path: Path, upstream: Upstream, capfd: pytest.CaptureFixture[str]
) -> None:
    recorded_req = {"model": "gpt-4o", "messages": [{"role": "user", "content": "a"}]}
    chunks = [e.decode() for e in openai_sse_events("gpt-4o")]
    recorded = [
        LlmCall(
            provider="openai",
            endpoint="/v1/chat/completions",
            model="gpt-4o",
            request_json=json.dumps(recorded_req),
            response_json=json.dumps({"recorded": 0}),
            http_status=200,
            latency_ms=1.0,
        ),
        LlmCall(
            provider="openai",
            endpoint="/v1/chat/completions",
            model="gpt-4o",
            request_json=json.dumps({**recorded_req, "stream": True}),
            response_json="{}",
            raw_sse_chunks=chunks,
            http_status=200,
            latency_ms=1.0,
        ),
    ]
    tape = tmp_path / "child.tape"
    router = ReplayRouter(recorded, fork_llm_idx=2, model="gpt-override")
    with running_proxy(repo, tape, upstream, router) as base:
        first = httpx.post(f"{base}/openai/v1/chat/completions", json=recorded_req, timeout=30)
        second = httpx.post(
            f"{base}/openai/v1/chat/completions",
            json={**recorded_req, "stream": True, "messages": []},  # diverged request
            timeout=30,
        )
        assert upstream.requests == []
        live = httpx.post(f"{base}/openai/v1/chat/completions", json=recorded_req, timeout=30)
    assert first.status_code == 200 and first.json() == {"recorded": 0}
    assert second.status_code == 200 and second.content.decode() == "".join(chunks)
    assert second.headers["content-type"].startswith("text/event-stream")
    assert [r.body["model"] for r in upstream.requests] == ["gpt-override"]
    assert live.json()["choices"][0]["message"]["content"] == "echo:1:gpt-override"
    err = capfd.readouterr().err
    assert "llm request 1 diverged from the parent tape" in err
    assert "llm request 0 diverged" not in err
    [step] = _steps(tape)  # mocked requests are not re-recorded
    assert step.llm is not None and json.loads(step.llm.request_json)["model"] == "gpt-override"


def test_proxy_env_injection() -> None:
    env = proxy_env(4321, Path("/x/s.tape"))
    assert env == {
        "OPENAI_BASE_URL": "http://127.0.0.1:4321/openai/v1",
        "ANTHROPIC_BASE_URL": "http://127.0.0.1:4321/anthropic",
        "AGENT_RR_TAPE_PATH": "/x/s.tape",
    }
