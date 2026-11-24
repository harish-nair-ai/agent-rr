"""SSE-transparent OpenAI/Anthropic reverse proxy that records every call into a tape (§5.2)."""

from __future__ import annotations

import asyncio
import json
import socket
import sys
import threading
import time
from collections.abc import AsyncIterator, Iterator
from contextlib import asynccontextmanager, contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import httpx
import uvicorn
from pydantic import TypeAdapter, ValidationError
from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import JSONResponse, Response, StreamingResponse
from starlette.routing import Route

from agent_rr.chaos.injector import ChaosInjector
from agent_rr.errors import AgentRRError
from agent_rr.git_engine.snapshot import capture_tree_sha
from agent_rr.models import ChaosEvent, LlmCall, StepData
from agent_rr.proxy.replay_router import ReplayRouter, canonical_json, request_digest
from agent_rr.proxy.sse import event_payloads, reassemble, split_events, token_usage
from agent_rr.tape.redactor import redact_payload, redact_text
from agent_rr.tape.store import TapeWriter

DEFAULT_OPENAI = "https://api.openai.com/v1"
DEFAULT_ANTHROPIC = "https://api.anthropic.com"
_HOP_HEADERS = frozenset(
    {"host", "content-length", "connection", "keep-alive", "transfer-encoding", "accept-encoding"}
)
_DROP_RESPONSE_HEADERS = frozenset(
    {"content-length", "transfer-encoding", "content-encoding", "connection", "keep-alive"}
)
_JSON_OBJECT: TypeAdapter[dict[str, Any]] = TypeAdapter(dict[str, Any])


@dataclass
class ProxyState:
    writer: TapeWriter
    snapshot_root: Path
    openai_upstream: str = DEFAULT_OPENAI
    anthropic_upstream: str = DEFAULT_ANTHROPIC
    router: ReplayRouter = field(default_factory=lambda: ReplayRouter([], 0))
    chaos: ChaosInjector | None = None
    llm_idx: int = 0
    client: httpx.AsyncClient | None = None


def proxy_env(port: int, tape_path: Path) -> dict[str, str]:
    base = f"http://127.0.0.1:{port}"
    return {
        "OPENAI_BASE_URL": f"{base}/openai/v1",
        "ANTHROPIC_BASE_URL": f"{base}/anthropic",
        "AGENT_RR_TAPE_PATH": str(tape_path),
    }


def build_app(state: ProxyState) -> Starlette:
    @asynccontextmanager
    async def lifespan(_app: Starlette) -> AsyncIterator[None]:
        async with httpx.AsyncClient(timeout=httpx.Timeout(None, connect=30.0)) as client:
            state.client = client
            yield

    async def openai(request: Request) -> Response:
        return await _handle(state, request, "openai", state.openai_upstream, "/v1/")

    async def anthropic(request: Request) -> Response:
        return await _handle(state, request, "anthropic", state.anthropic_upstream, "/")

    methods = ["GET", "POST", "PUT", "PATCH", "DELETE"]
    return Starlette(
        routes=[
            Route("/openai/v1/{path:path}", openai, methods=methods),
            Route("/anthropic/{path:path}", anthropic, methods=methods),
        ],
        lifespan=lifespan,
    )


async def _handle(
    state: ProxyState, request: Request, provider: str, upstream: str, endpoint_prefix: str
) -> Response:
    raw = await request.body()
    try:
        body = _JSON_OBJECT.validate_json(raw or b"{}")
    except ValidationError:
        return JSONResponse({"error": "agent-rr: request body must be a JSON object"}, 400)
    idx = state.llm_idx
    state.llm_idx += 1
    path = request.path_params["path"]
    endpoint = endpoint_prefix + path

    mock = state.router.mock_for(idx)
    if mock is not None:
        return _serve_mock(idx, body, mock)
    if state.chaos is not None and (rule := state.chaos.llm_fault(idx)) is not None:
        await _record_rate_limit(state, state.chaos.event(rule, {"llm_idx": idx, "retry_after": 1}))
        return JSONResponse(
            {"error": {"type": "rate_limit_error", "message": "agent-rr chaos: rate-limit-429"}},
            429,
            headers={"retry-after": "1"},
        )

    live = state.router.live_body(provider, body)
    content = raw if live == body else json.dumps(live).encode()
    url = f"{upstream.rstrip('/')}/{path}"
    if request.url.query:
        url += f"?{request.url.query}"
    headers = {k: v for k, v in request.headers.items() if k.lower() not in _HOP_HEADERS}
    headers["accept-encoding"] = "identity"

    pre_task = asyncio.create_task(asyncio.to_thread(capture_tree_sha, state.snapshot_root))
    ts_start, t0 = time.time(), time.perf_counter()

    async def finish(status: int, data: bytes, streamed: bool) -> None:
        """Persist the call; recording failures warn but never break the agent's request."""
        try:
            sha = await pre_task
            await asyncio.to_thread(
                state.writer.append_step,
                _llm_step(provider, endpoint, live, status, data, streamed, sha, ts_start, t0),
            )
        except Exception as exc:
            print(f"agent-rr: warning: failed to record step: {exc}", file=sys.stderr)

    assert state.client is not None
    upstream_req = state.client.build_request(request.method, url, headers=headers, content=content)
    try:
        resp = await state.client.send(upstream_req, stream=True)
    except httpx.HTTPError as exc:
        message = f"agent-rr: upstream request failed: {type(exc).__name__}: {exc}"
        await finish(502, json.dumps({"error": message}).encode(), streamed=False)
        return JSONResponse({"error": message}, 502)

    out_headers = {k: v for k, v in resp.headers.items() if k.lower() not in _DROP_RESPONSE_HEADERS}
    if not resp.headers.get("content-type", "").startswith("text/event-stream"):
        data = await resp.aread()
        await resp.aclose()
        await finish(resp.status_code, data, streamed=False)
        return Response(data, resp.status_code, headers=out_headers)

    async def relay() -> AsyncIterator[bytes]:
        chunks: list[bytes] = []
        try:
            async for chunk in resp.aiter_raw():
                chunks.append(chunk)
                yield chunk  # forwarded immediately; Starlette awaits send() => backpressure
        finally:
            await resp.aclose()
        # Recorded before the response terminates so step order matches the agent's order.
        await finish(resp.status_code, b"".join(chunks), streamed=True)

    return StreamingResponse(relay(), resp.status_code, headers=out_headers)


async def _record_rate_limit(state: ProxyState, event: ChaosEvent) -> None:
    try:
        sha = await asyncio.to_thread(capture_tree_sha, state.snapshot_root)
        now = time.time()
        await asyncio.to_thread(
            state.writer.append_step,
            StepData(
                ts_start=now,
                ts_end=now,
                kind="chaos_fault",
                status="injected_fault",
                pre_tree_sha=sha,
                post_tree_sha=sha,
                chaos=event,
            ),
        )
    except Exception as exc:
        print(f"agent-rr: warning: failed to record chaos fault: {exc}", file=sys.stderr)


def _serve_mock(idx: int, body: dict[str, Any], mock: LlmCall) -> Response:
    if request_digest(redact_payload(body)) != request_digest(json.loads(mock.request_json)):
        print(
            f"agent-rr: warning: llm request {idx} diverged from the parent tape "
            "(request hash mismatch); serving the recorded response",
            file=sys.stderr,
        )
    if body.get("stream") and mock.raw_sse_chunks is not None:
        chunks = mock.raw_sse_chunks

        async def replay() -> AsyncIterator[bytes]:
            for chunk in chunks:
                yield chunk.encode()

        return StreamingResponse(replay(), 200, media_type="text/event-stream")
    return Response(mock.response_json, 200, media_type="application/json")


def _llm_step(
    provider: str,
    endpoint: str,
    request_body: dict[str, Any],
    status: int,
    data: bytes,
    streamed: bool,
    tree_sha: str,
    ts_start: float,
    t0: float,
) -> StepData:
    text = data.decode("utf-8", errors="replace")
    events = split_events(text) if streamed else None
    try:
        response: Any = (
            reassemble(event_payloads(events)) if events is not None else json.loads(text)
        )
    except json.JSONDecodeError:
        response = {"raw": text}
    if not isinstance(response, dict):
        response = {"raw": response}
    response = redact_payload(response)
    inp, out = token_usage(response)
    return StepData(
        ts_start=ts_start,
        ts_end=time.time(),
        kind="llm_call",
        status="ok" if status < 400 else "error",
        pre_tree_sha=tree_sha,  # LLM calls never touch the workspace: post == pre
        post_tree_sha=tree_sha,
        llm=LlmCall(
            provider="anthropic" if provider == "anthropic" else "openai",
            endpoint=endpoint,
            model=str(request_body.get("model") or response.get("model") or "unknown"),
            request_json=canonical_json(redact_payload(request_body)),
            response_json=canonical_json(response),
            raw_sse_chunks=[redact_text(e) for e in events] if events is not None else None,
            http_status=status,
            input_tokens=inp,
            output_tokens=out,
            latency_ms=(time.perf_counter() - t0) * 1000,
        ),
    )


@contextmanager
def serve(app: Starlette) -> Iterator[int]:
    """Run `app` on uvicorn at 127.0.0.1:<ephemeral> in a background thread; yield the port."""
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.bind(("127.0.0.1", 0))
    port: int = sock.getsockname()[1]
    server = uvicorn.Server(
        uvicorn.Config(
            app, log_level="warning", access_log=False, lifespan="on", timeout_graceful_shutdown=5
        )
    )
    thread = threading.Thread(target=server.run, kwargs={"sockets": [sock]}, daemon=True)
    thread.start()
    try:
        while not server.started:
            if not thread.is_alive():
                raise AgentRRError("LLM proxy failed to start", exit_code=1)
            time.sleep(0.005)
        yield port
    finally:
        server.should_exit = True
        thread.join(15)
        sock.close()
