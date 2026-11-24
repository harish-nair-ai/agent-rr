"""`@tape.record`: records tool calls of custom Python agents into $AGENT_RR_TAPE_PATH.

In a forked run (tape_meta.fork_step_idx set) the first tool calls are replayed from the
tape's copied pre-fork steps instead of executing, mirroring the LLM replay router.
"""

from __future__ import annotations

import asyncio
import functools
import inspect
import json
import os
import sys
import threading
import time
import uuid
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, ParamSpec, TypeVar, cast, overload

from agent_rr.errors import AgentRRError
from agent_rr.git_engine.snapshot import capture_tree_sha, diff_trees, head_sha, repo_toplevel
from agent_rr.models import StepData, TapeMeta, ToolCall
from agent_rr.tape.redactor import redact_payload, redact_text
from agent_rr.tape.store import TapeReader, TapeWriter

P = ParamSpec("P")
R = TypeVar("R")


@dataclass
class _Call:
    tool_name: str
    arguments_json: str
    ts_start: float
    pre_sha: str
    mock: ToolCall | None = None


class _Session:
    """Per-process recording state for one tape path."""

    def __init__(self, tape_path: Path, repo_root: Path) -> None:
        self.tape_path = tape_path
        self.repo_root = repo_root
        self.lock = threading.Lock()
        self.tool_idx = 0
        with TapeWriter(tape_path) as writer:
            writer.ensure_meta(
                TapeMeta(
                    tape_id=uuid.uuid4().hex,
                    created_at=time.time(),
                    command_argv=list(sys.argv),
                    repo_root=str(repo_root),
                    base_commit_sha=head_sha(repo_root),
                    metadata={"producer": "agent-rr"},
                )
            )
        with TapeReader(tape_path) as reader:
            fork = reader.meta.fork_step_idx
            self.mocks = [
                s.tool for s in reader.steps() if fork is not None and s.step_idx < fork and s.tool
            ]

    def start(self, tool_name: str, arguments: dict[str, Any]) -> _Call:
        with self.lock:
            idx = self.tool_idx
            self.tool_idx += 1
        args_json = json.dumps(redact_payload(arguments), sort_keys=True, default=repr)
        if idx < len(self.mocks):
            return _Call(tool_name, args_json, time.time(), "", mock=self.mocks[idx])
        return _Call(tool_name, args_json, time.time(), capture_tree_sha(self.repo_root))

    def finish(self, call: _Call, result_text: str, is_error: bool) -> None:
        post = capture_tree_sha(self.repo_root)
        step = StepData(
            ts_start=call.ts_start,
            ts_end=time.time(),
            kind="tool_call",
            status="error" if is_error else "ok",
            pre_tree_sha=call.pre_sha,
            post_tree_sha=post,
            diff_text=redact_text(diff_trees(self.repo_root, call.pre_sha, post)),
            tool=ToolCall(
                tool_name=call.tool_name,
                arguments_json=call.arguments_json,
                result_text=redact_text(result_text),
                is_error=is_error,
            ),
        )
        with TapeWriter(self.tape_path) as writer:
            writer.append_step(step)


_sessions: dict[str, _Session | None] = {}
_sessions_lock = threading.Lock()


def _session() -> _Session | None:
    tape = os.environ.get("AGENT_RR_TAPE_PATH")
    if not tape:
        return None
    with _sessions_lock:
        if tape not in _sessions:
            try:
                _sessions[tape] = _Session(Path(tape), repo_toplevel(Path.cwd()))
            except AgentRRError as exc:
                print(f"agent-rr: warning: tool recording disabled: {exc}", file=sys.stderr)
                _sessions[tape] = None
        return _sessions[tape]


def _arguments(
    sig: inspect.Signature, args: tuple[Any, ...], kwargs: dict[str, Any]
) -> dict[str, Any]:
    try:
        return dict(sig.bind(*args, **kwargs).arguments)
    except TypeError:
        return {"args": list(args), "kwargs": kwargs}


def _text(result: object) -> str:
    return result if isinstance(result, str) else json.dumps(result, sort_keys=True, default=repr)


def _replayed(mock: ToolCall) -> Any:
    if mock.is_error:
        raise RuntimeError(mock.result_text)
    return mock.result_text


def _wrap(fn: Callable[P, R], tool_name: str) -> Callable[P, R]:
    sig = inspect.signature(fn)

    if inspect.iscoroutinefunction(fn):
        afn: Callable[..., Awaitable[Any]] = fn

        @functools.wraps(fn)
        async def awrapper(*args: Any, **kwargs: Any) -> Any:
            session = _session()
            if session is None:
                return await afn(*args, **kwargs)
            call = await asyncio.to_thread(session.start, tool_name, _arguments(sig, args, kwargs))
            if call.mock is not None:
                return _replayed(call.mock)
            try:
                result = await afn(*args, **kwargs)
            except Exception as exc:
                await asyncio.to_thread(session.finish, call, f"{type(exc).__name__}: {exc}", True)
                raise
            await asyncio.to_thread(session.finish, call, _text(result), False)
            return result

        return cast(Callable[P, R], awrapper)

    @functools.wraps(fn)
    def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
        session = _session()
        if session is None:
            return fn(*args, **kwargs)
        call = session.start(tool_name, _arguments(sig, args, kwargs))
        if call.mock is not None:
            return cast(R, _replayed(call.mock))
        try:
            result = fn(*args, **kwargs)
        except Exception as exc:
            session.finish(call, f"{type(exc).__name__}: {exc}", True)
            raise
        session.finish(call, _text(result), False)
        return result

    return wrapper


@overload
def record(func: Callable[P, R], *, name: str | None = None) -> Callable[P, R]: ...
@overload
def record(
    func: None = None, *, name: str | None = None
) -> Callable[[Callable[P, R]], Callable[P, R]]: ...
def record(
    func: Callable[P, R] | None = None, *, name: str | None = None
) -> Callable[P, R] | Callable[[Callable[P, R]], Callable[P, R]]:
    """Decorate a sync or async tool function so each call becomes a `tool_call` step."""
    if func is not None:
        return _wrap(func, name or func.__name__)
    return lambda fn: _wrap(fn, name or fn.__name__)
