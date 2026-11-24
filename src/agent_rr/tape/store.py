"""TapeWriter / TapeReader over the portable SQLite WAL .tape file."""

from __future__ import annotations

import json
import sqlite3
import threading
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Self, TypeVar

from agent_rr.errors import AgentRRError
from agent_rr.models import ChaosEvent, LlmCall, Step, StepData, TapeMeta, ToolCall
from agent_rr.tape.schema import DDL, connect


class TapeWriter:
    """Append-only writer. Safe to share across threads; multi-process safe via D11."""

    def __init__(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self.path = path
        self.connection = connect(path)
        self._lock = threading.Lock()
        self.connection.executescript(DDL)

    def ensure_meta(self, meta: TapeMeta) -> None:
        """Insert the single tape_meta row unless another writer already did."""
        with self._lock, self._immediate() as conn:
            if conn.execute("SELECT 1 FROM tape_meta").fetchone():
                return
            conn.execute(
                "INSERT INTO tape_meta VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    meta.tape_id,
                    meta.schema_version,
                    meta.created_at,
                    json.dumps(meta.command_argv),
                    meta.repo_root,
                    meta.base_commit_sha,
                    meta.parent_tape_id,
                    meta.fork_step_idx,
                    meta.chaos_seed,
                    json.dumps(meta.metadata, sort_keys=True),
                ),
            )

    def append_step(self, step: StepData) -> int:
        """Allocate the next step_idx and insert the step with its child rows atomically."""
        with self._lock, self._immediate() as conn:
            idx: int = conn.execute("SELECT COALESCE(MAX(step_idx), -1) + 1 FROM steps").fetchone()[
                0
            ]
            conn.execute(
                "INSERT INTO steps VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    idx,
                    step.ts_start,
                    step.ts_end,
                    step.kind,
                    step.pre_tree_sha,
                    step.post_tree_sha,
                    step.diff_text,
                    step.status,
                ),
            )
            if step.llm:
                c = step.llm
                conn.execute(
                    "INSERT INTO llm_calls (step_idx, provider, endpoint, model, request_json,"
                    " response_json, raw_sse_chunks, http_status, input_tokens, output_tokens,"
                    " latency_ms) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    (
                        idx,
                        c.provider,
                        c.endpoint,
                        c.model,
                        c.request_json,
                        c.response_json,
                        None if c.raw_sse_chunks is None else json.dumps(c.raw_sse_chunks),
                        c.http_status,
                        c.input_tokens,
                        c.output_tokens,
                        c.latency_ms,
                    ),
                )
            if step.tool:
                t = step.tool
                conn.execute(
                    "INSERT INTO tool_calls (step_idx, tool_name, arguments_json, result_text,"
                    " is_error, exit_code) VALUES (?, ?, ?, ?, ?, ?)",
                    (
                        idx,
                        t.tool_name,
                        t.arguments_json,
                        t.result_text,
                        int(t.is_error),
                        t.exit_code,
                    ),
                )
            if step.chaos:
                e = step.chaos
                conn.execute(
                    "INSERT INTO chaos_events (step_idx, preset, seed, target, detail_json)"
                    " VALUES (?, ?, ?, ?, ?)",
                    (idx, e.preset, e.seed, e.target, e.detail_json),
                )
            return idx

    def close(self) -> None:
        with self._lock:
            self.connection.execute("PRAGMA wal_checkpoint(TRUNCATE)")
            self.connection.close()

    @contextmanager
    def _immediate(self) -> Iterator[sqlite3.Connection]:
        """`BEGIN IMMEDIATE` ... COMMIT, or ROLLBACK on any exception."""
        self.connection.execute("BEGIN IMMEDIATE")
        try:
            yield self.connection
        except BaseException:
            self.connection.execute("ROLLBACK")
            raise
        self.connection.execute("COMMIT")

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()


class TapeReader:
    """Read-only view of a tape. Raises AgentRRError (exit 2) for missing or invalid tapes."""

    def __init__(self, path: Path) -> None:
        if not path.is_file():
            raise AgentRRError(f"tape not found: {path}")
        self.path = path
        try:
            self.connection = connect(path)
        except sqlite3.DatabaseError as exc:
            raise AgentRRError(f"not a valid .tape file: {path} ({exc})") from exc
        try:
            self.meta = self._parse_meta()
        except BaseException:
            self.connection.close()
            raise

    def _parse_meta(self) -> TapeMeta:
        with _corruption_as_error(self.path):
            row = self.connection.execute(
                "SELECT tape_id, schema_version, created_at, command_argv, repo_root,"
                " base_commit_sha, parent_tape_id, fork_step_idx, chaos_seed, metadata_json"
                " FROM tape_meta"
            ).fetchone()
            if row is None:
                raise AgentRRError(f"tape has no tape_meta row: {self.path}")
            return TapeMeta(
                tape_id=row[0],
                schema_version=row[1],
                created_at=row[2],
                command_argv=json.loads(row[3]),
                repo_root=row[4],
                base_commit_sha=row[5],
                parent_tape_id=row[6],
                fork_step_idx=row[7],
                chaos_seed=row[8],
                metadata=json.loads(row[9]),
            )

    def steps(self) -> list[Step]:
        with _corruption_as_error(self.path):
            return self._parse_steps()

    def _parse_steps(self) -> list[Step]:
        conn = self.connection
        llm = _first_by_step(
            conn,
            "SELECT step_idx, provider, endpoint, model, request_json, response_json,"
            " raw_sse_chunks, http_status, input_tokens, output_tokens, latency_ms"
            " FROM llm_calls ORDER BY id",
            lambda r: LlmCall(
                provider=r[1],
                endpoint=r[2],
                model=r[3],
                request_json=r[4],
                response_json=r[5],
                raw_sse_chunks=None if r[6] is None else json.loads(r[6]),
                http_status=r[7],
                input_tokens=r[8] or 0,
                output_tokens=r[9] or 0,
                latency_ms=r[10],
            ),
        )
        tool = _first_by_step(
            conn,
            "SELECT step_idx, tool_name, arguments_json, result_text, is_error, exit_code"
            " FROM tool_calls ORDER BY id",
            lambda r: ToolCall(
                tool_name=r[1],
                arguments_json=r[2],
                result_text=r[3],
                is_error=bool(r[4]),
                exit_code=r[5],
            ),
        )
        chaos = _first_by_step(
            conn,
            "SELECT step_idx, preset, seed, target, detail_json FROM chaos_events ORDER BY id",
            lambda r: ChaosEvent(preset=r[1], seed=r[2], target=r[3], detail_json=r[4]),
        )
        return [
            Step(
                step_idx=r[0],
                ts_start=r[1],
                ts_end=r[2],
                kind=r[3],
                pre_tree_sha=r[4],
                post_tree_sha=r[5],
                diff_text=r[6],
                status=r[7],
                llm=llm.get(r[0]),
                tool=tool.get(r[0]),
                chaos=chaos.get(r[0]),
            )
            for r in conn.execute(
                "SELECT step_idx, ts_start, ts_end, kind, pre_tree_sha, post_tree_sha, diff_text,"
                " status FROM steps ORDER BY step_idx"
            )
        ]

    def close(self) -> None:
        self.connection.close()

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()


T = TypeVar("T")


@contextmanager
def _corruption_as_error(path: Path) -> Iterator[None]:
    """Malformed rows/schema surface as an actionable exit-2 error, never a traceback."""
    try:
        yield
    except (sqlite3.DatabaseError, ValueError, TypeError, IndexError) as exc:
        raise AgentRRError(f"corrupt or incompatible .tape file {path}: {exc}") from exc


def _first_by_step(conn: sqlite3.Connection, sql: str, build: Callable[[Any], T]) -> dict[int, T]:
    out: dict[int, T] = {}
    for row in conn.execute(sql):
        out.setdefault(row[0], build(row))
    return out


def read_steps(path: Path) -> tuple[TapeMeta, list[Step]]:
    with TapeReader(path) as reader:
        return reader.meta, reader.steps()
