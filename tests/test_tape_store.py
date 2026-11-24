import json
import multiprocessing
import sqlite3
from pathlib import Path

import pytest

from agent_rr.errors import AgentRRError
from agent_rr.models import ChaosEvent, LlmCall, StepData, TapeMeta, ToolCall
from agent_rr.tape.schema import SCHEMA_VERSION
from agent_rr.tape.store import TapeReader, TapeWriter

SHA_A = "a" * 40
SHA_B = "b" * 40


def _meta(**kw: object) -> TapeMeta:
    base: dict[str, object] = {
        "tape_id": "0" * 32,
        "created_at": 1.5,
        "command_argv": ["python", "agent.py"],
        "repo_root": "/repo",
        "base_commit_sha": "c" * 40,
        "metadata": {"producer": "agent-rr"},
    }
    base.update(kw)
    return TapeMeta.model_validate(base)


def _llm_step(n: int = 0) -> StepData:
    return StepData(
        ts_start=1.0,
        ts_end=2.0,
        kind="llm_call",
        pre_tree_sha=SHA_A,
        post_tree_sha=SHA_A,
        llm=LlmCall(
            provider="openai",
            endpoint="/v1/chat/completions",
            model="gpt-x",
            request_json=json.dumps({"n": n}),
            response_json="{}",
            raw_sse_chunks=["data: 1\n\n"],
            http_status=200,
            input_tokens=3,
            output_tokens=4,
            latency_ms=12.5,
        ),
    )


def _tool_step() -> StepData:
    return StepData(
        ts_start=2.0,
        ts_end=3.0,
        kind="tool_call",
        pre_tree_sha=SHA_A,
        post_tree_sha=SHA_B,
        diff_text="diff --git a/x b/x\n",
        tool=ToolCall(tool_name="edit", arguments_json='{"p":1}', result_text="ok", exit_code=0),
    )


def test_roundtrip_meta_steps_and_children(tmp_path: Path) -> None:
    path = tmp_path / "s.tape"
    with TapeWriter(path) as w:
        w.ensure_meta(_meta())
        assert w.append_step(_llm_step()) == 0
        assert w.append_step(_tool_step()) == 1
        fault = StepData(
            ts_start=3.0,
            ts_end=3.0,
            kind="chaos_fault",
            status="injected_fault",
            pre_tree_sha=SHA_B,
            post_tree_sha=SHA_B,
            chaos=ChaosEvent(preset="rate-limit-429", seed=42, target="llm", detail_json="{}"),
        )
        assert w.append_step(fault) == 2

    with TapeReader(path) as r:
        assert r.meta == _meta()
        assert r.meta.schema_version == SCHEMA_VERSION == "0.1"
        steps = r.steps()
    assert [s.step_idx for s in steps] == [0, 1, 2]
    assert steps[0].llm == _llm_step().llm
    assert steps[1].tool == _tool_step().tool
    assert steps[1].diff_text == "diff --git a/x b/x\n"
    assert steps[2].chaos is not None and steps[2].chaos.preset == "rate-limit-429"


def test_writer_close_leaves_portable_single_file(tmp_path: Path) -> None:
    path = tmp_path / "s.tape"
    with TapeWriter(path) as w:
        w.ensure_meta(_meta())
        w.append_step(_llm_step())
    assert path.exists()
    assert not Path(f"{path}-wal").exists()
    assert not Path(f"{path}-shm").exists()


def test_connection_pragmas_schema_and_indexes(tmp_path: Path) -> None:
    path = tmp_path / "s.tape"
    with TapeWriter(path) as w:
        w.ensure_meta(_meta())
        conn = w.connection
        assert conn.execute("PRAGMA journal_mode").fetchone()[0] == "wal"
        assert conn.execute("PRAGMA synchronous").fetchone()[0] == 1  # NORMAL
        assert conn.execute("PRAGMA foreign_keys").fetchone()[0] == 1
        assert conn.execute("PRAGMA busy_timeout").fetchone()[0] == 5000
    conn = sqlite3.connect(path)
    names = {row[0] for row in conn.execute("SELECT name FROM sqlite_master")}
    conn.close()
    assert {
        "tape_meta",
        "steps",
        "llm_calls",
        "tool_calls",
        "chaos_events",
        "idx_llm_calls_step",
        "idx_tool_calls_step",
        "idx_chaos_events_step",
    } <= names


def test_ensure_meta_is_first_writer_wins(tmp_path: Path) -> None:
    path = tmp_path / "s.tape"
    with TapeWriter(path) as w:
        w.ensure_meta(_meta(tape_id="1" * 32))
        w.ensure_meta(_meta(tape_id="2" * 32))
    with TapeReader(path) as r:
        assert r.meta.tape_id == "1" * 32
        assert r.connection.execute("SELECT COUNT(*) FROM tape_meta").fetchone()[0] == 1


def test_check_constraints_reject_unknown_kind(tmp_path: Path) -> None:
    path = tmp_path / "s.tape"
    with TapeWriter(path) as w:
        w.ensure_meta(_meta())
        with pytest.raises(sqlite3.IntegrityError):
            w.connection.execute(
                "INSERT INTO steps VALUES (9, 0, 0, 'bogus', ?, ?, '', 'ok')", (SHA_A, SHA_A)
            )


def _hammer(path: str, n: int) -> None:
    with TapeWriter(Path(path)) as w:
        for i in range(n):
            w.append_step(_llm_step(i))


def test_concurrent_writers_allocate_contiguous_step_idx(tmp_path: Path) -> None:
    path = tmp_path / "s.tape"
    with TapeWriter(path) as w:
        w.ensure_meta(_meta())
    ctx = multiprocessing.get_context("spawn")
    procs = [ctx.Process(target=_hammer, args=(str(path), 25)) for _ in range(4)]
    for p in procs:
        p.start()
    for p in procs:
        p.join(60)
        assert p.exitcode == 0
    with TapeReader(path) as r:
        steps = r.steps()
    assert [s.step_idx for s in steps] == list(range(100))
    assert all(s.llm is not None for s in steps)


def test_reader_missing_file_is_actionable_error(tmp_path: Path) -> None:
    missing = tmp_path / "nope.tape"
    with pytest.raises(AgentRRError) as exc:
        TapeReader(missing)
    assert exc.value.exit_code == 2
    assert "not found" in str(exc.value)
    assert not missing.exists()  # reader never creates files


def test_reader_rejects_non_tape_file(tmp_path: Path) -> None:
    junk = tmp_path / "junk.tape"
    junk.write_text("not sqlite at all, definitely not a database file" * 10)
    with pytest.raises(AgentRRError) as exc:
        TapeReader(junk)
    assert exc.value.exit_code == 2


def test_models_are_frozen_and_strict() -> None:
    meta = _meta()
    with pytest.raises(ValueError):
        TapeMeta.model_validate({**meta.model_dump(), "surprise": 1})
    with pytest.raises(ValueError):
        meta.tape_id = "x"  # type: ignore[misc]
