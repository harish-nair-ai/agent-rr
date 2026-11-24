import json
import sqlite3
import sys
import textwrap
import time
from pathlib import Path

import pytest
from agent_rr.chaos.injector import (
    TRUNCATION_MARKER,
    ChaosInjector,
    InjectedToolError,
    load_chaos,
    truncate_stdout,
)
from conftest import run_cli
from upstream import Upstream

from agent_rr.errors import AgentRRError
from agent_rr.models import ChaosConfig, ChaosRule
from agent_rr.proxy.decorator import record
from agent_rr.tape.store import TapeReader


def _cfg(*rules: dict[str, object], seed: int = 42) -> ChaosConfig:
    return ChaosConfig(seed=seed, faults=[ChaosRule.model_validate(r) for r in rules])


def _llm_seq(inj: ChaosInjector, n: int = 60) -> list[bool]:
    return [inj.llm_fault(i) is not None for i in range(n)]


def test_same_seed_same_decisions_different_seed_differs() -> None:
    cfg = _cfg({"preset": "rate-limit-429", "probability": 0.3})
    a, b = _llm_seq(ChaosInjector(cfg)), _llm_seq(ChaosInjector(cfg))
    assert a == b and any(a) and not all(a)
    other = _llm_seq(ChaosInjector(cfg.model_copy(update={"seed": 7})))
    assert other != a


def test_isolated_rng_ignores_global_random_state() -> None:
    import random

    cfg = _cfg({"preset": "rate-limit-429", "probability": 0.5})
    random.seed(1)
    a = _llm_seq(ChaosInjector(cfg))
    random.seed(999)
    random.random()
    assert _llm_seq(ChaosInjector(cfg)) == a


def test_at_steps_are_explicit_and_draw_nothing() -> None:
    inj = ChaosInjector(_cfg({"preset": "rate-limit-429", "at_steps": [1, 3]}))
    assert [i for i in range(6) if inj.llm_fault(i)] == [1, 3]


def test_tool_glob_filters_eligibility_without_consuming_draws() -> None:
    rule = {"preset": "truncate-stdout", "probability": 0.5, "tool_glob": "run_*"}
    mixed = ChaosInjector(_cfg(rule))
    only = ChaosInjector(_cfg(rule))
    got = []
    for i in range(40):
        if mixed.tool_fault(i, "read_file", {}) is not None:
            pytest.fail("non-matching tool must never fault")
        got.append(mixed.tool_fault(i, "run_tests", {}) is not None)
    assert got == [only.tool_fault(i, "run_tests", {}) is not None for i in range(40)]


def test_flaky_test_fires_once_then_passes_on_retry() -> None:
    inj = ChaosInjector(_cfg({"preset": "flaky-test", "probability": 1.0}))
    assert [inj.tool_fault(i, "pytest", {}) is not None for i in range(4)] == [
        True,
        False,
        False,
        False,
    ]


def test_permission_denied_requires_matching_path_argument() -> None:
    inj = ChaosInjector(
        _cfg({"preset": "permission-denied", "probability": 1.0, "path_glob": "secrets/*"})
    )
    assert inj.tool_fault(0, "write", {"path": "src/a.py"}) is None
    assert inj.tool_fault(1, "write", {"path": "secrets/key.pem"}) is not None


def test_llm_and_tool_presets_route_to_their_target() -> None:
    inj = ChaosInjector(
        _cfg(
            {"preset": "rate-limit-429", "at_steps": [0]},
            {"preset": "tool-timeout", "at_steps": [0]},
        )
    )
    llm = inj.llm_fault(0)
    tool = inj.tool_fault(0, "x", {})
    assert llm is not None and llm.preset == "rate-limit-429"
    assert tool is not None and tool.preset == "tool-timeout"


def test_truncate_stdout() -> None:
    text = "é" * 400  # multi-byte: cut on a byte budget, never split a character
    out = truncate_stdout(text)
    assert out.endswith(TRUNCATION_MARKER) and TRUNCATION_MARKER == "\n...[TRUNCATED BY CHAOS]"
    assert len(out.removesuffix(TRUNCATION_MARKER).encode()) == 300


def test_preset_flag_and_chaos_toml(tmp_path: Path) -> None:
    assert load_chaos(["rate-limit-429"], seed=5) == _cfg(
        {"preset": "rate-limit-429", "probability": 0.3}, seed=5
    )
    toml = tmp_path / "chaos.toml"
    toml.write_text(
        textwrap.dedent(
            """
            seed = 42
            [[faults]]
            preset = "rate-limit-429"
            target = "llm"
            at_steps = [1]
            [[faults]]
            preset = "permission-denied"
            path_glob = "*.env"
            probability = 0.5
            """
        )
    )
    cfg = load_chaos([str(toml)], seed=None)
    assert cfg.seed == 42
    assert cfg.faults[0].at_steps == [1] and cfg.faults[1].target == "tool"
    assert load_chaos([str(toml)], seed=9).seed == 9  # --seed overrides the file


@pytest.mark.parametrize(
    "body",
    [
        'seed = 1\n[[faults]]\npreset = "nope"\n',
        'seed = 1\n[[faults]]\npreset = "rate-limit-429"\ntarget = "tool"\n',
        "seed = 1\nsurprise = true\n",
        "seed = [",
    ],
)
def test_invalid_chaos_config_is_exit_2(tmp_path: Path, body: str) -> None:
    toml = tmp_path / "chaos.toml"
    toml.write_text(body)
    with pytest.raises(AgentRRError) as exc:
        load_chaos([str(toml)], seed=None)
    assert exc.value.exit_code == 2


def test_unknown_preset_name_is_exit_2() -> None:
    with pytest.raises(AgentRRError, match="unknown chaos preset"):
        load_chaos(["meteor-strike"], seed=None)


# --- tool presets through the real @record wrapper ---------------------------------------


@pytest.fixture
def chaos_tape(repo: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    tape = tmp_path / "chaos.tape"
    monkeypatch.setenv("AGENT_RR_TAPE_PATH", str(tape))
    monkeypatch.chdir(repo)
    return tape


def _use(monkeypatch: pytest.MonkeyPatch, *rules: dict[str, object]) -> None:
    monkeypatch.setenv("AGENT_RR_CHAOS", _cfg(*rules).model_dump_json())


def test_truncate_stdout_tool(chaos_tape: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _use(monkeypatch, {"preset": "truncate-stdout", "at_steps": [0]})

    @record
    def run_tests() -> str:
        return "x" * 1000

    out = run_tests()
    assert out == "x" * 300 + TRUNCATION_MARKER
    assert run_tests() == "x" * 1000
    with TapeReader(chaos_tape) as r:
        steps = r.steps()
    assert [(s.kind, s.status) for s in steps] == [
        ("chaos_fault", "injected_fault"),
        ("tool_call", "ok"),
    ]
    assert steps[0].tool is not None and steps[0].tool.result_text == out
    assert steps[0].chaos is not None
    assert (steps[0].chaos.preset, steps[0].chaos.seed, steps[0].chaos.target) == (
        "truncate-stdout",
        42,
        "tool",
    )
    assert json.loads(steps[0].chaos.detail_json) == {
        "original_bytes": 1000,
        "tool_idx": 0,
        "tool_name": "run_tests",
    }


def test_flaky_timeout_and_permission_tools(
    chaos_tape: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _use(
        monkeypatch,
        {"preset": "flaky-test", "at_steps": [0, 1], "tool_glob": "run_pytest"},
        {"preset": "tool-timeout", "at_steps": [2], "delay_s": 0.2},
        {"preset": "permission-denied", "at_steps": [3], "path_glob": "*.pem"},
    )
    calls: list[str] = []

    @record
    def run_pytest(args: str) -> str:
        calls.append("pytest")
        return "1 passed"

    @record
    def write(path: str) -> str:
        calls.append(path)
        return "ok"

    with pytest.raises(InjectedToolError, match="flaky-test"):
        run_pytest("-q")
    assert run_pytest("-q") == "1 passed"  # passes on retry (fires once)
    t0 = time.monotonic()
    with pytest.raises(TimeoutError):
        write("a.txt")
    assert time.monotonic() - t0 >= 0.2
    with pytest.raises(PermissionError, match="EACCES: Permission denied"):
        write("key.pem")
    assert calls == ["pytest"]  # faulted calls never executed the tool
    with TapeReader(chaos_tape) as r:
        steps = r.steps()
    assert [(s.kind, s.chaos.preset if s.chaos else None) for s in steps] == [
        ("chaos_fault", "flaky-test"),
        ("tool_call", None),
        ("chaos_fault", "tool-timeout"),
        ("chaos_fault", "permission-denied"),
    ]
    assert all(s.tool is not None and s.tool.is_error for s in steps if s.kind == "chaos_fault")
    assert steps[3].chaos is not None
    assert json.loads(steps[3].chaos.detail_json)["path"] == "key.pem"


# --- acceptance: full `agent-rr record --chaos` run, twice, byte-identical --------------

AGENT = textwrap.dedent(
    """
    import json, os, urllib.request, urllib.error
    from agent_rr import tape

    @tape.record
    def run_tool(i: int) -> str:
        return f"line {i}\\n" * 80

    base = os.environ["OPENAI_BASE_URL"]
    rate_limited = truncated = 0
    for i in range(8):
        req = urllib.request.Request(
            f"{base}/chat/completions",
            data=json.dumps({"model": "m", "messages": [{"role": "user", "content": i}]}).encode(),
            headers={"Content-Type": "application/json"},
        )
        while True:
            try:
                urllib.request.urlopen(req).read()
                break
            except urllib.error.HTTPError as e:
                assert e.code == 429 and e.headers["Retry-After"] == "1"
                rate_limited += 1
        if run_tool(i).endswith("[TRUNCATED BY CHAOS]"):
            truncated += 1
    print(json.dumps({"rate_limited": rate_limited, "truncated": truncated}))
    """
)


def _faults(tape: Path) -> tuple[list[tuple[object, ...]], list[tuple[object, ...]]]:
    conn = sqlite3.connect(tape)
    try:
        steps = conn.execute(
            "SELECT step_idx, kind, status FROM steps WHERE kind = 'chaos_fault' ORDER BY step_idx"
        ).fetchall()
        events = conn.execute(
            "SELECT step_idx, preset, seed, target, detail_json FROM chaos_events ORDER BY id"
        ).fetchall()
    finally:
        conn.close()
    return steps, events


def test_record_with_chaos_is_reproducible(repo: Path, tmp_path: Path, upstream: Upstream) -> None:
    agent = tmp_path / "agent.py"
    agent.write_text(AGENT)
    results = []
    for run in ("a", "b"):
        out = tmp_path / f"{run}.tape"
        proc = run_cli(
            "record",
            "--out",
            str(out),
            "--chaos",
            "rate-limit-429",
            "--chaos",
            "truncate-stdout",
            "--seed",
            "42",
            "--",
            sys.executable,
            str(agent),
            cwd=repo,
            env={"OPENAI_BASE_URL": f"{upstream.url}/v1"},
        )
        assert proc.returncode == 0, proc.stderr
        results.append((json.loads(proc.stdout), _faults(out)))
        with TapeReader(out) as r:
            assert r.meta.chaos_seed == 42
    (summary_a, faults_a), (summary_b, faults_b) = results
    assert faults_a == faults_b  # identical fault step indices and chaos_events rows
    assert summary_a == summary_b
    assert summary_a["rate_limited"] > 0 and summary_a["truncated"] > 0
    presets = {e[1] for e in faults_a[1]}
    assert presets == {"rate-limit-429", "truncate-stdout"}
