"""Run an agent command behind the recording proxy, with zombie-free child cleanup."""

import os
import subprocess
from pathlib import Path

from agent_rr.chaos.injector import ChaosInjector
from agent_rr.errors import AgentRRError
from agent_rr.models import ChaosConfig
from agent_rr.proxy.llm_proxy import (
    DEFAULT_ANTHROPIC,
    DEFAULT_OPENAI,
    ProxyState,
    build_app,
    proxy_env,
    serve,
)
from agent_rr.proxy.replay_router import ReplayRouter
from agent_rr.tape.store import TapeWriter


def run_child(cmd: list[str], cwd: Path, env: dict[str, str]) -> int:
    """Run `cmd`; on any interruption terminate (then kill) it so no zombie survives."""
    try:
        proc = subprocess.Popen(cmd, cwd=cwd, env=env)
    except OSError as exc:
        raise AgentRRError(f"cannot run {cmd[0]!r}: {exc.strerror or exc}") from exc
    try:
        code = proc.wait()
    finally:
        if proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(5)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait()
    return 128 - code if code < 0 else code


def run_recorded(
    cmd: list[str],
    cwd: Path,
    writer: TapeWriter,
    snapshot_root: Path,
    router: ReplayRouter,
    chaos: ChaosConfig | None,
) -> int:
    state = ProxyState(
        writer=writer,
        snapshot_root=snapshot_root,
        openai_upstream=os.environ.get("OPENAI_BASE_URL") or DEFAULT_OPENAI,
        anthropic_upstream=os.environ.get("ANTHROPIC_BASE_URL") or DEFAULT_ANTHROPIC,
        router=router,
        chaos=ChaosInjector(chaos) if chaos else None,
    )
    with serve(build_app(state)) as port:
        env = {**os.environ, **proxy_env(port, writer.path.resolve())}
        if chaos:
            env["AGENT_RR_CHAOS"] = chaos.model_dump_json()
        else:
            env.pop("AGENT_RR_CHAOS", None)
        return run_child(cmd, cwd, env)
