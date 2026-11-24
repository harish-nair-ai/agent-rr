"""Typer CLI: record, inspect, fork, diff, freeze, bisect (spec §6)."""

from __future__ import annotations

import functools
import sys
import time
import uuid
from collections.abc import Callable
from pathlib import Path
from typing import Annotated, Any, TypeVar, cast

import typer

from agent_rr import __version__
from agent_rr.chaos.injector import load_chaos
from agent_rr.errors import AgentRRError
from agent_rr.fork import fork
from agent_rr.git_engine.snapshot import head_sha, repo_toplevel
from agent_rr.git_engine.worktree import signal_cleanup
from agent_rr.models import TapeMeta
from agent_rr.proxy.replay_router import ReplayRouter
from agent_rr.runner import run_recorded
from agent_rr.tape.store import TapeWriter

app = typer.Typer(
    pretty_exceptions_enable=False,
    no_args_is_help=True,
    add_completion=False,
    help="Flight recorder, time-travel debugger, and git bisect engine for coding agents.",
)

F = TypeVar("F", bound=Callable[..., Any])
Cmd = Annotated[list[str], typer.Argument(help="Agent command, after `--`.")]


def _guard(fn: F) -> F:
    """Users never see tracebacks: known errors -> documented code, others -> exit 1."""

    @functools.wraps(fn)
    def inner(*args: Any, **kwargs: Any) -> Any:
        try:
            with signal_cleanup():
                return fn(*args, **kwargs)
        except AgentRRError as exc:
            typer.echo(f"error: {exc}", err=True)
            raise typer.Exit(exc.exit_code) from None
        except KeyboardInterrupt:
            raise typer.Exit(130) from None
        except (typer.Exit, typer.Abort):
            raise
        except Exception as exc:
            typer.echo(f"error: internal: {type(exc).__name__}: {exc}", err=True)
            raise typer.Exit(1) from None

    return cast(F, inner)


def _version(value: bool) -> None:
    if value:
        typer.echo(f"agent-rr {__version__}")
        raise typer.Exit()


@app.callback()
def main(
    version: Annotated[
        bool,
        typer.Option("--version", callback=_version, is_eager=True, help="Show version and exit."),
    ] = False,
) -> None:
    """agent-rr command line."""


def _new_tape_path(out: Path | None, repo: Path, tape_id: str) -> Path:
    path = out if out is not None else repo / ".agent-rr" / "tapes" / f"{tape_id}.tape"
    if path.exists():
        raise AgentRRError(f"tape already exists (tapes are append-only): {path}")
    return path


@app.command()
@_guard
def record(
    cmd: Cmd,
    out: Annotated[Path | None, typer.Option("--out", help="Tape path to create.")] = None,
    chaos: Annotated[
        list[str] | None,
        typer.Option("--chaos", help="Chaos preset name or chaos.toml path (repeatable)."),
    ] = None,
    seed: Annotated[int | None, typer.Option("--seed", help="Chaos RNG seed.")] = None,
) -> None:
    """Run CMD behind the recording proxy and write a redacted .tape."""
    cwd = Path.cwd()
    repo = repo_toplevel(cwd)
    base = head_sha(repo)
    config = load_chaos(chaos, seed) if chaos else None
    tape_id = uuid.uuid4().hex
    path = _new_tape_path(out, repo, tape_id)
    with TapeWriter(path) as writer:
        writer.ensure_meta(
            TapeMeta(
                tape_id=tape_id,
                created_at=time.time(),
                command_argv=cmd,
                repo_root=str(repo),
                base_commit_sha=base,
                chaos_seed=config.seed if config else None,
                metadata={"producer": "agent-rr"},
            )
        )
        code = run_recorded(cmd, cwd, writer, repo, ReplayRouter([], 0), config)
    print(f"agent-rr: tape written to {path}", file=sys.stderr)
    raise typer.Exit(code)


@app.command("fork")
@_guard
def fork_cmd(
    tape: Annotated[Path, typer.Argument(help="Parent tape.")],
    cmd: Cmd,
    at_step: Annotated[int, typer.Option("--at-step", help="Step to fork at.")],
    model: Annotated[str | None, typer.Option("--model", help="Override the model.")] = None,
    system_prompt_file: Annotated[
        Path | None, typer.Option("--system-prompt-file", help="Replacement system prompt.")
    ] = None,
    out: Annotated[Path | None, typer.Option("--out", help="Child tape path.")] = None,
) -> None:
    """Restore step N in an isolated worktree, mock steps < N, run CMD live from N."""
    code, child = fork(tape, at_step, cmd, out, model, system_prompt_file)
    print(f"agent-rr: child tape written to {child}", file=sys.stderr)
    raise typer.Exit(code)
