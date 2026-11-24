"""Fork a tape at step N into an ephemeral worktree and a new child tape (contracts §6.3)."""

import time
import uuid
from pathlib import Path

from agent_rr.errors import AgentRRError
from agent_rr.git_engine.snapshot import repo_toplevel
from agent_rr.git_engine.worktree import WorktreeContext
from agent_rr.models import Step, TapeMeta
from agent_rr.proxy.replay_router import ReplayRouter
from agent_rr.runner import run_recorded
from agent_rr.tape.store import TapeWriter, read_steps


def validate_step(steps: list[Step], at_step: int) -> Step:
    if not 0 <= at_step < len(steps):
        valid = f"0..{len(steps) - 1}" if steps else "none"
        raise AgentRRError(f"invalid step {at_step}: tape has {len(steps)} steps (valid: {valid})")
    return steps[at_step]


def start_child_tape(
    writer: TapeWriter, parent: TapeMeta, steps: list[Step], at_step: int, cmd: list[str]
) -> None:
    """Child meta links to the parent; steps < N are copied so indices stay aligned."""
    writer.ensure_meta(
        TapeMeta(
            tape_id=uuid.uuid4().hex,
            created_at=time.time(),
            command_argv=cmd,
            repo_root=parent.repo_root,
            base_commit_sha=parent.base_commit_sha,
            parent_tape_id=parent.tape_id,
            fork_step_idx=at_step,
            metadata={"producer": "agent-rr"},
        )
    )
    for step in steps[:at_step]:
        writer.append_step(step)


def fork_router(
    steps: list[Step], at_step: int, model: str | None, system_prompt: str | None
) -> ReplayRouter:
    prefix = [s.llm for s in steps[:at_step] if s.llm is not None]
    return ReplayRouter(prefix, len(prefix), model, system_prompt)


def fork(
    parent_path: Path,
    at_step: int,
    cmd: list[str],
    out: Path | None,
    model: str | None,
    system_prompt_file: Path | None,
) -> tuple[int, Path]:
    meta, steps = read_steps(parent_path)
    target = validate_step(steps, at_step)
    system_prompt = None
    if system_prompt_file is not None:
        if not system_prompt_file.is_file():
            raise AgentRRError(f"system prompt file not found: {system_prompt_file}")
        system_prompt = system_prompt_file.read_text()
    child = out or parent_path.with_name(
        f"{parent_path.stem}.fork{at_step}.{uuid.uuid4().hex[:8]}.tape"
    )
    if child.exists():
        raise AgentRRError(f"tape already exists (tapes are append-only): {child}")
    if not Path(meta.repo_root).is_dir():
        raise AgentRRError(f"recorded repo_root is not available: {meta.repo_root}")
    repo = repo_toplevel(Path(meta.repo_root))
    with (
        WorktreeContext(repo, meta.base_commit_sha, target.pre_tree_sha) as wt,
        TapeWriter(child) as writer,
    ):
        start_child_tape(writer, meta, steps, at_step, cmd)
        router = fork_router(steps, at_step, model, system_prompt)
        code = run_recorded(cmd, wt, writer, wt, router, chaos=None)
    return code, child
