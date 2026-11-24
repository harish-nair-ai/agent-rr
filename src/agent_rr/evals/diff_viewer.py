"""Rich trajectory timeline, step detail, and side-by-side tape diff (spec §6).

Consoles passed in must use `markup=False`: every rendered string may contain user data.
"""

import difflib
import json
from pathlib import Path
from typing import Any

from rich.console import Console, Group
from rich.panel import Panel
from rich.syntax import Syntax
from rich.table import Table
from rich.text import Text

from agent_rr.errors import AgentRRError
from agent_rr.git_engine.snapshot import diff_trees
from agent_rr.models import Step, TapeMeta
from agent_rr.proxy.replay_router import canonical_json
from agent_rr.testing import changed_lines, tool_sequence


def llm_output(response_json: str) -> Any:
    """The model's actual output, ignoring volatile ids/timestamps/usage."""
    try:
        resp = json.loads(response_json)
    except json.JSONDecodeError:
        return response_json
    if not isinstance(resp, dict):
        return resp
    if isinstance(resp.get("choices"), list):
        return [c.get("message") or c.get("text") for c in resp["choices"] if isinstance(c, dict)]
    if "content" in resp:
        return resp["content"]
    if "output" in resp:
        return resp["output"]
    return resp


def signature(step: Step) -> str:
    parts: list[Any] = [step.kind, step.status, step.post_tree_sha]
    if step.llm:
        parts += [step.llm.model, step.llm.http_status, llm_output(step.llm.response_json)]
    if step.tool:
        t = step.tool
        parts += [t.tool_name, t.arguments_json, t.result_text, t.is_error]
    if step.chaos:
        parts += [step.chaos.preset, step.chaos.detail_json]
    return canonical_json(parts)


def first_divergence(a: list[Step], b: list[Step]) -> int | None:
    for i, (x, y) in enumerate(zip(a, b, strict=False)):
        if signature(x) != signature(y):
            return i
    return None if len(a) == len(b) else min(len(a), len(b))


def _label(step: Step) -> str:
    if step.llm:
        return step.llm.model
    if step.tool:
        return step.tool.tool_name
    return step.chaos.preset if step.chaos else ""


def _diff_stat(step: Step) -> str:
    lines = changed_lines(step.diff_text)
    plus = sum(1 for _, sign, _ in lines if sign == "+")
    return f"+{plus} -{len(lines) - plus}" if lines else ""


def _tokens(steps: list[Step]) -> tuple[int, int]:
    return (
        sum(s.llm.input_tokens for s in steps if s.llm),
        sum(s.llm.output_tokens for s in steps if s.llm),
    )


def render_timeline(console: Console, meta: TapeMeta, steps: list[Step]) -> None:
    head = (
        f"tape {meta.tape_id}  base {meta.base_commit_sha[:12]}  cmd {' '.join(meta.command_argv)}"
    )
    if meta.parent_tape_id:
        head += f"\nforked from {meta.parent_tape_id} at step {meta.fork_step_idx}"
    if meta.chaos_seed is not None:
        head += f"\nchaos seed {meta.chaos_seed}"
    table = Table(title=head, title_justify="left")
    for col in ("step", "kind", "status", "model/tool", "tokens in/out", "diff", "latency ms"):
        table.add_column(col, no_wrap=True)
    for s in steps:
        latency = s.llm.latency_ms if s.llm else (s.ts_end - s.ts_start) * 1000
        table.add_row(
            str(s.step_idx),
            s.kind,
            s.status,
            _label(s),
            f"{s.llm.input_tokens}/{s.llm.output_tokens}" if s.llm else "",
            _diff_stat(s),
            f"{latency:.0f}",
        )
    console.print(table)


def _pretty(text: str) -> str:
    try:
        return json.dumps(json.loads(text), indent=2, sort_keys=True)
    except json.JSONDecodeError:
        return text


def render_step(console: Console, steps: list[Step], idx: int) -> None:
    if not 0 <= idx < len(steps):
        raise AgentRRError(f"invalid step {idx}: tape has {len(steps)} steps")
    s = steps[idx]
    parts: list[Any] = [
        Text(
            f"step {s.step_idx}  {s.kind}  status={s.status}\n"
            f"pre {s.pre_tree_sha}\npost {s.post_tree_sha}"
        )
    ]
    if s.llm:
        c = s.llm
        parts.append(
            Text(
                f"\n{c.provider} {c.endpoint}  model={c.model}  http={c.http_status}  "
                f"tokens={c.input_tokens}/{c.output_tokens}  latency={c.latency_ms:.0f}ms"
                + (f"  sse_chunks={len(c.raw_sse_chunks)}" if c.raw_sse_chunks is not None else "")
            )
        )
        parts += [Text("\nrequest:"), Syntax(_pretty(c.request_json), "json", word_wrap=True)]
        parts += [Text("response:"), Syntax(_pretty(c.response_json), "json", word_wrap=True)]
    if s.tool:
        t = s.tool
        parts.append(Text(f"\ntool {t.tool_name}  error={t.is_error}  exit_code={t.exit_code}"))
        parts += [Text("arguments:"), Syntax(_pretty(t.arguments_json), "json", word_wrap=True)]
        parts += [Text("result:"), Text(t.result_text)]
    if s.chaos:
        e = s.chaos
        parts.append(Text(f"\nchaos {e.preset}  target={e.target}  seed {e.seed}"))
        parts.append(Syntax(_pretty(e.detail_json), "json", word_wrap=True))
    if s.diff_text:
        parts += [Text("\ndiff:"), Syntax(s.diff_text, "diff", word_wrap=True)]
    console.print(Panel(Group(*parts), title=f"step {idx}", title_align="left"))


def render_diff(
    console: Console, a: tuple[TapeMeta, list[Step]], b: tuple[TapeMeta, list[Step]]
) -> bool:
    """Print the comparison; return True iff the trajectories and final trees are identical."""
    (meta_a, steps_a), (_, steps_b) = a, b
    div = first_divergence(steps_a, steps_b)
    final_a = steps_a[-1].post_tree_sha if steps_a else None
    final_b = steps_b[-1].post_tree_sha if steps_b else None
    if div is None and final_a == final_b:
        console.print(f"identical: {len(steps_a)} steps, same final tree")
        return True

    console.print(
        Text(f"diverged at step {div}" if div is not None else "final trees differ", style="bold")
    )
    side = Table(title="trajectory (A | B)", title_justify="left")
    for col in ("step", "A", "B"):
        side.add_column(col, no_wrap=True)
    for i in range(max(len(steps_a), len(steps_b))):
        cells = [
            f"{s[i].kind} {_label(s[i])} {s[i].status}" if i < len(s) else "-"
            for s in (steps_a, steps_b)
        ]
        style = "red" if div is not None and i >= div else None
        side.add_row(str(i), *cells, style=style)
    console.print(side)

    delta = difflib.unified_diff(
        tool_sequence(steps_a), tool_sequence(steps_b), "A tools", "B tools", lineterm=""
    )
    console.print(Text("tool sequence delta:\n" + ("\n".join(delta) or "(same tools)")))
    (in_a, out_a), (in_b, out_b) = _tokens(steps_a), _tokens(steps_b)
    console.print(f"tokens in: {in_a} -> {in_b} ({in_b - in_a:+d})")
    console.print(f"tokens out: {out_a} -> {out_b} ({out_b - out_a:+d})")

    if final_a and final_b and final_a != final_b:
        if not Path(meta_a.repo_root).is_dir():
            console.print(f"final tree diff unavailable: repo {meta_a.repo_root} not found")
            return False
        try:
            tree_diff = diff_trees(Path(meta_a.repo_root), final_a, final_b)
        except AgentRRError as exc:
            console.print(f"final tree diff unavailable: {exc}")
        else:
            console.print(Text("final tree diff (A -> B):"))
            console.print(Syntax(tree_diff, "diff", word_wrap=True))
    return False
