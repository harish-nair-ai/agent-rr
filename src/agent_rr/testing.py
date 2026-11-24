"""Behavioral assertion helpers used by `agent-rr freeze` generated tests (spec §5.4)."""

import fnmatch
import os
import re
from collections import Counter
from pathlib import Path

from agent_rr.models import Step
from agent_rr.tape.store import read_steps

TapePath = str | os.PathLike[str]

_ASSERTION = re.compile(r"^\s*(assert\b|self\.assert\w*\(|expect\()")
_TEST_FILE = re.compile(
    r"(^|/)(tests?/|test_[^/]*\.py$|[^/]*_test\.py$|[^/]*\.(test|spec)\.[jt]sx?$)"
)
_DIFF_HEADER = re.compile(r"^diff --git a/(.+) b/(.+)$")


def _steps(tape: TapePath) -> list[Step]:
    return read_steps(Path(tape))[1]


def applied_diffs(steps: list[Step]) -> list[str]:
    """Diffs that really changed the workspace (steered steps hold rejected proposals, D12)."""
    return [s.diff_text for s in steps if s.diff_text and s.status != "steered"]


def changed_lines(diff: str) -> list[tuple[str, str, str]]:
    """(path, '+' or '-', line) for every changed line of a git unified diff."""
    out: list[tuple[str, str, str]] = []
    path = ""
    for line in diff.splitlines():
        if header := _DIFF_HEADER.match(line):
            path = header.group(2)
        elif line.startswith(("+++", "---")):
            continue
        elif line[:1] in ("+", "-") and path:
            out.append((path, line[0], line[1:]))
    return out


def edited_paths(steps: list[Step]) -> set[str]:
    paths: set[str] = set()
    for diff in applied_diffs(steps):
        for line in diff.splitlines():
            if header := _DIFF_HEADER.match(line):
                paths |= {header.group(1), header.group(2)}
    return paths


def tool_sequence(steps: list[Step]) -> list[str]:
    return [s.tool.tool_name for s in steps if s.tool is not None]


def assert_max_steps(tape: TapePath, limit: int) -> None:
    count = len(_steps(tape))
    assert count <= limit, f"run took {count} steps > limit {limit}"


def assert_no_edits_matching(tape: TapePath, globs: list[str]) -> None:
    touched = sorted(
        p for p in edited_paths(_steps(tape)) if any(fnmatch.fnmatchcase(p, g) for g in globs)
    )
    assert not touched, f"protected paths were edited: {touched}"


def assert_tool_called_before(tape: TapePath, first_tool: str, second_tool: str) -> None:
    tools = tool_sequence(_steps(tape))
    missing = [t for t in (first_tool, second_tool) if t not in tools]
    assert not missing, f"tool(s) never called: {missing}"
    assert tools.index(first_tool) < tools.index(second_tool), (
        f"{second_tool!r} was called before {first_tool!r}"
    )


def assert_no_repeated_error_loop(tape: TapePath, max_repeats: int = 3) -> None:
    errors = Counter(
        (s.tool.tool_name, s.tool.result_text)
        for s in _steps(tape)
        if s.tool is not None and s.tool.is_error
    )
    for (tool, text), count in errors.items():
        assert count <= max_repeats, (
            f"identical {tool!r} error repeated {count} times (max {max_repeats}): {text[:200]}"
        )


def assert_final_diff_clean(tape: TapePath) -> None:
    net_asserts: Counter[str] = Counter()
    ignores = 0
    for diff in applied_diffs(_steps(tape)):
        for path, sign, line in changed_lines(diff):
            delta = 1 if sign == "+" else -1
            if _TEST_FILE.search(path) and _ASSERTION.match(line):
                net_asserts[path] += delta
            if "# type: ignore" in line:
                ignores += delta
    deleted = {p: -n for p, n in net_asserts.items() if n < 0}
    assert not deleted, f"run deleted {sum(deleted.values())} assertion(s): {deleted}"
    assert ignores <= 0, f"run added {ignores} '# type: ignore' comment(s)"
