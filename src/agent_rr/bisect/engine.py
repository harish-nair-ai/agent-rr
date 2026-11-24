"""Binary search over harness commits with K-trial majority voting (spec §5.5, D15, D16)."""

import fnmatch
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path

from agent_rr.errors import AgentRRError
from agent_rr.fork import fork_router, start_child_tape, validate_step
from agent_rr.git_engine.snapshot import run_git, temp_index
from agent_rr.git_engine.worktree import WorktreeContext
from agent_rr.models import Step, TapeMeta
from agent_rr.runner import run_recorded
from agent_rr.tape.store import TapeWriter, read_steps

DEFAULT_HARNESS_PATHS = ("AGENTS.md", "CLAUDE.md", "envelope.toml", "prompts/**", ".cursorrules")
INCONCLUSIVE = 3


def majority_threshold(trials: int) -> int:
    """ceil(2K/3): 1/1, 2/3, 4/5 (D15)."""
    return (2 * trials + 2) // 3


def _is_harness(path: str, patterns: list[str]) -> bool:
    return any(fnmatch.fnmatchcase(path, p) for p in patterns)


def overlay_tree(repo: Path, commit: str, workspace_tree: str, harness: list[str]) -> str:
    """Tree = workspace_tree's files, except harness paths, which come from `commit`."""

    def entries(treeish: str) -> list[str]:
        out = run_git(["ls-tree", "-r", "-z", treeish], repo)
        return [e for e in out.split("\0") if e]

    keep = [e for e in entries(workspace_tree) if not _is_harness(e.split("\t", 1)[1], harness)]
    keep += [e for e in entries(commit) if _is_harness(e.split("\t", 1)[1], harness)]
    with temp_index() as env:
        run_git(["update-index", "-z", "--index-info"], repo, env, stdin="\0".join(keep) + "\0")
        return run_git(["write-tree"], repo, env).strip()


@dataclass(frozen=True)
class Bisect:
    repo: Path
    meta: TapeMeta
    steps: list[Step]
    at_step: int
    check: str
    cmd: list[str]
    trials: int
    harness: list[str]

    def trial(self, commit: str) -> bool:
        tree = overlay_tree(self.repo, commit, self.steps[self.at_step].pre_tree_sha, self.harness)
        with (
            WorktreeContext(self.repo, commit, tree) as wt,
            tempfile.TemporaryDirectory(prefix="agent-rr-tape-") as tmp,
            TapeWriter(Path(tmp) / "trial.tape") as writer,
        ):
            start_child_tape(writer, self.meta, self.steps, self.at_step, self.cmd)
            router = fork_router(self.steps, self.at_step, None, None)
            run_recorded(self.cmd, wt, writer, wt, router, None, stdout=sys.stderr)
            return subprocess.run(self.check, shell=True, cwd=wt, stdout=sys.stderr).returncode == 0

    def is_good(self, commit: str) -> bool:
        passes = sum(self.trial(commit) for _ in range(self.trials))
        good = passes >= majority_threshold(self.trials)
        print(
            f"agent-rr: {commit[:12]} {'GOOD' if good else 'BAD'} "
            f"({passes}/{self.trials} trials passed)",
            file=sys.stderr,
        )
        return good


def _rev(repo: Path, rev: str) -> str:
    try:
        return run_git(["rev-parse", "--verify", "-q", f"{rev}^{{commit}}"], repo).strip()
    except AgentRRError:
        raise AgentRRError(f"unknown revision: {rev}") from None


def bisect(
    repo: Path,
    good: str,
    bad: str,
    tape: Path,
    at_step: int,
    check: str,
    cmd: list[str],
    trials: int,
    harness_paths: list[str],
    skip_endpoint_check: bool,
) -> tuple[str, int]:
    """Return (first bad commit, number of evaluations). Raises AgentRRError(exit 3)."""
    if trials < 1:
        raise AgentRRError("--trials must be >= 1")
    meta, steps = read_steps(tape)
    validate_step(steps, at_step)
    good_sha, bad_sha = _rev(repo, good), _rev(repo, bad)
    run_git(["cat-file", "-e", f"{steps[at_step].pre_tree_sha}^{{tree}}"], repo)
    commits = run_git(["rev-list", "--reverse", f"{good_sha}..{bad_sha}"], repo).split()
    if not commits:
        raise AgentRRError(f"inconclusive: no commits in {good}..{bad}", INCONCLUSIVE)
    engine = Bisect(repo, meta, steps, at_step, check, cmd, trials, harness_paths)
    if not skip_endpoint_check:
        if not engine.is_good(good_sha):
            raise AgentRRError(f"inconclusive: --good {good} does not pass --check", INCONCLUSIVE)
        if engine.is_good(bad_sha):
            raise AgentRRError(f"inconclusive: --bad {bad} passes --check", INCONCLUSIVE)
    lo, hi, evaluations = 0, len(commits) - 1, 0
    while lo < hi:
        mid = (lo + hi) // 2
        evaluations += 1
        if engine.is_good(commits[mid]):
            lo = mid + 1
        else:
            hi = mid
    return commits[lo], evaluations


def describe(repo: Path, commit: str, harness_paths: list[str]) -> str:
    header = run_git(
        ["show", "-s", "--format=Author: %an <%ae>%nDate:   %ad%n%n    %B", commit], repo
    )
    pathspecs = [f":(glob){p}" for p in harness_paths]
    diff = run_git(
        ["diff", "--no-color", "--no-ext-diff", f"{commit}^", commit, "--", *pathspecs], repo
    )
    return f"first bad commit: {commit}\n{header.rstrip()}\n\n{diff}"
