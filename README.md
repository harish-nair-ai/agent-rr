# agent-rr

Flight recorder, time-travel debugger, chaos injector, pytest freezer, and git bisect
engine for coding agents. Every LLM call, tool call, and exact workspace tree is recorded
into a portable `.tape` (single SQLite WAL file + git tree SHAs).

```
agent-rr record [--out s.tape] [--chaos PRESET|chaos.toml]... [--seed 42] -- <agent cmd>
agent-rr inspect s.tape [--step N]
agent-rr fork s.tape --at-step N [--model M] [--system-prompt-file F] [--out c.tape] -- <cmd>
agent-rr diff a.tape b.tape                       # exit 0 identical, 1 diverged
agent-rr freeze s.tape --out tests/evals/test_x.py
agent-rr bisect --good G --bad B --tape s.tape --at-step N --check "<shell>" \
    [--trials 3] [--harness-paths GLOB]... [--skip-endpoint-check] -- <agent cmd>
```

- `record` starts a local SSE-transparent proxy and sets `OPENAI_BASE_URL`,
  `ANTHROPIC_BASE_URL`, and `AGENT_RR_TAPE_PATH` for the child. Default tape:
  `<repo>/.agent-rr/tapes/<tape_id>.tape`. Exit code = the child's.
- Snapshots use a throwaway `GIT_INDEX_FILE`; your index, staged changes, HEAD and refs
  are never touched. `fork`/`bisect` run in ephemeral `git worktree`s under the temp dir,
  removed on exit, error, SIGINT, or SIGTERM.
- Python tool functions are recorded with `@tape.record` (`from agent_rr import tape`);
  tool-side chaos presets apply there. In a forked run, tool calls before the fork step
  are replayed from the tape instead of executing.
- Chaos presets: `rate-limit-429` (LLM proxy), `truncate-stdout`, `flaky-test`,
  `tool-timeout`, `permission-denied` (tool wrapper). Same seed + same call sequence
  gives byte-identical `chaos_events`.
- Frozen tests import only `pytest`, `pathlib`, `os`, `agent_rr.testing`, and evaluate
  `$AGENT_RR_TAPE_PATH` when set, else the reference tape copied next to the test.

Exit codes: 0 ok, 1 internal error / `diff` diverged, 2 bad args / missing tape / invalid
step, 3 bisect inconclusive; `record`/`fork` return the child's code.

## How it works, and why it's built this way

agent-rr applies record/replay debugging (in the style of rr) to coding agents. A `.tape` is a single portable SQLite file holding every LLM call, tool call and git tree SHA.

- Snapshots never touch your git state. Each one builds a throwaway index (`GIT_INDEX_FILE` → `read-tree HEAD` → `add -A` → `write-tree`), so your staged changes, HEAD and refs stay exactly as they were. Tests check the index bytes before and after.
- The LLM reverse proxy streams SSE chunks to the agent as they arrive, using asyncio backpressure, and rebuilds the full response for the tape in parallel. Time to first token is unchanged; a test with a gated upstream fails if the proxy buffers.
- `fork --at-step N` restores step N's tree in an ephemeral `git worktree`, serves earlier LLM calls from the tape at zero cost, warns when a replayed request no longer matches the recording, then goes live with your `--model` or system-prompt override. Worktrees are removed in `finally` and on SIGINT/SIGTERM.
- Chaos faults (429, truncated stdout, flaky tests, timeouts, EACCES) come from an isolated `random.Random(seed)`, so the same seed gives byte-identical fault logs.
- `bisect` uses ⌈2K/3⌉ majority voting over K trials per commit, so a flaky agent doesn't throw the search off. Prompt and harness files come from the commit under test while the rest of the workspace comes from the tape.
- Secrets (auth headers and OpenAI, Anthropic, GitHub and AWS key patterns) are scrubbed before anything is written to SQLite.

Cross-package file formats are documented in `CONTRACTS.md`.
