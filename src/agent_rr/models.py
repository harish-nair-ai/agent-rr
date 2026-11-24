"""Frozen pydantic v2 models for every external boundary: tape rows and chaos.toml."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

StepKind = Literal["llm_call", "tool_call", "chaos_fault"]
StepStatus = Literal["ok", "error", "steered", "injected_fault"]
Preset = Literal[
    "rate-limit-429", "truncate-stdout", "flaky-test", "tool-timeout", "permission-denied"
]
Target = Literal["llm", "tool"]


class Frozen(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class TapeMeta(Frozen):
    tape_id: str
    schema_version: str = "0.1"
    created_at: float
    command_argv: list[str]
    repo_root: str
    base_commit_sha: str
    parent_tape_id: str | None = None
    fork_step_idx: int | None = None
    chaos_seed: int | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class LlmCall(Frozen):
    provider: Literal["openai", "anthropic", "generic"]
    endpoint: str
    model: str
    request_json: str
    response_json: str
    raw_sse_chunks: list[str] | None = None
    http_status: int
    input_tokens: int = 0
    output_tokens: int = 0
    latency_ms: float


class ToolCall(Frozen):
    tool_name: str
    arguments_json: str
    result_text: str
    is_error: bool = False
    exit_code: int | None = None


class ChaosEvent(Frozen):
    preset: Preset
    seed: int
    target: Target
    detail_json: str = "{}"


class StepData(Frozen):
    """A step as written: `step_idx` is allocated by the writer (D11)."""

    ts_start: float
    ts_end: float
    kind: StepKind
    pre_tree_sha: str
    post_tree_sha: str
    diff_text: str = ""
    status: StepStatus = "ok"
    llm: LlmCall | None = None
    tool: ToolCall | None = None
    chaos: ChaosEvent | None = None


class Step(StepData):
    step_idx: int


PRESET_TARGET: dict[str, Target] = {
    "rate-limit-429": "llm",
    "truncate-stdout": "tool",
    "flaky-test": "tool",
    "tool-timeout": "tool",
    "permission-denied": "tool",
}


class ChaosRule(Frozen):
    preset: Preset
    target: Target
    at_steps: list[int] | None = None
    probability: float = Field(default=0.0, ge=0.0, le=1.0)
    tool_glob: str = "*"
    path_glob: str = "*"
    delay_s: float = Field(default=0.5, ge=0.0)

    @model_validator(mode="before")
    @classmethod
    def _default_target(cls, data: Any) -> Any:
        if isinstance(data, dict) and "target" not in data and data.get("preset") in PRESET_TARGET:
            return {**data, "target": PRESET_TARGET[data["preset"]]}
        return data

    @model_validator(mode="after")
    def _target_matches_preset(self) -> ChaosRule:
        expected = PRESET_TARGET[self.preset]
        if self.target != expected:
            raise ValueError(f"preset {self.preset!r} requires target = {expected!r}")
        return self


class ChaosConfig(Frozen):
    seed: int = 0
    faults: list[ChaosRule] = Field(default_factory=list)
