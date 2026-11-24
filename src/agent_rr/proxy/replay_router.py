"""Hybrid fork replay: LLM requests < fork_llm_idx are mocked from the parent tape (§6.3)."""

import copy
import hashlib
import json
from typing import Any

from agent_rr.models import LlmCall


def canonical_json(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def request_digest(obj: Any) -> str:
    return hashlib.sha256(canonical_json(obj).encode()).hexdigest()


def apply_overrides(
    provider: str, body: dict[str, Any], model: str | None, system_prompt: str | None
) -> dict[str, Any]:
    """Return a copy of `body` with --model / --system-prompt-file applied."""
    out = copy.deepcopy(body)
    if model is not None:
        out["model"] = model
    if system_prompt is not None:
        if provider == "anthropic":
            out["system"] = system_prompt
        else:
            messages = list(out.get("messages") or [])
            if messages and isinstance(messages[0], dict) and messages[0].get("role") == "system":
                messages[0] = {**messages[0], "content": system_prompt}
            else:
                messages.insert(0, {"role": "system", "content": system_prompt})
            out["messages"] = messages
    return out


class ReplayRouter:
    def __init__(
        self,
        recorded: list[LlmCall],
        fork_llm_idx: int,
        model: str | None = None,
        system_prompt: str | None = None,
    ) -> None:
        self.recorded = recorded
        self.fork_llm_idx = fork_llm_idx
        self.model = model
        self.system_prompt = system_prompt

    def mock_for(self, llm_idx: int) -> LlmCall | None:
        if llm_idx < min(self.fork_llm_idx, len(self.recorded)):
            return self.recorded[llm_idx]
        return None

    def live_body(self, provider: str, body: dict[str, Any]) -> dict[str, Any]:
        return apply_overrides(provider, body, self.model, self.system_prompt)
