"""Deterministic seeded fault injection (spec §5.3, contracts §6.4 / D14).

All randomness comes from one isolated `random.Random(seed)` per injector; a draw happens
once per eligible probability-mode fault, in call order, so the same seed and call sequence
reproduce byte-identical `chaos_events`.
"""

import fnmatch
import json
import random
import tomllib
from pathlib import Path
from typing import Any, get_args

from pydantic import ValidationError

from agent_rr.errors import AgentRRError, describe
from agent_rr.models import ChaosConfig, ChaosEvent, ChaosRule, Preset

TRUNCATION_MARKER = "\n...[TRUNCATED BY CHAOS]"
TRUNCATE_BYTES = 300
PRESET_FLAG_PROBABILITY = 0.3
PRESETS: tuple[str, ...] = get_args(Preset)


class InjectedToolError(RuntimeError):
    """Synthetic transient tool failure (flaky-test)."""


def load_chaos(specs: list[str], seed: int | None) -> ChaosConfig:
    """Each `--chaos` value is a preset name or a chaos.toml path; `--seed` wins over the file."""
    faults: list[ChaosRule] = []
    file_seed: int | None = None
    for spec in specs:
        if spec in PRESETS:
            faults.append(
                ChaosRule.model_validate({"preset": spec, "probability": PRESET_FLAG_PROBABILITY})
            )
            continue
        path = Path(spec)
        if not path.is_file():
            raise AgentRRError(
                f"unknown chaos preset or missing chaos.toml: {spec!r} "
                f"(presets: {', '.join(PRESETS)})"
            )
        try:
            cfg = ChaosConfig.model_validate(tomllib.loads(path.read_text()))
        except ValidationError as exc:
            raise AgentRRError(f"invalid chaos config {path}: {describe(exc)}") from exc
        except tomllib.TOMLDecodeError as exc:
            raise AgentRRError(f"invalid chaos config {path}: {exc}") from exc
        file_seed = cfg.seed if file_seed is None else file_seed
        faults.extend(cfg.faults)
    effective = seed if seed is not None else (file_seed if file_seed is not None else 0)
    return ChaosConfig(seed=effective, faults=faults)


def truncate_stdout(text: str) -> str:
    clipped = text.encode()[:TRUNCATE_BYTES].decode(errors="ignore")
    return clipped + TRUNCATION_MARKER


def matching_path(rule: ChaosRule, arguments: dict[str, Any]) -> str | None:
    for value in arguments.values():
        if isinstance(value, str) and fnmatch.fnmatchcase(value, rule.path_glob):
            return value
    return None


class ChaosInjector:
    def __init__(self, config: ChaosConfig) -> None:
        self.config = config
        self._rng = random.Random(config.seed)
        self._spent: set[int] = set()  # flaky-test rules that already fired

    def llm_fault(self, llm_idx: int) -> ChaosRule | None:
        return self._first(
            [(i, r) for i, r in enumerate(self.config.faults) if r.target == "llm"], llm_idx
        )

    def tool_fault(
        self, tool_idx: int, tool_name: str, arguments: dict[str, Any]
    ) -> ChaosRule | None:
        eligible = [
            (i, r)
            for i, r in enumerate(self.config.faults)
            if r.target == "tool"
            and i not in self._spent
            and fnmatch.fnmatchcase(tool_name, r.tool_glob)
            and (r.preset != "permission-denied" or matching_path(r, arguments) is not None)
        ]
        return self._first(eligible, tool_idx)

    def event(self, rule: ChaosRule, detail: dict[str, Any]) -> ChaosEvent:
        return ChaosEvent(
            preset=rule.preset,
            seed=self.config.seed,
            target=rule.target,
            detail_json=json.dumps(detail, sort_keys=True, separators=(",", ":")),
        )

    def _first(self, eligible: list[tuple[int, ChaosRule]], idx: int) -> ChaosRule | None:
        fired: ChaosRule | None = None
        for i, rule in eligible:
            hit = (
                idx in rule.at_steps
                if rule.at_steps is not None
                else (self._rng.random() < rule.probability)
            )
            if hit and fired is None:
                fired = rule
                if rule.preset == "flaky-test":
                    self._spent.add(i)
        return fired
