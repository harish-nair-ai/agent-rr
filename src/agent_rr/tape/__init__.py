"""Tape storage. `tape.record` is the tool-call decorator (lazy to avoid an import cycle)."""

from typing import Any


def __getattr__(name: str) -> Any:
    if name == "record":
        from agent_rr.proxy.decorator import record

        return record
    raise AttributeError(name)
