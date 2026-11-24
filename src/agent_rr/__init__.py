"""agent-rr: flight recorder, time-travel debugger, and git bisect for coding agents."""

from agent_rr.proxy.decorator import record
from agent_rr.tape.store import TapeReader, TapeWriter

__all__ = ["TapeReader", "TapeWriter", "record"]
__version__ = "0.1.0"
