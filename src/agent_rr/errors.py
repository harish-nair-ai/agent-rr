"""The single user-facing error type: rendered as `error: <message>` with a documented exit code."""

from pydantic import ValidationError


class AgentRRError(Exception):
    def __init__(self, message: str, exit_code: int = 2) -> None:
        super().__init__(message)
        self.exit_code = exit_code


WORDING = (
    ("Field required", "missing required key"),
    ("Input should be", "must be"),
    ("String should", "must"),
    ("List should", "must"),
)


def describe(exc: ValidationError) -> str:
    """`key.path: problem (got value)` per error, without pydantic model names or URLs."""
    parts = []
    for err in exc.errors(include_url=False):
        loc = ".".join(str(p) for p in err["loc"]) or "(top level)"
        if err["type"] == "extra_forbidden":
            parts.append(f"{loc}: unknown key")
            continue
        msg = err["msg"].removeprefix("Value error, ")
        for pydantic_text, human in WORDING:
            msg = msg.replace(pydantic_text, human)
        got = err.get("input")
        suffix = "" if err["type"] == "missing" or isinstance(got, dict) else f" (got {got!r})"
        parts.append(f"{loc}: {msg}{suffix}")
    return "; ".join(parts)
