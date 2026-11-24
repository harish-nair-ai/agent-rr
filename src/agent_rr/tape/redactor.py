"""Deterministic secret and header scrubber, applied before anything enters SQLite (§6)."""

import re
from collections.abc import Mapping
from typing import Any

REDACTED_HEADER = "<REDACTED_HEADER>"
REDACTED_SECRET = "<REDACTED_SECRET>"

SENSITIVE_HEADERS = frozenset(
    {"authorization", "x-api-key", "api-key", "cookie", "set-cookie", "proxy-authorization"}
)

# Order matters: the Anthropic pattern must win over the broader OpenAI one.
_SECRET_PATTERNS = tuple(
    re.compile(p)
    for p in (
        r"sk-ant-[A-Za-z0-9_-]{20,}",
        r"sk-[A-Za-z0-9_-]{20,}",
        r"gh[pousr]_[A-Za-z0-9]{30,}",
        r"AKIA[0-9A-Z]{16}",
        r"(?i)bearer\s+[A-Za-z0-9._~+/=-]{16,}",
    )
)


def redact_text(text: str) -> str:
    for pattern in _SECRET_PATTERNS:
        text = pattern.sub(REDACTED_SECRET, text)
    return text


def redact_headers(headers: Mapping[str, str]) -> dict[str, str]:
    return {
        k: REDACTED_HEADER if k.lower() in SENSITIVE_HEADERS else redact_text(v)
        for k, v in headers.items()
    }


def redact_payload(value: Any) -> Any:
    """Return a scrubbed deep copy of a JSON-like value."""
    if isinstance(value, str):
        return redact_text(value)
    if isinstance(value, Mapping):
        return {
            k: REDACTED_HEADER
            if isinstance(k, str) and k.lower() in SENSITIVE_HEADERS
            else redact_payload(v)
            for k, v in value.items()
        }
    if isinstance(value, list | tuple):
        return [redact_payload(v) for v in value]
    return value
