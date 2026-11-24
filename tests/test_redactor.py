from agent_rr.tape.redactor import (
    REDACTED_HEADER,
    REDACTED_SECRET,
    redact_headers,
    redact_payload,
    redact_text,
)

OPENAI = "sk-proj_" + "A" * 30
ANTHROPIC = "sk-ant-api03-" + "b" * 30
GITHUB = "ghp_" + "c" * 36
AWS = "AKIA" + "ABCDEFGHIJKLMNOP"
BEARER = "Bearer abcdefghijklmnopqrstuvwxyz0123"


def test_each_secret_pattern_is_scrubbed() -> None:
    for secret in (OPENAI, ANTHROPIC, GITHUB, AWS, BEARER):
        out = redact_text(f"before {secret} after")
        assert out == f"before {REDACTED_SECRET} after", secret


def test_anthropic_key_is_one_redaction_not_a_partial_openai_match() -> None:
    assert redact_text(ANTHROPIC) == REDACTED_SECRET


def test_short_lookalikes_are_preserved() -> None:
    text = "sk-short ghp_tiny AKIA123 bearer xyz"
    assert redact_text(text) == text


def test_headers_case_insensitive() -> None:
    headers = {
        "Authorization": "Bearer x",
        "X-API-Key": "k",
        "api-key": "k",
        "Cookie": "c",
        "Set-Cookie": "c",
        "Proxy-Authorization": "p",
        "Content-Type": "application/json",
    }
    out = redact_headers(headers)
    assert out["Content-Type"] == "application/json"
    assert all(v == REDACTED_HEADER for k, v in out.items() if k != "Content-Type")


def test_payload_recurses_and_scrubs_header_named_keys() -> None:
    payload = {
        "messages": [{"role": "user", "content": f"my key is {OPENAI}"}],
        "nested": {"Authorization": "token-without-pattern", "n": 3, "ok": True, "none": None},
    }
    out = redact_payload(payload)
    assert out == {
        "messages": [{"role": "user", "content": f"my key is {REDACTED_SECRET}"}],
        "nested": {"Authorization": REDACTED_HEADER, "n": 3, "ok": True, "none": None},
    }
    assert payload["messages"][0]["content"].endswith(OPENAI)  # input not mutated
