"""Pure SSE parsing and canonical response reassembly for OpenAI and Anthropic streams."""

import json
from typing import Any


def split_events(text: str) -> list[str]:
    """Split a raw SSE body into events, each keeping its terminating blank line."""
    events: list[str] = []
    start = 0
    while (end := text.find("\n\n", start)) != -1:
        events.append(text[start : end + 2])
        start = end + 2
    if text[start:]:
        events.append(text[start:])
    return events


def event_payloads(events: list[str]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for event in events:
        data = "\n".join(
            line[5:].removeprefix(" ") for line in event.splitlines() if line.startswith("data:")
        )
        if not data or data == "[DONE]":
            continue
        try:
            obj = json.loads(data)
        except json.JSONDecodeError:
            continue
        if isinstance(obj, dict):
            out.append(obj)
    return out


def reassemble(payloads: list[dict[str, Any]]) -> dict[str, Any]:
    if any(p.get("type") == "message_start" for p in payloads):
        return _anthropic(payloads)
    for p in payloads:
        if p.get("type") == "response.completed" and isinstance(p.get("response"), dict):
            return dict(p["response"])  # OpenAI Responses API
    if any("choices" in p for p in payloads):
        return _openai_chat(payloads)
    return {"events": payloads}


def _openai_chat(payloads: list[dict[str, Any]]) -> dict[str, Any]:
    out: dict[str, Any] = {"object": "chat.completion", "choices": []}
    choices: dict[int, dict[str, Any]] = {}
    for p in payloads:
        for key in ("id", "created", "model", "system_fingerprint"):
            if key in p:
                out[key] = p[key]
        if p.get("usage"):
            out["usage"] = p["usage"]
        for ch in p.get("choices") or []:
            slot = choices.setdefault(
                ch.get("index", 0),
                {"index": ch.get("index", 0), "message": {"role": "assistant", "content": None}},
            )
            delta = ch.get("delta") or {}
            msg = slot["message"]
            if "role" in delta:
                msg["role"] = delta["role"]
            if delta.get("content") is not None:
                msg["content"] = (msg["content"] or "") + delta["content"]
            for tc in delta.get("tool_calls") or []:
                calls = msg.setdefault("tool_calls", [])
                while len(calls) <= tc.get("index", 0):
                    calls.append(
                        {"id": None, "type": "function", "function": {"name": "", "arguments": ""}}
                    )
                call = calls[tc.get("index", 0)]
                call["id"] = tc.get("id") or call["id"]
                call["type"] = tc.get("type") or call["type"]
                fn = tc.get("function") or {}
                call["function"]["name"] += fn.get("name") or ""
                call["function"]["arguments"] += fn.get("arguments") or ""
            if ch.get("finish_reason") is not None:
                slot["finish_reason"] = ch["finish_reason"]
    out["choices"] = [choices[i] for i in sorted(choices)]
    return out


def _anthropic(payloads: list[dict[str, Any]]) -> dict[str, Any]:
    msg: dict[str, Any] = {}
    blocks: dict[int, dict[str, Any]] = {}
    partial_json: dict[int, str] = {}
    for p in payloads:
        kind = p.get("type")
        if kind == "message_start":
            msg = dict(p.get("message") or {})
        elif kind == "content_block_start":
            blocks[p["index"]] = dict(p.get("content_block") or {})
        elif kind == "content_block_delta":
            block = blocks.setdefault(p["index"], {})
            delta = p.get("delta") or {}
            dtype = delta.get("type")
            if dtype == "text_delta":
                block["text"] = block.get("text", "") + delta.get("text", "")
            elif dtype == "input_json_delta":
                partial_json[p["index"]] = partial_json.get(p["index"], "") + delta.get(
                    "partial_json", ""
                )
            elif dtype == "thinking_delta":
                block["thinking"] = block.get("thinking", "") + delta.get("thinking", "")
            elif dtype == "signature_delta":
                block["signature"] = block.get("signature", "") + delta.get("signature", "")
        elif kind == "message_delta":
            msg.update(p.get("delta") or {})
            msg["usage"] = {**(msg.get("usage") or {}), **(p.get("usage") or {})}
    for idx, raw in partial_json.items():
        try:
            blocks[idx]["input"] = json.loads(raw) if raw else {}
        except json.JSONDecodeError:
            blocks[idx]["input"] = raw
    msg["content"] = [blocks[i] for i in sorted(blocks)]
    return msg


def token_usage(response: dict[str, Any]) -> tuple[int, int]:
    usage = response.get("usage")
    if not isinstance(usage, dict):
        return 0, 0
    inp = usage.get("prompt_tokens", usage.get("input_tokens", 0))
    out = usage.get("completion_tokens", usage.get("output_tokens", 0))
    return (inp if isinstance(inp, int) else 0), (out if isinstance(out, int) else 0)
