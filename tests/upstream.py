"""A real HTTP/1.1 mock LLM upstream (OpenAI + Anthropic shapes) on an ephemeral 127.0.0.1 port."""

from __future__ import annotations

import json
import threading
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

SSE_CHUNK_COUNT = 10


def openai_sse_events(model: str) -> list[bytes]:
    """Exactly SSE_CHUNK_COUNT events: role, 7 words, finish+usage, [DONE]."""

    def ev(obj: dict[str, Any]) -> bytes:
        return f"data: {json.dumps(obj)}\n\n".encode()

    base = {"id": "chatcmpl-1", "object": "chat.completion.chunk", "created": 1, "model": model}
    events = [ev({**base, "choices": [{"index": 0, "delta": {"role": "assistant"}}]})]
    events += [
        ev({**base, "choices": [{"index": 0, "delta": {"content": f"w{i} "}}]}) for i in range(7)
    ]
    events.append(
        ev(
            {
                **base,
                "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 11, "completion_tokens": 7, "total_tokens": 18},
            }
        )
    )
    events.append(b"data: [DONE]\n\n")
    assert len(events) == SSE_CHUNK_COUNT
    return events


def anthropic_sse_events(model: str) -> list[bytes]:
    def ev(name: str, obj: dict[str, Any]) -> bytes:
        return f"event: {name}\ndata: {json.dumps({'type': name, **obj})}\n\n".encode()

    return [
        ev(
            "message_start",
            {
                "message": {
                    "id": "msg_1",
                    "type": "message",
                    "role": "assistant",
                    "model": model,
                    "content": [],
                    "stop_reason": None,
                    "usage": {"input_tokens": 21, "output_tokens": 1},
                }
            },
        ),
        ev("content_block_start", {"index": 0, "content_block": {"type": "text", "text": ""}}),
        ev("content_block_delta", {"index": 0, "delta": {"type": "text_delta", "text": "Hel"}}),
        ev("content_block_delta", {"index": 0, "delta": {"type": "text_delta", "text": "lo"}}),
        ev("content_block_stop", {"index": 0}),
        ev(
            "content_block_start",
            {
                "index": 1,
                "content_block": {"type": "tool_use", "id": "tu_1", "name": "bash", "input": {}},
            },
        ),
        ev(
            "content_block_delta",
            {"index": 1, "delta": {"type": "input_json_delta", "partial_json": '{"cmd": "l'}},
        ),
        ev(
            "content_block_delta",
            {"index": 1, "delta": {"type": "input_json_delta", "partial_json": 's"}'}},
        ),
        ev("content_block_stop", {"index": 1}),
        ev(
            "message_delta",
            {
                "delta": {"stop_reason": "tool_use", "stop_sequence": None},
                "usage": {"output_tokens": 9},
            },
        ),
        ev("message_stop", {}),
    ]


@dataclass
class Received:
    path: str
    headers: dict[str, str]
    body: dict[str, Any]


@dataclass
class Upstream:
    requests: list[Received] = field(default_factory=list)
    gate: threading.Event | None = None  # if set, pause after the first SSE chunk until released
    gate_released_in_time: bool | None = None
    fail_status: int | None = None
    sent: list[bytes] = field(default_factory=list)
    url: str = ""

    def start(self) -> str:
        upstream = self

        class Handler(BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"

            def log_message(self, *args: Any) -> None:
                return

            def do_POST(self) -> None:
                raw = self.rfile.read(int(self.headers["Content-Length"]))
                body = json.loads(raw)
                upstream.requests.append(
                    Received(self.path, {k.lower(): v for k, v in self.headers.items()}, body)
                )
                if upstream.fail_status:
                    self._send_json(upstream.fail_status, {"error": {"message": "upstream down"}})
                    return
                model = body.get("model", "m")
                anthropic = self.path.endswith("/messages")
                if body.get("stream"):
                    events = anthropic_sse_events(model) if anthropic else openai_sse_events(model)
                    self._send_sse(events)
                elif anthropic:
                    self._send_json(
                        200,
                        {
                            "id": "msg_2",
                            "type": "message",
                            "role": "assistant",
                            "model": model,
                            "content": [{"type": "text", "text": f"echo:{len(body['messages'])}"}],
                            "usage": {"input_tokens": 5, "output_tokens": 2},
                        },
                    )
                else:
                    self._send_json(
                        200,
                        {
                            "id": "chatcmpl-2",
                            "object": "chat.completion",
                            "model": model,
                            "choices": [
                                {
                                    "index": 0,
                                    "message": {
                                        "role": "assistant",
                                        "content": f"echo:{len(body['messages'])}:{model}",
                                    },
                                    "finish_reason": "stop",
                                }
                            ],
                            "usage": {"prompt_tokens": 3, "completion_tokens": 2},
                        },
                    )

            def _send_json(self, status: int, obj: dict[str, Any]) -> None:
                data = json.dumps(obj).encode()
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

            def _send_sse(self, events: list[bytes]) -> None:
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream")
                self.send_header("Transfer-Encoding", "chunked")
                self.end_headers()
                for i, event in enumerate(events):
                    self.wfile.write(f"{len(event):x}\r\n".encode() + event + b"\r\n")
                    self.wfile.flush()
                    upstream.sent.append(event)
                    if i == 0 and upstream.gate is not None:
                        upstream.gate_released_in_time = upstream.gate.wait(10)
                self.wfile.write(b"0\r\n\r\n")
                self.wfile.flush()

        self._server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self._thread = threading.Thread(target=self._server.serve_forever, daemon=True)
        self._thread.start()
        self.url = f"http://127.0.0.1:{self._server.server_address[1]}"
        return self.url

    def stop(self) -> None:
        self._server.shutdown()
        self._server.server_close()
        self._thread.join(10)
