from __future__ import annotations

import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer


HOST = "127.0.0.1"
PORT = 8500
MODEL = "webai2api-model"


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def _json(self, payload: dict, status: int = 200) -> None:
        raw = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("content-type", "application/json; charset=utf-8")
        self.send_header("content-length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)

    def do_GET(self) -> None:
        if self.path.rstrip("/") == "/v1/models":
            self._json({"object": "list", "data": [{"id": MODEL, "object": "model"}]})
            return
        self._json({"ok": True})

    def do_POST(self) -> None:
        length = int(self.headers.get("content-length") or "0")
        body = self.rfile.read(length).decode("utf-8") if length else "{}"
        try:
            payload = json.loads(body)
        except Exception:
            payload = {}
        messages = payload.get("messages") if isinstance(payload.get("messages"), list) else []
        joined = "\n".join(str(m.get("content") or "") for m in messages if isinstance(m, dict))
        if "Tool result for read_file" in joined:
            content = "mock upstream received the tool result and continued"
        elif "read" in joined.lower() or "读取" in joined:
            content = '```tool_json\n{"name":"read_file","args":{"path":"README.md"}}\n```'
        else:
            content = "mock upstream normal reply"
        self._json(
            {
                "id": "mock-chatcmpl",
                "object": "chat.completion",
                "choices": [
                    {
                        "index": 0,
                        "finish_reason": "stop",
                        "message": {"role": "assistant", "content": content},
                    }
                ],
            }
        )

    def log_message(self, fmt: str, *args: object) -> None:
        return


ThreadingHTTPServer((HOST, PORT), Handler).serve_forever()
