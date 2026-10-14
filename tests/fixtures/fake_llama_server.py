"""测试用的极简假 llama-server（票 15b/15c 自包含夹具）。

实现生命周期与最低限度的 OpenAI 兼容面：
* `GET /health`               → `{"status":"ok"}`
* `GET /props`                → `{"model_path": <--model 传入值>}`
* `GET /v1/models`            → 单个模型（供 IslandEngine.load_model 解析模型名）
* `POST /v1/chat/completions` → 固定回显（供委托链路验证）

它**故意忽略**其它参数（`-c` / `-ngl` / `--cpu-moe` / `--spec-type` …），
因为这些只由被测代码组装与断言 —— 这样测试不依赖 20 GiB 真模型。

`--startup-delay N`（秒）模拟"进程已起但模型还没加载完"，用于验证就绪判据
必须是 `/health` 而不是 TCP connect。
"""

from __future__ import annotations

import http.server
import json
import socketserver
import sys
import time

MODEL_ID = "fake-llama-server-model"


def _arg_value(argv: list[str], name: str, default: str = "") -> str:
    if name in argv:
        index = argv.index(name)
        if index + 1 < len(argv):
            return argv[index + 1]
    return default


def main(argv: list[str]) -> int:
    port = int(_arg_value(argv, "--port", "0"))
    host = _arg_value(argv, "--host", "127.0.0.1")
    model = _arg_value(argv, "-m", "")
    delay = float(_arg_value(argv, "--startup-delay", "0") or 0)

    if delay > 0:
        time.sleep(delay)

    class Handler(http.server.BaseHTTPRequestHandler):
        def _send_json(self, payload: dict, status: int = 200) -> None:
            body = json.dumps(payload).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self) -> None:  # noqa: N802 - http.server 约定
            if self.path == "/health":
                self._send_json({"status": "ok"})
            elif self.path == "/props":
                self._send_json({"model_path": model})
            elif self.path == "/v1/models":
                self._send_json({"object": "list", "data": [{"id": MODEL_ID, "object": "model"}]})
            else:
                self.send_response(404)
                self.end_headers()

        def do_POST(self) -> None:  # noqa: N802
            length = int(self.headers.get("Content-Length") or 0)
            raw = self.rfile.read(length) if length else b"{}"
            try:
                request = json.loads(raw or b"{}")
            except ValueError:
                request = {}
            if self.path == "/v1/chat/completions":
                last = ""
                messages = request.get("messages") or []
                if messages and isinstance(messages[-1], dict):
                    last = str(messages[-1].get("content") or "")
                self._send_json({
                    "id": "chatcmpl-fake",
                    "object": "chat.completion",
                    "model": MODEL_ID,
                    "choices": [{
                        "index": 0,
                        "message": {"role": "assistant", "content": f"echo:{last}"},
                        "finish_reason": "stop",
                    }],
                    "usage": {
                        "prompt_tokens": 1,
                        "completion_tokens": 1,
                        "total_tokens": 2,
                    },
                })
            else:
                self.send_response(404)
                self.end_headers()

        def log_message(self, *args: object) -> None:  # 静音
            pass

    socketserver.TCPServer.allow_reuse_address = True
    with socketserver.TCPServer((host, port), Handler) as httpd:
        httpd.serve_forever()
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
