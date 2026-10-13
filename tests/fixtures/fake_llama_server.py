"""测试用的极简假 llama-server（票 15b 自包含夹具）。

只实现生命周期测试需要的两个端点：
* `GET /health` → `{"status":"ok"}`
* `GET /props`  → `{"model_path": <--model 传入值>}`

它**故意忽略**所有其它参数（`-m` / `-c` / `-ngl` / `--cpu-moe` / `--spec-type` …），
因为这些参数只由被测代码组装与断言，不需要真解析 —— 这样测试不依赖 20 GiB 真模型。

支持 `--startup-delay N`（秒）来模拟"进程已起但模型还没加载完"，
用于验证就绪判据必须是 `/health` 而不是 TCP connect。
"""

from __future__ import annotations

import http.server
import json
import socketserver
import sys
import time


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
        def do_GET(self) -> None:  # noqa: N802 - http.server 约定
            if self.path == "/health":
                body = json.dumps({"status": "ok"}).encode()
            elif self.path == "/props":
                body = json.dumps({"model_path": model}).encode()
            else:
                self.send_response(404)
                self.end_headers()
                return
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *args: object) -> None:  # 静音
            pass

    socketserver.TCPServer.allow_reuse_address = True
    with socketserver.TCPServer((host, port), Handler) as httpd:
        httpd.serve_forever()
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
