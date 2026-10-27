"""TUI SSE transport regression tests."""

from __future__ import annotations

import os
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from tui_api import ApiClient, iter_chat_payloads  # noqa: E402


@pytest.mark.parametrize("chunked", [True, False], ids=["chunked", "connection-close"])
def test_sse_delivers_small_events_before_stream_closes(chunked):
    """A complete SSE event must be visible while later bytes are withheld."""
    send_token = threading.Event()
    send_done = threading.Event()
    finish_stream = threading.Event()
    stop = threading.Event()

    class PausedSseHandler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def do_POST(self):  # noqa: N802 - stdlib handler API
            self.rfile.read(int(self.headers.get("Content-Length", "0")))
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Connection", "close")
            if chunked:
                self.send_header("Transfer-Encoding", "chunked")
            self.end_headers()
            frames = [
                (None, b": keep-alive\n\n"),
                (None, b'data: {"start":true}\n\n'),
                (send_token, b'data: {"token":"ok"}\n\n'),
                (send_done, b'data: {"done":true,"response":"ok"}\n\n'),
            ]
            for gate, frame in frames:
                if gate is not None and (not gate.wait(5) or stop.is_set()):
                    return
                if chunked:
                    frame = f"{len(frame):x}\r\n".encode("ascii") + frame + b"\r\n"
                self.wfile.write(frame)
                self.wfile.flush()
            if finish_stream.wait(5) and not stop.is_set() and chunked:
                self.wfile.write(b"0\r\n\r\n")
                self.wfile.flush()

        def log_message(self, *_args):
            return

    server = ThreadingHTTPServer(("127.0.0.1", 0), PausedSseHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    api = ApiClient(host="127.0.0.1", port=server.server_address[1])
    payloads = iter_chat_payloads(api, "hello", read_timeout=1)
    try:
        assert next(payloads) == {"start": True}
        send_token.set()
        assert next(payloads) == {"token": "ok"}
        send_done.set()
        assert next(payloads) == {"done": True, "response": "ok"}
        finish_stream.set()
        with pytest.raises(StopIteration):
            next(payloads)
    finally:
        stop.set()
        send_token.set()
        send_done.set()
        finish_stream.set()
        payloads.close()
        server.shutdown()
        thread.join(timeout=3)
        server.server_close()
