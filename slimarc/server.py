"""OpenAI-compatible chat completions over the layer chain.

Scope, deliberately narrow: one endpoint, streaming and non-streaming, no auth, no
model switching. The point is to let existing OpenAI clients point at a split model
without a translation layer, not to reimplement the API surface.

Two honesty constraints shape this file:

  * Decoding is argmax only -- the shim returns one token per step and never samples.
    `temperature`/`top_p`/`seed` are accepted so clients do not error, recorded in the
    response as ignored, and NOT silently pretended to work. A caller who asks for
    temperature 0.9 and gets deterministic output should be told why.
  * There is no multi-turn memory. Every request replays its own prompt from
    position 0; the KV cache is reset between requests. Sending a long `messages`
    list works because it is all re-prefilled, not because context is retained.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import threading
import time
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import config  # noqa: E402
import topology  # noqa: E402
from chain_client import StageClient, spawn_worker, _terminate  # noqa: E402
from stream import generate_stream  # noqa: E402

DEFAULT_SHIM = Path(os.environ.get("SA_SHIM") or
                    (Path(__file__).resolve().parent / "shim" / "bin" / "qlh_keep_head.dll"))

# Parameters this server cannot honour. Reported per response instead of dropped, so
# a client can see why its output does not vary.
UNSUPPORTED = ("temperature", "top_p", "top_k", "min_p", "repetition_penalty",
               "presence_penalty", "frequency_penalty", "seed", "logit_bias")


class Chain:
    """Owns the stage processes for the server's lifetime.

    Loading a 40-layer split costs tens of seconds, so the stages stay up across
    requests. A lock serialises requests because the KV cache is shared state: two
    concurrent conversations would interleave positions and corrupt both.
    """

    def __init__(self, stages: tuple, shim: Path, ctx: int) -> None:
        self._stages = stages
        self._shim = shim
        self._ctx = ctx
        self._lock = threading.Lock()
        self._clients: list[StageClient] = []
        self._procs: list[tuple] = []
        self.ready = False

    def start(self) -> None:
        for stage in self._stages:
            if stage.is_local:
                self._procs.append((*spawn_worker(stage, self._shim, self._ctx), stage))
        self._clients = [StageClient(s) for s in self._stages]
        for client in self._clients:
            client.connect()
        self.ready = True

    def generate(self, prompt: str, max_tokens: int):
        """Yield Chunk, serialised under the chain lock."""
        with self._lock:
            for client in self._clients:
                client.reset()
            yield from generate_stream(self._clients, prompt, max_tokens=max_tokens)

    def stop(self) -> None:
        for client in self._clients:
            try:
                client.close()
            except Exception:
                pass
        if self._procs:
            _terminate(self._procs)
        self.ready = False


def flatten_messages(messages: list[dict]) -> str:
    """Collapse a chat into one prompt.

    There is no chat template here: the split model may not be a chat model at all,
    and inventing `<|im_start|>` markers for it would produce confidently wrong
    output. Role prefixes are kept as plain text so multi-turn content survives.
    """
    parts: list[str] = []
    for msg in messages:
        role = str(msg.get("role", "user"))
        content = msg.get("content", "")
        if isinstance(content, list):
            # OpenAI's multimodal shape; keep only the text parts.
            content = "".join(p.get("text", "") for p in content
                              if isinstance(p, dict) and p.get("type") == "text")
        content = str(content)
        if role == "system":
            parts.append(f"系统：{content}")
        elif role == "assistant":
            parts.append(f"助手：{content}")
        else:
            parts.append(f"用户：{content}")
    parts.append("助手：")
    return "\n".join(parts)


class Handler(BaseHTTPRequestHandler):
    server_version = "slimarc/0.1"
    protocol_version = "HTTP/1.1"

    chain: Chain
    default_max_tokens: int

    def log_message(self, fmt: str, *args) -> None:  # noqa: A003 - base class name
        # Default logs one line per request to stderr; keep that but stamp the time.
        sys.stderr.write("%s [slimarc] %s\n" % (time.strftime("%H:%M:%S"), fmt % args))

    # -- helpers ---------------------------------------------------------------

    def _json(self, code: int, payload: dict) -> None:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(body)

    def _sse_head(self) -> None:
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream; charset=utf-8")
        self.send_header("Cache-Control", "no-cache")
        # No Content-Length is sent for a stream, so the body must be terminated by
        # closing the connection. keep-alive here makes every client wait forever
        # after [DONE] -- measured: the reader never reaches EOF.
        self.send_header("Connection", "close")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()

    def _sse(self, payload: dict | str) -> None:
        data = payload if isinstance(payload, str) else json.dumps(payload, ensure_ascii=False)
        self.wfile.write(f"data: {data}\n\n".encode("utf-8"))
        self.wfile.flush()

    # -- routes ----------------------------------------------------------------

    def do_OPTIONS(self) -> None:  # noqa: N802 - base class name
        self.send_response(204)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Headers", "*")
        self.send_header("Access-Control-Allow-Methods", "POST, GET, OPTIONS")
        self.end_headers()

    def do_GET(self) -> None:  # noqa: N802
        path = self.path.split("?")[0].rstrip("/")
        if path in ("/v1/models", "/models"):
            self._json(200, {"object": "list", "data": [{
                "id": "slimarc", "object": "model", "owned_by": "local",
            }]})
            return
        if path in ("/healthz", "/"):
            self._json(200, {"ok": self.chain.ready, "stages": len(self.chain._stages)})
            return
        self._json(404, {"error": {"message": f"no route {path}", "type": "not_found"}})

    def do_POST(self) -> None:  # noqa: N802
        path = self.path.split("?")[0].rstrip("/")
        if path not in ("/v1/chat/completions", "/chat/completions"):
            self._json(404, {"error": {"message": f"no route {path}", "type": "not_found"}})
            return
        try:
            length = int(self.headers.get("Content-Length", "0"))
            body = json.loads(self.rfile.read(length) or b"{}")
        except (ValueError, OSError) as exc:
            self._json(400, {"error": {"message": f"bad request body: {exc}",
                                        "type": "invalid_request_error"}})
            return

        messages = body.get("messages")
        if not isinstance(messages, list) or not messages:
            self._json(400, {"error": {"message": "messages must be a non-empty array",
                                        "type": "invalid_request_error"}})
            return

        prompt = flatten_messages(messages)
        max_tokens = int(body.get("max_tokens") or self.default_max_tokens)
        stream = bool(body.get("stream"))
        model = str(body.get("model") or "slimarc")
        ignored = [k for k in UNSUPPORTED if k in body]
        completion_id = f"chatcmpl-{uuid.uuid4().hex[:24]}"
        created = int(time.time())

        if not self.chain.ready:
            self._json(503, {"error": {"message": "stages not ready",
                                        "type": "service_unavailable"}})
            return

        if stream:
            self._stream_response(completion_id, created, model, prompt, max_tokens, ignored)
        else:
            self._full_response(completion_id, created, model, prompt, max_tokens, ignored)

    # -- response shapes -------------------------------------------------------

    def _full_response(self, cid: str, created: int, model: str, prompt: str,
                       max_tokens: int, ignored: list[str]) -> None:
        text_parts: list[str] = []
        tokens = 0
        started = time.perf_counter()
        try:
            for chunk in self.chain.generate(prompt, max_tokens):
                text_parts.append(chunk.text)
                tokens += 1
        except Exception as exc:  # noqa: BLE001 - report, do not kill the server
            self._json(500, {"error": {"message": f"{type(exc).__name__}: {exc}",
                                       "type": "internal_error"}})
            return
        elapsed = time.perf_counter() - started
        payload = {
            "id": cid, "object": "chat.completion", "created": created, "model": model,
            "choices": [{
                "index": 0,
                "message": {"role": "assistant", "content": "".join(text_parts)},
                "finish_reason": "length" if tokens >= max_tokens else "stop",
            }],
            "usage": {"prompt_tokens": len(prompt), "completion_tokens": tokens,
                      "total_tokens": len(prompt) + tokens},
            "slimarc": {
                "tokens_per_second": round(tokens / elapsed, 2) if elapsed > 0 else 0,
                "elapsed_seconds": round(elapsed, 3),
                "ignored_parameters": ignored,
                "note": "decoding is argmax; sampling parameters have no effect"
                        if ignored else "",
            },
        }
        self._json(200, payload)

    def _stream_response(self, cid: str, created: int, model: str, prompt: str,
                         max_tokens: int, ignored: list[str]) -> None:
        self._sse_head()
        first = {"id": cid, "object": "chat.completion.chunk", "created": created,
                 "model": model,
                 "choices": [{"index": 0, "delta": {"role": "assistant"},
                              "finish_reason": None}]}
        if ignored:
            first["slimarc"] = {"ignored_parameters": ignored,
                                "note": "decoding is argmax; sampling parameters "
                                        "have no effect"}
        self._sse(first)
        tokens = 0
        try:
            for chunk in self.chain.generate(prompt, max_tokens):
                if not chunk.text:
                    continue        # a partial UTF-8 byte pair; wait for the rest
                tokens += 1
                self._sse({"id": cid, "object": "chat.completion.chunk", "created": created,
                           "model": model,
                           "choices": [{"index": 0, "delta": {"content": chunk.text},
                                        "finish_reason": None}]})
        except (BrokenPipeError, ConnectionResetError):
            return              # client hung up; the chain lock is released by the generator
        except Exception as exc:  # noqa: BLE001
            self._sse({"error": {"message": f"{type(exc).__name__}: {exc}"}})
            self._sse("[DONE]")
            return
        self._sse({"id": cid, "object": "chat.completion.chunk", "created": created,
                   "model": model,
                   "choices": [{"index": 0, "delta": {},
                                "finish_reason": "length" if tokens >= max_tokens else "stop"}],
                   "usage": {"completion_tokens": tokens}})
        self._sse("[DONE]")


def main() -> int:
    ap = argparse.ArgumentParser(description="OpenAI-compatible front end for the chain")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8080)
    ap.add_argument("--shim", default=str(DEFAULT_SHIM))
    ap.add_argument("--topology", default="")
    ap.add_argument("--max-tokens", type=int, default=256)
    args = ap.parse_args()

    if args.topology:
        os.environ[config.TOPOLOGY_ENV] = args.topology
    stages = topology.load(config.topology_path())
    shim = Path(args.shim)
    if not shim.is_file():
        raise SystemExit(f"keep-head shim not found: {shim}")

    # Same guard as chain_client: loading every local artifact at once is the whole
    # memory cost, and Windows answers an overshoot by thrashing swap rather than
    # failing.
    from chain_client import mem_budget_bytes
    local_bytes = sum(s.model.stat().st_size for s in stages if s.is_local)
    budget = mem_budget_bytes()
    if local_bytes > budget:
        raise SystemExit(
            f"refusing to start: local artifacts total {local_bytes / 2**30:.2f} GiB "
            f"but only {budget / 2**30:.2f} GiB is safe (available x 0.75). "
            f"Move a stage to another host.")

    chain = Chain(stages, shim, config.ctx())
    print(f"loading {len(stages)} stage(s)…", flush=True)
    chain.start()
    print(f"ready on http://{args.host}:{args.port}  "
          f"(stages={len(stages)}, argmax decoding)", flush=True)

    Handler.chain = chain
    Handler.default_max_tokens = args.max_tokens
    server = ThreadingHTTPServer((args.host, args.port), Handler)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nshutting down", flush=True)
    finally:
        server.server_close()
        chain.stop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
