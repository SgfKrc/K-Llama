"""OpenAI-compatible chat completions over the layer chain.

Scope, deliberately narrow: one endpoint, streaming and non-streaming, no auth, no
model switching. The point is to let existing OpenAI clients point at a split model
without a translation layer, not to reimplement the API surface.

Two honesty constraints shape this file:

  * Sampling happens inside the shim, on the tail stage, using llama.cpp's own
    sampler chain. Every parameter that reaches the tail is echoed back in the
    response as the value that actually took effect -- read out of the native handle,
    not out of the request, because the clamping rules ("top_k >= n_vocab means off",
    "repeat_last_n 0 means the default") live there. Parameters this server cannot
    honour are reported in `ignored_parameters` rather than dropped.
  * There is no multi-turn memory. Every request replays its own prompt from
    position 0; the KV cache is reset between requests. Sending a long `messages`
    list works because it is all re-prefilled, not because context is retained.
    The sampler is reset on the same boundary, so penalties do not leak from one
    conversation into the next.
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

# Parameters this server still cannot honour. Reported per response instead of
# dropped, so a client can see why its output did not change. Everything else in the
# OpenAI sampling surface now maps onto a real sampler in llama.dll.
#
#   logit_bias -- needs llama_sampler_init_logit_bias plus a per-token table, which
#                 the shim does not expose yet.
#   min_p      -- a first-class llama.cpp sampler, but the shim's setter signature
#                 (SAMPLING.md §4.2) has no min_p argument, so it stays unsupported
#                 rather than being silently approximated by top_p.
UNSUPPORTED = ("logit_bias", "min_p")

#: OpenAI request field -> shim setter keyword. `top_k` and `repetition_penalty` are
#: not OpenAI parameters, but llama.cpp's own API takes both and clients of local
#: servers routinely send them, so they are honoured instead of ignored.
SAMPLING_FIELDS = {
    "temperature": "temperature",
    "top_p": "top_p",
    "top_k": "top_k",
    "seed": "seed",
    "frequency_penalty": "frequency_penalty",
    "presence_penalty": "presence_penalty",
    "repetition_penalty": "repeat_penalty",
}


def sampling_request(body: dict) -> dict:
    """Translate an OpenAI request body into shim setter arguments.

    Only fields the client actually sent are forwarded: the shim treats every
    argument's low end as "off", so inventing a temperature the client never asked
    for would silently change its output.
    """
    params: dict = {}
    for field, keyword in SAMPLING_FIELDS.items():
        if field not in body or body[field] is None:
            continue
        value = body[field]
        try:
            params[keyword] = int(value) if keyword in ("top_k", "seed") else float(value)
        except (TypeError, ValueError):
            continue        # a malformed value is not worth a 400; the default applies
    return params


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

    def generate(self, prompt: str, max_tokens: int, sampling: dict | None = None):
        """Yield Chunk, serialised under the chain lock.

        The tail's sampler is configured inside the same lock, right after the reset,
        so a request can never sample with another request's penalties history.

        ``sampling`` is either a non-empty parameter dict, or ``None`` meaning "this
        request asked for nothing". The two are NOT the same: an empty dict would be
        passed straight to the setter, whose defaults install a greedy chain, and the
        handle would stay in that state for every later request. ``None`` therefore
        clears the chain instead.
        """
        with self._lock:
            for client in self._clients:
                client.reset()
            if self._clients:
                tail = self._clients[-1]
                if sampling:
                    # The tail is the only stage that turns logits into an id; every
                    # other kind ignores this and returns {}.
                    tail.set_sampling(**sampling)
                else:
                    # Restore the pristine argmax path, not a greedy chain.
                    tail.clear_sampling()
            yield from generate_stream(self._clients, prompt, max_tokens=max_tokens)

    def sampling_state(self) -> dict:
        """What the tail stage currently has in effect, for the response body."""
        if not self._clients:
            return {}
        try:
            return self._clients[-1].sampling()
        except Exception:  # noqa: BLE001 - an echo must never fail a completion
            return {}

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
        # `None` (not `{}`) when the client asked for nothing: the chain then *clears*
        # the sampler instead of installing a greedy one, which keeps the pristine
        # argmax path and stops one request's configuration leaking into the next.
        sampling = sampling_request(body) or None
        completion_id = f"chatcmpl-{uuid.uuid4().hex[:24]}"
        created = int(time.time())

        if not self.chain.ready:
            self._json(503, {"error": {"message": "stages not ready",
                                        "type": "service_unavailable"}})
            return

        if stream:
            self._stream_response(completion_id, created, model, prompt, max_tokens,
                                  ignored, sampling)
        else:
            self._full_response(completion_id, created, model, prompt, max_tokens,
                                ignored, sampling)

    # -- response shapes -------------------------------------------------------

    def _full_response(self, cid: str, created: int, model: str, prompt: str,
                       max_tokens: int, ignored: list[str],
                       sampling: dict | None = None) -> None:
        text_parts: list[str] = []
        tokens = 0
        started = time.perf_counter()
        try:
            for chunk in self.chain.generate(prompt, max_tokens, sampling):
                text_parts.append(chunk.text)
                tokens += 1
        except Exception as exc:  # noqa: BLE001 - report, do not kill the server
            self._json(500, {"error": {"message": f"{type(exc).__name__}: {exc}",
                                       "type": "internal_error"}})
            return
        elapsed = time.perf_counter() - started
        # Read back from the native handle while the lock is free: this is what the
        # client actually got, not what it asked for (clamping happens in C).
        effective = self.chain.sampling_state()
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
                "sampling": effective,
            },
        }
        self._json(200, payload)

    def _stream_response(self, cid: str, created: int, model: str, prompt: str,
                         max_tokens: int, ignored: list[str],
                         sampling: dict | None = None) -> None:
        self._sse_head()
        first = {"id": cid, "object": "chat.completion.chunk", "created": created,
                 "model": model,
                 "choices": [{"index": 0, "delta": {"role": "assistant"},
                              "finish_reason": None}]}
        # The stream announces the requested configuration; the closing chunk carries
        # the effective one, because the shim can only be asked after the first
        # generate() has configured it.
        announced = {"ignored_parameters": ignored} if ignored else {}
        announced["requested_sampling"] = dict(sampling or {})
        first["slimarc"] = announced
        self._sse(first)
        tokens = 0
        try:
            for chunk in self.chain.generate(prompt, max_tokens, sampling):
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
                   "usage": {"completion_tokens": tokens},
                   "slimarc": {"sampling": self.chain.sampling_state()}})
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
    # failing. The estimate accounts for mmap residency, not file size sum.
    from chain_client import estimate_ram_bytes, mem_budget_bytes
    estimate = estimate_ram_bytes(stages)
    budget = mem_budget_bytes()
    if estimate > budget:
        raise SystemExit(
            f"refusing to start: local stages need an estimated "
            f"{estimate / 2**30:.2f} GiB resident but only "
            f"{budget / 2**30:.2f} GiB is safe. Move a stage to another host.")

    chain = Chain(stages, shim, config.ctx())
    print(f"loading {len(stages)} stage(s)…", flush=True)
    chain.start()
    print(f"ready on http://{args.host}:{args.port}  "
          f"(stages={len(stages)}, sampling on the tail stage; argmax by default)",
          flush=True)

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
