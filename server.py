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

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

import config  # noqa: E402
import topology  # noqa: E402
from chat import Chat, ChatError  # noqa: E402 - after the sys.path insert above
from chain_client import (StageClient, _terminate, bring_up, context_shortfall,
                          tail_rs_depth, vram_after_load, vram_before_load,
                          vram_report)  # noqa: E402
from stream import generate_stream  # noqa: E402


class ContextOverflow(ValueError):
    """The request cannot fit the KV window; the HTTP layer answers 400, not 500."""


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

    def __init__(self, stages: tuple, shim: Path, ctx: int,
                 spec_k: int = 0, dflash: str = "", dflash_block: int = 6,
                 cascade_mtp: int = 0) -> None:
        self._stages = stages
        self._shim = shim
        self._ctx = ctx
        self._spec_k = spec_k
        self._dflash = dflash
        self._dflash_block = dflash_block
        self._cascade_mtp = int(cascade_mtp)
        self._lock = threading.Lock()
        self._clients: list[StageClient] = []
        self._procs: list[tuple] = []
        self.ready = False

    @property
    def model_path(self) -> Path:
        """The checkpoint whose ``tokenizer.chat_template`` renders this chain's prompts.

        Every stage of a chain loads the same file (a split model is one artifact with
        layer ranges), so stage 0 speaks for all of them. Read rather than configured a
        second time: a second model path is a second thing that can disagree.
        """
        return self._stages[0].model

    def start(self) -> None:
        # Local stages spawned and launchable remote ones asked, by the same function the
        # CLI and the TUI use: `start_cmd`/`ssh_user` used to be read in chain_client only,
        # so a cross-machine topology started from here waited 120 s for a worker that
        # nobody had launched.
        bring_up(self._stages, self._shim, self._ctx,
                 tail_rs_depth(self._dflash, self._dflash_block, self._cascade_mtp),
                 procs=self._procs)
        self._clients = [StageClient(s) for s in self._stages]
        # Pass the process so connect() can fail fast: without it a worker that dies on
        # load is indistinguishable from one that is still loading, and the server sits
        # through the whole 120 s deadline before saying anything.
        local = {stage.index: proc for proc, _log, stage in self._procs}
        for client in self._clients:
            client.connect(proc=local.get(client.stage.index))
        self.ready = True

    def spec_plan(self, sampling: dict | None) -> dict:
        """This request's speculation outcome: ``{requested, active, reason}``.

        The greedy path keeps the startup configuration unchanged. A sampled request can
        also speculate now (`chain_client.speculate`'s rejection sampler), but only when the
        sampler is temperature-only: p and q are built from raw logits as softmax(logits/T),
        and top_k / top_p / penalties live inside the shim's sampler chain without ever being
        exposed as a distribution. Asking for them therefore *degrades* this request to plain
        decode -- and says so in the response. A silent downgrade would be indistinguishable
        from network jitter: the same client gets speculative speed until it passes top_p,
        then suddenly does not.

        The rule itself is `config.sampled_spec_problem`, shared with the CLI and the TUI;
        those two can refuse the combination outright because their flags are fixed for the
        run, while here it is decided per request.
        """
        greedy = config.is_greedy(sampling)
        if not (self._spec_k or self._dflash):
            return {"requested": False, "active": False, "reason": ""}
        if greedy:
            return {"requested": True, "active": True, "reason": ""}
        if self._dflash:
            # The block drafter answers with argmax rows only; there is no sampled verifier.
            return {"requested": True, "active": False,
                    "reason": config.DFLASH_NEEDS_GREEDY}
        params = sampling or {}
        try:
            temperature = float(params.get("temperature", 1.0))
            top_k = int(params.get("top_k", 0) or 0)
            top_p = float(params.get("top_p", 1.0))
        except (TypeError, ValueError):
            return {"requested": True, "active": False,
                    "reason": "malformed sampler parameters"}
        extra = {k: v for k, v in params.items()
                 if k not in ("temperature", "top_k", "top_p", "seed")}
        why = config.sampled_spec_problem(temperature, top_k, top_p, extra)
        return {"requested": True, "active": not why, "reason": why}

    def generate(self, prompt: str, max_tokens: int, sampling: dict | None = None,
                 stats: dict | None = None):
        """Yield Chunk, serialised under the chain lock.

        The tail's sampler is configured inside the same lock, right after the reset,
        so a request can never sample with another request's penalties history.

        ``sampling`` is either a non-empty parameter dict, or ``None`` meaning "this
        request asked for nothing". The two are NOT the same: an empty dict would be
        passed straight to the setter, whose defaults install a greedy chain, and the
        handle would stay in that state for every later request. ``None`` therefore
        clears the chain instead.

        ``stats`` (optional) receives the tokenized prompt length. It still comes from the
        generator, which owns the count reported to the caller: the tokenize below exists
        only for the fit check, and asks once more than strictly necessary so that an
        unfittable request is refused before the KV is reset. Passing pre-tokenized ids
        would avoid that round trip but means changing `generate_stream`'s contract.
        """
        with self._lock:
            # Refuse a request that cannot fit before touching any state. An overrun does
            # not fail cleanly: llama_decode returns rc=-1 partway through the answer, and
            # the client sees a truncated stream rather than a rejected request.
            if self._clients:
                n_prompt = len(self._clients[0].tokenize(prompt))
                if why := context_shortfall(n_prompt, max_tokens, self._ctx):
                    raise ContextOverflow(why)
            for client in self._clients:
                client.reset()
            greedy = config.is_greedy(sampling)
            if self._clients:
                tail = self._clients[-1]
                if sampling:
                    # The tail is the only stage that turns logits into an id; every
                    # other kind ignores this and returns {}.
                    tail.set_sampling(**sampling)
                else:
                    # Restore the pristine argmax path, not a greedy chain.
                    tail.clear_sampling()
            spec = self.spec_plan(sampling)
            if stats is not None:
                stats["speculation"] = spec
            spec_sampling = None
            if not greedy and spec["active"]:
                # Temperature-only by construction (spec_plan refused everything else), so
                # the rejection sampler's p/q match the tail's own transform.
                spec_sampling = {
                    "temperature": float((sampling or {}).get("temperature", 1.0)),
                    "rng": np.random.default_rng((sampling or {}).get("seed")),
                }
            yield from generate_stream(
                self._clients, prompt, max_tokens=max_tokens,
                spec_k=self._spec_k if spec["active"] else 0,
                dflash_path=self._dflash if (spec["active"] and greedy) else "",
                dflash_block=self._dflash_block,
                cascade_mtp=self._cascade_mtp,
                sampling=spec_sampling,
                stats=stats,
            )

    def drain(self, prompt: str, max_tokens: int, sampling: dict | None,
              sink=None):
        """One generation pass, shared by all four response shapes.

        Returns ``(text, completion_tokens, prompt_tokens, elapsed, stats)``. The four shapes
        used to carry their own loops, and the counting differed between them: the
        streaming variants skipped chunks whose text was empty (a partial UTF-8 byte
        group the assembler is still holding) before counting, the full variants counted
        them -- so the same answer reported a different ``completion_tokens`` depending
        on ``stream``. Counting tokens rather than chunks also stops the final flush
        chunk (``token == -1``) from being counted as one.

        ``stats`` is returned as well because it now carries the per-request speculation
        verdict: whether the configured drafts actually ran, and why not when they did not.
        """
        parts: list[str] = []
        tokens = 0
        stats: dict = {}
        started = time.perf_counter()
        for chunk in self.generate(prompt, max_tokens, sampling, stats=stats):
            if chunk.text:
                parts.append(chunk.text)
                if sink is not None:
                    sink(chunk.text)
            if chunk.token >= 0:
                tokens += 1
        return ("".join(parts), tokens, int(stats.get("prompt_tokens", 0)),
                time.perf_counter() - started, stats)

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


def _renderer(model):
    """One handle per request, open across render -> generate -> parse.

    `parse()` needs the `apply()` state of the *same* handle (`shim/qlh_chat.cpp`: parse
    before apply returns -1), so the handle cannot be closed when the prompt comes back.
    The old `flatten_messages` did exactly that, which left no way to parse the answer
    without a second, stateless handle -- and that one can only fail.
    """
    return Chat.for_model(model)


def normalize_tool_call_arguments(messages: list[dict]) -> list[dict]:
    """Turn historical `tool_calls[].function.arguments` strings into mappings.

    OpenAI sends the arguments of a past tool call as a JSON *string*, and the template
    renders each parameter with `tool_call.arguments | items`, where iterating a string
    yields nothing. Upstream parses the string back for templates whose jinja caps say
    `supports_object_arguments` (`chat.cpp`: `func_args_not_string`), and Qwen3.5 does hit
    that workaround -- measured here: the string and mapping forms render byte for byte
    identical. Doing it in Python anyway keeps the behaviour from depending on that
    template analysis: a template it classifies differently would silently drop every
    argument, with no error and no trace in the prompt.

    Malformed JSON is refused instead of passed through. Upstream would raise "chat
    template apply failed: Failed to parse tool call arguments as JSON" from inside the
    render, naming neither the message nor the field; a 400 does. An empty string means
    "this call took no arguments" (llama.cpp renders a zero-argument call as `{}`), not a
    syntax error.
    """
    out: list[dict] = []
    for msg in messages:
        if not isinstance(msg, dict) or msg.get("role") != "assistant":
            out.append(msg)
            continue
        calls = msg.get("tool_calls")
        if not isinstance(calls, list):
            out.append(msg)
            continue
        fixed = []
        for call in calls:
            function = call.get("function") if isinstance(call, dict) else None
            if not isinstance(function, dict) or not isinstance(function.get("arguments"), str):
                fixed.append(call)      # already a mapping, or not ours to reinterpret
                continue
            raw = function["arguments"].strip()
            if not raw:
                arguments: object = {}
            else:
                try:
                    arguments = json.loads(raw)
                except ValueError as exc:
                    raise ValueError(
                        f"tool_calls[].function.arguments is not valid JSON: "
                        f"{raw[:80]!r} ({exc})") from exc
                if not isinstance(arguments, dict):
                    raise ValueError(
                        f"tool_calls[].function.arguments must be a JSON object, got "
                        f"{type(arguments).__name__}: {raw[:80]!r}")
            fixed.append({**call, "function": {**function, "arguments": arguments}})
        out.append({**msg, "tool_calls": fixed})
    return out


def render_messages(renderer, messages: list[dict], *, tools=None, tool_choice=None,
                    parallel_tool_calls=None):
    """Render a request's messages through an already-open handle.

    This is the only prompt builder: the streaming chat path, the non-streaming one and
    `/v1/responses` all call it, so `test_contract`'s "no root Python file may spell the
    ChatML markers" cannot be bypassed by a second renderer.

    `tools` and `tool_choice` go through untouched -- the template decides how they read,
    and an empty tool list means "no tools", the same as OpenAI. `parallel_tool_calls` is
    forwarded only when the client sent it, so the DLL default (false) stays a default
    rather than becoming a value this server invented.
    """
    from chat import merge_system

    system, turns = merge_system(messages)
    if system is not None:
        turns = [{"role": "system", "content": system}, *turns]
    return renderer.render(normalize_tool_call_arguments(turns), tools=tools,
                           tool_choice=tool_choice,
                           parallel_tool_calls=parallel_tool_calls)


def _new_tool_call_id() -> str:
    return f"call_{uuid.uuid4().hex[:24]}"


def _tool_arguments(value) -> str:
    """OpenAI's wire form is a JSON string; the DLL already produces one."""
    if isinstance(value, str):
        return value
    return json.dumps(value, ensure_ascii=False)


def openai_assistant_message(parsed: dict, *, tokens: int, max_tokens: int,
                             new_id=_new_tool_call_id) -> tuple[dict, str]:
    """Build the OpenAI `message` and `finish_reason` from one parsed answer.

    `parsed` is `qlh_chat_parse`'s `msg`: already OpenAI-shaped, with `arguments` as a
    JSON string but no `id` (the DLL only emits one when the model wrote it). An id is
    generated per call because the client needs it to pair up the `role: "tool"` message
    it sends back, and without it the next turn's history is unpairable.

    `content` is `null` when a tool call carries no prose -- what the OpenAI spec shows
    for tool calls, and a more precise statement than an empty string. Prose the model
    wrote *before* calling is kept, because the Qwen template explicitly allows it.

    `reasoning_content` is the field name DSH reads from streaming deltas (its provider
    lists `reasoning_content` / `reasoning` / `reasoning_text`); its non-streaming path
    never runs, so carrying it here costs nothing and keeps the thinking text instead of
    dropping it.

    `finish_reason` stays inside DSH's vocabulary (`stop` / `length` / `tool_calls`; any
    other value becomes an error there). A tool call wins over the length cutoff: the call
    is what the client has to act on.
    """
    calls = parsed.get("tool_calls") or []
    content = parsed.get("content")
    message: dict = {"role": "assistant"}
    if calls:
        message["content"] = content or None
        message["tool_calls"] = [
            {"id": new_id(), "type": "function",
             "function": {
                 "name": str(call.get("function", {}).get("name", "")),
                 "arguments": _tool_arguments(call.get("function", {}).get("arguments", "{}")),
             }}
            for call in calls
        ]
        finish = "tool_calls"
    else:
        message["content"] = content or ""
        finish = "length" if tokens >= max_tokens else "stop"
    if reasoning := parsed.get("reasoning_content"):
        message["reasoning_content"] = reasoning
    return message, finish


# Responses request/response translation. Kept in its own module because the two
# protocol shapes differ more than the code that carries them.
from proto_responses import (  # noqa: E402 - after the sys.path insert above
    flatten_input, rejected_responses_field, responses_message, responses_object,
    responses_status, responses_usage,
)
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
        if path not in ("/v1/chat/completions", "/chat/completions", "/v1/responses"):
            self._json(404, {"error": {"message": f"no route {path}", "type": "not_found"}})
            return
        try:
            length = int(self.headers.get("Content-Length", "0"))
            body = json.loads(self.rfile.read(length) or b"{}")
        except (ValueError, OSError) as exc:
            self._json(400, {"error": {"message": f"bad request body: {exc}",
                                        "type": "invalid_request_error"}})
            return
        if not isinstance(body, dict):
            self._json(400, {"error": {"message": "request body must be an object",
                                        "type": "invalid_request_error"}})
            return
        if path == "/v1/responses":
            self._responses(body)
            return

        messages = body.get("messages")
        if not isinstance(messages, list) or not messages:
            self._json(400, {"error": {"message": "messages must be a non-empty array",
                                        "type": "invalid_request_error"}})
            return

        # `tools: []` is "no tools", the same as absent: the template then renders no tool
        # section, and the answer is not parsed for tool calls.
        tools = body.get("tools") or None
        tool_choice = body.get("tool_choice")
        parallel = body.get("parallel_tool_calls")
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

        # One handle for the whole request: `parse()` shares the `apply()` state, so the
        # renderer has to outlive generation. Streaming does not parse yet (fourth phase),
        # but it renders tools into the prompt, and closing the handle early there would
        # let the two paths diverge on exactly the input this phase is about.
        try:
            with _renderer(self.chain.model_path) as renderer:
                prompt = render_messages(renderer, messages, tools=tools,
                                         tool_choice=tool_choice,
                                         parallel_tool_calls=parallel).prompt
                if stream:
                    self._stream_response(completion_id, created, model, prompt,
                                          max_tokens, ignored, sampling)
                else:
                    self._full_response(completion_id, created, model, prompt,
                                        max_tokens, ignored, sampling,
                                        renderer=renderer, tools=bool(tools))
        except (ChatError, ValueError) as exc:
            # A template that refuses the request (bad tool_choice type, malformed tool
            # definition, unparsable historical arguments) is the client's mistake, and
            # the DLL's message says what it rejected.
            self._json(400, {"error": {"message": str(exc),
                                        "type": "invalid_request_error"}})

    # -- response shapes -------------------------------------------------------

    def _full_response(self, cid: str, created: int, model: str, prompt: str,
                       max_tokens: int, ignored: list[str],
                       sampling: dict | None = None, renderer=None,
                       tools: bool = False) -> None:
        try:
            text, tokens, prompt_tokens, elapsed, drain_stats = self.chain.drain(
                prompt, max_tokens, sampling)
        except ContextOverflow as exc:
            # The request asks for more than ctx can hold; that is the client's mistake,
            # so it is a 400 rather than the 500 every other failure is.
            self._json(400, {"error": {"message": str(exc),
                                       "type": "invalid_request_error"}})
            return
        except Exception as exc:  # noqa: BLE001 - report, do not kill the server
            self._json(500, {"error": {"message": f"{type(exc).__name__}: {exc}",
                                       "type": "internal_error"}})
            return
        # Read back from the native handle while the lock is free: this is what the
        # client actually got, not what it asked for (clamping happens in C).
        effective = self.chain.sampling_state()
        message: dict = {"role": "assistant", "content": text}
        finish = "length" if tokens >= max_tokens else "stop"
        parse_error = ""
        if tools:
            # Parsed only when the request carried tools. Without a tool definition the
            # PEG parser still splits `thinking` out of the text, which would change the
            # existing answer silently -- measured: a text that never closes its thinking
            # block comes back with content "" and everything in reasoning_content.
            try:
                parsed = renderer.parse(text, partial=False)["msg"]
            except ChatError as exc:
                # Parsing happens after generation, so failing the request would throw
                # away a complete answer. Answer with the raw text and say why.
                parse_error = str(exc)
            else:
                message, finish = openai_assistant_message(
                    parsed, tokens=tokens, max_tokens=max_tokens)
        slimarc = {
            "tokens_per_second": round(tokens / elapsed, 2) if elapsed > 0 else 0,
            "elapsed_seconds": round(elapsed, 3),
            "ignored_parameters": ignored,
            "sampling": effective,
            # Whether the configured drafts ran for *this* request, and the reason when
            # they did not. A sampled request that also asked for top_k/top_p degrades
            # to plain decode; without this field that slowdown would be invisible.
            "speculation": drain_stats.get("speculation"),
        }
        if parse_error:
            # A tool_calls-shaped answer that could not be parsed is not a 500: the text
            # is complete, only the split into content/reasoning/tool calls failed.
            slimarc["parse_error"] = parse_error
        payload = {
            "id": cid, "object": "chat.completion", "created": created, "model": model,
            "choices": [{
                "index": 0,
                "message": message,
                "finish_reason": finish,
            }],
            "usage": {"prompt_tokens": prompt_tokens, "completion_tokens": tokens,
                      "total_tokens": prompt_tokens + tokens},
            "slimarc": slimarc,
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

        def emit_text(text: str) -> None:
            self._sse({"id": cid, "object": "chat.completion.chunk", "created": created,
                       "model": model,
                       "choices": [{"index": 0, "delta": {"content": text},
                                    "finish_reason": None}]})

        try:
            _text, tokens, prompt_tokens, _elapsed, drain_stats = self.chain.drain(
                prompt, max_tokens, sampling, sink=emit_text)
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
                   "usage": {"prompt_tokens": prompt_tokens, "completion_tokens": tokens},
                   "slimarc": {"sampling": self.chain.sampling_state(),
                               "speculation": drain_stats.get("speculation")}})
        self._sse("[DONE]")

    # -- /v1/responses ---------------------------------------------------------

    def _responses(self, body: dict) -> None:
        bad = rejected_responses_field(body)
        if bad is not None:
            self._json(400, {"error": bad})
            return
        try:
            messages = flatten_input(body.get("input"), body.get("instructions"))
        except ValueError as exc:
            self._json(400, {"error": {"message": str(exc), "type": "invalid_request_error",
                                        "param": "input"}})
            return

        try:
            # The Responses route keeps the same renderer as the chat route, so the same
            # conversation still produces the same prompt on both endpoints.
            with _renderer(self.chain.model_path) as renderer:
                prompt = render_messages(renderer, messages).prompt
        except (ChatError, ValueError) as exc:
            self._json(400, {"error": {"message": str(exc),
                                        "type": "invalid_request_error"}})
            return
        # Same fallback and the same cast as the chat route, so an identical request
        # produces an identical `max_tokens` and therefore identical text.
        max_tokens = int(body.get("max_output_tokens") or self.default_max_tokens)
        sampling = sampling_request(body) or None
        model = str(body.get("model") or "slimarc")
        rid = f"resp_{uuid.uuid4().hex[:24]}"
        mid = f"msg_{uuid.uuid4().hex[:24]}"
        created = time.time()

        if not self.chain.ready:
            self._json(503, {"error": {"message": "stages not ready",
                                        "type": "service_unavailable"}})
            return

        meta = {"max_output_tokens": max_tokens,
                "temperature": body.get("temperature"),
                "top_p": body.get("top_p"),
                "metadata": body.get("metadata")}
        if body.get("stream"):
            self._responses_stream(rid, mid, created, model, prompt, max_tokens, meta,
                                   sampling)
        else:
            self._responses_full(rid, mid, created, model, prompt, max_tokens, meta,
                                 sampling)

    def _responses_full(self, rid: str, mid: str, created: float, model: str,
                        prompt: str, max_tokens: int, meta: dict,
                        sampling: dict | None) -> None:
        try:
            text, tokens, prompt_tokens, _elapsed, drain_stats = self.chain.drain(
                prompt, max_tokens, sampling)
        except ContextOverflow as exc:
            self._json(400, {"error": {"message": str(exc),
                                       "type": "invalid_request_error"}})
            return
        except Exception as exc:  # noqa: BLE001 - report, do not kill the server
            self._json(500, {"error": {"message": f"{type(exc).__name__}: {exc}",
                                        "type": "internal_error"}})
            return

        status, reason, item_status = responses_status(tokens, max_tokens)
        payload = responses_object(
            rid, created, model, status,
            [responses_message(mid, text, item_status)],
            responses_usage(prompt_tokens, tokens), incomplete_reason=reason,
            **{k: v for k, v in meta.items() if v is not None})
        # An extra field beside the protocol object, for the same reason the chat endpoint
        # has carried `slimarc` all along: a request whose sampler cannot be verified gets
        # plain decode, and the client has to be able to see that instead of guessing.
        payload["slimarc"] = {"speculation": drain_stats.get("speculation")}
        self._json(200, payload)

    def _responses_stream(self, rid: str, mid: str, created: float, model: str,
                          prompt: str, max_tokens: int, meta: dict,
                          sampling: dict | None) -> None:
        self._sse_head()
        seq = 0
        # Events are addressed by (response, item, index), so the same ids are repeated
        # in every event rather than being re-generated per event: the SDK groups
        # deltas by item_id, and a fresh id per event would split one answer into many.
        addr = {"item_id": mid, "output_index": 0, "content_index": 0}

        def emit(kind: str, **fields) -> None:
            nonlocal seq
            self._sse({"type": kind, "sequence_number": seq, **addr, **fields})
            seq += 1

        in_progress = responses_object(
            rid, created, model, "in_progress",
            [responses_message(mid, "", "in_progress")], None, **meta)
        self._sse({"type": "response.created", "sequence_number": seq,
                   "response": in_progress})
        seq += 1
        self._sse({"type": "response.in_progress", "sequence_number": seq,
                   "response": in_progress})
        seq += 1
        self._sse({"type": "response.output_item.added", "sequence_number": seq,
                   "output_index": 0,
                   "item": responses_message(mid, "", "in_progress")})
        seq += 1
        self._sse({"type": "response.content_part.added", "sequence_number": seq,
                   "output_index": 0, "content_index": 0, "item_id": mid,
                   "part": {"type": "output_text", "text": "",
                            "annotations": [], "logprobs": []}})
        seq += 1

        try:
            full, tokens, prompt_tokens, _elapsed, drain_stats = self.chain.drain(
                prompt, max_tokens, sampling, sink=lambda text: emit(
                    # `logprobs` is part of the schema, not an option here: the SDK's
                    # typed model declares it required, so it is always sent, empty.
                    "response.output_text.delta", delta=text, logprobs=[]))
        except (BrokenPipeError, ConnectionResetError):
            return              # client hung up; the chain lock is released by the generator
        except Exception as exc:  # noqa: BLE001
            self._sse({"type": "response.failed", "sequence_number": seq,
                       "response": responses_object(
                           rid, created, model, "failed", [], None,
                           error={"code": "server_error",
                                  "message": f"{type(exc).__name__}: {exc}"})})
            self._sse("[DONE]")
            return

        emit("response.output_text.done", text=full, logprobs=[])
        self._sse({"type": "response.content_part.done", "sequence_number": seq,
                   "output_index": 0, "content_index": 0, "item_id": mid,
                   "part": {"type": "output_text", "text": full,
                            "annotations": [], "logprobs": []}})
        seq += 1
        # Now that the loop is over it is known whether the cap was hit, so the item
        # can carry the same status the Response object is about to report.
        status, reason, item_status = responses_status(tokens, max_tokens)
        self._sse({"type": "response.output_item.done", "sequence_number": seq,
                   "output_index": 0,
                   "item": responses_message(mid, full, item_status)})
        seq += 1
        final = responses_object(
            rid, created, model, status,
            [responses_message(mid, full, item_status)],
            responses_usage(prompt_tokens, tokens), incomplete_reason=reason, **meta)
        final["slimarc"] = {"speculation": drain_stats.get("speculation")}
        self._sse({"type": "response.completed", "sequence_number": seq,
                   "response": final})
        self._sse("[DONE]")


def main() -> int:
    ap = argparse.ArgumentParser(description="OpenAI-compatible front end for the chain")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8080)
    ap.add_argument("--shim", default="", help="覆盖 SA_SHIM（见 config.find_shim）")
    ap.add_argument("--topology", default="")
    ap.add_argument("--max-tokens", type=int, default=256)
    ap.add_argument("--spec-k", type=int, default=0,
                    help="MTP 递归草稿数（默认 0 = 关闭）。贪心的请求全速生效；"
                         "采样请求只在 sampler 仅含 temperature 时生效，否则该请求降级"
                         "为普通 decode 并在响应里说明")
    ap.add_argument("--dflash", default="",
                    help="DFlash 草稿 GGUF；给了就走并行块草稿，只在 temperature=0 生效")
    ap.add_argument("--cascade-mtp", type=int, default=0,
                    help="DFlash 块整块接受后再用 MTP 追加几个草稿（0=关，见 chain_client）")
    ap.add_argument("--dflash-block", type=int, default=6,
                    help="DFlash 每轮草稿数 = block-1（默认 6 = k=5，实测拐点）")
    args = ap.parse_args()

    if args.topology:
        os.environ[config.TOPOLOGY_ENV] = args.topology
    stages = topology.load(config.topology_path())
    shim = config.find_shim(args.shim)
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

    chain = Chain(stages, shim, config.ctx(),
                  spec_k=args.spec_k, dflash=args.dflash, dflash_block=args.dflash_block,
                  cascade_mtp=args.cascade_mtp)
    # Must be set BEFORE the workers spawn: they inherit the environment at start-up, and a
    # later assignment would not reach the shim that tokenizes. Without it the ChatML
    # markers the renderer emits are split into ordinary fragments, which is
    # worse than sending plain text -- the model then sees broken control tokens.
    from chat import chat_token_env
    os.environ.update(chat_token_env())
    print(f"loading {len(stages)} stage(s)…", flush=True)
    # Sampled before the spawn for the same reason as in chain_client: the delta afterwards
    # is then attributable to this run, and the loaded-state check is the only thing that
    # catches a spill the up-front estimate got wrong.
    vram_before = vram_before_load(stages)
    chain.start()
    if report := vram_report(vram_after_load(stages, vram_before)):
        print("\n" + report, flush=True)
    spec_note = []
    if args.dflash:
        spec_note.append(f"DFlash block={args.dflash_block}")
    if args.spec_k:
        spec_note.append(f"MTP k={args.spec_k}")
    spec_txt = ("; speculation: " + ", ".join(spec_note)
                + " (greedy requests always; sampled requests only when the sampler is "
                  "temperature-only -- otherwise the response says so)") if spec_note else ""
    print(f"ready on http://{args.host}:{args.port}  "
          f"(stages={len(stages)}, sampling on the tail stage; argmax by default{spec_txt})",
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
