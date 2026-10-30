"""Responses API (`/v1/responses`) translation layer.

Split out of `server.py` because the two protocol shapes are genuinely different: Chat
Completions is a flat list of messages, Responses is an indexed event sequence closing
on a whole Response object. What they share -- the chain, the sampler, the UTF-8
assembler -- lives below both, so only translation belongs here.

Everything in this module is pure: a request body, or a list of already-decoded strings,
goes in; JSON-ready dicts or a normalised message list comes out. No socket, no chain,
no server state. That is what makes it testable without loading a model.

`flatten_input` returns *messages*, not a prompt, and `server.py` runs them through the
same `flatten_messages` the chat endpoint uses. Deliberate: the acceptance criterion is
that an identical request produces identical text on both endpoints, and a second prompt
formatter is exactly how those two drift apart.

UTF-8 is not handled here at all. `stream.generate_stream` already feeds token bytes
through `stream.Utf8Assembler`, so every string this module sees is valid `str`.
Decoding again here is the mistake the assembler exists to prevent.
"""
from __future__ import annotations

# -- Responses API (`/v1/responses`) -------------------------------------------
#
# The two protocols differ in shape, not in substance: both end up as one prompt
# through the same chain, so this whole layer is a translation pair -- `flatten_input`
# in, an event sequence or a Response object out -- and everything from
# `Chain.generate` down is shared, unchanged.
#
# UTF-8 is not handled here at all, deliberately. `stream.generate_stream` feeds raw
# token bytes through `stream.Utf8Assembler` and yields `Chunk.text` that is valid str
# by construction, so this layer only ever concatenates or forwards str. Decoding
# again here is exactly the mistake the assembler exists to prevent.

#: Fields whose value would make a client believe a capability is active when it is
#: not. Rejected with a 4xx rather than ignored: a client that asked for tools and got
#: plain text back has a much harder bug to find than one that got a 400 naming the
#: field. An empty `tools: []` or `include: []` is not a request for anything and is
#: accepted -- which falls out of testing truthiness of the value.
#:
#: Layout: ``(field, reported param, is_requested, message)``. `param` is what the
#: OpenAI error envelope should point at, so it carries a nested path where the field
#: itself is a container -- hence `text` appears twice, once per nested offender.
#:
#: Fields that only carry metadata (`metadata`, `user`, `safety_identifier`,
#: `prompt_cache_key`, `service_tier`) are absent from this table on purpose: they do
#: not change a single produced token, so ignoring them cannot mislead a client.
RESPONSES_UNSUPPORTED = (
    ("tools", "tools", bool,
     "tools is not supported: this server only does single-turn text generation"),
    ("tool_choice", "tool_choice", lambda v: v != "none",
     "tool_choice is not supported without tools"),
    ("previous_response_id", "previous_response_id", bool,
     "previous_response_id is not supported: this server keeps no response store"),
    ("store", "store", bool,
     "store is not supported: this server keeps no response store"),
    ("background", "background", bool,
     "background is not supported"),
    ("include", "include", bool,
     "include is not supported"),
    ("truncation", "truncation", lambda v: v == "auto",
     "truncation=auto is not supported: the prompt is always sent whole"),
    ("reasoning", "reasoning", lambda v: True,
     "reasoning is not supported: this model has no reasoning phase"),
    ("conversation", "conversation", bool,
     "conversation is not supported: this server keeps no response store"),
    ("prompt", "prompt", bool,
     "prompt templates are not supported"),
    ("max_tool_calls", "max_tool_calls", lambda v: True,
     "max_tool_calls is not supported without tools"),
    ("moderation", "moderation", bool,
     "moderation is not supported"),
    ("top_logprobs", "top_logprobs", lambda v: bool(v),
     "top_logprobs is not supported"),
    ("text", "text.format",
     lambda v: (isinstance(v, dict) and isinstance(v.get("format"), dict)
                and v["format"].get("type") not in (None, "text")),
     "structured outputs (text.format) are not supported"),
    ("text", "text.verbosity",
     lambda v: isinstance(v, dict) and v.get("verbosity") is not None,
     "text.verbosity is not supported"),
)


def rejected_responses_field(body: dict) -> dict | None:
    """First field the request asked for that this server cannot honour."""
    for field, param, requested, message in RESPONSES_UNSUPPORTED:
        if field not in body or body[field] is None:
            continue
        try:
            if not requested(body[field]):
                continue
        except (AttributeError, TypeError):
            continue        # malformed value for a rejected field: not a rejection
        return {"message": message, "type": "unsupported_parameter", "param": param}
    return None


def _content_text(content) -> str:
    """Plain text out of one input item's `content`.

    Responses names text parts `input_text` / `output_text`; chat completions names
    them `text`. All three are read here because handing this list to
    `flatten_messages` untouched would drop the Responses spellings -- it filters on
    ``type == "text"`` -- and the prompt would silently lose its body. Non-text parts
    are refused rather than dropped for the same reason.
    """
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        return str(content)
    parts: list[str] = []
    for part in content:
        if isinstance(part, str):
            parts.append(part)
            continue
        if not isinstance(part, dict):
            continue
        kind = part.get("type")
        if kind in ("input_text", "output_text", "text"):
            parts.append(str(part.get("text", "")))
        elif kind is not None:
            raise ValueError(f"input content part {kind!r} is not supported: text only")
    return "".join(parts)


def flatten_input(payload, instructions=None) -> list[dict]:
    """Normalise a Responses ``input`` into chat-completions ``messages``.

    Returns messages rather than a prompt, on purpose: `flatten_messages` then builds
    the prompt, so `/v1/responses` and `/v1/chat/completions` cannot drift apart for the
    same conversation. Keeping the two prompt builders equal is an acceptance
    criterion, so it is enforced structurally instead of by keeping two formatters in
    sync by hand.
    """
    if payload is None:
        items: list = []
    elif isinstance(payload, str):
        items = [{"role": "user", "content": payload}]
    elif isinstance(payload, list):
        items = payload
    else:
        raise ValueError("input must be a string or an array of input items")

    messages: list[dict] = []
    if instructions:
        messages.append({"role": "system", "content": str(instructions)})
    for item in items:
        if not isinstance(item, dict):
            raise ValueError("every input item must be an object")
        role = item.get("role")
        if role is None:
            # Items without a role are bookkeeping for tool calling (function calls,
            # reasoning, ...), all of which is rejected earlier. Refusing here keeps a
            # turn from vanishing out of the prompt.
            raise ValueError(
                f"input item of type {item.get('type')!r} is not supported: "
                "only message items with a role are accepted")
        # `developer` is this protocol's system role. flatten_messages maps an unknown
        # role onto "用户", which would invert the meaning of an instruction.
        role = "system" if role == "developer" else str(role)
        messages.append({"role": role, "content": _content_text(item.get("content"))})
    if not messages:
        raise ValueError("input must contain at least one message")
    return messages


def responses_usage(prompt_chars: int, tokens: int) -> dict:
    """The `usage` block, with every sub-object the response schema requires.

    The cache and reasoning counters have no counterpart in this server; they are
    reported as 0 instead of omitted because the field is required, and 0 is the
    truthful value for a server that neither caches nor reasons. Input tokens are
    counted the same way the chat endpoint counts them (prompt characters), so the two
    endpoints report comparable numbers.
    """
    return {
        "input_tokens": prompt_chars,
        "input_tokens_details": {"cached_tokens": 0, "cache_write_tokens": 0},
        "output_tokens": tokens,
        "output_tokens_details": {"reasoning_tokens": 0},
        "total_tokens": prompt_chars + tokens,
    }


def responses_message(mid: str, text: str, status: str) -> dict:
    """One `message` output item.

    `output[]` is a list because a future phase adds `function_call` items to it;
    nothing here assumes this is the only kind of item that can appear.
    """
    return {
        "id": mid, "type": "message", "role": "assistant", "status": status,
        # `annotations` and `logprobs` are required members of the output_text part.
        "content": [{"type": "output_text", "text": text,
                     "annotations": [], "logprobs": []}],
    }


def responses_status(tokens: int, max_tokens: int) -> tuple[str, str | None, str]:
    """Map "how generation ended" onto the protocol's status vocabulary.

    Returns ``(response status, incomplete_details.reason, output item status)``.

    The three agree by construction, which is the point: a stream that finishes an item
    as `completed` and then closes on an `incomplete` response is the same endpoint
    telling a client two different things about one answer. Kept as a function so both
    branches are testable without a model that happens to stop -- a base model under
    greedy decoding runs to the cap every time, so the `completed` branch would
    otherwise be code nobody ever executed.

    `tokens >= max_tokens` is the same test the chat endpoint uses for its
    `finish_reason: length`, so the two endpoints cannot disagree about truncation.
    """
    if tokens >= max_tokens:
        return "incomplete", "max_output_tokens", "incomplete"
    return "completed", None, "completed"


def responses_object(rid: str, created: float, model: str, status: str,
                     output: list, usage: dict | None, *,
                     max_output_tokens: int | None = None,
                     temperature: float | None = None, top_p: float | None = None,
                     metadata: dict | None = None,
                     incomplete_reason: str | None = None,
                     error: dict | None = None) -> dict:
    """A Response object.

    `parallel_tool_calls`, `tool_choice` and `tools` are not optional in the schema;
    they are filled with the only values this server can honestly claim.
    """
    body = {
        "id": rid, "object": "response", "created_at": created, "model": model,
        "status": status, "output": output, "usage": usage,
        "parallel_tool_calls": False, "tool_choice": "none", "tools": [],
    }
    if max_output_tokens is not None:
        body["max_output_tokens"] = max_output_tokens
    if temperature is not None:
        body["temperature"] = temperature
    if top_p is not None:
        body["top_p"] = top_p
    if metadata:
        body["metadata"] = metadata
    if incomplete_reason is not None:
        body["incomplete_details"] = {"reason": incomplete_reason}
    if error is not None:
        body["error"] = error
    return body
