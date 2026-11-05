"""Turn OpenAI-style messages into the prompt the model expects -- through qlh_chat.dll.

Why this file is thin now: rendering belongs to llama.cpp's own `common/chat`, vendored
in `shim/chat/` and exported behind the C ABI in `shim/qlh_chat.cpp`. This module only
moves JSON across that boundary. The hand-written ChatML renderer that used to live here
was a second implementation of the template, and it had already drifted: on the 2B
checkpoint the template opens the assistant turn with `thinking`, while the hand-written
one said `thinking`, so the prompt differed by one token before the model ever saw it.
`checks/verify_equiv.py --mode prompt` is what keeps that from coming back.

The two facts that made ChatML necessary in the first place still hold, and they are why
`chat_token_env()` stays: both checkpoints carry a `tokenizer.chat_template`, and the shim
must tokenize with parse_special=true or `<|im_start|>` splits into ordinary fragments.

One handle is one conversation, and stream and non-stream must not share it: `parse()` is
stateful (it diffs against the previous full text), and every `apply()` clears that state.
`parse()` takes the **whole accumulated text**, never a delta.

Sampling parameters (temperature / top_k / top_p / seed) are deliberately absent from the
request JSON: they live in the shim's sampler chain, and `apply()` does not consume them.
What does go into the request are the *generation policy* knobs the template reads --
`enable_thinking`, `reasoning_format`, `tool_choice`, `parallel_tool_calls`, `json_schema`
and `chat_template_kwargs` -- because those decide what the prompt looks like.
"""
from __future__ import annotations

import ctypes
import json
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import config

#: GGUF metadata key holding the Jinja chat template.
TEMPLATE_KEY = "tokenizer.chat_template"

#: The four C ABI calls this module uses, named once.
_SIG = {
    "qlh_chat_new": (ctypes.c_void_p, [ctypes.c_char_p, ctypes.c_char_p, ctypes.c_size_t]),
    "qlh_chat_del": (None, [ctypes.c_void_p]),
    "qlh_chat_apply": (ctypes.c_int, [ctypes.c_void_p, ctypes.c_char_p,
                                      ctypes.POINTER(ctypes.c_char_p)]),
    "qlh_chat_parse": (ctypes.c_int, [ctypes.c_void_p, ctypes.c_char_p, ctypes.c_int,
                                      ctypes.POINTER(ctypes.c_char_p)]),
    "qlh_chat_str_free": (None, [ctypes.c_char_p]),
}


class ChatError(RuntimeError):
    """The DLL refused the template or the request; the message is the DLL's own."""


@lru_cache(maxsize=8)
def _read_template(path: str, mtime_ns: int, size: int) -> str:
    """Uncached body of `read_template`; the stat values are the cache key's identity.

    Both are in the key so a re-quantized or replaced checkpoint is re-read rather than
    silently keeping the old template. Nothing else about the file is inspected.
    """
    try:
        from gguf import GGUFReader
    except ImportError as exc:  # pragma: no cover - environment problem, not a code path
        raise ChatError("reading tokenizer.chat_template needs the `gguf` package "
                        "(pip install gguf)") from exc
    reader = GGUFReader(path, "r")
    field = reader.fields.get(TEMPLATE_KEY)
    if field is None:
        raise ChatError(f"{Path(path).name} carries no {TEMPLATE_KEY}; it is not a chat "
                        f"model, and this project only supports the Qwen family")
    raw = b"".join(bytes(part.tobytes() if hasattr(part, "tobytes") else part)
                   for part in (field.parts[i] for i in field.data))
    return raw.decode("utf-8")


def read_template(model: str | Path) -> str:
    """Read ``tokenizer.chat_template`` out of the GGUF the model lives in.

    The template is stored as an array of uint8, so it has to be reassembled from the
    reader's parts before it is a string. Read from the artifact rather than shipped
    beside the code: the template is a property of the checkpoint, and the whole point of
    the DLL is that it renders *that* checkpoint's conversation format.

    Cached by (path, mtime, size) because opening a GGUF to read one metadata string costs
    **7 s** on the 35B checkpoint, and `server._renderer` does it on every request.
    Building a handle costs 6 ms and is not cached -- a handle carries parse state.
    """
    st = Path(model).stat()
    return _read_template(str(Path(model).resolve()), st.st_mtime_ns, st.st_size)


@dataclass(frozen=True)
class Rendered:
    """What one ``apply`` decided: the prompt, and how to parse what comes back."""

    prompt: str
    generation_prompt: str
    stops: tuple[str, ...]
    thinking: bool
    format: str
    supports_thinking: bool
    #: What the template says it opens an assistant turn with. The DLL derives this from
    #: the template itself, which is the thing that decides it -- guessing it out of the
    #: last line of ``generation_prompt`` worked only because every checkpoint in use
    #: happens to agree.
    thinking_start_tag: str = ""

    @property
    def thinking_start(self) -> str:
        """The marker an assistant turn opens with (`` thinking``).

        Taken from the DLL rather than reverse-engineered from ``generation_prompt``.
        Measured: with thinking on the two agree (``<think>``); with thinking off the
        template closes an empty block itself (``<think>\n\n</think>\n\n``), so no
        single marker describes it and an assistant history built from one is lossy.
        `Conversation.add_user` then sees the prefix break and replays the whole prompt --
        slower, still correct. No entry point turns thinking off today.
        """
        return self.thinking_start_tag


class Chat:
    """One handle, one conversation. Render messages in, parse the answer out.

    A handle is not shareable between a streaming and a non-streaming path: ``parse()``
    remembers the previous full text so it can diff, and ``apply()`` resets that memory.
    """

    def __init__(self, template: str, *, dll: str | Path | None = None) -> None:
        self.dll_path = config.find_chat_dll(dll)
        self._lib = ctypes.CDLL(str(self.dll_path))
        for name, (restype, argtypes) in _SIG.items():
            fn = getattr(self._lib, name)
            fn.restype, fn.argtypes = restype, argtypes
        err = ctypes.create_string_buffer(4096)
        handle = self._lib.qlh_chat_new(template.encode("utf-8"), err, len(err))
        if not handle:
            raise ChatError(err.value.decode("utf-8", "replace") or "qlh_chat_new failed")
        self._handle: int | None = handle
        #: What the template opens an assistant turn with, for callers that have to write
        #: a *historical* assistant message back in the shape the template will render.
        #: A template is one checkpoint's, so this never changes within a handle.
        self._thinking_start: str | None = None

    @classmethod
    def for_model(cls, model: str | Path, *, dll: str | Path | None = None) -> "Chat":
        """Open a conversation for the checkpoint at ``model`` (template read from GGUF)."""
        return cls(read_template(model), dll=dll)

    # -- rendering ------------------------------------------------------------

    def render(self, messages: Sequence[Mapping[str, Any]], *, tools: Sequence | None = None,
               thinking: bool = True, reasoning_format: str | None = None,
               tool_choice: str | None = None, parallel_tool_calls: bool | None = None,
               json_schema: str | None = None,
               template_kwargs: Mapping[str, Any] | None = None,
               add_generation_prompt: bool = True) -> Rendered:
        """Render ``messages`` into the prompt for the next assistant turn.

        Every knob here is one the template reads; sampling parameters are not among them
        (see the module docstring). Consecutive same-role messages stay separate blocks --
        merging them changes the token sequence, and the model was trained on the
        separate form.
        """
        req: dict[str, Any] = {"messages": list(messages),
                               "enable_thinking": bool(thinking),
                               "add_generation_prompt": bool(add_generation_prompt)}
        if tools:
            req["tools"] = list(tools)
        if reasoning_format:
            req["reasoning_format"] = reasoning_format
        if tool_choice:
            req["tool_choice"] = tool_choice
        if parallel_tool_calls is not None:
            req["parallel_tool_calls"] = bool(parallel_tool_calls)
        if json_schema:
            req["json_schema"] = json_schema
        if template_kwargs:
            req["chat_template_kwargs"] = dict(template_kwargs)
        out = self._call("qlh_chat_apply", req)
        return Rendered(prompt=out["prompt"],
                        generation_prompt=out.get("generation_prompt", ""),
                        stops=tuple(out.get("stops") or ()),
                        thinking=bool(thinking) and bool(out.get("supports_thinking", True)),
                        format=str(out.get("format", "")),
                        supports_thinking=bool(out.get("supports_thinking", False)),
                        thinking_start_tag=str(out.get("thinking_start_tag", "")))

    def thinking_start(self) -> str:
        """The marker an assistant turn opens with, e.g. ``<think>``.

        Templates are asymmetric: the generation prompt opens the assistant turn with this
        marker, but a *historical* assistant message renders without it. A conversation
        re-rendered from history therefore diverges from the KV exactly at the assistant
        header -- which is the one place prefix reuse has to match. Cached because it is a
        property of the template, not of the request.
        """
        if self._thinking_start is None:
            probe = self.render([{"role": "user", "content": ""}],
                                add_generation_prompt=True)
            self._thinking_start = probe.thinking_start_tag
        return self._thinking_start

    def parse(self, text: str, *, partial: bool = True) -> dict:
        """Parse the **whole accumulated** answer text into a message plus the new diffs.

        ``text`` is never a delta: the handle keeps the previous full text and returns
        only what is new. Sending a delta silently drops everything before it.
        """
        return self._call("qlh_chat_parse", text, int(bool(partial)))

    # -- plumbing -------------------------------------------------------------

    def _call(self, name: str, payload: Any, *extra: Any) -> dict:
        if self._handle is None:
            raise ChatError("chat handle is closed")
        fn = getattr(self._lib, name)
        raw = (payload if isinstance(payload, str) else
               json.dumps(payload, ensure_ascii=False))
        out = ctypes.c_char_p()
        try:
            rc = fn(self._handle, raw.encode("utf-8"), *extra, ctypes.byref(out))
            body = out.value.decode("utf-8", "replace") if out.value else ""
            if rc != 0:
                try:
                    why = json.loads(body).get("error", body)
                except (ValueError, AttributeError):
                    why = body or f"{name} returned {rc}"
                raise ChatError(why)
            return json.loads(body) if body else {}
        finally:
            self._lib.qlh_chat_str_free(out)

    def close(self) -> None:
        if self._handle is not None:
            self._lib.qlh_chat_del(self._handle)
            self._handle = None

    def __enter__(self) -> "Chat":
        return self

    def __exit__(self, *_exc) -> None:
        self.close()


class Conversation:
    """One conversation's rendering state: messages, the last full prompt, the delta.

    The full render is authoritative -- the template lives in C++ now. "Only send the new
    part" is derived from it: when this turn's prompt starts with the previous turn's
    prompt followed by the text the model generated, the tail is exactly the new turn and
    goes into the chain as-is. When it does not (the template trimmed the generated text,
    dropped its thinking block, or the system prompt changed), the caller has to replay
    the whole sequence; `add_user` says which by returning ``replay``.

    Why the previous prompt is kept as *text* rather than as tokens: the chain already
    holds the generated tokens, and round-tripping them through text is not lossless for
    byte-level BPE. The text comparison is only there to decide whether the *new* tail can
    be appended -- the old tokens stay untouched.
    """

    def __init__(self, chat: Chat, *, system: str | None = None,
                 thinking: bool = True) -> None:
        self.chat = chat
        self.system = system
        self.thinking = thinking
        self.messages: list[dict] = []
        #: 上一轮的完整 prompt（含它自己的 generation_prompt），增量判定的基准。
        self.previous_full: str | None = None
        #: 上一轮 render 的结果：add_assistant 要从它的 generation_prompt 里取思考标记。
        self.rendered: Rendered | None = None

    def add_user(self, question: str, *, closed: bool = False) -> tuple[str, bool]:
        """Append a user turn; return ``(text to send, replay)``.

        ``closed`` says the previous round stopped on the eog token, so the chain already
        holds the turn boundary and the tail must not repeat it.
        """
        self.messages.append({"role": "user", "content": question})
        self.rendered = self.chat.render(self._with_system(), thinking=self.thinking)
        full = self.rendered.prompt
        if self.previous_full is None:
            self.previous_full = full
            return full, True
        if not full.startswith(self.previous_full):
            # 模板对历史 assistant 做了有损处理（trim / 丢思考），或 system 变了。
            # 整段重放是唯一与模板一致的选择：接着旧 KV 写会得到一个模型从未见过的上下文。
            self.previous_full = full
            return full, True
        tail = full[len(self.previous_full):]
        self.previous_full = full
        if closed:
            # 模板在历史 assistant 段末尾补的是「结束标记 + 换行」，而 eog token 已经在
            # KV 里了；跳过标记本身，保留它后面那个换行。
            marker = tail.split("\n", 1)[0]
            if not marker or not tail.startswith(marker + "\n"):
                return full, True
            tail = tail[len(marker):]
        return tail, False

    def add_assistant(self, generated: str) -> None:
        """Record what the model wrote, in the form the template will render back.

        The generated text is what came after ``generation_prompt``; the template renders
        a historical assistant turn as ``assistant header + trim(content)``. One leading
        space bridges the two: trim eats it, and what remains is exactly
        ``generation_prompt's marker + newline + generated``. Get this wrong and every
        turn replays, which is slow but still correct -- `checks/verify_multiturn.py` is
        what measures which path a real checkpoint takes.
        """
        if self.rendered is None:
            raise ChatError("add_assistant before add_user")
        self.messages.append({"role": "assistant",
                              "content": " " + self.rendered.thinking_start + "\n"
                                         + generated})
        # 下一轮的 prompt 必须以上一轮「prompt + 生成文本」为前缀，才算增量。
        # 生成文本是模板渲染历史 assistant 段的一部分（在 generation_prompt 之后）。
        if self.previous_full is not None:
            self.previous_full += generated

    def _with_system(self) -> list[dict]:
        if self.system is None:
            return self.messages
        return [{"role": "system", "content": self.system}, *self.messages]


def chat_token_env() -> dict[str, str]:
    """Environment the shim needs to treat the markers as special tokens."""
    return {"SA_PARSE_SPECIAL": "1"}


def merge_system(messages: Iterable[Mapping[str, Any]]) -> tuple[str | None, list[dict]]:
    """Split OpenAI messages into ``(system, turns)``, keeping system turns in order.

    The template takes a system message as a message, not as a side channel, so this is
    only a convenience for callers that receive a flat list.
    """
    system: str | None = None
    turns: list[dict] = []
    for msg in messages:
        if str(msg.get("role", "user")) == "system":
            content = msg.get("content", "")
            if isinstance(content, list):
                content = "".join(p.get("text", "") for p in content
                                  if isinstance(p, dict) and p.get("type") == "text")
            text = str(content)
            system = text if system is None else f"{system}\n{text}"
        else:
            turns.append(dict(msg))
    return system, turns
