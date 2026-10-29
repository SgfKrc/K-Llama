"""Streaming text generation over the layer chain.

This sits between the transport (chain_client) and whatever consumes the output
(chat_tui or an OpenAI endpoint). It owns two things the numeric path does not need:

  * UTF-8 reassembly. Byte-level BPE splits multi-byte characters across tokens, so
    a single piece is often not decodable on its own. Emitting raw pieces produces
    mojibake; buffering the incomplete tail and decoding only complete characters
    does not.
  * Stop conditions. End-of-generation is a vocabulary property, not a fixed id, so
    it has to be asked of the tail stage rather than hardcoded.

The generator yields ``Chunk`` as each fragment becomes decodable, so a TUI can
render incrementally instead of waiting for the whole sentence.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Iterator, Sequence

import numpy as np


@dataclass
class Chunk:
    text: str            # newly decoded text (may be empty while bytes accumulate)
    token: int           # the token that produced it
    elapsed_ms: float    # wall time for this token across the whole chain
    stage_ms: list[float] = field(default_factory=list)


class Utf8Assembler:
    """Turn a stream of arbitrary byte fragments into a stream of valid str.

    A partial trailing sequence is held back until the next fragment completes it.
    Without this, a Chinese character split across two tokens shows up as two
    replacement characters.
    """

    def __init__(self) -> None:
        self._pending = b""

    def feed(self, data: bytes) -> str:
        self._pending += data
        # Decode the longest well-formed prefix and keep the incomplete tail. Trying
        # cuts from long to short matters: a 3-byte character split as 2+1 must not be
        # decoded as one valid byte plus a stray continuation byte.
        for cut in range(len(self._pending), -1, -1):
            try:
                text = self._pending[:cut].decode("utf-8")
            except UnicodeDecodeError:
                continue
            self._pending = self._pending[cut:]
            return text
        # Unreachable in practice: cut=0 always decodes (to the empty string), so the
        # loop returns above. Kept so the branch cannot silently swallow bytes if the
        # range ever changes.
        return ""

    def flush(self) -> str:
        """Emit whatever is left, replacing any truly invalid tail."""
        rest, self._pending = self._pending, b""
        return rest.decode("utf-8", errors="replace")


def generate_stream(
    clients: Sequence,
    prompt: str,
    *,
    max_tokens: int = 256,
    tail_index: int | None = None,
) -> Iterator[Chunk]:
    """Tokenize ``prompt``, run the chain, yield text as it becomes readable.

    ``clients`` is the connected stage list from chain_client; index 0 must be the
    gpu stage and the last entry the tail stage.
    """
    if not clients:
        raise ValueError("generate_stream needs at least one stage")

    tail = clients[-1] if tail_index is None else clients[tail_index]
    stage_ms: list[float] = []

    # Tokenize through the head stage: every stage loads the same vocab, so any of
    # them can do it, and the head is the one that consumes ids anyway.
    prompt_tokens = clients[0].tokenize(prompt)
    if not prompt_tokens:
        return

    assembler = Utf8Assembler()
    t0 = time.perf_counter()

    # Prefill the whole prompt in one pass; decode then continues one token at a time.
    if len(clients) == 1:
        # Single-stage topology: protocol smoke test only. A tail stage expects hidden,
        # so feed zeros; the numbers are meaningless but every wire path is real.
        head = clients[0]
        prefix = np.zeros((len(prompt_tokens), head.n_embd), dtype=np.float32)
        token = head.forward_token(prefix, 0)
    else:
        hidden = clients[0].forward_tokens(prompt_tokens, 0)
        for client in clients[1:-1]:
            hidden = client.forward_hidden(hidden, 0)
        token = tail.forward_token(hidden, 0)
    n_past = len(prompt_tokens)

    for _ in range(max_tokens):
        piece = tail.piece(token)
        text = assembler.feed(piece)
        elapsed = (time.perf_counter() - t0) * 1000.0
        yield Chunk(text=text, token=token, elapsed_ms=elapsed, stage_ms=list(stage_ms))

        if tail.is_eog(token):
            break

        step_start = time.perf_counter()
        stage_ms.clear()
        for i, client in enumerate(clients):
            seg = time.perf_counter()
            if i == 0:
                hidden = client.forward_tokens([token], n_past)
            elif i == len(clients) - 1:
                token = client.forward_token(hidden, n_past)
            else:
                hidden = client.forward_hidden(hidden, n_past)
            stage_ms.append((time.perf_counter() - seg) * 1000.0)
        n_past += 1
        # Guard against a stage that returns instantly and spins the loop.
        if (time.perf_counter() - step_start) < 1e-6:
            break

    tail_text = assembler.flush()
    if tail_text:
        yield Chunk(text=tail_text, token=-1, elapsed_ms=(time.perf_counter() - t0) * 1000.0)
