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

from chain_client import DFlashDriver, forward, speculate


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
    spec_k: int = 0,
    dflash_path: str = "",
    dflash_block: int = 6,
) -> Iterator[Chunk]:
    """Tokenize ``prompt``, drive the chain, yield text as it becomes readable.

    The decode loop is not implemented here: `chain_client` owns it (`forward`,
    `speculate`, and the DFlash driver). Going through those is what makes a server or TUI
    get the same speculative paths as the CLI -- an earlier version of this module carried
    its own plain loop, so the OpenAI endpoint silently ran without any drafting.

    Speculation is opt-in and only valid at temperature 0. The verifier compares drafts
    against the chain's own argmax, so a sampling tail would make it accept tokens the
    sampler never produced.
    """
    if not clients:
        raise ValueError("generate_stream needs at least one stage")

    tail = clients[-1]
    prompt_tokens = clients[0].tokenize(prompt)
    if not prompt_tokens:
        return

    assembler = Utf8Assembler()
    single_stage = len(clients) == 1
    stage_ms: list[list[float]] = [[] for _ in clients]
    stats: dict = {}
    t0 = time.perf_counter()

    driver = None
    if not single_stage and dflash_path:
        driver = DFlashDriver(clients, dflash_path, dflash_block, 1024, stage_ms, stats)

    # Prefill. A single-stage topology walks the protocol with zero hidden states -- the
    # numbers are meaningless there, only the wire paths are real.
    if single_stage:
        prefix = np.zeros((len(prompt_tokens), tail.n_embd), dtype=np.float32)
        queued = [tail.forward_token(prefix, 0)]
    else:
        queued = [forward(clients, prompt_tokens, 0, stage_ms)]
    n_past = len(prompt_tokens)

    emitted = 0
    while emitted < max_tokens:
        token = queued.pop(0)
        yield Chunk(
            text=assembler.feed(tail.piece(token)),
            token=token,
            elapsed_ms=(time.perf_counter() - t0) * 1000.0,
            stage_ms=[seg[-1] if seg else 0.0 for seg in stage_ms],
        )
        emitted += 1
        if tail.is_eog(token) or emitted >= max_tokens:
            break
        if queued:
            # A speculative round settles several tokens at once; emit the rest without
            # touching the chain again.
            continue

        for seg in stage_ms:
            seg.clear()
        if single_stage:
            empty = np.zeros((1, tail.n_embd), dtype=np.float32)
            queued = [tail.forward_token(empty, n_past)]
        elif driver is not None:
            queued = driver.step(token, n_past)
        elif spec_k > 0:
            queued = speculate(clients, token, n_past, stage_ms, spec_k, stats)
        else:
            queued = [forward(clients, [token], n_past, stage_ms)]
        n_past += len(queued)

    tail_text = assembler.flush()
    if tail_text:
        yield Chunk(text=tail_text, token=-1,
                    elapsed_ms=(time.perf_counter() - t0) * 1000.0)
