"""Streaming text generation over the layer chain.

This sits between the transport (chain_client) and whatever consumes the output
(the TUI or an OpenAI endpoint). It owns two things the numeric path does not need:

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

import codecs
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

    A byte-level BPE tokenizer splits multi-byte characters across tokens, so a single
    piece is often not decodable on its own. The stdlib already does exactly this job:
    an incremental decoder buffers the incomplete tail and releases it once the next
    fragment completes the sequence. The hand-written version that used to live here
    retried `bytes.decode` from the longest cut down, which is O(n²) in the pending size.
    """

    def __init__(self) -> None:
        self._decoder = codecs.getincrementaldecoder("utf-8")(errors="replace")

    def feed(self, data: bytes) -> str:
        return self._decoder.decode(data, final=False)

    def flush(self) -> str:
        """Emit whatever is left, replacing any truly invalid tail."""
        return self._decoder.decode(b"", final=True)


def generate_stream(
    clients: Sequence,
    prompt: str,
    *,
    max_tokens: int = 256,
    spec_k: int = 0,
    dflash_path: str = "",
    dflash_block: int = 6,
    stats: dict | None = None,
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
    if stats is not None:
        # Callers used to report `len(prompt)` -- a character count -- as the prompt's
        # token usage. The real number is right here and was being thrown away.
        stats["prompt_tokens"] = len(prompt_tokens)

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
