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
import threading
import time
from dataclasses import dataclass, field
from typing import Iterator, Sequence

import numpy as np

import config
from chain_client import DFlashDriver, forward, prefill_block, speculate


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
    cascade_mtp: int = 0,
    stats: dict | None = None,
    ctx: int | None = None,
    prefix_ids: Sequence[int] | None = None,
    n_past: int = 0,
    sampling: dict | None = None,
    stop: threading.Event | None = None,
) -> Iterator[Chunk]:
    """Tokenize ``prompt``, drive the chain, yield text as it becomes readable.

    ``prefix_ids`` / ``n_past`` are the multi-turn form: the caller hands over already
    tokenized *new* tokens and says how much of the KV is already filled, so this call
    continues an existing sequence instead of starting at position 0. Defaults (``None`` /
    ``0``) are the single-turn behaviour every existing caller already has.

    Why here and not in a second loop: this module is the decode loop for the TUI and the
    server, and `chain_client` is the one for the CLI (AGENTS §7 forbids a third). The
    multi-turn driver needs exactly this loop with a different start position, so the
    ability lives here as a parameter rather than as a copy in `checks/`.

    ``sampling`` is ``{"temperature": T, "rng": Generator}`` and is handed to
    `chain_client.speculate`, which uses rejection sampling to make the speculative stream
    distributed exactly like T-sample. ``None`` is the greedy contract (argmax-verified
    drafts). Only a temperature-only chain is supported: top_k / top_p / penalties live
    inside the shim's sampler and are never exposed as a distribution, so a caller that
    asked for them must refuse or degrade *and say so* -- this loop does not silently
    approximate them. A non-greedy caller therefore either leaves ``spec_k`` at 0 or passes
    a temperature-only sampler; passing a DFlash path with ``sampling`` raises, because the
    block drafter only produces argmax.

    ``stop`` is checked once per token, so an interactive caller can end a round without
    killing the chain. A stopped round is a *complete* round from the caller's point of
    view: it still settles ``stats`` (below), so the next round can continue from a KV that
    agrees with what the user was shown. What was already emitted stays emitted.

    What a continuation needs is reported through ``stats``, because a generator cannot return
    a value:

    * ``stats["n_past"]`` -- the position this round finished at. Updated after every step,
      not only at the end, so a consumer that stops the round (or the generator that is
      closed early) still sees the position the KV really reached.
    * ``stats["pending"]`` -- the token that was emitted but never entered the KV (the loop
      breaks after emitting, before the forward that would place it). Dropping it makes the
      next round write a different token at that position, and the KV then silently disagrees
      with the transcript by one token.
    * ``stats["kv_tail"]`` -- tokens that ARE in the KV but were never emitted. Non-empty only
      when a speculative block was cut short by the end of the round; llama.cpp will not roll
      the last position back, so these have to be replayed to keep the history aligned.
    * ``stats["stopped"]`` -- this round ended because ``stop`` was set.
    * ``stats["last_token"]`` -- the last token emitted, whether or not it reached the KV.
      A caller continuing a ChatML conversation needs it to decide whether the previous
      assistant turn already wrote its closing marker.
    * ``stats["emitted"]`` -- tokens emitted in this round.

    The decode loop is not implemented here: `chain_client` owns it (`forward`,
    `speculate`, and the DFlash driver). Going through those is what makes a server or TUI
    get the same speculative paths as the CLI -- an earlier version of this module carried
    its own plain loop, so the OpenAI endpoint silently ran without any drafting.

    Speculation is opt-in. The greedy path is argmax-verified; the sampled path uses
    `chain_client.speculate`'s rejection sampler and needs ``sampling``.

    ``ctx`` is the draft model's KV window. ``None`` means ask ``config.ctx()``, which is
    the run's only source of truth for the KV budget; a caller that already holds that
    value may pass it. This used to be a literal 1024 here, so every run whose real ctx
    was larger overflowed the draft KV past 1024 tokens, the drafter raised, and the loop
    dropped back to plain decode without printing anything: slower, never visibly wrong.
    """
    if not clients:
        raise ValueError("generate_stream needs at least one stage")
    if sampling is not None and dflash_path:
        # The block drafter answers with argmax rows only; its verifier has no sampling
        # contract. Refusing beats running a greedy block under a sampling request and
        # reporting text that is not distributed as asked.
        raise ValueError(
            "DFlash drafting is greedy-only: a sampled round must use spec_k / MTP "
            "drafting, or drop --dflash")

    tail = clients[-1]
    if prefix_ids is None:
        prompt_tokens = clients[0].tokenize(prompt)
    else:
        # Already tokenized by the caller: a continuation has to feed back token *ids* it
        # generated (including the one that is still pending), and round-tripping those
        # through decode/text/encode is not lossless for byte-level BPE.
        prompt_tokens = [int(t) for t in prefix_ids]
    if not prompt_tokens:
        return
    if stats is not None:
        # Callers used to report `len(prompt)` -- a character count -- as the prompt's
        # token usage. The real number is right here and was being thrown away.
        stats["prompt_tokens"] = len(prompt_tokens)

    assembler = Utf8Assembler()
    single_stage = len(clients) == 1
    stage_ms: list[list[float]] = [[] for _ in clients]
    # This used to be an unconditional `stats: dict = {}`, which shadowed the parameter and
    # sent every speculation counter into a throwaway local: the caller only ever received
    # `prompt_tokens` and the acceptor's numbers were silently dropped. Server usage reports
    # were built on that, so they were wrong for any run that speculated.
    if stats is None:
        stats = {}
    t0 = time.perf_counter()

    driver = None
    if not single_stage and dflash_path:
        driver = DFlashDriver(clients, dflash_path, dflash_block,
                              ctx if ctx is not None else config.ctx(),
                              stage_ms, stats, cascade_mtp=cascade_mtp)

    # Prefill. A single-stage topology walks the protocol with zero hidden states -- the
    # numbers are meaningless there, only the wire paths are real.
    round_start = n_past          # 本轮起点；块内截断时按它算出哪些位置已在 KV 里
    if single_stage:
        block = prefill_block(clients)
        token = -1
        for start in range(0, len(prompt_tokens), block):
            chunk = prompt_tokens[start:start + block]
            prefix = np.zeros((len(chunk), tail.n_embd), dtype=np.float32)
            token = tail.forward_token(prefix, n_past + start)
        queued = [token]
    else:
        queued = [forward(clients, prompt_tokens, n_past, stage_ms)]
    # `n_past` is the first free position, so it advances by what was just written -- not
    # reset to the length of this round's tokens, which would only be right at n_past == 0.
    n_past += len(prompt_tokens)

    emitted = 0
    # The last emitted token does not get a forward of its own: the loop breaks after
    # emitting, and the forward that would place it never runs. A continuation has to write it
    # back, or the next round puts a different token at that position. Exception: a token that
    # came out of a speculative block is already placed, so `pending` stays None (see below).
    pending: int | None = None
    #: Tokens a speculative block computed but that this round neither emitted nor kept:
    #: diagnostic only, a continuation generates its own. Non-empty means the round ended
    #: inside a block, i.e. the extra drafts were paid for and thrown away.
    orphans: list[int] = []
    #: Positions that really are in the KV but were never emitted (see the break below).
    kv_tail: list[int] = []
    last_token: int | None = None
    # Incremental, not only at the end: a caller that stops the round (or closes the
    # generator) still has to know where the KV actually got to, and the end-of-round
    # settlement below may never run for it.
    stats["n_past"] = n_past
    stats["emitted"] = 0
    while emitted < max_tokens:
        if stop is not None and stop.is_set():
            # One check per token: the loop body is exactly one token, so this is the
            # finest granularity that costs nothing. The token already emitted stays
            # emitted; the settlement below makes the KV agree with the transcript.
            stats["stopped"] = True
            break
        token = queued.pop(0)
        last_token = token
        yield Chunk(
            text=assembler.feed(tail.piece(token)),
            token=token,
            elapsed_ms=(time.perf_counter() - t0) * 1000.0,
            stage_ms=[seg[-1] if seg else 0.0 for seg in stage_ms],
        )
        emitted += 1
        stats["emitted"] = emitted
        stats["last_token"] = token
        stats["n_past"] = n_past
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
            queued = speculate(clients, token, n_past, stage_ms, spec_k, stats,
                               sampling=sampling)
        else:
            queued = [forward(clients, [token], n_past, stage_ms)]
        n_past += len(queued)
        stats["n_past"] = n_past

    # One settlement for every way a round can end -- eog, the token cap, or `stop`. Three
    # copies of this arithmetic is how a stopped round ends up with a KV that disagrees
    # with the text on screen; the caller has no other way to learn it.
    written = n_past - round_start - len(prompt_tokens)
    if last_token is not None:
        pending = last_token
    if emitted <= written:
        # A speculative step writes its whole verification block to the KV and rolls back
        # only the rejected tail. If the round ends on the *first* token of that block, the
        # rest of the block is already placed: the model holds positions the user never saw.
        # Trimming them back is what this needs and what llama.cpp will not do -- on a
        # hybrid model `llama_memory_seq_rm` refuses to drop the last position
        # (measured: seq_rm p0=82 over pos_max=82 -> removed=0, and the next decode then
        # dies on "pos_max X must be < n_past Y"). So the positions stay, and the round
        # reports them instead: `pending` is None (the token just emitted is already in the
        # KV, re-writing it at the next position would duplicate it) and `kv_tail` names the
        # tokens that are in play but never shown. A single-turn caller ignores both; a
        # continuation has to replay `kv_tail` or its history is short by that many tokens.
        pending = None
        kv_tail = list(queued[:max(0, written - emitted)])
    orphans = list(queued)

    if stats is not None:
        # Reported even when nothing continues this sequence, so a caller never has to guess
        # whether the number it holds is current.
        stats["n_past"] = n_past
        stats["pending"] = pending
        stats["orphans"] = orphans
        stats["kv_tail"] = kv_tail
        stats["last_token"] = last_token

    tail_text = assembler.flush()
    if tail_text:
        yield Chunk(text=tail_text, token=-1,
                    elapsed_ms=(time.perf_counter() - t0) * 1000.0)
