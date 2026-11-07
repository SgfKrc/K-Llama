"""Two chain-level behaviours that the HTTP layer depends on being true.

First, prefill slicing: one forward call may never carry more than a stage's n_batch.
The head raises "batch too large" and the whole request dies before a single token is
generated -- which is exactly what a normal chat turns into the moment the transcript
outgrows 512 tokens.

Second, session recovery: the worker drops an idle session after a minute, and the
driver has to notice rather than write into the closed socket forever.
"""
from __future__ import annotations

import unittest

import numpy as np

import chain_client


class FakeStage:
    """Two stages worth of the surface `_walk` touches, recording what it was handed."""

    def __init__(self, kind: str, n_batch: int, log: list) -> None:
        self.stage = type("Stage", (), {"kind": kind})()
        self.n_batch = n_batch
        self.n_embd = 3
        self.log = log

    def forward_tokens(self, tokens, n_past: int):
        self.log.append(("gpu", len(tokens), n_past))
        return np.zeros((len(tokens), self.n_embd), dtype=np.float32)

    def forward_tokens_out(self, hidden, n_past: int):
        self.log.append(("tail", int(hidden.shape[0]), n_past))
        return [7] * int(hidden.shape[0])

    def forward_token(self, hidden, n_past: int) -> int:
        self.log.append(("tail", 1, n_past))
        return 7


def chain(n_batch: int = 4, tail_batch: int | None = None) -> tuple[list, list]:
    log: list = []
    return ([FakeStage("gpu", n_batch, log),
             FakeStage("tail", n_batch if tail_batch is None else tail_batch, log)], log)


class PrefillSlicing(unittest.TestCase):
    def test_a_prompt_that_fits_stays_one_call(self):
        clients, log = chain()
        token = chain_client.forward(clients, [1, 2, 3], 0, [[], []])
        self.assertEqual([("gpu", 3, 0), ("tail", 3, 0)], log)
        self.assertEqual(7, token)

    def test_a_long_prompt_is_sliced_at_n_batch(self):
        clients, log = chain()
        prompt = list(range(10))
        chain_client.forward(clients, prompt, 0, [[], []])
        gpu = [entry for entry in log if entry[0] == "gpu"]
        self.assertEqual([("gpu", 4, 0), ("gpu", 4, 4), ("gpu", 2, 8)], gpu)
        # Nothing may exceed the stage's ceiling: that is the bug this guards.
        self.assertTrue(all(n <= 4 for _, n, _ in gpu))

    def test_a_continuation_slices_from_its_own_start(self):
        clients, log = chain()
        chain_client.forward(clients, list(range(5)), 100, [[], []])
        gpu = [entry for entry in log if entry[0] == "gpu"]
        self.assertEqual([("gpu", 4, 100), ("gpu", 1, 104)], gpu)

    def test_the_smallest_ceiling_wins(self):
        clients, log = chain(n_batch=8, tail_batch=2)
        self.assertEqual(2, chain_client.prefill_block(clients))
        chain_client.forward(clients, list(range(5)), 0, [[], []])
        gpu = [entry for entry in log if entry[0] == "gpu"]
        self.assertEqual([("gpu", 2, 0), ("gpu", 2, 2), ("gpu", 1, 4)], gpu)

    def test_a_stage_that_never_announced_one_uses_the_default(self):
        clients, _log = chain()
        clients[0].n_batch = 0                     # a worker that predates the HELLO field
        self.assertEqual(4, chain_client.prefill_block(clients[1:]))
        self.assertEqual(512, chain_client.prefill_block([object()]))


class ResetReconnects(unittest.TestCase):
    """An idle-closed session has to heal on the next request, not stay dead forever.

    Measured before this existed: the first request after 95 s of silence answered 500
    ConnectionAbortedError, and so did every request after it until a restart.
    """

    def dead_then_alive(self) -> tuple[object, dict]:
        client = object.__new__(chain_client.StageClient)
        client.stage = type("Stage", (), {"index": 0, "kind": "gpu"})()
        seen = {"reset": 0, "reconnect": 0}

        def reset_once() -> None:
            seen["reset"] += 1
            if seen["reset"] == 1:
                raise ConnectionAbortedError(10053)

        client._reset_once = reset_once
        client.reconnect = lambda: seen.__setitem__("reconnect", seen["reconnect"] + 1)
        return client, seen

    def test_a_dead_socket_is_reconnected_and_the_reset_retried(self):
        client, seen = self.dead_then_alive()
        client.reset()
        self.assertEqual(2, seen["reset"])       # tried once, then retried after reconnect
        self.assertEqual(1, seen["reconnect"])

    def test_a_healthy_socket_is_never_reconnected(self):
        client = object.__new__(chain_client.StageClient)
        client.stage = type("Stage", (), {"index": 0, "kind": "gpu"})()
        client._reset_once = lambda: None
        client.reconnect = lambda: self.fail("must not reconnect a healthy session")
        client.reset()


if __name__ == "__main__":
    unittest.main()
