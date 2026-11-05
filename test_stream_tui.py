"""Contract tests for the streaming loop, the multi-turn ledger and the TUI's sampling rules.

These run without a model. The decode loop is driven through a single-stage fake that
implements exactly the surface `stream.generate_stream` touches (tokenize / piece / is_eog /
forward_token / n_embd), so the KV bookkeeping -- the part that decides whether the next
round continues a sequence or silently writes over it -- is exercised for real.

What each group is protecting:

  * `StreamBookkeeping` -- the settlement at the end of a round. A stopped round, an eog
    round and a capped round all have to report the same three numbers (n_past, pending,
    kv_tail); three copies of that arithmetic is how a stopped round ends up with a KV that
    disagrees with what is on screen.
  * `SessionLedger` -- the invariant the TUI's multi-turn depends on: KV position p holds
    ids[p]. Only a speculative block cut short by the end of a round breaks it, and the
    ledger either trims it back or falls back to replaying the sequence.
  * `ChatML` -- the continuation renderer only renders the *new* turn. `test_contract.py`
    already guards that it lives in chat.py; here it is checked for shape.
  * `SamplingRules` -- which sampler configurations a sampled speculative round can honour.
    This is the rule all three entry points ask, so it is checked once.
"""
from __future__ import annotations

import threading
import unittest

import chat
import config
import stream
from koakuma import Session


class FakeTail:
    """A single-stage chain: no network, no model, but a real position counter."""

    def __init__(self, script: list[int], eog: int = -1) -> None:
        self.n_embd = 4
        self.script = list(script)
        self.eog = eog
        #: n_past handed to every forward, in order. The multi-turn test reads this.
        self.forwards: list[int] = []
        self.tokenize_calls: list[str] = []

    def tokenize(self, text: str) -> list[int]:
        self.tokenize_calls.append(text)
        return [1 + (ord(ch) % 40) for ch in text]

    def piece(self, token: int) -> bytes:
        # One byte per token keeps the UTF-8 assembler out of the picture.
        return b"x"

    def is_eog(self, token: int) -> bool:
        return token == self.eog

    def forward_token(self, hidden, n_past: int) -> int:
        self.forwards.append(n_past)
        return self.script.pop(0)


class StreamBookkeeping(unittest.TestCase):
    """How a round reports what it did to the KV."""

    def test_eog_round_leaves_pending(self):
        stats: dict = {}
        tail = FakeTail([10, 20, 99], eog=99)
        chunks = list(stream.generate_stream([tail], "ab", max_tokens=8, stats=stats))
        self.assertEqual([10, 20, 99], [c.token for c in chunks])
        # Prefill wrote 2 positions; two forwards wrote positions 2 and 3; the eog token
        # was emitted but never forwarded -- that is exactly `pending`.
        self.assertEqual(4, stats["n_past"])
        self.assertEqual(99, stats["pending"])
        self.assertEqual([], stats["kv_tail"])
        self.assertEqual(99, stats["last_token"])
        self.assertEqual(3, stats["emitted"])
        self.assertNotIn("stopped", stats)

    def test_capped_round_looks_like_an_eog_round(self):
        stats: dict = {}
        tail = FakeTail([10, 20, 30], eog=-1)
        list(stream.generate_stream([tail], "abcd", max_tokens=2, stats=stats))
        self.assertEqual(2, stats["emitted"])
        self.assertEqual(5, stats["n_past"])       # 4 prefill + 1 forward
        self.assertEqual(20, stats["pending"])     # the second token was never forwarded
        self.assertEqual([], stats["kv_tail"])

    def test_stop_ends_the_round_after_the_current_token(self):
        stats: dict = {}
        stop = threading.Event()
        tail = FakeTail([10, 20, 30, 40], eog=-1)
        emitted = 0
        for _chunk in stream.generate_stream([tail], "ab", max_tokens=16, stats=stats,
                                             stop=stop):
            emitted += 1
            if emitted == 2:
                stop.set()
        self.assertTrue(stats["stopped"])
        self.assertEqual(2, stats["emitted"])
        # The consumer sets `stop` after seeing the second token, but the loop only checks
        # before the next token -- so it still ran one forward, which wrote that token into
        # the KV. The round therefore settles with pending=None and a KV that is exactly the
        # ids sequence: the next round continues instead of replaying a token.
        self.assertEqual(4, stats["n_past"])
        self.assertIsNone(stats["pending"])
        self.assertEqual(20, stats["last_token"])

    def test_stop_before_the_first_token(self):
        stats: dict = {}
        stop = threading.Event()
        stop.set()
        tail = FakeTail([10, 20], eog=-1)
        chunks = list(stream.generate_stream([tail], "ab", max_tokens=8, stats=stats,
                                             stop=stop))
        self.assertEqual([], [c.token for c in chunks])
        self.assertTrue(stats["stopped"])
        self.assertEqual(0, stats["emitted"])
        self.assertIsNone(stats["pending"])
        self.assertEqual(2, stats["n_past"])       # the prefill did happen

    def test_closing_the_generator_mid_round_keeps_stats_current(self):
        """A consumer that stops iterating (e.g. a killed worker) must not lose the position."""
        stats: dict = {}
        tail = FakeTail([10, 20, 30, 40], eog=-1)
        gen = stream.generate_stream([tail], "ab", max_tokens=16, stats=stats)
        first = next(gen)
        self.assertEqual(10, first.token)
        self.assertEqual(2, stats["n_past"])       # prefill only, updated incrementally
        gen.close()
        self.assertEqual(2, stats["n_past"])

    def test_continuation_starts_at_the_reported_position(self):
        """The second round's prefill must land exactly where the first one stopped."""
        tail = FakeTail([10, 20, 99, 30, 99], eog=99)
        stats1: dict = {}
        list(stream.generate_stream([tail], "ab", max_tokens=8, stats=stats1))
        # Round 1: prefill [0,2), forwards at 2 and 3 -> n_past 4, pending 99.
        self.assertEqual([0, 2, 3], tail.forwards)
        self.assertEqual(4, stats1["n_past"])

        stats2: dict = {}
        prefix = [stats1["pending"], 7, 8]         # replay the pending token, then next turn
        list(stream.generate_stream([tail], "", prefix_ids=prefix, n_past=stats1["n_past"],
                                    max_tokens=8, stats=stats2))
        # The prefill of round 2 starts at 4 and writes 3 tokens; the next forward is at 7.
        self.assertEqual([0, 2, 3, 4, 7], tail.forwards)
        self.assertEqual(8, stats2["n_past"])      # 4 + 3 prefill + 1 forward
        self.assertEqual(99, stats2["pending"])

    def test_dflash_under_sampling_is_refused(self):
        tail = FakeTail([10])
        with self.assertRaises(ValueError):
            list(stream.generate_stream([tail], "a", dflash_path="nope.gguf",
                                        sampling={"temperature": 0.7, "rng": None}))

    def test_stats_reports_prompt_tokens_from_tokenize(self):
        stats: dict = {}
        tail = FakeTail([99], eog=99)
        list(stream.generate_stream([tail], "hello world", max_tokens=4, stats=stats))
        # The fake tokenizer is one token per character; the number comes from its length,
        # not from len(prompt) happening to be the same thing.
        self.assertEqual(11, stats["prompt_tokens"])


class FakeChain:
    """Client-side surface `Session` uses, with every side effect recorded."""

    def __init__(self, eog: int = -1) -> None:
        self.eog = eog
        self.reset_calls = 0
        self.removed: list[int] = []
        self.texts: list[str] = []
        self.rm_error = False

    def tokenize(self, text: str) -> list[int]:
        self.texts.append(text)
        return [len(text)]                 # one token per render, easy to reason about

    def is_eog(self, token: int) -> bool:
        return token == self.eog

    def reset(self) -> None:
        self.reset_calls += 1

    def seq_rm(self, n_past: int) -> None:
        if self.rm_error:
            raise RuntimeError("stage refused the rollback")
        self.removed.append(n_past)


class SessionLedger(unittest.TestCase):
    """The KV/ids invariant that makes a second question a continuation, not a reset."""

    def setUp(self) -> None:
        self.chain = FakeChain()
        self.clients = [self.chain]
        self.session = Session()

    def test_first_round_starts_at_zero_and_renders_chatml(self):
        prefix, n_past, fresh = self.session.prepare(
            self.clients, "你好", plain=False, system=None)
        self.assertEqual(0, n_past)
        self.assertEqual(fresh, prefix)
        text = self.chain.texts[0]
        # First turn renders a whole conversation: the user block and an open assistant turn.
        self.assertIn("user\n", text)
        self.assertTrue(text.endswith(chat.ASSISTANT_OPEN))
        self.assertNotIn("system\n", text)      # no system message injected

    def test_system_block_only_on_the_first_turn(self):
        self.session.prepare(self.clients, "q1", plain=False, system="你是助手")
        self.assertIn("system\n你是助手", self.chain.texts[0])
        self.session.record({"n_past": 1, "pending": 5, "last_token": 5}, [1], [5])
        self.session.prepare(self.clients, "q2", plain=False, system="你是助手")
        self.assertNotIn("system\n", self.chain.texts[1])

    def test_pending_is_replayed_as_the_first_prefix_token(self):
        # Round 1: prefill wrote 1 position, the answer's only token was never forwarded.
        self.session.record({"n_past": 1, "pending": 5, "last_token": 5}, [1], [5])
        self.assertEqual([1, 5], self.session.ids)
        self.assertEqual(1, self.session.kv_target())
        self.assertEqual(0, self.session.stray())
        prefix, n_past, fresh = self.session.prepare(
            self.clients, "q2", plain=False, system=None)
        self.assertEqual(1, n_past)
        self.assertEqual(5, prefix[0])           # the un-forwarded token goes first
        self.assertEqual(fresh, prefix[1:])
        # Round 2's ChatML delta continues the conversation instead of re-rendering it.
        self.assertIn("user\nq2", self.chain.texts[0])
        self.assertTrue(self.chain.texts[0].endswith(chat.ASSISTANT_OPEN))

    def test_closed_previous_turn_gets_no_extra_end_marker(self):
        self.chain.eog = 7
        self.session.record({"n_past": 1, "pending": 7, "last_token": 7}, [1], [7])
        self.session.prepare(self.clients, "q2", plain=False, system=None)
        # The eog token itself is the end marker, so the delta opens with a newline only.
        self.assertTrue(self.chain.texts[0].startswith("\n"))
        self.assertEqual(1, self.chain.texts[0].count("user\n"))

    def test_open_previous_turn_gets_its_end_marker(self):
        self.chain.eog = 99
        self.session.record({"n_past": 1, "pending": 7, "last_token": 7}, [1], [7])
        self.session.prepare(self.clients, "q2", plain=False, system=None)
        # Truncated answer: without the marker the next user block would interrupt it.
        self.assertTrue(self.chain.texts[0].startswith(chat.IM_END))

    def test_plain_mode_sends_no_chat_markup(self):
        self.session.prepare(self.clients, "q1", plain=True, system=None)
        self.assertEqual("q1", self.chain.texts[0])
        self.session.record({"n_past": 1, "pending": 5, "last_token": 5}, [1], [5])
        self.session.prepare(self.clients, "q2", plain=True, system=None)
        self.assertEqual("\nq2\n", self.chain.texts[1])

    def test_speculative_leftovers_force_a_replay(self):
        """A block cut short by the end of a round leaves positions the user never saw.

        The KV cannot be trimmed back on a hybrid model (measured: seq_rm p0=236 rc=-6,
        the position stays, the next decode dies on "X < Y"), so the ledger does not even
        try: it replays the sequence. Slower, and the only version that is verifiably
        consistent.
        """
        # Round 1 ended inside a speculative block: n_past is 2 positions ahead of ids.
        self.session.record({"n_past": 4, "pending": None, "last_token": 7,
                             "kv_tail": [7, 8]}, [1], [7])
        self.assertEqual([1, 7], self.session.ids)
        self.assertEqual(2, self.session.stray())
        prefix, n_past, fresh = self.session.prepare(
            self.clients, "q2", plain=False, system=None)
        self.assertEqual([], self.chain.removed)    # no optimistic rollback attempt
        self.assertEqual(1, self.chain.reset_calls)
        self.assertEqual(0, n_past)
        self.assertEqual(1, self.session.replays)
        self.assertEqual([1, 7], prefix[:2])        # the whole history is replayed
        self.assertEqual(fresh, prefix[2:])
        self.assertEqual(0, self.session.stray())   # the ledger is consistent again

    def test_dirty_session_replays_without_touching_align(self):
        self.session.record({"n_past": 1, "pending": 5, "last_token": 5}, [1], [5])
        self.session.dirty = True
        prefix, n_past, fresh = self.session.prepare(
            self.clients, "q2", plain=False, system=None)
        self.assertEqual(1, self.chain.reset_calls)
        self.assertEqual([], self.chain.removed)    # dirty means the KV cannot be trusted
        self.assertEqual(0, n_past)
        self.assertFalse(self.session.dirty)
        self.assertEqual([1, 5], prefix[:2])
        self.assertEqual(fresh, prefix[2:])

    def test_reset_forgets_everything(self):
        self.session.record({"n_past": 4, "pending": 5, "last_token": 5}, [1], [5])
        self.session.reset()
        self.assertEqual([], self.session.ids)
        self.assertEqual(0, self.session.n_past)
        self.assertIsNone(self.session.pending)
        self.assertEqual(0, self.session.rounds)


class ChatML(unittest.TestCase):
    """The continuation renderer, checked for the two things callers depend on."""

    def test_next_turn_is_additive(self):
        text = chat.next_turn("下一个问题", assistant_closed=True)
        self.assertIn(f"user\n下一个问题{chat.IM_END}", text)
        self.assertTrue(text.endswith(chat.ASSISTANT_OPEN))
        # It must NOT re-render history: one assistant header, no system block.
        self.assertEqual(1, text.count("assistant\n"))

    def test_truncated_turn_is_closed(self):
        self.assertTrue(chat.next_turn("q", assistant_closed=False).startswith(chat.IM_END))

    def test_assistant_open_is_shared_with_the_first_render(self):
        first = chat.apply_chatml([{"role": "user", "content": "q"}])
        self.assertTrue(first.endswith(chat.ASSISTANT_OPEN))


class SamplingRules(unittest.TestCase):
    """Which samplers a sampled speculative round can honour (one rule, three callers)."""

    def test_temperature_only_is_allowed(self):
        self.assertEqual("", config.sampled_spec_problem(0.7, 0, 1.0))

    def test_greedy_sampling_speculation_is_refused(self):
        self.assertIn("temperature", config.sampled_spec_problem(0.0, 0, 1.0))

    def test_top_k_is_refused(self):
        self.assertIn("top_k", config.sampled_spec_problem(0.7, 20, 1.0))

    def test_top_p_is_refused(self):
        self.assertIn("top_p", config.sampled_spec_problem(0.7, 0, 0.95))

    def test_penalties_are_refused(self):
        why = config.sampled_spec_problem(0.7, 0, 1.0, {"frequency_penalty": 0.5})
        self.assertIn("frequency_penalty", why)

    def test_server_spec_plan_degrades_visibly(self):
        import server

        chain = object.__new__(server.Chain)
        chain._spec_k, chain._dflash = 2, ""
        # Greedy request: unchanged behaviour.
        self.assertTrue(chain.spec_plan(None)["active"])
        self.assertTrue(chain.spec_plan({"temperature": 0.0})["active"])
        # Temperature-only sampling: now active too.
        self.assertTrue(chain.spec_plan({"temperature": 0.7})["active"])
        # top_k lives inside the chain: degrade, and say why.
        plan = chain.spec_plan({"temperature": 0.7, "top_k": 20})
        self.assertFalse(plan["active"])
        self.assertTrue(plan["requested"])
        self.assertIn("top_k", plan["reason"])

        chain._spec_k, chain._dflash = 0, "draft.gguf"
        plan = chain.spec_plan({"temperature": 0.7})
        self.assertFalse(plan["active"])
        self.assertEqual(config.DFLASH_NEEDS_GREEDY, plan["reason"])

        chain._spec_k, chain._dflash = 0, ""
        self.assertFalse(chain.spec_plan(None)["requested"])


class TuiWiring(unittest.TestCase):
    """The TUI's own argument surface: refusal happens before any weight is loaded."""

    def test_escape_is_bound_to_stop(self):
        import koakuma

        keys = {binding.key for binding in koakuma.KLlamaApp.BINDINGS}
        self.assertIn("escape", keys)

    def test_sampled_speculation_with_default_top_k_is_refused(self):
        import koakuma

        with self.assertRaises(SystemExit) as caught:
            koakuma.main(["--spec-k", "2", "--temperature", "0.7"])
        self.assertIn("top_k", str(caught.exception))

    def test_dflash_under_sampling_is_refused(self):
        import koakuma

        with self.assertRaises(SystemExit) as caught:
            koakuma.main(["--dflash", "draft.gguf", "--temperature", "0.7"])
        self.assertIn("贪心", str(caught.exception))


if __name__ == "__main__":
    unittest.main()
