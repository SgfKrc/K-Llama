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
        return [1 + (ord(ch) % 40) for ch in text]

    def piece(self, token: int) -> bytes:
        # One byte per token keeps the UTF-8 assembler out of the picture.
        return b"x"

    def is_eog(self, token: int) -> bool:
        return token == self.eog

    def forward_token(self, hidden, n_past: int) -> int:
        self.forwards.append(n_past)
        return self.script.pop(0)

    def reset(self) -> None:
        self.reset_calls += 1

    def seq_rm(self, n_past: int) -> None:
        if self.rm_error:
            raise RuntimeError("stage refused the rollback")
        self.removed.append(n_past)


class FakeRenderer:
    """A template with the shape the Qwen ones have, without the DLL.

    Historical assistant turns render as ``header + trim(content) + end``; the generation
    prompt is ``header + thinking marker``. That is the structure the delta logic depends
    on: `chat.Conversation` records the generated text behind one leading space, the trim
    eats that space, and what is left is exactly the generation prompt plus the text.
    """

    HEAD = "<|assistant|>\n"
    THINK = "<|think|>"
    END = "<|end|>"

    def __init__(self) -> None:
        self.renders = 0

    @property
    def generation_prompt(self) -> str:
        return f"{self.HEAD}{self.THINK}\n"

    def render(self, messages, *, thinking: bool = True, **_kw):
        self.renders += 1
        parts: list[str] = []
        for msg in messages:
            role, content = msg["role"], msg.get("content", "")
            if role == "assistant":
                parts.append(f"{self.HEAD}{content.strip()}{self.END}\n")
            else:
                parts.append(f"<|{role}|>\n{content}{self.END}\n")
        generation = self.generation_prompt
        return chat.Rendered(prompt="".join(parts) + generation,
                             generation_prompt=generation, stops=(),
                             thinking=thinking, format="fake", supports_thinking=True)


class SessionLedger(unittest.TestCase):
    """The KV/ids invariant that makes a second question a continuation, not a reset."""

    def setUp(self) -> None:
        self.chain = FakeChain()
        self.clients = [self.chain]
        self.renderer = FakeRenderer()
        self.session = Session(self.renderer)

    def _first(self, question: str = "q1", system: str | None = None):
        return self.session.prepare(self.clients, question, plain=False, system=system)

    def test_first_round_starts_at_zero_and_renders_the_whole_prompt(self):
        prefix, n_past, fresh = self._first("你好")
        self.assertEqual(0, n_past)
        self.assertEqual(fresh, prefix)          # 首轮没有历史，整段 prompt 就是新增
        text = self.chain.texts[0]
        self.assertIn("<|user|>", text)
        self.assertTrue(text.endswith(self.renderer.generation_prompt))
        self.assertNotIn("<|system|>", text)     # 没有注入默认 system

    def test_system_block_only_on_the_first_turn(self):
        self._first("q1", system="你是助手")
        self.assertIn("<|system|>", self.chain.texts[0])
        self.assertIn("你是助手", self.chain.texts[0])
        fresh = self.session.ids[:]
        self.session.record({"n_past": len(fresh), "pending": 5, "last_token": 5},
                            fresh, [5], text="答")
        self.session.prepare(self.clients, "q2", plain=False, system="你是助手")
        self.assertNotIn("<|system|>", self.chain.texts[1])

    def test_pending_is_replayed_as_the_first_prefix_token(self):
        # 第 1 轮：prefill 全写进 KV，答案唯一那个 token 从没 forward 过。
        _prefix, _n, fresh = self._first("q1")
        self.session.record({"n_past": len(fresh), "pending": 5, "last_token": 5},
                            fresh, [5], text="答")
        self.assertEqual(fresh + [5], self.session.ids)
        self.assertEqual(len(fresh), self.session.kv_target())
        self.assertEqual(0, self.session.stray())

        prefix, n_past, fresh2 = self.session.prepare(
            self.clients, "q2", plain=False, system=None)
        self.assertEqual(len(fresh), n_past)
        self.assertEqual(5, prefix[0])           # 未 forward 的 token 排在最前
        self.assertEqual(fresh2, prefix[1:])
        # 第 2 轮发的是「块尾 + 新 user 块 + 生成头」，不是重渲染整段。
        self.assertIn("<|user|>", self.chain.texts[1])
        self.assertIn("q2", self.chain.texts[1])
        self.assertTrue(self.chain.texts[1].endswith(self.renderer.generation_prompt))

    def test_closed_previous_turn_gets_no_extra_end_marker(self):
        self.chain.eog = 7
        _prefix, _n, fresh = self._first("q1")
        self.session.record({"n_past": len(fresh), "pending": 7, "last_token": 7},
                            fresh, [7], text="答")
        self.session.prepare(self.clients, "q2", plain=False, system=None)
        # eog token 本身就是结束标记，增量只从一个换行开始。
        self.assertTrue(self.chain.texts[1].startswith("\n"))
        self.assertEqual(1, self.chain.texts[1].count("<|user|>"))

    def test_open_previous_turn_gets_its_end_marker(self):
        self.chain.eog = 99
        _prefix, _n, fresh = self._first("q1")
        self.session.record({"n_past": len(fresh), "pending": 7, "last_token": 7},
                            fresh, [7], text="答")
        self.session.prepare(self.clients, "q2", plain=False, system=None)
        # 被截断的回答：没有结束标记，新 user 块会直接截断它。
        self.assertTrue(self.chain.texts[1].startswith("<|end|>"))

    def test_plain_mode_sends_no_chat_markup(self):
        self.session.prepare(self.clients, "q1", plain=True, system=None)
        self.assertEqual("q1", self.chain.texts[0])
        self.session.record({"n_past": 2, "pending": 5, "last_token": 5}, [1, 2], [5])
        self.session.prepare(self.clients, "q2", plain=True, system=None)
        self.assertEqual("\nq2\n", self.chain.texts[1])

    def test_speculative_leftovers_force_a_replay(self):
        """A block cut short by the end of a round leaves positions the user never saw.

        The KV cannot be trimmed back on a hybrid model (measured: seq_rm p0=236 rc=-6,
        the position stays, the next decode dies on "X < Y"), so the ledger does not even
        try: it replays the sequence. Slower, and the only version that is verifiably
        consistent.
        """
        _prefix, _n, fresh = self._first("q1")
        # 第 1 轮停在投机块内：n_past 比 ids 多 2 个位置。
        self.session.record({"n_past": len(fresh) + 3, "pending": None,
                             "last_token": 7, "kv_tail": [7, 8]},
                            fresh, [7], text="答")
        self.assertEqual(fresh + [7], self.session.ids)
        self.assertEqual(2, self.session.stray())
        prefix, n_past, fresh2 = self.session.prepare(
            self.clients, "q2", plain=False, system=None)
        self.assertEqual([], self.chain.removed)     # 不尝试乐观回滚
        self.assertEqual(1, self.chain.reset_calls)
        self.assertEqual(0, n_past)
        self.assertEqual(1, self.session.replays)
        # 重放的是这一轮渲染出的完整 prompt（历史 + 新块），从 0 写。
        self.assertEqual(fresh2, prefix)
        self.assertIn("q2", self.chain.texts[1])
        self.assertEqual(0, self.session.stray())    # 账本重新一致

    def test_dirty_session_replays_without_touching_align(self):
        _prefix, _n, fresh = self._first("q1")
        self.session.record({"n_past": len(fresh), "pending": 5, "last_token": 5},
                            fresh, [5], text="答")
        self.session.dirty = True
        prefix, n_past, fresh2 = self.session.prepare(
            self.clients, "q2", plain=False, system=None)
        self.assertEqual(1, self.chain.reset_calls)
        self.assertEqual([], self.chain.removed)     # dirty 表示 KV 不可信
        self.assertEqual(0, n_past)
        self.assertFalse(self.session.dirty)
        self.assertEqual(fresh2, prefix)             # 整段重放

    def test_reset_forgets_everything(self):
        _prefix, _n, fresh = self._first("q1")
        self.session.record({"n_past": len(fresh), "pending": 5, "last_token": 5},
                            fresh, [5], text="答")
        self.session.reset()
        self.assertEqual([], self.session.ids)
        self.assertEqual(0, self.session.n_past)
        self.assertIsNone(self.session.pending)
        self.assertEqual(0, self.session.rounds)
        self.assertIsNone(self.session.conversation)  # /clear 之后是一条新对话


class Continuation(unittest.TestCase):
    """「只发新增」的判定：前缀成立就发尾块，不成立就整段。"""

    def setUp(self) -> None:
        self.renderer = FakeRenderer()
        self.conv = chat.Conversation(self.renderer)

    def test_first_turn_renders_the_whole_prompt(self):
        text, replay = self.conv.add_user("q")
        self.assertTrue(replay)
        self.assertTrue(text.endswith(self.renderer.generation_prompt))

    def test_second_turn_is_additive(self):
        self.conv.add_user("q1")
        self.conv.add_assistant("答")
        text, replay = self.conv.add_user("q2", closed=True)
        self.assertFalse(replay)
        # 增量是「块尾之后的换行 + 新 user 块 + 生成头」，历史一个字都不重发。
        self.assertTrue(text.startswith("\n"))
        self.assertEqual(1, text.count("<|user|>"))
        self.assertTrue(text.endswith(self.renderer.generation_prompt))

    def test_truncated_turn_keeps_its_end_marker(self):
        self.conv.add_user("q1")
        self.conv.add_assistant("答")
        text, replay = self.conv.add_user("q2", closed=False)
        self.assertFalse(replay)
        self.assertTrue(text.startswith("<|end|>"))

    def test_trimmed_generated_text_forces_a_replay(self):
        """模板 trim 掉生成文本的尾部空白时，前缀对不上，必须整段重放。"""
        self.conv.add_user("q1")
        self.conv.add_assistant("答  ")      # 尾部空白会被模板 trim 吃掉
        text, replay = self.conv.add_user("q2", closed=True)
        self.assertTrue(replay)
        self.assertEqual(1, text.count("q2"))   # 整段 prompt，不是增量块
        self.assertTrue(text.endswith(self.renderer.generation_prompt))

    def test_changed_system_prompt_forces_a_replay(self):
        self.conv.add_user("q1")
        self.conv.add_assistant("答")
        self.conv.system = "另一个系统提示"
        _text, replay = self.conv.add_user("q2", closed=True)
        self.assertTrue(replay)


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
