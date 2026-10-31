"""Contract tests between the two shim clients and their workers.

Two transports reach the same shim, and each names its operations independently:

    StageClient  --TCP-->   stage_worker        (tokenize, draft_open, seq_rm, ...)
    KeepHeadUpstream --pipe--> keep_head_worker (tokens, hidden, hidden_token, ...)

Neither side is generated from the other, and until now nothing checked that they
stayed in step. An op the client sends but the worker does not answer surfaces at
runtime as ``unknown_op``, which the chain reports as "keep-head worker exited before
responding" -- about as far from the real cause as an error message gets.

These tests are deliberately textual rather than reflective: the dispatchers are
if/elif chains, so there is no table to import. Parsing the source is what actually
catches "added a method, forgot the branch", which is the mistake they exist for.
"""
from __future__ import annotations

import inspect
import re
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent

# The operations both clients expose under one name. They are the shim's own surface --
# everything the daemon can be asked to do -- so a rename on one side must show up as a
# red test rather than as a runtime failure.
CONTRACT_OPS = (
    "draft_open", "draft_n_embd_enc", "draft_inject", "draft_block",
    "draft_reset", "draft_rollback", "draft_close",
    "add_layer_inp", "layer_inp", "last_hidden",
    "mtp_draft", "mtp_reset", "seq_rm",
    "tokenize", "piece", "is_eog",
    "set_sampling", "sampling", "clear_sampling",
    "reset", "close",
)


def _read(rel: str) -> str:
    return (ROOT / rel).read_text(encoding="utf-8", errors="replace")


def _sent_ops(rel: str, *extra_patterns: str) -> set[str]:
    """Operations a client sends, from ``{"op": "x"}`` literals and extra patterns."""
    text = _read(rel)
    ops = set(re.findall(r'"op":\s*"([a-z_0-9]+)"', text))
    for pattern in extra_patterns:
        ops |= set(re.findall(pattern, text))
    return ops


def _handled_ops(rel: str) -> set[str]:
    """Operations a dispatcher answers: ``op == "x"`` / ``operation == "x"``."""
    return set(re.findall(r'\b(?:op|operation)\s*==\s*"([a-z_0-9]+)"', _read(rel)))


class TcpContract(unittest.TestCase):
    """StageClient -> stage_worker."""

    def test_every_op_the_client_sends_is_answered(self):
        sent = _sent_ops("chain_client.py")
        handled = _handled_ops("stage_worker.py")
        missing = sorted(sent - handled)
        self.assertEqual(
            [], missing,
            "chain_client sends ops the stage worker does not answer: %s" % missing)

    def test_the_baseline_forward_paths_are_covered(self):
        """`forward_tokens`/`forward_token` send no `op` at all -- they are the default
        (empty op) branch. Guard the branch exists so a refactor of the dispatcher
        cannot silently turn the hot path into an unknown_op."""
        text = _read("stage_worker.py")
        self.assertRegex(text, r"op\s*=\s*msg\.meta\.get\(\"op\"",
                         "stage worker no longer reads op out of the frame")
        self.assertIn("if op:", text,
                      "stage worker lost the empty-op (plain forward) branch")


class PipeContract(unittest.TestCase):
    """KeepHeadUpstream -> keep_head_worker."""

    def test_every_op_the_client_sends_is_answered(self):
        # Three shapes carry an op: an inline literal, a keyword argument, and the
        # op name handed to _hidden_payload by the forward helpers.
        sent = _sent_ops(
            "keep_head.py",
            r'_hidden_payload\(\s*"([a-z_0-9]+)"',
            r'_request\(\{"op":\s*"([a-z_0-9]+)"',
        )
        handled = _handled_ops("keep_head_worker.py")
        missing = sorted(sent - handled)
        self.assertEqual(
            [], missing,
            "keep_head sends ops the pipe worker does not answer: %s" % missing)


class MethodSurface(unittest.TestCase):
    """Both clients expose the shim's operations under one name."""

    def test_shared_ops_exist_on_both_clients(self):
        import chain_client
        import keep_head

        tcp = {n for n, _ in inspect.getmembers(chain_client.StageClient, inspect.isfunction)
               if not n.startswith("_")}
        pipe = {n for n, _ in inspect.getmembers(keep_head.KeepHeadUpstream, inspect.isfunction)
                if not n.startswith("_")}
        missing_tcp = [op for op in CONTRACT_OPS if op not in tcp]
        missing_pipe = [op for op in CONTRACT_OPS if op not in pipe]
        self.assertEqual([], missing_tcp, "StageClient lost: %s" % missing_tcp)
        self.assertEqual([], missing_pipe, "KeepHeadUpstream lost: %s" % missing_pipe)

    def test_no_client_gained_an_undocumented_op(self):
        """A new public method is fine, but it has to be added to CONTRACT_OPS -- that is
        the step people forget, and the one that makes the two transports drift."""
        import chain_client
        import keep_head

        known = set(CONTRACT_OPS) | {
            # transport-specific or naming variants, not shim operations
            "connect", "sock", "n_embd",
            "forward_tokens", "forward_hidden", "forward_token", "forward_tokens_out",
            "forward_tokens_to_hidden", "forward_hidden_to_hidden_or_none",
            "forward_hidden_to_hidden", "forward_hidden_to_token", "forward_hidden_to_tokens",
            "is_available", "prime", "step",
        }
        for cls, label in ((chain_client.StageClient, "StageClient"),
                           (keep_head.KeepHeadUpstream, "KeepHeadUpstream")):
            public = {n for n, _ in inspect.getmembers(cls, inspect.isfunction)
                      if not n.startswith("_")}
            unknown = sorted(public - known)
            self.assertEqual([], unknown,
                             f"{label} gained public methods not in the contract list: "
                             f"{unknown} -- add them to CONTRACT_OPS if they map to a shim op")


class SingleSourceOfTruth(unittest.TestCase):
    """One fact, one place.

    Every bug this class guards against had the same shape: a fact written down a second
    time, where the copy was correct-looking and only drifted on a path the tests never
    exercised. ``doc-style.md`` states the rule; these assertions are what makes it
    mechanical.
    """

    def test_only_chat_renders_chatml(self):
        """A second prompt renderer is how the server ended up sending plain text.

        Qwen answered "是 是 是 ..." because `flatten_messages` had its own idea of a
        prompt while `chat.apply_chatml` had the right one. Two ways to write a second
        renderer: spell the markers out, or reuse chat.IM_START. Both are caught;
        prose that merely mentions the marker is not.
        """
        literal = re.compile(r'''["']\s*<\|im_start\|>''')
        offenders = []
        for path in sorted(ROOT.glob("*.py")):
            if path.name in {"chat.py", "test_contract.py"}:
                continue
            for i, line in enumerate(path.read_text(encoding="utf-8", errors="replace")
                                     .splitlines(), 1):
                code = line.split("#", 1)[0]
                if literal.search(code) or "IM_START" in code:
                    offenders.append(f"{path.name}:{i}: {line.strip()}")
        self.assertEqual([], offenders,
                         "prompt rendering belongs to chat.apply_chatml; found a second "
                         "implementation: %s" % offenders)

    def test_needs_logits_has_one_implementation(self):
        """`SA_NEEDS_LOGITS` is *computed* in config.needs_logits() and nowhere else.

        Reading it (the shim-facing worker takes it out of the environment) is fine;
        only assignments count as a second place where the rule lives.
        """
        assign = re.compile(r'SA_NEEDS_LOGITS"?\]\s*=\s*["\d]')
        offenders = []
        for path in sorted(ROOT.glob("*.py")):
            if path.name in {"config.py", "test_contract.py"}:
                continue
            for i, line in enumerate(path.read_text(encoding="utf-8", errors="replace")
                                     .splitlines(), 1):
                code = line.split("#", 1)[0]
                if not assign.search(code) or "config.needs_logits(" in code:
                    continue
                offenders.append(f"{path.name}:{i}: {line.strip()}")
        self.assertEqual([], offenders,
                         "spell the rule once in config.needs_logits(); found it inline: %s"
                         % offenders)

    def test_prompt_tokens_is_not_a_character_count(self):
        """`len(prompt)` is characters. The token count comes from tokenize()."""
        offenders = []
        for name in ("server.py", "proto_responses.py"):
            for i, line in enumerate(_read(name).splitlines(), 1):
                code = line.split("#", 1)[0]
                if "len(prompt)" in code:
                    offenders.append(f"{name}:{i}: {line.strip()}")
        self.assertEqual([], offenders,
                         "prompt_tokens must come from generate_stream's stats, not len(): %s"
                         % offenders)

    def test_server_has_one_generation_loop(self):
        """Four response shapes, one drain. A fifth hand-rolled loop is how `usage`
        started disagreeing between stream and non-stream."""
        text = _read("server.py")
        count = text.count("for chunk in self.generate(")
        self.assertEqual(
            1, count,
            "server.py has %d generation loops; route new shapes through Chain.drain()" % count)


if __name__ == "__main__":
    unittest.main()
