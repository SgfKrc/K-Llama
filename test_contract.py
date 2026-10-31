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


if __name__ == "__main__":
    unittest.main()
