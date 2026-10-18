"""One stop decision for token limits, EOS and incremental text streaming."""
from __future__ import annotations
from qlh_core.contracts.execution import ExecutionError


class StopPolicy:
    def __init__(self, *, max_tokens, eos_ids=(), stops=()):
        if type(max_tokens) is not int or max_tokens < 1:
            raise ExecutionError("stop_policy_invalid")
        if any(not isinstance(s, str) for s in stops):
            raise ExecutionError("route_a_stop_configuration_invalid:non_string_stop")
        self.max_tokens = max_tokens
        self.eos_ids = frozenset(eos_ids)
        self.stops = tuple(dict.fromkeys(s for s in stops if isinstance(s, str) and s))
        self.count = 0
        self.text = ""
        self.published = ""
        self.reason = ""
        self.matched = ""

    def push(self, token, decoded):
        if self.reason:
            raise ExecutionError("stop_policy_already_finished")
        self.count += 1
        if token in self.eos_ids:
            self.reason = "eos_token"
            return self.finish()
        # A tokenizer can revise an incomplete UTF-8 suffix on the next token.
        decoded = decoded.rstrip("\ufffd")
        positions = [(decoded.find(s), s) for s in self.stops if s in decoded]
        if positions:
            index, self.matched = min(positions, key=lambda pair: pair[0])
            self.text = decoded[:index]
            self.reason = "text_stop"
        else:
            hold = max((size for stop in self.stops for size in range(1, len(stop))
                        if decoded.endswith(stop[:size])), default=0)
            self.text = decoded[:-hold] if hold else decoded
            if self.count >= self.max_tokens:
                self.reason = "max_new_tokens"
        return self._publish(self.text)

    def _publish(self, visible):
        if not visible.startswith(self.published):
            raise ExecutionError("tokenizer_stream_prefix_changed")
        delta = visible[len(self.published):]
        self.published = visible
        return delta

    def finish(self):
        return self._publish(self.text)
