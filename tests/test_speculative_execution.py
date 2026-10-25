"""Causal three-segment speculation contracts; no native model or network."""
from __future__ import annotations

import threading

import pytest

from qlh_core.contracts.deadline import RequestDeadline
from qlh_core.contracts.execution import ExecutionError, ExecutionPlan, SamplingPolicy, Segment
from qlh_core.contracts.token_output import TokenOutputError
from qlh_core.execution.speculative import SpeculativeExecutor, draft_ngram, greedy_accept

pytestmark = pytest.mark.refactor_contract
EOS = 999


def pipeline():
    # Fixed three-stage topology, including a middle owner that must be reset.
    return ExecutionPlan("deployment", "boot", "authority", 6, (
        Segment("master", "prefix", "master:1", 0, 0, 2, True, False, "{}"),
        Segment("middle", "middle", "middle:1", 1, 2, 4, False, False, "{}"),
        Segment("tail", "tail", "tail:1", 1, 4, 6, False, True, "{}"),
    ))


def request(name="request"):
    return {"request_id": name, "attempt_id": name + ":attempt",
            "cancel": threading.Event(), "deadline": RequestDeadline.start(60)}


class CausalTarget:
    """A causal prefix trie; a wrong input switches all later predictions.

    Unlike a list-backed target iterator, this evaluates the *actual* cache
    contents after each row. The fallback depends on the entire input prefix.
    """
    def __init__(self, tokens=tuple(range(1, 20)), *, prompt=(90, 91), pieces=None):
        self.prompt = tuple(prompt)
        self.tokens = tuple(tokens)
        self.pieces = pieces or {}

    def next(self, prefix):
        if len(prefix) < len(self.prompt):
            assert tuple(prefix) == self.prompt[:len(prefix)]
            return self.prompt[len(prefix)]
        assert tuple(prefix[:len(self.prompt)]) == self.prompt
        suffix = tuple(prefix[len(self.prompt):])
        if suffix == self.tokens[:len(suffix)] and len(suffix) <= len(self.tokens):
            return self.tokens[len(suffix)] if len(suffix) < len(self.tokens) else EOS
        return 10000 + sum((i + 1) * token for i, token in enumerate(prefix)) % 100003

    def oracle(self, count):
        prefix, generated = list(self.prompt), []
        for _ in range(count):
            token = self.next(prefix)
            generated.append(token)
            if token == EOS:
                break
            prefix.append(token)
        return generated

    def decode(self, tokens):
        parts = [self.pieces.get(token, chr(64 + token) if 1 <= token <= 26 else f"[{token}]")
                 for token in tokens]
        if parts and isinstance(parts[0], bytes):
            return b"".join(parts).decode("utf-8", "replace")
        return "".join(parts)


class LoopTarget(CausalTarget):
    """Lookup-friendly prompt, with a causal change at history length seven."""
    def __init__(self):
        super().__init__(prompt=(1, 2, 1, 2))

    def next(self, prefix):
        if len(prefix) < len(self.prompt):
            return super().next(prefix)
        assert tuple(prefix[:4]) == self.prompt
        if len(prefix) == 7:
            return 3
        return 2 if prefix[-1] == 1 else 1


class StrictPorts:
    def __init__(self, target=None):
        self.target = target or CausalTarget()
        self.caches = {node: [] for node in ("master", "middle", "tail")}
        self.lock = threading.Lock()
        self.held = self.admitted = False
        self.valid = True
        self.calls, self.trace, self.stage_calls = [], [], []
        self.hooks = {}
        self.eos = (EOS,)
        self.admission_error = ""
        self.bad_replay = False
        self.failed_clear = ""
        self.corrupt = None
        self.scalar_mapping = False
        self.prompt_override = None
        self.last_step = -1

    def touch(self, name):
        self.trace.append((name, {node: list(ids) for node, ids in self.caches.items()}))
        if name in self.hooks:
            self.hooks[name](self)

    def current(self, plan):
        return self.valid

    def prepare(self, plan, record):
        assert self.lock.acquire(blocking=False), "request execution overlapped"
        self.held = True
        assert all(not ids for ids in self.caches.values()), "dirty KV reused by next request"
        self.record = record
        self.last_step = -1
        self.touch("prepare")

    def prepare_speculation(self, plan, record):
        assert self.held and record is self.record
        if self.admission_error:
            raise ExecutionError(self.admission_error)
        self.admitted = True
        self.touch("admission")

    def eos_ids(self):
        self.touch("eos")
        return self.eos

    def prompt_tokens(self):
        self.touch("prompt")
        return list(self.target.prompt) if self.prompt_override is None else self.prompt_override

    def take_step(self, kind, ids, step, **fields):
        assert self.held and self.admitted
        assert type(step) is int and step == self.last_step + 1, "reused/skipped operation step"
        assert isinstance(ids, list) and ids and all(type(token) is int for token in ids)
        self.last_step = step
        self.calls.append({"kind": kind, "ids": list(ids), "step": step, **fields})

    def execute(self, plan, ids, position):
        assert all(len(cache) == position for cache in self.caches.values()), "noncausal append"
        assert all(cache == self.caches["master"] for cache in self.caches.values())
        hidden = None
        for segment in plan.segments:
            cache = self.caches[segment.node_id]
            rows = []
            for token in ids:
                cache.append(token)
                rows.append(tuple(cache))
            # A segment receives every causal row from its predecessor.
            if hidden is not None:
                assert rows == hidden
            hidden = rows
            self.stage_calls.append((segment.node_id, position, tuple(ids)))
        return [self.target.next(prefix) for prefix in hidden]

    def scalar(self, token):
        return {"token_argmax": token} if self.scalar_mapping else token

    def forward_tokens(self, plan, record, ids, position, step, *, all_tokens=False):
        self.take_step("forward", ids, step, position=position, all_tokens=all_tokens)
        rows = self.execute(plan, ids, position)
        self.touch("prefill" if position == 0 else "verify" if all_tokens else "scalar")
        if not all_tokens:
            return self.scalar(rows[-1])
        value = {"token_argmax": rows[-1], "token_argmax_by_position": rows,
                 "positions": list(range(position, position + len(ids))),
                 "n_tokens": len(ids), "n_vocab": 200000}
        if self.corrupt:
            self.corrupt(value)
        return value

    def restore_prefix(self, plan, record, ids, step):
        self.take_step("restore", ids, step)
        for segment in plan.segments:
            if segment.node_id == self.failed_clear:
                raise ExecutionError("speculative_restore_cleanup_pending")
            self.caches[segment.node_id].clear()
            self.trace.append(("clear:" + segment.node_id, None))
        rows = self.execute(plan, ids, 0)
        self.touch("restore")
        return self.scalar(rows[-1] + (1 if self.bad_replay else 0))

    def decode(self, tokens):
        assert EOS not in tokens or EOS not in self.eos
        text = self.target.decode(tokens)
        self.touch("decode")
        return text

    def cleanup(self):
        for cache in self.caches.values():
            cache.clear()
        self.admitted = False
        if self.held:
            self.held = False
            self.lock.release()


class ScriptedDrafter:
    def __init__(self, proposals=()):
        self.proposals = list(proposals)
        self.calls = []

    def __call__(self, history, limit):
        self.calls.append((list(history), limit))
        return list(self.proposals.pop(0))[:limit] if self.proposals else []


class MatchingDrafter:
    """Test-only oracle drafter; production uses only draft_ngram."""
    def __init__(self, target):
        self.target, self.calls = target, []

    def __call__(self, history, limit):
        self.calls.append((list(history), limit))
        proposed = []
        for _ in range(limit):
            proposed.append(self.target.next(history + proposed))
        return proposed


@pytest.fixture
def ports_factory():
    instances = []
    def make(target=None):
        ports = StrictPorts(target)
        instances.append(ports)
        return ports
    yield make
    for ports in instances:
        ports.cleanup()


def run(ports, *, gamma=4, drafter=None, max_tokens=8, stops=(), record=None, emit=None):
    return SpeculativeExecutor(ports, gamma, drafter).run(
        pipeline(), request() if record is None else record, max_tokens=max_tokens,
        sampling=SamplingPolicy(), stops=stops, emit=emit)


def tokens(result):
    return [step["token"] for step in result["metrics"]["steps"]]


def test_greedy_accept_longest_prefix_and_bonus():
    assert greedy_accept([2, 3, 4], [2, 3, 4, 5]) == (3, 5)
    assert greedy_accept([2, 3, 4], [9, 3, 4, 5]) == (0, 9)
    assert greedy_accept([2, 3, 4], [2, 9, 4, 5]) == (1, 9)
    assert greedy_accept([], [5]) == (0, 5)
    with pytest.raises(ExecutionError, match="verification_length"):
        greedy_accept([1], [1])


@pytest.mark.parametrize("history,limit,expected", [
    ([], 3, []), ([1], 3, []), ([1, 2, 3], 3, []), ([1, 2, 1], 0, []),
    ([1, 5, 1, 6, 1], 1, [6]),
    ([1, 2, 3, 8, 3, 9, 1, 2, 3], 1, [8]),
    ([1, 2, 7, 1, 2, 8, 1, 2], 1, [8]),
    ([1, 2, 1, 2], 6, [1, 2, 1, 2, 1, 2]),
    ([0, 0], 3, [0, 0, 0]),
])
def test_ngram_longest_then_latest_known_successor(history, limit, expected):
    original = list(history)
    assert draft_ngram(history, limit) == expected
    assert history == original
    assert draft_ngram(history, limit) == expected


@pytest.mark.parametrize("gamma", [2, 4, 8])
def test_default_ngram_loop_has_proposals_and_real_restore(ports_factory, gamma):
    target = LoopTarget()
    ports = ports_factory(target)
    result = run(ports, gamma=gamma, max_tokens=14)
    assert tokens(result) == target.oracle(14)
    stats = result["metrics"]["speculation"]
    assert stats["drafter"] == "ngram"
    assert stats["proposed_tokens"] > 0
    assert stats["rejected_rounds"] > 0 and stats["replay_calls"] > 0
    assert any(call["kind"] == "restore" for call in ports.calls)
    assert [call["step"] for call in ports.calls] == list(range(len(ports.calls)))
    assert [node for node, _, _ in ports.stage_calls] == ["master", "middle", "tail"] * (
        len(ports.stage_calls) // 3)


def test_full_acceptance_bonus_trace_and_physical_metrics(ports_factory):
    ports = ports_factory()
    events = []
    result = run(ports, drafter=ScriptedDrafter([[2, 3, 4], [6, 7, 8]]), emit=events.append)
    assert tokens(result) == list(range(1, 9))
    assert result["response"] == "ABCDEFGH" == "".join(event["token"] for event in events)
    metrics, stats = result["metrics"], result["metrics"]["speculation"]
    assert metrics["fallback"] is False and metrics["tokens_generated"] == 8
    assert metrics["workers_used"] == ["middle", "tail"]
    assert metrics["layer_segments"] == [[0, 2], [2, 4], [4, 6]]
    assert [(step["step"], step["mode"]) for step in metrics["steps"]] == [
        (0, "prefill"), *((i, "decode") for i in range(1, 8))]
    assert stats["verification_calls"] == 3  # prefill + two verification forwards
    assert stats["rounds"] == stats["proposed_rounds"] == stats["accepted_rounds"] == 2
    assert stats["proposed_tokens"] == stats["accepted_tokens"] == stats["committed_draft_tokens"] == 5
    assert stats["committed_tokens"] == stats["committed_targets"] == 8
    assert stats["acceptance_ratio"] == stats["committed_acceptance_ratio"] == 1.0
    assert stats["replay_calls"] == stats["rejected_tokens"] == stats["replayed_input_tokens"] == 0
    assert [call["position"] for call in ports.calls] == [0, 2, 6]
    assert all(stats[field] >= 0 for field in (
        "draft_seconds", "verify_seconds", "restore_seconds", "total_seconds"))
    assert stats["total_seconds"] >= sum(stats[field] for field in (
        "draft_seconds", "verify_seconds", "restore_seconds"))


@pytest.mark.parametrize("proposal,accepted", [([88, 3, 4], 0), ([2, 88, 4], 1), ([2, 3, 88], 2)])
def test_rejection_replays_before_any_round_publication(ports_factory, proposal, accepted):
    ports = ports_factory()
    snapshots = []
    def emit(event):
        ports.touch("emit")
        snapshots.append((event["token"], len(ports.calls), list(ports.caches["tail"])))
    result = run(ports, drafter=ScriptedDrafter([proposal]), max_tokens=6, emit=emit)
    assert tokens(result) == list(range(1, 7))
    replay = next(call for call in ports.calls if call["kind"] == "restore")
    expected = list(ports.target.prompt) + list(range(1, accepted + 2))
    assert replay["ids"] == expected
    assert snapshots[1][1] == 3 and snapshots[1][2] == expected
    names = [name for name, _ in ports.trace]
    assert names.index("restore") < names.index("emit", names.index("emit") + 1)
    assert [name for name in names if name.startswith("clear:")] == [
        "clear:master", "clear:middle", "clear:tail"]
    stats = result["metrics"]["speculation"]
    assert stats["accepted_tokens"] == stats["committed_draft_tokens"] == accepted
    assert stats["rejected_tokens"] == 3 - accepted
    assert stats["acceptance_ratio"] == pytest.approx(accepted / 3)
    assert stats["replayed_input_tokens"] == len(expected)
    assert [call["step"] for call in ports.calls] == list(range(len(ports.calls)))


def test_consecutive_rejections_keep_all_caches_causal(ports_factory):
    ports = ports_factory()
    result = run(ports, drafter=ScriptedDrafter([[77, 78, 79]] * 8), max_tokens=6)
    assert tokens(result) == list(range(1, 7))
    stats = result["metrics"]["speculation"]
    assert stats["rejected_rounds"] == stats["replay_calls"] == 4
    assert stats["replayed_input_tokens"] == 3 + 4 + 5 + 6
    assert stats["verification_calls"] == 6
    assert stats["proposed_tokens"] == stats["rejected_tokens"] == 9
    assert stats["accepted_tokens"] == stats["committed_draft_tokens"] == 0
    assert all(cache == list(ports.target.prompt) + [1, 2, 3, 4, 5] for cache in ports.caches.values())


@pytest.mark.parametrize("scalar_mapping", [False, True])
def test_no_draft_is_scalar_without_restore(ports_factory, scalar_mapping):
    ports = ports_factory()
    ports.scalar_mapping = scalar_mapping
    result = run(ports, drafter=ScriptedDrafter(), max_tokens=5)
    assert tokens(result) == [1, 2, 3, 4, 5]
    assert all(call["kind"] == "forward" and not call["all_tokens"] for call in ports.calls)
    assert [call["position"] for call in ports.calls] == [0, 2, 3, 4, 5]
    stats = result["metrics"]["speculation"]
    assert stats["verification_calls"] == 5 and stats["replay_calls"] == 0
    assert stats["proposed_tokens"] == stats["acceptance_ratio"] == 0


@pytest.mark.parametrize("gamma", [2, 4, 8])
@pytest.mark.parametrize("budget", [1, 2, 3, 4, 8, 9])
def test_max_budget_reserves_correction_or_bonus(ports_factory, gamma, budget):
    ports = ports_factory()
    drafter = MatchingDrafter(ports.target)
    result = run(ports, gamma=gamma, drafter=drafter, max_tokens=budget)
    assert tokens(result) == ports.target.oracle(budget)
    assert result["finish_reason"] == "length"
    assert result["metrics"]["tokens_generated"] == budget
    assert result["metrics"]["speculation"]["committed_targets"] == budget
    for history, limit in drafter.calls:
        remaining = budget - (len(history) - len(ports.target.prompt))
        assert limit == min(gamma - 1, remaining - 1)
    assert all(len(call["ids"]) <= gamma for call in ports.calls[1:])


@pytest.mark.parametrize("sequence,proposal,expected,replays", [
    ([EOS, 7], [], [EOS], 0),
    ([1, 2, EOS, 4, 5], [2, EOS, 4], [1, 2, EOS], 0),
    ([1, 2, 3, 4, EOS], [2, 3, 4], [1, 2, 3, 4, EOS], 0),
    ([1, EOS, 4, 5], [88, 3, 4], [1, EOS], 1),
    ([1, 2, EOS, 4, 5], [2, EOS, 88], [1, 2, EOS], 1),
])
def test_eos_at_prefill_draft_correction_or_bonus(ports_factory, sequence, proposal, expected, replays):
    ports = ports_factory(CausalTarget(sequence))
    events = []
    result = run(ports, drafter=ScriptedDrafter([proposal]), max_tokens=10, emit=events.append)
    assert tokens(result) == expected
    assert result["metrics"]["stop_reason"] == "eos_token"
    assert result["metrics"]["tokens_generated"] == len(expected) - 1
    stats = result["metrics"]["speculation"]
    assert stats["replay_calls"] == replays
    assert stats["committed_tokens"] == len(expected) - 1
    assert stats["committed_targets"] == len(expected)
    assert result["response"] == "".join(event["token"] for event in events)
    if proposal == [2, EOS, 4]:
        assert stats["accepted_tokens"] == 3
        assert stats["committed_draft_tokens"] == 1  # EOS and later verified rows earn no throughput
        assert stats["acceptance_ratio"] == 1
        assert stats["committed_acceptance_ratio"] == pytest.approx(1 / 3)


@pytest.mark.parametrize("proposal,replays", [([2, 3, 4], 0), ([2, 88, 4], 1)])
def test_text_stop_inside_verified_batch_never_leaks_later_tokens(ports_factory, proposal, replays):
    ports = ports_factory(CausalTarget(pieces={1: "answer S", 2: "T", 3: "OP private", 4: "LEAK"}))
    events = []
    result = run(ports, drafter=ScriptedDrafter([proposal]), stops=("STOP",), emit=events.append)
    assert tokens(result) == [1, 2, 3]
    assert result["response"] == "answer " == "".join(event["token"] for event in events)
    assert result["metrics"]["matched_stop"] == "STOP"
    stats = result["metrics"]["speculation"]
    assert stats["replay_calls"] == replays
    assert stats["committed_draft_tokens"] == (2 if not replays else 1)
    assert stats["committed_tokens"] == stats["committed_targets"] == 3


def test_utf8_cumulative_decode_and_held_stop_prefix(ports_factory):
    target = CausalTarget([1, 2, 3, 4, EOS], pieces={
        1: b"\xe4", 2: b"\xbd", 3: b"\xa0", 4: b"!ST"})
    ports = ports_factory(target)
    events = []
    result = run(ports, drafter=MatchingDrafter(target), stops=("STOP",), emit=events.append)
    assert result["response"] == "你!" == "".join(event["token"] for event in events)
    assert "\ufffd" not in result["response"]


@pytest.mark.parametrize("cause,code", [
    ("cancel", "generation_cancelled"), ("deadline", "request_deadline_exceeded"),
    ("fence", "execution_deployment_fenced"),
])
@pytest.mark.parametrize("operation", ["before", "prepare", "admission", "eos", "prompt", "prefill",
                                       "draft", "verify", "restore", "decode", "emit"])
def test_stop_signals_fence_every_operation_and_publication(ports_factory, cause, code, operation):
    ports = ports_factory()
    record, events = request(), []
    def trip(_ports=None):
        if cause == "cancel":
            record["cancel"].set()
        elif cause == "deadline":
            record["deadline"] = RequestDeadline(0, 0, 1)
        else:
            ports.valid = False
    if operation == "before":
        trip()
    elif operation not in {"draft", "emit"}:
        ports.hooks[operation] = trip
    def drafter(history, limit):
        if operation == "draft":
            trip()
        return [88, 3, 4][:limit]
    def emit(event):
        events.append(event)
        if operation == "emit":
            trip()
    with pytest.raises(ExecutionError, match=code):
        run(ports, drafter=drafter, record=record, emit=emit)
    assert len(events) == (1 if operation in {"draft", "verify", "restore", "emit"} else 0)
    if operation in {"before", "prepare", "admission", "eos", "prompt"}:
        assert ports.calls == []
    if operation == "draft":
        assert len(ports.calls) == 1
    if operation == "verify":
        assert not any(call["kind"] == "restore" for call in ports.calls)


@pytest.mark.parametrize("fault", ["positions", "length", "missing", "scalar", "vocab"])
def test_verification_binds_rows_to_exact_input_offer(ports_factory, fault):
    ports = ports_factory()
    def corrupt(value):
        if fault == "positions":
            value["positions"] = [position + 1 for position in value["positions"]]
        elif fault == "length":
            value["n_tokens"] -= 1
            value["positions"].pop()
            value["token_argmax_by_position"].pop(0)
        elif fault == "missing":
            value.clear()
            value["token_argmax"] = 5
        elif fault == "scalar":
            value["token_argmax"] += 1
        else:
            value["n_vocab"] = 1
    ports.corrupt = corrupt
    events = []
    with pytest.raises(TokenOutputError):
        run(ports, drafter=ScriptedDrafter([[2, 3, 4]]), emit=events.append)
    assert events == [{"token": "A"}]
    assert not any(call["kind"] == "restore" for call in ports.calls)


@pytest.mark.parametrize("failure", ["mismatch", "master", "middle", "tail"])
def test_replay_failure_blocks_all_tokens_from_rejected_round(ports_factory, failure):
    ports = ports_factory()
    ports.bad_replay = failure == "mismatch"
    ports.failed_clear = "" if failure == "mismatch" else failure
    events = []
    with pytest.raises(ExecutionError, match="replay_mismatch" if failure == "mismatch" else "cleanup_pending"):
        run(ports, drafter=ScriptedDrafter([[2, 88, 4]]), emit=events.append)
    assert events == [{"token": "A"}]


@pytest.mark.parametrize("gamma", [True, False, 1, 0, 9, 2.0, "4", None])
def test_gamma_is_explicit_and_strict(ports_factory, gamma):
    ports = ports_factory()
    with pytest.raises(ExecutionError, match="gamma_invalid"):
        SpeculativeExecutor(ports, gamma)
    assert not ports.held and ports.calls == []


def test_sampling_and_admission_fail_before_target_work(ports_factory):
    ports = ports_factory()
    with pytest.raises(ExecutionError, match="sampling_unsupported"):
        SpeculativeExecutor(ports, 4).run(pipeline(), {}, max_tokens=8, sampling=SamplingPolicy(0.5))
    assert not ports.held
    ports.admission_error = "speculative_attention_state_unsupported"
    with pytest.raises(ExecutionError, match="attention_state_unsupported"):
        run(ports)
    assert ports.held and ports.calls == []


@pytest.mark.parametrize("budget", [0, -1, True, 1.0])
def test_invalid_budget_fails_before_prefill(ports_factory, budget):
    ports = ports_factory()
    with pytest.raises(ExecutionError, match="stop_policy_invalid"):
        run(ports, max_tokens=budget)
    assert ports.calls == []


def test_missing_eos_requires_text_stop(ports_factory):
    ports = ports_factory()
    ports.eos = ()
    with pytest.raises(ExecutionError, match="no_eos_or_text_stop"):
        run(ports)
    assert ports.calls == []
    ports.cleanup()
    result = run(ports, max_tokens=2, stops=("STOP",))
    assert result["metrics"]["stop_diagnostic"] == "route_a_eos_unavailable_text_stop"


@pytest.mark.parametrize("bad", [None, [1, 2, 3, 4], [True], [-1]])
def test_invalid_draft_never_reaches_target(ports_factory, bad):
    ports = ports_factory()
    with pytest.raises((ExecutionError, TokenOutputError)):
        run(ports, drafter=lambda history, limit: bad)
    assert len(ports.calls) == 1


@pytest.mark.parametrize("terminal", ["length", "text", "eos", "replay_failure"])
def test_request_owner_cleanup_restores_next_request_equivalence(ports_factory, terminal):
    target = CausalTarget([1, 2, 3, 4, EOS] if terminal == "eos" else tuple(range(1, 20)))
    ports = ports_factory(target)
    executor = SpeculativeExecutor(ports, 4, ScriptedDrafter([[2, 88, 4]]))
    ports.bad_replay = terminal == "replay_failure"
    try:
        if ports.bad_replay:
            with pytest.raises(ExecutionError, match="replay_mismatch"):
                executor.run(pipeline(), request("first"), max_tokens=8, sampling=SamplingPolicy())
        else:
            executor.run(pipeline(), request("first"), max_tokens=5, sampling=SamplingPolicy(),
                         stops=("BC",) if terminal == "text" else ())
    finally:
        # The application/request owner, not the executor, owns final cleanup.
        ports.cleanup()
    ports.bad_replay = False
    second = executor.run(pipeline(), request("second"), max_tokens=8, sampling=SamplingPolicy())
    fresh = ports_factory(target)
    expected = run(fresh, drafter=ScriptedDrafter(), max_tokens=8)
    assert tokens(second) == tokens(expected) == target.oracle(8)
    assert second["response"] == expected["response"]
    assert second["metrics"]["stop_reason"] == expected["metrics"]["stop_reason"]
