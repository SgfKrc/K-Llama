"""Opt-in full-attention, single-request greedy speculation over frozen ports.

``prepare`` acquires the request's execution lock; ``prepare_speculation`` must
admit full attention, single-sequence execution, all-row tail argmax and strict
KV reset/replay while that lock is held. The request owner retains responsibility
for final cleanup and lock release, including failures and stops inside a batch.

``prompt_tokens()`` returns a nonempty list of Python ints. ``forward_tokens``
executes the local prefix followed by every remote segment, binding receipts to
its request/attempt/operation step, and returns the tail output. ``all_tokens``
requires a position-bound argmax row for every input token; otherwise the result
is a scalar token or a ``token_argmax`` mapping. ``restore_prefix`` must strictly
clear *all* segments, replay the entire supplied committed input prefix, and
return its final argmax. It must fail if any cleanup/replay is unconfirmed.

Port operation steps start at zero and increase across forwards and restores.
The public ``steps`` trace instead numbers committed target tokens, as in the
linear executor. ``metrics.speculation`` counts physical forwards (including
prefill, excluding replays), entire replay input prefixes, and verifier-matched
drafts, even when a stop ends publication inside a verified batch.
``accepted_tokens`` / ``acceptance_ratio`` describe verifier matches.
``committed_draft_tokens`` and ``committed_tokens`` count only non-EOS tokens
actually passed to StopPolicy; ``committed_targets`` also includes EOS.
``committed_acceptance_ratio`` uses committed non-EOS drafts / proposals.
Throughput must use committed non-EOS tokens and application elapsed time. Timings are
elapsed seconds for draft, forward, restore, and the complete successful run.
No model is loaded by the history-only drafter, and no fallback is performed.
"""
from __future__ import annotations

from collections.abc import Mapping
from time import perf_counter

from qlh_core.contracts.execution import ExecutionError
from qlh_core.contracts.token_output import tail_token, validate_token_output
from qlh_core.execution.linear import LinearExecutor
from qlh_core.execution.stopping import StopPolicy


def draft_ngram(history, limit, *, max_ngram=4):
    """Copy successors of the longest suffix, using the latest known match.

    Only ``history`` is searched, never newly proposed tokens. Proposed tokens
    extend the query suffix for the next lookup. A match must have a successor
    inside the known history; a miss ends the draft without inventing a token.
    """
    if type(limit) is not int or limit < 0:
        raise ExecutionError("speculation_draft_limit_invalid")
    if type(max_ngram) is not int or not 1 <= max_ngram <= 8:
        raise ExecutionError("speculation_ngram_size_invalid")
    known = list(history)
    for token in known:
        tail_token(token)
    context = list(known)
    proposed = []
    for _ in range(limit):
        successor = None
        for size in range(min(max_ngram, len(context), len(known) - 1), 0, -1):
            suffix = context[-size:]
            for start in range(len(known) - size - 1, -1, -1):
                if known[start:start + size] == suffix:
                    successor = known[start + size]
                    break
            if successor is not None:
                break
        if successor is None:
            break
        proposed.append(successor)
        context.append(successor)
    return proposed


def greedy_accept(draft, target):
    """Return (matching draft count, mismatch correction or all-accepted bonus).

    ``target[i]`` is the argmax *after* batch input row i, where the batch is
    ``[last_committed_token] + draft``. There must be len(draft) + 1 rows.
    """
    if len(target) != len(draft) + 1:
        raise ExecutionError("speculation_verification_length_invalid")
    for token in (*draft, *target):
        tail_token(token)
    accepted = 0
    while accepted < len(draft) and draft[accepted] == target[accepted]:
        accepted += 1
    return accepted, target[accepted]


def _scalar_token(output):
    if isinstance(output, Mapping):
        validate_token_output(output, allow_all=False)
    return tail_token(output)


class SpeculativeExecutor(LinearExecutor):
    """Greedy target verification with a callable ``drafter(history, limit=...)``.

    ``gamma`` is the verification input width, including the previous token,
    and must be 2..8. The composition layer chooses LinearExecutor for gamma=1.
    Injected drafters return a list/tuple of at most ``limit`` Python token ints;
    the default is deterministic ngram lookup with a suffix cap of four.
    """

    def __init__(self, ports, gamma, drafter=None):
        super().__init__(ports)
        if type(gamma) is not int or not 2 <= gamma <= 8:
            raise ExecutionError("speculation_gamma_invalid")
        self.gamma = gamma
        self.drafter = draft_ngram if drafter is None else drafter
        if not callable(self.drafter):
            raise ExecutionError("speculation_drafter_invalid")

    def run(self, plan, record, *, max_tokens, sampling, stops=(), emit=None):
        started = perf_counter()
        sampling.require_greedy()
        self._check(plan, record)
        speculation = {
            "enabled": True, "gamma": self.gamma,
            "drafter": "ngram" if self.drafter is draft_ngram else getattr(
                self.drafter, "__name__", type(self.drafter).__name__),
            "rounds": 0, "proposed_rounds": 0, "accepted_rounds": 0, "rejected_rounds": 0,
            "proposed_tokens": 0, "accepted_tokens": 0, "rejected_tokens": 0,
            "committed_draft_tokens": 0, "committed_tokens": 0, "committed_targets": 0,
            "acceptance_ratio": 0.0, "committed_acceptance_ratio": 0.0,
            "verification_calls": 0, "replay_calls": 0, "replayed_input_tokens": 0,
            "draft_seconds": 0.0, "verify_seconds": 0.0, "restore_seconds": 0.0,
            "total_seconds": 0.0,
        }

        def checked(operation, *args, **kwargs):
            self._check(plan, record)
            value = operation(*args, **kwargs)
            self._check(plan, record)
            return value

        def timed(field, operation, *args, **kwargs):
            begin = perf_counter()
            value = checked(operation, *args, **kwargs)
            speculation[field] += perf_counter() - begin
            return value

        checked(self.ports.prepare, plan, record)
        checked(self.ports.prepare_speculation, plan, record)
        eos_ids = checked(self.ports.eos_ids)
        if not eos_ids and not stops:
            raise ExecutionError("route_a_stop_configuration_invalid:no_eos_or_text_stop")
        policy = checked(StopPolicy, max_tokens=max_tokens, eos_ids=eos_ids, stops=stops)
        prompt = checked(self.ports.prompt_tokens)
        if not isinstance(prompt, list) or not prompt:
            raise ExecutionError("speculation_prompt_tokens_invalid")
        prompt = list(prompt)
        for token in prompt:
            tail_token(token)
        generated, steps = [], []
        operation_step = 0

        def forward(ids, position, *, all_tokens=False):
            nonlocal operation_step
            step = operation_step
            operation_step += 1
            speculation["verification_calls"] += 1
            return timed("verify_seconds", self.ports.forward_tokens, plan, record, ids,
                         position=position, step=step, all_tokens=all_tokens)

        def restore(ids):
            nonlocal operation_step
            step = operation_step
            operation_step += 1
            speculation["replay_calls"] += 1
            speculation["replayed_input_tokens"] += len(ids)
            return timed("restore_seconds", self.ports.restore_prefix, plan, record, ids, step=step)

        def commit(token, *, draft_token=False):
            self._check(plan, record)
            next_generated = generated + ([token] if token not in policy.eos_ids else [])
            decoded = checked(self.ports.decode, next_generated)
            self._check(plan, record)
            if token not in policy.eos_ids:
                generated.append(token)
            delta = policy.push(token, decoded)
            speculation["committed_tokens"] = len(generated)
            speculation["committed_targets"] = policy.count
            if draft_token and token not in policy.eos_ids:
                speculation["committed_draft_tokens"] += 1
            index = len(steps)
            steps.append({"step": index, "token": token,
                          "mode": "prefill" if index == 0 else "decode"})
            self._check(plan, record)
            if delta and emit:
                checked(emit, {"token": delta})
            self._check(plan, record)

        commit(_scalar_token(forward(prompt, 0)))
        while not policy.reason:
            remaining = max_tokens - policy.count
            limit = min(self.gamma - 1, remaining - 1)
            draft = timed("draft_seconds", self.drafter, prompt + generated, limit=limit)
            if not isinstance(draft, (list, tuple)) or len(draft) > limit:
                raise ExecutionError("speculation_draft_invalid")
            draft = list(draft)
            for token in draft:
                tail_token(token)
            speculation["rounds"] += 1
            speculation["proposed_tokens"] += len(draft)
            position = len(prompt) + len(generated) - 1
            output = forward([generated[-1]] + draft, position, all_tokens=bool(draft))
            accepted = 0
            if draft:
                validate_token_output(output, n_tokens=len(draft) + 1,
                                      positions=range(position, position + len(draft) + 1),
                                      require_all=True)
                accepted, next_token = greedy_accept(draft, output["token_argmax_by_position"])
                speculation["proposed_rounds"] += 1
                speculation["accepted_tokens"] += accepted
                speculation["rejected_tokens"] += len(draft) - accepted
                if accepted < len(draft):
                    speculation["rejected_rounds"] += 1
                    # No token from this round may be published before replay
                    # confirms the correction from the restored causal prefix.
                    replay = restore(prompt + generated + draft[:accepted])
                    if _scalar_token(replay) != next_token:
                        raise ExecutionError("speculation_replay_mismatch")
                else:
                    speculation["accepted_rounds"] += 1
                committed = draft[:accepted] + [next_token]
            else:
                committed = [_scalar_token(output)]
            for index, token in enumerate(committed):
                commit(token, draft_token=index < accepted)
                if policy.reason:
                    break

        self._check(plan, record)
        proposed = speculation["proposed_tokens"]
        if proposed:
            speculation["acceptance_ratio"] = speculation["accepted_tokens"] / proposed
            speculation["committed_acceptance_ratio"] = speculation["committed_draft_tokens"] / proposed
        speculation["total_seconds"] = perf_counter() - started
        return {
            "response": policy.text, "full_text": policy.text,
            "finish_reason": "length" if policy.reason == "max_new_tokens" else "stop",
            "metrics": {
                "execution_mode": "distributed_pipeline", "distributed_requested": True,
                "distributed_used": True, "fallback": False, "config_id": plan.deployment_id,
                "steps": steps, "tokens_generated": len(generated), "stop_reason": policy.reason,
                "matched_stop": policy.matched, "eos_token_ids": sorted(policy.eos_ids),
                "stop_diagnostic": "" if policy.eos_ids else "route_a_eos_unavailable_text_stop",
                "workers_used": [s.node_id for s in plan.segments[1:]],
                "claimed_layers": [plan.segments[1].start, plan.total_layers],
                "layer_segments": [[s.start, s.end] for s in plan.segments],
                "speculation": speculation,
            },
        }
