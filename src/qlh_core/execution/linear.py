"""Sequential master relay; consumes one frozen plan through injected runtime ports."""
from __future__ import annotations
from qlh_core.contracts.deadline import request_stop_reason
from qlh_core.contracts.execution import ExecutionError, StageIdentity
from qlh_core.execution.stopping import StopPolicy
from qlh_core.contracts.token_output import tail_token


class LinearExecutor:
    def __init__(self, ports):
        self.ports = ports
        self._receipts = {}

    def accept(self, expected, receipt):
        if receipt.identity != expected:
            raise ExecutionError("execution_stage_identity_mismatch")
        previous = self._receipts.get(expected)
        if previous is not None and previous != receipt.output_digest:
            raise ExecutionError("execution_stage_result_conflict")
        self._receipts[expected] = receipt.output_digest
        return receipt.value

    def run(self, plan, record, *, max_tokens, sampling, stops=(), emit=None):
        sampling.require_greedy()
        self._check(plan, record)
        self.ports.prepare(plan, record)
        eos_ids = self.ports.eos_ids()
        if not eos_ids and not stops:
            raise ExecutionError("route_a_stop_configuration_invalid:no_eos_or_text_stop")
        policy = StopPolicy(max_tokens=max_tokens, eos_ids=eos_ids, stops=stops)
        generated = []
        steps = []
        for step in range(max_tokens):
            self._check(plan, record)
            hidden = self.ports.prefix(plan.segments[0], record, step, generated[-1] if generated else None)
            for segment in plan.segments[1:]:
                self._check(plan, record)
                identity = StageIdentity.for_segment(plan, segment, record["request_id"], record["attempt_id"], step)
                receipt = self.ports.stage(segment, identity, hidden, record)
                self._check(plan, record)
                hidden = self.accept(identity, receipt)
            token = tail_token(hidden) if plan.segments[-1].head else int(self.ports.head(hidden))
            if token not in policy.eos_ids:
                generated.append(token)
            delta = policy.push(token, self.ports.decode(generated))
            self._check(plan, record)
            if delta and emit:
                emit({"token": delta})
            steps.append({"step": step, "token": token, "mode": "prefill" if step == 0 else "decode"})
            if policy.reason:
                break
        return {"response": policy.text, "full_text": policy.text, "finish_reason": "length" if policy.reason == "max_new_tokens" else "stop",
                "metrics": {"execution_mode": "distributed_pipeline", "distributed_requested": True,
                            "distributed_used": True, "fallback": False, "config_id": plan.deployment_id,
                            "steps": steps, "tokens_generated": len(generated), "stop_reason": policy.reason,
                            "matched_stop": policy.matched, "eos_token_ids": sorted(policy.eos_ids),
                            "stop_diagnostic": "" if policy.eos_ids else "route_a_eos_unavailable_text_stop",
                            "workers_used": [s.node_id for s in plan.segments[1:]],
                            "claimed_layers": [plan.segments[1].start, plan.total_layers],
                            "layer_segments": [[s.start, s.end] for s in plan.segments]}}

    def _check(self, plan, record):
        reason = request_stop_reason(record["cancel"], record["deadline"])
        if reason:
            raise ExecutionError(reason)
        if not self.ports.current(plan):
            raise ExecutionError("execution_deployment_fenced")
