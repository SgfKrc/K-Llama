"""Explicit, bounded admission for the full-attention greedy batch path."""
from __future__ import annotations
import re
from qlh_core.contracts.execution import ExecutionError


def resolve_speculative_gamma(value=None, *, environment=None):
    """One is the existing scalar path; request values override the environment."""
    if value is None:
        raw = (environment or {}).get("QLH_SPECULATIVE_GAMMA", "1")
        if not isinstance(raw, str) or not re.fullmatch(r"[1-8]", raw.strip()):
            raise ExecutionError("speculative_gamma_invalid")
        return int(raw.strip())
    if type(value) is not int or not 1 <= value <= 8:
        raise ExecutionError("speculative_gamma_invalid")
    return value


def require_speculative_plan(plan, descriptor, engine):
    """First execution gate: Qwen2 GGUF, CPU, one sequence on every owner."""
    import json
    if not plan.segments[-1].head:
        raise ExecutionError("speculative_tail_head_required")
    if (engine != "llama_cpp" or not isinstance(descriptor, dict)
            or descriptor.get("model_type") != "qwen2"
            or descriptor.get("total_layers") != plan.total_layers
            or not (descriptor.get("loaded_artifact") or descriptor.get("model_path"))):
        raise ExecutionError("speculative_model_unsupported")
    for segment in plan.segments:
        assignment = json.loads(segment.payload)
        spec = assignment["load_spec"]
        if (spec["device"] != "cpu" or spec["kv_device"] != "cpu"
                or spec["sequence_count"] != 1 or spec["kv_policy"] == "disabled"
                or spec["context_size"] != json.loads(plan.segments[0].payload)["load_spec"]["context_size"]):
            raise ExecutionError("speculative_load_policy_unsupported")
        if segment is not plan.segments[0]:
            identity = assignment["worker_prepare_identity"]["stage_model_identity"]
            if identity.get("engine") != "llama_cpp" or identity.get("format") != "gguf":
                raise ExecutionError("speculative_model_unsupported")
