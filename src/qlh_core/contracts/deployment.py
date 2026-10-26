"""Model control contracts; the journal domain is independent of workflow events."""
from __future__ import annotations

from typing import Any, Mapping, Protocol
import hashlib
import json

MODEL_CONTROL_DOMAIN = "model-control:v1"
OPERATION_TERMINAL = frozenset({"ready", "failed", "cancelled", "superseded"})
DEPLOYMENT_TERMINAL = frozenset({"aborted", "retired"})
DEPLOYMENT_TRANSITIONS = {
    "candidate": {"preparing", "aborting"},
    "preparing": {"prepared", "aborting"},
    "prepared": {"committing", "aborting"},
    "committing": {"active", "aborting"},
    "active": {"invalidated", "retiring"},
    "invalidated": {"retiring"},
    "retiring": {"retired"},
    "aborting": {"aborted"},
    "retired": set(), "aborted": set(),
}


class ModelControlError(ValueError):
    def __init__(self, code):
        self.code = code
        super().__init__(code)


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False)


def digest(value):
    return hashlib.sha256(canonical(value).encode("utf-8")).hexdigest()


def command_key(node_id, payload):
    return digest({"node_id": node_id, "payload": payload})


class ModelControlRepository(Protocol):
    def read(self) -> dict: ...
    def transact(self, mutation, *, domain: str = MODEL_CONTROL_DOMAIN) -> dict: ...
    def recovery_status(self) -> dict: ...


def frozen_assignments(expected):
    return {node: {key: value for key, value in item.items() if key != "phase"}
            for node, item in expected.items()}


def materialization_budget(assignment: Mapping[str, Any], materialization: str) -> int:
    """该次物化的**可用预算**（字节）。

    ★ 2026-10-26（LOCAL-EXECUTORS-01）：**预算与实测量必须同口径**。

    `required_bytes` 是求解器按**源描述符层宽**（fp16）估的；而 worker 实际加载的是
    它那份**量化 GGUF 段**。两者差异来自 `build_gguf_segment` **总是**把完整词表
    （`token_embd` + `output`）打进每一段 —— 实测 `[24,28)` 的 4 层 middle 段
    1850 MB 里，约 1375 MB 是词表，只剩 475 MB 是解码层：

        materialized = 1929171200 (1839.8 MB)   ← 含词表
        required     = 1164094976 (1110 MB)     ← 只算 4 层解码层
        capacity     = 12167577600 (11.3 GB)    ← 物理上绰绰有余

    按 `required` 判 ⇒ 误杀 `deployment_materialization_budget_exceeded`
    ⇒ 回执被拒 ⇒ **生命周期持久化失败** ⇒ 事务卡死直到超时（实测 10:41 起完全静默）。
    这是同一判据在 `worker_prepare_coordinator` 与这里的**两份实现**，改一处会漏另一处
    —— 因此抽成这个函数，两处共用。

    口径：
      * `gguf_segment`：量化段可能大于按源估算的值 ⇒ 取「估算」与「物理可用
        (capacity - reserve)」的**较大者**。仍 fail-closed —— 真的超过物理可用时照拒。
      * 其它物化：保持**严格**，`required_bytes` 就是权威
        （`test_resident_scalar_weights_plus_kv_check_remains_strict` 锚定这一点）。
    """
    required = assignment.get("required_bytes", 0) or 0
    capacity = assignment.get("capacity_bytes", 0) or 0
    reserve = assignment.get("reserve_bytes", 0) or 0
    if materialization == "gguf_segment":
        physical = max(0, int(capacity) - int(reserve)) if capacity else 0
        return max(int(required), int(physical))
    return int(required or capacity)


def require_runtime_receipt(assignment, receipt):
    """Validate runtime evidence against the complete frozen assignment."""
    from qlh_core.contracts.load import RuntimeLoadSpec
    identity = assignment.get("worker_prepare_identity", assignment.get("identity", {}))
    try:
        spec = RuntimeLoadSpec.from_dict(assignment["load_spec"])
        actual = RuntimeLoadSpec.from_dict(receipt["load_spec"])
        valid = (actual == spec and identity.get("load_spec_digest") == spec.digest
                 and receipt["layer_range"] == identity["layer_range"]
                 and receipt["artifact_sha256"] == identity["artifact_sha256"]
                 and receipt["memory_scope"] == "weight_tensors"
                 and type(receipt["materialized_bytes"]) is int and receipt["materialized_bytes"] > 0)
    except (ValueError, TypeError, KeyError):
        valid = False
    if not valid:
        raise ModelControlError("deployment_runtime_receipt_mismatch")
    budget = materialization_budget(assignment, spec.materialization)
    if spec.materialization == "safetensors_layer_offload":
        # CPU mmap owners and the largest CUDA execution unit have independent
        # budgets. The complete host residency must never be compared to VRAM.
        names = ("device_weight_bytes", "host_source_bytes", "host_conversion_bytes")
        if (not spec.device.startswith("cuda:")
                or any(type(receipt.get(name)) is not int or receipt[name] < 0 for name in names)
                or receipt["device_weight_bytes"] <= 0
                or receipt["host_source_bytes"] != receipt["materialized_bytes"]):
            raise ModelControlError("deployment_runtime_receipt_mismatch")
        host_budget = assignment.get("host_required_bytes")
        host_capacity = assignment.get("host_capacity_bytes")
        if (type(host_budget) is not int or type(host_capacity) is not int
                or not 0 < host_budget <= host_capacity):
            raise ModelControlError("deployment_host_budget_missing")
        if (receipt["host_source_bytes"] + receipt["host_conversion_bytes"] > host_budget
                or (budget and receipt["device_weight_bytes"] + spec.kv_cache_bytes > budget)):
            raise ModelControlError("deployment_materialization_budget_exceeded")
    elif budget and receipt["materialized_bytes"] + spec.kv_cache_bytes > budget:
        raise ModelControlError("deployment_materialization_budget_exceeded")
