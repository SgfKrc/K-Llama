"""Frozen request, placement and stage fencing contracts for linear execution."""
from __future__ import annotations
from dataclasses import dataclass, asdict
import math
from qlh_core.contracts.deployment import canonical, digest
from qlh_core.contracts.load import RuntimeLoadSpec

REQUEST_DOMAIN = "requests:v1"


class ExecutionError(RuntimeError):
    def __init__(self, code):
        self.code = code
        super().__init__(code)


@dataclass(frozen=True)
class SamplingPolicy:
    temperature: float = 0.0
    top_p: float = 1.0

    def __post_init__(self):
        if (not math.isfinite(self.temperature) or not 0 <= self.temperature <= 2
                or not math.isfinite(self.top_p) or not 0 <= self.top_p <= 1):
            raise ExecutionError("sampling_policy_invalid")

    @classmethod
    def for_request(cls, request):
        if request.model_fields_set & {"temperature", "top_p"}:
            return cls(float(request.temperature), float(request.top_p))
        return cls()

    def require_greedy(self):
        if self.temperature != 0:
            raise ExecutionError("route_a_sampling_unsupported")


@dataclass(frozen=True)
class Segment:
    node_id: str
    assignment_id: str
    incarnation: str
    connection: int
    start: int
    end: int
    embedding: bool
    head: bool
    payload: str
    # ★ 2026-10-25（LOCAL-EXECUTORS-01）：**节点内执行器**。一台物理机可以承载多个
    #   执行器（master 的 GPU 段 + 同机 CPU 段），二者 node_id **相同**（同一台机器、
    #   同一个传输身份、同一份 RAM 域）而执行器不同。
    #   ⚠️ 必须留在**末尾且带默认值**：`Segment` 已有 9 个位置参数，
    #   `tests/test_refactor_execution.py` 等按位置构造；插在中间会静默错位。
    #   缺省 "" = 该节点的默认/主执行器 = 今天的行为（唯一性判据因此逐位等价）。
    executor_id: str = ""

    @classmethod
    def freeze(cls, node, value, connection, *, local=False, incarnation="",
               executor_id=""):
        identity = value.get("worker_prepare_identity", value.get("identity", {}))
        spec = RuntimeLoadSpec.from_dict(value["load_spec"])
        start, end = identity.get("layer_range", [value.get("start_layer"), value.get("end_layer")])
        assignment = str(identity.get("assignment_id", value.get("assignment_id", "")))
        incarnation = str(incarnation or identity.get("node_incarnation", ""))
        if not assignment or (not local and (not incarnation or connection < 1)):
            raise ExecutionError("execution_assignment_identity_missing")
        return cls(node, assignment, incarnation, connection, int(start), int(end),
                   "embedding" in spec.components, "lm_head" in spec.components, canonical(value),
                   str(executor_id or identity.get("executor_id", "") or ""))

    @property
    def identity_key(self):
        """唯一性判据用的键：同一 node 上的不同执行器互不冲突。"""
        return (self.node_id, self.executor_id)


@dataclass(frozen=True)
class ExecutionPlan:
    deployment_id: str
    boot_id: str
    authority: str
    total_layers: int
    segments: tuple[Segment, ...]

    def __post_init__(self):
        # ★ 2026-10-25：唯一性判据由 `node_id` 放宽到 `(node_id, executor_id)` ——
        #   同一物理节点上的 GPU 段与 CPU 段**必须**能共存（这正是本票的目标形态）。
        #   `executor_id` 缺省 "" ⇒ 键退化为 `(node_id, "")`，与旧的"node_id 唯一"
        #   完全等价，既有 fixture 全为不同 node ⇒ 零回归。
        #   "至少两段"的语义不变（分布式接力的前提是有东西可接力）。
        if (not self.deployment_id or not self.boot_id or len(self.segments) < 2
                or len({s.identity_key for s in self.segments}) != len(self.segments)):
            raise ExecutionError("execution_requires_two_nodes")
        cursor = 0
        for segment in self.segments:
            if segment.start != cursor or not segment.start < segment.end < self.total_layers + 1:
                raise ExecutionError("execution_segment_coverage_invalid")
            if segment.end - segment.start >= self.total_layers:
                raise ExecutionError("execution_full_model_forbidden")
            cursor = segment.end
        if (cursor != self.total_layers or not self.segments[0].embedding
                or sum(s.embedding for s in self.segments) != 1
                or sum(s.head for s in self.segments) != 1
                or any(s.head for s in self.segments[1:-1])):
            raise ExecutionError("execution_component_ownership_invalid")

    @property
    def digest(self):
        return digest(asdict(self))


@dataclass(frozen=True)
class StageIdentity:
    authority: str
    deployment_id: str
    assignment_id: str
    incarnation: str
    connection: int
    request_id: str
    attempt_id: str
    step: int

    @classmethod
    def for_segment(cls, plan, segment, request_id, attempt_id, step):
        return cls(plan.authority, plan.deployment_id, segment.assignment_id, segment.incarnation,
                   segment.connection, request_id, attempt_id, step)


@dataclass(frozen=True)
class StageReceipt:
    identity: StageIdentity
    output_digest: str
    value: object
