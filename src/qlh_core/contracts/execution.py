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

    @classmethod
    def freeze(cls, node, value, connection, *, local=False, incarnation=""):
        identity = value.get("worker_prepare_identity", value.get("identity", {}))
        spec = RuntimeLoadSpec.from_dict(value["load_spec"])
        start, end = identity.get("layer_range", [value.get("start_layer"), value.get("end_layer")])
        assignment = str(identity.get("assignment_id", value.get("assignment_id", "")))
        incarnation = str(incarnation or identity.get("node_incarnation", ""))
        if not assignment or (not local and (not incarnation or connection < 1)):
            raise ExecutionError("execution_assignment_identity_missing")
        return cls(node, assignment, incarnation, connection, int(start), int(end),
                   "embedding" in spec.components, "lm_head" in spec.components, canonical(value))


@dataclass(frozen=True)
class ExecutionPlan:
    deployment_id: str
    boot_id: str
    authority: str
    total_layers: int
    segments: tuple[Segment, ...]

    def __post_init__(self):
        if (not self.deployment_id or not self.boot_id or len(self.segments) < 2
                or len({s.node_id for s in self.segments}) != len(self.segments)):
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
