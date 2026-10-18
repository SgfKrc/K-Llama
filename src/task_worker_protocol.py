"""Versioned task-worker messages with strict validation and no transport."""

from __future__ import annotations

import base64
import hashlib
import json
import math
import re
from dataclasses import dataclass
from typing import Any, Mapping

from task_worker_capabilities import validate_worker_engine_capabilities

try:
    from .execution_provider import canonical_model_engine
except ImportError:  # pragma: no cover - top-level import in worker bundles
    from execution_provider import canonical_model_engine


PROTOCOL_NAME = "qlh.task_worker"
PROTOCOL_VERSION = 3
MIN_PROTOCOL_VERSION = 1
#: ★ 2026-09-20：v3 引入**层段**（`layer_forward`）。v1/v2 只支持整模型
#: （`full_inference`）与聚合（`aggregate`），因此**不具备**参与层流水线的能力。
#: 加版本而不改旧版：v1/v2 客户端仍可协商成功，只是在 `layer_forward` 上会被
#: `unsupported_stage_type` 拒绝（fail-closed，不做静默降级）。
MAX_PROTOCOL_VERSION = 3
MAX_MESSAGE_BYTES = 8 * 1024 * 1024
# Match SocketTaskWorkerTransport: outer metadata + body length + raw bytes;
# the first metadata length prefix is excluded. Legacy JSON remains 8 MiB inside.
MAX_BINARY_FRAME_BYTES = MAX_MESSAGE_BYTES + 1024
FULL_WORKER_KINDS = frozenset({"pc_full_worker", "android_full_worker"})

#: ★ 2026-10-07（DIST-NEXT-2）：层段 hidden 的 **wire 预算**。
#:
#: 层段的 `root_input.hidden_f32` 与中间段的 `output.hidden_out_f32` 都是 raw 数据
#: 经 base64 承载，而单条消息的上限是 `MAX_MESSAGE_BYTES`。**必须在 dispatch 前**按
#: `n_tokens * n_embd * dtype` 估算，否则合法的长 prefill / 大 `n_embd` 会先通过准入、
#: 在执行完成后才触发 `message_too_large`（过晚的契约发现，表现为 stage 超时/回退）。
_BASE64_EXPANSION = 4 / 3
HIDDEN_DTYPE_BYTES = {"float32": 4, "float16": 2}
#: 帧内除 hidden 之外的余量：JSON 外壳、`layer_range`/`hidden_spec`、
#: `seq_ids`/`positions`（n_tokens 量级）与 metadata。
STAGE_FRAME_RESERVE_BYTES = 256 * 1024

#: ★ 2026-10-07（DIST-NEXT-2b）：大 payload 的**有序分片**契约。
#:
#: 超单帧预算的输入（长 prefill / 大 `n_embd`）不再只能被拒：coordinator 先发
#: `chunk_count` 条 `stage_chunk`（每条带 attempt 身份与自身摘要），再发引用它们的
#: `stage_offer`；worker 收齐并校验后装配。三个上限都是**硬上限**：分片不能用来绕过
#: 帧预算（总数 × 单片上限 = 装配上限）。
MAX_STAGE_CHUNKS = 64
STAGE_CHUNK_BYTES = 1 * 1024 * 1024
MAX_STAGE_PAYLOAD_BYTES = MAX_STAGE_CHUNKS * STAGE_CHUNK_BYTES

# Runtime profiles are part of the release capability contract.  Keep this
# dependency-free so protocol validation can run in the slim worker process.
RUNTIME_PROFILES = ("llama_cpp_only", "torch_cpu", "torch_cuda")
RUNTIME_PROFILE_UNSPECIFIED = "unspecified"

MESSAGE_TYPES = frozenset({
    "hello",
    "hello_ack",
    "stage_offer",
    "stage_accept",
    "lease_renew",
    "stage_result",
    "stage_error",
    "stage_cancel",
    "stage_cancelled",
    #: ★ 2026-10-07（DIST-NEXT-2b）：大 payload 的有序分片（v3 起）。
    "stage_chunk",
    #: ★ 2026-10-11（三机实测）：回程分片（worker → master）。见 `_PAYLOAD_FIELDS`
    #: 里同名字段的说明；只对声明 `stage_chunked_output` 的对端使用。
    "stage_result_chunk",
})

_MESSAGE_ID = re.compile(r"^msg_[A-Za-z0-9_-]{8,96}$")
_WORKFLOW_ID = re.compile(r"^wf_[A-Za-z0-9_-]{8,96}$")
_ATTEMPT_ID = re.compile(r"^att_[A-Za-z0-9_-]{8,96}$")
_LEASE_ID = re.compile(r"^lease_[A-Za-z0-9_-]{8,96}$")
_SAFE_ID = re.compile(r"^[A-Za-z0-9_.:-]{1,128}$")
_SAFE_CODE = re.compile(r"^[a-z][a-z0-9_]{0,63}$")
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_LAYER_SEGMENT_MODES = frozenset({"head", "middle", "tail"})

# 受支持的推理引擎标识（与 execution_provider.ModelIdentity 的校验集保持一致）:
#   pytorch / llama_cpp   本地引擎
#   island                TP 孤岛（路线 A，指纹为端点摘要）
#   external_api          外部推理服务（路线 B，指纹为外部端点摘要）
_SUPPORTED_ENGINES = frozenset({
    "pytorch", "llama_cpp", "island", "external_api",
})
_TEXT_STAGE_TYPES = frozenset({"full_inference", "aggregate", "layer_forward"})

#: ★ 2026-09-20：**层段**（v3）要求的 `stage_offer` 附加字段。
#: 层段节点用 llama.cpp 承一段层：加载**裁层 GGUF**，接收上游 hidden 并注入
#: （`llama_batch.embd`），只算自己负责的层区间。字段设计对齐
#: `src/pipeline_node_contract.py` 的 `PipelineNode(Layer_range/handoff_at)` 与
#: `src/relay_contract.py` 的判据（per-token argmax，不认 bitwise）。
_LAYER_FORWARD_OFFER_FIELDS = {
    #: 本节点负责的**源模型**层区间 [start, end)（闭开）。必须与裁层 GGUF 的
    #: `block_count` 自洽，由接收方核对，防止「声明与实际加载的工件不符」。
    "layer_range",
    #: 上游交接点：hidden 从源模型第 `handoff_at` 层之后跨边界。与
    #: `PipelineNode.handoff_at` 同义，供 `cross_framework` 节点使用。
    "handoff_at",
    #: 上游 hidden 的 wire 摘要（算法 + 值），用于对账与防串话。
    "hidden_sha256",
    #: hidden 的形状：`{"n_tokens": int, "n_embd": int, "dtype": "float32"}`。
    "hidden_spec",
}

#: ★ 2026-09-23：层段 offer 的**可选**字段（出现才允许；缺省 = 旧行为）。
#: * `middle_channel` —— 中间段取 hidden 的通道，允许值见 `_LAYER_FORWARD_MIDDLE_CHANNELS`，
#:   与 Android JNI 能力上报的同名键**同值**：`keep_head_layer_out` = 末层输出
#:   （`output_norm` **之前**，层段接力所需的形态）；`extract_hidden` = `output_norm(H)`（旧默认）。
#: ⚠️ **不能**并入 `_LAYER_FORWARD_OFFER_FIELDS`：那里的字段集合是**精确**校验（缺失即
#: `field_mismatch`）⇒ 会把可选字段变成必填、破坏既有对端（实测踩到）。可选字段在
#: `_validate_payload` 里按「payload 是否真的出现」动态放宽。
_LAYER_FORWARD_OPTIONAL_FIELDS = {"middle_channel", "seq_ids", "positions", "prepare_identity"}

# Request deadlines are request-scoped and must not be encoded as worker
# ownership leases. The field is optional so peers without an end-to-end
# deadline continue to emit the existing stage_offer shape.
_STAGE_OFFER_OPTIONAL_FIELDS = {"request_deadline_ms"}

#: `middle_channel` 的允许值（协议两侧必须同集合）。
#: * `extract_hidden` —— `llama_get_embeddings_ith` 通道，返回 `output_norm(H)`（旧默认）；
#: * `keep_head_layer_out` —— `layer_inp` 的 `lid == n_layer` 槽位，返回**末层输出**。
_LAYER_FORWARD_MIDDLE_CHANNELS = {"extract_hidden", "keep_head_layer_out"}

#: ★ 2026-10-07（DIST-NEXT-6）：层段工件诊断的稳定 error code（与 Android
#: `LayerArtifactDiagnostic` 同集合）。能力广告只带**不可用**项，因此不含"可用"。
_LAYER_ARTIFACT_DIAGNOSTIC_CODES = frozenset({
    "manifest_unreadable",
    "manifest_invalid",
    "source_digest_mismatch",
    "artifact_missing",
    "artifact_digest_mismatch",
    "fallback_storage_unavailable",
    "artifact_ambiguous",
})
_LAYER_ARTIFACT_DIAGNOSTIC_MAX = 32

#: ★ 2026-10-07（DIST-NEXT-1）：`stage_cancelled` 的**可选**执行状态。
#: 取消合同把「已取消」拆成两个可分别取证的事实：ACK 先到，执行可能仍在
#: in-flight（native 层段前向不可被一次调用打断）。取值：
#: * `execution_stopped` —— 该 attempt 的执行已确实停止（终态，可释放）；
#: * `execution_in_flight` —— 已确认收到 `stage_cancel`，执行尚未停止。
#: **缺省 = 旧对端**（未携带状态），按 `unknown` 记录，不改变既有语义。
#: 与 `_LAYER_FORWARD_OPTIONAL_FIELDS` 同一机制：只在 payload 真出现时放宽精确字段校验，
#: 避免把可选字段变成必填而破坏既有对端。
_STAGE_CANCELLED_OPTIONAL_FIELDS = {"execution_state"}
_STAGE_CANCELLED_EXECUTION_STATES = frozenset({
    "execution_stopped", "execution_in_flight",
})

#: 层段 `stage_result` 的结果字段：**放在 `output` 或 `metadata` 对象内**，不扩顶层。
#: 原因：`stage_result` 的 payload 里没有 `stage_type`，无法按类型做动态字段校验；
#: 而 `output` / `metadata` 本就是自由对象（仅校验类型与摘要一致性），可安全承载。
_LAYER_FORWARD_RESULT_FIELDS = {
    "hidden_out_sha256",
    "token_argmax",
}

_ENVELOPE_FIELDS = {
    "protocol", "version", "message_type", "message_id", "sent_at_ms",
    "payload",
}
_IDENTITY_FIELDS = {
    "workflow_id", "stage_id", "attempt_id", "lease_id", "lease_epoch",
}
_PAYLOAD_FIELDS = {
    "hello": {
        "node_id", "worker_kind", "min_version", "max_version",
        "capabilities",
    },
    "hello_ack": {
        "coordinator_node_id", "accepted", "selected_version", "reason_code",
    },
    "stage_offer": _IDENTITY_FIELDS | {
        "request_id", "stage_type", "provider_id", "lease_expires_at_ms",
        "root_input", "dependencies", "input_sha256",
    },
    "stage_accept": _IDENTITY_FIELDS | {
        "provider_id", "accepted", "reason_code",
    },
    "lease_renew": _IDENTITY_FIELDS | {"lease_expires_at_ms"},
    "stage_result": _IDENTITY_FIELDS | {
        "provider_id", "output", "output_sha256", "metadata",
    },
    "stage_error": _IDENTITY_FIELDS | {
        "provider_id", "error_code", "retryable",
    },
    "stage_cancel": _IDENTITY_FIELDS | {"reason_code"},
    "stage_cancelled": _IDENTITY_FIELDS | {
        "provider_id", "reason_code",
    },
    #: ★ 2026-10-07（DIST-NEXT-2b）：大 payload 分片（v3 起，见 `MAX_STAGE_CHUNKS`）。
    #: `payload_b64` 是**原始字节**的 base64；`payload_sha256` 是这些字节的摘要，
    #: 使接收端能在装配前逐片校验（装配完仍以 `stage_offer.hidden_sha256` 兜底）。
    "stage_chunk": _IDENTITY_FIELDS | {
        "provider_id", "chunk_index", "chunk_count", "payload_b64",
        "payload_sha256", "total_bytes",
    },
    #: ★ 2026-10-11（ROG+Y700+本机三机实测）：**回程**分片。
    #: `stage_result.output.hidden_out_f32` 超过单帧预算时，worker 先发 N 条本类型，
    #: 再发一条引用它们的 `stage_result`（`output.hidden_out_ref`）。字段与 `stage_chunk`
    #: **同形**，以便复用两侧既有装配器；仅对在 HELLO 里声明 `stage_chunked_output`
    #: 的对端使用，旧对端行为不变。
    "stage_result_chunk": _IDENTITY_FIELDS | {
        "provider_id", "chunk_index", "chunk_count", "payload_b64",
        "payload_sha256", "total_bytes",
    },
}
_PAYLOAD_FIELDS_V2 = {
    **_PAYLOAD_FIELDS,
    "stage_offer": _PAYLOAD_FIELDS["stage_offer"] | {"model_identity"},
    "stage_accept": _PAYLOAD_FIELDS["stage_accept"] | {"retryable"},
}
#: ★ 2026-09-20（v3）：层段承载。**注意这里不扩任何顶层字段** ——
#: 层段字段**只对 `stage_type == "layer_forward"` 生效**，由 `_validate_payload`
#: 在精确字段校验时按 `stage_type` **动态**并入（否则 `full_inference` 也会被要求
#: 提供 hidden 字段 —— 实测会让既有任务图全线 500）。
#:
#: 层段的**结果**字段（`hidden_out_sha256` / `token_argmax`）刻意**不扩顶层**：
#: `stage_result` 的 payload 里**没有** `stage_type`，无法做动态判断 ⇒ 放进本来就
#: 自由的 `output` / `metadata` 对象内（见 `_LAYER_FORWARD_RESULT_FIELDS` 的说明）。
_PAYLOAD_FIELDS_V3 = {
    **_PAYLOAD_FIELDS_V2,
    # ★ 2026-10-09（稳定性 #73）：`stage_error` 增加**可选** `reason`（worker 的异常文本），
    #   让 master / 用户端能看到真因（例：`ValueError: hidden 形状应为 [n_tokens, 2048]，
    #   实际 (33, 896)`），而不是只剩一句「remote worker reported a Stage error」。
    #   接收端对 stage_error 只做正向校验、**不拒绝未知字段**（见 `_validate_payload`）
    #   ⇒ 老 master 收到新 worker 的消息时会忽略该字段，天然向后兼容。
    "stage_error": _PAYLOAD_FIELDS_V2["stage_error"] | {"reason"},
}


class WorkerProtocolError(ValueError):
    """Stable protocol failure that is safe to return without raw payloads."""

    def __init__(self, message: str, *, code: str, field: str = ""):
        self.code = code
        self.field = field
        super().__init__(message)


@dataclass(frozen=True)
class WorkerMessage:
    protocol: str
    version: int
    message_type: str
    message_id: str
    sent_at_ms: int
    _payload_json: str

    @property
    def payload(self) -> dict[str, Any]:
        value = json.loads(self._payload_json)
        if not isinstance(value, dict):
            raise RuntimeError("validated WorkerMessage payload is not an object")
        return value

    def snapshot(self) -> dict[str, Any]:
        return {
            "protocol": self.protocol,
            "version": self.version,
            "message_type": self.message_type,
            "message_id": self.message_id,
            "sent_at_ms": self.sent_at_ms,
            "payload": self.payload,
        }


def _error(code: str, field: str, message: str) -> WorkerProtocolError:
    return WorkerProtocolError(message, code=code, field=field)


def _require_int(value: Any, field: str, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise _error("invalid_integer", field, f"{field} must be an integer")
    return value


def _require_bool(value: Any, field: str) -> bool:
    if not isinstance(value, bool):
        raise _error("invalid_boolean", field, f"{field} must be a boolean")
    return value


def _require_string(
    value: Any,
    field: str,
    *,
    pattern: re.Pattern[str] | None = None,
    allow_empty: bool = False,
    max_length: int = 256,
) -> str:
    if not isinstance(value, str) or len(value) > max_length:
        raise _error("invalid_string", field, f"{field} must be a string")
    if not value and not allow_empty:
        raise _error("invalid_string", field, f"{field} must not be empty")
    if value and pattern is not None and pattern.fullmatch(value) is None:
        raise _error("invalid_identifier", field, f"{field} is invalid")
    return value


def _require_object(value: Any, field: str) -> dict[str, Any]:
    if not isinstance(value, dict) or any(
        not isinstance(key, str) for key in value
    ):
        raise _error("invalid_object", field, f"{field} must be an object")
    try:
        canonical_json(value)
    except (TypeError, ValueError) as exc:
        raise _error(
            "invalid_json_value", field, f"{field} must contain strict JSON"
        ) from exc
    return value


def _require_exact_fields(
    value: Mapping[str, Any], expected: set[str], field: str,
) -> None:
    actual = set(value)
    if actual == expected:
        return
    missing = sorted(expected - actual)
    unknown = sorted(actual - expected)
    reason = ""
    if missing:
        reason += f" missing={missing}"
    if unknown:
        reason += f" unknown={unknown}"
    raise _error(
        "invalid_fields", field, f"{field} fields do not match schema:{reason}"
    )


def canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def stage_input_sha256(
    root_input: dict,
    dependencies: dict,
    transfer_plan: dict | None = None,
) -> str:
    body: dict[str, Any] = {
        "dependencies": dependencies,
        "root_input": root_input,
    }
    if transfer_plan is not None:
        body["transfer_plan"] = transfer_plan
    return canonical_sha256(body)


def stage_output_sha256(output: dict, transfer_plan: dict | None = None) -> str:
    if transfer_plan is None:
        return canonical_sha256(output)
    return canonical_sha256({"output": output, "transfer_plan": transfer_plan})


def _validate_version_range(
    minimum: Any, maximum: Any, *, prefix: str,
) -> tuple[int, int]:
    min_version = _require_int(minimum, f"{prefix}.min_version", minimum=1)
    max_version = _require_int(maximum, f"{prefix}.max_version", minimum=1)
    if min_version > max_version:
        raise _error(
            "invalid_version_range",
            prefix,
            "minimum protocol version exceeds maximum",
        )
    return min_version, max_version


def negotiate_protocol_version(
    remote_min_version: int,
    remote_max_version: int,
    *,
    local_min_version: int = MIN_PROTOCOL_VERSION,
    local_max_version: int = MAX_PROTOCOL_VERSION,
) -> int:
    remote_min, remote_max = _validate_version_range(
        remote_min_version, remote_max_version, prefix="remote"
    )
    local_min, local_max = _validate_version_range(
        local_min_version, local_max_version, prefix="local"
    )
    if local_min < MIN_PROTOCOL_VERSION or local_max > MAX_PROTOCOL_VERSION:
        raise _error(
            "invalid_version_range",
            "local",
            "local range includes an unimplemented protocol version",
        )
    selected = min(remote_max, local_max)
    if selected < max(remote_min, local_min):
        raise _error(
            "unsupported_protocol_version",
            "version",
            "worker and coordinator protocol ranges do not overlap",
        )
    return selected


def _validate_identity(payload: dict[str, Any]) -> None:
    _require_string(payload["workflow_id"], "payload.workflow_id", pattern=_WORKFLOW_ID)
    _require_string(payload["stage_id"], "payload.stage_id", pattern=_SAFE_ID)
    _require_string(payload["attempt_id"], "payload.attempt_id", pattern=_ATTEMPT_ID)
    _require_string(payload["lease_id"], "payload.lease_id", pattern=_LEASE_ID)
    _require_int(payload["lease_epoch"], "payload.lease_epoch", minimum=1)


def _validate_model_identity(value: Any, field: str) -> dict[str, Any]:
    model = _require_object(value, field)
    _require_exact_fields(
        model,
        {"model_id", "engine", "format", "revision", "sha256"},
        field,
    )
    _require_string(model["model_id"], f"{field}.model_id", pattern=_SAFE_ID)
    # "island": TP 孤岛引擎（网关整请求转发，指纹为端点摘要）
    # "external_api": 外部推理服务（路线 B，指纹为外部端点摘要）
    engine = _require_string(
        model["engine"], f"{field}.engine", max_length=64,
    )
    try:
        engine = canonical_model_engine(engine)
    except ValueError as exc:
        raise _error(
            "invalid_model_identity", f"{field}.engine",
            "model engine is unsupported",
        ) from exc
    _require_string(model["format"], f"{field}.format", pattern=_SAFE_ID)
    _require_string(model["revision"], f"{field}.revision", pattern=_SAFE_ID)
    _require_string(model["sha256"], f"{field}.sha256", pattern=_SHA256)
    normalized = dict(model)
    normalized["engine"] = engine
    return normalized


def _validate_layer_budget(value: Any) -> None:
    """校验设备自荐的层容量（`capabilities.layer_budget`，可选字段）。"""
    budget = _require_object(value, "payload.capabilities.layer_budget")
    for key in ("available_bytes", "per_layer_bytes", "max_layers"):
        if key not in budget:
            raise _error(
                "invalid_capabilities",
                f"payload.capabilities.layer_budget.{key}",
                f"layer_budget.{key} is required",
            )
    _require_int(
        budget["available_bytes"],
        "payload.capabilities.layer_budget.available_bytes",
        minimum=0,
    )
    _require_int(
        budget["per_layer_bytes"],
        "payload.capabilities.layer_budget.per_layer_bytes",
        minimum=1,
    )
    max_layers = _require_int(
        budget["max_layers"],
        "payload.capabilities.layer_budget.max_layers",
        minimum=1,
    )
    # 上限只防荒谬值，远大于任何现役模型的层数。
    if max_layers > 1024:
        raise _error(
            "invalid_capabilities",
            "payload.capabilities.layer_budget.max_layers",
            "max_layers must be <= 1024",
        )
    if "local_cut" in budget:
        _require_bool(
            budget["local_cut"],
            "payload.capabilities.layer_budget.local_cut",
        )


def _validate_layer_artifact_diagnostics(value: Any) -> None:
    """★ 2026-10-07（DIST-NEXT-6）：校验不可用层段工件的诊断广告。

    键集固定、**不含本地路径**（路径只留在设备日志与本地诊断）；这样主节点能看到
    「哪个区间为什么不可用」，而不是只看到 `layer_range_not_advertised`。
    """
    field = "payload.capabilities.layer_artifact_diagnostics"
    if (
        not isinstance(value, list)
        or not value
        or len(value) > _LAYER_ARTIFACT_DIAGNOSTIC_MAX
    ):
        raise _error(
            "invalid_capabilities", field,
            "layer_artifact_diagnostics must be a non-empty list with at most "
            f"{_LAYER_ARTIFACT_DIAGNOSTIC_MAX} entries",
        )
    for index, entry in enumerate(value):
        item_field = f"{field}[{index}]"
        entry = _require_object(entry, item_field)
        _require_exact_fields(
            entry,
            {
                "error_code", "manifest", "architecture", "mode",
                "layer_range", "artifact_present",
            },
            item_field,
        )
        error_code = _require_string(
            entry["error_code"], f"{item_field}.error_code",
            pattern=_SAFE_CODE, max_length=64,
        )
        if error_code not in _LAYER_ARTIFACT_DIAGNOSTIC_CODES:
            raise _error(
                "invalid_capabilities", f"{item_field}.error_code",
                "error_code must be one of "
                + ", ".join(sorted(_LAYER_ARTIFACT_DIAGNOSTIC_CODES)),
            )
        _require_string(entry["manifest"], f"{item_field}.manifest", pattern=_SAFE_ID)
        _require_string(
            entry["architecture"], f"{item_field}.architecture",
            pattern=_SAFE_ID, allow_empty=True, max_length=64,
        )
        mode = _require_string(
            entry["mode"], f"{item_field}.mode",
            pattern=_SAFE_ID, allow_empty=True, max_length=32,
        )
        if mode and mode not in _LAYER_SEGMENT_MODES:
            raise _error(
                "invalid_capabilities", f"{item_field}.mode",
                "mode must be empty or one of head, middle, tail",
            )
        _require_bool(entry["artifact_present"], f"{item_field}.artifact_present")
        layer_range = entry["layer_range"]
        if layer_range is None:
            continue
        if (
            not isinstance(layer_range, list)
            or len(layer_range) != 2
            or any(
                isinstance(bound, bool) or not isinstance(bound, int)
                for bound in layer_range
            )
            or layer_range[0] < 0
            or layer_range[1] <= layer_range[0]
        ):
            raise _error(
                "invalid_capabilities", f"{item_field}.layer_range",
                "layer_range must be null or a non-empty [start, end) integer range",
            )


def _validate_capabilities(value: Any, *, version: int) -> None:
    capabilities = _require_object(value, "payload.capabilities")
    expected_fields = {"stage_types", "engines", "models", "max_concurrency"}
    # Optional for backwards compatibility with existing PC workers. Android
    # workers are admitted only when this gate is present and explicitly true;
    # the scheduler performs that role-specific check after hello validation.
    if "engine_capabilities" in capabilities:
        expected_fields.add("engine_capabilities")
    if "load_spec_version" in capabilities:
        expected_fields.add("load_spec_version")
        if type(capabilities["load_spec_version"]) is not int or capabilities["load_spec_version"] != 1:
            raise _error("invalid_capabilities", "payload.capabilities.load_spec_version", "unsupported load policy schema")
    if "resource_gate" in capabilities:
        expected_fields.add("resource_gate")
    if "layer_ranges" in capabilities:
        expected_fields.add("layer_ranges")
    # A layer-only worker advertises a deterministic source-SHA alias instead
    # of a full-model identity.  This marker is optional for older workers.
    if "layer_worker" in capabilities:
        expected_fields.add("layer_worker")
    if "runtime_profile" in capabilities:
        expected_fields.add("runtime_profile")
    # ★ 2026-09-23：中间段通道能力（可选，向后兼容）—— 让调度侧知道该节点中间段
    #   实际能走哪条通道（值域同 `_LAYER_FORWARD_MIDDLE_CHANNELS`）；缺失 = 未声明。
    if "middle_channel" in capabilities:
        expected_fields.add("middle_channel")
    # ★ 2026-09-23：M-RoPE 模型的位置分量数（1 或 4）—— 供调度侧构造 hidden spec / 判据用。
    if "n_pos_per_embd" in capabilities:
        expected_fields.add("n_pos_per_embd")
    # ★ 2026-10-03：层容量自荐（可选，向后兼容）—— 设备按自身可用内存申报"本地裁层
    #   之后最多能承载多少层"，并声明是否具备本地裁层条件。与 `layer_ranges` 分工：
    #   后者是"当前已就绪、马上能跑的区间"，前者是"能承载的上限"。
    if "layer_budget" in capabilities:
        expected_fields.add("layer_budget")
    # ★ 2026-10-03：relay 中段角色（本节点不加载层，段由远端 relay 服务持有）。
    #   声明侧早已在发这个键（`scheduler_task_worker._task_worker_capabilities`），
    #   但这里没有放行 ⇒ 该 worker 的 hello 会以 `unknown=['relay_middle']` 被判
    #   `field_mismatch`。与其余可选键同样按「出现才允许」处理。
    if "relay_middle" in capabilities:
        expected_fields.add("relay_middle")
    # ★ 2026-10-05（DIST-3 三机实测）：工件段类型（可选）—— `head`/`middle`/`tail`，
    #   取值同主仓 `scripts/cut_layers.py` 的 `mode`。让调度侧能拒绝「中间段接末段」
    #   （中间段工件没有 lm_head / final_norm）。
    #   与其余可选键同样按「出现才允许」处理 —— 漏掉这一行，该 worker 的 hello 会以
    #   `unknown=['segment_mode']` 被判 `invalid_fields` ⇒ **TCP 连上但握手中断**
    #   （实测：Y700 每 30s 重连一次，master 侧只剩 `task_worker_handshake_pending`）。
    if "segment_mode" in capabilities:
        expected_fields.add("segment_mode")
    # Per-artifact metadata is required when one worker owns heterogeneous
    # ranges (for example a middle segment and a tail segment).  A single
    # node-level ``segment_mode`` cannot describe that safely.
    if "layer_artifacts" in capabilities:
        expected_fields.add("layer_artifacts")
    # ★ 2026-10-07（DIST-NEXT-6）：不可用层段工件的结构化原因（可选，向后兼容）。
    #   广告侧只带 error code 与身份/区间，**不含本地路径**。与其余可选键同样
    #   按「出现才允许」处理 —— 两侧必须同时放行，否则 hello 自校验失败后
    #   连接建立即 0.7s 静默断开（`segment_mode` 那次实测的坑）。
    if "layer_artifact_diagnostics" in capabilities:
        expected_fields.add("layer_artifact_diagnostics")
    # ★ 2026-10-07（DIST-NEXT-2b）：大 payload 分片接收能力（可选，默认关）。
    #   只有声明 `stage_chunked_input == true` 的 worker 才会收到 `stage_chunk` +
    #   `stage_offer.root_input.hidden_ref`；未声明者继续走内联 hidden（由 DIST-NEXT-2a
    #   的 dispatch 前预检负责拒绝超预算请求）。两侧必须同时放行。
    if "stage_chunked_input" in capabilities:
        expected_fields.add("stage_chunked_input")
    if "node_incarnation" in capabilities:
        expected_fields.add("node_incarnation")
        _require_string(capabilities["node_incarnation"], "payload.capabilities.node_incarnation")
    if "stage_kv_cleanup" in capabilities:
        expected_fields.add("stage_kv_cleanup")
        _require_bool(capabilities["stage_kv_cleanup"], "payload.capabilities.stage_kv_cleanup")
    # ★ 票 21a（COMM-WIRE-01）：层段 hidden 的**二进制帧**接收能力（可选，默认关）。
    #   声明 `binary_hidden == true` 的对端，才可能收到 `format:"binary"` 的帧
    #   （见 `build_message` 的 `binary_body`）；未声明者继续走 base64 内联，
    #   字节与旧行为一致。与其余可选键同样按「出现才允许」处理 —— 两侧
    #   （主仓这里与 android `TaskWorkerProtocol`）必须**同时**放行，只改一边
    #   会让对端 hello 自校验失败（`segment_mode` 那次实测的静默断连坑）。
    if "binary_hidden" in capabilities:
        expected_fields.add("binary_hidden")
    _require_exact_fields(
        capabilities,
        expected_fields,
        "payload.capabilities",
    )
    if "layer_budget" in capabilities:
        _validate_layer_budget(capabilities["layer_budget"])
    if "middle_channel" in capabilities:
        channel = _require_string(
            capabilities["middle_channel"], "payload.capabilities.middle_channel",
        )
        if channel not in _LAYER_FORWARD_MIDDLE_CHANNELS:
            raise _error(
                "invalid_capabilities", "payload.capabilities.middle_channel",
                "middle_channel must be one of "
                + ", ".join(sorted(_LAYER_FORWARD_MIDDLE_CHANNELS)),
            )
    if "n_pos_per_embd" in capabilities:
        n_pos = _require_int(
            capabilities["n_pos_per_embd"], "payload.capabilities.n_pos_per_embd", minimum=1,
        )
        if n_pos not in {1, 4}:
            raise _error(
                "invalid_capabilities", "payload.capabilities.n_pos_per_embd",
                "n_pos_per_embd must be 1 or 4",
            )
    if "layer_worker" in capabilities:
        _require_bool(
            capabilities["layer_worker"], "payload.capabilities.layer_worker",
        )
    if "runtime_profile" in capabilities:
        runtime_profile = _require_string(
            capabilities["runtime_profile"],
            "payload.capabilities.runtime_profile",
            pattern=_SAFE_ID,
        )
        if runtime_profile != RUNTIME_PROFILE_UNSPECIFIED and runtime_profile not in RUNTIME_PROFILES:
            raise _error(
                "invalid_capabilities", "payload.capabilities.runtime_profile",
                "runtime_profile is not a supported release profile",
            )
    stage_types = capabilities["stage_types"]
    if not isinstance(stage_types, list) or not stage_types:
        raise _error(
            "invalid_capabilities", "payload.capabilities.stage_types",
            "stage_types must be a non-empty list",
        )
    if any(value not in _TEXT_STAGE_TYPES for value in stage_types):
        raise _error(
            "invalid_capabilities", "payload.capabilities.stage_types",
            "unsupported stage type",
        )
    if len(stage_types) != len(set(stage_types)):
        raise _error(
            "invalid_capabilities", "payload.capabilities.stage_types",
            "stage_types must not contain duplicates",
        )
    if "layer_ranges" in capabilities:
        ranges = capabilities["layer_ranges"]
        if not isinstance(ranges, list) or any(
            not isinstance(item, list) or len(item) != 2
            or any(isinstance(value, bool) or not isinstance(value, int) for value in item)
            or item[0] < 0 or item[1] <= item[0]
            for item in ranges
        ):
            raise _error(
                "invalid_capabilities", "payload.capabilities.layer_ranges",
                "layer_ranges must contain non-empty [start, end) integer ranges",
            )
        if len(ranges) != len({tuple(item) for item in ranges}):
            raise _error(
                "invalid_capabilities", "payload.capabilities.layer_ranges",
                "layer_ranges must not contain duplicates",
            )
    if "segment_mode" in capabilities:
        segment_mode = _require_string(
            capabilities["segment_mode"],
            "payload.capabilities.segment_mode",
        )
        if segment_mode not in _LAYER_SEGMENT_MODES:
            raise _error(
                "invalid_capabilities", "payload.capabilities.segment_mode",
                "segment_mode must be head, middle, or tail",
            )
    if "layer_artifacts" in capabilities:
        artifacts = capabilities["layer_artifacts"]
        field = "payload.capabilities.layer_artifacts"
        if not isinstance(artifacts, list) or not artifacts or len(artifacts) > 64:
            raise _error(
                "invalid_capabilities", field,
                "layer_artifacts must be a non-empty list with at most 64 entries",
            )
        artifact_ranges: list[tuple[int, int]] = []
        artifact_modes: set[str] = set()
        for index, artifact in enumerate(artifacts):
            item_field = f"{field}[{index}]"
            artifact = _require_object(artifact, item_field)
            expected_artifact_fields = {
                "layer_range", "segment_mode", "model_id", "artifact_sha256",
            }
            if "layer_weight_bytes" in artifact:
                expected_artifact_fields.add("layer_weight_bytes")
            if "source_model_sha256" in artifact:
                expected_artifact_fields.add("source_model_sha256")
            contract_fields = {
                "source_model_id", "hidden_size", "tokenizer_sha256",
            }
            present_contract_fields = contract_fields.intersection(artifact)
            if present_contract_fields and present_contract_fields != contract_fields:
                raise _error(
                    "invalid_capabilities", item_field,
                    "source_model_id, hidden_size, and tokenizer_sha256 "
                    "must be declared together",
                )
            expected_artifact_fields.update(present_contract_fields)
            _require_exact_fields(artifact, expected_artifact_fields, item_field)
            layer_range = artifact["layer_range"]
            if (
                not isinstance(layer_range, list)
                or len(layer_range) != 2
                or any(
                    isinstance(bound, bool) or not isinstance(bound, int)
                    for bound in layer_range
                )
                or layer_range[0] < 0
                or layer_range[1] <= layer_range[0]
            ):
                raise _error(
                    "invalid_capabilities", f"{item_field}.layer_range",
                    "layer_range must be a non-empty [start, end) integer range",
                )
            if "layer_weight_bytes" in artifact:
                sizes = artifact["layer_weight_bytes"]
                if not isinstance(sizes, list) or len(sizes) != layer_range[1] - layer_range[0]:
                    raise _error(
                        "invalid_capabilities", f"{item_field}.layer_weight_bytes",
                        "layer_weight_bytes must match the segment span",
                    )
                for offset, size in enumerate(sizes):
                    _require_int(
                        size, f"{item_field}.layer_weight_bytes[{offset}]", minimum=1,
                    )
            mode = _require_string(
                artifact["segment_mode"], f"{item_field}.segment_mode",
            )
            if mode not in _LAYER_SEGMENT_MODES:
                raise _error(
                    "invalid_capabilities", f"{item_field}.segment_mode",
                    "segment_mode must be head, middle, or tail",
                )
            _require_string(
                artifact["model_id"], f"{item_field}.model_id", pattern=_SAFE_ID,
            )
            _require_string(
                artifact["artifact_sha256"],
                f"{item_field}.artifact_sha256",
                pattern=_SHA256,
            )
            if "source_model_sha256" in artifact:
                _require_string(
                    artifact["source_model_sha256"],
                    f"{item_field}.source_model_sha256",
                    pattern=_SHA256,
                )
            if present_contract_fields:
                _require_string(
                    artifact["source_model_id"],
                    f"{item_field}.source_model_id",
                    pattern=_SAFE_ID,
                )
                _require_int(
                    artifact["hidden_size"],
                    f"{item_field}.hidden_size",
                    minimum=1,
                )
                _require_string(
                    artifact["tokenizer_sha256"],
                    f"{item_field}.tokenizer_sha256",
                    pattern=_SHA256,
                )
            artifact_ranges.append((layer_range[0], layer_range[1]))
            artifact_modes.add(mode)
        if len(artifact_ranges) != len(set(artifact_ranges)):
            raise _error(
                "invalid_capabilities", field,
                "layer_artifacts must not contain duplicate layer_range values",
            )
        if "layer_ranges" not in capabilities or set(artifact_ranges) != {
            tuple(item) for item in capabilities.get("layer_ranges", [])
        }:
            raise _error(
                "invalid_capabilities", field,
                "layer_artifacts and layer_ranges must advertise the same ranges",
            )
        if (
            "segment_mode" in capabilities
            and artifact_modes != {capabilities["segment_mode"]}
        ):
            raise _error(
                "invalid_capabilities", "payload.capabilities.segment_mode",
                "node-level segment_mode must agree with every layer_artifact",
            )
    if "layer_artifact_diagnostics" in capabilities:
        _validate_layer_artifact_diagnostics(
            capabilities["layer_artifact_diagnostics"],
        )
    if "stage_chunked_input" in capabilities:
        _require_bool(
            capabilities["stage_chunked_input"],
            "payload.capabilities.stage_chunked_input",
        )
    if "binary_hidden" in capabilities:
        _require_bool(
            capabilities["binary_hidden"], "payload.capabilities.binary_hidden",
        )
    engines = capabilities["engines"]
    if not isinstance(engines, list) or not engines:
        raise _error(
            "invalid_capabilities", "payload.capabilities.engines",
            "engines must contain supported engine identifiers",
        )
    canonical_engines = []
    for index, value in enumerate(engines):
        try:
            engine = canonical_model_engine(_require_string(
                value,
                f"payload.capabilities.engines[{index}]",
                max_length=64,
            ))
        except (ValueError, WorkerProtocolError) as exc:
            raise _error(
                "invalid_capabilities", "payload.capabilities.engines",
                "engines must contain supported engine identifiers",
            ) from exc
        canonical_engines.append(engine)
    engines = canonical_engines
    capabilities["engines"] = engines
    if len(engines) != len(set(engines)):
        raise _error(
            "invalid_capabilities", "payload.capabilities.engines",
            "engines must not contain duplicates",
        )
    models = capabilities["models"]
    if not isinstance(models, list):
        raise _error(
            "invalid_capabilities", "payload.capabilities.models",
            "models must be a list",
        )
    model_ids = []
    for index, model in enumerate(models):
        field = f"payload.capabilities.models[{index}]"
        model = _validate_model_identity(model, field)
        models[index] = model
        model_ids.append(model["model_id"])
        if model["engine"] not in engines:
            raise _error(
                "invalid_capabilities", f"{field}.engine",
                "model engine was not declared by the worker",
            )
    if len(model_ids) != len(set(model_ids)):
        raise _error(
            "invalid_capabilities", "payload.capabilities.models",
            "model_id values must be unique",
        )
    if "layer_artifacts" in capabilities:
        models_by_id = {
            str(model.get("model_id", "")): model
            for model in models if isinstance(model, dict)
        }
        for index, artifact in enumerate(capabilities["layer_artifacts"]):
            artifact_model_id = str(artifact.get("model_id", ""))
            advertised_model = models_by_id.get(artifact_model_id)
            if advertised_model is None:
                raise _error(
                    "invalid_capabilities",
                    f"payload.capabilities.layer_artifacts[{index}].model_id",
                    "layer artifact model_id must reference capabilities.models",
                )
            if advertised_model.get("sha256") != artifact.get("artifact_sha256"):
                raise _error(
                    "invalid_capabilities",
                    f"payload.capabilities.layer_artifacts[{index}].artifact_sha256",
                    "layer artifact digest must match its advertised model identity",
                )
            if (advertised_model.get("engine"), advertised_model.get("format")) not in {
                ("llama_cpp", "gguf"), ("pytorch", "safetensors"),
            }:
                raise _error(
                    "invalid_capabilities",
                    f"payload.capabilities.layer_artifacts[{index}].model_id",
                    "layer artifact identity must reference a supported engine/format pair",
                )
    max_concurrency = _require_int(
        capabilities["max_concurrency"],
        "payload.capabilities.max_concurrency",
        minimum=1,
    )
    if max_concurrency > 32:
        raise _error(
            "invalid_capabilities", "payload.capabilities.max_concurrency",
            "max_concurrency must not exceed 32",
        )
    if "resource_gate" in capabilities:
        _validate_resource_gate(capabilities["resource_gate"])
    try:
        validate_worker_engine_capabilities(capabilities)
    except (ValueError, TypeError) as exc:
        raise _error(
            "invalid_capabilities", "payload.capabilities.engine_capabilities", str(exc),
        ) from exc



def _validate_resource_gate(value: Any) -> None:
    field = "payload.capabilities.resource_gate"
    gate = _require_object(value, field)
    _require_exact_fields(gate, {"admitted", "reason_code"}, field)
    _require_bool(gate["admitted"], f"{field}.admitted")
    reason = _require_string(
        gate["reason_code"], f"{field}.reason_code",
        pattern=_SAFE_CODE, allow_empty=True, max_length=64,
    )
    if gate["admitted"] and reason:
        raise _error(
            "invalid_resource_gate", f"{field}.reason_code",
            "an admitted resource gate cannot carry a rejection reason",
        )
    if not gate["admitted"] and not reason:
        raise _error(
            "invalid_resource_gate", f"{field}.reason_code",
            "a rejected resource gate must carry a reason code",
        )


def _validate_metadata(value: Any, *, version: int) -> None:
    metadata = _require_object(value, "payload.metadata")
    allowed = {"usage", "usage_estimated", "tokens_per_second", "model"}
    # ★ 2026-10-03：层段（`layer_forward`）结果的**对账字段** —— Android executor 会
    #   回记实际生效的中间段通道、交接点与是否尾段，供 master 与请求对账。它们与 v2
    #   的 `model` 同属审计信息，此前被这里拒掉 ⇒ 回程 result 解码失败、请求只能等到
    #   超时（真机表现为 `remote Stage response timed out`）。
    allowed |= {"stage", "middle_channel", "handoff_at", "tail"}
    if not set(metadata).issubset(allowed):
        raise _error(
            "invalid_fields", "payload.metadata",
            "metadata contains unsupported fields",
        )
    if "usage_estimated" in metadata:
        _require_bool(metadata["usage_estimated"], "payload.metadata.usage_estimated")
    if "model" in metadata:
        _require_string(
            metadata["model"], "payload.metadata.model", max_length=256,
        )
    if "tokens_per_second" in metadata:
        value = metadata["tokens_per_second"]
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(value)
            or value < 0
        ):
            raise _error(
                "invalid_number", "payload.metadata.tokens_per_second",
                "tokens_per_second must be finite and non-negative",
            )
    usage = metadata.get("usage")
    if usage is not None:
        usage = _require_object(usage, "payload.metadata.usage")
        allowed_usage = {
            "prompt_tokens", "completion_tokens", "total_tokens", "cached_tokens",
        }
        if not set(usage).issubset(allowed_usage):
            raise _error(
                "invalid_fields", "payload.metadata.usage",
                "usage contains unsupported fields",
            )
        for key, item in usage.items():
            _require_int(item, f"payload.metadata.usage.{key}", minimum=0)


def _validate_payload(
    message_type: str,
    payload: dict[str, Any],
    version: int,
) -> None:
    fields = (
        _PAYLOAD_FIELDS_V3 if version >= 3
        else _PAYLOAD_FIELDS_V2 if version >= 2
        else _PAYLOAD_FIELDS
    )
    required = fields[message_type]
    if message_type == "stage_offer" and isinstance(payload, Mapping):
        required = required | (_STAGE_OFFER_OPTIONAL_FIELDS & set(payload))
    # ★ 2026-09-20（v3）：层段字段**按 `stage_type` 动态并入**。
    #   只有声明 `layer_forward` 的 `stage_offer` 才要求 hidden 交接字段；
    #   非层段 stage 携带这些字段会被精确字段校验拒绝（既不漏也不多）。
    if (
        message_type == "stage_offer"
        and version >= 3
        and isinstance(payload, Mapping)
        and payload.get("stage_type") == "layer_forward"
    ):
        required = required | _LAYER_FORWARD_OFFER_FIELDS
        # ★ 2026-09-23：**可选**层段字段只在 payload 里**真的出现**时放宽 —— 精确校验是双向的，
        #   提前并入会把它们变成必填、破坏既有对端（实测踩到）。
        required = required | (_LAYER_FORWARD_OPTIONAL_FIELDS & set(payload))
    # ★ 2026-10-07（DIST-NEXT-1）：取消 ACK 的执行状态同样是**可选**字段。
    #   ⚠️ 限定 `version >= 3`：v1/v2 的字段表里没有它，若不加判定，v2 对端带字段
    #      会静默通过而不是稳定的 `invalid_fields`。
    if (
        message_type == "stage_cancelled"
        and version >= 3
        and isinstance(payload, Mapping)
    ):
        required = required | (_STAGE_CANCELLED_OPTIONAL_FIELDS & set(payload))
    if message_type == "hello_ack" and "binary_hidden" in payload:
        required = required | {"binary_hidden"}
    _require_exact_fields(payload, required, "payload")
    if message_type == "hello":
        _require_string(payload["node_id"], "payload.node_id", pattern=_SAFE_ID)
        if payload["worker_kind"] not in FULL_WORKER_KINDS:
            raise _error(
                "unsupported_worker_kind", "payload.worker_kind",
                "worker_kind is not supported for this protocol version",
            )
        _validate_version_range(
            payload["min_version"], payload["max_version"], prefix="payload"
        )
        _validate_capabilities(payload["capabilities"], version=version)
        return
    if message_type == "hello_ack":
        _require_string(
            payload["coordinator_node_id"], "payload.coordinator_node_id",
            pattern=_SAFE_ID,
        )
        accepted = _require_bool(payload["accepted"], "payload.accepted")
        selected = _require_int(
            payload["selected_version"], "payload.selected_version", minimum=0,
        )
        reason = _require_string(
            payload["reason_code"], "payload.reason_code", pattern=_SAFE_CODE,
            allow_empty=True, max_length=64,
        )
        if "binary_hidden" in payload:
            binary_hidden = _require_bool(payload["binary_hidden"], "payload.binary_hidden")
            if binary_hidden and (not accepted or version < 3 or selected < 3):
                raise _error(
                    "invalid_negotiation_result", "payload.binary_hidden",
                    "binary hidden requires an accepted v3 hello",
                )
        if accepted and selected != version:
            raise _error(
                "invalid_selected_version", "payload.selected_version",
                "accepted negotiation must select the supported version",
            )
        if (accepted and reason) or (not accepted and (selected != 0 or not reason)):
            raise _error(
                "invalid_negotiation_result", "payload",
                "hello_ack accepted, selected_version and reason_code disagree",
            )
        return

    _validate_identity(payload)
    if "provider_id" in payload:
        _require_string(
            payload["provider_id"], "payload.provider_id", pattern=_SAFE_ID,
        )
    if message_type == "stage_offer":
        _require_string(
            payload["request_id"], "payload.request_id", pattern=_SAFE_ID,
            allow_empty=True,
        )
        if payload["stage_type"] not in _TEXT_STAGE_TYPES:
            raise _error(
                "unsupported_stage_type", "payload.stage_type",
                "unsupported stage type",
            )
        _require_int(
            payload["lease_expires_at_ms"], "payload.lease_expires_at_ms",
            minimum=1,
        )
        if "request_deadline_ms" in payload:
            _require_int(
                payload["request_deadline_ms"], "payload.request_deadline_ms",
                minimum=1,
            )
        root_input = _require_object(payload["root_input"], "payload.root_input")
        dependencies = _require_object(
            payload["dependencies"], "payload.dependencies"
        )
        digest = _require_string(
            payload["input_sha256"], "payload.input_sha256", pattern=_SHA256,
        )
        expected_input_digest = stage_input_sha256(root_input, dependencies)
        if digest != expected_input_digest:
            raise _error(
                "input_digest_mismatch", "payload.input_sha256",
                "stage input digest does not match payload",
            )
        # ★ 2026-09-20（v3）：**层段**专项校验。缺字段由上面的精确字段集校验先拦下；
        #   这里校验**值的合法性**，防止「声明与实际加载的工件不符」的静默错配。
        #   ⚠️ 必须限定 `version >= 3`：v1/v2 的字段表里**没有**层段字段，
        #      若不加该判定，v2 客户端声明 `layer_forward` 会在此处 KeyError 而非
        #      返回稳定的 `field_mismatch`（实测）。
        if version >= 3 and payload["stage_type"] == "layer_forward":
            # 字段存在性已由上面的精确字段集校验保证，这里只校验值。
            layer_range = payload["layer_range"]
            if not isinstance(layer_range, (list, tuple)) or len(layer_range) != 2:
                raise _error(
                    "invalid_layer_range", "payload.layer_range",
                    "layer_range must be [start, end)",
                )
            start = _require_int(layer_range[0], "payload.layer_range[0]", minimum=0)
            end = _require_int(layer_range[1], "payload.layer_range[1]", minimum=1)
            if end <= start:
                raise _error(
                    "invalid_layer_range", "payload.layer_range",
                    "layer_range must satisfy end > start",
                )
            _require_int(payload["handoff_at"], "payload.handoff_at", minimum=0)
            _require_string(
                payload["hidden_sha256"], "payload.hidden_sha256", pattern=_SHA256,
            )
            hidden_spec = _require_object(
                payload["hidden_spec"], "payload.hidden_spec"
            )
            _require_exact_fields(
                hidden_spec, {"n_tokens", "n_embd", "dtype"}, "payload.hidden_spec"
            )
            _require_int(hidden_spec["n_tokens"], "payload.hidden_spec.n_tokens", minimum=1)
            _require_int(hidden_spec["n_embd"], "payload.hidden_spec.n_embd", minimum=1)
            dtype = _require_string(hidden_spec["dtype"], "payload.hidden_spec.dtype")
            if dtype not in {"float32", "float16"}:
                raise _error(
                    "unsupported_hidden_dtype", "payload.hidden_spec.dtype",
                    "hidden dtype must be float32 or float16",
                )
            # ★ 2026-09-23：`middle_channel` 可选；一旦出现必须落在允许集合内（fail-closed）。
            #   字段不存在 = `extract_hidden`（旧行为），因此不破坏既有对端。
            if "prepare_identity" in payload:
                from worker_prepare_contract import validate_stage_identity, WorkerPrepareError
                try:
                    validate_stage_identity(payload["prepare_identity"])
                except WorkerPrepareError as exc:
                    raise _error(exc.reason_code, "payload.prepare_identity", str(exc)) from exc
            if "middle_channel" in payload:
                channel = _require_string(
                    payload["middle_channel"], "payload.middle_channel",
                )
                if channel not in _LAYER_FORWARD_MIDDLE_CHANNELS:
                    raise _error(
                        "unsupported_middle_channel", "payload.middle_channel",
                        "middle_channel must be one of "
                        + ", ".join(sorted(_LAYER_FORWARD_MIDDLE_CHANNELS)),
                    )
            # ★ 2026-09-23（A12）：**多序列显式位置**（可选）—— `seq_ids` / `positions`
            #   长度必须等于 `hidden_spec.n_tokens`，且每个元素是非负整数；与 Android 侧
            #   `layerForward(seqIds=…, positions=…)` 同一契约（多序列交错推进时必需）。
            n_tokens = int(hidden_spec["n_tokens"])
            for field in ("seq_ids", "positions"):
                if field not in payload:
                    continue
                values = payload[field]
                if not isinstance(values, (list, tuple)) or len(values) != n_tokens:
                    raise _error(
                        f"invalid_{field}", f"payload.{field}",
                        f"{field} must be a list of length n_tokens ({n_tokens})",
                    )
                for index, item in enumerate(values):
                    # 错误码与 Android 侧一致（`invalid_<field>`）：长度与取值问题都归同一码，
                    # 便于两侧对账；不用通用 `_require_int`（那会抛 `invalid_integer`）。
                    if isinstance(item, bool) or not isinstance(item, int) or item < 0:
                        raise _error(
                            f"invalid_{field}", f"payload.{field}[{index}]",
                            f"{field} entries must be non-negative integers",
                        )
        elif "layer_range" in payload or "handoff_at" in payload:
            # 非层段 stage 不得携带层段字段（精确字段集已拦，这里是双保险）
            raise _error(
                "unexpected_layer_fields", "payload",
                "layer fields are only valid for layer_forward",
            )
        if version >= 2:
            payload["model_identity"] = _validate_model_identity(
                payload["model_identity"], "payload.model_identity",
            )
    elif message_type == "stage_accept":
        accepted = _require_bool(payload["accepted"], "payload.accepted")
        reason = _require_string(
            payload["reason_code"], "payload.reason_code", pattern=_SAFE_CODE,
            allow_empty=True, max_length=64,
        )
        if accepted == bool(reason):
            raise _error(
                "invalid_acceptance_result", "payload",
                "accepted offers require no reason; rejected offers require one",
            )
        if version >= 2:
            retryable = _require_bool(
                payload["retryable"], "payload.retryable",
            )
            if accepted and retryable:
                raise _error(
                    "invalid_acceptance_result", "payload.retryable",
                    "accepted offers cannot be retryable failures",
                )
    elif message_type == "lease_renew":
        _require_int(
            payload["lease_expires_at_ms"], "payload.lease_expires_at_ms",
            minimum=1,
        )
    elif message_type == "stage_result":
        output = _require_object(payload["output"], "payload.output")
        digest = _require_string(
            payload["output_sha256"], "payload.output_sha256", pattern=_SHA256,
        )
        expected_output_digest = stage_output_sha256(output)
        if digest != expected_output_digest:
            raise _error(
                "output_digest_mismatch", "payload.output_sha256",
                "stage output digest does not match output",
            )
        _validate_metadata(payload["metadata"], version=version)
    elif message_type == "stage_error":
        _require_string(
            payload["error_code"], "payload.error_code", pattern=_SAFE_CODE,
            max_length=64,
        )
        _require_bool(payload["retryable"], "payload.retryable")
    elif message_type in {"stage_chunk", "stage_result_chunk"}:
        # ★ 2026-10-07（DIST-NEXT-2b）：分片的形状与硬上限。顺序语义（不得乱序、
        #   不得重复、集齐才装配）由接收端状态机 `task_worker_chunks.StageChunkAssembler`
        #   保证 —— 协议层只拒绝「单条就不合法」的分片。
        # ★ 2026-10-11：**回程**分片 `stage_result_chunk` 与输入分片**同形**，因此共用本
        #   分支的全部约束（片数/单片/总字节上限、摘要格式、v3 起支持）。
        if version < 3:
            raise _error(
                "unsupported_message_type", "message_type",
                "stage_chunk requires protocol v3 or newer",
            )
        chunk_count = _require_int(
            payload["chunk_count"], "payload.chunk_count", minimum=1,
        )
        if chunk_count > MAX_STAGE_CHUNKS:
            raise _error(
                "chunk_count_out_of_range", "payload.chunk_count",
                f"chunk_count must be <= {MAX_STAGE_CHUNKS}",
            )
        chunk_index = _require_int(
            payload["chunk_index"], "payload.chunk_index", minimum=0,
        )
        if chunk_index >= chunk_count:
            raise _error(
                "chunk_index_out_of_range", "payload.chunk_index",
                "chunk_index must be less than chunk_count",
            )
        total_bytes = _require_int(
            payload["total_bytes"], "payload.total_bytes", minimum=1,
        )
        if total_bytes > MAX_STAGE_PAYLOAD_BYTES:
            raise _error(
                "stage_payload_too_large", "payload.total_bytes",
                f"total_bytes must be <= {MAX_STAGE_PAYLOAD_BYTES}",
            )
        chunk = payload["payload_b64"]
        if not isinstance(chunk, str) or not chunk:
            raise _error(
                "invalid_string", "payload.payload_b64",
                "payload.payload_b64 must be a non-empty string",
            )
        # 单片上限：把 base64 字符数换算回 raw 字节（`±2` 是 padding 的量化误差；
        # **精确**判定在装配器 `StageChunkAssembler.add`，那里按真实字节数比较）。
        approx_raw_bytes = 3 * (-(-len(chunk) // 4))
        if approx_raw_bytes > STAGE_CHUNK_BYTES + 2:
            raise _error(
                "chunk_too_large", "payload.payload_b64",
                f"a single chunk must not exceed {STAGE_CHUNK_BYTES} raw bytes",
            )
        _require_string(
            payload["payload_sha256"], "payload.payload_sha256", pattern=_SHA256,
        )
    else:
        _require_string(
            payload["reason_code"], "payload.reason_code", pattern=_SAFE_CODE,
            max_length=64,
        )
        # ★ 2026-10-07（DIST-NEXT-1）：取消 ACK 的**执行状态**（可选）。
        #   存在的目的是让 master 不把「请求已取消」当成「执行已停止」；
        #   因此值域必须封闭（fail-closed），未知状态一律拒收而不是照抄。
        if message_type == "stage_cancelled" and "execution_state" in payload:
            state = _require_string(
                payload["execution_state"], "payload.execution_state",
                max_length=32,
            )
            if state not in _STAGE_CANCELLED_EXECUTION_STATES:
                raise _error(
                    "unsupported_execution_state", "payload.execution_state",
                    "execution_state must be one of "
                    + ", ".join(sorted(_STAGE_CANCELLED_EXECUTION_STATES)),
                )


def binary_hidden_frame_bytes(
    message: Mapping[str, Any], raw_bytes: int, *,
    control_certificate: Mapping[str, Any] | None = None,
) -> int:
    """Exact TCP binary frame budget, excluding the first metadata length header.

    Match tcp_comm.build_message's default json.dumps encoding, including spaces
    and ASCII escaping. Canonical inner JSON is smaller and is not a wire budget.
    Callers can include the certificate that TCP appends before serializing data.
    """
    data = dict(message)
    if control_certificate is not None and "control_certificate" not in data:
        data["control_certificate"] = dict(control_certificate)
    meta = {
        "type": "task_worker", "format": "binary",
        "binary_bytes": raw_bytes, "data": data,
    }
    return len(json.dumps(meta).encode("utf-8")) + _BINARY_BODY_HEADER_BYTES + raw_bytes


def validate_message(
    value: Mapping[str, Any], *, binary_hidden: bool = False,
) -> WorkerMessage:
    if not isinstance(value, Mapping):
        raise _error("invalid_envelope", "message", "message must be an object")
    _require_exact_fields(value, _ENVELOPE_FIELDS, "message")
    protocol = _require_string(value["protocol"], "protocol", max_length=64)
    if protocol != PROTOCOL_NAME:
        raise _error("unsupported_protocol", "protocol", "unsupported protocol")
    version = _require_int(value["version"], "version", minimum=1)
    if version < MIN_PROTOCOL_VERSION or version > MAX_PROTOCOL_VERSION:
        raise _error(
            "unsupported_protocol_version", "version",
            "unsupported protocol version",
        )
    message_type = _require_string(
        value["message_type"], "message_type", max_length=32,
    )
    if message_type not in MESSAGE_TYPES:
        raise _error(
            "unsupported_message_type", "message_type",
            "unsupported message type",
        )
    message_id = _require_string(
        value["message_id"], "message_id", pattern=_MESSAGE_ID,
    )
    sent_at_ms = _require_int(value["sent_at_ms"], "sent_at_ms", minimum=0)
    payload = _require_object(value["payload"], "payload")
    _validate_payload(message_type, payload, version)
    if message_type in {"stage_offer", "lease_renew"} and (
        payload["lease_expires_at_ms"] <= sent_at_ms
    ):
        raise _error(
            "invalid_lease_deadline", "payload.lease_expires_at_ms",
            "lease deadline must be later than the message timestamp",
        )
    message = WorkerMessage(
        protocol=protocol,
        version=version,
        message_type=message_type,
        message_id=message_id,
        sent_at_ms=sent_at_ms,
        _payload_json=canonical_json(payload),
    )
    try:
        wire_snapshot, body = split_inline_hidden(
            message.snapshot(), enabled=binary_hidden,
        )
        message_size = len(canonical_json(wire_snapshot).encode("utf-8"))
        if body is not None:
            message_size = binary_hidden_frame_bytes(wire_snapshot, len(body))
    except UnicodeEncodeError as exc:
        raise _error(
            "invalid_encoding", "message", "message must be valid UTF-8"
        ) from exc
    message_limit = MAX_BINARY_FRAME_BYTES if body is not None else MAX_MESSAGE_BYTES
    if message_size > message_limit:
        raise _error(
            "message_too_large", "message", "message exceeds maximum size"
        )
    return message


def build_message(
    message_type: str,
    payload: Mapping[str, Any],
    *,
    message_id: str,
    sent_at_ms: int,
    version: int = PROTOCOL_VERSION,
    binary_hidden: bool = False,
) -> WorkerMessage:
    return validate_message({
        "protocol": PROTOCOL_NAME,
        "version": version,
        "message_type": message_type,
        "message_id": message_id,
        "sent_at_ms": sent_at_ms,
        "payload": dict(payload),
    }, binary_hidden=binary_hidden)


def decode_message(
    raw: bytes | str | Mapping[str, Any], *, binary_hidden: bool = False,
) -> WorkerMessage:
    # Restored hidden is base64 in memory, but only an explicitly binary frame
    # may use the larger representation. validate_message checks its wire size.
    representation_limit = (
        _base64_wire_length(MAX_BINARY_FRAME_BYTES) if binary_hidden else MAX_MESSAGE_BYTES
    )
    if isinstance(raw, bytes):
        if len(raw) > representation_limit:
            raise _error(
                "message_too_large", "message", "message exceeds maximum size"
            )
        try:
            raw = raw.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise _error(
                "invalid_encoding", "message", "message must be UTF-8"
            ) from exc
    if isinstance(raw, str):
        try:
            raw_size = len(raw.encode("utf-8"))
        except UnicodeEncodeError as exc:
            raise _error(
                "invalid_encoding", "message", "message must be valid UTF-8"
            ) from exc
        if raw_size > representation_limit:
            raise _error(
                "message_too_large", "message", "message exceeds maximum size"
            )
        try:
            decoded = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise _error("invalid_json", "message", "message is not JSON") from exc
    else:
        decoded = raw
    if not isinstance(decoded, Mapping):
        raise _error("invalid_envelope", "message", "message must be an object")
    return validate_message(decoded, binary_hidden=binary_hidden)


def canonical_message_bytes(message: WorkerMessage) -> bytes:
    return canonical_json(message.snapshot()).encode("utf-8")


def worker_protocol_status(
    adapter_status: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Report schema readiness and optional TC-N2 adapter runtime state."""
    runtime = dict(adapter_status or {})
    return {
        **runtime,
        "protocol": PROTOCOL_NAME,
        "min_version": MIN_PROTOCOL_VERSION,
        "max_version": MAX_PROTOCOL_VERSION,
        "fixture_version": 1,
        "preferred_version": PROTOCOL_VERSION,
        "schema_ready": True,
        # TC-N2.4 adds an explicit experimental gate. Physical-device
        # validation and production admission remain fenced.
        "adapter_connected": bool(runtime.get("adapter_connected", False)),
        "transport": runtime.get(
            "transport", "existing_tcp_length_prefixed"
        ),
        "admission_state": runtime.get(
            "admission_state", "n2_4_experiment_disabled"
        ),
    }


def _base64_wire_length(raw_bytes: int) -> int:
    """标准 base64 编码后的字符数：`4 × ceil(n / 3)`。

    ⚠️ 不要用 `ceil(n × 4/3)` 近似：对 1 MiB 这类 n 会少算 2 个字符，
    使「刚好一片」的分片被误判为超限（实测踩到）。
    """
    count = int(raw_bytes)
    return 4 * (-(-count // 3))


#: ★ 票 21a（COMM-WIRE-01）：二进制体的长度头字节数（`tcp_comm.pack_data` 的
#: 4 字节大端长度前缀）。二进制帧 = `[meta 长度][meta JSON]` + `[本长度][raw 字节]`。
_BINARY_BODY_HEADER_BYTES = 4


def hidden_wire_bytes(
    n_tokens: int, n_embd: int, dtype: str = "float32", *,
    binary: bool = False,
) -> int:
    """层段 hidden 在帧里占用的字节数。

    默认（`binary=False`）按 **JSON + base64** 计（raw 的 `4×ceil(n/3)`）；协议两侧
    （主仓 Python 与 `android` 的 `TaskWorkerProtocol`）用同一公式与同一余量常量。

    ★ 票 21a（COMM-WIRE-01）：`binary=True` 表示该帧走 `format:"binary"` 的**原始
    字节体** ⇒ 只有 `raw` 加一个 4 字节长度头，省掉 33% 膨胀（同样预算下单帧容量
    +≈33%）。**只有对端声明了 `binary_hidden` 才能传 True** —— 否则实际发出的仍是
    base64 帧，会以 `message_too_large` 失败。
    """
    if dtype not in HIDDEN_DTYPE_BYTES:
        raise _error(
            "unsupported_hidden_dtype", "hidden_spec.dtype",
            "hidden dtype must be one of "
            + ", ".join(sorted(HIDDEN_DTYPE_BYTES)),
        )
    tokens = int(n_tokens)
    embd = int(n_embd)
    if tokens < 1 or embd < 1:
        raise _error(
            "invalid_hidden_spec", "hidden_spec",
            "hidden_spec.n_tokens/n_embd must be positive",
        )
    raw_bytes = tokens * embd * HIDDEN_DTYPE_BYTES[dtype]
    if binary:
        return raw_bytes + _BINARY_BODY_HEADER_BYTES
    return _base64_wire_length(raw_bytes)


#: ★ 票 21a（COMM-WIRE-01）：内联 hidden 在消息里的**唯一**落点，按消息类型推导。
#: 因此二进制体**不需要任何新协议字段** —— raw 的位置由「帧 `format:"binary"` + 消息
#: 类型」共同确定，逐位完整性仍由既有的摘要字段承担（`hidden_sha256` /
#: `hidden_out_sha256` / 分片自己的 `payload_sha256`）。
#:
#: 容器键为 `None` 表示**载荷就在 `payload` 顶层**（分片路径：`payload_b64` 本身就是
#: 那一段 raw 字节的 base64）。入程 `stage_chunk` 与回程 `stage_result_chunk` 共用同一
#: 形状，所以这里一条定义覆盖两个方向。
BINARY_HIDDEN_PATHS = {
    "stage_offer": ("root_input", "hidden_f32"),
    "stage_result": ("output", "hidden_out_f32"),
    "stage_chunk": (None, "payload_b64"),
    "stage_result_chunk": (None, "payload_b64"),
}


def inline_hidden_path(message_type: str) -> tuple[str | None, str] | None:
    """该消息类型里内联 hidden 的 `(容器键, 字段名)`；不承载内联 hidden 则 `None`。

    容器键为 `None` 表示载荷就在 `payload` 顶层（分片路径）。
    """
    return BINARY_HIDDEN_PATHS.get(str(message_type))


def _split_field(container: Mapping[str, Any], field_key: str):
    """把 `container[field_key]`（base64 字符串）抽成 raw 字节并删除该字段。

    返回 `(新容器, raw)`；不可抽（非字符串 / 非法 base64 / 空）则返回 `None`。
    """
    encoded = container.get(field_key)
    if not isinstance(encoded, str):
        return None
    try:
        raw = base64.b64decode(encoded, validate=True)
    except Exception:  # noqa: BLE001 - 非法 base64 交给既有协议校验报错
        return None
    if not raw or base64.b64encode(raw).decode("ascii") != encoded:
        return None
    new_container = {key: value for key, value in container.items()}
    new_container.pop(field_key, None)
    return new_container, raw


def _restore_field(container: Mapping[str, Any], field_key: str, raw: bytes):
    """`_split_field` 的逆：把 raw 以 base64 放回 `field_key`。"""
    new_container = {key: value for key, value in container.items()}
    new_container[field_key] = base64.b64encode(raw).decode("ascii")
    return new_container


def split_inline_hidden(
    message: Mapping[str, Any], *, enabled: bool,
) -> "tuple[dict[str, Any], bytes | None]":
    """★ 票 21a：把内联 base64 hidden 抽成**帧的二进制体**（能力位开启时）。

    返回 `(snapshot, raw)`：

    * `enabled` 为假、该消息类型不承载内联 hidden、值是 `hidden_ref`（已走分片）、
      或值不是合法 base64 ⇒ `raw is None` 且结果与输入**逐字段相同**
      （零行为变化：旧路径的帧字节完全不变）；
    * 否则 `raw` = `base64.b64decode(hidden_f32)` 的原始字节，并把该字段从 JSON 里
      **删除** —— 同一份数据不能同帧两份，否则体积比 base64 还大。
    """
    snapshot = {key: value for key, value in message.items()}
    if not enabled:
        return snapshot, None
    path = inline_hidden_path(snapshot.get("message_type", ""))
    if path is None:
        return snapshot, None
    container_key, field_key = path
    payload = snapshot.get("payload")
    if not isinstance(payload, Mapping):
        return snapshot, None
    if container_key is None:
        # 分片路径：载荷就在 `payload` 顶层（`payload_b64` 自身即那一段 raw）。
        result = _split_field(payload, field_key)
        if result is None:
            return snapshot, None
        snapshot["payload"] = result[0]
        return snapshot, result[1]
    container = payload.get(container_key)
    if not isinstance(container, Mapping):
        return snapshot, None
    result = _split_field(container, field_key)
    if result is None:
        # `hidden_ref`（已分片）或缺失 ⇒ 原样保留（该路径由分片消息自己承载）。
        return snapshot, None
    new_payload = {key: value for key, value in payload.items()}
    new_payload[container_key] = result[0]
    snapshot["payload"] = new_payload
    return snapshot, result[1]


def restore_inline_hidden(
    message: Mapping[str, Any], raw: bytes | None,
) -> dict[str, Any]:
    """★ 票 21a：把帧的二进制体放回 JSON 内联 hidden（`split_inline_hidden` 的逆）。

    位置同样按消息类型推导；`raw` 为 `None` 时结果与输入逐字段相同（零行为变化）。
    放回的是 base64 字符串 ⇒ **下游代码完全不必知道二进制路径存在**。
    """
    snapshot = {key: value for key, value in message.items()}
    if raw is None:
        return snapshot
    path = inline_hidden_path(snapshot.get("message_type", ""))
    payload = snapshot.get("payload")
    if (not isinstance(raw, (bytes, bytearray)) or not raw
            or path is None or not isinstance(payload, Mapping)):
        raise _error("invalid_binary_hidden", "message", "binary hidden has no destination")
    container_key, field_key = path
    container = payload if container_key is None else payload.get(container_key)
    ref_key = {"hidden_f32": "hidden_ref", "hidden_out_f32": "hidden_out_ref"}.get(field_key)
    if (not isinstance(container, Mapping) or field_key in container
            or (ref_key is not None and ref_key in container)):
        raise _error("invalid_binary_hidden", "payload", "binary hidden destination is ambiguous")
    if container_key is None:
        snapshot["payload"] = _restore_field(container, field_key, raw)
        return snapshot
    new_payload = {key: value for key, value in payload.items()}
    new_payload[container_key] = _restore_field(container, field_key, raw)
    snapshot["payload"] = new_payload
    return snapshot


def stage_payload_budget_bytes(
    *, max_message_bytes: int = MAX_MESSAGE_BYTES,
) -> int:
    """单条 stage 消息里可承载的 hidden 预算（已扣除帧内其它字段的余量）。"""
    limit = int(max_message_bytes)
    if limit <= STAGE_FRAME_RESERVE_BYTES:
        raise _error(
            "invalid_message_limit", "message",
            "max_message_bytes must exceed the frame reserve",
        )
    return limit - STAGE_FRAME_RESERVE_BYTES


def hidden_fits_stage_frame(
    n_tokens: int,
    n_embd: int,
    dtype: str = "float32",
    *,
    max_message_bytes: int = MAX_MESSAGE_BYTES,
    binary: bool = False,
) -> bool:
    """该 hidden 能否装进单帧（不装得下就必须在 dispatch 前拒绝）。

    `binary=True` 表示该帧会走 `format:"binary"` 的原始字节体 ⇒ 同样预算能多装
    ≈33%（去 base64 膨胀）。**调用方必须先确认对端确实声明了 `binary_hidden`**，
    否则判据与实际发帧方式不一致，帧会在发送时以 `message_too_large` 失败。
    """
    return hidden_wire_bytes(
        n_tokens, n_embd, dtype, binary=binary,
    ) <= stage_payload_budget_bytes(max_message_bytes=max_message_bytes)


def max_hidden_tokens(
    n_embd: int,
    dtype: str = "float32",
    *,
    max_message_bytes: int = MAX_MESSAGE_BYTES,
    binary: bool = False,
) -> int:
    """给定额度下每帧可承载的最大 token 数（诊断与日志用）。"""
    if dtype not in HIDDEN_DTYPE_BYTES:
        raise _error(
            "unsupported_hidden_dtype", "hidden_spec.dtype",
            "hidden dtype must be one of "
            + ", ".join(sorted(HIDDEN_DTYPE_BYTES)),
        )
    embd = int(n_embd)
    if embd < 1:
        raise _error(
            "invalid_hidden_spec", "hidden_spec.n_embd",
            "hidden_spec.n_embd must be positive",
        )
    per_token_raw = embd * HIDDEN_DTYPE_BYTES[dtype]
    budget = stage_payload_budget_bytes(max_message_bytes=max_message_bytes)
    if binary:
        # The body has one length header per frame, not one per token.
        return max(0, budget - _BINARY_BODY_HEADER_BYTES) // per_token_raw
    return int(budget // (per_token_raw * _BASE64_EXPANSION))
