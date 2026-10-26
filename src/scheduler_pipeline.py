"""Distributed inference and pipeline methods mixed into Scheduler."""

from __future__ import annotations

from contextlib import ExitStack

import logging
import json
import os
import threading
import time
import uuid
import base64
import hashlib
import math

from koakuma_engine import Capability, backend_id_for, runtime_supports
from config import (
    PIPELINE_MODEL_SYNC_TIMEOUT,
    PIPELINE_RELAY_ENABLED,
    PIPELINE_RELAY_PROBE_ONLY,
    PIPELINE_RELAY_SEGMENTS,
    PIPELINE_ROUTE_A_STAGE_OFFER_ENABLED,
    TASK_WORKER_EXPERIMENTAL_ENABLED,
    pipeline_stage_lease_seconds,
)
# ★ #31 M2：层流水线支持的架构**单一事实来源**（此前在 4 处各写了一份 `{"qwen","qwen2"}`）
from pipeline_model_descriptor import PIPELINE_RUNTIME_MODEL_TYPES


from qlh_core.execution.stopping import native_thinking_suppression_required

# ★ 2026-10-07（DIST-NEXT-7）：A1 relay 的隔离边界（唯一门 + 独立诊断 namespace）。
from relay_a1_legacy import (
    a1_isolation_status,
    a1_production_enabled,
    parse_relay_segment_map,
)
from relay_segment_client import (
    RelaySegmentClient,
    RelaySegmentError,
)
from relay_transport import is_loopback_host
from request_deadline import (
    REQUEST_DEADLINE_EXCEEDED,
    DeadlineCancelEvent,
    RequestDeadline,
    coerce_request_deadline,
    request_stop_reason,
)
from request_outcome import (
    REASON_GENERATION_CANCELLED,
    TerminalOutcomeLatch,
)
from scheduler_types import WORKER_HEARTBEAT_MAX_AGE
from task_worker_protocol import (
    hidden_fits_stage_frame,
    hidden_wire_bytes,
    max_hidden_tokens,
    stage_payload_budget_bytes,
)
# ★ 2026-10-07（DIST-NEXT-3）：assignment 权威视图的相位与 reason code。
from worker_assignment_state import (
    PHASE_ACKED,
    PHASE_READY,
    REASON_CONFIG_ACKED,
    REASON_CONFIG_CLEARED,
    REASON_WORKER_RELEASED,
    new_assignment_id,
)
from torch_runtime import loaded_torch

logger = logging.getLogger("scheduler")


RELAY_HIDDEN_WIRE_FORMAT = "qlh.relay_hidden.f32.v1"


def _route_a_text_stop_boundary(text: str, stop_sequences: list[str]) -> tuple[str, str]:
    """Return text safe to publish, retaining split stop prefixes and UTF-8 tails."""
    matches = [(text.find(stop), stop) for stop in stop_sequences if stop]
    matches = [(index, stop) for index, stop in matches if index >= 0]
    if matches:
        index, stop = min(matches, key=lambda match: match[0])
        return text[:index], stop
    # A token can end in an incomplete UTF-8 character. Decode the complete
    # generated sequence again next step instead of publishing replacement chars.
    safe_text = text.rstrip("\ufffd")
    withheld = 0
    for stop in stop_sequences:
        for size in range(min(len(stop) - 1, len(safe_text)), withheld, -1):
            if safe_text.endswith(stop[:size]):
                withheld = size
                break
    return safe_text[:-withheld] if withheld else safe_text, ""


def _route_a_eos_ids(value) -> set[int]:
    """Ignore missing, negative and non-integral EOS metadata."""
    from numbers import Integral

    values = value if isinstance(value, (list, tuple, set)) else [value]
    return {
        int(token_id) for token_id in values
        if isinstance(token_id, Integral) and not isinstance(token_id, bool) and token_id >= 0
    }


def _request_terminal_result(
    *,
    cancel_event: threading.Event | None,
    deadline: RequestDeadline | None,
    terminal_outcome: TerminalOutcomeLatch | None,
    source: str,
) -> dict | None:
    """Return one canonical stop result without overwriting its first reason."""
    reason = request_stop_reason(cancel_event, deadline)
    if not reason:
        return None
    latch = terminal_outcome or TerminalOutcomeLatch()
    reason = latch.decide(reason, source=source)
    cancelled = reason == REASON_GENERATION_CANCELLED
    return {
        "response": "",
        "error": (
            "pipeline generation was cancelled"
            if cancelled else "request deadline exceeded"
        ),
        "cancelled": cancelled,
        "reason_code": reason,
        "metrics": {
            "distributed_used": False,
            "fallback": False,
            "terminal_reason": latch.snapshot(),
        },
    }


class LayerStageFrameTooLarge(RuntimeError):
    """★ 2026-10-07（DIST-NEXT-2）：层段 hidden 超出单帧预算。

    在 **dispatch 前**抛出（offer 尚未 reserve/execute）：超限的 hidden 装不进
    `MAX_MESSAGE_BYTES`，继续发送只会在执行完成后触发 `message_too_large`，
    表现为 stage 超时与回退 —— 那是过晚的契约发现，不是有效的 fail-closed。

    不可恢复：换一个更小的 prompt（更少 token）或更小的模型（更少 `n_embd`）
    才能重试；reason code 稳定，供上层直接收敛为具名失败。
    """

    code = "route_a_stage_frame_too_large"

    def __init__(
        self, *, node_id: str, wire_bytes: int, budget_bytes: int,
        n_tokens: int, n_embd: int,
    ) -> None:
        self.node_id = str(node_id)
        self.wire_bytes = int(wire_bytes)
        self.budget_bytes = int(budget_bytes)
        self.n_tokens = int(n_tokens)
        self.n_embd = int(n_embd)
        super().__init__(
            f"{self.code}:node={self.node_id}:wire={self.wire_bytes}"
            f":budget={self.budget_bytes}:n_tokens={self.n_tokens}"
            f":n_embd={self.n_embd}"
        )

#: 从节点心跳的**容忍上限**（秒）。★ 2026-10-05（DIST-2）：定义上移到
#: `scheduler_types.WORKER_HEARTBEAT_MAX_AGE`，让**容量规划**与 **readiness**
#: 共用同一阈值；此处保留 `_WORKER_HEARTBEAT_MAX_AGE` 别名，避免改动既有读取点。
_WORKER_HEARTBEAT_MAX_AGE = WORKER_HEARTBEAT_MAX_AGE

_PIPELINE_LIFECYCLE_STATE_KEY = "pipeline_config_lifecycle_v1"
_PIPELINE_LIFECYCLE_SCHEMA_VERSION = 4
_PIPELINE_RECOVERY_PHASES = frozenset({
    "preparing", "committing_local", "committing", "committing_global",
    "ready",
})


class _PipelineSupplyError(ValueError):
    """A registered segment must never fall back to a different artifact."""

    def __init__(self, field: str, *, changed: bool = False):
        self.reason_code = (
            "pipeline_supply_selection_changed" if changed
            else "pipeline_supply_identity_mismatch"
        )
        super().__init__(f"{self.reason_code}: {field}")


def _validate_pipeline_supply(
    selected: dict, *, model_id: str, model_sha256: str,
    total_layers: int, layer_range: tuple[int, int],
    tokenizer_sha256: str = "", model_type: str = "",
) -> None:
    manifest = selected["manifest"]
    expected = {
        "source_model_id": model_id,
        "source_model_sha256": model_sha256.lower(),
        "total_layers": total_layers,
        "layer_range": list(layer_range),
    }
    if tokenizer_sha256:
        expected["tokenizer_sha256"] = tokenizer_sha256.lower()
    if model_type:
        expected["model_type"] = model_type
    for field, value in expected.items():
        if manifest.get(field) != value:
            raise _PipelineSupplyError(field)
    directory = str(selected.get("directory", "") or "")
    model_path = str(selected.get("model_path", "") or "")
    if (
        not directory or not os.path.isdir(directory)
        or os.path.normcase(os.path.abspath(model_path))
        != os.path.normcase(os.path.abspath(directory))
    ):
        raise _PipelineSupplyError("directory")


def _kv_state_seq_len(past, model_type: str) -> tuple[int, int]:
    """从 KV 状态里取 `(槽位数, 已缓存序列长度)` —— **同时支持 tuple 与 Cache 对象**。

    ⚠️ 为什么需要它（`已知问题记录.md` #31 M4）：hybrid（`qwen3_5`）在层流水线上，
    `forward_layers` 返回的 `past_key_values` **tuple 是"有损兼容通道"** ——
    `linear_attention` 层的 recurrent state 在 tuple 里是 `None` 占位
    （实测：12 层里 **9 层是 `None`，且第 0 层就是**）⇒ 旧代码 `past_kv[0][0].shape`
    会 `None[0]` 直接 `TypeError` 硬崩。
    完整的 recurrent state 只在 `result["cache"]` 里（`cache.layers[i].recurrent_states`），
    所以这里**优先吃 cache 对象**，并**跳过 `None` 槽位**。

    序列长度取**第一个非空槽位**的对应维度（`qwen` 系是 `[1]`，其余是 `[2]`，与既有口径一致）；
    一个可读的槽位都没有时 `seq_len=0`（recurrent 槽位没有 `keys`，不参与 seq_len 判定）。
    """
    layers = getattr(past, "layers", None)      # Cache 对象（DynamicCache 等）
    entries = list(layers) if layers is not None else list(past or ())
    for entry in entries:
        if entry is None:
            continue
        keys = getattr(entry, "keys", None)     # Cache 的 KV 层
        if keys is not None and hasattr(keys, "shape"):
            shape = keys.shape
        elif hasattr(entry, "shape"):
            shape = entry.shape                 # tuple 的 (k, v) ⇒ 取 k
        elif isinstance(entry, (list, tuple)) and entry and hasattr(entry[0], "shape"):
            shape = entry[0].shape
        else:
            continue                            # recurrent 槽位：没有 keys，不提供 seq_len
        if len(shape) >= 3:
            return len(entries), int(shape[1] if model_type == "qwen" else shape[2])
    return len(entries), 0


def _prefer_cache_state(result) -> object:
    """★ #31 M4：优先取**完整**的 `cache` 对象，回落到 `past_key_values`（tuple）。

    与 `scripts/relay_experiment.py:1008-1010` 同一模式 —— hybrid 下 tuple 会丢 recurrent
    state（`linear_attention` 层是 `None` 占位）。
    """
    cache = result.get("cache") if hasattr(result, "get") else None
    if cache is not None:
        return cache
    return result["past_key_values"]


def _hidden_to_raw_f32(hidden) -> tuple[bytes, int, int]:
    """把 hidden 归一成 `(raw_le_f32_bytes, n_tokens, n_embd)` —— **不要求 torch**。

    线格式本来就是 raw little-endian f32（见 `_encode_relay_hidden` 与
    `transport_port.serialize_tensor`），所以这里要做的只有「二维 → CPU f32 连续 →
    bytes」。给 torch tensor 就沿用它的 `.detach().to(...)`（语义最准）；没有 torch 的
    边缘构建（免安装版）走 numpy —— 那种运行时里 hidden 本来就出自 `llama_cpp`/numpy，
    不该为了转一次字节序而要求装 torch。
    """
    if hidden is None:
        raise ValueError("layer stage requires a hidden tensor")
    detach = getattr(hidden, "detach", None)
    if callable(detach):
        torch = loaded_torch()
        if torch is None:
            raise ValueError(
                "收到 torch 张量但当前运行时没有 torch —— "
                "边缘构建请直接传 numpy 数组"
            )
        cpu = detach().to(device="cpu", dtype=torch.float32).contiguous()
        if cpu.ndim < 2:
            raise ValueError(
                "layer stage hidden tensor must have token and embedding dimensions"
            )
        shape = tuple(int(size) for size in cpu.shape)
        return (
            cpu.numpy().tobytes(),
            int(math.prod(shape[:-1])),
            int(shape[-1]),
        )
    import numpy as _np

    array = _np.asarray(hidden)
    if array.ndim < 2:
        raise ValueError(
            "layer stage hidden tensor must have token and embedding dimensions"
        )
    if array.dtype != _np.float32:
        array = array.astype(_np.float32)
    array = _np.ascontiguousarray(array)
    shape = tuple(int(size) for size in array.shape)
    return array.tobytes(), int(math.prod(shape[:-1])), int(shape[-1])


def _encode_relay_hidden(tensor) -> tuple[str, list[int]]:
    """Encode relay input as the explicit raw-f32 wire contract."""
    raw, n_tokens, n_embd = _hidden_to_raw_f32(tensor)
    shape = getattr(tensor, "shape", None)
    if shape is None:
        import numpy as _np
        shape = _np.asarray(tensor).shape
    shape = [int(size) for size in shape]
    if len(shape) < 2 or math.prod(shape[:-1]) != n_tokens or shape[-1] != n_embd:
        raise ValueError("relay hidden shape is inconsistent with raw f32 payload")
    return base64.b64encode(raw).decode("ascii"), shape


def _decode_relay_hidden(raw: bytes, shape: object):
    """Decode and validate a relay raw-f32 payload.

    返回 torch 张量（有 torch 时）或 numpy 数组（无 torch 的边缘构建）—— 两者在下游
    都按 `[tokens, embedding]` f32 使用。
    """
    if not isinstance(shape, list) or not shape or any(
        isinstance(size, bool) or not isinstance(size, int) or size <= 0
        for size in shape
    ):
        raise ValueError("relay hidden_shape must be a non-empty positive integer list")
    expected_items = 1
    for size in shape:
        expected_items *= size
    if len(raw) != expected_items * 4:
        raise ValueError(
            f"relay raw f32 length mismatch: bytes={len(raw)} expected={expected_items * 4}"
        )
    torch = loaded_torch()
    if torch is None:
        import numpy as _np

        return _np.frombuffer(memoryview(raw), dtype=_np.float32).reshape(shape).copy()
    return torch.frombuffer(memoryview(raw), dtype=torch.float32).reshape(shape).clone()


def _decode_layer_stage_hidden_output(
    output: object, *, n_tokens: int, n_embd: int,
):
    """Decode one v3 intermediate result into the pipeline tensor contract."""
    if not isinstance(output, dict):
        raise ValueError("layer stage output must be an object")
    encoded = output.get("hidden_out_f32")
    digest = output.get("hidden_out_sha256")
    if not isinstance(encoded, str) or not isinstance(digest, str):
        raise ValueError("layer stage hidden output requires bytes and digest")
    try:
        raw = base64.b64decode(encoded, validate=True)
    except Exception as exc:
        raise ValueError("layer stage hidden output is not valid base64") from exc
    expected_bytes = int(n_tokens) * int(n_embd) * 4
    if len(raw) != expected_bytes:
        raise ValueError(
            f"layer stage hidden length mismatch: {len(raw)} != {expected_bytes}"
        )
    actual_digest = hashlib.sha256(raw).hexdigest()
    if actual_digest != digest:
        raise ValueError("layer stage hidden digest mismatch")
    return _decode_relay_hidden(raw, [int(n_tokens), int(n_embd)])


def _layer_stage_result_to_pipeline_value(
    output: object, *, hidden_spec: dict, token_output: str | None = None, positions=None,
):
    """Map a v3 StageResult to the old pipeline step's explicit value shape."""
    if not isinstance(output, dict):
        raise ValueError("layer stage result must be an object")
    n_tokens = int(hidden_spec.get("n_tokens", 0) or 0)
    n_embd = int(hidden_spec.get("n_embd", 0) or 0)
    if n_tokens < 1 or n_embd < 1:
        raise ValueError("hidden_spec dimensions must be positive")
    from qlh_core.contracts.token_output import validate_token_output, ALL_POSITIONS
    multiple = validate_token_output(output, n_tokens=n_tokens, positions=positions,
        require_all=token_output == ALL_POSITIONS, allow_all=token_output == ALL_POSITIONS)
    if "hidden_out_f32" in output:
        return {
            "kind": "hidden",
            "hidden_states": _decode_layer_stage_hidden_output(
                output, n_tokens=n_tokens, n_embd=n_embd,
            ),
            "token_argmax": output.get("token_argmax"),
        }
    token = output.get("token_argmax")
    if isinstance(token, bool) or not isinstance(token, int) or token < 0:
        raise ValueError("tail layer stage result requires token_argmax")
    value = {"kind": "token", "token_argmax": token}
    if multiple:
        value.update({key: output[key] for key in
            ("token_argmax_by_position", "positions", "n_vocab", "n_tokens")})
    return value


def _assert_layer_stage_offer_fits_frame(
    *, node_id: str, n_tokens: int, n_embd: int, dtype: str = "float32",
    chunked_input: bool = False, binary_hidden: bool = False,
) -> None:
    """★ 2026-10-07（DIST-NEXT-2）：dispatch 前的层段 wire 大小预检。

    抽出为模块级纯函数（不依赖 `Scheduler` 状态），使两侧共用同一判据并可独立单测。

    ★ 2026-10-08（DIST-NEXT-2b）：`chunked_input=True` 表示目标 worker 声明了
    `stage_chunked_input` —— 此时**不得**在此拒绝。分片发生在 provider 层
    （`task_worker_adapter._maybe_send_stage_chunks`，先发 `chunk_count` 条 `stage_chunk`
    再发 offer），而本预检在 provider **之前**，无条件拒绝会让分片路径**永远走不到**：
    实测 ≈2475 tokens 的 prompt 得到 `route_a_stage_frame_too_large:…:wire=24160940`
    而非分片。未声明分片的对端仍保持 fail-closed。

    ★ 票 21a（COMM-WIRE-01）：`binary_hidden=True` 表示目标 worker 声明了
    `binary_hidden` ⇒ 帧走原始字节体，同样预算多装 ≈33%。判据顺序因此是
    **先按 base64 免费判**，只有超预算时才把二进制与分片两条放宽路径依次算进来 ——
    与 `chunked_input` 同因：能力探测含全量 capabilities 深拷贝，不能每步每段都付。
    """
    wire_bytes = hidden_wire_bytes(n_tokens, n_embd, dtype)
    budget_bytes = stage_payload_budget_bytes()
    if hidden_fits_stage_frame(n_tokens, n_embd, dtype):
        return
    # ★ 票 21a：base64 判不下、但走二进制体就装得下 ⇒ 放行（provider 会抽出体发送）。
    if binary_hidden and hidden_fits_stage_frame(n_tokens, n_embd, dtype, binary=True):
        logger.debug(
            "Route-A stage 帧超 base64 预算但声明 binary_hidden，改走二进制体: "
            "node=%s wire=%dB binary_wire=%dB budget=%dB n_tokens=%d",
            node_id, wire_bytes,
            hidden_wire_bytes(n_tokens, n_embd, dtype, binary=True),
            budget_bytes, n_tokens,
        )
        return
    if chunked_input:
        # ★ 2026-10-08（DIST-NEXT-2b）：声明分片只放宽**单帧**预算。分片本身仍有总量上限
        #   （`MAX_STAGE_CHUNKS × STAGE_CHUNK_BYTES` 原始字节），越过它必须**仍在此**
        #   fail-closed —— 否则会在 provider 的分片计划中途才抛
        #   （`stage_payload_too_large: payload needs N chunks, at most 64 are allowed`），
        #   又退回成"过晚的契约发现"。
        from task_worker_protocol import (
            MAX_STAGE_PAYLOAD_BYTES,
            _base64_wire_length,
        )

        chunked_wire_limit = int(_base64_wire_length(int(MAX_STAGE_PAYLOAD_BYTES)))
        if wire_bytes <= chunked_wire_limit:
            logger.debug(
                "Route-A stage 帧超单帧预算但对端声明分片，交由 provider 分片发送: "
                "node=%s wire=%dB budget=%dB n_tokens=%d",
                node_id, wire_bytes, budget_bytes, n_tokens,
            )
            return
        logger.warning(
            "Route-A stage 分片总量超限: node=%s n_tokens=%d n_embd=%d wire=%dB "
            "max_chunked_wire=%dB",
            node_id, n_tokens, n_embd, wire_bytes, chunked_wire_limit,
        )
        raise LayerStageFrameTooLarge(
            node_id=node_id,
            wire_bytes=wire_bytes,
            budget_bytes=chunked_wire_limit,
            n_tokens=n_tokens,
            n_embd=n_embd,
        )
    logger.warning(
        "Route-A stage 帧预算不足: node=%s n_tokens=%d n_embd=%d dtype=%s "
        "wire=%dB budget=%dB max_tokens=%d",
        node_id, n_tokens, n_embd, dtype, wire_bytes, budget_bytes,
        max_hidden_tokens(n_embd, dtype),
    )
    raise LayerStageFrameTooLarge(
        node_id=node_id,
        wire_bytes=wire_bytes,
        budget_bytes=budget_bytes,
        n_tokens=n_tokens,
        n_embd=n_embd,
    )


def _node_declares_stage_chunked_input(control: Any, node_id: str) -> bool:
    """★ 2026-10-08（DIST-NEXT-2b）：该节点是否声明 `stage_chunked_input`。

    声明 ⇒ 超预算的 hidden 由 provider 切 `stage_chunk` 分片发送，预检不得提前拒绝；
    未声明（含查询失败）⇒ 返回 False，维持 fail-closed 的既有语义。
    """
    return _node_capabilities(control, node_id).get("stage_chunked_input") is True


def _node_declares_binary_hidden(control: Any, node_id: str) -> bool:
    """★ 票 21a（COMM-WIRE-01）：该节点是否声明 `binary_hidden`。

    声明 ⇒ 该节点能收 `format:"binary"` 的帧（帧体为原始字节），单帧容量 +≈33%；
    未声明（含查询失败）⇒ False，维持 fail-closed（继续走 base64 或分片）。
    """
    return _node_capabilities(control, node_id).get("binary_hidden") is True


def _node_capabilities(control: Any, node_id: str) -> dict:
    """一次取回该节点的自荐能力字典；不可得 ⇒ `{}`。

    ⚠️ `control.status(role="master")` 内含**全量 worker capabilities 深拷贝**
    （见 :2521 的注释），因此**多个能力判据必须共用同一次查询** —— 分别调用
    `_node_declares_*` 会各付一次深拷贝。
    """
    if control is None:
        return {}
    try:
        status = control.status(role="master")
    except Exception:
        logger.debug("查询 task worker 能力失败: node=%s", node_id, exc_info=True)
        return {}
    worker_ids = [
        item.get("node_id")
        for item in ((status or {}).get("workers") or [])
        if isinstance(item, dict)
    ]
    for worker in (status or {}).get("workers", []) or []:
        if not isinstance(worker, dict) or worker.get("node_id") != node_id:
            continue
        capabilities = worker.get("capabilities")
        declared_chunked = (
            isinstance(capabilities, dict)
            and capabilities.get("stage_chunked_input") is True
        )
        # ★ 2026-10-08（诊断，定位后降级为 debug）：分辨「worker 不在控制面」与
        #   「hello 没带该键」——2b 真机回归已用它确认 Android 侧 `declared=True`
        #   且分片真实生效（`task_worker_stage_chunks_sent chunks=18`）。
        logger.debug(
            "event=stage_chunked_probe node=%s declared=%s caps_keys=%s worker_ids=%s",
            node_id, declared_chunked,
            sorted(capabilities.keys()) if isinstance(capabilities, dict) else None,
            worker_ids,
        )
        if not isinstance(capabilities, dict):
            return {}
        # Frame admission must use the same accepted ACK bit as the sender.
        # A coordinator-wide disable still preserves the raw HELLO inventory.
        return {
            **capabilities,
            "binary_hidden": (
                worker.get("binary_hidden") is True
                and capabilities.get("binary_hidden") is True
            ),
        }
    return {}


from worker_prepare_coordinator import WorkerPrepareCoordinatorMixin


class SchedulerPipelineMixin(WorkerPrepareCoordinatorMixin):
    def _selected_pipeline_safetensors(self) -> dict | None:
        from model_supply_worker import get_selected_model

        worker = getattr(self, "_model_supply_worker", None)
        selected = get_selected_model(getattr(worker, "root", None))
        if selected and selected.get("manifest", {}).get("format") == "safetensors":
            return selected
        return None

    def _pipeline_lifecycle_snapshot_locked(self) -> dict | None:
        """Return a bounded JSON-safe transaction record for restart recovery."""
        transaction = self._pipeline_load_transaction
        cleanup_ledger = getattr(self, "_pipeline_model_cleanup_ledger", {})
        recovery = getattr(self, "_pipeline_recovery_state", None)
        recovering = not isinstance(transaction, dict) and isinstance(recovery, dict)
        if not isinstance(transaction, dict) and not cleanup_ledger and not recovering:
            return None
        transaction = transaction if isinstance(transaction, dict) else {}
        plan = transaction.get("plan") if isinstance(transaction.get("plan"), dict) else {}
        assignments = []
        for item in plan.get("assignments", []):
            if not isinstance(item, dict):
                continue
            assignments.append({
                key: item[key]
                for key in (
                    "node_id", "start_layer", "end_layer", "layers_count",
                    "execution", "stage_type", "has_embedding", "has_lm_head",
                    "worker_prepare_identity",
                )
                if key in item
            })
        cleanup_records = []
        for record in cleanup_ledger.values():
            if not isinstance(record, dict):
                continue
            payload = record.get("payload")
            cleanup_records.append({
                "cleanup_id": str(record.get("cleanup_id", "") or ""),
                "legacy_identity": bool(record.get("legacy_identity", False)),
                "node_id": str(record.get("node_id", "") or ""),
                "candidate_config_id": str(
                    record.get("candidate_config_id", "") or ""
                ),
                "assignment_id": str(record.get("assignment_id", "") or ""),
                "generation": int(record.get("generation", 0) or 0),
                "action": str(record.get("action", "") or ""),
                "created_at": float(record.get("created_at", 0.0) or 0.0),
                "payload": dict(payload) if isinstance(payload, dict) else {},
                **({key: record[key] for key in ("process_boot_id", "process_id", "process_start_time") if key in record}),
            })
        if recovering:
            # Cleanup replay runs before a fresh transaction exists. Preserve
            # the persisted model and assignments through every ledger update;
            # otherwise a second restart loses the authoritative recovery input.
            return {
                **recovery,
                "schema_version": recovery.get("schema_version", _PIPELINE_LIFECYCLE_SCHEMA_VERSION),
                "cleanup_ledger": cleanup_records,
                "updated_at": time.time(),
            }
        return {
            "schema_version": _PIPELINE_LIFECYCLE_SCHEMA_VERSION,
            "config_id": str(transaction.get("config_id", "") or ""),
            "generation": int(transaction.get("generation", 0) or 0),
            "phase": str(transaction.get("phase", "") or ""),
            "published": bool(transaction.get("published") or transaction.get("phase") == "ready"),
            "model_id": str(plan.get("model_id", "") or ""),
            "model_type": str(plan.get("model_type", "") or ""),
            "model_sha256": str(plan.get("model_sha256", "") or ""),
            "total_layers": int(plan.get("total_layers", 0) or 0),
            "tokenizer_sha256": str(plan.get("tokenizer_sha256", "") or ""),
            "quant_type": str(plan.get("quant_type", "") or ""),
            "plan_id": str(plan.get("plan_id", "") or ""),
            "worker_ids": sorted(
                str(value) for value in transaction.get("worker_ids", set())
                if str(value)
            ),
            "prepared_nodes": sorted(
                str(value) for value in transaction.get("prepared_nodes", set())
                if str(value)
            ),
            "ready_nodes": sorted(
                str(value) for value in transaction.get("ready_nodes", set())
                if str(value)
            ),
            "assignments": assignments,
            "local_assignment": transaction.get("local_assignment"),
            # ★ 2026-10-26（LOCAL-EXECUTORS-01）：**同机附加执行器必须进快照**。
            #
            #   这是"同机跨框架两段并存"的最后一环，也是最隐蔽的一环：`freeze` 明明
            #   成功（日志 `local_executor_freeze_done frozen=['cpu']`），但执行期
            #   `execution_segments local_executors=[]` —— 因为
            #   `qlh_core/control/deployment.py:293` 是从 **lifecycle（本快照）** 读
            #   `local_executors` 再写进 deployment 的，而这里没输出该字段
            #   ⇒ `lifecycle.get(...)` 为 None ⇒ 下游跳过写入 ⇒ deployment 里没有
            #   ⇒ 执行期段列表缺 CPU 段的 `[8,24)` ⇒ 覆盖不连续
            #   ⇒ `/api/chat` 503 `execution_segment_coverage_invalid`。
            #   实测（2026-10-26 12:58）：plan 四段齐全、供给 ready、freeze 成功，
            #   唯独执行期只有 3 段。
            #   缺省 `{}`（而非 None）：下游 `if ... is not None` 才会写入，
            #   与 `local_assignment` 的语义一致。
            "local_executors": dict(transaction.get("local_executors") or {}),
            "local_receipt": transaction.get("local_receipt"),
            "candidate_id": transaction.get("candidate_id", ""),
            "process_identity": transaction.setdefault("process_identity", self._model_control_process_identity()),
            "cleanup_ledger": cleanup_records,
            "updated_at": time.time(),
        }

    def _model_control_process_identity(self):
        import psutil
        owner = getattr(self, "_model_control", None)
        return {"process_boot_id": owner.boot_id if owner else "",
                "process_id": os.getpid(), "process_start_time": psutil.Process(os.getpid()).create_time()}

    def _model_control_authority(self):
        fence = getattr(self, "_control_fence", None)
        authority = getattr(fence, "authority", None)
        if fence is not None and getattr(fence, "required", False):
            certificate = authority.current_certificate() if authority is not None else None
            fence.admit(certificate, action="model_deployment", source="scheduler")
        identity = authority.authority_identity() if authority is not None else None
        if identity is not None:
            from dataclasses import asdict
            return asdict(identity)
        return {"mode": "static", "node_id": self.get_effective_node_id(), "role": self._effective_role()}

    def _model_control_connections(self, transaction=None):
        transaction = transaction or self._pipeline_load_transaction or {}
        return {node: self._assignment_connection_generation(node)
                for node in transaction.get("worker_ids", ())}

    def _model_control_input_revisions(self, transaction):
        from qlh_core.contracts.deployment import digest
        plan = transaction.get("plan", {})
        catalog = getattr(self, "_catalog", None)
        model_id = plan.get("model_id", "")
        catalog_revision = catalog.snapshot(model_id).registry_revision if catalog is not None and model_id else 0
        capabilities = {}
        for node in transaction.get("worker_ids", ()):
            worker = self._task_worker_control.worker_snapshot(node) or {}
            value = worker.get("capabilities") or {}
            capabilities[node] = digest({key: value.get(key) for key in ("engine_capabilities", "load_spec_version", "engines")})
        return {"catalog": catalog_revision, "capabilities": capabilities,
                "resources": digest([{key: item.get(key) for key in ("node_id", "capacity_bytes", "required_bytes", "reserve_bytes", "execution_device", "capacity_source")}
                                     for item in plan.get("assignments", [])])}

    def _save_model_control_lifecycle(self, state):
        transaction = self._pipeline_load_transaction or {}
        owner = self._model_control
        owner.start()
        inputs = self._model_control_input_revisions(transaction)
        frozen = transaction.setdefault("model_control_inputs", inputs)
        if frozen != inputs and state.get("phase") not in {"aborted", "invalidated", "rejected"}:
            raise RuntimeError("deployment_input_revision_changed")
        state["input_revisions"] = frozen
        expected = transaction.get("expected", {})
        receipts = dict(transaction.get("acks", {}))
        # Legacy transport stores validated receipts in its assignment registry.
        # Copy exact current assignment observations into the durable owner.
        for node in transaction.get("worker_ids", ()):
            if not transaction.get("candidate_id") and node not in receipts:
                item = self._worker_assignments.expected(node)
                ack = self._worker_assignments.acknowledgement(node)
                if item and ack:
                    expected = {**expected, node: item}
                    receipts[node] = ack
        return owner.save_execution(state, expected=expected,
            acknowledgements=receipts, authority=self._model_control_authority(),
            connections=self._model_control_connections(transaction),
            recovery_input=self._pipeline_load_transaction is None and bool(getattr(self, "_pipeline_recovery_state", None)))

    def _persist_pipeline_lifecycle_locked(self) -> None:
        """Persist lifecycle state while the transaction lock is held."""
        if (
            (not getattr(self, "_running", False) and getattr(self, "_model_control", None) is None)
            or self._effective_role() != "master"
        ):
            return
        if not self._pipeline_recovery_validated:
            # A partial ledger cannot replace the original unvalidated record.
            # An exact ACK may settle one known debt, but cannot erase evidence
            # of obligations that failed restart validation.
            return
        state = self._pipeline_lifecycle_snapshot_locked()
        if state is None:
            return
        try:
            from local_store import set_local_setting

            if getattr(self, "_model_control", None) is not None:
                self._save_model_control_lifecycle(state)
            else:
                set_local_setting(_PIPELINE_LIFECYCLE_STATE_KEY, state)
            self._pipeline_lifecycle_persist_ok = True
            if (self._pipeline_recovery_pending
                    and self._pipeline_recovery_failure == "pipeline_lifecycle_persist_failed"):
                self._pipeline_recovery_failure = "pipeline_recovery_pending"
        except Exception:
            self._pipeline_lifecycle_persist_ok = False
            self._pipeline_recovery_pending = True
            self._pipeline_recovery_failure = "pipeline_lifecycle_persist_failed"
            logger.warning("failed to persist pipeline lifecycle state", exc_info=True)

    def _record_pipeline_model_cleanup_locked(
        self, *, node_id: str, candidate_config_id: str,
        generation: int, action: str, assignment_id: str = "",
        payload: dict | None = None,
    ) -> dict:
        """Create or return one durable terminal cleanup obligation."""
        node_id = str(node_id or "")
        candidate_config_id = str(candidate_config_id or "")
        action = str(action or "")
        assignment_id = str(assignment_id or "")
        if not node_id or not candidate_config_id:
            raise ValueError("pipeline cleanup requires node and candidate config")
        if action not in {"abort", "finalize", "release"}:
            raise ValueError(f"unsupported pipeline cleanup action: {action}")
        ledger = self._pipeline_model_cleanup_ledger
        for record in ledger.values():
            if (
                record.get("node_id") == node_id
                and record.get("candidate_config_id") == candidate_config_id
                and record.get("assignment_id", "") == assignment_id
                and record.get("action") == action
            ):
                return record
        cleanup_id = uuid.uuid4().hex
        wire_payload = dict(payload or {})
        wire_payload.update({
            "node_id": node_id,
            "config_id": cleanup_id,
            "generation": int(generation or 0),
            "assignment_id": assignment_id,
            "cleanup_action": action,
            "candidate_config_id": candidate_config_id,
        })
        record = {
            "cleanup_id": cleanup_id,
            "node_id": node_id,
            "candidate_config_id": candidate_config_id,
            "assignment_id": assignment_id,
            "generation": int(generation or 0),
            "action": action,
            "state": "pending",
            "attempts": 1,
            "next_retry": time.monotonic() + 5.0,
            "created_at": time.time(),
            "payload": wire_payload,
        }
        if node_id == "master":
            record.update(self._model_control_process_identity())
        ledger[cleanup_id] = record
        return record

    def _complete_pipeline_model_cleanup(
        self, cleanup_id: str, *, persist: bool = True, evidence: dict | None = None,
    ) -> bool:
        with self._layer_config_lock:
            debt = self._pipeline_model_cleanup_ledger.get(str(cleanup_id or ""))
            owner = getattr(self, "_model_control", None)
            if debt is not None and owner is not None:
                owner.settle_cleanup(str(cleanup_id), evidence=evidence or {"kind": "receipt", "action": debt["action"],
                    "identity": debt.get("payload", {}).get("worker_prepare_identity")})
            removed = self._pipeline_model_cleanup_ledger.pop(
                str(cleanup_id or ""), None,
            )
            if removed is not None and persist:
                self._persist_pipeline_lifecycle_locked()
        return removed is not None

    def _dispatch_pipeline_model_cleanup(self, record: dict) -> bool:
        """Attempt one cleanup without dropping its ledger entry on failure."""
        cleanup_id = str(record.get("cleanup_id", "") or "")
        node_id = str(record.get("node_id", "") or "")
        action = str(record.get("action", "") or "")
        candidate_config_id = str(
            record.get("candidate_config_id", "") or ""
        )
        if not cleanup_id or not node_id or not candidate_config_id:
            return False
        owner = getattr(self, "_model_control", None)
        if owner is not None and cleanup_id not in owner.snapshot()["cleanup"]:
            return False
        if node_id == "master":
            if owner is not None and record.get("process_boot_id") != owner.boot_id:
                import psutil
                if not all(record.get(key) for key in ("process_boot_id", "process_id", "process_start_time")):
                    return False
                try:
                    old = psutil.Process(record["process_id"])
                    ended = old.create_time() != record["process_start_time"]
                except psutil.NoSuchProcess:
                    ended = True
                except (psutil.Error, OSError):
                    return False
                if not ended:
                    return False
                self._complete_pipeline_model_cleanup(cleanup_id, evidence={"kind": "process_ended",
                    **{key: record[key] for key in ("process_boot_id", "process_id", "process_start_time")},
                    "boot_id": record["process_boot_id"]})
                return True
            method = getattr(
                self._host,
                (
                    "abort_pipeline_candidate"
                    if action == "abort"
                    else "finalize_pipeline_candidate"
                ),
                None,
            )
            if not callable(method):
                return False
            try:
                method(candidate_config_id)
            except Exception:
                logger.warning(
                    "主节点候选模型终局清理失败: action=%s candidate=%s",
                    action, candidate_config_id, exc_info=True,
                )
                return False
            self._complete_pipeline_model_cleanup(cleanup_id)
            return True
        server = self._tcp_server
        if server is None or not getattr(server, "_running", False):
            return False
        try:
            if self._send_pipeline_worker_config(
                node_id, dict(record.get("payload", {})),
            ) is False:
                return False
        except Exception:
            logger.warning(
                "发送候选终局清理失败: action=%s node=%s candidate=%s",
                action, node_id, candidate_config_id, exc_info=True,
            )
            return False
        with self._layer_config_lock:
            current = self._pipeline_model_cleanup_ledger.get(cleanup_id)
            if current is not None:
                current["state"] = "pending"
        return True

    def _pending_pipeline_model_cleanup_for_node(
        self, node_id: str,
    ) -> list[dict]:
        with self._layer_config_lock:
            records = [
                dict(record)
                for record in self._pipeline_model_cleanup_ledger.values()
                if record.get("node_id") == node_id
            ]
        return sorted(records, key=lambda item: item.get("created_at", 0.0))

    def _replay_pipeline_model_cleanup_for_node(self, node_id: str) -> bool:
        """Replay terminal cleanup before a reconnected node is readmitted."""
        records = self._pending_pipeline_model_cleanup_for_node(node_id)
        if not records:
            return False
        # Candidate transactions are serialized per ModelHost.  Replaying the
        # oldest debt first preserves that ordering if multiple durable records
        # ever survive a long disconnect.
        self._dispatch_pipeline_model_cleanup(records[0])
        return True

    def _ack_pipeline_model_cleanup(self, client_id: str, data: dict, *, native_ack: dict | None = None) -> bool:
        cleanup_id = str(data.get("config_id", "") or "")
        action = str(data.get("cleanup_action", "") or "")
        assignment_id = str(data.get("assignment_id", "") or "")
        status = str(data.get("status", "") or "")
        candidate_config_id = str(data.get("candidate_config_id", "") or "")
        finalized_config_id = str(data.get("finalized_config_id", "") or "")
        aborted_config_id = str(data.get("aborted_config_id", "") or "")
        if action == "finalize":
            identity_valid = bool(
                status == "finalized"
                and candidate_config_id
                and finalized_config_id == candidate_config_id
                and not aborted_config_id
            )
        elif action == "release":
            identity_valid = bool(status == "released" and data.get("release") is True
                                  and candidate_config_id and not aborted_config_id and not finalized_config_id)
        elif action == "abort":
            identity_valid = bool(
                status == "released"
                and candidate_config_id
                and aborted_config_id == candidate_config_id
                and not finalized_config_id
            )
        else:
            identity_valid = False
        if not identity_valid:
            logger.warning(
                "ignoring malformed candidate cleanup ACK: node=%s "
                "cleanup=%s action=%s status=%s candidate=%s",
                client_id, cleanup_id, action, status, candidate_config_id,
            )
            return False
        try:
            generation = int(data.get("generation", 0) or 0)
        except (TypeError, ValueError):
            return False
        with self._layer_config_lock:
            record = self._pipeline_model_cleanup_ledger.get(cleanup_id)
            if not isinstance(record, dict):
                return False
            native_identity = record.get("payload", {}).get("worker_prepare_identity")
            if isinstance(native_identity, dict):
                from worker_prepare_contract import validate_ack, WorkerPrepareError
                try:
                    checked = validate_ack(native_ack)
                except WorkerPrepareError:
                    return False
                if (checked["identity"] != native_identity or checked["phase"] != action
                        or checked.get("cleanup_id") != cleanup_id or checked["status"] == "error"):
                    return False
            if (
                record.get("node_id") != client_id
                or record.get("action") != action
                or record.get("candidate_config_id") != candidate_config_id
                or str(record.get("assignment_id", "") or "")
                != assignment_id
                or int(record.get("generation", 0) or 0) != generation
            ):
                logger.warning(
                    "忽略不匹配的候选终局 ACK: node=%s cleanup=%s action=%s candidate=%s",
                    client_id, cleanup_id, action, candidate_config_id,
                )
                return False
            owner = getattr(self, "_model_control", None)
            if owner is not None:
                owner.settle_cleanup(cleanup_id, evidence={"kind": "receipt", "identity": native_identity, "action": action})
            self._pipeline_model_cleanup_ledger.pop(cleanup_id, None)
            transaction = self._pipeline_load_transaction
            if isinstance(transaction, dict):
                pending = set(transaction.get("finalize_pending", set()))
                if action == "finalize":
                    pending.discard(client_id)
                    transaction["finalize_pending"] = pending
                    transaction.get("finalize_retry_state", {}).pop(
                        client_id, None,
                    )
                    if not pending:
                        transaction["finalize_payloads"] = {}
                        transaction["finalize_retry_state"] = {}
            self._persist_pipeline_lifecycle_locked()
            has_more = any(
                item.get("node_id") == client_id
                for item in self._pipeline_model_cleanup_ledger.values()
            )
            transaction_active = bool(
                isinstance(transaction, dict)
                and str(transaction.get("phase", "") or "") in {
                    "preparing", "committing_local", "committing",
                    "committing_global",
                }
            )
        logger.info(
            "worker 候选终局清理已确认: node=%s action=%s candidate=%s",
            client_id, action, candidate_config_id,
        )
        if has_more:
            self._replay_pipeline_model_cleanup_for_node(client_id)
        elif not transaction_active:
            try:
                self.push_layer_config_to_clients()
            except Exception:
                logger.debug(
                    "候选清理完成后重新同步层配置失败", exc_info=True,
                )
        return True

    def _load_pipeline_recovery_state(self) -> None:
        """Arm a restart fence from the last active transaction."""
        self._pipeline_recovery_validated = False
        try:
            from local_store import get_local_setting

            owner = getattr(self, "_model_control", None)
            owned_state = owner.start() if owner is not None else None
            raw = owned_state.get("lifecycle") if owned_state else None
            if raw is None:
                # One-way import of recovery evidence. Subsequent writes use only
                # the model-control domain and never update the legacy setting.
                raw = get_local_setting(_PIPELINE_LIFECYCLE_STATE_KEY, None)
        except Exception:
            self._pipeline_lifecycle_persist_ok = False
            self._pipeline_recovery_pending = True
            self._pipeline_recovery_failure = "pipeline_recovery_state_unavailable"
            logger.warning("failed to load pipeline lifecycle state", exc_info=True)
            return
        self._pipeline_lifecycle_persist_ok = True
        if raw is None:
            self._pipeline_recovery_validated = True
            return
        try:
            schema_version = int(raw.get("schema_version", 0) or 0)
        except (AttributeError, TypeError, ValueError):
            schema_version = 0
        if not isinstance(raw, dict) or schema_version not in {1, 2, 3, 4}:
            self._pipeline_lifecycle_persist_ok = False
            self._pipeline_recovery_pending = True
            self._pipeline_recovery_failure = "pipeline_recovery_state_invalid"
            return
        self._pipeline_recovery_state = dict(raw)
        restored_cleanup = {}
        raw_cleanup = raw.get("cleanup_ledger", [])
        invalid_cleanup = not isinstance(raw_cleanup, list)
        for item in raw_cleanup if isinstance(raw_cleanup, list) else []:
            if not isinstance(item, dict):
                invalid_cleanup = True
                break
            cleanup_id = str(item.get("cleanup_id", "") or "")
            node_id = str(item.get("node_id", "") or "")
            candidate_config_id = str(
                item.get("candidate_config_id", "") or ""
            )
            assignment_id = str(item.get("assignment_id", "") or "")
            action = str(item.get("action", "") or "")
            payload = item.get("payload")
            if node_id == "master":
                if (not cleanup_id or not candidate_config_id or action not in {"abort", "finalize", "release"}
                        or not isinstance(payload, dict)):
                    invalid_cleanup = True
                    break
                restored_cleanup[cleanup_id] = {**item, "state": "unknown", "attempts": 0, "next_retry": 0.0}
                continue
            if (
                not cleanup_id
                or not node_id
                or not candidate_config_id
                or action not in {"abort", "finalize", "release"}
                or not isinstance(payload, dict)
            ):
                invalid_cleanup = True
                break
            try:
                generation = int(item.get("generation", 0) or 0)
                created_at = float(item.get("created_at", 0.0) or 0.0)
            except (TypeError, ValueError):
                invalid_cleanup = True
                break
            canonical_payload = dict(payload)
            if "legacy_identity" in item and type(item["legacy_identity"]) is not bool:
                invalid_cleanup = True
                break
            legacy_identity = schema_version < 4 or item.get("legacy_identity") is True
            if legacy_identity and "worker_prepare_identity" in canonical_payload:
                invalid_cleanup = True
                break
            if legacy_identity:
                canonical_payload.update({
                    "node_id": node_id,
                    "config_id": cleanup_id,
                    "generation": generation,
                    "assignment_id": "",
                    "cleanup_action": action,
                    "candidate_config_id": candidate_config_id,
                })
                if action == "finalize":
                    canonical_payload["finalized_config_id"] = candidate_config_id
                    canonical_payload.pop("aborted_config_id", None)
                else:
                    canonical_payload["aborted_config_id"] = candidate_config_id
                    canonical_payload.pop("finalized_config_id", None)
                assignment_id = ""
            else:
                try:
                    payload_generation = int(
                        canonical_payload.get("generation", 0) or 0
                    )
                except (TypeError, ValueError):
                    invalid_cleanup = True
                    break
                payload_identity_matches = (
                    bool(assignment_id)
                    and str(canonical_payload.get("node_id", "") or "") == node_id
                    and str(canonical_payload.get("config_id", "") or "") == cleanup_id
                    and payload_generation == generation
                    and str(canonical_payload.get("assignment_id", "") or "")
                    == assignment_id
                    and str(canonical_payload.get("cleanup_action", "") or "")
                    == action
                    and str(canonical_payload.get("candidate_config_id", "") or "")
                    == candidate_config_id
                )
                if action == "release":
                    payload_identity_matches = (payload_identity_matches
                                                and canonical_payload.get("release") is True
                                                and isinstance(canonical_payload.get("worker_prepare_identity"), dict))
                elif action == "finalize":
                    payload_identity_matches = (
                        payload_identity_matches
                        and str(canonical_payload.get("finalized_config_id", "") or "")
                        == candidate_config_id
                        and not canonical_payload.get("aborted_config_id")
                    )
                else:
                    payload_identity_matches = (
                        payload_identity_matches
                        and str(canonical_payload.get("aborted_config_id", "") or "")
                        == candidate_config_id
                        and not canonical_payload.get("finalized_config_id")
                    )
                if not payload_identity_matches:
                    invalid_cleanup = True
                    break
            restored_cleanup[cleanup_id] = {
                "cleanup_id": cleanup_id,
                "node_id": node_id,
                "candidate_config_id": candidate_config_id,
                "assignment_id": assignment_id,
                "legacy_identity": legacy_identity,
                "generation": generation,
                "action": action,
                "state": "await_reconnect",
                "attempts": 0,
                "next_retry": 0.0,
                "created_at": created_at,
                "payload": canonical_payload,
            }
        if invalid_cleanup:
            self._pipeline_lifecycle_persist_ok = False
            self._pipeline_recovery_pending = True
            self._pipeline_recovery_failure = "pipeline_recovery_state_invalid"
            logger.error("pipeline recovery cleanup ledger is inconsistent")
            return
        self._pipeline_model_cleanup_ledger.update(restored_cleanup)
        phase = str(raw.get("phase", "") or "")
        if phase in _PIPELINE_RECOVERY_PHASES or raw.get("published"):
            # A crashed master cannot reuse remote ready evidence. Retire the
            # old native generation before publishing a fresh one, including
            # candidates whose prepare completed before the crash.
            action = "release" if phase == "ready" or raw.get("published") else "abort"
            with self._layer_config_lock:
                for assignment in raw.get("assignments", []):
                    identity = assignment.get("worker_prepare_identity") if isinstance(assignment, dict) else None
                    if not isinstance(identity, dict):
                        continue
                    from worker_prepare_contract import validate_identity, WorkerPrepareError
                    try:
                        identity = validate_identity(identity)
                    except WorkerPrepareError:
                        self._pipeline_lifecycle_persist_ok = False
                        self._pipeline_recovery_pending = True
                        self._pipeline_recovery_failure = "pipeline_recovery_state_invalid"
                        return
                    if (identity["config_id"] != raw.get("config_id") or identity["generation"] != raw.get("generation")
                            or identity["node_id"] != assignment.get("node_id")):
                        self._pipeline_lifecycle_persist_ok = False
                        self._pipeline_recovery_pending = True
                        self._pipeline_recovery_failure = "pipeline_recovery_state_invalid"
                        return
                    payload = {"worker_prepare_identity": identity, "release": True}
                    if action == "abort":
                        payload.update(abort=True, aborted_config_id=identity["config_id"])
                    self._record_pipeline_model_cleanup_locked(
                        node_id=identity["node_id"], candidate_config_id=identity["config_id"],
                        generation=int(raw["generation"]), action=action,
                        assignment_id=identity["assignment_id"], payload=payload,
                    )
        if (raw.get("local_assignment") and raw.get("candidate_id")
                and (phase in {"committing_local", "committing", "committing_global", "ready"} or raw.get("published"))):
            action = "release" if raw.get("published") or phase == "ready" else "abort"
            with self._layer_config_lock:
                old = next((item for item in self._pipeline_model_cleanup_ledger.values()
                            if item.get("node_id") == "master" and item.get("candidate_config_id") == raw["candidate_id"]), None)
                if old is None:
                    record = self._record_pipeline_model_cleanup_locked(node_id="master",
                        candidate_config_id=raw["candidate_id"], generation=int(raw.get("generation", 0)), action=action)
                    for key in ("process_boot_id", "process_id", "process_start_time"):
                        record[key] = (raw.get("process_identity") or {}).get(key)
        if phase in _PIPELINE_RECOVERY_PHASES:
            persisted_sha256 = str(raw.get("model_sha256", "") or "").lower()
            if (
                len(persisted_sha256) != 64
                or any(char not in "0123456789abcdef" for char in persisted_sha256)
            ):
                # Schema v1 did not require artifact identity.  Guessing
                # between a directory and a GGUF that share one model_id can
                # silently restore different bytes, so old incomplete state
                # is fenced for an explicit model reload instead.
                self._pipeline_lifecycle_persist_ok = False
                self._pipeline_recovery_pending = True
                self._pipeline_recovery_failure = "pipeline_recovery_state_invalid"
                logger.error(
                    "pipeline recovery state lacks an authoritative model digest"
                )
                return
            self._pipeline_recovery_pending = True
            self._pipeline_recovery_failure = "pipeline_recovery_pending"
            logger.warning(
                "pipeline restart recovery fenced until authoritative republish: "
                "config=%s generation=%s phase=%s",
                raw.get("config_id", ""), raw.get("generation", ""), phase,
            )

        self._pipeline_recovery_validated = True
        if self._pipeline_model_cleanup_ledger:
            with self._layer_config_lock:
                self._persist_pipeline_lifecycle_locked()

    def _restore_pipeline_model_for_recovery(self) -> bool:
        """Restore distributed-only model metadata before workers reconnect.

        The lifecycle record intentionally contains no absolute path. Prefer
        the current verified supply and its persisted local assignment; ordinary
        registry resolution also accepts the historical directory-name identity.
        Prepare only metadata without materializing a full model.
        """
        if not self._pipeline_recovery_pending:
            return True
        if not self._pipeline_recovery_validated:
            return False
        state = self._pipeline_recovery_state
        if not isinstance(state, dict):
            return False
        model_id = str(state.get("model_id", "") or "").strip()
        if not model_id:
            self._pipeline_recovery_failure = "pipeline_recovery_model_identity_missing"
            return False
        supplied = self._selected_pipeline_safetensors()
        supply_range = None
        if supplied is not None:
            try:
                local_ids = {"master", self.get_effective_node_id()}
                assignment = next((
                    item for item in state.get("assignments", [])
                    if isinstance(item, dict) and item.get("node_id") in local_ids
                ), None)
                if assignment is None:
                    raise _PipelineSupplyError("layer_range_missing")
                supply_range = (
                    int(assignment["start_layer"]), int(assignment["end_layer"]),
                )
                _validate_pipeline_supply(
                    supplied, model_id=model_id,
                    model_sha256=str(state.get("model_sha256", "") or ""),
                    # Earlier lifecycle records did not persist total_layers.
                    total_layers=int(
                        state.get("total_layers", 0)
                        or supplied["manifest"]["total_layers"]
                    ),
                    layer_range=supply_range,
                    tokenizer_sha256=str(state.get("tokenizer_sha256", "") or ""),
                    model_type=str(state.get("model_type", "") or "").lower(),
                )
            except (ValueError, TypeError, KeyError) as exc:
                self._pipeline_recovery_failure = getattr(
                    exc, "reason_code", "pipeline_supply_identity_mismatch",
                )
                logger.error("pipeline supply recovery rejected: %s", exc)
                return False
        if self._get_active_pipeline_model_info():
            return True

        model_paths: list[str] = []
        if supplied is not None:
            model_paths.append(supplied["directory"])
        else:
            try:
                from model_config import (
                    get_builtin_models,
                    get_model_config,
                    resolve_model_path,
                )
                from local_store import get_local_experimental_models

                try:
                    db_models = get_local_experimental_models()
                except Exception:
                    db_models = []
                    logger.warning(
                        "failed to read local model registry during pipeline recovery",
                        exc_info=True,
                    )
                configured = get_model_config(model_id, db_models)
                candidates = [configured] if configured is not None else []
                candidates.extend(
                    item for item in get_builtin_models()
                    if item is not configured
                )
                for candidate in candidates:
                    resolved_paths = [
                        resolve_model_path(str(getattr(candidate, key, "") or ""))
                        for key in ("model_path", "gguf_path")
                    ]
                    candidate_matches = (
                        str(getattr(candidate, "model_id", "") or "") == model_id
                        or any(
                            path and (
                                os.path.basename(os.path.normpath(path)) == model_id
                                or os.path.splitext(os.path.basename(path))[0] == model_id
                            )
                            for path in resolved_paths
                        )
                    )
                    if not candidate_matches:
                        continue
                    for resolved in resolved_paths:
                        if resolved and (os.path.isdir(resolved) or os.path.isfile(resolved)):
                            normalized = os.path.abspath(resolved)
                            if normalized not in model_paths:
                                model_paths.append(normalized)
            except Exception:
                logger.warning(
                    "failed to resolve persisted pipeline model: model=%s",
                    model_id,
                    exc_info=True,
                )

        prepare = getattr(self._host, "prepare_pipeline_model", None)
        if not model_paths or not callable(prepare):
            self._pipeline_recovery_failure = "pipeline_recovery_model_unavailable"
            logger.error(
                "pipeline recovery model unavailable: model=%s paths=%s",
                model_id, model_paths or "unresolved",
            )
            return False
        expected_sha256 = str(state.get("model_sha256", "") or "").lower()
        descriptor = None
        selected_path = ""
        restore_errors: list[tuple[str, str]] = []
        for model_path in model_paths:
            try:
                candidate_descriptor = prepare(
                    model_id=model_id,
                    model_path=model_path,
                    quant_type=(str(state.get("quant_type", "") or "") or None),
                    # Supply selection verifies the live segment and its source
                    # identity. Ordinary registry artifacts recompute their SHA.
                    model_sha256=(
                        supplied["manifest"]["source_model_sha256"]
                        if supplied is not None else None
                    ),
                    **({"layer_range": supply_range} if supplied is not None else {}),
                )
                actual_sha256 = str(
                    candidate_descriptor.get("model_sha256", "")
                    if isinstance(candidate_descriptor, dict) else ""
                ).lower()
                if expected_sha256 and actual_sha256 != expected_sha256:
                    restore_errors.append((model_path, "digest_mismatch"))
                    continue
                descriptor = candidate_descriptor
                selected_path = model_path
                break
            except Exception as exc:
                restore_errors.append((model_path, type(exc).__name__))
                logger.warning(
                    "pipeline recovery model candidate rejected: model=%s path=%s",
                    model_id, model_path, exc_info=True,
                )
        if descriptor is None:
            unload = getattr(self._host, "unload_model", None)
            if callable(unload):
                try:
                    unload()
                except Exception:
                    logger.warning(
                        "failed to clear mismatched recovery model metadata",
                        exc_info=True,
                    )
            only_digest_mismatch = bool(restore_errors) and all(
                reason == "digest_mismatch" for _path, reason in restore_errors
            )
            self._pipeline_recovery_failure = (
                "pipeline_recovery_model_digest_mismatch"
                if only_digest_mismatch else "pipeline_recovery_model_restore_failed"
            )
            logger.error(
                "pipeline recovery model restore exhausted: model=%s expected=%s errors=%s",
                model_id, expected_sha256 or "unspecified", restore_errors,
            )
            return False
        logger.info(
            "pipeline recovery model metadata restored: model=%s path=%s type=%s layers=%s",
            model_id,
            selected_path,
            descriptor.get("model_type", "") if isinstance(descriptor, dict) else "",
            descriptor.get("total_layers", "") if isinstance(descriptor, dict) else "",
        )
        return True

    def _clear_pipeline_recovery_fence(self) -> None:
        # Successful retirement or a fresh ready generation ends this recovery
        # identity. Native cleanup obligations remain in their ACK-owned ledger.
        self._pipeline_recovery_state = None
        self._pipeline_recovery_pending = False
        self._pipeline_recovery_failure = ""

    def retire_pipeline_recovery_for_model_change(self) -> bool:
        """Persist explicit retirement without forgiving any native cleanup.

        The model-change owner calls this only after a successful local change
        and invalidation of the old transaction. A restart recovery identity is
        no longer a target after that explicit choice; its native debts retain
        their separate, ACK-owned lifecycle.
        """
        with self._layer_config_lock:
            if not self._pipeline_recovery_validated:
                return False
            recovery = self._pipeline_recovery_state
            if not isinstance(recovery, dict):
                return not self._pipeline_recovery_pending
            if (self._pipeline_load_transaction is not None
                    or self._active_pipeline_capacity_plan is not None
                    or not getattr(self, "_running", False)
                    or self._effective_role() != "master"):
                return False
            if (not self._pipeline_lifecycle_persist_ok
                    and self._pipeline_recovery_failure not in {
                        "pipeline_lifecycle_persist_failed",
                        # ★ 2026-10-23（#117）：**"模型已变"与"持久化坏"性质不同**。
                        #   前者意味着这条恢复记录指向的模型已经不存在了，放弃它正是
                        #   本入口的职责；后者是记录不可读/不可重建，换模型也洗不掉，
                        #   必须保留围栏。此前只放行 persist_failed ⇒ 模型不匹配时恒
                        #   return False ⇒ 围栏清不掉 ⇒ 层配置推送恒 `return None` ⇒
                        #   供给恒报 `pipeline_transaction_not_started`，死锁。
                        "pipeline_recovery_model_mismatch",
                    }):
                # Invalid/unreadable cleanup records cannot be reconstructed by
                # replacing the local model. Preserve their existing fence.
                return False
            self._pipeline_recovery_state = {
                **recovery,
                "phase": "invalidated",
                "published": False,
                "updated_at": time.time(),
            }
            self._persist_pipeline_lifecycle_locked()
            if not self._pipeline_lifecycle_persist_ok:
                return False
            self._clear_pipeline_recovery_fence()
            return True

    def _recovery_model_matches(self, model_info: dict) -> bool:
        """Fence a fresh generation to the model persisted before restart."""
        if not self._pipeline_recovery_pending:
            return True
        persisted = self._pipeline_recovery_state
        if not isinstance(persisted, dict):
            return True
        expected_id = str(persisted.get("model_id", "") or "")
        expected_type = str(persisted.get("model_type", "") or "").lower()
        expected_sha256 = str(persisted.get("model_sha256", "") or "").lower()
        actual_id = str(model_info.get("model_id", "") or "")
        actual_type = str(model_info.get("model_type", "") or "").lower()
        actual_sha256 = str(model_info.get("model_sha256", "") or "").lower()
        # ★ 2026-10-23（#117）：**模型身份看 id/type，摘要不参与否决**。
        #   持久化记录里的摘要与当前描述器的摘要来自**不同的格式域**（实测
        #   `lifecycle.model_sha256 = d6e6d9ab…` 是 HF 侧，而当前 GGUF 侧是
        #   `ada47e14…`），而 `model_id` / `model_type` **完全相同**
        #   （`qwen3-5-9b` / `qwen3_5`）。这说明是**同一模型的不同量化/格式**，
        #   不是换了模型。此前把摘要不等也算 mismatch ⇒ 恒判失败 ⇒ 层配置推送
        #   恒 `return None` ⇒ 供给恒报 `pipeline_transaction_not_started`。
        #   摘要不同只告警：真的换了模型时 id/type 也会变，仍会被下面拦下。
        if (
            (expected_id and actual_id != expected_id)
            or (expected_type and actual_type != expected_type)
        ):
            self._pipeline_recovery_failure = "pipeline_recovery_model_mismatch"
            logger.error(
                "pipeline recovery model mismatch: expected=%s/%s actual=%s/%s",
                expected_id, expected_type, actual_id, actual_type,
            )
            # 模型不匹配时**保留围栏并记录失败**：清围栏是
            # `retire_pipeline_recovery_for_model_change()` 的职责（在"显式更换模型"
            # 时把 lifecycle 落成 `invalidated/published=False` 再清）。此处不得擅自
            # 放弃 —— 本地模型替换失败的场景必须保留恢复身份，
            # `tests/test_pipeline_recovery_retirement.py` 对此有明确断言。
            return False
        if expected_sha256 and actual_sha256 and expected_sha256 != actual_sha256:
            # 同 id/type 但摘要不同：同一模型的不同量化或格式域，只记录不否决。
            logger.info(
                "pipeline recovery digest differs but identity matches: "
                "model=%s type=%s persisted_sha=%s current_sha=%s",
                actual_id, actual_type, expected_sha256[:16], actual_sha256[:16],
            )
        return True

    def _maybe_finish_pipeline_recovery(self) -> None:
        """Release the restart fence only after the fresh generation is ready."""
        if not self._pipeline_recovery_pending:
            return
        if (not self._pipeline_lifecycle_persist_ok
                or not self._pipeline_recovery_validated):
            return
        with self._layer_config_lock:
            transaction = self._pipeline_load_transaction
            if not transaction:
                return
            phase = str(transaction.get("phase", "") or "")
            if phase in {"rejected", "aborted", "invalidated"}:
                self._pipeline_recovery_failure = (
                    str(transaction.get("reason_code", "") or "")
                    or "pipeline_recovery_failed"
                )
                return
            if phase != "ready":
                return
            config_id = str(transaction.get("config_id", "") or "")
            if not config_id:
                return
            try:
                generation = int(transaction.get("generation", 0) or 0)
            except (TypeError, ValueError):
                return
            persisted = self._pipeline_recovery_state
            if isinstance(persisted, dict):
                persisted_config_id = str(persisted.get("config_id", "") or "")
                try:
                    persisted_generation = int(
                        persisted.get("generation", 0) or 0
                    )
                except (TypeError, ValueError):
                    persisted_generation = 0
                if (
                    persisted_config_id
                    and config_id == persisted_config_id
                    and generation <= persisted_generation
                ):
                    return
            plan = transaction.get("plan")
            plan = dict(plan) if isinstance(plan, dict) else {}
            for node_id, expected in self._worker_assignments.expected_configs().items():
                if expected.get("release"):
                    return
                if not self._effective_layer_config_pushed(node_id):
                    return
        for assignment in plan.get("assignments", []):
            if not isinstance(assignment, dict):
                continue
            if assignment.get("execution") != "stage_offer_v3":
                continue
            ready, _reason = self._stage_offer_assignment_ready(
                str(assignment.get("node_id", "") or ""), assignment,
            )
            if not ready:
                return
        with self._layer_config_lock:
            transaction = self._pipeline_load_transaction
            if (
                transaction
                and transaction.get("config_id") == config_id
                and transaction.get("phase") == "ready"
            ):
                self._clear_pipeline_recovery_fence()

    @property
    def _layer_config_pushed(self) -> frozenset:
        """★ 2026-10-07（DIST-NEXT-3）：**派生视图**，不再是事实源。

        集合语义 = 「该节点已确认收到本代际层配置」，唯一来源是
        `_worker_assignments` 的相位（`ACKED` / `READY`）。返回 `frozenset`：
        任何遗留的 `add` / `discard` / `clear` 会立刻 `AttributeError`（fail-loud），
        而不是静默失效 —— 这是"降级为派生视图"能安全落地的前提。
        """
        return frozenset(
            node_id for node_id, state in self._worker_assignments.snapshot().items()
            if state["phase"] in (PHASE_ACKED, PHASE_READY)
        )

    def _effective_layer_config_pushed(
        self, node_id: str, _legacy_pushed: bool = False,
    ) -> bool:
        """Return the authoritative ACK/READY verdict for one assignment."""
        state = self._worker_assignments.state(node_id)
        return state is not None and state.phase in (PHASE_ACKED, PHASE_READY)

    def _effective_layer_config_pushed_nodes(
        self, _legacy_pushed: set[str] | None = None,
        _expected_configs: dict | None = None,
    ) -> set[str]:
        """Return all ACKED/READY nodes from the single assignment source."""
        return set(self._layer_config_pushed)

    @property
    def _active_layer_config(self):
        """当前生效的层段配置（`None` = 本节点不是层段 worker）。

        用 property 而不是裸属性：它的**每一次变化**都会改变本节点在 hello 里上报的
        `layer_worker` / `layer_ranges` / `models` 语义，因此必须同步给主节点
        （`refresh_task_worker_capabilities()`）。此前层段路径**全都不 refresh**
        —— 整模路径都 refresh、层段路径一条都没有 —— 于是主节点一直拿 hello 旧快照
        判身份，远端 Stage 被 `model_identity_mismatch` 拒（`#28`）。

        收口成 setter 就不会再漏：那 7 个赋值点一行都不用改，将来新增的也会自动生效。
        """
        return getattr(self, "_active_layer_config_value", None)

    @_active_layer_config.setter
    def _active_layer_config(self, value):
        previous = getattr(self, "_active_layer_config_value", None)
        self._active_layer_config_value = value
        if previous == value:
            return
        # `refresh_task_worker_capabilities()` 是异步的（只起线程，内部只碰
        # `_task_worker_refresh_lock`），所以在层配置锁内调用是安全的。构造早期
        # `_tcp_client` 还没就位时它自己会返回 False。
        refresh = getattr(self, "refresh_task_worker_capabilities", None)
        if callable(refresh):
            try:
                refresh()
            except Exception:
                logger.debug("层段状态变化后刷新 hello 失败（忽略）", exc_info=True)

    def request_authoritative_layer_sync(
        self, *, require_distributed: bool = False,
    ) -> bool:
        """让在线 PC 从节点服从主节点当前模型和分层配置。

        从节点显式执行本地模型操作后会暂时退出分层 worker。主节点模型
        加载完成或收到新的分布式请求时，通过这个一次性标记重新取得
        配置权威；普通拓扑刷新仍尊重从节点的临时退出状态。
        """
        if not self.get_distributed_inference_enabled() or self._effective_role() != "master":
            return False
        # Check the fence and publish under one lock order.  A check performed
        # before acquiring `_layer_config_push_lock` can race with begin/end
        # and publish a transient release during a model transition.
        with self._layer_config_push_lock:
            with self._layer_config_lock:
                if self._layer_config_model_change_depth:
                    self._layer_config_push_deferred = True
                    self._layer_config_push_deferred_authoritative = True
                    self._layer_config_push_deferred_require_distributed = (
                        self._layer_config_push_deferred_require_distributed
                        or bool(require_distributed)
                    )
                    logger.info(
                        "defer authoritative layer sync during model transition depth=%d",
                        self._layer_config_model_change_depth,
                    )
                    return True
                self._authoritative_layer_sync_requests += 1
            try:
                # Explicit distributed requests bypass a single-node capacity plan;
                # ordinary authoritative refreshes preserve the existing policy.
                self._push_layer_config_to_clients_locked(
                    require_distributed=require_distributed,
                )
            finally:
                with self._layer_config_lock:
                    self._authoritative_layer_sync_requests = max(
                        0, self._authoritative_layer_sync_requests - 1,
                    )
        return True

    def start_pipeline_model_transaction(
        self, *, candidate_id: str, descriptor: dict,
        model_path: str, quant_type: str = "",
        require_distributed: bool = True, on_commit=None, operation_id: str = "",
        capacity_evidence=None,
    ) -> dict:
        """Start one explicit model candidate without replacing active state."""
        candidate_id = str(candidate_id or "").strip()
        if not candidate_id:
            raise ValueError("candidate_id is required")
        if self._effective_role() != "master":
            return {
                "status": "rejected", "admitted": False,
                "reason_code": "pipeline_master_required", "assignments": [],
            }
        connected_for_cleanup = set(self._connected_client_ids())
        with self._layer_config_push_lock:
            with self._layer_config_lock:
                transaction = self._pipeline_load_transaction
                transaction_active = bool(
                    transaction
                    and str(transaction.get("phase", "") or "") in {
                        "preparing", "committing_local", "committing",
                        "committing_global",
                    }
                )
                if transaction_active:
                    existing_candidate = str(
                        transaction.get("candidate_id", "") or ""
                    )
                    if existing_candidate == candidate_id:
                        plan = dict(transaction.get("plan", {}) or {})
                        plan.update({
                            "config_id": str(transaction.get("config_id", "") or ""),
                            "generation": int(transaction.get("generation", 0) or 0),
                            "transaction_phase": str(transaction.get("phase", "") or ""),
                        })
                        return plan
                    return {
                        "status": "rejected", "admitted": False,
                        "reason_code": "MODEL_TXN_CONFLICT",
                        "reason": "another model transaction is already active",
                        "transaction_id": str(
                            transaction.get("config_id", "") or ""
                        ),
                        "assignments": [],
                    }
                cleanup_blockers = [
                    record
                    for record in self._pipeline_model_cleanup_ledger.values()
                    if record.get("node_id") == "master"
                    or record.get("node_id") in connected_for_cleanup
                ]
                if cleanup_blockers:
                    blocker = min(
                        cleanup_blockers,
                        key=lambda item: item.get("created_at", 0.0),
                    )
                    return {
                        "status": "rejected", "admitted": False,
                        "reason_code": "MODEL_TXN_CLEANUP_PENDING",
                        "reason": (
                            "previous model transaction cleanup is pending "
                            f"for node {blocker.get('node_id', '')}"
                        ),
                        "transaction_id": str(
                            blocker.get("candidate_config_id", "") or ""
                        ),
                        "assignments": [],
                    }
                if self._layer_config_model_change_depth:
                    return {
                        "status": "rejected", "admitted": False,
                        "reason_code": "MODEL_TXN_CONFLICT",
                        "reason": "local model replacement is in progress",
                        "assignments": [],
                    }
                self._authoritative_layer_sync_requests += 1
            try:
                result = self._push_layer_config_to_clients_locked(
                    require_distributed=require_distributed,
                    candidate={
                        "candidate_id": candidate_id,
                        "descriptor": dict(descriptor or {}),
                        "model_path": str(model_path or ""),
                        "quant_type": str(quant_type or ""),
                        "on_commit": on_commit,
                        "operation_id": operation_id,
                        "capacity_evidence": capacity_evidence,
                    },
                )
            finally:
                with self._layer_config_lock:
                    self._authoritative_layer_sync_requests = max(
                        0, self._authoritative_layer_sync_requests - 1,
                    )
        if isinstance(result, dict):
            return result
        return {
            "status": "rejected", "admitted": False,
            "reason_code": "pipeline_transaction_not_started",
            "assignments": [],
        }


    def _task_worker_layer_stage_ids(self, connected_ids: set[str]) -> set[str]:
        """Return connected Android peers admitted for Route-A layer stages."""
        if (
            self._effective_role() != "master"
            or not TASK_WORKER_EXPERIMENTAL_ENABLED
        ):
            return set()
        try:
            if self._pipeline_recovery_pending:
                # Use accepted HELLO capabilities while rebuilding the first
                # fresh assignment. The normal status projection is
                # heartbeat-gated and can still report ``not_configured``.
                raw_workers = self._task_worker_control.connected_layer_stage_workers(
                    set(connected_ids),
                )
                admitted = {
                    str(worker.get("node_id", ""))
                    for worker in raw_workers
                    if isinstance(worker, dict)
                    and int(worker.get("selected_version", 0) or 0) >= 2
                }
                # ★ 2026-10-08（真机闸门卡点）：**恢复期不得比正常路径更严**。
                #
                #   v3 层段 worker 的能力来自它自己的 hello（`layer_stage_dispatch_enabled`
                #   = healthy ∧ version≥2 ∧ `layer_forward` ∧ `layer_ranges` 非空），与 legacy
                #   层配置的「重启恢复闸门」是两回事。实测：master 重启后的恢复窗口里本分支
                #   返回**空集** ⇒ capacity 候选缺 Y700 ⇒ 请求被判
                #   `pipeline_layer_range_coverage_insufficient`（把"恢复中"误导成"区间覆盖不足"），
                #   而 **1.2 秒后**恢复期一解除，**同一个请求**就 `admitted=True`
                #   （00:47:06 / 00:47:08 的对照日志）。
                #   这里在空集时回退到正常准入判据（三分量齐备的 worker 照常参与规划），
                #   并记一条诊断 —— 恢复期里到底有没有可用的 v3 worker，从此可查。
                if not admitted:
                    status = self._task_worker_control.status(role="master")
                    fallback_admitted = {
                        str(worker.get("node_id", ""))
                        for worker in status.get("workers", []) or []
                        if isinstance(worker, dict)
                        and worker.get("healthy")
                        and worker.get("layer_stage_dispatch_enabled")
                        and str(worker.get("node_id", "")) in connected_ids
                    }
                    logger.info(
                        "event=stage_admission_recovery_empty raw=%s fallback=%s",
                        sorted(admitted), sorted(fallback_admitted),
                    )
                    admitted = fallback_admitted
            else:
                status = self._task_worker_control.status(role="master")
                # ★ 2026-10-08（诊断，定位后降级）：逐 worker 打出**准入三分量**，用来回答
                #   "为什么状态显示健康的 v3 层段 worker 仍被判 not_eligible"。capacity 候选的
                #   白名单正取自本函数的返回值（`stage_releasable_worker_ids`），所以这里缺哪个
                #   分量，就是候选缺它的原因。
                for _worker in status.get("workers", []) or []:
                    if not isinstance(_worker, dict):
                        continue
                    _caps = _worker.get("capabilities")
                    _ok = (
                        _worker.get("healthy") is True
                        and _worker.get("layer_stage_dispatch_enabled") is True
                        and isinstance(_caps, dict)
                        and bool(_caps.get("layer_ranges"))
                        and _worker.get("node_id") in connected_ids
                    )
                    if _ok:
                        # 合格 ⇒ 不打日志（避免每次请求每个 worker 一条）。
                        continue
                    # 只有**不合格**时才记 —— 那正是"候选为什么缺它"的答案。
                    logger.info(
                        "event=stage_admission_rejected node=%s healthy=%s dispatch=%s "
                        "ranges=%s in_connected=%s version=%s",
                        _worker.get("node_id"),
                        _worker.get("healthy"),
                        _worker.get("layer_stage_dispatch_enabled"),
                        (str(_caps.get("layer_ranges"))[:48]
                         if isinstance(_caps, dict) else "n/a"),
                        _worker.get("node_id") in connected_ids,
                        _worker.get("selected_version"),
                    )
                admitted = {
                    str(worker.get("node_id", ""))
                    for worker in status.get("workers", [])
                    if isinstance(worker, dict)
                    and worker.get("healthy")
                    and worker.get("layer_stage_dispatch_enabled")
                    and str(worker.get("node_id", "")) in connected_ids
                }
        except Exception:
            return set()
        with self._nodes_lock:
            # ★ 2026-10-03：PC 也能承层段 —— v3 `layer_forward` 已在
            #   `EngineHost.execute_task_worker_stage` 实现，工件与 shim 经 env 配置。
            #   此前这里硬编码只认 `android` ⇒ PC worker 即使声明了 `layer_forward`
            #   + `layer_ranges` 也永远拿不到 `stage_offer_v3` 标记，层段永远派不到它。
            #   `admitted` 已过 `layer_stage_dispatch_enabled` 门控（即已声明层段能力），
            #   节点类型只用于排除不具备该能力的旧式节点。
            selected = {
                node_id for node_id in admitted
                if getattr(self.nodes.get(node_id), "node_type", "")
                in ("android", "pc")
            }
        logger.info(
            "层段 worker 准入: admitted=%s selected=%s",
            sorted(admitted), sorted(selected),
        )
        return selected


    def _push_layer_config_to_clients_locked(
        self, *, require_distributed: bool = False,
        candidate: dict | None = None,
    ) -> dict | None:
        """
        向所有 TCP 连接的从节点推送其分层配置。

        assignment 携带当前 PyTorch 模型身份和摘要。从节点缺少或模型不一致时
        先从主节点同步模型，校验成功并加载层范围后再返回 ready ACK。
        """
        if not self.get_distributed_inference_enabled():
            return {
                "status": "rejected", "admitted": False,
                "reason_code": "local_inference_mode", "assignments": [],
            } if candidate is not None else None
        # `getattr` 而非直接取属性：`push_layer_config_to_clients()` 现在会在 task-worker
        # hello 之后被调用（legacy 重算），而测试/嵌入场景下的 `_tcp_server` 桩可能没有
        # `_running` ⇒ 直接取会 `AttributeError`（实测回归）。
        if self._pipeline_recovery_pending:
            # The persisted record is an epoch fence, not an executor
            # assignment that may be replayed blindly.  Recompute a fresh
            # authoritative generation from current worker capabilities.
            require_distributed = True
        if not self._tcp_server or not getattr(self._tcp_server, "_running", False):
            if candidate is not None:
                return {
                    "status": "rejected", "admitted": False,
                    "reason_code": "pipeline_workers_unavailable",
                    "assignments": [],
                }
            return None
        if candidate is None:
            with self._layer_config_lock:
                current = self._pipeline_load_transaction
                if isinstance(current, dict) and current.get("phase") in {
                    "preparing", "committing_local", "committing", "committing_global",
                }:
                    # HELLO/inventory refresh must not replace a model operation
                    # whose native load is still in flight.
                    return {**dict(current.get("plan", {})), "transaction_phase": current["phase"]}
                if self._pipeline_model_cleanup_ledger:
                    return dict(self._active_pipeline_capacity_plan or {})
                active = dict(self._active_pipeline_capacity_plan or {})
            stage_items = [item for item in active.get("assignments", [])
                           if item.get("execution") == "stage_offer_v3"]
            if stage_items and all(self._stage_offer_assignment_ready(item["node_id"], item)[0]
                                   for item in stage_items):
                return active
            supply = getattr(self, "_model_supply", None)
            active_operation = getattr(supply, "_active", "")
            if active_operation:
                operation = supply.get(active_operation)
                matching_operation = any(item.get("worker_prepare_identity", {}).get("operation_id") == active_operation
                                         for item in stage_items)
                if not matching_operation and operation.get("status") not in {"failed", "cancelled"}:
                    # SUPPLY_ASSETS_AVAILABLE is a registration receipt. Explicit
                    # prepare owns the transition to a loaded execution plan.
                    return active
        get_client_ids = getattr(self._tcp_server, "get_client_ids", None)
        connected_ids = (
            get_client_ids()
            if callable(get_client_ids)
            else list(getattr(self._tcp_server, "clients", {}).keys())
        )
        if not connected_ids:
            if candidate is not None:
                return {
                    "status": "rejected", "admitted": False,
                    "reason_code": "pipeline_workers_unavailable",
                    "assignments": [],
                }
            return None

        with self._nodes_lock:
            # 这是旧版 PyTorch LAYER_CONFIG 线路。Android 的 llama.cpp
            # worker 只能走 v3 task_worker stage offer，不能收到这份配置。
            releasable_legacy_ids = {
                node_id for node_id, node in self.nodes.items()
                if node_id in connected_ids
                and node_id != self.get_effective_node_id()
                and (
                    getattr(node, "node_type", "pc") == "pc"
                    # Relay is an endpoint-backed role, not a local model
                    # capability. Preserve it even when the host advertises
                    # itself as Android so it can receive relay_middle rather
                    # than being silently excluded before role resolution.
                    or self._is_relay_host(node_id)
                )
            }
        try:
            pending_worker_ids = self._task_worker_control.pending_worker_ids()
        except Exception:
            pending_worker_ids = set()
        if pending_worker_ids:
            releasable_legacy_ids.difference_update(pending_worker_ids)
            logger.info(
                "legacy layer config fenced pending task-worker hello: %s",
                sorted(pending_worker_ids & set(connected_ids)),
            )
        # ★ 2026-10-03：**已声明 v3 层段能力的节点必须排除在这条线路之外** —— 不只是
        #   不给它派 legacy 层段，而是**连配置都不要推**。否则它会按 legacy 语义去
        #   `ensure_pipeline_assignment_available` 同步工件，在跨机（非 loopback、
        #   不在信任 CIDR）时被 `MODEL_API_SOURCE_UNTRUSTED` 拒绝，进而
        #   「主节点已释放本设备的分层 worker 预留」⇒ 它直接掉出层段 worker 名单
        #   （实测：Surface 因此从 `admitted` 里消失）。
        #   它手上有工件，v3 stage offer 走的是 offer 里带的 hidden，不需要 master 推模型。
        # Record endpoint-backed relay hosts before removing v3 stage workers
        # from the legacy candidate set. A relay worker may advertise
        # ``layer_forward`` from a local artifact, but it still needs the
        # logical ``relay_middle`` assignment after a master restart; the
        # remote relay service owns that segment's data path.
        relay_worker_ids = {
            node_id for node_id in releasable_legacy_ids
            if self._is_relay_host(node_id)
        }
        releasable_legacy_ids -= self._task_worker_layer_stage_ids(set(connected_ids))
        # Route A Android workers use v3 stage_offer and must never receive a
        # legacy LAYER_CONFIG (which would make a layer worker look like a
        # full-model worker and reintroduce the old opt-out path).
        stage_releasable_worker_ids = self._task_worker_layer_stage_ids(
            set(connected_ids)
        )

        # A healthy full-model Task Worker has priority over legacy automatic
        # layer assignment. Keep its local model intact and release any stale
        # layer reservation instead of reassigning a subset of layers.
        full_worker_release_ids = (
            releasable_legacy_ids & self._task_worker_full_model_ids()
        )
        # A node explicitly assigned to an endpoint-backed relay segment is
        # still a pipeline participant even if it also advertises a full model.
        full_worker_release_ids.difference_update(relay_worker_ids)
        layer_releasable_worker_ids = releasable_legacy_ids - full_worker_release_ids

        with self._layer_config_lock:
            # Clear stale opt-outs created before this node was assigned as a
            # relay; compute_layer_assignment filters opted-out nodes first.
            self._pipeline_worker_opt_out.difference_update(relay_worker_ids)
            if full_worker_release_ids:
                self._pipeline_worker_opt_out.update(full_worker_release_ids)
            authoritative_sync = bool(
                self._authoritative_layer_sync_requests
            )
            reenabled_nodes = (
                self._pipeline_worker_opt_out & layer_releasable_worker_ids
                if authoritative_sync else set()
            )
            if reenabled_nodes:
                self._pipeline_worker_opt_out.difference_update(reenabled_nodes)
        if reenabled_nodes:
            logger.info(
                "主节点权威模型同步重新启用分层 worker: %s",
                sorted(reenabled_nodes),
            )

        with self._layer_config_lock:
            self._layer_config_generation = max(
                self._layer_config_generation + 1,
                time.time_ns(),
            )
            generation = self._layer_config_generation
        config_id = uuid.uuid4().hex

        candidate_descriptor = (
            dict(candidate.get("descriptor", {}) or {})
            if isinstance(candidate, dict) else None
        )
        if isinstance(candidate, dict):
            model_info = self._get_active_pipeline_model_info(
                descriptor=candidate_descriptor,
                model_path=str(candidate.get("model_path", "") or ""),
                quant_type=str(candidate.get("quant_type", "") or ""),
            )
        else:
            model_info = self._get_active_pipeline_model_info()
        master_sha256 = model_info.get("model_sha256", "")
        model_id = model_info.get("model_id", "")
        model_type = model_info.get("model_type", "")
        if model_info and not self._recovery_model_matches(model_info):
            return None
        # ★ #31 M2：走**单一事实来源**（此前这里硬编码 `{"qwen","qwen2"}`
        #   ⇒ hybrid 会被静默拦掉，master **不推层配置**）
        if (not master_sha256 or not model_id
                or model_type not in PIPELINE_RUNTIME_MODEL_TYPES):
            # Relay hosts can be rehydrated without local model metadata.
            # During master restart, releasing them here races the next
            # assignment/ACK and leaves the relay with no active config.
            relay_assignments = {}
            relay_total_layers = self._get_total_model_layers()
            for node_id in relay_worker_ids:
                relay_segment = self._relay_segment_for_worker(node_id)
                if relay_segment is None:
                    continue
                relay_assignments[node_id] = {
                    "node_id": node_id,
                    "config_id": config_id,
                    "generation": generation,
                    "start_layer": relay_total_layers,
                    "end_layer": relay_total_layers,
                    "has_embedding": False,
                    "has_lm_head": False,
                    "model_id": model_id,
                    "model_sha256": master_sha256,
                    "model_type": model_type,
                    "total_layers": relay_total_layers,
                    "engine": "relay_middle",
                    "phase": "commit",
                    "relay_segment": relay_segment,
                }
            releases = {
                node_id: {
                    "node_id": node_id,
                    "config_id": config_id,
                    "generation": generation,
                    "release": True,
                }
                for node_id in releasable_legacy_ids
                if node_id not in relay_assignments
            }
            self._publish_layer_configs({**relay_assignments, **releases})
            logger.warning("主节点尚未加载可校验的 PyTorch 模型，暂不推送层配置")
            return None

        distributed_only = bool(
            require_distributed
            or candidate is not None
            or getattr(self._host, "is_pipeline_prepared", False)
        )
        capacity_plan = None
        if distributed_only:
            manual_override = bool(self._runtime_layer_override)
            if manual_override and not require_distributed:
                layer_info = self.get_layer_assignments()
                capacity_plan = self._build_manual_pipeline_capacity_plan(
                    layer_info.get("assignments", [])
                )
            else:
                if manual_override:
                    logger.info(
                        "分布式请求忽略单机手动分层覆盖，改用多节点容量求解"
                    )
                eligible_node_ids = set(layer_releasable_worker_ids)
                eligible_node_ids.update(stage_releasable_worker_ids)
                eligible_node_ids.update({"master", self.get_effective_node_id()})
                capacity_plan = self.get_pipeline_capacity_plan(
                    eligible_node_ids,
                    descriptor=candidate_descriptor,
                    require_distributed=require_distributed,
                    **({"capacity_evidence": candidate["capacity_evidence"]}
                       if candidate and candidate.get("capacity_evidence") is not None else {}),
                )
            if isinstance(capacity_plan, dict):
                capacity_plan = dict(capacity_plan)
                capacity_plan["quant_type"] = str(
                    model_info.get("quant_type", "") or ""
                )
            if not capacity_plan.get("admitted"):
                releases = {
                    node_id: {
                        "node_id": node_id,
                        "config_id": config_id,
                        "generation": generation,
                        "release": True,
                        "abort": True,
                        "reason_code": capacity_plan.get(
                            "reason_code", "pipeline_capacity_rejected"
                        ),
                    }
                for node_id in releasable_legacy_ids
                # ★ 与下面的 `:537` releases 同理：relay 宿主**不**收本地层配置，
                #   容量求解的否定结论不该顺带释放它的预留 —— 否则它每次 hello 后
                #   都「确认退出分层 worker」，relay 链再也拿不到中间段（实测）。
                if not self._is_relay_host(node_id)
                }
                with self._layer_config_lock:
                    self._pipeline_load_transaction = {
                        "config_id": config_id,
                        "generation": generation,
                        "phase": "rejected",
                        "plan": dict(capacity_plan),
                        "prepared_nodes": set(),
                        "reason_code": capacity_plan.get("reason_code", ""),
                        "candidate_id": (
                            str(candidate.get("candidate_id", "") or "")
                            if isinstance(candidate, dict) else ""
                        ),
                    }
                    if candidate is None:
                        self._active_pipeline_capacity_plan = None
                    self._persist_pipeline_lifecycle_locked()
                if candidate is None:
                    self._publish_layer_configs(releases)
                self._maybe_finish_pipeline_recovery()
                logger.warning(
                    "集群容量准入拒绝流水线加载: reason=%s",
                    capacity_plan.get("reason_code", "unknown"),
                )
                return dict(capacity_plan)
            layer_info = {
                "assignments": capacity_plan.get("assignments", []),
            }
        else:
            layer_info = self.get_layer_assignments()
        assignments = {}
        stage_assignments = {}
        operation_id = str((candidate or {}).get("operation_id", ""))
        if not operation_id and candidate is None:
            from model_supply_worker import get_selected_model
            supply_worker = getattr(self, "_model_supply_worker", None)
            selected = get_selected_model(supply_worker.root) if supply_worker is not None else None
            if selected is not None and selected["manifest"]["source_model_id"] == model_id:
                operation_id = selected["manifest"]["operation_id"]
        from config import API_PORT

        for a in layer_info["assignments"]:
            nid = a["node_id"]
            if (
                nid in {"master", self.get_effective_node_id()}
                or (
                    nid not in layer_releasable_worker_ids
                    and nid not in stage_releasable_worker_ids
                )
            ):
                continue

            if (
                nid in stage_releasable_worker_ids
                # ★ relay 段不属于 v3 stage 名单：它的层由远端 relay_mid_service
                #   代跑，本节点只转发（下面 `relay_segment` 分支会给它
                #   `engine="relay_middle"`）。若这里把它标成 `stage_offer_v3`，
                #   `_run_pipeline` 就会按 v3 数据面去要它的层区间，而它手上根本没有
                #   那段工件 ⇒ `layer_range_not_advertised`（实测：relay 链被误路由到
                #   Route-A stage 路径后卡在这里）。
                and not self._is_relay_host(nid)
                # ★ 2026-10-05（DIST-4「发布闸门」）：A3 的独立开关。置 0 时不打
                #   `stage_offer_v3` 标记 ⇒ `stage_offer_nodes` 为空，层段链整体不
                #   参与，请求按既有路径处理。默认 1 = 保持现状行为（本开关是给
                #   发布/部署侧显式关掉实验线路用的，不是改变已验收的行为）。
                and PIPELINE_ROUTE_A_STAGE_OFFER_ENABLED
            ):
                # Keep the assignment in the active capacity plan for the
                # execution/readiness contract, but do not materialize a
                # local model segment or publish LAYER_CONFIG to Android.
                a["execution"] = "stage_offer_v3"
                a["stage_type"] = "layer_forward"
                try:
                    stage_assignments[nid] = self._build_worker_prepare_assignment(
                        a, config_id=config_id, generation=generation,
                        model_info=model_info,
                        operation_id=operation_id,
                    )
                except ValueError as exc:
                    return {"status": "rejected", "admitted": False,
                            "reason_code": str(exc), "reason": str(exc),
                            "assignments": [], "transaction_phase": "rejected"}
                if candidate is None:
                    self._clear_layer_config_state(nid)
                continue

            # 新一轮配置开始后，旧 ACK 立即失效。
            if candidate is None:
                self._clear_layer_config_state(nid)

            assignments[nid] = {
                "node_id": nid,
                "config_id": config_id,
                "generation": generation,
                "start_layer": a["start_layer"],
                "end_layer": a["end_layer"],
                "has_embedding": a.get("has_embedding", False),
                "has_lm_head": a.get("has_lm_head", False),
                "model_id": model_id,
                "model_sha256": master_sha256,
                "model_type": model_type,
                "total_layers": int(model_info["total_layers"]),
                "master_quant_type": model_info.get("quant_type", ""),
                "engine": (
                    "relay_middle"
                    if self._is_relay_host(nid)
                    else "pytorch"
                ),
                "sync_policy": (
                    "master_authoritative" if authoritative_sync else "normal"
                ),
                "authoritative_sync": authoritative_sync,
                "master_api_port": API_PORT,
            }
            relay_segment = self._relay_segment_for_worker(nid)
            if relay_segment is not None:
                assignments[nid]["relay_segment"] = relay_segment
                logger.info(
                    "relay 段委派已下发: node=%s → %s@%s:%s（该节点不加载本地层）",
                    nid, relay_segment.get("role"), relay_segment.get("host"),
                    relay_segment.get("port"),
                )
            if capacity_plan is not None:
                assignments[nid].update({
                    "phase": "prepare",
                    "assignment_manifest": True,
                    "plan_id": capacity_plan.get("plan_id", ""),
                    "required_bytes": int(a.get("required_bytes", 0) or 0),
                    "capacity_bytes": int(a.get("capacity_bytes", 0) or 0),
                    "capacity_source": a.get("capacity_source", ""),
                    "safety_margin": capacity_plan.get("safety_margin", 1.0),
                })

        releases = {
            node_id: {
                    "node_id": node_id,
                    "config_id": config_id,
                    "generation": generation,
                    "release": True,
            }
            for node_id in (layer_releasable_worker_ids | full_worker_release_ids)
            if node_id not in assignments and node_id not in stage_assignments
            # ★ relay 宿主**不**收本地层配置（它的层由远端 relay_mid_service 代跑），
            #   因此它永远不在 `assignments` 里 —— 但那不等于该释放它的预留：它要的
            #   恰恰是那份 legacy 层配置（`engine="relay_middle"`，由请求路径的
            #   `_push_layer_config_to_clients_locked` 下发；hello 时还没有
            #   capacity_plan，所以那时释放等于把它踢出链路）。此前没有这一条，
            #   实测 Surface 每次 hello 后立刻「确认退出分层 worker」，
            #   relay 链永远拿不到中间段。
            and not self._is_relay_host(node_id)
        }
        configs = (
            {**assignments, **stage_assignments}
            if candidate is not None
            else {**assignments, **stage_assignments, **releases}
        )
        if candidate is not None:
            # Candidate assignments exist beside the active registry until
            # global commit. Allocate their wire identity before prepare so
            # prepare, commit, retry and promotion all retain one ID.
            for payload in configs.values():
                payload.setdefault("assignment_id", new_assignment_id())
        local_frozen = None
        local_executors_frozen: dict[str, dict] = {}
        if capacity_plan is not None and candidate is not None:
            master_ids = {"master", self.get_effective_node_id()}
            freeze = getattr(self._host, "freeze_pipeline_candidate_assignment", None)
            # ★ 2026-10-25（LOCAL-EXECUTORS-01）：**必须排除附加执行器**。本机 CPU 段与
            #   主段**同 node_id**（同一台物理机），若不加这个过滤，`next()` 可能把 CPU 段
            #   当成"本地主段" ⇒ `local_assignment` 语义被污染、主段丢失。
            local = next((item for item in capacity_plan.get("assignments", [])
                          if item.get("node_id") in master_ids
                          and not item.get("executor_id")), None)
            if local is not None and callable(freeze):
                try:
                    local_frozen = freeze(candidate["candidate_id"], local, config_id=config_id, generation=generation)
                except Exception as exc:
                    return {"status": "rejected", "admitted": False, "reason_code": str(exc),
                            "assignments": [], "transaction_phase": "rejected"}
            # ★ 同机**附加执行器**（例如 CPU 段）进 `local_executors`，**进不了**
            #   `assignments`：后者同时是 wire 目的地、连接世代、incarnation 来源与
            #   readiness 关联四种语义，而本机执行器**没有 TCP 连接** —— 进去必然被要求
            #   ACK（`deployment_commit_receipts_missing`）。它们与 `local_assignment`
            #   共用 master 的 node_id/boot_id 作 incarnation，因此传输身份与故障归属
            #   仍是"这台机器"，而不是续航一台虚拟物理机。
            if callable(freeze):
                # ★ 2026-10-26（LOCAL-EXECUTORS-01）：**这一步的产物是执行期段连续性的
                #   唯一来源**。实测（12:48）`bootstrap/execution` 报
                #   `local_executors=[]` ⇒ segments 缺 `[8,24)` ⇒
                #   `execution_segment_coverage_invalid`，而 plan 里四段齐全、
                #   供给 `ready`。需要看清是"plan 里没有"、"master_ids 不匹配"
                #   还是"freeze 返回 None"。
                logger.info(
                    "event=local_executor_freeze_begin plan=%s master_ids=%s",
                    [(item.get("node_id"), str(item.get("executor_id") or ""))
                     for item in capacity_plan.get("assignments", [])],
                    sorted(master_ids),
                )
                for item in capacity_plan.get("assignments", []):
                    executor_id = str(item.get("executor_id") or "")
                    # ★ 2026-10-27：这里**刻意只处理本机节点的附加执行器**，
                    #   而 `executor_id` 为空（主执行器）一律跳过。
                    #
                    #   为什么：主执行器有自己的装载路径（`ModelManager` 走
                    #   `get_selected_model()` 与既有的 worker prepare/commit 链），
                    #   本机执行器则是**独立执行器** —— 它的工件由它自己按
                    #   `QLH_LAYER_GGUF` 加载，不走远端那套工件传输与回执
                    #   （那些是**物理节点之间**的义务）。两者混在同一循环里会互相
                    #   覆盖身份块。
                    #
                    # ⚠ **已知边界（不是缺陷，但要知道）**：本机执行器因此**不参与
                    #   就绪检查**（`scheduler_pipeline` 的 `pipeline_nodes` 用
                    #   `node_id not in master_ids` 把它整体排除）。所以本机 CPU 段
                    #   若加载失败，供给仍可能报 `ready`，失败要到**第一次 stage**
                    #   才具名暴露。当前接受这一点：本机段没有 TCP 回执链，硬造一个
                    #   就绪信号等于把"进程内"伪装成"有握手"，反而更难判真伪。
                    #   要补这个门，得先有本机执行器自己的就绪上报通道。
                    if not executor_id or item.get("node_id") not in master_ids:
                        continue
                    try:
                        frozen_executor = freeze(candidate["candidate_id"], item,
                                                 config_id=config_id, generation=generation)
                    except Exception as exc:
                        logger.warning(
                            "event=local_executor_freeze_failed executor=%s err=%s: %s",
                            executor_id, type(exc).__name__, exc,
                        )
                        return {"status": "rejected", "admitted": False,
                                "reason_code": str(exc), "assignments": [],
                                "transaction_phase": "rejected"}
                    if frozen_executor is not None:
                        local_executors_frozen[executor_id] = frozen_executor
                logger.info(
                    "event=local_executor_freeze_done frozen=%s",
                    sorted(local_executors_frozen),
                )
            else:
                logger.warning("event=local_executor_freeze_unavailable freeze=%r", freeze)
        if capacity_plan is not None:
            if not assignments and not stage_assignments:
                capacity_plan = dict(capacity_plan)
                capacity_plan.update({
                    "status": "rejected",
                    "admitted": False,
                    "reason_code": "pipeline_capacity_workers_unavailable",
                    "assignments": [],
                    "transaction_phase": "rejected",
                })
            elif stage_assignments:
                # Every remote segment participates in the native-load barrier.
                capacity_plan = dict(capacity_plan)
                capacity_plan["assignments"] = [
                    (
                        stage_assignments.get(item.get("node_id"), item)
                        if item.get("node_id") in stage_assignments else item
                    )
                    for item in capacity_plan.get("assignments", [])
                ]
            with self._layer_config_lock:
                # Cancellation can arrive while candidate metadata or capacity
                # planning is being prepared. Check the supply guard under the
                # same lock that installs the transaction, so cancellation either
                # prevents its creation or finds it and aborts it.
                if operation_id:
                    self._model_supply.ensure_prepare_not_cancelled(operation_id)
                previous_control_state = {
                    "retry_state": {
                        node_id: dict(value)
                        for node_id, value in self._layer_config_retry_state.items()
                    },
                    "assignments": self._worker_assignments.snapshot(),
                    "active_plan": (
                        dict(self._active_pipeline_capacity_plan)
                        if self._active_pipeline_capacity_plan else None
                    ),
                }
                self._pipeline_load_transaction = {
                    "config_id": config_id,
                    "generation": generation,
                    "phase": "preparing" if assignments or stage_assignments else "rejected",
                    "plan": dict(capacity_plan),
                    "worker_ids": set(assignments) | set(stage_assignments),
                    "connection_generations": {
                        node: self._assignment_connection_generation(node)
                        for node in stage_assignments
                    },
                    "prepared_nodes": set(),
                    "local_assignment": local_frozen,
                    # ★ 2026-10-25（LOCAL-EXECUTORS-01）：同一物理节点上的**附加执行器**
                    #   （`{executor_id: frozen_assignment}`）。空 ⇒ 与改动前逐位相同。
                    "local_executors": local_executors_frozen,
                    "local_receipt": None,
                    "candidate_id": (
                        str(candidate.get("candidate_id", "") or "")
                        if isinstance(candidate, dict) else ""
                    ),
                    "started_at": time.monotonic(),
                    "deadline": time.monotonic() + PIPELINE_MODEL_SYNC_TIMEOUT,
                    "previous_control_state": previous_control_state,
                    "expected": {
                        node_id: dict(value)
                        for node_id, value in {**assignments, **stage_assignments}.items()
                    },
                    "acks": {},
                    "retry_state": {
                        node_id: {
                            "attempts": 1,
                            "next_retry": time.monotonic() + 5.0,
                        }
                        for node_id in {**assignments, **stage_assignments}
                    },
                    "on_commit": (
                        candidate.get("on_commit")
                        if isinstance(candidate, dict) else None
                    ),
                    "retire_configs": {
                        node_id: dict(value)
                        for node_id, value in releases.items()
                    } if candidate is not None else {},
                }
                if candidate is None:
                    self._active_pipeline_capacity_plan = None
                self._persist_pipeline_lifecycle_locked()
                if capacity_plan.get("status") == "rejected":
                    self._pipeline_recovery_failure = (
                        str(capacity_plan.get("reason_code", "") or "")
                        or "pipeline_recovery_failed"
                    )
        if candidate is not None:
            self._publish_pipeline_candidate_configs(configs)
        else:
            self._publish_layer_configs(configs)
        if assignments:
            logger.info(
                f"分层配置已推送到 {len(assignments)} 个从节点，"
                f"等待加载 ACK (config_id={config_id}, generation={generation})"
            )
        else:
            logger.warning("没有可用的从节点接收分层配置")
        if not isinstance(capacity_plan, dict):
            return None
        result = dict(capacity_plan)
        with self._layer_config_lock:
            transaction = self._pipeline_load_transaction
            if transaction and transaction.get("config_id") == config_id:
                phase = str(transaction.get("phase", "") or "")
                result.update({
                    "config_id": config_id,
                    "generation": generation,
                    "transaction_phase": phase,
                })
                if phase in {"aborted", "invalidated", "rejected"}:
                    result.update({
                        "status": "rejected",
                        "admitted": False,
                        "reason_code": str(
                            transaction.get("reason_code", "")
                            or f"pipeline_transaction_{phase}"
                        ),
                        "reason": str(transaction.get("reason", "") or ""),
                    })
        return result


    def _stage_offer_assignment_ready(
        self, node_id: str, assignment: dict,
    ) -> tuple[bool, str]:
        """Check a Route-A worker without consulting legacy layer ACK state."""
        loaded, reason = self._worker_prepare_assignment_loaded(node_id, assignment)
        if not loaded:
            return False, reason
        provider_factory = getattr(self, "_ensure_remote_task_worker_provider", None)
        if not callable(provider_factory):
            return False, "task_worker_provider_unavailable"
        try:
            provider = provider_factory(node_id)
            capabilities = provider.inspect()
            if not capabilities.healthy:
                return False, "task_worker_provider_unhealthy"
            if "layer_forward" not in capabilities.supported_stage_types:
                return False, "layer_forward_not_advertised"
            snapshot = self._task_worker_control.worker_snapshot(node_id)
            if not snapshot.get("layer_stage_dispatch_enabled"):
                return False, "layer_stage_dispatch_not_admitted"
            raw_caps = snapshot.get("capabilities", {})
            ranges = raw_caps.get("layer_ranges", []) if isinstance(raw_caps, dict) else []
            artifacts = (
                raw_caps.get("layer_artifacts", [])
                if isinstance(raw_caps, dict) else []
            )
            layer_budget = (
                raw_caps.get("layer_budget", {})
                if isinstance(raw_caps, dict) else {}
            )
            requested = (
                int(assignment.get("start_layer", -1)),
                int(assignment.get("end_layer", -1)),
            )
            if artifacts:
                planned_artifact = assignment.get("layer_artifact")
                expected_artifact = (
                    planned_artifact
                    if isinstance(planned_artifact, dict) else None
                )

                def artifact_matches(item):
                    if not (
                        isinstance(item, dict)
                        and isinstance(item.get("layer_range"), (list, tuple))
                        and len(item["layer_range"]) == 2
                        and tuple(int(value) for value in item["layer_range"]) == requested
                    ):
                        return False
                    if expected_artifact is None:
                        return True
                    return all(
                        str(item.get(key, "") or "") == str(
                            expected_artifact.get(key, "") or ""
                        )
                        for key in (
                            "segment_mode", "model_id", "artifact_sha256",
                            "source_model_sha256", "source_model_id",
                            "hidden_size", "tokenizer_sha256",
                        )
                    )

                matches = any(
                    artifact_matches(item)
                    for item in artifacts
                )
                if not matches:
                    return False, "layer_artifact_contract_changed"
            elif isinstance(assignment.get("layer_artifact"), dict):
                return False, "layer_artifact_contract_changed"
            elif ranges and not (
                isinstance(layer_budget, dict) and layer_budget.get("local_cut") is True
            ):
                matches = any(
                    isinstance(item, (list, tuple)) and len(item) == 2
                    and int(item[0]) == requested[0]
                    and int(item[1]) == requested[1]
                    for item in ranges
                )
                if not matches:
                    return False, "layer_range_not_advertised"
            planned_identity = assignment.get("stage_model_identity")
            if isinstance(planned_identity, dict):
                models = raw_caps.get("models", []) if isinstance(raw_caps, dict) else []
                current_identity = next((
                    item for item in models
                    if isinstance(item, dict)
                    and all(
                        str(item.get(key, "") or "")
                        == str(planned_identity.get(key, "") or "")
                        for key in (
                            "model_id", "engine", "format", "revision", "sha256",
                        )
                    )
                ), None) if isinstance(models, list) else None
                if current_identity is None:
                    return False, "stage_model_identity_changed"
                expected_capability_sha256 = str(
                    assignment.get("stage_capability_sha256", "") or ""
                )
                planned_artifact = assignment.get("layer_artifact")
                if expected_capability_sha256 and isinstance(planned_artifact, dict):
                    from model_preflight import stage_capability_sha256

                    if stage_capability_sha256(
                        planned_artifact, current_identity,
                    ) != expected_capability_sha256:
                        return False, "stage_capability_changed"
            return True, "ready"
        except Exception as exc:
            logger.debug(
                "stage offer readiness probe failed node=%s", node_id,
                exc_info=True,
            )
            return False, type(exc).__name__


    def _route_a_stage_model_identity(
        self, node_id: str, assignment: dict | None = None,
    ):
        """Return the physical artifact identity a Route-A stage worker advertised.

        Route A 的层段在**设备**上执行，用的是设备手上那份 GGUF 工件 ⇒ offer 必须
        带该工件的身份（`engine=llama_cpp` / `format=gguf` / 该工件 manifest 的源模型
        摘要），而不是 master 自己那份模型的摘要 —— 后者是 safetensors，两者永远不等。
        `task_worker_adapter._layer_model_matches` 比对的正是 engine/format/sha256。

        取不到时返回 None，由调用方 fail-closed（不退回 master 身份：那必然不匹配）。
        """
        from execution_provider import ModelIdentity

        try:
            status = self._task_worker_control.status(role="master")
        except Exception:
            logger.warning("Route-A stage identity lookup failed", exc_info=True)
            return None
        for worker in (status or {}).get("workers", []):
            if not isinstance(worker, dict):
                continue
            if str(worker.get("node_id", "")) != node_id:
                continue
            capabilities = worker.get("capabilities")
            models = (
                capabilities.get("models") if isinstance(capabilities, dict) else None
            )
            if not isinstance(models, list) or not models:
                return None
            expected_model_id = ""
            expected_sha256 = ""
            artifacts = capabilities.get("layer_artifacts", [])
            planned_artifact = (
                assignment.get("layer_artifact")
                if isinstance(assignment, dict) else None
            )
            if isinstance(planned_artifact, dict):
                expected_model_id = str(planned_artifact.get("model_id", "") or "")
                expected_sha256 = str(
                    planned_artifact.get("artifact_sha256", "") or ""
                )
                # ★ 2026-10-23（#117）：自裁节点（`layer_budget.local_cut`）的工件是
                #   **供给阶段按本次分配区间新裁出来的**，不可能出现在 worker 上一轮
                #   通告的 `layer_artifacts` 里（那里是旧切点的段）。此前要求逐字段
                #   完全匹配 ⇒ 恒返回 None ⇒ prepare 以
                #   `prepare_stage_model_identity_missing` 失败，即使段已裁好并校验。
                #   此时身份已由供给收据权威写入 planned_artifact，直接采用。
                local_cut = bool(
                    isinstance(capabilities, dict)
                    and isinstance(capabilities.get("layer_budget"), dict)
                    and capabilities["layer_budget"].get("local_cut") is True
                )
                current_artifact = None if local_cut else next((
                    item for item in artifacts
                    if isinstance(item, dict)
                    and item.get("layer_range") == planned_artifact.get("layer_range")
                    and all(
                        str(item.get(key, "") or "") == str(
                            planned_artifact.get(key, "") or ""
                        )
                        for key in (
                            "segment_mode", "model_id", "artifact_sha256",
                            "source_model_sha256", "source_model_id",
                            "hidden_size", "tokenizer_sha256",
                        )
                    )
                ), None) if isinstance(artifacts, list) else None
                if not local_cut and current_artifact is None:
                    return None
            elif assignment is not None and isinstance(artifacts, list) and artifacts:
                requested = (
                    int(assignment.get("start_layer", -1)),
                    int(assignment.get("end_layer", -1)),
                )
                artifact = next((
                    item for item in artifacts
                    if isinstance(item, dict)
                    and isinstance(item.get("layer_range"), (list, tuple))
                    and len(item["layer_range"]) == 2
                    and tuple(int(value) for value in item["layer_range"]) == requested
                ), None)
                if artifact is None:
                    return None
                expected_model_id = str(artifact.get("model_id", "") or "")
                expected_sha256 = str(
                    artifact.get("artifact_sha256", "") or ""
                )
            model = next((
                item for item in models
                if isinstance(item, dict)
                and (not expected_model_id or item.get("model_id") == expected_model_id)
                and (not expected_sha256 or item.get("sha256") == expected_sha256)
            ), None)
            if not isinstance(model, dict):
                return None
            try:
                return ModelIdentity(
                    model_id=str(model.get("model_id", "")),
                    engine=str(model.get("engine", "")),
                    format=str(model.get("format", "")),
                    revision=str(model.get("revision", "")),
                    sha256=str(model.get("sha256", "")),
                )
            except ValueError:
                return None
        return None

    def _execute_layer_stage_offer(
        self,
        *,
        node_id: str,
        assignment: dict,
        hidden_states,
        model_identity,
        workflow_id: str,
        request_id: str,
        stage_id: str,
        context_size: int,
        pos_base: int = 0,
        want_hidden: bool = True,
        want_all_tokens: bool = False,
        middle_channel: str = "extract_hidden",
        seq_ids: list[int] | None = None,
        positions: list[int] | None = None,
        cancel_event: threading.Event | None = None,
        request_deadline: RequestDeadline | None = None,
        execution_attempts: list | None = None,
        cleanup_only: bool = False,
        executor_id: str = "",
    ) -> dict:
        """Execute one Route-A v3 layer stage and map its result to a step.

        This helper is intentionally independent from the legacy token loop;
        callers must own the surrounding prefill/decode/KV state machine.
        """
        from execution_provider import StageAttempt, StageRequest
        from task_worker_adapter import remote_provider_id

        raw, n_tokens, n_embd = _hidden_to_raw_f32(hidden_states)
        # ★ 2026-10-07（DIST-NEXT-2）：**offer 前**按 `n_tokens * n_embd * dtype` 预检
        #   wire 大小。输入与中间段输出（`hidden_out_f32`）同尺寸，所以一次预检覆盖往返；
        #   超限时在 reserve/execute 之前以稳定 reason 结束 —— 不再让大 payload 走到
        #   「执行完成后才 `message_too_large`」。
        # ★ 2026-10-09（接口税）：`chunked_input` 探测内含 `control.status(role="master")` 的
        #   **全量 worker capabilities 深拷贝**，而此前每步每段都无条件执行 —— decode 每步
        #   hidden 仅 ~11.5KB，远低于 8.1MB 单帧预算 ⇒ 探测结果必然用不上（白付深拷贝）。
        #   改为只在实际超预算时才查询；fits 时 `_assert_layer_stage_offer_fits_frame` 本就
        #   提前 return（不看该参数），故**语义完全等价**。
        _fits_single_frame = hidden_fits_stage_frame(n_tokens, n_embd, "float32")
        # ★ 2026-10-26（LOCAL-EXECUTORS-01）：**本机执行器没有网络帧**。
        #   帧大小上限是 TCP 传输约束（base64 / 分片 / 二进制体），而本机执行器
        #   走进程内 provider（见下方 `if executor_id:` 分支），根本不序列化成帧。
        #   此前无条件校验 ⇒ 本机 CPU 段在 hidden 稍大时被
        #   `route_a_stage_frame_too_large` 拒绝（实测 11:45:11：CPU 段第一次真正
        #   被 stage，就死在这里，而它并不是远端）。
        if not executor_id:
            # ★ 票 21a（COMM-WIRE-01）：只有 base64 判不下时才查能力（该查询含全量
            #   capabilities 深拷贝），且**一次查询**同时取出分片与二进制两个键 ——
            #   分别调 `_node_declares_*` 会各付一次深拷贝。
            _worker_caps = (
                {}
                if _fits_single_frame and not want_all_tokens
                else _node_capabilities(
                    getattr(self, "_task_worker_control", None), str(node_id)
                )
            )
            _assert_layer_stage_offer_fits_frame(
                node_id=str(node_id), n_tokens=n_tokens, n_embd=n_embd,
                # ★ 票 21a：对端声明 `binary_hidden` ⇒ 帧走原始字节体，单帧容量 +≈33%。
                binary_hidden=_worker_caps.get("binary_hidden") is True,
                # ★ 2026-10-08（DIST-NEXT-2b）：对端声明 `stage_chunked_input` ⇒ 超预算的
                #   hidden 交由 provider 切 `stage_chunk` 分片发送，不在此提前拒绝。
                chunked_input=_worker_caps.get("stage_chunked_input") is True,
            )
        hidden_spec = {
            "n_tokens": n_tokens,
            "n_embd": n_embd,
            "dtype": "float32",
        }
        stage_fields = {
            "layer_range": [
                int(assignment["start_layer"]),
                int(assignment["end_layer"]),
            ],
            "handoff_at": int(assignment["end_layer"]),
            "hidden_sha256": hashlib.sha256(raw).hexdigest(),
            "hidden_spec": hidden_spec,
            "middle_channel": middle_channel,
        }
        from worker_prepare_contract import stage_identity
        # ★ 2026-10-26（LOCAL-EXECUTORS-01）：本机执行器的身份块键名是 `identity`
        #   （`freeze_pipeline_candidate_assignment` 写入），远端才是
        #   `worker_prepare_identity`。`Segment.freeze` 早已按此回退，这里此前硬取
        #   ⇒ 本机 CPU 段 stage 时 `KeyError`。两处现已一致。
        identity = (assignment.get("worker_prepare_identity")
                    or assignment.get("identity") or {})
        ready, reason = self._worker_prepare_assignment_loaded(node_id, assignment)
        if not ready:
            raise RuntimeError(reason)
        stage_fields["prepare_identity"] = stage_identity(identity)
        if seq_ids is not None:
            stage_fields["seq_ids"] = list(seq_ids)
        if positions is not None:
            stage_fields["positions"] = list(positions)
        if want_all_tokens:
            from qlh_core.contracts.token_output import validate_token_output_request, ALL_POSITIONS
            # Validate caller intent even if this peer requires the scalar fallback.
            validate_token_output_request({**stage_fields, "token_output": ALL_POSITIONS},
                                          {"want_hidden": want_hidden})
            if _worker_caps.get("multi_position_argmax") is True:
                stage_fields["token_output"] = ALL_POSITIONS
        # ★ 2026-10-03：把每步的 hidden 摘要打出来。跨机 D→L 的分叉定位需要与 relay
        #   路径（同层范围、同 prompt）的输出**逐位对照**，而此前链路上没有任何可对照
        #   的中间量 —— 只能看到"最终 token 不同"，无法判断差异出在 master 段还是设备段。
        logger.info(
            "Route-A stage handoff: node=%s stage=%s tokens=%d hidden_sha256=%s "
            "workflow_id=%s stage_request_id=%s",
            node_id, stage_id, n_tokens, stage_fields["hidden_sha256"],
            workflow_id, request_id,
        )
        # ★ 2026-10-25（LOCAL-EXECUTORS-01）：**本机执行器走进程内 provider，不发帧**。
        #   本机执行器（例如 master 上的 CPU 段）与远端 worker **共用同一套**
        #   `StageRequest`/`ProviderExecutor` 契约，但没有 TCP 连接 ⇒ 不能走
        #   `RemoteFullWorkerProvider`（它按 node_id 发帧，并把故障归属绑到连接世代上）。
        #   注意 provider 选择必须提到 `StageRequest` 构造**之前**：`provider_id` 要与
        #   最终选中的 provider 一致，否则 `reserve()` 会以 `provider_request_mismatch` 拒绝。
        #   `_ensure_local_task_worker_provider` 取 `_task_worker_stage_lock`（RLock），
        #   与调用方可能已持有的同一把锁可重入。
        if executor_id:
            provider = self._ensure_local_task_worker_provider(str(node_id), str(executor_id))
            provider_id = str(provider.provider_id)
            logger.info(
                "Route-A stage handoff (local executor): node=%s executor=%s stage=%s "
                "tokens=%d hidden_sha256=%s",
                node_id, executor_id, stage_id, n_tokens, stage_fields["hidden_sha256"],
            )
        else:
            provider = self._ensure_remote_task_worker_provider(str(node_id))
            provider_id = remote_provider_id(str(node_id))
        request = StageRequest(
            workflow_id=str(workflow_id),
            request_id=str(request_id),
            stage_id=str(stage_id),
            stage_type="layer_forward",
            provider_id=provider_id,
            dependencies={},
            root_input={
                "hidden_f32": base64.b64encode(raw).decode("ascii"),
                "context_size": int(context_size),
                "pos_base": int(pos_base),
                "want_hidden": bool(want_hidden),
                **({"cleanup_only": True} if cleanup_only else {}),
            },
            model_identity=model_identity,
            stage_fields=stage_fields,
            runtime_context={"pipeline_route": "route_a"},
            # ★ 2026-10-26（LOCAL-EXECUTORS-01）：把执行器身份交给执行侧 ——
            #   它需要据此判断"这一段的工件由谁持有"（本机附加执行器有自己的
            #   GGUF 层段，不能落到全局主模型）。缺省 "" 与改动前逐位等价。
            executor_id=str(executor_id or ""),
        )
        try:
            reservation = provider.reserve(request)
        except Exception as exc:
            # ★ 2026-10-03：`reserve()` 有七个拒绝分支（provider_request_mismatch /
            #   unsupported_stage_type / stage_dispatch_not_admitted /
            #   model_identity_required / model_identity_mismatch /
            #   remote_worker_unavailable / remote_worker_busy），但它们都只抛一句笼统
            #   消息 ⇒ 跨机层段失败时看不出是哪一条。把 code 与两端身份一并打出来。
            logger.warning(
                "Route-A stage reserve 被拒: node=%s code=%s detail=%s "
                "requested_identity=%s advertised=%s",
                node_id, getattr(exc, "code", ""), exc,
                getattr(request.model_identity, "snapshot", lambda: None)(),
                [
                    model for model in (
                        provider._snapshot().get("capabilities", {}) or {}
                    ).get("models", [])
                ] if hasattr(provider, "_snapshot") else None,
                exc_info=True,
            )
            raise
        # ★ 2026-10-24（#117）：租约此前硬编码 60 秒。远端每个 stage 尝试只有这一段
        #   租约，三机首次推理（冷启动 + KV 初始化 + 网络往返）必然超过它 ⇒ 租约到期
        #   ⇒ `/api/chat` 实测 63 秒返回 `lease_expired`（60 s + 开销，完全吻合）。
        #   改为设置（默认 300 s，见 `config.pipeline_stage_lease_seconds`）。
        lease_expires_at = time.time() + pipeline_stage_lease_seconds()
        if request_deadline is not None:
            request_deadline.require_remaining()
            lease_expires_at = min(
                lease_expires_at, request_deadline.expires_at_epoch,
            )
        attempt = StageAttempt(
            attempt_id=f"att_{uuid.uuid4().hex}",
            request=request,
            provider_id=request.provider_id,
            lease_id=f"lease_{uuid.uuid4().hex}",
            lease_epoch=1,
            lease_expires_at=lease_expires_at,
            request_deadline=request_deadline,
        )
        if execution_attempts is not None:
            execution_attempts.append((provider, attempt.attempt_id))
        try:
            result = provider.execute(
                attempt,
                reservation,
                cancel_event or threading.Event(),
            )
            if execution_attempts is not None:
                # A returned StageResult already proves physical execution
                # stopped. Its provider cache expires after release; retain
                # only attempts that still need a late stop/cancel receipt.
                execution_attempts.remove((provider, attempt.attempt_id))
            if cleanup_only:
                if result.output != {"kv_cleared": True}:
                    raise RuntimeError("execution_kv_cleanup_receipt_invalid")
                value = {"kind": "cleanup", "kv_cleared": True}
            else:
                value = _layer_stage_result_to_pipeline_value(result.output, hidden_spec=hidden_spec,
                    token_output=stage_fields.get("token_output"), positions=positions)
            # A dispatch log precedes execution and cannot prove a handoff
            # succeeded. Emit this receipt only after the result is validated.
            logger.info(
                "Route-A stage completed: %s",
                json.dumps({
                    "workflow_id": str(workflow_id),
                    "stage_request_id": str(request_id),
                    "stage_id": str(stage_id),
                    "node_id": str(node_id),
                    "layer_range": stage_fields["layer_range"],
                    **stage_fields["prepare_identity"],
                    "tokens": n_tokens,
                    "hidden_sha256": stage_fields["hidden_sha256"],
                    "result_kind": value["kind"],
                    "completed_at": time.time(),
                }, sort_keys=True, separators=(",", ":")),
            )
            return value
        finally:
            provider.release(reservation.reservation_id)


    def _assignment_connection_generation(self, node_id: str) -> int:
        server = getattr(self, "_tcp_server", None)
        getter = getattr(server, "get_authenticated_peer_epoch", None)
        if not callable(getter):
            return 0
        try:
            return int(getter(node_id) or 0)
        except (TypeError, ValueError):
            return 0


    def _publish_layer_configs(self, configs: dict[str, dict]) -> None:
        """Register every assignment/release before sending so both are retried."""
        if not configs:
            return
        with self._layer_config_lock:
            for node_id, config in configs.items():
                payload = dict(config)
                current = self._worker_assignments.state(node_id)
                identity = str(payload.get("assignment_id", "") or "")
                if (
                    current is not None
                    and not current.terminal
                    and identity
                    and identity == current.assignment_id
                ):
                    state = self._worker_assignments.update_expected(
                        node_id, payload,
                    )
                else:
                    state = self._worker_assignments.begin(
                        node_id,
                        assignment_id=identity,
                        expected_config=payload,
                        connection_generation=(
                            self._assignment_connection_generation(node_id)
                        ),
                    )
                payload = self._worker_assignments.expected(node_id)
                configs[node_id] = payload
                self._layer_config_retry_state[node_id] = {
                    "attempts": 1,
                    "next_retry": time.monotonic() + 5.0,
                }
                if state is None:
                    raise RuntimeError(
                        f"cannot publish terminal assignment without renewal: {node_id}"
                    )
        self._start_layer_config_retry_monitor()
        for node_id, config in configs.items():
            try:
                self._send_pipeline_worker_config(node_id, config)
            except Exception:
                logger.warning(
                    "分层配置首次发送失败，将由退避线程重试: node=%s",
                    node_id,
                    exc_info=True,
                )


    def _publish_pipeline_candidate_configs(
        self, configs: dict[str, dict],
    ) -> None:
        """Send candidate configs without replacing the active control view."""
        if not configs:
            return
        with self._layer_config_lock:
            transaction = self._pipeline_load_transaction
            for node_id, config in configs.items():
                payload = dict(config)
                payload.setdefault("assignment_id", new_assignment_id())
                configs[node_id] = payload
                if (
                    isinstance(transaction, dict)
                    and transaction.get("config_id") == payload.get("config_id")
                    and transaction.get("candidate_id")
                ):
                    transaction.setdefault("expected", {})[node_id] = dict(payload)
        self._start_layer_config_retry_monitor()
        for node_id, config in configs.items():
            try:
                self._send_pipeline_worker_config(node_id, config)
            except Exception:
                logger.warning(
                    "候选分层配置首次发送失败，将由退避线程重试: node=%s",
                    node_id,
                    exc_info=True,
                )


    def _clear_layer_config_state(self, node_id: str) -> None:
        """清除节点的层配置期望、ACK 和 ready 状态。"""
        with self._layer_config_lock:
            self._layer_config_retry_state.pop(node_id, None)
            self._worker_assignments.release(
                node_id, reason_code=REASON_CONFIG_CLEARED,
            )


    def _record_pipeline_candidate_abort_cleanup_locked(
        self, transaction: dict, *, generation: int, reason_code: str,
    ) -> list[dict]:
        """Record every terminal abort owed by one isolated candidate."""
        candidate_id = str(transaction.get("candidate_id", "") or "")
        config_id = str(transaction.get("config_id", "") or "")
        if not candidate_id or not config_id:
            return []
        candidate_expected = transaction.get("expected", {})
        model_id = str(transaction.get("plan", {}).get("model_id", "") or "")
        records = [
            self._record_pipeline_model_cleanup_locked(
                node_id="master",
                candidate_config_id=candidate_id,
                generation=generation,
                action="abort",
            )
        ]
        for node_id in set(transaction.get("worker_ids", set())):
            records.append(
                self._record_pipeline_model_cleanup_locked(
                    node_id=node_id,
                    candidate_config_id=config_id,
                    generation=generation,
                    action="abort",
                    assignment_id=str(
                        candidate_expected.get(node_id, {}).get(
                            "assignment_id", ""
                        )
                        if isinstance(candidate_expected, dict) else ""
                    ),
                    payload={
                        **({"worker_prepare_identity": candidate_expected[node_id]["worker_prepare_identity"]}
                           if candidate_expected.get(node_id, {}).get("worker_prepare_identity") else {}),
                        "release": True,
                        "abort": True,
                        "aborted_config_id": config_id,
                        "model_id": model_id,
                        "reason_code": reason_code,
                    },
                )
            )
        return records


    def _abort_pipeline_load_transaction(
        self, config_id: str, reason_code: str, reason: str = "", *,
        _deferred_dispatch: list | None = None,
    ) -> None:
        """Abort state now; optionally dispatch cleanup after an outer guard."""
        cleanup_records = []
        with self._layer_config_lock:
            transaction = self._pipeline_load_transaction
            if not transaction or transaction.get("config_id") != config_id:
                return
            retire_published = bool(transaction.get("published") or transaction.get("phase") == "ready")
            if transaction.get("phase") in {"aborted", "rejected", "invalidated"} and not retire_published:
                return
            if not retire_published:
                worker_ids = set(transaction.get("worker_ids", set()))
                candidate_id = str(transaction.get("candidate_id", "") or "")
                legacy_host = self._host if not candidate_id else None
                self._layer_config_generation = max(
                    self._layer_config_generation + 1,
                    time.time_ns(),
                )
                generation = self._layer_config_generation
                transaction["phase"] = "aborted"
                transaction["reason_code"] = reason_code
                transaction["reason"] = reason
                model_id = str(transaction.get("plan", {}).get("model_id", "") or "")
                previous = transaction.get("previous_control_state")
                if isinstance(previous, dict) and not candidate_id:
                    self._layer_config_retry_state.clear()
                    self._layer_config_retry_state.update({
                        node_id: dict(value)
                        for node_id, value in previous.get("retry_state", {}).items()
                        if isinstance(value, dict)
                    })
                    self._worker_assignments.restore(
                        previous.get("assignments", {})
                    )
                    active_plan = previous.get("active_plan")
                    self._active_pipeline_capacity_plan = (
                        dict(active_plan) if isinstance(active_plan, dict) else None
                    )
                elif not candidate_id:
                    self._active_pipeline_capacity_plan = None
                if candidate_id:
                    cleanup_records.extend(
                        self._record_pipeline_candidate_abort_cleanup_locked(
                            transaction,
                            generation=generation,
                            reason_code=reason_code,
                        )
                    )
                self._persist_pipeline_lifecycle_locked()
        if retire_published:
            self._invalidate_pipeline_load_transaction(
                reason_code, reason, expected_config_id=config_id,
                _deferred_dispatch=_deferred_dispatch,
            )
            return
        def dispatch():
            if candidate_id:
                for record in cleanup_records:
                    self._dispatch_pipeline_model_cleanup(record)
            else:
                execution_lock = getattr(
                    legacy_host, "full_chat_execution_lock", self._layer_execution_lock,
                )
                with execution_lock:
                    # Legacy loads share this execution lock. Recheck only after
                    # waiting: an aborted transaction may have been superseded
                    # before its deferred cleanup reached the native boundary.
                    with self._layer_config_lock:
                        cleanup_current = (
                            self._host is legacy_host
                            and self._pipeline_load_transaction is transaction
                            and transaction.get("phase") == "aborted"
                        )
                    if cleanup_current:
                        abort_materialization = getattr(
                            legacy_host, "abort_pipeline_materialization", None,
                        )
                        if callable(abort_materialization):
                            try:
                                abort_materialization()
                                # Update only the captured, guarded host. A new
                                # materialization cannot run until lock release.
                                legacy_host.model_loaded = False
                            except Exception:
                                logger.warning("主节点回滚流水线层段失败", exc_info=True)
                abort_id = uuid.uuid4().hex
                native_records = []
                with self._layer_config_lock:
                    for node, expected in transaction.get("expected", {}).items():
                        identity = expected.get("worker_prepare_identity")
                        if isinstance(identity, dict):
                            native_records.append(self._record_pipeline_model_cleanup_locked(
                                node_id=node, candidate_config_id=config_id, generation=generation,
                                action="abort", assignment_id=identity["assignment_id"],
                                payload={"worker_prepare_identity": identity, "release": True, "abort": True,
                                         "aborted_config_id": config_id},
                            ))
                    self._persist_pipeline_lifecycle_locked()
                for record in native_records:
                    self._dispatch_pipeline_model_cleanup(record)
                configs = {
                    node_id: {
                        "node_id": node_id,
                        "config_id": abort_id,
                        "generation": generation,
                        "release": True,
                        "abort": True,
                        "aborted_config_id": config_id,
                        "model_id": model_id,
                        "reason_code": reason_code,
                    }
                    for node_id in worker_ids
                    if not transaction.get("expected", {}).get(node_id, {}).get("worker_prepare_identity")
                }
                for node_id, payload in configs.items():
                    try:
                        self._tcp_server.send_layer_config(node_id, payload)
                    except Exception:
                        logger.warning(
                            "发送候选事务中止失败: node=%s config=%s",
                            node_id, config_id, exc_info=True,
                        )
            logger.error(
                "流水线加载事务已中止: config=%s reason_code=%s reason=%s",
                config_id, reason_code, reason,
            )

        if _deferred_dispatch is None:
            dispatch()
        else:
            _deferred_dispatch.append(dispatch)


    def _invalidate_pipeline_load_transaction(
        self, reason_code: str = "pipeline_model_changed", reason: str = "", *, expected_config_id: str = "",
        _deferred_dispatch: list | None = None,
    ) -> None:
        """Fence a pipeline transaction before replacing the local model."""
        cleanup_records = []
        with self._layer_config_lock:
            transaction = self._pipeline_load_transaction
            if expected_config_id and (not transaction or transaction.get("config_id") != expected_config_id):
                return
            if transaction and (transaction.get("phase") not in {
                "aborted", "rejected", "invalidated",
            } or transaction.get("published")):
                published = bool(transaction.get("published") or transaction.get("phase") == "ready")
                transaction["phase"] = "invalidated"
                transaction["reason_code"] = reason_code
                transaction["reason"] = reason
                self._layer_config_generation = max(
                    self._layer_config_generation + 1,
                    time.time_ns(),
                )
                # Ready is the publication boundary, even if finalize still
                # owes cleanup of the previous runtime. Retiring that published
                # assignment cannot roll it back or abort a finalized candidate.
                # Keep existing finalize obligations fenced and retryable; the
                # model-change owner releases the active runtime separately.
                if published:
                    for node, expected in transaction.get("expected", {}).items():
                        identity = expected.get("worker_prepare_identity")
                        if isinstance(identity, dict):
                            cleanup_records.append(self._record_pipeline_model_cleanup_locked(
                                node_id=node, candidate_config_id=identity["config_id"],
                                generation=self._layer_config_generation, action="release",
                                assignment_id=identity["assignment_id"],
                                payload={"worker_prepare_identity": identity, "release": True},
                            ))
                if not published:
                    cleanup_records.extend(
                        self._record_pipeline_candidate_abort_cleanup_locked(
                            transaction,
                            generation=self._layer_config_generation,
                            reason_code=reason_code,
                        )
                    )
                self._persist_pipeline_lifecycle_locked()
            self._active_pipeline_capacity_plan = None
            for node in self._worker_assignments.expected_configs():
                # Revoke execution immediately. The separate native cleanup
                # ledger keeps ownership until actual unload is acknowledged.
                self._worker_assignments.release(node, reason_code=reason_code, aborted=True)
            self._prepared_layer_configs.clear()
            self._layer_config_inflight.clear()
        def dispatch():
            for record in cleanup_records:
                self._dispatch_pipeline_model_cleanup(record)

        if _deferred_dispatch is None:
            dispatch()
        else:
            _deferred_dispatch.append(dispatch)


    def _commit_pipeline_load_transaction(self, config_id: str) -> None:
        """Materialize the local candidate, then ask prepared workers to commit."""
        with self._layer_config_lock:
            transaction = self._pipeline_load_transaction
            if (
                not transaction
                or transaction.get("config_id") != config_id
                or transaction.get("phase") != "preparing"
            ):
                return
            plan = dict(transaction.get("plan", {}))
            worker_ids = set(transaction.get("worker_ids", set()))
            candidate_id = str(transaction.get("candidate_id", "") or "")
            legacy_host = self._host if not candidate_id else None
            candidate_expected = transaction.get("expected", {})
            expected = {
                node_id: dict(
                    candidate_expected.get(node_id, {})
                    if candidate_id
                    else self._worker_assignments.expected(node_id)
                )
                for node_id in worker_ids
            }
            transaction["phase"] = "committing_local"
            self._persist_pipeline_lifecycle_locked()
            if not self._pipeline_lifecycle_persist_ok:
                return

        master_ids = {"master", self.get_effective_node_id()}
        local_assignment = next((
            item for item in plan.get("assignments", [])
            if item.get("node_id") in master_ids
        ), None)
        # ★ 2026-10-24（#125）：**取证"跨事务能否跳过本地物化"**。
        #   `materialize_pipeline_candidate` 的幂等只在**单个事务内**有效（靠
        #   `transaction.materialization_key`），而每个新供给都是新事务 ⇒ 总要对同一份
        #   配置重新加载段并重跑 torch.compile（实测 138 s，占供给最大头，且与上一轮
        #   完全重复）。这里先记录"当前已加载状态"与"本次要加载的候选/区间"，用真实
        #   供给确认二者一致后再启用跳过 —— 直接跳过若判据不严会导致静默加载错段。
        try:
            _local_range = (
                [local_assignment.get("start_layer"), local_assignment.get("end_layer")]
                if isinstance(local_assignment, dict) else None
            )
            _active_plan = self._active_pipeline_capacity_plan or {}
            logger.info(
                "event=pipeline_candidate_materialize_probe config=%s candidate=%s "
                "host_prepared=%s local_range=%s active_config=%s active_generation=%s "
                "active_phase=%s",
                config_id, candidate_id,
                bool(getattr(self._host, "is_pipeline_prepared", False)),
                _local_range, _active_plan.get("config_id"), _active_plan.get("generation"),
                _active_plan.get("transaction_phase"),
            )
        except Exception:  # 诊断本身绝不影响主流程
            pass
        try:
            if candidate_id:
                materialize_candidate = getattr(
                    self._host, "materialize_pipeline_candidate", None,
                )
                if not callable(materialize_candidate):
                    raise RuntimeError(
                        "model host does not support candidate materialization"
                    )
                if local_assignment is None:
                    materialize_candidate(
                        candidate_id,
                        start_layer=None,
                        end_layer=None,
                        has_embedding=False,
                        has_lm_head=False,
                        total_layers=int(plan.get("total_layers", 0) or 0),
                        model_id=str(plan.get("model_id", "") or ""),
                    )
                else:
                    local_result = materialize_candidate(
                        candidate_id,
                        start_layer=int(local_assignment["start_layer"]),
                        end_layer=int(local_assignment["end_layer"]),
                        has_embedding=bool(
                            local_assignment.get("has_embedding")
                        ),
                        has_lm_head=bool(
                            local_assignment.get("has_lm_head")
                        ),
                        total_layers=int(plan.get("total_layers", 0) or 0),
                        model_id=str(plan.get("model_id", "") or ""),
                        **({"load_assignment": transaction["local_assignment"]}
                           if transaction.get("local_assignment") else {}),
                    )
                    if transaction.get("local_assignment"):
                        from qlh_core.contracts.deployment import require_runtime_receipt
                        receipt = local_result.get("runtime_receipt")
                        require_runtime_receipt(transaction["local_assignment"], receipt)
                        transaction["local_receipt"] = receipt
            else:
                execution_lock = getattr(
                    legacy_host, "full_chat_execution_lock", self._layer_execution_lock,
                )
                with execution_lock:
                    # Match the deferred legacy abort's execution -> config
                    # order. No native/stage operation runs under config or
                    # registration locks, and stale commits cannot reload later.
                    with self._layer_config_lock:
                        if (self._host is not legacy_host
                                or self._pipeline_load_transaction is not transaction
                                or transaction.get("phase") != "committing_local"):
                            return
                    # Compatibility for legacy callers that do not own an isolated
                    # ModelHost candidate yet.
                    if local_assignment is not None and not getattr(
                        legacy_host, "is_pipeline_prepared", False
                    ):
                        prepare_local = getattr(
                            legacy_host, "prepare_pipeline_model", None,
                        )
                        local_model_path = (
                            getattr(legacy_host, "_full_model_path", None)
                            or getattr(legacy_host, "_model_path", None)
                            or getattr(legacy_host, "model_path", None)
                        )
                        if callable(prepare_local) and local_model_path:
                            prepare_local(
                                model_id=str(plan.get("model_id", "") or ""),
                                model_path=str(local_model_path),
                                quant_type=getattr(legacy_host, "quant_type", None),
                                layer_range=(
                                    int(local_assignment["start_layer"]),
                                    int(local_assignment["end_layer"]),
                                ),
                                model_sha256=None,
                            )
                    prepare_tokenizer = getattr(
                        legacy_host, "prepare_pipeline_tokenizer", None,
                    )
                    if callable(prepare_tokenizer):
                        prepare_tokenizer()
                    if local_assignment is not None:
                        legacy_host.load_layer_range(
                                int(local_assignment["start_layer"]),
                                int(local_assignment["end_layer"]),
                            has_embedding=bool(
                                local_assignment.get("has_embedding")
                            ),
                            has_lm_head=bool(
                                local_assignment.get("has_lm_head")
                            ),
                            model_path=getattr(
                                legacy_host, "_full_model_path", None,
                            ),
                            quant_type=getattr(legacy_host, "quant_type", None),
                            total_layers=int(plan.get("total_layers", 0) or 0),
                            model_id=str(plan.get("model_id", "") or ""),
                        )
        except Exception as exc:
            self._abort_pipeline_load_transaction(
                config_id, "pipeline_local_commit_failed", str(exc)
            )
            return

        with self._layer_config_lock:
            transaction = self._pipeline_load_transaction
            if (
                not transaction
                or transaction.get("config_id") != config_id
                or transaction.get("phase") != "committing_local"
            ):
                logger.info(
                    "discarding superseded pipeline commit: config=%s phase=%s",
                    config_id,
                    transaction.get("phase", "") if transaction else "missing",
                )
                return
            changed_workers = [
                node_id for node_id, item in expected.items()
                if item.get("worker_prepare_identity")
                and transaction.get("connection_generations", {}).get(node_id)
                != self._assignment_connection_generation(node_id)
            ]
        if changed_workers:
            self._abort_pipeline_load_transaction(
                config_id, "pipeline_worker_connection_changed",
                "worker connection changed during local materialization: " + ", ".join(sorted(changed_workers)),
            )
            return

        commit_configs = {}
        for node_id, item in expected.items():
            if not item or item.get("release"):
                continue
            item["phase"] = "commit"
            commit_configs[node_id] = item
        if not commit_configs:
            self._abort_pipeline_load_transaction(
                config_id, "pipeline_commit_workers_missing",
                "prepared worker set disappeared before commit",
            )
            return
        with self._layer_config_lock:
            transaction = self._pipeline_load_transaction
            if (not transaction or transaction.get("config_id") != config_id
                    or transaction.get("phase") != "committing_local"):
                return
            if transaction and transaction.get("config_id") == config_id:
                transaction["phase"] = "committing"
                transaction["expected"] = {
                    node_id: dict(value)
                    for node_id, value in commit_configs.items()
                }
                if candidate_id:
                    transaction["retry_state"] = {
                        node_id: {
                            "attempts": 1,
                            "next_retry": time.monotonic() + 5.0,
                        }
                        for node_id in commit_configs
                    }
                self._persist_pipeline_lifecycle_locked()
                persist_ok = self._pipeline_lifecycle_persist_ok
        if not persist_ok:
            self._abort_pipeline_load_transaction(config_id, "pipeline_lifecycle_persist_failed")
            return
        if candidate_id:
            self._publish_pipeline_candidate_configs(commit_configs)
        else:
            self._publish_layer_configs(commit_configs)
        logger.info(
            "流水线 prepare 全部通过，已下发 commit: config=%s workers=%s",
            config_id, sorted(commit_configs),
        )


    def _complete_pipeline_load_transaction(self, config_id: str) -> None:
        """Atomically publish a fully ready candidate and retire old state."""
        with self._layer_config_lock:
            transaction = self._pipeline_load_transaction
            if (
                not transaction
                or transaction.get("config_id") != config_id
                or transaction.get("phase") != "committing_global"
            ):
                return
            plan = dict(transaction.get("plan", {}))
            candidate_id = str(transaction.get("candidate_id", "") or "")
            worker_ids = set(transaction.get("worker_ids", set()))
            expected = {
                node_id: dict(value)
                for node_id, value in transaction.get("expected", {}).items()
                if isinstance(value, dict)
            }
            candidate_acks = {
                node_id: dict(value)
                for node_id, value in transaction.get("acks", {}).items()
                if isinstance(value, dict)
            }
            retire_configs = {
                node_id: dict(value)
                for node_id, value in transaction.get(
                    "retire_configs", {}
                ).items()
                if isinstance(value, dict)
            }
            generation = int(transaction.get("generation", 0) or 0)
            on_commit = transaction.get("on_commit")

        reshard_result = None
        active_plan = dict(plan)
        active_plan.update({
            "computed_at": time.time(),
            "transaction_phase": "ready",
            "config_id": config_id,
            "generation": generation,
        })
        cleanup_records = []
        try:
            with self._host.full_chat_execution_lock:
                # Mode changes can invalidate this snapshot while we wait for
                # full-chat. Fence publication before swapping any runtime or
                # advancing the irreversible reshard epoch.
                with self._layer_config_lock:
                    transaction = self._pipeline_load_transaction
                    if (
                        not transaction
                        or transaction.get("config_id") != config_id
                        or transaction.get("phase") != "committing_global"
                    ):
                        return
                if not self.get_distributed_inference_enabled():
                    reason = "local mode selected before pipeline publication"
                    if candidate_id:
                        self._abort_pipeline_load_transaction(
                            config_id, "distributed_inference_disabled", reason,
                        )
                    else:
                        # A legacy transaction has no isolated runtime to abort;
                        # never tear down the now-local model during rejection.
                        self._invalidate_pipeline_load_transaction(
                            reason_code="distributed_inference_disabled",
                            reason=reason,
                        )
                    return
                if candidate_id:
                    commit_candidate = getattr(
                        self._host, "commit_pipeline_candidate", None,
                    )
                    if not callable(commit_candidate):
                        raise RuntimeError(
                            "model host does not support candidate commit"
                        )
                    commit_candidate(candidate_id)

                with self._layer_config_lock:
                    transaction = self._pipeline_load_transaction
                    if (not transaction or transaction.get("config_id") != config_id
                            or transaction.get("phase") != "committing_global"
                            or transaction.get("cancel_requested")):
                        raise RuntimeError("pipeline transaction superseded before reshard commit")
                    for node, item in expected.items():
                        identity = item.get("worker_prepare_identity")
                        if isinstance(identity, dict) and (
                            node not in self._connected_client_ids()
                            or node in self._pipeline_worker_opt_out
                            or transaction.get("connection_generations", {}).get(node) != self._assignment_connection_generation(node)
                            or transaction.get("acks", {}).get(node, {}).get("worker_prepare_identity") != identity
                            or transaction.get("acks", {}).get(node, {}).get("loaded") is not True
                            or transaction.get("acks", {}).get(node, {}).get("phase") != "commit"
                        ):
                            raise RuntimeError("worker_native_ready_revoked_before_publication")

                    if getattr(self, "_model_control", None) is not None:
                        # Revalidate authority, inputs, and exact local/remote
                        # commit observations before the irreversible epoch CAS.
                        self._save_model_control_lifecycle(self._pipeline_lifecycle_snapshot_locked())
                    # Candidate commit is reversible until finalize.  Cross the
                    # reshard CAS only after the local runtime can be restored on
                    # failure; the coordinator has no reverse-epoch operation.
                    reshard_result = self._commit_ready_pipeline_reshard(active_plan)
                    if reshard_result is False:
                        raise RuntimeError("pipeline_reshard_commit_failed")

                    transaction = self._pipeline_load_transaction
                    if (
                        not transaction
                        or transaction.get("config_id") != config_id
                        or transaction.get("phase") != "committing_global"
                    ):
                        raise RuntimeError(
                            "pipeline transaction superseded before publication"
                        )
                    for node, item in expected.items():
                        identity = item.get("worker_prepare_identity")
                        if isinstance(identity, dict) and (
                            node not in self._connected_client_ids()
                            or node in self._pipeline_worker_opt_out
                            or transaction.get("connection_generations", {}).get(node) != self._assignment_connection_generation(node)
                            or transaction.get("acks", {}).get(node, {}).get("worker_prepare_identity") != identity
                            or transaction.get("acks", {}).get(node, {}).get("loaded") is not True
                            or transaction.get("acks", {}).get(node, {}).get("phase") != "commit"
                        ):
                            raise RuntimeError("worker_native_ready_revoked_before_publication")

                    for retired_node, old in self._worker_assignments.expected_configs().items():
                        identity = old.get("worker_prepare_identity")
                        if retired_node not in worker_ids and isinstance(identity, dict):
                            cleanup_records.append(self._record_pipeline_model_cleanup_locked(
                                node_id=retired_node, candidate_config_id=identity["config_id"],
                                generation=generation, action="release", assignment_id=identity["assignment_id"],
                                payload={"worker_prepare_identity": identity, "release": True},
                            ))
                    # Freeze all post-publication retirement obligations in the
                    # same SQLite transaction as the active pointer. Native work
                    # has completed; only committed publication grants admission.
                    if candidate_id:
                        self._record_pipeline_model_cleanup_locked(node_id="master",
                            candidate_config_id=candidate_id, generation=generation, action="finalize")
                    for node_id in worker_ids:
                        self._record_pipeline_model_cleanup_locked(node_id=node_id,
                            candidate_config_id=config_id, generation=generation, action="finalize",
                            assignment_id=str(expected.get(node_id, {}).get("assignment_id", "")),
                            payload={**({"worker_prepare_identity": expected[node_id]["worker_prepare_identity"]}
                                      if expected.get(node_id, {}).get("worker_prepare_identity") else {}),
                                     "finalize": True, "finalized_config_id": config_id})
                    if getattr(self, "_model_control", None) is not None:
                        publication = self._pipeline_lifecycle_snapshot_locked()
                        publication.update(phase="ready", published=True)
                        self._save_model_control_lifecycle(publication)

                    for node_id in worker_ids:
                        item = expected.get(node_id)
                        if not item:
                            continue
                        self._layer_config_retry_state.pop(node_id, None)
                        self._worker_assignments.begin(
                            node_id,
                            assignment_id=str(
                                item.get("assignment_id", "") or ""
                            ),
                            config_id=config_id,
                            connection_generation=(
                                transaction.get("connection_generations", {}).get(node_id,
                                    self._assignment_connection_generation(node_id))
                            ),
                            expected_config=item,
                            acknowledgement=candidate_acks.get(node_id, {}),
                        )
                        self._worker_assignments.transition(
                            node_id,
                            phase=PHASE_READY,
                            reason_code=REASON_CONFIG_ACKED,
                        )
                    transaction["phase"] = "ready"
                    transaction["published"] = True
                    transaction["deadline"] = 0.0
                    self._active_pipeline_capacity_plan = dict(active_plan)
                    self._local_pipeline_steps.clear()
                    if candidate_id:
                        cleanup_records.append(
                            self._record_pipeline_model_cleanup_locked(
                                node_id="master",
                                candidate_config_id=candidate_id,
                                generation=generation,
                                action="finalize",
                            )
                        )
                    finalize_payloads = {}
                    for node_id in worker_ids:
                        record = self._record_pipeline_model_cleanup_locked(
                            node_id=node_id,
                            candidate_config_id=config_id,
                            generation=generation,
                            action="finalize",
                            assignment_id=str(
                                expected.get(node_id, {}).get(
                                    "assignment_id", ""
                                )
                            ),
                            payload={
                                **({"worker_prepare_identity": expected[node_id]["worker_prepare_identity"]}
                                   if expected.get(node_id, {}).get("worker_prepare_identity") else {}),
                                "finalize": True,
                                "finalized_config_id": config_id,
                            },
                        )
                        cleanup_records.append(record)
                        finalize_payloads[node_id] = dict(record["payload"])
                    transaction["finalize_pending"] = set(worker_ids)
                    transaction["finalize_payloads"] = finalize_payloads
                    worker_cleanup_records = {
                        str(record.get("node_id", "") or ""): record
                        for record in cleanup_records
                        if record.get("node_id") in worker_ids
                    }
                    transaction["finalize_retry_state"] = {
                        node_id: {
                            "attempts": int(
                                worker_cleanup_records[node_id].get(
                                    "attempts", 1,
                                ) or 1
                            ),
                            "next_retry": float(
                                worker_cleanup_records[node_id].get(
                                    "next_retry", 0.0,
                                ) or 0.0
                            ),
                        }
                        for node_id in worker_ids
                        if node_id in worker_cleanup_records
                    }
                    self._persist_pipeline_lifecycle_locked()
                if callable(on_commit):
                    try:
                        on_commit()
                    except Exception:
                        # The runtime and epoch are already committed.  Cache /
                        # telemetry cleanup is post-commit maintenance and must
                        # never manufacture a false rollback over a committed
                        # reshard epoch.
                        logger.warning(
                            "流水线模型提交后回调失败: config=%s",
                            config_id,
                            exc_info=True,
                        )
        except Exception as exc:
            self._abort_pipeline_load_transaction(
                config_id, "pipeline_global_commit_failed", str(exc),
            )
            return

        if reshard_result is None:
            self._activate_pipeline_reshard_coordinator(active_plan)

        relay_clients = getattr(self, "_relay_segment_clients", None)
        if isinstance(relay_clients, dict):
            for task_id in list(relay_clients):
                self._close_relay_segment_client(task_id)

        for record in cleanup_records:
            self._dispatch_pipeline_model_cleanup(record)
        if retire_configs:
            self._publish_layer_configs(retire_configs)
        self._maybe_finish_pipeline_recovery()
        logger.info(
            "流水线模型事务已原子发布: config=%s candidate=%s workers=%s",
            config_id, candidate_id or "legacy", sorted(worker_ids),
        )


    def _invalidate_worker_layer_ready(
        self, node_id: str, config_id: str, reason: str,
    ) -> bool:
        """Revoke one worker's ready ACK and immediately resend its generation."""
        with self._layer_config_lock:
            expected = self._worker_assignments.expected(node_id)
            if not expected or expected.get("config_id") != config_id:
                return False
            assignment = dict(expected)
            # ★ 2026-10-07（DIST-NEXT-3）：撤销 ready ACK = 就绪证据作废 ⇒ 相位退回 `pushing`
            #   （保留 assignment_id / generation，迟到 ACK 仍无法冒充）。
            self._worker_assignments.invalidate(
                node_id, reason_code="layer_config_ack_revoked",
            )
            self._worker_assignments.record_ack(node_id, {
                "node_id": node_id,
                "assignment_id": assignment.get("assignment_id", ""),
                "config_id": config_id,
                "generation": assignment.get("generation", 0),
                "status": "error",
                "error": reason,
            })
            state = self._layer_config_retry_state.setdefault(
                node_id, {"attempts": 0, "next_retry": 0.0},
            )
            state["attempts"] = int(state.get("attempts", 0)) + 1
            state["next_retry"] = time.monotonic() + 5.0

        try:
            if self._tcp_server and self._tcp_server._running:
                self._send_pipeline_worker_config(node_id, assignment)
                logger.warning(
                    "worker 层配置已失效，立即重发: node=%s config=%s reason=%s",
                    node_id, config_id, reason,
                )
        except Exception:
            logger.warning(
                "worker 层配置立即重发失败，将由退避线程重试: node=%s",
                node_id, exc_info=True,
            )
        return True


    def _start_layer_config_retry_monitor(self) -> None:
        if (self._layer_config_retry_thread is not None
                and self._layer_config_retry_thread.is_alive()):
            return
        self._layer_config_retry_thread = threading.Thread(
            target=self._layer_config_retry_loop,
            name="layer-config-retry",
            daemon=True,
        )
        self._layer_config_retry_thread.start()


    def _layer_config_retry_loop(self) -> None:
        """重发未确认配置；节点错误或 ACK 丢失不能永久禁用流水线。"""
        while self._running and self._effective_role() == "master":
            self._retry_pending_layer_configs()
            time.sleep(1.0)


    def _retry_pending_layer_configs(self, now: float = None) -> int:
        """执行一次层配置重发扫描，返回成功发出的配置数量。"""
        now = time.monotonic() if now is None else now
        pending = []
        cleanup_pending = []
        timed_out = None
        with self._layer_config_lock:
            transaction = self._pipeline_load_transaction
            if (
                isinstance(transaction, dict)
                and str(transaction.get("phase", "") or "") in {
                    "preparing", "committing_local", "committing",
                    "committing_global",
                }
                and float(transaction.get("deadline", 0.0) or 0.0) > 0.0
                and now >= float(transaction.get("deadline", 0.0) or 0.0)
            ):
                timed_out = (
                    str(transaction.get("config_id", "") or ""),
                    "pipeline_model_transaction_timeout",
                    "pipeline model prepare/commit deadline exceeded",
                )
            if (
                isinstance(transaction, dict)
                and transaction.get("candidate_id")
                and str(transaction.get("phase", "") or "") in {
                    "preparing", "committing",
                }
            ):
                candidate_retry = transaction.setdefault("retry_state", {})
                confirmed_nodes = transaction.get(
                    "prepared_nodes" if transaction["phase"] == "preparing" else "ready_nodes", set(),
                )
                for node_id, expected in transaction.get("expected", {}).items():
                    if node_id in confirmed_nodes:
                        continue
                    state = candidate_retry.setdefault(
                        node_id, {"attempts": 0, "next_retry": now},
                    )
                    if now < state.get("next_retry", 0):
                        continue
                    state["attempts"] = int(state.get("attempts", 0)) + 1
                    delay = min(
                        60.0,
                        5.0 * (2 ** min(state["attempts"] - 1, 4)),
                    )
                    state["next_retry"] = now + delay
                    pending.append((node_id, dict(expected), state["attempts"]))
            for cleanup_id, record in list(
                self._pipeline_model_cleanup_ledger.items()
            ):
                if now < float(record.get("next_retry", 0.0) or 0.0):
                    continue
                attempts = int(record.get("attempts", 0) or 0) + 1
                delay = min(60.0, 5.0 * (2 ** min(attempts - 1, 4)))
                record["attempts"] = attempts
                record["next_retry"] = now + delay
                cleanup_pending.append(dict(record))
                if isinstance(transaction, dict) and record.get("action") == "finalize":
                    node_id = str(record.get("node_id", "") or "")
                    retry = transaction.setdefault(
                        "finalize_retry_state", {}
                    ).setdefault(node_id, {})
                    retry.update({
                        "attempts": attempts,
                        "next_retry": record["next_retry"],
                    })
            for node_id, expected in self._worker_assignments.expected_configs().items():
                if self._effective_layer_config_pushed(node_id):
                    continue
                state = self._layer_config_retry_state.setdefault(
                    node_id, {"attempts": 0, "next_retry": now}
                )
                if now < state.get("next_retry", 0):
                    continue
                state["attempts"] = int(state.get("attempts", 0)) + 1
                delay = min(60.0, 5.0 * (2 ** min(state["attempts"] - 1, 4)))
                state["next_retry"] = now + delay
                pending.append((node_id, dict(expected), state["attempts"]))

        if timed_out is not None:
            self._abort_pipeline_load_transaction(*timed_out)
            return 0

        cleanup_sent = 0
        for record in cleanup_pending:
            if self._dispatch_pipeline_model_cleanup(record):
                cleanup_sent += 1

        connected = set()
        if self._tcp_server and self._tcp_server._running:
            try:
                connected = set(self._tcp_server.get_client_ids())
            except Exception:
                logger.debug("读取层配置重试节点失败", exc_info=True)
        sent = cleanup_sent
        for node_id, assignment, attempt in pending:
            if node_id not in connected:
                continue
            try:
                self._send_pipeline_worker_config(node_id, assignment)
                sent += 1
                logger.info(
                    "重发分层配置: node=%s config=%s attempt=%d",
                    node_id, assignment.get("config_id", ""), attempt,
                )
            except Exception:
                logger.warning(
                    "重发分层配置失败: node=%s attempt=%d",
                    node_id, attempt, exc_info=True,
                )
        return sent


    def _get_master_model_sha256(self) -> str:
        """
        获取主节点当前加载模型的 SHA256。

        口径由**主节点引擎自己的描述器**给出：PyTorch 侧是目录内 artifact 的联合哈希
        （`model_sync.compute_model_sha256`），llama.cpp 侧是整份 GGUF 的文件摘要
        （`LlamaCppEngine._pipeline_file_sha256`）。两者都只用于「各节点是否握着同一份
        权重」的自证，混合部署时以主节点为准。

        旧注释写的是「llama.cpp/GGUF 不支持层拆分，不得作为流水线模型基准」—— 那是
        `BackendId.LLAMA_CPP` 声明 `FORWARD_LAYERS` 之前的判据，已不成立。
        """
        from model_sync import compute_model_sha256

        mgr = self._host
        if not mgr or not runtime_supports(mgr, Capability.FORWARD_LAYERS):
            return ""

        get_descriptor = getattr(mgr, "get_pipeline_descriptor", None)
        if callable(get_descriptor):
            try:
                cached = str((get_descriptor() or {}).get("model_sha256", ""))
                if cached:
                    return cached
            except Exception:
                logger.debug("读取流水线描述器摘要失败", exc_info=True)

        model_path = (
            getattr(mgr, '_full_model_path', '')
            or getattr(mgr, '_model_path', '')
            or ''
        )
        if not model_path or not os.path.isdir(model_path):
            # 单文件（GGUF）走不到这里：描述器已给出 `model_sha256` 并在上面返回。
            # `compute_model_sha256` 的语义是「目录内 artifact 联合哈希」，对单文件
            # 无意义，故不放宽 —— 缺描述器的 GGUF 引擎视为不可校验。
            return ""

        try:
            return compute_model_sha256(model_path)
        except Exception:
            logger.warning("计算主节点 PyTorch 模型摘要失败", exc_info=True)
            return ""


    def _clear_pipeline_runtime_state(self, task_id: str) -> None:
        """清理主节点侧单个流水线任务的等待结果与链路 ACK 状态。"""
        if not task_id:
            return
        self._close_relay_segment_client(task_id)
        prefix = f"{task_id}:"
        with self._pipeline_lock:
            self._pipeline_active_tasks.discard(task_id)
            for key in list(self._pipeline_results):
                if key.startswith(prefix):
                    self._pipeline_results.pop(key, None)
            for key in list(self._pipeline_events):
                if key.startswith(prefix):
                    self._pipeline_events.pop(key, None)
            self._chain_ack_state.pop(task_id, None)
            self._pipeline_task_contracts.pop(task_id, None)


    def has_pipeline_worker_reservation(self) -> bool:
        """Return whether this PC is reserved for a master's layer pipeline."""
        with self._layer_config_lock:
            return bool(self._pipeline_worker_reserved)


    def release_pipeline_worker_for_local_model(self) -> bool:
        """Opt this client out before an explicit local model operation."""
        with self._layer_config_lock:
            self._pipeline_worker_reserved = False
            self._pipeline_worker_opted_out = True
            self._active_layer_config = None
            self._last_layer_config_ack_payload = None
            self._local_pipeline_steps.clear()
        if self._effective_role() != "client":
            return True
        client = getattr(self, "_tcp_client", None)
        if not client or not getattr(client, "_running", False):
            logger.warning("本地模型切换时主节点未连接，已仅清理本地分层预留")
            return False
        try:
            from transport_port import MessageType

            client.send_data(
                {
                    "node_id": self.get_effective_node_id(),
                    "reason": "explicit_local_model_change",
                },
                MessageType.LAYER_WORKER_OPT_OUT,
            )
            return True
        except Exception:
            logger.warning("通知主节点退出分层 worker 失败", exc_info=True)
            return False


    def _begin_local_pipeline_task(self, task_id: str) -> None:
        """Track local work so layer reconfiguration cannot replace an active model."""
        if not task_id:
            return
        with self._layer_config_lock:
            self._active_pipeline_task_ids.add(task_id)


    def _finish_local_pipeline_task(self, task_id: str) -> None:
        """Release local work state and apply the newest deferred layer config."""
        pending = None
        with self._layer_config_lock:
            self._active_pipeline_task_ids.discard(task_id)
            self._local_pipeline_steps.pop(task_id, None)
            if not self._active_pipeline_task_ids and self._pending_layer_config is not None:
                pending_config_id = str(
                    self._pending_layer_config[1].get("config_id", "")
                )
                if not pending_config_id or pending_config_id not in self._layer_config_inflight:
                    pending = self._pending_layer_config
                    self._pending_layer_config = None
        self._close_relay_segment_client(task_id)
        if pending is not None:
            client_id, data = pending
            logger.info("当前流水线任务已结束，开始应用延后的分层配置")
            self._schedule_layer_config(client_id, data)


    def _relay_segment_client_for_task(
        self, task_id: str, spec: dict[str, object], *, n_embd: int,
    ) -> RelaySegmentClient:
        """Reuse one relay TCP session for all steps in a pipeline task."""
        cache = getattr(self, "_relay_segment_clients", None)
        if cache is None:
            cache = {}
            self._relay_segment_clients = cache
        key = (
            str(spec["host"]), int(spec["port"]), int(n_embd),
            str(spec.get("role", "middle")),
        )
        current = cache.get(task_id)
        if current is not None and current[0] == key:
            return current[1]
        if current is not None:
            try:
                current[1].close()
            except Exception:
                logger.debug("close stale relay session failed", exc_info=True)
        client = RelaySegmentClient(
            str(spec["host"]), int(spec["port"]), n_embd=int(n_embd),
            role=str(spec.get("role", "middle")), timeout=float(spec["timeout"]),
        )
        cache[task_id] = (key, client)
        logger.info("relay 段会话已建立并缓存: task=%s cache_id=%s size=%d",
                    task_id, id(cache), len(cache))
        return client


    def _close_relay_segment_client(
        self, task_id: str, *, grace_timeout: float | None = None,
    ) -> None:
        cache = getattr(self, "_relay_segment_clients", None)
        if not cache:
            logger.info("关闭 relay 段会话: task=%s 无缓存（cache=%r cache_id=%s self_id=%s）",
                        task_id, cache, id(cache) if cache is not None else None,
                        id(getattr(self, "_relay_segment_clients", None)))
            return
        current = cache.pop(task_id, None)
        logger.info("关闭 relay 段会话: task=%s found=%s cache_id=%s keys=%r",
                    task_id, current is not None, id(cache), list(cache))
        if current is not None:
            try:
                try:
                    current[1].close(grace_timeout=grace_timeout)
                except TypeError:
                    current[1].close()
            except Exception:
                logger.warning("close relay session failed: task=%s", task_id, exc_info=True)


    def _mark_local_pipeline_cancelled(self, task_id: str) -> None:
        if not task_id:
            return
        with self._layer_config_lock:
            if task_id not in self._local_pipeline_cancelled:
                self._local_pipeline_cancelled.add(task_id)
                self._local_pipeline_cancelled_order.append(task_id)
            while len(self._local_pipeline_cancelled_order) > 4096:
                expired = self._local_pipeline_cancelled_order.popleft()
                self._local_pipeline_cancelled.discard(expired)


    def _fail_pending_pipeline_results_for_node(self, node_id: str,
                                                reason: str) -> None:
        """节点断连/不可用时，立即失败所有正在等待该节点的流水线步骤。"""
        if not node_id:
            return
        failed = []
        with self._pipeline_lock:
            for key, event in list(self._pipeline_events.items()):
                try:
                    task_id, waiting_node_id = key.split(":", 1)
                except ValueError:
                    continue
                if waiting_node_id != node_id:
                    continue
                self._pipeline_results[key] = {
                    "task_id": task_id,
                    "node_id": node_id,
                    "error": reason,
                    "step": -1,
                }
                event.set()
                failed.append(task_id)
        if failed:
            logger.warning(
                "节点 %s 不可用，已唤醒 %d 个流水线等待任务: %s",
                node_id, len(failed), ", ".join(failed),
            )


    def _handle_chain_forward_ack(self, client_id: str, msg: dict) -> None:
        """主节点：记录链式转发每跳 ACK/错误，并在错误时立即唤醒流水线。"""
        data = msg.get("data", {})
        task_id = str(data.get("task_id", "") or "")
        try:
            step = int(data.get("step", -1))
        except (TypeError, ValueError):
            logger.warning("丢弃 step 无效的链式 ACK: task=%s", task_id or "-")
            return
        status = data.get("status", "received")
        error = data.get("error", "")
        config_id = str(data.get("config_id", ""))
        assignment_id = str(data.get("assignment_id", "") or "")
        reporter_node_id = str(data.get("node_id", client_id))
        target_node_id = data.get("target_node_id", "")
        node_id = target_node_id if status in ("sent", "error") and target_node_id else reporter_node_id

        if not task_id or not node_id:
            return
        if reporter_node_id != client_id:
            logger.warning(
                "丢弃来源不一致的链式 ACK: connection=%s payload=%s",
                client_id, reporter_node_id,
            )
            return
        if status not in {"sent", "received", "error"}:
            logger.warning("丢弃未知链式 ACK 状态: %s", status)
            return

        now = time.time()
        with self._pipeline_lock:
            if task_id not in self._pipeline_active_tasks:
                return
            contract = self._pipeline_task_contracts.get(task_id, {})
            if contract.get("execution_protocol") != "legacy_layer_v1":
                logger.warning(
                    "discarding chain ACK for non-legacy task: task=%s protocol=%s",
                    task_id, contract.get("execution_protocol", "missing"),
                )
                return
            worker_ids = list(contract.get("worker_ids", []))
            assignment_ids = dict(contract.get("assignment_ids", {}))
            expected_nodes = set(worker_ids)
            identities_complete = (
                set(assignment_ids) == expected_nodes
                and all(str(assignment_ids.get(worker_id, "") or "")
                        for worker_id in worker_ids)
            )
            if (step != contract.get("current_step")
                    or config_id != contract.get("config_id")
                    or not identities_complete
                    or not assignment_id
                    or assignment_id != assignment_ids.get(reporter_node_id, "")
                    or reporter_node_id not in expected_nodes
                    or node_id not in expected_nodes):
                logger.warning(
                    "丢弃不符合执行契约的链式 ACK: task=%s step=%s "
                    "node=%s config=%s",
                    task_id, step, node_id, config_id,
                )
                return
            if status == "sent" or (status == "error" and target_node_id):
                reporter_index = worker_ids.index(reporter_node_id)
                expected_target = (
                    worker_ids[reporter_index + 1]
                    if reporter_index + 1 < len(worker_ids) else ""
                )
                if not target_node_id or target_node_id != expected_target:
                    logger.warning(
                        "丢弃非相邻链路 %s ACK: task=%s reporter=%s "
                        "target=%s expected=%s",
                        status, task_id, reporter_node_id,
                        target_node_id, expected_target,
                    )
                    return
            elif status == "received":
                receiver_index = worker_ids.index(reporter_node_id)
                expected_source = (
                    worker_ids[receiver_index - 1] if receiver_index > 0 else ""
                )
                if str(data.get("from_node_id", "")) != expected_source:
                    logger.warning(
                        "丢弃非相邻链路 received ACK: task=%s receiver=%s "
                        "source=%s expected=%s",
                        task_id, reporter_node_id,
                        data.get("from_node_id", ""), expected_source,
                    )
                    return
            task_state = self._chain_ack_state.setdefault(task_id, {})
            step_state = task_state.setdefault(step, {})
            existing = step_state.get(node_id, {})
            new_state = {
                "status": status,
                "error": error,
                "from_node_id": data.get("from_node_id", reporter_node_id),
                "target_node_id": target_node_id or node_id,
                "reporter_node_id": reporter_node_id,
                "assignment_id": assignment_id,
                "updated_at": now,
            }
            if status == "sent":
                new_state["sent_at"] = now
                new_state["source_assignment_id"] = assignment_id
                if existing.get("status") == "received":
                    # 下游 ACK 可能比上游 sent 回报更早到达；不要把
                    # received 状态倒退为 sent，也不要用来源段身份覆盖
                    # 已确认接收的目标段身份。
                    new_state["status"] = "received"
                    new_state["acked_at"] = existing.get("acked_at", existing.get("updated_at", now))
                    new_state["error"] = existing.get("error", "")
                    new_state["reporter_node_id"] = existing.get(
                        "reporter_node_id", node_id,
                    )
                    new_state["assignment_id"] = existing.get(
                        "assignment_id", "",
                    )
                    new_state["target_assignment_id"] = existing.get(
                        "target_assignment_id",
                        existing.get("assignment_id", ""),
                    )
            elif status == "received":
                new_state["acked_at"] = now
                new_state["target_assignment_id"] = assignment_id
                if existing.get("sent_at"):
                    new_state["sent_at"] = existing["sent_at"]
                if existing.get("source_assignment_id"):
                    new_state["source_assignment_id"] = existing[
                        "source_assignment_id"
                    ]
            step_state[node_id] = new_state

        if status == "error" or error:
            message = error or f"链式转发节点 {node_id} 返回错误 ACK"
            logger.error(
                "链式转发 ACK 错误: task=%s step=%s node=%s error=%s",
                task_id, step, node_id, message,
            )
            self._set_pipeline_result_error(task_id, node_id, message, step)


    def _get_chain_ack_failure(self, task_id: str, step: int,
                               expected_node_ids: list,
                               ack_timeout: float) -> Optional[dict]:
        """检测已发送但迟迟未被下游确认接收的链式转发。"""
        if not task_id or not expected_node_ids:
            return None
        now = time.time()
        with self._pipeline_lock:
            step_state = self._chain_ack_state.get(task_id, {}).get(step, {})
            for node_id in expected_node_ids:
                state = step_state.get(node_id)
                if not state:
                    continue
                if state.get("status") == "error" or state.get("error"):
                    return {
                        "task_id": task_id,
                        "node_id": node_id,
                        "error": state.get("error") or f"链式转发到 {node_id} 失败",
                        "step": step,
                    }
                if state.get("status") == "sent":
                    sent_at = state.get("sent_at", state.get("updated_at", now))
                    if now - sent_at >= ack_timeout:
                        return {
                            "task_id": task_id,
                            "node_id": node_id,
                            "error": (
                                f"链式转发到 {node_id} 未收到接收 ACK "
                                f"({ack_timeout:.1f}s)"
                            ),
                            "step": step,
                        }
        return None


    def handle_infer_forward(self, client_id: str, msg: dict) -> None:
        """
        处理从节点转发的推理请求（统一流水线调度）。

        主节点收到 INFER_FORWARD 后:
          1. 创建推理任务
          2. 通过 run_pipeline_safe() 统一调度:
             - 流水线节点就绪 → 分布式流水线推理
             - 流水线节点未就绪 → 自动回退到主节点全模型推理
          3. 将结果通过 INFER_RESULT 回传给请求方

        路径 A 和路径 B 已统一 — 无论请求来自 HTTP /api/chat 还是
        TCP INFER_FORWARD，都走同一套 run_pipeline_safe() 调度。
        """
        data = msg.get("data", {})
        prompt = data.get("prompt", "")
        max_new_tokens = data.get("max_new_tokens", 512)
        temperature = data.get("temperature", 0.7)
        top_p = data.get("top_p", 0.9)
        routing_preference = str(data.get("routing_preference", "auto") or "auto")
        show_thinking = data.get("show_thinking", False)
        # ★ 2026-09-19：主节点收到转发请求后同样透传深度思考**开关**。
        enable_thinking = data.get("enable_thinking")
        session_id = data.get("session_id")
        messages = data.get("messages")
        request_id = data.get("request_id")   # L5: 链路追踪
        request_deadline_ms = int(data.get("request_deadline_ms", 0) or 0)
        request_deadline = (
            RequestDeadline.from_epoch_ms(request_deadline_ms)
            if request_deadline_ms else None
        )
        forward_request_id = str(data.get("forward_request_id", ""))
        speculation = {}
        # Omitted/null stays omitted: only ExecutionApplication owns env defaults.
        if data.get("speculative_gamma") is not None:
            from qlh_core.contracts.speculation import resolve_speculative_gamma
            from qlh_core.contracts.execution import ExecutionError
            try:
                gamma = resolve_speculative_gamma(data["speculative_gamma"])
                if gamma > 1 and routing_preference == "local_only":
                    raise ExecutionError("speculative_route_unsupported")
            except ExecutionError as exc:
                self._send_infer_result(
                    client_id, "", "", {"reason_code": exc.code},
                    forward_request_id=forward_request_id,
                    status="error", error=str(exc),
                )
                return
            speculation["speculative_gamma"] = gamma
        cancel_key = forward_request_id or f"legacy_{uuid.uuid4().hex}"

        import threading as _thr

        if not self._forward_infer_slots.acquire(blocking=False):
            self._send_infer_result(
                client_id, "", "", {},
                forward_request_id=forward_request_id,
                status="error",
                error="主节点转发请求已达并发上限，请稍后重试",
            )
            return

        cancel_event = threading.Event()
        with self._forward_cancel_lock:
            self._forward_cancel_events[(client_id, cancel_key)] = cancel_event

        def _run_inference():
            task_id = ""
            try:
                task_id = self.start_infer_task(prompt, request_id=request_id)
                logger.info(
                    "event=infer_forward_recv task_id=%s request_id=%s "
                    "client_id=%s prompt_len=%d max_tokens=%d",
                    task_id, request_id or "-", client_id, len(prompt), max_new_tokens,
                )

                # ★ 统一流水线调度（替代原来的 mgr.chat() 全模型直调）
                pipeline_result = self.run_pipeline_safe(
                    prompt=prompt,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    **speculation,
                    routing_preference=routing_preference,
                    session_id=session_id,
                    messages=messages,
                    show_thinking=show_thinking,
                    enable_thinking=enable_thinking,
                    _require_distributed=(routing_preference == "distributed_required"),
                    _force_distributed_assignment=(routing_preference != "local_only"),
                    _cancel_event=cancel_event,
                    _request_deadline=request_deadline,
                )

                content = pipeline_result.get("response", "")
                thinking_content = pipeline_result.get("thinking")
                error = pipeline_result.get("error")
                metrics = pipeline_result.get("metrics", {})
                reason_code = str(pipeline_result.get("reason_code") or "")
                if reason_code:
                    metrics = {**metrics, "reason_code": reason_code}

                if error:
                    self.fail_infer_task(task_id, error)
                    logger.warning(
                        f"⚠️ 流水线推理失败 → {client_id}: task={task_id}, "
                        f"error={error}"
                    )
                    self._send_infer_result(
                        client_id, task_id, content,
                        {**metrics, "error": error},
                        thinking_content=thinking_content,
                        forward_request_id=forward_request_id,
                        status="error",
                        error=error,
                    )
                    return

                # 保存到对话历史（主节点侧）
                if content:
                    try:
                        import local_store
                        local_store.save_local_conversation_turn(
                            session_id=session_id or "default",
                            user_message=prompt,
                            assistant_message=content,
                            metrics=metrics,
                            operation_id=f"pipeline:{task_id}",
                        )
                    except Exception:
                        pass

                if not metrics.get("distributed_used"):
                    try:
                        self.record_task_complete(success=True)
                    except Exception:
                        pass

                self.complete_infer_task(task_id, content, metrics)
                self._send_infer_result(
                    client_id, task_id, content, metrics,
                    thinking_content=thinking_content,
                    forward_request_id=forward_request_id,
                )
                logger.info(
                    f"✅ 推理完成 → {client_id}: task={task_id}, "
                    f"len={len(content)}, engine={metrics.get('engine', '?')}"
                )

            except Exception as e:
                if task_id:
                    self.fail_infer_task(task_id, str(e))
                logger.error(f"转发推理执行失败: {e}", exc_info=True)
                self._send_infer_result(
                    client_id, task_id, "",
                    {"error": str(e)},
                    forward_request_id=forward_request_id,
                    status="error",
                    error=str(e),
                )
            finally:
                with self._forward_cancel_lock:
                    self._forward_cancel_events.pop((client_id, cancel_key), None)
                self._forward_infer_slots.release()

        _thr.Thread(
            target=_run_inference,
            name=f"infer-{client_id}-{cancel_key[-8:]}",
            daemon=True,
        ).start()


    def _send_infer_result(self, client_id: str, task_id: str,
                           content: str, metrics: dict = None,
                           thinking_content: str = None,
                           followups: list = None,
                           forward_request_id: str = "",
                           status: str = "ok",
                           error: str = "") -> None:
        """向从节点回传推理结果"""
        if self._tcp_server and self._tcp_server._running:
            try:
                from transport_port import MessageType
                result_data = {
                    "task_id": task_id,
                    "forward_request_id": forward_request_id,
                    "status": status,
                    "content": content,
                    "metrics": metrics or {},
                }
                if error:
                    result_data["error"] = error
                if thinking_content:
                    result_data["thinking_content"] = thinking_content
                if followups:
                    result_data["followups"] = followups
                self._tcp_server.send_to_client(
                    client_id,
                    result_data,
                    msg_type=MessageType.INFER_RESULT,
                )
            except Exception as e:
                logger.error(f"回传推理结果失败 ({client_id}): {e}")


    def _schedule_layer_config(self, client_id: str, data: dict) -> None:
        config_id = str(data.get("config_id", "")) if isinstance(data, dict) else ""
        assignment_id = str(
            data.get("assignment_id", "") or ""
        ) if isinstance(data, dict) else ""
        incoming_phase = str(
            data.get("phase", "commit") or "commit"
        ) if isinstance(data, dict) else "commit"
        authoritative_sync = bool(
            isinstance(data, dict) and data.get("authoritative_sync")
        )
        if not self.get_distributed_inference_enabled() and not data.get("release"):
            self.release_pipeline_worker_for_local_model()
            return
        resend_opt_out = False
        authoritative_opt_in = False
        with self._layer_config_lock:
            # The fast check above can precede an explicit local-mode switch.
            # Recheck before authoritative sync can clear opt-out or enqueue work.
            if not self.get_distributed_inference_enabled() and not data.get("release"):
                resend_opt_out = True
            elif (
                authoritative_sync
                and self._pipeline_worker_opted_out
                and isinstance(data, dict)
                and not data.get("release")
            ):
                self._pipeline_worker_opted_out = False
                authoritative_opt_in = True
            if (self._pipeline_worker_opted_out
                    and isinstance(data, dict)
                    and not data.get("release")):
                resend_opt_out = True
            if resend_opt_out:
                payload = None
                receive_sequence = 0
                generation = 0
            else:
                cached = self._last_layer_config_ack_payload
                if (config_id and cached
                        and cached.get("config_id") == config_id
                        and str(cached.get("assignment_id", "") or "")
                        == assignment_id
                        and str(cached.get("phase", "commit") or "commit")
                        == incoming_phase
                        and config_id not in self._layer_config_inflight):
                    payload = dict(cached)
                else:
                    payload = None
            if resend_opt_out:
                pass
            elif config_id and config_id in self._layer_config_inflight:
                return
            if resend_opt_out:
                pass
            elif payload is not None:
                receive_sequence = 0
                generation = 0
            else:
                self._layer_config_receive_sequence += 1
                receive_sequence = self._layer_config_receive_sequence
                try:
                    generation = int(data.get("generation", 0) or 0)
                except (TypeError, ValueError):
                    generation = 0
            if resend_opt_out:
                pass
            elif payload is not None:
                pass
            elif (self._latest_layer_config_generation
                  and generation < self._latest_layer_config_generation):
                logger.info(
                    "忽略过期分层配置: config=%s generation=%s latest=%s",
                    config_id,
                    generation,
                    self._latest_layer_config_generation,
                )
                return
            elif (
                self._latest_layer_assignment_id
                and generation == self._latest_layer_config_generation
                and assignment_id != self._latest_layer_assignment_id
            ):
                logger.info(
                    "忽略同 generation 的异 assignment 配置: config=%s "
                    "assignment=%s latest=%s",
                    config_id,
                    assignment_id or "legacy",
                    self._latest_layer_assignment_id,
                )
                return
            else:
                self._latest_layer_config_receive_sequence = receive_sequence
                self._latest_layer_config_generation = max(
                    self._latest_layer_config_generation,
                    generation,
                )
                if assignment_id:
                    self._latest_layer_assignment_id = assignment_id
                if isinstance(data, dict) and not data.get("release"):
                    self._pipeline_worker_reserved = True
                if config_id:
                    self._layer_config_inflight.add(config_id)

        if authoritative_opt_in:
            logger.info(
                "收到主节点权威模型配置，自动重新加入分层 worker: config=%s",
                config_id,
            )
        if resend_opt_out:
            logger.info(
                "本设备已选择本地模型，拒绝分层配置并重发退出请求: config=%s",
                config_id,
            )
            self.release_pipeline_worker_for_local_model()
            return
        if payload is not None:
            self._send_layer_config_ack(payload)
            return

        def _load() -> None:
            try:
                self._handle_layer_config(
                    client_id,
                    data,
                    receive_sequence=receive_sequence,
                    generation=generation,
                )
            finally:
                if config_id:
                    with self._layer_config_lock:
                        self._layer_config_inflight.discard(config_id)
                        pending_matches = bool(
                            self._pending_layer_config
                            and str(self._pending_layer_config[1].get(
                                "config_id", ""
                            )) == config_id
                        )
                    if pending_matches:
                        with self._layer_config_lock:
                            can_apply = not self._active_pipeline_task_ids
                            pending = (
                                self._pending_layer_config
                                if can_apply else None
                            )
                            if pending is not None:
                                self._pending_layer_config = None
                        if pending is not None:
                            pending_client_id, pending_data = pending
                            self._schedule_layer_config(
                                pending_client_id, pending_data
                            )

        threading.Thread(
            target=_load,
            name=f"layer-config-{config_id[-8:] or 'legacy'}",
            daemon=True,
        ).start()


    def _handle_layer_config(
        self,
        client_id: str,
        data: dict,
        *,
        receive_sequence: int = None,
        generation: int = None,
    ) -> None:
        # Match API model changes: full-chat must precede layer execution.
        # The full-chat lock also excludes worker Stages; do not reacquire their
        # non-reentrant inference lock here. Candidate commit/abort can reenter
        # full-chat while these locks are held.
        # Injected endpoint/control-only hosts may have no chat lifecycle;
        # retain their existing layer-only serialization.
        full_chat_lock = getattr(
            self._host, "full_chat_execution_lock", self._layer_execution_lock,
        )
        with full_chat_lock, self._layer_execution_lock:
            # Admission may have happened before an explicit opt-out. Check
            # again under the model lifecycle lock before touching any runtime.
            # Release/abort cleanup must remain available in local mode.
            if not self.get_distributed_inference_enabled() and not data.get("release"):
                self.release_pipeline_worker_for_local_model()
                return
            if receive_sequence is not None:
                with self._layer_config_lock:
                    if (receive_sequence
                            != self._latest_layer_config_receive_sequence):
                        logger.info(
                            "跳过已被新消息取代的分层配置: config=%s sequence=%s latest=%s",
                            data.get("config_id", ""),
                            receive_sequence,
                            self._latest_layer_config_receive_sequence,
                        )
                        return
                    if (self._latest_layer_config_generation
                            and generation
                            < self._latest_layer_config_generation):
                        return
            self._handle_layer_config_locked(
                client_id,
                data,
                receive_sequence=receive_sequence,
                generation=generation,
            )


    def _handle_layer_config_locked(
        self,
        client_id: str,
        data: dict,
        *,
        receive_sequence: int = None,
        generation: int = None,
    ) -> None:
        """
        从节点：收到主节点推送的分层配置 → 加载指定层范围。

        新版主节点只发送本节点的 assignment；同时兼容旧版
        ``{node_id: assignment}`` 外层映射。加载结束后必须发送 ACK，
        主节点收到当前 config_id 的成功 ACK 才会将本节点视为 ready。
        """
        with self._layer_config_lock:
            if self._active_pipeline_task_ids:
                self._pending_layer_config = (client_id, dict(data))
                logger.warning(
                    "本节点仍有流水线任务执行中，分层配置已延后: active=%s",
                    sorted(self._active_pipeline_task_ids),
                )
                return

        cleanup_action = data.get("cleanup_action") if isinstance(data, dict) else None
        if cleanup_action and (
            (cleanup_action == "abort" and (data.get("abort") is not True or data.get("release") is not True or data.get("finalize")))
            or (cleanup_action == "finalize" and (data.get("finalize") is not True or data.get("abort") or data.get("release")))
            or cleanup_action not in {"abort", "finalize"}
            or not data.get("candidate_config_id")
        ):
            logger.warning("event=pipeline_cleanup_shape_rejected action=%s", cleanup_action)
            return
        node_id = self.get_effective_node_id()
        ack_generation = (
            data.get("generation", 0) if isinstance(data, dict) else 0
        )
        # ★ 2026-10-03：本节点若是 **v3 层段 worker**（env 已指定层段工件），就不该接受
        #   legacy `LAYER_CONFIG`。它手上有工件、层段由 v3 stage offer 驱动；legacy 语义
        #   却要求它 `ensure_pipeline_assignment_available` 去同步模型 —— 跨机时被
        #   `MODEL_API_SOURCE_UNTRUSTED` 拒（非 loopback、不在信任 CIDR），随后
        #   「主节点已释放本设备的分层 worker 预留」⇒ 它掉出层段 worker 名单，永远进不了
        #   `admitted`（实测：Surface 注册后 6 ms 就被释放，此后每轮重连重复一次）。
        #   判据必须落在 worker 自己身上：主节点在**节点注册那一刻**就推送 legacy 配置，
        #   早于 hello 往返 ⇒ master 侧按 capabilities 排除在时序上不可靠（已踩到）。
        legacy_candidate = data
        if (
            isinstance(data, dict)
            and node_id in data
            and isinstance(data.get(node_id), dict)
        ):
            legacy_candidate = data[node_id]
        if isinstance(legacy_candidate, dict):
            candidate_generation = legacy_candidate.get("generation")
            if candidate_generation is not None:
                ack_generation = candidate_generation
        ack_assignment_id = (
            str(legacy_candidate.get("assignment_id", "") or "")
            if isinstance(legacy_candidate, dict) else ""
        )
        ack_phase = (
            str(legacy_candidate.get("phase", "commit") or "commit")
            if isinstance(legacy_candidate, dict) else "commit"
        )
        is_relay_assignment = (
            isinstance(legacy_candidate, dict)
            and str(legacy_candidate.get("engine", "") or "").lower()
            == "relay_middle"
        )
        if (
            isinstance(legacy_candidate, dict)
            and not legacy_candidate.get("release")
            and not is_relay_assignment
            and os.environ.get("QLH_LAYER_GGUF", "").strip()
        ):
            logger.info(
                "本节点是 v3 层段 worker，拒绝 legacy 分层配置: config=%s",
                legacy_candidate.get("config_id", ""),
            )
            self._send_layer_config_ack({
                "node_id": node_id,
                "assignment_id": ack_assignment_id,
                "config_id": str(legacy_candidate.get("config_id", "")),
                "generation": ack_generation,
                "phase": ack_phase,
                "status": "error",
                "error": "layer_stage_worker_rejects_legacy_config",
                "timestamp": time.time(),
            })
            return
        if isinstance(data, dict) and data.get("finalize"):
            target_node_id = str(data.get("node_id", node_id))
            if target_node_id != node_id:
                return
            finalized_config_id = str(
                data.get("candidate_config_id", "")
                or data.get("finalized_config_id", "")
                or data.get("config_id", "")
                or ""
            )
            with self._layer_config_lock:
                rollback = self._pipeline_assignment_rollbacks.get(
                    finalized_config_id, {}
                )
            candidate_status = getattr(
                self._host, "pipeline_candidate_status", None,
            )
            status = (
                candidate_status(finalized_config_id)
                if callable(candidate_status) else {}
            )
            if rollback.get("candidate_runtime") or (
                isinstance(status, dict)
                and status.get("matches")
                and status.get("phase") in {"committed", "finalized"}
            ):
                finalize_candidate = getattr(
                    self._host, "finalize_pipeline_candidate", None,
                )
                if callable(finalize_candidate) and finalized_config_id:
                    finalize_candidate(finalized_config_id)
            with self._layer_config_lock:
                self._prepared_layer_configs.pop(finalized_config_id, None)
                self._pipeline_assignment_rollbacks.pop(
                    finalized_config_id, None,
                )
            self._send_layer_config_ack({
                "node_id": node_id,
                "assignment_id": str(data.get("assignment_id", "") or ""),
                "config_id": str(data.get("config_id", "") or ""),
                "generation": ack_generation,
                "status": "finalized",
                "cleanup_action": str(
                    data.get("cleanup_action", "finalize") or "finalize"
                ),
                "candidate_config_id": finalized_config_id,
                "finalized_config_id": finalized_config_id,
                "timestamp": time.time(),
            })
            return
        if isinstance(data, dict) and data.get("release"):
            target_node_id = str(data.get("node_id", node_id))
            if target_node_id != node_id:
                logger.warning(
                    "忽略目标不匹配的分层释放: target=%s local=%s",
                    target_node_id, node_id,
                )
                return
            aborted_config_id = str(
                data.get("candidate_config_id", "")
                or data.get("aborted_config_id", "")
                or ""
            )
            rollback = None
            prepared = None
            with self._layer_config_lock:
                if aborted_config_id:
                    rollback = self._pipeline_assignment_rollbacks.get(
                        aborted_config_id,
                    )
                    prepared = self._prepared_layer_configs.get(
                        aborted_config_id,
                    )
                if not data.get("abort"):
                    self._pipeline_worker_reserved = False
                    self._active_layer_config = None
                    self._last_layer_config_ack_payload = None
                    self._local_pipeline_steps.clear()
            if data.get("abort"):
                abort_candidate = getattr(
                    self._host, "abort_pipeline_candidate", None,
                )
                candidate_status = getattr(
                    self._host, "pipeline_candidate_status", None,
                )
                status = (
                    candidate_status(aborted_config_id)
                    if callable(candidate_status) and aborted_config_id else {}
                )
                candidate_runtime = bool(
                    isinstance(rollback, dict)
                    and rollback.get("candidate_runtime")
                ) or bool(
                    isinstance(prepared, dict)
                    and prepared.get("plan_id")
                ) or bool(
                    isinstance(status, dict)
                    and status.get("matches")
                    and status.get("phase") not in {"idle", "aborted"}
                )
                if (
                    callable(abort_candidate)
                    and aborted_config_id
                    and candidate_runtime
                ):
                    abort_candidate(aborted_config_id)
                if isinstance(rollback, dict):
                    with self._layer_config_lock:
                        previous = rollback.get("active_layer_config")
                        self._active_layer_config = (
                            dict(previous) if isinstance(previous, dict) else None
                        )
                        self._pipeline_worker_reserved = bool(
                            rollback.get("pipeline_worker_reserved", False)
                        )
                        self._last_layer_config_ack_payload = rollback.get(
                            "last_layer_config_ack_payload"
                        )
                        self._local_pipeline_steps.clear()
                with self._layer_config_lock:
                    self._pipeline_assignment_rollbacks.pop(
                        aborted_config_id, None,
                    )
                    self._prepared_layer_configs.pop(
                        aborted_config_id, None,
                    )
                self._host.model_loaded = bool(
                    getattr(self._host, "is_loaded", False)
                    and not getattr(
                        self._host, "is_pipeline_prepared", False,
                    )
                )
                try:
                    from model_sync import remove_pipeline_assignment_cache

                    model_id = str(data.get("model_id", "") or "")
                    if model_id and aborted_config_id:
                        remove_pipeline_assignment_cache(
                            model_id, aborted_config_id, node_id,
                        )
                except Exception:
                    logger.warning("清理已中止的 assignment 缓存失败", exc_info=True)
            release_ack = {
                "node_id": node_id,
                "assignment_id": str(data.get("assignment_id", "") or ""),
                "config_id": str(data.get("config_id", "")),
                "generation": ack_generation,
                "status": "released",
                "release": True,
                "timestamp": time.time(),
            }
            if data.get("abort") or data.get("cleanup_action"):
                release_ack.update({
                    "cleanup_action": str(
                        data.get("cleanup_action", "abort") or "abort"
                    ),
                    "candidate_config_id": aborted_config_id,
                    "aborted_config_id": aborted_config_id,
                })
            self._send_layer_config_ack(release_ack)
            logger.info("主节点已释放本设备的分层 worker 预留")
            return
        # TP 孤岛网关节点不参与 PyTorch 层拆分：直接拒绝分层配置并退出
        # 分层 worker 池（与 llama_cpp 全模型节点的语义一致，防止孤岛引擎
        # 被 load_layer_range 覆盖为 PyTorch 层段）。
        try:
            import config as _island_cfg
            island_gateway = bool(getattr(_island_cfg, "ISLAND_ENABLED", False))
        except Exception:
            island_gateway = False
        if island_gateway:
            error = "本设备为 TP 孤岛网关节点，不参与 PyTorch 层拆分"
            logger.info(error)
            self._send_layer_config_ack({
                "node_id": node_id,
                "assignment_id": ack_assignment_id,
                "config_id": str(data.get("config_id", "")) if isinstance(data, dict) else "",
                "generation": ack_generation,
                "phase": ack_phase,
                "status": "error",
                "error": error,
            })
            self.release_pipeline_worker_for_local_model()
            return

        if node_id in data and isinstance(data.get(node_id), dict):
            cfg = dict(data[node_id])
        elif isinstance(data, dict) and "start_layer" in data and "end_layer" in data:
            cfg = dict(data)
        else:
            error = f"分层配置中未找到本节点 {node_id} 的有效 assignment"
            logger.warning(error)
            self._send_layer_config_ack({
                "node_id": node_id,
                "assignment_id": ack_assignment_id,
                "config_id": data.get("config_id", "") if isinstance(data, dict) else "",
                "generation": ack_generation,
                "phase": ack_phase,
                "status": "error",
                "error": error,
            })
            return

        config_id = str(cfg.get("config_id", ""))
        ack_assignment_id = str(cfg.get("assignment_id", "") or "")
        ack_generation = cfg.get("generation", ack_generation)
        target_node_id = str(cfg.get("node_id", node_id))
        start = cfg.get("start_layer", 0)
        end = cfg.get("end_layer", 24)
        has_embed = cfg.get("has_embedding", False)
        has_lm = cfg.get("has_lm_head", False)
        model_id = str(cfg.get("model_id", ""))
        expected_sha256 = str(cfg.get("model_sha256", ""))
        expected_model_type = str(cfg.get("model_type", "")).lower()
        expected_engine = str(cfg.get("engine", "pytorch") or "pytorch").lower()
        master_quant_type = str(cfg.get("master_quant_type", "") or "")
        phase = str(cfg.get("phase", "commit") or "commit").lower()
        plan_id = str(cfg.get("plan_id", "") or "")
        try:
            start = int(start)
            end = int(end)
            total_layers = int(cfg.get("total_layers", 0) or 0)
            master_api_port = int(cfg.get("master_api_port", 8000) or 8000)
            required_bytes = int(cfg.get("required_bytes", 0) or 0)
        except (TypeError, ValueError) as exc:
            error = f"分层配置数字字段无效: {exc}"
            logger.warning(error)
            self._send_layer_config_ack({
                "node_id": node_id,
                "assignment_id": ack_assignment_id,
                "config_id": config_id,
                "generation": ack_generation,
                "phase": phase,
                "status": "error",
                "error": error,
            })
            return
        configuration_invalidated = False

        logger.info(
            f"🔧 收到分层配置: 节点={node_id}, "
            f"Layer {start}-{end}, embed={has_embed}, lm_head={has_lm}, "
            f"config_id={config_id or 'legacy'}"
        )

        try:
            if target_node_id != node_id:
                raise ValueError(f"层配置目标节点 {target_node_id} 与本节点 {node_id} 不一致")
            # ★ #31 M2：同上，走单一事实来源
            relay_assignment = expected_engine == "relay_middle"
            # Endpoint-backed relay workers own their model artifact remotely;
            # allow a restart-time metadata gap to rehydrate their logical
            # assignment using the relay segment contract alone.
            if not relay_assignment and expected_model_type not in PIPELINE_RUNTIME_MODEL_TYPES:
                raise ValueError(f"不支持的流水线模型架构: {expected_model_type or 'unknown'}")
            if expected_engine not in {"pytorch", "relay_middle"}:
                raise ValueError(
                    f"分层配置引擎必须为 pytorch 或 relay_middle，实际为 {expected_engine}"
                )
            contract_fields = (
                (("config_id", config_id),)
                if relay_assignment
                else (
                    ("config_id", config_id),
                    ("model_id", model_id),
                    ("model_sha256", expected_sha256),
                    ("total_layers", total_layers),
                )
            )
            missing_contract = [name for name, value in contract_fields if not value]
            if missing_contract:
                raise ValueError(
                    "分层配置执行契约不完整: " + ", ".join(missing_contract)
                )
            if phase not in {"prepare", "commit"}:
                raise ValueError(f"不支持的分层加载阶段: {phase}")
            if phase == "prepare" and not plan_id:
                raise ValueError("prepare 阶段缺少 plan_id")
            if phase == "prepare" and expected_engine != "relay_middle" and required_bytes <= 0:
                raise ValueError("prepare 阶段缺少 required_bytes")
            relay_spec = None
            if relay_assignment:
                relay_spec = self._normalize_relay_segment(
                    cfg.get("relay_segment")
                )
                if relay_spec is None:
                    raise ValueError(
                        "relay_middle requires a valid relay_segment"
                    )
            if phase == "commit" and plan_id:
                with self._layer_config_lock:
                    active_config = (
                        dict(self._active_layer_config)
                        if isinstance(self._active_layer_config, dict)
                        else {}
                    )
                    previous_ready = (
                        dict(self._last_layer_config_ack_payload)
                        if isinstance(self._last_layer_config_ack_payload, dict)
                        else {}
                    )
                if (
                    active_config.get("config_id") == config_id
                    and active_config.get("assignment_id") == ack_assignment_id
                    and active_config.get("layer_range") == [start, end]
                    and active_config.get("model_sha256") == expected_sha256
                    and active_config.get("engine") == expected_engine
                    and (
                        not relay_assignment
                        or active_config.get("relay_segment") == relay_spec
                    )
                    and previous_ready.get("status") == "ready"
                    and previous_ready.get("config_id") == config_id
                    and previous_ready.get("assignment_id") == ack_assignment_id
                    and previous_ready.get("plan_id") == plan_id
                    and previous_ready.get("layer_range") == [start, end]
                    and previous_ready.get("model_sha256") == expected_sha256
                    and previous_ready.get("engine") == expected_engine
                    and (
                        not relay_assignment
                        or previous_ready.get("relay_segment") == relay_spec
                    )
                    and int(previous_ready.get("generation", 0) or 0)
                    == int(ack_generation or 0)
                ):
                    # A delayed/lost ready ACK makes the master resend commit.
                    # Both PyTorch and endpoint-backed relay assignments have
                    # already consumed their prepared record at this point, so
                    # replay the canonical receipt before either branch tries
                    # to validate that now-absent record.  A duplicate commit
                    # must never roll an already-published worker back.
                    previous_ready["timestamp"] = time.time()
                    self._send_layer_config_ack(previous_ready)
                    return
            if expected_engine == "relay_middle":
                if not PIPELINE_RELAY_ENABLED:
                    raise ValueError("relay_middle requires an enabled valid relay_segment")
                # relay_middle is endpoint-backed.  The independently
                # supervised relay_mid_service owns its segment artifact;
                # this scheduler worker must not load a full model or a
                # PyTorch layer range just to accept the logical assignment.
                active_config = {
                    "node_id": node_id, "assignment_id": ack_assignment_id,
                    "config_id": config_id,
                    "model_id": model_id, "model_sha256": expected_sha256,
                    "model_type": expected_model_type, "layer_range": [start, end],
                    "engine": expected_engine, "relay_segment": relay_spec,
                }
                if phase == "prepare":
                    with self._layer_config_lock:
                        self._prepared_layer_configs[config_id] = {
                            **active_config,
                            "plan_id": plan_id,
                            "previous_active_config": (
                                dict(self._active_layer_config)
                                if isinstance(self._active_layer_config, dict)
                                else None
                            ),
                            "previous_reserved": bool(
                                self._pipeline_worker_reserved
                            ),
                            "previous_ack": self._last_layer_config_ack_payload,
                        }
                    self._send_layer_config_ack({
                        "node_id": node_id, "config_id": config_id,
                        "assignment_id": ack_assignment_id,
                        "generation": ack_generation, "status": "prepared", "phase": phase,
                        "plan_id": plan_id, "layer_range": [start, end],
                        "model_sha256": expected_sha256, "model_type": expected_model_type,
                        "engine": expected_engine, "relay_segment": relay_spec,
                        "timestamp": time.time(),
                    })
                    return
                prepared = {}
                if plan_id:
                    with self._layer_config_lock:
                        prepared = dict(self._prepared_layer_configs.get(config_id, {}))
                    if (
                        prepared.get("plan_id") != plan_id
                        or prepared.get("assignment_id") != ack_assignment_id
                        or prepared.get("layer_range") != [start, end]
                        or prepared.get("model_sha256") != expected_sha256
                    ):
                        raise RuntimeError("commit 未命中同代际 relay prepared 记录")
                active_config = {
                    "node_id": node_id, "assignment_id": ack_assignment_id,
                    "config_id": config_id,
                    "model_id": model_id, "model_sha256": expected_sha256,
                    "model_type": expected_model_type, "layer_range": [start, end],
                    "engine": expected_engine, "relay_segment": relay_spec,
                }
                with self._layer_config_lock:
                    if plan_id:
                        self._pipeline_assignment_rollbacks[config_id] = {
                            "candidate_runtime": False,
                            "active_layer_config": prepared.get(
                                "previous_active_config"
                            ),
                            "pipeline_worker_reserved": bool(
                                prepared.get("previous_reserved", False)
                            ),
                            "last_layer_config_ack_payload": prepared.get(
                                "previous_ack"
                            ),
                        }
                    self._pipeline_worker_reserved = True
                    self._active_layer_config = dict(active_config)
                    self._local_pipeline_steps.clear()
                    self._prepared_layer_configs.pop(config_id, None)
                self._send_layer_config_ack({
                    "node_id": node_id, "config_id": config_id,
                    "assignment_id": ack_assignment_id,
                    "generation": ack_generation, "status": "ready", "phase": phase,
                    "plan_id": plan_id, "layer_range": [start, end],
                    "has_embedding": has_embed, "has_lm_head": has_lm,
                    "model_sha256": expected_sha256, "model_type": expected_model_type,
                    "engine": expected_engine, "relay_segment": relay_spec,
                    "timestamp": time.time(),
                })
                return
            prepared = {}
            if phase == "commit" and plan_id:
                with self._layer_config_lock:
                    prepared = dict(
                        self._prepared_layer_configs.get(config_id, {})
                    )
                if (
                    prepared.get("plan_id") != plan_id
                    or prepared.get("assignment_id") != ack_assignment_id
                    or prepared.get("layer_range") != [start, end]
                    or prepared.get("model_sha256") != expected_sha256
                    or any(
                        field in prepared and prepared[field] != value
                        for field, value in (
                            ("generation", ack_generation),
                            ("total_layers", total_layers),
                            ("model_id", model_id),
                            ("model_type", expected_model_type),
                            ("has_embedding", has_embed),
                            ("has_lm_head", has_lm),
                            ("tokenizer_sha256", str(cfg.get("tokenizer_sha256", "") or "")),
                        )
                    )
                ):
                    raise RuntimeError("commit 未命中同代际 prepared 记录")

            local_sha256 = ""
            local_model_path = None
            # Commit retains the already materialized transaction even when
            # the registration pointer has changed or has been revoked.
            supplied = (
                None if phase == "commit" and plan_id
                else self._selected_pipeline_safetensors()
            )
            if supplied is not None:
                _validate_pipeline_supply(
                    supplied, model_id=model_id, model_sha256=expected_sha256,
                    total_layers=total_layers, layer_range=(start, end),
                    tokenizer_sha256=str(cfg.get("tokenizer_sha256", "") or ""),
                    model_type=expected_model_type,
                )
            if phase == "commit" and plan_id:
                local_sha256 = str(prepared.get("model_sha256", "") or "")
                local_model_path = prepared.get("model_path")
            elif supplied is not None:
                local_model_path = supplied["directory"]
                local_sha256 = supplied["manifest"]["source_model_sha256"]
            elif phase == "prepare" and plan_id and cfg.get("assignment_manifest"):
                from model_sync import (
                    ensure_pipeline_assignment_available,
                    resolve_worker_model_path,
                )
                from transport_port import compute_local_model_sha256

                tcp_client = getattr(self, "_tcp_client", None)
                master_host = getattr(tcp_client, "server_host", "")
                if not master_host:
                    raise RuntimeError("无法确定主节点模型下载地址")
                # Prefer an already provisioned full model. The assignment
                # manifest is only a cold-start fallback for workers that do
                # not own the same revision locally.
                local_model_path = resolve_worker_model_path(model_id)
                local_sha256 = compute_local_model_sha256(
                    model_path=local_model_path,
                    model_id=model_id,
                )
                if local_sha256 == expected_sha256:
                    logger.info(
                        "worker local model revision matches; skip assignment weight transfer: model=%s",
                        model_id,
                    )
                else:
                    local_model_path, assignment_manifest = ensure_pipeline_assignment_available(
                        master_host,
                        master_api_port,
                        {
                            **cfg,
                            "model_id": model_id,
                            "model_sha256": expected_sha256,
                        },
                    )
                    local_sha256 = expected_sha256
            elif expected_sha256:
                from transport_port import compute_local_model_sha256
                if model_id:
                    from model_sync import (
                        ensure_model_available,
                        resolve_worker_model_path,
                    )

                    local_model_path = resolve_worker_model_path(model_id)
                    local_sha256 = compute_local_model_sha256(
                        model_path=local_model_path,
                        model_id=model_id,
                    )
                    if local_sha256 != expected_sha256:
                        tcp_client = getattr(self, "_tcp_client", None)
                        master_host = getattr(tcp_client, "server_host", "")
                        if not master_host:
                            raise RuntimeError("无法确定主节点模型下载地址")
                        logger.info("从主节点同步流水线模型: %s", model_id)
                        local_model_path = ensure_model_available(
                            master_host,
                            master_api_port,
                            model_id,
                            expected_sha256,
                        )
                        local_sha256 = compute_local_model_sha256(
                            model_path=local_model_path,
                            model_id=model_id,
                        )
                else:
                    local_sha256 = compute_local_model_sha256()
                if not local_sha256:
                    raise FileNotFoundError("本节点未找到可校验的 PyTorch 模型权重")
                if local_sha256 != expected_sha256:
                    raise ValueError(
                        f"模型 SHA256 不一致: local={local_sha256[:16]}... "
                        f"master={expected_sha256[:16]}..."
                    )

            if phase == "prepare":
                from pipeline_model_descriptor import inspect_pipeline_model

                try:
                    descriptor = inspect_pipeline_model(
                        local_model_path,
                        model_id=model_id,
                        layer_range=(start, end),
                    )
                except TypeError as exc:
                    # Keep compatibility with older test/sidecar adapters
                    # that still expose the C1 two-argument inspector.
                    if "layer_range" not in str(exc):
                        raise
                    descriptor = inspect_pipeline_model(
                        local_model_path, model_id=model_id,
                    )
                if (
                    descriptor.get("model_type") != expected_model_type
                    or int(descriptor.get("total_layers", 0) or 0) != total_layers
                    or start < 0 or end <= start or end > total_layers
                ):
                    raise ValueError("worker 工件描述器与 prepare 契约不一致")
                if supplied is not None:
                    manifest = supplied["manifest"]
                    for field in (
                        "source_model_id", "source_model_sha256", "tokenizer_sha256",
                    ):
                        if descriptor.get(field) != manifest[field]:
                            raise _PipelineSupplyError(field)
                    if descriptor.get("supply_manifest_sha256") != manifest["manifest_sha256"]:
                        raise _PipelineSupplyError("manifest_sha256")
                    if descriptor.get("assignment_layer_range") != [start, end]:
                        raise _PipelineSupplyError("layer_range")
                profile = dict(self._local_device_profile or {})
                gpu = self._select_scoring_gpu(profile)
                cuda_discrete = bool(
                    isinstance(gpu, dict)
                    and gpu.get("cuda_available", False)
                    and not self._gpu_is_integrated(gpu)
                )
                if cuda_discrete:
                    free_gb = float(gpu.get("vram_free_gb", 0) or 0)
                    capacity_source = "gpu.vram_free_gb"
                else:
                    ram = profile.get("ram", {})
                    free_gb = float(
                        ram.get("available_gb", 0) or 0
                    ) if isinstance(ram, dict) else 0.0
                    capacity_source = "ram.available_gb"
                available_bytes = max(0, int(free_gb * 1024 ** 3))
                if available_bytes < required_bytes:
                    raise RuntimeError(
                        "worker 实时容量不足: "
                        f"required={required_bytes}, available={available_bytes}"
                    )
                prepare_candidate = getattr(
                    self._host, "prepare_pipeline_candidate", None,
                )
                materialize_candidate = getattr(
                    self._host, "materialize_pipeline_candidate", None,
                )
                if not callable(prepare_candidate) or not callable(
                    materialize_candidate
                ):
                    raise RuntimeError(
                        "model host does not support transactional pipeline candidates"
                    )
                prepare_candidate(
                    config_id,
                    model_id=model_id,
                    model_path=local_model_path,
                    quant_type=master_quant_type or None,
                    layer_range=(start, end),
                    model_sha256=local_sha256 or expected_sha256,
                )
                materialize_candidate(
                    config_id,
                    start_layer=start,
                    end_layer=end,
                    has_embedding=has_embed,
                    has_lm_head=has_lm,
                    total_layers=total_layers or None,
                    model_id=model_id or None,
                )
                if supplied is not None:
                    current = self._selected_pipeline_safetensors()
                    if (
                        current is None
                        or current["directory"] != supplied["directory"]
                        or current["manifest"]["manifest_sha256"]
                        != supplied["manifest"]["manifest_sha256"]
                    ):
                        raise _PipelineSupplyError("registration", changed=True)
                prepared_record = {
                    "generation": ack_generation,
                    "total_layers": total_layers,
                    "has_embedding": has_embed,
                    "has_lm_head": has_lm,
                    "tokenizer_sha256": str(cfg.get("tokenizer_sha256", "") or ""),
                    "assignment_id": ack_assignment_id,
                    "config_id": config_id,
                    "plan_id": plan_id,
                    "model_id": model_id,
                    "model_sha256": local_sha256,
                    "model_type": expected_model_type,
                    "model_path": local_model_path,
                    "layer_range": [start, end],
                    "required_bytes": required_bytes,
                    "available_bytes": available_bytes,
                    "capacity_source": capacity_source,
                    "previous_active_config": (
                        dict(self._active_layer_config)
                        if isinstance(self._active_layer_config, dict)
                        else None
                    ),
                    "previous_reserved": bool(self._pipeline_worker_reserved),
                    "previous_ack": self._last_layer_config_ack_payload,
                }
                with self._layer_config_lock:
                    self._prepared_layer_configs[config_id] = prepared_record
                self._send_layer_config_ack({
                    "node_id": node_id,
                    "assignment_id": ack_assignment_id,
                    "config_id": config_id,
                    "generation": ack_generation,
                    "status": "prepared",
                    "phase": "prepare",
                    "plan_id": plan_id,
                    "layer_range": [start, end],
                    "model_sha256": local_sha256,
                    "model_type": expected_model_type,
                    "engine": "pytorch",
                    "required_bytes": required_bytes,
                    "available_bytes": available_bytes,
                    "capacity_source": capacity_source,
                    "timestamp": time.time(),
                })
                return

            mgr = self._host
            if plan_id:
                commit_candidate = getattr(
                    self._host, "commit_pipeline_candidate", None,
                )
                if not callable(commit_candidate):
                    raise RuntimeError(
                        "model host does not support transactional pipeline commit"
                    )
                commit_candidate(config_id)
            else:
                # Legacy one-phase assignments retain the old destructive path.
                with self._layer_config_lock:
                    self._pipeline_worker_reserved = True
                    self._active_layer_config = None
                    self._last_layer_config_ack_payload = None
                    self._local_pipeline_steps.clear()
                self._host.model_loaded = False
                configuration_invalidated = True

            if not plan_id and mgr and mgr.is_loaded:
                # 如果已加载完整模型，重新加载指定层范围
                logger.info(f"🔄 重新加载模型层范围: {start}-{end}")
                mgr.load_layer_range(
                    start, end,
                    has_embedding=has_embed,
                    has_lm_head=has_lm,
                    model_path=local_model_path,
                    quant_type=master_quant_type or None,
                    total_layers=total_layers or None,
                    model_id=model_id or None,
                )
            elif not plan_id and mgr:
                # 模型尚未加载，先加载层范围
                logger.info(f"📥 首次加载模型层范围: {start}-{end}")
                mgr.load_layer_range(
                    start, end,
                    has_embedding=has_embed,
                    has_lm_head=has_lm,
                    model_path=local_model_path,
                    quant_type=master_quant_type or None,
                    total_layers=total_layers or None,
                    model_id=model_id or None,
                )
            elif not plan_id:
                raise RuntimeError("model_manager 不可用，无法加载层范围")

            actual_range = getattr(mgr, 'layer_range', None)
            if actual_range is not None and tuple(actual_range) != (start, end):
                raise RuntimeError(
                    f"模型层范围加载结果不一致: actual={actual_range}, expected=({start}, {end})"
                )
            engine = backend_id_for(mgr, default='pytorch') or 'pytorch'
            if engine != 'pytorch':
                raise RuntimeError(f"层拆分要求 PyTorch 引擎，实际为 {engine}")
            loaded_config = getattr(getattr(mgr, "model", None), "config", None)
            actual_model_type = str(
                getattr(loaded_config, "model_type", "") or ""
            ).lower()
            if actual_model_type != expected_model_type:
                raise RuntimeError(
                    f"模型架构不一致: actual={actual_model_type}, "
                    f"expected={expected_model_type}"
                )

            # Model loading can take minutes. A newer assignment/release or a
            # master disconnect may arrive while this thread owns the execution
            # lock; never publish the obsolete load as ready afterwards.
            if receive_sequence is not None:
                with self._layer_config_lock:
                    if (receive_sequence
                            != self._latest_layer_config_receive_sequence
                            or (self._latest_layer_config_generation
                                and generation
                                < self._latest_layer_config_generation)):
                        raise RuntimeError(
                            "分层配置在模型加载期间已被更新代际取代"
                        )

            active_config = {
                "node_id": node_id,
                "assignment_id": ack_assignment_id,
                "config_id": config_id,
                "model_id": model_id,
                "model_sha256": local_sha256 or expected_sha256,
                "model_type": actual_model_type,
                "layer_range": [start, end],
                "engine": engine,
                "master_quant_type": master_quant_type,
                "runtime_quant_type": getattr(mgr, "quant_type", "") or "",
            }
            with self._layer_config_lock:
                if plan_id:
                    self._pipeline_assignment_rollbacks[config_id] = {
                        "candidate_runtime": True,
                        "active_layer_config": prepared.get(
                            "previous_active_config"
                        ),
                        "pipeline_worker_reserved": bool(
                            prepared.get("previous_reserved", False)
                        ),
                        "last_layer_config_ack_payload": prepared.get(
                            "previous_ack"
                        ),
                    }
                self._pipeline_worker_reserved = True
                self._active_layer_config = dict(active_config)
                self._local_pipeline_steps.clear()
                self._prepared_layer_configs.pop(config_id, None)
            # Layer config may be the first model load on a clean worker. Keep the
            # API's compatibility globals aligned so the first forwarded chat does
            # not auto-load a full/GGUF model over this segment.
            self._host.model_loaded = bool(
                getattr(self._host, "is_loaded", False)
                and not getattr(self._host, "is_pipeline_prepared", False)
            )
            self._host.current_quant = getattr(mgr, "quant_type", None) or "fp16"

            self._send_layer_config_ack({
                "node_id": node_id,
                "assignment_id": ack_assignment_id,
                "config_id": config_id,
                "generation": ack_generation,
                "status": "ready",
                "phase": phase,
                "plan_id": plan_id,
                "layer_range": [start, end],
                "has_embedding": has_embed,
                "has_lm_head": has_lm,
                "model_sha256": local_sha256 or expected_sha256,
                "model_type": actual_model_type,
                "engine": engine,
                "master_quant_type": master_quant_type,
                "runtime_quant_type": getattr(mgr, "quant_type", "") or "",
                "timestamp": time.time(),
            })
            logger.info(
                f"✅ 模型层加载完成并已确认: node={node_id}, "
                f"Layer {start}-{end}, config_id={config_id or 'legacy'}"
            )
        except Exception as e:
            if plan_id:
                abort_candidate = getattr(
                    self._host, "abort_pipeline_candidate", None,
                )
                if callable(abort_candidate):
                    try:
                        abort_candidate(config_id)
                    except Exception:
                        logger.warning(
                            "worker 候选模型回滚失败: config=%s",
                            config_id, exc_info=True,
                        )
            if configuration_invalidated:
                with self._layer_config_lock:
                    self._active_layer_config = None
                    self._last_layer_config_ack_payload = None
                    self._local_pipeline_steps.clear()
                self._host.model_loaded = False
            logger.error(f"加载层范围失败: {e}", exc_info=True)
            self._send_layer_config_ack({
                "node_id": node_id,
                "assignment_id": ack_assignment_id,
                "config_id": config_id,
                "generation": ack_generation,
                "status": "error",
                "phase": phase,
                "layer_range": [start, end],
                "model_sha256": "",
                "model_type": expected_model_type,
                "engine": expected_engine,
                "error": str(e),
                **({"reason_code": e.reason_code} if isinstance(e, _PipelineSupplyError) else {}),
                "timestamp": time.time(),
            })


    def _send_layer_config_ack(self, payload: dict) -> bool:
        """从节点向主节点回传层配置加载结果。"""
        from transport_port import MessageType

        if (
            payload.get("config_id")
            and payload.get("status") in {"prepared", "ready"}
        ):
            with self._layer_config_lock:
                self._last_layer_config_ack_payload = dict(payload)

        client = getattr(self, '_tcp_client', None)
        if client is None:
            logger.warning("TCP 客户端未连接，无法发送层配置 ACK")
            return False
        try:
            client.send_data(payload, MessageType.LAYER_CONFIG_ACK)
            return True
        except Exception as e:
            logger.error(f"发送层配置 ACK 失败: {e}", exc_info=True)
            return False


    def _handle_layer_config_ack(self, client_id: str, msg: dict) -> None:
        """Validate legacy ready ACKs and capacity prepare/commit phases."""
        data = msg.get("data", {})
        node_id = str(data.get("node_id", client_id))
        config_id = str(data.get("config_id", ""))
        if node_id != client_id:
            logger.warning(
                "忽略节点标识不一致的层配置 ACK: connection=%s payload=%s",
                client_id, node_id,
            )
            return

        if (
            data.get("status") in {"finalized", "released"}
            and data.get("cleanup_action") in {"abort", "finalize"}
        ):
            self._ack_pipeline_model_cleanup(client_id, data)
            return

        commit_config_id = ""
        complete_config_id = ""
        abort_details = None
        with self._layer_config_lock:
            transaction = self._pipeline_load_transaction
            candidate_transaction = bool(
                isinstance(transaction, dict)
                and transaction.get("candidate_id")
                and transaction.get("config_id") == config_id
            )
            candidate_expected = (
                transaction.get("expected", {}).get(client_id)
                if candidate_transaction else None
            )
            expected = (
                candidate_expected
                if isinstance(candidate_expected, dict)
                else self._worker_assignments.expected(client_id)
            )
            if expected.get("worker_prepare_identity"):
                logger.warning("event=legacy_ack_rejected_for_native_prepare node=%s", client_id)
                return
            if not expected:
                logger.warning("忽略未请求的层配置 ACK: node=%s", client_id)
                return
            if config_id != expected.get("config_id"):
                logger.warning(
                    "忽略过期层配置 ACK: node=%s config_id=%s expected=%s",
                    client_id, config_id, expected.get("config_id"),
                )
                return

            expected_assignment_id = str(
                expected.get("assignment_id", "") or ""
            )
            ack_assignment_id = str(data.get("assignment_id", "") or "")
            if expected_assignment_id and ack_assignment_id != expected_assignment_id:
                logger.warning(
                    "忽略过期层配置 ACK assignment: node=%s config=%s "
                    "ack=%s expected=%s",
                    client_id,
                    config_id,
                    ack_assignment_id or "missing",
                    expected_assignment_id,
                )
                return

            # A versioned assignment is fenced by both config_id and
            # generation. This rejects a delayed prepare/ready ACK after a
            # reconnect or replacement assignment, while preserving the
            # legacy path for expectations that never carried generation.
            if not expected.get("release") and "generation" in expected:
                try:
                    expected_generation = int(expected.get("generation"))
                    ack_generation = int(data.get("generation"))
                except (TypeError, ValueError):
                    logger.warning(
                        "忽略缺少或无效 generation 的层配置 ACK: node=%s config=%s",
                        client_id, config_id,
                    )
                    return
                if ack_generation != expected_generation:
                    logger.warning(
                        "忽略过期层配置 ACK generation: node=%s config=%s ack=%s expected=%s",
                        client_id, config_id, ack_generation, expected_generation,
                    )
                    return

            expected_phase = str(expected.get("phase", "commit") or "commit")
            if (
                expected_assignment_id
                and not expected.get("release")
                and data.get("status") == "error"
                and str(data.get("phase", "") or "") != expected_phase
            ):
                logger.warning(
                    "忽略过期层配置 error ACK phase: node=%s config=%s "
                    "ack=%s expected=%s assignment=%s",
                    client_id,
                    config_id,
                    data.get("phase", "missing"),
                    expected_phase,
                    expected_assignment_id,
                )
                return

            if expected.get("release"):
                try:
                    ack_generation = int(data.get("generation", 0) or 0)
                except (TypeError, ValueError):
                    ack_generation = -1
                released = (
                    data.get("status") == "released"
                    and data.get("release") is True
                    and ack_generation == int(expected.get("generation", 0) or 0)
                )
                if not candidate_transaction:
                    self._worker_assignments.record_ack(client_id, data)
                if released:
                    self._layer_config_retry_state.pop(client_id, None)
                    self._worker_assignments.release(
                        client_id, reason_code=REASON_WORKER_RELEASED,
                    )
                else:
                    state = self._layer_config_retry_state.setdefault(
                        client_id, {"attempts": 0, "next_retry": 0.0}
                    )
                    state["next_retry"] = time.monotonic() + 5.0
                release_ack = True
                ready = False
                prepared = False
                prepared_late = False
                expected_range = []
            else:
                release_ack = False
                expected_range = [expected["start_layer"], expected["end_layer"]]
                prepared = (
                    expected_phase == "prepare"
                    and data.get("status") == "prepared"
                    and data.get("phase") == "prepare"
                    and data.get("plan_id") == expected.get("plan_id")
                    and data.get("layer_range") == expected_range
                    and data.get("model_sha256") == expected.get("model_sha256")
                    and data.get("model_type") == expected.get("model_type")
                    and data.get("engine") == expected.get("engine", "pytorch")
                    and int(data.get("available_bytes", 0) or 0)
                    >= int(expected.get("required_bytes", 0) or 0)
                )
                # A fast worker may deliver prepare after the coordinator has
                # already published commit. This is a valid same-generation
                # state update, not a worker failure.
                prepared_late = (
                    expected_phase == "commit"
                    and data.get("status") == "prepared"
                    and data.get("phase") == "prepare"
                    and data.get("plan_id") == expected.get("plan_id")
                    and data.get("layer_range") == expected_range
                    and data.get("model_sha256") == expected.get("model_sha256")
                    and data.get("model_type") == expected.get("model_type")
                    and data.get("engine") == expected.get("engine", "pytorch")
                )
                ready = (
                    expected_phase == "commit"
                    and data.get("status") == "ready"
                    and data.get("layer_range") == expected_range
                    and data.get("model_sha256") == expected.get("model_sha256")
                    and data.get("model_type") == expected.get("model_type")
                    and data.get("engine") == expected.get("engine", "pytorch")
                    and (
                        "has_embedding" not in data
                        or bool(data.get("has_embedding"))
                        == bool(expected.get("has_embedding"))
                    )
                    and (
                        "has_lm_head" not in data
                        or bool(data.get("has_lm_head"))
                        == bool(expected.get("has_lm_head"))
                    )
                    and (
                        not expected.get("plan_id")
                        or data.get("plan_id") == expected.get("plan_id")
                    )
                )
                if candidate_transaction:
                    transaction.setdefault("acks", {})[client_id] = dict(data)
                else:
                    self._worker_assignments.record_ack(client_id, data)
                # ★ 2026-10-03：v3 层段 worker 会**明确拒绝** legacy 配置（它手上有工件、
                #   层段由 stage offer 驱动，见 `_handle_layer_config_locked` 里的同名分流）。
                #   这不是"未就绪"，而是"不参与这条通道" ⇒ 把它从待 ACK 集合里摘掉，
                #   既不计入 commit 门槛、也不阻塞请求。否则整个 `distributed_required`
                #   会以 `pipeline workers not ready: layer_stage_worker_rejects_legacy_config`
                #   失败（实测）。摘除这一步必须在这里做：master 是在**节点注册那一刻**
                #   推送 legacy 配置的，那时 hello 还没往返，按 capabilities 排除不可靠。
                if (
                    str(data.get("status", "")) == "error"
                    and str(data.get("error", ""))
                    == "layer_stage_worker_rejects_legacy_config"
                ):
                    if candidate_transaction:
                        transaction.get("expected", {}).pop(client_id, None)
                        transaction.get("retry_state", {}).pop(client_id, None)
                    else:
                        self._layer_config_retry_state.pop(client_id, None)
                    # ★ 2026-10-07（DIST-NEXT-3）：该节点不参与 legacy 通道 ⇒ 权威视图进
                    #   终止态（派生视图随之不含它）。
                    if not candidate_transaction:
                        self._worker_assignments.release(
                            client_id,
                            reason_code="layer_stage_worker_rejects_legacy_config",
                        )
                    if (
                        transaction
                        and transaction.get("config_id") == expected.get("config_id")
                    ):
                        remaining = set(transaction.get("worker_ids", set()))
                        remaining.discard(client_id)
                        transaction["worker_ids"] = remaining
                    logger.info(
                        "v3 层段 worker 不参与 legacy 分层通道，已摘除: node=%s",
                        client_id,
                    )
                    return
                if not (ready or prepared or prepared_late):
                    # ★ 诊断：把 ACK 与期望的**逐字段差异**一次打全 ——
                    #   排障跨机 relay 时，ACK 恒判失败却完全看不出是哪个字段不等
                    #   （`expected` 来自 `_publish_layer_configs` 写入的原始 config，
                    #    含 `phase="prepare"` ⇒ 正常应命中 `prepared` 判据）。
                    logger.warning(
                        "层配置 ACK 未通过 node=%s expected_phase=%s 差异=%s",
                        client_id, expected_phase,
                        {
                            key: {"got": data.get(key), "want": expected.get(key)}
                            for key in (
                                "status", "phase", "plan_id", "layer_range",
                                "model_sha256", "model_type", "engine",
                                "required_bytes", "available_bytes",
                            )
                            if data.get(key) != expected.get(key)
                        },
                    )
                if ready:
                    # ★ 2026-10-07（DIST-NEXT-3 第三步）：legacy ready ACK 推进**权威视图**；
                    #   `_layer_config_pushed` 是派生视图，无需再单独 add。
                    if candidate_transaction:
                        transaction.get("retry_state", {}).pop(client_id, None)
                    else:
                        self._worker_assignments.transition(
                            client_id,
                            phase=PHASE_READY,
                            reason_code=REASON_CONFIG_ACKED,
                        )
                        self._layer_config_retry_state.pop(client_id, None)
                    if (
                        transaction
                        and transaction.get("config_id") == config_id
                        and transaction.get("phase") == "committing"
                    ):
                        ready_nodes = set(transaction.get("ready_nodes", set()))
                        ready_nodes.add(client_id)
                        transaction["ready_nodes"] = ready_nodes
                        if ready_nodes == set(transaction.get("worker_ids", set())):
                            transaction["phase"] = "committing_global"
                            self._persist_pipeline_lifecycle_locked()
                            complete_config_id = config_id
                elif prepared:
                    # ★ 2026-10-07（DIST-NEXT-3）：派生视图不因**迟到的 prepared** 撤回 ready
                    #   （正常流程 prepared 先于 ready；旧写法在这里 discard 会把已就绪的节点
                    #   踢出 `pushed`，那本身是个隐患）。
                    if candidate_transaction:
                        transaction.get("retry_state", {}).pop(client_id, None)
                    else:
                        self._layer_config_retry_state.pop(client_id, None)
                    if (
                        transaction
                        and transaction.get("config_id") == config_id
                        and transaction.get("phase") == "preparing"
                    ):
                        prepared_nodes = set(transaction.get("prepared_nodes", set()))
                        prepared_nodes.add(client_id)
                        transaction["prepared_nodes"] = prepared_nodes
                        if prepared_nodes == set(transaction.get("worker_ids", set())):
                            commit_config_id = config_id
                elif prepared_late:
                    if candidate_transaction:
                        transaction.get("retry_state", {}).pop(client_id, None)
                    else:
                        self._layer_config_retry_state.pop(client_id, None)
                else:
                    # ★ 2026-10-07（DIST-NEXT-3）：ACK 未通过任一判据 ⇒ 就绪证据作废
                    #   （相位退回 `pushing`），派生视图随之不含该节点。
                    if candidate_transaction:
                        retry_state = transaction.setdefault("retry_state", {})
                    else:
                        self._worker_assignments.invalidate(
                            client_id, reason_code="layer_config_ack_rejected",
                        )
                        retry_state = self._layer_config_retry_state
                    state = retry_state.setdefault(
                        client_id, {"attempts": 0, "next_retry": 0.0},
                    )
                    state["next_retry"] = time.monotonic() + 5.0
                    if (
                        transaction
                        and transaction.get("config_id") == config_id
                        and data.get("status") == "error"
                    ):
                        abort_details = (
                            config_id,
                            f"pipeline_{expected_phase}_failed",
                            str(data.get("error", "") or "worker rejected phase"),
                        )

        self._maybe_finish_pipeline_recovery()
        if release_ack:
            if released:
                logger.info(
                    "从节点已确认退出分层 worker: node=%s config_id=%s",
                    client_id, config_id,
                )
            else:
                logger.error("从节点分层释放 ACK 未通过: node=%s", client_id)
            return
        if abort_details is not None:
            self._abort_pipeline_load_transaction(*abort_details)
            return
        if complete_config_id:
            self._complete_pipeline_load_transaction(complete_config_id)
            return
        if commit_config_id:
            self._commit_pipeline_load_transaction(commit_config_id)
            return
        if prepared:
            logger.info(
                "流水线 worker prepare 通过: node=%s config=%s",
                client_id, config_id,
            )
        elif ready:
            logger.info(
                "从节点层配置已就绪: node=%s range=%s config=%s",
                client_id, expected_range, config_id,
            )
        elif prepared_late:
            logger.info(
                "忽略已进入 commit 的迟到 prepare ACK: node=%s config=%s",
                client_id, config_id,
            )
        else:
            logger.error(
                "从节点层配置阶段未通过: node=%s status=%s error=%s",
                client_id, data.get("status"), data.get("error", ""),
            )


    def _handle_layer_worker_opt_out(
        self, client_id: str, msg: dict, *, hello_work=None,
    ) -> None:
        """Remove a connected client from future layer assignments."""
        data = msg.get("data", {})
        node_id = str(data.get("node_id", client_id))
        if node_id != client_id:
            logger.warning(
                "忽略来源不一致的分层退出请求: connection=%s payload=%s",
                client_id,
                node_id,
            )
            return
        deferred_dispatch = []
        with self._layer_config_lock, ExitStack() as owner_guard:
            if hello_work is not None:
                server = hello_work[0]
                if self._tcp_server is not server:
                    return
                # Lock order: config -> transport owner -> control owner.
                # REGISTER replacement and HELLO snapshot updates use these
                # existing owner locks. Validate only after the config wait,
                # then keep identity fixed through ALL control-state changes.
                owner_guard.enter_context(server._clients_lock)
                owner_guard.enter_context(self._task_worker_control._lock)
                if not self._task_worker_hello_followup_current(client_id, hello_work):
                    return
                # Wall-clock HELLO timestamps can coincide. Check the live
                # role under the same owner lock before releasing reservations.
                snapshot = self._task_worker_control.worker_snapshot(client_id)
                capabilities = snapshot.get("capabilities") or {}
                if (not snapshot.get("accepted") or not capabilities.get("models")
                        or capabilities.get("layer_worker", False)
                        or ("layer_forward" in capabilities.get("stage_types", [])
                            and capabilities.get("layer_ranges"))):
                    return
            relay = self._is_relay_host(client_id)
            if relay:
                self._pipeline_worker_opt_out.discard(client_id)
            else:
                self._pipeline_worker_opt_out.add(client_id)
                transaction = self._pipeline_load_transaction
                if transaction and client_id in set(transaction.get("worker_ids", set())):
                    if transaction.get("phase") in {"preparing", "committing_local", "committing", "committing_global"}:
                        self._abort_pipeline_load_transaction(
                            transaction["config_id"], "pipeline_worker_opted_out",
                            "worker left distributed inference",
                            _deferred_dispatch=deferred_dispatch,
                        )
                    elif transaction.get("phase") == "ready":
                        self._invalidate_pipeline_load_transaction(
                            "pipeline_worker_opted_out",
                            expected_config_id=transaction["config_id"],
                            _deferred_dispatch=deferred_dispatch,
                        )
                self._clear_layer_config_state(client_id)
        # Host/stage work and network sends must run outside registration and
        # config locks. The existing ledger retains exact cleanup identities.
        for dispatch in deferred_dispatch:
            dispatch()
        if relay:
            logger.info("忽略 relay 节点的本地分层退出通知: node=%s", client_id)
        else:
            logger.info("从节点已退出 PyTorch 分层计算: node=%s", client_id)
        self.push_layer_config_to_clients()


    def _handle_layer_worker_opt_in(self, client_id: str, msg: dict) -> None:
        """Allow a connected client to receive layer assignments again."""
        data = msg.get("data", {})
        node_id = str(data.get("node_id", client_id))
        if node_id != client_id:
            logger.warning(
                "忽略来源不一致的分层加入请求: connection=%s payload=%s",
                client_id,
                node_id,
            )
            return
        with self._layer_config_lock:
            self._pipeline_worker_opt_out.discard(client_id)
        logger.info("从节点已重新加入 PyTorch 分层计算: node=%s", client_id)
        # A worker may still carry a local opt-out from its previous model
        # operation.  When the master has a prepared distributed artifact,
        # use the authoritative path so the assignment clears that stale
        # local state instead of immediately eliciting another opt-out.
        if getattr(self._host, "is_pipeline_prepared", False):
            self.request_authoritative_layer_sync()
        else:
            self.push_layer_config_to_clients()


    def _run_master_lm_head(self, hidden_states):
        """在主节点对 worker 返回的尾层 hidden states 执行 Norm + LM Head。"""
        mgr = self._host
        if (
            not mgr
            or not mgr.is_loaded
            or not runtime_supports(mgr, Capability.FORWARD_LAYERS)
        ):
            raise RuntimeError("主节点 PyTorch 模型未加载，无法执行 LM Head")
        project = getattr(mgr, "forward_lm_head", None)
        if not callable(project):
            raise RuntimeError("当前模型管理器不支持架构感知 LM Head")
        return project(hidden_states)


    def _handle_layer_forward(self, client_id: str, msg: dict) -> None:
        if client_id != "master":
            logger.warning(
                "丢弃非 master 来源的层前向指令: source=%s task=%s",
                client_id,
                msg.get("data", {}).get("task_id", "-") if isinstance(msg, dict) else "-",
            )
            return
        with self._layer_execution_lock:
            self._handle_layer_forward_locked(client_id, msg)


    def _handle_layer_forward_locked(self, client_id: str, msg: dict) -> None:
        """
        从节点：收到主节点的 LAYER_FORWARD → 执行本节点层前向 → 返回 LAYER_RESULT。

        消息格式:
            LAYER_FORWARD: { task_id, step, use_kv_cache,
                             input_ids?, hidden_states?,
                             attention_mask?, position_ids?,
                             temperature, top_p }

        **KV Cache 支持 (Phase 3)**:
         - use_kv_cache=True: 从本地 _kv_cache[task_id] 读取缓存的 KV，
           仅处理新 token（增量解码），计算后将新 KV 存回。
         - use_kv_cache=False: Prefill 模式，处理完整序列，构建新 KV cache。

        处理流程:
            1. 反序列化输入（input_ids 或 hidden_states）
            2. 根据 use_kv_cache 读取/写入本地 KV cache
            3. 调用 model_manager.forward_layers()
            4. 序列化输出（hidden_states 或 logits，不含 KV cache）
            5. 发送 LAYER_RESULT 回主节点
        """
        from transport_port import MessageType

        data = msg.get("data", {})
        task_id = str(data.get("task_id", "unknown") or "unknown")
        try:
            step = int(data.get("step", 0))
        except (TypeError, ValueError):
            step = -1
        use_kv_cache = data.get("use_kv_cache", False)
        config_id = str(data.get("config_id", ""))
        assignment_id = str(data.get("assignment_id", "") or "")
        model_sha256 = str(data.get("model_sha256", ""))
        model_type = str(data.get("model_type", "")).lower()
        layer_config_invalid = False
        received_chain_path = data.get("chain_path", [])
        if not isinstance(received_chain_path, list):
            received_chain_path = []
        logical_predecessor = (
            str(data.get("_chain_predecessor", "") or client_id)
            if client_id == "master" else client_id
        )

        logger.info(
            f"🔬 收到层前向指令: task={task_id}, step={step}, from={client_id}, "
            f"kv_cache={'on' if use_kv_cache else 'off'}"
        )

        try:
            if received_chain_path:
                normalized_path = [str(item) for item in received_chain_path]
                if (normalized_path[-1] != logical_predecessor
                        or self.get_effective_node_id() in normalized_path
                        or len(normalized_path) != len(set(normalized_path))):
                    raise RuntimeError(
                        f"链式转发路径与 TCP 前驱不一致: "
                        f"path={normalized_path}, predecessor={logical_predecessor}"
                    )
                received_chain_path = normalized_path
            with self._layer_config_lock:
                if task_id in self._local_pipeline_cancelled:
                    logger.info("忽略已取消任务的迟到层前向: task=%s", task_id)
                    return
                active_config = dict(self._active_layer_config or {})
                last_step = self._local_pipeline_steps.get(task_id)
                task_active = task_id in self._active_pipeline_task_ids
            if not active_config:
                layer_config_invalid = True
                raise RuntimeError("本节点没有已确认的活动层配置")
            for field, actual in (
                ("assignment_id", assignment_id),
                ("config_id", config_id),
                ("model_sha256", model_sha256),
                ("model_type", model_type),
            ):
                if not actual or actual != str(active_config.get(field, "")):
                    raise RuntimeError(
                        f"流水线执行契约不一致: {field}={actual or '-'}, "
                        f"expected={active_config.get(field, '-') }"
                    )
            if step < 0:
                raise RuntimeError(f"无效流水线 step: {step}")
            if step == 0:
                if use_kv_cache:
                    raise RuntimeError("prefill step 0 不得声明使用既有 KV cache")
                if task_active or last_step is not None:
                    raise RuntimeError(f"重复 prefill: task={task_id}")
            else:
                if not use_kv_cache:
                    raise RuntimeError(f"decode step {step} 必须使用 KV cache")
                if not task_active or last_step != step - 1:
                    raise RuntimeError(
                        f"流水线 step 越序: task={task_id}, step={step}, "
                        f"last_step={last_step}"
                    )
            if str(active_config.get("engine", "pytorch") or "pytorch").lower() == "relay_middle":
                # relay_middle is endpoint-backed. The separately supervised
                # relay_mid_service owns the segment artifact; this scheduler
                # host does not need a local ModelHost/model loaded.
                relay_spec = self._normalize_relay_segment(data.get("relay_segment"))
                if not PIPELINE_RELAY_ENABLED or relay_spec is None:
                    layer_config_invalid = True
                    raise RuntimeError("relay_middle requires an enabled valid relay_segment")
                if relay_spec != active_config.get("relay_segment"):
                    layer_config_invalid = True
                    raise RuntimeError("relay segment does not match active layer config")
                return self._handle_layer_forward_via_relay(
                    relay_spec, data=data, task_id=task_id, step=step,
                    config_id=config_id, model_sha256=model_sha256,
                    model_type=model_type, received_chain_path=received_chain_path,
                )
            mgr = self._host
            if not mgr or not mgr.is_loaded:
                layer_config_invalid = True
                raise RuntimeError("模型未加载")
            loaded_config = getattr(getattr(mgr, "model", None), "config", None)
            actual_model_type = str(
                getattr(loaded_config, "model_type", "") or ""
            ).lower()
            if backend_id_for(mgr) != "pytorch":
                # ★ A1 / X 档（2026-09-24）：本节点不跑 pytorch 层段时的**保守** Relay 委托。
                #   仅当 ① 全局开关打开 且 ② 本步携带合法的 `relay_segment` 规格（middle 角色）
                #   才把本段交给远端 relay 段；否则**保持原行为** —— 直接拒绝，绝不静默降级。
                relay_spec = self._normalize_relay_segment(data.get("relay_segment"))
                if not PIPELINE_RELAY_ENABLED or relay_spec is None:
                    layer_config_invalid = True
                    raise RuntimeError(f"worker 引擎已变化: {backend_id_for(mgr)}")
                if relay_spec != active_config.get("relay_segment"):
                    layer_config_invalid = True
                    raise RuntimeError("relay segment does not match active layer config")
                return self._handle_layer_forward_via_relay(
                    relay_spec, data=data, task_id=task_id, step=step,
                    config_id=config_id, model_sha256=model_sha256,
                    model_type=model_type, received_chain_path=received_chain_path,
                )
            # 数据面经 `transport_port.serialize_tensor` ⇒ `serialize_tensor_fast`
            # （`TNR0` magic + numpy frombuffer），**不经过 torch**。此前在这里前置
            # `require_torch()`，会让无 torch 的边缘节点在走到这一段时直接失败。
            from transport_port import deserialize_tensor, serialize_tensor
            if actual_model_type != model_type:
                layer_config_invalid = True
                raise RuntimeError(
                    f"worker 模型架构已变化: actual={actual_model_type}, expected={model_type}"
                )
            if str(getattr(mgr, "active_model_id", "") or "") != str(
                active_config.get("model_id", "")
            ):
                layer_config_invalid = True
                raise RuntimeError(
                    f"worker 模型 ID 已变化: actual={getattr(mgr, 'active_model_id', '')}, "
                    f"expected={active_config.get('model_id', '')}"
                )
            if list(getattr(mgr, "layer_range", ()) or ()) != active_config.get("layer_range"):
                layer_config_invalid = True
                raise RuntimeError(
                    f"worker 层范围已变化: actual={getattr(mgr, 'layer_range', None)}, "
                    f"expected={active_config.get('layer_range')}"
                )

            # ---- 反序列化输入 ----
            input_ids = None
            hidden_states = None
            attention_mask = None
            position_ids = None

            if "input_ids" in data and data["input_ids"] is not None:
                input_ids = self._runtime_setting('torch').tensor(data["input_ids"], dtype=self._runtime_setting('torch').long)
                if input_ids.dim() == 1:
                    input_ids = input_ids.unsqueeze(0)  # (seq_len,) → (1, seq_len)

            if "hidden_states" in data and data["hidden_states"] is not None:
                hs_bytes = data["hidden_states"]
                if isinstance(hs_bytes, str):
                    import base64
                    hs_bytes = base64.b64decode(hs_bytes)
                elif isinstance(hs_bytes, list):
                    hs_bytes = bytes(hs_bytes)
                hidden_states = deserialize_tensor(hs_bytes)

            if "attention_mask" in data and data["attention_mask"] is not None:
                attention_mask = self._runtime_setting('torch').tensor(data["attention_mask"], dtype=self._runtime_setting('torch').long)
                if attention_mask.dim() == 1:
                    attention_mask = attention_mask.unsqueeze(0)

            if "position_ids" in data and data["position_ids"] is not None:
                position_ids = self._runtime_setting('torch').tensor(data["position_ids"], dtype=self._runtime_setting('torch').long)
                if position_ids.dim() == 1:
                    position_ids = position_ids.unsqueeze(0)

            # ---- KV Cache: 读取缓存的 past_key_values ----
            past_kv = None
            if use_kv_cache:
                with self._kv_cache_lock:
                    if task_id in self._kv_cache:
                        past_kv = self._kv_cache[task_id]
                if past_kv is None:
                    raise RuntimeError(
                        f"decode step {step} 缺少本地 KV cache: task={task_id}"
                    )
                if past_kv is not None:
                    # ★ #31 M4：`past_kv` 现在可能是 **Cache 对象**（hybrid 必须），也可能是旧 tuple
                    #   ⇒ 统一走 helper 取形状，并**跳过 hybrid 的 `None` 槽位**
                    #   （旧写法 `past_kv[0][0].shape` 在第 0 层是 None 时直接 TypeError）。
                    cached_layers, cached_seq_len = _kv_state_seq_len(past_kv, model_type)
                    logger.debug(
                        f"📦 KV cache 命中: task={task_id}, "
                        f"layers={cached_layers}, "
                        f"seq_len={cached_seq_len}"
                    )

            # ---- 执行前向传播 ----
            self._begin_local_pipeline_task(task_id)
            t_start = time.time()
            result = mgr.forward_layers(
                input_ids=input_ids,
                hidden_states=hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_kv,
                use_cache=True,  # 始终缓存 KV（prefill 构建，decode 更新）
            )
            with self._layer_config_lock:
                task_cancelled = task_id in self._local_pipeline_cancelled
            if task_cancelled:
                with self._kv_cache_lock:
                    self._kv_cache.pop(task_id, None)
                self._finish_local_pipeline_task(task_id)
                with self._layer_config_lock:
                    self._local_pipeline_cancelled.discard(task_id)
                logger.info("丢弃已取消任务的迟到计算结果: task=%s", task_id)
                return
            elapsed_ms = (time.time() - t_start) * 1000
            # ---- KV Cache: 存储更新后的 past_key_values ----
            # ★ #31 M4：**优先持有 `result["cache"]`** —— hybrid 的 tuple 会丢 recurrent state
            #   （`linear_attention` 层在 tuple 里是 `None` 占位）。旧代码只搬 tuple ⇒
            #   ① 后续 decode 的状态是错的，② 取形状时 `[0][0]` 直接 TypeError 硬崩。
            if result.get("cache") is not None or result.get("past_key_values"):
                state = _prefer_cache_state(result)
                with self._kv_cache_lock:
                    self._kv_cache[task_id] = state
                kv_layers, kv_seq_len = _kv_state_seq_len(state, model_type)
                logger.debug(
                    f"💾 KV cache 已更新: task={task_id}, "
                    f"layers={kv_layers}, seq_len={kv_seq_len}"
                )
            else:
                raise RuntimeError("分层前向未返回 KV cache")
            with self._layer_config_lock:
                self._local_pipeline_steps[task_id] = step

            # ---- 序列化输出 ----
            response = {
                "task_id": task_id,
                "node_id": self.get_effective_node_id(),
                "step": step,
                "assignment_id": assignment_id,
                "config_id": config_id,
                "model_sha256": model_sha256,
                "model_type": model_type,
                "chain_path": [
                    *[str(item) for item in received_chain_path],
                    self.get_effective_node_id(),
                ],
                "metrics": {
                    "time_ms": round(elapsed_ms, 1),
                    "kv_cache": use_kv_cache,  # 标记是否使用了 KV cache
                    "kv_seq_len": (
                        kv_seq_len
                        if result.get("past_key_values") else 0
                    ),
                    "memory_allocated_gb": (
                        round(self._runtime_setting('torch').cuda.memory_allocated() / (1024**3), 2)
                        if self._runtime_setting('torch').cuda.is_available() else 0
                    ),
                },
            }

            if "hidden_states" in result:
                # 中间节点：返回隐藏状态
                hs_cpu = result["hidden_states"].detach().cpu()
                response["hidden_states"] = serialize_tensor(hs_cpu)
                response["hidden_shape"] = list(hs_cpu.shape)
                logger.info(
                    f"✅ 层前向完成: task={task_id}, step={step}, "
                    f"output=hidden_states {list(hs_cpu.shape)}, "
                    f"kv={'on' if use_kv_cache else 'prefill'}, "
                    f"time={elapsed_ms:.0f}ms"
                )

            if "logits" in result:
                # 末节点：返回 logits
                logits_cpu = result["logits"].detach().cpu()
                response["logits"] = serialize_tensor(logits_cpu)
                response["logits_shape"] = list(logits_cpu.shape)
                logger.info(
                    f"✅ 层前向完成: task={task_id}, step={step}, "
                    f"output=logits {list(logits_cpu.shape)}, "
                    f"kv={'on' if use_kv_cache else 'prefill'}, "
                    f"time={elapsed_ms:.0f}ms"
                )

            # ---- 星状转发：经主节点把 hidden 交给下一层段 ----
            chain_next = data.get("chain_next")
            chain_remaining = data.get("chain_remaining", [])

            if chain_next and isinstance(chain_next, dict) and chain_next.get("node_id"):
                # 保留逻辑层段接力，但 worker 之间不建立数据连接；master
                # 校验相邻段后再转发，避免 legacy 回退路径破坏星状拓扑。
                import base64 as _b64
                _hs = response.get("hidden_states")
                chain_data = {
                    "task_id": task_id,
                    "step": step,
                    "assignment_id": assignment_id,
                    "config_id": config_id,
                    "model_sha256": model_sha256,
                    "model_type": model_type,
                    "chain_path": response["chain_path"],
                    "hidden_states": _b64.b64encode(_hs).decode("ascii") if _hs else None,
                    # Preserve the explicit relay raw-f32 contract across a
                    # CHAIN_FORWARD hop. Legacy tensor-fast frames leave this
                    # unset and keep their existing decoder path.
                    "hidden_wire_format": response.get("hidden_wire_format"),
                    "hidden_shape": response.get("hidden_shape"),
                    "chain_next": chain_remaining[0] if chain_remaining else None,
                    "chain_remaining": chain_remaining[1:] if len(chain_remaining) > 1 else [],
                    "use_kv_cache": use_kv_cache,
                    "temperature": data.get("temperature", 0.7),
                    "top_p": data.get("top_p", 0.9),
                }

                chain_data["_relay_to"] = chain_next["node_id"]
                sent = self._send_layer_result("master", task_id, result_data=chain_data)
                if not sent:
                    logger.error(
                        "星状层段转发未能回到主节点: task=%s target=%s",
                        task_id, chain_next["node_id"],
                    )
            else:
                # 末节点（或无链配置）：发送 LAYER_RESULT 回主节点
                self._send_layer_result("master", task_id, result_data=response)

        except Exception as e:
            with self._kv_cache_lock:
                self._kv_cache.pop(task_id, None)
            with self._layer_config_lock:
                self._local_pipeline_steps.pop(task_id, None)
                if layer_config_invalid:
                    self._active_layer_config = None
                    self._last_layer_config_ack_payload = None
            self._record_local_pipeline_participation(task_id, success=False)
            self._finish_local_pipeline_task(task_id)
            if layer_config_invalid:
                try:
                    self._host.model_loaded = False
                except Exception:
                    logger.debug("worker 层配置失效后更新 API 状态失败", exc_info=True)
            logger.error(f"层前向传播失败: task={task_id}, error={e}", exc_info=True)
            error_result = {
                "node_id": self.get_effective_node_id(),
                "step": step,
                "assignment_id": assignment_id,
                "config_id": config_id,
                "model_sha256": model_sha256,
                "model_type": model_type,
            }
            if layer_config_invalid:
                error_result["layer_config_invalid"] = True
            self._send_layer_result(
                "master",
                task_id,
                result_data=error_result,
                error=str(e),
            )
            self._send_chain_forward_ack(
                task_id=task_id,
                step=step,
                config_id=config_id,
                assignment_id=assignment_id,
                from_node_id=client_id,
                status="error",
                error=str(e),
            )


    @staticmethod
    def _normalize_relay_segment(raw: object) -> dict[str, object] | None:
        """★ A1 / X 档：校验并规范化 `LAYER_FORWARD` 里**可选**的 `relay_segment` 规格。

        返回 `None` 表示"不适用"（缺失 / 非法 / 超出 X 档范围）—— 调用方据此走**既有**拒绝路径。
        严格到"多一个未知键就整条不认"，避免"半懂"的规格被误用。

        **角色（当前 hidden 输入执行路径）**：`middle` / `tail` 二选一。`head` 需要
        token → hidden 的独立上游协议，当前 scheduler/PeerClient 不接收该输入，因此在
        配置解析层直接拒绝，避免 readiness 看似成功后首帧才失败。`middle` =
        hidden → hidden，与本节点"中间节点返回 hidden_states"的既有契约对齐；`tail` =
        hidden → token，末段返回 token。`head` 的 token → hidden 输入协议仍未接线。
        角色的**区间语义**（tail 须到 total_layers 止）在**切分校验**处检查 ——
        这里只保证"角色合法 + 层区间存在且非空"。
        **★ Y 档第二条：层区间必填**。relay 段必须声明它**认领**的层区间
        `[layer_start, layer_end)` —— 这是修复「relay 段的层范围从未进入调度 ⇒ 主节点跑满
        全部层、该段在"已过全部层"的 hidden 上重算」这一根因的**契约前提**：没有区间就无从
        把该段的层从主节点层范围里扣除，也无从校验切分是否恰好覆盖。缺区间 ⇒ 整条不认
        （fail-closed，绝不静默降级成"只带 n_embd"的半懂规格）。
        """
        if not isinstance(raw, dict):
            return None
        if set(raw) - {"role", "host", "port", "n_embd", "timeout",
                       "layer_start", "layer_end"}:
            return None
        role = str(raw.get("role", "")).strip().lower()
        if role not in {"middle", "tail"}:
            return None     # head 需要 token→hidden，上游协议尚未接线
        host = str(raw.get("host", "")).strip()
        if not is_loopback_host(host):
            return None     # 跨机必须走本地 SSH 隧道端点（Relay 传输层自身也强制 loopback）
        try:
            port = int(raw.get("port", 0))
            n_embd = int(raw.get("n_embd", 0))
            timeout = float(raw.get("timeout", 60.0))
            layer_start = int(raw.get("layer_start", -1))
            layer_end = int(raw.get("layer_end", -1))
        except (TypeError, ValueError):
            return None
        if not (0 < port <= 65535) or n_embd < 1 or not (0.0 < timeout <= 3600.0):
            return None
        if layer_start < 0 or layer_end <= layer_start:
            return None     # 层区间必填且非空
        return {"role": role, "host": host, "port": port, "n_embd": n_embd,
                "timeout": timeout, "layer_start": layer_start, "layer_end": layer_end}

    def _handle_layer_forward_via_relay(self, spec: dict[str, object], *, data: dict,
                                        task_id: str, step: int, config_id: str,
                                        model_sha256: str, model_type: str,
                                        received_chain_path: list) -> None:
        """★ A1 / X 档：把本步委托给远端 **middle** relay 段（hidden → hidden），再回传主节点。

        与中间节点语义对齐（吃 hidden、吐 hidden）；KV 由远端段自管，本节点**不碰**本地
        `_kv_cache`（所以本分支在 KV 检查之前就 return，见 `_handle_layer_forward_locked`）。

        范围：**只做 2 段拓扑** —— 请求里带 `chain_next` 时显式拒绝（>2 段属 Y 档）。
        失败：抛 :class:`RelaySegmentError` ⇒ 被外层 `except` 捕获 ⇒ 经既有
        `_send_layer_result(..., error=str(e))` 回传**具名**错误（形如
        `relay_segment_failed:runner_failed#middle@127.0.0.1:50183`），**绝不**静默产出空 hidden
        （那会退化成"模型算错"，无从区分）。
        """
        if data.get("chain_next"):
            raise RuntimeError("relay 段委托不支持链式转发（>2 段拓扑属 Y 档）")

        width = int(spec["n_embd"])
        raw_hidden = data.get("hidden_states")
        if isinstance(raw_hidden, str):
            import base64
            hidden_bytes = base64.b64decode(raw_hidden)
        elif isinstance(raw_hidden, (bytes, bytearray)):
            hidden_bytes = bytes(raw_hidden)
        else:
            raise RuntimeError("relay 段委托需要 hidden_states（token 输入不在 X 档范围）")
        if not hidden_bytes or len(hidden_bytes) % (width * 4):
            raise RuntimeError("relay 段委托的 hidden 长度与 n_embd 不匹配（需 f32 且整除）")
        n_tokens = len(hidden_bytes) // (width * 4)
        request_deadline_ms = int(data.get("request_deadline_ms", 0) or 0)
        request_deadline = (
            RequestDeadline.from_epoch_ms(request_deadline_ms)
            if request_deadline_ms else None
        )
        if request_deadline is not None and request_deadline.expired():
            raise RelaySegmentError(
                REQUEST_DEADLINE_EXCEEDED,
                role=str(spec.get("role", "middle")),
                endpoint=f'{spec["host"]}:{spec["port"]}',
            )
        deadline_monotonic = (
            request_deadline.expires_at_monotonic
            if request_deadline is not None else None
        )
        hidden_shape = data.get("hidden_shape")
        if isinstance(hidden_shape, list):
            if not hidden_shape or any(
                isinstance(size, bool) or not isinstance(size, int) or size <= 0
                for size in hidden_shape
            ) or hidden_shape[-1] != width:
                raise RuntimeError("relay hidden_shape must end in n_embd and contain positive integers")
            shape_items = 1
            for size in hidden_shape:
                shape_items *= size
            if shape_items != n_tokens * width:
                raise RuntimeError("relay hidden_shape does not match raw f32 payload")
        else:
            hidden_shape = [n_tokens, width]

        # ★ P3：显式给了 seq_ids / positions 就走 `HIDDEN_SEQ`（多序列必须逐 token 绑定）。
        seq_ids = data.get("seq_ids")
        positions = data.get("positions")
        seq_meta = None
        if seq_ids is not None or positions is not None:
            if not (isinstance(seq_ids, list) and isinstance(positions, list)
                    and len(seq_ids) == n_tokens and len(positions) == n_tokens):
                raise RuntimeError("relay 段委托的 seq_ids/positions 必须与 token 数等长")
            n_seq_id = data.get("n_seq_id")
            seq_meta = {
                "n_seq_id": [int(v) for v in (n_seq_id or [1] * n_tokens)],
                "seq_ids": [int(v) for v in seq_ids],
                "positions": [int(v) for v in positions],
            }

        self._begin_local_pipeline_task(task_id)   # 与既有执行路径对齐（保证 begin/finish 平衡）
        started = time.time()
        client = self._relay_segment_client_for_task(task_id, spec, n_embd=width)
        try:
            role = str(spec.get("role", "middle"))
            if role == "tail":
                if seq_meta is None:
                    outcome = client.forward_hidden_to_token(
                        hidden_bytes,
                        n_tokens=n_tokens,
                        deadline_monotonic=deadline_monotonic,
                    )
                else:
                    outcome = client.forward_hidden_to_token(
                        hidden_bytes,
                        n_tokens=n_tokens,
                        seq_meta=seq_meta,
                        deadline_monotonic=deadline_monotonic,
                    )
            elif role == "middle":
                if seq_meta is None:
                    outcome = client.forward_hidden(
                        hidden_bytes,
                        n_tokens=n_tokens,
                        deadline_monotonic=deadline_monotonic,
                    )
                else:
                    outcome = client.forward_hidden(
                        hidden_bytes,
                        n_tokens=n_tokens,
                        seq_meta=seq_meta,
                        deadline_monotonic=deadline_monotonic,
                    )
            else:
                raise RelaySegmentError("relay_role_unsupported", role=role,
                                        detail="unsupported scheduler relay role")
        except Exception:
            self._close_relay_segment_client(task_id)
            raise
        elapsed_ms = (time.time() - started) * 1000

        if not outcome.ok:
            self._close_relay_segment_client(task_id, grace_timeout=0.0)
            # ★ 这一层的语义是「**调度层的段委托**失败」⇒ 消息要能让主节点直接落进
            #   `_fallback_reason`（形如 `pipeline_error_result: ... relay_segment_failed:runner_failed#...`）。
            #   `detail` 只进可读消息；`RelaySegmentError.code` 仍是白名单码（可供线上/日志使用）。
            _code = outcome.error or "relay_internal_error"
            raise RelaySegmentError(_code, role=str(spec.get("role", "middle")), endpoint=outcome.endpoint,
                                    detail=f"relay_segment_failed:{_code}")
        if request_deadline is not None and request_deadline.expired():
            self._close_relay_segment_client(task_id, grace_timeout=0.0)
            raise RelaySegmentError(
                REQUEST_DEADLINE_EXCEEDED,
                role=str(spec.get("role", "middle")),
                endpoint=outcome.endpoint,
            )

        layer_lock = getattr(self, "_layer_config_lock", None)
        steps = getattr(self, "_local_pipeline_steps", None)
        if steps is None:
            steps = {}
            self._local_pipeline_steps = steps
        if layer_lock is None:
            steps[task_id] = step
        else:
            with layer_lock:
                steps[task_id] = step

        response = {
            "task_id": task_id,
            "node_id": self.get_effective_node_id(),
            "step": step,
            "assignment_id": str(data.get("assignment_id", "") or ""),
            "config_id": config_id,
            "model_sha256": model_sha256,
            "model_type": model_type,
            "chain_path": [*[str(item) for item in received_chain_path],
                           self.get_effective_node_id()],
            "metrics": {
                "time_ms": round(elapsed_ms, 1),
                "kv_cache": False,       # KV 在远端段，本节点没有本地 KV
                "kv_seq_len": 0,
                "relay_executed": True,
                # relay_segment / relay_frames / relay_tokens / relay_payload_bytes / relay_error
                **outcome.to_metrics(),
            },
        }
        if role == "tail":
            if outcome.token is None:
                raise RelaySegmentError("relay_internal_error", role=role,
                                        detail="relay tail did not return token")
            response["token"] = int(outcome.token)
        else:
            response.update({
                "hidden_states": bytes(outcome.hidden),
                "hidden_wire_format": RELAY_HIDDEN_WIRE_FORMAT,
                "hidden_shape": hidden_shape,
            })
        logger.info(
            f"🔁 relay 段委托完成: task={task_id}, step={step}, "
            f"段={outcome.role}@{spec['host']}:{spec['port']}, tokens={n_tokens}, "
            f"time={elapsed_ms:.0f}ms"
        )
        self._send_layer_result("master", task_id, result_data=response)

    def _handle_chain_forward(self, client_id: str, msg: dict) -> None:
        """
        从节点：收到 master 的 CHAIN_FORWARD → 执行本节点层前向 → 回传 master。

        CHAIN_FORWARD 的消息结构与 LAYER_FORWARD 一致（均为 hidden_states + chain 信息），
        直接委托 _handle_layer_forward 处理（其内部根据 chain_next 决定下一步动作）。
        """
        data = msg.get("data", {})
        task_id = data.get("task_id", "")
        step = data.get("step", -1)
        if client_id != "master":
            logger.warning(
                "丢弃非 master 来源的逻辑段转发: source=%s task=%s",
                client_id, task_id or "-",
            )
            return
        logger.info(f"🔗 收到 master 层段转发: task={task_id or '?'}")
        self._send_chain_forward_ack(
            task_id=task_id,
            step=step,
            config_id=str(data.get("config_id", "")),
            assignment_id=str(data.get("assignment_id", "") or ""),
            from_node_id=(
                str(data.get("_chain_predecessor", "") or client_id)
                if client_id == "master" else client_id
            ),
            status="received",
        )
        self._handle_layer_forward(client_id, msg)


    def _send_layer_result(self, client_id: str, task_id: str,
                           result_data: dict = None, error: str = None) -> bool:
        """从节点 → 主节点：发送层前向传播结果"""
        if not self._tcp_client or not self._tcp_client._running:
            logger.error("TCP 客户端未连接，无法发送层前向结果")
            self._record_local_pipeline_participation(task_id, success=False)
            self._finish_local_pipeline_task(task_id)
            return False

        from transport_port import MessageType
        import base64

        payload = result_data or {}
        payload["task_id"] = task_id
        if error:
            payload["error"] = error

        # 将 bytes 字段转为 base64 字符串（JSON 兼容）
        safe_payload = {}
        for k, v in payload.items():
            if isinstance(v, bytes):
                safe_payload[k] = base64.b64encode(v).decode("ascii")
            else:
                safe_payload[k] = v

        try:
            self._tcp_client.send_data(safe_payload, MessageType.LAYER_RESULT)
            return True
        except Exception as e:
            logger.error(f"发送层前向结果失败: {e}")
            self._record_local_pipeline_participation(task_id, success=False)
            self._finish_local_pipeline_task(task_id)
            try:
                self._tcp_client.disconnect()
            except Exception:
                pass
            return False


    def _send_chain_forward_ack(self, task_id: str, step: int,
                                config_id: str = "",
                                assignment_id: str = "",
                                from_node_id: str = "",
                                target_node_id: str = "",
                                status: str = "received",
                                error: str = "") -> bool:
        """从节点 → 主节点：发送链式转发接收/错误 ACK。"""
        if not task_id:
            return False
        if not self._tcp_client or not self._tcp_client._running:
            logger.error("TCP 客户端未连接，无法发送链式转发 ACK")
            return False

        from transport_port import MessageType

        payload = {
            "task_id": task_id,
            "step": step,
            "config_id": config_id,
            "assignment_id": assignment_id,
            "node_id": self.get_effective_node_id(),
            "from_node_id": from_node_id,
            "status": status,
        }
        if target_node_id:
            payload["target_node_id"] = target_node_id
        if error:
            payload["error"] = error
        try:
            self._tcp_client.send_data(payload, MessageType.CHAIN_FORWARD_ACK)
            return True
        except Exception as e:
            logger.error(f"发送链式转发 ACK 失败: {e}")
            try:
                self._tcp_client.disconnect()
            except Exception:
                pass
            return False


    @staticmethod
    def _extract_relay_metrics(metrics: object) -> dict:
        """★ A1 / X 档：从末节点回传的 `metrics` 里取出 relay 五字段（没走 relay 时返回空 dict）。

        只挑 `relay_segment` / `relay_frames` / `relay_tokens` / `relay_payload_bytes` /
        `relay_error` 这五个键，且要求 `relay_segment` 非空 —— worker 侧只有**真走了 relay 分支**
        才会写它们（`_handle_layer_forward_via_relay` 里的 `**outcome.to_metrics()`）⇒
        "键存在且非空"就等价于"这一步确实委托出去了"，普通 pytorch 路径不会被误标成 relay。
        """

        if not isinstance(metrics, dict):
            return {}
        if not metrics.get("relay_segment"):
            return {}
        return {
            "relay_segment": metrics.get("relay_segment"),
            "relay_frames": metrics.get("relay_frames"),
            "relay_tokens": metrics.get("relay_tokens"),
            "relay_payload_bytes": metrics.get("relay_payload_bytes"),
            "relay_error": metrics.get("relay_error"),
        }

    def _handle_layer_result(self, client_id: str, msg: dict) -> None:
        """
        主节点：收到从节点的 LAYER_RESULT → 存储到流水线结果字典，
        唤醒正在等待的 run_pipeline() 主循环。

        特殊处理: 如果 data 中包含 _relay_to 字段，说明从节点请求
        master 将 hidden_states 转发给下一逻辑段，此时
        主节点转发后直接返回，不存储结果也不唤醒 run_pipeline()。
        """
        data = msg.get("data", {})
        task_id = data.get("task_id", "")
        node_id = str(data.get("node_id", client_id))
        try:
            step = int(data.get("step", -1))
        except (TypeError, ValueError):
            step = -1
        config_id = str(data.get("config_id", ""))
        assignment_id = str(data.get("assignment_id", "") or "")

        if node_id != client_id:
            logger.warning(
                "丢弃来源不一致的层结果: connection=%s payload=%s task=%s",
                client_id, node_id, task_id or "-",
            )
            return

        with self._pipeline_lock:
            is_active = task_id in self._pipeline_active_tasks
            contract = dict(self._pipeline_task_contracts.get(task_id, {}))
        if not is_active:
            logger.warning(
                "丢弃非活跃流水线任务结果: task=%s node=%s",
                task_id or "-", node_id,
            )
            return
        worker_ids = list(contract.get("worker_ids", []))
        assignment_ids = dict(contract.get("assignment_ids", {}))
        expected_nodes = set(worker_ids)
        identities_complete = (
            set(assignment_ids) == expected_nodes
            and all(str(assignment_ids.get(worker_id, "") or "")
                    for worker_id in worker_ids)
        )
        if contract.get("execution_protocol") != "legacy_layer_v1":
            logger.warning(
                "discarding layer result for non-legacy task: "
                "task=%s protocol=%s",
                task_id, contract.get("execution_protocol", "missing"),
            )
            return
        if not identities_complete:
            self._set_pipeline_result_error(
                task_id,
                node_id,
                "流水线执行契约 assignment 身份不完整",
                step,
            )
            return
        if (node_id not in expected_nodes
                or step != contract.get("current_step")
                or not assignment_id
                or assignment_id != assignment_ids.get(node_id, "")
                or config_id != contract.get("config_id")
                or data.get("model_sha256") != contract.get("model_sha256")
                or data.get("model_type") != contract.get("model_type")):
            logger.warning(
                "丢弃不符合执行契约的层结果: task=%s node=%s step=%s config=%s",
                task_id, node_id, step, config_id,
            )
            return

        # ★ 星状中转请求：worker 只声明下一逻辑段，由 master 校验并转发。
        relay_target = data.get("_relay_to")
        if relay_target:
            source_index = worker_ids.index(node_id)
            expected_target = (
                worker_ids[source_index + 1]
                if source_index + 1 < len(worker_ids) else ""
            )
            if relay_target != expected_target:
                error = (
                    f"非相邻中转目标: source={node_id}, "
                    f"target={relay_target}, expected={expected_target or '-'}"
                )
                logger.error(
                    "终止非法链路中转: task=%s %s",
                    task_id, error,
                )
                self._set_pipeline_result_error(
                    task_id, node_id, error, step
                )
                return
            logger.info(
                f"🔄 主节点中转: {node_id} → {relay_target} "
                f"(task={task_id}, step={data.get('step', '?')})"
            )
            try:
                # 构建转发 payload（去掉 _relay_to 内部标记）
                relay_data = {
                    k: v for k, v in data.items()
                    if k != "_relay_to"
                }
                target_assignment_id = str(
                    assignment_ids.get(relay_target, "") or ""
                )
                if not target_assignment_id:
                    self._set_pipeline_result_error(
                        task_id,
                        node_id,
                        f"目标段 assignment 身份缺失: {relay_target}",
                        step,
                    )
                    return
                relay_data["assignment_id"] = target_assignment_id
                relay_data["_chain_predecessor"] = node_id
                from transport_port import MessageType
                self._send_to_worker(relay_target, relay_data,
                                     MessageType.CHAIN_FORWARD)
                self._handle_chain_forward_ack(
                    node_id,
                    {
                        "data": {
                            "task_id": task_id,
                            "step": step,
                            "config_id": config_id,
                            "assignment_id": assignment_id,
                            "node_id": node_id,
                            "target_node_id": relay_target,
                            "status": "sent",
                        }
                    },
                )
                logger.info(f"✅ 中转成功: master → {relay_target}")
                return  # 不存储结果，不唤醒 run_pipeline，链继续
            except Exception as e:
                logger.error(
                    f"❌ 主节点中转失败 → {relay_target}: {e}，"
                    f"触发全模型回退"
                )
                # 中转失败 → 存储错误，唤醒 run_pipeline
                self._set_pipeline_result_error(
                    task_id,
                    relay_target,
                    f"主节点中转到 {relay_target} 失败: {e}",
                    data.get("step", -1),
                )
                return

        if data.get("layer_config_invalid"):
            self._invalidate_worker_layer_ready(
                node_id, config_id, str(data.get("error", "") or "worker 层配置失效")
            )

        if not data.get("error") and node_id != contract.get("last_node_id"):
            logger.warning(
                "丢弃非末节点成功结果: task=%s node=%s expected=%s",
                task_id, node_id, contract.get("last_node_id"),
            )
            return
        if (not data.get("error")
                and data.get("chain_path") != worker_ids):
            error = (
                f"链路路径不完整: path={data.get('chain_path')}, "
                f"expected={worker_ids}"
            )
            logger.error(
                "终止链路路径不完整的任务: task=%s path=%s expected=%s",
                task_id, data.get("chain_path"), worker_ids,
            )
            self._set_pipeline_result_error(
                task_id, node_id, error, step
            )
            return

        logger.info(
            f"📥 收到层前向结果: task={task_id}, node={node_id}, "
            f"step={data.get('step', '?')}, "
            f"error={data.get('error', 'none')}"
        )

        # 解码 base64 bytes 字段
        import base64
        decoded = {}
        for k, v in data.items():
            if isinstance(v, str) and k in ("hidden_states", "logits"):
                try:
                    decoded[k] = base64.b64decode(v)
                except Exception:
                    decoded[k] = v  # 保持原样
            else:
                decoded[k] = v

        # ★ A1 / X 档（Y 档第一条）：把末节点回传的 relay 指标读出来存到主节点，供
        #   `_get_pipeline_status()` 展示。X 档只支持 **2 段**，relay 段执行完**直接**
        #   `_send_layer_result("master", ...)`（`_handle_layer_forward_via_relay` 明确拒绝
        #   `chain_next`）⇒ 指标本来就在这一帧的 `metrics` 里，**不需要**跨节点聚合。
        #   （真跨节点聚合要等 >2 段拓扑，那属 Y 档的另一条。）
        relay_metrics = self._extract_relay_metrics(decoded.get("metrics"))

        key = f"{task_id}:{node_id}"
        with self._pipeline_lock:
            self._pipeline_results[key] = decoded
            if relay_metrics:
                self._last_relay_metrics = {
                    "node_id": node_id,
                    "task_id": task_id,
                    "step": decoded.get("step"),
                    **relay_metrics,
                }
            if key in self._pipeline_events:
                self._pipeline_events[key].set()


    def _get_pipeline_readiness(self) -> dict:
        """返回流水线 worker 的真实就绪状态和首个阻塞原因。"""
        if not self._tcp_server or not self._tcp_server._running:
            return {
                "ready": False,
                "reason_code": "tcp_server_not_running",
                "reason": "主节点 TCP 服务未运行",
                "workers": [],
            }

        if self._pipeline_recovery_pending:
            return {
                "ready": False,
                "reason_code": self._pipeline_recovery_failure
                or "pipeline_recovery_pending",
                "reason": (
                    "pipeline configuration recovery is pending; waiting for "
                    "authoritative republish"
                ),
                "workers": [],
            }

        with self._layer_config_lock:
            transaction = self._pipeline_load_transaction
            transaction_phase = str(
                transaction.get("phase", "") if transaction else ""
            )
        if transaction_phase in {
            "committing_local", "committing", "committing_global",
        }:
            return {
                "ready": False,
                "reason_code": "pipeline_model_transaction_committing",
                "reason": "pipeline model transaction is crossing the commit fence",
                "workers": [],
            }

        assignments = self.get_layer_assignments()
        master_ids = {"master", self.get_effective_node_id()}
        # ★ A1 / X 档（Y 档第二条缺口 3）：relay 段节点**也**是流水线成员，尽管它 `layers_count=0`
        #   （缺口 1 把它摘成不占层）。只按"层数 > 0"过滤会把它排除 ⇒ 若它是唯一 worker，
        #   `pipeline_nodes` 直接为空 ⇒ `no_pipeline_workers`（实测日志
        #   `reason=未分配任何 PC 从节点参与模型层计算`）。
        relay_for_worker = getattr(self, "_relay_segment_for_worker", None)

        def _is_relay_member(node_id: str) -> bool:
            """该 node_id 是否被配成「由远端 relay 段代跑本段」。"""
            return callable(relay_for_worker) and relay_for_worker(node_id) is not None

        pipeline_nodes = [
            a for a in assignments.get("assignments", [])
            if a.get("node_id") not in master_ids
            and (a.get("layers_count", 1) > 0 or _is_relay_member(a.get("node_id", "")))
        ]
        if not pipeline_nodes:
            return {
                "ready": False,
                "reason_code": "no_pipeline_workers",
                "reason": "未分配任何 PC 从节点参与模型层计算",
                "workers": [],
            }

        with self._nodes_lock:
            nodes_snapshot = dict(self.nodes)
        with self._layer_config_lock:
            expected_configs = self._worker_assignments.expected_configs()
            ack_snapshot = self._worker_assignments.acknowledgements()
            ready_nodes = self._effective_layer_config_pushed_nodes()
        get_client_ids = getattr(self._tcp_server, "get_client_ids", None)
        connected = set(
            get_client_ids()
            if callable(get_client_ids)
            else getattr(self._tcp_server, "clients", {}).keys()
        )

        first_failure = None
        worker_status = []
        now = time.time()
        for assignment in pipeline_nodes:
            node_id = assignment["node_id"]
            node_info = nodes_snapshot.get(node_id)
            online = bool(node_info and node_info.is_available())
            tcp_connected = node_id in connected
            heartbeat_age = (
                max(0.0, now - node_info.last_heartbeat)
                if node_info and node_info.last_heartbeat else None
            )
            expected = expected_configs.get(node_id, {})
            ack = ack_snapshot.get(node_id, {})
            expected_range = [
                assignment.get("start_layer", 0),
                assignment.get("end_layer", 0),
            ]
            relay_for_assignment = getattr(self, "_relay_segment_for_worker", None)
            is_relay_worker = (
                callable(relay_for_assignment)
                and relay_for_assignment(node_id) is not None
            )
            is_stage_offer_worker = assignment.get("execution") == "stage_offer_v3"
            if is_stage_offer_worker:
                layer_ready, stage_reason = self._stage_offer_assignment_ready(
                    node_id, assignment,
                )
                ack = {
                    "status": "ready" if layer_ready else "error",
                    "error": "" if layer_ready else stage_reason,
                }
            elif is_relay_worker:
                # ★ A1 / X 档（Y 档第二条缺口 3）：relay 段节点**不需要**加载本地层
                #   （`:2255` 分支明说段工件由外边监督的 relay_mid_service 持有 ——
                #   "this scheduler host does not need a local ModelHost/model loaded"）
                #   ⇒ 就绪判据退化为「在线 + TCP 连着」。段**不可达**会在**运行时**具名失败
                #   （`relay_transport_error` ⇒ 具名回退 ⇒ `_fallback_reason` 带得出原因），
                #   不靠这里探活 —— 主节点也没法直接探远端 loopback 上的段服务。
                layer_ready = online and tcp_connected
            else:
                layer_ready = (
                    node_id in ready_nodes
                    and ack.get("assignment_id") == expected.get("assignment_id")
                    and ack.get("config_id") == expected.get("config_id")
                    and ack.get("layer_range") == expected_range
                    and ack.get("model_sha256") == expected.get("model_sha256")
                    and ack.get("model_type") == expected.get("model_type")
                    and ack.get("engine") == expected.get("engine", "pytorch")
                )
            layer_status = "ready" if layer_ready else (
                "error" if ack.get("status") == "error" else
                "loading" if expected else "not_configured"
            )
            error = str(ack.get("error", ""))

            # ★ 2026-10-03：v3 层段 worker 的 legacy ACK 故意是 error（见
            #   `_handle_layer_config_locked` 的同名分流）⇒ 它**不参与** legacy 就绪判据。
            #   否则会先命中 `not layer_ready and error` 分支，报成「模型同步或层加载
            #   失败」并卡住整个请求 —— 而它其实是通过 v3 stage offer 就绪的。
            #
            # ★ 2026-10-05（DIST-2）：但它**不能因此跳过整个 failure 判定**。此前这里
            #   是裸 `continue`，于是下方 `worker_stage_offer_not_ready` 分支**永远
            #   不可达** ⇒ stage worker 不健康时 readiness 仍可能报 `ready=True`。
            #   现在改为：基础存活检查（未注册/离线/TCP 断/心跳过期）照常参与；
            #   legacy 专属的 `worker_layer_load_failed` 只对非 stage worker 生效；
            #   stage worker 自己的就绪判据用 `_stage_offer_assignment_ready` 的结果。

            failure = None
            if node_info is None:
                failure = ("worker_not_registered", f"从节点 {node_id} 未注册")
            elif not online:
                failure = ("worker_offline", f"从节点 {node_id} 已离线")
            elif not tcp_connected:
                failure = ("worker_tcp_disconnected", f"从节点 {node_id} TCP 已断开")
            elif heartbeat_age is None or heartbeat_age > _WORKER_HEARTBEAT_MAX_AGE:
                age_text = "未知" if heartbeat_age is None else f"{heartbeat_age:.1f}s"
                failure = (
                    "worker_heartbeat_stale",
                    f"从节点 {node_id} 心跳已过期 ({age_text})",
                )
            elif not layer_ready and is_stage_offer_worker:
                failure = (
                    "worker_stage_offer_not_ready",
                    f"从节点 {node_id} v3 layer_forward 未就绪: {stage_reason or error}",
                )
            elif not layer_ready and error:
                failure = (
                    "worker_layer_load_failed",
                    f"从节点 {node_id} 模型同步或层加载失败: {error}",
                )
            elif not layer_ready and expected:
                failure = (
                    "worker_layer_loading",
                    f"从节点 {node_id} 正在同步同款 PyTorch 模型或加载分配层",
                )
            elif not layer_ready:
                failure = (
                    "worker_layer_not_configured",
                    f"从节点 {node_id} 尚未收到模型分层配置",
                )

            if first_failure is None and failure is not None:
                first_failure = failure
            worker_status.append({
                "node_id": node_id,
                "online": online,
                "tcp_connected": tcp_connected,
                "heartbeat_age_seconds": (
                    round(heartbeat_age, 1) if heartbeat_age is not None else None
                ),
                "layer_ready": layer_ready,
                "layer_status": layer_status,
                "layer_error": error,
                "config_id": expected.get("config_id", ""),
                "model_id": expected.get("model_id", ""),
                "layer_range": expected_range,
                "execution": assignment.get("execution", "legacy_layer_config"),
            })

        if first_failure is None:
            return {
                "ready": True,
                "reason_code": "ready",
                "reason": "所有 PC 从节点已确认同款 PyTorch 模型和分配层",
                "workers": worker_status,
            }
        return {
            "ready": False,
            "reason_code": first_failure[0],
            "reason": first_failure[1],
            "workers": worker_status,
        }


    def _all_pipeline_nodes_ready(self) -> bool:
        """检查所有流水线节点是否在线并已确认模型层加载完成。"""
        readiness = self._get_pipeline_readiness()
        if readiness["ready"]:
            logger.info(
                "✅ 所有流水线节点就绪: %s",
                [worker["node_id"] for worker in readiness["workers"]],
            )
            return True
        logger.warning("流水线未就绪: %s", readiness["reason"])
        return False


    def _connected_client_ids(self) -> set[str]:
        """Return node ids of the TCP clients currently connected to this master."""
        server = getattr(self, "_tcp_server", None)
        if not server or not getattr(server, "_running", False):
            return set()
        get_client_ids = getattr(server, "get_client_ids", None)
        return set(
            get_client_ids()
            if callable(get_client_ids)
            else getattr(server, "clients", {}).keys()
        )


    def _distributable_worker_ids(self) -> list[str]:
        """Workers this master may hand work to, PC **or** admitted Android stage node.

        ★ 2026-10-01 真机实测（Route A 阶段 1）：原先强制分布式路径只看
        `_connected_pc_worker_ids()`，而它按 `node_type == "pc"` 过滤 ⇒ 一个已经
        声明并**被准入**的 Android `layer_forward` 节点仍被挡在外面，
        `force_distributed_assignment` 于是直接 `pipeline_distributed_workers_unavailable`
        → 回退整模（实测 `fallback_reason` 正是"没有在线 PC 从节点可参与强制分布式分层"）。
        这里把 stage 节点一并计入；准入判据**复用** `_task_worker_layer_stage_ids()`
        （单一判据来源），不再另开一套。
        """
        worker_ids = self._connected_pc_worker_ids()
        stage_ids = self._task_worker_layer_stage_ids(self._connected_client_ids())
        if not stage_ids:
            return worker_ids
        return sorted({*worker_ids, *stage_ids})


    def _connected_pc_worker_ids(self) -> list[str]:
        """Return online PC clients that can receive an authoritative config."""
        connected = self._connected_client_ids()
        local_node_id = self.get_effective_node_id()
        with self._nodes_lock:
            return sorted(
                node_id for node_id, node in self.nodes.items()
                if node_id in connected
                and node_id != local_node_id
                and getattr(node, "node_type", "pc") == "pc"
                and node.is_available()
                and not self._node_is_island_gateway(node.device_info)
            )


    def _has_active_distributed_pipeline_plan(self) -> bool:
        """Return whether the current generation actually uses two nodes."""
        with self._layer_config_lock:
            plan = self._active_pipeline_capacity_plan
            assignments = plan.get("assignments", []) if isinstance(plan, dict) else []
            return len(assignments) >= 2


    def _synchronize_pipeline_workers_for_request(
        self,
        timeout: float = PIPELINE_MODEL_SYNC_TIMEOUT,
        *,
        force_distributed_assignment: bool = False,
    ) -> dict:
        """Synchronize worker model segments before falling back to the master."""
        readiness = self._get_pipeline_readiness()
        if readiness.get("ready") and (
            not force_distributed_assignment
            or self._has_active_distributed_pipeline_plan()
        ):
            return readiness

        worker_ids = self._distributable_worker_ids()
        if not worker_ids:
            if force_distributed_assignment:
                return {
                    **readiness,
                    "ready": False,
                    "reason_code": "pipeline_distributed_workers_unavailable",
                    "reason": "没有在线 PC 从节点可参与强制分布式分层",
                }
            return readiness

        recoverable = {
            "no_pipeline_workers",
            "worker_layer_not_configured",
            "worker_layer_loading",
            "worker_layer_load_failed",
            "pipeline_recovery_pending",
        }
        # ★ 2026-10-05（DIST-2）：把「不可恢复的就绪原因」从 force 短路里剥离出来。
        #
        #   此前 `not force_distributed_assignment and reason not in recoverable`
        #   的组合意味着：**force 路径下连不可恢复的原因也会等满
        #   `PIPELINE_MODEL_SYNC_TIMEOUT`（60s）** —— 包括 `worker_offline` /
        #   `worker_tcp_disconnected` / `worker_heartbeat_stale`。这些状态等下去不会
        #   变好：对端要么已经没了，要么需要重新连上，而重连本身会触发一次权威重发
        #   （见 `_handle_task_worker_message` 的 hello 分支）。在这里死等只是把
        #   「确定的失败」延迟成「超时」。
        #
        #   DIST-2 明确要求「禁止等待多个互相独立的超时后才 fallback」，故这几种
        #   原因在 force 路径下同样快速具名返回。
        #
        #   只收编最无歧义的三种：`worker_not_registered` 可能是「刚注册、hello
        #   还没到位」，仍有等待价值，保持原行为。
        unrecoverable = {
            "worker_offline",
            "worker_tcp_disconnected",
            "worker_heartbeat_stale",
            "pipeline_lifecycle_persist_failed",
            "pipeline_recovery_state_unavailable",
            "pipeline_recovery_state_invalid",
            "pipeline_recovery_model_identity_missing",
            "pipeline_recovery_model_unavailable",
            "pipeline_recovery_model_restore_failed",
            "pipeline_recovery_model_mismatch",
            "pipeline_recovery_model_digest_mismatch",
        }
        reason_code = str(readiness.get("reason_code", "") or "")
        if reason_code in unrecoverable:
            logger.info(
                "分布式请求快速失败（不可恢复的就绪原因）: workers=%s reason_code=%s",
                worker_ids, reason_code,
            )
            return readiness
        if not force_distributed_assignment and reason_code not in recoverable:
            return readiness

        # An existing loading generation should finish without being superseded.
        # Other recoverable states need a fresh authoritative generation.
        if (
            force_distributed_assignment
            or readiness.get("reason_code") != "worker_layer_loading"
        ):
            logger.info(
                "分布式请求触发主节点权威模型同步: workers=%s reason=%s",
                worker_ids,
                readiness.get("reason", ""),
            )
            if force_distributed_assignment or self._pipeline_recovery_pending:
                self.request_authoritative_layer_sync(require_distributed=True)
            else:
                self.request_authoritative_layer_sync()

        deadline = time.monotonic() + max(0.0, float(timeout))
        while True:
            readiness = self._get_pipeline_readiness()
            if force_distributed_assignment:
                with self._layer_config_lock:
                    transaction = self._pipeline_load_transaction or {}
                    phase = str(transaction.get("phase", "") or "")
                    plan = transaction.get("plan") or {}
                if phase in {"rejected", "aborted"}:
                    # ★ 2026-10-09（稳定性 #73-④）：**把 `invalidated` 从"不可恢复"里摘出来**。
                    #   `invalidated` 是「模型被替换、事务正常作废」的标记（见 api_server 的
                    #   `_invalidate_pipeline_load_transaction(reason_code="pipeline_model_changed")`），
                    #   它**必须允许重新提交** —— 否则实测会出现：切一次模型（尤其切到 GGUF 这类
                    #   master 引擎不支持的格式）后事务停在失败态，**之后每个请求都 503，连切回
                    #   正确模型、甚至重启后端都无效**。
                    #   而 `rejected`（本次容量计划不成立）与 `aborted`（本地提交失败）仍保持
                    #   fail-closed；它们由模型切换路径的清理逻辑负责复位。
                    return {
                        **readiness,
                        "ready": False,
                        "reason_code": (
                            transaction.get("reason_code")
                            or plan.get("reason_code")
                            or "pipeline_distributed_capacity_rejected"
                        ),
                        "reason": (
                            transaction.get("reason")
                            or plan.get("reason")
                            or "强制分布式容量计划未获准入"
                        ),
                    }
            if readiness.get("ready") and (
                not force_distributed_assignment
                or self._has_active_distributed_pipeline_plan()
            ):
                logger.info(
                    "主从模型配置同步完成，流水线已就绪: workers=%s",
                    worker_ids,
                )
                return readiness
            # 旧安装包不认识 authoritative_sync，会再次发送 opt-out。
            # 这是明确的不可恢复信号；不应让本次推理无谓等待完整同步
            # 超时，直接按既有安全路径回退到主节点。
            relay_for_worker = getattr(self, "_relay_segment_for_worker", None)
            with self._layer_config_lock:
                opted_out = sorted(
                    node_id for node_id in set(worker_ids)
                    if node_id in self._pipeline_worker_opt_out
                    and not (
                        callable(relay_for_worker)
                        and relay_for_worker(node_id) is not None
                    )
                )
            if opted_out:
                logger.warning(
                    "从节点拒绝主节点权威模型同步，立即回退: workers=%s",
                    opted_out,
                )
                return readiness
            if (
                not force_distributed_assignment
                and readiness.get("reason_code") not in recoverable
            ):
                return readiness
            if readiness.get("ready") and force_distributed_assignment:
                readiness = {
                    **readiness,
                    "ready": False,
                    "reason_code": "pipeline_single_node_plan_active",
                    "reason": "当前计划仍是单节点，等待多节点容量计划生效",
                }
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                logger.warning(
                    "等待主从模型配置同步超时: timeout=%.1fs reason=%s",
                    float(timeout),
                    readiness.get("reason", ""),
                )
                return readiness
            time.sleep(min(0.1, remaining))


    def _verify_pipeline_readiness(self, pipeline_nodes: list
                                   ) -> tuple:
        """
        二次就绪检查（出队后 / 立即执行前调用）。

        与 _all_pipeline_nodes_ready 的区别：
        - _all_pipeline_nodes_ready: 入队前的快速筛选（Pre-queue gate）
        - _verify_pipeline_readiness: tokenize 前的最终确认（Post-queue gate）

        入队等待期间节点可能离线 / 心跳超时 / TCP 断开，
        此检查在即将开始推理前做最后验证，避免浪费 prefill 计算。

        Returns:
            (ok: bool, reason: str)
        """
        if not self._tcp_server or not self._tcp_server._running:
            return False, "TCP 服务端未运行"

        # Phase 2.1+: 快照避免循环中并发修改
        with self._nodes_lock:
            nodes_snapshot = dict(self.nodes)

        for node in pipeline_nodes:
            node_id = node["node_id"]
            node_info = nodes_snapshot.get(node_id)
            if not node_info:
                return False, f"节点 {node_id} 已消失（可能被注销）"
            if not node_info.is_available():
                return False, f"节点 {node_id} 已离线 (state={node_info.state.value})"
            get_client_ids = getattr(self._tcp_server, "get_client_ids", None)
            connected_ids = (
                get_client_ids()
                if callable(get_client_ids)
                else getattr(self._tcp_server, "clients", {}).keys()
            )
            if node_id not in connected_ids:
                return False, f"节点 {node_id} TCP 连接已断开"

            # 心跳新鲜度
            heartbeat_age = time.time() - node_info.last_heartbeat
            if heartbeat_age > _WORKER_HEARTBEAT_MAX_AGE:
                return False, (
                    f"节点 {node_id} 心跳过期 "
                    f"({heartbeat_age:.1f}s > {_WORKER_HEARTBEAT_MAX_AGE:.0f}s)"
                )

            # ★ A1 / X 档（Y 档第二条缺口 3 的**第二处**同型判据）：relay 段节点**不需要**
            #   确认"层配置加载" —— 段工件由远端 relay_mid_service 持有，本节点不加载模型
            #   （`scheduler_pipeline.py:2255` 注释即 "this scheduler host does not need a
            #   local ModelHost/model loaded"）。上面已检查过在线 / TCP / 心跳，这里对
            #   relay 节点直接放行；段**不可达**会在**运行时**具名失败。
            relay_for_check = getattr(self, "_relay_segment_for_worker", None)
            is_relay_worker = (
                callable(relay_for_check) and relay_for_check(node_id) is not None
            )
            if node.get("execution") == "stage_offer_v3":
                stage_ready, stage_reason = self._stage_offer_assignment_ready(
                    node_id, node,
                )
                if not stage_ready:
                    return False, (
                        f"node {node_id} v3 layer_forward not ready: "
                        f"{stage_reason}"
                    )
            elif not is_relay_worker:
                with self._layer_config_lock:
                    expected = self._worker_assignments.expected(node_id)
                    ack = self._worker_assignments.acknowledgement(node_id)
                    expected_range = [node.get("start_layer"), node.get("end_layer")]
                    layer_ready = (
                        self._effective_layer_config_pushed(node_id)
                        and ack.get("assignment_id") == expected.get("assignment_id")
                        and ack.get("config_id") == expected.get("config_id")
                        and ack.get("layer_range") == expected_range
                        and ack.get("model_sha256") == expected.get("model_sha256")
                        and ack.get("model_type") == expected.get("model_type")
                        and ack.get("engine") == expected.get("engine", "pytorch")
                    )
                if not layer_ready:
                    return False, f"节点 {node_id} 尚未确认层配置加载成功"

        logger.info(
            f"✅ 二次就绪检查通过: "
            f"{' → '.join(n['node_id'] for n in pipeline_nodes)}"
        )
        return True, "ok"


    def _broadcast_pipeline_abort(self, pipeline_nodes: list, task_id: str,
                                   reason: str, count_error: bool = True) -> None:
        """向所有流水线节点广播 PIPELINE_ABORT（清理各节点 + master 本地 KV cache）。"""
        from transport_port import MessageType
        failed_nodes = []
        for n in pipeline_nodes:
            node_id = n.get("node_id")
            if not node_id:
                continue
            try:
                self._send_to_worker(
                    node_id,
                    {
                        "task_id": task_id,
                        "reason": reason,
                        "count_error": count_error,
                    },
                    MessageType.PIPELINE_ABORT,
                )
            except Exception as e:
                failed_nodes.append(f"{node_id}: {e}")
                logger.warning(
                    "PIPELINE_ABORT 发送失败: node=%s task=%s error=%s",
                    node_id, task_id, e,
                    exc_info=True,
                )
        if failed_nodes:
            logger.warning(
                "PIPELINE_ABORT 部分节点清理失败: task=%s failed=%s",
                task_id, "; ".join(failed_nodes),
            )
        # ★ 同时清理 master 自身 KV cache（master_participates 路径会产生本地缓存）
        if task_id:
            with self._kv_cache_lock:
                if task_id in self._kv_cache:
                    del self._kv_cache[task_id]
            with self._pipeline_lock:
                self._chain_ack_state.pop(task_id, None)


    def _send_to_worker(self, worker_id: str, data: dict,
                        msg_type=None) -> None:
        """主节点 → 从节点：发送消息"""
        from transport_port import MessageType
        if msg_type is None:
            msg_type = MessageType.LAYER_FORWARD
        if not self._tcp_server or not self._tcp_server._running:
            raise ConnectionError("TCP 服务端未运行")
        self._tcp_server.send_to_client(worker_id, data, msg_type)


    @staticmethod
    def _build_star_chain_route(pipeline_nodes: list[dict]) -> list[dict[str, str]]:
        """Return the logical segment order without exposing peer addresses.

        Workers only need the next segment identity.  The master owns address
        resolution and every inter-segment forward, which keeps the data plane
        hub-and-spoke while preserving D→L/L→L multi-segment semantics.
        """
        route: list[dict[str, str]] = []
        for node in pipeline_nodes:
            node_id = str(node.get("node_id", "") or "").strip()
            if not node_id:
                raise ValueError("pipeline segment is missing node_id")
            route.append({"node_id": node_id})
        return route

    @staticmethod
    def _parse_relay_segment_map(raw: str) -> dict[str, dict[str, object]]:
        """★ A1 / X 档（Y 档第二条扩层区间）：解析 `QLH_RELAY_SEGMENTS`。

        ★ 2026-10-07（DIST-NEXT-7）：实现已迁到 `relay_a1_legacy.parse_relay_segment_map`
        （A1 的隔离边界），判据仍复用 `_normalize_relay_segment`（单一真源）。
        保留本静态方法名以兼容既有调用点与测试。
        """
        return parse_relay_segment_map(
            raw, SchedulerPipelineMixin._normalize_relay_segment,
        )

    def _relay_segment_for_worker(self, worker_id: str,
                                  routing_preference: str = "auto") -> Optional[dict]:
        """★ A1 / X 档：该 worker 是否由远端 relay 段代跑本段（主节点侧配置，解析一次后缓存）。

        **三重闸门**，任一不成立即 `None`（对既有路径零影响）：

        1. **全局开关** `QLH_RELAY_ENABLED`（默认关）；
        2. **请求级** `routing_preference == "local_only"` ⇒ 不委派（语义一致：「只要本地算」）。
           注：API 层在 `local_only` 时**本就不会进流水线路径**（`api_server.py` 的
           `req.routing_preference != "local_only"` 判断），这里是**防御性**的第二道边界 ——
           即使将来有别的入口把 `local_only` 请求送进流水线，也不会被派给远端段；
        3. 该 worker 在 `QLH_RELAY_SEGMENTS` 映射里。

        更细的「请求级 relay 取舍」（例如按请求挑不同段）**不在 X 档**：`routing_preference`
        现有取值集不含这个维度，扩它等于改 API 契约 ⇒ 归 Y 档。

        缓存用 `getattr` 惰性挂在实例上，**不**改 `__init__`（本方法是 mixin 方法，
        实例可能来自多种构造路径）。
        """
        # ★ 2026-10-04 产品裁定（分票规划 DIST-0）：A1 relay 已从**产品调度入口
        #   剔除** —— 它依赖 probe/SSH 隧道与 loopback 段服务，只保留为技术探针与
        #   历史证据（产品主线是 A3 `stage_offer_v3`）。默认 `PROBE_ONLY=1` 时本方法
        #   直接不供给 relay 段：生产请求即使配了 `QLH_RELAY_ENABLED` /
        #   `QLH_RELAY_SEGMENTS`，也只记一份具名诊断并继续走 A3，**不做静默切换**。
        #   探针/实验要恢复 A1 行为时显式设 `QLH_RELAY_PROBE_ONLY=0`。
        if not a1_production_enabled(
            probe_only=PIPELINE_RELAY_PROBE_ONLY,
            relay_enabled=PIPELINE_RELAY_ENABLED,
        ):
            # ★ 2026-10-07（DIST-NEXT-7）：门集中在 `relay_a1_legacy.a1_production_enabled`；
            #   这里只保留「为什么不供给」的两种具名表现：探针闸门（默认，一次性告警）
            #   与总开关关闭（静默，保持既有行为）。
            if PIPELINE_RELAY_PROBE_ONLY:
                if not getattr(self, "_relay_probe_only_warned", False):
                    self._relay_probe_only_warned = True
                    logger.warning(
                        "A1 relay 已从产品调度入口剔除（QLH_RELAY_PROBE_ONLY=1，默认）："
                        "QLH_RELAY_ENABLED/QLH_RELAY_SEGMENTS 不再作为生产能力，生产请求"
                        "继续走 A3 stage_offer_v3。需要 A1 探针行为请显式设 "
                        "QLH_RELAY_PROBE_ONLY=0。首个受影响 worker=%s", worker_id,
                    )
            return None
        if str(routing_preference or "auto") == "local_only":
            return None
        cache = getattr(self, "_relay_segment_map_cache", None)
        if cache is None:
            cache = self._parse_relay_segment_map(PIPELINE_RELAY_SEGMENTS)
            self._relay_segment_map_cache = cache
        worker_key = str(worker_id or "").strip()
        spec = cache.get(worker_key)
        if spec is not None:
            return spec

        # Client workers conventionally register as ``client_<hostname>``
        # while deployment profiles often use the stable hostname alone. Keep
        # the exact key authoritative, then allow only the one unambiguous
        # client-prefix alias so a profile typo cannot silently assign a relay
        # segment to a different node.
        alias = (
            worker_key[7:]
            if worker_key.startswith("client_")
            else f"client_{worker_key}"
        )
        if alias and alias != worker_key:
            aliased = cache.get(alias)
            if aliased is not None:
                logger.info(
                    "relay 节点名按 client_ 别名匹配: worker=%s configured=%s",
                    worker_key, alias,
                )
                return aliased
        return None

    def _is_relay_host(self, node_id: str) -> bool:
        """该节点是否是 **relay 宿主** —— 「谁是 relay 宿主」的**唯一判据入口**。

        relay 宿主是**第三种角色**，不是「v3 层段 worker 的例外」。它的能力声明与层段
        worker **正好相反**：

        * **不能**声明 `FORWARD_LAYERS` —— 声明了它就会拒绝 legacy 层配置，而 relay
          委派恰恰走那条通道（实测：Surface 报「本节点是 v3 层段 worker，拒绝 legacy
          分层配置」）；
        * **要的**恰恰是那份 legacy 层配置（`engine="relay_middle"`），因此既不能被当
          Full Worker 释放预留，也不能被排出层段名单。

        这两条结论此前被复写在**五条**判定路径上（hello 的 opt-out、`_task_worker_full_
        model_ids`、容量求解的两处 releases、`_task_worker_layer_stage_ids`），每处各写
        一遍 `_relay_segment_for_worker(...) is not None`。少写一处就退化成「relay 链丢掉
        中间段」，而症状（该节点被释放预留）离原因很远 —— 实测连追了五轮。

        **刻意不接 `routing_preference`**：这是**节点角色**判定，不该随单次请求变化。
        `_relay_segment_for_worker()` 里的 `local_only` 闸门是**请求级**的（「本次只要本地
        算」），把它混进角色判定会让同一个节点在不同请求下被判成不同角色。需要请求级
        取舍的调用点直接调 `_relay_segment_for_worker(id, routing_preference)` 取 spec。

        ⚠️ 只传 `node_id` 一个位置参数：调用方（含测试与嵌入方）常注入单参 stub。
        """
        return self._relay_segment_for_worker(node_id) is not None

    def get_relay_a1_status(self) -> dict:
        """★ 2026-10-07（DIST-NEXT-7）：A1 的**独立诊断 namespace**（`relay_a1`）。

        把「A1 现在能不能被调度选中」收敛成单一入口：返回开关、段配置与 relay 宿主
        计数，其中 `production_enabled` 与 `assignment_selectable` 同源取值。生产侧
        （health / 日志 / 测试）据此判断，而不必逐个开关去猜，也不会与 Route A 的
        原因码混在一起。
        """
        cache = getattr(self, "_relay_segment_map_cache", None)
        if cache is None:
            cache = self._parse_relay_segment_map(PIPELINE_RELAY_SEGMENTS)
            self._relay_segment_map_cache = cache
        nodes = getattr(self, "nodes", {}) or {}
        relay_hosts = [
            node_id for node_id in nodes
            if self._relay_segment_for_worker(node_id) is not None
        ]
        return a1_isolation_status(
            probe_only=PIPELINE_RELAY_PROBE_ONLY,
            relay_enabled=PIPELINE_RELAY_ENABLED,
            segments_raw=PIPELINE_RELAY_SEGMENTS,
            configured_nodes=cache,
            relay_host_count=len(relay_hosts),
        )


    def _wait_for_layer_result(self, task_id: str, node_ids,
                               timeout: float = 30.0,
                               ack_node_ids: list = None,
                               ack_step: int = None,
                               ack_timeout: float = None,
                               cancel_event: threading.Event = None,
                               request_deadline: RequestDeadline = None) -> Optional[dict]:
        """
        主节点：等待指定节点的 LAYER_RESULT。

        node_ids 可以是单个 str 或 list[str]。当传入 list 时，
        等待其中任一节点返回结果（链式拓扑中错误可能来自任意节点）。

        使用 threading.Event 实现同步等待，由 _handle_layer_result 唤醒。
        """
        import base64

        if isinstance(node_ids, str):
            node_ids = [node_ids]

        keys = [f"{task_id}:{nid}" for nid in node_ids]

        # 先消费已经到达的结果，避免 worker 极快返回时发生
        # "结果先写入、event 后创建" 的竞态。
        events = []
        result = None
        signaled_key = None
        with self._pipeline_lock:
            for key in keys:
                data = self._pipeline_results.pop(key, None)
                if data is not None:
                    result = data
                    signaled_key = key
                    break
            if result is None:
                for key in keys:
                    event = threading.Event()
                    self._pipeline_events[key] = event
                    events.append((key, event))

        # 等待任一 event 触发
        deadline = time.monotonic() + timeout
        if request_deadline is not None:
            deadline = min(
                deadline, request_deadline.expires_at_monotonic,
            )
        while result is None and time.monotonic() < deadline:
            stop_reason = request_stop_reason(cancel_event, request_deadline)
            if stop_reason:
                result = {
                    "task_id": task_id,
                    "error": (
                        "流水线任务已取消"
                        if stop_reason == REASON_GENERATION_CANCELLED
                        else "request deadline exceeded"
                    ),
                    "cancelled": stop_reason == REASON_GENERATION_CANCELLED,
                    "reason_code": stop_reason,
                    "step": ack_step if ack_step is not None else -1,
                }
                break
            if ack_node_ids and ack_step is not None and ack_timeout is not None:
                ack_failure = self._get_chain_ack_failure(
                    task_id, ack_step, ack_node_ids, ack_timeout,
                )
                if ack_failure is not None:
                    result = ack_failure
                    signaled_key = f"{task_id}:{ack_failure.get('node_id')}"
                    break
            for key, event in events:
                if event.is_set():
                    signaled_key = key
                    break
            if signaled_key:
                break
            time.sleep(0.05)

        if result is None:
            stop_reason = request_stop_reason(cancel_event, request_deadline)
            if stop_reason:
                result = {
                    "task_id": task_id,
                    "error": (
                        "流水线任务已取消"
                        if stop_reason == REASON_GENERATION_CANCELLED
                        else "request deadline exceeded"
                    ),
                    "cancelled": stop_reason == REASON_GENERATION_CANCELLED,
                    "reason_code": stop_reason,
                    "step": ack_step if ack_step is not None else -1,
                }

        # 清理所有 events，收集结果
        with self._pipeline_lock:
            for key, _ in events:
                self._pipeline_events.pop(key, None)

            if result is None:
                # 查找第一个有结果或超时的 key。即使 event 轮询刚好错过
                # 最后一瞬间，也以实际结果为准。
                for key in keys:
                    data = self._pipeline_results.pop(key, None)
                    if data is not None:
                        result = data
                        signaled_key = key
                        break

        if result is None:
            logger.error(f"⏰ 等待流水线结果超时 ({timeout}s), task={task_id}")
            return None

        # 解码 base64 → bytes（供调用方反序列化张量）
        decoded = {}
        for k, v in result.items():
            if isinstance(v, str) and k in ("hidden_states", "logits"):
                try:
                    decoded[k] = base64.b64decode(v)
                except Exception:
                    decoded[k] = v
            else:
                decoded[k] = v
        return decoded


    def _wait_for_layer_result_with_ack(self, task_id: str, node_ids,
                                        timeout: float,
                                        ack_node_ids: list,
                                        ack_step: int,
                                        ack_timeout: float,
                                        cancel_event: threading.Event = None,
                                        request_deadline: RequestDeadline = None) -> Optional[dict]:
        """等待流水线结果；兼容测试或旧扩展中替换掉的三参数等待函数。"""
        try:
            return self._wait_for_layer_result(
                task_id,
                node_ids,
                timeout=timeout,
                ack_node_ids=ack_node_ids,
                ack_step=ack_step,
                ack_timeout=ack_timeout,
                cancel_event=cancel_event,
                request_deadline=request_deadline,
            )
        except TypeError as e:
            if ("ack_node_ids" not in str(e)
                    and "cancel_event" not in str(e)
                    and "request_deadline" not in str(e)):
                raise
            logger.debug(
                "_wait_for_layer_result 不支持 ACK 参数，退回旧签名调用",
                exc_info=True,
            )
            stop_reason = request_stop_reason(
                cancel_event, request_deadline,
            )
            if stop_reason:
                return {
                    "task_id": task_id,
                    "error": (
                        "pipeline task cancelled"
                        if stop_reason == REASON_GENERATION_CANCELLED
                        else "request deadline exceeded"
                    ),
                    "cancelled": stop_reason == REASON_GENERATION_CANCELLED,
                    "reason_code": stop_reason,
                    "step": ack_step,
                }
            legacy_timeout = float(timeout)
            if request_deadline is not None:
                legacy_timeout = request_deadline.clamp(legacy_timeout)
            if legacy_timeout <= 0:
                return {
                    "task_id": task_id,
                    "error": "request deadline exceeded",
                    "cancelled": False,
                    "reason_code": REQUEST_DEADLINE_EXCEEDED,
                    "step": ack_step,
                }
            result = self._wait_for_layer_result(
                task_id, node_ids, legacy_timeout,
            )
            stop_reason = request_stop_reason(
                cancel_event, request_deadline,
            )
            if stop_reason:
                return {
                    "task_id": task_id,
                    "error": (
                        "pipeline task cancelled"
                        if stop_reason == REASON_GENERATION_CANCELLED
                        else "request deadline exceeded"
                    ),
                    "cancelled": stop_reason == REASON_GENERATION_CANCELLED,
                    "reason_code": stop_reason,
                    "step": ack_step,
                }
            return result


    def run_pipeline_stream(self, prompt: str, **kwargs):
        """
        流式版本：逐 token yield 事件字典，用于 SSE 推送。

        内部通过线程+队列包装 run_pipeline() 的 _stream_callback，
        将 callback 调用转为 generator yield。

        Yields:
            {"token": str}       — 新生成的 token 文本
            {"done": True, "response": str, "metrics": dict, ...}
                                  — 完成信号（含完整响应和指标）
            {"done": True, "error": str}
                                  — 错误信号
        """
        import queue
        import threading as _thr

        q = queue.Queue()
        callback_called = _thr.Event()
        request_deadline = kwargs.get("_request_deadline")
        cancel_event = kwargs.pop("_cancel_event", None)
        if not isinstance(cancel_event, DeadlineCancelEvent):
            cancel_event = DeadlineCancelEvent(
                cancel_event, request_deadline,
            )

        def on_token(event):
            if "done" in event:
                callback_called.set()
            q.put(event)

        def _run():
            try:
                result = self.run_pipeline_safe(
                    prompt,
                    _stream_callback=on_token,
                    _cancel_event=cancel_event,
                    **kwargs,
                )
                # 错误路径：run_pipeline 直接返回了 error（未走 callback）
                if not callback_called.is_set():
                    q.put({
                        "done": True,
                        "error": result.get("error", "unknown"),
                        "reason_code": result.get("reason_code"),
                        "cancelled": result.get("cancelled", False),
                        "response": result.get("response", ""),
                        "metrics": result.get("metrics", {}),
                    })
            except Exception as e:
                logger.error(f"流式推理异常: {e}", exc_info=True)
                q.put({"done": True, "error": str(e)})

        _thr.Thread(target=_run, name="pipeline-stream", daemon=True).start()

        try:
            while True:
                stop_reason = request_stop_reason(
                    cancel_event, request_deadline,
                )
                if stop_reason:
                    cancel_event.set()
                    yield {
                        "done": True,
                        "error": (
                            "generation cancelled"
                            if stop_reason == REASON_GENERATION_CANCELLED
                            else "request deadline exceeded"
                        ),
                        "cancelled": stop_reason == REASON_GENERATION_CANCELLED,
                        "reason_code": stop_reason,
                    }
                    break
                wait_timeout = 0.1
                if request_deadline is not None:
                    remaining = request_deadline.remaining()
                    if remaining <= 0:
                        continue
                    wait_timeout = min(wait_timeout, remaining)
                try:
                    event = q.get(timeout=wait_timeout)
                except queue.Empty:
                    continue
                yield event
                if "done" in event:
                    break
        finally:
            if not callback_called.is_set():
                cancel_event.set()


    @staticmethod
    def _stream_output_started(kwargs: dict) -> bool:
        """返回流式调用是否已向客户端发送过正文 token。"""
        callback = kwargs.get("_stream_callback")
        return bool(getattr(callback, "_qlh_tokens_emitted", False))


    @staticmethod
    def _track_stream_output(kwargs: dict) -> None:
        """包装流式回调，供失败回退判断是否会造成回答重放。"""
        callback = kwargs.get("_stream_callback")
        if not callable(callback) or getattr(callback, "_qlh_stream_tracker", False):
            return

        def tracked_callback(event):
            if isinstance(event, dict) and event.get("token"):
                tracked_callback._qlh_tokens_emitted = True
            callback(event)

        tracked_callback._qlh_stream_tracker = True
        tracked_callback._qlh_tokens_emitted = False
        kwargs["_stream_callback"] = tracked_callback




    def run_pipeline_safe(self, prompt: str, **kwargs) -> dict:
        """Admit only an explicitly published deployment into the FIFO owner."""
        return self._execution_app.run(prompt, **kwargs)


    # ★ 2026-10-09（#78 缺口 2）：`reason_code` → 用户可读的中文说明。
    #   容量求解器给出的 code 是给机器看的（如 `pipeline_capacity_workers_unavailable`），
    #   直接抛给用户等于没抛；这里补一层人话，让 UI 能自助诊断。
    _PIPELINE_REASON_HINT = {
        "pipeline_capacity_workers_unavailable": (
            "没有可用的分层 worker（从节点未连上，或未声明/未通过校验层段工件）"
        ),
        "pipeline_distributed_workers_unavailable": "分布式放置至少需要两个可用节点",
        "pipeline_capacity_nodes_unavailable": "当前没有满足条件的可用节点",
        "pipeline_segment_contract_unsatisfied": "层段契约不满足（声明区间与所需区间不符）",
        "pipeline_layer_range_coverage_insufficient": "各节点声明的层区间覆盖不足",
        "pipeline_capacity_single_node_insufficient": (
            "区间被约束后，没有任何单个节点能在自己的空闲内存里装下所属层段"
            "（总容量够，但切分后单节点不够 ⇒ 检查各节点空闲内存/降低精度）"
        ),
        "pipeline_distributed_capacity_insufficient": "分布式各节点总容量不足",
        "pipeline_required_node_unavailable": "指定执行节点当前没有可用的模型容量",
        "node_capacity_unavailable": "节点容量不足（内存/显存）",
        "pipeline_capacity_rejected": "集群容量准入被拒",
        "pipeline_reshard_capacity_insufficient": "重新分片后容量不足",
        "model_identity_mismatch": "参与节点的模型身份不一致（算子/工件不匹配）",
        "pipeline_model_changed": "流水线准备期间模型被切换",
        # ★ 2026-10-09（子 agent 复查 N2）：补齐**生产路径会真实产生**但此前漏收录的码 ——
        #   漏了就等于"有码没说明"，UI 只能退回显示裸码。
        "pipeline_local_commit_failed": "本地提交层段失败（master 侧无法按分配裁层/物化）",
        "pipeline_full_model_fallback_forbidden": (
            "该模型只以分布式流水线模式准备，禁止退回整模推理（无可用原因信息）"
        ),
        "distributed_full_model_forbidden": "分布式模式禁止整模回退；需要整模基线时请手动切换到本地推理",
        "distributed_inference_disabled": "本节点已选择本地推理，分布式候选停止发布",
        "pipeline_layer_range_not_advertised": (
            "节点未声明层区间，无法判断它能承担哪一段"
        ),
        "pipeline_capacity_descriptor_invalid": "模型描述信息不合法，无法做容量求解",
        "relay_layer_claim_invalid": "relay 段认领的层区间不合法（重叠/角色/不连续）",
        # ★ 2026-10-09：由"该红必须红"测试扫出的其余生产码（见
        #   `tests/test_pipeline_fallback_diagnostic.py::test_every_production_reason_code_has_a_hint`）。
        "pipeline_capacity_manual_insufficient": "手动切分的容量不足",
        "pipeline_capacity_manual_node_unavailable": "手动切分指定的节点不可用",
        "pipeline_capacity_manual_range_invalid": "手动切分的层区间不合法",
        "pipeline_capacity_not_computed": "容量尚未计算（集群还没就绪）",
        "pipeline_descriptor_unavailable": "模型层段描述不可用（无法确定总层数/切分点）",
        "pipeline_node_contract_invalid": "层段布局不满足「恰好连续覆盖每一层」契约",
        "pipeline_reshard_descriptor_unavailable": "重分片缺少模型描述",
        "pipeline_reshard_layout_invalid": "重分片后的层布局不合法",
        "pipeline_reshard_plan_mismatch": "重分片计划与当前计划不一致",
        "pipeline_runtime_unsupported": "当前运行时/引擎不支持该模型的层段执行",
        "pipeline_single_node_plan_active": "已存在单机分层计划，与分布式请求冲突",
        # ★ 2026-10-10（票4 MODEL-TXN-01）：模型/pipeline 原子切换新产生的码，
        #   由 `test_every_production_reason_code_has_a_hint` 扫出（漏了 UI 只能显示裸码）。
        "pipeline_master_required": "只有 master 节点能编排模型切换（当前节点角色不是 master）",
        "pipeline_model_transaction_committing": (
            "模型事务正在跨越提交栅栏（local/global commit 进行中），请稍后重试"
        ),
        "pipeline_transaction_not_started": "模型事务未能启动（候选准备没有返回结果）",
        "pipeline_workers_unavailable": (
            "没有可用的分层 worker（TCP 服务未运行，或没有在线的从节点）"
        ),
    }

    #: ★ 这些码出现在 `admitted=True`（成功）的 plan 里，**不是失败原因** ⇒ 不参与
    #: 「每个失败码都要有中文说明」的检查（但仍保留在本表方便 UI 直接查）。
    _PIPELINE_SUCCESS_REASON_CODES = frozenset({"distributed_forced"})

    def _pipeline_unavailable_diagnostic(self) -> tuple:
        """返回 `(reason_code, 人类可读说明)`，用于把「流水线不可用」的真实原因透给调用方/UI。

        ★ 2026-10-09（#78 缺口 2）：此前回退路径只吐一句**硬编码**文案
        「当前模型仅以分布式流水线模式准备，禁止整模回退；请等待从节点就绪」——
        真因（如 `pipeline_capacity_workers_unavailable` = 从节点未声明工件）只留在
        logcat 里，用户无从自助诊断。现在把容量求解器给出的 `reason_code` / `reason` /
        `excluded_nodes` 一并带出去。
        """
        code = ""
        parts = []
        excluded = []
        try:
            txn = self._pipeline_load_transaction or {}
            code = str(txn.get("reason_code") or "")
            plan = txn.get("plan") if isinstance(txn.get("plan"), dict) else {}
            if not code:
                plan = plan or {}
            reason_text = str(plan.get("reason") or "")
            excluded = list(plan.get("excluded_nodes") or [])
        except Exception:  # pragma: no cover - 诊断路径绝不抛
            reason_text = ""
            excluded = []
        if not code:
            active = getattr(self, "_active_pipeline_capacity_plan", None)
            if isinstance(active, dict):
                code = str(active.get("reason_code") or "")
                reason_text = str(active.get("reason") or "") or reason_text
                excluded = list(active.get("excluded_nodes") or []) or excluded
        if not code:
            return "", ""
        hint = self._PIPELINE_REASON_HINT.get(code, "")
        for item in (hint, reason_text):
            if item:
                parts.append(item)
        if excluded:
            preview = []
            for node in excluded[:4]:
                if isinstance(node, dict):
                    nid = node.get("node_id") or node.get("id") or "?"
                    why = node.get("reason") or node.get("reason_code") or ""
                    preview.append(f"{nid}({why})" if why else str(nid))
                else:
                    preview.append(str(node))
            parts.append("被排除的节点: " + ", ".join(preview))
        return code, "；".join(parts)

    def _run_full_model_inference(self, prompt: str,
                                   max_new_tokens: int = 512,
                                   temperature: float = 0.7,
                                   top_p: float = 0.9,
                                   session_id: str = None,
                                   **kwargs) -> dict:
        """
        回退模式：在主节点本地执行完整模型推理。

        当流水线节点不可用时，使用 model_manager.chat() 直接推理。
        若调用方传入 _stream_callback，则使用 chat_stream() 逐 token 推送。
        """
        request_deadline = kwargs.pop("_request_deadline", None)
        terminal_outcome = kwargs.pop("_terminal_outcome", None)
        cancel_event = kwargs.pop("_cancel_event", None)
        if request_deadline is not None and not isinstance(
            cancel_event, DeadlineCancelEvent,
        ):
            cancel_event = DeadlineCancelEvent(
                cancel_event, request_deadline,
            )
        terminal = _request_terminal_result(
            cancel_event=cancel_event,
            deadline=request_deadline,
            terminal_outcome=terminal_outcome,
            source="fallback_entry",
        )
        if terminal is not None:
            return terminal
        mgr = self._host
        if mgr and getattr(mgr, "is_pipeline_prepared", False):
            # ★ 2026-10-09（#78 缺口 2）：附上真实 `reason_code` + 人话说明 + 被排除的节点。
            #   此前这里只吐硬编码文案，真因（如从节点未声明工件）只在 logcat 里，
            #   用户看到「请等待从节点就绪」无从判断到底缺什么。
            _code, _detail = "", ""
            try:
                _code, _detail = self._pipeline_unavailable_diagnostic()
            except Exception:  # pragma: no cover - 诊断失败不影响主流程
                pass
            _suffix = ""
            if _code:
                _suffix = f"（原因: {_code}"
                if _detail:
                    _suffix += f" —— {_detail}"
                _suffix += "）"
            return {
                "response": "",
                "error": (
                    "当前模型仅以分布式流水线模式准备，禁止整模回退；"
                    "请等待从节点就绪或显式执行普通模型加载" + _suffix
                ),
                "reason_code": _code or "pipeline_full_model_fallback_forbidden",
            }
        if not mgr or not mgr.is_loaded:
            return {"response": "", "error": "模型未加载"}
        if self.get_distributed_inference_enabled():
            return {
                "response": "", "reason_code": "distributed_full_model_forbidden",
                "error": "分布式模式禁止整模回退；请先手动切换到本地推理模式。",
            }

        _stream_callback = kwargs.pop('_stream_callback', None)
        fallback_reason = kwargs.pop('_fallback_reason', '') or 'pipeline_fallback_full_model'
        show_thinking = bool(kwargs.pop("show_thinking", False))
        # ★ 2026-09-19：深度思考**开关**（与 show_thinking 的「展示」语义区分开）。
        #   None ⇒ 不干预，沿用模型模板默认（Qwen3 模板默认会思考，故会出现超长 <think>）。
        #   显式 False ⇒ 引擎 `_set_thinking_mode(False)` 会经 chat template 的
        #   `enable_thinking=False` **真正阻止**模型生成思考内容（省算力），
        #   而不是靠事后剥离（后者依赖模板含 `<think>` 且能找到 `</think>`，任一不成立即失效）。
        enable_thinking = kwargs.pop("enable_thinking", None)
        # ★ 若 master 刚执行过流水线裁剪（layer_range != None），
        #   需要先重新加载完整模型，否则 chat()/chat_stream() 会因
        #   缺 Embedding/LM Head 而报错（如 RuntimeError: 缺少 lm_head）。
        try:
            ensure_full = getattr(mgr, 'ensure_full_model', None)
            logger.info(
                "回退前完整模型检查: mgr=%s callable=%s is_pipeline_prepared=%s",
                type(mgr).__name__, callable(ensure_full),
                getattr(mgr, "is_pipeline_prepared", None),
            )
            if callable(ensure_full):
                ensure_full()
        except Exception as e:
            logger.error(f"完整模型重载失败: {e}")
            return {"response": "", "error": f"完整模型恢复失败: {e}"}

        terminal = _request_terminal_result(
            cancel_event=cancel_event,
            deadline=request_deadline,
            terminal_outcome=terminal_outcome,
            source="fallback_prepare",
        )
        if terminal is not None:
            return terminal

        # ★ 2026-10-06（docs/已知问题记录.md #38）：**回退前必须确认 master 真的持有全层
        #   模型**。此前只依赖 `is_pipeline_prepared`（= `_pipeline_distributed_only and
        #   _pipeline_descriptor`）＋ `ensure_full_model()` 两道守卫，但二者都可能不成立却
        #   仍放着裁层工件往下跑：实测 master 加载的是 `qwen25-05b-f16-head8.gguf`
        #   （日志 `llama.cpp 层段已加载: [0,8) embed=True lm_head=False`），回退却按
        #   「全层」硬跑 ⇒ 产出垃圾文本**却返回 HTTP 200**（fail-open 最坏的一种：掩盖故障
        #   且伪装成功，对拍/验收/目视都会据此误判）。
        #   这里改用**直接判据**：`layer_range`（`src/model_module.py:836` 明确
        #   `None` = 完整模型）＋ `lm_head` —— 任一不满足即 fail-closed 且给具名原因。
        _layer_range = getattr(mgr, "layer_range", None)
        _desc: dict = {}
        try:
            _get_desc = getattr(mgr, "get_pipeline_descriptor", None)
            if callable(_get_desc):
                _d = _get_desc() or {}
                if isinstance(_d, dict):
                    _desc = _d
        except Exception:  # noqa: BLE001 - 描述符不可用时按「未声明」处理
            _desc = {}
        # 判据用**描述符里的裁层痕迹**，因为两侧通用：PyTorch 侧另有 `layer_range`
        # （`model_module.py:836`，None=整模），而 **llama.cpp 侧没有该实例属性**，
        # 它的裁层状态只体现在 `_pipeline_descriptor` 的
        # `assignment_layer_range` / `partial_assignment` / `loaded_artifact` 与
        # `lm_head`（`llama_engine.py:516-528`）。实测 #38 正发生在 **llama.cpp** master：
        # 日志 `llama.cpp 层段已加载: [0,8) embed=True lm_head=False` ——
        # 而 LlamaCppEngine **既无 `ensure_full_model` 也无 `layer_range`** ⇒
        # 上面两道旧守卫对它完全无效，只能靠这里挡住。
        _partial = bool(
            _desc.get("assignment_layer_range")
            or _desc.get("partial_assignment")
            or _desc.get("loaded_artifact")
        )
        _lm_head = _desc.get("lm_head") if "lm_head" in _desc else None
        if _layer_range is not None or _partial or _lm_head is False:
            logger.error(
                "拒绝整模回退：master 并未持有全层模型 "
                "(layer_range=%s partial=%s lm_head=%s)",
                _layer_range, _partial, _lm_head,
            )
            return {
                "response": "",
                "error": (
                    "refusing_full_model_fallback_with_partial_artifact: "
                    f"layer_range={_layer_range} partial={_partial} lm_head={_lm_head}；"
                    "master 持有的是裁层工件，整模回退会产出垃圾且伪装成功"
                ),
            }

        try:
            messages = kwargs.pop("messages", None) or [{"role": "user", "content": prompt}]
            callbacks = self._require_callbacks()
            if show_thinking and not any(item.get("role") == "system" for item in messages):
                messages = [
                    {"role": "system", "content": callbacks.thinking_system_prompt},
                    *messages,
                ]
            try:
                fallback_prompt = callbacks.build_model_chat_prompt(mgr.tokenizer, messages)
                native_thinking_prompt = "<think>" in fallback_prompt[-128:].lower()
            except Exception:
                native_thinking_prompt = False

            if _stream_callback:
                # 流式路径：逐 token 推送
                full_text_parts = []
                visible_buffer = ""
                suppress_thinking = bool(native_thinking_prompt and not show_thinking)
                # ★ 2026-10-24（#125）：**思考必须边想边出**。
                #   此前只在 `suppress_thinking`（= 有思考模板但未要求展示）时把整段思考
                #   攒进 `visible_buffer`，直到 `</think>` 出现才吐正文；而 `show_thinking`
                #   为真时虽然不抑制，却把 `<think>…</think>` 原文当成正文 token 推出去。
                #   两种情况下客户端都**看不到"正在思考"的独立信号**：实测一个 prompt_len=2
                #   的请求，首字节等了 88.7 s（= 纯思考时长），期间界面完全空转。
                #
                #   现在：`show_thinking` 为真时按标签**分流**——思考增量发
                #   `{"thinking": …}`，正文发 `{"token": …}`，客户端可即时渲染思考区。
                #   `show_thinking` 为假时保持原行为（抑制思考，仅发正文）。
                split_thinking = bool(show_thinking)
                think_state = False      # 当前是否处于 <think> 区段
                think_pending = ""       # 跨 chunk 的标签残留缓冲
                t0 = time.time()
                for chunk in mgr.chat_stream(
                    messages=messages,
                    max_tokens=max_new_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    enable_thinking=enable_thinking,
                    _cancel_event=cancel_event,
                ):
                    if not chunk:
                        continue
                    full_text_parts.append(chunk)
                    if suppress_thinking:
                        visible_buffer += chunk
                        marker = visible_buffer.lower().find("</think>")
                        if marker >= 0:
                            visible = visible_buffer[marker + len("</think>"):]
                            suppress_thinking = False
                            visible_buffer = ""
                            if visible:
                                _stream_callback({"token": visible})
                        continue
                    if not split_thinking:
                        _stream_callback({"token": chunk})
                        continue
                    # 分流：把 chunk 按 <think> / </think> 切成 thinking / token 片段。
                    think_pending += chunk
                    segments = []
                    while think_pending:
                        if not think_state:
                            index = think_pending.lower().find("<think>")
                            if index < 0:
                                # 无开标签 ⇒ 全是正文（模型没走思考模板时也安全）。
                                segments.append(("token", think_pending))
                                think_pending = ""
                                break
                            if index > 0:
                                segments.append(("token", think_pending[:index]))
                            think_pending = think_pending[index + len("<think>"):]
                            think_state = True
                        else:
                            index = think_pending.lower().find("</think>")
                            if index < 0:
                                segments.append(("thinking", think_pending))
                                think_pending = ""
                                break
                            if index > 0:
                                segments.append(("thinking", think_pending[:index]))
                            think_pending = think_pending[index + len("</think>"):]
                            think_state = False
                    for kind, text in segments:
                        if text:
                            _stream_callback({kind: text})
                raw_response_text = "".join(full_text_parts)
                terminal = _request_terminal_result(
                    cancel_event=cancel_event,
                    deadline=request_deadline,
                    terminal_outcome=terminal_outcome,
                    source="fallback_execute",
                )
                if terminal is not None:
                    return terminal
                response_text, thinking_content = callbacks.format_model_response(
                    raw_response_text,
                    show_thinking,
                    native_thinking_prompt=native_thinking_prompt,
                )
                elapsed = time.time() - t0
                metrics = {
                    "engine": backend_id_for(mgr, default='unknown') or 'unknown',
                    "mode": "fallback_full_model_streaming",
                    "execution_mode": "fallback_full_model_streaming",
                    "distributed_requested": True,
                    "distributed_used": False,
                    "fallback": True,
                    "fallback_reason": fallback_reason,
                    "route": "master_pipeline_fallback_full_model_streaming",
                    "serving_node_id": self.get_effective_node_id(),
                    "workers_used": [],
                    "layer_assignments": [],
                    "tokens_per_second": len(full_text_parts) / elapsed if elapsed > 0 else 0,
                    "chunks": len(full_text_parts),
                    "elapsed_seconds": round(elapsed, 3),
                }
                # ★ 发送完成信号（与 run_pipeline 一致）
                _stream_callback({
                    "done": True,
                    "response": response_text,
                    "thinking": thinking_content,
                    "metrics": metrics,
                })
            else:
                result = mgr.chat(
                    messages=messages,
                    max_tokens=max_new_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    enable_thinking=enable_thinking,
                    _cancel_event=cancel_event,
                )
                terminal = _request_terminal_result(
                    cancel_event=cancel_event,
                    deadline=request_deadline,
                    terminal_outcome=terminal_outcome,
                    source="fallback_execute",
                )
                if terminal is not None:
                    return terminal
                raw_response_text = result.get("content", "")
                response_text, thinking_content = callbacks.format_model_response(
                    raw_response_text,
                    show_thinking,
                    native_thinking_prompt=native_thinking_prompt,
                )
                usage = result.get("usage", {}) or {}
                completion_tokens = usage.get("completion_tokens", 0)
                metrics = {
                    "engine": backend_id_for(mgr, default='unknown') or 'unknown',
                    "mode": "fallback_full_model",
                    "execution_mode": "fallback_full_model",
                    "distributed_requested": True,
                    "distributed_used": False,
                    "fallback": True,
                    "fallback_reason": fallback_reason,
                    "route": "master_pipeline_fallback_full_model",
                    "serving_node_id": self.get_effective_node_id(),
                    "workers_used": [],
                    "layer_assignments": [],
                    "tokens_per_second": result.get("tokens_per_second", 0),
                    "generated_tokens": completion_tokens,
                    "completion_tokens": completion_tokens,
                    "usage": usage,
                }

            return {
                "response": response_text,
                "thinking": thinking_content,
                "metrics": metrics,
            }
        except Exception as e:
            terminal = _request_terminal_result(
                cancel_event=cancel_event,
                deadline=request_deadline,
                terminal_outcome=terminal_outcome,
                source="fallback_exception",
            )
            if terminal is not None:
                return terminal
            logger.error(f"全模型回退推理失败: {e}")
            return {"response": "", "error": str(e)}


    def _run_full_model_inference_stream(self, prompt: str, **kwargs):
        """
        单机 PyTorch 流式推理 — 逐 token yield 事件字典，用于 SSE 推送。

        通过线程+队列包装 model_manager.chat_stream()，
        将文本 chunk 转为 {"token": text} 事件。

        Yields:
            {"token": str}       — 增量文本 chunk
            {"done": True, "response": str, "metrics": dict}
                                  — 完成信号
            {"done": True, "error": str}
                                  — 错误信号
        """
        import queue
        import threading as _thr

        request_deadline = kwargs.pop("_request_deadline", None)
        if request_deadline is not None and not isinstance(
            request_deadline, RequestDeadline,
        ):
            raise TypeError("request deadline must be a RequestDeadline")
        terminal_outcome = kwargs.pop(
            "_terminal_outcome", None,
        ) or TerminalOutcomeLatch()
        cancel_event = kwargs.pop("_cancel_event", None)
        if not isinstance(cancel_event, DeadlineCancelEvent):
            cancel_event = DeadlineCancelEvent(
                cancel_event, request_deadline,
            )
        stop_reason = request_stop_reason(
            cancel_event, request_deadline,
        )
        if stop_reason:
            terminal_outcome.decide(stop_reason, source="stream_entry")
            yield {
                "done": True,
                "error": (
                    "generation cancelled"
                    if stop_reason == REASON_GENERATION_CANCELLED
                    else "request deadline exceeded"
                ),
                "cancelled": stop_reason == REASON_GENERATION_CANCELLED,
                "reason_code": stop_reason,
            }
            return

        mgr = self._host
        if mgr and getattr(mgr, "is_pipeline_prepared", False):
            # ★ 2026-10-09（#78 缺口 2）：与非流式路径同样附上真实 reason_code + 人话说明。
            _code, _detail = "", ""
            try:
                _code, _detail = self._pipeline_unavailable_diagnostic()
            except Exception:  # pragma: no cover - 诊断失败不影响主流程
                pass
            _suffix = ""
            if _code:
                _suffix = f"（原因: {_code}"
                if _detail:
                    _suffix += f" —— {_detail}"
                _suffix += "）"
            yield {
                "done": True,
                "error": (
                    "当前模型仅以分布式流水线模式准备，禁止整模回退；"
                    "请等待从节点就绪或显式执行普通模型加载" + _suffix
                ),
                "reason_code": _code or "pipeline_full_model_fallback_forbidden",
            }
            return
        if not mgr or not mgr.is_loaded:
            yield {"done": True, "error": "模型未加载"}
            return
        if self.get_distributed_inference_enabled():
            yield {
                "done": True, "reason_code": "distributed_full_model_forbidden",
                "error": "分布式模式禁止整模回退；请先手动切换到本地推理模式。",
            }
            return

        try:
            callbacks = self._require_callbacks()
        except RuntimeError as e:
            yield {"done": True, "error": str(e)}
            return

        if request_deadline is None:
            self._inference_lock.acquire()
        else:
            remaining = request_deadline.require_remaining()
            if not self._inference_lock.acquire(timeout=remaining):
                terminal_outcome.decide(
                    REQUEST_DEADLINE_EXCEEDED,
                    source="stream_inference_lock",
                )
                yield {
                    "done": True,
                    "error": "request deadline exceeded",
                    "reason_code": REQUEST_DEADLINE_EXCEEDED,
                }
                return
        try:
            ensure_full = getattr(mgr, "ensure_full_model", None)
            if callable(ensure_full):
                ensure_full()
            if request_deadline is not None:
                request_deadline.require_remaining()
        except Exception as e:
            self._inference_lock.release()
            if request_deadline is not None and request_deadline.expired():
                terminal_outcome.decide(
                    REQUEST_DEADLINE_EXCEEDED,
                    source="stream_prepare",
                )
                yield {
                    "done": True,
                    "error": "request deadline exceeded",
                    "reason_code": REQUEST_DEADLINE_EXCEEDED,
                }
                return
            yield {"done": True, "error": f"完整模型恢复失败: {e}"}
            return

        # A direct llama.cpp layer engine may not expose ``ensure_full_model``.
        # Inspect its descriptor too so streaming fallback cannot execute a
        # partial GGUF as though it were a whole model.
        _layer_range = getattr(mgr, "layer_range", None)
        _desc = {}
        try:
            _get_desc = getattr(mgr, "get_pipeline_descriptor", None)
            if callable(_get_desc):
                _candidate = _get_desc() or {}
                if isinstance(_candidate, dict):
                    _desc = _candidate
        except Exception:  # noqa: BLE001 - unavailable descriptor stays fail-closed
            _desc = {}
        _partial = bool(
            _desc.get("assignment_layer_range")
            or _desc.get("partial_assignment")
            or _desc.get("loaded_artifact")
        )
        _lm_head = _desc.get("lm_head") if "lm_head" in _desc else None
        if _layer_range is not None or _partial or _lm_head is False:
            self._inference_lock.release()
            yield {
                "done": True,
                "error": (
                    "refusing_full_model_fallback_with_partial_artifact: "
                    f"layer_range={_layer_range} partial={_partial} lm_head={_lm_head};"
                    " master holds a partial artifact; refusing full-model fallback"
                ),
            }
            return

        max_new_tokens = kwargs.pop('max_new_tokens', 512)
        temperature = kwargs.pop('temperature', 0.7)
        top_p = kwargs.pop('top_p', 0.9)
        show_thinking = bool(kwargs.pop('show_thinking', False))
        # ★ 2026-09-19：深度思考**开关**（同另一处路径；None ⇒ 不干预，沿用模板默认）。
        enable_thinking = kwargs.pop('enable_thinking', None)
        messages = kwargs.pop("messages", None) or [{"role": "user", "content": prompt}]
        engine_name = backend_id_for(mgr, default="pytorch") or "pytorch"
        try:
            model_prompt = callbacks.build_model_chat_prompt(mgr.tokenizer, messages)
            # ★ 2026-10-07（真机复验根因，与 Route-A 同源）：判据必须是「模板注入的思考块
            #   **尚未闭合**」。旧判据只看 `"<think>"` 是否出现，而 qwen3-5-2b 模板在
            #   `enable_thinking` 非 true 时注入的是**已闭合**的 `'<think>\n\n</think>\n\n'`
            #   ⇒ 生成段只含正文、永远等不到 `</think>`，正文被整段丢掉（真机实测：PyTorch
            #   单机流式档 180s 超时且回答为空）。
            native_thinking_prompt = native_thinking_suppression_required(
                show_thinking, model_prompt,
            )
        except Exception:
            native_thinking_prompt = bool(
                engine_name == "llama_cpp"
                and getattr(mgr, "_chat_template", "") == "qwen3_chat_v1"
                and not getattr(mgr, "_thinking_controlled", False)
            )

        q = queue.Queue()
        full_text_parts = []
        error_info = [None]
        metrics_info = [{}]
        worker_started = False

        def _run():
            try:
                t0 = time.time()
                token_count = 0
                for chunk in mgr.chat_stream(
                    messages=messages,
                    max_tokens=max_new_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    show_thinking=show_thinking,
                    enable_thinking=enable_thinking,
                    _cancel_event=cancel_event,
                ):
                    if chunk:
                        full_text_parts.append(chunk)
                        token_count += 1
                        q.put({"token": chunk})
                elapsed = time.time() - t0
                metrics_info[0] = {
                    "mode": "single_streaming",
                    "engine": engine_name,
                    "chunks": token_count,
                    "elapsed_seconds": round(elapsed, 3),
                }
            except Exception as e:
                logger.error(f"单机流式推理异常: {e}", exc_info=True)
                error_info[0] = str(e)
            finally:
                q.put(None)  # sentinel
                self._inference_lock.release()

        worker = _thr.Thread(target=_run, name="full-model-stream", daemon=True)
        suppress_thinking = bool(native_thinking_prompt and not show_thinking)
        visible_buffer = ""
        terminal_reason = ""
        try:
            worker.start()
            worker_started = True
            while True:
                terminal_reason = request_stop_reason(
                    cancel_event, request_deadline,
                )
                if terminal_reason:
                    terminal_outcome.decide(
                        terminal_reason, source="stream_execute",
                    )
                    cancel_event.set()
                    break
                wait_timeout = 0.1
                if request_deadline is not None:
                    remaining = request_deadline.remaining()
                    if remaining <= 0:
                        terminal_reason = REQUEST_DEADLINE_EXCEEDED
                        terminal_outcome.decide(
                            terminal_reason, source="stream_execute",
                        )
                        cancel_event.set()
                        break
                    wait_timeout = min(
                        wait_timeout, remaining,
                    )
                try:
                    event = q.get(timeout=wait_timeout)
                except queue.Empty:
                    continue
                if event is None:
                    break
                chunk = event.get("token", "")
                if suppress_thinking:
                    visible_buffer += chunk
                    marker = visible_buffer.lower().find("</think>")
                    if marker >= 0:
                        visible = visible_buffer[marker + len("</think>"):]
                        suppress_thinking = False
                        visible_buffer = ""
                        if visible:
                            yield {"token": visible}
                else:
                    yield event
        finally:
            cancel_event.set()
            if not worker_started:
                self._inference_lock.release()
            elif not worker.is_alive():
                worker.join(timeout=0)

        if terminal_reason:
            yield {
                "done": True,
                "error": (
                    "generation cancelled"
                    if terminal_reason == REASON_GENERATION_CANCELLED
                    else "request deadline exceeded"
                ),
                "cancelled": terminal_reason == REASON_GENERATION_CANCELLED,
                "reason_code": terminal_reason,
            }
            return

        raw_response_text = "".join(full_text_parts)
        response_text, thinking_content = callbacks.format_model_response(
            raw_response_text,
            show_thinking,
            native_thinking_prompt=native_thinking_prompt,
        )
        if error_info[0]:
            yield {
                "done": True,
                "error": error_info[0],
                "response": response_text,
                "thinking": thinking_content,
                "metrics": metrics_info[0],
            }
        else:
            yield {
                "done": True,
                "response": response_text,
                "thinking": thinking_content,
                "metrics": metrics_info[0],
            }


    def _get_pipeline_status(self) -> dict:
        """
        获取流水线模式状态（供前端展示）。

        Returns:
            {
                "available": bool,       # 条件是否满足（PyTorch + 分布式 + 有从节点）
                "active": bool,          # 当前是否可用（所有节点在线）
                "degraded": bool,        # 降级模式（部分从节点离线）
                "worker_count": int,     # 流水线从节点总数
                "online_worker_count": int,  # 在线从节点数
                "engine_compatible": bool,   # 引擎是否兼容（PyTorch）
                "workers": [             # 各从节点详情
                    {node_id, online, layer_range, has_embedding, has_lm_head}
                ],
            }
        """
        # 检查引擎兼容性
        mgr = self._host
        engine_ok = (
            mgr is not None
            and (
                getattr(mgr, 'is_loaded', False)
                or getattr(mgr, 'is_pipeline_prepared', False)
            )
            and runtime_supports(mgr, Capability.FORWARD_LAYERS)
        )

        # 获取分层配置
        layer_info = self.get_layer_assignments()
        workers = [
            a for a in layer_info.get("assignments", [])
            if a.get("node_id") != "master"
        ]
        workers.sort(key=lambda a: a.get("start_layer", 0))

        readiness = self._get_pipeline_readiness()
        readiness_by_node = {
            item["node_id"]: item for item in readiness.get("workers", [])
        }
        # Read the connection state from the master-side TCPServer directly.
        # Readiness intentionally returns no worker details while the restart
        # recovery fence is active, so falling back to ``detail.get(...,
        # False)`` would report every live peer as disconnected and obscure
        # the actual recovery state.
        connected_ids = self._connected_client_ids()
        worker_status = []
        online_count = 0
        with self._nodes_lock:
            nodes_snapshot = dict(self.nodes)
        for w in workers:
            nid = w["node_id"]
            node = nodes_snapshot.get(nid)
            is_online = node.is_available() if node else False
            if is_online:
                online_count += 1
            detail = readiness_by_node.get(nid, {})
            worker_status.append({
                "node_id": nid,
                "online": is_online,
                "tcp_connected": nid in connected_ids,
                "heartbeat_age_seconds": detail.get("heartbeat_age_seconds"),
                "layer_ready": detail.get("layer_ready", False),
                "layer_status": detail.get("layer_status", "not_configured"),
                "layer_error": detail.get("layer_error", ""),
                "model_id": detail.get("model_id", ""),
                "layer_range": [w.get("start_layer", 0), w.get("end_layer", 24)],
                "has_embedding": w.get("has_embedding", False),
                "has_lm_head": w.get("has_lm_head", False),
            })

        distributed_enabled = self.get_distributed_inference_enabled()
        available = (
            engine_ok
            and self._runtime_setting('RUN_MODE') == "distributed"
            and self._effective_role() == "master"
            and len(workers) > 0
            and distributed_enabled
        )
        active = available and readiness.get("ready", False)
        degraded = available and not active and online_count > 0

        if not distributed_enabled:
            reason_code = "distributed_disabled"
            reason = "分布式推理开关已关闭"
        elif self._runtime_setting('RUN_MODE') != "distributed":
            reason_code = "not_distributed_mode"
            reason = "当前不是 distributed 运行模式"
        elif self._effective_role() != "master":
            reason_code = "not_master"
            reason = "当前节点不是主节点"
        elif not engine_ok:
            reason_code = "engine_not_pytorch"
            reason = "主节点必须加载 PyTorch 引擎模型才能进行模型层拆分"
        else:
            reason_code = readiness.get("reason_code", "unknown")
            reason = readiness.get("reason", "流水线状态未知")

        return {
            "available": available,
            "active": active,
            "degraded": degraded,
            "worker_count": len(workers),
            "online_worker_count": online_count,
            "engine_compatible": engine_ok,
            "distributed_enabled": distributed_enabled,
            "readiness_reason_code": reason_code,
            "readiness_reason": reason,
            "workers": worker_status,
            # ★ A1 / X 档（Y 档第一条）：最近一次 relay 段委托的真实指标（没走过则为空 dict）。
            #   取自末节点 `LAYER_RESULT.metrics` —— worker 侧只在**真走 relay 分支**时才写这五个键，
            #   所以空 dict 就等价于"本次没走 relay"，不会误报。
            "relay": dict(getattr(self, "_last_relay_metrics", None) or {}),
        }
