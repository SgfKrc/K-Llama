"""Distributed inference and pipeline methods mixed into Scheduler."""

from __future__ import annotations

import logging
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
)
# ★ #31 M2：层流水线支持的架构**单一事实来源**（此前在 4 处各写了一份 `{"qwen","qwen2"}`）
from pipeline_model_descriptor import PIPELINE_RUNTIME_MODEL_TYPES


def native_thinking_suppression_required(show_thinking: bool, model_prompt: str) -> bool:
    """★ 2026-10-07（真机复验根因）：是否需要抑制「模型原生思考」的外露。

    Route-A stage 链路此前只判 `"<think" in model_prompt[-128:]` —— 把**模板给模型的指令**
    当成了「模型已进入思考」的证据。`qwen3-5-2b` 的 chat template 在 `enable_thinking` 非 true
    时注入的是**已闭合**的 `'<think>\\n\\n</think>\\n\\n'`（见
    `models/qwen3-5-2b/tokenizer_config.json` 的 `chat_template`），生成段因此只含正文、
    永远等不到 `</think>` ⇒ 流式 token 全被吞进缓冲（真机 `tokens=0`），非流式则被
    `_format_model_response` 判成空正文（「流水线返回空响应」）。

    正确判据是「模板注入的思考块**尚未闭合**」：只有那种情况下，生成文本里的思考才需要
    等到 `</think>` 之后再外露。
    """
    tail = (model_prompt or "")[-128:].lower()
    return bool(not show_thinking and "<think" in tail and "</think>" not in tail)
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
from scheduler_types import PreemptState, WORKER_HEARTBEAT_MAX_AGE
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
    output: object, *, hidden_spec: dict,
):
    """Map a v3 StageResult to the old pipeline step's explicit value shape."""
    if not isinstance(output, dict):
        raise ValueError("layer stage result must be an object")
    n_tokens = int(hidden_spec.get("n_tokens", 0) or 0)
    n_embd = int(hidden_spec.get("n_embd", 0) or 0)
    if n_tokens < 1 or n_embd < 1:
        raise ValueError("hidden_spec dimensions must be positive")
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
    return {"kind": "token", "token_argmax": token}


def _assert_layer_stage_offer_fits_frame(
    *, node_id: str, n_tokens: int, n_embd: int, dtype: str = "float32",
    chunked_input: bool = False,
) -> None:
    """★ 2026-10-07（DIST-NEXT-2）：dispatch 前的层段 wire 大小预检。

    抽出为模块级纯函数（不依赖 `Scheduler` 状态），使两侧共用同一判据并可独立单测。

    ★ 2026-10-08（DIST-NEXT-2b）：`chunked_input=True` 表示目标 worker 声明了
    `stage_chunked_input` —— 此时**不得**在此拒绝。分片发生在 provider 层
    （`task_worker_adapter._maybe_send_stage_chunks`，先发 `chunk_count` 条 `stage_chunk`
    再发 offer），而本预检在 provider **之前**，无条件拒绝会让分片路径**永远走不到**：
    实测 ≈2475 tokens 的 prompt 得到 `route_a_stage_frame_too_large:…:wire=24160940`
    而非分片。未声明分片的对端仍保持 fail-closed。
    """
    wire_bytes = hidden_wire_bytes(n_tokens, n_embd, dtype)
    budget_bytes = stage_payload_budget_bytes()
    if hidden_fits_stage_frame(n_tokens, n_embd, dtype):
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
    if control is None:
        return False
    try:
        status = control.status(role="master")
    except Exception:
        logger.debug("查询 task worker 分片能力失败: node=%s", node_id, exc_info=True)
        return False
    worker_ids = [
        item.get("node_id")
        for item in ((status or {}).get("workers") or [])
        if isinstance(item, dict)
    ]
    for worker in (status or {}).get("workers", []) or []:
        if not isinstance(worker, dict) or worker.get("node_id") != node_id:
            continue
        capabilities = worker.get("capabilities")
        declared = (
            isinstance(capabilities, dict)
            and capabilities.get("stage_chunked_input") is True
        )
        # ★ 2026-10-08（诊断，定位后降级为 debug）：分辨「worker 不在控制面」与
        #   「hello 没带该键」——2b 真机回归已用它确认 Android 侧 `declared=True`
        #   且分片真实生效（`task_worker_stage_chunks_sent chunks=18`）。
        logger.debug(
            "event=stage_chunked_probe node=%s declared=%s caps_keys=%s worker_ids=%s",
            node_id, declared,
            sorted(capabilities.keys()) if isinstance(capabilities, dict) else None,
            worker_ids,
        )
        return declared
    logger.debug(
        "event=stage_chunked_probe node=%s declared=False worker_ids=%s",
        node_id, worker_ids,
    )
    return False


class SchedulerPipelineMixin:
    def _pipeline_lifecycle_snapshot_locked(self) -> dict | None:
        """Return a bounded JSON-safe transaction record for restart recovery."""
        transaction = self._pipeline_load_transaction
        cleanup_ledger = getattr(self, "_pipeline_model_cleanup_ledger", {})
        if not isinstance(transaction, dict) and not cleanup_ledger:
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
                "node_id": str(record.get("node_id", "") or ""),
                "candidate_config_id": str(
                    record.get("candidate_config_id", "") or ""
                ),
                "assignment_id": str(record.get("assignment_id", "") or ""),
                "generation": int(record.get("generation", 0) or 0),
                "action": str(record.get("action", "") or ""),
                "created_at": float(record.get("created_at", 0.0) or 0.0),
                "payload": dict(payload) if isinstance(payload, dict) else {},
            })
        return {
            "schema_version": _PIPELINE_LIFECYCLE_SCHEMA_VERSION,
            "config_id": str(transaction.get("config_id", "") or ""),
            "generation": int(transaction.get("generation", 0) or 0),
            "phase": str(transaction.get("phase", "") or ""),
            "model_id": str(plan.get("model_id", "") or ""),
            "model_type": str(plan.get("model_type", "") or ""),
            "model_sha256": str(plan.get("model_sha256", "") or ""),
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
            "cleanup_ledger": cleanup_records,
            "updated_at": time.time(),
        }

    def _persist_pipeline_lifecycle_locked(self) -> None:
        """Persist lifecycle state while the transaction lock is held."""
        if (
            not getattr(self, "_running", False)
            or self._effective_role() != "master"
        ):
            return
        state = self._pipeline_lifecycle_snapshot_locked()
        if state is None:
            return
        try:
            from local_store import set_local_setting

            set_local_setting(_PIPELINE_LIFECYCLE_STATE_KEY, state)
            self._pipeline_lifecycle_persist_ok = True
            if self._pipeline_recovery_pending and self._pipeline_recovery_failure in {
                "pipeline_lifecycle_persist_failed",
                "pipeline_recovery_state_unavailable",
                "pipeline_recovery_state_invalid",
            }:
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
        if action not in {"abort", "finalize"}:
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
        ledger[cleanup_id] = record
        return record

    def _complete_pipeline_model_cleanup(
        self, cleanup_id: str, *, persist: bool = True,
    ) -> bool:
        with self._layer_config_lock:
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
        if node_id == "master":
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
            server.send_layer_config(node_id, dict(record.get("payload", {})))
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

    def _ack_pipeline_model_cleanup(self, client_id: str, data: dict) -> bool:
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
        try:
            from local_store import get_local_setting

            raw = get_local_setting(_PIPELINE_LIFECYCLE_STATE_KEY, None)
        except Exception:
            self._pipeline_lifecycle_persist_ok = False
            self._pipeline_recovery_pending = True
            self._pipeline_recovery_failure = "pipeline_recovery_state_unavailable"
            logger.warning("failed to load pipeline lifecycle state", exc_info=True)
            return
        self._pipeline_lifecycle_persist_ok = True
        if raw is None:
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
                continue
            if (
                not cleanup_id
                or not node_id
                or not candidate_config_id
                or action not in {"abort", "finalize"}
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
            legacy_identity = schema_version < 4
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
                if action == "finalize":
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

    def _restore_pipeline_model_for_recovery(self) -> bool:
        """Restore distributed-only model metadata before workers reconnect.

        The lifecycle record intentionally contains no absolute path. Resolve
        the persisted model identity through the local model registry (also
        accepting the historical directory-name identity), then prepare only
        metadata. No full model is materialized here.
        """
        if not self._pipeline_recovery_pending:
            return True
        state = self._pipeline_recovery_state
        if not isinstance(state, dict):
            return False
        model_id = str(state.get("model_id", "") or "").strip()
        if not model_id:
            self._pipeline_recovery_failure = "pipeline_recovery_model_identity_missing"
            return False
        if self._get_active_pipeline_model_info():
            return True

        model_paths: list[str] = []
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
                    # The persisted digest is an expectation, not evidence about
                    # the bytes present after this restart. Recompute locally.
                    model_sha256=None,
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
        self._pipeline_recovery_pending = False
        self._pipeline_recovery_failure = ""

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
        if (
            (expected_id and actual_id != expected_id)
            or (expected_type and actual_type != expected_type)
            or (expected_sha256 and actual_sha256 != expected_sha256)
        ):
            self._pipeline_recovery_failure = "pipeline_recovery_model_mismatch"
            logger.error(
                "pipeline recovery model mismatch: expected=%s/%s actual=%s/%s",
                expected_id, expected_type, actual_id, actual_type,
            )
            return False
        return True

    def _maybe_finish_pipeline_recovery(self) -> None:
        """Release the restart fence only after the fresh generation is ready."""
        if not self._pipeline_recovery_pending:
            return
        if not self._pipeline_lifecycle_persist_ok:
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
        require_distributed: bool = True, on_commit=None,
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
                stage_assignments[nid] = {
                    **dict(a),
                }
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
            if node_id not in assignments
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
            dict(assignments)
            if candidate is not None
            else {**assignments, **releases}
        )
        if candidate is not None:
            # Candidate assignments exist beside the active registry until
            # global commit. Allocate their wire identity before prepare so
            # prepare, commit, retry and promotion all retain one ID.
            for payload in configs.values():
                payload.setdefault("assignment_id", new_assignment_id())
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
                # The solver's plan is retained, while only legacy workers
                # participate in the prepare/commit ACK transaction.
                capacity_plan = dict(capacity_plan)
                capacity_plan["assignments"] = [
                    (
                        stage_assignments.get(item.get("node_id"), item)
                        if item.get("node_id") in stage_assignments else item
                    )
                    for item in capacity_plan.get("assignments", [])
                ]
            with self._layer_config_lock:
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
                    "worker_ids": set(assignments),
                    "prepared_nodes": set(),
                    "candidate_id": (
                        str(candidate.get("candidate_id", "") or "")
                        if isinstance(candidate, dict) else ""
                    ),
                    "started_at": time.monotonic(),
                    "deadline": time.monotonic() + PIPELINE_MODEL_SYNC_TIMEOUT,
                    "previous_control_state": previous_control_state,
                    "expected": {
                        node_id: dict(value)
                        for node_id, value in assignments.items()
                    } if candidate is not None else {},
                    "acks": {},
                    "retry_state": {
                        node_id: {
                            "attempts": 1,
                            "next_retry": time.monotonic() + 5.0,
                        }
                        for node_id in assignments
                    } if candidate is not None else {},
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
        if capacity_plan is not None and stage_assignments and not assignments:
            # No legacy worker has an ACK to drive the transaction forward.
            # Commit the master's local prefix explicitly, then expose the
            # stage-only plan as active.
            self._commit_pipeline_load_transaction(config_id)
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
        from task_provider import ModelIdentity

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
                current_artifact = next((
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
                if current_artifact is None:
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
        middle_channel: str = "extract_hidden",
        seq_ids: list[int] | None = None,
        positions: list[int] | None = None,
        cancel_event: threading.Event | None = None,
        request_deadline: RequestDeadline | None = None,
    ) -> dict:
        """Execute one Route-A v3 layer stage and map its result to a step.

        This helper is intentionally independent from the legacy token loop;
        callers must own the surrounding prefill/decode/KV state machine.
        """
        from task_provider import StageAttempt, StageRequest
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
        _assert_layer_stage_offer_fits_frame(
            node_id=str(node_id), n_tokens=n_tokens, n_embd=n_embd,
            # ★ 2026-10-08（DIST-NEXT-2b）：对端声明 `stage_chunked_input` ⇒ 超预算的
            #   hidden 交由 provider 切 `stage_chunk` 分片发送，不在此提前拒绝。
            chunked_input=(
                False
                if _fits_single_frame
                else _node_declares_stage_chunked_input(
                    getattr(self, "_task_worker_control", None), str(node_id),
                )
            ),
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
        if seq_ids is not None:
            stage_fields["seq_ids"] = list(seq_ids)
        if positions is not None:
            stage_fields["positions"] = list(positions)
        # ★ 2026-10-03：把每步的 hidden 摘要打出来。跨机 D→L 的分叉定位需要与 relay
        #   路径（同层范围、同 prompt）的输出**逐位对照**，而此前链路上没有任何可对照
        #   的中间量 —— 只能看到"最终 token 不同"，无法判断差异出在 master 段还是设备段。
        logger.info(
            "Route-A stage handoff: node=%s stage=%s tokens=%d hidden_sha256=%s",
            node_id, stage_id, n_tokens, stage_fields["hidden_sha256"],
        )
        request = StageRequest(
            workflow_id=str(workflow_id),
            request_id=str(request_id),
            stage_id=str(stage_id),
            stage_type="layer_forward",
            provider_id=remote_provider_id(str(node_id)),
            dependencies={},
            root_input={
                "hidden_f32": base64.b64encode(raw).decode("ascii"),
                "context_size": int(context_size),
                "pos_base": int(pos_base),
                "want_hidden": bool(want_hidden),
            },
            model_identity=model_identity,
            stage_fields=stage_fields,
            runtime_context={"pipeline_route": "route_a"},
        )
        provider = self._ensure_remote_task_worker_provider(str(node_id))
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
        lease_expires_at = time.time() + 60.0
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
        try:
            result = provider.execute(
                attempt,
                reservation,
                cancel_event or threading.Event(),
            )
            return _layer_stage_result_to_pipeline_value(
                result.output, hidden_spec=hidden_spec,
            )
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
                self._tcp_server.send_layer_config(node_id, config)
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
                self._tcp_server.send_layer_config(node_id, config)
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
        self, config_id: str, reason_code: str, reason: str = "",
    ) -> None:
        """Abort one candidate and restore the last committed control state."""
        cleanup_records = []
        with self._layer_config_lock:
            transaction = self._pipeline_load_transaction
            if not transaction or transaction.get("config_id") != config_id:
                return
            worker_ids = set(transaction.get("worker_ids", set()))
            candidate_id = str(transaction.get("candidate_id", "") or "")
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
        if candidate_id:
            for record in cleanup_records:
                self._dispatch_pipeline_model_cleanup(record)
        else:
            abort_materialization = getattr(
                self._host, "abort_pipeline_materialization", None
            )
            if callable(abort_materialization):
                try:
                    abort_materialization()
                    self._host.model_loaded = False
                except Exception:
                    logger.warning("主节点回滚流水线层段失败", exc_info=True)
            abort_id = uuid.uuid4().hex
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


    def _invalidate_pipeline_load_transaction(
        self, reason_code: str = "pipeline_model_changed", reason: str = "",
    ) -> None:
        """Fence a pipeline transaction before replacing the local model."""
        cleanup_records = []
        with self._layer_config_lock:
            transaction = self._pipeline_load_transaction
            if transaction and transaction.get("phase") not in {
                "aborted", "rejected", "invalidated",
            }:
                transaction["phase"] = "invalidated"
                transaction["reason_code"] = reason_code
                transaction["reason"] = reason
                self._layer_config_generation = max(
                    self._layer_config_generation + 1,
                    time.time_ns(),
                )
                cleanup_records.extend(
                    self._record_pipeline_candidate_abort_cleanup_locked(
                        transaction,
                        generation=self._layer_config_generation,
                        reason_code=reason_code,
                    )
                )
                self._persist_pipeline_lifecycle_locked()
            self._active_pipeline_capacity_plan = None
            self._prepared_layer_configs.clear()
            self._layer_config_inflight.clear()
        for record in cleanup_records:
            self._dispatch_pipeline_model_cleanup(record)


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

        master_ids = {"master", self.get_effective_node_id()}
        local_assignment = next((
            item for item in plan.get("assignments", [])
            if item.get("node_id") in master_ids
        ), None)
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
                    materialize_candidate(
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
                    )
            else:
                # Compatibility for legacy callers that do not own an isolated
                # ModelHost candidate yet.
                if local_assignment is not None and not getattr(
                    self._host, "is_pipeline_prepared", False
                ):
                    prepare_local = getattr(
                        self._host, "prepare_pipeline_model", None,
                    )
                    local_model_path = (
                        getattr(self._host, "_full_model_path", None)
                        or getattr(self._host, "_model_path", None)
                        or getattr(self._host, "model_path", None)
                    )
                    if callable(prepare_local) and local_model_path:
                        prepare_local(
                            model_id=str(plan.get("model_id", "") or ""),
                            model_path=str(local_model_path),
                            quant_type=getattr(self._host, "quant_type", None),
                            layer_range=(
                                int(local_assignment["start_layer"]),
                                int(local_assignment["end_layer"]),
                            ),
                            model_sha256=None,
                        )
                prepare_tokenizer = getattr(
                    self._host, "prepare_pipeline_tokenizer", None,
                )
                if callable(prepare_tokenizer):
                    prepare_tokenizer()
                if local_assignment is not None:
                    self._host.load_layer_range(
                            int(local_assignment["start_layer"]),
                            int(local_assignment["end_layer"]),
                        has_embedding=bool(
                            local_assignment.get("has_embedding")
                        ),
                        has_lm_head=bool(
                            local_assignment.get("has_lm_head")
                        ),
                        model_path=getattr(
                            self._host, "_full_model_path", None,
                        ),
                        quant_type=getattr(self._host, "quant_type", None),
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

        commit_configs = {}
        for node_id, item in expected.items():
            if not item or item.get("release"):
                continue
            item["phase"] = "commit"
            commit_configs[node_id] = item
        if not commit_configs:
            stage_only = bool(
                not worker_ids
                and any(
                    item.get("execution") == "stage_offer_v3"
                    for item in plan.get("assignments", [])
                )
            )
            if stage_only:
                with self._layer_config_lock:
                    transaction = self._pipeline_load_transaction
                    if (
                        transaction
                        and transaction.get("config_id") == config_id
                        and transaction.get("phase") == "committing_local"
                    ):
                        transaction["phase"] = "committing_global"
                        self._persist_pipeline_lifecycle_locked()
                self._complete_pipeline_load_transaction(config_id)
                return
            self._abort_pipeline_load_transaction(
                config_id, "pipeline_commit_workers_missing",
                "prepared worker set disappeared before commit",
            )
            return
        with self._layer_config_lock:
            transaction = self._pipeline_load_transaction
            if transaction and transaction.get("config_id") == config_id:
                transaction["phase"] = "committing"
                if candidate_id:
                    transaction["expected"] = {
                        node_id: dict(value)
                        for node_id, value in commit_configs.items()
                    }
                    transaction["retry_state"] = {
                        node_id: {
                            "attempts": 1,
                            "next_retry": time.monotonic() + 5.0,
                        }
                        for node_id in commit_configs
                    }
                self._persist_pipeline_lifecycle_locked()
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

                # Candidate commit is reversible until finalize.  Cross the
                # reshard CAS only after the local runtime can be restored on
                # failure; the coordinator has no reverse-epoch operation.
                reshard_result = self._commit_ready_pipeline_reshard(active_plan)
                if reshard_result is False:
                    raise RuntimeError("pipeline_reshard_commit_failed")

                with self._layer_config_lock:
                    transaction = self._pipeline_load_transaction
                    if (
                        not transaction
                        or transaction.get("config_id") != config_id
                        or transaction.get("phase") != "committing_global"
                    ):
                        raise RuntimeError(
                            "pipeline transaction superseded before publication"
                        )
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
                                self._assignment_connection_generation(node_id)
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
                self._tcp_server.send_layer_config(node_id, assignment)
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
                for node_id, expected in transaction.get("expected", {}).items():
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
                self._tcp_server.send_layer_config(node_id, assignment)
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
                ):
                    raise RuntimeError("commit 未命中同代际 prepared 记录")

            local_sha256 = ""
            local_model_path = None
            if phase == "commit" and plan_id:
                local_sha256 = str(prepared.get("model_sha256", "") or "")
                local_model_path = prepared.get("model_path")
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
                prepared_record = {
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


    def _handle_layer_worker_opt_out(self, client_id: str, msg: dict) -> None:
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
        if self._is_relay_host(client_id):
            with self._layer_config_lock:
                self._pipeline_worker_opt_out.discard(client_id)
            logger.info(
                "忽略 relay 节点的本地分层退出通知: node=%s", client_id,
            )
            self.push_layer_config_to_clients()
            return
        with self._layer_config_lock:
            self._pipeline_worker_opt_out.add(client_id)
        self._clear_layer_config_state(client_id)
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
                input_ids = self._scheduler_facade_global('torch').tensor(data["input_ids"], dtype=self._scheduler_facade_global('torch').long)
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
                attention_mask = self._scheduler_facade_global('torch').tensor(data["attention_mask"], dtype=self._scheduler_facade_global('torch').long)
                if attention_mask.dim() == 1:
                    attention_mask = attention_mask.unsqueeze(0)

            if "position_ids" in data and data["position_ids"] is not None:
                position_ids = self._scheduler_facade_global('torch').tensor(data["position_ids"], dtype=self._scheduler_facade_global('torch').long)
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
                        round(self._scheduler_facade_global('torch').cuda.memory_allocated() / (1024**3), 2)
                        if self._scheduler_facade_global('torch').cuda.is_available() else 0
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


    def _run_route_a_stage_pipeline(
        self,
        *,
        prompt: str,
        max_new_tokens: int,
        temperature: float,
        top_p: float,
        session_id: str | None,
        messages: list | None,
        show_thinking: bool,
        routing_preference: str,
        stage_nodes: list[dict],
        master_assignment: dict,
        stop: list[str] | str | None = None,
        enable_thinking: bool | None = None,
        _stream_callback=None,
        _cancel_event: threading.Event | None = None,
        _request_deadline: RequestDeadline | None = None,
        _terminal_outcome: TerminalOutcomeLatch | None = None,
    ) -> dict:
        """Run a Route-A chain made only of v3 Android layer stages.

        The legacy LAYER_FORWARD loop cannot share its KV ownership or wire
        contract with a v3 task worker.  This path therefore owns the whole
        prefill/decode sequence: the master computes its prefix, then each
        Android assignment receives a v3 hidden handoff.  Mixed legacy/v3
        chains remain rejected by the caller until they have one KV contract.
        """
        mgr = self._host
        if not master_assignment or not master_assignment.get("layers_count", 0):
            return {
                "response": "",
                "error": "route_a_requires_master_prefix",
            }
        if not mgr or not getattr(mgr, "tokenizer", None):
            return {"response": "", "error": "route_a_tokenizer_not_ready"}

        ok, readiness_error = self._verify_pipeline_readiness(stage_nodes)
        if not ok:
            return {"response": "", "error": readiness_error}

        from task_provider import ModelIdentity

        stage_model_identities = []
        for assignment in stage_nodes:
            raw_identity = assignment.get("stage_model_identity")
            if not isinstance(raw_identity, dict):
                legacy_identity = self._route_a_stage_model_identity(
                    str(assignment.get("node_id", "") or ""), assignment,
                )
                if legacy_identity is None:
                    return {
                        "response": "",
                        "error": (
                            "route_a_stage_model_identity_unavailable:"
                            f"{assignment.get('node_id', '')}"
                        ),
                    }
                stage_model_identities.append(legacy_identity)
                continue
            try:
                stage_model_identities.append(ModelIdentity(**raw_identity))
            except (TypeError, ValueError):
                return {
                    "response": "",
                    "error": (
                        "route_a_stage_model_identity_invalid:"
                        f"{assignment.get('node_id', '')}"
                    ),
                }

        callbacks = self._require_callbacks()
        try:
            model_identity = callbacks.active_task_graph_model_identity()
        except Exception as exc:
            logger.warning("Route-A model identity lookup failed", exc_info=True)
            model_identity = None
        if model_identity is None:
            return {"response": "", "error": "route_a_model_identity_unavailable"}

        terminal = _request_terminal_result(
            cancel_event=_cancel_event,
            deadline=_request_deadline,
            terminal_outcome=_terminal_outcome,
            source="route_a_prepare",
        )
        if terminal is not None:
            return terminal
        try:
            ensure_layer_range = getattr(mgr, "ensure_layer_range", None)
            if callable(ensure_layer_range):
                ensure_layer_range(
                    master_assignment["start_layer"],
                    master_assignment["end_layer"],
                    has_embedding=master_assignment.get("has_embedding", True),
                    has_lm_head=master_assignment.get("has_lm_head", True),
                )
            else:
                mgr.load_layer_range(
                    master_assignment["start_layer"],
                    master_assignment["end_layer"],
                    has_embedding=master_assignment.get("has_embedding", True),
                    has_lm_head=master_assignment.get("has_lm_head", True),
                )
        except Exception as exc:
            logger.error("Route-A master prefix load failed", exc_info=True)
            return {"response": "", "error": f"route_a_master_prefix_load_failed: {exc}"}
        terminal = _request_terminal_result(
            cancel_event=_cancel_event,
            deadline=_request_deadline,
            terminal_outcome=_terminal_outcome,
            source="route_a_prepare",
        )
        if terminal is not None:
            return terminal

        tokenizer = mgr.tokenizer
        chat_messages = messages or [{"role": "user", "content": prompt}]
        thinking_prompt = callbacks.thinking_system_prompt if show_thinking else None
        thinking_prefill = "思考\n" if show_thinking else None
        model_prompt = callbacks.build_model_chat_prompt(
            tokenizer,
            chat_messages,
            system_prompt=thinking_prompt,
            assistant_prefill=thinking_prefill,
            **({"enable_thinking": enable_thinking} if enable_thinking is not None else {}),
        )
        inputs = tokenizer(model_prompt, return_tensors="pt")
        input_ids = inputs["input_ids"]
        attention_mask = inputs.get("attention_mask")
        prompt_len = int(input_ids.shape[1])
        torch = self._scheduler_facade_global("torch")
        config = getattr(getattr(mgr, "model", None), "config", None)
        context_size = int(
            getattr(config, "max_position_embeddings", None)
            or getattr(config, "max_seq_len", None)
            or 2048
        )

        task_id = uuid.uuid4().hex[:12]
        worker_ids = [node["node_id"] for node in stage_nodes]
        # ★ 产品裁定「分布式可用必须 fail-closed」：成功请求必须带非空
        #   `claimed_layers`（与 A1 同口径：远端实际承了哪段层）。此前 A3 恒缺
        #   该字段（它只在 `pipeline_capacity` 的 relay 零层条目里被填），按裁定
        #   就不能计为分布式成功。这里由 `stage_nodes` 的每段 start/end 派生：
        #   顶层给并集（判据只用非空 + 覆盖性），另给逐段明细供可观测性。
        _stage_ranges: list[tuple[int, int]] = []
        for _node in stage_nodes:
            _start = _node.get("start_layer")
            _end = _node.get("end_layer")
            if _start is None or _end is None:
                continue
            _stage_ranges.append((int(_start), int(_end)))
        _claimed_layers = (
            [min(r[0] for r in _stage_ranges), max(r[1] for r in _stage_ranges)]
            if _stage_ranges
            else []
        )
        pipeline_metrics = {
            "steps": [],
            "total_time_ms": 0,
            "kv_cache": True,
            "chain_topology": True,
            "engine": "distributed_pipeline",
            "execution_mode": "route_a_stage_offer_v3",
            "distributed_requested": True,
            "distributed_used": True,
            "fallback": False,
            "fallback_reason": "",
            "route": "master_pipeline_route_a",
            "task_id": task_id,
            "serving_node_id": self.get_effective_node_id(),
            "workers_used": worker_ids,
            "layer_assignments": stage_nodes,
            "claimed_layers": _claimed_layers,
            "layer_segments": [[s, e] for s, e in _stage_ranges],
            # ★ 2026-10-05（DIST-4）：把层配置代际写进**请求级** metrics。
            #   此前 `config_id` 只存在于内存 contract（`_pipeline_task_contracts`）
            #   与 `/api/cluster/pipeline-capacity`，响应与日志里都看不到 ⇒ 出问题时
            #   无法从单条请求回溯它用的是哪一代层配置。取值与本链的 contract 一致
            #   （`route_a:{task_id}`）。
            "config_id": f"route_a:{task_id}",
        }
        generated_ids: list[int] = []
        full_input_ids = input_ids
        new_token_id: int | None = None
        try:
            from llama_engine import _merge_stop_sequences

            merge_stops = getattr(mgr, "_merge_stop_sequences", _merge_stop_sequences)
            stop_sequences = merge_stops([stop] if isinstance(stop, str) else stop)
            get_eos = getattr(mgr, "_get_generation_eos_token_ids", None)
            native_eos = getattr(tokenizer, "eos_token_id", None)
            eos_ids = _route_a_eos_ids(native_eos)
            if callable(get_eos):
                eos_ids.update(_route_a_eos_ids(get_eos(stop_sequences)))
        except Exception as exc:
            return {"response": "", "error": f"route_a_stop_configuration_invalid: {exc}"}
        if not eos_ids:
            if not stop_sequences:
                return {"response": "", "error": "route_a_stop_configuration_invalid:no_eos_or_text_stop"}
            diagnostic = "route_a_eos_unavailable_text_stop"
            logger.warning("%s eos=%r stops=%d", diagnostic, native_eos, len(stop_sequences))
            pipeline_metrics["stop_diagnostic"] = diagnostic
        pipeline_metrics["eos_token_ids"] = sorted(eos_ids)
        raw_new_text = ""
        published_text_len = 0
        finish_reason = "length"
        stop_reason = "max_new_tokens"
        matched_stop = ""
        native_thinking_prompt = native_thinking_suppression_required(
            show_thinking, model_prompt,
        )
        suppress_native_thinking = native_thinking_prompt
        stream_buffer = ""
        t_pipeline_start = time.time()

        with self._pipeline_lock:
            self._pipeline_active_tasks.add(task_id)
            self._pipeline_task_contracts[task_id] = {
                "execution_protocol": "route_a_stage_v3",
                "config_id": f"route_a:{task_id}",
                "model_sha256": getattr(model_identity, "sha256", ""),
                "model_type": getattr(model_identity, "model_id", ""),
                "worker_ids": worker_ids,
                "last_node_id": worker_ids[-1],
                "current_step": -1,
            }
        pipeline_stack = getattr(self._pipeline_context, "stack", None)
        if pipeline_stack is None:
            pipeline_stack = []
            self._pipeline_context.stack = pipeline_stack
        pipeline_stack.append({
            "task_id": task_id,
            "pipeline_nodes": stage_nodes,
        })

        try:
            for step in range(max_new_tokens):
                step_start = time.time()
                terminal = _request_terminal_result(
                    cancel_event=_cancel_event,
                    deadline=_request_deadline,
                    terminal_outcome=_terminal_outcome,
                    source="route_a_step",
                )
                if terminal is not None:
                    return terminal
                with self._pipeline_lock:
                    contract = self._pipeline_task_contracts.get(task_id)
                    if contract is None:
                        return {"response": "", "error": "route_a_task_contract_expired"}
                    contract["current_step"] = step

                is_prefill = step == 0
                past_kv = None
                if not is_prefill:
                    with self._kv_cache_lock:
                        past_kv = self._kv_cache.get(task_id)
                    if past_kv is None:
                        return {
                            "response": "",
                            "error": f"route_a_decode_step_{step}_missing_kv",
                        }
                if is_prefill:
                    local_input_ids = input_ids
                elif hasattr(input_ids, "detach"):
                    # PyTorch 引擎：decode 步只喂新 token。
                    local_input_ids = torch.tensor([[new_token_id]], dtype=torch.long)
                else:
                    # llama.cpp / 去 torch 节点：载体是 numpy。这里若不判载体，
                    # `torch_runtime.require_torch()` 会抛 `ModuleNotFoundError`，
                    # 表现为 `route_a_stage_execution_failed: No module named 'torch'`。
                    import numpy as _np

                    local_input_ids = _np.array([[new_token_id]], dtype=_np.int64)
                # ★ 2026-10-09（接口税量化）：per-step 分解打点。此前只有三段**近似**值
                #   （相邻日志时间戳之差）—— 无法区分「master 本地算」与「等对端」，
                #   于是 54.5ms 的 master 段里到底多少是 GPU→CPU 同步纯属猜测。
                #   这三个时间戳把一步拆成：fwd（本地层前向）/ hidden（GPU→CPU 同步 +
                #   形状整理）/ stage（全部层段 offer，含网络往返 + 对端计算）。
                _perf_t_fwd = time.perf_counter()
                local_result = mgr.forward_layers(
                    input_ids=local_input_ids,
                    attention_mask=attention_mask if is_prefill else None,
                    past_key_values=past_kv,
                    use_cache=True,
                    apply_lm_head=False,
                )
                _perf_t_hid = time.perf_counter()
                if local_result.get("cache") is not None or local_result.get("past_key_values"):
                    with self._kv_cache_lock:
                        self._kv_cache[task_id] = _prefer_cache_state(local_result)
                hidden = local_result.get("hidden_states")
                if hidden is None:
                    return {"response": "", "error": "route_a_master_missing_hidden"}
                # hidden 的载体随引擎不同：PyTorch 给张量（要 `.detach().cpu()`），
                # llama.cpp 给 numpy `[tokens, n_embd]` f32（legacy 首段路径同款判断）。
                # 缺了这条分支，去 torch 的 master 会在这里抛
                # `'numpy.ndarray' object has no attribute 'detach'`，被外层收敛成
                # `route_a_stage_execution_failed` 后回退到全层主节点模式。
                if hasattr(hidden, "detach"):
                    hidden = hidden.detach().to(device="cpu", dtype=torch.float32).contiguous()
                else:
                    import numpy as _np

                    hidden = _np.ascontiguousarray(hidden, dtype=_np.float32)
                if hidden.ndim == 3 and int(hidden.shape[0]) == 1:
                    hidden = hidden.squeeze(0)
                if hidden.ndim != 2:
                    return {
                        "response": "",
                        "error": f"route_a_invalid_master_hidden_rank:{hidden.ndim}",
                    }
                n_tokens = int(hidden.shape[0])
                positions = (
                    list(range(n_tokens))
                    if is_prefill
                    else [prompt_len + step - 1] * n_tokens
                )
                current_hidden = hidden
                _perf_t_stage = time.perf_counter()
                stage_token = None
                for index, assignment in enumerate(stage_nodes):
                    last_stage = index == len(stage_nodes) - 1
                    # The exact physical identity was frozen by model preflight.
                    # Runtime readiness above still rejects capability drift, but
                    # the per-token loop no longer rescans worker capabilities.
                    stage_model_identity = stage_model_identities[index]
                    try:
                        stage_result = self._execute_layer_stage_offer(
                            node_id=assignment["node_id"],
                            assignment=assignment,
                            hidden_states=current_hidden,
                            model_identity=stage_model_identity,
                            # ★ 协议要求 `wf_` 前缀（`_WORKFLOW_ID = ^wf_[A-Za-z0-9_-]{8,96}$`）；
                            #   Route A 原先直接传裸 `task_id`（12 位 hex），会被 offer 校验拒掉。
                            workflow_id=f"wf_{task_id}",
                            request_id=f"{task_id}:step:{step}",
                            stage_id=f"{assignment['node_id']}:step:{step}",
                            context_size=context_size,
                            pos_base=0,
                            want_hidden=not last_stage,
                            # ★ 2026-10-03：层段接力必须走 **keep-head** 通道（末层输出，
                            #   `output_norm` **之前**）—— 协议里 `extract_hidden` 是旧默认，
                            #   会多一次 `output_norm`。传错通道会让跨机 D→L 从 decode 起
                            #   分叉（真机实测：首 token 一致、第 3 个 token 起偏）。
                            middle_channel="keep_head_layer_out",
                            seq_ids=[0] * n_tokens,
                            positions=positions,
                            cancel_event=_cancel_event,
                            request_deadline=_request_deadline,
                        )
                    except LayerStageFrameTooLarge as exc:
                        # ★ 2026-10-07（DIST-NEXT-2）：dispatch 前的不可恢复拒绝。
                        #   具名 reason（含所需/上限字节数与维度）直接回给调用方，
                        #   不再让请求走「发不出的 offer → 执行超时 → 回退」。
                        return {"response": "", "error": str(exc)}
                    if last_stage:
                        if stage_result.get("kind") != "token":
                            return {"response": "", "error": "route_a_tail_missing_token"}
                        stage_token = int(stage_result["token_argmax"])
                    else:
                        if stage_result.get("kind") != "hidden":
                            return {"response": "", "error": "route_a_middle_missing_hidden"}
                        current_hidden = stage_result["hidden_states"]
                        n_tokens = int(current_hidden.shape[0])
                _perf_t_end = time.perf_counter()
                # ★ 短格式：master.log 的每条消息在**写入端被截断到 ~119 字符**
                #   （实测：`Route-A stage handoff:` 也正好停在 119）⇒ 时间戳 + `request_id=`
                #   已占 75 字符，长字段会整段丢失。故这里用 `f=/h=/s=` 短名 + 整数毫秒。
                logger.info(
                    "perf step=%d f=%.0f h=%.0f s=%.0f",
                    step,
                    (_perf_t_hid - _perf_t_fwd) * 1000.0,
                    (_perf_t_stage - _perf_t_hid) * 1000.0,
                    (_perf_t_end - _perf_t_stage) * 1000.0,
                )
                if stage_token is None:
                    return {"response": "", "error": "route_a_stage_chain_empty"}
                new_token_id = stage_token
                # ★ 2026-10-07：逐 token 取证日志。此前 Route-A 生成循环**不打印任何 token**，
                #   导致「空响应」无法区分「没生成」与「生成的全是特殊 token 被
                #   skip_special_tokens 滤掉」。与 Android 侧（无逐 token 日志）配合时，
                #   这是唯一能定位数值分歧的观测点。
                logger.info(
                    "Route-A 生成 step=%d token=%d eos=%s text=%r",
                    step, int(new_token_id), new_token_id in eos_ids,
                    tokenizer.decode([int(new_token_id)]),
                )
                if new_token_id in eos_ids:
                    finish_reason = "stop"
                    stop_reason = "eos_token"
                    break
                generated_ids.append(new_token_id)
                raw_new_text, matched_stop = _route_a_text_stop_boundary(
                    tokenizer.decode(generated_ids, skip_special_tokens=False),
                    stop_sequences,
                )
                if _stream_callback:
                    token_text = raw_new_text[published_text_len:]
                    published_text_len = len(raw_new_text)
                    if suppress_native_thinking:
                        stream_buffer += token_text
                        marker = stream_buffer.lower().find("</think>")
                        if marker >= 0:
                            visible = stream_buffer[marker + len("</think>"):]
                            suppress_native_thinking = False
                            stream_buffer = ""
                            if visible:
                                _stream_callback({"token": visible})
                    elif token_text:
                        _stream_callback({"token": token_text})
                # 载体随引擎不同：PyTorch 给张量，llama.cpp 给 numpy `[1, n_tokens]`。
                # 必须跟 `full_input_ids` 同源拼接 —— 否则在去 torch 的节点上
                # `torch_runtime.require_torch()` 会抛 `ModuleNotFoundError`，表现
                # 为 `route_a_stage_execution_failed: No module named 'torch'`
                # （legacy 首段的 decode 步早前已修同款，Route-A 这条漏了）。
                if hasattr(full_input_ids, "detach"):
                    new_token_tensor = torch.tensor([[new_token_id]], dtype=torch.long)
                    full_input_ids = torch.cat([full_input_ids, new_token_tensor], dim=1)
                else:
                    import numpy as _np

                    full_input_ids = _np.concatenate(
                        [full_input_ids,
                         _np.array([[new_token_id]], dtype=_np.int64)],
                        axis=1,
                    )
                step_ms = (time.time() - step_start) * 1000
                pipeline_metrics["steps"].append({
                    "step": step,
                    "token": new_token_id,
                    "time_ms": round(step_ms, 1),
                    "mode": "prefill" if is_prefill else "decode",
                })
                if matched_stop:
                    finish_reason = "stop"
                    stop_reason = "text_stop"
                    break
        except Exception as exc:
            terminal = _request_terminal_result(
                cancel_event=_cancel_event,
                deadline=_request_deadline,
                terminal_outcome=_terminal_outcome,
                source="route_a_exception",
            )
            if terminal is not None:
                logger.info(
                    "event=route_a_stage_pipeline_stopped task_id=%s reason=%s",
                    task_id, terminal.get("reason_code", ""),
                )
                return terminal
            if getattr(exc, "code", "") == "provider_cancelled":
                reason = (_terminal_outcome or TerminalOutcomeLatch()).decide(
                    REASON_GENERATION_CANCELLED,
                    source="route_a_provider_cancelled",
                )
                return {
                    "response": "",
                    "error": "pipeline generation was cancelled",
                    "cancelled": True,
                    "reason_code": reason,
                }
            logger.error("Route-A stage pipeline failed", exc_info=True)
            return {"response": "", "error": f"route_a_stage_execution_failed: {exc}"}
        finally:
            with self._kv_cache_lock:
                self._kv_cache.pop(task_id, None)
            self._clear_pipeline_runtime_state(task_id)
            if pipeline_stack and pipeline_stack[-1].get("task_id") == task_id:
                pipeline_stack.pop()

        # ★ 2026-10-07（真机复验根因）：抑制门若一直没等到 `</think>`，循环结束时**必须**
        #   把缓冲内容交出去 —— 否则已生成的正文既不在 SSE 事件里、也不在 response 里
        #   （真机表现：流式 `tokens=0`、非流式「流水线返回空响应」）。放在组装 response
        #   之前，让两条出口看到同一份文本。
        if _stream_callback and stream_buffer and not suppress_native_thinking:
            _stream_callback({"token": stream_buffer})
        stream_buffer = ""
        suppress_native_thinking = False
        # Reuse the same stop-checked completion as the stream. Decoding the
        # token list again here would restore the withheld stop and next turn.
        response_text = (
            tokenizer.decode(input_ids.squeeze(0), skip_special_tokens=True)
            + raw_new_text
        )
        new_text, thinking_content = callbacks.format_model_response(
            raw_new_text,
            show_thinking,
            native_thinking_prompt=native_thinking_prompt,
        )
        pipeline_metrics["total_time_ms"] = round(
            (time.time() - t_pipeline_start) * 1000, 1
        )
        pipeline_metrics["tokens_generated"] = len(generated_ids)
        pipeline_metrics["generated_tokens"] = len(generated_ids)
        pipeline_metrics["finish_reason"] = finish_reason
        pipeline_metrics["stop_reason"] = stop_reason
        if matched_stop:
            pipeline_metrics["matched_stop"] = matched_stop
        pipeline_metrics["nodes_used"] = len(stage_nodes)
        pipeline_metrics["elapsed_seconds"] = round(
            pipeline_metrics["total_time_ms"] / 1000, 3,
        )
        pipeline_metrics["tokens_per_second"] = round(
            len(generated_ids) / (pipeline_metrics["total_time_ms"] / 1000)
            if generated_ids and pipeline_metrics["total_time_ms"] > 0 else 0,
            1,
        )
        accounting = self._record_pipeline_task_accounting(
            task_id=task_id,
            pipeline_nodes=stage_nodes,
            success=True,
        )
        pipeline_metrics["node_task_accounting"] = accounting
        pipeline_metrics["workers_counted"] = accounting.get("workers_counted", [])
        pipeline_metrics["counted_nodes"] = accounting.get("counted_nodes", [])
        result = {
            "response": new_text,
            "full_text": response_text,
            "thinking": thinking_content,
            "finish_reason": finish_reason,
            "metrics": pipeline_metrics,
        }
        if _stream_callback:
            _stream_callback({"done": True, **result})
        return result


    def _check_preempt_conditions(self, current_step: int) -> bool:
        """
        检查是否满足抢占条件（防抖动 + 最小 token 阈值）。

        条件:
        1. PIPELINE_PREEMPT_ENABLED=True
        2. 未被禁用（_preempt_disabled=False）
        3. 当前未在执行抢占（防嵌套）
        4. 已生成 >= MIN_TOKENS 个 token
        5. 距上次抢占 >= MIN_INTERVAL 秒
        """
        if not self._scheduler_facade_global('PIPELINE_PREEMPT_ENABLED') or self._preempt_disabled:
            return False
        if self._preempting:  # ★ 防嵌套：Q0 内部不触发二次抢占
            return False
        if current_step < self._scheduler_facade_global('PIPELINE_PREEMPT_MIN_TOKENS'):
            return False
        if self._preempt_last_time > 0:
            if time.time() - self._preempt_last_time < self._scheduler_facade_global('PIPELINE_PREEMPT_MIN_INTERVAL'):
                return False
        return True


    def _save_preempt_state(self, *, task_id: str, generated_ids: list,
                            full_input_ids, current_step: int,
                            max_new_tokens: int, temperature: float,
                            top_p: float, prompt: str,
                            pipeline_nodes: list, first_node_id: str,
                            _stream_callback=None) -> PreemptState:
        """
        保存当前 decode 循环的所有局部状态到 PreemptState。

        generated_ids 做 shallow copy（list 在恢复后独立 append）。
        full_input_ids 仅保存 tensor 引用（只读，不会被 Q0 修改）。
        """
        state = PreemptState(
            task_id=task_id,
            generated_ids=generated_ids,
            full_input_ids=full_input_ids,
            current_step=current_step,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            prompt=prompt,
            pipeline_nodes=pipeline_nodes,
            first_node_id=first_node_id,
            _stream_callback=_stream_callback,
        )
        self._preempted_task = state
        return state


    def _execute_q0_inline(
        self,
        q0_task: QueueTask,
        preempt_state: PreemptState,
        request_deadline: RequestDeadline | None = None,
        inference_lock_state: dict | None = None,
    ) -> None:
        """
        内联执行 Q0 抢占任务。

        调用方已释放 _inference_lock 并设置 _preempting=True，本方法负责:
        1. 标记 Q0 为 current_task（try/finally 保护恢复）
        2. 检查节点就绪 → 获取推理锁 → 执行 Q0 → 释放推理锁
        3. 存储 Q0 结果 + 唤醒等待的 API 线程
        4. 恢复被抢占任务为 current_task
        5. 重新获取推理锁（为被抢占任务继续执行）

        Q0 自身异常不影响被抢占任务——错误结果照常存储并唤醒调用方。
        """
        q0_id = q0_task.task_id
        preempted_id = preempt_state.task_id
        q0_result = None
        q0_error = None
        lock_reacquired = False  # ★ BUG1 fix: track if lock was re-acquired in step 5

        try:
            # 1. 标记 Q0 为当前执行任务（在 try 内，异常时由外层的 except 恢复）
            with self.pipeline_queue._lock:
                self.pipeline_queue._current_task_id = q0_id
                if q0_id not in self.pipeline_queue._results:
                    self.pipeline_queue._results[q0_id] = {"status": "pending", "created_at": time.time()}
                self.pipeline_queue._results[q0_id]["status"] = "running"
                self.pipeline_queue._results[q0_id]["started_at"] = time.time()

            t_q0_start = time.time()

            # 2. 获取推理锁 → 执行 Q0（★ 含节点就绪检查与回退）
            extra = q0_task._extra_kwargs if q0_task._extra_kwargs else {}
            q0_deadline = extra.get("_request_deadline")
            if q0_deadline is None:
                self._inference_lock.acquire()
            elif not self._inference_lock.acquire(
                timeout=q0_deadline.require_remaining(),
            ):
                raise TimeoutError(
                    "Q0 request deadline exceeded while waiting for inference lock"
                )
            if inference_lock_state is not None:
                inference_lock_state["held"] = True
            try:
                terminal = _request_terminal_result(
                    cancel_event=q0_task.cancel_event,
                    deadline=q0_deadline,
                    terminal_outcome=extra.get("_terminal_outcome"),
                    source="q0_lock",
                )
                if terminal is not None:
                    q0_result = terminal
                elif not self._all_pipeline_nodes_ready():
                    logger.warning("Q0 抢占: 流水线节点不可用，回退到全模型推理")
                    q0_result = self._run_full_model_inference(
                        prompt=q0_task.prompt,
                        max_new_tokens=q0_task.max_new_tokens,
                        temperature=q0_task.temperature,
                        top_p=q0_task.top_p,
                        session_id=q0_task.session_id,
                        _cancel_event=q0_task.cancel_event,
                        **extra,
                    )
                else:
                    # 透传 QueueTask 中保存的额外参数（如 _stream_callback）
                    q0_result = self.run_pipeline(
                        prompt=q0_task.prompt,
                        max_new_tokens=q0_task.max_new_tokens,
                        temperature=q0_task.temperature,
                        top_p=q0_task.top_p,
                        session_id=q0_task.session_id,
                        _cancel_event=q0_task.cancel_event,
                        **extra,
                    )
            except Exception as e:
                q0_error = str(e)
                logger.error(f"❌ Q0 抢占任务执行失败: {q0_id} — {e}")
            finally:
                self._inference_lock.release()
                if inference_lock_state is not None:
                    inference_lock_state["held"] = False

            q0_elapsed = time.time() - t_q0_start

            # 3. 存储 Q0 结果 + 唤醒 API 线程 + 恢复 current_task
            with self.pipeline_queue._lock:
                if q0_error:
                    self.pipeline_queue._results[q0_id] = {
                        "status": "error", "error": q0_error,
                        "created_at": self.pipeline_queue._results.get(q0_id, {}).get("created_at", 0),
                        "completed_at": time.time(),
                        "elapsed_s": round(q0_elapsed, 2),
                    }
                else:
                    self.pipeline_queue._results[q0_id] = {
                        "status": "done", "result": q0_result,
                        "created_at": self.pipeline_queue._results.get(q0_id, {}).get("created_at", 0),
                        "started_at": self.pipeline_queue._results.get(q0_id, {}).get("started_at", 0),
                        "completed_at": time.time(),
                        "elapsed_s": round(q0_elapsed, 2),
                    }
                event = self.pipeline_queue._events.get(q0_id)
                if event:
                    event.set()
                # 4. 恢复被抢占任务为 current_task
                self.pipeline_queue._current_task_id = preempted_id

            # 5. 重新获取推理锁（为被抢占任务继续）
            reacquire_timeout = (
                None
                if request_deadline is None
                else request_deadline.remaining()
            )
            if reacquire_timeout is None:
                self._inference_lock.acquire()
            elif not self._inference_lock.acquire(timeout=reacquire_timeout):
                raise TimeoutError(
                    "request deadline exceeded while restoring preempted inference lock"
                )
            if inference_lock_state is not None:
                inference_lock_state["held"] = True
            lock_reacquired = True  # ★ 标记：在此点之后异常需释放锁

            logger.info(
                f"✅ Q0 抢占完成: {q0_id} ({q0_elapsed:.1f}s) "
                f"→ 恢复 {preempted_id}"
            )
        except Exception:
            # ★ C2 修复: _current_task_id 损坏保护
            with self.pipeline_queue._lock:
                if self.pipeline_queue._current_task_id == q0_id:
                    self.pipeline_queue._current_task_id = preempted_id
            # ★ BUG1 修复: 若锁已被重新获取，释放它以防死锁
            if lock_reacquired:
                try:
                    self._inference_lock.release()
                    if inference_lock_state is not None:
                        inference_lock_state["held"] = False
                except RuntimeError:
                    pass
            raise


    def _update_preempt_stats(self, overhead_ms: float) -> None:
        """
        更新抢占统计。

        若单次抢占开销超过 PIPELINE_PREEMPT_MAX_OVERHEAD_MS，
        自动禁用后续抢占（防止 thrashing）。
        统计同步到 PipelineQueue 以支持 get_queue_detail()。
        """
        self._preempt_count += 1
        self._preempt_total_overhead_ms += overhead_ms
        self._preempt_last_time = time.time()

        if overhead_ms > self._scheduler_facade_global('PIPELINE_PREEMPT_MAX_OVERHEAD_MS'):
            self._preempt_disabled = True
            logger.warning(
                f"⚠️ 抢占开销 {overhead_ms:.1f}ms 超过阈值 "
                f"({self._scheduler_facade_global('PIPELINE_PREEMPT_MAX_OVERHEAD_MS')}ms)，已禁用后续抢占"
            )

        # 同步到 PipelineQueue（get_queue_detail 读取此处）
        with self.pipeline_queue._lock:
            self.pipeline_queue._preempt_count = self._preempt_count
            self.pipeline_queue._preempt_total_overhead_ms = self._preempt_total_overhead_ms
            self.pipeline_queue._last_preempt_time = self._preempt_last_time


    def run_pipeline(self, *args, **kwargs) -> dict:
        """Run one pipeline task and abort every task context added by this call on exceptions."""
        stack = getattr(self._pipeline_context, "stack", None)
        if stack is None:
            stack = []
            self._pipeline_context.stack = stack
        initial_depth = len(stack)
        try:
            return self._run_pipeline(*args, **kwargs)
        except Exception as exc:
            for context in reversed(stack[initial_depth:]):
                self._broadcast_pipeline_abort(
                    context["pipeline_nodes"], context["task_id"], str(exc)
                )
                self._clear_pipeline_runtime_state(context["task_id"])
            raise
        finally:
            del stack[initial_depth:]


    def _run_pipeline(self, prompt: str, max_new_tokens: int = 512,
                     temperature: float = 0.7, top_p: float = 0.9,
                     session_id: str = None,
                     messages: list = None,
                     show_thinking: bool = False,
                     # ★ A1 / X 档：请求级路由偏好（`local_only` ⇒ 本段不委派给远端 relay 段）。
                     #   默认 "auto" ⇒ 旧调用方**逐比特不变**。
                     routing_preference: str = "auto",
                     # ★ 既有缺陷修复（Y 档第二条跑通时暴露）：`api_server.py:3609` 一直传
                     #   `enable_thinking`，而 `run_pipeline_safe`（`:5094`）pop 它之后透传
                     #   `run_pipeline(**kwargs)` ⇒ 本方法收不了 ⇒
                     #   `TypeError: _run_pipeline() got an unexpected keyword argument
                     #   'enable_thinking'` ⇒ **任何**走分布式流水线的请求都 503。
                     #   此前从未暴露，因为流水线路径一直没跑通过。
                     #   语义与 `run_pipeline_safe` 一致：`None` ⇒ 不干预模型模板默认。
                     #   流水线路径下它由 master 首段的 prompt 决定；这里接收是为**消除崩溃**，
                     #   并让签名与"回退到全模型"那条路保持一致。
                     enable_thinking: bool | None = None,
                     _stream_callback=None,
                     _cancel_event: threading.Event = None,
                     _request_deadline: RequestDeadline = None,
                     _terminal_outcome: TerminalOutcomeLatch = None,
                     _inference_lock_state: dict | None = None,
                     stop: list[str] | str | None = None) -> dict:
        """
        主节点：协调多节点流水线推理。

        **KV Cache 支持 (Phase 3)**:
         - Prefill (step 0): use_kv_cache=False，发送完整 prompt input_ids，
           各节点构建 KV cache 并本地存储。
         - Decode (step 1+): use_kv_cache=True，仅发送最后 1 个 token
           (shape 1×1)，各节点基于本地 KV cache 增量计算。
         - 通信量: hidden_states 从 O(seq_len×2048) FP16 降至 O(1×2048) FP16
         - 计算量: 每 step 从 O(seq_len) 降至 O(1)

        流程:
            1. 获取当前分层配置
            2. 确定流水线节点顺序（按 start_layer 排序）
            3. Tokenize prompt → input_ids
            4. 自回归生成循环:
               a. Prefill (step 0): 发送完整 input_ids + chain_info 给首节点
               b. Decode (step 1+): 发送新 token + chain_info 给首节点
               c. 每个中间段把 hidden_states 回传 master，由 master 转发给下一段
               d. 中间节点处理 → 继续链式转发
               e. 末节点处理 → 直接返回 logits 给主节点（LAYER_RESULT）
               f. 主节点从 logits 采样下一个 token
               g. 判断 EOS / max_tokens → 继续或结束
            5. 广播 PIPELINE_DONE，各节点清理 KV cache

        **星状数据拓扑**:
            - master 是唯一协调与转发节点，worker 之间不互连
            - hidden_states 仍按逻辑层段顺序接力，master 校验相邻目标后转发
            - 每个 step 网络传输: 每个 worker 一次上行与下一段一次下行
            6. 解码完整序列 → 返回 response text

        Returns:
            {"response": str, "thinking": str, "metrics": dict, ...}
        """
        # 流水线数据面只经 `serialize_tensor_fast` / `deserialize_tensor_fast`
        # （`TNR0` magic + numpy frombuffer），**不需要 torch**。此前这里前置
        # `require_torch()`，与 `koakuma_engine` 里「llama.cpp 不需要 torch 也能做层
        # 前向」的能力声明自相矛盾，也让无 torch 的边缘主节点在流水线入口就失败。
        import uuid
        from transport_port import MessageType, deserialize_tensor, serialize_tensor
        # ★ 与 worker 的 `deserialize_tensor_fast` 对称：fast 走 `TNR0` magic（numpy
        #   frombuffer），**不经过 `torch.load`** —— torch ≥2.6 的 `weights_only=True` 默认
        #   会让普通 pickle 载入失败（实测 `UnpicklingError: Unsupported operand 26`，
        #   在跨机 relay 上表现为"末节点响应超时"）。
        from tcp_comm import serialize_tensor_fast

        mgr = self._host
        if not mgr:
            return {"response": "", "error": "模型运行时不可用"}

        # ---- Step 1: 获取分层配置 ----
        layer_info = self.get_layer_assignments()
        # ★ A1 / X 档（Y 档第二条缺口 7）：**不能**只按 `layers_count > 0` 过滤 —— relay 段节点是
        #   **零层**条目（不占层，段工件在远端 relay_mid_service），却必须留在流水线里，
        #   否则 `pipeline_nodes` 为空 ⇒ 直接返回「没有可用的流水线从节点」
        #   （实测 `503 没有可用的流水线从节点`）。
        #   判据与 `_get_pipeline_readiness` 一致：用 `_relay_segment_for_worker()` 真判据放行。
        relay_for_worker = getattr(self, "_relay_segment_for_worker", None)

        def _participates(item: dict) -> bool:
            if item.get("layers_count", 0) > 0:
                return True
            return (
                callable(relay_for_worker)
                and relay_for_worker(item.get("node_id", "")) is not None
            )

        assignments = [
            a for a in layer_info.get("assignments", []) if _participates(a)
        ]
        logger.info(
            "Route-A 分配原始: raw=%s filtered=%s",
            [(a.get("node_id"), a.get("start_layer"), a.get("end_layer"),
              a.get("layers_count"), a.get("execution"))
             for a in layer_info.get("assignments", [])],
            [a.get("node_id") for a in assignments],
        )
        assignments.sort(key=lambda a: a.get("start_layer", 0))

        master_ids = {"master", self.get_effective_node_id()}
        master_assignment = next(
            (a for a in assignments if a.get("node_id") in master_ids),
            None,
        )
        master_participates = bool(
            master_assignment and master_assignment.get("layers_count", 0) > 0
        )
        pipeline_nodes = [
            a for a in assignments
            if a.get("node_id") not in master_ids
        ]
        # 按 start_layer 排序，确保 worker 流水线顺序正确
        pipeline_nodes.sort(key=lambda a: a.get("start_layer", 0))

        stage_offer_nodes = [
            node for node in pipeline_nodes
            if node.get("execution") == "stage_offer_v3"
        ]
        logger.info(
            "Route-A 节点筛选: pipeline=%s stage_offer=%s execution=%s",
            [n.get("node_id") for n in pipeline_nodes],
            [n.get("node_id") for n in stage_offer_nodes],
            [(n.get("node_id"), n.get("execution")) for n in pipeline_nodes],
        )
        if stage_offer_nodes:
            # Route A owns the complete prefill/decode/KV sequence for a
            # stage-only Android chain.  A mixed legacy/v3 chain remains
            # fail-closed until both protocols share one KV state machine.
            if len(stage_offer_nodes) == len(pipeline_nodes):
                return self._run_route_a_stage_pipeline(
                    prompt=prompt,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    session_id=session_id,
                    enable_thinking=enable_thinking,
                    messages=messages,
                    show_thinking=show_thinking,
                    routing_preference=routing_preference,
                    stage_nodes=stage_offer_nodes,
                    master_assignment=master_assignment or {},
                    stop=stop,
                    _stream_callback=_stream_callback,
                    _cancel_event=_cancel_event,
                    _request_deadline=_request_deadline,
                    _terminal_outcome=_terminal_outcome,
                )
            return {
                "response": "",
                "error": "route_a_mixed_legacy_execution_bridge_not_ready",
                "execution": "stage_offer_v3",
                "stage_nodes": [node.get("node_id") for node in stage_offer_nodes],
            }

        if not pipeline_nodes:
            return {"response": "", "error": "没有可用的流水线从节点"}

        # master 参与时，在本地保留 Embedding + 首段 Transformer + LM Head。
        # 后续 step 由 master.forward_layers(input_ids) 生成 hidden_states，
        # 再交给第一个 worker；避免 RTX 独显主节点只做调度而不计算。
        if master_participates:
            terminal = _request_terminal_result(
                cancel_event=_cancel_event,
                deadline=_request_deadline,
                terminal_outcome=_terminal_outcome,
                source="legacy_prepare",
            )
            if terminal is not None:
                return terminal
            try:
                ensure_layer_range = getattr(mgr, "ensure_layer_range", None)
                if callable(ensure_layer_range):
                    ensure_layer_range(
                        master_assignment["start_layer"],
                        master_assignment["end_layer"],
                        has_embedding=master_assignment.get("has_embedding", True),
                        has_lm_head=master_assignment.get("has_lm_head", True),
                    )
                else:
                    mgr.load_layer_range(
                        master_assignment["start_layer"],
                        master_assignment["end_layer"],
                        has_embedding=master_assignment.get("has_embedding", True),
                        has_lm_head=master_assignment.get("has_lm_head", True),
                    )
            except Exception as e:
                logger.error(f"❌ 主节点本地层范围加载失败: {e}", exc_info=True)
                return {"response": "", "error": f"主节点本地层范围加载失败: {e}"}
            terminal = _request_terminal_result(
                cancel_event=_cancel_event,
                deadline=_request_deadline,
                terminal_outcome=_terminal_outcome,
                source="legacy_prepare",
            )
            if terminal is not None:
                return terminal

        if not mgr.tokenizer:
            return {"response": "", "error": "流水线 tokenizer 未加载"}
        tokenizer = mgr.tokenizer
        device = mgr.get_device()

        # ★ 二次就绪检查（出队后 / 立即执行前）
        #   入队等待期间节点可能离线，tokenize 前最后确认。
        ok, err_msg = self._verify_pipeline_readiness(pipeline_nodes)
        if not ok:
            logger.error(f"❌ 流水线就绪检查失败: {err_msg}")
            # Keep the assigned segment. A readiness failure cannot authorize
            # loading a whole model while this node participates in the cluster.
            return {"response": "", "error": err_msg}

        # Freeze the worker configuration generation for the complete task.
        # A topology/model refresh after this point must not be mixed into an
        # already running token sequence.
        worker_ids = [node["node_id"] for node in pipeline_nodes]
        with self._layer_config_lock:
            worker_contracts = [
                self._worker_assignments.expected(node_id)
                for node_id in worker_ids
            ]
        config_ids = {item.get("config_id") for item in worker_contracts if item}
        model_hashes = {item.get("model_sha256") for item in worker_contracts if item}
        model_types = {item.get("model_type") for item in worker_contracts if item}
        if (len(worker_contracts) != len(worker_ids)
                or any(not item for item in worker_contracts)
                or len(config_ids) != 1
                or len(model_hashes) != 1
                or len(model_types) != 1):
            return {"response": "", "error": "worker 层配置代际不一致，请等待重新就绪"}
        pipeline_config_id = str(next(iter(config_ids)))
        pipeline_model_sha256 = str(next(iter(model_hashes)))
        pipeline_model_type = str(next(iter(model_types)))
        assignment_ids = {
            node_id: str(contract.get("assignment_id", "") or "")
            for node_id, contract in zip(worker_ids, worker_contracts)
        }
        if any(not value for value in assignment_ids.values()):
            return {"response": "", "error": "worker assignment 身份不完整"}
        pipeline_nodes = [
            {**node, "assignment_id": assignment_ids.get(node["node_id"], "")}
            for node in pipeline_nodes
        ]
        # 主节点的模型类型一律取自**它自己的描述器**：PyTorch 侧能从 `mgr.model.config`
        # 读，llama.cpp 侧根本没有 `model`/`config`（权重在 GGUF 里）——原实现只看后者
        # ⇒ 去 torch 的主节点这里恒为空串，与 worker 契约永远对不上，每次请求都被判
        # 「模型已变化」。描述器是两条引擎路径共同的单一事实来源。
        master_descriptor = {}
        get_master_descriptor = getattr(mgr, "get_pipeline_descriptor", None)
        if callable(get_master_descriptor):
            try:
                master_descriptor = get_master_descriptor() or {}
            except Exception:
                master_descriptor = {}
        master_model_type = str(
            master_descriptor.get("model_type", "")
            or getattr(
                getattr(getattr(mgr, "model", None), "config", None),
                "model_type", "",
            )
            or ""
        ).lower()
        if (master_model_type != pipeline_model_type
                or self._get_master_model_sha256() != pipeline_model_sha256):
            return {"response": "", "error": "主节点模型已变化，请等待层配置重新同步"}

        full_chain = ([master_assignment] if master_participates else []) + pipeline_nodes
        logger.info(
            f"🚀 启动流水线推理: prompt_len={len(prompt)}, "
            f"max_tokens={max_new_tokens}, worker数={len(pipeline_nodes)}, "
            f"顺序: {' → '.join(n['node_id'] for n in full_chain)}, "
            f"master_local={'✅' if master_participates else '❌'}, KV Cache: ✅"
        )

        # ---- Step 2: Tokenize ----
        chat_messages = messages or [{"role": "user", "content": prompt}]
        callbacks = self._require_callbacks()
        thinking_prompt = callbacks.thinking_system_prompt if show_thinking else None
        thinking_prefill = "【思考】\n" if show_thinking else None
        model_prompt = callbacks.build_model_chat_prompt(
            tokenizer,
            chat_messages,
            system_prompt=thinking_prompt,
            assistant_prefill=thinking_prefill,
            **({"enable_thinking": enable_thinking} if enable_thinking is not None else {}),
        )
        inputs = tokenizer(model_prompt, return_tensors="pt")
        input_ids = inputs["input_ids"]  # (1, prompt_len)
        attention_mask = inputs.get("attention_mask")
        prompt_len = input_ids.shape[1]
        # ★ 数值对照诊断：把**真正**的 token 数与 prompt 首尾打出来，便于与
        #   `local_only`（`metrics.prompt_tokens`）逐字对齐 —— 跨机 relay 的对照里，
        #   两条路若输入不同，任何"数值不一致"的判断都不成立。
        logger.info(
            "流水线 prompt: tokens=%d chars=%d head=%r tail=%r",
            prompt_len, len(model_prompt), model_prompt[:70], model_prompt[-50:],
        )

        # ---- Step 3: 自回归生成 ----
        task_id = uuid.uuid4().hex[:12]
        with self._pipeline_lock:
            self._pipeline_active_tasks.add(task_id)
            self._pipeline_task_contracts[task_id] = {
                "execution_protocol": "legacy_layer_v1",
                "config_id": pipeline_config_id,
                "model_sha256": pipeline_model_sha256,
                "model_type": pipeline_model_type,
                "worker_ids": worker_ids,
                "assignment_ids": dict(assignment_ids),
                "last_node_id": pipeline_nodes[-1]["node_id"],
                "current_step": -1,
            }
        self._pipeline_context.stack.append({
            "task_id": task_id,
            "pipeline_nodes": pipeline_nodes,
        })
        generated_ids = []
        merge_stops = getattr(mgr, "_merge_stop_sequences", None)
        stop_sequences = merge_stops(None) if callable(merge_stops) else []
        get_eos = getattr(mgr, "_get_generation_eos_token_ids", None)
        eos_token_ids = get_eos(stop_sequences) if callable(get_eos) else tokenizer.eos_token_id
        if eos_token_ids is None:
            eos_ids = {tokenizer.eos_token_id}
        elif isinstance(eos_token_ids, int):
            eos_ids = {eos_token_ids}
        else:
            eos_ids = set(eos_token_ids)
        native_thinking_prompt = bool(
            not show_thinking and "<think" in model_prompt[-128:].lower()
        )
        suppress_native_thinking = native_thinking_prompt
        stream_buffer = ""
        workers_used = [n["node_id"] for n in pipeline_nodes]
        # ★ 2026-10-05（DIST-4）：与 A3 链同口径派生层区间（见
        #   `_run_route_a_stage_pipeline` 里 `_stage_ranges` 的算法）。
        _legacy_stage_ranges: list[tuple[int, int]] = []
        for _node in pipeline_nodes:
            _start = _node.get("start_layer")
            _end = _node.get("end_layer")
            if _start is None or _end is None:
                continue
            _legacy_stage_ranges.append((int(_start), int(_end)))
        _legacy_claimed_layers = (
            [min(r[0] for r in _legacy_stage_ranges),
             max(r[1] for r in _legacy_stage_ranges)]
            if _legacy_stage_ranges else []
        )
        pipeline_metrics = {
            "steps": [],
            "total_time_ms": 0,
            "kv_cache": True,
            "chain_topology": True,
            "engine": "distributed_pipeline",
            "execution_mode": "distributed_pipeline",
            "distributed_requested": True,
            "distributed_used": True,
            "fallback": False,
            "fallback_reason": "",
            "route": "master_pipeline",
            "task_id": task_id,
            "serving_node_id": self.get_effective_node_id(),
            "workers_used": workers_used,
            "layer_assignments": pipeline_nodes,
            # ★ 2026-10-05（DIST-4）：与 A3 链对齐。
            #   ① 补 `claimed_layers` / `layer_segments`：此前 legacy 链成功时这两个
            #      字段完全缺失，而 DIST-4 的判据是「成功即 `claimed_layers` 非空」
            #      —— 缺字段就等于无法证明它真的做了分层。
            #   ② 补 `config_id`：使层配置代际在**请求级**可见（此前只存在于内存
            #      contract 与 `/api/cluster/pipeline-capacity`）。
            "claimed_layers": _legacy_claimed_layers,
            "layer_segments": [[s, e] for s, e in _legacy_stage_ranges],
            "config_id": pipeline_config_id,
        }
        t_pipeline_start = time.time()

        # 仅用于最终解码，不再用于发送
        full_input_ids = input_ids

        for step in range(max_new_tokens):
            terminal = _request_terminal_result(
                cancel_event=_cancel_event,
                deadline=_request_deadline,
                terminal_outcome=_terminal_outcome,
                source="legacy_pipeline_step",
            )
            if terminal is not None:
                step_error = terminal["error"]
                self._broadcast_pipeline_abort(
                    pipeline_nodes, task_id, step_error, count_error=False
                )
                self._clear_pipeline_runtime_state(task_id)
                return terminal

            # ---- Phase 2: 协同抢占检查 ----
            # 在每个 decode 步边界检测 Q0 任务，若存在则执行内联抢占。
            # Prefill (step=0) 不抢占——此时尚未生成任何 token。
            if (step > 0
                    and self._scheduler_facade_global('PIPELINE_PREEMPT_ENABLED')
                    and not self._preempt_disabled
                    and self._check_preempt_conditions(step)):

                # ★ 原子检查 + 弹出（消除 TOCTOU 窗口）
                q0_task = None
                with self.pipeline_queue._lock:
                    if self.pipeline_queue._q0:
                        q0_task = self.pipeline_queue._q0.popleft()

                if q0_task is not None:
                    t_preempt = time.time()

                    # 保存被抢占任务的执行状态
                    preempt_state = self._save_preempt_state(
                        task_id=task_id,
                        generated_ids=generated_ids,
                        full_input_ids=full_input_ids,
                        current_step=step,
                        max_new_tokens=max_new_tokens,
                        temperature=temperature,
                        top_p=top_p,
                        prompt=prompt,
                        pipeline_nodes=pipeline_nodes,
                        first_node_id=pipeline_nodes[0]["node_id"],
                        _stream_callback=_stream_callback,
                    )

                    logger.info(
                        f"⚡ 抢占触发: step={step}, {task_id} "
                        f"→ Q0={q0_task.task_id} "
                        f"(已生成 {len(generated_ids)} tokens)"
                    )

                    # 释放推理锁，内联执行 Q0
                    self._inference_lock.release()
                    if _inference_lock_state is not None:
                        _inference_lock_state["held"] = False

                    self._preempting = True  # ★ 防嵌套抢占
                    try:
                        self._execute_q0_inline(
                            q0_task,
                            preempt_state,
                            _request_deadline,
                            _inference_lock_state,
                        )
                        terminal = _request_terminal_result(
                            cancel_event=_cancel_event,
                            deadline=_request_deadline,
                            terminal_outcome=_terminal_outcome,
                            source="q0_restore",
                        )
                        if terminal is not None:
                            self._broadcast_pipeline_abort(
                                pipeline_nodes,
                                task_id,
                                terminal["error"],
                                count_error=False,
                            )
                            self._preempted_task = None
                            self._clear_pipeline_runtime_state(task_id)
                            return terminal
                        ensure_layer_range = getattr(mgr, "ensure_layer_range", None)
                        if callable(ensure_layer_range):
                            ensure_layer_range(
                                master_assignment["start_layer"],
                                master_assignment["end_layer"],
                                has_embedding=master_assignment.get("has_embedding", True),
                                has_lm_head=master_assignment.get("has_lm_head", True),
                            )
                        terminal = _request_terminal_result(
                            cancel_event=_cancel_event,
                            deadline=_request_deadline,
                            terminal_outcome=_terminal_outcome,
                            source="q0_prepare_restore",
                        )
                        if terminal is not None:
                            self._broadcast_pipeline_abort(
                                pipeline_nodes,
                                task_id,
                                terminal["error"],
                                count_error=False,
                            )
                            self._preempted_task = None
                            self._clear_pipeline_runtime_state(task_id)
                            return terminal
                    except Exception as e:
                        logger.error(
                            f"❌ Q0 抢占异常: {e}，中止 {task_id}"
                        )
                        # 恢复锁平衡也必须受原请求 deadline 约束。锁持有状态
                        # 与外层 finally 共享，避免超时后误释放其他请求的锁。
                        lock_held = bool(
                            _inference_lock_state
                            and _inference_lock_state.get("held", False)
                        )
                        if not lock_held:
                            restore_timeout = (
                                None
                                if _request_deadline is None
                                else _request_deadline.remaining()
                            )
                            if restore_timeout is None:
                                self._inference_lock.acquire()
                                lock_held = True
                            elif restore_timeout > 0:
                                lock_held = self._inference_lock.acquire(
                                    timeout=restore_timeout,
                                )
                            if _inference_lock_state is not None:
                                _inference_lock_state["held"] = lock_held
                        self._broadcast_pipeline_abort(
                            pipeline_nodes, task_id, f"抢占失败: {e}"
                        )
                        self._preempted_task = None
                        self._preempting = False
                        self._clear_pipeline_runtime_state(task_id)
                        terminal = _request_terminal_result(
                            cancel_event=_cancel_event,
                            deadline=_request_deadline,
                            terminal_outcome=_terminal_outcome,
                            source="q0_restore_lock",
                        )
                        if terminal is not None:
                            return terminal
                        return {"response": "", "error": f"抢占失败: {e}"}
                    finally:
                        self._preempting = False

                    # 恢复被抢占任务状态
                    generated_ids = preempt_state.generated_ids
                    full_input_ids = preempt_state.full_input_ids
                    temperature = preempt_state.temperature
                    top_p = preempt_state.top_p
                    prompt = preempt_state.prompt
                    _stream_callback = preempt_state._stream_callback
                    self._preempted_task = None  # ★ M2: 清除泄漏

                    overhead_ms = (time.time() - t_preempt) * 1000
                    self._update_preempt_stats(overhead_ms)

                    logger.info(
                        f"🔄 抢占恢复: {task_id} step {step} "
                        f"(剩余 {max_new_tokens - step} tokens)"
                    )
                    # ★ 循环继续，step 不变——被推迟的这一步现在执行

            step_start = time.time()
            logits = None
            step_error = None

            with self._pipeline_lock:
                contract = self._pipeline_task_contracts.get(task_id)
            if contract is None:
                # ★ 这一处必须走与其它失败路径相同的**统一中止流程**：任务已经派发过
                #   （worker 侧的 `_active_pipeline_task_ids` 已置位），只 `return` 会让
                #   它在 worker 上**永远**留着 —— 此后每一次层配置都会被判「本节点仍有
                #   流水线任务执行中」而延后，节点从此再也收不到新配置
                #   （实测：Surface 卡在 `active=['bae88f27f26e']`，master 连发 6 次配置
                #   都无人 ACK，链路整体失效）。
                step_error = "流水线任务执行契约已失效"
                self._broadcast_pipeline_abort(pipeline_nodes, task_id, step_error)
                self._clear_pipeline_runtime_state(task_id)
                return {"response": "", "error": step_error}
            with self._pipeline_lock:
                contract["current_step"] = step
                prefix = f"{task_id}:"
                for stale_key in list(self._pipeline_results):
                    if stale_key.startswith(prefix):
                        self._pipeline_results.pop(stale_key, None)

            # 判断 Prefill vs Decode
            is_prefill = (step == 0)

            # ---- 星状数据拓扑：构建逻辑层段顺序 ----
            # 保留逻辑层段顺序，但所有 worker 数据流都经 master 中转。
            chain_info = self._build_star_chain_route(pipeline_nodes)

            first_node_id = pipeline_nodes[0]["node_id"]
            last_node_id = pipeline_nodes[-1]["node_id"]
            has_chain = len(pipeline_nodes) >= 2

            # ---- 构建 LAYER_FORWARD 消息（发给首个 worker）----
            forward_data = {
                "task_id": task_id,
                "step": step,
                "assignment_id": assignment_ids[first_node_id],
                "config_id": pipeline_config_id,
                "model_sha256": pipeline_model_sha256,
                "model_type": pipeline_model_type,
                "chain_path": [],
                "temperature": temperature,
                "top_p": top_p,
                "use_kv_cache": not is_prefill,  # ★ Prefill=False, Decode=True
            }
            if _request_deadline is not None:
                forward_data["request_deadline_ms"] = (
                    _request_deadline.expires_at_epoch_ms
                )

            if master_participates:
                # master 本地首段：input_ids → Embedding + master layers → hidden_states。
                # worker 不再需要 Embedding，因此收到的一定是 hidden_states。
                try:
                    past_kv = None
                    if not is_prefill:
                        with self._kv_cache_lock:
                            past_kv = self._kv_cache.get(task_id)
                        if past_kv is None:
                            raise RuntimeError(
                                f"主节点 decode step {step} 缺少 KV cache"
                            )
                    if is_prefill:
                        local_input_ids = input_ids
                    else:
                        # decode 步的输入载体随引擎而变：PyTorch 要张量，llama.cpp 要 numpy
                        # （`LlamaCppEngine.forward_layers` 内部 `np.asarray`）。照 prefill 时
                        # `input_ids` 的载体来造，别**无条件**造 torch —— 那会让无 torch 的
                        # 主节点在**第二个 step** 就 `ModuleNotFoundError`（实测：relay 链的
                        # 首段 prefill 已经过了、relay 段也回了前向结果，才炸在这里）。
                        # 用 `loaded_torch()`（不触发 import）而不是调度门面的 `torch`。
                        torch_mod = loaded_torch()
                        if torch_mod is not None and hasattr(input_ids, "detach"):
                            local_input_ids = torch_mod.tensor(
                                [[new_token_id]], dtype=torch_mod.long
                            )
                        else:
                            import numpy as _np

                            local_input_ids = _np.asarray(
                                [[new_token_id]], dtype=_np.int64
                            )
                    local_attention_mask = attention_mask if is_prefill else None

                    t_master = time.time()
                    local_result = mgr.forward_layers(
                        input_ids=local_input_ids,
                        attention_mask=local_attention_mask,
                        past_key_values=past_kv,
                        use_cache=True,
                        apply_lm_head=False,
                    )
                    master_elapsed_ms = (time.time() - t_master) * 1000
                    # ★ #31 M4：同上 —— master 首段也要**优先持有 cache 对象**
                    #   （hybrid 的 tuple 会丢 recurrent state）。
                    if (local_result.get("cache") is not None
                            or local_result.get("past_key_values")):
                        with self._kv_cache_lock:
                            self._kv_cache[task_id] = _prefer_cache_state(local_result)
                    if "hidden_states" not in local_result:
                        raise RuntimeError("主节点首段未返回 hidden_states")
                    # hidden 的载体随引擎不同：PyTorch 给张量（要 `.detach().cpu()`），
                    # llama.cpp 给 numpy `[tokens, n_embd]` f32。下游只把它当 payload 用。
                    raw_hidden = local_result["hidden_states"]
                    hs_cpu = (
                        raw_hidden.detach().cpu()
                        if hasattr(raw_hidden, "detach")
                        else raw_hidden
                    )
                    import base64 as _b64
                    relay_segment = (
                        self._relay_segment_for_worker(first_node_id, routing_preference)
                        if master_participates else None
                    )
                    if relay_segment is not None:
                        forward_data["hidden_states"], forward_data["hidden_shape"] = (
                            _encode_relay_hidden(hs_cpu)
                        )
                        forward_data["hidden_wire_format"] = RELAY_HIDDEN_WIRE_FORMAT
                        # Relay stages must see the same absolute RoPE/KV
                        # positions as the local first stage.  A new TCP
                        # session is no longer opened per step, but explicit
                        # metadata also keeps tail/middle correct for callers
                        # that use more than one sequence.
                        if hs_cpu.ndim < 2:
                            raise RuntimeError("relay hidden must have token and embedding dimensions")
                        hidden_seq = int(hs_cpu.shape[-2])
                        # `numel()` 是 torch 的；llama.cpp 侧的 hidden 是 numpy，用 `.size`。
                        hidden_items = (
                            hs_cpu.numel() if hasattr(hs_cpu, "numel") else hs_cpu.size
                        )
                        hidden_batch = int(hidden_items // (hidden_seq * int(hs_cpu.shape[-1])))
                        prompt_tokens = int(input_ids.shape[-1])
                        if is_prefill:
                            positions_per_seq = list(range(hidden_seq))
                        else:
                            positions_per_seq = [prompt_tokens + step - 1] * hidden_seq
                        forward_data["seq_ids"] = [
                            seq_id for seq_id in range(hidden_batch) for _ in range(hidden_seq)
                        ]
                        forward_data["positions"] = positions_per_seq * hidden_batch
                    else:
                        forward_data["hidden_states"] = _b64.b64encode(
                            serialize_tensor_fast(hs_cpu)
                        ).decode("ascii")
                        forward_data["hidden_shape"] = list(hs_cpu.shape)
                    logger.debug(
                        f"🏠 Master 本地 Step {step}: Layer "
                        f"{master_assignment['start_layer']}-{master_assignment['end_layer']} "
                        f"hidden_states={list(hs_cpu.shape)}, time={master_elapsed_ms:.0f}ms"
                    )
                except Exception as e:
                    step_error = f"主节点本地首段 forward 失败: {e}"
                    logger.error(step_error, exc_info=True)
            else:
                if is_prefill:
                    # 兼容旧配置：首 worker 含 Embedding，发送完整 prompt input_ids
                    forward_data["input_ids"] = input_ids.cpu().tolist()
                    if attention_mask is not None:
                        forward_data["attention_mask"] = attention_mask.cpu().tolist()
                else:
                    # Decode: 仅发送最后 1 个 token
                    forward_data["input_ids"] = [[new_token_id]]

            if has_chain:
                # 只下发逻辑相邻节点身份；peer 地址不进入 worker 数据面。
                forward_data["chain_next"] = chain_info[1] if len(chain_info) > 1 else None
                forward_data["chain_remaining"] = chain_info[2:] if len(chain_info) > 2 else []
                logger.debug(
                    f"🔗 Step {step} 链式路由: "
                    f"{'master → ' if master_participates else ''}"
                    f"{' → '.join(c['node_id'] for c in chain_info)}"
                )
            else:
                forward_data["chain_next"] = None
                forward_data["chain_remaining"] = []

            # ★ A1 / X 档（2026-09-24）：若该节点被配置为「由远端 relay 段代跑本段」，
            #   随 LAYER_FORWARD 下发规格；worker 侧仅在开关打开时才会认它（默认关 ⇒ 零影响）。
            relay_segment = (
                relay_segment if "relay_segment" in locals()
                else self._relay_segment_for_worker(first_node_id, routing_preference)
            )
            if relay_segment is not None:
                forward_data["relay_segment"] = relay_segment

            # ★ A1 / X 档：把请求级路由偏好一并下发。此前 worker 侧
            #   `data.get("routing_preference", "auto")` 因主节点**从不下发**该字段而永远读到
            #   默认值（死读）；补上这一行后才真正贯通（worker 侧 `_handle_layer_forward`
            #   已用它决定 `_require_distributed` / `_force_distributed_assignment`）。
            forward_data["routing_preference"] = routing_preference

            # ---- 发送给首个 worker ----
            try:
                if not step_error:
                    self._send_to_worker(first_node_id, forward_data, MessageType.LAYER_FORWARD)
            except Exception as e:
                step_error = f"发送到首节点 {first_node_id} 失败: {e}"
                logger.error(step_error)

            if step_error:
                self._broadcast_pipeline_abort(pipeline_nodes, task_id, step_error)
                self._clear_pipeline_runtime_state(task_id)
                return {"response": "", "error": step_error}

            # ---- 等待链上任一节点返回结果（末节点=成功，其他=错误）----
            result = self._wait_for_layer_result_with_ack(
                task_id,
                [n["node_id"] for n in pipeline_nodes],  # 任一节点都可能报错
                timeout=self._scheduler_facade_global('PIPELINE_STEP_TIMEOUT'),
                ack_node_ids=[n["node_id"] for n in pipeline_nodes[1:]] if has_chain else [],
                ack_step=step,
                ack_timeout=min(5.0, max(1.0, self._scheduler_facade_global('PIPELINE_STEP_TIMEOUT') / 6)),
                cancel_event=_cancel_event,
                request_deadline=_request_deadline,
            )
            terminal = _request_terminal_result(
                cancel_event=_cancel_event,
                deadline=_request_deadline,
                terminal_outcome=_terminal_outcome,
                source="legacy_pipeline_wait",
            )
            if terminal is not None:
                step_error = terminal["error"]
                self._broadcast_pipeline_abort(
                    pipeline_nodes, task_id, step_error, count_error=False
                )
                self._clear_pipeline_runtime_state(task_id)
                return terminal
            if result is None:
                step_error = f"末节点 {last_node_id} 响应超时"
                logger.error(step_error)
                self._broadcast_pipeline_abort(pipeline_nodes, task_id, step_error)
                self._clear_pipeline_runtime_state(task_id)
                return {"response": "", "error": step_error}

            if result.get("error"):
                if result.get("reason_code") == REQUEST_DEADLINE_EXCEEDED:
                    terminal = _request_terminal_result(
                        cancel_event=_cancel_event,
                        deadline=_request_deadline,
                        terminal_outcome=_terminal_outcome,
                        source="legacy_pipeline_result",
                    )
                    if terminal is not None:
                        self._broadcast_pipeline_abort(
                            pipeline_nodes,
                            task_id,
                            terminal["error"],
                            count_error=False,
                        )
                        self._clear_pipeline_runtime_state(task_id)
                        return terminal
                step_error = f"流水线错误: {result['error']}"
                logger.error(step_error)
                self._broadcast_pipeline_abort(pipeline_nodes, task_id, step_error)
                self._clear_pipeline_runtime_state(task_id)
                return {"response": "", "error": step_error}

            # 提取末端输出。推荐拓扑由 worker 返回 hidden_states，主节点在
            # CUDA 上执行 Norm + LM Head；兼容旧配置直接返回 logits。
            # ★ Y-(b)：末节点是 relay **tail** 段时回的是 **token**（远端已做完 argmax），
            #   既没有 logits 也没有 hidden —— 这种拓扑下 master 不需要跑 LM Head。
            relay_token = result.get("token")
            if relay_token is not None:
                pass    # 由下方的 token 分支消费（不设 step_error）
            elif "logits" in result and result["logits"] is not None:
                logits_data = result["logits"]
                if isinstance(logits_data, bytes):
                    logits = deserialize_tensor(logits_data).to(device=device)
                elif (torch_mod := loaded_torch()) is not None and isinstance(
                    logits_data, torch_mod.Tensor
                ):
                    logits = logits_data.to(device=device)
                else:
                    step_error = f"未知 logits 类型: {type(logits_data).__name__}"
                    logger.error(step_error)
            elif "hidden_states" in result and result["hidden_states"] is not None:
                hidden_data = result["hidden_states"]
                if isinstance(hidden_data, bytes):
                    if result.get("hidden_wire_format") == RELAY_HIDDEN_WIRE_FORMAT:
                        try:
                            final_hidden = _decode_relay_hidden(
                                hidden_data, result.get("hidden_shape")
                            )
                        except Exception as exc:
                            step_error = f"relay hidden 解码失败: {exc}"
                            logger.error(step_error)
                            final_hidden = None
                    else:
                        final_hidden = deserialize_tensor(hidden_data)
                elif (torch_mod := loaded_torch()) is not None and isinstance(
                    hidden_data, torch_mod.Tensor
                ):
                    final_hidden = hidden_data
                else:
                    step_error = (
                        f"未知 hidden_states 类型: {type(hidden_data).__name__}"
                    )
                    logger.error(step_error)
                    final_hidden = None
                if final_hidden is not None:
                    try:
                        logits = self._run_master_lm_head(final_hidden)
                    except Exception as e:
                        step_error = f"主节点 LM Head 执行失败: {e}"
                        logger.error(step_error, exc_info=True)
            else:
                step_error = "末节点未返回 logits"
                logger.error(step_error)

            if step_error:
                # ★ 统一中止路径：广播 ABORT → 清理各节点 KV cache → 返回错误
                self._broadcast_pipeline_abort(pipeline_nodes, task_id, step_error)
                self._clear_pipeline_runtime_state(task_id)
                return {"response": "", "error": step_error}

            # ---- Step 4: 从 logits 选择下一个 token ----
            # ★ Y-(b)：末节点为 relay **tail** 段时，远端已跑完本段并做完 argmax ⇒ 直接给
            #   token，此时**没有 logits** 可用。该协议语义就是 argmax ⇒ 与**贪心**等价；
            #   `temperature > 0` 的采样在远端无法执行 ⇒ 显式**具名拒绝**
            #   （绝不把 token 当成"采样结果"，那会静默改变请求语义）。
            if relay_token is not None:
                if float(temperature or 0.0) > 0.0:
                    step_error = (
                        "relay tail 段只回 token（远端 argmax）⇒ 仅支持贪心解码，"
                        f"但请求 temperature={temperature}"
                    )
                    logger.error(step_error)
                    self._broadcast_pipeline_abort(pipeline_nodes, task_id, step_error)
                    self._clear_pipeline_runtime_state(task_id)
                    return {"response": "", "error": step_error}
                new_token_id = int(relay_token)
            else:
                # temperature=0 与单机路径一致采用贪心解码；正温度才执行
                # FP32 top-p 采样并在进入 CUDA multinomial 前校验概率。
                new_token_id = self._scheduler_facade_global('_sample_pipeline_token_id')(
                    logits, temperature=temperature, top_p=top_p,
                )

            # 检查 EOS
            if new_token_id in eos_ids:
                logger.info(f"🏁 EOS token 生成于 step {step}")
                break

            generated_ids.append(new_token_id)

            # ★ 流式回调：每生成一个 token 立即推送
            if _stream_callback:
                new_token_text = tokenizer.decode([new_token_id])
                if suppress_native_thinking:
                    stream_buffer += new_token_text
                    marker = stream_buffer.lower().find("</think>")
                    if marker >= 0:
                        visible = stream_buffer[marker + len("</think>"):]
                        suppress_native_thinking = False
                        stream_buffer = ""
                        if visible:
                            _stream_callback({"token": visible})
                else:
                    _stream_callback({"token": new_token_text})

            # 更新完整序列仅用于最终解码（不再发送给首节点）。
            # 用 numpy 而不是 torch：`full_input_ids` 起初来自 tokenizer（两条引擎路径现在
            # 都给 numpy），这里只做**累积**，`_save_preempt_state` 也只做**暂存** ——
            # 全程没有任何张量运算。此前无条件造 `torch.tensor` + `torch.cat` ⇒ 无 torch
            # 的主节点在**第一个 decode 步**就 `ModuleNotFoundError`（实测：整条 relay 链
            # 已走通、relay 段真回了前向结果，才在这里炸，此前一直被外层包装吞成一句
            # `str(e)`）。
            import numpy as _np

            full_input_ids = _np.concatenate(
                [
                    _np.asarray(full_input_ids),
                    _np.asarray([[new_token_id]], dtype=_np.int64),
                ],
                axis=1,
            )

            step_ms = (time.time() - step_start) * 1000
            pipeline_metrics["steps"].append({
                "step": step,
                "token": new_token_id,
                "time_ms": round(step_ms, 1),
                "mode": "prefill" if is_prefill else "decode",
            })
            logger.info(
                f"🪜 Step {step}: token={new_token_id}, "
                f"seq_len={full_input_ids.shape[1]}, "
                f"mode={'prefill' if is_prefill else 'decode'}, "
                f"time={step_ms:.0f}ms"
            )

        # ---- Step 5: 广播 PIPELINE_DONE（各节点清理 KV cache） ----
        for n in pipeline_nodes:
            try:
                self._send_to_worker(
                    n["node_id"],
                    {"task_id": task_id},
                    MessageType.PIPELINE_DONE,
                )
            except Exception as e:
                logger.warning(
                    "发送 PIPELINE_DONE 失败: node=%s task=%s error=%s",
                    n.get("node_id"), task_id, e,
                )

        # ★ 清理 master 自身 KV cache（master_participates 路径会产生本地缓存）
        with self._kv_cache_lock:
            if task_id in self._kv_cache:
                del self._kv_cache[task_id]

        # ---- Step 6: 解码结果 ----
        # 拼接用 numpy：`input_ids` 来自 tokenizer（两条引擎路径现在都给 numpy），
        # 而 `tokenizer.decode` 两侧都接受 numpy。此前无条件走 `torch.cat` ⇒ 无 torch 的
        # 主节点在**生成结束后**这一步炸（整条链跑完才现形）。
        import numpy as _np

        if generated_ids:
            full_ids = _np.concatenate(
                [
                    _np.asarray(input_ids).squeeze(0),
                    _np.asarray(generated_ids, dtype=_np.int64),
                ],
                axis=0,
            )
            response_text = tokenizer.decode(full_ids, skip_special_tokens=True)
            raw_new_text = tokenizer.decode(
                generated_ids, skip_special_tokens=True
            )
        else:
            response_text = tokenizer.decode(
                _np.asarray(input_ids).squeeze(0), skip_special_tokens=True
            )
            raw_new_text = ""

        new_text, thinking_content = self._require_callbacks().format_model_response(
            raw_new_text,
            show_thinking,
            native_thinking_prompt=native_thinking_prompt,
        )

        pipeline_metrics["total_time_ms"] = round(
            (time.time() - t_pipeline_start) * 1000, 1
        )
        pipeline_metrics["tokens_generated"] = len(generated_ids)
        pipeline_metrics["generated_tokens"] = len(generated_ids)
        pipeline_metrics["nodes_used"] = len(pipeline_nodes)
        pipeline_metrics["elapsed_seconds"] = round(pipeline_metrics["total_time_ms"] / 1000, 3)

        tokens_per_sec = (
            len(generated_ids) / (pipeline_metrics["total_time_ms"] / 1000)
            if pipeline_metrics["total_time_ms"] > 0 and generated_ids
            else 0
        )
        pipeline_metrics["tokens_per_second"] = round(tokens_per_sec, 1)

        accounting = self._record_pipeline_task_accounting(
            task_id=task_id,
            pipeline_nodes=pipeline_nodes,
            success=True,
        )
        pipeline_metrics["node_task_accounting"] = accounting
        pipeline_metrics["workers_counted"] = accounting.get("workers_counted", [])
        pipeline_metrics["counted_nodes"] = accounting.get("counted_nodes", [])

        logger.info(
            f"✅ 流水线推理完成: {len(generated_ids)} tokens, "
            f"{pipeline_metrics['total_time_ms']:.0f}ms, "
            f"{tokens_per_sec:.1f} tok/s (KV Cache: ✅)"
        )

        result = {
            "response": new_text,
            "full_text": response_text,
            "thinking": thinking_content,
            "metrics": pipeline_metrics,
        }

        # ★ 流式完成通知
        if _stream_callback:
            _stream_callback({"done": True, **result})

        self._clear_pipeline_runtime_state(task_id)
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


    def _process_queued_pipeline_task(self, prompt: str, **kwargs) -> dict:
        """
        队列工作线程的回调：执行流水线推理并返回结果。

        ★ 直接调用 run_pipeline（绕过 run_pipeline_safe 的排队检查），
           避免死锁：队列 worker 已设置 _current_task_id，若走 run_pipeline_safe
           会再次检测 is_busy=True → enqueue → 永久等待自己完成。

        ★ 手动管理 _inference_lock：正常路径在 finally 中释放；
           抢占路径中 run_pipeline 内部会 release/re-acquire，
           返回时锁仍被持有，由 finally 统一释放。
        """
        request_deadline = kwargs.get("_request_deadline")
        terminal_outcome = kwargs.get("_terminal_outcome")
        if request_deadline is None:
            self._inference_lock.acquire()
            lock_held = True
        else:
            lock_held = self._inference_lock.acquire(
                timeout=request_deadline.remaining(),
            )
            if not lock_held:
                return _request_terminal_result(
                    cancel_event=kwargs.get("_cancel_event"),
                    deadline=request_deadline,
                    terminal_outcome=terminal_outcome,
                    source="queue_inference_lock",
                )
        inference_lock_state = {"held": lock_held}
        try:
            terminal = _request_terminal_result(
                cancel_event=kwargs.get("_cancel_event"),
                deadline=request_deadline,
                terminal_outcome=terminal_outcome,
                source="queue_dispatch",
            )
            if terminal is not None:
                return terminal
            # 检查节点是否就绪
            require_distributed = bool(kwargs.pop("_require_distributed", False))
            force_distributed = bool(
                kwargs.pop("_force_distributed_assignment", require_distributed)
            )
            sync_timeout = kwargs.pop(
                "_pipeline_model_sync_timeout", self._scheduler_facade_global('PIPELINE_MODEL_SYNC_TIMEOUT'),
            )
            if request_deadline is not None:
                sync_timeout = request_deadline.clamp(sync_timeout)
                if sync_timeout <= 0:
                    return _request_terminal_result(
                        cancel_event=kwargs.get("_cancel_event"),
                        deadline=request_deadline,
                        terminal_outcome=terminal_outcome,
                        source="queue_prepare",
                    )
            pipeline_ready = self._all_pipeline_nodes_ready()
            if force_distributed and not self._has_active_distributed_pipeline_plan():
                pipeline_ready = False
            if not pipeline_ready and force_distributed:
                readiness = self._synchronize_pipeline_workers_for_request(
                    timeout=sync_timeout,
                    force_distributed_assignment=True,
                )
                pipeline_ready = bool(readiness.get("ready"))
            terminal = _request_terminal_result(
                cancel_event=kwargs.get("_cancel_event"),
                deadline=request_deadline,
                terminal_outcome=terminal_outcome,
                source="queue_prepare",
            )
            if terminal is not None:
                return terminal
            if not pipeline_ready:
                if require_distributed:
                    return {
                        "response": "",
                        "error": (
                            "distributed_required: queued pipeline workers not ready"
                        ),
                        "metrics": {"distributed_used": False, "fallback": False},
                    }
                logger.warning("流水线节点不可用，队列任务回退到全模型推理")
                # ★ H1 修复: 保持 lock_held=True，回退推理在锁保护下执行（防止 GPU 并发）
                return self._run_full_model_inference(
                    prompt,
                    _fallback_reason="queue_pipeline_nodes_not_ready",
                    **kwargs,
                )
            run_kwargs = dict(kwargs)
            run_kwargs["_inference_lock_state"] = inference_lock_state
            result = self.run_pipeline(prompt, **run_kwargs)
            if result.get("error"):
                if result.get("reason_code") == REQUEST_DEADLINE_EXCEEDED:
                    return result
                if require_distributed:
                    return result
                if self._stream_output_started(kwargs):
                    logger.warning(
                        "流水线已输出部分内容，跳过全模型回退以避免重复回答: %s",
                        result.get("error"),
                    )
                    return result
                logger.warning(
                    "队列任务流水线单步失败，回退到全模型推理: %s",
                    result.get("error"),
                )
                return self._run_full_model_inference(
                    prompt,
                    _fallback_reason=f"queue_pipeline_error_result: {result.get('error')}",
                    **kwargs,
                )
            return result
        except Exception as e:
            terminal = _request_terminal_result(
                cancel_event=kwargs.get("_cancel_event"),
                deadline=kwargs.get("_request_deadline"),
                terminal_outcome=kwargs.get("_terminal_outcome"),
                source="queue_exception",
            )
            if terminal is not None:
                return terminal
            if require_distributed:
                return {
                    "response": "",
                    "error": f"distributed_required: queued pipeline failed: {e}",
                    "metrics": {"distributed_used": False, "fallback": False},
                }
            if self._stream_output_started(kwargs):
                logger.error(
                    "流水线已输出部分内容后异常，跳过全模型回退: %s",
                    e,
                    exc_info=True,
                )
                return {"response": "", "error": str(e)}
            logger.error(f"队列任务流水线推理失败: {e}，回退到全模型推理", exc_info=True)
            # 锁可能在抢占异常路径中已被释放；只释放本请求仍持有的锁。
            if inference_lock_state["held"]:
                self._inference_lock.release()
                lock_held = False
                inference_lock_state["held"] = False
            # Phase 5 review H2: 回退推理需持有推理锁
            if request_deadline is None:
                self._inference_lock.acquire()
                lock_held = True
            else:
                lock_held = self._inference_lock.acquire(
                    timeout=request_deadline.remaining(),
                )
                if not lock_held:
                    return _request_terminal_result(
                        cancel_event=kwargs.get("_cancel_event"),
                        deadline=request_deadline,
                        terminal_outcome=terminal_outcome,
                        source="queue_fallback_lock",
                    )
            inference_lock_state["held"] = lock_held
            return self._run_full_model_inference(
                prompt,
                _fallback_reason=f"queue_pipeline_error: {e}",
                **kwargs,
            )
        finally:
            if inference_lock_state["held"]:
                self._inference_lock.release()
                inference_lock_state["held"] = False


    def run_pipeline_safe(self, prompt: str, **kwargs) -> dict:
        """
        带自动回退的流水线推理（支持排队）。

        规则:
        - 流水线节点不可用 → 回退到全模型推理
        - 队列中有任务执行中 → 新请求自动入队等待
        - 队列空闲 → 立即执行

        ★ 立即执行路径与 is_busy 检查在同一锁内完成，消除 TOCTOU 竞态：
          多个调用方线程不可能同时看到 is_busy=False 并绕过队列。
        """
        # ---- 引擎检查：流水线仅支持 PyTorch 引擎 ----
        # llama.cpp(GGUF) 不支持层拆分，直接走全模型推理。
        # 已显式准备的 distributed-only 模型尚未物化任何权重，也允许进入；
        # 主节点首段会在 worker 就绪后由 run_pipeline 按分配范围首次加载。
        queue_timeout = kwargs.pop('_queue_timeout', self._scheduler_facade_global('PIPELINE_TIMEOUT'))
        request_timeout = kwargs.pop(
            "_request_timeout_seconds", queue_timeout,
        )
        request_deadline = coerce_request_deadline(
            kwargs.pop("_request_deadline", None),
            timeout_seconds=request_timeout,
        )
        terminal_outcome = kwargs.pop(
            "_terminal_outcome", None,
        ) or TerminalOutcomeLatch()
        cancel_event = kwargs.get("_cancel_event")
        if not isinstance(cancel_event, DeadlineCancelEvent):
            cancel_event = DeadlineCancelEvent(
                cancel_event, request_deadline,
            )
        kwargs["_cancel_event"] = cancel_event
        kwargs["_request_deadline"] = request_deadline
        kwargs["_terminal_outcome"] = terminal_outcome
        terminal = _request_terminal_result(
            cancel_event=cancel_event,
            deadline=request_deadline,
            terminal_outcome=terminal_outcome,
            source="request_entry",
        )
        if terminal is not None:
            return terminal
        require_distributed = bool(kwargs.pop("_require_distributed", False))
        force_distributed_assignment = bool(
            kwargs.pop("_force_distributed_assignment", require_distributed)
        )
        model_sync_timeout = kwargs.pop(
            '_pipeline_model_sync_timeout', self._scheduler_facade_global('PIPELINE_MODEL_SYNC_TIMEOUT'),
        )
        model_sync_timeout = request_deadline.clamp(model_sync_timeout)
        if model_sync_timeout <= 0:
            return _request_terminal_result(
                cancel_event=cancel_event,
                deadline=request_deadline,
                terminal_outcome=terminal_outcome,
                source="prepare",
            )
        self._track_stream_output(kwargs)
        mgr = self._host
        pipeline_prepared = bool(
            mgr and getattr(mgr, "is_pipeline_prepared", False)
        )
        if not mgr or (not mgr.is_loaded and not pipeline_prepared):
            if self._pipeline_recovery_pending:
                return {
                    "response": "",
                    "error": (
                        "pipeline_recovery_pending: "
                        "waiting for authoritative layer-config recovery"
                    ),
                    "metrics": {"distributed_used": False, "fallback": False},
                }
            logger.warning("模型未加载，无法执行流水线推理")
            if require_distributed:
                return {
                    "response": "",
                    "error": "distributed_required: pipeline model is not prepared",
                    "metrics": {"distributed_used": False, "fallback": False},
                }
            return self._run_full_model_inference(
                prompt,
                _fallback_reason="model_not_loaded_for_pipeline",
                **kwargs,
            )
        engine_type = backend_id_for(mgr)
        if engine_type and not runtime_supports(mgr, Capability.FORWARD_LAYERS):
            logger.info(
                f"引擎类型为 {engine_type}，不支持流水线层拆分，"
                f"使用全模型推理"
            )
            if require_distributed:
                return {
                    "response": "",
                    "error": (
                        "distributed_required: pipeline execution requires "
                        f"pytorch, got {engine_type}"
                    ),
                    "metrics": {"distributed_used": False, "fallback": False},
                }
            if self._pipeline_recovery_pending:
                return {
                    "response": "",
                    "error": (
                        "pipeline_recovery_pending: "
                        "waiting for authoritative layer-config recovery"
                    ),
                    "metrics": {"distributed_used": False, "fallback": False},
                }
            return self._run_full_model_inference(
                prompt,
                _fallback_reason=f"engine {engine_type} does not support layer-split pipeline",
                **kwargs,
            )

        # ---- 自动回退：节点不可用 → 全模型推理 ----
        readiness = None
        try:
            pipeline_ready = (
                False
                if force_distributed_assignment
                else self._all_pipeline_nodes_ready()
            )
            if not pipeline_ready:
                if force_distributed_assignment:
                    readiness = self._synchronize_pipeline_workers_for_request(
                        timeout=model_sync_timeout,
                        force_distributed_assignment=True,
                    )
                else:
                    readiness = self._synchronize_pipeline_workers_for_request(
                        timeout=model_sync_timeout,
                    )
                pipeline_ready = bool(readiness.get("ready"))
        except Exception:
            logger.warning(
                "请求前主从模型配置同步失败，将按未就绪处理",
                exc_info=True,
            )
            pipeline_ready = False

        terminal = _request_terminal_result(
            cancel_event=cancel_event,
            deadline=request_deadline,
            terminal_outcome=terminal_outcome,
            source="prepare",
        )
        if terminal is not None:
            return terminal

        if not pipeline_ready:
            try:
                readiness = readiness or self._get_pipeline_readiness()
                readiness_reason = readiness.get("reason") or "未知原因"
            except Exception:
                readiness_reason = "就绪状态检查失败"
            logger.warning(
                "部分流水线节点未就绪，回退到全层主节点模式: %s",
                readiness_reason,
            )
            if self._pipeline_recovery_pending:
                return {
                    "response": "",
                    "error": (
                        "pipeline_recovery_pending: "
                        f"{readiness_reason}"
                    ),
                    "metrics": {
                        "distributed_used": False,
                        "fallback": False,
                        "pipeline_readiness": readiness or {},
                    },
                }
            if require_distributed:
                return {
                    "response": "",
                    "error": (
                        "distributed_required: pipeline workers not ready: "
                        f"{readiness_reason}"
                    ),
                    "metrics": {
                        "distributed_used": False,
                        "fallback": False,
                        "pipeline_readiness": readiness or {},
                    },
                }
            # 回退仍会执行完整模型推理，必须与其他 GPU 推理共享同一把锁。
            # 这里阻塞等待，避免锁被占用时直接绕过互斥保护。
            if not self._inference_lock.acquire(
                timeout=request_deadline.remaining(),
            ):
                return _request_terminal_result(
                    cancel_event=cancel_event,
                    deadline=request_deadline,
                    terminal_outcome=terminal_outcome,
                    source="fallback_lock",
                )
            try:
                return self._run_full_model_inference(
                    prompt,
                    _fallback_reason=(
                        f"pipeline_nodes_not_ready: {readiness_reason}"
                    ),
                    **kwargs,
                )
            finally:
                self._inference_lock.release()

        # ---- 排队逻辑（锁内原子判断 + 入队/执行）----
        # ★ 同时检查 is_busy 和 queue_size，消除竞态缺口：
        #   T1 刚完成（_current_task_id=None）但队列还残留 T2 的请求，
        #   此时 T3 若仅检查 is_busy 会绕过队列直接执行 → T2 被插队。
        with self.pipeline_queue._lock:
            if self.pipeline_queue.is_busy or self.pipeline_queue.queue_size > 0:
                # 有任务执行中 或 队列非空 → 入队（保证 FIFO 顺序）
                queued_kwargs = dict(kwargs)
                # Preserve request-scoped routing semantics across the queue.
                # These flags are consumed by _process_queued_pipeline_task
                # before it calls run_pipeline, so they never reach the model
                # engine as unexpected keyword arguments.
                queued_kwargs.update({
                    "_require_distributed": require_distributed,
                    "_force_distributed_assignment": force_distributed_assignment,
                    "_pipeline_model_sync_timeout": model_sync_timeout,
                })
                task_id = self.pipeline_queue.enqueue(
                    prompt=prompt, **queued_kwargs,
                )
            else:
                # 空闲且队列空 → 标记为"即将执行"（阻止其他线程绕过队列）
                task_id = None
                self.pipeline_queue._current_task_id = "__reserved__"

        if task_id is not None:
            # 入队路径：阻塞等待结果
            logger.info(
                f"⏳ 流水线正忙，请求已排队: task={task_id}, "
                f"queue_depth={self.pipeline_queue.queue_size}"
            )
            result = self.pipeline_queue.wait_for_result(
                task_id,
                timeout=min(queue_timeout, request_deadline.remaining()),
                cancel_event=cancel_event,
            )
            terminal = _request_terminal_result(
                cancel_event=cancel_event,
                deadline=request_deadline,
                terminal_outcome=terminal_outcome,
                source="queue_wait",
            )
            if terminal is not None:
                self.pipeline_queue.cancel_task(task_id)
                return terminal
            if result.get("status") == "done":
                payload = result.get("result", {})
                if isinstance(payload, dict) and payload.get("error"):
                    if self._stream_output_started(kwargs):
                        return payload
                    if require_distributed:
                        return payload
                    if not self._inference_lock.acquire(
                        timeout=request_deadline.remaining(),
                    ):
                        return _request_terminal_result(
                            cancel_event=cancel_event,
                            deadline=request_deadline,
                            terminal_outcome=terminal_outcome,
                            source="queue_fallback_lock",
                        )
                    try:
                        logger.warning(
                            "排队流水线任务返回错误，回退到全模型推理: %s",
                            payload.get("error"),
                        )
                        return self._run_full_model_inference(
                            prompt,
                            _fallback_reason=f"queued_pipeline_error_result: {payload.get('error')}",
                            **kwargs,
                        )
                    finally:
                        self._inference_lock.release()
                return payload
            elif result.get("status") == "timeout":
                terminal = _request_terminal_result(
                    cancel_event=cancel_event,
                    deadline=request_deadline,
                    terminal_outcome=terminal_outcome,
                    source="queue_wait",
                )
                if terminal is None:
                    reason = terminal_outcome.decide(
                        "queue_timeout", source="queue_wait",
                    )
                    terminal = {
                        "response": "",
                        "error": f"排队超时 ({queue_timeout}s)",
                        "reason_code": reason,
                        "metrics": {
                            "distributed_used": False,
                            "fallback": False,
                            "terminal_reason": terminal_outcome.snapshot(),
                        },
                    }
                self.pipeline_queue.cancel_task(task_id)
                return terminal
            else:
                return {"response": "", "error": result.get("error", "排队请求失败")}

        # ---- 立即执行（已通过原子检查）----
        # ★ 非阻塞获取推理锁：防止与 _process_loop 残留任务并发
        if not self._inference_lock.acquire(blocking=False):
            logger.warning("推理引擎正忙（锁竞争），返回繁忙错误")
            self.pipeline_queue._current_task_id = None
            return {"response": "", "error": "推理引擎正忙，请稍后重试"}
        inference_lock_state = {"held": True}
        try:
            try:
                run_kwargs = dict(kwargs)
                run_kwargs["_inference_lock_state"] = inference_lock_state
                result = self.run_pipeline(prompt, **run_kwargs)
                if result.get("error"):
                    if result.get("reason_code") == REQUEST_DEADLINE_EXCEEDED:
                        return result
                    if self._stream_output_started(kwargs):
                        return result
                    if require_distributed:
                        return result
                    logger.warning(
                        "流水线推理返回错误，回退到全层主节点模式: %s",
                        result.get("error"),
                    )
                    return self._run_full_model_inference(
                        prompt,
                        _fallback_reason=f"pipeline_error_result: {result.get('error')}",
                        **kwargs,
                    )
                return result
            except Exception as e:
                terminal = _request_terminal_result(
                    cancel_event=cancel_event,
                    deadline=request_deadline,
                    terminal_outcome=terminal_outcome,
                    source="pipeline_exception",
                )
                if terminal is not None:
                    return terminal
                if self._stream_output_started(kwargs):
                    logger.warning("流水线推理失败（流式已开始）: %s", e, exc_info=True)
                    return {"response": "", "error": str(e)}
                if require_distributed:
                    # `distributed_required` 的 error 会**原样抛给客户端**，所以必须先把
                    # 完整 traceback 落盘 —— 否则排障只剩一句 `str(e)`（实测踩过：只看到
                    # `No module named 'torch'`，看不出是哪条路径在 import，只能靠猜）。
                    logger.error(
                        "distributed_required 流水线失败: %s", e, exc_info=True,
                    )
                    return {
                        "response": "",
                        "error": f"distributed_required: {e}",
                        "metrics": {"distributed_used": False, "fallback": False},
                    }
                logger.error(f"流水线推理失败: {e}，回退到全层主节点模式", exc_info=True)
                return self._run_full_model_inference(
                    prompt,
                    _fallback_reason=f"pipeline_error: {e}",
                    **kwargs,
                )
        finally:
            if inference_lock_state["held"]:
                self._inference_lock.release()
                inference_lock_state["held"] = False
            # ★ 释放预留标记（无论成功/失败/回退）
            if task_id is None:
                self.pipeline_queue._current_task_id = None


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
                t0 = time.time()
                for chunk in mgr.chat_stream(
                    messages=messages,
                    max_tokens=max_new_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    enable_thinking=enable_thinking,
                    _cancel_event=cancel_event,
                ):
                    if chunk:
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
                        else:
                            _stream_callback({"token": chunk})
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
            and self._scheduler_facade_global('RUN_MODE') == "distributed"
            and self._effective_role() == "master"
            and len(workers) > 0
            and distributed_enabled
        )
        active = available and readiness.get("ready", False)
        degraded = available and not active and online_count > 0

        if not distributed_enabled:
            reason_code = "distributed_disabled"
            reason = "分布式推理开关已关闭"
        elif self._scheduler_facade_global('RUN_MODE') != "distributed":
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
