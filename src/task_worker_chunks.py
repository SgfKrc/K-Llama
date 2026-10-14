"""★ 2026-10-07（DIST-NEXT-2b）：大 payload 的**有序分片**（装配 / 切分 / 建消息）。

契约见 `task_worker_protocol` 的 `stage_chunk`：coordinator 先发 `chunk_count` 条
`stage_chunk`（每条带 attempt 身份、序号、总数与自身摘要），再发引用它们的
`stage_offer`；worker 收齐、逐片校验、装配后仍以 `stage_offer.hidden_sha256` 兜底。

为什么需要它：超单帧预算的输入（长 prefill / 大 `n_embd`）在 DIST-NEXT-2a 里只能被
**具名拒绝**；本模块给出「装得下」的路径，同时保持三件事不变：

* **有序**：序号必须落在 `[0, chunk_count)`，重复即拒（不静默覆盖）；
* **有上限**：片数、单片字节、装配总字节都是硬上限 —— 分片不能用来绕过帧预算；
* **可校验**：每片带 `payload_sha256`，装配后再由 offer 的 `hidden_sha256` 兜底。

两侧共享这里的**纯逻辑**（装配器与切分计划不依赖调度器状态）；Android 侧为同语义的
Kotlin 实现。
"""

from __future__ import annotations

import base64
import hashlib
import math
from dataclasses import dataclass, field
from typing import Any

from task_worker_protocol import (
    MAX_STAGE_CHUNKS,
    MAX_STAGE_PAYLOAD_BYTES,
    PROTOCOL_VERSION,
    WorkerProtocolError,
    build_message,
)
from task_worker_protocol import STAGE_CHUNK_BYTES


def _error(code: str, field_name: str, message: str) -> WorkerProtocolError:
    return WorkerProtocolError(message, code=code, field=field_name)


# ---------------------------------------------------------------------------
# 接收端：有序装配（fail-closed）
# ---------------------------------------------------------------------------

@dataclass
class _AttemptChunks:
    chunk_count: int
    total_bytes: int
    received: dict[int, bytes] = field(default_factory=dict)


class StageChunkAssembler:
    """按 attempt 累积分片，集齐才给出装配结果。

    错误码（稳定、可 grep）：`invalid_chunk_count` / `invalid_chunk_index` /
    `duplicate_chunk` / `chunk_too_large` / `chunk_layout_mismatch` /
    `stage_payload_too_large` / `chunk_digest_mismatch` / `incomplete_chunks`。
    """

    def __init__(
        self,
        *,
        max_chunks: int = MAX_STAGE_CHUNKS,
        max_chunk_bytes: int = STAGE_CHUNK_BYTES,
        max_total_bytes: int = MAX_STAGE_PAYLOAD_BYTES,
    ) -> None:
        self.max_chunks = int(max_chunks)
        self.max_chunk_bytes = int(max_chunk_bytes)
        self.max_total_bytes = int(max_total_bytes)
        self._state: dict[str, _AttemptChunks] = {}

    def add(
        self,
        *,
        attempt_id: str,
        chunk_index: int,
        chunk_count: int,
        payload: bytes,
        payload_sha256: str,
    ) -> int:
        """接收一条分片，返回该 attempt 已收条数；不合法即抛错（不落盘半条）。"""
        key = str(attempt_id or "")
        if not key:
            raise _error("invalid_attempt", "attempt_id", "attempt id is required")
        if chunk_count < 1 or chunk_count > self.max_chunks:
            raise _error(
                "invalid_chunk_count", "chunk_count",
                f"chunk_count must be in 1..{self.max_chunks}",
            )
        if chunk_index < 0 or chunk_index >= chunk_count:
            raise _error(
                "invalid_chunk_index", "chunk_index",
                "chunk_index must be less than chunk_count",
            )
        chunk = bytes(payload or b"")
        if len(chunk) > self.max_chunk_bytes:
            raise _error(
                "chunk_too_large", "payload",
                f"a single chunk must not exceed {self.max_chunk_bytes} bytes",
            )
        actual = hashlib.sha256(chunk).hexdigest()
        if actual != str(payload_sha256 or "").lower():
            raise _error(
                "chunk_digest_mismatch", "payload_sha256",
                "chunk digest does not match its payload",
            )
        state = self._state.get(key)
        if state is None:
            state = _AttemptChunks(
                chunk_count=int(chunk_count), total_bytes=0, received={},
            )
            self._state[key] = state
        elif (
            state.chunk_count != int(chunk_count)
        ):
            raise _error(
                "chunk_layout_mismatch", "chunk_count",
                "chunk_count changed for an in-flight assembly",
            )
        if chunk_index in state.received:
            raise _error(
                "duplicate_chunk", "chunk_index",
                f"chunk {chunk_index} was already received",
            )
        if state.total_bytes + len(chunk) > self.max_total_bytes:
            raise _error(
                "stage_payload_too_large", "payload",
                f"assembled payload must not exceed {self.max_total_bytes} bytes",
            )
        state.received[chunk_index] = chunk
        state.total_bytes += len(chunk)
        return len(state.received)

    def received_count(self, attempt_id: str) -> int:
        state = self._state.get(str(attempt_id or ""))
        return 0 if state is None else len(state.received)

    def expected_count(self, attempt_id: str) -> int:
        state = self._state.get(str(attempt_id or ""))
        return 0 if state is None else state.chunk_count

    def assembled_bytes(self, attempt_id: str) -> int:
        state = self._state.get(str(attempt_id or ""))
        return 0 if state is None else state.total_bytes

    def missing_indices(self, attempt_id: str) -> list[int]:
        state = self._state.get(str(attempt_id or ""))
        if state is None:
            return []
        return [
            index for index in range(state.chunk_count)
            if index not in state.received
        ]

    def is_complete(self, attempt_id: str) -> bool:
        state = self._state.get(str(attempt_id or ""))
        return state is not None and len(state.received) == state.chunk_count

    def assemble(self, attempt_id: str) -> bytes:
        """集齐则返回拼装后的字节；未集齐抛 `incomplete_chunks`（带缺失序号）。"""
        key = str(attempt_id or "")
        state = self._state.get(key)
        if state is None:
            raise _error(
                "incomplete_chunks", "attempt_id",
                "no chunks were received for this attempt",
            )
        missing = self.missing_indices(key)
        if missing:
            raise _error(
                "incomplete_chunks", "chunk_index",
                f"missing chunk indices: {missing}",
            )
        return b"".join(state.received[index] for index in range(state.chunk_count))

    def discard(self, attempt_id: str) -> None:
        self._state.pop(str(attempt_id or ""), None)

    def pending(self) -> dict[str, int]:
        """诊断：`{attempt_id: 已收条数}`（只含未集齐的装配）。"""
        return {
            key: len(state.received)
            for key, state in self._state.items()
            if len(state.received) != state.chunk_count
        }


# ---------------------------------------------------------------------------
# 发送端：切分计划与建消息
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class StageChunkPlan:
    chunks: tuple[bytes, ...]
    total_bytes: int
    payload_sha256: str

    @property
    def chunk_count(self) -> int:
        return len(self.chunks)

    def chunk_digests(self) -> tuple[str, ...]:
        return tuple(hashlib.sha256(chunk).hexdigest() for chunk in self.chunks)


def plan_stage_payload_chunks(
    payload: bytes,
    *,
    chunk_bytes: int = STAGE_CHUNK_BYTES,
    max_chunks: int = MAX_STAGE_CHUNKS,
) -> StageChunkPlan:
    """把 raw payload 切成**有序**分片；超出硬上限即抛错（不做静默截断）。"""
    raw = bytes(payload or b"")
    if not raw:
        raise _error("invalid_payload", "payload", "payload must not be empty")
    if chunk_bytes < 1:
        raise _error("invalid_chunk_size", "chunk_bytes", "chunk_bytes must be >= 1")
    count = int(math.ceil(len(raw) / chunk_bytes))
    if count > max_chunks:
        raise _error(
            "stage_payload_too_large", "payload",
            f"payload needs {count} chunks, at most {max_chunks} are allowed",
        )
    chunks = tuple(
        raw[offset:offset + chunk_bytes]
        for offset in range(0, len(raw), chunk_bytes)
    )
    return StageChunkPlan(
        chunks=chunks,
        total_bytes=len(raw),
        payload_sha256=hashlib.sha256(raw).hexdigest(),
    )


def build_stage_chunk(
    *,
    workflow_id: str,
    stage_id: str,
    attempt_id: str,
    lease_id: str,
    lease_epoch: int,
    provider_id: str,
    chunk_index: int,
    chunk_count: int,
    payload: bytes,
    total_bytes: int,
    message_id: str,
    sent_at_ms: int,
    version: int = PROTOCOL_VERSION,
    message_type: str = "stage_chunk",
) -> Any:
    """构造一条分片消息（摘要按**本条 payload** 计算）。

    `message_type` 默认 `stage_chunk`（coordinator → worker 的输入分片）。
    ★ 2026-10-11（三机实测）：**回程**分片复用同一形状，调用方传
    `stage_result_chunk` —— 因此两侧装配器与切分计划可以完全共享，不需要第二套实现。
    """
    return build_message(
        str(message_type or "stage_chunk"),
        {
            "workflow_id": workflow_id,
            "stage_id": stage_id,
            "attempt_id": attempt_id,
            "lease_id": lease_id,
            "lease_epoch": int(lease_epoch),
            "provider_id": provider_id,
            "chunk_index": int(chunk_index),
            "chunk_count": int(chunk_count),
            "payload_b64": base64.b64encode(bytes(payload)).decode("ascii"),
            "payload_sha256": hashlib.sha256(bytes(payload)).hexdigest(),
            "total_bytes": int(total_bytes),
        },
        message_id=message_id,
        sent_at_ms=sent_at_ms,
        version=version,
    )


def stage_chunk_ref(plan: StageChunkPlan) -> dict[str, Any]:
    """`stage_offer.root_input` 里替代内联 payload 的引用对象。"""
    return {
        "chunk_count": plan.chunk_count,
        "total_bytes": plan.total_bytes,
        "payload_sha256": plan.payload_sha256,
    }
