"""回程分片 `stage_result_chunk` 的协议与装配往返。

★ 2026-10-11（ROG+Y700+本机三机实测）：协议此前只做了**输入方向**分片，中间段 hidden
回程必须整体塞进 `MAX_MESSAGE_BYTES`（8 MiB）单帧 ⇒ 长 prefill（实测 1004 tokens，
f32 ≈ 8.2 MiB）必然 `message exceeds maximum size`。本文件锁定回程分片的两个不变量：
协议接受该类型（字段与 `stage_chunk` 同形），且既有装配器能字节级还原。
"""

from __future__ import annotations

import hashlib
import sys

import pytest

sys.path.insert(0, "src")

from task_worker_chunks import (  # noqa: E402
    StageChunkAssembler,
    build_stage_chunk,
    plan_stage_payload_chunks,
    stage_chunk_ref,
)
from task_worker_protocol import STAGE_CHUNK_BYTES, decode_message  # noqa: E402

_ATTEMPT = "att_" + "b" * 12


def _identity() -> dict:
    return {
        "workflow_id": "wf_" + "a" * 12,
        "stage_id": "node:step:0",
        "attempt_id": _ATTEMPT,
        "lease_id": "lease_" + "c" * 12,
        "lease_epoch": 1,
    }


def _payload() -> bytes:
    # 略大于两片，确保 chunk_count == 3（边界也要被覆盖）。
    return bytes((index * 7) % 256 for index in range(2 * STAGE_CHUNK_BYTES + 3))


def test_stage_result_chunk_passes_protocol_validation_and_round_trips():
    raw = _payload()
    plan = plan_stage_payload_chunks(raw)
    assert plan.chunk_count == 3

    message = build_stage_chunk(
        **_identity(),
        provider_id="remote_node-1",
        chunk_index=0,
        chunk_count=plan.chunk_count,
        payload=plan.chunks[0],
        total_bytes=plan.total_bytes,
        message_id="msg_" + "d" * 12,
        sent_at_ms=1,
        message_type="stage_result_chunk",
    )

    decoded = decode_message(message.snapshot())
    assert decoded.message_type == "stage_result_chunk"
    assert decoded.payload["chunk_count"] == plan.chunk_count
    # 与输入方向同形：字段集合一致（只有 message_type 不同）。
    assert set(decoded.payload) == {
        "workflow_id", "stage_id", "attempt_id", "lease_id", "lease_epoch",
        "provider_id", "chunk_index", "chunk_count", "payload_b64",
        "payload_sha256", "total_bytes",
    }


def test_assembler_restores_exact_bytes_and_reports_missing_chunks():
    raw = _payload()
    plan = plan_stage_payload_chunks(raw)
    assembler = StageChunkAssembler()

    for index, chunk in enumerate(plan.chunks):
        assembler.add(
            attempt_id=_ATTEMPT,
            chunk_index=index,
            chunk_count=plan.chunk_count,
            payload=chunk,
            payload_sha256=hashlib.sha256(chunk).hexdigest(),
        )

    assert assembler.is_complete(_ATTEMPT)
    restored = assembler.assemble(_ATTEMPT)
    assert restored == raw
    # 装配后的摘要必须与 worker 放进 `hidden_out_ref` 的摘要一致（对端据此校验）。
    assert hashlib.sha256(restored).hexdigest() == stage_chunk_ref(plan)["payload_sha256"]

    # 缺片 ⇒ fail-closed（带缺失序号）。
    partial = StageChunkAssembler()
    partial.add(
        attempt_id=_ATTEMPT, chunk_index=0, chunk_count=2, payload=b"x",
        payload_sha256=hashlib.sha256(b"x").hexdigest(),
    )
    with pytest.raises(Exception) as incomplete:
        partial.assemble(_ATTEMPT)
    assert incomplete.value.code == "incomplete_chunks"


def test_chunk_digest_mismatch_is_rejected():
    assembler = StageChunkAssembler()
    with pytest.raises(Exception) as mismatch:
        assembler.add(
            attempt_id=_ATTEMPT, chunk_index=0, chunk_count=1, payload=b"payload",
            payload_sha256=hashlib.sha256(b"other").hexdigest(),
        )
    assert mismatch.value.code == "chunk_digest_mismatch"
