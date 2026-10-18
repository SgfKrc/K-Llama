"""DIST-NEXT-2b 接线回归（主仓两侧）：能力位 + 发送分片 + 接收装配。

判据：
1. 只有声明 `stage_chunked_input == true` 的 worker 才会收到分片；
2. 分片在 `stage_offer` **之前**发出，offer 的 `root_input` 用 `hidden_ref` 取代内联
   `hidden_f32`，且 `input_sha256` 按改写后的 root_input 计算；
3. 未声明能力 / 未超预算 ⇒ 与接线前**逐字节一致**（内联路径原样透传）；
4. 接收侧装配 fail-closed：分片未齐备或摘要不符都以具名原因失败。
"""

import base64
import hashlib
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import api_server  # noqa: F401,E402
from scheduler import Scheduler  # noqa: E402
from execution_provider import (  # noqa: E402
    ModelIdentity,
    StageAttempt,
    StageRequest,
)
from task_worker_adapter import (  # noqa: E402
    RemoteFullWorkerProvider,
)
from task_worker_chunks import StageChunkAssembler  # noqa: E402
from task_worker_protocol import (  # noqa: E402
    WorkerProtocolError,
    build_message,
    stage_input_sha256,
)

# 1 token × 2_000_000 维 f32 = 8 MB raw ⇒ base64 ≈ 10.7 MB > 帧预算（8 MiB − 256 KiB）
OVERSIZED_EMBD = 2_000_000
OVERSIZED_RAW = b"\x01" * (OVERSIZED_EMBD * 4)


def _snapshot(*, chunked: bool):
    capabilities = {
        "stage_types": ["layer_forward"],
        "engines": ["llama_cpp"],
        "models": [{
            "model_id": "qwen35_2b_mid4_16",
            "engine": "llama_cpp",
            "format": "gguf",
            "revision": "local-v1",
            "sha256": "b" * 64,
        }],
        "max_concurrency": 1,
        "layer_ranges": [[4, 16]],
    }
    if chunked:
        capabilities["stage_chunked_input"] = True
    return {
        "capabilities": capabilities,
        "healthy": True,
        "selected_version": 3,
        "worker_kind": "pc_full_worker",
    }


def _stage_request(provider_id, *, hidden_b64: str, embd: int):
    return StageRequest(
        workflow_id="wf_chunkedinput1",
        request_id="request-chunked-input-1",
        stage_id="candidate_a",
        stage_type="layer_forward",
        provider_id=provider_id,
        dependencies={},
        root_input={
            "hidden_f32": hidden_b64,
            "context_size": 2048,
            "pos_base": 0,
            "want_hidden": True,
        },
        model_identity=ModelIdentity(
            model_id="qwen35_2b_mid4_16", engine="llama_cpp", format="gguf",
            revision="local-v1", sha256="b" * 64,
        ),
        stage_fields={
            "layer_range": [4, 16],
            "handoff_at": 16,
            "hidden_sha256": hashlib.sha256(OVERSIZED_RAW).hexdigest(),
            "hidden_spec": {"n_tokens": 1, "n_embd": embd, "dtype": "float32"},
            "middle_channel": "keep_head_layer_out",
        },
    )


def _attempt(provider_id, *, hidden_b64, embd, attempt_id="att_chunkedinput1"):
    return StageAttempt(
        attempt_id=attempt_id,
        request=_stage_request(provider_id, hidden_b64=hidden_b64, embd=embd),
        provider_id=provider_id,
        lease_id="lease_chunkedinput1",
        lease_epoch=1,
        lease_expires_at=1_700_000_000.0,
    )


def _provider(snapshot, sent):
    return RemoteFullWorkerProvider(
        node_id="worker_01",
        peer_snapshot=lambda: snapshot,
        send_message=sent.append,
    )


def test_capability_flag_is_admitted_and_validated():
    """主仓侧放行 `stage_chunked_input`（布尔），并拒绝非布尔值。"""
    def hello(capabilities):
        return build_message(
            "hello",
            {
                "node_id": "pc_worker_01",
                "worker_kind": "pc_full_worker",
                "min_version": 1,
                "max_version": 3,
                "capabilities": capabilities,
            },
            message_id="msg_hello_chunkflag1",
            sent_at_ms=1_700_000_000_000,
            version=3,
        )

    base = {
        "stage_types": ["layer_forward"],
        "engines": ["llama_cpp"],
        "models": [{
            "model_id": "qwen35_2b_mid4_16",
            "engine": "llama_cpp",
            "format": "gguf",
            "revision": "local-v1",
            "sha256": "b" * 64,
        }],
        "max_concurrency": 1,
    }
    assert hello({**base, "stage_chunked_input": True}).payload["capabilities"][
        "stage_chunked_input"
    ] is True

    with pytest.raises(WorkerProtocolError) as invalid:
        hello({**base, "stage_chunked_input": "yes"})
    assert invalid.value.code == "invalid_boolean"


def test_declared_chunked_worker_receives_chunks_before_the_offer():
    sent = []
    provider = _provider(_snapshot(chunked=True), sent)
    attempt = _attempt(
        provider.provider_id,
        hidden_b64=base64.b64encode(OVERSIZED_RAW).decode("ascii"),
        embd=OVERSIZED_EMBD,
    )

    root_input = provider._maybe_send_stage_chunks(attempt)

    chunked = [m for m in sent if m.message_type == "stage_chunk"]
    assert len(chunked) == 8                       # ceil(8_000_000 / 1 MiB)
    assert [m.payload["chunk_index"] for m in chunked] == list(range(8))
    assert all(m.payload["chunk_count"] == 8 for m in chunked)
    assert all(m.payload["attempt_id"] == attempt.attempt_id for m in chunked)
    # 逐片摘要 + 总量自洽
    assert sum(m.payload["total_bytes"] == len(OVERSIZED_RAW) for m in chunked) == 8
    for message in chunked:
        payload = base64.b64decode(message.payload["payload_b64"])
        assert hashlib.sha256(payload).hexdigest() == message.payload["payload_sha256"]

    # 改写后的 root_input：内联 hidden 消失、换成引用
    assert "hidden_f32" not in root_input
    ref = root_input["hidden_ref"]
    assert ref["chunk_count"] == 8
    assert ref["total_bytes"] == len(OVERSIZED_RAW)
    assert ref["payload_sha256"] == hashlib.sha256(OVERSIZED_RAW).hexdigest()
    # 其它字段原样保留（context_size / pos_base / want_hidden）
    assert root_input["want_hidden"] is True


def test_undeclared_worker_keeps_the_inline_path():
    """未声明能力 ⇒ 不发分片、root_input 原样（零行为变化）。"""
    sent = []
    provider = _provider(_snapshot(chunked=False), sent)
    hidden_b64 = base64.b64encode(OVERSIZED_RAW).decode("ascii")
    attempt = _attempt(provider.provider_id, hidden_b64=hidden_b64, embd=OVERSIZED_EMBD)

    root_input = provider._maybe_send_stage_chunks(attempt)

    assert sent == []
    assert root_input == attempt.request.root_input


def test_small_hidden_never_chunks_even_when_declared():
    sent = []
    provider = _provider(_snapshot(chunked=True), sent)
    raw = b"\x02" * (1024 * 4)
    attempt = _attempt(
        provider.provider_id,
        hidden_b64=base64.b64encode(raw).decode("ascii"),
        embd=1024,
    )

    root_input = provider._maybe_send_stage_chunks(attempt)

    assert sent == []
    assert root_input == attempt.request.root_input


def test_worker_assembles_hidden_ref_and_verifies_the_digest():
    raw = b"\x03" * 5000
    assembler = StageChunkAssembler()
    worker = Scheduler()
    plan_chunks = [
        raw[offset:offset + 2000] for offset in range(0, len(raw), 2000)
    ]
    for index, chunk in enumerate(plan_chunks):
        assembler.add(
            attempt_id="att_chunkedinput2", chunk_index=index,
            chunk_count=len(plan_chunks), payload=chunk,
            payload_sha256=hashlib.sha256(chunk).hexdigest(),
        )
    worker._task_worker_chunk_state = assembler
    offer = {
        "attempt_id": "att_chunkedinput2",
        "stage_type": "layer_forward",
        "hidden_sha256": hashlib.sha256(raw).hexdigest(),
        "root_input": {
            "hidden_ref": {
                "chunk_count": len(plan_chunks),
                "total_bytes": len(raw),
                "payload_sha256": hashlib.sha256(raw).hexdigest(),
            },
            "want_hidden": True,
        },
    }

    assembled = worker._assemble_stage_root_input(offer)

    assert base64.b64decode(assembled["hidden_f32"]) == raw
    assert assembled["want_hidden"] is True
    # 改写后移除 `hidden_ref`（执行侧只应看到一个来源）
    assert "hidden_ref" not in assembled
    # 装配完成后分片状态被丢弃（不长期占用内存）
    assert assembler.pending() == {}


def test_worker_fails_closed_when_chunks_are_incomplete_or_mismatched():
    worker = Scheduler()
    incomplete_offer = {
        "attempt_id": "att_missing",
        "stage_type": "layer_forward",
        "hidden_sha256": hashlib.sha256(b"x").hexdigest(),
        "root_input": {"hidden_ref": {"chunk_count": 2, "total_bytes": 2,
                                      "payload_sha256": hashlib.sha256(b"xx").hexdigest()}},
    }
    with pytest.raises(RuntimeError) as incomplete:
        worker._assemble_stage_root_input(incomplete_offer)
    assert "分片未齐备" in str(incomplete.value)

    assembler = StageChunkAssembler()
    assembler.add(
        attempt_id="att_baddigest", chunk_index=0, chunk_count=1,
        payload=b"abc", payload_sha256=hashlib.sha256(b"abc").hexdigest(),
    )
    worker._task_worker_chunk_state = assembler
    mismatch_offer = {
        "attempt_id": "att_baddigest",
        "stage_type": "layer_forward",
        "hidden_sha256": hashlib.sha256(b"different").hexdigest(),
        "root_input": {"hidden_ref": {"chunk_count": 1, "total_bytes": 3,
                                      "payload_sha256": hashlib.sha256(b"abc").hexdigest()}},
    }
    with pytest.raises(RuntimeError) as mismatch:
        worker._assemble_stage_root_input(mismatch_offer)
    assert "摘要不符" in str(mismatch.value)


def test_inline_root_input_is_passed_through_unchanged():
    """内联路径（无 hidden_ref）逐字节透传 —— 接线不影响既有链路。"""
    worker = Scheduler()
    root = {"hidden_f32": "AAAA", "context_size": 2048, "want_hidden": False}

    assembled = worker._assemble_stage_root_input(
        {"attempt_id": "att_inline", "root_input": root},
    )

    assert assembled is root


def test_offer_digest_is_computed_over_the_rewritten_root_input():
    """改写后的 root_input 决定 `input_sha256`（两端一致，否则 worker 校验失败）。"""
    sent = []
    provider = _provider(_snapshot(chunked=True), sent)
    attempt = _attempt(
        provider.provider_id,
        hidden_b64=base64.b64encode(OVERSIZED_RAW).decode("ascii"),
        embd=OVERSIZED_EMBD,
    )
    root_input = provider._maybe_send_stage_chunks(attempt)

    assert stage_input_sha256(root_input, {}) == stage_input_sha256(
        root_input, attempt.request.dependencies,
    )
    assert stage_input_sha256(root_input, {}) != stage_input_sha256(
        attempt.request.root_input, {},
    )
