"""DIST-NEXT-2 回归：层段 hidden 的 **dispatch 前** wire 大小预检。

背景
----
层段的 `root_input.hidden_f32` 与中间段的 `output.hidden_out_f32` 都是 raw 数据经
base64 承载，而单条 task-worker 消息的上限是 `MAX_MESSAGE_BYTES`（8 MiB）。旧实现
没有任何预检 ⇒ 合法的长 prefill / 大 `n_embd` 会**先通过准入**，在 worker 执行完成
后才由协议层抛 `message_too_large`（表现为 stage 超时、回退或被 `distributed_required`
拒绝）—— 这是过晚的契约发现，而不是有效的 fail-closed。

本文件锁定两条不变量：
1. 预算函数与协议常量同源（`task_worker_protocol`），f32 / f16 双档一致；
2. `_execute_layer_stage_offer` 在 **reserve/execute 之前**就以稳定 reason code 拒绝。
"""

import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np  # noqa: E402

import api_server  # noqa: F401,E402  （加载 API composition root）
from scheduler import Scheduler  # noqa: E402
from scheduler_pipeline import (  # noqa: E402
    LayerStageFrameTooLarge,
    _assert_layer_stage_offer_fits_frame,
)
from task_worker_protocol import (  # noqa: E402
    MAX_MESSAGE_BYTES,
    MAX_STAGE_PAYLOAD_BYTES,
    STAGE_CHUNK_BYTES,
    STAGE_FRAME_RESERVE_BYTES,
    WorkerProtocolError,
    _base64_wire_length,
    hidden_fits_stage_frame,
    hidden_wire_bytes,
    max_hidden_tokens,
    stage_payload_budget_bytes,
)


def test_hidden_wire_bytes_includes_base64_expansion():
    # 1024 个 f32 元素 = 4096 B ⇒ base64 后 4 × ceil(4096/3) = 5464 字符
    assert hidden_wire_bytes(1, 1024, "float32") == 5464
    # f16 是同一公式的一半
    assert hidden_wire_bytes(1, 1024, "float16") == 2732
    # 1 MiB 边界：必须与真实 base64 长度一致（近似公式会少算 2 个字符）
    assert _base64_wire_length(STAGE_CHUNK_BYTES) == 1_398_104
    with pytest.raises(WorkerProtocolError) as captured:
        hidden_wire_bytes(1, 1024, "bfloat16")
    assert captured.value.code == "unsupported_hidden_dtype"
    with pytest.raises(WorkerProtocolError) as invalid:
        hidden_wire_bytes(0, 1024)
    assert invalid.value.code == "invalid_hidden_spec"


def test_stage_frame_budget_tracks_protocol_limit():
    assert stage_payload_budget_bytes() == MAX_MESSAGE_BYTES - STAGE_FRAME_RESERVE_BYTES
    # 单帧预算内的最大 token 数：再 +1 就装不下（判据自洽）
    n_embd = 2048
    limit = max_hidden_tokens(n_embd, "float32")
    assert hidden_fits_stage_frame(limit, n_embd, "float32")
    assert not hidden_fits_stage_frame(limit + 1, n_embd, "float32")
    # f16 能把同一预算装下约两倍的 token
    assert max_hidden_tokens(n_embd, "float16") >= 2 * limit - 1


def test_precheck_passes_within_budget():
    _assert_layer_stage_offer_fits_frame(
        node_id="worker-b", n_tokens=512, n_embd=2048,
    )


def test_precheck_rejects_oversized_hidden_with_stable_reason():
    with pytest.raises(LayerStageFrameTooLarge) as captured:
        _assert_layer_stage_offer_fits_frame(
            node_id="worker-b", n_tokens=1, n_embd=2_000_000,
        )
    error = captured.value
    assert error.code == "route_a_stage_frame_too_large"
    text = str(error)
    assert text.startswith(error.code)
    assert f"wire={error.wire_bytes}" in text
    assert f"budget={error.budget_bytes}" in text
    assert error.wire_bytes > error.budget_bytes


def test_layer_stage_offer_rejects_before_touching_the_provider():
    """超预算的 hidden 必须在 reserve/execute **之前**结束。"""
    sched = Scheduler()
    provider_calls = []

    def _provider(node_id):
        provider_calls.append(node_id)
        raise AssertionError("预检失败时不得触碰远端 provider")

    sched._ensure_remote_task_worker_provider = _provider  # type: ignore[assignment]

    hidden = np.zeros((1, 2_000_000), dtype=np.float32)

    with pytest.raises(LayerStageFrameTooLarge) as captured:
        sched._execute_layer_stage_offer(
            node_id="worker-b",
            assignment={"start_layer": 2, "end_layer": 4},
            hidden_states=hidden,
            model_identity=object(),
            workflow_id="wf_framebudget01",
            request_id="request-framebudget01",
            stage_id="worker-b:step:0",
            context_size=2048,
        )

    assert captured.value.code == "route_a_stage_frame_too_large"
    assert provider_calls == []


def test_precheck_allows_oversized_hidden_when_peer_declares_chunks():
    """★ 2026-10-08（DIST-NEXT-2b）：对端声明 `stage_chunked_input` 时，预检**不得**拒绝。

    否则 provider 层（`task_worker_adapter._maybe_send_stage_chunks`）的 `stage_chunk`
    分片路径**永远走不到** —— 这正是 2b 实测到的现象：≈2475 tokens 的 prompt 只能得到
    `route_a_stage_frame_too_large:…:wire=24160940:budget=8126464`。
    """
    _assert_layer_stage_offer_fits_frame(
        node_id="worker-b", n_tokens=1, n_embd=2_000_000, chunked_input=True,
    )


def test_node_declares_stage_chunked_input_reads_capabilities():
    """★ 2026-10-08（DIST-NEXT-2b）：能力判定三态 —— 命中 / 未命中 / 查询失败。"""
    from scheduler_pipeline import _node_declares_stage_chunked_input

    class _Control:
        def __init__(self, workers=None, boom=False):
            self._workers = workers or []
            self._boom = boom

        def status(self, role: str = "master"):
            assert role == "master"
            if self._boom:
                raise RuntimeError("control plane down")
            return {"workers": self._workers}

    def _worker(node_id, value):
        return {"node_id": node_id, "capabilities": {"stage_chunked_input": value}}

    # 无控制面 / 查询失败 ⇒ 保守为「不支持」（维持 fail-closed）
    assert _node_declares_stage_chunked_input(None, "w1") is False
    assert _node_declares_stage_chunked_input(_Control(boom=True), "w1") is False
    # 未命中节点、或声明值不是布尔真 ⇒ False
    assert _node_declares_stage_chunked_input(_Control([_worker("w2", True)]), "w1") is False
    assert _node_declares_stage_chunked_input(_Control([_worker("w1", "yes")]), "w1") is False
    # 命中且为 True ⇒ True
    assert _node_declares_stage_chunked_input(_Control([_worker("w1", True)]), "w1") is True


def test_layer_stage_offer_reaches_provider_when_peer_declares_chunks():
    """声明分片的节点：预检放行，流程继续到 provider —— 分片发送就发生在那里。"""
    sched = Scheduler()
    touched = []

    def _provider(node_id):
        touched.append(node_id)
        raise AssertionError("分片路径：预检放行后应触碰 provider")

    sched._ensure_remote_task_worker_provider = _provider  # type: ignore[assignment]
    # ★ 票 12 WORKER-PREPARE-01：层段 offer 要求 assignment 的 worker native prepare
    #   已就绪（`scheduler_pipeline.py:2500`，否则抛 RuntimeError）。本用例只验证帧预算
    #   预检放行与分片路径，故把 prepare 就绪判据直接置为通过。
    sched._worker_prepare_assignment_loaded = lambda node_id, assignment: (True, "")  # type: ignore[assignment]
    sched._task_worker_control = type("_Control", (), {
        "status": lambda self, role="master": {
            "workers": [{
                "node_id": "worker-b",
                "capabilities": {"stage_chunked_input": True},
            }],
        },
    })()

    hidden = np.zeros((1, 2_000_000), dtype=np.float32)

    with pytest.raises(AssertionError):
        sched._execute_layer_stage_offer(
            node_id="worker-b",
            assignment={
                "start_layer": 2,
                "end_layer": 4,
                # ★ 票 12 WORKER-PREPARE-01：阶段转发前会用 `stage_identity()` 校验
                #   （worker_prepare_contract.py STAGE_IDENTITY_FIELDS 五个字段齐全才通过）。
                "worker_prepare_identity": {
                    "operation_id": "op-framebudget02",
                    "config_id": "cfg-framebudget02",
                    "generation": 1,
                    "assignment_id": "asg-framebudget02",
                    "artifact_sha256": "a" * 64,
                },
            },
            hidden_states=hidden,
            model_identity=object(),
            workflow_id="wf_framebudget02",
            request_id="request-framebudget02",
            stage_id="worker-b:step:0",
            context_size=2048,
        )

    assert touched == ["worker-b"]


def test_precheck_rejects_when_chunked_total_exceeds_chunk_limit():
    """★ 2026-10-08（DIST-NEXT-2b）：分片只放宽**单帧**预算。

    总量越过 `MAX_STAGE_CHUNKS × STAGE_CHUNK_BYTES` 时仍须在 dispatch 前具名失败，
    否则会在 provider 的分片计划里才抛 `stage_payload_too_large` —— 又是过晚的契约发现。
    """
    limit_wire = int(_base64_wire_length(int(MAX_STAGE_PAYLOAD_BYTES)))

    inside = 8000  # 8000 × 2048 × 4 B ≈ 63 片，正好在 64 片上限内
    assert hidden_wire_bytes(inside, 2048, "float32") <= limit_wire
    _assert_layer_stage_offer_fits_frame(
        node_id="worker-b", n_tokens=inside, n_embd=2048, chunked_input=True,
    )

    outside = 9000  # 超过 64 片 ⇒ 分片也装不下
    assert hidden_wire_bytes(outside, 2048, "float32") > limit_wire
    with pytest.raises(LayerStageFrameTooLarge) as captured:
        _assert_layer_stage_offer_fits_frame(
            node_id="worker-b", n_tokens=outside, n_embd=2048, chunked_input=True,
        )
    error = captured.value
    assert error.code == "route_a_stage_frame_too_large"
    assert error.budget_bytes == limit_wire
    assert error.wire_bytes > error.budget_bytes
