"""DIST-NEXT-8 回归：请求相位与**互斥终态**。

审计「两者共同要求」要求一次请求的日志与响应 metrics 可重建，且「取消」「具名拒绝」
「链路失败」不被普通 `fallback` 覆盖。此前 metrics 只有 `fallback` 布尔与一段自由文本
`error`，聚合时这几类事件分不开。
"""

import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from request_outcome import (  # noqa: E402
    OUTCOME_CANCELLED,
    OUTCOME_COMPLETED,
    OUTCOME_FALLBACK_COMPLETED,
    OUTCOME_FAILED,
    OUTCOME_INCOMPLETE,
    OUTCOME_REFUSED,
    OUTCOMES,
    PHASE_FIELDS,
    REASON_GENERATION_CANCELLED,
    REASON_REQUEST_FAILED,
    REASON_REQUEST_REFUSED,
    derive_outcome,
    merge_phase_metrics,
    request_phase_metrics,
)


def test_outcome_is_mutually_exclusive():
    """优先级：取消 > 拒绝 > 失败 > 回退完成 > 完成 > 未完成。"""
    assert derive_outcome(cancelled=True, refused=True, failed=True,
                          completed=True, fallback=True) == OUTCOME_CANCELLED
    assert derive_outcome(refused=True, failed=True, completed=True,
                          fallback=True) == OUTCOME_REFUSED
    assert derive_outcome(failed=True, completed=True,
                          fallback=True) == OUTCOME_FAILED
    assert derive_outcome(completed=True, fallback=True) == OUTCOME_FALLBACK_COMPLETED
    assert derive_outcome(completed=True) == OUTCOME_COMPLETED
    assert derive_outcome() == OUTCOME_INCOMPLETE
    # 只有这四个「终态」允许出现在 outcome 里（incomplete 表示没有终态）
    assert {OUTCOME_CANCELLED, OUTCOME_REFUSED, OUTCOME_FAILED,
            OUTCOME_COMPLETED, OUTCOME_FALLBACK_COMPLETED,
            OUTCOME_INCOMPLETE} == set(OUTCOMES)


def test_cancellation_is_not_swallowed_by_fallback_or_error():
    """核心判据：取消带 fallback/错误文本时，终态仍是 `cancelled`。"""
    metrics = request_phase_metrics(
        admitted=True, started=True, cancelled=True,
        fallback=True, reason_code=REASON_GENERATION_CANCELLED,
    )

    assert metrics["outcome"] == OUTCOME_CANCELLED
    assert metrics["cancelled"] is True
    assert metrics["fallback"] is True          # 相位如实保留
    assert metrics["outcome_reason"] == REASON_GENERATION_CANCELLED
    # 终态只有 cancelled 一个 —— 不会被 `fallback` 伪装成 fallback_completed
    assert metrics["outcome"] != OUTCOME_FALLBACK_COMPLETED


def test_refusal_and_link_failure_are_distinct():
    refused = request_phase_metrics(started=True, refused=True)
    failed = request_phase_metrics(started=True, failed=True)

    assert refused["outcome"] == OUTCOME_REFUSED
    assert refused["refused"] is True and refused["failed"] is False
    assert failed["outcome"] == OUTCOME_FAILED
    assert failed["failed"] is True and failed["refused"] is False


def test_phase_field_set_is_stable():
    metrics = request_phase_metrics()

    assert set(metrics) == set(PHASE_FIELDS) | {"outcome", "outcome_reason"}
    assert metrics["outcome"] == OUTCOME_INCOMPLETE
    assert metrics["outcome_reason"] == ""
    # 布尔化：给非布尔值也只产出布尔相位
    coerced = request_phase_metrics(admitted=1, started="yes", completed=0)
    assert coerced["admitted"] is True
    assert coerced["started"] is True
    assert coerced["completed"] is False


def test_merge_keeps_existing_metrics():
    """相位合并进既有 metrics 时不得丢掉其它键。"""
    merged = merge_phase_metrics(
        {"engine": "llama_cpp", "fallback_reason": "x", "config_id": "cfg-1"},
        admitted=True, started=True, completed=True,
    )

    assert merged["engine"] == "llama_cpp"
    assert merged["fallback_reason"] == "x"
    assert merged["config_id"] == "cfg-1"
    assert merged["outcome"] == OUTCOME_COMPLETED
    assert merged["admitted"] is True

    # 空 metrics 也安全
    assert merge_phase_metrics(None, completed=True)["outcome"] == OUTCOME_COMPLETED


def test_reason_codes_are_stable_and_greppable():
    assert REASON_GENERATION_CANCELLED == "generation_cancelled"
    assert REASON_REQUEST_FAILED == "request_failed"
    assert REASON_REQUEST_REFUSED == "request_refused"


def test_typed_reason_preserves_causal_chain_without_text_parsing():
    from qlh_core.contracts.snapshots import reason
    cause = reason("worker", "worker_stage_offer_not_ready", "worker_receipt", retryability="retry")
    value = reason("request", "deployment_unavailable", "admission", cause=cause,
                   details={"message": "a: b | readiness=opaque"})
    assert value["cause"]["code"] == "worker_stage_offer_not_ready"
    assert value["details"]["message"] == "a: b | readiness=opaque"
    import api_server
    assert not hasattr(api_server, "parse_pipeline_readiness")
