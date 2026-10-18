"""Capacity diagnostics are a display projection, never another admission solver."""
from copy import deepcopy

import pytest

from tui_shared import format_model_intent, model_intent_ready


CAPACITY_CODE = "pipeline_distributed_capacity_insufficient"
HOST_CAPACITY_CODE = "SUPPLY_RUNTIME_HOST_CAPACITY_INSUFFICIENT"
MIB = 1024**2
GIB = 1024**3


def capacity_operation(*, code=CAPACITY_CODE, scope="candidate_plan"):
    return {
        "schema_version": 2, "revision": 7, "identity": {"boot_id": "tui-test"}, "observed_at": 100, "validity": "valid", "runnable": False,
        "operation_id": "supply_" + "d" * 32,
        "intent": {"model_id": "qwen35-9b"},
        "status": "waiting_nodes",
        "reason_code": code,
        "reason": "当前在线节点无法容纳连续层段",
        "loaded_ready": False,
        "nodes": {
            "master": {"status": "online"},
            "rog": {"status": "online"},
            "y700": {"status": "online"},
        },
        "capacity_diagnostic": {
            "reason_code": code,
            "reason": "候选层段超出节点预算",
            "scope": scope,
            "nodes": [{
                "node_id": "rog",
                "engine": "llama_cpp",
                "execution_device": "cpu",
                "resource": "ram",
                "available_bytes": GIB,
                "required_bytes": GIB + 256 * MIB,
                "deficit_bytes": 256 * MIB,
                "reserve_bytes": 128 * MIB,
                "layer_range": [20, 24],
            }],
            "suggested_action": "释放节点内存或增加可用节点后重新规划",
        },
    }


def test_online_nodes_waiting_for_capacity_show_owner_budget_and_keep_status():
    operation = capacity_operation()
    before = deepcopy(operation)

    text = format_model_intent(operation)

    assert text.splitlines()[0] == "qwen35-9b · 等待容量 (waiting_nodes)"
    for fragment in (
        CAPACITY_CODE, "当前在线节点无法容纳连续层段", "候选层段超出节点预算",
        "容量不足（候选方案", "释放所示缺口不保证可执行", "rog · llama.cpp · CPU · RAM",
        "层段 [20, 24)", "可用 1.00 GiB (1073741824 bytes)",
        "所需 1.25 GiB (1342177280 bytes)", "缺口 256.00 MiB (268435456 bytes)",
        "含边界的预留 128.00 MiB (134217728 bytes)", "建议：释放节点内存或增加可用节点后重新规划",
        "master: online", "rog: online", "y700: online", "loaded_ready: false",
    ):
        assert fragment in text
    assert not model_intent_ready(operation)
    assert operation == before


def test_actual_worker_joining_ignores_stale_capacity_diagnostic():
    operation = capacity_operation()
    operation.update(reason_code="SUPPLY_WAITING_NODES", reason="等待从节点完成加入")
    operation["nodes"] = {"rog": {"status": "joining"}}
    operation["capacity_diagnostic"]["pending_node_ids"] = ["joining-worker"]

    text = format_model_intent(operation)

    assert text.splitlines()[0] == "qwen35-9b · 等待节点 (waiting_nodes)"
    assert "SUPPLY_WAITING_NODES: 等待从节点完成加入" in text
    assert "rog: joining" in text
    for stale in ("等待容量", "容量不足（", "候选层段超出节点预算", "1342177280 bytes", "建议：",
                  "仍在加入的节点", "joining-worker"):
        assert stale not in text


def test_host_staging_is_not_reported_as_cuda_device_memory():
    operation = capacity_operation(code=HOST_CAPACITY_CODE, scope="minimum_segment")
    operation["capacity_diagnostic"]["nodes"][0].update(
        engine="pytorch", execution_device="cuda", resource="host_ram",
        layer_range=[0, 4], available_bytes=33554432, required_bytes=67154432,
        deficit_bytes=33600000,
    )

    text = format_model_intent(operation)

    assert "等待容量 (waiting_nodes)" in text
    assert "容量不足（最小层段" in text
    assert "rog · PyTorch · CUDA · 主机暂存 RAM" in text
    assert " · 预留 128.00 MiB (134217728 bytes)" in text
    assert "含边界的预留" not in text
    assert "显存" not in text
    assert "层段 [0, 4)" in text
    assert "可用 32.00 MiB (33554432 bytes)" in text
    assert "所需 64.04 MiB (67154432 bytes)" in text
    assert "缺口 32.04 MiB (33600000 bytes)" in text
    assert not model_intent_ready(operation)


def test_vram_label_is_explicit_and_keeps_zero_deficit_from_owner():
    operation = capacity_operation()
    operation["capacity_diagnostic"]["nodes"].append({
        "node_id": "master", "engine": "pytorch", "execution_device": "cuda", "resource": "vram",
        "available_bytes": 4 * GIB, "required_bytes": 3 * GIB, "deficit_bytes": 0,
    })

    text = format_model_intent(operation)

    assert "master · PyTorch · CUDA · 显存（VRAM）" in text
    assert "可用 4.00 GiB (4294967296 bytes)" in text
    assert "所需 3.00 GiB (3221225472 bytes) · 缺口 0 bytes" in text
    assert not model_intent_ready(operation)


@pytest.mark.parametrize("status, ready, label", [
    ("ready", True, "就绪"), ("ready", False, "等待加载确认"),
    ("cancelled", False, "已取消"),
    ("superseded", False, "已被替代"), ("planning", False, "规划层段"),
    ("preparing", False, "准备中"),
])
@pytest.mark.parametrize("code", [CAPACITY_CODE, "pipeline_distributed_workers_unavailable"])
def test_old_diagnostic_cannot_override_current_lifecycle(status, ready, label, code):
    operation = capacity_operation(code=code)
    operation.update(status=status, loaded_ready=ready, runnable=ready)
    operation["capacity_diagnostic"]["pending_node_ids"] = ["joining-worker"]

    text = format_model_intent(operation)

    assert text.splitlines()[0] == f"qwen35-9b · {label} ({status})"
    for stale in ("等待容量", "容量不足（", "候选层段超出节点预算", "1342177280 bytes", "建议：",
                  "仍在加入的节点", "joining-worker"):
        assert stale not in text
    assert model_intent_ready(operation) is ready


@pytest.mark.parametrize("status, label", [("waiting_nodes", "等待容量"), ("failed", "失败")])
def test_diagnostic_from_another_reason_is_not_displayed(status, label):
    operation = capacity_operation(code=HOST_CAPACITY_CODE)
    operation["status"] = status
    operation["capacity_diagnostic"]["reason_code"] = CAPACITY_CODE
    operation["capacity_diagnostic"]["pending_node_ids"] = ["joining-worker"]

    text = format_model_intent(operation)

    assert text.splitlines()[0] == f"qwen35-9b · {label} ({status})"
    assert "仍在加入的节点" not in text
    assert "joining-worker" not in text
    assert HOST_CAPACITY_CODE in text
    assert "候选层段超出节点预算" not in text
    assert "1342177280 bytes" not in text
    assert "建议：" not in text


@pytest.mark.parametrize("code", [
    "pipeline_cluster_capacity_insufficient", "pipeline_capacity_single_node_insufficient",
    "pipeline_capacity_manual_insufficient", "pipeline_reshard_capacity_insufficient",
    "node_capacity_unavailable", "SUPPLY_CAPACITY_INSUFFICIENT", "SUPPLY_MASTER_CAPACITY_INSUFFICIENT",
])
def test_other_explicit_capacity_rejections_use_the_same_projection(code):
    text = format_model_intent(capacity_operation(code=code))
    assert "等待容量 (waiting_nodes)" in text
    assert code in text
    assert "容量不足（候选方案" in text


@pytest.mark.parametrize("diagnostic", [None, {}, [], "legacy"])
def test_no_usable_diagnostic_does_not_invent_numbers(diagnostic):
    operation = capacity_operation()
    operation["capacity_diagnostic"] = diagnostic
    operation["raw_model_deficit"] = 123456789
    operation["plan"] = {"raw_model_deficit": 987654321}

    text = format_model_intent(operation)

    assert "等待容量 (waiting_nodes)" in text
    assert "可用 " not in text
    assert "所需 " not in text
    assert "缺口 " not in text
    assert "123456789" not in text
    assert "987654321" not in text


def test_partial_diagnostic_shows_only_available_owner_fields():
    operation = capacity_operation(scope="minimum_segment")
    operation["capacity_diagnostic"]["nodes"] = [None, {
        "node_id": "y700", "resource": "ram", "available_bytes": 5 * GIB,
        "raw_model_deficit": 1122334455,
    }]

    text = format_model_intent(operation)

    assert "y700 · RAM · 可用 5.00 GiB (5368709120 bytes)" in text
    assert "所需 " not in text
    assert "缺口 " not in text
    assert "预留 " not in text
    assert "层段 [" not in text
    assert "1122334455" not in text


def test_absent_deficit_is_not_computed_even_when_both_budgets_are_present():
    operation = capacity_operation()
    del operation["capacity_diagnostic"]["nodes"][0]["deficit_bytes"]

    text = format_model_intent(operation)

    assert "可用 1.00 GiB" in text
    assert "所需 1.25 GiB" in text
    assert "缺口 " not in text


@pytest.mark.parametrize("invalid", [None, True, -1, 1.5, "268435456"])
def test_invalid_byte_fields_are_not_coerced_into_precision(invalid):
    operation = capacity_operation()
    operation["capacity_diagnostic"]["nodes"] = [{
        "node_id": "rog", "available_bytes": invalid, "required_bytes": invalid,
        "deficit_bytes": invalid, "reserve_bytes": invalid,
    }]

    text = format_model_intent(operation)

    for label in ("可用 ", "所需 ", "缺口 ", "预留 "):
        assert label not in text
    assert "rog" in text


@pytest.mark.parametrize("layer_range", [None, [], [2], [4, 2], [-1, 2], [0, True], ["0", "4"]])
def test_invalid_or_missing_range_does_not_become_a_claimed_layer_segment(layer_range):
    operation = capacity_operation()
    operation["capacity_diagnostic"]["nodes"][0]["layer_range"] = layer_range

    text = format_model_intent(operation)

    assert "层段 [" not in text
    assert "可用 1.00 GiB" in text


def test_unknown_reason_without_explicit_deficit_keeps_legacy_waiting_label():
    operation = capacity_operation(code="FUTURE_REASON")
    del operation["capacity_diagnostic"]["nodes"][0]["deficit_bytes"]

    text = format_model_intent(operation)

    assert "等待节点 (waiting_nodes)" in text
    assert "FUTURE_REASON" in text
    assert "容量不足（" not in text
    assert "1342177280 bytes" not in text


def test_legacy_data_without_diagnostic_retains_reason_and_node_receipts():
    operation = {
        "status": "waiting_nodes", "intent": {"model_id": "legacy-model"},
        "reason_code": "SUPPLY_WAITING_NODES", "reason": "等待节点加入",
        "nodes": {"worker-old": {"status": "receiving", "bytes_received": 33, "total_bytes": 44}},
    }

    text = format_model_intent(operation)

    assert text.splitlines()[0] == "legacy-model · 等待节点 (waiting_nodes)"
    assert "SUPPLY_WAITING_NODES: 等待节点加入" in text
    assert "worker-old: receiving · 33/44 bytes" in text
    assert "容量不足（" not in text


@pytest.mark.parametrize("scope", [None, "future_scope", []])
def test_missing_or_unknown_scope_is_not_presented_as_a_complete_candidate_plan(scope):
    operation = capacity_operation(scope=scope)

    text = format_model_intent(operation)

    assert "容量不足（范围未提供" in text
    assert "容量不足（候选方案" not in text
    assert "可用 1.00 GiB" in text


def test_legacy_non_string_reason_retains_status_without_interpreting_diagnostic():
    operation = capacity_operation()
    operation["reason_code"] = {}

    text = format_model_intent(operation)

    assert "等待节点 (waiting_nodes)" in text
    assert "容量不足（" not in text


@pytest.mark.parametrize("code, scope, resource, reason", [
    (HOST_CAPACITY_CODE, "minimum_segment", "host_ram", "主机 RAM 不足，无法暂存最小层段"),
    (CAPACITY_CODE, "candidate_plan", "ram", "等待容量到达截止时间，保留本次容量拒绝原因"),
])
def test_failed_capacity_cause_preserves_precise_diagnostic(code, scope, resource, reason):
    operation = capacity_operation(code=code, scope=scope)
    operation.update(status="failed", reason=reason)
    operation["capacity_diagnostic"]["nodes"][0]["resource"] = resource
    before = deepcopy(operation)

    text = format_model_intent(operation)

    assert text.splitlines()[0] == "qwen35-9b · 失败 (failed)"
    assert code in text and reason in text
    assert "容量不足（" in text
    assert "候选层段超出节点预算" in text
    assert "所需 1.25 GiB (1342177280 bytes)" in text
    assert "缺口 256.00 MiB (268435456 bytes)" in text
    assert "建议：释放节点内存或增加可用节点后重新规划" in text
    assert "等待容量" not in text.splitlines()[0]
    assert not model_intent_ready(operation)
    assert operation == before


@pytest.mark.parametrize("reason_code", ["SUPPLY_WORKER_FAILED", "SUPPLY_DEADLINE_EXCEEDED"])
def test_unrelated_failure_does_not_inherit_previous_capacity_or_pending_nodes(reason_code):
    operation = capacity_operation()
    operation.update(status="failed", reason_code=reason_code, reason="当前失败原因已改变")
    operation["capacity_diagnostic"]["pending_node_ids"] = ["joining-worker"]

    text = format_model_intent(operation)

    assert text.splitlines()[0] == "qwen35-9b · 失败 (failed)"
    assert reason_code in text
    assert "容量不足（" not in text
    assert "1342177280 bytes" not in text
    assert "仍在加入的节点" not in text
    assert "joining-worker" not in text


@pytest.mark.parametrize("status, label", [("waiting_nodes", "等待容量"), ("failed", "失败")])
def test_pending_nodes_supplement_capacity_cause_without_replacing_it(status, label):
    operation = capacity_operation()
    operation["status"] = status
    operation["capacity_diagnostic"]["pending_node_ids"] = ["surface-new", "worker-new"]
    before = deepcopy(operation)

    text = format_model_intent(operation)

    assert text.splitlines()[0] == f"qwen35-9b · {label} ({status})"
    assert "仍在加入的节点：surface-new, worker-new" in text
    assert CAPACITY_CODE in text
    assert "当前在线节点无法容纳连续层段" in text
    assert "缺口 256.00 MiB (268435456 bytes)" in text
    assert "rog: online" in text
    assert "SUPPLY_WAITING_NODES" not in text
    assert not model_intent_ready(operation)
    assert operation == before


@pytest.mark.parametrize("pending", [None, [], "worker-new", [None, "", "   ", True]])
def test_empty_or_invalid_pending_nodes_do_not_claim_that_nodes_are_joining(pending):
    operation = capacity_operation()
    operation["capacity_diagnostic"]["pending_node_ids"] = pending

    text = format_model_intent(operation)

    assert "仍在加入的节点" not in text
    assert "容量不足（" in text
    assert "缺口 256.00 MiB (268435456 bytes)" in text


def test_absent_pending_nodes_do_not_claim_that_nodes_are_joining():
    text = format_model_intent(capacity_operation())
    assert "仍在加入的节点" not in text


def test_mixed_legacy_pending_list_shows_only_explicit_node_id_strings():
    operation = capacity_operation()
    operation["capacity_diagnostic"]["pending_node_ids"] = [None, "surface-new", 42, "", "   "]

    text = format_model_intent(operation)

    assert "仍在加入的节点：surface-new\n" in text
    assert "仍在加入的节点：None" not in text
    assert "仍在加入的节点：42" not in text
    assert "缺口 256.00 MiB (268435456 bytes)" in text


@pytest.mark.parametrize("status, label", [("waiting_nodes", "等待容量"), ("failed", "失败")])
def test_connected_nodes_excluded_by_reserve_still_show_owner_candidate_deficits(status, label):
    code = "pipeline_distributed_workers_unavailable"
    operation = capacity_operation(code=code)
    operation.update(status=status, reason="在线节点可用 RAM 未超过预留，被容量求解器排除")
    operation["nodes"] = {"master": {"status": "online"}, "rog": {"status": "online"}}
    operation["capacity_diagnostic"]["nodes"] = [
        {"node_id": "master", "engine": "llama_cpp", "execution_device": "cpu", "resource": "ram",
         "available_bytes": 128 * MIB, "required_bytes": 640 * MIB, "reserve_bytes": 128 * MIB,
         "deficit_bytes": 512 * MIB, "layer_range": [0, 16]},
        {"node_id": "rog", "engine": "llama_cpp", "execution_device": "cpu", "resource": "ram",
         "available_bytes": 64 * MIB, "required_bytes": 640 * MIB, "reserve_bytes": 128 * MIB,
         "deficit_bytes": 576 * MIB, "layer_range": [16, 32]},
    ]
    before = deepcopy(operation)

    text = format_model_intent(operation)

    assert text.splitlines()[0] == f"qwen35-9b · {label} ({status})"
    assert code in text
    assert "容量不足（候选方案" in text
    assert "在线节点可用 RAM 未超过预留，被容量求解器排除" in text
    assert "master: online" in text and "rog: online" in text
    assert "可用 128.00 MiB (134217728 bytes)" in text
    assert "所需 640.00 MiB (671088640 bytes)" in text
    assert "缺口 512.00 MiB (536870912 bytes)" in text
    assert "缺口 576.00 MiB (603979776 bytes)" in text
    assert "层段 [0, 16)" in text and "层段 [16, 32)" in text
    assert "仍在加入的节点" not in text
    assert "SUPPLY_WAITING_NODES" not in text
    assert not model_intent_ready(operation)
    assert operation == before


def test_unknown_code_with_matching_explicit_deficit_does_not_need_whitelist_update():
    operation = capacity_operation(code="FUTURE_REASON")
    text = format_model_intent(operation)
    assert "等待容量 (waiting_nodes)" in text
    assert "FUTURE_REASON" in text
    assert "容量不足（候选方案" in text
    assert "缺口 256.00 MiB (268435456 bytes)" in text


@pytest.mark.parametrize("status", ["waiting_nodes", "failed"])
def test_explicit_joining_reason_never_becomes_capacity_even_with_matching_positive_deficit(status):
    operation = capacity_operation(code="SUPPLY_WAITING_NODES")
    operation.update(status=status, reason="等待从节点完成加入")

    text = format_model_intent(operation)

    label = "等待节点" if status == "waiting_nodes" else "失败"
    assert text.splitlines()[0] == f"qwen35-9b · {label} ({status})"
    assert "SUPPLY_WAITING_NODES: 等待从节点完成加入" in text
    assert "等待容量" not in text
    assert "容量不足（" not in text
    assert "1342177280 bytes" not in text


@pytest.mark.parametrize("deficit", [None, 0, -1, True, 1.5, "268435456"])
def test_ambiguous_code_needs_positive_integer_deficit_without_deriving_one(deficit):
    operation = capacity_operation(code="pipeline_distributed_workers_unavailable")
    operation["capacity_diagnostic"]["nodes"][0]["deficit_bytes"] = deficit
    operation["capacity_diagnostic"]["raw_model_deficit"] = 268435456

    text = format_model_intent(operation)

    assert text.splitlines()[0] == "qwen35-9b · 等待节点 (waiting_nodes)"
    assert "容量不足（" not in text
    assert "1342177280 bytes" not in text
    assert "缺口 " not in text


@pytest.mark.parametrize("nodes", [None, [], [None], [{"node_id": "rog", "required_bytes": None}]])
def test_pure_segment_contract_failure_without_candidate_or_known_requirement_is_not_memory_shortage(nodes):
    operation = capacity_operation(code="pipeline_segment_contract_unsatisfied")
    operation["capacity_diagnostic"].update(nodes=nodes, reason="层段契约不满足，无法形成候选")

    text = format_model_intent(operation)

    assert text.splitlines()[0] == "qwen35-9b · 等待节点 (waiting_nodes)"
    assert "pipeline_segment_contract_unsatisfied" in text
    assert "容量不足（" not in text
    assert "所需 " not in text
    assert "缺口 " not in text


@pytest.mark.parametrize("status, label", [("waiting_nodes", "等待节点"), ("failed", "失败")])
def test_ambiguous_code_does_not_reuse_positive_deficit_from_another_cause(status, label):
    operation = capacity_operation(code="pipeline_distributed_workers_unavailable")
    operation["status"] = status
    operation["capacity_diagnostic"]["reason_code"] = CAPACITY_CODE
    operation["capacity_diagnostic"]["pending_node_ids"] = ["joining-worker"]

    text = format_model_intent(operation)

    assert text.splitlines()[0] == f"qwen35-9b · {label} ({status})"
    assert "容量不足（" not in text
    assert "1342177280 bytes" not in text
    assert "joining-worker" not in text


@pytest.mark.parametrize("status", ["waiting_nodes", "failed"])
def test_identical_capacity_reason_is_displayed_once_on_the_operation(status):
    operation = capacity_operation()
    reason = "当前各节点容量无法满足分布式层段预算：rog 缺口 268435456 字节。"
    operation.update(status=status, reason=reason)
    operation["capacity_diagnostic"]["reason"] = reason
    before = deepcopy(operation)

    text = format_model_intent(operation)

    assert text.count(reason) == 1
    assert f"{CAPACITY_CODE}: {reason}" in text.splitlines()
    assert "容量不足（候选方案" in text
    assert "所需 1.25 GiB (1342177280 bytes)" in text
    assert "缺口 256.00 MiB (268435456 bytes)" in text
    assert operation == before


@pytest.mark.parametrize("diagnostic_reason", ["主机暂存预算还有额外限制", "当前在线节点无法容纳连续层段 "])
def test_distinct_diagnostic_reason_is_preserved_by_exact_equality_only(diagnostic_reason):
    operation = capacity_operation()
    operation["capacity_diagnostic"]["reason"] = diagnostic_reason

    text = format_model_intent(operation)

    assert f"{CAPACITY_CODE}: {operation['reason']}" in text.splitlines()
    assert diagnostic_reason in text.splitlines()
    assert "缺口 256.00 MiB (268435456 bytes)" in text


@pytest.mark.parametrize("reason_fields", [{}, {"reason": None}, {"reason": ""}])
def test_diagnostic_reason_is_preserved_when_operation_reason_is_absent(reason_fields):
    operation = capacity_operation()
    del operation["reason"]
    operation.update(reason_fields)

    text = format_model_intent(operation)

    assert text.count(operation["capacity_diagnostic"]["reason"]) == 1
    assert operation["capacity_diagnostic"]["reason"] in text.splitlines()
    assert "缺口 256.00 MiB (268435456 bytes)" in text


@pytest.mark.parametrize("resource", ["ram", "vram"])
def test_solver_reserve_is_labelled_as_including_boundaries_without_changing_bytes(resource):
    operation = capacity_operation()
    operation["capacity_diagnostic"]["nodes"][0].update(resource=resource, reserve_bytes=3912472986)
    before = deepcopy(operation)

    text = format_model_intent(operation)

    assert " · 含边界的预留 3.64 GiB (3912472986 bytes)" in text
    assert " · 预留 3.64 GiB" not in text
    assert text.count("3912472986 bytes") == 1
    assert operation == before
