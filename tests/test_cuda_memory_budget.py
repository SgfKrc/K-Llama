"""CUDA inventory budgets use the same physical GPU under WDDM and remapping."""
from __future__ import annotations

import json
import subprocess
import sys
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

import cuda_memory_budget as budget

MIB = 1024**2
TARGET = "962bd173-a5f8-4f61-9840-05d364605846"
OTHER = "11111111-2222-3333-4444-555555555555"
SMI_PATH = r"C:\Program Files\NVIDIA Corporation\nvidia-smi.exe"


def fake_torch(free=10 * MIB, device_uuid=TARGET):
    return SimpleNamespace(
        cuda=SimpleNamespace(
            mem_get_info=Mock(return_value=(free, 16 * MIB)),
            get_device_properties=Mock(return_value=SimpleNamespace(uuid=device_uuid)),
        ),
        empty=Mock(return_value=SimpleNamespace(element_size=lambda: 2)),
    )


def install_smi(monkeypatch, output, *, returncode=0, error=None):
    locate = Mock(return_value=SMI_PATH)
    run = Mock(return_value=SimpleNamespace(returncode=returncode, stdout=output,
                                           stderr="driver diagnostics"), side_effect=error)
    monkeypatch.setattr(budget.shutil, "which", locate)
    monkeypatch.setattr(budget.subprocess, "run", run)
    return locate, run


@pytest.mark.parametrize("device,device_uuid,visible,output", [
    ("cuda:0", TARGET, "1,0", f"GPU-{OTHER}, 1\nGPU-{TARGET.upper()}, 7\n"),
    ("cuda:1", "GPU-" + TARGET.upper(), "1,0", f"{TARGET}, 7\nGPU-{OTHER}, 1\n"),
    ("cuda:0", TARGET, "GPU-" + TARGET, f" GPU-{OTHER}, 1\r\n GPU-{TARGET}, 7 \r\n"),
])
def test_uuid_matching_survives_row_order_prefix_case_and_visible_device_remapping(
        monkeypatch, device, device_uuid, visible, output):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", visible)
    torch = fake_torch(device_uuid=device_uuid)
    locate, run = install_smi(monkeypatch, output)
    result = budget.sample_cuda_memory_budget(torch, device)
    assert result["available_bytes"] == 7 * MIB
    assert result["budget_source"] == budget.UUID_MIN_BUDGET_SOURCE
    assert result["budget_observations"] == {
        "torch_free_bytes": 10 * MIB, "device_uuid": TARGET,
        "nvidia_smi_free_bytes": 7 * MIB, "nvidia_smi_status": "matched",
    }
    torch.cuda.mem_get_info.assert_called_once_with(device)
    torch.cuda.get_device_properties.assert_called_once_with(device)
    locate.assert_called_once_with("nvidia-smi")
    args, kwargs = run.call_args
    assert args[0] == [SMI_PATH, "--query-gpu=uuid,memory.free", "--format=csv,noheader,nounits"]
    assert kwargs["shell"] is False and 0 < kwargs["timeout"] <= 2
    assert kwargs["creationflags"] == getattr(subprocess, "CREATE_NO_WINDOW", 0)
    json.dumps(result)


@pytest.mark.parametrize("torch_free,global_mib", [
    (7451181056, 7057),  # Reported WDDM repro: torch misses another process.
    (3 * MIB, 7),       # The global query must never increase a smaller torch sample.
    (0, 7),            # A real zero torch budget stays zero.
    (10 * MIB, 0),     # A real zero global budget stays zero.
])
def test_global_sample_only_reduces_torch_budget_and_zero_is_not_fallback(monkeypatch, torch_free, global_mib):
    install_smi(monkeypatch, f"GPU-{TARGET}, {global_mib}\n")
    result = budget.sample_cuda_memory_budget(fake_torch(torch_free), "cuda:0")
    assert result["available_bytes"] == min(torch_free, global_mib * MIB)
    assert result["budget_source"] == budget.UUID_MIN_BUDGET_SOURCE
    assert result["budget_observations"]["nvidia_smi_free_bytes"] == global_mib * MIB
    assert result["budget_observations"]["nvidia_smi_status"] == "matched"


def test_each_inventory_sample_observes_new_global_pressure_without_caching(monkeypatch):
    _, run = install_smi(monkeypatch, "")
    run.side_effect = [SimpleNamespace(returncode=0, stdout=f"GPU-{TARGET}, {free}\n")
                       for free in (9, 4)]
    torch = fake_torch()
    first = budget.sample_cuda_memory_budget(torch, "cuda:0")
    second = budget.sample_cuda_memory_budget(torch, "cuda:0")
    assert first["available_bytes"] == 9 * MIB
    assert second["available_bytes"] == 4 * MIB
    assert torch.cuda.mem_get_info.call_count == run.call_count == 2


@pytest.mark.parametrize("device_uuid", [None, "", "0", "GPU-0", "MIG-" + TARGET, 0, b"unknown"])
def test_missing_or_unknown_device_identity_never_queries_or_uses_the_first_gpu(monkeypatch, device_uuid):
    locate, run = install_smi(monkeypatch, f"GPU-{OTHER}, 1\n")
    result = budget.sample_cuda_memory_budget(fake_torch(device_uuid=device_uuid), "cuda:0")
    assert result["available_bytes"] == 10 * MIB
    assert result["budget_source"] == budget.TORCH_BUDGET_SOURCE
    observations = result["budget_observations"]
    assert observations["device_uuid"] is observations["nvidia_smi_free_bytes"] is None
    assert observations["nvidia_smi_status"] == ("uuid_missing" if device_uuid is None else "uuid_invalid")
    locate.assert_not_called()
    run.assert_not_called()


@pytest.mark.parametrize("error", [AttributeError("old API"), RuntimeError("CUDA props failed"), OSError("driver")])
def test_device_properties_failure_retains_torch_budget_and_evidence(monkeypatch, error):
    torch = fake_torch()
    torch.cuda.get_device_properties.side_effect = error
    _, run = install_smi(monkeypatch, f"GPU-{OTHER}, 1\n")
    result = budget.sample_cuda_memory_budget(torch, "cuda:0")
    assert result["available_bytes"] == 10 * MIB
    assert result["budget_source"] == budget.TORCH_BUDGET_SOURCE
    assert result["budget_observations"]["nvidia_smi_status"] == "device_properties_failed"
    run.assert_not_called()


def test_missing_nvidia_smi_retains_torch_budget_without_spawning(monkeypatch):
    locate, run = install_smi(monkeypatch, "")
    locate.return_value = None
    result = budget.sample_cuda_memory_budget(fake_torch(), "cuda:0")
    assert result["available_bytes"] == 10 * MIB
    assert result["budget_source"] == budget.TORCH_BUDGET_SOURCE
    assert result["budget_observations"]["nvidia_smi_status"] == "unavailable"
    assert result["budget_observations"]["device_uuid"] == TARGET
    run.assert_not_called()


@pytest.mark.parametrize("error,returncode,status", [
    (FileNotFoundError("missing"), 0, "query_failed"),
    (PermissionError("denied"), 0, "query_failed"),
    (subprocess.TimeoutExpired("nvidia-smi", 2), 0, "query_timeout"),
    (subprocess.SubprocessError("query failed"), 0, "query_failed"),
    (None, 1, "query_failed"),
])
def test_global_query_failures_do_not_replace_or_drop_torch_budget(monkeypatch, error, returncode, status):
    install_smi(monkeypatch, f"GPU-{TARGET}, 0\n", error=error, returncode=returncode)
    result = budget.sample_cuda_memory_budget(fake_torch(), "cuda:0")
    assert result["available_bytes"] == 10 * MIB
    assert result["budget_source"] == budget.TORCH_BUDGET_SOURCE
    assert result["budget_observations"]["nvidia_smi_status"] == status
    assert result["budget_observations"]["nvidia_smi_free_bytes"] is None
    assert "driver diagnostics" not in json.dumps(result)


@pytest.mark.parametrize("output,status", [
    (f"GPU-{OTHER}, 1\n", "uuid_not_found"),
    (f"GPU-{TARGET}, 1\n{TARGET.upper()}, 2\n", "uuid_ambiguous"),
    (f"GPU-{TARGET}, N/A\n", "invalid_free"),
    (f"GPU-{TARGET}, [Not Supported]\n", "invalid_free"),
    (f"GPU-{TARGET}, -1\n", "invalid_free"),
    (f"GPU-{TARGET}, 1.5\n", "invalid_free"),
    (f"GPU-{TARGET}, 7 MiB\n", "invalid_free"),
    (f"GPU-{TARGET}, 1, 2\n", "invalid_output"),
    ('"unterminated', "invalid_output"),
    ("\n", "uuid_not_found"),
])
def test_unmatched_ambiguous_or_invalid_global_output_falls_back_with_reason(monkeypatch, output, status):
    install_smi(monkeypatch, output)
    result = budget.sample_cuda_memory_budget(fake_torch(), "cuda:0")
    assert result["available_bytes"] == 10 * MIB
    assert result["budget_source"] == budget.TORCH_BUDGET_SOURCE
    assert result["budget_observations"]["nvidia_smi_status"] == status
    assert result["budget_observations"]["nvidia_smi_free_bytes"] is None


def test_torch_query_failure_is_not_hidden_by_the_optional_global_query(monkeypatch):
    torch = fake_torch()
    torch.cuda.mem_get_info.side_effect = RuntimeError("torch budget unavailable")
    _, run = install_smi(monkeypatch, f"GPU-{TARGET}, 7\n")
    with pytest.raises(RuntimeError, match="torch budget unavailable"):
        budget.sample_cuda_memory_budget(torch, "cuda:0")
    run.assert_not_called()


@pytest.fixture
def inventory_worker(tmp_path, monkeypatch):
    import config
    import psutil
    import torch_runtime
    from model_supply_worker import ModelSupplyWorker

    monkeypatch.setattr(config, "RUN_MODE", "distributed")
    monkeypatch.setattr(config, "INFERENCE_ENGINE", "llama_cpp")
    monkeypatch.setattr(torch_runtime, "torch_available", lambda: True)
    # model_supply_worker 自 #117 起还要读 virtual.total（主节点 offload 上限按
    # 「物理总量 × 比例」封顶），桩必须同时给出该字段，否则 AttributeError。
    monkeypatch.setattr(
        psutil, "virtual_memory",
        lambda: SimpleNamespace(available=100 * MIB, total=1024 * MIB),
    )
    scheduler = SimpleNamespace(_effective_role=lambda: "master",
                                get_effective_node_id=lambda: "master",
                                get_distributed_inference_enabled=lambda: True)
    worker = ModelSupplyWorker(scheduler, tmp_path / "supply" / "received")
    monkeypatch.setattr(worker, "_source_candidates", lambda: iter(()))

    def configure(device="cuda:0", torch=None):
        torch = fake_torch() if torch is None else torch
        monkeypatch.setattr(torch_runtime, "require_torch", lambda: torch)
        monkeypatch.setitem(sys.modules, "model_module", SimpleNamespace(
            _select_layer_runtime=lambda: (device, "float16")))
        return worker, torch

    return configure


def test_inventory_returns_uuid_bound_global_budget_and_json_safe_evidence(inventory_worker, monkeypatch):
    worker, torch = inventory_worker()
    install_smi(monkeypatch, f"GPU-{OTHER}, 1\nGPU-{TARGET}, 7\n")
    inventory = worker.inventory(model_id="")
    runtime = inventory["pytorch_runtime"]
    assert runtime["available_bytes"] == 7 * MIB
    assert runtime["execution_device"] == "cuda" and runtime["target_element_size"] == 2
    assert runtime["budget_source"] == budget.UUID_MIN_BUDGET_SOURCE
    assert runtime["budget_observations"]["device_uuid"] == TARGET
    assert runtime["materializations"] == ["safetensors_key_filtered", "safetensors_layer_offload"]
    assert inventory["memory_available_bytes"] == 100 * MIB
    torch.cuda.mem_get_info.assert_called_once_with("cuda:0")
    json.dumps(inventory)


def test_cpu_inventory_uses_host_budget_without_any_cuda_or_global_query(inventory_worker, monkeypatch):
    worker, torch = inventory_worker(device="cpu")
    locate, run = install_smi(monkeypatch, f"GPU-{TARGET}, 1\n")
    runtime = worker.inventory(model_id="")["pytorch_runtime"]
    assert runtime["execution_device"] == "cpu"
    assert runtime["available_bytes"] == 100 * MIB
    assert runtime["materializations"] == ["safetensors_key_filtered"]
    torch.cuda.mem_get_info.assert_not_called()
    torch.cuda.get_device_properties.assert_not_called()
    locate.assert_not_called()
    run.assert_not_called()


@pytest.mark.parametrize("torch_free,global_mib", [(0, 7), (10 * MIB, 0)])
def test_zero_cuda_budget_reaches_existing_positive_free_admission_guard(
        inventory_worker, monkeypatch, torch_free, global_mib):
    from model_supply import ModelSupplyError
    from model_supply_capacity import project_inventory_capacity

    worker, _ = inventory_worker(torch=fake_torch(torch_free))
    install_smi(monkeypatch, f"GPU-{TARGET}, {global_mib}\n")
    inventory = worker.inventory(model_id="")
    assert inventory["pytorch_runtime"]["available_bytes"] == 0
    with pytest.raises(ModelSupplyError) as exc:
        project_inventory_capacity({"node_id": "master"}, inventory, "pytorch")
    assert exc.value.code == "SUPPLY_RUNTIME_BUDGET_MISSING"


def test_global_budget_reaches_fresh_solver_with_reserve_and_safety_unchanged(inventory_worker, monkeypatch):
    from model_supply_capacity import project_inventory_capacity
    from pipeline_capacity import solve_pipeline_capacity
    from tests.test_pc_supply_layer_budget import descriptor

    worker, _ = inventory_worker(torch=fake_torch(20 * MIB))
    install_smi(monkeypatch, f"GPU-{TARGET}, 9\n")
    inventory = worker.inventory(model_id="")
    node = project_inventory_capacity({"node_id": "master", "role": "master", "reserve_bytes": MIB},
                                      inventory, "pytorch")
    assert node["capacity_bytes"] == 9 * MIB
    assert node["reserve_bytes"] == MIB and node["runtime_multiplier"] == 1.0
    source = descriptor(sizes=(2 * MIB,) * 4)
    # 8 MiB weights * 1.25 + 1 MiB reserve: torch's 20 MiB admits, global 9 rejects.
    rejected = solve_pipeline_capacity(source, [node], safety_margin=1.25, required_node_id="master")
    admitted = solve_pipeline_capacity(source, [{**node, "capacity_bytes": 20 * MIB}],
                                       safety_margin=1.25, required_node_id="master")
    assert not rejected["admitted"], rejected
    assert admitted["admitted"], admitted
    assert admitted["assignments"][0]["required_bytes"] == 11 * MIB


@pytest.mark.parametrize("rendered", [TARGET, "GPU-" + TARGET.upper()])
def test_torch_cuuuid_object_string_form_is_strictly_matched(monkeypatch, rendered):
    class CUuuidLike:
        def __str__(self):
            return rendered

    install_smi(monkeypatch, f"GPU-{OTHER}, 1\nGPU-{TARGET}, 7\n")
    result = budget.sample_cuda_memory_budget(fake_torch(device_uuid=CUuuidLike()), "cuda:0")
    assert result["available_bytes"] == 7 * MIB
    assert result["budget_source"] == budget.UUID_MIN_BUDGET_SOURCE
    assert result["budget_observations"]["device_uuid"] == TARGET
    assert result["budget_observations"]["nvidia_smi_status"] == "matched"


@pytest.mark.parametrize("error", [TypeError("unreadable"), RuntimeError("unreadable"), KeyError("unreadable")])
def test_unreadable_uuid_object_falls_back_without_global_query(monkeypatch, error):
    class UnreadableUuid:
        def __str__(self):
            raise error

    locate, run = install_smi(monkeypatch, f"GPU-{OTHER}, 1\n")
    result = budget.sample_cuda_memory_budget(fake_torch(device_uuid=UnreadableUuid()), "cuda:0")
    assert result["available_bytes"] == 10 * MIB
    assert result["budget_source"] == budget.TORCH_BUDGET_SOURCE
    assert result["budget_observations"]["nvidia_smi_status"] == "uuid_invalid"
    locate.assert_not_called()
    run.assert_not_called()
