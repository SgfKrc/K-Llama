"""发行 profile 的上报契约（最小发行包 P0-1 的"能力广告"切片）。

覆盖三件事：

1. `detect_runtime_profile()` 的取值规则：只认 `RUNTIME_PROFILES`，其余一律 `unspecified`；
2. src 侧与打包侧（`packaging/packaging/runtime_guard.py`）的 profile 集合一致，防两端漂移；
3. 上报链路：设备画像 → `device_info` → 主节点节点表（`NodeInfo.to_dict`）全程可见。
"""

from __future__ import annotations

import importlib.util
import os
import shutil
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

import device_profiler as dp
from scheduler_types import NodeInfo, NodeState

ROOT = Path(__file__).resolve().parents[1]
RUNTIME_GUARD = ROOT / "packaging" / "packaging" / "runtime_guard.py"


def _load_runtime_guard(path: Path, monkeypatch):
    if not path.is_file():
        raise FileNotFoundError(
            f"Required packaging contract is missing: {path}. "
            "Run git submodule update --init --depth 1 packaging."
        )
    name = "_qlh_packaging_runtime_guard_contract"
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    # dataclasses resolves annotations through sys.modules during exec_module.
    monkeypatch.setitem(sys.modules, name, module)
    spec.loader.exec_module(module)
    return module


def test_detect_runtime_profile_accepts_each_declared_value(monkeypatch):
    for profile in dp.RUNTIME_PROFILES:
        monkeypatch.setenv("QLH_RUNTIME_PROFILE", profile)
        assert dp.detect_runtime_profile() == profile


def test_detect_runtime_profile_normalizes_case_and_spaces(monkeypatch):
    monkeypatch.setenv("QLH_RUNTIME_PROFILE", "  LLAMA_CPP_ONLY  ")
    assert dp.detect_runtime_profile() == "llama_cpp_only"


def test_detect_runtime_profile_falls_back_to_unspecified(monkeypatch):
    monkeypatch.delenv("QLH_RUNTIME_PROFILE", raising=False)
    assert dp.detect_runtime_profile() == dp.RUNTIME_PROFILE_UNSPECIFIED

    monkeypatch.setenv("QLH_RUNTIME_PROFILE", "gpu_only")
    assert dp.detect_runtime_profile() == dp.RUNTIME_PROFILE_UNSPECIFIED


def test_runtime_profile_set_matches_packaging_contract(monkeypatch):
    """src 与打包侧的 profile 集合是同一份合同，任一侧新增必须同时改另一侧。"""
    runtime_guard = _load_runtime_guard(RUNTIME_GUARD, monkeypatch)

    assert Path(runtime_guard.__file__).resolve() == RUNTIME_GUARD.resolve()
    assert dp.RUNTIME_PROFILES == runtime_guard.RUNTIME_PROFILES


def test_packaging_contract_requires_the_checked_out_source(tmp_path, monkeypatch):
    # An unrelated installed module must never make a missing submodule pass.
    monkeypatch.setitem(sys.modules, "runtime_guard", SimpleNamespace(
        RUNTIME_PROFILES=dp.RUNTIME_PROFILES,
    ))
    with pytest.raises(FileNotFoundError, match="git submodule update --init"):
        _load_runtime_guard(tmp_path / "missing" / "runtime_guard.py", monkeypatch)


def test_packaging_contract_does_not_reuse_an_installed_module(monkeypatch):
    monkeypatch.setitem(sys.modules, "runtime_guard", SimpleNamespace(
        RUNTIME_PROFILES=("unrelated_installed_profile",),
    ))
    guard = _load_runtime_guard(RUNTIME_GUARD, monkeypatch)
    assert guard.RUNTIME_PROFILES == dp.RUNTIME_PROFILES


def test_device_profile_dict_carries_runtime_profile(monkeypatch):
    monkeypatch.setenv("QLH_RUNTIME_PROFILE", "llama_cpp_only")
    payload = dp.DeviceProfiler.mock_mobile().to_dict()

    assert payload["runtime_profile"] == "llama_cpp_only"


def test_peer_report_carries_runtime_profile(monkeypatch):
    """从节点上报设备画像时把 profile 写进 device_info 与客户端对象。"""
    monkeypatch.setenv("QLH_RUNTIME_PROFILE", "llama_cpp_only")
    from inference_service.peer import PeerClient

    peer = PeerClient.__new__(PeerClient)
    peer._client = SimpleNamespace(device_info={})
    peer._report_device_profile()

    assert peer._device_info["runtime_profile"] == "llama_cpp_only"
    assert peer._client.device_info["runtime_profile"] == "llama_cpp_only"


def test_master_node_table_exposes_runtime_profile():
    """主节点 `/api/cluster/nodes` 走的 `NodeInfo.to_dict` 必须透出该字段。"""
    info = NodeInfo(
        node_id="edge-1",
        role="client",
        node_type="pc",
        state=NodeState.ONLINE,
        device_info={"runtime_profile": "llama_cpp_only"},
    )

    assert info.to_dict()["device_info"]["runtime_profile"] == "llama_cpp_only"


def test_pc_task_worker_hello_exposes_runtime_profile(monkeypatch):
    monkeypatch.setenv("QLH_RUNTIME_PROFILE", "llama_cpp_only")
    from scheduler import Scheduler

    scheduler = Scheduler()
    capabilities = scheduler._task_worker_capabilities()

    assert capabilities["runtime_profile"] == "llama_cpp_only"


@pytest.mark.parametrize(
    ("include_tools", "missing_core", "expected_code"),
    [(False, False, 0), (True, False, 2), (False, True, 2)],
    ids=["core-stays-collected", "tool-opt-in-keeps-import-errors", "core-import-errors-stay-fatal"],
)
def test_workspace_tool_collection_ownership(
    tmp_path, include_tools, missing_core, expected_code,
):
    shutil.copyfile(ROOT / "tests" / "conftest.py", tmp_path / "conftest.py")
    (tmp_path / "pytest.ini").write_text("[pytest]\n", encoding="utf-8")
    (tmp_path / "test_core.py").write_text(
        "def test_core_contract(): assert True\n", encoding="utf-8",
    )
    for name in (
        "test_docagent_gate.py", "test_doc_maintenance_audit.py",
        "test_model_fleet_explicit_layer_pager.py", "test_model_fleet_memory_transport.py",
        "test_model_fleet_n2f_fixed_tile.py", "test_model_fleet_n2r_artifact_gate.py",
        "test_model_fleet_n2r_compare.py", "test_offline_bundle.py",
    ):
        (tmp_path / name).write_text(
            "raise ImportError('external tool source required')\n", encoding="utf-8",
        )
    core_contracts = (
        "test_model_fleet_core_dependency.py", "test_experiment_gemma_quality_unit.py",
        "test_experiment_production_quality.py", "test_llama_dependency_contract.py",
    )
    if missing_core:
        for name in core_contracts:
            (tmp_path / name).write_text(
                "raise ImportError('core dependency required')\n", encoding="utf-8",
            )
    env = os.environ.copy()
    env.pop("QLH_INCLUDE_WORKSPACE_TOOL_TESTS", None)
    if include_tools:
        env["QLH_INCLUDE_WORKSPACE_TOOL_TESTS"] = "1"
    env["PYTEST_DISABLE_PLUGIN_AUTOLOAD"] = "1"
    result = subprocess.run(
        [sys.executable, "-B", "-m", "pytest", str(tmp_path), "--collect-only",
         "-q", "-c", str(tmp_path / "pytest.ini"), "--confcutdir", str(tmp_path)],
        cwd=tmp_path, env=env, capture_output=True, text=True,
        encoding="utf-8", errors="replace", timeout=90,
    )
    output = result.stdout + result.stderr
    assert result.returncode == expected_code, output
    assert "test_core.py::test_core_contract" in output, output
    if include_tools:
        assert "external tool source required" in output
    elif missing_core:
        assert "core dependency required" in output
        assert all(name in output for name in core_contracts)
    else:
        assert "1 test collected" in output
        assert "skipped" not in output
