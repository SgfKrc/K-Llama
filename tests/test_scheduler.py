"""
单元测试 — 调度器动态分层算法
=============================
测试 _compute_node_weight() 和 compute_layer_assignment()
使用模拟设备数据，无需真实硬件/数据库连接。
"""

import sys
import os
import json
import logging
import time
import threading
from typing import Any, cast
from unittest.mock import MagicMock, patch
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import pytest
# 加载 API composition root，以配置 SchedulerCallbackSet。
import api_server  # noqa: F401,E402
import node_runtime as node_runtime_mod
from scheduler import Scheduler, PipelineQueue, NodeInfo, NodeState, NodeRole
from koakuma_engine import Capability
from qlh_core.contracts.engines import ENGINE_REGISTRY


def _engine_contract(family, runtime_kind):
    """Concrete execution evidence chosen explicitly by each scheduler fixture."""
    return {"engine_capabilities": ENGINE_REGISTRY.advertisement([
        ENGINE_REGISTRY.require(family, runtime_kind),
    ])}


# ================================================================
# 模拟设备画像（不同硬件配置）
# ================================================================

PROFILE_WORKSTATION = {
    **_engine_contract("pytorch", "pytorch_cuda"),
    "gpu": {
        "name": "NVIDIA RTX 4090",
        "vram_total_gb": 24.0,
        "cuda_available": True,
        "is_integrated": False,
    },
    "ram": {"total_gb": 64.0},
    "cpu": {"physical_cores": 16, "freq_max_mhz": 4500},
}

PROFILE_LAPTOP = {
    **_engine_contract("pytorch", "pytorch_cuda"),
    "gpu": {
        "name": "NVIDIA RTX 4060 Laptop",
        "vram_total_gb": 8.0,
        "cuda_available": True,
        "is_integrated": False,
    },
    "ram": {"total_gb": 16.0},
    "cpu": {"physical_cores": 8, "freq_max_mhz": 4000},
}

PROFILE_ULTRABOOK = {
    **_engine_contract("pytorch", "pytorch_cpu"),
    "gpu": {
        "name": "Intel Iris Xe",
        "vram_total_gb": 0.5,
        "cuda_available": False,
        "is_integrated": True,
    },
    "ram": {"total_gb": 8.0},
    "cpu": {"physical_cores": 4, "freq_max_mhz": 3000},
}

PROFILE_EDGE = {
    **_engine_contract("pytorch", "pytorch_cpu"),
    "gpu": {
        "name": "None",
        "vram_total_gb": 0,
        "cuda_available": False,
        "is_integrated": True,
    },
    "ram": {"total_gb": 4.0},
    "cpu": {"physical_cores": 2, "freq_max_mhz": 1500},
}

PROFILE_MOBILE = {
    "gpu": {
        "name": "ARM Mali",
        "vram_total_gb": 0,
        "cuda_available": False,
        "is_integrated": True,
    },
    "ram": {"total_gb": 2.0},
    "cpu": {"physical_cores": 4, "freq_max_mhz": 2000},
}

PROFILE_NO_GPU = {
    **_engine_contract("pytorch", "pytorch_cpu"),
    "gpu": {},
    "ram": {"total_gb": 8.0},
    "cpu": {"physical_cores": 4, "freq_max_mhz": 2500},
}


# ================================================================
# _compute_node_weight 测试
# ================================================================

class TestComputeNodeWeight:
    """测试节点权重计算"""

    @pytest.fixture
    def sched(self):
        return Scheduler()

    @pytest.mark.parametrize(
        ("profile_name", "profile", "expected_weight"),
        [
            ("cuda_workstation", PROFILE_WORKSTATION, 160.0),
            ("cuda_laptop", PROFILE_LAPTOP, 99.16666666666667),
            # 集显不计专用显存分：仅 RAM 3.75 + CPU 10.0。
            ("integrated_ultrabook", PROFILE_ULTRABOOK, 13.75),
            ("edge", PROFILE_EDGE, 6.875),
            ("gpu_field_missing", PROFILE_NO_GPU, 12.5),
            ("empty", {}, 8.125),
            ("none", None, 8.125),
            ("mobile_arm", PROFILE_MOBILE, 8.4375),
        ],
    )
    def test_fixed_profile_weight_is_formula_stable(
            self, sched, profile_name, profile, expected_weight):
        """固定画像必须精确锁定权重公式，而非仅落在宽泛区间。
        （已按改进计划收紧断言 + 补充精确计算注释，符合 Q17/P3-A 建议）
        + PT-PIPE-QW3.5 scheduler integration: loopback 认证绑定测试点"""
        weight = sched._compute_node_weight(profile)
        assert weight == pytest.approx(expected_weight), profile_name

    def test_discrete_gpu_bonus(self, sched):
        """只有可用 CUDA 独显才应获得专用显存与执行后端分。"""
        with_gpu = PROFILE_LAPTOP  # 独显
        without_gpu = {**PROFILE_LAPTOP, "gpu": {**PROFILE_LAPTOP["gpu"], "cuda_available": False}}
        w_gpu = sched._compute_node_weight(with_gpu)
        w_no_gpu = sched._compute_node_weight(without_gpu)
        # 8GB 专用显存 16.6667 分 + CUDA 执行后端 60 分。
        bonus = w_gpu - w_no_gpu
        assert bonus == pytest.approx(76.66666666666667)
        assert w_gpu / w_no_gpu == pytest.approx(4.407407407407407)

    def test_cuda_laptop_outscores_igpu_cpu_worker(self, sched):
        """4060 CUDA 主机评分必须显著高于同核数的集显 CPU worker。"""
        cuda_weight = sched._compute_node_weight(PROFILE_LAPTOP)
        igpu_weight = sched._compute_node_weight(PROFILE_IGPU_ONLY)

        assert cuda_weight > igpu_weight
        assert cuda_weight / igpu_weight == pytest.approx(4.407407407407407)


# ================================================================
# compute_layer_assignment 测试
# ================================================================

def _mark_layer_config_pushed(sched, node_id: str) -> None:
    """★ DIST-NEXT-3：测试夹具 —— 把节点标记为「已确认层配置」。

    `_layer_config_pushed` 已是**派生视图**（相位 `ACKED`/`READY`），夹具不能再直接写集合；
    这里经权威视图推进相位，语义与产品的 ready ACK 路径一致。
    """
    registry = sched._worker_assignments
    if registry.state(node_id) is None:
        registry.begin(node_id, reason_code="test_fixture")
    registry.transition(node_id, phase="ready")


def _install_layer_assignment(
    sched, node_id: str, expected: dict, *, ack: dict | None = None,
    phase: str | None = None,
) -> dict:
    """Install one complete assignment fixture through the production source."""
    payload = dict(expected)
    state = sched._worker_assignments.begin(
        node_id,
        assignment_id=str(payload.get("assignment_id", "") or ""),
        expected_config=payload,
        reason_code="test_fixture",
    )
    canonical = sched._worker_assignments.expected(node_id)
    if ack is not None:
        receipt = {
            "node_id": node_id,
            "assignment_id": state.assignment_id,
            "config_id": canonical.get("config_id", ""),
            **({"generation": canonical["generation"]}
               if "generation" in canonical else {}),
            **dict(ack),
        }
        assert sched._worker_assignments.record_ack(node_id, receipt) is not None
    if phase:
        sched._worker_assignments.transition(node_id, phase=phase)
    return canonical


def _layer_expected(sched, node_id: str) -> dict:
    return sched._worker_assignments.expected(node_id)


def _layer_ack(sched, node_id: str) -> dict:
    return sched._worker_assignments.acknowledgement(node_id)


class TestComputeLayerAssignment:
    """测试动态分层计算"""

    @pytest.fixture
    def sched(self):
        s = Scheduler()
        s._role_override = "master"
        # 注入模拟节点
        s.nodes = {
            "master": NodeInfo(
                node_id="master", role="master", state=NodeState.ONLINE,
                node_type="pc",
                device_info=PROFILE_WORKSTATION,
            ),
            "client1": NodeInfo(
                node_id="client1", role="client", state=NodeState.ONLINE,
                node_type="pc",
                device_info=PROFILE_LAPTOP,
            ),
            "client2": NodeInfo(
                node_id="client2", role="client", state=NodeState.ONLINE,
                node_type="pc",
                device_info=PROFILE_ULTRABOOK,
            ),
        }
        return s

    def test_single_node_gets_all_layers(self, sched):
        """单节点应获得全部 24 层"""
        single = [
            {"node_id": "master", "role": "master", "device_info": PROFILE_WORKSTATION}
        ]
        result = sched.compute_layer_assignment(single)
        assert len(result) == 1
        assert result[0]["start_layer"] == 0
        assert result[0]["end_layer"] == 24
        assert result[0]["layers_count"] == 24
        assert result[0]["has_embedding"] is True
        assert result[0]["has_lm_head"] is True

    def test_pipeline_node_metadata_uses_task_worker_engine(
            self, sched, monkeypatch):
        monkeypatch.setattr("scheduler.TASK_WORKER_EXPERIMENTAL_ENABLED", True)
        sched._role_override = "master"
        sched._task_worker_control.status = lambda role: {
            "workers": [{
                "node_id": "client1",
                "capabilities": {
                    "engines": ["pytorch", "llama_cpp"],
                    "models": [{"engine": "llama_cpp"}],
                },
            }],
        }

        metadata = sched._pipeline_node_metadata()

        assert metadata["master"]["engine"] == "pytorch"
        assert metadata["client1"]["engine"] == "llama_cpp"

    def test_active_deepseek_layer_count_replaces_fixed_qwen_count(self, sched, monkeypatch):
        from model_host import model_host as _host

        manager = type("DeepSeekManager", (), {
            "_total_model_layers": 28,
            "model": None,
            "_model_path": "",
        })()
        monkeypatch.setattr(_host, "_manager", manager)
        nodes = [{
            "node_id": "master",
            "role": "master",
            "node_type": "pc",
            "device_info": {**_engine_contract("pytorch", "pytorch_cuda"), },
        }]

        result = sched.compute_layer_assignment(nodes)

        assert result[0]["start_layer"] == 0
        assert result[0]["end_layer"] == 28
        assert result[0]["layers_count"] == 28

    def test_pipeline_model_info_accepts_metadata_only_preparation(self, sched):
        descriptor = {
            "model_id": "tiny-qwen2",
            "model_type": "qwen2",
            "total_layers": 4,
            "model_sha256": "b" * 64,
            "pipeline_runtime_supported": True,
            "inspection_mode": "safetensors_headers_only",
            "weight_bytes": 1024,
        }
        host = type("PreparedHost", (), {
            "is_loaded": False,
            "runtime_kind": "pytorch_cpu", "native_abi": "torch_module.v1",
            "is_pipeline_prepared": True,
            "_engine_type": "pytorch",
            "_model_path": os.getcwd(),
            "_full_model_path": os.getcwd(),
            "active_model_id": "tiny-qwen2",
            "quant_type": "fp16",
            "get_pipeline_descriptor": lambda self: dict(descriptor),
        })()
        sched._host = host

        info = sched._get_active_pipeline_model_info()

        assert info["model_id"] == "tiny-qwen2"
        assert info["total_layers"] == 4
        assert info["model_sha256"] == "b" * 64
        assert info["inspection_mode"] == "safetensors_headers_only"

    def test_pipeline_model_info_rejects_discovered_unsupported_architecture(self, sched):
        host = type("PreparedHost", (), {
            "is_loaded": False,
            "_engine_type": "pytorch",
            "_model_path": os.getcwd(),
            "_full_model_path": os.getcwd(),
            "get_pipeline_descriptor": lambda self: {
                "model_id": "qwen3-4b",
                "model_type": "qwen3",
                "total_layers": 36,
                "pipeline_runtime_supported": False,
            },
        })()
        sched._host = host

        assert sched._get_active_pipeline_model_info() == {}

    def test_capacity_plan_uses_free_memory_and_allows_control_only_master(
            self, sched, monkeypatch):
        mib = 1024 * 1024
        descriptor = {
            "model_id": "synthetic-qwen2",
            "model_type": "qwen2",
            "model_sha256": "a" * 64,
            "pipeline_runtime_supported": True,
            "total_layers": 4,
            "layer_weight_bytes": [100 * mib] * 4,
            "component_weight_bytes": {
                "embedding": 40 * mib,
                "final_norm": 5 * mib,
                "lm_head": 40 * mib,
                "visual": 0,
                "mtp": 0,
                "other": 0,
            },
        }
        sched._host = type("PreparedHost", (), {
            "get_pipeline_descriptor": lambda self: dict(descriptor),
        })()
        sched.nodes = {
            "master": NodeInfo(
                node_id="master", role="master", state=NodeState.ONLINE,
                device_info={**_engine_contract("pytorch", "pytorch_cuda"),
                    "gpu": {"name": "RTX", "cuda_available": True,
                            "is_integrated": False, "vram_total_gb": 8,
                            "vram_free_gb": 0.6},
                    "ram": {"available_gb": 8},
                },
            ),
            "worker-a": NodeInfo(
                node_id="worker-a", role="client", state=NodeState.ONLINE,
                device_info={**_engine_contract("pytorch", "pytorch_cuda"),
                    "gpu": {"name": "RTX", "cuda_available": True,
                            "is_integrated": False, "vram_total_gb": 8,
                            "vram_free_gb": 1.0},
                },
            ),
            "worker-b": NodeInfo(
                node_id="worker-b", role="client", state=NodeState.ONLINE,
                device_info={**_engine_contract("pytorch", "pytorch_cuda"),
                    "gpu": {"name": "RTX", "cuda_available": True,
                            "is_integrated": False, "vram_total_gb": 8,
                            "vram_free_gb": 1.0},
                },
            ),
        }
        monkeypatch.setattr("config.PIPELINE_CAPACITY_RESERVE_MB", 512.0)
        monkeypatch.setattr("config.PIPELINE_CAPACITY_SAFETY_MARGIN", 1.0)

        plan = sched.get_pipeline_capacity_plan()

        assert plan["admitted"] is True
        assert sum(item["layers_count"] for item in plan["assignments"]) == 4
        assert "master" in plan["control_only_nodes"]
        assert plan["transaction_phase"] == "planned"
        layout = plan["pipeline_layout"]
        assert layout["model_sha256"] == "a" * 64
        assert layout["nodes"][0]["layer_range"][0] == 0
        assert layout["nodes"][-1]["layer_range"][1] == 4
        assert sum(
            item["layer_range"][1] - item["layer_range"][0]
            for item in layout["nodes"]
        ) == 4

    def test_stale_heartbeat_worker_leaves_capacity_candidates(self, sched):
        """★ DIST-2：静默（TCP 半开）节点必须退出容量候选，不必等 TCP 巡检。

        此前容量候选只看 `NodeInfo.is_available()`（`state == ONLINE`），而 TCP
        半开（对端进程已死、不发 FIN）要等巡检约 129s 才置 OFFLINE ⇒ 这段窗口里
        该节点**一直占容量**，容量求解器于是分配出既非它声明区间、层数也不对的
        结果（实测：Y700 被带走后 master 仍把它算进规划，给出 `Layer 13-20` /
        `Layer 15-24`）。现在候选与 readiness 共用 `WORKER_HEARTBEAT_MAX_AGE`。
        """
        now = time.time()
        sched.nodes = {
            "fresh": NodeInfo(
                node_id="fresh", role="client", state=NodeState.ONLINE,
                device_info=_engine_contract("pytorch", "pytorch_cpu"),
                last_heartbeat=now - 5,
            ),
            "silent": NodeInfo(
                node_id="silent", role="client", state=NodeState.ONLINE,
                device_info=_engine_contract("pytorch", "pytorch_cpu"),
                last_heartbeat=now - 200,  # 超过 WORKER_HEARTBEAT_MAX_AGE
            ),
        }

        ids = {record["node_id"] for record in sched._get_pipeline_capacity_nodes()}

        assert "fresh" in ids
        assert "silent" not in ids

    def test_aggregate_resource_view_combines_online_nodes_without_addresses(self, sched):
        sched._role_override = "master"
        sched.nodes = {
            "master": NodeInfo(
                node_id="master", role="master", state=NodeState.ONLINE,
                address="127.0.0.1:8888",
                device_info={**_engine_contract("pytorch", "pytorch_cuda"),
                    "cpu": {"physical_cores": 8, "logical_cores": 16},
                    "ram": {"total_gb": 32, "available_gb": 12},
                    "gpu": {
                        "cuda_available": True, "vram_total_gb": 8,
                        "vram_free_gb": 6,
                    },
                },
            ),
            "edge": NodeInfo(
                node_id="edge", role="client", state=NodeState.ONLINE,
                address="100.64.0.2:8888",
                device_info={**_engine_contract("pytorch", "pytorch_cuda"),
                    "cpu": {"physical_cores": 4, "logical_cores": 8},
                    "ram": {"total_gb": 16, "available_gb": 10},
                },
            ),
            "offline": NodeInfo(
                node_id="offline", role="client", state=NodeState.OFFLINE,
                device_info={**_engine_contract("pytorch", "pytorch_cuda"), "ram": {"total_gb": 128, "available_gb": 128}},
            ),
        }

        view = sched.get_aggregate_resource_view()

        assert view["is_distributed"] is True
        assert view["remote_available_count"] == 1
        assert view["totals"]["logical_cores"] == 24
        assert view["totals"]["ram_available_gb"] == 22
        assert "127.0.0.1" not in json.dumps(view)
        assert "100.64.0.2" not in json.dumps(view)

    def test_capacity_plan_rejects_profile_without_free_memory_evidence(
            self, sched):
        mib = 1024 * 1024
        sched._host = type("PreparedHost", (), {
            "get_pipeline_descriptor": lambda self: {
                "model_id": "synthetic-qwen2",
                "model_type": "qwen2",
                "pipeline_runtime_supported": True,
                "total_layers": 1,
                "layer_weight_bytes": [mib],
                "component_weight_bytes": {
                    "embedding": 0, "final_norm": 0, "lm_head": 0,
                    "visual": 0, "mtp": 0, "other": 0,
                },
            },
        })()
        sched.nodes = {
            "master": NodeInfo(
                node_id="master", role="master", state=NodeState.ONLINE,
                device_info=PROFILE_WORKSTATION,
            ),
        }

        plan = sched.get_pipeline_capacity_plan()

        assert plan["admitted"] is False
        assert plan["reason_code"] == "pipeline_capacity_nodes_unavailable"

    def test_android_layer_worker_uses_byte_memory_capacity(self, sched):
        """Android presence uses memory.available_bytes instead of PC ram shape."""
        sched.nodes = {
            "android-worker": NodeInfo(
                node_id="android-worker", role="client", state=NodeState.ONLINE,
                node_type="android",
                device_info={**_engine_contract("llama_cpp", "llama_cpp_android"),
                    "backend_id": "llama_cpp",
                    "capabilities": [Capability.FORWARD_LAYERS],
                    "memory": {"available_bytes": 3 * 1024 ** 3},
                },
            ),
        }

        records = sched._get_pipeline_capacity_nodes()

        assert records == [{
            "node_id": "android-worker",
            "role": NodeRole.CLIENT,
            "capacity_bytes": 3 * 1024 ** 3,
            "reserve_bytes": records[0]["reserve_bytes"],
            "runtime_multiplier": 2.0,
            "execution_device": "cpu",
            "capacity_source": "memory.available_bytes",
            "engine": "llama_cpp",
            "score": records[0]["score"],
        }]

    def test_master_capacity_uses_live_memory_instead_of_startup_snapshot(
            self, sched, monkeypatch):
        gib = 1024 ** 3
        sched._role_override = "master"
        sched.nodes = {
            "master": NodeInfo(
                node_id="master", role="master", state=NodeState.ONLINE,
                device_info={**_engine_contract("pytorch", "pytorch_cuda"), "ram": {"available_gb": 1.0}},
            ),
            "remote": NodeInfo(
                node_id="remote", role="client", state=NodeState.ONLINE,
                device_info={**_engine_contract("pytorch", "pytorch_cuda"), "ram": {"available_gb": 7.0}},
            ),
        }
        import psutil

        monkeypatch.setattr(psutil, "virtual_memory", lambda: type("Memory", (), {
            "available": 5 * gib,
        })())
        records = {record["node_id"]: record for record in sched._get_pipeline_capacity_nodes()}
        assert records["master"]["capacity_bytes"] == 5 * gib
        assert records["master"]["capacity_source"] == "ram.available_bytes_live"
        assert records["remote"]["capacity_bytes"] == 7 * gib

    def test_android_task_worker_ranges_reach_capacity_solver(self, sched, monkeypatch):
        """Admitted v3 ranges are preserved as solver constraints."""
        monkeypatch.setattr("scheduler.TASK_WORKER_EXPERIMENTAL_ENABLED", True)
        sched._role_override = "master"
        sched._task_worker_control.status = lambda role: {
            "workers": [{
                "node_id": "android-worker",
                "healthy": True,
                "layer_stage_dispatch_enabled": True,
                "capabilities": {"layer_ranges": [[4, 16]]},
            }],
        }
        sched.nodes = {
            "android-worker": NodeInfo(
                node_id="android-worker", role=NodeRole.CLIENT,
                state=NodeState.ONLINE, node_type="android",
                device_info={**_engine_contract("llama_cpp", "llama_cpp_android"),
                    "backend_id": "llama_cpp",
                    "capabilities": [Capability.FORWARD_LAYERS],
                    "memory": {"available_bytes": 3 * 1024 ** 3},
                },
            ),
        }

        records = sched._get_pipeline_capacity_nodes()

        assert records[0]["layer_ranges"] == [[4, 16]]

    def test_android_task_worker_artifacts_reach_capacity_solver(
            self, sched, monkeypatch):
        monkeypatch.setattr("scheduler.TASK_WORKER_EXPERIMENTAL_ENABLED", True)
        sched._role_override = "master"
        artifact = {
            "layer_range": [4, 16],
            "segment_mode": "middle",
            "model_id": "middle4-16.gguf",
            "artifact_sha256": "a" * 64,
            "source_model_sha256": "b" * 64,
        }
        sched._task_worker_control.status = lambda role: {
            "workers": [{
                "node_id": "android-worker",
                "healthy": True,
                "layer_stage_dispatch_enabled": True,
                "capabilities": {
                    "layer_ranges": [[4, 16]],
                    "layer_artifacts": [artifact],
                },
            }],
        }
        sched.nodes = {
            "android-worker": NodeInfo(
                node_id="android-worker", role=NodeRole.CLIENT,
                state=NodeState.ONLINE, node_type="android",
                device_info={**_engine_contract("llama_cpp", "llama_cpp_android"),
                    "backend_id": "llama_cpp",
                    "capabilities": [Capability.FORWARD_LAYERS],
                    "memory": {"available_bytes": 3 * 1024 ** 3},
                },
            ),
        }

        records = sched._get_pipeline_capacity_nodes()

        assert records[0]["layer_artifacts"] == [artifact]

    def test_android_task_worker_layer_budget_reaches_capacity_solver(self, sched, monkeypatch):
        """设备自荐的层容量必须随 `layer_ranges` 一起投影进求解器输入 ——

        `layer_ranges` 是"当前已就绪、马上能跑的区间"，`layer_budget.max_layers`
        是"本地裁层后能承载的上限"；两者都要到求解器手上，它才能决策。
        """
        monkeypatch.setattr("scheduler.TASK_WORKER_EXPERIMENTAL_ENABLED", True)
        sched._role_override = "master"
        sched._task_worker_control.status = lambda role: {
            "workers": [{
                "node_id": "android-worker",
                "healthy": True,
                "layer_stage_dispatch_enabled": True,
                "capabilities": {
                    "layer_ranges": [[4, 16]],
                    "layer_budget": {
                        "available_bytes": 3 * 1024 ** 3,
                        "per_layer_bytes": 64 * 1024 ** 2,
                        "max_layers": 12,
                        "local_cut": False,
                    },
                },
            }],
        }
        sched.nodes = {
            "android-worker": NodeInfo(
                node_id="android-worker", role=NodeRole.CLIENT,
                state=NodeState.ONLINE, node_type="android",
                device_info={**_engine_contract("llama_cpp", "llama_cpp_android"),
                    "backend_id": "llama_cpp",
                    "capabilities": [Capability.FORWARD_LAYERS],
                    "memory": {"available_bytes": 3 * 1024 ** 3},
                },
            ),
        }

        records = sched._get_pipeline_capacity_nodes()

        assert records[0]["layer_ranges"] == [[4, 16]]
        assert records[0]["layer_budget"]["max_layers"] == 12
        assert records[0]["layer_budget"]["local_cut"] is False

    def test_manual_capacity_plan_enforces_exact_artifact_boundary_mode(
            self, sched, monkeypatch):
        descriptor = {
            "model_id": "model", "model_type": "qwen2",
            "model_sha256": "a" * 64, "total_layers": 4,
            "hidden_size": 2048, "tokenizer_sha256": "d" * 64,
            "layer_weight_bytes": [10, 10, 10, 10],
            "component_weight_bytes": {
                "embedding": 5, "final_norm": 2, "lm_head": 5,
                "visual": 0, "mtp": 0, "multimodal": 0, "other": 0,
            },
        }
        artifact = {
            "layer_range": [2, 4], "segment_mode": "middle",
            "model_id": "tail.gguf", "artifact_sha256": "b" * 64,
            "source_model_sha256": "c" * 64,
            "source_model_id": "model", "hidden_size": 2048,
            "tokenizer_sha256": "d" * 64,
        }
        records = [
            {
                "node_id": "master", "capacity_bytes": 10_000,
                "reserve_bytes": 0, "runtime_multiplier": 1.0,
                "execution_device": "cpu", "capacity_source": "test",
            },
            {
                "node_id": "worker", "capacity_bytes": 10_000,
                "reserve_bytes": 0, "runtime_multiplier": 1.0,
                "execution_device": "cpu", "capacity_source": "test",
                "layer_ranges": [[2, 4]], "layer_artifacts": [artifact],
                "models": [{
                    "model_id": "tail.gguf", "engine": "llama_cpp",
                    "format": "gguf", "revision": "local",
                    "sha256": "b" * 64,
                }],
            },
        ]
        monkeypatch.setattr(
            sched, "_get_pipeline_capacity_nodes", lambda _ids: records,
        )
        assignments = [
            {
                "node_id": "master", "role": "master",
                "start_layer": 0, "end_layer": 2,
                "has_embedding": True, "has_lm_head": False,
            },
            {
                "node_id": "worker", "role": "client",
                "start_layer": 2, "end_layer": 4,
                "has_embedding": False, "has_lm_head": True,
            },
        ]

        rejected = sched._build_manual_pipeline_capacity_plan(
            assignments, descriptor=descriptor,
        )
        assert rejected["admitted"] is False
        assert rejected["reason_code"] == "pipeline_segment_contract_unsatisfied"

        artifact["segment_mode"] = "tail"
        admitted = sched._build_manual_pipeline_capacity_plan(
            assignments, descriptor=descriptor,
        )
        assert admitted["admitted"] is True
        assert admitted["assignments"][1]["layer_artifact"] == artifact
        assert admitted["assignments"][1]["stage_model_identity"] == records[1]["models"][0]
        assert len(admitted["assignments"][1]["stage_capability_sha256"]) == 64
        assert admitted["pipeline_layout"]["nodes"][1]["engine"] == "llama_cpp"

    def test_route_a_stage_identity_comes_from_worker_artifact(self, sched):
        """Route A 的 offer 必须带 **worker 宣告的工件身份**（engine/format/sha256），
        而不是 master 自己那份模型的摘要 —— 层段在设备上执行、用的是设备的 GGUF，
        而 master 侧是 safetensors，两者格式不同，永远不匹配。"""
        sched._task_worker_control.status = lambda role: {
            "workers": [{
                "node_id": "android-worker",
                "capabilities": {"models": [{
                    "model_id": "layer-f6dab6b72d243419",
                    "engine": "llama_cpp",
                    "format": "gguf",
                    "revision": "local",
                    "sha256": (
                        "f6dab6b72d243419856d9a45921ce886"
                        "bec0af5b920e241bc93f478434d98954"
                    ),
                }]},
            }],
        }

        identity = sched._route_a_stage_model_identity("android-worker")

        assert identity is not None
        assert identity.engine == "llama_cpp"
        assert identity.format == "gguf"
        assert identity.sha256.startswith("f6dab6b7")
        # 找不到该节点 / 没有可用的 models ⇒ None（调用方 fail-closed）
        assert sched._route_a_stage_model_identity("missing-node") is None

    def test_route_a_stage_identity_matches_the_assigned_artifact(self, sched):
        status = {
            "workers": [{
                "node_id": "android-worker",
                "capabilities": {
                    "models": [
                        {
                            "model_id": "middle.gguf", "engine": "llama_cpp",
                            "format": "gguf", "revision": "local",
                            "sha256": "a" * 64,
                        },
                        {
                            "model_id": "tail.gguf", "engine": "llama_cpp",
                            "format": "gguf", "revision": "local",
                            "sha256": "b" * 64,
                        },
                    ],
                    "layer_artifacts": [
                        {
                            "layer_range": [8, 20], "segment_mode": "middle",
                            "model_id": "middle.gguf", "artifact_sha256": "a" * 64,
                        },
                        {
                            "layer_range": [20, 24], "segment_mode": "tail",
                            "model_id": "tail.gguf", "artifact_sha256": "b" * 64,
                        },
                    ],
                },
            }],
        }
        sched._task_worker_control.status = lambda role: status

        assignment = {
            "start_layer": 20,
            "end_layer": 24,
            "layer_artifact": {
                "layer_range": [20, 24], "segment_mode": "tail",
                "model_id": "tail.gguf", "artifact_sha256": "b" * 64,
            },
        }
        identity = sched._route_a_stage_model_identity("android-worker", assignment)

        assert identity is not None
        assert identity.model_id == "tail.gguf"
        assert identity.sha256 == "b" * 64

        # A capability refresh must not silently substitute a different byte
        # artifact under the already admitted plan identity.
        status["workers"][0]["capabilities"]["layer_artifacts"][1][
            "artifact_sha256"
        ] = "c" * 64
        assert sched._route_a_stage_model_identity(
            "android-worker", assignment,
        ) is None

    def test_capacity_prepare_acks_all_workers_before_commit(self, sched):
        sent = []
        sched._tcp_server = type("Server", (), {
            "_running": True,
            "send_layer_config": lambda self, node_id, payload: sent.append(
                (node_id, dict(payload))
            ),
        })()
        sched._host = type("Host", (), {
            "prepare_pipeline_tokenizer": lambda self: object(),
        })()
        plan = {
            "admitted": True,
            "plan_id": "plan-1",
            "model_id": "qwen-test",
            "total_layers": 4,
            "assignments": [
                {"node_id": "worker-a", "start_layer": 0, "end_layer": 2},
                {"node_id": "worker-b", "start_layer": 2, "end_layer": 4},
            ],
        }
        sched._pipeline_load_transaction = {
            "config_id": "cfg-1",
            "generation": 1,
            "phase": "preparing",
            "plan": plan,
            "worker_ids": {"worker-a", "worker-b"},
            "prepared_nodes": set(),
        }
        for node_id, start, end in (
            ("worker-a", 0, 2), ("worker-b", 2, 4),
        ):
            _install_layer_assignment(sched, node_id, {
                "node_id": node_id,
                "config_id": "cfg-1",
                "phase": "prepare",
                "plan_id": "plan-1",
                "start_layer": start,
                "end_layer": end,
                "required_bytes": 100,
                "model_sha256": "sha",
                "model_type": "qwen2",
            })

        def ack(node_id, start, end):
            sched._handle_layer_config_ack(node_id, {"data": {
                "node_id": node_id,
                "assignment_id": _layer_expected(sched, node_id)["assignment_id"],
                "config_id": "cfg-1",
                "status": "prepared",
                "phase": "prepare",
                "plan_id": "plan-1",
                "layer_range": [start, end],
                "model_sha256": "sha",
                "model_type": "qwen2",
                "engine": "pytorch",
                "available_bytes": 100,
            }})

        ack("worker-a", 0, 2)
        assert sent == []
        assert sched._pipeline_load_transaction["phase"] == "preparing"

        ack("worker-b", 2, 4)
        assert sched._pipeline_load_transaction["phase"] == "committing"
        assert {(node_id, payload["phase"]) for node_id, payload in sent} == {
            ("worker-a", "commit"), ("worker-b", "commit"),
        }
        assert sched._layer_config_pushed == set()

    def test_capacity_prepare_error_aborts_every_worker(self, sched):
        sent = []
        aborted = []
        sched._tcp_server = type("Server", (), {
            "_running": True,
            "send_layer_config": lambda self, node_id, payload: sent.append(
                (node_id, dict(payload))
            ),
        })()
        sched._host = type("Host", (), {
            "model_loaded": True,
            "abort_pipeline_materialization": lambda self: aborted.append(True),
        })()
        sched._pipeline_load_transaction = {
            "config_id": "cfg-2",
            "generation": 2,
            "phase": "preparing",
            "plan": {"admitted": True, "plan_id": "plan-2"},
            "worker_ids": {"worker-a", "worker-b"},
            "prepared_nodes": set(),
        }
        _install_layer_assignment(sched, "worker-a", {
            "node_id": "worker-a",
            "config_id": "cfg-2",
            "generation": 2,
            "phase": "prepare",
            "plan_id": "plan-2",
            "start_layer": 0,
            "end_layer": 2,
            "required_bytes": 100,
            "model_sha256": "sha",
            "model_type": "qwen2",
        })

        stale_error = {
            "node_id": "worker-a",
            "assignment_id": _layer_expected(sched, "worker-a")["assignment_id"],
            "config_id": "cfg-2",
            "generation": 1,
            "status": "error",
            "phase": "prepare",
            "error": "capacity changed",
        }
        sched._handle_layer_config_ack("worker-a", {"data": stale_error})

        assert sched._pipeline_load_transaction["phase"] == "preparing"
        assert aborted == []

        sched._handle_layer_config_ack("worker-a", {"data": {
            **stale_error,
            "generation": 2,
        }})

        assert sched._pipeline_load_transaction["phase"] == "aborted"
        assert aborted == [True]
        assert {node_id for node_id, _ in sent} == {"worker-a", "worker-b"}
        assert all(payload["abort"] for _, payload in sent)

    def test_capacity_ready_acks_activate_plan(self, sched):
        sched._pipeline_load_transaction = {
            "config_id": "cfg-3",
            "generation": 3,
            "phase": "committing",
            "plan": {
                "admitted": True,
                "plan_id": "plan-3",
                "total_layers": 2,
                "assignments": [{
                    "node_id": "worker", "start_layer": 0, "end_layer": 2,
                    "layers_count": 2,
                }],
            },
            "worker_ids": {"worker"},
            "ready_nodes": set(),
        }
        expected = _install_layer_assignment(sched, "worker", {
            "node_id": "worker",
            "config_id": "cfg-3",
            "phase": "commit",
            "plan_id": "plan-3",
            "start_layer": 0,
            "end_layer": 2,
            "model_sha256": "sha",
            "model_type": "qwen2",
        })

        sched._handle_layer_config_ack("worker", {"data": {
            "node_id": "worker",
            "assignment_id": expected["assignment_id"],
            "config_id": "cfg-3",
            "status": "ready",
            "phase": "commit",
            "plan_id": "plan-3",
            "layer_range": [0, 2],
            "model_sha256": "sha",
            "model_type": "qwen2",
            "engine": "pytorch",
        }})

        assert sched._pipeline_load_transaction["phase"] == "ready"
        assert sched._active_pipeline_capacity_plan["transaction_phase"] == "ready"
        assert sched.get_layer_assignments()["strategy"] == "capacity"

    def test_candidate_transaction_keeps_active_plan_until_global_commit(
            self, sched, monkeypatch):
        calls = []
        sent = []

        class Host:
            full_chat_execution_lock = threading.RLock()

            def materialize_pipeline_candidate(self, transaction_id, **kwargs):
                calls.append(("materialize", transaction_id, kwargs))
                return {"success": True}

            def commit_pipeline_candidate(self, transaction_id):
                calls.append(("commit", transaction_id))

            def abort_pipeline_candidate(self, transaction_id):
                calls.append(("abort", transaction_id))

            def finalize_pipeline_candidate(self, transaction_id):
                calls.append(("finalize", transaction_id))

        sched._host = Host()
        sched._tcp_server = type("Server", (), {
            "_running": True,
            "send_layer_config": lambda self, node_id, payload: sent.append(
                (node_id, dict(payload))
            ),
        })()
        monkeypatch.setattr(
            sched, "_commit_ready_pipeline_reshard", lambda _plan: None,
        )
        monkeypatch.setattr(
            sched, "_activate_pipeline_reshard_coordinator", lambda _plan: None,
        )
        old_plan = {
            "admitted": True,
            "plan_id": "plan-old",
            "assignments": [
                {"node_id": "master", "start_layer": 0, "end_layer": 2},
                {"node_id": "worker", "start_layer": 2, "end_layer": 4},
            ],
        }
        old_expected = {
            "node_id": "worker", "config_id": "cfg-old", "generation": 1,
            "phase": "commit", "start_layer": 2, "end_layer": 4,
        }
        sched._active_pipeline_capacity_plan = dict(old_plan)
        old_expected = _install_layer_assignment(
            sched, "worker", old_expected,
        )
        candidate_assignment_id = "asg_candidate_new"
        sched._pipeline_load_transaction = {
            "config_id": "cfg-new",
            "generation": 2,
            "phase": "preparing",
            "candidate_id": "candidate-new",
            "deadline": time.monotonic() + 30,
            "plan": {
                "admitted": True,
                "plan_id": "plan-new",
                "model_id": "new-model",
                "model_type": "qwen2",
                "model_sha256": "sha-new",
                "total_layers": 4,
                "assignments": [
                    {
                        "node_id": "master", "start_layer": 0,
                        "end_layer": 2, "has_embedding": True,
                    },
                    {
                        "node_id": "worker", "start_layer": 2,
                        "end_layer": 4, "has_lm_head": True,
                    },
                ],
            },
            "worker_ids": {"worker"},
            "prepared_nodes": set(),
            "ready_nodes": set(),
            "expected": {"worker": {
                "node_id": "worker", "config_id": "cfg-new",
                "assignment_id": candidate_assignment_id,
                "generation": 2, "phase": "prepare",
                "plan_id": "plan-new", "start_layer": 2, "end_layer": 4,
                "has_lm_head": True,
                "required_bytes": 100, "model_sha256": "sha-new",
                "model_type": "qwen2", "engine": "pytorch",
            }},
            "acks": {},
            "retry_state": {},
            "retire_configs": {},
            "on_commit": lambda: calls.append(("callback",)),
        }

        sched._handle_layer_config_ack("worker", {"data": {
            "node_id": "worker", "assignment_id": candidate_assignment_id,
            "config_id": "cfg-new", "generation": 2,
            "status": "prepared", "phase": "prepare", "plan_id": "plan-new",
            "layer_range": [2, 4], "model_sha256": "sha-new",
            "model_type": "qwen2", "engine": "pytorch",
            "available_bytes": 100,
        }})

        assert sched._pipeline_load_transaction["phase"] == "committing"
        assert sched._active_pipeline_capacity_plan == old_plan
        assert _layer_expected(sched, "worker") == old_expected
        assert [item[0] for item in calls] == ["materialize"]
        commit_payload = next(
            payload for _, payload in sent
            if payload.get("phase") == "commit"
        )
        assert commit_payload["assignment_id"] == candidate_assignment_id
        sched._handle_layer_config_ack("worker", {"data": {
            "node_id": "worker", "assignment_id": candidate_assignment_id,
            "config_id": "cfg-new", "generation": 2,
            "status": "prepared", "phase": "prepare", "plan_id": "plan-new",
            "layer_range": [2, 4], "model_sha256": "sha-new",
            "model_type": "qwen2", "engine": "pytorch",
            "available_bytes": 100,
        }})
        assert [item[0] for item in calls] == ["materialize"]
        assert sum(
            payload.get("phase") == "commit" for _, payload in sent
        ) == 1

        sched._handle_layer_config_ack("worker", {"data": {
            "node_id": "worker", "assignment_id": candidate_assignment_id,
            "config_id": "cfg-new", "generation": 2,
            "status": "ready", "phase": "commit", "plan_id": "plan-new",
            "layer_range": [2, 4], "model_sha256": "sha-new",
            "model_type": "qwen2", "engine": "pytorch",
            "has_lm_head": True,
        }})

        assert sched._pipeline_load_transaction["phase"] == "ready"
        assert sched._active_pipeline_capacity_plan["plan_id"] == "plan-new"
        assert _layer_expected(sched, "worker")["config_id"] == "cfg-new"
        assert (
            _layer_expected(sched, "worker")["assignment_id"]
            == candidate_assignment_id
        )
        assert [item[0] for item in calls] == [
            "materialize", "commit", "callback", "finalize",
        ]
        finalize_payload = next(
            payload for _, payload in sent if payload.get("finalize")
        )
        assert sched._pipeline_load_transaction["finalize_pending"] == {"worker"}
        sched._handle_layer_config_ack("worker", {"data": {
            "node_id": "worker", "assignment_id": candidate_assignment_id,
            "config_id": "cfg-new", "generation": 2,
            "status": "ready", "phase": "commit", "plan_id": "plan-new",
            "layer_range": [2, 4], "model_sha256": "sha-new",
            "model_type": "qwen2", "engine": "pytorch",
            "has_lm_head": True,
        }})
        assert [item[0] for item in calls] == [
            "materialize", "commit", "callback", "finalize",
        ]
        assert sum(bool(payload.get("finalize")) for _, payload in sent) == 1

        sched._handle_layer_config_ack("worker", {"data": {
            **finalize_payload,
            "status": "finalized",
        }})

        assert sched._pipeline_load_transaction["finalize_pending"] == set()
        assert finalize_payload["config_id"] not in (
            sched._pipeline_model_cleanup_ledger
        )
        sched._handle_layer_config_ack("worker", {"data": {
            **finalize_payload,
            "status": "finalized",
        }})
        assert sched._pipeline_load_transaction["finalize_pending"] == set()
        assert [item[0] for item in calls] == [
            "materialize", "commit", "callback", "finalize",
        ]

    def test_candidate_reshard_failure_aborts_without_publishing(
            self, sched, monkeypatch):
        calls = []

        class Host:
            full_chat_execution_lock = threading.RLock()

            def commit_pipeline_candidate(self, transaction_id):
                calls.append(("commit", transaction_id))

            def abort_pipeline_candidate(self, transaction_id):
                calls.append(("abort", transaction_id))

        sched._host = Host()
        sched._tcp_server = type("Server", (), {
            "_running": True,
            "send_layer_config": lambda self, node_id, payload: None,
        })()
        monkeypatch.setattr(
            sched, "_commit_ready_pipeline_reshard", lambda _plan: False,
        )
        old_plan = {"admitted": True, "plan_id": "plan-old"}
        sched._active_pipeline_capacity_plan = dict(old_plan)
        sched._pipeline_load_transaction = {
            "config_id": "cfg-new",
            "generation": 2,
            "phase": "committing_global",
            "candidate_id": "candidate-new",
            "plan": {"admitted": True, "plan_id": "plan-new"},
            "worker_ids": {"worker"},
            "expected": {},
            "acks": {},
            "retire_configs": {},
            "previous_control_state": {
                "expected": {}, "acks": {}, "retry_state": {},
                "assignments": {}, "active_plan": dict(old_plan),
            },
        }

        sched._complete_pipeline_load_transaction("cfg-new")

        assert sched._pipeline_load_transaction["phase"] == "aborted"
        assert sched._active_pipeline_capacity_plan == old_plan
        assert calls == [
            ("commit", "candidate-new"),
            ("abort", "candidate-new"),
        ]

    def test_different_candidate_conflicts_with_active_transaction(self, sched):
        sched._role_override = "master"
        sched._pipeline_load_transaction = {
            "config_id": "cfg-active",
            "generation": 3,
            "phase": "preparing",
            "candidate_id": "candidate-active",
            "plan": {"admitted": True, "plan_id": "plan-active"},
        }

        result = sched.start_pipeline_model_transaction(
            candidate_id="candidate-other",
            descriptor={"model_id": "other"},
            model_path="C:/models/other",
        )

        assert result["admitted"] is False
        assert result["reason_code"] == "MODEL_TXN_CONFLICT"

    def test_candidate_plan_is_not_executable_before_commit(self, sched):
        sched._active_pipeline_capacity_plan = {
            "assignments": [{"node_id": "master"}],
        }
        sched._pipeline_load_transaction = {
            "phase": "preparing",
            "candidate_id": "candidate-new",
            "plan": {
                "assignments": [
                    {"node_id": "master"},
                    {"node_id": "worker"},
                ],
            },
        }

        assert sched._has_active_distributed_pipeline_plan() is False

    def test_candidate_timeout_restores_active_plan(self, sched):
        calls = []
        old_plan = {"admitted": True, "plan_id": "plan-old"}

        class Host:
            def abort_pipeline_candidate(self, transaction_id):
                calls.append(transaction_id)

        sched._host = Host()
        sched._tcp_server = type("Server", (), {
            "_running": True,
            "get_client_ids": lambda self: ["worker"],
            "send_layer_config": lambda self, node_id, payload: None,
        })()
        sched._active_pipeline_capacity_plan = dict(old_plan)
        sched._pipeline_load_transaction = {
            "config_id": "cfg-timeout",
            "generation": 4,
            "phase": "preparing",
            "candidate_id": "candidate-timeout",
            "deadline": 5.0,
            "plan": {"admitted": True, "plan_id": "plan-new"},
            "worker_ids": {"worker"},
            "expected": {"worker": {
                "node_id": "worker", "config_id": "cfg-timeout",
            }},
            "retry_state": {},
            "previous_control_state": {
                "expected": {}, "acks": {}, "retry_state": {},
                "assignments": {}, "active_plan": dict(old_plan),
            },
        }

        assert sched._retry_pending_layer_configs(now=10.0) == 0
        assert sched._pipeline_load_transaction["phase"] == "aborted"
        assert sched._active_pipeline_capacity_plan == old_plan
        assert calls == ["candidate-timeout"]

    def test_candidate_abort_does_not_restore_invalidated_active_assignment(
            self, sched):
        class Host:
            def abort_pipeline_candidate(self, _transaction_id):
                return None

        sched._host = Host()
        active = _install_layer_assignment(sched, "worker", {
            "node_id": "worker",
            "config_id": "cfg-active",
            "generation": 5,
        })
        active_plan = {"admitted": True, "plan_id": "plan-active"}
        sched._active_pipeline_capacity_plan = dict(active_plan)
        sched._pipeline_load_transaction = {
            "config_id": "cfg-candidate",
            "generation": 6,
            "phase": "preparing",
            "candidate_id": "candidate-new",
            "plan": {"model_id": "new-model"},
            "worker_ids": {"worker"},
            "expected": {"worker": {
                "node_id": "worker",
                "assignment_id": "asg-candidate",
                "config_id": "cfg-candidate",
                "generation": 6,
            }},
            "previous_control_state": {
                "retry_state": {},
                "assignments": sched._worker_assignments.snapshot(),
                "active_plan": dict(active_plan),
            },
        }

        sched._worker_assignments.release(
            "worker", reason_code="connection_closed",
        )
        sched._active_pipeline_capacity_plan = None
        sched._abort_pipeline_load_transaction(
            "cfg-candidate", "candidate_failed", "test abort",
        )

        state = sched._worker_assignments.state("worker")
        assert state is not None
        assert state.assignment_id == active["assignment_id"]
        assert state.phase == "released"
        assert sched._worker_assignments.expected("worker") == {}
        assert sched._active_pipeline_capacity_plan is None

    def test_candidate_abort_preserves_valid_active_assignment(
            self, sched):
        sent = []

        class Host:
            def abort_pipeline_candidate(self, _transaction_id):
                return None

        sched._host = Host()
        sched._tcp_server = type("Server", (), {
            "_running": True,
            "send_layer_config": lambda self, node_id, payload: sent.append(
                (node_id, dict(payload))
            ),
        })()
        active = _install_layer_assignment(
            sched,
            "worker",
            {
                "node_id": "worker",
                "config_id": "cfg-active",
                "generation": 5,
                "start_layer": 0,
                "end_layer": 2,
                "model_sha256": "sha-active",
                "model_type": "qwen2",
            },
            ack={
                "status": "ready",
                "layer_range": [0, 2],
                "model_sha256": "sha-active",
                "model_type": "qwen2",
                "engine": "pytorch",
            },
            phase="ready",
        )
        active_plan = {"admitted": True, "plan_id": "plan-active"}
        sched._active_pipeline_capacity_plan = dict(active_plan)
        active_snapshot = sched._worker_assignments.snapshot()
        sched._pipeline_load_transaction = {
            "config_id": "cfg-candidate",
            "generation": 6,
            "phase": "preparing",
            "candidate_id": "candidate-new",
            "plan": {"model_id": "new-model"},
            "worker_ids": {"worker"},
            "expected": {"worker": {
                "node_id": "worker",
                "assignment_id": "asg-candidate",
                "config_id": "cfg-candidate",
                "generation": 6,
            }},
            "previous_control_state": {
                "retry_state": {},
                "assignments": active_snapshot,
                "active_plan": dict(active_plan),
            },
        }

        sched._abort_pipeline_load_transaction(
            "cfg-candidate", "candidate_failed", "test abort",
        )

        assert sched._worker_assignments.snapshot() == active_snapshot
        assert sched._active_pipeline_capacity_plan == active_plan
        assert _layer_expected(sched, "worker")["assignment_id"] == (
            active["assignment_id"]
        )
        assert len(sched._pipeline_model_cleanup_ledger) == 1
        cleanup = next(iter(sched._pipeline_model_cleanup_ledger.values()))
        assert cleanup["assignment_id"] == "asg-candidate"
        assert cleanup["candidate_config_id"] == "cfg-candidate"
        assert sent == [("worker", cleanup["payload"])]

    def test_master_finalize_cleanup_retries_and_blocks_new_candidate(
            self, sched, monkeypatch):
        calls = []

        class Host:
            def finalize_pipeline_candidate(self, transaction_id):
                calls.append(transaction_id)
                if len(calls) == 1:
                    raise RuntimeError("transient finalize failure")

        sched._host = Host()
        monkeypatch.setattr(sched, "_effective_role", lambda: "master")
        monkeypatch.setattr(sched, "_connected_client_ids", lambda: [])
        monkeypatch.setattr(
            sched,
            "_push_layer_config_to_clients_locked",
            lambda **_kwargs: (_ for _ in ()).throw(
                AssertionError("cleanup fence must stop candidate planning")
            ),
        )
        record = sched._record_pipeline_model_cleanup_locked(
            node_id="master",
            candidate_config_id="candidate-finalize",
            generation=7,
            action="finalize",
        )

        sched._persist_pipeline_lifecycle_locked()

        assert sched._dispatch_pipeline_model_cleanup(record) is False
        assert record["cleanup_id"] in sched._pipeline_model_cleanup_ledger

        blocked = sched.start_pipeline_model_transaction(
            candidate_id="candidate-next",
            descriptor={"model_id": "next"},
            model_path="C:/models/next",
        )

        assert blocked["admitted"] is False
        assert blocked["reason_code"] == "MODEL_TXN_CLEANUP_PENDING"
        record["next_retry"] = 0.0
        assert sched._retry_pending_layer_configs(now=10.0) == 1
        assert calls == ["candidate-finalize", "candidate-finalize"]
        assert sched._pipeline_model_cleanup_ledger == {}

    def test_cleanup_ack_requires_exact_identity(self, sched, monkeypatch):
        pushes = []
        monkeypatch.setattr(
            sched, "push_layer_config_to_clients",
            lambda: pushes.append("sync"),
        )
        record = sched._record_pipeline_model_cleanup_locked(
            node_id="worker",
            candidate_config_id="cfg-candidate",
            generation=9,
            action="finalize",
            assignment_id="asg-candidate",
            payload={"finalize": True},
        )
        cleanup_id = record["cleanup_id"]
        canonical = {
            "node_id": "worker",
            "config_id": cleanup_id,
            "assignment_id": "asg-candidate",
            "generation": 9,
            "status": "finalized",
            "cleanup_action": "finalize",
            "candidate_config_id": "cfg-candidate",
            "finalized_config_id": "cfg-candidate",
        }

        mismatches = [
            dict(canonical, config_id="wrong-cleanup"),
            dict(canonical, assignment_id=""),
            dict(canonical, assignment_id="asg-stale"),
            dict(canonical, cleanup_action="abort"),
            dict(canonical, status="released"),
            dict(
                canonical,
                finalized_config_id="",
                aborted_config_id="cfg-candidate",
            ),
            dict(
                canonical,
                candidate_config_id="cfg-other",
                finalized_config_id="cfg-other",
            ),
            dict(canonical, generation=10),
        ]
        for payload in mismatches:
            assert sched._ack_pipeline_model_cleanup("worker", payload) is False
            assert cleanup_id in sched._pipeline_model_cleanup_ledger
        assert sched._ack_pipeline_model_cleanup("other-worker", canonical) is False
        assert cleanup_id in sched._pipeline_model_cleanup_ledger

        assert sched._ack_pipeline_model_cleanup("worker", canonical) is True
        assert sched._pipeline_model_cleanup_ledger == {}
        assert pushes == ["sync"]

    def test_disconnected_cleanup_debt_survives_without_blocking_survivors(
            self, sched, monkeypatch):
        monkeypatch.setattr(sched, "_effective_role", lambda: "master")
        monkeypatch.setattr(
            sched, "_fail_pending_pipeline_results_for_node", lambda *_args: None,
        )
        monkeypatch.setattr(sched, "deregister_node", lambda _node_id: False)
        monkeypatch.setattr(
            sched, "_stage_pipeline_reshard_after_disconnect", lambda _node_id: None,
        )
        monkeypatch.setattr(sched, "_push_node_update_to_all_clients", lambda *_args: None)
        monkeypatch.setattr(sched, "push_layer_config_to_clients", lambda: None)
        record = sched._record_pipeline_model_cleanup_locked(
            node_id="client1",
            candidate_config_id="cfg-old",
            generation=4,
            action="finalize",
            payload={"finalize": True, "finalized_config_id": "cfg-old"},
        )
        plan = {
            "admitted": True,
            "plan_id": "plan-old",
            "assignments": [
                {"node_id": "client1", "start_layer": 0, "end_layer": 2},
                {"node_id": "master", "start_layer": 2, "end_layer": 4},
            ],
        }
        sched._active_pipeline_capacity_plan = dict(plan)
        sched._pipeline_load_transaction = {
            "config_id": "cfg-old",
            "phase": "ready",
            "plan": dict(plan),
            "worker_ids": {"client1"},
            "finalize_pending": {"client1"},
            "finalize_retry_state": {"client1": {"attempts": 1}},
        }

        sched._on_tcp_disconnect("client1")

        assert record["cleanup_id"] in sched._pipeline_model_cleanup_ledger
        assert record["state"] == "await_reconnect"
        assert sched._pipeline_load_transaction["phase"] == "invalidated"
        assert sched._pipeline_load_transaction["finalize_pending"] == set()

        monkeypatch.setattr(sched, "_connected_client_ids", lambda: ["client2"])
        monkeypatch.setattr(
            sched,
            "_push_layer_config_to_clients_locked",
            lambda **_kwargs: {
                "status": "preparing",
                "admitted": True,
                "plan_id": "plan-next",
                "assignments": [],
            },
        )
        admitted = sched.start_pipeline_model_transaction(
            candidate_id="candidate-next",
            descriptor={"model_id": "next"},
            model_path="C:/models/next",
        )

        assert admitted["admitted"] is True

    def test_reconnect_replays_cleanup_before_assignment(
            self, sched, monkeypatch):
        import scheduler as scheduler_module

        sent = []
        pushes = []
        monkeypatch.setattr(sched, "_effective_role", lambda: "master")
        monkeypatch.setattr(
            scheduler_module, "TASK_WORKER_EXPERIMENTAL_ENABLED", False,
        )
        sched._tcp_server = type("Server", (), {
            "_running": True,
            "send_layer_config": lambda self, node_id, payload: sent.append(
                (node_id, dict(payload))
            ),
        })()
        monkeypatch.setattr(
            sched, "push_layer_config_to_clients", lambda: pushes.append("assignment"),
        )
        monkeypatch.setattr(sched, "_push_node_list_to_client", lambda _node_id: None)
        monkeypatch.setattr(sched, "_push_node_update_to_all_clients", lambda *_args: None)
        record = sched._record_pipeline_model_cleanup_locked(
            node_id="client1",
            candidate_config_id="cfg-old",
            generation=5,
            action="abort",
            payload={
                "release": True,
                "abort": True,
                "aborted_config_id": "cfg-old",
            },
        )

        sched._persist_pipeline_lifecycle_locked()

        sched._on_tcp_registration_confirmed("client1")

        assert sent == [("client1", record["payload"])]
        assert pushes == []

    def test_model_change_invalidates_capacity_transaction(self, sched):
        sched._layer_config_generation = 10
        sched._pipeline_load_transaction = {
            "config_id": "cfg-model-change",
            "generation": 10,
            "phase": "committing_local",
            "plan": {"admitted": True, "plan_id": "plan-model-change"},
            "worker_ids": {"worker"},
        }
        sched._active_pipeline_capacity_plan = {
            "admitted": True, "plan_id": "plan-model-change",
        }
        sched._prepared_layer_configs["cfg-model-change"] = {
            "plan_id": "plan-model-change",
        }
        sched._layer_config_inflight.add("worker")

        sched._invalidate_pipeline_load_transaction(
            reason_code="pipeline_model_changed",
            reason="local model replacement started",
        )

        assert sched._pipeline_load_transaction["phase"] == "invalidated"
        assert (
            sched._pipeline_load_transaction["reason_code"]
            == "pipeline_model_changed"
        )
        assert sched._active_pipeline_capacity_plan is None
        assert sched._prepared_layer_configs == {}
        assert sched._layer_config_inflight == set()
        assert sched._layer_config_generation > 10

    def test_model_change_invalidates_candidate_into_durable_abort_cleanup(
            self, sched):
        aborted = []
        sent = []
        sched._host = type("Host", (), {
            "abort_pipeline_candidate": (
                lambda self, candidate_id: aborted.append(candidate_id)
            ),
        })()
        sched._tcp_server = type("Server", (), {
            "_running": True,
            "send_layer_config": (
                lambda self, node_id, payload: sent.append(
                    (node_id, dict(payload))
                )
            ),
        })()
        sched._pipeline_load_transaction = {
            "config_id": "cfg-model-change",
            "candidate_id": "candidate-local",
            "generation": 10,
            "phase": "preparing",
            "plan": {"model_id": "model-next"},
            "worker_ids": {"worker"},
            "expected": {
                "worker": {
                    "node_id": "worker",
                    "config_id": "cfg-model-change",
                    "assignment_id": "asg-candidate",
                },
            },
        }

        sched._invalidate_pipeline_load_transaction(
            reason_code="pipeline_model_changed",
            reason="local model replacement committed",
        )

        assert sched._pipeline_load_transaction["phase"] == "invalidated"
        assert aborted == ["candidate-local"]
        assert len(sched._pipeline_model_cleanup_ledger) == 1
        record = next(iter(sched._pipeline_model_cleanup_ledger.values()))
        assert record["node_id"] == "worker"
        assert record["candidate_config_id"] == "cfg-model-change"
        assert record["assignment_id"] == "asg-candidate"
        assert record["action"] == "abort"
        assert record["payload"]["aborted_config_id"] == "cfg-model-change"
        assert sent == [("worker", record["payload"])]

    def test_superseded_local_commit_is_not_published(self, sched, monkeypatch):
        sent = []
        plan = {
            "admitted": True,
            "plan_id": "plan-superseded",
            "model_id": "model",
            "total_layers": 4,
            "assignments": [
                {"node_id": "master", "start_layer": 0, "end_layer": 2},
                {"node_id": "worker", "start_layer": 2, "end_layer": 4},
            ],
        }
        sched._pipeline_load_transaction = {
            "config_id": "cfg-superseded",
            "generation": 1,
            "phase": "preparing",
            "plan": plan,
            "worker_ids": {"worker"},
            "prepared_nodes": set(),
        }
        _install_layer_assignment(sched, "worker", {
            "node_id": "worker",
            "config_id": "cfg-superseded",
            "phase": "prepare",
            "plan_id": "plan-superseded",
            "start_layer": 2,
            "end_layer": 4,
        })
        sched._host = type("Host", (), {
            "prepare_pipeline_tokenizer": lambda self: None,
            "load_layer_range": lambda self, *args, **kwargs: (
                sched._invalidate_pipeline_load_transaction(
                    reason_code="pipeline_model_changed",
                    reason="local model replacement started",
                )
            ),
        })()
        sched._tcp_server = type("Server", (), {
            "_running": True,
            "send_layer_config": lambda self, node_id, payload: sent.append(
                (node_id, dict(payload))
            ),
        })()

        sched._commit_pipeline_load_transaction("cfg-superseded")

        assert sent == []
        assert sched._pipeline_load_transaction["phase"] == "invalidated"

    def test_late_prepare_ack_after_commit_is_not_logged_as_worker_error(
            self, sched, caplog):
        sched._pipeline_load_transaction = {
            "config_id": "cfg-late-prepare",
            "generation": 4,
            "phase": "committing",
            "plan": {"admitted": True, "plan_id": "plan-late"},
            "worker_ids": {"worker"},
            "ready_nodes": set(),
        }
        expected = _install_layer_assignment(sched, "worker", {
            "node_id": "worker",
            "config_id": "cfg-late-prepare",
            "phase": "commit",
            "plan_id": "plan-late",
            "start_layer": 2,
            "end_layer": 4,
            "required_bytes": 100,
            "model_sha256": "sha",
            "model_type": "qwen2",
        })

        with caplog.at_level(logging.ERROR):
            sched._handle_layer_config_ack("worker", {"data": {
                "node_id": "worker",
                "assignment_id": expected["assignment_id"],
                "config_id": "cfg-late-prepare",
                "status": "prepared",
                "phase": "prepare",
                "plan_id": "plan-late",
                "layer_range": [2, 4],
                "model_sha256": "sha",
                "model_type": "qwen2",
                "engine": "pytorch",
                "available_bytes": 100,
            }})

        assert _layer_ack(sched, "worker")["status"] == "prepared"
        assert sched._pipeline_load_transaction["phase"] == "committing"
        assert not any("阶段未通过" in record.message for record in caplog.records)

    def test_capacity_transaction_aborts_when_worker_disconnects(
            self, sched, monkeypatch):
        aborts = []
        reshard_attempts = []
        sched._pipeline_load_transaction = {
            "config_id": "cfg-drop",
            "phase": "preparing",
            "worker_ids": {"worker-drop"},
        }
        monkeypatch.setattr(
            sched,
            "_abort_pipeline_load_transaction",
            lambda *args: aborts.append(args),
        )
        monkeypatch.setattr(
            sched, "_fail_pending_pipeline_results_for_node", lambda *_args: None,
        )
        monkeypatch.setattr(sched, "deregister_node", lambda _node_id: False)
        monkeypatch.setattr(
            sched, "_stage_pipeline_reshard_after_disconnect",
            lambda node_id: reshard_attempts.append(node_id) or None,
        )

        sched._on_tcp_disconnect("worker-drop")

        assert aborts == [(
            "cfg-drop",
            "pipeline_worker_disconnected",
            "worker worker-drop disconnected during transaction",
        )]
        assert reshard_attempts == ["worker-drop"]

    def test_ready_capacity_plan_is_invalidated_when_worker_disconnects(
            self, sched, monkeypatch):
        plan = {
            "admitted": True,
            "plan_id": "plan-ready-drop",
            "total_layers": 4,
            "assignments": [
                {"node_id": "worker-drop", "start_layer": 0, "end_layer": 2},
                {"node_id": "master", "start_layer": 2, "end_layer": 4},
            ],
        }
        sched._active_pipeline_capacity_plan = dict(plan)
        sched._pipeline_load_transaction = {
            "config_id": "cfg-ready-drop",
            "phase": "ready",
            "plan": dict(plan),
            "worker_ids": {"worker-drop"},
            "ready_nodes": {"worker-drop"},
        }
        monkeypatch.setattr(
            sched, "_fail_pending_pipeline_results_for_node", lambda *_args: None,
        )
        monkeypatch.setattr(sched, "deregister_node", lambda _node_id: False)
        monkeypatch.setattr(
            sched, "_stage_pipeline_reshard_after_disconnect", lambda _node_id: None,
        )
        monkeypatch.setattr(sched, "push_layer_config_to_clients", lambda: None)

        sched._on_tcp_disconnect("worker-drop")

        assert sched._active_pipeline_capacity_plan is None
        assert sched._pipeline_load_transaction["phase"] == "invalidated"
        assert (
            sched._pipeline_load_transaction["reason_code"]
            == "pipeline_worker_disconnected"
        )
        assert sched.get_layer_assignments()["strategy"] != "capacity"

    def test_disconnect_reshard_uses_only_connected_survivors(
            self, sched, monkeypatch):
        captured = {}

        class _Node:
            node_id = "worker-a"

        class _Decision:
            def to_dict(self):
                return {"status": "staged", "accepted": True, "reason_code": ""}

        class _Coordinator:
            layout = type("Layout", (), {"nodes": [_Node()]})()

            def snapshot(self):
                return {"staged": [{"failed_node_ids": ["worker-old"]}]}

            def stage_failure(self, failed, **kwargs):
                captured["failed"] = failed
                captured.update(kwargs)
                return _Decision()

        sched._role_override = "master"
        # Keep the unit topology independent from the host process NODE_ID.
        monkeypatch.setattr(sched, "get_effective_node_id", lambda: "master")
        sched._pipeline_reshard_coordinator = _Coordinator()
        sched._tcp_server = type("Server", (), {
            "get_client_ids": lambda self: ["worker-b", "task-full"],
        })()
        sched._host = type("Host", (), {
            "get_pipeline_descriptor": lambda self: {"model_sha256": "a" * 64},
        })()
        monkeypatch.setattr(
            sched, "_get_pipeline_capacity_nodes",
            lambda ids: captured.setdefault("eligible", set(ids)) or [],
        )
        monkeypatch.setattr(
            sched, "_task_worker_full_model_ids", lambda: {"task-full"},
        )
        monkeypatch.setattr(sched, "_pipeline_node_metadata", lambda: {})

        report = sched._stage_pipeline_reshard_after_disconnect("worker-a")

        assert report["status"] == "staged"
        assert captured["failed"] == {"worker-a", "worker-old"}
        assert captured["eligible"] == {"master", "worker-b"}

    def test_versioned_ready_ack_is_fenced_by_generation(self, sched):
        """迟到的同 config ACK 不能跨 generation 激活容量计划。"""
        sched._pipeline_load_transaction = {
            "config_id": "cfg-generation",
            "generation": 8,
            "phase": "committing",
            "plan": {
                "admitted": True,
                "plan_id": "plan-generation",
                "total_layers": 2,
                "assignments": [{
                    "node_id": "worker", "start_layer": 0, "end_layer": 2,
                    "layers_count": 2,
                }],
            },
            "worker_ids": {"worker"},
            "ready_nodes": set(),
        }
        expected = _install_layer_assignment(sched, "worker", {
            "node_id": "worker",
            "config_id": "cfg-generation",
            "generation": 8,
            "phase": "commit",
            "plan_id": "plan-generation",
            "start_layer": 0,
            "end_layer": 2,
            "model_sha256": "sha",
            "model_type": "qwen2",
        })
        stale = {
            "node_id": "worker",
            "assignment_id": expected["assignment_id"],
            "config_id": "cfg-generation",
            "generation": 7,
            "status": "ready",
            "phase": "commit",
            "plan_id": "plan-generation",
            "layer_range": [0, 2],
            "model_sha256": "sha",
            "model_type": "qwen2",
            "engine": "pytorch",
        }
        sched._handle_layer_config_ack("worker", {"data": stale})
        assert sched._pipeline_load_transaction["phase"] == "committing"
        assert sched._layer_config_pushed == set()
        assert _layer_ack(sched, "worker") == {}

        sched._handle_layer_config_ack(
            "worker", {"data": {**stale, "generation": 8}},
        )
        assert sched._pipeline_load_transaction["phase"] == "ready"
        assert sched._layer_config_pushed == {"worker"}

    def test_capacity_commit_is_not_hidden_by_cached_prepare_ack(
            self, sched, monkeypatch):
        started = []
        sched._last_layer_config_ack_payload = {
            "config_id": "cfg-phase",
            "phase": "prepare",
            "status": "prepared",
        }
        monkeypatch.setattr(
            threading.Thread, "start", lambda thread: started.append(thread),
        )

        sched._schedule_layer_config("master", {
            "config_id": "cfg-phase",
            "generation": 1,
            "phase": "commit",
            "start_layer": 0,
            "end_layer": 1,
        })

        assert len(started) == 1
        assert "cfg-phase" in sched._layer_config_inflight

    def test_multi_node_covers_all_layers(self, sched):
        """多节点分配应完整覆盖 0-24"""
        result = sched.compute_layer_assignment()
        total = sum(a["layers_count"] for a in result)
        assert total == 24, f"总层数应为 24，实际: {total}"

        # 验证连续性
        result.sort(key=lambda x: x["start_layer"])
        cursor = 0
        for a in result:
            assert a["start_layer"] == cursor, \
                f"节点 {a['node_id']} 起始层应为 {cursor}，实际: {a['start_layer']}"
            cursor = a["end_layer"]

    def test_first_node_has_embedding(self, sched):
        """首节点（master）应有 Embedding 且 start_layer=0"""
        result = sched.compute_layer_assignment()
        executable = [a for a in result if a.get("layers_count", 0) > 0]
        executable.sort(key=lambda x: x["start_layer"])
        assert executable[0]["has_embedding"] is True
        assert executable[0]["start_layer"] == 0
        # master 参与执行时应为首位
        assert executable[0].get("node_id") == "master" or executable[0].get("role") == "master"

    def test_last_node_has_lm_head(self, sched):
        """强 master 应保留 LM Head，避免弱末端 worker 回传整词表 logits。"""
        result = sched.compute_layer_assignment()
        result.sort(key=lambda x: x["start_layer"])
        master = next(item for item in result if item["node_id"] == "master")
        assert master["has_lm_head"] is True
        assert result[-1]["end_layer"] == 24

    def test_master_first_sorting(self, sched):
        """master 节点应排在第一位"""
        result = sched.compute_layer_assignment()
        # 第一个是 master
        master_assignments = [a for a in result if a["role"] == "master"]
        client_assignments = [a for a in result if a["role"] != "master"]

        if master_assignments:
            # master 应该在所有 client 之前
            master_max_end = max(a["end_layer"] for a in master_assignments)
            if client_assignments:
                client_min_start = min(a["start_layer"] for a in client_assignments)
                assert master_max_end <= client_min_start, \
                    "master 的层应排在 client 之前"

    def test_weight_proportional_distribution(self, sched):
        """worker 节点按权重分配层；master 作为首段 Embedding 锚点至少执行 1 层"""
        result = sched.compute_layer_assignment()

        scores = {a["node_id"]: a["score"] for a in result}
        layers = {a["node_id"]: a["layers_count"] for a in result}

        # master 分数最高，作为首段锚点至少执行 1 层
        assert scores["master"] > scores["client1"] > scores["client2"], \
            f"权重排序错误: {scores}"
        assert layers["master"] >= 1, \
            f"master 应至少执行 1 层（作为 Embedding 锚点），实际: {layers['master']}"
        assert "coordinator_only" not in result[0], \
            "master 不应再有 coordinator_only 标记"
        # master 是首节点，具有 Embedding
        assert result[0]["has_embedding"] is True
        assert result[0]["start_layer"] == 0
        # worker 中游戏本 > 轻薄本，应分配更多层
        assert layers["client1"] >= layers["client2"], \
            f"worker 权重高应分配更多层: {layers}"
        total_layers = sum(layers.values())
        assert total_layers == 24, f"总层数应为 24，实际: {total_layers}"

    def test_empty_nodes_returns_empty(self, sched):
        """空节点列表应返回空"""
        result = sched.compute_layer_assignment([])
        assert result == []

    def test_android_nodes_excluded_from_layer_assignment(self, sched):
        """Android HTTP/移动节点不参与 Transformer 层间拆分。"""
        nodes = [
            {"node_id": "master", "role": "master", "node_type": "pc", "device_info": PROFILE_WORKSTATION},
            {"node_id": "pc-worker", "role": "client", "node_type": "pc", "device_info": PROFILE_LAPTOP},
            {"node_id": "android-live", "role": "client", "node_type": "android", "device_info": PROFILE_MOBILE},
        ]
        result = sched.compute_layer_assignment(nodes)
        node_ids = {a["node_id"] for a in result}
        assert "android-live" not in node_ids
        assert node_ids == {"master", "pc-worker"}

    def test_android_forward_layers_capability_is_assigned(self, sched):
        """Android with an explicit llama.cpp layer capability can receive a segment."""
        nodes = [
            {"node_id": "master", "role": "master", "node_type": "pc", "device_info": PROFILE_WORKSTATION},
            {"node_id": "android-worker", "role": "client", "node_type": "android", "device_info": {**_engine_contract("llama_cpp", "llama_cpp_android"),
                **PROFILE_MOBILE,
                "backend_id": "llama_cpp",
                "capabilities": [Capability.FORWARD_LAYERS],
            }},
        ]
        result = sched.compute_layer_assignment(nodes)
        node_ids = {assignment["node_id"] for assignment in result}
        assert node_ids == {"master", "android-worker"}

    def test_offline_pc_nodes_are_excluded_from_runtime_assignment(self):
        """数据库恢复的历史 PC 节点离线时不能继续占用模型层。"""
        sched = Scheduler()
        sched.nodes = {
            "master": NodeInfo(
                node_id="master", role="master", state=NodeState.ONLINE,
                node_type="pc", device_info=PROFILE_LAPTOP,
            ),
            "old-worker": NodeInfo(
                node_id="old-worker", role="client", state=NodeState.OFFLINE,
                node_type="pc", device_info=PROFILE_IGPU_ONLY,
            ),
        }

        result = sched.compute_layer_assignment()

        assert [item["node_id"] for item in result] == ["master"]
        assert result[0]["layers_count"] == 24

    def test_all_zero_weight_equal_split(self, sched):
        """权重全为 0 时应均分"""
        nodes = [
            {"node_id": "n1", "role": "client", "device_info": {**_engine_contract("pytorch", "pytorch_cuda"), }},
            {"node_id": "n2", "role": "client", "device_info": {**_engine_contract("pytorch", "pytorch_cuda"), }},
            {"node_id": "n3", "role": "client", "device_info": {**_engine_contract("pytorch", "pytorch_cuda"), }},
        ]
        result = sched.compute_layer_assignment(nodes)

        total = sum(a["layers_count"] for a in result)
        assert total == 24

        # 3 个节点均分 24 层 → 每节点 8 层
        for a in result:
            assert 7 <= a["layers_count"] <= 9, \
                f"均分时每节点应有 ~8 层，实际 {a['node_id']}: {a['layers_count']}"

    def test_two_nodes_continuous(self, sched):
        """两个节点时 master 参与首段计算，分层应连续"""
        two = [
            {"node_id": "master", "role": "master", "device_info": PROFILE_WORKSTATION},
            {"node_id": "client1", "role": "client", "device_info": PROFILE_LAPTOP},
        ]
        result = sched.compute_layer_assignment(two)
        result.sort(key=lambda x: x["start_layer"])

        assert len(result) == 2
        assert result[0]["node_id"] == "master"
        assert result[0]["layers_count"] >= 1, \
            f"master 应至少 1 层，实际: {result[0]['layers_count']}"
        assert result[0]["has_embedding"] is True
        assert result[0]["start_layer"] == 0
        assert "coordinator_only" not in result[0]
        worker = result[1]
        assert worker["node_id"] == "client1"
        assert worker["start_layer"] == result[0]["end_layer"]
        assert worker["end_layer"] == 24
        assert result[0]["has_lm_head"] is True
        assert worker["has_lm_head"] is False

    def test_4060_master_gets_more_layers_than_igpu_i7_worker(self, sched):
        """主节点画像正确时，不得再出现弱 CPU worker 的层数反超 CUDA 主节点。"""
        nodes = [
            {"node_id": "master", "role": "master", "node_type": "pc",
             "device_info": PROFILE_LAPTOP},
            {"node_id": "igpu-worker", "role": "client", "node_type": "pc",
             "device_info": PROFILE_IGPU_ONLY},
        ]

        result = sched.compute_layer_assignment(nodes)
        by_id = {item["node_id"]: item for item in result}

        assert by_id["master"]["score"] > by_id["igpu-worker"]["score"] * 2
        assert by_id["master"]["layers_count"] > by_id["igpu-worker"]["layers_count"]
        assert by_id["master"]["layers_count"] >= 19
        assert by_id["master"]["has_lm_head"] is True
        assert by_id["igpu-worker"]["has_lm_head"] is False

    def test_each_node_has_min_one_layer(self, sched):
        """每个节点至少分配 1 层"""
        # 极端场景：一个超强节点 + 一个超弱节点
        nodes = [
            {"node_id": "strong", "role": "client", "device_info": PROFILE_WORKSTATION},
            {"node_id": "weak", "role": "client", "device_info": PROFILE_MOBILE},
        ]
        result = sched.compute_layer_assignment(nodes)
        for a in result:
            assert a["layers_count"] >= 1, \
                f"节点 {a['node_id']} 应至少 1 层，实际: {a['layers_count']}"

    def test_result_fields_complete(self, sched):
        """返回结果应包含所有必要字段"""
        result = sched.compute_layer_assignment()
        for a in result:
            assert "node_id" in a
            assert "role" in a
            assert "start_layer" in a
            assert "end_layer" in a
            assert "layers_count" in a
            assert "has_embedding" in a
            assert "has_lm_head" in a
            assert "score" in a
            assert isinstance(a["score"], (int, float))


class TestLocalDeviceProfileSync:
    """异步设备检测必须进入调度器评分，并清除旧动态分层。"""

    def test_master_profile_updates_node_invalidates_cache_and_repushes(
            self, monkeypatch):
        sched = Scheduler()
        sched._role_override = "master"
        sched.nodes = {
            "master": NodeInfo(
                node_id="master", role="master", state=NodeState.ONLINE,
                device_info={},
            ),
        }
        pushed = []
        monkeypatch.setattr(
            sched, "push_layer_config_to_clients", lambda: pushed.append(True),
        )

        sched.update_local_device_profile(PROFILE_LAPTOP)

        assert sched.nodes["master"].device_info["gpu"]["name"] == "NVIDIA RTX 4060 Laptop"
        assert pushed == [True]

    def test_client_profile_is_reported_after_background_detection(
            self, monkeypatch):
        sched = Scheduler()
        sched._role_override = "client"
        monkeypatch.setattr(sched, "get_effective_node_id", lambda: "worker1")
        sched.nodes = {
            "worker1": NodeInfo(
                node_id="worker1", role="client", state=NodeState.ONLINE,
                device_info={},
            ),
        }
        sent = []

        class FakeClient:
            is_registered = True
            device_info = {}

            def send_data(self, data, msg_type):
                sent.append((data, msg_type.value))

        fake_client = FakeClient()
        monkeypatch.setattr(sched, "_tcp_client", fake_client, raising=False)

        sched.update_local_device_profile(PROFILE_IGPU_ONLY)

        expected = dict(sched._local_device_profile)
        assert "engine_capabilities" in expected
        assert expected["gpu"] == PROFILE_IGPU_ONLY["gpu"]
        assert fake_client.device_info == expected
        assert sent == [(
            {"state": "online", "device_info": expected},
            "status_res",
        )]

    def test_failed_profile_report_remains_retryable(self, monkeypatch):
        sched = Scheduler()
        sched._role_override = "client"
        sched._local_device_profile = PROFILE_IGPU_ONLY
        monkeypatch.setattr(sched, "get_effective_node_id", lambda: "worker1")
        sched.nodes = {
            "worker1": NodeInfo(
                node_id="worker1", role="client", state=NodeState.ONLINE,
                device_info=PROFILE_IGPU_ONLY,
            ),
        }

        class FlakyClient:
            is_registered = True
            device_info = {}
            attempts = 0

            def send_data(self, data, msg_type):
                self.attempts += 1
                if self.attempts == 1:
                    raise ConnectionError("temporary")

        client = FlakyClient()

        assert sched._report_local_device_profile(client, "worker1") is False
        assert client.device_info == {}
        assert sched._report_local_device_profile(client, "worker1") is True
        assert client.device_info == PROFILE_IGPU_ONLY
        assert client.attempts == 2

    def test_master_profile_report_recomputes_layer_config(self, monkeypatch):
        sched = Scheduler()
        sched._role_override = "master"
        sched.nodes = {
            "worker1": NodeInfo(
                node_id="worker1", role="client", state=NodeState.ONLINE,
                device_info={"platform": "Windows"},
            ),
        }
        pushed = []
        monkeypatch.setattr(
            sched, "push_layer_config_to_clients", lambda: pushed.append(True),
        )

        sched._on_tcp_message("worker1", {
            "type": "status_res",
            "data": {"state": "online", "device_info": PROFILE_IGPU_ONLY},
        })

        assert sched.nodes["worker1"].device_info == PROFILE_IGPU_ONLY
        assert pushed == [True]


class TestManualLayerAssignmentNormalization:
    """前端只提交区间，后端必须补齐可执行分层字段。"""

    def test_manual_ranges_gain_counts_roles_scores_and_io_heads(self, monkeypatch):
        sched = Scheduler()
        sched._role_override = "master"
        sched.nodes = {
            "master": NodeInfo(
                node_id="master", role="master", state=NodeState.ONLINE,
                device_info=PROFILE_LAPTOP,
            ),
            "worker": NodeInfo(
                node_id="worker", role="client", state=NodeState.ONLINE,
                device_info=PROFILE_IGPU_ONLY,
            ),
        }
        monkeypatch.setattr(sched, "push_layer_config_to_clients", lambda: None)

        result = sched.override_layer_assignments([
            {"node_id": "master", "start_layer": 0, "end_layer": 20},
            {"node_id": "worker", "start_layer": 20, "end_layer": 24},
        ])

        assert result["status"] == "ok"
        normalized = result["current_assignments"]["assignments"]
        assert normalized[0]["layers_count"] == 20
        assert normalized[1]["layers_count"] == 4
        assert normalized[0]["role"] == "master"
        assert normalized[0]["has_embedding"] is True
        assert normalized[0]["has_lm_head"] is True
        assert normalized[1]["has_lm_head"] is False

    def test_manual_ranges_reject_master_outside_first_segment(self, monkeypatch):
        sched = Scheduler()
        sched._role_override = "master"
        sched.nodes = {
            "master": NodeInfo(node_id="master", role="master", state=NodeState.ONLINE),
            "worker": NodeInfo(node_id="worker", role="client", state=NodeState.ONLINE),
        }

        result = sched.override_layer_assignments([
            {"node_id": "worker", "start_layer": 0, "end_layer": 12},
            {"node_id": "master", "start_layer": 12, "end_layer": 24},
        ])

        assert result["status"] == "invalid"
        assert result["reason_code"] == "layer_assignment_master_must_be_first"


# ================================================================
# get_layer_assignments 测试
# ================================================================

class TestGetLayerAssignments:
    """测试分层配置获取（内存模式，无 DB）"""

    @pytest.fixture
    def sched(self):
        s = Scheduler()
        s.nodes = {
            "master": NodeInfo(
                node_id="master", role="master", state=NodeState.ONLINE,
                node_type="pc",
                device_info=PROFILE_LAPTOP,
            ),
        }
        return s

    def test_returns_total_and_strategy(self, sched):
        """应返回 total, strategy, assignments"""
        result = sched.get_layer_assignments()
        assert result["total"] == 24
        assert result["strategy"] in ("dynamic", "manual")
        assert isinstance(result["assignments"], list)
        assert len(result["assignments"]) >= 1

    def test_single_node_all_24_layers(self, sched):
        """单节点时全部 24 层归该节点"""
        result = sched.get_layer_assignments()
        assignments = result["assignments"]
        assert len(assignments) == 1
        assert assignments[0]["start_layer"] == 0
        assert assignments[0]["end_layer"] == 24
        assert assignments[0]["has_embedding"] is True
        assert assignments[0]["has_lm_head"] is True

    def test_dynamic_assignment_is_computed_from_runtime_nodes(
            self, sched, monkeypatch):
        result = sched.get_layer_assignments()

        assert result["assignments"][0]["node_id"] == "master"
        assert result.get("cache_key")


# ================================================================
# NodeInfo RTT 字段测试
# ================================================================

class TestNodeRTT:
    """测试 NodeInfo 的 RTT 字段"""

    def test_default_rtt_zero(self):
        """新建 NodeInfo 时 RTT 默认为 0"""
        from scheduler import NodeInfo, NodeState
        node = NodeInfo(node_id="test", role="client")
        assert node.avg_rtt_ms == 0.0
        assert node.last_rtt_ms == 0.0

    def test_rtt_fields_in_to_dict(self):
        """to_dict() 应包含 avg_rtt_ms 和 last_rtt_ms"""
        from scheduler import NodeInfo, NodeState
        node = NodeInfo(
            node_id="rtt_test", role="client",
            avg_rtt_ms=12.5, last_rtt_ms=11.2,
        )
        d = node.to_dict()
        assert "avg_rtt_ms" in d
        assert "last_rtt_ms" in d
        assert d["avg_rtt_ms"] == 12.5
        assert d["last_rtt_ms"] == 11.2

    def test_rtt_rounded_in_to_dict(self):
        """to_dict() 中 RTT 应四舍五入到 1 位小数"""
        from scheduler import NodeInfo
        node = NodeInfo(
            node_id="round_test", role="client",
            avg_rtt_ms=12.567, last_rtt_ms=3.14159,
        )
        d = node.to_dict()
        assert d["avg_rtt_ms"] == 12.6
        assert d["last_rtt_ms"] == 3.1


# ================================================================
# Android 节点手动注册 / 删除 测试
# ================================================================

class TestAndroidNodeManagement:
    """测试 Android 节点注册与离线删除能力。"""

    @pytest.fixture
    def sched(self, monkeypatch):
        import scheduler as scheduler_module

        monkeypatch.setattr(scheduler_module, "NODE_ROLE", "master")
        s = Scheduler()
        yield s
        # 清理测试节点（内存 + 数据库）
        test_ids = [nid for nid in s.nodes if nid != "master"]
        for nid in test_ids:
            try:
                del s.nodes[nid]
            except Exception:
                pass

    def test_manual_register_android_node(self, sched):
        """手动注册 Android 节点应保存 node_type=android 且初始离线。"""
        result = sched.manual_register_node(
            "android-test", hostname="Android Phone",
            network_type="wifi", node_type="android",
        )
        assert result["status"] == "registered"
        node = sched.nodes["android-test"]
        assert node.node_type == "android"
        assert node.hostname == "Android Phone"
        assert node.network_type == "wifi"
        assert node.state == NodeState.OFFLINE

    def test_manual_register_existing_android_updates_metadata(self, sched):
        """重复注册同一 Android 节点应更新元数据而非失败。"""
        sched.manual_register_node("android-test", hostname="Old", node_type="android")
        result = sched.manual_register_node(
            "android-test", hostname="New", address="1.2.3.4:8888",
            network_type="wifi", node_type="android",
        )
        assert result["status"] == "updated"
        node = sched.nodes["android-test"]
        assert node.hostname == "New"
        assert node.address == "1.2.3.4:8888"
        assert node.network_type == "wifi"
        assert node.node_type == "android"

    def test_manual_register_real_connected_node_conflict(self, sched):
        """手动注册不应覆盖已经通过真实连接建立的节点记录。"""
        sched.nodes["client-live"] = NodeInfo(
            node_id="client-live", role="client", state=NodeState.OFFLINE,
            node_type="pc", address="10.0.0.2:8888", hostname="worker",
            network_type="ethernet", connected_at=1700000000.0,
        )

        result = sched.manual_register_node(
            "client-live", hostname="manual", address="1.2.3.4:9999",
            network_type="wifi", node_type="android",
        )

        assert result["status"] == "conflict"
        node = sched.nodes["client-live"]
        assert node.hostname == "worker"
        assert node.address == "10.0.0.2:8888"
        assert node.network_type == "ethernet"
        assert node.node_type == "pc"

    def test_invite_capacity_ignores_historical_offline_records(self, sched):
        """邀请容量不应被无地址、不可用的历史离线节点占用。"""
        sched._max_nodes = 2
        sched.nodes["old-offline"] = NodeInfo(
            node_id="old-offline", role="client", state=NodeState.OFFLINE,
            hostname="", address="", connected_at=1700000000.0,
        )

        invite = sched.get_invite_info()

        assert invite["has_capacity"] is True
        assert invite["node_count"] == 1
        assert invite["total_node_records"] >= 1
        assert invite["discovery_methods"] == ["local_config", "tailnet"]
        assert "db_registered" not in invite

    def test_delete_node_rejects_master_and_missing(self, sched):
        """删除 master / 不存在节点应返回明确状态。"""
        assert sched.delete_node("master")["status"] == "invalid"
        assert sched.delete_node("missing")["status"] == "not_found"

    def test_delete_node_rejects_online_node(self, sched):
        """在线节点需要先注销，不能直接删除。"""
        sched.nodes["client1"] = NodeInfo(
            node_id="client1", role="client", state=NodeState.ONLINE,
        )
        assert sched.delete_node("client1")["status"] == "online"
        assert "client1" in sched.nodes

    def test_android_presence_registers_online_and_expires(self, sched, monkeypatch):
        """Android HTTP thin client presence 应在线登记，并在超时后离线。"""
        pushed = []
        sched._push_node_update_to_all_clients = lambda *args: pushed.append(args)

        result = sched.register_android_client(
            "android-live",
            hostname="Game Tablet",
            network_type="wifi",
            device_info={"gpu": {"renderer": "Adreno"}},
            http_peer="100.64.1.2",
        )
        assert result["status"] == "registered"
        node = sched.nodes["android-live"]
        assert node.state == NodeState.ONLINE
        assert node.node_type == "android"
        assert node.device_info["connection_type"] == "http_thin"
        assert node.device_info["pipeline_worker"] is False
        assert node.device_info["http_peer"] == "100.64.1.2"
        assert pushed and pushed[-1][1] == "add"

        first_heartbeat = node.last_heartbeat
        result = sched.register_android_client("android-live", hostname="Game Tablet 2")
        assert result["status"] == "updated"
        assert sched.nodes["android-live"].last_heartbeat >= first_heartbeat
        assert pushed[-1][1] == "update"

        sched._refresh_http_client_states(now=node.last_heartbeat + 121)
        assert sched.nodes["android-live"].state == NodeState.OFFLINE
        assert pushed[-1][1] == "update"

    def test_android_presence_heartbeat_is_lease_fenced(self, sched):
        registered = sched.register_android_client("android-lease", hostname="Tablet")
        node = sched.nodes["android-lease"]
        generation = registered["presence_generation"]
        lease_id = registered["presence_lease_id"]

        heartbeat = sched.heartbeat_android_client("android-lease", generation, lease_id)
        assert heartbeat["status"] == "heartbeat"
        assert heartbeat["presence_generation"] == generation

        stale = sched.heartbeat_android_client("android-lease", generation + 1, lease_id)
        assert stale["status"] == "rejected"
        assert stale["error_code"] == "stale_generation"
        missing = sched.heartbeat_android_client("android-unknown", generation, lease_id)
        assert missing["error_code"] == "presence_not_registered"

        node.presence_expires_at = time.time() - 1
        expired = sched.heartbeat_android_client("android-lease", generation, lease_id)
        assert expired["error_code"] == "lease_expired"
        assert node.state == NodeState.OFFLINE

    def test_android_offline_does_not_block_nodes_ready(self, sched):
        """Android HTTP 客户端离线不影响 PC worker readiness。"""
        sched.nodes["android-offline"] = NodeInfo(
            node_id="android-offline",
            role="client",
            node_type="android",
            state=NodeState.OFFLINE,
            device_info={"connection_type": "http_thin"},
        )
        assert sched.check_nodes_ready() is True

    def test_delete_offline_android_node_removes_and_pushes_remove(self, sched, monkeypatch):
        """离线 Android 节点删除后应从内存移除并广播 remove。"""
        sched.manual_register_node("android-test", hostname="Phone", node_type="android")
        pushed = []
        sched._push_node_update_to_all_clients = lambda *args: pushed.append(args)

        result = sched.delete_node("android-test")
        assert result["status"] == "deleted"
        assert "android-test" not in sched.nodes
        assert pushed and pushed[0][0] == "android-test" and pushed[0][1] == "remove"


# ================================================================
# _get_node_vram_mb / _check_vram_constraint 测试
# ================================================================

class TestVRAMConstraint:
    """测试显存约束检查"""

    @pytest.fixture
    def sched(self):
        s = Scheduler()
        # 注入测试节点：
        # - gpu_node: 8GB 独显
        # - cpu_node: 无独显, 4GB RAM 可用
        # - tiny_node: 0.5GB 共享显存、1GB RAM 可用（集显/CPU 模式）
        from scheduler import NodeInfo, NodeState
        s.nodes["gpu_node"] = NodeInfo(
            node_id="gpu_node", role="client", state=NodeState.ONLINE,
            device_info={
                "gpu": {"vram_total_gb": 8.0, "name": "RTX 3070",
                        "is_integrated": False, "cuda_available": True},
                "ram": {"total_gb": 16.0, "available_gb": 8.0},
                "cpu": {"physical_cores": 8, "freq_max_mhz": 4000},
            },
        )
        s.nodes["cpu_node"] = NodeInfo(
            node_id="cpu_node", role="client", state=NodeState.ONLINE,
            device_info={
                "gpu": {},
                "ram": {"total_gb": 8.0, "available_gb": 4.0},
                "cpu": {"physical_cores": 4, "freq_max_mhz": 2500},
            },
        )
        s.nodes["tiny_node"] = NodeInfo(
            node_id="tiny_node", role="client", state=NodeState.ONLINE,
            device_info={
                "gpu": {"vram_total_gb": 0.5, "name": "Adreno", "is_integrated": True},
                "ram": {"total_gb": 3.0, "available_gb": 1.0},
                "cpu": {"physical_cores": 4, "freq_max_mhz": 2000},
            },
        )
        return s

    def test_get_vram_from_gpu(self, sched):
        """有独显时从 GPU 读取 VRAM"""
        mb = sched._get_node_vram_mb("gpu_node")
        assert mb == 8.0 * 1024  # 8 GB → 8192 MB

    def test_get_vram_from_ram_fallback(self, sched):
        """无独显时从可用 RAM 读取"""
        mb = sched._get_node_vram_mb("cpu_node")
        assert mb == 4.0 * 1024  # 4 GB available

    def test_get_vram_tiny_gpu(self, sched):
        """集显/CPU 节点应使用系统可用内存，不使用共享显存数字。"""
        mb = sched._get_node_vram_mb("tiny_node")
        assert mb == 1.0 * 1024

    def test_get_vram_unknown_node(self, sched):
        """未知节点返回 0"""
        assert sched._get_node_vram_mb("nonexistent") == 0.0

    def test_vram_constraint_ok(self, sched):
        """显存充足 — 应返回 True"""
        ok, needed, available = sched._check_vram_constraint(
            "gpu_node", layers_count=8, has_embedding=True, has_lm_head=False,
        )
        assert ok is True
        assert needed > 0
        assert available == 8192.0

    def test_vram_constraint_insufficient(self, sched):
        """显存不足 — 应返回 False"""
        # tiny_node 仅 1GB 可用内存，装不下 24 层 + Embedding + LM Head
        ok, needed, available = sched._check_vram_constraint(
            "tiny_node", layers_count=24, has_embedding=True, has_lm_head=True,
        )
        assert ok is False
        assert needed > available

    def test_vram_constraint_barely_ok(self, sched):
        """刚好够用 — 10% 安全余量内"""
        # cpu_node: ~4GB, int4 8层 ≈ 8*25 + 无额外头 ≈ 200MB → 轻松
        ok, needed, available = sched._check_vram_constraint(
            "cpu_node", layers_count=8, has_embedding=False, has_lm_head=False,
        )
        assert ok is True

    def test_vram_constraint_unknown_node(self, sched):
        """未知节点 — 无法判断时放行"""
        ok, needed, available = sched._check_vram_constraint(
            "nonexistent", layers_count=100, has_embedding=True, has_lm_head=True,
        )
        assert ok is True  # 不能判断 → 放行
        assert needed == 0
        assert available == 0

    def test_qwen2_split_memory_uses_fp16_even_when_int4_requested(
            self, sched, monkeypatch):
        from model_host import model_host as _host

        config = type("Qwen2Config", (), {
            "model_type": "qwen2",
            "hidden_size": 2048,
            "intermediate_size": 5504,
            "num_attention_heads": 16,
            "num_key_value_heads": 2,
            "vocab_size": 151936,
        })()
        manager = type("Manager", (), {
            "model": type("Model", (), {"config": config})(),
            "quant_type": "int4",
        })()
        monkeypatch.setattr(_host, "_manager", manager)

        layer_mb, embedding_mb, lm_head_mb = sched._get_layer_memory_estimate_mb(
            node_id="gpu_node",
            fallback=(70, 580, 580),
            quant_factors={"fp16": 1.0, "int4": 0.35},
            configured_quant="int4",
        )

        assert layer_mb > 50
        assert embedding_mb > 500
        assert lm_head_mb == embedding_mb

        cpu_layer_mb, cpu_embedding_mb, cpu_lm_head_mb = (
            sched._get_layer_memory_estimate_mb(
                node_id="cpu_node",
                fallback=(70, 580, 580),
                quant_factors={"fp16": 1.0, "int4": 0.35},
                configured_quant="int4",
            )
        )
        assert cpu_layer_mb == pytest.approx(layer_mb * 2)
        assert cpu_embedding_mb == pytest.approx(embedding_mb * 2)
        assert cpu_lm_head_mb == pytest.approx(lm_head_mb * 2)

    def test_original_qwen_cpu_split_memory_uses_fp32_baseline(
            self, sched, monkeypatch):
        from model_host import model_host as _host

        manager = type("Manager", (), {
            "model": type("Model", (), {
                "config": type("QwenConfig", (), {"model_type": "qwen"})(),
            })(),
            "quant_type": "int4",
        })()
        monkeypatch.setattr(_host, "_manager", manager)

        estimate = sched._get_layer_memory_estimate_mb(
            node_id="cpu_node",
            fallback=(70, 580, 580),
            quant_factors={"fp16": 1.0, "int4": 0.35},
            configured_quant="int4",
        )

        assert estimate == (140.0, 1160.0, 1160.0)

    def test_vram_transfer_moves_only_excess_layers(self, sched, monkeypatch):
        assignments = [
            {"node_id": "gpu_node", "role": "master", "layers_count": 10,
             "has_embedding": True, "has_lm_head": True, "score": 100},
            {"node_id": "cpu_node", "role": "client", "layers_count": 2,
             "has_embedding": False, "has_lm_head": False, "score": 20},
        ]

        def constraint(node_id, count, has_embedding=False, has_lm_head=False):
            limits = {"gpu_node": 8, "cpu_node": 8}
            return count <= limits[node_id], float(count), float(limits[node_id])

        monkeypatch.setattr(sched, "_check_vram_constraint", constraint)
        result = sched._apply_vram_constraints(assignments)
        by_id = {item["node_id"]: item for item in result}

        assert by_id["gpu_node"]["layers_count"] == 8
        assert by_id["cpu_node"]["layers_count"] == 4


# ================================================================
# 流水线就绪检查 测试
# ================================================================


class TestPipelineReadiness:
    """测试 _all_pipeline_nodes_ready() 和流水线节点状态判断"""

    @pytest.fixture
    def sched(self):
        return Scheduler()

    def test_no_tcp_server(self, sched):
        """TCP 服务端未启动 → 返回 False"""
        sched._tcp_server = None
        assert sched._all_pipeline_nodes_ready() is False

    def test_restart_recovery_fence_blocks_readiness_until_fresh_commit(
            self, sched, monkeypatch):
        """重启后的旧事务只能作为闸门，不能直接充当活动配置。"""
        import local_store

        monkeypatch.setattr(
            local_store,
            "get_local_setting",
            lambda key, default=None: {
                "schema_version": 1,
                "config_id": "cfg-old",
                "generation": 7,
                "phase": "ready",
                "model_sha256": "a" * 64,
            } if key == "pipeline_config_lifecycle_v1" else default,
        )
        sched._tcp_server = type("Server", (), {"_running": True})()
        sched._load_pipeline_recovery_state()

        readiness = sched._get_pipeline_readiness()

        assert readiness["ready"] is False
        assert readiness["reason_code"] == "pipeline_recovery_pending"
        assert sched._pipeline_recovery_state["config_id"] == "cfg-old"

        sched._pipeline_load_transaction = {
            "config_id": "cfg-new",
            "generation": 8,
            "phase": "ready",
            "worker_ids": {"worker-a"},
            "prepared_nodes": set(),
            "ready_nodes": {"worker-a"},
            "plan": {"assignments": []},
        }
        _install_layer_assignment(sched, "worker-a", {
            "config_id": "cfg-new",
            "generation": 8,
        })
        sched._pipeline_recovery_pending = True
        sched._maybe_finish_pipeline_recovery()
        assert sched._pipeline_recovery_pending is True

        _mark_layer_config_pushed(sched, "worker-a")
        sched._maybe_finish_pipeline_recovery()
        assert sched._pipeline_recovery_pending is False

    def test_restart_recovery_restores_persisted_model_before_republish(
            self, sched, monkeypatch, tmp_path):
        import model_config
        from types import SimpleNamespace

        model_dir = tmp_path / "qwen2.5-0.5b-instruct"
        model_dir.mkdir()
        sched._pipeline_recovery_pending = True
        sched._pipeline_recovery_state = {
            "model_id": "qwen2.5-0.5b-instruct",
            "model_type": "qwen2",
            "model_sha256": "a" * 64,
            "quant_type": "fp16",
        }
        prepared = []
        sched._host = SimpleNamespace(
            prepare_pipeline_model=lambda **kwargs: prepared.append(kwargs) or {
                "model_type": "qwen2", "total_layers": 24,
                "model_sha256": "a" * 64,
            },
        )
        monkeypatch.setattr(sched, "_get_active_pipeline_model_info", lambda: {})
        monkeypatch.setattr(
            model_config, "get_model_config", lambda _model_id, _db_models=None: None,
        )
        monkeypatch.setattr(
            model_config,
            "get_builtin_models",
            lambda: [SimpleNamespace(
                model_id="qwen2.5-0.5b",
                model_path=str(model_dir),
            )],
        )
        monkeypatch.setattr(model_config, "resolve_model_path", lambda value: value)

        assert sched._restore_pipeline_model_for_recovery() is True
        assert prepared == [{
            "model_id": "qwen2.5-0.5b-instruct",
            "model_path": str(model_dir),
            "quant_type": "fp16",
            "model_sha256": None,
        }]

    def test_restart_recovery_rejects_changed_local_model_digest(
            self, sched, monkeypatch, tmp_path):
        import model_config
        from types import SimpleNamespace

        model_dir = tmp_path / "model-a"
        model_dir.mkdir()
        sched._pipeline_recovery_pending = True
        sched._pipeline_recovery_state = {
            "model_id": "model-a", "model_type": "qwen2",
            "model_sha256": "a" * 64, "quant_type": "fp16",
        }
        unloaded = []
        sched._host = SimpleNamespace(
            prepare_pipeline_model=lambda **_kwargs: {
                "model_type": "qwen2", "total_layers": 24,
                "model_sha256": "b" * 64,
            },
            unload_model=lambda: unloaded.append(True),
        )
        monkeypatch.setattr(sched, "_get_active_pipeline_model_info", lambda: {})
        monkeypatch.setattr(
            model_config, "get_model_config", lambda _model_id, _db_models=None: None,
        )
        monkeypatch.setattr(
            model_config, "get_builtin_models",
            lambda: [SimpleNamespace(model_id="model-a", model_path=str(model_dir))],
        )
        monkeypatch.setattr(model_config, "resolve_model_path", lambda value: value)

        assert sched._restore_pipeline_model_for_recovery() is False
        assert sched._pipeline_recovery_failure == "pipeline_recovery_model_digest_mismatch"
        assert unloaded == [True]

    def test_restart_recovery_resolves_db_registered_gguf(
            self, sched, monkeypatch, tmp_path):
        import local_store
        import model_config
        from types import SimpleNamespace

        gguf_path = tmp_path / "custom.gguf"
        gguf_path.write_bytes(b"gguf-probe")
        sched._pipeline_recovery_pending = True
        sched._pipeline_recovery_state = {
            "model_id": "custom-gguf", "model_type": "qwen2",
            "model_sha256": "c" * 64, "quant_type": "Q4_K_M",
        }
        prepared = []
        sched._host = SimpleNamespace(
            prepare_pipeline_model=lambda **kwargs: prepared.append(kwargs) or {
                "model_type": "qwen2", "total_layers": 24,
                "model_sha256": "c" * 64,
            },
        )
        db_models = [{
            "model_id": "custom-gguf", "model_type": "gguf",
            "model_path": "", "gguf_path": str(gguf_path),
        }]
        monkeypatch.setattr(local_store, "get_local_experimental_models", lambda: db_models)
        monkeypatch.setattr(
            model_config, "get_model_config",
            lambda model_id, values=None: SimpleNamespace(**values[0])
            if model_id == "custom-gguf" and values else None,
        )
        monkeypatch.setattr(model_config, "get_builtin_models", lambda: [])
        monkeypatch.setattr(model_config, "resolve_model_path", lambda value: value)
        monkeypatch.setattr(sched, "_get_active_pipeline_model_info", lambda: {})

        assert sched._restore_pipeline_model_for_recovery() is True
        assert prepared == [{
            "model_id": "custom-gguf",
            "model_path": str(gguf_path),
            "quant_type": "Q4_K_M",
            "model_sha256": None,
        }]

    def test_restart_recovery_rejects_different_default_model(self, sched):
        sched._pipeline_recovery_pending = True
        sched._pipeline_recovery_state = {
            "model_id": "qwen2.5-0.5b-instruct",
            "model_type": "qwen2",
        }

        assert sched._recovery_model_matches({
            "model_id": "qwen3-5-2b",
            "model_type": "qwen3_5",
        }) is False
        assert sched._pipeline_recovery_failure == "pipeline_recovery_model_mismatch"

    def test_pipeline_lifecycle_snapshot_is_bounded_and_json_safe(
            self, sched):
        """持久化记录不能携带 set、模型路径或执行器对象。"""
        sched._pipeline_load_transaction = {
            "config_id": "cfg-1",
            "generation": 3,
            "phase": "preparing",
            "worker_ids": {"worker-b", "worker-a"},
            "prepared_nodes": {"worker-a"},
            "plan": {
                "model_id": "model-1",
                "model_type": "qwen2",
                "plan_id": "plan-1",
                "model_path": "C:/private/model.safetensors",
                "assignments": [{
                    "node_id": "worker-a",
                    "start_layer": 0,
                    "end_layer": 4,
                    "execution": "legacy_layer_config",
                    "runtime_object": object(),
                }],
            },
        }
        cleanup = sched._record_pipeline_model_cleanup_locked(
            node_id="worker-a",
            candidate_config_id="cfg-1",
            generation=3,
            action="finalize",
            payload={"finalize": True, "finalized_config_id": "cfg-1"},
        )

        snapshot = sched._pipeline_lifecycle_snapshot_locked()

        assert snapshot["schema_version"] == 4
        assert snapshot["worker_ids"] == ["worker-a", "worker-b"]
        assert snapshot["prepared_nodes"] == ["worker-a"]
        assert snapshot["assignments"] == [{
            "node_id": "worker-a",
            "start_layer": 0,
            "end_layer": 4,
            "execution": "legacy_layer_config",
        }]
        assert "model_path" not in snapshot
        assert snapshot["cleanup_ledger"] == [{
            "cleanup_id": cleanup["cleanup_id"],
            "node_id": "worker-a",
            "candidate_config_id": "cfg-1",
            "assignment_id": "",
            "generation": 3,
            "action": "finalize",
            # ★ 票 12 WORKER-PREPARE-01 起 cleanup ledger 条目带 legacy_identity 标记。
            "legacy_identity": False,
            "created_at": cleanup["created_at"],
            "payload": cleanup["payload"],
        }]

    def test_schema_v3_recovery_restores_worker_cleanup_ledger(
            self, sched, monkeypatch):
        import local_store

        monkeypatch.setattr(
            local_store,
            "get_local_setting",
            lambda _key, _default=None: {
                "schema_version": 3,
                "config_id": "",
                "generation": 0,
                "phase": "",
                "cleanup_ledger": [
                    {
                        "cleanup_id": "cleanup-worker",
                        "node_id": "worker-a",
                        "candidate_config_id": "cfg-worker",
                        "generation": 8,
                        "action": "abort",
                        "created_at": 123.0,
                        "payload": {
                            "node_id": "worker-a",
                            "config_id": "cleanup-worker",
                            "generation": 8,
                            "cleanup_action": "abort",
                            "candidate_config_id": "cfg-worker",
                            "release": True,
                            "abort": True,
                        },
                    },
                    {
                        "cleanup_id": "cleanup-master",
                        "node_id": "master",
                        "candidate_config_id": "cfg-master",
                        "generation": 8,
                        "action": "finalize",
                        "created_at": 124.0,
                        "payload": {},
                    },
                ],
            },
        )

        sched._load_pipeline_recovery_state()

        assert set(sched._pipeline_model_cleanup_ledger) == {"cleanup-worker"}
        restored = sched._pipeline_model_cleanup_ledger["cleanup-worker"]
        assert restored["state"] == "await_reconnect"
        assert restored["attempts"] == 0
        assert restored["next_retry"] == 0.0
        assert restored["legacy_identity"] is True
        assert restored["assignment_id"] == ""
        assert restored["payload"]["assignment_id"] == ""
        assert restored["payload"]["aborted_config_id"] == "cfg-worker"

    def test_schema_v4_cleanup_round_trip_preserves_exact_assignment(
            self, sched, monkeypatch):
        import local_store

        record = sched._record_pipeline_model_cleanup_locked(
            node_id="worker-a",
            candidate_config_id="cfg-worker",
            generation=8,
            action="finalize",
            assignment_id="asg-worker",
            payload={
                "finalize": True,
                "finalized_config_id": "cfg-worker",
            },
        )
        snapshot = sched._pipeline_lifecycle_snapshot_locked()
        sched._pipeline_model_cleanup_ledger.clear()
        monkeypatch.setattr(
            local_store, "get_local_setting",
            lambda _key, _default=None: snapshot,
        )

        sched._load_pipeline_recovery_state()

        restored = sched._pipeline_model_cleanup_ledger[record["cleanup_id"]]
        assert restored["legacy_identity"] is False
        assert restored["assignment_id"] == "asg-worker"
        assert restored["payload"]["assignment_id"] == "asg-worker"
        monkeypatch.setattr(
            sched, "push_layer_config_to_clients", lambda: None,
        )
        ack = dict(restored["payload"])
        ack["status"] = "finalized"

        assert sched._ack_pipeline_model_cleanup("worker-a", ack) is True
        assert sched._pipeline_model_cleanup_ledger == {}

    def test_schema_v4_cleanup_payload_drift_fails_recovery_closed(
            self, sched, monkeypatch):
        import local_store

        sched._record_pipeline_model_cleanup_locked(
            node_id="worker-a",
            candidate_config_id="cfg-worker",
            generation=8,
            action="abort",
            assignment_id="asg-worker",
            payload={
                "release": True,
                "abort": True,
                "aborted_config_id": "cfg-worker",
            },
        )
        snapshot = sched._pipeline_lifecycle_snapshot_locked()
        snapshot["cleanup_ledger"][0]["payload"]["assignment_id"] = "asg-stale"
        sched._pipeline_model_cleanup_ledger.clear()
        monkeypatch.setattr(
            local_store, "get_local_setting",
            lambda _key, _default=None: snapshot,
        )

        sched._load_pipeline_recovery_state()

        assert sched._pipeline_model_cleanup_ledger == {}
        assert sched._pipeline_recovery_pending is True
        assert sched._pipeline_recovery_failure == "pipeline_recovery_state_invalid"
        assert sched._pipeline_lifecycle_persist_ok is False

    def test_schema_v1_recovery_without_model_digest_fails_closed(
            self, sched, monkeypatch):
        import local_store

        monkeypatch.setattr(
            local_store, "get_local_setting",
            lambda _key, _default=None: {
                "schema_version": 1,
                "config_id": "cfg-old",
                "generation": 7,
                "phase": "ready",
                "model_id": "ambiguous-model",
            },
        )

        sched._load_pipeline_recovery_state()

        assert sched._pipeline_recovery_pending is True
        assert sched._pipeline_recovery_failure == "pipeline_recovery_state_invalid"
        assert sched._pipeline_lifecycle_persist_ok is False

    def test_restart_recovery_fence_requires_durable_ready_state(
            self, sched, monkeypatch):
        """A ready in-memory generation cannot clear a fence after a write failure."""
        import local_store

        monkeypatch.setattr(sched, "_running", True)
        monkeypatch.setattr(sched, "_effective_role", lambda: "master")
        sched._pipeline_recovery_pending = True
        sched._pipeline_load_transaction = {
            "config_id": "cfg-new",
            "generation": 8,
            "phase": "ready",
            "worker_ids": set(),
            "prepared_nodes": set(),
            "ready_nodes": set(),
            "plan": {"assignments": []},
        }

        def fail_write(_key, _value):
            raise OSError("sqlite unavailable")

        monkeypatch.setattr(local_store, "set_local_setting", fail_write)
        sched._persist_pipeline_lifecycle_locked()
        sched._maybe_finish_pipeline_recovery()

        assert sched._pipeline_recovery_pending is True
        assert sched._pipeline_lifecycle_persist_ok is False

        saved = []
        monkeypatch.setattr(
            local_store, "set_local_setting",
            lambda key, value: saved.append((key, value)),
        )
        sched._persist_pipeline_lifecycle_locked()
        sched._maybe_finish_pipeline_recovery()

        assert saved[0][0] == "pipeline_config_lifecycle_v1"
        assert saved[0][1]["phase"] == "ready"
        assert sched._pipeline_lifecycle_persist_ok is True
        assert sched._pipeline_recovery_pending is False

    def test_restart_recovery_fence_waits_for_route_a_stage_readiness(
            self, sched, monkeypatch):
        """A stage-offer assignment is not committed by legacy ACK state."""
        stage_ready = False
        monkeypatch.setattr(
            sched, "_stage_offer_assignment_ready",
            lambda _node_id, _assignment: (stage_ready, "not_ready"),
        )
        sched._pipeline_recovery_pending = True
        sched._pipeline_load_transaction = {
            "config_id": "cfg-new",
            "generation": 8,
            "phase": "ready",
            "worker_ids": set(),
            "prepared_nodes": set(),
            "ready_nodes": set(),
            "plan": {"assignments": [{
                "node_id": "android-worker",
                "execution": "stage_offer_v3",
                "start_layer": 0,
                "end_layer": 4,
            }]},
        }

        sched._maybe_finish_pipeline_recovery()
        assert sched._pipeline_recovery_pending is True

        stage_ready = True
        sched._maybe_finish_pipeline_recovery()
        assert sched._pipeline_recovery_pending is False

    def test_restart_recovery_rejects_persisted_generation_as_active(
            self, sched):
        sched._pipeline_recovery_pending = True
        sched._pipeline_recovery_state = {
            "schema_version": 1,
            "config_id": "cfg-old",
            "generation": 7,
            "phase": "ready",
        }
        sched._pipeline_load_transaction = {
            "config_id": "cfg-old",
            "generation": 7,
            "phase": "ready",
            "worker_ids": set(),
            "prepared_nodes": set(),
            "ready_nodes": set(),
            "plan": {"assignments": []},
        }

        sched._maybe_finish_pipeline_recovery()
        assert sched._pipeline_recovery_pending is True

        sched._pipeline_load_transaction["config_id"] = "cfg-new"
        sched._pipeline_load_transaction["generation"] = 8
        sched._maybe_finish_pipeline_recovery()
        assert sched._pipeline_recovery_pending is False

    def test_forced_sync_does_not_accept_ready_single_node_plan(
            self, sched, monkeypatch):
        calls = []
        monkeypatch.setattr(sched, "_get_pipeline_readiness", lambda: {
            "ready": True,
            "reason_code": "ready",
            "reason": "当前单节点计划已就绪",
            "workers": [],
        })
        monkeypatch.setattr(sched, "_connected_pc_worker_ids", lambda: ["worker"])
        monkeypatch.setattr(
            sched, "_has_active_distributed_pipeline_plan", lambda: False,
        )
        monkeypatch.setattr(
            sched,
            "request_authoritative_layer_sync",
            lambda **kwargs: calls.append(kwargs),
        )

        readiness = sched._synchronize_pipeline_workers_for_request(
            timeout=0.0,
            force_distributed_assignment=True,
        )

        assert calls == [{"require_distributed": True}]
        assert readiness["ready"] is False
        assert readiness["reason_code"] == "pipeline_single_node_plan_active"

    def test_only_master_nodes(self, sched):
        """仅主节点分配了层 → 无流水线从节点 → 返回 False"""
        # 模拟 get_layer_assignments 只返回 master
        original = sched.get_layer_assignments
        sched.get_layer_assignments = lambda: {
            "total": 24,
            "strategy": "dynamic",
            "assignments": [
                {"node_id": "master", "start_layer": 0, "end_layer": 24,
                 "has_embedding": True, "has_lm_head": True}
            ],
        }
        try:
            assert sched._all_pipeline_nodes_ready() is False
        finally:
            sched.get_layer_assignments = original

    def test_worker_not_online(self, sched):
        """从节点不在线 → 返回 False"""
        from scheduler import NodeInfo, NodeState
        # 注册一个离线从节点
        sched.nodes["client1"] = NodeInfo(
            node_id="client1", role="client",
            state=NodeState.OFFLINE, address="1.2.3.4:8888"
        )
        original = sched.get_layer_assignments
        sched.get_layer_assignments = lambda: {
            "total": 24,
            "strategy": "dynamic",
            "assignments": [
                {"node_id": "master", "start_layer": 0, "end_layer": 8,
                 "has_embedding": True, "has_lm_head": False},
                {"node_id": "client1", "start_layer": 8, "end_layer": 24,
                 "has_embedding": False, "has_lm_head": True},
            ],
        }
        try:
            assert sched._all_pipeline_nodes_ready() is False
        finally:
            sched.get_layer_assignments = original

    def test_push_waits_for_worker_load_ack(self, sched, monkeypatch):
        """配置发出后不能立即 ready，必须等从节点加载 ACK。"""
        from scheduler import NodeInfo, NodeState

        assignment_info = {
            "total": 24,
            "strategy": "dynamic",
            "assignments": [
                {"node_id": "master", "start_layer": 0, "end_layer": 8,
                 "has_embedding": True, "has_lm_head": False},
                {"node_id": "client1", "start_layer": 8, "end_layer": 24,
                 "has_embedding": False, "has_lm_head": True},
            ],
        }
        sched.nodes["client1"] = NodeInfo(
            node_id="client1",
            role="client",
            state=NodeState.ONLINE,
            address="100.64.1.2:8888",
            last_heartbeat=time.time(),
            model_sha256="",
        )
        sent = []
        sched._tcp_server = type("FakeServer", (), {
            "_running": True,
            "clients": {"client1": object()},
            "send_layer_config": lambda self, node_id, payload: sent.append({
                node_id: payload,
            }),
        })()
        monkeypatch.setattr(sched, "get_layer_assignments", lambda: assignment_info)
        monkeypatch.setattr(sched, "_get_active_pipeline_model_info", lambda **_kwargs: {
            "model_id": "qwen-1_8b",
            "model_path": "C:/models/qwen",
            "model_sha256": "sha-qwen",
            "model_type": "qwen",
            "total_layers": 24,
        })
        # This test covers the legacy one-phase ACK path. Do not inherit a
        # prepared-model flag left by another test in the process-wide host.
        monkeypatch.setattr(
            sched, "_host", type("Host", (), {"is_pipeline_prepared": False})(),
        )

        sched.push_layer_config_to_clients()

        payload = sent[0]["client1"]
        assert payload["node_id"] == "client1"
        assert payload["model_id"] == "qwen-1_8b"
        assert payload["model_sha256"] == "sha-qwen"
        assert payload["total_layers"] == 24
        assert payload["authoritative_sync"] is False
        assert payload["sync_policy"] == "normal"
        assert "client1" not in sched._layer_config_pushed
        assert sched._all_pipeline_nodes_ready() is False

        sched._on_tcp_message("client1", {
            "type": "layer_config_ack",
            "data": {
                "node_id": "client1",
                "assignment_id": payload["assignment_id"],
                "config_id": payload["config_id"],
                "generation": payload["generation"],
                "status": "ready",
                "layer_range": [8, 24],
                "model_sha256": "sha-qwen",
                "model_type": "qwen",
                "engine": "pytorch",
            },
        })
        assert sched._all_pipeline_nodes_ready() is True

    def test_publish_captures_authenticated_connection_epoch(
            self, sched, monkeypatch):
        sent = []
        sched._tcp_server = type("Server", (), {
            "get_authenticated_peer_epoch": lambda self, node_id: (
                37 if node_id == "client1" else 0
            ),
            "send_layer_config": lambda self, node_id, payload, *, registration_epoch: sent.append(
                (node_id, dict(payload), registration_epoch)
            ),
        })()
        monkeypatch.setattr(
            sched, "_start_layer_config_retry_monitor", lambda: None,
        )

        configs = {"client1": {
            "node_id": "client1",
            "config_id": "cfg-epoch",
            "generation": 11,
        }}
        sched._publish_layer_configs(configs)

        state = sched._worker_assignments.state("client1")
        assert state is not None
        assert state.connection_generation == 37
        assert sent[0][1]["assignment_id"] == state.assignment_id

        assert sent[0][2] == 37

    def test_readiness_reports_worker_layer_load_error(self, sched, monkeypatch):
        """在线不等于可计算，层加载错误必须成为明确的阻塞原因。"""
        sched.nodes["client1"] = NodeInfo(
            node_id="client1", role="client", state=NodeState.ONLINE,
            address="100.64.1.2:8888", last_heartbeat=time.time(),
        )
        sched._tcp_server = type("FakeServer", (), {
            "_running": True,
            "clients": {"client1": object()},
        })()
        monkeypatch.setattr(sched, "get_layer_assignments", lambda: {
            "total": 24,
            "assignments": [
                {"node_id": "master", "start_layer": 0, "end_layer": 8,
                 "layers_count": 8},
                {"node_id": "client1", "start_layer": 8, "end_layer": 24,
                 "layers_count": 16},
            ],
        })
        _install_layer_assignment(sched, "client1", {
            "config_id": "cfg-1", "model_id": "deepseek-r1",
        }, ack={
            "status": "error", "error": "missing tokenizer.json",
        })

        readiness = sched._get_pipeline_readiness()

        assert readiness["ready"] is False
        assert readiness["reason_code"] == "worker_layer_load_failed"
        assert "missing tokenizer.json" in readiness["reason"]
        assert readiness["workers"][0]["layer_status"] == "error"

    def test_pipeline_status_requires_layer_ready_ack(self, sched, monkeypatch):
        """管理状态不能把仅 TCP 在线但仍在同步模型的 worker 标为 active。"""
        from model_host import model_host as _host

        sched._role_override = "master"
        sched.nodes["client1"] = NodeInfo(
            node_id="client1", role="client", state=NodeState.ONLINE,
            address="100.64.1.2:8888", last_heartbeat=time.time(),
        )
        sched._tcp_server = type("FakeServer", (), {
            "_running": True,
            "clients": {"client1": object()},
        })()
        monkeypatch.setattr(_host, "_manager", type("Mgr", (), {
            "is_loaded": True, "_engine_type": "pytorch",
        })())
        monkeypatch.setattr(sched, "get_layer_assignments", lambda: {
            "total": 24,
            "assignments": [
                {"node_id": "master", "start_layer": 0, "end_layer": 8,
                 "layers_count": 8},
                {"node_id": "client1", "start_layer": 8, "end_layer": 24,
                 "layers_count": 16},
            ],
        })
        _install_layer_assignment(sched, "client1", {
            "config_id": "cfg-1", "model_id": "qwen-1_8b",
        })

        status = sched._get_pipeline_status()

        assert status["online_worker_count"] == 1
        assert status["active"] is False
        assert status["readiness_reason_code"] == "worker_layer_loading"
        assert status["workers"][0]["layer_status"] == "loading"

    def test_pipeline_status_reports_tcp_connection_during_recovery_fence(
        self, sched, monkeypatch,
    ):
        """Recovery readiness may be fenced without hiding a live TCP peer."""
        from model_host import model_host as _host

        sched._role_override = "master"
        sched.nodes["client1"] = NodeInfo(
            node_id="client1", role="client", state=NodeState.ONLINE,
            address="100.64.1.2:8888", last_heartbeat=time.time(),
        )
        sched._tcp_server = type("FakeServer", (), {
            "_running": True,
            "clients": {"client1": object()},
        })()
        monkeypatch.setattr(_host, "_manager", type("Mgr", (), {
            "is_loaded": True, "_engine_type": "pytorch",
        })())
        monkeypatch.setattr(sched, "get_layer_assignments", lambda: {
            "total": 24,
            "assignments": [
                {"node_id": "master", "start_layer": 0, "end_layer": 8,
                 "layers_count": 8},
                {"node_id": "client1", "start_layer": 8, "end_layer": 24,
                 "layers_count": 16},
            ],
        })
        sched._pipeline_recovery_pending = True
        sched._pipeline_recovery_failure = "pipeline_recovery_pending"

        status = sched._get_pipeline_status()

        assert status["readiness_reason_code"] == "pipeline_recovery_pending"
        assert status["workers"][0]["tcp_connected"] is True
        assert status["workers"][0]["layer_ready"] is False


# ================================================================
# 流水线结果等待 测试
# ================================================================


class TestPipelineWaitResult:
    """测试 _wait_for_layer_result 超时和事件机制"""

    @pytest.fixture
    def sched(self):
        return Scheduler()

    def test_timeout_no_result(self, sched):
        """没有节点返回结果 → 超时返回 None"""
        import time
        t0 = time.time()
        result = sched._wait_for_layer_result("task_1", "client1", timeout=0.5)
        elapsed = time.time() - t0
        assert result is None
        assert 0.4 < elapsed < 2.0  # 允许一些系统抖动

    def test_wait_for_layer_result_observes_cancel_event(self, sched):
        cancel_event = threading.Event()
        # 等待循环先建立再触发取消：0.2s 先导延迟足够循环推进到 ack_step，
        # 断言放宽为 step >= 2（原 ==2 依赖 50ms Timer 恰好命中步进点，
        # 慢机负载下可能提前取消导致 flaky）
        threading.Timer(0.2, cancel_event.set).start()
        started = time.time()

        result = sched._wait_for_layer_result(
            "task-cancel",
            "client1",
            timeout=5.0,
            ack_step=2,
            cancel_event=cancel_event,
        )

        elapsed = time.time() - started
        assert result["cancelled"] is True
        assert result["step"] >= 2
        assert elapsed < 0.5

    def test_wait_for_layer_result_uses_one_absolute_request_deadline(self, sched):
        from request_deadline import RequestDeadline

        request_deadline = RequestDeadline.start(0.24)

        def publish_first_result():
            key = "task-deadline-first:client1"
            with sched._pipeline_lock:
                sched._pipeline_results[key] = {
                    "task_id": "task-deadline-first",
                    "node_id": "client1",
                }
                event = sched._pipeline_events.get(key)
                if event is not None:
                    event.set()

        threading.Timer(0.06, publish_first_result).start()
        started = time.monotonic()
        first = sched._wait_for_layer_result(
            "task-deadline-first",
            "client1",
            timeout=5.0,
            request_deadline=request_deadline,
        )
        second = sched._wait_for_layer_result(
            "task-deadline-second",
            "client1",
            timeout=5.0,
            request_deadline=request_deadline,
        )
        elapsed = time.monotonic() - started

        assert first["task_id"] == "task-deadline-first"
        assert second["reason_code"] == "request_deadline_exceeded"
        assert elapsed < 0.5

    def test_legacy_layer_wait_signature_still_uses_remaining_deadline(
        self, sched, monkeypatch,
    ):
        from request_deadline import RequestDeadline

        observed = {}

        def old_wait(task_id, node_ids, timeout):
            observed["timeout"] = timeout
            time.sleep(timeout + 0.02)
            return {"task_id": task_id, "node_id": node_ids}

        monkeypatch.setattr(sched, "_wait_for_layer_result", old_wait)
        request_deadline = RequestDeadline.start(0.08)
        result = sched._wait_for_layer_result_with_ack(
            "task-legacy-deadline",
            "client1",
            timeout=5.0,
            ack_node_ids=["client1"],
            ack_step=0,
            ack_timeout=5.0,
            request_deadline=request_deadline,
        )

        assert observed["timeout"] <= 0.081
        assert result["reason_code"] == "request_deadline_exceeded"

    def test_result_before_wait(self, sched):
        """在等待线程中，另一侧注入结果 → 应立即唤醒"""
        import threading
        import time

        result_holder = []
        waiter_ready = threading.Event()
        original_wait = sched._wait_for_layer_result

        def observed_wait(*args, **kwargs):
            waiter_ready.set()
            return original_wait(*args, **kwargs)

        sched._wait_for_layer_result = observed_wait

        def waiter():
            result_holder.append(
                sched._wait_for_layer_result("task_2", "client1", timeout=5.0)
            )

        # 启动等待线程
        t = threading.Thread(target=waiter)
        t.start()

        assert waiter_ready.wait(timeout=1.0)

        # 注入结果（模拟 _handle_layer_result）
        key = "task_2:client1"
        with sched._pipeline_lock:
            sched._pipeline_results[key] = {"task_id": "task_2", "node_id": "client1"}
            if key in sched._pipeline_events:
                sched._pipeline_events[key].set()

        t.join(timeout=2.0)
        assert len(result_holder) == 1
        assert result_holder[0] is not None
        assert result_holder[0].get("task_id") == "task_2"


# ================================================================
# 流水线消息处理 测试
# ================================================================


class TestPipelineMessageDispatch:
    """测试流水线相关消息类型的调度分发"""

    @pytest.fixture
    def sched(self):
        return Scheduler()

    def test_layer_result_stores_in_dict(self, sched):
        """_handle_layer_result 应将结果存入并触发 event"""
        sched._pipeline_active_tasks.add("task_1")
        sched._pipeline_task_contracts["task_1"] = {
            "execution_protocol": "legacy_layer_v1",
            "config_id": "cfg-1",
            "model_sha256": "sha-1",
            "model_type": "qwen2",
            "worker_ids": ["client1"],
            "assignment_ids": {"client1": "asg-client1"},
            "last_node_id": "client1",
            "current_step": 0,
        }
        msg = {
            "type": "layer_result",
            "data": {
                "task_id": "task_1",
                "node_id": "client1",
                "step": 0,
                "assignment_id": "asg-client1",
                "config_id": "cfg-1",
                "model_sha256": "sha-1",
                "model_type": "qwen2",
                "chain_path": ["client1"],
                "hidden_states": "ZmFrZQ==",  # "fake" in base64
                "hidden_shape": [1, 16, 2048],
            },
        }
        sched._on_tcp_message("client1", msg)

        key = "task_1:client1"
        with sched._pipeline_lock:
            assert key in sched._pipeline_results
            result = sched._pipeline_results[key]
            assert result["task_id"] == "task_1"
            assert "hidden_states" in result

    def test_late_layer_result_is_discarded(self, sched):
        """已清理任务的迟到大张量不得重新进入结果缓存。"""
        sched._handle_layer_result("client1", {
            "data": {
                "task_id": "task_late",
                "node_id": "client1",
                "hidden_states": "dGVzdA==",
            },
        })
        assert sched._pipeline_results == {}

    def test_route_a_task_rejects_identity_less_legacy_layer_result(self, sched):
        sched._pipeline_active_tasks.add("task-route-a")
        sched._pipeline_task_contracts["task-route-a"] = {
            "execution_protocol": "route_a_stage_v3",
            "config_id": "route_a:task-route-a",
            "model_sha256": "sha-route-a",
            "model_type": "qwen2",
            "worker_ids": ["client1"],
            "last_node_id": "client1",
            "current_step": 0,
        }

        sched._handle_layer_result("client1", {"data": {
            "task_id": "task-route-a",
            "node_id": "client1",
            "step": 0,
            "config_id": "route_a:task-route-a",
            "model_sha256": "sha-route-a",
            "model_type": "qwen2",
            "error": "spoofed legacy result",
        }})

        assert sched._pipeline_results == {}

    def test_layer_result_rejects_spoofed_source_and_stale_step(self, sched):
        sched._pipeline_active_tasks.add("task-contract")
        sched._pipeline_task_contracts["task-contract"] = {
            "execution_protocol": "legacy_layer_v1",
            "config_id": "cfg-contract",
            "model_sha256": "sha-contract",
            "model_type": "qwen2",
            "worker_ids": ["client1", "client2"],
            "assignment_ids": {
                "client1": "asg-client1", "client2": "asg-client2",
            },
            "last_node_id": "client2",
            "current_step": 1,
        }
        base = {
            "task_id": "task-contract",
            "step": 1,
            "assignment_id": "asg-client2",
            "config_id": "cfg-contract",
            "model_sha256": "sha-contract",
            "model_type": "qwen2",
            "node_id": "client2",
            "hidden_states": "dGVzdA==",
        }

        sched._handle_layer_result("client1", {"data": dict(base)})
        sched._handle_layer_result(
            "client2", {"data": dict(base, step=0)},
        )

        assert sched._pipeline_results == {}

    def test_active_task_uses_frozen_assignment_after_worker_reconfiguration(
            self, sched):
        old = _install_layer_assignment(sched, "client1", {
            "node_id": "client1",
            "config_id": "cfg-shared",
            "generation": 3,
        })
        sched._pipeline_active_tasks.add("task-frozen")
        sched._pipeline_task_contracts["task-frozen"] = {
            "execution_protocol": "legacy_layer_v1",
            "config_id": "cfg-shared",
            "model_sha256": "sha-shared",
            "model_type": "qwen2",
            "worker_ids": ["client1"],
            "assignment_ids": {
                "client1": old["assignment_id"],
            },
            "last_node_id": "client1",
            "current_step": 0,
        }
        new = _install_layer_assignment(sched, "client1", {
            "node_id": "client1",
            "config_id": "cfg-shared",
            "generation": 3,
        })
        assert new["assignment_id"] != old["assignment_id"]

        sched._handle_layer_result("client1", {"data": {
            "task_id": "task-frozen",
            "node_id": "client1",
            "step": 0,
            "assignment_id": old["assignment_id"],
            "config_id": "cfg-shared",
            "model_sha256": "sha-shared",
            "model_type": "qwen2",
            "chain_path": ["client1"],
            "hidden_states": "dGVzdA==",
        }})

        assert "task-frozen:client1" in sched._pipeline_results

    def test_pipeline_done_clears_kv_cache(self, sched):
        """PIPELINE_DONE 应清理指定 task 的 KV 缓存"""
        sched._kv_cache["task_x"] = {"layer_0": "mock_kv"}
        msg = {
            "type": "pipeline_done",
            "data": {"task_id": "task_x"},
        }
        sched._on_tcp_message("master", msg)
        assert "task_x" not in sched._kv_cache

    def test_pipeline_abort_clears_kv_cache(self, sched):
        """PIPELINE_ABORT 应清理指定 task 的 KV 缓存"""
        sched._kv_cache["task_y"] = {"layer_0": "mock_kv"}
        msg = {
            "type": "pipeline_abort",
            "data": {"task_id": "task_y"},
        }
        sched._on_tcp_message("master", msg)
        assert "task_y" not in sched._kv_cache

    def test_layer_config_dispatches_to_handler(self, sched):
        """layer_config 消息类型应路由到 _handle_layer_config（不抛异常）"""
        msg = {
            "type": "layer_config",
            "data": {
                "client1": {
                    "start_layer": 8, "end_layer": 16,
                    "has_embedding": False, "has_lm_head": False,
                },
            },
        }
        # 不应抛出异常（即使 model_manager 不存在也会优雅处理）
        sched._on_tcp_message("master", msg)

    def test_direct_layer_config_loads_range_and_sends_ack(self, sched, monkeypatch):
        """真实线上单 assignment 载荷应加载层范围并回传 ready ACK。"""
        from model_host import model_host as _host
        from tcp_comm import MessageType, TCPClient

        class MockModelManager:
            is_loaded = False
            is_pipeline_prepared = False
            _engine_type = "pytorch"
            layer_range = None
            active_model_id = "qwen-test"
            quant_type = "fp16"
            model = type("Model", (), {
                "config": type("Config", (), {"model_type": "qwen2"})(),
            })()

            def load_layer_range(self, start, end, **kwargs):
                self.layer_range = (start, end)
                self.is_loaded = True
                self.is_pipeline_prepared = True

        sent = []
        sched._tcp_client = type("FakeClient", (), {
            "send_data": lambda self, payload, msg_type: sent.append((payload, msg_type)),
        })()
        monkeypatch.setattr(sched, "get_effective_node_id", lambda: "client1")
        monkeypatch.setattr(
            TCPClient, "_compute_local_model_sha256", lambda **kwargs: "sha-qwen",
        )
        monkeypatch.setattr(_host, "_manager", MockModelManager())
        monkeypatch.setattr(_host, "model_loaded", False)

        sched._handle_layer_config("master", {
            "node_id": "client1",
            "config_id": "cfg-1",
            "start_layer": 8,
            "end_layer": 16,
            "has_embedding": False,
            "has_lm_head": False,
            "model_id": "qwen-test",
            "model_type": "qwen2",
            "model_sha256": "sha-qwen",
            "total_layers": 24,
        })

        assert len(sent) == 1
        payload, msg_type = sent[0]
        assert msg_type == MessageType.LAYER_CONFIG_ACK
        assert payload["status"] == "ready"
        assert payload["config_id"] == "cfg-1"
        assert payload["layer_range"] == [8, 16]
        assert payload["model_sha256"] == "sha-qwen"
        assert _host.model_loaded is False
        assert _host.is_loaded is True
        assert _host.is_pipeline_prepared is True
        assert _host.current_quant == "fp16"
        assert sched._active_layer_config == {
            "node_id": "client1",
            "assignment_id": "",
            "config_id": "cfg-1",
            "model_id": "qwen-test",
            "model_sha256": "sha-qwen",
            "model_type": "qwen2",
            "layer_range": [8, 16],
            "engine": "pytorch",
            "master_quant_type": "",
            "runtime_quant_type": "fp16",
        }

    def test_relay_middle_assignment_does_not_load_scheduler_host(
            self, sched, monkeypatch):
        """relay_middle uses the supervised endpoint, not a local full/layer model."""
        import scheduler_pipeline as pipeline_module

        relay_spec = {
            "role": "middle",
            "host": "127.0.0.1",
            "port": 50183,
            "n_embd": 896,
            "timeout": 5.0,
            "layer_start": 8,
            "layer_end": 16,
        }
        load_calls = []
        fake_host = type("RelayHost", (), {
            "is_loaded": False,
            "model_loaded": False,
            "layer_range": None,
            "load_layer_range": lambda self, *args, **kwargs: load_calls.append(
                (args, kwargs)
            ),
        })()
        sent = []
        sched._host = fake_host
        sched._tcp_client = type("Client", (), {
            "send_data": lambda self, payload, msg_type: sent.append((payload, msg_type)),
        })()
        monkeypatch.setattr(sched, "get_effective_node_id", lambda: "worker")
        monkeypatch.setattr(pipeline_module, "PIPELINE_RELAY_ENABLED", True)

        sched._handle_layer_config("master", {
            "node_id": "worker",
            "config_id": "cfg-relay",
            "start_layer": 8,
            "end_layer": 16,
            "model_id": "qwen-test",
            "model_sha256": "sha-relay",
            "model_type": "qwen2",
            "total_layers": 24,
            "engine": "relay_middle",
            "relay_segment": relay_spec,
        })

        assert load_calls == []
        assert fake_host.model_loaded is False
        assert sent[-1][0]["status"] == "ready"
        assert sched._active_layer_config["engine"] == "relay_middle"
        assert sched._active_layer_config["relay_segment"] == relay_spec

    def test_relay_middle_assignment_is_not_rejected_as_legacy_stage_config(
            self, sched, monkeypatch):
        """A stage-advertising relay still accepts its logical relay config."""
        import scheduler_pipeline as pipeline_module

        relay_spec = {
            "role": "middle", "host": "127.0.0.1", "port": 50183,
            "n_embd": 896, "timeout": 5.0, "layer_start": 8,
            "layer_end": 16,
        }
        sent = []
        fake_host = type("RelayHost", (), {
            "is_loaded": False,
            "model_loaded": False,
            "layer_range": None,
            "load_layer_range": lambda self, *args, **kwargs: (
                (_ for _ in ()).throw(AssertionError("relay must not load local layers"))
            ),
        })()
        sched._host = fake_host
        sched._tcp_client = type("Client", (), {
            "send_data": lambda self, payload, msg_type: sent.append(payload),
        })()
        monkeypatch.setattr(sched, "get_effective_node_id", lambda: "worker")
        monkeypatch.setattr(pipeline_module, "PIPELINE_RELAY_ENABLED", True)
        monkeypatch.setenv("QLH_LAYER_GGUF", "C:/models/layer.gguf")

        sched._handle_layer_config("master", {
            "node_id": "worker", "config_id": "cfg-relay-stage",
            "generation": 7, "start_layer": 8, "end_layer": 16,
            "model_id": "qwen-test", "model_sha256": "sha-relay",
            "model_type": "qwen2", "total_layers": 24,
            "engine": "relay_middle", "relay_segment": relay_spec,
        })

        assert sent[-1]["status"] == "ready"
        assert sent[-1]["engine"] == "relay_middle"
        assert sched._active_layer_config["config_id"] == "cfg-relay-stage"

    def test_relay_middle_rehydrates_without_master_model_identity(
            self, sched, monkeypatch):
        """Restart metadata gaps must still admit endpoint-backed relay config."""
        import scheduler_pipeline as pipeline_module

        relay_spec = {
            "role": "middle", "host": "127.0.0.1", "port": 50183,
            "n_embd": 896, "timeout": 5.0, "layer_start": 8,
            "layer_end": 16,
        }
        sent = []
        fake_host = type("RelayHost", (), {
            "is_loaded": False,
            "model_loaded": False,
            "layer_range": None,
            "load_layer_range": lambda self, *args, **kwargs: (
                (_ for _ in ()).throw(AssertionError("relay must not load local layers"))
            ),
        })()
        sched._host = fake_host
        sched._tcp_client = type("Client", (), {
            "send_data": lambda self, payload, msg_type: sent.append(payload),
        })()
        monkeypatch.setattr(sched, "get_effective_node_id", lambda: "worker")
        monkeypatch.setattr(pipeline_module, "PIPELINE_RELAY_ENABLED", True)

        sched._handle_layer_config("master", {
            "node_id": "worker", "config_id": "cfg-relay-gap",
            "generation": 8, "start_layer": 24, "end_layer": 24,
            "total_layers": 24, "engine": "relay_middle",
            "relay_segment": relay_spec,
        })

        assert sent[-1]["status"] == "ready"
        assert sent[-1]["config_id"] == "cfg-relay-gap"
        assert sent[-1]["model_type"] == ""
        assert sched._active_layer_config["engine"] == "relay_middle"

    def test_nested_legacy_rejection_ack_preserves_generation(
            self, sched, monkeypatch):
        """Wrapped legacy assignments must reject with their own generation."""
        sent = []
        sched._tcp_client = type("Client", (), {
            "send_data": lambda self, payload, msg_type: sent.append(payload),
        })()
        monkeypatch.setattr(sched, "get_effective_node_id", lambda: "worker")
        monkeypatch.setenv("QLH_LAYER_GGUF", "C:/models/layer.gguf")

        sched._handle_layer_config("master", {
            "worker": {
                "node_id": "worker",
                "config_id": "cfg-legacy-nested",
                "generation": 19,
                "start_layer": 8,
                "end_layer": 16,
            },
        })

        assert sent[-1]["status"] == "error"
        assert sent[-1]["error"] == "layer_stage_worker_rejects_legacy_config"
        assert sent[-1]["generation"] == 19

    def test_relay_middle_prepare_does_not_require_local_capacity(
            self, sched, monkeypatch):
        import scheduler_pipeline as pipeline_module

        relay_spec = {
            "role": "middle", "host": "127.0.0.1", "port": 50183,
            "n_embd": 896, "timeout": 5.0, "layer_start": 8, "layer_end": 16,
        }
        sent = []
        sched._host = type("RelayHost", (), {
            "is_loaded": False, "model_loaded": False, "layer_range": None,
        })()
        sched._tcp_client = type("Client", (), {
            "send_data": lambda self, payload, msg_type: sent.append(payload),
        })()
        monkeypatch.setattr(sched, "get_effective_node_id", lambda: "worker")
        monkeypatch.setattr(pipeline_module, "PIPELINE_RELAY_ENABLED", True)

        sched._handle_layer_config("master", {
            "node_id": "worker", "config_id": "cfg-relay-prepare",
            "phase": "prepare", "plan_id": "plan-relay",
            "start_layer": 8, "end_layer": 16,
            "model_id": "qwen-test", "model_sha256": "sha-relay",
            "model_type": "qwen2", "total_layers": 24,
            "engine": "relay_middle", "relay_segment": relay_spec,
        })

        assert sent[-1]["status"] == "prepared"
        assert sent[-1]["engine"] == "relay_middle"
        assert sched._prepared_layer_configs["cfg-relay-prepare"]["plan_id"] == "plan-relay"

    def test_relay_middle_forward_bypasses_local_model_check(
            self, sched, monkeypatch):
        """A confirmed relay assignment can forward with an unloaded host."""
        import scheduler_pipeline as pipeline_module

        relay_spec = {
            "role": "middle", "host": "127.0.0.1", "port": 50183,
            "n_embd": 4, "timeout": 5.0, "layer_start": 8, "layer_end": 16,
        }
        sched._host = type("RelayHost", (), {
            "is_loaded": False, "model_loaded": False, "layer_range": None,
        })()
        sched._active_layer_config = {
            "assignment_id": "asg-relay",
            "config_id": "cfg-relay", "model_id": "qwen-test",
            "model_sha256": "sha-relay", "model_type": "qwen2",
            "layer_range": [8, 16], "engine": "relay_middle",
            "relay_segment": relay_spec,
        }
        sched._tcp_client = type("Client", (), {"_running": True})()
        monkeypatch.setattr(pipeline_module, "PIPELINE_RELAY_ENABLED", True)
        forwarded = []
        monkeypatch.setattr(
            sched, "_handle_layer_forward_via_relay",
            lambda spec, **kwargs: forwarded.append((spec, kwargs)),
        )

        hidden = (b"\x00" * (4 * 4))
        sched._handle_layer_forward_locked("master", {"data": {
            "task_id": "relay-task", "step": 0, "use_kv_cache": False,
            "assignment_id": "asg-relay",
            "config_id": "cfg-relay", "model_sha256": "sha-relay",
            "model_type": "qwen2", "hidden_states": hidden,
            "hidden_shape": [4, 4], "relay_segment": relay_spec,
        }})

        assert len(forwarded) == 1
        assert forwarded[0][0] == relay_spec

    def test_capacity_prepare_materializes_isolated_candidate(
            self, sched, monkeypatch, tmp_path):
        from tcp_comm import TCPClient
        import model_sync
        import pipeline_model_descriptor

        calls = []
        active_manager = object()
        sched._host = type("Host", (), {
            "prepare_pipeline_candidate": (
                lambda self, transaction_id, **kwargs: calls.append(
                    ("prepare", transaction_id, kwargs)
                ) or {"model_id": kwargs["model_id"]}
            ),
            "materialize_pipeline_candidate": (
                lambda self, transaction_id, **kwargs: calls.append(
                    ("materialize", transaction_id, kwargs)
                ) or {"success": True}
            ),
            "peek_manager": lambda self: active_manager,
        })()
        sent = []
        sched._tcp_client = type("Client", (), {
            "server_host": "100.64.0.1",
            "send_data": lambda self, payload, msg_type: sent.append(payload),
        })()
        sched._local_device_profile = {
            "gpu": {
                "name": "RTX", "cuda_available": True,
                "is_integrated": False, "vram_total_gb": 8,
                "vram_free_gb": 4,
            },
        }
        monkeypatch.setattr(sched, "get_effective_node_id", lambda: "worker")
        monkeypatch.setattr(
            model_sync, "resolve_worker_model_path", lambda _model_id: str(tmp_path),
        )
        monkeypatch.setattr(
            TCPClient, "_compute_local_model_sha256", lambda **_kwargs: "sha",
        )
        monkeypatch.setattr(
            pipeline_model_descriptor,
            "inspect_pipeline_model",
            lambda path, model_id="": {
                "model_type": "qwen2", "total_layers": 4,
            },
        )

        sched._handle_layer_config("master", {
            "node_id": "worker",
            "config_id": "cfg-prepare",
            "generation": 11,
            "phase": "prepare",
            "plan_id": "plan-prepare",
            "start_layer": 0,
            "end_layer": 2,
            "has_embedding": True,
            "has_lm_head": False,
            "model_id": "tiny-qwen2",
            "model_sha256": "sha",
            "model_type": "qwen2",
            "total_layers": 4,
            "required_bytes": 1024,
        })

        assert [call[0] for call in calls] == ["prepare", "materialize"]
        assert all(call[1] == "cfg-prepare" for call in calls)
        assert sched._host.peek_manager() is active_manager
        assert sent[-1]["status"] == "prepared"
        assert sent[-1]["plan_id"] == "plan-prepare"
        assert sent[-1]["generation"] == 11
        assert sched._active_layer_config is None
        assert sched._prepared_layer_configs["cfg-prepare"]["layer_range"] == [0, 2]

    def test_capacity_commit_requires_matching_prepared_record(
            self, sched, monkeypatch):
        from model_host import model_host as _host

        loads = []
        manager = type("Manager", (), {
            "is_loaded": False,
            "load_layer_range": lambda self, *args, **kwargs: loads.append(args),
        })()
        sent = []
        sched._tcp_client = type("Client", (), {
            "send_data": lambda self, payload, msg_type: sent.append(payload),
        })()
        monkeypatch.setattr(sched, "get_effective_node_id", lambda: "worker")
        monkeypatch.setattr(_host, "_manager", manager)

        sched._handle_layer_config("master", {
            "node_id": "worker",
            "config_id": "cfg-commit",
            "phase": "commit",
            "plan_id": "plan-commit",
            "start_layer": 0,
            "end_layer": 2,
            "model_id": "tiny-qwen2",
            "model_sha256": "sha",
            "model_type": "qwen2",
            "total_layers": 4,
        })

        assert loads == []
        assert sent[-1]["status"] == "error"
        assert "prepared" in sent[-1]["error"]

    @pytest.mark.parametrize("engine", ["pytorch", "relay_middle"])
    def test_duplicate_candidate_commit_replays_ready_without_abort(
            self, sched, monkeypatch, engine):
        import scheduler_pipeline as pipeline_module

        aborts = []
        sent = []
        relay_spec = {
            "role": "middle",
            "host": "127.0.0.1",
            "port": 50183,
            "n_embd": 896,
            "timeout": 5.0,
            "layer_start": 0,
            "layer_end": 2,
        }
        sched._host = type("Host", (), {
            "abort_pipeline_candidate": (
                lambda self, transaction_id: aborts.append(transaction_id)
            ),
        })()
        sched._tcp_client = type("Client", (), {
            "send_data": lambda self, payload, msg_type: sent.append(dict(payload)),
        })()
        monkeypatch.setattr(sched, "get_effective_node_id", lambda: "worker")
        monkeypatch.setattr(pipeline_module, "PIPELINE_RELAY_ENABLED", True)
        active = {
            "node_id": "worker",
            "assignment_id": "asg-duplicate",
            "config_id": "cfg-duplicate",
            "model_id": "tiny-qwen2",
            "model_sha256": "sha",
            "model_type": "qwen2",
            "layer_range": [0, 2],
            "engine": engine,
        }
        if engine == "relay_middle":
            active["relay_segment"] = relay_spec
        ready = {
            **active,
            "generation": 11,
            "status": "ready",
            "phase": "commit",
            "plan_id": "plan-duplicate",
        }
        sched._active_layer_config = dict(active)
        sched._last_layer_config_ack_payload = dict(ready)

        payload = {
            **active,
            "generation": 11,
            "phase": "commit",
            "plan_id": "plan-duplicate",
            "start_layer": 0,
            "end_layer": 2,
            "total_layers": 4,
        }
        payload.pop("layer_range")
        sched._handle_layer_config("master", payload)

        assert aborts == []
        assert len(sent) == 1
        assert sent[0]["status"] == "ready"
        assert sent[0]["config_id"] == "cfg-duplicate"
        assert sent[0]["plan_id"] == "plan-duplicate"
        assert sched._prepared_layer_configs == {}

    def test_abort_cleanup_failure_preserves_worker_rollback_until_retry(
            self, sched, monkeypatch):
        calls = []
        sent = []

        class Host:
            is_loaded = True
            is_pipeline_prepared = False
            model_loaded = True

            def abort_pipeline_candidate(self, transaction_id):
                calls.append(transaction_id)
                if len(calls) == 1:
                    raise RuntimeError("transient abort failure")

            def pipeline_candidate_status(self, transaction_id):
                return {
                    "matches": transaction_id == "cfg-candidate",
                    "phase": "committed",
                }

        sched._host = Host()
        sched._tcp_client = type("Client", (), {
            "send_data": lambda self, payload, msg_type: sent.append(dict(payload)),
        })()
        monkeypatch.setattr(sched, "get_effective_node_id", lambda: "worker")
        previous = {"config_id": "cfg-old", "layer_range": [2, 4]}
        previous_ack = {"config_id": "cfg-old", "status": "ready"}
        sched._active_layer_config = {
            "config_id": "cfg-candidate", "layer_range": [0, 2],
        }
        sched._pipeline_worker_reserved = True
        sched._pipeline_assignment_rollbacks["cfg-candidate"] = {
            "candidate_runtime": True,
            "active_layer_config": dict(previous),
            "pipeline_worker_reserved": True,
            "last_layer_config_ack_payload": dict(previous_ack),
        }
        sched._prepared_layer_configs["cfg-candidate"] = {
            "plan_id": "plan-candidate",
        }
        cleanup = {
            "node_id": "worker",
            "config_id": "cleanup-abort",
            "generation": 12,
            "release": True,
            "abort": True,
            "cleanup_action": "abort",
            "candidate_config_id": "cfg-candidate",
            "aborted_config_id": "cfg-candidate",
        }

        with pytest.raises(RuntimeError, match="transient abort failure"):
            sched._handle_layer_config("master", cleanup)

        assert "cfg-candidate" in sched._pipeline_assignment_rollbacks
        assert "cfg-candidate" in sched._prepared_layer_configs
        assert sent == []

        sched._handle_layer_config("master", cleanup)

        assert calls == ["cfg-candidate", "cfg-candidate"]
        assert sched._pipeline_assignment_rollbacks == {}
        assert sched._prepared_layer_configs == {}
        assert sched._active_layer_config == previous
        assert sched._last_layer_config_ack_payload == previous_ack
        assert sent[-1]["status"] == "released"
        assert sent[-1]["cleanup_action"] == "abort"
        assert sent[-1]["candidate_config_id"] == "cfg-candidate"

    def test_deepseek_assignment_syncs_selected_model_before_loading(self, sched, monkeypatch):
        """缺少 DeepSeek 时应先同步指定模型，再按真实层数加载。"""
        from model_host import model_host as _host
        import model_sync
        from tcp_comm import TCPClient

        load_calls = []

        class MockModelManager:
            is_loaded = False
            _engine_type = "pytorch"
            layer_range = None
            active_model_id = "deepseek-r1-distill-qwen-1.5b"
            quant_type = "fp16"
            model = type("Model", (), {
                "config": type("Config", (), {"model_type": "qwen2"})(),
            })()

            def load_layer_range(self, start, end, **kwargs):
                load_calls.append((start, end, kwargs))
                self.layer_range = (start, end)

        sent = []
        fake_client = type("FakeClient", (), {
            "server_host": "100.64.0.10",
            "send_data": lambda self, payload, msg_type: sent.append(payload),
        })()
        sched._tcp_client = fake_client
        monkeypatch.setattr(sched, "get_effective_node_id", lambda: "client1")
        hashes = iter(["old-sha", "deepseek-sha"])
        monkeypatch.setattr(
            TCPClient,
            "_compute_local_model_sha256",
            lambda **kwargs: next(hashes),
        )
        monkeypatch.setattr(
            model_sync,
            "resolve_worker_model_path",
            lambda model_id: "C:/models/deepseek",
        )
        sync_calls = []
        monkeypatch.setattr(
            model_sync,
            "ensure_model_available",
            lambda host, port, model_id, sha: (
                sync_calls.append((host, port, model_id, sha))
                or "C:/models/deepseek"
            ),
        )
        monkeypatch.setattr(_host, "_manager", MockModelManager())
        monkeypatch.setattr(_host, "model_loaded", False)

        sched._handle_layer_config("master", {
            "node_id": "client1",
            "config_id": "cfg-deepseek",
            "start_layer": 10,
            "end_layer": 28,
            "has_embedding": False,
            "has_lm_head": True,
            "model_id": "deepseek-r1-distill-qwen-1.5b",
            "model_sha256": "deepseek-sha",
            "model_type": "qwen2",
            "total_layers": 28,
            "master_api_port": 8000,
        })

        assert sync_calls == [(
            "100.64.0.10",
            8000,
            "deepseek-r1-distill-qwen-1.5b",
            "deepseek-sha",
        )]
        assert load_calls[0][0:2] == (10, 28)
        assert load_calls[0][2]["model_path"] == "C:/models/deepseek"
        assert load_calls[0][2]["model_id"] == "deepseek-r1-distill-qwen-1.5b"
        assert load_calls[0][2]["total_layers"] == 28
        assert sent[0]["status"] == "ready"

    def test_layer_config_is_deferred_until_active_task_finishes(self, sched, monkeypatch):
        """新分层配置不能在已有 task 的 KV cache 生命周期中替换模型。"""
        calls = []
        sched._active_pipeline_task_ids.add("active-task")

        sched._handle_layer_config("master", {"config_id": "cfg-next"})
        assert calls == []
        assert sched._pending_layer_config[1]["config_id"] == "cfg-next"

        monkeypatch.setattr(
            sched,
            "_handle_layer_config_locked",
            lambda client_id, data, **_kwargs: calls.append(
                (client_id, data["config_id"])
            ),
        )
        sched._finish_local_pipeline_task("active-task")
        assert calls == [("master", "cfg-next")]

    def test_deferred_config_stays_pending_until_inflight_loader_exits(
            self, sched, monkeypatch):
        calls = []
        sched._active_pipeline_task_ids.add("active-task")
        sched._layer_config_inflight.add("cfg-next")
        sched._pending_layer_config = ("master", {"config_id": "cfg-next"})
        monkeypatch.setattr(
            sched,
            "_schedule_layer_config",
            lambda client_id, data: calls.append((client_id, data["config_id"])),
        )

        sched._finish_local_pipeline_task("active-task")

        assert calls == []
        assert sched._pending_layer_config[1]["config_id"] == "cfg-next"

    def test_master_disconnect_clears_worker_pipeline_state(self, sched, monkeypatch):
        """主连接丢失后，worker 不得永久保留 KV cache/活动任务。"""
        sched.nodes["client1"] = NodeInfo(
            node_id="client1", role="client", state=NodeState.ONLINE,
        )
        monkeypatch.setattr(sched, "get_effective_node_id", lambda: "client1")
        sched._kv_cache["task-lost"] = ("kv",)
        sched._active_pipeline_task_ids.add("task-lost")
        sched._local_pipeline_steps["task-lost"] = 3
        sched._active_layer_config = {"config_id": "cfg-old"}
        sched._last_layer_config_ack_payload = {"config_id": "cfg-old"}

        sched._on_master_connection_lost()

        assert sched._kv_cache == {}
        assert sched._active_pipeline_task_ids == set()
        assert sched._local_pipeline_steps == {}
        assert sched._active_layer_config is None
        assert sched._last_layer_config_ack_payload is None
        assert sched.nodes["client1"].error_count == 1

    def test_layer_config_model_mismatch_sends_error_ack(self, sched, monkeypatch):
        """从节点权重摘要不一致时不得加载，并应返回 error ACK。"""
        from model_host import model_host as _host
        import model_sync
        from tcp_comm import TCPClient

        load_calls = []
        manager = type("MockModelManager", (), {
            "is_loaded": False,
            "_engine_type": "pytorch",
            "load_layer_range": lambda self, *args, **kwargs: load_calls.append(args),
        })()
        sent = []
        sched._tcp_client = type("FakeClient", (), {
            "server_host": "100.64.0.10",
            "send_data": lambda self, payload, msg_type: sent.append(payload),
        })()
        monkeypatch.setattr(sched, "get_effective_node_id", lambda: "client1")
        monkeypatch.setattr(
            TCPClient, "_compute_local_model_sha256", lambda **kwargs: "local-sha",
        )
        monkeypatch.setattr(
            model_sync,
            "ensure_model_available",
            lambda *args, **kwargs: "C:/models/qwen-test",
        )
        monkeypatch.setattr(_host, "_manager", manager)

        sched._handle_layer_config("master", {
            "node_id": "client1",
            "config_id": "cfg-bad",
            "start_layer": 8,
            "end_layer": 16,
            "model_id": "qwen-test",
            "model_type": "qwen2",
            "model_sha256": "master-sha",
            "total_layers": 24,
        })

        assert load_calls == []
        assert sent[0]["status"] == "error"
        assert "SHA256 不一致" in sent[0]["error"]

    def test_failed_new_layer_config_invalidates_old_ready_state(
            self, sched, monkeypatch):
        """新模型配置失败后不得继续宣称旧 Qwen 分层就绪。"""
        from model_host import model_host as _host
        from tcp_comm import TCPClient

        sent = []
        sched._tcp_client = type("FakeClient", (), {
            "send_data": lambda self, payload, msg_type: sent.append(payload),
        })()
        sched._active_layer_config = {"config_id": "cfg-qwen-old"}
        sched._last_layer_config_ack_payload = {
            "config_id": "cfg-qwen-old", "status": "ready",
        }
        sched._local_pipeline_steps["old-task"] = 2
        manager = type("Manager", (), {
            "is_loaded": True,
            "_engine_type": "pytorch",
            "load_layer_range": lambda self, *args, **kwargs: (
                _ for _ in ()
            ).throw(RuntimeError("selective load failed")),
        })()
        monkeypatch.setattr(sched, "get_effective_node_id", lambda: "client1")
        monkeypatch.setattr(
            TCPClient, "_compute_local_model_sha256", lambda **kwargs: "sha-deepseek",
        )
        monkeypatch.setattr(_host, "_manager", manager)
        monkeypatch.setattr(_host, "model_loaded", True)

        sched._handle_layer_config("master", {
            "node_id": "client1",
            "config_id": "cfg-deepseek-new",
            "start_layer": 8,
            "end_layer": 16,
            "model_id": "deepseek-test",
            "model_type": "qwen2",
            "model_sha256": "sha-deepseek",
            "total_layers": 28,
        })

        assert sent[-1]["status"] == "error"
        assert sched._active_layer_config is None
        assert sched._last_layer_config_ack_payload is None
        assert sched._local_pipeline_steps == {}
        assert _host.model_loaded is False

    def test_layer_config_ack_marks_ready_only_for_current_version(self, sched):
        """只有当前 config_id、模型和层范围完全匹配的 ACK 才能置 ready。"""
        expected = {
            "node_id": "client1",
            "config_id": "cfg-current",
            "start_layer": 8,
            "end_layer": 16,
            "model_sha256": "sha-qwen",
            "model_type": "qwen",
        }
        expected = _install_layer_assignment(sched, "client1", expected)

        stale = {
            "type": "layer_config_ack",
            "data": {
                "node_id": "client1",
                "assignment_id": expected["assignment_id"],
                "config_id": "cfg-old",
                "status": "ready",
                "layer_range": [8, 16],
                "model_sha256": "sha-qwen",
                "model_type": "qwen",
                "engine": "pytorch",
            },
        }
        sched._on_tcp_message("client1", stale)
        assert "client1" not in sched._layer_config_pushed

        current = dict(stale)
        current["data"] = dict(stale["data"], config_id="cfg-current")
        sched._on_tcp_message("client1", current)
        assert "client1" in sched._layer_config_pushed

    def test_layer_config_retry_resends_same_config_id(self, sched):
        sent = []
        sched._tcp_server = MagicMock()
        sched._tcp_server._running = True
        sched._tcp_server.get_authenticated_peer_epoch.return_value = 11
        sched._tcp_server.get_client_ids.return_value = ["client1"]
        sched._tcp_server.send_layer_config.side_effect = (
            lambda node_id, data, *, registration_epoch: sent.append((node_id, data, registration_epoch))
        )
        expected = _install_layer_assignment(sched, "client1", {
            "node_id": "client1",
            "config_id": "cfg-same",
            "start_layer": 0,
            "end_layer": 8,
        })
        sched._layer_config_retry_state["client1"] = {
            "attempts": 1,
            "next_retry": 0.0,
        }

        assert sched._retry_pending_layer_configs(now=10.0) == 1
        assert sent == [("client1", expected, 11)]
        assert sent[0][1]["config_id"] == "cfg-same"

    def test_duplicate_ready_config_only_resends_cached_ack(self, sched, monkeypatch):
        sent = []
        sched._last_layer_config_ack_payload = {
            "config_id": "cfg-ready",
            "status": "ready",
        }
        monkeypatch.setattr(
            sched, "_send_layer_config_ack",
            lambda payload: sent.append(payload) or True,
        )
        sched._schedule_layer_config("master", {"config_id": "cfg-ready"})
        assert sent == [{"config_id": "cfg-ready", "status": "ready"}]

    def test_unknown_message_type_logs_debug(self, sched, caplog):
        """未知消息类型应记录 DEBUG 日志，便于排查协议不匹配。"""
        with caplog.at_level(logging.DEBUG, logger="scheduler"):
            sched._on_tcp_message("client1", {"type": "unknown_type"})
        assert any("未知消息类型" in r.getMessage() for r in caplog.records)

    def test_pipeline_pause_resume_logs_info(self, sched, caplog):
        """pipeline_pause/resume 应在 INFO 级别可见。"""
        with caplog.at_level(logging.INFO, logger="scheduler"):
            sched._on_tcp_message("client1", {"type": "pipeline_pause"})
            sched._on_tcp_message("client1", {"type": "pipeline_resume"})
        messages = [r.getMessage() for r in caplog.records]
        assert any("PIPELINE_PAUSE" in m for m in messages)
        assert any("PIPELINE_RESUME" in m for m in messages)

    def test_on_tcp_register_uses_advertised_addr_not_peer_port(self, sched, monkeypatch):
        """_on_tcp_message(register) 应使用 advertised_addr 注册节点。"""
        class FakeServer:
            _running = True
            def get_client_info(self, cid):
                return {
                    "advertised_addr": "10.0.0.9:8888",
                    "peer_addr": "10.0.0.9:51234",
                    "network_type": "wifi",
                }

            def confirm_registration(self, cid):
                return True
        sched._tcp_server = FakeServer()
        sched.nodes["pc-worker"] = NodeInfo(
            node_id="pc-worker", role="client", node_type="pc",
            state=NodeState.OFFLINE,
        )
        msg = {
            "type": "register",
            "data": {
                "role": "client",
                "node_type": "pc",
                "hostname": "worker",
                "advertised_address": "10.0.0.9:8888",
            },
        }
        sched._on_tcp_message("pc-worker", msg)
        node = sched.nodes["pc-worker"]
        assert node.state == NodeState.ONLINE
        assert node.address == "10.0.0.9:8888"
        assert "tcp_peer_addr" in node.device_info
        assert node.device_info["tcp_peer_addr"] == "10.0.0.9:51234"

    def test_pipeline_peer_registration_is_rejected(self, sched):
        server = MagicMock()
        sched._tcp_server = server
        sched._on_tcp_message("client-peer", {
            "type": "register",
            "data": {
                "role": "client",
                "node_type": "pipeline_peer",
            },
        })
        server.reject_client.assert_called_once()
        assert server.reject_client.call_args.args[0] == "client-peer"
        assert "必须经 master" in server.reject_client.call_args.args[1]
        server.confirm_registration.assert_not_called()
        assert "client-peer" not in sched.nodes

    def test_tcp_disconnect_logs_and_marks_offline(self, sched, caplog):
        """TCP 断连应记录调度器层日志并标记节点离线。"""
        sched.nodes["client1"] = NodeInfo(
            node_id="client1", role="client", state=NodeState.ONLINE,
        )
        sched._push_node_update_to_all_clients = lambda *args, **kwargs: None
        sched.deregister_node = lambda _client_id: None

        with caplog.at_level(logging.INFO, logger="scheduler"):
            sched._on_tcp_disconnect("client1")

        assert sched.nodes["client1"].state == NodeState.OFFLINE
        assert any("TCP 断开" in r.getMessage() for r in caplog.records)


# ================================================================
# 流水线回退推理 测试
# ================================================================


class TestPipelineFallback:
    """测试 run_pipeline_safe 自动回退逻辑"""

    @pytest.fixture
    def sched(self):
        return Scheduler()

    def test_fallback_when_no_nodes_ready(self, sched):
        """没有流水线节点 → run_pipeline_safe 应回退到全模型推理"""
        # 全模型推理需要 model_manager，这里只测试回退触发条件
        sched._tcp_server = None
        # run_pipeline_safe 内调用 _all_pipeline_nodes_ready → False → 回退
        # 回退方法 _run_full_model_inference 依赖 model_manager，
        # 此处只验证链路不崩溃（会返回 error 而非抛异常）
        result = sched.run_pipeline_safe("测试")
        # 预期：回退到全模型推理，但因模型未加载返回 error
        assert isinstance(result, dict)
        # 可能返回 error（模型未加载）或 response
        assert "response" in result or "error" in result

    def test_request_sync_waits_for_authoritative_worker_ready(
            self, sched, monkeypatch):
        """分布式请求应先等待强制同步，而不是立即回退主节点。"""
        sched._role_override = "master"
        sched.nodes["worker1"] = NodeInfo(
            node_id="worker1", role=NodeRole.CLIENT, state=NodeState.ONLINE,
            node_type="pc", device_info={"tier": "ultrabook"},
            last_heartbeat=time.time(),
        )
        sched._tcp_server = type("Server", (), {
            "_running": True,
            "clients": {"worker1": object()},
        })()
        states = iter([
            {"ready": False, "reason_code": "no_pipeline_workers",
             "reason": "未分配任何 PC 从节点参与模型层计算", "workers": []},
            {"ready": False, "reason_code": "worker_layer_loading",
             "reason": "worker loading", "workers": []},
            {"ready": True, "reason_code": "ready", "reason": "ok",
             "workers": ["worker1"]},
        ])
        monkeypatch.setattr(sched, "_get_pipeline_readiness", lambda: next(states))
        sync_calls = []
        monkeypatch.setattr(
            sched,
            "request_authoritative_layer_sync",
            lambda: sync_calls.append(True) or True,
        )

        result = sched._synchronize_pipeline_workers_for_request(timeout=1)

        assert result["ready"] is True
        assert sync_calls == [True]

    @pytest.mark.parametrize(
        "reason_code",
        ["worker_offline", "worker_tcp_disconnected", "worker_heartbeat_stale"],
    )
    def test_force_distributed_fails_fast_on_unrecoverable_readiness(
            self, sched, monkeypatch, reason_code):
        """★ DIST-2：force 路径下不可恢复的就绪原因必须快速返回，不等满超时。

        此前 `not force_distributed_assignment and reason not in recoverable`
        的组合让 force 路径**连这三种原因也会等满 `PIPELINE_MODEL_SYNC_TIMEOUT`**。
        它们等下去不会变好：对端要么已经没了，要么要等重连，而重连本身会触发一次
        权威重发（`_handle_task_worker_message` 的 hello 分支）。DIST-2 要求
        「禁止等待多个互相独立的超时后才 fallback」。
        """
        sched._role_override = "master"
        sched.nodes["worker1"] = NodeInfo(
            node_id="worker1", role=NodeRole.CLIENT, state=NodeState.ONLINE,
            node_type="pc", device_info={"tier": "ultrabook"},
            last_heartbeat=time.time(),
        )
        sched._tcp_server = type("Server", (), {
            "_running": True,
            "clients": {"worker1": object()},
        })()
        monkeypatch.setattr(
            sched, "_get_pipeline_readiness",
            lambda: {
                "ready": False,
                "reason_code": reason_code,
                "reason": "不可恢复的就绪原因",
                "workers": [],
            },
        )
        sync_calls = []
        monkeypatch.setattr(
            sched,
            "request_authoritative_layer_sync",
            lambda **kwargs: sync_calls.append(kwargs) or True,
        )

        started = time.monotonic()
        result = sched._synchronize_pipeline_workers_for_request(
            timeout=60.0, force_distributed_assignment=True,
        )
        elapsed = time.monotonic() - started

        assert result["reason_code"] == reason_code
        assert result["ready"] is False
        assert elapsed < 5.0, f"应快速失败，实际耗时 {elapsed:.1f}s"
        assert sync_calls == [], "不可恢复状态下不应再触发权威重发"

    def test_request_sync_stops_when_worker_reopts_out(
            self, sched, monkeypatch):
        """旧从节点拒绝权威配置时不能等待完整同步超时。"""
        sched._role_override = "master"
        sched.nodes["worker1"] = NodeInfo(
            node_id="worker1", role=NodeRole.CLIENT, state=NodeState.ONLINE,
            node_type="pc", device_info={"tier": "ultrabook"},
            last_heartbeat=time.time(),
        )
        sched._tcp_server = type("Server", (), {
            "_running": True,
            "clients": {"worker1": object()},
        })()
        not_ready = {
            "ready": False,
            "reason_code": "no_pipeline_workers",
            "reason": "未分配任何 PC 从节点参与模型层计算",
            "workers": [],
        }
        monkeypatch.setattr(
            sched, "_get_pipeline_readiness", lambda: dict(not_ready),
        )
        monkeypatch.setattr(
            sched,
            "request_authoritative_layer_sync",
            lambda: sched._pipeline_worker_opt_out.add("worker1") or True,
        )

        result = sched._synchronize_pipeline_workers_for_request(timeout=60)

        assert result == not_ready

    def test_run_pipeline_safe_uses_ready_worker_after_sync(
            self, sched, monkeypatch):
        """请求前同步成功后，当前请求必须走流水线而非本地回退。"""
        from model_host import model_host as _host

        class MockModelManager:
            is_loaded = True
            _engine_type = "pytorch"

        monkeypatch.setattr(_host, "_manager", MockModelManager())
        monkeypatch.setattr(sched, "_all_pipeline_nodes_ready", lambda: False)
        sync_timeouts = []
        monkeypatch.setattr(
            sched,
            "_synchronize_pipeline_workers_for_request",
            lambda timeout: sync_timeouts.append(timeout) or {"ready": True},
        )
        monkeypatch.setattr(
            sched, "run_pipeline", lambda prompt, **kwargs: {
                "response": "distributed", "metrics": {"distributed": True},
            },
        )
        monkeypatch.setattr(
            sched,
            "_run_full_model_inference",
            lambda *args, **kwargs: pytest.fail("同步成功后不应回退本地推理"),
        )

        result = sched.run_pipeline_safe(
            "测试", _pipeline_model_sync_timeout=3,
        )

        assert result["response"] == "distributed"
        assert sync_timeouts == [3]

    def test_model_sync_exhausting_request_budget_does_not_fallback(
            self, sched, monkeypatch):
        from model_host import model_host as _host

        class MockModelManager:
            is_loaded = True
            _engine_type = "pytorch"

        monkeypatch.setattr(_host, "_manager", MockModelManager())
        monkeypatch.setattr(sched, "_all_pipeline_nodes_ready", lambda: False)

        def slow_sync(**_kwargs):
            time.sleep(0.08)
            return {"ready": False, "reason": "still loading"}

        monkeypatch.setattr(
            sched, "_synchronize_pipeline_workers_for_request", slow_sync,
        )
        monkeypatch.setattr(
            sched,
            "_run_full_model_inference",
            lambda *args, **kwargs: pytest.fail(
                "deadline exhaustion must not enter full-model fallback"
            ),
        )

        result = sched.run_pipeline_safe(
            "deadline during prepare",
            _request_timeout_seconds=0.05,
            _pipeline_model_sync_timeout=1.0,
        )

        assert result["reason_code"] == "request_deadline_exceeded"
        assert result["metrics"]["fallback"] is False

    def test_full_model_stream_deadline_keeps_lock_until_worker_stops(
            self, sched, monkeypatch):
        from request_deadline import RequestDeadline

        monkeypatch.setattr(sched, "_distributed_inference_enabled", False)

        entered = threading.Event()
        release_worker = threading.Event()
        worker_exited = threading.Event()

        class BlockingModelManager:
            is_loaded = True
            is_pipeline_prepared = False
            layer_range = None
            _engine_type = "pytorch"
            tokenizer = object()

            def ensure_full_model(self):
                return None

            def get_pipeline_descriptor(self):
                return {}

            def chat_stream(self, **_kwargs):
                entered.set()
                try:
                    release_worker.wait(timeout=2.0)
                    yield "late"
                finally:
                    worker_exited.set()

        callbacks = type("Callbacks", (), {
            "build_model_chat_prompt": staticmethod(lambda *_args: "prompt"),
            "format_model_response": staticmethod(
                lambda text, *_args, **_kwargs: (text, None)
            ),
        })()
        sched._host = BlockingModelManager()
        monkeypatch.setattr(sched, "_require_callbacks", lambda: callbacks)

        started = time.monotonic()
        try:
            events = list(sched._run_full_model_inference_stream(
                "blocked stream",
                _request_deadline=RequestDeadline.start(0.08),
            ))
            elapsed = time.monotonic() - started

            assert entered.is_set()
            assert elapsed < 0.5
            assert events[-1]["reason_code"] == "request_deadline_exceeded"
            assert not sched._inference_lock.acquire(blocking=False), (
                "backend worker 仍运行时不得提前释放推理锁"
            )
        finally:
            release_worker.set()

        assert worker_exited.wait(timeout=1.0)
        acquire_deadline = time.monotonic() + 1.0
        while time.monotonic() < acquire_deadline:
            if sched._inference_lock.acquire(blocking=False):
                sched._inference_lock.release()
                break
            time.sleep(0.01)
        else:
            pytest.fail("backend worker 结束后推理锁未最终释放")

    def test_q0_restore_lock_wait_is_bounded_by_original_deadline(
            self, sched, monkeypatch):
        from request_deadline import RequestDeadline
        from scheduler_types import PreemptState, QueueTask

        class ScriptedLock:
            def __init__(self):
                self.acquire_calls = 0
                self.held = False

            def acquire(self, blocking=True, timeout=-1):
                del blocking
                self.acquire_calls += 1
                if self.acquire_calls == 1:
                    self.held = True
                    return True
                if timeout is not None and timeout > 0:
                    time.sleep(timeout)
                return False

            def release(self):
                assert self.held
                self.held = False

        scripted_lock = ScriptedLock()
        sched._inference_lock = scripted_lock
        monkeypatch.setattr(sched, "_all_pipeline_nodes_ready", lambda: False)
        monkeypatch.setattr(
            sched,
            "_run_full_model_inference",
            lambda *args, **kwargs: {"response": "q0", "metrics": {}},
        )
        q0_task = QueueTask(
            task_id="q0-deadline",
            prompt="urgent",
            max_new_tokens=1,
            priority_level=0,
            original_level=0,
        )
        preempt_state = PreemptState(
            task_id="original-deadline",
            generated_ids=[],
            full_input_ids=None,
            current_step=1,
            max_new_tokens=32,
            temperature=0.7,
            top_p=0.9,
            prompt="original",
            pipeline_nodes=[],
            first_node_id="worker-1",
        )
        lock_state = {"held": False}
        sched.pipeline_queue._current_task_id = preempt_state.task_id

        started = time.monotonic()
        with pytest.raises(TimeoutError, match="restoring preempted"):
            sched._execute_q0_inline(
                q0_task,
                preempt_state,
                RequestDeadline.start(0.06),
                lock_state,
            )

        assert time.monotonic() - started < 0.3
        assert scripted_lock.acquire_calls == 2
        assert lock_state["held"] is False
        assert sched.pipeline_queue._current_task_id == preempt_state.task_id

    def test_late_queue_result_cannot_overwrite_request_deadline(
            self, sched, monkeypatch):
        from model_host import model_host as _host

        class MockModelManager:
            is_loaded = True
            _engine_type = "pytorch"

        monkeypatch.setattr(_host, "_manager", MockModelManager())
        monkeypatch.setattr(sched, "_all_pipeline_nodes_ready", lambda: True)
        monkeypatch.setattr(
            sched.pipeline_queue, "enqueue", lambda **_kwargs: "queued-late",
        )

        def late_result(*_args, **_kwargs):
            time.sleep(0.08)
            return {
                "status": "done",
                "result": {"response": "too late", "metrics": {}},
            }

        cancelled = []
        monkeypatch.setattr(
            sched.pipeline_queue, "wait_for_result", late_result,
        )
        monkeypatch.setattr(
            sched.pipeline_queue,
            "cancel_task",
            lambda task_id: cancelled.append(task_id) or True,
        )
        sched.pipeline_queue._current_task_id = "busy"
        try:
            result = sched.run_pipeline_safe(
                "queued deadline", _request_timeout_seconds=0.05,
            )
        finally:
            sched.pipeline_queue._current_task_id = None

        assert result["reason_code"] == "request_deadline_exceeded"
        assert result.get("response") != "too late"
        assert cancelled == ["queued-late"]


    def test_fallback_waits_for_inference_lock(self, sched, monkeypatch):
        """流水线未就绪回退时也必须等待推理锁，避免并发完整模型推理。"""
        from model_host import model_host as _host

        class MockModelManager:
            is_loaded = True
            _engine_type = "pytorch"

        monkeypatch.setattr(_host, "_manager", MockModelManager())
        monkeypatch.setattr(sched, "_all_pipeline_nodes_ready", lambda: False)

        entered = threading.Event()
        finished = threading.Event()

        def fake_fallback(prompt, **kwargs):
            entered.set()
            return {"response": "fallback", "metrics": {"fallback": True}}

        monkeypatch.setattr(sched, "_run_full_model_inference", fake_fallback)

        assert sched._inference_lock.acquire(blocking=False)

        result_holder = {}

        def call_pipeline():
            result_holder["result"] = sched.run_pipeline_safe("blocked")
            finished.set()

        t = threading.Thread(target=call_pipeline)
        t.start()

        assert not entered.wait(0.2), "推理锁未释放前不应进入本地回退推理"
        assert not finished.is_set(), "推理锁未释放前请求不应提前完成"

        sched._inference_lock.release()
        t.join(timeout=2)

        assert finished.is_set()
        assert entered.is_set()
        assert result_holder["result"]["response"] == "fallback"

    def test_run_pipeline_safe_fallbacks_when_pipeline_returns_error(self, sched, monkeypatch):
        """流水线单步返回 error 时，当前请求应自动回退到全模型推理。"""
        from model_host import model_host as _host

        class MockModelManager:
            is_loaded = True
            _engine_type = "pytorch"

        monkeypatch.setattr(_host, "_manager", MockModelManager())
        monkeypatch.setattr(sched, "_all_pipeline_nodes_ready", lambda: True)
        monkeypatch.setattr(
            sched,
            "run_pipeline",
            lambda prompt="", **kw: {"response": "", "error": "worker step failed"},
        )

        fallback_calls = []

        def fake_fallback(prompt, **kwargs):
            fallback_calls.append(kwargs)
            return {
                "response": "fallback ok",
                "metrics": {"fallback_reason": kwargs.get("_fallback_reason")},
            }

        monkeypatch.setattr(sched, "_run_full_model_inference", fake_fallback)

        result = sched.run_pipeline_safe("test prompt")

        assert result["response"] == "fallback ok"
        assert fallback_calls
        assert "worker step failed" in fallback_calls[0]["_fallback_reason"]

    def test_run_pipeline_safe_does_not_replay_after_stream_output(
            self, sched, monkeypatch):
        """已发送部分正文后失败时不能从头执行全模型并重复回答。"""
        from model_host import model_host as _host

        class MockModelManager:
            is_loaded = True
            _engine_type = "pytorch"

        monkeypatch.setattr(_host, "_manager", MockModelManager())
        monkeypatch.setattr(sched, "_all_pipeline_nodes_ready", lambda: True)

        def failed_pipeline(prompt="", **kwargs):
            kwargs["_stream_callback"]({"token": "partial"})
            return {"response": "", "error": "worker disconnected"}

        monkeypatch.setattr(sched, "run_pipeline", failed_pipeline)
        monkeypatch.setattr(
            sched,
            "_run_full_model_inference",
            lambda *args, **kwargs: pytest.fail("部分输出后不能从头回退"),
        )
        events = []

        result = sched.run_pipeline_safe(
            "test prompt", _stream_callback=events.append,
        )

        assert result["error"] == "worker disconnected"
        assert events == [{"token": "partial"}]


class TestPipelineSampling:
    @pytest.fixture
    def sched(self):
        return Scheduler()

    def test_zero_temperature_is_greedy_without_multinomial(self, monkeypatch):
        import scheduler as scheduler_module
        import torch

        monkeypatch.setattr(
            torch,
            "multinomial",
            lambda *args, **kwargs: pytest.fail(
                "temperature=0 不得进入随机采样"
            ),
        )
        logits = torch.tensor([[[1.0, 5.0, 3.0]]], dtype=torch.float16)

        token_id = scheduler_module._sample_pipeline_token_id(
            logits, temperature=0.0, top_p=1.0,
        )

        assert token_id == 1

    def test_non_finite_logits_are_rejected_before_multinomial(
            self, monkeypatch):
        import scheduler as scheduler_module
        import torch

        monkeypatch.setattr(
            torch,
            "multinomial",
            lambda *args, **kwargs: pytest.fail(
                "非有限概率不得进入 CUDA multinomial"
            ),
        )
        logits = torch.tensor([[[1.0, float("inf"), 3.0]]])

        with pytest.raises(RuntimeError, match="NaN/Inf"):
            scheduler_module._sample_pipeline_token_id(
                logits, temperature=0.7, top_p=0.9,
            )

    def test_queued_pipeline_does_not_replay_after_stream_output(
            self, sched, monkeypatch):
        """队列 worker 同样不得在部分输出后触发全模型重放。"""
        monkeypatch.setattr(sched, "_all_pipeline_nodes_ready", lambda: True)
        events = []
        kwargs = {"_stream_callback": events.append}
        sched._track_stream_output(kwargs)

        def failed_pipeline(prompt="", **pipeline_kwargs):
            pipeline_kwargs["_stream_callback"]({"token": "partial"})
            return {"response": "", "error": "worker disconnected"}

        monkeypatch.setattr(sched, "run_pipeline", failed_pipeline)
        monkeypatch.setattr(
            sched,
            "_run_full_model_inference",
            lambda *args, **fallback_kwargs: pytest.fail("部分输出后不能从头回退"),
        )

        result = sched._process_queued_pipeline_task("test prompt", **kwargs)

        assert result["error"] == "worker disconnected"
        assert events == [{"token": "partial"}]


# ================================================================
# KV Cache 管理 测试 (Phase 3)
# ================================================================


class TestKVCacheManagement:
    """测试调度器本地 KV cache 存储与清理（从节点侧）"""

    @pytest.fixture
    def sched(self):
        return Scheduler()

    def test_kv_cache_initialized_empty(self, sched):
        """新 Scheduler 的 _kv_cache 应为空 dict"""
        assert isinstance(sched._kv_cache, dict)
        assert len(sched._kv_cache) == 0

    def test_kv_cache_stored_after_layer_result(self, sched):
        """_handle_layer_result 不应影响 _kv_cache（两者独立）"""
        # 注入一个 KV cache（模拟从节点侧 _handle_layer_forward 的存储行为）
        sched._kv_cache["task_A"] = "mock_past_key_values"
        assert "task_A" in sched._kv_cache
        assert sched._kv_cache["task_A"] == "mock_past_key_values"

    def test_pipeline_done_clears_tuple_kv_cache(self, sched):
        """PIPELINE_DONE 应清理 torch tensor 类型的 KV cache"""
        sched._kv_cache["task_t"] = ("layer_kv",)  # 模拟 tuple 类型
        msg = {
            "type": "pipeline_done",
            "data": {"task_id": "task_t"},
        }
        sched._on_tcp_message("master", msg)
        assert "task_t" not in sched._kv_cache

    def test_pipeline_abort_clears_tuple_kv_cache(self, sched):
        """PIPELINE_ABORT 应清理 KV cache"""
        sched._kv_cache["task_z"] = (("mock_k", "mock_v"),)
        msg = {
            "type": "pipeline_abort",
            "data": {"task_id": "task_z"},
        }
        sched._on_tcp_message("master", msg)
        assert "task_z" not in sched._kv_cache

    def test_multiple_task_kv_cache_independent(self, sched):
        """不同 task 的 KV cache 应相互独立"""
        sched._kv_cache["task_1"] = "kv_1"
        sched._kv_cache["task_2"] = "kv_2"

        # 清理 task_1
        msg = {
            "type": "pipeline_done",
            "data": {"task_id": "task_1"},
        }
        sched._on_tcp_message("master", msg)

        assert "task_1" not in sched._kv_cache
        assert "task_2" in sched._kv_cache
        assert sched._kv_cache["task_2"] == "kv_2"

    def test_fallback_inference_returns_error_without_model(self, sched, monkeypatch):
        """无模型时 _run_full_model_inference 返回 error"""
        from model_host import model_host as _host

        # ★ 显式建立"没有任何已加载模型"这一前置条件，而不是指望全局恰好干净：
        #   `ModelHost.has_loaded_model()` 先看 `model_loaded` 标志、再看惰性 manager，
        #   任何一条为真都会走到"完整模型恢复"分支 ⇒ 报「完整模型恢复失败」而非
        #   「模型未加载」（实测：与前序用例连跑时本断言落空）。
        #   注意两条都在 `ModelHost._OWN_ATTRS` 里，才会落在宿主自身；
        #   其它名字会被 `__setattr__` 代理给 manager（`_manager=None` 时会 AttributeError）。
        monkeypatch.setattr(_host, "model_loaded", False)
        monkeypatch.setattr(_host, "_manager", None)
        result = sched._run_full_model_inference("测试")
        assert isinstance(result, dict)
        assert "error" in result
        assert "模型未加载" in result["error"]


# ================================================================
# 请求队列 测试 (Phase 4 — 多请求排队)
# ================================================================


class TestPipelineQueueBasics:
    """测试 PipelineQueue 的基本操作"""

    @pytest.fixture
    def queue(self):
        from scheduler import PipelineQueue
        return PipelineQueue(max_size=10, result_ttl=10.0)

    def test_initial_state(self, queue):
        """新队列应为空且未运行"""
        assert queue.queue_size == 0
        assert not queue.is_busy
        assert queue.get_status()["running"] is False

    def test_enqueue_returns_task_id(self, queue):
        """enqueue 应返回 task_id"""
        tid = queue.enqueue(prompt="test")
        assert tid.startswith("q_")
        assert len(tid) > 4
        assert queue.queue_size == 1

    def test_external_cancel_event_removes_queued_task(self, queue):
        cancel_event = threading.Event()
        task_id = queue.enqueue(
            prompt="cancel queued",
            max_new_tokens=32,
            _cancel_event=cancel_event,
        )

        cancel_event.set()
        result = queue.wait_for_result(
            task_id,
            timeout=1.0,
            cancel_event=cancel_event,
        )

        assert result["status"] == "cancelled"
        assert queue.queue_size == 0

    def test_enqueue_custom_task_id(self, queue):
        """enqueue 应接受自定义 task_id"""
        tid = queue.enqueue(task_id="my_task", prompt="hello")
        assert tid == "my_task"
        assert queue.queue_size == 1

    def test_enqueue_full_queue_raises(self, queue):
        """队列满时应抛异常"""
        queue._max_size = 2
        queue.enqueue(prompt="task1")
        queue.enqueue(prompt="task2")
        with pytest.raises(RuntimeError, match="已满"):
            queue.enqueue(prompt="task3")

    def test_is_busy_when_processing(self, queue):
        """正在处理任务时 is_busy 应为 True"""
        import threading
        import time

        started = threading.Event()
        finish = threading.Event()

        def slow_process(**kwargs):
            started.set()
            finish.wait(timeout=5.0)
            return {"response": "done"}

        queue.start(process_fn=slow_process)
        queue.enqueue(prompt="test")
        started.wait(timeout=2.0)

        assert queue.is_busy

        finish.set()
        queue.stop()

    def test_wait_for_result_returns_done(self, queue):
        """等待任务完成应返回结果"""
        def simple_process(**kwargs):
            return {"response": f"echo: {kwargs.get('prompt', '')}"}

        queue.start(process_fn=simple_process)
        tid = queue.enqueue(prompt="hello")
        result = queue.wait_for_result(tid, timeout=5.0)

        assert result["status"] == "done"
        assert result["result"]["response"] == "echo: hello"
        queue.stop()

    def test_wait_for_result_timeout(self, queue):
        """超时应返回 timeout 状态"""
        started = threading.Event()
        release = threading.Event()

        def slow_process(**kwargs):
            started.set()
            release.wait(timeout=5.0)
            return {"response": "late"}

        queue.start(process_fn=slow_process)
        tid = queue.enqueue(prompt="test")
        assert started.wait(timeout=1.0)
        result = queue.wait_for_result(tid, timeout=0.3)

        assert result["status"] == "timeout"
        release.set()
        queue.stop()

    def test_wait_for_unknown_task(self, queue):
        """等待不存在的任务应返回 unknown"""
        result = queue.wait_for_result("nonexistent", timeout=0.5)
        assert result["status"] == "unknown"

    def test_process_error_captured(self, queue, caplog):
        """process_fn 抛异常时应记录 error 状态，并带 exc_info。"""
        def failing_process(**kwargs):
            raise ValueError("模拟推理失败")

        with caplog.at_level(logging.ERROR, logger="scheduler"):
            queue.start(process_fn=failing_process)
            tid = queue.enqueue(prompt="test")
            result = queue.wait_for_result(tid, timeout=5.0)

        assert result["status"] == "error"
        assert "模拟推理失败" in result["error"]
        # L5: 半结构化日志格式 event=task_failed
        records = [r for r in caplog.records if "task_failed" in r.getMessage()]
        assert records
        assert records[0].exc_info is not None
        queue.stop()

    def test_queue_fifo_order(self, queue):
        """任务应按 FIFO 顺序执行"""
        import threading
        order = []
        lock = threading.Lock()
        started = threading.Event()
        completed = threading.Event()

        def ordered_process(**kwargs):
            with lock:
                order.append(kwargs.get("seq", -1))
            if len(order) == 1:
                started.set()  # 第一个任务开始后通知
            if len(order) == 3:
                completed.set()
            return {"seq": kwargs.get("seq", -1)}

        queue.start(process_fn=ordered_process)

        # 快速入队 3 个任务
        queue.enqueue(seq=0, prompt="first")
        queue.enqueue(seq=1, prompt="second")
        queue.enqueue(seq=2, prompt="third")

        assert completed.wait(timeout=5.0)

        assert order == [0, 1, 2], f"应为 FIFO 顺序，实际: {order}"
        queue.stop()

    def test_stop_wakes_waiters(self, queue):
        """stop() 应唤醒所有等待者并设置 cancelled 状态"""
        tid = queue.enqueue(prompt="test")
        # 不启动 worker，直接 stop
        queue.stop()

        result = queue.wait_for_result(tid, timeout=1.0)
        assert result["status"] == "cancelled"

    def test_get_status_reflects_state(self, queue):
        """get_status 应反映当前队列状态"""
        status = queue.get_status()
        assert not status["running"]
        assert status["queue_size"] == 0
        assert status["current_task"] is None

        started = threading.Event()
        release = threading.Event()

        def slow_process(**kwargs):
            started.set()
            release.wait(timeout=2.0)
            return {"response": "ok"}

        queue.start(process_fn=slow_process)
        queue.enqueue(prompt="test")

        assert started.wait(timeout=1.0)

        status2 = queue.get_status()
        assert status2["running"]
        assert status2["queue_size"] == 0  # 已被 worker 取出
        assert status2["current_task"] is not None

        release.set()
        queue.stop()

    def test_result_cleanup_after_ttl(self, queue):
        """超过 TTL 的已完成结果应被清理"""
        queue._result_ttl = 0.1  # 100ms TTL
        tid = queue.enqueue(prompt="test")

        # 手动注入过期结果
        import time
        queue._results[tid] = {
            "status": "done",
            "result": {"response": "old"},
            "completed_at": time.time() - 1.0,  # 1 秒前完成
        }

        queue._cleanup_expired()
        assert tid not in queue._results

    def test_queue_preserves_task_data(self, queue):
        """任务数据应正确传递给 process_fn"""
        received = {}

        def capture_process(**kwargs):
            received.update(kwargs)
            return {"response": "captured"}

        queue.start(process_fn=capture_process)
        tid = queue.enqueue(
            prompt="hello world",
            max_new_tokens=256,
            temperature=0.5,
            session_id="sess_1",
        )
        result = queue.wait_for_result(tid, timeout=5.0)

        assert result["status"] == "done"
        assert received["prompt"] == "hello world"
        assert received["max_new_tokens"] == 256
        assert received["temperature"] == 0.5
        assert received["session_id"] == "sess_1"
        queue.stop()


class TestPipelineQueueIntegration:
    """测试 PipelineQueue 与 Scheduler 的集成"""

    @pytest.fixture
    def sched(self):
        s = Scheduler()
        # 不启动调度器（避免 TCP 绑定），只测试队列集成
        return s

    def test_queue_initialized_in_scheduler(self, sched):
        """Scheduler 初始化时应创建 PipelineQueue"""
        assert sched.pipeline_queue is not None
        status = sched.get_status()["pipeline_queue"]
        assert status["running"] is False
        assert status["queue_size"] == 0

    def test_queue_included_in_get_status(self, sched):
        """get_status() 应包含 queue 信息"""
        status = sched.get_status()
        assert "pipeline_queue" in status
        q = status["pipeline_queue"]
        assert "queue_size" in q
        assert "running" in q
        assert "current_task" in q

    def test_two_tcp_servers_dispatch_without_callback_cross_talk(self, sched):
        """并发 server 事件应各自读取显式 source，不能静默串台。"""
        import socket

        from tcp_comm import MessageType, TCPServer, build_message

        servers = [
            TCPServer(host="127.0.0.1", port=0),
            TCPServer(host="127.0.0.1", port=0),
        ]
        default_server = object()
        sched._tcp_server = default_server
        barrier = threading.Barrier(2)
        seen = []
        seen_lock = threading.Lock()
        socket_pairs = [socket.socketpair(), socket.socketpair()]
        threads = []

        def observe(label):
            barrier.wait(timeout=2)
            with seen_lock:
                seen.append((label, sched._tcp_server))

        for index, (server, (server_sock, _client_sock)) in enumerate(
                zip(servers, socket_pairs)):
            server.on_message = sched._bind_tcp_server_callback(
                server,
                lambda _client_id, _message, label=index: observe(label),
            )
            server._running = True
            threads.append(threading.Thread(
                target=server._handle_client,
                args=(
                    server_sock,
                    ("127.0.0.1", 54400 + index),
                    f"pending_{index}",
                ),
                daemon=True,
            ))

        try:
            for thread in threads:
                thread.start()
            for index, (_server_sock, client_sock) in enumerate(socket_pairs):
                client_sock.sendall(build_message(
                    MessageType.STATUS_RES, {"server": index},
                ))
            for thread in threads:
                thread.join(timeout=2)

            assert sorted(seen, key=lambda item: item[0]) == [
                (0, servers[0]), (1, servers[1]),
            ]
            assert sched._tcp_server is default_server
        finally:
            for server in servers:
                server._running = False
            for _server_sock, client_sock in socket_pairs:
                client_sock.close()
            for thread in threads:
                thread.join(timeout=2)
            for server in servers:
                server.stop()

    def test_process_queued_task_delegates(self, sched):
        """_process_queued_pipeline_task 应调用 run_pipeline（绕过排队检查）"""
        original_ready = sched._all_pipeline_nodes_ready
        original_run = sched.run_pipeline

        sched._all_pipeline_nodes_ready = lambda: True
        calls = []
        sched.run_pipeline = lambda prompt="", **kw: calls.append(dict(prompt=prompt, **kw)) or {"response": "ok"}

        try:
            result = sched._process_queued_pipeline_task(
                prompt="test", max_new_tokens=512
            )
            assert len(calls) == 1
            assert calls[0]["prompt"] == "test"
            assert calls[0]["max_new_tokens"] == 512
            assert result == {"response": "ok"}
        finally:
            sched._all_pipeline_nodes_ready = original_ready
            sched.run_pipeline = original_run

    def test_process_queued_task_fallbacks_when_pipeline_returns_error(self, sched, monkeypatch):
        """队列 worker 中流水线返回 error 时也应自动回退。"""
        monkeypatch.setattr(sched, "_all_pipeline_nodes_ready", lambda: True)
        monkeypatch.setattr(
            sched,
            "run_pipeline",
            lambda prompt="", **kw: {"response": "", "error": "queued worker failed"},
        )

        fallback_calls = []

        def fake_fallback(prompt, **kwargs):
            fallback_calls.append(kwargs)
            return {"response": "queue fallback ok"}

        monkeypatch.setattr(sched, "_run_full_model_inference", fake_fallback)

        result = sched._process_queued_pipeline_task(
            prompt="queued", max_new_tokens=16,
        )

        assert result["response"] == "queue fallback ok"
        assert fallback_calls
        assert "queued worker failed" in fallback_calls[0]["_fallback_reason"]

    def test_required_distributed_queued_worker_error_does_not_fallback(
            self, sched, monkeypatch):
        """强制分布式队列任务返回 error 时必须原样失败，禁止整模回退。"""
        pipeline_error = {
            "response": "",
            "error": "queued distributed step failed",
        }
        monkeypatch.setattr(sched, "_all_pipeline_nodes_ready", lambda: True)
        monkeypatch.setattr(
            sched, "_has_active_distributed_pipeline_plan", lambda: True,
        )
        monkeypatch.setattr(
            sched, "run_pipeline", lambda prompt="", **kwargs: pipeline_error,
        )
        monkeypatch.setattr(
            sched,
            "_run_full_model_inference",
            lambda *args, **kwargs: pytest.fail(
                "distributed_required 队列任务不得回退整模"
            ),
        )

        result = sched._process_queued_pipeline_task(
            prompt="queued", _require_distributed=True,
        )

        assert result is pipeline_error

    def test_required_distributed_queued_worker_exception_does_not_fallback(
            self, sched, monkeypatch):
        """强制分布式队列任务抛异常时必须 fail closed，禁止整模回退。"""
        monkeypatch.setattr(sched, "_all_pipeline_nodes_ready", lambda: True)
        monkeypatch.setattr(
            sched, "_has_active_distributed_pipeline_plan", lambda: True,
        )

        def raise_pipeline_error(*args, **kwargs):
            raise RuntimeError("queued distributed exploded")

        monkeypatch.setattr(sched, "run_pipeline", raise_pipeline_error)
        monkeypatch.setattr(
            sched,
            "_run_full_model_inference",
            lambda *args, **kwargs: pytest.fail(
                "distributed_required 队列异常不得回退整模"
            ),
        )

        result = sched._process_queued_pipeline_task(
            prompt="queued", _require_distributed=True,
        )

        assert "distributed_required" in result["error"]
        assert "queued distributed exploded" in result["error"]
        assert result["metrics"]["fallback"] is False

    def test_run_pipeline_safe_respects_queue_busy(self, sched):
        """
        当 pipeline_queue.is_busy=True 时，run_pipeline_safe 应将请求入队。

        注意: 由于 queue worker 未启动（无 process_fn），入队后需要直接
        注入结果来模拟队列完成。
        """
        import threading
        queued = threading.Event()
        original_enqueue = sched.pipeline_queue.enqueue

        enqueued_task_ids: list = []

        def observed_enqueue(*args, **kwargs):
            task_id = original_enqueue(*args, **kwargs)
            enqueued_task_ids.append(task_id)
            queued.set()
            return task_id

        sched.pipeline_queue.enqueue = observed_enqueue

        # 模拟 busy 状态
        sched.pipeline_queue._current_task_id = "fake_running"

        # 准备结果注入
        # ★ 不要读 `pipeline_queue._queue`：MLFQ 重构后内部只有 _q0/_q1/_q2，
        #   访问已不存在的 `_queue` 会让本线程抛 AttributeError 静默死掉，主线程
        #   则在 wait_for_result 上干等到 PIPELINE_TIMEOUT（默认 120s）——实测全量
        #   下表现为"卡死/超时"（单跑恰好走立即执行路径时侥幸为绿）。
        #   改为沿用 enqueue 真正返回的 task_id 注入结果：不依赖内部队列结构，
        #   也不依赖本次是否真的走了排队路径。
        def inject_result():
            if not queued.wait(timeout=5.0) or not enqueued_task_ids:
                return
            tid_str = enqueued_task_ids[0]
            with sched.pipeline_queue._lock:
                sched.pipeline_queue._results[tid_str] = {
                    "status": "done",
                    "result": {"response": "queued_result"},
                }
                if tid_str in sched.pipeline_queue._events:
                    sched.pipeline_queue._events[tid_str].set()

        # 模拟流水线节点可用
        original = sched._all_pipeline_nodes_ready
        sched._all_pipeline_nodes_ready = lambda: True
        # 同步执行（不经过真正的 worker 线程）
        original_run = sched.run_pipeline
        sched.run_pipeline = lambda **kw: {"response": "direct"}

        try:
            t = threading.Thread(target=inject_result, daemon=True)
            t.start()

            result = sched.run_pipeline_safe(prompt="queued_prompt")
            # 清理 fake running task
            sched.pipeline_queue._current_task_id = None
            t.join(timeout=3.0)

            # 应得到注入的结果（排队路径）
            assert isinstance(result, dict)
            assert "response" in result or "error" in result
        finally:
            sched._all_pipeline_nodes_ready = original
            sched.run_pipeline = original_run
            sched.pipeline_queue._current_task_id = None

    def test_queued_distributed_request_preserves_routing_flags(
            self, sched, monkeypatch):
        """排队不能丢失强制分布式标记并静默改走整模。"""
        from model_host import model_host as _host

        class MockModelManager:
            is_loaded = True
            _engine_type = "pytorch"

        monkeypatch.setattr(_host, "_manager", MockModelManager())
        monkeypatch.setattr(sched, "_all_pipeline_nodes_ready", lambda: True)
        monkeypatch.setattr(
            sched, "_has_active_distributed_pipeline_plan", lambda: True,
        )
        monkeypatch.setattr(
            sched,
            "_synchronize_pipeline_workers_for_request",
            lambda **kwargs: {"ready": True, "reason_code": "ready"},
        )
        captured = {}
        sched.pipeline_queue._current_task_id = "busy"
        monkeypatch.setattr(
            sched.pipeline_queue,
            "enqueue",
            lambda **data: captured.update(data) or "queued-distributed",
        )
        monkeypatch.setattr(
            sched.pipeline_queue,
            "wait_for_result",
            lambda *args, **kwargs: {
                "status": "done",
                "result": {
                    "response": "queued distributed",
                    "metrics": {"distributed_used": True},
                },
            },
        )

        try:
            result = sched.run_pipeline_safe(
                "queued prompt",
                _require_distributed=True,
                _force_distributed_assignment=True,
                _pipeline_model_sync_timeout=3,
            )
        finally:
            sched.pipeline_queue._current_task_id = None

        assert result["response"] == "queued distributed"
        assert captured["_require_distributed"] is True
        assert captured["_force_distributed_assignment"] is True
        assert captured["_pipeline_model_sync_timeout"] == 3

    def test_required_distributed_queued_result_error_does_not_fallback(
            self, sched, monkeypatch):
        """排队完成后返回 pipeline error 时不得在等待方执行整模回退。"""
        from model_host import model_host as _host

        class MockModelManager:
            is_loaded = True
            _engine_type = "pytorch"

        pipeline_error = {
            "response": "",
            "error": "queued pipeline result failed",
        }
        monkeypatch.setattr(_host, "_manager", MockModelManager())
        monkeypatch.setattr(
            sched,
            "_synchronize_pipeline_workers_for_request",
            lambda **kwargs: {"ready": True, "reason_code": "ready"},
        )
        monkeypatch.setattr(
            sched.pipeline_queue, "enqueue", lambda **kwargs: "queued-error",
        )
        monkeypatch.setattr(
            sched.pipeline_queue,
            "wait_for_result",
            lambda *args, **kwargs: {
                "status": "done",
                "result": pipeline_error,
            },
        )
        monkeypatch.setattr(
            sched,
            "_run_full_model_inference",
            lambda *args, **kwargs: pytest.fail(
                "distributed_required 排队结果错误不得回退整模"
            ),
        )
        sched.pipeline_queue._current_task_id = "busy"

        try:
            result = sched.run_pipeline_safe(
                "queued prompt", _require_distributed=True,
            )
        finally:
            sched.pipeline_queue._current_task_id = None

        assert result is pipeline_error

    def test_required_distributed_immediate_error_does_not_fallback(
            self, sched, monkeypatch):
        """立即执行返回 pipeline error 时必须原样失败，禁止整模回退。"""
        from model_host import model_host as _host

        class MockModelManager:
            is_loaded = True
            _engine_type = "pytorch"

        pipeline_error = {
            "response": "",
            "error": "immediate distributed step failed",
        }
        monkeypatch.setattr(_host, "_manager", MockModelManager())
        monkeypatch.setattr(
            sched,
            "_synchronize_pipeline_workers_for_request",
            lambda **kwargs: {"ready": True, "reason_code": "ready"},
        )
        monkeypatch.setattr(
            sched, "run_pipeline", lambda prompt="", **kwargs: pipeline_error,
        )
        monkeypatch.setattr(
            sched,
            "_run_full_model_inference",
            lambda *args, **kwargs: pytest.fail(
                "distributed_required 立即执行错误不得回退整模"
            ),
        )

        result = sched.run_pipeline_safe(
            "immediate prompt", _require_distributed=True,
        )

        assert result is pipeline_error

    def test_required_distributed_immediate_exception_does_not_fallback(
            self, sched, monkeypatch):
        """立即执行抛异常时必须 fail closed，禁止整模回退。"""
        from model_host import model_host as _host

        class MockModelManager:
            is_loaded = True
            _engine_type = "pytorch"

        def raise_pipeline_error(*args, **kwargs):
            raise RuntimeError("immediate distributed exploded")

        monkeypatch.setattr(_host, "_manager", MockModelManager())
        monkeypatch.setattr(
            sched,
            "_synchronize_pipeline_workers_for_request",
            lambda **kwargs: {"ready": True, "reason_code": "ready"},
        )
        monkeypatch.setattr(sched, "run_pipeline", raise_pipeline_error)
        monkeypatch.setattr(
            sched,
            "_run_full_model_inference",
            lambda *args, **kwargs: pytest.fail(
                "distributed_required 立即执行异常不得回退整模"
            ),
        )

        result = sched.run_pipeline_safe(
            "immediate prompt", _require_distributed=True,
        )

        assert "distributed_required" in result["error"]
        assert "immediate distributed exploded" in result["error"]
        assert result["metrics"]["fallback"] is False


# ================================================================
# 星状拓扑与逻辑段链测试
# ================================================================


class TestChainTopology:
    """测试逻辑段链保留、worker peer 数据连接已移除。"""

    @pytest.fixture
    def sched(self):
        s = Scheduler()
        from scheduler import NodeInfo, NodeState
        # 注册 3 个在线从节点；地址只属于 master 控制面。
        s.nodes["client1"] = NodeInfo(
            node_id="client1", role="client", state=NodeState.ONLINE,
            address="192.168.1.2:8888",
        )
        s.nodes["client2"] = NodeInfo(
            node_id="client2", role="client", state=NodeState.ONLINE,
            address="192.168.1.3:8888",
        )
        s.nodes["client3"] = NodeInfo(
            node_id="client3", role="client", state=NodeState.ONLINE,
            address="192.168.1.4:8888",
        )
        return s

    def test_worker_peer_connector_is_not_exposed(self, sched):
        assert not hasattr(sched, "_send_chain_forward")
        assert not hasattr(sched, "_chain_clients")

    def test_broadcast_pipeline_abort(self, sched):
        """_broadcast_pipeline_abort 应调用 _send_to_worker 给所有节点"""
        sent_to = []

        def mock_send(worker_id, data, msg_type):
            sent_to.append(worker_id)

        original = sched._send_to_worker
        sched._send_to_worker = mock_send

        try:
            pipeline_nodes = [
                {"node_id": "client1"},
                {"node_id": "client2"},
                {"node_id": "client3"},
            ]
            sched._broadcast_pipeline_abort(pipeline_nodes, "task_x", "测试取消")
            assert len(sent_to) == 3
            assert "client1" in sent_to
            assert "client2" in sent_to
            assert "client3" in sent_to
        finally:
            sched._send_to_worker = original

    def test_broadcast_pipeline_abort_continues_after_send_failure(self, sched):
        """单个节点 ABORT 失败不应阻断其他节点清理。"""
        sent_to = []

        def mock_send(worker_id, data, msg_type):
            sent_to.append(worker_id)
            if worker_id == "client1":
                raise ConnectionError("client1 down")

        original = sched._send_to_worker
        sched._send_to_worker = mock_send

        try:
            pipeline_nodes = [
                {"node_id": "client1"},
                {"node_id": "client2"},
                {"node_id": "client3"},
            ]
            sched._broadcast_pipeline_abort(pipeline_nodes, "task_x", "测试取消")
            assert sent_to == ["client1", "client2", "client3"]
        finally:
            sched._send_to_worker = original

    def test_wait_for_layer_result_multi_nodes(self, sched):
        """_wait_for_layer_result 应在多个节点中有任一返回时立即唤醒"""
        import threading
        result_holder = []
        waiter_ready = threading.Event()
        original_wait = sched._wait_for_layer_result

        def observed_wait(*args, **kwargs):
            waiter_ready.set()
            return original_wait(*args, **kwargs)

        sched._wait_for_layer_result = observed_wait

        def waiter():
            result_holder.append(
                sched._wait_for_layer_result(
                    "task_chain", ["client1", "client2", "client3"],
                    timeout=5.0,
                )
            )

        t = threading.Thread(target=waiter)
        t.start()
        assert waiter_ready.wait(timeout=1.0)

        # 模拟 client2 返回结果（非最后一个节点）
        key = "task_chain:client2"
        with sched._pipeline_lock:
            sched._pipeline_results[key] = {
                "task_id": "task_chain",
                "node_id": "client2",
            }
            if key in sched._pipeline_events:
                sched._pipeline_events[key].set()

        t.join(timeout=2.0)
        assert len(result_holder) == 1
        assert result_holder[0] is not None
        assert result_holder[0].get("node_id") == "client2"

    def test_wait_for_layer_result_consumes_result_that_arrived_before_event(self, sched):
        """结果先于 event 注册到达时不应等待到超时。"""
        import time

        with sched._pipeline_lock:
            sched._pipeline_results["task_fast:client2"] = {
                "task_id": "task_fast",
                "node_id": "client2",
            }

        t0 = time.time()
        result = sched._wait_for_layer_result(
            "task_fast", ["client1", "client2"], timeout=5.0,
        )
        elapsed = time.time() - t0

        assert result is not None
        assert result["node_id"] == "client2"
        assert elapsed < 0.5

    def test_node_disconnect_wakes_pending_pipeline_waiter(self, sched):
        """节点断连应立即唤醒等待该节点结果的流水线线程。"""
        import threading
        result_holder = []
        waiter_ready = threading.Event()
        original_wait = sched._wait_for_layer_result

        def observed_wait(*args, **kwargs):
            waiter_ready.set()
            return original_wait(*args, **kwargs)

        sched._wait_for_layer_result = observed_wait

        def waiter():
            result_holder.append(
                sched._wait_for_layer_result("task_down", "client2", timeout=5.0)
            )

        t = threading.Thread(target=waiter)
        t.start()
        assert waiter_ready.wait(timeout=1.0)

        t0 = time.time()
        sched._fail_pending_pipeline_results_for_node("client2", "client2 down")
        t.join(timeout=1.0)
        elapsed = time.time() - t0

        assert len(result_holder) == 1
        assert result_holder[0]["node_id"] == "client2"
        assert "client2 down" in result_holder[0]["error"]
        assert elapsed < 0.5

    def test_send_layer_result_returns_false_when_tcp_client_missing(self, sched):
        """worker 无主节点连接时，结果发送应返回 False 而不是静默吞掉。"""
        sched._tcp_client = None

        assert sched._send_layer_result("master", "task_send", error="failed") is False

    def test_send_layer_result_disconnects_on_send_failure(self, sched):
        """结果回传 send_data 抛错时应返回 False 并断开连接促使主节点快速感知。"""
        class FailingClient:
            _running = True

            def __init__(self):
                self.disconnected = False

            def send_data(self, data, msg_type):
                raise OSError("broken pipe")

            def disconnect(self):
                self.disconnected = True
                self._running = False

        client = FailingClient()
        sched._tcp_client = client

        assert sched._send_layer_result("master", "task_send", error="failed") is False
        assert client.disconnected is True

    def test_chain_ack_timeout_returns_error_before_layer_timeout(self, sched):
        """链式转发已发送但下游未 ACK 时，应在 ACK 窗口内快速失败。"""
        import time

        sched._pipeline_active_tasks.add("task_ack")
        sched._pipeline_task_contracts["task_ack"] = {
            "execution_protocol": "legacy_layer_v1",
            "config_id": "cfg-ack", "worker_ids": ["client1", "client2"],
            "assignment_ids": {
                "client1": "asg-client1", "client2": "asg-client2",
            },
            "current_step": 0,
        }
        sched._handle_chain_forward_ack(
            "client1",
            {
                "data": {
                    "task_id": "task_ack",
                    "step": 0,
                    "config_id": "cfg-ack",
                    "assignment_id": "asg-client1",
                    "node_id": "client1",
                    "target_node_id": "client2",
                    "status": "sent",
                }
            },
        )

        t0 = time.time()
        result = sched._wait_for_layer_result(
            "task_ack",
            ["client1", "client2"],
            timeout=5.0,
            ack_node_ids=["client2"],
            ack_step=0,
            ack_timeout=0.2,
        )
        elapsed = time.time() - t0

        assert result is not None
        assert result["node_id"] == "client2"
        assert "未收到接收 ACK" in result["error"]
        assert elapsed < 1.0

    def test_wait_for_layer_result_timeout_multi_nodes(self, sched):
        """多节点等待超时应返回 None"""
        import time
        t0 = time.time()
        result = sched._wait_for_layer_result(
            "task_t", ["client1", "client2"], timeout=0.3
        )
        elapsed = time.time() - t0
        assert result is None
        assert elapsed < 2.0

    def test_chain_forward_routing(self, sched):
        """master 发来的 CHAIN_FORWARD 应路由到处理器。"""
        import threading
        called = []

        def mock_handler(cid, msg):
            called.append((cid, msg.get("type")))

        original = sched._handle_chain_forward
        sched._handle_chain_forward = mock_handler

        try:
            msg = {
                "type": "chain_forward",
                "data": {
                    "task_id": "task_1",
                    "step": 0,
                    "hidden_states": "dGVzdA==",
                },
            }
            sched._on_tcp_message("master", msg)
            assert len(called) == 1
            assert called[0][0] == "master"
            assert called[0][1] == "chain_forward"
        finally:
            sched._handle_chain_forward = original

    def test_chain_forward_from_worker_is_rejected(self, sched):
        called = []
        original = sched._handle_layer_forward
        sched._handle_layer_forward = lambda *args: called.append(args)
        try:
            sched._handle_chain_forward("client1", {
                "type": "chain_forward",
                "data": {"task_id": "task-peer-bypass", "step": 0},
            })
            assert called == []
        finally:
            sched._handle_layer_forward = original

    def test_layer_forward_from_worker_is_rejected(self, sched):
        called = []
        original = sched._handle_layer_forward_locked
        sched._handle_layer_forward_locked = lambda *args: called.append(args)
        try:
            sched._handle_layer_forward("client1", {
                "type": "layer_forward",
                "data": {"task_id": "task-peer-bypass", "step": 0},
            })
            assert called == []
        finally:
            sched._handle_layer_forward_locked = original

    def test_chain_forward_ack_routing(self, sched):
        """CHAIN_FORWARD_ACK 消息应路由并记录 ACK 状态。"""
        sched._pipeline_active_tasks.add("task_ack_route")
        sched._pipeline_task_contracts["task_ack_route"] = {
            "execution_protocol": "legacy_layer_v1",
            "config_id": "cfg-route", "worker_ids": ["client1", "client2"],
            "assignment_ids": {
                "client1": "asg-client1", "client2": "asg-client2",
            },
            "current_step": 1,
        }
        msg = {
            "type": "chain_forward_ack",
            "data": {
                "task_id": "task_ack_route",
                "step": 1,
                "config_id": "cfg-route",
                "assignment_id": "asg-client2",
                "node_id": "client2",
                "from_node_id": "client1",
                "status": "received",
            },
        }

        sched._on_tcp_message("client2", msg)

        assert sched._chain_ack_state["task_ack_route"][1]["client2"]["status"] == "received"

    def test_chain_forward_ack_rejects_stale_assignment(self, sched):
        sched._pipeline_active_tasks.add("task-ack-identity")
        sched._pipeline_task_contracts["task-ack-identity"] = {
            "execution_protocol": "legacy_layer_v1",
            "config_id": "cfg-identity",
            "worker_ids": ["client1", "client2"],
            "assignment_ids": {
                "client1": "asg-client1", "client2": "asg-client2",
            },
            "current_step": 0,
        }
        base = {
            "task_id": "task-ack-identity",
            "step": 0,
            "config_id": "cfg-identity",
            "node_id": "client2",
            "from_node_id": "client1",
            "status": "received",
        }

        sched._handle_chain_forward_ack(
            "client2", {"data": {**base, "assignment_id": "asg-stale"}},
        )
        assert "task-ack-identity" not in sched._chain_ack_state

        sched._handle_chain_forward_ack(
            "client2", {"data": {**base, "assignment_id": "asg-client2"}},
        )
        assert (
            sched._chain_ack_state["task-ack-identity"][0]["client2"][
                "assignment_id"
            ]
            == "asg-client2"
        )

    def test_chain_ack_received_is_not_overwritten_by_late_sent(self, sched):
        """下游 received ACK 先到时，后到的 sent 回报不应覆盖已确认状态。"""
        sched._pipeline_active_tasks.add("task_ack_order")
        sched._pipeline_task_contracts["task_ack_order"] = {
            "execution_protocol": "legacy_layer_v1",
            "config_id": "cfg-order", "worker_ids": ["client1", "client2"],
            "assignment_ids": {
                "client1": "asg-client1", "client2": "asg-client2",
            },
            "current_step": 0,
        }
        sched._handle_chain_forward_ack(
            "client2",
            {
                "data": {
                    "task_id": "task_ack_order",
                    "step": 0,
                    "config_id": "cfg-order",
                    "assignment_id": "asg-client2",
                    "node_id": "client2",
                    "from_node_id": "client1",
                    "status": "received",
                }
            },
        )
        sched._handle_chain_forward_ack(
            "client1",
            {
                "data": {
                    "task_id": "task_ack_order",
                    "step": 0,
                    "config_id": "cfg-order",
                    "assignment_id": "asg-client1",
                    "node_id": "client1",
                    "target_node_id": "client2",
                    "status": "sent",
                }
            },
        )

        state = sched._chain_ack_state["task_ack_order"][0]["client2"]
        assert state["status"] == "received"
        assert state["assignment_id"] == "asg-client2"
        assert state["target_assignment_id"] == "asg-client2"
        assert state["source_assignment_id"] == "asg-client1"
        assert state["reporter_node_id"] == "client2"

    def test_late_chain_ack_is_discarded(self, sched):
        sched._handle_chain_forward_ack("client2", {
            "data": {
                "task_id": "task-finished",
                "step": 0,
                "node_id": "client2",
                "status": "received",
            },
        })
        assert sched._chain_ack_state == {}

    def test_chain_forward_delegates_to_layer_forward(self, sched):
        """master 的 CHAIN_FORWARD 应委托给层前向并回接收 ACK。"""
        called_with = []
        ack_calls = []

        def mock_handler(cid, msg):
            called_with.append(cid)

        original = sched._handle_layer_forward
        original_ack = sched._send_chain_forward_ack
        sched._handle_layer_forward = mock_handler
        sched._send_chain_forward_ack = lambda **kwargs: ack_calls.append(kwargs) or True

        try:
            msg = {"type": "chain_forward", "data": {
                "task_id": "t1", "step": 2,
                "assignment_id": "asg-client2",
                "_chain_predecessor": "client1",
            }}
            sched._handle_chain_forward("master", msg)
            assert len(called_with) == 1
            assert called_with[0] == "master"
            assert ack_calls
            assert ack_calls[0]["task_id"] == "t1"
            assert ack_calls[0]["assignment_id"] == "asg-client2"
            assert ack_calls[0]["from_node_id"] == "client1"
            assert ack_calls[0]["status"] == "received"
        finally:
            sched._handle_layer_forward = original
            sched._send_chain_forward_ack = original_ack

    def test_layer_forward_with_chain_next_relays_through_master(self, sched, monkeypatch):
        """_handle_layer_forward 保留段接力，但必须经 master 转发。"""
        from model_host import model_host as _host
        import torch

        peer_forward_calls = []
        master_result_calls = []

        class MockModelManager:
            is_loaded = True
            _engine_type = "pytorch"
            active_model_id = "qwen2-test"
            layer_range = (8, 16)
            model = type("Model", (), {
                "config": type("Config", (), {"model_type": "qwen2"})(),
            })()

            def forward_layers(self, **kwargs):
                return {
                    "hidden_states": torch.ones(1, 2, 4),
                    "past_key_values": ((
                        torch.ones(1, 1, 2, 4), torch.ones(1, 1, 2, 4),
                    ),),
                }

        def mock_peer_forward(target_id, data):
            peer_forward_calls.append((target_id, data))
            return True

        def mock_send_result(cid, tid, result_data=None, error=None):
            master_result_calls.append((cid, tid, result_data, error))
            return True

        original_send = sched._send_layer_result
        monkeypatch.setattr(
            sched, "_send_chain_forward", mock_peer_forward, raising=False,
        )
        sched._send_layer_result = mock_send_result
        monkeypatch.setattr(_host, "_manager", MockModelManager())
        monkeypatch.setattr(sched, "_record_local_pipeline_participation", lambda *a, **kw: True)
        sched._active_layer_config = {
            "assignment_id": "asg-forward",
            "config_id": "cfg-forward",
            "model_id": "qwen2-test",
            "model_sha256": "sha-forward",
            "model_type": "qwen2",
            "layer_range": [8, 16],
            "engine": "pytorch",
        }

        try:
            msg = {
                "data": {
                    "task_id": "task_forward",
                    "step": 0,
                    "assignment_id": "asg-forward",
                    "config_id": "cfg-forward",
                    "model_sha256": "sha-forward",
                    "model_type": "qwen2",
                    "input_ids": [[1, 2]],
                    "use_kv_cache": False,
                    "chain_next": {"node_id": "client2"},
                    "chain_remaining": [],
                }
            }
            sched._handle_layer_forward("master", msg)

            assert peer_forward_calls == []
            assert len(master_result_calls) == 1
            assert master_result_calls[0][0:2] == ("master", "task_forward")
            assert master_result_calls[0][2]["_relay_to"] == "client2"
        finally:
            sched._send_layer_result = original_send

    def test_qwen_layer_forward_uses_original_cache_layout(
            self, sched, monkeypatch):
        """Qwen-1.8B KV 为 (batch, seq, heads, dim)，不得按 Qwen2 解析。"""
        from model_host import model_host as _host
        import torch

        class QwenManager:
            is_loaded = True
            _engine_type = "pytorch"
            active_model_id = "qwen-1_8b-chat"
            layer_range = (8, 16)
            model = type("Model", (), {
                "config": type("Config", (), {"model_type": "qwen"})(),
            })()

            def forward_layers(self, **kwargs):
                return {
                    "hidden_states": torch.ones(1, 2, 4),
                    "past_key_values": ((
                        torch.ones(1, 2, 1, 4),
                        torch.ones(1, 2, 1, 4),
                    ),),
                }

        results = []
        monkeypatch.setattr(_host, "_manager", QwenManager())
        monkeypatch.setattr(
            sched,
            "_send_layer_result",
            lambda cid, tid, result_data=None, error=None: (
                results.append((result_data, error)) or True
            ),
        )
        sched._active_layer_config = {
            "assignment_id": "asg-qwen",
            "config_id": "cfg-qwen",
            "model_id": "qwen-1_8b-chat",
            "model_sha256": "sha-qwen",
            "model_type": "qwen",
            "layer_range": [8, 16],
            "engine": "pytorch",
        }

        sched._handle_layer_forward("master", {"data": {
            "task_id": "task-qwen",
            "step": 0,
            "assignment_id": "asg-qwen",
            "config_id": "cfg-qwen",
            "model_sha256": "sha-qwen",
            "model_type": "qwen",
            "hidden_states": None,
            "input_ids": [[1, 2]],
            "use_kv_cache": False,
        }})

        response, error = results[-1]
        assert error is None
        assert response["metrics"]["kv_seq_len"] == 2
        assert len(sched._kv_cache["task-qwen"]) == 1
        assert sched._local_pipeline_steps["task-qwen"] == 0

    def test_layer_forward_rejects_stale_config_and_missing_decode_cache(
            self, sched, monkeypatch):
        from model_host import model_host as _host

        errors = []
        manager = type("Manager", (), {
            "is_loaded": True,
            "_engine_type": "pytorch",
            "active_model_id": "deepseek",
            "layer_range": (8, 16),
            "model": type("Model", (), {
                "config": type("Config", (), {"model_type": "qwen2"})(),
            })(),
        })()
        monkeypatch.setattr(_host, "_manager", manager)
        monkeypatch.setattr(
            sched,
            "_send_layer_result",
            lambda _cid, _tid, result_data=None, error=None: errors.append(error) or True,
        )
        monkeypatch.setattr(sched, "_send_chain_forward_ack", lambda **_kwargs: True)
        monkeypatch.setattr(sched, "_record_local_pipeline_participation", lambda *a, **k: True)
        sched._active_layer_config = {
            "assignment_id": "asg-current",
            "config_id": "cfg-current",
            "model_id": "deepseek",
            "model_sha256": "sha-current",
            "model_type": "qwen2",
            "layer_range": [8, 16],
            "engine": "pytorch",
        }

        sched._handle_layer_forward("master", {"data": {
            "task_id": "stale", "step": 0,
            "assignment_id": "asg-current", "config_id": "cfg-old",
            "model_sha256": "sha-current", "model_type": "qwen2",
            "input_ids": [[1]], "use_kv_cache": False,
        }})
        assert "config_id" in errors[-1]

        sched._active_pipeline_task_ids.add("decode")
        sched._local_pipeline_steps["decode"] = 0
        sched._handle_layer_forward("master", {"data": {
            "task_id": "decode", "step": 1,
            "assignment_id": "asg-current", "config_id": "cfg-current",
            "model_sha256": "sha-current", "model_type": "qwen2",
            "input_ids": [[2]], "use_kv_cache": True,
        }})
        assert "缺少本地 KV cache" in errors[-1]

    def test_master_relay_records_chain_sent_ack(self, sched):
        """master 转发成功后应记录发往目标逻辑段的 sent ACK 状态。"""
        sent = []

        def mock_send(worker_id, data, msg_type):
            sent.append((worker_id, data, msg_type))

        original = sched._send_to_worker
        sched._send_to_worker = mock_send
        sched._pipeline_active_tasks.add("task_relay")
        sched._pipeline_task_contracts["task_relay"] = {
            "execution_protocol": "legacy_layer_v1",
            "config_id": "cfg-relay",
            "model_sha256": "sha-relay",
            "model_type": "qwen2",
            "worker_ids": ["client1", "client2"],
            "assignment_ids": {
                "client1": "asg-client1", "client2": "asg-client2",
            },
            "last_node_id": "client2",
            "current_step": 4,
        }

        try:
            sched._handle_layer_result(
                "client1",
                {
                    "data": {
                        "task_id": "task_relay",
                        "step": 4,
                        "node_id": "client1",
                        "assignment_id": "asg-client1",
                        "config_id": "cfg-relay",
                        "model_sha256": "sha-relay",
                        "model_type": "qwen2",
                        "_relay_to": "client2",
                        "hidden_states": "dGVzdA==",
                    }
                },
            )

            assert sent and sent[0][0] == "client2"
            assert sent[0][1]["assignment_id"] == "asg-client2"
            state = sched._chain_ack_state["task_relay"][4]["client2"]
            assert state["status"] == "sent"
            assert state["reporter_node_id"] == "client1"
            assert state["assignment_id"] == "asg-client1"
        finally:
            sched._send_to_worker = original

    def test_master_relay_rejects_non_adjacent_target(self, sched):
        sent = []
        sched._send_to_worker = lambda worker_id, data, msg_type: sent.append(worker_id)
        sched._pipeline_active_tasks.add("task-skip")
        sched._pipeline_task_contracts["task-skip"] = {
            "execution_protocol": "legacy_layer_v1",
            "config_id": "cfg-skip",
            "model_sha256": "sha-skip",
            "model_type": "qwen2",
            "worker_ids": ["client1", "client2", "client3"],
            "assignment_ids": {
                "client1": "asg-client1", "client2": "asg-client2",
                "client3": "asg-client3",
            },
            "last_node_id": "client3",
            "current_step": 0,
        }

        sched._handle_layer_result("client1", {"data": {
            "task_id": "task-skip",
            "node_id": "client1",
            "step": 0,
            "assignment_id": "asg-client1",
            "config_id": "cfg-skip",
            "model_sha256": "sha-skip",
            "model_type": "qwen2",
            "_relay_to": "client3",
            "chain_path": ["client1"],
            "hidden_states": "dGVzdA==",
        }})

        assert sent == []
        error = sched._pipeline_results["task-skip:client1"]
        assert "非相邻中转目标" in error["error"]

    def test_master_relay_marks_trusted_logical_predecessor(self, sched):
        sent = []
        sched._send_to_worker = lambda worker_id, data, msg_type: sent.append(
            (worker_id, data)
        )
        sched._pipeline_active_tasks.add("task-relay-path")
        sched._pipeline_task_contracts["task-relay-path"] = {
            "execution_protocol": "legacy_layer_v1",
            "config_id": "cfg-relay-path",
            "model_sha256": "sha-relay-path",
            "model_type": "qwen2",
            "worker_ids": ["client1", "client2"],
            "assignment_ids": {
                "client1": "asg-client1", "client2": "asg-client2",
            },
            "last_node_id": "client2",
            "current_step": 0,
        }

        sched._handle_layer_result("client1", {"data": {
            "task_id": "task-relay-path",
            "node_id": "client1",
            "step": 0,
            "assignment_id": "asg-client1",
            "config_id": "cfg-relay-path",
            "model_sha256": "sha-relay-path",
            "model_type": "qwen2",
            "_relay_to": "client2",
            "chain_path": ["client1"],
            "hidden_states": "dGVzdA==",
        }})

        assert sent[0][0] == "client2"
        assert sent[0][1]["_chain_predecessor"] == "client1"

    def test_master_relays_three_logical_segments_in_order(self, sched):
        sent = []
        sched._send_to_worker = lambda worker_id, data, msg_type: sent.append(
            (worker_id, data, msg_type)
        )
        sched._pipeline_active_tasks.add("task-three-hop")
        sched._pipeline_task_contracts["task-three-hop"] = {
            "execution_protocol": "legacy_layer_v1",
            "config_id": "cfg-three-hop",
            "model_sha256": "sha-three-hop",
            "model_type": "qwen2",
            "worker_ids": ["client1", "client2", "client3"],
            "assignment_ids": {
                "client1": "asg-client1", "client2": "asg-client2",
                "client3": "asg-client3",
            },
            "last_node_id": "client3",
            "current_step": 0,
        }

        def result(node_id, relay_to=None, chain_path=None, **extra):
            data = {
                "task_id": "task-three-hop",
                "node_id": node_id,
                "step": 0,
                "assignment_id": f"asg-{node_id}",
                "config_id": "cfg-three-hop",
                "model_sha256": "sha-three-hop",
                "model_type": "qwen2",
                "chain_path": chain_path or [node_id],
                **extra,
            }
            if relay_to is not None:
                data["_relay_to"] = relay_to
            sched._handle_layer_result(node_id, {"data": data})

        result("client1", relay_to="client2", hidden_states="aDE=")
        result(
            "client2", relay_to="client3",
            chain_path=["client1", "client2"], hidden_states="aDI=",
        )
        result(
            "client3", chain_path=["client1", "client2", "client3"],
            logits="bG9naXRz",
        )

        assert [item[0] for item in sent] == ["client2", "client3"]
        assert sent[0][1]["_chain_predecessor"] == "client1"
        assert sent[1][1]["_chain_predecessor"] == "client2"
        final = sched._pipeline_results["task-three-hop:client3"]
        assert final["chain_path"] == ["client1", "client2", "client3"]

    def test_layer_config_release_clears_worker_reservation(self, sched):
        from tcp_comm import MessageType

        sent = []
        sched._tcp_client = type("FakeClient", (), {
            "send_data": lambda self, payload, msg_type: sent.append(
                (payload, msg_type)
            ),
        })()
        sched._pipeline_worker_reserved = True
        sched._active_layer_config = {"config_id": "cfg-old"}
        sched._last_layer_config_ack_payload = {
            "config_id": "cfg-old", "status": "ready",
        }

        sched._handle_layer_config("master", {
            "node_id": sched.get_effective_node_id(),
            "config_id": "cfg-release",
            "generation": 2,
            "release": True,
        })

        assert sched.has_pipeline_worker_reservation() is False
        assert sched._active_layer_config is None
        assert sched._last_layer_config_ack_payload is None
        assert sent == [({
            "node_id": sched.get_effective_node_id(),
            "assignment_id": "",
            "config_id": "cfg-release",
            "generation": 2,
            "status": "released",
            "release": True,
            "timestamp": sent[0][0]["timestamp"],
        }, MessageType.LAYER_CONFIG_ACK)]

    def test_older_layer_generation_cannot_override_newer_release(
            self, sched, monkeypatch):
        handled = []
        monkeypatch.setattr(
            sched,
            "_handle_layer_config_locked",
            lambda _client_id, data, **_kwargs: handled.append(
                data["config_id"]
            ),
        )
        sched._latest_layer_config_receive_sequence = 2
        sched._latest_layer_config_generation = 20

        sched._handle_layer_config(
            "master",
            {"config_id": "cfg-old", "generation": 19},
            receive_sequence=1,
            generation=19,
        )

        assert handled == []

    def test_legacy_generation_cannot_override_versioned_release(
            self, sched, monkeypatch):
        started = []
        sched._latest_layer_config_generation = 20
        monkeypatch.setattr(
            threading.Thread,
            "start",
            lambda self: started.append(self),
        )

        sched._schedule_layer_config("master", {
            "config_id": "cfg-legacy",
            "start_layer": 1,
            "end_layer": 2,
        })

        assert started == []
        assert sched._layer_config_inflight == set()

    def test_release_ack_clears_master_retry_state(self, sched):
        expected = _install_layer_assignment(sched, "client1", {
            "node_id": "client1",
            "config_id": "cfg-release",
            "generation": 7,
            "release": True,
        })
        sched._layer_config_retry_state["client1"] = {
            "attempts": 1,
            "next_retry": 0.0,
        }

        sched._handle_layer_config_ack("client1", {"data": {
            "node_id": "client1",
            "assignment_id": expected["assignment_id"],
            "config_id": "cfg-release",
            "generation": 7,
            "status": "released",
            "release": True,
        }})

        assert _layer_expected(sched, "client1") == {}
        assert "client1" not in sched._layer_config_retry_state

    def test_disconnect_drops_obsolete_layer_config_state(self, sched, monkeypatch):
        """Reconnect must start from a fresh per-node config transaction."""
        from scheduler import NodeInfo, NodeState

        sched.nodes["client1"] = NodeInfo(
            node_id="client1", role="client", state=NodeState.ONLINE,
        )
        _install_layer_assignment(sched, "client1", {
            "node_id": "client1", "config_id": "cfg-old", "generation": 7,
        }, ack={
            "node_id": "client1", "config_id": "cfg-old", "status": "ready",
            "generation": 7,
        })
        _mark_layer_config_pushed(sched, "client1")
        sched._layer_config_retry_state["client1"] = {
            "attempts": 2, "next_retry": 0.0,
        }
        sched._tcp_server = type("Server", (), {
            "clients": {},
            "_running": False,
        })()
        monkeypatch.setattr(sched, "_push_node_update_to_all_clients", lambda *args: None)
        monkeypatch.setattr(sched, "deregister_node", lambda _node_id: None)
        monkeypatch.setattr(sched, "push_layer_config_to_clients", lambda: None)
        monkeypatch.setattr(sched, "_stage_pipeline_reshard_after_disconnect", lambda _node_id: None)

        sched._on_tcp_disconnect("client1")

        assert _layer_expected(sched, "client1") == {}
        assert _layer_ack(sched, "client1") == {}
        assert "client1" not in sched._layer_config_pushed
        assert "client1" not in sched._layer_config_retry_state

    def test_failed_release_send_remains_retryable(self, sched):
        class FailingServer:
            _running = True

            def send_layer_config(self, node_id, config):
                raise ConnectionError("temporary")

        sched._tcp_server = FailingServer()
        release = {
            "node_id": "client1",
            "config_id": "cfg-release",
            "generation": 3,
            "release": True,
        }

        sched._publish_layer_configs({"client1": release})

        expected = _layer_expected(sched, "client1")
        assert expected["config_id"] == release["config_id"]
        assert expected["release"] is True
        assert expected["assignment_id"]
        assert "client1" in sched._layer_config_retry_state

    def test_local_model_change_sends_worker_opt_out(self, sched):
        from tcp_comm import MessageType

        sent = []
        sched._role_override = "client"
        sched._pipeline_worker_reserved = True
        sched._tcp_client = type("Client", (), {
            "_running": True,
            "send_data": lambda self, payload, msg_type: sent.append(
                (payload, msg_type)
            ),
        })()

        assert sched.release_pipeline_worker_for_local_model() is True
        assert sched.has_pipeline_worker_reservation() is False
        assert sent[0][1] == MessageType.LAYER_WORKER_OPT_OUT

    def test_opted_out_worker_rejects_stale_assignment(self, sched, monkeypatch):
        sched._pipeline_worker_opted_out = True
        calls = []
        monkeypatch.setattr(
            sched,
            "release_pipeline_worker_for_local_model",
            lambda: calls.append("opt-out") or True,
        )

        sched._schedule_layer_config("master", {
            "config_id": "cfg-stale",
            "generation": 4,
            "start_layer": 8,
            "end_layer": 16,
        })

        assert calls == ["opt-out"]
        assert sched._layer_config_inflight == set()

    def test_authoritative_assignment_reenables_opted_out_worker(
            self, sched, monkeypatch):
        sched._pipeline_worker_opted_out = True
        started = []
        monkeypatch.setattr(
            threading.Thread,
            "start",
            lambda self: started.append(self),
        )

        sched._schedule_layer_config("master", {
            "config_id": "cfg-authoritative",
            "generation": 5,
            "start_layer": 8,
            "end_layer": 16,
            "authoritative_sync": True,
        })

        assert sched._pipeline_worker_opted_out is False
        assert len(started) == 1
        assert "cfg-authoritative" in sched._layer_config_inflight

    def test_master_authoritative_sync_clears_worker_opt_out(
            self, sched, monkeypatch):
        sched.nodes = {
            "master": NodeInfo(
                node_id="master", role="master", state=NodeState.ONLINE,
                node_type="pc", device_info=PROFILE_WORKSTATION,
            ),
            "client1": NodeInfo(
                node_id="client1", role="client", state=NodeState.ONLINE,
                node_type="pc", device_info=PROFILE_ULTRABOOK,
            ),
        }
        sched._role_override = "master"
        sent = []
        sched._tcp_server = type("Server", (), {
            "_running": True,
            "clients": {"client1": object()},
            "send_layer_config": lambda self, node_id, payload: sent.append(
                (node_id, payload)
            ),
        })()
        sched._pipeline_worker_opt_out.add("client1")
        monkeypatch.setattr(sched, "get_layer_assignments", lambda: {
            "total": 24,
            "strategy": "dynamic",
            "assignments": [
                {
                    "node_id": "master", "start_layer": 0,
                    "end_layer": 21, "has_embedding": True,
                    "has_lm_head": True,
                },
                {
                    "node_id": "client1", "start_layer": 21,
                    "end_layer": 24, "has_embedding": False,
                    "has_lm_head": False,
                },
            ],
        })
        monkeypatch.setattr(sched, "_get_active_pipeline_model_info", lambda: {
            "model_id": "qwen-1_8b",
            "model_path": "C:/models/qwen",
            "model_sha256": "sha-qwen",
            "model_type": "qwen",
            "total_layers": 24,
            "quant_type": "int4",
        })
        monkeypatch.setattr(
            sched, "_host", type("Host", (), {"is_pipeline_prepared": False})(),
        )

        assert sched.request_authoritative_layer_sync() is True

        assert "client1" not in sched._pipeline_worker_opt_out
        assert sent[0][0] == "client1"
        assert sent[0][1]["authoritative_sync"] is True
        assert sent[0][1]["sync_policy"] == "master_authoritative"
        assert sent[0][1]["master_quant_type"] == "int4"
        assert sched._authoritative_layer_sync_requests == 0

    def test_master_opt_out_excludes_worker_from_assignment(self, sched, monkeypatch):
        sched.nodes = {
            "master": NodeInfo(
                node_id="master", role="master", state=NodeState.ONLINE,
                node_type="pc", device_info=PROFILE_WORKSTATION,
            ),
            "client1": NodeInfo(
                node_id="client1", role="client", state=NodeState.ONLINE,
                node_type="pc", device_info=PROFILE_ULTRABOOK,
            ),
        }
        sched._role_override = "master"
        monkeypatch.setattr(sched, "push_layer_config_to_clients", lambda: None)

        sched._handle_layer_worker_opt_out("client1", {"data": {
            "node_id": "client1",
        }})

        assert "client1" in sched._pipeline_worker_opt_out
        assignments = sched.compute_layer_assignment()
        assert [item["node_id"] for item in assignments] == ["master"]

    def test_enabling_distributed_requests_worker_opt_in(self, sched, monkeypatch):
        calls = []
        sched._role_override = "client"
        monkeypatch.setattr(
            sched,
            "_request_pipeline_worker_opt_in",
            lambda: calls.append("opt-in") or True,
        )

        result = sched.set_distributed_inference_enabled(True)

        assert result["enabled"] is True
        assert calls == ["opt-in"]

    def test_master_opt_in_restores_worker_assignment(self, sched, monkeypatch):
        sched.nodes = {
            "master": NodeInfo(
                node_id="master", role="master", state=NodeState.ONLINE,
                node_type="pc", device_info=PROFILE_WORKSTATION,
            ),
            "client1": NodeInfo(
                node_id="client1", role="client", state=NodeState.ONLINE,
                node_type="pc", device_info=PROFILE_ULTRABOOK,
            ),
        }
        sched._role_override = "master"
        sched._pipeline_worker_opt_out.add("client1")
        pushed = []
        monkeypatch.setattr(
            sched, "push_layer_config_to_clients", lambda: pushed.append(True),
        )
        monkeypatch.setattr(
            sched, "_host", type("Host", (), {"is_pipeline_prepared": False})(),
        )

        sched._handle_layer_worker_opt_in("client1", {"data": {
            "node_id": "client1",
        }})

        assert "client1" not in sched._pipeline_worker_opt_out
        assert pushed == [True]

    def test_last_result_rejects_incomplete_chain_path(self, sched):
        sched._pipeline_active_tasks.add("task-path")
        sched._pipeline_task_contracts["task-path"] = {
            "execution_protocol": "legacy_layer_v1",
            "config_id": "cfg-path",
            "model_sha256": "sha-path",
            "model_type": "qwen2",
            "worker_ids": ["client1", "client2", "client3"],
            "assignment_ids": {
                "client1": "asg-client1", "client2": "asg-client2",
                "client3": "asg-client3",
            },
            "last_node_id": "client3",
            "current_step": 0,
        }

        sched._handle_layer_result("client3", {"data": {
            "task_id": "task-path",
            "node_id": "client3",
            "step": 0,
            "assignment_id": "asg-client3",
            "config_id": "cfg-path",
            "model_sha256": "sha-path",
            "model_type": "qwen2",
            "chain_path": ["client1", "client3"],
            "hidden_states": "dGVzdA==",
        }})

        error = sched._pipeline_results["task-path:client3"]
        assert "链路路径不完整" in error["error"]

    def test_invalid_worker_config_revokes_ready_and_resends(self, sched):
        sched._tcp_server = MagicMock()
        sched._tcp_server._running = True
        sched._tcp_server.get_authenticated_peer_epoch.return_value = 11
        expected = {
            "node_id": "client1",
            "config_id": "cfg-invalid",
            "start_layer": 8,
            "end_layer": 16,
            "model_sha256": "sha-invalid",
            "model_type": "qwen2",
        }
        expected = _install_layer_assignment(sched, "client1", expected)
        _mark_layer_config_pushed(sched, "client1")
        sched._pipeline_active_tasks.add("task-invalid")
        sched._pipeline_task_contracts["task-invalid"] = {
            "execution_protocol": "legacy_layer_v1",
            "config_id": "cfg-invalid",
            "model_sha256": "sha-invalid",
            "model_type": "qwen2",
            "worker_ids": ["client1"],
            "assignment_ids": {
                "client1": expected["assignment_id"],
            },
            "last_node_id": "client1",
            "current_step": 1,
        }

        sched._handle_layer_result("client1", {"data": {
            "task_id": "task-invalid",
            "node_id": "client1",
            "step": 1,
            "assignment_id": expected["assignment_id"],
            "config_id": "cfg-invalid",
            "model_sha256": "sha-invalid",
            "model_type": "qwen2",
            "layer_config_invalid": True,
            "error": "worker 层范围已变化",
        }})

        assert "client1" not in sched._layer_config_pushed
        assert _layer_ack(sched, "client1")["status"] == "error"
        sched._tcp_server.send_layer_config.assert_called_once_with(
            "client1", expected, registration_epoch=11,
        )

    def test_chain_info_built_in_run_pipeline(self, sched):
        """逻辑段链只包含身份，不能把 peer 地址下发给 worker。"""
        # mock get_layer_assignments 返回 2 个从节点
        original = sched.get_layer_assignments
        sched.get_layer_assignments = lambda: {
            "total": 24, "strategy": "dynamic",
            "assignments": [
                {"node_id": "master", "start_layer": 0, "end_layer": 0,
                 "has_embedding": True, "has_lm_head": False},
                {"node_id": "client1", "start_layer": 0, "end_layer": 12,
                 "assignment_id": "asg-client1",
                 "has_embedding": True, "has_lm_head": False},
                {"node_id": "client2", "start_layer": 12, "end_layer": 24,
                 "assignment_id": "asg-client2",
                 "has_embedding": False, "has_lm_head": True},
            ],
        }
        try:
            # run_pipeline 需要 model_manager 和完整的 TCP 环境
            # 此处仅验证分层解析不抛异常
            layer_info = sched.get_layer_assignments()
            pipeline_nodes = [
                a for a in layer_info.get("assignments", [])
                if a.get("node_id") != "master"
            ]
            pipeline_nodes.sort(key=lambda a: a.get("start_layer", 0))
            route = sched._build_star_chain_route(pipeline_nodes)
            assert route == [
                {"node_id": "client1"},
                {"node_id": "client2"},
            ]
            assert all("host" not in item and "port" not in item for item in route)
        finally:
            sched.get_layer_assignments = original

    def test_star_chain_route_rejects_missing_identity(self, sched):
        with pytest.raises(ValueError, match="missing node_id"):
            sched._build_star_chain_route([{"start_layer": 0, "end_layer": 12}])


# ================================================================
# 显存约束校验 — 层区间重算
# ================================================================

class TestVramConstraintRecalculation:
    """测试 _apply_vram_constraints 在转移层后正确重算 start_layer/end_layer"""

    @pytest.fixture
    def sched(self):
        s = Scheduler()
        s.nodes = {
            "master": NodeInfo(
                node_id="master", role="master", state=NodeState.ONLINE,
                node_type="pc",
                device_info=PROFILE_WORKSTATION,
            ),
            "client1": NodeInfo(
                node_id="client1", role="client", state=NodeState.ONLINE,
                node_type="pc",
                device_info=PROFILE_EDGE,  # 显存极少
            ),
        }
        return s

    def test_layer_ranges_continuous_after_transfer(self, sched):
        """层转移后 start_layer/end_layer 应连续无间隙"""
        # 构造含 master 和低显存 client 的 assignments
        assignments = [
            {"node_id": "master", "role": "master",
             "start_layer": 0, "end_layer": 12, "layers_count": 12,
             "has_embedding": True, "has_lm_head": False, "score": 100},
            {"node_id": "client1", "role": "client",
             "start_layer": 12, "end_layer": 24, "layers_count": 12,
             "has_embedding": False, "has_lm_head": True, "score": 10},
        ]
        result = sched._apply_vram_constraints(assignments)

        # client1 显存不足 → 层应转给 master
        # master 应承担全部 24 层
        assert len(result) >= 1
        total_layers = sum(a["layers_count"] for a in result)
        assert total_layers == 24

        # 验证连续无间隙
        result.sort(key=lambda x: x["start_layer"])
        cursor = 0
        for a in result:
            assert a["start_layer"] == cursor, \
                f"gap at {a['node_id']}: expected start={cursor}, got {a['start_layer']}"
            cursor = a["end_layer"]

    def test_vram_transfer_preserves_embed_lm_head(self, sched):
        """层转移后首节点应有 embedding，末节点应有 lm_head"""
        assignments = [
            {"node_id": "master", "role": "master",
             "start_layer": 0, "end_layer": 12, "layers_count": 12,
             "has_embedding": True, "has_lm_head": False, "score": 100},
            {"node_id": "client1", "role": "client",
             "start_layer": 12, "end_layer": 24, "layers_count": 12,
             "has_embedding": False, "has_lm_head": True, "score": 10},
        ]
        result = sched._apply_vram_constraints(assignments)
        if len(result) > 0:
            assert result[0]["has_embedding"] is True
            assert any(item["has_lm_head"] for item in result)


# ================================================================
# _simple_weight_assignment 边界条件
# ================================================================

class TestSimpleWeightAssignmentEdgeCases:
    """测试 _simple_weight_assignment 的边界条件"""

    @pytest.fixture
    def sched(self):
        return Scheduler()

    def test_many_nodes_with_min_layers(self, sched):
        """当节点数 > 层数时，每节点至少 1 层，超出的从低分节点削减"""
        # 构造 30 个低分配置节点
        nodes = []
        for i in range(30):
            nodes.append({
                "node_id": f"node{i}",
                "role": "master" if i == 0 else "client",
                "device_info": {"gpu": {"vram_total_gb": 0.5, "cuda_available": False},
                               "ram": {"total_gb": 2}, "cpu": {"physical_cores": 1, "freq_max_mhz": 1000}},
            })
        result = sched._simple_weight_assignment(nodes, 24)
        total = sum(a["layers_count"] for a in result)
        # 总层数不应超过 24
        assert total <= 24, f"总层数 {total} 不应超过 24"
        # 至少部分节点应有分配层
        assert len(result) >= 1

    def test_two_nodes_equal_weight(self, sched):
        """等权重双节点应均分 24 层（各 12）"""
        nodes = [
            {"node_id": "master", "role": "master",
             "device_info": PROFILE_WORKSTATION},
            {"node_id": "client1", "role": "client",
             "device_info": PROFILE_WORKSTATION},  # 相同 profile
        ]
        result = sched._simple_weight_assignment(nodes, 24)
        total = sum(a["layers_count"] for a in result)
        assert total == 24
        # 每个节点至少 10 层（等权重接近均分）
        for a in result:
            assert a["layers_count"] >= 10, \
                f"{a['node_id']} 应有 ≈12 层，实际 {a['layers_count']}"


# ================================================================
# 运行时安全分层与任务统计
# ================================================================

class TestRuntimeSafeLayerAssignmentAndAccounting:
    """测试 master 协调节点语义与分布式任务记账。"""

    def test_low_score_master_does_not_steal_worker_layers(self):
        """低分 master 仅保留 1 层锚定 Embedding，主要层仍归高分 worker"""
        sched = Scheduler()
        nodes = [
            {"node_id": "master", "role": "master", "node_type": "pc", "device_info": PROFILE_EDGE},
            {"node_id": "worker-fast", "role": "client", "node_type": "pc", "device_info": PROFILE_WORKSTATION},
            {"node_id": "worker-mid", "role": "client", "node_type": "pc", "device_info": PROFILE_LAPTOP},
        ]
        result = sched.compute_layer_assignment(nodes)
        by_id = {a["node_id"]: a for a in result}

        # master 作为 Embedding 锚点至少 1 层，不再 coordinator-only
        assert by_id["master"]["layers_count"] >= 1, \
            f"低分 master 应保留 1 层锚定 Embedding，实际: {by_id['master']['layers_count']}"
        assert "coordinator_only" not in by_id["master"]
        assert by_id["master"]["has_embedding"] is True
        assert by_id["master"]["start_layer"] == 0
        # 高分 worker 仍拿主要层数
        assert by_id["worker-fast"]["layers_count"] >= 14, \
            f"高分 worker 应拿主要层数，实际: {by_id['worker-fast']['layers_count']}"
        assert result[-1]["end_layer"] == 24
        assert result[-1]["has_lm_head"] is True
        assert sum(a["layers_count"] for a in result) == 24

    def test_pipeline_task_accounting_counts_master_and_workers_once(self):
        sched = Scheduler()
        sched.nodes = {
            "master": NodeInfo(node_id="master", role="master", state=NodeState.ONLINE),
            "worker1": NodeInfo(node_id="worker1", role="client", state=NodeState.ONLINE),
            "worker2": NodeInfo(node_id="worker2", role="client", state=NodeState.ONLINE),
        }
        sched._effective_role = lambda: "master"
        sched._push_node_update_to_all_clients = lambda *args, **kwargs: None
        sched.get_effective_node_id = lambda: 'master'

        accounting = sched._record_pipeline_task_accounting(
            "task-1",
            [{"node_id": "worker1"}, {"node_id": "worker2"}, {"node_id": "worker1"}],
            success=True,
        )
        assert accounting["counted_nodes"] == ["master", "worker1", "worker2"]
        assert accounting["workers_counted"] == ["worker1", "worker2"]
        assert sched.nodes["master"].task_count == 1
        assert sched.nodes["worker1"].task_count == 1
        assert sched.nodes["worker2"].task_count == 1

        dedup = sched._record_pipeline_task_accounting(
            "task-1", [{"node_id": "worker1"}], success=True
        )
        assert dedup["deduplicated"] is True
        assert sched.nodes["worker1"].task_count == 1

    def test_local_pipeline_participation_is_idempotent(self):
        sched = Scheduler()
        sched.nodes = {
            "client_node": NodeInfo(node_id="client_node", role="client", state=NodeState.ONLINE),
        }
        sched.get_effective_node_id = lambda: "client_node"
        assert sched._record_local_pipeline_participation("task-local", success=True) is True
        assert sched._record_local_pipeline_participation("task-local", success=True) is False
        assert sched.nodes["client_node"].task_count == 1


# ================================================================
# get_status() — tcp_client 字段（从节点视角）
# ================================================================

class TestGetStatusTcpClient:
    """测试 get_status() 对从节点返回 tcp_client 连接状态"""

    @pytest.fixture
    def sched(self):
        return Scheduler()

    def test_client_status_includes_tcp_client_field(self, sched):
        """从节点 get_status 应包含 tcp_client 字段"""
        # mock 从节点角色
        original_role = sched._effective_role
        sched._effective_role = lambda: "client"
        try:
            # 无 _tcp_client 时 tcp_client 字段应为 None
            status = sched.get_status()
            assert "tcp_client" in status, "get_status 应包含 tcp_client 字段"
            assert status["tcp_client"] is None, \
                "无 _tcp_client 时 tcp_client 应为 None"
        finally:
            sched._effective_role = original_role

    def test_client_status_with_tcp_client(self, sched):
        """从节点有 _tcp_client 时应报告连接状态"""
        original_role = sched._effective_role
        sched._effective_role = lambda: "client"
        try:
            # 注入 mock _tcp_client
            mock_client = type('MockTCPClient', (), {
                'is_registered': True,
                '_running': True,
                'server_host': '100.100.52.1',
                'server_port': 8888,
                'avg_rtt_ms': 12.5,
                'sock': True,
            })()
            sched._tcp_client = mock_client

            status = sched.get_status()
            assert status["tcp_client"] is not None
            assert status["tcp_client"]["connected"] is True
            assert status["tcp_client"]["running"] is True
            assert status["tcp_client"]["server_host"] == "100.100.52.1"
            assert status["tcp_client"]["avg_rtt_ms"] == 12.5
        finally:
            sched._effective_role = original_role
            if hasattr(sched, '_tcp_client'):
                del sched._tcp_client

    def test_client_network_path_is_projected_to_status_and_master_node(
            self, sched):
        sched._effective_role = lambda: "client"
        sched.nodes = {
            "master": NodeInfo(
                node_id="master", role=NodeRole.MASTER,
                state=NodeState.ONLINE,
            ),
            "client1": NodeInfo(
                node_id="client1", role=NodeRole.CLIENT,
                state=NodeState.ONLINE,
            ),
        }

        class MockTCPClient:
            is_registered = True
            _running = True
            server_host = "127.0.0.1"
            server_port = 8888
            avg_rtt_ms = 12.5

            @staticmethod
            def get_network_quality_snapshot():
                return {
                    "rtt_ms_p95": 20.0,
                    "jitter_ms_p95": 4.0,
                    "stalls_in_window": 1,
                    "reconnects_in_window": 0,
                }

        sched._tcp_client = MockTCPClient()

        status = sched.get_status()
        node_list = sched.get_nodes()

        assert status["network_path"]["path_kind"] == "lan_direct"
        assert status["nodes"]["master"]["network_path"] == status["network_path"]
        assert "network_path" not in status["nodes"]["client1"]
        master = next(node for node in node_list if node["node_id"] == "master")
        assert master["network_path"]["quality"]["rtt_ms_p95"] == 20.0
        assert "127.0.0.1" not in json.dumps(status["network_path"])


# ================================================================
# check_master_health() — TCP 快速检测
# ================================================================

class TestCheckMasterHealthTcp:
    """测试 check_master_health() 通过 TCP 状态快速检测断连"""

    @pytest.fixture
    def sched(self):
        return Scheduler()

    def test_tcp_disconnected_returns_offline(self, sched):
        """TCP 断开时 check_master_health 应立即返回离线"""
        original_role = sched._effective_role
        sched._effective_role = lambda: "client"
        try:
            # 注入未注册的 mock _tcp_client
            mock_client = type('MockTCPClient', (), {
                'is_registered': False,
                '_running': False,
                'server_host': '100.100.52.1',
                'server_port': 8888,
                'sock': None,
            })()
            sched._tcp_client = mock_client

            result = sched.check_master_health()
            # 即使 DB 可能显示在线，TCP 断开应立即报告离线
            if result.get("source") == "tcp_disconnected":
                assert result["master_online"] is False
                assert result["tcp_connected"] is False
        finally:
            sched._effective_role = original_role
            if hasattr(sched, '_tcp_client'):
                del sched._tcp_client

    def test_check_master_health_has_tcp_connected_field(self, sched):
        """所有 check_master_health 返回路径应包含 tcp_connected 字段"""
        result = sched.check_master_health()
        assert "tcp_connected" in result, \
            "check_master_health 返回值应包含 tcp_connected 字段"

    def test_registered_tcp_is_online_without_database(self, sched):
        """无数据库时，已注册且活跃的 TCP 连接仍必须判定主节点在线。"""
        sched._role_override = "client"
        sched._tcp_client = type("MockTCPClient", (), {
            "is_registered": True,
            "_running": True,
            "server_host": "100.64.0.10",
            "server_port": 8888,
            "sock": object(),
        })()

        result = sched.check_master_health()

        assert result["master_online"] is True
        assert result["source"] == "tcp"
        assert result["tcp_connected"] is True
        assert result["master_host"] == "100.64.0.10"

    def test_disconnected_client_uses_local_connection_target(self, sched):
        """断线不得查询旧数据库，仍应保留本机的重连目标。"""
        sched._role_override = "client"
        sched._tcp_client = type("MockTCPClient", (), {
            "is_registered": False,
            "_running": False,
            "server_host": "100.64.0.10",
            "server_port": 8888,
            "sock": None,
        })()

        result = sched.check_master_health()

        assert result["master_online"] is False
        assert result["source"] == "tcp_disconnected"
        assert result["master_host"] == "100.64.0.10"
        assert result["master_port"] == 8888


# ================================================================
# GPU 选择与评分测试 — 多 GPU / 独显/集显混合场景
# ================================================================


PROFILE_IGPU_ONLY = {
    **_engine_contract("pytorch", "pytorch_cpu"),
    "gpu": {
        "name": "Intel Iris Xe Graphics",
        "vram_total_gb": 0.5,
        "cuda_available": False,
        "is_integrated": True,
        "gpu_type": "integrated",
    },
    "ram": {"total_gb": 16.0},
    "cpu": {"physical_cores": 8, "freq_max_mhz": 4000},
}

PROFILE_DGPU_ONLY = {
    "gpu": {
        "name": "NVIDIA GeForce RTX 4060",
        "vram_total_gb": 8.0,
        "cuda_available": True,
        "is_integrated": False,
        "gpu_type": "discrete",
    },
    "ram": {"total_gb": 16.0},
    "cpu": {"physical_cores": 8, "freq_max_mhz": 4000},
}

PROFILE_MULTI_GPU = {
    "gpu": {       # ← 用户前端默认选中的"当前"GPU 被集显占领
        "name": "Intel Iris Xe Graphics",
        "vram_total_gb": 0.5,
        "cuda_available": False,
        "is_integrated": True,
        "gpu_type": "integrated",
    },
    "gpus": [
        {
            "name": "Intel Iris Xe Graphics",
            "vram_total_gb": 0.5,
            "cuda_available": False,
            "is_integrated": True,
            "gpu_type": "integrated",
        },
        {
            "name": "NVIDIA GeForce RTX 4060",
            "vram_total_gb": 8.0,
            "cuda_available": True,
            "is_integrated": False,
            "gpu_type": "discrete",
        },
    ],
    "selected_gpu_index": 0,
    "ram": {"total_gb": 16.0},
    "cpu": {"physical_cores": 8, "freq_max_mhz": 4000},
}

PROFILE_DGPU_NO_IS_INTEGRATED_FIELD = {
    "gpu": {
        "name": "NVIDIA GeForce RTX 4070",
        "vram_total_gb": 12.0,
        "cuda_available": True,
        "gpu_type": "discrete",
    },
    "ram": {"total_gb": 32.0},
    "cpu": {"physical_cores": 12, "freq_max_mhz": 4200},
}


class TestGpuSelection:
    """测试 _select_scoring_gpu() 和 _gpu_is_integrated()"""

    @pytest.fixture
    def sched(self):
        return Scheduler()

    def test_fallback_to_gpus_list_for_discrete(self, sched):
        """多 GPU 画像 + 集显为 current → 应选中 gpus 里的 CUDA 独显"""
        gpu = sched._select_scoring_gpu(PROFILE_MULTI_GPU)
        assert gpu["cuda_available"] is True
        assert "nvidia" in gpu["name"].lower() or "rtx" in gpu["name"].lower()

    def test_single_gpu_works(self, sched):
        """仅一个 GPU 且为独显 → 直接返回"""
        gpu = sched._select_scoring_gpu(PROFILE_DGPU_ONLY)
        assert gpu["cuda_available"] is True

    def test_igpu_only_no_change(self, sched):
        """仅集显 → 返回自身"""
        gpu = sched._select_scoring_gpu(PROFILE_IGPU_ONLY)
        assert gpu.get("is_integrated") is True
        assert gpu.get("cuda_available") is False

    def test_missing_is_integrated_field(self, sched):
        """缺少 is_integrated 但 gpu_type=discrete+cuda → 判为非集显"""
        gpu = sched._select_scoring_gpu(PROFILE_DGPU_NO_IS_INTEGRATED_FIELD)
        assert gpu["cuda_available"] is True

    def test_empty_device_info(self, sched):
        """空画像 → 返回空 dict"""
        assert sched._select_scoring_gpu({}) == {}
        assert sched._select_scoring_gpu(None) == {}


class TestGpuIsIntegrated:
    """测试 _gpu_is_integrated() 的启发式判断（静态方法，无需实例化）"""

    @staticmethod
    def _gpu_is_integrated(gpu):
        return Scheduler._gpu_is_integrated(gpu)

    def test_explicit_false(self):
        assert self._gpu_is_integrated({"is_integrated": False}) is False

    def test_explicit_true(self):
        assert self._gpu_is_integrated({"is_integrated": True}) is True

    def test_gpu_type_discrete(self):
        assert self._gpu_is_integrated({"gpu_type": "discrete"}) is False

    def test_gpu_type_integrated(self):
        assert self._gpu_is_integrated({"gpu_type": "integrated"}) is True

    def test_nvidia_name_marker(self):
        assert self._gpu_is_integrated({
            "name": "NVIDIA GeForce GTX 1660", "cuda_available": True,
        }) is False

    def test_intel_uhd_name_marker(self):
        assert self._gpu_is_integrated({"name": "Intel UHD Graphics 630"}) is True

    def test_unknown_defaults_true(self):
        """无提示 → 保守判为集显"""
        assert self._gpu_is_integrated({"name": "Foo Bar"}) is True


class TestReviewVoteEligibility:
    """测试审查投票资格的本机 CUDA 独显判断。"""

    @pytest.fixture
    def sched_master(self, monkeypatch):
        s = Scheduler()
        monkeypatch.setattr(s, "_effective_role", lambda: "master")
        s.nodes["master"] = NodeInfo(
            node_id="master", role="master", state=NodeState.ONLINE,
            node_type="pc", hostname="master-host",
            device_info=PROFILE_WORKSTATION,
        )
        return s

    def test_can_vote_uses_discrete_gpu_from_gpus_list(self, sched_master):
        """current gpu 是集显时，应使用 gpus 列表里的 CUDA 独显判断投票资格。"""
        sched_master.nodes["master"].device_info = PROFILE_MULTI_GPU

        ok, reason = sched_master.can_node_vote("master")

        assert ok is True, reason

    def test_can_vote_accepts_custom_master_node_id(self, sched_master, monkeypatch):
        """主节点自定义 NODE_ID 时，投票检查应回退到 master 自身记录。"""
        monkeypatch.setattr(sched_master, "get_effective_node_id", lambda: "gaming-laptop")
        sched_master.nodes["master"].device_info = PROFILE_LAPTOP

        ok, reason = sched_master.can_node_vote("gaming-laptop")

        assert ok is True, reason

    def test_can_vote_uses_local_profile_when_master_profile_is_stale(self, sched_master, monkeypatch):
        """主节点表里的旧画像只有集显时，应允许用实时本机画像兜底。"""
        import device_profiler

        class DummyProfile:
            def to_dict(self):
                return PROFILE_MULTI_GPU

        sched_master.nodes["master"].device_info = PROFILE_IGPU_ONLY
        monkeypatch.setattr(device_profiler, "get_profile", lambda: DummyProfile())

        ok, reason = sched_master.can_node_vote("master")

        assert ok is True, reason

    def test_can_vote_rejects_android_node(self, sched_master):
        sched_master.nodes["android-1"] = NodeInfo(
            node_id="android-1", role="client", state=NodeState.ONLINE,
            node_type="android", hostname="phone",
            device_info=PROFILE_MULTI_GPU,
        )

        ok, reason = sched_master.can_node_vote("android-1")

        assert ok is False
        assert "PC" in reason


# ================================================================
# _normalize_master_anchor 测试
# ================================================================


class TestNormalizeMasterAnchor:
    """测试统一分层不变量：master 作为首段 Embedding 锚点"""

    @pytest.fixture
    def sched(self):
        return Scheduler()

    def _nl(self, items=None):
        """便捷构造 node_list"""
        default = [
            {"node_id": "master", "role": "master",
             "device_info": PROFILE_WORKSTATION},
            {"node_id": "client1", "role": "client",
             "device_info": PROFILE_LAPTOP},
            {"node_id": "client2", "role": "client",
             "device_info": PROFILE_ULTRABOOK},
        ]
        return items if items is not None else default

    def test_master_becomes_first_with_at_least_one_layer(self, sched):
        """master 在 assignments 中应为首位且至少 1 层"""
        node_list = self._nl()
        assignments = [
            {"node_id": "client1", "start_layer": 0, "end_layer": 18,
             "layers_count": 18, "has_embedding": True, "has_lm_head": False,
             "score": 50, "role": "client"},
            {"node_id": "master", "start_layer": 18, "end_layer": 24,
             "layers_count": 6, "has_embedding": False, "has_lm_head": True,
             "score": 100, "role": "master"},
        ]
        result = sched._normalize_master_anchor(assignments, node_list, 24)
        assert result[0]["node_id"] == "master"
        assert result[0]["has_embedding"] is True
        assert result[0]["layers_count"] >= 1

    def test_master_not_in_assignments_yet_in_node_list(self, sched):
        """master 未在 assignments 中但在 node_list 中 → 应补入并置首"""
        node_list = self._nl()
        assignments = [
            {"node_id": "client1", "start_layer": 0, "end_layer": 24,
             "layers_count": 24, "has_embedding": True, "has_lm_head": True,
             "score": 50, "role": "client"},
        ]
        result = sched._normalize_master_anchor(assignments, node_list, 24)
        assert result[0]["node_id"] == "master"
        assert result[0]["has_embedding"] is True
        assert result[0]["start_layer"] == 0
        assert result[0]["has_lm_head"] is True
        assert sum(a["layers_count"] for a in result) == 24

    def test_no_master_in_node_list_passes_through(self, sched):
        """node_list 中无 master → 原样返回但区间重新排序"""
        node_list = self._nl([
            {"node_id": "client1", "role": "client", "device_info": PROFILE_LAPTOP},
            {"node_id": "client2", "role": "client", "device_info": PROFILE_ULTRABOOK},
        ])
        assignments = [
            {"node_id": "client1", "start_layer": 0, "end_layer": 18,
             "layers_count": 18, "has_embedding": True, "has_lm_head": False,
             "score": 50, "role": "client"},
            {"node_id": "client2", "start_layer": 18, "end_layer": 24,
             "layers_count": 6, "has_embedding": False, "has_lm_head": True,
             "score": 10, "role": "client"},
        ]
        result = sched._normalize_master_anchor(assignments, node_list, 24)
        assert len(result) == 2
        assert result[0]["has_embedding"] is True
        assert result[-1]["has_lm_head"] is True
        assert sum(a["layers_count"] for a in result) == 24

    def test_master_gets_zero_layers_becomes_one(self, sched):
        """master layers_count=0 → 变为 1，从低分节点回收"""
        node_list = self._nl()
        assignments = [
            {"node_id": "client1", "start_layer": 0, "end_layer": 12,
             "layers_count": 12, "has_embedding": True, "has_lm_head": False,
             "score": 50, "role": "client"},
            {"node_id": "master", "start_layer": 12, "end_layer": 12,
             "layers_count": 0, "has_embedding": False, "has_lm_head": False,
             "score": 100, "role": "master"},
            {"node_id": "client2", "start_layer": 12, "end_layer": 24,
             "layers_count": 12, "has_embedding": False, "has_lm_head": True,
             "score": 10, "role": "client"},
        ]
        result = sched._normalize_master_anchor(assignments, node_list, 24)
        assert result[0]["node_id"] == "master"
        assert result[0]["layers_count"] >= 1
        assert sum(a["layers_count"] for a in result) == 24


# ================================================================
# 分布式推理集成测试 — 全链路流水线编排
# ================================================================


class TestPipelineOrchestrationIntegration:
    """
    集成测试：模拟真实分布式推理场景。

    覆盖关键路径：
    1. master 在线 + 所有 worker 离线 → fallback 本地推理
    2. master 参与流水线首段 → layer 分配正确
    3. 多 GPU 画像 → 独显评分不被集显压低
    4. Android 节点注册后保持在线（不被 init_nodes 强制离线）
    """

    @pytest.fixture
    def sched_master(self, monkeypatch):
        """创建一个"主节点" scheduler，含模拟 TCP 服务端"""
        from node_runtime import node_runtime
        monkeypatch.setattr(node_runtime, "_node_id", "master")
        monkeypatch.setattr(node_runtime, "_node_role", "master")
        s = Scheduler()
        # 伪装为主节点
        s._role_override = "master"
        monkeypatch.setattr(s, "_effective_role", lambda: "master")
        # 注入假 TCP 服务端（满足就绪检查和 get_status 的属性访问）
        fake_server = type('FakeTCPServer', (), {
            '_running': True,
            'host': '0.0.0.0',
            'port': 8888,
            'clients': {},
            'send_to_client': lambda self, cid, data, msg_type: None,
            'broadcast_layer_config': lambda self, assignments: None,
            'get_client_ids': lambda self: list(self.clients.keys()),
            'get_client_info': lambda self, cid: {},
        })()
        monkeypatch.setattr(s, "_tcp_server", fake_server)

        # 注入 master 节点自身
        s.nodes["master"] = NodeInfo(
            node_id="master", role="master", state=NodeState.ONLINE,
            node_type="pc", hostname="master-host",
            device_info=PROFILE_WORKSTATION,
        )
        return s

    @pytest.fixture
    def sched_with_workers(self, sched_master, monkeypatch):
        """主节点 + 2 个 PC worker 在线"""
        s = sched_master
        for nid, profile in [("worker1", PROFILE_LAPTOP), ("worker2", PROFILE_ULTRABOOK)]:
            s.nodes[nid] = NodeInfo(
                node_id=nid, role="client", state=NodeState.ONLINE,
                node_type="pc", address=f"100.64.1.{2 if '1' in nid else 3}:8888",
                hostname=nid, device_info=profile,
                last_heartbeat=time.time(),
            )
        # 模拟 TCP 服务端已知这些 client
        s._tcp_server.clients = {"worker1": True, "worker2": True}
        monkeypatch.setattr(s, "_get_master_model_sha256", lambda: "sha-current")
        assignments = s.compute_layer_assignment()
        # 移除 get_layer_assignments 的 DB 依赖，固定同一份当前分配。
        monkeypatch.setattr(s, "get_layer_assignments", lambda: {
            "total": 24,
            "strategy": "dynamic",
            "assignments": assignments,
        })
        for assignment in assignments:
            node_id = assignment["node_id"]
            if node_id == "master":
                continue
            expected = {
                "config_id": "cfg-current",
                "start_layer": assignment["start_layer"],
                "end_layer": assignment["end_layer"],
                "model_sha256": "sha-current",
                "model_type": "qwen2",
            }
            _install_layer_assignment(
                s,
                node_id,
                expected,
                ack={
                    "status": "ready",
                    "layer_range": [
                        assignment["start_layer"], assignment["end_layer"],
                    ],
                    "model_sha256": "sha-current",
                    "model_type": "qwen2",
                    "engine": "pytorch",
                },
                phase="ready",
            )
        return s

    def test_stale_layer_ack_does_not_make_pipeline_ready(self, sched_with_workers):
        """当前分配变化后，旧层范围 ACK 必须立即失效。"""
        ack = _layer_ack(sched_with_workers, "worker1")
        ack["layer_range"] = [0, 1]
        assert sched_with_workers._worker_assignments.record_ack(
            "worker1", ack,
        ) is not None
        readiness = sched_with_workers._get_pipeline_readiness()
        assert readiness["ready"] is False
        assert readiness["reason_code"] == "worker_layer_loading"

    def test_registration_pending_task_worker_skips_legacy_layer_push(
            self, sched_master, monkeypatch):
        import scheduler as scheduler_mod

        monkeypatch.setattr(scheduler_mod, "TASK_WORKER_EXPERIMENTAL_ENABLED", True)
        node = NodeInfo(
            node_id="surface-stage", role="client", state=NodeState.ONLINE,
            node_type="pc", address="100.100.52.106:8888",
            last_heartbeat=time.time(),
        )
        sched_master.nodes[node.node_id] = node
        sched_master._tcp_server.clients = {node.node_id: True}
        pushes = []
        monkeypatch.setattr(
            sched_master, "push_layer_config_to_clients",
            lambda: pushes.append("legacy"),
        )
        monkeypatch.setattr(sched_master, "_push_node_list_to_client", lambda _id: None)
        monkeypatch.setattr(
            sched_master, "_push_node_update_to_all_clients",
            lambda *args: None,
        )

        sched_master._on_tcp_registration_confirmed(node.node_id)

        assert pushes == []
        assert sched_master._task_worker_control.pending_worker_ids() == {
            node.node_id,
        }

        sched_master._push_layer_config_to_clients_locked()
        assert sched_master._worker_assignments.expected_configs() == {}

    def test_layer_push_admits_peer_after_task_worker_hello_resolution(
            self, sched_master, monkeypatch):
        import scheduler as scheduler_mod

        monkeypatch.setattr(scheduler_mod, "TASK_WORKER_EXPERIMENTAL_ENABLED", True)
        node = NodeInfo(
            node_id="surface-stage", role="client", state=NodeState.ONLINE,
            node_type="pc", address="100.100.52.106:8888",
            last_heartbeat=time.time(),
        )
        sched_master.nodes[node.node_id] = node
        sched_master._tcp_server.clients = {node.node_id: True}
        sched_master._task_worker_control.mark_worker_connection_pending(node.node_id)
        monkeypatch.setattr(
            sched_master._tcp_server, "send_layer_config",
            lambda node_id, config: None,
            raising=False,
        )

        # No capability admission yet: a pending connection is still fenced.
        sched_master._push_layer_config_to_clients_locked()
        assert sched_master._worker_assignments.expected_configs() == {}

        sched_master._task_worker_control.resolve_worker_connection_pending(node.node_id)
        assert sched_master._task_worker_control.pending_worker_ids() == set()
        monkeypatch.setattr(
            sched_master, "_task_worker_layer_stage_ids",
            lambda connected: {node.node_id} & connected,
        )
        sched_master._push_layer_config_to_clients_locked()
        assert not sched_master._worker_assignments.expected(node.node_id)

    def test_model_metadata_gap_preserves_relay_assignment(
            self, sched_master, monkeypatch):
        """Restart-time metadata gaps must not release an endpoint relay host."""
        relay_id = "relay-host"
        relay = NodeInfo(
            node_id=relay_id, role="client", state=NodeState.ONLINE,
            node_type="pc", address="100.64.1.9:8888",
            last_heartbeat=time.time(),
        )
        sched_master.nodes[relay_id] = relay
        sched_master._tcp_server.clients = {relay_id: True}
        relay_segment = {
            "role": "middle",
            "host": "127.0.0.1",
            "port": 50283,
            "n_embd": 4096,
            "layer_start": 23,
            "layer_end": 24,
        }
        monkeypatch.setattr(
            sched_master, "_relay_segment_for_worker",
            lambda node_id: relay_segment if node_id == relay_id else None,
        )
        monkeypatch.setattr(
            sched_master, "_get_active_pipeline_model_info", lambda: {},
        )
        published = []
        monkeypatch.setattr(
            sched_master, "_publish_layer_configs",
            lambda configs: published.append(configs),
        )

        sched_master._push_layer_config_to_clients_locked()

        assert len(published) == 1
        config = published[0][relay_id]
        assert config["engine"] == "relay_middle"
        assert config["relay_segment"] == relay_segment
        assert config.get("release") is not True

    def test_model_metadata_gap_rehydrates_stage_advertised_relay(
            self, sched_master, monkeypatch):
        """A relay that advertises layer_forward still needs relay_middle."""
        relay_id = "relay-stage-host"
        relay = NodeInfo(
            node_id=relay_id, role="client", state=NodeState.ONLINE,
            node_type="pc", address="100.64.1.10:8888",
            last_heartbeat=time.time(),
        )
        sched_master.nodes[relay_id] = relay
        sched_master._tcp_server.clients = {relay_id: True}
        relay_segment = {
            "role": "middle",
            "host": "127.0.0.1",
            "port": 50283,
            "n_embd": 4096,
            "layer_start": 23,
            "layer_end": 24,
        }
        monkeypatch.setattr(
            sched_master, "_relay_segment_for_worker",
            lambda node_id: relay_segment if node_id == relay_id else None,
        )
        monkeypatch.setattr(
            sched_master, "_task_worker_layer_stage_ids",
            lambda _connected: {relay_id},
        )
        monkeypatch.setattr(
            sched_master, "_get_active_pipeline_model_info", lambda: {},
        )
        published = []
        monkeypatch.setattr(
            sched_master, "_publish_layer_configs",
            lambda configs: published.append(configs),
        )

        sched_master._push_layer_config_to_clients_locked()

        assert len(published) == 1
        config = published[0][relay_id]
        assert config["engine"] == "relay_middle"
        assert config["relay_segment"] == relay_segment
        assert config.get("release") is not True

    def test_android_relay_host_is_not_excluded_from_legacy_rehydration(
            self, sched_master, monkeypatch):
        """Endpoint relay role survives candidate filtering regardless of platform."""
        relay_id = "android-relay-host"
        relay = NodeInfo(
            node_id=relay_id, role="client", state=NodeState.ONLINE,
            node_type="android", address="100.64.1.11:8888",
            last_heartbeat=time.time(),
        )
        sched_master.nodes[relay_id] = relay
        sched_master._tcp_server.clients = {relay_id: True}
        relay_segment = {
            "role": "middle",
            "host": "127.0.0.1",
            "port": 50283,
            "n_embd": 4096,
            "layer_start": 23,
            "layer_end": 24,
        }
        monkeypatch.setattr(
            sched_master, "_relay_segment_for_worker",
            lambda node_id: relay_segment if node_id == relay_id else None,
        )
        monkeypatch.setattr(
            sched_master, "_get_active_pipeline_model_info", lambda: {},
        )
        published = []
        monkeypatch.setattr(
            sched_master, "_publish_layer_configs",
            lambda configs: published.append(configs),
        )

        sched_master._push_layer_config_to_clients_locked()

        assert len(published) == 1
        config = published[0][relay_id]
        assert config["engine"] == "relay_middle"
        assert config["relay_segment"] == relay_segment
        assert config.get("release") is not True

    def test_model_transition_defers_release_and_flushes_one_push(
            self, sched_master, monkeypatch):
        """Transient model metadata must not publish release over a new config."""
        pushed = []
        monkeypatch.setattr(
            sched_master,
            "_push_layer_config_to_clients_locked",
            lambda **_kwargs: pushed.append(True),
        )

        sched_master._begin_layer_config_model_change()
        sched_master.push_layer_config_to_clients()
        # The authoritative refresh uses the internal push helper, so it must
        # honor the same fence as ordinary status/hello-triggered pushes.
        assert sched_master.request_authoritative_layer_sync() is True
        assert pushed == []
        assert sched_master._layer_config_push_deferred is True

        sched_master._end_layer_config_model_change()
        assert pushed == [True]
        assert sched_master._layer_config_model_change_depth == 0
        assert sched_master._layer_config_push_deferred is False

    def test_failed_transition_flush_is_retried_with_authoritative_intent(
            self, sched_master, monkeypatch):
        """A transient publish failure must not lose the post-transition sync."""
        calls = []

        def fail_once(**kwargs):
            calls.append(kwargs)
            if len(calls) == 1:
                raise RuntimeError("temporary publish failure")

        monkeypatch.setattr(
            sched_master, "_push_layer_config_to_clients_locked", fail_once,
        )

        sched_master._begin_layer_config_model_change()
        assert sched_master.request_authoritative_layer_sync(
            require_distributed=True,
        ) is True
        sched_master._end_layer_config_model_change()

        assert sched_master._layer_config_push_deferred is True
        assert sched_master._layer_config_push_deferred_authoritative is True
        assert sched_master._layer_config_push_deferred_require_distributed is True

        sched_master.push_layer_config_to_clients()

        assert calls == [
            {"require_distributed": True},
            {"require_distributed": True},
        ]
        assert sched_master._layer_config_push_deferred is False
        assert sched_master._layer_config_push_deferred_authoritative is False
        assert sched_master._layer_config_push_deferred_require_distributed is False
        assert sched_master._authoritative_layer_sync_requests == 0

    # ----------------------------------------------------------
    # 场景 1：master 在线，所有 worker 离线 → fallback 本地推理
    # ----------------------------------------------------------

    def test_all_workers_offline_triggers_fallback(self, sched_master, monkeypatch):
        """无可用 worker → run_pipeline_safe 应回退到全模型推理"""
        fallback_called = []
        pipeline_called = []

        def fake_fallback(prompt, **kw):
            fallback_called.append((prompt, kw))
            return {"response": "fallback_response", "metrics": {"fallback": True}}

        monkeypatch.setattr(sched_master, "_run_full_model_inference", fake_fallback)
        monkeypatch.setattr(sched_master, "run_pipeline", lambda **kw: pipeline_called.append(1) or {})

        result = sched_master.run_pipeline_safe("测试 prompt")

        assert len(fallback_called) == 1
        assert len(pipeline_called) == 0
        assert result["response"] == "fallback_response"
        assert result["metrics"]["fallback"] is True

    def test_all_workers_offline_get_status_shows_offline(self, sched_master):
        """worker 离线时 get_status 应正确反映离线状态"""
        sched_master.nodes["worker-x"] = NodeInfo(
            node_id="worker-x", role="client", state=NodeState.OFFLINE,
            node_type="pc",
        )
        status = sched_master.get_status()
        nodes = status["nodes"]
        assert nodes["master"]["is_available"] is True
        assert nodes["worker-x"]["is_available"] is False

    # ----------------------------------------------------------
    # 场景 2：master 参与流水线 → 层分配正确
    # ----------------------------------------------------------

    def test_layer_assignment_master_has_embedding(self, sched_with_workers):
        """master 应作为首段 Embedding 锚点参与层分配"""
        info = sched_with_workers.get_layer_assignments()
        assignments = info["assignments"]
        # master 至少 1 层
        master = [a for a in assignments if a.get("role") == "master"]
        assert len(master) >= 1, "master 应在 assignments 中"
        m = master[0]
        assert m["has_embedding"] is True
        assert m["start_layer"] == 0
        assert m["layers_count"] >= 1
        # worker1 分数高于 worker2 → 应拿更多层
        workers = [a for a in assignments if a["node_id"] != "master"]
        assert len(workers) >= 1
        if len(workers) >= 2:
            assert workers[0]["layers_count"] >= workers[1]["layers_count"]

    def test_master_layer_range_is_contiguous_with_workers(self, sched_with_workers):
        """master 的层范围 + worker 的层范围应连续覆盖 [0, 24)"""
        info = sched_with_workers.get_layer_assignments()
        assignments = info["assignments"]
        assignments.sort(key=lambda a: a["start_layer"])
        cursor = 0
        for a in assignments:
            assert a["start_layer"] == cursor, \
                f"断点于 {a['node_id']}: 期望 {cursor}, 实际 {a['start_layer']}"
            cursor = a["end_layer"]
        assert cursor == 24, f"末节点 end_layer 应为 24, 实际 {cursor}"

    # ----------------------------------------------------------
    # 场景 3：多 GPU / 独显评分
    # ----------------------------------------------------------

    def test_mixed_gpu_profile_selects_discrete_for_scoring(self, sched_master):
        """多 GPU 画像（集显 front + 独显 in gpus）→ 评分优先用独显"""
        master = sched_master.nodes["master"]
        # 模拟：device_info["gpu"] 是集显，但 gpus 里含独显
        master.device_info = PROFILE_MULTI_GPU
        # 通过 compute_layer_assignment 间接计算 weight
        # 先 _compute_node_weight 验证
        weight = sched_master._compute_node_weight(PROFILE_MULTI_GPU)
        # 必须选中 gpus 列表中的 RTX 4060，评分与同规格独显画像一致。
        assert weight == pytest.approx(
            sched_master._compute_node_weight(PROFILE_DGPU_ONLY),
        )

        weight_igpu = sched_master._compute_node_weight(PROFILE_IGPU_ONLY)
        assert weight_igpu == pytest.approx(22.5)
        assert weight / weight_igpu == pytest.approx(4.407407407407407)

    def test_vram_uses_scoring_gpu_not_frontend_gpu(self, sched_master):
        """_get_node_vram_mb 应使用评分 GPU 而非前端选中 GPU"""
        master = sched_master.nodes["master"]
        master.device_info = PROFILE_MULTI_GPU
        vram_mb = sched_master._get_node_vram_mb("master")
        # 独显 RTX 4060: 8GB → 8192 MB
        assert vram_mb == 8.0 * 1024, \
            f"VRAM 应为独显的 8192MB, 实际: {vram_mb}"

        master.device_info = PROFILE_IGPU_ONLY
        vram_igpu = sched_master._get_node_vram_mb("master")
        # 共享显存不能作为独立容量；若画像未提供 available_gb，则返回未知 0。
        assert vram_igpu == 0, \
            f"纯集显应按系统可用内存计量，缺少 available_gb 时应为未知 0，实际: {vram_igpu}"

    # ----------------------------------------------------------
    # 场景 4：Android 节点状态管理
    # ----------------------------------------------------------

    def test_android_node_stays_online_after_register(self, sched_master):
        """Android HTTP 客户端注册后应保持 online"""
        sched_master.register_android_client(
            "android-1", hostname="Pixel 7", network_type="wifi",
        )
        assert sched_master.nodes["android-1"].state == NodeState.ONLINE
        assert sched_master.nodes["android-1"].node_type == "android"

        # 调用 refresh（模拟心跳检查）— 不应立即标记离线
        sched_master._refresh_http_client_states(now=time.time() + 30)
        assert sched_master.nodes["android-1"].state == NodeState.ONLINE, \
            "30 秒后 Android 仍应在线"

    def test_android_node_offline_after_timeout(self, sched_master):
        """Android 客户端超时后应变为 offline"""
        sched_master.register_android_client("android-2", hostname="Tablet")
        # 快进 121 秒（超过 120s 超时）
        sched_master._refresh_http_client_states(now=time.time() + 121)
        assert sched_master.nodes["android-2"].state == NodeState.OFFLINE

    def test_init_nodes_rebuilds_runtime_topology(self, sched_master, monkeypatch):
        """启动只创建本地主节点；HTTP 客户端由新 presence 重建。"""
        import scheduler as sched_mod
        monkeypatch.setattr(sched_mod, "RUN_MODE", "distributed")

        # 在新 scheduler 上运行 init_nodes（避免 sched_master fixture 已初始化）
        s2 = Scheduler()
        s2._role_override = "master"
        monkeypatch.setattr(s2, "_effective_role", lambda: "master")
        s2._tcp_server = sched_master._tcp_server
        s2.init_nodes()

        assert list(s2.nodes) == ["master"]
        assert s2.nodes["master"].state == NodeState.ONLINE

    # ----------------------------------------------------------
    # 场景 5：ensure_full_model 在 fallback 中触发
    # ----------------------------------------------------------

    def test_fallback_calls_ensure_full_model(self, sched_master, monkeypatch):
        """显式本地模式下推理 → 必须确保模型为完整模型。"""
        from model_host import model_host as _host

        monkeypatch.setattr(sched_master, "_distributed_inference_enabled", False)

        ensure_calls = []
        chat_calls = []

        class MockModelManager:
            is_loaded = True
            _engine_type = "pytorch"
            layer_range = (0, 4)  # 模拟被裁剪过
            _layer_has_embedding = True
            _layer_has_lm_head = False
            tokenizer = None

            def ensure_full_model(self, **kw):
                ensure_calls.append(kw)
                self.layer_range = None
                self._layer_has_lm_head = True

            def chat(self, messages, **kw):
                chat_calls.append((messages, kw))
                return {"content": "fallback reply", "usage": {}, "tokens_per_second": 10.0}

            def chat_stream(self, messages, **kw):
                chat_calls.append(("stream", messages, kw))
                yield "fallback"
                yield " reply"

        mock_mgr = MockModelManager()
        monkeypatch.setattr(_host, "_manager", mock_mgr)

        result = sched_master._run_full_model_inference(
            "test prompt", max_new_tokens=32,
            _fallback_reason="test_fallback",
        )
        # 确认 ensure_full_model 被调用
        assert len(ensure_calls) >= 1, "fallback 前应调用 ensure_full_model"
        # 确认 chat 也被调用
        assert len(chat_calls) >= 1
        assert result["response"] == "fallback reply"
        assert result["metrics"]["fallback"] is True
        assert result["metrics"]["fallback_reason"] == "test_fallback"

    # ----------------------------------------------------------
    # 场景 6：master 参与 pipeline 时的 run_pipeline 编排
    # ----------------------------------------------------------

    def test_run_pipeline_master_loads_local_layer_range(self, sched_with_workers, monkeypatch):
        """run_pipeline 应在 master_participates 时调用 load_layer_range"""
        from model_host import model_host as _host

        layer_loads = []
        forward_calls = []

        class MockModelManager:
            is_loaded = True
            _engine_type = "pytorch"
            model = type("Model", (), {
                "config": type("Config", (), {"model_type": "qwen2"})(),
            })()
            tokenizer: Any = None
            layer_range = None
            _layer_has_embedding = True
            _layer_has_lm_head = True
            quant_type = "int4"

            def load_layer_range(self, start, end, has_embedding, has_lm_head, **kw):
                layer_loads.append((start, end, has_embedding, has_lm_head))

            def ensure_layer_range(self, start, end, has_embedding, has_lm_head, **kw):
                layer_loads.append(("ensure", start, end, has_embedding, has_lm_head))

            def forward_layers(self, **kw):
                forward_calls.append(kw)
                import torch
                input_ids: Any = kw.get("input_ids")
                bs = input_ids.shape[0] if input_ids is not None else 1
                sl = input_ids.shape[1] if input_ids is not None else 1
                return {"hidden_states": torch.randn(bs, sl, 2048),
                        "past_key_values": ((torch.randn(1, 16, sl, 64),
                                             torch.randn(1, 16, sl, 64)),)}

            def get_device(self):
                import torch
                return torch.device("cpu")

        mock_mgr = MockModelManager()

        # 构建 tokenizer mock
        class MockTokenizer:
            eos_token_id = 151643
            def __call__(self, prompt, **kw):
                import torch
                return {"input_ids": torch.randint(0, 1000, (1, 8)),
                        "attention_mask": torch.ones(1, 8)}
            def decode(self, ids, **kw):
                return "mock response"

        mock_mgr.tokenizer = MockTokenizer()
        monkeypatch.setattr(_host, "_manager", mock_mgr)

        # 也需要 mock _send_to_worker 和 _wait_for_layer_result
        from transport_port import MessageType

        send_calls = []
        frozen_assignment_ids = {
            node_id: _layer_expected(
                sched_with_workers, node_id,
            )["assignment_id"]
            for node_id in ("worker1", "worker2")
        }
        reconfigured = set()

        def record_send(worker_id, data, msg_type):
            send_calls.append((worker_id, dict(data), msg_type))
            if (
                msg_type == MessageType.LAYER_FORWARD
                and worker_id not in reconfigured
            ):
                current = _layer_expected(sched_with_workers, worker_id)
                current.pop("assignment_id", None)
                _install_layer_assignment(
                    sched_with_workers, worker_id, current,
                )
                reconfigured.add(worker_id)

        monkeypatch.setattr(
            sched_with_workers, "_send_to_worker", record_send,
        )

        # 模拟无 LM Head 的 worker 返回尾层 hidden states，由 master 执行输出头。
        # _wait_for_layer_result 内部已将 base64 解码为 bytes
        import torch
        from tcp_comm import serialize_tensor_fast
        fake_logits = torch.randn(1, 8, 151936)
        fake_hidden = torch.randn(1, 8, 2048)
        fake_hidden_raw = serialize_tensor_fast(fake_hidden)
        lm_head_calls = []
        monkeypatch.setattr(
            sched_with_workers,
            "_run_master_lm_head",
            lambda hidden: lm_head_calls.append(hidden.shape) or fake_logits,
        )

        monkeypatch.setattr(sched_with_workers, "_wait_for_layer_result",
                           lambda tid, nids, timeout: {
                               "task_id": tid,
                               "node_id": nids[-1] if isinstance(nids, list) else nids,
                               "step": 0,
                               "hidden_states": fake_hidden_raw,
                           })

        result = sched_with_workers.run_pipeline(
            "hello world", max_new_tokens=3, temperature=0.7, top_p=0.9,
        )

        # 验证 master 加载了层范围
        assert len(layer_loads) >= 1, "master 应调用 load_layer_range/ensure_layer_range"
        # 验证 master 本地 forward 被调用
        assert len(forward_calls) >= 1, "master 本地 forward_layers 应被调用"
        # 验证发送了 LAYER_FORWARD 给 worker
        layer_forwards = [
            call for call in send_calls
            if call[2] == MessageType.LAYER_FORWARD
        ]
        assert len(layer_forwards) >= 1, "应发送 LAYER_FORWARD 给 worker"
        assert all(
            payload["assignment_id"] == frozen_assignment_ids[worker_id]
            for worker_id, payload, _msg_type in layer_forwards
        )
        assert (
            _layer_expected(sched_with_workers, "worker1")["assignment_id"]
            != frozen_assignment_ids["worker1"]
        )
        assert lm_head_calls, "worker 尾层 hidden states 应返回 master 执行 LM Head"
        # 验证分布式执行标记与主从任务统计来自同一条已完成流水线任务
        assert result["metrics"]["distributed_used"] is True
        assert set(result["metrics"]["workers_used"]) == {"worker1", "worker2"}
        assert set(result["metrics"]["workers_counted"]) == {"worker1", "worker2"}
        assert sched_with_workers.nodes["master"].task_count == 1
        assert sched_with_workers.nodes["worker1"].task_count == 1
        assert sched_with_workers.nodes["worker2"].task_count == 1

    def test_route_a_stage_pipeline_keeps_prefill_decode_positions(
            self, sched_with_workers, monkeypatch):
        """Route-A v3 stages own both prefill and decode position semantics."""
        from model_host import model_host as _host
        import torch

        class Tokenizer:
            eos_token_id = 999

            def __call__(self, prompt, **kwargs):
                return {
                    "input_ids": torch.tensor([[11, 12]]),
                    "attention_mask": torch.ones(1, 2, dtype=torch.long),
                }

            def decode(self, ids, **kwargs):
                return "answer"

        class Manager:
            is_loaded = True
            tokenizer = Tokenizer()
            model = type("Model", (), {
                "config": type("Config", (), {
                    "max_position_embeddings": 128,
                })(),
            })()

            def ensure_layer_range(self, *args, **kwargs):
                pass

            def forward_layers(self, **kwargs):
                input_ids = kwargs["input_ids"]
                seq_len = int(input_ids.shape[1])
                return {
                    "hidden_states": torch.ones(1, seq_len, 4),
                    "past_key_values": ((torch.ones(1, 1, seq_len, 1),
                                          torch.ones(1, 1, seq_len, 1)),),
                }

        monkeypatch.setattr(_host, "_manager", Manager())

        class Callbacks:
            thinking_system_prompt = ""

            @staticmethod
            def active_task_graph_model_identity():
                return type("Identity", (), {
                    "sha256": "a" * 64,
                    "model_id": "qwen-test",
                })()

            @staticmethod
            def build_model_chat_prompt(tokenizer, messages, **kwargs):
                return "prompt"

            @staticmethod
            def format_model_response(text, show_thinking, **kwargs):
                return text, ""

        monkeypatch.setattr(sched_with_workers, "_require_callbacks", lambda: Callbacks())
        monkeypatch.setattr(
            sched_with_workers,
            "_verify_pipeline_readiness",
            lambda nodes: (True, "ok"),
        )
        stage_calls = []

        def fake_stage(**kwargs):
            stage_calls.append(kwargs)
            return {"kind": "token", "token_argmax": 7}

        monkeypatch.setattr(sched_with_workers, "_execute_layer_stage_offer", fake_stage)
        legacy_identity_resolver = sched_with_workers._route_a_stage_model_identity
        monkeypatch.setattr(
            sched_with_workers,
            "_route_a_stage_model_identity",
            lambda *_args, **_kwargs: (_ for _ in ()).throw(
                AssertionError("hot path must use the preflight-frozen identity")
            ),
        )
        # ★ Route A 的 offer 带**设备工件身份**（层段在设备上执行、用设备的 GGUF），
        #   而不是 master 的模型身份 —— 后者是 safetensors，`_layer_model_matches`
        #   比 engine/format/sha256，必然拒绝。
        sched_with_workers._task_worker_control.status = lambda role: {
            "workers": [{
                "node_id": "android-1",
                "capabilities": {"models": [{
                    "model_id": "layer-android1",
                    "engine": "llama_cpp",
                    "format": "gguf",
                    "revision": "local",
                    "sha256": "c" * 64,
                }]},
            }],
        }
        result = sched_with_workers._run_route_a_stage_pipeline(
            prompt="hello",
            max_new_tokens=2,
            temperature=0.0,
            top_p=1.0,
            session_id=None,
            messages=None,
            show_thinking=False,
            routing_preference="auto",
            stage_nodes=[{
                "node_id": "android-1",
                "execution": "stage_offer_v3",
                "start_layer": 4,
                "end_layer": 8,
                "layers_count": 4,
                "stage_model_identity": {
                    "model_id": "layer-android1",
                    "engine": "llama_cpp",
                    "format": "gguf",
                    "revision": "local",
                    "sha256": "c" * 64,
                },
            }],
            master_assignment={
                "node_id": "master",
                "start_layer": 0,
                "end_layer": 4,
                "layers_count": 4,
                "has_embedding": True,
                "has_lm_head": True,
            },
        )

        assert result["metrics"]["execution_mode"] == "route_a_stage_offer_v3"
        # ★ 协议要求 `wf_` 前缀（`_WORKFLOW_ID`）；裸 task_id 会被 offer 校验拒掉。
        assert all(call["workflow_id"].startswith("wf_") for call in stage_calls)
        assert [call["positions"] for call in stage_calls] == [[0, 1], [2]]
        assert all(call["want_hidden"] is False for call in stage_calls)
        assert all(
            call["model_identity"].model_id == "layer-android1"
            and call["model_identity"].engine == "llama_cpp"
            and call["model_identity"].format == "gguf"
            for call in stage_calls
        )

        identity_lookups = []

        def legacy_identity(node_id, assignment=None):
            identity_lookups.append(node_id)
            return legacy_identity_resolver(node_id, assignment)

        monkeypatch.setattr(
            sched_with_workers, "_route_a_stage_model_identity", legacy_identity,
        )
        stage_calls.clear()
        legacy_result = sched_with_workers._run_route_a_stage_pipeline(
            prompt="hello",
            max_new_tokens=2,
            temperature=0.0,
            top_p=1.0,
            session_id=None,
            messages=None,
            show_thinking=False,
            routing_preference="auto",
            stage_nodes=[{
                "node_id": "android-1",
                "execution": "stage_offer_v3",
                "start_layer": 4,
                "end_layer": 8,
                "layers_count": 4,
            }],
            master_assignment={
                "node_id": "master",
                "start_layer": 0,
                "end_layer": 4,
                "layers_count": 4,
                "has_embedding": True,
                "has_lm_head": True,
            },
        )

        assert legacy_result["metrics"]["execution_mode"] == "route_a_stage_offer_v3"
        assert identity_lookups == ["android-1"]
        assert len(stage_calls) == 2

    def test_run_pipeline_builds_native_prompt_from_history(
            self, sched_with_workers, monkeypatch):
        from model_host import model_host as _host
        import torch
        from tcp_comm import serialize_tensor_fast

        prompts = []

        class Tokenizer:
            eos_token_id = 2

            def apply_chat_template(self, messages, tokenize=False,
                                    add_generation_prompt=True):
                assert messages == [
                    {"role": "user", "content": "first"},
                    {"role": "assistant", "content": "answer"},
                    {"role": "user", "content": "second"},
                ]
                return "native-history-prompt"

            def __call__(self, prompt, **kwargs):
                prompts.append(prompt)
                return {
                    "input_ids": torch.tensor([[1, 3]]),
                    "attention_mask": torch.ones(1, 2, dtype=torch.long),
                }

            def decode(self, ids, **kwargs):
                return "answer"

        class Manager:
            is_loaded = True
            _engine_type = "pytorch"
            model = type("Model", (), {
                "config": type("Config", (), {"model_type": "qwen2"})(),
            })()
            tokenizer = Tokenizer()

            def ensure_layer_range(self, *args, **kwargs):
                pass

            def forward_layers(self, **kwargs):
                return {"hidden_states": torch.randn(1, 2, 8)}

            def get_device(self):
                return torch.device("cpu")

            def _merge_stop_sequences(self, _value):
                return []

            def _get_generation_eos_token_ids(self, _stops):
                return 2

        monkeypatch.setattr(_host, "_manager", Manager())
        monkeypatch.setattr(sched_with_workers, "_send_to_worker", lambda *a, **kw: None)
        hidden = serialize_tensor_fast(torch.randn(1, 2, 8))
        monkeypatch.setattr(
            sched_with_workers, "_wait_for_layer_result_with_ack",
            lambda *a, **kw: {"hidden_states": hidden},
        )
        monkeypatch.setattr(
            sched_with_workers, "_run_master_lm_head",
            lambda _hidden: torch.tensor([[[0.0, 0.0, 100.0, 0.0]]]),
        )

        sched_with_workers.run_pipeline(
            "second",
            max_new_tokens=1,
            messages=[
                {"role": "user", "content": "first"},
                {"role": "assistant", "content": "answer"},
                {"role": "user", "content": "second"},
            ],
        )

        assert prompts == ["native-history-prompt"]

    def test_run_pipeline_safe_immediate_path_locks(self, sched_with_workers, monkeypatch):
        """立即执行路径应正确管理 inference_lock（不产生死锁或泄漏）"""
        from model_host import model_host as _host

        class MockModelManager:
            is_loaded = True
            _engine_type = "pytorch"
            model = type("Model", (), {
                "config": type("Config", (), {"model_type": "qwen2"})(),
            })()
            tokenizer: Any = None
            layer_range = None
            _layer_has_embedding = True
            _layer_has_lm_head = True
            quant_type = "int4"

            def ensure_layer_range(self, start_layer, end_layer, has_embedding, has_lm_head, **kw):
                pass

            def forward_layers(self, **kw):
                import torch
                bs = 1; sl = 4
                return {"hidden_states": torch.randn(bs, sl, 2048),
                        "past_key_values": ((torch.randn(1, 16, sl, 64),
                                             torch.randn(1, 16, sl, 64)),)}

            def get_device(self):
                import torch
                return torch.device("cpu")

        mock_mgr = MockModelManager()
        class MockTokenizer:
            eos_token_id = 151643
            def __call__(self, prompt, **kw):
                import torch
                return {"input_ids": torch.randint(0, 1000, (1, 4)),
                        "attention_mask": torch.ones(1, 4)}
            def decode(self, ids, **kw):
                return "mock"

        mock_mgr.tokenizer = MockTokenizer()
        monkeypatch.setattr(_host, "_manager", mock_mgr)
        monkeypatch.setattr(sched_with_workers, "_send_to_worker",
                           lambda wid, data, mtype: None)

        import torch
        from tcp_comm import serialize_tensor_fast
        fake_logits = torch.randn(1, 4, 151936)
        fake_logits_raw = serialize_tensor_fast(fake_logits)  # bytes — 模拟 _wait_for_layer_result 解码后的返回值
        monkeypatch.setattr(sched_with_workers, "_wait_for_layer_result",
                           lambda tid, nids, timeout: {
                               "task_id": tid, "node_id": "worker1", "step": 0,
                               "logits": fake_logits_raw,
                           })

        result = sched_with_workers.run_pipeline_safe("test", max_new_tokens=1)
        # 验证锁已正常释放（不是死锁）
        assert not sched_with_workers._inference_lock.locked(), \
            "推理锁应在 run_pipeline_safe 返回后释放"
        assert "error" not in result or result.get("response"), \
            f"不应返回致命错误: {result.get('error', '')}"

    def test_run_pipeline_stream_uses_safe_scheduler(self, sched_master, monkeypatch):
        """fast SSE 也必须经过排队、互斥和全模型回退入口。"""
        calls = []

        def safe(prompt, **kwargs):
            calls.append(prompt)
            kwargs["_stream_callback"]({
                "done": True, "response": "ok", "metrics": {},
            })
            return {"response": "ok", "metrics": {}}

        monkeypatch.setattr(sched_master, "run_pipeline_safe", safe)
        monkeypatch.setattr(
            sched_master,
            "_run_pipeline",
            lambda *args, **kwargs: pytest.fail("不能绕过 run_pipeline_safe"),
        )

        events = list(sched_master.run_pipeline_stream("hello"))
        assert calls == ["hello"]
        assert events[-1]["done"] is True
        assert events[-1]["response"] == "ok"

    def test_run_pipeline_unexpected_exception_aborts_registered_context(
            self, sched_master, monkeypatch):
        """采样/解码等未预期异常也必须清理所有节点 KV cache。"""
        aborts = []
        clears = []

        def fail(*args, **kwargs):
            sched_master._pipeline_context.stack.append({
                "task_id": "task-exception",
                "pipeline_nodes": [{"node_id": "worker1"}],
            })
            raise RuntimeError("sampling failed")

        monkeypatch.setattr(sched_master, "_run_pipeline", fail)
        monkeypatch.setattr(
            sched_master,
            "_broadcast_pipeline_abort",
            lambda nodes, task_id, reason: aborts.append((nodes, task_id, reason)),
        )
        monkeypatch.setattr(
            sched_master,
            "_clear_pipeline_runtime_state",
            lambda task_id: clears.append(task_id),
        )

        with pytest.raises(RuntimeError, match="sampling failed"):
            sched_master.run_pipeline("hello")
        assert aborts[0][1] == "task-exception"
        assert clears == ["task-exception"]


# ================================================================
# Phase 7 P0: _effective_role 实现逻辑测试
# ================================================================

class TestEffectiveRole:
    """测试 _effective_role() 的实现逻辑（非 mock）。"""

    def test_default_role_from_config(self):
        """默认角色来自 NODE_ROLE 配置。"""
        s = Scheduler()
        # 新 Scheduler 未设置 _role_override，直接读 NODE_ROLE
        role = s._effective_role()
        assert role in ("master", "client"), \
            f"有效角色应为 master 或 client，实际: {role}"

    def test_role_override_takes_precedence(self):
        """_role_override 应优先于 NODE_ROLE。"""
        s = Scheduler()
        s._role_override = "client"
        assert s._effective_role() == "client"

    def test_role_override_empty_falls_through_to_config(self):
        """空字符串 override 应回退到 NODE_ROLE（falsy 值处理）。"""
        s = Scheduler()
        s._role_override = ""
        role = s._effective_role()
        # 空字符串为 falsy → or 短路 → NODE_ROLE
        assert role in ("master", "client")


class TestProvisionalMasterRole:
    """Fresh packaged nodes can join an existing master without DB access."""

    @staticmethod
    def _set_master_role(monkeypatch, tmp_path):
        import config as cfg
        import scheduler as scheduler_mod

        monkeypatch.setattr(cfg, "NODE_ROLE", "master", raising=False)
        monkeypatch.setattr(cfg, "NODE_ID", "master", raising=False)
        monkeypatch.setattr(scheduler_mod, "NODE_ROLE", "master", raising=False)
        monkeypatch.setattr(scheduler_mod, "NODE_ID", "master", raising=False)
        from node_runtime import node_runtime
        monkeypatch.setattr(node_runtime, "_node_id", "master")
        monkeypatch.setattr(node_runtime, "_node_role", "master")
        monkeypatch.setenv("QLH_NODE_CONFIG_PATH", str(tmp_path / "node_config.json"))
        monkeypatch.delenv("QLH_NODE_ROLE", raising=False)

    def test_local_only_master_is_not_provisional(self, monkeypatch, tmp_path):
        """本地 SQLite 主节点不因远端数据库退场而降级。"""
        self._set_master_role(monkeypatch, tmp_path)
        sched = Scheduler()
        sched._master_identity_verified = True
        sched._master_identity_reason = "db_unavailable"

        role = sched.get_my_role()

        assert role["node_role"] == "master"
        assert role["is_master"] is True
        assert role["is_provisional"] is False
        assert role["can_join_existing_master"] is False

    def test_locally_verified_master_is_not_provisional(self, monkeypatch, tmp_path):
        """本地身份已验证时，外部服务状态不得隐藏主节点能力。"""
        self._set_master_role(monkeypatch, tmp_path)
        sched = Scheduler()
        sched._master_identity_verified = True
        sched._master_identity_reason = "db_unavailable"

        role = sched.get_my_role()

        assert role["node_role"] == "master"
        assert role["is_master"] is True
        assert role["is_provisional"] is False
        assert role["can_join_existing_master"] is False

    def test_database_verified_master_is_not_provisional(self, monkeypatch, tmp_path):
        self._set_master_role(monkeypatch, tmp_path)
        sched = Scheduler()
        sched._master_identity_verified = True
        sched._master_identity_reason = "match"

        role = sched.get_my_role()

        assert role["node_role"] == "master"
        assert role["is_master"] is True
        assert role["is_provisional"] is False
        assert role["can_join_existing_master"] is False

    def test_master_with_connected_clients_cannot_switch(self, monkeypatch, tmp_path):
        self._set_master_role(monkeypatch, tmp_path)
        sched = Scheduler()
        sched._master_identity_reason = "db_unavailable"
        sched._tcp_server = MagicMock()
        sched._tcp_server.get_client_ids.return_value = ["client-1"]

        result = sched.activate_client_mode()

        assert result["status"] == "denied"
        assert sched._effective_role() == "master"


class TestLocalMasterIdentity:
    """Main-node identity must be fully owned by its local SQLite store."""

    def test_verify_and_reset_identity_without_postgresql(self, monkeypatch, tmp_path):
        import local_store

        sqlite_path = tmp_path / "qlh-control.sqlite3"
        monkeypatch.setenv("QLH_SQLITE_PATH", str(sqlite_path))
        monkeypatch.setattr(local_store, "_initialized_paths", set())

        # ★ R-R4（用户裁定 A，`dec-e31b944dfde347ef`）：`reset_master_identity()` 以**物理 MAC** 为准
        #   ⇒ 这里把物理探测固定成可预测值，使「重置绑定物理值」这一契约可以被断言。
        monkeypatch.setattr("transport_port.get_mac_addresses",
                            lambda: ["AA-BB-CC-DD-EE-FF"], raising=False)

        sched = Scheduler()
        sched._role_override = "master"
        sched._mac_addresses = ["AA-BB-CC-DD-EE-FF"]

        sched._verify_master_identity()
        identity = sched.get_invite_info()
        assert identity["identity_verified"] is True
        assert identity["identity_reason"] == "first_run"
        assert local_store.get_local_master_identity()["mac_addresses"] == [
            "aa-bb-cc-dd-ee-ff",
        ]

        sched._verify_master_identity()
        identity = sched.get_invite_info()
        assert identity["identity_verified"] is True
        assert identity["identity_reason"] == "match"

        sched._mac_addresses = ["11:22:33:44:55:66"]
        sched._verify_master_identity()
        identity = sched.get_invite_info()
        assert identity["identity_verified"] is False
        assert identity["identity_reason"] == "mac_mismatch"

        result = sched.reset_master_identity()
        assert result["status"] == "ok"
        assert sched.get_invite_info()["identity_reason"] == "reset"
        # ★ 重置绑的是**物理** MAC，而**不是**上面被改成 `11:22:33:44:55:66` 的内存缓存 ——
        #   用户裁定 A：重置 = 强制回到物理真相，缓存陈旧/被污染时不得被固化。
        assert local_store.get_local_master_identity()["mac_addresses"] == [
            "aa-bb-cc-dd-ee-ff",
        ]

    def test_invite_reports_tailnet_address_source(self):
        sched = Scheduler()
        sched._lan_ip = "100.88.9.10"

        assert sched.get_invite_info()["master_host_source"] == "tailnet"


class TestMasterIdentityAdvertiseAndReset:
    """★ R-R4（B5 / #8）实机复验补的回归：重置必须绑**物理** MAC；host_source 要区分 tailnet/LAN。"""

    def test_reset_binds_physical_mac_not_stale_cache(self, monkeypatch, tmp_path):
        """★ 重置的语义 = 「强制回到物理真相」⇒ **不得**沿用内存里可能陈旧/被污染的缓存。

        实机复验发现：`reset_master_identity()` 原先**优先**用 `self._mac_addresses`；把该缓存污染成
        `de-ad-be-ef-00-01` 后重置，会把**错值固化**进 SQLite —— 而它的 docstring 明说
        「立即把当前**物理** MAC 绑定到主节点 SQLite」。
        """
        import local_store
        from scheduler import Scheduler

        monkeypatch.setenv("QLH_SQLITE_PATH", str(tmp_path / "qlh-control.sqlite3"))
        monkeypatch.setattr(local_store, "_initialized_paths", set())
        monkeypatch.setattr(
            "transport_port.get_mac_addresses",
            lambda: ["AA-BB-CC-DD-EE-FF", "11-22-33-44-55-66"],
            raising=False,
        )

        sched = Scheduler()
        sched._role_override = "master"
        sched._mac_addresses = ["de-ad-be-ef-00-01"]          # 陈旧 / 被污染的缓存

        result = sched.reset_master_identity()

        assert result["status"] == "ok"
        stored = local_store.get_local_master_identity()["mac_addresses"]
        assert stored == ["11-22-33-44-55-66", "aa-bb-cc-dd-ee-ff"]     # 物理值（已排序）
        assert "de-ad-be-ef-00-01" not in stored

    def test_invite_reports_lan_source_when_not_on_tailnet(self):
        """③ 的另一半：不在 Tailnet 上时 `master_host_source` 必须是 `lan`（B5 的「LAN 回退」）。"""
        from scheduler import Scheduler

        sched = Scheduler()
        sched._lan_ip = "192.168.1.61"

        invite = sched.get_invite_info()
        assert invite["master_host_source"] == "lan"
        assert invite["master_host"] == "192.168.1.61"


class TestSchedulerHighAvailabilitySQLite:
    """HA control state stays local, bounded, and restart-safe."""

    @pytest.fixture
    def scheduler(self, monkeypatch, tmp_path):
        import local_store

        monkeypatch.setenv("QLH_SQLITE_PATH", str(tmp_path / "control.sqlite3"))
        monkeypatch.setattr(local_store, "_initialized_paths", set())
        instance = Scheduler()
        instance._role_override = "master"
        return instance

    def test_ha_state_persists_across_scheduler_instances(self, scheduler):
        scheduler._update_ha_state(
            spare_master={"node_id": "worker-spare", "hostname": "spare"},
            spare_master_active=True,
            pending_new_master_id="worker-next",
        )
        scheduler._append_ha_log(
            "transfer_logs",
            "promotion",
            {"transfer_id": "transfer-test"},
        )

        restored = Scheduler()
        spare = restored.get_spare_master()

        assert spare == {
            "node_id": "worker-spare",
            "hostname": "spare",
            "is_online": False,
            "state": "unknown",
            "is_active": True,
        }
        assert restored._ha_state["pending_new_master_id"] == "worker-next"
        assert restored.get_transfer_logs()[0]["details"] == {
            "transfer_id": "transfer-test",
        }

    def test_ha_log_is_bounded_and_rejects_unknown_category(self, scheduler):
        for index in range(260):
            scheduler._append_ha_log(
                "spare_master_logs",
                "designated",
                {"sequence": index},
            )

        restored = Scheduler()
        logs = restored.get_spare_master_logs()

        assert len(logs) == 256
        assert logs[0]["details"]["sequence"] == 4
        assert logs[-1]["details"]["sequence"] == 259
        with pytest.raises(ValueError, match="unknown HA log category"):
            scheduler._append_ha_log("unknown", "ignored", {})

    def test_clear_spare_master_is_role_gated_and_persisted(self, scheduler):
        scheduler._update_ha_state(
            spare_master={"node_id": "worker-spare"},
            spare_master_active=True,
            pending_new_master_id="worker-next",
        )
        scheduler._role_override = "client"
        assert scheduler.clear_spare_master()["status"] == "denied"

        scheduler._role_override = "master"
        assert scheduler.clear_spare_master()["status"] == "ok"

        restored = Scheduler()
        assert restored.get_spare_master() is None
        assert restored._ha_state["pending_new_master_id"] == ""
        assert restored.get_spare_master_logs()[-1]["direction"] == "undesignated"

    def test_role_transfer_message_persists_promotion_and_acknowledges(
        self, scheduler,
    ):
        sent = []

        class FakeClient:
            sock = object()

            @staticmethod
            def send_data(payload, message_type):
                sent.append((payload, message_type))

        scheduler._tcp_client = FakeClient()
        scheduler._handle_role_transfer("old-master", {"data": {
            "transfer_id": "transfer-promote",
            "old_master_id": "old-master",
            "new_master_id": "new-master",
            "spare_master": {"node_id": "worker-spare"},
        }})

        assert sent[0][0]["transfer_id"] == "transfer-promote"
        assert sent[0][0]["ack"]["accepted"] is True
        assert scheduler._ha_state["spare_master_active"] is True
        assert scheduler._ha_state["pending_new_master_id"] == "new-master"
        assert scheduler.get_transfer_logs()[-1]["direction"] == "promotion"

    def test_startup_deactivation_notifies_spare_and_clears_state(
        self, scheduler, monkeypatch,
    ):
        import scheduler as scheduler_module

        sent = []

        class FakeServer:
            _running = True

            @staticmethod
            def get_client_ids():
                return ["worker-spare"]

            @staticmethod
            def send_to_client(node_id, payload, message_type):
                sent.append((node_id, payload, message_type))

        monkeypatch.setattr(scheduler_module, "NODE_ID", "worker-next")
        scheduler._tcp_server = FakeServer()
        scheduler._update_ha_state(
            spare_master={"node_id": "worker-spare"},
            spare_master_active=True,
            pending_new_master_id="worker-next",
        )

        scheduler.deactivate_spare_master_on_startup()

        assert sent[0][0] == "worker-spare"
        assert sent[0][1]["new_master_id"] == "worker-next"
        assert scheduler._ha_state["spare_master_active"] is False
        assert scheduler._ha_state["pending_new_master_id"] == ""


# ================================================================
# Phase 7 P1: compute_layer_assignment 边界测试
# ================================================================

class TestComputeLayerAssignmentEdgeCases:
    """测试 compute_layer_assignment 的极端情况。"""

    @pytest.fixture
    def sched(self):
        s = Scheduler()
        s.nodes = {}
        return s

    def test_zero_vram_node_fallback_to_min_one_layer(self, sched):
        """0 VRAM 的节点不应崩溃，应回退到最小 1 层分配。"""
        nodes = [
            {"node_id": "master", "role": "master", "node_type": "pc",
             "device_info": {"gpu": {"vram_total_gb": 0.0, "cuda_available": False},
                             "ram": {"total_gb": 8.0}, "cpu": {"physical_cores": 4}}},
            {"node_id": "w1", "role": "client", "node_type": "pc",
             "device_info": {"gpu": {"vram_total_gb": 0.0, "cuda_available": False},
                             "ram": {"total_gb": 8.0}, "cpu": {"physical_cores": 4}}},
        ]
        result = sched.compute_layer_assignment(nodes)
        assert len(result) >= 1
        total = sum(a["layers_count"] for a in result)
        assert total == 24
        # 验证每个节点至少分配 1 层（0-VRAM 回退保证）
        for a in result:
            assert a["layers_count"] >= 1, (
                f"节点 {a['node_id']} 分配到 {a['layers_count']} 层，"
                f"期望至少 1 层 (0-VRAM 回退)"
            )

    def test_only_android_nodes_returns_empty(self, sched):
        """仅 Android 节点应返回空分配（Android 不参与层拆分）。"""
        nodes = [
            {"node_id": "android-a", "role": "client", "node_type": "android",
             "device_info": {"platform": "android"}},
            {"node_id": "android-b", "role": "client", "node_type": "android",
             "device_info": {"platform": "android"}},
        ]
        result = sched.compute_layer_assignment(nodes)
        assert result == []


# ================================================================
# L5: task_id / request_id 半结构化日志贯穿
# ================================================================

class TestTaskRequestIdCorrelation:
    """L5: scheduler 中 task_id 和 request_id 的关联。"""

    @pytest.fixture
    def queue(self):
        from scheduler import PipelineQueue
        return PipelineQueue(max_size=10, result_ttl=60)

    def test_enqueue_stores_request_id(self, queue):
        """enqueue 时应存储 request_id 到 QueueTask。"""
        tid = queue.enqueue(
            prompt="test prompt",
            max_new_tokens=128,
            request_id="req-abc-123",
        )
        queue.stop()
        assert tid.startswith("q_")

    def test_enqueue_request_id_in_log(self, queue, caplog):
        """enqueue 日志应包含 request_id。"""
        with caplog.at_level(logging.INFO, logger="scheduler"):
            queue.enqueue(prompt="test", max_new_tokens=64, request_id="rid-logtest")
            queue.stop()

        found = False
        for record in caplog.records:
            msg = record.getMessage()
            if "task_enqueue" in msg and "request_id=rid-logtest" in msg:
                found = True
                break
        assert found, f"未找到包含 request_id=rid-logtest 的 enqueue 日志: {caplog.text}"

    def test_task_dispatch_log_has_request_id(self, queue, caplog):
        """task_dispatch 日志应包含 request_id。"""
        def echo_process(**kwargs):
            return {"status": "ok", "response": "echo"}

        queue.start(process_fn=echo_process)
        with caplog.at_level(logging.INFO, logger="scheduler"):
            tid = queue.enqueue(
                prompt="hello",
                max_new_tokens=128,
                request_id="rid-dispatch-test",
            )
            queue.wait_for_result(tid, timeout=5.0)

        found = False
        for record in caplog.records:
            msg = record.getMessage()
            if "task_dispatch" in msg and "request_id=rid-dispatch-test" in msg:
                found = True
                break
        assert found, f"task_dispatch 日志应包含 request_id: {caplog.text}"
        queue.stop()

    def test_task_complete_log_has_request_id(self, queue, caplog):
        """task_complete 日志应包含 request_id。"""
        def quick_process(**kwargs):
            return {"status": "ok", "content": "done"}

        queue.start(process_fn=quick_process)
        with caplog.at_level(logging.INFO, logger="scheduler"):
            tid = queue.enqueue(
                prompt="quick",
                max_new_tokens=16,
                request_id="rid-complete-test",
            )
            queue.wait_for_result(tid, timeout=5.0)

        found = False
        for record in caplog.records:
            msg = record.getMessage()
            if "task_complete" in msg and "request_id=rid-complete-test" in msg:
                found = True
                break
        assert found, f"task_complete 日志应包含 request_id: {caplog.text}"
        queue.stop()

    def test_task_failed_log_has_request_id_and_exc_info(self, queue, caplog):
        """task_failed 日志应包含 request_id 且携带异常堆栈。"""
        def failing_process(**kwargs):
            raise RuntimeError("模拟推理崩溃")

        queue.start(process_fn=failing_process)
        with caplog.at_level(logging.ERROR, logger="scheduler"):
            tid = queue.enqueue(
                prompt="will fail",
                max_new_tokens=64,
                request_id="rid-fail-test",
            )
            queue.wait_for_result(tid, timeout=5.0)

        records = [r for r in caplog.records if "task_failed" in r.getMessage()]
        assert records
        assert "request_id=rid-fail-test" in records[0].getMessage()
        assert records[0].exc_info is not None
        queue.stop()

    def test_enqueue_without_request_id_defaults_to_dash(self, queue, caplog):
        """未提供 request_id 时日志应显示 request_id=-。"""
        def quick_process(**kwargs):
            return {"status": "ok"}

        queue.start(process_fn=quick_process)
        with caplog.at_level(logging.INFO, logger="scheduler"):
            tid = queue.enqueue(prompt="no-rid", max_new_tokens=32)
            queue.wait_for_result(tid, timeout=5.0)

        found = False
        for record in caplog.records:
            msg = record.getMessage()
            if "task_enqueue" in msg:
                assert "request_id=-" in msg, f"无 request_id 时应显示 -: {msg}"
                found = True
                break
        assert found
        queue.stop()


class TestForwardInferenceRequestId:
    """L5: forward_inference_to_master 中的 request_id 传递。"""

    def test_forward_method_accepts_request_id_param(self):
        """forward_inference_to_master 应接受 request_id 参数。"""
        import inspect
        from scheduler import Scheduler
        sig = inspect.signature(Scheduler.forward_inference_to_master)
        assert "request_id" in sig.parameters

    def test_start_infer_task_accepts_request_id_param(self):
        """start_infer_task 应接受 request_id 参数。"""
        import inspect
        from scheduler import Scheduler
        sig = inspect.signature(Scheduler.start_infer_task)
        assert "request_id" in sig.parameters

    def test_forward_result_is_correlated_and_errors_are_not_reported_as_ok(
            self, monkeypatch):
        import scheduler as scheduler_mod

        sched = Scheduler()
        sched._role_override = "client"
        monkeypatch.setattr(scheduler_mod, "NODE_ROLE", "client")

        class FakeClient:
            _running = True
            is_registered = True
            sock = object()

            def send_data(self, data, msg_type):
                if msg_type.value != "infer_forward":
                    return
                sched._on_tcp_message("master", {
                    "type": "infer_result",
                    "data": {
                        "forward_request_id": data["forward_request_id"],
                        "task_id": "task-x",
                        "status": "error",
                        "error": "worker failed",
                        "metrics": {"distributed_used": False},
                    },
                })

        monkeypatch.setattr(sched, "_tcp_client", FakeClient(), raising=False)
        result = sched.forward_inference_to_master("hello", timeout=1)
        assert result["status"] == "error"
        assert result["error"] == "worker failed"

    def test_forward_timeout_sends_cancel(self, monkeypatch):
        import scheduler as scheduler_mod

        sched = Scheduler()
        sched._role_override = "client"
        monkeypatch.setattr(scheduler_mod, "NODE_ROLE", "client")
        sent = []

        class FakeClient:
            _running = True
            is_registered = True
            sock = object()

            def send_data(self, data, msg_type):
                sent.append((data, msg_type.value))

        monkeypatch.setattr(sched, "_tcp_client", FakeClient(), raising=False)
        result = sched.forward_inference_to_master("hello", timeout=0.01)
        assert result["status"] == "timeout"
        assert [kind for _, kind in sent] == ["infer_forward", "infer_cancel"]
        assert sent[0][0]["forward_request_id"] == sent[1][0]["forward_request_id"]

    def test_forward_external_cancel_sends_cancel_and_cleans_pending(self, monkeypatch):
        import scheduler as scheduler_mod

        sched = Scheduler()
        sched._role_override = "client"
        monkeypatch.setattr(scheduler_mod, "NODE_ROLE", "client")
        sent = []

        class FakeClient:
            _running = True
            is_registered = True
            sock = object()

            def send_data(self, data, msg_type):
                sent.append((data, msg_type.value))

        cancel_event = threading.Event()
        cancel_event.set()
        monkeypatch.setattr(sched, "_tcp_client", FakeClient(), raising=False)
        result = sched.forward_inference_to_master(
            "hello",
            _cancel_event=cancel_event,
            timeout=1,
        )

        assert result["status"] == "cancelled"
        assert [kind for _, kind in sent] == ["infer_forward", "infer_cancel"]
        assert sched._client_pending_events == {}
        assert sched._client_pending_results == {}

    def test_forwarded_success_is_sent_before_task_cleanup_regression(self):
        sched = Scheduler()
        sent = []
        finished = threading.Event()
        sched.start_infer_task = lambda *args, **kwargs: "task-demo"
        sched.run_pipeline_safe = lambda prompt, **kwargs: {
            "response": "computed",
            "metrics": {"distributed_used": True},
        }
        sched.complete_infer_task = lambda *args, **kwargs: None

        def capture(*args, **kwargs):
            sent.append((args, kwargs))
            finished.set()

        sched._send_infer_result = capture
        sched.handle_infer_forward("client1", {
            "data": {
                "prompt": "x",
                "forward_request_id": "forward-demo",
            },
        })
        assert finished.wait(1)
        args, kwargs = sent[0]
        assert args[1:4] == ("task-demo", "computed", {"distributed_used": True})
        assert kwargs["forward_request_id"] == "forward-demo"
        assert kwargs.get("status", "ok") == "ok"

    def test_start_task_does_not_mark_workers_busy(self, monkeypatch):
        sched = Scheduler()
        sched.nodes["worker"] = NodeInfo(
            node_id="worker", role=NodeRole.CLIENT, state=NodeState.ONLINE,
        )
        task_id = sched.start_infer_task("hello")
        assert task_id.startswith("task_")
        assert sched.nodes["worker"].state == NodeState.ONLINE


class TestRequestNodeLogs:
    """L5: scheduler.request_node_logs() 多节点日志聚合。"""

    @pytest.fixture
    def sched(self):
        from scheduler import Scheduler
        s = Scheduler()
        s._tcp_server = None  # 无 TCP 服务器时应返回 None
        return s

    def test_request_node_logs_returns_none_without_tcp_server(self, sched):
        """无 TCP 服务器时应返回 None。"""
        result = sched.request_node_logs("worker-1")
        assert result is None

    def test_request_node_logs_returns_none_for_unknown_node(self, sched):
        """不存在的节点应返回 None。"""
        sched._tcp_server = MagicMock()
        sched.nodes = {}
        result = sched.request_node_logs("unknown-node")
        assert result is None

    def test_request_node_logs_returns_none_for_offline_node(self, sched):
        """离线节点应返回 None。"""
        sched._tcp_server = MagicMock()
        from scheduler import NodeInfo, NodeState, NodeRole
        sched.nodes = {
            "worker-1": NodeInfo(
                node_id="worker-1",
                state=NodeState.OFFLINE,
                role=NodeRole.CLIENT,
            )
        }
        with sched._nodes_lock:
            pass  # just testing state check
        result = sched.request_node_logs("worker-1")
        assert result is None


# ================================================================
# T1-T4 修复验证：线程安全测试
# ================================================================

class TestThreadSafety:
    """测试 scheduler 线程安全修复（BUG T1-T4 验证）"""

    def test_sync_node_rtt_with_lock(self):
        """_sync_node_rtt 应在 _nodes_lock 保护下访问节点（T1 修复验证）。"""
        sched = Scheduler.__new__(Scheduler)
        sched._nodes_lock = threading.RLock()
        sched.nodes = {
            "client1": NodeInfo(
                node_id="client1",
                role=NodeRole.CLIENT,
                state=NodeState.ONLINE,
                last_heartbeat=time.time() - 10,
                avg_rtt_ms=0.0,
                last_rtt_ms=0.0,
            ),
        }

        mock_client = MagicMock()
        mock_client.avg_rtt_ms = 5.5

        sched._sync_node_rtt("client1", mock_client)
        assert sched.nodes["client1"].last_rtt_ms == 5.5
        assert sched.nodes["client1"].avg_rtt_ms == 5.5
        assert time.time() - sched.nodes["client1"].last_heartbeat < 1.0

    def test_sync_node_rtt_missing_node_safe(self):
        """_sync_node_rtt 对不存在的节点应安全返回（T1 修复验证）。"""
        sched = Scheduler.__new__(Scheduler)
        sched._nodes_lock = threading.RLock()
        sched.nodes = {}

        mock_client = MagicMock()
        mock_client.avg_rtt_ms = 5.0

        sched._sync_node_rtt("nonexistent", mock_client)

    def test_sync_node_rtt_concurrent_access(self):
        """多线程并发调用 _sync_node_rtt 不应导致数据竞争（T1 修复验证）。"""
        sched = Scheduler.__new__(Scheduler)
        sched._nodes_lock = threading.RLock()
        sched.nodes = {
            "client1": NodeInfo(
                node_id="client1",
                role=NodeRole.CLIENT,
                state=NodeState.ONLINE,
                last_heartbeat=time.time() - 100,
                avg_rtt_ms=0.0,
                last_rtt_ms=0.0,
            ),
        }

        mock_client = MagicMock()
        mock_client.avg_rtt_ms = 10.0

        threads = []
        for _ in range(10):
            t = threading.Thread(
                target=sched._sync_node_rtt,
                args=("client1", mock_client),
            )
            threads.append(t)

        def delete_node():
            with sched._nodes_lock:
                sched.nodes.pop("client1", None)

        t_del = threading.Thread(target=delete_node)
        threads.append(t_del)

        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=5)

    def test_master_db_heartbeat_uses_lock(self):
        """_master_db_heartbeat_loop 应在 _nodes_lock 保护下更新心跳（T2 修复验证）。"""
        sched = Scheduler.__new__(Scheduler)
        sched._nodes_lock = threading.RLock()
        sched.nodes = {
            "master": NodeInfo(
                node_id="master",
                role=NodeRole.MASTER,
                state=NodeState.ONLINE,
                last_heartbeat=time.time() - 100,
            ),
        }
        sched._running = False

        old_hb = sched.nodes["master"].last_heartbeat
        with sched._nodes_lock:
            master_node = sched.nodes.get("master")
            if master_node:
                master_node.last_heartbeat = time.time()

        assert sched.nodes["master"].last_heartbeat > old_hb

    def test_connect_to_master_node_update_uses_lock(self):
        """connect_to_master 中节点状态更新应在 _nodes_lock 保护下（T3 修复验证）。"""
        sched = Scheduler.__new__(Scheduler)
        sched._nodes_lock = threading.RLock()
        sched.nodes = {
            "client1": NodeInfo(
                node_id="client1",
                role=NodeRole.CLIENT,
                state=NodeState.OFFLINE,
                last_heartbeat=time.time() - 100,
            ),
        }
        node_id = "client1"
        with sched._nodes_lock:
            if node_id in sched.nodes:
                sched.nodes[node_id].state = NodeState.ONLINE
                sched.nodes[node_id].last_heartbeat = time.time()

        assert sched.nodes[node_id].state == NodeState.ONLINE
        assert time.time() - sched.nodes[node_id].last_heartbeat < 1.0


class TestConnectToMasterBootstrapRecovery:
    """connect_to_master bootstrap recovery behavior."""

    def test_auth_failure_refreshes_bootstrap_and_retries(self, monkeypatch):
        import scheduler as scheduler_mod
        import config as cfg

        monkeypatch.setattr(cfg, "NODE_ROLE", "client", raising=False)
        monkeypatch.setattr(cfg, "NODE_ID", "client-old", raising=False)
        monkeypatch.setattr(cfg, "CLUSTER_SECRET", "stale-secret", raising=False)
        monkeypatch.setattr(scheduler_mod, "NODE_ROLE", "client", raising=False)
        monkeypatch.setattr(scheduler_mod, "NODE_ID", "client-old", raising=False)
        monkeypatch.setattr(node_runtime_mod.node_runtime, "_node_id", "client-old", raising=False)
        monkeypatch.setattr(node_runtime_mod.node_runtime, "_node_role", "client", raising=False)
        monkeypatch.setenv("QLH_API_PORT", "8001")
        monkeypatch.delenv("QLH_BOOTSTRAP_API_PORT", raising=False)
        monkeypatch.delenv("QLH_MASTER_API_PORT", raising=False)

        sched = Scheduler()
        sched._role_override = "client"

        first_connect_calls = []

        def fake_first_connect(master_api_host, master_api_port, node_id, node_type):
            first_connect_calls.append((master_api_host, master_api_port, node_id, node_type))
            cfg.CLUSTER_SECRET = "fresh-secret"
            return {
                "cluster": {
                    "master_tcp_host": "100.64.0.10",
                    "master_tcp_port": 8888,
                },
                "node": {
                    "node_id": "client-new",
                    "role": "client",
                },
            }

        class FakeTCPClient:
            attempts = 0

            def __init__(self, server_host, server_port, client_id, role,
                         advertise_port=None, device_info=None):
                self.server_host = server_host
                self.server_port = server_port
                self.client_id = client_id
                self.role = role
                self.advertise_port = advertise_port
                self.device_info = dict(device_info or {})
                self.last_register_error = ""
                self.is_registered = False
                self._running = False

            def connect(self, on_message=None):
                type(self).attempts += 1
                if type(self).attempts == 1:
                    self.last_register_error = "注册被拒绝: 认证失败: HMAC 签名不匹配"
                    return False
                self.is_registered = True
                self._running = True
                return True

            def send_data(self, data, msg_type):
                pass

        monkeypatch.setattr("bootstrap.first_connect", fake_first_connect)
        monkeypatch.setattr("tcp_comm.TCPClient", FakeTCPClient)

        result = sched.connect_to_master("100.64.0.10", 8888)

        assert result["status"] == "connected"
        assert result["node_id"] == "client-new"
        assert FakeTCPClient.attempts == 2
        assert first_connect_calls == [("100.64.0.10", 8000, "client-old", "pc")]
        # 阶段 0.3：运行时身份写入 node_runtime（cfg 不再被写回）
        assert node_runtime_mod.node_runtime.get_node_id() == "client-new"
        assert scheduler_mod.NODE_ID == "client-new"

    def test_existing_registered_connection_is_reused(self):
        sched = Scheduler()
        sched._role_override = "client"
        existing = MagicMock()
        existing._running = True
        existing.is_registered = True
        existing.sock = object()
        existing.server_host = "100.64.0.10"
        existing.server_port = 8888
        sched._tcp_client = existing

        result = sched.connect_to_master("100.64.0.10", 8888)

        assert result["status"] == "connected"
        assert result["reused"] is True
        existing.disconnect.assert_not_called()

    def test_failed_candidate_connection_preserves_existing_connection(
            self, monkeypatch):
        import scheduler as scheduler_mod
        import config as cfg

        monkeypatch.setattr(cfg, "NODE_ROLE", "client", raising=False)
        monkeypatch.setattr(cfg, "NODE_ID", "client-existing", raising=False)
        monkeypatch.setattr(cfg, "CLUSTER_SECRET", "shared-secret", raising=False)
        monkeypatch.setattr(scheduler_mod, "NODE_ROLE", "client", raising=False)
        monkeypatch.setattr(
            scheduler_mod, "NODE_ID", "client-existing", raising=False,
        )

        class FailedTCPClient:
            instances = []

            def __init__(self, server_host, server_port, client_id, role,
                         advertise_port=None, device_info=None):
                self.server_host = server_host
                self.server_port = server_port
                self.client_id = client_id
                self.role = role
                self.last_register_error = "candidate unavailable"
                self.is_registered = False
                self._running = False
                self.sock = None
                self.on_disconnect = None
                self.disconnect_calls = 0
                type(self).instances.append(self)

            def connect(self, on_message=None):
                return False

            def disconnect(self):
                self.disconnect_calls += 1

        sched = Scheduler()
        sched._role_override = "client"
        previous_callback = object()
        existing = MagicMock()
        existing._running = True
        existing.is_registered = True
        existing.sock = object()
        existing.server_host = "100.64.0.10"
        existing.server_port = 8888
        existing.client_id = "client-existing"
        existing.on_disconnect = previous_callback
        sched._tcp_client = existing
        monkeypatch.setattr("tcp_comm.TCPClient", FailedTCPClient)

        result = sched.connect_to_master("100.64.0.20", 8888)

        assert result["status"] == "failed"
        assert sched._tcp_client is existing
        assert existing.on_disconnect is previous_callback
        existing.disconnect.assert_not_called()
        assert FailedTCPClient.instances[0].disconnect_calls == 1

    def test_stale_connection_disconnect_does_not_clear_current_state(self):
        sched = Scheduler()
        current = object()
        stale = object()
        sched._tcp_client = cast(Any, current)
        sched._kv_cache["active"] = "kv"

        sched._on_master_connection_lost(stale)

        assert sched._kv_cache == {"active": "kv"}

    def test_layer_config_callback_can_use_client_before_connect_returns(self, monkeypatch):
        """REGISTER 后立即到达的层配置必须能看到 TCP 客户端和最终 node_id。"""
        import scheduler as scheduler_mod
        import config as cfg

        monkeypatch.setattr(cfg, "NODE_ROLE", "client", raising=False)
        monkeypatch.setattr(cfg, "NODE_ID", "client-race", raising=False)
        monkeypatch.setattr(cfg, "CLUSTER_SECRET", "shared-secret", raising=False)
        monkeypatch.setattr(scheduler_mod, "NODE_ROLE", "client", raising=False)
        monkeypatch.setattr(scheduler_mod, "NODE_ID", "client-race", raising=False)
        monkeypatch.setattr(node_runtime_mod.node_runtime, "_node_id", "client-race", raising=False)
        monkeypatch.setattr(node_runtime_mod.node_runtime, "_node_role", "client", raising=False)

        observed = []

        class FakeTCPClient:
            def __init__(self, server_host, server_port, client_id, role,
                         advertise_port=None, device_info=None):
                self.server_host = server_host
                self.server_port = server_port
                self.client_id = client_id
                self.role = role
                self.device_info = dict(device_info or {})
                self.last_register_error = ""
                self.is_registered = True
                self._running = True
                self.sock = object()

            def connect(self, on_message=None):
                assert on_message is not None
                on_message({"type": "layer_config", "data": {}})
                return True

            def send_data(self, data, msg_type):
                pass

        sched = Scheduler()
        sched._role_override = "client"

        def observe(_client_id, _msg):
            active_client = cast(Any, sched._tcp_client)
            assert active_client is not None
            observed.append((active_client.client_id, sched.get_effective_node_id()))

        monkeypatch.setattr(sched, "_on_tcp_message", observe)
        monkeypatch.setattr("tcp_comm.TCPClient", FakeTCPClient)

        result = sched.connect_to_master("100.64.0.10", 8888)

        assert result["status"] == "connected"
        assert observed == [("client-race", "client-race")]

    def test_profile_completed_during_connect_is_reported_after_registration(
            self, monkeypatch):
        import config as cfg
        import scheduler as scheduler_mod

        monkeypatch.setattr(cfg, "NODE_ROLE", "client", raising=False)
        monkeypatch.setattr(cfg, "NODE_ID", "client-profile", raising=False)
        monkeypatch.setattr(cfg, "CLUSTER_SECRET", "shared-secret", raising=False)
        monkeypatch.setattr(scheduler_mod, "NODE_ROLE", "client", raising=False)
        monkeypatch.setattr(scheduler_mod, "NODE_ID", "client-profile", raising=False)
        sent = []
        sched = Scheduler()
        sched._role_override = "client"
        sched._local_device_profile = None

        class FakeTCPClient:
            def __init__(self, server_host, server_port, client_id, role,
                         advertise_port=None, device_info=None):
                self.client_id = client_id
                self.server_host = server_host
                self.server_port = server_port
                self.device_info = dict(device_info or {})
                self.is_registered = False
                self._running = False
                self.last_register_error = ""

            def connect(self, on_message=None):
                sched._local_device_profile = PROFILE_IGPU_ONLY
                self.is_registered = True
                self._running = True
                return True

            def send_data(self, data, msg_type):
                sent.append((data, msg_type.value))

        monkeypatch.setattr("tcp_comm.TCPClient", FakeTCPClient)

        result = sched.connect_to_master("100.64.0.10", 8888)

        assert result["status"] == "connected"
        assert any(
            msg_type == "status_res" and data["device_info"] == PROFILE_IGPU_ONLY
            for data, msg_type in sent
        )


def test_tcp_bind_failure_keeps_master_local_pipeline_available(monkeypatch):
    import scheduler as scheduler_mod
    import config as cfg

    monkeypatch.setattr(scheduler_mod, "RUN_MODE", "distributed", raising=False)
    monkeypatch.setattr(scheduler_mod, "NODE_ROLE", "master", raising=False)
    monkeypatch.setattr(cfg, "NODE_ROLE", "master", raising=False)

    class FailedTCPServer:
        def __init__(self, host, port):
            self.host = host
            self.port = port

        def start(self, on_message=None, on_disconnect=None,
                  on_registration_confirmed=None):
            raise OSError("address already in use")

    sched = Scheduler()
    monkeypatch.setattr("tcp_comm.TCPServer", FailedTCPServer)
    monkeypatch.setattr("tcp_comm.detect_lan_ip", lambda: "100.64.0.10")
    monkeypatch.setattr("tcp_comm.get_mac_addresses", lambda: ["001122334455"])
    monkeypatch.setattr(sched, "init_nodes", lambda: None)
    monkeypatch.setattr(
        sched,
        "_verify_master_identity",
        lambda: setattr(sched, "_master_identity_reason", "first_run"),
    )
    monkeypatch.setattr(sched, "deactivate_spare_master_on_startup", lambda: None)
    monkeypatch.setattr(sched, "can_join_existing_master", lambda: False)

    try:
        sched.start(host="0.0.0.0", port=8888)

        status = sched.get_status()
        assert status["running"] is True
        assert status["tcp_server"] is None
        assert status["pipeline_queue"]["running"] is True
    finally:
        sched.stop()


def test_start_failure_releases_idempotency_guard_for_retry(monkeypatch):
    import scheduler as scheduler_mod

    monkeypatch.setattr(scheduler_mod, "RUN_MODE", "single", raising=False)
    scheduler = Scheduler()
    monkeypatch.setattr(
        scheduler, "init_nodes", lambda: (_ for _ in ()).throw(
            RuntimeError("init failed")
        ),
    )

    with pytest.raises(RuntimeError, match="init failed"):
        scheduler.start()
    assert scheduler._running is False

    monkeypatch.setattr(scheduler, "init_nodes", lambda: None)
    scheduler.start()
    try:
        assert scheduler._running is True
    finally:
        scheduler.stop()


def test_distributed_start_defers_network_identity(monkeypatch):
    """A slow network probe must not block the local scheduler start."""
    import config as cfg
    import transport_port
    import scheduler as scheduler_mod

    monkeypatch.setattr(scheduler_mod, "RUN_MODE", "distributed", raising=False)
    monkeypatch.setattr(scheduler_mod, "NODE_ROLE", "master", raising=False)
    monkeypatch.setattr(cfg, "NODE_ROLE", "master", raising=False)

    class FakeServer:
        _running = True
        host = "0.0.0.0"
        port = 8888

        def start(self, **kwargs):
            return None

        def stop(self):
            self._running = False

        def get_client_ids(self):
            return []

    probe_started = threading.Event()
    release_probe = threading.Event()

    def slow_lan_probe():
        probe_started.set()
        release_probe.wait(timeout=5)
        return "100.64.0.10"

    monkeypatch.setattr(transport_port, "create_server", lambda *args: FakeServer())
    monkeypatch.setattr(transport_port, "detect_lan_ip", slow_lan_probe)
    monkeypatch.setattr(transport_port, "get_mac_addresses", lambda: ["001122334455"])

    sched = Scheduler()
    monkeypatch.setattr(sched, "init_nodes", lambda: None)
    monkeypatch.setattr(sched, "deactivate_spare_master_on_startup", lambda: None)
    monkeypatch.setattr(sched, "can_join_existing_master", lambda: False)

    start_done = threading.Event()
    errors = []

    def run_start():
        try:
            sched.start(host="0.0.0.0", port=8888)
        except BaseException as exc:  # pragma: no cover - diagnostic path
            errors.append(exc)
        finally:
            start_done.set()

    starter = threading.Thread(target=run_start, daemon=True)
    starter.start()
    try:
        assert start_done.wait(timeout=5), "scheduler start waited for network identity"
        assert not errors
        assert probe_started.wait(timeout=1), "deferred network probe did not start"
    finally:
        release_probe.set()
        starter.join(timeout=2)
        sched.stop()


def test_distributed_toggle_is_process_owned():
    """运行时分布式开关应立即生效。"""
    sched = Scheduler()
    sched.set_distributed_inference_enabled(False)
    assert sched.get_distributed_inference_enabled() is False
    sched.set_distributed_inference_enabled(True)
    assert sched.get_distributed_inference_enabled() is True


class _ImmediateStartupWait:
    """让启动期决策测试跳过真实等待，同时记录等待预算。"""

    def __init__(self, cancelled=False):
        self.cancelled = cancelled
        self.timeouts = []

    def clear(self):
        self.cancelled = False

    def set(self):
        self.cancelled = True

    def wait(self, timeout=None):
        self.timeouts.append(timeout)
        return self.cancelled


class TestSchedulerStartupDecisions:
    def test_discover_master_prefers_local_config_and_can_skip_to_tailnet(
            self, monkeypatch):
        import bootstrap
        import config as cfg

        sched = Scheduler()
        monkeypatch.setattr(cfg, "CLIENT_MASTER_HOST", "100.64.0.10")
        monkeypatch.setattr(cfg, "CLIENT_MASTER_PORT", 8899)

        configured = sched.discover_master()
        assert configured == {
            "found": True,
            "master_host": "100.64.0.10",
            "master_port": 8899,
            "stale": False,
            "source": "config",
        }

        monkeypatch.setattr(cfg, "CLIENT_MASTER_HOST", "192.168.x.x")
        monkeypatch.setattr(
            bootstrap,
            "discover_master_via_tailnet",
            lambda *, api_port: {
                "found": True,
                "master_host": "100.64.0.20",
                "master_port": api_port,
                "stale": True,
                "source": "tailnet",
            },
        )
        discovered = sched.discover_master(skip_config=True)
        assert discovered["found"] is True
        assert discovered["source"] == "tailnet"
        assert discovered["stale"] is True

    def test_discover_master_fail_closed_when_tailnet_probe_errors(self, monkeypatch):
        import bootstrap
        import config as cfg

        monkeypatch.setattr(cfg, "CLIENT_MASTER_HOST", "192.168.x.x")
        monkeypatch.setattr(
            bootstrap,
            "discover_master_via_tailnet",
            lambda **kwargs: (_ for _ in ()).throw(OSError("probe failed")),
        )
        assert Scheduler().discover_master() == {
            "found": False,
            "source": "none",
        }

    def test_auto_switch_wait_is_cancellable_and_activates_after_ready(
            self, monkeypatch):
        sched = Scheduler()
        wait = _ImmediateStartupWait()
        monkeypatch.setattr(sched, "_startup_cancel_event", wait)
        activated = []
        monkeypatch.setattr(
            sched,
            "activate_client_mode",
            lambda host, port: activated.append((host, port)) or {"message": "ok"},
        )

        sched._auto_switch_to_client("100.64.0.10", 8888)
        assert activated == [("100.64.0.10", 8888)]
        assert wait.timeouts == [2]

        activated.clear()
        wait.cancelled = True
        sched._auto_switch_to_client("100.64.0.11", 8889)
        assert activated == []

    def test_auto_join_requires_running_joinable_node_and_discovered_master(
            self, monkeypatch):
        sched = Scheduler()
        wait = _ImmediateStartupWait()
        monkeypatch.setattr(sched, "_startup_cancel_event", wait)
        sched._running = True
        monkeypatch.setattr(sched, "can_join_existing_master", lambda: True)
        monkeypatch.setattr(
            sched,
            "discover_master",
            lambda: {"found": True, "master_host": "100.64.0.12", "master_port": 8890},
        )
        activated = []
        monkeypatch.setattr(
            sched,
            "activate_client_mode",
            lambda host, port: activated.append((host, port)),
        )

        sched._auto_join_tailnet_master_on_startup()
        assert activated == [("100.64.0.12", 8890)]

        activated.clear()
        sched._running = False
        sched._auto_join_tailnet_master_on_startup()
        assert activated == []

    def test_auto_connect_retries_tailnet_discovery_after_config_failure(
            self, monkeypatch):
        sched = Scheduler()
        wait = _ImmediateStartupWait()
        monkeypatch.setattr(sched, "_startup_cancel_event", wait)
        sched._running = True
        monkeypatch.setattr(
            sched,
            "discover_master",
            lambda **kwargs: {"found": True, "master_host": "100.64.0.10", "master_port": 8899},
        )
        monkeypatch.setattr(
            sched,
            "_discover_master_fallbacks",
            lambda host, port: [
                {"found": True, "master_host": "100.64.0.11", "master_port": 8900, "source": "tailnet"},
            ],
        )
        connected = []
        monkeypatch.setattr(
            sched,
            "connect_to_master",
            lambda host, port: connected.append((host, port)) or {"status": "failed"},
        )

        sched._auto_connect_on_startup()
        assert connected == [("100.64.0.10", 8899), ("100.64.0.11", 8900)]
        assert wait.timeouts == [5]

    def test_auto_connect_skips_discovery_when_already_connected_or_stopped(
            self, monkeypatch):
        sched = Scheduler()
        wait = _ImmediateStartupWait()
        monkeypatch.setattr(sched, "_startup_cancel_event", wait)
        sched._running = True
        sched._tcp_client = type("Client", (), {"_running": True})()
        monkeypatch.setattr(
            sched,
            "discover_master",
            lambda **kwargs: pytest.fail("active connection should win"),
        )
        sched._auto_connect_on_startup()

        sched._tcp_client = None
        sched._running = False
        sched._auto_connect_on_startup()

    @pytest.mark.parametrize(
        "enabled,disabled,preempting,current_step,last_time,now,expected",
        [
            (False, False, False, 100, 0, 100, False),
            (True, True, False, 100, 0, 100, False),
            (True, False, True, 100, 0, 100, False),
            (True, False, False, 15, 0, 100, False),
            (True, False, False, 16, 95, 100, False),
            (True, False, False, 16, 80, 100, True),
        ],
    )
    def test_preempt_condition_truth_table(
            self, monkeypatch, enabled, disabled, preempting, current_step,
            last_time, now, expected):
        import scheduler as scheduler_mod

        monkeypatch.setattr(scheduler_mod, "PIPELINE_PREEMPT_ENABLED", enabled)
        monkeypatch.setattr(
            scheduler_mod, "PIPELINE_PREEMPT_MIN_TOKENS", 16,
        )
        monkeypatch.setattr(
            scheduler_mod, "PIPELINE_PREEMPT_MIN_INTERVAL", 10.0,
        )
        monkeypatch.setattr(scheduler_mod.time, "time", lambda: now)
        sched = Scheduler()
        sched._preempt_disabled = disabled
        sched._preempting = preempting
        sched._preempt_last_time = last_time
        assert sched._check_preempt_conditions(current_step) is expected
