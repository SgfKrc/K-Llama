"""阶段 0 复核补充测试：API 响应 key 契约（防误替换回归）

阶段 0.2 迁移 model_loaded/current_quant 等运行时状态到 model_host 时，
响应字典的 key 曾被误替换为字面量 "model_host.current_quant"（破坏前端
MetricsPanel/ChatPanel 读取）。本测试锁定关键端点响应 key 不被迁移破坏。
"""
import os
import sys
from types import SimpleNamespace

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import pytest
from fastapi.testclient import TestClient

import api_server as api_server_mod
from api_server import app
from model_host import model_host


@pytest.fixture
def client():
    return TestClient(app)


class TestApiResponseKeys:
    """响应 JSON 的字段名必须保持 api_server 既有契约（前端/TUI 依赖）。"""

    def test_presets_returns_current_quant_key(self, client):
        res = client.get("/api/presets")
        assert res.status_code == 200
        body = res.json()
        assert "current_quant" in body, "key 被误替换为 model_host.current_quant"
        assert "presets" in body
        assert "max_new_tokens" in body

    def test_status_returns_current_quant_key(self, client, monkeypatch):
        monkeypatch.setattr(api_server_mod, "model_manager", SimpleNamespace(is_loaded=True, active_model_id="fixture", is_pipeline_prepared=False, get_model_info=lambda: {"model_id":"fixture"}))
        monkeypatch.setattr(model_host, "model_loaded", True)
        res = client.get("/api/status")
        assert res.status_code == 200
        body = res.json()
        assert "current_quant" in body, "key 被误替换为 model_host.current_quant"
        assert "model_loaded" in body
        # 迁移后值来自 host（与 API 层一致）
        assert body["model_loaded"] is model_host.model_loaded

    def test_status_quant_value_follows_host(self, client, monkeypatch):
        monkeypatch.setattr(api_server_mod, "model_manager", SimpleNamespace(is_loaded=True, active_model_id="fixture", is_pipeline_prepared=False, get_model_info=lambda: {"model_id":"fixture"}))
        monkeypatch.setattr(model_host, "model_loaded", True)
        monkeypatch.setattr(model_host, "current_quant", "int8")
        res = client.get("/api/status")
        assert res.json()["current_quant"] == "int8"

    def test_health_shape(self, client):
        res = client.get("/api/health")
        assert res.status_code == 200
        body = res.json()
        assert set(body.keys()) == {"status", "timestamp"}

    def test_readiness_is_separate_from_health(self, client):
        health = client.get("/api/health")
        readiness = client.get("/api/ready")
        assert set(health.json().keys()) == {"status", "timestamp"}
        body = readiness.json()
        assert {"process_ready", "ready", "status", "components"} <= set(body)
        assert set(body["components"]) == {
            "local_store", "scheduler", "device_profile",
        }

    def test_auth_capability_is_explicit_when_running_direct_api(self, client, monkeypatch):
        """★ 2026-09-19：认证改为 **monolith 内实现**（抛弃 control-svc 反代）。

        原契约假设认证由独立 control-svc 承载 ⇒ `available=False` +
        `reason_code="auth_control_plane_unavailable"`，并允许用 `QLH_CONTROL_URL` 探测。
        改造后**不再有反代层**，本进程**自带**认证实现 ⇒ `available=True`、`service="api_server"`，
        且 `QLH_CONTROL_URL` **不再被读取**。

        保留的原意：**端点必须显式返回能力对象，而不是 404**（让 UI 能区分
        “未启用认证”与“端点不存在”）。
        """
        # 该变量已不再影响行为（回归断言：设了也不改变结果）
        monkeypatch.setenv("QLH_CONTROL_URL", "http://127.0.0.1:1")
        res = client.get("/api/auth/capability")
        assert res.status_code == 200, "应显式返回 200 能力对象，而不是 404"
        body = res.json()
        assert body["service"] == "api_server", "实现已回到 monolith 进程内"
        assert body["available"] is True, "本进程自带认证实现"
        assert body["mode"] == "local_totp"
        # 默认不强制登录（保持既有可用性）
        assert body["required"] is False
        assert body["enforced"] is False
        # 新增字段：首次引导可见性（原控制面的 bootstrap 能力的本地替代）
        assert "bootstrap_open" in body and "user_count" in body
        # 不再有「控制面不可用」这一失败态
        assert "reason_code" not in body or body.get("reason_code") != "auth_control_plane_unavailable"

    def test_storage_health_is_local_first_and_retired_remote(self, client, monkeypatch):
        monkeypatch.setattr(
            api_server_mod._local_store,
            "local_store_health",
            lambda: {"status": "ok", "backend": "sqlite", "writable": True},
        )
        res = client.get("/api/storage/health")
        assert res.status_code == 200
        body = res.json()
        assert body["local"]["backend"] == "sqlite"
        assert body["effective_mode"] == "local_only"
        assert body["remote"] == {"status": "retired", "backend": "postgresql", "mode": "retired"}
        assert body["projection"]["pending_events"] == 0

    def test_speculative_capability_is_zero_network_and_fail_closed(self, client, monkeypatch):
        res = client.get("/api/experimental/speculative/capability")
        assert res.status_code == 410
        assert res.json()["detail"]["code"] == "CLIENT_FEATURE_REMOVED"

    def test_android_heartbeat_returns_lease_contract(self, client, monkeypatch):
        monkeypatch.setattr(
            api_server_mod.scheduler,
            "heartbeat_android_client",
            lambda **kwargs: {
                "status": "heartbeat",
                "node_id": kwargs["node_id"],
                "state": "online",
                "server_time_ms": 1700000000000,
                "presence_generation": kwargs["presence_generation"],
                "presence_lease_id": kwargs["presence_lease_id"],
                "lease_expires_at_ms": 1700000120000,
                "heartbeat_interval_seconds": 45,
            },
        )
        response = client.post(
            "/api/cluster/android/heartbeat",
            json={
                "node_id": "android-1",
                "presence_generation": 2,
                "presence_lease_id": "lease-2",
            },
        )
        assert response.status_code == 200
        body = response.json()
        assert body["status"] == "heartbeat"
        assert body["presence_generation"] == 2
        assert body["heartbeat_interval_seconds"] == 45

    def test_android_heartbeat_rejection_has_stable_error_code(self, client, monkeypatch):
        monkeypatch.setattr(
            api_server_mod.scheduler,
            "heartbeat_android_client",
            lambda **kwargs: {
                "status": "rejected",
                "reason": "stale",
                "error_code": "stale_generation",
            },
        )
        response = client.post(
            "/api/cluster/android/heartbeat",
            json={"node_id": "android-1", "presence_generation": 1, "presence_lease_id": "old"},
        )
        assert response.status_code == 409
        assert response.json()["error_code"] == "stale_generation"

    @staticmethod
    def _cluster_status_payload(network_path=None):
        payload = {
            "run_mode": "single",
            "nodes_ready": True,
            "nodes": {
                "master": {
                    "node_id": "master",
                    "role": "master",
                    "state": "online",
                },
            },
            "current_task": None,
            "tcp_server": None,
            "pipeline": None,
            "pipeline_queue": None,
        }
        if network_path is not None:
            payload["network_path"] = network_path
            payload["nodes"]["master"]["network_path"] = network_path
        return payload

    def test_cluster_status_omits_optional_network_path_for_old_snapshot(
            self, client, monkeypatch):
        monkeypatch.setattr(
            api_server_mod.scheduler,
            "get_status",
            lambda: self._cluster_status_payload(),
        )

        response = client.get("/api/cluster/status")

        assert response.status_code == 200
        body = response.json()
        assert "network_path" not in body
        assert "network_path" not in body["nodes"]["master"]
        assert set(body) == {
            "run_mode", "nodes_ready", "nodes", "current_task",
            "tcp_server", "pipeline", "pipeline_queue",
        }

    def test_cluster_status_projects_network_path_without_changing_old_keys(
            self, client, monkeypatch):
        network_path = {
            "schema_version": 1,
            "path_kind": "derp",
            "availability": "available",
            "endpoint": {"role": "master", "host_scope": "tailscale_ipv4", "port": 8888},
            "tailscale": None,
            "tcp_probe": {"state": "available", "reason": "existing_connection", "elapsed_ms": None},
            "quality": {"schema_version": 1, "rtt_ms_p95": 30.0},
        }
        monkeypatch.setattr(
            api_server_mod.scheduler,
            "get_status",
            lambda: self._cluster_status_payload(network_path),
        )

        response = client.get("/api/cluster/status")

        assert response.status_code == 200
        body = response.json()
        assert body["network_path"] == network_path
        assert body["nodes"]["master"]["network_path"] == network_path
        assert body["nodes_ready"] is True

    def test_cluster_status_projects_pipeline_capacity_and_node_participation(
            self, client, monkeypatch):
        """「在线但不干活」必须在 /api/cluster/status 上可见（2026-10-08 第 4 条静默路径）。"""
        payload = self._cluster_status_payload()
        payload["pipeline_capacity"] = {
            "status": "rejected",
            "admitted": False,
            "reason_code": "pipeline_distributed_workers_unavailable",
            "participating_node_ids": [],
            "control_only_nodes": ["master"],
            "worker_count": 0,
            "prepared_node_count": 0,
            "ready_node_count": 0,
        }
        payload["nodes"]["master"]["pipeline_participating"] = False
        payload["nodes"]["master"]["pipeline_exclusion_reason"] = (
            "capacity_plan_control_only"
        )
        monkeypatch.setattr(
            api_server_mod.scheduler,
            "get_status",
            lambda: payload,
        )

        response = client.get("/api/cluster/status")

        assert response.status_code == 200
        body = response.json()
        assert body["pipeline_capacity"]["admitted"] is False
        assert body["pipeline_capacity"]["reason_code"] == (
            "pipeline_distributed_workers_unavailable"
        )
        assert body["pipeline_capacity"]["worker_count"] == 0
        assert body["nodes"]["master"]["pipeline_participating"] is False
        assert body["nodes"]["master"]["pipeline_exclusion_reason"] == (
            "capacity_plan_control_only"
        )

    def test_cluster_resources_is_read_only_aggregate_projection(
            self, client, monkeypatch):
        projection = {
            "schema_version": 1,
            "scope": "cluster",
            "is_distributed": True,
            "node_count": 2,
            "available_node_count": 2,
            "remote_available_count": 1,
            "available": {"local": {"node_id": "master"}, "remote": []},
            "totals": {"logical_cores": 16, "ram_available_gb": 20},
        }
        monkeypatch.setattr(
            api_server_mod.scheduler,
            "get_aggregate_resource_view",
            lambda: dict(projection),
        )

        response = client.get("/api/cluster/resources")

        assert response.status_code == 200
        assert response.json() == projection
