"""★ 2026-09-30：**非 release** 的层配置 ACK 也必须回显 `generation`。

与 `test_peer_release_ack.py`（release ACK）同源但不同分支：主节点
`scheduler_pipeline.py:1926-1941` 的 generation 门闩对**所有**非 release 期望都生效

    if not expected.get("release") and "generation" in expected:
        … int(data["generation"]) == int(expected["generation"]) …

而主节点 assignment registry 的 canonical expected contract 本就带
`generation`。worker 的 `prepared` / `ready` ACK 若不带该字段 ⇒ 被
`忽略缺少或无效 generation 的层配置 ACK` 整条丢弃 ⇒ 主节点每 60s
`重发分层配置 attempt=N`（2026-09-30 三机验收实测刷到 attempt=8、9…），
worker 侧则反复「收到分层配置 → 段配置就绪」。

这些用例不需要真模型：`prepared` 分支只用 `_layer_config_lock` / `_active_layer_config`；
`ready` 分支把 `model_sync` 与 `_host` 换成替身。
"""

from __future__ import annotations

import sys
import threading
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
for _candidate in (str(ROOT), str(ROOT / "src")):
    if _candidate not in sys.path:
        sys.path.insert(0, _candidate)

from inference_service.peer import PeerClient  # noqa: E402


def _peer(node_id: str = "client1"):
    peer = object.__new__(PeerClient)
    peer._node_id = node_id
    peer._layer_execution_lock = threading.RLock()
    peer._layer_config_lock = threading.RLock()
    peer._active_layer_config = None
    peer._local_pipeline_steps = {}
    peer._latest_layer_config_generation = 0
    peer._latest_layer_config_id = ""
    peer._latest_layer_assignment_id = ""
    peer._layer_assignment_identity_required = False
    sent: list = []
    peer._send_layer_config_ack = lambda payload: (sent.append(payload), True)[1]
    return peer, sent


def _master_generation_gate(ack: dict, expected: dict) -> bool:
    """复刻 `scheduler_pipeline.py:1926-1941` 的 generation 门闩（非 release 分支）。

    该判据内联在方法体里（无法直接 import），故照抄一份；并用**不同** generation
    反向断言它真的参与判定（不是摆设）。
    """
    if expected.get("release") or "generation" not in expected:
        return True                     # 期望未带 generation ⇒ 不校验（旧路径）
    try:
        return int(ack.get("generation")) == int(expected.get("generation"))
    except (TypeError, ValueError):
        return False


def test_relay_prepared_ack_echoes_generation():
    """relay_middle（零层）的 `prepared` ACK 必须带 `generation`。

    该红必须红：把 `"generation": …` 从 ACK 里去掉（修复前的形态），本用例在
    `_master_generation_gate(...) is True` 上失败 —— 与线上「每 60s 重发分层配置」同型。
    """
    peer, sent = _peer()

    peer._handle_layer_config({
        "start_layer": 8, "end_layer": 24,
        "node_id": "client1", "config_id": "cfg-1", "generation": 7,
        "assignment_id": "asg-prepare",
        "engine": "relay_middle", "plan_id": "plan-1",
        "model_sha256": "deadbeef", "model_type": "qwen2",
        "phase": "prepare",
    })

    assert len(sent) == 1, sent
    ack = sent[0]
    assert ack["status"] == "prepared"
    assert ack["phase"] == "prepare"
    assert ack["generation"] == 7, ack
    assert ack["assignment_id"] == "asg-prepare"
    expected = {"config_id": "cfg-1", "generation": 7, "phase": "prepare"}
    assert _master_generation_gate(ack, expected) is True, ack
    # 反向：期望别的 generation 就该判失败 ⇒ 证明字段真的参与门闩。
    assert _master_generation_gate(ack, {"generation": 6}) is False


def _master_ready_gate(ack: dict, expected: dict) -> bool:
    """复刻 `scheduler_pipeline.py:1997-2017` 的 `ready` 判据的关键项。

    `layer_range` 必须是 **list**（`expected_range = [start_layer, end_layer]`）。
    """
    return (
        str(expected.get("phase", "commit")) == "commit"
        and ack.get("status") == "ready"
        and ack.get("layer_range") == [expected["start_layer"], expected["end_layer"]]
        and ack.get("model_sha256") == expected.get("model_sha256")
        and ack.get("model_type") == expected.get("model_type")
        and ack.get("engine") == expected.get("engine", "pytorch")
    )


def test_relay_commit_ack_is_ready_with_list_range():
    """`phase="commit"` 时 relay 段必须回 `status="ready"`。

    ★ 2026-09-30：此前本分支**不区分 phase**、永远回 `prepared` ⇒ commit 阶段的
    `ready` 判据永不成立 ⇒ 主节点每 1s `重发分层配置`（worker 日志也每 1s
    「收到分层配置 → 段配置就绪」）。该红必须红：把 ACK 退回 `status="prepared"`，
    本用例在 `_master_ready_gate(...) is True` 上失败。
    """
    peer, sent = _peer()

    peer._handle_layer_config({
        "start_layer": 8, "end_layer": 24,
        "node_id": "client1", "config_id": "cfg-1", "generation": 7,
        "assignment_id": "asg-commit",
        "engine": "relay_middle", "plan_id": "plan-1",
        "model_sha256": "deadbeef", "model_type": "qwen2",
        "phase": "commit",
    })

    assert len(sent) == 1, sent
    ack = sent[0]
    assert ack["status"] == "ready", ack
    assert ack["layer_range"] == [8, 24], ack          # 必须是 list，不是 "8-24"
    assert ack["generation"] == 7, ack
    assert ack["assignment_id"] == "asg-commit"
    expected = {
        "config_id": "cfg-1", "generation": 7, "phase": "commit",
        "start_layer": 8, "end_layer": 24,
        "model_sha256": "deadbeef", "model_type": "qwen2", "engine": "relay_middle",
    }
    assert _master_ready_gate(ack, expected) is True, ack


def test_prepare_stage_is_not_reported_as_ready():
    """`phase="prepare"` 时不得提前回 `ready`（否则 `prepared` 阶段判据失效）。"""
    peer, sent = _peer()
    peer._handle_layer_config({
        "start_layer": 8, "end_layer": 24,
        "node_id": "client1", "config_id": "cfg-1", "generation": 7,
        "engine": "relay_middle", "plan_id": "plan-1",
        "model_sha256": "deadbeef", "model_type": "qwen2",
        "phase": "prepare",
    })
    assert sent[0]["status"] == "prepared"
    assert "assignment_id" not in sent[0]


def test_prepare_commit_and_retry_reuse_the_same_assignment_id():
    peer, sent = _peer()
    payload = {
        "start_layer": 8, "end_layer": 24,
        "node_id": "client1", "config_id": "cfg-stable", "generation": 8,
        "assignment_id": "asg-stable",
        "engine": "relay_middle", "plan_id": "plan-stable",
        "model_sha256": "deadbeef", "model_type": "qwen2",
        "phase": "prepare",
    }

    peer._handle_layer_config(payload)
    peer._handle_layer_config({**payload, "phase": "commit"})
    peer._handle_layer_config({**payload, "phase": "commit"})

    assert [ack["status"] for ack in sent] == ["prepared", "ready", "ready"]
    assert {ack["assignment_id"] for ack in sent} == {"asg-stable"}
    assert peer._latest_layer_assignment_id == "asg-stable"
    assert peer._active_layer_config["assignment_id"] == "asg-stable"


def test_same_generation_and_config_with_other_assignment_is_not_a_replay():
    peer, sent = _peer()
    current = {
        "start_layer": 8, "end_layer": 24,
        "node_id": "client1", "config_id": "cfg-current", "generation": 9,
        "assignment_id": "asg-current",
        "engine": "relay_middle", "model_sha256": "deadbeef",
        "model_type": "qwen2", "phase": "commit",
    }

    peer._handle_layer_config(current)
    peer._handle_layer_config({**current, "assignment_id": "asg-other"})

    assert len(sent) == 1
    assert sent[0]["assignment_id"] == "asg-current"
    assert peer._latest_layer_assignment_id == "asg-current"
    assert peer._active_layer_config["assignment_id"] == "asg-current"


def test_identity_bearing_config_makes_missing_assignment_fail_closed():
    peer, sent = _peer()
    current = {
        "start_layer": 8, "end_layer": 24,
        "node_id": "client1", "config_id": "cfg-current", "generation": 10,
        "assignment_id": "asg-current",
        "engine": "relay_middle", "model_sha256": "deadbeef",
        "model_type": "qwen2", "phase": "commit",
    }

    peer._handle_layer_config(current)
    peer._handle_layer_config({
        **current,
        "config_id": "cfg-newer",
        "generation": 11,
        "assignment_id": None,
    })
    legacy_shaped = dict(current)
    legacy_shaped.update(config_id="cfg-newer", generation=11)
    legacy_shaped.pop("assignment_id")
    peer._handle_layer_config(legacy_shaped)

    assert len(sent) == 1
    assert peer._latest_layer_config_generation == 10
    assert peer._latest_layer_config_id == "cfg-current"
    assert peer._latest_layer_assignment_id == "asg-current"
    assert peer._active_layer_config["config_id"] == "cfg-current"


class _HostInner:
    model_loaded = False

    def __init__(self) -> None:
        self.calls: list = []

    def load_model(self, **kwargs) -> None:
        self.calls.append(("load_model", kwargs))

    def load_layer_range(self, **kwargs) -> None:
        self.calls.append(("load_layer_range", kwargs))


class _Host:
    def __init__(self) -> None:
        self._host = _HostInner()

    def _ensure_device_profile(self):
        """与 `inference_service/engine_host.py` 的同名方法对齐（惰性探测，未探测时为 None）。

        `peer._handle_layer_config_locked`（peer.py:511）在解析层段模型加载时会取设备画像；
        桩缺这个方法会让该分支直接 AttributeError，与实现脱节。
        """
        return None


@pytest.fixture()
def _stub_model_sync(monkeypatch):
    """把 `model_sync` 的模型解析挡掉（ready 分支只需走到发 ACK）。"""
    import model_sync

    monkeypatch.setattr(model_sync, "resolve_worker_model_path",
                        lambda model_id: f"/tmp/{model_id}", raising=False)
    monkeypatch.setattr(model_sync, "ensure_model_available",
                        lambda model_id: None, raising=False)
    return model_sync


def test_layer_ready_ack_echoes_generation(_stub_model_sync):
    """普通层（非 relay）的 `ready` ACK 同样必须带 `generation`。"""
    peer, sent = _peer()
    peer._host = _Host()

    peer._handle_layer_config({
        "start_layer": 0, "end_layer": 8,
        "node_id": "client1", "config_id": "cfg-2", "generation": 11,
        "assignment_id": "asg-ready",
        "model_id": "qwen2.5-0.5b-instruct", "model_sha256": "abc",
        "model_type": "qwen2", "total_layers": 24,
    })

    assert len(sent) == 1, sent
    ack = sent[0]
    assert ack["status"] == "ready"
    assert ack["layer_range"] == [0, 8], ack
    assert ack["generation"] == 11, ack
    assert ack["assignment_id"] == "asg-ready"
    assert _master_generation_gate(ack, {"generation": 11}) is True, ack
    assert _master_generation_gate(ack, {"generation": 12}) is False


def test_layer_config_parse_error_ack_echoes_generation():
    peer, sent = _peer()

    peer._handle_layer_config({
        "start_layer": "not-an-int", "end_layer": 8,
        "node_id": "client1", "config_id": "cfg-parse", "generation": 13,
        "assignment_id": "asg-parse",
    })

    assert sent[0]["status"] == "error"
    assert sent[0]["generation"] == 13
    assert sent[0]["assignment_id"] == "asg-parse"


def test_missing_assignment_error_ack_echoes_top_level_generation():
    peer, sent = _peer()

    peer._handle_layer_config({
        "node_id": "client1", "config_id": "cfg-missing", "generation": 14,
        "assignment_id": "asg-missing",
        "assignments": {},
    })

    assert sent[0]["status"] == "error"
    assert sent[0]["generation"] == 14
    assert sent[0]["assignment_id"] == "asg-missing"


def test_layer_load_error_ack_echoes_generation(_stub_model_sync):
    peer, sent = _peer()
    peer._host = _Host()

    def fail_load(**_kwargs):
        raise RuntimeError("simulated load failure")

    # ★ 桩要打在真实调用点上：peer 解析后调的是 `load_layer_range`（peer.py:531），
    #   不是 `load_model` —— 打在旧名字上会让本测试静默变成"加载成功"。
    peer._host._host.load_layer_range = fail_load
    peer._handle_layer_config({
        "start_layer": 0, "end_layer": 8,
        "node_id": "client1", "config_id": "cfg-load", "generation": 16,
        "assignment_id": "asg-load-error",
        "model_id": "qwen2.5-0.5b-instruct", "model_sha256": "abc",
        "model_type": "qwen2", "total_layers": 24,
    })

    assert sent[0]["status"] == "error"
    assert sent[0]["generation"] == 16
    assert sent[0]["assignment_id"] == "asg-load-error"


def test_relay_unknown_phase_is_rejected_with_generation():
    peer, sent = _peer()

    peer._handle_layer_config({
        "start_layer": 8, "end_layer": 24,
        "node_id": "client1", "config_id": "cfg-phase", "generation": 15,
        "assignment_id": "asg-phase-error",
        "engine": "relay_middle", "phase": "commit-ish",
    })

    assert sent[0]["status"] == "error"
    assert sent[0]["generation"] == 15
    assert sent[0]["assignment_id"] == "asg-phase-error"
    assert peer._active_layer_config is None


def test_gate_is_inert_when_expected_has_no_generation():
    """旧路径（期望不带 generation）不得因新增字段而改变行为。"""
    assert _master_generation_gate({"status": "ready"}, {"config_id": "x"}) is True
