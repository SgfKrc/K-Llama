"""★ A1 / X 档（Y 档第二条缺口 2）：worker 的 release ACK 必须满足主节点的 `released` 判据。

实测（2026-09-28 跨机 relay 首次跑）：新注册的 worker 收到 release，回的 ACK **只有**
`status="released"`；而主节点 `scheduler_pipeline.py:1930-1934` 的判据是
`status == "released"` **且** `release is True` **且** `generation` 相等
⇒ 恒判「从节点分层释放 ACK 未通过」⇒ 每 5 秒重发（实测刷到 99 次 attempt）
⇒ `pipeline_distributed_workers_unavailable` ⇒ `/api/chat` 503。
"""

from __future__ import annotations

import sys
import threading
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for _candidate in (str(ROOT), str(ROOT / "src")):
    if _candidate not in sys.path:
        sys.path.insert(0, _candidate)

from inference_service.peer import PeerClient  # noqa: E402


def _peer(node_id: str = "client1"):
    """最小 PeerClient：只补 `_handle_layer_config_locked` 用到的属性，并记录发出的 ACK。"""
    peer = object.__new__(PeerClient)
    peer._node_id = node_id
    peer._layer_execution_lock = threading.RLock()
    peer._layer_config_lock = threading.RLock()
    peer._active_layer_config = {"stale": True}
    peer._local_pipeline_steps = {"t1": 0}
    peer._latest_layer_config_generation = 0
    peer._latest_layer_config_id = ""
    peer._latest_layer_assignment_id = ""
    peer._layer_assignment_identity_required = False
    sent: list = []
    peer._send_layer_config_ack = lambda payload: (sent.append(payload), True)[1]
    return peer, sent


def _master_released_predicate(ack: dict, expected: dict) -> bool:
    """复刻 `scheduler_pipeline.py:1930-1934` 的 `released` 判据。

    该判据内联在方法体里（无法直接 import），所以这里照抄一份并**在用例中断言它真的参与判定**
    （`test_release_generation_must_echo_the_request` 用不同 generation 反向验证）。
    """
    try:
        ack_generation = int(ack.get("generation", 0) or 0)
    except (TypeError, ValueError):
        ack_generation = -1
    return (
        ack.get("status") == "released"
        and ack.get("release") is True
        and ack_generation == int(expected.get("generation", 0) or 0)
    )


def test_release_ack_satisfies_the_master_predicate():
    """★ 缺口 2 的判据：release ACK 必须让主节点的 `released` 判据成立。"""
    peer, sent = _peer()

    peer._handle_layer_config({
        "release": True, "node_id": "client1", "config_id": "cfg-1", "generation": 7,
    })

    assert len(sent) == 1, sent
    ack = sent[0]
    assert ack["status"] == "released"
    assert ack["config_id"] == "cfg-1"
    assert _master_released_predicate(ack, {"generation": 7}) is True, ack

    # release 成功后本地层配置必须被清掉（否则残留的旧配置会继续被当有效）。
    assert peer._active_layer_config is None
    assert peer._local_pipeline_steps == {}


def test_release_generation_must_echo_the_request():
    """`generation` 必须**回显请求值**（主节点按它防"陈旧 release"）。"""
    peer, sent = _peer()

    peer._handle_layer_config({
        "release": True, "node_id": "client1", "config_id": "cfg-9", "generation": 42,
    })

    assert sent[0]["generation"] == 42
    assert _master_released_predicate(sent[0], {"generation": 42}) is True
    # 主节点若期望别的 generation 就该判失败 ⇒ 证明这个字段**真的**参与判据，不是摆设。
    assert _master_released_predicate(sent[0], {"generation": 41}) is False


def test_matching_assignment_release_echoes_identity_and_clears_state():
    peer, sent = _peer()
    peer._handle_layer_config({
        "start_layer": 8,
        "end_layer": 24,
        "node_id": "client1",
        "config_id": "cfg-current",
        "generation": 12,
        "assignment_id": "asg-current",
        "engine": "relay_middle",
        "model_sha256": "deadbeef",
        "model_type": "qwen2",
        "phase": "commit",
    })
    peer._local_pipeline_steps = {"task-current": 0}

    peer._handle_layer_config({
        "release": True,
        "node_id": "client1",
        "config_id": "cfg-current",
        "generation": 12,
        "assignment_id": "asg-current",
    })

    assert [ack["status"] for ack in sent] == ["ready", "released"]
    assert sent[-1]["assignment_id"] == "asg-current"
    assert peer._active_layer_config is None
    assert peer._local_pipeline_steps == {}


def test_other_assignment_release_cannot_clear_current_config():
    peer, sent = _peer()
    current = {
        "start_layer": 8,
        "end_layer": 24,
        "node_id": "client1",
        "config_id": "cfg-current",
        "generation": 13,
        "assignment_id": "asg-current",
        "engine": "relay_middle",
        "model_sha256": "deadbeef",
        "model_type": "qwen2",
        "phase": "commit",
    }
    peer._handle_layer_config(current)
    peer._local_pipeline_steps = {"task-current": 0}

    peer._handle_layer_config({
        "release": True,
        "node_id": "client1",
        "config_id": "cfg-current",
        "generation": 13,
        "assignment_id": "asg-old",
    })

    assert len(sent) == 1
    assert peer._active_layer_config["assignment_id"] == "asg-current"
    assert peer._local_pipeline_steps == {"task-current": 0}


def test_identity_less_release_is_rejected_after_assignment_fencing():
    peer, sent = _peer()
    current = {
        "start_layer": 8,
        "end_layer": 24,
        "node_id": "client1",
        "config_id": "cfg-current",
        "generation": 14,
        "assignment_id": "asg-current",
        "engine": "relay_middle",
        "model_sha256": "deadbeef",
        "model_type": "qwen2",
        "phase": "commit",
    }
    peer._handle_layer_config(current)
    peer._local_pipeline_steps = {"task-current": 0}

    peer._handle_layer_config({
        "release": True,
        "node_id": "client1",
        "config_id": "cfg-current",
        "generation": 14,
    })

    assert len(sent) == 1
    assert peer._active_layer_config["assignment_id"] == "asg-current"
    assert peer._local_pipeline_steps == {"task-current": 0}


def test_release_for_another_node_is_ignored():
    """目标不匹配的 release 必须被忽略，且**不得**回 ACK（既有保护，勿退化）。"""
    peer, sent = _peer(node_id="client1")

    peer._handle_layer_config({
        "release": True, "node_id": "client2", "config_id": "cfg-1", "generation": 1,
    })

    assert sent == []
    assert peer._active_layer_config == {"stale": True}


def test_foreign_release_does_not_advance_generation_watermark():
    """A release addressed to another peer must not fence local configs."""
    peer, sent = _peer(node_id="client1")

    peer._handle_layer_config({
        "release": True,
        "node_id": "client2",
        "config_id": "foreign",
        "generation": 99,
    })
    peer._handle_layer_config({
        "release": True,
        "node_id": "client1",
        "config_id": "local",
        "generation": 1,
    })

    assert peer._latest_layer_config_generation == 1
    assert sent[-1]["config_id"] == "local"


def test_stale_release_cannot_clear_newer_relay_config():
    """A delayed restart release must not erase the current relay assignment."""
    peer, sent = _peer(node_id="client1")
    peer._latest_layer_config_generation = 20
    peer._latest_layer_config_id = "cfg-current"
    peer._active_layer_config = {
        "node_id": "client1",
        "config_id": "cfg-current",
        "generation": 20,
        "engine": "relay_middle",
    }
    peer._local_pipeline_steps = {"task-current": 0}

    peer._handle_layer_config({
        "release": True,
        "node_id": "client1",
        "config_id": "cfg-old-release",
        "generation": 19,
    })

    assert sent == []
    assert peer._active_layer_config["config_id"] == "cfg-current"
    assert peer._local_pipeline_steps == {"task-current": 0}


def test_unversioned_release_cannot_clear_newer_relay_config():
    """Legacy delayed releases are fenced after a versioned assignment."""
    peer, sent = _peer(node_id="client1")
    peer._latest_layer_config_generation = 20
    peer._latest_layer_config_id = "cfg-current"
    peer._active_layer_config = {
        "node_id": "client1",
        "config_id": "cfg-current",
        "generation": 20,
        "engine": "relay_middle",
    }
    peer._local_pipeline_steps = {"task-current": 0}

    peer._handle_layer_config({
        "release": True,
        "node_id": "client1",
        "config_id": "cfg-legacy-release",
    })

    assert sent == []
    assert peer._active_layer_config["config_id"] == "cfg-current"
    assert peer._local_pipeline_steps == {"task-current": 0}


def test_same_generation_release_without_config_id_is_fenced():
    """A versioned release without its identity must not clear active state."""
    peer, sent = _peer(node_id="client1")
    peer._latest_layer_config_generation = 20
    peer._latest_layer_config_id = "cfg-current"
    peer._active_layer_config = {
        "node_id": "client1",
        "config_id": "cfg-current",
        "generation": 20,
    }
    peer._local_pipeline_steps = {"task-current": 0}

    peer._handle_layer_config({
        "release": True,
        "node_id": "client1",
        "generation": 20,
    })

    assert sent == []
    assert peer._active_layer_config["config_id"] == "cfg-current"
    assert peer._local_pipeline_steps == {"task-current": 0}


def test_same_generation_release_with_other_config_id_is_fenced():
    """A same-generation release for another config is not a duplicate."""
    peer, sent = _peer(node_id="client1")
    peer._latest_layer_config_generation = 20
    peer._latest_layer_config_id = "cfg-current"
    peer._active_layer_config = {
        "node_id": "client1",
        "config_id": "cfg-current",
        "generation": 20,
    }
    peer._local_pipeline_steps = {"task-current": 0}

    peer._handle_layer_config({
        "release": True,
        "node_id": "client1",
        "config_id": "cfg-old",
        "generation": 20,
    })

    assert sent == []
    assert peer._active_layer_config["config_id"] == "cfg-current"
    assert peer._local_pipeline_steps == {"task-current": 0}
