"""R-R11 的回归测试：journal 复制的最小可用接线。

**该红必须红**的用例（照 `tests/test_ci_relay_gates.py` 的惯例）：
* `test_replication_is_disabled_by_default` —— 谁把默认改成开启，它必须红；
* `test_wrong_secret_is_rejected_by_receiver` —— 谁把认证绕过，它必须红。
"""

from __future__ import annotations

import asyncio
import os
import sys
import types
from dataclasses import replace
from pathlib import Path

import pytest
from fastapi import HTTPException

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

from journal_replication import JournalReplicationError  # noqa: E402
from journal_replication_service import (  # noqa: E402
    JournalReplicaReceiver,
    JournalReplicationDisabled,
    JournalReplicationSettings,
    replicate_journal_once,
)
from journal_replication_transport import JournalReplicationTransportError  # noqa: E402
from recovery_journal import JournalEvent, SQLiteTaskJournal  # noqa: E402

SECRET = b"r" * 40


def _snapshot(workflow_id: str, sequence: int, *, state: str = "running") -> dict:
    return {
        "workflow_id": workflow_id,
        "last_sequence": sequence,
        "state": state,
        "created_at": 100.0,
        "updated_at": 100.0 + sequence,
        "stages": [
            {
                "stage_id": "stage-1",
                "state": "running",
                "retry_safe": True,
                "pure": True,
                "attempts": [{"state": "running"}],
            }
        ],
    }


def _append(journal: SQLiteTaskJournal, workflow_id: str, sequence: int) -> None:
    journal.append_event(
        JournalEvent(
            event_id=f"evt-{workflow_id}-{sequence}",
            workflow_id=workflow_id,
            sequence=sequence,
            entity_type="workflow",
            entity_id=workflow_id,
            event_type="workflow_state_changed",
            occurred_at=100.0 + sequence,
            payload={"state": "running", "sequence": sequence},
        ),
        _snapshot(workflow_id, sequence),
    )


def _settings(tmp_path: Path, **overrides) -> JournalReplicationSettings:
    base = JournalReplicationSettings(
        enabled=True,
        source_node_id="node-a",
        stream_id="journal-1",
        replica_path=str(tmp_path / "replica.sqlite3"),
        secret=SECRET,
        timeout_s=5.0,
    )
    return replace(base, **overrides) if overrides else base


def _clear_env(monkeypatch) -> None:
    for name in list(os.environ):
        if name.startswith("QLH_JOURNAL_REPLICATION"):
            monkeypatch.delenv(name, raising=False)


# --------------------------------------------------------------------------- 默认关

def test_replication_is_disabled_by_default(tmp_path, monkeypatch):
    """★ 该红必须红：默认状态必须是**关**，且关着时推送直接 fail-closed（不发起任何连接）。"""

    _clear_env(monkeypatch)
    settings = JournalReplicationSettings.from_env()

    assert settings.enabled is False
    assert settings.public()["enabled"] is False

    journal = SQLiteTaskJournal(str(tmp_path / "source.sqlite3"))
    _append(journal, "wf-disabled", 1)
    try:
        with pytest.raises(JournalReplicationDisabled):
            replicate_journal_once(journal, workflow_id="wf-disabled", settings=settings)
    finally:
        journal.close()


def test_enabled_settings_require_secret_and_peer(tmp_path):
    """开启之后缺任一必需项都必须 fail-closed —— 绝不降级成"连 loopback:0 试试"。"""

    journal = SQLiteTaskJournal(str(tmp_path / "source.sqlite3"))
    _append(journal, "wf-guard", 1)
    try:
        with pytest.raises(JournalReplicationError):
            replicate_journal_once(
                journal, workflow_id="wf-guard", settings=_settings(tmp_path, secret=b"short")
            )
        # secret 够了、但没有对端 ⇒ 同样必须拒。
        with pytest.raises(JournalReplicationError):
            replicate_journal_once(
                journal, workflow_id="wf-guard", settings=_settings(tmp_path, secret=SECRET)
            )
    finally:
        journal.close()


def test_from_env_reads_the_documented_names(monkeypatch):
    _clear_env(monkeypatch)
    monkeypatch.setenv("QLH_JOURNAL_REPLICATION_ENABLED", "1")
    monkeypatch.setenv("QLH_JOURNAL_REPLICATION_SECRET", "s" * 40)
    monkeypatch.setenv("QLH_JOURNAL_REPLICATION_PEER_HOST", "10.0.0.9")
    monkeypatch.setenv("QLH_JOURNAL_REPLICATION_PEER_PORT", "50999")
    monkeypatch.setenv("QLH_NODE_ID", "node-env")

    settings = JournalReplicationSettings.from_env()

    assert settings.enabled is True
    assert settings.source_node_id == "node-env"      # 回落到 QLH_NODE_ID
    assert settings.peer_host == "10.0.0.9"
    assert settings.peer_port == 50999
    assert settings.public()["secret_present"] is True
    assert "secret" not in settings.public()          # 绝不回显密钥本身


# --------------------------------------------------------------------- 真跑（loopback）

def test_replicates_to_a_live_loopback_receiver(tmp_path):
    """真接线：起一个真接收端（port=0 由系统分配），推一个 checkpoint，对端可恢复。"""

    source = SQLiteTaskJournal(str(tmp_path / "source.sqlite3"))
    _append(source, "wf-e2e", 1)
    settings = _settings(tmp_path)
    receiver = JournalReplicaReceiver(settings)
    host, port = receiver.start()
    try:
        pushed = replicate_journal_once(
            source,
            workflow_id="wf-e2e",
            settings=replace(settings, peer_host=host, peer_port=port),
        )
        assert pushed["status"] == "replicated"
        assert pushed["durable_sequence"] == 1
        assert pushed["event_count"] == 1

        status = receiver.replica.recovery_status("wf-e2e")
        assert status["durable_sequence"] == 1
        assert status["verified"] is True
    finally:
        receiver.close()
        source.close()


def test_wrong_secret_is_rejected_by_receiver(tmp_path):
    """★ 该红必须红：密钥不对必须被**对端拒绝** —— 证明 HMAC 认证真的在跑。"""

    source = SQLiteTaskJournal(str(tmp_path / "source.sqlite3"))
    _append(source, "wf-sec", 1)
    settings = _settings(tmp_path)
    receiver = JournalReplicaReceiver(settings)
    host, port = receiver.start()
    try:
        with pytest.raises(JournalReplicationTransportError):
            replicate_journal_once(
                source,
                workflow_id="wf-sec",
                settings=replace(
                    settings, peer_host=host, peer_port=port, secret=b"x" * 40
                ),
            )
        # 被拒之后对端**不得**留下任何可用状态（fail-closed，不暴露半份数据）。
        status = receiver.replica.recovery_status("wf-sec")
        assert status["durable"] is False
        assert status["reason"] == "workflow_not_replicated"
    finally:
        receiver.close()
        source.close()


def test_receiver_refuses_to_start_when_disabled():
    disabled = JournalReplicationSettings(
        enabled=False, source_node_id="node-a", stream_id="journal-1"
    )
    with pytest.raises(JournalReplicationDisabled):
        JournalReplicaReceiver(disabled)


# ----------------------------------------------------------------------- 端点 fail-closed
