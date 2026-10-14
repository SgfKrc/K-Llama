"""#93「回包前栅栏」：checkpoint 推送必须**多数耐久**才回 OK。

用**真实** TCP replica server（不是 mock）：
- 3 个目标全可达 ⇒ durable_acks=3 ≥ 多数(2) ⇒ accepted；
- 只有 1 个可达 ⇒ durable_acks=1 < 2 ⇒ 拒绝回包，并逐个记下失败目标；
- 阈值默认取多数、可显式指定；阈值 > 目标数 ⇒ 配置即拒绝；
- `public()` 绝不回显 secret。
"""

from __future__ import annotations

import asyncio
import sys
import types
from pathlib import Path

import pytest
from fastapi import HTTPException

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

from journal_replication import DurableJournalReplica, JournalReplicationError  # noqa: E402
from journal_replication_service import (  # noqa: E402
    JournalReplicationSettings,
    _parse_quorum_peers,
    replicate_journal_to_quorum,
)
from journal_replication_transport import JournalReplicationTcpServer  # noqa: E402
from task_journal import JournalEvent, SQLiteTaskJournal  # noqa: E402

SECRET = b"q" * 40
SRC = "node-leader"
STREAM = "task-graph-journal"


def _snapshot(workflow_id: str, sequence: int) -> dict:
    return {
        "workflow_id": workflow_id,
        "last_sequence": sequence,
        "state": "running",
        "created_at": 100.0,
        "updated_at": 100.0 + sequence,
        "stages": [{"stage_id": "s1", "state": "running", "retry_safe": True, "pure": True, "attempts": []}],
    }


def _journal(tmp_path: Path, workflow_id: str, sequences: int = 2) -> SQLiteTaskJournal:
    journal = SQLiteTaskJournal(str(tmp_path / "source.sqlite3"))
    for sequence in range(1, sequences + 1):
        journal.append_event(
            JournalEvent(
                event_id=f"evt-{workflow_id}-{sequence}",
                workflow_id=workflow_id,
                sequence=sequence,
                entity_type="workflow",
                entity_id=workflow_id,
                event_type="workflow_state_changed",
                occurred_at=100.0 + sequence,
                payload={"sequence": sequence},
            ),
            _snapshot(workflow_id, sequence),
        )
    return journal


def _replica_servers(tmp_path: Path, count: int):
    servers = []
    for index in range(count):
        replica = DurableJournalReplica(
            str(tmp_path / f"replica-{index}.sqlite3"), source_node_id=SRC, stream_id=STREAM,
        )
        server = JournalReplicationTcpServer(replica, secret=SECRET)
        servers.append((server, server.start()))
    return servers


def _settings(peers, **overrides) -> JournalReplicationSettings:
    base = dict(
        enabled=True, source_node_id=SRC, stream_id=STREAM, replica_path="x",
        timeout_s=1.0, secret=SECRET, quorum_peers=tuple(peers),
    )
    base.update(overrides)
    return JournalReplicationSettings(**base)


def test_quorum_barrier_accepts_when_majority_is_durable(tmp_path):
    journal = _journal(tmp_path, "wf-q1")
    servers = _replica_servers(tmp_path, 3)
    try:
        peers = [("127.0.0.1", address[1]) for _server, address in servers]
        result = replicate_journal_to_quorum(
            journal, workflow_id="wf-q1", settings=_settings(peers),
        )
    finally:
        for server, _address in servers:
            server.close()
        journal.close()
    assert result["accepted"] is True
    assert result["status"] == "replicated"
    assert result["durable_acks"] == 3
    assert result["required_quorum"] == 2
    assert result["failures"] == {}
    # 三个副本都真的耐久持有到同一点、同一个摘要
    assert len(set(result["durable_peers"])) == 3


def test_quorum_barrier_refuses_to_answer_without_majority(tmp_path):
    """栅栏的核心：达不到多数 ⇒ accepted=False（调用方据此回错误，绝不回 OK）。"""
    journal = _journal(tmp_path, "wf-q2")
    servers = _replica_servers(tmp_path, 1)
    try:
        reachable_port = servers[0][1][1]
        peers = [("127.0.0.1", reachable_port), ("127.0.0.1", 1), ("127.0.0.1", 2)]
        result = replicate_journal_to_quorum(
            journal, workflow_id="wf-q2", settings=_settings(peers),
        )
    finally:
        servers[0][0].close()
        journal.close()
    assert result["accepted"] is False
    assert result["status"] == "quorum_not_reached"
    assert result["durable_acks"] == 1
    assert result["required_quorum"] == 2
    assert len(result["failures"]) == 2


def test_required_quorum_defaults_to_majority():
    assert _settings([]).required_quorum == 0  # 未配置 ⇒ 不做多数判定（单 peer 语义不变）
    assert _settings([("a", 1)]).required_quorum == 1
    assert _settings([("a", 1), ("b", 2)]).required_quorum == 2
    assert _settings([("a", 1), ("b", 2), ("c", 3)]).required_quorum == 2
    assert _settings([("a", 1), ("b", 2), ("c", 3), ("d", 4), ("e", 5)]).required_quorum == 3


def test_explicit_quorum_size_is_honoured():
    assert _settings([("a", 1), ("b", 2), ("c", 3)], quorum_size=3).required_quorum == 3
    assert _settings([("a", 1), ("b", 2), ("c", 3)], quorum_size=1).required_quorum == 1


def test_quorum_size_above_peer_count_is_rejected():
    with pytest.raises(JournalReplicationError):
        _settings([("a", 1)], quorum_size=2).require_quorum_peers()


def test_parse_quorum_peers_drops_invalid_and_dedupes():
    assert _parse_quorum_peers("h1:1, h2:2 ,bad,h3:x, h1:1,") == (("h1", 1), ("h2", 2))
    assert _parse_quorum_peers("") == ()
    assert _parse_quorum_peers("h:0") == ()  # 端口 0 不合法（不猜端口）


def test_public_never_leaks_secret_and_reports_quorum():
    payload = _settings([("h1", 1111)], quorum_size=1).public()
    assert payload["secret_present"] is True
    assert "secret" not in payload
    assert SECRET.decode() not in repr(payload)
    assert payload["quorum"] == {"peers": ["h1:1111"], "size": 1, "required": 1}


def test_from_env_reads_quorum_peers(monkeypatch):
    monkeypatch.setenv("QLH_JOURNAL_REPLICATION_ENABLED", "1")
    monkeypatch.setenv("QLH_JOURNAL_REPLICATION_SOURCE_NODE_ID", SRC)
    monkeypatch.setenv("QLH_JOURNAL_REPLICATION_REPLICA_PATH", "x")
    monkeypatch.setenv("QLH_JOURNAL_REPLICATION_SECRET", SECRET.decode())
    monkeypatch.setenv(
        "QLH_JOURNAL_REPLICATION_QUORUM_PEERS", "10.0.0.1:9001,10.0.0.2:9002"
    )
    settings = JournalReplicationSettings.from_env()
    assert settings.quorum_peers == (("10.0.0.1", 9001), ("10.0.0.2", 9002))
    assert settings.required_quorum == 2
    settings.validate()


# ------------------------------------------------------------------ 端点：栅栏的 HTTP 语义

def _fake_api_module(monkeypatch, *, role: str, journal):
    from api import routes_tasks

    scheduler = types.SimpleNamespace(_effective_role=lambda: role)
    coordinator = types.SimpleNamespace(journal=journal)
    fake = types.SimpleNamespace(
        scheduler=scheduler,
        task_graph_coordinator=coordinator,
        HTTPException=HTTPException,
    )
    monkeypatch.setattr(routes_tasks, "_api_module", fake)
    return routes_tasks


def _set_env(monkeypatch, peers: str) -> None:
    monkeypatch.setenv("QLH_JOURNAL_REPLICATION_ENABLED", "1")
    monkeypatch.setenv("QLH_JOURNAL_REPLICATION_SOURCE_NODE_ID", SRC)
    monkeypatch.setenv("QLH_JOURNAL_REPLICATION_STREAM_ID", STREAM)
    monkeypatch.setenv("QLH_JOURNAL_REPLICATION_REPLICA_PATH", "x")
    monkeypatch.setenv("QLH_JOURNAL_REPLICATION_SECRET", SECRET.decode())
    monkeypatch.setenv("QLH_JOURNAL_REPLICATION_QUORUM_PEERS", peers)
    monkeypatch.setenv("QLH_JOURNAL_REPLICATION_TIMEOUT_S", "1")


def test_api_endpoint_answers_only_when_quorum_is_durable(monkeypatch, tmp_path):
    journal = _journal(tmp_path, "wf-api1")
    servers = _replica_servers(tmp_path, 3)
    try:
        peers = ",".join(f"127.0.0.1:{address[1]}" for _server, address in servers)
        _set_env(monkeypatch, peers)
        routes = _fake_api_module(monkeypatch, role="master", journal=journal)
        payload = asyncio.run(routes.replicate_journal(workflow_id="wf-api1"))
    finally:
        for server, _address in servers:
            server.close()
        journal.close()
    assert payload["accepted"] is True
    assert payload["durable_acks"] == 3
    assert payload["required_quorum"] == 2


def test_api_endpoint_returns_503_when_quorum_is_not_durable(monkeypatch, tmp_path):
    """栅栏必须体现在 HTTP 语义上：多数未达成 ⇒ 503，而不是 200。"""
    journal = _journal(tmp_path, "wf-api2")
    servers = _replica_servers(tmp_path, 1)
    try:
        peers = f"127.0.0.1:{servers[0][1][1]},127.0.0.1:1,127.0.0.1:2"
        _set_env(monkeypatch, peers)
        routes = _fake_api_module(monkeypatch, role="master", journal=journal)
        with pytest.raises(HTTPException) as excinfo:
            asyncio.run(routes.replicate_journal(workflow_id="wf-api2"))
    finally:
        servers[0][0].close()
        journal.close()
    assert excinfo.value.status_code == 503
    assert "1/2" in str(excinfo.value.detail)
