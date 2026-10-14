"""#93：落后副本的**主动追赶**（跨机重建的拉取方向）。

链路对偶关系：
- 栅栏（推）：`replicate_journal_to_quorum` ⇒ 多数耐久才回 OK；
- 追赶（拉）：`request_checkpoint` + `recover_journal_prefix` ⇒ 落后的一方向持有完整
  前缀的对端讨回一个**已自洽**的 checkpoint 并应用，之后才允许接管。

两条硬约束由测试钉住：
1. 不自洽的副本**导出不了** checkpoint（不许把没人担保的前缀当已提交发出去）；
2. 一个对端都没成功 ⇒ `recovered=False`（调用方必须继续拒绝接管）。
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

from cluster_auto_role_runtime import AutoRoleRuntimeError, JournalPrefixAuthority  # noqa: E402
from cluster_control_contract import CommitWatermark  # noqa: E402
from journal_replication import (  # noqa: E402
    DurableJournalReplica,
    JournalReplicationError,
    build_checkpoint,
)
from journal_replication_service import (  # noqa: E402
    JournalReplicationSettings,
    recover_journal_prefix,
)
from journal_replication_transport import (  # noqa: E402
    JournalReplicationTcpServer,
    request_checkpoint,
)
from task_journal import JournalEvent, SQLiteTaskJournal  # noqa: E402

SECRET = b"r" * 40
SRC = "node-leader"
STREAM = "task-graph-journal"
WF = "wf-recover"


def _snapshot(workflow_id: str, sequence: int) -> dict:
    return {
        "workflow_id": workflow_id,
        "last_sequence": sequence,
        "state": "running",
        "created_at": 100.0,
        "updated_at": 100.0 + sequence,
        "stages": [{"stage_id": "s1", "state": "running", "retry_safe": True, "pure": True, "attempts": []}],
    }


def _append(journal: SQLiteTaskJournal, sequence: int) -> None:
    journal.append_event(
        JournalEvent(
            event_id=f"evt-{WF}-{sequence}",
            workflow_id=WF,
            sequence=sequence,
            entity_type="workflow",
            entity_id=WF,
            event_type="workflow_state_changed",
            occurred_at=100.0 + sequence,
            payload={"sequence": sequence},
        ),
        _snapshot(WF, sequence),
    )


def _replica(tmp_path: Path, name: str) -> DurableJournalReplica:
    return DurableJournalReplica(
        str(tmp_path / f"{name}.sqlite3"), source_node_id=SRC, stream_id=STREAM,
    )


def _apply(source: SQLiteTaskJournal, replica: DurableJournalReplica, *, created_at: float = 1.0):
    return replica.apply_checkpoint(build_checkpoint(
        source, source_node_id=SRC, stream_id=STREAM, workflow_id=WF, created_at=created_at,
    ))


def _settings(peers, **overrides) -> JournalReplicationSettings:
    base = dict(
        enabled=True, source_node_id=SRC, stream_id=STREAM, replica_path="x",
        timeout_s=2.0, secret=SECRET, quorum_peers=tuple(peers),
    )
    base.update(overrides)
    return JournalReplicationSettings(**base)


def test_export_checkpoint_refuses_an_inconsistent_replica(tmp_path):
    source = SQLiteTaskJournal(str(tmp_path / "source.sqlite3"))
    for sequence in (1, 2, 3):
        _append(source, sequence)
    replica = _replica(tmp_path, "r-inconsistent")
    _apply(source, replica)
    # 截断尾部后快照与事件链不再自洽 ⇒ 不许导出
    replica.truncate_to(WF, last_sequence=1)
    with pytest.raises(JournalReplicationError):
        replica.export_checkpoint(WF)
    replica.close()
    source.close()


def test_request_checkpoint_round_trip(tmp_path):
    source = SQLiteTaskJournal(str(tmp_path / "source.sqlite3"))
    for sequence in (1, 2, 3):
        _append(source, sequence)
    donor = _replica(tmp_path, "donor")
    _apply(source, donor)
    server = JournalReplicationTcpServer(donor, secret=SECRET)
    _host, port = server.start()
    try:
        pulled = request_checkpoint(
            "127.0.0.1", port,
            source_node_id=SRC, stream_id=STREAM, workflow_id=WF,
            secret=SECRET, timeout=2.0,
        )
    finally:
        server.close()
    assert pulled.last_sequence == 3
    # 拉到的东西必须能直接喂给另一个副本并让它自洽
    fresh = _replica(tmp_path, "fresh")
    fresh.apply_checkpoint(pulled)
    assert fresh.recovery_status(WF)["verified"] is True
    fresh.close()
    donor.close()
    source.close()


def test_behind_replica_catches_up_from_a_peer(tmp_path):
    source = SQLiteTaskJournal(str(tmp_path / "source.sqlite3"))
    _append(source, 1)
    behind = _replica(tmp_path, "behind")
    _apply(source, behind)
    assert behind.durable_prefix(WF)["last_sequence"] == 1

    for sequence in (2, 3):
        _append(source, sequence)
    donor = _replica(tmp_path, "donor")
    _apply(source, donor)
    server = JournalReplicationTcpServer(donor, secret=SECRET)
    _host, port = server.start()
    try:
        result = recover_journal_prefix(
            behind, workflow_id=WF, settings=_settings([("127.0.0.1", port)]),
        )
    finally:
        server.close()
    assert result["recovered"] is True
    assert result["failures"] == {}
    assert behind.durable_prefix(WF)["last_sequence"] == 3
    assert behind.recovery_status(WF)["verified"] is True
    behind.close()
    donor.close()
    source.close()


def test_recover_reports_failure_when_no_peer_can_serve(tmp_path):
    source = SQLiteTaskJournal(str(tmp_path / "source.sqlite3"))
    for sequence in (1, 2):
        _append(source, sequence)
    lonely = _replica(tmp_path, "lonely")
    try:
        result = recover_journal_prefix(
            lonely, workflow_id=WF, settings=_settings([("127.0.0.1", 1), ("127.0.0.1", 2)]),
        )
    finally:
        lonely.close()
        source.close()
    assert result["recovered"] is False
    assert result["status"] == "unrecovered"
    assert len(result["failures"]) == 2


def test_authority_recovers_before_deciding(tmp_path):
    """落后 + 有 recovery 钩子 ⇒ 钩子被调用、追赶成功后**允许**接管。"""
    source = SQLiteTaskJournal(str(tmp_path / "source.sqlite3"))
    _append(source, 1)
    behind = _replica(tmp_path, "behind2")
    _apply(source, behind)

    for sequence in (2, 3):
        _append(source, sequence)
    donor = _replica(tmp_path, "donor2")
    _apply(source, donor)
    server = JournalReplicationTcpServer(donor, secret=SECRET)
    _host, port = server.start()

    calls: list[str] = []
    settings = _settings([("127.0.0.1", port)])

    def _recovery(workflow_id: str) -> None:
        calls.append(workflow_id)
        recover_journal_prefix(behind, workflow_id=workflow_id, settings=settings)

    authority = JournalPrefixAuthority(behind, workflows=(WF,), recovery=_recovery)
    committed = authority.held(WF)
    # 本地**自洽**但只到 1：这不等于"没有前缀"，而是"前缀落后于承诺"。
    assert committed is not None and committed.last_sequence == 1
    # donor 自洽 ⇒ 用它的水位当承诺
    donor_authority = JournalPrefixAuthority(donor, workflows=(WF,))
    target = donor_authority.held(WF)
    assert target is not None
    try:
        authority.enforce(type("C", (), {"commit_watermark": (target,)})())
        assert calls == [WF]
        # 追赶之后本地前缀已对齐 —— 这才是"允许接管"的依据。
        assert behind.recovery_status(WF)["verified"] is True
        assert behind.durable_prefix(WF)["last_sequence"] == 3
    finally:
        server.close()
        behind.close()
        donor.close()
        source.close()


def test_authority_without_recovery_still_refuses(tmp_path):
    """没有 recovery 钩子时，落后就是落后 ⇒ 必须拒绝（不许凑合上线）。"""
    source = SQLiteTaskJournal(str(tmp_path / "source.sqlite3"))
    _append(source, 1)
    behind = _replica(tmp_path, "behind3")
    _apply(source, behind)
    authority = JournalPrefixAuthority(behind, workflows=(WF,))
    target = CommitWatermark(workflow_id=WF, last_sequence=3, checkpoint_digest="f" * 64)
    with pytest.raises(AutoRoleRuntimeError) as excinfo:
        authority.enforce(type("C", (), {"commit_watermark": (target,)})())
    assert excinfo.value.code == "committed_prefix_behind"
    behind.close()
    source.close()
