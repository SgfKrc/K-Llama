"""#93：把「提交证明」接到自动选主的两端。

- leader 侧：`commit_watermark_provider` 提议的已提交前缀必须真的进入证书（因此也进入
  被签名 payload），否则 voter 的耐久持有校验毫无意义；
- 接管侧：`committed_prefix_guard` 决定能不能接管 —— 不通过必须 fail-closed 停在
  read_only，绝不允许带着「未提交尾部」或「落后前缀」上位。
"""

from __future__ import annotations

import base64
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

from cluster_auto_role import AutoRoleController  # noqa: E402
from cluster_auto_role_runtime import AutoRoleRuntimeError, JournalPrefixAuthority  # noqa: E402
from cluster_control_contract import (  # noqa: E402
    CommitWatermark,
    ControlPlaneAuthority,
    VoterIdentity,
    VoterSet,
)
from cluster_quorum import (  # noqa: E402
    QuorumCollector,
    QuorumVoter,
    SQLiteVoterLedger,
)
from journal_replication import DurableJournalReplica, build_checkpoint  # noqa: E402
from task_journal import JournalEvent, SQLiteTaskJournal  # noqa: E402

WF = "wf-wired"
DIGEST = "c" * 64
SOURCE = "node-a"
STREAM = "journal-1"


def _watermark(sequence: int = 10, digest: str = DIGEST) -> CommitWatermark:
    return CommitWatermark(workflow_id=WF, last_sequence=sequence, checkpoint_digest=digest)


def _voters(tmp_path: Path, *, count: int = 3, holders: dict | None = None):
    from cryptography.hazmat.primitives import serialization
    from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

    pairs, identities = [], []
    for index in range(count):
        private = Ed25519PrivateKey.generate()
        public = base64.urlsafe_b64encode(
            private.public_key().public_bytes(
                encoding=serialization.Encoding.Raw,
                format=serialization.PublicFormat.Raw,
            )
        ).decode("ascii").rstrip("=")
        voter_id = f"voter-{index}"
        pairs.append((voter_id, private))
        identities.append(VoterIdentity(voter_id=voter_id, public_key=public))
    voter_set = VoterSet(cluster_id="cluster-wired", voter_set_epoch=1, voters=identities)

    def _held_for(voter_id: str):
        if holders is None:
            return None
        entry = holders.get(voter_id)
        return None if entry is None else _watermark(*entry)

    voters = {}
    for voter_id, private in pairs:
        voters[voter_id] = QuorumVoter(
            voter_id=voter_id,
            private_key=private,
            voter_set=voter_set,
            ledger=SQLiteVoterLedger(
                tmp_path / f"{voter_id}.sqlite3",
                voter_id=voter_id,
                cluster_id=voter_set.cluster_id,
                voter_set_epoch=voter_set.voter_set_epoch,
            ),
            watermark_source=(
                None if holders is None
                else (lambda workflow_id, vid=voter_id: _held_for(vid))
            ),
        )
    collector = QuorumCollector(voter_set=voter_set, voters=voters)
    authority = ControlPlaneAuthority(
        cluster_id=voter_set.cluster_id, voter_set=voter_set, static_role="unknown",
    )
    return voter_set, collector, authority


def test_provider_watermark_reaches_the_certificate(tmp_path):
    _voter_set, collector, authority = _voters(
        tmp_path, holders={"voter-0": (10, DIGEST), "voter-1": (10, DIGEST)},
    )
    controller = AutoRoleController(
        "voter-0", mode="auto", authority=authority, collector=collector,
        commit_watermark_provider=lambda: (_watermark(),),
    )
    decision = controller.start(available_voter_ids=("voter-0", "voter-1", "voter-2"), now_ms=1_000)
    assert decision.accepted is True
    assert controller.certificate is not None
    assert controller.certificate.commit_watermark == (_watermark(),)


def test_provider_without_voter_durability_cannot_form_quorum(tmp_path):
    """提议了前缀但多数 voter 没有耐久日志 ⇒ 必须失败，且原因点明是水位问题。"""
    _voter_set, collector, authority = _voters(tmp_path)  # 无 holders
    controller = AutoRoleController(
        "voter-0", mode="auto", authority=authority, collector=collector,
        commit_watermark_provider=lambda: (_watermark(),),
    )
    decision = controller.start(available_voter_ids=("voter-0", "voter-1", "voter-2"), now_ms=1_000)
    assert decision.accepted is False
    assert decision.reason == "quorum_watermark_unavailable"
    assert decision.runtime_role == "client"


def test_failing_prefix_guard_keeps_controller_read_only(tmp_path):
    _voter_set, collector, authority = _voters(
        tmp_path, holders={"voter-0": (10, DIGEST), "voter-1": (10, DIGEST)},
    )

    def _reject(certificate):
        raise AutoRoleRuntimeError("committed_prefix_behind", "replica is behind")

    controller = AutoRoleController(
        "voter-0", mode="auto", authority=authority, collector=collector,
        commit_watermark_provider=lambda: (_watermark(),),
        committed_prefix_guard=_reject,
    )
    decision = controller.start(available_voter_ids=("voter-0", "voter-1", "voter-2"), now_ms=1_000)
    assert decision.accepted is False
    assert decision.state == "read_only"
    assert decision.reason == "committed_prefix_unverified"
    assert controller.certificate is None


# ---------------------------------------------------------------------------
# JournalPrefixAuthority：用真副本覆盖「截断未提交尾部 / 落后必须追赶」
# ---------------------------------------------------------------------------


def _snapshot(workflow_id: str, sequence: int) -> dict:
    return {
        "workflow_id": workflow_id,
        "last_sequence": sequence,
        "state": "running",
        "created_at": 100.0,
        "updated_at": 100.0 + sequence,
        "stages": [{"stage_id": "s1", "state": "running", "retry_safe": True, "pure": True, "attempts": []}],
    }


def _replica(tmp_path: Path, workflow_id: str, sequences: int) -> DurableJournalReplica:
    source = SQLiteTaskJournal(str(tmp_path / "source.sqlite3"))
    for sequence in range(1, sequences + 1):
        source.append_event(
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
    replica = DurableJournalReplica(
        str(tmp_path / "replica.sqlite3"), source_node_id=SOURCE, stream_id=STREAM,
    )
    replica.apply_checkpoint(build_checkpoint(
        source, source_node_id=SOURCE, stream_id=STREAM, workflow_id=workflow_id, created_at=1.0,
    ))
    source.close()
    return replica


def test_authority_held_and_propose_reflect_durable_prefix(tmp_path):
    replica = _replica(tmp_path, WF, 3)
    authority = JournalPrefixAuthority(replica, workflows=(WF,))
    held = authority.held(WF)
    assert held is not None
    assert held.last_sequence == 3
    assert authority.propose() == (held,)
    replica.close()


def test_authority_truncates_uncommitted_tail_then_requires_catch_up(tmp_path):
    """本地有 5 点、证书只承诺 3 点 ⇒ 先截断尾部，再因为尚不自洽而拒绝接管。"""
    replica = _replica(tmp_path, WF, 5)
    authority = JournalPrefixAuthority(replica, workflows=(WF,))
    committed = authority.held(WF)
    assert committed is not None and committed.last_sequence == 5

    stale = CommitWatermark(workflow_id=WF, last_sequence=3, checkpoint_digest="d" * 64)
    with pytest.raises(AutoRoleRuntimeError) as excinfo:
        authority.enforce(type("C", (), {"commit_watermark": (stale,)})())
    assert excinfo.value.code == "committed_prefix_behind"
    assert authority.truncated[WF] == 2
    assert replica.durable_prefix(WF)["last_sequence"] == 3
    replica.close()


def test_authority_rejects_prefix_behind_the_commitment(tmp_path):
    replica = _replica(tmp_path, WF, 3)
    authority = JournalPrefixAuthority(replica, workflows=(WF,))
    ahead = authority.held(WF)
    assert ahead is not None
    later = CommitWatermark(workflow_id=WF, last_sequence=9, checkpoint_digest=ahead.checkpoint_digest)
    with pytest.raises(AutoRoleRuntimeError) as excinfo:
        authority.enforce(type("C", (), {"commit_watermark": (later,)})())
    assert excinfo.value.code == "committed_prefix_behind"
    replica.close()


def test_authority_rejects_divergent_prefix(tmp_path):
    replica = _replica(tmp_path, WF, 3)
    authority = JournalPrefixAuthority(replica, workflows=(WF,))
    held = authority.held(WF)
    assert held is not None
    forked = CommitWatermark(workflow_id=WF, last_sequence=3, checkpoint_digest="e" * 64)
    with pytest.raises(AutoRoleRuntimeError) as excinfo:
        authority.enforce(type("C", (), {"commit_watermark": (forked,)})())
    assert excinfo.value.code == "committed_prefix_forked"
    replica.close()


def test_authority_accepts_matching_commitment(tmp_path):
    replica = _replica(tmp_path, WF, 3)
    authority = JournalPrefixAuthority(replica, workflows=(WF,))
    held = authority.held(WF)
    assert held is not None
    authority.enforce(type("C", (), {"commit_watermark": (held,)})())  # 不抛即通过
    assert authority.truncated == {}
    replica.close()


# ---------------------------------------------------------------------------
# #93 远程 voter：拒签原因必须能被主节点看见（真机演练暴露的吞码缺陷）
# ---------------------------------------------------------------------------

def test_voter_rpc_keeps_the_watermark_code(tmp_path):
    """voter 的 RPC 层必须保留 `quorum_watermark_*`，否则主节点会把「多数不持有」
    误当成「多数不可用」—— 这正是本票要消除的歧义（真机演练里就踩到了）。"""
    from cluster_auto_role_runtime import AutoRoleRuntime
    from cluster_control_contract import QuorumCertificate

    voter_set, collector, _authority = _voters(tmp_path)  # 无 holders ⇒ voter 没有耐久来源
    local = collector.voters["voter-0"]
    runtime = AutoRoleRuntime(voter=local, collector=collector, secret="s" * 40)
    certificate = QuorumCertificate(
        cluster_id=voter_set.cluster_id,
        voter_set_epoch=voter_set.voter_set_epoch,
        term=1, leader_id="voter-0", lease_id="lease-x", expires_at_ms=10**12,
        signatures=(), commit_watermark=(_watermark(),),
    )
    result = runtime.handle_rpc({"op": "sign_vote", "certificate": certificate.to_dict()})
    assert result["ok"] is False
    assert result["code"] == "quorum_watermark_unavailable"


def test_remote_voter_call_propagates_watermark_code(monkeypatch):
    """`HTTPRemoteVoter` 不得把水位码降级成通用码（白名单必须放行它）。"""
    import json as _json
    import threading
    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

    from cluster_auto_role_runtime import HTTPRemoteVoter
    from cluster_quorum import QuorumError

    monkeypatch.setenv("HTTP_PROXY", "http://127.0.0.1:1")
    monkeypatch.setenv("HTTPS_PROXY", "http://127.0.0.1:1")
    monkeypatch.setenv("NO_PROXY", "")

    class _Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, *args):  # noqa: D102 - silence
            return

        def do_POST(self):  # noqa: N802 - http.server API
            body = _json.dumps({"ok": False, "code": "quorum_watermark_unheld"}).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Connection", "close")
            self.end_headers()
            self.wfile.write(body)

    server = ThreadingHTTPServer(("127.0.0.1", 0), _Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        host, port = server.server_address[:2]
        remote = HTTPRemoteVoter("rog-voter", f"http://{host}:{port}", secret="s" * 40, timeout=3.0)
        with pytest.raises(QuorumError) as excinfo:
            remote.snapshot()
        assert excinfo.value.code == "quorum_watermark_unheld"
    finally:
        server.shutdown()
        server.server_close()
