from __future__ import annotations

import base64
import json
import sys
import threading
import time
from types import SimpleNamespace
from pathlib import Path

import pytest

sys.path.insert(0, "src")

from cluster_auto_role_runtime import (  # noqa: E402
    AutoRoleRuntime,
    AutoRoleRuntimeError,
    build_auto_role_runtime,
    install_auto_role_controller,
)
from cluster_control_contract import VoterIdentity, VoterSet  # noqa: E402


def _key_material():
    from cryptography.hazmat.primitives import serialization
    from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

    keys = [Ed25519PrivateKey.generate() for _ in range(3)]
    voters = {}
    for index, key in enumerate(keys):
        raw = key.public_key().public_bytes(
            encoding=serialization.Encoding.Raw,
            format=serialization.PublicFormat.Raw,
        )
        voters[f"voter-{index}"] = base64.urlsafe_b64encode(raw).decode("ascii").rstrip("=")
    return keys, VoterSet(cluster_id="runtime-test", voter_set_epoch=1, voters=voters)


def _private_key_text(key) -> str:
    from cryptography.hazmat.primitives import serialization

    raw = key.private_bytes(
        encoding=serialization.Encoding.Raw,
        format=serialization.PrivateFormat.Raw,
        encryption_algorithm=serialization.NoEncryption(),
    )
    return base64.urlsafe_b64encode(raw).decode("ascii").rstrip("=")


def test_runtime_requires_explicit_voter_material(monkeypatch):
    monkeypatch.delenv("QLH_CONTROL_VOTER_SET", raising=False)
    with pytest.raises(AutoRoleRuntimeError) as error:
        build_auto_role_runtime()
    assert error.value.code == "auto_role_voter_set_missing"


def test_node_config_preserves_explicit_auto_role(monkeypatch):
    import node_config

    monkeypatch.setenv("QLH_NODE_ROLE", "auto")
    assert node_config.resolve_initial_node_role() == "auto"


def test_runtime_builds_local_voter_and_peer_collector(monkeypatch, tmp_path: Path):
    keys, voter_set = _key_material()
    monkeypatch.setenv("QLH_CONTROL_VOTER_SET", json.dumps(voter_set.to_dict()))
    monkeypatch.setenv("QLH_CONTROL_VOTER_ID", "voter-0")
    monkeypatch.setenv("QLH_CONTROL_VOTER_PRIVATE_KEY", _private_key_text(keys[0]))
    monkeypatch.setenv("QLH_CONTROL_VOTER_LEDGER", str(tmp_path / "voter.sqlite3"))
    monkeypatch.setenv("QLH_CONTROL_VOTER_RPC_SECRET", "test-secret")
    monkeypatch.setenv(
        "QLH_CONTROL_VOTER_PEERS",
        json.dumps({"voter-1": "http://127.0.0.1:18001"}),
    )
    runtime = build_auto_role_runtime()
    assert runtime.available_voter_ids == ("voter-0", "voter-1")
    assert 0 < runtime.collector.voters["voter-1"]._timeout <= 1.25
    assert runtime.authenticate("test-secret") is True
    assert runtime.authenticate("bad") is False
    snapshot = runtime.handle_rpc({"op": "snapshot"})
    assert snapshot["ok"] is True
    assert snapshot["snapshot"]["voter_id"] == "voter-0"


def test_auto_role_runtime_rejects_too_short_lease(monkeypatch, tmp_path: Path):
    keys, voter_set = _key_material()
    monkeypatch.setenv("QLH_CONTROL_VOTER_SET", json.dumps(voter_set.to_dict()))
    monkeypatch.setenv("QLH_CONTROL_VOTER_ID", "voter-0")
    monkeypatch.setenv("QLH_CONTROL_VOTER_PRIVATE_KEY", _private_key_text(keys[0]))
    monkeypatch.setenv("QLH_CONTROL_VOTER_LEDGER", str(tmp_path / "voter.sqlite3"))
    monkeypatch.setenv("QLH_CONTROL_VOTER_RPC_SECRET", "test-secret")
    monkeypatch.setenv("QLH_CONTROL_VOTER_PEERS", json.dumps({"voter-1": "http://127.0.0.1:18001"}))
    monkeypatch.setenv("QLH_CONTROL_VOTER_LEASE_MS", "100")
    with pytest.raises(AutoRoleRuntimeError) as rejected:
        build_auto_role_runtime()
    assert rejected.value.code == "auto_role_lease_too_short"


def test_install_attaches_controller_to_scheduler(monkeypatch, tmp_path: Path):
    keys, voter_set = _key_material()
    monkeypatch.setenv("QLH_NODE_ROLE", "auto")
    monkeypatch.setenv("QLH_CONTROL_VOTER_SET", json.dumps(voter_set.to_dict()))
    monkeypatch.setenv("QLH_CONTROL_VOTER_ID", "voter-0")
    monkeypatch.setenv("QLH_CONTROL_VOTER_PRIVATE_KEY", _private_key_text(keys[0]))
    monkeypatch.setenv("QLH_CONTROL_VOTER_LEDGER", str(tmp_path / "voter.sqlite3"))
    monkeypatch.setenv("QLH_CONTROL_VOTER_RPC_SECRET", "test-secret")
    monkeypatch.setenv("QLH_CONTROL_VOTER_PEERS", json.dumps({"voter-1": "http://127.0.0.1:18001"}))

    class SchedulerStub:
        def set_auto_role_controller(self, controller):
            self.controller = controller

        def set_control_fence(self, fence):
            self.fence = fence

    scheduler = SchedulerStub()
    runtime = install_auto_role_controller(scheduler)
    assert runtime is not None
    assert scheduler.controller.mode == "auto"
    assert scheduler.controller.collector is runtime.collector
    assert scheduler.fence.role_guard is not None
    assert scheduler.fence.role_guard() is False


def test_runtime_renews_live_lease_and_stops_supervisor():
    from cluster_quorum import QuorumPolicy

    collector = SimpleNamespace(
        policy=QuorumPolicy(lease_duration_ms=90), voters={"voter-0": object()},
    )
    runtime = AutoRoleRuntime(voter=object(), collector=collector, secret="test")
    called = threading.Event()

    class Controller:
        state = "leader"
        renew_count = 0

        def renew_lease(self, *, available_voter_ids):
            assert available_voter_ids == ("voter-0",)
            self.renew_count += 1
            called.set()
            return SimpleNamespace(accepted=True)

    controller = Controller()
    runtime.start_lease_renewal(controller)
    try:
        assert called.wait(timeout=2)
        runtime.start_lease_renewal(controller)
    finally:
        runtime.stop_lease_renewal()
    count = controller.renew_count
    time.sleep(0.1)
    assert controller.renew_count == count
