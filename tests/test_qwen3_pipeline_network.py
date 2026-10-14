from __future__ import annotations

import hashlib
import json
from pathlib import Path
import socket
import subprocess
import sys
import time
from urllib.parse import urlsplit

from fastapi import FastAPI
from fastapi.testclient import TestClient
import pytest

# ★ 本文件起真实 socket 与子进程（进程目标链 / 副作用释放），并行跑会与其它用例抢
#   端口和进程资源：`-n 4` 全量下实测偶发失败，而单跑与 `-n 4` 单文件各 3 次均通过。
#   按 `pytest.ini` 中 `external` 的既有语义（"run serially"），整文件交给
#   `scripts/run_test_channels.py` 的串行 external 段，不进 unit 并行段。
pytestmark = pytest.mark.external


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import scheduler as scheduler_module  # noqa: E402
import scheduler_svc_http  # noqa: E402
from qwen3_pipeline_contract import build_kv_contract  # noqa: E402
from qwen3_pipeline_control import (  # noqa: E402
    Qwen3LoopbackNetworkControlClient,
    router as network_control_router,
)
from qwen3_pipeline_data_plane import (  # noqa: E402
    Qwen3ArtifactTransferRuntime,
    router as transfer_router,
)
from qwen3_pipeline_multisidecar import (  # noqa: E402
    Qwen3MultiSidecarError,
    Qwen3PipelineMultiSidecar,
)
from qwen3_pipeline_network import (  # noqa: E402
    Qwen3NetworkError,
    Qwen3NetworkHandoffTransport,
    Qwen3NetworkTarget,
    Qwen3NetworkTransferCoordinator,
    validate_qwen3_artifact_reference,
)
from qwen3_pipeline_peer_auth import (  # noqa: E402
    QWEN3_PEER_PROOF_HEADER,
    Qwen3PeerAuthError,
    Qwen3PeerAuthMiddleware,
    Qwen3PeerRequestSigner,
    Qwen3PeerRequestVerifier,
)
from qwen3_pipeline_transaction import build_qwen3_dry_run_contract  # noqa: E402
from qwen3_pipeline_transfer import (  # noqa: E402
    QWEN3_TRANSFER_PREFIX,
    Qwen3ArtifactTransferClient,
    TransferResponse,
)
from qwen3_pipeline_transfer import default_transfer_request  # noqa: E402


SECRET = "qwen3-network-contract-secret-value!!"


class _NetworkLedger:
    def __init__(self) -> None:
        self.value = {}

    def load(self):
        return json.loads(json.dumps(self.value)) if self.value else {
            "schema_version": 1,
            "local_node_id": "",
            "last_generation": -1,
            "active_contract": {},
            "transfers": {},
            "outputs": {},
            "updated_at": "",
        }

    def save(self, value):
        self.value = json.loads(json.dumps(value))
        return self.load()


def _contract(*, segment_count=3, generation=5):
    nodes = ["node-a", "node-b", "node-c"][:segment_count]
    segments = []
    for index, node_id in enumerate(nodes):
        segments.append({
            "node_id": node_id,
            "layer_range": [index * 2, (index + 1) * 2],
            "has_embedding": index == 0,
            "has_lm_head": index == segment_count - 1,
            "required_bytes": 100,
            "assignment_manifest_sha256": f"{index + 1}" * 64,
            "execution_device": "cpu",
            "dtype": "float32",
        })
    return build_qwen3_dry_run_contract(
        config_id=f"cfg-{generation}",
        plan_id=f"plan-{generation}",
        generation=generation,
        model_id="qwen3-4b",
        model_sha256="a" * 64,
        total_layers=segment_count * 2,
        hidden_size=4,
        execution_mode="node_local_sidecar",
        segments=segments,
    )


def _requester(client, *, calls=None, disconnect_after_patch=None):
    state = disconnect_after_patch

    def request(method, url, headers, body):
        parsed = urlsplit(url)
        target = parsed.path + (f"?{parsed.query}" if parsed.query else "")
        response = client.request(
            method,
            target,
            headers=dict(headers),
            content=body,
        )
        if calls is not None:
            calls.append({
                "method": method,
                "target": target,
                "body_bytes": 0 if body is None else len(body),
                "status": response.status_code,
            })
        if method == "PATCH" and state is not None and state["pending"]:
            state["pending"] = False
            raise OSError("simulated disconnect after persisted PATCH")
        return TransferResponse(
            status_code=response.status_code,
            headers=dict(response.headers),
            content=response.content,
        )

    return request


def _node(tmp_path, node_id, allowed_peers, *, now=None):
    clock = (lambda: now[0]) if now is not None else None
    runtime = Qwen3ArtifactTransferRuntime.create(
        state_dir=tmp_path / node_id,
        cluster_secret=SECRET,
        clock=clock,
    )
    coordinator = Qwen3NetworkTransferCoordinator(
        local_node_id=node_id,
        runtime=runtime,
    )
    verifier = Qwen3PeerRequestVerifier(
        SECRET,
        is_authenticated_peer=lambda peer: peer in allowed_peers,
        **({"clock": clock} if clock is not None else {}),
    )
    app = FastAPI()
    app.state.qwen3_artifact_transfer = runtime
    app.add_middleware(Qwen3PeerAuthMiddleware, verifier=verifier)
    app.include_router(transfer_router)
    return runtime, coordinator, TestClient(app)


def _attach_memory_ledger(coordinator):
    state = {
        "value": {
            "schema_version": 1,
            "local_node_id": coordinator.local_node_id,
            "last_generation": -1,
            "active_contract": {},
            "transfers": {},
            "outputs": {},
            "updated_at": "",
        },
    }

    def load():
        return json.loads(json.dumps(state["value"]))

    def save(value):
        state["value"] = json.loads(json.dumps(value))
        return load()

    coordinator.configure_persistent_ledger(load=load, save=save)
    return state


class _Session:
    def __init__(self, index, layer_range, *, fail_phase=None):
        self.index = index
        self.layer_range = layer_range
        self.fail_phase = fail_phase
        self.calls = []
        self.input_paths = []

    def prepare(self):
        self.calls.append("prepare")
        return {"status": "prepared", "gate_passed": True}

    def commit(self):
        self.calls.append("commit")
        return {
            "status": "committed",
            "gate_passed": True,
            "full_model_materialized": False,
        }

    def execute(self, **request):
        phase = request["phase"]
        self.calls.append(phase)
        self.input_paths.append(Path(request["input_ref"]).resolve())
        if self.fail_phase == phase:
            raise RuntimeError(f"{phase} failed on segment {self.index}")
        incoming = Path(request["input_ref"]).read_bytes()
        data = f"{phase}:{self.index}:".encode("ascii") + incoming
        output = Path(request["output_ref"])
        output.write_bytes(data)
        digest = hashlib.sha256(data).hexdigest()
        sequence = int(request["sequence_length"])
        batch = int(request["batch_size"])
        generation = int(request["generation"])
        dtype = str(request["dtype"])
        device = str(request["device"])
        return {
            "status": "executed",
            "gate_passed": True,
            "execution": {
                "full_model_materialized": False,
                "segment_materialized": True,
                "artifact_bytes": len(data),
                "artifact_sha256": digest,
            },
            "kv_contract": build_kv_contract(
                chain_id=request["chain_id"],
                segment_index=self.index,
                layer_range=self.layer_range,
                sequence_length=sequence,
                batch_size=batch,
                dtype=dtype,
                device=device,
                phase=phase,
                generation=generation,
            ),
            "hidden_handoff": (
                {
                    "schema_version": 1,
                    "chain_id": request["chain_id"],
                    "from_segment": self.index,
                    "to_segment": self.index + 1,
                    "shape": [batch, sequence, 4],
                    "batch_size": batch,
                    "sequence_length": sequence,
                    "hidden_size": 4,
                    "dtype": dtype,
                    "device": device,
                }
                if request["has_next_segment"]
                else None
            ),
        }

    def release(self):
        self.calls.append("release")
        return {"status": "released"}

    def abort(self):
        self.calls.append("abort")
        return {"status": "aborted"}


def _network_chain(tmp_path, *, segment_count=3, fail_index=None, disconnect=False):
    contract = _contract(segment_count=segment_count)
    root = tmp_path / "chain"
    root.mkdir()
    allowed = {"node-a", "node-b"}
    targets = {}
    calls = []
    disconnect_state = {"pending": bool(disconnect)}
    for index, node_id in enumerate(["node-b", "node-c"][:segment_count - 1], start=1):
        _runtime, coordinator, client = _node(root, node_id, allowed)
        targets[node_id] = Qwen3NetworkTarget(
            node_id=node_id,
            base_url=f"http://127.0.0.1:{9800 + index}",
            coordinator=coordinator,
            requester=_requester(
                client,
                calls=calls,
                disconnect_after_patch=(
                    disconnect_state if disconnect and node_id == "node-b" else None
                ),
            ),
        )
    transport = Qwen3NetworkHandoffTransport(
        artifact_root=root,
        targets=targets,
        peer_signers={
            node_id: Qwen3PeerRequestSigner(
                SECRET,
                peer_node_id=node_id,
            )
            for node_id in ["node-a", "node-b"][:segment_count - 1]
        },
        chunk_bytes=7,
    )
    sessions = []

    def factory(frame):
        index = int(frame["segment_index"])
        session = _Session(
            index,
            list(frame["layer_range"]),
            fail_phase="prefill" if fail_index == index else None,
        )
        sessions.append(session)
        return session

    chain = Qwen3PipelineMultiSidecar.from_contract(
        contract=contract,
        artifact_root=root,
        session_factory=factory,
        handoff_transport=transport,
    )
    source = root / "input.pt"
    source.write_bytes(bytes(range(40)))
    return chain, sessions, source, root, targets, calls, disconnect_state


def test_peer_proof_binds_request_ticket_tcp_registry_and_nonce():
    now = [1000.0]
    allowed = {"node-a"}
    signer = Qwen3PeerRequestSigner(
        SECRET,
        peer_node_id="node-a",
        clock=lambda: now[0],
    )
    verifier = Qwen3PeerRequestVerifier(
        SECRET,
        is_authenticated_peer=lambda peer: peer in allowed,
        clock=lambda: now[0],
    )
    path = f"{QWEN3_TRANSFER_PREFIX}/qtx_{'1' * 32}"
    proof = signer.proof("GET", path, "ticket", nonce="a" * 32)

    assert verifier.verify(
        proof,
        method="GET",
        path=path,
        transfer_ticket="ticket",
    ) == "node-a"
    with pytest.raises(Qwen3PeerAuthError) as replay:
        verifier.verify(proof, method="GET", path=path, transfer_ticket="ticket")
    assert replay.value.reason_code == "qwen3_peer_proof_replay"

    scoped = signer.proof("PATCH", path, "ticket", nonce="b" * 32)
    with pytest.raises(Qwen3PeerAuthError) as wrong_method:
        verifier.verify(scoped, method="GET", path=path, transfer_ticket="ticket")
    assert wrong_method.value.reason_code == "qwen3_peer_proof_scope"

    now[0] = 1031.0
    stale = signer.proof("GET", path, "ticket", nonce="c" * 32)
    now[0] = 1062.0
    with pytest.raises(Qwen3PeerAuthError) as expired:
        verifier.verify(stale, method="GET", path=path, transfer_ticket="ticket")
    assert expired.value.reason_code == "qwen3_peer_proof_expired"

    now[0] = 1070.0
    denied = signer.proof("GET", path, "ticket", nonce="d" * 32)
    allowed.clear()
    with pytest.raises(Qwen3PeerAuthError) as not_registered:
        verifier.verify(denied, method="GET", path=path, transfer_ticket="ticket")
    assert not_registered.value.reason_code == "qwen3_peer_not_authenticated"


def test_peer_auth_middleware_rejects_missing_proof_and_replay(tmp_path):
    runtime, coordinator, http = _node(tmp_path, "node-b", {"node-a"})
    contract = _contract(segment_count=2)
    coordinator.activate(contract)
    coordinator.begin_phase("prefill", contract["generation"])
    data = b"proof-bound"
    plan = coordinator.begin_receive(
        base_url="http://127.0.0.1:9876",
        source_peer_id="node-a",
        chain_id=contract["contract_sha256"],
        generation=contract["generation"],
        phase="prefill",
        from_segment=0,
        to_segment=1,
        size_bytes=len(data),
        sha256=hashlib.sha256(data).hexdigest(),
    )
    path = f"{QWEN3_TRANSFER_PREFIX}/{plan['transfer_id']}"
    authorization = {"Authorization": f"Bearer {plan['ticket']}"}
    missing = http.get(path, headers=authorization)
    assert missing.status_code == 401
    assert missing.json()["detail"]["code"] == "qwen3_peer_proof_missing"

    signer = Qwen3PeerRequestSigner(SECRET, peer_node_id="node-a")
    proof = signer.proof("GET", path, plan["ticket"])
    headers = {**authorization, QWEN3_PEER_PROOF_HEADER: proof}
    accepted = http.get(path, headers=headers)
    replay = http.get(path, headers=headers)
    assert accepted.status_code == 200
    assert replay.status_code == 401
    assert replay.json()["detail"]["code"] == "qwen3_peer_proof_replay"


def test_active_coordinator_fences_topology_generation_and_release(tmp_path):
    runtime, coordinator, _http = _node(tmp_path, "node-b", {"node-a"})
    contract = _contract(segment_count=2, generation=5)
    assert coordinator.activate(contract)["active"] is True
    data = b"contract-bound"
    fields = {
        "base_url": "http://127.0.0.1:9876",
        "chain_id": contract["contract_sha256"],
        "generation": 5,
        "phase": "prefill",
        "from_segment": 0,
        "to_segment": 1,
        "size_bytes": len(data),
        "sha256": hashlib.sha256(data).hexdigest(),
    }
    with pytest.raises(Qwen3NetworkError) as wrong_peer:
        coordinator.begin_receive(source_peer_id="node-c", **fields)
    assert wrong_peer.value.reason_code == "qwen3_network_contract_mismatch"
    with pytest.raises(Qwen3NetworkError) as stale_decode:
        coordinator.begin_receive(
            source_peer_id="node-a",
            **{**fields, "phase": "decode", "generation": 5},
        )
    assert stale_decode.value.reason_code == "qwen3_network_contract_mismatch"

    with pytest.raises(Qwen3NetworkError) as before_phase:
        coordinator.begin_receive(source_peer_id="node-a", **fields)
    assert before_phase.value.reason_code == "qwen3_network_contract_mismatch"
    coordinator.begin_phase("prefill", 5)
    plan = coordinator.begin_receive(source_peer_id="node-a", **fields)
    assert runtime.receiver.root.joinpath(f".{plan['transfer_id']}.part").is_file()
    cleanup = coordinator.release()
    assert cleanup["cleanup_complete"] is True
    assert not runtime.receiver.root.joinpath(f".{plan['transfer_id']}.part").exists()
    with pytest.raises(Qwen3NetworkError) as fenced:
        coordinator.activate(contract)
    assert fenced.value.reason_code == "qwen3_network_generation_stale"
    assert coordinator.activate(_contract(segment_count=2, generation=6))["generation"] == 6


@pytest.mark.parametrize("segment_count", [2, 3])
def test_network_chain_runs_prefill_decode_with_path_free_references_and_cleanup(
    tmp_path,
    segment_count,
):
    chain, sessions, source, root, targets, _calls, _disconnect = _network_chain(
        tmp_path,
        segment_count=segment_count,
    )
    chain.prepare()
    chain.commit()
    prefill = chain.prefill(input_ref=source, batch_size=1, sequence_length=3)
    decode = chain.decode(input_ref=source, batch_size=1, sequence_length=4)

    assert prefill["handoff_modes"] == ["network"]
    assert decode["handoff_reference_count"] == 2 * (segment_count - 1)
    for phase in ("prefill", "decode"):
        references = chain.handoff_references(phase)
        assert len(references) == segment_count - 1
        for reference in references:
            assert validate_qwen3_artifact_reference(reference) == reference
            serialized = json.dumps(reference)
            assert "path" not in serialized.lower()
            assert "ticket" not in serialized.lower()
            assert reference["mode"] == "network"
    for index in range(1, segment_count):
        expected_root = targets[["node-b", "node-c"][index - 1]].coordinator.runtime.receiver.root
        assert any(path.is_relative_to(expected_root) for path in sessions[index].input_paths)
    assert chain.snapshot["full_model_materialized"] is False

    released = chain.release()
    assert released["phase"] == "released"
    assert released["cleanup_complete"] is True
    assert not list(root.glob("**/qtx_*.pt"))
    assert not list(root.glob("**/*.part"))


def test_network_chain_resumes_after_lost_ack_without_duplicate_bytes(tmp_path):
    chain, _sessions, source, _root, _targets, calls, disconnect = _network_chain(
        tmp_path,
        segment_count=2,
        disconnect=True,
    )
    chain.prepare()
    chain.commit()
    result = chain.prefill(input_ref=source, batch_size=1, sequence_length=3)

    assert result["phase"] == "prefilled"
    assert disconnect["pending"] is False
    gets = [call for call in calls if call["method"] == "GET"]
    patches = [call for call in calls if call["method"] == "PATCH"]
    assert len(gets) >= 2
    assert patches[0]["status"] == 200
    assert sum(call["body_bytes"] for call in patches) == len(
        Path(chain.artifact_refs("prefill")[0]).read_bytes()
    )
    chain.cancel()


def test_downstream_failure_cancels_network_artifacts_and_all_sidecars(tmp_path):
    chain, sessions, source, root, targets, _calls, _disconnect = _network_chain(
        tmp_path,
        segment_count=2,
        fail_index=1,
    )
    chain.prepare()
    chain.commit()
    with pytest.raises(Qwen3MultiSidecarError):
        chain.prefill(input_ref=source, batch_size=1, sequence_length=3)
    assert chain.phase == "aborted"
    assert all("abort" in session.calls for session in sessions)
    assert targets["node-b"].coordinator.snapshot()["active"] is False
    assert not list(root.glob("**/qtx_*.pt"))
    assert not list(root.glob("**/*.part"))


def test_runtime_startup_reconciles_only_scoped_orphan_files(tmp_path):
    root = tmp_path / "state" / "qwen3" / "network_artifacts"
    root.mkdir(parents=True)
    temporary = root / f".qtx_{'1' * 32}.part"
    committed = root / f"qtx_{'2' * 32}.pt"
    unrelated = root / "user-data.pt"
    temporary.write_bytes(b"partial")
    committed.write_bytes(b"committed")
    unrelated.write_bytes(b"keep")

    runtime = Qwen3ArtifactTransferRuntime.create(
        state_dir=tmp_path / "state",
        cluster_secret=SECRET,
    )

    assert runtime.receiver.root == root.resolve()
    assert not temporary.exists()
    assert not committed.exists()
    assert unrelated.read_bytes() == b"keep"


def test_scheduler_service_wires_data_plane_to_live_tcp_peer_predicate(
    monkeypatch,
    tmp_path,
):
    class _TcpServer:
        _running = True

        def __init__(self):
            self.allowed = {"node-a"}

        def is_authenticated_loopback_client(self, peer):
            return peer in self.allowed

    sched = scheduler_module.Scheduler()
    server = _TcpServer()
    sched._tcp_server = server
    monkeypatch.setattr(sched, "_effective_role", lambda: "master")
    monkeypatch.setattr(sched, "_qwen3_cluster_secret", lambda: SECRET)
    runtime = Qwen3ArtifactTransferRuntime.create(
        state_dir=tmp_path,
        cluster_secret=SECRET,
    )
    configured = sched.configure_qwen3_artifact_transfer(runtime)
    assert configured["production_admitted"] is False
    http = TestClient(scheduler_svc_http.build_scheduler_app(sched))
    data = b"scheduler-wired"
    plan = runtime.begin_receive(
        base_url="http://127.0.0.1:9876",
        peer_node_id="node-a",
        chain_id="a" * 64,
        generation=1,
        phase="prefill",
        from_segment=0,
        to_segment=1,
        size_bytes=len(data),
        sha256=hashlib.sha256(data).hexdigest(),
    )
    path = f"{QWEN3_TRANSFER_PREFIX}/{plan['transfer_id']}"
    signer = Qwen3PeerRequestSigner(SECRET, peer_node_id="node-a")

    def headers():
        return {
            "Authorization": f"Bearer {plan['ticket']}",
            **signer.headers("GET", path, plan["ticket"]),
        }

    assert http.get(path, headers=headers()).status_code == 200
    server.allowed.clear()
    denied = http.get(path, headers=headers())
    assert denied.status_code == 401
    assert denied.json()["detail"]["code"] == "qwen3_peer_not_authenticated"


def test_scheduler_service_registers_authenticated_network_control_router(
    monkeypatch,
    tmp_path,
):
    class _TcpServer:
        _running = True

        def is_authenticated_loopback_client(self, peer):
            return peer == "node-a"

    sched = scheduler_module.Scheduler()
    sched._tcp_server = _TcpServer()
    monkeypatch.setattr(sched, "_effective_role", lambda: "master")
    monkeypatch.setattr(sched, "_qwen3_cluster_secret", lambda: SECRET)
    runtime = Qwen3ArtifactTransferRuntime.create(
        state_dir=tmp_path,
        cluster_secret=SECRET,
    )
    coordinator = Qwen3NetworkTransferCoordinator(
        local_node_id="node-b",
        runtime=runtime,
    )
    configured = sched.configure_qwen3_artifact_transfer(
        runtime,
        network_coordinator=coordinator,
    )
    assert configured["network_control"] is True
    http = TestClient(scheduler_svc_http.build_scheduler_app(sched))
    response = http.post(
        "/internal/v1/qwen3/network-control/activate",
        json={"contract": _contract(segment_count=2)},
    )
    assert response.status_code == 401
    assert response.json()["detail"]["code"] == "qwen3_peer_proof_missing"


def test_loopback_process_control_client_fences_control_and_data_plane(tmp_path):
    runtime, coordinator, http = _node(tmp_path, "node-b", {"node-a"})
    http.app.state.qwen3_network_transfer_coordinator = coordinator
    http.app.include_router(network_control_router)
    contract = _contract(segment_count=2, generation=8)
    signer = Qwen3PeerRequestSigner(SECRET, peer_node_id="node-a")

    def request(method, url, headers, body):
        parsed = urlsplit(url)
        return TransferResponse(
            status_code=(response := http.request(
                method,
                parsed.path + (f"?{parsed.query}" if parsed.query else ""),
                headers=dict(headers),
                content=body,
            )).status_code,
            headers=dict(response.headers),
            content=response.content,
        )

    remote = Qwen3LoopbackNetworkControlClient(
        node_id="node-b",
        base_url="http://127.0.0.1:9876",
        artifact_root=runtime.receiver.root,
        signer=signer,
        requester=request,
    )
    activated = remote.activate(contract)
    assert activated["active"] is True
    remote.begin_phase("prefill", contract["generation"])
    source = tmp_path / "control-source.pt"
    source.write_bytes(b"control-process-bound")
    plan = remote.begin_receive(
        base_url="http://127.0.0.1:9876",
        source_peer_id="node-a",
        chain_id=contract["contract_sha256"],
        generation=contract["generation"],
        phase="prefill",
        from_segment=0,
        to_segment=1,
        size_bytes=source.stat().st_size,
        sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
        ttl_seconds=60,
    )
    transfer = Qwen3ArtifactTransferClient(
        tmp_path,
        request,
        chunk_bytes=5,
        peer_proof_headers=signer.headers,
    )
    assert transfer.upload(source=source, plan=plan)["status"] == "committed"
    resolved = remote.resolve(plan["transfer_id"])
    assert resolved.path.read_bytes() == source.read_bytes()
    resolved.path.unlink()
    with pytest.raises(Qwen3NetworkError) as missing:
        remote.resolve(plan["transfer_id"])
    assert missing.value.reason_code == "qwen3_network_artifact_missing"
    remote.finish_phase("prefill", contract["generation"])
    released = remote.release()
    assert released["cleanup_complete"] is True
    assert not resolved.path.exists()


def test_loopback_process_control_rejects_body_rebinding(tmp_path):
    runtime, coordinator, http = _node(tmp_path, "node-b", {"node-a"})
    http.app.state.qwen3_network_transfer_coordinator = coordinator
    http.app.include_router(network_control_router)
    contract = _contract(segment_count=2, generation=9)
    signer = Qwen3PeerRequestSigner(SECRET, peer_node_id="node-a")
    first = json.dumps({"contract": contract}, ensure_ascii=True, sort_keys=True, separators=(",", ":")).encode()
    digest = hashlib.sha256(first).hexdigest()
    path = "/internal/v1/qwen3/network-control/activate"
    headers = {
        "Authorization": f"Bearer {digest}",
        "Content-Type": "application/json",
        "Content-Length": str(len(first)),
        **signer.headers("POST", path, digest),
    }
    changed = first.replace(b"cfg-9", b"cfg-x")
    response = http.post(path, headers=headers, content=changed)
    assert response.status_code == 401
    assert response.json()["detail"]["code"] == "qwen3_network_control_body_binding_mismatch"


def _free_port():
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _spawn_network_node(tmp_path, node_id, port, allowed_peers, *, synthetic_sidecar=False):
    helper = Path(__file__).parent / "helpers" / "qwen3_network_node.py"
    state_dir = tmp_path / node_id
    command = [
        sys.executable,
        str(helper),
        "--node-id", node_id,
        "--port", str(port),
        "--state-dir", str(state_dir),
        "--secret", SECRET,
    ]
    for peer in allowed_peers:
        command.extend(["--allowed-peer", peer])
    if synthetic_sidecar:
        command.append("--synthetic-sidecar")
    process = subprocess.Popen(
        command,
        cwd=str(Path(__file__).resolve().parents[1]),
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    url = f"http://127.0.0.1:{port}{QWEN3_TRANSFER_PREFIX}/status"
    deadline = time.monotonic() + 15
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError(f"Qwen3 test node {node_id} exited with {process.returncode}")
        try:
            response = default_transfer_request("GET", url, {}, None)
            if response.status_code == 200:
                return process, state_dir
        except Exception:
            pass
        time.sleep(0.05)
    process.terminate()
    process.wait(timeout=5)
    raise RuntimeError(f"Qwen3 test node {node_id} did not become ready")


def _stop_network_node(process):
    if process.poll() is None:
        process.terminate()
    try:
        process.wait(timeout=10)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait(timeout=5)


def test_same_machine_three_process_network_chain_prefill_decode(tmp_path):
    contract = _contract(segment_count=3, generation=12)
    root = tmp_path / "chain"
    root.mkdir()
    ports = {"node-b": _free_port(), "node-c": _free_port()}
    processes = []
    targets = {}
    try:
        for node_id, allowed in (("node-b", ["node-a"]), ("node-c", ["node-b"])):
            process, state_dir = _spawn_network_node(
                root, node_id, ports[node_id], allowed,
            )
            processes.append(process)
            targets[node_id] = Qwen3NetworkTarget(
                node_id=node_id,
                base_url=f"http://127.0.0.1:{ports[node_id]}",
                coordinator=Qwen3LoopbackNetworkControlClient(
                    node_id=node_id,
                    base_url=f"http://127.0.0.1:{ports[node_id]}",
                    artifact_root=state_dir / "qwen3" / "network_artifacts",
                    signer=Qwen3PeerRequestSigner(
                        SECRET,
                        peer_node_id="node-a" if node_id == "node-b" else "node-b",
                    ),
                ),
            )
        transport = Qwen3NetworkHandoffTransport(
            artifact_root=root,
            targets=targets,
            peer_signers={
                "node-a": Qwen3PeerRequestSigner(SECRET, peer_node_id="node-a"),
                "node-b": Qwen3PeerRequestSigner(SECRET, peer_node_id="node-b"),
            },
            chunk_bytes=7,
        )
        sessions = []

        def factory(frame):
            session = _Session(int(frame["segment_index"]), list(frame["layer_range"]))
            sessions.append(session)
            return session

        chain = Qwen3PipelineMultiSidecar.from_contract(
            contract=contract,
            artifact_root=root,
            session_factory=factory,
            handoff_transport=transport,
        )
        source = root / "input.pt"
        source.write_bytes(bytes(range(40)))
        chain.prepare()
        chain.commit()
        assert chain.prefill(input_ref=source, batch_size=1, sequence_length=3)["phase"] == "prefilled"
        assert chain.decode(input_ref=source, batch_size=1, sequence_length=4)["phase"] == "decoded"
        assert chain.handoff_references("prefill")[0]["mode"] == "network"
        assert chain.handoff_references("decode")[1]["target_node_id"] == "node-c"
        chain.release()
        assert not list(root.glob("**/qtx_*.pt"))
        assert not list(root.glob("**/*.part"))
    finally:
        for process in reversed(processes):
            _stop_network_node(process)


@pytest.mark.parametrize("segment_count", [2, 3])
def test_same_machine_process_target_chain_executes_and_releases_outputs(
    tmp_path, segment_count,
):
    contract = _contract(segment_count=segment_count, generation=41 + segment_count)
    root = tmp_path / "target-chain"
    root.mkdir()
    node_ids = ["node-b", "node-c"][:segment_count - 1]
    ports = {node_id: _free_port() for node_id in node_ids}
    processes = []
    targets = {}
    try:
        for node_id in node_ids:
            allowed = ["node-a", "node-c"] if node_id == "node-b" else ["node-b"]
            process, state_dir = _spawn_network_node(
                root, node_id, ports[node_id], allowed, synthetic_sidecar=True,
            )
            processes.append(process)
            signer_peer = "node-a" if node_id == "node-b" else "node-b"
            targets[node_id] = Qwen3NetworkTarget(
                node_id=node_id,
                base_url=f"http://127.0.0.1:{ports[node_id]}",
                coordinator=Qwen3LoopbackNetworkControlClient(
                    node_id=node_id,
                    base_url=f"http://127.0.0.1:{ports[node_id]}",
                    artifact_root=state_dir / "qwen3" / "network_artifacts",
                    signer=Qwen3PeerRequestSigner(SECRET, peer_node_id=signer_peer),
                ),
            )
        transport = Qwen3NetworkHandoffTransport(
            artifact_root=root,
            targets=targets,
            peer_signers={
                node_id: Qwen3PeerRequestSigner(SECRET, peer_node_id=node_id)
                for node_id in ["node-a", "node-b", "node-c"][:segment_count]
            },
            chunk_bytes=7,
            target_execution=True,
        )
        transport.activate(contract)
        source = root / "target-chain-input.pt"
        source.write_bytes(bytes(range(40)))
        phases = (
            ("prefill", contract["generation"], 3),
            ("decode", contract["generation"] + 1, 4),
        )
        phase_results = []
        for phase, generation, sequence_length in phases:
            transport.begin_phase(phase, generation)
            result = transport.execute_target_chain(
                source_path=source,
                phase=phase,
                generation=generation,
                batch_size=1,
                sequence_length=sequence_length,
            )
            phase_results.append(result)
            transport.finish_phase(phase, generation)
            assert result["completed"] is True
            assert result["target_execution_count"] == segment_count - 1
            assert result["released_output_ids"] == (
                [result["executions"][0]["output_reference"]["artifact_id"]]
                if segment_count == 3 else []
            )
            assert result["executions"][-1]["output_reference"] is None
            encoded = json.dumps(result, ensure_ascii=True).lower()
            assert "path" not in encoded
            assert "ticket" not in encoded
            assert "tensor" not in encoded
        assert all(
            any(
                record["status"] == "released"
                for record in json.loads(
                    (root / node_id / "qwen3-network-ledger.json").read_text(encoding="utf-8")
                )["outputs"].values()
            )
            for node_id in (["node-b"] if segment_count == 3 else [])
        )
        transport.cleanup()
        assert not list(root.glob("**/qtx_*.pt"))
        assert not list(root.glob("**/*.part"))
    finally:
        for process in reversed(processes):
            _stop_network_node(process)


def test_same_machine_process_target_restart_invalidates_output_reference(tmp_path):
    contract = _contract(segment_count=3, generation=47)
    root = tmp_path / "target-restart"
    root.mkdir()
    ports = {"node-b": _free_port(), "node-c": _free_port()}
    processes = []
    try:
        targets = {}
        state_dirs = {}
        for node_id, allowed in (
            ("node-b", ["node-a", "node-c"]),
            ("node-c", ["node-b"]),
        ):
            process, state_dir = _spawn_network_node(
                root, node_id, ports[node_id], allowed, synthetic_sidecar=True,
            )
            processes.append(process)
            state_dirs[node_id] = state_dir
            signer_peer = "node-a" if node_id == "node-b" else "node-b"
            targets[node_id] = Qwen3NetworkTarget(
                node_id=node_id,
                base_url=f"http://127.0.0.1:{ports[node_id]}",
                coordinator=Qwen3LoopbackNetworkControlClient(
                    node_id=node_id,
                    base_url=f"http://127.0.0.1:{ports[node_id]}",
                    artifact_root=state_dir / "qwen3" / "network_artifacts",
                    signer=Qwen3PeerRequestSigner(SECRET, peer_node_id=signer_peer),
                ),
            )
        transport = Qwen3NetworkHandoffTransport(
            artifact_root=root,
            targets=targets,
            peer_signers={
                node_id: Qwen3PeerRequestSigner(SECRET, peer_node_id=node_id)
                for node_id in ["node-a", "node-b", "node-c"]
            },
            chunk_bytes=7,
            target_execution=True,
        )
        transport.activate(contract)
        transport.begin_phase("prefill", contract["generation"])
        source = root / "restart-input.pt"
        source.write_bytes(bytes(range(31)))
        first = transport.transfer_and_consume(
            source_path=source,
            chain_id=contract["contract_sha256"], generation=contract["generation"],
            phase="prefill", from_segment=0, to_segment=1,
            source_node_id="node-a", target_node_id="node-b",
            batch_size=1, sequence_length=3, dtype="float32", device="cpu",
            has_next_segment=True,
        )
        output_reference = first["output_reference"]
        _stop_network_node(processes[0])
        processes[0], _ = _spawn_network_node(
            root, "node-b", ports["node-b"], ["node-a", "node-c"],
            synthetic_sidecar=True,
        )
        with pytest.raises(Qwen3NetworkError) as stale:
            transport.transfer_registered_output(
                output_reference=output_reference,
                chain_id=contract["contract_sha256"], generation=contract["generation"],
                phase="prefill", from_segment=1, to_segment=2,
                source_node_id="node-b", target_node_id="node-c",
            )
        assert stale.value.reason_code == "qwen3_network_output_missing"
        assert not list((state_dirs["node-c"] / "qwen3" / "network_artifacts").glob("qtx_*.pt"))
        assert not list((state_dirs["node-c"] / "qwen3" / "network_artifacts").glob("*.part"))
    finally:
        for process in reversed(processes):
            _stop_network_node(process)


def test_same_machine_network_node_restart_fences_stale_contract_and_reconciles_files(tmp_path):
    contract = _contract(segment_count=2, generation=14)
    port = _free_port()
    process, state_dir = _spawn_network_node(tmp_path / "restart", "node-b", port, ["node-a"])
    try:
        artifact_root = state_dir / "qwen3" / "network_artifacts"
        remote = Qwen3LoopbackNetworkControlClient(
            node_id="node-b",
            base_url=f"http://127.0.0.1:{port}",
            artifact_root=artifact_root,
            signer=Qwen3PeerRequestSigner(SECRET, peer_node_id="node-a"),
        )
        remote.activate(contract)
        artifact_root.mkdir(parents=True, exist_ok=True)
        orphan_part = artifact_root / f".qtx_{'a' * 32}.part"
        orphan_final = artifact_root / f"qtx_{'b' * 32}.pt"
        unrelated = artifact_root / "keep.pt"
        orphan_part.write_bytes(b"orphan")
        orphan_final.write_bytes(b"orphan")
        unrelated.write_bytes(b"keep")
        _stop_network_node(process)
        process, _ = _spawn_network_node(tmp_path / "restart", "node-b", port, ["node-a"])
        assert not orphan_part.exists()
        assert not orphan_final.exists()
        assert unrelated.read_bytes() == b"keep"
        with pytest.raises(Qwen3NetworkError) as stale:
            remote.begin_phase("prefill", contract["generation"])
        assert stale.value.reason_code == "qwen3_network_contract_inactive"
    finally:
        _stop_network_node(process)


def test_expired_transfer_reconciles_to_terminal_ledger_and_removes_staging(tmp_path):
    now = [1000.0]
    runtime, coordinator, _http = _node(
        tmp_path, "node-b", {"node-a"}, now=now,
    )
    ledger = _attach_memory_ledger(coordinator)
    contract = _contract(segment_count=2, generation=61)
    coordinator.activate(contract)
    coordinator.authorize_control_peer("node-a", contract=contract, peer_epoch=0)
    coordinator.begin_phase("prefill", contract["generation"])
    data = b"expires-before-upload"
    plan = coordinator.begin_receive(
        base_url="http://127.0.0.1:9876",
        source_peer_id="node-a",
        chain_id=contract["contract_sha256"],
        generation=contract["generation"],
        phase="prefill",
        from_segment=0,
        to_segment=1,
        size_bytes=len(data),
        sha256=hashlib.sha256(data).hexdigest(),
        ttl_seconds=1,
        peer_epoch=0,
    )
    transfer_id = plan["transfer_id"]
    assert list((tmp_path / "node-b" / "qwen3" / "network_artifacts").glob("*.part"))
    now[0] += 2
    result = coordinator.cleanup_expired()
    assert result["reconciled_transfers"] == 1
    assert result["cleanup_complete"] is True
    assert coordinator.snapshot()["transfer_count"] == 0
    assert ledger["value"]["transfers"][transfer_id]["status"] == "expired"
    assert not list((tmp_path / "node-b" / "qwen3" / "network_artifacts").glob("*.part"))


def test_committed_transfer_progress_replay_keeps_terminal_ledger_state(tmp_path):
    runtime, coordinator, _http = _node(tmp_path, "node-b", {"node-a"})
    ledger = _attach_memory_ledger(coordinator)
    contract = _contract(segment_count=2, generation=62)
    coordinator.activate(contract)
    coordinator.authorize_control_peer("node-a", contract=contract, peer_epoch=0)
    coordinator.begin_phase("prefill", contract["generation"])
    data = b"committed-progress-replay"
    plan = coordinator.begin_receive(
        base_url="http://127.0.0.1:9876",
        source_peer_id="node-a",
        chain_id=contract["contract_sha256"],
        generation=contract["generation"],
        phase="prefill",
        from_segment=0,
        to_segment=1,
        size_bytes=len(data),
        sha256=hashlib.sha256(data).hexdigest(),
    )
    transfer_id = plan["transfer_id"]
    runtime.receiver.write(
        transfer_id,
        ticket=plan["ticket"],
        authenticated_peer_id="node-a",
        offset=0,
        data=data,
    )
    runtime.receiver.commit(
        transfer_id,
        ticket=plan["ticket"],
        authenticated_peer_id="node-a",
    )
    reference = coordinator.commit_reference(transfer_id)
    assert coordinator.record_transfer_progress(transfer_id, len(data))["status"] == "committed"
    assert coordinator.record_transfer_progress(transfer_id, len(data)) == {
        "transfer_id": transfer_id,
        "received_bytes": len(data),
        "size_bytes": len(data),
        "status": "committed",
    }
    assert ledger["value"]["transfers"][transfer_id]["status"] == "committed"
    assert ledger["value"]["transfers"][transfer_id]["received_bytes"] == len(data)
    assert ledger["value"]["transfers"][transfer_id]["input_reference"] == reference


def test_same_machine_process_ttl_epoch_revoke_and_terminal_ledger(tmp_path):
    contract = _contract(segment_count=2, generation=63)
    root = tmp_path / "fault-matrix"
    root.mkdir()
    port = _free_port()
    process = None
    try:
        process, state_dir = _spawn_network_node(
            root, "node-b", port, ["node-a"], synthetic_sidecar=True,
        )
        epoch = {"value": 0}
        signer = Qwen3PeerRequestSigner(
            SECRET,
            peer_node_id="node-a",
            peer_epoch_provider=lambda: epoch["value"],
        )
        target = Qwen3NetworkTarget(
            node_id="node-b",
            base_url=f"http://127.0.0.1:{port}",
            coordinator=Qwen3LoopbackNetworkControlClient(
                node_id="node-b",
                base_url=f"http://127.0.0.1:{port}",
                artifact_root=state_dir / "qwen3" / "network_artifacts",
                signer=signer,
            ),
        )
        target.coordinator.activate(contract)
        target.coordinator.begin_phase("prefill", contract["generation"])
        payload = b"ttl-process-input"
        plan = target.coordinator.begin_receive(
            base_url=f"http://127.0.0.1:{port}",
            source_peer_id="node-a",
            chain_id=contract["contract_sha256"],
            generation=contract["generation"],
            phase="prefill",
            from_segment=0,
            to_segment=1,
            size_bytes=len(payload),
            sha256=hashlib.sha256(payload).hexdigest(),
            ttl_seconds=1,
        )
        transfer_id = plan["transfer_id"]
        time.sleep(1.25)
        status_url = f"http://127.0.0.1:{port}{QWEN3_TRANSFER_PREFIX}/status"
        status = default_transfer_request("GET", status_url, {}, None)
        assert status.status_code == 200
        ledger_path = state_dir / "qwen3-network-ledger.json"
        expired = json.loads(ledger_path.read_text(encoding="utf-8"))
        assert expired["transfers"][transfer_id]["status"] == "expired"
        assert not list((state_dir / "qwen3" / "network_artifacts").glob("*.part"))

        # A new registration epoch is accepted by the fresh proof.  The
        # expired transfer remains terminal and cannot be resurrected.
        registration_url = f"http://127.0.0.1:{port}/__fixture/registration"
        body = json.dumps({"peer": "node-a", "action": "register"}).encode("utf-8")
        registration = default_transfer_request(
            "POST", registration_url,
            {"Content-Type": "application/json", "Content-Length": str(len(body))}, body,
        )
        assert registration.status_code == 200
        epoch["value"] = 1
        expired_cancel = target.coordinator.cancel_transfer(transfer_id)
        assert expired_cancel["status"] == "missing"
        terminal = json.loads(ledger_path.read_text(encoding="utf-8"))
        assert terminal["transfers"][transfer_id]["status"] == "expired"
        assert not list((state_dir / "qwen3" / "network_artifacts").glob("**/*.part"))

        active = target.coordinator.begin_receive(
            base_url=f"http://127.0.0.1:{port}",
            source_peer_id="node-a",
            chain_id=contract["contract_sha256"],
            generation=contract["generation"],
            phase="prefill",
            from_segment=0,
            to_segment=1,
            size_bytes=len(payload),
            sha256=hashlib.sha256(payload).hexdigest(),
            ttl_seconds=60,
        )
        active_id = active["transfer_id"]
        body = json.dumps({"peer": "node-a", "action": "register"}).encode("utf-8")
        registration = default_transfer_request(
            "POST", registration_url,
            {"Content-Type": "application/json", "Content-Length": str(len(body))}, body,
        )
        assert registration.status_code == 200
        epoch["value"] = 2
        fenced_cancel = target.coordinator.cancel_transfer(active_id)
        assert fenced_cancel["status"] == "missing"
        fenced = json.loads(ledger_path.read_text(encoding="utf-8"))
        assert fenced["transfers"][active_id]["status"] == "invalidated"
        assert not list((state_dir / "qwen3" / "network_artifacts").glob("**/*.part"))
    finally:
        if process is not None:
            _stop_network_node(process)


def test_peer_registration_epoch_fences_reconnect_proof_and_transfer_ticket():
    epochs = {"node-a": 1}
    signer = Qwen3PeerRequestSigner(
        SECRET, peer_node_id="node-a", peer_epoch_provider=lambda: epochs["node-a"],
    )
    verifier = Qwen3PeerRequestVerifier(
        SECRET,
        is_authenticated_peer=lambda peer: peer in epochs,
        is_authenticated_peer_epoch=lambda peer, epoch: epochs.get(peer) == epoch,
    )
    ticket = "digest"
    proof = signer.proof("GET", "/internal/v1/qwen3/artifact-transfer/status", ticket)
    assert verifier.verify(
        proof,
        method="GET",
        path="/internal/v1/qwen3/artifact-transfer/status",
        transfer_ticket=ticket,
    ) == "node-a"
    epochs["node-a"] = 2
    fresh = signer.proof("GET", "/internal/v1/qwen3/artifact-transfer/status", ticket)
    with pytest.raises(Qwen3PeerAuthError) as stale:
        verifier.verify(
            proof,
            method="GET",
            path="/internal/v1/qwen3/artifact-transfer/status",
            transfer_ticket=ticket,
        )
    assert stale.value.reason_code == "qwen3_peer_epoch_mismatch"
    assert verifier.verify(
        fresh,
        method="GET",
        path="/internal/v1/qwen3/artifact-transfer/status",
        transfer_ticket=ticket,
    ) == "node-a"


def test_target_consume_returns_path_free_execution_contract(tmp_path):
    runtime = Qwen3ArtifactTransferRuntime.create(
        state_dir=tmp_path, cluster_secret=SECRET,
    )
    coordinator = Qwen3NetworkTransferCoordinator(
        local_node_id="node-b", runtime=runtime,
    )
    contract = _contract(segment_count=2, generation=18)
    coordinator.activate(contract)
    coordinator.begin_phase("prefill", contract["generation"])
    data = b"target-local-artifact"
    plan = coordinator.begin_receive(
        base_url="http://127.0.0.1:9876",
        source_peer_id="node-a",
        chain_id=contract["contract_sha256"],
        generation=contract["generation"],
        phase="prefill",
        from_segment=0,
        to_segment=1,
        size_bytes=len(data),
        sha256=hashlib.sha256(data).hexdigest(),
    )
    transfer_id = plan["transfer_id"]
    runtime.receiver.write(
        transfer_id,
        ticket=plan["ticket"],
        authenticated_peer_id="node-a",
        offset=0,
        data=data,
    )
    runtime.receiver.commit(
        transfer_id,
        ticket=plan["ticket"],
        authenticated_peer_id="node-a",
    )
    coordinator.resolve(transfer_id)
    result = coordinator.consume(
        transfer_id,
        phase="prefill",
        generation=contract["generation"],
        batch_size=1,
        sequence_length=4,
        dtype="float32",
        device="cpu",
        has_next_segment=False,
    )
    encoded = json.dumps(result, ensure_ascii=True)
    assert "path" not in encoded.lower()
    assert result["execution"]["artifact_sha256"] == hashlib.sha256(data).hexdigest()
    assert result["hidden_handoff"]["has_next_segment"] is False


class _TargetConsumeExecutor:
    def __init__(self, *, fail: bool = False) -> None:
        self.fail = fail
        self.calls = 0
        self.cleaned: list[str] = []
        self.outputs: list[Path] = []

    def __call__(self, input_path: Path, request: dict) -> dict:
        self.calls += 1
        output = input_path.parent / (
            f"qwen3-consume-{request['transfer_id']}-{request['phase']}-"
            f"{request['generation']}-{request['segment_index']}.pt"
        )
        output.write_bytes(b"target-sidecar-output")
        self.outputs.append(output)
        if self.fail:
            raise RuntimeError("synthetic target sidecar failure")
        return {
            "status": "executed",
            "gate_passed": True,
            "output_path": str(output),
            "execution": {
                "full_model_materialized": False,
                "segment_materialized": True,
            },
            "hidden_handoff": {
                "dtype": request["dtype"],
                "device": request["device"],
                "shape": [request["batch_size"], request["sequence_length"], 4],
            },
            "kv_contract": {
                "present": True,
                "shape": [request["batch_size"], request["sequence_length"]],
            },
        }

    def cleanup(self, request: dict, reason_code: str) -> None:
        self.cleaned.append(reason_code)
        for output in self.outputs:
            output.unlink(missing_ok=True)


def _committed_target_transfer(tmp_path, *, epoch: int = 1):
    runtime = Qwen3ArtifactTransferRuntime.create(
        state_dir=tmp_path, cluster_secret=SECRET,
    )
    coordinator = Qwen3NetworkTransferCoordinator(local_node_id="node-b", runtime=runtime)
    contract = _contract(segment_count=3, generation=21)
    coordinator.activate(contract)
    coordinator.authorize_control_peer("node-a", contract=contract, peer_epoch=epoch)
    coordinator.begin_phase("prefill", contract["generation"])
    data = b"target-sidecar-input"
    plan = coordinator.begin_receive(
        base_url="http://127.0.0.1:9876",
        source_peer_id="node-a",
        chain_id=contract["contract_sha256"],
        generation=contract["generation"],
        phase="prefill",
        from_segment=0,
        to_segment=1,
        size_bytes=len(data),
        sha256=hashlib.sha256(data).hexdigest(),
        peer_epoch=epoch,
    )
    transfer_id = plan["transfer_id"]
    runtime.receiver.write(
        transfer_id,
        ticket=plan["ticket"],
        authenticated_peer_id="node-a",
        authenticated_peer_epoch=epoch,
        offset=0,
        data=data,
    )
    runtime.receiver.commit(
        transfer_id,
        ticket=plan["ticket"],
        authenticated_peer_id="node-a",
        authenticated_peer_epoch=epoch,
    )
    coordinator.resolve(transfer_id)
    return coordinator, contract, transfer_id


def test_target_consume_registers_next_reference_and_is_idempotent(tmp_path):
    coordinator, contract, transfer_id = _committed_target_transfer(tmp_path)
    executor = _TargetConsumeExecutor()
    result = coordinator.consume(
        transfer_id,
        phase="prefill",
        generation=contract["generation"],
        batch_size=1,
        sequence_length=4,
        dtype="float32",
        device="cpu",
        has_next_segment=True,
        executor=executor,
    )
    encoded = json.dumps(result, ensure_ascii=True)
    assert "output_path" not in encoded
    assert "path" not in encoded.lower()
    output_reference = result["output_reference"]
    assert output_reference["source_node_id"] == "node-b"
    assert output_reference["target_node_id"] == "node-c"
    assert output_reference["from_segment"] == 1
    assert output_reference["to_segment"] == 2
    assert coordinator.consume(
        transfer_id,
        phase="prefill",
        generation=contract["generation"],
        batch_size=1,
        sequence_length=4,
        dtype="float32",
        device="cpu",
        has_next_segment=True,
        executor=executor,
    ) == result
    assert executor.calls == 1
    with pytest.raises(Qwen3NetworkError) as mismatch:
        coordinator.consume(
            transfer_id,
            phase="prefill",
            generation=contract["generation"],
            batch_size=1,
            sequence_length=5,
            dtype="float32",
            device="cpu",
            has_next_segment=True,
            executor=executor,
        )
    assert mismatch.value.reason_code == "qwen3_network_consume_duplicate_mismatch"
    assert coordinator.snapshot()["output_count"] == 1
    coordinator.cancel_transfer(transfer_id)
    assert coordinator.snapshot()["output_count"] == 0
    assert executor.cleaned == ["cancelled"]


def test_registered_output_chunk_read_revalidates_digest_and_release(tmp_path):
    coordinator, contract, transfer_id = _committed_target_transfer(tmp_path)
    executor = _TargetConsumeExecutor()
    result = coordinator.consume(
        transfer_id,
        phase="prefill",
        generation=contract["generation"],
        batch_size=1,
        sequence_length=4,
        dtype="float32",
        device="cpu",
        has_next_segment=True,
        executor=executor,
    )
    reference = result["output_reference"]
    first = coordinator.read_output_chunk(
        reference["artifact_id"], requester_peer_id="node-c", offset=0, max_bytes=7,
    )
    second = coordinator.read_output_chunk(
        reference["artifact_id"], requester_peer_id="node-c", offset=7, max_bytes=64,
    )
    assert first["data"] + second["data"] == b"target-sidecar-output"
    assert first["reference"]["artifact_id"] == reference["artifact_id"]
    assert second["eof"] is True

    output = executor.outputs[0]
    output.write_bytes(b"tampered")
    with pytest.raises(Qwen3NetworkError) as mismatch:
        coordinator.read_output_chunk(
            reference["artifact_id"], requester_peer_id="node-c", offset=0, max_bytes=7,
        )
    assert mismatch.value.reason_code == "qwen3_network_output_digest_mismatch"
    assert coordinator.snapshot()["output_count"] == 0
    assert coordinator.release_output_reference(reference["artifact_id"])["status"] == "missing"


def test_transfer_registered_output_is_path_free(tmp_path):
    contract = _contract(segment_count=3, generation=25)
    root = tmp_path / "chain"
    root.mkdir()
    source_runtime, source, _ = _node(root, "node-b", {"node-a", "node-c"})
    target_runtime, target, target_http = _node(root, "node-c", {"node-b"})
    source_exec = _TargetConsumeExecutor()
    source.configure_sidecar_executor(source_exec)
    target = Qwen3NetworkTarget(
        node_id="node-c", base_url="http://127.0.0.1:9877", coordinator=target,
        requester=_requester(target_http),
    )
    source_target = Qwen3NetworkTarget(
        node_id="node-b", base_url="http://127.0.0.1:9876", coordinator=source,
    )
    transport = Qwen3NetworkHandoffTransport(
        artifact_root=root,
        targets={"node-b": source_target, "node-c": target},
        peer_signers={
            "node-a": Qwen3PeerRequestSigner(SECRET, peer_node_id="node-a"),
            "node-b": Qwen3PeerRequestSigner(SECRET, peer_node_id="node-b"),
        },
        chunk_bytes=5,
    )
    transport.activate(contract)
    transport.begin_phase("prefill", contract["generation"])
    data = b"incoming-to-b"
    plan = source.begin_receive(
        base_url="http://127.0.0.1:9876", source_peer_id="node-a",
        chain_id=contract["contract_sha256"], generation=contract["generation"],
        phase="prefill", from_segment=0, to_segment=1,
        size_bytes=len(data), sha256=hashlib.sha256(data).hexdigest(),
    )
    source_runtime.receiver.write(
        plan["transfer_id"], ticket=plan["ticket"], authenticated_peer_id="node-a",
        offset=0, data=data,
    )
    source_runtime.receiver.commit(
        plan["transfer_id"], ticket=plan["ticket"], authenticated_peer_id="node-a",
    )
    source.commit_reference(plan["transfer_id"])
    consumed = source.consume(
        plan["transfer_id"], phase="prefill", generation=contract["generation"],
        batch_size=1, sequence_length=4, dtype="float32", device="cpu",
        has_next_segment=True,
    )
    output_reference = consumed["output_reference"]
    moved = transport.transfer_registered_output(
        output_reference=output_reference,
        chain_id=contract["contract_sha256"], generation=contract["generation"],
        phase="prefill", from_segment=1, to_segment=2,
        source_node_id="node-b", target_node_id="node-c",
    )
    encoded = json.dumps(moved, ensure_ascii=True)
    assert "path" not in encoded.lower()
    assert moved["input_reference"] == output_reference
    assert moved["target_reference"]["target_node_id"] == "node-c"
    target_path = target_runtime.receiver.artifact_path(moved["transfer_id"])
    assert target_path.read_bytes() == b"target-sidecar-output"


def test_persistent_ledger_tracks_prefill_decode_kv_and_terminal_release(tmp_path):
    contract = _contract(segment_count=3, generation=31)
    runtime = Qwen3ArtifactTransferRuntime.create(
        state_dir=tmp_path / "node-b", cluster_secret=SECRET,
    )
    coordinator = Qwen3NetworkTransferCoordinator(local_node_id="node-b", runtime=runtime)
    ledger = _attach_memory_ledger(coordinator)
    executor = _TargetConsumeExecutor()
    coordinator.configure_sidecar_executor(executor)
    coordinator.activate(contract)
    coordinator.begin_phase("prefill", contract["generation"])

    def consume_phase(phase, generation, payload):
        plan = coordinator.begin_receive(
            base_url="http://127.0.0.1:9876", source_peer_id="node-a",
            chain_id=contract["contract_sha256"], generation=generation,
            phase=phase, from_segment=0, to_segment=1,
            size_bytes=len(payload), sha256=hashlib.sha256(payload).hexdigest(),
        )
        runtime.receiver.write(
            plan["transfer_id"], ticket=plan["ticket"], authenticated_peer_id="node-a",
            offset=0, data=payload,
        )
        runtime.receiver.commit(
            plan["transfer_id"], ticket=plan["ticket"], authenticated_peer_id="node-a",
        )
        coordinator.commit_reference(plan["transfer_id"])
        result = coordinator.consume(
            plan["transfer_id"], phase=phase, generation=generation,
            batch_size=1, sequence_length=4, dtype="float32", device="cpu",
            has_next_segment=True,
        )
        return plan["transfer_id"], result

    prefill_id, prefill = consume_phase("prefill", contract["generation"], b"prefill-input")
    coordinator.finish_phase("prefill", contract["generation"])
    coordinator.begin_phase("decode", contract["generation"] + 1)
    decode_id, decode = consume_phase("decode", contract["generation"] + 1, b"decode-input")
    coordinator.finish_phase("decode", contract["generation"] + 1)

    persisted = ledger["value"]
    assert persisted["transfers"][prefill_id]["kv_contract"] == {
        "present": True, "shape": [1, 4],
    }
    assert persisted["transfers"][prefill_id]["generation"] == contract["generation"]
    assert persisted["transfers"][decode_id]["generation"] == contract["generation"] + 1
    assert persisted["transfers"][decode_id]["phase"] == "decode"
    assert persisted["outputs"][prefill["output_reference"]["artifact_id"]]["status"] == "registered"
    assert persisted["outputs"][decode["output_reference"]["artifact_id"]]["status"] == "registered"

    output_id = prefill["output_reference"]["artifact_id"]
    next_transfer_id = "qtx_" + "f" * 32
    assert coordinator.lease_output_reference(output_id, next_transfer_id)["status"] == "leased"
    coordinator.record_output_progress(
        output_id, next_transfer_id, prefill["output_reference"]["size_bytes"],
    )
    assert coordinator.commit_output_reference(output_id, next_transfer_id)["status"] == "committed"
    assert coordinator.commit_output_reference(output_id, next_transfer_id)["status"] == "committed"
    assert coordinator.lease_output_reference(output_id, next_transfer_id)["status"] == "committed"
    coordinator.release_output_reference(output_id)
    coordinator.release()
    assert ledger["value"]["active_contract"]["phase"] == "released"
    assert all(
        record["status"] == "released" for record in ledger["value"]["outputs"].values()
    )


def test_persistent_ledger_invalidates_restart_references_and_recovers_contract(tmp_path):
    contract = _contract(segment_count=3, generation=33)
    state_dir = tmp_path / "node-b"
    runtime = Qwen3ArtifactTransferRuntime.create(state_dir=state_dir, cluster_secret=SECRET)
    coordinator = Qwen3NetworkTransferCoordinator(local_node_id="node-b", runtime=runtime)
    ledger = _attach_memory_ledger(coordinator)
    coordinator.activate(contract)
    coordinator.begin_phase("prefill", contract["generation"])
    data = b"restart-input"
    plan = coordinator.begin_receive(
        base_url="http://127.0.0.1:9876", source_peer_id="node-a",
        chain_id=contract["contract_sha256"], generation=contract["generation"],
        phase="prefill", from_segment=0, to_segment=1,
        size_bytes=len(data), sha256=hashlib.sha256(data).hexdigest(),
    )
    runtime.receiver.write(
        plan["transfer_id"], ticket=plan["ticket"], authenticated_peer_id="node-a",
        offset=0, data=data,
    )
    runtime.receiver.commit(
        plan["transfer_id"], ticket=plan["ticket"], authenticated_peer_id="node-a",
    )
    coordinator.commit_reference(plan["transfer_id"])
    result = coordinator.consume(
        plan["transfer_id"], phase="prefill", generation=contract["generation"],
        batch_size=1, sequence_length=4, dtype="float32", device="cpu",
        has_next_segment=True, executor=_TargetConsumeExecutor(),
    )
    output_id = result["output_reference"]["artifact_id"]

    restarted_runtime = Qwen3ArtifactTransferRuntime.create(
        state_dir=state_dir, cluster_secret=SECRET,
    )
    restarted = Qwen3NetworkTransferCoordinator(
        local_node_id="node-b", runtime=restarted_runtime,
    )
    restarted.configure_persistent_ledger(
        load=lambda: json.loads(json.dumps(ledger["value"])),
        save=lambda value: ledger.update(value=json.loads(json.dumps(value))) or ledger["value"],
    )
    snapshot = restarted.ledger_snapshot()
    assert snapshot["restart_pending"] is True
    assert snapshot["invalidated_transfers"] >= 1
    assert snapshot["invalidated_outputs"] == 1
    with pytest.raises(Qwen3NetworkError) as stale:
        restarted.read_output_chunk(
            output_id, requester_peer_id="node-c", offset=0, max_bytes=4,
        )
    assert stale.value.reason_code == "qwen3_network_output_missing"
    assert restarted.activate(contract)["phase"] == "prepared"


def test_registered_output_reuses_pending_plan_after_cross_call_disconnect(tmp_path):
    contract = _contract(segment_count=3, generation=35)
    root = tmp_path / "chain"
    root.mkdir()
    source_runtime, source, _ = _node(root, "node-b", {"node-a", "node-c"})
    target_runtime, target_coordinator, target_http = _node(root, "node-c", {"node-b"})
    source_ledger = _attach_memory_ledger(source)
    target_ledger = _attach_memory_ledger(target_coordinator)
    source.configure_sidecar_executor(_TargetConsumeExecutor())
    disconnect = {"pending": True}
    transport = Qwen3NetworkHandoffTransport(
        artifact_root=root,
        targets={
            "node-b": Qwen3NetworkTarget(
                node_id="node-b", base_url="http://127.0.0.1:9876", coordinator=source,
            ),
            "node-c": Qwen3NetworkTarget(
                node_id="node-c", base_url="http://127.0.0.1:9877",
                coordinator=target_coordinator,
                requester=_requester(target_http, disconnect_after_patch=disconnect),
            ),
        },
        peer_signers={
            "node-a": Qwen3PeerRequestSigner(SECRET, peer_node_id="node-a"),
            "node-b": Qwen3PeerRequestSigner(SECRET, peer_node_id="node-b"),
        },
        chunk_bytes=5,
        max_attempts=1,
    )
    transport.activate(contract)
    transport.begin_phase("prefill", contract["generation"])
    data = b"ledger-source-input"
    plan = source.begin_receive(
        base_url="http://127.0.0.1:9876", source_peer_id="node-a",
        chain_id=contract["contract_sha256"], generation=contract["generation"],
        phase="prefill", from_segment=0, to_segment=1,
        size_bytes=len(data), sha256=hashlib.sha256(data).hexdigest(),
    )
    source_runtime.receiver.write(
        plan["transfer_id"], ticket=plan["ticket"], authenticated_peer_id="node-a",
        offset=0, data=data,
    )
    source_runtime.receiver.commit(
        plan["transfer_id"], ticket=plan["ticket"], authenticated_peer_id="node-a",
    )
    source.commit_reference(plan["transfer_id"])
    consumed = source.consume(
        plan["transfer_id"], phase="prefill", generation=contract["generation"],
        batch_size=1, sequence_length=4, dtype="float32", device="cpu",
        has_next_segment=True,
    )
    reference = consumed["output_reference"]
    fields = {
        "output_reference": reference,
        "chain_id": contract["contract_sha256"], "generation": contract["generation"],
        "phase": "prefill", "from_segment": 1, "to_segment": 2,
        "source_node_id": "node-b", "target_node_id": "node-c",
    }
    with pytest.raises(Qwen3NetworkError) as interrupted:
        transport.transfer_registered_output(**fields)
    assert interrupted.value.reason_code == "qwen3_transfer_connection_failed"
    next_transfer_id = source_ledger["value"]["outputs"][reference["artifact_id"]]["next_transfer_id"]
    assert next_transfer_id.startswith("qtx_")

    moved = transport.transfer_registered_output(**fields)
    assert moved["transfer_id"] == next_transfer_id
    assert target_runtime.receiver.artifact_path(next_transfer_id).read_bytes() == b"target-sidecar-output"
    source_record = source_ledger["value"]["outputs"][reference["artifact_id"]]
    assert source_record["status"] == "committed"
    assert source_record["confirmed_offset"] == reference["size_bytes"]
    assert target_ledger["value"]["transfers"][next_transfer_id]["status"] == "committed"
    released = transport.release_registered_output(reference)
    assert released["status"] == "released"
    assert source_ledger["value"]["outputs"][reference["artifact_id"]]["status"] == "released"


def test_loopback_output_identity_survives_phase_completion(tmp_path, monkeypatch):
    contract = _contract(segment_count=3, generation=37)
    client = Qwen3LoopbackNetworkControlClient(
        node_id="node-b",
        base_url="http://127.0.0.1:9876",
        artifact_root=tmp_path,
        signer=Qwen3PeerRequestSigner(SECRET, peer_node_id="node-c"),
        requester=lambda *_args: None,
    )
    client._contract = contract
    reference = {
        "schema_version": 1,
        "mode": "network",
        "artifact_id": "qout_" + "e" * 32,
        "source_node_id": "node-b",
        "target_node_id": "node-c",
        "chain_id": contract["contract_sha256"],
        "generation": contract["generation"],
        "phase": "prefill",
        "from_segment": 1,
        "to_segment": 2,
        "size_bytes": 16,
        "sha256": "d" * 64,
        "status": "committed",
        "full_model_materialized": False,
    }
    calls = []
    monkeypatch.setattr(
        client, "_call",
        lambda action, payload: calls.append((action, dict(payload))) or {"status": "released"},
    )
    client.bind_output_reference(reference)
    client._phase = "decoded"

    client.release_output_reference(reference["artifact_id"])

    assert calls == [(
        "release-output",
        {
            "chain_id": contract["contract_sha256"],
            "generation": contract["generation"],
            "phase": "prefill",
            "output_id": reference["artifact_id"],
        },
    )]


def test_target_consume_failure_and_epoch_change_reclaim_sidecar_artifacts(tmp_path):
    coordinator, contract, transfer_id = _committed_target_transfer(tmp_path)
    executor = _TargetConsumeExecutor(fail=True)
    with pytest.raises(Qwen3NetworkError) as failed:
        coordinator.consume(
            transfer_id,
            phase="prefill",
            generation=contract["generation"],
            batch_size=1,
            sequence_length=4,
            dtype="float32",
            device="cpu",
            has_next_segment=True,
            executor=executor,
        )
    assert failed.value.reason_code == "qwen3_network_execution_failed"
    assert coordinator.snapshot()["transfer_count"] == 0
    assert executor.cleaned == ["consume_failed"]
    assert all(not path.exists() for path in executor.outputs)

    coordinator, contract, transfer_id = _committed_target_transfer(tmp_path / "epoch")
    executor = _TargetConsumeExecutor()
    result = coordinator.consume(
        transfer_id,
        phase="prefill",
        generation=contract["generation"],
        batch_size=1,
        sequence_length=4,
        dtype="float32",
        device="cpu",
        has_next_segment=True,
        executor=executor,
    )
    assert result["output_reference"]
    coordinator.authorize_control_peer("node-a", contract=contract, peer_epoch=2)
    assert coordinator.snapshot()["transfer_count"] == 0
    assert coordinator.snapshot()["output_count"] == 0
    assert executor.cleaned == ["peer_epoch_changed"]
    assert all(not path.exists() for path in executor.outputs)


def test_transfer_and_consume_keeps_target_path_out_of_handoff_contract(tmp_path):
    contract = _contract(segment_count=2, generation=23)
    root = tmp_path / "source"
    root.mkdir()
    source = root / "source.pt"
    source.write_bytes(b"source-network-artifact")
    runtime, target_coordinator, target_http = _node(
        tmp_path, "node-b", {"node-a"},
    )
    executor = _TargetConsumeExecutor()
    target_coordinator.configure_sidecar_executor(executor)
    target = Qwen3NetworkTarget(
        node_id="node-b",
        base_url="http://127.0.0.1:9876",
        coordinator=target_coordinator,
        requester=_requester(target_http),
    )
    transport = Qwen3NetworkHandoffTransport(
        artifact_root=root,
        targets={"node-b": target},
        peer_signers={"node-a": Qwen3PeerRequestSigner(SECRET, peer_node_id="node-a")},
        chunk_bytes=5,
        target_execution=True,
    )
    transport.activate(contract)
    transport.begin_phase("prefill", contract["generation"])
    result = transport.transfer_and_consume(
        source_path=source,
        chain_id=contract["contract_sha256"],
        generation=contract["generation"],
        phase="prefill",
        from_segment=0,
        to_segment=1,
        source_node_id="node-a",
        target_node_id="node-b",
        batch_size=1,
        sequence_length=4,
        dtype="float32",
        device="cpu",
        has_next_segment=False,
    )
    encoded = json.dumps(result, ensure_ascii=True)
    assert "path" not in encoded.lower()
    assert result["input_reference"]["target_node_id"] == "node-b"
    assert result["output_reference"] is None
    assert executor.calls == 1
    transport.cleanup()
    assert executor.cleaned == ["release"]
