"""Supply plans create missing segments without claiming loaded executors."""
import copy
import json
import threading
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

from model_supply import ModelSupplyCoordinator, ModelSupplyError, plan_segment_supply
from task_worker_adapter import TaskWorkerControlPlane


def descriptor(layers=8, layer_bytes=10, components=4):
    return {"model_id": "fixture", "model_type": "qwen2", "total_layers": layers,
            "hidden_size": 16, "model_sha256": "a" * 64, "source_model_sha256": "a" * 64,
            "tokenizer_sha256": "b" * 64, "pipeline_runtime_supported": True,
            "layer_weight_bytes": [layer_bytes] * layers,
            "component_weight_bytes": {"embedding": components, "final_norm": components,
                                       "lm_head": components, "other": 0}}


def node(node_id, capacity=80):
    return {"node_id": node_id, "role": "master" if node_id == "master" else "client",
            "capacity_bytes": capacity, "reserve_bytes": 1, "runtime_multiplier": 1.0,
            "score": 100 if node_id == "master" else 50, "engine": "llama_cpp"}


def test_target_plan_uses_aligned_missing_ranges_without_mutating_execution_inventory():
    records = [node("master"), node("middle"), node("tail")]
    records[1]["layer_ranges"] = [[2, 6]]
    records[2]["layer_ranges"] = [[6, 8]]
    before = copy.deepcopy(records)
    plan = plan_segment_supply({"llama_cpp": descriptor()}, records, master_engine="llama_cpp", alignment=2,
                               prefer_all_workers=True)
    assert plan["admitted"]
    assert len(plan["assignments"]) == 3
    assert plan["assignments"][0]["node_id"] == "master"
    assert plan["assignments"][0]["start_layer"] == 0
    assert plan["assignments"][-1]["end_layer"] == 8
    assert all(item["start_layer"] % 2 == item["end_layer"] % 2 == 0 for item in plan["assignments"])
    assert all(item["layers_count"] < 8 for item in plan["assignments"])
    assert not plan["loaded_ready"] and plan["supply_only"]
    assert records == before


def test_non_block_gguf_tensors_are_charged_on_every_executor():
    # Layer weights alone fit. Duplicated GGUF boundary tensors do not.
    plan = plan_segment_supply({"llama_cpp": descriptor(layers=4, components=20)},
                              [node("master", 55), node("tail", 55)], master_engine="llama_cpp")
    assert not plan["admitted"]


def test_single_node_full_model_supply_is_rejected():
    plan = plan_segment_supply({"llama_cpp": descriptor()}, [node("master", 1000)], master_engine="llama_cpp")
    assert not plan["admitted"]


@pytest.mark.parametrize("alignment", [0, 3, True])
def test_invalid_cut_alignment_is_named(alignment):
    with pytest.raises(ModelSupplyError, match="alignment") as exc:
        plan_segment_supply({"llama_cpp": descriptor()}, [node("master"), node("tail")],
                            master_engine="llama_cpp", alignment=alignment)
    assert exc.value.code == "SUPPLY_ALIGNMENT_INVALID"


def test_mixed_engines_are_charged_using_their_own_tensor_sizes():
    descriptors = {"llama_cpp": descriptor(), "pytorch": descriptor(layer_bytes=40)}
    records = [dict(node("master", 200), engine="pytorch"), node("middle", 90), node("tail", 90)]
    plan = plan_segment_supply(descriptors, records, master_engine="pytorch", alignment=2)
    assert plan["admitted"]
    first = plan["assignments"][0]
    assert first["engine"] == "pytorch" and first["required_bytes"] <= 200
    assert first["end_layer"] <= 4


class Server:
    def __init__(self):
        self.clients = {"worker": {"connected_at": 1.0, "registration_epoch": 1, "registration_confirmed": True}}
        self.sent = []
    def get_client_info(self, node_id):
        return self.clients.get(node_id)
    def get_authenticated_peer_epoch(self, node_id):
        info = self.get_client_info(node_id)
        return info.get("registration_epoch") if info and info.get("registration_confirmed") is True else None
    def send_to_client(self, node_id, payload, message_type):
        self.sent.append((node_id, payload, message_type))


def worker_hello(scheduler, node_id, *, acknowledge=True):
    # The transport fake uses the actual control/ACK owner, even before it has
    # any model or layer-range capabilities. Supply must install those first.
    control = scheduler._task_worker_control
    generations = getattr(scheduler, "_hello_test_generations", {})
    generation = generations.get(node_id, 0) + 1
    generations[node_id] = generation
    scheduler._hello_test_generations = generations
    scheduler._tcp_server.clients[node_id]["registration_epoch"] = generation
    control.mark_worker_connection_pending(node_id, connection_generation=generation)
    worker = TaskWorkerControlPlane()
    hello = worker.begin_worker_hello(node_id=node_id,
        capabilities={"stage_types": ["layer_forward"], "engines": ["llama_cpp"], "models": [], "max_concurrency": 1})
    ack = control.receive_on_coordinator(node_id, hello.snapshot(), coordinator_node_id="master")
    assert ack.payload["accepted"] is True
    if acknowledge:
        assert worker.receive_on_worker(ack.snapshot()).payload["accepted"] is True
        control.resolve_worker_connection_pending(node_id)
    return worker, ack


def coordinator(tmp_path):
    scheduler = SimpleNamespace(_effective_role=lambda: "master", get_distributed_inference_enabled=lambda: True,
                                _tcp_server=Server(), _task_worker_control=TaskWorkerControlPlane(),
                                _model_supply_worker=SimpleNamespace(cancel=lambda op: None))
    worker_hello(scheduler, "worker")
    c = ModelSupplyCoordinator(scheduler, tmp_path)
    op_id = "supply_" + "a" * 32
    c._active = op_id
    c._operations[op_id] = {"operation_id": op_id, "intent": {"model_id": "fixture"},
                            "status": "supplying", "nodes": {"worker": {"status": "supplying"}}}
    c._events[op_id] = threading.Event()
    c._connections[(op_id, "worker")] = 1.0
    file = tmp_path / "weights.gguf"
    file.write_bytes(b"verified")
    c._leases[(op_id, "worker")] = {"token": "secret", "files": {"weights.gguf": str(file)}}
    return c, op_id, file


def test_supply_file_authorization_is_assignment_scoped(tmp_path):
    c, op_id, file = coordinator(tmp_path)
    assert c.authorize_file(op_id, "worker", "weights.gguf", "secret") == file
    for node_id, filename, token in [("other", "weights.gguf", "secret"),
                                      ("worker", "../weights.gguf", "secret"),
                                      ("worker", "weights.gguf", "wrong")]:
        with pytest.raises(ModelSupplyError):
            c.authorize_file(op_id, node_id, filename, token)


def test_connection_replacement_revokes_transfer_and_late_receipt(tmp_path):
    c, op_id, _ = coordinator(tmp_path)
    c.scheduler._tcp_server.clients["worker"]["connected_at"] = 2.0
    with pytest.raises(ModelSupplyError):
        c.authorize_file(op_id, "worker", "weights.gguf", "secret")
    c.handle("worker", {"action": "result", "operation_id": op_id, "node_id": "worker", "status": "available"})
    assert not c._responses


def test_cancel_revokes_open_file_leases(tmp_path):
    c, op_id, _ = coordinator(tmp_path)
    result = c.cancel(op_id)
    assert result["status"] == "cancelled"
    assert not c._leases
    assert c.scheduler._tcp_server.sent[0][1]["action"] == "cancel"
    with pytest.raises(ModelSupplyError):
        c.authorize_file(op_id, "worker", "weights.gguf", "secret")


def test_master_restart_does_not_replay_supply(tmp_path):
    op_id = "supply_" + "b" * 32
    path = tmp_path / op_id
    path.mkdir()
    (path / "operation.json").write_text(json.dumps({"operation_id": op_id, "status": "supplying"}))
    c, _, _ = coordinator(tmp_path)
    result = c.get(op_id)
    assert result["status"] == "failed" and result["reason_code"] == "SUPPLY_MASTER_RESTARTED"


def test_inventory_cannot_impersonate_another_node(tmp_path):
    c, op_id, _ = coordinator(tmp_path)
    c.handle("worker", {"action": "inventory_result", "operation_id": op_id, "node_id": "other"})
    assert not c._responses


def test_supply_intent_stays_behind_model_api_boundary(monkeypatch):
    import api_server
    from fastapi.testclient import TestClient
    called = []
    monkeypatch.setattr(api_server.scheduler._model_supply, "create", lambda value: called.append(value))
    response = TestClient(api_server.app, client=("192.0.2.50", 1234)).post("/api/models/supply", json={"model_id": "fixture"})
    assert response.status_code == 403
    assert called == []


def test_supply_token_does_not_authorize_other_model_routes(monkeypatch):
    import api_server
    from fastapi.testclient import TestClient
    response = TestClient(api_server.app, client=("192.0.2.50", 1234)).get(
        "/api/models/current", headers={"X-QLH-Supply-Token": "secret"})
    assert response.status_code == 403


def test_assigned_range_file_endpoint_requires_lease_and_supports_resume(tmp_path, monkeypatch):
    import api_server
    from fastapi.testclient import TestClient
    file = tmp_path / "weights.gguf"
    file.write_bytes(b"0123456789")
    def authorize(op_id, node_id, filename, token):
        if token != "secret" or node_id != "worker" or filename != "weights.gguf":
            raise ModelSupplyError("SUPPLY_FILE_FORBIDDEN", "Invalid file lease")
        return file
    monkeypatch.setattr(api_server.scheduler._model_supply, "authorize_file", authorize)
    monkeypatch.setattr(api_server.auth_service, "auth_required", lambda: True)
    client = TestClient(api_server.app, client=("192.0.2.50", 1234))
    url = "/api/models/supply/supply_" + "a" * 32 + "/files/worker/weights.gguf"
    assert client.get(url).status_code == 403
    response = client.get(url, headers={"X-QLH-Supply-Token": "secret", "Range": "bytes=4-"})
    assert response.status_code == 206 and response.content == b"456789"
    assert response.headers["content-range"] == "bytes 4-9/10"
    assert client.get(url, headers={"X-QLH-Supply-Token": "secret", "Range": "bytes=10-"}).status_code == 416


def test_cancelled_operation_cannot_be_published_by_background_update(tmp_path):
    c, op_id, _ = coordinator(tmp_path)
    c.cancel(op_id)
    c._update(op_id, status="available", nodes={"worker": {"status": "available"}})
    assert c.get(op_id)["status"] == "cancelled"
    c._update(op_id, status="building")
    assert c.get(op_id)["status"] == "cancelled"
    with pytest.raises(ModelSupplyError) as exc:
        c.prepared_source(op_id)
    assert exc.value.code == "SUPPLY_ASSETS_NOT_AVAILABLE"


def test_fp32_request_is_rejected_before_source_work(tmp_path):
    c, _, _ = coordinator(tmp_path)
    with pytest.raises(ModelSupplyError) as exc:
        c.create({"model_id": "fixture", "quant_type": "fp32"})
    assert exc.value.code == "SUPPLY_RUNTIME_PRECISION_UNSUPPORTED"


def test_real_supplied_safetensors_pack_enters_partial_descriptor(tmp_path):
    import importlib.util
    spec = importlib.util.spec_from_file_location("supply_fixture", Path(__file__).with_name("test_model_supply_assets.py"))
    fixture = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(fixture)
    _safetensors_fixture, _kwargs = fixture._safetensors_fixture, fixture._kwargs
    from model_supply_assets import build_safetensors_segment
    from pipeline_model_descriptor import inspect_pipeline_model
    root = tmp_path / "source"
    _safetensors_fixture(root)
    manifest = build_safetensors_segment(root, tmp_path / "packs", **_kwargs(root, start=0, end=2))
    directory = Path(next(iter(manifest["local_files"].values()))).parent
    descriptor = inspect_pipeline_model(directory, model_id="fixture")
    assert descriptor["assignment_layer_range"] == [0, 2]
    assert descriptor["source_model_sha256"] == manifest["source_model_sha256"]
    assert descriptor["tokenizer_sha256"] == manifest["tokenizer_sha256"]
    assert descriptor["layer_weight_bytes"] == manifest["layer_weight_bytes"]


def test_progress_reports_transfer_and_reuse_separately(tmp_path):
    c, op_id, _ = coordinator(tmp_path)
    c.handle("worker", {"action": "progress", "operation_id": op_id, "node_id": "worker",
                        "downloaded_bytes": 90, "total_bytes": 100,
                        "actual_downloaded_bytes": 20, "reused_bytes": 60, "resumed_bytes": 10,
                        "phase": "verifying", "token": "must-not-publish"})
    public = c.get(op_id)["nodes"]["worker"]
    assert public["downloaded_bytes"] == 90
    assert public["actual_downloaded_bytes"] == 20
    assert public["reused_bytes"] == 60 and public["resumed_bytes"] == 10
    assert "token" not in public
    c.handle("worker", {"action": "progress", "operation_id": op_id, "node_id": "worker",
                        "actual_downloaded_bytes": -1, "total_bytes": True, "phase": "secret"})
    assert c.get(op_id)["nodes"]["worker"] == public


@pytest.mark.parametrize("disk_free_bytes", [256 * 1024**2, 32 * 1024**2])
def test_master_dispatches_all_receivers_before_waiting_for_receipts(tmp_path, monkeypatch, disk_free_bytes):
    import model_supply as module
    import model_supply_assets as assets
    c, op_id, _ = coordinator(tmp_path)
    c.scheduler._nodes_lock = threading.Lock()
    c.scheduler.nodes = {node_id: SimpleNamespace(is_available=lambda: True, is_heartbeat_fresh=lambda *args: True)
                         for node_id in ("worker", "tail")}
    c.scheduler._tcp_server.clients["tail"] = {"connected_at": 2.0, "registration_epoch": 2, "registration_confirmed": True}
    worker_hello(c.scheduler, "tail")
    c._operations[op_id]["intent"].update(engine="llama_cpp", strategy="distribute", node_ids=["worker", "tail"])
    c.scheduler._get_pipeline_capacity_nodes = lambda *args: []
    c.scheduler._model_supply_worker.inventory = lambda **kw: {"engines": ["llama_cpp"], "distributed": True,
        "disk_free_bytes": 10**12, "memory_available_bytes": 10**12, "sources": []}
    c.scheduler._model_supply_worker.activate_local = lambda manifest, directory, **kw: {"status": "available", "model_path": str(directory)}
    desc = {"layer_weight_bytes": [1]*6, "component_weight_bytes": {"layers": 10**12}}
    monkeypatch.setattr(module.shutil, "disk_usage", lambda path: SimpleNamespace(free=disk_free_bytes))
    c._sources = lambda *args, **kw: {"descriptors": {"llama_cpp": desc}, "gguf": tmp_path / "source.gguf", "hf": None,
        "master_engine": "llama_cpp", "alignment": 1, "source_model_id": "fixture", "source_model_sha256": "a"*64,
        "tokenizer_sha256": "b"*64, "conversion_proof": None}
    assignments = [{"node_id": node_id, "start_layer": start, "end_layer": start+2, "raw_weight_bytes": 2, "engine": "llama_cpp"}
                   for node_id, start in (("master",0),("worker",2),("tail",4))]
    monkeypatch.setattr(module, "plan_segment_supply", lambda *args, **kw: {"admitted": True, "assignments": assignments, "total_layers": 6})
    manifests = {}
    def build(source, directory, **kw):
        assert callable(kw["cancelled"]) and not kw["cancelled"]()
        assert time.monotonic() < kw["deadline"] <= time.monotonic() + c._operations[op_id]["intent"].get("timeout_seconds", 3600)
        file = directory / "segment.gguf"
        file.write_bytes(b"segment")
        manifest = {"operation_id":op_id, "node_id":kw["node_id"], "layer_range":[kw["start_layer"],kw["end_layer"]],
                    "segment_mode":"head" if kw["start_layer"] == 0 else "tail" if kw["end_layer"] == 6 else "middle",
                    "manifest_sha256":"c"*64, "artifact_sha256":"d"*64, "source_model_sha256":"a"*64,
                    "files":[{"path":"segment.gguf", "size_bytes":7}], "local_files":{"segment.gguf":str(file)}}
        manifests[kw["node_id"]] = dict(manifest)
        return manifest
    monkeypatch.setattr(assets, "build_gguf_segment", build)
    monkeypatch.setattr(assets, "gguf_segment_disk_weight_bytes", lambda source, start, end: end - start)
    def wait(operation_id, node_id, action, timeout):
        if action == "inventory_result":
            return c.scheduler._model_supply_worker.inventory()
        ensured = [item[0] for item in c.scheduler._tcp_server.sent if item[1]["action"] == "ensure"]
        assert ensured == ["worker", "tail"]
        return {**manifests[node_id], "status":"available", "actual_downloaded_bytes":7, "reused_bytes":0, "resumed_bytes":0}
    c._wait = wait
    if disk_free_bytes < 64 * 1024**2:
        with pytest.raises(ModelSupplyError) as exc:
            c._execute(op_id)
        assert exc.value.code == "SUPPLY_MASTER_DISK_INSUFFICIENT"
        assert manifests == {}
        return
    c._execute(op_id)
    result = c.get(op_id)
    assert result["status"] == "available" and result["loaded_ready"] is False
    assert result["nodes"]["tail"]["actual_downloaded_bytes"] == 7


def test_failed_supply_cancels_receivers_and_revokes_file_leases(tmp_path):
    c, op_id, _ = coordinator(tmp_path)
    cancelled = []
    c.scheduler._model_supply_worker.cancel = cancelled.append
    c._execute = lambda operation_id: (_ for _ in ()).throw(ModelSupplyError("SUPPLY_WORKER_FAILED", "Worker failed"))
    c._run(op_id)
    assert c.get(op_id)["status"] == "failed"
    assert not c._leases and cancelled == [op_id]
    assert c.scheduler._tcp_server.sent[0][1]["action"] == "cancel"


def test_tail_failure_ends_wait_for_slow_first_receiver(tmp_path):
    c, op_id, _ = coordinator(tmp_path)
    c.scheduler._tcp_server.clients["tail"] = {"connected_at":2.0}
    c._connections[(op_id,"tail")] = 2.0
    c._operations[op_id]["nodes"]["tail"] = {"status":"supplying"}
    c.handle("tail", {"action":"result", "operation_id":op_id, "node_id":"tail", "status":"failed", "reason_code":"SUPPLY_DIGEST_MISMATCH"})
    with pytest.raises(ModelSupplyError) as error:
        c._wait(op_id,"worker","result",3600)
    assert error.value.code == "SUPPLY_DIGEST_MISMATCH"


def test_cancel_boundary_rejects_later_ensure(tmp_path):
    c, op_id, _ = coordinator(tmp_path)
    c.cancel(op_id)
    with pytest.raises(ModelSupplyError):
        c._send(op_id,"worker",{"action":"ensure"})
    assert [item[1]["action"] for item in c.scheduler._tcp_server.sent] == ["cancel"]


def test_parallel_receiver_connection_replacement_ends_wait(tmp_path):
    c, op_id, _ = coordinator(tmp_path)
    c.scheduler._tcp_server.clients["tail"] = {"connected_at":3.0}
    c._connections[(op_id,"tail")] = 2.0
    with pytest.raises(ModelSupplyError) as error:
        c._wait(op_id,"worker","result",3600)
    assert error.value.code == "SUPPLY_WORKER_DISCONNECTED"


def test_prepare_guard_prevents_new_intent_until_transaction_entry(tmp_path):
    c, op_id, _ = coordinator(tmp_path)
    c.scheduler._task_worker_stage_lock = threading.RLock()
    c.scheduler._task_worker_active_attempts = {}
    c.scheduler._active_pipeline_task_ids = set()
    c.prepared_source = lambda operation_id: {"operation_id":operation_id}
    with c.prepare_guard(op_id) as source:
        assert source["operation_id"] == op_id
        with pytest.raises(ModelSupplyError) as error:
            c.create({"model_id":"new-fixture"})
        assert error.value.code == "SUPPLY_PREPARE_BUSY"
    assert c._preparing == ""
    c.scheduler._pipeline_load_transaction = {"phase":"preparing"}
    with pytest.raises(ModelSupplyError) as error:
        c.create({"model_id":"new-fixture"})
    assert error.value.code == "SUPPLY_PREPARE_BUSY"


def test_failed_prepare_releases_guard(tmp_path):
    c, op_id, _ = coordinator(tmp_path)
    c.prepared_source = lambda operation_id: (_ for _ in ()).throw(ModelSupplyError("SUPPLY_OPERATION_STALE", "changed"))
    with pytest.raises(ModelSupplyError):
        with c.prepare_guard(op_id):
            pytest.fail("stale preparation must not enter the transaction")
    assert c._preparing == ""


def test_runtime_projection_is_applied_per_node_without_changing_disk_descriptor():
    sources = {"pytorch": descriptor(layers=4, layer_bytes=1024**2, components=0)}
    original = copy.deepcopy(sources)
    peak = descriptor(layers=4, layer_bytes=5*1024**2, components=0)
    records = [dict(node("master", 3*1024**2),engine="pytorch"),
               dict(node("tail", 3*1024**2),engine="pytorch")]
    plan = plan_segment_supply(sources, records, master_engine="pytorch", safety_margin=1.0,
                               runtime_descriptors={"master":peak,"tail":peak})
    assert not plan["admitted"]
    assert sources == original


def test_pytorch_head_does_not_reserve_unloaded_output_projection():
    pytorch = descriptor(layers=4,layer_bytes=10,components=0)
    pytorch["component_weight_bytes"] = {"embedding":10,"final_norm":2,"lm_head":1000,"other":0}
    gguf = descriptor(layers=4,layer_bytes=2,components=2)
    records = [dict(node("master",70),engine="pytorch"),node("tail",100)]
    plan = plan_segment_supply({"pytorch":pytorch,"llama_cpp":gguf},records,master_engine="pytorch",safety_margin=1.0)
    assert plan["admitted"]
    head = plan["assignments"][0]
    assert head["has_embedding"] and not head["has_lm_head"]
    assert head["reserve_bytes"] == 1+10+2
    assert head["required_bytes"] <= 70


def test_existing_range_preference_is_grouped_without_imposing_old_inventory():
    records=[node("master",100),node("middle",100),node("tail",100)]
    records[1]["layer_ranges"]=[[4,6]]
    plan=plan_segment_supply({"llama_cpp":descriptor(layers=8)},records,master_engine="llama_cpp",alignment=2,
                             preferred_ranges={"tail":[[6,8]]}, prefer_all_workers=True)
    assert plan["admitted"]
    tail=next(item for item in plan["assignments"] if item["node_id"]=="tail")
    assert [tail["start_layer"],tail["end_layer"]]==[6,8]
    assert tail["layer_ranges"]==[[6,8]]


@pytest.mark.parametrize("preferences",[{"master":[[0,4]]},{"tail":[[1,8]]},{"tail":[[0,8]]},{"tail":[[True,4]]}])
def test_invalid_range_preference_cannot_bypass_alignment_or_full_model_gate(preferences):
    with pytest.raises(ModelSupplyError) as error:
        plan_segment_supply({"llama_cpp":descriptor(layers=8)},[node("master",1000),node("tail",1000)],
                            master_engine="llama_cpp",alignment=2,preferred_ranges=preferences)
    assert error.value.code=="SUPPLY_PREFERRED_RANGE_INVALID"


@pytest.mark.parametrize("engine,start,end,tied,expected", [
    ("pytorch", 0, 2, False, 42),
    ("pytorch", 2, 4, False, 22),
    ("pytorch", 4, 6, False, 59),
    ("pytorch", 4, 6, True, 79),
])
def test_segment_disk_budget_selects_source_tensors_without_counting_decoder_twice(engine, start, end, tied, expected):
    from model_supply import _segment_disk_weight_bytes
    source = descriptor(layers=6, layer_bytes=10)
    source.update(tie_word_embeddings=tied, component_weight_bytes={
        "layers": 10**12, "embedding": 20, "final_norm": 2, "lm_head": 30,
        "other": 7, "visual": 100, "mtp": 40, "multimodal": 49,
    })
    before = copy.deepcopy(source)
    assert _segment_disk_weight_bytes(source, {"engine": engine, "start_layer": start, "end_layer": end}) == expected
    assert source == before


@pytest.mark.parametrize("entered_guard", [False, True])
def test_opt_out_revokes_available_prepare_permission_without_deleting_assets(tmp_path, entered_guard):
    c, op, file = coordinator(tmp_path)
    c._operations[op].update(status="available", master_model_path=str(file),
                            plan={"assignments": [{"engine": "llama_cpp"}]})
    c.scheduler._model_supply_worker.selected_model_path = lambda: file
    cancelled = []
    c.scheduler.cancel_worker_prepare_operation = cancelled.append
    if entered_guard:
        with c.prepare_guard(op):
            c.cancel_active()
            with pytest.raises(ModelSupplyError) as error:
                c.ensure_prepare_not_cancelled(op)
            assert error.value.code == "SUPPLY_PREPARE_CANCELLED"
    else:
        c.cancel_active()
    assert c.get(op)["status"] == "available" and file.read_bytes() == b"verified"
    assert cancelled == [op] and c._events[op].is_set()
    # Rejoining cannot reset this operation's cancellation event.
    with pytest.raises(ModelSupplyError):
        with c.prepare_guard(op):
            pytest.fail("opted-out operation regained prepare permission")
    with pytest.raises(ModelSupplyError) as error:
        c.ensure_prepare_not_cancelled(op)
    assert error.value.code == "SUPPLY_PREPARE_CANCELLED"


def test_cancelled_guard_fences_late_native_thread_after_guard_exit(tmp_path):
    c, op, file = coordinator(tmp_path)
    c._operations[op].update(status="available", master_model_path=str(file),
                            plan={"assignments": [{"engine": "llama_cpp"}]})
    c.scheduler._model_supply_worker.selected_model_path = lambda: file
    with c.prepare_guard(op):
        c.cancel(op)
    assert not c._preparing and not c._prepare_cancelled
    with pytest.raises(ModelSupplyError) as error:
        c.ensure_prepare_not_cancelled(op)
    assert error.value.code == "SUPPLY_PREPARE_CANCELLED"


def test_progress_accepts_android_downloading_phase_without_opening_allowlist(tmp_path):
    c, op, _ = coordinator(tmp_path)
    c.handle("worker", {"action": "progress", "operation_id": op, "node_id": "worker",
                        "phase": "downloading", "downloaded_bytes": 7, "total_bytes": 10,
                        "token": "private", "arbitrary_field": "must-not-publish"})
    projected = c.get(op)["nodes"]["worker"]
    assert projected["phase"] == "downloading" and projected["downloaded_bytes"] == 7
    assert "token" not in projected and "arbitrary_field" not in projected
    c.handle("worker", {"action": "progress", "operation_id": op, "node_id": "worker", "phase": "arbitrary-phase"})
    assert c.get(op)["nodes"]["worker"] == projected



@pytest.mark.parametrize("via_opt_out", [False, True])
def test_cancel_persist_failure_still_revokes_native_and_remote_receivers(tmp_path, monkeypatch, caplog, via_opt_out):
    from worker_prepare_coordinator import WorkerPrepareCoordinatorMixin
    c, op, file = coordinator(tmp_path)
    original_persist = c._persist
    remote = []
    native = []
    c.scheduler._layer_config_lock = threading.RLock()
    c.scheduler._pipeline_load_transaction = {
        "config_id": "fixture-config", "phase": "preparing",
        "expected": {"worker": {"worker_prepare_identity": {"operation_id": op}}},
    }
    def native_abort(config_id, reason_code, reason):
        assert not c._condition._is_owned(), "cancel took native lock under supply condition"
        native.append((config_id, reason_code))
        c.scheduler._pipeline_load_transaction["phase"] = "aborted"
    c.scheduler._abort_pipeline_load_transaction = native_abort
    c.scheduler.cancel_worker_prepare_operation = lambda operation_id: WorkerPrepareCoordinatorMixin.cancel_worker_prepare_operation(c.scheduler, operation_id)
    def cancel_receiver(operation_id):
        assert not c._condition._is_owned(), "receiver cleanup ran under supply condition"
        remote.append(operation_id)
    c.scheduler._model_supply_worker.cancel = cancel_receiver
    disk_error = OSError("injected cancel persistence failure")
    def broken_persist(operation):
        assert c._events[op].is_set() and not c._leases
        raise disk_error
    monkeypatch.setattr(c, "_persist", broken_persist)
    with pytest.raises(OSError) as error:
        c.cancel_active() if via_opt_out else c.cancel(op)
    assert error.value is disk_error
    assert c.get(op)["status"] == "cancelled" and c._events[op].is_set()
    assert not c._leases and file.read_bytes() == b"verified"
    assert native == [("fixture-config", "pipeline_prepare_cancelled")]
    assert remote == [op]
    assert c.scheduler._tcp_server.sent[0][1]["action"] == "cancel"
    assert "stage=persist reason_code=SUPPLY_CANCEL_FAILED" in caplog.text
    monkeypatch.setattr(c, "_persist", original_persist)
    c.cancel_active() if via_opt_out else c.cancel(op)  # Retry save and cleanup.
    saved = json.loads((tmp_path / op / "operation.json").read_text(encoding="utf-8"))
    assert saved["status"] == "cancelled" and remote == [op, op]


def test_supply_cancel_preserves_first_error_and_attempts_every_cleanup(tmp_path, monkeypatch, caplog):
    c, op, _ = coordinator(tmp_path)
    first = OSError("disk failed first")
    stages = []
    def fail_save(operation):
        raise first
    def fail_native(operation_id):
        assert not c._condition._is_owned()
        stages.append("native")
        raise RuntimeError("native cleanup also failed")
    def fail_receiver(operation_id):
        assert not c._condition._is_owned()
        stages.append("receiver")
        raise RuntimeError("receiver cleanup also failed")
    monkeypatch.setattr(c, "_persist", fail_save)
    c.scheduler.cancel_worker_prepare_operation = fail_native
    c.scheduler._model_supply_worker.cancel = fail_receiver
    with pytest.raises(OSError) as error:
        c.cancel(op)
    assert error.value is first and stages == ["native", "receiver"]
    assert c._events[op].is_set() and not c._leases
    assert all("stage=" + stage in caplog.text for stage in ("persist", "native", "receivers"))


@pytest.fixture
def inventory_case(tmp_path, monkeypatch):
    """Only inventory I/O is faked; control receipts and supply waits are real."""
    c, op, _ = coordinator(tmp_path)
    scheduler = c.scheduler
    scheduler._nodes_lock = threading.RLock()
    scheduler.nodes = {"worker": SimpleNamespace(is_available=lambda: True, is_heartbeat_fresh=lambda *args: True)}
    c._operations[op]["intent"].update(timeout_seconds=30)
    waiting, dispatched, sources_reached = threading.Event(), threading.Event(), threading.Event()
    inventories = []
    original_update = c._update
    def update(operation_id, **values):
        original_update(operation_id, **values)
        if values.get("status") == "waiting_nodes":
            waiting.set()
    monkeypatch.setattr(c, "_update", update)
    def master_inventory(**kwargs):
        inventories.append("master")
        return {"distributed": True, "engines": ["llama_cpp"]}
    scheduler._model_supply_worker.inventory = master_inventory
    original_send = scheduler._tcp_server.send_to_client
    def send(node_id, payload, message_type):
        original_send(node_id, payload, message_type)
        if payload.get("action") == "inventory":
            dispatched.set()
            c.handle(node_id, {"action": "inventory_result", "operation_id": op,
                "node_id": node_id, "distributed": True, "engines": ["llama_cpp"]})
    monkeypatch.setattr(scheduler._tcp_server, "send_to_client", send)
    def sources(*args, **kwargs):
        sources_reached.set()
        raise ModelSupplyError("SUPPLY_TEST_SOURCE_BOUNDARY", "Stop before any source or weight I/O")
    monkeypatch.setattr(c, "_sources", sources)
    return SimpleNamespace(c=c, scheduler=scheduler, op=op, waiting=waiting,
        dispatched=dispatched, sources_reached=sources_reached, inventories=inventories)


@pytest.mark.parametrize("requested", [[], ["worker", "tail"]])
@pytest.mark.parametrize("pending", ["hello", "ack"])
def test_startup_waits_for_current_hello_ack_before_automatic_inventory(inventory_case, requested, pending):
    case = inventory_case
    c, scheduler, op = case.c, case.scheduler, case.op
    scheduler._tcp_server.clients["tail"] = {"connected_at": time.time(), "registration_epoch": 2,
                                            "registration_confirmed": True}
    scheduler.nodes["tail"] = scheduler.nodes["worker"]
    scheduler._task_worker_control.mark_worker_connection_pending("tail")
    worker, ack = worker_hello(scheduler, "tail", acknowledge=False) if pending == "ack" else (None, None)
    c._operations[op].update(wait_for_nodes=True)
    c._operations[op]["intent"]["node_ids"] = requested
    runner = threading.Thread(target=c._run, args=(op,))
    runner.start()
    try:
        assert case.waiting.wait(2), "supply did not enter the handshake wait"
        assert c.get(op)["status"] == "waiting_nodes"
        assert not scheduler._tcp_server.sent and not case.inventories
        assert not case.dispatched.is_set() and not case.sources_reached.is_set()
        if pending == "hello":
            worker, ack = worker_hello(scheduler, "tail", acknowledge=False)
        # Accepting hello alone is not the ACK boundary: the original pending
        # owner stays set until the ACK write completes.
        assert scheduler._task_worker_control.worker_snapshot("tail")["manual_stage_dispatch_enabled"]
        assert not c._worker_handshake_ready("tail", scheduler._tcp_server.get_client_info("tail"))
        assert not case.dispatched.is_set()
        assert worker.receive_on_worker(ack.snapshot()).payload["accepted"] is True
        scheduler._task_worker_control.resolve_worker_connection_pending("tail")
        assert case.sources_reached.wait(2), "accepted current-session ACK did not start inventory"
        runner.join(2)
        assert not runner.is_alive()
        assert [item[0] for item in scheduler._tcp_server.sent if item[1].get("action") == "inventory"] == ["worker", "tail"]
        assert c.get(op)["reason_code"] == "SUPPLY_TEST_SOURCE_BOUNDARY"
    finally:
        c._events[op].set()
        runner.join(2)


@pytest.mark.parametrize("requested,code", [([], "SUPPLY_WORKERS_UNAVAILABLE"),
                                            (["worker"], "SUPPLY_REQUESTED_NODE_UNAVAILABLE")])
@pytest.mark.parametrize("proof", ["hello_pending", "ack_pending", "register_pending", "stale_hello", "no_owner"])
def test_default_no_wait_requires_current_handshake_owner_evidence(inventory_case, requested, code, proof):
    case = inventory_case
    c, scheduler, op = case.c, case.scheduler, case.op
    control = scheduler._task_worker_control
    if proof == "hello_pending":
        control.disconnect_worker("worker")
        control.mark_worker_connection_pending(
            "worker", connection_generation=scheduler._tcp_server.clients["worker"]["registration_epoch"] + 1,
        )
    elif proof == "ack_pending":
        worker_hello(scheduler, "worker", acknowledge=False)
    elif proof == "register_pending":
        scheduler._tcp_server.clients["worker"]["registration_confirmed"] = False
    elif proof == "stale_hello":
        # A new TCP object can appear before the post-REGISTER pending callback;
        # its old accepted hello must not qualify the replacement connection.
        scheduler._tcp_server.clients["worker"] = {
            "connected_at": control.worker_snapshot("worker")["hello_received_at"] + 1,
            "registration_epoch": 2, "registration_confirmed": True}
    else:
        del scheduler._task_worker_control
    c._operations[op]["intent"]["node_ids"] = requested
    with pytest.raises(ModelSupplyError) as error:
        c._execute(op)
    assert error.value.code == code
    assert not case.dispatched.is_set() and not case.sources_reached.is_set() and not case.inventories
    assert not scheduler._tcp_server.sent


def test_supply_inventory_rechecks_hello_pending_at_dispatch_boundary(inventory_case):
    case = inventory_case
    epoch = case.scheduler._tcp_server.clients["worker"]["registration_epoch"] + 1
    case.scheduler._task_worker_control.mark_worker_connection_pending("worker", connection_generation=epoch)
    with pytest.raises(ModelSupplyError) as error:
        case.c._send(case.op, "worker", {"action": "inventory", "model_id": "fixture"})
    assert error.value.code == "SUPPLY_WORKERS_UNAVAILABLE"
    assert not case.scheduler._tcp_server.sent


def test_confirmed_worker_without_selected_model_can_enter_inventory(inventory_case):
    case = inventory_case
    snapshot = case.scheduler._task_worker_control.worker_snapshot("worker")
    assert snapshot["manual_stage_dispatch_enabled"]
    assert not snapshot["layer_stage_dispatch_enabled"] and not snapshot["capabilities"]["models"]
    with pytest.raises(ModelSupplyError) as error:
        case.c._execute(case.op)
    assert error.value.code == "SUPPLY_TEST_SOURCE_BOUNDARY"
    assert case.dispatched.is_set() and case.sources_reached.is_set() and case.inventories == ["master"]


@pytest.mark.parametrize("changed_node", ["worker", "tail"])
@pytest.mark.parametrize("buffered", [False, True])
def test_inventory_wait_fails_immediately_when_any_dispatched_session_is_replaced(tmp_path, changed_node, buffered):
    c, op, _ = coordinator(tmp_path)
    server = c.scheduler._tcp_server
    server.clients["tail"] = {"connected_at": 2.0, "registration_epoch": 2, "registration_confirmed": True}
    c._connections[(op, "tail")] = 2.0
    c._operations[op]["nodes"]["tail"] = {"status": "inventory"}
    if buffered:
        c.handle("worker", {"action": "inventory_result", "operation_id": op, "node_id": "worker", "distributed": True})
    # Old receipt queued before reconnect must be rejected before it is popped.
    server.clients[changed_node] = {"connected_at": 3.0, "registration_epoch": 3, "registration_confirmed": True}
    started = time.monotonic()
    with pytest.raises(ModelSupplyError) as error:
        c._wait(op, "worker", "inventory_result", 90)
    assert error.value.code == "SUPPLY_WORKER_DISCONNECTED"
    assert time.monotonic() - started < 1


def test_inventory_wait_wakes_and_fails_on_live_same_id_reconnect(tmp_path):
    c, op, _ = coordinator(tmp_path)
    entered = threading.Event()
    errors = []
    original_check = c._check_receivers
    def check(operation_id):
        original_check(operation_id)
        entered.set()
    c._check_receivers = check
    def wait():
        try:
            c._wait(op, "worker", "inventory_result", 90)
        except ModelSupplyError as exc:
            errors.append(exc.code)
    runner = threading.Thread(target=wait)
    runner.start()
    try:
        assert entered.wait(2)
        c.scheduler._tcp_server.clients["worker"] = {"connected_at": 2.0, "registration_epoch": 2,
                                                     "registration_confirmed": True}
        # No model-supply response or condition notification accompanies reconnect;
        # the existing poll must detect it instead of waiting out the 90s timer.
        runner.join(2)
        assert not runner.is_alive() and errors == ["SUPPLY_WORKER_DISCONNECTED"]
    finally:
        c._events[op].set()
        with c._condition:
            c._condition.notify_all()
        runner.join(2)


@pytest.fixture
def startup_pool_case(tmp_path, monkeypatch):
    """Real supply waits, hello receipts and capacity planning; fake only source/build I/O."""
    import model_supply as module
    import model_supply_assets as assets
    import config
    c, op, _ = coordinator(tmp_path)
    c._connections.clear()
    c._leases.clear()
    c._operations[op].update(wait_for_nodes=True, plan=None, reason_code="", reason="")
    c._operations[op]["intent"].update(engine="llama_cpp", node_ids=[], timeout_seconds=5)
    scheduler = c.scheduler
    scheduler.role, scheduler.enabled = "master", True
    scheduler._effective_role = lambda: scheduler.role
    scheduler.get_distributed_inference_enabled = lambda: scheduler.enabled
    scheduler._nodes_lock = threading.RLock()
    scheduler.nodes = {}
    case = SimpleNamespace(c=c, op=op, scheduler=scheduler, inventory_nodes=[], source_calls=[],
        build_calls=[], capacity_pools=[], inventory_hooks={}, blocked_inventory=set(),
        source_entered=threading.Event(), release_source=threading.Event(), waiting=threading.Event(),
        inventory_sent={}, on_source=None, on_build=None, budgets={}, engines={}, runtime_source=None,
        proof={"source_owner": "verified-conversion"})
    monkeypatch.setattr(config, "PIPELINE_CAPACITY_RESERVE_MB", 0)
    monkeypatch.setattr(config, "PIPELINE_CAPACITY_SAFETY_MARGIN", 1.0)
    monkeypatch.setattr(config, "PIPELINE_PREFER_ALL_WORKERS", True)
    monkeypatch.setattr(module.shutil, "disk_usage", lambda path: SimpleNamespace(free=10**12))

    def join(node_id="android", *, acknowledge=True, available=True, fresh=True, connected=True):
        record = SimpleNamespace(available=available, fresh=fresh)
        record.is_available = lambda: record.available
        record.is_heartbeat_fresh = lambda *args: record.fresh
        with scheduler._nodes_lock:
            scheduler.nodes[node_id] = record
        if connected:
            scheduler._tcp_server.clients[node_id] = {"connected_at": time.time(),
                "registration_epoch": len(scheduler._tcp_server.clients) + 1, "registration_confirmed": True}
            scheduler._task_worker_control.mark_worker_connection_pending(node_id)
            if acknowledge:
                worker_hello(scheduler, node_id)
        case.inventory_sent.setdefault(node_id, threading.Event())
        return record
    case.join = join
    # Preserve the existing worker's accepted control receipt and TCP session.
    join("worker", connected=False)
    case.inventory_sent["master"] = threading.Event()

    def inventory(node_id):
        value = {"distributed": True, "engines": case.engines.get(node_id, ["llama_cpp"]),
                 "memory_available_bytes": case.budgets.get(node_id, 10**4),
                 "disk_free_bytes": 10**12, "sources": []}
        if "pytorch" in value["engines"]:
            value["pytorch_runtime"] = {"execution_device": "cpu", "target_element_size": 2,
                                        "available_bytes": value["memory_available_bytes"]}
        return value
    scheduler._model_supply_worker.inventory = lambda **kwargs: inventory("master")
    scheduler._model_supply_worker.activate_local = lambda manifest, directory, **kwargs: {
        "status": "available", "model_path": str(directory)}
    original_update = c._update
    def update(operation_id, **values):
        original_update(operation_id, **values)
        if values.get("status") == "waiting_nodes":
            case.waiting.set()
    monkeypatch.setattr(c, "_update", update)

    original_send = scheduler._tcp_server.send_to_client
    def send(node_id, payload, message_type):
        original_send(node_id, payload, message_type)
        if payload["action"] == "inventory":
            assert node_id in c.get(op)["nodes"], "new recipient was not published before dispatch"
            case.inventory_nodes.append(node_id)
            if node_id not in case.blocked_inventory:
                c.handle(node_id, {"action": "inventory_result", "operation_id": op, "node_id": node_id,
                                   **inventory(node_id)})
            if node_id in case.inventory_hooks:
                case.inventory_hooks[node_id]()
            case.inventory_sent[node_id].set()
        elif payload["action"] == "ensure":
            c.handle(node_id, {"action": "result", "operation_id": op, "node_id": node_id,
                               **payload["manifest"], "status": "available"})
    monkeypatch.setattr(scheduler._tcp_server, "send_to_client", send)

    def sources(operation_id, intent, *, want_gguf):
        assert not c._condition._is_owned(), "source work held the supply lock"
        case.source_calls.append(want_gguf)
        if len(case.source_calls) == 1:
            case.source_entered.set()
            assert case.release_source.wait(5), "test did not release source inspection"
        if case.on_source:
            case.on_source(want_gguf)
        desc = descriptor(layers=6, layer_bytes=10, components=0)
        engine = intent["engine"]
        descriptors = {engine: desc}
        if want_gguf:
            descriptors["llama_cpp"] = copy.deepcopy(desc)
        return {"descriptors": descriptors, "master_engine": engine, "alignment": 1,
                "gguf": tmp_path / "source.gguf" if want_gguf else None, "hf": case.runtime_source,
                "source_model_id": "fixture", "source_model_sha256": "a" * 64,
                "tokenizer_sha256": "b" * 64, "conversion_proof": case.proof if engine == "pytorch" and want_gguf else None}
    monkeypatch.setattr(c, "_sources", sources)

    def capacity(eligible):
        case.capacity_pools.append(set(eligible))
        return [{**node(node_id, case.budgets.get(node_id, 10**4)), "reserve_bytes": 0}
                for node_id in sorted(eligible)]
    scheduler._get_pipeline_capacity_nodes = capacity
    def build(source, directory, **kwargs):
        assert not c._condition._is_owned(), "builder held the supply lock"
        case.build_calls.append(copy.deepcopy(kwargs))
        if case.on_build:
            case.on_build(kwargs)
        file = directory / "segment.gguf"
        file.write_bytes(b"segment")
        return {"operation_id": op, "node_id": kwargs["node_id"],
                "layer_range": [kwargs["start_layer"], kwargs["end_layer"]],
                "segment_mode": "head" if kwargs["start_layer"] == 0 else "tail" if kwargs["end_layer"] == 6 else "middle",
                "manifest_sha256": "c" * 64, "artifact_sha256": "d" * 64,
                "source_model_sha256": kwargs["source_model_sha256"],
                "files": [{"path": "segment.gguf", "size_bytes": 7}], "local_files": {"segment.gguf": str(file)}}
    monkeypatch.setattr(assets, "build_gguf_segment", build)
    monkeypatch.setattr(assets, "build_safetensors_segment", build)
    monkeypatch.setattr(assets, "gguf_segment_disk_weight_bytes", lambda source, start, end: end - start)
    threads = []
    def start():
        runner = threading.Thread(target=c._run, args=(op,))
        threads.append(runner)
        runner.start()
        assert case.source_entered.wait(2), "initial worker never reached source inspection"
        return runner
    case.start = start
    yield case
    case.release_source.set()
    c._events[op].set()
    with c._condition:
        c._condition.notify_all()
    for runner in threads:
        runner.join(2)
        assert not runner.is_alive(), "test left a supply thread running"


def _assert_startup_available(case, runner, expected):
    runner.join(3)
    assert not runner.is_alive(), case.c.get(case.op)
    result = case.c.get(case.op)
    assert result["status"] == "available", result
    assert result["plan"]["admitted"] and not result["loaded_ready"]
    assert {item["node_id"] for item in result["plan"]["assignments"]} == set(expected)
    assert set(result["nodes"]) == set(expected)
    assert len(case.build_calls) == len(expected)
    return result


@pytest.mark.parametrize("requested,free,expected,reason", [
    (None, 10**12, "local_cut", "SUPPLY_LOCAL_CUT_PREFERRED"),
    ("distribute", 10**12, "distribute", "SUPPLY_DISTRIBUTE_REQUESTED"),
    (None, 1, "distribute", "SUPPLY_LOCAL_CUT_DISK_INSUFFICIENT"),
])
def test_default_local_cut_precedence_and_named_distribution(startup_pool_case, monkeypatch,
                                                             requested, free, expected, reason):
    case = startup_pool_case
    if requested is not None:
        case.c._operations[case.op]["intent"]["strategy"] = requested
    server = case.scheduler._tcp_server
    original = server.send_to_client
    def send(node_id, payload, message_type):
        if payload["action"] == "inventory":
            case.c.handle(node_id, {"action": "inventory_result", "operation_id": case.op,
                "node_id": node_id, "distributed": True, "engines": ["llama_cpp"],
                "memory_available_bytes": 10**4, "disk_free_bytes": free, "local_cut": True,
                "sources": [{"source_model_sha256": "a" * 64, "tokenizer_sha256": "b" * 64}]})
        else:
            original(node_id, payload, message_type)
    monkeypatch.setattr(server, "send_to_client", send)
    observed = []
    def collect(**kwargs):
        assert case.c.get(case.op)["status"] == "available"
        observed.append(kwargs)
        return {"removed": 1, "reclaimed_bytes": 7, "skipped": {}, "examined": 2}
    case.scheduler._model_supply_worker.collect_garbage = collect
    case.release_source.set()
    runner = case.start()
    result = _assert_startup_available(case, runner, {"master", "worker"})
    ensured = next(payload for node_id, payload, _ in server.sent if payload["action"] == "ensure")
    assert ensured["strategy"] == expected and ensured["strategy_reason_code"] == reason
    assert ensured["allow_fallback"] is (requested is None)
    assert 1 <= ensured["local_cut_timeout_seconds"] <= 300
    assert result["nodes"]["worker"]["strategy_reason_code"] == reason
    assert all(call["reuse_root"] == case.c.root for call in case.build_calls)
    assert observed and case.c.root / case.op in observed[0]["protected_paths"]
    assert result["retention"]["removed"] == 1


def test_source_window_new_android_waits_pending_ack_then_enters_three_node_plan(startup_pool_case):
    case = startup_pool_case
    runner = case.start()
    case.join(acknowledge=False)
    case.release_source.set()
    assert case.waiting.wait(2)
    assert case.inventory_nodes == ["worker"] and not case.capacity_pools and not case.build_calls
    worker, ack = worker_hello(case.scheduler, "android", acknowledge=False)
    assert case.scheduler._task_worker_control.worker_snapshot("android")["manual_stage_dispatch_enabled"]
    assert not case.c._worker_handshake_ready("android", case.scheduler._tcp_server.get_client_info("android"))
    assert not case.inventory_sent["android"].is_set()
    assert worker.receive_on_worker(ack.snapshot()).payload["accepted"] is True
    case.scheduler._task_worker_control.resolve_worker_connection_pending("android")
    result = _assert_startup_available(case, runner, {"master", "worker", "android"})
    assert case.inventory_nodes == ["worker", "android"] and case.source_calls == [True]
    assert case.capacity_pools == [{"master", "worker", "android"}]
    assert result["plan"]["assignments"][0]["start_layer"] == 0
    assert result["plan"]["assignments"][-1]["end_layer"] == 6


@pytest.mark.parametrize("wait_for_nodes,selected", [(False, []), (True, ["worker"])])
def test_source_window_expansion_keeps_explicit_and_no_wait_pools_fixed(startup_pool_case, wait_for_nodes, selected):
    case = startup_pool_case
    case.c._operations[case.op].update(wait_for_nodes=wait_for_nodes)
    case.c._operations[case.op]["intent"]["node_ids"] = selected
    runner = case.start()
    case.join(acknowledge=False)
    case.release_source.set()
    _assert_startup_available(case, runner, {"master", "worker"})
    assert case.inventory_nodes == ["worker"] and case.source_calls == [True]


@pytest.mark.parametrize("eligibility", ["unavailable", "stale_heartbeat", "no_tcp"])
def test_source_window_rescan_uses_live_availability_heartbeat_and_tcp(startup_pool_case, eligibility):
    case = startup_pool_case
    runner = case.start()
    case.join(available=eligibility != "unavailable", fresh=eligibility != "stale_heartbeat",
              connected=eligibility != "no_tcp")
    case.release_source.set()
    _assert_startup_available(case, runner, {"master", "worker"})
    assert case.inventory_nodes == ["worker"]


@pytest.mark.parametrize("boundary", ["sources", "added_inventory", "before_building"])
def test_expanding_pool_never_recaptures_replaced_session(startup_pool_case, monkeypatch, boundary):
    case = startup_pool_case
    runner = case.start()
    old = case.c._connections[(case.op, "worker")]
    case.join()
    def replace(node_id):
        case.scheduler._tcp_server.clients[node_id] = {"connected_at": time.time() + 1,
            "registration_epoch": 99, "registration_confirmed": True}
    if boundary == "sources":
        replace("worker")
    elif boundary == "added_inventory":
        case.inventory_hooks["android"] = lambda: replace("android")
    else:
        original_extend = case.c._extend_auto_inventory
        def extend(*args, **kwargs):
            expanded = original_extend(*args, **kwargs)
            if case.capacity_pools:
                replace("worker")
            return expanded
        monkeypatch.setattr(case.c, "_extend_auto_inventory", extend)
    case.release_source.set()
    runner.join(3)
    assert not runner.is_alive()
    result = case.c.get(case.op)
    assert result["status"] == "failed" and result["reason_code"] == "SUPPLY_WORKER_DISCONNECTED", result
    assert case.c._connections[(case.op, "worker")] == old
    assert not case.build_calls
    if boundary == "sources":
        assert case.inventory_nodes == ["worker"]
    elif boundary == "added_inventory":
        assert case.c._connections[(case.op, "android")] != case.scheduler._tcp_server.clients["android"]["connected_at"]


@pytest.mark.parametrize("action", ["inventory", "ensure"])
def test_later_dispatch_cannot_overwrite_operation_session_capture(tmp_path, action):
    c, op, _ = coordinator(tmp_path)
    c.scheduler._tcp_server.clients["worker"] = {"connected_at": time.time(), "registration_epoch": 2,
                                                 "registration_confirmed": True}
    worker_hello(c.scheduler, "worker")
    with pytest.raises(ModelSupplyError) as error:
        c._send(op, "worker", {"action": action})
    assert error.value.code == "SUPPLY_WORKER_DISCONNECTED"
    assert c._connections[(op, "worker")] == 1.0 and not c.scheduler._tcp_server.sent


@pytest.mark.parametrize("boundary", ["sources", "pending_ack", "added_inventory", "before_building"])
@pytest.mark.parametrize("stop", ["cancel", "local", "demotion"])
def test_pool_expansion_cancel_and_mode_exit_prevent_late_build(startup_pool_case, monkeypatch, boundary, stop):
    case = startup_pool_case
    runner = case.start()
    def stop_operation():
        if stop == "cancel":
            case.c.cancel(case.op)
        elif stop == "local":
            case.scheduler.enabled = False
        else:
            case.scheduler.role = "client"
    if boundary == "pending_ack":
        case.join(acknowledge=False)
        case.release_source.set()
        assert case.waiting.wait(2)
    elif boundary == "added_inventory":
        case.blocked_inventory.add("android")
        case.join()
        case.release_source.set()
        assert case.inventory_sent["android"].wait(2)
    elif boundary == "before_building":
        original_extend = case.c._extend_auto_inventory
        def extend(*args, **kwargs):
            expanded = original_extend(*args, **kwargs)
            if case.capacity_pools:
                stop_operation()
            return expanded
        monkeypatch.setattr(case.c, "_extend_auto_inventory", extend)
    if boundary != "before_building":
        stop_operation()
    case.release_source.set()
    runner.join(3)
    assert not runner.is_alive()
    expected = {"cancel": "SUPPLY_CANCELLED", "local": "SUPPLY_LOCAL_MODE", "demotion": "SUPPLY_MASTER_REQUIRED"}[stop]
    result = case.c.get(case.op)
    assert result["reason_code"] == expected, result
    assert not case.build_calls
    assert not any(payload["action"] == "ensure" for _, payload, _ in case.scheduler._tcp_server.sent)
    if boundary == "pending_ack":
        worker_hello(case.scheduler, "android")
        assert case.inventory_nodes == ["worker"]


def test_new_peer_after_building_cannot_change_published_plan(startup_pool_case):
    case = startup_pool_case
    published = []
    def build(kwargs):
        if kwargs["node_id"] == "master":
            result = case.c.get(case.op)
            assert result["status"] == "building"
            published.append(result["plan"])
            case.join()
    case.on_build = build
    case.release_source.set()
    runner = case.start()
    result = _assert_startup_available(case, runner, {"master", "worker"})
    assert result["plan"] == published[0]
    assert case.inventory_nodes == ["worker"] and case.capacity_pools == [{"master", "worker"}]


def test_capacity_wait_replans_only_when_new_node_inventory_arrives(startup_pool_case):
    case = startup_pool_case
    case.budgets = {node_id: 45 for node_id in ("master", "worker", "android")}
    case.release_source.set()
    runner = case.start()
    assert case.waiting.wait(2)
    assert case.capacity_pools == [{"master", "worker"}]
    assert not case.build_calls and case.c.get(case.op)["plan"] is None
    case.join()
    _assert_startup_available(case, runner, {"master", "worker", "android"})
    assert case.capacity_pools == [{"master", "worker"}, {"master", "worker", "android"}]
    assert case.source_calls == [True] and case.inventory_nodes == ["worker", "android"]


@pytest.mark.parametrize("pending", [False, True])
def test_pool_expansion_wait_is_bounded_by_original_operation_deadline(startup_pool_case, pending):
    case = startup_pool_case
    case.c._operations[case.op]["intent"]["timeout_seconds"] = 1
    if not pending:
        case.budgets = {"master": 45, "worker": 45}
    runner = case.start()
    if pending:
        case.join(acknowledge=False)
    case.release_source.set()
    runner.join(3)
    assert not runner.is_alive()
    result = case.c.get(case.op)
    expected = "SUPPLY_WORKERS_TIMEOUT" if pending else "pipeline_distributed_capacity_insufficient"
    assert result["status"] == "failed" and result["reason_code"] == expected, result
    assert case.source_calls == [True] and not case.build_calls
    assert len(case.capacity_pools) == (0 if pending else 1)


@pytest.mark.parametrize("identity_changed", [False, True])
def test_late_android_gguf_requirement_reuses_source_owner_and_provenance(startup_pool_case, tmp_path, monkeypatch, identity_changed):
    from tests.test_model_supply_runtime_capacity import _write_shard
    case = startup_pool_case
    case.c._operations[case.op]["intent"]["engine"] = "pytorch"
    case.engines = {"master": ["pytorch"], "worker": ["pytorch"]}
    hf = tmp_path / "hf"
    hf.mkdir()
    _write_shard(hf, {f"model.layers.{index}.weight": ("F16", [5]) for index in range(6)})
    case.runtime_source = hf
    original_sources = case.c._sources
    def sources(*args, **kwargs):
        value = original_sources(*args, **kwargs)
        if identity_changed and len(case.source_calls) == 2:
            value["source_model_sha256"] = "e" * 64
        return value
    monkeypatch.setattr(case.c, "_sources", sources)
    runner = case.start()
    case.join()
    case.release_source.set()
    if identity_changed:
        runner.join(3)
        assert not runner.is_alive()
        assert case.c.get(case.op)["reason_code"] == "SUPPLY_SOURCE_CONTRACT_MISMATCH"
        assert not case.build_calls
    else:
        _assert_startup_available(case, runner, {"master", "worker", "android"})
        assert all(call["conversion_proof"] == case.proof for call in case.build_calls)
        assert [call["node_id"] for call in case.build_calls].count("android") == 1
    assert case.source_calls == [False, True]
    assert case.inventory_nodes == ["worker", "android"]


def test_gc_blocked_collector_leaves_status_and_cancel_responsive(tmp_path):
    c, op_id, file = coordinator(tmp_path)
    entered, release, completed = threading.Event(), threading.Event(), threading.Event()
    captured, results, failures = {}, {}, []

    def collect(**kwargs):
        captured.update(kwargs)
        entered.set()
        if not release.wait(5):
            raise AssertionError("test did not release collector")
        return {"removed": 0, "reclaimed_bytes": 0, "skipped": {}, "examined": 0}

    def run_gc():
        try:
            c._collect_garbage(op_id)
        except BaseException as exc:
            failures.append(exc)

    def control():
        try:
            results["status"] = c.get(op_id)
            results["cancel"] = c.cancel(op_id)
        except BaseException as exc:
            failures.append(exc)
        finally:
            completed.set()

    c.scheduler._model_supply_worker.collect_garbage = collect
    collector = threading.Thread(target=run_gc)
    observer = threading.Thread(target=control)
    collector.start()
    try:
        assert entered.wait(2)
        assert c.root / op_id in captured["protected_paths"]
        assert file in captured["protected_paths"]
        assert not captured["cancelled"]()
        observer.start()
        assert completed.wait(1), "status/cancel blocked behind GC disk IO"
        assert results["status"]["status"] == "supplying"
        assert results["cancel"]["status"] == "cancelled"
        assert captured["cancelled"]()
    finally:
        release.set()
        collector.join(2)
        if observer.ident is not None:
            observer.join(2)
    assert not collector.is_alive() and not observer.is_alive()
    assert not failures
    assert c.get(op_id)["status"] == "cancelled"
    assert "retention" not in c.get(op_id)


@pytest.mark.parametrize("change", ["active", "preparing", "lease"])
def test_gc_snapshot_fence_rejects_new_operation_prepare_or_lease(tmp_path, change):
    c, op_id, _ = coordinator(tmp_path)
    def collect(**kwargs):
        assert not c._condition._is_owned()
        assert not kwargs["cancelled"]()
        with c._condition:
            if change == "active":
                c._active = "supply_" + "b" * 32
            elif change == "preparing":
                c._preparing = "supply_" + "b" * 32
            else:
                c._leases[(op_id, "new-worker")] = {"files": {"weight": str(tmp_path / "new-weight")}}
        assert kwargs["cancelled"]()
        return {"removed": 0, "reclaimed_bytes": 0, "skipped": {}, "examined": 0}
    c.scheduler._model_supply_worker.collect_garbage = collect
    c._collect_garbage(op_id)


@pytest.mark.parametrize("node_id", ["master", "worker"])
def test_capacity_wait_refreshes_existing_node_without_new_membership(startup_pool_case, node_id):
    case = startup_pool_case
    case.budgets = {"master": 45, "worker": 45}
    case.release_source.set()
    runner = case.start()
    assert case.waiting.wait(2)
    assert not case.build_calls
    case.budgets[node_id] = 200
    result = _assert_startup_available(case, runner, {"master", "worker"})
    assert len(case.capacity_pools) == 2
    assert case.inventory_nodes == ["worker", "worker"]
    assert case.source_calls == [True]
    assert result["operation_id"] == case.op


def test_unused_candidate_disconnect_does_not_abort_published_minimum_hop_plan(startup_pool_case, monkeypatch):
    import config
    monkeypatch.setattr(config, "PIPELINE_PREFER_ALL_WORKERS", False)
    case = startup_pool_case
    case.join()
    def disconnect_unused(_kwargs):
        published = case.c.get(case.op)
        assigned = {item["node_id"] for item in published["plan"]["assignments"]}
        for node_id in set(published["candidate_node_ids"]) - assigned:
            case.scheduler._tcp_server.clients.pop(node_id, None)
    case.on_build = disconnect_unused
    case.release_source.set()
    runner = case.start()
    runner.join(3)
    assert not runner.is_alive()
    operation = case.c.get(case.op)
    assert operation["status"] == "available", operation
    assigned = {item["node_id"] for item in operation["plan"]["assignments"]}
    assert len(assigned) == 2 and "master" in assigned
    assert set(operation["nodes"]) == assigned
    assert set(operation["candidate_node_ids"]) == {"master", "worker", "android"}
    unused = set(operation["candidate_node_ids"]) - assigned
    assert all((case.op, node_id) not in case.c._connections for node_id in unused)
    assert all(value["status"] == "available" for value in operation["nodes"].values())
