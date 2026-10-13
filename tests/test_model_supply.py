"""Supply plans create missing segments without claiming loaded executors."""
import copy
import json
import threading
from pathlib import Path
from types import SimpleNamespace

import pytest

from model_supply import ModelSupplyCoordinator, ModelSupplyError, plan_segment_supply


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
    plan = plan_segment_supply({"llama_cpp": descriptor()}, records, master_engine="llama_cpp", alignment=2)
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
        self.clients = {"worker": {"connected_at": 1.0}}
        self.sent = []
    def get_client_info(self, node_id):
        return self.clients.get(node_id)
    def send_to_client(self, node_id, payload, message_type):
        self.sent.append((node_id, payload, message_type))


def coordinator(tmp_path):
    scheduler = SimpleNamespace(_effective_role=lambda: "master", get_distributed_inference_enabled=lambda: True,
                                _tcp_server=Server(), _model_supply_worker=SimpleNamespace(cancel=lambda op: None))
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
    c.scheduler._tcp_server.clients["tail"] = {"connected_at": 2.0}
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
                             preferred_ranges={"tail":[[6,8]]})
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
