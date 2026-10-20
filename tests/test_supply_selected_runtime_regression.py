"""Supply available -> fresh evidence -> Scheduler admission and frozen load policy."""
from copy import deepcopy
import math
from types import SimpleNamespace

import pytest

import config
from model_preflight import REASON_ARTIFACT_IDENTITY_INVALID
from model_supply import ModelSupplyError
from model_supply_capacity import bind_supplied_runtime_capacity, project_inventory_capacity
from qlh_core.planning.load_spec import assignment_load_spec
from tests.test_model_preflight import _chain, _preflight
from tests.test_model_supply import Server, startup_pool_case
from tests.test_model_supply_runtime_capacity import _write_shard
from tests.test_supply_capacity_evidence import _accept_supply_hello, _supplied_capabilities


@pytest.mark.parametrize("engine,fmt", [("pytorch", "safetensors"), ("llama_cpp", "gguf")])
@pytest.mark.parametrize("explicit", [False, True])
def test_preflight_freezes_exact_selected_engine_identity(engine, fmt, explicit):
    assignments, nodes = _chain()
    for item, node in zip(assignments[1:], nodes):
        node["models"][0].update(engine=engine, format=fmt)
        if explicit:
            node["engine"] = item["engine"] = engine
    result = _preflight(assignments, nodes)
    assert result.ok, result.reason
    assert all(binding["stage_model_identity"]["engine"] == engine
               and binding["stage_model_identity"]["format"] == fmt
               for binding in result.stage_bindings)


@pytest.mark.parametrize("change", ["wrong_format", "selected_engine", "conflicting_selection",
                                    "unknown_engine", "wrong_sha", "wrong_id", "ambiguous"])
def test_preflight_engine_extension_preserves_strict_physical_identity(change):
    assignments, nodes = _chain()
    node = nodes[0]
    node["engine"] = "pytorch"
    model = node["models"][0]
    model.update(engine="pytorch", format="safetensors")
    if change == "wrong_format":
        model["format"] = "gguf"
    elif change == "selected_engine":
        node["engine"] = "llama_cpp"
    elif change == "conflicting_selection":
        assignments[1]["engine"] = "llama_cpp"
    elif change == "unknown_engine":
        node["engine"] = "unknown"
    elif change == "wrong_sha":
        model["sha256"] = "f" * 64
    elif change == "wrong_id":
        model["model_id"] = "another-artifact"
    else:
        node["models"].append(deepcopy(model))
    result = _preflight(assignments, nodes)
    assert not result.ok and result.reason_code == REASON_ARTIFACT_IDENTITY_INVALID


@pytest.mark.parametrize("device,size,dtype", [("cuda", 4, "float32"), ("cuda:0", 2, "float16"),
                                               ("cpu", 2, "float16"), ("cpu", 4, "float32")])
def test_runtime_projection_precision_overrides_device_default(device, size, dtype):
    assignment = {"execution_device": device, "has_embedding": True, "has_lm_head": False,
                  "runtime_target_element_size": size, "runtime_target_dtype": dtype}
    spec = assignment_load_spec(assignment, artifact_format="safetensors")
    assert spec.dtype == spec.kv_dtype == dtype
    assert spec.quant_type == {2: "fp16", 4: "fp32"}[size]
    assert "embedding" in spec.components and "lm_head" not in spec.components
    assignment["load_spec"] = spec.to_dict()
    assert assignment_load_spec(assignment, artifact_format="safetensors") == spec


@pytest.mark.parametrize("hints", [
    {"runtime_target_element_size": True}, {"runtime_target_element_size": 3},
    {"runtime_target_dtype": None}, {"runtime_target_dtype": []}, {"runtime_target_dtype": "bfloat16"},
    {"runtime_target_dtype": "float16", "runtime_target_element_size": 4},
])
def test_invalid_or_conflicting_runtime_precision_is_rejected(hints):
    with pytest.raises(ValueError, match="deployment_runtime_dtype"):
        assignment_load_spec({"execution_device": "cuda", **hints}, artifact_format="safetensors")


def test_frozen_dtype_cannot_drift_and_unquoted_legacy_policy_is_unchanged():
    assignment = {"execution_device": "cuda", "has_embedding": True}
    spec = assignment_load_spec(assignment, artifact_format="safetensors")
    assert spec.dtype == "float16"
    assignment.update(load_spec=spec.to_dict(), runtime_target_element_size=4)
    with pytest.raises(ValueError, match="deployment_runtime_dtype_mismatch"):
        assignment_load_spec(assignment, artifact_format="safetensors")
    native = assignment_load_spec({"execution_device": "cuda", "runtime_target_element_size": 4},
                                  artifact_format="gguf")
    assert native.dtype == native.quant_type == "artifact" and native.device == "cpu"


def _available_case(case, tmp_path, monkeypatch, *, offload, target_size=2, source_dtype="F32"):
    case.c._operations[case.op]["intent"].update(engine="pytorch", timeout_seconds=2)
    case.engines = {"master": ["pytorch"], "worker": ["pytorch"]}
    case.budgets = {"master": 1000, "worker": 1000}
    case.runtime_source = tmp_path / "source"
    case.runtime_source.mkdir()
    _write_shard(case.runtime_source, {
        **{f"model.layers.{i}.weight": (source_dtype, [5]) for i in range(6)},
        "model.embed_tokens.weight": (source_dtype, [6]),
        "model.norm.weight": (source_dtype, [1]), "lm_head.weight": (source_dtype, [4]),
    })
    monkeypatch.setattr(config, "PIPELINE_CAPACITY_SAFETY_MARGIN", 1.25)
    original_sources = case.c._sources
    source_size = {"F16": 2, "F32": 4}[source_dtype]
    def sources(*args, **kwargs):
        result = original_sources(*args, **kwargs)
        descriptor = result["descriptors"]["pytorch"]
        descriptor.update(layer_weight_bytes=[5 * source_size] * 6,
                          component_weight_bytes={"embedding": 6 * source_size, "final_norm": source_size,
                                                  "lm_head": 4 * source_size, "other": 0})
        case.descriptor = deepcopy(descriptor)
        return result
    monkeypatch.setattr(case.c, "_sources", sources)
    original_capacity = case.scheduler._get_pipeline_capacity_nodes
    def capacity(eligible):
        return [{**record, "reserve_bytes": 7} for record in original_capacity(eligible)]
    case.scheduler._get_pipeline_capacity_nodes = capacity
    master_capacity = 7 + math.ceil(6 * target_size * 1.25) if offload else 1000
    case.inventories = {
        node: {"distributed": True, "engines": ["pytorch"], "memory_available_bytes": 1000,
               "disk_free_bytes": 10**12, "sources": [], "pytorch_runtime": {
                   "execution_device": "cuda", "target_element_size": target_size,
                   "available_bytes": master_capacity if node == "master" else (60 if target_size == 2 else 120),
                   "materializations": ["safetensors_key_filtered", "safetensors_layer_offload"]}}
        for node in ("master", "worker")}
    case.scheduler._model_supply_worker.inventory = lambda **kwargs: deepcopy(case.inventories["master"])
    server = case.scheduler._tcp_server
    original_send = server.send_to_client
    def send(node, payload, message_type):
        if payload["action"] != "inventory":
            return original_send(node, payload, message_type)
        Server.send_to_client(server, node, payload, message_type)
        case.c.handle(node, {**deepcopy(case.inventories[node]), "action": "inventory_result",
                             "operation_id": payload["operation_id"], "node_id": node})
    monkeypatch.setattr(server, "send_to_client", send)
    case.release_source.set()
    runner = case.start()
    runner.join(3)
    assert not runner.is_alive()
    operation = case.c.get(case.op)
    assert operation["status"] == "available", operation
    assert operation["loaded_ready"] is False
    case.server = server
    case.scheduler._model_supply_worker.selected_model_path = lambda: operation["master_model_path"]
    case.records = []
    for assignment in operation["plan"]["assignments"]:
        node = assignment["node_id"]
        caps = _supplied_capabilities(case, node)
        start, end = assignment["start_layer"], assignment["end_layer"]
        caps["layer_artifacts"][0]["layer_weight_bytes"] = case.descriptor["layer_weight_bytes"][start:end]
        record = {"node_id": node, "role": "master" if node == "master" else "client", "score": 100,
                  "reserve_bytes": 7, "capacity_bytes": 10**12, "execution_device": "cuda", "engine": "pytorch",
                  "layer_ranges": [[start, end]]}
        if node != "master":
            _accept_supply_hello(case, node, caps)
            record.update(layer_artifacts=deepcopy(caps["layer_artifacts"]), models=deepcopy(caps["models"]))
        case.records.append(record)
    # Model preparation sees the supplied head headers, not the complete source.
    case.ready_descriptor = deepcopy(case.descriptor)
    master_end = operation["plan"]["assignments"][0]["end_layer"]
    case.ready_descriptor["layer_weight_bytes"][master_end:] = [0] * (6 - master_end)
    case.ready_scheduler = SimpleNamespace(
        _get_pipeline_capacity_nodes=lambda eligible: deepcopy(case.records),
        _relay_claimed_layer_prefix=lambda total: None, _relay_layer_claim_ranges=lambda total: None,
        _attach_pipeline_node_contract=lambda plan, **kwargs: plan)
    return operation


def _ready(case, evidence):
    from scheduler import Scheduler
    return Scheduler.get_pipeline_capacity_plan(case.ready_scheduler, {"master", "worker"},
        descriptor=case.ready_descriptor, require_distributed=True, capacity_evidence=evidence)


@pytest.mark.parametrize("offload", [False, True], ids=["resident", "offload"])
@pytest.mark.parametrize("target_size,source_dtype", [(2, "F32"), (4, "F16")], ids=["f32-to-f16", "f16-to-f32"])
def test_available_to_fresh_solver_preflight_and_local_freeze(startup_pool_case, tmp_path, monkeypatch,
                                                             offload, target_size, source_dtype):
    from model_host import ModelHost
    case = startup_pool_case
    operation = _available_case(case, tmp_path, monkeypatch, offload=offload,
                                target_size=target_size, source_dtype=source_dtype)
    before = deepcopy(case.records)
    supplied = operation["plan"]["assignments"]
    mode = "safetensors_layer_offload" if offload else "safetensors_key_filtered"
    assert supplied[0]["load_spec"]["materialization"] == mode
    from model_supply import _digest
    assert operation["plan"]["plan_id"] == _digest({
        "source": case.descriptor["source_model_sha256"], "assignments": supplied})
    with case.c.prepare_guard(case.op):
        evidence = case.c.prepare_capacity_evidence(case.op)
        projected, provenance = evidence.project(case.records)
        for record, item in zip(projected, provenance):
            assert item["host_required_bytes"] == record["host_required_bytes"]
            assert item["range_weight_bytes"] == record["range_weight_bytes"]
        provenance[0]["range_weight_bytes"][0]["weight_bytes"] = 1
        assert projected[0]["range_weight_bytes"][0]["weight_bytes"] != 1
        plan = _ready(case, evidence)
        assert plan["admitted"], plan
        assert plan["model_preflight"]["ok"]
        for assignment, frozen in zip(plan["assignments"], supplied):
            assert [assignment["start_layer"], assignment["end_layer"]] == [frozen["start_layer"], frozen["end_layer"]]
            assert assignment["range_weight_quote"]["weight_bytes"] == frozen["device_weight_bytes"]
            assert assignment["raw_weight_bytes"] == frozen["device_weight_bytes"]
            assert assignment["required_bytes"] == 7 + math.ceil(frozen["device_weight_bytes"] * 1.25)
            assert assignment["reserve_bytes"] == 7 and assignment["runtime_multiplier"] == 1
            assert assignment["host_required_bytes"] == frozen["host_required_bytes"]
            assert assignment["host_capacity_bytes"] == 1000
            assert assignment["load_spec_digest"] == frozen["load_spec_digest"]
        tail = plan["assignments"][1]
        assert tail["stage_model_identity"]["engine"] == "pytorch"
        assert tail["stage_model_identity"]["format"] == "safetensors"
        assert tail["layer_artifact"] == before[1]["layer_artifacts"][0]
        local = plan["assignments"][0]
        host = SimpleNamespace(_require_pipeline_candidate=lambda txn: SimpleNamespace(
            manager=SimpleNamespace(engine_type="pytorch"), model_sha256="d" * 64, descriptor=case.ready_descriptor))
        frozen = ModelHost.freeze_pipeline_candidate_assignment(host, "candidate", local,
                                                                config_id="config", generation=1)
        assert frozen["load_spec"] == supplied[0]["load_spec"]
        assert frozen["identity"]["load_spec_digest"] == supplied[0]["load_spec_digest"]
        assert frozen["host_required_bytes"] == supplied[0]["host_required_bytes"]
        assert frozen["range_weight_quote"] == local["range_weight_quote"]
        assert "embedding" in frozen["load_spec"]["components"] and "lm_head" not in frozen["load_spec"]["components"]
        # A new independent host sample changes identity without changing CUDA cost.
        case.inventories["master"]["memory_available_bytes"] += 1
        newer = _ready(case, case.c.prepare_capacity_evidence(case.op))
        assert newer["admitted"] and newer["plan_id"] != plan["plan_id"]
        assert newer["assignments"][0]["required_bytes"] == local["required_bytes"]
    assert case.records == before and case.c.get(case.op)["plan"] == operation["plan"]


@pytest.mark.parametrize("offload", [False, True])
@pytest.mark.parametrize("change", ["host_shortage", "device_shortage", "dtype_drift", "artifact_sha",
                                    "source_sha", "range", "tokenizer"])
def test_fresh_boundary_rejects_resource_or_identity_drift(startup_pool_case, tmp_path, monkeypatch,
                                                         offload, change):
    case = startup_pool_case
    operation = _available_case(case, tmp_path, monkeypatch, offload=offload)
    master = operation["plan"]["assignments"][0]
    if change == "host_shortage":
        case.inventories["master"]["memory_available_bytes"] = master["host_required_bytes"] - 1
    elif change == "device_shortage":
        case.inventories["master"]["pytorch_runtime"]["available_bytes"] = 7 + math.ceil(master["device_weight_bytes"] * 1.25) - 1
    elif change == "dtype_drift":
        case.inventories["master"]["pytorch_runtime"]["target_element_size"] = 4
    with case.c.prepare_guard(case.op):
        evidence = case.c.prepare_capacity_evidence(case.op)
        artifact = case.records[1]["layer_artifacts"][0]
        if change == "artifact_sha":
            artifact["artifact_sha256"] = "f" * 64
        elif change == "source_sha":
            artifact["source_model_sha256"] = "f" * 64
        elif change == "range":
            artifact["layer_range"][0] += 1
        elif change == "tokenizer":
            artifact["tokenizer_sha256"] = "f" * 64
        if change == "device_shortage":
            plan = _ready(case, evidence)
            assert not plan["admitted"] and not plan["assignments"]
        else:
            code = ("SUPPLY_RUNTIME_HOST_CAPACITY_INSUFFICIENT" if change == "host_shortage" else
                    "SUPPLY_RUNTIME_LOAD_SPEC_INVALID" if change == "dtype_drift" else
                    "SUPPLY_RUNTIME_CAPABILITY_CHANGED")
            with pytest.raises(ModelSupplyError) as exc:
                _ready(case, evidence)
            assert exc.value.code == code


@pytest.mark.parametrize("mode", ["safetensors_key_filtered", "safetensors_layer_offload"])
@pytest.mark.parametrize("change", ["missing_spec", "digest", "device", "boundary", "zero_weight", "negative_host",
                                    "host_underquote", "dtype", "range", "materialization"])
def test_runtime_binder_rejects_incomplete_or_inconsistent_frozen_quotes(mode, change):
    assignment = {"node_id": "master", "engine": "pytorch", "execution_device": "cuda",
        "start_layer": 0, "end_layer": 2, "has_embedding": True, "has_lm_head": False,
        "runtime_materialization": mode, "runtime_target_dtype": "float32", "runtime_target_element_size": 4,
        "host_source_bytes": 40, "host_conversion_bytes": 20, "host_required_bytes": 80,
        "device_weight_bytes": 40}
    spec = assignment_load_spec(assignment, artifact_format="safetensors")
    assignment.update(load_spec=spec.to_dict(), load_spec_digest=spec.digest)
    if change == "missing_spec":
        assignment.pop("load_spec")
    elif change == "digest":
        assignment["load_spec_digest"] = "f" * 64
    elif change == "device":
        assignment["execution_device"] = "cpu"
    elif change == "boundary":
        assignment["has_lm_head"] = True
    elif change == "zero_weight":
        assignment["device_weight_bytes"] = 0
    elif change == "negative_host":
        assignment["host_conversion_bytes"] = -1
    elif change == "host_underquote":
        assignment["host_required_bytes"] = 59
    elif change == "dtype":
        assignment["runtime_target_element_size"] = 2
    elif change == "range":
        assignment["end_layer"] = True
    else:
        assignment["runtime_materialization"] = "unknown"
    with pytest.raises(ModelSupplyError) as exc:
        bind_supplied_runtime_capacity({"node_id": "master", "reserve_bytes": 7}, assignment,
                                      {"memory_available_bytes": 1000})
    assert exc.value.code == "SUPPLY_RUNTIME_LOAD_SPEC_INVALID"


@pytest.mark.parametrize("mode", ["safetensors_key_filtered", "safetensors_layer_offload"])
def test_exact_runtime_quote_preserves_source_contract_and_charges_budget_once(mode):
    from pipeline_capacity import solve_pipeline_capacity
    from tests.test_model_supply import descriptor, node
    source = descriptor(layers=4, layer_bytes=200, components=20)
    master = {**node("master", 25), "reserve_bytes": 7, "execution_device": "cuda", "engine": "pytorch",
              "layer_ranges": [[0, 2]]}
    assignment = {"node_id": "master", "engine": "pytorch", "execution_device": "cuda", "start_layer": 0,
        "end_layer": 2, "has_embedding": True, "has_lm_head": False, "runtime_materialization": mode,
        "runtime_target_dtype": "float32", "runtime_target_element_size": 4,
        "host_source_bytes": 440, "host_conversion_bytes": 20, "host_required_bytes": 582,
        "device_weight_bytes": 100}
    spec = assignment_load_spec(assignment, artifact_format="safetensors")
    assignment.update(load_spec=spec.to_dict(), load_spec_digest=spec.digest)
    inventory = {"distributed": True, "engines": ["pytorch"], "memory_available_bytes": 800,
                 "pytorch_runtime": {"execution_device": "cuda", "target_element_size": 4,
                    "available_bytes": 132, "materializations": ["safetensors_key_filtered", "safetensors_layer_offload"]}}
    before = deepcopy((master, assignment, inventory))
    record = project_inventory_capacity(bind_supplied_runtime_capacity(master, assignment, inventory), inventory, "pytorch")
    tail = {**node("worker", 10000), "layer_ranges": [[2, 4]]}
    plan = solve_pipeline_capacity(source, [record, tail], safety_margin=1.25,
                                   require_distributed=True, required_node_id="master")
    assert plan["admitted"], plan
    selected = plan["assignments"][0]
    assert selected["raw_weight_bytes"] == 100 and selected["required_bytes"] == 132
    assert selected["host_required_bytes"] == 582 and selected["host_capacity_bytes"] == 800
    assert selected["load_spec"]["dtype"] == "float32" and selected["load_spec_digest"] == spec.digest
    assert (master, assignment, inventory) == before
    record["host_required_bytes"] += 1
    changed = solve_pipeline_capacity(source, [record, tail], safety_margin=1.25,
                                      require_distributed=True, required_node_id="master")
    assert changed["admitted"] and changed["plan_id"] != plan["plan_id"]
    assert changed["assignments"][0]["required_bytes"] == 132


@pytest.mark.parametrize("change,reason", [("source_model_id", "model_preflight_model_id_mismatch"),
    ("source_model_sha256", "model_preflight_source_model_mismatch"),
    ("hidden_size", "model_preflight_hidden_size_mismatch"),
    ("tokenizer_sha256", "model_preflight_tokenizer_mismatch"),
    ("segment_mode", "model_preflight_segment_role_mismatch")])
def test_pytorch_preflight_keeps_logical_and_segment_contracts(change, reason):
    from model_preflight import preflight_pipeline_artifacts
    from tests.test_model_preflight import MODEL, TOTAL, HIDDEN, TOKENIZER, SOURCE_SHA
    assignments, nodes = _chain()
    for item, node in zip(assignments[1:], nodes):
        node["models"][0].update(engine="pytorch", format="safetensors")
        node["engine"] = item["engine"] = "pytorch"
    artifact = nodes[0]["layer_artifacts"][0]
    artifact[change] = {"source_model_id": "wrong-model", "source_model_sha256": "f" * 64,
                        "hidden_size": 1, "tokenizer_sha256": "f" * 64, "segment_mode": "tail"}[change]
    assignments[1]["layer_artifact"] = deepcopy(artifact)
    result = preflight_pipeline_artifacts(model_id=MODEL, total_layers=TOTAL, hidden_size=HIDDEN,
        tokenizer_sha256=TOKENIZER, source_model_sha256=SOURCE_SHA,
        assignments=assignments, nodes=nodes, require_distributed=True)
    assert not result.ok and result.reason_code == reason
