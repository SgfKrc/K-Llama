"""R2 capacity budgets and R3 frozen materialization contracts.

New deployments require negotiated specs and actual receipts. Legacy commands
retain their wire shape with deterministic FP32 compatibility materialization.
"""
from copy import deepcopy
from dataclasses import FrozenInstanceError, replace
import hashlib
import json
import logging
import os
import subprocess
import sys
import threading
import time
from types import ModuleType, SimpleNamespace

import pytest

from qlh_core.contracts.load import RuntimeLoadSpec
from qlh_core.planning.capacity import PipelineCapacityError, capacity_budget, solve_pipeline_capacity
from worker_prepare_contract import (
    IDENTITY_FIELDS, STAGE_IDENTITY_FIELDS, WorkerPrepareError,
    stage_identity, validate_ack, validate_command, validate_identity,
)
from tests.test_worker_prepare import command, identity

pytestmark = pytest.mark.refactor_contract


def spec(**changes):
    values = dict(device="cpu", dtype="float32", quant_type="fp32", context_size=64,
                  kv_device="cpu", kv_dtype="float32", kv_policy="dynamic",
                  components=("layers", "final_norm"), materialization="safetensors_key_filtered",
                  batch_size=32)
    values.update(changes)
    return RuntimeLoadSpec(**values)


def descriptor(count=4):
    return {"model_id": "tiny", "model_type": "qwen2", "model_sha256": "a" * 64,
            "total_layers": count, "layer_weight_bytes": [20] * count,
            "component_weight_bytes": {"embedding": 8, "final_norm": 2, "lm_head": 8}}


def node(name="pc", capacity=200, **changes):
    return {"node_id": name, "capacity_bytes": capacity, "reserve_bytes": 5,
            "runtime_multiplier": 1.0, **changes}


def bound_command(policy=None, *, fmt="safetensors"):
    policy = policy or spec()
    value = identity(fmt=fmt)
    value["load_spec_digest"] = policy.digest
    return command("prepare", value, load_spec=policy.to_dict())


def test_load_spec_freezes_nested_inputs_and_exports_independent_json():
    components = ["layers", "final_norm"]
    policy = spec(components=components)
    digest = policy.digest
    components.append("embedding")
    exported = policy.to_dict()
    exported["components"].append("lm_head")
    assert policy.components == ("final_norm", "layers") and policy.digest == digest
    with pytest.raises(FrozenInstanceError):
        policy.device = "mps"
    canonical = json.dumps(policy.to_dict(), sort_keys=True, separators=(",", ":"))
    assert digest == hashlib.sha256(canonical.encode()).hexdigest()
    assert RuntimeLoadSpec.from_dict(json.loads(canonical)) == policy
    assert spec(components=("final_norm", "layers")).digest == digest


@pytest.mark.parametrize("change", [
    {"device": "cuda:0"}, {"dtype": "float16"}, {"quant_type": "fp16"},
    {"context_size": 128}, {"kv_device": "mps"}, {"kv_dtype": "float16"},
    {"kv_policy": "preallocated"}, {"kv_cache_bytes": 10},
    {"components": ("layers", "final_norm", "embedding")},
    {"materialization": "gguf_segment"}, {"batch_size": 16}, {"sequence_count": 2},
])
def test_every_execution_policy_field_changes_digest_and_cannot_be_tampered(change):
    policy = spec()
    changed = replace(policy, **change)
    assert changed.digest != policy.digest
    data = bound_command(policy)
    data["load_spec"] = changed.to_dict()
    with pytest.raises(WorkerPrepareError, match="digest_mismatch"):
        validate_command(data)


@pytest.mark.parametrize("change", [
    {"device": "auto"}, {"device": "cuda"}, {"dtype": "auto"}, {"quant_type": ""},
    {"context_size": True}, {"context_size": 0}, {"batch_size": 65},
    {"sequence_count": 0}, {"kv_device": "unknown"}, {"kv_dtype": "auto"},
    {"kv_policy": "auto"}, {"kv_cache_bytes": -1},
    {"kv_policy": "disabled", "kv_cache_bytes": 1},
    {"components": ["layers", "layers"]}, {"components": ["unknown"]},
    {"materialization": "auto"}, {"schema_version": 2},
])
def test_load_spec_rejects_unresolved_and_invalid_execution_values(change):
    with pytest.raises(ValueError):
        spec(**change)


def test_load_spec_json_schema_is_exact():
    for data in ({**spec().to_dict(), "unknown": 1}, {"device": "cpu"}, None):
        with pytest.raises(ValueError):
            RuntimeLoadSpec.from_dict(data)


def test_solver_does_not_use_environment_clock_io_or_logger_and_copies_inputs(monkeypatch):
    import builtins

    partial = descriptor(2)
    partial["layer_weight_bytes"] = [20, 0]
    partial["source_model_sha256"] = "a" * 64
    candidates = [node("master", role="master", layer_ranges=[[0, 1]]), node(
        "worker", layer_ranges=[[1, 2]], layer_artifacts=[{
            "layer_range": [1, 2], "segment_mode": "tail", "model_id": "tail.gguf",
            "artifact_sha256": "b" * 64, "source_model_sha256": "a" * 64,
            "layer_weight_bytes": [20],
        }],
    )]
    original = deepcopy((partial, candidates))

    def forbidden(*args, **kwargs):
        raise AssertionError("pure solver accessed ambient state")

    class NoEnvironment(dict):
        __getitem__ = get = __iter__ = __contains__ = forbidden

    with monkeypatch.context() as guard:
        guard.setattr(os, "environ", NoEnvironment())
        guard.setattr(builtins, "open", forbidden)
        guard.setattr(logging.Logger, "_log", forbidden)
        for name in ("time", "monotonic", "perf_counter"):
            guard.setattr(time, name, forbidden)
        first = solve_pipeline_capacity(partial, candidates, safety_margin=1.0,
                                        require_distributed=True)
        second = solve_pipeline_capacity(partial, list(reversed(candidates)), safety_margin=1.0,
                                         require_distributed=True)
    assert first == second and first["admitted"]
    first["assignments"][1]["layer_artifact"]["layer_weight_bytes"][0] = 999
    assert (partial, candidates) == original


def test_core_import_has_no_runtime_or_config_dependencies():
    script = r"""
import sys
import importlib.abc
class BanRuntime(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, *args):
        if fullname.split('.')[0] in {'torch', 'transformers', 'config', 'scheduler', 'model_module'}:
            raise AssertionError(fullname)
sys.meta_path.insert(0, BanRuntime())
from qlh_core.planning.capacity import solve_pipeline_capacity
from pipeline_capacity import solve_pipeline_capacity as legacy
assert legacy is solve_pipeline_capacity
"""
    result = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True,
                            env={**os.environ, "PYTHONPATH": os.pathsep.join(sys.path)}, timeout=30)
    assert result.returncode == 0, result.stderr


def test_budget_admission_and_exclusion_use_identical_rounded_numbers():
    model = descriptor(1)  # 20 layers + 8 embedding + 2 norm + 8 head = 38.
    candidate = node(capacity=76, runtime_multiplier=1.5)
    expected = dict(required_bytes=77, available_bytes=76, reserved_bytes=5,
                    shortfall_bytes=1, allocatable_bytes=71, headroom_bytes=-1)
    assert capacity_budget(38, capacity_bytes=76, reserve_bytes=5,
                           runtime_multiplier=1.5, safety_margin=1.25) == expected
    rejected = solve_pipeline_capacity(model, [candidate], safety_margin=1.25)
    assert not rejected["admitted"]
    check = next(c for c in rejected["capacity_checks"] if c["layer_range"] == [0, 1])
    exclusion = rejected["candidate_exclusions"][0]
    assert check["reason_code"] == exclusion["reason_code"] == "node_capacity_insufficient"
    assert {key: check[key] for key in expected} == expected
    assert {key: exclusion[key] for key in expected} == expected
    candidate["capacity_bytes"] = 77
    admitted = solve_pipeline_capacity(model, [candidate], safety_margin=1.25)
    assignment = admitted["assignments"][0]
    from pipeline_capacity import _required_bytes
    assert admitted["admitted"] and assignment["shortfall_bytes"] == 0
    assert assignment["required_bytes"] == _required_bytes(38, candidate, 1.25) == 77
    assert assignment["reserved_bytes"] == assignment["reserve_bytes"] == 5
    assert assignment["available_bytes"] == assignment["capacity_bytes"] == 77


def test_legacy_required_bytes_accepts_requirement_only_budget_records():
    # model_supply_diagnostics asks for an upper requirement before it has an
    # available-memory sample. This private compatibility API never required it.
    from pipeline_capacity import _required_bytes
    assert _required_bytes(38, {"reserve_bytes": 5, "runtime_multiplier": 1.5}, 1.25) == 77


@pytest.mark.parametrize("available", [76, 77, 78])
def test_manual_scheduler_uses_core_requirement_and_matching_legacy_budget_fields(monkeypatch, available):
    import config
    import pipeline_capacity
    from qlh_core.planning import capacity as core_capacity
    from scheduler import Scheduler

    assert pipeline_capacity._required_bytes is core_capacity._required_bytes
    monkeypatch.setattr(config, "PIPELINE_CAPACITY_SAFETY_MARGIN", 1.25)
    candidate = node("master", capacity=available, runtime_multiplier=1.5,
                     execution_device="cpu", capacity_source="test")
    original = deepcopy(candidate)
    expected = capacity_budget(38, capacity_bytes=available, reserve_bytes=5,
                               runtime_multiplier=1.5, safety_margin=1.25)
    observed = []
    real_budget = core_capacity.capacity_budget

    def track_budget(raw_bytes, **options):
        budget = real_budget(raw_bytes, **options)
        observed.append((raw_bytes, options, budget))
        return budget

    monkeypatch.setattr(core_capacity, "capacity_budget", track_budget)
    # Use an explicit capacity snapshot so the actual manual admission method
    # is tested independently of capability discovery and its separate fixtures.
    scheduler = SimpleNamespace(
        _get_pipeline_capacity_nodes=lambda ids: [candidate],
        _attach_pipeline_node_contract=lambda plan, **options: plan,
    )
    plan = Scheduler._build_manual_pipeline_capacity_plan(scheduler, [{
        "node_id": "master", "start_layer": 0, "end_layer": 1,
        "has_embedding": True, "has_lm_head": True,
    }], descriptor=descriptor(1))
    assert observed == [(38, {
        "capacity_bytes": available, "reserve_bytes": 5,
        "runtime_multiplier": 1.5, "safety_margin": 1.25,
    }, expected)]
    assert candidate == original
    if expected["shortfall_bytes"]:
        assert not plan["admitted"]
        assert plan["reason_code"] == "pipeline_capacity_manual_insufficient"
        assert f"requires {expected['required_bytes']} bytes" in plan["reason"]
        assert f"has {expected['available_bytes']} bytes" in plan["reason"]
    else:
        assert plan["admitted"]
        assignment = plan["assignments"][0]
        assert assignment["required_bytes"] == expected["required_bytes"]
        assert assignment["capacity_bytes"] == expected["available_bytes"]
        assert assignment["reserve_bytes"] == expected["reserved_bytes"]
        assert assignment["headroom_bytes"] == expected["headroom_bytes"]
        assert assignment["raw_weight_bytes"] == 38


def test_reservation_only_exclusions_keep_the_same_budget_semantics():
    rejected = solve_pipeline_capacity(descriptor(), [node(capacity=2)])
    excluded = rejected["excluded_nodes"][0]
    assert excluded["reason_code"] == "node_capacity_unavailable"
    assert (excluded["required_bytes"], excluded["available_bytes"],
            excluded["reserved_bytes"], excluded["shortfall_bytes"]) == (5, 2, 5, 3)


@pytest.mark.parametrize("changes", [
    {"raw_bytes": True}, {"reserve_bytes": -1}, {"capacity_bytes": 1.5},
    {"runtime_multiplier": float("nan")}, {"safety_margin": 0.9},
    {"runtime_multiplier": float("inf")}, {"raw_bytes": 10 ** 1000},
])
def test_budget_rejects_unsafe_arithmetic(changes):
    args = dict(raw_bytes=10, capacity_bytes=100)
    args.update(changes)
    with pytest.raises(PipelineCapacityError):
        capacity_budget(**args)


@pytest.mark.parametrize("score", [float("nan"), float("inf"), "invalid"])
def test_solver_rejects_non_finite_or_unordered_scores(score):
    with pytest.raises(PipelineCapacityError, match="score must be finite"):
        solve_pipeline_capacity(descriptor(), [node(score=score)])


def test_spec_changes_plan_digest_and_output_cannot_mutate_caller():
    policy = spec(components=("layers", "final_norm", "embedding", "lm_head"))
    candidates = [node(load_spec=policy.to_dict(), load_spec_digest=policy.digest)]
    original = deepcopy(candidates)
    first = solve_pipeline_capacity(descriptor(), candidates)
    assert first["admitted"]
    assignment = first["assignments"][0]
    assert assignment["load_spec_digest"] == policy.digest
    assert assignment["load_spec"] == policy.to_dict()
    assignment["load_spec"]["components"].append("other")
    assert candidates == original
    changed = replace(policy, context_size=128)
    second = solve_pipeline_capacity(descriptor(), [node(load_spec=changed.to_dict())])
    assert first["plan_id"] != second["plan_id"]
    assert first["assignments"][0]["required_bytes"] == second["assignments"][0]["required_bytes"]
    with pytest.raises(PipelineCapacityError, match="digest mismatch"):
        solve_pipeline_capacity(descriptor(), [node(load_spec=changed.to_dict(), load_spec_digest=policy.digest)])
    with pytest.raises(PipelineCapacityError, match="device mismatch"):
        solve_pipeline_capacity(descriptor(), [node(load_spec=policy.to_dict(), execution_device="cuda:0")])


def test_kv_is_covered_by_reservation_and_component_constraints_are_not_relaxed():
    policy = spec(kv_cache_bytes=6)
    with pytest.raises(PipelineCapacityError, match="cover KV"):
        solve_pipeline_capacity(descriptor(), [node(load_spec=policy.to_dict())])
    full = replace(policy, components=("layers", "final_norm", "embedding", "lm_head"))
    plan = solve_pipeline_capacity(descriptor(), [node(load_spec=full.to_dict(), reserve_bytes=6)])
    assert plan["admitted"]
    assert plan["assignments"][0]["reserved_bytes"] == 6
    rejected = solve_pipeline_capacity(descriptor(), [node(load_spec=policy.to_dict(), reserve_bytes=6)])
    assert not rejected["admitted"]
    assert {row["reason_code"] for row in rejected["candidate_exclusions"]} == {
        "node_load_spec_components_mismatch",
    }


def test_legacy_wire_shape_stays_exact_and_optional_digest_survives_ack():
    legacy = command("prepare", identity())
    assert validate_command(legacy) == legacy
    assert set(legacy["identity"]) == IDENTITY_FIELDS
    assert set(stage_identity(legacy["identity"])) == STAGE_IDENTITY_FIELDS
    extended = bound_command()
    assert validate_command(extended) == extended
    assert validate_identity(extended["identity"]) == extended["identity"]
    ack = {"action": "ack", "phase": "prepare", "identity": extended["identity"],
           "status": "prepared", "loaded": True}
    assert validate_ack(ack)["identity"]["load_spec_digest"] == spec().digest
    assert set(stage_identity(extended["identity"])) == STAGE_IDENTITY_FIELDS | {"load_spec_digest"}
    with pytest.raises(WorkerPrepareError, match="invalid_prepare_command"):
        validate_command({**legacy, "wire_version": 2})


def test_prepare_requires_both_digest_and_spec_and_matching_context():
    extended = bound_command()
    del extended["load_spec"]
    with pytest.raises(WorkerPrepareError, match="load_spec_missing"):
        validate_command(extended)
    extended = bound_command()
    del extended["identity"]["load_spec_digest"]
    with pytest.raises(WorkerPrepareError, match="digest_mismatch"):
        validate_command(extended)
    extended = bound_command()
    extended["context_size"] = 128
    with pytest.raises(WorkerPrepareError, match="context_mismatch"):
        validate_command(extended)
    extended = bound_command()
    extended["identity"] = []
    with pytest.raises(WorkerPrepareError, match="invalid_prepare_identity"):
        validate_command(extended)


@pytest.fixture
def recording_loader(monkeypatch):
    made = []

    class Parameter:
        device = "cpu"
        dtype = "torch.float32"

        def is_floating_point(self):
            return True

        def numel(self):
            return 4

        def element_size(self):
            return 4

    class Manager:
        def __init__(self):
            self.is_loaded = False
            self.layer_range = None
            self.quants = []
            self.model = SimpleNamespace(parameters=lambda: iter([Parameter()]))
            self.quant_type = "fp32"
            self.unloads = 0
            made.append(self)

        def prepare_pipeline_model(self, **kwargs):
            self.quants.append(kwargs["quant_type"])

        def load_layer_range(self, **kwargs):
            self.quants.append(kwargs["quant_type"])
            self.layer_range = (kwargs["start_layer"], kwargs["end_layer"])
            self.is_loaded = True

        def unload_model(self):
            self.is_loaded = False
            self.model = None
            self.unloads += 1

    module = ModuleType("model_module")
    module.ModelManager = Manager
    module._select_layer_runtime = lambda quant: ("cpu", "torch.float32")
    monkeypatch.setitem(sys.modules, "model_module", module)
    return SimpleNamespace(made=made, module=module, parameter=Parameter)


def selection(value):
    return {"directory": "unused-segment", "manifest": {**deepcopy(value), "format": "safetensors"}}


def test_spec_bound_materialization_ignores_global_quant_between_retries(monkeypatch, recording_loader):
    import config
    from inference_service.engine_host import EngineHost

    data = bound_command()
    host = EngineHost.__new__(EngineHost)
    runtimes = []
    for local_quant in ("int4", "int8"):
        monkeypatch.setattr(config, "QUANT_TYPE", local_quant)
        runtime = host._build_worker_runtime(data, selection(data["identity"]), threading.Event())
        assert runtime.loaded and runtime.identity == data["identity"]
        assert runtime.manager.quants == ["fp32", "fp32"]
        assert runtime.load_spec == runtime.manager.runtime_load_spec == spec()
        runtimes.append(runtime)
    assert len(recording_loader.made) == 2
    assert runtimes[0].manager.quants == runtimes[1].manager.quants
    # Removing QUANT_TYPE altogether proves neither prepare nor materialize reads it.
    monkeypatch.delattr(config, "QUANT_TYPE")
    assert host._build_worker_runtime(data, selection(data["identity"]), threading.Event()).loaded


def test_legacy_materialization_uses_deterministic_compatibility_policy(monkeypatch, recording_loader):
    import config
    from inference_service.engine_host import EngineHost

    host = EngineHost.__new__(EngineHost)
    data = command("prepare", identity())
    for local_quant in ("int4", "int8"):
        monkeypatch.setattr(config, "QUANT_TYPE", local_quant)
        runtime = host._build_worker_runtime(data, selection(data["identity"]), threading.Event())
        assert runtime.manager.quants == ["fp32", "fp32"]
    assert recording_loader.made[0].quants == recording_loader.made[1].quants


def test_device_change_rejects_before_load_and_dtype_drift_unloads(monkeypatch, recording_loader):
    from inference_service.engine_host import EngineHost
    host = EngineHost.__new__(EngineHost)
    data = bound_command()
    selected = selection(data["identity"])
    recording_loader.module._select_layer_runtime = lambda quant: ("cuda:0", "torch.float16")
    # The concrete CPU spec overrides the loader's mutable automatic choice.
    assert host._build_worker_runtime(data, selected, threading.Event()).loaded
    recording_loader.made.clear()
    recording_loader.module._select_layer_runtime = lambda quant: ("cpu", "torch.float32")
    recording_loader.parameter.dtype = "torch.float16"
    with pytest.raises(WorkerPrepareError, match="runtime_mismatch"):
        host._build_worker_runtime(data, selected, threading.Event())
    assert len(recording_loader.made) == 1 and recording_loader.made[0].unloads == 1
    assert not recording_loader.made[0].is_loaded


@pytest.mark.parametrize("change", [
    {"components": ("layers", "embedding")}, {"materialization": "gguf_segment"},
    {"kv_policy": "preallocated"}, {"sequence_count": 2}, {"quant_type": "int4"},
])
def test_unsupported_spec_is_never_silently_materialized(change, recording_loader):
    from inference_service.engine_host import EngineHost
    data = bound_command(spec(**change))
    host = EngineHost.__new__(EngineHost)
    with pytest.raises(WorkerPrepareError, match="load_spec_unsupported"):
        host._build_worker_runtime(data, selection(data["identity"]), threading.Event())
    assert not recording_loader.made


def test_native_spec_uses_frozen_context_batch_and_sequence_count(monkeypatch):
    import keep_head_shim
    import llama_keep_head
    from inference_service.engine_host import EngineHost
    from tests.test_worker_prepare_runtime import Upstream

    environment = {"PATH": "frozen-path"}
    monkeypatch.setattr(keep_head_shim, "resolve_keep_head_shim", lambda **kwargs: "shim")
    monkeypatch.setattr(llama_keep_head, "KeepHeadUpstream", Upstream)
    Upstream.made = []
    monkeypatch.setattr(Upstream, "materialization_observation", lambda self: {
        "materialized_bytes": 128, "context_size": 64, "batch_size": 16, "device": "cpu",
        "dtype": "artifact", "quant_type": "artifact", "kv_device": "cpu", "kv_dtype": "float16", "sequence_count": 2,
    }, raising=False)
    policy = spec(dtype="artifact", quant_type="artifact", kv_dtype="float16", kv_policy="preallocated",
                  components=("layers",), materialization="gguf_segment", sequence_count=2, batch_size=16)
    data = bound_command(policy, fmt="gguf")
    selected = {"model_path": "segment.gguf", "manifest": deepcopy(data["identity"])}
    host = EngineHost.__new__(EngineHost)
    host._environment = environment
    runtime = host._build_worker_runtime(data, selected, threading.Event())
    assert runtime.loaded and runtime.load_spec == policy
    assert runtime.upstream.options == {
        "mode": "nextn", "n_ctx": 64, "n_batch": 16, "n_threads": 8,
        "n_seq_max": 2, "environment": environment,
    }
    assert runtime.upstream.resets == 1


def test_real_filtered_loader_uses_bound_spec(tiny_supply, monkeypatch):
    import config
    import model_module
    import torch
    from model_supply_worker import get_selected_model
    from tests.test_worker_prepare_runtime import supply_identity

    manifest = tiny_supply.activate(0)
    policy = spec(components=("layers", "final_norm", "embedding"))
    value = supply_identity(manifest, 1)
    value["load_spec_digest"] = policy.digest
    # Use CPU availability as an explicit test input; global quant is unavailable.
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.delattr(config, "QUANT_TYPE")
    monkeypatch.delattr(model_module, "QUANT_TYPE")
    host = tiny_supply.host()
    host.prepare_worker_runtime(command("prepare", value, load_spec=policy.to_dict()),
                                get_selected_model(), threading.Event())
    manager = host._worker_runtime_candidate.manager
    assert manager.runtime_load_spec == policy and manager.quant_type == "fp32"
    assert manager._layer_load_metrics["mode"] == "safetensors_key_filtered"
    assert all(p.device.type == "cpu" and p.dtype == torch.float32 for p in manager.model.parameters())
    host.release_worker_runtime()


# Reuse the existing real tiny model fixture without editing its owner tests.
from tests.test_safetensors_supply_worker import tiny_supply
