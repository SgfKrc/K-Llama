"""R2 engine-registry contracts, shared fixtures, and real hello builder wiring."""
from __future__ import annotations

from copy import deepcopy
from dataclasses import replace
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from qlh_core.contracts.engines import (
    ENGINE_REGISTRY, FULL_REQUEST, LAYER_RANGE, FORWARD_LAYERS,
    LEGACY_ENGINE_SPECS, LLAMA_CPP_BINDING, LLAMA_CPP_SERVER,
)
from task_worker_protocol import WorkerProtocolError, build_message, decode_message

pytestmark = pytest.mark.refactor_contract
FIXTURE_PATH = ROOT / "fixtures/engine-capabilities-v1.json"
FIXTURE = json.loads(FIXTURE_PATH.read_text(encoding="utf-8"))


def hello(capabilities):
    return build_message("hello", {
        "node_id": "pc_registry_fixture", "worker_kind": "pc_full_worker",
        "min_version": 3, "max_version": 3, "capabilities": capabilities,
    }, message_id="msg_registry_fixture", sent_at_ms=1, version=3)


def test_fixture_copies_are_identical_and_cover_the_entire_static_contract():
    android = ROOT / "android/app/src/test/resources/engine-capabilities-v1.json"
    assert FIXTURE_PATH.read_bytes() == android.read_bytes()
    assert FIXTURE["registry"] == ENGINE_REGISTRY.snapshot()
    assert len({(e.family, e.runtime_kind) for e in ENGINE_REGISTRY.entries}) == len(ENGINE_REGISTRY.entries)
    for entry in ENGINE_REGISTRY.entries:
        assert entry.native_abi and entry.input_kinds and entry.output_kinds
        assert entry.optimization_constraints
        assert set(dict(entry.architectures)) == entry.operations
        assert set(dict(entry.precisions)) == entry.operations
        if entry.artifact_kind:
            assert entry.artifact_kind in entry.artifact_formats


def test_core_contract_imports_only_stdlib_and_has_no_runtime_side_effects():
    source = """
import builtins, sys
sys.path.insert(0, 'src')
original = builtins.__import__
def guarded(name, globals=None, locals=None, fromlist=(), level=0):
    # level > 0 即相对导入：`from . import x` 的 name 是空串，`from ._constants import *`
    # 的 name 是**包内子模块名**（如 re 的 _constants）。两者都不是顶层模块，拿
    # sys.stdlib_module_names 判会误报 —— Python 3.12 的 re 包正是这样导入
    # _compiler/_parser/_constants（只在 name 上空判不够，必须看 level）。
    if level == 0 and name and name.split('.')[0] not in sys.stdlib_module_names and not name.startswith('qlh_core'):
        raise AssertionError('non-stdlib/legacy import: ' + name)
    return original(name, globals, locals, fromlist, level)
builtins.__import__ = guarded
from qlh_core.contracts.engines import ENGINE_REGISTRY
assert ENGINE_REGISTRY.entries
assert 'torch' not in sys.modules and 'scheduler' not in sys.modules
"""
    result = subprocess.run([sys.executable, "-I", "-c", source], cwd=ROOT,
                            capture_output=True, text=True, timeout=15)
    assert result.returncode == 0, result.stderr


def test_both_legacy_tables_are_derived_projections_of_the_same_entries():
    import koakuma_engine as koakuma
    import model_load_resolver as resolver

    assert koakuma.LEGACY_ENGINE_SPECS is resolver.LEGACY_ENGINE_SPECS is LEGACY_ENGINE_SPECS
    for family, spec in LEGACY_ENGINE_SPECS.items():
        assert koakuma._BACKEND_CAPABILITIES[family].capabilities == spec.capabilities
        assert resolver.ENGINE_CAPABILITIES[family].artifact_kind == spec.artifact_kind
        assert resolver.ENGINE_CAPABILITIES[family].layer_range_mode == spec.layer_range_mode
    with pytest.raises(TypeError):
        koakuma._BACKEND_CAPABILITIES["new"] = None
    with pytest.raises(TypeError):
        resolver.ENGINE_CAPABILITIES["new"] = None


@pytest.mark.parametrize("case", FIXTURE["builder_cases"], ids=lambda case: case["name"])
def test_real_pc_builder_matches_shared_fixture_and_roundtrips_through_validator(case, monkeypatch):
    import config
    import scheduler_task_worker as worker

    spec = ENGINE_REGISTRY.require(case["family"], case["runtime_kind"])
    requested = set(case["probed_operations"] if case["probed_operations"] is not None else spec.operations)
    # Installed-runtime discovery is isolated; the actual concrete host is
    # supplied, and neither path may materialize a model merely to build hello.
    monkeypatch.setattr(worker, "torch_available", lambda: False)
    monkeypatch.setattr(worker.importlib.util, "find_spec", lambda name: None)
    monkeypatch.setattr(worker, "keep_head_shim_available", lambda: False)
    monkeypatch.setattr(config, "ISLAND_ENABLED", False)
    monkeypatch.setattr(config, "EXTERNAL_ENABLED", False)
    monkeypatch.delenv("QLH_LAYER_GGUF", raising=False)
    def supports(capability):
        if capability == "forward_layers":
            return FORWARD_LAYERS in requested
        if capability == "chat":
            return FULL_REQUEST in requested
        return capability in spec.capabilities
    def forbidden_load(*args, **kwargs):
        raise AssertionError("hello must never load/ensure a full model")
    host = SimpleNamespace(
        engine_type=spec.family, runtime_kind=spec.runtime_kind, native_abi=spec.native_abi,
        layer_range=None, supports=supports, load_model=forbidden_load,
        ensure_full_model=forbidden_load,
    )
    harness = worker.SchedulerTaskWorkerMixin()
    harness._model_control = SimpleNamespace(boot_id="fixture-worker-boot")
    harness._host = host
    ranges = case["layer_ranges"]
    harness._active_layer_config = {"engine": spec.family, "layer_range": ranges[0]} if ranges else {}
    model = {"model_id": "fixture_model", "engine": spec.family, "format": spec.artifact_kind,
             "revision": "1", "sha256": "a" * 64}
    harness._task_worker_full_model_identity = lambda engines: model if case["full_inference_available"] else None
    harness._configured_layer_artifact = lambda: None
    harness._task_worker_layer_artifact_identity = lambda artifact: None
    harness._runtime_profile_for_capabilities = lambda: "unspecified"

    actual = harness._task_worker_capabilities()

    assert actual["node_incarnation"] == "fixture-worker-boot"
    assert actual["engine_capabilities"] == case["engine_capabilities"]
    assert actual["stage_types"] == case["pc_stage_types"]
    assert actual["layer_ranges"] == ranges
    if not case["full_inference_available"]:
        assert "full_inference" not in actual["stage_types"]
        assert all(FULL_REQUEST not in e["operations"] for e in actual["engine_capabilities"]["runtimes"])
    encoded = hello(actual).snapshot()
    assert decode_message(encoded).payload["capabilities"] == actual


@pytest.mark.parametrize("case", FIXTURE["invalid_cases"], ids=lambda case: case["name"])
def test_real_pc_hello_validator_rejects_unknown_abi_and_runtime_overclaims(case):
    with pytest.raises(WorkerProtocolError) as error:
        hello(deepcopy(case["capabilities"]))
    assert error.value.code == "invalid_capabilities"


def test_dynamic_probe_cannot_expand_server_or_unknown_runtime_capabilities():
    from koakuma_engine import runtime_supports
    server = SimpleNamespace(engine_type="llama_cpp", runtime_kind=LLAMA_CPP_SERVER,
                             native_abi="llama_server_http.v1", supports=lambda capability: True)
    assert runtime_supports(server, "chat")
    assert not runtime_supports(server, FORWARD_LAYERS)
    assert not runtime_supports(server, "load_layer_range")
    assert not runtime_supports(server, "custom")
    unknown = SimpleNamespace(engine_type="llama_cpp", supports=lambda capability: True)
    assert not runtime_supports(unknown, FORWARD_LAYERS)
    server.native_abi = "unknown.v99"
    assert not runtime_supports(server, "chat")


def test_planner_uses_schema_abi_and_path_consistency_instead_of_platform_or_family():
    from scheduler import _node_supports_forward_layers
    binding = ENGINE_REGISTRY.advertisement([ENGINE_REGISTRY.require("llama_cpp", LLAMA_CPP_BINDING)])
    server = ENGINE_REGISTRY.advertisement([ENGINE_REGISTRY.require("llama_cpp", LLAMA_CPP_SERVER)])
    for platform in ("pc", "android", "unknown"):
        assert not _node_supports_forward_layers({"node_type": platform})
        assert not _node_supports_forward_layers({"node_type": platform, "device_info": {"backend_id": "llama_cpp"}})
        assert _node_supports_forward_layers({"node_type": platform, "device_info": {"engine_capabilities": binding}})
        assert not _node_supports_forward_layers({"node_type": platform, "device_info": {"engine_capabilities": server}})
    for extra in ({"inference_path": LLAMA_CPP_SERVER}, {"backend_id": "island"},
                  {"native_abi": "unknown.v99"}, {"serves_layer_segments": False},
                  {"capabilities": ["chat"]}, {"forward_layers": False}):
        assert not _node_supports_forward_layers({"device_info": {"engine_capabilities": binding, **extra}})
    poisoned = deepcopy(binding)
    poisoned["runtimes"][0]["native_abi"] = "unknown.v99"
    assert not _node_supports_forward_layers({"device_info": {"engine_capabilities": poisoned}})


def test_resolver_explicit_server_kind_and_schema_share_one_contract():
    from model_load_resolver import ModelLoadFacts, ModelLoadResolutionError, resolve_model_load
    facts = ModelLoadFacts(model_id="fixture", has_gguf=True, gguf_path="fixture.gguf")
    server = resolve_model_load(facts, requested_engine="llama_cpp", requested_quant="Q4_K_M",
                                runtime_kind=LLAMA_CPP_SERVER)
    assert server.runtime_kind == server.runtime == LLAMA_CPP_SERVER
    assert server.layer_range_mode == "none"
    assert server.operations == (FULL_REQUEST,)
    assert server.engine_capabilities == ENGINE_REGISTRY.advertisement([
        ENGINE_REGISTRY.require("llama_cpp", LLAMA_CPP_SERVER)])
    for operation in ("layer_range", "dynamic_layer_range", "distributed_load"):
        with pytest.raises(ModelLoadResolutionError) as error:
            resolve_model_load(facts, requested_engine="llama_cpp", requested_quant="gguf",
                               operation=operation, runtime_kind=LLAMA_CPP_SERVER)
        assert error.value.code == "LAYER_RANGE_MODE_UNSUPPORTED"
    with pytest.raises(ModelLoadResolutionError) as error:
        resolve_model_load(replace(facts, runtime_kind=LLAMA_CPP_BINDING),
                           requested_engine="llama_cpp", requested_quant="gguf", runtime_kind=LLAMA_CPP_SERVER)
    assert error.value.code == "ENGINE_RUNTIME_MISMATCH"
    for changed, code in ((replace(facts, runtime_abi="unknown.v99"), "ENGINE_ABI_UNSUPPORTED"),
                          (replace(facts, runtime_kind="pytorch_cpu"), "ENGINE_RUNTIME_UNSUPPORTED"),
                          (replace(facts, probed_operations=()), "ENGINE_OPERATION_UNSUPPORTED")):
        with pytest.raises(ModelLoadResolutionError) as error:
            resolve_model_load(changed, requested_engine="llama_cpp", requested_quant="gguf")
        assert error.value.code == code


def test_resolver_architecture_and_actual_precision_are_bounded_by_the_static_operation():
    from model_load_resolver import ModelLoadFacts, ModelLoadResolutionError, resolve_model_load
    facts = ModelLoadFacts(model_id="fixture", has_safetensors=True, safetensors_path="fixture",
                           cuda_available=True, model_architecture="qwen2")
    result = resolve_model_load(facts, requested_engine="pytorch", requested_quant="int4", operation="layer_range")
    assert result.runtime_quant == "fp16"
    assert ENGINE_REGISTRY.require("pytorch", result.runtime_kind).supports(
        LAYER_RANGE, architecture="qwen2", precision=result.runtime_quant)
    with pytest.raises(ModelLoadResolutionError):
        resolve_model_load(replace(facts, model_architecture="unknown_architecture"),
                           requested_engine="pytorch", requested_quant="int4", operation="layer_range")


def test_legacy_wire_remains_decodable_without_granting_planner_layer_capabilities():
    from scheduler import _node_supports_forward_layers
    old = {"stage_types": ["full_inference"], "engines": ["llama_cpp"], "models": [], "max_concurrency": 1}
    assert hello(old).payload["capabilities"] == old
    assert not _node_supports_forward_layers({"node_type": "pc", "device_info": {"capabilities": old}})


def test_production_local_profile_and_remote_hello_merge_keep_concrete_evidence(monkeypatch):
    from scheduler import Scheduler, NodeInfo, NodeState, _node_supports_forward_layers
    from qlh_core.contracts.engines import LLAMA_CPP_ANDROID
    from task_worker_capabilities import build_engine_capabilities
    from scheduler_task_worker import SchedulerTaskWorkerMixin

    sched = Scheduler()
    sched._role_override = "master"
    sched.nodes = {
        "master": NodeInfo(node_id="master", role="master", state=NodeState.ONLINE),
        "remote": NodeInfo(node_id="remote", role="client", state=NodeState.ONLINE,
                           device_info={"backend_id": "pytorch", "inference_path": LLAMA_CPP_SERVER}),
    }
    binding = build_engine_capabilities([("llama_cpp", LLAMA_CPP_BINDING)])
    monkeypatch.setattr(sched, "_task_worker_engine_schema", lambda: binding)
    monkeypatch.setattr(sched, "push_layer_config_to_clients", lambda: None)
    sched.update_local_device_profile({"ram": {"available_gb": 8}})
    assert sched._local_device_profile["engine_capabilities"] == binding
    assert _node_supports_forward_layers(sched.nodes["master"])
    remote = build_engine_capabilities([("llama_cpp", LLAMA_CPP_ANDROID)])
    SchedulerTaskWorkerMixin._merge_task_worker_capabilities_into_node(sched, "remote", {
        "stage_types": ["layer_forward"], "engines": ["llama_cpp"], "engine_capabilities": remote,
    })
    assert sched.nodes["remote"].device_info["engine_capabilities"] == remote
    assert _node_supports_forward_layers(sched.nodes["remote"])
    assert "inference_path" not in sched.nodes["remote"].device_info
    # Hardware-only refreshes must retain the accepted hello contract.
    sched._on_tcp_message("remote", {"type": "status_res", "data": {
        "state": "online", "device_info": {"ram": {"available_gb": 7}},
    }})
    assert sched.nodes["remote"].device_info["engine_capabilities"] == remote
    assert _node_supports_forward_layers(sched.nodes["remote"])
    server = build_engine_capabilities([("llama_cpp", LLAMA_CPP_SERVER)])
    monkeypatch.setattr(sched, "_task_worker_engine_schema", lambda: server)
    sched.refresh_task_worker_capabilities()
    assert sched.nodes["master"].device_info["engine_capabilities"] == server
    assert not _node_supports_forward_layers(sched.nodes["master"])
    # Old hello, missing schema, and rejection revoke an old layer grant.
    SchedulerTaskWorkerMixin._merge_task_worker_capabilities_into_node(sched, "remote", {})
    assert not _node_supports_forward_layers(sched.nodes["remote"])


def test_local_discovery_failure_is_closed_and_does_not_load_a_full_model(monkeypatch):
    import config
    import scheduler_task_worker as worker
    monkeypatch.setattr(worker, "torch_available", lambda: False)
    monkeypatch.setattr(worker.importlib.util, "find_spec", lambda name: None)
    monkeypatch.setattr(worker, "keep_head_shim_available", lambda: False)
    monkeypatch.setattr(config, "ISLAND_ENABLED", False)
    monkeypatch.setattr(config, "EXTERNAL_ENABLED", False)
    harness = worker.SchedulerTaskWorkerMixin()
    harness._host = SimpleNamespace(engine_type="llama_cpp", supports=lambda capability: True)
    schema = harness._task_worker_engine_schema()
    assert schema == {"schema_version": 1, "runtimes": []}
