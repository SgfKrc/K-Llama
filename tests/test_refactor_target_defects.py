"""Correct target assertions for confirmed legacy defects, retired by named stages.

Only a precise, observed defect signature raises a defect-specific exception.
Unexpected exceptions and harness failures remain ordinary test failures; a fix
produces strict XPASS until the marker and migration record are reviewed.
"""
from copy import deepcopy
from dataclasses import asdict
import sys
import threading
from types import ModuleType, SimpleNamespace

import numpy as np
import pytest

pytestmark = pytest.mark.refactor_contract


class RegistryLostUpdate(AssertionError):
    pass


class HostMirrorDiverged(AssertionError):
    pass


class RuntimePolicyDrift(AssertionError):
    pass


class SamplingSilentlyIgnored(AssertionError):
    pass


def assert_target(actual, expected, *, known_bad, defect):
    if actual == known_bad:
        raise defect(f"observed legacy defect: {actual!r}")
    assert actual == expected


def test_stale_download_cannot_replace_supply_publication(tmp_path, monkeypatch):
    import local_store
    import model_download_jobs as jobs
    import model_supply_sources as sources
    from model_config import ModelConfig

    cfg = ModelConfig(model_id="r0-f02", name="fixture", model_path=str(tmp_path / "unused-model"))
    old = {**asdict(cfg), "artifact_sha256": "a" * 64, "source": "seed"}
    assert local_store.save_local_experimental_model(cfg.model_id, old)
    published = []

    def infer_after_publication(*_args):
        sources._save(cfg, old, registry_snapshot=deepcopy(old), artifact_sha256="d" * 64,
                      cross_format_provenance_sha256="e" * 64, source="supply")
        published.extend(deepcopy(local_store.get_local_experimental_models()))
        assert len(published) == 1
        return {"model_type": "safetensors", "model_path": cfg.model_path, "gguf_path": "", "files": []}

    monkeypatch.setattr(jobs, "_infer_artifact", infer_after_publication)
    jobs._register(cfg.model_id, tmp_path / "unused-model", {
        "sha256": "a" * 64,
        "manifest": {"artifact_sha256": "a" * 64, "manifest_sha256": "b" * 64},
    }, revision="older-download")
    rows = local_store.get_local_experimental_models()
    assert len(rows) == 1
    row = rows[0]
    assert_target((row.get("cross_format_provenance_sha256"), row.get("artifact_sha256"), row.get("source")),
                  ("e" * 64, "d" * 64, "supply"),
                  known_bad=(None, "a" * 64, "download-service"), defect=RegistryLostUpdate)
    assert rows == published


@pytest.mark.xfail(strict=True, raises=HostMirrorDiverged,
                   reason="F05 / R3-R4: host mirror shadows manager update; remove when published runtime becomes a consistent readonly snapshot")
def test_published_host_and_manager_cannot_diverge_after_assignment():
    from model_host import ModelHost

    manager = SimpleNamespace(model_path="published-old")
    host = object.__new__(ModelHost)
    object.__setattr__(host, "_manager", manager)
    host._apply_pipeline_candidate_host_state({
        "model_loaded": False, "current_quant": "int4", "mirrors": {"model_path": "published-old"},
    })
    assert (host.model_path, manager.model_path) == ("published-old", "published-old")
    host.model_path = "new"
    assert_target((host.model_path, manager.model_path), ("new", "new"),
                  known_bad=("published-old", "new"), defect=HostMirrorDiverged)


def test_same_prepare_identity_cannot_drift_with_local_quant(monkeypatch):
    import config
    from inference_service.engine_host import EngineHost
    from worker_prepare_contract import validate_command
    from tests.test_worker_prepare import identity, command

    class RecordingManager:
        def __init__(self):
            self.is_loaded = False
            self.layer_range = None
            self.quants = []

        def prepare_pipeline_model(self, **kwargs):
            self.quants.append(kwargs["quant_type"])

        def load_layer_range(self, **kwargs):
            self.quants.append(kwargs["quant_type"])
            self.layer_range = (kwargs["start_layer"], kwargs["end_layer"])
            self.is_loaded = True

    fake_module = ModuleType("model_module")
    fake_module.ModelManager = RecordingManager
    monkeypatch.setitem(sys.modules, "model_module", fake_module)
    value = identity()
    cmd = validate_command(command("prepare", value))
    selected = {"directory": "unused-segment", "manifest": {**deepcopy(value), "format": "safetensors"}}
    host = EngineHost.__new__(EngineHost)
    runtimes = []
    for local_quant in ("int4", "int8"):
        monkeypatch.setattr(config, "QUANT_TYPE", local_quant)
        runtime = host._build_worker_runtime(deepcopy(cmd), deepcopy(selected), threading.Event())
        assert runtime.loaded and runtime.identity == cmd["identity"]
        assert runtime.manager.layer_range == (1, 3)
        runtimes.append(runtime)
    actual = tuple(tuple(r.manager.quants) for r in runtimes)
    assert actual[0] == actual[1] == ("fp32", "fp32")


def run_controlled_route_a(temperature, top_p):
    from scheduler_pipeline import SchedulerPipelineMixin

    calls = []

    class Tokenizer:
        eos_token_id = 999

        def __call__(self, *_args, **_kwargs):
            return {"input_ids": np.array([[1]], dtype=np.int64)}

        def decode(self, ids, **_kwargs):
            return "".join({1: "P", 42: "A", 999: ""}[int(i)] for i in ids)

    def forward(**_kwargs):
        calls.append("forward")
        return {"hidden_states": np.zeros((1, 1, 2), dtype=np.float32), "cache": 1}

    def offer(**_kwargs):
        calls.append("offer")
        return {"kind": "token", "token_argmax": 42}

    callbacks = SimpleNamespace(
        thinking_system_prompt="",
        active_task_graph_model_identity=lambda: SimpleNamespace(sha256="a" * 64, model_id="fixture"),
        build_model_chat_prompt=lambda *_args, **_kwargs: "P",
        format_model_response=lambda text, *_args, **_kwargs: (text, ""),
    )
    host = SimpleNamespace(tokenizer=Tokenizer(),
                           ensure_layer_range=lambda *_args, **_kwargs: calls.append("ensure"), forward_layers=forward)
    harness = SimpleNamespace(
        _host=host, _verify_pipeline_readiness=lambda _nodes: (True, ""), _require_callbacks=lambda: callbacks,
        _scheduler_facade_global=lambda _name: None, get_effective_node_id=lambda: "master",
        _execute_layer_stage_offer=offer, _pipeline_lock=threading.Lock(), _kv_cache_lock=threading.Lock(),
        _pipeline_context=threading.local(), _pipeline_active_tasks=set(), _pipeline_task_contracts={},
        _kv_cache={}, _clear_pipeline_runtime_state=lambda _task_id: None,
        _record_pipeline_task_accounting=lambda **_kwargs: {},
    )
    result = SchedulerPipelineMixin._run_route_a_stage_pipeline(
        harness, prompt="hello", max_new_tokens=1, temperature=temperature, top_p=top_p,
        session_id=None, messages=None, show_thinking=False, routing_preference="auto",
        master_assignment={"start_layer": 0, "end_layer": 1, "layers_count": 1},
        stage_nodes=[{"node_id": "tail", "start_layer": 1, "end_layer": 2,
                      "stage_model_identity": {"model_id": "tail", "engine": "llama_cpp", "format": "gguf",
                                               "revision": "1", "sha256": "b" * 64}}],
    )
    return result, tuple(calls)


def test_route_a_greedy_control_reaches_real_decode_loop():
    result, calls = run_controlled_route_a(0.0, 1.0)
    assert "error" not in result
    assert result["response"] == "A"
    assert result["metrics"]["steps"][0]["token"] == 42
    assert calls == ("ensure", "forward", "offer")


@pytest.mark.parametrize("temperature,top_p", [(0.7, 1.0), (0.7, 0.2)])
@pytest.mark.xfail(strict=True, raises=SamplingSilentlyIgnored,
                   reason="F15 / R0-R4: greedy-only stage output ignores sampling; remove when unsupported sampling is rejected before runtime work")
def test_route_a_rejects_unsupported_sampling_before_execution(temperature, top_p):
    result, calls = run_controlled_route_a(temperature, top_p)
    actual = (result.get("reason_code"), result.get("response"), calls)
    assert_target(actual, ("route_a_sampling_unsupported", "", ()),
                  known_bad=(None, "A", ("ensure", "forward", "offer")), defect=SamplingSilentlyIgnored)
