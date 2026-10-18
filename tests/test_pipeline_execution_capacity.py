"""Capacity admission follows the selected runtime, including uncommitted candidates."""
import math
import threading
from types import SimpleNamespace

import pytest

import config
import model_host as host_module
import model_module
from llama_engine import LlamaCppEngine
from model_host import ModelHost
from model_module import ModelManager
from scheduler import NodeInfo, NodeState, Scheduler
from qlh_core.contracts.engines import ENGINE_REGISTRY, LLAMA_CPP_BINDING, PYTORCH_CUDA


def _engine_evidence(backend="pytorch"):
    kind = LLAMA_CPP_BINDING if backend == "llama_cpp" else PYTORCH_CUDA
    return ENGINE_REGISTRY.advertisement([ENGINE_REGISTRY.require(backend, kind)])


GIB = 1024 ** 3
MIB = 1024 ** 2


def _descriptor():
    return {
        "model_id": "capacity-test", "model_type": "qwen2",
        "model_sha256": "a" * 64, "total_layers": 1,
        "pipeline_runtime_supported": True,
        "layer_weight_bytes": [512 * MIB],
        "component_weight_bytes": {
            "embedding": 0, "final_norm": 0, "lm_head": 0,
            "visual": 0, "mtp": 0, "other": 0,
        },
    }


def _torch_runtime(device=None):
    manager = ModelManager()
    manager._engine_type = "pytorch"
    if device is not None:
        # Exercise the production device getter without allocating GPU weights.
        manager.model = SimpleNamespace(device=device)
    return manager


def _profile(*, vram_gb=8, ram_gb=3, backend="pytorch"):
    return {
        "backend_id": backend,
        "engine_capabilities": _engine_evidence(backend),
        "gpu": {
            "name": "RTX", "cuda_available": True, "is_integrated": False,
            "vram_total_gb": 8, "vram_free_gb": vram_gb,
        },
        "ram": {"available_gb": ram_gb},
    }


@pytest.fixture
def make_scheduler(monkeypatch):
    monkeypatch.setattr("scheduler.TASK_WORKER_EXPERIMENTAL_ENABLED", True)
    monkeypatch.setattr("psutil.virtual_memory", lambda: SimpleNamespace(available=3 * GIB))

    def make(host, *, profile=None, local_id="master", role="master"):
        sched = Scheduler(host=host)
        sched._role_override = role
        monkeypatch.setattr(sched, "get_effective_node_id", lambda: "pc-local")
        monkeypatch.setattr(sched, "_configured_layer_artifact", lambda: None)
        selected_profile = dict(profile if profile is not None else _profile())
        # This fixture explicitly installs a PyTorch CUDA execution path; CPU
        # budgeting/candidate selection is the independent behavior under test.
        selected_profile.setdefault("engine_capabilities", _engine_evidence())
        sched.nodes = {
            local_id: NodeInfo(
                node_id=local_id, role=role, state=NodeState.ONLINE,
                device_info=selected_profile,
            ),
        }
        return sched

    return make


def _project(sched):
    records = sched._get_pipeline_capacity_nodes()
    assert len(records) == 1
    return records[0]


def test_ll_cpu_uses_live_ram_even_when_profile_and_config_say_cuda(make_scheduler, monkeypatch):
    monkeypatch.setattr(config, "INFERENCE_ENGINE", "pytorch")
    monkeypatch.setattr(config, "DEVICE", "cuda")
    engine = LlamaCppEngine()
    sched = make_scheduler(ModelHost(manager=engine))

    record = _project(sched)

    assert engine.get_device() == "cpu"
    assert record["execution_device"] == "cpu"
    assert record["capacity_source"] == "ram.available_bytes_live"
    assert record["capacity_bytes"] == 3 * GIB
    assert record["runtime_multiplier"] == 2.0
    assert record["reserve_bytes"] == int(config.PIPELINE_CAPACITY_RESERVE_MB * MIB)


@pytest.mark.parametrize("device", ["cpu", "cuda:0"])
def test_pytorch_materialized_device_wins_over_hardware_profile(device, make_scheduler):
    runtime = _torch_runtime(device)
    record = _project(make_scheduler(ModelHost(manager=runtime)))

    assert str(runtime.get_device()) == device
    if device == "cpu":
        assert record["capacity_source"] == "ram.available_bytes_live"
        assert record["capacity_bytes"] == 3 * GIB
        assert record["runtime_multiplier"] == 2.0
    else:
        assert record["execution_device"] == "cuda"
        assert record["capacity_source"] == "gpu.vram_free_gb"
        assert record["capacity_bytes"] == 8 * GIB
        assert record["runtime_multiplier"] == 1.0


def test_legacy_manager_wrapping_ll_does_not_use_its_torch_device_getter(make_scheduler, monkeypatch):
    monkeypatch.setattr(model_module, "DEVICE", "cuda")
    manager = _torch_runtime()
    manager._engine_type = "llama_cpp"
    manager._llama_engine = LlamaCppEngine()

    record = _project(make_scheduler(ModelHost(manager=manager)))

    assert str(manager.get_device()) == "cuda"
    assert manager._llama_engine.get_device() == "cpu"
    assert record["execution_device"] == "cpu"
    assert record["capacity_bytes"] == 3 * GIB


@pytest.fixture
def prepare_candidate(monkeypatch, tmp_path):
    def metadata_only(manager, **kwargs):
        if isinstance(manager, ModelManager):
            manager._engine_type = "pytorch"
        manager._pipeline_distributed_only = True
        manager._pipeline_descriptor = _descriptor()
        manager.quant_type = kwargs.get("quant_type") or "fp16"
        return _descriptor()

    monkeypatch.setattr(LlamaCppEngine, "prepare_pipeline_model", metadata_only)
    monkeypatch.setattr(ModelManager, "prepare_pipeline_model", metadata_only)
    # Keep teardown independent of an actual CUDA context when availability is mocked.
    monkeypatch.setattr(ModelManager, "unload_model", lambda manager: setattr(manager, "model", None))

    def prepare(host, engine):
        if engine == "llama_cpp":
            path = tmp_path / "candidate.gguf"
            path.write_bytes(b"metadata-only-fixture")
        else:
            path = tmp_path / "candidate"
            path.mkdir()
        return host.prepare_pipeline_candidate("capacity-txn", "candidate", str(path), "fp16")

    return prepare


def test_uncommitted_ll_candidate_uses_ram_while_active_torch_stays_cuda(
    make_scheduler, prepare_candidate,
):
    active = _torch_runtime("cuda:0")
    host = ModelHost(manager=active)
    sched = make_scheduler(host)
    assert _project(sched)["capacity_source"] == "gpu.vram_free_gb"

    prepare_candidate(host, "llama_cpp")
    record = _project(sched)

    assert host.pipeline_candidate_status()["phase"] == "prepared"
    assert host.peek_manager() is active
    assert host.engine_type == "pytorch"
    assert record["execution_device"] == "cpu"
    assert record["capacity_bytes"] == 3 * GIB
    host.abort_pipeline_candidate("capacity-txn")
    assert _project(sched)["capacity_source"] == "gpu.vram_free_gb"


@pytest.mark.parametrize("cuda_available", [False, True])
def test_uncommitted_torch_candidate_uses_layer_loader_device_not_old_config(
    cuda_available, make_scheduler, prepare_candidate, monkeypatch,
):
    monkeypatch.setattr(model_module.torch.cuda, "is_available", lambda: cuda_available)
    monkeypatch.setattr(model_module, "DEVICE", "cpu")
    active = LlamaCppEngine()
    host = ModelHost(manager=active)
    sched = make_scheduler(host)
    prepare_candidate(host, "pytorch")

    record = _project(sched)

    assert host.peek_manager() is active
    assert host.engine_type == "llama_cpp"
    assert host.pipeline_candidate_status()["phase"] == "prepared"
    expected_device = model_module._select_layer_runtime("fp16")[0].split(":")[0]
    assert record["execution_device"] == expected_device
    assert record["capacity_bytes"] == (8 if cuda_available else 3) * GIB
    assert record["capacity_source"] == (
        "gpu.vram_free_gb" if cuda_available else "ram.available_bytes_live"
    )


def test_lazy_host_projection_does_not_instantiate_manager(make_scheduler, monkeypatch):
    def forbidden(_self):
        raise AssertionError("capacity projection must not initialize a runtime")

    monkeypatch.setattr(host_module._LazyModelManager, "_get_instance", forbidden)
    monkeypatch.setattr(config, "INFERENCE_ENGINE", "llama_cpp")
    host = ModelHost()

    record = _project(make_scheduler(host))

    assert host.peek_manager() is None
    assert record["execution_device"] == "cpu"
    assert record["capacity_bytes"] == 3 * GIB


@pytest.mark.parametrize("use_hello", [False, True])
def test_remote_ll_worker_on_gpu_pc_uses_ram(use_hello, make_scheduler):
    sched = make_scheduler(ModelHost(manager=LlamaCppEngine()))
    sched.nodes = {
        "remote": NodeInfo(
            node_id="remote", role="client", state=NodeState.ONLINE,
            device_info=_profile(ram_gb=5, backend="pytorch" if use_hello else "llama_cpp"),
        ),
    }
    if use_hello:
        sched._task_worker_control.status = lambda role: {
            "workers": [{
                "node_id": "remote", "layer_stage_dispatch_enabled": True,
                "capabilities": {
                    "engines": ["pytorch", "llama_cpp"],
                    "models": [{"engine": "llama_cpp"}],
                },
            }],
        }

    from scheduler_task_worker import SchedulerTaskWorkerMixin
    SchedulerTaskWorkerMixin._merge_task_worker_capabilities_into_node(sched, "remote", {
        "engines": ["llama_cpp"], "stage_types": ["layer_forward"],
        "models": [{"model_id": "remote-segment", "engine": "llama_cpp"}],
        "engine_capabilities": _engine_evidence("llama_cpp"),
    })
    # Production merge supersedes the old hardware/family hint.
    assert sched.nodes["remote"].device_info["backend_id"] == "llama_cpp"
    record = _project(sched)

    assert record["execution_device"] == "cpu"
    assert record["capacity_source"] == "ram.available_gb"
    assert record["capacity_bytes"] == 5 * GIB


def test_cpu_missing_ram_cannot_borrow_vram(make_scheduler):
    record = _project(make_scheduler(ModelHost(manager=LlamaCppEngine()), profile={
        "gpu": _profile()["gpu"],
    }))

    assert record["execution_device"] == "cpu"
    assert record["capacity_bytes"] == 0
    assert record["capacity_source"] == ""


@pytest.mark.parametrize("gpu", [{}, _profile(vram_gb=0)["gpu"]])
def test_cuda_missing_vram_cannot_borrow_ram(gpu, make_scheduler):
    record = _project(make_scheduler(ModelHost(manager=_torch_runtime("cuda:0")), profile={
        "gpu": gpu, "ram": {"available_gb": 64},
    }))

    assert record["execution_device"] == "cuda"
    assert record["capacity_bytes"] == 0
    assert record["capacity_source"] == ""


@pytest.mark.parametrize("engine", ["llama_cpp", "pytorch"])
def test_manual_admission_uses_execution_memory_and_keeps_safety_margin(
    engine, make_scheduler, monkeypatch,
):
    monkeypatch.setattr("psutil.virtual_memory", lambda: SimpleNamespace(available=GIB))
    runtime = LlamaCppEngine() if engine == "llama_cpp" else _torch_runtime("cuda:0")
    sched = make_scheduler(ModelHost(manager=runtime))

    plan = sched._build_manual_pipeline_capacity_plan([{
        "node_id": "master", "start_layer": 0, "end_layer": 1,
        "has_embedding": True, "has_lm_head": True,
    }], descriptor=_descriptor())

    if engine == "llama_cpp":
        assert plan["admitted"] is False
        assert plan["reason_code"] == "pipeline_capacity_manual_insufficient"
        assert f"has {GIB} bytes" in plan["reason"]
    else:
        assert plan["admitted"] is True
        assignment = plan["assignments"][0]
        assert assignment["capacity_bytes"] == 8 * GIB
        assert assignment["required_bytes"] == int(config.PIPELINE_CAPACITY_RESERVE_MB * MIB) + math.ceil(
            512 * MIB * config.PIPELINE_CAPACITY_SAFETY_MARGIN
        )
        assert assignment["capacity_source"] == "gpu.vram_free_gb"


@pytest.mark.parametrize("use_candidate", [False, True])
def test_device_callback_does_not_reverse_full_then_candidate_lock_order(
    use_candidate, make_scheduler, prepare_candidate, monkeypatch,
):
    host = ModelHost(manager=_torch_runtime("cuda:0"))
    if use_candidate:
        prepare_candidate(host, "pytorch")

        def materialize(manager, **kwargs):
            manager.model = SimpleNamespace(device="cuda:0")
            return {"success": True}

        monkeypatch.setattr(ModelManager, "load_layer_range", materialize)
        host.materialize_pipeline_candidate("capacity-txn", 0, 1, True, True)
        runtime = host._pipeline_candidate_transaction.manager
    else:
        runtime = host.peek_manager()
    sched = make_scheduler(host)
    full_held = threading.Event()
    device_entered = threading.Event()
    observations = {}
    original_get_device = runtime.get_device

    def get_device():
        device_entered.set()
        with host.full_chat_execution_lock:
            return original_get_device()

    monkeypatch.setattr(runtime, "get_device", get_device)

    def full_then_candidate_reader():
        with host.full_chat_execution_lock:
            full_held.set()
            observations["device_entered"] = device_entered.wait(3)
            # A finite acquire turns a reversed edge into a test failure, not a
            # permanently wedged pytest process. This is the status/commit order.
            acquired = host._pipeline_candidate_lock.acquire(timeout=1)
            observations["candidate_acquired"] = acquired
            if acquired:
                try:
                    observations["status"] = host.pipeline_candidate_status()
                finally:
                    host._pipeline_candidate_lock.release()

    reader = threading.Thread(target=full_then_candidate_reader, daemon=True)
    reader.start()
    assert full_held.wait(3)
    try:
        record = _project(sched)
    finally:
        reader.join(4)

    assert not reader.is_alive()
    assert observations["device_entered"] is True
    assert observations["candidate_acquired"] is True
    assert record["execution_device"] == "cuda"


def test_candidate_abort_during_device_lookup_invalidates_capacity(
    make_scheduler, prepare_candidate, monkeypatch,
):
    host = ModelHost(manager=LlamaCppEngine())
    prepare_candidate(host, "pytorch")
    runtime = host._pipeline_candidate_transaction.manager
    runtime.model = SimpleNamespace(device="cuda:0")

    def abort_during_lookup():
        host.abort_pipeline_candidate("capacity-txn")
        return "cuda:0"

    monkeypatch.setattr(runtime, "get_device", abort_during_lookup)

    record = _project(make_scheduler(host))

    assert host.pipeline_candidate_status()["phase"] == "aborted"
    assert record["capacity_bytes"] == 0
    assert record["capacity_source"] == ""
    assert record["execution_device"] == "unknown"


def test_device_lookup_failure_rejects_capacity_and_has_readable_diagnostic(
    make_scheduler, monkeypatch, caplog,
):
    runtime = _torch_runtime("cuda:0")

    def unavailable():
        raise RuntimeError("device lookup failed")

    monkeypatch.setattr(runtime, "get_device", unavailable)

    record = _project(make_scheduler(ModelHost(manager=runtime)))

    assert record["capacity_bytes"] == 0
    assert record["capacity_source"] == ""
    assert "runtime capacity device unavailable: engine=pytorch" in caplog.text


def test_local_client_uses_own_live_ram_instead_of_gpu_profile(make_scheduler):
    record = _project(make_scheduler(
        ModelHost(manager=LlamaCppEngine()), local_id="pc-local", role="client",
    ))

    assert record["execution_device"] == "cpu"
    assert record["capacity_source"] == "ram.available_bytes_live"
    assert record["capacity_bytes"] == 3 * GIB
