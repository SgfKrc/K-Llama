"""Exercise real load argument boundaries and state, without allocating weights."""
from types import SimpleNamespace
import ctypes

import pytest

import inference_switches as switches
import llama_engine
from llama_engine import LlamaCppEngine


@pytest.fixture
def binding(monkeypatch, tmp_path):
    seen = []
    def new(kwargs, n_seq_max):
        # The actual ctypes fields reject symbolic strings; exercise that ABI.
        import llama_cpp
        params = llama_cpp.llama_context_default_params()
        for key in ("type_k", "type_v", "n_threads", "n_ctx"):
            if key in kwargs:
                setattr(params, key, kwargs[key])
        seen.append(dict(kwargs))
        return SimpleNamespace(context_params=params, close=lambda: None)
    monkeypatch.setattr(llama_engine, "_new_llama_with_seq_max", new)
    monkeypatch.setattr(LlamaCppEngine, "_install_chat_template_controls", lambda self: None)
    monkeypatch.setattr(LlamaCppEngine, "_auto_threads", lambda self: 8)
    monkeypatch.setattr("device_profiler.get_profile", lambda: SimpleNamespace(to_dict=lambda: {}))
    engine = LlamaCppEngine()
    engine._environment = {"QLH_STATE_DIR": str(tmp_path)}
    yield engine, seen
    engine.unload()


def load(engine, **kwargs):
    engine._load_model_runtime("dummy.gguf", n_ctx=512, **kwargs)


def test_environment_threads_reach_constructor_and_state(binding):
    engine, seen = binding
    engine._environment[switches.ENV_THREADS] = "3"
    load(engine)
    assert seen[0]["n_threads"] == 3
    assert engine._n_threads == 3
    assert engine.get_model_info()["inference_switches"]["effective"]["n_threads"] == 3
    engine.unload()
    assert engine.get_model_info()["inference_switches"] is None


def test_call_threads_win_over_environment(binding):
    engine, seen = binding
    engine._environment[switches.ENV_THREADS] = "3"
    load(engine, n_threads=5)
    assert seen[0]["n_threads"] == 5
    assert engine.get_model_info()["inference_switches"]["sources"]["n_threads"] == "call_argument"


def test_explicit_flash_off_reaches_constructor(binding):
    engine, seen = binding
    engine._environment[switches.ENV_FLASH_ATTN] = "off"
    load(engine)
    assert seen[0]["flash_attn"] is False


def test_explicit_call_flash_overrides_environment_conflict(binding):
    engine, seen = binding
    engine._environment.update({switches.ENV_KV_CACHE: "q8_0", switches.ENV_FLASH_ATTN: "off"})
    load(engine, flash_attn=True)
    assert seen[0]["flash_attn"] is True
    assert type(seen[0]["type_k"]) is int


def test_quantized_cache_uses_real_ctypes_enum(binding):
    engine, seen = binding
    engine._environment[switches.ENV_KV_CACHE] = "q8_0"
    load(engine)
    assert seen[0]["type_k"] == switches.binding_kv_type("q8_0")
    assert seen[0]["type_v"] == switches.binding_kv_type("q8_0")
    assert seen[0]["flash_attn"] is True


def test_final_merged_quantized_flash_conflict_fails_before_native(binding):
    engine, seen = binding
    engine._environment[switches.ENV_KV_CACHE] = "q8_0"
    with pytest.raises(ValueError, match="quantized_kv_requires_flash_attn"):
        load(engine, flash_attn=False)
    assert seen == []
    assert engine._inference_switches is None


def test_environment_conflict_is_not_swallowed(binding):
    engine, seen = binding
    engine._environment.update({switches.ENV_KV_CACHE: "q8_0", switches.ENV_FLASH_ATTN: "off"})
    with pytest.raises(ValueError, match="flash-attn"):
        load(engine)
    assert seen == []


def test_explicit_f16_and_flash_off_override_environment(binding):
    engine, seen = binding
    engine._environment.update({switches.ENV_KV_CACHE: "q8_0", switches.ENV_FLASH_ATTN: "on"})
    load(engine, type_k="f16", type_v="f16", flash_attn=False)
    assert seen[0]["type_k"] == switches.binding_kv_type("f16")
    assert seen[0]["flash_attn"] is False


def test_quantized_kv_preference_does_not_enable_pytorch_flash(monkeypatch):
    import model_module
    monkeypatch.setenv(switches.ENV_KV_CACHE, "q8_0")
    monkeypatch.delenv(switches.ENV_FLASH_ATTN, raising=False)
    assert model_module._resolve_pytorch_flash_attn() is False


def test_server_does_not_advertise_binding_isa(monkeypatch):
    monkeypatch.setattr(switches, "_cached_engine_isa", lambda: pytest.fail("binding ISA queried"))
    assert switches.resolve_inference_switches(env={}, execution_path="llama_server").engine_isa == {}


def test_unknown_calibration_path_falls_back_with_reason(tmp_path):
    state = switches.resolve_for_load("missing.gguf", {"n_ctx": 512}, execution_path="llama_cpp_binding",
        env={switches.ENV_AUTO: "0", switches.ENV_CALIBRATION_PATH: str(tmp_path / "missing.json")})
    assert state.selection["reason"] == "auto_disabled"
    assert state.is_default


@pytest.fixture
def server(monkeypatch, tmp_path):
    from llama_server_engine import LlamaServerEngine
    class Process:
        def __init__(self, spec, **kwargs):
            self.spec, self.base_url = spec, "http://127.0.0.1:12345"
        def start(self): pass
        def stop(self): pass
    class Client:
        is_loaded = True
        def load_model(self, **kwargs): pass
        def unload(self): pass
        def get_model_info(self): return {"engine": "llama_cpp"}
    monkeypatch.setattr("llama_server_process.LlamaServerProcess", Process)
    monkeypatch.setattr("island_engine.IslandEngine", Client)
    monkeypatch.setattr("device_profiler.get_profile", lambda: SimpleNamespace(to_dict=lambda: {}))
    engine = LlamaServerEngine(exe="dummy-server.exe")
    engine._environment = {"QLH_STATE_DIR": str(tmp_path)}
    yield engine
    engine.unload()


def test_server_environment_flags_and_status_agree(server):
    server._environment.update({switches.ENV_THREADS: "3", switches.ENV_FLASH_ATTN: "off"})
    server.load_model("dummy.gguf", n_ctx=512)
    argv = server.server.spec.to_argv(host="127.0.0.1", port=1)
    assert argv[argv.index("--threads") + 1] == "3"
    assert argv[argv.index("--flash-attn") + 1] == "off"
    state = server.get_model_info()["inference_switches"]
    assert state["effective"]["n_threads"] == 3
    assert state["effective"]["flash_attn"] is False
    server.unload()
    assert server.get_model_info()["inference_switches"] is None


def test_server_default_retains_own_auto_policy(server):
    server.load_model("dummy.gguf", n_ctx=512)
    state = server.get_model_info()["inference_switches"]
    assert state["effective"]["flash_attn"] == "server_auto"
    assert state["effective"]["n_threads"] == "server_auto"
    assert server.server.spec.inference_args == []


def test_server_explicit_arguments_win(server):
    server._environment.update({switches.ENV_THREADS: "3", switches.ENV_FLASH_ATTN: "on"})
    server.load_model("dummy.gguf", n_ctx=512, n_threads=5, flash_attn=False)
    assert server.get_model_info()["inference_switches"]["effective"]["n_threads"] == 5
    assert server.server.spec.inference_args == ["--threads", "5", "--flash-attn", "off"]


def test_server_final_kv_conflict_is_rejected(server):
    with pytest.raises(ValueError, match="quantized_kv_requires_flash_attn"):
        server.load_model("dummy.gguf", n_ctx=512, type_v="q8_0", flash_attn=False)
    assert server.server is None


def test_server_explicit_quantized_v_enables_flash(server):
    server.load_model("dummy.gguf", n_ctx=512, type_v="q8_0")
    assert server.server.spec.inference_args == ["--cache-type-v", "q8_0", "--flash-attn", "on"]


@pytest.mark.parametrize("preferences", [
    {switches.ENV_KV_CACHE: "q8_0"}, {switches.ENV_FLASH_ATTN: "off"},
    {switches.ENV_FLASH_ATTN: "on"},
])
def test_frozen_native_contract_rejects_unavailable_tuning(monkeypatch, tmp_path, preferences):
    from qlh_core.contracts.load import RuntimeLoadSpec
    path = tmp_path / "segment.gguf"
    path.write_bytes(b"partial artifact")
    engine = LlamaCppEngine()
    engine._environment = preferences
    engine._model_path = str(path)
    monkeypatch.setattr(engine, "_find_layer_artifact", lambda *args: str(path))
    monkeypatch.setattr(engine, "_verify_catalog_artifact", lambda *args, **kwargs: None)
    spec = RuntimeLoadSpec(device="cpu", dtype="artifact", quant_type="artifact", context_size=512,
        kv_device="cpu", kv_dtype="float16", kv_policy="preallocated", components=("layers",),
        materialization="gguf_segment")
    with pytest.raises(ValueError, match="deployment_inference_switch_unsupported"):
        engine.load_layer_range(1, 2, load_spec=spec.to_dict())
    assert engine._inference_switches is None


def test_actual_native_thread_argument_uses_selected_value(monkeypatch, tmp_path):
    import llama_keep_head
    path = tmp_path / "segment.gguf"
    path.write_bytes(b"partial artifact")
    engine = LlamaCppEngine()
    engine._model_path = str(path)
    engine._pipeline_descriptor = {"loaded_artifact": str(path), "assignment_layer_range": [1, 2]}
    engine._inference_switches = {"effective": {"n_threads": 3}}
    seen = []
    monkeypatch.setattr(llama_engine, "resolve_keep_head_shim", lambda: str(tmp_path / "shim.dll"))
    def native(*args, **kwargs):
        seen.append(kwargs)
        return SimpleNamespace(is_available=True, close=lambda: None)
    monkeypatch.setattr(llama_keep_head, "KeepHeadUpstream", native)
    assert engine._keep_head_upstream() is not None
    assert seen[0]["n_threads"] == 3
    engine.unload()


def test_default_context_respects_explicit_flash_over_shadowed_environment(binding, monkeypatch):
    engine, seen = binding
    engine._environment.update({switches.ENV_KV_CACHE: "q8_0", switches.ENV_FLASH_ATTN: "off"})
    monkeypatch.setattr(switches, "suggest_n_ctx_from_gguf", lambda *args, **kwargs: 512)
    engine._load_model_runtime("dummy.gguf", flash_attn=True)
    assert seen[0]["flash_attn"] is True
    assert seen[0]["n_ctx"] == 512


def test_server_context_batch_flags_are_actually_applied(server):
    server.load_model("dummy.gguf", n_ctx=512, n_batch=128, n_ubatch=64, n_seq_max=2)
    argv = server.server.spec.to_argv(host="127.0.0.1", port=1)
    assert argv[argv.index("--batch-size") + 1] == "128"
    assert argv[argv.index("--ubatch-size") + 1] == "64"
    assert argv[argv.index("--parallel") + 1] == "2"


def test_fresh_cuda_budget_keeps_physical_uuid_and_resource_domain(monkeypatch):
    import sys
    fake = SimpleNamespace(cuda=SimpleNamespace(is_available=lambda: True, current_device=lambda: 0))
    monkeypatch.setitem(sys.modules, "torch", fake)
    monkeypatch.setattr("cuda_memory_budget.sample_cuda_memory_budget", lambda runtime, device: {
        "available_bytes": 3 * 1024**3, "budget_source": "cuda.min_mem_get_info_nvidia_smi_uuid",
        "budget_observations": {"device_uuid": "962bd173-a5f8-4f61-9840-05d364605846"}})
    profile = switches._fresh_context_profile("cuda:0")
    sample = profile["gpus"][0]["memory_sample"]
    assert sample["device"] == "cuda:0"
    assert sample["free_bytes"] == 3 * 1024**3
    assert sample["device_uuid"] == profile["gpus"][0]["device_uuid"]
    assert "ram" not in profile


def test_server_auto_context_observes_sequence_count(server, monkeypatch):
    seen = {}
    def context(path, **kwargs):
        seen.update(kwargs)
        kwargs["observation"].update(reason="ok", n_ctx=1024)
        return 1024
    monkeypatch.setattr(llama_engine, "_default_n_ctx", context)
    server.load_model("dummy.gguf", n_gpu_layers=0, n_seq_max=2)
    assert seen["n_seq_max"] == 2
    assert seen["execution_device"] == "cpu"
    state = server.get_model_info()["inference_switches"]
    assert state["effective"]["n_seq_max"] == 2
    assert state["effective"]["n_ctx"] == 1024
    assert state["sources"]["n_seq_max"] == "call_argument"
    assert state["sources"]["n_ctx"] == "context_budget"
    assert state["argv"][-2:] == ["--parallel", "2"]


def test_server_kv_source_uses_public_effective_key(server):
    server._environment[switches.ENV_KV_CACHE] = "q8_0"
    server.load_model("dummy.gguf", n_ctx=512)
    state = server.get_model_info()["inference_switches"]
    assert state["sources"]["type_k"] == "environment"
    assert state["sources"]["type_v"] == "environment"
    assert state["sources"]["n_ctx"] == "call_argument"


def test_binding_explicit_context_and_kv_dependency_sources(binding):
    engine, _ = binding
    load(engine, type_k="q8_0")
    state = engine.get_model_info()["inference_switches"]
    assert state["sources"]["type_k"] == "call_argument"
    assert state["sources"]["flash_attn"] == "kv_dependency"
    assert state["sources"]["n_ctx"] == "call_argument"


@pytest.mark.parametrize("materialized", [False, True])
def test_current_model_projection_keeps_owner_state_and_cold_read(monkeypatch, materialized):
    from api import routes_models
    owner_state = {"scope": "loaded_model", "execution_path": "llama_keep_head",
                   "effective": {"n_threads": 3}}
    class Manager:
        is_loaded = materialized
        def get_pipeline_descriptor(self):
            return {}
        def get_model_info(self):
            assert materialized, "cold query must not construct a native model"
            return {"inference_switches": owner_state}
    api = SimpleNamespace(_pipeline_model_is_prepared=lambda: True,
        _node_inference_mode=lambda: "distributed", model_manager=Manager(),
        model_host=SimpleNamespace(current_quant="gguf"))
    monkeypatch.setattr(routes_models, "_api_module", api)
    projected = routes_models._current_model_payload()
    assert projected["inference_switches"] == (owner_state if materialized else None)
    assert projected["segment_materialized"] is materialized



@pytest.mark.parametrize("options,expected_domain", [
    ({"n_gpu_layers": 99, "offload_kqv": False}, "cpu"),
    ({"n_gpu_layers": 99}, "cuda:0"),
    ({"n_gpu_layers": 99, "tensor_split": [0.5, 0.5]}, "multi_gpu"),
])
def test_binding_context_uses_cache_resource_domain(binding, monkeypatch, options, expected_domain):
    import llama_cpp
    engine, _ = binding
    monkeypatch.setattr(llama_cpp, "llama_supports_gpu_offload", lambda: True)
    seen = {}
    def context(path, **kwargs):
        seen.update(kwargs)
        return 512
    monkeypatch.setattr(llama_engine, "_default_n_ctx", context)
    engine._load_model_runtime("dummy.gguf", **options)
    assert seen["execution_device"] == expected_domain
