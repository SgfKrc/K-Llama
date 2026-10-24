import asyncio
import os
import sys
import types
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import pytest
from fastapi.testclient import TestClient

import api_server
from model_host import model_host
import model_config as mc
from model_module import ModelManager
from worker_assignment_state import WorkerAssignmentRegistry


def test_pipeline_capacity_endpoint_returns_scheduler_plan(monkeypatch):
    expected = {
        "status": "admitted",
        "admitted": True,
        "plan_id": "plan-test",
        "assignments": [],
    }
    monkeypatch.setattr(
        api_server.scheduler,
        "get_pipeline_capacity_plan",
        lambda: dict(expected),
    )

    response = TestClient(api_server.app).get("/api/cluster/pipeline-capacity")

    assert response.status_code == 200
    assert response.json() == expected


def test_pipeline_reshard_endpoint_returns_scheduler_projection(monkeypatch):
    expected = {
        "status": "staged",
        "epoch": 3,
        "active_contract_sha256": "a" * 64,
        "staged": [],
        "last_decision": {"reason_code": "pipeline_reshard_assets_required"},
    }
    monkeypatch.setattr(
        api_server.scheduler,
        "get_pipeline_reshard_status",
        lambda: dict(expected),
    )

    response = TestClient(api_server.app).get("/api/cluster/pipeline-reshard")

    assert response.status_code == 200
    assert response.json() == expected


def test_pipeline_assignment_file_supports_http_range(monkeypatch, tmp_path):
    model_file = tmp_path / "model.safetensors"
    model_file.write_bytes(b"0123456789")
    monkeypatch.setattr(api_server, "_require_trusted_model_peer", lambda _request: None)
    monkeypatch.setattr(api_server, "_active_pytorch_model", lambda: {
        "model_id": "fixture",
        "model_path": str(tmp_path),
        "model_sha256": "a" * 64,
        "total_layers": 1,
    })
    response = TestClient(api_server.app).get(
        "/api/models/files/fixture/model.safetensors",
        headers={"Range": "bytes=4-"},
    )
    assert response.status_code == 206
    assert response.headers["content-range"] == "bytes 4-9/10"
    assert response.content == b"456789"


def _both_model(tmp_path, model_id="custom-both"):
    hf_dir = tmp_path / f"{model_id}-hf"
    hf_dir.mkdir()
    (hf_dir / "config.json").write_text("{}", encoding="utf-8")
    (hf_dir / "model.safetensors").write_text("x", encoding="utf-8")
    gguf = tmp_path / f"{model_id}.Q4_K_M.gguf"
    gguf.write_text("x", encoding="utf-8")
    return mc.ModelConfig(
        model_id=model_id,
        name="Custom Both",
        model_type="both",
        model_path=str(hf_dir),
        gguf_path=str(gguf),
        is_experimental=True,
        quant_types=["fp16", "int4", "Q4_K_M"],
    )


def _resolved_load(
    *,
    model_id,
    engine="pytorch",
    model_path="models/test-hf",
    quant_type="int4",
    layer_range_mode="dynamic",
):
    return api_server.ModelLoadResolution(
        model_id=model_id,
        requested_engine=engine,
        engine=engine,
        artifact_kind="gguf" if engine == "llama_cpp" else "safetensors",
        model_path=model_path,
        requested_quant=quant_type,
        quant_type=quant_type,
        runtime="llama_cpp_native" if engine == "llama_cpp" else "pytorch_cuda",
        runtime_quant=quant_type,
        layer_range_mode=layer_range_mode,
        reason_code="TEST_RESOLUTION",
        reason="test fixture",
    )


class _TemplateTokenizer:
    def __init__(self, template_name):
        self.template_name = template_name

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True):
        assert tokenize is False
        assert add_generation_prompt is True
        user = next(m["content"] for m in messages if m["role"] == "user")
        if self.template_name == "deepseek":
            return f"<｜begin▁of▁sentence｜><｜User｜>{user}<｜Assistant｜><think>\n"
        return f"<|im_start|>user\n{user}<|im_end|>\n<|im_start|>assistant\n"


def test_build_model_chat_prompt_uses_deepseek_native_template():
    prompt = api_server._build_model_chat_prompt(
        _TemplateTokenizer("deepseek"),
        [{"role": "user", "content": "Ubuntu和Debian有什么区别？"}],
    )

    assert "<｜User｜>Ubuntu和Debian有什么区别？<｜Assistant｜><think>" in prompt
    assert "<|im_start|>" not in prompt


def test_build_model_chat_prompt_keeps_qwen_chatml_template():
    prompt = api_server._build_model_chat_prompt(
        _TemplateTokenizer("qwen"),
        [{"role": "user", "content": "Ubuntu和Debian有什么区别？"}],
    )

    assert "<|im_start|>user" in prompt
    assert "<｜User｜>" not in prompt


def test_build_model_chat_prompt_does_not_mix_prefill_with_native_think():
    prompt = api_server._build_model_chat_prompt(
        _TemplateTokenizer("deepseek"),
        [{"role": "user", "content": "test"}],
        assistant_prefill="【思考】\n",
    )

    assert prompt.endswith("<think>\n")
    assert "【思考】" not in prompt


def test_unclosed_native_thinking_is_not_exposed_as_answer():
    answer, thinking = api_server._format_model_response(
        "internal reasoning without a closing tag",
        show_thinking=False,
        native_thinking_prompt=True,
    )

    assert answer == ""
    assert thinking is None


def test_model_payload_prefers_pytorch_when_cuda_has_both_formats(tmp_path, monkeypatch):
    monkeypatch.setattr(api_server.mc, "is_cuda_available", lambda: True)
    monkeypatch.setattr(api_server, "cuda_available", lambda load=False: True)
    payload = api_server._model_api_payload(_both_model(tmp_path))

    assert payload["supported_engines"] == ["llama_cpp", "pytorch"]
    assert payload["preferred_engine"] == "pytorch"
    assert payload["default_quant_type"] == "int4"


def test_model_payload_prefers_llama_cpp_without_cuda_when_both_formats_exist(tmp_path, monkeypatch):
    monkeypatch.setattr(api_server.mc, "is_cuda_available", lambda: False)
    monkeypatch.setattr(api_server, "cuda_available", lambda load=False: False)
    payload = api_server._model_api_payload(_both_model(tmp_path))

    assert payload["supported_engines"] == ["llama_cpp", "pytorch"]
    assert payload["preferred_engine"] == "llama_cpp"
    assert payload["default_quant_type"] == "Q4_K_M"


def test_available_models_scans_all_registered_model_formats(tmp_path, monkeypatch):
    import config as runtime_config

    cfg = _both_model(tmp_path)
    monkeypatch.setattr(api_server, "_get_all_model_configs", lambda: [cfg])
    monkeypatch.setattr(api_server.mc, "is_cuda_available", lambda: True)
    monkeypatch.setattr(api_server.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(model_host, "model_loaded", False)
    # This test isolates local artifact scanning. A developer's .env may enable
    # optional external routing, which is orthogonal to the asserted engine set.
    monkeypatch.setattr(runtime_config, "EXTERNAL_ENABLED", False)
    monkeypatch.setattr(runtime_config, "EXTERNAL_BASE_URL", "")

    result = asyncio.run(api_server.list_available_models())
    engine_ids = {engine["id"] for engine in result["available_engines"]}

    assert engine_ids == {"llama_cpp", "pytorch"}
    assert result["current"] is None
    assert result["current_engine"] is None


class TestChatMlStopHandling:
    """Regression tests for ChatML leakage in local PyTorch generation."""

    class _FakeTokenizer:
        eos_token_id = 2
        unk_token_id = 0

        def __init__(self, decoded=""):
            self.decoded = decoded

        def convert_tokens_to_ids(self, token):
            return {
                "<|im_end|>": 10,
                "<|im_start|>": 11,
            }.get(token, self.unk_token_id)

        def encode(self, text, add_special_tokens=False):
            mapping = {
                "<|im_end|>": [10],
                "<|im_start|>": [11],
                "<not-a-real-token>": [self.unk_token_id],
            }
            return mapping.get(text, [99])

        def decode(self, ids, skip_special_tokens=False):
            return self.decoded

    def test_decode_generated_ids_cuts_at_chatml_stop(self):
        mgr = ModelManager()
        mgr.tokenizer = self._FakeTokenizer(
            "Ubuntu 和 Debian 的区别如下。<|im_end|><|im_start|>user\n下一轮"
        )

        text = mgr._decode_generated_ids([1, 2, 3], mgr._merge_stop_sequences(None))

        assert text == "Ubuntu 和 Debian 的区别如下。"

    def test_unknown_stop_token_is_not_used_as_eos_id(self):
        mgr = ModelManager()
        mgr.tokenizer = self._FakeTokenizer()

        eos_ids = mgr._get_generation_eos_token_ids(["<not-a-real-token>"])

        assert eos_ids == 2

    def test_stream_stop_filter_preserves_chunk_boundary_spaces(self):
        mgr = ModelManager()
        chunks = ["Hello", " world", "<|im_end|>", "ignored"]

        text = "".join(mgr._iter_stream_until_stop(chunks, mgr._merge_stop_sequences(None)))

        assert text == "Hello world"


def test_switch_model_accepts_external_model_path_not_in_builtin_registry(monkeypatch):
    calls = []

    def fake_load_model(self, **kwargs):
        calls.append(kwargs)
        self._active_model_id = kwargs["model_id"]
        self._engine_type = kwargs["engine"]
        self._model_path = kwargs["model_path"]

    monkeypatch.setattr(ModelManager, "load_model", fake_load_model)

    mgr = ModelManager()
    result = mgr.switch_model(
        "db-custom-gguf",
        quant_type="Q4_K_M",
        engine="llama_cpp",
        model_path="models/custom.Q4_K_M.gguf",
    )

    assert result["success"] is True
    assert result["model_id"] == "db-custom-gguf"
    assert calls[0]["model_path"] == "models/custom.Q4_K_M.gguf"


def test_api_switch_model_resolves_db_registered_gguf(tmp_path, monkeypatch):
    monkeypatch.setattr(api_server.scheduler, "_distributed_inference_enabled", False)
    gguf = tmp_path / "custom.Q4_K_M.gguf"
    gguf.write_text("x", encoding="utf-8")
    db_entry = {
        "model_id": "db-custom-gguf",
        "name": "DB Custom GGUF",
        "model_type": "gguf",
        "model_path": "",
        "gguf_path": str(gguf),
        "recommended_vram_gb": 4.0,
        "max_context": 4096,
        "quant_types": ["Q4_K_M"],
        "description": "test",
    }
    calls = []

    class FakeManager:
        active_model_id = ""
        is_loaded = False

        def switch_model(self, **kwargs):
            calls.append(kwargs)
            self.active_model_id = kwargs["model_id"]
            self.is_loaded = True
            return {
                "success": True,
                "model_id": kwargs["model_id"],
                "model_name": "DB Custom GGUF",
                "error": None,
            }

    monkeypatch.setattr(api_server, "_get_registered_experimental_models", lambda: [db_entry])
    monkeypatch.setattr(api_server.mc, "is_cuda_available", lambda: False)
    monkeypatch.setattr(api_server, "model_manager", FakeManager())
    monkeypatch.setattr(api_server, "kv_cache", None)
    monkeypatch.setattr(model_host, "model_loaded", False)
    monkeypatch.setattr(api_server, "_init_kv_cache", lambda: None)

    req = api_server.SwitchModelRequest(
        model_id="db-custom-gguf",
        quant_type="Q4_K_M",
        engine="auto",
    )
    result = asyncio.run(api_server.switch_model(req))

    assert result["success"] is True
    assert calls[0]["engine"] == "llama_cpp"
    assert calls[0]["model_path"] == str(gguf)


def test_model_payload_marks_builtin_slots(monkeypatch):
    monkeypatch.setattr(api_server.mc, "is_cuda_available", lambda: False)

    builtin_payload = api_server._model_api_payload(
        mc.get_builtin_model("deepseek-r1-distill-qwen-7b")
    )
    custom_payload = api_server._model_api_payload(
        mc.ModelConfig(
            model_id="custom-gguf",
            name="Custom GGUF",
            model_type="gguf",
            gguf_path="models/custom.gguf",
            is_experimental=True,
        )
    )

    assert builtin_payload["is_builtin"] is True
    assert custom_payload["is_builtin"] is False


def test_local_assets_endpoint_returns_discovery_inventory(monkeypatch):
    import local_model_assets

    expected = {
        "assets": [{"model_id": "qwen3-4b"}],
        "summary": {"total": 1, "total_bytes": 1},
    }
    monkeypatch.setattr(
        local_model_assets, "discover_local_model_assets", lambda **_kwargs: expected
    )

    assert asyncio.run(api_server.list_local_model_assets()) == expected


def test_local_asset_preflight_delegates_to_read_only_inventory_gate(monkeypatch):
    import local_model_assets

    expected = {"model_id": "qwen3-4b", "gate_passed": True, "status": "ready_for_qwen3_smoke"}
    monkeypatch.setattr(local_model_assets, "preflight_local_model_asset", lambda model_id: expected)

    assert asyncio.run(api_server.preflight_local_model_asset("qwen3-4b")) == expected


def test_register_gguf_model_is_allowed_without_cuda(monkeypatch, tmp_path):
    saved = []
    monkeypatch.setattr(
        api_server._local_store,
        "publish_local_model_registry",
        lambda model_id, config_json, *, expected_revision: saved.append((model_id, config_json)) or True,
    )
    monkeypatch.setattr(api_server.mc, "is_cuda_available", lambda: False)
    gguf = tmp_path / "custom.Q4_K_M.gguf"
    gguf.write_bytes(b"fake-gguf")

    req = api_server.RegisterModelRequest(
        model_id="custom-gguf",
        name="Custom GGUF",
        model_type="gguf",
        gguf_path=str(gguf),
    )

    result = asyncio.run(api_server.register_model(req))

    assert result == {"status": "registered", "model_id": "custom-gguf"}
    assert saved and saved[0][0] == "custom-gguf"


def test_register_safetensors_model_without_cuda_is_allowed(monkeypatch, tmp_path):
    saved = []
    monkeypatch.setattr(
        api_server._local_store,
        "publish_local_model_registry",
        lambda model_id, config_json, *, expected_revision: saved.append((model_id, config_json)) or True,
    )
    monkeypatch.setattr(api_server.mc, "is_cuda_available", lambda: False)
    model_dir = tmp_path / "custom-hf"
    model_dir.mkdir()
    (model_dir / "config.json").write_text("{}", encoding="utf-8")
    (model_dir / "model.safetensors").write_bytes(b"fake-weights")

    req = api_server.RegisterModelRequest(
        model_id="custom-hf",
        name="Custom HF",
        model_type="safetensors",
        model_path=str(model_dir),
    )

    result = asyncio.run(api_server.register_model(req))

    assert result == {"status": "registered", "model_id": "custom-hf"}
    assert saved and saved[0][0] == "custom-hf"


def test_register_model_rejects_empty_or_missing_assets(monkeypatch, tmp_path):
    from fastapi import HTTPException
    monkeypatch.setattr(api_server._local_store, "save_local_experimental_model", lambda *_args: True)
    req = api_server.RegisterModelRequest(
        model_id="missing-hf",
        name="Missing HF",
        model_type="safetensors",
        model_path=str(tmp_path / "missing"),
    )
    with pytest.raises(HTTPException) as exc_info:
        asyncio.run(api_server.register_model(req))
    assert exc_info.value.status_code == 400


# ================================================================
# /api/models/load 端点测试
# ================================================================


def test_exclusive_local_model_change_opts_out_pipeline_worker(monkeypatch):
    calls = []
    assignments = WorkerAssignmentRegistry()
    assignments.begin(
        "worker",
        expected_config={"config_id": "cfg-old", "generation": 7},
    )
    retry_state = {"worker": {"attempts": 2, "next_retry": 10.0}}
    transaction = {
        "config_id": "cfg-old",
        "phase": "ready",
        "plan": {"admitted": True, "plan_id": "plan-old"},
    }
    active_plan = {"admitted": True, "plan_id": "plan-old"}
    monkeypatch.setattr(api_server.scheduler, "_worker_assignments", assignments)
    monkeypatch.setattr(api_server.scheduler, "_layer_config_retry_state", retry_state)
    monkeypatch.setattr(api_server.scheduler, "_pipeline_load_transaction", transaction)
    monkeypatch.setattr(api_server.scheduler, "_active_pipeline_capacity_plan", active_plan)
    monkeypatch.setattr(
        api_server.scheduler,
        "release_pipeline_worker_for_local_model",
        lambda: calls.append("released") or True,
    )
    monkeypatch.setattr(api_server, "_refresh_pipeline_layer_config", lambda: None)

    result = api_server._run_exclusive_model_change(
        lambda: {"success": True},
        release_worker_reservation=True,
    )

    assert result == {"success": True}
    assert calls == ["released"]
    assert assignments.snapshot() == {}
    assert retry_state == {}
    assert api_server.scheduler._pipeline_load_transaction is None
    assert api_server.scheduler._active_pipeline_capacity_plan is None


def test_failed_exclusive_model_change_preserves_active_control_state(monkeypatch):
    calls = []
    assignments = WorkerAssignmentRegistry()
    assignments.begin(
        "worker",
        expected_config={"config_id": "cfg-old", "generation": 7},
    )
    assignment_snapshot = assignments.snapshot()
    retry_state = {"worker": {"attempts": 2, "next_retry": 10.0}}
    transaction = {
        "config_id": "cfg-old",
        "phase": "ready",
        "plan": {"admitted": True, "plan_id": "plan-old"},
    }
    active_plan = {"admitted": True, "plan_id": "plan-old"}
    active_config = {"config_id": "cfg-old"}
    monkeypatch.setattr(api_server.scheduler, "_worker_assignments", assignments)
    monkeypatch.setattr(api_server.scheduler, "_layer_config_retry_state", retry_state)
    monkeypatch.setattr(api_server.scheduler, "_pipeline_load_transaction", transaction)
    monkeypatch.setattr(api_server.scheduler, "_active_pipeline_capacity_plan", active_plan)
    monkeypatch.setattr(
        api_server.scheduler, "_active_layer_config", dict(active_config),
    )
    monkeypatch.setattr(
        api_server.scheduler,
        "release_pipeline_worker_for_local_model",
        lambda: calls.append("released") or True,
    )
    monkeypatch.setattr(
        api_server, "_refresh_pipeline_layer_config",
        lambda: calls.append("refreshed"),
    )

    result = api_server._run_exclusive_model_change(
        lambda: {"success": False, "error": "load failed"},
        prepare=lambda: calls.append("prepared"),
        release_worker_reservation=True,
    )

    assert result["success"] is False
    assert assignments.snapshot() == assignment_snapshot
    assert retry_state == {"worker": {"attempts": 2, "next_retry": 10.0}}
    assert api_server.scheduler._pipeline_load_transaction == transaction
    assert api_server.scheduler._active_pipeline_capacity_plan == active_plan
    assert api_server.scheduler._active_layer_config == active_config
    assert calls == []


def test_master_model_refresh_requests_authoritative_layer_sync(monkeypatch):
    """主节点加载模型后应以权威配置重新接管在线分层 worker。"""
    calls = []
    monkeypatch.setattr(api_server.scheduler, "_effective_role", lambda: "master")
    monkeypatch.setattr(
        api_server.scheduler,
        "request_authoritative_layer_sync",
        lambda: calls.append("authoritative") or True,
    )
    monkeypatch.setattr(
        api_server.scheduler,
        "push_layer_config_to_clients",
        lambda: pytest.fail("新调度器不应退回普通层配置推送"),
    )

    api_server._refresh_pipeline_layer_config()

    assert calls == ["authoritative"]


def test_load_model_rejects_invalid_engine(monkeypatch):
    """无效 engine 参数 → 400"""
    monkeypatch.setattr(api_server.mc, "is_cuda_available", lambda: True)
    req = api_server.LoadModelRequest(
        engine="invalid_engine",
        quant_type="int4",
    )
    with pytest.raises(api_server.HTTPException) as exc:
        asyncio.run(api_server.load_model(req))
    assert exc.value.status_code == 400


def test_load_model_rejects_invalid_quant_for_pytorch(monkeypatch):
    """PyTorch 引擎 + 无效 quant → 400"""
    monkeypatch.setattr(api_server.mc, "is_cuda_available", lambda: True)
    req = api_server.LoadModelRequest(
        engine="pytorch",
        quant_type="q4_k_m",  # GGUF 量化，PyTorch 不支持
    )
    with pytest.raises(api_server.HTTPException) as exc:
        asyncio.run(api_server.load_model(req))
    assert exc.value.status_code == 400


@pytest.mark.skipif(
    not os.path.isdir(os.path.join(os.path.dirname(__file__), "..", "models", "qwen3-0.6b")),
    reason="缺本地工件 models/qwen3-0.6b（CI 不拉模型）",
)
def test_normalize_quant_marks_gguf_precision_as_artifact_owned():
    """llama_cpp 的 effective quant 不得伪装成请求的 PyTorch dtype。"""
    api_server._validate_model_load_request(None, "llama_cpp")
    assert api_server._normalize_quant_for_engine("Q4_K_M", "llama_cpp") == "gguf"
    with pytest.raises(api_server.HTTPException) as exc:
        api_server._normalize_quant_for_engine("Q4_K_M", "pytorch")
    assert exc.value.status_code == 400


def test_load_model_auto_uses_real_gguf_model_resolution(monkeypatch, tmp_path):
    monkeypatch.setattr(api_server.scheduler, "_distributed_inference_enabled", False)
    """engine=auto 解析到 llama.cpp 时，GGUF 量化占位值应合法。"""
    switch_calls = []

    class FakeManager:
        active_model_id = ""
        is_loaded = False
        _engine_type = "llama_cpp"
        quant_type = None

        def switch_model(self, **kwargs):
            switch_calls.append(kwargs)
            self.is_loaded = True
            self.active_model_id = kwargs["model_id"]
            self.quant_type = kwargs["quant_type"]
            return {
                "success": True,
                "model_id": self.active_model_id,
                "model_name": "GGUF Model",
                "error": None,
            }

    async def fake_get_status():
        return {"status": "ok"}

    gguf_path = tmp_path / "gguf-model.Q4_K_M.gguf"
    gguf_path.write_bytes(b"gguf")
    registered_model = {
        "model_id": "gguf-model",
        "name": "GGUF Model",
        "model_type": "gguf",
        "gguf_path": str(gguf_path),
        "quant_types": ["Q4_K_M"],
    }

    import config as runtime_config

    monkeypatch.setattr(api_server, "model_manager", FakeManager())
    monkeypatch.setattr(api_server, "_get_registered_experimental_models", lambda: [registered_model])
    monkeypatch.setattr(api_server, "_init_kv_cache", lambda: None)
    monkeypatch.setattr(api_server, "get_status", fake_get_status)
    monkeypatch.setattr(model_host, "model_loaded", False)
    monkeypatch.setattr(runtime_config, "INFERENCE_ENGINE", "auto")
    monkeypatch.setattr(runtime_config, "QUANT_TYPE", "int4")
    monkeypatch.setattr(runtime_config, "USE_COMPILE", False)

    req = api_server.LoadModelRequest(
        engine="auto",
        quant_type="Q4_K_M",
        model_id="gguf-model",
    )
    result = asyncio.run(api_server.load_model(req))

    assert result["status"] == "ok"
    assert switch_calls[0]["engine"] == "llama_cpp"
    assert switch_calls[0]["quant_type"] == "gguf"
    assert switch_calls[0]["resolution"].requested_quant == "Q4_K_M"
    assert switch_calls[0]["model_path"] == str(gguf_path)


def test_api_auto_load_safetensors_only_uses_resolved_fp32(monkeypatch, tmp_path):
    monkeypatch.setattr(api_server.scheduler, "_distributed_inference_enabled", False)
    import config as runtime_config

    model_dir = tmp_path / "safetensors-only"
    model_dir.mkdir()
    calls = []

    class FakeManager:
        quant_type = None

        def load_model(self, **kwargs):
            calls.append(dict(kwargs))
            self.quant_type = kwargs["quant_type"]

    manager = FakeManager()
    monkeypatch.setattr(api_server, "model_manager", manager)
    monkeypatch.setattr(api_server, "_resolution_cuda_available", lambda: False)
    monkeypatch.setattr(api_server, "_init_kv_cache", lambda: None)
    monkeypatch.setattr(
        api_server,
        "_run_exclusive_model_change",
        lambda operation, **_kwargs: operation(),
    )
    monkeypatch.setattr(
        api_server.scheduler,
        "refresh_task_worker_capabilities",
        lambda: None,
    )
    monkeypatch.setattr(runtime_config, "get_active_model_paths", lambda: {})
    monkeypatch.setattr(runtime_config, "GGUF_MODEL_PATH", str(tmp_path / "missing.gguf"))
    monkeypatch.setattr(runtime_config, "MODEL_PATH", str(model_dir))
    monkeypatch.setattr(runtime_config, "PREFER_PYTORCH", False)
    monkeypatch.setattr(runtime_config, "QUANT_TYPE", "int4")

    api_server._auto_load_default_model()

    assert len(calls) == 1
    assert calls[0]["engine"] == "pytorch"
    assert calls[0]["model_path"] == str(model_dir)
    assert calls[0]["quant_type"] == "fp32"
    assert calls[0]["resolution"].runtime == "pytorch_cpu"


def test_api_auto_load_never_rebinds_active_id_to_global_artifact(
    monkeypatch, tmp_path,
):
    monkeypatch.setattr(api_server.scheduler, "_distributed_inference_enabled", False)
    import config as runtime_config

    foreign = tmp_path / "foreign.gguf"
    foreign.write_bytes(b"gguf")
    calls = []

    class FakeManager:
        def load_model(self, **kwargs):
            calls.append(dict(kwargs))

    monkeypatch.setattr(api_server, "model_manager", FakeManager())
    monkeypatch.setattr(api_server, "_resolution_cuda_available", lambda: False)
    monkeypatch.setattr(
        runtime_config,
        "get_active_model_paths",
        lambda: {
            "model_id": "model-a",
            "gguf_path": str(tmp_path / "missing-a.gguf"),
            "model_path": str(tmp_path / "missing-a"),
        },
    )
    monkeypatch.setattr(runtime_config, "GGUF_MODEL_PATH", str(foreign))
    monkeypatch.setattr(runtime_config, "MODEL_PATH", str(tmp_path / "foreign-safe"))

    with pytest.raises(FileNotFoundError, match="未找到可自动加载的模型文件"):
        api_server._auto_load_default_model()
    assert calls == []


def test_load_model_rejects_nonexistent_model_id(monkeypatch):
    """不存在的 model_id → 404"""
    monkeypatch.setattr(api_server.mc, "is_cuda_available", lambda: True)
    monkeypatch.setattr(api_server, "_get_registered_experimental_models", lambda: [])
    req = api_server.LoadModelRequest(
        engine="auto",
        quant_type="int4",
        model_id="nonexistent-model-xyz",
    )
    with pytest.raises(api_server.HTTPException) as exc:
        asyncio.run(api_server.load_model(req))
    assert exc.value.status_code == 404


@pytest.mark.skipif(
    not os.path.isdir(os.path.join(os.path.dirname(__file__), "..", "models", "qwen3-0.6b")),
    reason="缺本地工件 models/qwen3-0.6b（CI 不拉模型）",
)
def test_load_model_uses_switch_model_internally(monkeypatch):
    monkeypatch.setattr(api_server.scheduler, "_distributed_inference_enabled", False)
    """验证 /api/models/load 使用 switch_model（B2 修复）"""
    switch_calls = []
    load_mode_calls = []

    class FakeManager:
        active_model_id = ""
        is_loaded = False
        _engine_type = ""
        quant_type = None

        def switch_model(self, **kwargs):
            switch_calls.append(kwargs)
            self.is_loaded = True
            self.active_model_id = kwargs.get("model_id", "")
            self._engine_type = kwargs.get("engine") or "llama_cpp"
            self.quant_type = kwargs.get("quant_type")
            return {
                "success": True,
                "model_id": self.active_model_id,
                "model_name": "Test Model",
                "error": None,
            }

    monkeypatch.setattr(api_server, "model_manager", FakeManager())
    monkeypatch.setattr(model_host, "model_loaded", False)
    monkeypatch.setattr(api_server, "kv_cache", None)
    monkeypatch.setattr(api_server, "_init_kv_cache", lambda: None)
    monkeypatch.setattr(api_server.mc, "is_cuda_available", lambda: True)
    monkeypatch.setattr(
        api_server,
        "_decide_pipeline_load_mode",
        lambda model_id, engine, quant: (
            load_mode_calls.append((model_id, engine, quant)) or ("full", "test")
        ),
    )

    async def fake_get_status():
        return {"status": "ok"}

    monkeypatch.setattr(api_server, "get_status", fake_get_status)

    req = api_server.LoadModelRequest(
        engine="auto",
        quant_type="int4",
    )
    result = asyncio.run(api_server.load_model(req))
    assert len(switch_calls) == 1, "应调用 switch_model 而非手动 unload/load"
    # ★ 2026-09-19：未指定 model_id 时按**设备画像**取默认（不再是固定常量）。
    default_model_id = api_server.mc.get_profile_default_model_id()
    assert switch_calls[0]["model_id"] == default_model_id
    assert load_mode_calls[0][0] == default_model_id
    assert result["status"] == "ok"


@pytest.fixture
def real_api_services():
    # Resolve the owning runtime before an API-only host stub replaces the facade.
    return api_server._get_app_runtime().services


def test_prepare_pipeline_model_does_not_mark_full_model_loaded(monkeypatch, tmp_path, real_api_services):
    calls = []
    transactions = []
    import config as cfg

    monkeypatch.setattr(cfg, "INFERENCE_ENGINE", "old-engine")
    monkeypatch.setattr(cfg, "QUANT_TYPE", "old-quant")

    def prepare_candidate(transaction_id, **kwargs):
        calls.append((transaction_id, kwargs))
        return {
            "schema_version": 1,
            "inspection_mode": "safetensors_headers_only",
            "model_id": kwargs["model_id"],
            "model_type": "qwen2",
            "total_layers": 4,
            "hidden_size": 128,
            "tokenizer_sha256": "d" * 64,
            "weight_bytes": 1024,
            "pipeline_runtime_supported": True,
            "model_sha256": "c" * 64,
        }

    fake_host = types.SimpleNamespace(
        model_loaded=True,
        current_quant="old-quant",
        prepare_pipeline_candidate=prepare_candidate,
    )
    monkeypatch.setattr(api_server, "model_host", fake_host)
    monkeypatch.setattr(
        api_server,
        "_resolve_model_load_plan",
        lambda *args, **kwargs: _resolved_load(
            model_id="tiny-qwen2",
            model_path=str(tmp_path),
            quant_type="fp16",
        ),
    )
    monkeypatch.setattr(
        api_server.scheduler,
        "refresh_task_worker_capabilities",
        lambda: True,
    )
    monkeypatch.setattr(
        api_server.scheduler,
        "start_pipeline_model_transaction",
        lambda **kwargs: transactions.append(kwargs) or {
            "admitted": True,
            "plan_id": "plan-test",
            "transaction_phase": "preparing",
            "model_preflight": {"ok": True},
        },
    )

    result = asyncio.run(api_server.prepare_pipeline_model(
        api_server.PreparePipelineModelRequest(
            model_id="tiny-qwen2",
            quant_type="fp16",
        )
    ))

    assert result["success"] is True
    assert result["loaded"] is False
    assert result["pipeline_prepared"] is True
    assert result["descriptor"]["model_type"] == "qwen2"
    assert result["model_preflight"] == {"ok": True}
    assert fake_host.model_loaded is True
    assert result["transaction_id"].startswith("modeltxn_")
    assert calls == [(result["transaction_id"], {
        "model_id": "tiny-qwen2",
        "model_path": str(tmp_path),
        "quant_type": "fp16",
    })]
    assert transactions[0]["candidate_id"] == result["transaction_id"]
    assert transactions[0]["descriptor"]["model_type"] == "qwen2"
    assert cfg.INFERENCE_ENGINE == "old-engine"
    assert cfg.QUANT_TYPE == "old-quant"

    # Scheduler invokes the callback only after ModelHost has published the
    # candidate and mirrored its effective runtime quant.
    fake_host.current_quant = "fp32"
    transactions[0]["on_commit"]()

    assert cfg.INFERENCE_ENGINE == "pytorch"
    assert cfg.QUANT_TYPE == "fp32"


def test_prepare_pipeline_model_rejects_named_model_preflight_failure(
    monkeypatch, tmp_path, real_api_services,
):
    aborted = []
    fake_host = types.SimpleNamespace(
        model_loaded=True,
        current_quant="old-quant",
        prepare_pipeline_candidate=lambda transaction_id, **kwargs: {
            "schema_version": 1,
            "inspection_mode": "safetensors_headers_only",
            "model_id": kwargs["model_id"],
            "model_type": "qwen2",
            "total_layers": 24,
            "hidden_size": 2048,
            "tokenizer_sha256": "d" * 64,
            "weight_bytes": 1024,
            "pipeline_runtime_supported": True,
            "model_sha256": "c" * 64,
        },
        abort_pipeline_candidate=(
            lambda transaction_id: aborted.append(transaction_id)
        ),
    )
    monkeypatch.setattr(api_server, "model_host", fake_host)
    monkeypatch.setattr(
        api_server,
        "_resolve_model_load_plan",
        lambda *args, **kwargs: _resolved_load(
            model_id="tiny-qwen2",
            model_path=str(tmp_path),
            quant_type="fp16",
        ),
    )
    monkeypatch.setattr(
        api_server.scheduler,
        "start_pipeline_model_transaction",
        lambda **kwargs: {
            "admitted": False,
            "reason_code": "model_preflight_hidden_size_mismatch",
            "reason": (
                "model preflight rejected model=tiny-qwen2 "
                "nodes=edge-1 missing_ranges=[8,24)"
            ),
            "missing_layer_ranges": [[8, 24]],
        },
    )

    with pytest.raises(api_server.HTTPException) as exc_info:
        asyncio.run(api_server.prepare_pipeline_model(
            api_server.PreparePipelineModelRequest(
                model_id="tiny-qwen2",
                quant_type="fp16",
            )
        ))

    assert exc_info.value.status_code == 409
    assert exc_info.value.error_code == "model_preflight_hidden_size_mismatch"
    assert "model=tiny-qwen2" in exc_info.value.detail
    assert "nodes=edge-1" in exc_info.value.detail
    assert "missing_ranges=[8,24)" in exc_info.value.detail
    assert len(aborted) == 1


def test_prepare_pipeline_model_rejects_synchronous_aborted_commit(
    monkeypatch, tmp_path, real_api_services,
):
    aborted = []
    monkeypatch.setattr(api_server, "model_host", types.SimpleNamespace(
        prepare_pipeline_candidate=lambda transaction_id, **kwargs: {
            "model_id": kwargs["model_id"],
            "model_type": "qwen3_5",
            "total_layers": 32,
            "pipeline_runtime_supported": True,
            "model_sha256": "a" * 64,
        },
        abort_pipeline_candidate=lambda transaction_id: aborted.append(transaction_id),
    ))
    monkeypatch.setattr(
        api_server,
        "_resolve_model_load_plan",
        lambda *args, **kwargs: _resolved_load(
            model_id="q35-artifacts", engine="llama_cpp",
            model_path=str(tmp_path / "head4.gguf"), quant_type="gguf",
        ),
    )
    monkeypatch.setattr(
        api_server.scheduler,
        "start_pipeline_model_transaction",
        lambda **kwargs: {
            "admitted": True,
            "transaction_phase": "aborted",
            "reason_code": "pipeline_local_commit_failed",
            "reason": "no matching local segment",
        },
    )

    with pytest.raises(api_server.HTTPException) as exc_info:
        asyncio.run(api_server.prepare_pipeline_model(
            api_server.PreparePipelineModelRequest(model_id="q35-artifacts")
        ))

    assert exc_info.value.status_code == 409
    assert exc_info.value.error_code == "pipeline_local_commit_failed"
    assert "no matching local segment" in exc_info.value.detail
    assert len(aborted) == 1


@pytest.mark.skipif(
    not os.path.isdir(os.path.join(os.path.dirname(__file__), "..", "models", "qwen3-0.6b")),
    reason="缺本地工件 models/qwen3-0.6b（CI 不拉模型）",
)
def test_load_model_reports_effective_cpu_quant(monkeypatch):
    monkeypatch.setattr(api_server.scheduler, "_distributed_inference_enabled", False)
    """CPU loader 归一为 FP32 后，API 状态不能继续上报请求的 FP16。"""
    class FakeManager:
        active_model_id = ""
        is_loaded = False
        _engine_type = "pytorch"
        quant_type = None

        def switch_model(self, **kwargs):
            self.is_loaded = True
            self.active_model_id = kwargs.get("model_id", "")
            self.quant_type = "fp32"
            return {
                "success": True,
                "model_id": self.active_model_id,
                "model_name": "CPU PyTorch Model",
                "error": None,
            }

    async def fake_get_status():
        return {"current_quant": model_host.current_quant}

    monkeypatch.setattr(api_server, "model_manager", FakeManager())
    monkeypatch.setattr(model_host, "model_loaded", False)
    monkeypatch.setattr(model_host, "current_quant", "int4")
    monkeypatch.setattr(api_server, "kv_cache", None)
    monkeypatch.setattr(api_server, "_init_kv_cache", lambda: None)
    monkeypatch.setattr(api_server, "get_status", fake_get_status)
    monkeypatch.setattr(api_server.mc, "is_cuda_available", lambda: False)
    monkeypatch.setattr(
        api_server,
        "_decide_pipeline_load_mode",
        lambda *_args: ("full", "量化上报测试固定整模路径"),
    )

    result = asyncio.run(api_server.load_model(api_server.LoadModelRequest(
        engine="pytorch",
        quant_type="fp16",
    )))

    assert model_host.current_quant == "fp32"
    assert result["current_quant"] == "fp32"


def test_load_model_rollback_on_failure(monkeypatch):
    monkeypatch.setattr(api_server.scheduler, "_distributed_inference_enabled", False)
    """加载失败时 switch_model 应触发回滚，/api/models/load 应正确报告"""
    class FakeManager:
        active_model_id = "previous-model"
        is_loaded = True
        _engine_type = "llama_cpp"
        quant_type = "Q4_K_M"
        model = None  # 供 _init_kv_cache 检查

        def switch_model(self, **kwargs):
            # 模拟: 卸载旧模型成功，加载新模型失败，回滚成功
            return {
                "success": False,
                "model_id": "previous-model",
                "model_name": "Previous Model",
                "error": "模型 'new-model' 加载失败: 模拟错误。已回滚到 'Previous Model'。",
            }

    monkeypatch.setattr(api_server, "model_manager", FakeManager())
    monkeypatch.setattr(model_host, "model_loaded", True)
    monkeypatch.setattr(api_server, "kv_cache", None)
    monkeypatch.setattr(api_server.mc, "is_cuda_available", lambda: True)
    monkeypatch.setattr(
        api_server,
        "_resolve_model_load_plan",
        lambda *args, **kwargs: _resolved_load(
            model_id="new-model",
            engine="pytorch",
            quant_type="int4",
        ),
    )
    # 绕过 _init_kv_cache（不需要真实模型）
    monkeypatch.setattr(api_server, "_init_kv_cache", lambda: None)

    req = api_server.LoadModelRequest(
        engine="auto",
        quant_type="int4",
        model_id="new-model",
    )
    with pytest.raises(api_server.HTTPException) as exc:
        asyncio.run(api_server.load_model(req))
    assert exc.value.status_code == 500
    assert exc.value.error_code == "MODEL_LOAD_FAILED"


def test_model_error_response_keeps_detail_and_adds_stable_code():
    response = TestClient(api_server.app).post(
        "/api/models/load",
        json={"engine": "unsupported-engine"},
    )

    assert response.status_code == 400
    body = response.json()
    assert isinstance(body["detail"], str)
    assert body["error_code"] == "MODEL_ENGINE_UNSUPPORTED"
    assert response.headers["X-QLH-Error-Code"] == "MODEL_ENGINE_UNSUPPORTED"


def test_local_pytorch_chat_restores_full_model_before_generate(monkeypatch):
    import torch

    ensure_calls = []

    class Tokenizer:
        eos_token_id = 2
        unk_token_id = 0

        def apply_chat_template(self, messages, tokenize=False,
                                add_generation_prompt=True):
            return "prompt"

        def __call__(self, prompt, return_tensors=None):
            return {"input_ids": torch.tensor([[1]])}

        def convert_tokens_to_ids(self, _token):
            return self.unk_token_id

        def encode(self, _text, add_special_tokens=False):
            return [99]

        def decode(self, _ids, skip_special_tokens=False):
            return "reply"

    class Model:
        def generate(self, **kwargs):
            return torch.tensor([[1, 3]])

    class Manager:
        _engine_type = "pytorch"
        tokenizer = Tokenizer()
        model = Model()
        layer_range = (0, 4)

        def ensure_full_model(self):
            ensure_calls.append(True)
            self.layer_range = None

        def get_device(self):
            return torch.device("cpu")

        def _merge_stop_sequences(self, _value):
            return []

        def _get_generation_eos_token_ids(self, _stops):
            return 2

        def _build_stop_criteria(self, _stops, _length):
            return None

        def _decode_generated_ids(self, _ids, _stops):
            return "reply"

    req = api_server.ChatRequest(message="hello", max_new_tokens=1)
    monkeypatch.setattr(api_server, "model_manager", Manager())
    monkeypatch.setattr(api_server.scheduler, "get_distributed_inference_enabled", lambda: False)
    monkeypatch.setattr(api_server._chat_context, "active_session_id", None)
    monkeypatch.setattr(api_server, "_generate_followups", lambda *a, **kw: [])
    monkeypatch.setattr(
        api_server._local_store,
        "save_local_conversation_turn",
        lambda *a, **kw: True,
    )

    result = api_server._execute_chat_full(req)

    assert result["content"] == "reply"
    assert ensure_calls == [True]


def test_reserved_worker_forward_failure_does_not_restore_full_model(monkeypatch):
    ensure_calls = []

    class SegmentManager:
        is_loaded = True
        _engine_type = "pytorch"

        def ensure_full_model(self):
            ensure_calls.append(True)

    req = api_server.ChatRequest(message="hello", max_new_tokens=1)
    monkeypatch.setattr(api_server, "model_manager", SegmentManager())
    monkeypatch.setattr(api_server, "RUN_MODE", "distributed")
    monkeypatch.setattr(
        api_server.scheduler, "get_distributed_inference_enabled", lambda: True,
    )
    monkeypatch.setattr(api_server.scheduler, "_effective_role", lambda: "client")
    monkeypatch.setattr(
        api_server.scheduler, "has_pipeline_worker_reservation", lambda: True,
    )
    monkeypatch.setattr(
        api_server.scheduler,
        "forward_inference_to_master",
        lambda **kwargs: {"status": "disconnected", "error": "master offline"},
    )
    monkeypatch.setattr(api_server._chat_context, "active_session_id", None)

    with pytest.raises(api_server.HTTPException) as exc_info:
        api_server._execute_chat_full(req)

    assert exc_info.value.status_code == 503
    assert ensure_calls == []


def test_mixed_pipeline_failure_cannot_bypass_partial_gguf_guard(monkeypatch):
    """A1/A3 mixed failure must not fall through to direct partial-GGUF chat."""
    calls = {"chat": 0, "ensure": 0}

    class PartialGguf:
        engine_type = "llama_cpp"
        is_loaded = True
        tokenizer = object()

        def ensure_full_model(self):
            calls["ensure"] += 1
            raise RuntimeError("裁层工件 head8，禁止整模回退")

        def chat(self, **kwargs):
            calls["chat"] += 1
            return {"content": "garbage", "usage": {}}

    req = api_server.ChatRequest(
        message="hello", routing_preference="distributed_preferred",
    )
    monkeypatch.setattr(api_server, "model_manager", PartialGguf())
    monkeypatch.setattr(api_server.model_host, "model_loaded", True)
    monkeypatch.setattr(api_server, "RUN_MODE", "distributed")
    monkeypatch.setattr(
        api_server.scheduler, "get_distributed_inference_enabled", lambda: True,
    )
    monkeypatch.setattr(api_server.scheduler, "_effective_role", lambda: "master")
    monkeypatch.setattr(
        api_server.scheduler, "has_pipeline_worker_reservation", lambda: False,
    )
    monkeypatch.setattr(api_server, "runtime_supports", lambda *args, **kwargs: True)
    monkeypatch.setattr(
        api_server.scheduler,
        "run_pipeline_safe",
        lambda *args, **kwargs: {
            "error": "route_a_mixed_legacy_execution_bridge_not_ready",
        },
    )
    monkeypatch.setattr(api_server._chat_context, "active_session_id", None)
    monkeypatch.setattr(
        api_server,
        "_external_route_decision",
        lambda _req: types.SimpleNamespace(use_external=False, reason="disabled"),
    )

    with pytest.raises(api_server.HTTPException) as exc_info:
        api_server._execute_chat_full(req)

    assert exc_info.value.status_code == 503
    assert exc_info.value.error_code == "DISTRIBUTED_FULL_MODEL_FORBIDDEN"
    assert "route_a_mixed_legacy_execution_bridge_not_ready" in str(exc_info.value.detail)
    assert calls == {"chat": 0, "ensure": 0}


def test_pipeline_deadline_is_504_and_never_enters_local_fallback(monkeypatch):
    from request_deadline import RequestDeadline

    calls = {"chat": 0, "ensure": 0}

    class DeadlineManager:
        is_loaded = True
        model_loaded = True
        _engine_type = "pytorch"

        def ensure_full_model(self):
            calls["ensure"] += 1
            pytest.fail("deadline terminal must not restore a full model")

        def chat(self, **_kwargs):
            calls["chat"] += 1
            pytest.fail("deadline terminal must not enter local chat")

    req = api_server.ChatRequest(message="deadline fence")
    monkeypatch.setattr(api_server, "model_manager", DeadlineManager())
    monkeypatch.setattr(api_server.model_host, "model_loaded", True)
    monkeypatch.setattr(api_server, "RUN_MODE", "distributed")
    monkeypatch.setattr(api_server, "runtime_supports", lambda *_a, **_k: True)
    monkeypatch.setattr(
        api_server,
        "_external_route_decision",
        lambda _req: types.SimpleNamespace(use_external=False, reason="disabled"),
    )
    monkeypatch.setattr(
        api_server,
        "_prepare_chat_context",
        lambda *_a, **_k: types.SimpleNamespace(
            session_id="default", history=[], revision=0,
        ),
    )
    monkeypatch.setattr(
        api_server.scheduler, "get_distributed_inference_enabled", lambda: True,
    )
    monkeypatch.setattr(
        api_server.scheduler, "_effective_role", lambda: "master",
    )
    monkeypatch.setattr(
        api_server.scheduler, "has_pipeline_worker_reservation", lambda: False,
    )
    monkeypatch.setattr(
        api_server.scheduler,
        "run_pipeline_safe",
        lambda *_a, **_k: {
            "response": "",
            "error": "request deadline exceeded",
            "reason_code": "request_deadline_exceeded",
            "metrics": {"fallback": False},
        },
    )

    with pytest.raises(api_server.HTTPException) as exc_info:
        api_server._execute_chat_full(
            req,
            request_deadline=RequestDeadline.start(1.0),
        )

    assert exc_info.value.status_code == 504
    assert getattr(exc_info.value, "error_code", "") == (
        "request_deadline_exceeded"
    )
    assert calls == {"chat": 0, "ensure": 0}


def test_prepared_gguf_pipeline_never_generates_local_followups(monkeypatch):
    req = api_server.ChatRequest(
        message="2+2?", routing_preference="distributed_required",
    )
    monkeypatch.setattr(api_server, "RUN_MODE", "distributed")
    monkeypatch.setattr(api_server, "_pipeline_model_is_prepared", lambda: True)
    monkeypatch.setattr(api_server, "runtime_supports", lambda *_a, **_kw: True)
    monkeypatch.setattr(
        api_server, "model_manager",
        types.SimpleNamespace(engine_type="llama_cpp"),
    )
    monkeypatch.setattr(
        api_server, "_external_route_decision",
        lambda _req: types.SimpleNamespace(use_external=False),
    )
    monkeypatch.setattr(
        api_server, "_prepare_chat_context",
        lambda *_a, **_kw: types.SimpleNamespace(
            session_id="test-prepared-gguf", revision=0,
            history=[
                {"role": "user", "content": "hello"},
                {"role": "assistant", "content": "hi"},
            ],
        ),
    )
    monkeypatch.setattr(api_server, "_commit_chat_context_turn", lambda *_a, **_kw: True)
    monkeypatch.setattr(
        api_server, "_generate_followups_llama",
        lambda *_a, **_kw: pytest.fail("partial model used for followups"),
    )
    monkeypatch.setattr(api_server, "_fallback_followups", lambda *_a, **_kw: ["next question?"])
    monkeypatch.setattr(api_server.scheduler, "get_distributed_inference_enabled", lambda: True)
    monkeypatch.setattr(api_server.scheduler, "_effective_role", lambda: "master")
    monkeypatch.setattr(api_server.scheduler, "has_pipeline_worker_reservation", lambda: False)
    monkeypatch.setattr(api_server.scheduler, "run_pipeline_safe", lambda *_a, **_kw: {
        "response": "4", "metrics": {
            "distributed_used": True, "fallback": False,
            "execution_mode": "route_a_stage_offer_v3",
            "workers_used": ["surface", "y700"],
        },
    })

    result = api_server._execute_chat_full(req)

    assert result["content"] == "4"
    assert result["followups"] == ["next question?"]


@pytest.mark.parametrize("engine", ["pytorch", "llama_cpp"])
def test_partial_pipeline_local_only_rejects_before_direct_chat(
        monkeypatch, engine):
    calls = []

    class PartialManager:
        _engine_type = engine
        is_loaded = True
        is_pipeline_prepared = True
        tokenizer = object()

        def chat(self, **kwargs):
            calls.append(("chat", kwargs))
            return {"content": "must not run", "usage": {}}

        def chat_stream(self, **kwargs):
            calls.append(("chat_stream", kwargs))
            yield "must not run"

    req = api_server.ChatRequest(
        message="hello", routing_preference="local_only",
    )
    monkeypatch.setattr(api_server, "model_manager", PartialManager())
    monkeypatch.setattr(api_server.model_host, "model_loaded", False)
    monkeypatch.setattr(api_server, "RUN_MODE", "standalone")
    monkeypatch.setattr(
        api_server.scheduler, "get_distributed_inference_enabled", lambda: False,
    )
    monkeypatch.setattr(
        api_server.scheduler, "has_pipeline_worker_reservation", lambda: False,
    )
    monkeypatch.setattr(
        api_server,
        "_external_route_decision",
        lambda _req: types.SimpleNamespace(use_external=False, reason="disabled"),
    )

    with pytest.raises(api_server.HTTPException) as exc_info:
        api_server._execute_chat_full(req)

    assert exc_info.value.status_code == 409
    assert "分布式流水线模式准备" in str(exc_info.value.detail)
    assert calls == []


def test_local_native_image_chat_uses_mtmd_and_removes_temp_file(monkeypatch):
    image = (
        "data:image/png;base64,"
        "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNk"
        "+A8AAQUBAScY42YAAAAASUVORK5CYII="
    )
    observed = {}
    persisted = []

    class NativeManager:
        _engine_type = "llama_cpp"
        is_loaded = True

        def native_vision_available(self):
            return True

        def chat_image(self, **kwargs):
            observed.update(kwargs)
            observed["exists_during_call"] = os.path.isfile(kwargs["image_path"])
            with open(kwargs["image_path"], "rb") as handle:
                observed["bytes"] = handle.read()
            return {
                "content": "一张测试图片",
                "usage": {"completion_tokens": 4, "total_tokens": 12},
                "tokens_per_second": 8.0,
                "native_mtmd": True,
            }

    req = api_server.ChatRequest(
        message="描述图片",
        image_data_urls=[image],
        routing_preference="local_only",
        max_new_tokens=32,
    )
    monkeypatch.setattr(api_server, "model_manager", NativeManager())
    monkeypatch.setattr(api_server.scheduler, "get_distributed_inference_enabled", lambda: False)
    monkeypatch.setattr(api_server._chat_context, "active_session_id", None)
    monkeypatch.setattr(api_server, "_generate_followups_llama", lambda *a, **kw: [])
    monkeypatch.setattr(
        api_server,
        "_persist_conversation_turn",
        lambda session_id, user, assistant, metrics: persisted.append(
            (session_id, user, assistant, metrics)
        ),
    )
    monkeypatch.setattr(
        api_server,
        "conversation_stats",
        {
            "total_prompt_tokens": 0,
            "total_generated_tokens": 0,
            "total_time_seconds": 0.0,
            "rounds": 0,
        },
    )

    result = api_server._execute_chat_full(req)

    assert result["content"] == "一张测试图片"
    assert result["metrics"]["native_mtmd"] is True
    assert result["metrics"]["execution_mode"] == "local_llama_cpp_mtmd"
    assert observed["exists_during_call"] is True
    assert observed["bytes"].startswith(b"\x89PNG\r\n\x1a\n")
    assert not os.path.exists(observed["image_path"])
    assert persisted[0][1:3] == ("描述图片", "一张测试图片")
    assert image not in repr(persisted)


def test_local_native_image_chat_rejects_multi_image_fallback(monkeypatch):
    image = (
        "data:image/png;base64,"
        "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNk"
        "+A8AAQUBAScY42YAAAAASUVORK5CYII="
    )

    class NativeManager:
        _engine_type = "llama_cpp"
        is_loaded = True

        def native_vision_available(self):
            return True

    req = api_server.ChatRequest(
        message="鎻忚堪鍥剧墖",
        image_data_urls=[image, image],
        allow_external=True,
        prefer_external=True,
    )
    monkeypatch.setattr(api_server, "model_manager", NativeManager())
    monkeypatch.setattr(
        api_server,
        "_external_route_decision",
        lambda _req: types.SimpleNamespace(use_external=False, reason="disabled"),
    )

    with pytest.raises(api_server.HTTPException, match="不能静默丢弃"):
        api_server._execute_chat_full(req)


# ================================================================
# /api/models 端点测试
# ================================================================

def test_list_models_includes_active_model_id(monkeypatch):
    """GET /api/models 应返回 active_model_id"""
    # ★ 2026-09-19：`qwen-1_8b` 已从内置列表**移除**（用户裁定），改用当前默认模型。
    class FakeManager:
        active_model_id = "qwen3-0.6b"

    monkeypatch.setattr(api_server, "model_manager", FakeManager())
    monkeypatch.setattr(model_host, "model_loaded", True)
    monkeypatch.setattr(api_server.mc, "is_cuda_available", lambda: False)

    result = asyncio.run(api_server.list_models())
    assert result["active_model_id"] == "qwen3-0.6b"
    assert isinstance(result["models"], list)
    assert len(result["models"]) > 0

    # 默认模型应在列表中
    model_ids = [m["model_id"] for m in result["models"]]
    assert "qwen3-0.6b" in model_ids


@pytest.mark.skipif(
    not os.path.isdir(os.path.join(os.path.dirname(__file__), "..", "models", "qwen3-0.6b")),
    reason="缺本地工件 models/qwen3-0.6b（CI 不拉模型）",
)
def test_list_models_exposes_new_model_assets_for_frontend_selection(monkeypatch):
    class FakeManager:
        active_model_id = ""

    monkeypatch.setattr(api_server, "model_manager", FakeManager())
    monkeypatch.setattr(model_host, "model_loaded", False)
    monkeypatch.setattr(api_server.mc, "is_cuda_available", lambda: False)

    result = asyncio.run(api_server.list_models())
    models = {item["model_id"]: item for item in result["models"]}
    expected = {
        "qwen2.5-0.5b",
        "qwen3-0.6b",
        "minicpm4-0.5b",
        "distilqwen25-ds3-0324-7b",
    }

    assert expected <= models.keys()
    assert all(models[model_id]["is_builtin"] for model_id in expected)
    assert all(models[model_id]["is_available"] for model_id in expected)
    assert all(models[model_id]["supported_engines"] == ["llama_cpp", "pytorch"] for model_id in expected)

    profiles = {model_id: models[model_id]["profile"] for model_id in expected}
    assert all(profile["profile_schema"] == "qlh.harness.model_profile.v1" for profile in profiles.values())
    assert all(profile["format"] == "both" for profile in profiles.values())
    assert all(len(profile["artifact_sha256"]) == 64 for profile in profiles.values())
    assert all(profile["evidence"]["artifact_digest_mode"] == "manifest" for profile in profiles.values())
    assert profiles["qwen3-0.6b"]["generation"]["thinking"] == "declared"
    assert profiles["minicpm4-0.5b"]["adaptation"]["prompt_family"] == "minicpm4_chat_v1"
    assert profiles["distilqwen25-ds3-0324-7b"]["resources"]["min_vram_gb"] == 8.0


def test_list_models_returns_null_when_not_loaded(monkeypatch):
    """模型未加载时 active_model_id 为 None"""
    class FakeManager:
        active_model_id = ""

    monkeypatch.setattr(api_server, "model_manager", FakeManager())
    monkeypatch.setattr(model_host, "model_loaded", False)
    monkeypatch.setattr(api_server.mc, "is_cuda_available", lambda: False)

    result = asyncio.run(api_server.list_models())
    assert result["active_model_id"] is None


# ================================================================
# /api/models/current 端点测试
# ================================================================

def test_current_model_when_not_loaded(monkeypatch):
    """模型未加载时 → loaded=False"""
    monkeypatch.setattr(model_host, "model_loaded", False)
    result = asyncio.run(api_server.get_current_model())
    assert result["loaded"] is False
    assert result["model_id"] is None


def test_current_model_when_loaded(monkeypatch):
    """模型已加载时 → 返回完整信息"""
    class FakeManager:
        active_model_id = "qwen-1_8b"
        _engine_type = "pytorch"

        def get_model_info(self):
            return {
                "model_id": "qwen-1_8b",
                "engine": "pytorch",
                "model_name": "Qwen-1.8B-Chat",
                "model_path": "/fake/path",
                "total_params": "1.8B",
                "device": "cuda:0",
            }

        def get_memory_usage(self):
            return {"gpu_allocated_gb": 1.8, "gpu_reserved_gb": 2.0}

    monkeypatch.setattr(api_server, "model_manager", FakeManager())
    monkeypatch.setattr(model_host, "model_loaded", True)
    monkeypatch.setattr(model_host, "current_quant", "int4")

    result = asyncio.run(api_server.get_current_model())
    assert result["loaded"] is True
    assert result["model_id"] == "qwen-1_8b"
    assert result["engine"] == "pytorch"
    assert result["model_name"] == "Qwen-1.8B-Chat"


@pytest.mark.parametrize("host_loaded", [False, True])
def test_current_model_pipeline_segment_is_not_reported_as_full(monkeypatch, host_loaded):
    class PartialManager:
        active_model_id = "tiny-qwen2"
        _engine_type = "pytorch"
        is_loaded = True
        is_pipeline_prepared = True

        def get_pipeline_descriptor(self):
            return {
                "schema_version": 1,
                "model_id": "tiny-qwen2",
                "model_type": "qwen2",
                "model_sha256": "a" * 64,
                "total_layers": 4,
                "hidden_size": 128,
                "tokenizer_sha256": "b" * 64,
                "pipeline_runtime_supported": True,
            }

        def get_model_info(self):
            # segment 已物化时生产代码会读 inference_switches（routes_models.py:52）
            return {"inference_switches": {"streaming_mode": "fast"}}

    monkeypatch.setattr(api_server, "model_manager", PartialManager())
    monkeypatch.setattr(model_host, "model_loaded", host_loaded)
    monkeypatch.setattr(model_host, "current_quant", "fp32")

    result = asyncio.run(api_server.get_current_model())

    assert result["loaded"] is False
    assert result["pipeline_prepared"] is True
    assert result["segment_materialized"] is True
    assert result["model_id"] == "tiny-qwen2"
    assert result["quant_type"] == "fp32"


# ================================================================
# /api/models/switch 端点测试
# ================================================================

def test_switch_model_rejects_invalid_engine(monkeypatch):
    """无效 engine → 400"""
    monkeypatch.setattr(api_server.mc, "is_cuda_available", lambda: True)
    req = api_server.SwitchModelRequest(
        model_id="qwen-1_8b",
        engine="invalid",
    )
    with pytest.raises(api_server.HTTPException) as exc:
        asyncio.run(api_server.switch_model(req))
    assert exc.value.status_code == 400


def test_switch_model_calls_manager_switch(monkeypatch):
    monkeypatch.setattr(api_server.scheduler, "_distributed_inference_enabled", False)
    """正常切换 → 调用 model_manager.switch_model 并更新全局状态"""
    switch_calls = []

    class FakeManager:
        active_model_id = ""
        is_loaded = False
        _engine_type = ""
        quant_type = None

        def switch_model(self, **kwargs):
            switch_calls.append(kwargs)
            self.is_loaded = True
            self.active_model_id = kwargs["model_id"]
            self._engine_type = kwargs.get("engine") or "llama_cpp"
            return {
                "success": True,
                "model_id": self.active_model_id,
                "model_name": "Test Model",
                "error": None,
            }

    monkeypatch.setattr(api_server, "model_manager", FakeManager())
    monkeypatch.setattr(model_host, "model_loaded", False)
    monkeypatch.setattr(api_server, "kv_cache", None)
    monkeypatch.setattr(api_server.mc, "is_cuda_available", lambda: True)
    monkeypatch.setattr(
        api_server,
        "_resolve_model_load_plan",
        lambda *args, **kwargs: _resolved_load(
            model_id="qwen2.5-7b-gguf",
            engine="llama_cpp",
            model_path="models/test.gguf",
            quant_type="Q4_K_M",
            layer_range_mode="precut_artifact",
        ),
    )
    monkeypatch.setattr(api_server, "_get_registered_experimental_models", lambda: [])
    monkeypatch.setattr(api_server, "_init_kv_cache", lambda: None)

    req = api_server.SwitchModelRequest(
        model_id="qwen2.5-7b-gguf",
        quant_type="Q4_K_M",
        engine="llama_cpp",
    )
    result = asyncio.run(api_server.switch_model(req))
    assert result["success"] is True
    assert len(switch_calls) == 1
    assert switch_calls[0]["model_id"] == "qwen2.5-7b-gguf"
    assert switch_calls[0]["engine"] == "llama_cpp"


def test_switch_model_rejects_gguf_quant_when_effective_engine_is_pytorch(monkeypatch, tmp_path):
    """engine=auto 解析到 PyTorch 时，不能把 Q4_K_M 传进 PyTorch 加载路径。"""
    switch_calls = []

    class FakeManager:
        active_model_id = ""
        is_loaded = False
        _engine_type = "pytorch"
        quant_type = None

        def switch_model(self, **kwargs):
            switch_calls.append(kwargs)
            return {"success": True, "model_id": kwargs["model_id"], "model_name": "bad", "error": None}

    monkeypatch.setattr(api_server, "model_manager", FakeManager())
    hf_dir = tmp_path / "hf-model"
    hf_dir.mkdir()
    (hf_dir / "config.json").write_text("{}", encoding="utf-8")
    (hf_dir / "model.safetensors").write_bytes(b"weights")
    monkeypatch.setattr(api_server, "_get_registered_experimental_models", lambda: [{
        "model_id": "hf-model",
        "name": "HF Model",
        "model_type": "safetensors",
        "model_path": str(hf_dir),
        "gguf_path": "",
    }])

    req = api_server.SwitchModelRequest(
        model_id="hf-model",
        quant_type="Q4_K_M",
        engine="auto",
    )
    with pytest.raises(api_server.HTTPException) as exc:
        asyncio.run(api_server.switch_model(req))

    assert exc.value.status_code == 400
    assert switch_calls == []


def test_switch_model_resets_runtime_conversation_state(monkeypatch):
    monkeypatch.setattr(api_server.scheduler, "_distributed_inference_enabled", False)
    switch_calls = []
    init_calls = []
    cleared = []

    class FakeKvCache:
        def clear(self):
            cleared.append(True)

    class FakeManager:
        active_model_id = ""
        is_loaded = False
        _engine_type = ""
        quant_type = None

        def switch_model(self, **kwargs):
            switch_calls.append(kwargs)
            self.is_loaded = True
            self.active_model_id = kwargs["model_id"]
            self._engine_type = kwargs.get("engine") or "llama_cpp"
            return {
                "success": True,
                "model_id": self.active_model_id,
                "model_name": "Test Model",
                "error": None,
            }

    monkeypatch.setattr(api_server, "model_manager", FakeManager())
    monkeypatch.setattr(model_host, "model_loaded", True)
    monkeypatch.setattr(api_server, "kv_cache", FakeKvCache())
    monkeypatch.setattr(
        api_server,
        "conversation_stats",
        {
            "total_prompt_tokens": 10,
            "total_generated_tokens": 20,
            "total_time_seconds": 3.5,
            "rounds": 2,
        },
    )
    monkeypatch.setattr(
        api_server._chat_context,
        "histories",
        {"default": [{"role": "assistant", "content": "DeepSeek-R1 intro"}]},
    )
    monkeypatch.setattr(
        api_server,
        "_resolve_model_load_plan",
        lambda *args, **kwargs: _resolved_load(
            model_id="qwen2.5-7b-gguf",
            engine="llama_cpp",
            model_path="models/test.gguf",
            quant_type="Q4_K_M",
            layer_range_mode="precut_artifact",
        ),
    )
    monkeypatch.setattr(api_server, "_get_registered_experimental_models", lambda: [])
    monkeypatch.setattr(api_server, "_init_kv_cache", lambda: init_calls.append(True))

    req = api_server.SwitchModelRequest(
        model_id="qwen2.5-7b-gguf",
        quant_type="Q4_K_M",
        engine="llama_cpp",
    )
    result = asyncio.run(api_server.switch_model(req))

    assert result["success"] is True
    assert len(switch_calls) == 1
    assert cleared == [True]
    assert api_server.kv_cache is None
    assert api_server._chat_context.histories == {}
    assert api_server.conversation_stats == {
        "total_prompt_tokens": 0,
        "total_generated_tokens": 0,
        "total_time_seconds": 0.0,
        "rounds": 0,
    }
    assert init_calls == [True]


# ================================================================
# 模型注册表 DB 集成测试
# ================================================================

def test_register_model_passes_db_experimental_models_to_switch(monkeypatch, tmp_path):
    monkeypatch.setattr(api_server.scheduler, "_distributed_inference_enabled", False)
    """注册的 DB 模型应可通过 switch_model 端点加载（B1 修复验证）"""
    gguf_file = tmp_path / "registered-model.Q4_K_M.gguf"
    gguf_file.write_text("x", encoding="utf-8")

    db_entry = {
        "model_id": "registered-gguf-model",
        "name": "Registered GGUF Model",
        "model_type": "gguf",
        "model_path": "",
        "gguf_path": str(gguf_file),
        "recommended_vram_gb": 4.0,
        "max_context": 4096,
        "quant_types": ["Q4_K_M"],
        "description": "test",
    }

    switch_calls = []

    class FakeManager:
        active_model_id = ""
        is_loaded = False
        _engine_type = ""
        quant_type = None

        def switch_model(self, **kwargs):
            switch_calls.append(kwargs)
            self.is_loaded = True
            self.active_model_id = kwargs["model_id"]
            self._engine_type = "llama_cpp"
            return {
                "success": True,
                "model_id": self.active_model_id,
                "model_name": "Registered GGUF Model",
                "error": None,
            }

    monkeypatch.setattr(api_server, "_get_registered_experimental_models", lambda: [db_entry])
    monkeypatch.setattr(api_server, "model_manager", FakeManager())
    monkeypatch.setattr(model_host, "model_loaded", False)
    monkeypatch.setattr(api_server, "kv_cache", None)
    monkeypatch.setattr(api_server.mc, "is_cuda_available", lambda: False)
    monkeypatch.setattr(api_server, "_init_kv_cache", lambda: None)

    req = api_server.SwitchModelRequest(
        model_id="registered-gguf-model",
        quant_type="Q4_K_M",
        engine="auto",
    )
    result = asyncio.run(api_server.switch_model(req))
    assert result["success"] is True
    assert switch_calls[0]["model_id"] == "registered-gguf-model"
    # 应传递 db_experimental_models（B1 修复验证）
    assert "db_experimental_models" in switch_calls[0]
    assert switch_calls[0]["db_experimental_models"] == [db_entry]


def test_list_model_registry_returns_db_models(monkeypatch):
    """GET /api/models/registry → 返回 DB 注册模型列表"""
    db_models = [
        {"model_id": "m1", "name": "Model 1", "model_type": "gguf"},
        {"model_id": "m2", "name": "Model 2", "model_type": "safetensors"},
    ]
    monkeypatch.setattr(api_server, "_get_registered_experimental_models", lambda: db_models)
    result = asyncio.run(api_server.list_model_registry())
    assert result["models"] == db_models


def test_list_model_registry_redacts_persisted_paths(monkeypatch):
    root = Path(api_server.mc._APP_ROOT).resolve()
    asset_dir = root / "audit-assets" / "custom"
    db_models = [{
        "model_id": "m1",
        "name": "Model 1",
        "model_type": "both",
        "model_path": str(asset_dir),
        "gguf_path": str(asset_dir / "model.gguf"),
        "expected_paths": [f"Safetensors: {asset_dir}"],
    }]
    monkeypatch.setattr(api_server, "_get_registered_experimental_models", lambda: db_models)

    result = asyncio.run(api_server.list_model_registry())
    payload = result["models"][0]
    assert payload["model_path"] == "audit-assets/custom"
    assert payload["gguf_path"] == "audit-assets/custom/model.gguf"
    assert payload["expected_paths"] == ["Safetensors: audit-assets/custom"]
    assert str(root) not in repr(payload)


# ================================================================
# _strip_native_thinking_tags 测试（P3修复: DeepSeek-R1 思考标记剥离）
# ================================================================

class TestStripNativeThinkingTags:
    """测试 _strip_native_thinking_tags 对各种思考标记格式的处理"""

    # 使用 chr() 构造 XML 标签，避免测试框架剥离 <xxx> 格式的文本
    LT = chr(60)  # <
    GT = chr(62)  # >

    def test_strips_think_block_with_response_marker(self):
        """  ...  块 +  标记 → 只保留最终回答"""
        L, G = self.LT, self.GT
        text = f"{L}think{G}thinking process here...{L}/think{G}\n{L}response{G}Hello! I'm DeepSeek-R1.{L}/response{G}{L}|im_end|{G}"
        result = api_server._strip_native_thinking_tags(text)
        assert "thinking process" not in result
        assert "Hello! I'm DeepSeek-R1." in result
        assert "<|im_end|>" not in result

    def test_strips_think_block_without_response_marker(self):
        """  ...  块（无  标记）→ 回答直接跟在 </think> 后"""
        L, G = self.LT, self.GT
        text = f"{L}think{G}Let me think about this...{L}/think{G}\nThe answer is 42.{L}|im_end|{G}"
        result = api_server._strip_native_thinking_tags(text)
        assert "Let me think about this" not in result
        assert "The answer is 42." in result

    def test_strips_leading_reasoning_before_closing_think(self):
        """Native DeepSeek completion may contain only the closing think tag."""
        L, G = self.LT, self.GT
        text = f"reasoning leaked from prompt prefill{L}/think{G}\nFinal answer."
        result = api_server._strip_native_thinking_tags(text)
        assert result == "Final answer."

    def test_strips_standalone_think_tags(self):
        """独立的  和 </think> 标记 → 剥离"""
        L, G = self.LT, self.GT
        # 成对的  块：内容也被移除
        text = f"Here is a response with {L}think{G}think block here.{L}/think{G} around."
        result = api_server._strip_native_thinking_tags(text)
        # 块内内容被移除
        assert "think block here" not in result
        # 普通文字应保留
        assert "Here is a response with" in result
        assert "around." in result

    def test_strips_im_end_tokens(self):
        """<|im_end|> 和 <|im_start|> token → 剥离"""
        text = "Hello world<|im_end|><|im_start|>assistant\n"
        result = api_server._strip_native_thinking_tags(text)
        assert "<|im_end|>" not in result
        assert "<|im_start|>" not in result
        assert "Hello world" in result

    def test_handles_empty_text(self):
        """空文本 → 返回空文本"""
        assert api_server._strip_native_thinking_tags("") == ""
        assert api_server._strip_native_thinking_tags(None) is None

    def test_clean_text_passes_through(self):
        """无思考标记的普通文本 → 原样返回"""
        text = "这是一个普通的回答，没有任何思考标记。"
        result = api_server._strip_native_thinking_tags(text)
        assert result == text

    def test_strips_qwen3_answer_tags(self):
        """Qwen3  格式 → 剥离思考和 answer 标签"""
        L, G = self.LT, self.GT
        text = f"{L}think{G}some reasoning{L}/think{G}\n{L}answer{G}Here is the final response{L}/answer{G}"
        result = api_server._strip_native_thinking_tags(text)
        assert "some reasoning" not in result
        assert "Here is the final response" in result

    def test_strips_real_deepseek_r1_output(self):
        """模拟 DeepSeek-R1 真实输出: 自我介绍在  块中泄漏"""
        L, G = self.LT, self.GT
        text = (
            f"{L}think{G}\n"
            "您好！我是由中国的深度求索（DeepSeek）公司独立开发的智能助手DeepSeek-R1，"
            "有关模型和产品的详细内容请参考官方文档。\n"
            f"{L}/think{G}\n"
            f"{L}response{G}\n"
            "关于您的问题，我的回答是：这是正确的答案。"
        )
        result = api_server._strip_native_thinking_tags(text)
        # 思考块中的内容应被完全移除
        assert "深度求索" not in result
        assert "DeepSeek-R1" not in result
        assert "官方文档" not in result
        # 回答内容应保留
        assert "关于您的问题" in result
        assert "这是正确的答案" in result

    def test_preserves_response_content_formatting(self):
        """保留回答内容的换行和格式"""
        L, G = self.LT, self.GT
        text = f"{L}think{G}reasoning here...{L}/think{G}\n{L}response{G}第一行\n第二行\n\n第三行"
        result = api_server._strip_native_thinking_tags(text)
        assert "reasoning here" not in result
        assert "第一行" in result
        assert "第二行" in result
        assert "第三行" in result


class TestParseThinkingResponseNative:
    """测试 _parse_thinking_response 对 DeepSeek-R1 本地格式的解析"""

    def test_parse_closing_only_native_think_format(self):
        """Completion starting inside native <think> should split at </think>."""
        L, G = chr(60), chr(62)
        text = f"reasoning from native template{L}/think{G}\nFinal answer."
        answer, thinking = api_server._parse_thinking_response(text)

        assert answer == "Final answer."
        assert thinking == "reasoning from native template"

    def test_parse_native_think_removes_legacy_prefix_if_present(self):
        """Native parser should not leak the legacy thinking prefill."""
        L, G = chr(60), chr(62)
        text = f"【思考】\nreasoning from native template{L}/think{G}\nFinal answer."
        answer, thinking = api_server._parse_thinking_response(text)

        assert answer == "Final answer."
        assert thinking == "reasoning from native template"

    def test_parse_native_think_format_extracts_thinking(self):
        """  ...  格式 → 正确提取思考和答案"""
        text = "【思考】\nthinking内容...\nresponse这是实际回答"
        # _parse_thinking_response 应先匹配【思考】，找不到【思考结束】则尝试本地格式
        answer, thinking = api_server._parse_thinking_response(text)
        # 由于【思考】标记存在但没有【思考结束】，fallthrough 到情况2
        # 情况2 调用 _strip_native_thinking_tags 清理
        assert answer is not None
        assert "这是实际回答" in answer

    def test_parse_fallback_strips_native_tags(self):
        """格式不匹配 → 回退清理本地标记"""
        text = "【思考】\nsome text\nNo end marker here..."
        answer, thinking = api_server._parse_thinking_response(text)
        assert answer is not None
        # 清理后不应有【思考】标记
        assert "【思考】" not in answer


def test_api_catalog_and_load_resolution_use_registered_builtin_override(tmp_path, monkeypatch):
    gguf = tmp_path / "replacement.gguf"
    gguf.write_bytes(b"existing")
    mid = "qwen3-0.6b"
    row = {"model_id": mid, "name": "Verified replacement", "model_type": "gguf",
           "model_path": "", "gguf_path": str(gguf)}
    monkeypatch.setattr(api_server, "_get_registered_experimental_models", lambda: [row])
    model = next(m for m in api_server._get_all_model_configs() if m.model_id == mid)
    assert model == mc.get_model_config(mid, [row])
    payload = api_server._model_api_payload(model)
    assert payload["supported_engines"] == ["llama_cpp"]
    assert payload["name"] == row["name"]
    assert payload["has_safetensors"] is False


def test_registry_public_payload_hides_external_manifest_server_paths():
    payload = api_server._public_registry_payload({"model_id": "fixture",
        "manifest_external_files": {"external/private.gguf": "/private/weights/private.gguf"}})
    assert "manifest_external_files" not in payload
