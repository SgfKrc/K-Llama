"""PC 层段 worker 的能力声明（`Scheduler._task_worker_capabilities`）。

补 #33 的接线：载了层段工件就必须**同时**声明 `layer_forward` 与 `layer_ranges` ——
否则调度侧不会把层段 Stage 派给这台 PC worker（执行路径早就实现了，只是永远轮不到）。

测试只验证「声明 ⇔ 就绪」这一条契约，不加载任何真实模型。
"""
import os
import sys
import hashlib
import threading
from types import SimpleNamespace

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from scheduler import Scheduler  # noqa: E402
import scheduler_task_worker  # noqa: E402
from execution_provider import ModelIdentity  # noqa: E402
from task_worker_adapter import remote_provider_id  # noqa: E402


def test_layer_weight_bytes_are_local_to_the_configured_segment(monkeypatch):
    import relay_segment_info

    monkeypatch.setattr(
        relay_segment_info, "read_gguf_tensor_bytes",
        lambda path: {"blk.0.attn_q.weight": 10, "blk.0.ffn.weight": 20,
                      "blk.1.attn_q.weight": 30, "output.weight": 1000},
    )
    assert scheduler_task_worker._layer_weight_bytes_of("cut.gguf", 2) == [30, 30]
    assert scheduler_task_worker._layer_weight_bytes_of("cut.gguf", 3) == []


def test_task_worker_capability_log_summary_is_bounded_and_exact():
    layer_weight_bytes = list(range(1, 21))
    summary = scheduler_task_worker._task_worker_capability_log_summary({
        "stage_types": ["layer_forward"],
        "layer_ranges": [[4, 24]],
        "models": [{
            "model_id": "q35-9b-local-mid4-24.gguf",
            "engine": "llama_cpp",
            "format": "gguf",
            "revision": "2",
            "sha256": "d" * 64,
        }],
        "layer_artifacts": [{
            "model_id": "q35-9b-local-mid4-24.gguf",
            "source_model_id": "qwen3-5-9b",
            "source_model_sha256": "a" * 64,
            "tokenizer_sha256": "b" * 64,
            "artifact_sha256": "d" * 64,
            "layer_range": [4, 24],
            "layer_weight_bytes": layer_weight_bytes,
        }],
        "resource_gate": {"admitted": True, "reason_code": ""},
    })

    assert summary == {
        "stage_types": ["layer_forward"],
        "binary_hidden": False,
        "layer_ranges": [[4, 24]],
        "models": [{
            "model_id": "q35-9b-local-mid4-24.gguf",
            "source_model_sha256_prefix": "a" * 12,
            "tokenizer_sha256_prefix": "b" * 12,
        }],
        "layer_artifacts": [{
            "source_model_id": "qwen3-5-9b",
            "source_model_sha256_prefix": "a" * 12,
            "tokenizer_sha256_prefix": "b" * 12,
            "artifact_sha256_prefix": "d" * 12,
            "layer_range": [4, 24],
            "weight_bytes_count": 20,
            "weight_bytes_all_positive": True,
        }],
        "resource_gate_admitted": True,
    }
    assert "layer_weight_bytes" not in repr(summary)
    assert repr(layer_weight_bytes) not in repr(summary)


class _FakeHost:
    """最小 host 打桩：capabilities 只读这几个属性。"""

    def __init__(self, *, layer_range=None, full_model_loaded=False,
                 forward_layers=None):
        self.layer_range = layer_range
        self.model_loaded = full_model_loaded
        self.is_loaded = full_model_loaded
        self.active_model_id = "fake-model"
        # ★ 票 15a：`forward_layers=None` => 不提供能力探测（维持既有广告行为）；
        #   给布尔值 => 模拟"引擎自带 supports"的 runtime（同时给出 backend 身份）。
        self._forward_layers = forward_layers
        if forward_layers is not None:
            self.engine_type = "llama_cpp"

    def has_loaded_model(self):
        return self.model_loaded

    def supports(self, capability):
        from koakuma_engine import Capability

        if capability == Capability.FORWARD_LAYERS:
            return bool(self._forward_layers)
        return True


def _scheduler(*, active_layer_config=None, layer_range=None,
               full_model_loaded=False, full_model_identity=None,
               forward_layers=None):
    """绕过 Scheduler.__init__ 构造最小能力快照目标。"""
    scheduler = Scheduler.__new__(Scheduler)
    scheduler._model_control = SimpleNamespace(boot_id="fixture-worker-boot")
    scheduler._distributed_inference_enabled = True
    scheduler._host = _FakeHost(
        layer_range=layer_range, full_model_loaded=full_model_loaded,
        forward_layers=forward_layers,
    )
    scheduler._active_layer_config = active_layer_config
    if full_model_identity is not None:
        scheduler._require_callbacks = lambda: SimpleNamespace(
            active_execution_model_identity=lambda: full_model_identity,
        )
    return scheduler


def _capabilities(*, active_layer_config=None, layer_range=None,
                  full_model_loaded=False, full_model_identity=None,
                  forward_layers=None):
    """绕过 Scheduler.__init__ 直接调能力快照。"""
    return _scheduler(
        active_layer_config=active_layer_config,
        layer_range=layer_range,
        full_model_loaded=full_model_loaded,
        full_model_identity=full_model_identity,
        forward_layers=forward_layers,
    )._task_worker_capabilities()


def _reject_stage_offer(scheduler, *, stage_type, model_identity):
    responses = []
    scheduler._task_worker_control = SimpleNamespace(
        coordinator_snapshot=lambda: {"manual_stage_dispatch_enabled": True},
    )
    scheduler._runtime_setting = lambda name: True
    scheduler.get_effective_node_id = lambda: "pc-worker"
    scheduler._task_worker_stage_lock = threading.Lock()
    scheduler._task_worker_active_attempts = {}
    scheduler._send_task_worker_stage_accept = (
        lambda request_message_id, offer, **response: responses.append(response)
    )
    scheduler._handle_task_worker_stage_offer(SimpleNamespace(
        message_id=f"msg-{stage_type}",
        sent_at_ms=1,
        payload={
            "attempt_id": f"attempt-{stage_type}",
            "provider_id": remote_provider_id("pc-worker"),
            "stage_type": stage_type,
            "model_identity": model_identity,
        },
    ))
    assert len(responses) == 1
    assert responses[0]["accepted"] is False
    return responses[0]["reason_code"]


class TestLayerCapabilities:
    def test_full_worker_keeps_legacy_stage_types(self):
        caps = _capabilities()
        assert caps["stage_types"] == ["full_inference", "aggregate"]
        assert caps["layer_worker"] is False
        assert caps["layer_ranges"] == []

    def test_layer_config_declares_layer_forward_and_range(self):
        caps = _capabilities(
            active_layer_config={"layer_range": [16, 20], "engine": "llama_cpp"},
        )
        assert caps["stage_types"] == ["layer_forward"]
        assert caps["layer_worker"] is True
        assert caps["layer_ranges"] == [[16, 20]]

    def test_layer_worker_declares_stage_chunked_input(self):
        """★ 2026-10-08（DIST-NEXT-2b）：层段 worker 必须声明能接收分片 hidden。

        接收侧（`_handle_task_worker_stage_chunk` + `_assemble_stage_root_input` 的
        装配/摘要校验/fail-closed）早已实现，但此前**从未声明** ⇒ provider 的
        `_maybe_send_stage_chunks` 直接跳过分片、dispatch 前的预检按超限拒绝 ⇒
        大 prompt 只能 503（2b 实测 `route_a_stage_frame_too_large:…:wire=24160940`）。
        """
        caps = _capabilities(
            active_layer_config={"layer_range": [16, 20], "engine": "llama_cpp"},
        )
        assert caps["stage_chunked_input"] is True

    def test_layer_worker_without_forward_layers_does_not_advertise_it(self):
        """★ 票 15a：**能起 server 不等于能参与层段接力**。

        server 引擎（llama-server 子进程）对外仍报 `engine_type="llama_cpp"`
        （家族一致），但 `FORWARD_LAYERS` 为 False ⇒ 即使本进程被配置成层段 worker，
        也不得广告 `layer_forward`，否则 master 会把层段派给一台执行不了的 worker。
        """
        caps = _capabilities(
            active_layer_config={"layer_range": [16, 20], "engine": "llama_cpp"},
            forward_layers=False,
        )
        # R2 keeps hello valid (stage_types must be non-empty) while granting
        # neither full-model work nor layers to this unavailable layer worker.
        assert caps["stage_types"] == ["aggregate"]
        assert "layer_forward" not in caps["stage_types"]
        assert "full_inference" not in caps["stage_types"]
        assert caps["layer_ranges"] == []

    def test_layer_worker_with_forward_layers_keeps_advertising_it(self):
        """绑定路径（真的支持层段前向）广告行为不变。"""
        caps = _capabilities(
            active_layer_config={"layer_range": [16, 20], "engine": "llama_cpp"},
            forward_layers=True,
        )
        assert caps["stage_types"] == ["layer_forward"]

    def test_full_worker_does_not_declare_chunked_input(self):
        """非层段 worker 不做 hidden 交接 ⇒ 保守不声明（维持既有语义）。"""
        assert _capabilities()["stage_chunked_input"] is False

    def test_malformed_layer_range_does_not_break_hello(self):
        # layer_ranges 必须是 [start, end) 整数对；畸形值宁可不上报，也不能让 hello 挂掉。
        caps = _capabilities(
            active_layer_config={"layer_range": "16-20", "engine": "llama_cpp"},
        )
        assert "layer_forward" in caps["stage_types"]
        assert caps["layer_ranges"] == []

    def test_artifact_derived_model_id_is_protocol_safe(self, tmp_path, monkeypatch):
        """从工件 manifest 推导的 `model_id` 必须能过协议正则。

        manifest 的 `artifact` 字段是**相对路径**（如 `models\\x.gguf`），而
        `task_worker_protocol._SAFE_ID = ^[A-Za-z0-9_.:-]{1,128}$` 不允许反斜杠 ⇒
        直接拿它当 `model_id` 会让整个 hello 被判
        `payload.capabilities.models[0].model_id is invalid`，worker 永远进不了
        `admitted`（实测踩到，且被 legacy 通道的噪声盖了很久）。
        """
        import json

        from task_worker_protocol import _SAFE_ID, _validate_capabilities

        gguf = tmp_path / "qwen25-05b-f16-cut-16-20.gguf"
        gguf.write_bytes(b"")
        artifact_sha256 = hashlib.sha256(b"").hexdigest()
        (tmp_path / "qwen25-05b-f16-cut-16-20.manifest.json").write_text(
            json.dumps({
                "source_layer_range": [16, 20],
                "artifact": "models\\qwen25-05b-f16-cut-16-20.gguf",
                "artifact_sha256": artifact_sha256,
                "source_model_sha256": "b" * 64,
                "source_model_id": "qwen3-5-2b",
                "hidden_size": 2048,
                "tokenizer_sha256": "c" * 64,
                "mode": "middle",
                "generator_version": 2,
            }),
            encoding="utf-8",
        )
        monkeypatch.setenv("QLH_LAYER_GGUF", str(gguf))
        monkeypatch.setattr(
            scheduler_task_worker, "_layer_weight_bytes_of",
            lambda artifact_path, local_layers: [1, 2, 3, 4],
        )
        monkeypatch.setattr(
            "relay_segment_info.read_gguf_tokenizer_sha256",
            lambda artifact_path: "c" * 64,
        )

        scheduler = _scheduler()
        caps = scheduler._task_worker_capabilities()
        model_id = caps["models"][0]["model_id"]
        assert _SAFE_ID.match(model_id), f"model_id 不合法: {model_id!r}"
        assert "\\" not in model_id
        assert caps["stage_types"] == ["layer_forward"]
        assert caps["layer_ranges"] == [[16, 20]]
        assert caps["segment_mode"] == "middle"
        assert caps["layer_artifacts"] == [{
            "layer_range": [16, 20],
            "segment_mode": "middle",
            "model_id": model_id,
            "artifact_sha256": artifact_sha256,
            "source_model_sha256": "b" * 64,
            "source_model_id": "qwen3-5-2b",
            "hidden_size": 2048,
            "tokenizer_sha256": "c" * 64,
            "layer_weight_bytes": [1, 2, 3, 4],
        }]
        artifact_identity = caps["models"][0]
        assert scheduler._task_worker_full_model_identities(caps) == []
        assert scheduler._task_worker_offer_model_identity_matches({
            "stage_type": "layer_forward",
            "model_identity": artifact_identity,
        }, caps) is True
        assert scheduler._task_worker_offer_model_identity_matches({
            "stage_type": "full_inference",
            "model_identity": artifact_identity,
        }, caps) is False
        assert scheduler._task_worker_offer_model_identity_matches({
            "stage_type": "aggregate",
            "model_identity": artifact_identity,
        }, caps) is False
        assert _reject_stage_offer(
            scheduler,
            stage_type="full_inference",
            model_identity=artifact_identity,
        ) == "unsupported_stage_type"
        assert _reject_stage_offer(
            scheduler,
            stage_type="aggregate",
            model_identity=artifact_identity,
        ) == "unsupported_stage_type"
        _validate_capabilities(caps, version=3)

    def test_full_model_and_layer_artifact_identities_are_both_advertised(
            self, tmp_path, monkeypatch):
        import json

        gguf = tmp_path / "cut-16-20.gguf"
        gguf.write_bytes(b"")
        artifact_sha256 = hashlib.sha256(b"").hexdigest()
        gguf.with_suffix(".manifest.json").write_text(json.dumps({
            "source_layer_range": [16, 20],
            "artifact_sha256": artifact_sha256,
            "source_model_sha256": "c" * 64,
            "mode": "middle",
            "generator_version": 2,
        }), encoding="utf-8")
        monkeypatch.setenv("QLH_LAYER_GGUF", str(gguf))
        monkeypatch.setattr(scheduler_task_worker, "torch_available", lambda: True)

        full_identity = ModelIdentity(
            model_id="full-model", engine="pytorch", format="safetensors",
            revision="local", sha256="a" * 64,
        )
        scheduler = _scheduler(
            full_model_loaded=True, full_model_identity=full_identity,
        )

        caps = scheduler._task_worker_capabilities()

        assert caps["stage_types"] == [
            "full_inference", "aggregate", "layer_forward",
        ]
        assert [item["model_id"] for item in caps["models"]] == [
            "full-model", gguf.name,
        ]
        assert caps["layer_artifacts"][0]["model_id"] == gguf.name
        advertised_full_identity, artifact_identity = caps["models"]
        assert scheduler._task_worker_full_model_identities(caps) == [
            advertised_full_identity,
        ]
        assert scheduler._task_worker_offer_model_identity_matches({
            "stage_type": "full_inference",
            "model_identity": advertised_full_identity,
        }, caps) is True
        assert scheduler._task_worker_offer_model_identity_matches({
            "stage_type": "full_inference",
            "model_identity": artifact_identity,
        }, caps) is False
        assert scheduler._task_worker_offer_model_identity_matches({
            "stage_type": "layer_forward",
            "model_identity": artifact_identity,
        }, caps) is True
        assert _reject_stage_offer(
            scheduler,
            stage_type="full_inference",
            model_identity=artifact_identity,
        ) == "model_identity_mismatch"

    def test_loaded_layer_artifact_is_never_a_full_model_identity(
            self, tmp_path, monkeypatch):
        import json

        gguf = tmp_path / "cut-16-20.gguf"
        gguf.write_bytes(b"cropped layer artifact")
        artifact_sha256 = hashlib.sha256(gguf.read_bytes()).hexdigest()
        gguf.with_suffix(".manifest.json").write_text(json.dumps({
            "source_layer_range": [16, 20],
            "artifact_sha256": artifact_sha256,
            "source_model_sha256": "c" * 64,
            "mode": "middle",
            "generator_version": 2,
        }), encoding="utf-8")
        monkeypatch.setenv("QLH_LAYER_GGUF", str(gguf))
        monkeypatch.setattr(
            scheduler_task_worker, "keep_head_shim_available", lambda: True,
        )
        monkeypatch.setattr(
            scheduler_task_worker.importlib.util, "find_spec", lambda name: None,
        )

        cropped_identity = ModelIdentity(
            model_id="misleading-full-model-alias",
            engine="llama_cpp",
            format="gguf",
            revision="loaded-through-ordinary-host",
            sha256=artifact_sha256,
        )
        scheduler = _scheduler(
            full_model_loaded=True,
            full_model_identity=cropped_identity,
        )

        caps = scheduler._task_worker_capabilities()

        assert caps["stage_types"] == ["layer_forward"]
        assert caps["models"] == [{
            "model_id": gguf.name,
            "engine": "llama_cpp",
            "format": "gguf",
            "revision": "2",
            "sha256": artifact_sha256,
        }]
        assert scheduler._task_worker_full_model_identities(caps) == []
        assert _reject_stage_offer(
            scheduler,
            stage_type="full_inference",
            model_identity=cropped_identity.snapshot(),
        ) == "unsupported_stage_type"
        assert _reject_stage_offer(
            scheduler,
            stage_type="aggregate",
            model_identity=cropped_identity.snapshot(),
        ) == "unsupported_stage_type"

    def test_corrupt_or_incomplete_artifact_is_not_advertised(
            self, tmp_path, monkeypatch):
        import json

        gguf = tmp_path / "cut-16-20.gguf"
        gguf.write_bytes(b"actual artifact")
        manifest = gguf.with_suffix(".manifest.json")
        manifest.write_text(json.dumps({
            "source_layer_range": [16, 20],
            "artifact_sha256": "a" * 64,
            "mode": "middle",
        }), encoding="utf-8")
        monkeypatch.setenv("QLH_LAYER_GGUF", str(gguf))

        caps = _capabilities(
            active_layer_config={"layer_range": [16, 20], "engine": "llama_cpp"},
        )

        assert caps["layer_worker"] is False
        assert caps["layer_ranges"] == []
        assert "layer_forward" not in caps["stage_types"]
        assert "layer_artifacts" not in caps

        # A valid digest without a boundary mode is equally unsafe: the
        # scheduler must not downgrade it into a legacy exact-range worker.
        manifest.write_text(json.dumps({
            "source_layer_range": [16, 20],
            "artifact_sha256": hashlib.sha256(b"actual artifact").hexdigest(),
        }), encoding="utf-8")
        caps = _capabilities()
        assert caps["layer_worker"] is False
        assert caps["layer_ranges"] == []


class TestLayerCapabilitiesSurviveProtocolValidation:
    """声明的形状必须能被协议侧接受（否则 hello 直接被拒）。"""

    def test_declared_capabilities_pass_validation(self):
        caps = _capabilities(
            active_layer_config={"layer_range": [16, 20], "engine": "llama_cpp"},
        )
        from task_worker_protocol import PROTOCOL_VERSION, _validate_capabilities

        _validate_capabilities(caps, version=PROTOCOL_VERSION)

    def test_full_worker_capabilities_pass_validation(self):
        caps = _capabilities()
        from task_worker_protocol import PROTOCOL_VERSION, _validate_capabilities

        _validate_capabilities(caps, version=PROTOCOL_VERSION)


def test_binary_hidden_compatibility_switch_omits_the_unknown_key(monkeypatch):
    monkeypatch.setenv("QLH_TASK_WORKER_BINARY_HIDDEN", "0")
    caps = _capabilities(active_layer_config={"layer_range": [4, 16]})
    assert "binary_hidden" not in caps
    assert caps["stage_chunked_input"] is True
    monkeypatch.setenv("QLH_TASK_WORKER_BINARY_HIDDEN", "1")
    assert _capabilities(active_layer_config={"layer_range": [4, 16]})["binary_hidden"] is True
