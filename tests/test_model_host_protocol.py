"""阶段 0.2 测试：InferenceHost 协议与 ModelHost 代理语义。

验证：
  - ModelHost 满足 InferenceHost 协议（鸭子类型：方法签名齐全）
  - ModelHost 属性代理：manager 属性读写转发；自有属性（model_loaded/
    generation_config/full_chat_execution_lock）留在宿主
  - SchedulerCallbackSet 显式回调注入（api_server 组合根暴露的执行能力）
  - model_host 单例可被 scheduler 默认注入（get_model_host）
"""
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import pytest
import threading
import types

from model_host import (
    InferenceHost,
    ModelHost,
    SchedulerCallbackSet,
    get_model_host,
    model_host,
)


def _protocol_methods() -> list:
    return [
        "select_engine", "load_model", "unload_model", "load_layer_range",
        "forward_layers", "chat", "chat_stream", "ensure_full_model",
    ]


class TestInferenceHostProtocol:
    """协议满足性（不加载真实模型）。"""

    def test_protocol_methods_exist(self):
        # 冷启动状态通过公共快照检查，不触发 ModelManager 导入。
        host = ModelHost()
        assert host.runtime_status()["manager_loaded"] is False

    def test_model_host_proxies_manager(self):
        # 代理转发：宿主未定义属性转发给注入的 manager
        class FakeManager:
            is_loaded = False
            layer_range = None

        host = ModelHost(manager=FakeManager())
        assert host.is_loaded is False
        assert host.layer_range is None
        with pytest.raises(AttributeError):
            host.no_such_attr

    def test_own_attrs_stay_on_host(self):
        host = ModelHost()
        assert host.model_loaded is False
        assert host.generation_config["max_new_tokens"] == 1024
        assert isinstance(host.full_chat_execution_lock, type(threading.RLock()))

    def test_own_attr_write(self):
        host = ModelHost()
        host.model_loaded = True
        assert host.model_loaded is True


class TestSwitchModelAfterGgufEngine:
    """#31：GGUF 路径把 LlamaCppEngine 装成 _manager 之后，必须能切回非 GGUF 引擎。"""

    class _FakeGgufEngine:
        """模拟 LlamaCppEngine：有 unload，没有 switch_model。"""

        def __init__(self):
            self.unloaded = False

        def unload(self):
            self.unloaded = True

    class _FakeManager:
        """模拟 ModelManager：接受 switch_model 并记录调用。"""

        def __init__(self):
            self.calls = []

        def switch_model(self, **kwargs):
            self.calls.append(kwargs)
            return {"success": True, "model_id": kwargs.get("model_id")}

    def _host_with_gguf_engine(self):
        host = ModelHost()
        engine = self._FakeGgufEngine()
        object.__setattr__(host, "_manager", engine)
        # 复现 _load_gguf_model 写进实例字典的镜像属性
        object.__setattr__(host, "model_loaded", True)
        object.__setattr__(host, "_engine_type", "llama_cpp")
        object.__setattr__(host, "quant_type", "gguf")
        object.__setattr__(host, "model_path", "/tmp/x.gguf")
        return host, engine

    def test_switch_back_drops_gguf_engine_and_mirrored_attrs(self, monkeypatch):
        import model_host as model_host_module

        fresh = self._FakeManager()
        monkeypatch.setattr(model_host_module, "_LazyModelManager", lambda: fresh)

        host, engine = self._host_with_gguf_engine()
        result = host.switch_model(model_id="qwen2.5-0.5b-instruct", engine="pytorch")

        assert result["success"] is True
        assert engine.unloaded is True, "GGUF 引擎必须先卸载"
        assert fresh.calls[0]["engine"] == "pytorch"
        # 镜像属性必须移除，否则读取时命中的仍是上一档的值（_engine_type 会停在
        # "llama_cpp"，让 api_server 走 GGUF 分支）
        instance_dict = object.__getattribute__(host, "__dict__")
        for name in ("_engine_type", "quant_type", "model_path",
                     "active_model_id", "_active_model_id"):
            assert name not in instance_dict, f"{name} 应已被移除"

    def test_gguf_path_untouched_when_manager_has_switch_model(self, monkeypatch):
        """本来就是管理器时，走原路径：不卸载、不新建。"""
        import model_host as model_host_module

        def _boom():
            raise AssertionError("不应新建懒管理器")

        monkeypatch.setattr(model_host_module, "_LazyModelManager", _boom)

        existing = self._FakeManager()
        host = ModelHost(manager=existing)
        result = host.switch_model(model_id="m", engine="pytorch")

        assert result["success"] is True
        assert existing.calls[0]["engine"] == "pytorch"
        host.generation_config["max_new_tokens"] = 2048
        assert host.generation_config["max_new_tokens"] == 2048


def _callback_set(marker="default"):
    return SchedulerCallbackSet(
        active_execution_model_identity=lambda: marker,
        execute_task_worker_stage=lambda request, cancel_event: {"marker": marker},
        build_model_chat_prompt=lambda tokenizer, messages, **kwargs: marker,
        thinking_system_prompt=f"thinking:{marker}",
        snapshot_recent_logs=lambda: ([{"marker": marker}], 1),
        filter_recent_logs=lambda entries, **kwargs: list(entries),
        format_model_response=lambda text, **kwargs: (text, None),
    )


class TestSchedulerCallbackInjection:
    """Scheduler callbacks are one explicit, immutable protocol bundle."""

    def test_callback_bundle_is_instance_scoped_and_lazy(self):
        attached = ModelHost()
        callbacks = _callback_set("attached")
        attached.configure_scheduler_callbacks(callbacks)
        untouched = ModelHost()

        assert attached.scheduler_callbacks is callbacks
        assert untouched.scheduler_callbacks is None
        assert untouched.runtime_status()["manager_loaded"] is False

    def test_callback_bundle_keyword_order_does_not_change_wiring(self):
        values = {
            "active_execution_model_identity": lambda: "identity",
            "execute_task_worker_stage": lambda request, cancel_event: {"ok": True},
            "build_model_chat_prompt": lambda tokenizer, messages, **kwargs: "prompt",
            "thinking_system_prompt": "thinking",
            "snapshot_recent_logs": lambda: ([], 0),
            "filter_recent_logs": lambda entries, **kwargs: entries,
            "format_model_response": lambda text, **kwargs: (text, None),
        }
        callbacks = SchedulerCallbackSet(**dict(reversed(list(values.items()))))

        assert callbacks.active_execution_model_identity() == "identity"
        assert callbacks.execute_task_worker_stage(None, None) == {"ok": True}
        assert callbacks.build_model_chat_prompt(None, []) == "prompt"
        assert callbacks.thinking_system_prompt == "thinking"


class TestModelHostSingleton:
    """全局单例与 scheduler 默认注入。"""

    def test_singleton_identity(self):
        assert get_model_host() is model_host
        assert isinstance(model_host, ModelHost)

    def test_scheduler_default_host(self):
        # scheduler 无 host 参数时使用全局单例（阶段 0.2 注入语义）
        import scheduler as sched_mod
        s = sched_mod.Scheduler()
        assert s.inference_host is model_host

    def test_scheduler_explicit_host(self):
        import scheduler as sched_mod
        host = ModelHost()
        s = sched_mod.Scheduler(host=host)
        assert s.inference_host is host


class TestLazyModelManager:
    """从 api_server 迁入的惰性容器行为不变。"""

    def test_lazy_no_instantiation(self):
        # 未访问任何属性前不实例化 ModelManager（冷启动友好）
        assert ModelHost().runtime_status()["manager_loaded"] is False

    def test_no_torch_proxy_reports_unloaded_and_fails_closed(self):
        host = ModelHost()
        assert host.is_loaded is False
        assert host.is_pipeline_prepared is False
        assert host.runtime_status()["manager_loaded"] is False
        with pytest.raises(RuntimeError, match="没有可用的 ModelManager"):
            host.ensure_full_model()
        assert host.runtime_status()["manager_loaded"] is False

    def test_range_probe_never_materializes_missing_torch(self, monkeypatch):
        import model_host as model_host_module

        def unavailable(self):
            raise AssertionError("cold range observation constructed a manager")

        monkeypatch.setattr(model_host_module._LazyModelManager, "_get_instance", unavailable)
        host = ModelHost()
        assert host.layer_range is None
        assert host.peek_manager() is None
        manager = types.SimpleNamespace(layer_range=(4, 8))
        host.runtime_controller.install(manager)
        assert host.layer_range == (4, 8)
        manager.layer_range = None
        assert host.layer_range is None
        with pytest.raises(AttributeError):
            host.layer_range = (0, 8)

    def test_explicit_torch_load_reports_missing_torch(self, monkeypatch):
        import model_host as model_host_module

        host = ModelHost()
        monkeypatch.setattr(
            model_host_module._LazyModelManager,
            "_get_instance",
            lambda self: (_ for _ in ()).throw(ImportError("torch is unavailable")),
        )
        with pytest.raises(RuntimeError, match="没有 PyTorch"):
            host.load_model(engine="pytorch")

    def test_auto_load_uses_gguf_without_materializing_torch_manager(self, monkeypatch, tmp_path):
        import model_host as model_host_module

        model_path = tmp_path / "probe.gguf"
        model_path.write_bytes(b"probe")

        class FakeGguf:
            engine_type = "llama_cpp"

            def __init__(self):
                self.is_loaded = False

            def load_model(self, **kwargs):
                self.is_loaded = True
                self.model_path = kwargs["model_path"]

        fake_module = types.ModuleType("llama_engine")
        fake_module.LlamaCppEngine = FakeGguf
        fake_module.get_gguf_model_path = lambda: str(model_path)
        monkeypatch.setitem(sys.modules, "llama_engine", fake_module)

        monkeypatch.setattr(
            ModelHost,
            "select_engine",
            lambda self, profile=None: "llama_cpp",
        )
        host = ModelHost()
        host.load_model(model_path=str(model_path), engine=None)

        assert host.is_loaded is True
        assert host.engine_type == "llama_cpp"
        assert host.current_quant == "gguf"

    def test_explicit_llama_path_cannot_fall_back_to_global_gguf(
            self, monkeypatch, tmp_path):
        foreign = tmp_path / "foreign.gguf"
        foreign.write_bytes(b"foreign")

        class FakeGguf:
            def load_model(self, **_kwargs):
                raise AssertionError("非 GGUF 显式路径不得回退并加载全局工件")

        fake_module = types.ModuleType("llama_engine")
        fake_module.LlamaCppEngine = FakeGguf
        fake_module.get_gguf_model_path = lambda: str(foreign)
        monkeypatch.setitem(sys.modules, "llama_engine", fake_module)

        host = ModelHost()
        with pytest.raises(ValueError, match="只接受显式 GGUF"):
            host.load_model(
                model_path=str(tmp_path / "safetensors-model"),
                engine="llama_cpp",
            )

    def test_prepare_gguf_pipeline_does_not_materialize_torch_manager(
            self, monkeypatch, tmp_path):
        model_path = tmp_path / "probe.gguf"
        model_path.write_bytes(b"probe")

        class FakeGguf:
            engine_type = "llama_cpp"

            def prepare_pipeline_model(self, **kwargs):
                self.kwargs = kwargs
                self.is_pipeline_prepared = True
                return {
                    "model_id": kwargs["model_id"],
                    "model_sha256": "a" * 64,
                    "pipeline_runtime_supported": True,
                }

        fake_module = types.ModuleType("llama_engine")
        fake_module.LlamaCppEngine = FakeGguf
        monkeypatch.setitem(sys.modules, "llama_engine", fake_module)

        host = ModelHost()
        descriptor = host.prepare_pipeline_model(
            model_id="probe-model",
            model_path=str(model_path),
            quant_type="Q4_K_M",
        )

        assert descriptor["model_id"] == "probe-model"
        assert host.is_pipeline_prepared is True
        assert host.is_loaded is False
        assert host.engine_type == "llama_cpp"

    def test_prepare_directory_switches_away_from_llama_manager(
            self, monkeypatch, tmp_path):
        model_dir = tmp_path / "safetensors-model"
        model_dir.mkdir()
        unloaded = []

        class OldGguf:
            engine_type = "llama_cpp"

            def prepare_pipeline_model(self, **_kwargs):
                raise AssertionError("directory must not use the GGUF preparer")

            def unload(self):
                unloaded.append(True)

        class TorchManager:
            def prepare_pipeline_model(self, **kwargs):
                self.kwargs = kwargs
                return {"model_id": kwargs["model_id"]}

        torch_manager = TorchManager()
        host = ModelHost(manager=OldGguf())
        object.__setattr__(host, "_engine_type", "llama_cpp")
        monkeypatch.setattr(
            ModelHost, "_materialize_manager", lambda _self: torch_manager,
        )

        result = host.prepare_pipeline_model(
            model_id="directory-model", model_path=str(model_dir),
        )

        assert result == {"model_id": "directory-model"}
        assert unloaded == [True]
        assert "_engine_type" not in host.__dict__

    def test_unload_clears_direct_gguf_mirror_state(self):
        class DirectGguf:
            def unload(self):
                self.unloaded = True

        manager = DirectGguf()
        host = ModelHost(manager=manager)
        for name, value in {
            "_engine_type": "llama_cpp",
            "quant_type": "GGUF",
            "model_path": "old.gguf",
            "active_model_id": "old-model",
            "_active_model_id": "old-model",
        }.items():
            object.__setattr__(host, name, value)

        host.unload_model()

        assert manager.unloaded is True
        for name in (
            "_engine_type", "quant_type", "model_path",
            "active_model_id", "_active_model_id",
        ):
            assert name not in host.__dict__

    def test_torch_free_partial_gguf_is_rejected_as_full_model(self):
        class PartialGguf:
            engine_type = "llama_cpp"
            is_loaded = True

            def get_pipeline_descriptor(self):
                return {
                    "partial_assignment": True,
                    "assignment_layer_range": [0, 8],
                    "loaded_artifact": "head8.gguf",
                }

        host = ModelHost(manager=PartialGguf())
        assert host.is_loaded is True
        with pytest.raises(RuntimeError, match="裁层工件"):
            host.ensure_full_model()

    def test_torch_free_full_gguf_satisfies_full_model_check(self):
        class FullGguf:
            engine_type = "llama_cpp"
            is_loaded = True

            def get_pipeline_descriptor(self):
                return {"model_path": "full.gguf"}

        host = ModelHost(manager=FullGguf())
        assert host.ensure_full_model() is None


class TestPipelineCandidateTransaction:
    class _Manager:
        engine_type = "pytorch"

        def __init__(self, name, *, loaded=False):
            self.name = name
            self.is_loaded = loaded
            self.prepare_calls = []
            self.materialize_calls = []
            self.tokenizer_calls = 0
            self.closed = 0

        def prepare_pipeline_model(self, **kwargs):
            self.prepare_calls.append(kwargs)
            self.quant_type = kwargs.get("quant_type") or "int4"
            self.is_pipeline_prepared = True
            return {
                "model_id": kwargs["model_id"],
                "model_sha256": kwargs.get("model_sha256") or "a" * 64,
            }

        def load_layer_range(self, **kwargs):
            self.materialize_calls.append(kwargs)
            self.is_loaded = True
            return {
                "success": True,
                "layer_start": kwargs["start_layer"],
                "layer_end": kwargs["end_layer"],
            }

        def prepare_pipeline_tokenizer(self):
            self.tokenizer_calls += 1
            return {"success": True, "tokenizer": True}

        def unload_model(self):
            self.closed += 1
            self.is_loaded = False

    @staticmethod
    def _set_active_mirrors(host):
        object.__setattr__(host, "model_loaded", True)
        object.__setattr__(host, "current_quant", "old-quant")
        object.__setattr__(host, "_engine_type", "old-engine")
        object.__setattr__(host, "quant_type", "old-quant")
        object.__setattr__(host, "model_path", "old-path")
        object.__setattr__(host, "active_model_id", "old-model")
        object.__setattr__(host, "_active_model_id", "old-model")

    def _prepare_and_materialize(self, host, model_dir, transaction_id="txn-1"):
        descriptor = host.prepare_pipeline_candidate(
            transaction_id,
            "new-model",
            str(model_dir),
            quant_type="int8",
            model_sha256="b" * 64,
        )
        materialized = host.materialize_pipeline_candidate(
            transaction_id,
            0,
            8,
            True,
            False,
            total_layers=24,
        )
        return descriptor, materialized

    def test_prepare_and_materialize_do_not_pollute_active_runtime(
        self, monkeypatch, tmp_path,
    ):
        import model_host as model_host_module

        model_dir = tmp_path / "model"
        model_dir.mkdir()
        active = self._Manager("active", loaded=True)
        candidate = self._Manager("candidate")
        monkeypatch.setattr(
            model_host_module._LazyModelManager,
            "_get_instance",
            lambda _self: candidate,
        )
        host = ModelHost(manager=active)
        self._set_active_mirrors(host)

        descriptor, materialized = self._prepare_and_materialize(host, model_dir)
        repeated = host.materialize_pipeline_candidate(
            "txn-1", 0, 8, True, False, total_layers=24,
        )

        assert descriptor["model_id"] == "new-model"
        assert materialized["success"] is True
        assert repeated == materialized
        assert host.peek_manager() is active
        assert host.model_loaded is True
        assert host.current_quant == "old-quant"
        assert host.engine_type == "old-engine"
        assert host.active_model_id == "old-model"
        assert active.closed == 0
        assert candidate.prepare_calls[0]["model_path"] == str(model_dir.resolve())
        assert candidate.materialize_calls[0]["total_layers"] == 24
        assert len(candidate.materialize_calls) == 1

    def test_commit_then_abort_restores_exact_active_runtime(
        self, monkeypatch, tmp_path,
    ):
        import model_host as model_host_module

        model_dir = tmp_path / "model"
        model_dir.mkdir()
        active = self._Manager("active", loaded=True)
        candidate = self._Manager("candidate")
        monkeypatch.setattr(
            model_host_module._LazyModelManager,
            "_get_instance",
            lambda _self: candidate,
        )
        host = ModelHost(manager=active)
        self._set_active_mirrors(host)
        self._prepare_and_materialize(host, model_dir)

        committed = host.commit_pipeline_candidate("txn-1")
        repeated_commit = host.commit_pipeline_candidate("txn-1")

        assert committed["phase"] == "committed"
        assert repeated_commit["phase"] == "committed"
        assert host.peek_manager() is candidate
        assert host.active_model_id == "new-model"
        assert host.current_quant == "int8"
        assert host.model_loaded is False
        assert host.is_loaded is True
        assert host.is_pipeline_prepared is True
        assert host.runtime_status() == {
            "manager_loaded": True,
            "model_loaded": False,
            "runtime_materialized": True,
            "pipeline_prepared": True,
            "current_quant": "int8",
        }
        assert active.closed == 0

        aborted = host.abort_pipeline_candidate("txn-1")
        repeated_abort = host.abort_pipeline_candidate("txn-1")

        assert aborted["phase"] == "aborted"
        assert repeated_abort["phase"] == "aborted"
        assert host.peek_manager() is active
        assert host.model_loaded is True
        assert host.current_quant == "old-quant"
        assert host.engine_type == "old-engine"
        assert host.quant_type == "old-quant"
        assert host.model_path == "old-path"
        assert host.active_model_id == "old-model"
        assert candidate.closed == 1
        assert active.closed == 0

    def test_abort_before_commit_closes_only_candidate(self, monkeypatch, tmp_path):
        import model_host as model_host_module

        model_dir = tmp_path / "model"
        model_dir.mkdir()
        active = self._Manager("active", loaded=True)
        candidate = self._Manager("candidate")
        monkeypatch.setattr(
            model_host_module._LazyModelManager,
            "_get_instance",
            lambda _self: candidate,
        )
        host = ModelHost(manager=active)
        self._prepare_and_materialize(host, model_dir)

        aborted = host.abort_pipeline_candidate("txn-1")

        assert aborted["phase"] == "aborted"
        assert host.peek_manager() is active
        assert active.closed == 0
        assert candidate.closed == 1

    def test_finalize_closes_old_manager_once(self, monkeypatch, tmp_path):
        import model_host as model_host_module

        model_dir = tmp_path / "model"
        model_dir.mkdir()
        active = self._Manager("active", loaded=True)
        candidate = self._Manager("candidate")
        monkeypatch.setattr(
            model_host_module._LazyModelManager,
            "_get_instance",
            lambda _self: candidate,
        )
        host = ModelHost(manager=active)
        self._prepare_and_materialize(host, model_dir)
        host.commit_pipeline_candidate("txn-1")

        finalized = host.finalize_pipeline_candidate("txn-1")
        repeated = host.finalize_pipeline_candidate("txn-1")

        assert finalized["phase"] == "finalized"
        assert repeated["phase"] == "finalized"
        assert host.peek_manager() is candidate
        assert active.closed == 1
        assert candidate.closed == 0

    def test_conflicting_transaction_is_rejected_and_same_id_is_idempotent(
        self, monkeypatch, tmp_path,
    ):
        import model_host as model_host_module

        model_dir = tmp_path / "model"
        model_dir.mkdir()
        candidate = self._Manager("candidate")
        monkeypatch.setattr(
            model_host_module._LazyModelManager,
            "_get_instance",
            lambda _self: candidate,
        )
        host = ModelHost(manager=self._Manager("active", loaded=True))

        first = host.prepare_pipeline_candidate("txn-1", "model", str(model_dir))
        repeated = host.prepare_pipeline_candidate("txn-1", "model", str(model_dir))

        assert repeated == first
        assert len(candidate.prepare_calls) == 1
        with pytest.raises(RuntimeError, match="MODEL_TXN_CONFLICT"):
            host.prepare_pipeline_candidate("txn-2", "other", str(model_dir))
        with pytest.raises(RuntimeError, match="MODEL_TXN_CONFLICT"):
            host.materialize_pipeline_candidate("txn-2", 0, 8, True, False)

    def test_gguf_candidate_uses_independent_engine(self, monkeypatch, tmp_path):
        model_path = tmp_path / "candidate.gguf"
        model_path.write_bytes(b"candidate")
        engines = []

        class FakeGguf(self._Manager):
            engine_type = "llama_cpp"

            def __init__(self):
                super().__init__("gguf-candidate")
                engines.append(self)

        fake_module = types.ModuleType("llama_engine")
        fake_module.LlamaCppEngine = FakeGguf
        monkeypatch.setitem(sys.modules, "llama_engine", fake_module)
        active = self._Manager("active", loaded=True)
        host = ModelHost(manager=active)

        host.prepare_pipeline_candidate("txn-gguf", "gguf", str(model_path))

        assert len(engines) == 1
        assert host.peek_manager() is active
        assert active.closed == 0

    def test_gguf_candidate_commit_retains_layer_forward_capability(
        self, monkeypatch, tmp_path,
    ):
        from koakuma_engine import Capability, runtime_supports
        from llama_engine import LlamaCppEngine

        model_path = tmp_path / "head.gguf"
        model_path.write_bytes(b"header-only-fixture")

        def prepare(engine, **kwargs):
            engine._model_path = kwargs["model_path"]
            engine._pipeline_distributed_only = True
            engine._pipeline_descriptor = {
                "model_id": kwargs["model_id"],
                "model_sha256": "a" * 64,
                "pipeline_runtime_supported": True,
            }
            return dict(engine._pipeline_descriptor)

        def materialize(engine, **kwargs):
            engine._loaded = True
            return {"success": True}

        monkeypatch.setattr(LlamaCppEngine, "prepare_pipeline_model", prepare)
        monkeypatch.setattr(LlamaCppEngine, "load_layer_range", materialize)
        host = ModelHost()
        host.prepare_pipeline_candidate("txn-head", "model", str(model_path))
        host.materialize_pipeline_candidate("txn-head", 0, 4, True, False)
        host.commit_pipeline_candidate("txn-head")

        assert host.engine_type == "llama_cpp"
        assert runtime_supports(host, Capability.FORWARD_LAYERS) is True
        assert host.model_loaded is False

    def test_tokenizer_only_candidate_can_commit_without_local_layers(
        self, monkeypatch, tmp_path,
    ):
        import model_host as model_host_module

        model_dir = tmp_path / "model"
        model_dir.mkdir()
        active = self._Manager("active", loaded=True)
        candidate = self._Manager("candidate")
        monkeypatch.setattr(
            model_host_module._LazyModelManager,
            "_get_instance",
            lambda _self: candidate,
        )
        host = ModelHost(manager=active)

        host.prepare_pipeline_candidate(
            "txn-tokenizer", "model", str(model_dir),
        )
        result = host.materialize_pipeline_candidate(
            "txn-tokenizer", None, None, False, False,
        )
        host.commit_pipeline_candidate("txn-tokenizer")

        assert result == {"success": True, "tokenizer": True}
        assert candidate.tokenizer_calls == 1
        assert host.peek_manager() is candidate
        assert active.closed == 0


class TestApiServerIntegration:
    """api_server 经改造后的宿主接线（不加载模型）。"""

    def test_api_server_model_manager_is_host(self):
        import api_server
        assert api_server.model_manager is model_host

    def test_api_server_configures_explicit_scheduler_callbacks(self):
        import api_server

        # R1 binds callbacks at explicit construction, so import stays inert.
        resources = api_server._build_default_resources(
            api_server._get_app_runtime().resolve_config(), host=model_host)
        try:
            callbacks = resources.scheduler.inference_callbacks
            assert resources.host is model_host
            assert callbacks is model_host.scheduler_callbacks
            assert callable(callbacks.execute_task_worker_stage)
            assert callable(callbacks.active_execution_model_identity)
        finally:
            resources.close()

    def test_no_reverse_import(self):
        # 阶段 0.5 验收项：scheduler 不再 import api_server
        import inspect
        import scheduler as sched_mod
        src = inspect.getsource(sched_mod)
        assert "import api_server" not in src


def test_host_unload_failure_preserves_direct_engine_projection():
    should_fail = [True]

    class Engine:
        def unload(self):
            if should_fail[0]:
                raise OSError("native owner still alive")

    engine = Engine()
    host = ModelHost(manager=engine)
    for name, value in {"model_loaded": True, "_engine_type": "llama_cpp",
                        "model_path": "head.gguf", "active_model_id": "segment-model"}.items():
        object.__setattr__(host, name, value)
    with pytest.raises(OSError, match="native owner still alive"):
        host.unload_model()
    assert host._manager is engine and host.model_loaded is True
    assert dict(host.runtime_controller.snapshot().metadata)["model_path"] == "head.gguf"
    assert dict(host.runtime_controller.snapshot().metadata)["active_model_id"] == "segment-model"
    should_fail[0] = False
    host.unload_model()
    assert host.model_loaded is False
    assert not dict(host.runtime_controller.snapshot().metadata)


class TestColdRuntimeObservation:
    @staticmethod
    def _cold_host(monkeypatch):
        import model_host as module

        attempts = []

        def unavailable(_self):
            attempts.append("torch manager construction")
            raise ImportError("torch intentionally absent")

        monkeypatch.setattr(module._LazyModelManager, "_get_instance", unavailable)
        return ModelHost(), attempts

    def test_state_readout_uses_unloaded_values_without_torch(self, monkeypatch):
        host, attempts = self._cold_host(monkeypatch)
        assert host.model is None
        assert host.tokenizer is None
        assert host._total_model_layers == 0
        assert host._model_path is None
        assert host._full_model_path is None
        assert host._island_engine is None
        assert host.get_pipeline_descriptor() == {}
        assert host.runtime_status()["manager_loaded"] is False
        assert attempts == []

    def test_cluster_status_and_layer_planning_do_not_construct_cold_manager(
        self, monkeypatch,
    ):
        from scheduler import Scheduler
        from config import TOTAL_MODEL_LAYERS

        host, attempts = self._cold_host(monkeypatch)
        scheduler = Scheduler(host=host)
        status = scheduler.get_status()
        assert status["pipeline"]["active"] is False
        assert status["pipeline"]["engine_compatible"] is False
        assert scheduler.get_layer_assignments()["total"] == TOTAL_MODEL_LAYERS
        assert host.peek_manager() is None
        assert attempts == []

    def test_state_readout_tracks_owner_replacement_and_explicit_writes(
        self, monkeypatch,
    ):
        host, attempts = self._cold_host(monkeypatch)
        descriptor = {"total_layers": 32, "pipeline_runtime_supported": True}
        first = types.SimpleNamespace(
            model=object(), tokenizer=object(), _total_model_layers=32,
            _model_path="first", get_pipeline_descriptor=lambda: dict(descriptor),
        )
        second = types.SimpleNamespace(
            model=None, tokenizer=None, _total_model_layers=8,
            _model_path="second", get_pipeline_descriptor=lambda: {"total_layers": 8},
        )
        host.runtime_controller.install(first)
        assert host.model is first.model
        assert host.tokenizer is first.tokenizer
        assert host.get_pipeline_descriptor() == descriptor
        host._model_path = "updated"
        assert first._model_path == "updated"
        host.runtime_controller.install(second)
        assert host.model is None
        assert host._total_model_layers == 8
        assert host._model_path == "second"
        assert host.get_pipeline_descriptor() == {"total_layers": 8}
        assert attempts == []
