import os
from pathlib import Path
import subprocess
import sys
import textwrap
import threading
import types
import asyncio
import pytest
from fastapi.testclient import TestClient

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import api_server
import model_host as model_host_module
from model_host import model_host


def _isolated_api_probe_env(tmp_path: Path) -> dict[str, str]:
    """Build a cold-start environment independent of developer runtime state."""

    repo_root = Path(__file__).resolve().parents[1]
    env = os.environ.copy()
    env.update({
        "QLH_INFERENCE_ENGINE": "llama_cpp",
        "QLH_STATE_DIR": str(tmp_path / "state"),
        "QLH_NODE_CONFIG_PATH": str(tmp_path / "node-config.json"),
        "QLH_TASK_GRAPH_ENABLED": "false",
        "QLH_TASK_WORKER_EXPERIMENTAL_ENABLED": "false",
        "QLH_TASK_GRAPH_INSTANCE_ID": f"cold-probe-{tmp_path.name}",
    })
    env["PYTHONPATH"] = os.pathsep.join(
        item for item in (str(repo_root / "src"), env.get("PYTHONPATH", "")) if item
    )
    return env


def test_shared_lifespans_reference_count_runtime(monkeypatch):
    """Two uvicorn servers must share one runtime lifetime and readiness state."""
    startup_calls = []
    shutdown_calls = []
    readiness_before = dict(api_server._runtime_readiness)
    with api_server._runtime_lifecycle_lock:
        api_server._runtime_lifecycle_users = 0

    def fake_startup():
        startup_calls.append(True)
        api_server._runtime_startup_done.set()

    async def fake_shutdown():
        shutdown_calls.append(True)

    monkeypatch.setattr(api_server, "_run_runtime_startup", fake_startup)
    monkeypatch.setattr(api_server, "_shutdown_resources", fake_shutdown)

    async def exercise():
        first = api_server._lifespan(api_server.app)
        await first.__aenter__()
        await asyncio.sleep(0)
        with api_server._runtime_readiness_lock:
            api_server._runtime_readiness["ready"] = True

        second = api_server._lifespan(api_server.app)
        await second.__aenter__()
        assert startup_calls == [True]
        with api_server._runtime_readiness_lock:
            assert api_server._runtime_readiness["ready"] is True

        await first.__aexit__(None, None, None)
        assert shutdown_calls == []
        with api_server._runtime_lifecycle_lock:
            assert api_server._runtime_lifecycle_users == 1

        await second.__aexit__(None, None, None)
        assert shutdown_calls == [True]
        with api_server._runtime_lifecycle_lock:
            assert api_server._runtime_lifecycle_users == 0

    try:
        asyncio.run(exercise())
    finally:
        with api_server._runtime_lifecycle_lock:
            api_server._runtime_lifecycle_users = 0
        with api_server._runtime_readiness_lock:
            api_server._runtime_readiness.clear()
            api_server._runtime_readiness.update(readiness_before)


def test_lazy_model_manager_defers_construction(monkeypatch):
    calls: list[str] = []

    fake_module = types.ModuleType("model_module")

    class FakeModelManager:
        def __init__(self):
            calls.append("init")
            self.is_loaded = True

    fake_module.ModelManager = FakeModelManager
    monkeypatch.setitem(sys.modules, "model_module", fake_module)

    proxy = model_host_module._LazyModelManager()

    assert calls == []
    assert repr(proxy) == "<_LazyModelManager unloaded>"

    assert proxy.is_loaded is True
    assert calls == ["init"]


def test_available_models_does_not_touch_model_manager_when_unloaded(monkeypatch):
    class BombModelManager:
        def __getattr__(self, name):
            raise AssertionError(f"model_manager should stay lazy, accessed {name}")

    monkeypatch.setattr(model_host, "model_loaded", False)
    monkeypatch.setattr(api_server, "model_manager", BombModelManager())
    # The endpoint serializes model_config availability itself.  Patch the
    # direct config source so this lazy-manager test cannot pass by relying on
    # whatever model files happen to exist on the developer machine.
    monkeypatch.setattr(api_server, "_get_all_model_configs", lambda: [])

    result = asyncio.run(api_server.list_available_models())

    assert result["current"] is None
    assert result["current_engine"] is None


def test_frontend_bootstrap_endpoints_keep_model_manager_lazy(monkeypatch):
    fake_module = types.ModuleType("model_module")

    def fail_on_model_manager_access(name):
        raise AssertionError(f"model_module should not be imported for bootstrap endpoints: {name}")

    fake_module.__getattr__ = fail_on_model_manager_access
    monkeypatch.setitem(sys.modules, "model_module", fake_module)
    monkeypatch.setattr(model_host, "model_loaded", False)
    monkeypatch.setattr(api_server, "model_manager", model_host_module._LazyModelManager())

    class FakeScheduler:
        _max_nodes = 3

        def _effective_role(self):
            return "master"

        def get_effective_node_id(self):
            return "test-master"

    monkeypatch.setattr(api_server, "scheduler", FakeScheduler())
    monkeypatch.setattr(api_server, "_get_registered_experimental_models", lambda: [])

    client = TestClient(api_server.app)

    for path in (
        "/api/status",
        "/api/models/current",
        "/api/models/available",
        "/api/models",
    ):
        response = client.get(path)
        assert response.status_code == 200, path


def test_l_tier_cold_start_gguf_inference_and_tui_without_torch(tmp_path):
    """The L-tier must boot and run its GGUF path when torch cannot be imported."""
    repo_root = Path(__file__).resolve().parents[1]
    src_root = repo_root / "src"
    env = _isolated_api_probe_env(tmp_path)
    probe = textwrap.dedent(
        """
        import importlib.abc
        import pathlib
        import sys
        import tempfile
        import types


        class BlockTorch(importlib.abc.MetaPathFinder):
            def find_spec(self, fullname, path=None, target=None):
                if fullname == "torch" or fullname.startswith("torch."):
                    raise ModuleNotFoundError("torch intentionally blocked for L-tier probe")
                return None


        sys.meta_path.insert(0, BlockTorch())
        sys.path.insert(0, {src_root!r})

        import api_server

        assert "torch" not in sys.modules
        assert api_server.model_host.runtime_status()["manager_loaded"] is False

        fake_llama = types.ModuleType("llama_engine")
        model_path = pathlib.Path(tempfile.gettempdir()) / "qlh-l-tier-probe.gguf"
        model_path.write_bytes(b"probe")

        class FakeLlamaCppEngine:
            def __init__(self):
                self.is_loaded = False

            def load_model(self, **kwargs):
                self.model_path = kwargs["model_path"]
                self.is_loaded = True

            def chat(self, messages, **kwargs):
                return {{"content": "probe-response"}}

        fake_llama.LlamaCppEngine = FakeLlamaCppEngine
        fake_llama.get_gguf_model_path = lambda: str(model_path)
        sys.modules["llama_engine"] = fake_llama

        api_server.model_host.load_model(
            model_path=str(model_path),
            engine="llama_cpp",
            model_id="l-tier-probe",
        )
        result = api_server.model_host.chat([{{"role": "user", "content": "ping"}}])
        assert result["content"] == "probe-response"
        assert "torch" not in sys.modules

        import tui_commands

        # 旧标准库 TUI 已归档；单命令薄层必须在**无 torch** 的子进程里可用。
        assert tui_commands.main(["help"]) == 0
        assert "torch" not in sys.modules
        model_path.unlink(missing_ok=True)
        print("L_TIER_PROBE_OK")
        """
    ).format(src_root=str(src_root))
    completed = subprocess.run(
        [sys.executable, "-c", probe],
        cwd=repo_root,
        env=env,
        capture_output=True,
        text=True,
        # ★ 必须显式指定编码：`text=True` 缺 `encoding` 时按 **locale 默认**解码
        #   （中文 Windows 是 GBK）⇒ 探针输出里出现非 GBK 字节时 `_readerthread`
        #   抛 UnicodeDecodeError，CPython 直接把 `stdout` 置为 **None** ⇒ 下面
        #   那条 `in completed.stdout` 变成 TypeError（症状与根因相隔很远）。
        encoding="utf-8",
        errors="replace",
        timeout=30,
    )
    assert completed.returncode == 0, completed.stderr or completed.stdout
    assert "L_TIER_PROBE_OK" in completed.stdout


def test_paged_kv_cache_import_keeps_torch_out_of_default_gguf_process(tmp_path):
    """The default GGUF process must stay torch-free even when torch is installed."""
    repo_root = Path(__file__).resolve().parents[1]
    env = _isolated_api_probe_env(tmp_path)
    completed = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import sys; import paged_kv_cache; "
                "assert 'torch' not in sys.modules; print('PAGED_IMPORT_OK')"
            ),
        ],
        cwd=repo_root,
        env=env,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert completed.returncode == 0, completed.stderr or completed.stdout
    assert "PAGED_IMPORT_OK" in completed.stdout


def test_default_gguf_api_import_and_bootstrap_queries_do_not_load_torch(tmp_path):
    """The installed Torch wheel must stay cold on the default llama.cpp path."""
    repo_root = Path(__file__).resolve().parents[1]
    env = _isolated_api_probe_env(tmp_path)
    probe = (
        "import asyncio, sys; "
        "import api_server; "
        "assert 'torch' not in sys.modules; "
        "asyncio.run(api_server.health()); "
        "asyncio.run(api_server.get_status()); "
        "asyncio.run(api_server.list_available_models()); "
        "assert 'torch' not in sys.modules; "
        "print('DEFAULT_GGUF_BOOTSTRAP_OK')"
    )
    completed = subprocess.run(
        [sys.executable, "-c", probe],
        cwd=repo_root,
        env=env,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert completed.returncode == 0, completed.stderr or completed.stdout
    assert "DEFAULT_GGUF_BOOTSTRAP_OK" in completed.stdout


def test_health_is_available_while_runtime_startup_is_still_running(monkeypatch):
    """Liveness must not wait for the full local runtime startup sequence."""
    started = threading.Event()
    release = threading.Event()

    def blocked_startup():
        started.set()
        release.wait(timeout=5)

    monkeypatch.setattr(api_server, "_startup_device_detection", blocked_startup)
    try:
        with TestClient(api_server.app) as client:
            assert started.wait(timeout=1), "runtime startup thread did not start"
            health = client.get("/api/health")
            readiness = client.get("/api/ready")
            assert health.status_code == 200
            assert health.json()["status"] == "ok"
            assert readiness.status_code == 200
            assert readiness.json()["ready"] is False
            assert readiness.json()["process_ready"] is True
            release.set()
    finally:
        release.set()


def test_reserved_pipeline_worker_does_not_auto_load_full_model(monkeypatch):
    class UnloadedManager:
        is_loaded = False

    monkeypatch.setattr(model_host, "model_loaded", False)
    monkeypatch.setattr(api_server, "model_manager", UnloadedManager())
    monkeypatch.setattr(
        api_server.scheduler, "get_distributed_inference_enabled", lambda: True,
    )
    monkeypatch.setattr(api_server.scheduler, "_effective_role", lambda: "client")
    monkeypatch.setattr(
        api_server.scheduler, "has_pipeline_worker_reservation", lambda: True,
    )
    monkeypatch.setattr(api_server, "RUN_MODE", "distributed")
    auto_load_calls = []
    monkeypatch.setattr(
        api_server, "_auto_load_default_model", lambda: auto_load_calls.append(True),
    )

    api_server._ensure_chat_model_or_forwarding()

    assert auto_load_calls == []


def test_required_route_does_not_auto_load_full_model_before_pipeline(monkeypatch):
    class UnloadedManager:
        is_loaded = False
        is_pipeline_prepared = False

    monkeypatch.setattr(model_host, 'model_loaded', False)
    monkeypatch.setattr(api_server, 'model_manager', UnloadedManager())
    monkeypatch.setattr(
        api_server.scheduler, 'get_distributed_inference_enabled', lambda: True,
    )
    monkeypatch.setattr(api_server.scheduler, '_effective_role', lambda: 'master')
    monkeypatch.setattr(api_server, 'RUN_MODE', 'distributed')
    auto_load_calls = []
    monkeypatch.setattr(
        api_server, '_auto_load_default_model', lambda: auto_load_calls.append(True),
    )

    with pytest.raises(api_server.HTTPException) as exc_info:
        api_server._ensure_chat_model_or_forwarding(
            api_server.ChatRequest(
                message='ping', routing_preference='distributed_required',
            ),
        )

    assert exc_info.value.status_code == 503
    assert exc_info.value.detail == 'distributed_required_pipeline_not_ready'
    assert auto_load_calls == []


def test_reserved_worker_rejects_local_model_even_when_already_loaded(monkeypatch):
    class SegmentManager:
        is_loaded = True

    monkeypatch.setattr(model_host, "model_loaded", True)
    monkeypatch.setattr(api_server, "model_manager", SegmentManager())
    monkeypatch.setattr(
        api_server.scheduler, "get_distributed_inference_enabled", lambda: False,
    )
    monkeypatch.setattr(
        api_server.scheduler, "has_pipeline_worker_reservation", lambda: True,
    )

    with pytest.raises(api_server.HTTPException) as exc_info:
        api_server._ensure_chat_model_or_forwarding()

    assert exc_info.value.status_code == 503
