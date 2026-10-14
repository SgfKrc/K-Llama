"""Exercise the real conftest and ModelHost without native engines or local config."""
import json
import sqlite3
from pathlib import Path

import pytest

pytest_plugins = ["pytester"]

_REPO = Path(__file__).resolve().parents[1]


def _isolated_suite(pytester, *, audit_teardown=True):
    # Copy the dependency-light host, but supply harmless config/API/model stubs.
    # Each child suite uses one Python process per worker across multiple tests.
    for name in ("model_host", "koakuma_engine", "model_load_resolver", "local_store"):
        pytester.makepyfile(**{name: (_REPO / "src" / f"{name}.py").read_text(encoding="utf-8")})
    pytester.makeini("[pytest]\naddopts = -q\n")
    pytester.makepyfile(
        transformers5_compat="def install(): pass",
        config="""
            INFERENCE_ENGINE = "auto"
            QUANT_TYPE = "int4"
            USE_COMPILE = False
            EXTERNAL_ENABLED = False
            DEFAULT_LAYER_CONFIG = {"master": [0, 8], "workers": {"edge": [8, 16]}}
            LAYER_VRAM_FACTOR = {"pytorch": 1.0}
        """,
        model_module="""
            class ModelManager:
                def __init__(self):
                    self.is_loaded = False
        """,
        api_server="""
            from contextvars import ContextVar
            from threading import RLock
            from types import SimpleNamespace
            from model_host import model_host
            model_manager = model_host
            _request_id_ctx = ContextVar("isolation_request_id", default="")
            _chat_context = SimpleNamespace(
                _lock=RLock(), _committed_operations={}, _revisions={},
                histories={}, _history_generations={}, active_session_id=None,
            )
            model_host.configure_scheduler_callbacks(object())
        """,
        state="""
            from copy import deepcopy
            import api_server
            import config
            import model_host as host_module

            host = host_module.model_host
            manager = host._manager
            generation = host.generation_config
            generation_before = deepcopy(generation)
            callbacks = host.scheduler_callbacks
            execution_lock = host.full_chat_execution_lock
            layer_config = config.DEFAULT_LAYER_CONFIG
            layer_config_before = deepcopy(layer_config)
            vram_factor = config.LAYER_VRAM_FACTOR

            def assert_clean():
                assert host_module.model_host is host
                assert api_server.model_manager is host
                assert host._manager is manager
                assert object.__getattribute__(manager, "_instance") is None
                assert host.model_loaded is False
                assert host.current_quant == "int4"
                assert host.generation_config is generation
                assert generation == generation_before
                assert host.scheduler_callbacks is callbacks
                assert host.full_chat_execution_lock is execution_lock
                for name in ("_engine_type", "quant_type", "model_path", "active_model_id", "_active_model_id"):
                    assert name not in vars(host)
                assert config.INFERENCE_ENGINE == "auto"
                assert config.QUANT_TYPE == "int4"
                assert config.USE_COMPILE is False
                assert config.EXTERNAL_ENABLED is False
                assert config.DEFAULT_LAYER_CONFIG is layer_config
                assert layer_config == layer_config_before
                assert config.LAYER_VRAM_FACTOR is vram_factor
                assert vram_factor == {"pytorch": 1.0}
                assert not hasattr(config, "ISOLATION_TEMP_SETTING")
                assert api_server._request_id_ctx.get() == ""
                service = api_server._chat_context
                assert not service._committed_operations
                assert not service._revisions
                assert not service.histories
                assert not service._history_generations
                assert service.active_session_id is None
        """,
    )
    conftest = (_REPO / "tests" / "conftest.py").read_text(encoding="utf-8")
    if audit_teardown:
        conftest += """

@pytest.hookimpl(wrapper=True, tryfirst=True)
def pytest_runtest_teardown(item, nextitem):
    yield
    from state import assert_clean
    assert_clean()
    worker = getattr(item.config, "workerinput", {}).get("workerid")
    if worker is not None:
        from pathlib import Path
        Path("seen-" + worker).touch()
"""
    pytester.makeconftest(conftest)


_SEQUENCE = """
    from types import SimpleNamespace
    import pytest
    import api_server
    import config
    import model_host as host_module
    import state
    from state import assert_clean

    def test_01_direct_host_changes():
        assert_clean()
        state.host.model_loaded = True
        state.host.current_quant = "fp32"
        state.host.generation_config["temperature"] = -1
        state.host.generation_config["nested"] = {"tokens": [1]}
        state.host._manager = SimpleNamespace(is_loaded=True)
        for name in ("_engine_type", "quant_type", "model_path", "active_model_id", "_active_model_id"):
            object.__setattr__(state.host, name, "test-only")
        state.host.scheduler_callbacks = object()
        state.host.full_chat_execution_lock = object()
        api_server._request_id_ctx.set("leftover-operation")
        service = api_server._chat_context
        for name in ("_committed_operations", "_revisions", "histories", "_history_generations"):
            getattr(service, name)["leftover"] = "test-only"
        service.active_session_id = "leftover"

    def test_02_observe_after_host_changes():
        assert_clean()

    def test_03_lazy_instance_and_monkeypatch(monkeypatch):
        assert_clean()
        # Forwarded setattr materializes the actual lazy proxy. If isolation
        # runs before monkeypatch.undo(), undo will materialize it a second time.
        monkeypatch.setattr(state.host, "is_loaded", True)
        assert object.__getattribute__(state.manager, "_instance") is not None
        monkeypatch.setattr(host_module, "_LazyModelManager", lambda: None)

    def test_04_observe_after_lazy_changes():
        assert_clean()

    def test_05_mutable_config_changes():
        assert_clean()
        config.INFERENCE_ENGINE = "pytorch"
        config.QUANT_TYPE = "fp32"
        config.USE_COMPILE = True
        config.EXTERNAL_ENABLED = True
        config.DEFAULT_LAYER_CONFIG["workers"]["edge"].append(99)
        config.DEFAULT_LAYER_CONFIG["master"].clear()
        config.LAYER_VRAM_FACTOR.clear()
        config.ISOLATION_TEMP_SETTING = {"nested": [1]}
        del config.USE_COMPILE

    def test_06_observe_after_config_changes():
        assert_clean()
        # Repeat an in-place mutation to catch a baseline reused by reference.
        config.DEFAULT_LAYER_CONFIG["workers"]["edge"].append(100)
        config.LAYER_VRAM_FACTOR = {"replacement": 2.0}
        state.host.generation_config = {"replacement": True}

    @pytest.fixture
    def teardown_writer():
        yield
        state.host._manager = None
        config.QUANT_TYPE = "teardown-quant"

    def test_07_fixture_teardown_and_monkeypatch(teardown_writer, monkeypatch):
        assert_clean()
        monkeypatch.setattr(config, "INFERENCE_ENGINE", "patched")
        monkeypatch.setattr(state.host, "model_loaded", True)
        assert config.INFERENCE_ENGINE == "patched"
        assert state.host.model_loaded is True

    def test_08_observe_after_all_finalizers():
        assert_clean()
"""


@pytest.mark.parametrize(
    "case_names",
    [
        ("01_direct_host_changes", "02_observe_after_host_changes"),
        ("03_lazy_instance_and_monkeypatch", "04_observe_after_lazy_changes"),
        ("05_mutable_config_changes", "06_observe_after_config_changes", "08_observe_after_all_finalizers"),
    ],
    ids=["host", "lazy", "config"],
)
def test_state_isolation_in_one_process(pytester, case_names):
    _isolated_suite(pytester)
    pytester.makepyfile(test_sequence=_SEQUENCE)
    result = pytester.runpytest_subprocess(
        *(f"test_sequence.py::test_{name}" for name in case_names), timeout=60,
    )
    result.assert_outcomes(passed=len(case_names))


@pytest.mark.parametrize("distribution", ["load", "loadfile"])
def test_state_isolation_with_four_workers(pytester, distribution):
    _isolated_suite(pytester)
    for index in range(4):
        pytester.makepyfile(**{f"test_worker_{index}": _SEQUENCE})
    arguments = ["-n", "4"]
    if distribution != "load":
        arguments.append("--dist=" + distribution)
    result = pytester.runpytest_subprocess(*arguments, timeout=60)
    result.assert_outcomes(passed=32)
    assert len(list(pytester.path.glob("seen-gw*"))) == 4


@pytest.mark.parametrize("failure_stage", ["call", "setup"])
def test_state_isolation_after_failure(pytester, failure_stage):
    _isolated_suite(pytester)
    pytester.makepyfile(test_failure=f"""
        import pytest
        import config
        import state
        from state import assert_clean

        def pollute():
            state.host.model_loaded = True
            state.host._manager = None
            config.EXTERNAL_ENABLED = True
            raise RuntimeError("intentional isolation probe failure")

        @pytest.fixture
        def failing_setup():
            if {failure_stage!r} == "setup":
                pollute()

        def test_01_failure(failing_setup):
            pollute()

        def test_02_observer():
            assert_clean()
    """)
    result = pytester.runpytest_subprocess(timeout=60)
    if failure_stage == "call":
        result.assert_outcomes(passed=1, failed=1)
    else:
        result.assert_outcomes(passed=1, errors=1)


def test_state_isolation_preserves_scoped_fixture_overrides(pytester):
    _isolated_suite(pytester, audit_teardown=False)
    pytester.makepyfile(test_scoped="""
        from copy import deepcopy
        from types import SimpleNamespace
        import pytest
        import config
        import state

        @pytest.fixture(scope="module", autouse=True)
        def scoped_state():
            host = state.host
            old_instance = object.__getattribute__(state.manager, "_instance")
            scoped_manager = SimpleNamespace(is_loaded=True)
            object.__setattr__(state.manager, "_instance", scoped_manager)
            old_quant = config.QUANT_TYPE
            old_layers = deepcopy(config.DEFAULT_LAYER_CONFIG)
            old_loaded = host.model_loaded
            config.QUANT_TYPE = "scoped-quant"
            config.DEFAULT_LAYER_CONFIG["master"] = [2, 6]
            host.model_loaded = True
            yield
            assert config.QUANT_TYPE == "scoped-quant"
            assert config.DEFAULT_LAYER_CONFIG["master"] == [2, 6]
            assert host.model_loaded is True
            assert object.__getattribute__(state.manager, "_instance") is scoped_manager
            object.__setattr__(state.manager, "_instance", old_instance)
            config.QUANT_TYPE = old_quant
            config.DEFAULT_LAYER_CONFIG.clear()
            config.DEFAULT_LAYER_CONFIG.update(old_layers)
            host.model_loaded = old_loaded

        def test_01_scoped_baseline(monkeypatch):
            assert config.QUANT_TYPE == "scoped-quant"
            monkeypatch.setattr(config, "QUANT_TYPE", "patched-quant")
            config.DEFAULT_LAYER_CONFIG["master"].append(99)
            state.host.model_loaded = False
            object.__setattr__(state.manager, "_instance", SimpleNamespace(is_loaded=False))

        def test_02_scoped_baseline_restored():
            assert config.QUANT_TYPE == "scoped-quant"
            assert config.DEFAULT_LAYER_CONFIG["master"] == [2, 6]
            assert state.host.model_loaded is True
            assert state.host.peek_manager().is_loaded is True
    """)
    result = pytester.runpytest_subprocess(timeout=60)
    result.assert_outcomes(passed=2)


@pytest.mark.parametrize("workers", [0, 4])
def test_local_store_isolation_excludes_inherited_lifecycle(pytester, monkeypatch, workers):
    _isolated_suite(pytester)
    inherited = pytester.path / "outside-test.sqlite3"
    record = {"schema_version": 4, "model_id": "outside-test-model", "phase": "ready"}
    with sqlite3.connect(inherited) as connection:
        connection.execute(
            "CREATE TABLE cluster_settings (key TEXT PRIMARY KEY, value TEXT NOT NULL, updated_at TEXT)"
        )
        connection.execute(
            "INSERT INTO cluster_settings VALUES (?, ?, ?)",
            ("pipeline_config_lifecycle_v1", json.dumps(record), "test-only"),
        )
    before = inherited.read_bytes()
    monkeypatch.setenv("QLH_SQLITE_PATH", str(inherited))
    pytester.makepyfile(test_persistence=f"""
        import os
        from pathlib import Path
        import pytest
        import local_store

        @pytest.mark.parametrize("iteration", range(8))
        def test_local_state_starts_empty_and_remains_writable(iteration):
            # Every test and worker must ignore the inherited durable lifecycle.
            assert local_store.get_local_setting("pipeline_config_lifecycle_v1") is None
            path = Path(local_store._sqlite_path())
            assert path != Path({str(inherited)!r})
            legacy_dir = Path(local_store._legacy_store_dir())
            assert path.parent == legacy_dir.parent
            assert not legacy_dir.exists()
            local_store.set_local_setting("pipeline_config_lifecycle_v1", {{"test_iteration": iteration}})
            assert local_store.get_local_setting("pipeline_config_lifecycle_v1") == {{"test_iteration": iteration}}

        def test_explicit_test_store_override(monkeypatch, tmp_path):
            override = tmp_path / "explicit.sqlite3"
            monkeypatch.setenv("QLH_SQLITE_PATH", str(override))
            local_store.set_local_setting("pipeline_config_lifecycle_v1", {{"explicit": True}})
            assert Path(local_store._sqlite_path()) == override
            assert local_store.get_local_setting("pipeline_config_lifecycle_v1") == {{"explicit": True}}
    """)
    arguments = ["-n", str(workers)]
    result = pytester.runpytest_subprocess(*arguments, timeout=60)
    result.assert_outcomes(passed=9)
    assert inherited.read_bytes() == before


def test_real_scheduler_recovers_only_explicit_test_lifecycle():
    import local_store
    from scheduler import Scheduler

    assert local_store.get_local_setting("pipeline_config_lifecycle_v1") is None
    cold = Scheduler()
    cold._load_pipeline_recovery_state()
    assert cold._pipeline_recovery_pending is False
    record = {
        "schema_version": 4,
        "config_id": "isolation-config",
        "model_id": "isolation-model",
        "model_sha256": "a" * 64,
        "generation": 1,
        "phase": "ready",
        "cleanup_ledger": [],
    }
    local_store.set_local_setting("pipeline_config_lifecycle_v1", record)
    restarted = Scheduler()
    restarted._load_pipeline_recovery_state()
    assert restarted._pipeline_recovery_pending is True
    assert restarted._pipeline_recovery_failure == "pipeline_recovery_pending"
    assert restarted._pipeline_recovery_state == record
