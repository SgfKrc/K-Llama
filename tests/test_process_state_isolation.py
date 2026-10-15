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


@pytest.mark.parametrize("workers", [0, 4])
def test_backend_thread_after_fixture_teardown_uses_session_store(
    pytester, monkeypatch, workers,
):
    """Release a real supervisor thread only after all function fixtures unwind."""
    monkeypatch.setenv("PYTHONUTF8", "1")
    monkeypatch.setenv("PYTHONIOENCODING", "utf-8")
    _isolated_suite(pytester, audit_teardown=False)
    pytester.makepyfile(
        tui_backend=(_REPO / "src" / "tui_backend.py").read_text(encoding="utf-8"),
    )
    inherited = pytester.path / "inherited-user.sqlite3"
    user_state = pytester.path / "user-state"
    user_state.mkdir()
    default_store = user_state / "qlh-control.sqlite3"
    identity = {
        "version": 1, "mac_addresses": ["aabbccddeeff"], "bound_at": "user-owned",
    }
    for store in (inherited, default_store):
        with sqlite3.connect(store) as connection:
            connection.execute(
                "CREATE TABLE cluster_settings (key TEXT PRIMARY KEY, value TEXT NOT NULL, updated_at TEXT)"
            )
            connection.execute(
                "INSERT INTO cluster_settings VALUES (?, ?, ?)",
                ("master_identity_v1", json.dumps(identity), "user-owned"),
            )
    before = {store: store.read_bytes() for store in (inherited, default_store)}
    monkeypatch.setenv("QLH_SQLITE_PATH", str(inherited))
    config_file = pytester.path / "config.py"
    config_file.write_text(
        config_file.read_text(encoding="utf-8") + f"\nSTATE_DIR = {str(user_state)!r}\n",
        encoding="utf-8",
    )
    pytester.makepyfile(late_backend="""
        import os
        import sys
        import threading
        from pathlib import Path
        from types import SimpleNamespace
        import local_store
        from tui_backend import BackendSupervisor

        release = threading.Event()
        entered = threading.Event()
        errors = []
        connections = []
        supervisors = []
        function_path = None
        session_path = None

        def audit_connection(event, args):
            if event == "sqlite3.connect" and threading.current_thread().name == "qlh-tui-backend":
                path = Path(args[0]).resolve()
                connections.append(path)
                if path != session_path:
                    # Catch a regression before it can open ANY durable store.
                    raise AssertionError("late backend SQLite escaped session isolation: " + str(path))

        sys.addaudithook(audit_connection)

        class FakeServer:
            def __init__(self, config):
                self.should_exit = False

            def run(self):
                entered.set()
                try:
                    assert release.wait(timeout=10), "teardown did not release backend"
                    local_store.set_local_master_identity(["001122334455"])
                except BaseException as exc:
                    errors.append(exc)

        def start(monkeypatch):
            global function_path, session_path
            # Read the session fallback from the actual copied conftest.
            import conftest
            function_path = Path(local_store._sqlite_path()).resolve()
            session_path = Path(conftest._session_store_dir, "local-store.sqlite3").resolve()
            assert function_path != session_path
            release.clear()
            entered.clear()
            errors.clear()
            connections.clear()
            monkeypatch.setitem(sys.modules, "uvicorn", SimpleNamespace(
                Config=lambda *args, **kwargs: None, Server=FakeServer,
            ))
            import api_server
            monkeypatch.setattr(api_server, "app", object(), raising=False)
            supervisor = BackendSupervisor()
            probes = iter([False, True])
            monkeypatch.setattr(supervisor, "probe", lambda **kwargs: next(probes))
            monkeypatch.setattr(supervisor, "_silence_console_logging", lambda: None)
            supervisors.append(supervisor)
            assert supervisor.ensure_ready() is True
            assert entered.wait(timeout=5)
            assert supervisor.thread.is_alive()
            assert not connections

        def finish_after_teardown():
            if not supervisors:
                return
            supervisor = supervisors.pop()
            try:
                # At this point monkeypatch and _isolate_local_store are gone.
                assert Path(os.environ["QLH_SQLITE_PATH"]).resolve() == session_path
                assert Path(local_store._sqlite_path()).resolve() == session_path
            finally:
                release.set()
                supervisor.thread.join(timeout=5)
                supervisor.stop()
            assert not supervisor.thread.is_alive(), "backend leaked past teardown audit"
            assert not errors, errors
            assert connections and set(connections) == {session_path}
            assert local_store.get_local_master_identity()["mac_addresses"] == ["001122334455"]
            assert not function_path.exists(), "late write landed in the finished test's store"
    """)
    conftest_file = pytester.path / "conftest.py"
    conftest_file.write_text(
        conftest_file.read_text(encoding="utf-8") + """

@pytest.hookimpl(wrapper=True, tryfirst=True)
def pytest_runtest_teardown(item, nextitem):
    yield
    from late_backend import finish_after_teardown
    finish_after_teardown()
""",
        encoding="utf-8",
    )
    for index in range(4):
        pytester.makepyfile(**{f"test_late_backend_{index}": """
            import pytest
            from late_backend import start

            @pytest.mark.parametrize("iteration", range(2))
            def test_delayed_backend_identity_write(monkeypatch, iteration):
                start(monkeypatch)
        """})
    result = pytester.runpytest_subprocess(
        "-n", str(workers), "--dist=loadfile", timeout=60,
    )
    result.assert_outcomes(passed=8)
    for store, original_bytes in before.items():
        assert store.read_bytes() == original_bytes
        with sqlite3.connect(store) as connection:
            value = connection.execute(
                "SELECT value FROM cluster_settings WHERE key = 'master_identity_v1'",
            ).fetchone()[0]
        assert json.loads(value) == identity



def _node_isolated_suite(pytester):
    """Real bootstrap/credentials/journal, with lightweight runtime consumers."""
    _isolated_suite(pytester, audit_teardown=False)
    for name in ("node_config", "node_runtime", "local_secret_store", "network_address",
                 "release_contract", "task_journal"):
        pytester.makepyfile(**{name: (_REPO / "src" / f"{name}.py").read_text(encoding="utf-8")})
    pytester.makefile(".json", **{
        "release-contract": (_REPO / "release-contract.json").read_text(encoding="utf-8"),
    })
    config_file = pytester.path / "config.py"
    config_file.write_text(config_file.read_text(encoding="utf-8") + """
import os
NODE_ID = "baseline-node"
NODE_ROLE = "master"
CLUSTER_SECRET = ""
CLUSTER_SECRET_EPOCH = 1
CLIENT_MASTER_HOST = ""
CLIENT_MASTER_PORT = 8888
SERVER_PORT = 8888
HEARTBEAT_INTERVAL = 10
_NODE_CONFIG = {"node": {"role": "master"}, "cluster": {"epochs": [1]}}
STATE_DIR = os.environ["QLH_STATE_DIR"]
TASK_GRAPH_JOURNAL_PATH = os.path.join(STATE_DIR, "task-graph.sqlite3")
JOURNAL_REPLICATION_REPLICA_PATH = os.environ["QLH_JOURNAL_REPLICATION_REPLICA_PATH"]
KV_COLD_CACHE_ROOT = os.environ["QLH_KV_COLD_CACHE_ROOT"]
""", encoding="utf-8")
    pytester.makepyfile(
        scheduler="from config import NODE_ID, NODE_ROLE, SERVER_PORT",
        tcp_comm="from config import SERVER_PORT, HEARTBEAT_INTERVAL",
        runtime_probe="""
            import os
            from copy import deepcopy
            import api_server
            import config
            import node_config
            import node_runtime
            import scheduler
            import tcp_comm

            runtime = node_runtime.node_runtime
            cache = config._NODE_CONFIG
            cache_before = deepcopy(cache)
            aliases = [(module, name, getattr(module, name)) for module, names in (
                (scheduler, ("NODE_ID", "NODE_ROLE", "SERVER_PORT")),
                (api_server, ("NODE_ID", "NODE_ROLE", "SERVER_PORT")),
                (tcp_comm, ("SERVER_PORT", "HEARTBEAT_INTERVAL")),
            ) for name in names]
            environment = {k: v for k, v in os.environ.items() if k.startswith("QLH_")}

            def assert_runtime_clean():
                assert config.NODE_ROLE == "master"
                assert config.CLUSTER_SECRET == ""
                assert config.CLUSTER_SECRET_EPOCH == 1
                assert config._NODE_CONFIG is cache
                assert cache == cache_before
                assert node_runtime.node_runtime is runtime
                assert runtime.get_node_id() == "baseline-node"
                assert runtime.get_node_role() == "master"
                for module, name, value in aliases:
                    assert getattr(module, name) == value, (module.__name__, name)
                assert "isolation-probe" not in node_config._ROLE_ALIASES

            def assert_environment_clean():
                assert {k: v for k, v in os.environ.items() if k.startswith("QLH_")} == environment
        """,
    )
    api_file = pytester.path / "api_server.py"
    api_file.write_text(api_file.read_text(encoding="utf-8") + """
from config import NODE_ID, NODE_ROLE, SERVER_PORT, TASK_GRAPH_JOURNAL_PATH
""", encoding="utf-8")
    return pytester.path / "conftest.py"


@pytest.mark.parametrize("workers", [0, 4])
def test_bootstrap_restores_runtime_aliases_environment_and_config_cache(pytester, workers):
    conftest_file = _node_isolated_suite(pytester)
    conftest_file.write_text(conftest_file.read_text(encoding="utf-8") + """

@pytest.hookimpl(wrapper=True, tryfirst=True)
def pytest_runtest_teardown(item, nextitem):
    yield
    from runtime_probe import assert_runtime_clean, assert_environment_clean
    assert_runtime_clean()
    assert_environment_clean()
""", encoding="utf-8")
    sequence = """
        import os
        import pytest
        import api_server
        import config
        import node_config
        import node_runtime
        import scheduler
        import tcp_comm
        from runtime_probe import assert_runtime_clean

        @pytest.fixture
        def teardown_bootstrap():
            yield
            # A function fixture finalizer can perform a final bootstrap sync.
            node_config.apply_runtime_config({"node": {"node_id": "late-client", "role": "client"}})

        def test_01_bootstrap_without_monkeypatch(teardown_bootstrap):
            assert_runtime_clean()
            node_config.apply_runtime_config({
                "node": {"node_id": "joined-client", "role": "client"},
                "cluster": {"cluster_secret": "probe-secret", "cluster_secret_epoch": 3},
            })
            # Real runtime sync deliberately leaves the static role unchanged.
            assert config.NODE_ROLE == "master"
            assert scheduler.NODE_ROLE == "client"
            assert node_runtime.node_runtime.get_node_role() == "client"
            api_server.NODE_ROLE = "client"
            api_server.NODE_ID = "polluted-api"
            scheduler.SERVER_PORT = 1234
            tcp_comm.SERVER_PORT = 1234
            config._NODE_CONFIG["cluster"]["epochs"].append(99)
            node_config._ROLE_ALIASES["isolation-probe"] = "client"
            os.environ["QLH_ISOLATION_TEMP_PROJECTION"] = "leaked"
            os.environ.pop("QLH_MASTER_API_PORT", None)

        def test_02_observer_and_replaced_cache():
            assert_runtime_clean()
            config._NODE_CONFIG = {"replacement": True}
            from node_runtime import NodeRuntime
            node_runtime.node_runtime = NodeRuntime()
            node_runtime.node_runtime.set_node_id("replacement")

        def test_03_observe_restored_cache():
            assert_runtime_clean()
    """
    for index in range(4):
        pytester.makepyfile(**{f"test_runtime_{index}": sequence})
    result = pytester.runpytest_subprocess("-n", str(workers), "--dist=loadfile", timeout=90)
    result.assert_outcomes(passed=12)


@pytest.mark.parametrize("workers", [0, 4])
def test_node_config_credentials_and_journal_exclude_shared_runner_state(
    pytester, monkeypatch, workers,
):
    _node_isolated_suite(pytester)
    shared = pytester.path / "inherited-runner-state"
    shared.mkdir()
    files = {name: shared / name for name in (
        "node_config.json", "node_secrets.json", "node_secrets.json.key",
        "task-graph.sqlite3", "journal-replica.sqlite3",
    )}
    for name, path in files.items():
        path.write_bytes(("user-owned-" + name).encode())
    before = {path: path.read_bytes() for path in files.values()}
    monkeypatch.setenv("QLH_STATE_DIR", str(shared))
    monkeypatch.setenv("QLH_NODE_CONFIG_PATH", str(files["node_config.json"]))
    monkeypatch.setenv("QLH_NODE_SECRET_STORE_PATH", str(files["node_secrets.json"]))
    monkeypatch.setenv("QLH_JOURNAL_REPLICATION_REPLICA_PATH", str(files["journal-replica.sqlite3"]))
    pytester.makepyfile(test_persistence=f"""
        import os
        from pathlib import Path
        import pytest
        import api_server
        import config
        import node_config
        from task_journal import SQLiteTaskJournal

        @pytest.mark.parametrize("iteration", range(12))
        def test_bootstrap_and_journal_are_per_test(iteration, tmp_path):
            assert node_config.load_node_config() == {{}}
            assert not node_config.get_node_secret_store_path().exists()
            assert Path(config.STATE_DIR) == tmp_path / "state"
            assert api_server.TASK_GRAPH_JOURNAL_PATH == config.TASK_GRAPH_JOURNAL_PATH
            assert not Path(config.TASK_GRAPH_JOURNAL_PATH).exists()
            response = {{
                "node": {{"node_id": "node-" + str(iteration), "role": "client"}},
                "cluster": {{"cluster_secret": "secret-" + str(iteration)}},
            }}
            path = node_config.persist_bootstrap_response(response)
            assert path == tmp_path / "node_config.json"
            assert node_config.get_local_cluster_secret() == "secret-" + str(iteration)
            assert node_config.load_node_config()["node"]["node_id"] == "node-" + str(iteration)
            assert node_config.get_node_secret_store_path().with_suffix(".json.key").is_file()
            for path in (config.TASK_GRAPH_JOURNAL_PATH, config.JOURNAL_REPLICATION_REPLICA_PATH):
                assert not Path(path).is_relative_to(Path({str(shared)!r}))
                journal = SQLiteTaskJournal(path)
                journal.close()

        def test_platform_fallback_and_legacy_root_are_safe(monkeypatch, tmp_path):
            # Deleting overrides must still use a private home/LocalAppData.
            monkeypatch.delenv("QLH_NODE_CONFIG_PATH")
            monkeypatch.delenv("QLH_NODE_SECRET_STORE_PATH")
            assert node_config.get_node_config_path().is_relative_to(tmp_path)
            assert node_config.get_node_secret_store_path().is_relative_to(tmp_path)
            assert node_config.load_node_config() == {{}}
    """)
    result = pytester.runpytest_subprocess("-n", str(workers), timeout=90)
    result.assert_outcomes(passed=13)
    assert {path: path.read_bytes() for path in before} == before


@pytest.mark.parametrize("workers", [0, 4])
def test_late_bootstrap_and_journal_thread_uses_session_fallback(pytester, workers):
    conftest_file = _node_isolated_suite(pytester)
    pytester.makepyfile(late_persistence="""
        import os
        import threading
        from pathlib import Path
        import api_server
        import config
        import node_config
        from task_journal import SQLiteTaskJournal

        pending = []

        def start(tmp_path):
            release = threading.Event()
            errors = []
            function_root = tmp_path.resolve()

            def write():
                try:
                    assert release.wait(10)
                    import conftest
                    session = Path(conftest._session_store_dir).resolve()
                    for path in (node_config.get_node_config_path(),
                                 node_config.get_node_secret_store_path(),
                                 Path(config.STATE_DIR), Path(config.TASK_GRAPH_JOURNAL_PATH),
                                 Path(api_server.TASK_GRAPH_JOURNAL_PATH)):
                        assert path.resolve().is_relative_to(session), str(path)
                        assert not path.resolve().is_relative_to(function_root)
                    node_config.persist_bootstrap_response({
                        "node": {"node_id": "late-session-client", "role": "client"},
                        "cluster": {"cluster_secret": "late-session-secret"},
                    })
                    journal = SQLiteTaskJournal(api_server.TASK_GRAPH_JOURNAL_PATH)
                    journal.close()
                except BaseException as exc:
                    errors.append(exc)

            thread = threading.Thread(target=write, daemon=True)
            thread.start()
            pending.append((thread, release, errors))

        def finish():
            if not pending:
                return
            thread, release, errors = pending.pop()
            environment = {k: v for k, v in os.environ.items() if k.startswith("QLH_")}
            release.set()
            thread.join(10)
            try:
                assert not thread.is_alive()
                assert not errors, errors
                assert node_config.get_local_cluster_secret() == "late-session-secret"
            finally:
                # This probe deliberately writes after the fixture boundary.
                # Retire its session data and direct env projection ourselves.
                for path in (node_config.get_node_config_path(), node_config.get_node_secret_store_path(),
                             node_config.get_node_secret_store_path().with_suffix(".json.key")):
                    path.unlink(missing_ok=True)
                for name in list(os.environ):
                    if name.startswith("QLH_") and name not in environment:
                        del os.environ[name]
                os.environ.update(environment)
    """)
    conftest_file.write_text(conftest_file.read_text(encoding="utf-8") + """

@pytest.hookimpl(wrapper=True, tryfirst=True)
def pytest_runtest_teardown(item, nextitem):
    yield
    from late_persistence import finish
    finish()
""", encoding="utf-8")
    for index in range(4):
        pytester.makepyfile(**{f"test_late_persistence_{index}": """
            import pytest
            from late_persistence import start

            @pytest.mark.parametrize("iteration", range(2))
            def test_late_thread(tmp_path, iteration):
                start(tmp_path)
        """})
    result = pytester.runpytest_subprocess("-n", str(workers), "--dist=loadfile", timeout=90)
    result.assert_outcomes(passed=8)


def test_scheduler_runtime_role_keeps_client_management_denied(monkeypatch):
    """Exercise the production guard even if static Config still says master."""
    import config
    import scheduler

    monkeypatch.setattr(config, "NODE_ROLE", "master")
    monkeypatch.setattr(scheduler, "NODE_ROLE", "client")
    instance = scheduler.Scheduler()
    assert instance._effective_role() == "client"
    assert instance.manual_register_node("test-only-edge")["status"] == "denied"
    assert instance.delete_node("test-only-edge")["status"] == "denied"


_seen_default_journals = []


@pytest.mark.parametrize("iteration", range(2))
def test_default_task_graph_journal_has_a_test_lifetime(monkeypatch, tmp_path, iteration):
    import api_server
    import config

    for previous in _seen_default_journals:
        assert previous._closed is True
    coordinator = api_server.task_graph_coordinator
    assert type(coordinator) is api_server._LazyTaskGraphCoordinator
    assert coordinator._is_materialized() is False
    monkeypatch.setattr(api_server, "TASK_GRAPH_ENABLED", True)
    status = coordinator.journal_status()
    assert status["available"] is True
    assert status["backend"] == "sqlite"
    assert Path(status["path"]) == Path(config.TASK_GRAPH_JOURNAL_PATH)
    assert Path(status["path"]).is_relative_to(tmp_path)
    _seen_default_journals.append(vars(coordinator)["_coordinator"]._journal)



def test_runtime_alias_isolation_preserves_module_fixture_baselines(pytester):
    _node_isolated_suite(pytester)
    pytester.makepyfile(test_scoped_runtime="""
        import pytest
        import config
        import node_runtime
        import scheduler
        import api_server
        import tcp_comm

        @pytest.fixture(scope="module", autouse=True)
        def scoped_runtime():
            with pytest.MonkeyPatch.context() as patch:
                patch.setattr(scheduler, "NODE_ROLE", "client")
                patch.setattr(api_server, "NODE_ROLE", "auto")
                patch.setattr(tcp_comm, "SERVER_PORT", 9999)
                patch.setattr(config, "_NODE_CONFIG", {"scoped": [1]})
                patch.setattr(node_runtime.node_runtime, "_node_id", "scoped-node")
                patch.setattr(node_runtime.node_runtime, "_node_role", "client")
                yield
                assert_scoped()

        def assert_scoped():
            assert config.NODE_ROLE == "master"
            assert scheduler.NODE_ROLE == "client"
            assert api_server.NODE_ROLE == "auto"
            assert tcp_comm.SERVER_PORT == 9999
            assert config._NODE_CONFIG == {"scoped": [1]}
            assert node_runtime.node_runtime.get_node_id() == "scoped-node"
            assert node_runtime.node_runtime.get_node_role() == "client"

        def test_01_scoped_mutation():
            assert_scoped()
            scheduler.NODE_ROLE = "master"
            api_server.NODE_ROLE = "client"
            tcp_comm.SERVER_PORT = 1234
            config._NODE_CONFIG["scoped"].append(2)
            node_runtime.node_runtime.set_node_id("function-node")
            node_runtime.node_runtime.set_node_role("master")

        def test_02_scoped_observer():
            assert_scoped()
    """)
    result = pytester.runpytest_subprocess("-n", "0", timeout=60)
    result.assert_outcomes(passed=2)


def test_bootstrap_runtime_isolation_restores_after_failure(pytester):
    _node_isolated_suite(pytester)
    pytester.makepyfile(test_failed_bootstrap="""
        import node_config
        import config
        from runtime_probe import assert_runtime_clean

        def test_01_bootstrap_then_failure():
            node_config.apply_runtime_config({"node": {"node_id": "failed-join", "role": "client"}})
            config._NODE_CONFIG["cluster"]["epochs"].append(99)
            raise RuntimeError("intentional bootstrap isolation failure")

        def test_02_after_failure():
            assert_runtime_clean()
    """)
    result = pytester.runpytest_subprocess("-n", "0", timeout=60)
    result.assert_outcomes(failed=1, passed=1)
