"""tests/conftest.py — 测试会话级准备

Install the compatibility shim early and isolate process-wide mutable state.

为什么需要在这里（而不是只靠 `src/model_module.py`）：
`transformers_stream_generator`（Qwen-1.8B remote code 的 `chat_stream` 依赖）在
transformers 5.x 下无法导入，而 transformers 5.x 的 `dynamic_module_utils.check_imports`
会在加载**任何** remote code 时把它列为待导入依赖，**非 "No module named" 的 ImportError 会直接抛**。

`src/model_module.py` 里已经装了一次（生产入口够用），但测试里有些用例**不经过**
`model_module` 就自己去 `from_pretrained(...)`（例如直接构造 remote code 模型的用例），
在那种执行顺序下 shim 尚未生效 ⇒ 会偶发 `ImportError: cannot import name
'DisjunctiveConstraint'`。会话最初装一次即可消除这种顺序依赖。

注意：**不要把它放到 `src/config.py`** —— 入口模块会导入 config，而
`test_entry_module_no_heavy_imports` 明确要求入口顶层不得拉入 transformers。
"""
from __future__ import annotations

import atexit
import os
import random
import shutil
import sys
import tempfile
from copy import deepcopy
from pathlib import Path

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

# Workspace-only suites: independent qlh-docagent, retired MF-MEM benchmarks,
# and external offline-asset release tooling (aa889bcb / module ownership docs).
# Their sources are intentionally absent from the main repo. Keep ownership
# stable even on developer checkouts; opt-in imports must still fail normally.
# Core EX-N3 quality infrastructure and dependency contracts remain collected.
collect_ignore_glob = (
    [
        "test_docagent_*.py", "test_doc_maintenance_*.py",
        "test_model_fleet_explicit_layer_pager.py",
        "test_model_fleet_memory_transport.py",
        "test_model_fleet_n2f_fixed_tile.py",
        "test_model_fleet_n2r_artifact_gate.py",
        "test_model_fleet_n2r_compare.py",
        "test_offline_bundle.py",
    ]
    if os.environ.get("QLH_INCLUDE_WORKSPACE_TOOL_TESTS") != "1"
    else []
)

# Install a distinct fallback in every worker before collection imports config.
# Function fixtures restore these paths; a late backend thread can still persist
# safely after its test ends. Never inherit the runner's shared node config,
# credential store or journal (nor a real user's platform-default directories).
_session_store_dir = tempfile.mkdtemp(prefix="qlh-pytest-store-")


def _persistence_environment(root):
    root = Path(root)
    return {
        "QLH_SQLITE_PATH": str(root / "local-store.sqlite3"),
        "QLH_STATE_DIR": str(root / "state"),
        "QLH_NODE_CONFIG_PATH": str(root / "node_config.json"),
        "QLH_NODE_SECRET_STORE_PATH": str(root / "node_secrets.json"),
        "QLH_KV_COLD_CACHE_ROOT": str(root / "state" / "kv-cold"),
        "QLH_JOURNAL_REPLICATION_REPLICA_PATH": str(root / "journal-replica.sqlite3"),
        "LOCALAPPDATA": str(root / "user-local"),
        "APPDATA": str(root / "user-roaming"),
        "USERPROFILE": str(root / "user-home"),
        "HOME": str(root / "user-home"),
        "XDG_CONFIG_HOME": str(root / "user-config"),
        "XDG_STATE_HOME": str(root / "user-state"),
    }


os.environ.update(_persistence_environment(_session_store_dir))
atexit.register(shutil.rmtree, _session_store_dir, True)

# Legacy discovery can migrate/delete checkout/node_config.json even when the
# primary path is isolated. Keep the real implementation for temporary frozen
# installations used by bootstrap migration tests; redirect user/install roots.
try:
    import node_config as _session_node_config

    _node_app_root = _session_node_config.get_app_root

    def _test_node_app_root():
        root = _node_app_root().resolve()
        if root.is_relative_to(Path(tempfile.gettempdir()).resolve()):
            return root
        return Path(os.environ.get("QLH_STATE_DIR", _session_store_dir)) / "legacy"

    _session_node_config.get_app_root = _test_node_app_root
except ImportError:
    pass  # Dependency-light pytester suites may omit node_config entirely.

# config paths are evaluated on import, and api_server opens logging at
# collection time. Install the safe log path before any test imports that API.
try:
    import config as _session_config

    _session_config.LOG_DIR = str(Path(_session_store_dir) / "logs")
except ImportError:
    pass


def pytest_addoption(parser) -> None:
    parser.addoption(
        "--qlh-order-seed",
        action="store",
        type=int,
        default=None,
        help="deterministically shuffle collected tests with this seed",
    )


def pytest_collection_modifyitems(config, items) -> None:
    seed = config.getoption("--qlh-order-seed")
    if seed is not None:
        random.Random(seed).shuffle(items)


def pytest_report_header(config) -> str:
    seed = config.getoption("--qlh-order-seed")
    return f"qlh test order seed: {seed}" if seed is not None else ""

try:  # 尽量早地装上；失败也不应让整个测试会话崩掉
    from transformers5_compat import install as _install_transformers5_compat

    _install_transformers5_compat()
except Exception:  # noqa: BLE001 - 环境缺 transformers 时静默跳过（相关用例会自行跳过）
    pass


def pytest_runtest_setup(item) -> None:  # noqa: ARG001 - 需要 item 签名
    """每个用例开始前，确保 transformers 的 5.x 兼容符号在位。

    为什么需要：有测试会**动全局的 transformers** —— 例如
    `tests/test_island_engine.py` 为在无 transformers 的容器里导入 `model_module`
    而把 `sys.modules["transformers"]` 换成桩；也可能有用例删/替换模块属性。
    这类改动若未彻底复原，会让后续用例（如
    `test_inference_service_protocol::test_build_app_client_role_gates_chat`，它经
    `inference_svc_main → model_module` 真的去加载 remote code）看到
    `cannot import name 'DisjunctiveConstraint'` —— 看起来像 transformers 版本问题，
    实为**测试间污染**。

    ⚠️ 判定条件必须基于「**shim 符号是否在位**」，而不是「模块是否存在/是不是桩」：
    实测被污染时 `transformers` 仍是真模块（有 `__version__`），只是少了那几个符号。

    正常情况下这是廉价 no-op。
    """
    import sys as _sys

    module = _sys.modules.get("transformers")
    if module is not None and hasattr(module, "DisjunctiveConstraint"):
        return  # 符号在位 ⇒ 无需干预
    # 情况一：模块被换成桩 ⇒ 清掉相关缓存，重新导入真模块
    if module is not None and not hasattr(module, "__version__"):
        _sys.modules.pop("transformers", None)
        _sys.modules.pop("model_module", None)
    try:
        from transformers5_compat import install as _install

        _install()  # 情况二：真模块但符号被删 ⇒ 直接补回
    except Exception:  # noqa: BLE001 - 兜底失败不应中断用例
        pass


def _is_config_setting(name: str) -> bool:
    return (name.isupper() and not name.startswith("_")) or name == "_NODE_CONFIG"


def _restore_mutable_value(original, snapshot):
    """Restore containers in place so aliases keep seeing the restored baseline."""
    if isinstance(original, dict):
        original.clear()
        original.update(snapshot)
    elif isinstance(original, (list, bytearray)):
        original[:] = snapshot
    elif isinstance(original, set):
        original.clear()
        original.update(snapshot)
    else:
        return snapshot
    return original


def _clear_chat_state(api_server) -> None:
    if api_server is None:
        return
    service = getattr(api_server, "_chat_context", None)
    if service is not None:
        with service._lock:
            service._committed_operations.clear()
            service._revisions.clear()
            service.histories.clear()
            service._history_generations.clear()
            service._active_by_client.clear()
            service._kv_session_id = None
    ctx = getattr(api_server, "_request_id_ctx", None)
    if ctx is not None:
        ctx.set("")


@pytest.fixture(autouse=True)
def _isolate_local_store(tmp_path):
    """Isolate every durable state path, retaining real persistence semantics."""
    # Bootstrap writes os.environ directly, outside the monkeypatch ledger.
    # Capture the scoped baseline before our overrides and restore after undo.
    environment = {k: v for k, v in os.environ.items() if k.startswith("QLH_")}
    try:
        with pytest.MonkeyPatch.context() as isolation:
            for name, value in _persistence_environment(tmp_path).items():
                isolation.setenv(name, value)
            import local_store
            import config

            isolation.setattr(local_store, "_legacy_store_dir", lambda: str(tmp_path / "legacy"))
            isolation.setattr(local_store, "_initialized_paths", set())
            paths = {
                "STATE_DIR": str(tmp_path / "state"),
                "LEGACY_RECOVERY_JOURNAL_PATH": str(tmp_path / "state" / "task-graph.sqlite3"),
                "JOURNAL_REPLICATION_REPLICA_PATH": str(tmp_path / "journal-replica.sqlite3"),
                "KV_COLD_CACHE_ROOT": str(tmp_path / "state" / "kv-cold"),
                "LOG_DIR": str(tmp_path / "logs"),
            }
            for name, value in paths.items():
                if hasattr(config, name):
                    isolation.setattr(config, name, value)
                # Imported path aliases must follow the test, too. Do not
                # synthesize settings in lightweight child-suite stubs.
                for module_name in ("scheduler", "api_server", "tcp_comm"):
                    module = sys.modules.get(module_name)
                    if module is not None and name in vars(module):
                        isolation.setattr(module, name, value)
            yield
    finally:
        for name in list(os.environ):
            if name.startswith("QLH_") and name not in environment:
                del os.environ[name]
        os.environ.update(environment)


@pytest.fixture(autouse=True)
def _reset_process_wide_chat_state(_isolate_local_store):
    """Restore each test's Config and shared ModelHost after normal fixture teardown.

    Auto-load and scheduler paths write host state directly; distributed tests
    also replace the shared manager and can leave it as None.

    Keep snapshots local to the test: a session baseline both loses scoped
    fixture overrides and, when reused by reference, can itself be mutated.
    Do not depend on monkeypatch or call undo(): its normal teardown must run
    first, especially for attributes forwarded through the lazy manager.
    Managers, callbacks and locks retain their identity; no engine is loaded,
    copied or unloaded by isolation.
    """
    try:
        import api_server
    except Exception:  # noqa: BLE001 - optional runtime dependencies may be absent
        api_server = None
    try:
        import config
    except ImportError:
        config = None

    config_originals = (
        {name: value for name, value in vars(config).items() if _is_config_setting(name)}
        if config is not None else {}
    )
    config_snapshot = deepcopy(config_originals)

    # Each module owns its runtime alias. Restoring Config alone leaves
    # scheduler._effective_role() on a bootstrap-written client role. Snapshot
    # aliases independently, preserving module-scoped fixture overrides.
    aliases = {}
    for module_name in ("scheduler", "api_server", "tcp_comm", "node_config"):
        module = sys.modules.get(module_name)
        if module is None:
            continue
        originals = {
            name: value for name, value in vars(module).items()
            if name in config_originals or (module_name == "node_config" and name.isupper())
        }
        aliases[module_name] = (module, originals, deepcopy(originals))

    runtime_module = sys.modules.get("node_runtime")
    runtime = vars(runtime_module).get("node_runtime") if runtime_module is not None else None
    runtime_snapshot = (
        (runtime.get_node_id(), runtime.get_node_role()) if runtime is not None else None
    )

    # Inspect the instance dictionary and lazy slot directly. getattr/hasattr on
    # ModelHost or _LazyModelManager can otherwise construct a real ModelManager.
    host_module = sys.modules.get("model_host")
    host = vars(host_module).get("model_host") if host_module is not None else None
    host_originals = dict(vars(host)) if host is not None else {}
    generation = host_originals.get("generation_config")
    generation_snapshot = deepcopy(generation)
    controller = host_originals.get("_runtime_controller")
    controller_originals = dict(vars(controller)) if controller is not None else None
    if controller_originals is not None:
        controller_originals["_metadata"] = deepcopy(controller_originals["_metadata"])
    manager = controller.manager() if controller is not None else host_originals.get("_manager")
    missing = object()
    try:
        lazy_instance = object.__getattribute__(manager, "_instance")
    except AttributeError:
        lazy_instance = missing

    _clear_chat_state(api_server)
    try:
        yield
    finally:
        # This autouse fixture is set up before function fixtures, so their
        # finalizers (including monkeypatch.undo) have already completed here.
        try:
            _clear_chat_state(api_server)
        finally:
            if host is not None:
                if controller is not None:
                    vars(controller).clear()
                    vars(controller).update(controller_originals)
                if lazy_instance is not missing:
                    object.__setattr__(manager, "_instance", lazy_instance)
                if generation is not None:
                    _restore_mutable_value(generation, deepcopy(generation_snapshot))
                namespace = vars(host)
                namespace.clear()
                namespace.update(host_originals)
            if config is not None:
                namespace = vars(config)
                for name in list(namespace):
                    if _is_config_setting(name) and name not in config_originals:
                        del namespace[name]
                restored = deepcopy(config_snapshot)
                for name, original in config_originals.items():
                    namespace[name] = _restore_mutable_value(original, restored[name])

            for module_name in ("scheduler", "api_server", "tcp_comm", "node_config"):
                module = sys.modules.get(module_name)
                if module is None:
                    continue
                baseline = aliases.get(module_name)
                if baseline is None:
                    # A first import during the test may have consumed polluted
                    # Config; restore only its configuration aliases.
                    originals = {name: value for name, value in config_originals.items()
                                 if name in vars(module)}
                    snapshot = deepcopy(config_snapshot)
                else:
                    original_module, originals, snapshot = baseline
                    sys.modules[module_name] = original_module
                    module = original_module
                restored = deepcopy(snapshot)
                for name, original in originals.items():
                    vars(module)[name] = _restore_mutable_value(original, restored[name])
            current_runtime_module = sys.modules.get("node_runtime")
            if current_runtime_module is not None:
                if runtime is not None:
                    current_runtime_module.node_runtime = runtime
                    runtime.set_node_id(runtime_snapshot[0])
                    runtime.set_node_role(runtime_snapshot[1])
                else:
                    current_runtime = vars(current_runtime_module).get("node_runtime")
                    if current_runtime is not None:
                        current_runtime.set_node_id(str(config_originals.get("NODE_ID", "master")))
                        current_runtime.set_node_role(str(config_originals.get("NODE_ROLE", "client")))


@pytest.fixture(autouse=True)
def _isolate_api_runtime(_reset_process_wide_chat_state):
    """Give each test an unstarted runtime and close its real resource owners.

    Config snapshots cannot restore AppRuntime: its immutable config, SQLite
    binding and identity store retain the paths from their first resolution.
    Cleanup runs after function monkeypatch teardown and before ModelHost and
    Config restoration. Dependency-light child suites may supply an API stub.
    """
    api = sys.modules.get("api_server")
    if api is None or "_app_runtime" not in vars(api):
        yield
        return
    from qlh_core.bootstrap.runtime import AppRuntime

    runtimes = []
    resolve = api._get_app_runtime

    def resolve_owned_runtime():
        runtime = resolve()
        if all(runtime is not owned for owned in runtimes):
            runtimes.append(runtime)
        return runtime

    runtime = AppRuntime(
        host=api.model_host,
        resources_factory=api._build_default_resources,
        config_observer=lambda config: api._initialize_legacy_runtime_settings(),
    )
    runtimes.append(runtime)
    with pytest.MonkeyPatch.context() as isolation:
        slots = {
            "_app_runtime": runtime,
            "_get_app_runtime": resolve_owned_runtime,
            "_runtime_readiness_lock": runtime._readiness_lock,
            "_runtime_readiness": runtime._readiness,
            "_runtime_startup_done": runtime.startup_done,
            "_runtime_lifecycle_lock": runtime._lock,
            "_join_ledger_instance": None,
        }
        for name, value in slots.items():
            isolation.setattr(api, name, value)
        try:
            yield
        finally:
            for owned in reversed(runtimes):
                owned.close()
                # A test may replace lifespan shutdown with a spy and still
                # mark the runtime closed. Retained services must drain too.
                services = owned._services
                if services is not None and not services._closed:
                    services.close()


@pytest.fixture(autouse=True)
def _close_test_tcp_clients(_isolate_api_runtime):
    """Cancel and join clients created by this test before patches disappear.

    Synthetic hello-owner tests run production recv-loop finalizers, which can
    schedule a real retry even though no public connect() was used. Stop those
    precise instances rather than letting retry threads consume later globals.
    Keep the constructor patch outside function monkeypatch teardown. Requesting
    that shared fixture here would undo its replacements after local mock
    contexts exit, publishing their old API mocks into the next test.
    """
    module = sys.modules.get("tcp_comm")
    if module is None or "TCPClient" not in vars(module):
        yield
        return
    import threading
    import time

    client_type = module.TCPClient
    initialize = client_type.__init__
    disconnect = client_type.disconnect
    clients = []

    def track(client, *args, **kwargs):
        initialize(client, *args, **kwargs)
        clients.append(client)

    def owned_thread(thread):
        target = getattr(thread, "_target", None)
        owner = getattr(target, "__self__", None)
        if any(owner is client for client in clients):
            return True
        # TCP's delayed retry is a closure rather than a bound method.
        if getattr(target, "__name__", "") == "_retry":
            return any(any(cell.cell_contents is client for client in clients)
                       for cell in (target.__closure__ or ()))
        return False

    # Function patches undo first, including wrappers around this tracker;
    # this independent ledger then restores the original constructor last.
    constructor_patch = pytest.MonkeyPatch()
    constructor_patch.setattr(client_type, "__init__", track)
    try:
        yield
    finally:
        constructor_patch.undo()
        for client in reversed(clients):
            client.on_disconnect = None
            # Some protocol tests use a marker or a send-only socket
            # double. It owns no descriptor and has no close operation.
            with client._disconnect_callback_lock:
                if client.sock is not None and not callable(getattr(client.sock, "close", None)):
                    client.sock = None
            disconnect(client)
        deadline = time.monotonic() + 5
        threads = [thread for thread in threading.enumerate()
                   if thread is not threading.current_thread() and owned_thread(thread)]
        for thread in threads:
            thread.join(max(0, deadline - time.monotonic()))
        remaining = [thread.name for thread in threads if thread.is_alive()]
        assert not remaining, f"TCP workers escaped test teardown: {remaining}"
