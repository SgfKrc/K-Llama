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

import os
import random
import sys
from copy import deepcopy

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))


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
    return name.isupper() and not name.startswith("_")


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
            service.active_session_id = None
    ctx = getattr(api_server, "_request_id_ctx", None)
    if ctx is not None:
        ctx.set("")


@pytest.fixture(autouse=True)
def _isolate_local_store(tmp_path):
    """Keep lifecycle/identity state and legacy imports out of the user's store.

    Use the real SQLite implementation with a separate path for every test and
    worker. Test fixtures may still override the path or persistence functions
    to exercise recovery; do not suppress lifecycle reads or model restoration.
    """
    with pytest.MonkeyPatch.context() as isolation:
        isolation.setenv("QLH_SQLITE_PATH", str(tmp_path / "local-store.sqlite3"))
        import local_store

        isolation.setattr(local_store, "_legacy_store_dir", lambda: str(tmp_path / "legacy"))
        isolation.setattr(local_store, "_initialized_paths", set())
        yield


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

    # Inspect the instance dictionary and lazy slot directly. getattr/hasattr on
    # ModelHost or _LazyModelManager can otherwise construct a real ModelManager.
    host_module = sys.modules.get("model_host")
    host = vars(host_module).get("model_host") if host_module is not None else None
    host_originals = dict(vars(host)) if host is not None else {}
    generation = host_originals.get("generation_config")
    generation_snapshot = deepcopy(generation)
    manager = host_originals.get("_manager")
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
