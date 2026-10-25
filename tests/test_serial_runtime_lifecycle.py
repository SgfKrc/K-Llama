"""Deterministic serial regressions for runtime ownership and test cleanup."""
from pathlib import Path

import pytest

pytest_plugins = ["pytester"]
_REPO = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("predecessor_fails", [False, True])
def test_runtime_and_tcp_owners_are_drained_between_tests(pytester, monkeypatch, predecessor_fails):
    monkeypatch.setenv("PYTHONUTF8", "1")
    monkeypatch.setenv("PYTHONIOENCODING", "utf-8")
    # Run the actual repository conftest in a separate interpreter. Keep the
    # runtime, encrypted store and retry owner alive across two child tests so
    # the second can audit teardown, including teardown after an assertion.
    source = (_REPO / "tests/conftest.py").read_text(encoding="utf-8")
    source = source.replace(
        'sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))',
        f"sys.path.insert(0, {str(_REPO / 'src')!r})",
    )
    pytester.makeconftest(source)
    pytester.makeini("[pytest]\naddopts = -q -n 0\n")
    pytester.makepyfile(test_owners=f'''
import os
from pathlib import Path
import threading

import api_server
import local_store
import tcp_comm

seen = {{}}
initial_constructor = tcp_comm.TCPClient.__init__


def test_01_leave_constructed_owners(monkeypatch, tmp_path):
    runtime = api_server._get_app_runtime()
    services = runtime.services
    ledger = api_server._get_join_ledger()
    ledger._private_store.set("test-pending", "predecessor")
    seen.update(runtime=runtime, services=services, ledger=ledger,
                private_bytes=ledger._private_store.path.read_bytes(),
                expected_root=tmp_path)
    # Match hello-owner-worker: a synthetic registered client retires its recv
    # loop, handing off to a real delayed retry. It must be stopped before any
    # function monkeypatch disappears or the next test changes runtime paths.
    # Constructor patches must share the ordinary monkeypatch undo order;
    # they must not leave the previous test's tracking closure installed.
    initialize = tcp_comm.TCPClient.__init__
    def wrapped_initialize(client, *args, **kwargs):
        initialize(client, *args, **kwargs)
    monkeypatch.setattr(tcp_comm.TCPClient, "__init__", wrapped_initialize)
    client = tcp_comm.TCPClient(client_id="serial-leaked-owner")
    seen["client"] = client
    monkeypatch.setattr(tcp_comm, "RECONNECT_DELAY", 60)
    client._schedule_reconnect_retry()
    seen["retry"] = next(t for t in threading.enumerate()
                         if t.name == "tcp-reconnect-serial-leaked-owner")
    assert client._reconnect_retry_scheduled
    assert Path(services.store.path).parent == tmp_path
    assert not {predecessor_fails!r}, "deliberate predecessor failure"


def test_02_audit_teardown_and_current_paths(tmp_path):
    import inspect
    tracker = inspect.getclosurevars(tcp_comm.TCPClient.__init__).nonlocals
    assert tracker["initialize"] is initial_constructor
    previous = seen["runtime"]
    assert previous._closed, "previous test retained its AppRuntime"
    assert seen["services"]._closed, "previous runtime services were not closed"
    assert not seen["client"]._reconnect_enabled
    assert not seen["client"]._reconnect_retry_scheduled
    assert not seen["retry"].is_alive(), "retry thread escaped its owning test"
    runtime = api_server._get_app_runtime()
    assert runtime is not previous
    services = runtime.services
    assert Path(services.store.path).parent == tmp_path
    with services.store.scope():
        assert Path(local_store._sqlite_path()).parent == tmp_path
    ledger = api_server._get_join_ledger()
    assert ledger is not seen["ledger"]
    assert Path(ledger.path).parent == tmp_path
    assert ledger._private_store.get("test-pending") == ""
    assert seen["ledger"]._private_store.path.read_bytes() == seen["private_bytes"]
''')
    result = pytester.runpytest_subprocess("--tb=short", timeout=90)
    result.assert_outcomes(passed=1 if predecessor_fails else 2,
                           failed=1 if predecessor_fails else 0)



def test_nested_api_mocks_do_not_escape_function_fixture_teardown(pytester, monkeypatch):
    """A local mock context must outlive the test's ordinary monkeypatch undo."""
    monkeypatch.setenv("PYTHONUTF8", "1")
    monkeypatch.setenv("PYTHONIOENCODING", "utf-8")
    source = (_REPO / "tests/conftest.py").read_text(encoding="utf-8")
    source = source.replace(
        'sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))',
        f"sys.path.insert(0, {str(_REPO / 'src')!r})",
    )
    pytester.makeconftest(source)
    pytester.makeini("[pytest]\naddopts = -q -n 0\n")
    pytester.makepyfile(test_api_aliases='''
import pytest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import api_server

original = {name: vars(api_server)[name]
            for name in ("scheduler", "model_host", "model_manager")}


@pytest.fixture
def scoped_api_mocks():
    with patch.multiple(api_server, **{name: MagicMock() for name in original}):
        yield


def test_01_nested_replacements(scoped_api_mocks, monkeypatch):
    # Logging tests combine a fixture's mock context with function replacements.
    # Undoing the function ledger after that context re-publishes its old mocks.
    for name in original:
        monkeypatch.setattr(api_server, name, SimpleNamespace())
    monkeypatch.setattr(api_server, "_runtime_readiness", {"ready": True})


def test_02_real_runtime_and_readiness_are_current():
    for name, value in original.items():
        assert vars(api_server)[name] is value, f"API alias escaped teardown: {name}"
    runtime = api_server._get_app_runtime()
    assert not runtime._closed and runtime.config is None
    assert api_server._runtime_readiness is runtime._readiness
    assert api_server._runtime_readiness_lock is runtime._readiness_lock
    assert api_server._runtime_startup_done is runtime.startup_done
    assert runtime.readiness()["ready"] is False
    assert not runtime.startup_done.is_set()
    assert api_server.scheduler._worker_prepare._runtime_factory is api_server._get_layer_stage_host
''')
    result = pytester.runpytest_subprocess("--tb=short", timeout=90)
    result.assert_outcomes(passed=2)
