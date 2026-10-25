"""Model-free tests for optional native capability, strict batches and isolated IPC."""
from __future__ import annotations

import base64
import ctypes
import importlib
import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import numpy as np
import pytest

from src import llama_keep_head as kh

ROOT = Path(__file__).resolve().parents[1]


class Function:
    def __init__(self, callback):
        self.callback = callback
        self.calls = []

    def __call__(self, *args):
        self.calls.append(args)
        return self.callback(*args)


class FakeLibrary(SimpleNamespace):
    """ctypes-shaped functions, used only in tests and disposable Python children."""

    def __init__(self, *, missing=(), guard=1, vocabulary=4):
        super().__init__()
        for name in kh.SHIM_SYMBOLS:
            setattr(self, name, Function(lambda *args: 0))
        self.qlh_kh_load = Function(self.load)
        self.qlh_kh_forward_embd_token = Function(self.scalar)
        self.qlh_kh_forward_embd_tokens = Function(self.multi)
        self.qlh_kh_n_vocab = Function(lambda handle: vocabulary)
        self.qlh_kh_finite_logits_guard = Function(lambda: guard)
        self.qlh_kh_last_error = Function(lambda handle: 7)
        self.output = None
        self.rc = 0
        for name in missing:
            delattr(self, name)

    @staticmethod
    def load(*args):
        ctypes.cast(args[7], ctypes.POINTER(ctypes.c_int32))[0] = 2
        ctypes.cast(args[8], ctypes.POINTER(ctypes.c_int32))[0] = 3
        return 2**48 + 123  # Exercise the real 64-bit handle declaration.

    @staticmethod
    def scalar(handle, hidden, count, past, nseq, seq, pos, output):
        ctypes.cast(output, ctypes.POINTER(ctypes.c_int32))[0] = 2
        return 0

    def multi(self, handle, hidden, count, past, nseq, seq, pos, output, capacity):
        assert capacity == count
        for i in range(count):
            output[i] = self.output[i] if self.output is not None else i % 4
        return self.rc


@pytest.fixture
def local_upstream(monkeypatch, tmp_path):
    monkeypatch.setattr(kh, "_llama_cpp_loaded", lambda: False)
    monkeypatch.setattr(kh, "_shim_abi_collides_with_llama_cpp", lambda path: False)
    monkeypatch.setattr(kh, "_add_dll_dirs", lambda *args, **kwargs: ([], []))
    shim = tmp_path / "shim.dll"
    model = tmp_path / "model.gguf"
    shim.write_bytes(b"fake native library")
    model.write_bytes(b"fake weights")
    owners = []

    def create(**kwargs):
        lib = FakeLibrary(**kwargs)
        monkeypatch.setattr(kh.ctypes, "CDLL", lambda path: lib)
        up = kh.KeepHeadUpstream(shim, model, n_ctx=256, n_seq_max=4, n_batch=8)
        owners.append(up)
        return up, lib

    yield create
    for owner in owners:
        owner.close()


def test_new_symbols_and_ctypes_signatures(local_upstream):
    up, lib = local_upstream()
    assert up.supports_multi_position_argmax is True and up.n_vocab == 4
    types = lib.qlh_kh_forward_embd_tokens.argtypes
    assert types == [ctypes.c_void_p, ctypes.POINTER(ctypes.c_float), ctypes.c_int32,
                     ctypes.c_int32, ctypes.POINTER(ctypes.c_int32), ctypes.POINTER(ctypes.c_int32),
                     ctypes.POINTER(ctypes.c_int32), ctypes.POINTER(ctypes.c_int32), ctypes.c_int32]
    assert lib.qlh_kh_forward_embd_tokens.restype is ctypes.c_int32
    assert lib.qlh_kh_finite_logits_guard.argtypes == []
    assert lib.qlh_kh_n_vocab.argtypes == [ctypes.c_void_p]
    assert lib.qlh_kh_n_vocab.calls == [(2**48 + 123,)]


@pytest.mark.parametrize("missing", ["qlh_kh_forward_embd_tokens", "qlh_kh_n_vocab",
                                    "qlh_kh_finite_logits_guard"])
def test_each_optional_symbol_is_required_but_scalar_still_works(local_upstream, missing):
    up, _ = local_upstream(missing=[missing])
    assert up.supports_multi_position_argmax is False
    with pytest.raises(kh.KeepHeadUnavailable, match="qlh_kh_forward_embd_tokens"):
        up.forward_hidden_to_tokens([[0, 0]])
    assert up.forward_hidden_to_token([[0, 0]]) == 2
    assert up.n_vocab == (0 if missing == "qlh_kh_n_vocab" else 4)


def test_old_shim_without_any_new_symbols_keeps_scalar(local_upstream):
    up, _ = local_upstream(missing=["qlh_kh_forward_embd_tokens", "qlh_kh_n_vocab",
                                   "qlh_kh_finite_logits_guard"])
    assert up.n_vocab == 0 and up.supports_multi_position_argmax is False
    assert up.forward_hidden_to_token([[float("nan"), 0]]) == 2  # Preserve the scalar input rule.
    with pytest.raises(kh.KeepHeadUnavailable, match="multi-position argmax unavailable"):
        up.forward_hidden_to_tokens([[0, 0]])


@pytest.mark.parametrize("kwargs", [{"guard": 0}, {"guard": -1}, {"guard": 2},
                                   {"vocabulary": 0}, {"vocabulary": -1}])
def test_capability_requires_finite_guard_and_loaded_vocabulary(local_upstream, kwargs):
    up, _ = local_upstream(**kwargs)
    assert up.supports_multi_position_argmax is False


@pytest.mark.parametrize("count", [1, 2, 4, 8])
def test_returns_exactly_one_token_per_row_with_nonzero_seq(local_upstream, count):
    up, lib = local_upstream()
    result = up.forward_hidden_to_tokens(np.zeros((count, 2)), n_past=np.int64(9),
                                        seq_ids=[np.int32(3)] * count,
                                        positions=list(range(9, 9 + count)))
    assert result == [i % 4 for i in range(count)]
    assert all(type(value) is int for value in result)
    args = lib.qlh_kh_forward_embd_tokens.calls[0]
    assert args[0] == 2**48 + 123 and args[2:5] == (count, 9, None)
    assert list(args[5]) == [3] * count
    assert list(args[6]) == list(range(9, 9 + count))
    assert args[8] == count


def test_defaults_single_row_and_noncontiguous_input(local_upstream):
    up, _ = local_upstream()
    assert up.forward_hidden_to_tokens([1, 2]) == [0]
    hidden = np.arange(12, dtype=np.float64).reshape(3, 4)[:, ::2]
    assert not hidden.flags.c_contiguous
    assert up.forward_hidden_to_tokens(hidden) == [0, 1, 2]


@pytest.mark.parametrize("hidden", [[], np.zeros((0, 2)), [[0]], np.zeros((1, 1, 2)),
                                    [[float("nan"), 0]], [[0, float("inf")]],
                                    [[float("-inf"), 0]], [[1e100, 0]]])
def test_invalid_hidden_never_calls_native(local_upstream, hidden):
    up, lib = local_upstream()
    with np.errstate(over="ignore"), pytest.raises(ValueError):
        up.forward_hidden_to_tokens(hidden)
    assert not lib.qlh_kh_forward_embd_tokens.calls


@pytest.mark.parametrize("kwargs", [
    {"seq_ids": [0]}, {"positions": [0]}, {"seq_ids": [0, 1]}, {"seq_ids": [-1, -1]},
    {"seq_ids": [4, 4]}, {"positions": [0, 0]}, {"positions": [0, 2]},
    {"positions": [1, 0]}, {"positions": [-1, 0]}, {"n_past": -1},
    {"n_past": 2**31}, {"n_past": -(2**31) - 1}, {"n_past": 2**31 - 1},
    {"seq_ids": [2**32, 2**32]}, {"positions": [0, 2**32 + 1]},
    {"n_past": float("nan")}, {"n_past": float("inf")}, {"n_past": 1.5}, {"n_past": True},
    {"seq_ids": [0.0, 0.0]}, {"seq_ids": [True, True]}, {"positions": [0, float("inf")]},
    {"positions": [0, float("nan")]}, {"positions": ["0", "1"]},
])
def test_invalid_metadata_never_calls_native(local_upstream, kwargs):
    up, lib = local_upstream()
    with pytest.raises(ValueError):
        up.forward_hidden_to_tokens([[0, 0]] * 2, **kwargs)
    assert not lib.qlh_kh_forward_embd_tokens.calls


def test_batch_context_and_closed_owner_fail_before_native(local_upstream):
    up, lib = local_upstream()
    with pytest.raises(kh.KeepHeadUnavailable, match="batch exceeds n_batch"):
        up.forward_hidden_to_tokens([[0, 0]] * 9)
    with pytest.raises(kh.KeepHeadUnavailable, match="context exceeds"):
        up.forward_hidden_to_tokens([[0, 0]] * 2, n_past=63)
    up.close()
    with pytest.raises(kh.KeepHeadUnavailable, match="unavailable"):
        up.forward_hidden_to_tokens([[0, 0]])
    assert not lib.qlh_kh_forward_embd_tokens.calls


@pytest.mark.parametrize("tokens", [[-1, 0], [0, 4], [2**31 - 1, 1]])
def test_native_token_range_is_validated(local_upstream, tokens):
    up, lib = local_upstream()
    lib.output = tokens
    with pytest.raises(kh.KeepHeadUnavailable, match="invalid multi-position tokens"):
        up.forward_hidden_to_tokens([[0, 0]] * 2)


@pytest.mark.parametrize("rc", [-2, -5, -6, -7])
def test_native_failure_never_returns_written_partial_output(local_upstream, rc):
    up, lib = local_upstream()
    lib.rc = rc
    with pytest.raises(kh.KeepHeadUnavailable, match=f"rc={rc}"):
        up.forward_hidden_to_tokens([[0, 0]] * 2)


@pytest.mark.parametrize("tokens", [None, [], [0], [0, 1, 2], [0, 4], [-1, 0],
                                    [True, 1], [0, 1.0], [0, "1"], [0, float("nan")]])
def test_parent_rejects_bad_ipc_length_types_or_vocabulary(local_upstream, monkeypatch, tokens):
    up, _ = local_upstream()
    up._worker = SimpleNamespace(poll=lambda: None)
    monkeypatch.setattr(up, "_worker_request", lambda payload: {"ok": True, "tokens": tokens})
    try:
        with pytest.raises(kh.KeepHeadUnavailable, match="invalid multi-position tokens"):
            up.forward_hidden_to_tokens([[0, 0]] * 2)
    finally:
        up._worker = None


def test_parent_serializes_hidden_and_validates_before_ipc(local_upstream, monkeypatch):
    up, _ = local_upstream()
    up._worker = SimpleNamespace(poll=lambda: None)
    requests = []
    monkeypatch.setattr(up, "_worker_request", lambda payload: requests.append(payload) or {"tokens": [1, 2]})
    try:
        assert up.forward_hidden_to_tokens([[3, 4], [5, 6]], n_past=7,
                                           seq_ids=[2, 2], positions=[7, 8]) == [1, 2]
        payload = requests[0]
        assert payload["op"] == "hidden_tokens" and payload["shape"] == [2, 2]
        assert payload["n_past"] == 7 and payload["seq_ids"] == [2, 2] and payload["positions"] == [7, 8]
        assert np.frombuffer(base64.b64decode(payload["data"]), dtype=np.float32).tolist() == [3, 4, 5, 6]
        with pytest.raises(ValueError):
            up.forward_hidden_to_tokens([[0, 0]] * 2, positions=[0, 3])
        assert len(requests) == 1
    finally:
        up._worker = None


def worker_module(monkeypatch):
    monkeypatch.syspath_prepend(str(ROOT / "src"))
    return importlib.import_module("llama_keep_head_worker")


@pytest.mark.parametrize("new", [True, False])
def test_worker_receipt_and_operation_with_actual_python_child(monkeypatch, tmp_path, new):
    """Use the real worker main + parent pipes; fake only the native library in the child."""
    shim = tmp_path / "shim.dll"
    model = tmp_path / "model.gguf"
    shim.write_bytes(b"fake")
    model.write_bytes(b"fake")
    real_popen = subprocess.Popen
    bootstrap = """
import sys
from pathlib import Path
root = Path(sys.argv.pop(1))
sys.path.insert(0, str(root / 'tests'))
sys.path.insert(0, str(root / 'src'))
from test_llama_keep_head_multi_position import FakeLibrary
import llama_keep_head as kh
missing = [] if sys.argv.pop(1) == 'new' else ['qlh_kh_forward_embd_tokens', 'qlh_kh_n_vocab', 'qlh_kh_finite_logits_guard']
lib = FakeLibrary(missing=missing)
kh.ctypes.CDLL = lambda path: lib
from llama_keep_head_worker import main
raise SystemExit(main())
"""

    def start(command, **kwargs):
        return real_popen([sys.executable, "-c", bootstrap, str(ROOT), "new" if new else "old",
                           *command[2:]], **kwargs)

    monkeypatch.setattr(kh.subprocess, "Popen", start)
    monkeypatch.setattr(kh, "_llama_cpp_loaded", lambda: True)
    environment = dict(os.environ, QLH_KEEP_HEAD_WORKER_LOG=str(tmp_path / "worker.err"))
    with kh.KeepHeadUpstream(shim, model, n_ctx=256, n_seq_max=4, n_batch=8,
                             environment=environment) as up:
        assert up._worker is not None and up._lib is None
        assert up.supports_multi_position_argmax is new
        assert up.n_vocab == (4 if new else 0)
        assert up.forward_hidden_to_token([[0, 0]]) == 2
        if new:
            assert up.forward_hidden_to_tokens([[0, 0]] * 4,
                                               seq_ids=[2] * 4, positions=[41, 42, 43, 44]) == [0, 1, 2, 3]
        else:
            with pytest.raises(kh.KeepHeadUnavailable, match="multi-position argmax unavailable"):
                up.forward_hidden_to_tokens([[0, 0]])
    assert up._worker is None


@pytest.mark.parametrize("receipt", [
    {"n_vocab": 4, "supports_multi_position_argmax": True},
    {},
    {"n_vocab": 0, "supports_multi_position_argmax": False},
])
def test_parent_initialization_receipt_defaults_and_capability(monkeypatch, tmp_path, receipt):
    up = object.__new__(kh.KeepHeadUpstream)
    up.shim_path = tmp_path / "shim.dll"
    up.model_path = tmp_path / "model.gguf"
    up.mode, up.cut_layer = "nextn", None
    up._environment = {"QLH_KEEP_HEAD_WORKER_LOG": str(tmp_path / "worker.err")}
    process = object()
    monkeypatch.setattr(kh.subprocess, "Popen", lambda *args, **kwargs: process)
    monkeypatch.setattr(up, "_read_worker_response", lambda: {"ok": True, "n_embd": 2, "n_layer": 3, **receipt})
    up._init_isolated_worker(n_ctx=64, n_threads=1, n_batch=8, n_seq_max=1, extra_dll_dirs=[])
    assert up.n_vocab == receipt.get("n_vocab", 0)
    assert up.supports_multi_position_argmax is receipt.get("supports_multi_position_argmax", False)
    up._worker = None


@pytest.mark.parametrize("receipt", [
    {"n_vocab": 0, "supports_multi_position_argmax": True},
    {"n_vocab": -1}, {"n_vocab": 2**31}, {"n_vocab": "4"}, {"n_vocab": True},
    {"n_vocab": 4, "supports_multi_position_argmax": 1},
])
def test_invalid_receipt_retires_worker(monkeypatch, tmp_path, receipt):
    up = object.__new__(kh.KeepHeadUpstream)
    up.shim_path = tmp_path / "shim.dll"
    up.model_path = tmp_path / "model.gguf"
    up.mode, up.cut_layer = "nextn", None
    up._environment = {"QLH_KEEP_HEAD_WORKER_LOG": str(tmp_path / "worker.err")}
    monkeypatch.setattr(kh.subprocess, "Popen", lambda *args, **kwargs: object())
    monkeypatch.setattr(up, "_read_worker_response", lambda: {"ok": True, "n_embd": 2, "n_layer": 3, **receipt})
    retired = []
    monkeypatch.setattr(up, "close", lambda: retired.append(True))
    with pytest.raises(kh.KeepHeadUnavailable, match="invalid capability receipt"):
        up._init_isolated_worker(n_ctx=64, n_threads=1, n_batch=8, n_seq_max=1, extra_dll_dirs=[])
    assert retired == [True] and up.is_available is False
    up._worker = None


@pytest.mark.parametrize("missing,guard,expected", [
    ([], 1, True), (["qlh_kh_forward_embd_tokens"], 1, False),
    (["qlh_kh_n_vocab"], 1, False), (["qlh_kh_finite_logits_guard"], 1, False),
    ([], 0, False), ([], 2, False),
])
def test_probe_worker_checks_symbols_without_loading_model(monkeypatch, tmp_path, capsys, missing, guard, expected):
    worker = worker_module(monkeypatch)
    lib = FakeLibrary(missing=missing, guard=guard)
    monkeypatch.setattr(worker, "_args", lambda: SimpleNamespace(
        probe_shim=True, shim=str(tmp_path / "fake.dll"), dll_dir=[]))
    monkeypatch.setattr(worker, "_add_dll_dirs", lambda *args: ([], []))
    monkeypatch.setattr(worker.ctypes, "CDLL", lambda path: lib)
    assert worker.main() == 0
    assert json.loads(capsys.readouterr().out) == {"ok": True, "supports_multi_position_argmax": expected}
    assert not lib.qlh_kh_load.calls
    if hasattr(lib, "qlh_kh_n_vocab"):
        assert not lib.qlh_kh_n_vocab.calls


@pytest.fixture
def probe_setup(monkeypatch, tmp_path):
    kh._keep_head_multi_position_probe_cached.cache_clear()
    shim = tmp_path / "shim.dll"
    shim.write_bytes(b"fake")
    # Any host DLL load would fail this test, including when probing succeeds.
    monkeypatch.setattr(kh.ctypes, "CDLL", lambda *args: pytest.fail("host must never load a DLL"))
    calls = []

    def run(command, **kwargs):
        calls.append((command, kwargs))
        return SimpleNamespace(returncode=0, stdout='{"ok":true,"supports_multi_position_argmax":true}')

    monkeypatch.setattr(kh.subprocess, "run", run)
    yield shim, calls
    kh._keep_head_multi_position_probe_cached.cache_clear()


def test_probe_is_isolated_model_free_and_cached_by_file_and_environment(probe_setup):
    shim, calls = probe_setup
    environment = {"PATH": "test-path", "QLH_KEEP_HEAD_DLL_DIRS": "extra"}
    assert kh.keep_head_multi_position_available(shim, environment=environment) is True
    assert kh.keep_head_multi_position_available(shim.parent / "." / shim.name, environment=environment) is True
    assert len(calls) == 1
    command, kwargs = calls[0]
    assert command[-3:] == ["--probe-shim", "--shim", str(shim.resolve())]
    assert "--model" not in command
    assert kwargs["timeout"] == 15 and kwargs["env"] == environment
    assert kwargs["stderr"] is subprocess.DEVNULL
    previous = shim.stat()
    os.utime(shim, ns=(previous.st_atime_ns, previous.st_mtime_ns + 100))
    assert kh.keep_head_multi_position_available(shim, environment=environment) is True
    assert len(calls) == 2
    shim.write_bytes(b"new bigger fake")
    assert kh.keep_head_multi_position_available(shim, environment=environment) is True
    assert len(calls) == 3
    assert kh.keep_head_multi_position_available(shim, environment={**environment, "OTHER": "1"}) is True
    assert len(calls) == 4


@pytest.mark.parametrize("stdout,returncode", [
    ('{"ok":true,"supports_multi_position_argmax":false}', 0),
    ('{"ok":true,"supports_multi_position_argmax":1}', 0),
    ('{"ok":false,"supports_multi_position_argmax":true}', 0),
    ('{"ok":true,"supports_multi_position_argmax":true}', 1),
    ('not json', 0), ('[]', 0), ('', 0),
])
def test_probe_rejects_missing_or_invalid_capability(probe_setup, monkeypatch, stdout, returncode):
    shim, _ = probe_setup
    monkeypatch.setattr(kh.subprocess, "run", lambda *args, **kwargs: SimpleNamespace(
        returncode=returncode, stdout=stdout))
    assert kh.keep_head_multi_position_available(shim) is False


@pytest.mark.parametrize("failure", [OSError("failed"), subprocess.TimeoutExpired("probe", 15)])
def test_probe_timeout_or_launch_failure_returns_false(probe_setup, monkeypatch, failure):
    shim, _ = probe_setup
    def fail(*args, **kwargs):
        raise failure
    monkeypatch.setattr(kh.subprocess, "run", fail)
    assert kh.keep_head_multi_position_available(shim) is False


def test_probe_missing_shim_does_not_launch(probe_setup):
    shim, calls = probe_setup
    assert kh.keep_head_multi_position_available(shim.parent / "missing.dll") is False
    assert kh.keep_head_multi_position_available(shim.parent) is False
    assert not calls


def test_probe_frozen_command_uses_worker_dispatch(probe_setup, monkeypatch):
    shim, calls = probe_setup
    monkeypatch.setattr(kh.sys, "frozen", True, raising=False)
    assert kh.keep_head_multi_position_available(shim) is True
    assert calls[0][0][:2] == [sys.executable, "--keephead-worker"]


@pytest.mark.parametrize("past", [0, -1, 2**31, 2**31 - 1, float("nan"), float("inf")])
def test_explicit_positions_override_past_for_validation_and_native_call(local_upstream, past):
    up, lib = local_upstream()
    assert up.forward_hidden_to_tokens([[0, 0]] * 3, n_past=past,
                                       positions=[41, 42, 43]) == [0, 1, 2]
    assert lib.qlh_kh_forward_embd_tokens.calls[0][3] == 41


def test_context_check_uses_explicit_position_base(local_upstream):
    up, lib = local_upstream()
    with pytest.raises(kh.KeepHeadUnavailable, match="context exceeds"):
        up.forward_hidden_to_tokens([[0, 0]] * 2, positions=[63, 64])
    assert not lib.qlh_kh_forward_embd_tokens.calls
    assert up.forward_hidden_to_tokens([[0, 0]] * 2, n_past=1000, positions=[62, 63]) == [0, 1]


def test_explicit_position_overflow_is_rejected_before_native(local_upstream):
    up, lib = local_upstream()
    with pytest.raises(ValueError, match="int32"):
        up.forward_hidden_to_tokens([[0, 0]] * 2, positions=[2**31 - 1, 0])
    assert not lib.qlh_kh_forward_embd_tokens.calls

@pytest.mark.parametrize("missing,guard,expected", [
    ([], 1, True), (["qlh_kh_forward_embd_tokens"], 1, False),
    (["qlh_kh_n_vocab"], 1, False), ([], 0, False),
])
def test_probe_uses_actual_python_child_without_host_native_load(monkeypatch, tmp_path, missing, guard, expected):
    shim = tmp_path / "fake.dll"
    shim.write_bytes(b"fake")
    kh._keep_head_multi_position_probe_cached.cache_clear()
    bootstrap = """
import json
import sys
from pathlib import Path
root = Path(sys.argv.pop(1))
sys.path.insert(0, str(root / 'tests'))
sys.path.insert(0, str(root / 'src'))
from test_llama_keep_head_multi_position import FakeLibrary
import llama_keep_head as kh
configuration = json.loads(sys.argv.pop(1))
lib = FakeLibrary(**configuration)
def forbidden_load(*args):
    raise AssertionError('ABI probe must not load a model')
lib.qlh_kh_load.callback = forbidden_load
kh.ctypes.CDLL = lambda path: lib
from llama_keep_head_worker import main
raise SystemExit(main())
"""
    command = [sys.executable, "-c", bootstrap, str(ROOT), json.dumps({"missing": missing, "guard": guard})]
    monkeypatch.setattr(kh, "_keep_head_worker_command", lambda: command)
    monkeypatch.setattr(kh.ctypes, "CDLL", lambda *args: pytest.fail("host must never load a DLL"))
    try:
        assert kh.keep_head_multi_position_available(shim) is expected
    finally:
        kh._keep_head_multi_position_probe_cached.cache_clear()
