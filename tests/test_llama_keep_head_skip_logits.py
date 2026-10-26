"""PERF-SKIP-LOGITS-01：`QLH_LAYER_SKIP_LOGITS` 的守卫用例。

背景：`qlh_keep_head.c` 过去对每个 token 都标 `batch.logits[i] = 1`，而"只要 hidden 的
入口"（`forward_tokens_to_hidden` / `forward_hidden_to_hidden`）**从不读 logits** ⇒
llama.cpp 每步仍要跑 `output.weight` 的矩阵乘。9B 中间段工件上这是每步几百 MB 的权重读取。

这里固定三条**不可退让**的性质：

1. 缺省（不设 `QLH_LAYER_SKIP_LOGITS`）行为与改动前一致 —— 不悄悄改语义；
2. 开了开关但 shim 没有 `qlh_kh_set_skip_hidden_logits` ⇒ **fail-loud**，且发生在
   加载模型**之前**（别为一次注定失败的初始化白读几 GB 权重）；
3. worker 隔离模式下，"已生效"必须由**子进程回执**确认，不能凭主进程设了环境变量就算数。

真模型上的数值门（hidden 逐位一致 + 末段 argmax 不变）由
`local_docs/perf-skiplogits-20261026/ab_keep_head_logits.py` 承担（要 3.9 GB 工件，不进单测）。
"""

from __future__ import annotations

import sys
import types
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import src.llama_keep_head as native  # noqa: E402
from src.llama_keep_head import (  # noqa: E402
    SHIM_SYMBOLS,
    SKIP_HIDDEN_LOGITS_ENV,
    KeepHeadUnavailable,
    KeepHeadUpstream,
    skip_hidden_logits_requested,
)


# ------------------------------------------------------------------ 纯逻辑
def test_flag_is_off_by_default_and_parses_the_usual_truthy_spellings():
    assert skip_hidden_logits_requested({}) is False
    assert skip_hidden_logits_requested() is False, "进程环境缺省必须是关闭"
    for value in ("1", "true", "TRUE", "yes", "on", " On "):
        assert skip_hidden_logits_requested({SKIP_HIDDEN_LOGITS_ENV: value}) is True
    for value in ("", "0", "false", "no", "off", "2", "maybe"):
        assert skip_hidden_logits_requested({SKIP_HIDDEN_LOGITS_ENV: value}) is False


def test_flag_is_not_part_of_the_mandatory_symbol_contract():
    """开关是**可选能力**：旧 shim 不该因为 SHIM_SYMBOLS 多一项而整个不可用。"""
    assert "qlh_kh_set_skip_hidden_logits" not in SHIM_SYMBOLS


# --------------------------------------------------- 缺符号必须 fail-loud
def _fake_inputs(tmp_path: Path) -> tuple[Path, Path]:
    shim = tmp_path / "qlh_keep_head.dll"
    shim.write_bytes(b"")               # 只要求存在：ctypes.CDLL 被替身接管
    model = tmp_path / "layer.gguf"
    model.write_bytes(b"")
    return shim, model


def _stub_library(*extra: str) -> types.SimpleNamespace:
    """最小假 DLL：必备符号齐全，可选符号按 `extra` 给。"""
    lib = types.SimpleNamespace()
    for name in (*SHIM_SYMBOLS, *extra):
        setattr(lib, name, lambda *args, **kwargs: 0)
    return lib


def _no_isolation(monkeypatch) -> None:
    """强制就地加载，避免替身测试真的去 spawn worker 子进程。"""
    monkeypatch.setattr(native, "_llama_cpp_loaded", lambda: False)
    monkeypatch.setattr(native, "_shim_abi_collides_with_llama_cpp", lambda shim_dir: False)


def test_missing_skip_symbol_fails_loud_before_reading_any_model(tmp_path, monkeypatch):
    shim, model = _fake_inputs(tmp_path)
    _no_isolation(monkeypatch)
    loaded: list = []
    lib = _stub_library()
    lib.qlh_kh_load = lambda *args: loaded.append(args) or 0x1
    monkeypatch.setattr(native.ctypes, "CDLL", lambda path: lib)
    monkeypatch.setenv(SKIP_HIDDEN_LOGITS_ENV, "1")

    with pytest.raises(KeepHeadUnavailable, match="qlh_kh_set_skip_hidden_logits"):
        KeepHeadUpstream(shim, model)
    assert loaded == [], "fail-fast：符号检查必须在 qlh_kh_load 之前"


def test_skip_symbol_is_switched_on_with_one(tmp_path, monkeypatch):
    shim, model = _fake_inputs(tmp_path)
    _no_isolation(monkeypatch)
    calls: list[tuple[int, int]] = []
    lib = _stub_library("qlh_kh_set_skip_hidden_logits")

    def _set(handle, value):
        calls.append((handle, value))
        return 0

    def _load(*args):
        args[7]._obj.value = 8      # out_n_embd
        args[8]._obj.value = 2      # out_n_layer
        return 0x1234

    lib.qlh_kh_set_skip_hidden_logits = _set
    lib.qlh_kh_load = _load
    monkeypatch.setattr(native.ctypes, "CDLL", lambda path: lib)
    monkeypatch.setenv(SKIP_HIDDEN_LOGITS_ENV, "1")

    upstream = KeepHeadUpstream(shim, model)
    assert calls == [(0x1234, 1)]
    assert upstream.skip_hidden_logits_applied is True
    upstream._handle = None          # 句柄是替身，别让 close() 去调 native


def test_flag_off_never_touches_the_optional_symbol(tmp_path, monkeypatch):
    """缺省路径**不得**调用新符号 —— 这是"缺省保持既有行为"的机械证据。"""
    shim, model = _fake_inputs(tmp_path)
    _no_isolation(monkeypatch)
    calls: list[tuple[int, int]] = []
    lib = _stub_library("qlh_kh_set_skip_hidden_logits")

    def _set(handle, value):       # pragma: no cover - 关闭时不该被调用
        calls.append((handle, value))
        return 0

    def _load(*args):
        args[7]._obj.value = 8
        args[8]._obj.value = 2
        return 0x1234

    lib.qlh_kh_set_skip_hidden_logits = _set
    lib.qlh_kh_load = _load
    monkeypatch.setattr(native.ctypes, "CDLL", lambda path: lib)
    monkeypatch.delenv(SKIP_HIDDEN_LOGITS_ENV, raising=False)

    upstream = KeepHeadUpstream(shim, model)
    assert calls == []
    assert upstream.skip_hidden_logits_applied is False
    upstream._handle = None


# --------------------------------------------------- worker 回执
def _bare_worker_upstream(tmp_path: Path, receipt: dict) -> KeepHeadUpstream:
    upstream = KeepHeadUpstream.__new__(KeepHeadUpstream)
    upstream.shim_path = tmp_path / "shim.dll"
    upstream.model_path = tmp_path / "model.gguf"
    upstream.mode, upstream.cut_layer = "nextn", None
    upstream._environment = {"QLH_KEEP_HEAD_WORKER_LOG": str(tmp_path / "worker.log")}
    upstream._skip_hidden_logits_requested = True
    upstream._skip_hidden_logits_applied = False
    upstream._read_worker_response = lambda: receipt
    upstream._worker = None
    return upstream


class _Process:
    """Popen 替身：`close()` 会正规地退掉 worker，所以这里要提供它调用的那几个方法。"""

    def __init__(self, command, **kwargs):
        self.command = command
        self.stdin = None
        self.stdout = None

    def poll(self):
        return None

    def wait(self, timeout=None):
        return 0


@pytest.fixture
def _stub_popen(monkeypatch):
    monkeypatch.setattr(native.subprocess, "Popen", _Process)


def test_worker_receipt_must_confirm_the_switch(tmp_path, _stub_popen):
    upstream = _bare_worker_upstream(
        tmp_path, {"ok": True, "n_embd": 8, "n_layer": 2, "skip_hidden_logits": False})

    with pytest.raises(KeepHeadUnavailable, match=SKIP_HIDDEN_LOGITS_ENV):
        upstream._init_isolated_worker(
            n_ctx=32, n_threads=1, n_batch=8, n_seq_max=1, extra_dll_dirs=())


def test_worker_receipt_without_the_field_stays_backward_compatible(tmp_path, _stub_popen):
    """旧 worker 回执没有这个字段：**没请求**跳过时不该因此失败。"""
    upstream = _bare_worker_upstream(
        tmp_path, {"ok": True, "n_embd": 8, "n_layer": 2})
    upstream._skip_hidden_logits_requested = False

    upstream._init_isolated_worker(
        n_ctx=32, n_threads=1, n_batch=8, n_seq_max=1, extra_dll_dirs=())

    assert upstream._skip_hidden_logits_applied is False
    assert upstream.skip_hidden_logits_applied is False


def test_worker_receipt_confirmation_is_recorded(tmp_path, _stub_popen):
    upstream = _bare_worker_upstream(
        tmp_path, {"ok": True, "n_embd": 8, "n_layer": 2, "skip_hidden_logits": True})

    upstream._init_isolated_worker(
        n_ctx=32, n_threads=1, n_batch=8, n_seq_max=1, extra_dll_dirs=())

    assert upstream.skip_hidden_logits_applied is True
