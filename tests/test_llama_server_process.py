"""llama-server 子进程生命周期契约（票 15b `LLAMA-SERVER-LIFECYCLE-01`）。

依据裁决（2026-10-12）：**有 CUDA 走 server 子进程，无 CUDA 老实走绑定**。
本文件只测进程生命周期与参数组装，用自包含假 server
（`tests/fixtures/fake_llama_server.py`）——不依赖真模型。

验收判据（对应票面）：
* 连续多轮「启动-就绪-请求-回收」后无残留进程、无端口泄漏；
* 进程被强杀后 `stop()` 不抛异常，下一次启动仍能就绪；
* 就绪判据是 HTTP `/health`，**不是** TCP connect（TCP 通 != 模型加载完）。
"""

from __future__ import annotations

import json
import pathlib
import socket
import sys
import time
import urllib.request

import pytest

from llama_server_process import LlamaServerProcess, ServerSpec

FIXTURE = str(pathlib.Path(__file__).parent / "fixtures" / "fake_llama_server.py")


def _spec(**overrides) -> ServerSpec:
    """默认指向假 server：`python <fixture> <server 参数...>`。"""
    base = dict(model_path="dummy.gguf", exe=sys.executable, argv_prefix=[FIXTURE])
    base.update(overrides)
    return ServerSpec(**base)


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def test_argv_assembly_includes_recommended_flags():
    """实测的推荐组合必须能组装出来：--cpu-moe + --spec-type + chat-template。"""
    spec = _spec(
        n_ctx=2048, n_gpu_layers=99, cpu_moe=True, spec_type="ngram-mod",
        chat_template_kwargs={"enable_thinking": False},
    )
    argv = spec.to_argv(host="127.0.0.1", port=12345)
    assert argv[:2] == [sys.executable, FIXTURE], "argv_prefix 必须紧跟解释器"
    assert argv[argv.index("-m") + 1] == "dummy.gguf"
    assert argv[argv.index("-c") + 1] == "2048"
    assert argv[argv.index("-ngl") + 1] == "99"
    assert "--cpu-moe" in argv
    assert argv[argv.index("--spec-type") + 1] == "ngram-mod"
    assert json.loads(argv[argv.index("--chat-template-kwargs") + 1]) == {
        "enable_thinking": False,
    }
    assert argv[argv.index("--port") + 1] == "12345"


def test_argv_omits_optional_flags_when_unset():
    argv = _spec().to_argv(host="127.0.0.1", port=1)
    assert "--cpu-moe" not in argv
    assert "--spec-type" not in argv
    assert "--chat-template-kwargs" not in argv


def test_missing_exe_raises_before_spawning():
    spec = _spec(exe=str(pathlib.Path("does-not-exist-anywhere.exe")))
    server = LlamaServerProcess(spec, port=_free_port(), startup_timeout=5)
    with pytest.raises(RuntimeError):
        server.start()
    assert server.is_alive is False


def test_start_ready_stop_three_rounds_without_leaks(tmp_path):
    """连续 3 轮：无残留进程、端口可复用（无泄漏）。"""
    port = _free_port()
    for round_index in range(3):
        server = LlamaServerProcess(
            _spec(), port=port, log_path=tmp_path / f"round{round_index}.log",
            startup_timeout=30,
        )
        with server:
            assert server.is_alive is True
            with urllib.request.urlopen(f"{server.base_url}/health", timeout=5) as r:
                assert json.loads(r.read())["status"] == "ok"
        assert server.is_alive is False
        assert server.proc is None, "stop() 必须清空句柄"


def test_ready_probe_is_health_not_tcp_connect():
    """假 server 延迟 3 秒才监听 => 期间绝不能提前判定就绪。"""
    port = _free_port()
    spec = _spec(model_path="dummy.gguf", extra_args=["--startup-delay", "3"])
    started = time.time()
    server = LlamaServerProcess(spec, port=port, startup_timeout=40)
    try:
        server.start()
        assert time.time() - started >= 2.5, "延迟启动期间不得提前判定就绪"
        assert server.is_alive is True
    finally:
        server.stop()


def test_stop_after_external_kill_is_safe_and_restartable():
    """外部强杀后 stop() 不抛，且能再次启动（票面验收）。"""
    port = _free_port()
    first = LlamaServerProcess(_spec(), port=port, startup_timeout=30)
    first.start()
    assert first.is_alive is True
    first.proc.kill()  # 模拟外部 kill -9
    time.sleep(0.5)
    first.stop()  # 不得抛异常

    second = LlamaServerProcess(_spec(), port=port, startup_timeout=30)
    try:
        second.start()
        assert second.is_alive is True
    finally:
        second.stop()


def test_start_timeout_stops_process_and_raises():
    """永不监听 => 超时后必须回收进程并具名报错。"""
    spec = _spec(exe=sys.executable, argv_prefix=["-c", "import time; time.sleep(60)"])
    server = LlamaServerProcess(spec, port=_free_port(), startup_timeout=3)
    with pytest.raises(RuntimeError):
        server.start()
    assert server.is_alive is False


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(pytest.main([__file__, "-q"]))
