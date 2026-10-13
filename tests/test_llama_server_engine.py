"""本机 llama-server 引擎契约（票 15b 接线 + 15c server 半边）。

`LlamaServerEngine` 把两件事组合起来：
* `LlamaServerProcess`（15b：进程生命周期）
* `IslandEngine`（既有的 OpenAI 兼容 HTTP 客户端 —— 票面要求的"复用"，不另写传输层）

它**对外仍报 `engine_type = "llama_cpp"`**：server 本来就是 llama.cpp 家族，
这样不必动 `BackendId` 枚举与所有 `backend_id_for` 分支；差异通过
`inference_path` / `capability()` 单独暴露（正合票 15a 的"能力声明"）。

测试全部走自包含假 server（`tests/fixtures/fake_llama_server.py`），不依赖真模型。
"""

from __future__ import annotations

import pathlib
import sys

import pytest

from llama_server_engine import LlamaServerEngine

FIXTURE = str(pathlib.Path(__file__).parent / "fixtures" / "fake_llama_server.py")


def _engine() -> LlamaServerEngine:
    """指向假 server 的引擎：`python <fixture> <server 参数...>`。"""
    return LlamaServerEngine(exe=sys.executable, argv_prefix=[FIXTURE])


def test_engine_reports_llama_cpp_family_and_server_path():
    engine = _engine()
    assert engine.engine_type == "llama_cpp", "对外仍是 llama.cpp 家族，避免动 BackendId"
    assert engine.inference_path == "llama_server"


def test_load_starts_process_and_chat_delegates():
    engine = _engine()
    engine.load_model(model_path="dummy.gguf", n_ctx=512, startup_timeout=30)
    try:
        assert engine.is_loaded is True
        result = engine.chat(
            messages=[{"role": "user", "content": "ping"}], max_tokens=8,
        )
        assert "echo:ping" in str(result.get("content", ""))
    finally:
        engine.unload()
    assert engine.is_loaded is False


def test_unload_stops_the_child_process():
    engine = _engine()
    engine.load_model(model_path="dummy.gguf", startup_timeout=30)
    proc = engine.server.proc if engine.server else None
    assert proc is not None and proc.poll() is None
    engine.unload()
    assert engine.server is None
    assert proc.poll() is not None, "unload 必须回收子进程"


def test_recommended_spec_reaches_argv():
    """实测推荐组合（cpu-moe + spec-type + thinking off）必须真的传给 server。"""
    engine = _engine()
    engine.load_model(
        model_path="dummy.gguf", n_ctx=2048, n_gpu_layers=99,
        cpu_moe=True, spec_type="ngram-mod", startup_timeout=30,
    )
    try:
        argv = engine.server.spec.to_argv(host="127.0.0.1", port=1)
        assert "--cpu-moe" in argv
        assert argv[argv.index("--spec-type") + 1] == "ngram-mod"
        assert argv[argv.index("-ngl") + 1] == "99"
    finally:
        engine.unload()


def test_capability_never_claims_layer_segments():
    engine = _engine()
    engine.load_model(model_path="dummy.gguf", startup_timeout=30)
    try:
        caps = engine.capability()
        assert caps["inference_path"] == "llama_server"
        assert caps["serves_layer_segments"] is False
    finally:
        engine.unload()


def test_double_unload_is_safe():
    engine = _engine()
    engine.load_model(model_path="dummy.gguf", startup_timeout=30)
    engine.unload()
    engine.unload()  # 不得抛异常


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(pytest.main([__file__, "-q"]))
