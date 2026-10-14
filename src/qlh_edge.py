"""QLH Edge —— 本地小模型 + 原生 RPC worker 的边缘入口。

基线定义见 local_docs/plans/边缘基线框定-2026-09-14.md：
- 只依赖白名单：llama-cpp-python + fastapi + uvicorn + psutil（+ numpy 传递依赖）
- 不 import torch / transformers / accelerate / 多模态 / RAG / 实验链
- 目标：运行时 venv ≤300MB、冷启动 ≤15s（无 torch import 链）

启动（在 edge venv 中）：
    QLH_EDGE_MODEL=models/Qwen-1_8B-Chat.Q4_K_M.gguf \\
        .venv-edge/Scripts/python.exe -m uvicorn qlh_edge:app --port 8010

端点：
    GET  /health            —— 进程/内存/模型状态（不含路径泄漏）
    GET  /status            —— 边缘基线元信息 + 模型懒加载状态
    POST /generate          —— 本地模型生成（llama.cpp，首次请求懒加载）
    GET  /capabilities      —— 本地推理与 RPC worker 能力合同
    GET  /rpc/status        —— 原生 llama.cpp RPC worker 状态
    POST /rpc/start|stop    —— 显式启动/停止本机 RPC worker
"""
from __future__ import annotations

import argparse
import os
import sys
import threading
import time
from pathlib import Path

_SRC_DIR = Path(__file__).resolve().parent
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

import psutil  # noqa: E402
from fastapi import FastAPI, HTTPException  # noqa: E402
from pydantic import BaseModel, Field  # noqa: E402

from edge_cluster import EdgeRpcWorker, EdgeRpcWorkerError  # noqa: E402
from release_contract import PRODUCT_VERSION  # noqa: E402

EDGE_BASELINE = {
    "edition": "edge",
    "runtime_whitelist": ["llama-cpp-python", "fastapi", "uvicorn", "psutil", "httpx"],
    "excluded": ["torch", "transformers", "accelerate", "multimodal", "rag", "tui"],
    "venv_budget_mb": 300,
    "cold_start_budget_s": 15,
}

app = FastAPI(title="QLH Edge", version=PRODUCT_VERSION)

_state_lock = threading.RLock()
_llm = None
_llm_path: str = ""
_rpc_worker = EdgeRpcWorker()


class GenerateRequest(BaseModel):
    prompt: str = Field(..., min_length=1, max_length=16_384)
    max_tokens: int = Field(default=128, ge=1, le=1024)
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    top_p: float = Field(default=0.8, gt=0.0, le=1.0)


def _model_path() -> str:
    path = os.environ.get("QLH_EDGE_MODEL", "").strip()
    if not path:
        raise HTTPException(status_code=503, detail="QLH_EDGE_MODEL not configured")
    candidate = Path(path)
    if not candidate.is_file():
        raise HTTPException(status_code=503, detail="configured model file not found")
    return str(candidate)


def _get_llm():
    """Lazy-load the GGUF model once (single-model edge profile)."""
    global _llm, _llm_path
    with _state_lock:
        path = _model_path()
        if _llm is None or _llm_path != path:
            from llama_cpp import Llama  # deferred: keeps cold start free of the engine

            _llm = Llama(
                model_path=path,
                n_ctx=int(os.environ.get("QLH_EDGE_N_CTX", "2048")),
                n_threads=int(os.environ.get("QLH_EDGE_THREADS", "0")) or None,
                verbose=False,
            )
            _llm_path = path
        return _llm


@app.get("/health")
def health():
    mem = psutil.virtual_memory()
    return {
        "status": "ok",
        "edition": "edge",
        "model_loaded": _llm is not None,
        "ram_total_mb": round(mem.total / 2**20),
        "ram_available_mb": round(mem.available / 2**20),
        "process_rss_mb": round(psutil.Process().memory_info().rss / 2**20),
        "pid": os.getpid(),
    }


@app.get("/status")
def status():
    payload = dict(EDGE_BASELINE)
    payload["model_loaded"] = _llm is not None
    payload["model_configured"] = bool(os.environ.get("QLH_EDGE_MODEL", "").strip())
    payload["distributed_inference"] = _distributed_capabilities()
    payload["started_at_unix"] = _STARTED_AT
    return payload


def _distributed_capabilities() -> dict:
    return {
        "local_inference": True,
        "local_engine": "llama_cpp",
        "local_model_policy": "prefer_le_1b",
        "worker_engine": "llama_cpp_rpc",
        "worker_role": "rpc_worker",
        "rpc_worker": _rpc_worker.snapshot(),
        "failure_bypass": ["other_shard_topology", "full_model_node", "local_small_model"],
    }


@app.get("/capabilities")
def capabilities():
    """Expose the Edge node contract without exposing local filesystem paths."""
    return {
        "schema": "qlh.edge.capabilities.v1",
        "edition": "edge",
        "engine": "llama_cpp",
        "model_format": "gguf",
        "default_model_policy": "prefer_le_1b",
        "distributed_inference": _distributed_capabilities(),
    }


@app.get("/rpc/status")
def rpc_status():
    return _rpc_worker.snapshot()


@app.post("/rpc/start")
def rpc_start():
    try:
        return _rpc_worker.start()
    except EdgeRpcWorkerError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


@app.post("/rpc/stop")
def rpc_stop():
    try:
        return _rpc_worker.stop()
    except EdgeRpcWorkerError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


@app.post("/generate")
def generate(req: GenerateRequest):
    started = time.perf_counter()
    llm = _get_llm()
    out = llm(
        req.prompt,
        max_tokens=req.max_tokens,
        temperature=req.temperature,
        top_p=req.top_p,
        echo=False,
    )
    text = out["choices"][0]["text"] if out.get("choices") else ""
    usage = out.get("usage", {})
    return {
        "text": text,
        "prompt_tokens": usage.get("prompt_tokens"),
        "completion_tokens": usage.get("completion_tokens"),
        "elapsed_s": round(time.perf_counter() - started, 3),
    }


_STARTED_AT = time.time()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run QLH Edge local inference and optional native RPC worker.")
    parser.add_argument("--host", default=os.environ.get("QLH_EDGE_HOST", "127.0.0.1"))
    parser.add_argument(
        "--port",
        type=int,
        default=int(os.environ.get("QLH_EDGE_PORT", "8010")),
    )
    args = parser.parse_args(argv)

    import uvicorn

    uvicorn.run(app, host=args.host, port=args.port)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
