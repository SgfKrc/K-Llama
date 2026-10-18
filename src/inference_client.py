"""InferenceClient —— scheduler-svc 注入的推理宿主 HTTP 客户端（计划 §1.4）。

与 `InferenceHost` Protocol（model_host.py:20-64）同构：scheduler.py 通过
`Scheduler(host=..., callbacks=client.scheduler_callbacks)` 使用本客户端；
控制面所需的 API 辅助函数以不可变 callback bundle 显式注入。api_server
进程内的 scheduler 继续使用进程内 model_host，
互不影响。回退：`QLH_MONOLITH=1` 时 scheduler-svc（1.7）构造直接用
进程内 model_host，随时一键回单进程。

映射表（host 方法 → inference-svc 端点）：
  load_model / unload_model / switch_model   → POST /v1/models/{load,unload,switch}
  chat                                       → POST /v1/chat（streaming_mode=full）
  chat_stream                                → POST /v1/chat/stream（SSE 迭代）
  forward_layers                             → POST /v1/layers/forward
                                               （tensor 走 tensor_transport loopback）
  load_layer_range / unload_layer_range      → POST /v1/layers/{load,unload}
  kv_init / kv_free                          → POST /v1/kv/{init,free}
  cancel_generation                          → POST /v1/chat/cancel
  execute_task_worker_stage                  → POST /v1/worker/stage
  active_execution_model_identity           → GET /v1/models/current 推断
  format_model_response / build_model_chat_prompt
                                             → 本地等价实现（复用
                                               inference_service.engine_host
                                               保真复制版，纯函数无需 HTTP）
  snapshot_recent_logs / filter_recent_logs   → 本地日志快照
  model_loaded / current_quant               → GET /v1/status（带 TTL 缓存）
  full_chat_execution_lock                   → 本地 threading.RLock
"""
import base64
import json
import logging
import threading
import time
from typing import Any, Dict, Iterator, List, Optional

import requests

from model_host import SchedulerCallbackSet

logger = logging.getLogger("inference_client")

_STATUS_TTL = 2.0  # /v1/status 缓存秒数


class InferenceClient:
    """推理宿主 HTTP 客户端（InferenceHost 同构）。"""

    def __init__(self, base_url: Optional[str] = None, timeout: float = 300.0):
        import os
        from network_address import build_url

        host = os.environ.get("QLH_INFERENCE_HOST", "127.0.0.1")
        port = os.environ.get("QLH_INFERENCE_PORT", "8010")
        self._base_url = base_url or build_url("http", host, int(port))
        self._timeout = timeout
        self._lock = threading.RLock()
        self.full_chat_execution_lock = threading.RLock()
        self._status_cache: Dict[str, Any] = {}
        self._status_at = 0.0
        self._logs: List[dict] = []
        self._logs_lock = threading.Lock()
        self._scheduler_callbacks: Optional[SchedulerCallbackSet] = None

    # ------------------------------------------------------------------
    # 内部：HTTP 封装 + 状态缓存
    # ------------------------------------------------------------------
    def _post(self, path: str, payload: Optional[dict] = None) -> dict:
        resp = requests.post(
            f"{self._base_url}{path}",
            json=payload or {},
            timeout=self._timeout,
        )
        if resp.status_code >= 400:
            detail = ""
            try:
                detail = resp.json().get("detail", "")
            except Exception:
                detail = resp.text[:200]
            raise RuntimeError(f"{path} 失败 ({resp.status_code}): {detail}")
        return resp.json()

    def _get(self, path: str) -> dict:
        resp = requests.get(f"{self._base_url}{path}", timeout=10)
        if resp.status_code >= 400:
            raise RuntimeError(f"{path} 失败 ({resp.status_code})")
        return resp.json()

    def _status(self, force: bool = False) -> dict:
        with self._lock:
            now = time.time()
            if force or not self._status_cache or now - self._status_at > _STATUS_TTL:
                self._status_cache = self._get("/v1/status")
                self._status_at = now
            return self._status_cache

    # ------------------------------------------------------------------
    # 状态属性（scheduler 读取面）
    # ------------------------------------------------------------------
    @property
    def model_loaded(self) -> bool:
        return bool(self._status().get("model_loaded", False))

    @property
    def _model_path(self) -> str:
        """scheduler 层分配缓存键读取面（scheduler.py:2367 getattr 兜底）。

        推理进程内模型路径在微服务边界不可直读；返回不存在的路径，
        使 _layer_assignment_cache_key 的 os.walk 文件指纹分支跳过——
        否则 abspath('') = cwd，会遍历整个仓库目录（2026-08-05 实测
        /cluster/status 4.5s，TUI 5s 超时告警的根因）。
        """
        return "remote-model"

    @property
    def _total_model_layers(self) -> int:
        """推理进程内真实层数不可直读；返回 0 走 config.TOTAL_MODEL_LAYERS 兜底。"""
        return 0

    @property
    def current_quant(self) -> Optional[str]:
        return self._status().get("quant_type")

    @property
    def is_loaded(self) -> bool:
        return self.model_loaded

    # ------------------------------------------------------------------
    # InferenceHost Protocol：模型生命周期
    # ------------------------------------------------------------------
    def select_engine(self, profile=None) -> str:
        return "pytorch"  # 引擎选择由 inference-svc 进程内决定

    def load_model(self, engine=None, quant_type=None, use_compile=False,
                   model_id=None) -> dict:
        result = self._post("/v1/models/load", {
            "engine": engine, "quant_type": quant_type,
            "use_compile": use_compile, "model_id": model_id,
        })
        self._status(force=True)
        return result

    def unload_model(self) -> dict:
        result = self._post("/v1/models/unload", {})
        self._status(force=True)
        return result

    def switch_model(self, model_id: str, engine: Optional[str] = None) -> dict:
        return self._post("/v1/models/switch", {
            "model_id": model_id, "engine": engine,
        })

    def ensure_full_model(self) -> None:
        if not self.model_loaded:
            self.load_model()

    # ------------------------------------------------------------------
    # InferenceHost Protocol：层段接口（tensor loopback）
    # ------------------------------------------------------------------
    def load_layer_range(self, layer_range: str, embed: bool = False,
                         lm_head: bool = False) -> dict:
        return self._post("/v1/layers/load", {
            "layer_range": layer_range, "embed": embed, "lm_head": lm_head,
        })

    def unload_layer_range(self, layer_range: str) -> dict:
        return self._post("/v1/layers/unload", {"layer_range": layer_range})

    def forward_layers(self, layer_range: str, hidden, past_key_values=None,
                       **kw) -> Any:
        """hidden → tensor_transport 序列化 → base64 → HTTP loopback。"""
        from inference_service.tensor_transport import serialize_tensor

        tensor_ref = base64.b64encode(serialize_tensor(hidden)).decode("ascii")
        payload = {"layer_range": layer_range, "tensor_ref": tensor_ref}
        task_id = getattr(past_key_values, "task_id", None)
        if task_id is not None:
            payload["past_key_values_ref"] = task_id
        result = self._post("/v1/layers/forward", payload)
        from inference_service.tensor_transport import deserialize_tensor

        if "output_ref" in result:
            return deserialize_tensor(base64.b64decode(result["output_ref"]))
        return result.get("output")

    # ------------------------------------------------------------------
    # InferenceHost Protocol：chat
    # ------------------------------------------------------------------
    def chat(self, messages: List[dict], **kw) -> dict:
        """完整对话（JSON，非流式）。messages 取最后一条作为 message；
        session_id 透传保证多轮历史/追问语义与进程内 model_host 一致。"""
        message = messages[-1]["content"] if messages else ""
        payload = {
            "message": message,
            "session_id": kw.get("session_id"),
            "max_new_tokens": kw.get("max_tokens", kw.get("max_new_tokens", 1024)),
            "temperature": kw.get("temperature", 0.7),
            "top_p": kw.get("top_p", 0.9),
            "show_thinking": bool(kw.get("show_thinking", False)),
            "streaming_mode": "full",
        }
        result = self._post("/v1/chat", payload)
        return {
            "content": result.get("content") or result.get("response", ""),
            "tokens_per_second": result.get("metrics", {}).get(
                "tokens_per_second", 0),
            "usage": result.get("metrics", {}).get("usage", {}),
            "followups": result.get("followups", []),
            "metrics": result.get("metrics", {}),
        }

    def chat_stream(self, messages: List[dict], **kw) -> Iterator[dict]:
        """SSE 流式（逐 token 事件 + 结束事件）。"""
        message = messages[-1]["content"] if messages else ""
        payload = {
            "message": message,
            "session_id": kw.get("session_id"),
            "max_new_tokens": kw.get("max_tokens", kw.get("max_new_tokens", 1024)),
            "temperature": kw.get("temperature", 0.7),
            "top_p": kw.get("top_p", 0.9),
            "show_thinking": bool(kw.get("show_thinking", False)),
            "streaming_mode": "fast",
        }
        with requests.post(
            f"{self._base_url}/v1/chat/stream",
            json=payload,
            stream=True,
            timeout=self._timeout,
        ) as resp:
            if resp.status_code >= 400:
                raise RuntimeError(f"chat/stream 失败 ({resp.status_code})")
            for raw in resp.iter_lines(decode_unicode=True):
                if not raw or not raw.startswith("data: "):
                    continue
                event = json.loads(raw[len("data: "):])
                if "token" in event:
                    yield {"token": event["token"]}
                elif event.get("done"):
                    yield {
                        "done": True,
                        "response": event.get("response", ""),
                        "metrics": event.get("metrics", {}),
                    }
                    return

    # ------------------------------------------------------------------
    # InferenceHost Protocol：KV 生命周期
    # ------------------------------------------------------------------
    def kv_init(self, task_id: Optional[str] = None, **kw) -> dict:
        return self._post("/v1/kv/init", {"task_id": task_id, **kw})

    def kv_free(self, task_id: str) -> dict:
        return self._post("/v1/kv/free", {"task_id": task_id})

    def cancel_generation(self, generation_id: str) -> dict:
        return self._post("/v1/chat/cancel", {"generation_id": generation_id})

    # ------------------------------------------------------------------
    # scheduler callback bundle：task-worker Stage 与模型身份
    # ------------------------------------------------------------------
    def execute_task_worker_stage(self, stage_request, cancel_event=None) -> dict:
        """远程执行 task-worker Stage（scheduler 调用面 → /v1/worker/stage）。"""
        from dataclasses import asdict

        payload = asdict(stage_request)
        if payload.get("model_identity") is not None:
            payload["model_identity"] = asdict(payload["model_identity"])
        result = self._post("/v1/worker/stage", payload)
        if not isinstance(result, dict):
            raise RuntimeError("remote Stage executor returned non-object output")
        return result

    def active_execution_model_identity(self):
        """从 /v1/models/current 推断当前模型身份（等价 api_server 版语义）。"""
        try:
            info = self._get("/v1/models/current")
        except Exception:
            return None
        if not info.get("loaded"):
            return None
        engine = str(info.get("engine") or "")
        model_id = str(info.get("model_id") or "")
        if engine not in {"pytorch", "llama_cpp", "island"} or not model_id:
            return None
        try:
            from execution_provider import ModelIdentity

            return ModelIdentity(
                model_id=model_id,
                engine=engine,
                format="",
                revision="",
                sha256="",
            )
        except Exception:
            return None

    # ------------------------------------------------------------------
    # scheduler callback bundle：辅助纯函数（本地等价实现，复用保真复制版）
    # ------------------------------------------------------------------
    def format_model_response(self, text: str, show_thinking: bool,
                              native_thinking_prompt: bool = False):
        from inference_service.engine_host import _format_model_response as _f

        return _f(text, show_thinking, native_thinking_prompt=native_thinking_prompt)

    def build_model_chat_prompt(self, tokenizer, messages, system_prompt=None,
                                assistant_prefill=None) -> str:
        from inference_service.engine_host import _build_model_chat_prompt as _b

        return _b(tokenizer, messages, system_prompt=system_prompt,
                  assistant_prefill=assistant_prefill)

    # ------------------------------------------------------------------
    # scheduler callback bundle：日志快照（本地记录，等价语义）
    # ------------------------------------------------------------------
    def snapshot_recent_logs(self) -> tuple:
        with self._logs_lock:
            return list(self._logs), len(self._logs)

    def filter_recent_logs(self, entries, level="", name="", limit=200) -> list:
        out = []
        for e in entries:
            if level and e.get("level", "").lower() != level.lower():
                continue
            if name and name not in e.get("name", ""):
                continue
            out.append(e)
            if len(out) >= limit:
                break
        return out

    @property
    def scheduler_callbacks(self) -> SchedulerCallbackSet:
        """Return the complete scheduler dependency bundle for this remote host.

        The bundle is built once so scheduler construction has no dependency on
        mutation order or private attributes of the host implementation.
        """
        with self._lock:
            callbacks = self._scheduler_callbacks
            if callbacks is None:
                from inference_service.engine_host import THINKING_SYSTEM_PROMPT

                callbacks = SchedulerCallbackSet(
                    active_execution_model_identity=(
                        self.active_execution_model_identity
                    ),
                    execute_task_worker_stage=self.execute_task_worker_stage,
                    build_model_chat_prompt=self.build_model_chat_prompt,
                    thinking_system_prompt=THINKING_SYSTEM_PROMPT,
                    snapshot_recent_logs=self.snapshot_recent_logs,
                    filter_recent_logs=self.filter_recent_logs,
                    format_model_response=self.format_model_response,
                )
                self._scheduler_callbacks = callbacks
            return callbacks

    def _record_local_log(self, level: str, name: str, message: str) -> None:
        with self._logs_lock:
            self._logs.append({
                "time": time.time(),
                "level": level,
                "name": name,
                "message": message,
            })
            if len(self._logs) > 1000:
                self._logs = self._logs[-500:]
