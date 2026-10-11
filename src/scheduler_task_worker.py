"""Task-worker protocol methods mixed into the Scheduler facade."""

from __future__ import annotations

import base64
import hashlib
import importlib
import logging
import os
import re
import threading
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Mapping, Optional

from scheduler_types import NodeRole
from task_provider import (
    ModelIdentity as TaskModelIdentity,
    StageRequest as TaskProviderStageRequest,
    sanitize_result_metadata as sanitize_task_result_metadata,
)
from task_worker_adapter import RemoteFullWorkerProvider, remote_provider_id
from task_worker_chunks import (
    build_stage_chunk,
    plan_stage_payload_chunks,
    stage_chunk_ref,
)
from task_worker_protocol import (
    MAX_MESSAGE_BYTES,
    PROTOCOL_VERSION as TASK_WORKER_PROTOCOL_VERSION,
    RUNTIME_PROFILE_UNSPECIFIED,
    WorkerMessage,
    WorkerProtocolError,
    build_message as build_task_worker_message,
    canonical_message_bytes as task_worker_message_bytes,
    canonical_sha256 as task_worker_sha256,
    decode_message as decode_task_worker_message,
    worker_protocol_status,
)
from torch_runtime import torch_available

logger = logging.getLogger("scheduler")

_LAYER_ARTIFACT_DIGEST_LOCK = threading.Lock()
_LAYER_ARTIFACT_DIGEST_CACHE: dict[str, tuple[tuple[int, int, int], str]] = {}
_TASK_WORKER_LOCK_WAIT_SLICE_SECONDS = 0.05
_REQUEST_DEADLINE_EXCEEDED = "request_deadline_exceeded"


def keep_head_shim_available() -> bool:
    """本机是否具备可用的 keep-head shim（llama.cpp 层段的另一条执行路径）。

    ★ 2026-10-11（ROG+Y700+本机三机实测）：`_task_worker_capabilities` 此前只用
    `importlib.util.find_spec("llama_cpp")` 判断"能不能跑 llama.cpp"。但**层段执行**
    走的是 shim（`QLH_KEEP_HEAD_SHIM` 指向的 `qlh_keep_head.dll` + 同目录私有
    libllama/ggml），**不需要 pip 包** ⇒ ROG 明明配了 shim 与裁层工件，却被协议判为
    「model engine was not declared by the worker」，HELLO 直接被拒。
    这里把 shim 作为同等有效的判据（事实源与 `keep_head_shim.resolve_keep_head_shim`
    一致，避免第二套推导）。
    """
    try:
        from keep_head_shim import resolve_keep_head_shim
    except ImportError:
        return False
    try:
        path = resolve_keep_head_shim()
    except Exception:  # noqa: BLE001 - 判据失败按"不可用"处理，绝不抛出
        return False
    # `resolve_keep_head_shim` 只负责"取值"（显式 > env > 仓库默认，**不校验存在性**）；
    # 本判据问的是**可用性** ⇒ 必须确认文件真的在，否则一个指错的 env 就会被当成能力。
    from pathlib import Path

    return bool(path) and Path(path).is_file()


def _verified_layer_artifact_sha256(path) -> str:
    """Hash one configured GGUF once per stable filesystem identity."""
    try:
        resolved = path.resolve(strict=True)
        stat = resolved.stat()
        if not resolved.is_file():
            return ""
        fingerprint = (
            int(stat.st_size),
            int(getattr(stat, "st_mtime_ns", 0)),
            int(getattr(stat, "st_ctime_ns", 0)),
        )
    except OSError:
        return ""
    cache_key = str(resolved)
    with _LAYER_ARTIFACT_DIGEST_LOCK:
        cached = _LAYER_ARTIFACT_DIGEST_CACHE.get(cache_key)
        if cached is not None and cached[0] == fingerprint:
            return cached[1]
        digest = hashlib.sha256()
        try:
            with resolved.open("rb") as handle:
                for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
                    digest.update(chunk)
        except OSError:
            return ""
        value = digest.hexdigest()
        _LAYER_ARTIFACT_DIGEST_CACHE[cache_key] = (fingerprint, value)
        return value


@dataclass
class _TaskWorkerActiveAttempt:
    workflow_id: str
    stage_id: str
    attempt_id: str
    lease_id: str
    lease_epoch: int
    provider_id: str
    lease_expires_at_ms: int
    lease_deadline_monotonic: float
    request_deadline_ms: Optional[int] = None
    request_deadline_monotonic: Optional[float] = None
    cancel_event: threading.Event = field(default_factory=threading.Event)
    done_event: threading.Event = field(default_factory=threading.Event)
    lease_expired: bool = False
    cancel_reason: str = ""
    terminal_reason: str = ""
    terminal_decided_at_monotonic: float = 0.0


class SchedulerTaskWorkerMixin:
    def _configured_layer_artifact(self) -> dict | None:
        """从 env 指定的层段工件推导身份与层区间；取不到返回 None。

        读同目录同名的 `.manifest.json`（`scripts/cut_layers.py` 产出）。这里刻意
        **不依赖** master 下发的 layer config —— 声明必须能先于配置成立，否则首次
        hello 会形成死锁（见 `_task_worker_capabilities` 里的说明）。

        返回的 `sha256` 用**工件自身**的摘要（不是源模型摘要）：Route A 的 offer 身份
        要与 worker 手上那份 GGUF 对齐，`task_worker_adapter._layer_model_matches`
        比对的正是这一项。
        """
        import json
        from pathlib import Path

        model_path = os.environ.get("QLH_LAYER_GGUF", "").strip()
        if not model_path:
            return None
        artifact_path = Path(model_path)
        if not artifact_path.is_file():
            return None
        manifest_path = artifact_path.with_suffix(".manifest.json")
        if not manifest_path.is_file():
            return None
        try:
            data = json.loads(manifest_path.read_text(encoding="utf-8"))
        except Exception:
            return None
        value = data.get("source_layer_range")
        if not (isinstance(value, list) and len(value) == 2):
            return None
        if any(isinstance(item, bool) or not isinstance(item, int) for item in value):
            return None
        if value[0] < 0 or value[1] <= value[0]:
            return None
        expected_sha256 = str(data.get("artifact_sha256", "") or "").lower()
        if (
            len(expected_sha256) != 64
            or any(char not in "0123456789abcdef" for char in expected_sha256)
            or _verified_layer_artifact_sha256(artifact_path) != expected_sha256
        ):
            return None
        segment_mode = str(data.get("mode", "") or "").lower()
        if segment_mode not in {"head", "middle", "tail"}:
            return None
        source_sha256 = str(data.get("source_model_sha256", "") or "").lower()
        if source_sha256 and (
            len(source_sha256) != 64
            or any(char not in "0123456789abcdef" for char in source_sha256)
        ):
            return None
        source_model_id = str(data.get("source_model_id", "") or "").strip()
        tokenizer_sha256 = str(data.get("tokenizer_sha256", "") or "").lower()
        try:
            hidden_size = int(data.get("hidden_size", 0) or 0)
        except (TypeError, ValueError):
            hidden_size = 0
        # New cut manifests always expose ``hidden_size``.  The logical-model
        # preflight contract is opt-in only when its two explicit identity
        # fields are present, so old invocations remain a visible legacy
        # artifact instead of disappearing from capabilities altogether.
        contract_present = bool(source_model_id or tokenizer_sha256)
        contract_complete = bool(
            source_model_id
            and hidden_size > 0
            and re.fullmatch(r"[0-9a-f]{64}", tokenizer_sha256)
        )
        if contract_present and not contract_complete:
            return None
        model_id = artifact_path.name
        if re.fullmatch(r"[A-Za-z0-9_.:-]{1,128}", model_id) is None:
            return None
        # `model_id` 取**文件名**，不用 manifest 的 `artifact` 字段：后者是相对路径
        # （含 `\`），而协议对它的要求是 `^[A-Za-z0-9_.:-]{1,128}$` —— 反斜杠不合法 ⇒
        # 整个 hello 会被判 `payload.capabilities.models[0].model_id is invalid`，worker
        # 永远进不了 `admitted`（实测：这条错误被 legacy 通道的噪声盖了很久才浮出来）。
        result = {
            "start": int(value[0]),
            "end": int(value[1]),
            "model_id": model_id,
            "sha256": expected_sha256,
            "source_model_sha256": source_sha256,
            "revision": str(data.get("generator_version", "") or ""),
            "segment_mode": segment_mode,
        }
        if contract_complete:
            result.update({
                "source_model_id": source_model_id,
                "hidden_size": hidden_size,
                "tokenizer_sha256": tokenizer_sha256,
            })
        return result

    def _task_worker_stage_result_messages(
        self,
        result_payload: dict,
        *,
        offer: Mapping[str, Any],
        attempt_id: str,
        provider_id: str,
    ) -> list:
        """构造 stage_result；`output.hidden_out_f32` 超单帧预算时先切**回程分片**。

        ★ 2026-10-11（ROG+Y700+本机三机实测）：协议此前只做了**输入方向**分片，回程
        hidden 必须整体塞进 `MAX_MESSAGE_BYTES`（8 MiB）⇒ 长 prefill（实测 1004 tokens，
        f32 ≈ 8.2 MiB、base64 后 ≈ 11 MiB）必然 `message exceeds maximum size`，主节点
        只看到 `remote worker reported a Stage error`。

        这里复用输入方向的 `plan_stage_payload_chunks` 与
        `build_stage_chunk(message_type="stage_result_chunk")`：先发有序分片，再发一条引用
        它们的 `stage_result`（`output.hidden_out_ref`）。对端若不认识该类型会以
        `unsupported_message_type` 拒绝 ⇒ 与「超限失败」等价的 fail-closed，无静默降级。
        `output_sha256` 仍按**原始完整 output** 计算（对端还原后可自行复核）。
        """
        now_ms = int(time.time() * 1000)

        def _result(payload: dict):
            return build_task_worker_message(
                "stage_result",
                payload,
                message_id=f"msg_result_{uuid.uuid4().hex}",
                sent_at_ms=now_ms,
                version=TASK_WORKER_PROTOCOL_VERSION,
            )

        output = result_payload.get("output")
        hidden = output.get("hidden_out_f32") if isinstance(output, Mapping) else None
        if not isinstance(hidden, str) or not hidden:
            return [_result(result_payload)]
        if task_worker_message_bytes(_result(result_payload)) <= MAX_MESSAGE_BYTES:
            return [_result(result_payload)]
        try:
            raw = base64.b64decode(hidden, validate=True)
            plan = plan_stage_payload_chunks(raw)
        except Exception:
            # 切分失败不得掩盖真因：照常返回原消息，让「超限」错误原样暴露。
            logger.warning(
                "event=task_worker_stage_result_chunk_plan_failed attempt_id=%s",
                attempt_id, exc_info=True,
            )
            return [_result(result_payload)]
        messages = []
        for index, chunk in enumerate(plan.chunks):
            messages.append(build_stage_chunk(
                workflow_id=str(offer["workflow_id"]),
                stage_id=str(offer["stage_id"]),
                attempt_id=attempt_id,
                lease_id=str(offer["lease_id"]),
                lease_epoch=int(offer["lease_epoch"]),
                provider_id=provider_id,
                chunk_index=index,
                chunk_count=plan.chunk_count,
                payload=chunk,
                total_bytes=plan.total_bytes,
                message_id=f"msg_chunk_{uuid.uuid4().hex}",
                sent_at_ms=now_ms,
                version=TASK_WORKER_PROTOCOL_VERSION,
                message_type="stage_result_chunk",
            ))
        shrunk = dict(result_payload)
        new_output = dict(output)
        new_output.pop("hidden_out_f32", None)
        new_output["hidden_out_ref"] = stage_chunk_ref(plan)
        shrunk["output"] = new_output
        shrunk["output_sha256"] = task_worker_sha256(output)
        messages.append(_result(shrunk))
        logger.info(
            "event=task_worker_stage_result_chunked attempt_id=%s chunks=%d bytes=%d",
            attempt_id, plan.chunk_count, plan.total_bytes,
        )
        return messages

    def _task_worker_capabilities(self) -> dict:
        """Build an honest PC Full Worker snapshot without loading a model."""
        engines = []
        if torch_available():
            engines.append("pytorch")
        try:
            if importlib.util.find_spec("llama_cpp") is not None:
                engines.append("llama_cpp")
            elif keep_head_shim_available():
                # shim 路径同样能执行 llama.cpp 层段（见 keep_head_shim_available 的说明）。
                engines.append("llama_cpp")
        except (ImportError, ValueError):
            pass
        # TP 孤岛网关：以 island 引擎承担整请求推理（配置启用即上报能力）
        try:
            import config as _island_cfg
            if (getattr(_island_cfg, "ISLAND_ENABLED", False)
                    and getattr(_island_cfg, "ISLAND_BASE_URL", "")):
                engines.append("island")
        except Exception:
            pass
        # 外部推理服务（路线 B）：以 external_api 引擎承担整请求推理。
        # 仅声明能力，实际外发仍受数据作用域门控约束（默认 opt_in 不出集群）。
        try:
            import config as _external_cfg
            if (getattr(_external_cfg, "EXTERNAL_ENABLED", False)
                    and getattr(_external_cfg, "EXTERNAL_BASE_URL", "")):
                engines.append("external_api")
        except Exception:
            pass
        if not engines:
            # The PC application currently requires the PyTorch runtime. Keeping
            # the schema valid also makes a broken installation visible in hello.
            engines.append("pytorch")

        models = []
        try:
            # 阶段 0.2：完整模型判定直接走 host 代理（不依赖内部 manager 容器
            # 的 _instance 结构，阶段 1 替换远程 host 适配器后依然成立）
            has_loaded_model = getattr(self._host, "has_loaded_model", None)
            if callable(has_loaded_model):
                # ModelHost owns the authoritative loaded-state check and
                # correctly accounts for its lazy manager proxy.  Requiring
                # the legacy ``is_loaded`` attribute here made a real full
                # Safetensors worker advertise an empty model list.
                full_model_loaded = bool(has_loaded_model())
            else:
                full_model_loaded = bool(
                    getattr(self._host, "model_loaded", False)
                    and getattr(self._host, "is_loaded", False)
                )
            full_model_loaded = bool(
                full_model_loaded and getattr(self._host, "layer_range", None) is None
            )
            if full_model_loaded:
                identity = self._require_callbacks().active_task_graph_model_identity()
                if identity is not None and identity.engine in engines:
                    models.append({
                        "model_id": identity.model_id,
                        "engine": identity.engine,
                        "format": identity.format,
                        "revision": identity.revision,
                        "sha256": identity.sha256,
                    })
        except Exception:
            logger.warning(
                "构建 PC Full Worker 模型能力快照失败，暂不上报模型",
                exc_info=True,
            )
        active_layer_config = getattr(self, "_active_layer_config", None) or {}
        layer_worker = bool(
            active_layer_config or getattr(self._host, "layer_range", None)
        )
        # ★ 2026-10-03：载了层段工件就同时承担 v3 层段 Stage。此前 `stage_types` 硬编码
        #   两个整模类型，而层段执行路径（`_handle_task_worker_stage_offer`）已实现 ——
        #   结果是 PC worker 永远不会被派到 `layer_forward`。
        stage_types = ["full_inference", "aggregate"]
        layer_ranges: list[list[int]] = []
        # ★ #28：**工件身份**（「我手上有哪份权重」）与**就绪区间**（「我现在能跑哪几层」）
        #   是两件事，此前共用一个 `if layer_worker / else` 分支 ⇒ 只要 `layer_range` 残留
        #   或 `_active_layer_config` 还在，整个工件身份分支就被跳过，`models` 变空。
        #   而主节点把这份 hello 快照当层分配与 Stage offer 的唯一身份来源 ⇒ 远端 Stage
        #   以 `model_identity_mismatch` 被拒（worker 侧 `_handle_task_worker_stage_offer`
        #   还会用**实时** capabilities 再比对一次，两侧都不一致时症状更隐蔽）。
        #   ⇒ 身份**不再**依附于就绪状态：只要手上有工件就上报。
        artifact = self._configured_layer_artifact()
        artifact_required = bool(
            os.environ.get("QLH_LAYER_GGUF", "").strip()
        )
        if artifact_required and artifact is None:
            # An explicitly configured but missing/corrupt GGUF must not keep
            # advertising a stale active range from the previous generation.
            layer_worker = False
        if artifact is not None:
            # A fixed GGUF segment is executable only for the exact manifest
            # range.  Never let a stale active config publish a second range
            # beside the per-artifact contract.
            layer_worker = True
            layer_ranges.append([artifact["start"], artifact["end"]])
        elif layer_worker:
            layer_range = active_layer_config.get("layer_range")
            if (
                isinstance(layer_range, (list, tuple))
                and len(layer_range) == 2
                and not any(
                    isinstance(value, bool) or not isinstance(value, int)
                    for value in layer_range
                )
            ):
                layer_ranges.append([int(layer_range[0]), int(layer_range[1])])
        if artifact is not None and artifact.get("sha256"):
            # 层段 worker 不加载整模 ⇒ 只靠上面的整模分支时 `models` 会是空的，而 Route A
            # 的 offer 身份正是从这里取（`_route_a_stage_model_identity`）⇒ 缺了它整条链
            # 会以 `route_a_stage_model_identity_unavailable` 失败。用**工件身份**顶上：
            # `task_worker_adapter._layer_model_matches` 比对的就是 engine/format/sha256。
            # 整模身份优先（上面已填 `models` 时不覆盖）—— 两者冲突说明本机同时握着整模
            # 与工件，此时以整模为准是保守选择。
            artifact_model = {
                "model_id": artifact["model_id"],
                "engine": "llama_cpp",
                "format": "gguf",
                "revision": artifact["revision"],
                "sha256": artifact["sha256"],
            }
            existing = next((
                item for item in models
                if item.get("model_id") == artifact_model["model_id"]
            ), None)
            if existing is None:
                models.append(artifact_model)
            elif existing != artifact_model:
                raise RuntimeError("layer_artifact_model_identity_conflict")
        if layer_worker and "layer_forward" not in stage_types:
            stage_types.append("layer_forward")
        capabilities = {
            "stage_types": stage_types,
            "engines": engines,
            "models": models,
            "max_concurrency": 1,
            "runtime_profile": self._runtime_profile_for_capabilities(),
            # A distributed-only layer/relay worker is intentionally not a
            # Full Worker.  Its segment capability is negotiated by the
            # layer-config contract, so advertising no full-model identity
            # must not cause the coordinator to opt it out of layer work.
            "layer_worker": layer_worker,
            # 当前**就绪、马上能跑**的层区间（与 `layer_budget` 的"承载上限"分工明确）。
            "layer_ranges": layer_ranges,
            # ★ 2026-10-08（DIST-NEXT-2b）：声明**能接收分片 hidden**。接收侧
            #   （`_handle_task_worker_stage_chunk` + `hidden_ref` 装配）早已实现，但此前
            #   没有任何声明处 —— 于是 provider 的 `_maybe_send_stage_chunks` 直接跳过
            #   分片、dispatch 前的预检按超限拒绝，大 prompt 只能 503
            #   （实测 `route_a_stage_frame_too_large:…:wire=24160940:budget=8126464`）。
            #   非层段 worker 不接收 hidden，故按 `layer_worker` 保守声明。
            "stage_chunked_input": bool(layer_worker),
            "relay_middle": bool(
                layer_worker
                and active_layer_config
                and str(active_layer_config.get("engine", ""))
                == "relay_middle"
            ),
        }
        if artifact is not None and artifact.get("segment_mode") in {
            "head", "middle", "tail",
        }:
            item = {
                "layer_range": [artifact["start"], artifact["end"]],
                "segment_mode": artifact["segment_mode"],
                "model_id": artifact["model_id"],
                "artifact_sha256": artifact["sha256"],
            }
            if artifact.get("source_model_sha256"):
                item["source_model_sha256"] = artifact["source_model_sha256"]
            if artifact.get("source_model_id"):
                item.update({
                    "source_model_id": artifact["source_model_id"],
                    "hidden_size": artifact["hidden_size"],
                    "tokenizer_sha256": artifact["tokenizer_sha256"],
                })
            capabilities["layer_artifacts"] = [item]
            capabilities["segment_mode"] = artifact["segment_mode"]
        return capabilities

    @staticmethod
    def _runtime_profile_for_capabilities() -> str:
        """Return the launcher-selected release profile without probing engines."""
        try:
            from device_profiler import detect_runtime_profile

            return detect_runtime_profile()
        except Exception:
            # A malformed or partial slim installation must remain visible as
            # unspecified; it must never claim a stronger release profile.
            return RUNTIME_PROFILE_UNSPECIFIED


    def _send_task_worker_hello(
        self, client=None, refresh_generation: int = 0,
    ) -> bool:
        """Send the TC-N2.0 hello after authenticated TCP registration."""
        if (
            self._effective_role() != "client"
            or not self._scheduler_facade_global('TASK_WORKER_EXPERIMENTAL_ENABLED')
        ):
            return False
        target = client or getattr(self, "_tcp_client", None)
        if not target or not getattr(target, "is_registered", False):
            return False
        try:
            from transport_port import MessageType

            message = self._task_worker_control.begin_worker_hello(
                node_id=self.get_effective_node_id(),
                capabilities=self._task_worker_capabilities(),
            )
            if message is None:
                return False
            with self._task_worker_refresh_lock:
                self._task_worker_refresh_requested = bool(
                    self._task_worker_refresh_generation > refresh_generation
                )
            target.send_data(message.snapshot(), MessageType.TASK_WORKER)
            logger.info(
                "event=task_worker_hello_sent node_id=%s version=%s",
                self.get_effective_node_id(), message.version,
            )
            return True
        except Exception as exc:
            self._task_worker_control.disconnect_coordinator()
            logger.warning(
                "event=task_worker_hello_failed node_id=%s error=%s",
                self.get_effective_node_id(),
                getattr(exc, "code", type(exc).__name__),
                exc_info=True,
            )
            return False


    def refresh_task_worker_capabilities(self) -> bool:
        """Refresh hello asynchronously after a local full-model change."""
        client = getattr(self, "_tcp_client", None)
        if (
            self._effective_role() != "client"
            or not self._scheduler_facade_global('TASK_WORKER_EXPERIMENTAL_ENABLED')
            or not client
            or not getattr(client, "is_registered", False)
        ):
            return False
        with self._task_worker_refresh_lock:
            self._task_worker_refresh_requested = True
            self._task_worker_refresh_generation += 1
            refresh_generation = self._task_worker_refresh_generation
        threading.Thread(
            target=self._send_task_worker_hello,
            args=(client, refresh_generation),
            name="task-worker-capability-refresh",
            daemon=True,
        ).start()
        return True


    def _send_task_worker_to_node(
        self, node_id: str, message: WorkerMessage,
    ) -> None:
        server = self._tcp_server
        if server is None:
            raise ConnectionError("task worker TCP server is unavailable")
        from transport_port import MessageType

        server.send_to_client(
            node_id, message.snapshot(), MessageType.TASK_WORKER,
        )


    def _send_task_worker_to_master(self, message: WorkerMessage) -> None:
        client = getattr(self, "_tcp_client", None)
        if not client or not getattr(client, "is_registered", False):
            raise ConnectionError("task worker is not connected to its master")
        from transport_port import MessageType

        client.send_data(message.snapshot(), MessageType.TASK_WORKER)


    def _ensure_remote_task_worker_provider(
        self, node_id: str,
    ) -> RemoteFullWorkerProvider:
        with self._task_worker_stage_lock:
            provider = self._remote_task_worker_providers.get(node_id)
            if provider is None:
                provider = RemoteFullWorkerProvider(
                    node_id=node_id,
                    peer_snapshot=lambda bound_node=node_id: (
                        self._task_worker_control.worker_snapshot(bound_node)
                    ),
                    send_message=lambda message, bound_node=node_id: (
                        self._send_task_worker_to_node(bound_node, message)
                    ),
                )
                self._remote_task_worker_providers[node_id] = provider
            return provider


    def remote_task_worker_providers(self) -> list[RemoteFullWorkerProvider]:
        """Return stable Provider objects for explicit TaskGraph registration."""
        with self._task_worker_stage_lock:
            return [
                self._remote_task_worker_providers[node_id]
                for node_id in sorted(self._remote_task_worker_providers)
            ]


    @staticmethod
    def _task_worker_message_digest(message: WorkerMessage) -> str:
        return hashlib.sha256(task_worker_message_bytes(message)).hexdigest()


    def _prepare_task_worker_request(
        self, message: WorkerMessage,
    ) -> tuple[bool, list[dict]]:
        digest = self._task_worker_message_digest(message)
        with self._task_worker_stage_lock:
            cached = self._task_worker_seen_messages.get(message.message_id)
            if cached is not None:
                if cached[0] != digest:
                    raise WorkerProtocolError(
                        "message_id was reused with different Stage content",
                        code="message_id_conflict",
                        field="message_id",
                    )
                return True, [dict(response) for response in cached[1]]
            self._task_worker_seen_messages[message.message_id] = (digest, [])
            self._task_worker_seen_order.append(message.message_id)
            while len(self._task_worker_seen_order) > 1024:
                expired = self._task_worker_seen_order.popleft()
                self._task_worker_seen_messages.pop(expired, None)
            return False, []


    def _cache_task_worker_response(
        self, request_message_id: str, response: WorkerMessage,
    ) -> None:
        with self._task_worker_stage_lock:
            cached = self._task_worker_seen_messages.get(request_message_id)
            if cached is not None:
                cached[1].append(response.snapshot())


    def _forget_task_worker_request(self, request_message_id: str) -> None:
        with self._task_worker_stage_lock:
            self._task_worker_seen_messages.pop(request_message_id, None)
            try:
                self._task_worker_seen_order.remove(request_message_id)
            except ValueError:
                pass


    def _send_task_worker_response(
        self, request_message_id: str, response: WorkerMessage,
    ) -> None:
        self._cache_task_worker_response(request_message_id, response)
        self._send_task_worker_to_master(response)


    def _replay_task_worker_responses(self, responses: list[dict]) -> None:
        for response in responses:
            self._send_task_worker_to_master(
                decode_task_worker_message(response)
            )


    @staticmethod
    def _task_worker_active_identity_matches(
        payload: dict, active: _TaskWorkerActiveAttempt,
    ) -> bool:
        return all((
            payload.get("workflow_id") == active.workflow_id,
            payload.get("stage_id") == active.stage_id,
            payload.get("attempt_id") == active.attempt_id,
            payload.get("lease_id") == active.lease_id,
            payload.get("lease_epoch") == active.lease_epoch,
        ))


    @staticmethod
    def _task_worker_next_boundary_monotonic(
        active: _TaskWorkerActiveAttempt,
    ) -> float:
        request_deadline = active.request_deadline_monotonic
        if request_deadline is None:
            return active.lease_deadline_monotonic
        return min(active.lease_deadline_monotonic, request_deadline)


    @staticmethod
    def _task_worker_decide_stop_locked(
        active: _TaskWorkerActiveAttempt,
        *,
        now: Optional[float] = None,
    ) -> str:
        """Latch the first observed attempt stop without merging its causes."""
        if active.terminal_reason:
            return active.terminal_reason

        observed_at = time.monotonic() if now is None else now
        due_boundaries = []
        request_deadline = active.request_deadline_monotonic
        if (
            request_deadline is not None
            and request_deadline <= observed_at
        ):
            # Request timeout is the immutable workflow boundary, while the
            # lease only fences ownership of this one attempt.  An unlatching
            # disconnect observation must not overwrite either boundary.
            due_boundaries.append((
                request_deadline, 0, _REQUEST_DEADLINE_EXCEEDED,
            ))
        if active.lease_deadline_monotonic <= observed_at:
            due_boundaries.append((
                active.lease_deadline_monotonic, 1, "lease_expired",
            ))
        if due_boundaries:
            reason = min(due_boundaries)[2]
        elif active.lease_expired:
            reason = "lease_expired"
        elif active.cancel_reason:
            reason = active.cancel_reason
        elif active.cancel_event.is_set():
            reason = "provider_cancelled"
        else:
            return ""

        active.terminal_reason = reason
        active.terminal_decided_at_monotonic = observed_at
        if reason == "lease_expired":
            active.lease_expired = True
        active.cancel_event.set()
        return reason


    def _acquire_task_worker_execution_lock(
        self,
        execution_lock,
        active: _TaskWorkerActiveAttempt,
    ) -> str:
        """Acquire one model lock in bounded slices or return the stop reason."""
        while True:
            with self._task_worker_stage_lock:
                current = self._task_worker_active_attempts.get(
                    active.attempt_id,
                )
                if current is not active:
                    if not active.terminal_reason:
                        active.terminal_reason = "provider_cancelled"
                        active.terminal_decided_at_monotonic = time.monotonic()
                        active.cancel_event.set()
                    return active.terminal_reason
                now = time.monotonic()
                stop_reason = self._task_worker_decide_stop_locked(
                    active, now=now,
                )
                if stop_reason:
                    return stop_reason
                remaining = (
                    self._task_worker_next_boundary_monotonic(active) - now
                )
                wait_seconds = min(
                    _TASK_WORKER_LOCK_WAIT_SLICE_SECONDS,
                    max(0.001, remaining),
                )

            if not execution_lock.acquire(timeout=wait_seconds):
                continue

            with self._task_worker_stage_lock:
                stop_reason = self._task_worker_decide_stop_locked(active)
            if stop_reason:
                execution_lock.release()
                return stop_reason
            return ""


    def _watch_task_worker_lease(self, attempt_id: str) -> None:
        while True:
            with self._task_worker_stage_lock:
                active = self._task_worker_active_attempts.get(attempt_id)
                if active is None:
                    return
                now = time.monotonic()
                if self._task_worker_decide_stop_locked(active, now=now):
                    return
                remaining = (
                    self._task_worker_next_boundary_monotonic(active) - now
                )
                done_event = active.done_event
            if done_event.wait(min(
                _TASK_WORKER_LOCK_WAIT_SLICE_SECONDS,
                max(0.001, remaining),
            )):
                return


    def _handle_task_worker_lease_renew(
        self, message: WorkerMessage,
    ) -> None:
        payload = message.payload
        attempt_id = str(payload["attempt_id"])
        with self._task_worker_stage_lock:
            active = self._task_worker_active_attempts.get(attempt_id)
            if active is None:
                raise WorkerProtocolError(
                    "lease renewal has no active attempt",
                    code="unknown_attempt",
                    field="payload.attempt_id",
                )
            if not self._task_worker_active_identity_matches(payload, active):
                raise WorkerProtocolError(
                    "lease renewal identity does not match the active attempt",
                    code="attempt_identity_mismatch",
                    field="payload",
                )
            stop_reason = self._task_worker_decide_stop_locked(active)
            if stop_reason:
                raise WorkerProtocolError(
                    "a stopped Stage attempt cannot be renewed",
                    code=stop_reason,
                    field="payload.lease_expires_at_ms",
                )
            deadline = int(payload["lease_expires_at_ms"])
            if deadline <= active.lease_expires_at_ms:
                raise WorkerProtocolError(
                    "lease renewal must extend the active deadline",
                    code="stale_lease",
                    field="payload.lease_expires_at_ms",
                )
            active.lease_expires_at_ms = deadline
            active.lease_deadline_monotonic = time.monotonic() + (
                deadline - message.sent_at_ms
            ) / 1000.0


    def _handle_task_worker_stage_cancel(
        self, message: WorkerMessage,
    ) -> None:
        payload = message.payload
        attempt_id = str(payload["attempt_id"])
        with self._task_worker_stage_lock:
            active = self._task_worker_active_attempts.get(attempt_id)
            if active is None:
                now = time.monotonic()
                completed = getattr(
                    self, "_task_worker_completed_attempts", {},
                )
                for completed_id, (_record, expires_at) in list(completed.items()):
                    if expires_at <= now:
                        completed.pop(completed_id, None)
                tombstone = completed.get(attempt_id)
                if tombstone is None:
                    raise WorkerProtocolError(
                        "Stage cancellation has no active attempt",
                        code="unknown_attempt",
                        field="payload.attempt_id",
                    )
                active = tombstone[0]
            if not self._task_worker_active_identity_matches(payload, active):
                raise WorkerProtocolError(
                    "Stage cancellation identity does not match the active attempt",
                    code="attempt_identity_mismatch",
                    field="payload",
                )
            terminal_reason = self._task_worker_decide_stop_locked(active)
            active.cancel_reason = active.cancel_reason or str(
                payload["reason_code"]
            )
            if not terminal_reason:
                terminal_reason = active.cancel_reason
                active.terminal_reason = terminal_reason
                active.terminal_decided_at_monotonic = time.monotonic()
                active.cancel_event.set()
            provider_id = active.provider_id
        response_payload = self._task_worker_attempt_payload(
            payload, provider_id=provider_id,
        )
        response_payload["reason_code"] = terminal_reason
        response = build_task_worker_message(
            "stage_cancelled",
            response_payload,
            message_id=f"msg_cancelled_{uuid.uuid4().hex}",
            sent_at_ms=int(time.time() * 1000),
            version=TASK_WORKER_PROTOCOL_VERSION,
        )
        self._send_task_worker_response(message.message_id, response)


    # ------------------------------------------------------------------
    # ★ 2026-10-07（DIST-NEXT-2b）：大 payload 分片（`stage_chunk`）
    # ------------------------------------------------------------------

    def _task_worker_chunk_assembler(self):
        """惰性创建分片装配器（进程内一份；按 attempt_id 分区）。"""
        assembler = getattr(self, "_task_worker_chunk_state", None)
        if assembler is None:
            from task_worker_chunks import StageChunkAssembler

            assembler = StageChunkAssembler()
            self._task_worker_chunk_state = assembler
        return assembler

    def _handle_task_worker_stage_chunk(self, message: WorkerMessage) -> None:
        """接收一条 `stage_chunk`（fail-closed：重复/漂移/摘要不符/超限一律拒）。"""
        import base64 as _b64

        payload = message.payload
        try:
            chunk = _b64.b64decode(str(payload["payload_b64"]), validate=True)
        except Exception as exc:
            raise WorkerProtocolError(
                "stage_chunk payload is not valid base64",
                code="invalid_chunk_payload",
                field="payload.payload_b64",
            ) from exc
        self._task_worker_chunk_assembler().add(
            attempt_id=str(payload["attempt_id"]),
            chunk_index=int(payload["chunk_index"]),
            chunk_count=int(payload["chunk_count"]),
            payload=chunk,
            payload_sha256=str(payload["payload_sha256"]),
        )

    def _assemble_stage_root_input(self, offer: dict) -> dict:
        """★ DIST-NEXT-2b：把 `hidden_ref` 的分片装配回 `hidden_f32`（fail-closed）。

        内联路径（offer 直接带 `hidden_f32`）**原样返回** —— 与接线前逐字节一致；
        分片路径则要求：分片齐备、`total_bytes` 与 spec 自洽、装配后摘要等于
        `hidden_sha256`。任何一条不成立就以具名原因失败，不做静默降级。
        """
        root_input = offer.get("root_input")
        if not isinstance(root_input, dict):
            raise RuntimeError("层段 Stage 的 root_input 必须是对象")
        # 非层段（`full_inference` / `aggregate`）没有 hidden 交接，原样透传。
        if str(offer.get("stage_type") or "") != "layer_forward":
            return root_input
        if isinstance(root_input.get("hidden_f32"), str):
            return root_input
        ref = root_input.get("hidden_ref")
        if not isinstance(ref, dict):
            raise RuntimeError("层段 Stage 缺少 root_input.hidden_f32")

        import base64 as _b64
        import hashlib as _hashlib

        assembler = self._task_worker_chunk_assembler()
        attempt_id = str(offer["attempt_id"])
        try:
            raw = assembler.assemble(attempt_id)
        except WorkerProtocolError as exc:
            raise RuntimeError(f"层段 Stage 的分片未齐备: {exc}") from exc
        declared = str(offer.get("hidden_sha256") or "")
        actual = _hashlib.sha256(raw).hexdigest()
        if declared and actual != declared:
            raise RuntimeError(
                f"层段 Stage 装配后的 hidden 摘要不符: {actual} != {declared}"
            )
        assembler.discard(attempt_id)
        # 改写后**移除** `hidden_ref`：执行侧只应看到内联 `hidden_f32`（避免两个来源并存）。
        assembled = {
            key: value for key, value in root_input.items() if key != "hidden_ref"
        }
        assembled["hidden_f32"] = _b64.b64encode(raw).decode("ascii")
        return assembled

    @staticmethod
    def _task_worker_attempt_payload(
        offer_payload: dict,
        *,
        provider_id: str,
    ) -> dict:
        return {
            "workflow_id": offer_payload["workflow_id"],
            "stage_id": offer_payload["stage_id"],
            "attempt_id": offer_payload["attempt_id"],
            "lease_id": offer_payload["lease_id"],
            "lease_epoch": offer_payload["lease_epoch"],
            "provider_id": provider_id,
        }


    def _send_task_worker_stage_accept(
        self,
        request_message_id: str,
        offer_payload: dict,
        *,
        accepted: bool,
        reason_code: str = "",
        retryable: bool = False,
    ) -> None:
        payload = self._task_worker_attempt_payload(
            offer_payload,
            provider_id=str(offer_payload.get("provider_id", "")),
        )
        payload.update({
            "accepted": bool(accepted),
            "reason_code": "" if accepted else reason_code,
            "retryable": False if accepted else bool(retryable),
        })
        response = build_task_worker_message(
            "stage_accept",
            payload,
            message_id=f"msg_accept_{uuid.uuid4().hex}",
            sent_at_ms=int(time.time() * 1000),
            version=TASK_WORKER_PROTOCOL_VERSION,
        )
        self._send_task_worker_response(request_message_id, response)


    def _handle_task_worker_stage_offer(
        self, message: WorkerMessage,
    ) -> None:
        """Validate and execute one explicit remote Stage on a PC worker."""
        offer = message.payload
        attempt_id = str(offer["attempt_id"])
        expected_provider = remote_provider_id(self.get_effective_node_id())
        reject_reason = ""
        reject_retryable = False
        active: Optional[_TaskWorkerActiveAttempt] = None

        coordinator = self._task_worker_control.coordinator_snapshot()
        if not self._scheduler_facade_global('TASK_WORKER_EXPERIMENTAL_ENABLED'):
            reject_reason = "worker_experiment_disabled"
            reject_retryable = True
        else:
            # Layer workers are admitted through the v3 layer capability gate;
            # full-model stages keep the legacy manual dispatch gate.  Older
            # coordinators do not publish the layer flag, so retain the
            # connected/manual fallback for wire compatibility.
            # ★ 2026-10-03：层段的准入判据必须来自**本机自己的** capabilities。
            #   两个坑：① 原来的 `coordinator["layer_stage_dispatch_enabled"]` 描述的是
            #   "别的 worker 里有没有层段就绪的"（见 `get_task_worker_protocol_status` 里
            #   `layer_stage_worker_ids` 的构造）⇒ 拿它判自己会把本机的合法 offer 拒成
            #   `worker_not_admitted`；② 想把"自己的判据"塞进 `coordinator_snapshot()`
            #   也走不通 —— 那返回的是**协调者**的快照，不是本机的状态。所以直接在本地算，
            #   口径与 master 侧 `task_worker_adapter.py:359` 一致。
            if offer.get("stage_type") == "layer_forward":
                capabilities = self._task_worker_capabilities()
                dispatch_enabled = bool(
                    capabilities.get("layer_ranges")
                    and "layer_forward" in (capabilities.get("stage_types") or [])
                )
            else:
                dispatch_enabled = coordinator.get("manual_stage_dispatch_enabled")
            if not dispatch_enabled:
                reject_reason = "worker_not_admitted"
                reject_retryable = True
        if not reject_reason and offer["provider_id"] != expected_provider:
            reject_reason = "provider_identity_mismatch"
        elif not reject_reason:
            capabilities = self._task_worker_capabilities()
            if offer["stage_type"] not in capabilities["stage_types"]:
                reject_reason = "unsupported_stage_type"
                reject_retryable = True
            elif offer["model_identity"] not in capabilities["models"]:
                reject_reason = "model_identity_mismatch"
                reject_retryable = True

        with self._task_worker_stage_lock:
            if not reject_reason and (
                attempt_id in self._task_worker_active_attempts
                or self._task_worker_active_attempts
            ):
                reject_reason = "remote_worker_busy"
                reject_retryable = True
            if not reject_reason:
                received_at_monotonic = time.monotonic()
                received_at_epoch = time.time()
                lease_expires_at_ms = int(offer["lease_expires_at_ms"])
                request_deadline_value = offer.get("request_deadline_ms")
                request_deadline_ms = (
                    int(request_deadline_value)
                    if request_deadline_value is not None else None
                )
                active = _TaskWorkerActiveAttempt(
                    workflow_id=str(offer["workflow_id"]),
                    stage_id=str(offer["stage_id"]),
                    attempt_id=attempt_id,
                    lease_id=str(offer["lease_id"]),
                    lease_epoch=int(offer["lease_epoch"]),
                    provider_id=expected_provider,
                    lease_expires_at_ms=lease_expires_at_ms,
                    lease_deadline_monotonic=received_at_monotonic + (
                        lease_expires_at_ms - message.sent_at_ms
                    ) / 1000.0,
                    request_deadline_ms=request_deadline_ms,
                    request_deadline_monotonic=(
                        received_at_monotonic + (
                            request_deadline_ms / 1000.0 - received_at_epoch
                        )
                        if request_deadline_ms is not None else None
                    ),
                )
                self._task_worker_active_attempts[attempt_id] = active

        if reject_reason:
            try:
                self._send_task_worker_stage_accept(
                    message.message_id,
                    offer,
                    accepted=False,
                    reason_code=reject_reason,
                    retryable=reject_retryable,
                )
            except Exception:
                self._forget_task_worker_request(message.message_id)
                logger.warning(
                    "event=task_worker_stage_reject_send_failed attempt_id=%s",
                    attempt_id, exc_info=True,
                )
            return
        if active is None:
            raise RuntimeError("accepted task-worker Stage has no active record")

        try:
            self._send_task_worker_stage_accept(
                message.message_id, offer, accepted=True,
            )
        except Exception:
            with self._task_worker_stage_lock:
                removed = self._task_worker_active_attempts.pop(
                    attempt_id, None,
                )
                if removed is not None:
                    removed.done_event.set()
            self._forget_task_worker_request(message.message_id)
            logger.warning(
                "event=task_worker_stage_accept_send_failed attempt_id=%s",
                attempt_id, exc_info=True,
            )
            return

        threading.Thread(
            target=self._watch_task_worker_lease,
            args=(attempt_id,),
            name=f"task-worker-lease-{attempt_id}",
            daemon=True,
        ).start()

        try:
            model_identity = TaskModelIdentity(**offer["model_identity"])
            # ★ 2026-10-07（DIST-NEXT-2b）：上游把超帧预算的 hidden 分片发送时，offer 的
            #   `root_input` 只带 `hidden_ref`（无内联 `hidden_f32`）⇒ 在这里装配并回填，
            #   使执行侧（`inference_service.engine_host`）无需感知分片。
            root_input = self._assemble_stage_root_input(offer)
            request = TaskProviderStageRequest(
                workflow_id=offer["workflow_id"],
                request_id=offer["request_id"],
                stage_id=offer["stage_id"],
                stage_type=offer["stage_type"],
                provider_id=offer["provider_id"],
                dependencies=offer["dependencies"],
                root_input=root_input,
                model_identity=model_identity,
                stage_fields={
                    key: offer[key]
                    for key in (
                        "layer_range", "handoff_at", "hidden_sha256",
                        "hidden_spec", "middle_channel", "seq_ids", "positions",
                    )
                    if key in offer
                },
            )
            acquired_full_chat_lock = False
            acquired_inference_lock = False
            try:
                stop_reason = self._acquire_task_worker_execution_lock(
                    self._host.full_chat_execution_lock, active,
                )
                if stop_reason:
                    raise RuntimeError(stop_reason)
                acquired_full_chat_lock = True

                stop_reason = self._acquire_task_worker_execution_lock(
                    self._inference_lock, active,
                )
                if stop_reason:
                    raise RuntimeError(stop_reason)
                acquired_inference_lock = True

                output = self._require_callbacks().execute_task_worker_stage(
                    request, active.cancel_event,
                )
            finally:
                if acquired_inference_lock:
                    self._inference_lock.release()
                if acquired_full_chat_lock:
                    self._host.full_chat_execution_lock.release()
            if not isinstance(output, dict):
                raise RuntimeError("remote Stage executor returned non-object output")
            with self._task_worker_stage_lock:
                current = self._task_worker_active_attempts.get(attempt_id)
                if current is None:
                    raise RuntimeError("remote Stage attempt is no longer active")
                stop_reason = self._task_worker_decide_stop_locked(current)
                if stop_reason:
                    raise RuntimeError(stop_reason)
            result_payload = self._task_worker_attempt_payload(
                offer, provider_id=expected_provider,
            )
            result_payload.update({
                "output": output,
                "output_sha256": task_worker_sha256(output),
                "metadata": sanitize_task_result_metadata(output),
            })
            responses = self._task_worker_stage_result_messages(
                result_payload,
                offer=offer,
                attempt_id=attempt_id,
                provider_id=expected_provider,
            )
            for response in responses:
                try:
                    self._send_task_worker_response(message.message_id, response)
                except Exception:
                    logger.warning(
                        "event=task_worker_stage_result_send_failed attempt_id=%s",
                        attempt_id, exc_info=True,
                    )
                    return
            logger.info(
                "event=task_worker_stage_result_sent workflow_id=%s stage_id=%s attempt_id=%s chunks=%d",
                offer["workflow_id"], offer["stage_id"], attempt_id,
                max(0, len(responses) - 1),
            )
        except Exception as exc:
            with self._task_worker_stage_lock:
                current = self._task_worker_active_attempts.get(attempt_id)
                terminal_reason = (
                    self._task_worker_decide_stop_locked(current)
                    if current is not None else active.terminal_reason
                )
                cancelled_by_coordinator = bool(
                    current is not None and current.cancel_reason
                )
            if cancelled_by_coordinator and terminal_reason != "lease_expired":
                logger.info(
                    "event=task_worker_stage_cancelled workflow_id=%s stage_id=%s attempt_id=%s",
                    offer["workflow_id"], offer["stage_id"], attempt_id,
                )
                return
            error_code = (
                terminal_reason
                if terminal_reason in {
                    "lease_expired", _REQUEST_DEADLINE_EXCEEDED,
                }
                else "provider_cancelled"
                if active.cancel_event.is_set()
                else "remote_stage_execution_failed"
            )
            error_payload = self._task_worker_attempt_payload(
                offer, provider_id=expected_provider,
            )
            # ★ 2026-10-09（稳定性 #73）：把异常文本**带出去**。此前 `stage_error` 只传
            #   `error_code`/`retryable`，master 侧一律显示「remote worker reported a Stage error」
            #   ⇒ 真因（例：`ValueError: hidden 形状应为 [n_tokens, 2048]，实际 (33, 896)`）
            #   只留在 worker 本地日志里，用户端完全看不到、无从修复。
            #   新增 `reason` 字段是**向后兼容**的：老 master 只读 `error_code`/`retryable`。
            error_payload.update({
                "error_code": error_code,
                "retryable": error_code == "lease_expired",
                "reason": f"{type(exc).__name__}: {exc}"[:200],
            })
            try:
                response = build_task_worker_message(
                    "stage_error",
                    error_payload,
                    message_id=f"msg_error_{uuid.uuid4().hex}",
                    sent_at_ms=int(time.time() * 1000),
                    version=TASK_WORKER_PROTOCOL_VERSION,
                )
                self._send_task_worker_response(
                    message.message_id, response,
                )
            except Exception:
                logger.warning(
                    "event=task_worker_stage_error_send_failed attempt_id=%s",
                    attempt_id, exc_info=True,
                )
            logger.warning(
                "event=task_worker_stage_execution_failed workflow_id=%s stage_id=%s attempt_id=%s reason=%s",
                offer["workflow_id"], offer["stage_id"], attempt_id,
                type(exc).__name__, exc_info=True,
            )
        finally:
            with self._task_worker_stage_lock:
                removed = self._task_worker_active_attempts.pop(
                    attempt_id, None,
                )
                if removed is not None:
                    removed.done_event.set()
                    completed = getattr(
                        self, "_task_worker_completed_attempts", None,
                    )
                    if completed is None:
                        completed = {}
                        self._task_worker_completed_attempts = completed
                    now = time.monotonic()
                    for completed_id, (_record, expires_at) in list(completed.items()):
                        if expires_at <= now:
                            completed.pop(completed_id, None)
                    completed[attempt_id] = (removed, now + 5.0)


    def _handle_task_worker_message(self, client_id: str, msg: dict) -> None:
        """Route N2.2 hello, Stage, renewal, and cancellation messages."""
        raw = msg.get("data")
        if not isinstance(raw, dict):
            self._task_worker_control.record_rejection()
            logger.warning(
                "event=task_worker_message_rejected peer=%s reason=invalid_outer_payload",
                client_id,
            )
            return
        try:
            message = decode_task_worker_message(raw)
            if self._effective_role() == "master":
                if client_id == "master":
                    raise WorkerProtocolError(
                        "master cannot register as its own task worker",
                        code="invalid_message_direction",
                        field="message_type",
                    )
                worker_kind = (
                    message.payload.get("worker_kind")
                    if message.message_type == "hello"
                    else ""
                )
                with self._nodes_lock:
                    registered_node = self.nodes.get(client_id)
                    registered_role = getattr(
                        getattr(registered_node, "role", ""),
                        "value",
                        getattr(registered_node, "role", ""),
                    )
                    admitted_worker_node = bool(
                        registered_node is not None
                        and registered_node.node_type in {"pc", "android"}
                        and registered_role == NodeRole.CLIENT.value
                    )
                if not admitted_worker_node:
                    self._task_worker_control.record_rejection()
                    raise WorkerProtocolError(
                        "only a registered PC or Android client may negotiate a full worker",
                        code="unsupported_worker_node",
                        field="payload.worker_kind",
                    )
                if message.message_type == "hello":
                    expected_kind = (
                        "android_full_worker"
                        if registered_node.node_type == "android"
                        else "pc_full_worker"
                    )
                    if worker_kind != expected_kind:
                        self._task_worker_control.record_rejection()
                        raise WorkerProtocolError(
                            "worker kind does not match the registered node type",
                            code="worker_kind_node_type_mismatch",
                            field="payload.worker_kind",
                        )
                if message.message_type == "hello":
                    # ★ 2026-10-05（DIST-3）：把 hello 校验的异常面显式记下来。
                    #   此前这里异常直接向上抛，而更外层（TCP 接收循环）会吞掉它
                    #   ⇒ worker 侧只看到"连上又断开"，master 侧**一行日志都没有**，
                    #   排查只能靠猜（实测：Y700 每 30s 重连一次、只知道
                    #   `task_worker_handshake_pending`，定位花了好几轮）。
                    try:
                        ack = self._task_worker_control.receive_on_coordinator(
                            client_id,
                            raw,
                            coordinator_node_id=self.get_effective_node_id(),
                        )
                    except Exception:
                        logger.error(
                            "task worker hello 校验失败: node=%s", client_id,
                            exc_info=True,
                        )
                        raise
                    self._send_task_worker_to_node(client_id, ack)
                    # The registration fence must end for both an accepted
                    # hello and a definitive rejection.  A rejected hello is
                    # no longer negotiating the task-worker path, so legacy
                    # scheduling can be recomputed for that connection.
                    self._task_worker_control.resolve_worker_connection_pending(client_id)
                    # ★ 幂等 hello（#28）：**只有 capabilities 真的变了**才重推层配置。
                    #   此前无条件 push，配合层段路径补 refresh 会形成
                    #   `hello → push → load_layer_range → refresh → hello` 自激环（每次 push
                    #   都取新 generation ⇒ worker 端永远判成"新配置"）。
                    #   字段缺失时按"变了"处理（保守：多重推一次总好过永远不推）。
                    worker_snapshot = self._task_worker_control.worker_snapshot(client_id)
                    # Recovery must use one authoritative publish. A normal
                    # capability push here would create a second generation.
                    recovery_sync = bool(
                        ack.payload["accepted"] and self._pipeline_recovery_pending
                    )
                    if not recovery_sync and worker_snapshot.get(
                        "capabilities_changed", True
                    ):
                        self.push_layer_config_to_clients()
                    # ★ 2026-10-05（DIST-1 三机重启实测）：权威重发此前**只在请求路径**
                    #   触发，而请求会被恢复闸门自身拒绝 ⇒ 没有任何路径去产生「新代际」
                    #   ⇒ 闸门永不解除（实测：三机全部在线、TCP 已重连、hello
                    #   accepted=True，请求仍返回 `pipeline_recovery_pending`；
                    #   readiness 停在 `layer_status=not_configured`，且准入名单在
                    #   同一次会话内由非空变为空）。
                    #   worker 重新 hello 是「这个节点回来了」的权威信号，恢复期就在
                    #   此刻补一次**权威重发**：`require_distributed=True` 的语义是
                    #   「按当前在线能力重算一份新计划」，而不是重放持久化的旧计划
                    #   （见 `push_layer_config_to_clients_locked` 里对
                    #   `_pipeline_recovery_pending` 的处理）。
                    if recovery_sync:
                        logger.info(
                            "重启恢复期收到 worker hello，触发权威重发: node=%s",
                            client_id,
                        )
                        self.request_authoritative_layer_sync(require_distributed=True)
                    if ack.payload["accepted"]:
                        self._ensure_remote_task_worker_provider(client_id)
                        # A node that has just advertised a complete model is
                        # a Full Worker, not a layer partition.  Registration
                        # may have pushed a layer config before the hello
                        # arrived; release that reservation so the two roles
                        # cannot race and make the Worker reject its Stage.
                        advertised_models = (
                            message.payload.get("capabilities", {}).get("models", [])
                            if isinstance(message.payload, dict)
                            else []
                        )
                        advertised_capabilities = (
                            message.payload.get("capabilities", {})
                            if isinstance(message.payload, dict)
                            else {}
                        )
                        is_layer_stage_worker = bool(
                            isinstance(advertised_capabilities, dict)
                            and "layer_forward" in advertised_capabilities.get(
                                "stage_types", []
                            )
                            and advertised_capabilities.get("layer_ranges")
                        )
                        # Layer/relay workers deliberately advertise no full
                        # model identity.  Do not turn their hello into an
                        # opt-out: the layer-config handshake is their role.
                        # ★ relay 宿主是第三类角色，判据唯一的出处在 `_is_relay_host()`：
                        #   它不能声明 `forward_layers`（声明了就会拒绝 legacy 层配置，而
                        #   relay 委派正是走那条通道），但它要的恰恰就是那份 legacy 配置，
                        #   因此不能被当 Full Worker 释放预留。
                        if (advertised_models
                                and not self._is_relay_host(client_id)
                                and not is_layer_stage_worker
                                and not bool(
                                    advertised_capabilities.get(
                                        "layer_worker", False
                                    )
                                )):
                            self._handle_layer_worker_opt_out(
                                client_id,
                                {"data": {"node_id": client_id}},
                            )
                    logger.info(
                        "event=task_worker_hello_acked node_id=%s accepted=%s version=%s",
                        client_id,
                        ack.payload["accepted"],
                        ack.payload["selected_version"],
                    )
                    # ★ 2026-10-07（DIST-NEXT-6）：把「哪些工件被判不可用、为什么」打到
                    #   master 日志 —— 否则节点被静默剔除时这里只剩
                    #   `layer_range_not_advertised`，看不出是缺文件、摘要不符还是架构不符。
                    unusable_artifacts = (
                        advertised_capabilities.get("layer_artifact_diagnostics")
                        if isinstance(advertised_capabilities, dict) else None
                    ) or []
                    if unusable_artifacts:
                        logger.warning(
                            "event=task_worker_layer_artifact_unusable node_id=%s "
                            "count=%d details=%s",
                            client_id,
                            len(unusable_artifacts),
                            [
                                {
                                    "error_code": entry.get("error_code"),
                                    "manifest": entry.get("manifest"),
                                    "layer_range": entry.get("layer_range"),
                                    "mode": entry.get("mode"),
                                    "architecture": entry.get("architecture"),
                                }
                                for entry in unusable_artifacts
                                if isinstance(entry, dict)
                            ],
                        )
                elif message.message_type in {
                    "stage_accept", "stage_result", "stage_error",
                    "stage_cancelled",
                }:
                    provider = self._remote_task_worker_providers.get(client_id)
                    if provider is None:
                        raise WorkerProtocolError(
                            "Stage response arrived before an accepted worker hello",
                            code="worker_not_admitted",
                            field="message_type",
                        )
                    provider.handle_message(raw)
                else:
                    raise WorkerProtocolError(
                        "message is not valid in the N2.2 coordinator direction",
                        code="invalid_message_direction",
                        field="message_type",
                    )
            else:
                if client_id != "master":
                    raise WorkerProtocolError(
                        "worker accepts task-worker control messages from master only",
                        code="invalid_message_direction",
                        field="message_type",
                    )
                if message.message_type == "hello_ack":
                    accepted = self._task_worker_control.receive_on_worker(raw)
                    logger.info(
                        "event=task_worker_hello_ack_received coordinator=%s accepted=%s version=%s",
                        accepted.payload["coordinator_node_id"],
                        accepted.payload["accepted"],
                        accepted.payload["selected_version"],
                    )
                    with self._task_worker_refresh_lock:
                        refresh_requested = self._task_worker_refresh_requested
                    if refresh_requested:
                        self.refresh_task_worker_capabilities()
                elif message.message_type == "stage_chunk":
                    # ★ 2026-10-07（DIST-NEXT-2b）：大 payload 分片 —— 只累积，装配在
                    #   随后的 `stage_offer` 路径完成（收到 offer 时分片应当已齐备）。
                    self._handle_task_worker_stage_chunk(message)
                elif message.message_type == "stage_offer":
                    duplicate, responses = self._prepare_task_worker_request(
                        message
                    )
                    if duplicate:
                        self._replay_task_worker_responses(responses)
                        return
                    threading.Thread(
                        target=self._handle_task_worker_stage_offer,
                        args=(message,),
                        name=f"task-worker-stage-{message.payload['attempt_id']}",
                        daemon=True,
                    ).start()
                elif message.message_type in {"lease_renew", "stage_cancel"}:
                    duplicate, responses = self._prepare_task_worker_request(
                        message
                    )
                    if duplicate:
                        self._replay_task_worker_responses(responses)
                        return
                    if message.message_type == "lease_renew":
                        self._handle_task_worker_lease_renew(message)
                    else:
                        self._handle_task_worker_stage_cancel(message)
                else:
                    raise WorkerProtocolError(
                        "message is not valid in the N2.2 worker direction",
                        code="invalid_message_direction",
                        field="message_type",
                    )
        except (WorkerProtocolError, ConnectionError) as exc:
            logger.warning(
                "event=task_worker_message_rejected peer=%s reason=%s",
                client_id,
                getattr(exc, "code", type(exc).__name__),
                exc_info=True,
            )
        except Exception as exc:
            # A malformed control-plane message must not collapse the shared TCP
            # receive loop or any existing inference path.
            logger.warning(
                "event=task_worker_message_failed peer=%s reason=%s",
                client_id, type(exc).__name__, exc_info=True,
            )


    def get_task_worker_protocol_status(self) -> dict:
        role = self._effective_role()
        runtime = self._task_worker_control.status(role=role)
        connected = bool(runtime.get("control_plane_connected", False))
        if role == "master":
            healthy_workers = [
                worker for worker in runtime.get("workers", [])
                if isinstance(worker, dict) and worker.get("healthy")
            ]
            full_model_worker_ids = sorted(
                str(worker.get("node_id", ""))
                for worker in healthy_workers
                if isinstance(worker.get("capabilities"), dict)
                and bool(worker["capabilities"].get("models"))
                and not bool(worker["capabilities"].get("layer_worker"))
                and (
                    worker.get("worker_kind") != "android_full_worker"
                    or (
                        isinstance(worker["capabilities"].get("resource_gate"), dict)
                        and worker["capabilities"]["resource_gate"].get("admitted") is True
                        and not worker["capabilities"]["resource_gate"].get("reason_code")
                    )
                )
            )
            workers_missing_full_model = sorted(
                str(worker.get("node_id", ""))
                for worker in healthy_workers
                if str(worker.get("node_id", "")) not in full_model_worker_ids
            )
            layer_stage_worker_ids = sorted(
                str(worker.get("node_id", ""))
                for worker in healthy_workers
                if worker.get("layer_stage_dispatch_enabled")
            )
        else:
            local_models = self._task_worker_capabilities().get("models", [])
            full_model_worker_ids = (
                [self.get_effective_node_id()] if connected and local_models else []
            )
            workers_missing_full_model = (
                [self.get_effective_node_id()] if connected and not local_models else []
            )
            local_caps = self._task_worker_capabilities()
            layer_stage_worker_ids = (
                [self.get_effective_node_id()]
                if connected
                and "layer_forward" in local_caps.get("stage_types", [])
                and local_caps.get("layer_ranges")
                else []
            )
        full_model_ready = bool(full_model_worker_ids)
        layer_stage_ready = bool(layer_stage_worker_ids)
        if not self._scheduler_facade_global('TASK_WORKER_EXPERIMENTAL_ENABLED'):
            readiness_reason = "task_worker_experiment_disabled"
        elif not connected:
            readiness_reason = "task_worker_control_plane_not_connected"
        elif not full_model_ready:
            readiness_reason = "task_worker_full_model_not_advertised"
        else:
            readiness_reason = "task_worker_manual_dispatch_ready"
        runtime.update({
            "phase": "TC-N2.4",
            "experiment_enabled": self._scheduler_facade_global('TASK_WORKER_EXPERIMENTAL_ENABLED'),
            "experimental_dispatch_enabled": bool(
                self._scheduler_facade_global('TASK_WORKER_EXPERIMENTAL_ENABLED') and connected and full_model_ready
            ),
            "auto_provider_selection_enabled": bool(
                self._scheduler_facade_global('TASK_WORKER_EXPERIMENTAL_ENABLED') and connected and full_model_ready
            ),
            "manual_stage_dispatch_enabled": bool(
                self._scheduler_facade_global('TASK_WORKER_EXPERIMENTAL_ENABLED') and connected and full_model_ready
            ),
            "layer_stage_dispatch_enabled": bool(
                self._scheduler_facade_global('TASK_WORKER_EXPERIMENTAL_ENABLED')
                and connected and layer_stage_ready
            ),
            # ★ 2026-10-03：上面那个字段的判据是 `layer_stage_worker_ids` —— 那是
            #   **别的** worker 的 id 集合（用于向调度侧汇报"当前有谁在承层段"）。
            #   而 worker 在 `_handle_task_worker_stage_offer` 里判"我是否被允许接
            #   层段 Stage"时读的也是它 ⇒ 语义错位：本机明明声明了 layer_forward +
            #   layer_ranges，却因为**别的**节点没就绪而把自己的 offer 拒成
            #   `worker_not_admitted`（实测：三段链的 Surface 因此一直拒收）。
            #   这里按**自己的** capabilities 给出同名字段，口径与 master 侧
            #   `task_worker_adapter.py:359` 一致。
            "self_layer_stage_dispatch_enabled": bool(
                self._scheduler_facade_global('TASK_WORKER_EXPERIMENTAL_ENABLED')
                and connected
                and "layer_forward" in (
                    self._task_worker_capabilities().get("stage_types") or []
                )
                and self._task_worker_capabilities().get("layer_ranges")
            ),
            "full_model_worker_count": len(full_model_worker_ids),
            "full_model_worker_ids": full_model_worker_ids,
            "workers_missing_full_model": workers_missing_full_model,
            "layer_stage_worker_count": len(layer_stage_worker_ids),
            "layer_stage_worker_ids": layer_stage_worker_ids,
            "workers_missing_layer_stage": sorted(
                str(worker.get("node_id", ""))
                for worker in healthy_workers
                if str(worker.get("node_id", ""))
                and not worker.get("layer_stage_dispatch_enabled")
            ) if role == "master" else [],
            "worker_readiness_reason": readiness_reason,
            "admission_state": (
                "n2_4_experimental_physical_validation_pending"
                if self._scheduler_facade_global('TASK_WORKER_EXPERIMENTAL_ENABLED') and connected
                else "n2_4_experiment_enabled_not_connected"
                if self._scheduler_facade_global('TASK_WORKER_EXPERIMENTAL_ENABLED')
                else "n2_4_experiment_disabled"
            ),
        })
        return worker_protocol_status(runtime)
