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
    PROTOCOL_VERSION as TASK_WORKER_PROTOCOL_VERSION,
    RUNTIME_PROFILE_UNSPECIFIED,
    WorkerMessage,
    WorkerProtocolError,
    build_message as build_task_worker_message,
    canonical_message_bytes as task_worker_message_bytes,
    canonical_sha256 as task_worker_sha256,
    decode_message as decode_task_worker_message,
    restore_inline_hidden as restore_task_worker_hidden,
    split_inline_hidden as split_task_worker_hidden,
    worker_protocol_status,
)
from torch_runtime import torch_available

logger = logging.getLogger("scheduler")

_LAYER_ARTIFACT_DIGEST_LOCK = threading.Lock()
_LAYER_ARTIFACT_DIGEST_CACHE: dict[str, tuple[tuple[int, int, int], str]] = {}
_TASK_WORKER_LOCK_WAIT_SLICE_SECONDS = 0.05
_REQUEST_DEADLINE_EXCEEDED = "request_deadline_exceeded"


def _task_worker_capability_log_summary(
    capabilities: Mapping[str, Any],
) -> dict[str, Any]:
    """Return a bounded, identity-focused summary of validated capabilities."""
    if not isinstance(capabilities, Mapping):
        capabilities = {}

    artifacts = [
        artifact
        for artifact in capabilities.get("layer_artifacts") or []
        if isinstance(artifact, Mapping)
    ]
    artifacts_by_model_id = {
        artifact.get("model_id"): artifact
        for artifact in artifacts
        if artifact.get("model_id")
    }

    def digest_prefix(value: Any) -> str | None:
        return value[:12] if isinstance(value, str) and value else None

    models = []
    for model in capabilities.get("models") or []:
        if not isinstance(model, Mapping):
            continue
        artifact = artifacts_by_model_id.get(model.get("model_id"))
        source_sha256 = (
            artifact.get("source_model_sha256")
            if artifact is not None else None
        ) or model.get("source_model_sha256") or model.get("sha256")
        tokenizer_sha256 = (
            artifact.get("tokenizer_sha256")
            if artifact is not None else None
        ) or model.get("tokenizer_sha256")
        models.append({
            "model_id": model.get("model_id"),
            "source_model_sha256_prefix": digest_prefix(source_sha256),
            "tokenizer_sha256_prefix": digest_prefix(tokenizer_sha256),
        })

    artifact_summaries = []
    for artifact in artifacts:
        weight_bytes = artifact.get("layer_weight_bytes")
        if not isinstance(weight_bytes, list):
            weight_bytes = []
        artifact_summaries.append({
            "source_model_id": artifact.get("source_model_id"),
            "source_model_sha256_prefix": digest_prefix(
                artifact.get("source_model_sha256")
            ),
            "tokenizer_sha256_prefix": digest_prefix(
                artifact.get("tokenizer_sha256")
            ),
            "artifact_sha256_prefix": digest_prefix(
                artifact.get("artifact_sha256")
            ),
            "layer_range": artifact.get("layer_range"),
            "weight_bytes_count": len(weight_bytes),
            "weight_bytes_all_positive": bool(weight_bytes) and all(
                isinstance(value, int)
                and not isinstance(value, bool)
                and value > 0
                for value in weight_bytes
            ),
        })

    resource_gate = capabilities.get("resource_gate")
    return {
        "stage_types": list(capabilities.get("stage_types") or []),
        "layer_ranges": list(capabilities.get("layer_ranges") or []),
        "models": models,
        "layer_artifacts": artifact_summaries,
        "resource_gate_admitted": (
            resource_gate.get("admitted")
            if isinstance(resource_gate, Mapping) else None
        ),
    }


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


def _layer_weight_bytes_of(artifact_path, local_layers: int) -> list[int]:
    """统计**本段局部**的逐层字节（按 `blk.N.` 前缀聚合 GGUF 张量表）。

    与 `llama_engine.get_pipeline_descriptor` 对工件的算法同源：现场按 ggml 类型算，
    **不用** offset 差值（差值含 32 字节对齐填充 ⇒ 与容量账不符）。

    ★ 2026-10-11（方案 A）：接力不加载整模 ⇒ 主节点只有本段字节，而容量求解需要整模的
    逐层字节 ⇒ 由各 worker 申报自己那一段，主节点按 `layer_range` 平移后拼成整模视图。

    返回**本段局部**口径（长度 = 段层数，下标 0 = 段内第一层）；任一层取不到字节即返回
    空列表（宁可不上报，也不上报一份残缺的容量账）。
    """
    try:
        from relay_segment_info import read_gguf_tensor_bytes
    except ImportError:  # 以包形式导入本模块时（src.scheduler_task_worker）
        try:
            from .relay_segment_info import read_gguf_tensor_bytes
        except ImportError:
            return []
    count = int(local_layers or 0)
    if count <= 0:
        return []
    try:
        tensor_bytes = read_gguf_tensor_bytes(str(artifact_path))
    except Exception:
        return []
    if not tensor_bytes:
        return []
    sizes = [0] * count
    pattern = re.compile(r"^blk\.(\d+)\.")
    for name, size in tensor_bytes.items():
        matched = pattern.match(str(name))
        if not matched:
            continue
        index = int(matched.group(1))
        if 0 <= index < count:
            sizes[index] += int(size)
    if any(value <= 0 for value in sizes):
        return []
    return sizes


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
    def _task_worker_full_model_identity(
        self, engines: list[str] | None = None,
    ) -> dict | None:
        """Return the currently loaded full-model identity, if executable."""
        try:
            has_loaded_model = getattr(self._host, "has_loaded_model", None)
            if callable(has_loaded_model):
                full_model_loaded = bool(has_loaded_model())
            else:
                full_model_loaded = bool(
                    getattr(self._host, "model_loaded", False)
                    and getattr(self._host, "is_loaded", False)
                )
            if (
                not full_model_loaded
                or getattr(self._host, "layer_range", None) is not None
            ):
                return None
            identity = self._require_callbacks().active_task_graph_model_identity()
            if identity is None or (
                engines is not None and identity.engine not in engines
            ):
                return None
            snapshot = {
                "model_id": identity.model_id,
                "engine": identity.engine,
                "format": identity.format,
                "revision": identity.revision,
                "sha256": identity.sha256,
            }
            layer_artifact = self._configured_layer_artifact()
            if (
                layer_artifact is not None
                and snapshot["sha256"] == layer_artifact["sha256"]
            ):
                # A registered segment may look loaded to the ordinary host,
                # but it is not a complete-model execution identity. Digest
                # equality is the physical fact; aliases must not turn it
                # back into a full_inference/aggregate capability.
                return None
            return snapshot
        except Exception:
            logger.warning(
                "failed to build PC full-worker model capability snapshot",
                exc_info=True,
            )
            return None

    @staticmethod
    def _task_worker_layer_artifact_identity(artifact: dict | None) -> dict | None:
        if artifact is None or not artifact.get("sha256"):
            return None
        return {
            "model_id": artifact["model_id"],
            "engine": artifact.get("engine", "llama_cpp"),
            "format": artifact.get("format", "gguf"),
            "revision": artifact["revision"],
            "sha256": artifact["sha256"],
        }

    @staticmethod
    def _task_worker_full_model_identities(
        capabilities: Mapping[str, Any],
    ) -> list[dict]:
        """Return only identities executable by a complete-model stage."""
        stage_types = capabilities.get("stage_types") or []
        if not any(
            stage_type in stage_types
            for stage_type in ("full_inference", "aggregate")
        ):
            return []
        layer_artifact_digests = {
            artifact.get("artifact_sha256")
            for artifact in capabilities.get("layer_artifacts") or []
            if isinstance(artifact, Mapping)
        }
        identities = []
        for model in capabilities.get("models") or []:
            if not isinstance(model, Mapping):
                continue
            if model.get("sha256") in layer_artifact_digests:
                continue
            identities.append(dict(model))
        return identities

    def _task_worker_offer_model_identity_matches(
        self, offer: Mapping[str, Any], capabilities: Mapping[str, Any],
    ) -> bool:
        """Match an offer against the identity valid for its execution path."""
        if offer.get("stage_type") == "layer_forward":
            expected = self._task_worker_layer_artifact_identity(
                self._configured_layer_artifact()
            )
            return expected is not None and offer.get("model_identity") == expected

        offered = offer.get("model_identity")
        if not isinstance(offered, Mapping):
            return False
        return dict(offered) in self._task_worker_full_model_identities(
            capabilities
        )

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

        supply_worker = getattr(self, "_model_supply_worker", None)
        if supply_worker is not None:
            from model_supply_worker import get_selected_model

            selected = get_selected_model(getattr(supply_worker, "root", None))
            if selected is not None and selected["manifest"]["format"] == "safetensors":
                # The selector validates pointer, inventory, bytes and source
                # proof. This is registration, not an engine-ready ACK.
                from pipeline_model_descriptor import PIPELINE_RUNTIME_MODEL_TYPES

                manifest = selected["manifest"]
                start, end = manifest["layer_range"]
                model_type = manifest.get("model_type")
                if not model_type:
                    config_path = Path(selected["directory"]) / "config.json"
                    model_type = json.loads(config_path.read_text(encoding="utf-8")).get("model_type")
                return {
                    "start": start,
                    "end": end,
                    "model_id": "supply-" + manifest["artifact_sha256"][:24],
                    "engine": "pytorch",
                    "format": "safetensors",
                    "runtime_supported": bool(
                        torch_available() and model_type in PIPELINE_RUNTIME_MODEL_TYPES
                    ),
                    "revision": "1",  # qlh.segment_supply.v1
                    "sha256": manifest["artifact_sha256"],
                    "source_model_id": manifest["source_model_id"],
                    "source_model_sha256": manifest["source_model_sha256"],
                    "tokenizer_sha256": manifest["tokenizer_sha256"],
                    "hidden_size": manifest["hidden_size"],
                    "segment_mode": manifest["segment_mode"],
                    # Supply uses absolute indices; v3 uses segment-local ones.
                    "layer_weight_bytes": manifest["layer_weight_bytes"][start:end],
                }
        supplied_path = supply_worker.selected_artifact_path() if supply_worker else ""
        model_path = supplied_path or os.environ.get("QLH_LAYER_GGUF", "").strip()
        if not model_path:
            return None
        artifact_path = Path(model_path)
        if not artifact_path.is_file():
            return None
        manifest_path = artifact_path.with_suffix(".manifest.json")
        if not manifest_path.is_file():
            manifest_path = Path(str(artifact_path) + ".manifest.json")
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
        if contract_complete:
            try:
                from relay_segment_info import read_gguf_tokenizer_sha256
            except ImportError:
                from .relay_segment_info import read_gguf_tokenizer_sha256
            if read_gguf_tokenizer_sha256(artifact_path) != tokenizer_sha256:
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
        # ★ 2026-10-11（方案 A：worker 上报本段逐层字节）：接力不加载整模 ⇒ 主节点只持有
        #   **本段**的逐层字节，而容量求解需要**整模**的逐层字节 ⇒ 各 worker 各报自己那一段，
        #   主节点按 `layer_range` 平移到源模型坐标系后拼出整模视图。
        #   这里是**本段局部**口径（0..段层数-1），与 `llama_engine.get_pipeline_descriptor`
        #   对工件的算法同源（现场按 ggml 类型算，不用 offset 差值）。
        segment_layer_bytes = _layer_weight_bytes_of(
            artifact_path, int(value[1]) - int(value[0])
        )
        if segment_layer_bytes:
            result["layer_weight_bytes"] = segment_layer_bytes
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
        # ⚠️ 不能用"构造一次看是否成功"之外的方式测量：`build_message`/`validate_message`
        #   **在超限时直接抛 `message_too_large`**，拿它的返回值做 `<= MAX_MESSAGE_BYTES`
        #   的判断永远走不到分片分支（真机实测：ROG 代码已更新仍报
        #   `message exceeds maximum size`）。这里按协议自身的判定分流：能构造就直接发，
        #   抛 `message_too_large` 才切分片。
        try:
            direct = _result(result_payload)
        except WorkerProtocolError as exc:
            if getattr(exc, "code", "") != "message_too_large":
                raise
            direct = None
        if direct is not None:
            return [direct]
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
        # ⚠️ 协议层（`_validate_payload` 的 stage_result 分支）强制
        #   `output_sha256 == stage_output_sha256(output)`，且校验的是**实际发送的 output
        #   对象** ⇒ 切分片后必须按**新 output**（含 `hidden_out_ref`）重算摘要。
        #   原始 hidden 的完整性由 `hidden_out_ref.payload_sha256`（装配摘要）保证，
        #   output 内既有的 `hidden_out_sha256` 因字节未变而继续有效。
        shrunk["output_sha256"] = task_worker_sha256(new_output)
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

        full_model_identity = self._task_worker_full_model_identity(engines)
        models = [full_model_identity] if full_model_identity is not None else []
        active_layer_config = getattr(self, "_active_layer_config", None) or {}
        layer_worker = bool(
            active_layer_config or getattr(self._host, "layer_range", None)
        )
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
            # A registered segment is executable only for the exact manifest
            # range.  Never let a stale active config publish a second range
            # beside the per-artifact contract.
            layer_worker = bool(artifact.get("runtime_supported", True))
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
        artifact_model = self._task_worker_layer_artifact_identity(artifact)
        if artifact_model is not None:
            # Protocol v3 requires layer_artifacts to reference a physical
            # identity in models.  This is an artifact registry entry only;
            # full-stage admission independently requires a loaded full model.
            existing = next((
                item for item in models
                if item.get("model_id") == artifact_model["model_id"]
            ), None)
            if existing is None:
                models.append(artifact_model)
            elif existing != artifact_model:
                raise RuntimeError("layer_artifact_model_identity_conflict")
        stage_types = ["full_inference", "aggregate"]
        if layer_worker and full_model_identity is None:
            stage_types = []
        if layer_worker and "layer_forward" not in stage_types:
            # ★ 票 15a（能力广告半边）：**能起 server 不等于能参与层段接力**。
            #   若当前 runtime 实际不支持 `FORWARD_LAYERS`（例：本机 llama-server
            #   子进程路径），就不得广告 `layer_forward` —— 否则 master 会把层段
            #   派给一台执行不了的 worker。判据走运行时能力探测而非 backend 名字：
            #   server 引擎对外仍报 `engine_type="llama_cpp"`。
            if self._advertises_layer_forward():
                stage_types.append("layer_forward")
            else:
                logger.warning(
                    "task worker 声明为层段 worker，但当前推理路径不支持层段前向 "
                    "-> 不广告 layer_forward（票 15a INFER-PATH-SPLIT-01）"
                )
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
            # Exact registered/assigned ranges, without a loaded-ready ACK.
            "layer_ranges": layer_ranges,
            # ★ 2026-10-08（DIST-NEXT-2b）：声明**能接收分片 hidden**。接收侧
            #   （`_handle_task_worker_stage_chunk` + `hidden_ref` 装配）早已实现，但此前
            #   没有任何声明处 —— 于是 provider 的 `_maybe_send_stage_chunks` 直接跳过
            #   分片、dispatch 前的预检按超限拒绝，大 prompt 只能 503
            #   （实测 `route_a_stage_frame_too_large:…:wire=24160940:budget=8126464`）。
            #   非层段 worker 不接收 hidden，故按 `layer_worker` 保守声明。
            "stage_chunked_input": bool(layer_worker),
            # ★ 票 21a（COMM-WIRE-01）：声明**能收发二进制 hidden 帧**
            #   （`tcp_comm.build_message(binary_body=...)`，见 `split_inline_hidden`）。
            #   声明后本节点与主节点之间的层段 hidden 走 raw 字节：省掉 base64 的 33%
            #   膨胀与编解码，并顺带让单帧容量 +33%（预算判据按 base64 后体积算）。
            #   未声明者（含所有旧对端与 Android）继续走 base64 内联，字节与旧行为
            #   一致 —— 主节点只在看到该键时才使用二进制帧，故关掉即天然回退。
            #   与 `stage_chunked_input` 同样按 `layer_worker` 保守声明：非层段 worker
            #   不承载 hidden。
            "binary_hidden": bool(layer_worker),
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
            if artifact.get("layer_weight_bytes"):
                item["layer_weight_bytes"] = artifact["layer_weight_bytes"]
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

    def _advertises_layer_forward(self) -> bool:
        """当前 runtime 能否**真的**执行层段前向（票 15a 的能力广告判据）。

        保守语义：只有拿到**明确的否定结论**才不广告 —— 探测不出结论时（测试替身、
        未加载模型、无可识别 backend）按"可以"处理，维持既有广告行为不变。
        """
        try:
            from koakuma_engine import Capability, backend_id_for

            host = self._host
            probe = getattr(host, "supports", None)
            if not callable(probe):
                return True
            if not backend_id_for(host):
                # 没有可识别的 backend（未加载/替身）=> 不下否定结论
                return True
            return bool(probe(Capability.FORWARD_LAYERS))
        except Exception:
            return True


    def _send_task_worker_hello(
        self, client=None, refresh_generation: int = 0,
        registration_owner: tuple | None = None,
    ) -> bool:
        """Send hello for exactly one authenticated client/epoch/socket owner."""
        if (
            not self.get_distributed_inference_enabled()
            or self._effective_role() != "client"
            or not self._scheduler_facade_global('TASK_WORKER_EXPERIMENTAL_ENABLED')
        ):
            return False
        target = client or getattr(self, "_tcp_client", None)
        if not target:
            return False
        message = None
        phase = "capabilities"
        try:
            from transport_port import MessageType

            if registration_owner is None:
                registration_owner = target.capture_registration_owner()
            if registration_owner is None:
                return False
            # Hashing artifacts must never hold either the registration lock or
            # the scheduler lock shared with client replacement and cleanup.
            capabilities = self._task_worker_capabilities()
            # ★ 票 21a（COMM-WIRE-01）：记下本进程是否声明了二进制 hidden，供**回程**
            #   发送时作判据。`_task_worker_capabilities()` 会重算工件摘要（注释明确
            #   不能持锁调用），因此不能每条出站消息都重算一次。
            self._task_worker_binary_hidden_advertised = bool(
                capabilities.get("binary_hidden")
            )
            node_id = self.get_effective_node_id()
            phase = "begin"
            with self._task_worker_refresh_lock:
                if getattr(self, "_tcp_client", None) is not target:
                    return False

                def begin():
                    return self._task_worker_control.begin_worker_hello(
                        node_id=node_id, capabilities=capabilities,
                        registration_owner=registration_owner,
                    )

                message = target.run_if_registration_owner(registration_owner, begin)
                if message is None:
                    return False
                self._task_worker_refresh_requested = bool(
                    self._task_worker_refresh_generation > refresh_generation
                )
            phase = "send"
            target.send_data(
                message.snapshot(), MessageType.TASK_WORKER,
                connection_sock=registration_owner[2],
            )
            advertised_layers = [
                (artifact.get("layer_range"), len(artifact.get("layer_weight_bytes") or []))
                for artifact in message.payload["capabilities"].get("layer_artifacts", [])
            ]
            logger.info(
                "event=task_worker_hello_sent node_id=%s registration_epoch=%s "
                "hello_id=%s version=%s artifact_byte_layers=%s",
                node_id, registration_owner[1], message.message_id,
                message.version, advertised_layers,
            )
            return True
        except Exception as exc:
            cleared = False
            if message is not None:
                cleared = self._task_worker_control.disconnect_coordinator(
                    owner=(*registration_owner, message.message_id),
                )
            logger.warning(
                "event=task_worker_hello_%s_failed registration_epoch=%s "
                "hello_id=%s coordinator_cleared=%s error=%s",
                phase,
                registration_owner[1] if registration_owner is not None else None,
                message.message_id if message is not None else None,
                cleared, getattr(exc, "code", type(exc).__name__), exc_info=True,
            )
            return False


    def refresh_task_worker_capabilities(self) -> bool:
        """Schedule a hello bound to the registration requesting this refresh."""
        client = getattr(self, "_tcp_client", None)
        if (
            not self.get_distributed_inference_enabled()
            or self._effective_role() != "client"
            or not self._scheduler_facade_global('TASK_WORKER_EXPERIMENTAL_ENABLED')
            or not client
            or not getattr(client, "is_registered", False)
        ):
            return False
        registration_owner = client.capture_registration_owner()
        if registration_owner is None:
            return False
        with self._task_worker_refresh_lock:
            if getattr(self, "_tcp_client", None) is not client:
                return False
            self._task_worker_refresh_requested = True
            self._task_worker_refresh_generation += 1
            refresh_generation = self._task_worker_refresh_generation
        threading.Thread(
            target=self._send_task_worker_hello,
            args=(client, refresh_generation, registration_owner),
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

        send_options = {}
        epoch_getter = getattr(server, "get_authenticated_peer_epoch", None)
        if callable(epoch_getter):
            epoch = epoch_getter(node_id)
            if epoch is None:
                raise ConnectionError("task worker registration is no longer current")
            send_options["registration_epoch"] = epoch
        # ★ 票 21a（COMM-WIRE-01）：对端声明 `binary_hidden` 时，把内联 hidden 抽成
        #   帧的二进制体（省 base64 的 33% 膨胀与编解码开销）；未声明者逐字段走旧
        #   路径、字节不变。能力位来自该 worker 的 hello 自荐。
        peer_capabilities = (
            self._task_worker_control.worker_snapshot(node_id).get("capabilities") or {}
        )
        binary_enabled = (
            isinstance(peer_capabilities, dict)
            and peer_capabilities.get("binary_hidden") is True
        )
        body_message, binary_body = split_task_worker_hidden(
            message.snapshot(), enabled=binary_enabled,
        )
        if binary_body is not None:
            logger.debug(
                "event=task_worker_binary_hidden_sent node_id=%s message_type=%s bytes=%d",
                node_id, body_message.get("message_type"), len(binary_body),
            )
        # ★ 票 21a：无二进制体时**连调用形状都不变**（不传 `binary_body`）——
        #   否则既有 send_to_client 的替身/包装会因多一个关键字参数而失败
        #   （实测：三处测试替身报 unexpected keyword argument）。
        send_kwargs = dict(send_options)
        if binary_body is not None:
            send_kwargs["binary_body"] = binary_body
        sent = server.send_to_client(
            node_id, body_message, MessageType.TASK_WORKER, **send_kwargs,
        )
        if sent is False:
            raise ConnectionError("task worker message was not delivered")


    def _send_task_worker_to_master(self, message: WorkerMessage) -> None:
        client = getattr(self, "_tcp_client", None)
        if not client or not getattr(client, "is_registered", False):
            raise ConnectionError("task worker is not connected to its master")
        from transport_port import MessageType

        # ★ 票 21a（COMM-WIRE-01）：本进程**声明了** `binary_hidden` 时，回程的
        #   `stage_result.output.hidden_out_f32` 也走帧的二进制体。声明之后仍能握手
        #   成功，就说明对端（主节点）认识这个键 ⇒ 支持二进制帧 —— 旧主节点会以
        #   `invalid_fields` 拒绝 hello，连接根本建立不起来。判据用 hello 时的缓存，
        #   不重算工件摘要。
        snapshot, binary_body = split_task_worker_hidden(
            message.snapshot(),
            enabled=getattr(self, "_task_worker_binary_hidden_advertised", False),
        )
        if binary_body is None:
            # 未声明 ⇒ 连调用形状都不变（不传 `binary_body`）。
            client.send_data(snapshot, MessageType.TASK_WORKER)
            return
        logger.debug(
            "event=task_worker_binary_hidden_sent_to_master message_type=%s bytes=%d",
            snapshot.get("message_type"), len(binary_body),
        )
        client.send_data(snapshot, MessageType.TASK_WORKER, binary_body=binary_body)


    def _task_worker_hello_followup_owner_current(self, node_id, work) -> bool:
        server, generation, _received_at, _changed = work
        return bool(
            self._running
            and not self._startup_cancel_event.is_set()
            and self._effective_role() == "master"
            and self.get_distributed_inference_enabled()
            and self._tcp_server is server
            and server is not None and getattr(server, "_running", False)
            and generation > 0
            and self._assignment_connection_generation(node_id) == generation
            and node_id not in self._task_worker_control.pending_worker_ids()
        )

    def _task_worker_hello_followup_current(self, node_id, work) -> bool:
        return bool(
            self._task_worker_hello_followup_owner_current(node_id, work)
            and self._task_worker_control.worker_snapshot(node_id).get(
                "hello_received_at"
            ) == work[2]
        )

    def _queue_task_worker_hello_followup(self, node_id, server, generation):
        snapshot = self._task_worker_control.worker_snapshot(node_id)
        work = (server, generation, snapshot.get("hello_received_at"),
                snapshot.get("capabilities_changed", True))
        # This is a bounded wakeup list, not another worker/connection registry.
        # All decisions below come from the live control and transport owners.
        with self._task_worker_refresh_lock:
            if not self._task_worker_hello_followup_current(node_id, work):
                return
            pending = getattr(self, "_task_worker_hello_followups", None)
            if pending is None:
                pending = self._task_worker_hello_followups = {}
            previous = pending.get(node_id)
            for peer, item in list(pending.items()):
                if not self._task_worker_hello_followup_owner_current(peer, item):
                    pending.pop(peer, None)
            if previous is not None and previous[:2] == work[:2]:
                # An identical refresh must not erase an earlier changed hello
                # that the monitor has not yet applied.
                work = (*work[:3], previous[3] or work[3])
            if node_id not in pending and len(pending) >= self._max_nodes:
                logger.warning("task worker hello followup bound reached node=%s", node_id)
                return
            pending[node_id] = work
        self._start_layer_config_retry_monitor()

    def _start_layer_config_retry_monitor(self):
        # Serialize starts from both hello and ordinary configuration publishes.
        # Reuse the one daemon and its existing _running/stop lifecycle.
        with self._task_worker_refresh_lock:
            super()._start_layer_config_retry_monitor()

    def _retry_pending_layer_configs(self, now=None) -> int:
        self._drain_task_worker_hello_followups()
        return super()._retry_pending_layer_configs(now=now)

    def _drain_task_worker_hello_followups(self):
        with self._task_worker_refresh_lock:
            pending = getattr(self, "_task_worker_hello_followups", {})
            items = list(pending.items())
        for node_id, work in items:
            if not self._task_worker_hello_followup_current(node_id, work):
                with self._task_worker_refresh_lock:
                    # A same-owner HELLO may have updated the control snapshot
                    # before reaching its ACK/queue edge. Keep changed intent
                    # for that successor to merge; only retire an expired owner.
                    if (pending.get(node_id) is work
                            and not self._task_worker_hello_followup_owner_current(node_id, work)):
                        pending.pop(node_id, None)
                continue
            # Never park the retry monitor behind GC, either. Recheck ownership
            # *after* acquiring the execution lock, before creating a provider.
            if not self._task_worker_stage_lock.acquire(blocking=False):
                continue
            try:
                if not self._task_worker_hello_followup_current(node_id, work):
                    continue
                snapshot = self._task_worker_control.worker_snapshot(node_id)
                if snapshot.get("accepted"):
                    self._ensure_remote_task_worker_provider(node_id)
            finally:
                self._task_worker_stage_lock.release()
            try:
                self._apply_task_worker_hello_followup(node_id, work)
            except Exception:
                # One failed publish must not kill the shared retry monitor.
                logger.warning("task worker hello followup failed node=%s", node_id,
                               exc_info=True)
                continue
            with self._task_worker_refresh_lock:
                if (pending.get(node_id) is work
                        and (self._task_worker_hello_followup_current(node_id, work)
                             or not self._task_worker_hello_followup_owner_current(node_id, work))):
                    pending.pop(node_id, None)

    def _apply_task_worker_hello_followup(self, node_id, work):
        def current():
            return self._task_worker_hello_followup_current(node_id, work)

        if not current():
            return
        snapshot = self._task_worker_control.worker_snapshot(node_id)
        accepted = snapshot.get("accepted")
        capabilities = snapshot.get("capabilities") or {}
        cleanup_pending = bool(self._pending_pipeline_model_cleanup_for_node(node_id))
        if accepted and cleanup_pending and current():
            self._replay_pipeline_model_cleanup_for_node(node_id)
        recovery_sync = bool(accepted and self._pipeline_recovery_pending)
        if not cleanup_pending and not recovery_sync and work[3] and current():
            self.push_layer_config_to_clients()
        if recovery_sync and not cleanup_pending and current():
            self.request_authoritative_layer_sync(require_distributed=True)
        is_layer_worker = bool(
            "layer_forward" in capabilities.get("stage_types", [])
            and capabilities.get("layer_ranges")
        )
        if (accepted and capabilities.get("models") and not is_layer_worker
                and not capabilities.get("layer_worker", False)
                and not self._is_relay_host(node_id) and current()):
            self._handle_layer_worker_opt_out(
                node_id, {"data": {"node_id": node_id}}, hello_work=work,
            )


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


    def _cleanup_task_worker_prepare(self, reason_code, *, client=None, close=False):
        if client is not None and getattr(self, "_tcp_client", None) is not client:
            return
        if client is None:
            with self._task_worker_refresh_lock:
                getattr(self, "_task_worker_hello_followups", {}).clear()
        with self._task_worker_stage_lock:
            for attempt in self._task_worker_active_attempts.values():
                attempt.cancel_reason = reason_code
                attempt.cancel_event.set()
        controller = getattr(self, "_worker_prepare", None)
        if controller is not None:
            if close:
                controller.close(wait=False)
            else:
                controller.invalidate(reason_code, client=client)

    def _check_task_worker_prepare_stage(self, offer):
        from worker_prepare_contract import WorkerPrepareError
        controller = getattr(self, "_worker_prepare", None)
        if controller is None:
            raise WorkerPrepareError("worker_prepare_executor_unavailable")
        controller.check_stage(offer)

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
        if not self.get_distributed_inference_enabled():
            reject_reason = "local_inference_mode"
        elif not self._scheduler_facade_global('TASK_WORKER_EXPERIMENTAL_ENABLED'):
            reject_reason = "worker_experiment_disabled"
            reject_retryable = True
        else:
            if offer.get("stage_type") == "layer_forward":
                try:
                    self._check_task_worker_prepare_stage(offer)
                    dispatch_enabled = True
                except Exception as exc:
                    reject_reason = getattr(exc, "reason_code", "worker_prepare_stage_rejected")
                    dispatch_enabled = False
            else:
                dispatch_enabled = coordinator.get("manual_stage_dispatch_enabled")
            if not dispatch_enabled and not reject_reason:
                reject_reason = "worker_not_admitted"
                reject_retryable = True
        if not reject_reason and offer["provider_id"] != expected_provider:
            reject_reason = "provider_identity_mismatch"
        elif not reject_reason and offer.get("stage_type") != "layer_forward":
            capabilities = self._task_worker_capabilities()
            if offer["stage_type"] not in capabilities["stage_types"]:
                reject_reason = "unsupported_stage_type"
                reject_retryable = True
            elif not self._task_worker_offer_model_identity_matches(
                offer, capabilities,
            ):
                reject_reason = "model_identity_mismatch"
                reject_retryable = True

        with self._task_worker_stage_lock:
            if not self.get_distributed_inference_enabled():
                reject_reason = "local_inference_mode"
                reject_retryable = False
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
                        "hidden_spec", "middle_channel", "seq_ids", "positions", "prepare_identity",
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
                if not self.get_distributed_inference_enabled():
                    raise RuntimeError("local_inference_mode")

                stop_reason = self._acquire_task_worker_execution_lock(
                    self._inference_lock, active,
                )
                if stop_reason:
                    raise RuntimeError(stop_reason)
                acquired_inference_lock = True

                if offer.get("stage_type") == "layer_forward":
                    self._check_task_worker_prepare_stage(offer)
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
        # ★ 票 21a（COMM-WIRE-01）：帧的二进制体**必须在 decode 之前**放回内联 hidden。
        #   精确字段校验会以 `invalid_fields` 拒绝顶层未知键 ⇒ `binary` 段绝不能随消息
        #   进 `decode_message`。放回的是 base64 字符串，因此其后所有既有代码
        #   （含 `_handle_task_worker_stage_offer`）都不必知道二进制路径存在。
        binary_body = msg.get("binary")
        if isinstance(binary_body, (bytes, bytearray)):
            raw = restore_task_worker_hidden(raw, bytes(binary_body))
            logger.debug(
                "event=task_worker_binary_hidden_received peer=%s "
                "message_type=%s bytes=%d",
                client_id, raw.get("message_type"), len(binary_body),
            )
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
                    hello_connection_generation = self._assignment_connection_generation(client_id)
                    try:
                        ack = self._task_worker_control.receive_on_coordinator(
                            client_id,
                            raw,
                            coordinator_node_id=self.get_effective_node_id(),
                            connection_generation=hello_connection_generation,
                        )
                    except Exception:
                        logger.error(
                            "task worker hello 校验失败: node=%s", client_id,
                            exc_info=True,
                        )
                        raise
                    advertised_capabilities = (
                        message.payload.get("capabilities", {})
                        if isinstance(message.payload, dict)
                        else {}
                    )
                    logger.info(
                        "event=task_worker_hello_capabilities node_id=%s summary=%s",
                        client_id,
                        _task_worker_capability_log_summary(
                            advertised_capabilities
                        ),
                    )
                    self._send_task_worker_to_node(client_id, ack)
                    # The registration fence must end for both an accepted
                    # hello and a definitive rejection.  A rejected hello is
                    # no longer negotiating the task-worker path, so legacy
                    # scheduling can be recomputed for that connection.
                    if not self._task_worker_control.resolve_worker_connection_pending(
                        client_id, connection_generation=hello_connection_generation,
                    ):
                        # A new REGISTER superseded this hello after its ACK.
                        # The old callback must not open the successor's fence.
                        return
                    # ACK and the registration fence belong to the receive
                    # thread. Provider admission and scheduling can wait behind
                    # supply GC's stage lock; leave them to the existing retry
                    # monitor so this socket can keep processing heartbeats.
                    self._queue_task_worker_hello_followup(
                        client_id, self._tcp_server, hello_connection_generation,
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
                    # ★ 2026-10-11（ROG+Y700+本机三机实测）：**回程分片**。
                    #   此前这一层方向白名单没有它 ⇒ 分片被逐条以
                    #   `invalid_message_direction` 拒掉（真机日志：7 条 rejected），
                    #   随后的 `stage_result` 于是装配不齐（`incomplete_chunks`），
                    #   主节点侧只能看到 60 s `remote Stage response timed out`。
                    "stage_result_chunk",
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
                and bool(self._task_worker_full_model_identities(
                    worker["capabilities"]
                ))
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
            local_caps = self._task_worker_capabilities()
            local_full_models = self._task_worker_full_model_identities(
                local_caps
            )
            full_model_worker_ids = (
                [self.get_effective_node_id()]
                if connected and local_full_models else []
            )
            workers_missing_full_model = (
                [self.get_effective_node_id()]
                if connected and not local_full_models else []
            )
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
