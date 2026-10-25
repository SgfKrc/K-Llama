"""Model one-click download job service (P0A).

后端下载 job 状态机：resolve -> download -> verify -> register -> ready，
并持久化到主节点 SQLite（``model_download_jobs`` 表），供前端轮询进度。

设计要点（对齐项目既有惯例）：
- 复用 ``import_model`` 的校验/注册语义（build_manifest / validate_model_artifact /
  write_manifest / save_local_experimental_model），但为可编程 service 重写，
  不依赖 scripts 目录的相对导入或 CLI 参数层。
- 资源门前置：GGUF 按 ``asset_size*1.2 + 512MB`` 估 RAM；safetensors 按
  recommended_vram + ``asset_size*0.55+512MB`` 估 VRAM，非 CUDA 允许 CPU 时按
  ``asset_size*2.2 + 1GB`` 估 RAM（与 ``llm_smoke_matrix._resource_rejection`` 对齐）。
- 幂等：同一 ``source + target + model_id`` 去重；ready 后不再重复排队。
- fail-closed：未配置显式 SHA-256 时仍做全量 SHA-256 记录；配置则严格校验，
  不匹配即失败并清理 staging。

下载链默认在调用方提供的 ``runner``（后台线程池）中执行；模块本身不持有全局后台线程。
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import tempfile
import threading
import time
import uuid
from pathlib import Path
from typing import Any, Callable

import local_store
import model_registry_validation as mrv
import proxy_config
from model_config import _APP_ROOT
from model_registry_validation import build_manifest, validate_model_artifact, write_manifest

# 下载成功后落盘子目录名（沿用 config.MODEL_PATH 的 ``models/<name>`` 约定）
_DEFAULT_MODELS_ROOT = os.path.join(_APP_ROOT, "models")
_WEIGHT_SUFFIXES = (".safetensors", ".gguf", ".bin")
_SAFE_SUFFIXES = (".safetensors", ".bin")

# job 状态机
STATUS_QUEUED = "queued"
STATUS_DOWNLOADING = "downloading"
STATUS_VERIFYING = "verifying"
STATUS_REGISTERING = "registering"
STATUS_READY = "ready"
STATUS_FAILED = "failed"
STATUS_CANCELLED = "cancelled"

_ACTIVE_STATUSES = {STATUS_QUEUED, STATUS_DOWNLOADING, STATUS_VERIFYING, STATUS_REGISTERING}

# 资源门错误码
ERR_INSUFFICIENT_RAM = "insufficient_ram"
ERR_INSUFFICIENT_VRAM = "insufficient_vram"
ERR_CUDA_REQUIRED = "cuda_required"

_LOCK = threading.Lock()

# 预设列表（P0）。先代码内置，便于审查；后续可迁到 DB。
# 每项枚举了双源 repo、默认文件名量化、resource_gate 与元数据。
# NOTE: 真实 repo id / 受控 SHA 需逐项核实后再启用严格校验（见 docs 计划）。
PRESETS: list[dict[str, Any]] = [
    # ★ 2026-09-19：原 `qwen-1_8b-gguf-q4` 预设**已移除**（用户裁定）——该模型已退役
    #   并从内置列表移除，保留下载入口只会让用户下到一个跑不起来的模型。
    #   模型文件与 GGUF 仍在 `models/` 保留，只是不再提供下载预设。
    {
        "id": "qwen3-3_8b-gguf-q4",
        "display": "Qwen3-3.8B Q4_K_M (GGUF)",
        "kind": "gguf",
        "default_engine": "llama_cpp",
        "default_quant": "Q4_K_M",
        "default_model_id": "qwen3-3_8b",
        "default_target": "qwen3-3_8b",
        "hf_repo": "Qwen/Qwen3-3.8B",
        "ms_path": "",
        "file_pattern": "qwen3-3_8b-q4_k_m.gguf",
        "expected_sha256": "",
        "resource_gate": {
            "min_ram_gb": 6.0,
            "min_vram_gb": 0.0,
            "min_disk_gb": 3.0,
            "allow_cpu": True,
        },
        "description": "Qwen3-3.8B 的 4-bit GGUF，CPU 可跑，推理质量与速度均衡。",
    },
    {
        "id": "deepseek-7b-gguf-q4",
        "display": "DeepSeek-7B Q4_K_M (GGUF)",
        "kind": "gguf",
        "default_engine": "llama_cpp",
        "default_quant": "Q4_K_M",
        "default_model_id": "deepseek-7b",
        "default_target": "deepseek-7b",
        "hf_repo": "deepseek-ai/deepseek-llm-7b-chat",
        "ms_path": "",
        "file_pattern": "deepseek-llm-7b-chat.Q4_K_M.gguf",
        "expected_sha256": "",
        "resource_gate": {
            "min_ram_gb": 8.0,
            "min_vram_gb": 0.0,
            "min_disk_gb": 5.0,
            "allow_cpu": True,
        },
        "description": "DeepSeek-7B 的 4-bit GGUF，适合中低配机器。",
    },
    {
        "id": "qwen2.5-0.5b-instruct",
        "display": "Qwen2.5-0.5B-Instruct (Safetensors)",
        "kind": "safetensors",
        "default_engine": "pytorch",
        "default_quant": "int4",
        "default_model_id": "qwen2.5-0.5b",
        "default_target": "qwen2.5-0.5b-instruct",
        "hf_repo": "Qwen/Qwen2.5-0.5B-Instruct",
        "ms_path": "",
        "file_pattern": "",
        "expected_sha256": "",
        "resource_gate": {
            "min_ram_gb": 3.0,
            "min_vram_gb": 1.5,
            "min_disk_gb": 2.0,
            "allow_cpu": True,
        },
        "description": "亚 1B Qwen2.5 基线，适合链路、模板和上下文策略验证。",
    },
    {
        "id": "qwen3-0.6b",
        "display": "Qwen3-0.6B (Safetensors)",
        "kind": "safetensors",
        "default_engine": "pytorch",
        "default_quant": "int4",
        "default_model_id": "qwen3-0.6b",
        "default_target": "qwen3-0.6b",
        "hf_repo": "Qwen/Qwen3-0.6B",
        "ms_path": "",
        "file_pattern": "",
        "expected_sha256": "",
        "resource_gate": {
            "min_ram_gb": 4.0,
            "min_vram_gb": 2.0,
            "min_disk_gb": 3.0,
            "allow_cpu": True,
        },
        "description": "亚 1B Qwen3 对照模型，需验证 thinking 开关与真实推理边界。",
    },
    {
        "id": "minicpm4-0.5b",
        "display": "MiniCPM4-0.5B (Safetensors)",
        "kind": "safetensors",
        "default_engine": "pytorch",
        "default_quant": "int4",
        "default_model_id": "minicpm4-0.5b",
        "default_target": "minicpm4-0.5b",
        "hf_repo": "openbmb/MiniCPM4-0.5B",
        "ms_path": "",
        "file_pattern": "",
        "expected_sha256": "",
        "resource_gate": {
            "min_ram_gb": 3.0,
            "min_vram_gb": 1.5,
            "min_disk_gb": 2.0,
            "allow_cpu": True,
        },
        "description": "亚 1B MiniCPM4 新架构候选，优先验证模板与 llama.cpp 兼容性。",
    },
    {
        "id": "distilqwen25-ds3-0324-7b",
        "display": "DistilQwen2.5-DS3-0324-7B (Safetensors)",
        "kind": "safetensors",
        "default_engine": "pytorch",
        "default_quant": "int4",
        "default_model_id": "distilqwen25-ds3-0324-7b",
        "default_target": "distilqwen25-ds3-0324-7b",
        "hf_repo": "alibaba-pai/DistilQwen2.5-DS3-0324-7B",
        "ms_path": "",
        "file_pattern": "",
        "expected_sha256": "",
        "resource_gate": {
            "min_ram_gb": 12.0,
            "min_vram_gb": 8.0,
            "min_disk_gb": 20.0,
            "allow_cpu": True,
        },
        "description": "Qwen2.5 架构快思考替代候选，需完成 GGUF 与预算内收敛验证。",
    },
    {
        "id": "qw3-vl-4b",
        "display": "QW3-VL-4B (Safetensors)",
        "kind": "safetensors",
        "default_engine": "pytorch",
        "default_quant": "fp16",
        "default_model_id": "qw3-vl-4b",
        "hf_repo": "Qwen/Qwen3-VL-4B",
        "ms_path": "",
        "file_pattern": "",
        "expected_sha256": "",
        "resource_gate": {
            "min_ram_gb": 10.0,
            "min_vram_gb": 8.0,
            "min_disk_gb": 10.0,
            "allow_cpu": False,
        },
        "description": "Qwen3-VL 4B 视觉多模态，需 CUDA GPU（约 8GB 显存）。",
    },
]
_PRESETS_BY_ID = {p["id"]: p for p in PRESETS}
# ★ 2026-09-20：pin manifest 原在 `docs/agent_tool/model-artifacts/`，而 `docs/agent_tool/`
#   是**开发工具**（已从主仓裁掉、只留本地）⇒ 该文件是**运行时数据**，已随迁到
#   `fixtures/model-artifacts/`（入库）。**不要**再指回 `docs/agent_tool/`。
_PIN_MANIFEST_PATH = Path(__file__).resolve().parents[1] / "fixtures" / "model-artifacts" / "remote-model-pins-2026-09-09.json"
_PINNED_PRESET_IDS = {
    "qwen2.5-0.5b-instruct",
    "qwen3-0.6b",
    "minicpm4-0.5b",
    "distilqwen25-ds3-0324-7b",
}
_REVISION_RE = re.compile(r"^[0-9a-f]{40}$")
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


class JobError(Exception):
    """job 执行失败的宿主异常，message 会脱敏写入 error 字段。"""

    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code
        self.message = message


# ---------------------------------------------------------------------------
# 数据库持久化
# ---------------------------------------------------------------------------

def _now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _row_to_job(row: Any) -> dict[str, Any]:
    return {
        "job_id": row["job_id"],
        "status": row["status"],
        "progress": float(row["progress"] or 0.0),
        "source": row["source"],
        "target": row["target"],
        "model_id": row["model_id"],
        "preset_id": row["preset_id"],
        "engine": row["engine"],
        "quant": row["quant"],
        "total_bytes": int(row["total_bytes"] or 0),
        "downloaded_bytes": int(row["downloaded_bytes"] or 0),
        "error_code": row["error_code"],
        "error": row["error"],
        "created_at": row["created_at"],
        "updated_at": row["updated_at"],
        "finished_at": row["finished_at"],
    }


def _insert_job(row: dict[str, Any]) -> None:
    with local_store._write_connection() as connection:
        connection.execute(
            """
            INSERT INTO model_download_jobs (
              job_id, status, progress, source, target, model_id, preset_id,
              engine, quant, total_bytes, downloaded_bytes, error_code, error,
              created_at, updated_at, finished_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                row["job_id"], row["status"], row["progress"], row["source"],
                row["target"], row["model_id"], row["preset_id"], row["engine"],
                row["quant"], row["total_bytes"], row["downloaded_bytes"],
                row["error_code"], row["error"], row["created_at"], row["updated_at"],
                row["finished_at"],
            ),
        )


def _update_job(job_id: str, **fields: Any) -> None:
    if not fields:
        return
    fields["updated_at"] = _now()
    cols = ", ".join(f"{key}=?" for key in fields)
    params = list(fields.values()) + [job_id]
    with local_store._write_connection() as connection:
        connection.execute(f"UPDATE model_download_jobs SET {cols} WHERE job_id=?", params)


def get_job(job_id: str) -> dict[str, Any] | None:
    connection = local_store._connect()
    try:
        row = connection.execute(
            "SELECT * FROM model_download_jobs WHERE job_id=?", (job_id,)
        ).fetchone()
        return _row_to_job(row) if row else None
    finally:
        connection.close()


def list_jobs(limit: int = 100) -> list[dict[str, Any]]:
    connection = local_store._connect()
    try:
        rows = connection.execute(
            "SELECT * FROM model_download_jobs ORDER BY created_at DESC LIMIT ?",
            (int(limit),),
        ).fetchall()
        return [_row_to_job(r) for r in rows]
    finally:
        connection.close()


def _job_key(source: str, target: str, model_id: str) -> str:
    return f"{source}|{target}|{model_id}"


def _active_job_exists(source: str, target: str, model_id: str) -> bool:
    connection = local_store._connect()
    try:
        rows = connection.execute(
            "SELECT job_id FROM model_download_jobs WHERE source=? AND target=? AND model_id=?",
            (source, target, model_id),
        ).fetchall()
        for r in rows:
            if r["job_id"]:
                status = get_job(r["job_id"])
                if status and status["status"] in _ACTIVE_STATUSES:
                    return True
        return False
    finally:
        connection.close()


# ---------------------------------------------------------------------------
# 预设查询 + 资源门
# ---------------------------------------------------------------------------

def _preset_source(preset: dict[str, Any], use_modelscope: bool) -> str:
    if use_modelscope:
        if not preset.get("ms_path"):
            raise JobError("PRESET_NO_MS_SOURCE", f"预设 {preset['id']} 无 ModelScope 源")
        return preset["ms_path"]
    if not preset.get("hf_repo"):
        raise JobError("PRESET_NO_HF_SOURCE", f"预设 {preset['id']} 无 HF 源")
    return preset["hf_repo"]


def _available_ram_bytes() -> int:
    try:
        import psutil
        return int(psutil.virtual_memory().available)
    except Exception:
        return 0


def _resource_gate(rejected: dict[str, str], preset: dict[str, Any]) -> None:
    gate = preset.get("resource_gate", {})
    min_ram_gb = float(gate.get("min_ram_gb", 0))
    allow_cpu = bool(gate.get("allow_cpu", True))
    ram = _available_ram_bytes()
    if min_ram_gb > 0 and ram < int(min_ram_gb * 1024**3):
        rejected["ram"] = ERR_INSUFFICIENT_RAM
    if not allow_cpu:
        allowed_vram = float(gate.get("min_vram_gb", 0))
        try:
            import torch
            if not torch.cuda.is_available():
                rejected["gpu"] = ERR_CUDA_REQUIRED
            elif allowed_vram > 0:
                free, total = torch.cuda.mem_get_info()
                if int(total) < int(allowed_vram * 1024**3 * 0.9) or int(free) < int(allowed_vram * 1024**3):
                    rejected["gpu"] = ERR_INSUFFICIENT_VRAM
        except Exception:
            rejected["gpu"] = ERR_CUDA_REQUIRED


def _validate_pin_path(value: Any, *, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise JobError("PRESET_PIN_INVALID", f"pin {field} 必须是非空相对路径")
    normalized = value.replace("\\", "/")
    path = Path(normalized)
    if path.is_absolute() or path.drive or ".." in path.parts:
        raise JobError("PRESET_PIN_INVALID", f"pin {field} 不能越出仓库根目录")
    return normalized


def _load_remote_pin(preset: dict[str, Any]) -> dict[str, Any] | None:
    """Load and validate the reviewed artifact pin for a remote preset."""
    preset_id = str(preset.get("id") or "")
    if preset_id not in _PINNED_PRESET_IDS:
        return None
    try:
        with _PIN_MANIFEST_PATH.open("r", encoding="utf-8") as handle:
            document = json.load(handle)
    except (OSError, json.JSONDecodeError) as exc:
        raise JobError("PRESET_PIN_INVALID", f"无法读取远程工件 pin: {_PIN_MANIFEST_PATH}") from exc

    if not isinstance(document, dict) or document.get("schema") != 1 or not isinstance(document.get("presets"), dict):
        raise JobError("PRESET_PIN_INVALID", "远程工件 pin schema 无效")
    entry = document["presets"].get(preset_id)
    if not isinstance(entry, dict):
        raise JobError("PRESET_PIN_MISSING", f"预设 '{preset_id}' 没有远程工件 pin")
    if entry.get("repo_id") != preset.get("hf_repo"):
        raise JobError("PRESET_PIN_INVALID", f"预设 '{preset_id}' 的 repo_id 与 pin 不一致")

    revision = entry.get("revision")
    if not isinstance(revision, str) or not _REVISION_RE.fullmatch(revision):
        raise JobError("PRESET_PIN_INVALID", f"预设 '{preset_id}' 的 revision 必须是完整 commit SHA")

    allow_patterns = entry.get("allow_patterns")
    required_files = entry.get("required_files")
    weight_files = entry.get("weight_files")
    if not isinstance(allow_patterns, list) or not allow_patterns:
        raise JobError("PRESET_PIN_INVALID", f"预设 '{preset_id}' 缺少 allow_patterns")
    if not isinstance(required_files, list) or not required_files:
        raise JobError("PRESET_PIN_INVALID", f"预设 '{preset_id}' 缺少 required_files")
    if not isinstance(weight_files, list) or not weight_files:
        raise JobError("PRESET_PIN_INVALID", f"预设 '{preset_id}' 缺少 weight_files")

    normalized_patterns = [_validate_pin_path(item, field="allow_patterns") for item in allow_patterns]
    normalized_required = [_validate_pin_path(item, field="required_files") for item in required_files]
    if len(set(normalized_patterns)) != len(normalized_patterns):
        raise JobError("PRESET_PIN_INVALID", f"预设 '{preset_id}' 的 allow_patterns 重复")
    if len(set(normalized_required)) != len(normalized_required):
        raise JobError("PRESET_PIN_INVALID", f"预设 '{preset_id}' 的 required_files 重复")
    if not set(normalized_required) <= set(normalized_patterns):
        raise JobError("PRESET_PIN_INVALID", f"预设 '{preset_id}' 的 required_files 未被 allow_patterns 覆盖")

    normalized_weights: list[dict[str, str]] = []
    seen_weights: set[str] = set()
    for item in weight_files:
        if not isinstance(item, dict):
            raise JobError("PRESET_PIN_INVALID", f"预设 '{preset_id}' 的 weight_files 项无效")
        path = _validate_pin_path(item.get("path"), field="weight_files.path")
        sha256 = item.get("sha256")
        if not isinstance(sha256, str) or not _SHA256_RE.fullmatch(sha256.lower()):
            raise JobError("PRESET_PIN_INVALID", f"预设 '{preset_id}' 的权重 SHA-256 无效")
        if path in seen_weights:
            raise JobError("PRESET_PIN_INVALID", f"预设 '{preset_id}' 的 weight_files 重复")
        if path not in normalized_patterns:
            raise JobError("PRESET_PIN_INVALID", f"预设 '{preset_id}' 的权重文件未被 allow_patterns 覆盖")
        seen_weights.add(path)
        normalized_weights.append({"path": path, "sha256": sha256.lower()})

    return {
        "repo_id": entry["repo_id"],
        "revision": revision,
        "allow_patterns": normalized_patterns,
        "required_files": normalized_required,
        "weight_files": normalized_weights,
    }


def list_presets() -> list[dict[str, Any]]:
    """返回带本机可装性评估的预设列表。"""
    out: list[dict[str, Any]] = []
    for preset in PRESETS:
        item = dict(preset)
        rejected: dict[str, str] = {}
        if preset["id"] in _PINNED_PRESET_IDS:
            try:
                pin = _load_remote_pin(preset)
                if pin is None:
                    raise JobError("PRESET_PIN_MISSING", f"预设 '{preset['id']}' 没有远程工件 pin")
                item["pin_status"] = "pinned"
                item["revision"] = pin["revision"]
                item["allow_patterns"] = pin["allow_patterns"]
                item["required_files"] = pin["required_files"]
                item["weight_files"] = pin["weight_files"]
            except JobError as exc:
                item["pin_status"] = "invalid"
                rejected["artifact_pin"] = exc.code
        _resource_gate(rejected, preset)
        if rejected:
            item["installable"] = False
            item["blocked_reasons"] = rejected
        else:
            item["installable"] = True
            item["blocked_reasons"] = {}
        out.append(item)
    return out


# ---------------------------------------------------------------------------
# 下载链（resolve -> download -> verify -> register）
# ---------------------------------------------------------------------------

def _weight_files(target: Path) -> list[Path]:
    if not target.is_dir():
        return []
    return sorted(
        path for path in target.rglob("*")
        if path.is_file() and path.name.lower().endswith(_WEIGHT_SUFFIXES)
    )


def _resolve_target(repo_or_path: str, target: str | None, models_root: str) -> Path:
    """返回发布目标（destination）。

    对本地目录 source：destination 由 target/models_root 决定（source 仅作内容来源），
    name 取自目录名。
    """
    root = Path(models_root).expanduser().resolve(strict=False)
    source = Path(repo_or_path).expanduser()
    if source.is_dir():
        name = source.name
    else:
        name = repo_or_path.strip("/").split("/")[-1]
    if not name:
        raise JobError("SOURCE_INVALID", f"无法从 source 解析模型名: {repo_or_path!r}")
    raw_target = Path(target).expanduser() if target else root / name
    candidate = raw_target if raw_target.is_absolute() else root / raw_target
    destination = candidate.resolve(strict=False)
    try:
        destination.relative_to(root)
    except ValueError as exc:
        raise JobError("TARGET_OUTSIDE_MODELS_ROOT", "模型目标目录必须位于 models 根目录内") from exc
    if destination == root:
        raise JobError("TARGET_INVALID", "模型目标目录不能等于 models 根目录")
    return destination


def _validate_remote_source(source: str) -> None:
    """Keep repository sources in the provider namespace shape."""
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,95}/[A-Za-z0-9][A-Za-z0-9_.-]{0,95}", source):
        raise JobError("SOURCE_INVALID", "远程模型 source 必须是 provider/仓库名")


def _resolve_staged_gguf(value: str, staging: Path) -> str:
    """Resolve an explicit GGUF selector without allowing external files."""
    if not value:
        return ""
    raw = Path(value).expanduser()
    candidate = raw if raw.is_absolute() else staging / raw
    resolved = candidate.resolve(strict=False)
    try:
        resolved.relative_to(staging.resolve())
    except ValueError as exc:
        raise JobError("GGUF_PATH_OUTSIDE_TARGET", "显式 GGUF 路径必须位于下载目标内") from exc
    if resolved.suffix.lower() != ".gguf":
        raise JobError("GGUF_PATH_INVALID", "显式 GGUF 路径必须以 .gguf 结尾")
    return str(resolved)


def _relocate_staged_path(value: str, staging: Path, destination: Path) -> str:
    if not value:
        return ""
    try:
        relative = Path(value).resolve().relative_to(staging.resolve())
    except ValueError as exc:
        raise JobError("GGUF_PATH_OUTSIDE_TARGET", "显式 GGUF 路径不在 staging 目录内") from exc
    return str((destination / relative).resolve(strict=False))


def _download(source: str, staging: Path, *, use_modelscope: bool, proxy: str,
              progress_cb: Callable[[Path, int, int], None] | None = None,
              revision: str = "", allow_patterns: list[str] | None = None) -> list[Path]:
    """下载到 staging，并回调（已下载字节，总字节）进度。

    本地目录作为 source 时整体拷贝到 staging（模拟下载产物），便于测试与本地导入。
    """
    staging.mkdir(parents=True, exist_ok=True)
    src_path = Path(source)
    if src_path.is_dir():
        if src_path.is_symlink():
            raise JobError("SOURCE_SYMLINK", "本地模型 source 不能是符号链接")
        for item in src_path.rglob("*"):
            if item.is_symlink():
                raise JobError("SOURCE_SYMLINK", "本地模型 source 不能包含符号链接")
            if item.is_file():
                rel = item.relative_to(src_path)
                dest = staging / rel
                dest.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(item, dest)
        files = _weight_files(staging)
        total = sum(f.stat().st_size for f in files)
        if progress_cb and files:
            progress_cb(files[-1], total, total)
        if not files:
            raise JobError("DOWNLOAD_NO_WEIGHTS", "本地目录未发现权重文件")
        return files
    _validate_remote_source(source)
    resolved_proxy = proxy_config.resolve_http_proxy(proxy or None)
    if use_modelscope:
        code = ("from modelscope import snapshot_download; "
                f"snapshot_download({source!r}, local_dir={str(staging)!r})")
        with proxy_config.proxy_environment(resolved_proxy) as environment:
            import subprocess
            import sys
            result = subprocess.run(
                [sys.executable, "-c", code], capture_output=True, text=True,
                encoding="utf-8", errors="replace", env=environment,
            )
        if result.returncode != 0:
            raise JobError("DOWNLOAD_FAILED",
                           f"ModelScope 下载失败: {(result.stderr or result.stdout)[-300:]}")
    else:
        import huggingface_hub
        with proxy_config.proxy_environment(resolved_proxy) as environment:
            keys = ("HTTP_PROXY", "HTTPS_PROXY", "http_proxy", "https_proxy")
            previous = {key: os.environ.get(key) for key in keys}
            try:
                for key in keys:
                    value = environment.get(key)
                    if value is None:
                        os.environ.pop(key, None)
                    else:
                        os.environ[key] = value
                options: dict[str, Any] = {
                    "repo_id": source,
                    "local_dir": str(staging),
                    # 注：`local_dir_use_symlinks` 在 huggingface_hub **1.x 已移除**（1.x 会 pop 掉它
                    # 并给警告，且默认不再使用 symlink）⇒ 不再传入，避免噪声与语义歧义。
                }
                if revision:
                    options["revision"] = revision
                if allow_patterns:
                    options["allow_patterns"] = allow_patterns
                huggingface_hub.snapshot_download(**options)
            finally:
                for key, value in previous.items():
                    if value is None:
                        os.environ.pop(key, None)
                    else:
                        os.environ[key] = value
    files = _weight_files(staging)
    if not files:
        raise JobError("DOWNLOAD_NO_WEIGHTS",
                       "下载完成但未发现 safetensors/gguf/bin 权重文件")
    return files


def _verify(files: list[Path], expected_sha256: str | None) -> dict[str, Any]:
    if not files:
        raise JobError("VERIFY_EMPTY", "没有可校验的权重文件")
    total_bytes = 0
    digest = hashlib.sha256()
    for path in files:
        if not path.is_file() or path.stat().st_size <= 0:
            raise JobError("VERIFY_EMPTY_FILE", f"权重文件缺失或为空: {path}")
        size = path.stat().st_size
        total_bytes += size
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
    raw = digest.hexdigest()
    summary = {"file_count": len(files), "total_bytes": total_bytes, "sha256": raw}
    if expected_sha256 and raw != expected_sha256.lower():
        raise JobError(
            "SHA256_MISMATCH",
            f"SHA-256 不匹配: 期望 {expected_sha256.lower()}，实际 {raw}",
        )
    return summary


def _verify_required_files(root: Path, required_files: list[str] | None) -> None:
    if not required_files:
        return
    root = root.resolve()
    for relative in required_files:
        candidate = (root / relative).resolve(strict=False)
        try:
            candidate.relative_to(root)
        except ValueError as exc:
            raise JobError("PRESET_PIN_INVALID", f"必需文件路径越出 staging: {relative}") from exc
        if not candidate.is_file() or candidate.stat().st_size <= 0:
            raise JobError("PINNED_FILE_MISSING", f"固定工件缺少必需文件: {relative}")


def _verify_pinned_files(root: Path, files: list[Path], expected_files: list[dict[str, str]]) -> None:
    """Verify the exact pinned weight set and each file's content hash."""
    root = root.resolve()
    actual: dict[str, Path] = {}
    for path in files:
        try:
            relative = path.resolve().relative_to(root).as_posix()
        except ValueError as exc:
            raise JobError("PRESET_PIN_INVALID", "权重文件路径越出 staging") from exc
        actual[relative] = path
    expected_paths = {item["path"] for item in expected_files}
    if set(actual) != expected_paths:
        missing = sorted(expected_paths - set(actual))
        unexpected = sorted(set(actual) - expected_paths)
        raise JobError(
            "PINNED_FILE_SET_MISMATCH",
            f"固定权重文件集合不一致: missing={missing}, unexpected={unexpected}",
        )
    for item in expected_files:
        path = actual[item["path"]]
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        actual_sha256 = digest.hexdigest()
        if actual_sha256 != item["sha256"].lower():
            raise JobError(
                "SHA256_MISMATCH",
                f"固定权重 SHA-256 不匹配: {item['path']}",
            )


def _infer_artifact(target: Path, gguf_path: str = "") -> dict[str, Any]:
    files = _weight_files(target)
    safe_files = [f for f in files if f.name.lower().endswith(_SAFE_SUFFIXES)]
    gguf_files = [f for f in files if f.suffix.lower() == ".gguf"]
    explicit = Path(gguf_path).expanduser().resolve(strict=False) if gguf_path else None
    if not explicit and len(gguf_files) > 1:
        raise JobError("MULTIPLE_GGUF", "发现多个 GGUF 文件；需指定 --gguf-path")
    selected = explicit or (gguf_files[0] if len(gguf_files) == 1 else None)
    model_type = ("both" if safe_files and selected else
                  "safetensors" if safe_files else
                  "gguf" if selected else "")
    if not model_type:
        raise JobError("ARTIFACT_INFER_FAILED", "无法推断模型类型：需 safetensors/bin 或 GGUF 权重")
    artifact = validate_model_artifact(
        model_type, str(target) if safe_files else "",
        str(selected) if selected else "",
    )
    if selected and selected not in files:
        if not selected.is_file():
            raise JobError("GGUF_PATH_INVALID", "显式 GGUF 文件不存在")
        files.append(selected)
        artifact["files"] = [*artifact["safetensors_files"], selected]
    return artifact


def _register(model_id: str, target: Path, summary: dict[str, Any],
              *, gguf_path: str = "", revision: str = "", registry_snapshot=None) -> bool:
    snapshot = registry_snapshot if registry_snapshot is not None else local_store.get_local_model_registry_snapshot(model_id)
    artifact = _infer_artifact(target, gguf_path)
    manifest = summary.get("manifest") or build_manifest(
        target, artifact["files"], model_type=artifact["model_type"],
        revision=revision, source="download-service",
    )
    config = {
        "model_id": model_id,
        "name": model_id,
        "model_type": artifact["model_type"],
        "model_path": artifact["model_path"],
        "gguf_path": artifact["gguf_path"],
        "quantization": "Q4_K_M" if artifact["model_type"] == "gguf" else "fp16",
        "sha256": summary["sha256"],
        "artifact_sha256": manifest["artifact_sha256"],
        "manifest_sha256": manifest["manifest_sha256"],
        "source": "download-service",
        "revision": revision,
        "manifest": manifest,
    }
    return local_store.publish_local_model_registry(model_id, config, expected_revision=snapshot.registry_revision)


# ---------------------------------------------------------------------------
# job 主流程
# ---------------------------------------------------------------------------

def _run_job(job_id: str, *, source: str, target: str, model_id: str,
             preset_id: str, engine: str, quant: str, use_modelscope: bool,
             proxy: str, expected_sha256: str, gguf_path: str,
             models_root: str, allow_cpu: bool,
             revision: str = "", allow_patterns: list[str] | None = None,
             required_files: list[str] | None = None,
             artifact_files: list[dict[str, str]] | None = None,
             progress_cb: Callable[[int, int], None] | None = None,
             registry_snapshot=None) -> None:
    """在调用方线程中执行 job 全流程；异常写回 job 并落盘清理。"""
    registry_snapshot = registry_snapshot if registry_snapshot is not None else local_store.get_local_model_registry_snapshot(model_id)
    staging: Path | None = None
    destination: Path | None = None
    published = False
    registered = False
    try:
        _update_job(job_id, status=STATUS_DOWNLOADING, progress=0.0,
                    downloaded_bytes=0, total_bytes=0, error=None, error_code=None)
        destination = _resolve_target(source, target, models_root)
        if destination.exists():
            raise JobError("TARGET_EXISTS", f"目标已存在，拒绝覆盖安装: {destination}")
        destination.parent.mkdir(parents=True, exist_ok=True)
        staging = Path(tempfile.mkdtemp(prefix=f".{destination.name}.qlh-dl-", dir=destination.parent))
        base_dir = staging
        total_bytes = 0
        downloaded = 0
        seen_files: set[str] = set()

        def _cb(path: Path, current: int, grand_total: int) -> None:
            nonlocal total_bytes, downloaded
            total_bytes = grand_total
            downloaded = current
            if progress_cb:
                progress_cb(downloaded, grand_total)
            _update_job(job_id, downloaded_bytes=current, total_bytes=grand_total,
                        progress=(current / grand_total if grand_total else 0.0))

        # 下载（含 fake/本地目录场景跳过）
        files = _download(source, base_dir, use_modelscope=use_modelscope,
                          proxy=proxy, progress_cb=_cb,
                          revision=revision, allow_patterns=allow_patterns)
        gguf_path = _resolve_staged_gguf(gguf_path, staging)
        _update_job(job_id, status=STATUS_VERIFYING)

        # 校验
        _verify_required_files(staging, required_files)
        summary = _verify(files, expected_sha256 or None)
        if artifact_files:
            _verify_pinned_files(staging, files, artifact_files)
        artifact = _infer_artifact(staging, gguf_path)
        files = artifact["files"]
        manifest = build_manifest(staging, artifact["files"],
                                  model_type=artifact["model_type"],
                                  revision=revision, source="download-service")
        write_manifest(staging, manifest)
        summary.update({
            "artifact_sha256": manifest["artifact_sha256"],
            "manifest_sha256": manifest["manifest_sha256"],
            "manifest": manifest,
            "model_type": artifact["model_type"],
        })

        _update_job(job_id, status=STATUS_REGISTERING, progress=1.0,
                    total_bytes=summary["total_bytes"],
                    downloaded_bytes=summary["total_bytes"])

        # staging -> 目标（原子 rename）
        staging.replace(destination)
        published = True
        registered = _register(
            model_id,
            destination,
            summary,
            gguf_path=_relocate_staged_path(gguf_path, staging, destination),
            revision=revision, registry_snapshot=registry_snapshot,
        )
        if not registered:
            raise JobError("REGISTER_FAILED", f"模型 '{model_id}' 注册失败")
        _update_job(job_id, status=STATUS_READY, progress=1.0,
                    finished_at=_now(), error=None, error_code=None)
    except JobError as exc:
        message = str(exc)
        if published and not registered and destination is not None:
            message = "cleanup_pending: published artifact retained; " + message
        _fail_job(job_id, exc.code, message)
    except Exception as exc:  # noqa: BLE001
        message = str(exc)
        if published and not registered and destination is not None:
            message = "cleanup_pending: published artifact retained; " + message
        _fail_job(job_id, "INTERNAL_ERROR", message)
    finally:
        if staging is not None and staging.exists() and not published:
            shutil.rmtree(staging, ignore_errors=True)


def retain_unregistered_publication(destination: Path) -> dict:
    """Return recovery evidence; published bytes await the R3 supply owner.

    A registry read cannot authorize deletion: bindings include typed/external
    locations, and a concurrent alias may adopt the artifact after that read.
    Staging remains the creating operation's responsibility.
    """
    from qlh_core.bootstrap.model_control import build_model_control
    from qlh_core.adapters.publication_cleanup import publication_evidence
    try:
        location = str(Path(destination).resolve(strict=True))
        evidence = publication_evidence(location)
        cleanup_id = build_model_control().retain_publication(location, evidence=evidence)
        return {"reason_code": "cleanup_pending", "path": location, "retained": True, "cleanup_id": cleanup_id}
    except Exception:
        import logging
        logging.getLogger(__name__).exception("event=publication_cleanup_debt_persist_failed path=%s", destination)
        return {"reason_code": "cleanup_pending", "path": str(destination), "retained": True,
                "cleanup_persisted": False}


def _fail_job(job_id: str, code: str, message: str) -> None:
    if len(message) > 500:
        message = message[:500]
    _update_job(job_id, status=STATUS_FAILED, error_code=code, error=message,
                finished_at=_now())


def create_job(*, source: str = "", target: str = "", model_id: str = "", preset_id: str = "",
               engine: str = "auto", quant: str = "", use_modelscope: bool = False,
               proxy: str = "", expected_sha256: str = "", gguf_path: str = "",
               models_root: str = "", allow_cpu: bool = True,
               progress_cb: Callable[[int, int], None] | None = None,
               executor: Callable[[Callable[[], None]], None] | None = None) -> dict[str, Any]:
    """排队一个下载 job。``executor`` 若提供则将 ``_run_job`` 交给其调度（后台线程池）。

    返回 job dict。幂等：同 source+target+model_id 已有活跃 job 则返回既有 job。
    """
    preset: dict[str, Any] | None = None
    revision = ""
    allow_patterns: list[str] | None = None
    required_files: list[str] | None = None
    artifact_files: list[dict[str, str]] | None = None
    if preset_id:
        preset = _PRESETS_BY_ID.get(preset_id)
        if preset is None:
            raise JobError("PRESET_NOT_FOUND", f"预设 '{preset_id}' 不存在")
        source = _preset_source(preset, use_modelscope)
        model_id = model_id or preset.get("default_model_id", "")
        if engine == "auto":
            engine = preset.get("default_engine", "auto")
        quant = quant or preset.get("default_quant", "")
        expected_sha256 = expected_sha256 or preset.get("expected_sha256", "")
        pin = _load_remote_pin(preset)
        if preset["id"] in _PINNED_PRESET_IDS:
            if pin is None:
                raise JobError("PRESET_PIN_MISSING", f"预设 '{preset['id']}' 没有远程工件 pin")
            revision = pin["revision"]
            allow_patterns = pin["allow_patterns"]
            required_files = pin["required_files"]
            artifact_files = pin["weight_files"]
        # 资源门前置
        rejected: dict[str, str] = {}
        _resource_gate(rejected, preset)
        if rejected:
            raise JobError("RESOURCE_BLOCKED", "；".join(f"{k}:{v}" for k, v in rejected.items()))
    else:
        if not source:
            raise JobError("SOURCE_REQUIRED", "非预设下载必须提供 source")

    root = models_root or _DEFAULT_MODELS_ROOT
    # Local imports historically allow an explicit destination outside the
    # application models directory. Treat that destination's parent as the
    # caller-selected import root, while remote downloads stay application-rooted.
    if not models_root and target and Path(source).expanduser().is_dir():
        explicit_target = Path(target).expanduser()
        if explicit_target.is_absolute():
            root = str(explicit_target.resolve(strict=False).parent)
    default_target = str(preset.get("default_target") or "") if preset else ""
    destination = _resolve_target(
        source,
        target or (str(Path(root) / default_target) if default_target else ""),
        root,
    )
    model_id = model_id or destination.name
    target = str(destination)

    with _LOCK:
        existing = _active_job_exists(source, target, model_id)
        if existing:
            # 返回已有活跃 job（首个匹配）
            for j in list_jobs(limit=100):
                if j["source"] == source and j["target"] == target and j["model_id"] == model_id \
                        and j["status"] in _ACTIVE_STATUSES:
                    return j
            # 若上面兜底没找到（竞态），再创建一个
        job_id = str(uuid.uuid4())
        now = _now()
        row = {
            "job_id": job_id, "status": STATUS_QUEUED, "progress": 0.0,
            "source": source, "target": target, "model_id": model_id,
            "preset_id": preset_id, "engine": engine or "auto", "quant": quant,
            "total_bytes": 0, "downloaded_bytes": 0, "error_code": None,
            "error": None, "created_at": now, "updated_at": now, "finished_at": None,
        }
        _insert_job(row)

    registry_snapshot = local_store.get_local_model_registry_snapshot(model_id)

    def _run() -> None:
        _run_job(
            job_id, source=source, target=target, model_id=model_id,
            preset_id=preset_id, engine=engine, quant=quant,
            use_modelscope=use_modelscope, proxy=proxy, expected_sha256=expected_sha256,
            gguf_path=gguf_path, models_root=root,
            allow_cpu=allow_cpu, revision=revision, allow_patterns=allow_patterns,
            required_files=required_files, artifact_files=artifact_files,
            progress_cb=progress_cb, registry_snapshot=registry_snapshot,
        )

    if executor is not None:
        executor(_run)
    return get_job(job_id) or row


def cancel_job(job_id: str) -> bool:
    """标记取消 job（仅 queued 可取消；执行中的由调用方线程池控制）。"""
    job = get_job(job_id)
    if not job:
        return False
    if job["status"] != STATUS_QUEUED:
        return False
    _update_job(job_id, status=STATUS_CANCELLED, finished_at=_now())
    return True


def delete_job(job_id: str) -> bool:
    """删除 job 记录（不删磁盘模型，仅清理记录）。"""
    with local_store._write_connection() as connection:
        result = connection.execute(
            "DELETE FROM model_download_jobs WHERE job_id=?", (job_id,)
        )
        return int(result.rowcount) > 0
