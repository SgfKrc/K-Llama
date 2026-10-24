"""MODEL-SUPPLY-01: receive/build and register assets without loading an engine.

The scheduler owns lifecycle hooks. Call ``cancel()`` when leaving distributed
mode or losing the master connection; every operation also checks those fences
itself. ``current.json`` points at a verified public manifest, not engine state.
"""
from __future__ import annotations

from contextlib import ExitStack, contextmanager, nullcontext
from copy import deepcopy
from dataclasses import dataclass, field
import hashlib
import importlib
import json
import logging
import os
from pathlib import Path, PurePosixPath, PureWindowsPath
import re
import sys
import threading
from typing import Any, Callable
import uuid
import weakref

_SHA = re.compile(r"[0-9a-f]{64}")
_ID = re.compile(r"[A-Za-z0-9_.:/-]{1,256}")
_MAX_JSON_BYTES = 2 * 1024 * 1024
_MAX_CANCELLED_OPERATIONS = 256
_TRANSFER_COUNTERS = ("downloaded_bytes", "actual_downloaded_bytes", "reused_bytes", "resumed_bytes")
_STRATEGY_REASONS = frozenset({
    "SUPPLY_CACHE_REUSED", "SUPPLY_LOCAL_CUT_PREFERRED", "SUPPLY_DISTRIBUTE_REQUESTED",
    "SUPPLY_LOCAL_SOURCE_UNAVAILABLE", "SUPPLY_LOCAL_CUT_TOOL_UNAVAILABLE",
    "SUPPLY_LOCAL_CUT_DISK_INSUFFICIENT", "SUPPLY_LOCAL_CUT_BUDGET_EXCEEDED",
    "SUPPLY_LOCAL_CUT_UNSUPPORTED", "SUPPLY_NODE_CUTTER_UNAVAILABLE",
})
_LOG = logging.getLogger(__name__)
# Keep product pointer readers out of atomic publish/restore, including bounded
# Windows retries. WSL UNC can reject rename even with FILE_SHARE_DELETE.
# Never hold this lock while hashing artifacts or refreshing capabilities.
_POINTER_IO_LOCK = threading.RLock()
_SELECTION_CACHE_LOCK = threading.Lock()
_SELECTION_CACHE: dict[str, tuple[tuple, dict]] = {}
_SELECTION_OWNERS_LOCK = threading.Lock()
_SELECTION_OWNERS: dict[str, weakref.WeakSet] = {}


class SupplyError(ValueError):
    def __init__(self, code: str):
        self.reason_code = code
        super().__init__(code)


def _default_root() -> Path:
    from config import STATE_DIR
    return Path(STATE_DIR) / "model-supply" / "received"


def _runtime_distributed() -> bool:
    import config
    return getattr(config, "RUN_MODE", "distributed") == "distributed"


def _json(path: Path) -> dict:
    if path.stat().st_size > _MAX_JSON_BYTES:
        raise SupplyError("invalid_manifest")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise SupplyError("invalid_manifest")
    return value


def _read_pointer(path: Path) -> bytes:
    with _POINTER_IO_LOCK:
        if path.is_symlink() or path.stat().st_size > _MAX_JSON_BYTES:
            raise SupplyError("invalid_manifest")
        return path.read_bytes()


def _log_registration_error(op, phase: str, exc: Exception) -> None:
    # Registration only touches validated local artifacts. Transport exceptions
    # remain unlogged because they may contain credential-bearing URLs.
    _LOG.error("Model supply %s failed (operation=%s, node=%s)",
               phase, op.operation_id, op.node_id, exc_info=True,
               extra={"supply_phase": phase, "supply_operation_id": op.operation_id,
                      "supply_node_id": op.node_id, "supply_errno": getattr(exc, "errno", None),
                      "supply_winerror": getattr(exc, "winerror", None)})


def _sha(value: Any) -> bool:
    return isinstance(value, str) and _SHA.fullmatch(value) is not None


def _relative(value: Any) -> str:
    if not isinstance(value, str) or not value or "\\" in value or ":" in value:
        raise SupplyError("unsafe_path")
    path = PurePosixPath(value)
    if path.is_absolute() or any(part in {"", ".", ".."} for part in value.split("/")):
        raise SupplyError("unsafe_path")
    if any(ord(char) < 32 for char in value):
        raise SupplyError("unsafe_path")
    return value


def _within(path: Path, parent: Path) -> Path:
    resolved = path.resolve()
    if not resolved.is_relative_to(parent.resolve()):
        raise SupplyError("unsafe_path")
    return resolved


def _public(value: Any) -> None:
    """Never persist credentials, download URLs or host filesystem paths."""
    if isinstance(value, dict):
        for key, item in value.items():
            if not isinstance(key, str) or key.lower() in {
                "token", "authorization", "access_token", "base_url", "url",
                "local_path", "source_path", "source_model_path", "absolute_path",
            }:
                raise SupplyError("invalid_manifest")
            _public(item)
    elif isinstance(value, list):
        for item in value:
            _public(item)
    elif isinstance(value, str):
        if PurePosixPath(value).is_absolute() or PureWindowsPath(value).is_absolute():
            raise SupplyError("unsafe_path")


def _manifest(value: dict) -> dict:
    # Builders attach a private local_files map. It never becomes part of the
    # registered/public contract or a TCP response.
    if not isinstance(value, dict):
        raise SupplyError("invalid_manifest")
    public = {key: item for key, item in value.items()
              if isinstance(key, str) and key not in {"local_files", "private"}
              and not key.startswith("_")}
    _public(public)
    try:
        from model_supply_assets import validate_supply_manifest
        return validate_supply_manifest(public)
    except ImportError:
        raise SupplyError("supply_assets_unavailable") from None
    except ValueError as exc:
        raise SupplyError(_error_code(exc, "invalid_manifest")) from None


_ASSET_CODES = frozenset({
    "invalid_manifest", "unsafe_manifest", "invalid_identity", "unsafe_path",
    "artifact_digest_mismatch", "manifest_digest_mismatch", "file_unavailable",
    "file_size_mismatch", "file_digest_mismatch", "unexpected_file",
    "insufficient_disk", "identity_mismatch", "invalid_artifact", "invalid_gguf",
    "layer_range_mismatch", "layer_bytes_mismatch", "source_missing",
    "source_digest_mismatch", "tokenizer_digest_mismatch", "cut_tool_unavailable",
    "cut_timeout", "cut_failed", "cut_manifest_mismatch", "tensor_digest_mismatch",
    "artifact_io_error", "unsupported_range", "invalid_safetensors",
    "unsupported_architecture", "cancelled", "download_failed", "http_error",
    "authorization_failed", "download_size_mismatch", "download_digest_mismatch",
    "native_manifest_mismatch", "http_redirect_rejected", "http_range_mismatch",
    "http_size_mismatch", "invalid_base_url", "invalid_token", "download_incomplete",
})


def _error_code(exc: Exception, fallback: str) -> str:
    if isinstance(exc, SupplyError):
        return exc.reason_code
    code = getattr(exc, "code", None)
    return code if isinstance(code, str) and code in _ASSET_CODES else fallback


def _change_time(path: Path, stat) -> int:
    if os.name != "nt":
        return stat.st_ctime_ns
    # Python 3.12 Windows st_ctime is still creation time. NTFS ChangeTime
    # detects byte changes even when an application restores LastWriteTime.
    import ctypes
    from ctypes import wintypes

    class FileBasicInfo(ctypes.Structure):
        _fields_ = [(name, ctypes.c_longlong) for name in (
            "CreationTime", "LastAccessTime", "LastWriteTime", "ChangeTime")]
        _fields_.append(("FileAttributes", wintypes.DWORD))

    kernel = ctypes.WinDLL("kernel32", use_last_error=True)
    create = kernel.CreateFileW
    create.argtypes = [wintypes.LPCWSTR, wintypes.DWORD, wintypes.DWORD,
                       ctypes.c_void_p, wintypes.DWORD, wintypes.DWORD, wintypes.HANDLE]
    create.restype = wintypes.HANDLE
    query = kernel.GetFileInformationByHandleEx
    query.argtypes = [wintypes.HANDLE, ctypes.c_int, ctypes.c_void_p, wintypes.DWORD]
    query.restype = wintypes.BOOL
    close = kernel.CloseHandle
    close.argtypes, close.restype = [wintypes.HANDLE], wintypes.BOOL
    filename = str(path.resolve())
    if not filename.startswith("\\\\?\\"):
        filename = "\\\\?\\UNC\\" + filename[2:] if filename.startswith("\\\\") else "\\\\?\\" + filename
    handle = create(filename, 0, 7, None, 3, 0x02000000, None)
    if handle == wintypes.HANDLE(-1).value:
        raise ctypes.WinError(ctypes.get_last_error())
    try:
        info = FileBasicInfo()
        if not query(handle, 0, ctypes.byref(info), ctypes.sizeof(info)):
            raise ctypes.WinError(ctypes.get_last_error())
        return int(info.ChangeTime) * 100
    finally:
        close(handle)


def _file_stamp(path: Path) -> tuple:
    stat = path.stat()
    return (stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns, stat.st_ino,
            _change_time(path, stat))


def _file_sha(path: Path, cancelled=None) -> str:
    digest = hashlib.sha256()
    before = _file_stamp(path)
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            if cancelled is not None and cancelled():
                raise SupplyError("cancelled")
            digest.update(chunk)
    if before != _file_stamp(path):
        raise SupplyError("artifact_changed")
    return digest.hexdigest()


# ★ 2026-10-24（#125）：整模 sha256 是必需的源身份证明，但**不必每轮重算**。
#   实测清点（inventory）阶段对 5.78 GB 的 GGUF 做全量 sha256 需约 200 s，占整个
#   供给近三分之一；#119 加的进程内摘要缓存**重启即失效、跨 op 也不复用** ⇒ 每次
#   供给都重付一遍。这里以 `_file_stamp`（size/mtime_ns/ctime_ns/ino/change_time）
#   作为失效判据，把结果**持久化**到 worker root，跨重启与跨供给复用。
_SOURCE_DIGEST_CACHE_NAME = "source-digests.json"
_source_digest_cache: dict | None = None


def _load_source_digest_cache(root: Path) -> dict:
    global _source_digest_cache
    if _source_digest_cache is not None:
        return _source_digest_cache
    import json
    cache: dict = {}
    try:
        payload = json.loads((root / _SOURCE_DIGEST_CACHE_NAME).read_text(encoding="utf-8"))
        entries = payload.get("entries") if isinstance(payload, dict) else None
        if isinstance(payload, dict) and payload.get("schema_version") == 1 and isinstance(entries, dict):
            cache = {key: value for key, value in entries.items() if isinstance(value, dict)}
    except (OSError, ValueError):
        cache = {}
    _source_digest_cache = cache
    return cache


def _store_source_digest_cache(root: Path) -> None:
    import json
    import os
    cache = _source_digest_cache
    if cache is None:
        return
    try:
        root.mkdir(parents=True, exist_ok=True)
        temp = root / (_SOURCE_DIGEST_CACHE_NAME + ".tmp")
        temp.write_text(json.dumps({"schema_version": 1, "entries": cache},
                                   ensure_ascii=True, sort_keys=True), encoding="utf-8")
        os.replace(temp, root / _SOURCE_DIGEST_CACHE_NAME)
    except OSError:
        pass


def _file_sha_cached(root: Path, path: Path) -> str:
    """Full-file sha256, reusing the persisted digest while the file is unchanged."""
    cache = _load_source_digest_cache(root)
    key = str(path)
    try:
        stamp = list(_file_stamp(path))
    except OSError:
        return _file_sha(path)
    entry = cache.get(key)
    if (isinstance(entry, dict) and entry.get("stamp") == stamp
            and isinstance(entry.get("sha256"), str) and _sha(entry["sha256"])):
        return str(entry["sha256"])
    digest = _file_sha(path)
    cache[key] = {"stamp": stamp, "sha256": digest}
    _store_source_digest_cache(root)
    return digest


def _verify_files(manifest: dict, directory: Path, cancelled=None) -> Path:
    directory = directory.resolve(strict=True)
    paths = []
    for entry in manifest["files"]:
        path = _within(directory / _relative(entry["path"]), directory)
        if not path.is_file() or path.stat().st_size != entry.get("size_bytes", entry.get("size")):
            raise SupplyError("artifact_size_mismatch")
        if _file_sha(path, cancelled) != entry["sha256"]:
            raise SupplyError("artifact_sha256_mismatch")
    try:
        from model_supply_assets import _verify_segment
        _verify_segment(directory, manifest)
    except ImportError:
        raise SupplyError("supply_assets_unavailable") from None
    except ValueError as exc:
        raise SupplyError(_error_code(exc, "artifact_metadata_mismatch")) from None
    weights = [directory / entry["path"] for entry in manifest["files"] if entry["kind"] == "weights"]
    return weights[0] if manifest["format"] == "gguf" else directory


def _selection_key(root: Path, pointer_bytes: bytes, directory: Path, manifest: dict) -> tuple:
    from model_supply_assets import _safe_path
    names = {item["path"] for item in manifest["files"]} | {"supply.manifest.json"}
    # Directory membership participates too: adding an unlisted shard must not
    # turn a previously checked Safetensors index into a different asset.
    actual = {path.relative_to(directory).as_posix() for path in directory.rglob("*") if path.is_file()}
    if actual != names:
        raise SupplyError("unexpected_file")
    fingerprint = []
    for name in sorted(names):
        path = directory / name if name == "supply.manifest.json" else _safe_path(directory, name)
        if path.is_symlink() or getattr(path, "is_junction", lambda: False)():
            raise SupplyError("unsafe_path")
        path = _within(path, directory)
        fingerprint.append((str(path), *_file_stamp(path)))
    return (hashlib.sha256(pointer_bytes).hexdigest(), tuple(fingerprint))


def _remember_selection(root: Path, pointer_bytes: bytes, directory: Path, manifest: dict, model_path: Path) -> None:
    key = _selection_key(root, pointer_bytes, directory, manifest)
    value = {"model_path": str(model_path), "directory": str(directory),
             "manifest": deepcopy(manifest), "manifest_sha256": manifest["manifest_sha256"],
             "artifact_sha256": manifest["artifact_sha256"]}
    with _SELECTION_CACHE_LOCK:
        if len(_SELECTION_CACHE) >= 64:
            _SELECTION_CACHE.clear()
        _SELECTION_CACHE[str(root.resolve())] = (key, value)


def _selection_distributed(root: Path) -> bool:
    if not _runtime_distributed():
        return False
    # EngineHost has no scheduler reference. Reuse the owning worker's live
    # participation check when present; this is a weak lookup, not mode state.
    with _SELECTION_OWNERS_LOCK:
        owners = list(_SELECTION_OWNERS.get(str(root.resolve()), ()))
    return all(owner._distributed() and not owner._registration_blocked for owner in owners)


def get_selected_model(root: Path | None = None) -> dict | None:
    """Return the current verified registration without rehashing unchanged GBs.

    Pointer bytes plus every file's resolved path, size, mtime and ctime fence
    the process cache. No engine is loaded. Instance access additionally checks
    the scheduler's runtime participation switch.
    """
    root = Path(root) if root is not None else _default_root()
    if not _selection_distributed(root):
        return None
    try:
        current = root / "current.json"
        if current.is_symlink() or current.stat().st_size > _MAX_JSON_BYTES:
            return None
        pointer_bytes = _read_pointer(current)
        pointer = json.loads(pointer_bytes)
        if not isinstance(pointer, dict) or pointer.get("kind") != "qlh.model_supply.current.v1":
            return None
        directory = _within(root.parent / _relative(pointer.get("directory")), root.parent)
        manifest_path = _within(directory / _relative(pointer.get("manifest")), directory)
        if manifest_path.name != "supply.manifest.json":
            return None
        manifest = _manifest(_json(manifest_path))
        if (pointer.get("manifest_sha256") != manifest["manifest_sha256"]
                or pointer.get("artifact_sha256") != manifest["artifact_sha256"]):
            return None
        # Serialize cache misses so concurrent first hello/stage discovery does
        # not fan out several full hashes of the same weight file.
        with _SELECTION_CACHE_LOCK:
            key = _selection_key(root, pointer_bytes, directory, manifest)
            cached = _SELECTION_CACHE.get(str(root.resolve()))
            if cached is not None and cached[0] == key:
                result = deepcopy(cached[1])
            else:
                model_path = _verify_files(manifest, directory)
                if key != _selection_key(root, pointer_bytes, directory, manifest):
                    return None
                result = {"model_path": str(model_path), "directory": str(directory),
                          "manifest": manifest, "manifest_sha256": manifest["manifest_sha256"],
                          "artifact_sha256": manifest["artifact_sha256"]}
                if len(_SELECTION_CACHE) >= 64:
                    _SELECTION_CACHE.clear()
                _SELECTION_CACHE[str(root.resolve())] = (key, deepcopy(result))
        if _read_pointer(current) != pointer_bytes or not _selection_distributed(root):
            return None
        return result
    except (OSError, ValueError, TypeError, KeyError, RuntimeError):
        return None


def selected_gguf_path(root: Path | None = None) -> str:
    """Return a safe GGUF registration in distributed runtime, without loading."""
    if not _runtime_distributed():
        return ""
    selected = get_selected_model(root)
    path = selected["model_path"] if selected else ""
    return path if Path(path).suffix.lower() == ".gguf" else ""


@dataclass(eq=False)
class _Operation:
    operation_id: str
    node_id: str
    client: Any = field(repr=False)
    epoch: Any = None
    generation: Any = None
    manifest: dict = field(default_factory=dict)
    strategy: str = "distribute"
    strategy_reason_code: str = ""
    allow_fallback: bool = False
    cut_timeout_seconds: int = 300
    token: str = field(default="", repr=False)
    api_port: int = 8000
    local: bool = False
    finished: bool = False
    transfer_counts: dict[str, int] = field(
        default_factory=lambda: dict.fromkeys(_TRANSFER_COUNTERS, 0), repr=False)
    committed: dict | None = field(default=None, repr=False)
    previous_pointer: bytes | None = field(default=None, repr=False)
    cancelled: threading.Event = field(default_factory=threading.Event, repr=False)
    cancelled_callback: Callable[[], bool] | None = field(default=None, repr=False)


class ModelSupplyWorker:
    def __init__(self, scheduler, root: Path | None = None):
        self.scheduler = scheduler
        self.root = (Path(root) if root is not None else _default_root()).absolute()
        self._lock = threading.Lock()
        self._execution_lock = threading.RLock()
        self._inventory_lock = threading.Lock()
        self._registration_blocked = False
        self._active: _Operation | None = None
        self._pending: _Operation | None = None
        self._runner: threading.Thread | None = None
        self._sources: dict[tuple[str, str, str], Path] = {}
        # Retain the captured client to prevent object-id reuse. Tombstones are
        # bounded in memory and scoped to its registration/connection fences.
        self._cancelled_operations: dict[tuple, Any] = {}
        with _SELECTION_OWNERS_LOCK:
            for stale in [key for key, owners in _SELECTION_OWNERS.items() if not owners]:
                _SELECTION_OWNERS.pop(stale, None)
            _SELECTION_OWNERS.setdefault(str(self.root), weakref.WeakSet()).add(self)

    def _distributed(self) -> bool:
        return bool(_runtime_distributed() and self.scheduler.get_distributed_inference_enabled())

    def _context_valid(self, op: _Operation) -> bool:
        if not self._distributed():
            return False
        if op.local:
            role = getattr(self.scheduler, "_effective_role", None)
            return callable(role) and role() == "master" and op.node_id == "master"
        if str(self.scheduler.get_effective_node_id()) != op.node_id:
            return False
        client = op.client
        return bool(client is not None and getattr(self.scheduler, "_tcp_client", None) is client
                    and getattr(client, "is_registered", False)
                    and getattr(client, "registration_epoch", None) == op.epoch
                    and getattr(client, "_connection_generation", None) == op.generation)

    def _current(self, op: _Operation, *, allow_cancelled=False) -> bool:
        with self._lock:
            valid = self._active is op and (allow_cancelled or not op.cancelled.is_set())
        return valid and self._context_valid(op)

    def _cancelled(self, op: _Operation) -> bool:
        if ((op.cancelled_callback is not None and op.cancelled_callback())
                or not self._current(op)):
            op.cancelled.set()
        return op.cancelled.is_set()

    def _check(self, op: _Operation) -> None:
        if self._cancelled(op):
            raise SupplyError("cancelled")

    @staticmethod
    def _cancel_key(op: _Operation) -> tuple:
        return (op.local, id(op.client), op.epoch, op.generation, op.operation_id)

    def _remember_cancel_locked(self, op: _Operation) -> None:
        key = self._cancel_key(op)
        self._cancelled_operations.pop(key, None)
        self._cancelled_operations[key] = op.client
        while len(self._cancelled_operations) > _MAX_CANCELLED_OPERATIONS:
            self._cancelled_operations.pop(next(iter(self._cancelled_operations)))

    def _cancel_operation(self, op: _Operation) -> None:
        with self._lock:
            self._remember_cancel_locked(op)
            if self._active is not None and self._cancel_key(self._active) == self._cancel_key(op):
                self._active.cancelled.set()

    def cancel(self, operation_id: str | None = None) -> None:
        """Cancel current work; explicit IDs also fence late admission in this scope."""
        captured = getattr(self.scheduler, "_tcp_client", None)
        op = _Operation(operation_id, "", captured,
                        getattr(captured, "registration_epoch", None),
                        getattr(captured, "_connection_generation", None), local=captured is None)
        with self._lock:
            if operation_id is not None:
                self._remember_cancel_locked(op)
            if self._active is not None and (operation_id is None or self._active.operation_id == operation_id):
                if operation_id is not None:
                    self._remember_cancel_locked(self._active)
                self._active.cancelled.set()

    def _send(self, op: _Operation, payload: dict, *, inventory=False, terminal=False) -> None:
        valid = self._context_valid(op) if inventory else self._current(op, allow_cancelled=terminal)
        if not valid or op.client is None:
            return
        if op.token and op.token in json.dumps(payload, ensure_ascii=True):
            return
        try:
            from transport_port import MessageType
            op.client.send_data(payload, MessageType.MODEL_SUPPLY)
        except Exception:
            # Transport errors may contain a credential-bearing URL. Never echo
            # or log their message, and never redirect to a replacement client.
            op.cancelled.set()

    def _reply(self, op: _Operation, status: str, code: str = "") -> dict:
        manifest = op.manifest if isinstance(op.manifest, dict) else {}
        hashes = {key: value if _sha(value) and value != op.token else ""
                  for key in ("manifest_sha256", "artifact_sha256", "source_model_sha256")
                  for value in [manifest.get(key)]}
        span = manifest.get("layer_range")
        if not isinstance(span, list) or len(span) != 2 or any(type(item) is not int for item in span):
            span = []
        return {"action": "result", "operation_id": op.operation_id, "node_id": op.node_id,
                "status": status, "reason_code": code, "layer_range": span,
                "strategy": op.strategy if op.strategy in {"distribute", "local_cut", "reuse"} else "",
                "strategy_reason_code": op.strategy_reason_code if op.strategy_reason_code in _STRATEGY_REASONS else "",
                **hashes, **op.transfer_counts}

    def handle(self, payload: dict, client=None) -> None:
        """Accept a MODEL_SUPPLY data object (or outer envelope) without I/O."""
        if not isinstance(payload, dict):
            return
        if "data" in payload:
            if payload.get("type", payload.get("message_type", "model_supply")) != "model_supply":
                return
            payload = payload["data"]
        if not isinstance(payload, dict):
            return
        operation_id = payload.get("operation_id")
        if not isinstance(operation_id, str) or not _ID.fullmatch(operation_id):
            return
        captured = client if client is not None else getattr(self.scheduler, "_tcp_client", None)
        op = _Operation(operation_id, str(self.scheduler.get_effective_node_id()), captured,
                        getattr(captured, "registration_epoch", None),
                        getattr(captured, "_connection_generation", None))
        if not self._context_valid(op):
            return
        action = payload.get("action")
        if action == "cancel":
            self._cancel_operation(op)
            return
        if action == "inventory":
            threading.Thread(target=self._inventory_reply, args=(op, payload.get("model_id")),
                             name="model-supply-inventory", daemon=True).start()
            return
        if action != "ensure":
            return
        # Copy only protocol fields. Caller mutation and credentials in unknown
        # fields cannot reach a later operation, manifest or response.
        op.manifest = deepcopy(payload.get("manifest", {}))
        op.strategy = payload.get("strategy", "")
        reason = payload.get("strategy_reason_code")
        if reason in _STRATEGY_REASONS:
            op.strategy_reason_code = reason
        op.allow_fallback = payload.get("allow_fallback") is True
        budget = payload.get("local_cut_timeout_seconds", 300)
        op.cut_timeout_seconds = budget if type(budget) is int and 1 <= budget <= 3600 else 300
        token = payload.get("token", "")
        op.token = token if isinstance(token, str) else ""
        op.api_port = payload.get("api_port", 8000)
        if payload.get("node_id") != op.node_id:
            return
        with self._lock:
            if self._cancel_key(op) in self._cancelled_operations:
                return
            previous = self._active
            if previous is not None:
                if (previous.operation_id == operation_id and previous.client is captured
                        and previous.epoch == op.epoch and previous.generation == op.generation
                        and not previous.finished and not previous.cancelled.is_set()):
                    return
                previous.cancelled.set()
            self._active = self._pending = op
            if self._runner is None:
                self._runner = threading.Thread(target=self._run, name="model-supply-worker", daemon=True)
                self._runner.start()

    def _inventory_reply(self, op: _Operation, model_id: str | None = None) -> None:
        try:
            # Keep legacy callers (including zero-argument inventory hooks) valid.
            result = self.inventory() if model_id is None else self.inventory(model_id=model_id)
            result.update(action="inventory_result", operation_id=op.operation_id, node_id=op.node_id)
            self._send(op, result, inventory=True)
        except Exception:
            self._send(op, {"action": "inventory_result", "operation_id": op.operation_id,
                           "node_id": op.node_id, "distributed": self._distributed(),
                           "disk_free_bytes": 0, "memory_available_bytes": 0,
                           "local_cut": False, "engines": [], "execution_device": "cpu",
                           "sources": [], "reason_code": "inventory_failed"}, inventory=True)

    def _source_candidates(self):
        """Trusted local registry/discovery only; no request-supplied paths."""
        from model_config import get_builtin_models, resolve_model_path
        from local_model_assets import discover_local_model_assets, _app_root
        import local_store
        rows = local_store.get_local_experimental_models()
        seen = set()
        # Local user registrations can override a builtin's source location.
        for row in rows:
            source_id, configured = row.get("model_id"), row.get("gguf_path")
            if isinstance(source_id, str) and _ID.fullmatch(source_id) and isinstance(configured, str) and configured:
                path = Path(resolve_model_path(configured)).resolve()
                if (source_id, path) not in seen:
                    seen.add((source_id, path))
                    yield source_id, path
        for model in get_builtin_models():
            if getattr(model, "gguf_path", ""):
                path = Path(resolve_model_path(model.gguf_path)).resolve()
                if (model.model_id, path) not in seen:
                    seen.add((model.model_id, path))
                    yield model.model_id, path
        app_root = _app_root()
        for asset in discover_local_model_assets(verify_hashes=False)["assets"]:
            if asset.get("gguf_path"):
                path = (app_root / asset["gguf_path"]).resolve()
                if (asset["model_id"], path) not in seen:
                    seen.add((asset["model_id"], path))
                    yield asset["model_id"], path

    def inventory(self, model_id: str | None = None) -> dict:
        """Verify only the selected source identity, or all sources for legacy callers."""
        import shutil
        import config
        from torch_runtime import torch_available
        from local_model_assets import _canonical_id
        from relay_segment_info import read_gguf_layer_info, read_gguf_tokenizer_sha256
        engines = []
        engine = getattr(config, "INFERENCE_ENGINE", "")
        if engine in {"llama_cpp", "gguf", "auto"}:
            engines.append("llama_cpp")
        else:
            try:
                from keep_head_shim import resolve_keep_head_shim
                shim = resolve_keep_head_shim()
                if importlib.util.find_spec("llama_cpp") is not None or (shim and Path(shim).is_file()):
                    engines.append("llama_cpp")
            except (ImportError, ValueError):
                pass
        if torch_available():
            engines.append("pytorch")
        existing = self.root
        while not existing.exists() and existing != existing.parent:
            existing = existing.parent
        free = int(shutil.disk_usage(existing).free)
        sources = []
        verified = {}
        selected_id = (_canonical_id(model_id)
                       if isinstance(model_id, str) and _ID.fullmatch(model_id) else "")
        with self._inventory_lock:
            for source_id, path in self._source_candidates():
                # Discovery pairs a registered *-gguf asset with its canonical
                # model ID. Compare those source IDs before any weight I/O;
                # filenames and paths never supply model identity.
                if model_id is not None and (not selected_id or _canonical_id(source_id) != selected_id):
                    continue
                try:
                    if not path.is_file() or path.suffix.lower() != ".gguf":
                        continue
                    # A cut segment is an artifact, not a complete local source.
                    if path.with_suffix(".manifest.json").exists() or path.with_suffix(path.suffix + ".manifest.json").exists():
                        continue
                    info = read_gguf_layer_info(path)
                    tokenizer = read_gguf_tokenizer_sha256(path)
                    if not info or not _sha(tokenizer):
                        continue
                    # ★ 2026-10-24（#125）：清点阶段对整模做全量 sha256 约需 200 s
                    #   （5.78 GB），改为带**持久化**缓存的摘要：文件未变即复用，
                    #   跨重启与跨供给都生效。
                    digest = _file_sha_cached(self.root, path)
                    verified[(source_id, digest, tokenizer)] = path
                    sources.append({"source_model_id": source_id, "source_model_sha256": digest,
                                    "tokenizer_sha256": tokenizer, "size_bytes": path.stat().st_size})
                except (OSError, ValueError):
                    continue
            with self._lock:
                self._sources = verified
        # Tool availability is independent of the selected execution engine.
        try:
            assets = importlib.import_module("model_supply_assets")
            capability = getattr(assets, "local_cut_available", None)
            if callable(capability):
                local_cut = bool(capability())
            else:
                import sys
                local_cut = bool(importlib.util.find_spec("gguf") is not None and (
                    getattr(sys, "frozen", False) or
                    (Path(__file__).resolve().parent.parent / "scripts" / "cut_layers.py").is_file()))
        except (ImportError, OSError, ValueError):
            local_cut = False
        role = getattr(self.scheduler, "_effective_role", None)
        node_id = "master" if callable(role) and role() == "master" else str(self.scheduler.get_effective_node_id())
        try:
            import psutil
            virtual = psutil.virtual_memory()
            available_memory = int(virtual.available)
            # ★ 2026-10-23（#117）：主机内存的**物理总量**也要上报 —— 主节点的
            #   offload 承载上限按"总量 × HOST_RAM_MAX_FRACTION"封顶（默认 60%），
            #   而不是把当前可用内存整块交出去，否则一次供给就能吃掉大半 RAM。
            total_memory = int(virtual.total)
        except (ImportError, OSError):
            available_memory = 0
            total_memory = 0
        runtime = None
        if "pytorch" in engines:
            try:
                from model_module import _select_layer_runtime
                from torch_runtime import require_torch
                torch = require_torch()
                device, dtype = _select_layer_runtime()
                is_cuda = str(device).startswith("cuda")
                runtime = {"execution_device": "cuda" if is_cuda else "cpu",
                           "target_element_size": torch.empty((), dtype=dtype).element_size(),
                           "available_bytes": max(0, available_memory),
                           "materializations": (["safetensors_key_filtered", "safetensors_layer_offload"]
                                                if is_cuda else ["safetensors_key_filtered"])}
                if is_cuda:
                    from cuda_memory_budget import sample_cuda_memory_budget
                    runtime.update(sample_cuda_memory_budget(torch, device))
            except (ImportError, OSError, RuntimeError):
                # The engine remains discoverable, but a new supply plan cannot
                # admit it without a real runtime dtype and memory budget.
                runtime = None
        return {"node_id": node_id, "distributed": self._distributed(),
                "disk_free_bytes": max(0, free), "memory_available_bytes": max(0, available_memory),
                "memory_total_bytes": max(0, total_memory),
                "local_cut": local_cut, "engines": engines,
                "execution_device": str(getattr(config, "DEVICE", "cpu")), "pytorch_runtime": runtime,
                "sources": sources}

    @staticmethod
    def _record_transfer_counts(op: _Operation, data: Any) -> None:
        if isinstance(data, dict):
            # The receiver reports cumulative snapshots for this call. Adding
            # successive events would double count files and resumed prefixes.
            for key in _TRANSFER_COUNTERS:
                item = data.get(key)
                if type(item) is int and item >= 0:
                    op.transfer_counts[key] = item

    def _progress(self, op: _Operation, value=None, **kwargs) -> None:
        data = value if isinstance(value, dict) else kwargs
        result = {"action": "progress", "operation_id": op.operation_id, "node_id": op.node_id}
        # Only numbers and a fixed vocabulary can cross the credential boundary.
        for key in ("received_bytes", *_TRANSFER_COUNTERS, "completed_bytes", "total_bytes", "files_completed", "files_total"):
            item = data.get(key)
            if type(item) is int and item >= 0:
                result[key] = item
        self._record_transfer_counts(op, result)
        phase = data.get("phase", data.get("stage"))
        if isinstance(phase, str) and phase in {"receiving", "building", "verifying", "registering", "download", "verify", "build"}:
            result["phase"] = phase
        if op.strategy in {"distribute", "local_cut", "reuse"}:
            result["strategy"] = op.strategy
        if op.strategy_reason_code in _STRATEGY_REASONS:
            result["strategy_reason_code"] = op.strategy_reason_code
        self._send(op, result)

    def _run(self) -> None:
        while True:
            with self._lock:
                op, self._pending = self._pending, None
                if op is None:
                    self._runner = None
                    return
            with self._execution_lock:
                self._ensure(op)

    @staticmethod
    def _directory(result: Any, fallback: Path) -> Path:
        if isinstance(result, dict):
            local_files = result.get("local_files")
            if isinstance(local_files, dict) and local_files:
                first_relative, first_path = next(iter(local_files.items()))
                path = Path(first_path)
                for _ in PurePosixPath(_relative(first_relative)).parts:
                    path = path.parent
                if any((path / _relative(relative)).resolve() != Path(local).resolve()
                       for relative, local in local_files.items()):
                    raise SupplyError("unsafe_path")
                return path
            for key in ("directory", "root", "artifact_path", "model_path", "path"):
                if result.get(key):
                    result = result[key]
                    break
            else:
                result = fallback
        if result is None:
            result = fallback
        path = Path(result)
        return path.parent if path.is_file() else path

    def _ensure(self, op: _Operation) -> None:
        stage = "invalid_manifest"
        try:
            self._check(op)
            op.manifest = _manifest(op.manifest)
            manifest = op.manifest
            if op.token and op.token in json.dumps(manifest, ensure_ascii=True):
                raise SupplyError("invalid_manifest")
            for key, expected in (("operation_id", op.operation_id), ("node_id", op.node_id)):
                if key in manifest and manifest[key] != expected:
                    raise SupplyError("manifest_identity_mismatch")
            if op.strategy not in {"distribute", "local_cut"}:
                raise SupplyError("invalid_strategy")
            try:
                assets = importlib.import_module("model_supply_assets")
            except ImportError:
                raise SupplyError("supply_assets_unavailable") from None
            self.root.mkdir(parents=True, exist_ok=True)
            if op.strategy == "local_cut":
                # A cached exact inventory wins before source hashing/cutting.
                cached = assets.reuse_segment(manifest, self.root,
                    cancelled=lambda: self._cancelled(op))
                if cached is not None:
                    result = cached
                    op.strategy = "reuse"
                    op.strategy_reason_code = "SUPPLY_CACHE_REUSED"
                    self._record_transfer_counts(op, cached)
                    directory = _within(self._directory(cached, self.root), self.root)
                else:
                    stage = "local_cut_failed"
                    op.strategy_reason_code = "SUPPLY_LOCAL_CUT_PREFERRED"
                    try:
                        directory = self._local_cut(op, assets)
                    except Exception as exc:
                        fallback = {
                            "local_source_unavailable": "SUPPLY_LOCAL_SOURCE_UNAVAILABLE",
                            "cut_tool_unavailable": "SUPPLY_LOCAL_CUT_TOOL_UNAVAILABLE",
                            "insufficient_disk": "SUPPLY_LOCAL_CUT_DISK_INSUFFICIENT",
                            "cut_timeout": "SUPPLY_LOCAL_CUT_BUDGET_EXCEEDED",
                            "unsupported_local_cut": "SUPPLY_LOCAL_CUT_UNSUPPORTED",
                            "unsupported_range": "SUPPLY_LOCAL_CUT_UNSUPPORTED",
                        }.get(_error_code(exc, stage))
                        if not op.allow_fallback or fallback is None:
                            raise
                        self._check(op)
                        op.strategy = "distribute"
                        op.strategy_reason_code = fallback
                        stage = "transfer_failed"
                        self._progress(op, {"phase": "receiving"})
                        directory = self._receive(op, assets)
            else:
                if not op.strategy_reason_code:
                    op.strategy_reason_code = "SUPPLY_DISTRIBUTE_REQUESTED"
                stage = "transfer_failed"
                directory = self._receive(op, assets)
            self._check(op)
            stage = "registration_failed"
            self._progress(op, {"phase": "verifying"})
            self._register(op, directory)
            self._send(op, self._reply(op, "available"))
            self.collect_garbage()
        except Exception as exc:
            code = _error_code(exc, stage)
            try:
                self._revoke(op)
            except SupplyError as revoke_error:
                code = revoke_error.reason_code
            self._send(op, self._reply(op, "failed", code), terminal=True)
        finally:
            op.token = ""
            op.finished = True

    def _receive(self, op: _Operation, assets) -> Path:
        if not isinstance(op.token, str) or not op.token or type(op.api_port) is not int or not 1 <= op.api_port <= 65535:
            raise SupplyError("invalid_authorization")
        from network_address import build_url
        host = getattr(op.client, "server_host", "")
        if not isinstance(host, str) or not host or any(char in host for char in "/\\@?#\r\n"):
            raise SupplyError("invalid_master_endpoint")
        result = assets.receive_segment(op.manifest, self.root,
            base_url=build_url("http", host, op.api_port), token=op.token,
            cancelled=lambda: self._cancelled(op),
            progress=lambda value=None, **kw: self._progress(op, value, **kw))
        self._record_transfer_counts(op, result)
        if isinstance(result, dict) and _manifest(result.get("manifest", result)) != op.manifest:
            raise SupplyError("manifest_identity_mismatch")
        return _within(self._directory(result, self.root), self.root)

    def _local_cut(self, op: _Operation, assets) -> Path:
        manifest = op.manifest
        if manifest["format"] != "gguf":
            raise SupplyError("unsupported_local_cut")
        key = tuple(manifest[item] for item in ("source_model_id", "source_model_sha256", "tokenizer_sha256"))
        with self._lock:
            source = self._sources.get(key)
            if source is None:
                source = next((path for (_, sha, tokenizer), path in self._sources.items()
                               if sha == key[1] and tokenizer == key[2]), None)
        if source is None:
            raise SupplyError("local_source_unavailable")
        if _file_sha(source, lambda: self._cancelled(op)) != manifest["source_model_sha256"]:
            raise SupplyError("source_model_sha256_mismatch")
        from relay_segment_info import read_gguf_layer_info, read_gguf_tokenizer_sha256
        info = read_gguf_layer_info(source)
        if not info or info["n_layer"] != manifest["total_layers"]:
            raise SupplyError("source_model_metadata_mismatch")
        if read_gguf_tokenizer_sha256(source) != manifest["tokenizer_sha256"]:
            raise SupplyError("tokenizer_sha256_mismatch")
        self._progress(op, {"phase": "building"})
        self._check(op)
        # Stable root allows disk-backed lookup after worker restart.
        result = assets.build_gguf_segment(source, self.root,
            source_model_id=manifest["source_model_id"],
            source_model_sha256=manifest["source_model_sha256"],
            tokenizer_sha256=manifest["tokenizer_sha256"],
            start_layer=manifest["layer_range"][0], end_layer=manifest["layer_range"][1],
            total_layers=manifest["total_layers"], operation_id=op.operation_id, node_id=op.node_id,
            conversion_proof=manifest.get("conversion_proof"), reuse_root=self.root,
            cut_timeout_seconds=op.cut_timeout_seconds, cancelled=lambda: self._cancelled(op))
        if _manifest(result) != manifest:
            raise SupplyError("manifest_identity_mismatch")
        op.transfer_counts["reused_bytes"] = int(result.get("_reused_bytes", 0))
        return _within(self._directory(result, self.root), self.root)

    @contextmanager
    def _native_gc_paths(self, root: Path):
        """Hold the actual native owners while proving their file references.

        Reading __dict__/slots avoids ModelHost's lazy manager creation. Unknown
        hosts/providers disable deletion; a current supply pointer is not proof
        that an older native runtime has released its mappings.
        """
        host = getattr(self.scheduler, "_host", None)
        if host is None or type(host).__module__ != "model_host":
            yield None
            return
        refs, identities = [], set()
        seen = set()
        class UnknownOwner(Exception):
            pass
        def path(value):
            if value:
                if not isinstance(value, (str, Path)):
                    raise UnknownOwner()
                refs.append(Path(value))
        def manager(owner):
            if owner is None or id(owner) in seen:
                return
            seen.add(id(owner))
            module = type(owner).__module__
            if module == "model_host" and type(owner).__name__ == "_LazyModelManager":
                manager(object.__getattribute__(owner, "_instance"))
                return
            if module not in {"model_module", "llama_engine", "llama_server_engine",
                              "llama_keep_head", "gemma4_pipeline_sidecar"}:
                raise UnknownOwner()
            state = vars(owner)
            for name in ("model_path", "_model_path", "_full_model_path", "_mmproj_path"):
                path(state.get(name))
            descriptor = state.get("_pipeline_descriptor") or {}
            path(descriptor.get("loaded_artifact"))
            key = state.get("_keep_head_key")
            if key:
                path(key[0])
            for name in ("_llama_engine", "_island_engine", "_keep_head", "_gemma4_runtime"):
                manager(state.get(name))
        with ExitStack() as locks:
            proof = True
            try:
                state = vars(host)
                candidate_lock = state.get("_pipeline_candidate_lock")
                if candidate_lock is None:
                    raise UnknownOwner()
                locks.enter_context(candidate_lock)
                manager(host.runtime_controller.manager())
                path(dict(host.runtime_controller.snapshot().metadata).get("model_path"))
                transaction = state.get("_pipeline_candidate_transaction")
                if transaction is not None:
                    path(transaction.model_path)
                    manager(transaction.manager)
                    manager(transaction.previous_manager)
                    previous = transaction.previous_host_state or {}
                    mirror = previous.get("mirrors", {}).get("model_path")
                    if mirror and mirror[0]:
                        path(mirror[1])
                controller = getattr(self.scheduler, "_worker_prepare", None)
                if controller is not None:
                    if type(controller).__module__ != "worker_prepare":
                        raise UnknownOwner()
                    # Snapshot queued/current prepare identities. Native work
                    # needs full_chat_execution_lock, already held by admission.
                    with controller._lock:
                        commands = [controller._current, *controller._queue]
                        for command in commands:
                            payload = getattr(command, "payload", None) or {}
                            digest = payload.get("identity", {}).get("manifest_sha256")
                            if digest:
                                identities.add(digest)
                        runtime = controller._runtime
                    if runtime is not None:
                        if type(runtime).__module__ != "inference_service.engine_host":
                            raise UnknownOwner()
                        locks.enter_context(runtime._worker_layer_lock())
                        # Includes active, prepared candidate, previous and
                        # partially closed orphan ownership, not just loaded.
                        owned = [runtime._capture_worker_runtime(),
                                 getattr(runtime, "_worker_runtime_candidate", None),
                                 getattr(runtime, "_worker_runtime_previous", None),
                                 *getattr(runtime, "_worker_runtime_orphans", [])]
                        for item in owned:
                            if item is None:
                                continue
                            selected = item.selected or {}
                            path(selected.get("model_path"))
                            path(selected.get("manifest_path"))
                            manager(item.manager)
                            if item.upstream is not None:
                                path(item.upstream.model_path)
                if identities:
                    from model_supply_assets import _segment_candidates, _read_json
                    for directory in _segment_candidates(root):
                        if _read_json(directory / "supply.manifest.json").get("manifest_sha256") in identities:
                            refs.append(directory)
            except (UnknownOwner, AttributeError, TypeError, ValueError, OSError):
                proof = False
            yield refs if proof else None

    def collect_garbage(self, *, root: Path | None = None, protected_paths=(),
                        cancelled=lambda: False) -> dict:
        """Owner entry: supply completion invokes this under execution admission."""
        from model_supply_assets import collect_supply_segments
        root = Path(root) if root is not None else self.root
        protected = list(protected_paths)
        # Reentrant for completion on the receiver thread; nonblocking for an
        # external maintenance caller. Excludes both writers and prefix donors.
        if not self._execution_lock.acquire(blocking=False):
            return {"removed": 0, "reclaimed_bytes": 0,
                    "skipped": {"receiver_busy": 1}, "examined": 0}
        try:
            with self._admission():
                if (getattr(self.scheduler, "_task_worker_active_attempts", {})
                        or getattr(self.scheduler, "_active_pipeline_task_ids", set())):
                    raise SupplyError("SUPPLY_WORKER_BUSY")
                with self._lock:
                    active, pending = self._active, self._pending
                def invalidated():
                    if cancelled():
                        return True
                    with self._lock:
                        return (self._active is not active or self._pending is not pending
                                or (active is not None and active.cancelled.is_set()))
                if active is not None:
                    self._assert_idle(active)
                    if active.manifest.get("manifest_sha256"):
                        protected.append(self.root / (".supply-receive-" + active.manifest["manifest_sha256"]))
                    if active.previous_pointer:
                        old = json.loads(active.previous_pointer)
                        protected.append(self.root.parent / old["directory"])
                pointer = self.root / "current.json"
                from model_supply_assets import _link
                if _link(pointer):
                    raise SupplyError("unsafe_path")
                if pointer.is_file():
                    protected.append(self.root.parent / json.loads(_read_pointer(pointer))["directory"])
                import config
                setting = getattr(self.scheduler, "_runtime_setting", None)
                def cache_setting(name):
                    return setting(name) if callable(setting) else getattr(config, name)
                key = ("MODEL_SUPPLY_CACHE_KEEP" if root.resolve() == self.root.resolve()
                       else "MODEL_SUPPLY_CACHE_MASTER_KEEP")
                with self._native_gc_paths(root) as native:
                    return collect_supply_segments(root, protected_paths=protected,
                        native_paths=native, keep=cache_setting(key),
                        max_retained_bytes=cache_setting("MODEL_SUPPLY_CACHE_MAX_BYTES"),
                        cancelled=invalidated, receive_idle=True)
        except Exception:
            # Retention is maintenance, never a second terminal supply result.
            return {"removed": 0, "reclaimed_bytes": 0,
                    "skipped": {"owner_admission_unavailable": 1}, "examined": 0}

        finally:
            self._execution_lock.release()

    @contextmanager
    def _admission(self):
        # Mode switches and full-model chat use full -> stage. Never wait for
        # full chat while holding our operation mutex or the stage mutex.
        host = getattr(self.scheduler, "_host", None)
        full = getattr(host, "full_chat_execution_lock", None)
        stage = getattr(self.scheduler, "_task_worker_stage_lock", None)
        if full is None or stage is None:
            raise SupplyError("SUPPLY_ADMISSION_UNAVAILABLE")
        if not full.acquire(blocking=False):
            raise SupplyError("SUPPLY_WORKER_BUSY")
        native = getattr(getattr(host, "runtime_controller", None), "execution_lock", None)
        native_held = False
        try:
            if native is not None:
                native_held = native.acquire(blocking=False)
                if not native_held:
                    raise SupplyError("SUPPLY_WORKER_BUSY")
            with stage:
                yield
        finally:
            if native_held:
                native.release()
            full.release()

    def _assert_idle(self, op: _Operation) -> None:
        if (getattr(self.scheduler, "_task_worker_active_attempts", {})
                or (op.local and getattr(self.scheduler, "_active_pipeline_task_ids", set()))):
            raise SupplyError("SUPPLY_WORKER_BUSY")

    def _register(self, op: _Operation, directory: Path) -> Path:
        try:
            return self._register_verified(op, directory)
        except Exception as exc:
            _log_registration_error(op, "registration", exc)
            raise

    def _register_verified(self, op: _Operation, directory: Path) -> Path:
        self._check(op)
        directory = _within(directory, self.root.parent)
        marker = directory / "supply.manifest.json"
        if not marker.is_file():
            matches = [candidate.parent for candidate in directory.glob("segment-*/supply.manifest.json")
                       if _json(candidate).get("manifest_sha256") == op.manifest["manifest_sha256"]]
            if len(matches) != 1:
                raise SupplyError("artifact_unavailable")
            directory = _within(matches[0], self.root.parent)
        self._check(op)
        verified_files = _selection_key(self.root, b"", directory, op.manifest)[1]
        self._check(op)
        model_path = _verify_files(op.manifest, directory, lambda: self._cancelled(op))
        self._check(op)
        if _selection_key(self.root, b"", directory, op.manifest)[1] != verified_files:
            raise SupplyError("artifact_changed")
        self._check(op)
        manifest_path = directory / "supply.manifest.json"
        if manifest_path.is_symlink() or _manifest(_json(manifest_path)) != op.manifest:
            raise SupplyError("manifest_identity_mismatch")
        self._check(op)
        pointer = {"kind": "qlh.model_supply.current.v1",
                   "directory": directory.relative_to(self.root.parent.resolve()).as_posix(),
                   "manifest": manifest_path.name, "manifest_sha256": op.manifest["manifest_sha256"],
                   "artifact_sha256": op.manifest["artifact_sha256"]}
        self.root.mkdir(parents=True, exist_ok=True)
        if (self.root / "current.json").is_symlink():
            raise SupplyError("unsafe_path")
        current = _within(self.root / "current.json", self.root)
        temporary = self.root / (".current-" + uuid.uuid4().hex + ".tmp")
        try:
            self._check(op)
            temporary.write_text(json.dumps(pointer, sort_keys=True), encoding="utf-8")
            # This is the same admission boundary used for incoming stages.
            # Hashing and transfer happen outside it; selection changes do not.
            with self._admission():
                self._check(op)
                self._assert_idle(op)
                self._check(op)
                try:
                    # Match the resume journal's bounded Windows backoff.
                    # A reader without FILE_SHARE_DELETE can briefly block an
                    # otherwise valid pointer switch. Keep full -> stage held,
                    # but release the operation mutex during cancellable waits.
                    with _POINTER_IO_LOCK:
                        for attempt in range(8):
                            self._check(op)
                            self._assert_idle(op)
                            if current.is_symlink() or temporary.is_symlink():
                                raise SupplyError("unsafe_path")
                            if attempt and _selection_key(self.root, b"", directory, op.manifest)[1] != verified_files:
                                raise SupplyError("artifact_changed")
                            with self._lock:
                                if self._active is not op or op.cancelled.is_set():
                                    raise SupplyError("cancelled")
                                op.previous_pointer = _read_pointer(current) if current.exists() else None
                                try:
                                    os.replace(temporary, current)
                                except OSError as exc:
                                    if (sys.platform != "win32" or getattr(exc, "winerror", None) not in {5, 32, 33}
                                            or attempt == 7):
                                        raise
                                else:
                                    op.committed = pointer
                                    break
                            op.cancelled.wait(0.01 * 2 ** attempt)
                    self._check(op)
                    if _selection_key(self.root, b"", directory, op.manifest)[1] != verified_files:
                        raise SupplyError("artifact_changed")
                    self._check(op)
                    published = _read_pointer(current)
                    if json.loads(published) != pointer:
                        raise SupplyError("registration_pointer_visibility_failed")
                    _remember_selection(self.root, published, directory, op.manifest, model_path)
                    self._check(op)
                    self.scheduler.refresh_task_worker_capabilities()
                    self._check(op)
                    self._registration_blocked = False
                except Exception as exc:
                    self._revoke(op, admission_held=True)
                    if getattr(exc, "reason_code", "") == "registration_pointer_visibility_failed":
                        self._registration_blocked = True
                    raise
            return model_path
        finally:
            temporary.unlink(missing_ok=True)

    def _revoke(self, op: _Operation, *, admission_held=False) -> None:
        if op.committed is None:
            return
        with nullcontext() if admission_held else self._admission():
            # Any admitted execution keeps its artifact stable, including a
            # full chat and a master's legacy pipeline task.
            self._assert_idle(op)
            current = self.root / "current.json"
            try:
                with _POINTER_IO_LOCK:
                    if current.is_file() and json.loads(_read_pointer(current)) == op.committed:
                        if op.previous_pointer is None:
                            current.unlink()
                        else:
                            temporary = self.root / (".restore-" + uuid.uuid4().hex + ".tmp")
                            try:
                                temporary.write_bytes(op.previous_pointer)
                                pause = threading.Event()
                                for attempt in range(8):
                                    try:
                                        os.replace(temporary, current)
                                        break
                                    except OSError as exc:
                                        if (sys.platform != "win32" or getattr(exc, "winerror", None) not in {5, 32, 33}
                                                or attempt == 7):
                                            raise
                                    pause.wait(0.01 * 2 ** attempt)
                            finally:
                                temporary.unlink(missing_ok=True)
            except (OSError, ValueError) as exc:
                self._registration_blocked = True
                with _SELECTION_CACHE_LOCK:
                    _SELECTION_CACHE.pop(str(self.root), None)
                _log_registration_error(op, "rollback", exc)
                raise SupplyError("registration_rollback_failed") from exc
            op.committed = None
            self._registration_blocked = False
            with _SELECTION_CACHE_LOCK:
                _SELECTION_CACHE.pop(str(self.root), None)

    def activate_local(self, manifest: dict, directory: Path, *, cancelled: Callable[[], bool] | None = None) -> dict:
        """Register a verified local master artifact; never load an engine.

        The directory must be within root.parent (the model-supply state tree).
        This synchronous method is intended for the master's supply job thread.
        The optional callback observes the coordinator's live cancellation event.
        """
        op = _Operation(str(manifest.get("operation_id", "local-" + uuid.uuid4().hex)),
                        "master", None, manifest=deepcopy(manifest), local=True,
                        cancelled_callback=cancelled)
        if not self._context_valid(op):
            return self._reply(op, "failed", "distributed_mode_required")
        if cancelled is not None and cancelled():
            return self._reply(op, "failed", "cancelled")
        with self._lock:
            if self._cancel_key(op) in self._cancelled_operations:
                return self._reply(op, "failed", "cancelled")
            if self._active is not None:
                self._active.cancelled.set()
            self._active = op
            self._pending = None
        with self._execution_lock:
            try:
                self._check(op)
                op.manifest = _manifest(op.manifest)
                if op.manifest["node_id"] != op.node_id:
                    raise SupplyError("manifest_identity_mismatch")
                model_path = self._register(op, Path(directory))
                self._check(op)
                result = self._reply(op, "available")
                result["model_path"] = str(model_path)
                return result
            except Exception as exc:
                code = _error_code(exc, "registration_failed")
                try:
                    self._revoke(op)
                except SupplyError as revoke_error:
                    code = revoke_error.reason_code
                return self._reply(op, "failed", code)
            finally:
                op.finished = True

    def selected_model_path(self) -> Path | None:
        if not self._distributed():
            return None
        selected = get_selected_model(self.root)
        if not self._distributed():
            return None
        return Path(selected["model_path"]) if selected else None

    def selected_artifact_path(self) -> Path | None:
        path = self.selected_model_path()
        return path if path is not None and path.suffix.lower() == ".gguf" else None
