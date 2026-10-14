"""relay_segment_info.py — 段构建标识（供 relay 记录与对账；§10.2 待办）。

**为什么要它**：relay 记录里的 `engines` 过去是**固定字符串**（"纯 llama.cpp（无 torch / 无 D 档
组件参与推理）"）、`ifaces` 只写模块名 ⇒ 从记录本身**无法回答**「这一段实际用哪套 llama.cpp 构建、
段工件是哪一份」。这不是理论问题：tail 段走 **pip 绑定**还是**自建 shim**，直接决定同一 head 段下
是 1/32 还是 32/32（见 `docs/跨框架接力…§10.7`），而记录里当时分辨不出来。

三个入口：

* :func:`collect_local_build` —— **本进程段**：shim 与同目录 `libllama`/`ggml*` 的 sha256
  （小文件，很快）、pip `llama_cpp` 版本与其 `lib/llama.dll` 摘要、段工件的大小/名字；
  **GB 级段工件默认不算 sha256**（会明显拖慢每轮实验），需要时用 `digest_artifacts=True` 显式开启；
* :func:`load_ready_build` —— **远端段**：读服务端 ready 文件里的 `build` 字段（`relay_mid_service`
  已在写）；读不到/字段缺失时**如实标注**原因，不抛异常；
* :func:`unknown_remote_build` —— 远端既没给 ready 文件也没法读时**显式标注** `remote_unknown`：
  记录必须能区分「未知」与「没写这个字段」。
"""

from __future__ import annotations

import hashlib
import json
import struct
from pathlib import Path
from typing import Any

__all__ = [
    "RELAY_SEGMENT_INFO_SCHEMA",
    "collect_local_build",
    "load_ready_build",
    "read_artifact_manifest",
    "read_gguf_layer_info",
    "read_gguf_tokenizer_sha256",
    "unknown_remote_build",
]

#: 段构建标识的 schema 版本（记录里逐段出现）。
RELAY_SEGMENT_INFO_SCHEMA = "qlh.relay_segment_info.v1"

#: 与 llama.cpp 构建同源的动态库（这些才是真正决定数值的构建产物）。
_LLAMA_LIBRARIES = ("libllama.dll", "llama.dll", "libllama.so")
_GGML_LIBRARIES = ("ggml-base.dll", "ggml.dll", "libggml-base.so", "libggml.so")

#: 超过这个大小就不默认算 sha256（段工件事常是 GB 级）。
_DIGEST_SIZE_LIMIT = 64 * 1024 * 1024


def _sha256_file(path: Path, chunk: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(chunk):
            digest.update(block)
    return digest.hexdigest()


def _utc_now() -> str:
    from datetime import datetime, timezone

    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


# ── 零依赖读 GGUF 头（★ A15b / #30）──────────────────────────────────────────
# ⚠️ **刻意不 import `gguf` / `llama_cpp`** —— 与下面 `collect_local_build` 不做隐式 import 同因：
#    `llama_cpp` 一旦进 `sys.modules` 就会改变 keep-head 的隔离判据（实测踩到）。
#    这里只用标准库 `struct` 解 GGUF 的 KV 区，读 `block_count` / `nextn_predict_layers`，
#    用来回答「**这一段覆盖多少层**」—— 工件此前完全无法自证这一点（见 `已知问题记录.md` #30）。

_GGUF_MAGIC = b"GGUF"

#: GGUF KV 值类型 → (struct 格式, 字节数)。只列**定长标量**；STRING / ARRAY 单独处理。
_GGUF_SCALAR_TYPES: dict[int, tuple[str, int]] = {
    0: ("<B", 1), 1: ("<b", 1), 2: ("<H", 2), 3: ("<h", 2),
    4: ("<I", 4), 5: ("<i", 4), 6: ("<f", 4), 7: ("<?", 1),
    10: ("<Q", 8), 11: ("<q", 8), 12: ("<d", 8),
}
_GGUF_TYPE_STRING = 8
_GGUF_TYPE_ARRAY = 9


def _gguf_read_string(handle) -> str:
    (length,) = struct.unpack("<Q", handle.read(8))
    return handle.read(int(length)).decode("utf-8", errors="replace")


def _gguf_read_scalar(handle, value_type: int) -> Any:
    entry = _GGUF_SCALAR_TYPES.get(value_type)
    if entry is None:
        return None
    fmt, size = entry
    return struct.unpack(fmt, handle.read(size))[0]


def _gguf_skip_value(handle, value_type: int) -> None:
    """结构化跳过一个 KV 值（数组递归）。未知类型 ⇒ `ValueError`（由调用方兜成 `None`）。"""
    if value_type == _GGUF_TYPE_STRING:
        _gguf_read_string(handle)
        return
    if value_type == _GGUF_TYPE_ARRAY:
        (element_type,) = struct.unpack("<I", handle.read(4))
        (count,) = struct.unpack("<Q", handle.read(8))
        for _ in range(int(count)):
            _gguf_skip_value(handle, element_type)
        return
    if value_type not in _GGUF_SCALAR_TYPES:
        raise ValueError(f"未知的 GGUF KV 类型: {value_type}")
    handle.read(_GGUF_SCALAR_TYPES[value_type][1])


def read_gguf_layer_info(path: str | Path) -> dict[str, Any] | None:
    """**零依赖**读 GGUF 头 ⇒ `{"architecture","block_count","nextn_predict_layers","n_layer"}`。

    只解到需要的几个 KV，其余按类型**结构化跳过**；KV 顺序不做假设（先把标量都收集起来，
    最后再按 `<arch>.block_count` 挑）。读不到 / 非 GGUF / 格式非法一律返回 `None`
    （**不猜、不抛** —— 调用方据此如实记 `null`，而不是编一个层数出来）。

    `n_layer = block_count - nextn_predict_layers`（MTP 层不计入），与
    `scripts/cut_layers.py:82` 同口径。
    """
    target = Path(path)
    if not target.is_file():
        return None
    try:
        with target.open("rb") as handle:
            if handle.read(4) != _GGUF_MAGIC:
                return None
            (version,) = struct.unpack("<I", handle.read(4))
            if version < 2:
                return None
            handle.read(8)  # n_tensors（本函数不需要）
            (n_kv,) = struct.unpack("<Q", handle.read(8))
            arch: str | None = None
            scalars: dict[str, Any] = {}
            for _ in range(int(n_kv)):
                key = _gguf_read_string(handle)
                (value_type,) = struct.unpack("<I", handle.read(4))
                if key == "general.architecture" and value_type == _GGUF_TYPE_STRING:
                    arch = _gguf_read_string(handle)
                    continue
                # 只留流水线合同需要的整数标量；浮点/字符串没有用还占地方。
                if key.endswith((
                    ".block_count", ".nextn_predict_layers", ".embedding_length",
                )):
                    value = _gguf_read_scalar(handle, value_type)
                    if isinstance(value, int):
                        scalars[key] = value
                    continue
                _gguf_skip_value(handle, value_type)
    except (OSError, ValueError, struct.error, UnicodeDecodeError):
        return None

    if not arch:
        return None
    block_count = scalars.get(f"{arch}.block_count")
    if not isinstance(block_count, int) or block_count <= 0:
        return None
    nextn = scalars.get(f"{arch}.nextn_predict_layers") or 0
    n_embd = scalars.get(f"{arch}.embedding_length") or 0
    return {
        "architecture": arch,
        "block_count": block_count,
        "nextn_predict_layers": int(nextn),
        "n_layer": max(0, block_count - int(nextn)),
        "n_embd": max(0, int(n_embd)),
    }


def read_gguf_tokenizer_sha256(path: str | Path) -> str | None:
    """Fingerprint serialized tokenizer GGUF metadata without reading tensor payloads."""
    target = Path(path)
    try:
        file_size = target.stat().st_size
        with target.open("rb") as handle:
            if handle.read(4) != _GGUF_MAGIC:
                return None
            (version,) = struct.unpack("<I", handle.read(4))
            if version < 2:
                return None
            handle.read(8)
            (metadata_count,) = struct.unpack("<Q", handle.read(8))
            if metadata_count > 100_000:
                return None
            values: dict[str, bytes] = {}
            for _ in range(metadata_count):
                key = _gguf_read_string(handle)
                (value_type,) = struct.unpack("<I", handle.read(4))
                start = handle.tell()
                _gguf_skip_value(handle, value_type)
                end = handle.tell()
                if end > file_size:
                    return None
                if not key.startswith("tokenizer."):
                    continue
                if key in values or end - start > 128 * 1024 * 1024:
                    return None
                field_digest = hashlib.sha256(struct.pack("<I", value_type))
                handle.seek(start)
                remaining = end - start
                while remaining:
                    chunk = handle.read(min(remaining, 1 << 20))
                    if not chunk:
                        return None
                    field_digest.update(chunk)
                    remaining -= len(chunk)
                values[key] = field_digest.digest()
            if not {"tokenizer.ggml.model", "tokenizer.ggml.tokens"}.issubset(values):
                return None
            digest = hashlib.sha256()
            for key in sorted(values):
                digest.update(key.encode("utf-8") + b"\0" + values[key])
            return digest.hexdigest()
    except (OSError, ValueError, struct.error, UnicodeDecodeError):
        return None


#: GGML 张量类型 → `(每个 block 的元素数, 每个 block 的字节数)`。
#: 与 `llama.cpp` 的 `ggml_blck_size()` / `ggml_type_size()` 同源。**刻意不 import
#: `gguf` / `llama_cpp`** —— 同本模块其余部分：一旦进 `sys.modules` 就会改变 keep-head
#: 的隔离判据（实测踩到）。
_GGML_TYPE_SIZES: dict[int, tuple[int, int]] = {
    0: (1, 4), 1: (1, 2), 2: (32, 18), 3: (32, 20),
    6: (32, 22), 7: (32, 24), 8: (32, 34), 9: (32, 36),
    10: (256, 84), 11: (256, 110), 12: (256, 144), 13: (256, 176),
    14: (256, 210), 15: (256, 292), 16: (256, 66), 17: (256, 74),
    18: (256, 98), 19: (256, 50), 20: (32, 18), 21: (256, 110),
    22: (256, 82), 23: (256, 136), 24: (1, 1), 25: (1, 2),
    26: (1, 4), 27: (1, 8), 28: (1, 8), 29: (256, 56),
    30: (1, 2),
}


def read_gguf_tensor_bytes(path: str | Path) -> dict[str, int] | None:
    """**零依赖**读 GGUF 张量表 ⇒ `{张量名: 字节数}`。

    字节数按 **ggml 类型表精确算**（`元素数 / block_elems * block_bytes`），**不用**
    「相邻 offset 之差」—— 后者含 GGUF 的 32 字节对齐填充，对不上 `pipeline_capacity`
    要的 exact 账。

    读不到 / 非 GGUF / 遇到未知张量类型一律返回 `None`（**不猜、不抛**）。
    """
    target = Path(path)
    if not target.is_file():
        return None
    try:
        with target.open("rb") as handle:
            if handle.read(4) != _GGUF_MAGIC:
                return None
            (version,) = struct.unpack("<I", handle.read(4))
            if version < 2:
                return None
            (n_tensors,) = struct.unpack("<Q", handle.read(8))
            (n_kv,) = struct.unpack("<Q", handle.read(8))
            for _ in range(int(n_kv)):
                _gguf_read_string(handle)  # key
                (value_type,) = struct.unpack("<I", handle.read(4))
                _gguf_skip_value(handle, value_type)
            sizes: dict[str, int] = {}
            for _ in range(int(n_tensors)):
                name = _gguf_read_string(handle)
                (n_dims,) = struct.unpack("<I", handle.read(4))
                dims = struct.unpack(f"<{int(n_dims)}Q", handle.read(8 * int(n_dims)))
                (tensor_type,) = struct.unpack("<I", handle.read(4))
                handle.read(8)  # offset（本函数不用）
                entry = _GGML_TYPE_SIZES.get(tensor_type)
                if entry is None:
                    return None
                block_elems, block_bytes = entry
                elements = 1
                for dim in dims:
                    elements *= int(dim)
                if elements % block_elems:
                    return None
                sizes[name] = (elements // block_elems) * block_bytes
    except (OSError, ValueError, struct.error, UnicodeDecodeError):
        return None
    return sizes


def read_artifact_manifest(path: str | Path) -> dict[str, Any] | None:
    """读**段工件旁的 manifest**（`<artifact>.gguf.manifest.json` 或显式路径）里的层范围。

    返回经过基本格式校验的层范围、源整模摘要和段工件摘要（能读到哪个给哪个）；
    读不到 ⇒ `None`。工件选择方必须同时校验这些字段，不能只按层区间猜测同源性。
    """
    target = Path(path)
    if not target.is_file():
        return None
    try:
        payload = json.loads(target.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    if not isinstance(payload, dict):
        return None
    info: dict[str, Any] = {}
    span = payload.get("source_layer_range")
    if isinstance(span, (list, tuple)) and len(span) == 2:
        try:
            start, end = int(span[0]), int(span[1])
        except (TypeError, ValueError):
            start = end = -1
        if 0 <= start < end:
            info["source_layer_range"] = [start, end]
    total = payload.get("n_layer")
    if isinstance(total, int) and total > 0:
        info["n_layer"] = total
    for field in (
        "source_model_sha256", "artifact_sha256", "tokenizer_sha256",
    ):
        digest = str(payload.get(field, "") or "").strip().lower()
        if len(digest) == 64 and all(char in "0123456789abcdef" for char in digest):
            info[field] = digest
    source_model_id = str(payload.get("source_model_id", "") or "").strip()
    if (
        1 <= len(source_model_id) <= 128
        and all(
            char in "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789_.:-"
            for char in source_model_id
        )
    ):
        info["source_model_id"] = source_model_id
    hidden_size = payload.get("hidden_size")
    if isinstance(hidden_size, int) and not isinstance(hidden_size, bool) and hidden_size > 0:
        info["hidden_size"] = hidden_size
    return info or None


def _file_info(path: Path, *, digest: bool) -> dict[str, Any]:
    """单个文件的标识：名字/大小/mtime；`digest=True` 且文件不大时附 sha256。"""
    info: dict[str, Any] = {"path": str(path), "name": path.name, "exists": path.is_file()}
    if not info["exists"]:
        return info
    stat = path.stat()
    info["size"] = int(stat.st_size)
    info["mtime"] = int(stat.st_mtime)
    if digest and stat.st_size <= _DIGEST_SIZE_LIMIT:
        info["sha256"] = _sha256_file(path)
    else:
        # 大工件默认不算：GB 级 sha256 会拖慢每轮实验；要算请显式开 digest_artifacts。
        info["sha256"] = None
    return info


def collect_local_build(
    *,
    shim: str | Path | None = None,
    model: str | Path | None = None,
    llama_cpp_module: Any = None,
    digest_artifacts: bool = False,
) -> dict[str, Any]:
    """收集**本进程段**的构建标识。

    Args:
        shim: keep-head shim 路径；给了就同时记录同目录的 `libllama` / `ggml*`（构建同源）。
        model: 段工件（裁层 GGUF / 整模）路径。
        llama_cpp_module: 已导入的 `llama_cpp` 模块（可选）。**本函数不会自行 import 它** ——
            隐式导入会污染 `sys.modules`、进而改变 keep-head 的进程隔离判据（实测踩到）。
        digest_artifacts: 对**段工件**也算 sha256（GB 级会明显变慢）。
    """
    build: dict[str, Any] = {
        "schema_version": RELAY_SEGMENT_INFO_SCHEMA,
        "collected_at": _utc_now(),
    }
    if shim is not None:
        shim_path = Path(shim)
        build["shim"] = _file_info(shim_path, digest=True)
        siblings = {
            name: _file_info(shim_path.parent / name, digest=True)
            for name in (*_LLAMA_LIBRARIES, *_GGML_LIBRARIES)
            if (shim_path.parent / name).is_file()
        }
        if siblings:
            build["llama_cpp_build"] = siblings

    # ⚠️ **不做隐式 import**：调用方可能只是想记 shim 摘要，而在主进程里 `import llama_cpp`
    #    会**改变 keep-head 的隔离判据**（一旦 `llama_cpp` 进了 `sys.modules`，后续
    #    `KeepHeadUpstream` 会改走 worker 子进程）⇒ 这种副作用不可接受（实测踩到：
    #    `relay_mid_service._runner_build` 曾因此把 tail 段打成 `runner_failed`）。
    module = llama_cpp_module
    if module is not None:
        build["llama_cpp_version"] = getattr(module, "__version__", None)
        module_file = getattr(module, "__file__", None)
        lib_dir = Path(module_file).parent / "lib" if module_file else None
        if lib_dir is not None and lib_dir.is_dir():
            libs = {
                name: _file_info(lib_dir / name, digest=True)
                for name in (*_LLAMA_LIBRARIES, *_GGML_LIBRARIES)
                if (lib_dir / name).is_file()
            }
            if libs:
                build["llama_cpp_libs"] = libs

    if model is not None:
        model_path = Path(model)
        build["model"] = _file_info(model_path, digest=bool(digest_artifacts))
        # ★ A15b / #30：让**工件自己**回答「我覆盖多少层」（此前无法自证 ⇒ 已误判过一次）。
        #   ① 层数：零依赖读 GGUF 头（**不 import** `llama_cpp` / `gguf`，见上面的隔离约束）；
        #   ② 来源层号：读工件旁的 manifest —— 工件头部**没有**这个信息（裁层只改 `block_count`）。
        layer_info = read_gguf_layer_info(model_path)
        if layer_info:
            build["model"]["layer_info"] = layer_info
        manifest_info = read_artifact_manifest(Path(str(model_path) + ".manifest.json"))
        if manifest_info:
            build["model"]["manifest"] = {
                "path": str(model_path) + ".manifest.json",
                **manifest_info,
            }
    return build


def load_ready_build(path: str | Path) -> dict[str, Any]:
    """读服务端 ready 文件里的 `build` 段；读不到就**如实标注原因**（不抛）。"""
    ready_path = Path(path)
    try:
        payload = json.loads(ready_path.read_text(encoding="utf-8"))
    except Exception as exc:  # noqa: BLE001 - 读不到就标 unknown
        return {
            "schema_version": RELAY_SEGMENT_INFO_SCHEMA,
            "source": "ready_file_unreadable",
            "ready_file": str(ready_path),
            "error": f"{type(exc).__name__}: {exc}",
        }
    build = payload.get("build")
    if not isinstance(build, dict):
        return {
            "schema_version": RELAY_SEGMENT_INFO_SCHEMA,
            "source": "ready_file_without_build",
            "ready_file": str(ready_path),
            "role": payload.get("role"),
        }
    result = dict(build)
    result.setdefault("schema_version", RELAY_SEGMENT_INFO_SCHEMA)
    result["source"] = "ready_file"
    result["ready_file"] = str(ready_path)
    result["role"] = payload.get("role")
    return result


def unknown_remote_build(endpoint: str | None) -> dict[str, Any]:
    """远端段未提供构建标识时的**显式**占位（`remote_unknown`）。"""
    return {
        "schema_version": RELAY_SEGMENT_INFO_SCHEMA,
        "source": "remote_unknown",
        "endpoint": endpoint,
    }
