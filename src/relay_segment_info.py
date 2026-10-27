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
import os
import struct
import threading
import weakref
from collections import OrderedDict
from contextlib import contextmanager
from copy import deepcopy
from functools import wraps
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
_MAX_ARRAY_ITEMS = 2_000_000
_MAX_GGUF_STRING_BYTES = 64 * 1024 * 1024

_GGUF_HEADER_CACHE = OrderedDict()
_GGUF_HEADER_LOCK = threading.RLock()
_GGUF_READER_LOCKS = weakref.WeakValueDictionary()


def _gguf_version(path):
    from model_supply_digest_cache import file_stamp
    return file_stamp(path)


def _cached_gguf_reader(reader):
    @wraps(reader)
    def read(path):
        try:
            version = _gguf_version(path)
            key = (reader.__name__, version)
            with _GGUF_HEADER_LOCK:
                if key in _GGUF_HEADER_CACHE:
                    _GGUF_HEADER_CACHE.move_to_end(key)
                    return deepcopy(_GGUF_HEADER_CACHE[key])
                lock = _GGUF_READER_LOCKS.get(key)
                if lock is None:
                    lock = threading.RLock()
                    _GGUF_READER_LOCKS[key] = lock
            with lock:
                with _GGUF_HEADER_LOCK:
                    if key in _GGUF_HEADER_CACHE:
                        _GGUF_HEADER_CACHE.move_to_end(key)
                        return deepcopy(_GGUF_HEADER_CACHE[key])
                if reader.__name__ in {"read_gguf_layer_info", "read_gguf_tokenizer_sha256"}:
                    from model_supply_digest_cache import cached_metadata

                    namespace = "relay-gguf-reader-v2-" + reader.__name__
                    result = cached_metadata(
                        path, namespace,
                        lambda source: {"value": reader(source)},
                    )["value"]
                else:
                    result = reader(path)
                    if version != _gguf_version(path):
                        return None
                if result is not None:
                    with _GGUF_HEADER_LOCK:
                        _GGUF_HEADER_CACHE[key] = deepcopy(result)
                        while len(_GGUF_HEADER_CACHE) > 128:
                            _GGUF_HEADER_CACHE.popitem(last=False)
                return deepcopy(result)
        except (OSError, ValueError):
            return None
    return read


class _HeaderStream:
    """GGUF header 的**只进不退**读视图：委托给文件对象，额外提供 `len()`。

    为什么需要它：解析器用 `len(handle)` 判断"读到这里是否会越过文件末尾"
    （`_gguf_skip_value` 的截断检查）。此前 `handle` 是 `mmap`，天然支持 `len()`
    与下标；换回顺序读之后 `BufferedReader` 没有 `__len__`，于是
    `TypeError: object of type '_io.BufferedReader' has no len()`（实测 2026-10-26 14:58）。

    `__len__` 取 `fstat().st_size` —— **只查元数据、不读内容**，因此不会破坏
    "preflight 不得触碰权重载荷"的契约（`tests/test_model_pipeline_availability.py`
    的 `HeaderOnly` 哨兵只允许 `read` 落在 header 范围内）。

    刻意**不**实现 `__getitem__`：整个模块对 header 的访问都是 `read/seek/tell`
    顺序式；若将来有人按下标取值，应当显式改成读取，而不是让这里悄悄退化成
    整文件映射。
    """

    __slots__ = ("_stream", "_size", "_pos")

    def __init__(self, stream) -> None:
        self._stream = stream
        self._pos = 0
        try:
            self._size = int(os.fstat(stream.fileno()).st_size)
        except (OSError, AttributeError, ValueError):
            # 底层可能是个**只提供 read 的哨兵**（测试用它禁止越界读）。
            # 这时拿不到真实大小 —— 必须表示成"无穷大"，否则
            # `_gguf_skip_value` 的截断检查（`tell() + length > len(handle)`）
            # 会把每一次跳过都误判成 truncated。
            self._size = -1

    # —— 只依赖底层 `read` ——
    #
    # 刻意**不**委托 `tell()` / `seek()`：本模块的解析器只做"顺序前进"
    # （`read(n)` 取值、`seek(n, 1)` 跳过），因此偏移量自己记账就够，
    # 也更贴合"只准读 header"的契约 —— `tests/test_model_pipeline_availability.py`
    # 的 `HeaderOnly` 哨兵**只**提供了 `read`（它把越界读取变成断言失败），
    # 委托 `fileno()` / `tell()` 都会直接 `AttributeError`（实测 2026-10-26 14:47/15:05）。
    def read(self, count=-1):
        data = self._stream.read(count)
        self._pos += len(data)
        return data

    def seek(self, offset, whence=0):
        """只支持"相对当前位置前进"——这正是解析器的唯一用法。"""
        if whence == 1 and offset >= 0:
            remaining = offset
            # 分块吞掉，避免为一个跳过而尝试分配巨大缓冲。
            while remaining > 0:
                chunk = self.read(min(remaining, 1 << 20))
                if not chunk:
                    break
                remaining -= len(chunk)
            return self._pos
        if whence == 0 and offset >= 0:
            # 绝对定位只用于"从头重来"这种极端情况：退化为吞到目标位置。
            if offset < self._pos:
                raise ValueError("GGUF header 视图不支持回退定位")
            return self.seek(offset - self._pos, 1)
        if whence == 2:
            raise ValueError("GGUF header 视图不支持从末尾定位")
        raise ValueError(f"不支持的 seek: offset={offset} whence={whence}")

    def tell(self):
        return self._pos

    # —— mmap 才有、而解析器确实用到的能力 ——
    def __len__(self) -> int:
        # 大小未知（见 `__init__`）时返回一个远大于任何真实 GGUF 的值，
        # 让截断检查退化为"不判截断"——宁可让解析器在 EOF 处读到短数据
        # 自行报 `truncated`，也不要因为哨兵没有 `fileno()` 就全部误判。
        return self._size if self._size >= 0 else (1 << 62)


@contextmanager
def _gguf_header(path):
    """只读地打开 GGUF，**按顺序**交出读视图（不做整文件 mmap）。

    ★ 2026-10-26：此前这里是 `mmap.mmap(stream.fileno(), 0)` —— 把**整个文件**
    映射进地址空间。GGUF header 是纯顺序读（`read(n)` 逐段前进），mmap 的随机
    访问能力一次都用不上，代价却很实在：

      1. **越界**：映射范围覆盖全部张量载荷与 `tokenizer.ggml.tokens`
         （本模型 248320 条字符串）⇒ "只读元数据"的契约名存实亡；
         `tests/test_model_pipeline_availability.py::test_preflight_never_reads_weight_payloads_or_hashes_models`
         用 `HeaderOnly` 哨兵把这条契约变成硬断言，于是它在 `fileno()` 上直接
         `AttributeError`（实测 2026-10-26 14:47）。该测试还同时禁止 `sha256`
         并要求总读取量 ≤ 2×header。
      2. **无谓开销**：解析 header 只需要前若干 KB ~ 数 MB。

    改回顺序读之后，读取量完全由解析器决定（解析器只前进到 header 结束），
    既满足契约，也顺带消除了"冷进程摸一遍整份权重"的开销。
    """
    with Path(path).open("rb") as stream:
        yield _HeaderStream(stream)


def _gguf_read_string(handle) -> str:
    (length,) = struct.unpack("<Q", handle.read(8))
    value = handle.read(int(length))
    if len(value) != length:
        raise ValueError("truncated GGUF string")
    return value.decode("utf-8", errors="replace")


def _gguf_read_scalar(handle, value_type: int) -> Any:
    entry = _GGUF_SCALAR_TYPES.get(value_type)
    if entry is None:
        return None
    fmt, size = entry
    return struct.unpack(fmt, handle.read(size))[0]


def _gguf_skip_value(handle, value_type: int) -> None:
    """结构化跳过一个 KV 值（数组递归）。未知类型 ⇒ `ValueError`（由调用方兜成 `None`）。"""
    if value_type == _GGUF_TYPE_STRING:
        (length,) = struct.unpack("<Q", handle.read(8))
        if length > _MAX_GGUF_STRING_BYTES or handle.tell() + length > len(handle):
            raise ValueError("truncated or excessive GGUF string")
        handle.seek(int(length), 1)
        return
    if value_type == _GGUF_TYPE_ARRAY:
        (element_type,) = struct.unpack("<I", handle.read(4))
        (count,) = struct.unpack("<Q", handle.read(8))
        if count > _MAX_ARRAY_ITEMS:
            raise ValueError("GGUF metadata array is excessive")
        if element_type in _GGUF_SCALAR_TYPES:
            length = int(count) * _GGUF_SCALAR_TYPES[element_type][1]
            if handle.tell() + length > len(handle):
                raise ValueError("truncated GGUF metadata array")
            handle.seek(length, 1)
            return
        for _ in range(int(count)):
            _gguf_skip_value(handle, element_type)
        return
    if value_type not in _GGUF_SCALAR_TYPES:
        raise ValueError(f"未知的 GGUF KV 类型: {value_type}")
    length = _GGUF_SCALAR_TYPES[value_type][1]
    if handle.tell() + length > len(handle):
        raise ValueError("truncated GGUF scalar")
    handle.seek(length, 1)


def _gguf_digest_value(handle, value_type: int, digest) -> None:
    """消费一个 KV 值（数组递归），**并把读到的原始字节喂给 `digest`** —— 只走一遍。

    ★ 2026-10-26：此前 `read_gguf_tokenizer_sha256` 的做法是

        start = handle.tell();  _gguf_skip_value(...);  end = handle.tell()
        handle.seek(start);     # 回退
        while remaining: ...    # 再读一遍算摘要

    也就是**同一个 KV 被完整遍历两次**。对 `tokenizer.ggml.tokens` 这种
    248320 条字符串的数组（本模型 qwen35），代价实打实翻倍；而且它必须先
    `seek` 回退 —— 那个回退在后来的"只进不退"读视图上直接 `ValueError`，
    把这段隐藏浪费变成了显式失败（实测 2026-10-26 15:05）。

    这里把"跳过"和"取摘要"合并成一次顺序读：语义与原来的
    `sha256(pack(value_type) + raw_bytes[start:end])` 逐字节等价，但只读一遍。

    `digest` 允许为 `None`（调用方只关心跳过时不必构造 hasher）。
    """

    def feed(chunk) -> None:
        if digest is not None and chunk:
            digest.update(chunk)

    if value_type == _GGUF_TYPE_STRING:
        header = handle.read(8)
        (length,) = struct.unpack("<Q", header)
        feed(header)
        if length > _MAX_GGUF_STRING_BYTES or handle.tell() + length > len(handle):
            raise ValueError("truncated or excessive GGUF string")
        remaining = int(length)
        while remaining > 0:
            chunk = handle.read(min(remaining, 1 << 20))
            if not chunk:
                raise ValueError("truncated GGUF string")
            feed(chunk)
            remaining -= len(chunk)
        return
    if value_type == _GGUF_TYPE_ARRAY:
        head = handle.read(12)
        (element_type,) = struct.unpack("<I", head[:4])
        (count,) = struct.unpack("<Q", head[4:12])
        feed(head)
        if count > _MAX_ARRAY_ITEMS:
            raise ValueError("GGUF metadata array is excessive")
        if element_type in _GGUF_SCALAR_TYPES:
            remaining = int(count) * _GGUF_SCALAR_TYPES[element_type][1]
            if handle.tell() + remaining > len(handle):
                raise ValueError("truncated GGUF metadata array")
            while remaining > 0:
                chunk = handle.read(min(remaining, 1 << 20))
                if not chunk:
                    raise ValueError("truncated GGUF metadata array")
                feed(chunk)
                remaining -= len(chunk)
            return
        for _ in range(int(count)):
            _gguf_digest_value(handle, element_type, digest)
        return
    if value_type not in _GGUF_SCALAR_TYPES:
        raise ValueError(f"未知的 GGUF KV 类型: {value_type}")
    length = _GGUF_SCALAR_TYPES[value_type][1]
    if handle.tell() + length > len(handle):
        raise ValueError("truncated GGUF scalar")
    chunk = handle.read(length)
    if len(chunk) != length:
        raise ValueError("truncated GGUF scalar")
    feed(chunk)


@_cached_gguf_reader
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
        with _gguf_header(target) as handle:
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
                if key == "general.architecture":
                    if arch is not None or value_type != _GGUF_TYPE_STRING:
                        return None
                    arch = _gguf_read_string(handle)
                    continue
                # Preserve original scalar types. A present invalid value must
                # not become an absent/default value or a coerced integer.
                if key.endswith((
                    ".block_count", ".nextn_predict_layers", ".embedding_length",
                )):
                    if key in scalars:
                        return None
                    if value_type in _GGUF_SCALAR_TYPES:
                        scalars[key] = _gguf_read_scalar(handle, value_type)
                    else:
                        _gguf_skip_value(handle, value_type)
                        scalars[key] = None
                    continue
                _gguf_skip_value(handle, value_type)
    except (OSError, ValueError, struct.error, UnicodeDecodeError):
        return None

    if not arch:
        return None
    block_count = scalars.get(f"{arch}.block_count")
    if type(block_count) is not int or block_count <= 0:
        return None
    nextn = scalars.get(f"{arch}.nextn_predict_layers", 0)
    if type(nextn) is not int or not 0 <= nextn < block_count:
        return None
    embedding_key = f"{arch}.embedding_length"
    n_embd = scalars.get(embedding_key, 0)
    if embedding_key in scalars and (type(n_embd) is not int or n_embd <= 0):
        return None
    # Missing embedding metadata remains explicit as 0 for layer-only
    # diagnostics. Executable descriptors/builders must require a positive size.
    return {
        "architecture": arch,
        "block_count": block_count,
        "nextn_predict_layers": nextn,
        "n_layer": block_count - nextn,
        "n_embd": n_embd,
    }


@_cached_gguf_reader
def read_gguf_tokenizer_sha256(path: str | Path) -> str | None:
    """Fingerprint serialized tokenizer GGUF metadata without reading tensor payloads."""
    target = Path(path)
    try:
        file_size = target.stat().st_size
        with _gguf_header(target) as handle:
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
                # ★ 一次遍历：`_gguf_digest_value` 读的同时把**原始字节**喂给 hasher。
                #   原实现是"先 `_gguf_skip_value` 定位 end，再 `seek(start)` 重读"
                #   —— `tokenizer.ggml.tokens` 有 248320 条字符串（本模型），
                #   等于把最大的那个 KV 完整走两遍。摘要口径不变：
                #   `sha256(pack(value_type) + raw_bytes)`。
                field_digest = hashlib.sha256(struct.pack("<I", value_type))
                wants_digest = key.startswith("tokenizer.")
                if wants_digest and key in values:
                    return None
                start = handle.tell()
                _gguf_digest_value(handle, value_type, field_digest if wants_digest else None)
                end = handle.tell()
                if end > file_size:
                    return None
                if not wants_digest:
                    continue
                if end - start > 128 * 1024 * 1024:
                    return None
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


@_cached_gguf_reader
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
        with _gguf_header(target) as handle:
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
