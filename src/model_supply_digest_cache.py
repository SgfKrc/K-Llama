"""Versioned local digest and metadata cache shared by supply owners."""
from __future__ import annotations

from copy import deepcopy
from contextlib import contextmanager
from functools import lru_cache
import hashlib
import json
import os
from pathlib import Path
import re
import threading
import uuid
import weakref

_SHA = re.compile(r"[0-9a-f]{64}\Z")
_LIMIT = 4096
_MAX_BYTES = 8 * 1024 * 1024
_LOCK = threading.RLock()
_STORES: dict[str, dict] = {}
_FILE_LOCKS: weakref.WeakValueDictionary = weakref.WeakValueDictionary()


class DigestCacheError(ValueError):
    def __init__(self, code: str):
        self.code = code
        super().__init__(code)


def _check(cancelled) -> None:
    if cancelled is not None and cancelled():
        raise DigestCacheError("cancelled")


@lru_cache(maxsize=1)
def _windows_api():
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
    return ctypes, wintypes, FileBasicInfo, create, query, close


def _change_time(path: Path, stat) -> int:
    if os.name != "nt":
        return stat.st_ctime_ns
    ctypes, wintypes, basic_info, create, query, close = _windows_api()
    filename = str(path.resolve())
    if not filename.startswith("\\\\?\\"):
        filename = "\\\\?\\UNC\\" + filename[2:] if filename.startswith("\\\\") else "\\\\?\\" + filename
    handle = create(filename, 0, 7, None, 3, 0x02000000, None)
    if handle == wintypes.HANDLE(-1).value:
        raise ctypes.WinError(ctypes.get_last_error())
    try:
        info = basic_info()
        if not query(handle, 0, ctypes.byref(info), ctypes.sizeof(info)):
            raise ctypes.WinError(ctypes.get_last_error())
        return int(info.ChangeTime) * 100
    finally:
        close(handle)


@contextmanager
def stable_file_read(path: str | Path):
    """Deny concurrent Windows writes while a verified hardlink is created."""
    if os.name != "nt":
        yield False
        return
    ctypes, wintypes, _, create, _, close = _windows_api()
    filename = str(Path(path).resolve(strict=True))
    if not filename.startswith("\\\\?\\"):
        filename = "\\\\?\\UNC\\" + filename[2:] if filename.startswith("\\\\") else "\\\\?\\" + filename
    handle = create(filename, 0x80000000, 0x1 | 0x4, None, 3, 0x80, None)
    if handle == wintypes.HANDLE(-1).value:
        yield False
        return
    try:
        yield True
    finally:
        close(handle)


def file_stamp(path: str | Path) -> tuple:
    path = Path(path)
    stat = path.stat()
    return (stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns, stat.st_ino,
            stat.st_dev, _change_time(path, stat))


def _store_path() -> Path:
    from config import STATE_DIR
    return Path(STATE_DIR).resolve() / "model-supply" / "file-digests.json"


def _legacy_digest(store: Path, path: Path, stamp: tuple, legacy_root) -> str | None:
    root = Path(legacy_root) if legacy_root is not None else store.parent / "received"
    legacy = root / "source-digests.json"
    try:
        if legacy.stat().st_size > _MAX_BYTES:
            return None
        payload = json.loads(legacy.read_text(encoding="utf-8"))
        if not isinstance(payload, dict) or payload.get("schema_version") != 1:
            return None
        entries = payload.get("entries")
        entry = entries.get(str(path)) if isinstance(entries, dict) else None
        if (isinstance(entry, dict) and entry.get("stamp") == [*stamp[:4], stamp[-1]]
                and isinstance(entry.get("sha256"), str) and _SHA.fullmatch(entry["sha256"])):
            return entry["sha256"]
    except (OSError, ValueError):
        pass
    return None


def _entries(store: Path) -> dict:
    key = str(store)
    if key not in _STORES:
        entries = {}
        try:
            if store.stat().st_size <= _MAX_BYTES:
                payload = json.loads(store.read_text(encoding="utf-8"))
                if isinstance(payload, dict) and payload.get("schema_version") == 1:
                    records = payload.get("entries")
                    if isinstance(records, dict):
                        entries = {name: value for name, value in list(records.items())[-_LIMIT:]
                                   if isinstance(value, dict)}
        except (OSError, ValueError):
            pass
        _STORES[key] = entries
    return _STORES[key]


def _file_lock(store: Path, path: Path):
    with _LOCK:
        key = (str(store), str(path))
        lock = _FILE_LOCKS.get(key)
        if lock is None:
            lock = threading.RLock()
            _FILE_LOCKS[key] = lock
        return lock


def _record(store: Path, path: Path, stamp: tuple) -> dict:
    with _LOCK:
        entry = _entries(store).get(str(path))
        return deepcopy(entry) if isinstance(entry, dict) and entry.get("stamp") == list(stamp) else {}


def _save(store: Path, path: Path, entry: dict) -> None:
    with _LOCK:
        entries = _entries(store)
        entries.pop(str(path), None)
        entries[str(path)] = entry
        while len(entries) > _LIMIT:
            entries.pop(next(iter(entries)))
        temporary = store.with_name(".file-digests-" + uuid.uuid4().hex + ".tmp")
        try:
            store.parent.mkdir(parents=True, exist_ok=True)
            encoded = json.dumps({"schema_version": 1, "entries": entries},
                                 ensure_ascii=True, sort_keys=True)
            if len(encoded) > _MAX_BYTES:
                return
            temporary.write_text(encoded, encoding="utf-8")
            os.replace(temporary, store)
        except OSError:
            pass
        finally:
            try:
                temporary.unlink(missing_ok=True)
            except OSError:
                pass


def _hash_file(path: Path, cancelled=None) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            _check(cancelled)
            digest.update(chunk)
    return digest.hexdigest()


def cached_sha256(path: str | Path, cancelled=None, *, hash_file=None, legacy_root=None) -> str:
    """Reuse only the same live file version; hash_file must read live bytes."""
    _check(cancelled)
    path = Path(path).resolve(strict=True)
    store = _store_path()
    with _file_lock(store, path):
        _check(cancelled)
        before = file_stamp(path)
        entry = _record(store, path, before)
        digest = entry.get("sha256")
        if not isinstance(digest, str) or _SHA.fullmatch(digest) is None:
            digest = _legacy_digest(store, path, before, legacy_root)
            if digest is None:
                digest = (hash_file or _hash_file)(path, cancelled)
            if not isinstance(digest, str) or _SHA.fullmatch(digest) is None:
                raise DigestCacheError("file_digest_mismatch")
            _check(cancelled)
            if before != file_stamp(path):
                raise DigestCacheError("file_digest_mismatch")
            entry.update(stamp=list(before), sha256=digest)
            _save(store, path, entry)
        _check(cancelled)
        if before != file_stamp(path):
            raise DigestCacheError("file_digest_mismatch")
        return digest


def remember_hardlink_sha256(source: str | Path, target: str | Path, digest: str,
                             source_stamp: tuple) -> bool:
    """Carry a verified digest across a hardlink-only timestamp transition."""
    if not isinstance(digest, str) or _SHA.fullmatch(digest) is None:
        return False
    source = Path(source).resolve(strict=True)
    target = Path(target).resolve(strict=True)
    store = _store_path()
    locks = sorted((_file_lock(store, source), _file_lock(store, target)), key=id)
    with locks[0], locks[1]:
        source_after = file_stamp(source)
        target_after = file_stamp(target)
        stable = (0, 1, 3, 4)
        if (source_after != target_after
                or tuple(source_stamp[index] for index in stable)
                != tuple(source_after[index] for index in stable)):
            return False
        entry = _record(store, source, source_stamp)
        if entry.get("sha256") != digest:
            return False
        entry.update(stamp=list(source_after), sha256=digest)
        _save(store, source, entry)
        _save(store, target, entry)
        return source_after == file_stamp(source) == file_stamp(target)


def remember_published_files(files) -> None:
    """Move verified cache records across an atomic parent-directory rename."""
    store = _store_path()
    for source, target, stamp, digest in files:
        source = Path(source).absolute()
        target = Path(target).absolute()
        if source.exists() or not target.is_file():
            continue
        try:
            current = file_stamp(target)
        except OSError:
            continue
        if current != tuple(stamp):
            continue
        entry = _record(store, source, stamp)
        if entry.get("sha256") != digest:
            continue
        _save(store, target.resolve(), entry)


def cached_metadata(path: str | Path, namespace: str, compute, cancelled=None) -> dict:
    """Cache parsed metadata under its parser version and the complete file stamp."""
    _check(cancelled)
    path = Path(path).resolve(strict=True)
    store = _store_path()
    with _file_lock(store, path):
        _check(cancelled)
        before = file_stamp(path)
        entry = _record(store, path, before)
        metadata = entry.get("metadata")
        metadata = metadata if isinstance(metadata, dict) else {}
        value = metadata.get(namespace)
        if not isinstance(value, dict):
            value = compute(path)
            if not isinstance(value, dict):
                raise DigestCacheError("invalid_metadata")
            _check(cancelled)
            if before != file_stamp(path):
                raise DigestCacheError("file_digest_mismatch")
            metadata[namespace] = deepcopy(value)
            entry.update(stamp=list(before), metadata=metadata)
            _save(store, path, entry)
        _check(cancelled)
        if before != file_stamp(path):
            raise DigestCacheError("file_digest_mismatch")
        return deepcopy(value)


def cached_metadata_items(path: str | Path, namespace: str, keys, compute, cancelled=None) -> dict:
    """Incrementally cache metadata for a verified content identity."""
    _check(cancelled)
    path = Path(path).resolve(strict=True)
    keys = tuple(dict.fromkeys(str(key) for key in keys))
    store = _store_path()
    with _file_lock(store, path):
        _check(cancelled)
        before = file_stamp(path)
        entry = _record(store, path, before)
        metadata = entry.get("metadata")
        metadata = metadata if isinstance(metadata, dict) else {}
        values = metadata.get(namespace)
        values = values if isinstance(values, dict) else {}
        changed = False
        known = {}
        with _LOCK:
            candidates = tuple(_entries(store).values())
        for candidate in candidates:
            candidate_metadata = candidate.get("metadata")
            candidate_values = candidate_metadata.get(namespace) if isinstance(candidate_metadata, dict) else None
            if not isinstance(candidate_values, dict):
                continue
            for key in keys:
                if key not in candidate_values:
                    continue
                if key in known and known[key] != candidate_values[key]:
                    raise DigestCacheError("metadata_conflict")
                known[key] = candidate_values[key]
        for key in keys:
            if key not in values and key in known:
                values[key] = known[key]
                changed = True
        missing = tuple(key for key in keys if key not in values)
        if missing:
            additions = compute(path, missing)
            if not isinstance(additions, dict) or set(additions) != set(missing):
                raise DigestCacheError("invalid_metadata")
            _check(cancelled)
            if before != file_stamp(path):
                raise DigestCacheError("file_digest_mismatch")
            values.update(additions)
            changed = True
        _check(cancelled)
        if before != file_stamp(path):
            raise DigestCacheError("file_digest_mismatch")
        if changed:
            metadata[namespace] = values
            entry.update(stamp=list(before), metadata=metadata)
            _save(store, path, entry)
        return deepcopy({key: values[key] for key in keys})
