"""Independently hash read-only Windows file mappings without resolving aliases.

The supplied paths are evidence labels: neither path is opened or normalized.
Only exact psutil path matches are read, and every matching region must prove
its own expected bytes. No model loader or privilege adjustment is used.
"""
from __future__ import annotations

import ctypes
import hashlib
import math
import ntpath
import re
import sys
import time

_WINDOWS = sys.platform == "win32"
_MAX_READ_BYTES = 2 * 1024 * 1024
_PROCESS_QUERY_INFORMATION = 0x0400
_PROCESS_VM_READ = 0x0010
_MEM_COMMIT = 0x1000
_MEM_MAPPED = 0x40000
_PAGE_READONLY = 0x02


class _SystemInfo(ctypes.Structure):
    _fields_ = [
        ("oem_id", ctypes.c_uint32),
        ("page_size", ctypes.c_uint32),
        ("minimum_address", ctypes.c_void_p),
        ("maximum_address", ctypes.c_void_p),
        ("processor_mask", ctypes.c_size_t),
        ("processor_count", ctypes.c_uint32),
        ("processor_type", ctypes.c_uint32),
        ("allocation_granularity", ctypes.c_uint32),
        ("processor_level", ctypes.c_uint16),
        ("processor_revision", ctypes.c_uint16),
    ]


class _MemoryBasicInformation(ctypes.Structure):
    # The native 32-bit ABI has no PartitionId; the 64-bit ABI includes it
    # in the padding before RegionSize. Use the caller's native pointer width.
    _fields_ = [
        ("base_address", ctypes.c_void_p),
        ("allocation_base", ctypes.c_void_p),
        ("allocation_protect", ctypes.c_uint32),
        *([("partition_id", ctypes.c_uint16)] if ctypes.sizeof(ctypes.c_void_p) == 8 else []),
        ("region_bytes", ctypes.c_size_t),
        ("state", ctypes.c_uint32),
        ("protect", ctypes.c_uint32),
        ("kind", ctypes.c_uint32),
    ]


class _WindowsMemoryBackend:
    """Small read-only OS boundary, replaceable by a fake in contract tests."""

    def __init__(self):
        import psutil

        self._psutil = psutil
        self._handle = None
        self._kernel = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel = self._kernel
        kernel.GetSystemInfo.argtypes = [ctypes.POINTER(_SystemInfo)]
        kernel.GetSystemInfo.restype = None
        kernel.OpenProcess.argtypes = [ctypes.c_uint32, ctypes.c_int, ctypes.c_uint32]
        kernel.OpenProcess.restype = ctypes.c_void_p
        kernel.CloseHandle.argtypes = [ctypes.c_void_p]
        kernel.CloseHandle.restype = ctypes.c_int
        kernel.VirtualQueryEx.argtypes = [
            ctypes.c_void_p, ctypes.c_void_p,
            ctypes.POINTER(_MemoryBasicInformation), ctypes.c_size_t,
        ]
        kernel.VirtualQueryEx.restype = ctypes.c_size_t
        kernel.ReadProcessMemory.argtypes = [
            ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
            ctypes.c_size_t, ctypes.POINTER(ctypes.c_size_t),
        ]
        kernel.ReadProcessMemory.restype = ctypes.c_int
        info = _SystemInfo()
        kernel.GetSystemInfo(ctypes.byref(info))
        self.page_size = int(info.page_size)
        self.minimum_address = int(info.minimum_address or 0)
        self.maximum_address = min(
            int(info.maximum_address or 0), (1 << (8 * ctypes.sizeof(ctypes.c_void_p))) - 1,
        )

    def identity(self, pid):
        # A fresh Process avoids psutil's cached creation time after PID reuse.
        process = self._psutil.Process(pid)
        return {"pid": process.pid, "created_at": process.create_time()}

    def maps(self, pid):
        return self._psutil.Process(pid).memory_maps(grouped=False)

    def open(self, pid):
        self._handle = self._kernel.OpenProcess(
            _PROCESS_QUERY_INFORMATION | _PROCESS_VM_READ, False, pid,
        )
        if not self._handle:
            raise ctypes.WinError(ctypes.get_last_error())

    def query(self, address):
        info = _MemoryBasicInformation()
        returned = self._kernel.VirtualQueryEx(
            self._handle, address, ctypes.byref(info), ctypes.sizeof(info),
        )
        if returned != ctypes.sizeof(info):
            raise ValueError("VirtualQueryEx failed or returned incomplete region information")
        return {
            "address": int(info.base_address or 0),
            "allocation_base": int(info.allocation_base or 0),
            "allocation_protect": int(info.allocation_protect),
            "region_bytes": int(info.region_bytes),
            "state": int(info.state),
            "protect": int(info.protect),
            "kind": int(info.kind),
        }

    def read_into(self, address, buffer, count):
        if not 0 < count <= len(buffer) <= _MAX_READ_BYTES:
            raise ValueError("ReadProcessMemory buffer bounds are invalid")
        read = ctypes.c_size_t()
        target = ctypes.addressof(ctypes.c_char.from_buffer(buffer))
        if not self._kernel.ReadProcessMemory(
            self._handle, address, target, count, ctypes.byref(read),
        ):
            raise ctypes.WinError(ctypes.get_last_error())
        return read.value

    def close(self):
        if self._handle:
            handle, self._handle = self._handle, None
            if not self._kernel.CloseHandle(handle):
                raise ctypes.WinError(ctypes.get_last_error())


def _validate_inputs(pid, created_at, mapped_path, artifact_path, expected_sha256, expected_size):
    if type(pid) is not int or not 0 < pid <= 0xFFFFFFFF:
        raise ValueError("pid must be a positive Windows process ID")
    if (type(created_at) not in (int, float)
            or not math.isfinite(created_at) or created_at <= 0):
        raise ValueError("created_at must be a finite positive process creation time")
    for label, path in (("mapped_path", mapped_path), ("artifact_path", artifact_path)):
        if not isinstance(path, str) or "\0" in path or not ntpath.isabs(path):
            raise ValueError(f"{label} must be an absolute Windows path without NULs")
    if not isinstance(expected_sha256, str) or re.fullmatch(r"[0-9a-f]{64}", expected_sha256) is None:
        raise ValueError("expected_sha256 must be a lowercase SHA-256 digest")
    if type(expected_size) is not int or expected_size <= 0:
        raise ValueError("expected_size must be a positive integer byte count")


def _check_identity(backend, pid, created_at):
    identity = backend.identity(pid)
    if (not isinstance(identity, dict) or type(identity.get("pid")) is not int
            or type(identity.get("created_at")) not in (int, float)
            or identity != {"pid": pid, "created_at": created_at}):
        raise ValueError("process identity changed or process restarted")
    return identity


def _snapshot(backend, pid):
    """Freeze every enumerated file mapping, retaining duplicates and all fields."""
    records = []
    for mapping in backend.maps(pid):
        if (not isinstance(mapping.addr, str) or not isinstance(mapping.perms, str)
                or not isinstance(mapping.path, str) or type(mapping.rss) is not int
                or mapping.rss <= 0):
            raise ValueError("memory maps enumeration returned a malformed region")
        records.append((mapping.addr, mapping.perms, mapping.path, mapping.rss))
    return tuple(sorted(records))


def _region_address(text, size, backend):
    if re.fullmatch(r"(?:0x)?[0-9a-fA-F]+", text) is None:
        raise ValueError("mapped region has an invalid address")
    address = int(text, 16)
    if (not backend.minimum_address <= address <= backend.maximum_address
            or address <= 0 or address % backend.page_size
            or size > backend.maximum_address - address + 1):
        raise ValueError("mapped region address is unaligned or out of bounds")
    return address


def _check_region(backend, address, region_bytes):
    region = backend.query(address)
    expected = {
        "address": address, "allocation_base": address, "region_bytes": region_bytes,
        "state": _MEM_COMMIT, "protect": _PAGE_READONLY, "kind": _MEM_MAPPED,
    }
    if not isinstance(region, dict) or any(
        type(region.get(key)) is not int or region[key] != value for key, value in expected.items()
    ):
        raise ValueError("VirtualQueryEx region must be a complete read-only committed MEM_MAPPED allocation")
    return dict(region)


def capture_windows_mapped_artifact(
    *, pid: int, created_at: float, mapped_path: str, artifact_path: str,
    expected_sha256: str, expected_size: int,
) -> dict:
    """Prove all exact-path mappings by independently reading their process bytes.

    Returns evidence only on complete success; any inability to enumerate,
    query, read, or keep process/map identities stable raises ValueError. Each
    region must be a whole read-only file view of exactly the page-rounded
    expected size. Hashing excludes page padding and uses one <=2 MiB buffer.
    artifact_path is retained verbatim as the caller's identity label, never
    opened, and is not proof by itself. The caller must bind this observation
    to its separately verified artifact/runtime contract.
    """
    started_at = time.time()
    if not _WINDOWS:
        raise ValueError("Windows mapped artifact capture requires Windows")
    try:
        _validate_inputs(pid, created_at, mapped_path, artifact_path, expected_sha256, expected_size)
        backend = _WindowsMemoryBackend()
        try:
            _check_identity(backend, pid, created_at)
            if (type(backend.page_size) is not int or backend.page_size <= 0
                    or backend.page_size & (backend.page_size - 1)
                    or type(backend.minimum_address) is not int or backend.minimum_address <= 0
                    or type(backend.maximum_address) is not int
                    or backend.maximum_address < backend.minimum_address):
                raise ValueError("Windows page size or application address bounds are unavailable")
            rounded_size = ((expected_size + backend.page_size - 1) // backend.page_size) * backend.page_size
            if rounded_size > backend.maximum_address - backend.minimum_address + 1:
                raise ValueError("expected_size exceeds application address bounds")
            backend.open(pid)
            _check_identity(backend, pid, created_at)
            before = _snapshot(backend, pid)
            matches = []
            for addr, perms, path, size in before:
                if path != mapped_path:
                    continue
                if perms != "r":
                    raise ValueError("mapped region must be read-only")
                if size != rounded_size:
                    raise ValueError("mapped region size differs from page-rounded expected_size")
                matches.append((_region_address(addr, size, backend), size))
            if not matches:
                raise ValueError("no exact mapped_path file region was enumerated")
            matches.sort()
            for previous, current in zip(matches, matches[1:]):
                if current[0] < previous[0] + previous[1]:
                    raise ValueError("mapped regions overlap or contain duplicate addresses")
            queried = {address: _check_region(backend, address, size) for address, size in matches}
            buffer = bytearray(min(expected_size, _MAX_READ_BYTES))
            view = memoryview(buffer)
            regions = []
            for address, size in matches:
                digest, offset = hashlib.sha256(), 0
                while offset < expected_size:
                    count = min(len(buffer), expected_size - offset)
                    if address + offset + count > address + size:
                        raise ValueError("ReadProcessMemory would exceed mapped region bounds")
                    read = backend.read_into(address + offset, buffer, count)
                    if type(read) is not int or read != count:
                        raise ValueError("ReadProcessMemory failed or returned a short/invalid read")
                    digest.update(view[:count])
                    offset += count
                actual = digest.hexdigest()
                if actual != expected_sha256:
                    raise ValueError("independently hashed mapped bytes differ from expected_sha256")
                regions.append({
                    "address": address, "region_bytes": size,
                    "hashed_bytes": offset, "sha256": actual,
                })
            for address, size in matches:
                if _check_region(backend, address, size) != queried[address]:
                    raise ValueError("VirtualQueryEx mapping changed during capture")
            if _snapshot(backend, pid) != before:
                raise ValueError("complete memory maps inventory changed during capture")
            identity_after = _check_identity(backend, pid, created_at)
            result = {
                "method": "windows_read_process_memory_sha256",
                "pid": pid, "created_at": created_at, "identity_after": identity_after,
                "mapped_path": mapped_path, "artifact_path": artifact_path,
                "artifact_size_bytes": expected_size, "sha256": expected_sha256,
                "page_size": backend.page_size, "regions": regions, "maps_stable": True,
                "started_at": started_at, "finished_at": time.time(),
            }
        finally:
            backend.close()
    except ValueError:
        raise
    except Exception as exc:
        raise ValueError(f"Windows mapped artifact capture failed ({type(exc).__name__}): {exc}") from exc
    return result
