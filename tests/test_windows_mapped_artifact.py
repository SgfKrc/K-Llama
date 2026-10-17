"""Strict mapped-byte proof tests; native integrations use only tiny temp files."""
from __future__ import annotations

from collections import namedtuple
from copy import deepcopy
import hashlib
import importlib.util
import os
from pathlib import Path
import subprocess
import sys
import time
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "windows_mapped_artifact_under_test",
    ROOT / "scripts/package_contract/windows_mapped_artifact.py",
)
helper = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(helper)

Mapping = namedtuple("Mapping", "addr perms path rss")
PID, CREATED = 4242, 12345.25
MAPPED = r"C:\deleted-operation\old-segment.gguf"
ARTIFACT = r"C:\current-operation\segment.gguf"
BASE = 0x10000
PAGE = 4096
DATA = (bytes(range(251)) * 25)[:6001]


def rounded(size, page=PAGE):
    return ((size + page - 1) // page) * page


def region(address=BASE, size=None):
    return {
        "address": address, "allocation_base": address, "allocation_protect": 2,
        "region_bytes": rounded(len(DATA)) if size is None else size,
        "state": 0x1000, "protect": 2, "kind": 0x40000,
    }


class FakeBackend:
    """No disk fallback; every byte returned is from an identified fake region."""

    def __init__(self, data=DATA):
        self.page_size = PAGE
        self.minimum_address = BASE
        self.maximum_address = (1 << 47) - 1
        maps = [
            Mapping(hex(BASE), "r", MAPPED, rounded(len(data))),
            Mapping("0x90000", "rx", r"C:\program\unrelated.dll", PAGE),
        ]
        self.snapshots = [deepcopy(maps), deepcopy(maps)]
        self.identities = [{"pid": PID, "created_at": CREATED}] * 3
        self.contents = {BASE: data}
        self.query_results = {BASE: region(size=rounded(len(data)))}
        self.events, self.reads = [], []
        self.identity_count = self.maps_count = self.query_count = 0
        self.opened, self.closed = [], 0
        self.read_error = self.open_error = self.close_error = None
        self.read_result = None
        self.query_change = None

    def identity(self, pid):
        self.events.append("identity")
        value = self.identities[min(self.identity_count, len(self.identities) - 1)]
        self.identity_count += 1
        if isinstance(value, Exception):
            raise value
        return deepcopy(value)

    def maps(self, pid):
        self.events.append("maps")
        value = self.snapshots[min(self.maps_count, len(self.snapshots) - 1)]
        self.maps_count += 1
        if isinstance(value, Exception):
            raise value
        return iter(value)

    def open(self, pid):
        self.events.append("open")
        if self.open_error:
            raise self.open_error
        self.opened.append(pid)

    def query(self, address):
        self.events.append("query")
        self.query_count += 1
        value = deepcopy(self.query_results[address])
        if self.query_change:
            value = self.query_change(self.query_count, value)
        if isinstance(value, Exception):
            raise value
        return value

    def read_into(self, address, buffer, count):
        self.events.append("read")
        self.reads.append((address, count, len(buffer), id(buffer)))
        if self.read_error:
            raise self.read_error
        for base, data in self.contents.items():
            offset = address - base
            if 0 <= offset and offset + count <= len(data):
                buffer[:count] = data[offset:offset + count]
                return count if self.read_result is None else self.read_result(count)
        raise AssertionError("helper requested bytes outside an actual fake mapping")

    def close(self):
        self.events.append("close")
        self.closed += 1
        if self.close_error:
            raise self.close_error

    def add_mapping(self, address, data=DATA):
        for snapshot in self.snapshots:
            snapshot.append(Mapping(hex(address), "r", MAPPED, rounded(len(data), self.page_size)))
        self.contents[address] = data
        self.query_results[address] = region(address, rounded(len(data), self.page_size))


@pytest.fixture
def fake(monkeypatch):
    backend = FakeBackend()
    monkeypatch.setattr(helper, "_WINDOWS", True)
    monkeypatch.setattr(helper, "_WindowsMemoryBackend", lambda: backend)
    return backend


def capture(**changes):
    arguments = dict(
        pid=PID, created_at=CREATED, mapped_path=MAPPED, artifact_path=ARTIFACT,
        expected_sha256=hashlib.sha256(DATA).hexdigest(), expected_size=len(DATA),
    )
    arguments.update(changes)
    return helper.capture_windows_mapped_artifact(**arguments)


def test_exact_public_result_and_all_matching_regions_are_independently_hashed(fake):
    fake.add_mapping(0x50000)
    began = time.time()
    result = capture()
    assert set(result) == {
        "method", "pid", "created_at", "identity_after", "mapped_path", "artifact_path",
        "artifact_size_bytes", "sha256", "page_size", "regions", "maps_stable",
        "started_at", "finished_at",
    }
    assert result["method"] == "windows_read_process_memory_sha256"
    assert (result["pid"], result["created_at"]) == (PID, CREATED)
    assert result["identity_after"] == {"pid": PID, "created_at": CREATED}
    assert result["mapped_path"] == MAPPED and result["artifact_path"] == ARTIFACT
    assert result["artifact_size_bytes"] == len(DATA)
    assert result["sha256"] == hashlib.sha256(DATA).hexdigest()
    assert result["page_size"] == PAGE and result["maps_stable"] is True
    assert result["regions"] == [
        {"address": base, "region_bytes": rounded(len(DATA)), "hashed_bytes": len(DATA),
         "sha256": hashlib.sha256(DATA).hexdigest()}
        for base in (BASE, 0x50000)
    ]
    assert began <= result["started_at"] <= result["finished_at"] <= time.time()
    assert fake.identity_count == 3 and fake.maps_count == 2
    assert fake.closed == 1 and fake.opened == [PID]
    assert fake.events[0] == "identity" and fake.events[-2:] == ["identity", "close"]
    assert {read[0] for read in fake.reads} == {BASE, 0x50000}


def test_missing_old_path_and_artifact_are_never_opened_or_checked(fake, monkeypatch):
    import builtins

    def forbidden(*args, **kwargs):
        pytest.fail("a supplied path was opened or checked instead of reading process memory")

    monkeypatch.setattr(builtins, "open", forbidden)
    monkeypatch.setattr(os.path, "exists", forbidden)
    monkeypatch.setattr(os.path, "isfile", forbidden)
    monkeypatch.setattr(os, "stat", forbidden)
    result = capture()
    assert result["mapped_path"] == MAPPED
    assert result["artifact_path"] == ARTIFACT
    assert fake.reads


def test_mapping_order_change_alone_is_not_an_inventory_change(fake):
    fake.snapshots[1].reverse()
    assert capture()["maps_stable"] is True


def test_windows_only_rejects_without_opening_a_process(fake, monkeypatch):
    monkeypatch.setattr(helper, "_WINDOWS", False)
    with pytest.raises(ValueError, match="requires Windows"):
        capture()
    assert not fake.events


@pytest.mark.parametrize("field,value", [
    ("pid", 0), ("pid", -1), ("pid", True), ("pid", 1.0), ("pid", 1 << 32),
    ("created_at", 0), ("created_at", -1), ("created_at", True),
    ("created_at", float("nan")), ("created_at", float("inf")), ("created_at", "12345.25"),
    ("mapped_path", "segment.gguf"), ("mapped_path", r"C:segment.gguf"),
    ("mapped_path", "C:\\bad\0name.gguf"), ("mapped_path", None),
    ("artifact_path", "segment.gguf"), ("artifact_path", ""), ("artifact_path", Path(ARTIFACT)),
    ("expected_sha256", "a" * 63), ("expected_sha256", "g" * 64),
    ("expected_sha256", "A" * 64), ("expected_sha256", None),
    ("expected_size", 0), ("expected_size", -1), ("expected_size", True),
    ("expected_size", 1.0), ("expected_size", "6001"),
])
def test_invalid_inputs_raise_value_error_before_process_access(fake, field, value):
    with pytest.raises(ValueError):
        capture(**{field: value})
    assert not fake.events


@pytest.mark.parametrize("moment", [0, 1, 2])
@pytest.mark.parametrize("change", ["created", "pid", "missing", "denied", "gone"])
def test_process_identity_must_match_before_open_after_open_and_after_capture(fake, moment, change):
    identity = {"pid": PID, "created_at": CREATED}
    if change == "created":
        identity["created_at"] += 0.000001
    elif change == "pid":
        identity["pid"] += 1
    elif change == "missing":
        identity.pop("created_at")
    elif change == "denied":
        identity = PermissionError("identity query denied")
    else:
        identity = ProcessLookupError("process exited")
    fake.identities = [deepcopy({"pid": PID, "created_at": CREATED}) for _ in range(3)]
    fake.identities[moment] = identity
    with pytest.raises(ValueError):
        capture()
    assert fake.closed == 1
    if moment < 2:
        assert not fake.reads


@pytest.mark.parametrize("stage", ["open", "read", "close"])
def test_access_denied_or_failed_handle_cleanup_is_not_a_success(fake, stage):
    setattr(fake, stage + "_error", PermissionError(stage + " denied"))
    with pytest.raises(ValueError, match="denied"):
        capture()
    assert fake.closed == 1
    if stage == "open":
        assert not fake.reads


@pytest.mark.parametrize("when", [0, 1])
@pytest.mark.parametrize("failure", [PermissionError("cannot enumerate"), RuntimeError("enumeration aborted")])
def test_maps_enumeration_failures_never_return_partial_evidence(fake, when, failure):
    fake.snapshots[when] = failure
    with pytest.raises(ValueError, match="enumerat"):
        capture()
    assert fake.closed == 1


def test_lazy_enumeration_error_after_a_matching_region_still_rejects(fake):
    first = fake.snapshots[0][0]

    def partial():
        yield first
        raise PermissionError("remaining maps unavailable")

    fake.snapshots[0] = partial()
    with pytest.raises(ValueError, match="remaining maps unavailable"):
        capture()
    assert not fake.reads and fake.closed == 1


@pytest.mark.parametrize("change", ["added", "removed", "address", "permissions", "path", "size", "duplicate"])
def test_entire_inventory_must_be_stable_including_unrelated_regions(fake, change):
    unrelated = fake.snapshots[1][1]
    if change == "added":
        fake.snapshots[1].append(Mapping("0xa0000", "r", r"C:\new.dll", PAGE))
    elif change == "removed":
        fake.snapshots[1].pop()
    elif change == "duplicate":
        fake.snapshots[1].append(unrelated)
    else:
        key, value = {
            "address": ("addr", "0xa0000"), "permissions": ("perms", "r"),
            "path": ("path", r"C:\changed.dll"), "size": ("rss", PAGE * 2),
        }[change]
        fake.snapshots[1][1] = unrelated._replace(**{key: value})
    with pytest.raises(ValueError, match="inventory changed"):
        capture()
    assert fake.closed == 1


@pytest.mark.parametrize("value", [None, 0, -1, True, "4096"])
def test_malformed_unrelated_map_cannot_be_silently_dropped(fake, value):
    fake.snapshots[0][1] = fake.snapshots[0][1]._replace(rss=value)
    with pytest.raises(ValueError, match="malformed"):
        capture()
    assert not fake.reads


@pytest.mark.parametrize("path", [
    r"C:\some-other-directory\old-segment.gguf", MAPPED.lower(), MAPPED.replace("\\", "/"),
    ARTIFACT, MAPPED + " (deleted)",
])
def test_basename_case_normalization_alias_and_deleted_suffix_are_not_path_proof(fake, path):
    for snapshot in fake.snapshots:
        snapshot[0] = snapshot[0]._replace(path=path)
    with pytest.raises(ValueError, match="no exact mapped_path"):
        capture()
    assert not fake.reads


def test_empty_map_list_is_not_proof_of_mapping(fake):
    fake.snapshots[0] = []
    with pytest.raises(ValueError, match="no exact mapped_path"):
        capture()


@pytest.mark.parametrize("perms", ["rw", "wc", "xr", "xrw", "xwc", "", "r--"])
def test_selected_mapping_must_be_exactly_readonly(fake, perms):
    fake.snapshots[0][0] = fake.snapshots[0][0]._replace(perms=perms)
    with pytest.raises(ValueError, match="read-only"):
        capture()
    assert not fake.reads


@pytest.mark.parametrize("size", [1, len(DATA), PAGE, rounded(len(DATA)) + PAGE])
def test_mapping_size_must_be_exactly_page_rounded_artifact_size(fake, size):
    fake.snapshots[0][0] = fake.snapshots[0][0]._replace(rss=size)
    with pytest.raises(ValueError, match="page-rounded"):
        capture()
    assert not fake.reads


@pytest.mark.parametrize("address", [
    "", "0", "0x0", "-0x10000", "+0x10000", " 0x10000", "0x10000-0x12000",
    "0x10001", "0x1000", "not-hex", "0x800000000000", "0x7ffffffff000",
])
def test_malformed_unaligned_overflowing_or_out_of_range_addresses_are_rejected(fake, address):
    fake.snapshots[0][0] = fake.snapshots[0][0]._replace(addr=address)
    with pytest.raises(ValueError, match="address"):
        capture()
    assert not fake.reads


@pytest.mark.parametrize("second", [BASE, BASE + PAGE])
def test_duplicate_or_overlapping_selected_regions_are_rejected(fake, second):
    fake.add_mapping(second)
    with pytest.raises(ValueError, match="overlap|duplicate"):
        capture()
    assert not fake.reads


@pytest.mark.parametrize("key,value", [
    ("address", BASE + PAGE), ("allocation_base", BASE - PAGE),
    ("region_bytes", PAGE), ("region_bytes", rounded(len(DATA)) + PAGE),
    ("state", 0x2000), ("state", 0), ("protect", 4), ("protect", 8),
    ("protect", 0x102), ("protect", 0x20), ("kind", 0x20000), ("kind", 0x1000000),
    ("region_bytes", True), ("address", str(BASE)),
])
def test_virtual_query_must_confirm_whole_committed_readonly_file_view(fake, key, value):
    fake.query_results[BASE][key] = value
    with pytest.raises(ValueError, match="VirtualQueryEx"):
        capture()
    assert not fake.reads


@pytest.mark.parametrize("result", [None, {}, PermissionError("query denied"), ValueError("query short")])
def test_virtual_query_failure_or_incomplete_result_is_rejected(fake, result):
    fake.query_change = lambda _count, _value: result
    with pytest.raises(ValueError):
        capture()
    assert not fake.reads and fake.closed == 1


def test_virtual_query_attributes_cannot_change_even_if_maps_path_and_hash_stay_equal(fake):
    def change(count, value):
        if count == 2:
            value["allocation_protect"] = 4
        return value

    fake.query_change = change
    with pytest.raises(ValueError, match="mapping changed"):
        capture()
    assert fake.maps_count == 1 and fake.reads


@pytest.mark.parametrize("reported", [lambda n: 0, lambda n: n - 1, lambda n: n + 1, lambda n: True, lambda n: None])
def test_short_zero_overreported_and_invalid_reads_are_rejected(fake, reported):
    fake.read_result = reported
    with pytest.raises(ValueError, match="short/invalid read"):
        capture()
    assert fake.closed == 1


def test_same_length_different_mapped_bytes_reject_even_when_caller_supplies_expected_hash(fake):
    fake.contents[BASE] = b"x" * len(DATA)
    with pytest.raises(ValueError, match="mapped bytes differ"):
        capture()
    assert fake.reads


def test_one_good_mapping_does_not_hide_another_mapping_of_different_bytes(fake):
    fake.add_mapping(0x50000, b"x" * len(DATA))
    with pytest.raises(ValueError, match="mapped bytes differ"):
        capture()
    assert [item[0] for item in fake.reads] == [BASE, 0x50000]


@pytest.mark.parametrize("field,value", [
    ("page_size", 0), ("page_size", -4096), ("page_size", 3000), ("page_size", True),
    ("minimum_address", 0), ("minimum_address", -1), ("maximum_address", BASE - 1),
])
def test_invalid_system_page_or_address_bounds_fail_closed(fake, field, value):
    setattr(fake, field, value)
    with pytest.raises(ValueError, match="bounds|page size"):
        capture()
    assert not fake.opened and not fake.reads


def test_impossible_artifact_size_is_rejected_before_allocation_or_process_open(fake):
    with pytest.raises(ValueError, match="exceeds application address bounds"):
        capture(expected_size=1 << 64)
    assert not fake.reads and not fake.opened


@pytest.mark.parametrize("size", [1, PAGE, PAGE + 1, 2 * 1024 * 1024 + 123])
def test_bounded_single_buffer_reads_exact_bytes_without_page_padding(fake, size):
    data = (bytes(range(251)) * (size // 251 + 1))[:size]
    replacement = FakeBackend(data)
    fake.__dict__.update(replacement.__dict__)
    digest = hashlib.sha256(data).hexdigest()
    result = capture(expected_size=size, expected_sha256=digest)
    assert result["regions"][0]["hashed_bytes"] == size
    assert sum(count for _, count, _, _ in fake.reads) == size
    assert all(0 < count <= capacity <= 2 * 1024 * 1024 for _, count, capacity, _ in fake.reads)
    assert len({buffer_id for _, _, _, buffer_id in fake.reads}) == 1
    assert all(address + count <= BASE + size for address, count, _, _ in fake.reads)
    assert result["regions"][0]["region_bytes"] == rounded(size)


def test_page_size_is_observed_instead_of_hardcoded(fake):
    fake.page_size = 8192
    assert capture()["page_size"] == 8192


# These tests read a real child process with real WinAPI calls. Its only mapped
# payloads are tiny temporary files, and it is idle during both map snapshots.
CHILD = r'''
import mmap, os, sys
path, count, writable = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
file = open(path, "r+b" if writable else "rb")
views = [mmap.mmap(file.fileno(), 0, access=mmap.ACCESS_WRITE if writable else mmap.ACCESS_READ)
         for _ in range(count)]
assert all(view[:4] == b"\x00\x01\x02\x03" for view in views)
print("ready", os.getpid(), flush=True)
try:
    sys.stdin.readline()
finally:
    for view in views: view.close()
    file.close()
'''


@pytest.fixture
def native_mapping(tmp_path):
    if sys.platform != "win32":
        pytest.skip("requires Windows ReadProcessMemory")
    import psutil

    original = tmp_path / "old-operation" / "segment.gguf"
    original.parent.mkdir()
    original.write_bytes(DATA)
    artifact = tmp_path / "current-operation" / "segment.gguf"
    artifact.parent.mkdir()
    processes = []

    def spawn(*, count=1, writable=False):
        process = subprocess.Popen(
            [sys.executable, "-B", "-X", "utf8", "-c", CHILD, str(original), str(count), str(int(writable))],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            text=True, creationflags=subprocess.CREATE_NO_WINDOW,
        )
        processes.append(process)
        ready = process.stdout.readline().split()
        assert len(ready) == 2 and ready[0] == "ready", (
            ready, process.stderr.read() if process.poll() is not None else "child not ready")
        # Windows venv executables can be redirector parents, so bind to the
        # interpreter PID reported by the mapping owner, not Popen's launcher.
        identity = psutil.Process(int(ready[1]))
        # Temp roots may use an 8.3 spelling and GetMappedFileName can retain
        # a hardlink name. Only this fixture locates its tiny file by file ID;
        # the helper still receives the exact observed path, unchanged.
        matches = []
        observed_maps = list(identity.memory_maps(grouped=False))
        for mapping in observed_maps:
            try:
                if os.path.samefile(mapping.path, original):
                    matches.append(mapping)
            except OSError:
                continue
        assert len(matches) == count, (str(original), [m for m in observed_maps if "segment" in m.path.lower()], len(observed_maps))
        assert len({m.path for m in matches}) == 1
        # Create the identity alias after mapping, so its name is not the one
        # cached by GetMappedFileName. No path repair happens in the helper.
        os.link(original, artifact)
        return dict(
            pid=identity.pid, created_at=identity.create_time(), mapped_path=matches[0].path,
            artifact_path=str(artifact), expected_sha256=hashlib.sha256(DATA).hexdigest(),
            expected_size=len(DATA),
        )

    yield SimpleNamespace(spawn=spawn, original=original, artifact=artifact)
    for process in processes:
        try:
            if process.stdin:
                process.stdin.close()
            process.wait(timeout=10)
        finally:
            if process.poll() is None:
                process.kill()
                process.wait(timeout=5)
            process.stdout.close()
            process.stderr.close()
        assert process.returncode == 0


def test_windows_real_two_mmaps_and_hardlink_use_memory_bytes(native_mapping):
    arguments = native_mapping.spawn(count=2)
    assert os.path.samefile(native_mapping.original, native_mapping.artifact)
    assert arguments["mapped_path"] != arguments["artifact_path"]
    result = helper.capture_windows_mapped_artifact(**arguments)
    assert result["maps_stable"] is True and len(result["regions"]) == 2
    assert all(r["hashed_bytes"] == len(DATA) and r["sha256"] == hashlib.sha256(DATA).hexdigest()
               for r in result["regions"])
    assert all(r["region_bytes"] == rounded(len(DATA), result["page_size"]) for r in result["regions"])
    assert len({r["address"] for r in result["regions"]}) == 2


def test_windows_real_same_size_disk_alias_cannot_replace_different_mapped_bytes(native_mapping, tmp_path):
    arguments = native_mapping.spawn()
    wrong = tmp_path / "wrong-current-segment.gguf"
    wrong.write_bytes(b"z" * len(DATA))
    arguments.update(artifact_path=str(wrong), expected_sha256=hashlib.sha256(wrong.read_bytes()).hexdigest())
    with pytest.raises(ValueError, match="mapped bytes differ"):
        helper.capture_windows_mapped_artifact(**arguments)


def test_windows_real_hardlink_name_is_not_an_exact_mapping_path(native_mapping):
    arguments = native_mapping.spawn()
    arguments["mapped_path"] = str(native_mapping.artifact)
    with pytest.raises(ValueError, match="no exact mapped_path"):
        helper.capture_windows_mapped_artifact(**arguments)


def test_windows_real_writable_file_mapping_is_rejected(native_mapping):
    arguments = native_mapping.spawn(writable=True)
    with pytest.raises(ValueError, match="read-only"):
        helper.capture_windows_mapped_artifact(**arguments)



