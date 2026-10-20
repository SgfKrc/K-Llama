"""Offline selective mmap provenance tests; no CUDA execution or product services."""
from copy import deepcopy
import os
from pathlib import Path
import sys
import threading
from types import SimpleNamespace as NS

import pytest
import psutil

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

import model_module
from package_contract.product_observations import derive_node_observation, ObservationError
from product_runtime_receipt import capture_runtime, RuntimeObservationError
from qlh_core.planning.load_spec import assignment_load_spec
from tests.product_observation_fixtures import node_observation
from tests.test_product_runtime_receipt import _api
from tests.test_product_runtime_integration import _raw
from tests.test_safetensors_supply_worker import tiny_supply
from tests.test_worker_prepare import command
from tests.test_worker_prepare_runtime import supply_identity


def example(directory="/opt/qlh/segments/proper"):
    assignments = [{"node_id": "master", "engine": "pytorch", "start_layer": 0, "end_layer": 2},
                   {"node_id": "pc", "engine": "llama_cpp", "start_layer": 2, "end_layer": 4}]
    identity = {"operation_id": "op", "config_id": "cfg", "generation": 3}
    raw, _, _, addresses = node_observation(identity, assignments[0], assignments, "a" * 40, 4)
    runtime = raw["facts"]["runtime"]
    runtime["artifact"]["path"] = directory
    proof = runtime["materialization"]
    proof["artifact_path"] = directory
    shards = [directory + "/model-00001-of-00002.safetensors",
              directory + "/model-00002-of-00002.safetensors"]
    for item, shard in zip(proof["tensor_metadata"].values(), shards):
        item.update(source_file=shard, source_element_size=2)
    proof["layer_offload"] = {"materialization": "safetensors_layer_offload",
        "source_device": "cpu", "source_dtype": "artifact", "active": False,
        "execution_device": "cuda:0", "execution_dtype": "float16",
        "resident_source_bytes": 16, "host_conversion_bytes": 0,
        "device_weight_bytes": 8, "target_weight_bytes": 16}
    raw["facts"]["process_tree"]["processes"][0]["model_mappings"] = shards
    expected = dict(expected_revision="a" * 40, expected_node_id="master", expected_identity=identity,
                    expected_assignment=assignments[0], node_addresses=addresses)
    return raw, expected


@pytest.mark.parametrize("directory", ["/opt/qlh/segments/proper", "C:/qlh/segments/proper"])
def test_verified_offload_allows_only_proof_listed_multi_shard_cpu_maps(directory):
    raw, expected = example(directory)
    before = deepcopy(raw)
    if directory.startswith("C:"):
        root = raw["facts"]["process_tree"]["processes"][0]
        root["model_mappings"] = [path.upper().replace("/", "\\") for path in root["model_mappings"]]
        before = deepcopy(raw)
    assert derive_node_observation(raw, **expected)["native"]["layer_range"] == [0, 2]
    assert raw == before


@pytest.mark.parametrize("extra", ["/opt/qlh/segments/proper/unknown.safetensors",
    "/opt/qlh/full/model.safetensors", "/opt/qlh/segments/proper-extra/model.safetensors",
    "/opt/qlh/segments/proper/../full/model.safetensors",
    "/opt/qlh/segments/proper/full.gguf"])
def test_unknown_outside_and_full_model_extra_mappings_still_reject(extra):
    raw, expected = example()
    raw["facts"]["process_tree"]["processes"][0]["model_mappings"].append(extra)
    with pytest.raises(ObservationError, match="unexpected/unproven/full-model mapping"):
        derive_node_observation(raw, **expected)


@pytest.mark.parametrize("field,value", [("materialization", "safetensors_key_filtered"),
    ("source_device", "cuda:0"), ("source_dtype", "float16"), ("active", True),
    ("execution_device", "cpu"), ("execution_dtype", "bfloat16"),
    ("resident_source_bytes", 15), ("resident_source_bytes", True),
    ("device_weight_bytes", 17), ("device_weight_bytes", 0),
    ("host_conversion_bytes", 9), ("host_conversion_bytes", -1), ("target_weight_bytes", None)])
def test_malformed_actual_offload_weight_domains_do_not_authorize_maps(field, value):
    raw, expected = example()
    raw["facts"]["runtime"]["materialization"]["layer_offload"][field] = value
    with pytest.raises(ObservationError):
        derive_node_observation(raw, **expected)


@pytest.mark.parametrize("change", ["no_actual_offload", "no_rehash", "wrong_directory",
    "outside_source", "cross_drive_source", "relative_source", "directory_source",
    "non_safetensors_source", "converted_source", "wrong_range", "extra_tensor"])
def test_policy_or_unverified_tensor_metadata_cannot_authorize_source_mmap(change):
    raw, expected = example()
    runtime = raw["facts"]["runtime"]
    proof = runtime["materialization"]
    item = next(iter(proof["tensor_metadata"].values()))
    if change == "no_actual_offload":
        proof.pop("layer_offload")
    elif change == "no_rehash":
        runtime["artifact"].pop("rehashed_sha256")
    elif change == "wrong_directory":
        proof["artifact_path"] = "/opt/qlh/other"
    elif change == "outside_source":
        item["source_file"] = "/opt/qlh/full/model.safetensors"
    elif change == "cross_drive_source":
        item["source_file"] = "D:/full/model.safetensors"
    elif change == "relative_source":
        item["source_file"] = "proper/model.safetensors"
    elif change == "directory_source":
        item["source_file"] = runtime["artifact"]["path"]
    elif change == "non_safetensors_source":
        item["source_file"] = runtime["artifact"]["path"] + "/full.gguf"
    elif change == "converted_source":
        item["source_element_size"] = 4
    elif change == "wrong_range":
        proof["layer_range"] = [0, 4]
    else:
        proof["loaded_tensor_names"].append("model.layers.3.weight")
        proof["tensor_metadata"]["model.layers.3.weight"] = deepcopy(item)
    with pytest.raises(ObservationError):
        derive_node_observation(raw, **expected)


def test_parent_observation_does_not_authorize_child_source_mappings():
    raw, expected = example()
    root = raw["facts"]["process_tree"]["processes"][0]
    child = deepcopy(root)
    child.update(pid=202, parent_pid=root["pid"], child_pids=[], connections=[],
                 identity_after={"pid": 202, "created_at": child["created_at"]})
    root["child_pids"] = [202]
    raw["facts"]["process_tree"]["processes"].append(child)
    with pytest.raises(ObservationError, match="unexpected/unproven/full-model mapping"):
        derive_node_observation(raw, **expected)


@pytest.fixture
def actual_offload(tiny_supply, monkeypatch):
    manifest = tiny_supply.activate(0)
    value = supply_identity(manifest, 1)
    spec = assignment_load_spec({"execution_device": "cuda:0", "has_embedding": True,
        "runtime_materialization": "safetensors_layer_offload"}, artifact_format="safetensors", context_size=64)
    value["load_spec_digest"] = spec.digest
    # Load only small CPU mmap tensors; no forward/transfer is performed.
    monkeypatch.setattr(model_module, "_layer_offload_device_available", lambda device: True)
    from model_supply_worker import get_selected_model
    host = tiny_supply.host()
    try:
        host.prepare_worker_runtime(command("prepare", value, load_spec=spec.to_dict()),
                                    get_selected_model(), threading.Event())
        host.commit_worker_runtime(value)
        host.finalize_worker_runtime(value)
        yield NS(manager=host._worker_runtime_active.manager, manifest=manifest)
    finally:
        host.release_worker_runtime()



def _windows_tensor_mapped_paths(tensors):
    # GetMappedFileNameW (also used by psutil) can return ERROR_MORE_DATA
    # when the NT device name exceeds MAX_PATH, even with a larger output
    # buffer. Query each live tensor address directly with a full-size native
    # buffer. GLOBALROOT opens that observed device path without guessing a
    # drive letter or rewriting an 8.3/hardlink name.
    import ctypes
    from ctypes import wintypes

    class UnicodeString(ctypes.Structure):
        _fields_ = [("length", wintypes.USHORT), ("maximum_length", wintypes.USHORT),
                    ("buffer", ctypes.c_void_p)]

    kernel = ctypes.WinDLL("kernel32")
    current = kernel.GetCurrentProcess
    current.argtypes, current.restype = [], wintypes.HANDLE
    query = ctypes.WinDLL("ntdll").NtQueryVirtualMemory
    query.argtypes = [wintypes.HANDLE, ctypes.c_void_p, ctypes.c_int,
                      ctypes.c_void_p, ctypes.c_size_t, ctypes.POINTER(ctypes.c_size_t)]
    query.restype = ctypes.c_long
    paths = set()
    for tensor in tensors:
        assert tensor.device.type == "cpu"
        if not tensor.numel():
            continue
        output = ctypes.create_string_buffer(65536 + ctypes.sizeof(UnicodeString))
        returned = ctypes.c_size_t()
        status = query(current(), ctypes.c_void_p(tensor.data_ptr()), 2,
                       output, len(output), ctypes.byref(returned))
        # Private CPU allocations, including generated buffers and copies of
        # weights, have no mapped filename. They provide no mmap evidence.
        if status & 0xffffffff == 0xc0000141:  # STATUS_INVALID_ADDRESS
            continue
        assert status == 0, f"tensor mapped-filename query failed: {status & 0xffffffff:#x}"
        name = UnicodeString.from_buffer(output)
        begin, end = ctypes.addressof(output), ctypes.addressof(output) + returned.value
        assert (ctypes.sizeof(UnicodeString) <= returned.value <= len(output)
                and name.length > 0 and name.length % 2 == 0
                and begin <= name.buffer < name.buffer + name.length <= end)
        path = ctypes.wstring_at(name.buffer, name.length // 2)
        assert path.startswith("\\Device\\"), path
        paths.add("\\\\?\\GLOBALROOT" + path)
    return paths


def _assert_source_mmaps(sources, tensors):
    if os.name == "nt":
        paths = _windows_tensor_mapped_paths(tensors)
    else:
        paths = {item.path for item in psutil.Process().memory_maps() if item.path}
    identities = {}
    for path in paths:
        try:
            info = os.stat(path)
        except OSError:
            continue
        if info.st_ino:
            identities[(info.st_dev, info.st_ino)] = path
    missing = []
    for source in sources:
        info = os.stat(source)
        assert info.st_ino, f"source file identity unavailable: {source}"
        if (info.st_dev, info.st_ino) not in identities:
            missing.append(source)
    assert not missing, f"source shards have no live CPU mmap: {missing}; observed: {sorted(paths)}"


@pytest.mark.skipif(os.name != "nt", reason="Windows native mapped filename/file identity")
def test_real_long_path_mmap_accepts_hardlink_but_rejects_copies(tmp_path):
    import torch

    directory = tmp_path
    while len(str(directory)) < 270:
        directory /= "long-directory-" + "x" * 48
    directory.mkdir(parents=True)
    source = directory / "assignment-00001-of-00001.safetensors"
    data = bytes(range(256)) * 16
    source.write_bytes(data)
    alias = tmp_path / "same-file-alias.safetensors"
    os.link(source, alias)
    other = tmp_path / "different-file-same-bytes.safetensors"
    other.write_bytes(data)
    tensor = torch.from_file(str(source), shared=False, size=len(data), dtype=torch.uint8)
    try:
        assert os.path.samefile(source, alias)
        _assert_source_mmaps({str(source), str(alias)}, [tensor])
        with pytest.raises(AssertionError, match="no live CPU mmap"):
            _assert_source_mmaps({str(other)}, [tensor])
        with pytest.raises(AssertionError, match="no live CPU mmap"):
            _assert_source_mmaps({str(source)}, [tensor.clone()])
    finally:
        del tensor


def test_real_cpu_mmap_and_actual_weight_domains_pass_independent_rehash_gate(actual_offload):
    manager = actual_offload.manager
    runtime = capture_runtime(_api(manager))
    proof = runtime["materialization"]
    assert proof["source"] == "safetensors_key_filtered"
    assert proof["layer_offload"] == manager.layer_offload_observation()
    assert proof["layer_offload"]["stage_count"] == 0
    assert proof["layer_offload"]["host_conversion_bytes"] > 0
    assert all(t.device.type == "cpu" for t in manager.model.parameters())
    sources = {os.path.normcase(os.path.abspath(item["source_file"]))
               for item in proof["tensor_metadata"].values()}
    # Match observed live tensor mappings by volume/file ID. Windows may
    # expose a long device name, an 8.3 spelling or another hardlink spelling.
    _assert_source_mmaps(sources, manager.model.parameters())
    raw, expected = _raw(actual_offload)
    raw["facts"]["process_tree"]["processes"][0]["model_mappings"] = sorted(sources)
    derived = derive_node_observation(raw, **expected)
    assert derived["native"]["layer_range"] == [0, 2]
    proof["layer_offload"]["device_weight_bytes"] = 1
    assert manager.layer_offload_observation()["device_weight_bytes"] > 1


def test_runtime_capture_rejects_unavailable_live_offload_observation(actual_offload, monkeypatch):
    def unavailable():
        raise ValueError("layer_offload_runtime_unavailable")
    monkeypatch.setattr(actual_offload.manager, "layer_offload_observation", unavailable)
    with pytest.raises(RuntimeObservationError, match="actual layer offload observation unavailable"):
        capture_runtime(_api(actual_offload.manager))
