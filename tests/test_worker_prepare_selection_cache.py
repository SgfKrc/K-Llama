"""Env GGUF digest reuse never substitutes for native transaction receipts."""
from collections import OrderedDict
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import struct
import threading
import time
from types import SimpleNamespace

import pytest

from inference_service.engine_host import EngineHost
from llama_engine import LlamaCppEngine
from model_supply_digest_cache import file_stamp
from qlh_core.planning.load_spec import assignment_load_spec
from tests.test_relay_segment_info import _arch_kvs, _kv_str, _write_gguf
from tests.test_worker_prepare import Client, command, drain, identity, offer, run
from tests.test_worker_prepare_runtime import Upstream
from worker_prepare import WorkerPrepareController, resolve_prepare_selection
from worker_prepare_contract import WorkerPrepareError


@pytest.fixture
def env_prepare(tmp_path, monkeypatch):
    import keep_head_shim
    import llama_keep_head
    import model_supply_assets
    import model_supply_worker
    from relay_segment_info import read_gguf_tokenizer_sha256

    path = _write_gguf(tmp_path / "segment.gguf", _arch_kvs("qwen2", 2, n_embd=16) + [
        ("tokenizer.ggml.model", 8, _kv_str("gpt2")),
        ("tokenizer.ggml.tokens", 9, struct.pack("<IQ", 8, 2) + _kv_str("a") + _kv_str("b")),
    ])
    path.write_bytes(path.read_bytes() + bytes(64))
    value = identity(fmt="gguf")
    value.update(manifest_sha256="", artifact_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                 tokenizer_sha256=read_gguf_tokenizer_sha256(path))
    value["stage_model_identity"]["sha256"] = value["artifact_sha256"]
    spec = assignment_load_spec({}, artifact_format="gguf", context_size=64)
    value["load_spec_digest"] = spec.digest
    marker = path.with_suffix(".manifest.json")
    manifest = {**deepcopy(value), "source_layer_range": value["layer_range"],
                "n_layer": value["total_layers"], "mode": value["segment_mode"]}
    marker.write_text(json.dumps(manifest), encoding="utf-8")
    monkeypatch.setenv("QLH_LAYER_GGUF", str(path))
    monkeypatch.setattr(model_supply_worker, "get_selected_model", lambda *args: None)
    monkeypatch.setattr(keep_head_shim, "resolve_keep_head_shim", lambda **kwargs: "shim")
    monkeypatch.setattr(llama_keep_head, "KeepHeadUpstream", Upstream)
    monkeypatch.setattr(Upstream, "materialization_observation", lambda self: {
        **spec.to_dict(), "materialized_bytes": 128,
    }, raising=False)
    monkeypatch.setattr(Upstream, "made", [])
    reads = []
    original_hash = model_supply_assets._hash_file

    def counted(source, cancelled=None):
        reads.append(Path(source))
        return original_hash(source, cancelled)

    monkeypatch.setattr(model_supply_assets, "_hash_file", counted)
    host = EngineHost.__new__(EngineHost)
    host._layer_upstream = None
    scheduler = SimpleNamespace(
        _host=SimpleNamespace(full_chat_execution_lock=threading.RLock()),
        _tcp_client=Client(), _control_fence=None, _effective_role=lambda: "client",
        get_effective_node_id=lambda: "pc", get_distributed_inference_enabled=lambda: True,
        _configured_layer_artifact=lambda: value["stage_model_identity"],
        _task_worker_layer_artifact_identity=lambda artifact: artifact,
    )
    controller = WorkerPrepareController(scheduler, lambda: host)
    result = SimpleNamespace(host=host, scheduler=scheduler, controller=controller,
                             value=value, path=path, marker=marker, manifest=manifest,
                             reads=reads, load_spec=spec.to_dict())
    try:
        yield result
    finally:
        controller.close()
        assert not host.worker_runtime_present()
        assert all(not upstream.is_available for upstream in Upstream.made)


def prepare(case, value=None):
    return run(case, "prepare", value or case.value, load_spec=case.load_spec)


def rewrite_same_version_fields(path):
    before = path.stat()
    stamp = file_stamp(path)
    payload = path.read_bytes()
    time.sleep(0.02)
    path.write_bytes(payload[:-1] + bytes([payload[-1] ^ 1]))
    os.utime(path, ns=(before.st_atime_ns, before.st_mtime_ns))
    assert path.stat().st_size == before.st_size
    assert path.stat().st_mtime_ns == before.st_mtime_ns
    assert file_stamp(path) != stamp


def test_prepare_and_recheck_hash_once_but_keep_all_native_ack_barriers(env_prepare):
    case = env_prepare
    prepared = prepare(case)
    assert prepared["status"] == "prepared" and prepared["loaded"]
    assert case.reads == [case.path]
    assert len(Upstream.made) == 1
    with pytest.raises(WorkerPrepareError):
        case.controller.check_stage(offer(case.value))
    committed = run(case, "commit", case.value)
    assert committed["status"] == "ready" and committed["loaded"]
    assert committed["identity"] == prepared["identity"] == case.value
    assert committed["runtime_receipt"] == prepared["runtime_receipt"]
    case.controller.check_stage(offer(case.value))
    assert run(case, "finalize", case.value)["status"] == "finalized"
    assert prepare(case)["status"] == "prepared"
    assert run(case, "commit", case.value)["status"] == "ready"
    assert case.reads == [case.path] and len(Upstream.made) == 1


def test_shared_digest_survives_retirement_but_new_identity_reloads_native(env_prepare):
    case = env_prepare
    assert LlamaCppEngine()._pipeline_file_sha256(str(case.path)) == case.value["artifact_sha256"]
    assert case.reads == [case.path]
    for value in (case.value, {**case.value, "operation_id": "op-new", "config_id": "cfg-new",
                              "assignment_id": "assignment-new", "generation": 2}):
        assert prepare(case, value)["status"] == "prepared"
        assert run(case, "commit", value)["status"] == "ready"
        assert run(case, "finalize", value)["status"] == "finalized"
        assert run(case, "release", value, cleanup_id="release-native")["status"] == "released"
        assert not case.host.worker_runtime_present()
    assert case.reads == [case.path]
    assert len(Upstream.made) == 2 and all(not upstream.is_available for upstream in Upstream.made)
    assert case.scheduler._tcp_client.acks[0]["identity"] != case.scheduler._tcp_client.acks[4]["identity"]


@pytest.mark.parametrize("change", ["in_place", "replace", "remove"])
def test_warm_digest_rejects_changed_artifact_before_native_load(env_prepare, change):
    case = env_prepare
    resolve_prepare_selection(case.scheduler, case.value)
    assert case.reads == [case.path]
    if change == "in_place":
        rewrite_same_version_fields(case.path)
    elif change == "replace":
        before = case.path.stat()
        payload = case.path.read_bytes()
        replacement = case.path.with_suffix(".replacement")
        replacement.write_bytes(payload[:-1] + b"x")
        os.utime(replacement, ns=(before.st_atime_ns, before.st_mtime_ns))
        os.replace(replacement, case.path)
    else:
        case.path.unlink()
    result = prepare(case)
    assert result["status"] == "error" and not result["loaded"]
    assert Upstream.made == []
    assert len(case.reads) == (1 if change == "remove" else 2)


@pytest.mark.parametrize("change", ["manifest", "tokenizer", "range", "model"])
def test_warm_digest_does_not_skip_manifest_or_identity_validation(env_prepare, change):
    case = env_prepare
    resolve_prepare_selection(case.scheduler, case.value)
    if change == "manifest":
        case.marker.unlink()
    elif change == "model":
        case.scheduler._configured_layer_artifact = lambda: {**case.value["stage_model_identity"], "revision": "other"}
    else:
        field, replacement = ("tokenizer_sha256", "d" * 64) if change == "tokenizer" else ("source_layer_range", [0, 2])
        case.marker.write_text(json.dumps({**case.manifest, field: replacement}), encoding="utf-8")
    result = prepare(case)
    assert result["status"] == "error" and not result["loaded"]
    assert case.reads == [case.path] and Upstream.made == []


def test_artifact_change_during_native_load_fails_recheck_without_prepared_ack(env_prepare, monkeypatch):
    case = env_prepare
    original = Upstream.__init__

    def changed(upstream, *args, **kwargs):
        original(upstream, *args, **kwargs)
        rewrite_same_version_fields(case.path)

    monkeypatch.setattr(Upstream, "__init__", changed)
    assert prepare(case)["status"] == "error"
    assert case.reads == [case.path, case.path]
    assert not any(ack["status"] in {"prepared", "ready"} for ack in case.scheduler._tcp_client.acks)
    with pytest.raises(WorkerPrepareError):
        case.controller.check_stage(offer(case.value))
    assert run(case, "abort", case.value)["status"] == "released"
    assert not Upstream.made[0].is_available and not case.host.worker_runtime_present()


@pytest.mark.parametrize("change", ["cancel", "reconnect"])
def test_warm_digest_cannot_publish_native_completion_after_fence_change(env_prepare, monkeypatch, change):
    case = env_prepare
    resolve_prepare_selection(case.scheduler, case.value)
    entered, complete = threading.Event(), threading.Event()
    original = Upstream.__init__

    def blocked(upstream, *args, **kwargs):
        original(upstream, *args, **kwargs)
        entered.set()
        assert complete.wait(3)

    monkeypatch.setattr(Upstream, "__init__", blocked)
    assert case.controller.handle(command("prepare", case.value, load_spec=case.load_spec))
    try:
        assert entered.wait(3)
        if change == "cancel":
            assert case.controller.handle(command("abort", case.value))
        else:
            case.scheduler._tcp_client._connection_generation += 1
            case.controller.invalidate("disconnect")
    finally:
        complete.set()
        drain(case.controller)
    assert not any(ack["status"] in {"prepared", "ready"} for ack in case.scheduler._tcp_client.acks)
    assert case.reads == [case.path]
    assert not Upstream.made[0].is_available and not case.host.worker_runtime_present()


def test_warm_digest_and_descriptor_cannot_hide_dead_native_sidecar(env_prepare):
    case = env_prepare
    assert prepare(case)["status"] == "prepared"
    assert run(case, "commit", case.value)["status"] == "ready"
    assert run(case, "finalize", case.value)["status"] == "finalized"
    Upstream.made[0].is_available = False
    result = prepare(case)
    assert result["status"] == "error" and not result["loaded"]
    assert result["error"] == "worker_prepare_runtime_unavailable"
    assert len(Upstream.made) == 1 and case.reads == [case.path]
    with pytest.raises(WorkerPrepareError):
        case.controller.check_stage(offer(case.value))


def test_prepare_recheck_uses_existing_versioned_parser_caches(env_prepare, monkeypatch):
    import gguf_metadata_contract
    import relay_segment_info
    import model_supply_digest_cache

    case = env_prepare
    monkeypatch.setattr(relay_segment_info, "_GGUF_HEADER_CACHE", OrderedDict())
    stamp = model_supply_digest_cache.file_stamp(case.path)
    entry = model_supply_digest_cache._record(
        model_supply_digest_cache._store_path(), case.path.resolve(), stamp,
    )
    metadata = entry.get("metadata", {})
    metadata.pop("relay-gguf-reader-v2-read_gguf_tokenizer_sha256", None)
    model_supply_digest_cache._save(
        model_supply_digest_cache._store_path(), case.path.resolve(), entry,
    )
    parses = {"tokenizer": 0, "table": 0}
    original_tokenizer_header = relay_segment_info._gguf_header
    original_table = gguf_metadata_contract._read_gguf_header

    def tokenizer_header(*args, **kwargs):
        parses["tokenizer"] += 1
        return original_tokenizer_header(*args, **kwargs)

    def table(*args, **kwargs):
        parses["table"] += 1
        return original_table(*args, **kwargs)

    monkeypatch.setattr(relay_segment_info, "_gguf_header", tokenizer_header)
    monkeypatch.setattr(gguf_metadata_contract, "_read_gguf_header", table)
    for value in (case.value, {**case.value, "operation_id": "op-new", "config_id": "cfg-new",
                              "assignment_id": "assignment-new", "generation": 2}):
        assert prepare(case, value)["status"] == "prepared"
        assert parses == {"tokenizer": 1, "table": 1}
        assert run(case, "commit", value)["status"] == "ready"
        assert run(case, "finalize", value)["status"] == "finalized"
        assert run(case, "release", value, cleanup_id="release-parsers")["status"] == "released"
    assert case.reads == [case.path] and len(Upstream.made) == 2

    rewrite_same_version_fields(case.path)
    changed = deepcopy(case.value)
    changed["artifact_sha256"] = hashlib.sha256(case.path.read_bytes()).hexdigest()
    changed["stage_model_identity"]["sha256"] = changed["artifact_sha256"]
    case.marker.write_text(json.dumps({**case.manifest, "artifact_sha256": changed["artifact_sha256"]}), encoding="utf-8")
    case.scheduler._configured_layer_artifact = lambda: changed["stage_model_identity"]
    assert resolve_prepare_selection(case.scheduler, changed)["artifact_sha256"] == changed["artifact_sha256"]
    assert parses == {"tokenizer": 2, "table": 2}
    assert case.reads == [case.path, case.path]
