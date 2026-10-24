"""Small, offline source-acquisition fixtures; no model runtime or large downloads."""

from __future__ import annotations

import hashlib
import json
import shutil
import struct
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace

import pytest

import local_store
import model_config as mc
import model_download_jobs as jobs
import model_registry_validation as mrv
import model_supply_sources as sources
import pipeline_model_descriptor as descriptor
from model_config import ModelConfig
from relay_segment_info import read_gguf_tokenizer_sha256
from scripts.model_tools import gguf_convert as convert

_REAL_PLAN = convert.plan_conversion
_REAL_EXECUTE = convert.execute_conversion
_REAL_DOWNLOAD = jobs._download


def test_source_identity_is_shared_with_segment_builders_and_invalidation(tmp_path, monkeypatch):
    import model_supply_assets as assets
    path = tmp_path / "source.gguf"
    path.write_bytes(b"original-weights")
    expected = hashlib.sha256(path.read_bytes()).hexdigest()
    assert sources._content_sha(path, lambda: False) == expected
    original = assets._hash_file
    reads = []
    def record(file_path, *args, **kwargs):
        reads.append(Path(file_path))
        return original(file_path, *args, **kwargs)
    monkeypatch.setattr(assets, "_hash_file", record)
    assert assets._hash_file_cached(path) == expected
    assert sources._content_sha(path, lambda: False) == expected
    assert reads == []
    _assert_code("SUPPLY_CANCELLED", lambda: sources._content_sha(path, lambda: True))
    path.write_bytes(b"new-weights-with-new-size")
    changed = hashlib.sha256(path.read_bytes()).hexdigest()
    assert sources._content_sha(path, lambda: False) == changed
    assert assets._hash_file_cached(path) == changed
    assert reads == [path]


def test_source_bundle_identity_reuses_source_verification_digests(tmp_path, monkeypatch):
    import model_supply_assets as assets
    from model_sync import compute_model_sha256
    path = tmp_path / "model.safetensors"
    path.write_bytes(b"verified-source-weights")
    file_digest = sources._content_sha(path, lambda: False)
    expected = hashlib.sha256(f"{path.name}\0{path.stat().st_size}\0{file_digest}\n".encode()).hexdigest()
    monkeypatch.setattr(assets, "_hash_file", lambda *args, **kwargs: pytest.fail("source bundle reread weight bytes"))
    assert compute_model_sha256(str(tmp_path), digest=assets._hash_file_cached) == expected


def _hf(root: Path, *, manifest: bool = False, tokenizer: bool = True) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    config = {"architectures": ["Qwen2ForCausalLM"], "model_type": "qwen2",
              "num_hidden_layers": 2, "hidden_size": 2}
    (root / "config.json").write_text(json.dumps(config), encoding="utf-8")
    (root / "tokenizer_config.json").write_text("{}", encoding="utf-8")
    if tokenizer:
        (root / "tokenizer.json").write_text('{"version":1}', encoding="utf-8")
    header = {
        "model.layers.0.input_layernorm.weight": {"dtype": "F16", "shape": [2], "data_offsets": [0, 4]},
        "model.layers.1.input_layernorm.weight": {"dtype": "F16", "shape": [2], "data_offsets": [4, 8]},
    }
    encoded = json.dumps(header, separators=(",", ":")).encode()
    (root / "model.safetensors").write_bytes(struct.pack("<Q", len(encoded)) + encoded + b"\0" * 8)
    if manifest:
        mrv.write_manifest(root, mrv.build_manifest(root, [root / "model.safetensors"],
                                                   model_type="safetensors"))
    return root


def _string(value: str) -> bytes:
    raw = value.encode()
    return struct.pack("<Q", len(raw)) + raw


def _gguf(path: Path, *, tokens: bool = True) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    metadata = [
        _string("general.architecture") + struct.pack("<I", 8) + _string("qwen2"),
        _string("qwen2.block_count") + struct.pack("<II", 4, 2),
        _string("tokenizer.ggml.model") + struct.pack("<I", 8) + _string("gpt2"),
    ]
    if tokens:
        metadata.append(_string("tokenizer.ggml.tokens") + struct.pack("<IIQ", 9, 8, 2)
                        + _string("a") + _string("b"))
    tensor = _string("blk.0.weight") + struct.pack("<IQIQ", 1, 4, 0, 0)
    header = b"GGUF" + struct.pack("<IQQ", 3, 1, len(metadata)) + b"".join(metadata) + tensor
    padding = (-len(header)) % 32
    path.write_bytes(header + b"\0" * padding + b"\0" * 16)
    return path


def _register(tmp_path: Path, *, root: Path | None = None, gguf: Path | None = None,
              model_id: str = "fixture", repo: str = "", **fields) -> ModelConfig:
    cfg = ModelConfig(model_id=model_id, name="Source fixture", model_type="safetensors",
                      model_path=str(root) if root else str(tmp_path / "models" / model_id),
                      gguf_path=str(gguf) if gguf else "", huggingface_id=repo)
    snapshot = local_store.get_local_model_registry_snapshot(model_id)
    assert local_store.save_local_experimental_model(model_id, {**asdict(cfg), **fields}, expected_revision=snapshot.registry_revision)
    return cfg


def _ensure(tmp_path: Path, model_id: str = "fixture", *, want_gguf=False,
            cancelled=lambda: False, progress=None):
    return sources.ensure_registered_sources(model_id, operation_dir=tmp_path / "operation",
                                              want_gguf=want_gguf, cancelled=cancelled,
                                              progress=progress)


def _assert_code(code: str, action):
    with pytest.raises(sources.ModelSupplySourceError) as exc:
        action()
    assert exc.value.code == code
    return exc.value


def _unexpected(*args, **kwargs):
    pytest.fail("Unexpected download, conversion or inference call")


@pytest.fixture
def fake_download(tmp_path, monkeypatch):
    template = _hf(tmp_path / "provider")
    calls = []

    def model_info(repo_id, **options):
        return SimpleNamespace(sha=options.get("revision") or "a" * 40,
                               siblings=[SimpleNamespace(rfilename=item.name, size=item.stat().st_size)
                                         for item in template.iterdir() if item.is_file()])

    monkeypatch.setattr("huggingface_hub.HfApi.model_info", lambda self, *a, **kw: model_info(*a, **kw))

    def download(repo, staging, **options):
        calls.append({"repo": repo, "staging": staging, **options})
        staging.mkdir(parents=True, exist_ok=True)
        for item in template.iterdir():
            shutil.copyfile(item, staging / item.name)
        weights = sorted(staging.glob("*.safetensors"))
        total = sum(item.stat().st_size for item in weights)
        options["progress_cb"](weights[0], total, total)
        return weights

    monkeypatch.setattr(jobs, "_download", download)
    return SimpleNamespace(template=template, calls=calls, model_info=model_info)


@pytest.fixture
def fake_conversion(tmp_path, monkeypatch):
    converter = tmp_path / "tools" / "convert_hf_to_gguf.py"
    converter.parent.mkdir()
    converter.write_text("# small converter fixture\n", encoding="utf-8")
    quantizer = converter.parent / "quantize.py"
    quantizer.write_text("# small quantizer fixture\n", encoding="utf-8")
    calls = []
    monkeypatch.setattr(convert, "_find_converter", lambda explicit: (converter, "available"))
    info = {"status": "available", "sha256": mrv.sha256_file(quantizer),
            "revision": "c" * 40, "verification": "verified", "provenance": "managed_package"}
    monkeypatch.setattr(convert, "_find_quantizer", lambda explicit: (quantizer, dict(info)))

    def plan(**options):
        return {"valid": True, "output_type": "Q4_K_M", "errors": [],
                "space": {"required_free_bytes": 0}}

    def execute(**options):
        calls.append(options)
        target = _gguf(options["target"])
        artifact = {"sha256": mrv.sha256_file(target)}
        return {"valid": True, "execution": {"started": True, "published": True, "artifact": artifact,
                "stages": [{"stage": name, "status": "passed"} for name in
                           ("convert_hf_to_gguf", "inspect_intermediate", "llama_quantize",
                            "inspect_verify", "atomic_publish")]}}

    monkeypatch.setattr(convert, "plan_conversion", plan)
    monkeypatch.setattr(convert, "execute_conversion", execute)
    return SimpleNamespace(converter=converter, quantizer=quantizer, info=info,
                           calls=calls, plan=plan, execute=execute)


def _old_proof(hf: Path, target: Path) -> dict:
    manifest = json.loads((hf / "model.manifest.json").read_text(encoding="utf-8"))
    payload = {"schema_version": 1, "kind": "qlh.cross_format_pipeline_identity.v1",
               "source": {"artifact_sha256": manifest["artifact_sha256"],
                          "manifest_sha256": manifest["manifest_sha256"],
                          "tokenizer_sha256": mrv.sha256_file(hf / "tokenizer.json")},
               "conversion": {"output_type": "Q4_K_M", "converter_revision": "a" * 40,
                              "converter_sha256": "b" * 64, "quantizer_revision": "c" * 40,
                              "quantizer_sha256": "d" * 64},
               "target": {"path": Path("..", target.parent.name, target.name).as_posix(),
                          "artifact_sha256": mrv.sha256_file(target),
                          "tokenizer_sha256": read_gguf_tokenizer_sha256(target)}}
    payload["provenance_sha256"] = sources._canonical(payload)
    (hf / sources.PROVENANCE).write_text(json.dumps(payload), encoding="utf-8")
    return payload


def test_unknown_id_is_named_and_never_interpreted_as_path(tmp_path, monkeypatch):
    monkeypatch.setattr(jobs, "_download", _unexpected)
    _assert_code("SUPPLY_MODEL_NOT_REGISTERED", lambda: _ensure(tmp_path, "../models/unregistered"))


def test_reuses_db_hf_with_header_inspection_and_no_runtime(tmp_path, monkeypatch):
    hf = _hf(tmp_path / "models" / "source")
    cfg = _register(tmp_path, root=hf, repo="owner/repo")
    monkeypatch.setattr(jobs, "_download", _unexpected)
    monkeypatch.setattr(convert, "execute_conversion", _unexpected)
    import builtins
    original_import = builtins.__import__

    def guarded(name, *args, **kwargs):
        if name.split(".")[0] in {"torch", "transformers", "llama_cpp", "model_module", "llama_engine"}:
            pytest.fail("Source preparation imported an inference runtime")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded)
    result = _ensure(tmp_path)
    assert result["hf"] == hf and result["gguf"] is None and result["conversion_proof"] is None
    assert result["model_config"].model_id == cfg.model_id
    assert result["model_config"].location == "external"
    assert not (hf / "model.manifest.json").exists()


def test_local_store_location_overrides_builtin_stale_path(tmp_path, monkeypatch):
    hf = _hf(tmp_path / "registered")
    builtin = ModelConfig(model_id="fixture", name="Builtin", model_path=str(tmp_path / "missing"))
    monkeypatch.setattr(mc, "get_builtin_model", lambda mid: builtin if mid == "fixture" else None)
    _register(tmp_path, root=hf)
    monkeypatch.setattr(jobs, "_download", _unexpected)
    assert _ensure(tmp_path)["hf"] == hf
    assert _ensure(tmp_path)["model_config"].name == "Source fixture"


def test_builtin_without_db_source_is_reused(tmp_path, monkeypatch):
    hf = _hf(tmp_path / "builtin")
    cfg = ModelConfig(model_id="fixture", name="Builtin", model_path=str(hf))
    monkeypatch.setattr(mc, "get_builtin_model", lambda mid: cfg if mid == "fixture" else None)
    assert _ensure(tmp_path)["hf"] == hf


@pytest.mark.parametrize("repo", ["", "https://example.invalid/model", "../private", "owner/repo;command"])
def test_missing_executable_source_has_named_failure(tmp_path, monkeypatch, repo):
    _register(tmp_path, repo=repo)
    monkeypatch.setattr(jobs, "_download", _unexpected)
    _assert_code("SUPPLY_SOURCE_MISSING", lambda: _ensure(tmp_path))


def test_present_invalid_source_does_not_trigger_download(tmp_path, monkeypatch):
    hf = tmp_path / "source"
    hf.mkdir()
    _register(tmp_path, root=hf, repo="owner/repo")
    monkeypatch.setattr(jobs, "_download", _unexpected)
    _assert_code("SUPPLY_SOURCE_METADATA_INVALID", lambda: _ensure(tmp_path))


def test_download_uses_existing_job_states_and_only_disk_gate(tmp_path, monkeypatch, fake_download):
    cfg = _register(tmp_path, repo="owner/repo")
    monkeypatch.setattr(jobs, "_resource_gate", _unexpected)
    monkeypatch.setattr(jobs, "_available_ram_bytes", lambda: 0)
    statuses = []
    original = jobs._update_job

    def update(job_id, **fields):
        if "status" in fields:
            statuses.append(fields["status"])
        original(job_id, **fields)

    monkeypatch.setattr(jobs, "_update_job", update)
    events = []
    result = _ensure(tmp_path, progress=events.append)
    assert result["hf"] == Path(cfg.model_path)
    assert result["gguf"] is None
    assert len(fake_download.calls) == 1
    assert fake_download.calls[0]["repo"] == "owner/repo"
    assert fake_download.calls[0]["revision"] == "a" * 40
    assert statuses == ["downloading", "verifying", "registering", "ready"]
    job = jobs.list_jobs()[0]
    assert job["status"] == "ready" and job["total_bytes"] > 0
    row = local_store.get_local_experimental_models()[0]
    assert row["huggingface_id"] == "owner/repo"
    assert row["supply_download_job_id"] == job["job_id"]
    sources._live_manifest(result["hf"], [result["hf"] / "model.safetensors"], lambda: False)
    assert any(event.get("downloaded_bytes", 0) > 0 for event in events)
    assert len(fake_download.calls) == 1
    _ensure(tmp_path)
    assert len(fake_download.calls) == 1


def test_low_disk_blocks_before_download(tmp_path, monkeypatch, fake_download):
    _register(tmp_path, repo="owner/repo")
    monkeypatch.setattr(sources.shutil, "disk_usage", lambda path: SimpleNamespace(free=0))
    _assert_code("SUPPLY_INSUFFICIENT_DISK", lambda: _ensure(tmp_path))
    assert fake_download.calls == []


def test_remote_sizes_admit_disk_not_whole_model_memory(tmp_path, monkeypatch, fake_download):
    _register(tmp_path, repo="owner/repo")
    original = fake_download.model_info

    def info(self, *args, **options):
        result = original(*args, **options)
        for item in result.siblings:
            if item.rfilename.endswith("safetensors"):
                item.size = 9 * 1024**3
        return result

    monkeypatch.setattr("huggingface_hub.HfApi.model_info", info)
    monkeypatch.setattr(sources.shutil, "disk_usage", lambda path: SimpleNamespace(free=2 * 1024**3))
    _assert_code("SUPPLY_INSUFFICIENT_DISK", lambda: _ensure(tmp_path))
    assert not fake_download.calls
    assert jobs.list_jobs()[0]["error_code"] == "SUPPLY_INSUFFICIENT_DISK"


def test_pin_applies_revision_file_selection_and_hashes(tmp_path, monkeypatch, fake_download):
    cfg = _register(tmp_path, repo="Qwen/Qwen3-0.6B", model_id="qwen3-0.6b")
    weights = [{"path": "model.safetensors", "sha256": mrv.sha256_file(fake_download.template / "model.safetensors")}]
    pin = {"revision": "f" * 40, "allow_patterns": sorted(item.name for item in fake_download.template.iterdir()),
           "required_files": ["config.json", "tokenizer_config.json", "model.safetensors"],
           "weight_files": weights}
    monkeypatch.setattr(jobs, "_load_remote_pin", lambda preset: pin)
    result = _ensure(tmp_path, cfg.model_id)
    assert result["hf"] == Path(cfg.model_path)
    assert fake_download.calls[0]["revision"] == "f" * 40
    assert fake_download.calls[0]["allow_patterns"] == pin["allow_patterns"]


def test_bad_pin_does_not_fall_back_to_unpinned_download(tmp_path, monkeypatch, fake_download):
    _register(tmp_path, repo="Qwen/Qwen3-0.6B", model_id="qwen3-0.6b")
    monkeypatch.setattr(jobs, "_load_remote_pin", lambda preset: None)
    _assert_code("SUPPLY_SOURCE_MISSING", lambda: _ensure(tmp_path, "qwen3-0.6b"))
    assert not fake_download.calls


def test_pin_hash_mismatch_fails_job_and_cleans_staging(tmp_path, monkeypatch, fake_download):
    cfg = _register(tmp_path, repo="Qwen/Qwen3-0.6B", model_id="qwen3-0.6b")
    pin = {"revision": "f" * 40, "allow_patterns": sorted(item.name for item in fake_download.template.iterdir()),
           "required_files": ["config.json"],
           "weight_files": [{"path": "model.safetensors", "sha256": "0" * 64}]}
    monkeypatch.setattr(jobs, "_load_remote_pin", lambda preset: pin)
    _assert_code("SUPPLY_DOWNLOAD_SHA256_MISMATCH", lambda: _ensure(tmp_path, cfg.model_id))
    assert not Path(cfg.model_path).exists()
    assert not list(Path(cfg.model_path).parent.glob(".qlh-supply-download-*"))
    assert jobs.list_jobs()[0]["status"] == "failed"


def test_download_transport_failure_is_named(tmp_path, monkeypatch, fake_download):
    cfg = _register(tmp_path, repo="owner/repo")

    def fail(*args, **kwargs):
        raise jobs.JobError("DOWNLOAD_FAILED", "sensitive remote details")

    monkeypatch.setattr(jobs, "_download", fail)
    error = _assert_code("SUPPLY_DOWNLOAD_FAILED", lambda: _ensure(tmp_path))
    assert "sensitive" not in error.message
    assert jobs.list_jobs()[0]["status"] == "failed"
    assert not Path(cfg.model_path).exists()


def test_invalid_download_metadata_is_not_registered(tmp_path, fake_download):
    cfg = _register(tmp_path, repo="owner/repo")
    (fake_download.template / "config.json").write_text("not json", encoding="utf-8")
    _assert_code("SUPPLY_DOWNLOAD_METADATA_INVALID", lambda: _ensure(tmp_path))
    assert jobs.list_jobs()[0]["status"] == "failed"
    assert not Path(cfg.model_path).exists()


def test_cancel_during_download_cleans_staging_and_marks_cancelled(tmp_path, fake_download):
    cfg = _register(tmp_path, repo="owner/repo")
    state = {"cancelled": False}

    def progress(event):
        if event.get("downloaded_bytes", 0):
            state["cancelled"] = True

    _assert_code("SUPPLY_CANCELLED", lambda: _ensure(tmp_path, cancelled=lambda: state["cancelled"], progress=progress))
    assert jobs.list_jobs()[0]["status"] == "cancelled"
    assert not Path(cfg.model_path).exists()


def test_initial_cancel_has_no_files_or_jobs(tmp_path):
    _register(tmp_path)
    _assert_code("SUPPLY_CANCELLED", lambda: _ensure(tmp_path, cancelled=lambda: True))
    assert not (tmp_path / "operation").exists()
    assert jobs.list_jobs() == []


def test_queued_existing_job_is_not_stolen(tmp_path, fake_download):
    cfg = _register(tmp_path, repo="owner/repo")
    target = Path(cfg.model_path)
    job = jobs.create_job(source=cfg.huggingface_id, target=str(target), model_id=cfg.model_id,
                          models_root=str(target.parent))
    _assert_code("SUPPLY_DOWNLOAD_BUSY", lambda: _ensure(tmp_path))
    assert not fake_download.calls
    assert jobs.get_job(job["job_id"])["status"] == "queued"


def test_register_failure_removes_only_this_download(tmp_path, monkeypatch, fake_download):
    cfg = _register(tmp_path, repo="owner/repo")
    def unavailable(*args, **kwargs):
        raise sources.ModelSupplySourceError("SUPPLY_REGISTER_FAILED", "Fixture persistence failure")

    monkeypatch.setattr(local_store, "save_local_experimental_model_cas", unavailable)
    _assert_code("SUPPLY_REGISTER_FAILED", lambda: _ensure(tmp_path))
    assert not Path(cfg.model_path).exists()
    assert jobs.list_jobs()[0]["status"] == "failed"
    assert fake_download.template.is_dir()


def test_valid_gguf_registry_path_reused_without_download(tmp_path, monkeypatch):
    target = _gguf(tmp_path / "models" / "exact.gguf")
    _register(tmp_path, gguf=target, model_type="gguf", model_path="")
    monkeypatch.setattr(jobs, "_download", _unexpected)
    result = _ensure(tmp_path, want_gguf=True)
    assert result["hf"] is None and result["gguf"] == target


def test_unregistered_adjacent_gguf_is_not_guessed(tmp_path, monkeypatch):
    hf = _hf(tmp_path / "models" / "hf")
    _gguf(hf.parent / "looks-like-fixture.gguf")
    _register(tmp_path, root=hf)
    monkeypatch.setattr(convert, "_find_converter", lambda explicit: (None, "missing"))
    monkeypatch.setattr(convert, "_find_quantizer", lambda explicit: (None, {"status": "missing"}))
    _assert_code("SUPPLY_CONVERSION_CONVERTER_MISSING", lambda: _ensure(tmp_path, want_gguf=True))


def test_conversion_produces_strict_proof_and_execution_evidence(tmp_path, fake_conversion):
    hf = _hf(tmp_path / "models" / "source")
    _register(tmp_path, root=hf)
    result = _ensure(tmp_path, want_gguf=True)
    proof = result["conversion_proof"]
    assert set(proof) == {"schema_version", "kind", "source", "conversion", "target", "provenance_sha256"}
    assert fake_conversion.calls[0]["confirmation"] == "CONVERT"
    assert fake_conversion.calls[0]["source"] == hf
    assert result["model_config"].gguf_path == str(result["gguf"])
    assert descriptor.cross_format_pipeline_identity(hf)["source_model_sha256"] == mrv.sha256_file(result["gguf"])
    assert proof["conversion"]["converter_sha256"] == mrv.sha256_file(fake_conversion.converter)
    assert proof["conversion"]["quantizer_revision"] == "c" * 40
    manifest = json.loads((hf / "model.manifest.json").read_text())
    assert {entry["path"] for entry in manifest["files"]} >= {"model.safetensors", "tokenizer.json", "config.json"}
    evidence_files = list((tmp_path / "operation").glob("source-conversion-*.json"))
    assert len(evidence_files) == 1
    evidence = json.loads(evidence_files[0].read_text())
    assert evidence["provenance_sha256"] == proof["provenance_sha256"]
    assert evidence["execution_report"]["execution"]["published"] is True
    assert evidence["converter_revision_kind"] == "content_sha256"
    _ensure(tmp_path, want_gguf=True)
    assert len(fake_conversion.calls) == 1


def test_real_conversion_transaction_with_small_copy_tools(tmp_path, fake_conversion, monkeypatch):
    hf = _hf(tmp_path / "models" / "source")
    _register(tmp_path, root=hf)
    fixture = _gguf(tmp_path / "tiny.gguf")
    fake_conversion.converter.write_text(
        "import shutil, sys\na=sys.argv[1:]\nshutil.copyfile(" + repr(str(fixture))
        + ", a[a.index('--outfile') + 1])\n", encoding="utf-8")
    fake_conversion.quantizer.write_text("import shutil, sys\nshutil.copyfile(sys.argv[1], sys.argv[2])\n", encoding="utf-8")
    fake_conversion.info["sha256"] = mrv.sha256_file(fake_conversion.quantizer)
    # Restore real plan/execute while retaining only fake, small tool discovery.
    monkeypatch.setattr(convert, "plan_conversion", _REAL_PLAN)
    monkeypatch.setattr(convert, "execute_conversion", _REAL_EXECUTE)
    monkeypatch.setattr(convert, "_find_converter", lambda explicit: (fake_conversion.converter, "available"))
    monkeypatch.setattr(convert, "_find_quantizer", lambda explicit: (fake_conversion.quantizer, dict(fake_conversion.info)))
    result = _ensure(tmp_path, want_gguf=True)
    assert result["gguf"].read_bytes() == fixture.read_bytes()
    assert result["gguf"].with_name(result["gguf"].name + ".sha256").is_file()
    assert descriptor.cross_format_pipeline_identity(hf)


@pytest.mark.parametrize("code", ["converter_missing", "quantizer_missing", "insufficient_space", "unsupported_architecture"])
def test_source_exists_but_conversion_preflight_failures_are_named(tmp_path, fake_conversion, monkeypatch, code):
    hf = _hf(tmp_path / "models" / "source")
    _register(tmp_path, root=hf)
    monkeypatch.setattr(convert, "plan_conversion", lambda **kw: {"valid": False, "errors": [{"code": code}]})
    monkeypatch.setattr(convert, "execute_conversion", _unexpected)
    _assert_code("SUPPLY_CONVERSION_" + code.upper(), lambda: _ensure(tmp_path, want_gguf=True))
    assert not (hf / sources.PROVENANCE).exists()


def test_executed_conversion_failure_keeps_report_without_proof(tmp_path, fake_conversion, monkeypatch):
    hf = _hf(tmp_path / "models" / "source")
    _register(tmp_path, root=hf)
    monkeypatch.setattr(convert, "execute_conversion", lambda **kw: {
        "valid": False, "execution": {"error": {"code": "stage_failed"}}})
    _assert_code("SUPPLY_CONVERSION_STAGE_FAILED", lambda: _ensure(tmp_path, want_gguf=True))
    assert len(list((tmp_path / "operation").glob("source-conversion-*.json"))) == 1
    assert not (hf / sources.PROVENANCE).exists()


def test_missing_source_tokenizer_has_named_proof_limit(tmp_path, fake_conversion):
    hf = _hf(tmp_path / "models" / "source", tokenizer=False)
    _register(tmp_path, root=hf)
    _assert_code("SUPPLY_CONVERSION_PROOF_UNAVAILABLE", lambda: _ensure(tmp_path, want_gguf=True))
    assert not fake_conversion.calls


def test_conversion_target_outside_models_root_has_named_limit(tmp_path, fake_conversion):
    hf = _hf(tmp_path / "models" / "source")
    _register(tmp_path, root=hf, gguf=tmp_path / "outside" / "model.gguf")
    _assert_code("SUPPLY_CONVERSION_PROOF_UNAVAILABLE", lambda: _ensure(tmp_path, want_gguf=True))
    assert not fake_conversion.calls


def test_legacy_core_manifest_and_extra_runtime_support_preserve_identity(tmp_path, monkeypatch):
    hf = _hf(tmp_path / "models" / "source", manifest=True)
    target = _gguf(tmp_path / "models" / "converted" / "non-guessable-name.gguf")
    _register(tmp_path, root=hf)
    proof = _old_proof(hf, target)
    original_manifest = (hf / "model.manifest.json").read_bytes()
    original_proof = (hf / sources.PROVENANCE).read_bytes()
    for name in ("chat_template.jinja", "merges.txt", "vocab.json"):
        (hf / name).write_text("runtime support fixture", encoding="utf-8")
    monkeypatch.setattr(jobs, "_download", _unexpected)
    monkeypatch.setattr(convert, "execute_conversion", _unexpected)
    assert descriptor.cross_format_pipeline_identity(hf)
    result = _ensure(tmp_path, want_gguf=True)
    assert result["gguf"] == target
    assert result["conversion_proof"] == proof
    assert (hf / "model.manifest.json").read_bytes() == original_manifest
    assert (hf / sources.PROVENANCE).read_bytes() == original_proof


@pytest.mark.parametrize("tamper", ["weights", "tokenizer", "target", "proof", "manifest"])
def test_invalid_old_proof_is_never_repaired_or_converted(tmp_path, monkeypatch, tamper):
    hf = _hf(tmp_path / "models" / "source", manifest=True)
    target = _gguf(tmp_path / "models" / "converted" / "model.gguf")
    _register(tmp_path, root=hf)
    _old_proof(hf, target)
    if tamper == "weights":
        path = hf / "model.safetensors"
        data = bytearray(path.read_bytes())
        data[-1] ^= 1
        path.write_bytes(data)
    elif tamper == "tokenizer":
        (hf / "tokenizer.json").write_text("changed", encoding="utf-8")
    elif tamper == "target":
        target.write_bytes(target.read_bytes() + b"changed")
    elif tamper == "proof":
        (hf / sources.PROVENANCE).write_text("{}", encoding="utf-8")
    else:
        (hf / "model.manifest.json").write_text("{}", encoding="utf-8")
    original = (hf / sources.PROVENANCE).read_bytes()
    monkeypatch.setattr(jobs, "_download", _unexpected)
    monkeypatch.setattr(convert, "execute_conversion", _unexpected)
    _assert_code("SUPPLY_CONVERSION_PROOF_INVALID", lambda: _ensure(tmp_path, want_gguf=True))
    assert (hf / sources.PROVENANCE).read_bytes() == original


def test_new_proof_requires_actual_source_and_execution_target_hashes(tmp_path, fake_conversion, monkeypatch):
    hf = _hf(tmp_path / "models" / "source")
    _register(tmp_path, root=hf)

    def execute(**options):
        report = fake_conversion.execute(**options)
        report["execution"]["artifact"]["sha256"] = "0" * 64
        return report

    monkeypatch.setattr(convert, "execute_conversion", execute)
    _assert_code("SUPPLY_CONVERSION_PROOF_UNAVAILABLE", lambda: _ensure(tmp_path, want_gguf=True))
    assert not (hf / sources.PROVENANCE).exists()


def test_download_then_convert_keeps_source_and_conversion_registered(tmp_path, fake_download, fake_conversion):
    cfg = _register(tmp_path, repo="owner/repo")
    result = _ensure(tmp_path, want_gguf=True)
    assert result["hf"] == Path(cfg.model_path)
    assert result["conversion_proof"] is not None
    row = local_store.get_local_experimental_models()[0]
    assert row["supply_download_job_id"]
    assert row["gguf_path"] == str(result["gguf"])
    assert jobs.list_jobs()[0]["status"] == "ready"


@pytest.mark.parametrize("problem", ["size", "revision", "files", "unsafe_path"])
def test_invalid_provider_metadata_fails_before_weight_download(tmp_path, fake_download, monkeypatch, problem):
    _register(tmp_path, repo="owner/repo")

    def info(self, *args, **options):
        result = fake_download.model_info(*args, **options)
        if problem == "size":
            result.siblings[0].size = None
        elif problem == "revision":
            result.sha = "main"
        elif problem == "files":
            result.siblings = []
        else:
            result.siblings[0].rfilename = "../outside"
        return result

    monkeypatch.setattr("huggingface_hub.HfApi.model_info", info)
    expected = {"size": "SUPPLY_DOWNLOAD_METADATA_INVALID", "revision": "SUPPLY_DOWNLOAD_METADATA_INVALID",
                "files": "SUPPLY_SOURCE_MISSING", "unsafe_path": "SUPPLY_DOWNLOAD_PRESET_PIN_INVALID"}[problem]
    _assert_code(expected, lambda: _ensure(tmp_path))
    assert not fake_download.calls
    assert jobs.list_jobs()[0]["status"] == "failed"


def test_reuses_real_download_implementation_with_fake_hub(tmp_path, fake_download, monkeypatch):
    cfg = _register(tmp_path, repo="owner/repo")
    calls = []

    def snapshot_download(**options):
        calls.append(options)
        staging = Path(options["local_dir"])
        for item in fake_download.template.iterdir():
            shutil.copyfile(item, staging / item.name)
        return str(staging)

    monkeypatch.setattr(jobs, "_download", _REAL_DOWNLOAD)
    monkeypatch.setattr("huggingface_hub.snapshot_download", snapshot_download)
    result = _ensure(tmp_path)
    assert result["hf"] == Path(cfg.model_path)
    assert calls[0]["repo_id"] == cfg.huggingface_id
    assert calls[0]["revision"] == "a" * 40
    assert calls[0]["allow_patterns"]
    assert jobs.list_jobs()[0]["status"] == "ready"


def test_remote_gguf_uses_explicit_registered_selector(tmp_path, fake_download, monkeypatch):
    _register(tmp_path, gguf=tmp_path / "missing" / "chosen.gguf", repo="owner/repo",
              model_type="gguf", model_path="")
    monkeypatch.setattr(mc, "_APP_ROOT", str(tmp_path))
    template = _gguf(tmp_path / "provider-gguf" / "chosen.gguf")
    monkeypatch.setattr("huggingface_hub.HfApi.model_info", lambda self, *a, **kw: SimpleNamespace(
        sha="a" * 40, siblings=[SimpleNamespace(rfilename=template.name, size=template.stat().st_size)]))
    calls = []

    def download(repo, staging, **options):
        calls.append(options)
        destination = staging / template.name
        shutil.copyfile(template, destination)
        return [destination]

    monkeypatch.setattr(jobs, "_download", download)
    result = _ensure(tmp_path, want_gguf=True)
    assert result["hf"] is None
    assert result["gguf"].name == "chosen.gguf"
    assert calls[0]["allow_patterns"] == ["chosen.gguf"]
    assert jobs.list_jobs()[0]["status"] == "ready"
    assert local_store.get_local_experimental_models()[0]["gguf_path"] == str(result["gguf"])


def test_support_not_in_old_manifest_does_not_change_new_conversion_source_identity(tmp_path, fake_conversion):
    hf = _hf(tmp_path / "models" / "source", manifest=True)
    _register(tmp_path, root=hf)
    before = (hf / "model.manifest.json").read_bytes()
    for name in ("chat_template.jinja", "merges.txt", "vocab.json"):
        (hf / name).write_text("extra runtime support", encoding="utf-8")
    result = _ensure(tmp_path, want_gguf=True)
    assert (hf / "model.manifest.json").read_bytes() == before
    original = json.loads(before)
    assert result["conversion_proof"]["source"]["artifact_sha256"] == original["artifact_sha256"]


def test_partial_or_forged_execution_report_cannot_create_proof(tmp_path, fake_conversion, monkeypatch):
    hf = _hf(tmp_path / "models" / "source")
    _register(tmp_path, root=hf)

    def execute(**options):
        report = fake_conversion.execute(**options)
        report["execution"]["stages"] = [{"stage": "atomic_publish", "status": "passed"}]
        return report

    monkeypatch.setattr(convert, "execute_conversion", execute)
    _assert_code("SUPPLY_CONVERSION_EVIDENCE_INVALID", lambda: _ensure(tmp_path, want_gguf=True))
    assert not (hf / sources.PROVENANCE).exists()


def test_source_changes_during_conversion_prevent_proof(tmp_path, fake_conversion, monkeypatch):
    hf = _hf(tmp_path / "models" / "source")
    _register(tmp_path, root=hf)

    def execute(**options):
        report = fake_conversion.execute(**options)
        path = hf / "model.safetensors"
        data = bytearray(path.read_bytes())
        data[-1] ^= 1
        path.write_bytes(data)
        return report

    monkeypatch.setattr(convert, "execute_conversion", execute)
    _assert_code("SUPPLY_CONVERSION_PROOF_UNAVAILABLE", lambda: _ensure(tmp_path, want_gguf=True))
    assert not (hf / sources.PROVENANCE).exists()


def test_cancel_after_conversion_preserves_report_and_does_not_publish_proof(tmp_path, fake_conversion, monkeypatch):
    hf = _hf(tmp_path / "models" / "source")
    _register(tmp_path, root=hf)
    state = {"cancelled": False}

    def execute(**options):
        report = fake_conversion.execute(**options)
        state["cancelled"] = True
        return report

    monkeypatch.setattr(convert, "execute_conversion", execute)
    _assert_code("SUPPLY_CANCELLED", lambda: _ensure(tmp_path, want_gguf=True, cancelled=lambda: state["cancelled"]))
    assert list((tmp_path / "operation").glob("source-conversion-*.json"))
    assert not (hf / sources.PROVENANCE).exists()


def test_unexpected_conversion_exception_still_has_supply_code(tmp_path, fake_conversion, monkeypatch):
    hf = _hf(tmp_path / "models" / "source")
    _register(tmp_path, root=hf)

    def execute(**options):
        raise RuntimeError("private tool failure")

    monkeypatch.setattr(convert, "execute_conversion", execute)
    error = _assert_code("SUPPLY_CONVERSION_FAILED", lambda: _ensure(tmp_path, want_gguf=True))
    assert "private" not in error.message


def test_unverified_quantizer_uses_actual_content_revision(tmp_path, fake_conversion):
    hf = _hf(tmp_path / "models" / "source")
    _register(tmp_path, root=hf)
    fake_conversion.info["verification"] = "unmanaged"
    fake_conversion.info["revision"] = None
    result = _ensure(tmp_path, want_gguf=True)
    conversion = result["conversion_proof"]["conversion"]
    assert conversion["quantizer_revision"] == conversion["quantizer_sha256"] == mrv.sha256_file(fake_conversion.quantizer)
    assert descriptor.cross_format_pipeline_identity(hf)



def test_existing_four_shard_indexed_identity_ignores_additional_support(tmp_path, monkeypatch):
    hf = _hf(tmp_path / "models" / "source")
    (hf / "model.safetensors").unlink()
    (hf / "config.json").write_text(json.dumps({
        "architectures": ["Qwen3_5ForConditionalGeneration"], "model_type": "qwen3_5",
        "text_config": {"num_hidden_layers": 4, "hidden_size": 2}}), encoding="utf-8")
    shards, mapping = [], {}
    for layer in range(4):
        name = f"model-{layer + 1:05d}-of-00004.safetensors"
        key = f"model.language_model.layers.{layer}.input_layernorm.weight"
        header = json.dumps({key: {"dtype": "F16", "shape": [2], "data_offsets": [0, 4]}}).encode()
        shard = hf / name
        shard.write_bytes(struct.pack("<Q", len(header)) + header + b"\0" * 4)
        shards.append(shard)
        mapping[key] = name
    (hf / "model.safetensors.index.json").write_text(json.dumps({"weight_map": mapping}), encoding="utf-8")
    mrv.write_manifest(hf, mrv.build_manifest(hf, shards, model_type="safetensors"))
    target = _gguf(tmp_path / "models" / "gguf" / "existing.gguf")
    proof = _old_proof(hf, target)
    _register(tmp_path, root=hf)
    manifest_bytes = (hf / "model.manifest.json").read_bytes()
    proof_bytes = (hf / sources.PROVENANCE).read_bytes()
    for name in ("chat_template.jinja", "merges.txt", "vocab.json"):
        (hf / name).write_text("existing runtime support", encoding="utf-8")
    monkeypatch.setattr(jobs, "_download", _unexpected)
    monkeypatch.setattr(convert, "execute_conversion", _unexpected)
    result = _ensure(tmp_path, want_gguf=True)
    assert result["conversion_proof"] == proof
    assert result["gguf"] == target
    assert (hf / "model.manifest.json").read_bytes() == manifest_bytes
    assert (hf / sources.PROVENANCE).read_bytes() == proof_bytes


def test_declared_index_changes_fail_existing_proof(tmp_path, monkeypatch):
    hf = _hf(tmp_path / "models" / "source")
    index = hf / "model.safetensors.index.json"
    mapping = {f"model.layers.{i}.input_layernorm.weight": "model.safetensors" for i in range(2)}
    index.write_text(json.dumps({"weight_map": mapping}), encoding="utf-8")
    mrv.write_manifest(hf, mrv.build_manifest(hf, [hf / "model.safetensors"], model_type="safetensors"))
    target = _gguf(tmp_path / "models" / "gguf" / "existing.gguf")
    _old_proof(hf, target)
    _register(tmp_path, root=hf)
    index.write_text(json.dumps({"weight_map": mapping, "unexpected": True}), encoding="utf-8")
    before = (hf / sources.PROVENANCE).read_bytes()
    monkeypatch.setattr(convert, "execute_conversion", _unexpected)
    _assert_code("SUPPLY_CONVERSION_PROOF_INVALID", lambda: _ensure(tmp_path, want_gguf=True))
    assert (hf / sources.PROVENANCE).read_bytes() == before


def _storage_link(target, link):
    import os
    if os.name == "nt":
        import _winapi
        _winapi.CreateJunction(str(target), str(link))
    else:
        link.symlink_to(target, target_is_directory=True)


def test_application_models_storage_mapping_is_accepted(tmp_path, monkeypatch):
    app = tmp_path / "app"
    app.mkdir()
    disk = tmp_path / "storage"
    source = _hf(disk / "fixture", manifest=True)
    alias = app / "models"
    _storage_link(disk, alias)
    monkeypatch.setattr(mc, "_APP_ROOT", str(app))
    try:
        _register(tmp_path, root=alias / "fixture")
        result = _ensure(tmp_path)
        assert Path(result["hf"]) == source.resolve()
        assert sources._path("models/fixture") == source.resolve()
    finally:
        alias.rmdir() if __import__("os").name == "nt" else alias.unlink()


def test_application_mapping_does_not_allow_link_inside_model(tmp_path, monkeypatch):
    app = tmp_path / "app"
    app.mkdir()
    disk = tmp_path / "storage"
    disk.mkdir()
    source = _hf(tmp_path / "outside", manifest=True)
    alias = app / "models"
    nested = disk / "fixture"
    _storage_link(disk, alias)
    _storage_link(source, nested)
    monkeypatch.setattr(mc, "_APP_ROOT", str(app))
    try:
        _register(tmp_path, root=alias / "fixture")
        _assert_code("SUPPLY_SOURCE_INVALID", lambda: _ensure(tmp_path))
    finally:
        for link in (nested, alias):
            link.rmdir() if __import__("os").name == "nt" else link.unlink()


def _both_registered(tmp_path, *, legacy=False):
    hf = _hf(tmp_path / "models" / "hf")
    gguf = _gguf(hf.parent / "registered.gguf")
    manifest = mrv.build_manifest(hf, [hf / "model.safetensors", gguf], model_type="both")
    mrv.write_manifest(hf, manifest)
    fields = {} if legacy else {"manifest_external_files": mrv.manifest_external_files(hf, [gguf])}
    _register(tmp_path, root=hf, gguf=gguf, model_type="both", **fields)
    return hf, gguf


@pytest.mark.parametrize("legacy", [True, False])
def test_both_source_external_manifest_uses_only_explicit_registered_file(tmp_path, monkeypatch, legacy):
    hf, gguf = _both_registered(tmp_path, legacy=legacy)
    before = (hf / "model.manifest.json").read_bytes()
    monkeypatch.setattr(jobs, "_download", _unexpected)
    monkeypatch.setattr(convert, "execute_conversion", _unexpected)
    result = _ensure(tmp_path, want_gguf=True)
    assert result["hf"] == hf and result["gguf"] == gguf
    assert (hf / "model.manifest.json").read_bytes() == before


@pytest.mark.parametrize("tamper", ["content", "missing", "unregistered"])
def test_external_manifest_binding_does_not_bypass_live_validation(tmp_path, tamper):
    hf, gguf = _both_registered(tmp_path, legacy=True)
    if tamper == "content":
        data = gguf.read_bytes()
        gguf.write_bytes(data[:-1] + bytes([data[-1] ^ 1]))
    elif tamper == "missing":
        gguf.unlink()
    else:
        cfg = mc.get_model_config("fixture", local_store.get_local_experimental_models())
        snapshot = local_store.get_local_model_registry_snapshot("fixture")
        assert local_store.save_local_experimental_model("fixture", {**asdict(cfg), "gguf_path": ""}, expected_revision=snapshot.registry_revision)
    _assert_code("SUPPLY_SOURCE_METADATA_INVALID", lambda: _ensure(tmp_path, want_gguf=True))


def test_mixed_both_without_proof_converts_once_preserving_original_source(tmp_path, fake_conversion):
    hf, original_gguf = _both_registered(tmp_path, legacy=True)
    original = original_gguf.read_bytes()
    manifest_bytes = (hf / "model.manifest.json").read_bytes()
    options = dict(operation_dir=tmp_path / "first", want_gguf=True,
                   cancelled=lambda: False, require_cross_format=True)
    first = sources.ensure_registered_sources("fixture", **options)
    assert first["gguf"] != original_gguf
    assert first["conversion_proof"]
    assert len(fake_conversion.calls) == 1
    assert original_gguf.read_bytes() == original
    assert (hf / "model.manifest.json").read_bytes() == manifest_bytes
    row = local_store.get_local_experimental_models()[0]
    assert row["manifest_external_files"]["external/registered.gguf"] == str(original_gguf)
    options["operation_dir"] = tmp_path / "second"
    second = sources.ensure_registered_sources("fixture", **options)
    assert second["gguf"] == first["gguf"]
    assert second["conversion_proof"] == first["conversion_proof"]
    assert len(fake_conversion.calls) == 1


def test_external_manifest_alias_collision_fails_registration(tmp_path):
    a = _gguf(tmp_path / "a" / "same.gguf")
    b = _gguf(tmp_path / "b" / "same.gguf")
    with pytest.raises(ValueError, match="ambiguous external"):
        mrv.build_manifest(tmp_path / "hf", [a, b], model_type="both")


def _register_manifest_identity(tmp_path, hf, manifest, *, storage="both", gguf=None):
    fields = {}
    if storage in {"flat", "both"}:
        fields.update(artifact_sha256=manifest["artifact_sha256"],
                      manifest_sha256=manifest["manifest_sha256"])
    if storage in {"nested", "both"}:
        fields["manifest"] = manifest
    if gguf is not None:
        fields["model_type"] = "both"
    return _register(tmp_path, root=hf, gguf=gguf, **fields)


@pytest.mark.parametrize("storage", ["flat", "nested", "both"])
@pytest.mark.parametrize("tamper", ["delete", "rewrite_weights", "rewrite_metadata"])
@pytest.mark.parametrize("want_gguf", [False, True])
def test_registered_manifest_identity_cannot_be_deleted_or_resigned(
    tmp_path, monkeypatch, storage, tamper, want_gguf,
):
    hf = _hf(tmp_path / "models" / "source", manifest=True)
    manifest = json.loads((hf / "model.manifest.json").read_text(encoding="utf-8"))
    _register_manifest_identity(tmp_path, hf, manifest, storage=storage)
    registered = local_store.get_local_experimental_models()
    if tamper == "delete":
        (hf / "model.manifest.json").unlink()
    else:
        if tamper == "rewrite_weights":
            weight = hf / "model.safetensors"
            data = bytearray(weight.read_bytes())
            data[-1] ^= 1  # Valid Safetensors layout; only the tensor payload changes.
            weight.write_bytes(data)
        replacement = mrv.build_manifest(hf, [hf / "model.safetensors"],
                                         model_type="safetensors",
                                         source="changed" if tamper == "rewrite_metadata" else "")
        mrv.write_manifest(hf, replacement)
        # The replacement is self-consistent and hashes its actual files.
        sources._live_manifest(hf, [hf / "model.safetensors"], lambda: False)
    monkeypatch.setattr(jobs, "_download", _unexpected)
    monkeypatch.setattr(convert, "execute_conversion", _unexpected)
    _assert_code("SUPPLY_SOURCE_METADATA_INVALID",
                 lambda: _ensure(tmp_path, want_gguf=want_gguf))
    assert local_store.get_local_experimental_models() == registered
    assert not (hf / sources.PROVENANCE).exists()
    if tamper == "delete":
        assert not (hf / "model.manifest.json").exists()


@pytest.mark.parametrize("storage", ["flat", "nested", "both"])
def test_pinned_mixed_source_stays_usable_across_two_operations(tmp_path, fake_conversion, storage):
    hf, original = _both_registered(tmp_path, legacy=True)
    manifest_bytes = (hf / "model.manifest.json").read_bytes()
    manifest = json.loads(manifest_bytes)
    _register_manifest_identity(tmp_path, hf, manifest, storage=storage, gguf=original)
    original_bytes = original.read_bytes()
    first = sources.ensure_registered_sources(
        "fixture", operation_dir=tmp_path / "first-pinned", want_gguf=True,
        require_cross_format=True, cancelled=lambda: False,
    )
    row = local_store.get_local_experimental_models()[0]
    assert row["manifest_external_files"]["external/registered.gguf"] == str(original)
    assert row["manifest_sha256"] == manifest["manifest_sha256"]
    assert row["artifact_sha256"] == manifest["artifact_sha256"]
    assert first["gguf"] != original
    second = sources.ensure_registered_sources(
        "fixture", operation_dir=tmp_path / "second-pinned", want_gguf=True,
        require_cross_format=True, cancelled=lambda: False,
    )
    assert second["conversion_proof"] == first["conversion_proof"]
    assert second["gguf"] == first["gguf"]
    assert len(fake_conversion.calls) == 1
    assert local_store.get_local_experimental_models()[0] == row
    assert original.read_bytes() == original_bytes
    assert (hf / "model.manifest.json").read_bytes() == manifest_bytes


@pytest.mark.parametrize("builtin_without_row", [False, True])
def test_conversion_reregistration_cannot_be_overwritten_by_old_snapshot(
    tmp_path, monkeypatch, fake_conversion, builtin_without_row,
):
    hf = _hf(tmp_path / "models" / "old", manifest=True)
    manifest_bytes = (hf / "model.manifest.json").read_bytes()
    if builtin_without_row:
        builtin = ModelConfig(model_id="fixture", name="Builtin", model_path=str(hf))
        monkeypatch.setattr(mc, "get_builtin_model", lambda mid: builtin if mid == "fixture" else None)
        assert local_store.get_local_experimental_models() == []
    else:
        # A legacy mixed source adds external bindings to its local working row.
        original = _gguf(hf.parent / "old-original.gguf")
        manifest = mrv.build_manifest(hf, [hf / "model.safetensors", original], model_type="both")
        mrv.write_manifest(hf, manifest)
        manifest_bytes = (hf / "model.manifest.json").read_bytes()
        _register_manifest_identity(tmp_path, hf, manifest, gguf=original)
    replacement = _hf(tmp_path / "models" / "new", manifest=True)
    replacement_manifest = json.loads((replacement / "model.manifest.json").read_text(encoding="utf-8"))
    captured = {}

    def execute(**options):
        report = fake_conversion.execute(**options)
        _register_manifest_identity(tmp_path, replacement, replacement_manifest)
        captured["row"] = local_store.get_local_experimental_models()[0]
        return report

    monkeypatch.setattr(convert, "execute_conversion", execute)
    _assert_code("SUPPLY_REGISTRY_CHANGED", lambda: sources.ensure_registered_sources(
        "fixture", operation_dir=tmp_path / "stale-conversion", want_gguf=True,
        require_cross_format=True, cancelled=lambda: False,
    ))
    assert len(fake_conversion.calls) == 1
    assert local_store.get_local_experimental_models() == [captured["row"]]
    assert captured["row"]["model_path"] == str(replacement)
    assert (hf / "model.manifest.json").read_bytes() == manifest_bytes
    assert _ensure(tmp_path)["hf"] == replacement


def test_builtin_without_registry_row_converts_then_reuses_inserted_identity(
    tmp_path, monkeypatch, fake_conversion,
):
    hf = _hf(tmp_path / "models" / "builtin")
    builtin = ModelConfig(model_id="fixture", name="Builtin", model_path=str(hf))
    monkeypatch.setattr(mc, "get_builtin_model", lambda mid: builtin if mid == "fixture" else None)
    assert local_store.get_local_experimental_models() == []
    first = _ensure(tmp_path, want_gguf=True)
    row = local_store.get_local_experimental_models()[0]
    assert row["manifest_sha256"] == first["conversion_proof"]["source"]["manifest_sha256"]
    assert row["artifact_sha256"] == first["conversion_proof"]["source"]["artifact_sha256"]
    assert _ensure(tmp_path, want_gguf=True)["gguf"] == first["gguf"]
    assert len(fake_conversion.calls) == 1


@pytest.mark.parametrize("tamper", ["delete", "resign"])
def test_existing_proof_remains_bound_to_registered_manifest(tmp_path, fake_conversion, tamper):
    hf = _hf(tmp_path / "models" / "source")
    _register(tmp_path, root=hf)
    first = _ensure(tmp_path, want_gguf=True)
    row = local_store.get_local_experimental_models()[0]
    if tamper == "delete":
        (hf / "model.manifest.json").unlink()
    else:
        manifest = json.loads((hf / "model.manifest.json").read_text(encoding="utf-8"))
        manifest["source"] = "resigned"
        manifest["manifest_sha256"] = sources._canonical({
            key: value for key, value in manifest.items()
            if key not in {"manifest_sha256", "artifact_sha256"}
        })
        mrv.write_manifest(hf, manifest)
        proof = first["conversion_proof"]
        proof["source"]["manifest_sha256"] = manifest["manifest_sha256"]
        proof["provenance_sha256"] = sources._canonical({
            key: value for key, value in proof.items() if key != "provenance_sha256"
        })
        (hf / sources.PROVENANCE).write_text(json.dumps(proof), encoding="utf-8")
        assert descriptor.cross_format_pipeline_identity(hf)  # Self-consistent, but not the registered identity.
    _assert_code("SUPPLY_CONVERSION_PROOF_INVALID", lambda: _ensure(tmp_path, want_gguf=True))
    assert local_store.get_local_experimental_models() == [row]
    assert len(fake_conversion.calls) == 1


@pytest.mark.parametrize("tamper", ["delete", "resign"])
def test_manifest_changed_between_validation_and_conversion_is_not_recreated(
    tmp_path, monkeypatch, fake_conversion, tamper,
):
    hf = _hf(tmp_path / "models" / "source", manifest=True)
    manifest = json.loads((hf / "model.manifest.json").read_text(encoding="utf-8"))
    _register_manifest_identity(tmp_path, hf, manifest)
    original_convert = sources._convert

    def changed_before_convert(*args, **kwargs):
        if tamper == "delete":
            (hf / "model.manifest.json").unlink()
        else:
            mrv.write_manifest(hf, mrv.build_manifest(hf, [hf / "model.safetensors"],
                                                     model_type="safetensors", source="changed"))
        return original_convert(*args, **kwargs)

    monkeypatch.setattr(sources, "_convert", changed_before_convert)
    _assert_code("SUPPLY_CONVERSION_PROOF_UNAVAILABLE", lambda: _ensure(tmp_path, want_gguf=True))
    assert not fake_conversion.calls
    assert not (hf / sources.PROVENANCE).exists()
    if tamper == "delete":
        assert not (hf / "model.manifest.json").exists()


def test_legacy_manifest_rewritten_during_conversion_cannot_publish_proof(
    tmp_path, monkeypatch, fake_conversion,
):
    hf = _hf(tmp_path / "models" / "source")
    _register(tmp_path, root=hf)

    def execute(**options):
        report = fake_conversion.execute(**options)
        mrv.write_manifest(hf, mrv.build_manifest(hf, [hf / "model.safetensors"],
                                                 model_type="safetensors", source="changed"))
        return report

    monkeypatch.setattr(convert, "execute_conversion", execute)
    _assert_code("SUPPLY_CONVERSION_PROOF_UNAVAILABLE", lambda: _ensure(tmp_path, want_gguf=True))
    assert not (hf / sources.PROVENANCE).exists()


def test_download_cas_conflict_preserves_source_adopted_by_new_registration(
    tmp_path, fake_download,
):
    cfg = _register(tmp_path, repo="owner/repo")
    captured = {}

    def progress(value):
        if value["phase"] == "registering":
            snapshot = local_store.get_local_model_registry_snapshot("fixture")
            assert local_store.save_local_experimental_model("fixture", {
                **asdict(cfg), "name": "Concurrent registration adopted this source",
            }, expected_revision=snapshot.registry_revision)
            captured["row"] = local_store.get_local_experimental_models()[0]

    _assert_code("SUPPLY_REGISTRY_CHANGED", lambda: _ensure(tmp_path, progress=progress))
    assert local_store.get_local_experimental_models() == [captured["row"]]
    assert Path(cfg.model_path).is_dir()
    assert jobs.list_jobs()[0]["status"] == "failed"


@pytest.mark.parametrize("changed", [False, True])
def test_download_rebuild_is_bound_to_existing_registered_identity(tmp_path, fake_download, changed):
    manifest = mrv.build_manifest(fake_download.template,
                                  [fake_download.template / "model.safetensors"],
                                  model_type="safetensors", revision="a" * 40, source="owner/repo")
    cfg = _register(tmp_path, repo="owner/repo", manifest=manifest,
                    artifact_sha256=manifest["artifact_sha256"],
                    manifest_sha256=manifest["manifest_sha256"])
    before = local_store.get_local_experimental_models()
    if changed:
        weight = fake_download.template / "model.safetensors"
        data = bytearray(weight.read_bytes())
        data[-1] ^= 1
        weight.write_bytes(data)
        _assert_code("SUPPLY_DOWNLOAD_METADATA_INVALID", lambda: _ensure(tmp_path))
        assert local_store.get_local_experimental_models() == before
        assert not Path(cfg.model_path).exists()
    else:
        first = _ensure(tmp_path)
        second = _ensure(tmp_path)
        assert first["hf"] == second["hf"] == Path(cfg.model_path)
        assert local_store.get_local_experimental_models()[0]["manifest_sha256"] == manifest["manifest_sha256"]
        assert len(fake_download.calls) == 1


def test_pinned_hf_tree_cannot_disappear_and_degrade_to_unproven_gguf(tmp_path, monkeypatch):
    hf, gguf = _both_registered(tmp_path, legacy=True)
    manifest = json.loads((hf / "model.manifest.json").read_text(encoding="utf-8"))
    _register_manifest_identity(tmp_path, hf, manifest, gguf=gguf)
    shutil.rmtree(hf)
    monkeypatch.setattr(jobs, "_download", _unexpected)
    monkeypatch.setattr(convert, "execute_conversion", _unexpected)
    _assert_code("SUPPLY_SOURCE_METADATA_INVALID", lambda: _ensure(tmp_path, want_gguf=True))
    assert gguf.is_file()


def test_conflicting_flat_and_nested_source_identity_fails_closed(tmp_path):
    hf = _hf(tmp_path / "models" / "source", manifest=True)
    manifest = json.loads((hf / "model.manifest.json").read_text(encoding="utf-8"))
    _register(tmp_path, root=hf, manifest=manifest, manifest_sha256="a" * 64)
    _assert_code("SUPPLY_REGISTRY_INVALID", lambda: _ensure(tmp_path))


@pytest.mark.parametrize("tamper", [None, "delete", "resign"])
def test_gguf_only_registration_is_bound_to_live_manifest(tmp_path, monkeypatch, tamper):
    gguf = _gguf(tmp_path / "models" / "gguf" / "registered.gguf")
    manifest = mrv.build_manifest(gguf.parent, [gguf], model_type="gguf")
    mrv.write_manifest(gguf.parent, manifest)
    _register(tmp_path, gguf=gguf, model_path="", model_type="gguf", manifest=manifest,
              artifact_sha256=manifest["artifact_sha256"], manifest_sha256=manifest["manifest_sha256"])
    row = local_store.get_local_experimental_models()[0]
    manifest_bytes = (gguf.parent / "model.manifest.json").read_bytes()
    monkeypatch.setattr(jobs, "_download", _unexpected)
    monkeypatch.setattr(convert, "execute_conversion", _unexpected)
    if tamper == "delete":
        (gguf.parent / "model.manifest.json").unlink()
    elif tamper == "resign":
        data = bytearray(gguf.read_bytes())
        data[-1] ^= 1
        gguf.write_bytes(data)
        sources._gguf_metadata(gguf)  # Structure-valid GGUF with a new payload.
        mrv.write_manifest(gguf.parent, mrv.build_manifest(gguf.parent, [gguf], model_type="gguf"))
    if tamper:
        _assert_code("SUPPLY_SOURCE_METADATA_INVALID", lambda: _ensure(tmp_path, want_gguf=True))
    else:
        assert _ensure(tmp_path, want_gguf=True)["gguf"] == gguf
        assert _ensure(tmp_path, want_gguf=True)["gguf"] == gguf
        assert (gguf.parent / "model.manifest.json").read_bytes() == manifest_bytes
    assert local_store.get_local_experimental_models() == [row]
