"""Distributed catalog/selection metadata preflight; no native loads or network."""
from __future__ import annotations

import asyncio
import hashlib
import json
import struct
import subprocess
import sys
import threading
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

import model_config as mc
import model_pipeline_availability as availability
from pipeline_model_descriptor import PIPELINE_RUNTIME_MODEL_TYPES


def _hf(root, model_type="qwen2", *, weights=True):
    root.mkdir(parents=True, exist_ok=True)
    (root / "config.json").write_text(json.dumps({
        "model_type": model_type, "num_hidden_layers": 4,
        "text_config": {"model_type": "qwen2", "num_hidden_layers": 4},
    }), encoding="utf-8")
    if weights:
        (root / "model.safetensors").write_bytes(b"weights must never be opened")
    return root


def _string(value):
    value = value.encode("utf-8")
    return struct.pack("<Q", len(value)) + value


def _gguf(path, architecture="qwen2", *, total=4, layers=None, mtp=0, tokenizer=False):
    metadata = [
        _string("general.architecture") + struct.pack("<I", 8) + _string(architecture),
        _string(architecture + ".block_count") + struct.pack("<II", 4, total + mtp),
        _string(architecture + ".nextn_predict_layers") + struct.pack("<II", 4, mtp),
    ]
    if tokenizer:
        metadata.extend([
            _string("tokenizer.ggml.model") + struct.pack("<I", 8) + _string("gpt2"),
            _string("tokenizer.ggml.tokens") + struct.pack("<IIQ", 9, 8, 2) + _string("a") + _string("b"),
        ])
    layers = list(range(total + mtp)) if layers is None else layers
    tensors = [_string(f"blk.{layer}.weight") + struct.pack("<IQIQ", 1, 4, 0, index * 32)
               for index, layer in enumerate(layers)]
    header = b"GGUF" + struct.pack("<IQQ", 3, len(tensors), len(metadata)) + b"".join(metadata + tensors)
    path.write_bytes(header + b"\0" * (-len(header) % 32) + b"\0" * max(32, 32 * len(tensors)))
    return path, len(header)


def _config(tmp_path, *, hf_type="qwen2", gguf_type="qwen2", model_id="fixture"):
    hf = _hf(tmp_path / "hf", hf_type) if hf_type is not None else None
    gguf = _gguf(tmp_path / "source.gguf", gguf_type)[0] if gguf_type is not None else None
    return mc.ModelConfig(model_id=model_id, name="metadata fixture", model_type="both",
                          model_path=str(hf) if hf else "", gguf_path=str(gguf) if gguf else "")


@pytest.fixture
def intent_api(monkeypatch, tmp_path):
    import api_server
    from api import routes_model_intent, routes_models

    rows, builtins = [], []
    monkeypatch.setattr(mc, "BUILTIN_MODELS", builtins)
    from tests.test_model_intent import IntentOwner
    scheduler = SimpleNamespace(
        _model_control=IntentOwner(),
        _effective_role=lambda: "master", _layer_config_lock=threading.RLock(),
        _pipeline_load_transaction={"phase": "ready", "config_id": "old", "generation": 7},
        _active_pipeline_capacity_plan={"transaction_phase": "ready", "config_id": "old", "generation": 7},
        _pipeline_model_cleanup_ledger={},
        _model_supply=SimpleNamespace(_check_master=lambda: None, create=Mock()),
    )
    coordinator = SimpleNamespace(current=lambda: None, create=AsyncMock(return_value={"status": "queued"}))
    api = SimpleNamespace(
        scheduler=scheduler, API_PORT=18000,
        _get_registered_experimental_models=lambda: rows,
        _get_all_model_configs=lambda: mc.get_all_model_configs(rows),
        _model_api_payload=api_server._model_api_payload,
        model_host=SimpleNamespace(model_loaded=False), model_manager=SimpleNamespace(active_model_id=None),
    )
    monkeypatch.setattr(routes_model_intent, "_api_module", api)
    monkeypatch.setattr(routes_models, "_api_module", api)
    monkeypatch.setattr(routes_model_intent, "coordinator", lambda: coordinator)
    monkeypatch.setattr(routes_model_intent, "get_binding", lambda *_: {"status": "connected", "is_master": True})
    app = FastAPI()
    app.add_api_route("/api/models/intent/catalog", routes_model_intent.catalog, methods=["GET"])
    app.add_api_route("/api/models/intent", routes_model_intent.create, methods=["POST"])
    app.add_api_route("/api/models", routes_models.list_models, methods=["GET"])
    return SimpleNamespace(client=TestClient(app), rows=rows, builtins=builtins, scheduler=scheduler,
                           coordinator=coordinator, routes=routes_model_intent, root=tmp_path)


@pytest.mark.parametrize("engine", ["auto", "pytorch", "llama_cpp"])
def test_qwen3_disk_formats_do_not_imply_pipeline_support(intent_api, engine):
    model = _config(intent_api.root, hf_type="qwen3", gguf_type="qwen3", model_id="qwen3-0.6b")
    intent_api.rows.append(asdict(model))
    local = intent_api.client.get("/api/models").json()["models"][0]
    distributed = intent_api.client.get("/api/models/intent/catalog").json()["models"][0]
    assert local["is_available"] is True and local["has_safetensors"] and local["has_gguf"]
    assert distributed["is_available"] is False
    assert "SUPPLY_MODEL_RUNTIME_UNSUPPORTED" in distributed["unavailable_reason"]
    assert "qwen3" in distributed["unavailable_reason"]
    assert distributed["supported_engines"] == []
    assert "model_path" not in distributed and "gguf_path" not in distributed
    response = intent_api.client.post("/api/models/intent", json={"model_id": model.model_id, "engine": engine})
    assert response.status_code == 409
    assert response.json()["detail"]["code"] == "SUPPLY_MODEL_RUNTIME_UNSUPPORTED"
    intent_api.coordinator.create.assert_not_awaited()
    assert "qwen3" not in PIPELINE_RUNTIME_MODEL_TYPES


@pytest.mark.parametrize("manifest_name", ["source.manifest.json", "source.gguf.manifest.json", "supply.manifest.json"])
@pytest.mark.parametrize("span", [[0, 2], [2, 4]])
def test_gguf_crop_catalog_and_selection_reject_before_retirement(intent_api, monkeypatch, manifest_name, span):
    from model_intent import ModelIntentCoordinator

    model = _config(intent_api.root, hf_type=None, gguf_type="qwen35", model_id="q35-artifacts")
    _gguf(Path(model.gguf_path), "qwen35", total=2)
    (intent_api.root / manifest_name).write_text(json.dumps({
        "n_layer": 4, "source_layer_range": span,
    }), encoding="utf-8")
    intent_api.rows.append(asdict(model))
    retire = AsyncMock()
    owner = ModelIntentCoordinator(intent_api.scheduler, AsyncMock(), retire=retire, root=intent_api.root / "intent")
    original_create = owner.create
    monkeypatch.setattr(owner, "create", AsyncMock(wraps=original_create))
    monkeypatch.setattr(intent_api.routes, "coordinator", lambda: owner)
    transaction = dict(intent_api.scheduler._pipeline_load_transaction)
    plan = dict(intent_api.scheduler._active_pipeline_capacity_plan)
    catalog = intent_api.client.get("/api/models/intent/catalog").json()["models"][0]
    assert catalog["is_available"] is False
    assert "SUPPLY_FULL_SOURCE_REQUIRED" in catalog["unavailable_reason"]
    assert intent_api.client.get("/api/models").json()["models"][0]["is_available"] is True
    response = intent_api.client.post("/api/models/intent", json={"model_id": model.model_id})
    assert response.status_code == 409
    assert response.json()["detail"]["code"] == "SUPPLY_FULL_SOURCE_REQUIRED"
    owner.create.assert_not_awaited()
    retire.assert_not_awaited()
    intent_api.scheduler._model_supply.create.assert_not_called()
    assert owner._retire_task is None
    assert intent_api.scheduler._pipeline_load_transaction == transaction
    assert intent_api.scheduler._active_pipeline_capacity_plan == plan


@pytest.mark.parametrize("bad_engine", ["pytorch", "llama_cpp"])
def test_catalog_and_auto_share_format_evidence_and_offer_explicit_supported_engine(intent_api, bad_engine):
    model = _config(intent_api.root, hf_type="qwen3" if bad_engine == "pytorch" else "qwen2",
                    gguf_type="qwen3" if bad_engine == "llama_cpp" else "qwen2")
    intent_api.rows.append(asdict(model))
    good_engine = "llama_cpp" if bad_engine == "pytorch" else "pytorch"
    catalog = intent_api.client.get("/api/models/intent/catalog").json()["models"][0]
    assert catalog["is_available"] is True
    assert catalog["supported_engines"] == [good_engine]
    assert catalog["preferred_engine"] == good_engine
    assert catalog["pipeline_auto_engine"] == "pytorch"
    assert catalog["pipeline_engine_availability"][bad_engine]["reason_code"] == "SUPPLY_MODEL_RUNTIME_UNSUPPORTED"
    rejected = intent_api.client.post("/api/models/intent", json={"model_id": model.model_id, "engine": bad_engine})
    assert rejected.status_code == 409
    assert good_engine in rejected.json()["detail"]["message"]
    automatic = intent_api.client.post("/api/models/intent", json={"model_id": model.model_id})
    assert automatic.status_code == (409 if bad_engine == "pytorch" else 200)
    if bad_engine == "pytorch":
        assert "SUPPLY_MODEL_RUNTIME_UNSUPPORTED" in catalog["pipeline_auto_unavailable_reason"]
    accepted = intent_api.client.post("/api/models/intent", json={"model_id": model.model_id, "engine": good_engine})
    assert accepted.status_code == 200
    assert intent_api.coordinator.create.await_args.args[0]["engine"] == good_engine


@pytest.mark.parametrize("hf_type,gguf_type", [("qwen3_5", "qwen35"), ("qwen3_5_text", "qwen2"), ("qwen", "qwen2")])
def test_runtime_support_uses_existing_set_and_gguf_alias(tmp_path, hf_type, gguf_type):
    model = _config(tmp_path, hf_type=hf_type, gguf_type=gguf_type)
    for engine in ("auto", "pytorch", "llama_cpp"):
        availability.require_model_pipeline_available(model, engine)
    formats = availability.inspect_model_pipeline_availability(model)
    assert formats["pytorch"].model_type in PIPELINE_RUNTIME_MODEL_TYPES
    assert formats["llama_cpp"].model_type in PIPELINE_RUNTIME_MODEL_TYPES
    assert availability.PIPELINE_RUNTIME_MODEL_TYPES is PIPELINE_RUNTIME_MODEL_TYPES


@pytest.mark.parametrize("case", ["unregistered", "missing", "missing_weights", "unknown_hf", "malformed_type", "unknown_gguf", "missing_selected_format"])
def test_missing_or_unknown_source_stays_with_original_supply_owner(intent_api, case):
    engine = "auto"
    if case != "unregistered":
        model = _config(intent_api.root, hf_type=None, gguf_type=None)
        if case == "missing":
            model.model_path = str(intent_api.root / "not-downloaded")
        elif case == "missing_weights":
            model.model_path = str(_hf(intent_api.root / "hf", "qwen3", weights=False))
        elif case in {"unknown_hf", "malformed_type"}:
            model.model_path = str(_hf(intent_api.root / "hf", 123 if case == "malformed_type" else None))
        elif case == "unknown_gguf":
            path = intent_api.root / "unknown.gguf"
            path.write_bytes(b"invalid metadata")
            model.gguf_path = str(path)
        elif case == "missing_selected_format":
            model.model_path = str(_hf(intent_api.root / "hf", "qwen3"))
            engine = "llama_cpp"
        intent_api.rows.append(asdict(model))
    response = intent_api.client.post("/api/models/intent", json={"model_id": "fixture", "engine": engine})
    assert response.status_code == 200
    intent_api.coordinator.create.assert_awaited_once()


@pytest.mark.parametrize("registered_type,expected", [("qwen2", 200), ("qwen3", 409)])
def test_catalog_and_create_use_shared_registered_override(intent_api, registered_type, expected):
    builtin = _config(intent_api.root / "builtin", hf_type="qwen3" if registered_type == "qwen2" else "qwen2", gguf_type=None,
                      model_id="qwen3-0.6b")
    registered = _config(intent_api.root / "registered", hf_type=registered_type, gguf_type=None,
                         model_id=builtin.model_id)
    intent_api.builtins.append(builtin)
    intent_api.rows.append(asdict(registered))
    catalog = intent_api.client.get("/api/models/intent/catalog").json()["models"][0]
    assert catalog["is_available"] is (expected == 200)
    assert catalog["pipeline_engine_availability"]["pytorch"]["runtime_model_type"] == registered_type
    assert intent_api.client.post("/api/models/intent", json={"model_id": builtin.model_id}).status_code == expected


def test_gguf_header_with_missing_decoder_layers_requires_full_source(tmp_path):
    model = _config(tmp_path, hf_type=None)
    _gguf(Path(model.gguf_path), total=4, layers=[0, 1])
    with pytest.raises(availability.ModelPipelineAvailabilityError) as error:
        availability.require_model_pipeline_available(model)
    assert error.value.code == "SUPPLY_FULL_SOURCE_REQUIRED"


def test_full_gguf_manifest_and_mtp_are_not_pre_cut(tmp_path):
    model = _config(tmp_path, hf_type=None)
    _gguf(Path(model.gguf_path), total=4, mtp=1)
    Path(model.gguf_path).with_suffix(".manifest.json").write_text(json.dumps({
        "source_layer_range": [0, 4], "n_layer": 4, "trim_layers": 0,
    }), encoding="utf-8")
    availability.require_model_pipeline_available(model)


def test_hf_supply_segment_is_not_a_complete_source(tmp_path):
    model = _config(tmp_path, gguf_type=None)
    (Path(model.model_path) / "supply.manifest.json").write_text(json.dumps({
        "format": "safetensors", "layer_range": [0, 2], "total_layers": 4,
    }), encoding="utf-8")
    with pytest.raises(availability.ModelPipelineAvailabilityError) as error:
        availability.require_model_pipeline_available(model, "pytorch")
    assert error.value.code == "SUPPLY_FULL_SOURCE_REQUIRED"


def test_preflight_never_reads_weight_payloads_or_hashes_models(tmp_path, monkeypatch):
    model = _config(tmp_path)
    path, header_end = _gguf(Path(model.gguf_path))
    original_open = Path.open
    reads = []

    class HeaderOnly:
        def __init__(self, handle):
            self.handle = handle
        def __enter__(self):
            return self
        def __exit__(self, *args):
            self.handle.close()
        def read(self, count=-1):
            assert 0 <= count <= header_end - self.handle.tell(), "read reached tensor payload"
            reads.append(count)
            return self.handle.read(count)

    def metadata_open(target, *args, **kwargs):
        assert target.suffix != ".safetensors", "HF weights must not be opened"
        handle = original_open(target, *args, **kwargs)
        return HeaderOnly(handle) if target == path else handle

    monkeypatch.setattr(Path, "open", metadata_open)
    monkeypatch.setattr(hashlib, "sha256", lambda *a, **kw: pytest.fail("preflight must not hash models"))
    availability.require_model_pipeline_available(model)
    assert reads and sum(reads) <= 2 * header_end


def test_helper_imports_without_model_runtime_dependencies():
    source = str(Path(__file__).resolve().parents[1] / "src")
    code = f"""
import sys
sys.path.insert(0, {source!r})
import model_pipeline_availability
for name in ('torch', 'transformers', 'llama_cpp', 'model_module', 'api_server', 'model_supply_sources'):
    assert name not in sys.modules, name
"""
    result = subprocess.run([sys.executable, "-I", "-X", "utf8", "-c", code], capture_output=True,
                            text=True, encoding="utf-8", timeout=30)
    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize("engine", ["pytorch", "llama_cpp"])
@pytest.mark.parametrize("declared_total", [None, 2])
def test_manifest_crop_uses_config_or_header_total_before_create(intent_api, engine, declared_total):
    model = _config(intent_api.root, hf_type="qwen2" if engine == "pytorch" else None,
                    gguf_type="qwen2" if engine == "llama_cpp" else None)
    manifest = (Path(model.model_path) / "supply.manifest.json" if engine == "pytorch"
                else Path(model.gguf_path).with_suffix(".manifest.json"))
    payload = {"layer_range": [0, 2]}
    if declared_total is not None:
        payload["total_layers"] = declared_total
    manifest.write_text(json.dumps(payload), encoding="utf-8")
    intent_api.rows.append(asdict(model))
    catalog = intent_api.client.get("/api/models/intent/catalog").json()["models"][0]
    assert catalog["is_available"] is False
    assert "SUPPLY_FULL_SOURCE_REQUIRED" in catalog["unavailable_reason"]
    response = intent_api.client.post("/api/models/intent", json={"model_id": model.model_id, "engine": engine})
    assert response.status_code == 409
    assert response.json()["detail"]["code"] == "SUPPLY_FULL_SOURCE_REQUIRED"
    intent_api.coordinator.create.assert_not_awaited()


def test_unsupported_runtime_preserves_existing_native_plan(intent_api, monkeypatch):
    from model_intent import ModelIntentCoordinator

    model = _config(intent_api.root, hf_type="qwen3", gguf_type="qwen3")
    intent_api.rows.append(asdict(model))
    retire = AsyncMock()
    owner = ModelIntentCoordinator(intent_api.scheduler, AsyncMock(), retire=retire, root=intent_api.root / "intent")
    monkeypatch.setattr(owner, "create", AsyncMock(wraps=owner.create))
    monkeypatch.setattr(intent_api.routes, "coordinator", lambda: owner)
    transaction = dict(intent_api.scheduler._pipeline_load_transaction)
    plan = dict(intent_api.scheduler._active_pipeline_capacity_plan)
    response = intent_api.client.post("/api/models/intent", json={"model_id": model.model_id})
    assert response.status_code == 409
    assert response.json()["detail"]["code"] == "SUPPLY_MODEL_RUNTIME_UNSUPPORTED"
    owner.create.assert_not_awaited()
    retire.assert_not_awaited()
    intent_api.scheduler._model_supply.create.assert_not_called()
    assert owner._retire_task is None
    assert intent_api.scheduler._pipeline_load_transaction == transaction
    assert intent_api.scheduler._active_pipeline_capacity_plan == plan



def _provenance(model, *, architecture="qwen35", total=4):
    import model_registry_validation as validation
    from relay_segment_info import read_gguf_tokenizer_sha256

    hf = Path(model.model_path)
    (hf / "tokenizer.json").write_text('{"fixture": "tokenizer"}', encoding="utf-8")
    manifest = validation.build_manifest(hf, [hf / "model.safetensors"], model_type="safetensors")
    validation.write_manifest(hf, manifest)
    target = hf.parent / "converted" / "explicit.gguf"
    target.parent.mkdir()
    _gguf(target, architecture, total=total, tokenizer=True)
    proof = {
        "schema_version": 1, "kind": "qlh.cross_format_pipeline_identity.v1",
        "source": {"artifact_sha256": manifest["artifact_sha256"],
                   "manifest_sha256": manifest["manifest_sha256"],
                   "tokenizer_sha256": validation.sha256_file(hf / "tokenizer.json")},
        "conversion": {"output_type": "Q4_K_M", "converter_revision": "a" * 40,
                       "converter_sha256": "b" * 64, "quantizer_revision": "c" * 40,
                       "quantizer_sha256": "d" * 64},
        "target": {"path": "../converted/explicit.gguf", "artifact_sha256": validation.sha256_file(target),
                   "tokenizer_sha256": read_gguf_tokenizer_sha256(target)},
    }
    _write_proof(hf, proof)
    return hf, target, proof


def _write_proof(hf, proof):
    unsigned = {key: value for key, value in proof.items() if key != "provenance_sha256"}
    proof["provenance_sha256"] = hashlib.sha256(json.dumps(
        unsigned, ensure_ascii=True, sort_keys=True, separators=(",", ":"),
    ).encode("utf-8")).hexdigest()
    (hf / "cross_format.provenance.json").write_text(json.dumps(proof), encoding="utf-8")


def test_registered_hf_provenance_exposes_ll_in_catalog_without_changing_local_inventory(intent_api):
    model = _config(intent_api.root / "registered", hf_type="qwen3_5", gguf_type=None, model_id="qwen3-5-9b")
    model.model_type = "safetensors"
    _provenance(model)
    builtin = _config(intent_api.root / "builtin", hf_type="qwen3", gguf_type="qwen3", model_id=model.model_id)
    intent_api.builtins.append(builtin)
    intent_api.rows.append(asdict(model))
    local = intent_api.client.get("/api/models").json()["models"][0]
    catalog = intent_api.client.get("/api/models/intent/catalog").json()["models"][0]
    assert local["supported_engines"] == ["pytorch"] and local["has_gguf"] is False
    assert catalog["supported_engines"] == ["pytorch", "llama_cpp"]
    assert catalog["preferred_engine"] == "pytorch" and catalog["default_quant_type"] == "int4"
    assert catalog["pipeline_auto_engine"] == "pytorch", "automatic source selection stays with supply semantics"
    assert catalog["is_available"] and catalog["has_gguf"]
    ll = catalog["pipeline_engine_availability"]["llama_cpp"]
    assert ll["status"] == "supported" and ll["runtime_model_type"] == "qwen3_5"
    assert ll["source"] == "provenance" and ll["content_verification"] == "supply_owner_required"
    assert "path" not in json.dumps(catalog)
    response = intent_api.client.post("/api/models/intent", json={
        "model_id": model.model_id, "engine": "llama_cpp", "quant_type": "Q4_K_M",
    })
    assert response.status_code == 200
    intent_api.coordinator.create.assert_awaited_once()
    assert intent_api.coordinator.create.await_args.args[0]["engine"] == "llama_cpp"


@pytest.mark.parametrize("case", ["absent", "bad_kind", "bad_digest", "source_mismatch", "bad_manifest",
                                 "missing_manifest", "changed_tokenizer", "absolute", "escape",
                                 "missing_target", "unknown_header"])
def test_unknown_or_invalid_provenance_does_not_invent_ll_availability(intent_api, case):
    model = _config(intent_api.root, hf_type="qwen3_5", gguf_type=None)
    hf, target, proof = _provenance(model)
    if case == "absent":
        (hf / "cross_format.provenance.json").unlink()
    elif case == "bad_kind":
        proof["kind"] = "unknown"
        _write_proof(hf, proof)
    elif case == "bad_digest":
        proof["provenance_sha256"] = "0" * 64
        (hf / "cross_format.provenance.json").write_text(json.dumps(proof), encoding="utf-8")
    elif case == "source_mismatch":
        proof["source"]["artifact_sha256"] = "0" * 64
        _write_proof(hf, proof)
    elif case == "bad_manifest":
        manifest = json.loads((hf / "model.manifest.json").read_text(encoding="utf-8"))
        manifest["artifact_sha256"] = "0" * 64
        (hf / "model.manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    elif case == "missing_manifest":
        (hf / "model.manifest.json").unlink()
    elif case == "changed_tokenizer":
        (hf / "tokenizer.json").write_text("changed", encoding="utf-8")
    elif case in {"absolute", "escape"}:
        proof["target"]["path"] = str(target.resolve()) if case == "absolute" else "../../outside.gguf"
        _write_proof(hf, proof)
    elif case == "missing_target":
        target.unlink()
    elif case == "unknown_header":
        target.write_bytes(b"unknown")
    intent_api.rows.append(asdict(model))
    catalog = intent_api.client.get("/api/models/intent/catalog").json()["models"][0]
    assert catalog["supported_engines"] == ["pytorch"]
    assert catalog["preferred_engine"] == "pytorch"
    assert catalog["pipeline_engine_availability"]["llama_cpp"]["status"] == "missing"
    # The supply owner still acquires missing sources or reports proof errors.
    response = intent_api.client.post("/api/models/intent", json={"model_id": model.model_id, "engine": "llama_cpp"})
    assert response.status_code == 200
    intent_api.coordinator.create.assert_awaited_once()


@pytest.mark.parametrize("architecture,cropped,code", [
    ("qwen3", False, "SUPPLY_MODEL_RUNTIME_UNSUPPORTED"),
    ("qwen35", True, "SUPPLY_FULL_SOURCE_REQUIRED"),
])
def test_provenance_target_shares_runtime_and_crop_preflight(intent_api, architecture, cropped, code):
    model = _config(intent_api.root, hf_type="qwen3_5", gguf_type=None)
    _, target, _ = _provenance(model, architecture=architecture, total=2 if cropped else 4)
    if cropped:
        target.with_suffix(".manifest.json").write_text(json.dumps({
            "n_layer": 4, "source_layer_range": [0, 2],
        }), encoding="utf-8")
    intent_api.rows.append(asdict(model))
    catalog = intent_api.client.get("/api/models/intent/catalog").json()["models"][0]
    assert catalog["supported_engines"] == ["pytorch"]
    assert catalog["pipeline_engine_availability"]["llama_cpp"]["reason_code"] == code
    response = intent_api.client.post("/api/models/intent", json={"model_id": model.model_id, "engine": "llama_cpp"})
    assert response.status_code == 409 and response.json()["detail"]["code"] == code
    intent_api.coordinator.create.assert_not_awaited()


def test_provenance_catalog_checks_metadata_without_hashing_model_payloads(intent_api, monkeypatch):
    import pipeline_model_descriptor as descriptor

    model = _config(intent_api.root, hf_type="qwen3_5", gguf_type=None)
    _provenance(model)
    intent_api.rows.append(asdict(model))
    original = descriptor._sha256_file
    checked = []
    def metadata_hash(path, *args, **kwargs):
        assert path.name == "tokenizer.json", "preflight must never hash weights or the full GGUF"
        checked.append(path)
        return original(path, *args, **kwargs)
    monkeypatch.setattr(descriptor, "_sha256_file", metadata_hash)
    catalog = intent_api.client.get("/api/models/intent/catalog").json()["models"][0]
    assert "llama_cpp" in catalog["supported_engines"]
    assert catalog["preferred_engine"] == "pytorch"
    assert checked


def test_main_ui_preserves_dl_default_for_provenance_ll_catalog(intent_api):
    pytest.importorskip("textual")
    import tui_textual as tt
    from textual.widgets import DataTable

    model = _config(intent_api.root, hf_type="qwen3_5", gguf_type=None, model_id="qwen3-5-9b")
    model.model_type = "safetensors"
    _provenance(model)
    intent_api.rows.append(asdict(model))
    intent_api.coordinator.create.return_value = {
        "operation_id": "supply_" + "a" * 32, "status": "queued", "intent": {"model_id": model.model_id},
    }
    from tui_api import ApiClient
    class UiApi:
        accept_snapshot = ApiClient.accept_snapshot
        _snapshot_cache = {}
        base_url, host, port, timeout, log_token = "http://fixture:18000", "127.0.0.1", 18000, 5.0, ""
        def get(self, path, **kwargs):
            if path == "/models/intent/catalog":
                return intent_api.client.get("/api" + path).json()
            return {"/health": {"status": "ok"}, "/ready": {"ready": True},
                    "/status": {"run_mode": "distributed", "node_role": "master", "model_loaded": False},
                    "/cluster/config/distributed-inference": {"enabled": True},
                    "/models/intent/current": None}.get(path, {})
        def request(self, method, path, body=None, **kwargs):
            assert method == "POST" and path == "/models/intent"
            response = intent_api.client.post("/api" + path, json=body)
            response.raise_for_status()
            return response.json()

    async def run():
        app = tt.KoakumaApp(UiApi(), interval=30)
        async with app.run_test(size=(140, 50)) as pilot:
            app.show_main()
            async def wait_for(predicate):
                for _ in range(200):
                    if predicate():
                        return
                    await pilot.pause(0.02)
                assert predicate(), "main model selection did not finish"
            await wait_for(lambda: isinstance(app.screen, tt.MainScreen) and model.model_id in app.screen.model_rows)
            screen = app.screen
            row = screen.model_rows[model.model_id]
            assert row["preferred_engine"] == "pytorch"
            assert "llama_cpp" in row["supported_engines"]
            screen.switch_page("models")
            screen.query_one("#models-table", DataTable).focus()
            await pilot.press("l")
            await wait_for(lambda: isinstance(app.screen, tt.ConfirmScreen))
            await pilot.press("y")
            await wait_for(lambda: intent_api.coordinator.create.await_count == 1)
            submitted = intent_api.coordinator.create.await_args.args[0]
            assert submitted["model_id"] == model.model_id
            assert submitted["engine"] == "pytorch" and submitted["quant_type"] == "int4"
    asyncio.run(run())



def test_provenance_preserves_preference_for_already_registered_gguf(intent_api, monkeypatch):
    import api_server

    model = _config(intent_api.root, hf_type="qwen3_5", gguf_type=None)
    _, target, _ = _provenance(model)
    model.gguf_path = str(target)
    intent_api.rows.append(asdict(model))
    monkeypatch.setattr(api_server, "_torch_cuda_available", lambda: True)
    catalog = intent_api.client.get("/api/models/intent/catalog").json()["models"][0]
    assert set(catalog["supported_engines"]) == {"pytorch", "llama_cpp"}
    assert catalog["preferred_engine"] == "pytorch" and catalog["default_quant_type"] == "int4"
    assert catalog["pipeline_engine_availability"]["llama_cpp"]["source"] == "provenance"
