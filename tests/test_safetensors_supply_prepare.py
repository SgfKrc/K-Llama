"""Registered Safetensors feed legacy prepare without downloading a full model."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import threading
from types import SimpleNamespace

import pytest
import torch
from safetensors import safe_open
from safetensors.torch import save_file

import config
import model_host
import model_supply_assets as assets
import model_supply_worker as supply
import model_sync
import pipeline_model_descriptor
import transport_port
from scheduler_pipeline import SchedulerPipelineMixin


class _Manager:
    engine_type = "pytorch"

    def __init__(self):
        self.is_loaded = False
        self.is_pipeline_prepared = False
        self.layer_range = None
        self.quant_type = "fp32"
        self.model = SimpleNamespace(config=SimpleNamespace(model_type="qwen2"))
        self.prepares = []
        self.loads = []
        self.closed = 0

    def prepare_pipeline_model(self, **kwargs):
        self.prepares.append(kwargs)
        descriptor = pipeline_model_descriptor.inspect_pipeline_model(
            kwargs["model_path"], model_id=kwargs["model_id"],
            layer_range=kwargs.get("layer_range"),
        )
        descriptor["model_sha256"] = kwargs["model_sha256"]
        self.is_pipeline_prepared = True
        return descriptor

    def load_layer_range(self, **kwargs):
        self.loads.append(kwargs)
        self.layer_range = (kwargs["start_layer"], kwargs["end_layer"])
        self.is_loaded = True
        return {"success": True}

    def load_model(self, **_kwargs):
        pytest.fail("a supplied segment must never load a full model")

    def unload_model(self):
        self.closed += 1
        self.is_loaded = False
        self.is_pipeline_prepared = False


class _Worker(SchedulerPipelineMixin):
    def __init__(self, root, host):
        self._host = host
        self._model_supply_worker = SimpleNamespace(root=root)
        self._layer_config_lock = threading.RLock()
        self._layer_execution_lock = threading.RLock()
        self._active_pipeline_task_ids = set()
        self._pending_layer_config = None
        self._active_layer_config = None
        self._last_layer_config_ack_payload = None
        self._pipeline_worker_reserved = False
        self._prepared_layer_configs = {}
        self._pipeline_assignment_rollbacks = {}
        self._local_pipeline_steps = {}
        self._local_device_profile = {"ram": {"available_gb": 1}}
        self._tcp_client = None  # Matching supply must work without a master URL.
        self._pipeline_recovery_validated = True
        self._pipeline_recovery_pending = False
        self._pipeline_recovery_state = None
        self._pipeline_recovery_failure = ""
        self.acks = []

    def get_distributed_inference_enabled(self):
        return True

    def get_effective_node_id(self):
        return "worker"

    def _send_layer_config_ack(self, payload):
        self.acks.append(dict(payload))
        if payload["status"] == "ready":
            self._last_layer_config_ack_payload = dict(payload)
        return True

    def _select_scoring_gpu(self, _profile):
        return None

    def _get_active_pipeline_model_info(self):
        return {}


@pytest.fixture
def prepared_supply(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "RUN_MODE", "distributed")
    monkeypatch.setattr(config, "STATE_DIR", str(tmp_path / "default-state"))
    monkeypatch.delenv("QLH_LAYER_GGUF", raising=False)
    source = tmp_path / "source"
    source.mkdir()
    (source / "config.json").write_text(json.dumps({
        "model_type": "qwen2", "num_hidden_layers": 4,
        "hidden_size": 2, "tie_word_embeddings": False,
    }), encoding="utf-8")
    (source / "tokenizer.json").write_text("{}", encoding="utf-8")
    save_file({
        **{f"model.layers.{i}.input_layernorm.weight": torch.ones(2) for i in range(4)},
        "model.embed_tokens.weight": torch.ones(4, 2),
        "model.norm.weight": torch.ones(2), "lm_head.weight": torch.ones(4, 2),
    }, str(source / "model.safetensors"))
    root = tmp_path / "supply" / "received"
    root.mkdir(parents=True)
    source_sha = assets._model_digest(source)
    tokenizer_sha = assets.sha256_file(source / "tokenizer.json")

    def build(start, end, operation):
        result = assets.build_safetensors_segment(
            source, root.parent / "built", source_model_id="source-qwen2",
            source_model_sha256=source_sha, tokenizer_sha256=tokenizer_sha,
            start_layer=start, end_layer=end, total_layers=4,
            operation_id=operation, node_id="worker",
        )
        directory = Path(result["local_files"]["config.json"]).parent
        return assets.validate_supply_manifest(result), directory

    def select(manifest, directory):
        (root / "current.json").write_text(json.dumps({
            "kind": "qlh.model_supply.current.v1",
            "directory": directory.relative_to(root.parent).as_posix(),
            "manifest": "supply.manifest.json",
            "manifest_sha256": manifest["manifest_sha256"],
            "artifact_sha256": manifest["artifact_sha256"],
        }), encoding="utf-8")
        selected = supply.get_selected_model(root)
        assert selected and selected["directory"] == str(directory)
        return selected

    manifest, directory = build(1, 3, "op-current")
    selected = select(manifest, directory)
    active, candidate = _Manager(), _Manager()
    host = model_host.ModelHost(manager=active)
    monkeypatch.setattr(model_host._LazyModelManager, "_get_instance", lambda _self: candidate)
    worker = _Worker(root, host)

    def forbidden(*_args, **_kwargs):
        pytest.fail("registered supply must not resolve, hash or download a full source")

    for name in ("resolve_worker_model_path", "ensure_model_available",
                 "ensure_pipeline_assignment_available"):
        monkeypatch.setattr(model_sync, name, forbidden)
    monkeypatch.setattr(transport_port, "compute_local_model_sha256", forbidden)
    payload = {
        "node_id": "worker", "config_id": "cfg-supply", "generation": 7,
        "assignment_id": "assignment-supply", "phase": "prepare", "plan_id": "plan-supply",
        "model_id": manifest["source_model_id"], "model_type": "qwen2",
        "model_sha256": manifest["source_model_sha256"],
        "tokenizer_sha256": manifest["tokenizer_sha256"],
        "total_layers": 4, "start_layer": 1, "end_layer": 3,
        "has_embedding": False, "has_lm_head": False,
        "engine": "pytorch", "required_bytes": 1024,
    }
    return SimpleNamespace(worker=worker, host=host, active=active, candidate=candidate,
                           root=root, manifest=manifest, directory=directory,
                           selected=selected, payload=payload, build=build, select=select,
                           source=source, source_sha=source_sha, tokenizer_sha=tokenizer_sha)


def test_safetensors_build_cancels_during_copy_and_cleans_stage(prepared_supply, monkeypatch):
    case = prepared_supply
    stopped = [False]
    original_copy = assets._copy_selected

    def cancel_before_copy(*args, **kwargs):
        stopped[0] = True
        return original_copy(*args, **kwargs)

    monkeypatch.setattr(assets, "_copy_selected", cancel_before_copy)
    with pytest.raises(assets.SupplyError, match="cancelled"):
        assets.build_safetensors_segment(
            case.source, case.root.parent / "cancelled-build",
            source_model_id="source-qwen2", source_model_sha256=case.source_sha,
            tokenizer_sha256=case.tokenizer_sha, start_layer=0, end_layer=2,
            total_layers=4, operation_id="op-cancel", node_id="worker",
            cancelled=lambda: stopped[0],
        )
    assert not list((case.root.parent / "cancelled-build").glob(".supply-build-*"))


@pytest.mark.parametrize("assignment_manifest", [False, True])
def test_matching_registration_uses_source_identity_and_exact_candidate(prepared_supply, assignment_manifest):
    case = prepared_supply
    case.worker._handle_layer_config("master", {
        **case.payload, "assignment_manifest": assignment_manifest,
    })
    assert case.worker.acks[-1]["status"] == "prepared"
    assert case.worker.acks[-1]["model_sha256"] == case.manifest["source_model_sha256"]
    assert case.host.peek_manager() is case.active
    assert case.worker._active_layer_config is None
    assert case.candidate.prepares == [{
        "model_id": case.payload["model_id"], "model_path": str(case.directory),
        "quant_type": None, "layer_range": (1, 3),
        "model_sha256": case.manifest["source_model_sha256"],
    }]
    assert case.candidate.loads[0]["model_path"] == str(case.directory)
    assert (case.candidate.loads[0]["start_layer"], case.candidate.loads[0]["end_layer"]) == (1, 3)
    assert case.manifest["source_model_sha256"] != case.manifest["artifact_sha256"]
    for entry in case.manifest["files"]:
        if entry["kind"] == "weights":
            with safe_open(str(case.directory / entry["path"]), framework="pt") as weights:
                assert not any(key.startswith(("model.layers.0.", "model.layers.3.")) for key in weights.keys())
    record = case.worker._prepared_layer_configs["cfg-supply"]
    assert record["model_path"] == str(case.directory)
    assert record["generation"] == 7


@pytest.mark.parametrize("changes,field", [
    ({"start_layer": 0}, "layer_range"),
    ({"end_layer": 2}, "layer_range"),
    ({"end_layer": 4}, "layer_range"),
    ({"model_sha256": "f" * 64}, "source_model_sha256"),
    ({"model_id": "different-source"}, "source_model_id"),
    ({"total_layers": 8}, "total_layers"),
    ({"tokenizer_sha256": "f" * 64}, "tokenizer_sha256"),
])
def test_registration_mismatch_fails_by_name_without_fallback(prepared_supply, changes, field):
    case = prepared_supply
    case.worker._handle_layer_config("master", {
        **case.payload, "assignment_manifest": True, **changes,
    })
    ack = case.worker.acks[-1]
    assert ack["status"] == "error"
    assert ack["reason_code"] == "pipeline_supply_identity_mismatch"
    assert field in ack["error"]
    assert case.candidate.prepares == case.candidate.loads == []
    assert case.host.peek_manager() is case.active
    assert case.worker._prepared_layer_configs == {}


@pytest.mark.parametrize("field", ["source_model_id", "source_model_sha256", "tokenizer_sha256"])
def test_descriptor_identity_cannot_override_registered_identity(prepared_supply, monkeypatch, field):
    case = prepared_supply
    original = pipeline_model_descriptor.inspect_pipeline_model

    def changed(*args, **kwargs):
        descriptor = original(*args, **kwargs)
        descriptor[field] = "different"
        return descriptor

    monkeypatch.setattr(pipeline_model_descriptor, "inspect_pipeline_model", changed)
    case.worker._handle_layer_config("master", case.payload)
    assert case.worker.acks[-1]["reason_code"] == "pipeline_supply_identity_mismatch"
    assert field in case.worker.acks[-1]["error"]
    assert case.candidate.prepares == []


@pytest.mark.parametrize("pointer_change", ["replace", "revoke"])
def test_commit_consumes_existing_transaction_after_pointer_change(prepared_supply, monkeypatch, pointer_change):
    case = prepared_supply
    case.worker._handle_layer_config("master", case.payload)
    assert case.worker.acks[-1]["status"] == "prepared"
    if pointer_change == "replace":
        manifest, directory = case.build(0, 1, "op-new")
        case.select(manifest, directory)
    else:
        (case.root / "current.json").unlink()
        assert supply.get_selected_model(case.root) is None

    def no_rediscovery(*_args, **_kwargs):
        pytest.fail("commit must consume the prepared transaction without pointer lookup")

    monkeypatch.setattr(supply, "get_selected_model", no_rediscovery)
    commit = {**case.payload, "phase": "commit"}
    case.worker._handle_layer_config("master", commit)
    assert case.worker.acks[-1]["status"] == "ready"
    assert case.host.peek_manager() is case.candidate
    assert case.host.pipeline_candidate_status("cfg-supply")["phase"] == "committed"
    assert case.worker._prepared_layer_configs == {}
    assert case.worker._active_layer_config["layer_range"] == [1, 3]
    assert len(case.candidate.loads) == len(case.candidate.prepares) == 1
    assert case.active.closed == 0
    # Retry replays the receipt even after the prepared record was consumed.
    case.worker._handle_layer_config("master", commit)
    assert case.worker.acks[-1]["status"] == "ready"
    assert case.host.peek_manager() is case.candidate
    assert len(case.candidate.loads) == 1


@pytest.mark.parametrize("changes", [
    {"generation": 8}, {"plan_id": "other-plan"},
    {"assignment_id": "other-assignment"}, {"total_layers": 8},
    {"has_embedding": True}, {"tokenizer_sha256": "f" * 64},
])
def test_commit_rejects_a_different_prepared_contract(prepared_supply, changes):
    case = prepared_supply
    case.worker._handle_layer_config("master", case.payload)
    assert case.worker.acks[-1]["status"] == "prepared"
    case.worker._handle_layer_config("master", {**case.payload, "phase": "commit", **changes})
    assert case.worker.acks[-1]["status"] == "error"
    assert "prepared" in case.worker.acks[-1]["error"]
    assert case.host.peek_manager() is case.active
    assert case.host.pipeline_candidate_status("cfg-supply")["phase"] == "aborted"
    assert len(case.candidate.loads) == 1


@pytest.mark.parametrize("pointer_change", ["replace", "revoke"])
def test_pointer_change_during_materialization_aborts_candidate(prepared_supply, monkeypatch, pointer_change):
    case = prepared_supply
    replacement = case.build(0, 1, "op-new") if pointer_change == "replace" else None
    original = case.candidate.load_layer_range

    def materialize(**kwargs):
        result = original(**kwargs)
        if replacement:
            case.select(*replacement)
        else:
            (case.root / "current.json").unlink()
        return result

    monkeypatch.setattr(case.candidate, "load_layer_range", materialize)
    case.worker._handle_layer_config("master", case.payload)
    assert case.worker.acks[-1]["reason_code"] == "pipeline_supply_selection_changed"
    assert case.worker._prepared_layer_configs == {}
    assert case.host.peek_manager() is case.active
    assert case.host.pipeline_candidate_status("cfg-supply")["phase"] == "aborted"


def _recovery(case):
    case.worker._pipeline_recovery_pending = True
    case.worker._pipeline_recovery_state = {
        "model_id": case.payload["model_id"], "model_type": "qwen2",
        "model_sha256": case.payload["model_sha256"], "total_layers": 4,
        "tokenizer_sha256": case.payload["tokenizer_sha256"], "quant_type": "fp32",
        "assignments": [{"node_id": "worker", "start_layer": 1, "end_layer": 3}],
    }


def test_cold_recovery_uses_verified_directory_source_sha_and_local_range(prepared_supply, monkeypatch):
    import model_config
    case = prepared_supply
    _recovery(case)

    def no_registry(*_args, **_kwargs):
        pytest.fail("current registered supply must precede the ordinary model registry")

    monkeypatch.setattr(model_config, "get_model_config", no_registry)
    assert case.worker._restore_pipeline_model_for_recovery()
    assert case.active.prepares[0]["model_path"] == str(case.directory)
    assert case.active.prepares[0]["model_sha256"] == case.manifest["source_model_sha256"]
    assert case.active.prepares[0]["layer_range"] == (1, 3)
    assert case.active.loads == []


@pytest.mark.parametrize("changes", [
    {"model_sha256": "f" * 64}, {"total_layers": 8},
    {"tokenizer_sha256": "f" * 64},
    {"assignments": [{"node_id": "worker", "start_layer": 0, "end_layer": 3}]},
    {"assignments": []},
])
def test_cold_recovery_mismatch_cannot_load_old_registered_segment(prepared_supply, changes):
    case = prepared_supply
    _recovery(case)
    case.worker._pipeline_recovery_state.update(changes)
    assert not case.worker._restore_pipeline_model_for_recovery()
    assert case.worker._pipeline_recovery_failure == "pipeline_supply_identity_mismatch"
    assert case.active.prepares == case.active.loads == []


def test_real_qwen2_candidate_materializes_only_supplied_layers(prepared_supply, monkeypatch, tmp_path):
    from transformers import PreTrainedTokenizerFast, Qwen2Config, Qwen2ForCausalLM
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    import model_module

    case = prepared_supply
    monkeypatch.setattr(config, "QUANT_TYPE", "fp32")
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(model_module.ModelManager, "_maybe_apply_compile", lambda _self: None)
    monkeypatch.setattr(model_module.ModelManager, "_maybe_apply_fused_rmsnorm", lambda _self: None)
    previous_threads, previous_rng = torch.get_num_threads(), torch.get_rng_state()
    manager = model_module.ModelManager()
    try:
        torch.set_num_threads(1)
        torch.manual_seed(17)
        model = Qwen2ForCausalLM(Qwen2Config(
            vocab_size=32, hidden_size=16, intermediate_size=32,
            num_hidden_layers=4, num_attention_heads=2, num_key_value_heads=1,
            max_position_embeddings=64, tie_word_embeddings=False,
        )).eval()
        source = tmp_path / "real-source"
        model.save_pretrained(source, safe_serialization=True)
        tokenizer = Tokenizer(WordLevel({"[UNK]": 0, **{f"t{i}": i for i in range(1, 32)}}, unk_token="[UNK]"))
        PreTrainedTokenizerFast(tokenizer_object=tokenizer, unk_token="[UNK]").save_pretrained(source)
        result = assets.build_safetensors_segment(
            source, case.root.parent / "real-built", source_model_id="source-qwen2",
            source_model_sha256=assets._model_digest(source),
            tokenizer_sha256=assets.sha256_file(source / "tokenizer.json"),
            start_layer=1, end_layer=3, total_layers=4,
            operation_id="op-real", node_id="worker",
        )
        manifest = assets.validate_supply_manifest(result)
        directory = Path(result["local_files"]["config.json"]).parent
        case.select(manifest, directory)
        monkeypatch.setattr(model_host._LazyModelManager, "_get_instance", lambda _self: manager)
        materialized = []
        original = model_module._LayerRangeLoadTracker.materialize

        index = json.loads((directory / "model.safetensors.index.json").read_text(encoding="utf-8"))
        expected_sources = {key: (directory / shard).resolve(strict=True)
                            for key, shard in index["weight_map"].items()}
        observed_sources = {}

        def tracked(tracker, handle, key, source_file=""):
            assert source_file, "filtered loading must identify the source shard"
            source = Path(source_file).resolve(strict=True)
            assert source == expected_sources[key]
            observed_sources[key] = source
            materialized.append(key)
            return original(tracker, handle, key, source_file=source_file)

        def no_full(*_args, **_kwargs):
            pytest.fail("legacy supply prepare attempted to load a full source model")

        monkeypatch.setattr(model_module._LayerRangeLoadTracker, "materialize", tracked)
        monkeypatch.setattr(model_module.ModelManager, "load_model", no_full)
        monkeypatch.setattr(model_module.ModelManager, "_load_pytorch", no_full)
        monkeypatch.setattr(model_module.AutoModelForCausalLM, "from_pretrained", no_full)
        payload = {**case.payload, "model_sha256": manifest["source_model_sha256"],
                   "tokenizer_sha256": manifest["tokenizer_sha256"], "master_quant_type": "fp32"}
        case.worker._handle_layer_config("master", payload)
        assert case.worker.acks[-1]["status"] == "prepared", case.worker.acks[-1]
        assert case.host.peek_manager() is case.active
        assert manager.layer_range == (1, 3)
        assert len(manager.model.model.layers) == 2
        assert materialized
        assert observed_sources == expected_sources
        proof = manager._layer_load_metrics["materialization_proof"]
        assert {tensor.name: Path(tensor.source_file).resolve(strict=True)
                for tensor in proof.tensors} == observed_sources
        assert not any(key.startswith(("model.layers.0.", "model.layers.3.")) for key in materialized)
        assert {int(key.split(".")[2]) for key in materialized if key.startswith("model.layers.")} == {1, 2}
        assert manager.get_pipeline_descriptor()["model_sha256"] == manifest["source_model_sha256"]
        case.worker._handle_layer_config("master", {**payload, "phase": "commit"})
        assert case.worker.acks[-1]["status"] == "ready", case.worker.acks[-1]
        assert case.host.peek_manager() is manager
        assert not case.host.model_loaded
    finally:
        manager.unload_model()
        torch.set_rng_state(previous_rng)
        torch.set_num_threads(previous_threads)



def test_real_metadata_prepare_keeps_supply_registration_across_unload(prepared_supply):
    import model_module

    case = prepared_supply
    manager = model_module.ModelManager()
    before = {path.name: path.read_bytes() for path in case.directory.iterdir()}
    try:
        # The master computes a physical model identity when no candidate SHA
        # was supplied. Those repeated metadata reads must leave the signed
        # directory usable by the next prepare, including after public unload.
        digests = []
        for _ in range(2):
            descriptor = manager.prepare_pipeline_model(
                model_id=case.manifest["source_model_id"],
                model_path=str(case.directory),
                quant_type="fp16",
                layer_range=tuple(case.manifest["layer_range"]),
            )
            digests.append(descriptor["model_sha256"])
            assert len(digests[-1]) == 64
            selected = supply.get_selected_model(case.root)
            assert selected and selected["manifest"] == case.manifest
            manager.unload_model()
            assert supply.get_selected_model(case.root)
        assert digests[0] == digests[1]
        assert {path.name: path.read_bytes() for path in case.directory.iterdir()} == before
    finally:
        manager.unload_model()


def test_cross_format_supply_uses_proven_target_identity(prepared_supply):
    case = prepared_supply
    # Portable segments retain the GGUF target identity and the original HF
    # tokenizer digest in their proof, even when that GGUF is not on the worker.
    manifest = dict(case.manifest)
    target_sha, target_tokenizer = "a" * 64, "b" * 64
    proof = {
        "schema_version": 1, "kind": "qlh.cross_format_pipeline_identity.v1",
        "source": {
            "artifact_sha256": case.manifest["source_model_sha256"],
            "manifest_sha256": "c" * 64,
            "tokenizer_sha256": case.manifest["tokenizer_sha256"],
        },
        "conversion": {
            "output_type": "Q4_K_M", "converter_revision": "d" * 40,
            "converter_sha256": "e" * 64, "quantizer_revision": "f" * 40,
            "quantizer_sha256": "1" * 64,
        },
        "target": {
            "path": "../absent-source.gguf", "artifact_sha256": target_sha,
            "tokenizer_sha256": target_tokenizer,
        },
    }

    def digest(payload):
        return hashlib.sha256(json.dumps(
            payload, ensure_ascii=True, sort_keys=True, separators=(",", ":"),
        ).encode()).hexdigest()

    proof["provenance_sha256"] = digest(proof)
    manifest.update(source_model_sha256=target_sha, tokenizer_sha256=target_tokenizer,
                    conversion_proof=proof)
    manifest.pop("manifest_sha256")
    manifest["manifest_sha256"] = digest(manifest)
    (case.directory / "supply.manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    manifest = assets.validate_supply_manifest(manifest)
    case.select(manifest, case.directory)
    payload = {**case.payload, "model_sha256": target_sha, "tokenizer_sha256": target_tokenizer}
    case.worker._handle_layer_config("master", payload)
    assert case.worker.acks[-1]["status"] == "prepared", case.worker.acks[-1]
    assert case.candidate.prepares[0]["model_sha256"] == target_sha
    descriptor = case.host.pipeline_candidate_status("cfg-supply")["descriptor"]
    assert descriptor["source_tokenizer_sha256"] == case.manifest["tokenizer_sha256"]
    assert descriptor["tokenizer_sha256"] == target_tokenizer
    assert descriptor["source_model_sha256"] == target_sha
    assert descriptor["model_sha256"] == target_sha


def test_next_prepare_rereads_pointer_and_rejects_old_range(prepared_supply):
    case = prepared_supply
    manifest, directory = case.build(0, 1, "op-replaced")
    case.select(manifest, directory)
    case.worker._handle_layer_config("master", case.payload)
    ack = case.worker.acks[-1]
    assert ack["reason_code"] == "pipeline_supply_identity_mismatch"
    assert "layer_range" in ack["error"]
    assert case.candidate.prepares == case.candidate.loads == []


def test_revoked_registration_is_not_cached_by_pipeline(prepared_supply):
    case = prepared_supply
    assert case.worker._selected_pipeline_safetensors()["directory"] == str(case.directory)
    (case.root / "current.json").unlink()
    assert case.worker._selected_pipeline_safetensors() is None
