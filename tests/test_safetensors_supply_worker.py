"""Bounded real Qwen2 supply registration and v3 segment execution tests."""
from __future__ import annotations

import base64
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import threading
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from transformers import PreTrainedTokenizerFast, Qwen2Config, Qwen2ForCausalLM
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace

import config
import model_module
import model_supply_assets as assets
import model_supply_worker as supply
from inference_service.engine_host import EngineHost
from model_supply_worker import ModelSupplyWorker, get_selected_model
from scheduler_task_worker import SchedulerTaskWorkerMixin
from task_provider import ModelIdentity
from task_worker_protocol import PROTOCOL_VERSION, WorkerProtocolError, _validate_capabilities


class _Scheduler(SchedulerTaskWorkerMixin):
    def __init__(self, root):
        self._host = SimpleNamespace(
            full_chat_execution_lock=threading.RLock(),
            model_loaded=False, is_loaded=False, layer_range=None,
        )
        self._task_worker_stage_lock = threading.RLock()
        self._task_worker_active_attempts = {}
        self._active_pipeline_task_ids = set()
        self._active_layer_config = None
        self.enabled = True
        self.refreshed = 0
        self._model_supply_worker = ModelSupplyWorker(self, root)

    def get_distributed_inference_enabled(self):
        return self.enabled

    def _effective_role(self):
        return "master"

    def get_effective_node_id(self):
        return "master"

    def refresh_task_worker_capabilities(self):
        self.refreshed += 1

    def _runtime_profile_for_capabilities(self):
        return "unspecified"


def _host():
    host = EngineHost.__new__(EngineHost)
    host._layer_upstream = None
    return host


def _request(manifest, hidden, *, want_hidden=True, pos_base=0,
             positions=None, seq_ids=None, workflow="wf-tiny"):
    hidden = np.asarray(hidden, dtype=np.float32).reshape(-1, manifest["hidden_size"])
    raw = hidden.tobytes()
    fields = {
        "layer_range": manifest["layer_range"],
        "handoff_at": manifest["layer_range"][1],
        "hidden_sha256": hashlib.sha256(raw).hexdigest(),
        "hidden_spec": {"n_tokens": hidden.shape[0], "n_embd": hidden.shape[1], "dtype": "float32"},
    }
    if positions is not None:
        fields["positions"] = positions
    if seq_ids is not None:
        fields["seq_ids"] = seq_ids
    return SimpleNamespace(
        stage_type="layer_forward", stage_id="stage-tiny", workflow_id=workflow,
        stage_fields=fields, root_input={
            "hidden_f32": base64.b64encode(raw).decode("ascii"),
            "want_hidden": want_hidden, "pos_base": pos_base, "context_size": 64,
        },
        model_identity=ModelIdentity(
            model_id="supply-" + manifest["artifact_sha256"][:24],
            engine="pytorch", format="safetensors", revision="1",
            sha256=manifest["artifact_sha256"],
        ),
    )


def _hidden(result, width=16):
    assert result["token_argmax"] is None
    raw = base64.b64decode(result["hidden_out_f32"], validate=True)
    assert result["hidden_out_sha256"] == hashlib.sha256(raw).hexdigest()
    return torch.from_numpy(np.frombuffer(raw, dtype=np.float32).copy().reshape(1, -1, width))


@pytest.fixture
def tiny_supply(tmp_path, monkeypatch):
    # Use a separate state tree and CPU. Neither source nor runtime can be 9B.
    monkeypatch.setattr(config, "STATE_DIR", str(tmp_path / "state"))
    monkeypatch.setattr(config, "RUN_MODE", "distributed")
    monkeypatch.setattr(config, "QUANT_TYPE", "fp32")
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.delenv("QLH_LAYER_GGUF", raising=False)
    monkeypatch.setattr(model_module.ModelManager, "_maybe_apply_compile", lambda self: None)
    monkeypatch.setattr(model_module.ModelManager, "_maybe_apply_fused_rmsnorm", lambda self: None)
    previous_threads = torch.get_num_threads()
    previous_rng = torch.get_rng_state()
    hosts = []
    try:
        torch.set_num_threads(1)
        torch.random.default_generator.manual_seed(17)
        model = Qwen2ForCausalLM(Qwen2Config(
            vocab_size=32, hidden_size=16, intermediate_size=32,
            num_hidden_layers=4, num_attention_heads=2, num_key_value_heads=1,
            max_position_embeddings=64, tie_word_embeddings=False,
            attention_dropout=0.0,
        )).eval()
        model.config._attn_implementation = "sdpa"
        source = tmp_path / "source"
        model.save_pretrained(source, safe_serialization=True)
        tokenizer = Tokenizer(WordLevel({"[UNK]": 0, **{f"t{i}": i for i in range(1, 32)}}, unk_token="[UNK]"))
        tokenizer.pre_tokenizer = Whitespace()
        PreTrainedTokenizerFast(tokenizer_object=tokenizer, unk_token="[UNK]").save_pretrained(source)
        source_sha = assets._model_digest(source)
        root = Path(config.STATE_DIR) / "model-supply" / "received"
        segments = []
        for start, end in ((0, 2), (2, 4), (1, 3)):
            result = assets.build_safetensors_segment(
                source, root.parent / "built", source_model_id="tiny-qwen2",
                source_model_sha256=source_sha,
                tokenizer_sha256=assets.sha256_file(source / "tokenizer.json"),
                start_layer=start, end_layer=end, total_layers=4,
                operation_id=f"op-{start}-{end}", node_id="master",
            )
            manifest = assets.validate_supply_manifest(result)
            directory = Path(result["local_files"]["config.json"]).parent
            segments.append((manifest, directory))
        scheduler = _Scheduler(root)

        def activate(index):
            manifest, directory = segments[index]
            result = scheduler._model_supply_worker.activate_local(manifest, directory)
            assert result["status"] == "available", result
            assert get_selected_model()["manifest"] == manifest
            return manifest

        def new_host():
            host = _host()
            hosts.append(host)
            return host

        yield SimpleNamespace(model=model, source=source, scheduler=scheduler,
                              segments=segments, activate=activate, host=new_host)
    finally:
        try:
            for host in hosts:
                host._unload_supplied_layer_manager()
        finally:
            torch.set_rng_state(previous_rng)
            torch.set_num_threads(previous_threads)


def test_registration_projects_exact_physical_identity_without_loading(tiny_supply, monkeypatch):
    def no_load(*args, **kwargs):
        pytest.fail("registration must not load any model")

    monkeypatch.setattr(model_module.ModelManager, "load_model", no_load)
    monkeypatch.setattr(model_module.ModelManager, "load_layer_range", no_load)
    monkeypatch.setattr(model_module.ModelManager, "prepare_pipeline_model", no_load)
    manifest = tiny_supply.activate(2)
    scheduler = tiny_supply.scheduler
    scheduler._active_layer_config = {"layer_range": [0, 1], "engine": "llama_cpp"}
    monkeypatch.setenv("QLH_LAYER_GGUF", str(tiny_supply.source / "missing.gguf"))
    caps = scheduler._task_worker_capabilities()
    identity = {
        "model_id": "supply-" + manifest["artifact_sha256"][:24],
        "engine": "pytorch", "format": "safetensors", "revision": "1",
        "sha256": manifest["artifact_sha256"],
    }
    assert caps["models"] == [identity]
    assert caps["stage_types"] == ["layer_forward"]
    assert caps["layer_ranges"] == [[1, 3]]
    assert caps["layer_artifacts"] == [{
        "layer_range": [1, 3], "segment_mode": "middle", "model_id": identity["model_id"],
        "artifact_sha256": manifest["artifact_sha256"],
        "source_model_id": manifest["source_model_id"],
        "source_model_sha256": manifest["source_model_sha256"],
        "tokenizer_sha256": manifest["tokenizer_sha256"],
        "hidden_size": manifest["hidden_size"],
        "layer_weight_bytes": manifest["layer_weight_bytes"][1:3],
    }]
    assert scheduler._task_worker_offer_model_identity_matches(
        {"stage_type": "layer_forward", "model_identity": identity}, caps,
    )
    assert not scheduler._task_worker_offer_model_identity_matches(
        {"stage_type": "full_inference", "model_identity": identity}, caps,
    )
    assert not scheduler._host.model_loaded
    assert scheduler._host.layer_range is None
    assert str(tiny_supply.source) not in json.dumps(caps)


def test_supply_capabilities_pass_v3_schema(tiny_supply):
    tiny_supply.activate(2)
    caps = tiny_supply.scheduler._task_worker_capabilities()
    _validate_capabilities(caps, version=PROTOCOL_VERSION)


@pytest.mark.parametrize("engine,fmt", [("pytorch", "gguf"), ("llama_cpp", "safetensors")])
def test_supply_capabilities_reject_mixed_engine_format(tiny_supply, engine, fmt):
    tiny_supply.activate(2)
    caps = tiny_supply.scheduler._task_worker_capabilities()
    caps["models"][0].update(engine=engine, format=fmt)
    # Declare the engine so rejection proves the pair check, not availability.
    if engine not in caps["engines"]:
        caps["engines"].append(engine)
    with pytest.raises(WorkerProtocolError, match="supported engine/format pair") as caught:
        _validate_capabilities(caps, version=PROTOCOL_VERSION)
    assert caught.value.code == "invalid_capabilities"
    assert caught.value.field == "payload.capabilities.layer_artifacts[0].model_id"


@pytest.mark.parametrize("full_loaded", [False, True])
def test_metadata_only_architecture_does_not_advertise_layer_forward(tiny_supply, monkeypatch, full_loaded):
    tiny_supply.activate(2)
    full_identity = ModelIdentity(model_id="full-model", engine="pytorch", format="safetensors",
                                  revision="1", sha256="a" * 64)
    if full_loaded:
        host = tiny_supply.scheduler._host
        host.model_loaded = host.is_loaded = True
        tiny_supply.scheduler._require_callbacks = lambda: SimpleNamespace(
            active_task_graph_model_identity=lambda: full_identity)
    selected = get_selected_model()
    selected["manifest"]["model_type"] = "gemma4_unified"
    monkeypatch.setattr(supply, "get_selected_model", lambda root=None: selected)
    caps = tiny_supply.scheduler._task_worker_capabilities()
    assert "layer_forward" not in caps["stage_types"]
    assert caps["layer_worker"] is False
    assert caps["models"][0]["engine"] == "pytorch"
    assert ("full_inference" in caps["stage_types"]) is full_loaded
    expected = [full_identity.snapshot()] if full_loaded else []
    assert tiny_supply.scheduler._task_worker_full_model_identities(caps) == expected
    _validate_capabilities(caps, version=PROTOCOL_VERSION)


def test_supplied_physical_identity_never_becomes_a_full_model_alias(tiny_supply):
    tiny_supply.activate(0)
    scheduler = tiny_supply.scheduler
    identity = scheduler._task_worker_layer_artifact_identity(scheduler._configured_layer_artifact())
    scheduler._host.model_loaded = scheduler._host.is_loaded = True
    scheduler._host.has_loaded_model = lambda: True
    scheduler._require_callbacks = lambda: SimpleNamespace(
        active_task_graph_model_identity=lambda: ModelIdentity(**identity),
    )
    assert scheduler._task_worker_full_model_identity(["pytorch"]) is None
    caps = scheduler._task_worker_capabilities()
    assert caps["stage_types"] == ["layer_forward"]
    # Even a mixed capability must exclude a segment from full-stage identity.
    caps["stage_types"].append("full_inference")
    assert scheduler._task_worker_full_model_identities(caps) == []


def test_two_real_segments_match_full_prefill_decode_reset_and_tail(tiny_supply, monkeypatch):
    model = tiny_supply.model
    head, tail, token_tail = tiny_supply.host(), tiny_supply.host(), tiny_supply.host()
    layers_loaded = []
    prepare_calls = []
    materialized = []
    original_load = model_module.ModelManager.load_layer_range
    original_prepare = model_module.ModelManager.prepare_pipeline_model
    original_materialize = model_module._LayerRangeLoadTracker.materialize

    def load(manager, *args, **kwargs):
        layers_loaded.append((kwargs["start_layer"], kwargs["end_layer"]))
        return original_load(manager, *args, **kwargs)

    def prepare(manager, *args, **kwargs):
        prepare_calls.append(kwargs)
        return original_prepare(manager, *args, **kwargs)

    expected_sources = {}
    for manifest, directory in tiny_supply.segments[:2]:
        index = json.loads((directory / "model.safetensors.index.json").read_text(encoding="utf-8"))
        expected_sources[tuple(manifest["layer_range"])] = {
            key: (directory / shard).resolve(strict=True)
            for key, shard in index["weight_map"].items()
        }

    def materialize(tracker, handle, key, source_file=""):
        assert source_file, "filtered loading must identify the source shard"
        source = Path(source_file).resolve(strict=True)
        assert source == expected_sources[(tracker.start_layer, tracker.end_layer)][key]
        materialized.append((tracker.start_layer, tracker.end_layer, key, source))
        return original_materialize(tracker, handle, key, source_file=source_file)

    def no_full_load(*args, **kwargs):
        pytest.fail("production worker attempted full-model loading")

    monkeypatch.setattr(model_module.ModelManager, "load_layer_range", load)
    monkeypatch.setattr(model_module.ModelManager, "prepare_pipeline_model", prepare)
    monkeypatch.setattr(model_module._LayerRangeLoadTracker, "materialize", materialize)
    monkeypatch.setattr(model_module.ModelManager, "load_model", no_full_load)
    monkeypatch.setattr(model_module.ModelManager, "_load_pytorch", no_full_load)
    monkeypatch.setattr(model_module.AutoModelForCausalLM, "from_pretrained", no_full_load)
    cancel = threading.Event()
    full_cache = None
    forward_caches = []
    max_abs_error = 0.0
    for step, ids in enumerate(([[1, 3, 7]], [[5]], [[9]], [[2, 4]])):
        reset = step in (0, 3)
        pos_base = 0 if reset else 2 + step
        if reset:
            full_cache = None
        input_ids = torch.tensor(ids)
        with torch.no_grad():
            expected = model(
                input_ids, past_key_values=full_cache, use_cache=True, output_hidden_states=True,
            )
            full_cache = expected.past_key_values
            embeddings = model.model.embed_tokens(input_ids)
        manifest = tiny_supply.activate(0)
        head_result = head._execute_layer_forward_stage(
            _request(manifest, embeddings.numpy(), pos_base=pos_base), cancel,
        )
        mid = _hidden(head_result)
        torch.testing.assert_close(mid, expected.hidden_states[2], rtol=2e-5, atol=2e-6)
        manifest = tiny_supply.activate(1)
        tail_result = tail._execute_layer_forward_stage(
            _request(manifest, mid.numpy(), pos_base=pos_base), cancel,
        )
        out = _hidden(tail_result)
        with torch.no_grad():
            actual_logits = tail._supplied_layer_manager.forward_lm_head(out)
        torch.testing.assert_close(actual_logits, expected.logits, rtol=2e-5, atol=2e-6)
        token_result = token_tail._execute_layer_forward_stage(
            _request(manifest, mid.numpy(), pos_base=pos_base, want_hidden=False), cancel,
        )
        assert token_result == {"token_argmax": expected.logits[0, -1].argmax().item()}
        for worker in (head, tail, token_tail):
            cache, next_position = worker._supplied_layer_caches[0]
            assert next_position == pos_base + input_ids.shape[1]
            assert cache.get_seq_length() == next_position
        max_abs_error = max(max_abs_error, (actual_logits - expected.logits).abs().max().item())
        assert len(head._supplied_layer_manager.model.model.layers) == 2
        assert len(tail._supplied_layer_manager.model.model.layers) == 2
        assert tail._supplied_layer_manager.model.model.embed_tokens is None
        forward_caches.append(head._supplied_layer_caches[0][0])

    assert forward_caches[0] is forward_caches[1] is forward_caches[2]
    assert forward_caches[3] is not forward_caches[2]
    print(f"tiny Qwen2 two-segment prefill/decode/reset max_logit_abs_error={max_abs_error:.3g}")
    assert layers_loaded == [(0, 2), (2, 4), (2, 4)]
    assert [call["layer_range"] for call in prepare_calls] == [(0, 2), (2, 4), (2, 4)]
    for start, end, key, source in materialized:
        assert source == expected_sources[(start, end)][key]
        if key.startswith("model.layers."):
            assert start <= int(key.split(".")[2]) < end
    for worker, (_, directory) in ((head, tiny_supply.segments[0]),
                                   (tail, tiny_supply.segments[1]),
                                   (token_tail, tiny_supply.segments[1])):
        index = json.loads((directory / "model.safetensors.index.json").read_text(encoding="utf-8"))
        metrics = worker._supplied_layer_manager._layer_load_metrics
        assert metrics["mode"] == "safetensors_key_filtered"
        assert metrics["source_tensor_bytes"] == index["metadata"]["total_size"]
        assert metrics["selected_tensor_count"] == len(index["weight_map"])
        proof = metrics["materialization_proof"]
        assert {tensor.name: Path(tensor.source_file).resolve(strict=True)
                for tensor in proof.tensors} == expected_sources[proof.layer_range]
        assert {key for start, end, key, _ in materialized
                if (start, end) == proof.layer_range} == set(index["weight_map"])
    # Return exactly v3's token format; run the same prompt from a fresh KV.
    with torch.no_grad():
        ids = torch.tensor([[1, 3, 7]])
        expected = model(ids, use_cache=False).logits[0, -1].argmax().item()
        embeddings = model.model.embed_tokens(ids)
    manifest = tiny_supply.activate(0)
    mid = _hidden(head._execute_layer_forward_stage(_request(manifest, embeddings.numpy()), cancel))
    manifest = tiny_supply.activate(1)
    result = tail._execute_layer_forward_stage(
        _request(manifest, mid.numpy(), want_hidden=False), cancel,
    )
    assert result == {"token_argmax": expected}
    assert layers_loaded == [(0, 2), (2, 4), (2, 4)]


def test_current_range_switch_unloads_old_segment_and_kv(tiny_supply):
    host = tiny_supply.host()
    hidden = np.ones((2, 16), dtype=np.float32)
    manifest = tiny_supply.activate(0)
    host._execute_layer_forward_stage(_request(manifest, hidden), threading.Event())
    previous = host._supplied_layer_manager
    previous_cache = host._supplied_layer_caches[0][0]
    manifest = tiny_supply.activate(2)
    host._execute_layer_forward_stage(_request(manifest, hidden), threading.Event())
    assert previous.model is None
    assert previous.tokenizer is None
    assert host._supplied_layer_manager is not previous
    assert host._supplied_layer_manager.layer_range == (1, 3)
    assert host._supplied_layer_caches[0][0] is not previous_cache
    assert host._supplied_layer_key[0] == manifest["manifest_sha256"]


def test_operation_change_with_same_artifact_reloads_and_clears_cache(tiny_supply):
    host = tiny_supply.host()
    manifest = tiny_supply.activate(0)
    hidden = np.ones((2, 16), dtype=np.float32)
    host._execute_layer_forward_stage(_request(manifest, hidden), threading.Event())
    previous = host._supplied_layer_manager
    old_id = tiny_supply.scheduler._configured_layer_artifact()["model_id"]
    manifest = deepcopy(manifest)
    manifest["operation_id"] = "op-replacement"
    manifest.pop("manifest_sha256")
    manifest["manifest_sha256"] = assets._digest(manifest)
    directory = tiny_supply.segments[0][1]
    (directory / "supply.manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    result = tiny_supply.scheduler._model_supply_worker.activate_local(manifest, directory)
    assert result["status"] == "available", result
    assert tiny_supply.scheduler._configured_layer_artifact()["model_id"] == old_id
    host._execute_layer_forward_stage(_request(manifest, hidden), threading.Event())
    assert previous.model is None
    assert host._supplied_layer_manager is not previous
    assert host._supplied_layer_key[0] == manifest["manifest_sha256"]


@pytest.mark.parametrize("change,error", [
    ("range", "supplied_layer_range_mismatch"),
    ("identity", "supplied_model_identity_mismatch"),
    ("tail", "supplied_segment_has_no_lm_head"),
    ("decode", "supplied_decode_requires_matching_prefill"),
    ("positions", "supplied_positions_must_be_contiguous"),
])
def test_rejects_invalid_stage_before_materializing(tiny_supply, monkeypatch, change, error):
    manifest = tiny_supply.activate(0)
    host = tiny_supply.host()
    request = _request(manifest, np.ones((2, 16), dtype=np.float32))
    if change == "range":
        request.stage_fields["layer_range"] = [0, 1]
    elif change == "identity":
        request.model_identity = {"engine": "llama_cpp"}
    elif change == "tail":
        request.root_input["want_hidden"] = False
    elif change == "decode":
        request.root_input["pos_base"] = 2
    else:
        request.stage_fields["positions"] = [0, 2]
    monkeypatch.setattr(model_module.ModelManager, "load_layer_range",
                        lambda *a, **kw: pytest.fail("invalid stage must not load"))
    with pytest.raises(Exception, match=error):
        host._execute_layer_forward_stage(request, threading.Event())
    assert host._supplied_layer_manager is None


def test_new_workflow_cannot_reuse_old_workflow_kv(tiny_supply):
    manifest = tiny_supply.activate(0)
    host = tiny_supply.host()
    hidden = np.ones((2, 16), dtype=np.float32)
    host._execute_layer_forward_stage(_request(manifest, hidden), threading.Event())
    with pytest.raises(Exception, match="supplied_decode_requires_matching_prefill"):
        host._execute_layer_forward_stage(
            _request(manifest, hidden, pos_base=2, workflow="wf-other"), threading.Event(),
        )
    assert host._supplied_layer_caches == {}


def test_interleaved_sequences_keep_independent_kv_and_positions(tiny_supply):
    manifest = tiny_supply.activate(0)
    mixed, first, second = tiny_supply.host(), tiny_supply.host(), tiny_supply.host()
    raw = np.arange(4 * 16, dtype=np.float32).reshape(4, 16) / 64
    cancel = threading.Event()
    result = _hidden(mixed._execute_layer_forward_stage(
        _request(manifest, raw, seq_ids=[2, 7, 2, 7], positions=[0, 0, 1, 1]), cancel,
    ))
    for seq, host, indices in ((2, first, [0, 2]), (7, second, [1, 3])):
        expected = _hidden(host._execute_layer_forward_stage(_request(manifest, raw[indices]), cancel))
        torch.testing.assert_close(result[:, indices, :], expected)
        assert mixed._supplied_layer_caches[seq][1] == 2
    next_raw = raw[[1, 3]]
    result = _hidden(mixed._execute_layer_forward_stage(
        _request(manifest, next_raw, seq_ids=[7, 2], positions=[2, 2]), cancel,
    ))
    for index, host in ((0, second), (1, first)):
        expected = _hidden(host._execute_layer_forward_stage(
            _request(manifest, next_raw[[index]], pos_base=2), cancel,
        ))
        torch.testing.assert_close(result[:, index:index + 1, :], expected)


def test_revoked_selection_releases_weights_before_gguf_dispatch(tiny_supply, monkeypatch):
    manifest = tiny_supply.activate(0)
    host = tiny_supply.host()
    request = _request(manifest, np.ones((2, 16), dtype=np.float32))
    host._execute_layer_forward_stage(request, threading.Event())
    previous = host._supplied_layer_manager
    (tiny_supply.scheduler._model_supply_worker.root / "current.json").unlink()

    class Upstream:
        def reset(self):
            pass

        def forward_hidden_to_hidden(self, hidden, **kwargs):
            assert previous.model is None
            return hidden

    monkeypatch.setattr(host, "_layer_forward_upstream", lambda: Upstream())
    with pytest.raises(Exception, match="supplied_registration_unavailable"):
        host._execute_layer_forward_stage(request, threading.Event())
    # An actual legacy GGUF request still uses its existing upstream.
    request.model_identity = None
    host._execute_layer_forward_stage(request, threading.Event())
    assert host._supplied_layer_manager is None
    assert host._supplied_layer_caches == {}


def test_forward_failure_discards_mutated_cache(tiny_supply, monkeypatch):
    manifest = tiny_supply.activate(0)
    host = tiny_supply.host()
    hidden = np.ones((2, 16), dtype=np.float32)
    host._execute_layer_forward_stage(_request(manifest, hidden), threading.Event())
    manager = host._supplied_layer_manager
    forward = manager.forward_layers

    def fail(**kwargs):
        forward(**kwargs)
        raise RuntimeError("forward-failed-after-cache-update")

    monkeypatch.setattr(manager, "forward_layers", fail)
    with pytest.raises(RuntimeError, match="forward-failed"):
        host._execute_layer_forward_stage(_request(manifest, hidden, pos_base=2), threading.Event())
    assert host._supplied_layer_caches == {}


def test_explicit_unload_releases_supplied_segment(tiny_supply):
    manifest = tiny_supply.activate(0)
    host = tiny_supply.host()
    host._execute_layer_forward_stage(
        _request(manifest, np.ones((2, 16), dtype=np.float32)), threading.Event(),
    )
    previous = host._supplied_layer_manager
    host._model_change_lock = threading.RLock()
    host._host = SimpleNamespace(
        full_chat_execution_lock=threading.RLock(), has_loaded_model=lambda: False,
        model_loaded=False,
    )
    host._layers = []
    host._reset_runtime_conversation_state = lambda **kw: None
    assert host.unload_model()["success"]
    assert previous.model is None
    assert host._supplied_layer_caches == {}


def test_cancelled_stage_does_not_materialize(tiny_supply, monkeypatch):
    manifest = tiny_supply.activate(0)
    host = tiny_supply.host()
    cancel = threading.Event()
    cancel.set()
    monkeypatch.setattr(model_module.ModelManager, "load_layer_range",
                        lambda *a, **kw: pytest.fail("cancelled stage must not load"))
    with pytest.raises(Exception):
        host._execute_layer_forward_stage(
            _request(manifest, np.ones((2, 16), dtype=np.float32)), cancel,
        )
    assert getattr(host, "_supplied_layer_manager", None) is None


def test_invalid_mixed_prefill_does_not_erase_other_valid_kv(tiny_supply):
    manifest = tiny_supply.activate(0)
    host = tiny_supply.host()
    cancel = threading.Event()
    raw = np.arange(4 * 16, dtype=np.float32).reshape(4, 16) / 64
    host._execute_layer_forward_stage(
        _request(manifest, raw, seq_ids=[2, 7, 2, 7], positions=[0, 0, 1, 1]), cancel,
    )
    before = dict(host._supplied_layer_caches)
    # Sequence 2 requests a reset, but sequence 7 has a broken decode span.
    with pytest.raises(Exception, match="supplied_positions_must_be_contiguous"):
        host._execute_layer_forward_stage(
            _request(manifest, raw[:3], seq_ids=[2, 7, 7], positions=[0, 2, 4]), cancel,
        )
    assert set(host._supplied_layer_caches) == {2, 7}
    for seq in (2, 7):
        assert host._supplied_layer_caches[seq][0] is before[seq][0]
        assert host._supplied_layer_caches[seq][1] == 2
        assert before[seq][0].get_seq_length() == 2
    host._execute_layer_forward_stage(
        _request(manifest, raw[:2], seq_ids=[2, 7], positions=[2, 2]), cancel,
    )
    assert {seq: entry[1] for seq, entry in host._supplied_layer_caches.items()} == {2: 3, 7: 3}


@pytest.mark.parametrize("action", ["load", "switch", "close", "exclusive"])
def test_model_lifecycle_releases_supplied_weights_before_replacement(action):
    host = _host()
    previous = SimpleNamespace(model=object())
    released = []

    def unload():
        previous.model = None
        released.append(True)

    def replacement(**kwargs):
        assert previous.model is None, "old supplied weights survived replacement"
        assert host._supplied_layer_caches == {}
        return {"success": True}

    previous.unload_model = unload
    host._supplied_layer_manager = previous
    host._supplied_layer_key = ("old-manifest", "old-directory")
    host._supplied_layer_workflow = "old-workflow"
    host._supplied_layer_caches = {0: (object(), 2)}
    host._model_change_lock = threading.RLock()
    host._host = SimpleNamespace(
        full_chat_execution_lock=threading.RLock(), load_model=replacement,
        switch_model=replacement, quant_type="fp32", model_loaded=False,
    )
    host._ensure_device_profile = lambda: {}
    host._reset_runtime_conversation_state = lambda **kwargs: None
    host._task_graph_coordinator = SimpleNamespace(close=replacement)
    host._scheduler = None
    if action == "load":
        host.load_model(engine="pytorch", quant_type="fp32")
    elif action == "switch":
        host.switch_model("next-model", engine="pytorch")
    elif action == "close":
        host.close()
        host.close()
    else:
        host._run_exclusive_model_change(replacement)
    assert released == [True]
    assert host._supplied_layer_manager is None
    assert host._supplied_layer_key is None
    assert host._supplied_layer_workflow is None


def test_cancel_during_prepare_never_materializes_weights(tiny_supply, monkeypatch):
    manifest = tiny_supply.activate(0)
    host = tiny_supply.host()
    cancel = threading.Event()
    prepare = model_module.ModelManager.prepare_pipeline_model

    def cancel_after_prepare(manager, *args, **kwargs):
        result = prepare(manager, *args, **kwargs)
        cancel.set()
        return result

    monkeypatch.setattr(model_module.ModelManager, "prepare_pipeline_model", cancel_after_prepare)
    monkeypatch.setattr(model_module.ModelManager, "load_layer_range",
                        lambda *a, **kw: pytest.fail("cancelled prepare must not materialize"))
    with pytest.raises(Exception, match="supplied_stage_cancelled"):
        host._execute_layer_forward_stage(
            _request(manifest, np.ones((2, 16), dtype=np.float32)), cancel,
        )
    assert host._supplied_layer_manager is None
    assert host._supplied_layer_caches == {}


@pytest.mark.parametrize("phase", ["load", "forward", "lm_head"])
def test_cancel_after_mutation_discards_supplied_state(tiny_supply, monkeypatch, phase):
    manifest = tiny_supply.activate(1)
    host = tiny_supply.host()
    raw = np.ones((2, 16), dtype=np.float32)
    cancel = threading.Event()
    if phase == "load":
        load = model_module.ModelManager.load_layer_range

        def cancel_after_load(manager, *args, **kwargs):
            load(manager, *args, **kwargs)
            cancel.set()

        monkeypatch.setattr(model_module.ModelManager, "load_layer_range", cancel_after_load)
        monkeypatch.setattr(model_module.ModelManager, "forward_layers",
                            lambda *a, **kw: pytest.fail("cancelled load must not forward"))
        request = _request(manifest, raw)
    else:
        host._execute_layer_forward_stage(_request(manifest, raw), cancel)
        manager = host._supplied_layer_manager
        cache = host._supplied_layer_caches[0][0]
        method_name = "forward_layers" if phase == "forward" else "forward_lm_head"
        callback = getattr(manager, method_name)

        def cancel_after_callback(*args, **kwargs):
            result = callback(*args, **kwargs)
            cancel.set()
            return result

        monkeypatch.setattr(manager, method_name, cancel_after_callback)
        request = _request(manifest, raw[:1], pos_base=2, want_hidden=phase == "forward")
    with pytest.raises(Exception, match="supplied_stage_cancelled"):
        host._execute_layer_forward_stage(request, cancel)
    assert host._supplied_layer_caches == {}
    if phase == "load":
        assert host._supplied_layer_manager is None
    else:
        assert cache.get_seq_length() == 3  # the mutation happened before cancellation
        cancel.clear()
        with pytest.raises(Exception, match="supplied_decode_requires_matching_prefill"):
            host._execute_layer_forward_stage(request, cancel)
        # Remove fault injection before testing recovery.
        monkeypatch.setattr(manager, method_name, callback)
        # A new prefill after cancellation must use a fresh cache.
        host._execute_layer_forward_stage(_request(manifest, raw), cancel)
        assert host._supplied_layer_caches[0][0] is not cache
        assert host._supplied_layer_caches[0][0].get_seq_length() == 2
