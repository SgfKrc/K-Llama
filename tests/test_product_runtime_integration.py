"""Real tiny selective load -> independent bytes -> v1 contract compatibility.

OS/source fixtures are synthetic: these tests are offline integration, never lab
or three-device acceptance. The model supply, materialization and digest are real.
"""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

import pytest
from tests.test_product_runtime_receipt import loaded, tiny_supply, _api
from product_runtime_receipt import capture_runtime
from package_contract.pc_product_observer import rehash_artifact, bind_artifact_rehash
from package_contract.product_observations import derive_node_observation, ObservationError
from tests.product_observation_fixtures import node_observation


def _raw(loaded):
    runtime = capture_runtime(_api(loaded.manager))
    digest = rehash_artifact(runtime)
    runtime = bind_artifact_rehash(runtime, digest)
    start, end = runtime["layer_range"]
    assignments = [{"node_id": "master", "engine": "pytorch", "start_layer": start, "end_layer": end},
                   {"node_id": "pc", "engine": "llama_cpp", "start_layer": end, "end_layer": 4}]
    identity = {key: runtime[key] for key in ("operation_id", "config_id", "generation")}
    raw, _, _, addresses = node_observation(identity, assignments[0], assignments, "a"*40, 4,
                                           captured_at=runtime["captured_at"])
    before = {key: runtime[key] for key in ("pid", "created_at")}
    facts = raw["facts"]
    facts.update(runtime=runtime, identity_before=before, identity_after=before)
    root = facts["process_tree"]["processes"][0]
    root.update(**before, identity_after=before, model_mappings=[])
    facts["source"].update(before)
    expected = dict(expected_revision="a"*40, expected_node_id="master", expected_identity=identity,
                    expected_assignment={**assignments[0], "_total_layers": 4,
                        "artifact_sha256": loaded.manifest["artifact_sha256"]}, node_addresses=addresses)
    return raw, expected


def test_real_loaded_tensor_proof_and_independent_supply_digest_match_gate(loaded):
    raw, expected = _raw(loaded)
    derived = derive_node_observation(raw, **expected)
    assert derived["native"]["engine"] == "pytorch"
    assert derived["native"]["layer_range"] == [1, 3]
    assert derived["native"]["full_model_loaded"] is False
    proof = raw["facts"]["runtime"]["materialization"]
    assert proof["tensor_metadata"] and proof["materialized_bytes"] > 0


def test_real_supply_tampering_prevents_independent_loaded_identity_proof(loaded):
    runtime = capture_runtime(_api(loaded.manager))
    source = next(iter(runtime["materialization"]["tensor_metadata"].values()))["source_file"]
    path = __import__("pathlib").Path(source)
    with path.open("r+b") as stream:
        stream.seek(-1, 2)
        old = stream.read(1)
        stream.seek(-1, 2)
        stream.write(bytes([old[0] ^ 1]))
    with pytest.raises(ValueError, match="digest"):
        rehash_artifact(runtime)


def test_real_tensor_proof_mutation_is_rejected_by_raw_interpreter(loaded):
    raw, expected = _raw(loaded)
    proof = raw["facts"]["runtime"]["materialization"]
    proof["materialized_bytes"] += 1
    with pytest.raises(ObservationError, match="byte accounting"):
        derive_node_observation(raw, **expected)


@pytest.fixture
def master_multi_shard_load(tiny_supply, tmp_path):
    """Actual master prepare path: no supply SHA override, four source shards."""
    from types import SimpleNamespace
    import json
    import shutil
    from safetensors.torch import save_file
    import model_module
    import model_supply_assets as assets
    source = tmp_path / "four-shard-source"
    source.mkdir()
    for name in ("config.json", "tokenizer.json", "tokenizer_config.json", "special_tokens_map.json"):
        member = tiny_supply.source / name
        if member.exists():
            shutil.copyfile(member, source / name)
    (source / "chat_template.jinja").write_text("{{ messages }}", encoding="utf-8")
    state = tiny_supply.model.state_dict()
    weight_map = {}
    for number in range(4):
        filename = f"model-{number + 1:05d}-of-00004.safetensors"
        keys = sorted(state)[number::4]
        save_file({key: state[key].contiguous() for key in keys}, str(source / filename), metadata={"format": "pt"})
        weight_map.update({key: filename for key in keys})
    (source / "model.safetensors.index.json").write_text(json.dumps({
        "metadata": {"total_size": sum(t.numel() * t.element_size() for t in state.values())},
        "weight_map": weight_map}), encoding="utf-8")
    result = assets.build_safetensors_segment(
        source, tmp_path / "master-segment", source_model_id="tiny-four-shard",
        source_model_sha256=assets._model_digest(source),
        tokenizer_sha256=assets.sha256_file(source / "tokenizer.json"),
        start_layer=0, end_layer=2, total_layers=4, operation_id="op-real", node_id="master")
    manifest = assets.validate_supply_manifest(result)
    directory = Path(result["local_files"]["config.json"]).parent
    assert len([item for item in manifest["files"] if item["kind"] == "weights"]) == 4
    manager = model_module.ModelManager()
    try:
        descriptor = manager.prepare_pipeline_model(model_id="tiny-four-shard", model_path=str(directory),
            quant_type="fp32", layer_range=(0, 2))
        assert descriptor["model_sha256"] != manifest["artifact_sha256"]
        manager.load_layer_range(0, 2, has_embedding=True, has_lm_head=False, total_layers=4)
        yield SimpleNamespace(manager=manager, manifest=manifest, directory=directory,
                              loaded_sha=descriptor["model_sha256"])
    finally:
        manager.unload_model()


def test_real_master_default_directory_load_is_bound_to_assignment_supply_digest(master_multi_shard_load, monkeypatch):
    import model_sync
    loaded = master_multi_shard_load
    proof = loaded.manager._layer_load_metrics["materialization_proof"]
    original = proof.as_dict()
    receipt = capture_runtime(_api(loaded.manager))
    assert receipt["artifact"]["sha256"] == loaded.loaded_sha != loaded.manifest["artifact_sha256"]
    assert len({t["source_file"] for t in receipt["materialization"]["tensor_metadata"].values()}) == 4
    def no_cache(*args, **kwargs):
        pytest.fail("observer reused the loader's cached digest")
    monkeypatch.setattr(model_sync, "compute_model_sha256", no_cache)
    monkeypatch.setattr(model_sync, "compute_file_sha256", no_cache)
    before = {p.name: p.read_bytes() for p in loaded.directory.iterdir()}
    raw, expected = _raw(loaded)
    derived = derive_node_observation(raw, **expected)
    runtime = raw["facts"]["runtime"]
    assert derived["native"]["layer_range"] == [0, 2]
    assert runtime["artifact"]["sha256"] == runtime["artifact"]["rehashed_sha256"] == loaded.manifest["artifact_sha256"]
    assert runtime["materialization"]["loaded_artifact_sha256"] == loaded.loaded_sha
    assert proof.as_dict() == original
    assert {p.name: p.read_bytes() for p in loaded.directory.iterdir()} == before


def test_real_multi_shard_master_still_rejects_changed_last_weight_shard(master_multi_shard_load):
    loaded = master_multi_shard_load
    member = sorted(loaded.directory.glob("*.safetensors"))[-1]
    with member.open("r+b") as stream:
        stream.seek(-1, 2)
        old = stream.read(1)
        stream.seek(-1, 2)
        stream.write(bytes([old[0] ^ 1]))
    with pytest.raises(ValueError, match="member digest"):
        rehash_artifact(capture_runtime(_api(loaded.manager)))
