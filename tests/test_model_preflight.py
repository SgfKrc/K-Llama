"""MODEL-PREFLIGHT-01: selected artifacts are admitted before pipeline use."""

from __future__ import annotations

import json
import sys
from pathlib import Path


SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from model_preflight import (  # noqa: E402
    MODEL_PREFLIGHT_SCHEMA,
    REASON_ARTIFACT_CONTRACT_CHANGED,
    REASON_CONTRACT_INCOMPLETE,
    REASON_DESCRIPTOR_INCOMPLETE,
    REASON_HIDDEN_SIZE_MISMATCH,
    REASON_MODEL_ID_MISMATCH,
    REASON_SEGMENT_ROLE_MISMATCH,
    REASON_SOURCE_MODEL_MISMATCH,
    REASON_TOKENIZER_MISMATCH,
    bind_model_preflight,
    preflight_pipeline_artifacts,
)


MODEL = "qwen3-5-2b"
TOTAL = 24
HIDDEN = 2048
TOKENIZER = "c" * 64
SOURCE_SHA = "d" * 64


def _artifact(start, end, *, mode, physical_id, artifact_sha, **overrides):
    value = {
        "layer_range": [start, end],
        "segment_mode": mode,
        "model_id": physical_id,
        "artifact_sha256": artifact_sha,
        "source_model_sha256": SOURCE_SHA,
        "source_model_id": MODEL,
        "hidden_size": HIDDEN,
        "tokenizer_sha256": TOKENIZER,
    }
    value.update(overrides)
    return value


def _node(node_id, artifact):
    return {
        "node_id": node_id,
        "layer_artifacts": [dict(artifact)],
        "models": [{
            "model_id": artifact["model_id"],
            "engine": "llama_cpp",
            "format": "gguf",
            "revision": "cut-v2",
            "sha256": artifact["artifact_sha256"],
        }],
    }


def _assignment(node_id, artifact):
    start, end = artifact["layer_range"]
    return {
        "node_id": node_id,
        "start_layer": start,
        "end_layer": end,
        "layer_artifact": dict(artifact),
    }


def _chain():
    middle = _artifact(
        8, 20, mode="middle", physical_id="middle.gguf", artifact_sha="a" * 64,
    )
    tail = _artifact(
        20, 24, mode="tail", physical_id="tail.gguf", artifact_sha="b" * 64,
    )
    assignments = [
        {"node_id": "master", "start_layer": 0, "end_layer": 8},
        _assignment("surface", middle),
        _assignment("y700", tail),
    ]
    nodes = [_node("surface", middle), _node("y700", tail)]
    return assignments, nodes


def _preflight(assignments, nodes):
    return preflight_pipeline_artifacts(
        model_id=MODEL,
        total_layers=TOTAL,
        hidden_size=HIDDEN,
        tokenizer_sha256=TOKENIZER,
        assignments=assignments,
        nodes=nodes,
        require_distributed=True,
    )


def test_schema_is_versioned_and_json_safe():
    assignments, nodes = _chain()
    payload = _preflight(assignments, nodes).as_dict()
    assert payload["schema"] == MODEL_PREFLIGHT_SCHEMA
    json.dumps(payload)


def test_malformed_descriptor_is_a_named_rejection_not_an_exception():
    assignments, nodes = _chain()
    plan = bind_model_preflight(
        {"admitted": True, "status": "admitted", "assignments": assignments},
        descriptor={
            "model_id": MODEL,
            "total_layers": "not-an-int",
            "hidden_size": "not-an-int",
            "tokenizer_sha256": "missing",
        },
        nodes=nodes,
        require_distributed=True,
    )

    assert plan["admitted"] is False
    assert plan["reason_code"] == REASON_DESCRIPTOR_INCOMPLETE
    assert plan["missing_layer_ranges"] == [[8, 24]]


def test_matching_chain_freezes_exact_physical_identities():
    assignments, nodes = _chain()
    verdict = _preflight(assignments, nodes)
    assert verdict.ok is True, verdict.reason
    assert verdict.serving_nodes == ("surface", "y700")
    assert len(verdict.stage_bindings) == 2
    assert verdict.stage_bindings[0]["stage_model_identity"]["engine"] == "llama_cpp"

    plan = bind_model_preflight(
        {"admitted": True, "status": "admitted", "assignments": assignments},
        descriptor={
            "model_id": MODEL,
            "total_layers": TOTAL,
            "hidden_size": HIDDEN,
            "tokenizer_sha256": TOKENIZER,
        },
        nodes=nodes,
        require_distributed=True,
    )
    assert plan["admitted"] is True
    assert plan["assignments"][1]["stage_model_identity"]["model_id"] == "middle.gguf"
    assert len(plan["assignments"][1]["stage_capability_sha256"]) == 64


def test_manifest_source_model_id_overrides_local_registry_alias():
    assignments, nodes = _chain()
    plan = bind_model_preflight(
        {"admitted": True, "status": "admitted", "assignments": assignments},
        descriptor={
            "model_id": "local-deployment-alias",
            "source_model_id": MODEL,
            "total_layers": TOTAL,
            "hidden_size": HIDDEN,
            "tokenizer_sha256": TOKENIZER,
            "source_model_sha256": SOURCE_SHA,
        },
        nodes=nodes,
        require_distributed=True,
    )

    assert plan["admitted"] is True
    assert plan["model_preflight"]["model_id"] == MODEL
    assert plan["assignments"][1]["stage_model_identity"]["model_id"] == "middle.gguf"


def test_logical_model_id_is_not_the_physical_artifact_id():
    assignments, nodes = _chain()
    nodes[1]["layer_artifacts"][0]["source_model_id"] = "qwen2.5-0.5b"
    verdict = _preflight(assignments, nodes)
    assert verdict.ok is False
    assert verdict.reason_code == REASON_MODEL_ID_MISMATCH
    assert verdict.missing_layer_ranges == ((20, 24),)
    assert "y700" in verdict.reason and MODEL in verdict.reason


def test_hidden_size_mismatch_is_rejected_before_step_zero():
    assignments, nodes = _chain()
    nodes[0]["layer_artifacts"][0]["hidden_size"] = 896
    verdict = _preflight(assignments, nodes)
    assert verdict.reason_code == REASON_HIDDEN_SIZE_MISMATCH
    assert verdict.missing_layer_ranges == ((8, 20),)
    problem = next(
        item for item in verdict.mismatched_nodes
        if item["problem"] == "hidden_size_mismatch"
    )
    assert problem["expected_hidden_size"] == HIDDEN


def test_llama_chain_source_model_digest_must_match_the_master_source():
    assignments, nodes = _chain()
    assignments[1]["layer_artifact"]["source_model_sha256"] = "e" * 64
    nodes[0]["layer_artifacts"][0]["source_model_sha256"] = "e" * 64
    plan = bind_model_preflight(
        {"admitted": True, "status": "admitted", "assignments": assignments},
        descriptor={
            "model_id": MODEL,
            "total_layers": TOTAL,
            "hidden_size": HIDDEN,
            "tokenizer_sha256": TOKENIZER,
            "source_model_sha256": SOURCE_SHA,
        },
        nodes=nodes,
        require_distributed=True,
    )

    assert plan["admitted"] is False
    assert plan["reason_code"] == REASON_SOURCE_MODEL_MISMATCH
    assert plan["missing_layer_ranges"] == [[8, 20]]


def test_tokenizer_mismatch_is_named():
    assignments, nodes = _chain()
    nodes[0]["layer_artifacts"][0]["tokenizer_sha256"] = "e" * 64
    verdict = _preflight(assignments, nodes)
    assert verdict.reason_code == REASON_TOKENIZER_MISMATCH
    problem = next(
        item for item in verdict.mismatched_nodes
        if item["problem"] == "tokenizer_sha256_mismatch"
    )
    assert problem["expected_tokenizer_sha256"] == TOKENIZER


def test_full_range_artifact_cannot_claim_a_partial_segment_role():
    artifact = _artifact(
        0, TOTAL, mode="head", physical_id="whole.gguf", artifact_sha="a" * 64,
    )
    verdict = _preflight([_assignment("edge", artifact)], [_node("edge", artifact)])

    assert verdict.ok is False
    assert verdict.reason_code == REASON_SEGMENT_ROLE_MISMATCH
    assert verdict.missing_layer_ranges == ((0, TOTAL),)


def test_legacy_artifact_contract_is_rejected_only_when_selected():
    assignments, nodes = _chain()
    for key in ("source_model_id", "hidden_size", "tokenizer_sha256"):
        nodes[0]["layer_artifacts"][0].pop(key)
        assignments[1]["layer_artifact"].pop(key)
    verdict = _preflight(assignments, nodes)
    assert verdict.reason_code == REASON_CONTRACT_INCOMPLETE
    assert set(verdict.mismatched_nodes[0]["missing_fields"]) == {
        "source_model_id", "hidden_size", "tokenizer_sha256",
    }

    legacy_only = preflight_pipeline_artifacts(
        model_id=MODEL,
        total_layers=TOTAL,
        hidden_size=HIDDEN,
        tokenizer_sha256=TOKENIZER,
        assignments=[{"node_id": "legacy", "start_layer": 8, "end_layer": 24}],
        nodes=[{"node_id": "legacy", "layer_ranges": [[8, 24]]}],
        require_distributed=True,
    )
    assert legacy_only.ok is True


def test_capability_change_invalidates_the_planned_artifact():
    assignments, nodes = _chain()
    nodes[1]["layer_artifacts"][0]["artifact_sha256"] = "f" * 64
    verdict = _preflight(assignments, nodes)
    assert verdict.reason_code == REASON_ARTIFACT_CONTRACT_CHANGED
    assert verdict.missing_layer_ranges == ((20, 24),)


def test_rejected_plan_keeps_proposed_assignments_for_diagnostics():
    assignments, nodes = _chain()
    nodes[0]["layer_artifacts"][0]["hidden_size"] = 896
    plan = bind_model_preflight(
        {"admitted": True, "status": "admitted", "assignments": assignments},
        descriptor={
            "model_id": MODEL,
            "total_layers": TOTAL,
            "hidden_size": HIDDEN,
            "tokenizer_sha256": TOKENIZER,
        },
        nodes=nodes,
        require_distributed=True,
    )
    assert plan["admitted"] is False
    assert plan["assignments"] == []
    assert plan["proposed_assignments"][1]["node_id"] == "surface"
    assert plan["missing_layer_ranges"] == [[8, 20]]
