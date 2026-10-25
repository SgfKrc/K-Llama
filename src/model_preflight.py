"""Fail-closed model/artifact admission before a pipeline is published.

The capacity solver decides *where* ranges fit. This module decides whether
the selected physical artifacts belong to the requested logical model and
freezes the exact worker-side ``ModelIdentity`` used by Route A.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from typing import Any, Mapping, Sequence


MODEL_PREFLIGHT_SCHEMA = "qlh.model_preflight.v1"

REASON_DESCRIPTOR_INCOMPLETE = "model_preflight_descriptor_incomplete"
REASON_CONTRACT_INCOMPLETE = "model_preflight_contract_incomplete"
REASON_MODEL_ID_MISMATCH = "model_preflight_model_id_mismatch"
REASON_SOURCE_MODEL_MISMATCH = "model_preflight_source_model_mismatch"
REASON_HIDDEN_SIZE_MISMATCH = "model_preflight_hidden_size_mismatch"
REASON_TOKENIZER_MISMATCH = "model_preflight_tokenizer_mismatch"
REASON_ARTIFACT_CONTRACT_CHANGED = "model_preflight_artifact_contract_changed"
REASON_ARTIFACT_IDENTITY_INVALID = "model_preflight_artifact_identity_invalid"
REASON_SEGMENT_ROLE_MISMATCH = "model_preflight_segment_role_mismatch"
REASON_LAYER_COVERAGE_GAP = "model_preflight_layer_coverage_gap"

_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_ARTIFACT_FIELDS = (
    "segment_mode",
    "model_id",
    "artifact_sha256",
    "source_model_sha256",
    "source_model_id",
    "hidden_size",
    "tokenizer_sha256",
)
_MODEL_IDENTITY_FIELDS = ("model_id", "engine", "format", "revision", "sha256")
_REASON_PRIORITY = {
    REASON_DESCRIPTOR_INCOMPLETE: 0,
    REASON_CONTRACT_INCOMPLETE: 1,
    REASON_MODEL_ID_MISMATCH: 2,
    REASON_SOURCE_MODEL_MISMATCH: 3,
    REASON_HIDDEN_SIZE_MISMATCH: 4,
    REASON_TOKENIZER_MISMATCH: 5,
    REASON_ARTIFACT_CONTRACT_CHANGED: 6,
    REASON_ARTIFACT_IDENTITY_INVALID: 7,
    REASON_SEGMENT_ROLE_MISMATCH: 8,
    REASON_LAYER_COVERAGE_GAP: 9,
}


def _json_safe(value: Any) -> Any:
    if isinstance(value, tuple):
        return [_json_safe(item) for item in value]
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    return value


def _range(value: Any) -> tuple[int, int] | None:
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        return None
    start, end = value
    if (
        isinstance(start, bool)
        or isinstance(end, bool)
        or not isinstance(start, int)
        or not isinstance(end, int)
        or start < 0
        or end <= start
    ):
        return None
    return int(start), int(end)


def _merge_ranges(ranges: Sequence[tuple[int, int]]) -> tuple[tuple[int, int], ...]:
    merged: list[list[int]] = []
    for start, end in sorted(set(ranges)):
        if not merged or start > merged[-1][1]:
            merged.append([start, end])
        else:
            merged[-1][1] = max(merged[-1][1], end)
    return tuple((start, end) for start, end in merged)


def stage_capability_sha256(
    artifact: Mapping[str, Any], model_identity: Mapping[str, Any],
) -> str:
    """Return the immutable capability subset bound into one assignment."""

    payload = {
        "artifact": {
            key: artifact.get(key, "")
            for key in ("layer_range",) + _ARTIFACT_FIELDS
        },
        "model_identity": {
            key: model_identity.get(key, "") for key in _MODEL_IDENTITY_FIELDS
        },
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


@dataclass(frozen=True)
class ModelPreflightVerdict:
    ok: bool
    reason_code: str
    reason: str
    model_id: str
    hidden_size: int
    tokenizer_sha256: str
    serving_nodes: tuple[str, ...] = ()
    missing_layer_ranges: tuple[tuple[int, int], ...] = ()
    mismatched_nodes: tuple[dict[str, Any], ...] = ()
    stage_bindings: tuple[dict[str, Any], ...] = ()
    schema: str = MODEL_PREFLIGHT_SCHEMA

    def as_dict(self) -> dict[str, Any]:
        return _json_safe({
            "schema": self.schema,
            "ok": self.ok,
            "reason_code": self.reason_code,
            "reason": self.reason,
            "model_id": self.model_id,
            "hidden_size": self.hidden_size,
            "tokenizer_sha256": self.tokenizer_sha256,
            "serving_nodes": self.serving_nodes,
            "missing_layer_ranges": self.missing_layer_ranges,
            "mismatched_nodes": self.mismatched_nodes,
            "stage_bindings": self.stage_bindings,
        })


def _rejection_reason(
    reason_code: str,
    *,
    model_id: str,
    problems: Sequence[Mapping[str, Any]],
    missing_ranges: Sequence[tuple[int, int]],
) -> str:
    nodes = sorted({
        str(item.get("node_id", "") or "")
        for item in problems if item.get("node_id")
    })
    ranges = ", ".join(f"[{start},{end})" for start, end in missing_ranges) or "none"
    details = "; ".join(
        f"{item.get('node_id', 'unknown')}:{item.get('problem', 'unknown')}"
        for item in problems
    )
    return (
        f"model preflight rejected model={model_id or 'unknown'} "
        f"nodes={','.join(nodes) or 'unknown'} missing_ranges={ranges} "
        f"reason_code={reason_code} details={details or 'none'}"
    )


def preflight_pipeline_artifacts(
    *,
    model_id: str,
    total_layers: Any,
    hidden_size: Any,
    tokenizer_sha256: str,
    source_model_sha256: str = "",
    assignments: Sequence[Mapping[str, Any]],
    nodes: Sequence[Mapping[str, Any]],
    require_distributed: bool = False,
) -> ModelPreflightVerdict:
    """Validate selected fixed artifacts and freeze their offer identities.

    Workers that only advertise legacy ``layer_ranges`` remain outside this
    first-phase artifact preflight. Once an assignment selects a concrete
    ``layer_artifact``, its logical-model contract and unique physical identity
    must agree with the selected PyTorch/Safetensors or llama.cpp/GGUF engine.
    """

    logical_model_id = str(model_id or "").strip()
    try:
        layer_count = int(total_layers)
    except (TypeError, ValueError):
        layer_count = 0
    try:
        expected_hidden = int(hidden_size)
    except (TypeError, ValueError):
        expected_hidden = 0
    expected_tokenizer = str(tokenizer_sha256 or "").strip().lower()
    expected_source_sha256 = str(source_model_sha256 or "").strip().lower()
    selected = [
        assignment for assignment in assignments
        if isinstance(assignment, Mapping)
        and isinstance(assignment.get("layer_artifact"), Mapping)
    ]
    if not selected:
        return ModelPreflightVerdict(
            ok=True,
            reason_code="",
            reason=(
                "no fixed layer artifacts selected"
                if require_distributed else "not applicable"
            ),
            model_id=logical_model_id,
            hidden_size=expected_hidden,
            tokenizer_sha256=expected_tokenizer,
        )

    node_by_id = {
        str(node.get("node_id", "") or ""): node
        for node in nodes if isinstance(node, Mapping)
    }
    problems: list[dict[str, Any]] = []
    bindings: list[dict[str, Any]] = []
    serving_nodes: list[str] = []
    missing_ranges: list[tuple[int, int]] = []

    descriptor_missing = []
    if not logical_model_id:
        descriptor_missing.append("model_id")
    if layer_count <= 0:
        descriptor_missing.append("total_layers")
    if expected_hidden <= 0:
        descriptor_missing.append("hidden_size")
    if _SHA256.fullmatch(expected_tokenizer) is None:
        descriptor_missing.append("tokenizer_sha256")
    if expected_source_sha256 and _SHA256.fullmatch(expected_source_sha256) is None:
        descriptor_missing.append("source_model_sha256")
    if descriptor_missing:
        for assignment in selected:
            requested = _range([
                assignment.get("start_layer"), assignment.get("end_layer"),
            ])
            if requested is not None:
                missing_ranges.append(requested)
            problems.append({
                "node_id": str(assignment.get("node_id", "") or ""),
                "model_id": logical_model_id,
                "problem": "descriptor_incomplete",
                "missing_fields": descriptor_missing,
                "layer_range": list(requested) if requested else [],
            })
        merged = _merge_ranges(missing_ranges)
        return ModelPreflightVerdict(
            ok=False,
            reason_code=REASON_DESCRIPTOR_INCOMPLETE,
            reason=_rejection_reason(
                REASON_DESCRIPTOR_INCOMPLETE,
                model_id=logical_model_id,
                problems=problems,
                missing_ranges=merged,
            ),
            model_id=logical_model_id,
            hidden_size=expected_hidden,
            tokenizer_sha256=expected_tokenizer,
            missing_layer_ranges=merged,
            mismatched_nodes=tuple(problems),
        )

    for assignment in selected:
        node_id = str(assignment.get("node_id", "") or "")
        requested = _range([
            assignment.get("start_layer"), assignment.get("end_layer"),
        ])
        expected_artifact = assignment.get("layer_artifact")
        node = node_by_id.get(node_id)
        node_problems: list[dict[str, Any]] = []
        if requested is None:
            node_problems.append({
                "node_id": node_id,
                "model_id": logical_model_id,
                "problem": "assignment_range_invalid",
                "layer_range": [],
                "reason_code": REASON_LAYER_COVERAGE_GAP,
            })
        current_artifact = None
        artifacts = node.get("layer_artifacts", []) if isinstance(node, Mapping) else []
        if requested is not None and isinstance(artifacts, (list, tuple)):
            current_artifact = next((
                item for item in artifacts
                if isinstance(item, Mapping)
                and _range(item.get("layer_range")) == requested
            ), None)
        if not isinstance(current_artifact, Mapping):
            node_problems.append({
                "node_id": node_id,
                "model_id": logical_model_id,
                "problem": "artifact_missing",
                "layer_range": list(requested) if requested else [],
                "reason_code": REASON_ARTIFACT_CONTRACT_CHANGED,
            })
        else:
            changed_fields = [
                key for key in _ARTIFACT_FIELDS
                if str(current_artifact.get(key, "") or "")
                != str(expected_artifact.get(key, "") or "")
            ]
            if _range(current_artifact.get("layer_range")) != _range(
                expected_artifact.get("layer_range")
            ):
                changed_fields.append("layer_range")
            if changed_fields:
                node_problems.append({
                    "node_id": node_id,
                    "model_id": logical_model_id,
                    "artifact_model_id": str(current_artifact.get("model_id", "") or ""),
                    "problem": "artifact_contract_changed",
                    "changed_fields": changed_fields,
                    "layer_range": list(requested) if requested else [],
                    "reason_code": REASON_ARTIFACT_CONTRACT_CHANGED,
                })

            contract_fields = ("source_model_id", "hidden_size", "tokenizer_sha256")
            missing_fields = [
                key for key in contract_fields
                if current_artifact.get(key) in (None, "")
            ]
            if missing_fields:
                node_problems.append({
                    "node_id": node_id,
                    "model_id": logical_model_id,
                    "artifact_model_id": str(current_artifact.get("model_id", "") or ""),
                    "problem": "artifact_contract_incomplete",
                    "missing_fields": missing_fields,
                    "layer_range": list(requested) if requested else [],
                    "reason_code": REASON_CONTRACT_INCOMPLETE,
                })
            else:
                source_model_id = str(current_artifact.get("source_model_id", "") or "")
                if source_model_id != logical_model_id:
                    node_problems.append({
                        "node_id": node_id,
                        "model_id": source_model_id,
                        "expected_model_id": logical_model_id,
                        "artifact_model_id": str(current_artifact.get("model_id", "") or ""),
                        "problem": "source_model_id_mismatch",
                        "layer_range": list(requested) if requested else [],
                        "reason_code": REASON_MODEL_ID_MISMATCH,
                    })
                try:
                    artifact_hidden = int(current_artifact.get("hidden_size", 0) or 0)
                except (TypeError, ValueError):
                    artifact_hidden = 0
                if artifact_hidden != expected_hidden:
                    node_problems.append({
                        "node_id": node_id,
                        "model_id": source_model_id,
                        "expected_model_id": logical_model_id,
                        "problem": "hidden_size_mismatch",
                        "hidden_size": artifact_hidden,
                        "expected_hidden_size": expected_hidden,
                        "layer_range": list(requested) if requested else [],
                        "reason_code": REASON_HIDDEN_SIZE_MISMATCH,
                    })
                artifact_tokenizer = str(
                    current_artifact.get("tokenizer_sha256", "") or ""
                ).lower()
                if artifact_tokenizer != expected_tokenizer:
                    node_problems.append({
                        "node_id": node_id,
                        "model_id": source_model_id,
                        "expected_model_id": logical_model_id,
                        "problem": "tokenizer_sha256_mismatch",
                        "tokenizer_sha256": artifact_tokenizer,
                        "expected_tokenizer_sha256": expected_tokenizer,
                        "layer_range": list(requested) if requested else [],
                        "reason_code": REASON_TOKENIZER_MISMATCH,
                    })

            mode = str(current_artifact.get("segment_mode", "") or "").lower()
            mode_valid = bool(requested) and (
                (
                    mode == "head"
                    and requested[0] == 0
                    and requested[1] < layer_count
                )
                or (
                    mode == "middle"
                    and requested[0] > 0
                    and requested[1] < layer_count
                )
                or (
                    mode == "tail"
                    and requested[0] > 0
                    and requested[1] == layer_count
                )
            )
            if not mode_valid:
                node_problems.append({
                    "node_id": node_id,
                    "model_id": str(current_artifact.get("source_model_id", "") or ""),
                    "problem": "segment_role_mismatch",
                    "segment_mode": mode,
                    "layer_range": list(requested) if requested else [],
                    "reason_code": REASON_SEGMENT_ROLE_MISMATCH,
                })

            artifact_sha = str(current_artifact.get("artifact_sha256", "") or "").lower()
            artifact_model_id = str(current_artifact.get("model_id", "") or "")
            source_sha = str(current_artifact.get("source_model_sha256", "") or "").lower()
            if expected_source_sha256 and source_sha != expected_source_sha256:
                node_problems.append({
                    "node_id": node_id,
                    "model_id": str(current_artifact.get("source_model_id", "") or ""),
                    "problem": "source_model_sha256_mismatch",
                    "source_model_sha256": source_sha,
                    "expected_source_model_sha256": expected_source_sha256,
                    "layer_range": list(requested) if requested else [],
                    "reason_code": REASON_SOURCE_MODEL_MISMATCH,
                })
            digest_invalid = (
                _SHA256.fullmatch(artifact_sha) is None
                or (source_sha and _SHA256.fullmatch(source_sha) is None)
            )
            models = node.get("models", []) if isinstance(node, Mapping) else []
            formats = {"llama_cpp": "gguf", "pytorch": "safetensors"}
            selected_engines = {
                str(owner["engine"]).strip().lower()
                for owner in (assignment, node or {}) if owner.get("engine")
            }
            # Legacy records may omit an engine. Resolve only from the exact
            # physical model, never from installed engines or another artifact.
            physical_matches = [
                model for model in models
                if isinstance(model, Mapping)
                and str(model.get("model_id", "") or "") == artifact_model_id
                and str(model.get("sha256", "") or "").lower() == artifact_sha
                and str(model.get("engine", "") or "").lower() in formats
                and str(model.get("format", "") or "").lower()
                    == formats[str(model.get("engine", "") or "").lower()]
                and (not selected_engines or selected_engines == {
                    str(model.get("engine", "") or "").lower()})
            ] if isinstance(models, (list, tuple)) else []
            physical_model = physical_matches[0] if len(physical_matches) == 1 else None
            if digest_invalid or not isinstance(physical_model, Mapping):
                node_problems.append({
                    "node_id": node_id,
                    "model_id": str(current_artifact.get("source_model_id", "") or ""),
                    "artifact_model_id": artifact_model_id,
                    "problem": (
                        "artifact_digest_invalid" if digest_invalid
                        else "physical_model_identity_missing"
                    ),
                    "artifact_sha256": artifact_sha,
                    "layer_range": list(requested) if requested else [],
                    "reason_code": REASON_ARTIFACT_IDENTITY_INVALID,
                })
            elif not node_problems:
                model_identity = {
                    key: str(physical_model.get(key, "") or "")
                    for key in _MODEL_IDENTITY_FIELDS
                }
                bindings.append({
                    "node_id": node_id,
                    "layer_range": list(requested),
                    "stage_model_identity": model_identity,
                    "stage_capability_sha256": stage_capability_sha256(
                        current_artifact, model_identity,
                    ),
                })
                serving_nodes.append(node_id)

        if node_problems:
            problems.extend(node_problems)
            if requested is not None:
                missing_ranges.append(requested)

    if problems:
        reason_code = min(
            (
                str(item.get("reason_code", REASON_LAYER_COVERAGE_GAP))
                for item in problems
            ),
            key=lambda code: _REASON_PRIORITY.get(code, 99),
        )
        merged = _merge_ranges(missing_ranges)
        return ModelPreflightVerdict(
            ok=False,
            reason_code=reason_code,
            reason=_rejection_reason(
                reason_code,
                model_id=logical_model_id,
                problems=problems,
                missing_ranges=merged,
            ),
            model_id=logical_model_id,
            hidden_size=expected_hidden,
            tokenizer_sha256=expected_tokenizer,
            serving_nodes=tuple(sorted(set(serving_nodes))),
            missing_layer_ranges=merged,
            mismatched_nodes=tuple(problems),
        )

    return ModelPreflightVerdict(
        ok=True,
        reason_code="",
        reason="all selected layer artifacts match the requested model contract",
        model_id=logical_model_id,
        hidden_size=expected_hidden,
        tokenizer_sha256=expected_tokenizer,
        serving_nodes=tuple(sorted(set(serving_nodes))),
        stage_bindings=tuple(bindings),
    )


def bind_model_preflight(
    plan: Mapping[str, Any],
    *,
    descriptor: Mapping[str, Any],
    nodes: Sequence[Mapping[str, Any]],
    require_distributed: bool = False,
) -> dict[str, Any]:
    """Attach a preflight verdict and immutable stage identities to a plan."""

    result = dict(plan)
    assignments = [dict(item) for item in result.get("assignments", [])]
    verdict = preflight_pipeline_artifacts(
        # Precut GGUF artifacts are often registered under a local deployment
        # alias. Their manifest-declared source_model_id is the logical model
        # identity that must match every remote segment. Descriptors without
        # that field retain the historical model_id contract.
        model_id=str(
            descriptor.get("source_model_id", "")
            or descriptor.get("model_id", "")
            or ""
        ),
        total_layers=descriptor.get("total_layers", 0),
        hidden_size=descriptor.get("hidden_size", 0),
        tokenizer_sha256=str(descriptor.get("tokenizer_sha256", "") or ""),
        source_model_sha256=str(
            descriptor.get("source_model_sha256", "") or ""
        ),
        assignments=assignments,
        nodes=nodes,
        require_distributed=require_distributed,
    )
    result["model_preflight"] = verdict.as_dict()
    if not verdict.ok:
        result.update({
            "status": "rejected",
            "admitted": False,
            "reason_code": verdict.reason_code,
            "reason": verdict.reason,
            "missing_layer_ranges": [
                list(item) for item in verdict.missing_layer_ranges
            ],
            "proposed_assignments": assignments,
            "assignments": [],
        })
        return result

    bindings = {
        (
            str(item.get("node_id", "") or ""),
            tuple(item.get("layer_range", [])),
        ): item
        for item in verdict.stage_bindings
    }
    for assignment in assignments:
        key = (
            str(assignment.get("node_id", "") or ""),
            (
                int(assignment.get("start_layer", -1)),
                int(assignment.get("end_layer", -1)),
            ),
        )
        binding = bindings.get(key)
        if binding:
            assignment["stage_model_identity"] = dict(
                binding["stage_model_identity"]
            )
            assignment["stage_capability_sha256"] = str(
                binding["stage_capability_sha256"]
            )
    result["assignments"] = assignments
    return result


__all__ = [
    "MODEL_PREFLIGHT_SCHEMA",
    "REASON_ARTIFACT_CONTRACT_CHANGED",
    "REASON_ARTIFACT_IDENTITY_INVALID",
    "REASON_CONTRACT_INCOMPLETE",
    "REASON_DESCRIPTOR_INCOMPLETE",
    "REASON_HIDDEN_SIZE_MISMATCH",
    "REASON_LAYER_COVERAGE_GAP",
    "REASON_MODEL_ID_MISMATCH",
    "REASON_SEGMENT_ROLE_MISMATCH",
    "REASON_SOURCE_MODEL_MISMATCH",
    "REASON_TOKENIZER_MISMATCH",
    "ModelPreflightVerdict",
    "bind_model_preflight",
    "preflight_pipeline_artifacts",
    "stage_capability_sha256",
]
