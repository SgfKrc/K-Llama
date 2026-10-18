"""Shared engine-schema adapters for worker hello and planner admission."""
from __future__ import annotations

from typing import Any, Iterable, Mapping

from qlh_core.contracts.engines import (
    ENGINE_REGISTRY, FULL_REQUEST, LAYER_RANGE, FORWARD_LAYERS,
    advertisement_supports,
)


def build_engine_capabilities(
    runtimes: Iterable[tuple[str, str]], *,
    allow_layers: bool = True,
) -> dict[str, Any]:
    entries = []
    for family, kind in dict.fromkeys(runtimes):
        spec = ENGINE_REGISTRY.require(family, kind)
        operations = spec.operations
        if not allow_layers:
            operations = operations - {LAYER_RANGE, FORWARD_LAYERS}
        entries.append(ENGINE_REGISTRY.narrow(family, kind, operations))
    return ENGINE_REGISTRY.advertisement(entries)


def validate_worker_engine_capabilities(capabilities: Mapping[str, Any]) -> None:
    """Validate the optional v1 projection against the existing wire fields.

    Absence keeps old hello messages decodable; it is never a planner grant.
    """
    if "engine_capabilities" not in capabilities:
        return
    entries = ENGINE_REGISTRY.validate_advertisement(capabilities["engine_capabilities"])
    engines = capabilities.get("engines", [])
    if any(entry.family not in engines for entry in entries):
        raise ValueError("engine capability family was not declared in engines")
    stages = capabilities.get("stage_types", [])
    if "full_inference" in stages and not any(entry.supports(FULL_REQUEST) for entry in entries):
        raise ValueError("full_inference requires a full_request runtime")
    if "layer_forward" in stages and not any(
        entry.supports(FORWARD_LAYERS) for entry in entries
    ):
        raise ValueError("layer_forward requires a forward_layers runtime")
    if capabilities.get("layer_ranges") or capabilities.get("layer_artifacts") or capabilities.get("layer_budget"):
        if not any(entry.supports(LAYER_RANGE) and entry.supports(FORWARD_LAYERS) for entry in entries):
            raise ValueError("layer ranges/artifacts/budget require a layer runtime")
    model_families = {item.get("model_id"): item.get("engine")
                      for item in capabilities.get("models", []) if isinstance(item, Mapping)}
    for artifact in capabilities.get("layer_artifacts", []):
        family = model_families.get(artifact.get("model_id"))
        if not any(entry.family == family and entry.supports(LAYER_RANGE)
                   and entry.supports(FORWARD_LAYERS) for entry in entries):
            raise ValueError("layer artifact engine has no layer runtime")


def engine_profile(profile: Mapping[str, Any], schema: Any) -> dict[str, Any]:
    """A current concrete schema supersedes historical profile runtime hints."""
    entries = ENGINE_REGISTRY.validate_advertisement(schema)
    result = dict(profile)
    for key in ("backend_id", "engine", "runtime_kind", "inference_path", "native_abi",
                "serves_layer_segments"):
        result.pop(key, None)
    result["engine_capabilities"] = schema
    families = {entry.family for entry in entries}
    if len(families) == 1:
        result["backend_id"] = next(iter(families))
    return result


def node_supports_forward_layers(node: Any) -> bool:
    """Use a concrete runtime or the validated worker schema, never node_type."""
    if isinstance(node, Mapping):
        info = node.get("device_info") or {}
    else:
        info = getattr(node, "device_info", None) or {}
    if not isinstance(info, Mapping):
        return False
    reported = info.get("capabilities")
    candidates = [info]
    if isinstance(reported, Mapping):
        candidates.insert(0, reported)
    for candidate in candidates:
        if "engine_capabilities" in candidate:
            schema = candidate["engine_capabilities"]
            if "engines" in candidate or "stage_types" in candidate:
                try:
                    validate_worker_engine_capabilities(candidate)
                except (ValueError, TypeError):
                    return False
            if not advertisement_supports(schema, FORWARD_LAYERS):
                return False
            if "stage_types" in candidate and "layer_forward" not in candidate["stage_types"]:
                return False
            if info.get("serves_layer_segments") is False or info.get("forward_layers") is False:
                return False
            if isinstance(reported, str) and reported != FORWARD_LAYERS:
                return False
            if isinstance(reported, (list, tuple, set, frozenset)) and FORWARD_LAYERS not in reported:
                return False
            family = info.get("backend_id") or info.get("engine")
            kind = info.get("runtime_kind") or info.get("inference_path")
            if info.get("runtime_kind") and info.get("inference_path") and info["runtime_kind"] != info["inference_path"]:
                return False
            entries = ENGINE_REGISTRY.validate_advertisement(schema)
            return any(entry.supports(FORWARD_LAYERS)
                       and (not family or entry.family == family)
                       and (not kind or entry.runtime_kind == kind)
                       and ("native_abi" not in info or entry.native_abi == info["native_abi"])
                       for entry in entries)
    family = info.get("backend_id") or info.get("engine")
    kind = info.get("runtime_kind") or info.get("inference_path")
    if info.get("runtime_kind") and info.get("inference_path") and info["runtime_kind"] != info["inference_path"]:
        return False
    if not isinstance(family, str) or not isinstance(kind, str):
        return False
    # Legacy hints can name a concrete path; an explicit ABI must still match.
    spec = ENGINE_REGISTRY.get(family, kind)
    if spec is None or info.get("native_abi") != spec.native_abi:
        return False
    if not ENGINE_REGISTRY.supports(family, kind, FORWARD_LAYERS):
        return False
    if info.get("serves_layer_segments") is False:
        return False
    if isinstance(reported, str):
        return reported == FORWARD_LAYERS
    if isinstance(reported, (list, tuple, set, frozenset)):
        return FORWARD_LAYERS in reported
    return True
