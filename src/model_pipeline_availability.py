"""Metadata preflight for distributed selection, independent of model runtimes.

This is negative evidence only: missing/unreadable metadata still belongs to the
supply owner. Passing preflight is never a verified source or a native receipt.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, replace
from pathlib import Path

import model_config as mc
from pipeline_model_descriptor import PIPELINE_RUNTIME_MODEL_TYPES, canonical_pipeline_model_type
from relay_segment_info import read_gguf_layer_info, read_gguf_tensor_bytes, read_gguf_tokenizer_sha256

_MAX_JSON_BYTES = 8 * 1024 * 1024
_RUNTIME_UNSUPPORTED = "SUPPLY_MODEL_RUNTIME_UNSUPPORTED"
_FULL_SOURCE_REQUIRED = "SUPPLY_FULL_SOURCE_REQUIRED"


class ModelPipelineAvailabilityError(ValueError):
    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code, self.message = code, message


@dataclass(frozen=True)
class FormatAvailability:
    present: bool = False
    model_type: str = ""
    reason_code: str = ""
    reason: str = ""
    source: str = "registered"

    def public(self) -> dict:
        result = {
            "status": ("missing" if not self.present else "unavailable" if self.reason_code
                       else "supported" if self.model_type else "unknown"),
            "runtime_model_type": self.model_type,
            "reason_code": self.reason_code,
            "reason": self.reason,
        }
        if self.source == "provenance":
            result.update(source="provenance", content_verification="supply_owner_required")
        return result


def _json(path: Path) -> dict:
    try:
        if not 0 < path.stat().st_size <= _MAX_JSON_BYTES:
            return {}
        value = json.loads(path.read_text(encoding="utf-8"))
        return value if isinstance(value, dict) else {}
    except (OSError, UnicodeError, ValueError):
        return {}


def _positive_integer(value):
    return value if type(value) is int and value > 0 else None


def _partial_manifest(payload: dict, local_total: int | None = None) -> bool:
    span = payload.get("source_layer_range", payload.get("layer_range"))
    total = _positive_integer(payload.get("n_layer", payload.get("total_layers")))
    range_total = max(total or 0, local_total or 0)
    if (isinstance(span, list) and len(span) == 2
            and all(type(value) is int for value in span) and 0 <= span[0] < span[1]):
        if span[0] > 0 or span[1] < range_total:
            return True
    if total is not None and local_total is not None and total > local_total:
        return True
    return bool(_positive_integer(payload.get("trim_layers"))
                or _positive_integer(payload.get("first_local_layer_maps_to")))


def _blocked(code: str, model_type: str) -> FormatAvailability:
    reason = ("Distributed supply requires a complete source; this artifact contains a pre-cut or incomplete layer range."
              if code == _FULL_SOURCE_REQUIRED else
              f"The distributed pipeline runtime does not support model_type={model_type}.")
    return FormatAvailability(True, model_type, code, reason)


def _inspect_hf(model: mc.ModelConfig) -> FormatAvailability:
    if not mc.has_safetensors_files(model):
        return FormatAvailability()
    root = Path(mc.resolve_model_path(model.model_path))
    config = _json(root / "config.json")
    # Use the outer config type, exactly as inspect_pipeline_model does. Never
    # turn an unsupported wrapper into a supported text model by inference.
    raw_type = config.get("model_type")
    model_type = raw_type.strip().lower() if isinstance(raw_type, str) else ""
    decoder = config.get("text_config")
    if not isinstance(decoder, dict):
        decoder = config
    total = _positive_integer(decoder.get("num_hidden_layers"))
    for filename in ("supply.manifest.json", "model.manifest.json", ".qlh-model-asset.json"):
        payload = _json(root / filename)
        if payload.get("format", "safetensors") == "safetensors" and _partial_manifest(payload, total):
            return _blocked(_FULL_SOURCE_REQUIRED, model_type)
    if model_type and model_type not in PIPELINE_RUNTIME_MODEL_TYPES:
        return _blocked(_RUNTIME_UNSUPPORTED, model_type)
    return FormatAvailability(True, model_type)


def _inspect_gguf_path(path: Path) -> FormatAvailability:
    info = read_gguf_layer_info(path) or {}
    model_type = canonical_pipeline_model_type(info.get("architecture"))
    total = _positive_integer(info.get("n_layer"))
    for manifest in (path.with_suffix(".manifest.json"), Path(str(path) + ".manifest.json"),
                     path.parent / "supply.manifest.json"):
        payload = _json(manifest)
        if payload.get("format", "gguf") == "gguf" and _partial_manifest(payload, total):
            return _blocked(_FULL_SOURCE_REQUIRED, model_type)
    if model_type and model_type not in PIPELINE_RUNTIME_MODEL_TYPES:
        return _blocked(_RUNTIME_UNSUPPORTED, model_type)
    if total is not None:
        sizes = read_gguf_tensor_bytes(path)
        if sizes is not None:
            layers = {int(match[1]) for name, size in sizes.items()
                      if size > 0 and (match := re.match(r"^blk\.(\d+)\.", name))}
            if sum(0 <= index < total for index in layers) != total:
                return _blocked(_FULL_SOURCE_REQUIRED, model_type)
    return FormatAvailability(True, model_type)


def _provenance_target(model: mc.ModelConfig) -> Path | None:
    """Resolve an explicitly bound target, leaving full content proof to supply."""
    if not model.model_path:
        return None
    root = Path(mc.resolve_model_path(model.model_path)).resolve()
    proof = _json(root / "cross_format.provenance.json")
    if not proof:
        return None
    # Both helpers validate shared metadata schemas and canonical JSON digests.
    # _source_manifest_identity hashes only the tokenizer metadata, no weights;
    # cross_format_pipeline_identity would hash the full GGUF, so do not call it.
    from model_supply_assets import _proof_source
    from pipeline_model_descriptor import _source_manifest_identity

    try:
        target = proof.get("target")
        if not isinstance(target, dict):
            return None
        source = _proof_source({"conversion_proof": proof,
                                "source_model_sha256": target.get("artifact_sha256"),
                                "tokenizer_sha256": target.get("tokenizer_sha256")})
        if not source or source != _source_manifest_identity(root):
            return None
        relative = Path(target["path"])
        if relative.is_absolute() or relative.drive:
            return None
        declared = root / relative
        resolved = declared.resolve()
        resolved.relative_to(root.parent)
        for component in (declared, *declared.parents):
            if component == root.parent:
                break
            if component.is_symlink() or (hasattr(component, "is_junction") and component.is_junction()):
                return None
        if not resolved.is_file() or resolved.suffix.lower() != ".gguf":
            return None
        if read_gguf_tokenizer_sha256(resolved) != target["tokenizer_sha256"]:
            return None
        return resolved
    except (OSError, ValueError, RuntimeError):
        # Invalid/unknown proof remains an owner error, never a new LL option.
        return None


def inspect_model_pipeline_availability(model) -> dict[str, FormatAvailability]:
    # Unresolved entries carry no registered source evidence. Keep their
    # acquisition and validation with the supply owner.
    if not isinstance(model, mc.ModelConfig):
        return {}
    formats = {"pytorch": _inspect_hf(model), "llama_cpp": FormatAvailability()}
    if mc.has_gguf_file(model):
        formats["llama_cpp"] = _inspect_gguf_path(Path(mc.resolve_model_path(model.gguf_path)))
    declared_target = _provenance_target(model)
    if declared_target is not None:
        # The supply owner follows valid HF provenance before the registered
        # GGUF field. Mirror that source choice without asserting live hashes.
        formats["llama_cpp"] = replace(_inspect_gguf_path(declared_target), source="provenance")
    return formats


def _auto_engine(formats: dict[str, FormatAvailability]) -> str:
    # Match ModelSupplyCoordinator._sources, independent of GPU availability.
    return "pytorch" if formats.get("pytorch", FormatAvailability()).present else "llama_cpp"


def require_model_pipeline_available(model, engine: str = "auto") -> None:
    formats = inspect_model_pipeline_availability(model)
    selected = _auto_engine(formats) if engine == "auto" else engine
    source = formats.get(selected)
    if source is None or not source.reason_code:
        return
    alternatives = [name for name, value in formats.items()
                    if value.present and value.model_type and not value.reason_code]
    message = f"{selected}: {source.reason}"
    if alternatives:
        message += " Explicitly select the supported format: " + ", ".join(alternatives) + "."
    raise ModelPipelineAvailabilityError(source.reason_code, message)


def model_pipeline_catalog_payload(model, payload: dict) -> dict:
    """Overlay distributed availability without changing ordinary disk inventory."""
    formats = inspect_model_pipeline_availability(model)
    if not formats:
        return dict(payload)
    result = dict(payload)
    result["pipeline_engine_availability"] = {name: value.public() for name, value in formats.items()}
    auto = _auto_engine(formats)
    result["pipeline_auto_engine"] = auto
    auto_source = formats[auto]
    result["pipeline_auto_unavailable_reason"] = (
        f"{auto_source.reason_code}: {auto_source.reason}" if auto_source.reason_code else ""
    )
    blocked = {name: value for name, value in formats.items() if value.reason_code}
    engines = [name for name in payload.get("supported_engines", []) if name not in blocked]
    ll = formats["llama_cpp"]
    bound_target = ll.source == "provenance"
    if bound_target:
        result["has_gguf"] = True
        result["available_formats"] = list(payload.get("available_formats", []))
        if "gguf" not in result["available_formats"]:
            result["available_formats"].append("gguf")
        if ll.model_type and not ll.reason_code:
            if "llama_cpp" not in engines:
                engines.append("llama_cpp")
            result["is_available"] = True
    result["supported_engines"] = engines
    if blocked and not any(value.present and not value.reason_code for value in formats.values()):
        result["is_available"] = False
        result["unavailable_reason"] = "; ".join(
            f"{value.reason_code}: {name}: {value.reason}" for name, value in blocked.items()
        )
    if engines and result.get("preferred_engine") not in engines:
        result["preferred_engine"] = engines[0]
        result["default_quant_type"] = "Q4_K_M" if engines[0] == "llama_cpp" else "int4"
    return result
