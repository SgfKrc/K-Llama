"""Strict identities, frozen load policies and actual runtime receipts.

Legacy field sets remain decodable. New deployments negotiate version 1 and
bind the complete policy digest into prepare, commit and stage identities.
"""
from __future__ import annotations
from copy import deepcopy
import re

from qlh_core.contracts.load import RuntimeLoadSpec

_ID = re.compile(r"[A-Za-z0-9_.:/-]{1,256}")
_SHA = re.compile(r"[0-9a-f]{64}")
IDENTITY_FIELDS = frozenset({
    "operation_id", "config_id", "generation", "assignment_id", "node_id",
    "model_id", "model_sha256", "stage_model_identity", "artifact_sha256",
    "manifest_sha256", "source_model_id", "source_model_sha256",
    "tokenizer_sha256", "layer_range", "total_layers", "hidden_size", "segment_mode",
})
OPTIONAL_IDENTITY_FIELDS = frozenset({"load_spec_digest"})
STAGE_IDENTITY_FIELDS = frozenset({"operation_id", "config_id", "generation", "assignment_id", "artifact_sha256"})


class WorkerPrepareError(ValueError):
    def __init__(self, code: str):
        self.reason_code = code
        super().__init__(code)


def validate_stage_identity(value):
    if (not isinstance(value, dict) or not STAGE_IDENTITY_FIELDS.issubset(value)
            or set(value) - STAGE_IDENTITY_FIELDS - OPTIONAL_IDENTITY_FIELDS):
        raise WorkerPrepareError("invalid_prepare_identity")
    for key in ("operation_id", "config_id", "assignment_id"):
        if not isinstance(value[key], str) or not _ID.fullmatch(value[key]):
            raise WorkerPrepareError("invalid_prepare_identity")
    if type(value["generation"]) is not int or value["generation"] <= 0:
        raise WorkerPrepareError("invalid_prepare_identity")
    if not isinstance(value["artifact_sha256"], str) or not _SHA.fullmatch(value["artifact_sha256"]):
        raise WorkerPrepareError("invalid_prepare_identity")
    if "load_spec_digest" in value and not _SHA.fullmatch(str(value["load_spec_digest"])):
        raise WorkerPrepareError("invalid_prepare_identity")
    return deepcopy(value)


def stage_identity(identity):
    return validate_stage_identity({key: identity[key] for key in STAGE_IDENTITY_FIELDS | OPTIONAL_IDENTITY_FIELDS if key in identity})


def validate_identity(value):
    if (not isinstance(value, dict) or not IDENTITY_FIELDS.issubset(value)
            or set(value) - IDENTITY_FIELDS - OPTIONAL_IDENTITY_FIELDS):
        raise WorkerPrepareError("invalid_prepare_identity")
    stage_identity(value)
    if "load_spec_digest" in value and (
            not isinstance(value["load_spec_digest"], str)
            or not _SHA.fullmatch(value["load_spec_digest"])):
        raise WorkerPrepareError("invalid_prepare_identity")
    for key in ("node_id", "model_id", "source_model_id"):
        if not isinstance(value[key], str) or not _ID.fullmatch(value[key]):
            raise WorkerPrepareError("invalid_prepare_identity")
    for key in ("model_sha256", "source_model_sha256", "tokenizer_sha256"):
        if not isinstance(value[key], str) or not _SHA.fullmatch(value[key]):
            raise WorkerPrepareError("invalid_prepare_identity")
    if value["manifest_sha256"] != "" and (not isinstance(value["manifest_sha256"], str) or not _SHA.fullmatch(value["manifest_sha256"])):
        raise WorkerPrepareError("invalid_prepare_identity")
    for key in ("total_layers", "hidden_size"):
        if type(value[key]) is not int or value[key] <= 0:
            raise WorkerPrepareError("invalid_prepare_identity")
    span = value["layer_range"]
    if (not isinstance(span, list) or len(span) != 2 or any(type(x) is not int for x in span)
            or not 0 <= span[0] < span[1] <= value["total_layers"]
            or span[1] - span[0] == value["total_layers"]):
        raise WorkerPrepareError("invalid_prepare_layer_range")
    mode = "head" if span[0] == 0 else "tail" if span[1] == value["total_layers"] else "middle"
    if value["segment_mode"] != mode:
        raise WorkerPrepareError("invalid_prepare_segment_mode")
    model = value["stage_model_identity"]
    if (not isinstance(model, dict) or set(model) != {"model_id", "engine", "format", "revision", "sha256"}
            or model["sha256"] != value["artifact_sha256"]
            or (model["engine"], model["format"]) not in {("llama_cpp", "gguf"), ("pytorch", "safetensors")}
            or any(not isinstance(model[key], str) or not _ID.fullmatch(model[key]) for key in ("model_id", "revision"))):
        raise WorkerPrepareError("invalid_prepare_model_identity")
    return deepcopy(value)


def validate_load_spec(value, *, identity=None, context_size=None):
    """Validate the opt-in PC extension; never derive policy from local config."""
    try:
        spec = value if isinstance(value, RuntimeLoadSpec) else RuntimeLoadSpec.from_dict(value)
    except (TypeError, ValueError) as exc:
        raise WorkerPrepareError("invalid_prepare_load_spec") from exc
    if identity is not None and identity.get("load_spec_digest") != spec.digest:
        raise WorkerPrepareError("prepare_load_spec_digest_mismatch")
    if context_size is not None and spec.context_size != context_size:
        raise WorkerPrepareError("prepare_load_spec_context_mismatch")
    return spec


def validate_command(value):
    if not isinstance(value, dict):
        raise WorkerPrepareError("invalid_prepare_command")
    action = value.get("action")
    if action not in {"prepare", "commit", "abort", "finalize", "release"}:
        raise WorkerPrepareError("invalid_prepare_action")
    expected = {"action", "identity"}
    if action == "prepare":
        expected |= {"context_size", "middle_channel"}
        if type(value.get("context_size")) is not int or not 1 <= value["context_size"] <= 131072 or value.get("middle_channel") != "keep_head_layer_out":
            raise WorkerPrepareError("invalid_prepare_options")
    if action == "prepare" and "load_spec" in value:
        expected.add("load_spec")
    if action in {"abort", "finalize"} or "cleanup_id" in value:
        expected.add("cleanup_id")
        if not isinstance(value.get("cleanup_id"), str) or not _ID.fullmatch(value["cleanup_id"]):
            raise WorkerPrepareError("invalid_prepare_cleanup")
    if set(value) != expected:
        raise WorkerPrepareError("invalid_prepare_command")
    validate_identity(value.get("identity"))
    if action == "prepare":
        if "load_spec" in value:
            # The wire boundary accepts only the complete JSON document.
            if not isinstance(value["load_spec"], dict):
                raise WorkerPrepareError("invalid_prepare_load_spec")
            validate_load_spec(value["load_spec"], identity=value["identity"],
                               context_size=value["context_size"])
        elif "load_spec_digest" in value["identity"]:
            raise WorkerPrepareError("prepare_load_spec_missing")
    return deepcopy(value)


def validate_ack(value):
    if not isinstance(value, dict) or value.get("action") != "ack":
        raise WorkerPrepareError("invalid_prepare_ack")
    phase = value.get("phase")
    success = {"prepare": "prepared", "commit": "ready", "abort": "released", "finalize": "finalized", "release": "released"}
    if phase not in success or value.get("status") not in {success[phase], "error"} or type(value.get("loaded")) is not bool:
        raise WorkerPrepareError("invalid_prepare_ack")
    expected = {"action", "phase", "identity", "status", "loaded"}
    if phase in {"abort", "finalize"} or "cleanup_id" in value:
        expected.add("cleanup_id")
        if not isinstance(value.get("cleanup_id"), str) or not _ID.fullmatch(value["cleanup_id"]):
            raise WorkerPrepareError("invalid_prepare_ack")
    if value.get("status") == "error":
        expected.add("error")
        if not isinstance(value.get("error"), str) or not _ID.fullmatch(value["error"]):
            raise WorkerPrepareError("invalid_prepare_ack")
    if "runtime_receipt" in value:
        expected.add("runtime_receipt")
        validate_runtime_receipt(value["runtime_receipt"], identity=value.get("identity"))
    if set(value) != expected:
        raise WorkerPrepareError("invalid_prepare_ack")
    if value["status"] != "error" and phase in {"prepare", "commit"} and not value["loaded"]:
        raise WorkerPrepareError("prepare_runtime_not_loaded")
    # Finalize may replay after a disconnected runtime has already retired.
    # It proves terminal cleanup only; it never grants execution readiness.
    # Abort may restore a previous runtime; release must actually unload it.
    if value["status"] != "error" and phase == "release" and value["loaded"]:
        raise WorkerPrepareError("prepare_runtime_not_released")
    validate_identity(value.get("identity"))
    return deepcopy(value)


def validate_runtime_receipt(value, *, identity):
    fields = {"load_spec", "materialized_bytes", "memory_scope", "layer_range", "artifact_sha256"}
    if (not isinstance(value, dict) or not fields.issubset(value)
            or value.get("memory_scope") != "weight_tensors"
            or type(value.get("materialized_bytes")) is not int or value["materialized_bytes"] <= 0
            or not isinstance(identity, dict) or value.get("layer_range") != identity.get("layer_range")
            or value.get("artifact_sha256") != identity.get("artifact_sha256")):
        raise WorkerPrepareError("prepare_runtime_receipt_invalid")
    spec = validate_load_spec(value["load_spec"], identity=identity)
    if spec.materialization == "safetensors_layer_offload":
        domains = {"device_weight_bytes", "host_source_bytes", "host_conversion_bytes"}
        fields |= domains
        if (any(type(value.get(name)) is not int or value[name] < 0 for name in domains)
                or value["device_weight_bytes"] <= 0
                or value["host_source_bytes"] != value["materialized_bytes"]):
            raise WorkerPrepareError("prepare_runtime_receipt_invalid")
    if set(value) != fields:
        raise WorkerPrepareError("prepare_runtime_receipt_invalid")
    return deepcopy(value)
