"""Strict, transport independent identity for remote segment preparation."""
from __future__ import annotations
from copy import deepcopy
import re

_ID = re.compile(r"[A-Za-z0-9_.:/-]{1,256}")
_SHA = re.compile(r"[0-9a-f]{64}")
IDENTITY_FIELDS = frozenset({
    "operation_id", "config_id", "generation", "assignment_id", "node_id",
    "model_id", "model_sha256", "stage_model_identity", "artifact_sha256",
    "manifest_sha256", "source_model_id", "source_model_sha256",
    "tokenizer_sha256", "layer_range", "total_layers", "hidden_size", "segment_mode",
})
STAGE_IDENTITY_FIELDS = frozenset({"operation_id", "config_id", "generation", "assignment_id", "artifact_sha256"})


class WorkerPrepareError(ValueError):
    def __init__(self, code: str):
        self.reason_code = code
        super().__init__(code)


def validate_stage_identity(value):
    if not isinstance(value, dict) or set(value) != STAGE_IDENTITY_FIELDS:
        raise WorkerPrepareError("invalid_prepare_identity")
    for key in ("operation_id", "config_id", "assignment_id"):
        if not isinstance(value[key], str) or not _ID.fullmatch(value[key]):
            raise WorkerPrepareError("invalid_prepare_identity")
    if type(value["generation"]) is not int or value["generation"] <= 0:
        raise WorkerPrepareError("invalid_prepare_identity")
    if not isinstance(value["artifact_sha256"], str) or not _SHA.fullmatch(value["artifact_sha256"]):
        raise WorkerPrepareError("invalid_prepare_identity")
    return deepcopy(value)


def stage_identity(identity):
    return validate_stage_identity({key: identity[key] for key in STAGE_IDENTITY_FIELDS})


def validate_identity(value):
    if not isinstance(value, dict) or set(value) != IDENTITY_FIELDS:
        raise WorkerPrepareError("invalid_prepare_identity")
    stage_identity(value)
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
    if action in {"abort", "finalize"} or "cleanup_id" in value:
        expected.add("cleanup_id")
        if not isinstance(value.get("cleanup_id"), str) or not _ID.fullmatch(value["cleanup_id"]):
            raise WorkerPrepareError("invalid_prepare_cleanup")
    if set(value) != expected:
        raise WorkerPrepareError("invalid_prepare_command")
    validate_identity(value.get("identity"))
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
