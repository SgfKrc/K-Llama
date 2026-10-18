"""Observe already materialized tensors; policy is verified, never inferred."""
from qlh_core.contracts.load import RuntimeLoadSpec


def observe_runtime(manager, assignment):
    spec = RuntimeLoadSpec.from_dict(assignment["load_spec"])
    identity = assignment["identity"]
    if tuple(manager.layer_range or ()) != tuple(identity["layer_range"]) or not manager.is_loaded:
        raise ValueError("deployment_local_range_mismatch")
    if spec.materialization == "gguf_segment":
        observation = manager.deployment_materialization_observation()
        for key in ("device", "dtype", "quant_type", "kv_device", "kv_dtype", "context_size", "batch_size", "sequence_count"):
            if observation.get(key) != getattr(spec, key):
                raise ValueError("deployment_local_policy_mismatch")
        if observation.get("artifact_sha256") != identity["artifact_sha256"]:
            raise ValueError("deployment_local_artifact_mismatch")
        materialized = observation["materialized_bytes"]
    else:
        parameters = list(manager.model.parameters())
        buffers = list(manager.model.buffers()) if callable(getattr(manager.model, "buffers", None)) else []
        tensors = parameters + buffers
        floating = [tensor for tensor in tensors if tensor.is_floating_point()]
        if (not floating or manager.quant_type != spec.quant_type
                or any(str(t.device) != spec.device or str(t.dtype).removeprefix("torch.") != spec.dtype for t in floating)):
            raise ValueError("deployment_local_policy_mismatch")
        materialized = sum(t.numel() * t.element_size() for t in tensors)
    receipt = {"load_spec": spec.to_dict(), "materialized_bytes": materialized,
               "memory_scope": "weight_tensors", "layer_range": identity["layer_range"],
               "artifact_sha256": identity["artifact_sha256"]}
    from qlh_core.contracts.deployment import require_runtime_receipt
    require_runtime_receipt(assignment, receipt)
    return receipt
