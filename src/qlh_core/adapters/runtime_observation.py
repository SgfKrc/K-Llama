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
    elif spec.materialization == "safetensors_layer_offload":
        observation = manager.layer_offload_observation()
        if (not spec.device.startswith("cuda:")
                or manager.quant_type != spec.quant_type
                or observation.get("execution_device") != spec.device
                or observation.get("execution_dtype") != spec.dtype):
            raise ValueError("deployment_local_policy_mismatch")
        materialized = observation["resident_source_bytes"]
    else:
        parameters = list(manager.model.parameters())
        buffers = list(manager.model.buffers()) if callable(getattr(manager.model, "buffers", None)) else []
        tensors = parameters + buffers
        # Non-persistent RoPE buffers are derived by the model, not checkpoint
        # weights. Transformers keeps these in FP32 even for FP16 parameters.
        # Persistent buffers still obey the frozen weight dtype; every buffer
        # must live on the assigned device and remains in byte accounting.
        derived = set()
        modules = getattr(manager.model, "modules", None)
        if callable(modules):
            for module in modules():
                names = getattr(module, "_non_persistent_buffers_set", ())
                for name, tensor in module.named_buffers(recurse=False):
                    if name in names:
                        derived.add(id(tensor))
        floating = [tensor for tensor in tensors if tensor.is_floating_point()]
        if (not floating or manager.quant_type != spec.quant_type
                or any(str(t.device) != spec.device for t in tensors)
                or any(str(t.dtype).removeprefix("torch.") not in (
                    {spec.dtype, "float32"} if id(t) in derived else {spec.dtype}) for t in floating)):
            raise ValueError("deployment_local_policy_mismatch")
        materialized = sum(t.numel() * t.element_size() for t in tensors)
    receipt = {"load_spec": spec.to_dict(), "materialized_bytes": materialized,
               "memory_scope": "weight_tensors", "layer_range": identity["layer_range"],
               "artifact_sha256": identity["artifact_sha256"]}
    if spec.materialization == "safetensors_layer_offload":
        receipt.update(device_weight_bytes=observation["device_weight_bytes"],
                       host_source_bytes=observation["resident_source_bytes"],
                       host_conversion_bytes=observation["host_conversion_bytes"])
    from qlh_core.contracts.deployment import require_runtime_receipt
    require_runtime_receipt(assignment, receipt)
    return receipt
