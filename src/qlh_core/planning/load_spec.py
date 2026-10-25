"""Resolve one concrete materialization policy from a frozen assignment."""
from qlh_core.contracts.load import RuntimeLoadSpec


def assignment_load_spec(assignment, *, artifact_format, context_size=4096):
    quoted_dtype = None
    if artifact_format == "safetensors":
        if "runtime_target_dtype" in assignment:
            quoted_dtype = assignment["runtime_target_dtype"]
            if not isinstance(quoted_dtype, str) or quoted_dtype not in {"float16", "float32"}:
                raise ValueError("deployment_runtime_dtype_unsupported")
        if "runtime_target_element_size" in assignment:
            size = assignment["runtime_target_element_size"]
            if type(size) is not int or size not in {2, 4}:
                raise ValueError("deployment_runtime_dtype_unsupported")
            element_dtype = {2: "float16", 4: "float32"}[size]
            if quoted_dtype is not None and quoted_dtype != element_dtype:
                raise ValueError("deployment_runtime_dtype_mismatch")
            quoted_dtype = element_dtype
    if "load_spec" in assignment:
        spec = RuntimeLoadSpec.from_dict(assignment["load_spec"])
        if quoted_dtype is not None and spec.dtype != quoted_dtype:
            raise ValueError("deployment_runtime_dtype_mismatch")
        return spec
    components = {"layers"}
    if artifact_format == "safetensors":
        components.add("final_norm")
    if assignment.get("has_embedding"):
        components.add("embedding")
    if assignment.get("has_lm_head"):
        components.update(("final_norm", "lm_head"))
    device = assignment.get("execution_device", "cpu")
    if device == "cuda":
        device = "cuda:0"
    if device not in {"cpu", "mps"} and not device.startswith("cuda:"):
        device = "cpu"
    native = artifact_format == "gguf"
    dtype = "artifact" if native else quoted_dtype or ("float32" if device == "cpu" else "float16")
    quant = "artifact" if native else "fp32" if dtype == "float32" else "fp16"
    materialization = assignment.get("runtime_materialization", "safetensors_key_filtered")
    if native:
        materialization = "gguf_segment"
    elif materialization not in {"safetensors_key_filtered", "safetensors_layer_offload"}:
        raise ValueError("deployment_materialization_unsupported")
    if materialization == "safetensors_layer_offload" and not device.startswith("cuda:"):
        raise ValueError("deployment_layer_offload_device_unsupported")
    return RuntimeLoadSpec(device="cpu" if native else device, dtype=dtype, quant_type=quant,
        context_size=context_size, kv_device="cpu" if native else device,
        kv_dtype="float16" if native else dtype, kv_policy="preallocated" if native else "dynamic",
        components=tuple(components), materialization=materialization,
        batch_size=min(context_size, 8192) if native else min(context_size, 512), sequence_count=1,
        kv_cache_bytes=int(assignment.get("kv_cache_bytes", 0)))
