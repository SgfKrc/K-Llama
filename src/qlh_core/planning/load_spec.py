"""Resolve one concrete materialization policy from a frozen assignment."""
from qlh_core.contracts.load import RuntimeLoadSpec


def assignment_load_spec(assignment, *, artifact_format, context_size=4096):
    if "load_spec" in assignment:
        return RuntimeLoadSpec.from_dict(assignment["load_spec"])
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
    dtype = "artifact" if native else "float32" if device == "cpu" else "float16"
    quant = "artifact" if native else "fp32" if dtype == "float32" else "fp16"
    return RuntimeLoadSpec(device="cpu" if native else device, dtype=dtype, quant_type=quant,
        context_size=context_size, kv_device="cpu" if native else device,
        kv_dtype="float16" if native else dtype, kv_policy="preallocated" if native else "dynamic",
        components=tuple(components), materialization="gguf_segment" if native else "safetensors_key_filtered",
        batch_size=min(context_size, 8192) if native else min(context_size, 512), sequence_count=1,
        kv_cache_bytes=int(assignment.get("kv_cache_bytes", 0)))
