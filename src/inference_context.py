"""Header-only KV/context estimates; callers retain authority over runtime settings.

Public API::

    context_budget_from_gguf(model_path, *, profile=None,
        execution_device="cpu", budget_bytes=None, n_seq_max=1,
        cache_type_k=None, cache_type_v=None, min_ctx=512, max_ctx=None) -> dict

``n_ctx`` is the total slot count, divisible by ``n_seq_max * 256``;
``min_ctx`` is a per-sequence minimum and ``max_ctx`` is a total ceiling.
A named ``reason`` accompanies every result, including unknown estimates.
Explicit budgets belong to the execution device declared by the caller.
Otherwise CPU uses available RAM / 8 and CUDA uses free VRAM / 8.

CUDA profile contract: a matching entry in ``gpus`` (or selected ``gpu``)
contains ``memory_sample={device: 'cuda:N', free_bytes: int, source: str,
sampled_at: unix_seconds_or_timezone_aware_iso8601}``. Sources are listed in
GPU_MEMORY_SOURCES; samples older than GPU_SAMPLE_MAX_AGE_SECONDS are unknown.
NVML/nvidia-smi samples also require matching ``device_uuid`` values on the
GPU entry and sample; their row order does not establish CUDA device identity.
Legacy ``vram_free_gb`` alone is deliberately insufficient. Direct GPU fields
``memory_source``, ``memory_sampled_at``, ``vram_free_bytes``/``vram_free_gb``
are also accepted when the entry identifies its CUDA index.

Estimates cover cache buffers, not weights, scratch graphs or process memory.
Qwen35 recurrent states use vendor defaults F32 and n_rs_seq=0: each sequence
still has one current-state row, not zero rows. Rollback adds n_rs_seq snapshots
per sequence, multiplying the R/S payload by (1 + n_rs_seq). Callers using those
snapshots, tensor splits, mixed KV offload or MTP contexts must keep explicit
RuntimeLoadSpec settings authoritative.

Without an explicit recurrent_layers array, Qwen35 chooses layer kinds by LOCAL
layer index modulo full_attention_interval (native default 4). Sidecar source
ranges never change that phase. Unaligned trunk boundaries are inapplicable;
head arrays and tensor directories must agree with the native local layout.

Evidence: android/app/src/main/cpp/llama.cpp/src/{llama-model.cpp,
llama-kv-cache.cpp,llama-memory-recurrent.cpp,llama-hparams.cpp},
models/qwen35.cpp and ggml/src/ggml-common.h. No native model library is used.
"""
from __future__ import annotations

import json
import math
import re
import struct
import time
from collections.abc import Mapping
from datetime import datetime
from pathlib import Path

from gguf_metadata_contract import normalize_gguf_metadata, read_gguf_header

__all__ = ["context_budget_from_gguf"]

KV_BUDGET_MEMORY_RATIO = 1 / 8
CONTEXT_ALIGNMENT = 256
GPU_SAMPLE_MAX_AGE_SECONDS = 30
GPU_MEMORY_SOURCES = frozenset({
    "cudaMemGetInfo", "torch.cuda.mem_get_info", "nvmlDeviceGetMemoryInfo",
    "nvml", "nvidia-smi", "cuda.mem_get_info",
    "cuda.min_mem_get_info_nvidia_smi_uuid",
})
_CACHE_BLOCKS = {"f16": (1, 2), "bf16": (1, 2), "f32": (1, 4),
                 "q8_0": (32, 34), "q4_0": (32, 18)}
_DENSE_ARCHS = {"llama", "qwen2", "qwen2moe", "qwen3", "qwen3moe",
                "phi2", "phi3", "gpt2", "gptneox", "stablelm"}
_HYBRID_ARCHS = {"qwen35", "qwen35moe"}
_MAX_INTEGER = (1 << 63) - 1
_MAX_CONTEXT = (1 << 32) - 1  # llama_cparams::n_ctx is uint32_t
_MAX_SEQUENCES = 256  # vendor src/llama-cparams.h: LLAMA_MAX_SEQ
_MAX_HEADER_BYTES = 64 * 1024 * 1024


class _Unknown(ValueError):
    pass


def _integer(value, reason, *, minimum=1, maximum=_MAX_INTEGER):
    if type(value) is not int or not minimum <= value <= maximum:
        raise _Unknown(reason)
    return value


def _number(value, reason, *, minimum=0):
    if type(value) not in (int, float):
        raise _Unknown(reason)
    if not 0 <= value <= _MAX_INTEGER or not math.isfinite(value) or value < minimum:
        raise _Unknown(reason)
    return value


def _layer_values(value, count, reason, *, default=None):
    if value is None:
        value = default
    if type(value) is list:
        if len(value) != count:
            raise _Unknown(reason + "_array_length")
        return [_integer(item, reason, minimum=0) for item in value]
    return [_integer(value, reason, minimum=0)] * count


def _supplement_arrays(path, raw, arch, count):
    """Retain the two cache arrays not retained by the shared contract reader.

    Use its bounded header first. This second metadata-only pass occurs only
    when that reader explicitly skipped a relevant array; tokenizers and tensor
    payloads are sought past, never retained. Do not mutate its global suffixes.
    """
    wanted = {f"{arch}.attention.{suffix}" for suffix in
              ("value_length", "recurrent_layers")}
    wanted = {key for key in wanted if key in raw and raw[key] is None}
    if not wanted:
        return
    from relay_segment_info import _GGUF_SCALAR_TYPES

    with Path(path).open("rb") as handle:
        limit = min(Path(path).stat().st_size, _MAX_HEADER_BYTES)

        def read(length):
            if length < 0 or handle.tell() + length > limit:
                raise _Unknown("invalid_gguf_header")
            data = handle.read(length)
            if len(data) != length:
                raise _Unknown("invalid_gguf_header")
            return data

        def unpack(fmt):
            return struct.unpack(fmt, read(struct.calcsize(fmt)))[0]

        def skip(length):
            if length < 0 or handle.tell() + length > limit:
                raise _Unknown("invalid_gguf_header")
            handle.seek(length, 1)

        def string():
            return read(unpack("<Q")).decode("utf-8")

        def skip_value(kind, depth=0):
            if depth > 2:
                raise _Unknown("invalid_gguf_header")
            if kind == 8:
                skip(unpack("<Q"))
            elif kind == 9:
                subtype, length = unpack("<I"), unpack("<Q")
                if length > 10_000_000:
                    raise _Unknown("invalid_gguf_header")
                if subtype in _GGUF_SCALAR_TYPES:
                    skip(length * _GGUF_SCALAR_TYPES[subtype][1])
                else:
                    for _ in range(length):
                        skip_value(subtype, depth + 1)
            elif kind in _GGUF_SCALAR_TYPES:
                skip(_GGUF_SCALAR_TYPES[kind][1])
            else:
                raise _Unknown("invalid_gguf_header")

        read(16)  # magic, version, tensor count: validated by the shared reader
        metadata_count = unpack("<Q")
        for _ in range(metadata_count):
            key, kind = string(), unpack("<I")
            if key not in wanted:
                skip_value(kind)
                continue
            if kind != 9:
                raise _Unknown("invalid_" + key.rsplit(".", 1)[-1])
            subtype, length = unpack("<I"), unpack("<Q")
            if subtype not in _GGUF_SCALAR_TYPES or length != count:
                raise _Unknown("invalid_" + key.rsplit(".", 1)[-1] + "_array")
            raw[key] = [unpack(_GGUF_SCALAR_TYPES[subtype][0]) for _ in range(length)]


def _source_span(path, count, main_count):
    """Read bounded, artifact-specific sidecars; never hash/copy the model."""
    spans, full = [], False
    for target in dict.fromkeys((Path(str(path) + ".manifest.json"),
                                 Path(path).with_suffix(".manifest.json"))):
        if not target.exists():
            continue
        try:
            with target.open("rb") as handle:
                data = handle.read(1024 * 1024 + 1)
            if len(data) > 1024 * 1024:
                raise ValueError("oversize")
            manifest = json.loads(data)
            if not isinstance(manifest, dict):
                raise ValueError("not an object")
        except (OSError, ValueError) as exc:
            raise _Unknown("invalid_artifact_manifest") from exc
        span = manifest.get("source_layer_range")
        if span is not None:
            if type(span) is not list or len(span) != 2:
                raise _Unknown("invalid_source_layer_range")
            start = _integer(span[0], "invalid_source_layer_range", minimum=0)
            end = _integer(span[1], "invalid_source_layer_range")
            if end - start != count:
                raise _Unknown("source_layer_range_length_mismatch")
            spans.append((start, end))
        elif manifest.get("mode") == "full":
            if "n_layer" in manifest:
                total = _integer(manifest["n_layer"], "invalid_manifest_layer_count")
                if total not in (count, main_count):
                    raise _Unknown("manifest_layer_count_mismatch")
            full = True
    if len(set(spans)) > 1 or (full and spans and spans[0][0] != 0):
        raise _Unknown("conflicting_artifact_manifests")
    return list(spans[0]) if spans else ([0, count] if full else None)


def _attention_mask(path, raw, arch, count, main_count, heads, observations):
    if arch in _DENSE_ARCHS:
        observations["attention_layout_source"] = "head_count_kv"
        return [head > 0 for head in heads[:main_count]]
    recurrent = raw.get(f"{arch}.attention.recurrent_layers")
    if recurrent is not None:
        if (type(recurrent) is not list or len(recurrent) != count
                or any(type(item) not in (int, bool) or item not in (0, 1)
                       for item in recurrent)):
            raise _Unknown("invalid_recurrent_layers")
        observations["attention_layout_source"] = "recurrent_layers"
        return [not item for item in recurrent[:main_count]]
    interval = _integer(raw.get(f"{arch}.full_attention_interval", 4),
                        "invalid_full_attention_interval")
    native_mask = [(index + 1) % interval == 0 for index in range(main_count)]
    span = _source_span(path, count, main_count)
    observations["source_layer_range"] = span
    if span is not None:
        # qwen35[moe].cpp never reads a sidecar or applies a source-layer offset.
        # Match cut_layers._validate_cut: require both trunk boundaries aligned.
        # Trailing NextN/MTP blocks do not belong to the main attention pattern.
        main_end = span[1] - (count - main_count)
        if span[0] % interval or main_end % interval:
            raise _Unknown("hybrid_source_span_not_interval_aligned")
    if type(raw.get(f"{arch}.attention.head_count_kv")) is list and 0 in heads[:main_count]:
        if [head > 0 for head in heads[:main_count]] != native_mask:
            raise _Unknown("hybrid_attention_layout_native_mismatch")
        observations["attention_layout_source"] = "head_count_kv_array"
        return native_mask
    if span is not None or interval == 1:
        observations["attention_layout_source"] = "source_layer_range_and_interval"
        return native_mask
    # With no sidecar, tensor descriptors can confirm that an unmarked crop is
    # compatible with the local pattern. They cannot override native's phase.
    try:
        _, tensors = read_gguf_header(path, include_tensors=True)
    except (OSError, ValueError, struct.error) as exc:
        raise _Unknown("hybrid_attention_phase_unknown") from exc
    mask = []
    for index in range(main_count):
        prefix = f"blk.{index}."
        linear = prefix + "ssm_conv1d.weight" in tensors
        attention = (prefix + "attn_k.weight" in tensors
                     and prefix + "attn_v.weight" in tensors)
        if linear == attention:
            raise _Unknown("hybrid_attention_phase_unknown")
        mask.append(attention)
    if mask != native_mask:
        raise _Unknown("hybrid_attention_layout_native_mismatch")
    observations["attention_layout_source"] = "tensor_directory"
    return native_mask


def _timestamp(value):
    if type(value) in (int, float):
        return _number(value, "invalid_gpu_sample_time")
    if type(value) is str:
        try:
            parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
            if parsed.tzinfo is not None:
                return parsed.timestamp()
        except (ValueError, OverflowError, OSError):
            pass
    raise _Unknown("invalid_gpu_sample_time")


def _budget(profile, execution_device, budget_bytes, observations):
    if type(execution_device) is not str or len(execution_device) > 64:
        raise _Unknown("invalid_execution_device")
    device = execution_device.strip().lower()
    match = re.fullmatch(r"cuda(?::([0-9]+))?", device)
    if device != "cpu" and match is None:
        raise _Unknown("unsupported_execution_device")
    if profile is not None and not isinstance(profile, Mapping):
        raise _Unknown("invalid_profile")
    profile = profile or {}
    index = None
    if match:
        raw_index = int(match[1]) if match[1] is not None else profile.get("selected_gpu_index")
        if raw_index is not None:
            index = _integer(raw_index, "invalid_gpu_index", minimum=0)
        device = f"cuda:{index}" if index is not None else "cuda"
    observations.update(resource_domain=device, gpu_index=index)
    if budget_bytes is not None:
        value = int(_number(budget_bytes, "invalid_budget_bytes", minimum=1))
        observations.update(budget_bytes=value, budget_source="explicit")
        return value
    if device == "cpu":
        ram = profile.get("ram")
        if ram is None and not profile:
            try:
                import psutil
                available = psutil.virtual_memory().available
            except Exception as exc:
                raise _Unknown("cpu_available_memory_unavailable") from exc
        elif isinstance(ram, Mapping):
            if "available_bytes" in ram:
                available = _number(ram["available_bytes"], "invalid_cpu_available_memory")
            else:
                available = _number(ram.get("available_gb"), "invalid_cpu_available_memory") * 1024**3
        else:
            raise _Unknown("cpu_available_memory_unavailable")
        source = "cpu_available_ram"
    else:
        if index is None:
            raise _Unknown("gpu_device_unknown")
        gpus = profile.get("gpus", [])
        if not isinstance(gpus, list):
            raise _Unknown("invalid_gpu_profile")
        selected = profile.get("gpu")
        candidates = [gpu for gpu in gpus if isinstance(gpu, Mapping)
                      and type(gpu.get("cuda_index", gpu.get("index"))) is int
                      and gpu.get("cuda_index", gpu.get("index")) == index]
        if not candidates and isinstance(selected, Mapping):
            if (type(selected.get("cuda_index", selected.get("index"))) is int
                    and selected.get("cuda_index", selected.get("index")) == index):
                candidates = [selected]
        if len(candidates) != 1:
            raise _Unknown("gpu_memory_sample_unavailable")
        gpu = candidates[0]
        sample = gpu.get("memory_sample")
        if sample is None:
            sample = {"device": device, "source": gpu.get("memory_source"),
                      "sampled_at": gpu.get("memory_sampled_at")}
            if "vram_free_bytes" in gpu:
                sample["free_bytes"] = gpu["vram_free_bytes"]
            elif "vram_free_gb" in gpu:
                sample["free_bytes"] = _number(gpu["vram_free_gb"], "invalid_gpu_free_memory") * 1024**3
        if not isinstance(sample, Mapping):
            raise _Unknown("invalid_gpu_memory_sample")
        if sample.get("device") != device:
            raise _Unknown("gpu_memory_sample_device_mismatch")
        source = sample.get("source")
        if type(source) is not str or source not in GPU_MEMORY_SOURCES:
            raise _Unknown("gpu_memory_sample_unreliable")
        if source in {"nvmlDeviceGetMemoryInfo", "nvml", "nvidia-smi",
                      "cuda.min_mem_get_info_nvidia_smi_uuid"}:
            # External GPU row order need not equal CUDA_VISIBLE_DEVICES order.
            def canonical_uuid(value):
                if type(value) is not str:
                    return None
                value = value.lower().removeprefix("gpu-")
                return value if re.fullmatch(
                    r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}", value) else None
            expected_uuid = canonical_uuid(gpu.get("device_uuid", gpu.get("uuid")))
            sampled_uuid = canonical_uuid(sample.get("device_uuid"))
            if expected_uuid is None or sampled_uuid != expected_uuid:
                raise _Unknown("gpu_memory_sample_identity_unknown")
            observations["gpu_device_uuid"] = expected_uuid
        sampled_at = _timestamp(sample.get("sampled_at"))
        age = time.time() - sampled_at
        observations.update(gpu_sample_age_seconds=age, gpu_memory_source=source)
        if not -1 <= age <= GPU_SAMPLE_MAX_AGE_SECONDS:
            raise _Unknown("gpu_memory_sample_stale")
        available = _number(sample.get("free_bytes"), "invalid_gpu_free_memory")
        if "total_bytes" in sample:
            total = _number(sample["total_bytes"], "invalid_gpu_total_memory")
            if available > total:
                raise _Unknown("invalid_gpu_free_memory")
    available = _number(available, "invalid_available_memory")
    value = int(available) // 8
    if value <= 0:
        raise _Unknown("available_memory_exhausted")
    observations.update(available_bytes=int(available), budget_bytes=value, budget_source=source)
    return value


def _dimensions(raw, arch, count, key):
    value = raw.get(f"{arch}.attention.{key}_length")
    if value is None:
        # Vendor defaults K and V independently to embedding_length/head_count.
        embedding = _integer(raw.get(f"{arch}.embedding_length"), "missing_" + key + "_length")
        heads = _layer_values(raw.get(f"{arch}.attention.head_count"), count,
                              "missing_" + key + "_length")
        if any(head <= 0 or embedding % head for head in heads):
            raise _Unknown("missing_" + key + "_length")
        return [embedding // head for head in heads]
    return _layer_values(value, count, "invalid_" + key + "_length")


def _estimate(path, raw, metadata, cache_k, cache_v, observations):
    arch = metadata["architecture"]
    count = _integer(metadata.get("block_count"), "invalid_block_count", maximum=100_000)
    observations.update(architecture=arch, layer_count=count)
    if arch not in _DENSE_ARCHS | _HYBRID_ARCHS:
        raise _Unknown("unsupported_architecture")
    nextn = metadata.get("nextn_predict_layers")
    nextn = 0 if nextn is None else _integer(nextn, "invalid_nextn_predict_layers", minimum=0)
    if nextn >= count or (nextn and arch not in _HYBRID_ARCHS):
        raise _Unknown("unsupported_nextn_layout")
    main_count = count - nextn
    observations.update(main_layer_count=main_count, mtp_layer_count=nextn)
    _supplement_arrays(path, raw, arch, count)
    raw_heads = metadata.get("head_count_kv")
    if raw_heads is None:
        raw_heads = raw.get(f"{arch}.attention.head_count")
    heads = _layer_values(raw_heads, count, "invalid_head_count_kv")
    mask = _attention_mask(path, raw, arch, count, main_count, heads, observations)
    kv_count, recurrent_count = sum(mask), main_count - sum(mask)
    observations.update(kv_layer_count=kv_count, recurrent_layer_count=recurrent_count)
    if recurrent_count and arch not in _HYBRID_ARCHS:
        raise _Unknown("recurrent_state_layout_unknown")
    state_per_seq = 0
    if recurrent_count:
        values = {}
        for key in ("conv_kernel", "inner_size", "state_size", "group_count", "time_step_rank"):
            values[key] = _integer(raw.get(f"{arch}.ssm.{key}"), "missing_or_invalid_recurrent_" + key)
        inner, state, groups = (values[key] for key in ("inner_size", "state_size", "group_count"))
        if (inner != state * values["time_step_rank"]
                or values["time_step_rank"] % groups):
            raise _Unknown("recurrent_dimensions_inconsistent")
        conv_elements = (values["conv_kernel"] - 1) * (inner + 2 * groups * state)
        state_elements = state * inner
        # llama_model::create_memory uses F32 R/S and mem_size=n_seq_max.
        # llama_memory_recurrent allocates n_rows=mem_size*(1+n_rs_seq), while
        # llama_context_default_params/common's no-speculation default is 0.
        # Thus one current-state row per sequence remains. Align each whole
        # layer tensor after multiplying rows, rather than padding each row.
        sequences = observations["n_seq_max"]
        aligned = lambda value: (value + 255) // 256 * 256
        state_per_seq = recurrent_count * 4 * (conv_elements + state_elements)
        state_bytes = recurrent_count * (aligned(4 * conv_elements * sequences)
                                         + aligned(4 * state_elements * sequences))
        observations["assumptions"].append("recurrent_f32_n_rs_seq_0")
    else:
        state_bytes = 0
    observations.update(recurrent_state_bytes_per_seq=state_per_seq, recurrent_state_bytes=state_bytes)
    if not kv_count:
        observations.update(kv_bytes_per_token=0, kv_allocation_bytes_per_token=0)
        return 0, state_bytes
    keys = _dimensions(raw, arch, count, "key")
    values = _dimensions(raw, arch, count, "value")
    block_k, width_k = _CACHE_BLOCKS[cache_k]
    block_v, width_v = _CACHE_BLOCKS[cache_v]
    rows = []
    for index, uses_kv in enumerate(mask):
        if not uses_kv:
            continue
        if min(heads[index], keys[index], values[index]) <= 0:
            raise _Unknown("invalid_attention_dimensions")
        if keys[index] % block_k or values[index] % block_v:
            raise _Unknown("cache_head_dimension_not_block_aligned")
        row_k = heads[index] * keys[index] // block_k * width_k
        row_v = heads[index] * values[index] // block_v * width_v
        rows.append({"layer": index, "kv_heads": heads[index],
                     "key_length": keys[index], "value_length": values[index],
                     "key_bytes": row_k, "value_bytes": row_v})
    logical = sum(row["key_bytes"] + row["value_bytes"] for row in rows)
    # Without a Flash Attention argument, cover vendor's transposed V layout,
    # which expands V to the largest layer width. Quantized V requires FA.
    max_v = max((heads[i] * values[i] for i in range(count)), default=0)
    allocated_v = max_v // block_v * width_v
    allocated = sum(row["key_bytes"] + (row["value_bytes"] if block_v > 1 else allocated_v)
                    for row in rows)
    if allocated != logical:
        observations["assumptions"].append("transposed_v_max_layer_width")
    observations.update(kv_layer_bytes=rows, kv_bytes_per_token=logical,
                        kv_allocation_bytes_per_token=allocated,
                        allocation_reserve_bytes=2 * kv_count * 255)
    return allocated, state_bytes


def context_budget_from_gguf(
    model_path, *, profile=None, execution_device="cpu", budget_bytes=None,
    n_seq_max=1, cache_type_k=None, cache_type_v=None, min_ctx=512, max_ctx=None,
) -> dict:
    """Return cache observations and a bounded total ``n_ctx``, or named unknown.

    The caller must apply this only to automatic context defaults. This function
    does not mutate a profile, environment, explicit n_ctx, or RuntimeLoadSpec.
    See the module docstring for GPU sample fields and accounting assumptions.
    """
    result = dict(schema_version=1, n_ctx=None, reason="context_estimation_failed",
                  resource_domain=None, gpu_index=None, budget_bytes=None,
                  budget_source=None, available_bytes=None, architecture=None,
                  layer_count=None, main_layer_count=None, mtp_layer_count=None,
                  kv_layer_count=None, recurrent_layer_count=None,
                  attention_layout_source=None, source_layer_range=None,
                  kv_bytes_per_token=None, kv_allocation_bytes_per_token=None,
                  kv_layer_bytes=[], recurrent_state_bytes=None,
                  recurrent_state_bytes_per_seq=None, allocation_reserve_bytes=0,
                  n_seq_max=None, min_ctx_per_seq=None, max_ctx=None,
                  context_alignment=CONTEXT_ALIGNMENT, total_alignment=None,
                  n_ctx_per_seq=None, estimated_cache_bytes=None, assumptions=[])
    try:
        sequences = _integer(n_seq_max, "invalid_n_seq_max", maximum=_MAX_SEQUENCES)
        minimum = _integer(min_ctx, "invalid_min_ctx")
        ceiling = None if max_ctx is None else _integer(max_ctx, "invalid_max_ctx")
        result.update(n_seq_max=sequences, min_ctx_per_seq=minimum, max_ctx=ceiling,
                      total_alignment=sequences * CONTEXT_ALIGNMENT)
        types = []
        for value, label in ((cache_type_k, "k"), (cache_type_v, "v")):
            value = "f16" if value is None else value
            if type(value) is not str or value not in _CACHE_BLOCKS:
                raise _Unknown("unsupported_cache_type_" + label)
            types.append(value)
        result.update(cache_type_k=types[0], cache_type_v=types[1])
        budget = _budget(profile, execution_device, budget_bytes, result)
        try:
            raw, _ = read_gguf_header(model_path, include_tensors=False)
            metadata = normalize_gguf_metadata(raw)
        except (OSError, ValueError, TypeError, struct.error) as exc:
            raise _Unknown("invalid_gguf_header") from exc
        per_token, fixed = _estimate(model_path, raw, metadata, *types, result)
        fixed += result["allocation_reserve_bytes"]
        if budget <= fixed:
            raise _Unknown("recurrent_state_exceeds_budget" if result["recurrent_state_bytes"]
                           else "cache_reserve_exceeds_budget")
        if per_token <= 0:
            raise _Unknown("recurrent_only_context_unbounded")
        slots = min((budget - fixed) // per_token, _MAX_CONTEXT)
        if ceiling is not None:
            slots = min(slots, ceiling)
        slots = slots // result["total_alignment"] * result["total_alignment"]
        if slots // sequences < minimum:
            raise _Unknown("below_min_ctx_per_sequence")
        result.update(n_ctx=int(slots), n_ctx_per_seq=int(slots // sequences),
                      estimated_cache_bytes=int(slots * per_token + fixed), reason="ok")
    except _Unknown as exc:
        result["reason"] = str(exc)
    except Exception as exc:  # Optional estimates must never interrupt a loader.
        result.update(reason="context_estimation_failed", error_type=type(exc).__name__)
    return result
