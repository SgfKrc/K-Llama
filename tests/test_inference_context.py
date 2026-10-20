"""Cache estimates use bounded headers, device samples and per-sequence floors."""
from __future__ import annotations

import builtins
import copy
import json
import struct
import time
from datetime import datetime, timezone

import pytest

import inference_context as context


def _string(value):
    data = value.encode("utf-8")
    return struct.pack("<Q", len(data)) + data


def _value(value):
    if type(value) is str:
        return 8, _string(value)
    if type(value) is bool:
        return 7, struct.pack("<?", value)
    if type(value) is float:
        return 12, struct.pack("<d", value)
    if type(value) is list:
        values = [_value(item) for item in value]
        kind = values[0][0] if values else 4
        assert all(item[0] == kind for item in values)
        return 9, struct.pack("<IQ", kind, len(value)) + b"".join(item[1] for item in values)
    return 11, struct.pack("<q", value)


def _header(tmp_path, *, arch="qwen3", blocks=4, updates=None, omit=(), tensors=None):
    metadata = {"general.architecture": arch, f"{arch}.block_count": blocks,
                f"{arch}.embedding_length": 1024, f"{arch}.attention.head_count": 8,
                f"{arch}.attention.head_count_kv": 2,
                f"{arch}.attention.key_length": 128, f"{arch}.attention.value_length": 64}
    if arch in {"qwen35", "qwen35moe"}:
        metadata.update({f"{arch}.full_attention_interval": 4,
                         f"{arch}.ssm.conv_kernel": 4, f"{arch}.ssm.inner_size": 2048,
                         f"{arch}.ssm.state_size": 128, f"{arch}.ssm.group_count": 16,
                         f"{arch}.ssm.time_step_rank": 16})
    metadata.update(updates or {})
    for key in omit:
        metadata.pop(key, None)
    fields = []
    for key, value in metadata.items():
        kind, data = _value(value)
        fields.append(_string(key) + struct.pack("<I", kind) + data)
    tensors = tensors or []
    data = b"GGUF" + struct.pack("<IQQ", 3, len(tensors), len(fields)) + b"".join(fields)
    for index, name in enumerate(tensors):
        data += _string(name) + struct.pack("<IQIQ", 1, 1, 0, index * 32)
    if tensors:
        data += bytes((-len(data)) % 32) + bytes(len(tensors) * 32)
    path = tmp_path / "model.gguf"
    path.write_bytes(data)
    return path


def _manifest(path, start, end):
    path.with_suffix(".manifest.json").write_text(json.dumps({
        "source_layer_range": [start, end], "mode": "middle"}), encoding="utf-8")


def _gpu_profile(*, index=1, free=8 * 1024**3, **sample_updates):
    sample = {"device": f"cuda:{index}", "free_bytes": free,
              "source": "cudaMemGetInfo", "sampled_at": time.time()}
    sample.update(sample_updates)
    return {"ram": {"available_gb": 512}, "selected_gpu_index": index,
            "gpus": [{"index": index, "cuda_available": True, "memory_sample": sample}]}


def test_dense_uses_independent_key_value_dimensions_and_only_reduces_ceiling(tmp_path):
    path = _header(tmp_path)
    result = context.context_budget_from_gguf(path, budget_bytes=10**8, max_ctx=2081)
    assert result["reason"] == "ok"
    assert result["kv_bytes_per_token"] == 4 * 2 * (128 * 2 + 64 * 2)
    assert result["n_ctx"] == 2048
    assert result["n_ctx_per_seq"] == 2048
    assert result["recurrent_state_bytes"] == 0
    assert result["estimated_cache_bytes"] <= result["budget_bytes"]


def test_layer_arrays_are_summed_and_transposed_value_width_is_reserved(tmp_path):
    path = _header(tmp_path, updates={"qwen3.attention.head_count_kv": [1, 2, 3, 4],
                                    "qwen3.attention.key_length": [32, 64, 96, 128],
                                    "qwen3.attention.value_length": [128, 96, 64, 32]})
    result = context.context_budget_from_gguf(path, budget_bytes=10**8, max_ctx=2048)
    assert result["reason"] == "ok"
    assert result["kv_bytes_per_token"] == sum(h * (k + v) * 2 for h, k, v in
                                              zip([1, 2, 3, 4], [32, 64, 96, 128], [128, 96, 64, 32]))
    assert result["kv_allocation_bytes_per_token"] > result["kv_bytes_per_token"]
    assert "transposed_v_max_layer_width" in result["assumptions"]
    assert [row["value_length"] for row in result["kv_layer_bytes"]] == [128, 96, 64, 32]


def test_qwen35_32_trunk_layers_have_eight_kv_layers_and_nonzero_recurrent_states(tmp_path):
    path = _header(tmp_path, arch="qwen35", blocks=33,
                   updates={"qwen35.nextn_predict_layers": 1})
    _manifest(path, 0, 33)
    result = context.context_budget_from_gguf(path, budget_bytes=100 * 1024**2, max_ctx=4096)
    assert result["reason"] == "ok"
    assert result["main_layer_count"] == 32
    assert result["mtp_layer_count"] == 1
    assert result["kv_layer_count"] == 8
    assert result["recurrent_layer_count"] == 24
    assert result["kv_bytes_per_token"] == 8 * 2 * (128 + 64) * 2
    assert result["recurrent_state_bytes"] == 24 * 4 * (3 * (2048 + 2 * 16 * 128) + 128 * 2048)
    assert [row["layer"] for row in result["kv_layer_bytes"]] == list(range(3, 32, 4))
    assert "recurrent_f32_n_rs_seq_0" in result["assumptions"]


@pytest.mark.parametrize("arch", ["qwen35", "qwen35moe"])
@pytest.mark.parametrize("start,end", [(1, 4), (1, 5), (4, 7), (0, 5)])
def test_unaligned_crop_span_cannot_override_native_local_phase(tmp_path, arch, start, end):
    path = _header(tmp_path, arch=arch, blocks=end - start)
    _manifest(path, start, end)
    result = context.context_budget_from_gguf(path, budget_bytes=10**8, max_ctx=2048)
    assert result["n_ctx"] is None
    assert result["reason"] == "hybrid_source_span_not_interval_aligned"
    assert result["source_layer_range"] == [start, end]
    assert result["kv_bytes_per_token"] is None


@pytest.mark.parametrize("arch", ["qwen35", "qwen35moe"])
def test_aligned_crop_span_uses_native_local_phase(tmp_path, arch):
    path = _header(tmp_path, arch=arch, blocks=8)
    _manifest(path, 4, 12)
    result = context.context_budget_from_gguf(path, budget_bytes=10**8, max_ctx=2048)
    assert result["reason"] == "ok"
    assert [row["layer"] for row in result["kv_layer_bytes"]] == [3, 7]
    assert result["kv_layer_count"] == 2
    assert result["recurrent_layer_count"] == 6


def test_crop_without_phase_evidence_is_named_unknown(tmp_path):
    path = _header(tmp_path, arch="qwen35", blocks=3)
    result = context.context_budget_from_gguf(path, budget_bytes=10**8)
    assert result["n_ctx"] is None
    assert result["reason"] == "hybrid_attention_phase_unknown"


def test_tensor_directory_can_prove_native_compatible_unmarked_layout(tmp_path):
    path = _header(tmp_path, arch="qwen35", tensors=[
        "blk.0.ssm_conv1d.weight", "blk.1.ssm_conv1d.weight",
        "blk.2.ssm_conv1d.weight", "blk.3.attn_k.weight", "blk.3.attn_v.weight"])
    result = context.context_budget_from_gguf(path, budget_bytes=10**8, max_ctx=2048)
    assert result["reason"] == "ok"
    assert result["attention_layout_source"] == "tensor_directory"
    assert result["kv_layer_count"] == 1
    assert result["recurrent_state_bytes"] > 0


def test_explicit_recurrent_array_takes_precedence_over_interval(tmp_path):
    path = _header(tmp_path, arch="qwen35", updates={
        "qwen35.attention.recurrent_layers": [1, 0, 1, 0]})
    result = context.context_budget_from_gguf(path, budget_bytes=10**8, max_ctx=2048)
    assert result["reason"] == "ok"
    assert [row["layer"] for row in result["kv_layer_bytes"]] == [1, 3]
    assert result["attention_layout_source"] == "recurrent_layers"


def test_head_array_dimensions_follow_explicit_recurrent_layout(tmp_path):
    path = _header(tmp_path, arch="qwen35", updates={
        "qwen35.attention.head_count_kv": [0, 3, 0, 2],
        "qwen35.attention.recurrent_layers": [1, 0, 1, 0]})
    result = context.context_budget_from_gguf(path, budget_bytes=10**8, max_ctx=2048)
    assert result["reason"] == "ok"
    assert result["kv_bytes_per_token"] == (3 + 2) * (128 + 64) * 2
    assert result["recurrent_layer_count"] == 2


@pytest.mark.parametrize("omit", ["conv_kernel", "state_size", "time_step_rank"])
def test_unknown_recurrent_cost_never_becomes_free(tmp_path, omit):
    path = _header(tmp_path, arch="qwen35", omit=[f"qwen35.ssm.{omit}"], updates={
        "qwen35.attention.head_count_kv": [0, 2, 0, 2],
        "qwen35.attention.recurrent_layers": [1, 0, 1, 0]})
    result = context.context_budget_from_gguf(path, budget_bytes=10**8)
    assert result["n_ctx"] is None
    assert result["reason"] == "missing_or_invalid_recurrent_" + omit
    assert result["recurrent_state_bytes"] is None


def test_all_recurrent_segment_reports_resident_cost_and_unknown_context(tmp_path):
    path = _header(tmp_path, arch="qwen35", updates={
        "qwen35.attention.head_count_kv": [0, 0, 0, 0],
        "qwen35.attention.recurrent_layers": [1, 1, 1, 1]})
    result = context.context_budget_from_gguf(path, budget_bytes=10**8)
    assert result["reason"] == "recurrent_only_context_unbounded"
    assert result["n_ctx"] is None
    assert result["recurrent_state_bytes"] > 0
    assert result["kv_bytes_per_token"] == 0


def test_recurrent_cost_scales_with_sequences_and_is_subtracted_before_kv(tmp_path):
    path = _header(tmp_path, arch="qwen35", updates={
        "qwen35.attention.head_count_kv": [0, 2, 0, 2],
        "qwen35.attention.recurrent_layers": [1, 0, 1, 0]})
    single = context.context_budget_from_gguf(path, budget_bytes=20 * 1024**2)
    multi = context.context_budget_from_gguf(path, budget_bytes=20 * 1024**2, n_seq_max=3)
    assert multi["recurrent_state_bytes"] == single["recurrent_state_bytes"] * 3
    assert multi["n_ctx"] < single["n_ctx"]
    assert multi["n_ctx"] % (3 * 256) == 0
    assert multi["estimated_cache_bytes"] <= multi["budget_bytes"]
    tiny = context.context_budget_from_gguf(path, budget_bytes=1024)
    assert tiny["reason"] == "recurrent_state_exceeds_budget"


def test_min_ctx_is_per_sequence_and_alignment_never_expands_maximum(tmp_path):
    path = _header(tmp_path)
    result = context.context_budget_from_gguf(path, budget_bytes=10**8, n_seq_max=3,
                                              min_ctx=512, max_ctx=1700)
    assert result["n_ctx"] == 1536
    assert result["n_ctx_per_seq"] == 512
    insufficient = context.context_budget_from_gguf(path, budget_bytes=10**8,
                                                    n_seq_max=3, max_ctx=1535)
    assert insufficient["reason"] == "below_min_ctx_per_sequence"
    assert insufficient["n_ctx"] is None
    unaligned = context.context_budget_from_gguf(path, budget_bytes=10**8,
                                                 n_seq_max=3, min_ctx=513, max_ctx=2303)
    assert unaligned["n_ctx"] is None


def test_budget_just_below_per_sequence_minimum_does_not_round_up(tmp_path):
    path = _header(tmp_path)
    result = context.context_budget_from_gguf(path, budget_bytes=1536 * 1024,
                                              n_seq_max=4, min_ctx=512)
    assert result["n_ctx"] is None
    assert result["reason"] == "below_min_ctx_per_sequence"


def test_cpu_uses_available_ram_rather_than_total(tmp_path):
    path = _header(tmp_path)
    profile = {"ram": {"available_gb": 2, "total_gb": 512}}
    before = copy.deepcopy(profile)
    result = context.context_budget_from_gguf(path, profile=profile, max_ctx=2048)
    assert result["budget_bytes"] == 2 * 1024**3 // 8
    assert result["resource_domain"] == "cpu"
    assert result["budget_source"] == "cpu_available_ram"
    assert profile == before


def test_cuda_uses_only_fresh_selected_gpu_and_excludes_host_ram(tmp_path):
    path = _header(tmp_path)
    profile = _gpu_profile()
    profile["gpus"].insert(0, {"index": 0, "vram_free_gb": 64})
    result = context.context_budget_from_gguf(path, profile=profile, execution_device="cuda", max_ctx=2048)
    assert result["reason"] == "ok"
    assert result["resource_domain"] == "cuda:1"
    assert result["gpu_index"] == 1
    assert result["budget_bytes"] == 1024**3
    assert result["gpu_memory_source"] == "cudaMemGetInfo"


@pytest.mark.parametrize("updates,reason", [
    ({"sampled_at": 0}, "gpu_memory_sample_stale"),
    ({"sampled_at": time.time() + 10000}, "gpu_memory_sample_stale"),
    ({"sampled_at": float("nan")}, "invalid_gpu_sample_time"),
    ({"source": "total-minus-torch-allocated"}, "gpu_memory_sample_unreliable"),
    ({"device": "cuda:0"}, "gpu_memory_sample_device_mismatch"),
    ({"free_bytes": float("inf")}, "invalid_gpu_free_memory"),
    ({"free_bytes": True}, "invalid_gpu_free_memory"),
    ({"free_bytes": 10, "total_bytes": 5}, "invalid_gpu_free_memory"),
])
def test_bad_cuda_samples_return_named_unknown_without_ram_fallback(tmp_path, updates, reason):
    path = _header(tmp_path)
    result = context.context_budget_from_gguf(path, profile=_gpu_profile(**updates), execution_device="cuda:1")
    assert result["n_ctx"] is None
    assert result["budget_bytes"] is None
    assert result["reason"] == reason


def test_legacy_gpu_profile_and_wrong_gpu_are_unknown(tmp_path):
    path = _header(tmp_path)
    profile = {"ram": {"available_gb": 512}, "selected_gpu_index": 0,
               "gpu": {"index": 0, "cuda_available": True, "vram_free_gb": 8}}
    result = context.context_budget_from_gguf(path, profile=profile, execution_device="cuda")
    assert result["n_ctx"] is None
    assert result["reason"] == "gpu_memory_sample_unreliable"
    wrong = context.context_budget_from_gguf(path, profile=_gpu_profile(), execution_device="cuda:0")
    assert wrong["reason"] == "gpu_memory_sample_unavailable"
    missing = context.context_budget_from_gguf(path, profile={"ram": {"available_gb": 512}}, execution_device="cuda")
    assert missing["reason"] == "gpu_device_unknown"


def test_timezone_aware_gpu_sample_time_and_selected_gpu_shape(tmp_path):
    path = _header(tmp_path)
    profile = _gpu_profile(sampled_at=datetime.now(timezone.utc).isoformat())
    profile["gpu"] = profile.pop("gpus")[0]
    result = context.context_budget_from_gguf(path, profile=profile, execution_device="cuda:1", max_ctx=2048)
    assert result["reason"] == "ok"


def test_explicit_budget_is_already_in_declared_device_domain(tmp_path):
    path = _header(tmp_path)
    result = context.context_budget_from_gguf(path, execution_device="cuda:2", budget_bytes=10**8, max_ctx=2048)
    assert result["reason"] == "ok"
    assert result["resource_domain"] == "cuda:2"
    assert result["budget_source"] == "explicit"
    assert result["budget_bytes"] == 10**8


@pytest.mark.parametrize("k,v,expected", [
    ("q8_0", "q4_0", 4 * 2 * (128 // 32 * 34 + 64 // 32 * 18)),
    ("f16", "q8_0", 4 * 2 * (128 * 2 + 64 // 32 * 34)),
    ("q4_0", "f32", 4 * 2 * (128 // 32 * 18 + 64 * 4)),
])
def test_cache_quantization_counts_block_scales_for_k_and_v_separately(tmp_path, k, v, expected):
    path = _header(tmp_path)
    result = context.context_budget_from_gguf(path, budget_bytes=10**8,
                                              cache_type_k=k, cache_type_v=v, max_ctx=2048)
    assert result["reason"] == "ok"
    assert result["kv_bytes_per_token"] == expected
    assert result["estimated_cache_bytes"] <= result["budget_bytes"]


def test_quantization_rejects_unsupported_per_head_block_alignment(tmp_path):
    path = _header(tmp_path, updates={"qwen3.attention.key_length": 48})
    result = context.context_budget_from_gguf(path, budget_bytes=10**8, cache_type_k="q8_0")
    assert result["reason"] == "cache_head_dimension_not_block_aligned"
    assert result["n_ctx"] is None


def test_missing_value_dimension_uses_vendor_embedding_default_not_key_dimension(tmp_path):
    path = _header(tmp_path, omit=["qwen3.attention.value_length"],
                   updates={"qwen3.attention.key_length": 256})
    result = context.context_budget_from_gguf(path, budget_bytes=10**8, max_ctx=2048)
    assert result["reason"] == "ok"
    assert result["kv_bytes_per_token"] == 4 * 2 * (256 + 1024 // 8) * 2


@pytest.mark.parametrize("field,value,reason", [
    ("budget_bytes", float("nan"), "invalid_budget_bytes"),
    ("budget_bytes", float("inf"), "invalid_budget_bytes"),
    ("budget_bytes", True, "invalid_budget_bytes"),
    ("budget_bytes", -1, "invalid_budget_bytes"),
    ("n_seq_max", True, "invalid_n_seq_max"),
    ("n_seq_max", 0, "invalid_n_seq_max"),
    ("n_seq_max", float("inf"), "invalid_n_seq_max"),
    ("min_ctx", float("nan"), "invalid_min_ctx"),
    ("max_ctx", True, "invalid_max_ctx"),
    ("max_ctx", float("inf"), "invalid_max_ctx"),
    ("cache_type_k", [], "unsupported_cache_type_k"),
    ("cache_type_v", "unknown", "unsupported_cache_type_v"),
    ("execution_device", True, "invalid_execution_device"),
    ("execution_device", "mps", "unsupported_execution_device"),
    ("profile", [], "invalid_profile"),
])
def test_malformed_inputs_do_not_interrupt_loader(tmp_path, field, value, reason):
    path = _header(tmp_path)
    kwargs = {"budget_bytes": 10**8, field: value}
    result = context.context_budget_from_gguf(path, **kwargs)
    assert result["n_ctx"] is None
    assert result["reason"] == reason


@pytest.mark.parametrize("key,value,reason", [
    ("qwen3.block_count", True, "invalid_block_count"),
    ("qwen3.attention.head_count_kv", [2, 2], "invalid_head_count_kv_array_length"),
    ("qwen3.attention.head_count_kv", True, "invalid_head_count_kv"),
    ("qwen3.attention.key_length", float("inf"), "invalid_key_length"),
    ("qwen3.attention.value_length", [64, 64], "invalid_value_length_array"),
])
def test_malformed_metadata_is_named_unknown(tmp_path, key, value, reason):
    path = _header(tmp_path, updates={key: value})
    result = context.context_budget_from_gguf(path, budget_bytes=10**8)
    assert result["n_ctx"] is None
    assert result["reason"] == reason


@pytest.mark.parametrize("value", [True, float("nan"), float("inf"), -1, "8"])
def test_invalid_cpu_memory_samples_are_named_unknown(tmp_path, value):
    path = _header(tmp_path)
    result = context.context_budget_from_gguf(path, profile={"ram": {"available_gb": value}})
    assert result["reason"] == "invalid_cpu_available_memory"
    assert result["n_ctx"] is None


def test_malformed_and_conflicting_crop_manifests_fail_closed(tmp_path):
    path = _header(tmp_path, arch="qwen35")
    _manifest(path, 1, 4)
    result = context.context_budget_from_gguf(path, budget_bytes=10**8)
    assert result["reason"] == "source_layer_range_length_mismatch"
    _manifest(path, 1, 5)
    other = type(path)(str(path) + ".manifest.json")
    other.write_text(json.dumps({"source_layer_range": [0, 4]}), encoding="utf-8")
    assert context.context_budget_from_gguf(path, budget_bytes=10**8)["reason"] == "conflicting_artifact_manifests"
    other.write_text("[", encoding="utf-8")
    assert context.context_budget_from_gguf(path, budget_bytes=10**8)["reason"] == "invalid_artifact_manifest"


def test_only_headers_are_read_and_native_packages_are_not_imported(tmp_path, monkeypatch):
    path = _header(tmp_path, updates={"qwen3.attention.value_length": [32, 64, 96, 128]})
    original_import = builtins.__import__
    original_open = type(path).open
    reads = []

    def guarded_import(name, *args, **kwargs):
        assert name.split(".")[0] not in {"llama_cpp", "torch", "numpy", "gguf"}
        return original_import(name, *args, **kwargs)

    class BoundedFile:
        def __init__(self, handle):
            self.handle = handle
        def __enter__(self):
            return self
        def __exit__(self, *args):
            return self.handle.__exit__(*args)
        def read(self, count=-1):
            assert 0 <= count <= 64 * 1024 * 1024
            reads.append(count)
            return self.handle.read(count)
        def __getattr__(self, name):
            return getattr(self.handle, name)

    def guarded_open(target, *args, **kwargs):
        handle = original_open(target, *args, **kwargs)
        return BoundedFile(handle) if target == path else handle

    monkeypatch.setattr(builtins, "__import__", guarded_import)
    monkeypatch.setattr(type(path), "open", guarded_open)
    result = context.context_budget_from_gguf(path, budget_bytes=10**8, max_ctx=2048)
    assert result["reason"] == "ok"
    assert sum(reads) < 2 * path.stat().st_size


def test_missing_and_bad_model_headers_are_named_unknown(tmp_path):
    path = tmp_path / "missing.gguf"
    assert context.context_budget_from_gguf(path, budget_bytes=10**8)["reason"] == "invalid_gguf_header"
    path.write_bytes(b"broken")
    assert context.context_budget_from_gguf(path, budget_bytes=10**8)["reason"] == "invalid_gguf_header"


def test_unexpected_reader_failure_still_preserves_loader_fallback(monkeypatch):
    def fail(*args, **kwargs):
        raise RuntimeError("optional reader failure")
    monkeypatch.setattr(context, "read_gguf_header", fail)
    result = context.context_budget_from_gguf("missing.gguf", budget_bytes=10**8)
    assert result["n_ctx"] is None
    assert result["reason"] == "context_estimation_failed"
    assert result["error_type"] == "RuntimeError"


def test_external_gpu_sample_requires_matching_physical_uuid(tmp_path):
    path = _header(tmp_path)
    profile = _gpu_profile(source="nvidia-smi")
    sample = profile["gpus"][0]["memory_sample"]
    result = context.context_budget_from_gguf(path, profile=profile, execution_device="cuda")
    assert result["reason"] == "gpu_memory_sample_identity_unknown"
    expected_uuid = "12345678-1234-1234-1234-123456789abc"
    profile["gpus"][0]["device_uuid"] = expected_uuid
    sample["device_uuid"] = "GPU-" + expected_uuid.upper()
    result = context.context_budget_from_gguf(path, profile=profile, execution_device="cuda", max_ctx=2048)
    assert result["reason"] == "ok"
    assert result["gpu_device_uuid"] == expected_uuid
    sample["device_uuid"] = "87654321-1234-1234-1234-123456789abc"
    result = context.context_budget_from_gguf(path, profile=profile, execution_device="cuda")
    assert result["reason"] == "gpu_memory_sample_identity_unknown"


def test_recurrent_only_segment_does_not_need_unused_attention_dimensions(tmp_path):
    path = _header(tmp_path, arch="qwen35", updates={
        "qwen35.attention.head_count_kv": [0, 0, 0, 0],
        "qwen35.attention.recurrent_layers": [1, 1, 1, 1]}, omit=[
            "qwen35.attention.key_length", "qwen35.attention.value_length",
            "qwen35.embedding_length", "qwen35.attention.head_count"])
    result = context.context_budget_from_gguf(path, budget_bytes=10**8)
    assert result["reason"] == "recurrent_only_context_unbounded"
    assert result["recurrent_state_bytes"] > 0


def test_vendor_integer_and_sequence_limits_are_respected(tmp_path):
    path = _header(tmp_path)
    large = context.context_budget_from_gguf(path, budget_bytes=(1 << 63) - 1)
    assert large["reason"] == "ok"
    assert large["n_ctx"] < 1 << 32
    assert large["n_ctx"] % 256 == 0
    too_many = context.context_budget_from_gguf(path, budget_bytes=10**8, n_seq_max=257)
    assert too_many["reason"] == "invalid_n_seq_max"


def test_full_manifest_must_not_claim_a_cropped_header_as_whole(tmp_path):
    path = _header(tmp_path, arch="qwen35")
    path.with_suffix(".manifest.json").write_text(json.dumps({"mode": "full", "n_layer": 32}), encoding="utf-8")
    result = context.context_budget_from_gguf(path, budget_bytes=10**8)
    assert result["reason"] == "manifest_layer_count_mismatch"


def test_gpu_duplicate_indices_and_boolean_indices_are_not_accepted(tmp_path):
    path = _header(tmp_path)
    profile = _gpu_profile()
    profile["gpus"].append(copy.deepcopy(profile["gpus"][0]))
    result = context.context_budget_from_gguf(path, profile=profile, execution_device="cuda")
    assert result["reason"] == "gpu_memory_sample_unavailable"
    profile["selected_gpu_index"] = True
    result = context.context_budget_from_gguf(path, profile=profile, execution_device="cuda")
    assert result["reason"] == "invalid_gpu_index"


@pytest.mark.parametrize("arch", ["qwen35", "qwen35moe"])
@pytest.mark.parametrize("layout", [[1, 0, 1, 1], [True, False, True, True]])
def test_explicit_recurrent_array_overrides_unaligned_sidecar_and_interval(tmp_path, arch, layout):
    path = _header(tmp_path, arch=arch, updates={
        f"{arch}.attention.recurrent_layers": layout,
        f"{arch}.full_attention_interval": 0})
    _manifest(path, 1, 5)
    result = context.context_budget_from_gguf(path, budget_bytes=10**8, max_ctx=2048)
    assert result["reason"] == "ok"
    assert result["attention_layout_source"] == "recurrent_layers"
    assert [row["layer"] for row in result["kv_layer_bytes"]] == [1]
    assert result["recurrent_layer_count"] == 3
    assert result["recurrent_state_bytes"] > 0


@pytest.mark.parametrize("evidence", ["heads", "tensors"])
def test_other_layout_evidence_cannot_bypass_unaligned_sidecar(tmp_path, evidence):
    updates = {"qwen35.attention.head_count_kv": [0, 0, 0, 2]} if evidence == "heads" else {}
    tensors = ["blk.0.ssm_conv1d.weight", "blk.1.ssm_conv1d.weight",
               "blk.2.ssm_conv1d.weight", "blk.3.attn_k.weight", "blk.3.attn_v.weight"]
    path = _header(tmp_path, arch="qwen35", updates=updates,
                   tensors=tensors if evidence == "tensors" else None)
    _manifest(path, 1, 5)
    result = context.context_budget_from_gguf(path, budget_bytes=10**8)
    assert result["n_ctx"] is None
    assert result["reason"] == "hybrid_source_span_not_interval_aligned"


@pytest.mark.parametrize("evidence", ["heads", "tensors"])
def test_layout_evidence_disagreeing_with_native_is_not_applicable(tmp_path, evidence):
    updates = {"qwen35.attention.head_count_kv": [0, 2, 0, 2]} if evidence == "heads" else {}
    path = _header(tmp_path, arch="qwen35", blocks=4 if evidence == "heads" else 3,
                   updates=updates, tensors=None if evidence == "heads" else [
                       "blk.0.ssm_conv1d.weight", "blk.1.ssm_conv1d.weight",
                       "blk.2.attn_k.weight", "blk.2.attn_v.weight"])
    result = context.context_budget_from_gguf(path, budget_bytes=10**8)
    assert result["n_ctx"] is None
    assert result["reason"] == "hybrid_attention_layout_native_mismatch"


def test_compatible_head_array_confirms_native_local_phase(tmp_path):
    path = _header(tmp_path, arch="qwen35", updates={
        "qwen35.attention.head_count_kv": [0, 0, 0, 2]})
    result = context.context_budget_from_gguf(path, budget_bytes=10**8, max_ctx=2048)
    assert result["reason"] == "ok"
    assert result["attention_layout_source"] == "head_count_kv_array"
    assert [row["layer"] for row in result["kv_layer_bytes"]] == [3]
    assert result["recurrent_layer_count"] == 3


def test_missing_interval_uses_native_default_for_span_alignment(tmp_path):
    path = _header(tmp_path, arch="qwen35", omit=["qwen35.full_attention_interval"])
    _manifest(path, 1, 5)
    result = context.context_budget_from_gguf(path, budget_bytes=10**8)
    assert result["n_ctx"] is None
    assert result["reason"] == "hybrid_source_span_not_interval_aligned"
    _manifest(path, 4, 8)
    aligned = context.context_budget_from_gguf(path, budget_bytes=10**8, max_ctx=2048)
    assert aligned["reason"] == "ok"
    assert [row["layer"] for row in aligned["kv_layer_bytes"]] == [3]


@pytest.mark.parametrize("sequences", [1, 3])
@pytest.mark.parametrize("cache_type", ["f16", "q8_0", "q4_0"])
def test_ssm_default_state_is_one_f32_current_row_per_sequence(tmp_path, sequences, cache_type):
    # Vendor n_embd_r=(4-1)*(4096+2*16*128)=24576;
    # n_embd_s=128*4096=524288, each F32. n_rs_seq=0 still has one row.
    path = _header(tmp_path, arch="qwen35", blocks=32, updates={
        "qwen35.ssm.inner_size": 4096, "qwen35.ssm.time_step_rank": 32})
    _manifest(path, 0, 32)
    result = context.context_budget_from_gguf(path, budget_bytes=256*1024**2,
                                              n_seq_max=sequences, max_ctx=8192,
                                              cache_type_k=cache_type, cache_type_v=cache_type)
    assert result["reason"] == "ok"
    assert result["recurrent_layer_count"] == 24
    assert result["recurrent_state_bytes_per_seq"] == 52_690_944
    assert result["recurrent_state_bytes"] == 52_690_944 * sequences
    assert "recurrent_f32_n_rs_seq_0" in result["assumptions"]
    expected_cache = (result["n_ctx"] * result["kv_allocation_bytes_per_token"]
                      + result["allocation_reserve_bytes"] + 52_690_944 * sequences)
    assert result["estimated_cache_bytes"] == expected_cache
    assert expected_cache <= result["budget_bytes"]
