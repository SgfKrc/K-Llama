"""Real CPU tensor execution of the layer transfer/restore mechanism.

CPU emulation checks mmap owners, dtype conversion, prefill/decode and cleanup;
the CUDA product gate is responsible for actual device/VRAM measurements.
"""
from copy import deepcopy
import weakref

import pytest
import torch
from torch import nn
from transformers import Qwen2Config, Qwen2ForCausalLM

import model_module
from model_module import ModelManager, _LayerOffloadRuntime
from model_supply_runtime_capacity import project_runtime_descriptor, runtime_range_capacity
from pipeline_model_descriptor import inspect_pipeline_model
from qlh_core.contracts.load import RuntimeLoadSpec
from tests.test_safetensors_supply_worker import tiny_supply


def policy(**changes):
    values = dict(device="cuda:0", dtype="float32", quant_type="fp32", context_size=64,
                  kv_device="cuda:0", kv_dtype="float32", kv_policy="dynamic",
                  components=("layers", "final_norm", "embedding"),
                  materialization="safetensors_layer_offload", batch_size=32)
    values.update(changes)
    return RuntimeLoadSpec(**values)


@pytest.mark.parametrize("fail", [False, True])
def test_unit_restores_original_cpu_storage_and_aliases_even_on_forward_error(fail):
    class Unit(nn.Module):
        def __init__(self):
            super().__init__()
            self.left = nn.Linear(3, 2, bias=False, dtype=torch.bfloat16)
            self.right = nn.Linear(3, 2, bias=False, dtype=torch.bfloat16)
            self.right.weight = self.left.weight
        def forward(self, value):
            assert self.left.weight is self.right.weight
            assert self.left.weight.dtype == torch.float32
            if fail:
                raise RuntimeError("forward failure")
            return self.left(value) + self.right(value)
    unit = Unit()
    original = unit.left.weight
    pointer = original.data_ptr()
    runtime = _LayerOffloadRuntime("cpu", torch.float32)
    if fail:
        with pytest.raises(RuntimeError, match="forward failure"):
            runtime.call(unit, torch.ones(1, 3))
    else:
        result = runtime.call(unit, torch.ones(1, 3))
        expected = 2 * torch.nn.functional.linear(torch.ones(1, 3), original.float())
        torch.testing.assert_close(result, expected)
        assert result.grad_fn is None
    assert unit.left.weight is unit.right.weight is original
    assert original.dtype == torch.bfloat16 and original.data_ptr() == pointer
    assert runtime.target_weight_peak_bytes == runtime.host_conversion_peak_bytes == 24
    assert not runtime.active


def test_failed_partial_transfer_restores_every_already_promoted_parameter(monkeypatch):
    unit = nn.Linear(3, 2, dtype=torch.bfloat16)
    originals = dict(unit.named_parameters())
    runtime = _LayerOffloadRuntime("cpu", torch.float32)
    transfer = runtime._promote
    attempts = []
    def fail_second(tensor):
        attempts.append(tensor)
        if len(attempts) == 2:
            raise RuntimeError("transfer OOM")
        return transfer(tensor)
    monkeypatch.setattr(runtime, "_promote", fail_second)
    with pytest.raises(RuntimeError, match="transfer OOM"):
        runtime.call(unit, torch.ones(1, 3))
    assert all(parameter is originals[name] for name, parameter in unit.named_parameters())
    assert not runtime.active
    monkeypatch.setattr(runtime, "_promote", transfer)
    assert runtime.call(unit, torch.ones(1, 3)).shape == (1, 2)


def test_unit_releases_converted_weights_before_the_next_unit(monkeypatch):
    runtime = _LayerOffloadRuntime("cpu", torch.float32)
    units = [nn.Linear(4, 4, dtype=torch.bfloat16) for _ in range(3)]
    references = []
    transfer = runtime._promote
    def record(tensor):
        staged = transfer(tensor)
        references.append(weakref.ref(staged))
        return staged
    monkeypatch.setattr(runtime, "_promote", record)
    value = torch.ones(1, 4)
    for unit in units:
        assert all(reference() is None for reference in references)
        value = runtime.call(unit, value)
    assert all(reference() is None for reference in references)
    assert runtime.stage_count == 3
    assert runtime.target_weight_peak_bytes == 80


def test_generated_buffer_replacement_survives_return_to_cpu():
    class Unit(nn.Module):
        def __init__(self):
            super().__init__()
            self.register_buffer("state", torch.tensor(1.), persistent=False)
        def forward(self, value):
            self.register_buffer("state", self.state + 1, persistent=False)
            return value * self.state
    unit = Unit()
    runtime = _LayerOffloadRuntime("cpu", torch.float32)
    assert runtime.call(unit, torch.ones(1)).item() == 2
    assert runtime.call(unit, torch.ones(1)).item() == 3
    assert unit.state.device.type == "cpu"


@pytest.mark.parametrize("changes", [
    {"device": "cpu", "kv_device": "cpu"},
    {"dtype": "float16", "kv_dtype": "float16"},
    {"kv_device": "cpu"}, {"sequence_count": 2},
])
def test_offload_rejects_unbound_or_inconsistent_policy_before_model_io(changes):
    with pytest.raises(ValueError, match="deployment_local_policy_unsupported"):
        ModelManager().load_layer_range(0, 2, has_lm_head=False,
                                       load_spec=policy(**changes).to_dict())


@pytest.mark.parametrize("architecture", ["qwen2", "qwen3_5_text"])
def test_real_mmap_source_load_and_prefill_decode_under_cpu_emulation(tmp_path, monkeypatch, architecture):
    torch.manual_seed(97)
    values = dict(vocab_size=32, hidden_size=16, intermediate_size=32,
                  num_hidden_layers=4, num_attention_heads=2,
                  num_key_value_heads=1, max_position_embeddings=64,
                  tie_word_embeddings=False)
    if architecture == "qwen2":
        source = Qwen2ForCausalLM(Qwen2Config(**values))
    else:
        from transformers import Qwen3_5TextConfig, Qwen3_5ForCausalLM
        import inspect
        from transformers.models.qwen3_5 import modeling_qwen3_5
        # The installed FLA kernel requires CUDA. CPU emulation deliberately
        # uses Transformers' real reference kernels for both comparison paths.
        for name in ("torch_chunk_gated_delta_rule", "torch_recurrent_gated_delta_rule"):
            monkeypatch.setattr(modeling_qwen3_5, name,
                                inspect.unwrap(getattr(modeling_qwen3_5, name)))
        config = Qwen3_5TextConfig(**values, head_dim=8,
            linear_key_head_dim=4, linear_value_head_dim=4,
            linear_num_key_heads=2, linear_num_value_heads=2,
            layer_types=["linear_attention", "full_attention"] * 2,
            rope_parameters={"rope_type": "default", "rope_theta": 10000.,
                             "partial_rotary_factor": 1., "mrope_section": [1, 1, 2]})
        source = Qwen3_5ForCausalLM(config)
    source = source.to(dtype=torch.bfloat16).eval()
    source.save_pretrained(tmp_path, safe_serialization=True)
    if architecture == "qwen3_5_text":
        # Match the official multimodal checkpoint prefix consumed by the
        # production text adapter while keeping a tiny text-only fixture.
        from safetensors.torch import load_file, save_file
        path = str(tmp_path / "model.safetensors")
        tensors = load_file(path)
        save_file({name.replace("model.", "model.language_model.", 1): tensor
                   for name, tensor in tensors.items()}, path, metadata={"format": "pt"})
    expected = deepcopy(source).float().eval()
    monkeypatch.setattr(model_module.AutoTokenizer, "from_pretrained", lambda *a, **k: object())
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    # The transfer target is replaced with CPU before execution below.
    monkeypatch.setattr(model_module, "_layer_offload_device_available", lambda device: True)
    materialized = {}
    materialize = model_module._LayerRangeLoadTracker.materialize
    def record(tracker, handle, key, **kwargs):
        tensor = materialize(tracker, handle, key, **kwargs)
        materialized[key] = tensor.data_ptr()
        return tensor
    monkeypatch.setattr(model_module._LayerRangeLoadTracker, "materialize", record)
    first, last = ModelManager(), ModelManager()
    try:
        for manager, start, end in ((first, 0, 2), (last, 2, 4)):
            materialized.clear()
            spec = policy(components=("layers", "final_norm", "embedding" if start == 0 else "lm_head"))
            manager.load_layer_range(start, end, has_embedding=start == 0,
                                     has_lm_head=end == 4, model_path=str(tmp_path),
                                     total_layers=4, load_spec=spec.to_dict())
            assert manager._compiled_layer_loop is None
            assert manager.get_device() == torch.device("cuda:0")
            observation = manager.layer_offload_observation()
            assert observation["execution_dtype"] == "float32"
            assert observation["resident_source_bytes"] > 0 and observation["stage_count"] == 0
            proof = manager._layer_load_metrics["materialization_proof"]
            import os
            import psutil
            if hasattr(psutil.Process(), "memory_maps"):
                mappings = {os.path.normcase(os.path.abspath(mapping.path))
                            for mapping in psutil.Process().memory_maps() if mapping.path}
                assert os.path.normcase(os.path.abspath(tmp_path / "model.safetensors")) in mappings
            assert all(tensor.element_size == tensor.source_element_size == 2 for tensor in proof.tensors)
            # The installed CPU Parameters share the exact storage returned by
            # safe_open, rather than allocating a target-dtype model on the host.
            for name, parameter in manager.model.named_parameters():
                original_index = int(name.split(".")[2]) if name.startswith("model.layers.") else None
                source_name = name if original_index is None else name.replace(
                    f"layers.{original_index}.", f"layers.{original_index + start}.", 1)
                if architecture == "qwen3_5_text":
                    source_name = source_name.replace("model.", "model.language_model.", 1)
                assert parameter.dtype == torch.bfloat16 and parameter.device.type == "cpu"
                assert parameter.data_ptr() == materialized[source_name]
            # This explicit substitution emulates only transfers on CPU; no
            # test claims CUDA execution or measured VRAM on the CPU wheel.
            manager._layer_offload.execution_device = torch.device("cpu")
        tokens = torch.tensor([[1, 2, 3]])
        with torch.no_grad():
            full = expected(input_ids=tokens, use_cache=True)
        head = first.forward_layers(input_ids=tokens)
        tail = last.forward_layers(hidden_states=head["hidden_states"])
        torch.testing.assert_close(tail["logits"], full.logits, atol=1e-5, rtol=1e-5)
        next_token = torch.tensor([[4]])
        with torch.no_grad():
            decoded = expected(input_ids=next_token, past_key_values=full.past_key_values, use_cache=True)
        head_next = first.forward_layers(input_ids=next_token, past_key_values=head["cache"])
        tail_next = last.forward_layers(hidden_states=head_next["hidden_states"], past_key_values=tail["cache"])
        torch.testing.assert_close(tail_next["logits"], decoded.logits, atol=1e-5, rtol=1e-5)
        projection = project_runtime_descriptor(tmp_path, inspect_pipeline_model(tmp_path), 4,
                                                target_dtype="float32")
        for manager, start, end in ((first, 0, 2), (last, 2, 4)):
            quote = runtime_range_capacity(projection, start, end,
                                           has_embedding=start == 0, has_lm_head=end == 4)
            observation = manager.layer_offload_observation()
            assert observation["stage_count"] > 0 and not observation["active"]
            assert observation["target_weight_peak_bytes"] == quote["device_weight_bytes"]
            assert observation["device_weight_bytes"] == quote["device_weight_bytes"]
            assert observation["host_conversion_peak_bytes"] == quote["host_conversion_bytes"]
            assert observation["host_conversion_bytes"] == quote["host_conversion_bytes"]
            assert observation["target_weight_bytes"] >= quote["target_weight_bytes"]
            assert all(parameter.device.type == "cpu" and parameter.dtype == torch.bfloat16
                       for parameter in manager.model.parameters())
    finally:
        first.unload_model()
        last.unload_model()
    assert first._layer_offload is last._layer_offload is None


def test_retained_forward_exception_keeps_no_staged_weight_references(monkeypatch):
    class Unit(nn.Linear):
        def forward(self, value):
            raise RuntimeError("keep traceback for diagnosis")
    unit = Unit(3, 2, dtype=torch.bfloat16)
    runtime = _LayerOffloadRuntime("cpu", torch.float32)
    references = []
    promote = runtime._promote
    def record(tensor):
        converted = promote(tensor)
        references.append(weakref.ref(converted))
        return converted
    monkeypatch.setattr(runtime, "_promote", record)
    with pytest.raises(RuntimeError) as error:
        runtime.call(unit, torch.ones(1, 3))
    assert error.value.__traceback__ is not None
    assert references and all(reference() is None for reference in references)
    assert not runtime.active


def test_offload_preserves_generated_fp32_rotary_buffer_precision():
    class Rotary(nn.Module):
        def __init__(self):
            super().__init__()
            self.register_buffer("inv_freq", torch.tensor([0.123456789]), persistent=False)
        def forward(self, value):
            assert self.inv_freq.dtype == torch.float32
            return value.float() * self.inv_freq
    unit = Rotary()
    runtime = _LayerOffloadRuntime("cpu", torch.float16)
    original = unit.inv_freq
    result = runtime.call(unit, torch.ones(1))
    torch.testing.assert_close(result, original)
    assert unit.inv_freq is original
    assert runtime.host_conversion_peak_bytes == 0
    assert runtime.target_weight_peak_bytes == 4


def test_public_offload_requires_the_selected_cuda_device(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    with pytest.raises(ValueError, match="deployment_layer_offload_device_unavailable"):
        ModelManager().load_layer_range(0, 2, has_lm_head=False, load_spec=policy().to_dict())


@pytest.mark.requires_gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="micro-CUDA verification requires a CUDA torch wheel")
def test_micro_cuda_executes_one_unit_and_releases_weights_to_cpu():
    # The main product gate schedules this isolated micro-test; it uses no
    # model files or service ports and leaves all source weights on CPU.
    runtime = _LayerOffloadRuntime("cuda:0", torch.float16)
    units = [nn.Linear(512, 512, bias=False, dtype=torch.bfloat16) for _ in range(8)]
    value = torch.ones(1, 512, device="cuda:0", dtype=torch.float16)
    warm = torch.ones(512, 512, device="cuda:0", dtype=torch.float16)
    torch.nn.functional.linear(value, warm)
    torch.cuda.synchronize()
    del warm
    torch.cuda.empty_cache()
    baseline = torch.cuda.memory_allocated()
    torch.cuda.reset_peak_memory_stats()
    for unit in units:
        original = unit.weight
        result = runtime.call(unit, value)
        torch.cuda.synchronize()
        assert result.device.type == "cuda" and result.dtype == torch.float16
        assert unit.weight is original and original.device.type == "cpu"
        assert torch.cuda.memory_allocated() - baseline < runtime.target_weight_peak_bytes
        del result
    peak = torch.cuda.max_memory_allocated() - baseline
    # CUDA allocator padding and GEMM scratch are independent of source dtype.
    # The bound excludes retaining all eight layer copies simultaneously.
    total_target = sum(unit.weight.numel() * 2 for unit in units)
    assert runtime.target_weight_peak_bytes == 512 * 512 * 2
    assert runtime.host_conversion_peak_bytes == 512 * 512 * 2
    assert peak < total_target
    print({"cuda_peak_delta_bytes": peak, "max_target_unit_bytes": runtime.target_weight_peak_bytes,
           "all_target_weights_bytes": total_target, "stages": runtime.stage_count})


def test_worker_offload_receipt_reports_real_source_and_device_domains(tiny_supply, monkeypatch):
    import threading
    from model_supply_worker import get_selected_model
    from qlh_core.contracts.deployment import require_runtime_receipt
    from tests.test_worker_prepare import command
    from tests.test_worker_prepare_runtime import supply_identity
    manifest = tiny_supply.activate(0)
    identity = supply_identity(manifest, 1)
    spec = policy(dtype="float16", quant_type="fp16", kv_dtype="float16")
    identity["load_spec_digest"] = spec.digest
    monkeypatch.setattr(model_module, "_layer_offload_device_available", lambda device: True)
    host = tiny_supply.host()
    try:
        host.prepare_worker_runtime(command("prepare", identity, load_spec=spec.to_dict()),
                                    get_selected_model(), threading.Event())
        receipt = host.worker_runtime_receipt(identity)
        manager = host._worker_runtime_candidate.manager
        observation = manager.layer_offload_observation()
        assert all(p.device.type == "cpu" and p.dtype == torch.float32
                   for p in manager.model.parameters())
        assert observation["stage_count"] == 0
        assert receipt["materialized_bytes"] == receipt["host_source_bytes"] == observation["resident_source_bytes"]
        assert receipt["device_weight_bytes"] == observation["device_weight_bytes"]
        assert receipt["host_conversion_bytes"] == observation["host_conversion_bytes"] > 0
        assert receipt["materialized_bytes"] > receipt["device_weight_bytes"]
        # Only the coordinator owns admission. It receives authentic tensor
        # measurements and validates each domain against its frozen budget.
        host_required = receipt["host_source_bytes"] + receipt["host_conversion_bytes"]
        require_runtime_receipt({"identity": identity, "load_spec": spec.to_dict(),
                                 "required_bytes": receipt["device_weight_bytes"],
                                 "host_required_bytes": host_required,
                                 "host_capacity_bytes": host_required}, receipt)
    finally:
        host.release_worker_runtime()


@pytest.mark.parametrize("target_dtype", [torch.float16, torch.float32])
def test_promote_drops_cpu_conversion_when_transfer_traceback_is_retained(monkeypatch, target_dtype):
    source = torch.ones(8, dtype=torch.bfloat16)
    runtime = _LayerOffloadRuntime("cpu", target_dtype)
    references = []
    real_to = torch.Tensor.to
    def transfer(tensor, *args, **kwargs):
        if "device" in kwargs:
            assert tensor.dtype == target_dtype and tensor.data_ptr() != source.data_ptr()
            references.append(weakref.ref(tensor))
            # Emulate a native transfer error: no lower Python frame owns the
            # input, leaving _promote itself responsible for the CPU staging.
            tensor = None
            raise RuntimeError("device transfer failed")
        return real_to(tensor, *args, **kwargs)
    with monkeypatch.context() as transfer_patch:
        transfer_patch.setattr(torch.Tensor, "to", transfer)
        with pytest.raises(RuntimeError, match="device transfer failed") as error:
            runtime._promote(source)
    assert error.value.__traceback__ is not None
    assert references and all(reference() is None for reference in references)
    assert source.dtype == torch.bfloat16
    assert runtime.host_conversion_peak_bytes == source.numel() * torch.empty((), dtype=target_dtype).element_size()


def test_retained_transfer_exception_releases_partial_unit_and_allows_retry(monkeypatch):
    unit = nn.Linear(3, 2, dtype=torch.bfloat16)
    originals = dict(unit.named_parameters())
    runtime = _LayerOffloadRuntime("cpu", torch.float16)
    converted = []
    real_to = torch.Tensor.to
    def transfer(tensor, *args, **kwargs):
        if "device" in kwargs:
            converted.append(weakref.ref(tensor))
            if len(converted) == 2:
                # The exception deliberately retains the receiver in this
                # lower Python frame, like a wrapped native transfer path.
                raise RuntimeError("partial transfer failure")
        return real_to(tensor, *args, **kwargs)
    with monkeypatch.context() as transfer_patch:
        transfer_patch.setattr(torch.Tensor, "to", transfer)
        with pytest.raises(RuntimeError, match="partial transfer failure") as error:
            runtime.call(unit, torch.ones(1, 3, dtype=torch.float16))
    assert error.value.__traceback__ is not None
    assert converted and all(reference() is None for reference in converted)
    assert all(parameter is originals[name] for name, parameter in unit.named_parameters())
    assert not runtime.active
    result = runtime.call(unit, torch.ones(1, 3, dtype=torch.float16))
    assert result.shape == (1, 2) and torch.isfinite(result).all()
    assert all(parameter is originals[name] for name, parameter in unit.named_parameters())


def test_retained_forward_traceback_releases_local_weight_views_and_retry_succeeds():
    references = []
    class Unit(nn.Linear):
        fail = True
        def forward(self, value):
            view = self.weight.view_as(self.weight)
            references.extend((weakref.ref(self.weight), weakref.ref(view)))
            if self.fail:
                raise RuntimeError("forward retained a weight view")
            return torch.nn.functional.linear(value, view, self.bias)
    unit = Unit(3, 2, dtype=torch.bfloat16)
    original = unit.weight
    runtime = _LayerOffloadRuntime("cpu", torch.float32)
    with pytest.raises(RuntimeError, match="forward retained a weight view") as error:
        runtime.call(unit, torch.ones(1, 3))
    assert error.value.__traceback__ is not None
    assert references and all(reference() is None for reference in references)
    assert unit.weight is original and not runtime.active
    unit.fail = False
    assert torch.isfinite(runtime.call(unit, torch.ones(1, 3))).all()
    assert all(reference() is None for reference in references)
