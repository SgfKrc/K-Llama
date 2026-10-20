"""Bounded real-artifact CUDA layer smoke; no service or model copies.

Run with a CUDA-enabled interpreter. CPU-only development checks use --help;
model loading requires an explicit local --model directory.
"""
from __future__ import annotations

import argparse
import gc
import json
import math
import os
from pathlib import Path
import sys
import tempfile
import time


def parser():
    value = argparse.ArgumentParser(description=__doc__)
    value.add_argument("--model", type=Path, required=True)
    value.add_argument("--device", default="cuda:0")
    value.add_argument("--start-layer", type=int, default=0)
    value.add_argument("--end-layer", type=int, default=4)
    value.add_argument("--retain-lm-head", action="store_true")
    value.add_argument("--context-size", type=int, default=128)
    value.add_argument("--batch-size", type=int, default=32)
    value.add_argument("--decode-steps", type=int, default=3)
    value.add_argument("--prompt", default="Explain what a memory mapped model is in one sentence.")
    value.add_argument("--dtype", choices=("float16", "float32"), default="float16")
    value.add_argument("--reserve-mib", type=int, default=512)
    value.add_argument("--safety-margin", type=float, default=1.2)
    value.add_argument("--compare-resident", action="store_true")
    value.add_argument("--atol", type=float, default=0.02)
    value.add_argument("--rtol", type=float, default=0.02)
    value.add_argument("--output", type=Path, required=True)
    return value


def run(args):
    import psutil
    import torch
    from model_module import ModelManager
    from model_supply_runtime_capacity import project_runtime_descriptor, runtime_range_capacity
    from pipeline_model_descriptor import inspect_pipeline_model
    from qlh_core.contracts.load import RuntimeLoadSpec

    if not args.device.startswith("cuda:") or not torch.cuda.is_available():
        raise ValueError("smoke requires an available explicit CUDA device")
    if (args.reserve_mib < 0 or not math.isfinite(args.safety_margin)
            or args.safety_margin < 1 or not 0 <= args.decode_steps <= 16):
        raise ValueError("invalid smoke budget or decode limit")
    descriptor = inspect_pipeline_model(args.model)
    total = descriptor["total_layers"]
    start, end = args.start_layer, args.end_layer
    if not 0 <= start < end <= total or end - start >= total:
        raise ValueError("smoke requires a proper layer range; use the product gate for a full model")
    has_embedding, has_lm_head = start == 0, end == total or args.retain_lm_head
    components = ["layers", "final_norm"]
    if has_embedding:
        components.append("embedding")
    if has_lm_head:
        components.append("lm_head")
    dtype = getattr(torch, args.dtype)
    projection = project_runtime_descriptor(args.model, descriptor, 2 if args.dtype == "float16" else 4,
                                             target_dtype=args.dtype)
    torch.cuda.set_device(args.device)
    torch.manual_seed(97)
    warm = torch.ones((32, 32), device=args.device, dtype=dtype)
    torch.mm(warm, warm)
    torch.cuda.synchronize()
    del warm
    torch.cuda.empty_cache()
    process = psutil.Process()
    report = {"model_type": descriptor["model_type"], "layer_range": [start, end],
              "has_embedding": has_embedding, "has_lm_head": has_lm_head,
              "torch": torch.__version__, "device": torch.cuda.get_device_name(),
              "passed": False, "runs": []}
    args._smoke_report = report
    outputs = []
    prompt_ids = None
    for materialization in (["safetensors_layer_offload", "safetensors_key_filtered"]
                            if args.compare_resident else ["safetensors_layer_offload"]):
        quote = runtime_range_capacity(projection, start, end, has_embedding=has_embedding,
                                        has_lm_head=has_lm_head, materialization=materialization)
        spec = RuntimeLoadSpec(device=args.device, dtype=args.dtype,
            quant_type="fp16" if dtype == torch.float16 else "fp32", context_size=args.context_size,
            kv_device=args.device, kv_dtype=args.dtype, kv_policy="dynamic",
            components=tuple(components), materialization=materialization,
            batch_size=args.batch_size, sequence_count=1)
        if args.decode_steps >= args.context_size or args.batch_size <= 0:
            raise ValueError("prompt and decode must fit context")
        reserve = args.reserve_mib * 1024 * 1024
        host_required = math.ceil(quote["host_weight_peak_bytes"] * args.safety_margin) + reserve
        device_required = math.ceil(quote["device_weight_bytes"] * args.safety_margin) + reserve
        host_available = psutil.virtual_memory().available
        device_available, _ = torch.cuda.mem_get_info()
        if host_required > host_available or device_required > device_available:
            raise RuntimeError("independent host/CUDA smoke admission failed")
        manager = ModelManager()
        item = {"load_spec": spec.to_dict(), "quote": quote,
                "host_required_bytes": host_required, "host_available_bytes": host_available,
                "device_required_bytes": device_required, "device_available_bytes": device_available,
                "steps": []}
        report["runs"].append(item)
        rss_baseline = process.memory_info().rss
        device_baseline = torch.cuda.memory_allocated()
        torch.cuda.reset_peak_memory_stats()
        cache = result = output = None
        cached = []
        try:
            begun = time.monotonic()
            manager.load_layer_range(start, end, has_embedding=has_embedding, has_lm_head=has_lm_head,
                                     model_path=str(args.model), total_layers=total, load_spec=spec.to_dict())
            torch.cuda.synchronize()
            item.update(load_seconds=time.monotonic() - begun,
                        load_peak_delta_bytes=torch.cuda.max_memory_allocated() - device_baseline,
                        rss_after_load_bytes=process.memory_info().rss)
            original = {name: (id(tensor), tensor.data_ptr(), tensor.dtype)
                        for name, tensor in manager.model.named_parameters()}
            if hasattr(process, "memory_maps"):
                mappings = {os.path.normcase(os.path.abspath(value.path))
                            for value in process.memory_maps() if value.path}
                proof = manager._layer_load_metrics["materialization_proof"]
                item["source_files_mapped"] = all(os.path.normcase(os.path.abspath(t.source_file)) in mappings
                                                  for t in proof.tensors if t.source_file)
                if materialization == "safetensors_layer_offload":
                    assert item["source_files_mapped"], "source mmap disappeared after load"
            if prompt_ids is None:
                prompt_ids = manager.tokenizer(args.prompt, return_tensors="pt", truncation=True,
                    max_length=min(args.batch_size, args.context_size - args.decode_steps))["input_ids"]
            pass_outputs = []
            forward_baseline = torch.cuda.memory_allocated()
            torch.cuda.reset_peak_memory_stats()
            for index in range(args.decode_steps + 1):
                ids = prompt_ids if index == 0 else prompt_ids[:, -1:]
                if has_embedding:
                    inputs = {"input_ids": ids}
                else:
                    # Identical synthetic input for both placement modes; this
                    # verifies segment execution rather than text generation.
                    generator = torch.Generator().manual_seed(97 + index)
                    inputs = {"hidden_states": torch.randn((1, ids.shape[1], descriptor["hidden_size"]),
                                                           generator=generator, dtype=dtype)}
                begun = time.monotonic()
                result = manager.forward_layers(**inputs, past_key_values=cache, use_cache=True)
                cache = result["cache"]  # Hybrid recurrent state needs the actual cache object.
                output = result["logits" if has_lm_head else "hidden_states"]
                torch.cuda.synchronize()
                assert output.device == torch.device(args.device) and torch.isfinite(output).all().item()
                pass_outputs.append(output.detach().cpu())
                cached = [value for layer in cache.layers for value in vars(layer).values()
                          if isinstance(value, torch.Tensor)]
                assert cached and all(value.device == torch.device(args.device) for value in cached)
                step = {"index": index, "seconds": time.monotonic() - begun,
                        "cuda_allocated_bytes": torch.cuda.memory_allocated(),
                        "cuda_reserved_bytes": torch.cuda.memory_reserved(),
                        "rss_bytes": process.memory_info().rss}
                if materialization == "safetensors_layer_offload":
                    assert all(tensor.device.type == "cpu" and
                               (id(tensor), tensor.data_ptr(), tensor.dtype) == original[name]
                               for name, tensor in manager.model.named_parameters())
                    step["observation"] = manager.layer_offload_observation()
                    assert not step["observation"]["active"]
                item["steps"].append(step)
                del output
            outputs.append(pass_outputs)
            item.update(forward_peak_delta_bytes=torch.cuda.max_memory_allocated() - forward_baseline,
                        cuda_peak_reserved_bytes=torch.cuda.max_memory_reserved(),
                        rss_delta_bytes=process.memory_info().rss - rss_baseline)
            if len(outputs) == 2:
                for offloaded, resident in zip(*outputs):
                    torch.testing.assert_close(offloaded, resident, atol=args.atol, rtol=args.rtol)
                report["resident_comparison_passed"] = True
        finally:
            cache = result = output = None
            cached.clear()
            manager.unload_model()
            gc.collect()
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
            item["cuda_after_unload_bytes"] = torch.cuda.memory_allocated()
    report["passed"] = True
    return report


def main():
    args = parser().parse_args()
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
    with tempfile.TemporaryDirectory(prefix="qlh-offload-smoke-") as temporary:
        # Isolate application state before importing any runtime modules.
        for name, suffix in {"QLH_STATE_DIR": "state", "QLH_SQLITE_PATH": "store.sqlite3",
                             "QLH_NODE_CONFIG_PATH": "node.json",
                             "QLH_NODE_SECRET_STORE_PATH": "secrets.json"}.items():
            os.environ[name] = str(Path(temporary) / suffix)
        try:
            report = run(args)
        except Exception as error:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            report = getattr(args, "_smoke_report", {})
            report.update(passed=False, error=str(error), type=type(error).__name__)
            args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
            raise
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
