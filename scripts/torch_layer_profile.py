"""Collect isolated, real PyTorch transformer-layer costs for offline planning.

The full model is used once to capture the real hidden-state inputs at every
layer. Each layer is then loaded as its own QLH layer-range manager and timed
without ``torch.profiler`` or dispatch hooks. The result is research evidence
only; it never enables runtime placement.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import statistics
import sys
import time
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
REPORT_SCHEMA = "qlh.torch_layer_profile.v1"


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _text_config(config: dict[str, Any]) -> dict[str, Any]:
    """Return the text-tower config for nested/hybrid architectures.

    `qwen3_5` declares its layer geometry under `text_config`
    (`Qwen3_5ForConditionalGeneration` has no top-level `num_hidden_layers`), so
    every geometry read must fall back there.
    """
    nested = config.get("text_config")
    return nested if isinstance(nested, dict) else {}


def _config_value(config: dict[str, Any], name: str, default: Any = 0) -> Any:
    value = config.get(name)
    if value is None:
        value = _text_config(config).get(name)
    return default if value is None else value


def _weight_files(model_dir: Path) -> list[Path]:
    """Resolve the safetensors shard set: single file, or `model-*-of-NNNNN` shards."""
    single = model_dir / "model.safetensors"
    if single.is_file():
        return [single]
    index_path = model_dir / "model.safetensors.index.json"
    if index_path.is_file():
        index = json.loads(index_path.read_text(encoding="utf-8"))
        names = sorted({str(name) for name in (index.get("weight_map") or {}).values()})
        shards = [model_dir / name for name in names]
        missing = [str(path) for path in shards if not path.is_file()]
        if missing:
            raise FileNotFoundError(f"safetensors index lists missing shards: {missing[:3]}")
        return shards
    shards = sorted(model_dir.glob("*.safetensors"))
    if not shards:
        raise FileNotFoundError(f"expected local safetensors weights under: {model_dir}")
    return shards


def _artifact_identity(model_dir: Path) -> dict[str, Any]:
    config = json.loads((model_dir / "config.json").read_text(encoding="utf-8"))
    manifest_path = model_dir / "model.manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.is_file() else {}
    weights = _weight_files(model_dir)
    # A sharded checkpoint is identified by its ordered per-shard digest set; a
    # single-file checkpoint degenerates to the original single digest.
    digests = [[path.name, _sha256_file(path)] for path in weights]
    return {
        "name": model_dir.name,
        "manifest_sha256": manifest.get("manifest_sha256"),
        "weights_sha256": digests[0][1] if len(digests) == 1 else None,
        "weights_shards": digests,
        "weights_bytes": sum(path.stat().st_size for path in weights),
        "model_type": config.get("model_type"),
        "num_hidden_layers": int(_config_value(config, "num_hidden_layers")),
        "hidden_size": int(_config_value(config, "hidden_size")),
    }


def _summary(samples: list[float]) -> dict[str, Any]:
    ordered = sorted(float(value) for value in samples)
    mean = statistics.fmean(ordered)
    cv = statistics.pstdev(ordered) / mean if mean else float("inf")
    return {
        "samples_ms": [round(value, 6) for value in ordered],
        "count": len(ordered),
        "median_ms": round(statistics.median(ordered), 6),
        "mean_ms": round(mean, 6),
        "cv": round(cv, 6),
        "min_ms": round(ordered[0], 6),
        "max_ms": round(ordered[-1], 6),
    }


def _sync(torch_module: Any, device: Any) -> None:
    if getattr(device, "type", "") == "cuda":
        torch_module.cuda.synchronize(device)


def _apply_cpu_affinity(spec: str | None) -> dict[str, Any] | None:
    if not spec:
        return None
    cpus = tuple(sorted({int(item.strip()) for item in spec.split(",") if item.strip()}))
    if not cpus or any(item < 0 for item in cpus):
        raise ValueError("cpu-affinity must contain non-negative logical CPU ids")
    mask = sum(1 << item for item in cpus)
    if os.name == "nt":
        import ctypes
        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel32.GetCurrentProcess.restype = ctypes.c_void_p
        kernel32.SetProcessAffinityMask.argtypes = (ctypes.c_void_p, ctypes.c_size_t)
        kernel32.SetProcessAffinityMask.restype = ctypes.c_int
        result = kernel32.SetProcessAffinityMask(
            kernel32.GetCurrentProcess(), ctypes.c_size_t(mask),
        )
        if result == 0:
            raise OSError("SetProcessAffinityMask failed")
        method = "windows_SetProcessAffinityMask"
    else:
        os.sched_setaffinity(0, cpus)
        method = "os.sched_setaffinity"
    return {"requested_logical_cpus": list(cpus), "mask": mask, "applied": True, "method": method}


def _cache(output: dict[str, Any]) -> Any:
    value = output.get("cache")
    if value is None:
        value = output.get("past_key_values")
    if value is None:
        raise RuntimeError("layer profile forward returned no cache")
    return value


def _shape_fingerprint(tensor: Any) -> str:
    payload = json.dumps({
        "shape": list(tensor.shape),
        "dtype": str(tensor.dtype),
    }, sort_keys=True).encode("utf-8")
    return "shape:sha256:" + hashlib.sha256(payload).hexdigest()


def _capture_real_inputs(torch_module: Any, manager: Any, prompt: Any, *, total_layers: int) -> tuple[dict[int, Any], dict[int, Any], int, str]:
    """Capture layer inputs only; no timings are taken while hooks are active."""
    import model_module

    transformer, layers_attr, _ = model_module._locate_text_transformer(manager.model)
    layers = list(getattr(transformer, layers_attr))
    if len(layers) != total_layers:
        raise RuntimeError(f"full model layer count mismatch: {len(layers)} != {total_layers}")
    captured: dict[str, dict[int, Any]] = {"prefill": {}, "decode": {}}
    current_phase = "prefill"

    def make_hook(index: int):
        def hook(_module: Any, args: tuple[Any, ...]) -> None:
            if index not in captured[current_phase]:
                if not args:
                    raise RuntimeError(f"layer {index} did not receive hidden states")
                captured[current_phase][index] = args[0].detach().clone()
        return hook

    handles = [layer.register_forward_pre_hook(make_hook(index)) for index, layer in enumerate(layers)]
    try:
        with torch_module.no_grad():
            prefill = manager.forward_layers(input_ids=prompt, use_cache=True)
            logits = prefill["logits"]
            token_id = int(logits[:, -1, :].argmax(dim=-1).detach().cpu().item())
            current_phase = "decode"
            manager.forward_layers(
                input_ids=prompt.new_tensor([[token_id]]),
                past_key_values=_cache(prefill),
                use_cache=True,
            )
    finally:
        for handle in handles:
            handle.remove()
    if len(captured["prefill"]) != total_layers or len(captured["decode"]) != total_layers:
        raise RuntimeError("full forward did not expose every layer input")
    return captured["prefill"], captured["decode"], token_id, _shape_fingerprint(prompt)


def _measure_layer(
    torch_module: Any,
    manager_module: Any,
    model_dir: Path,
    layer_index: int,
    total_layers: int,
    prefill_input: Any,
    decode_input: Any,
    device: Any,
    *,
    precision: str,
    warmup: int,
    repeats: int,
    inner_loops: int,
) -> dict[str, Any]:
    manager = manager_module.ModelManager()
    manager.load_layer_range(
        layer_index, layer_index + 1,
        has_embedding=False,
        has_lm_head=False,
        model_path=str(model_dir),
        quant_type="fp16",
        total_layers=total_layers,
    )
    manager.model.eval()
    if precision == "fp32" or device.type == "cpu":
        manager.model.float()
    try:
        def prefill_once() -> Any:
            with torch_module.no_grad():
                return manager.forward_layers(hidden_states=prefill_input, use_cache=True)

        def decode_once(cache: Any) -> Any:
            with torch_module.no_grad():
                return manager.forward_layers(
                    hidden_states=decode_input,
                    past_key_values=cache,
                    use_cache=True,
                )

        for _ in range(warmup):
            for _ in range(inner_loops):
                decode_once(_cache(prefill_once()))
        _sync(torch_module, device)
        prefill_samples: list[float] = []
        decode_samples: list[float] = []
        for _ in range(repeats):
            _sync(torch_module, device)
            started = time.perf_counter()
            prefill_output = None
            for _ in range(inner_loops):
                prefill_output = prefill_once()
            _sync(torch_module, device)
            prefill_samples.append((time.perf_counter() - started) * 1000.0 / inner_loops)
            decode_caches = [_cache(prefill_once()) for _ in range(inner_loops)]
            _sync(torch_module, device)
            started = time.perf_counter()
            decode_output = None
            for decode_cache in decode_caches:
                decode_output = decode_once(decode_cache)
            _sync(torch_module, device)
            decode_samples.append((time.perf_counter() - started) * 1000.0 / inner_loops)
            for output in (prefill_output, decode_output):
                hidden = output.get("hidden_states")
                if hidden is None or not torch_module.isfinite(hidden).all().item():
                    raise RuntimeError(f"layer {layer_index} produced non-finite output")
        parameter_bytes = sum(
            int(parameter.numel()) * int(parameter.element_size())
            for parameter in manager.model.parameters()
        )
        dtype = str(next(manager.model.parameters()).dtype).removeprefix("torch.")
        node = {
            "node_id": f"layer-{layer_index}",
            "operator_id": "transformer_layer_loop",
            "layer_index": layer_index,
            "dtype": dtype,
            "shape_fingerprint": _shape_fingerprint(prefill_input),
            "resident_bytes": parameter_bytes,
            "workspace_bytes": 0,
        }
        return {
            "node": node,
            "prefill_shape_fingerprint": _shape_fingerprint(prefill_input),
            "decode_shape_fingerprint": _shape_fingerprint(decode_input),
            "prefill": _summary(prefill_samples),
            "decode": _summary(decode_samples),
            "warmup_iterations": warmup,
            "correctness": {"finite_hidden": True, "end_to_end_argmax": "captured_by_full_reference"},
        }
    finally:
        manager.unload_model()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, default=ROOT / "models" / "qwen2.5-0.5b-instruct")
    parser.add_argument("--json-out", type=Path, required=True)
    parser.add_argument("--prompt", default="Explain distributed inference in one sentence.")
    parser.add_argument("--prefill-tokens", type=int, default=64)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--interop-threads", type=int, default=1)
    parser.add_argument("--layer-start", type=int, default=0)
    parser.add_argument("--layer-end", type=int, default=None)
    parser.add_argument("--precision", choices=("fp32", "device_default"), default="fp32")
    parser.add_argument("--inner-loops", type=int, default=1)
    parser.add_argument("--cpu-affinity", default=None, help="comma-separated logical CPU ids")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if (args.prefill_tokens < 1 or args.warmup < 1 or args.repeats < 3
            or args.threads < 1 or args.layer_start < 0 or args.inner_loops < 1):
        print("FAIL: invalid measurement configuration", file=sys.stderr)
        return 2
    if not args.model_dir.is_dir():
        print(f"FAIL: model directory does not exist: {args.model_dir}", file=sys.stderr)
        return 2
    cpu_affinity = _apply_cpu_affinity(args.cpu_affinity)
    sys.path.insert(0, str(ROOT))
    sys.path.insert(0, str(ROOT / "src"))
    import torch
    import config as qlh_config
    import model_module
    from transformers import AutoConfig, AutoTokenizer

    qlh_config.TRUST_REMOTE_CODE = False
    model_module.TRUST_REMOTE_CODE = False
    model_module.USE_COMPILE = False
    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(args.interop_threads)
    torch.manual_seed(0)
    config = AutoConfig.from_pretrained(str(args.model_dir), trust_remote_code=False, local_files_only=True)
    # ★ 2026-10-25：`qwen3_5` 是 hybrid 架构（`layer_types` 混有 linear_attention 与
    #   full_attention），层数声明在 `text_config` 下 —— `Qwen3_5ForConditionalGeneration`
    #   的顶层 config 没有 `num_hidden_layers`，直接取会 AttributeError。这里按
    #   「顶层优先、text_config 兜底」读取，对既有标准架构行为不变。
    _text_config = getattr(config, "text_config", None)
    total_layers = int(getattr(config, "num_hidden_layers", None)
                       or getattr(_text_config, "num_hidden_layers", 0))
    if total_layers <= 0:
        print("FAIL: cannot resolve num_hidden_layers", file=sys.stderr)
        return 2
    layer_end = total_layers if args.layer_end is None else args.layer_end
    if layer_end <= args.layer_start or layer_end > total_layers:
        print("FAIL: invalid layer range", file=sys.stderr)
        return 2
    tokenizer = AutoTokenizer.from_pretrained(str(args.model_dir), trust_remote_code=False, local_files_only=True)
    token_ids = tokenizer(args.prompt, return_tensors="pt", add_special_tokens=False)["input_ids"]
    if token_ids.numel() == 0:
        raise RuntimeError("prompt tokenized to an empty input")
    repeat_count = (args.prefill_tokens + token_ids.shape[1] - 1) // token_ids.shape[1]
    prompt = token_ids.repeat(1, repeat_count)[:, :args.prefill_tokens].contiguous()
    full = model_module.ModelManager()
    full.load_layer_range(
        0, total_layers, has_embedding=True, has_lm_head=True,
        model_path=str(args.model_dir), quant_type="fp16", total_layers=total_layers,
    )
    device = full.get_device()
    if args.precision == "fp32" or device.type == "cpu":
        full.model.float()
    prompt = prompt.to(device=device)
    try:
        prefill_inputs, decode_inputs, token_id, prompt_shape = _capture_real_inputs(
            torch, full, prompt, total_layers=total_layers,
        )
    finally:
        full.unload_model()
    identity = _artifact_identity(args.model_dir)
    model_fp = "model:sha256:" + str(identity["weights_sha256"])
    workload_fp = "workload:sha256:" + hashlib.sha256(
        json.dumps({"prompt_shape": prompt_shape, "prefill_tokens": args.prefill_tokens, "decode_token": token_id}, sort_keys=True).encode("utf-8")
    ).hexdigest()
    profiles = []
    for index in range(args.layer_start, layer_end):
        print(f"[layer-profile] layer {index + 1}/{total_layers}", flush=True)
        profiles.append(_measure_layer(
            torch, model_module, args.model_dir, index, total_layers,
            prefill_inputs[index], decode_inputs[index], device,
            precision=args.precision,
            warmup=args.warmup, repeats=args.repeats, inner_loops=args.inner_loops,
        ))
    hidden_size = int(identity["hidden_size"])
    dtype_bytes = 4 if profiles[0]["node"]["dtype"] == "float32" else 2
    phase_profiles: dict[str, Any] = {}
    for phase in ("prefill", "decode"):
        shape_key = f"{phase}_shape_fingerprint"
        hidden_bytes = (args.prefill_tokens if phase == "prefill" else 1) * hidden_size * dtype_bytes
        phase_profiles[phase] = {
            "nodes": [
                {**item["node"], "shape_fingerprint": item[shape_key]}
                for item in profiles
            ],
            "edges": [
                {"source_node_id": f"layer-{index}", "destination_node_id": f"layer-{index + 1}", "tensor_bytes": hidden_bytes}
                for index in range(args.layer_start, layer_end - 1)
            ],
            "costs": [
                {
                    "node_id": item["node"]["node_id"],
                    "device_id": device.type,
                    "operator_id": item["node"]["operator_id"],
                    "implementation_id": f"pytorch.eager.{device.type}.transformer_layer_loop",
                    "phase": phase,
                    "dtype": item["node"]["dtype"],
                    "shape_fingerprint": item[shape_key],
                    "model_fingerprint": model_fp,
                    "workload_fingerprint": workload_fp,
                    "samples_ms": item[phase]["samples_ms"],
                    "warmup_consistent": item[phase]["cv"] <= 0.10,
                    "instrumented": False,
                    "source_ref": f"layer_profile:{phase}:layer-{item['node']['layer_index']}",
                }
                for item in profiles
            ],
        }
    report = {
        "schema_version": REPORT_SCHEMA,
        "ticket": "TORCH-HETERO-PLAN-01",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "model_identity": identity,
        "model_fingerprint": model_fp,
        "workload_fingerprint": workload_fp,
        "runtime": {
            "python": sys.version.split()[0],
            "torch": torch.__version__,
            "torch_cuda_runtime": torch.version.cuda,
            "device": {"type": device.type, "name": str(device)},
            "host_identity": platform.node(),
            "threads": args.threads,
            "interop_threads": int(torch.get_num_interop_threads()),
            "cpu_affinity": cpu_affinity,
            "dtype": profiles[0]["node"]["dtype"],
            "precision_mode": args.precision,
        },
        "workload": {"prefill_tokens": args.prefill_tokens, "decode_steps": 1, "decode_token_id": token_id},
        "profiled_layer_range": [args.layer_start, layer_end],
        "measurement": {
            "warmup_iterations": args.warmup,
            "repeats": args.repeats,
            "inner_loops": args.inner_loops,
            "timing": "isolated layer forward wall clock with CUDA synchronization; no profiler or dispatch hooks during timing",
            "instrumented": False,
            "correctness_scope": "full-model input capture and finite isolated-layer output; planner still requires external end-to-end evidence",
        },
        "phase_profiles": phase_profiles,
    }
    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    args.json_out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[layer-profile] {args.json_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
