#!/usr/bin/env python3
"""Compare pinned CPU repack kernels in isolated native processes.

This measures an existing vendor mechanism. Repack is already enabled in the
production baseline; on/off speedups are not gains over the shipping engine.
"""
from __future__ import annotations

import argparse
from contextlib import ExitStack
from datetime import datetime, timezone
import importlib.util
import json
import math
import os
from pathlib import Path
import platform
import statistics
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
for folder in (ROOT, ROOT / "src", ROOT / "scripts"):
    if str(folder) not in sys.path:
        sys.path.insert(0, str(folder))
from edge_bench import BenchError, _protected_model, _sha256
from gguf_metadata_contract import read_gguf_header, normalize_gguf_metadata
from scripts.model_tools.sync_llama_cpp import inspect_sync

SCHEMA = "qlh.llama_kernel_bench.v1"
NATIVE_SCHEMA = "qlh.llama_kernel_probe.v1"
PROMPT = "40,1101,525,264,1588,13,198,32,40,1101,525,264,1588,13,198,32"
FIXED = "785,1633,315,220,17,10,17,374,220,19,13,198,791,1920,374,264,2311,311,1492,13,198,40,1101,525,264,1588,13,198,32,25,220,19"


def _write(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def _tokens(text):
    try:
        values = [int(item) for item in text.split(",")]
    except (AttributeError, ValueError) as exc:
        raise BenchError("tokens must be a nonempty comma-separated integer list") from exc
    if not values or any(n < 0 or n > 2**31 - 1 for n in values) or len(values) > 4096:
        raise BenchError("tokens must contain 1..4096 nonnegative int32 IDs")
    return values


def _cache(path):
    values = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if line and not line.startswith(("#", "//")) and ":" in line and "=" in line:
            key, value = line.split("=", 1)
            values[key.split(":", 1)[0]] = value
    return values


def _require_pinned_platform():
    """本实验的固定基线只在 Windows x64 上成立。"""
    if os.name != "nt" or platform.machine().lower() not in {"amd64", "x86_64"}:
        raise BenchError("this pinned AVX2 experiment requires Windows x64")


def _preflight(args):
    # 平台门单独成函数：os.name 是只读属性、monkeypatch 改不动，测试要单独验证
    # 它**之后**的判定（补丁状态、CMake cache 基线）就只能替换这个函数。
    _require_pinned_platform()
    source = ROOT / "android/app/src/main/cpp/llama.cpp"
    sync = inspect_sync(android_root=ROOT / "android", dry_run=True)
    if sync["state"] != "applied" or any(p["state"] != "applied_committed" for p in sync["patches"]):
        raise BenchError("kernel experiment requires committed, applied vendor patches")
    build = args.build_root.resolve()
    cache_path = build / "CMakeCache.txt"
    cache = _cache(cache_path)
    expected = {"GGML_CPU": "ON", "GGML_NATIVE": "OFF", "GGML_AVX2": "ON",
                "GGML_AVX512": "OFF", "GGML_AVX_VNNI": "OFF", "GGML_CUDA": "OFF",
                "GGML_VULKAN": "OFF", "GGML_CPU_REPACK": "ON"}
    if any(cache.get(k) != v for k, v in expected.items()):
        raise BenchError("actual CMake cache does not match the fixed CPU AVX2/repack baseline")
    if Path(cache.get("CMAKE_HOME_DIRECTORY", "")).resolve() != source.resolve():
        raise BenchError("CMake cache names a different source tree")
    compiler = Path(cache.get("CMAKE_CXX_COMPILER", ""))
    if not compiler.is_file():
        raise BenchError("the configured C++ compiler is missing")
    spec = importlib.util.spec_from_file_location("qlh_kernel_pe", ROOT / "packaging/packaging/keep_head_resources.py")
    pe = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(pe)
    # The isolated helper instance validates the explicitly selected build group.
    pe.NATIVE_RELATIVE = build / "bin"
    closure = pe.collect_keep_head_binaries(ROOT)
    binaries = [Path(item[1]) for item in closure]
    model = args.model.resolve(strict=True)
    raw, tensors = read_gguf_header(model)
    meta = normalize_gguf_metadata(raw)
    if any(k.startswith("split.") for k in raw):
        raise BenchError("split GGUF is unsupported")
    if not tensors or not any(entry[3] == 12 for entry in tensors.values()):
        raise BenchError("the Q4_K experiment requires actual Q4_K tensors")
    if meta["nextn_predict_layers"] not in (None, 0):
        raise BenchError("MTP tensors require a separate numerical experiment")
    if meta["architecture"] not in {"qwen2", "qwen35", "llama", "qwen3", "qwen3moe", "qwen35moe", "qwen3next", "gemma4", "openai-moe"}:
        raise BenchError("architecture has no registered final layer-input boundary")
    vocabulary_tensor = tensors.get("output.weight", tensors.get("token_embd.weight"))
    if vocabulary_tensor is None or len(vocabulary_tensor[2]) != 2:
        raise BenchError("cannot bind vocabulary to the actual model tensors")
    dimensions = {"n_embd": meta["n_embd"], "n_layer": meta["block_count"], "n_vocab": vocabulary_tensor[2][1]}
    if any(type(value) is not int or value <= 0 for value in dimensions.values()):
        raise BenchError("invalid actual model dimensions")
    tensor_types = {}
    for _, length, _, kind in tensors.values():
        record = tensor_types.setdefault(str(kind), {"tensors": 0, "bytes": 0})
        record["tensors"] += 1
        record["bytes"] += length
    lock_path = ROOT / "scripts/model_tools/llama_quantize.lock.json"
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    paths = [model, cache_path, compiler, source / "include/llama.h", lock_path,
             ROOT / "scripts/model_tools/sync_llama_cpp.py", Path(__file__).resolve(),
             ROOT / "scripts/model_tools/llama_kernel_probe.cpp", build / "src/libllama.dll.a", *binaries]
    paths.extend(ROOT / patch["path"] for patch in lock["patches"])
    return {"model": str(model), "metadata": meta, "dimensions": dimensions, "tensor_types": tensor_types,
            "source_sync": sync, "cache": {k: cache[k] for k in expected},
            "build_root": str(build), "compiler": str(compiler),
            "paths": paths}


def _command(argv, directory, stem, *, environment, timeout):
    start = time.monotonic()
    try:
        result = subprocess.run(argv, cwd=ROOT, env=environment, stdin=subprocess.DEVNULL,
                                capture_output=True, text=True, encoding="utf-8", errors="replace",
                                timeout=timeout, shell=False,
                                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
    except (OSError, subprocess.TimeoutExpired) as exc:
        for stream in ("stdout", "stderr"):
            partial = getattr(exc, stream, None) or ""
            if isinstance(partial, bytes):
                partial = partial.decode("utf-8", errors="replace")
            (directory / (stem + "." + stream + ".log")).write_text(partial, encoding="utf-8")
        _write(directory / (stem + ".command.json"), {"argv": argv, "error": str(exc),
               "returncode": 124 if isinstance(exc, subprocess.TimeoutExpired) else None,
               "elapsed_seconds": time.monotonic() - start})
        raise BenchError(f"{stem} did not finish: {exc}") from exc
    (directory / (stem + ".stdout.log")).write_text(result.stdout, encoding="utf-8")
    (directory / (stem + ".stderr.log")).write_text(result.stderr, encoding="utf-8")
    _write(directory / (stem + ".command.json"), {"argv": argv, "returncode": result.returncode,
           "elapsed_seconds": time.monotonic() - start})
    if result.returncode:
        raise BenchError(f"{stem} failed with exit code {result.returncode}; see raw logs")
    return result


def _check_native(data, *, policy, rows, greedy, repeats, dimensions, hidden, request):
    import numpy as np
    if not isinstance(data, dict) or data.get("schema") != NATIVE_SCHEMA or data.get("ok") is not True:
        raise BenchError("native report lacks a successful matching schema")
    if data.get("requested_repack") is not bool(policy):
        raise BenchError("native repack identity differs")
    if data.get("default_repack") is not True:
        raise BenchError("runtime default differs from the pinned repack baseline")
    for key in ("n_ctx", "n_batch", "n_ubatch", "n_seq_max", "n_threads", "n_threads_batch"):
        if type(data.get(key)) is not int or data[key] <= 0:
            raise BenchError(f"invalid actual native {key}")
    if (data["n_ctx"] < request["ctx"] or data["n_batch"] < len(request["prompt"])
            or not len(request["prompt"]) <= data["n_ubatch"] <= data["n_batch"]
            or data["n_seq_max"] != 1 or data["n_threads"] != request["threads"]
            or data["n_threads_batch"] != request["threads"]):
        raise BenchError("actual native configuration differs from the requested experiment")
    for key, expected in (("prompt_tokens", request["prompt"]), ("fixed_tokens", request["fixed"])):
        if data.get(key) != expected or any(type(n) is not int for n in data[key]):
            raise BenchError("native input token echo differs")
    for key in ("load_seconds", "context_seconds"):
        if type(data.get(key)) not in (int, float) or not math.isfinite(data[key]) or data[key] <= 0:
            raise BenchError(f"invalid native {key}")
    for key in ("n_embd", "n_layer", "n_vocab"):
        if type(data.get(key)) is not int or data[key] <= 0 or data[key] != dimensions[key]:
            raise BenchError(f"native {key} differs from the actual GGUF header")
    for key, length in (("fixed_argmax", rows), ("greedy_tokens", greedy)):
        values = data.get(key)
        if not isinstance(values, list) or len(values) != length or any(type(n) is not int or not 0 <= n < data["n_vocab"] for n in values):
            raise BenchError(f"native {key} is incomplete or outside vocabulary")
    for key, length in (("fixed_margins", rows), ("prefill_seconds", repeats), ("decode_seconds", repeats)):
        values = data.get(key)
        if (not isinstance(values, list) or len(values) != length
                or any(type(n) not in (int, float) or not math.isfinite(n) or n < 0 for n in values)
                or (key.endswith("seconds") and any(n <= 0 for n in values))):
            raise BenchError(f"native {key} is incomplete or nonfinite")
    if data.get("hidden_rows") != rows or data.get("hidden_columns") != data["n_embd"]:
        raise BenchError("native hidden shape differs")
    if hidden.stat().st_size != rows * data["n_embd"] * 4:
        raise BenchError("native hidden byte count differs")
    values = np.fromfile(hidden, dtype="<f4").reshape(rows, data["n_embd"])
    if not np.isfinite(values).all():
        raise BenchError("native hidden contains NaN or Inf")
    return values


def _compare(runs, prompt_count, fixed_count):
    import numpy as np
    if not runs or {r["policy"] for r in runs} != {0, 1}:
        raise BenchError("comparison requires both CPU buffer policies")
    first = runs[0]
    config_keys = ("n_ctx", "n_batch", "n_ubatch", "n_seq_max", "n_threads", "n_threads_batch")
    if any(any(run["result"][key] != first["result"][key] for key in config_keys) for run in runs):
        raise BenchError("cannot compare different effective native configurations")
    numerical = {"passed": True, "fixed_argmax_equal": True, "greedy_equal": True,
                 "hidden_close": True, "hidden_atol": 1e-4, "hidden_rtol": 1e-4,
                 "hidden_max_absolute_difference": 0.0}
    details = []
    for run in runs:
        current, reference = run["result"], first["result"]
        mismatches = [{"position": i, "reference_token": a, "token": b,
                       "reference_margin": reference["fixed_margins"][i], "margin": current["fixed_margins"][i]}
                      for i, (a, b) in enumerate(zip(reference["fixed_argmax"], current["fixed_argmax"])) if a != b]
        details.append({"name": run.get("name"), "policy": run["policy"], "argmax_mismatches": mismatches,
                        "hidden_max_abs": float(np.max(np.abs(run["hidden"] - first["hidden"]))),
                        "hidden_relative_l2": float(np.linalg.norm(run["hidden"] - first["hidden"]) /
                                                    max(np.linalg.norm(first["hidden"]), np.finfo(np.float32).tiny))})
    numerical["per_run_against_reference"] = details
    repeatable = True
    for policy in (0, 1):
        selected = [r for r in runs if r["policy"] == policy]
        ref = selected[0]
        repeatable &= all(r["result"]["fixed_argmax"] == ref["result"]["fixed_argmax"]
                          and r["result"]["greedy_tokens"] == ref["result"]["greedy_tokens"]
                          and np.array_equal(r["hidden"], ref["hidden"]) for r in selected)
    numerical["same_policy_bitwise_repeatable"] = bool(repeatable)
    for run in runs[1:]:
        numerical["fixed_argmax_equal"] &= run["result"]["fixed_argmax"] == first["result"]["fixed_argmax"]
        numerical["greedy_equal"] &= run["result"]["greedy_tokens"] == first["result"]["greedy_tokens"]
        delta = np.abs(run["hidden"] - first["hidden"])
        numerical["hidden_max_absolute_difference"] = max(numerical["hidden_max_absolute_difference"], float(delta.max()))
        numerical["hidden_close"] &= bool(np.allclose(run["hidden"], first["hidden"], atol=1e-4, rtol=1e-4))
    numerical["passed"] = all(numerical[k] for k in ("fixed_argmax_equal", "greedy_equal", "hidden_close", "same_policy_bitwise_repeatable"))
    performance = {}
    for policy, label in ((0, "plain_cpu"), (1, "repacked_cpu")):
        selected = [r for r in runs if r["policy"] == policy]
        pp = [prompt_count / t for r in selected for t in r["result"]["prefill_seconds"]]
        tg = [fixed_count / t for r in selected for t in r["result"]["decode_seconds"]]
        performance[label] = {"prefill_tokens_per_second": pp, "decode_tokens_per_second": tg,
                              "prefill_median": statistics.median(pp), "decode_median": statistics.median(tg)}
    performance["repack_over_plain_prefill"] = performance["repacked_cpu"]["prefill_median"] / performance["plain_cpu"]["prefill_median"]
    performance["repack_over_plain_decode"] = performance["repacked_cpu"]["decode_median"] / performance["plain_cpu"]["decode_median"]
    observed = all(r["repack_buffer_observed"] for r in runs if r["policy"] == 1) and not any(r["repack_buffer_observed"] for r in runs if r["policy"] == 0)
    return {"numerical": numerical, "performance": performance, "repack_buffer_gate_passed": observed,
            "production_default": "repacked_cpu", "new_migration_gain_proven": False,
            "performance_eligible_for_admission": bool(numerical["passed"] and observed),
            "scope": "same DLLs/model/threads, fixed-input native prefill/decode; greedy parity separately. Existing repack on/off is not a new gain over production or end-to-end product throughput."}


def run(args, report):
    if not (1 <= args.threads <= 128 and 1 <= args.repeats <= 20 and 2 <= args.rounds <= 10
            and 1 <= args.greedy_steps <= 4096 and 1 <= args.timeout <= 1800):
        raise BenchError("threads/repeats/rounds/greedy/timeout outside supported bounds")
    prompt, fixed = _tokens(args.prompt_tokens), _tokens(args.fixed_tokens)
    if not len(prompt) + max(len(fixed), args.greedy_steps) <= args.ctx <= 32768:
        raise BenchError("context cannot hold the complete fixed and greedy sequences")
    if args.out_dir.exists() and (not args.out_dir.is_dir() or any(args.out_dir.iterdir())):
        raise BenchError("choose a fresh empty output directory")
    preflight = _preflight(args)
    if any(n >= preflight["dimensions"]["n_vocab"] for n in prompt + fixed):
        raise BenchError("token ID exceeds actual vocabulary")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    paths = preflight.pop("paths")
    with ExitStack() as stack:
        for path in dict.fromkeys(paths):
            stack.enter_context(_protected_model(path))
        bindings = {str(path.resolve()): _sha256(path) for path in paths}
        report["preflight"] = preflight
        report["input_sha256"] = bindings
        snapshots = {}
        for path in paths:
            if path.is_relative_to(ROOT) and path.suffix.lower() in {".py", ".cpp", ".c", ".h", ".patch", ".json", ".txt"}:
                target = args.out_dir / "source-snapshots" / path.relative_to(ROOT)
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(path.read_bytes())
                snapshots[str(path.resolve())] = target.relative_to(args.out_dir).as_posix()
        report["source_snapshots"] = snapshots
        report["invocation"] = {"threads": args.threads, "ctx": args.ctx, "repeats": args.repeats,
                                "rounds": args.rounds, "prompt": prompt, "fixed": fixed, "greedy_steps": args.greedy_steps}
        if args.verify_only:
            report.update(ok=True, status="verified_only", native_executed=False)
            return
        environment = {k: v for k, v in os.environ.items() if not k.upper().startswith("GIT_")}
        binary_dir = Path(preflight["build_root"]) / "bin"
        compiler_dir = str(Path(preflight["compiler"]).parent)
        environment["PATH"] = os.pathsep.join((str(binary_dir), compiler_dir, environment.get("PATH", "")))
        probe = args.out_dir.resolve() / "llama-kernel-probe.exe"
        source = ROOT / "android/app/src/main/cpp/llama.cpp"
        argv = [preflight["compiler"], "-std=c++17", "-O2", "-DNDEBUG",
                str(ROOT / "scripts/model_tools/llama_kernel_probe.cpp"),
                "-I" + str(source / "include"), "-I" + str(source / "ggml/include"),
                "-L" + str(Path(preflight["build_root"]) / "src"), "-lllama", "-o", str(probe)]
        _command(argv, args.out_dir, "compile", environment=environment, timeout=args.timeout)
        stack.enter_context(_protected_model(probe))
        report["probe_sha256"] = _sha256(probe)
        dimensions = preflight["dimensions"]
        runs = []
        report["runs"] = []
        for round_index in range(args.rounds):
            for policy in ((0, 1) if round_index % 2 == 0 else (1, 0)):
                name = f"round-{round_index + 1}-repack-{policy}"
                hidden = args.out_dir.resolve() / (name + ".hidden.f32")
                argv = [str(probe), preflight["model"], str(policy), str(args.threads), str(args.ctx),
                        args.prompt_tokens, args.fixed_tokens, str(args.greedy_steps), str(args.repeats), str(hidden)]
                print(f"[run] {name}", flush=True)
                result = _command(argv, args.out_dir, name, environment=environment, timeout=args.timeout)
                try:
                    data = json.loads(result.stdout)
                except (ValueError, UnicodeError) as exc:
                    raise BenchError("native stdout is not one JSON report") from exc
                values = _check_native(data, policy=policy, rows=len(prompt) + len(fixed), greedy=args.greedy_steps,
                                       repeats=args.repeats, dimensions=dimensions, hidden=hidden, request=report["invocation"])
                observed = any("CPU_REPACK" in line and "model buffer size" in line for line in result.stderr.splitlines())
                entry = {"name": name, "policy": policy, "result": data, "hidden_sha256": _sha256(hidden),
                         "repack_buffer_observed": observed}
                report["runs"].append(entry)
                runs.append({**entry, "hidden": values})
                _write(args.out_dir / "report.json", report)
        report["comparison"] = _compare(runs, len(prompt), len(fixed))
        if any(_sha256(Path(path)) != sha for path, sha in bindings.items()):
            raise BenchError("a bound experiment input changed")
        if inspect_sync(android_root=ROOT / "android", dry_run=True) != preflight["source_sync"]:
            raise BenchError("vendor source contract changed during the experiment")
        passed = report["comparison"]["numerical"]["passed"] and report["comparison"]["repack_buffer_gate_passed"]
        report.update(ok=passed, status="passed" if passed else "comparison_failed", native_executed=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--build-root", type=Path, default=ROOT / "build/keephead/build-cpu")
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--ctx", type=int, default=256)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--rounds", type=int, default=2)
    parser.add_argument("--greedy-steps", type=int, default=32)
    parser.add_argument("--timeout", type=int, default=300)
    parser.add_argument("--prompt-tokens", default=PROMPT)
    parser.add_argument("--fixed-tokens", default=FIXED)
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args(argv)
    report = {"schema": SCHEMA, "ok": False, "status": "failed",
              "started_at": datetime.now(timezone.utc).isoformat(),
              "environment": {"platform": platform.platform(), "processor": platform.processor(), "logical_cpus": os.cpu_count(),
                              "python": sys.version, "executable": sys.executable}}
    existed = args.out_dir.exists()
    try:
        run(args, report)
    except (BenchError, OSError, RuntimeError, ValueError) as exc:
        report["error"] = str(exc)
        if existed and "preflight" not in report:
            print(json.dumps(report, ensure_ascii=False, allow_nan=False))
            return 1
    if args.out_dir.is_dir():
        _write(args.out_dir / "report.json", report)
    print(json.dumps({k: v for k, v in report.items() if k not in {"runs", "input_sha256"}}, ensure_ascii=False, indent=2, allow_nan=False))
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
