#!/usr/bin/env python
"""Reproducible CPU GGUF segment forward benchmark (EDGE-BENCH-01).

Verification reads headers only. Benchmarking requires a proper head/middle/tail
segment manifest; each cell runs in a fresh process using the production shim.
No tokenizer, whole-model completion, network or inference fallback is used.

python scripts/edge_bench.py --model segment.gguf --verify-manifest segment.manifest.json --json-out build/edge-bench/verify.json
python scripts/edge_bench.py --model segment.gguf --threads 4,8 --ctx 512,4096 --prompt-tokens 64 --gen 16 --json-out build/edge-bench/report.json
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import ctypes
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import statistics
import tempfile
import shutil
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))
SCHEMA = "qlh.edge_bench.v2"
KEYS = ("block_count", "head_count_kv", "key_length", "full_attention_interval")


class BenchError(ValueError):
    pass


def _load_gguf_meta(path: Path) -> dict:
    from gguf_metadata_contract import read_gguf_metadata, read_gguf_header
    result = read_gguf_metadata(path)
    raw, tensors = read_gguf_header(path)
    if any(key.startswith("split.") for key in raw):
        raise BenchError("split GGUF is unsupported: every native input must be one verified segment file")
    layers = set()
    for name in tensors:
        if name.startswith("blk."):
            try:
                layers.add(int(name.split(".")[1]))
            except (ValueError, IndexError) as exc:
                raise BenchError("malformed layer tensor name") from exc
    if layers != set(range(result["block_count"])):
        raise BenchError("tensor layer indices differ from block_count")
    return result


def _verify(meta: dict, expect: dict, source: str) -> list[str]:
    from gguf_metadata_contract import verify_gguf_metadata
    return [f"{source}: {item}" for item in verify_gguf_metadata(meta, expect)]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _parse_json(data: bytes) -> dict:
    def unique(pairs):
        result = {}
        for name, value in pairs:
            if name in result:
                raise BenchError(f"duplicate JSON key: {name}")
            result[name] = value
        return result
    value = json.loads(data.decode("utf-8-sig"), object_pairs_hook=unique)
    if not isinstance(value, dict):
        raise BenchError("manifest must be an object")
    return value


def _json(path: Path) -> dict:
    return _parse_json(path.read_bytes())


@contextmanager
def _protected_model(path: Path):
    """Keep the bytes verified by preflight immutable through native close."""
    if os.name == "nt":
        from ctypes import wintypes
        kernel = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel.CreateFileW.argtypes = [wintypes.LPCWSTR, wintypes.DWORD, wintypes.DWORD,
                                      ctypes.c_void_p, wintypes.DWORD, wintypes.DWORD, wintypes.HANDLE]
        kernel.CreateFileW.restype = wintypes.HANDLE
        kernel.CloseHandle.argtypes = [wintypes.HANDLE]
        kernel.CloseHandle.restype = wintypes.BOOL
        # FILE_SHARE_READ only: native can read/map; writers/deleters cannot open.
        handle = kernel.CreateFileW(str(path.resolve()), 0x80000000, 1, None, 3, 0x80, None)
        if handle == ctypes.c_void_p(-1).value:
            raise ctypes.WinError(ctypes.get_last_error())
        try:
            yield path
        finally:
            kernel.CloseHandle(handle)
    else:
        # A private copy provides a stable pathname/inode for native loaders on
        # platforms where advisory flock cannot prevent replacement or writes.
        with tempfile.TemporaryDirectory(prefix="qlh-segment-bench-") as private:
            snapshot = Path(private) / "segment.gguf"
            with path.open("rb") as source, snapshot.open("xb") as target:
                shutil.copyfileobj(source, target, 4 * 1024 * 1024)
            snapshot.chmod(0o400)
            yield snapshot


def _expect(text: str | None) -> dict:
    if not text:
        return {}
    # JSON supports per-layer integer arrays and explicit null (absent GGUF key).
    if text.lstrip().startswith("{"):
        result = json.loads(text)
    else:
        result = {}
        for item in text.split(","):
            if "=" not in item:
                raise BenchError(f"invalid --expect item: {item}")
            key, value = item.split("=", 1)
            if key.strip() in result:
                raise BenchError(f"duplicate --expect key: {key}")
            result[key.strip()] = json.loads(value)
    if not isinstance(result, dict) or set(result) - set(KEYS):
        raise BenchError("--expect accepts only the four GGUF metadata keys")
    return result


def _declaration(manifest: dict) -> dict:
    if "gguf_metadata" in manifest:
        value = manifest["gguf_metadata"]
        if not isinstance(value, dict):
            raise BenchError("gguf_metadata must be an object")
        result = dict(value)
    else:
        result = {key: manifest[key] for key in KEYS if key in manifest}
    # Top-level block_count describes the source, not the cropped artifact.
    if "kept_block_count" in manifest:
        kept = manifest["kept_block_count"]
        if "gguf_metadata" in manifest and result.get("block_count") != kept:
            raise BenchError("block_count: gguf_metadata differs from kept_block_count")
        result["block_count"] = kept
    return result


def _inspect(model: Path, manifest_path: Path | None, explicit: dict,
             *, bench: bool, layers: int = 0) -> dict:
    actual = _load_gguf_meta(model)
    manifest_bytes = manifest_path.read_bytes() if manifest_path else b"{}"
    manifest = _parse_json(manifest_bytes)
    expected = _declaration(manifest)
    # An explicit expectation may fill old missing declarations; it never
    # overrides a conflicting declaration to turn a failed check into a pass.
    for key, value in explicit.items():
        if key in expected and (type(expected[key]) is not type(value) or expected[key] != value):
            raise BenchError(f"{key}: --expect conflicts with manifest")
        expected[key] = value
    differences = _verify(actual, expected, str(manifest_path or "--expect"))
    info = {"actual": actual, "expected": expected, "differences": differences,
            "manifest": str(manifest_path) if manifest_path else None}
    if differences or not bench:
        return info
    blocks = actual.get("block_count")
    if type(blocks) is not int or blocks <= 0:
        raise BenchError("block_count: expected a positive integer")
    if layers and layers != blocks:
        raise BenchError(f"block_count: --layers={layers} differs from actual={blocks}")
    if actual.get("nextn_predict_layers", 0):
        raise BenchError("MTP blocks are not supported by this decoder-segment benchmark")
    total, interval = manifest.get("n_layer"), manifest.get("source_layer_range")
    if (type(total) is not int or not isinstance(interval, list) or len(interval) != 2
            or any(type(n) is not int for n in interval)):
        raise BenchError("a native segment manifest with n_layer/source_layer_range is required")
    start, end = interval
    if not (0 <= start < end <= total and end - start == blocks < total):
        raise BenchError("proper segment required: refuse whole-model or inconsistent layer range")
    role = "head" if start == 0 else ("tail" if end == total else "middle")
    if manifest.get("mode") != role or manifest.get("kept_block_count") != blocks:
        raise BenchError("mode/kept_block_count differs from actual segment range")
    if manifest.get("architecture") != actual.get("architecture"):
        raise BenchError("architecture differs from native manifest")
    digest = _sha256(model)
    if manifest.get("artifact_sha256") != digest:
        raise BenchError("artifact_sha256 differs from native manifest")
    info.update(role=role, layers=blocks, layer_range=interval,
                total_layers=total, artifact_sha256=digest,
                manifest_sha256=hashlib.sha256(manifest_bytes).hexdigest())
    return info


def _environment() -> dict:
    out = {"platform": platform.platform(), "python": sys.version,
           "executable": sys.executable, "pid": os.getpid(), "cpu_count": os.cpu_count()}
    try:
        import psutil
        memory = psutil.virtual_memory()
        out.update(available_ram_bytes=memory.available, total_ram_bytes=memory.total,
                   cpu_percent_snapshot=psutil.cpu_percent(interval=0.1))
        out["other_qlh_processes"] = [
            {"pid": p.info["pid"], "command": p.info["cmdline"]}
            for p in psutil.process_iter(["pid", "cmdline"])
            if p.info["pid"] != os.getpid() and any(
                marker in " ".join(p.info["cmdline"] or [])
                for marker in ("llama_keep_head_worker.py", "uvicorn", "api_server"))]
    except (ImportError, OSError) as exc:
        out["resource_snapshot_error"] = str(exc)
    return out


def _bench_one(model_path: Path, *, shim: str, info: dict, n_ctx: int,
               n_threads: int, gen: int, warmup: int, prompt_tokens: int,
               samples: int = 3, token_id: int = 42, hidden_input: str | None = None,
               factory=None, clock=None) -> dict:
    """One cell in the dedicated native process; timings exclude prefill/reset/hash."""
    import numpy as np
    if factory is None:
        from llama_keep_head import KeepHeadUpstream
        factory = KeepHeadUpstream
    clock = clock or time.perf_counter_ns
    target = min(prompt_tokens, n_ctx - gen)
    if target < 1:
        raise BenchError("context must fit at least one prompt position plus all decode steps")
    role = info["role"]
    # Seeded input positions are intentionally not a semantic chat/generation.
    # A supplied capture allows later ISA A/B with exactly the same upstream data.
    load_retries = 0
    while True:
        try:
            runner = factory(shim, model_path, mode="nextn",
                             n_ctx=n_ctx, n_threads=n_threads, n_batch=min(512, n_ctx),
                             n_seq_max=1, _worker_process=True)
            break
        except RuntimeError as exc:
            # LoadLibrary can be temporarily denied by a Windows file scanner.
            # Retry only that pre-allocation failure, with a small fixed bound;
            # model/context initialization failures must never be retried here.
            cause = exc.__cause__
            if (not isinstance(cause, OSError) or getattr(cause, "winerror", None) not in (32, 33)
                    or load_retries >= 3):
                raise
            load_retries += 1
            time.sleep(0.1 * load_retries)
    try:
        if getattr(runner, "_worker", None) is not None:
            raise BenchError("native benchmark must run without nested worker IPC")
        if runner.n_embd != info["actual"]["n_embd"]:
            raise BenchError("native embedding length differs from GGUF header")
        if runner.n_layer != info["layers"]:
            raise BenchError("native block_count differs from header before benchmark")
        lib = runner._lib
        if not hasattr(lib, "qlh_kh_n_vocab") or not hasattr(lib, "qlh_kh_finite_logits_guard"):
            raise BenchError("benchmark requires a shim with vocabulary/finite-logits guards; rebuild shim")
        lib.qlh_kh_n_vocab.argtypes = [ctypes.c_void_p]
        lib.qlh_kh_n_vocab.restype = ctypes.c_int32
        lib.qlh_kh_finite_logits_guard.argtypes = []
        lib.qlh_kh_finite_logits_guard.restype = ctypes.c_int32
        vocab = int(lib.qlh_kh_n_vocab(runner._handle))
        if vocab <= 0 or lib.qlh_kh_finite_logits_guard() != 1:
            raise BenchError("native vocabulary/finite-logits guard is invalid")
        if role == "head" and token_id >= vocab:
            raise BenchError("token-id is outside the native vocabulary")
        if role == "head":
            if hidden_input:
                raise BenchError("--hidden-input is only valid for middle/tail")
            inputs = np.full(target + gen, token_id, dtype=np.int32)
            input_kind = "repeated_token_ids"
        else:
            if hidden_input:
                inputs = np.load(hidden_input, allow_pickle=False)
                if (inputs.dtype != np.float32 or inputs.ndim != 2
                        or inputs.shape != (target + gen, runner.n_embd)
                        or not np.isfinite(inputs).all()):
                    raise BenchError("hidden-input must be finite float32 [prompt_actual+gen,n_embd]")
                inputs = np.ascontiguousarray(inputs)
                input_kind = "captured_upstream_hidden"
            else:
                inputs = np.random.default_rng(1234).standard_normal(
                    (target + gen, runner.n_embd), dtype=np.float32)
                input_kind = "seeded_synthetic_hidden"
        input_sha = hashlib.sha256(inputs.tobytes()).hexdigest()

        def forward(offset, count):
            value = inputs[offset:offset + count]
            if role == "head":
                return runner.forward_tokens_to_hidden(value, n_past=offset)
            method = (runner.forward_hidden_to_token if role == "tail"
                      else runner.forward_hidden_to_hidden)
            return method(value, n_past=offset)

        def validate_output(output, count):
            if role == "tail":
                if type(output) is not int or not 0 <= output < vocab:
                    raise BenchError("native tail returned an invalid argmax")
            elif (not isinstance(output, np.ndarray) or output.dtype != np.float32
                    or output.shape != (count, runner.n_embd) or not np.isfinite(output).all()):
                raise BenchError("native hidden must be finite float32 [count,n_embd]")

        measured = []
        for run in range(warmup + samples):
            runner.reset()
            prefill_start = clock()
            prefill_outputs = []
            for offset in range(0, target, runner.n_batch):
                count = min(runner.n_batch, target - offset)
                prefill_outputs.append((forward(offset, count), count))
            prefill_ms = (clock() - prefill_start) / 1_000_000
            for output, count in prefill_outputs:
                validate_output(output, count)
            del prefill_outputs
            steps, tokens = [], []
            digest = hashlib.sha256()
            for step in range(gen):
                started = clock()
                output = forward(target + step, 1)
                elapsed = (clock() - started) / 1_000_000
                if not math.isfinite(elapsed) or elapsed <= 0:
                    raise BenchError("non-positive or non-finite decode sample")
                validate_output(output, 1)
                steps.append(elapsed)
                if role == "tail":
                    token = int(output)
                    if token < 0:
                        raise BenchError("native tail returned an invalid argmax")
                    tokens.append(token)
                    digest.update(token.to_bytes(8, "little"))
                else:
                    if not np.isfinite(output).all():
                        raise BenchError("native segment returned non-finite hidden")
                    digest.update(np.asarray(output, dtype="<f4").tobytes())
            if run >= warmup:
                measured.append({"prefill_wall_ms": prefill_ms,
                                 "decode_step_ms": steps, "decode_wall_ms": sum(steps),
                                 "output_sha256": digest.hexdigest(), "tail_argmax": tokens})
        ms = statistics.mean(s["decode_wall_ms"] / gen for s in measured)
        stable = len({s["output_sha256"] for s in measured}) == 1
        if not stable:
            raise BenchError("output changed across identical reset samples")
        return {"n_ctx": n_ctx, "n_threads": n_threads, "layers": info["layers"],
                "prompt_tokens_actual": target, "decode_positions_per_sample": gen,
                "decode_ms_per_token": ms, "decode_tokens_per_s": 1000 / ms,
                "ms_per_layer": ms / info["layers"], "samples": measured,
                "input_kind": input_kind, "input_sha256": input_sha,
                "output_stable": stable, "n_batch": runner.n_batch,
                "n_seq_max": 1, "measurement_pid": os.getpid(),
                "native_model_path": str(model_path), "native_n_layer": runner.n_layer,
                "native_n_embd": runner.n_embd, "nested_native_worker": False,
                "native_n_vocab": vocab, "finite_logits_guard": True, "native_load_retries": load_retries}
    finally:
        runner.close()


def _write(path: str | Path, record: dict) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(record, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
                   encoding="utf-8")


def _worker_locked(request: dict, model: Path) -> dict:
    # Recheck declarations/digest in the fresh child before loading native code.
    info = _inspect(model, Path(request["manifest"]), request["explicit"], bench=True,
                    layers=request["layers"])
    if info["differences"]:
        raise BenchError("; ".join(info["differences"]))
    if info["artifact_sha256"] != request["artifact_sha256"]:
        raise BenchError("artifact changed between preflight and measurement")
    if info["manifest_sha256"] != request["manifest_sha256"]:
        raise BenchError("manifest changed between preflight and measurement")
    from relay_segment_info import collect_local_build
    build_before = collect_local_build(shim=request["shim"])
    environment_before = _environment()
    row = _bench_one(model, shim=request["shim"], info=info, **request["cell"])
    build_after = collect_local_build(shim=request["shim"])
    for field in ("shim", "llama_cpp_build"):
        if build_before.get(field) != build_after.get(field):
            raise BenchError("native build changed during measurement")
    row.update(role=info["role"], layer_range=info["layer_range"],
               artifact_sha256=info["artifact_sha256"], manifest_sha256=info["manifest_sha256"])
    row["build"] = build_before
    row["environment_before"] = environment_before
    if _sha256(model) != info["artifact_sha256"]:
        raise BenchError("artifact changed during measurement")
    return {"passed": True, "row": row, "environment": _environment()}


def _worker(request: dict) -> dict:
    with _protected_model(Path(request["model"])) as model:
        return _worker_locked(request, model)


def _validate_cell(result: dict, request: dict, info: dict) -> dict:
    row = result["row"]
    for key, expected in {"n_ctx": request["cell"]["n_ctx"],
                          "n_threads": request["cell"]["n_threads"],
                          "role": info["role"], "layer_range": info["layer_range"],
                          "artifact_sha256": info["artifact_sha256"],
                          "manifest_sha256": info["manifest_sha256"]}.items():
        if row.get(key) != expected:
            raise BenchError(f"cell result identity mismatch: {key}")
    return row


def _positive_list(text: str, name: str) -> list[int]:
    try:
        values = [int(value.strip()) for value in text.split(",")]
    except ValueError as exc:
        raise BenchError(f"{name} must contain positive comma-separated integers") from exc
    if not values or any(value <= 0 for value in values) or len(set(values)) != len(values):
        raise BenchError(f"{name} must contain distinct positive integers")
    return values


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model")
    ap.add_argument("--manifest", help="Native manifest for benchmark (default: model.with_suffix('.manifest.json'))")
    ap.add_argument("--verify-manifest", help="Verify only; no native model load")
    ap.add_argument("--expect", help="Four metadata declarations as JSON or name=value (null means absent)")
    ap.add_argument("--layers", type=int, default=0, help="Optional assertion; never overrides actual layer count")
    ap.add_argument("--threads", default="8")
    ap.add_argument("--ctx", default="512")
    ap.add_argument("--gen", type=int, default=16)
    ap.add_argument("--warmup", type=int, default=1)
    ap.add_argument("--samples", type=int, default=3)
    ap.add_argument("--prompt-tokens", type=int, default=20)
    ap.add_argument("--token-id", type=int, default=42)
    ap.add_argument("--hidden-input")
    ap.add_argument("--shim")
    ap.add_argument("--cell-timeout", type=int, default=600)
    ap.add_argument("--json-out", required=True)
    ap.add_argument("--_cell-request", help=argparse.SUPPRESS)
    args = ap.parse_args(argv)
    record = {"schema": SCHEMA, "passed": False, "rows": []}
    try:
        if args._cell_request:
            record.update(_worker(_json(Path(args._cell_request))))
        else:
            if not args.model:
                raise BenchError("--model is required")
            model = Path(args.model).resolve(strict=True)
            explicit = _expect(args.expect)
            verify = bool(args.verify_manifest or (args.expect and not args.manifest))
            record.update(mode="verify" if verify else "bench", artifact=str(model))
            manifest = (Path(args.verify_manifest or args.manifest).resolve()
                        if args.verify_manifest or args.manifest else
                        (None if verify else model.with_suffix(".manifest.json")))
            if (args.gen <= 0 or args.samples <= 0 or args.prompt_tokens <= 0
                    or args.warmup < 0 or args.layers < 0 or args.token_id < 0 or args.cell_timeout <= 0):
                raise BenchError("gen/samples/prompt-tokens/cell-timeout must be positive; warmup/layers/token-id nonnegative")
            threads, ctxs = _positive_list(args.threads, "threads"), _positive_list(args.ctx, "ctx")
            if not verify and min(ctxs) <= args.gen:
                raise BenchError("every context must fit a prompt plus gen positions")
            info = _inspect(model, manifest, explicit, bench=not verify, layers=args.layers)
            record.update(info)
            if info["differences"]:
                raise BenchError("; ".join(info["differences"]))
            if not verify:
                from keep_head_shim import resolve_keep_head_shim
                from relay_segment_info import collect_local_build
                shim = resolve_keep_head_shim(args.shim)
                if not shim or not Path(shim).is_file():
                    raise BenchError("keep-head shim is unavailable; pass --shim")
                record["build"] = collect_local_build(shim=shim, model=model)
                record["environment_before"] = _environment()
                record["protocol"] = {
                    "gen_tokens": args.gen, "warmup_runs": args.warmup,
                    "prompt_tokens_target": args.prompt_tokens, "samples_per_cell": args.samples,
                    "criterion": "one-position segment forward wall time through native CPU shim; includes Python marshaling",
                    "excluded": ["model load", "reset", "prefill", "input creation", "output hash", "network", "IPC"],
                    "input_semantics": "fixed inputs; segment microbenchmark, not autonomous text generation",
                    "layer_normalization": "actual GGUF blocks; all roles include final norm/vocabulary projection; head also includes embeddings",
                    "comparable_to": "known issue #71 c_D; compare matched role/model/context/input/build; excludes network RTT",
                    "native_process_per_cell": True, "cpu_only": True,
                }
                out = Path(args.json_out).resolve()
                cell_dir = out.with_name(out.stem + "-cells")
                cell_dir.mkdir(parents=True, exist_ok=True)
                for ctx in ctxs:
                    for count in threads:
                        stem = cell_dir / f"ctx-{ctx}-threads-{count}"
                        request = {"model": str(model), "manifest": str(manifest),
                                   "explicit": explicit, "layers": args.layers, "shim": str(Path(shim).resolve()),
                                   "artifact_sha256": info["artifact_sha256"],
                                   "manifest_sha256": info["manifest_sha256"], "cell": {
                                       "n_ctx": ctx, "n_threads": count, "gen": args.gen,
                                       "warmup": args.warmup, "prompt_tokens": args.prompt_tokens,
                                       "samples": args.samples, "token_id": args.token_id,
                                       "hidden_input": str(Path(args.hidden_input).resolve()) if args.hidden_input else None}}
                        request_path, result_path = Path(str(stem) + "-request.json"), Path(str(stem) + "-result.json")
                        _write(request_path, request)
                        with Path(str(stem) + ".log").open("w", encoding="utf-8") as log:
                            child = subprocess.run([sys.executable, "-B", str(Path(__file__).resolve()),
                                                    "--_cell-request", str(request_path), "--json-out", str(result_path)],
                                                   stdout=log, stderr=subprocess.STDOUT, timeout=args.cell_timeout,
                                                   creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0)
                        result = _json(result_path)
                        if child.returncode or not result.get("passed"):
                            raise BenchError(f"cell ctx={ctx} threads={count}: {result.get('error', 'failed')}; see {stem}.log")
                        row = _validate_cell(result, request, info)
                        row["environment"] = result["environment"]
                        record["rows"].append(row)
                        print(f"[bench] ctx={ctx} threads={count}: {row['decode_ms_per_token']:.3f} ms/token; {row['ms_per_layer']:.3f} ms/layer", flush=True)
                        _write(out, record)
                record["environment_after"] = _environment()
            record["passed"] = True
    except Exception as exc:  # CLI boundary: preserve a failed report, including unexpected parser/native errors.
        record["error"] = f"{type(exc).__name__}: {exc}"
        print(f"[FAIL] {record['error']}", file=sys.stderr)
    _write(args.json_out, record)
    return 0 if record["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
