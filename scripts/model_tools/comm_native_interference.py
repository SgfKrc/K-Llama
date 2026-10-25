"""Local native segment interference under real task-worker loopback traffic.

Uses a fresh AVX2/repack-ON receipt from LLAMA-BUILD-ISA-01. The full GGUF is
loaded once and all its materialized layers execute through the real shim.
Use a proper head artifact to measure only a segment. cut_layer selects the
exported hidden boundary; it does not truncate a full model graph.
Communication's executor/admission are injected; this is a CPU contention
experiment, not a distributed model or a product deployment gate.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import ctypes
import hashlib
import json
import os
from pathlib import Path
import statistics
import sys
import threading
import time

ROOT = Path(__file__).resolve().parents[2]
for directory in (ROOT, ROOT / "src", ROOT / "scripts"):
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))

from comm_merge_bench import Loopback, binary_switch
from llama_isa_bench import _validated_build, _model
from scripts.model_tools.sync_llama_cpp import inspect_sync
from llama_keep_head import KeepHeadUpstream

SCHEMA = "qlh.comm_native_interference.v1"
SOURCES = ("src/scheduler_task_worker.py", "src/task_worker_io.py", "src/llama_keep_head.py",
           "src/tcp_comm.py", "src/task_worker_protocol.py", "scripts/comm_merge_bench.py",
           "scripts/model_tools/comm_native_interference.py")


def digest(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def fingerprints():
    return {name: digest(ROOT / name) for name in SOURCES}


def summary(rows):
    return dict(count=len(rows), median_ns=statistics.median(rows), min_ns=min(rows), max_ns=max(rows),
                cv=statistics.stdev(rows) / statistics.mean(rows) if len(rows) > 1 else 0)


@contextmanager
def io_switch(enabled):
    key = "QLH_TASK_WORKER_IO_ISOLATION"
    old = os.environ.get(key)
    os.environ[key] = "1" if enabled else "0"
    try:
        yield
    finally:
        if old is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = old


def loaded_modules(names):
    kernel = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel.GetModuleHandleW.argtypes = [ctypes.c_wchar_p]
    kernel.GetModuleHandleW.restype = ctypes.c_void_p
    kernel.GetModuleFileNameW.argtypes = [ctypes.c_void_p, ctypes.c_wchar_p, ctypes.c_uint]
    paths = {}
    for name in names:
        handle = kernel.GetModuleHandleW(name)
        buffer = ctypes.create_unicode_buffer(32768)
        if not handle or not kernel.GetModuleFileNameW(handle, buffer, len(buffer)):
            raise RuntimeError("missing loaded native dependency: " + name)
        paths[name] = str(Path(buffer.value).resolve())
    return paths


def run(args):
    if os.name != "nt" or not 5 <= args.samples <= 500 or not 1 <= args.rounds <= 6:
        raise ValueError("Windows and samples 5..500 / rounds 1..6 required")
    if not 1 <= args.threads <= 64 or args.mask <= 0 or args.mask.bit_count() < args.threads:
        raise ValueError("invalid native affinity or thread count")
    if not 4 <= args.hidden_bytes <= 4 * 1024 * 1024 or args.hidden_bytes % 4:
        raise ValueError("traffic hidden bytes must be a multiple of 4 within 4MiB")
    source = fingerprints()
    sync = inspect_sync(android_root=ROOT / "android", dry_run=True)
    if sync["state"] != "applied" or any(p["state"] != "applied_committed" for p in sync["patches"]):
        raise RuntimeError("committed applied source required")
    build = _validated_build(args.build, "avx2", sync)
    metadata, dimensions = _model(args.model)
    if not 1 <= args.cut_layer <= dimensions["n_layer"]:
        raise ValueError("invalid measured layer boundary")
    bound_paths = list(build["paths"]) + [args.model.resolve()]
    bound_sha = {str(path): digest(path) for path in bound_paths}
    kernel = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel.GetCurrentProcess.restype = ctypes.c_void_p
    kernel.SetProcessAffinityMask.argtypes = [ctypes.c_void_p, ctypes.c_size_t]
    kernel.GetProcessAffinityMask.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_size_t), ctypes.POINTER(ctypes.c_size_t)]
    process = kernel.GetCurrentProcess()
    previous, system = ctypes.c_size_t(), ctypes.c_size_t()
    if not kernel.GetProcessAffinityMask(process, ctypes.byref(previous), ctypes.byref(system)):
        raise RuntimeError("cannot read affinity")
    if not kernel.SetProcessAffinityMask(process, args.mask):
        raise RuntimeError("cannot set affinity")
    native = None
    rows = []
    try:
        export_mode = "nextn" if args.cut_layer == dimensions["n_layer"] else "layer_inp"
        native = KeepHeadUpstream(Path(build["out_dir"]) / "qlh_keep_head.dll", args.model,
                                  mode=export_mode, cut_layer=args.cut_layer, n_ctx=256,
                                  n_threads=args.threads, n_batch=32, _worker_process=True)
        actual = loaded_modules([item["name"] for item in build["receipt"]["artifacts"]])
        for name, path in actual.items():
            if Path(path) != (Path(build["out_dir"]) / name).resolve():
                raise RuntimeError("loaded native dependency differs from receipt: " + name)
        tokens = [40, 1101, 525, 264, 1588, 13, 198, 32] * 2
        def forward():
            native.reset()
            start = time.perf_counter_ns()
            out = native.forward_tokens_to_hidden(tokens)
            return time.perf_counter_ns() - start, hashlib.sha256(out.tobytes()).hexdigest()
        _, reference = forward()
        for _ in range(3):
            assert forward()[1] == reference
        conditions = [("idle", False, False), ("legacy-json", False, False),
                      ("isolated-json", True, False), ("isolated-binary", True, True),
                      ("legacy-binary", False, True)]
        for round_index in range(args.rounds):
            order = conditions if round_index % 2 == 0 else list(reversed(conditions))
            for name, isolated, binary in order:
                stop, first = threading.Event(), threading.Event()
                traffic_rows, errors = [], []
                link, thread = None, None
                with io_switch(isolated), binary_switch(binary):
                    try:
                        if name != "idle":
                            link = Loopback(binary=binary, compute_delay_ms=0, master_accept_delay_ms=0, timeout=15)
                            raw = b"\x00\x00\x80\x3f" * (args.hidden_bytes // 4)
                            def traffic():
                                try:
                                    index = 0
                                    while not stop.is_set():
                                        sample = link.sample(index, raw)
                                        traffic_rows.append(dict(execute_ns=sample["execute_total_ns"],
                                                                 start_ns=sample["events"][0]["ns"],
                                                                 end_ns=sample["events"][-1]["ns"]))
                                        first.set()
                                        index += 1
                                except BaseException as exc:
                                    errors.append(type(exc).__name__ + ": " + str(exc))
                                    first.set()
                            thread = threading.Thread(target=traffic, name="native-interference-traffic", daemon=True)
                            thread.start()
                            if not first.wait(15) or errors:
                                raise RuntimeError("traffic startup failed: " + str(errors))
                        measurements, hashes = [], []
                        cpu_start, wall_start = time.process_time_ns(), time.perf_counter_ns()
                        for _ in range(args.samples):
                            elapsed, output_sha = forward()
                            measurements.append(elapsed)
                            hashes.append(output_sha)
                        wall_end, cpu_end = time.perf_counter_ns(), time.process_time_ns()
                    finally:
                        stop.set()
                        if thread is not None:
                            thread.join(20)
                        sender = getattr(getattr(link, "worker", None), "_task_worker_io_sender", None)
                        if link is not None:
                            link.close()
                        if sender is not None:
                            sender.close(wait=True)
                        if thread is not None and thread.is_alive():
                            raise RuntimeError("traffic thread did not stop")
                    if errors:
                        raise RuntimeError(str(errors))
                    overlap = sum(r["start_ns"] < wall_end and r["end_ns"] > wall_start for r in traffic_rows)
                    if name != "idle" and overlap < 1:
                        raise RuntimeError("no observed compute/traffic time overlap")
                    if any(value != reference for value in hashes):
                        raise RuntimeError("native fixed hidden changed during traffic")
                    rows.append(dict(condition=name, round=round_index, summary=summary(measurements),
                                     native_forward_ns=measurements, process_cpu_ns=cpu_end-cpu_start,
                                     measurement_wall_ns=wall_end-wall_start,
                                     traffic_calls=len(traffic_rows), overlapping_calls=overlap,
                                     traffic=traffic_rows, output_sha256=reference,
                                     sender=None if sender is None else sender.snapshot(),
                                     cleanup=None if link is None else link.cleanup))
    finally:
        try:
            if native is not None:
                native.close()
        finally:
            kernel.SetProcessAffinityMask(process, previous.value)
    if source != fingerprints() or any(digest(path) != sha for path, sha in bound_sha.items()):
        raise RuntimeError("source/model/native input changed during capture")
    return dict(schema=SCHEMA, passed=True, product_benefit_proven=False,
                scope="same-process ctypes native all materialized GGUF layers; cut_layer selects hidden export, not graph truncation; injected communication executor/admission over real loopback TCP; shared affinity; no device/product claim",
                config=dict(samples=args.samples, rounds=args.rounds, threads=args.threads, affinity_mask=args.mask,
                            executed_local_layer_range=[0,dimensions["n_layer"]], hidden_export_boundary=args.cut_layer, shim_mode=export_mode, native_tokens=tokens, traffic_hidden_bytes=args.hidden_bytes),
                model=dict(path=str(args.model.resolve()), metadata=metadata, dimensions=dimensions),
                source_sha256=source, input_sha256=bound_sha, loaded_modules=actual, rows=rows,
                aggregate={name:summary([v for row in rows if row["condition"]==name for v in row["native_forward_ns"]])
                           for name,_,_ in conditions})


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--cut-layer", type=int, default=4)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--mask", type=lambda value:int(value,0), default=0x55)
    parser.add_argument("--samples", type=int, default=31)
    parser.add_argument("--rounds", type=int, default=2)
    parser.add_argument("--hidden-bytes", type=int, default=262144)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        report = run(args)
    except Exception as exc:
        report = dict(schema=SCHEMA, passed=False, product_benefit_proven=False, error=type(exc).__name__+": "+str(exc))
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(report,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    print(json.dumps({k:v for k,v in report.items() if k in ("passed","error","aggregate")}))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
