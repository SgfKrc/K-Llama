#!/usr/bin/env python
"""Autoregressive 21b gate over production request ownership and native TCP ports.

Uses existing proper GGUF segments in place. Normal ngram measurements include
admission, prefill, draft, verification, recovery and final cleanup. Oracle
injections are numerical fault tests and never performance evidence. The probe
uses raw token IDs and reversible ID text, with an out-of-vocabulary EOS sentinel.
"""
from __future__ import annotations
import argparse
from contextlib import ExitStack, contextmanager
from copy import deepcopy
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import platform
import sqlite3
import subprocess
import sys
import time
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
for folder in (ROOT / "src", ROOT / "scripts"):
    if str(folder) not in sys.path: sys.path.insert(0, str(folder))
from comm_multipos_bench import inspect_pair, ProductionLoopback, BenchError
from edge_bench import _expect, _protected_model, _sha256


def injected_drafter(reference, prompt_length, mode, vocabulary):
    """Oracle candidates expose deterministic accept/reject recovery, not speed."""
    injected = False
    def draft(history, limit):
        nonlocal injected
        count = len(history) - prompt_length
        candidates = list(reference[count:count + limit])
        if candidates and mode != "accept" and not injected:
            index = 0 if mode == "reject_first" else min(1, len(candidates) - 1)
            candidates[index] = (candidates[index] + 1) % vocabulary
            injected = True
        return candidates
    return draft


def compare_runs(reference, runs):
    if not reference or not runs:
        raise BenchError("complete reference and nonempty generation evidence required")
    expected = {(gamma, mode) for gamma in (2, 4, 8)
                for mode in ("ngram", "accept", "reject_first", "reject_middle")}
    identities = [(run.get("gamma"), run.get("mode")) for run in runs]
    if len(identities) != len(expected) or set(identities) != expected:
        raise BenchError("complete, unique gamma=2/4/8 and four-mode evidence required")
    mismatches, compared = [], 0
    for run in runs:
        for field in ("tokens", "next_request_tokens"):
            values = run.get(field)
            if not isinstance(values, list) or len(values) != len(reference):
                raise BenchError("incomplete generated sequence: " + field)
            for i, (expected, actual) in enumerate(zip(reference, values)):
                compared += 1
                if actual != expected: mismatches.append({"gamma": run["gamma"], "mode": run["mode"],
                    "field": field, "index": i, "expected": expected, "actual": actual})
        metrics = run.get("speculation", {})
        if run["mode"] in {"reject_first", "reject_middle"} and not metrics.get("replay_calls", 0):
            raise BenchError("rejection injection did not exercise restoration")
        if run["mode"] == "accept" and not metrics.get("accepted_tokens", 0):
            raise BenchError("accept injection did not exercise draft acceptance")
        if run["mode"] == "accept" and metrics.get("replay_calls", 0):
            raise BenchError("all-accept injection unexpectedly required restoration")
        if not run.get("physically_stopped") or not run.get("next_request_stopped"):
            raise BenchError("request cleanup was not confirmed")
    return {"passed": not mismatches, "compared_tokens": compared,
            "mismatch_count": len(mismatches), "mismatches": mismatches}


def build_application(head, tail, wire, segments, declarations, args, journal, trace):
    import numpy as np
    from model_host import ModelHost
    from qlh_core.bootstrap.execution import ExecutionApplication
    from qlh_core.contracts.load import RuntimeLoadSpec
    from qlh_core.execution.requests import RequestService
    from qlh_core.adapters.request_repository import SqliteRequestRepository
    dimension, total = head.n_embd, segments[0]["total_layers"]
    boundary = segments[0]["layer_range"][1]
    vocabulary = tail.n_vocab
    class Tokenizer:
        eos_token_id = vocabulary
        def __call__(self, *_a, **_kw): return {"input_ids": np.array([args.prompt], np.int64)}
        def decode(self, ids, **_kw): return "".join(f"<{i}>" for i in ids)
    def forward(**kw):
        ids = np.asarray(kw["input_ids"]).reshape(-1).tolist()
        position = kw["past_key_values"] or 0
        if position == 0: head.reset()
        hidden = head.forward_tokens_to_hidden(ids, n_past=position)
        trace.append({"kind": "head", "position": position, "tokens": ids})
        return {"hidden_states": hidden, "cache": position + len(ids)}
    def local_reset():
        head.reset(); trace.append({"kind": "reset", "node": "master"})
    descriptor = {"model_type": "qwen2", "total_layers": total, "hidden_size": dimension,
        "model_path": "source_metadata_only.gguf", "loaded_artifact": str(Path(args.head).resolve())}
    manager = SimpleNamespace(layer_range=(0, boundary), engine_type="llama_cpp", tokenizer=Tokenizer(),
        forward_layers=forward, clear_pipeline_kv=local_reset, get_pipeline_descriptor=lambda: deepcopy(descriptor))
    model_identity = {"model_id": declarations[0].get("source_model_id", "q25-native"),
        "engine": "llama_cpp", "format": "gguf", "revision": "comm-batch-v1",
        "sha256": declarations[0]["source_model_sha256"]}
    worker_identity = {"operation_id": "supply_" + "a" * 32, "config_id": "comm-batch-native", "generation": 1,
        "assignment_id": "tail", "artifact_sha256": segments[1]["artifact_sha256"], "hidden_size": dimension,
        "source_model_sha256": declarations[1]["source_model_sha256"],
        "layer_range": [boundary, total], "stage_model_identity": model_identity}
    def spec(components):
        return RuntimeLoadSpec(device="cpu", dtype="artifact", quant_type="artifact", context_size=args.ctx,
            kv_device="cpu", kv_dtype="float16", kv_policy="preallocated", components=tuple(components),
            materialization="gguf_segment", batch_size=args.ctx, sequence_count=1).to_dict()
    local = {"start_layer": 0, "end_layer": boundary, "load_spec": spec(["layers", "embedding"]),
        "identity": {"assignment_id": "master", "layer_range": [0, boundary]}}
    remote = {"start_layer": boundary, "end_layer": total, "load_spec": spec(["layers", "lm_head", "final_norm"]),
        "worker_prepare_identity": worker_identity}
    state = {"active_deployment": "comm-batch-native", "lifecycle": {"total_layers": total},
        "deployments": {"comm-batch-native": {"published": True, "status": "active", "boot_id": "native-boot",
        "authority": {"term": 1}, "connections": {"tail": 1}, "local_assignment": local, "assignments": {"tail": remote}}}}
    def offer(**kw):
        if kw.get("cleanup_only"):
            tail.reset(); trace.append({"kind": "reset", "node": "tail", "stage_id": kw["stage_id"]})
            return {"kind": "cleanup", "kv_cleared": True}
        positions = kw["positions"]
        if positions != list(range(positions[0], positions[0] + len(positions))):
            raise BenchError("noncontiguous production batch")
        multi = kw.get("want_all_tokens", False)
        hidden, sent = wire(kw["hidden_states"], n_past=positions[0], multi=multi)
        if positions[0] == 0: tail.reset()
        options = {"n_past": positions[0], "positions": positions, "seq_ids": [0] * len(positions)}
        tokens = tail.forward_hidden_to_tokens(hidden, **options) if multi else [tail.forward_hidden_to_token(hidden, **options)]
        received = wire.result(tokens, positions, vocabulary, multi=multi)
        trace.append({"kind": "tail", "positions": positions, "tokens": tokens, "multi": multi,
            "stage_id": kw["stage_id"], "request_id": kw["request_id"], "wire": sent, "result": received})
        result = {"token_argmax": tokens[-1]}
        if multi: result.update(token_argmax_by_position=tokens, positions=positions, n_tokens=len(tokens), n_vocab=vocabulary)
        return result
    callbacks = SimpleNamespace(thinking_system_prompt="", build_model_chat_prompt=lambda *_a, **_kw: "raw-id-prompt",
        format_model_response=lambda text, *_a, **_kw: (text, None))
    scheduler = SimpleNamespace(_host=ModelHost(manager=manager),
        _model_control=SimpleNamespace(boot_id="native-boot", snapshot=lambda: deepcopy(state), runnable=lambda *_a, **_kw: True),
        _model_control_authority=lambda: {"term": 1}, _model_control_connections=lambda: {"tail": 1},
        _task_worker_control=SimpleNamespace(worker_snapshot=lambda _n: {"capabilities": {"node_incarnation": "tail-boot",
            "stage_kv_cleanup": True, "multi_position_argmax": True}}),
        get_effective_node_id=lambda: "master", _execute_layer_stage_offer=offer, _require_callbacks=lambda: callbacks)
    def connect():
        connection = sqlite3.connect(journal); connection.row_factory = sqlite3.Row
        return connection
    @contextmanager
    def transaction():
        connection = connect()
        try:
            connection.execute("BEGIN IMMEDIATE"); yield connection; connection.commit()
        except BaseException:
            connection.rollback(); raise
        finally: connection.close()
    requests = RequestService(SqliteRequestRepository(connect=connect, transaction=transaction), boot_id="native-boot")
    return ExecutionApplication(scheduler, requests)


def run_native(args, report):
    if os.name != "nt":
        raise BenchError("native no-copy model protection requires Windows")
    import qlh_core.execution.speculative as implementation
    from llama_keep_head import KeepHeadUpstream, KeepHeadCleanupError, keep_head_multi_position_available
    runners, processes = [], []
    try:
        with ExitStack() as stack:
            files = [stack.enter_context(_protected_model(Path(x).resolve())) for x in (args.head, args.tail)]
            manifests = [stack.enter_context(_protected_model(Path(x).resolve())) for x in (args.head_manifest, args.tail_manifest)]
            segments, declarations = inspect_pair(files, manifests, [_expect(args.head_expect), _expect(args.tail_expect)])
            shim = Path(args.shim).resolve()
            binaries = sorted(shim.parent.glob("*.dll"))
            for f in binaries: stack.enter_context(_protected_model(f))
            before = {str(f): _sha256(f) for f in binaries}
            if not keep_head_multi_position_available(shim): raise BenchError("native multipos finite-logits probe failed")
            report.update(segments=segments, binaries=before, shim=str(shim), finite_logits_probe_passed=True)
            for model in files:
                try:
                    runner = KeepHeadUpstream(shim, model, mode="nextn", n_ctx=args.ctx, n_threads=args.threads,
                        n_batch=args.ctx, n_seq_max=1)
                except KeepHeadCleanupError as exc:
                    # Construction can already own a live child. Retain it even
                    # though no normal runner was returned to this scope.
                    runners.append(exc.cleanup_owner)
                    if getattr(exc.cleanup_owner, "_worker", None) is not None:
                        processes.append(exc.cleanup_owner._worker)
                    raise
                runners.append(runner)
                if runner._worker is None: raise BenchError("native benchmark requires isolated owners")
                processes.append(runner._worker)
                stack.callback(runner.close)
            report["native_owners"] = []
            for runner, segment in zip(runners, segments):
                if (runner.n_embd != segment["actual"]["n_embd"] or runner.n_layer != segment["layers"]
                        or runner.n_vocab != segment["header"]["n_vocab"] or not runner.supports_multi_position_argmax):
                    raise BenchError("native owner does not match verified artifact")
                report["native_owners"].append({"n_embd": runner.n_embd, "n_layer": runner.n_layer,
                    "n_vocab": runner.n_vocab, "observation": runner.materialization_observation(), "pid": runner._worker.pid})
            if any(not 0 <= t < runners[0].n_vocab for t in args.prompt): raise BenchError("prompt outside vocabulary")
            identity = {"model_id": declarations[0].get("source_model_id", "q25-native"), "engine": "llama_cpp",
                "format": "gguf", "revision": "comm-batch-v1", "sha256": declarations[0]["source_model_sha256"]}
            wire = ProductionLoopback(identity, segments[1]["layer_range"]); stack.callback(wire.close)
            trace = []
            app = build_application(*runners, wire, segments, declarations, args, args.json_out.parent / "native-requests.sqlite3", trace)
            stack.callback(app.requests.close)
            original_draft = implementation.draft_ngram
            def run(gamma, mode="ngram", reference=None):
                start_index = len(trace); streamed = []
                implementation.draft_ngram = (original_draft if mode == "ngram" else
                    injected_drafter(reference, len(args.prompt), mode, runners[1].n_vocab))
                try:
                    begin = time.perf_counter()
                    result = app.run("raw-id-prompt", max_new_tokens=args.gen, temperature=0,
                        speculative_gamma=gamma, _request_timeout_seconds=120, _stream_callback=streamed.append)
                    elapsed = time.perf_counter() - begin
                finally: implementation.draft_ngram = original_draft
                if result.get("error"): raise BenchError(str(result))
                metrics = result["metrics"]; tokens = [s["token"] for s in metrics["steps"]]
                if mode != "ngram":
                    metrics["speculation"]["drafter"] = "oracle_injection_" + mode
                if "".join(e["token"] for e in streamed if "token" in e) != result["response"]:
                    raise BenchError("stream differs from committed output")
                stopped = metrics["request"]["execution_state"] == "stopped"
                return {"gamma": gamma, "mode": mode, "tokens": tokens, "elapsed_seconds": elapsed,
                    "committed_tokens_per_second": metrics["tokens_generated"] / elapsed,
                    "speculation": metrics.get("speculation", {}), "physically_stopped": stopped,
                    "performance_evidence": mode == "ngram", "trace_start": start_index, "trace_end": len(trace)}
            baseline = run(1); report["baseline"] = baseline
            reference = baseline["tokens"]; report["runs"] = []
            for mode in ("ngram", "accept", "reject_first", "reject_middle"):
                for gamma in (2, 4, 8):
                    observed = run(gamma, mode, reference)
                    following = run(1)
                    observed["next_request_tokens"] = following["tokens"]
                    observed["next_request_stopped"] = following["physically_stopped"]
                    report["runs"].append(observed)
                    print(f"mode={mode} gamma={gamma} committed={len(observed['tokens'])} "
                          f"replays={observed['speculation'].get('replay_calls')} "
                          f"tps={observed['committed_tokens_per_second']:.3f}", flush=True)
            report["trace"] = trace
            report["gate"] = compare_runs(reference, report["runs"])
            report["binary_integrity_passed"] = before == {str(f): _sha256(f) for f in binaries}
            if not report["binary_integrity_passed"]: raise BenchError("binary changed during run")
    finally:
        errors = []
        for runner in reversed(runners):
            process = getattr(runner, "_worker", None)
            try:
                runner.close()
            except Exception as exc:
                errors.append(f"{type(exc).__name__}: {exc}")
                if process is not None and process.poll() is None:
                    try:
                        process.terminate()
                        try:
                            process.wait(timeout=2)
                        except subprocess.TimeoutExpired:
                            process.kill()
                            process.wait(timeout=2)
                    except Exception as retire_error:
                        errors.append(f"retire: {type(retire_error).__name__}: {retire_error}")
        report["worker_cleanup_errors"] = errors
        report["workers_exited"] = bool(processes) and all(p.poll() is not None for p in processes)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("head", "tail", "head-manifest", "tail-manifest", "shim"):
        parser.add_argument("--" + name, required=True)
    parser.add_argument("--head-expect"); parser.add_argument("--tail-expect")
    parser.add_argument("--ctx", type=int, default=128); parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--gen", type=int, default=32)
    parser.add_argument("--prompt-tokens", default="40,1101,525,264,1588,13,198,32,40,1101,525,264,1588,13,198,32")
    parser.add_argument("--json-out", type=Path, required=True)
    args = parser.parse_args(argv)
    try: args.prompt = [int(v) for v in args.prompt_tokens.split(",")]
    except ValueError: parser.error("prompt tokens must be comma-separated integers")
    if (not args.prompt or args.gen < 4 or args.threads < 1 or args.ctx < len(args.prompt) + args.gen
            or any(t < 0 for t in args.prompt)):
        parser.error("valid prompt, gen>=4, threads>=1 and fitting context required")
    output = args.json_out.resolve()
    inputs = [Path(getattr(args, name)).resolve() for name in ("head", "tail", "head_manifest", "tail_manifest", "shim")]
    if output.suffix != ".json" or output.parent in {p.parent for p in inputs} or output == Path(__file__).resolve():
        parser.error("evidence must be a separate JSON outside model/shim directories")
    output.parent.mkdir(parents=True, exist_ok=True)
    report = {"schema": "qlh.comm_batch_bench.v1", "passed": False,
        "started_at": datetime.now(timezone.utc).isoformat(), "environment": {"platform": platform.platform(), "python": sys.version},
        "invocation": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        "scope": "same host CPU, isolated native owners, injected native ports with production request ownership and TCP codec; raw ID tokenizer, EOS sentinel; no device throughput claim"}
    status = 1
    try:
        sources = ["scripts/comm_batch_bench.py", "scripts/comm_multipos_bench.py", "scripts/comm_wire_bench.py",
            "src/qlh_core/bootstrap/execution.py", "src/qlh_core/execution/speculative.py", "src/qlh_core/execution/linear.py",
            "src/qlh_core/execution/requests.py", "src/qlh_core/contracts/speculation.py", "src/task_worker_protocol.py",
            "src/scheduler_pipeline.py", "src/llama_keep_head.py", "src/llama_keep_head_worker.py"]
        report["source_sha256"] = {p: _sha256(ROOT / p) for p in sources}
        report["git_head"] = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True, cwd=ROOT).strip()
        run_native(args, report)
        report["source_integrity_passed"] = report["source_sha256"] == {p: _sha256(ROOT / p) for p in sources}
        report["passed"] = (report["gate"]["passed"] and report["workers_exited"]
                            and report["binary_integrity_passed"] and report["source_integrity_passed"]
                            and not report["worker_cleanup_errors"])
        status = 0 if report["passed"] else 1
    except Exception as exc:
        report["error"] = f"{type(exc).__name__}: {exc}"
        print(report["error"], file=sys.stderr, flush=True)
    finally:
        report["finished_at"] = datetime.now(timezone.utc).isoformat()
        output.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    return status


if __name__ == "__main__": raise SystemExit(main())
