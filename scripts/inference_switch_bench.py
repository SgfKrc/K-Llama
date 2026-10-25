#!/usr/bin/env python3
"""Explicit local A/B runner; imports never run benchmarks or publish settings.

Examples (choose a prompt that freely generates at least 32 tokens):
  python scripts/inference_switch_bench.py --model MODEL.gguf --prompt TEXT \
    --threads 4,8 --flash-attn off,on --json-out report.json
  python scripts/inference_switch_bench.py --path llama_server --server-exe EXE \
    --model MODEL.gguf --prompt TEXT --calibration-out calibration.json

Default is reporting only. Publication needs both passing evidence and the
explicit --calibration-out flag. Server baseline omits threads/flash flags;
binding baseline uses the production engine's _auto_threads and Llama defaults.
Server requires complete raw logits for finite-logits evidence. Stock native
/completion does not expose them; its cells report an explicit fallback.
Probabilities, including full-vocabulary probabilities, are not a substitute.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import hashlib
import itertools
import re
import json
import math
from pathlib import Path
import sys
import tempfile
import time
import urllib.request
import uuid

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import inference_calibration as calibration
from atomic_json import atomic_write_json


def _csv_threads(text):
    try:
        values = [int(s.strip()) for s in text.split(",")]
        if not values or any(v <= 0 for v in values):
            raise ValueError
        return list(dict.fromkeys(values))
    except ValueError as exc:
        raise argparse.ArgumentTypeError("threads must be positive integers, e.g. 4,8") from exc


def _csv_flash(text):
    values = [s.strip() for s in text.split(",")]
    if not values or any(s not in {"off", "on"} for s in values):
        raise argparse.ArgumentTypeError("flash-attn must be off,on (or either one)")
    return list(dict.fromkeys(s == "on" for s in values))


def _parser():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--path", choices=(calibration.BINDING, calibration.SERVER), default=calibration.BINDING)
    parser.add_argument("--server-exe")
    parser.add_argument("--model", required=True)
    parser.add_argument("--n-ctx", type=int, default=4096)
    parser.add_argument("--n-gpu-layers", type=int, default=None)
    parser.add_argument("--baseline-threads", type=int, default=None, help="binding: production auto (max 8); server: omit flag")
    parser.add_argument("--baseline-flash-attn", choices=("off", "on", "auto"), default=None)
    parser.add_argument("--threads", type=_csv_threads, default=[4, 8])
    parser.add_argument("--flash-attn", type=_csv_flash, default=[False, True])
    parser.add_argument("--n-batch", type=int)
    parser.add_argument("--n-ubatch", type=int)
    parser.add_argument("--n-seq-max", type=int, help="server only; omit for production auto")
    parser.add_argument("--cpu-moe", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--spec-type", default="ngram-mod", help="server only; production default ngram-mod")
    parser.add_argument("--chat-template-kwargs", type=json.loads, default={"enable_thinking": False})
    parser.add_argument("--chat-format", help="binding only; match the production chat format, e.g. chatml")
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--gen", type=int, default=64)
    parser.add_argument("--samples", type=int, default=3)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--json-out")
    parser.add_argument("--calibration-out")
    return parser


def _baseline_kwargs(args):
    binding = args.path == calibration.BINDING
    threads = args.baseline_threads
    if binding and threads is None:
        from llama_engine import LlamaCppEngine

        threads = LlamaCppEngine._auto_threads(None)
    flash = args.baseline_flash_attn
    if binding and flash == "auto":
        raise ValueError("binding flash-attn baseline must be off or on")
    kwargs = {"n_ctx": args.n_ctx, "n_gpu_layers": args.n_gpu_layers if args.n_gpu_layers is not None else (0 if binding else 99),
              "n_threads": threads, "flash_attn": (flash == "on") if flash in {"on", "off"} else (False if binding else None)}
    for name in ("n_batch", "n_ubatch"):
        if getattr(args, name) is not None:
            kwargs[name] = getattr(args, name)
    if binding:
        if args.n_seq_max is not None:
            raise ValueError("binding benchmark uses the normal single-sequence constructor")
        kwargs["verbose"] = False
        if args.chat_format is not None:
            kwargs["chat_format"] = args.chat_format
    else:
        if args.chat_format is not None:
            raise ValueError("--chat-format is binding only; server uses --chat-template-kwargs")
        kwargs.update(cpu_moe=args.cpu_moe, spec_type=args.spec_type,
                      chat_template_kwargs=args.chat_template_kwargs, mmproj_path="", argv_prefix=[])
        if args.n_seq_max is not None:
            kwargs["n_seq_max"] = args.n_seq_max
    return kwargs


def _context(args, kwargs):
    if args.path == calibration.BINDING:
        return calibration.build_binding_context(args.model, load_kwargs=kwargs)
    return calibration.build_server_context(args.model, exe=args.server_exe, load_kwargs=kwargs)


def _empty_run(warmup):
    return {"tokens": [], "finite_logits": True, "prefill_tps": None, "decode_tps": None, "warmup": warmup, "stop": "error"}


def _binding_once(model, prompt, gen, warmup):
    import numpy as np
    import llama_cpp

    run = _empty_run(warmup)
    try:
        model.reset()
        start = time.perf_counter()
        model.eval(prompt)
        duration = time.perf_counter() - start
        run["prefill_tps"] = len(prompt) / duration
        decode_time = 0.0
        decoded = 0
        for index in range(gen):
            pointer = model._ctx.get_logits()
            if not pointer:
                raise ValueError("missing raw logits")
            logits = np.ctypeslib.as_array(pointer, shape=(model.n_vocab(),))
            if not np.isfinite(logits).all():
                run["finite_logits"] = False
                raise ValueError("nonfinite raw logits")
            token = int(np.argmax(logits))
            run["tokens"].append(token)
            if llama_cpp.llama_vocab_is_eog(model._model.vocab, token):
                run["stop"] = "eos"
                break
            if index == gen - 1:
                run["stop"] = "length"
                break
            # Feed the ACTUAL greedy result back. No reference-token replay.
            start = time.perf_counter()
            model.eval([token])
            decode_time += time.perf_counter() - start
            decoded += 1
        run["decode_tps"] = decoded / decode_time if decode_time > 0 else None
        return run, None
    except Exception as exc:
        return run, f"{type(exc).__name__}: {exc}"


def _http_json(base, endpoint, payload=None):
    data = None if payload is None else json.dumps(payload, allow_nan=False).encode("utf-8")
    request = urllib.request.Request(base + endpoint, data=data, headers={"Content-Type": "application/json"})
    # Local private server: bypass ambient HTTP proxies.
    with urllib.request.build_opener(urllib.request.ProxyHandler({})).open(request, timeout=600) as response:
        raw = response.read(16 * 1024 * 1024 + 1)
    if len(raw) > 16 * 1024 * 1024:
        raise ValueError("oversized server metadata")
    return json.loads(raw)


def _raw_logits_finite(values, vocab):
    return (type(values) is list and len(values) == vocab and
            all(type(v) in (int, float) and math.isfinite(v) for v in values))


def _server_once(base, prompt, gen, warmup, vocab):
    run = _empty_run(warmup)
    final, raw_tokens, finite = None, [], True
    try:
        body = {"prompt": prompt, "n_predict": gen, "temperature": -1.0,
                "top_k": 0, "top_p": 1.0, "min_p": 0.0, "repeat_penalty": 1.0,
                "frequency_penalty": 0.0, "presence_penalty": 0.0,
                "samplers": ["temperature"], "seed": 0, "ignore_eos": False,
                "cache_prompt": False, "return_tokens": True, "stream": True,
                "n_probs": 1, "post_sampling_probs": False}
        request = urllib.request.Request(base + "/completion", data=json.dumps(body).encode("utf-8"),
                                         headers={"Content-Type": "application/json"})
        opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
        with opener.open(request, timeout=600) as response:
            while True:
                line = response.readline(64 * 1024 * 1024 + 1)
                if not line:
                    break
                if len(line) > 64 * 1024 * 1024:
                    raise ValueError("oversized native completion event")
                if not line.startswith(b"data: ") or line.strip() == b"data: [DONE]":
                    continue
                chunk = json.loads(line[6:])
                if "error" in chunk:
                    raise ValueError(f"native completion failed: {chunk['error']}")
                tokens = chunk.get("tokens", [])
                if type(tokens) is not list or any(type(t) is not int or t < 0 for t in tokens):
                    raise ValueError("invalid raw token IDs")
                if chunk.get("stop") is True:
                    final = chunk
                    if tokens:
                        if tokens[:len(run["tokens"])] == run["tokens"]:
                            run["tokens"] = tokens
                        else:
                            run["tokens"].extend(tokens)
                else:
                    run["tokens"].extend(tokens)
                # Stock llama-server has no raw_logits field. It can report
                # timings/tokens, but can never produce a passing calibration.
                # A native extension must expose the entire pre-sampling vector
                # for each one-token event; top probabilities prove nothing here.
                if tokens and chunk.get("stop") is not True:
                    if len(tokens) == 1 and _raw_logits_finite(chunk.get("raw_logits"), vocab):
                        raw_tokens.extend(tokens)
                    else:
                        finite = False
        if final is None:
            raise ValueError("native completion ended without final result")
        run["finite_logits"] = finite and bool(raw_tokens) and raw_tokens == run["tokens"]
        if final.get("truncated") is not False or final.get("tokens_cached", 0) != 0:
            raise ValueError("truncated generation or cached prefill")
        if final.get("tokens_predicted") != len(run["tokens"]):
            raise ValueError("incomplete native token sequence (including EOS)")
        stop = final.get("stop_type")
        if stop not in {"eos", "limit"}:
            raise ValueError("native completion did not end normally")
        run["stop"] = "eos" if stop == "eos" else "length"
        timings = final["timings"]
        for phase, count_key, ms_key in (("prefill", "prompt_n", "prompt_ms"), ("decode", "predicted_n", "predicted_ms")):
            count, elapsed = timings[count_key], timings[ms_key]
            if type(count) is not int or count <= 0 or type(elapsed) not in (int, float) or not math.isfinite(elapsed) or elapsed <= 0:
                raise ValueError("invalid native phase timing")
            if phase == "prefill" and count != len(prompt):
                raise ValueError("prefill did not evaluate the complete prompt")
            run[f"{phase}_tps"] = count * 1000.0 / elapsed
        if not run["finite_logits"]:
            return run, "complete raw-logits evidence unavailable/nonfinite; native probabilities are insufficient"
        return run, None
    except Exception as exc:
        run["finite_logits"] = False
        return run, f"{type(exc).__name__}: {exc}"


def _server_spec(args, kwargs):
    from llama_server_process import ServerSpec

    flags = []
    if kwargs.get("n_threads") is not None:
        flags += ["--threads", str(kwargs["n_threads"])]
    if kwargs.get("flash_attn") is not None:
        flags += ["--flash-attn", "on" if kwargs["flash_attn"] else "off"]
    for name, flag in (("n_batch", "--batch-size"), ("n_ubatch", "--ubatch-size"), ("n_seq_max", "--parallel")):
        if kwargs.get(name) is not None:
            flags += [flag, str(kwargs[name])]
    return ServerSpec(model_path=str(Path(args.model).resolve()), exe=str(Path(args.server_exe).resolve()),
                      n_ctx=kwargs["n_ctx"], n_gpu_layers=kwargs["n_gpu_layers"],
                      cpu_moe=kwargs["cpu_moe"], spec_type=kwargs["spec_type"],
                      chat_template_kwargs=kwargs["chat_template_kwargs"], inference_args=flags)


def _measure(args, kwargs):
    cell = {"settings": {k: kwargs[k] for k in ("flash_attn", "n_threads")},
            "context": None, "prompt_tokens": None, "runs": [], "errors": []}
    model = None
    try:
        cell["context"] = _context(args, kwargs)
        if args.path == calibration.BINDING:
            from llama_cpp import Llama

            model = Llama(model_path=args.model, **kwargs)
            prompt = list(model.tokenize(args.prompt.encode("utf-8"), add_bos=True, special=True))
            if not prompt or len(prompt) + args.gen > model.n_ctx():
                raise ValueError("prompt/generation does not fit actual context")
            cell["prompt_tokens"] = prompt
            for index in range(args.warmup + args.samples):
                run, error = _binding_once(model, prompt, args.gen, index < args.warmup)
                cell["runs"].append(run)
                if error:
                    cell["errors"].append({"run": index, "error": error})
        else:
            from llama_server_process import LlamaServerProcess

            with tempfile.TemporaryDirectory(prefix="qlh-inference-bench-") as directory:
                log = Path(directory) / "server.log"
                with LlamaServerProcess(_server_spec(args, kwargs), log_path=log) as server:
                    if cell["context"]["execution"]["device"] == "cuda":
                        text = log.read_text(encoding="utf-8", errors="replace")
                        if not re.search(r"offloaded\s+[1-9]\d*/\d+\s+layers to GPU", text):
                            raise ValueError("server did not attest actual GPU layer offload")
                    prompt = _http_json(server.base_url, "/tokenize", {"content": args.prompt, "add_special": True, "parse_special": True})["tokens"]
                    models = _http_json(server.base_url, "/v1/models")
                    vocab = models["data"][0]["meta"]["n_vocab"]
                    if type(vocab) is not int or not 1 <= vocab <= 1048576:
                        raise ValueError("server vocabulary size unavailable")
                    cell["prompt_tokens"] = prompt
                    for index in range(args.warmup + args.samples):
                        run, error = _server_once(server.base_url, prompt, args.gen, index < args.warmup, vocab)
                        cell["runs"].append(run)
                        if error:
                            cell["errors"].append({"run": index, "error": error})
                cell["server_log"] = log.read_text(encoding="utf-8", errors="replace")[-65536:]
        if _context(args, kwargs) != cell["context"]:
            raise ValueError("context changed during benchmark cell")
    except Exception as exc:
        cell["errors"].append({"error": f"{type(exc).__name__}: {exc}"})
    finally:
        if model is not None:
            try:
                model.close()
            except Exception as exc:
                cell["errors"].append({"error": f"close failed: {exc}"})
    return cell


def _record(args, baseline, candidate):
    return {"record_id": uuid.uuid4().hex, "context": deepcopy(baseline["context"]),
            "baseline": dict(baseline["settings"]), "candidate": dict(candidate["settings"]),
            "protocol": {"warmup": args.warmup, "samples": args.samples, "gen_tokens": args.gen,
                         "prompt_tokens": baseline["prompt_tokens"], "prompt_sha256": hashlib.sha256(args.prompt.encode("utf-8")).hexdigest(),
                         "greedy": True, "free_generation": True},
            "baseline_runs": deepcopy(baseline["runs"]), "candidate_runs": deepcopy(candidate["runs"]), "accepted": False}


def _publish(record, path):
    verdict = calibration.evaluate_record(record)
    if not verdict["settings"]:
        raise ValueError("refusing to publish rejected evidence")
    calibration._check_live(record["context"])
    target = calibration.calibration_path(path)
    store = calibration.read_calibration_store(target) if target.exists() else {"schema_version": 1, "records": []}
    incoming = {**record["baseline"], **record["candidate"]}
    ids = set()
    for previous in store["records"]:
        previous_verdict = calibration.evaluate_record(previous)
        if previous_verdict["reason"].startswith("invalid_evidence") or previous["record_id"] in ids:
            raise ValueError("existing calibration store contains invalid/duplicate evidence")
        ids.add(previous["record_id"])
        if previous["context"] == record["context"]:
            if not previous_verdict["settings"] or {**previous["baseline"], **previous["candidate"]} != incoming:
                raise ValueError("existing same-context evidence conflicts; publication refused")
    if len(store["records"]) >= calibration.MAX_RECORDS:
        raise ValueError("calibration record limit reached")
    store["records"].append(record)
    if len(json.dumps(store, allow_nan=False).encode("utf-8")) > calibration.MAX_STORE_BYTES:
        raise ValueError("calibration store size limit reached")
    target.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_json(target, store, ensure_ascii=True, sort_keys=True)
    return str(target)


def _output_safe(value):
    if isinstance(value, float) and not math.isfinite(value):
        return repr(value)
    if isinstance(value, dict):
        return {k: _output_safe(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_output_safe(v) for v in value]
    return value


def main(argv=None):
    parser = _parser()
    args = parser.parse_args(argv)
    if (args.gen < 32 or args.gen > 65536 or args.samples < 3 or args.warmup < 1
            or args.samples + args.warmup > 1024 or args.n_ctx <= 0
            or (args.baseline_threads is not None and args.baseline_threads <= 0)
            or any(v is not None and v <= 0 for v in (args.n_batch, args.n_ubatch, args.n_seq_max))):
        parser.error("require gen>=32, samples>=3, warmup>=1 and positive context/threads/batches")
    if args.path == calibration.SERVER and not args.server_exe:
        parser.error("--server-exe is required for llama_server")
    inputs = [Path(args.model).resolve()]
    if args.server_exe:
        inputs += [Path(args.server_exe).resolve()]
    outputs = [Path(p).resolve() for p in (args.json_out, args.calibration_out) if p]
    if len(outputs) != len(set(outputs)) or any(p in inputs for p in outputs):
        parser.error("outputs must be distinct and cannot overwrite model/runtime inputs")
    report = {"schema_version": 1, "path": args.path, "baseline": None, "candidates": [],
              "publication": {"requested": bool(args.calibration_out), "path": None, "reason": "report_only"}, "errors": []}
    published = False
    try:
        kwargs = _baseline_kwargs(args)
        context = _context(args, kwargs)
        identities = [context["model"], *context["runtime"]["files"]]
        if any(p == Path(f["resolved_path"]) for p in outputs for f in identities):
            parser.error("output would overwrite a runtime identity file")
        baseline = _measure(args, kwargs)
        report["baseline"] = baseline
        passed = []
        for flash, threads in itertools.product(args.flash_attn, args.threads):
            candidate_kwargs = {**kwargs, "flash_attn": flash, "n_threads": threads}
            cell = _measure(args, candidate_kwargs)
            record = _record(args, baseline, cell)
            verdict = calibration.evaluate_record(record)
            if baseline["prompt_tokens"] != cell["prompt_tokens"]:
                verdict = {"settings": {}, "reason": "prompt_token_mismatch", "ratios": {}}
            if baseline["errors"] or cell["errors"]:
                verdict = {"settings": {}, "reason": "cell_failed", "ratios": {}}
            # Runtime/model/host/device must be identical across all A/B cells.
            if baseline["context"] is None or cell["context"] is None or any(
                    baseline["context"][key] != cell["context"][key]
                    for key in ("path", "model", "runtime", "host", "execution")):
                verdict = {"settings": {}, "reason": "cell_context_mismatch", "ratios": {}}
            record["accepted"] = bool(verdict["settings"])
            report["candidates"].append({"cell": cell, "record": record, "verdict": verdict})
            if verdict["settings"]:
                passed.append((record, verdict))
        if args.calibration_out:
            if passed:
                # Publish exactly one winner; retain every positive and negative
                # candidate in the report without creating a conflicting store.
                record, _ = max(passed, key=lambda pair: (pair[1]["ratios"]["decode"], pair[1]["ratios"]["prefill"], json.dumps(pair[0]["candidate"], sort_keys=True)))
                if _context(args, kwargs) != record["context"]:
                    raise ValueError("baseline context drift before publication")
                report["publication"].update(path=_publish(record, args.calibration_out), reason="published", record_id=record["record_id"])
                published = True
            else:
                report["publication"]["reason"] = "no_passing_candidate"
        success = bool(passed) and (not args.calibration_out or published)
    except Exception as exc:
        report["errors"].append(f"{type(exc).__name__}: {exc}")
        if args.calibration_out:
            report["publication"]["reason"] = "publication_or_benchmark_failed"
        success = False
    report = _output_safe(report)
    if args.json_out:
        path = Path(args.json_out)
        path.parent.mkdir(parents=True, exist_ok=True)
        atomic_write_json(path, report, ensure_ascii=True, sort_keys=True)
    summary = {"path": args.path, "candidates": len(report["candidates"]),
               "passed": sum(bool(v["verdict"]["settings"]) for v in report["candidates"]),
               "verdicts": [{"settings": v["record"]["candidate"], "reason": v["verdict"]["reason"],
                             "errors": v["cell"]["errors"]} for v in report["candidates"]],
               "baseline_errors": report["baseline"]["errors"] if report["baseline"] else [],
               "publication": report["publication"], "errors": report["errors"], "json_out": args.json_out}
    # With a report file, keep stdout small. Without one, preserve all evidence.
    print(json.dumps(summary if args.json_out else report, ensure_ascii=True, indent=2, allow_nan=False))
    return 0 if success else 1


if __name__ == "__main__":
    raise SystemExit(main())
