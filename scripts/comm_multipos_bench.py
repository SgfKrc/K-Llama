#!/usr/bin/env python
"""21b-0 full-attention multi-position argmax numerical acceptance (Windows).

Uses existing, contiguous proper head/tail GGUF segments in place. Each gamma
replays the SAME fixed token stream from reset + a nonempty, identical prefix.
The reference recomputes head(token)->hidden->tail(argmax) one token at a time;
it never slices bulk hidden into a purported reference. Timings are observations
for this numerical probe, not a throughput or performance claim.

Requires an explicitly selected, freshly built shim with the multi-position API.
No model conversion, downloads, tokenizer binding, or model copies are performed.
"""
from __future__ import annotations

import argparse
from contextlib import ExitStack
import ctypes
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import re
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
for folder in (ROOT / "src", ROOT / "scripts"):
    if str(folder) not in sys.path:
        sys.path.insert(0, str(folder))

from edge_bench import _expect, _inspect, _json, _protected_model, _sha256
from gguf_metadata_contract import read_gguf_header

SCHEMA = "qlh.comm_multipos_bench.v1"
GAMMAS = (2, 4, 8)
DEFAULT_PREFIX = (40, 1101, 525, 264, 1588, 13, 198, 32)
FIXED_TOKENS = (785, 1633, 315, 220, 17, 10, 17, 374,
                220, 19, 13, 198, 791, 1920, 374, 264,
                2311, 311, 1492, 13, 198, 40, 1101, 525,
                264, 1588, 13, 198, 32, 25, 220, 19)
SHA256 = re.compile(r"[0-9a-f]{64}\Z")


class BenchError(ValueError):
    pass


def _integers(text):
    try:
        values = json.loads(text) if text.lstrip().startswith("[") else [int(x) for x in text.split(",")]
    except (ValueError, TypeError) as exc:
        raise BenchError("token/gamma lists must contain comma-separated integers or a JSON integer array") from exc
    if not isinstance(values, list) or not values or any(type(x) is not int for x in values):
        raise BenchError("token/gamma lists must be nonempty integer arrays")
    return values


def _workload(args):
    gammas = _integers(args.gammas)
    if sorted(gammas) != list(GAMMAS):
        raise BenchError("the numerical gate requires exactly gamma=2,4,8, without duplicates")
    if args.batches < 3 or args.threads < 1:
        raise BenchError("--batches must be >=3 at gamma=8 and --threads must be positive")
    prefix = _integers(args.prefix_tokens)
    count = max(GAMMAS) * args.batches
    tokens = (_integers(args.tokens) if args.tokens is not None else
              [FIXED_TOKENS[i % len(FIXED_TOKENS)] for i in range(count)])
    if len(tokens) != count:
        raise BenchError("--tokens must contain exactly 8 * --batches tokens (one shared stream for all gammas)")
    if args.ctx < len(prefix) + len(tokens):
        raise BenchError("--ctx must fit the entire nonzero initial KV prefix and fixed token stream")
    return prefix, tokens


def _full_attention_header(model, info):
    raw, tensors = read_gguf_header(model)
    meta = info["actual"]
    if meta["architecture"] not in {"qwen2", "llama"}:
        raise BenchError("full-attention gate accepts only qwen2/llama; qwen35 hybrid is not full attention")
    if meta["full_attention_interval"] not in (None, 1) or meta.get("nextn_predict_layers", 0):
        raise BenchError("full-attention gate refuses hybrid intervals or MTP blocks")
    for key, value in raw.items():
        if ("ssm" in key.lower() or "recurrent" in key.lower()
                or ("sliding_window" in key.lower() and (type(value) is not int or value != 0))):
            raise BenchError(f"full-attention header contains unsupported hybrid/window metadata: {key}")
    if any("ssm" in name.lower() for name in tensors):
        raise BenchError("full-attention header contains SSM tensors")
    dimension = meta["n_embd"]
    if type(dimension) is not int or dimension <= 0:
        raise BenchError("header n_embd must be a positive integer")
    kv = meta["head_count_kv"]
    if not ((type(kv) is int and kv > 0) or
            (type(kv) is list and len(kv) == info["layers"] and all(type(n) is int and n > 0 for n in kv))):
        raise BenchError("full-attention head_count_kv must be positive for every layer")
    embedding = tensors.get("token_embd.weight")
    output = tensors.get("output.weight")
    if embedding is None or len(embedding[2]) != 2 or embedding[2][0] != dimension:
        raise BenchError("header requires a token embedding matrix matching n_embd")
    vocabulary = embedding[2][1]
    if output is not None and output[2] != (dimension, vocabulary):
        raise BenchError("output/embedding header vocabulary dimensions differ")
    if info["role"] == "tail":
        norm = tensors.get("output_norm.weight")
        if norm is None or norm[2] != (dimension,):
            raise BenchError("tail header requires output_norm.weight")
    header_bytes = min(value[0] for value in tensors.values())
    if not 0 < header_bytes <= 64 * 1024 * 1024:
        raise BenchError("GGUF header exceeds the bounded header budget")
    with model.open("rb") as handle:
        header_sha = hashlib.sha256(handle.read(header_bytes)).hexdigest()
    return {"n_vocab": vocabulary, "header_bytes": header_bytes, "header_sha256": header_sha,
            "tensor_count": len(tensors), "token_embd_shape": list(embedding[2]),
            "output_shape": list(output[2]) if output else None,
            "output_uses_tied_embedding": output is None,
            "attention_head_count": raw.get(f'{meta["architecture"]}.attention.head_count'),
            "full_attention": True}


def inspect_pair(models, manifests, expectations):
    segments, declarations = [], []
    for model, manifest, explicit in zip(models, manifests, expectations):
        info = _inspect(model, manifest, explicit, bench=True)
        segments.append(info)
        if info["differences"]:
            raise BenchError("segment metadata differs from manifest: " + "; ".join(info["differences"]))
        declaration = _json(manifest)
        # Tie the separately parsed declaration to the exact bytes preflight read.
        if _sha256(manifest) != info["manifest_sha256"]:
            raise BenchError("manifest changed during preflight")
        info.update(model=str(model.resolve()), manifest=str(manifest.resolve()),
                    file_bytes=model.stat().st_size, header=_full_attention_header(model, info))
        if "hidden_size" in declaration and declaration["hidden_size"] != info["actual"]["n_embd"]:
            raise BenchError("manifest hidden_size differs from header n_embd")
        declarations.append(declaration)
    head, tail = segments
    if (head["role"] != "head" or tail["role"] != "tail"
            or head["layer_range"][0] != 0 or tail["layer_range"][1] != head["total_layers"]
            or head["layer_range"][1] != tail["layer_range"][0]
            or head["total_layers"] != tail["total_layers"]):
        raise BenchError("head/tail must cover contiguous proper decoder ranges from 0 to n_layer")
    for key in ("architecture", "n_embd"):
        if head["actual"][key] != tail["actual"][key]:
            raise BenchError(f"head/tail {key} differs")
    if head["header"]["n_vocab"] != tail["header"]["n_vocab"]:
        raise BenchError("head/tail header n_vocab differs")
    identities = [item.get("source_model_sha256") for item in declarations]
    if not all(isinstance(value, str) and SHA256.fullmatch(value) for value in identities) or identities[0] != identities[1]:
        raise BenchError("head/tail source_model_sha256 identities differ or are missing")
    for key in ("tokenizer_sha256", "source_model_id"):
        if any(key in item for item in declarations):
            if not declarations[0].get(key) or declarations[0].get(key) != declarations[1].get(key):
                raise BenchError(f"head/tail {key} identities differ or are missing")
    return segments, declarations


def _native_contract(runner, info):
    if getattr(runner, "_worker", None) is not None:
        raise BenchError("native gate requires direct owners in this dedicated benchmark process")
    if runner.n_embd != info["actual"]["n_embd"] or runner.n_layer != info["layers"]:
        raise BenchError("native embedding/layer counts differ from verified header/manifest")
    if getattr(runner, "supports_multi_position_argmax", None) is not True:
        raise BenchError("shim lacks supports_multi_position_argmax=True; build the new shim")
    if not callable(getattr(runner, "forward_hidden_to_tokens", None)):
        raise BenchError("runner lacks forward_hidden_to_tokens")
    lib = runner._lib
    if not all(hasattr(lib, name) for name in ("qlh_kh_n_vocab", "qlh_kh_finite_logits_guard")):
        raise BenchError("native gate requires vocabulary and finite-logits guards; rebuild shim")
    lib.qlh_kh_n_vocab.argtypes = [ctypes.c_void_p]
    lib.qlh_kh_n_vocab.restype = ctypes.c_int32
    lib.qlh_kh_finite_logits_guard.argtypes = []
    lib.qlh_kh_finite_logits_guard.restype = ctypes.c_int32
    vocabulary = lib.qlh_kh_n_vocab(runner._handle)
    if (type(runner.n_vocab) is not int or runner.n_vocab <= 0
            or vocabulary != runner.n_vocab or vocabulary != info["header"]["n_vocab"]):
        raise BenchError("real native n_vocab, public n_vocab and header vocabulary differ")
    if lib.qlh_kh_finite_logits_guard() != 1:
        raise BenchError("native finite-logits guard must be enabled")
    return {"n_embd": runner.n_embd, "n_layer": runner.n_layer, "n_vocab": vocabulary,
            "supports_multi_position_argmax": True, "finite_logits_guard": True}


def _check_tokens(tokens, vocabulary, label):
    if type(tokens) is not list or not tokens or any(type(n) is not int or not 0 <= n < vocabulary for n in tokens):
        raise BenchError(f"{label}: expected nonempty list[int] inside real n_vocab={vocabulary}")


class ProductionLoopback:
    """Negotiate both capabilities, then move real input and token output over TCP."""
    def __init__(self, identity, layer_range):
        from comm_wire_bench import Peer, _master_switch
        from task_worker_adapter import TaskWorkerControlPlane
        from task_worker_protocol import build_message
        from tcp_comm import build_message as build_frame, MessageType
        self.peer = Peer(TaskWorkerControlPlane(), "comm_multipos_native", True)
        self.identity, self.layer_range = dict(identity), list(layer_range)
        self.counter = 0
        self.transfer = self._transfer_hidden
        try:
            # A refreshed HELLO is the production mechanism for newly available
            # runtime capabilities. Bind the advertisement to this same socket.
            capabilities = {"stage_types": ["layer_forward"], "engines": ["llama_cpp"],
                "models": [self.identity], "max_concurrency": 1, "binary_hidden": True,
                "multi_position_argmax": True, "layer_ranges": [self.layer_range]}
            hello = self.peer.worker.begin_worker_hello(node_id=self.peer.name,
                capabilities=capabilities, registration_owner=self.peer.owner)
            frame = self.peer.link.exchange(build_frame(MessageType.TASK_WORKER, hello.snapshot()))
            with _master_switch(True):
                ack = self.peer.coordinator.receive_on_coordinator(self.peer.name, frame["data"],
                    coordinator_node_id="comm_wire_master", connection_generation=self.peer.generation)
            frame = self.peer.link.exchange(build_frame(MessageType.TASK_WORKER, ack.snapshot()), reverse=True)
            self.peer.worker.receive_on_worker(frame["data"], registration_owner=self.peer.owner)
            snapshot = self.peer.coordinator.worker_snapshot(self.peer.name)
            if not ack.payload["accepted"] or snapshot["capabilities"].get("multi_position_argmax") is not True:
                raise BenchError("multi-position capability was not negotiated")
        except BaseException:
            self.close()
            raise

    def _transfer_hidden(self, peer, raw, *, multi, **fields):
        from comm_wire_bench import _message
        from task_worker_protocol import build_message, decode_message, split_inline_hidden, restore_inline_hidden
        from tcp_comm import build_message as build_frame, MessageType
        import base64
        begin = time.perf_counter_ns()
        base = _message(raw, binary_hidden=peer.enabled, **fields)
        payload = base.payload
        self.counter += 1
        payload.update(attempt_id=f"att_multipos{self.counter:08d}", lease_id=f"lease_multipos{self.counter:08d}")
        if multi:
            payload["token_output"] = "all_positions"
        message = build_message("stage_offer", payload, message_id=f"msg_multipos{self.counter:08d}",
                                sent_at_ms=int(time.time() * 1000), binary_hidden=peer.enabled)
        meta, body = split_inline_hidden(message.snapshot(), enabled=peer.enabled)
        packet = build_frame(MessageType.TASK_WORKER, meta, binary_body=body)
        frame = peer.link.exchange(packet)
        restored = restore_inline_hidden(frame["data"], frame.get("binary"))
        received = decode_message(restored, binary_hidden=frame.get("binary") is not None)
        if received.snapshot() != message.snapshot():
            raise BenchError("production offer metadata changed during roundtrip")
        self.offer = received.payload
        data = base64.b64decode(self.offer["root_input"]["hidden_f32"], validate=True)
        return data, {"wall_ms": (time.perf_counter_ns() - begin) / 1e6,
            "format": frame["format"], "frame_bytes": len(packet), "hidden_raw_bytes": len(raw),
            "multi_position_argmax_negotiated": True, "token_output": self.offer.get("token_output", "last")}

    def __call__(self, hidden, *, n_past, multi=False):
        import numpy as np
        raw = np.asarray(hidden, dtype="<f4", order="C").tobytes()
        received, measure = self.transfer(self.peer, raw, n_embd=hidden.shape[1], multi=multi,
            n_past=n_past, model_identity=self.identity, layer_range=self.layer_range)
        if received != raw or measure.get("format") != "binary":
            raise BenchError("production protocol roundtrip changed hidden bytes or binary format")
        restored = np.frombuffer(received, dtype="<f4").reshape(hidden.shape)
        digest = hashlib.sha256(raw).hexdigest()
        return restored, {**measure, "roundtrip_passed": True, "input_sha256": digest,
                          "output_sha256": hashlib.sha256(received).hexdigest()}

    def result(self, tokens, positions, vocabulary, *, multi):
        from task_worker_protocol import build_message, decode_message, stage_output_sha256
        from tcp_comm import build_message as build_frame, MessageType
        from scheduler_pipeline import _layer_stage_result_to_pipeline_value
        output = {"token_argmax": tokens[-1]}
        if multi:
            output.update(token_argmax_by_position=list(tokens), positions=list(positions),
                          n_tokens=len(tokens), n_vocab=vocabulary)
        identity = {key: self.offer[key] for key in
            ("workflow_id", "stage_id", "attempt_id", "lease_id", "lease_epoch", "provider_id")}
        message = build_message("stage_result", {**identity, "output": output,
            "output_sha256": stage_output_sha256(output), "metadata": {}},
            message_id=f"msg_multipos_result{self.counter:08d}", sent_at_ms=int(time.time() * 1000))
        packet = build_frame(MessageType.TASK_WORKER, message.snapshot())
        frame = self.peer.link.exchange(packet, reverse=True)
        received = decode_message(frame["data"])
        if received.snapshot() != message.snapshot():
            raise BenchError("production token output changed during roundtrip")
        value = _layer_stage_result_to_pipeline_value(received.payload["output"],
            hidden_spec=self.offer["hidden_spec"], token_output=self.offer.get("token_output"),
            positions=self.offer["positions"])
        observed = value["token_argmax_by_position"] if multi else [value["token_argmax"]]
        if observed != tokens:
            raise BenchError("production consumer changed native argmax tokens")
        return {"passed": True, "format": frame["format"], "frame_bytes": len(packet),
            "output_sha256": received.payload["output_sha256"], "position_binding_passed": True,
            "n_tokens": len(tokens), "n_vocab": vocabulary, "multi_position": multi}

    def close(self):
        self.peer.close()


def _step(head, tail, tokens, past, *, multi, vocabulary, roundtrip=None):
    import numpy as np
    positions = list(range(past, past + len(tokens)))
    begin = time.perf_counter_ns()
    hidden = head.forward_tokens_to_hidden(list(tokens), n_past=past)
    head_ns = time.perf_counter_ns() - begin
    if (not isinstance(hidden, np.ndarray) or hidden.dtype != np.float32
            or hidden.shape != (len(tokens), head.n_embd) or not np.isfinite(hidden).all()):
        raise BenchError("head must return a finite float32 [input_tokens, n_embd] matrix")
    # These row hashes are diagnostics only. The singleton reference obtains its
    # own hidden by calling head.forward_tokens_to_hidden([input_token]).
    row_hashes = [hashlib.sha256(np.asarray(row, dtype="<f4").tobytes()).hexdigest() for row in hidden]
    incoming, wire = roundtrip(hidden, n_past=past, multi=multi) if roundtrip is not None else (hidden, None)
    begin = time.perf_counter_ns()
    if multi:
        result = tail.forward_hidden_to_tokens(incoming, n_past=past,
            positions=positions, seq_ids=[0] * len(tokens))
    else:
        result = [tail.forward_hidden_to_token(incoming, n_past=past,
            positions=positions, seq_ids=[0] * len(tokens))]
    tail_ns = time.perf_counter_ns() - begin
    _check_tokens(result, vocabulary, "tail argmax")
    if len(result) != len(tokens):
        raise BenchError("tail argmax must return exactly one token for every input row")
    if roundtrip is not None:
        wire["result"] = roundtrip.result(result, positions, vocabulary, multi=multi)
    return {"n_past": past, "input_tokens": list(tokens), "positions": positions,
            "seq_ids": [0] * len(tokens), "tokens": result,
            "hidden_shape": list(hidden.shape), "hidden_row_sha256": row_hashes,
            "head_calls": 1, "tail_calls": 1,
            "tail_api": "forward_hidden_to_tokens" if multi else "forward_hidden_to_token",
            "measure": {"head_wall_ms": head_ns / 1e6, "tail_wall_ms": tail_ns / 1e6}, "wire": wire}


def _run_pass(head, tail, prefix, tokens, gamma, *, bulk, vocabulary, progress, roundtrip=None):
    progress.update(reset=False, prefix=[], steps=[])
    head.reset()
    tail.reset()
    progress["reset"] = True
    for past, token in enumerate(prefix):
        progress["prefix"].append(_step(head, tail, [token], past, multi=False, vocabulary=vocabulary, roundtrip=roundtrip))
    width = gamma if bulk else 1
    for offset in range(0, len(tokens), width):
        progress["steps"].append(_step(head, tail, tokens[offset:offset + width], len(prefix) + offset,
                                       multi=bulk, vocabulary=vocabulary, roundtrip=roundtrip))


def _validated_rows(steps, tokens, start, width, multi, vocabulary, require_loopback=False):
    if type(steps) is not list or len(steps) != len(tokens) // width:
        raise BenchError("incomplete numerical evidence: missing input steps")
    rows = []
    for index, step in enumerate(steps):
        offset = index * width
        chunk = tokens[offset:offset + width]
        positions = list(range(start + offset, start + offset + width))
        api = "forward_hidden_to_tokens" if multi else "forward_hidden_to_token"
        if (step["input_tokens"] != chunk or step["positions"] != positions
                or step["n_past"] != positions[0] or step["seq_ids"] != [0] * width
                or step["head_calls"] != 1 or step["tail_calls"] != 1 or step["tail_api"] != api
                or len(step["hidden_shape"]) != 2 or step["hidden_shape"][0] != width):
            raise BenchError("numerical evidence changed input/continuous positions or head/tail call contract")
        if require_loopback:
            wire = step.get("wire") or {}
            if (wire.get("roundtrip_passed") is not True or wire.get("format") != "binary"
                    or wire.get("hidden_raw_bytes") != width * step["hidden_shape"][1] * 4
                    or not isinstance(wire.get("input_sha256"), str)
                    or not SHA256.fullmatch(wire["input_sha256"])
                    or wire.get("input_sha256") != wire.get("output_sha256")):
                raise BenchError("missing or invalid production protocol roundtrip evidence")
            receipt = wire.get("result") or {}
            if (wire.get("multi_position_argmax_negotiated") is not True
                    or wire.get("token_output") != ("all_positions" if multi else "last")
                    or receipt.get("passed") is not True or receipt.get("position_binding_passed") is not True
                    or receipt.get("n_tokens") != width or receipt.get("n_vocab") != vocabulary
                    or receipt.get("multi_position") is not multi):
                raise BenchError("missing or invalid production token result roundtrip evidence")
        _check_tokens(step["tokens"], vocabulary, "recorded argmax")
        hashes = step["hidden_row_sha256"]
        if len(step["tokens"]) != width or len(hashes) != width or not all(isinstance(h, str) and SHA256.fullmatch(h) for h in hashes):
            raise BenchError("incomplete numerical evidence: one argmax/hash per row is required")
        rows.extend({"position": pos, "input_token": token, "token": argmax, "hidden_sha256": digest}
                    for pos, token, argmax, digest in zip(positions, chunk, step["tokens"], hashes))
    return rows


def compare_groups(groups, prefix, tokens, vocabulary, *, require_loopback=False):
    if (not prefix or len(tokens) < 24 or len(tokens) % 8
            or [group["gamma"] for group in groups] != list(GAMMAS)):
        raise BenchError("complete gamma=2,4,8 evidence with nonzero prefix and >=3 gamma=8 batches is required")
    initial = None
    mismatches = []
    for group in groups:
        gamma = group["gamma"]
        outputs = []
        for name in ("bulk", "sequential"):
            branch = group[name]
            if branch.get("reset") is not True:
                raise BenchError("each branch must reset both owners before prefix replay")
            prefix_rows = _validated_rows(branch["prefix"], prefix, 0, 1, False, vocabulary, require_loopback)
            if initial is None:
                initial = prefix_rows
            if prefix_rows != initial:
                raise BenchError("reset + prefix replay differs; initial KV inputs are not identical")
            outputs.append(_validated_rows(branch["steps"], tokens, len(prefix),
                gamma if name == "bulk" else 1, name == "bulk", vocabulary, require_loopback))
        group["rows"] = []
        for index, (left, right) in enumerate(zip(*outputs)):
            equal = left["token"] == right["token"]
            row = {"row": index, "batch": index // gamma, "row_in_batch": index % gamma,
                   "position": left["position"], "input_token": left["input_token"],
                   "bulk_token": left["token"], "sequential_token": right["token"], "matched": equal,
                   "bulk_hidden_sha256": left["hidden_sha256"],
                   "sequential_hidden_sha256": right["hidden_sha256"]}
            group["rows"].append(row)
            if not equal:
                mismatches.append({"gamma": gamma, **row})
        group.update(passed=all(row["matched"] for row in group["rows"]),
                     batch_count=len(tokens) // gamma, compared_rows=len(tokens))
    return {"passed": not mismatches, "gammas": list(GAMMAS), "initial_n_past": len(prefix),
            "identical_reset_prefix": True, "production_protocol_roundtrip": require_loopback, "tokens_per_gamma": len(tokens),
            "compared_rows": len(tokens) * len(GAMMAS), "mismatch_count": len(mismatches),
            "mismatches": mismatches}


def numerical_bench(head, tail, prefix, tokens, vocabulary, progress, *, roundtrip=None):
    _check_tokens(prefix, vocabulary, "prefix tokens")
    _check_tokens(tokens, vocabulary, "fixed input tokens")
    progress["protocol_loopback"] = {"enabled": roundtrip is not None, "format": "binary" if roundtrip is not None else None}
    progress["groups"] = []
    for gamma in GAMMAS:
        group = {"gamma": gamma, "bulk": {}, "sequential": {}}
        progress["groups"].append(group)
        for name, bulk in (("bulk", True), ("sequential", False)):
            _run_pass(head, tail, prefix, tokens, gamma, bulk=bulk,
                      vocabulary=vocabulary, progress=group[name], roundtrip=roundtrip)
        print(f"gamma={gamma}: {len(tokens) // gamma} bulk batches, {len(tokens)} singleton reference steps", flush=True)
    progress["gate"] = compare_groups(progress["groups"], prefix, tokens, vocabulary, require_loopback=roundtrip is not None)
    return progress["gate"]["passed"]


def native_bench(args, progress, *, factory=None):
    prefix, tokens = _workload(args)
    progress.update(prefix_tokens=prefix, fixed_tokens=tokens, gammas=list(GAMMAS),
                    n_ctx=args.ctx, n_threads=args.threads, n_seq_max=1,
                    scope="fixed inputs; real head token->hidden and tail hidden->argmax; wall measurements are not performance claims")
    # edge_bench's POSIX protection makes a private model copy. This tool must
    # never enter that branch: Windows FILE_SHARE_READ keeps verified bytes in place.
    if os.name != "nt":
        raise BenchError("native no-copy protected_model requires Windows")
    with ExitStack() as stack:
        models = [stack.enter_context(_protected_model(Path(p).resolve())) for p in (args.head, args.tail)]
        manifests = [stack.enter_context(_protected_model(Path(p).resolve())) for p in (args.head_manifest, args.tail_manifest)]
        expectations = [_expect(args.head_expect), _expect(args.tail_expect)]
        segments, declarations = inspect_pair(models, manifests, expectations)
        progress.update(segments=segments, segment_declarations=declarations,
                        explicit_expectations=expectations, preflight_passed=True)
        if args.verify_only:
            return False
        shim = Path(args.shim).resolve()
        if not shim.is_file():
            raise BenchError(f"explicit new shim does not exist: {shim}")
        binaries = sorted(set([shim, *shim.parent.glob("*.dll")]))
        for path in binaries:
            stack.enter_context(_protected_model(path))
        hashes = {str(path): _sha256(path) for path in binaries}
        progress.update(shim=str(shim), binaries=hashes)
        if any(name == "llama_cpp" or name.startswith("llama_cpp.") for name in sys.modules):
            raise BenchError("dedicated patched-native benchmark process must not load pip llama_cpp")
        if factory is None:
            from llama_keep_head import KeepHeadUpstream
            factory = KeepHeadUpstream
        runners = []
        for model in models:
            runner = factory(shim, model, mode="nextn", n_ctx=args.ctx, n_threads=args.threads,
                             n_batch=args.ctx, n_seq_max=1, _worker_process=True)
            stack.callback(runner.close)
            runners.append(runner)
        progress["native_contracts"] = [_native_contract(runner, info) for runner, info in zip(runners, segments)]
        vocabulary = progress["native_contracts"][0]["n_vocab"]
        roundtrip = None
        if getattr(args, "protocol_loopback", False):
            identity = {"model_id": declarations[0].get("source_model_id", "comm_multipos_full_attention"),
                        "engine": "llama_cpp", "format": "gguf", "revision": "comm_multipos_fixed_inputs",
                        "sha256": declarations[0]["source_model_sha256"]}
            roundtrip = ProductionLoopback(identity, segments[1]["layer_range"])
            stack.callback(roundtrip.close)
        passed = numerical_bench(*runners, prefix, tokens, vocabulary, progress, roundtrip=roundtrip)
        if hashes != {str(path): _sha256(path) for path in binaries}:
            raise BenchError("native binaries changed during numerical comparison")
        progress["binary_integrity_passed"] = True
        return passed


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("head", "tail", "head-manifest", "tail-manifest"):
        parser.add_argument("--" + name, required=True)
    parser.add_argument("--head-expect")
    parser.add_argument("--tail-expect")
    parser.add_argument("--shim", help="explicit new shim; mandatory for native execution")
    parser.add_argument("--verify-only", action="store_true", help="check files without loading shim; numerical passed stays false")
    parser.add_argument("--protocol-loopback", action="store_true", help="route every real head hidden through negotiated production binary TCP before tail")
    parser.add_argument("--gammas", default="2,4,8")
    parser.add_argument("--batches", type=int, default=4, help="number of gamma=8 batches; shared stream has 8*batches tokens (>=3)")
    parser.add_argument("--tokens", help="fixed input token IDs as comma list or JSON array")
    parser.add_argument("--prefix-tokens", default=",".join(map(str, DEFAULT_PREFIX)))
    parser.add_argument("--ctx", type=int, default=128)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--json-out", type=Path, required=True)
    args = parser.parse_args(argv)
    if not args.verify_only and not args.shim:
        parser.error("native execution requires an explicit --shim built with the new multi-position API")
    inputs = [Path(p).resolve() for p in (args.head, args.tail, args.head_manifest, args.tail_manifest, args.shim) if p]
    output = args.json_out.resolve()
    if output.suffix.lower() != ".json" or output in inputs or output == Path(__file__).resolve() or output.parent in {p.parent for p in inputs}:
        parser.error("--json-out must be a separate evidence JSON outside model/manifest/shim directories")
    report = {"schema": SCHEMA, "passed": False, "status": "failed",
              "started_at": datetime.now(timezone.utc).isoformat(),
              "invocation": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
              "environment": {"platform": platform.platform(), "python": sys.version, "executable": sys.executable},
              "native": {}}
    code = 1
    try:
        sources = ("scripts/comm_multipos_bench.py", "scripts/edge_bench.py", "src/gguf_metadata_contract.py",
                   "src/llama_keep_head.py", "src/llama_keep_head_worker.py", "scripts/model_tools/keep_head_shim/qlh_keep_head.c",
                   "scripts/comm_wire_bench.py", "src/task_worker_protocol.py", "src/task_worker_adapter.py", "src/tcp_comm.py",
                   "src/scheduler_pipeline.py", "src/qlh_core/contracts/token_output.py")
        report["source_sha256"] = {path: _sha256(ROOT / path) for path in sources}
        report["git_head"] = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
        passed = native_bench(args, report["native"])
        report["passed"] = passed
        report["status"] = "passed" if passed else ("verified_only" if args.verify_only else "mismatch")
        code = 0 if passed or args.verify_only else 1
    except Exception as exc:
        report["error"] = f"{type(exc).__name__}: {exc}"
        print(report["error"], file=sys.stderr, flush=True)
    finally:
        report["finished_at"] = datetime.now(timezone.utc).isoformat()
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    return code


if __name__ == "__main__":
    raise SystemExit(main())
