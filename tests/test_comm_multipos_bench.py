"""Contracts for the real full-attention numerical driver; no native DLL loads."""
from __future__ import annotations

from copy import deepcopy
import importlib.util
import json
import os
from pathlib import Path
import struct
from types import SimpleNamespace

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("comm_multipos_bench", ROOT / "scripts/comm_multipos_bench.py")
bench = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(bench)
PREFIX = [3, 5, 7]
TOKENS = list(range(8, 32))


class Function:
    def __init__(self, value):
        self.value = value

    def __call__(self, *args):
        return self.value


class Runner:
    """Stateful fake makes missing reset/prefix/positions and hidden reuse visible."""
    n_embd, n_layer, n_vocab = 4, 1, 128
    supports_multi_position_argmax = True
    _worker, _handle = None, None

    def __init__(self, *, mismatch=False):
        self.events, self.state, self.closed = [], [], False
        self.mismatch = mismatch
        self._lib = SimpleNamespace(qlh_kh_n_vocab=Function(128), qlh_kh_finite_logits_guard=Function(1))

    def reset(self):
        self.events.append(("reset",))
        self.state = []

    def forward_tokens_to_hidden(self, tokens, *, n_past):
        assert n_past == len(self.state)
        self.events.append(("head", list(tokens), n_past))
        rows = []
        for token in tokens:
            rows.append([token, len(self.state), (sum(self.state) + token) % 128, 1])
            self.state.append(token)
        return np.array(rows, dtype=np.float32)

    def _tail(self, hidden, *, n_past, positions, seq_ids, multi):
        assert n_past == len(self.state)
        assert positions == list(range(n_past, n_past + len(hidden)))
        assert seq_ids == [0] * len(hidden)
        self.events.append(("multi" if multi else "single", hidden.copy(), n_past))
        result = []
        for row in hidden:
            assert int(row[1]) == len(self.state)
            result.append((int(row[0]) + int(row[2]) + sum(self.state)) % 128)
            self.state.append(int(row[0]))
        if self.mismatch and multi and n_past == len(PREFIX) + 8:
            result[-1] = (result[-1] + 1) % 128
        return result

    def forward_hidden_to_token(self, hidden, **kwargs):
        assert len(hidden) == 1, "reference must compute genuine singleton hidden"
        return self._tail(hidden, multi=False, **kwargs)[0]

    def forward_hidden_to_tokens(self, hidden, **kwargs):
        return self._tail(hidden, multi=True, **kwargs)

    def close(self):
        self.closed = True


def run_probe(*, mismatch=False, roundtrip=None):
    head, tail, result = Runner(), Runner(mismatch=mismatch), {}
    passed = bench.numerical_bench(head, tail, PREFIX, TOKENS, 128, result, roundtrip=roundtrip)
    return head, tail, result, passed


def test_each_gamma_recomputes_real_singleton_head_from_reset_nonzero_prefix():
    head, tail, report, passed = run_probe()
    assert passed and report["gate"]["compared_rows"] == 72
    assert report["gate"]["initial_n_past"] == 3
    assert [group["batch_count"] for group in report["groups"]] == [12, 6, 3]
    expected = []
    for gamma in (2, 4, 8):
        for width in (gamma, 1):
            expected.append(("reset",))
            expected.extend(("head", [token], i) for i, token in enumerate(PREFIX))
            expected.extend(("head", TOKENS[i:i + width], len(PREFIX) + i)
                            for i in range(0, len(TOKENS), width))
    assert head.events == expected
    assert sum(event[0] == "reset" for event in tail.events) == 6
    assert sum(event[0] == "multi" for event in tail.events) == 21
    assert sum(event[0] == "single" for event in tail.events) == 90
    for group in report["groups"]:
        assert [row["input_token"] for row in group["rows"]] == TOKENS
        assert [row["position"] for row in group["rows"]] == list(range(3, 27))
        assert all(row["matched"] for row in group["rows"])
        assert len(group["sequential"]["steps"]) == len(TOKENS)
        assert all("measure" in step for step in group["bulk"]["steps"])


def test_mismatch_retains_every_row_and_continues_all_gammas():
    _, _, report, passed = run_probe(mismatch=True)
    assert not passed
    gate = report["gate"]
    assert gate["mismatch_count"] == 3 and len(report["groups"]) == 3
    assert [row["position"] for row in gate["mismatches"]] == [12, 14, 18]
    for group in report["groups"]:
        assert not group["passed"] and len(group["rows"]) == len(TOKENS)
        mismatch = next(row for row in group["rows"] if not row["matched"])
        assert mismatch["bulk_token"] != mismatch["sequential_token"]
        assert mismatch["bulk_hidden_sha256"] == mismatch["sequential_hidden_sha256"]


@pytest.mark.parametrize("fault", ["missing_gamma", "duplicate_gamma", "missing_batch", "missing_singleton",
    "position", "seq_id", "wrong_input", "argmax_count", "prefix_drift", "no_reset", "hidden_reuse", "empty_prefix"])
def test_gate_rejects_incomplete_or_invalid_evidence(fault):
    _, _, report, _ = run_probe()
    groups = deepcopy(report["groups"])
    prefix = PREFIX
    if fault == "missing_gamma":
        groups.pop()
    elif fault == "duplicate_gamma":
        groups[1]["gamma"] = 2
    elif fault == "missing_batch":
        groups[2]["bulk"]["steps"].pop()
    elif fault == "missing_singleton":
        groups[1]["sequential"]["steps"].pop()
    elif fault == "position":
        groups[0]["bulk"]["steps"][0]["positions"][0] += 1
    elif fault == "seq_id":
        groups[0]["bulk"]["steps"][0]["seq_ids"][0] = 1
    elif fault == "wrong_input":
        groups[0]["sequential"]["steps"][0]["input_tokens"][0] += 1
    elif fault == "argmax_count":
        groups[0]["bulk"]["steps"][0]["tokens"].pop()
    elif fault == "prefix_drift":
        groups[1]["sequential"]["prefix"][0]["hidden_row_sha256"][0] = "0" * 64
    elif fault == "no_reset":
        groups[0]["bulk"]["reset"] = False
    elif fault == "hidden_reuse":
        groups[0]["sequential"]["steps"][0]["head_calls"] = 0
    else:
        prefix = []
    with pytest.raises(bench.BenchError):
        bench.compare_groups(groups, prefix, TOKENS, 128)


@pytest.mark.parametrize("bad", [np.ones((2, 4), dtype=np.float64), np.ones((2, 3), dtype=np.float32),
    np.ones((1, 4), dtype=np.float32), np.full((2, 4), np.nan, dtype=np.float32),
    np.full((2, 4), np.inf, dtype=np.float32)])
def test_invalid_hidden_is_rejected_before_tail(bad):
    head, tail = Runner(), Runner()
    head.forward_tokens_to_hidden = lambda *a, **kw: bad
    with pytest.raises(bench.BenchError, match="finite float32"):
        bench._step(head, tail, [1, 2], 0, multi=True, vocabulary=128)
    assert not tail.events


@pytest.mark.parametrize("bad", [[1], [1, 2, 3], [1, 128], [1, -1], [1, True], [1, 1.5], (1, 2), 1])
def test_bulk_must_return_list_of_exactly_one_valid_argmax_per_row(bad):
    head, tail = Runner(), Runner()
    tail.forward_hidden_to_tokens = lambda *a, **kw: bad
    with pytest.raises(bench.BenchError, match="argmax"):
        bench._step(head, tail, [1, 2], 0, multi=True, vocabulary=128)


@pytest.mark.parametrize("fault", ["capability_missing", "capability_truthy", "api_missing", "guard_missing",
    "guard_disabled", "public_vocab", "native_vocab", "header_vocab", "embedding", "layers", "worker"])
def test_native_contract_refuses_old_shim_and_false_or_inconsistent_guards(fault):
    runner = Runner()
    info = {"actual": {"n_embd": 4}, "layers": 1, "header": {"n_vocab": 128}}
    if fault == "capability_missing":
        runner.supports_multi_position_argmax = False
    elif fault == "capability_truthy":
        runner.supports_multi_position_argmax = 1
    elif fault == "api_missing":
        runner.forward_hidden_to_tokens = None
    elif fault == "guard_missing":
        del runner._lib.qlh_kh_finite_logits_guard
    elif fault == "guard_disabled":
        runner._lib.qlh_kh_finite_logits_guard.value = 0
    elif fault == "public_vocab":
        runner.n_vocab = 128.0
    elif fault == "native_vocab":
        runner._lib.qlh_kh_n_vocab.value = 129
    elif fault == "header_vocab":
        info["header"]["n_vocab"] = 129
    elif fault == "embedding":
        runner.n_embd = 8
    elif fault == "layers":
        runner.n_layer = 8
    else:
        runner._worker = object()
    with pytest.raises(bench.BenchError):
        bench._native_contract(runner, info)
    assert not runner.events


def _string(text):
    encoded = text.encode("utf-8")
    return struct.pack("<Q", len(encoded)) + encoded


def _gguf(path, *, arch="qwen2", extra=(), tail=False):
    fields = [("general.architecture", arch), (f"{arch}.block_count", 1),
              (f"{arch}.embedding_length", 4), (f"{arch}.attention.head_count", 2),
              (f"{arch}.attention.head_count_kv", 1), *extra]
    kv = b""
    for key, value in fields:
        kind = 8 if isinstance(value, str) else 4
        kv += _string(key) + struct.pack("<I", kind) + (_string(value) if kind == 8 else struct.pack("<I", value))
    shapes = [("token_embd.weight", (4, 128)), ("blk.0.attn_q.weight", (4, 4))]
    if tail:
        shapes.extend([("output_norm.weight", (4,)), ("output.weight", (4, 128))])
    table, offset = b"", 0
    for name, shape in shapes:
        table += _string(name) + struct.pack("<I", len(shape)) + struct.pack("<" + "Q" * len(shape), *shape) + struct.pack("<IQ", 0, offset)
        size = int(np.prod(shape)) * 4
        offset += (size + 31) // 32 * 32
    header = b"GGUF" + struct.pack("<IQQ", 3, len(shapes), len(fields)) + kv + table
    path.write_bytes(header + bytes((-len(header)) % 32) + bytes(offset))


def make_pair(root, *, arch="qwen2", extra=()):
    root.mkdir()
    models, manifests = [], []
    for role in ("head", "tail"):
        model, manifest = root / f"{role}.gguf", root / f"{role}.manifest.json"
        _gguf(model, arch=arch, extra=extra, tail=role == "tail")
        declaration = {"mode": role, "n_layer": 2, "architecture": arch,
            "source_layer_range": [0, 1] if role == "head" else [1, 2], "kept_block_count": 1,
            "source_model_sha256": "a" * 64, "artifact_sha256": bench._sha256(model),
            "gguf_metadata": {"block_count": 1, "head_count_kv": 1, "key_length": None,
                              "full_attention_interval": dict(extra).get(f"{arch}.full_attention_interval")}}
        manifest.write_text(json.dumps(declaration), encoding="utf-8")
        models.append(model)
        manifests.append(manifest)
    return models, manifests


def change_manifest(path, **updates):
    data = json.loads(path.read_text(encoding="utf-8"))
    data.update(updates)
    path.write_text(json.dumps(data), encoding="utf-8")


def arguments(models, manifests, **updates):
    values = dict(head=str(models[0]), tail=str(models[1]), head_manifest=str(manifests[0]),
        tail_manifest=str(manifests[1]), head_expect=None, tail_expect=None,
        shim=str(models[0].parent.parent / "native" / "shim.dll"), verify_only=False,
        gammas="2,4,8", batches=3, threads=1, ctx=32, prefix_tokens=json.dumps(PREFIX), tokens=json.dumps(TOKENS))
    values.update(updates)
    return SimpleNamespace(**values)


def test_real_header_and_manifest_preflight_needs_no_native_or_tokenizer(tmp_path):
    models, manifests = make_pair(tmp_path / "models")
    segments, declarations = bench.inspect_pair(models, manifests, [{}, {}])
    assert [info["role"] for info in segments] == ["head", "tail"]
    assert all(info["header"]["full_attention"] for info in segments)
    assert all(info["header"]["n_vocab"] == 128 for info in segments)
    assert all(info["header"]["header_bytes"] < 1000 for info in segments)
    assert segments[0]["model"] == str(models[0].resolve())
    assert "hidden_size" not in declarations[0]  # existing legacy cutter manifests remain usable
    for segment in segments:
        assert bench.SHA256.fullmatch(segment["header"]["header_sha256"])
        assert bench.SHA256.fullmatch(segment["manifest_sha256"])
        assert segment["artifact_sha256"] == bench._sha256(Path(segment["model"]))


@pytest.mark.parametrize("arch,extra", [("qwen35", ()), ("qwen2", (("qwen2.full_attention_interval", 4),)),
    ("qwen2", (("qwen2.ssm.state_size", 16),)), ("llama", (("llama.attention.sliding_window", 32),))])
def test_full_attention_gate_rejects_hybrid_and_sliding_window_headers(tmp_path, arch, extra):
    models, manifests = make_pair(tmp_path / "models", arch=arch, extra=extra)
    with pytest.raises(bench.BenchError, match="full-attention"):
        bench.inspect_pair(models, manifests, [{}, {}])


@pytest.mark.parametrize("fault", ["sha", "metadata", "source", "range", "hidden", "tokenizer", "split", "truncated"])
def test_sha_header_manifest_and_chain_faults_fail_preflight(tmp_path, fault):
    models, manifests = make_pair(tmp_path / "models")
    if fault == "sha":
        change_manifest(manifests[0], artifact_sha256="0" * 64)
    elif fault == "metadata":
        change_manifest(manifests[0], gguf_metadata={"block_count": 1, "head_count_kv": 2,
                                                  "key_length": None, "full_attention_interval": None})
    elif fault == "source":
        change_manifest(manifests[1], source_model_sha256="b" * 64)
    elif fault == "range":
        change_manifest(manifests[1], source_layer_range=[2, 3], n_layer=3)
    elif fault == "hidden":
        change_manifest(manifests[1], hidden_size=8)
    elif fault == "tokenizer":
        change_manifest(manifests[1], tokenizer_sha256="b" * 64)
    elif fault == "split":
        _gguf(models[0], extra=(("split.count", 2),))
        change_manifest(manifests[0], artifact_sha256=bench._sha256(models[0]))
    else:
        models[0].write_bytes(b"GGUF")
    with pytest.raises(ValueError):
        bench.inspect_pair(models, manifests, [{}, {}])


@pytest.mark.parametrize("updates", [{"gammas": "2,4"}, {"gammas": "2,4,8,8"}, {"batches": 2},
    {"ctx": 26}, {"threads": 0}, {"prefix_tokens": "[]"}, {"tokens": "[1,2]"}])
def test_workload_cannot_silently_weaken_numerical_gate(updates):
    with pytest.raises(bench.BenchError):
        bench._workload(arguments([Path("h"), Path("t")], [Path("hm"), Path("tm")], **updates))


def test_invalid_input_vocab_is_rejected_before_any_reset_or_forward():
    head, tail = Runner(), Runner()
    with pytest.raises(bench.BenchError, match="real n_vocab"):
        bench.numerical_bench(head, tail, PREFIX, [128, *TOKENS[1:]], 128, {})
    assert head.events == tail.events == []


@pytest.mark.skipif(os.name != "nt", reason="Windows mandatory no-copy file protection")
@pytest.mark.parametrize("failure", [False, True])
@pytest.mark.parametrize("protocol", [False, True])
def test_native_workflow_protects_in_place_and_closes_owners_on_success_or_failure(tmp_path, failure, protocol):
    models, manifests = make_pair(tmp_path / "models")
    args = arguments(models, manifests, protocol_loopback=protocol)
    shim = Path(args.shim)
    shim.parent.mkdir()
    shim.write_bytes(b"not a DLL; native factory is injected by this contract test")
    owners = []
    def factory(shim_path, model, **kwargs):
        assert model in models and model.resolve() == model
        assert kwargs == dict(mode="nextn", n_ctx=32, n_threads=1, n_batch=32, n_seq_max=1, _worker_process=True)
        with pytest.raises(OSError):
            model.write_bytes(b"refuse protected model mutation")
        owner = Runner()
        if failure and len(owners) == 1:
            owner.forward_hidden_to_tokens = lambda *a, **k: (_ for _ in ()).throw(RuntimeError("injected failure"))
        owners.append(owner)
        return owner
    result = {}
    if failure:
        with pytest.raises(RuntimeError, match="injected failure"):
            bench.native_bench(args, result, factory=factory)
        assert result["groups"][0]["bulk"]["prefix"]
    else:
        assert bench.native_bench(args, result, factory=factory)
        assert result["binary_integrity_passed"]
    assert len(owners) == 2 and all(owner.closed for owner in owners)
    assert result["native_contracts"][0]["n_vocab"] == 128
    assert result["binaries"][str(shim)] == bench._sha256(shim)


@pytest.mark.skipif(os.name != "nt", reason="Windows mandatory no-copy file protection")
def test_verify_only_does_not_open_shim_or_claim_numerical_pass(tmp_path):
    models, manifests = make_pair(tmp_path / "models")
    report = {}
    assert bench.native_bench(arguments(models, manifests, verify_only=True), report,
        factory=lambda *a, **kw: pytest.fail("verify-only must not load native")) is False
    assert report["preflight_passed"] and "native_contracts" not in report


def test_nonwindows_does_not_enter_model_copy_branch(monkeypatch):
    monkeypatch.setattr(bench, "os", SimpleNamespace(name="posix"))
    monkeypatch.setattr(bench, "_protected_model", lambda *a: pytest.fail("must not copy models"))
    args = arguments([Path("h"), Path("t")], [Path("hm"), Path("tm")])
    with pytest.raises(bench.BenchError, match="no-copy"):
        bench.native_bench(args, {})


@pytest.mark.parametrize("outcome", ["mismatch", "error", "verify"])
def test_cli_keeps_machine_failure_partial_rows_and_verify_status(tmp_path, monkeypatch, outcome):
    models, manifests = make_pair(tmp_path / "models")
    output = tmp_path / "evidence" / "report.json"
    def run(args, progress):
        progress["groups"] = [{"gamma": 2, "bulk": {"steps": [{"tokens": [3, 4]}]}}]
        if outcome == "error":
            raise bench.BenchError("injected partial failure")
        progress["gate"] = {"passed": False, "mismatches": [{"position": 5, "bulk_token": 3, "sequential_token": 4}]}
        return False
    monkeypatch.setattr(bench, "native_bench", run)
    argv = ["--head", str(models[0]), "--tail", str(models[1]), "--head-manifest", str(manifests[0]),
            "--tail-manifest", str(manifests[1]), "--shim", str(tmp_path / "native" / "shim.dll"), "--json-out", str(output)]
    if outcome == "verify":
        argv.append("--verify-only")
    assert bench.main(argv) == (0 if outcome == "verify" else 1)
    report = json.loads(output.read_text(encoding="utf-8"))
    assert report["passed"] is False
    assert report["native"]["groups"][0]["bulk"]["steps"][0]["tokens"] == [3, 4]
    assert report["source_sha256"]["scripts/comm_multipos_bench.py"] == bench._sha256(ROOT / "scripts/comm_multipos_bench.py")
    assert report["status"] == {"error": "failed", "mismatch": "mismatch", "verify": "verified_only"}[outcome]


def test_evidence_output_cannot_overwrite_native_input(tmp_path):
    models, manifests = make_pair(tmp_path / "models")
    original = manifests[0].read_bytes()
    with pytest.raises(SystemExit):
        bench.main(["--head", str(models[0]), "--tail", str(models[1]), "--head-manifest", str(manifests[0]),
                    "--tail-manifest", str(manifests[1]), "--verify-only", "--json-out", str(manifests[0])])
    assert manifests[0].read_bytes() == original


def test_optional_production_protocol_loopback_matches_direct_gate_and_closes_sockets():
    identity = {"model_id": "qwen2_contract", "engine": "llama_cpp", "format": "gguf",
                "revision": "multipos_contract", "sha256": "a" * 64}
    loop = bench.ProductionLoopback(identity, [1, 2])
    link = loop.peer.link
    try:
        _, _, direct, direct_passed = run_probe()
        head, tail, report, passed = run_probe(roundtrip=loop)
        assert direct_passed and passed
        assert report["protocol_loopback"] == {"enabled": True, "format": "binary"}
        assert report["gate"]["production_protocol_roundtrip"] is True
        for group, reference in zip(report["groups"], direct["groups"]):
            assert group["rows"] == reference["rows"]
            for branch in (group["bulk"], group["sequential"]):
                for step in [*branch["prefix"], *branch["steps"]]:
                    wire = step["wire"]
                    assert wire["format"] == "binary" and wire["roundtrip_passed"]
                    assert wire["input_sha256"] == wire["output_sha256"]
                    assert wire["hidden_raw_bytes"] == len(step["input_tokens"]) * 4 * 4
                    assert wire["multi_position_argmax_negotiated"] is True
                    receipt = wire["result"]
                    assert receipt["passed"] and receipt["position_binding_passed"]
                    assert receipt["n_tokens"] == len(step["input_tokens"])
                    assert receipt["n_vocab"] == 128
                    assert receipt["multi_position"] is (step["tail_api"] == "forward_hidden_to_tokens")
        assert any(event[0] == "head" and len(event[1]) == 8 for event in head.events)
        assert any(event[0] == "multi" and len(event[1]) == 8 for event in tail.events)
        broken = deepcopy(report["groups"])
        broken[0]["bulk"]["steps"][0]["wire"] = None
        with pytest.raises(bench.BenchError, match="protocol roundtrip evidence"):
            bench.compare_groups(broken, PREFIX, TOKENS, 128, require_loopback=True)
    finally:
        loop.close()
    assert link.closed and link.sender.fileno() == link.receiver.fileno() == -1


def test_corrupt_protocol_restoration_cannot_reach_tail():
    identity = {"model_id": "qwen2_contract", "engine": "llama_cpp", "format": "gguf",
                "revision": "multipos_contract", "sha256": "a" * 64}
    loop = bench.ProductionLoopback(identity, [1, 2])
    head, tail = Runner(), Runner()
    try:
        loop.transfer = lambda peer, raw, **kw: (bytes([raw[0] ^ 1]) + raw[1:], {"format": "binary"})
        with pytest.raises(bench.BenchError, match="roundtrip changed hidden bytes"):
            bench._step(head, tail, [1, 2], 0, multi=True, vocabulary=128, roundtrip=loop)
        assert not tail.events
    finally:
        loop.close()


def test_legacy_manifest_can_fill_missing_metadata_but_cannot_override_it(tmp_path):
    models, manifests = make_pair(tmp_path / "models")
    for path in manifests:
        declaration = json.loads(path.read_text(encoding="utf-8"))
        declaration.pop("gguf_metadata")
        path.write_text(json.dumps(declaration), encoding="utf-8")
    explicit = {"head_count_kv": 1, "key_length": None, "full_attention_interval": None}
    segments, _ = bench.inspect_pair(models, manifests, [explicit, explicit])
    assert all(segment["differences"] == [] for segment in segments)
    with pytest.raises(bench.BenchError, match="metadata differs"):
        bench.inspect_pair(models, manifests, [{**explicit, "head_count_kv": 2}, explicit])


def test_numerical_gate_requires_real_result_receipts_in_addition_to_hidden_roundtrip():
    identity = {"model_id": "qwen2_contract", "engine": "llama_cpp", "format": "gguf",
                "revision": "multipos_contract", "sha256": "a" * 64}
    loop = bench.ProductionLoopback(identity, [1, 2])
    try:
        _, _, report, passed = run_probe(roundtrip=loop)
        assert passed
        report["groups"][0]["bulk"]["steps"][0]["wire"]["result"] = None
        with pytest.raises(bench.BenchError, match="token result roundtrip evidence"):
            bench.compare_groups(report["groups"], PREFIX, TOKENS, 128, require_loopback=True)
    finally:
        loop.close()


def test_real_result_roundtrip_binds_tokens_to_the_offer_positions():
    from qlh_core.contracts.token_output import TokenOutputError
    identity = {"model_id": "qwen2_contract", "engine": "llama_cpp", "format": "gguf",
                "revision": "multipos_contract", "sha256": "a" * 64}
    loop = bench.ProductionLoopback(identity, [1, 2])
    try:
        loop(np.zeros((2, 4), dtype=np.float32), n_past=41, multi=True)
        with pytest.raises(TokenOutputError, match="request_mismatch"):
            loop.result([2, 3], [42, 43], 128, multi=True)
    finally:
        loop.close()
