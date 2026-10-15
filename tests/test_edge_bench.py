"""Contracts for the segment benchmark, without loading native model weights."""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("edge_bench_contract", ROOT / "scripts/edge_bench.py")
edge_bench = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(edge_bench)


def metadata(**updates):
    return {"architecture": "qwen35", "block_count": 8, "head_count_kv": 2,
            "key_length": 128, "full_attention_interval": 4,
            "n_embd": 4, "nextn_predict_layers": 0, **updates}


def manifest(model, **updates):
    return {"mode": "head", "n_layer": 32, "source_layer_range": [0, 8],
            "architecture": "qwen35", "block_count": 32, "kept_block_count": 8,
            "artifact_sha256": edge_bench._sha256(model),
            "gguf_metadata": {key: metadata()[key] for key in edge_bench.KEYS}, **updates}


def test_missing_actual_never_passes():
    diffs = edge_bench._verify({}, {key: metadata()[key] for key in edge_bench.KEYS}, "manifest")
    assert any("block_count" in item for item in diffs)
    assert len(diffs) >= 4


def test_missing_declaration_differs_from_explicit_absence():
    actual = metadata(key_length=None)
    expected = {key: actual[key] for key in edge_bench.KEYS}
    assert edge_bench._verify(actual, expected, "manifest") == []
    expected.pop("key_length")
    assert any("key_length" in item for item in edge_bench._verify(actual, expected, "manifest"))


def test_top_level_source_blocks_are_not_artifact_blocks(tmp_path):
    model = tmp_path / "segment.gguf"
    model.write_bytes(b"segment")
    declared = manifest(model)
    assert edge_bench._declaration(declared)["block_count"] == 8
    declared["gguf_metadata"]["block_count"] = 16
    with pytest.raises(ValueError, match="block_count"):
        edge_bench._declaration(declared)


def run_main(tmp_path, monkeypatch, *, meta=None, declared=None, extra=()):
    model = tmp_path / "segment.gguf"
    model.write_bytes(b"segment")
    path = model.with_suffix(".manifest.json")
    path.write_text(json.dumps(manifest(model) if declared is None else declared), encoding="utf-8")
    monkeypatch.setattr(edge_bench, "_load_gguf_meta", lambda p: metadata() if meta is None else meta)
    output = tmp_path / "report.json"
    argv = ["--model", str(model), "--json-out", str(output), *extra]
    code = edge_bench.main(argv)
    return code, json.loads(output.read_text(encoding="utf-8"))


def test_verify_only_does_not_load_native(tmp_path, monkeypatch):
    path = tmp_path / "segment.manifest.json"
    code, report = run_main(tmp_path, monkeypatch, extra=["--verify-manifest", str(path)])
    assert code == 0 and report["passed"]
    assert report["actual"]["block_count"] == 8
    assert report["rows"] == []


def test_legacy_manifest_does_not_silently_skip_metadata(tmp_path, monkeypatch):
    code, report = run_main(tmp_path, monkeypatch, declared={"kept_block_count": 8},
                            extra=["--verify-manifest", str(tmp_path / "segment.manifest.json")])
    assert code == 1 and not report["passed"]
    assert any("head_count_kv" in item for item in report["differences"])


def test_benchmark_preflights_before_any_native_child(tmp_path, monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("must not launch native for invalid metadata")
    monkeypatch.setattr(edge_bench.subprocess, "run", forbidden)
    code, report = run_main(tmp_path, monkeypatch, meta=metadata(key_length=256))
    assert code == 1
    assert any("key_length" in item for item in report["differences"])


@pytest.mark.parametrize("extra,match", [
    (["--threads", "0"], "threads"),
    (["--threads", "4,4"], "threads"),
    (["--ctx", "8", "--gen", "8"], "context"),
    (["--gen", "0"], "positive"),
    (["--warmup", "-1"], "nonnegative"),
    (["--layers", "16"], "block_count"),
    (["--manifest", "MISSING.json"], "MISSING"),
])
def test_bad_arguments_are_reported_fail_closed(tmp_path, monkeypatch, extra, match):
    code, report = run_main(tmp_path, monkeypatch, extra=extra)
    assert code == 1 and not report["passed"]
    assert match in report["error"]


def test_whole_model_is_refused_before_loading(tmp_path, monkeypatch):
    model = tmp_path / "segment.gguf"
    model.write_bytes(b"segment")
    code, report = run_main(tmp_path, monkeypatch,
                            declared=manifest(model, n_layer=8))
    assert code == 1 and "proper segment" in report["error"]


def test_expect_cannot_override_wrong_manifest(tmp_path, monkeypatch):
    code, report = run_main(tmp_path, monkeypatch,
                            extra=["--manifest", str(tmp_path / "segment.manifest.json"),
                                   "--expect", "block_count=16"])
    assert code == 1 and "conflicts" in report["error"]


class FakeRunner:
    n_layer, n_embd, n_batch = 8, 4, 2

    def __init__(self, clock, *, fail=False):
        self.clock, self.fail, self.closed = clock, fail, False
        self.calls, self.resets = [], 0
        class Function:
            def __init__(self, result): self.result = result
            def __call__(self, *args): return self.result
        from types import SimpleNamespace
        self._lib = SimpleNamespace(qlh_kh_n_vocab=Function(1024), qlh_kh_finite_logits_guard=Function(1))
        self._handle = None

    def reset(self):
        self.resets += 1
        self.clock[0] += 100_000_000

    def forward_tokens_to_hidden(self, value, *, n_past):
        self.calls.append((n_past, len(value)))
        self.clock[0] += (1000 if len(value) > 1 else 5) * 1_000_000
        if self.fail:
            raise RuntimeError("forward failure")
        return np.ones((len(value), self.n_embd), dtype=np.float32)

    def forward_hidden_to_token(self, value, *, n_past):
        self.forward_tokens_to_hidden(value, n_past=n_past)
        return 123

    def forward_hidden_to_hidden(self, value, *, n_past):
        return self.forward_tokens_to_hidden(value, n_past=n_past)

    def close(self):
        self.closed = True


def bench_fake(role="head", fail=False):
    clock = [0]
    runner = FakeRunner(clock, fail=fail)
    kwargs_seen = {}
    def factory(*args, **kwargs):
        kwargs_seen.update(kwargs)
        return runner
    kwargs = dict(shim="unused", info={"role": role, "layers": 8, "actual": {"n_embd": 4}},
                  n_ctx=8, n_threads=4, gen=2, warmup=1, prompt_tokens=4,
                  samples=3, factory=factory, clock=lambda: clock[0])
    return runner, kwargs_seen, kwargs


@pytest.mark.parametrize("role", ["head", "middle", "tail"])
def test_decode_excludes_reset_prefill_and_uses_actual_layers(role):
    runner, seen, kwargs = bench_fake(role)
    result = edge_bench._bench_one(Path("unused"), **kwargs)
    assert result["decode_ms_per_token"] == 5
    assert result["decode_tokens_per_s"] == 200
    assert result["ms_per_layer"] == 5 / 8
    assert result["prompt_tokens_actual"] == 4
    assert result["output_stable"]
    assert runner.resets == 4 and runner.closed
    assert runner.calls[:4] == [(0, 2), (2, 2), (4, 1), (5, 1)]
    assert seen["_worker_process"] is True and seen["n_seq_max"] == 1
    assert seen["mode"] == "nextn"
    assert len(result["samples"]) == 3
    if role == "tail":
        assert result["samples"][0]["tail_argmax"] == [123, 123]


def test_context_target_is_exact_and_clamped():
    runner, seen, kwargs = bench_fake()
    kwargs.update(prompt_tokens=100, n_ctx=6)
    result = edge_bench._bench_one(Path("unused"), **kwargs)
    assert result["prompt_tokens_actual"] == 4
    assert runner.calls[-1] == (5, 1)


def test_native_owner_is_closed_on_forward_failure():
    runner, seen, kwargs = bench_fake(fail=True)
    with pytest.raises(RuntimeError, match="forward failure"):
        edge_bench._bench_one(Path("unused"), **kwargs)
    assert runner.closed


def test_report_never_serializes_nan(tmp_path):
    with pytest.raises(ValueError):
        edge_bench._write(tmp_path / "r.json", {"measurement": float("nan")})


def test_header_native_embedding_mismatch_closes_owner():
    runner, seen, kwargs = bench_fake()
    kwargs["info"]["actual"]["n_embd"] = 8
    with pytest.raises(ValueError, match="embedding"):
        edge_bench._bench_one(Path("unused"), **kwargs)
    assert runner.closed and not runner.calls


@pytest.mark.parametrize("bad", [np.zeros((0, 4), dtype=np.float32),
                                  np.zeros((1, 3), dtype=np.float32),
                                  np.zeros((1, 4), dtype=np.float64),
                                  np.full((1, 4), np.nan, dtype=np.float32)])
def test_empty_wrong_shape_or_nonfinite_hidden_is_rejected(bad):
    runner, seen, kwargs = bench_fake()
    runner.forward_tokens_to_hidden = lambda *a, **k: bad
    with pytest.raises(ValueError, match="finite float32"):
        edge_bench._bench_one(Path("unused"), **kwargs)
    assert runner.closed


def test_worker_rejects_manifest_drift_before_loading(monkeypatch):
    monkeypatch.setattr(edge_bench, "_inspect", lambda *a, **k: {
        "differences": [], "artifact_sha256": "weights", "manifest_sha256": "new"})
    monkeypatch.setattr(edge_bench, "_bench_one", lambda *a, **k: pytest.fail("must not load"))
    with pytest.raises(ValueError, match="manifest changed"):
        edge_bench._worker_locked({"manifest": "m.json", "explicit": {}, "layers": 0,
                                   "artifact_sha256": "weights", "manifest_sha256": "old"}, Path("segment"))


@pytest.mark.parametrize("field", ["role", "layer_range", "manifest_sha256", "n_ctx", "n_threads"])
def test_parent_rejects_wrong_cell_identity(field):
    info = {"role": "head", "layer_range": [0, 8], "artifact_sha256": "weights", "manifest_sha256": "native"}
    row = {**info, "n_ctx": 128, "n_threads": 4}
    request = {"cell": {"n_ctx": 128, "n_threads": 4}}
    assert edge_bench._validate_cell({"row": row}, request, info) is row
    row[field] = "wrong"
    with pytest.raises(ValueError, match="identity mismatch"):
        edge_bench._validate_cell({"row": row}, request, info)


def test_split_gguf_is_refused_before_native(monkeypatch):
    import gguf_metadata_contract
    monkeypatch.setattr(gguf_metadata_contract, "read_gguf_metadata", lambda p: metadata())
    monkeypatch.setattr(gguf_metadata_contract, "read_gguf_header", lambda p: ({"split.count": 2}, {}))
    with pytest.raises(ValueError, match="split GGUF"):
        edge_bench._load_gguf_meta(Path("unused"))


@pytest.mark.skipif(__import__("os").name != "nt", reason="Windows mandatory file sharing contract")
def test_native_model_guard_blocks_write_replace_and_allows_reads(tmp_path):
    path = tmp_path / "segment.gguf"
    path.write_bytes(b"segment")
    replacement = tmp_path / "replacement"
    replacement.write_bytes(b"full")
    with edge_bench._protected_model(path) as protected:
        assert protected.read_bytes() == b"segment"
        with pytest.raises(OSError):
            path.write_bytes(b"full")
        with pytest.raises(OSError):
            replacement.replace(path)
    path.write_bytes(b"writable after close")


def test_old_shim_without_finite_logits_guard_is_refused():
    runner, seen, kwargs = bench_fake("tail")
    del runner._lib.qlh_kh_finite_logits_guard
    with pytest.raises(ValueError, match="finite-logits guards"):
        edge_bench._bench_one(Path("unused"), **kwargs)
    assert runner.closed and not runner.calls


def test_token_id_outside_vocabulary_is_refused_before_forward():
    runner, seen, kwargs = bench_fake()
    kwargs["token_id"] = 9999
    with pytest.raises(ValueError, match="outside the native vocabulary"):
        edge_bench._bench_one(Path("unused"), **kwargs)
    assert runner.closed and not runner.calls


def test_tail_argmax_outside_vocabulary_is_refused():
    runner, seen, kwargs = bench_fake("tail")
    runner.forward_hidden_to_token = lambda *a, **k: 9999
    with pytest.raises(ValueError, match="invalid argmax"):
        edge_bench._bench_one(Path("unused"), **kwargs)
    assert runner.closed


def test_only_transient_native_load_sharing_error_is_retried(monkeypatch):
    runner, seen, kwargs = bench_fake()
    factory = kwargs["factory"]
    calls, delays = [], []
    def busy_then_load(*args, **options):
        calls.append(1)
        if len(calls) == 1:
            cause = OSError("busy DLL")
            cause.winerror = 32
            raise RuntimeError("LoadLibrary") from cause
        return factory(*args, **options)
    kwargs["factory"] = busy_then_load
    monkeypatch.setattr(edge_bench.time, "sleep", delays.append)
    row = edge_bench._bench_one(Path("unused"), **kwargs)
    assert row["native_load_retries"] == 1 and len(calls) == 2
    assert delays == [0.1] and runner.closed


def test_transient_native_load_retry_has_a_fixed_bound(monkeypatch):
    runner, seen, kwargs = bench_fake()
    calls = []
    def always_busy(*args, **options):
        calls.append(1)
        cause = OSError("busy DLL")
        cause.winerror = 32
        raise RuntimeError("LoadLibrary") from cause
    kwargs["factory"] = always_busy
    monkeypatch.setattr(edge_bench.time, "sleep", lambda _: None)
    with pytest.raises(RuntimeError, match="LoadLibrary"):
        edge_bench._bench_one(Path("unused"), **kwargs)
    assert len(calls) == 4 and not runner.calls
