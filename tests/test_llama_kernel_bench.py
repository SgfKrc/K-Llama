"""Fail-closed numerical and provenance gates for the CPU kernel A/B driver."""
from __future__ import annotations

from copy import deepcopy
import importlib.util
import json
from pathlib import Path
import subprocess
from types import SimpleNamespace

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("llama_kernel_bench", ROOT / "scripts/llama_kernel_bench.py")
bench = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(bench)
DIMS = {"n_embd": 2, "n_layer": 3, "n_vocab": 8}
REQUEST = {"prompt": [1, 2], "fixed": [3], "threads": 4, "ctx": 128}


def native():
    return {"schema": bench.NATIVE_SCHEMA, "ok": True, "requested_repack": False, "default_repack": True, **DIMS,
            "fixed_argmax": [1, 2, 3], "fixed_margins": [0.5, 0.2, 0.7],
            "greedy_tokens": [4, 5], "hidden_rows": 3, "hidden_columns": 2,
            "prefill_seconds": [0.01, 0.02], "decode_seconds": [0.02, 0.03],
            "n_ctx": 128, "n_batch": 32, "n_ubatch": 32, "n_seq_max": 1,
            "n_threads": 4, "n_threads_batch": 4, "prompt_tokens": [1, 2], "fixed_tokens": [3],
            "load_seconds": 0.5, "context_seconds": 0.05}


def check(data, path):
    return bench._check_native(data, policy=0, rows=3, greedy=2, repeats=2, dimensions=DIMS, hidden=path, request=REQUEST)


@pytest.fixture
def hidden(tmp_path):
    path = tmp_path / "hidden.f32"
    np.arange(6, dtype="<f4").tofile(path)
    return path


@pytest.mark.parametrize("field,value", [
    ("default_repack", False), ("n_ctx", 64), ("n_batch", 1), ("n_ubatch", 1), ("n_seq_max", 2),
    ("n_threads", 8), ("n_threads_batch", 8), ("prompt_tokens", [1, 3]),
    ("fixed_tokens", [4]), ("load_seconds", -1), ("context_seconds", float("nan")),
    ("ok", False), ("schema", "old"), ("requested_repack", 0),
    ("requested_repack", True), ("n_embd", 3), ("n_layer", 2), ("n_vocab", 9),
    ("fixed_argmax", [1, 2]), ("fixed_argmax", [1, True, 3]),
    ("fixed_argmax", [1, 8, 3]), ("greedy_tokens", []),
    ("fixed_margins", [0.1, float("nan"), 0.2]),
    ("prefill_seconds", [0.1, 0]), ("decode_seconds", [float("inf"), 0.1]),
    ("decode_seconds", [0.1]), ("hidden_rows", 2), ("hidden_columns", 4),
])
def test_native_identity_partial_or_nonfinite_result_never_passes(hidden, field, value):
    data = native()
    data[field] = value
    with pytest.raises(bench.BenchError):
        check(data, hidden)


def test_valid_native_report_reads_exact_f32_rows(hidden):
    np.testing.assert_array_equal(check(native(), hidden), np.arange(6).reshape(3, 2))


@pytest.mark.parametrize("mode", ["short", "nan", "inf"])
def test_hidden_truncation_or_nonfinite_never_passes(hidden, mode):
    values = np.arange(6, dtype="<f4")
    if mode == "short":
        values = values[:-1]
    else:
        values[2] = float(mode)
    values.tofile(hidden)
    with pytest.raises(bench.BenchError, match="hidden"):
        check(native(), hidden)


def runs():
    base = {"policy": 0, "result": native(), "hidden": np.arange(6, dtype="f4").reshape(3, 2),
            "repack_buffer_observed": False}
    candidate = deepcopy(base)
    candidate.update(policy=1, repack_buffer_observed=True)
    candidate["result"]["requested_repack"] = True
    candidate["result"]["decode_seconds"] = [0.01, 0.015]
    return [base, candidate, deepcopy(candidate), deepcopy(base)]


def test_abba_uses_all_repeats_but_never_claims_new_migration_gain():
    result = bench._compare(runs(), 2, 1)
    assert result["numerical"]["passed"]
    assert result["repack_buffer_gate_passed"]
    assert result["performance"]["repack_over_plain_decode"] == pytest.approx(2)
    assert len(result["performance"]["plain_cpu"]["decode_tokens_per_second"]) == 4
    assert result["production_default"] == "repacked_cpu"
    assert result["new_migration_gain_proven"] is False


@pytest.mark.parametrize("mutation,key", [
    ("argmax", "fixed_argmax_equal"), ("greedy", "greedy_equal"), ("hidden", "hidden_close"),
])
def test_any_run_divergence_blocks_the_numerical_gate(mutation, key):
    values = runs()
    if mutation == "argmax":
        values[2]["result"]["fixed_argmax"][0] = 7
    elif mutation == "greedy":
        values[2]["result"]["greedy_tokens"][0] = 7
    else:
        values[2]["hidden"][0, 0] += 1
    result = bench._compare(values, 2, 1)
    assert result["numerical"][key] is False
    assert result["numerical"]["passed"] is False


@pytest.mark.parametrize("wrong_policy", [0, 1])
def test_buffer_observation_must_distinguish_the_actual_policies(wrong_policy):
    values = runs()
    values[wrong_policy]["repack_buffer_observed"] = not bool(wrong_policy)
    assert bench._compare(values, 2, 1)["repack_buffer_gate_passed"] is False


@pytest.mark.parametrize("text", ["", "1,", "-1,2", "2147483648", "1,2.1", "x"])
def test_bad_token_lists_fail_before_native(text):
    with pytest.raises(bench.BenchError):
        bench._tokens(text)


def test_ready_to_apply_source_cannot_reach_native_or_cache(monkeypatch, tmp_path):
    monkeypatch.setattr(bench, "inspect_sync", lambda **kw: {"state": "ready_to_apply", "patches": []})
    with pytest.raises(bench.BenchError, match="committed, applied"):
        bench._preflight(SimpleNamespace(build_root=tmp_path, model=tmp_path / "absent.gguf"))


def test_worktree_patch_source_cannot_reach_native_or_cache(monkeypatch, tmp_path):
    monkeypatch.setattr(bench, "inspect_sync", lambda **kw: {"state": "applied", "patches": [{"state": "applied"}]})
    with pytest.raises(bench.BenchError, match="committed, applied"):
        bench._preflight(SimpleNamespace(build_root=tmp_path, model=tmp_path / "absent.gguf"))


def test_failed_native_command_keeps_exit_code_and_both_raw_logs(monkeypatch, tmp_path):
    calls = []
    def fake(argv, **kw):
        calls.append((argv, kw))
        return subprocess.CompletedProcess(argv, 37, "partial output", "native error")
    monkeypatch.setattr(bench.subprocess, "run", fake)
    with pytest.raises(bench.BenchError, match="37"):
        bench._command(["probe.exe", "a path"], tmp_path, "bad", environment={}, timeout=5)
    assert len(calls) == 1 and calls[0][1]["shell"] is False
    assert (tmp_path / "bad.stdout.log").read_text() == "partial output"
    assert (tmp_path / "bad.stderr.log").read_text() == "native error"
    assert json.loads((tmp_path / "bad.command.json").read_text())["returncode"] == 37


def test_nonempty_output_is_preserved_and_no_native_is_started(monkeypatch, tmp_path):
    marker = tmp_path / "report.json"
    marker.write_text("existing evidence")
    monkeypatch.setattr(bench, "_preflight", lambda *a: pytest.fail("must refuse before preflight"))
    assert bench.main(["--model", str(tmp_path / "m.gguf"), "--out-dir", str(tmp_path)]) == 1
    assert marker.read_text() == "existing evidence"


def test_insufficient_context_never_reaches_native(monkeypatch, tmp_path):
    monkeypatch.setattr(bench, "_preflight", lambda *a: pytest.fail("bad context reached preflight"))
    assert bench.main(["--model", "unused.gguf", "--ctx", "1", "--out-dir", str(tmp_path / "fresh")]) == 1
    assert not (tmp_path / "fresh").exists()


def test_different_effective_batch_configurations_cannot_be_averaged():
    values = runs()
    values[2]["result"]["n_batch"] = 64
    with pytest.raises(bench.BenchError, match="different effective"):
        bench._compare(values, 2, 1)


def test_timeout_keeps_partial_output_and_is_never_replayed(monkeypatch, tmp_path):
    calls = []
    def timeout(argv, **kw):
        calls.append(argv)
        raise subprocess.TimeoutExpired(argv, 5, output=b"partial stdout", stderr=b"partial stderr")
    monkeypatch.setattr(bench.subprocess, "run", timeout)
    with pytest.raises(bench.BenchError, match="did not finish"):
        bench._command(["probe.exe"], tmp_path, "timeout", environment={}, timeout=5)
    assert len(calls) == 1
    assert (tmp_path / "timeout.stdout.log").read_text() == "partial stdout"
    assert (tmp_path / "timeout.stderr.log").read_text() == "partial stderr"
    assert json.loads((tmp_path / "timeout.command.json").read_text())["returncode"] == 124
