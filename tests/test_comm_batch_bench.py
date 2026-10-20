"""Numerical evidence must prove exact generation and actual recovery."""
from copy import deepcopy
import importlib.util
from pathlib import Path
import pytest

spec = importlib.util.spec_from_file_location("comm_batch_bench", Path(__file__).resolve().parents[1] / "scripts/comm_batch_bench.py")
bench = importlib.util.module_from_spec(spec)
spec.loader.exec_module(bench)


def observation(mode="ngram", gamma=4):
    return {"gamma": gamma, "mode": mode, "tokens": [1, 2, 3, 4], "next_request_tokens": [1, 2, 3, 4],
        "speculation": {"replay_calls": 0 if mode == "accept" else 1, "accepted_tokens": 2}, "physically_stopped": True,
        "next_request_stopped": True}


def complete(run=None):
    values = [observation(mode, gamma) for mode in ("ngram", "accept", "reject_first", "reject_middle")
              for gamma in (2, 4, 8)]
    if run is not None:
        values = [run if (r["gamma"], r["mode"]) == (run["gamma"], run["mode"]) else r for r in values]
    return values


def test_exact_comparison_counts_both_generation_and_next_request():
    gate = bench.compare_runs([1, 2, 3, 4], complete())
    assert gate["passed"] and gate["compared_tokens"] == 96 and gate["mismatch_count"] == 0


@pytest.mark.parametrize("field", ["tokens", "next_request_tokens"])
def test_same_length_mismatch_is_visible_at_exact_token_index(field):
    run = observation(); run[field][2] = 8
    gate = bench.compare_runs([1, 2, 3, 4], complete(run))
    assert not gate["passed"] and gate["mismatch_count"] == 1
    assert gate["mismatches"][0]["index"] == 2 and gate["mismatches"][0]["field"] == field


@pytest.mark.parametrize("field,value", [("tokens", []), ("tokens", [1, 2]), ("next_request_tokens", None),
    ("next_request_tokens", [1, 2, 3, 4, 5]), ("physically_stopped", False)])
def test_partial_or_unclean_evidence_cannot_pass(field, value):
    run = observation(); run[field] = value
    with pytest.raises(bench.BenchError): bench.compare_runs([1, 2, 3, 4], complete(run))


@pytest.mark.parametrize("mode", ["reject_first", "reject_middle"])
def test_injected_rejection_requires_observed_replay(mode):
    run = observation(mode); run["speculation"]["replay_calls"] = 0
    with pytest.raises(bench.BenchError, match="restoration"):
        bench.compare_runs([1, 2, 3, 4], complete(run))


def test_all_accepted_injection_requires_verified_acceptance():
    run = observation("accept"); run["speculation"]["accepted_tokens"] = 0
    with pytest.raises(bench.BenchError, match="acceptance"):
        bench.compare_runs([1, 2, 3, 4], complete(run))


@pytest.mark.parametrize("reference,runs", [([], [observation()]), ([1, 2, 3, 4], [])])
def test_empty_reference_or_no_runs_is_rejected(reference, runs):
    with pytest.raises(bench.BenchError): bench.compare_runs(reference, runs)


@pytest.mark.parametrize("mode,index", [("accept", None), ("reject_first", 0), ("reject_middle", 1)])
def test_fault_injection_changes_exactly_one_candidate_once(mode, index):
    draft = bench.injected_drafter([1, 2, 3, 4, 5, 6], 2, mode, 128)
    first = draft([9, 9, 1], 3)
    expected = [2, 3, 4]
    if index is not None: expected[index] += 1
    assert first == expected
    assert draft([9, 9, 1], 3) == [2, 3, 4]


@pytest.mark.parametrize("mode", ["missing", "duplicate", "next_unclean", "accept_replayed"])
def test_complete_matrix_and_every_physical_cleanup_are_required(mode):
    runs = complete()
    if mode == "missing": runs.pop()
    if mode == "duplicate": runs[-1] = deepcopy(runs[0])
    if mode == "next_unclean": runs[0]["next_request_stopped"] = False
    if mode == "accept_replayed":
        next(r for r in runs if r["mode"] == "accept")["speculation"]["replay_calls"] = 1
    with pytest.raises(bench.BenchError): bench.compare_runs([1, 2, 3, 4], runs)


def test_non_windows_refuses_before_file_protection_or_copy(monkeypatch):
    from types import SimpleNamespace
    monkeypatch.setattr(bench, "os", SimpleNamespace(name="posix"))
    monkeypatch.setattr(bench, "_protected_model", lambda *_a: pytest.fail("model protection entered"))
    with pytest.raises(bench.BenchError, match="no-copy"):
        bench.run_native(SimpleNamespace(), {})


@pytest.mark.parametrize("close_failure", [False, True])
def test_failed_constructor_cleanup_owner_is_retired_and_included_in_receipt(tmp_path, monkeypatch, close_failure):
    from contextlib import contextmanager
    from types import SimpleNamespace
    import llama_keep_head
    class Process:
        dead = False
        def poll(self): return 0 if self.dead else None
        def terminate(self): self.dead = True
        def wait(self, **_kw): return 0
    class Owner:
        _worker = Process()
        closes = 0
        def close(self):
            self.closes += 1
            if close_failure: raise RuntimeError("close unavailable")
            self._worker.dead = True
    owner = Owner()
    def fail(*_a, **_kw):
        raise llama_keep_head.KeepHeadCleanupError(owner, RuntimeError("initialization"), RuntimeError("first cleanup"))
    @contextmanager
    def protected(path): yield path
    monkeypatch.setattr(bench, "os", SimpleNamespace(name="nt"))
    monkeypatch.setattr(bench, "_protected_model", protected)
    monkeypatch.setattr(bench, "inspect_pair", lambda *_a: ([{}, {}], [{}, {}]))
    monkeypatch.setattr(bench, "_sha256", lambda *_a: "0" * 64)
    monkeypatch.setattr(llama_keep_head, "KeepHeadUpstream", fail)
    monkeypatch.setattr(llama_keep_head, "keep_head_multi_position_available", lambda *_a: True)
    args = SimpleNamespace(head=tmp_path/"head.gguf", tail=tmp_path/"tail.gguf", head_manifest=tmp_path/"head.json",
        tail_manifest=tmp_path/"tail.json", head_expect=None, tail_expect=None, shim=tmp_path/"shim.dll", ctx=128, threads=4)
    report = {}
    with pytest.raises(llama_keep_head.KeepHeadCleanupError): bench.run_native(args, report)
    assert owner.closes == 1 and owner._worker.dead
    assert report["workers_exited"] is True
    assert bool(report["worker_cleanup_errors"]) is close_failure
