"""COMM-WIRE evidence rejects incomplete runs and uses real framed TCP."""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("comm_wire_bench", ROOT / "scripts/comm_wire_bench.py")
bench = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(bench)


def passes():
    return [{"name": name, "steps": [
        {"token_argmax": index, "hidden_sha256": "a" * 64,
         "wire": {"format": "binary" if name == "binary" or
                  (name == "mixed" and index % 2) else "json"}}
        for index in range(32)]} for name in ("json", "binary", "mixed")]


@pytest.mark.parametrize("mutation", ["truncated", "missing_pass", "duplicate_pass",
                                      "argmax", "hidden", "fake_format"])
def test_numerical_gate_rejects_incomplete_or_false_comparisons(mutation):
    rows = passes()
    if mutation == "truncated":
        rows[1]["steps"].pop()
    elif mutation == "missing_pass":
        rows.pop()
    elif mutation == "duplicate_pass":
        rows[1]["name"] = "json"
    elif mutation == "argmax":
        rows[1]["steps"][31]["token_argmax"] += 1
    elif mutation == "hidden":
        rows[2]["steps"][31]["hidden_sha256"] = "b" * 64
    else:
        rows[2]["steps"][31]["wire"]["format"] = "json"
    with pytest.raises(bench.BenchError):
        bench.compare_passes(rows, 32)


def test_numerical_gate_rejects_short_runs_even_when_all_tokens_match():
    rows = passes()
    for row in rows:
        row["steps"].pop()
    with pytest.raises(bench.BenchError, match=">=32"):
        bench.compare_passes(rows, 31)


def test_numerical_gate_accepts_complete_three_format_comparison():
    assert bench.compare_passes(passes(), 32)["passed"]


def test_tcp_mixed_peers_reconnect_and_disabled_master_use_production_code():
    report = bench.wire_bench(samples=1, warmup=0, sizes=(16384,))
    assert report["passed"]
    assert len(report["mixed_peers"]) == 12
    assert [row["binary_hidden"] for row in report["reconnects"]] == [True, False, True]
    for row in report["cases"]:
        binary = row["binary"]["samples"][0]
        old = row["json"]["samples"][0]
        assert binary["hidden_wire_bytes"] == 16384
        assert binary["frame_bytes"] < old["frame_bytes"]


def test_cli_failure_retains_partial_native_evidence(tmp_path, monkeypatch):
    monkeypatch.setattr(bench, "wire_bench", lambda **kw: {"passed": True})
    def fail(args, progress):
        progress["passes"] = [{"name": "json", "steps": []}]
        raise bench.BenchError("native test failed")
    monkeypatch.setattr(bench, "native_bench", fail)
    output = tmp_path / "failure.json"
    code = bench.main(["--native", "--head", "head.gguf", "--tail", "tail.gguf",
                       "--head-manifest", "head.json", "--tail-manifest", "tail.json",
                       "--json-out", str(output)])
    report = json.loads(output.read_text(encoding="utf-8"))
    assert code == 1 and not report["passed"]
    assert report["native"]["passes"][0]["steps"] == []
    assert "native test failed" in report["error"]


def test_handshake_failure_closes_both_tcp_sockets(monkeypatch):
    links = []
    original = bench.LoopbackLink
    def tracked():
        link = original()
        links.append(link)
        return link
    def reject(*args, **kwargs):
        raise bench.BenchError("injected ACK failure")
    monkeypatch.setattr(bench, "LoopbackLink", tracked)
    monkeypatch.setattr(bench.TaskWorkerControlPlane, "receive_on_worker", reject)
    with pytest.raises(bench.BenchError, match="injected ACK"):
        bench.Peer(bench.TaskWorkerControlPlane(), "failed_handshake", True)
    assert len(links) == 1 and links[0].closed
    assert links[0].sender.fileno() == links[0].receiver.fileno() == -1


@pytest.mark.parametrize("binary", [False, True])
def test_transfer_rejects_changed_but_schema_valid_positions(binary):
    peer = bench.Peer(bench.TaskWorkerControlPlane(), "metadata_check", binary)
    exchange = peer.link.exchange
    def changed(packet, **kwargs):
        frame = exchange(packet, **kwargs)
        frame["data"]["payload"]["positions"] = [7]
        return frame
    peer.link.exchange = changed
    try:
        with pytest.raises(bench.BenchError, match="metadata changed"):
            bench.transfer(peer, bytes(16384))
    finally:
        peer.close()


@pytest.mark.parametrize("fault", ["old_shim", "guard_off", "vocab", "declaration", "header", "layers"])
def test_native_preflight_rejects_invalid_loaded_contract(fault):
    from types import SimpleNamespace
    class Function:
        def __init__(self, value): self.value = value
        def __call__(self, *args): return self.value
    runner = SimpleNamespace(_worker=None, n_embd=4, n_layer=1, _handle=None,
        _lib=SimpleNamespace(qlh_kh_n_vocab=Function(128), qlh_kh_finite_logits_guard=Function(1)))
    info, declaration = {"actual": {"n_embd": 4}, "layers": 1}, {"hidden_size": 4}
    if fault == "old_shim":
        del runner._lib.qlh_kh_finite_logits_guard
    elif fault == "guard_off":
        runner._lib.qlh_kh_finite_logits_guard.value = 0
    elif fault == "vocab":
        runner._lib.qlh_kh_n_vocab.value = 129
    elif fault == "declaration":
        declaration["hidden_size"] = 777
    elif fault == "header":
        info["actual"]["n_embd"] = 777
    else:
        runner.n_layer = 99
    with pytest.raises(bench.BenchError):
        bench._native_contract(runner, info, declaration, 128)
