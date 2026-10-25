"""Adversarial calibration, identities, free-generation and publication tests.

No real model, benchmark or server is started. Runtime/host probes are isolated;
content/stat checks, JSON selection and atomic publication use actual temp files.
"""
from __future__ import annotations

from copy import deepcopy
import ctypes
import importlib.util
import json
import math
import os
from pathlib import Path
import sys
from types import SimpleNamespace, ModuleType

import pytest

import inference_calibration as cal

REAL_BINDING_RUNTIME = cal._binding_runtime

HOST = {"machine_id": "machine-1", "hostname": "local", "os": "TestOS", "release": "1",
        "version": "1", "architecture": "x86_64", "cpu": [{"model": "CPU-1"}], "isa": ["AVX2"],
        "logical_cores": 16, "physical_cores": 8, "affinity": list(range(16))}
CPU = {"name": "CPU", "description": "CPU-1", "backend": "CPU", "type": 0}
GPU = {"name": "CUDA0", "description": "GPU-1", "backend": "CUDA", "type": 1}
GPU_ID = {"index": "0", "uuid": "GPU-123", "pci_bus_id": "0000:01:00.0", "name": "GPU-1", "driver": "1"}
HELP = """-b, --batch-size N  logical batch (default: 2048)
-ub, --ubatch-size N physical batch (default: 512)
-tb, --threads-batch N threads (default:
                      same as --threads)
-ctk, --cache-type-k TYPE (default: f16)
-ctv, --cache-type-v TYPE (default: f16)
-fa, --flash-attn [on|off|auto] default: 'auto')
"""


class Constructor:
    def __init__(self, model_path, *, n_ctx=512, n_batch=512, n_ubatch=512,
                 n_threads=None, n_threads_batch=None, n_gpu_layers=0, flash_attn=False,
                 embedding=False, vocab_only=False, draft_model=None,
                 tokenizer=None, chat_handler=None, lora_base=None, lora_path=None,
                 offload_kqv=True, op_offload=None, swa_full=None, logits_all=False,
                 kv_overrides=None, type_k=None, type_v=None, chat_format=None, verbose=True):
        pass


@pytest.fixture
def local(tmp_path, monkeypatch):
    cal._HASH_CACHE.clear()
    model = tmp_path / "model.gguf"
    model.write_bytes(b"GGUF-test-model-1")
    native = tmp_path / "llama.dll"
    native.write_bytes(b"native-build-1")
    source = tmp_path / "llama.py"
    source.write_text("binding = 1", encoding="utf-8")
    binding = ModuleType("llama_cpp")
    binding.Llama = Constructor
    binding.__version__ = "0.3.35"
    binding.__file__ = str(tmp_path / "__init__.py")
    low = ModuleType("llama_cpp.llama_cpp")
    low.llama_context_default_params = lambda: SimpleNamespace(n_seq_max=1)
    low._lib = SimpleNamespace(_name=str(native))
    low.llama_print_system_info = lambda: b"CPU : AVX2 = 1 |"
    low.llama_backend_init = lambda: pytest.fail("identity probing must not initialize native backend")
    binding.llama_cpp = low
    binding.llama_vocab_is_eog = lambda vocab, token: False
    monkeypatch.setitem(sys.modules, "llama_cpp", binding)
    monkeypatch.setitem(sys.modules, "llama_cpp.llama_cpp", low)
    runtime = {"version": "0.3.35", "system_info": "CPU : AVX2 = 1 |", "devices": [deepcopy(CPU)],
               "files": [cal._file_identity(native), cal._file_identity(source)]}
    monkeypatch.setattr(cal, "_binding_runtime", lambda: (binding, deepcopy(runtime)))
    monkeypatch.setattr(cal, "_host_identity", lambda: deepcopy(HOST))
    monkeypatch.setattr(cal, "_cuda_inventory", lambda: [deepcopy(GPU_ID)])
    context = cal.build_binding_context(model, load_kwargs={"n_ctx": 4096, "n_threads": 8, "verbose": False})
    return SimpleNamespace(model=model, native=native, source=source, binding=binding, low=low,
                           runtime=runtime, context=context, tmp=tmp_path)


def make_record(context, **kwargs):
    baseline = {k: context["load_kwargs"][k] for k in ("flash_attn", "n_threads")}
    def runs(decode):
        return [{"tokens": list(range(32)), "finite_logits": True, "prefill_tps": 100.0,
                 "decode_tps": decode, "warmup": i == 0, "stop": "length"} for i in range(4)]
    record = {"record_id": "record-1", "context": deepcopy(context), "baseline": baseline,
              "candidate": {"flash_attn": True, "n_threads": 4},
              "protocol": {"warmup": 1, "samples": 3, "gen_tokens": 32, "prompt_tokens": [1, 2, 3],
                           "prompt_sha256": "a" * 64, "greedy": True, "free_generation": True},
              "baseline_runs": runs(100.0), "candidate_runs": runs(104.0), "accepted": True}
    record.update(kwargs)
    return record


def write_store(path, records):
    path.write_text(json.dumps({"schema_version": 1, "records": records}), encoding="utf-8")
    return path


def select(local, record):
    path = write_store(local.tmp / "calibration.json", [record])
    return cal.select_calibration(local.context, path=path)


def test_public_contract_and_positive(local):
    record = make_record(local.context)
    verdict = select(local, record)
    assert verdict["settings"] == record["candidate"]
    assert verdict.keys() == {"settings", "reason", "record_id", "source"}
    assert verdict["record_id"] == "record-1" and verdict["reason"] == "accepted"
    assert Path(verdict["source"]).is_absolute()
    assert local.context["execution"]["device"] == "cpu"
    assert local.context["load_kwargs"]["n_batch"] == 512
    assert local.context["load_kwargs"]["n_ubatch"] == 512


def test_default_path_and_override(local, monkeypatch):
    import config
    monkeypatch.setattr(config, "STATE_DIR", str(local.tmp))
    monkeypatch.delenv("QLH_INFER_CALIBRATION_PATH", raising=False)
    assert cal.calibration_path() == local.tmp / "inference-calibration.json"
    assert cal.select_calibration(local.context)["settings"] == {}
    path = write_store(local.tmp / "override.json", [make_record(local.context)])
    monkeypatch.setenv("QLH_INFER_CALIBRATION_PATH", str(path))
    assert cal.select_calibration(local.context)["settings"]["n_threads"] == 4


@pytest.mark.parametrize("target,key,value", [
    ("protocol", "warmup", 0), ("protocol", "samples", 2), ("protocol", "gen_tokens", 31),
    ("protocol", "greedy", False), ("protocol", "free_generation", False),
    ("protocol", "prompt_tokens", []), ("protocol", "prompt_sha256", "not-sha"),
    ("candidate", "n_threads", True), ("candidate", "n_threads", 0),
    ("candidate", "n_threads", -2), ("candidate", "n_threads", "4"),
    ("candidate", "flash_attn", 1), ("candidate", "type_k", "q8_0"),
    ("baseline", "n_threads", 4), ("baseline", "flash_attn", True),
    ("protocol", "teacher_forced", True), ("candidate", "future_setting", True),
])
def test_adversarial_fields(local, target, key, value):
    record = make_record(local.context)
    record[target][key] = value
    assert select(local, record)["settings"] == {}


@pytest.mark.parametrize("field,value", [
    ("finite_logits", False), ("finite_logits", 1), ("decode_tps", float("nan")),
    ("decode_tps", float("inf")), ("decode_tps", 0), ("prefill_tps", -1),
    ("prefill_tps", True), ("warmup", True), ("stop", "error"),
    ("tokens", list(range(31))), ("tokens", list(range(31)) + [True]),
    ("tokens", list(range(31)) + [999]),
])
def test_each_run_must_validate(local, field, value):
    record = make_record(local.context)
    record["candidate_runs"][-1][field] = value
    assert select(local, record)["settings"] == {}


@pytest.mark.parametrize("level", ["record", "context", "runtime", "host", "execution", "run", "model", "stat"])
def test_strict_fields_at_each_level(local, level):
    record = make_record(local.context)
    targets = {"record": record, "context": record["context"], "runtime": record["context"]["runtime"],
               "host": record["context"]["host"], "execution": record["context"]["execution"],
               "run": record["baseline_runs"][0], "model": record["context"]["model"],
               "stat": record["context"]["model"]["stat"]}
    targets[level]["untrusted_extra"] = True
    assert select(local, record)["settings"] == {}


@pytest.mark.parametrize("phase,rate", [("decode_tps", 102.99), ("prefill_tps", 97.99)])
def test_performance_thresholds_and_untrusted_accepted(local, phase, rate):
    record = make_record(local.context, accepted=True)
    for run in record["candidate_runs"]:
        run[phase] = rate
    assert select(local, record)["reason"] == "performance_gate_failed"


def test_median_thresholds_warmups_excluded(local):
    record = make_record(local.context)
    record["candidate_runs"][0].update(prefill_tps=0.001, decode_tps=0.001)
    record["baseline_runs"][0].update(prefill_tps=1e9, decode_tps=1e9)
    for run in record["candidate_runs"][1:]:
        run.update(prefill_tps=98, decode_tps=103)
    assert select(local, record)["settings"]
    record["baseline_runs"][0]["tokens"][-1] = 999
    assert select(local, record)["settings"] == {}


def test_all_run_counts_and_baseline_repeats_checked(local):
    record = make_record(local.context)
    record["candidate_runs"].pop()
    assert select(local, record)["settings"] == {}
    record = make_record(local.context)
    record["baseline_runs"][2]["tokens"][-1] = 999
    assert select(local, record)["settings"] == {}


@pytest.mark.parametrize("key,value", [("n_ctx", 2048), ("n_gpu_layers", 2), ("n_batch", 256),
                                      ("n_ubatch", 256), ("n_threads", 4), ("n_seq_max", 2),
                                      ("offload_kqv", False), ("flash_attn", True), ("type_k", 8)])
def test_parameter_drift_never_reuses_record(local, key, value):
    path = write_store(local.tmp / "record.json", [make_record(local.context)])
    changed = deepcopy(local.context)
    changed["load_kwargs"][key] = value
    assert cal.select_calibration(changed, path=path)["settings"] == {}


def test_model_and_native_file_drift_with_mtime_restored(local):
    path = write_store(local.tmp / "record.json", [make_record(local.context)])
    for target in (local.model, local.native):
        original = target.read_bytes()
        info = target.stat()
        target.write_bytes(b"X" * len(original))
        os.utime(target, ns=(info.st_atime_ns, info.st_mtime_ns))
        assert cal.select_calibration(local.context, path=path)["settings"] == {}
        assert cal._file_identity(target)["sha256"] != cal.__dict__["hashlib"].sha256(original).hexdigest()
        target.write_bytes(original)


def test_same_contents_different_path_and_profile_not_identity(local):
    other = local.tmp / "renamed.gguf"
    other.write_bytes(local.model.read_bytes())
    context = cal.build_binding_context(other, load_kwargs={"n_ctx": 4096, "n_threads": 8, "verbose": False}, profile={"tier": "workstation"})
    path = write_store(local.tmp / "record.json", [make_record(local.context)])
    assert cal.select_calibration(context, path=path)["reason"] == "context_mismatch"
    same = cal.build_binding_context(local.model, load_kwargs={"n_ctx": 4096, "n_threads": 8, "verbose": False}, profile={"tier": "mobile", "gpu": {"cuda_available": True}})
    assert same == local.context


def test_environment_and_host_drift(local, monkeypatch):
    path = write_store(local.tmp / "record.json", [make_record(local.context)])
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "7")
    assert cal.select_calibration(local.context, path=path)["settings"] == {}
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES")
    monkeypatch.setattr(cal, "_host_identity", lambda: {**HOST, "hostname": "different"})
    assert cal.select_calibration(local.context, path=path)["settings"] == {}


def test_conflicts_and_duplicate_ids_never_pick_first(local):
    first = make_record(local.context)
    second = make_record(local.context, record_id="record-2")
    second["candidate"]["n_threads"] = 6
    path = write_store(local.tmp / "record.json", [first, second])
    assert cal.select_calibration(local.context, path=path)["reason"] == "conflicting_records"
    second["candidate"] = dict(first["candidate"])
    write_store(path, [second, first])
    assert cal.select_calibration(local.context, path=path)["record_id"] == "record-1"
    second["record_id"] = first["record_id"]
    write_store(path, [first, second])
    assert cal.select_calibration(local.context, path=path)["settings"] == {}


@pytest.mark.parametrize("raw", ['{"schema_version":1,"records":[],"records":[]}',
                                  '{"schema_version":1,"records":[],"extra":true}',
                                  '{"schema_version":true,"records":[]}',
                                  '{"schema_version":1,"records":[NaN]}',
                                  '{"llama-bench":"old positive","accepted":true}',
                                  'not json', '[]'])
def test_corrupt_json_and_old_bench_rejected(local, raw):
    path = local.tmp / "corrupt.json"
    path.write_text(raw, encoding="utf-8")
    assert cal.select_calibration(local.context, path=path)["settings"] == {}


def test_size_limit_and_bounded_stat_cache(local, monkeypatch):
    path = write_store(local.tmp / "record.json", [make_record(local.context)])
    monkeypatch.setattr(cal, "MAX_STORE_BYTES", 20)
    assert cal.select_calibration(local.context, path=path)["settings"] == {}
    monkeypatch.setattr(cal, "_HASH_CACHE_LIMIT", 3)
    for index in range(10):
        target = local.tmp / f"hash-{index}.bin"
        target.write_bytes(bytes([index]))
        cal._file_identity(target)
    assert len(cal._HASH_CACHE) == 3


@pytest.mark.parametrize("kwargs", [{"rpc_servers": []}, {"devices": [0]}, {"embedding": True},
                                    {"draft_model": object()}, {"tokenizer": lambda: None},
                                    {"n_ctx": 0}, {"n_threads": True}, {"flash_attn": 1},
                                    {"kv_overrides": {"bad": lambda: 0}}])
def test_unsupported_binding_contexts_rejected(local, kwargs):
    with pytest.raises(ValueError):
        cal.build_binding_context(local.model, load_kwargs={"n_ctx": 4096, "n_threads": 8, **kwargs})


def test_nonlocal_model_rejected(local):
    for path in ("https://host/model.gguf", "//server/share/model.gguf"):
        with pytest.raises(ValueError):
            cal.build_binding_context(path, load_kwargs={"n_threads": 8})


def server_setup(local, monkeypatch, cuda=True):
    exe = local.tmp / "llama-server.exe"
    exe.write_bytes(b"server-build-1")
    def probe(argv):
        if argv[-1] == "--version":
            return "version: server-1 compiler-1"
        if argv[-1] == "--help":
            return HELP
        if argv[-1] == "--list-devices":
            return "Available devices:\n CUDA0: GPU-1 (8000 MiB, 123 MiB free)" if cuda else "Available devices:"
        pytest.fail("unexpected identity probe")
    monkeypatch.setattr(cal, "_run_probe", probe)
    params = {"n_ctx": 4096, "n_gpu_layers": 99, "n_threads": None, "cpu_moe": True,
              "spec_type": "ngram-mod", "chat_template_kwargs": {"enable_thinking": False}, "mmproj_path": "", "argv_prefix": []}
    return exe, params


def test_server_auto_context_and_selection_isolated(local, monkeypatch):
    exe, params = server_setup(local, monkeypatch)
    context = cal.build_server_context(local.model, exe=exe, load_kwargs=params)
    assert context["path"] == "llama_server"
    assert context["load_kwargs"]["n_threads"] is None
    assert context["load_kwargs"]["flash_attn"] is None
    assert context["load_kwargs"]["n_threads_batch"] is None
    assert context["runtime"]["system_info"] is None
    assert context["execution"]["device"] == "cuda"
    path = write_store(local.tmp / "server-record.json", [make_record(context)])
    assert cal.select_calibration(context, path=path)["settings"] == {"flash_attn": True, "n_threads": 4}
    assert cal.select_calibration(local.context, path=path)["settings"] == {}
    explicit = cal.build_server_context(local.model, exe=exe, load_kwargs={**params, "flash_attn": False, "n_threads": 8})
    assert cal.select_calibration(explicit, path=path)["settings"] == {}
    exe.write_bytes(b"server-build-2")
    assert cal.select_calibration(context, path=path)["settings"] == {}


def test_no_binding_isa_or_gpu_hardware_substitution_for_server(local, monkeypatch):
    exe, params = server_setup(local, monkeypatch, cuda=False)
    monkeypatch.setattr(cal, "_binding_runtime", lambda: pytest.fail("server must not inspect binding"))
    monkeypatch.setattr(cal, "_cuda_inventory", lambda: pytest.fail("CPU server must not claim hardware CUDA"))
    context = cal.build_server_context(local.model, exe=exe, load_kwargs=params, profile={"gpu": {"cuda_available": True}})
    assert context["execution"]["device"] == "cpu"


@pytest.mark.parametrize("change", [{"argv_prefix": ["wrapper"]}, {"mmproj_path": "vision.gguf"},
                                   {"rpc_servers": "localhost:1"}, {"spec_type": "draft-eagle3"},
                                   {"n_threads": -1}, {"chat_template_kwargs": lambda: None}])
def test_server_unsupported_paths(local, monkeypatch, change):
    exe, params = server_setup(local, monkeypatch)
    with pytest.raises(ValueError):
        cal.build_server_context(local.model, exe=exe, load_kwargs={**params, **change})


def test_frozen_binding_uses_real_exe_and_pyz_without_source(local, monkeypatch):
    exe = local.tmp / "product.exe"
    pyz = local.tmp / "PYZ-00.pyz"
    exe.write_bytes(b"frozen-python-code")
    pyz.write_bytes(b"packed-binding-code")
    monkeypatch.setattr(sys, "frozen", True, raising=False)
    monkeypatch.setattr(sys, "executable", str(exe))
    monkeypatch.setattr(sys, "_MEIPASS", str(local.tmp), raising=False)
    monkeypatch.setattr(cal, "_mapped_libraries", lambda: [local.native])
    monkeypatch.setattr(cal, "_binding_devices", lambda paths: [deepcopy(CPU)])
    monkeypatch.setattr(cal.inspect, "getsource", lambda _: (_ for _ in ()).throw(OSError("no source in PYZ")))
    binding, runtime = REAL_BINDING_RUNTIME()
    paths = {Path(f["path"]) for f in runtime["files"]}
    assert paths == {exe, pyz, local.native}
    params = cal._binding_parameters(binding, {"n_ctx": 4096, "n_threads": 8})
    assert params["n_batch"] == 512 and params["n_threads_batch"] >= 1
    local.binding.__version__ = "future-unknown"
    with pytest.raises(ValueError, match="unknown binding batch semantics"):
        cal._binding_parameters(local.binding, {"n_ctx": 4096, "n_threads": 8})


def test_binding_probe_never_initializes_native_and_rejects_foreign_loaded_dll(local, monkeypatch):
    monkeypatch.setattr(cal, "_mapped_libraries", lambda: [local.native])
    monkeypatch.setattr(cal, "_binding_devices", lambda paths: [deepcopy(CPU)])
    REAL_BINDING_RUNTIME()  # fixture's init function raises if called
    foreign_dir = local.tmp / "other-build"
    foreign_dir.mkdir()
    foreign = foreign_dir / "ggml-base.dll"
    foreign.write_bytes(b"foreign-native")
    monkeypatch.setattr(cal, "_mapped_libraries", lambda: [local.native, foreign])
    with pytest.raises(ValueError, match="foreign"):
        REAL_BINDING_RUNTIME()


@pytest.fixture
def bench():
    location = Path(__file__).resolve().parents[1] / "scripts" / "inference_switch_bench.py"
    spec = importlib.util.spec_from_file_location("calibration_bench_under_test", location)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_cli_production_defaults_and_server_flag_omissions(bench, monkeypatch):
    from llama_engine import LlamaCppEngine
    monkeypatch.setattr(LlamaCppEngine, "_auto_threads", lambda self: 8)
    args = bench._parser().parse_args(["--model", "model.gguf", "--prompt", "hello"])
    kwargs = bench._baseline_kwargs(args)
    assert kwargs["n_threads"] == 8 and kwargs["flash_attn"] is False
    assert "n_batch" not in kwargs and "n_ubatch" not in kwargs
    args = bench._parser().parse_args(["--path", "llama_server", "--server-exe", "server.exe", "--model", "model.gguf", "--prompt", "hello"])
    kwargs = bench._baseline_kwargs(args)
    argv = bench._server_spec(args, kwargs).to_argv(host="127.0.0.1", port=1234)
    assert kwargs["chat_template_kwargs"] == {"enable_thinking": False}
    assert kwargs["n_threads"] is None and kwargs["flash_attn"] is None
    assert "--threads" not in argv and "--flash-attn" not in argv
    assert "--batch-size" not in argv and "--parallel" not in argv


def test_binding_greedy_freely_feeds_generated_tokens_and_counts_phases(local, bench):
    import numpy as np
    class FakeModel:
        def __init__(self):
            self.values = (ctypes.c_float * 128)()
            self._ctx = SimpleNamespace(get_logits=lambda: self.values)
            self._model = SimpleNamespace(vocab=object())
            self.calls = []
            self.step = 0
        def reset(self):
            self.step = 0
        def n_vocab(self):
            return 128
        def eval(self, tokens):
            self.calls.append(list(tokens))
            self.step += 1
            self.values[:] = np.zeros(128)
            self.values[self.step] = 10.0
    model = FakeModel()
    run, error = bench._binding_once(model, [90, 91], 32, False)
    assert error is None
    assert run["tokens"] == list(range(1, 33))
    assert model.calls == [[90, 91], *[[t] for t in range(1, 32)]]
    assert run["prefill_tps"] > 0 and run["decode_tps"] > 0
    assert run["finite_logits"] is True and run["stop"] == "length"
    model.values[0] = float("nan")
    model.eval = lambda tokens: None
    run, error = bench._binding_once(model, [90, 91], 32, False)
    assert "nonfinite" in error and run["finite_logits"] is False


def test_server_top_probabilities_do_not_attest_rawfinite(bench, monkeypatch):
    events = []
    for token in range(32):
        events.append({"tokens": [token], "stop": False,
                       "completion_probabilities": [{"id": token, "top_logprobs": [{"id": token, "logprob": 0.0}]}]})
    events.append({"tokens": [], "stop": True, "stop_type": "limit", "truncated": False,
                   "tokens_predicted": 32, "timings": {"prompt_n": 3, "prompt_ms": 10.0, "predicted_n": 32, "predicted_ms": 100.0}})
    import io
    class Opener:
        def open(self, request, timeout):
            self.body = json.loads(request.data)
            return io.BytesIO(b"".join(b"data: " + json.dumps(v).encode() + b"\n\n" for v in events))
    opener = Opener()
    monkeypatch.setattr(bench.urllib.request, "build_opener", lambda *args: opener)
    run, error = bench._server_once("http://127.0.0.1:1", [1, 2, 3], 32, False, 128)
    assert run["tokens"] == list(range(32)) and run["decode_tps"] == 320
    assert run["finite_logits"] is False and "raw-logits" in error
    assert opener.body["cache_prompt"] is False and opener.body["n_probs"] == 1
    assert not bench._raw_logits_finite([0.0, float("inf")], 2)


def test_atomic_publication_rechecks_evidence_and_conflicts(local, bench):
    record = make_record(local.context)
    path = local.tmp / "published.json"
    assert bench._publish(record, path) == str(path)
    assert cal.select_calibration(local.context, path=path)["settings"]
    before = path.read_bytes()
    rejected = deepcopy(record)
    rejected["candidate_runs"][-1]["tokens"][0] = 999
    with pytest.raises(ValueError, match="rejected"):
        bench._publish(rejected, path)
    assert path.read_bytes() == before
    conflict = deepcopy(record)
    conflict["record_id"] = "conflict"
    conflict["candidate"]["n_threads"] = 6
    with pytest.raises(ValueError, match="conflicts"):
        bench._publish(conflict, path)
    assert path.read_bytes() == before
    assert list(local.tmp.glob(".*.tmp")) == []


def test_cli_load_close_order_and_default_report_only(local, bench, monkeypatch, capsys):
    order = []
    class Model:
        def __init__(self, model_path, **kwargs):
            assert not order or order[-1] == "close"
            order.append("load")
        def tokenize(self, *args, **kwargs):
            return [1, 2, 3]
        def n_ctx(self):
            return 4096
        def close(self):
            order.append("close")
    monkeypatch.setattr(local.binding, "Llama", Model)
    monkeypatch.setattr(bench, "_context", lambda args, kwargs: deepcopy(local.context))
    monkeypatch.setattr(bench, "_baseline_kwargs", lambda args: {"n_ctx": 4096, "n_threads": 8, "flash_attn": False})
    count = 0
    def once(model, prompt, gen, warmup):
        nonlocal count
        count += 1
        return {"tokens": list(range(32)), "finite_logits": True, "prefill_tps": 100.0,
                "decode_tps": 100.0 if count <= 4 else 110.0, "warmup": warmup, "stop": "length"}, None
    monkeypatch.setattr(bench, "_binding_once", once)
    monkeypatch.setattr(bench, "_publish", lambda *args: pytest.fail("default must not publish"))
    output = local.tmp / "report.json"
    assert bench.main(["--model", str(local.model), "--prompt", "hello", "--gen", "32",
                       "--threads", "4", "--flash-attn", "on", "--json-out", str(output)]) == 0
    assert order == ["load", "close", "load", "close"]
    report = json.loads(output.read_text(encoding="utf-8"))
    assert len(report["baseline"]["runs"]) == 4
    assert report["candidates"][0]["verdict"]["settings"]
    assert report["publication"]["reason"] == "report_only"
    assert len(capsys.readouterr().out) < 1500


def test_cli_rejected_cells_preserve_counterexamples_and_never_publish(local, bench, monkeypatch, capsys):
    baseline = {"settings": {"flash_attn": False, "n_threads": 8}, "context": local.context,
                "prompt_tokens": [1, 2, 3], "runs": make_record(local.context)["baseline_runs"], "errors": []}
    candidate = deepcopy(baseline)
    candidate["settings"] = {"flash_attn": True, "n_threads": 4}
    candidate["runs"][-1]["tokens"][0] = 999
    cells = iter([baseline, candidate])
    monkeypatch.setattr(bench, "_baseline_kwargs", lambda args: {"n_threads": 8, "flash_attn": False})
    monkeypatch.setattr(bench, "_context", lambda args, kwargs: local.context)
    monkeypatch.setattr(bench, "_measure", lambda args, kwargs: next(cells))
    monkeypatch.setattr(bench, "_publish", lambda *args: pytest.fail("counterexample must not publish"))
    output = local.tmp / "counterexamples.json"
    publish = local.tmp / "forbidden.json"
    assert bench.main(["--model", str(local.model), "--prompt", "hello", "--gen", "32", "--threads", "4",
                       "--flash-attn", "on", "--json-out", str(output), "--calibration-out", str(publish)]) == 1
    assert not publish.exists()
    assert json.loads(output.read_text())["candidates"][0]["record"]["candidate_runs"][-1]["tokens"][0] == 999
    capsys.readouterr()


def test_windows_mapped_network_drive_rejected(local, monkeypatch):
    class RemoteDrive:
        def __call__(self, anchor):
            return 4  # DRIVE_REMOTE, even when accessed via a drive letter
    monkeypatch.setattr(sys, "platform", "win32")
    monkeypatch.setattr(cal.ctypes, "WinDLL", lambda *args, **kwargs: SimpleNamespace(GetDriveTypeW=RemoteDrive()), raising=False)
    with pytest.raises(ValueError, match="non-local"):
        cal._local_path(local.model)


@pytest.mark.parametrize("mode,expected_threads,expected_reason", [
    ("accepted", 4, "accepted"),
    ("chatml", 4, "accepted"),
    ("disabled", 8, "auto_disabled"),
    ("different_context", 8, "context_mismatch"),
    ("corrupt", 8, "invalid_or_drifted"),
    ("explicit", 6, "explicit_configuration"),
    ("environment", 5, "explicit_configuration"),
])
def test_calibration_record_reaches_loader_and_current_model_api(local, monkeypatch, mode,
                                                                expected_threads, expected_reason):
    """Synthetic throughput proves wiring; it makes no hardware speed claim."""
    import config
    import llama_engine
    import inference_switches as switches
    from api import routes_models
    context = local.context
    if mode == "chatml":
        context = cal.build_binding_context(local.model, load_kwargs={
            "n_ctx": 4096, "n_threads": 8, "verbose": False, "chat_format": "chatml"})
    path = write_store(local.tmp / "loader-calibration.json", [make_record(context)])
    if mode == "corrupt":
        path.write_text("{broken", encoding="utf-8")
    monkeypatch.setattr(config, "NGRAM_SPEC_ENABLED", False)
    monkeypatch.setattr(llama_engine.LlamaCppEngine, "_auto_threads", lambda self: 8)
    monkeypatch.setattr(llama_engine.LlamaCppEngine, "_install_chat_template_controls", lambda self: None)
    monkeypatch.setattr("device_profiler.get_profile", lambda: SimpleNamespace(to_dict=lambda: {}))
    seen = []
    def constructor(kwargs, sequences):
        seen.append(dict(kwargs))
        return SimpleNamespace(close=lambda: None)
    monkeypatch.setattr(llama_engine, "_new_llama_with_seq_max", constructor)
    engine = llama_engine.LlamaCppEngine()
    engine._environment = {switches.ENV_CALIBRATION_PATH: str(path)}
    if mode == "disabled":
        engine._environment[switches.ENV_AUTO] = "0"
    elif mode == "environment":
        engine._environment[switches.ENV_THREADS] = "5"
    kwargs = {"n_ctx": 2048 if mode == "different_context" else 4096}
    if mode == "explicit":
        kwargs["n_threads"] = 6
    if mode == "chatml":
        kwargs["model_profile"] = {"template": "qwen_chat_v1"}
    try:
        engine._load_model_runtime(str(local.model), **kwargs)
        state = engine.get_model_info()["inference_switches"]
        assert seen[0]["n_threads"] == expected_threads
        assert state["effective"]["n_threads"] == expected_threads
        assert state["requested"]["selection"]["reason"].startswith(expected_reason)
        if mode in {"accepted", "chatml"}:
            assert seen[0]["flash_attn"] is True
            assert state["sources"]["n_threads"] == "local_calibration"
        manager = SimpleNamespace(active_model_id="test-model", get_model_info=engine.get_model_info,
                                  get_memory_usage=lambda: {})
        api = SimpleNamespace(_pipeline_model_is_prepared=lambda: False,
            _node_inference_mode=lambda: "local", model_manager=manager,
            model_host=SimpleNamespace(model_loaded=True, current_quant="gguf"))
        monkeypatch.setattr(routes_models, "_api_module", api)
        assert routes_models._current_model_payload()["inference_switches"] == state
    finally:
        engine.unload()
    assert engine.get_model_info()["inference_switches"] is None



def test_cli_binding_chat_format_matches_production_context(local, bench):
    args = bench._parser().parse_args(["--model", str(local.model), "--prompt", "test",
                                      "--baseline-threads", "8", "--chat-format", "chatml"])
    kwargs = bench._baseline_kwargs(args)
    assert kwargs["chat_format"] == "chatml"
    expected = cal.build_binding_context(local.model, load_kwargs={
        "n_ctx": 4096, "n_threads": 8, "verbose": False, "chat_format": "chatml"})
    assert cal.build_binding_context(local.model, load_kwargs=kwargs) == expected
    args.path = cal.SERVER
    with pytest.raises(ValueError, match="binding only"):
        bench._baseline_kwargs(args)
