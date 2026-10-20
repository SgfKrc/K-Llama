"""Local whole-model A/B calibration. No model load or benchmark on selection.

Public builders return JSON context or raise ValueError. Profiles never replace
local identity probes. Paths are llama_cpp_binding and llama_server, isolated.
Server baseline n_threads=None / flash_attn=None mean omitted flags (EXE auto),
never CPU-thread guesses or flash-attn off. Candidate settings ONLY permit
flash_attn: bool and n_threads: positive int. KV settings are identity, not knobs.

Strict JSON schema (no additional fields at any of these levels):
store: {schema_version, records}
record: {record_id, context, baseline, candidate, protocol, baseline_runs,
         candidate_runs, accepted} (accepted is informational and recomputed)
context: {schema_version, path, model, runtime, host, execution, load_kwargs}
protocol: {warmup, samples, gen_tokens, prompt_tokens, prompt_sha256, greedy,
           free_generation}
run: {tokens, finite_logits, prefill_tps, decode_tps, warmup, stop}
Baseline has exactly flash_attn/n_threads; candidate is a nonempty subset.
Runs contain complete freely generated token IDs, including EOS when sampled;
warmups precede measured samples. Every run, including warmup, must agree.
Warmups are excluded from medians. Rates are positive finite tokens/second.
File identities include SHA256, path, resolved_path and stat stability checks.
"""
from __future__ import annotations

from collections import OrderedDict
import ctypes
import hashlib
import inspect
import json
import math
import multiprocessing
import os
from pathlib import Path
import platform
import re
import stat
import statistics
import sys
import itertools
import subprocess
import threading
from typing import Any, Mapping

SCHEMA_VERSION = 1
BINDING = "llama_cpp_binding"
SERVER = "llama_server"
MAX_STORE_BYTES = 16 * 1024 * 1024
MAX_RECORDS = 256
_HASH_CACHE_LIMIT = 128
_HASH_CACHE: OrderedDict[tuple, str] = OrderedDict()
_HASH_LOCK = threading.RLock()
_CONTEXT_FIELDS = {"schema_version", "path", "model", "runtime", "host", "execution", "load_kwargs"}
_RECORD_FIELDS = {"record_id", "context", "baseline", "candidate", "protocol", "baseline_runs", "candidate_runs", "accepted"}
_PROTOCOL_FIELDS = {"warmup", "samples", "gen_tokens", "prompt_tokens", "prompt_sha256", "greedy", "free_generation"}
_RUN_FIELDS = {"tokens", "finite_logits", "prefill_tps", "decode_tps", "warmup", "stop"}
_FILE_FIELDS = {"path", "resolved_path", "sha256", "stat"}
_STAT_FIELDS = {"dev", "ino", "size", "mtime_ns", "ctime_ns"}
_SHA = re.compile(r"[0-9a-f]{64}\Z")
_BINDING_KEYS = set("""n_gpu_layers split_mode main_gpu tensor_split vocab_only use_mmap
use_mlock kv_overrides seed n_ctx n_batch n_ubatch n_threads n_threads_batch
rope_scaling_type pooling_type attention_type rope_freq_base rope_freq_scale
yarn_ext_factor yarn_attn_factor yarn_beta_fast yarn_beta_slow yarn_orig_ctx
logits_all embedding offload_kqv flash_attn op_offload swa_full no_perf
last_n_tokens_size lora_base lora_scale lora_path numa chat_format chat_handler
draft_model tokenizer type_k type_v spm_infill verbose""".split())
_SERVER_KEYS = set("""n_ctx n_gpu_layers n_threads n_threads_batch n_batch n_ubatch
n_seq_max flash_attn cpu_moe spec_type cache_type_k cache_type_v main_gpu
split_mode tensor_split rope_scaling_type rope_freq_base rope_freq_scale
yarn_ext_factor yarn_attn_factor yarn_beta_fast yarn_beta_slow yarn_orig_ctx
numa op_offload swa_full cache_reuse ctx_checkpoints use_mmap use_mlock
kv_unified cont_batch fit chat_template_kwargs mmproj_path argv_prefix""".split())


def _require(condition: Any, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _integer(value: Any, *, minimum: int = 1) -> bool:
    return type(value) is int and minimum <= value <= 2**31 - 1


def _json_value(value: Any) -> Any:
    if value is None or type(value) in (str, bool, int):
        return value
    if type(value) is float:
        _require(math.isfinite(value), "nonfinite parameter")
        return value
    if type(value) in (list, tuple):
        return [_json_value(v) for v in value]
    if isinstance(value, Mapping):
        _require(all(type(k) is str for k in value), "non-string load key")
        return {k: _json_value(v) for k, v in sorted(value.items())}
    raise ValueError("unsupported callable/pointer/load parameter")


def _local_path(value: Any) -> Path:
    raw = os.fspath(value)
    _require(isinstance(raw, str) and bool(raw), "empty/non-local path")
    _require(not raw.startswith(("\\\\", "//")) and "://" not in raw, "non-local path")
    path = Path(os.path.abspath(os.path.expanduser(raw)))
    if sys.platform == "win32":
        drive_type = ctypes.WinDLL("kernel32", use_last_error=True).GetDriveTypeW
        drive_type.argtypes, drive_type.restype = [ctypes.c_wchar_p], ctypes.c_uint
        _require(drive_type(path.anchor) in {2, 3, 6}, "non-local or unknown filesystem device")
    _require(not str(path.resolve()).startswith(("\\\\", "//")), "non-local resolved path")
    return path


def _stat(value: os.stat_result) -> dict:
    _require(stat.S_ISREG(value.st_mode), "identity requires regular file")
    return {"dev": value.st_dev, "ino": value.st_ino, "size": value.st_size,
            "mtime_ns": value.st_mtime_ns, "ctime_ns": value.st_ctime_ns}


def _file_identity(value: Any) -> dict:
    path = _local_path(value)
    resolved = path.resolve(strict=True)
    path_before = _stat(path.stat())
    # On Windows Python 3.12, path.stat ctime can be birth time while fstat
    # exposes NTFS change time. Compare like with like and cache by fstat.
    with path.open("rb") as handle:
        before = _stat(os.fstat(handle.fileno()))
        _require(all(before[k] == path_before[k] for k in ("dev", "ino", "size", "mtime_ns")),
                 "file changed before hashing")
        key = (str(path), str(resolved), *before.values())
        with _HASH_LOCK:
            digest = _HASH_CACHE.get(key)
            if digest is not None:
                _HASH_CACHE.move_to_end(key)
        if digest is None:
            h = hashlib.sha256()
            while chunk := handle.read(4 * 1024 * 1024):
                h.update(chunk)
            digest = h.hexdigest()
        _require(_stat(os.fstat(handle.fileno())) == before, "file changed while hashing")
        _require(_stat(path.stat()) == path_before and path.resolve(strict=True) == resolved,
                 "file changed during identity check")
    with _HASH_LOCK:
        _HASH_CACHE[key] = digest
        _HASH_CACHE.move_to_end(key)
        while len(_HASH_CACHE) > _HASH_CACHE_LIMIT:
            _HASH_CACHE.popitem(last=False)
    return {"path": str(path), "resolved_path": str(resolved), "sha256": digest, "stat": before}


def _environment() -> dict:
    prefixes = ("CUDA_", "GGML_", "LLAMA_", "OMP_", "GOMP_", "KMP_", "MKL_", "HIP_", "ROCR_", "CUBLAS_", "NVIDIA_")
    values = {k: v for k, v in sorted(os.environ.items()) if k.startswith(prefixes)}
    for key in ("CUDA_VISIBLE_DEVICES", "LD_LIBRARY_PATH", "LD_PRELOAD", "DYLD_LIBRARY_PATH"):
        values.setdefault(key, os.environ.get(key))
    return values


def _host_identity() -> dict:
    import psutil

    system = platform.system()
    if system == "Windows":
        import winreg

        with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, r"SOFTWARE\Microsoft\Cryptography") as key:
            machine_id = winreg.QueryValueEx(key, "MachineGuid")[0]
        cpus = []
        with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, r"HARDWARE\DESCRIPTION\System\CentralProcessor") as root:
            for index in range(winreg.QueryInfoKey(root)[0]):
                with winreg.OpenKey(root, winreg.EnumKey(root, index)) as key:
                    cpus.append({name: winreg.QueryValueEx(key, name)[0]
                                 for name in ("Identifier", "ProcessorNameString", "VendorIdentifier")})
        probe = ctypes.WinDLL("kernel32", use_last_error=True).IsProcessorFeaturePresent
        probe.argtypes, probe.restype = [ctypes.c_uint], ctypes.c_int
        isa = [i for i in range(64) if probe(i)]
    elif system == "Linux":
        machine_id = Path("/etc/machine-id").read_text(encoding="ascii").strip()
        cpus, isa = [], []
        fields = {"vendor_id", "model name", "cpu family", "model", "stepping", "CPU implementer", "CPU part", "CPU revision", "Hardware"}
        for block in Path("/proc/cpuinfo").read_text(encoding="utf-8").split("\n\n"):
            pairs = dict((k.strip(), v.strip()) for line in block.splitlines()
                         if ":" in line for k, v in [line.split(":", 1)])
            if pairs:
                cpus.append({k: pairs[k] for k in sorted(fields & pairs.keys())})
                isa.append(sorted((pairs.get("flags") or pairs.get("Features") or "").split()))
        _require(isa and all(isa), "host ISA unavailable")
    else:
        raise ValueError("host identity unavailable for this OS")
    process = psutil.Process()
    affinity = sorted(process.cpu_affinity()) if hasattr(process, "cpu_affinity") else sorted(os.sched_getaffinity(0))
    _require(machine_id and cpus and all(cpus) and isa and affinity, "incomplete host/CPU/ISA identity")
    return {"machine_id": machine_id, "hostname": platform.node(), "os": system,
            "release": platform.release(), "version": platform.version(),
            "architecture": platform.machine(), "cpu": cpus, "isa": isa,
            "logical_cores": psutil.cpu_count(), "physical_cores": psutil.cpu_count(logical=False), "affinity": affinity}


def _native(path: Path) -> bool:
    return bool(re.search(r"(?:\.dll|\.dylib|\.so(?:\.\d+)*)$", path.name, re.I))


def _mapped_libraries() -> list[Path]:
    import psutil

    if sys.platform.startswith("linux"):
        paths = set()
        for line in Path("/proc/self/maps").read_text(encoding="utf-8").splitlines():
            fields = line.split(None, 5)
            if len(fields) != 6 or not fields[5].startswith("/"):
                continue
            raw = fields[5]
            if raw.endswith(" (deleted)") and _native(Path(raw[:-10])):
                raise ValueError("loaded native image was deleted/replaced")
            path = Path(raw)
            if not _native(path):
                continue
            info = path.stat()
            major, minor = (int(v, 16) for v in fields[3].split(":"))
            _require(info.st_ino == int(fields[4]) and os.major(info.st_dev) == major
                     and os.minor(info.st_dev) == minor, "mapped native inode drift")
            paths.add(path.resolve(strict=True))
        return sorted(paths, key=str)
    return sorted({Path(m.path).resolve(strict=True) for m in psutil.Process().memory_maps()
                   if _native(Path(m.path))}, key=str)


def _binding_code_files(binding) -> list[Path]:
    if getattr(sys, "frozen", False):
        executable = _local_path(sys.executable)
        files = {executable}
        roots = {executable.parent}
        extracted = getattr(sys, "_MEIPASS", None)
        if extracted:
            roots.add(_local_path(extracted))
        for root in roots:
            # Bounded, one directory level only. Embedded PYZ is covered by EXE.
            entries = list(itertools.islice(root.iterdir(), 513))
            _require(len(entries) <= 512, "unbounded frozen code directory")
            files.update(p for p in entries if p.is_file() and p.suffix.lower() == ".pyz")
        return sorted(files, key=str)
    root = Path(binding.__file__).resolve().parent
    entries = list(itertools.islice(root.iterdir(), 257))
    _require(len(entries) <= 256, "unbounded binding package directory")
    files = [p for p in entries if p.is_file() and p.suffix in {".py", ".pyc", ".pyd"}]
    _require(files, "binding code container unavailable")
    return sorted(files, key=str)


def _binding_runtime() -> tuple[Any, dict]:
    import llama_cpp
    import llama_cpp.llama_cpp as low

    library = _local_path(low._lib._name).resolve(strict=True)
    mapped = _mapped_libraries()
    _require(library in mapped, "binding library not actually loaded")
    related = []
    dependency = re.compile(r"^(?:lib)?(?:llama|ggml|gomp|iomp|omp|vcomp|cuda|cublas|nvrtc|nvcuda|stdc\+\+|gcc_s)", re.I)
    for path in mapped:
        if path.parent == library.parent or dependency.match(path.name):
            if re.match(r"^(?:lib)?(?:llama|ggml)", path.name, re.I):
                _require(path.parent == library.parent, "foreign llama/ggml runtime loaded")
            related.append(path)
    sources = _binding_code_files(llama_cpp)
    _require(0 < len(sources) <= 256 and len(related) <= 256, "incomplete/unbounded binding runtime")
    raw = low.llama_print_system_info()
    info = raw.decode("utf-8", "strict") if isinstance(raw, bytes) else raw
    _require(type(info) is str and bool(info.strip()), "binding ISA unavailable")
    devices = _binding_devices(related)
    version = getattr(llama_cpp, "__version__", None)
    _require(type(version) is str and bool(version), "binding version unavailable")
    return llama_cpp, {"version": version, "system_info": info.strip(), "devices": devices,
                       "files": [_file_identity(p) for p in sorted(set(related + sources), key=str)]}


def _binding_devices(libraries: list[Path]) -> list[dict]:
    handles = [ctypes.CDLL(str(p)) for p in libraries if re.match(r"^(?:lib)?(?:llama|ggml)", p.name, re.I)]

    def symbol(name, args, result):
        for handle in handles:
            try:
                fn = getattr(handle, name)
                fn.argtypes, fn.restype = args, result
                return fn
            except AttributeError:
                continue
        raise ValueError(f"device API unavailable: {name}")

    count = symbol("ggml_backend_dev_count", [], ctypes.c_size_t)()
    _require(0 < count <= 64, "invalid device count")
    get = symbol("ggml_backend_dev_get", [ctypes.c_size_t], ctypes.c_void_p)
    name = symbol("ggml_backend_dev_name", [ctypes.c_void_p], ctypes.c_char_p)
    desc = symbol("ggml_backend_dev_description", [ctypes.c_void_p], ctypes.c_char_p)
    kind = symbol("ggml_backend_dev_type", [ctypes.c_void_p], ctypes.c_int)
    reg = symbol("ggml_backend_dev_backend_reg", [ctypes.c_void_p], ctypes.c_void_p)
    reg_name = symbol("ggml_backend_reg_name", [ctypes.c_void_p], ctypes.c_char_p)
    devices = []
    for index in range(count):
        pointer = get(index)
        _require(pointer, "null device")
        device = {"name": (name(pointer) or b"").decode("utf-8", "strict"),
                  "description": (desc(pointer) or b"").decode("utf-8", "strict"), "type": int(kind(pointer)),
                  "backend": (reg_name(reg(pointer)) or b"").decode("utf-8", "strict")}
        device["backend"] = device["backend"].upper()
        _require(all(device[k] for k in ("name", "description", "backend")), "incomplete device identity")
        _require(device["backend"].upper() in {"CPU", "CUDA"}, "unsupported/non-local backend")
        devices.append(device)
    return devices


def _run_probe(argv: list[str]) -> str:
    result = subprocess.run(argv, capture_output=True, timeout=20, encoding="utf-8", errors="strict",
                            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
    _require(result.returncode == 0, "runtime identity probe failed")
    text = result.stdout + "\n" + result.stderr
    _require(len(text) <= 1024 * 1024, "runtime probe output too large")
    return text.strip()


def _cuda_inventory() -> list[dict]:
    text = _run_probe(["nvidia-smi", "--query-gpu=index,uuid,pci.bus_id,name,driver_version", "--format=csv,noheader,nounits"])
    rows = []
    for line in text.splitlines():
        values = [s.strip() for s in line.split(",")]
        _require(len(values) == 5 and all(values) and values[1].startswith("GPU-"), "unreliable CUDA identity")
        rows.append(dict(zip(("index", "uuid", "pci_bus_id", "name", "driver"), values)))
    _require(rows, "CUDA identity unavailable")
    return rows


def _execution(runtime: dict, params: dict) -> dict:
    devices = runtime["devices"]
    _require(all(d["backend"].upper() in {"CPU", "CUDA"} for d in devices), "unsupported execution backend")
    cuda = [d for d in devices if d["backend"].upper() == "CUDA"]
    # Hardware presence alone never proves CUDA execution. The selected runtime
    # must expose CUDA devices, and this load must request GPU layers.
    use_cuda = bool(cuda) and params["n_gpu_layers"] != 0
    return {"device": "cuda" if use_cuda else "cpu", "devices": devices if use_cuda else [d for d in devices if d["backend"].upper() == "CPU"],
            "gpu_identity": _cuda_inventory() if use_cuda else [],
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"), "environment": _environment()}


def _validate_load(params: dict, *, binding: bool) -> None:
    for name in ("n_ctx", "n_batch", "n_ubatch"):
        _require(_integer(params.get(name)), f"unknown effective {name}")
    for name in ("n_threads", "n_threads_batch", "n_seq_max"):
        value = params.get(name)
        _require(_integer(value) or (not binding and value is None), f"invalid {name}")
    _require(_integer(params.get("n_gpu_layers"), minimum=-1), "invalid n_gpu_layers")
    fa = params.get("flash_attn")
    _require(type(fa) is bool or (not binding and fa is None), "invalid flash_attn baseline")
    if binding:
        _require(params["n_ubatch"] <= params["n_batch"] <= params["n_ctx"], "invalid effective batches")


def _binding_parameters(binding: Any, supplied: Mapping) -> dict:
    _require(isinstance(supplied, Mapping), "load_kwargs must be a mapping")
    given = dict(supplied)
    given.pop("model_path", None)
    _require(not (given.keys() - _BINDING_KEYS - {"n_seq_max"}), "unsupported binding load parameter")
    params = {}
    for name, parameter in inspect.signature(binding.Llama.__init__).parameters.items():
        if name in {"self", "model_path"} or parameter.kind in {parameter.VAR_POSITIONAL, parameter.VAR_KEYWORD}:
            continue
        _require(name in _BINDING_KEYS, f"unknown binding default: {name}")
        value = given.get(name, parameter.default)
        _require(value is not inspect.Parameter.empty, f"missing binding parameter: {name}")
        params[name] = _json_value(value)
    _require(not (given.keys() - params.keys() - {"n_seq_max"}), "parameter unsupported by binding")
    for name in ("n_ctx", "n_batch", "n_ubatch"):
        _require(_integer(params.get(name)), f"explicit positive {name} required")
    # 0.3.35 semantics are verified against the installed production binding.
    # inspect.signature survives PYZ compilation; source text often does not.
    known_semantics = getattr(binding, "__version__", None) == "0.3.35"
    try:
        source = inspect.getsource(binding.Llama.__init__) if not known_semantics else ""
    except (OSError, TypeError):
        source = ""
    _require(known_semantics or ("min(n_ctx, n_batch)" in source
             and "min(self.n_batch, n_ubatch)" in source), "unknown binding batch semantics")
    if params.get("n_threads") is None:
        _require(known_semantics or "max(multiprocessing.cpu_count() // 2, 1)" in source, "unknown default n_threads")
        params["n_threads"] = max(multiprocessing.cpu_count() // 2, 1)
    if params.get("n_threads_batch") is None:
        _require(known_semantics or "n_threads_batch or multiprocessing.cpu_count()" in source, "unknown default batch threads")
        params["n_threads_batch"] = multiprocessing.cpu_count()
    params["n_batch"] = min(params["n_ctx"], params["n_batch"])
    params["n_ubatch"] = min(params["n_batch"], params["n_ubatch"])
    import llama_cpp.llama_cpp as low

    params["n_seq_max"] = given.get("n_seq_max", int(low.llama_context_default_params().n_seq_max))
    _validate_load(params, binding=True)
    for name in ("draft_model", "chat_handler", "tokenizer", "lora_base", "lora_path"):
        _require(params.get(name) is None, f"unsupported {name}")
    _require(params.get("embedding") is False and params.get("vocab_only") is False, "embedding/vocab-only unsupported")
    return params


def _build(path: str, model_path: Any, runtime: dict, params: dict) -> dict:
    return {"schema_version": SCHEMA_VERSION, "path": path, "model": _file_identity(model_path),
            "runtime": runtime, "host": _host_identity(), "execution": _execution(runtime, params), "load_kwargs": params}


def build_binding_context(model_path, *, load_kwargs, profile=None) -> dict:
    """Fingerprint local whole-model binding; unknown facts raise ValueError.

    n_seq_max describes the value actually applied by the production wrapper.
    Other defaults follow this binding's constructor. RPC, embedding, draft,
    adapters, custom tokenizer and callable/pointer flags are unsupported.
    """
    try:
        if "model_path" in load_kwargs:
            _require(_local_path(load_kwargs["model_path"]) == _local_path(model_path), "conflicting model_path")
        binding, runtime = _binding_runtime()
        return _build(BINDING, model_path, runtime, _binding_parameters(binding, load_kwargs))
    except ValueError:
        raise
    except Exception as exc:
        raise ValueError(f"binding context unavailable: {exc}") from exc


def _help_default(help_text: str, flag: str) -> str:
    lines = help_text.splitlines()
    for index, line in enumerate(lines):
        if re.search(re.escape(flag) + r"(?=[\s,]|$)", line) and line.lstrip().startswith("-"):
            block = [line]
            for following in lines[index + 1:]:
                if following.lstrip().startswith("-"):
                    break
                block.append(following)
            match = re.search(r"default:\s*([^\)]+)", " ".join(block))
            _require(match, f"unresolved server default {flag}")
            return match.group(1).strip().strip("'\"")
    raise ValueError(f"missing server flag {flag}")


def _server_runtime(exe: Any) -> tuple[dict, str]:
    executable = _local_path(exe)
    files = [executable] + sorted((p for p in executable.parent.iterdir() if _native(p)), key=str)
    _require(len(files) <= 256, "unbounded server runtime directory")
    before = [_file_identity(p) for p in files]
    version = _run_probe([str(executable), "--version"])
    help_text = _run_probe([str(executable), "--help"])
    listed = _run_probe([str(executable), "--list-devices"])
    devices = []
    for line in listed.splitlines():
        match = re.match(r"\s*(CUDA\d+):\s*(.*?)\s*\(.*\)\s*$", line)
        if match:
            devices.append({"name": match[1], "description": match[2], "backend": "CUDA", "type": 1})
        elif re.match(r"\s*\S+\d+:\s*", line):
            raise ValueError("unsupported/non-local server device")
    _require("Available devices:" in listed and version, "server device/build identity unavailable")
    devices.insert(0, {"name": "CPU", "description": "host CPU", "backend": "CPU", "type": 0})
    _require(before == [_file_identity(p) for p in files], "server runtime changed during probing")
    return {"version": version, "system_info": None, "devices": devices, "files": before}, help_text


def _server_parameters(supplied: Mapping, help_text: str) -> dict:
    _require(isinstance(supplied, Mapping), "load_kwargs must be a mapping")
    params = _json_value(dict(supplied))
    params.pop("model_path", None)
    _require(not (params.keys() - _SERVER_KEYS), "unsupported server load parameter")
    for name in ("n_ctx", "n_gpu_layers", "cpu_moe", "spec_type"):
        _require(name in params, f"server requires explicit {name}")
    for name, default in (("n_threads", None), ("flash_attn", None), ("n_seq_max", None),
                          ("chat_template_kwargs", None), ("mmproj_path", ""), ("argv_prefix", [])):
        params.setdefault(name, default)
    _require(params["mmproj_path"] == "" and params["argv_prefix"] == [], "prefix/mmproj unsupported")
    _require(params["chat_template_kwargs"] is None or type(params["chat_template_kwargs"]) is dict, "invalid chat_template_kwargs")
    for name, flag in (("n_batch", "--batch-size"), ("n_ubatch", "--ubatch-size")):
        if name not in params:
            params[name] = int(_help_default(help_text, flag))
    if "n_threads_batch" not in params:
        default = _help_default(help_text, "--threads-batch")
        params["n_threads_batch"] = params["n_threads"] if default == "same as --threads" else int(default)
    for key, flag in (("cache_type_k", "--cache-type-k"), ("cache_type_v", "--cache-type-v")):
        if key not in params:
            params[key] = _help_default(help_text, flag)
    _require(type(params["cpu_moe"]) is bool, "invalid cpu_moe")
    _require(params["spec_type"] in {"", "none", "ngram-simple", "ngram-map-k", "ngram-map-k4v", "ngram-mod", "ngram-cache"}, "external draft/speculation unsupported")
    _validate_load(params, binding=False)
    return params


def build_server_context(model_path, *, exe, load_kwargs, profile=None) -> dict:
    """Fingerprint this EXE/runtime, independently of Python binding ISA.

    None threads/flash/sequence-count preserve upstream auto policy. Caller must
    use exactly these flags (or omissions); arbitrary argv/prefix/mmproj and
    external draft models are rejected. Missing batch/cache defaults are read
    from THIS executable. CUDA requires its CUDA backend plus GPU-layer request.
    """
    try:
        if "model_path" in load_kwargs:
            _require(_local_path(load_kwargs["model_path"]) == _local_path(model_path), "conflicting model_path")
        runtime, help_text = _server_runtime(exe)
        return _build(SERVER, model_path, runtime, _server_parameters(load_kwargs, help_text))
    except ValueError:
        raise
    except Exception as exc:
        raise ValueError(f"server context unavailable: {exc}") from exc


def _validate_context(context: Any) -> None:
    _require(type(context) is dict and context.keys() == _CONTEXT_FIELDS, "invalid context fields")
    _require(type(context["schema_version"]) is int and context["schema_version"] == SCHEMA_VERSION, "unsupported context version")
    _require(context["path"] in {BINDING, SERVER}, "unsupported calibration path")
    params = context["load_kwargs"]
    _require(type(params) is dict, "invalid context load parameters")
    allowed = _BINDING_KEYS | {"n_seq_max"} if context["path"] == BINDING else _SERVER_KEYS
    _require(not (params.keys() - allowed), "unsupported context load parameters")
    _json_value(context)
    _validate_load(params, binding=context["path"] == BINDING)
    runtime = context["runtime"]
    _require(type(runtime) is dict and runtime.keys() == {"version", "system_info", "devices", "files"}, "invalid runtime fields")
    _require(type(runtime["version"]) is str and bool(runtime["version"]), "invalid runtime version")
    _require(type(runtime["files"]) is list and 0 < len(runtime["files"]) <= 512, "invalid runtime files")
    for file in [context["model"], *runtime["files"]]:
        _require(type(file) is dict and file.keys() == _FILE_FIELDS, "invalid file identity fields")
        _require(type(file["sha256"]) is str and _SHA.fullmatch(file["sha256"]), "invalid file SHA256")
        _require(type(file["stat"]) is dict and file["stat"].keys() == _STAT_FIELDS, "invalid stat identity fields")
        _require(all(type(v) is int and v >= 0 for v in file["stat"].values()), "invalid stat identity")
        _require(all(type(file[k]) is str and bool(file[k]) for k in ("path", "resolved_path")), "invalid identity path")
    host = context["host"]
    _require(type(host) is dict and host.keys() == {"machine_id", "hostname", "os", "release", "version", "architecture", "cpu", "isa", "logical_cores", "physical_cores", "affinity"}, "invalid host fields")
    _require(all(host[k] for k in ("machine_id", "hostname", "architecture", "cpu", "isa", "affinity")), "missing host identity")
    execution = context["execution"]
    _require(type(execution) is dict and execution.keys() == {"device", "devices", "gpu_identity", "cuda_visible_devices", "environment"}, "invalid execution fields")
    _require(execution["device"] in {"cpu", "cuda"}, "unsupported execution device")
    for devices in (runtime["devices"], execution["devices"]):
        _require(type(devices) is list and bool(devices), "missing devices")
        for device in devices:
            _require(type(device) is dict and device.keys() == {"name", "description", "backend", "type"}, "invalid device fields")
            _require(device["backend"] in {"CPU", "CUDA"} and device["name"] and device["description"], "unsupported/non-local device")
    _require(type(execution["gpu_identity"]) is list and type(execution["environment"]) is dict, "invalid device identity")
    for gpu in execution["gpu_identity"]:
        _require(type(gpu) is dict and gpu.keys() == {"index", "uuid", "pci_bus_id", "name", "driver"} and all(gpu.values()), "invalid GPU fields")
    if execution["device"] == "cuda":
        _require(execution["gpu_identity"] and params["n_gpu_layers"] != 0
                 and any(d["backend"] == "CUDA" for d in runtime["devices"]), "CUDA execution not established")


def _check_live(context: dict) -> None:
    for file in [context["model"], *context["runtime"]["files"]]:
        _require(_file_identity(file["path"]) == file, "file identity drift")
    _require(_host_identity() == context["host"], "host identity drift")
    _require(_environment() == context["execution"]["environment"], "execution environment drift")
    _require(os.environ.get("CUDA_VISIBLE_DEVICES") == context["execution"]["cuda_visible_devices"], "CUDA visibility drift")
    if context["execution"]["device"] == "cuda":
        _require(_cuda_inventory() == context["execution"]["gpu_identity"], "CUDA device/driver drift")
    if context["path"] == BINDING:
        _, current = _binding_runtime()
    else:
        current, _ = _server_runtime(context["runtime"]["files"][0]["path"])
    _require(current == context["runtime"], "loaded runtime drift")


def _settings(value: Any, *, baseline: bool = False, server: bool = False) -> None:
    _require(type(value) is dict and bool(value) and not (value.keys() - {"flash_attn", "n_threads"}), "invalid settings fields")
    if baseline:
        _require(value.keys() == {"flash_attn", "n_threads"}, "incomplete baseline")
    if "flash_attn" in value:
        _require(type(value["flash_attn"]) is bool or (baseline and server and value["flash_attn"] is None), "invalid flash_attn")
    if "n_threads" in value:
        _require(_integer(value["n_threads"]) or (baseline and server and value["n_threads"] is None), "invalid n_threads")


def evaluate_record(record: dict) -> dict:
    """Recompute evidence verdict. Does NOT prove current machine identity.

    Only select_calibration additionally checks that context is still live.
    This helper is also the CLI's evidence gate; accepted:true is never trusted.
    """
    try:
        _require(type(record) is dict and record.keys() == _RECORD_FIELDS, "invalid record fields")
        _require(type(record["record_id"]) is str and 0 < len(record["record_id"]) <= 128, "invalid record_id")
        _require(type(record["accepted"]) is bool, "invalid accepted field")
        context = record["context"]
        _validate_context(context)
        baseline, candidate = record["baseline"], record["candidate"]
        _settings(baseline, baseline=True, server=context["path"] == SERVER)
        _settings(candidate)
        _require(baseline == {k: context["load_kwargs"][k] for k in ("flash_attn", "n_threads")}, "baseline/context mismatch")
        _require(any(v != baseline[k] for k, v in candidate.items()), "candidate equals baseline")
        protocol = record["protocol"]
        _require(type(protocol) is dict and protocol.keys() == _PROTOCOL_FIELDS, "invalid protocol fields")
        for name, minimum in (("warmup", 1), ("samples", 3), ("gen_tokens", 32)):
            _require(_integer(protocol[name], minimum=minimum), f"insufficient {name}")
        _require(protocol["warmup"] + protocol["samples"] <= 1024 and protocol["gen_tokens"] <= 65536, "unbounded evidence")
        _require(protocol["greedy"] is True and protocol["free_generation"] is True, "not greedy free generation")
        prompt = protocol["prompt_tokens"]
        _require(type(prompt) is list and 0 < len(prompt) <= context["load_kwargs"]["n_ctx"]
                 and all(_integer(t, minimum=0) for t in prompt), "invalid prompt tokens")
        sequences = context["load_kwargs"]["n_seq_max"] or 1
        _require(len(prompt) + protocol["gen_tokens"] <= context["load_kwargs"]["n_ctx"] // sequences, "generation exceeds context")
        _require(type(protocol["prompt_sha256"]) is str and _SHA.fullmatch(protocol["prompt_sha256"]), "invalid prompt SHA256")
        reference = None
        measured = []
        for key in ("baseline_runs", "candidate_runs"):
            runs = record[key]
            _require(type(runs) is list and len(runs) == protocol["warmup"] + protocol["samples"], "run count mismatch")
            rates = {"prefill": [], "decode": []}
            for index, run in enumerate(runs):
                _require(type(run) is dict and run.keys() == _RUN_FIELDS, "invalid run fields")
                _require(type(run["warmup"]) is bool and run["warmup"] == (index < protocol["warmup"]), "warmup ordering mismatch")
                _require(run["finite_logits"] is True, "nonfinite/unverified logits")
                tokens = run["tokens"]
                _require(type(tokens) is list and 32 <= len(tokens) <= protocol["gen_tokens"]
                         and all(_integer(t, minimum=0) for t in tokens), "short/incomplete generation")
                _require(run["stop"] in {"eos", "length"} and (run["stop"] == "eos" or len(tokens) == protocol["gen_tokens"]), "incomplete token sequence")
                if reference is None:
                    reference = (tokens, run["stop"])
                _require((tokens, run["stop"]) == reference, "token sequence mismatch")
                for phase in ("prefill", "decode"):
                    rate = run[f"{phase}_tps"]
                    _require(type(rate) in (int, float) and math.isfinite(rate) and rate > 0, "invalid phase sampling")
                    if not run["warmup"]:
                        rates[phase].append(rate)
            measured.append({k: statistics.median(v) for k, v in rates.items()})
        ratios = {k: measured[1][k] / measured[0][k] for k in ("prefill", "decode")}
        _require(all(math.isfinite(v) and v > 0 for v in ratios.values()), "invalid performance ratio")
        if ratios["decode"] < 1.03 or ratios["prefill"] < 0.98:
            return {"settings": {}, "reason": "performance_gate_failed", "ratios": ratios}
        return {"settings": dict(candidate), "reason": "accepted", "ratios": ratios}
    except (ValueError, TypeError, KeyError, OverflowError, RecursionError) as exc:
        return {"settings": {}, "reason": f"invalid_evidence: {exc}", "ratios": {}}


def calibration_path(path=None) -> Path:
    """Explicit path > QLH_INFER_CALIBRATION_PATH > config.STATE_DIR."""
    if path is None:
        path = os.environ.get("QLH_INFER_CALIBRATION_PATH", "").strip()
        if not path:
            import config

            path = Path(config.STATE_DIR) / "inference-calibration.json"
    return _local_path(path)


def _unique_object(pairs: list) -> dict:
    value = {}
    for key, item in pairs:
        _require(key not in value, "duplicate JSON field")
        value[key] = item
    return value


def read_calibration_store(path) -> dict:
    """Strict bounded JSON reader, shared with explicit CLI publication."""
    with Path(path).open("rb") as handle:
        raw = handle.read(MAX_STORE_BYTES + 1)
    _require(len(raw) <= MAX_STORE_BYTES, "oversized calibration store")
    store = json.loads(raw, object_pairs_hook=_unique_object,
                       parse_constant=lambda _: (_ for _ in ()).throw(ValueError("nonfinite JSON")))
    _require(type(store) is dict and store.keys() == {"schema_version", "records"}, "invalid store fields")
    _require(type(store["schema_version"]) is int and store["schema_version"] == SCHEMA_VERSION, "unsupported store version")
    _require(type(store["records"]) is list and len(store["records"]) <= MAX_RECORDS, "invalid records list")
    return store


def select_calibration(context, *, path=None) -> dict:
    """Return settings/reason/record_id/source; failures always settings={}.

    Same-context records must all revalidate and agree. Conflicting positive/
    negative evidence or candidate settings fails closed, with no first match.
    """
    result = {"settings": {}, "reason": "missing", "record_id": None, "source": None}
    try:
        source = calibration_path(path)
        result["source"] = str(source)
        _validate_context(context)
        if not source.exists():
            return result
        _check_live(context)
        store = read_calibration_store(source)
        matches, ids = [], set()
        for record in store["records"]:
            verdict = evaluate_record(record)
            if verdict["reason"].startswith("invalid_evidence"):
                result["reason"] = verdict["reason"]
                return result
            _require(record["record_id"] not in ids, "duplicate record_id")
            ids.add(record["record_id"])
            if record["context"] == context:
                matches.append((record, verdict))
        if not matches:
            result["reason"] = "context_mismatch"
            return result
        effective = [{**r["baseline"], **r["candidate"]} for r, _ in matches]
        decisions = [bool(v["settings"]) for _, v in matches]
        if any(v != effective[0] for v in effective) or any(v != decisions[0] for v in decisions):
            result["reason"] = "conflicting_records"
            return result
        record, verdict = min(matches, key=lambda pair: pair[0]["record_id"])
        result.update(settings=verdict["settings"], reason=verdict["reason"], record_id=record["record_id"])
        return result
    except FileNotFoundError:
        result["reason"] = "missing_or_drifted_file"
    except Exception as exc:
        result["reason"] = f"invalid_or_drifted: {type(exc).__name__}: {exc}"
    return result


__all__ = ["build_binding_context", "build_server_context", "select_calibration",
           "evaluate_record", "calibration_path", "read_calibration_store"]
