"""Acquire registry-owned master sources without initializing an inference engine.

``progress`` receives dictionaries with ``phase`` and optional byte/job counters.
Cancellation is cooperative at stage and hash boundaries; the existing snapshot
and conversion APIs cannot interrupt an in-flight provider call/subprocess.
Conversion uses the existing Q4_K_M policy. Selecting a model authorizes the
local download and ``confirmation="CONVERT"`` transaction.
"""

from __future__ import annotations

import fnmatch
import hashlib
import importlib
import json
import os
import shutil
import sys
import tempfile
from copy import deepcopy
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any, Callable

import local_store
import model_config as mc
import model_download_jobs as jobs
import model_registry_validation as mrv
import pipeline_model_descriptor as descriptor
import proxy_config
from model_config import ModelConfig

PROVENANCE = "cross_format.provenance.json"
_MAX_JSON_BYTES = 32 * 1024 * 1024
_DISK_RESERVE = 512 * 1024 * 1024
_OUTTYPE = "Q4_K_M"


class ModelSupplySourceError(ValueError):
    """A path-free failure suitable for the coordinator's reason_code/message."""

    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code
        self.message = message


def _fail(code: str, message: str) -> None:
    raise ModelSupplySourceError(code, message)


def _check(cancelled: Callable[[], bool]) -> None:
    if cancelled():
        _fail("SUPPLY_CANCELLED", "Model source preparation was cancelled")


def _notify(progress: Callable | None, phase: str, **values: Any) -> None:
    if progress is not None:
        progress({"phase": phase, **values})


def _linked(path: Path) -> bool:
    return path.is_symlink() or bool(getattr(path, "is_junction", lambda: False)())


def _json(path: Path) -> dict:
    if _linked(path) or not path.is_file() or not 0 < path.stat().st_size <= _MAX_JSON_BYTES:
        raise ValueError("missing, unsafe or oversized metadata")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("metadata must be a JSON object")
    return payload


def _sha(path: Path, cancelled: Callable[[], bool]) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            _check(cancelled)
            digest.update(chunk)
    _check(cancelled)
    return digest.hexdigest()


def _content_sha(path: Path, cancelled: Callable[[], bool]) -> str:
    """Share verified immutable source identities with builders and preparation."""
    from model_supply_assets import _hash_file_cached
    def check_cancelled():
        _check(cancelled)
        return False
    _check(cancelled)
    return _hash_file_cached(path, check_cancelled)


def _not_cancelled() -> bool:
    return False


def _content_digest(path: Path) -> str:
    """Digest callback for injected-hash APIs that have no cancellation hook."""
    return _content_sha(path, _not_cancelled)


def _canonical(payload: dict) -> str:
    return hashlib.sha256(json.dumps(payload, ensure_ascii=True, sort_keys=True,
                                     separators=(",", ":")).encode("utf-8")).hexdigest()


def _publish_json(path: Path, payload: dict) -> None:
    """Atomically publish a new record, preserving any existing record."""
    with tempfile.NamedTemporaryFile(dir=path.parent, prefix=".qlh-source-",
                                     suffix=".part", delete=False) as handle:
        temporary = Path(handle.name)
        try:
            handle.write((json.dumps(payload, ensure_ascii=True, sort_keys=True,
                                     indent=2) + "\n").encode("utf-8"))
            handle.flush()
            os.fsync(handle.fileno())
        except BaseException:
            handle.close()
            temporary.unlink(missing_ok=True)
            raise
    try:
        os.link(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _registered(model_id: str) -> tuple[ModelConfig, dict]:
    try:
        snapshot = local_store.get_local_model_registry_snapshot(model_id)
    except (TypeError, ValueError):
        _fail("SUPPLY_REGISTRY_INVALID", "Registered model configuration is invalid")
    row = local_store._RegistryPayload(snapshot.as_payload() or {}, snapshot.registry_revision)
    cfg = mc.get_model_config(model_id, [row] if row else [])
    if cfg is None:
        code = "SUPPLY_REGISTRY_INVALID" if row else "SUPPLY_MODEL_NOT_REGISTERED"
        _fail(code, "Registered model configuration is invalid" if row else "Select a registered model")
    return cfg, row


def _trusted_storage_path(path: Path) -> Path:
    """Resolve only the application's configured models storage mapping.

    The models root may be a user-managed junction to another disk. Its
    canonical root is trusted; links inside model contents remain forbidden.
    """
    candidate = Path(os.path.abspath(path.expanduser()))
    models = Path(os.path.abspath(Path(mc._APP_ROOT) / "models"))
    if _linked(models) and candidate.is_relative_to(models):
        return models.resolve() / candidate.relative_to(models)
    return candidate


def _path(value: str) -> Path | None:
    return _trusted_storage_path(Path(mc.resolve_model_path(value))) if value else None


def _existing(path: Path | None, *, directory: bool) -> Path | None:
    if path is None:
        return None
    path = _trusted_storage_path(path)
    if any(_linked(item) for item in (path, *path.parents)):
        _fail("SUPPLY_SOURCE_INVALID", "Registered source cannot traverse a filesystem link")
    if not path.exists():
        return None
    if (directory and not path.is_dir()) or (not directory and not path.is_file()):
        _fail("SUPPLY_SOURCE_INVALID", "Registered source has the wrong filesystem type")
    return path.resolve()


def _hf_metadata(root: Path, cancelled: Callable[[], bool]) -> list[Path]:
    """Validate loader metadata and Safetensors headers, never tensor payloads."""
    _check(cancelled)
    artifact = mrv.validate_model_artifact("safetensors", str(root))
    config = _json(root / "config.json")
    _json(root / "tokenizer_config.json")
    if not isinstance(config.get("model_type"), str) or not config["model_type"]:
        raise ValueError("config has no model_type")
    decoder = config.get("text_config", config)
    if not isinstance(decoder, dict):
        raise ValueError("invalid decoder config")
    for name in ("num_hidden_layers", "hidden_size"):
        value = decoder.get(name)
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError("config has invalid decoder dimensions")
    weights = artifact["safetensors_files"]
    # Validate references before either the converter or a metadata mmap opens
    # them. Provider cache metadata is not part of the source identity.
    for count, item in enumerate(root.rglob("*"), 1):
        if count > 100_000 or _linked(item):
            raise ValueError("unsafe or excessive source entries")
    observed: dict[str, str] = {}
    for weight in weights:
        _check(cancelled)
        if weight.suffix.lower() != ".safetensors":
            continue  # legacy .bin is checked for layout/index, never unpickled
        # safe_open exposes shapes/slices without get_tensor or model loading.
        from safetensors import safe_open
        with safe_open(str(weight), framework="numpy") as handle:
            for key in handle.keys():
                if key in observed:
                    raise ValueError("duplicate tensor key")
                handle.get_slice(key).get_shape()
                observed[key] = weight.relative_to(root).as_posix()
    for name in ("model.safetensors.index.json", "pytorch_model.bin.index.json"):
        index_path = root / name
        if not index_path.exists():
            continue
        mapping = _json(index_path).get("weight_map")
        if not isinstance(mapping, dict) or not mapping:
            raise ValueError("invalid weight index")
        for key, relative in mapping.items():
            if not isinstance(key, str):
                raise ValueError("invalid tensor key")
            relative = jobs._validate_pin_path(relative, field="weight index")
            shard = root / relative
            if not shard.is_file() or shard.suffix.lower() not in {".safetensors", ".bin"}:
                raise ValueError("missing weight shard")
            if shard.suffix.lower() == ".safetensors" and observed.get(key) != relative:
                raise ValueError("index references a missing tensor")
    return weights


def _model_tools(name: str):
    if getattr(sys, "frozen", False):
        # The frozen importer can resolve scripts.model_tools from PYZ before
        # the disk package, importing the developer probes initializer. Bind a
        # private package explicitly to signed resources under the runtime root.
        import _imp
        from importlib import util as import_util

        if name not in {"gguf", "gguf_convert", "llama_quantize_toolchain", "conversion_runtime"}:
            raise ImportError(f"model_supply_module_unavailable: {name}")
        resource_root = getattr(sys, "_MEIPASS", None)
        if resource_root is None:
            raise ImportError("model_supply_resource_missing: frozen runtime root")
        directory = Path(resource_root).resolve() / "scripts" / "model_tools"
        initializer = directory / "__init__.py"
        module_file = directory / f"{name}.py"
        if not initializer.is_file() or not module_file.is_file():
            raise ImportError(f"model_supply_resource_missing: {name}")
        namespace = "_qlh_supply_tools"
        _imp.acquire_lock()
        try:
            package = sys.modules.get(namespace)
            if package is None:
                spec = import_util.spec_from_file_location(
                    namespace, initializer,
                    submodule_search_locations=[str(directory)],
                )
                if spec is None or spec.loader is None:
                    raise ImportError("model_supply_resource_missing: package loader")
                package = import_util.module_from_spec(spec)
                sys.modules[namespace] = package
                try:
                    spec.loader.exec_module(package)
                except BaseException:
                    sys.modules.pop(namespace, None)
                    raise
            elif Path(getattr(package, "__file__", "")).resolve() != initializer:
                raise ImportError("model_supply_package_conflict: frozen resource root changed")
            # PyInstaller maps __path__ back to a PYZ path even for an alias.
            # Load the tiny relative-import closure explicitly, rather than
            # allowing its path finder to reuse archived developer modules.
            closure = ({
                "gguf_convert": ("gguf", "llama_quantize_toolchain", "conversion_runtime", "gguf_convert"),
            }.get(name, (name,)))
            for child in closure:
                fullname = namespace + "." + child
                child_file = directory / f"{child}.py"
                loaded = sys.modules.get(fullname)
                if loaded is not None:
                    if Path(getattr(loaded, "__file__", "")).resolve() != child_file:
                        raise ImportError("model_supply_package_conflict: child resource root changed")
                    continue
                if not child_file.is_file():
                    raise ImportError(f"model_supply_resource_missing: {child}")
                child_spec = import_util.spec_from_file_location(fullname, child_file)
                if child_spec is None or child_spec.loader is None:
                    raise ImportError(f"model_supply_resource_missing: {child} loader")
                loaded = import_util.module_from_spec(child_spec)
                sys.modules[fullname] = loaded
                try:
                    child_spec.loader.exec_module(loaded)
                except BaseException:
                    sys.modules.pop(fullname, None)
                    raise
                setattr(package, child, loaded)
            return sys.modules[namespace + "." + name]
        finally:
            _imp.release_lock()
    # A direct ``python src/api_server.py`` launch has only src on sys.path.
    # Keep the developer package identity so existing tests can monkeypatch it.
    root = Path(__file__).resolve().parent.parent
    if (root / "scripts" / "model_tools").is_dir() and str(root) not in sys.path:
        sys.path.insert(0, str(root))
    return importlib.import_module("scripts.model_tools." + name)

def _gguf_metadata(path: Path) -> dict:
    verify_gguf = _model_tools("gguf").verify_gguf
    from relay_segment_info import read_gguf_tokenizer_sha256
    mrv.validate_model_artifact("gguf", gguf_path=str(path))
    report = verify_gguf(path, full_hash=True, digest=_content_digest)
    if not report.get("valid") or not report.get("structure_valid"):
        raise ValueError("GGUF structure or SHA256 validation failed")
    tokenizer = read_gguf_tokenizer_sha256(path)
    if not tokenizer:
        raise ValueError("GGUF tokenizer metadata is missing")
    return {"artifact_sha256": report["sha256"], "tokenizer_sha256": tokenizer}


def _registered_external_files(cfg: ModelConfig, row: dict) -> dict[str, Path]:
    bindings = row.get("manifest_external_files", {})
    if not isinstance(bindings, dict):
        _fail("SUPPLY_REGISTRY_INVALID", "External manifest bindings must be an object")
    bindings = dict(bindings)
    # Legacy v1 both registrations recorded external/<name> without the map.
    # Recover only the explicitly registered GGUF, never an adjacent candidate.
    registered = _path(cfg.gguf_path)
    if registered is not None:
        bindings.setdefault("external/" + registered.name, str(registered))
    result = {}
    for alias, value in bindings.items():
        if (not isinstance(alias, str) or not alias.startswith("external/")
                or len(Path(alias).parts) != 2 or not isinstance(value, str) or not value):
            _fail("SUPPLY_REGISTRY_INVALID", "External manifest binding is invalid")
        path = _path(value)
        if path is None or path.name != Path(alias).name:
            _fail("SUPPLY_REGISTRY_INVALID", "External manifest binding filename differs")
        path = _existing(path, directory=False)
        if path is not None:
            result[alias] = path
    return result


def _registered_manifest_identity(row: dict) -> dict[str, str]:
    """Recover the pinned source identity without trusting a replacement file."""
    manifest = row.get("manifest")
    if manifest is not None and not isinstance(manifest, dict):
        _fail("SUPPLY_REGISTRY_INVALID", "Registered source manifest must be an object")
    expected = {}
    for field in ("artifact_sha256", "manifest_sha256"):
        digests = []
        for value in (row.get(field), (manifest or {}).get(field)):
            if value is None or value == "":
                continue
            if not isinstance(value, str) or not jobs._SHA256_RE.fullmatch(value.lower()):
                _fail("SUPPLY_REGISTRY_INVALID", "Registered source identity is invalid")
            digests.append(value.lower())
        if len(set(digests)) > 1:
            _fail("SUPPLY_REGISTRY_INVALID", "Registered source identities disagree")
        if digests:
            expected[field] = digests[0]
    return expected


def _live_manifest(root: Path, weights: list[Path], cancelled: Callable[[], bool],
                   *, external_files: dict[str, Path] | None = None,
                   expected_identity: dict[str, str] | None = None,
                   require_hf_metadata: bool = True) -> dict:
    """Rehash declared core identity without redefining an existing source.

    Extra runtime support files (chat_template.jinja, merges/vocab, etc.) are
    bound by the downstream supply manifest, not folded into this old identity.
    """
    manifest = _json(root / "model.manifest.json")
    if any(manifest.get(field) != digest for field, digest in (expected_identity or {}).items()):
        raise ValueError("source manifest differs from its registered identity")
    entries = manifest.get("files")
    if not isinstance(entries, list) or not entries:
        raise ValueError("invalid source manifest files")
    names: set[str] = set()
    lines = []
    for entry in entries:
        if not isinstance(entry, dict):
            raise ValueError("invalid source manifest entry")
        relative = jobs._validate_pin_path(entry.get("path"), field="source manifest")
        if relative in names:
            raise ValueError("duplicate source manifest path")
        names.add(relative)
        if relative.startswith("external/"):
            path = (external_files or {}).get(relative)
            if path is None:
                raise ValueError("external source manifest file is not explicitly registered")
        else:
            path = root / relative
            path.resolve().relative_to(root.resolve())
        if any(_linked(item) for item in (path, *path.parents)) or not path.is_file():
            raise ValueError("source manifest file missing or unsafe")
        size = entry.get("size_bytes")
        digest = entry.get("sha256")
        if (isinstance(size, bool) or not isinstance(size, int) or size < 0
                or not isinstance(digest, str) or not jobs._SHA256_RE.fullmatch(digest)):
            raise ValueError("invalid source manifest size or digest")
        if size != path.stat().st_size or digest != _content_sha(path, cancelled):
            raise ValueError("source manifest content changed")
        lines.append(f"{relative}\0{entry['size_bytes']}\0{entry['sha256']}")
    required = {item.relative_to(root).as_posix() for item in weights}
    if require_hf_metadata:
        required.update({"config.json", "tokenizer_config.json"})
    if not required <= names:
        raise ValueError("source manifest does not bind all weights and metadata")
    unsigned = {key: value for key, value in manifest.items()
                if key not in {"artifact_sha256", "manifest_sha256"}}
    if (manifest.get("schema") != 1
            or manifest.get("artifact_sha256") != hashlib.sha256("\n".join(lines).encode()).hexdigest()
            or manifest.get("manifest_sha256") != _canonical(unsigned)):
        raise ValueError("source manifest identity is invalid")
    return manifest


def _proof(root: Path, weights: list[Path], cancelled: Callable[[], bool],
           *, external_files: dict[str, Path] | None = None,
           expected_identity: dict[str, str] | None = None) -> tuple[dict, Path]:
    try:
        _live_manifest(root, weights, cancelled, external_files=external_files,
                       expected_identity=expected_identity)
        identity = descriptor.cross_format_pipeline_identity(root, digest=_content_digest)
        if not identity:
            raise ValueError("missing cross-format identity")
        proof = _json(root / PROVENANCE)
        # Only resolve the recorded target after the strict schema, confinement,
        # source/target hashes and tokenizer checks have all passed.
        target = _existing(root / proof["target"]["path"], directory=False)
        if target is None:
            raise ValueError("provenance target disappeared")
        return proof, target
    except ModelSupplySourceError:
        raise
    except Exception as exc:
        raise ModelSupplySourceError("SUPPLY_CONVERSION_PROOF_INVALID",
                                    "Existing conversion proof failed source/target verification") from exc


def _disk(path: Path, required: int) -> None:
    path = _trusted_storage_path(path)
    parent = path.absolute()
    while not parent.exists() and parent != parent.parent:
        parent = parent.parent
    if any(_linked(item) for item in (path, *path.parents)):
        _fail("SUPPLY_SOURCE_INVALID", "Source destination cannot traverse a filesystem link")
    if shutil.disk_usage(parent).free < required:
        _fail("SUPPLY_INSUFFICIENT_DISK", "Insufficient disk space for model source staging")


def _download_spec(cfg: ModelConfig, row: dict) -> dict:
    source = cfg.huggingface_id
    try:
        jobs._validate_remote_source(source)
        if Path(source).is_dir():
            raise ValueError("configured provider ID resolves to a local directory")
        preset = next((item for item in jobs.PRESETS
                       if item.get("default_model_id") == cfg.model_id
                       and item.get("hf_repo") == source), None)
        pin = jobs._load_remote_pin(preset) if preset else None
        if preset and preset["id"] in jobs._PINNED_PRESET_IDS and pin is None:
            raise ValueError("reviewed pin missing")
    except (ValueError, jobs.JobError, OSError) as exc:
        raise ModelSupplySourceError("SUPPLY_SOURCE_MISSING",
                                    "No executable pinned or configured download source") from exc
    is_gguf = cfg.model_type == "gguf"
    selector = Path(cfg.gguf_path).name if is_gguf and cfg.gguf_path else ""
    if is_gguf and not selector:
        _fail("SUPPLY_SOURCE_MISSING", "Registered GGUF download has no explicit file selector")
    revision = pin["revision"] if pin else str(row.get("revision") or "")
    # Fleet/profile revision labels are not provider revisions.
    if not jobs._REVISION_RE.fullmatch(revision):
        revision = ""
    patterns = pin["allow_patterns"] if pin else ([selector] if is_gguf else [
        "*.safetensors", "*.bin", "config.json", "tokenizer*.json", "generation_config.json",
        "*.index.json", "special_tokens_map.json", "added_tokens.json", "vocab.json",
        "*.model", "*.txt", "*.tiktoken", "*.jinja",
    ])
    min_disk = (preset or {}).get("resource_gate", {}).get("min_disk_gb", 0)
    return {"source": source, "revision": revision, "allow_patterns": patterns,
            "required_files": pin["required_files"] if pin else None,
            "artifact_files": pin["weight_files"] if pin else None,
            "selector": selector, "disk_bytes": max(_DISK_RESERVE, int(min_disk * 1024**3))}



def _remote_inventory(spec: dict) -> int:
    """Resolve immutable provider metadata before reserving source disk space."""
    from huggingface_hub import HfApi
    # Match the proxy handling already used by jobs._download; restore process
    # settings even if metadata lookup fails. No model runtime is imported.
    proxy = proxy_config.resolve_http_proxy()
    with proxy_config.proxy_environment(proxy) as environment:
        keys = ("HTTP_PROXY", "HTTPS_PROXY", "http_proxy", "https_proxy")
        previous = {key: os.environ.get(key) for key in keys}
        try:
            for key in keys:
                if key in environment:
                    os.environ[key] = environment[key]
                else:
                    os.environ.pop(key, None)
            info = HfApi().model_info(spec["source"], revision=spec["revision"] or None,
                                      files_metadata=True, timeout=30)
        finally:
            for key, value in previous.items():
                if value is None:
                    os.environ.pop(key, None)
                else:
                    os.environ[key] = value
    revision = str(info.sha or "")
    if not jobs._REVISION_RE.fullmatch(revision) or (spec["revision"] and spec["revision"] != revision):
        _fail("SUPPLY_DOWNLOAD_METADATA_INVALID", "Provider metadata has no matching immutable revision")
    siblings = info.siblings
    if not isinstance(siblings, list) or len(siblings) > 100_000:
        _fail("SUPPLY_DOWNLOAD_METADATA_INVALID", "Provider file inventory is unavailable or excessive")
    selected: dict[str, int] = {}
    for item in siblings:
        relative = jobs._validate_pin_path(item.rfilename, field="provider file")
        if not any(fnmatch.fnmatchcase(relative, pattern) for pattern in spec["allow_patterns"]):
            continue
        size = item.size
        if relative in selected or isinstance(size, bool) or not isinstance(size, int) or size < 0:
            _fail("SUPPLY_DOWNLOAD_METADATA_INVALID", "Provider file sizes are missing or invalid")
        selected[relative] = size
    required = set(spec["required_files"] or [])
    required.update({spec["selector"]} if spec["selector"] else {"config.json", "tokenizer_config.json"})
    if not required <= selected.keys() or not any(name.endswith(jobs._WEIGHT_SUFFIXES) for name in selected):
        _fail("SUPPLY_SOURCE_MISSING", "Configured repository has no executable source file set")
    spec["revision"] = revision
    # Exact names freeze this lookup's file selection as well as its revision.
    spec["allow_patterns"] = sorted(selected)
    spec["remote_files"] = selected
    return sum(selected.values())


def _save(cfg: ModelConfig, row: dict, *, registry_snapshot: dict | None,
          **evidence: Any) -> None:
    if not local_store.save_local_experimental_model_cas(
            cfg.model_id, {**row, **asdict(cfg), **evidence}, expected=registry_snapshot):
        _fail("SUPPLY_REGISTRY_CHANGED", "Model registration changed during source preparation")


def _download(cfg: ModelConfig, row: dict, operation: Path,
              cancelled: Callable[[], bool], progress: Callable | None) -> ModelConfig:
    registry_snapshot = deepcopy(row)
    expected_identity = _registered_manifest_identity(registry_snapshot or {})
    spec = _download_spec(cfg, row)
    target = _path(cfg.model_path) if cfg.model_type != "gguf" else None
    if target is None:
        # Generated storage names cannot be controlled by a request model_id.
        key = hashlib.sha256(cfg.model_id.encode()).hexdigest()[:20]
        target = _trusted_storage_path(Path(mc._APP_ROOT) / "models" / ("model-supply-" + key))
    _disk(target.parent, spec["disk_bytes"])
    _check(cancelled)
    # Use the existing persisted queue and state vocabulary without invoking
    # create_job(preset_id=...), which applies inference RAM/VRAM admission.
    claimed = False

    def claim(_runner: Callable) -> None:
        nonlocal claimed
        claimed = True  # executor is invoked only for a newly created job

    job = jobs.create_job(source=spec["source"], target=str(target),
                          model_id=cfg.model_id, models_root=str(target.parent), executor=claim)
    job_id = job["job_id"]
    if not claimed or job["status"] != jobs.STATUS_QUEUED:
        _fail("SUPPLY_DOWNLOAD_BUSY", "Registered source already has an active download")
    staging = None
    published = False
    registered = False
    registry_changed = False
    try:
        _notify(progress, "resolving", job_id=job_id)
        _check(cancelled)
        source_bytes = _remote_inventory(spec)
        _disk(target.parent, max(spec["disk_bytes"], source_bytes + source_bytes // 10 + _DISK_RESERVE))
        _check(cancelled)
        jobs._update_job(job_id, status=jobs.STATUS_DOWNLOADING, total_bytes=source_bytes)
        _notify(progress, "downloading", job_id=job_id, total_bytes=source_bytes)
        if target.exists() or _linked(target):
            _fail("SUPPLY_SOURCE_INVALID", "Source download destination already exists")
        target.parent.mkdir(parents=True, exist_ok=True)
        staging = Path(tempfile.mkdtemp(prefix=".qlh-supply-download-", dir=target.parent))

        def notify(path: Path, current: int, total: int) -> None:
            _check(cancelled)
            _disk(staging, _DISK_RESERVE)
            jobs._update_job(job_id, downloaded_bytes=current, total_bytes=total,
                             progress=current / total if total else 0)
            _notify(progress, "downloading", job_id=job_id,
                    downloaded_bytes=current, total_bytes=total)

        files = jobs._download(spec["source"], staging, use_modelscope=False, proxy="",
                               progress_cb=notify, revision=spec["revision"],
                               allow_patterns=spec["allow_patterns"])
        _check(cancelled)
        jobs._update_job(job_id, status=jobs.STATUS_VERIFYING)
        _notify(progress, "verifying", job_id=job_id)
        jobs._verify_required_files(staging, spec["required_files"])
        for item in staging.rglob("*"):
            if _linked(item):
                _fail("SUPPLY_DOWNLOAD_METADATA_INVALID", "Downloaded source contains filesystem links")
        for path in files:
            path.resolve().relative_to(staging.resolve())
        for relative, size in spec["remote_files"].items():
            item = staging / relative
            if _linked(item) or not item.is_file() or item.stat().st_size != size:
                _fail("SUPPLY_DOWNLOAD_METADATA_INVALID", "Downloaded file differs from provider metadata")
        summary = jobs._verify(files, None)
        if spec["artifact_files"]:
            jobs._verify_pinned_files(staging, files, spec["artifact_files"])
        selector = spec["selector"]
        if selector:
            selected = staging / selector
            if files != [selected]:
                _fail("SUPPLY_DOWNLOAD_FILE_SET_INVALID", "Downloaded GGUF differs from its registered selector")
            _gguf_metadata(selected)
            artifact = mrv.validate_model_artifact("gguf", gguf_path=str(selected))
        else:
            try:
                weights = _hf_metadata(staging, cancelled)
            except ModelSupplySourceError:
                raise
            except Exception as exc:
                raise ModelSupplySourceError("SUPPLY_DOWNLOAD_METADATA_INVALID",
                                            "Downloaded HF source failed metadata validation") from exc
            if set(files) != set(weights):
                _fail("SUPPLY_DOWNLOAD_FILE_SET_INVALID", "Download contains unrequested source formats")
            artifact = mrv.validate_model_artifact("safetensors", str(staging))
        manifest = mrv.build_manifest(staging, artifact["files"],
                                      model_type=artifact["model_type"],
                                      revision=spec["revision"], source=spec["source"])
        if any(manifest.get(field) != digest for field, digest in expected_identity.items()):
            _fail("SUPPLY_DOWNLOAD_METADATA_INVALID", "Downloaded source differs from its registered identity")
        mrv.write_manifest(staging, manifest)
        _check(cancelled)
        _disk(staging, _DISK_RESERVE)
        jobs._update_job(job_id, status=jobs.STATUS_REGISTERING, progress=1,
                         total_bytes=summary["total_bytes"], downloaded_bytes=summary["total_bytes"])
        _notify(progress, "registering", job_id=job_id)
        _check(cancelled)
        if target.exists() or _linked(target):
            _fail("SUPPLY_SOURCE_INVALID", "Source destination appeared before publication")
        staging.rename(target)
        published = True
        updated = replace(cfg, model_path="" if selector else str(target),
                          gguf_path=str(target / selector) if selector else cfg.gguf_path)
        _save(updated, row, registry_snapshot=registry_snapshot,
              manifest=manifest, artifact_sha256=manifest["artifact_sha256"],
              manifest_sha256=manifest["manifest_sha256"], revision=spec["revision"],
              supply_download_job_id=job_id)
        registered = True
        jobs._update_job(job_id, status=jobs.STATUS_READY, progress=1,
                         finished_at=jobs._now(), error=None, error_code=None)
        _notify(progress, "ready", job_id=job_id)
        return updated
    except Exception as exc:
        registry_changed = getattr(exc, "code", "") == "SUPPLY_REGISTRY_CHANGED"
        code = str(getattr(exc, "code", "SUPPLY_DOWNLOAD_FAILED"))
        if not code.startswith("SUPPLY_"):
            code = "SUPPLY_" + code.upper() if code.startswith("DOWNLOAD_") else "SUPPLY_DOWNLOAD_" + code.upper()
        if code == "SUPPLY_CANCELLED":
            jobs._update_job(job_id, status=jobs.STATUS_CANCELLED, finished_at=jobs._now())
        else:
            jobs._fail_job(job_id, code, "Model source download or verification failed")
        if isinstance(exc, ModelSupplySourceError):
            raise
        raise ModelSupplySourceError(code, "Model source download or verification failed") from exc
    finally:
        # Delete only this call's staging/publication, after resolved confinement.
        # A newer registration may have adopted the published source. A CAS
        # conflict cannot authorize deleting that source out from under it.
        cleanup = target if published and not registered and not registry_changed else staging if not published else None
        if cleanup is not None and cleanup.exists():
            cleanup.resolve().relative_to(target.parent.resolve())
            if cleanup.resolve() == target.parent.resolve() or _linked(cleanup):
                _fail("SUPPLY_SOURCE_INVALID", "Unsafe download cleanup destination")
            shutil.rmtree(cleanup)


def _conversion_error(report: dict) -> None:
    error = report.get("execution", {}).get("error") or next(iter(report.get("errors", [])), {})
    code = str(error.get("code") or "failed").upper()
    _fail("SUPPLY_CONVERSION_" + code, "HF source exists, but GGUF conversion failed: " + code)


def _convert(cfg: ModelConfig, row: dict, hf: Path, weights: list[Path], operation: Path,
             cancelled: Callable[[], bool], progress: Callable | None,
             *, registry_snapshot: dict | None) -> tuple[ModelConfig, dict, Path]:
    convert = _model_tools("gguf_convert")
    _check(cancelled)
    manifest_path = hf / "model.manifest.json"
    external_files = _registered_external_files(cfg, row)
    expected_identity = _registered_manifest_identity(registry_snapshot or {})
    try:
        if expected_identity or manifest_path.exists() or _linked(manifest_path):
            manifest = _live_manifest(hf, weights, cancelled, external_files=external_files,
                                      expected_identity=expected_identity)
        else:
            manifest = mrv.build_manifest(hf, weights, model_type="safetensors",
                                          revision=str(row.get("revision") or ""),
                                          source=cfg.huggingface_id)
            _publish_json(manifest_path, manifest)
        source_identity = descriptor._source_manifest_identity(hf)
        manifest_identity = {field: manifest[field] for field in ("artifact_sha256", "manifest_sha256")}
        if any(source_identity[field] != digest for field, digest in manifest_identity.items()):
            raise ValueError("source identity changed during inspection")
    except ModelSupplySourceError:
        raise
    except Exception as exc:
        raise ModelSupplySourceError("SUPPLY_CONVERSION_PROOF_UNAVAILABLE",
                                    "HF source exists, but its manifest/tokenizer cannot establish conversion identity") from exc
    target = _path(cfg.gguf_path)
    if target is None:
        key = hashlib.sha256(cfg.model_id.encode()).hexdigest()[:20]
        target = hf.parent / (".qlh-supply-" + key) / "model.Q4_K_M.gguf"
    try:
        target.resolve().relative_to(hf.parent.resolve())
        if target.resolve().is_relative_to(hf) or any(_linked(item) for item in (target, *target.parents)):
            raise ValueError("unsafe target")
    except ValueError as exc:
        raise ModelSupplySourceError("SUPPLY_CONVERSION_PROOF_UNAVAILABLE",
                                    "Registered conversion target is outside the source identity's models root") from exc
    _disk(target.parent, _DISK_RESERVE)
    target.parent.mkdir(parents=True, exist_ok=True)
    converter, converter_status = convert._find_converter(None)
    quantizer, quantizer_info = convert._find_quantizer(None)
    options = {"source": hf, "target": target, "outtype": _OUTTYPE,
               "converter": converter, "quantizer": quantizer}
    plan = convert.plan_conversion(**options)
    if not plan.get("valid"):
        _conversion_error(plan)
    if converter is None or quantizer is None or converter_status != "available" or quantizer_info.get("status") != "available":
        _fail("SUPPLY_CONVERSION_PROOF_UNAVAILABLE", "Conversion tool identities are unavailable")
    if _linked(converter) or _linked(quantizer):
        _fail("SUPPLY_CONVERSION_PROOF_UNAVAILABLE", "Conversion tools cannot be filesystem links")
    converter_sha = _sha(converter, cancelled)
    quantizer_sha = _sha(quantizer, cancelled)
    if quantizer_info.get("sha256") != quantizer_sha:
        _fail("SUPPLY_CONVERSION_PROOF_UNAVAILABLE", "Quantizer content differs from its discovered identity")
    # Schema v1 explicitly accepts 64-character revisions. For tools lacking a
    # verified upstream revision use the actual content revision, never invent
    # a Git commit. Evidence records which kind was available.
    quantizer_revision = (quantizer_info.get("revision")
                          if quantizer_info.get("verification") == "verified" else None) or quantizer_sha
    if not isinstance(quantizer_revision, str) or not (
            jobs._REVISION_RE.fullmatch(quantizer_revision) or jobs._SHA256_RE.fullmatch(quantizer_revision)):
        _fail("SUPPLY_CONVERSION_PROOF_UNAVAILABLE", "Quantizer revision is not an immutable identity")
    conversion = {"output_type": _OUTTYPE, "converter_revision": converter_sha,
                  "converter_sha256": converter_sha, "quantizer_revision": quantizer_revision,
                  "quantizer_sha256": quantizer_sha}
    _check(cancelled)
    _notify(progress, "converting")
    _check(cancelled)
    report = convert.execute_conversion(**options, confirmation="CONVERT")
    evidence = {"schema_version": 1, "model_id": cfg.model_id, "source": source_identity,
                "conversion": conversion, "converter_revision_kind": "content_sha256",
                "quantizer_revision_kind": "upstream" if quantizer_revision != quantizer_sha else "content_sha256",
                "plan": plan, "execution_report": report}
    # Keep execution evidence separate: the strict provenance schema has no
    # extension field. Each invocation owns a new, never-overwritten record.
    fd, evidence_name = tempfile.mkstemp(prefix="source-conversion-", suffix=".json", dir=operation)
    with os.fdopen(fd, "w", encoding="utf-8") as handle:
        json.dump(evidence, handle, ensure_ascii=True, sort_keys=True, indent=2)
    _check(cancelled)
    if not report.get("valid"):
        _conversion_error(report)
    execution = report.get("execution", {})
    stages = execution.get("stages", [])
    required = {"convert_hf_to_gguf", "llama_quantize", "inspect_verify", "atomic_publish"}
    if (execution.get("started") is not True or execution.get("published") is not True
            or not required <= {item.get("stage") for item in stages if item.get("status") == "passed"}
            or any(item.get("status") != "passed" for item in stages)):
        _fail("SUPPLY_CONVERSION_EVIDENCE_INVALID", "Converter returned incomplete execution evidence")
    try:
        _live_manifest(hf, weights, cancelled, external_files=external_files,
                       expected_identity=manifest_identity)
        if _sha(converter, cancelled) != converter_sha or _sha(quantizer, cancelled) != quantizer_sha:
            raise ValueError("conversion tools changed during execution")
        target_identity = _gguf_metadata(target)
        if execution.get("artifact", {}).get("sha256") != target_identity["artifact_sha256"]:
            raise ValueError("execution target digest mismatch")
        proof = {"schema_version": 1, "kind": "qlh.cross_format_pipeline_identity.v1",
                 "source": source_identity, "conversion": conversion,
                 "target": {"path": Path(os.path.relpath(target, hf)).as_posix(), **target_identity}}
        proof["provenance_sha256"] = _canonical(proof)
        _check(cancelled)
        _publish_json(hf / PROVENANCE, proof)
        proof, target = _proof(hf, weights, cancelled, external_files=external_files,
                               expected_identity=manifest_identity)
    except ModelSupplySourceError:
        raise
    except Exception as exc:
        raise ModelSupplySourceError("SUPPLY_CONVERSION_PROOF_UNAVAILABLE",
                                    "Conversion executed, but its source/tools/target could not be proven") from exc
    # Append the binding only to the evidence record owned by this invocation.
    evidence["provenance_sha256"] = proof["provenance_sha256"]
    Path(evidence_name).write_text(json.dumps(evidence, ensure_ascii=True, sort_keys=True,
                                            indent=2) + "\n", encoding="utf-8")
    updated = replace(cfg, model_type="both", model_path=str(hf), gguf_path=str(target))
    _save(updated, row, registry_snapshot=registry_snapshot,
          manifest=manifest, artifact_sha256=manifest["artifact_sha256"],
          manifest_sha256=manifest["manifest_sha256"], supply_conversion_evidence=evidence_name,
          cross_format_provenance_sha256=proof["provenance_sha256"])
    _notify(progress, "ready")
    return updated, proof, target


def ensure_registered_sources(model_id: str, *, operation_dir: Path, want_gguf: bool,
                              cancelled: Callable[[], bool], progress: Callable | None = None,
                              require_cross_format: bool = False) -> dict:
    """Return verified registered sources, downloading/converting only if absent.

    Result keys are ``model_config``, ``hf``, ``gguf`` and ``conversion_proof``.
    Paths come exclusively from the registry or a verified provenance target;
    operation_dir holds local execution evidence, never request commands.
    Present invalid sources/proofs fail closed. Errors expose ``code/message``.
    """
    _check(cancelled)
    _notify(progress, "resolving")
    cfg, row = _registered(model_id)
    registry_snapshot = deepcopy(row)
    expected_identity = _registered_manifest_identity(registry_snapshot or {})
    operation = Path(operation_dir).absolute()
    if any(_linked(item) for item in (operation, *operation.parents)):
        _fail("SUPPLY_SOURCE_INVALID", "Operation directory cannot traverse a filesystem link")
    operation.mkdir(parents=True, exist_ok=True)
    hf = _existing(_path(cfg.model_path), directory=True)
    gguf = _existing(_path(cfg.gguf_path), directory=False)
    if cfg.model_path and expected_identity and hf is None and gguf is not None:
        _fail("SUPPLY_SOURCE_METADATA_INVALID", "Registered HF source manifest is unavailable")
    if hf is None and gguf is None:
        cfg = _download(cfg, row, operation, cancelled, progress)
        # Reload persisted download evidence while retaining builtin metadata.
        cfg, row = _registered(model_id)
        registry_snapshot = deepcopy(row)
        hf = _existing(_path(cfg.model_path), directory=True)
        gguf = _existing(_path(cfg.gguf_path), directory=False)
    weights: list[Path] = []
    proof = None
    external_files = _registered_external_files(cfg, row)
    expected_identity = _registered_manifest_identity(registry_snapshot or {})
    if hf is not None:
        proof_present = (hf / PROVENANCE).exists() or _linked(hf / PROVENANCE)
        try:
            weights = _hf_metadata(hf, cancelled)
            if proof_present:
                proof, gguf = _proof(hf, weights, cancelled, external_files=external_files,
                                     expected_identity=expected_identity)
            elif expected_identity or (hf / "model.manifest.json").exists() or _linked(hf / "model.manifest.json"):
                _live_manifest(hf, weights, cancelled, external_files=external_files,
                               expected_identity=expected_identity)
        except ModelSupplySourceError:
            raise
        except Exception as exc:
            code = "SUPPLY_CONVERSION_PROOF_INVALID" if proof_present else "SUPPLY_SOURCE_METADATA_INVALID"
            raise ModelSupplySourceError(code, "Registered HF source failed metadata/content verification") from exc
    if gguf is not None:
        try:
            if hf is None and expected_identity:
                _live_manifest(gguf.parent, [gguf], cancelled, external_files=external_files,
                               expected_identity=expected_identity, require_hf_metadata=False)
            _gguf_metadata(gguf)
        except Exception as exc:
            raise ModelSupplySourceError("SUPPLY_SOURCE_METADATA_INVALID",
                                        "Registered GGUF source failed metadata/content verification") from exc
    _check(cancelled)
    needs_proof = require_cross_format and proof is None
    if want_gguf and hf is not None and (gguf is None or needs_proof):
        if gguf is not None and needs_proof:
            # The old GGUF remains a verified member of the old source manifest.
            # Convert to a distinct managed target instead of overwriting it.
            row = {**row, "manifest_external_files": {key: str(value) for key, value in external_files.items()}}
            cfg = replace(cfg, gguf_path="")
        try:
            cfg, proof, gguf = _convert(cfg, row, hf, weights, operation, cancelled, progress,
                                        registry_snapshot=registry_snapshot)
        except ModelSupplySourceError:
            raise
        except Exception as exc:
            raise ModelSupplySourceError("SUPPLY_CONVERSION_FAILED",
                                        "HF source exists, but conversion could not complete") from exc
    if hf is None and gguf is None:
        _fail("SUPPLY_SOURCE_MISSING", "No registered complete on-disk source is available")
    cfg = replace(cfg, model_path=str(hf) if hf else "", gguf_path=str(gguf) if gguf else "")
    _check(cancelled)
    return {"model_config": cfg, "hf": hf, "gguf": gguf, "conversion_proof": proof}


__all__ = ["ModelSupplySourceError", "ensure_registered_sources"]
