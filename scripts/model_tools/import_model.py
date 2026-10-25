#!/usr/bin/env python3
"""Fail-closed model import wizard with staging, manifests and SQLite registration."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

import local_store

try:
    from model_registry_validation import build_manifest, validate_model_artifact, write_manifest
    from proxy_config import proxy_environment, resolve_http_proxy
except ImportError:
    from src.model_registry_validation import build_manifest, validate_model_artifact, write_manifest
    from src.proxy_config import proxy_environment, resolve_http_proxy

_WEIGHT_SUFFIXES = (".safetensors", ".gguf", ".bin")
_SAFE_SUFFIXES = (".safetensors", ".bin")


def resolve_target(repo_or_path: str, target: str | None, models_root: str = "models") -> Path:
    source = Path(repo_or_path)
    if source.is_dir():
        return source.absolute()
    name = repo_or_path.strip("/").split("/")[-1]
    if not name:
        raise ValueError(f"cannot resolve model name from source: {repo_or_path!r}")
    return Path(target or os.path.join(models_root, name)).absolute()


def _weight_files(target: Path) -> list[Path]:
    if not target.is_dir():
        return []
    return sorted(path for path in target.rglob("*") if path.is_file() and path.name.lower().endswith(_WEIGHT_SUFFIXES))


def download_model(repo_or_path: str, target: Path, *, use_modelscope: bool = False, proxy: str = "") -> list[Path]:
    """Download into a caller-owned staging directory and return weight files."""
    source = Path(repo_or_path)
    if source.is_dir():
        return _weight_files(target)
    target.mkdir(parents=True, exist_ok=True)
    resolved_proxy = resolve_http_proxy(proxy or None)
    if use_modelscope:
        code = (
            "from modelscope import snapshot_download; "
            f"snapshot_download({repo_or_path!r}, local_dir={str(target)!r})"
        )
        with proxy_environment(resolved_proxy) as environment:
            result = subprocess.run(
                [sys.executable, "-c", code],
                capture_output=True, text=True,
                encoding="utf-8", errors="replace",
                env=environment,
            )
        if result.returncode != 0:
            raise RuntimeError(f"ModelScope download failed: {(result.stderr or result.stdout)[-300:]}")
    else:
        import huggingface_hub
        # huggingface_hub reads standard proxy variables at request time.
        with proxy_environment(resolved_proxy) as environment:
            keys = ("HTTP_PROXY", "HTTPS_PROXY", "http_proxy", "https_proxy")
            previous = {key: os.environ.get(key) for key in keys}
            try:
                os.environ.update({key: environment[key] for key in keys})
                # 注：`local_dir_use_symlinks` 在 huggingface_hub **1.x 已被移除**；1.x 会把它 pop 掉
                # 并给警告（不会 TypeError），且默认行为就是不再使用 symlink ⇒ 直接不传。
                huggingface_hub.snapshot_download(repo_id=repo_or_path, local_dir=str(target))
            finally:
                for key, value in previous.items():
                    if value is None:
                        os.environ.pop(key, None)
                    else:
                        os.environ[key] = value
    files = _weight_files(target)
    if not files:
        raise RuntimeError("download completed but no safetensors/gguf/bin weights were found")
    return files


def verify_files(files: list[Path], expected_sha256: str | None = None) -> dict[str, Any]:
    """Return the backwards-compatible raw digest and fail on empty assets."""
    if not files:
        raise ValueError("no model weight files found; empty directories are not importable")
    total_bytes = 0
    digest = hashlib.sha256()
    for path in files:
        if not path.is_file() or path.stat().st_size <= 0:
            raise ValueError(f"weight file is missing or empty: {path}")
        size = path.stat().st_size
        total_bytes += size
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
    raw_sha256 = digest.hexdigest()
    summary = {"file_count": len(files), "total_bytes": total_bytes, "sha256": raw_sha256}
    if expected_sha256 and raw_sha256 != expected_sha256.lower():
        raise ValueError(f"SHA-256 不匹配 (mismatch): expected {expected_sha256.lower()}, got {raw_sha256}")
    return summary


def _infer_artifact(target: Path, gguf_path: str = "") -> dict[str, Any]:
    files = _weight_files(target)
    safe_files = [item for item in files if item.name.lower().endswith(_SAFE_SUFFIXES)]
    gguf_files = [item for item in files if item.suffix.lower() == ".gguf"]
    explicit = Path(gguf_path).expanduser().absolute() if gguf_path else None
    if not explicit and len(gguf_files) > 1:
        raise ValueError("multiple GGUF files found; pass --gguf-path to select one")
    selected_gguf = explicit or (gguf_files[0] if len(gguf_files) == 1 else None)
    model_type = "both" if safe_files and selected_gguf else "safetensors" if safe_files else "gguf" if selected_gguf else ""
    if not model_type:
        raise ValueError("cannot infer model type: expected safetensors/bin or GGUF weights")
    artifact = validate_model_artifact(model_type, str(target) if safe_files else "", str(selected_gguf) if selected_gguf else "")
    if selected_gguf and selected_gguf not in files:
        files.append(selected_gguf)
        artifact["files"] = [*artifact["safetensors_files"], selected_gguf]
    return artifact


def register_model(model_id: str, target: Path, summary: dict[str, Any], *, gguf_path: str = "", revision: str = "", registry_snapshot=None) -> bool:
    """Validate and publish against the version observed before import work."""
    snapshot = registry_snapshot if registry_snapshot is not None else local_store.get_local_model_registry_snapshot(model_id)
    artifact = _infer_artifact(target, gguf_path)
    manifest = summary.get("manifest") or build_manifest(target, artifact["files"], model_type=artifact["model_type"], revision=revision, source="import-model")
    config = {
        "model_id": model_id,
        "name": model_id,
        "model_type": artifact["model_type"],
        "model_path": artifact["model_path"],
        "gguf_path": artifact["gguf_path"],
        "quantization": "Q4_K_M" if artifact["model_type"] == "gguf" else "fp16",
        "sha256": summary["sha256"],
        "artifact_sha256": manifest["artifact_sha256"],
        "manifest_sha256": manifest["manifest_sha256"],
        "source": "import-model",
        "revision": revision,
        "manifest": manifest,
    }
    return local_store.publish_local_model_registry(model_id, config, expected_revision=snapshot.registry_revision)


def main(argv: list[str] | argparse.Namespace | None = None) -> int:
    parser = argparse.ArgumentParser(description="model import: resolve -> download -> verify -> register")
    parser.add_argument("source", help="Hugging Face repo id, ModelScope path or local directory")
    parser.add_argument("--target", default="", help="target model directory (default models/<repo name>)")
    parser.add_argument("--model-id", default="", help="registered model_id (default target name)")
    parser.add_argument("--expected-sha256", default="", help="strict raw SHA-256 verification")
    parser.add_argument("--gguf-path", default="", help="explicit GGUF registration path")
    parser.add_argument("--proxy", default="", help="user HTTP(S) proxy, overriding environment settings")
    parser.add_argument("--revision", default="", help="source revision recorded in the manifest")
    parser.add_argument("--modelscope", action="store_true", help="download via ModelScope")
    parser.add_argument("--skip-download", action="store_true", help="verify/register an existing local directory only")
    parser.add_argument("--register", action="store_true", help="persist to main-node SQLite model_registry")
    parser.add_argument("--json", action="store_true", help="output JSON summary")
    args = parser.parse_args(argv) if not isinstance(argv, argparse.Namespace) else argv

    target: Path | None = None
    staging: Path | None = None
    published = False
    remote_import = False
    registered = False
    try:
        target = resolve_target(args.source, getattr(args, "target", ""))
        registry_model_id = getattr(args, "model_id", "") or target.name
        registry_snapshot = local_store.get_local_model_registry_snapshot(registry_model_id) if getattr(args, "register", False) else None
        local_source = Path(args.source).is_dir()
        if not getattr(args, "skip_download", False) and not local_source:
            if target.exists():
                raise RuntimeError(f"target already exists; refusing in-place import: {target}")
            target.parent.mkdir(parents=True, exist_ok=True)
            staging = Path(tempfile.mkdtemp(prefix=f".{target.name}.qlh-import-", dir=target.parent))
            remote_import = True
            print(f"[1/4] downloading {args.source} -> staging ...", flush=True)
            files = download_model(args.source, staging, use_modelscope=getattr(args, "modelscope", False), proxy=getattr(args, "proxy", ""))
            target_for_validation = staging
        else:
            target_for_validation = target
            files = _weight_files(target_for_validation)
            if not files and not local_source:
                raise RuntimeError("target directory has no model weights")

        print(f"[2/4] validating {len(files)} weight files ...", flush=True)
        artifact = _infer_artifact(target_for_validation, getattr(args, "gguf_path", ""))
        files = artifact["files"]
        summary = verify_files(files, getattr(args, "expected_sha256", "") or None)
        manifest = build_manifest(target_for_validation, artifact["files"], model_type=artifact["model_type"], revision=getattr(args, "revision", ""), source=args.source)
        if staging is None:
            from model_registry_validation import publish_registration_manifest
            publish_registration_manifest(target_for_validation, manifest)
        else:
            write_manifest(target_for_validation, manifest)
        summary.update({"artifact_sha256": manifest["artifact_sha256"], "manifest_sha256": manifest["manifest_sha256"], "manifest": manifest, "model_type": artifact["model_type"]})

        if staging is not None:
            staging.replace(target)
            published = True
        model_id = getattr(args, "model_id", "") or target.name
        print(f"[3/4] registration ready: model_id={model_id}")
        registered = False
        if getattr(args, "register", False):
            registered = register_model(model_id, target, summary, gguf_path=getattr(args, "gguf_path", ""), revision=getattr(args, "revision", ""), registry_snapshot=registry_snapshot)
            print(f"[4/4] registration {'succeeded' if registered else 'failed'} (main-node SQLite)")
            if remote_import and not registered:
                from model_download_jobs import retain_unregistered_publication
                print(json.dumps(retain_unregistered_publication(target)), file=sys.stderr)
        else:
            print("[4/4] not registered (use --register to write main-node SQLite)")
        result = {"model_id": model_id, "target": str(target), "model_type": artifact["model_type"], "file_count": summary["file_count"], "total_bytes": summary["total_bytes"], "sha256": summary["sha256"], "artifact_sha256": summary["artifact_sha256"], "manifest_sha256": summary["manifest_sha256"], "registered": registered}
        if getattr(args, "json", False):
            print(json.dumps(result, ensure_ascii=False, indent=1))
        else:
            print(f"complete: {result['file_count']} files / {result['total_bytes']} bytes")
            print(f"  SHA-256: {result['sha256']}")
        return 0 if registered or not getattr(args, "register", False) else 2
    except Exception as exc:
        if staging is not None and staging.exists() and not published:
            shutil.rmtree(staging, ignore_errors=True)
        if remote_import and published and not registered and target is not None:
            from model_download_jobs import retain_unregistered_publication
            print(json.dumps(retain_unregistered_publication(target)), file=sys.stderr)
        print(f"[error] import failed: {str(exc)[:300]}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
