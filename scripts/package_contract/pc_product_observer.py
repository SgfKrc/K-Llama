"""Read-only PC/master OS observation bracketed by in-process loader receipts.

The HTTP endpoint must already be running with QLH_PRODUCT_OBSERVATION=1.
This script runs under the external experiment controller's Python and psutil;
these are not dependencies added to the observed frozen product. installed_exe
observations verify source bytes, independently of release signing policy.
No model preparation, loading, configuration or process launch occurs here.
"""
from __future__ import annotations

import argparse
import hashlib
import os
import ntpath
from copy import deepcopy
import json
from pathlib import Path
import sys
import time
import urllib.parse
import urllib.error
import urllib.request
import uuid


def _identity(process):
    return {"pid": process.pid, "created_at": process.create_time()}


def process_observation(pid, api_port):
    import psutil
    start = time.time()
    result = {"started_at": start, "process_tree": {"complete": False, "processes": []}}
    try:
        root = psutil.Process(pid)
        result["identity_before"] = _identity(root)
        processes = [root, *root.children(recursive=True)]
        complete = True
        for process in processes:
            entry = {"pid": process.pid}
            try:
                entry.update(_identity(process), parent_pid=process.ppid(), command=process.cmdline(), cwd=process.cwd())
                # Legacy/source mocks need not implement exe(); frozen provenance
                # explicitly requires this OS fact in rehash_installed_exe/validator.
                exe = getattr(process, "exe", None)
                if callable(exe):
                    try:
                        entry["executable"] = exe()
                    except (psutil.Error, OSError) as exc:
                        entry["executable_error"] = type(exc).__name__ + ": " + str(exc)
                children = sorted(p.pid for p in process.children())
                entry.update(child_pids=children, children_complete=True)
                entry["model_mappings"] = sorted({m.path for m in process.memory_maps(grouped=False)
                    if m.path.lower().endswith((".gguf", ".safetensors", ".bin", ".pt", ".pth"))})
                entry["maps_complete"] = True
                entry["connections"] = [{"status": c.status, "local": list(c.laddr or []), "remote": list(c.raddr or [])}
                                         for c in process.net_connections(kind="tcp")]
                entry["connections_complete"] = True
                entry["identity_after"] = _identity(psutil.Process(process.pid))
                entry["children_complete"] = children == sorted(p.pid for p in process.children())
                complete = complete and entry["children_complete"] and entry["identity_after"] == _identity(process)
            except (psutil.Error, OSError) as exc:
                entry["error"] = type(exc).__name__ + ": " + str(exc)
                complete = False
            result["process_tree"]["processes"].append(entry)
        root_entry = result["process_tree"]["processes"][0]
        if not any(c["status"] == "LISTEN" and c["local"] and c["local"][1] == api_port
                   for c in root_entry.get("connections", [])):
            raise ValueError("runtime receipt PID does not own the API listener")
        result["identity_after"] = _identity(psutil.Process(pid))
        if {p.pid for p in processes} != {pid, *(p.pid for p in root.children(recursive=True))}:
            complete = False
            result["error"] = "native process tree changed during observation"
        result["process_tree"]["complete"] = complete
    except (psutil.Error, OSError, ValueError) as exc:
        result["error"] = type(exc).__name__ + ": " + str(exc)
    result["finished_at"] = time.time()
    return result


def _request(base, nonce):
    request = urllib.request.Request(base + "/api/diagnostics/product-runtime?" + urllib.parse.urlencode({"nonce": nonce}))
    token = os.environ.get("QLH_PRODUCT_AUTH_TOKEN")
    if token:
        request.add_header("Authorization", "Bearer " + token)
    with urllib.request.urlopen(request, timeout=120) as response:
        contract = _frozen_contract()
        return json.load(response, object_pairs_hook=contract._object, parse_constant=contract._constant)


def rehash_artifact(runtime):
    """Verify loaded bytes and derive their canonical supply identity.

    Master preparation can bind the model_sync directory digest, while worker
    preparation binds the supply weights inventory digest. Both must be proved
    from live bytes; they are different digests of the same verified directory.
    No model_sync hashing API is used here (its file cache is not independent).
    """
    source = Path(__file__).resolve().parents[2] / "src"
    if str(source) not in sys.path:
        sys.path.insert(0, str(source))
    from model_supply_assets import sha256_file, validate_supply_manifest, _safe_path, _link, _MODEL_SUFFIXES
    artifact = runtime.get("artifact") or {}
    path = Path(artifact.get("path", ""))
    if not path.is_absolute() or _link(path):
        raise ValueError("loaded artifact path must be absolute and not a filesystem link")

    def stamp(member):
        info = member.stat()
        return info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns

    if path.is_file():
        if path.suffix.lower() != ".gguf":
            raise ValueError("unsupported loaded artifact file format")
        before = stamp(path)
        actual = sha256_file(path)
        if before != stamp(path):
            raise ValueError("loaded GGUF changed during independent rehash")
        files = [{"path": str(path), "sha256": actual}]
        binding = {"size_bytes": before[2]}
    elif path.is_dir():
        marker = path / "supply.manifest.json"
        marker_before = sha256_file(marker)
        manifest = validate_supply_manifest(json.loads(marker.read_text(encoding="utf-8")))
        if manifest["format"] != "safetensors":
            raise ValueError("directory artifact must use the safetensors supply contract")
        if (manifest["layer_range"] != artifact.get("layer_range")
                or manifest["total_layers"] != runtime.get("total_layers")):
            raise ValueError("on-disk supply range/total differs from loaded artifact")
        expected = {item["path"] for item in manifest["files"]} | {marker.name}

        def inventory():
            found = set()
            for member in path.rglob("*"):
                if _link(member):
                    raise ValueError("loaded segment directory contains a filesystem link")
                if member.is_file():
                    found.add(member.relative_to(path).as_posix())
                elif not member.is_dir():
                    raise ValueError("loaded segment directory contains a special file")
            if found != expected:
                raise ValueError("loaded segment file inventory differs from supply contract")
            return {name: stamp(_safe_path(path, name)) if name != marker.name else stamp(marker)
                    for name in sorted(found)}

        before = inventory()
        files = []
        for item in manifest["files"]:
            member = _safe_path(path, item["path"])
            if member.stat().st_size != item["size_bytes"]:
                raise ValueError("loaded segment member size differs from supply contract")
            actual_member = sha256_file(member)
            if actual_member != item["sha256"]:
                raise ValueError("loaded segment member digest differs from supply contract")
            files.append({**item, "sha256": actual_member})
        if before != inventory() or marker_before != sha256_file(marker):
            raise ValueError("loaded segment changed during independent rehash")
        # model_supply_assets._weight_digest: canonical JSON weight inventory.
        weights = [item for item in files if item["kind"] == "weights"]
        actual = hashlib.sha256(json.dumps(weights, ensure_ascii=True, sort_keys=True,
            separators=(",", ":"), allow_nan=False).encode("utf-8")).hexdigest()
        # Reproduce compute_model_sha256's exact path/size/file-SHA stream,
        # including support files AND the supply marker. All member SHAs above
        # came from fresh byte reads, not the manifest or model_sync file cache.
        # Path ordering deliberately uses Path, as does the loader on each OS.
        entries = [*files, {"path": marker.name, "size_bytes": before[marker.name][2],
                           "sha256": marker_before}]
        directory = hashlib.sha256()
        for item in sorted(entries, key=lambda entry: path / entry["path"]):
            member = Path(item["path"])
            if (member.name in {"model.sha256", "model.sha256.meta.json"}
                    or member.name.endswith(".part")
                    or not member.name.lower().endswith(_MODEL_SUFFIXES)
                    or any(part.startswith(".") for part in member.parts)):
                continue
            directory.update(f"{item['path']}\0{item['size_bytes']}\0{item['sha256']}\n".encode("utf-8"))
        directory_sha = directory.hexdigest()
        loaded_sha = artifact.get("sha256")
        if loaded_sha == actual:
            scheme = "safetensors_supply_weights_v1"
        elif loaded_sha == directory_sha:
            scheme = "model_sync_directory_v1"
        else:
            raise ValueError("independently rehashed artifact differs from loaded identity "
                             "(neither supply weights nor model_sync directory digest matches)")
        binding = {"loaded_sha256": loaded_sha, "loaded_digest_scheme": scheme,
                   "directory_sha256": directory_sha, "manifest_sha256": manifest["manifest_sha256"],
                   "marker_sha256": marker_before}
    else:
        raise ValueError("loaded artifact is unavailable")
    if path.is_file() and actual != artifact.get("sha256"):
        raise ValueError("independently rehashed artifact differs from loaded identity")
    return {"path": str(path), "sha256": actual, "files": files, **binding}


def _frozen_contract():
    directory = str(Path(__file__).resolve().parent)
    if directory not in sys.path:
        sys.path.insert(0, directory)
    import product_observations
    return product_observations


def rehash_installed_exe(root):
    """Fresh filesystem-only package reads anchored to the observed OS EXE.

    The shared helper reads the entire actual package twice without a cache.
    No API inventory, API rehashed_files, revision environment or installation
    configuration chooses the files or root. psutil runs in this external
    experiment controller; the product gains no Python/psutil dependency.
    """
    import psutil
    contract = _frozen_contract()
    source = str(Path(__file__).resolve().parents[2] / "src")
    if source not in sys.path:
        sys.path.insert(0, source)
    import runtime_frozen_source as frozen

    started = time.time()
    identity = {key: root.get(key) for key in ("pid", "created_at")}
    contract._identity(identity)
    observed_exe = contract._frozen_absolute(root.get("executable"))

    def check_process():
        # Use a fresh psutil object; Windows caches exe()/create_time() per object.
        process = psutil.Process(identity["pid"])
        executable = Path(process.exe())
        if (_identity(process) != identity
                or contract._frozen_absolute(str(executable)) != observed_exe):
            raise ValueError("installed EXE OS process identity/executable changed during rehash")
        if contract._frozen_absolute(process.cwd()) != contract._frozen_absolute(str(executable.parent)):
            raise ValueError("installed EXE OS process cwd differs from executable parent during rehash")
        return executable

    executable = check_process()
    package_root = frozen._root(executable.parent)
    current = frozen._verify_package(package_root)
    # Retain actual text, including whitespace/BOM, so the pure validator can
    # bind canonical JSON contents AND byte SHA to the build-info file entry.
    marker, _, marker_bytes = frozen._hash_member(package_root, frozen.MANIFEST_NAME,
                                                 limit=frozen._MANIFEST_LIMIT)
    info, _, info_bytes = frozen._hash_member(package_root, frozen.BUILD_INFO_NAME,
                                             limit=frozen._BUILD_INFO_LIMIT)
    if (marker["sha256"] != current["_manifest_file_sha256"]
            or info["sha256"] != current["build_info_sha256"]):
        raise ValueError("installed EXE metadata changed during independent rehash")
    if check_process() != executable:
        raise ValueError("installed EXE OS executable changed during rehash")
    manifest = current["manifest"]
    if (package_root / manifest["executable_relative"]) != executable:
        raise ValueError("installed EXE manifest differs from OS root executable")
    return {"method": "independent_frozen_package_rehash", "observer": "external_python_controller",
            "started_at": started, "finished_at": time.time(), **identity,
            "source_root": str(package_root), "executable": str(executable),
            **{key: current[key] for key in ("executable_sha256", "build_info", "build_info_sha256",
                                            "manifest", "manifest_sha256", "rehashed_files")},
            "build_info_text": info_bytes.decode("utf-8"), "manifest_text": marker_bytes.decode("utf-8")}


def bind_artifact_rehash(runtime, rehashed):
    """Return detached v1 facts after proving the loaded-to-supply digest binding.

    The original endpoint receipts remain verbatim in raw.receipts/commands.
    Preserve the load digest on the derived facts, too; do not rewrite either
    the in-process immutable materialization proof or any historical evidence.
    """
    result = deepcopy(runtime)
    artifact = result["artifact"]
    loaded_sha, supply_sha = artifact["sha256"], rehashed["sha256"]
    if loaded_sha != supply_sha:
        proof = result.get("materialization") or {}
        if (result.get("engine") != "pytorch"
                or rehashed.get("loaded_sha256") != loaded_sha
                or rehashed.get("directory_sha256") != loaded_sha
                or rehashed.get("loaded_digest_scheme") != "model_sync_directory_v1"
                or proof.get("artifact_sha256") != loaded_sha
                or proof.get("artifact_path") != artifact.get("path")
                or proof.get("layer_range") != artifact.get("layer_range")):
            raise ValueError("independent digest binding differs from loaded materialization proof")
        artifact["load_identity"] = {"sha256": loaded_sha, "scheme": "model_sync_directory_v1"}
        proof["loaded_artifact_sha256"] = loaded_sha
        proof["artifact_sha256"] = supply_sha
    artifact["sha256"] = supply_sha
    artifact["rehashed_sha256"] = supply_sha
    if "size_bytes" in rehashed:
        if type(rehashed["size_bytes"]) is not int or rehashed["size_bytes"] <= 0:
            raise ValueError("independently rehashed artifact has invalid size")
        artifact["size_bytes"] = rehashed["size_bytes"]
    return result



def _capture_windows_mapping(**kwargs):
    # Direct CLI execution and package imports share the same sibling helper.
    directory = str(Path(__file__).resolve().parent)
    if directory not in sys.path:
        sys.path.insert(0, directory)
    from windows_mapped_artifact import capture_windows_mapped_artifact
    return capture_windows_mapped_artifact(**kwargs)


def observe_mapping_aliases(observed, runtime, rehashed, commands):
    """Prove Windows hardlink/retired-name mappings from actual mapped bytes.

    Keep every OS path verbatim. A disk hash, equal basename or missing old path
    is never an alias proof. The native mappings themselves must match the full
    independently rehashed proper GGUF, within stable PID and map inventories.
    """
    if runtime.get("engine") != "llama_cpp":
        return
    artifact = runtime.get("artifact") or {}
    expected = artifact.get("path", "")
    normalize = lambda name: ntpath.normcase(ntpath.normpath(name))
    for process in observed.get("process_tree", {}).get("processes", []):
        aliases = sorted({name for name in process.get("model_mappings", [])
                          if normalize(name) != normalize(expected)})
        for mapped in aliases:
            began = time.time()
            record = {"command": ["ReadProcessMemory", "mapped-artifact", str(process["pid"]), mapped],
                      "started_at": began, "stdout": "", "stderr": ""}
            try:
                if (not expected.lower().endswith(".gguf") or not mapped.lower().endswith(".gguf")
                        or type(rehashed.get("size_bytes")) is not int):
                    raise ValueError("unexpected model mapping lacks a complete GGUF byte identity")
                proof = _capture_windows_mapping(pid=process["pid"], created_at=process["created_at"],
                    mapped_path=mapped, artifact_path=expected, expected_sha256=rehashed["sha256"],
                    expected_size=rehashed["size_bytes"])
                process.setdefault("mapped_artifact_proofs", []).append(proof)
                record.update(returncode=0, stdout=json.dumps(proof, ensure_ascii=False))
            except Exception as exc:
                record.update(returncode=1, stderr=type(exc).__name__ + ": " + str(exc))
                raise
            finally:
                record["finished_at"] = time.time()
                commands.append(record)


def _same_runtime(before, after):
    first, second = deepcopy(before), deepcopy(after)
    first.pop("captured_at", None)
    second.pop("captured_at", None)
    return first == second


def observe_pc(*, node_id, platform, base="http://127.0.0.1:8000", operation_id=None):
    start, commands, captures = time.time(), [], []
    result = {"schema_version": 1, "kind": "product-node-observation", "node_id": node_id,
              "platform": platform, "started_at": start, "commands": commands,
              "facts": {}, "collection_errors": [], "receipts": captures}
    parsed = urllib.parse.urlsplit(base)
    if parsed.scheme != "http" or parsed.hostname not in {"127.0.0.1", "localhost", "::1"} or parsed.path not in {"", "/"}:
        raise ValueError("runtime collector requires a loopback HTTP endpoint")
    api_port = parsed.port or 80

    def receipt():
        nonce, began = uuid.uuid4().hex, time.time()
        command = {"command": ["HTTP GET", base + "/api/diagnostics/product-runtime", "nonce=" + nonce], "started_at": began}
        try:
            value = _request(base.rstrip("/"), nonce)
            command.update(returncode=0, stdout=json.dumps(value, ensure_ascii=False), stderr="")
            if not isinstance(value, dict) or value.get("nonce") != nonce or value.get("errors"):
                raise ValueError("runtime receipt failed identity/error checks")
            if type(value.get("pid")) is not int or value["pid"] <= 0:
                raise ValueError("runtime receipt lacks process PID")
            runtime = value.get("runtime")
            if not isinstance(runtime, dict) or runtime.get("node_id") != node_id:
                raise ValueError("running node ID differs from requested observer")
            if operation_id is not None and runtime.get("operation_id") != operation_id:
                raise ValueError("running model operation differs from requested operation")
            captures.append(value)
            return value
        except Exception as exc:
            command["returncode"] = 1
            command.setdefault("stdout", "")
            if isinstance(exc, urllib.error.HTTPError):
                # Preserve the endpoint's precise missing-fact response, not
                # merely urllib's generic "HTTP Error 409" text.
                try:
                    command["stdout"] = exc.read().decode("utf-8", "replace")
                finally:
                    exc.close()
            command["stderr"] = type(exc).__name__ + ": " + str(exc)
            raise
        finally:
            command["finished_at"] = time.time()
            commands.append(command)

    try:
        before = receipt()
        observed = process_observation(before["pid"], api_port)
        commands.append({"command": ["psutil", "recursive-process-tree", str(before["pid"])],
                         "started_at": observed["started_at"], "finished_at": observed["finished_at"],
                         "returncode": 0 if not observed.get("error") and observed["process_tree"]["complete"] else 1,
                         "stdout": json.dumps(observed, ensure_ascii=False), "stderr": observed.get("error", "")})
        result["facts"].update({key: observed[key] for key in ("identity_before", "identity_after", "process_tree") if key in observed})
        if (before.get("source") or {}).get("kind") == "installed_exe":
            roots = [entry for entry in observed.get("process_tree", {}).get("processes", [])
                     if entry.get("pid") == before["pid"]]
            if len(roots) != 1:
                raise ValueError("installed EXE root process missing/duplicate in OS observation")
            record = {"command": ["external-python-observer", "independent-installed-exe-rehash",
                                  str(before["pid"]), roots[0].get("executable", "")],
                      "started_at": time.time(), "stdout": "", "stderr": ""}
            try:
                package_rehash = rehash_installed_exe(roots[0])
                result["facts"]["package_rehash"] = package_rehash
                record.update(returncode=0, stdout=json.dumps(package_rehash, ensure_ascii=False))
            except Exception as exc:
                record.update(returncode=1, stderr=type(exc).__name__ + ": " + str(exc))
                raise
            finally:
                record["finished_at"] = time.time()
                commands.append(record)
        began = time.time()
        record = {"command": ["independent-artifact-rehash", before["runtime"].get("artifact", {}).get("path", "")],
                  "started_at": began, "stdout": "", "stderr": ""}
        try:
            rehashed = rehash_artifact(before["runtime"])
            record.update(returncode=0, stdout=json.dumps(rehashed, ensure_ascii=False))
        except Exception as exc:
            record.update(returncode=1, stderr=type(exc).__name__ + ": " + str(exc))
            raise
        finally:
            record["finished_at"] = time.time()
            commands.append(record)
        observe_mapping_aliases(observed, before["runtime"], rehashed, commands)
        after = receipt()
        if (before.get("pid"), before.get("created_at")) != (after.get("pid"), after.get("created_at")):
            raise ValueError("API process restarted during observation")
        if (before.get("source") or {}).get("kind") == "installed_exe":
            contract = _frozen_contract()
            contract._finite_json(before["source"])
            contract._finite_json(after.get("source"))
            if contract._canonical_hash(before["source"]) != contract._canonical_hash(after.get("source")):
                raise ValueError("actual runtime/source changed during OS observation")
        if not _same_runtime(before["runtime"], after["runtime"]) or before.get("source") != after.get("source"):
            raise ValueError("actual runtime/source changed during OS observation")
        runtime = bind_artifact_rehash(after["runtime"], rehashed)
        result["facts"].update(runtime=runtime, source=after["source"])
        if observed.get("error") or not observed["process_tree"]["complete"]:
            raise ValueError(observed.get("error", "incomplete process observation"))
    except Exception as exc:
        result["collection_errors"].append(type(exc).__name__ + ": " + str(exc))
    result["finished_at"] = time.time()
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--node-id", required=True)
    parser.add_argument("--platform", choices=("master", "pc"), default="pc")
    parser.add_argument("--base", default="http://127.0.0.1:8001")
    parser.add_argument("--operation-id")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    result = observe_pc(node_id=args.node_id, platform=args.platform, base=args.base, operation_id=args.operation_id)
    rendered = json.dumps(result, ensure_ascii=False, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("x", encoding="utf-8") as stream:
            stream.write(rendered)
    print(rendered, end="")
    return 1 if result["collection_errors"] else 0


if __name__ == "__main__":
    sys.exit(main())
