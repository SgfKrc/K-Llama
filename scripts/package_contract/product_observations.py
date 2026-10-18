"""Pure, fail-closed interpretation of product-node-observation v1.

Public API: derive_node_observation(raw, *, expected_revision, expected_node_id,
expected_identity, expected_assignment, node_addresses) -> {topology, native}.
No I/O, process inspection, clock reads, plan defaults, or source inference occur here.
ObservationError means no passing snapshot may be emitted. Old diagnostic probes are
not v1 observations and cannot be upgraded by adding a revision or wrapping timestamp.

Required raw structure (all timestamps are original observer timestamps):
  schema_version: 1; kind: product-node-observation; node_id; platform
  started_at, finished_at; commands: [{command, started_at, finished_at,
                                      returncode, stdout, stderr}]
  facts:
    identity_before/identity_after: {pid, created_at}
    runtime: {pid, created_at, captured_at, operation_id, config_id, generation,
              engine, layer_range, total_layers, loaded, full_model_loaded,
              artifact: {path, sha256, rehashed_sha256, layer_range,
                         decoder_layers}, loader_pid}
    process_tree: {complete, processes: [{pid, created_at, parent_pid, command,
        cwd, identity_after: {pid, created_at}, children_complete, child_pids,
        maps_complete, model_mappings: [path], connections_complete,
        connections: [{status, local: [IP, port], remote: [IP, port]}]}]}
    source (PC/master): {kind: runtime_manifest|synced_manifest, pid, created_at,
        source_root, source_revision, source_dirty, manifest_sha256,
        runtime_manifest_sha256, manifest: {source_revision, source_dirty,
        files: [{path, sha256}]}, rehashed_files: [{path, sha256}]}
      runtime_manifest_sha256 is the manifest identity observed from the running
      process's startup/deployment receipt, NOT a CLI revision/deployment note.
    source (frozen PC/master): {kind: installed_exe, pid, created_at,
        source_root, executable, executable_sha256, build_info, build_info_sha256,
        manifest, manifest_sha256, runtime_manifest_sha256, rehashed_files,
        source_revision, source_dirty}
      build_info_sha256 is the real file-byte SHA256. The canonical JSON SHA
      is additionally compared to independently parsed build-info file bytes.
      facts.package_rehash retains the external Python controller's fresh full
      inventory and actual UTF-8 manifest/build-info text. This proves observed
      package bytes and startup identity, without asserting a release signature.
    source (Android): {kind: installed_apk, pid, created_at, package,
        path_before, path_after, sha256_before, sha256_after, pulled_sha256,
        build_info_sha256, build_info, build_info_entry, extraction: apk_zip}
      The producer reads assets/build-info.json from bytes pulled from that
      installed path while PID/start identity and installed path/hash stay stable.
    runtime.materialization (PyTorch only): {source: safetensors_key_filtered,
        pid, created_at, operation_id, config_id, generation, artifact_sha256,
        layer_range, decoder_prefix, loaded_tensor_names, tensor_metadata:
        {name: {shape, element_size, source_file}}, materialized_bytes,
        allowed_boundary_prefixes: [prefix]}
      An empty mmap list or a descriptor/plan/count alone is never materialization
      evidence. Observed tensor names, shapes and byte accounting are required.
"""
from __future__ import annotations

import hashlib
import ipaddress
import json
import math
import ntpath
import posixpath
import re


class ObservationError(ValueError):
    """Missing, incomplete, contradictory, or unauthenticated observation."""


def require(condition, message):
    if not condition:
        raise ObservationError(message)


def _object(pairs):
    result = {}
    for key, value in pairs:
        require(key not in result, f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _constant(value):
    raise ObservationError(f"non-finite JSON constant: {value}")


def parse_json_output(stdout: str) -> dict:
    """Read exactly one complete JSON object, including pretty multiline output.

    Non-JSON log lines before/after the object are retained by the collector.
    Truncated JSON, duplicate keys, non-finite values and multiple objects fail.
    """
    require(isinstance(stdout, str), "probe stdout is not text")
    decoder = json.JSONDecoder(object_pairs_hook=_object, parse_constant=_constant)
    values, end = [], 0
    for match in re.finditer(r"(?m)^[ \t]*(?=\{)", stdout):
        if match.end() < end:
            continue
        try:
            value, end = decoder.raw_decode(stdout, match.end())
        except (ValueError, RecursionError) as exc:
            raise ObservationError(f"probe stdout contains invalid/truncated JSON: {exc}") from exc
        values.append(value)
    require(len(values) == 1 and isinstance(values[0], dict),
            "probe stdout must contain exactly one complete JSON object")
    return values[0]


def _dict(value, name):
    require(isinstance(value, dict), f"missing object: {name}")
    return value


def _list(value, name, *, nonempty=False):
    require(isinstance(value, list) and (value or not nonempty), f"missing list: {name}")
    return value


def _number(value, name):
    require(type(value) in (int, float) and math.isfinite(value) and value > 0,
            f"invalid {name}")
    return value


def _int(value, name, minimum=1):
    require(type(value) is int and value >= minimum, f"invalid integer: {name}")
    return value


def _hash(value, name, length=64):
    require(isinstance(value, str) and re.fullmatch(rf"[0-9a-f]{{{length}}}", value),
            f"missing/invalid {name}")
    return value


def _identity(value):
    value = _dict(value, "process identity")
    return (_int(value.get("pid"), "pid"), _number(value.get("created_at"), "created_at"))


def _selection(value):
    value = _dict(value, "runtime selection")
    for key in ("operation_id", "config_id"):
        require(isinstance(value.get(key), str) and bool(value[key].strip()), f"missing {key}")
    return value["operation_id"], value["config_id"], _int(value.get("generation"), "generation")


def _path(value):
    require(isinstance(value, str) and bool(value.strip()), "missing observed file path")
    # Portable comparisons for Linux/Android and Windows captures on any gate host.
    return ntpath.normcase(ntpath.normpath(value)) if "\\" in value or re.match(r"^[A-Za-z]:", value) else posixpath.normpath(value)


def _span(value, total):
    require(isinstance(value, list) and len(value) == 2 and all(type(n) is int for n in value),
            "missing observed layer_range")
    require(0 <= value[0] < value[1] <= total and value[1]-value[0] < total,
            "invalid/full-model observed layer_range")
    return value


def _ip(value):
    try:
        ip = ipaddress.ip_address(value)
        return str(ip.ipv4_mapped or ip) if isinstance(ip, ipaddress.IPv6Address) else str(ip)
    except (ValueError, TypeError):
        raise ObservationError(f"invalid observed/mapped IP address: {value!r}") from None


def _canonical_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     ensure_ascii=True, allow_nan=False).encode()).hexdigest()


_FROZEN_SOURCE_FIELDS = {
    "kind", "pid", "created_at", "source_root", "executable", "executable_sha256",
    "build_info", "build_info_sha256", "manifest", "manifest_sha256",
    "runtime_manifest_sha256", "rehashed_files", "source_revision", "source_dirty",
}
_FROZEN_MANIFEST_FIELDS = {
    "schema_version", "source_revision", "source_dirty", "executable_relative",
    "build_info_sha256", "files",
}
_FROZEN_REHASH_FIELDS = {
    "method", "observer", "started_at", "finished_at", "pid", "created_at",
    "source_root", "executable", "executable_sha256", "build_info", "build_info_sha256",
    "build_info_text", "manifest", "manifest_sha256", "manifest_text", "rehashed_files",
}


def _exact_object(value, fields, name):
    value = _dict(value, name)
    require(set(value) == fields, f"{name} has missing or extra fields")
    return value


def _frozen_relative(value):
    require(isinstance(value, str) and value and not value.startswith("/")
            and not any(ord(c) < 32 or c in '\\:<>"|?*' for c in value),
            "unsafe installed EXE package path")
    for part in value.split("/"):
        require(part not in ("", ".", "..") and not part.endswith((" ", "."))
                and not re.fullmatch(r"(?i:con|prn|aux|nul|com[1-9]|lpt[1-9])(?:\..*)?", part),
                "unsafe installed EXE package path")
    return value


def _frozen_absolute(value):
    require(isinstance(value, str) and value and not value.startswith(("\\", "//")),
            "unsafe installed EXE absolute path")
    windows = bool(re.match(r"^[A-Za-z]:[\\/]", value))
    require(windows or value.startswith("/"), "installed EXE path must be absolute")
    rest = value[3:] if windows else value[1:]
    _frozen_relative(rest.replace("\\", "/") if windows else rest)
    return _path(value)


# Fixed exclusions are never supplied by the endpoint, environment or caller.
# Only installation-root user state is omitted; nested DLL/script trees remain
# part of the program inventory, including directories named config or models.
_FROZEN_USER_ROOTS = {"models", "chat_history", "logs", "config", "local_docs", "state"}
_FROZEN_USER_FILES = {".env", ".tailscale_joined", "node_config.json", "node_secrets.json",
                      "qlh-control.sqlite3", "qlh-control.sqlite3-wal", "qlh-control.sqlite3-shm",
                      "qlh-control.sqlite3-journal"}


def _frozen_excluded(name):
    parts = name.casefold().split("/")
    return (parts[0] in _FROZEN_USER_ROOTS
            or name.casefold() == "product-package-observation.json"
            or name == "manifest/install-manifest.json"
            or (len(parts) == 1 and parts[0] in _FROZEN_USER_FILES))


def _frozen_entries(value, label):
    records, names = {}, []
    for item in _list(value, f"installed EXE {label} files", nonempty=True):
        item = _exact_object(item, {"path", "size_bytes", "sha256"}, f"installed EXE {label} file")
        name = _frozen_relative(item.get("path"))
        require(not _frozen_excluded(name), "installed EXE inventory includes observation marker or runtime user data")
        key = name.casefold()
        require(key not in records, f"duplicate installed EXE {label} file")
        records[key] = (name, _int(item.get("size_bytes"), f"installed EXE {label} size_bytes", 0),
                        _hash(item.get("sha256"), f"installed EXE {label} file SHA256"))
        names.append(name)
    require(names == sorted(names), f"installed EXE {label} file inventory must be sorted")
    return records


def _finite_json(value):
    if isinstance(value, dict):
        require(all(isinstance(k, str) for k in value), "invalid installed EXE JSON key")
        for member in value.values():
            _finite_json(member)
    elif isinstance(value, list):
        for member in value:
            _finite_json(member)
    else:
        require(value is None or type(value) in (str, bool, int)
                or (type(value) is float and math.isfinite(value)), "invalid/non-finite installed EXE JSON value")


def _frozen_json_text(value, name):
    require(isinstance(value, str), f"missing installed EXE {name} text")
    try:
        parsed = json.loads(value.removeprefix("\ufeff"), object_pairs_hook=_object, parse_constant=_constant)
    except (ValueError, RecursionError) as exc:
        raise ObservationError(f"invalid installed EXE {name} JSON: {exc}") from exc
    _finite_json(parsed)
    return _dict(parsed, f"installed EXE {name}"), value.encode("utf-8")


def _installed_exe_source(source, root, revision, package_rehash, window):
    _exact_object(source, _FROZEN_SOURCE_FIELDS, "installed EXE source")
    _finite_json(source)
    observed_exe = _frozen_absolute(root.get("executable"))
    require(_frozen_absolute(source.get("executable")) == observed_exe,
            "installed EXE differs from OS root executable")
    flavor = ntpath if re.match(r"^[A-Za-z]:", observed_exe) else posixpath
    package_root = _frozen_absolute(source.get("source_root"))
    require(package_root == flavor.dirname(observed_exe), "installed EXE source root differs from OS executable parent")
    require(_frozen_absolute(root.get("cwd")) == package_root, "installed EXE OS process cwd differs from package root")
    manifest = _exact_object(source.get("manifest"), _FROZEN_MANIFEST_FIELDS, "installed EXE manifest")
    require(type(manifest.get("schema_version")) is int and manifest["schema_version"] == 1,
            "invalid installed EXE manifest schema_version")
    relative = _frozen_relative(manifest.get("executable_relative"))
    require("/" not in relative and relative.lower().endswith(".exe")
            and _path(flavor.join(package_root, relative)) == observed_exe,
            "installed EXE manifest executable differs from OS root executable")
    digest = _canonical_hash(manifest)
    require(digest == source.get("manifest_sha256") == source.get("runtime_manifest_sha256"),
            "installed EXE manifest differs from running process startup manifest")
    expected = _frozen_entries(manifest.get("files"), "manifest")
    require(expected == _frozen_entries(source.get("rehashed_files"), "runtime rehash"),
            "installed EXE runtime rehash missing files, extra files, or content mismatch")
    exe_digest = _hash(source.get("executable_sha256"), "installed EXE SHA256")
    require(relative.casefold() in expected and expected[relative.casefold()][0] == relative
            and expected[relative.casefold()][1] > 0 and expected[relative.casefold()][2] == exe_digest,
            "installed EXE hash is not bound to manifest file entry")
    require("build-info.json" in expected, "installed EXE manifest missing build-info.json file entry")
    info = _dict(source.get("build_info"), "installed EXE build-info")
    require(type(info.get("schema_version")) is int and info["schema_version"] == 1,
            "invalid installed EXE build-info schema_version")
    info_digest = _canonical_hash(info)
    # The public SHA field is the real file-byte digest, matching the builder.
    # Canonical JSON SHA is compared independently to the fresh parsed bytes.
    build_digest = _hash(source.get("build_info_sha256"), "installed EXE build-info SHA256")
    require(build_digest == manifest.get("build_info_sha256") == expected["build-info.json"][2],
            "installed EXE build-info SHA256 differs from manifest file entry")
    for record in (source, manifest, info):
        _hash(record.get("source_revision"), "installed EXE source revision", 40)
        require(record.get("source_revision") == revision, "running source revision differs from expected revision")
        require(record.get("source_dirty") is False, "installed EXE source_dirty must be explicitly false")

    fresh = _exact_object(package_rehash, _FROZEN_REHASH_FIELDS, "independent installed EXE package rehash")
    _finite_json(fresh)
    require(fresh.get("method") == "independent_frozen_package_rehash"
            and fresh.get("observer") == "external_python_controller",
            "installed EXE package lacks independent external observer rehash")
    require(_identity(fresh) == _identity(root), "installed EXE independent rehash belongs to another process")
    began = _number(fresh.get("started_at"), "installed EXE rehash started_at")
    ended = _number(fresh.get("finished_at"), "installed EXE rehash finished_at")
    require(window is not None and window[0] <= began <= ended <= window[1],
            "installed EXE independent rehash outside original observation window")
    require(_frozen_absolute(fresh.get("source_root")) == package_root
            and _frozen_absolute(fresh.get("executable")) == observed_exe,
            "installed EXE independent rehash path differs from OS root executable")
    require(fresh.get("executable_sha256") == exe_digest,
            "installed EXE independent executable SHA256 mismatch")
    disk_manifest, _ = _frozen_json_text(fresh.get("manifest_text"), "manifest")
    disk_info, info_bytes = _frozen_json_text(fresh.get("build_info_text"), "build-info")
    require(disk_manifest == fresh.get("manifest") == manifest
            and _canonical_hash(disk_manifest) == _canonical_hash(fresh.get("manifest"))
            == fresh.get("manifest_sha256") == digest,
            "installed EXE actual manifest differs from startup manifest")
    require(disk_info == fresh.get("build_info") == info
            and _canonical_hash(disk_info) == _canonical_hash(fresh.get("build_info")) == info_digest
            and fresh.get("build_info_sha256") == build_digest,
            "installed EXE actual build-info differs from runtime build-info")
    require(expected == _frozen_entries(fresh.get("rehashed_files"), "independent rehash"),
            "installed EXE independent rehash missing files, extra files, or content mismatch")
    build_entry = expected["build-info.json"]
    require(build_entry[0] == "build-info.json" and build_entry[1] > 0 and build_entry[1] == len(info_bytes)
            and build_entry[2] == hashlib.sha256(info_bytes).hexdigest(),
            "installed EXE actual build-info bytes differ from manifest file entry")
    return info


def _source(source, root, platform, revision, package_rehash=None, window=None):
    source = _dict(source, "facts.source: runtime-bound source provenance")
    require(_identity(source) == _identity(root), "source provenance belongs to another process")
    if platform == "android":
        require(source.get("kind") == "installed_apk" and source.get("extraction") == "apk_zip",
                "Android source must be extracted from actual installed APK bytes")
        require(source.get("build_info_entry") == "assets/build-info.json", "unexpected APK build-info entry")
        package = source.get("package")
        require(isinstance(package, str) and re.fullmatch(r"com\.qlh\.inference(?:\.debug)?", package),
                "missing installed APK package identity")
        before, after = source.get("path_before"), source.get("path_after")
        require(isinstance(before, str) and before.startswith("/data/app/") and before.endswith("/base.apk")
                and before == after and package in before, "installed APK package path changed/unbound")
        digest = _hash(source.get("sha256_before"), "installed APK SHA256")
        require(source.get("sha256_after") == digest == source.get("pulled_sha256"),
                "installed APK changed during extraction or pulled bytes differ")
        info = _dict(source.get("build_info"), "installed APK build-info")
        require(_canonical_hash(info) == source.get("build_info_sha256"), "APK build-info digest mismatch")
    elif source.get("kind") == "installed_exe":
        info = _installed_exe_source(source, root, revision, package_rehash, window)
    else:
        require(source.get("kind") == ("synced_manifest" if platform == "pc" else "runtime_manifest"),
                "missing PID-bound runtime/deployment manifest and full content rehash (checkout/CLI is not running source)")
        require(_path(source.get("source_root")) == _path(root.get("cwd")), "source root differs from running process cwd")
        manifest = _dict(source.get("manifest"), "runtime source content manifest")
        digest = _canonical_hash(manifest)
        require(digest == source.get("manifest_sha256") == source.get("runtime_manifest_sha256"),
                "source manifest is not bound to the running process's loaded/startup manifest")
        expected, actual = {}, {}
        for label, entries, target in (("manifest", manifest.get("files"), expected),
                                       ("rehash", source.get("rehashed_files"), actual)):
            for item in _list(entries, f"source {label} files", nonempty=True):
                item = _dict(item, f"source {label} file")
                name = item.get("path")
                require(isinstance(name, str) and name and not name.startswith(("/", "\\"))
                        and not ntpath.isabs(name) and ".." not in name.replace("\\", "/").split("/"),
                        f"unsafe source {label} path")
                name = name.replace("\\", "/").casefold()
                require(name not in target, f"duplicate source {label} file")
                target[name] = _hash(item.get("sha256"), f"source {label} file SHA256")
        require(expected == actual, "source manifest rehash missing files, extra files, or content mismatch")
        require(any(p.endswith(".py") for p in expected), "source manifest has no executable Python source")
        info = source
        require(manifest.get("source_revision") == info.get("source_revision")
                and manifest.get("source_dirty") is False, "source manifest revision/cleanliness unknown or dirty")
    require(info.get("source_revision") == revision, "running source revision differs from expected revision")
    _hash(info.get("source_revision"), "running source revision", 40)
    require(info.get("source_dirty") is False, "running source_dirty must be explicitly false")
    return info


def _materialization(runtime, identity, root, artifact, span):
    proof = _dict(runtime.get("materialization"),
                  "runtime.materialization: DL requires selective tensor names/shapes/source files/bytes, not empty mmap")
    require(proof.get("source") == "safetensors_key_filtered", "DL selective materialization provenance missing")
    require(_identity(proof) == _identity(root) and _selection(proof) == identity,
            "DL materialization PID or operation identity mismatch")
    require(proof.get("artifact_sha256") == artifact["sha256"] and proof.get("layer_range") == span,
            "DL materialization belongs to another artifact/range")
    prefix = proof.get("decoder_prefix")
    require(isinstance(prefix, str) and prefix and prefix.endswith("."), "DL decoder key prefix missing")
    names = _list(proof.get("loaded_tensor_names"), "DL loaded_tensor_names", nonempty=True)
    require(all(isinstance(n, str) for n in names) and len(set(names)) == len(names), "invalid/duplicate DL tensor names")
    metadata = _dict(proof.get("tensor_metadata"), "DL tensor_metadata")
    require(set(names) == set(metadata), "DL loaded keys and source metadata differ")
    # Boundary ownership is derived from the observed segment, never arbitrary
    # prefixes supplied by a caller that could permit visual/MTP/full decoder keys.
    # Existing selective adapters retain the final norm on every segment for
    # model/device discovery. It is a fixed loader-owned allowlist, not a prefix
    # list accepted from observer input. Every norm byte remains accounted for.
    boundaries = ["model.norm.", "model.language_model.norm.", "transformer.ln_f."]
    if span[0] == 0:
        boundaries += ["model.embed_tokens.", "model.language_model.embed_tokens.", "transformer.wte."]
    if span[1] == runtime["total_layers"]:
        boundaries += ["lm_head."]
    observed_layers, count = set(), 0
    for name in names:
        if name.startswith(prefix):
            match = re.fullmatch(re.escape(prefix) + r"([0-9]+)\..+", name)
            require(match is not None, "DL malformed decoder tensor key")
            layer = int(match[1]); observed_layers.add(layer)
            require(span[0] <= layer < span[1], "DL loaded tensor outside selected layer range")
        else:
            require(any(name.startswith(p) for p in boundaries), "DL loaded unowned/full-model/visual/MTP tensor")
        item = _dict(metadata[name], "DL tensor source metadata")
        shape = _list(item.get("shape"), "DL tensor shape")
        require(all(type(d) is int and d > 0 for d in shape), "DL invalid tensor shape")
        _path(item.get("source_file"))
        count += math.prod(shape) * _int(item.get("element_size"), "DL tensor element_size")
    require(observed_layers == set(range(*span)), "DL selective materialization omits decoder layers")
    require(count == _int(proof.get("materialized_bytes"), "DL materialized_bytes"), "DL materialization byte accounting mismatch")



def _mapped_artifact_aliases(process, artifact, paths, *, platform, engine, start, end):
    """Validate optional Windows mapped-byte proofs without changing OS paths."""
    artifact_path = _path(artifact.get("path"))
    proved, intervals = set(), []
    for proof in _list(process.get("mapped_artifact_proofs", []), "mapped artifact proofs"):
        proof = _dict(proof, "mapped artifact proof")
        require(platform in ("master", "pc") and engine == "llama_cpp"
                and re.match(r"^[a-zA-Z]:", artifact_path)
                and proof.get("method") == "windows_read_process_memory_sha256",
                "unsupported mapped artifact identity proof")
        require(_identity(proof) == _identity(process)
                and _identity(proof.get("identity_after")) == _identity(process)
                and proof.get("maps_stable") is True, "mapped artifact PID/maps changed or unbound")
        mapped = _path(proof.get("mapped_path"))
        require(mapped in paths and mapped != artifact_path and mapped not in proved
                and mapped.lower().endswith(".gguf"), "mapped artifact alias missing/duplicate/unobserved")
        require(_path(proof.get("artifact_path")) == artifact_path
                and proof.get("sha256") == artifact.get("sha256"), "mapped bytes differ from loaded artifact")
        size = _int(artifact.get("size_bytes"), "independent artifact size_bytes")
        require(type(proof.get("artifact_size_bytes")) is int and proof["artifact_size_bytes"] == size,
                "mapped artifact byte length differs from independent rehash")
        ps, pe = _number(proof.get("started_at"), "mapped proof started_at"), _number(proof.get("finished_at"), "mapped proof finished_at")
        require(start <= ps <= pe <= end, "mapped artifact proof outside original observation")
        page = _int(proof.get("page_size"), "mapped page size")
        require(4096 <= page <= 65536 and page & (page - 1) == 0, "invalid mapped page size")
        extent = ((size + page - 1) // page) * page
        for region in _list(proof.get("regions"), "mapped file regions", nonempty=True):
            region = _dict(region, "mapped file region")
            address = _int(region.get("address"), "mapped address")
            require(address % page == 0 and address + extent <= 2**64,
                    "unaligned/overflow mapped address")
            require(type(region.get("region_bytes")) is int and region["region_bytes"] == extent
                    and type(region.get("hashed_bytes")) is int and region["hashed_bytes"] == size
                    and region.get("sha256") == artifact.get("sha256"),
                    "mapped bytes are partial, wrong extent or another artifact")
            require(all(address + extent <= a or address >= b for a, b in intervals),
                    "duplicate/overlapping mapped artifact regions")
            intervals.append((address, address + extent))
        proved.add(mapped)
    return proved


def derive_node_observation(raw, *, expected_revision, expected_node_id,
                            expected_identity, expected_assignment, node_addresses):
    """Return independently derived topology/native blocks or raise ObservationError."""
    raw = _dict(raw, "raw observation")
    require(type(raw.get("schema_version")) is int and raw["schema_version"] == 1
            and raw.get("kind") == "product-node-observation",
            "unsupported raw schema; legacy/tail-only probes cannot prove product observations")
    require(raw.get("node_id") == expected_node_id, "raw node_id differs from expected node")
    require(not raw.get("collection_errors"), "collection reported errors: " + str(raw.get("collection_errors")))
    platform = raw.get("platform")
    require(platform in ("master", "pc", "android"), "unknown observation platform")
    require((platform == "master") == (expected_node_id == "master"), "master process role/node mismatch")
    _hash(expected_revision, "expected revision", 40)
    identity = _selection(expected_identity)
    start, end = _number(raw.get("started_at"), "observation started_at"), _number(raw.get("finished_at"), "observation finished_at")
    require(start <= end, "reversed observation window")
    for command in _list(raw.get("commands"), "raw commands", nonempty=True):
        command = _dict(command, "raw command")
        argv = _list(command.get("command"), "raw command argv", nonempty=True)
        require(all(isinstance(a, str) for a in argv), "invalid raw command argv")
        require(type(command.get("returncode")) is int and command["returncode"] == 0,
                "observation command failed or did not execute")
        require(isinstance(command.get("stdout"), str) and isinstance(command.get("stderr"), str),
                "raw full stdout/stderr must be retained")
        cs, ce = _number(command.get("started_at"), "command started_at"), _number(command.get("finished_at"), "command finished_at")
        require(start <= cs <= ce <= end, "raw command outside original observation window")
    if "transport" in raw:
        transport = _dict(raw["transport"], "probe transport")
        require(type(transport.get("returncode")) is int and transport["returncode"] == 0, "probe transport failed")
        require(isinstance(transport.get("stdout"), str) and isinstance(transport.get("stderr"), str), "transport stdout/stderr missing")
        require(_number(transport.get("started_at"), "transport started_at") <= start <= end <= _number(transport.get("finished_at"), "transport finished_at"),
                "original remote observation window outside SSH transport")
    facts = _dict(raw.get("facts"), "facts")
    before, after = _identity(facts.get("identity_before")), _identity(facts.get("identity_after"))
    require(before == after, "root process restarted/PID identity changed")
    runtime = _dict(facts.get("runtime"), "facts.runtime: running model/selection receipt missing")
    require(_identity(runtime) == before, "runtime receipt is not bound to root PID/start identity")
    require(_selection(runtime) == identity, "runtime belongs to another operation/config/generation")
    captured = _number(runtime.get("captured_at"), "runtime captured_at")
    require(start <= captured <= end, "runtime receipt outside original observation window")
    total = _int(runtime.get("total_layers"), "observed total_layers", 2)
    if "_total_layers" in expected_assignment:
        require(type(expected_assignment["_total_layers"]) is int and total == expected_assignment["_total_layers"],
                "observed total_layers differs from intent plan")
    span = _span(runtime.get("layer_range"), total)
    require(span == [expected_assignment.get("start_layer"), expected_assignment.get("end_layer")], "observed range differs from assignment")
    engine = runtime.get("engine")
    require(engine in ("pytorch", "llama_cpp") and engine == expected_assignment.get("engine"), "observed engine differs from assignment")
    require(runtime.get("loaded") is True and runtime.get("full_model_loaded") is False,
            "native load not observed or full-model state missing/true")
    tree = _dict(facts.get("process_tree"), "process_tree")
    require(tree.get("complete") is True, "process tree observation incomplete/permission denied")
    processes = _list(tree.get("processes"), "process tree processes", nonempty=True)
    by_pid = {}
    for proc in processes:
        proc = _dict(proc, "observed process")
        ident = _identity(proc)
        require(ident[0] not in by_pid, "duplicate observed process")
        require(_identity(proc.get("identity_after")) == ident, "observed process PID/start changed")
        for field in ("children_complete", "maps_complete", "connections_complete"):
            require(proc.get(field) is True, f"PID {ident[0]} {field} incomplete/permission denied")
        _list(proc.get("command"), "process command", nonempty=True)
        _path(proc.get("cwd"))
        by_pid[ident[0]] = proc
    require(before[0] in by_pid and _identity(by_pid[before[0]]) == before, "root PID not in observed tree")
    root = by_pid[before[0]]
    descendants = {before[0]}
    for proc in processes:
        children = _list(proc.get("child_pids"), "observed child_pids")
        require(all(type(p) is int and p in by_pid for p in children) and len(set(children)) == len(children),
                "missing/duplicate native child process observation")
        for pid in children:
            require(by_pid[pid].get("parent_pid") == proc["pid"], "child-parent ownership mismatch")
        if proc["pid"] != before[0]:
            require(proc.get("parent_pid") in by_pid and proc["pid"] in by_pid[proc["parent_pid"]]["child_pids"], "unowned process in observation tree")
    for _ in processes:
        descendants.update(pid for proc in processes if proc["pid"] in descendants for pid in proc["child_pids"])
    require(descendants == set(by_pid), "disconnected/cyclic process tree")
    info = _source(facts.get("source"), root, platform, expected_revision,
                   facts.get("package_rehash"), (start, end))
    if platform == "android":
        require(facts["source"].get("package") == runtime.get("package_name"), "Android runtime/APK package mismatch")
    artifact = _dict(runtime.get("artifact"), "runtime artifact: loaded segment provenance missing")
    artifact_path = _path(artifact.get("path"))
    digest = _hash(artifact.get("sha256"), "loaded artifact SHA256")
    require(artifact.get("rehashed_sha256") == digest, "loaded segment artifact was not rehashed/mismatched")
    require(artifact.get("layer_range") == span and artifact.get("decoder_layers") == span[1]-span[0], "loaded artifact range/native decoder count mismatch")
    require(type(artifact.get("decoder_layers")) is int, "invalid native decoder count")
    if expected_assignment.get("artifact_sha256") is not None:
        require(digest == expected_assignment["artifact_sha256"], "loaded artifact differs from assignment artifact")
    loader = _int(runtime.get("loader_pid"), "native loader_pid")
    require(loader in by_pid, "native loader child missing from process observation")
    mapped = []
    for proc in processes:
        paths = [_path(p) for p in _list(proc.get("model_mappings"), "process model_mappings")]
        aliases = _mapped_artifact_aliases(proc, artifact, paths, platform=platform,
                                           engine=engine, start=start, end=end)
        require(all(p == artifact_path or p in aliases for p in paths),
                "unexpected/unproven/full-model mapping in process tree")
        if proc["pid"] == loader:
            mapped = [artifact_path if p in aliases else p for p in paths]
    if engine == "pytorch":
        require(loader == root["pid"], "DL materialization loader differs from root process")
        _materialization(runtime, identity, root, artifact, span)
    else:
        require(artifact_path in mapped, "native loader has no mapping for verified loaded segment")
        require(runtime.get("native_layer_count") == span[1]-span[0]
                and type(runtime.get("native_layer_count")) is int, "native runtime decoder count missing/mismatched")
    addresses = _dict(node_addresses, "explicit node->address mapping")
    reverse = {}
    for node, ips in addresses.items():
        for ip in _list(ips, f"addresses for {node}", nonempty=True):
            normalized = _ip(ip)
            require(normalized not in reverse or reverse[normalized] == node, "ambiguous node->address mapping")
            reverse[normalized] = node
    require(expected_node_id in addresses and "master" in addresses, "observer/master address mapping missing")
    peers = set()
    for proc in processes:
        for conn in _list(proc.get("connections"), "OS process connections"):
            conn = _dict(conn, "OS connection")
            if conn.get("status") != "ESTABLISHED":
                continue
            local, remote = conn.get("local"), conn.get("remote")
            require(isinstance(local, list) and len(local) == 2 and isinstance(remote, list) and len(remote) == 2,
                    "established OS connection endpoints missing")
            lip, rip = _ip(local[0]), _ip(remote[0])
            lport, rport = _int(local[1], "local port"), _int(remote[1], "remote port")
            require(lport <= 65535 and rport <= 65535, "invalid OS connection port")
            # Inspect every edge to a known cluster node, not only intended plan peers.
            peer = reverse.get(rip)
            if peer is not None and peer != expected_node_id:
                require(reverse.get(lip) == expected_node_id, "cluster socket local address does not belong to observed node")
                peers.add(peer)
            elif (lport == 8888 or rport == 8888) and not ipaddress.ip_address(rip).is_loopback:
                require(peer is not None, "worker TCP peer lacks explicit node address mapping")
    require(bool(peers), "no actual OS cluster peer connections observed")
    require(platform == "master" or peers == {"master"}, "worker-to-worker connection or missing master edge")
    common = {"node_id": expected_node_id, "operation_id": identity[0], "config_id": identity[1],
              "generation": identity[2], "captured_at": captured,
              "observation_started_at": start, "observation_finished_at": end, "complete": True}
    return {"topology": {**common, "worker_peers": sorted(peers)},
            "native": {**common, "source_revision": info["source_revision"], "source_dirty": False,
                       "source": "native_loader_receipt", "engine": engine, "layer_range": list(span),
                       "full_model_loaded": False}}


def validate_node_snapshot(raw, snapshot, **expected):
    """Return discrepancies; gate callers may instead call derive and compare directly."""
    try:
        derived = derive_node_observation(raw, **expected)
    except (ObservationError, TypeError, KeyError, ValueError) as exc:
        return [str(exc)]
    errors = []
    for section, values in derived.items():
        observed = snapshot.get(section, {}) if isinstance(snapshot, dict) else {}
        for key, value in values.items():
            if type(observed.get(key)) is not type(value) or observed.get(key) != value:
                errors.append(f"{section}.{key}: snapshot differs from raw observation")
    return errors
