"""Synthetic frozen packages only: no EXE execution, model or native loading."""
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest

import runtime_frozen_source as frozen
import runtime_source as source

REVISION = "5d157b05eab6911328b3edcb87d9188eb7d09e85"
EXE = "QLH-Edge-Inference.exe"


def _write_json(path, value):
    path.write_text(json.dumps(value, sort_keys=True, indent=2) + "\n", encoding="utf-8")


@pytest.fixture
def app(tmp_path):
    root = tmp_path / "app"
    root.mkdir()
    for name, data in {
        EXE: b"synthetic PE and embedded PYZ bytes\x00",
        "_internal/python312.dll": b"synthetic interpreter bytes",
        "_internal/engine.pyd": b"synthetic extension bytes",
        "_internal/scripts/tool.py": b"pass\n",
        "_internal/release-contract.json": b'{"schema_version":1}',
        "pubkeys/public.json": b'{"public":"synthetic test data"}',
        "manifest/companion.json": b'{"companion":true}',
    }.items():
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
    _write_json(root / frozen.BUILD_INFO_NAME, {
        "schema_version": 1, "source_revision": REVISION, "source_dirty": False,
        "source_tree_sha256": "a" * 64, "components": {".": {"source_dirty": False}},
    })
    return root


@pytest.fixture
def frozen_process(app, monkeypatch):
    state = {"pid": 123, "created_at": 10.0, "cwd": str(app), "exe": str(app / EXE)}

    def process(*_):
        return SimpleNamespace(pid=state["pid"], create_time=lambda: state["created_at"],
                               cwd=lambda: state["cwd"], exe=lambda: state["exe"])

    import psutil
    monkeypatch.setattr(psutil, "Process", process)
    monkeypatch.setattr(sys, "frozen", True, raising=False)
    monkeypatch.setattr(sys, "executable", str(app / EXE))
    monkeypatch.setenv("QLH_PRODUCT_OBSERVATION", "1")
    monkeypatch.setenv("QLH_RUNTIME_SOURCE_MANIFEST", str(app / "must-not-read-source.json"))
    monkeypatch.setattr(source, "_STARTUP", None)
    monkeypatch.setattr(source, "_ERROR", None)
    monkeypatch.setattr(source, "checkout_manifest", lambda *_: pytest.fail("frozen capture attempted Git"))
    monkeypatch.setattr(source, "_export_manifest", lambda *_: pytest.fail("frozen capture attempted source export"))
    return state


def _initialize(app):
    frozen.export_package_manifest(app)
    source.initialize()
    return source.observe_source("pc")


def test_export_uses_actual_complete_package_and_build_info(app, monkeypatch):
    monkeypatch.setenv("SOURCE_REVISION", "b" * 40)
    monkeypatch.setenv("QLH_SOURCE_DIRTY", "false")
    manifest = frozen.export_package_manifest(app)
    assert set(manifest) == {"schema_version", "source_revision", "source_dirty",
                             "executable_relative", "build_info_sha256", "files"}
    assert manifest["source_revision"] == REVISION and manifest["source_dirty"] is False
    assert manifest["executable_relative"] == EXE
    assert json.loads((app / frozen.MANIFEST_NAME).read_bytes()) == manifest
    entries = {entry["path"]: entry for entry in manifest["files"]}
    assert set(entries) == {path.relative_to(app).as_posix() for path in app.rglob("*")
                            if path.is_file() and path.name != frozen.MANIFEST_NAME}
    for name, entry in entries.items():
        data = (app / name).read_bytes()
        assert entry == {"path": name, "size_bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()}
    assert manifest["build_info_sha256"] == entries[frozen.BUILD_INFO_NAME]["sha256"]


def test_startup_is_installed_exe_pid_bound_and_detached(app, frozen_process):
    observed = _initialize(app)
    assert set(observed) == {"kind", "pid", "created_at", "source_root", "executable",
                             "executable_sha256", "build_info", "build_info_sha256", "manifest",
                             "manifest_sha256", "runtime_manifest_sha256", "rehashed_files",
                             "source_revision", "source_dirty"}
    assert observed["kind"] == "installed_exe" and observed["pid"] == 123 and observed["created_at"] == 10.0
    assert observed["source_root"] == str(app) and observed["executable"] == str(app / EXE)
    assert observed["executable_sha256"] == hashlib.sha256((app / EXE).read_bytes()).hexdigest()
    assert observed["build_info"] == json.loads((app / frozen.BUILD_INFO_NAME).read_bytes())
    assert observed["build_info_sha256"] == hashlib.sha256((app / frozen.BUILD_INFO_NAME).read_bytes()).hexdigest()
    assert observed["manifest_sha256"] == observed["runtime_manifest_sha256"] == source.canonical_hash(observed["manifest"])
    assert observed["rehashed_files"] == observed["manifest"]["files"]
    observed["manifest"]["files"].clear()
    observed["build_info"].clear()
    startup = source.startup_source()
    startup["manifest"]["files"].clear()
    assert source.observe_source("master")["kind"] == "installed_exe"
    assert source.observe_source("pc")["manifest"]["files"]


@pytest.mark.parametrize("dirty", [False, True])
def test_dirty_facts_are_preserved_without_environment_override(app, frozen_process, monkeypatch, dirty):
    path = app / frozen.BUILD_INFO_NAME
    info = json.loads(path.read_bytes())
    info["source_dirty"] = dirty
    _write_json(path, info)
    monkeypatch.setenv("SOURCE_REVISION", "b" * 40)
    monkeypatch.setenv("SOURCE_DIRTY", "false")
    observed = _initialize(app)
    assert observed["source_dirty"] is dirty and observed["manifest"]["source_dirty"] is dirty
    assert observed["build_info"]["source_dirty"] is dirty and observed["source_revision"] == REVISION


@pytest.mark.parametrize("name", [EXE, "build-info.json", "_internal/engine.pyd",
                                  "_internal/python312.dll", "_internal/scripts/tool.py",
                                  "pubkeys/public.json", "manifest/companion.json"])
@pytest.mark.parametrize("when", ["startup", "observe"])
def test_changed_package_bytes_fail_closed(app, frozen_process, name, when):
    frozen.export_package_manifest(app)
    if when == "observe":
        source.initialize()
    path = app / name
    data = path.read_bytes()
    path.write_bytes(bytes([data[0] ^ 1]) + data[1:])  # same length, actual byte hash required
    if when == "startup":
        source.initialize()
        with pytest.raises(source.SourceObservationError):
            source.startup_source()
    else:
        with pytest.raises(ValueError):
            source.observe_source("pc")


@pytest.mark.parametrize("mutation", ["extra", "missing", "nested-models", "extra-manifest", "size"])
def test_inventory_missing_extra_and_size_changes_rejected(app, frozen_process, mutation):
    _initialize(app)
    if mutation == "extra":
        (app / "extra.dll").write_bytes(b"unlisted")
    elif mutation == "missing":
        (app / "_internal/engine.pyd").unlink()
    elif mutation == "nested-models":
        path = app / "_internal/models/extra.dll"
        path.parent.mkdir()
        path.write_bytes(b"not top-level runtime data")
    elif mutation == "extra-manifest":
        (app / "manifest/extra.json").write_bytes(b"{}")
    else:
        with (app / EXE).open("ab") as stream:
            stream.write(b"extra")
    with pytest.raises(ValueError, match="inventory/content"):
        source.observe_source("pc")


def test_only_exact_post_export_install_manifest_is_excluded(app, frozen_process):
    observed = _initialize(app)
    path = app / frozen.INSTALL_MANIFEST_NAME
    path.write_bytes(b'{"signature":"synthetic post-export bytes"}')
    assert source.observe_source("pc") == observed
    path.write_bytes(b'{"signature":"different lab signing bytes"}')
    assert source.observe_source("pc") == observed
    assert "manifest/companion.json" in {entry["path"] for entry in observed["rehashed_files"]}


def test_top_level_runtime_data_changes_do_not_enter_inventory(app, frozen_process):
    observed = _initialize(app)
    for name in frozen.EXCLUDED_TOP_LEVEL_DIRECTORIES:
        path = app / name
        path.mkdir()
        (path / "mutable.txt").write_text("runtime data", encoding="utf-8")
    for name in frozen.EXCLUDED_TOP_LEVEL_FILES:
        (app / name).write_bytes(b"runtime data must not be read")
    assert source.observe_source("pc") == observed


@pytest.mark.parametrize("mutation", ["pid", "created_at", "cwd", "exe", "sys-executable"])
def test_process_identity_cwd_or_executable_drift_rejected(app, frozen_process, monkeypatch, mutation):
    _initialize(app)
    if mutation == "pid":
        frozen_process["pid"] += 1
    elif mutation == "created_at":
        frozen_process["created_at"] += 1
    elif mutation == "cwd":
        frozen_process["cwd"] = str(app.parent)
    elif mutation == "exe":
        frozen_process["exe"] = str(app / "different.exe")
    else:
        monkeypatch.setattr(sys, "executable", str(app / "different.exe"))
    with pytest.raises(ValueError):
        source.observe_source("pc")


@pytest.mark.parametrize("mutation", ["cwd", "exe", "root"])
def test_startup_binds_actual_exe_and_install_root(app, frozen_process, mutation):
    frozen.export_package_manifest(app)
    if mutation == "cwd":
        frozen_process["cwd"] = str(app.parent)
    elif mutation == "exe":
        frozen_process["exe"] = str(app / "other.exe")
    source.initialize(root=app.parent if mutation == "root" else None)
    with pytest.raises(source.SourceObservationError):
        source.startup_source()


@pytest.mark.parametrize("missing", [frozen.MANIFEST_NAME, frozen.BUILD_INFO_NAME, EXE])
def test_missing_metadata_or_exe_is_terminal_startup_failure(app, frozen_process, missing):
    frozen.export_package_manifest(app)
    path = app / missing
    data = path.read_bytes()
    path.unlink()
    source.initialize()
    error = source._ERROR
    assert error
    path.write_bytes(data)
    source.initialize()
    assert source._ERROR == error
    with pytest.raises(source.SourceObservationError):
        source.observe_source("pc")


@pytest.mark.parametrize("name", ["../outside", "/absolute", "C:/escape", "C:relative", "a\\b",
                                  "_internal//bad", "./file", "x/../file", "x:stream",
                                  "trailing.", "trailing ", "CON", "nul.json", "x\x00y"])
def test_unsafe_manifest_paths_rejected(app, frozen_process, name):
    manifest = frozen.export_package_manifest(app)
    manifest["files"][0]["path"] = name
    _write_json(app / frozen.MANIFEST_NAME, manifest)
    source.initialize()
    with pytest.raises(source.SourceObservationError, match="unsafe"):
        source.startup_source()


@pytest.mark.parametrize("mutation", ["duplicate", "case-duplicate", "size-bool", "hash", "dirty-string",
                                      "schema-bool", "revision", "missing-exe", "missing-info", "extra-field",
                                      "self", "excluded-data", "install-manifest", "unsorted", "wrong-exe",
                                      "info-hash", "info-revision", "info-dirty"])
def test_invalid_or_contradictory_manifest_rejected(app, frozen_process, mutation):
    manifest = frozen.export_package_manifest(app)
    if mutation in {"duplicate", "case-duplicate"}:
        entry = deepcopy(manifest["files"][0])
        if mutation == "case-duplicate":
            entry["path"] = entry["path"].upper()
        manifest["files"].append(entry)
    elif mutation == "size-bool":
        manifest["files"][0]["size_bytes"] = True
    elif mutation == "hash":
        manifest["files"][0]["sha256"] = "invalid"
    elif mutation == "dirty-string":
        manifest["source_dirty"] = "false"
    elif mutation == "schema-bool":
        manifest["schema_version"] = True
    elif mutation == "revision":
        manifest["source_revision"] = "invalid"
    elif mutation in {"missing-exe", "missing-info"}:
        name = EXE if mutation == "missing-exe" else frozen.BUILD_INFO_NAME
        manifest["files"] = [item for item in manifest["files"] if item["path"] != name]
    elif mutation == "extra-field":
        manifest["signature_verified"] = True
    elif mutation in {"self", "excluded-data", "install-manifest"}:
        manifest["files"][0]["path"] = {"self": frozen.MANIFEST_NAME,
                                         "excluded-data": "models/hidden.dll",
                                         "install-manifest": frozen.INSTALL_MANIFEST_NAME}[mutation]
    elif mutation == "unsorted":
        manifest["files"].reverse()
    elif mutation == "wrong-exe":
        manifest["executable_relative"] = "other.exe"
    elif mutation == "info-hash":
        manifest["build_info_sha256"] = "0" * 64
    elif mutation == "info-revision":
        manifest["source_revision"] = "b" * 40
    else:
        manifest["source_dirty"] = True
    _write_json(app / frozen.MANIFEST_NAME, manifest)
    source.initialize()
    with pytest.raises(source.SourceObservationError):
        source.startup_source()


@pytest.mark.parametrize("name", [frozen.MANIFEST_NAME, frozen.BUILD_INFO_NAME])
@pytest.mark.parametrize("data", [b'{"schema_version":1,"schema_version":1}',
                                  b'{"invalid":NaN}', b'{"invalid":1e999}', b'[]', b'{', b'\xff'])
def test_metadata_requires_strict_json(app, frozen_process, name, data):
    frozen.export_package_manifest(app)
    (app / name).write_bytes(data)
    source.initialize()
    with pytest.raises(source.SourceObservationError):
        source.startup_source()


@pytest.mark.parametrize("mutation", ["revision", "dirty", "schema", "missing", "too-large"])
def test_export_refuses_invalid_build_info(app, mutation):
    path = app / frozen.BUILD_INFO_NAME
    info = json.loads(path.read_bytes())
    if mutation == "revision":
        info["source_revision"] = "not-a-revision"
    elif mutation == "dirty":
        info["source_dirty"] = "false"
    elif mutation == "schema":
        info["schema_version"] = True
    elif mutation == "missing":
        del info["source_dirty"]
    else:
        info["padding"] = "a" * (64 * 1024)
    _write_json(path, info)
    with pytest.raises(ValueError):
        frozen.export_package_manifest(app)
    assert not (app / frozen.MANIFEST_NAME).exists()


def test_export_refuses_overwrite_and_runtime_never_adopts_reexport(app, frozen_process):
    _initialize(app)
    metadata = app / frozen.MANIFEST_NAME
    original = metadata.read_bytes()
    with pytest.raises(ValueError, match="overwrite"):
        frozen.export_package_manifest(app)
    assert metadata.read_bytes() == original
    metadata.unlink()
    (app / EXE).write_bytes(b"new package bytes")
    frozen.export_package_manifest(app)
    source.initialize()
    with pytest.raises(ValueError, match="startup receipt"):
        source.observe_source("pc")


def test_reformatted_metadata_is_not_recaptured(app, frozen_process):
    observed = _initialize(app)
    metadata = app / frozen.MANIFEST_NAME
    metadata.write_text(json.dumps(observed["manifest"]), encoding="utf-8")
    with pytest.raises(ValueError, match="startup receipt"):
        source.observe_source("pc")


def test_manifest_exe_must_be_running_os_exe(app, frozen_process):
    (app / "other.exe").write_bytes(b"another executable")
    frozen.export_package_manifest(app, executable_relative="other.exe")
    source.initialize()
    with pytest.raises(source.SourceObservationError, match="manifest executable"):
        source.startup_source()


@pytest.mark.parametrize("where", ["file", "directory", "metadata", "excluded-root", "install-manifest", "root"])
def test_symlinks_or_junctions_refused(app, tmp_path, where):
    target = tmp_path / "outside"
    target.mkdir()
    (target / "file").write_bytes(b"outside")
    if where == "root":
        link = tmp_path / "app-link"
        command = ["cmd", "/c", "mklink", "/J", str(link), str(app)] if os.name == "nt" else None
    elif where in {"directory", "excluded-root"}:
        link = app / ("linked" if where == "directory" else "models")
        command = ["cmd", "/c", "mklink", "/J", str(link), str(target)] if os.name == "nt" else None
    else:
        link = app / {"file": "linked.dll", "metadata": frozen.MANIFEST_NAME,
                      "install-manifest": frozen.INSTALL_MANIFEST_NAME}[where]
        command = None
    if command:
        result = subprocess.run(command, capture_output=True, check=False)
        if result.returncode:
            pytest.skip("test junction creation unavailable")
    else:
        try:
            link.symlink_to(app if where == "root" else target if where in {"directory", "excluded-root"}
                            else target / "file", target_is_directory=where in {"root", "directory", "excluded-root"})
        except OSError:
            pytest.skip("test symlink creation unavailable")
    with pytest.raises(ValueError):
        frozen.export_package_manifest(link if where == "root" else app)


def test_reparse_attribute_refused_without_following_link(app, monkeypatch):
    original = Path.lstat

    def lstat(path, *args, **kwargs):
        result = original(path, *args, **kwargs)
        if path.name == "engine.pyd":
            return SimpleNamespace(st_mode=result.st_mode, st_file_attributes=0x400)
        return result

    monkeypatch.setattr(Path, "lstat", lstat)
    with pytest.raises(ValueError, match="reparse"):
        frozen.export_package_manifest(app)


def test_file_changed_during_stream_hash_is_rejected(app, monkeypatch):
    original = Path.open

    class MutatingReader:
        def __init__(self, stream):
            self.stream = stream
            self.changed = False

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return self.stream.__exit__(*args)

        def fileno(self):
            return self.stream.fileno()

        def read(self, size):
            result = self.stream.read(size)
            if result and not self.changed:
                self.changed = True
                with original(app / EXE, "ab") as writer:
                    writer.write(b"mutation")
            return result

    def patched_open(path, *args, **kwargs):
        stream = original(path, *args, **kwargs)
        return MutatingReader(stream) if path == app / EXE and args == ("rb",) else stream

    monkeypatch.setattr(Path, "open", patched_open)
    with pytest.raises(ValueError, match="changed"):
        frozen.export_package_manifest(app)


def test_metadata_changed_during_verification_rejected(app, frozen_process, monkeypatch):
    frozen.export_package_manifest(app)
    original = frozen._package_bytes

    def changed(root):
        result = original(root)
        with (root / frozen.MANIFEST_NAME).open("ab") as stream:
            stream.write(b"\n")
        return result

    monkeypatch.setattr(frozen, "_package_bytes", changed)
    source.initialize()
    with pytest.raises(source.SourceObservationError, match="metadata changed"):
        source.startup_source()



@pytest.mark.parametrize("when", ["startup", "observe"])
@pytest.mark.parametrize("field", ["pid", "created_at", "cwd", "exe"])
def test_process_drift_during_package_hash_is_rejected(app, frozen_process, monkeypatch, when, field):
    frozen.export_package_manifest(app)
    if when == "observe":
        source.initialize()
    original = frozen._verify_package

    def changed(root):
        result = original(root)
        frozen_process[field] = {"pid": 456, "created_at": 20.0,
                                 "cwd": str(app.parent), "exe": str(app / "other.exe")}[field]
        return result

    monkeypatch.setattr(frozen, "_verify_package", changed)
    if when == "startup":
        source.initialize()
        with pytest.raises(source.SourceObservationError):
            source.startup_source()
    else:
        with pytest.raises(ValueError):
            source.observe_source("pc")

def test_frozen_default_disabled_does_no_package_or_process_io(app, frozen_process, monkeypatch):
    monkeypatch.delenv("QLH_PRODUCT_OBSERVATION")
    monkeypatch.setattr(frozen, "capture_startup", lambda *_args, **_kwargs: pytest.fail("disabled package capture"))
    import psutil
    monkeypatch.setattr(psutil, "Process", lambda *_: pytest.fail("disabled process inspection"))
    source.initialize()
    assert source._STARTUP is None and source._ERROR is None
    with pytest.raises(source.SourceObservationError, match="not captured"):
        source.startup_source()


def test_source_inventory_contract_is_unchanged(app):
    with pytest.raises(source.SourceObservationError, match="directories missing"):
        source._inventory(app)


def test_imports_do_not_load_model_or_native_runtime():
    script = """
import sys
sys.path.insert(0, sys.argv[1])
import runtime_source
import runtime_frozen_source
assert not any(name == p or name.startswith(p + '.') for name in sys.modules
               for p in ('torch', 'llama_cpp', 'model_module', 'api_server', 'psutil'))
"""
    result = subprocess.run([sys.executable, "-B", "-Xutf8", "-c", script,
                             str(Path(__file__).resolve().parents[1] / "src")],
                            capture_output=True, text=True, check=False)
    assert result.returncode == 0, result.stderr
