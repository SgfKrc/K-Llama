"""Offline frozen-source contract checks; fixtures are not product evidence."""
from copy import deepcopy
import hashlib
import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


contract = load("frozen_product_contract_test", ROOT / "scripts/package_contract/product_observations.py")
fixtures = load("frozen_product_fixture_test", ROOT / "tests/product_observation_fixtures.py")
REVISION = "a" * 40
IDENTITY = {"operation_id": "op", "config_id": "cfg", "generation": 3}


def frozen_facts(pid, created_at, root="C:/Program Files/QLH"):
    info = {"schema_version": 1, "source_revision": REVISION, "source_dirty": False,
            "source_tree_sha256": "c" * 64, "components": {}}
    info_text = json.dumps(info, sort_keys=True, indent=2) + "\n"
    contents = {"QLH.exe": b"synthetic frozen executable", "build-info.json": info_text.encode(),
                "_internal/PYZ.pyz": b"synthetic PYZ", "_internal/native.pyd": b"synthetic PYD",
                "_internal/python312.dll": b"synthetic DLL", "scripts/probe.py": b"pass\n"}
    files = [{"path": name, "size_bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()}
             for name, data in sorted(contents.items())]
    manifest = {"schema_version": 1, "source_revision": REVISION, "source_dirty": False,
                "executable_relative": "QLH.exe", "build_info_sha256": hashlib.sha256(info_text.encode()).hexdigest(),
                "files": files}
    source = {"kind": "installed_exe", "pid": pid, "created_at": created_at, "source_root": root,
              "executable": root + "/QLH.exe", "executable_sha256": hashlib.sha256(contents["QLH.exe"]).hexdigest(),
              "build_info": info, "build_info_sha256": hashlib.sha256(info_text.encode()).hexdigest(),
              "manifest": manifest, "manifest_sha256": contract._canonical_hash(manifest),
              "runtime_manifest_sha256": contract._canonical_hash(manifest), "rehashed_files": deepcopy(files),
              "source_revision": REVISION, "source_dirty": False}
    fresh = {key: deepcopy(value) for key, value in source.items() if key in contract._FROZEN_REHASH_FIELDS}
    fresh.update(method="independent_frozen_package_rehash", observer="external_python_controller",
                 started_at=104.0, finished_at=105.0, build_info_text=info_text,
                 manifest_text=json.dumps(manifest, sort_keys=True, indent=2) + "\n")
    return source, fresh


def example(platform="pc"):
    assignments = [{"node_id": "master", "engine": "llama_cpp", "start_layer": 0, "end_layer": 2},
                   {"node_id": "rog", "engine": "llama_cpp", "start_layer": 2, "end_layer": 4}]
    assignment = assignments[0 if platform == "master" else 1]
    raw, _, _, addresses = fixtures.node_observation(IDENTITY, assignment, assignments, REVISION, 4)
    root = raw["facts"]["process_tree"]["processes"][0]
    source, fresh = frozen_facts(root["pid"], root["created_at"])
    raw["facts"].update(source=source, package_rehash=fresh)
    # Frozen startup and independent OS observations share the EXE package root.
    root.update(executable=source["executable"], cwd=source["source_root"])
    expected = dict(expected_revision=REVISION, expected_node_id=assignment["node_id"],
                    expected_identity=IDENTITY, expected_assignment=assignment, node_addresses=addresses)
    return raw, expected


def derive(raw, expected):
    return contract.derive_node_observation(raw, **expected)


def resign_source(raw):
    source = raw["facts"]["source"]
    source["manifest_sha256"] = source["runtime_manifest_sha256"] = contract._canonical_hash(source["manifest"])


@pytest.mark.parametrize("platform", ["pc", "master"])
def test_unsigned_frozen_package_is_accepted_by_source_bytes_without_release_signature(platform):
    raw, expected = example(platform)
    original = deepcopy(raw)
    result = derive(raw, expected)
    assert result["native"]["source_revision"] == REVISION
    assert result["native"]["source_dirty"] is False
    assert raw == original
    assert "signature" not in raw["facts"]["source"]


@pytest.mark.parametrize("field", sorted(contract._FROZEN_SOURCE_FIELDS))
def test_each_required_frozen_source_field_is_mandatory(field):
    raw, expected = example()
    raw["facts"]["source"].pop(field)
    with pytest.raises(contract.ObservationError):
        derive(raw, expected)


@pytest.mark.parametrize("field", sorted(contract._FROZEN_REHASH_FIELDS))
def test_each_independent_rehash_field_is_mandatory(field):
    raw, expected = example()
    raw["facts"]["package_rehash"].pop(field)
    with pytest.raises(contract.ObservationError):
        derive(raw, expected)


@pytest.mark.parametrize("field", sorted(contract._FROZEN_MANIFEST_FIELDS))
def test_each_manifest_field_is_mandatory(field):
    raw, expected = example()
    raw["facts"]["source"]["manifest"].pop(field)
    resign_source(raw)
    with pytest.raises(contract.ObservationError):
        derive(raw, expected)


@pytest.mark.parametrize("section", ["source", "package_rehash", "manifest", "file"])
def test_unknown_fields_cannot_extend_frozen_contract(section):
    raw, expected = example()
    source = raw["facts"]["source"]
    target = (source["manifest"] if section == "manifest" else source["manifest"]["files"][0]
              if section == "file" else raw["facts"][section])
    target["arbitrary"] = "untrusted"
    resign_source(raw)
    with pytest.raises(contract.ObservationError):
        derive(raw, expected)


@pytest.mark.parametrize("section", ["manifest", "runtime", "independent"])
@pytest.mark.parametrize("mutation", ["missing", "extra", "duplicate", "case-duplicate", "sha", "size", "bool-size", "extra-field"])
def test_all_three_inventories_require_exact_unique_typed_byte_records(section, mutation):
    raw, expected = example()
    source = raw["facts"]["source"]
    entries = (source["manifest"]["files"] if section == "manifest" else source["rehashed_files"]
               if section == "runtime" else raw["facts"]["package_rehash"]["rehashed_files"])
    if mutation == "missing": entries.pop()
    if mutation == "extra": entries.append({"path": "unlisted.dll", "size_bytes": 1, "sha256": "e" * 64})
    if mutation == "duplicate": entries.append(deepcopy(entries[0]))
    if mutation == "case-duplicate": entries.append({**entries[0], "path": entries[0]["path"].swapcase()})
    if mutation == "sha": entries[0]["sha256"] = "e" * 64
    if mutation == "size": entries[0]["size_bytes"] += 1
    if mutation == "bool-size": entries[0]["size_bytes"] = True
    if mutation == "extra-field": entries[0]["trusted"] = True
    resign_source(raw)
    with pytest.raises(contract.ObservationError):
        derive(raw, expected)


@pytest.mark.parametrize("path", ["../outside.dll", "/outside.dll", "C:/outside.dll", "C:outside.dll",
    "a/../outside.dll", "a//b.dll", "./x.dll", "a/./b.dll", "a\\b.dll", "a/x.dll:stream",
    "a/x.dll.", "a/x.dll ", "a/NUL.dll", "a/COM1", "a/*", "x\x00.dll", "//server/share/x.dll",
    "product-package-observation.json", "logs/x.dll", "models/private.gguf", "state/x.sqlite3",
    "node_secrets.json", "manifest/install-manifest.json"])
def test_frozen_inventory_refuses_unsafe_or_reserved_paths_even_if_all_lists_agree(path):
    raw, expected = example()
    source, fresh = raw["facts"]["source"], raw["facts"]["package_rehash"]
    for entries in (source["manifest"]["files"], source["rehashed_files"], fresh["rehashed_files"]):
        entries[0]["path"] = path
    resign_source(raw)
    fresh["manifest"] = deepcopy(source["manifest"])
    fresh["manifest_sha256"] = source["manifest_sha256"]
    fresh["manifest_text"] = json.dumps(fresh["manifest"])
    with pytest.raises(contract.ObservationError):
        derive(raw, expected)


@pytest.mark.parametrize("section,field,value", [
    (section, field, value)
    for section in ("source", "manifest", "build_info")
    for field, value in (("schema_version", True), ("source_dirty", 0), ("source_dirty", "false"),
                         ("source_dirty", True), ("source_revision", "a" * 8),
                         ("source_revision", "A" * 40), ("source_revision", "f" * 40))
    if section != "source" or field != "schema_version"
])
def test_source_manifest_and_build_info_require_clean_full_observed_revision(section, field, value):
    raw, expected = example()
    source = raw["facts"]["source"]
    target = source if section == "source" else source[section]
    target[field] = value
    if section == "build_info":
        source["build_info_sha256"] = contract._canonical_hash(source["build_info"])
        source["manifest"]["build_info_sha256"] = source["build_info_sha256"]
    resign_source(raw)
    with pytest.raises(contract.ObservationError):
        derive(raw, expected)


@pytest.mark.parametrize("section,field,value", [
    ("source", "runtime_manifest_sha256", "f" * 64), ("source", "manifest_sha256", "f" * 64),
    ("source", "executable_sha256", "f" * 64), ("source", "build_info_sha256", "f" * 64),
    ("source", "executable", "C:/other/QLH.exe"), ("source", "source_root", "C:/other"),
    ("fresh", "executable", "C:/other/QLH.exe"), ("fresh", "source_root", "C:/other"),
    ("fresh", "executable_sha256", "f" * 64), ("fresh", "build_info_sha256", "f" * 64),
    ("fresh", "manifest_sha256", "f" * 64), ("fresh", "pid", True), ("fresh", "created_at", float("nan")),
    ("fresh", "observer", "product_api"), ("fresh", "method", "api_cached_rehash"),
    ("fresh", "started_at", 1), ("fresh", "finished_at", 999),
])
def test_binding_paths_hashes_process_identity_and_external_capture_window(section, field, value):
    raw, expected = example()
    raw["facts"]["source" if section == "source" else "package_rehash"][field] = value
    with pytest.raises(contract.ObservationError):
        derive(raw, expected)


@pytest.mark.parametrize("value", [None, "", "QLH.exe", "C:QLH.exe", "C:/other/QLH.exe", "C:/pkg/../QLH.exe",
                                 "\\\\server\\share\\QLH.exe", "C:/QLH.exe:stream"])
def test_root_executable_must_be_os_absolute_path_bound_to_package(value):
    raw, expected = example()
    raw["facts"]["process_tree"]["processes"][0]["executable"] = value
    with pytest.raises(contract.ObservationError):
        derive(raw, expected)


@pytest.mark.parametrize("member", ["QLH.exe", "build-info.json"])
def test_mandatory_byte_identities_cannot_be_omitted_with_self_consistent_lists(member):
    raw, expected = example()
    source, fresh = raw["facts"]["source"], raw["facts"]["package_rehash"]
    for entries in (source["manifest"]["files"], source["rehashed_files"], fresh["rehashed_files"]):
        entries[:] = [entry for entry in entries if entry["path"] != member]
    resign_source(raw)
    fresh["manifest"] = deepcopy(source["manifest"])
    fresh["manifest_sha256"] = source["manifest_sha256"]
    fresh["manifest_text"] = json.dumps(fresh["manifest"])
    with pytest.raises(contract.ObservationError):
        derive(raw, expected)


@pytest.mark.parametrize("key", ["manifest_text", "build_info_text"])
@pytest.mark.parametrize("mutation", ["duplicate", "nan", "overflow", "invalid", "different", "bool-schema"])
def test_actual_json_bytes_cannot_hide_duplicates_nonfinite_values_or_changed_metadata(key, mutation):
    raw, expected = example()
    fresh = raw["facts"]["package_rehash"]
    text = fresh[key]
    if mutation == "duplicate": text = text.replace('"schema_version": 1', '"schema_version": 1, "schema_version": 1')
    if mutation == "nan": text = text.replace('"schema_version": 1', '"schema_version": NaN')
    if mutation == "overflow": text = text.replace('"schema_version": 1', '"schema_version": 1e999')
    if mutation == "invalid": text = "{"
    if mutation == "different": text = text.replace(REVISION, "f" * 40)
    if mutation == "bool-schema": text = text.replace('"schema_version": 1', '"schema_version": true')
    fresh[key] = text
    with pytest.raises(contract.ObservationError):
        derive(raw, expected)


def test_actual_build_info_byte_hash_matters_when_parsed_canonical_hash_matches():
    raw, expected = example()
    raw["facts"]["package_rehash"]["build_info_text"] += "\n"
    with pytest.raises(contract.ObservationError, match="actual build-info bytes"):
        derive(raw, expected)


def test_api_rehashed_inventory_alone_cannot_satisfy_independent_evidence():
    raw, expected = example()
    raw["facts"].pop("package_rehash")
    with pytest.raises(contract.ObservationError, match="independent"):
        derive(raw, expected)


def test_android_does_not_accept_installed_exe_source():
    raw, expected = example()
    raw["platform"] = "android"
    with pytest.raises(contract.ObservationError, match="installed APK"):
        derive(raw, expected)


@pytest.mark.parametrize("section", ["manifest", "runtime", "independent"])
def test_inventory_requires_canonical_sort_order(section):
    raw, expected = example()
    source = raw["facts"]["source"]
    entries = (source["manifest"]["files"] if section == "manifest" else source["rehashed_files"]
               if section == "runtime" else raw["facts"]["package_rehash"]["rehashed_files"])
    entries.reverse()
    resign_source(raw)
    with pytest.raises(contract.ObservationError, match="sorted"):
        derive(raw, expected)


def test_empty_executable_cannot_be_accepted_with_matching_hashes_and_inventories():
    raw, expected = example()
    source, fresh = raw["facts"]["source"], raw["facts"]["package_rehash"]
    for entries in (source["manifest"]["files"], source["rehashed_files"], fresh["rehashed_files"]):
        entry = next(item for item in entries if item["path"] == "QLH.exe")
        entry.update(size_bytes=0, sha256=hashlib.sha256(b"").hexdigest())
    source["executable_sha256"] = fresh["executable_sha256"] = hashlib.sha256(b"").hexdigest()
    resign_source(raw)
    fresh["manifest"] = deepcopy(source["manifest"])
    fresh["manifest_sha256"] = source["manifest_sha256"]
    fresh["manifest_text"] = json.dumps(fresh["manifest"])
    with pytest.raises(contract.ObservationError, match="manifest file entry"):
        derive(raw, expected)


@pytest.mark.parametrize("platform", ["pc", "master"])
def test_frozen_os_cwd_substitution_is_rejected_with_all_other_facts_unchanged(platform):
    raw, expected = example(platform)
    original = deepcopy(raw)
    derive(raw, expected)
    raw["facts"]["process_tree"]["processes"][0]["cwd"] = "C:/different-os-working-directory"
    with pytest.raises(contract.ObservationError, match="OS process cwd differs from package root"):
        derive(raw, expected)
    raw["facts"]["process_tree"]["processes"][0]["cwd"] = original["facts"]["process_tree"]["processes"][0]["cwd"]
    assert raw == original


def test_frozen_os_cwd_uses_the_same_windows_path_normalization_as_package_root():
    raw, expected = example()
    raw["facts"]["process_tree"]["processes"][0]["cwd"] = raw["facts"]["source"]["source_root"].upper().replace("/", "\\")
    assert derive(raw, expected)["native"]["source_dirty"] is False
