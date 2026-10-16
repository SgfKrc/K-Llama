"""Static release-build checks/staging; never manufacture product observations.

The workflow owns compilation. This helper authenticates its outputs without
installing or launching them. Signing keys are provisioned by the trusted runner;
only paths/configuration are handled here, never private-key bytes.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path, PureWindowsPath
import re
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]


class BuildError(ValueError):
    """Public static diagnostic: never include paths, configuration values or tool output."""


def require(value, message):
    if not value:
        raise BuildError(message)


def tool(root, relative):
    spec = importlib.util.spec_from_file_location("release_build_" + Path(relative).stem, root / relative)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def revision_pair(revision, event_sha):
    require(isinstance(revision, str) and re.fullmatch(r"[0-9a-f]{40}", revision), "Full lowercase revision required")
    require(revision == event_sha, "revision must equal event github.sha / run.head_sha")


def clean_info(info, revision):
    require(type(info) is dict and type(info.get("schema_version")) is int and info["schema_version"] == 1,
            "Invalid build-info schema")
    require(info.get("source_revision") == revision and info.get("source_dirty") is False,
            "Build requires the exact revision and explicit source_dirty=false")
    require(re.fullmatch(r"[0-9a-f]{64}", info.get("source_tree_sha256", "")), "Source tree identity missing")
    components = info.get("components")
    require(isinstance(components, dict) and bool(components), "Fixed submodule provenance missing")
    for component in components.values():
        require(component.get("source_dirty") is False and component.get("revision_matches_pin") is True,
                "Submodule pin or clean source check failed")
    return info


def observe(root, revision):
    # Canonical provider checks the fixed Git graph, actual file bytes, dirty
    # index/worktrees and ignored production sources. No CLI revision override.
    return clean_info(tool(root, "scripts/package_contract/build_provenance.py").collect_build_info(root), revision)


def capture(command):
    result = subprocess.run([str(arg) for arg in command], capture_output=True, text=True,
                            encoding="utf-8", errors="replace", timeout=1200, check=False)
    # Do not echo arbitrary tool errors/arguments, which may contain local config.
    require(result.returncode == 0, "Required static artifact verifier/tool failed")
    return result.stdout


def digest(path):
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def configured_file(name):
    value = os.environ.get(name, "")
    require(bool(value), name + " must reference a pre-provisioned runner file")
    path = Path(value)
    require(path.is_absolute() and path.is_file(), name + " must reference an existing absolute file")
    return path


def work_path(root, platform):
    run = os.environ.get("GITHUB_RUN_ID", "")
    attempt = os.environ.get("GITHUB_RUN_ATTEMPT", "")
    require(bool(re.fullmatch(r"[0-9]+", run)) and bool(re.fullmatch(r"[0-9]+", attempt)), "GitHub run/attempt required")
    return root / "build" / "release-build" / (run + "-" + attempt) / platform


def preflight(root, platform, revision, event_sha):
    revision_pair(revision, event_sha)
    work = work_path(root, platform)
    require(not work.exists(), "Build work directory already exists; refuse a stale run")
    if platform in ("windows", "linux"):
        configured_file("QLH_SIGNING_KEY")  # existence only: never open private keys
    if platform == "windows":
        require(os.environ.get("QLH_BUILD_PROFILE") == "llama_cpp_only", "Explicit self-contained Windows edge profile required")
        require((root / "packaging/.venv-packaging/Scripts/python.exe").is_file(), "Provision packaging/.venv-packaging first")
        for name in ("QLH-Edge-Inference", "QLH-Launcher"):
            require(not (root / "packaging/dist" / name).exists(), "Old Windows output exists; use a fresh checkout")
    elif platform == "linux":
        require(not list((root / "packaging/packaging/linux").glob("qlh-edge-inference-cpu_*.deb")), "Old Debian output exists")
    elif platform == "android":
        require(not list((root / "android/app/build/outputs/apk/release").glob("*.apk")), "Old APK output exists")
        pin = os.environ.get("ANDROID_RELEASE_CERT_SHA256", "").replace(":", "").lower()
        require(bool(re.fullmatch(r"[0-9a-f]{64}", pin)), "Independent Android release certificate pin required")
        # actions/checkout clean=true removes ignored configuration. Use an
        # existing runner-owned properties file, copied byte-for-byte for Gradle.
        config = configured_file("QLH_RELEASE_ANDROID_KEYSTORE_PROPERTIES")
        require(not config.resolve().is_relative_to(root.resolve()), "Keystore configuration must be provisioned outside the checkout")
        contents = config.read_bytes()
        fields = dict(re.findall(r"^\s*(storeFile|storePassword|keyAlias|keyPassword)\s*=\s*(.+?)\s*$",
                                 contents.decode("utf-8"), re.M))
        require(set(fields) == {"storeFile", "storePassword", "keyAlias", "keyPassword"}, "Incomplete pre-provisioned keystore configuration")
        destination = root / "android/keystore.properties"
        require(not destination.exists(), "Refuse to overwrite existing Android signing configuration")
        # Paths in storeFile must be absolute (Java properties escaping applies),
        # so the provisioned config does not depend on its copied location.
        store = fields["storeFile"].replace("\\:", ":").replace("\\\\", "\\")
        require(Path(store).is_absolute() and Path(store).is_file(), "Configured Android keystore file is unavailable")
        destination.write_bytes(contents)
        destination.chmod(0o600)
    else:
        raise BuildError("Unsupported build platform")
    info = observe(root, revision)
    work.mkdir(parents=True, exist_ok=False)
    snapshot = work / "source" / "build-info.json"
    snapshot.parent.mkdir()
    snapshot.write_text(json.dumps(info, ensure_ascii=True, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    return work


def expected_info(root, platform, revision):
    snapshot = work_path(root, platform) / "source/build-info.json"
    provider = tool(root, "scripts/package_contract/build_provenance.py")
    # The existing strict verifier retains all field types, component pins and
    # source-byte identities. It never refreshes/relabels the original snapshot.
    return clean_info(provider.verify_build_info(root, snapshot), revision)


def same_info(actual, expected):
    require(json.dumps(actual, sort_keys=True) == json.dumps(expected, sort_keys=True),
            "Artifact build-info differs from original pre-build source snapshot")


def verified_bundle(gate, app, platform, variant, contract, expected, work):
    report = gate.Report(platform)
    work.mkdir(parents=True, exist_ok=False)
    manifest = gate.check_bundle(report, app=app, contract=contract, platform=platform,
                                 variant=variant, version_key="launcher" if variant == "launcher" else "product",
                                 kind="launcher" if variant == "launcher" else "application", work=work,
                                 required_revision=expected["source_revision"])
    require(manifest is not None and report.as_dict()["passed"] is True, "Trusted bundle/version/contract/revision verification failed")
    same_info(gate.read_object(app / "build-info.json"), expected)
    return manifest


def copy_file(source, target):
    require(source.is_file(), "Final artifact/attachment is missing")
    before = digest(source)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("xb") as output, source.open("rb") as stream:
        shutil.copyfileobj(stream, output, 1024 * 1024)
    require(before == digest(target) == digest(source), "Artifact changed while staging")


def stage_windows(root, output, expected, gate, contract, work):
    for variant, directory in (("cpu", "QLH-Edge-Inference"), ("launcher", "QLH-Launcher")):
        app = root / "packaging/dist" / directory
        manifest = verified_bundle(gate, app, "windows", variant, contract, expected, work / variant)
        # Only the authenticated file set, plus its signed manifest, is uploaded.
        # Unreferenced local state/configuration never enters the artifact.
        gate._clone_bundle(app, manifest, output / directory)
        verified_bundle(gate, output / directory, "windows", variant, contract, expected, work / (variant + "-staged"))


def stage_linux(root, output, expected, gate, contract, work):
    matches = list((root / "packaging/packaging/linux").glob("qlh-edge-inference-cpu_*.deb"))
    require(len(matches) == 1, "Exactly one freshly built CPU .deb required")
    package = matches[0]
    original = digest(package)
    fields = capture(["dpkg-deb", "--field", package, "Package", "Version", "Architecture"])
    metadata = dict(line.split(": ", 1) for line in fields.splitlines() if ": " in line)
    require(metadata == {"Package": "qlh-edge-inference-cpu", "Version": contract["version"]["product"], "Architecture": "amd64"},
            "Debian package identity does not match release contract")
    work.mkdir(parents=True, exist_ok=False)
    extracted = work / "extracted"
    capture(["dpkg-deb", "--extract", package, extracted])  # no install/maintainer script/service
    verified_bundle(gate, extracted / "opt/qlh-edge-inference", "linux", "cpu", contract, expected, work / "deb-verify")
    require(digest(package) == original, "Debian artifact changed during verification")
    copy_file(package, output / package.name)


def stage_android(root, output, expected, gate, contract, work):
    apks = list((root / "android/app/build/outputs/apk/release").glob("*.apk"))
    require(len(apks) == 1 and apks[0].name == "app-release.apk", "Exactly one signed release APK required; unsigned/debug outputs are rejected")
    apk = apks[0]
    before = digest(apk)
    gate.inspect_apk(apk)
    aapt, signer = gate.android_tool("aapt"), gate.android_tool("apksigner")
    require(aapt and signer, "Android static verification tools unavailable")
    metadata = gate.apk_metadata(capture([aapt, "dump", "badging", apk]))
    require(metadata == {"package": "com.qlh.inference", "version": contract["version"]["product"],
                         "version_code": contract["version"]["android_code"]}, "APK metadata does not match release contract")
    pin = os.environ.get("ANDROID_RELEASE_CERT_SHA256", "").replace(":", "").lower()
    require(bool(re.fullmatch(r"[0-9a-f]{64}", pin)), "Release certificate pin required")
    result = capture(gate.android_command(signer, ["verify", "--verbose", "--print-certs", str(apk)]))
    require(gate.signer_digest(result) == pin, "APK signer differs from independently configured release certificate")
    import zipfile
    with zipfile.ZipFile(apk) as archive:
        require(archive.getinfo("assets/build-info.json").file_size <= 65536, "APK build-info too large")
        same_info(json.loads(archive.read("assets/build-info.json")), expected)
    require(digest(apk) == before, "APK changed during static verification")
    copy_file(apk, output / apk.name)


def stage(root, platform, revision, event_sha):
    revision_pair(revision, event_sha)
    expected = expected_info(root, platform, revision)
    work = work_path(root, platform)
    output = work / "stage"
    require(not output.exists(), "Refuse existing artifact staging directory")
    gate = tool(root, "scripts/package_contract/release_artifacts_gate.py")
    contract = gate.read_object(root / "release-contract.json")
    gate.validate_contract(contract)
    output.mkdir()
    {"windows": stage_windows, "linux": stage_linux, "android": stage_android}[platform](root, output, expected, gate, contract, work / "verify")
    same_info(expected_info(root, platform, revision), expected)
    return output


def attachment_refs(value):
    """Read references in structured summaries, keeping raw captures opaque."""
    if isinstance(value, dict):
        if "path" in value and "sha256" in value:
            yield value
        else:
            for item in value.values():
                yield from attachment_refs(item)
    elif isinstance(value, list):
        for item in value:
            yield from attachment_refs(item)


def reference_files(bundle, base, parser):
    """Only the gate's declared attachment closure, never copy the lab directory."""
    refs = []
    for run in bundle["runs"]:
        for key in ("intent", "response", "handoffs", "ui", "topology", "native", "stream", "intent_before"):
            if key in run:
                refs.append(run[key])
        for key in ("topology", "native", "ui"):
            data = parser((base / run[key]["path"]).read_text(encoding="utf-8-sig"))
            refs.extend(attachment_refs(data))
    result = {}
    for ref in refs:
        name, sha = ref["path"], ref["sha256"]
        path = Path(name)
        require(not path.is_absolute() and not PureWindowsPath(name).drive and "\\" not in name
                and ".." not in path.parts and name != "bundle.json", "Unsafe attachment path")
        require(path.suffix.lower() not in {".key", ".pem", ".p12", ".pfx", ".jks", ".keystore"}, "Sensitive attachment extension forbidden")
        source = base / path
        require(source.resolve().is_relative_to(base.resolve()) and source.is_file() and digest(source) == sha,
                "Attachment missing, escaped or changed")
        name = path.as_posix()
        require(name != "bundle.json", "Attachment collides with bundle")
        require(name not in result or result[name] == sha, "Conflicting attachment reference")
        result[name] = sha
    return result


def product(root, revision, event_sha):
    revision_pair(revision, event_sha)
    original_info = observe(root, revision)
    bundle_path = configured_file("QLH_RELEASE_PRODUCT_BUNDLE")
    gate = tool(root, "scripts/product_relay_gate.py")
    require(gate.validate_bundle(bundle_path, revision)["passed"] is True, "Lab bundle failed strict DL/LL product validation")
    raw = bundle_path.read_bytes()
    refs = reference_files(gate.parse_json(raw.decode("utf-8-sig")), bundle_path.parent, gate.parse_json)
    work = work_path(root, "product")
    require(not work.exists(), "Product staging already exists")
    output = work / "stage"
    output.mkdir(parents=True, exist_ok=False)
    for name, sha in refs.items():
        copy_file(bundle_path.parent / name, output / name)
        require(digest(output / name) == sha, "Staged product attachment differs from original reference")
    (output / "bundle.json").write_bytes(raw)
    require(bundle_path.read_bytes() == raw, "Lab bundle changed during staging")
    require(gate.validate_bundle(output / "bundle.json", revision)["passed"] is True, "Staged product bundle failed strict validation")
    same_info(observe(root, revision), original_info)
    return output


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("preflight", "stage", "product"))
    parser.add_argument("--platform", choices=("windows", "linux", "android"))
    parser.add_argument("--root", type=Path, default=ROOT)
    args = parser.parse_args(argv)
    revision = os.environ.get("EXPECTED_REVISION", "")
    event_sha = os.environ.get("GITHUB_SHA", "")
    try:
        if args.command == "product":
            output = product(args.root.resolve(), revision, event_sha)
        else:
            require(args.platform is not None, "--platform required")
            action = preflight if args.command == "preflight" else stage
            output = action(args.root.resolve(), args.platform, revision, event_sha)
        print(json.dumps({"command": args.command, "platform": args.platform,
                          "source_revision": revision, "output": str(output)}))
        return 0
    except (ValueError, OSError, KeyError, TypeError, subprocess.SubprocessError) as exc:
        # BuildError messages are fixed public text (optionally naming an env
        # interface). Never echo external verifier/Git/OS exception messages:
        # they may contain secrets, config values or runner-local paths.
        reason = str(exc) if type(exc) is BuildError else (
            "Source inventory or static artifact verifier failed; external error details suppressed")
        print(json.dumps({"command": args.command, "platform": args.platform,
                          "passed": False, "reason": reason}), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())