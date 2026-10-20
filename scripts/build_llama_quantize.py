#!/usr/bin/env python3
"""Build and package the pinned llama-quantize tool without network access."""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import tempfile
import zipfile
from pathlib import Path
from typing import Any

if __package__:
    from .model_tools.llama_quantize_toolchain import (
        ROOT,
        default_package_root,
        expected_source_revision,
        file_sha256,
        host_target_id,
        llama_cpp_source,
        load_lock,
        verify_managed_package,
    )
    from .model_tools.sync_llama_cpp import SyncError, inspect_sync
else:
    from model_tools.llama_quantize_toolchain import (
        ROOT,
        default_package_root,
        expected_source_revision,
        file_sha256,
        host_target_id,
        llama_cpp_source,
        load_lock,
        verify_managed_package,
    )
    from model_tools.sync_llama_cpp import SyncError, inspect_sync


class BuildError(RuntimeError):
    """Raised when the pinned tool cannot be built or packaged."""


def _run(command: list[str], *, cwd: Path, timeout: float, accepted_codes: set[int] = {0}) -> subprocess.CompletedProcess[str]:
    try:
        result = subprocess.run(
            command,
            cwd=str(cwd),
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=timeout,
            check=False,
            shell=False,
            env={key: value for key, value in os.environ.items() if not key.upper().startswith("GIT_")},
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise BuildError("toolchain command failed to start or timed out") from exc
    if result.returncode not in accepted_codes:
        tail = "\n".join(result.stdout.splitlines()[-20:])
        raise BuildError(f"toolchain command failed with exit code {result.returncode}:\n{tail}")
    return result


def _android_sdk_roots(project_root: Path) -> list[Path]:
    roots: list[Path] = []
    for name in ("ANDROID_HOME", "ANDROID_SDK_ROOT"):
        if os.environ.get(name):
            roots.append(Path(os.environ[name]))
    local_properties = project_root / "android" / "local.properties"
    if local_properties.is_file():
        for line in local_properties.read_text(encoding="utf-8", errors="ignore").splitlines():
            if line.startswith("sdk.dir="):
                value = line.split("=", 1)[1].replace("\\:", ":").replace("\\\\", "\\")
                roots.append(Path(value))
    if os.environ.get("LOCALAPPDATA"):
        roots.append(Path(os.environ["LOCALAPPDATA"]) / "Android" / "Sdk")
    return list(dict.fromkeys(root.expanduser() for root in roots))


def find_cmake(project_root: Path, explicit: Path | None = None) -> tuple[Path, Path | None]:
    candidates: list[Path] = []
    if explicit is not None:
        candidates.append(explicit.expanduser())
    found = shutil.which("cmake")
    if found:
        candidates.append(Path(found))
    sdk_bins: list[Path] = []
    for sdk_root in _android_sdk_roots(project_root):
        cmake_root = sdk_root / "cmake"
        if cmake_root.is_dir():
            versions = sorted((item for item in cmake_root.iterdir() if item.is_dir()), reverse=True)
            sdk_bins.extend(version / "bin" for version in versions)
    executable = "cmake.exe" if os.name == "nt" else "cmake"
    candidates.extend(directory / executable for directory in sdk_bins)
    for candidate in candidates:
        if candidate.is_file():
            # The first CMake on PATH need not be the SDK installation that
            # supplies Ninja. Find the build tool independently of that choice.
            ninja_name = "ninja.exe" if os.name == "nt" else "ninja"
            ninja_candidates = [candidate.parent / ninja_name]
            if found_ninja := shutil.which("ninja"):
                ninja_candidates.append(Path(found_ninja))
            ninja_candidates.extend(directory / ninja_name for directory in sdk_bins)
            ninja = next((item.absolute() for item in ninja_candidates if item.is_file()), None)
            return candidate.absolute(), ninja
    raise BuildError("cmake was not found; pass --cmake or install CMake")


def _compiler_pair(c_compiler: Path | None, cxx_compiler: Path | None) -> tuple[Path | None, Path | None, str]:
    if bool(c_compiler) != bool(cxx_compiler):
        raise BuildError("--c-compiler and --cxx-compiler must be provided together")
    if c_compiler and cxx_compiler:
        family = "msvc" if cxx_compiler.name.lower() in {"cl", "cl.exe"} else "gnu"
        return c_compiler.absolute(), cxx_compiler.absolute(), family
    if os.name == "nt":
        gcc = shutil.which("gcc")
        gxx = shutil.which("g++")
        if gcc and gxx:
            return Path(gcc), Path(gxx), "gnu"
        if shutil.which("cl"):
            return None, None, "msvc"
        raise BuildError("no supported Windows C/C++ compiler was found")
    cc = shutil.which("cc") or shutil.which("gcc")
    cxx = shutil.which("c++") or shutil.which("g++")
    if not cc or not cxx:
        raise BuildError("no supported Linux C/C++ compiler was found")
    return Path(cc), Path(cxx), "gnu"


def _configured_toolchain(build_dir: Path, cc: Path | None, cxx: Path | None,
                          ninja: Path | None, compiler_family: str,
                          cmake_version: str) -> dict[str, Any]:
    """Check CMake's generated compiler facts, not only user-supplied cache values."""
    try:
        lines = (build_dir / "CMakeCache.txt").read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeError) as exc:
        raise BuildError("CMake did not produce a readable toolchain cache") from exc
    cache: dict[str, str] = {}
    for line in lines:
        if line and not line.startswith(("#", "//")) and "=" in line:
            key, value = line.split("=", 1)
            cache[key.split(":", 1)[0]] = value
    generator = cache.get("CMAKE_GENERATOR")
    if not generator or (ninja is not None and generator != "Ninja"):
        raise BuildError("CMake's actual generator differs from the selected toolchain")
    version = re.fullmatch(r"cmake version ([0-9]+\.[0-9]+\.[0-9]+[^\s/\\]*)", cmake_version)
    if version is None:
        raise BuildError("CMake version could not be identified")
    configurations = cache.get("CMAKE_CONFIGURATION_TYPES", "").split(";")
    multi_configuration = any(configurations)
    if (multi_configuration and "Release" not in configurations
            or not multi_configuration and cache.get("CMAKE_BUILD_TYPE") != "Release"):
        raise BuildError("CMake's actual configuration does not support the selected Release build")
    compiler_info = {}
    for language, requested in (("C", cc), ("CXX", cxx)):
        # Visual Studio can leave -D...COMPILER in the cache while using its
        # own compiler. These generated files describe the compiler it uses.
        info_path = build_dir / "CMakeFiles" / version[1] / f"CMake{language}Compiler.cmake"
        try:
            text = info_path.read_text(encoding="utf-8")
        except (OSError, UnicodeError) as exc:
            raise BuildError("CMake did not produce readable compiler identity") from exc
        values = dict(re.findall(r'^set\((CMAKE_[A-Z_]+) "([^"\n]*)"\)$', text, re.MULTILINE))
        actual = values.get(f"CMAKE_{language}_COMPILER")
        identity = values.get(f"CMAKE_{language}_COMPILER_ID")
        release = values.get(f"CMAKE_{language}_COMPILER_VERSION")
        if not actual or not identity or not release:
            raise BuildError("CMake's actual compiler identity is incomplete")
        if requested is not None and Path(actual).resolve() != requested.resolve():
            raise BuildError(f"CMake's actual CMAKE_{language}_COMPILER differs from the selected compiler")
        if (identity == "MSVC") != (compiler_family == "msvc"):
            raise BuildError("CMake's actual compiler family differs from the selected toolchain")
        compiler_info[language] = f"{identity} {release}"
    return {"generator": generator, "compiler": compiler_info["CXX"],
            "multi_configuration": multi_configuration}


def cmake_definitions(target_id: str, compiler_family: str) -> list[str]:
    definitions = [
        "-DCMAKE_BUILD_TYPE=Release",
        "-DBUILD_SHARED_LIBS=OFF",
        "-DLLAMA_BUILD_COMMON=ON",
        "-DLLAMA_BUILD_TOOLS=ON",
        "-DLLAMA_BUILD_TESTS=OFF",
        "-DLLAMA_BUILD_EXAMPLES=OFF",
        "-DLLAMA_BUILD_SERVER=OFF",
        "-DLLAMA_BUILD_APP=OFF",
        "-DLLAMA_BUILD_UI=OFF",
        "-DLLAMA_USE_PREBUILT_UI=OFF",
        "-DLLAMA_OPENSSL=OFF",
        "-DGGML_CUDA=OFF",
        "-DGGML_NATIVE=OFF",
        "-DGGML_OPENMP=OFF",
        "-DGGML_CCACHE=OFF",
        "-DGGML_BACKEND_DL=OFF",
        "-DGGML_LLAMAFILE=OFF",
    ]
    if target_id.startswith("windows-") and compiler_family == "gnu":
        definitions.append("-DCMAKE_EXE_LINKER_FLAGS=-static -static-libgcc -static-libstdc++")
    if target_id.startswith("windows-") and compiler_family == "msvc":
        definitions.extend(["-DCMAKE_POLICY_DEFAULT_CMP0091=NEW", "-DCMAKE_MSVC_RUNTIME_LIBRARY=MultiThreaded"])
    return definitions


def _source_revision(source: Path, project_root: Path) -> str:
    actual_root = _run(["git", "-C", str(source), "rev-parse", "--show-toplevel"], cwd=project_root, timeout=30)
    if Path(actual_root.stdout.strip()).resolve() != source.resolve():
        raise BuildError("llama.cpp source is not its own Git checkout")
    result = _run(["git", "-C", str(source), "rev-parse", "HEAD"], cwd=project_root, timeout=30)
    return result.stdout.strip()


def _source_contract(source: Path, project_root: Path, lock: dict[str, Any]) -> dict[str, Any]:
    revision = _source_revision(source, project_root)
    if revision != expected_source_revision(lock):
        raise BuildError("llama.cpp source HEAD does not match the lock's source revision")
    lock_path = project_root / "scripts/model_tools/llama_quantize.lock.json"
    if load_lock(lock_path) != lock:
        raise BuildError("llama.cpp source lock changed during verification")
    # Validate the selected checkout, including external Android roots, against
    # the shared upstream/tree/patch contract. This gate never applies patches.
    try:
        report = inspect_sync(
            android_root=source.resolve().parents[4], lock_path=lock_path, dry_run=True,
        )
    except SyncError as exc:
        raise BuildError(f"llama.cpp source patch contract failed: {exc}") from exc
    if report["revision"] != revision or load_lock(lock_path) != lock:
        raise BuildError("llama.cpp source lock changed during verification")
    if report["state"] != "applied" or any(
        item["state"] not in {"applied", "applied_committed"} for item in report["patches"]
    ):
        raise BuildError("llama.cpp build requires all managed patches applied; pending patches are not accepted")
    status = _run(["git", "-C", str(source), "status", "--porcelain", "--untracked-files=all"],
                  cwd=project_root, timeout=30).stdout
    # Git status can hide a complete worktree patch via index flags. The
    # shared tree check proves whether each patch is actually present in HEAD.
    dirty = bool(status.strip()) or any(item["state"] == "applied" for item in report["patches"])
    if "revision_local" in lock and dirty:
        raise BuildError("llama.cpp committed-patch source has worktree patches or unrelated changes")
    receipts = [
        {key: item[key] for key in ("id", "patch_sha256", "state")}
        for item in report["patches"]
    ]
    return {"source_revision": revision, "source_dirty": dirty, "source_patches": receipts}


def _version_line(command: list[str], project_root: Path) -> str:
    result = _run(command, cwd=project_root, timeout=30)
    return result.stdout.splitlines()[0].strip() if result.stdout.splitlines() else "unknown"


def _runtime_dependencies(executable: Path, cxx: Path | None, target_id: str, project_root: Path) -> list[str]:
    if target_id.startswith("windows-"):
        objdump = (cxx.parent / "objdump.exe") if cxx is not None else None
        if objdump is None or not objdump.is_file():
            found = shutil.which("objdump")
            objdump = Path(found) if found else None
        if objdump is not None:
            output = _run([str(objdump), "-p", str(executable)], cwd=project_root, timeout=30).stdout
            dependencies = re.findall(r"DLL Name:\s*([^\s]+)", output, flags=re.IGNORECASE)
        else:
            dumpbin = shutil.which("dumpbin")
            if not dumpbin:
                raise BuildError("objdump or dumpbin is required to verify Windows runtime dependencies")
            output = _run([dumpbin, "/dependents", str(executable)], cwd=project_root, timeout=30).stdout
            dependencies = re.findall(r"^\s+([^\s]+\.dll)\s*$", output, flags=re.IGNORECASE | re.MULTILINE)
        windows_root = Path(os.environ.get("WINDIR", "C:/Windows"))
        for dependency in dependencies:
            lowered = dependency.lower()
            is_api_set = lowered.startswith(("api-ms-win-", "ext-ms-win-"))
            is_system = (windows_root / "System32" / dependency).is_file()
            if not is_api_set and not is_system:
                raise BuildError(f"Windows package has a non-system runtime dependency: {dependency}")
        return sorted(set(dependencies), key=str.lower)
    ldd = shutil.which("ldd")
    if not ldd:
        raise BuildError("ldd is required to verify Linux runtime dependencies")
    output = _run([ldd, str(executable)], cwd=project_root, timeout=30).stdout
    if "not found" in output.lower():
        raise BuildError("Linux package has an unresolved runtime dependency")
    dependencies = []
    for line in output.splitlines():
        token = line.strip().split(" ", 1)[0]
        if token and (".so" in token or token.startswith("linux-vdso")):
            dependencies.append(Path(token).name)
    return sorted(set(dependencies))


def _write_deterministic_zip(package_dir: Path, archive: Path) -> str:
    root_name = f"llama-quantize-{package_dir.name}"
    with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as handle:
        for source in sorted(item for item in package_dir.iterdir() if item.is_file()):
            info = zipfile.ZipInfo(f"{root_name}/{source.name}", date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            mode = 0o755 if source.name.startswith("llama-quantize") else 0o644
            info.external_attr = mode << 16
            handle.writestr(info, source.read_bytes())
    digest = file_sha256(archive)
    archive.with_name(archive.name + ".sha256").write_text(f"{digest}  {archive.name}\n", encoding="ascii")
    return digest


def build_and_package(
    *,
    project_root: Path = ROOT,
    output_root: Path | None = None,
    build_dir: Path | None = None,
    cmake: Path | None = None,
    c_compiler: Path | None = None,
    cxx_compiler: Path | None = None,
    jobs: int = 4,
    timeout_seconds: float = 1800,
    replace: bool = False,
) -> dict[str, Any]:
    if not 1 <= jobs <= 64:
        raise BuildError("jobs must be between 1 and 64")
    lock = load_lock(project_root / "scripts/model_tools/llama_quantize.lock.json")
    target_id = host_target_id()
    target = lock["targets"].get(target_id)
    if target is None:
        raise BuildError(f"host target is not locked: {target_id}")
    source = llama_cpp_source(project_root)
    if not source.is_dir() or source.is_symlink():
        raise BuildError("pinned llama.cpp submodule is missing or unsafe")
    source_contract = _source_contract(source, project_root, lock)
    revision = source_contract["source_revision"]

    cmake_path, ninja_path = find_cmake(project_root, cmake)
    cc, cxx, compiler_family = _compiler_pair(c_compiler, cxx_compiler)
    if target_id.startswith("windows-") and compiler_family == "gnu" and ninja_path is None:
        raise BuildError("Ninja is required for Windows GNU builds; install Ninja or an Android SDK CMake toolchain")
    cmake_version = _version_line([str(cmake_path), "--version"], project_root)
    output_root = (output_root or default_package_root(project_root)).absolute()
    build_dir = (build_dir or output_root / "work" / target_id).absolute()
    build_dir.mkdir(parents=True, exist_ok=True)
    configure = [str(cmake_path), "-S", str(source), "-B", str(build_dir)]
    if ninja_path is not None:
        configure.extend(["-G", "Ninja", f"-DCMAKE_MAKE_PROGRAM={ninja_path.as_posix()}"])
    if cc is not None and cxx is not None:
        configure.extend([f"-DCMAKE_C_COMPILER={cc.as_posix()}", f"-DCMAKE_CXX_COMPILER={cxx.as_posix()}"])
    configure.extend(cmake_definitions(target_id, compiler_family))
    _run(configure, cwd=project_root, timeout=timeout_seconds)
    toolchain = _configured_toolchain(build_dir, cc, cxx, ninja_path, compiler_family, cmake_version)
    _run(
        [str(cmake_path), "--build", str(build_dir), "--config", "Release",
         "--target", lock["cmake_target"], "--parallel", str(jobs)],
        cwd=project_root,
        timeout=timeout_seconds,
    )

    executable_name = target["executable"]
    binary_dir = build_dir / "bin"
    if toolchain["multi_configuration"]:
        binary_dir /= "Release"
    executable = binary_dir / executable_name
    if not executable.is_file() or executable.is_symlink():
        raise BuildError("llama-quantize build did not produce the locked executable")
    smoke_result = _run([str(executable), "--help"], cwd=project_root, timeout=30, accepted_codes={0, 1})
    smoke_text = smoke_result.stdout
    if "allowed quantization types" not in smoke_text or "Q4_K_M" not in smoke_text:
        raise BuildError("llama-quantize help smoke did not expose Q4_K_M")
    runtime_dependencies = _runtime_dependencies(executable, cxx, target_id, project_root)

    if _source_contract(source, project_root, lock) != source_contract:
        raise BuildError("llama.cpp source contract changed during the build")

    packages_root = output_root / "packages"
    packages_root.mkdir(parents=True, exist_ok=True)
    destination = packages_root / target_id
    executable_digest = file_sha256(executable)
    if destination.exists():
        existing = verify_managed_package(destination, expected_target=target_id, lock=lock)
        if existing.get("valid") and existing.get("sha256") == executable_digest:
            package_dir = destination
        elif not replace:
            raise BuildError("managed package already exists with different or invalid content")
        else:
            shutil.rmtree(destination)
            package_dir = None
    else:
        package_dir = None
    if package_dir is None:
        with tempfile.TemporaryDirectory(prefix=".llama-quantize-package-", dir=str(packages_root)) as staging_name:
            staging = Path(staging_name)
            packaged_executable = staging / executable_name
            shutil.copy2(executable, packaged_executable)
            if target["platform"] == "linux":
                packaged_executable.chmod(0o755)
            license_file = staging / "LICENSE.llama.cpp"
            shutil.copy2(source / "LICENSE", license_file)
            files = []
            for item in sorted((packaged_executable, license_file), key=lambda path: path.name):
                files.append({"path": item.name, "size_bytes": item.stat().st_size, "sha256": file_sha256(item)})
            manifest = {
                "schema_version": 1,
                "tool": lock["tool"],
                "target_id": target_id,
                "upstream": {"repository": lock["upstream"]["repository"], "revision": lock["upstream"]["revision"]},
                **source_contract,
                "executable": executable_name,
                "build": {
                    "configuration": "Release",
                    **toolchain,
                    "cmake": cmake_version,
                    "shared_libraries": False,
                    "cuda": False,
                    "native_optimizations": False,
                    "openmp": False,
                    "windows_static_runtime": target["platform"] == "windows",
                },
                "smoke": {
                    "help": True,
                    "q4_k_m_listed": True,
                    "runtime_dependencies_verified": True,
                    "runtime_dependencies": runtime_dependencies,
                },
                "files": files,
            }
            (staging / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=True, indent=2) + "\n", encoding="utf-8")
            os.replace(staging, destination)
            package_dir = destination

    verification = verify_managed_package(package_dir, expected_target=target_id, lock=lock)
    if not verification["valid"]:
        raise BuildError("new managed package failed verification")
    archive = packages_root / f"llama-quantize-{revision[:8]}-{target_id}.zip"
    archive_digest = _write_deterministic_zip(package_dir, archive)
    return {
        "schema_version": 1,
        "tool": lock["tool"],
        "valid": True,
        "target_id": target_id,
        "revision": revision,
        "upstream_revision": lock["upstream"]["revision"],
        **source_contract,
        "package": package_dir.name,
        "executable": verification["executable"],
        "executable_size_bytes": verification["size_bytes"],
        "executable_sha256": verification["sha256"],
        "archive": archive.name,
        "archive_sha256": archive_digest,
        "smoke": {
            "help": True,
            "q4_k_m_listed": True,
            "runtime_dependencies_verified": True,
            "runtime_dependencies": runtime_dependencies,
        },
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=None)
    parser.add_argument("--build-dir", type=Path, default=None)
    parser.add_argument("--cmake", type=Path, default=None)
    parser.add_argument("--c-compiler", type=Path, default=None)
    parser.add_argument("--cxx-compiler", type=Path, default=None)
    parser.add_argument("--jobs", type=int, default=min(os.cpu_count() or 4, 8))
    parser.add_argument("--timeout-seconds", type=float, default=1800)
    parser.add_argument("--replace", action="store_true", help="replace an existing invalid or different managed package")
    parser.add_argument("--json", action="store_true", dest="as_json")
    args = parser.parse_args(argv)
    try:
        report = build_and_package(
            output_root=args.output_root,
            build_dir=args.build_dir,
            cmake=args.cmake,
            c_compiler=args.c_compiler,
            cxx_compiler=args.cxx_compiler,
            jobs=args.jobs,
            timeout_seconds=args.timeout_seconds,
            replace=args.replace,
        )
    except (BuildError, ValueError) as exc:
        report = {"tool": "llama-quantize", "valid": False, "error": str(exc)}
    if args.as_json:
        print(json.dumps(report, ensure_ascii=True, indent=2))
    else:
        state = "PACKAGED" if report.get("valid") else "FAILED"
        print(f"{state}: llama-quantize")
        if report.get("valid"):
            print(f"target={report['target_id']} revision={report['revision'][:12]}")
            print(f"executable_sha256={report['executable_sha256']} archive={report['archive']}")
        else:
            print(report.get("error"))
    return 0 if report.get("valid") else 1


if __name__ == "__main__":
    raise SystemExit(main())
