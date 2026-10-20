"""Windows native build contracts using tiny fake tools and local Git fixtures.

Run without optional pytest dependencies:
    python -m unittest discover -s tests -p test_keep_head_cpu_build.py -v
No model, device, signing key, package build, download, or production DLL is used.
"""
from __future__ import annotations

from contextlib import ExitStack
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import time
import unittest
from types import SimpleNamespace
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
BUILDER = ROOT / "scripts/model_tools/build_keep_head_cpu.ps1"
BATCH = ROOT / "packaging/packaging/build-cpu.bat"
DLLS = (
    "qlh_keep_head.dll", "libllama.dll", "ggml.dll", "ggml-base.dll", "ggml-cpu.dll",
    "libgcc_s_seh-1.dll", "libstdc++-6.dll", "libwinpthread-1.dll", "libgomp-1.dll",
)

# One framework executable linked under the four tool names. It logs argv as
# individual lines so the tests exercise real PowerShell/Windows quoting.
FAKE_TOOL = r'''using System;
using System.IO;
using System.Text;
using System.Collections.Generic;
class NativeTool {
    static int Failure(string stage) {
        string value = Environment.GetEnvironmentVariable("QLH_TEST_FAIL_" + stage);
        return String.IsNullOrEmpty(value) ? 0 : Int32.Parse(value);
    }
    static string After(string[] args, string flag) {
        return args[Array.IndexOf(args, flag) + 1];
    }
    static void Write(string path, string text) {
        Directory.CreateDirectory(Path.GetDirectoryName(path));
        File.WriteAllText(path, text, new UTF8Encoding(false));
    }
    static void Mutate(string stage) {
        if (Environment.GetEnvironmentVariable("QLH_TEST_MUTATE_AT") == stage)
            File.AppendAllText(Environment.GetEnvironmentVariable("QLH_TEST_MUTATE_PATH"),
                               Environment.GetEnvironmentVariable("QLH_TEST_MUTATE_TEXT"), new UTF8Encoding(false));
    }
    static int Main(string[] args) {
        string name = Path.GetFileNameWithoutExtension(Environment.GetCommandLineArgs()[0]);
        string log = Environment.GetEnvironmentVariable("QLH_TEST_TOOL_LOG");
        File.AppendAllText(log, "TOOL=" + name + "\n" + String.Join("\n", args) + "\nEND\n");
        if (Array.IndexOf(args, "--version") >= 0) {
            Console.WriteLine(name + " fake 1.0"); return Failure("VERSION");
        }
        if (Array.IndexOf(args, "-dumpmachine") >= 0) {
            Console.WriteLine(Environment.GetEnvironmentVariable("QLH_TEST_MACHINE") ?? "x86_64-w64-mingw32");
            return 0;
        }
        if (Array.IndexOf(args, "-S") >= 0) {
            if (Failure("CONFIGURE") != 0) return Failure("CONFIGURE");
            string dir = After(args, "-B");
            var cache = new List<string>();
            foreach (string arg in args) {
                if (arg.StartsWith("-D")) {
                    int equal = arg.IndexOf('=');
                    cache.Add(arg.Substring(2, equal - 2) + ":STRING=" + arg.Substring(equal + 1).Replace('\\', '/'));
                }
            }
            cache.Add("CMAKE_GENERATOR:INTERNAL=" + After(args, "-G"));
            cache.Add("CMAKE_HOME_DIRECTORY:INTERNAL=" + After(args, "-S").Replace('\\', '/'));
            if (Environment.GetEnvironmentVariable("QLH_TEST_BAD_CACHE") == "1")
                cache.Add("GGML_NATIVE:BOOL=ON");
            Write(Path.Combine(dir, "CMakeCache.txt"), String.Join("\n", cache));
            Mutate("CONFIGURE");
            return 0;
        }
        if (Array.IndexOf(args, "--build") >= 0) {
            if (Failure("BUILD") != 0) return Failure("BUILD");
            string dir = After(args, "--build");
            foreach (string dll in new string[]{"libllama.dll", "ggml.dll", "ggml-base.dll", "ggml-cpu.dll"})
                if (dll != Environment.GetEnvironmentVariable("QLH_TEST_MISSING_DLL"))
                    Write(Path.Combine(dir, "bin", dll), "fresh fake " + dll);
            Write(Path.Combine(dir, "src", "libllama.dll.a"), "fresh fake import library");
            Mutate("BUILD");
            return 0;
        }
        if (Array.IndexOf(args, "-shared") >= 0) {
            if (Failure("SHIM") != 0) return Failure("SHIM");
            Write(After(args, "-o"), "fresh fake shim");
            Mutate("SHIM");
            return 0;
        }
        return 98;
    }
}
'''


def retry_fixture_sharing(operation):
    """Only retry Windows sharing violations while preparing fake tools."""
    for attempt in range(12):
        try:
            return operation()
        except OSError as error:
            if getattr(error, "winerror", None) != 32 or attempt == 11:
                raise
            time.sleep(0.5 * (attempt + 1))


def exclusive_read(path):
    # A completed managed process can leave a scanner holding the image briefly.
    # Unlike Python's open(), FileShare.None also detects those remaining handles.
    import ctypes
    from ctypes import wintypes

    kernel = ctypes.WinDLL("kernel32", use_last_error=True)
    create = kernel.CreateFileW
    create.argtypes = (wintypes.LPCWSTR, wintypes.DWORD, wintypes.DWORD,
                       wintypes.LPVOID, wintypes.DWORD, wintypes.DWORD, wintypes.HANDLE)
    create.restype = wintypes.HANDLE
    handle = create(str(path), 0x80000000, 0, None, 3, 0, None)  # GENERIC_READ, OPEN_EXISTING
    if handle == ctypes.c_void_p(-1).value:
        raise ctypes.WinError(ctypes.get_last_error())
    close = kernel.CloseHandle
    close.argtypes = (wintypes.HANDLE,)
    close.restype = wintypes.BOOL
    if not close(handle):
        raise ctypes.WinError(ctypes.get_last_error())


def pin_fixture_image(path):
    """Keep a fake image readable/executable while excluding transient writers."""
    import ctypes
    from ctypes import wintypes

    kernel = ctypes.WinDLL("kernel32", use_last_error=True)
    create = kernel.CreateFileW
    create.argtypes = (wintypes.LPCWSTR, wintypes.DWORD, wintypes.DWORD,
                       wintypes.LPVOID, wintypes.DWORD, wintypes.DWORD, wintypes.HANDLE)
    create.restype = wintypes.HANDLE
    # Share reads and deletion, but no write opens between successive tool starts.
    handle = create(str(path), 0x80000000, 0x1 | 0x4, None, 3, 0, None)
    if handle == ctypes.c_void_p(-1).value:
        raise ctypes.WinError(ctypes.get_last_error())
    close = kernel.CloseHandle
    close.argtypes = (wintypes.HANDLE,)
    close.restype = wintypes.BOOL
    def release():
        if not close(handle):
            raise ctypes.WinError(ctypes.get_last_error())
    return release


@unittest.skipUnless(os.name == "nt" and shutil.which("powershell.exe") and shutil.which("git.exe"),
                     "requires Windows PowerShell and Git; no native compiler required")
class KeepHeadCpuBuildTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Build a small managed fake once with Windows' existing framework.
        # `ignore_cleanup_errors`: the tool images below are spawned by the builder,
        # and Windows can still hold them while removing the tree — that is not part
        # of the contract under test, so cleanup must not mask a real failure.
        cls.suite_tmp = tempfile.TemporaryDirectory(
            prefix="qlh keep head tools ", ignore_cleanup_errors=True)
        cls.addClassCleanup(cls.suite_tmp.cleanup)
        cls.fake = Path(cls.suite_tmp.name) / "fake.exe"
        code_path = cls.fake.with_suffix(".cs")
        code_path.write_text(FAKE_TOOL, encoding="utf-8")
        command = ["powershell.exe", "-NoProfile", "-ExecutionPolicy", "Bypass", "-Command",
                   "Add-Type -TypeDefinition ([IO.File]::ReadAllText($env:QLH_TEST_CS)) "
                   "-OutputAssembly $env:QLH_TEST_EXE -OutputType ConsoleApplication"]
        env = dict(os.environ, QLH_TEST_CS=str(code_path), QLH_TEST_EXE=str(cls.fake))
        result = subprocess.run(command, env=env, capture_output=True, text=True, errors="replace", timeout=45)
        if result.returncode:
            raise AssertionError(result.stdout + result.stderr)

    def setUp(self):
        # Same rationale as `suite_tmp`: the builder spawns freshly copied tool images
        # out of this tree, and Windows may still hold them when Python removes it.
        self.tmp = tempfile.TemporaryDirectory(
            prefix="qlh native contract ", ignore_cleanup_errors=True)
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name) / "source with spaces"
        self.root.mkdir()
        self.env = {key: value for key, value in os.environ.items()
                    if not key.upper().startswith(("GIT_", "QLH_KEEP_HEAD_", "QLH_TEST_"))}
        self.env.update(GIT_CONFIG_NOSYSTEM="1", GIT_CONFIG_GLOBAL=os.devnull,
                        GIT_OPTIONAL_LOCKS="0", GIT_TERMINAL_PROMPT="0")
        self.git(self.root, "init", "-q")
        (self.root / ".gitignore").write_text("build/\n", encoding="ascii")
        self.source = self.root / "android/app/src/main/cpp/llama.cpp"
        self.source.mkdir(parents=True)
        (self.source / "include").mkdir()
        self.header = self.source / "include/llama.h"
        self.header.write_bytes(b"/* fake upstream header */\n" + b"/* context */\n" * 20)
        (self.source / "CMakeLists.txt").write_bytes(b"# fake pinned source\n")
        self.git(self.source, "init", "-q")
        for key, value in (("user.name", "QLH test"), ("user.email", "qlh-test@example.invalid"),
                           ("core.autocrlf", "false"), ("commit.gpgsign", "false")):
            self.git(self.source, "config", key, value)
        self.upstream = self.commit_source("upstream fixture")
        scripts = self.root / "scripts/model_tools"
        (scripts / "keep_head_shim").mkdir(parents=True)
        # Execute the real SourceRoot copy: sync derives its ROOT and lock from
        # __file__, so importing or running the original would inspect this repo.
        self.sync = scripts / "sync_llama_cpp.py"
        shutil.copyfile(ROOT / "scripts/model_tools/sync_llama_cpp.py", self.sync)
        shutil.copyfile(ROOT / "scripts/model_tools/build_keep_head_shim.ps1", scripts / "build_keep_head_shim.ps1")
        (scripts / "keep_head_shim/qlh_keep_head.c").write_bytes(b"/* fake shim */\n")
        self.header.write_bytes(b"/* QLH managed API */\n" + self.header.read_bytes())
        self.managed_patch = scripts / "committed.patch"
        self.managed_patch.write_bytes(self.git(self.source, "diff", "--binary").stdout.encode("ascii"))
        self.revision = self.commit_source("managed patch fixture")
        self.lock_path = scripts / "llama_quantize.lock.json"
        self.lock = {"source": "android/app/src/main/cpp/llama.cpp", "revision_local": self.revision,
                     "revision_local_note": "Windows UTF-8 fixture: \u8865\u4e01",
                     "upstream": {"revision": self.upstream},
                     "patches": [{"id": "fixture-api", "path": "scripts/model_tools/committed.patch",
                                  "targets": ["include/llama.h"], "marker": "QLH managed API",
                                  "sha256": hashlib.sha256(self.managed_patch.read_bytes()).hexdigest()}]}
        self.write_lock()
        self.tools = self.root / "build/toolchain with spaces"
        self.tools.mkdir(parents=True)
        # Reuse one compiled image instead of writing four fresh EXEs for every
        # case. Each path still exercises the real tool name and space quoting.
        for name in ("gcc.exe", "g++.exe", "mingw32-make.exe", "cmake.exe"):
            os.link(self.fake, self.tools / name)
        for name in DLLS[5:]:
            (self.tools / name).write_bytes(("runtime " + name).encode())
        self.log = self.root / "build/tool-argv.log"
        self.env["QLH_TEST_TOOL_LOG"] = str(self.log)
        self.probe_log = self.root / "build/tool-readiness.log"
        for name in ("gcc.exe", "g++.exe", "mingw32-make.exe", "cmake.exe"):
            self.ready_fake_tool(self.tools / name)
        self.build = self.root / "build/fresh build"
        self.output = self.root / "build/separate output"
        self.command = ["powershell.exe", "-NoProfile", "-ExecutionPolicy", "Bypass", "-File", str(BUILDER),
                        "-SourceRoot", str(self.root), "-BuildDir", str(self.build),
                        "-OutDir", str(self.output), "-Python", sys.executable, "-ToolchainBin", str(self.tools)]

    def git(self, directory, *args):
        return subprocess.run(["git.exe", "-C", str(directory), *args], env=self.env,
                              capture_output=True, text=True, errors="replace", check=True, timeout=20)

    def commit_source(self, message):
        self.git(self.source, "add", ".")
        self.git(self.source, "commit", "-qm", message)
        return self.git(self.source, "rev-parse", "HEAD").stdout.strip()

    def write_lock(self):
        self.lock_path.write_text(json.dumps(self.lock, ensure_ascii=False), encoding="utf-8")

    def run_sync(self):
        return subprocess.run([sys.executable, "-B", "-X", "utf8", str(self.sync),
                               "--android-root", str(self.root / "android"), "--dry-run"],
                              env=self.env, capture_output=True, text=True, errors="replace", timeout=30)

    def assert_no_build(self):
        self.assertFalse(self.log.exists())
        self.assertFalse(self.build.exists())
        self.assertFalse(self.output.exists())

    def assert_drift_rejected(self, path, stage, detail, *, code=1, addition="\n/* drift */\n"):
        before = path.read_bytes()
        result = self.run_builder(code, QLH_TEST_MUTATE_AT=stage, QLH_TEST_MUTATE_PATH=str(path),
                                  QLH_TEST_MUTATE_TEXT=addition)
        self.assertIn(detail, result.stdout)
        self.assertEqual(path.read_bytes(), before + addition.encode("utf-8"))
        self.assertIn("--build\n", self.log.read_text())
        self.assertTrue((self.build / "bin/libllama.dll").exists())
        self.assertFalse((self.build / "keep-head-cpu-build.json").exists())

    def ready_fake_tool(self, path):
        def probe():
            result = subprocess.run([str(path), "--version"], cwd=self.tmp.name,
                                    env=dict(self.env, QLH_TEST_TOOL_LOG=str(self.probe_log)),
                                    capture_output=True, text=True, errors="replace", timeout=20)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            self.assertEqual(result.stdout.strip(), f"{path.stem} fake 1.0",
                             result.stdout + result.stderr)
        retry_fixture_sharing(probe)
        retry_fixture_sharing(lambda: exclusive_read(path))

    def run_builder(self, code=0, **env):
        # Never replay configuration/build against outputs from a previous launch.
        with ExitStack() as images:
            tools = getattr(self, "tools", None)
            for path in sorted(tools.glob("*.exe")) if tools is not None else ():
                images.callback(retry_fixture_sharing(lambda: pin_fixture_image(path)))
            result = subprocess.run(self.command, cwd=self.tmp.name, env=dict(self.env, **env),
                                    capture_output=True, text=True, errors="replace", timeout=45)
        self.assertEqual(result.returncode, code, result.stdout + result.stderr)
        return result

    def test_fresh_build_spaces_custom_output_and_safe_config(self):
        # Host CFLAGS must not change the portable ISA selection.
        self.run_builder(CFLAGS="-march=native -mavx512f", CXXFLAGS="-m32",
                         QLH_ANDROID_ROOT=str(self.root / "wrong android"),
                         QLH_KEEP_HEAD_PYTHON=str(self.root / "wrong python.exe"))
        receipt = json.loads((self.build / "keep-head-cpu-build.json").read_text(encoding="utf-8-sig"))
        self.assertEqual(receipt["exit_code"], 0)
        sync = receipt["source_sync"]
        self.assertEqual(Path(sync["python"]), Path(sys.executable))
        self.assertEqual(Path(sync["script"]), self.sync)
        self.assertIn("--dry-run", sync["arguments"])
        self.assertNotIn("--apply", sync["arguments"])
        for phase in ("before", "after"):
            self.assertTrue(sync[phase]["ok"])
            self.assertTrue(sync[phase]["dry_run"])
            self.assertEqual(sync[phase]["revision"], self.revision)
            self.assertEqual(sync[phase]["state"], "applied")
            self.assertEqual([item["state"] for item in sync[phase]["patches"]], ["applied_committed"])
        self.assertEqual(sync["script_sha256"], hashlib.sha256(self.sync.read_bytes()).hexdigest())
        for name, digest in receipt["input_sha256"].items():
            self.assertEqual(hashlib.sha256(Path(name).read_bytes()).hexdigest(), digest, name)
        self.assertTrue(receipt["source_git_clean"])
        self.assertEqual({p.name for p in self.output.iterdir()}, set(DLLS))
        self.assertEqual(len(receipt["artifacts"]), 9)
        argv = self.log.read_text()
        self.assertIn(str(self.source), argv)
        self.assertIn(str(self.output / "qlh_keep_head.dll"), argv)
        for flag in ("-DGGML_NATIVE=OFF", "-DGGML_AVX2=ON", "-DGGML_AVX512=OFF", "-DGGML_CUDA=OFF",
                     "-DGGML_VULKAN=OFF", "-DGGML_SYCL=OFF", "-DLLAMA_BUILD_TOOLS=OFF",
                     "-DLLAMA_BUILD_UI=OFF", "-DLLAMA_BUILD_SERVER=OFF", "-DCMAKE_CXX_FLAGS=",
                     "-DFETCHCONTENT_FULLY_DISCONNECTED=ON"):
            self.assertIn(flag + "\n", argv)
        self.assertNotIn("-m32", argv)
        self.assertEqual(self.git(self.source, "status", "--porcelain").stdout, "")
        for name in DLLS[5:]:
            self.assertEqual((self.output / name).read_bytes(), (self.tools / name).read_bytes())

    def test_configure_failure_preserves_code_and_never_builds(self):
        self.run_builder(23, QLH_TEST_FAIL_CONFIGURE="23")
        self.assertNotIn("--build\n", self.log.read_text())
        self.assertFalse(self.output.exists())

    def test_compile_failure_preserves_code_and_never_runs_shim(self):
        self.run_builder(37, QLH_TEST_FAIL_BUILD="37")
        self.assertNotIn("-shared\n", self.log.read_text())
        self.assertFalse(self.output.exists())

    def test_shim_gcc_failure_preserves_code_and_writes_no_success_receipt(self):
        self.run_builder(41, QLH_TEST_FAIL_SHIM="41")
        self.assertIn("-shared\n", self.log.read_text())
        self.assertFalse((self.build / "keep-head-cpu-build.json").exists())

    def test_missing_each_tool_fails_before_any_tool_or_output(self):
        for name in ("cmake.exe", "gcc.exe", "g++.exe", "mingw32-make.exe"):
            with self.subTest(name=name):
                path = self.tools / name
                contents = path.read_bytes()
                retry_fixture_sharing(path.unlink)
                try:
                    result = self.run_builder(1)
                    self.assertIn(name, result.stdout)
                    self.assertFalse(self.log.exists())
                    self.assertFalse(self.build.exists())
                finally:
                    retry_fixture_sharing(lambda: path.write_bytes(contents))
                    self.ready_fake_tool(path)

    def test_missing_runtime_fails_before_configuration(self):
        (self.tools / "libgomp-1.dll").unlink()
        self.assertIn("libgomp-1.dll", self.run_builder(1).stdout)
        self.assertFalse(self.log.exists())
        self.assertFalse(self.build.exists())

    def test_existing_cache_is_rejected_without_changing_bytes(self):
        self.build.mkdir()
        cache = self.build / "CMakeCache.txt"
        cache.write_bytes(b"CMAKE_GENERATOR:INTERNAL=Ninja\nGGML_NATIVE:BOOL=ON\n")
        self.assertIn("non-empty", self.run_builder(1).stdout)
        self.assertEqual(cache.read_bytes(), b"CMAKE_GENERATOR:INTERNAL=Ninja\nGGML_NATIVE:BOOL=ON\n")
        self.assertFalse(self.log.exists())

    def test_existing_output_is_rejected_without_overwriting(self):
        self.output.mkdir()
        sentinel = self.output / "qlh_keep_head.dll"
        sentinel.write_bytes(b"existing group")
        self.assertIn("non-empty", self.run_builder(1).stdout)
        self.assertEqual(sentinel.read_bytes(), b"existing group")
        self.assertFalse(self.build.exists())

    def test_dirty_vendor_is_rejected_without_repair(self):
        header = self.source / "include/llama.h"
        header.write_bytes(b"modified header\n")
        self.assertIn("Git-clean", self.run_builder(1).stdout)
        self.assertEqual(header.read_bytes(), b"modified header\n")
        self.assertFalse(self.build.exists())

    def test_unmanaged_committed_change_is_rejected_by_shared_sync(self):
        cmake = self.source / "CMakeLists.txt"
        cmake.write_bytes(cmake.read_bytes() + b"# unmanaged committed change\n")
        self.lock["revision_local"] = self.commit_source("unmanaged fixture")
        self.write_lock()
        before = cmake.read_bytes()
        index = (self.source / ".git/index").read_bytes()
        objects = self.git(self.source, "count-objects", "-v").stdout
        self.assertEqual(self.git(self.source, "status", "--porcelain").stdout, "")

        self.assertIn("unmanaged", self.run_builder(2).stdout)

        self.assertEqual(cmake.read_bytes(), before)
        self.assertEqual((self.source / ".git/index").read_bytes(), index)
        self.assertEqual(self.git(self.source, "count-objects", "-v").stdout, objects)
        self.assert_no_build()

    def test_fake_upstream_is_rejected_before_any_output(self):
        for upstream in ("a" * 40, self.revision):
            with self.subTest(upstream=upstream):
                self.lock["upstream"]["revision"] = upstream
                self.write_lock()
                result = self.run_builder(2)
                self.assertIn("upstream" if upstream == "a" * 40 else "unmanaged", result.stdout)
                self.assert_no_build()

    def test_pending_patch_sync_exit_zero_is_rejected_without_applying(self):
        self.git(self.source, "checkout", "-q", "--detach", self.upstream)
        self.lock["revision_local"] = self.upstream
        self.write_lock()
        before = self.header.read_bytes()
        sync = self.run_sync()
        self.assertEqual(sync.returncode, 0, sync.stdout + sync.stderr)
        report = json.loads(sync.stdout)
        self.assertEqual(report["state"], "ready_to_apply")
        self.assertEqual(report["pending"], ["fixture-api"])

        result = self.run_builder(1)

        self.assertIn("pending or uncommitted", result.stdout)
        self.assertIn("fixture-api=ready_to_apply", result.stdout)
        self.assertEqual(self.header.read_bytes(), before)
        self.assertEqual(self.git(self.source, "status", "--porcelain").stdout, "")
        self.assert_no_build()

    def test_hidden_complete_worktree_patch_is_rejected_before_any_output(self):
        # HEAD and both pins identify unpatched upstream; the complete patch
        # exists only in the worktree, hidden from porcelain by index flags.
        self.git(self.source, "checkout", "-q", "--detach", self.upstream)
        self.lock["revision_local"] = self.upstream
        self.write_lock()
        self.git(self.source, "apply", str(self.managed_patch))
        before = self.header.read_bytes()
        self.assertEqual(self.git(self.source, "rev-parse", "HEAD").stdout.strip(), self.upstream)
        self.assertNotEqual(self.git(self.source, "show", "HEAD:include/llama.h").stdout.encode("ascii"), before)

        for flag, undo in (("--skip-worktree", "--no-skip-worktree"),
                           ("--assume-unchanged", "--no-assume-unchanged")):
            with self.subTest(flag=flag):
                self.git(self.source, "update-index", flag, "--", "include/llama.h")
                try:
                    self.assertEqual(self.git(self.source, "status", "--porcelain", "--untracked-files=all").stdout, "")
                    index = (self.source / ".git/index").read_bytes()

                    result = self.run_builder(1)

                    self.assertIn("pending or uncommitted", result.stdout)
                    self.assertIn("fixture-api=applied", result.stdout)
                    self.assert_no_build()
                    self.assertEqual(self.header.read_bytes(), before)
                    self.assertEqual((self.source / ".git/index").read_bytes(), index)
                    self.assertEqual(self.git(self.source, "rev-parse", "HEAD").stdout.strip(), self.upstream)
                finally:
                    self.git(self.source, "update-index", undo, "--", "include/llama.h")

    def test_hidden_worktree_edit_is_rejected_by_shared_sync(self):
        cmake = self.source / "CMakeLists.txt"
        self.git(self.source, "update-index", "--skip-worktree", "CMakeLists.txt")
        cmake.write_bytes(cmake.read_bytes() + b"# hidden edit\n")
        self.assertEqual(self.git(self.source, "status", "--porcelain").stdout, "")
        before = cmake.read_bytes()

        self.assertIn("unmanaged", self.run_builder(2).stdout)

        self.assertEqual(cmake.read_bytes(), before)
        self.assert_no_build()

    def test_source_drift_during_configure_never_writes_success_receipt(self):
        self.assert_drift_rejected(self.source / "CMakeLists.txt", "CONFIGURE", "Git-clean")

    def test_hidden_source_drift_is_rechecked_by_shared_sync_before_success(self):
        self.git(self.source, "update-index", "--skip-worktree", "CMakeLists.txt")
        self.assert_drift_rejected(self.source / "CMakeLists.txt", "BUILD", "unmanaged", code=2)
        self.assertEqual(self.git(self.source, "status", "--porcelain").stdout, "")

    def test_lock_drift_with_identical_contract_is_rejected(self):
        # Still valid JSON and the same pins: re-running sync alone cannot bind
        # the receipt to the lock bytes used at build entry.
        self.assert_drift_rejected(self.lock_path, "BUILD", "Build input changed", addition="\n \n")

    def test_shim_source_drift_during_build_is_rejected(self):
        self.assert_drift_rejected(self.root / "scripts/model_tools/keep_head_shim/qlh_keep_head.c",
                                   "BUILD", "Build input changed")

    def test_shim_script_drift_during_shim_compile_is_rejected(self):
        self.assert_drift_rejected(self.root / "scripts/model_tools/build_keep_head_shim.ps1",
                                   "SHIM", "Build input changed", addition="\n# drift\n")

    def test_python_environment_override_without_explicit_parameter(self):
        index = self.command.index("-Python")
        del self.command[index:index + 2]
        self.run_builder(QLH_KEEP_HEAD_PYTHON=sys.executable)
        receipt = json.loads((self.build / "keep-head-cpu-build.json").read_text(encoding="utf-8-sig"))
        self.assertEqual(Path(receipt["source_sync"]["python"]), Path(sys.executable))

    def test_missing_explicit_python_does_not_fall_back_or_create_output(self):
        self.command[self.command.index("-Python") + 1] = str(self.root / "missing python.exe")
        self.assertIn("missing python.exe", self.run_builder(1, QLH_KEEP_HEAD_PYTHON=sys.executable).stdout)
        self.assert_no_build()

    def test_missing_build_dll_never_runs_shim_or_writes_receipt(self):
        self.assertIn("ggml-cpu.dll", self.run_builder(1, QLH_TEST_MISSING_DLL="ggml-cpu.dll").stdout)
        self.assertNotIn("-shared\n", self.log.read_text())
        self.assertFalse((self.build / "keep-head-cpu-build.json").exists())

    def test_unsafe_generated_cache_is_rejected_before_compile(self):
        self.assertIn("GGML_NATIVE", self.run_builder(1, QLH_TEST_BAD_CACHE="1").stdout)
        self.assertNotIn("--build\n", self.log.read_text())

    def test_x86_compiler_is_rejected(self):
        self.assertIn("x64 MinGW", self.run_builder(1, QLH_TEST_MACHINE="i686-w64-mingw32").stdout)
        self.assertFalse(self.build.exists())

    def test_toolchain_and_cmake_environment_overrides(self):
        self.command = self.command[:-2]
        self.run_builder(QLH_KEEP_HEAD_TOOLCHAIN_BIN=str(self.tools),
                         QLH_KEEP_HEAD_CMAKE=str(self.tools / "cmake.exe"))
        self.assertEqual({p.name for p in self.output.iterdir()}, set(DLLS))

    def test_non_ignored_output_is_rejected(self):
        self.build = self.root / "tracked source build"
        index = self.command.index("-BuildDir") + 1
        self.command[index] = str(self.build)
        self.assertIn("Git-ignored", self.run_builder(1).stdout)
        self.assertFalse(self.build.exists())


class CpuBatchNativeGateTests(unittest.TestCase):
    def test_formal_gate_precedes_pyinstaller_and_keeps_signing_gate(self):
        text = BATCH.read_text(encoding="utf-8")
        native = text.index("build_keep_head_cpu.ps1")
        self.assertLess(text.index("if not defined QLH_SIGNING_KEY"), native)
        self.assertLess(native, text.index('"%PYTHON%" -m PyInstaller'))
        self.assertIn('if /i "!QLH_BUILD_PROFILE!"=="llama_cpp_only" (', text[:native])
        self.assertIn("if errorlevel 1 exit /b !errorlevel!", text[native:text.index("REM ---- PyInstaller")])

    @unittest.skipUnless(os.name == "nt", "requires cmd.exe")
    def test_cmd_propagates_exact_builder_error_with_space_path(self):
        # Execute the exact integration block in isolation: formal packaging has
        # existing environment/signing gates and cannot be invoked as a unit test.
        text = BATCH.read_text(encoding="utf-8")
        start = text.index('if /i "!QLH_BUILD_PROFILE!"=="llama_cpp_only" (', text.index("REM Build the pinned"))
        end = text.index("REM ---- PyInstaller", start)
        block = text[start:end]
        with tempfile.TemporaryDirectory(prefix="qlh batch contract ") as directory:
            root = Path(directory)
            scripts = root / "scripts/model_tools"
            scripts.mkdir(parents=True)
            (scripts / "build_keep_head_cpu.ps1").write_text(
                "param([string]$SourceRoot, [string]$Python)\n"
                "if ($Python -cne $env:QLH_TEST_SELECTED_PYTHON) { exit 46 }\nexit 47\n", encoding="ascii")
            batch = root / "test.bat"
            batch.write_text("@echo off\nsetlocal enabledelayedexpansion\n" + block + "exit /b 99\n", encoding="ascii")
            selected_python = str(root / "selected python/python.exe")
            env = dict(os.environ, QLH_CORE_ROOT=str(root), QLH_BUILD_PROFILE="llama_cpp_only",
                       PYTHON=selected_python, QLH_TEST_SELECTED_PYTHON=selected_python,
                       QLH_KEEP_HEAD_PYTHON="must not override packaging Python")
            result = subprocess.run([os.environ.get("COMSPEC", "cmd.exe"), "/d", "/c", str(batch)],
                                    env=env, capture_output=True, text=True, errors="replace", timeout=20)
            self.assertEqual(result.returncode, 47, result.stdout + result.stderr)
            env["QLH_BUILD_PROFILE"] = "torch_cpu"
            result = subprocess.run([os.environ.get("COMSPEC", "cmd.exe"), "/d", "/c", str(batch)],
                                    env=env, capture_output=True, text=True, errors="replace", timeout=20)
            self.assertEqual(result.returncode, 99, result.stdout + result.stderr)


class FakeToolReadinessTests(unittest.TestCase):
    """Fixture retries must never turn a real builder failure into a rerun."""

    def setUp(self):
        self.builder = KeepHeadCpuBuildTests("test_fresh_build_spaces_custom_output_and_safe_config")
        self.builder.command = ["unused fixture builder"]
        self.builder.tmp = SimpleNamespace(name="unused fixture directory")
        self.builder.env = {"QLH_TEST_TOOL_LOG": "formal.log"}
        self.builder.probe_log = Path("readiness.log")
        self.process = self.enterContext(patch.object(subprocess, "run"))
        self.sleep = self.enterContext(patch.object(time, "sleep"))
        self.exclusive = self.enterContext(patch(f"{__name__}.exclusive_read"))

    @staticmethod
    def result(code=0, output="gcc fake 1.0\n"):
        return subprocess.CompletedProcess([], code, output, "")

    @staticmethod
    def windows_error(code):
        error = OSError(f"fixture WinError {code}")
        error.winerror = code
        return error

    def test_probe_retries_only_launch_sharing_and_keeps_separate_log(self):
        self.process.side_effect = [self.windows_error(32), self.result()]

        self.builder.ready_fake_tool(Path("gcc.exe"))

        self.assertEqual(self.process.call_count, 2)
        self.sleep.assert_called_once_with(0.5)
        self.exclusive.assert_called_once_with(Path("gcc.exe"))
        for call in self.process.call_args_list:
            self.assertEqual(call.args[0], ["gcc.exe", "--version"])
            self.assertEqual(call.kwargs["env"]["QLH_TEST_TOOL_LOG"], "readiness.log")
        self.assertEqual(self.builder.env["QLH_TEST_TOOL_LOG"], "formal.log")

    def test_exclusive_read_waits_after_successful_probe_without_relaunch(self):
        self.process.return_value = self.result()
        self.exclusive.side_effect = [self.windows_error(32), self.windows_error(32), None]

        self.builder.ready_fake_tool(Path("gcc.exe"))

        self.process.assert_called_once()
        self.assertEqual(self.exclusive.call_count, 3)
        self.assertEqual([call.args for call in self.sleep.call_args_list], [(0.5,), (1.0,)])

    def test_probe_access_denied_and_missing_file_propagate_without_retry(self):
        for code in (5, 2):
            with self.subTest(winerror=code):
                error = self.windows_error(code)
                self.process.side_effect = error
                with self.assertRaises(OSError) as caught:
                    self.builder.ready_fake_tool(Path("gcc.exe"))
                self.assertIs(caught.exception, error)
        self.assertEqual(self.process.call_count, 2)
        self.sleep.assert_not_called()
        self.exclusive.assert_not_called()

    def test_nonzero_or_empty_probe_output_fails_without_retry(self):
        for result in (self.result(1, "being used by another process"), self.result(0, "")):
            with self.subTest(returncode=result.returncode):
                self.process.return_value = result
                with self.assertRaises(AssertionError):
                    self.builder.ready_fake_tool(Path("gcc.exe"))
        self.assertEqual(self.process.call_count, 2)
        self.sleep.assert_not_called()
        self.exclusive.assert_not_called()

    def test_file_delete_and_restore_retry_only_sharing(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "fake.exe"
            path.write_bytes(b"fixture")
            for operation in (path.unlink, lambda: path.write_bytes(b"restored")):
                calls = []
                def locked_once():
                    calls.append(1)
                    if len(calls) == 1:
                        raise self.windows_error(32)
                    return operation()
                retry_fixture_sharing(locked_once)
                self.assertEqual(len(calls), 2)
            self.assertEqual(path.read_bytes(), b"restored")
        self.assertEqual([call.args for call in self.sleep.call_args_list], [(0.5,), (0.5,)])

    def test_file_operation_nonsharing_error_is_never_retried(self):
        for code in (5, 2, None):
            with self.subTest(winerror=code):
                error = self.windows_error(code)
                with patch.object(Path, "unlink", side_effect=error) as unlink:
                    with self.assertRaises(OSError) as caught:
                        retry_fixture_sharing(Path("fake.exe").unlink)
                    self.assertIs(caught.exception, error)
                    unlink.assert_called_once()
        self.sleep.assert_not_called()

    def test_sharing_exhaustion_is_bounded_and_preserves_original_error(self):
        self.process.return_value = self.result()
        error = self.windows_error(32)
        self.exclusive.side_effect = error

        with self.assertRaises(OSError) as caught:
            self.builder.ready_fake_tool(Path("gcc.exe"))

        self.assertIs(caught.exception, error)
        self.process.assert_called_once()
        self.assertEqual(self.exclusive.call_count, 12)
        self.assertEqual([call.args for call in self.sleep.call_args_list],
                         [(0.5 * attempt,) for attempt in range(1, 12)])

    def test_builder_failures_are_never_replayed_or_replaced(self):
        outputs = (
            "[FAIL] Unexpected CMake cache value for GGML_NATIVE\n",
            "[FAIL] Expected x64 MinGW compiler, got \n",
            "[FAIL] Program 'gcc.exe' failed to run: Access is denied [WinError 5]\n",
            "Program 'gcc.exe' failed to run: being used by another process\n",
        )
        for output in outputs:
            with self.subTest(output=output):
                expected = self.result(1, output)
                self.process.return_value = expected
                self.assertIs(self.builder.run_builder(1), expected)
                with self.assertRaises(AssertionError):
                    self.builder.run_builder(23)
        self.assertEqual(self.process.call_count, 2 * len(outputs))
        self.sleep.assert_not_called()
        self.exclusive.assert_not_called()


if __name__ == "__main__":
    unittest.main(verbosity=2)
