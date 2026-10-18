"""Windows native build contracts using tiny fake tools and local Git fixtures.

Run without optional pytest dependencies:
    python -m unittest discover -s tests -p test_keep_head_cpu_build.py -v
No model, device, signing key, package build, download, or production DLL is used.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import time
import unittest

ROOT = Path(__file__).resolve().parents[1]
BUILDER = ROOT / "scripts/model_tools/build_keep_head_cpu.ps1"
BATCH = ROOT / "packaging/packaging/build-cpu.bat"
DLLS = (
    "qlh_keep_head.dll", "libllama.dll", "ggml.dll", "ggml-base.dll", "ggml-cpu.dll",
    "libgcc_s_seh-1.dll", "libstdc++-6.dll", "libwinpthread-1.dll", "libgomp-1.dll",
)

# One framework executable copied under the four tool names. It logs argv as
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
            return 0;
        }
        if (Array.IndexOf(args, "--build") >= 0) {
            if (Failure("BUILD") != 0) return Failure("BUILD");
            string dir = After(args, "--build");
            foreach (string dll in new string[]{"libllama.dll", "ggml.dll", "ggml-base.dll", "ggml-cpu.dll"})
                if (dll != Environment.GetEnvironmentVariable("QLH_TEST_MISSING_DLL"))
                    Write(Path.Combine(dir, "bin", dll), "fresh fake " + dll);
            Write(Path.Combine(dir, "src", "libllama.dll.a"), "fresh fake import library");
            return 0;
        }
        if (Array.IndexOf(args, "-shared") >= 0) {
            if (Failure("SHIM") != 0) return Failure("SHIM");
            Write(After(args, "-o"), "fresh fake shim");
            return 0;
        }
        return 98;
    }
}
'''


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
        (self.source / "include/llama.h").write_text("/* fake pinned header */\n", encoding="ascii")
        (self.source / "CMakeLists.txt").write_text("# fake pinned source\n", encoding="ascii")
        self.git(self.source, "init", "-q")
        self.git(self.source, "add", ".")
        self.git(self.source, "-c", "user.name=QLH test", "-c", "user.email=qlh-test@example.invalid",
                 "-c", "commit.gpgsign=false", "commit", "-qm", "fixture")
        revision = self.git(self.source, "rev-parse", "HEAD").stdout.strip()
        scripts = self.root / "scripts/model_tools"
        (scripts / "keep_head_shim").mkdir(parents=True)
        # Exercise the existing shim unchanged, including its exit and copy rules.
        shutil.copyfile(ROOT / "scripts/model_tools/build_keep_head_shim.ps1", scripts / "build_keep_head_shim.ps1")
        (scripts / "keep_head_shim/qlh_keep_head.c").write_text("/* fake shim */\n", encoding="ascii")
        patch = scripts / "committed.patch"
        patch.write_bytes(b"patch fixture\r\n")
        lock = {"source": "android/app/src/main/cpp/llama.cpp", "revision_local": revision,
                "revision_local_note": "Windows UTF-8 fixture: \u8865\u4e01",  # PS 5.1 must read UTF-8.
                "upstream": {"revision": "a" * 40},
                "patches": [{"path": "scripts/model_tools/committed.patch",
                             "sha256": hashlib.sha256(b"patch fixture\n").hexdigest()}]}
        (scripts / "llama_quantize.lock.json").write_text(json.dumps(lock, ensure_ascii=False), encoding="utf-8")
        self.tools = self.root / "build/toolchain with spaces"
        self.tools.mkdir(parents=True)
        for name in ("gcc.exe", "g++.exe", "mingw32-make.exe", "cmake.exe"):
            shutil.copyfile(self.fake, self.tools / name)
        for name in DLLS[5:]:
            (self.tools / name).write_bytes(("runtime " + name).encode())
        self.log = self.root / "build/tool-argv.log"
        self.env["QLH_TEST_TOOL_LOG"] = str(self.log)
        self.build = self.root / "build/fresh build"
        self.output = self.root / "build/separate output"
        self.command = ["powershell.exe", "-NoProfile", "-ExecutionPolicy", "Bypass", "-File", str(BUILDER),
                        "-SourceRoot", str(self.root), "-BuildDir", str(self.build),
                        "-OutDir", str(self.output), "-ToolchainBin", str(self.tools)]

    def git(self, directory, *args):
        return subprocess.run(["git.exe", "-C", str(directory), *args], env=self.env,
                              capture_output=True, text=True, errors="replace", check=True, timeout=20)

    def run_builder(self, code=0, **env):
        # Windows can refuse to launch a tool image that a just-written copy — or the
        # security/indexer scan of it — still holds. PowerShell surfaces that either as
        #   "Program 'x.exe' failed to run: The process cannot access the file ...
        #    being used by another process"                        (WinError 32)
        # or as a plain access denial in the middle of the same race             (5).
        # It is a timing race on freshly created images and clears on retry. Only these
        # signatures are retried, so any real contract violation still fails at once.
        racing = ("being used by another process", "Access is denied", "拒绝访问",
                  "[WinError 32]", "[WinError 5]")
        for attempt in range(12):
            result = subprocess.run(self.command, cwd=self.tmp.name, env=dict(self.env, **env),
                                    capture_output=True, text=True, errors="replace", timeout=45)
            output = result.stdout + result.stderr
            if result.returncode == code or not any(token in output for token in racing):
                break
            time.sleep(0.5 * (attempt + 1))
        self.assertEqual(result.returncode, code, result.stdout + result.stderr)
        return result

    def test_fresh_build_spaces_custom_output_and_safe_config(self):
        # Host CFLAGS must not change the portable ISA selection.
        self.run_builder(CFLAGS="-march=native -mavx512f", CXXFLAGS="-m32")
        receipt = json.loads((self.build / "keep-head-cpu-build.json").read_text(encoding="utf-8-sig"))
        self.assertEqual(receipt["exit_code"], 0)
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
                path.unlink()
                try:
                    result = self.run_builder(1)
                    self.assertIn(name, result.stdout)
                    self.assertFalse(self.log.exists())
                    self.assertFalse(self.build.exists())
                finally:
                    path.write_bytes(contents)

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
            (scripts / "build_keep_head_cpu.ps1").write_text("param([string]$SourceRoot)\nexit 47\n", encoding="ascii")
            batch = root / "test.bat"
            batch.write_text("@echo off\nsetlocal enabledelayedexpansion\n" + block + "exit /b 99\n", encoding="ascii")
            env = dict(os.environ, QLH_CORE_ROOT=str(root), QLH_BUILD_PROFILE="llama_cpp_only")
            result = subprocess.run([os.environ.get("COMSPEC", "cmd.exe"), "/d", "/c", str(batch)],
                                    env=env, capture_output=True, text=True, errors="replace", timeout=20)
            self.assertEqual(result.returncode, 47, result.stdout + result.stderr)
            env["QLH_BUILD_PROFILE"] = "torch_cpu"
            result = subprocess.run([os.environ.get("COMSPEC", "cmd.exe"), "/d", "/c", str(batch)],
                                    env=env, capture_output=True, text=True, errors="replace", timeout=20)
            self.assertEqual(result.returncode, 99, result.stdout + result.stderr)


if __name__ == "__main__":
    unittest.main(verbosity=2)
