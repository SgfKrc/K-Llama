# Build the committed, pinned Windows x64 CPU keep-head closure without downloads.
# Relative paths are resolved from SourceRoot. All output directories must be fresh
# (or empty), and outputs inside SourceRoot must be Git-ignored. Never clean them.
# ToolchainBin defaults to C:\msys64\ucrt64\bin; CMake defaults to its cmake.exe.
# Override via parameters or QLH_KEEP_HEAD_TOOLCHAIN_BIN / QLH_KEEP_HEAD_CMAKE.
# Example: -BuildDir 'build/native trial' -OutDir 'build/native trial/output'
# ASCII only for Windows PowerShell 5.1.
[CmdletBinding()]
param(
    [string]$SourceRoot = '',
    [string]$BuildDir = 'build/keephead/build-cpu',
    [string]$OutDir = '',
    [string]$ToolchainBin = $env:QLH_KEEP_HEAD_TOOLCHAIN_BIN,
    [string]$CMake = $env:QLH_KEEP_HEAD_CMAKE,
    [ValidateRange(1, 128)][int]$Jobs = 8
)

$ErrorActionPreference = 'Stop'
Set-StrictMode -Version Latest
$originalPath = $env:PATH
$originalOptionalLocks = $env:GIT_OPTIONAL_LOCKS

function Full-Path([string]$Path, [string]$Base) {
    if (-not [IO.Path]::IsPathRooted($Path)) { $Path = Join-Path $Base $Path }
    return [IO.Path]::GetFullPath($Path).TrimEnd('\', '/')
}

function Is-Within([string]$Path, [string]$Parent) {
    return $Path.Equals($Parent, [StringComparison]::OrdinalIgnoreCase) -or
        $Path.StartsWith($Parent + [IO.Path]::DirectorySeparatorChar, [StringComparison]::OrdinalIgnoreCase)
}

function Require-File([string]$Path) {
    if (-not (Test-Path -LiteralPath $Path -PathType Leaf)) { throw "Missing required file: $Path" }
}

function Invoke-Checked([string]$Program, [string[]]$Arguments) {
    Write-Host ('[run] "{0}" {1}' -f $Program, (($Arguments | ForEach-Object { '"' + $_ + '"' }) -join ' '))
    & $Program @Arguments
    $code = $LASTEXITCODE
    if ($code -ne 0) {
        Write-Host "[FAIL] $Program exit=$code"
        exit $code
    }
}

function Git-Output([string]$Directory, [string[]]$Arguments) {
    return @(Invoke-Checked $git (@('-C', $Directory) + $Arguments))
}

function Assert-CleanSource {
    $status = @(Git-Output $source @('status', '--porcelain', '--untracked-files=all', '--ignore-submodules=none'))
    if ($status.Count -ne 0) { throw "llama.cpp source must be Git-clean: $source" }
    $head = (Git-Output $source @('rev-parse', 'HEAD')) -join ''
    if ($head -cne $revision) { throw "llama.cpp HEAD must match pinned revision $revision (got $head)" }
}

function Assert-FreshOutput([string]$Path) {
    if (Is-Within $SourceRoot $Path) { throw "Output cannot contain the source root: $Path" }
    # Reject junctions/symlinks, including existing ancestors of a new directory.
    $cursor = $Path
    while ($cursor) {
        if (Test-Path -LiteralPath $cursor) {
            $item = Get-Item -LiteralPath $cursor -Force
            if ($item.Attributes -band [IO.FileAttributes]::ReparsePoint) {
                throw "Output cannot traverse a reparse point: $cursor"
            }
        }
        $cursor = Split-Path -Parent $cursor
    }
    if (Test-Path -LiteralPath $Path) {
        if (-not (Test-Path -LiteralPath $Path -PathType Container) -or
            @(Get-ChildItem -LiteralPath $Path -Force).Count -ne 0) {
            throw "Refusing non-empty output (including existing CMake caches); choose a fresh directory: $Path"
        }
    }
    if (Is-Within $Path $SourceRoot) {
        & $git -C $SourceRoot check-ignore --quiet -- ($Path + [IO.Path]::DirectorySeparatorChar)
        if ($LASTEXITCODE -ne 0) { throw "Output inside SourceRoot must be Git-ignored: $Path" }
    }
}

# Patch pins describe Git's LF bytes; a clean Windows checkout can use CRLF.
# Record both physical bytes and normalized pin bytes, without rewriting input.
function Hash-Patch([string]$Path) {
    $bytes = [IO.File]::ReadAllBytes($Path)
    $text = [Text.Encoding]::UTF8.GetString($bytes).Replace("`r`n", "`n")
    $algorithm = [Security.Cryptography.SHA256]::Create()
    try {
        return [BitConverter]::ToString($algorithm.ComputeHash([Text.Encoding]::UTF8.GetBytes($text))).Replace('-', '').ToLowerInvariant()
    } finally { $algorithm.Dispose() }
}

function Hash-File([string]$Path) {
    $stream = [IO.File]::OpenRead($Path)
    $algorithm = [Security.Cryptography.SHA256]::Create()
    try { return [BitConverter]::ToString($algorithm.ComputeHash($stream)).Replace('-', '').ToLowerInvariant() }
    finally { $stream.Dispose(); $algorithm.Dispose() }
}

try {
    $scriptRepo = Full-Path '..\..' $PSScriptRoot
    if (-not $SourceRoot) { $SourceRoot = $scriptRepo }
    $SourceRoot = Full-Path $SourceRoot $scriptRepo
    $source = Join-Path $SourceRoot 'android/app/src/main/cpp/llama.cpp'
    $lockPath = Join-Path $SourceRoot 'scripts/model_tools/llama_quantize.lock.json'
    $shimScript = Join-Path $SourceRoot 'scripts/model_tools/build_keep_head_shim.ps1'
    $shimSource = Join-Path $SourceRoot 'scripts/model_tools/keep_head_shim/qlh_keep_head.c'
    foreach ($path in @($lockPath, $shimScript, $shimSource, (Join-Path $source 'CMakeLists.txt'),
                        (Join-Path $source 'include/llama.h'))) { Require-File $path }
    $lock = Get-Content -LiteralPath $lockPath -Raw -Encoding UTF8 | ConvertFrom-Json
    if ($lock.source -cne 'android/app/src/main/cpp/llama.cpp') { throw 'Unexpected locked source path' }
    # Only committed patches are accepted. This builder never applies patches.
    if (-not $lock.PSObject.Properties['revision_local'] -or
        $lock.revision_local -cnotmatch '^[0-9a-f]{40}$') { throw 'A committed patched revision_local pin is required' }
    $revision = $lock.revision_local
    $git = (Get-Command git.exe -CommandType Application -ErrorAction Stop).Source
    $env:GIT_OPTIONAL_LOCKS = '0'
    $actualSource = (Git-Output $source @('rev-parse', '--show-toplevel')) -join ''
    if ((Full-Path $actualSource $SourceRoot) -ine $source) { throw 'llama.cpp must be its own Git checkout' }
    Assert-CleanSource
    $tree = (Git-Output $source @('rev-parse', 'HEAD^{tree}')) -join ''
    $patches = @(
        foreach ($patch in $lock.patches) {
            $patchPath = Full-Path $patch.path $SourceRoot
            if (-not (Is-Within $patchPath $SourceRoot)) { throw 'Patch pin escapes SourceRoot' }
            Require-File $patchPath
            $hash = Hash-Patch $patchPath
            if ($hash -ine $patch.sha256) { throw "Patch SHA256 differs from lock: $patchPath" }
            [ordered]@{ path = $patch.path; sha256 = $hash; checkout_sha256 = (Hash-File $patchPath); state = 'committed' }
        }
    )

    $BuildDir = Full-Path $BuildDir $SourceRoot
    if (-not $OutDir) { $OutDir = Join-Path $BuildDir 'bin' }
    $OutDir = Full-Path $OutDir $SourceRoot
    Assert-FreshOutput $BuildDir
    Assert-FreshOutput $OutDir
    if ((Is-Within $BuildDir $OutDir) -and $BuildDir -ine $OutDir) {
        throw 'OutDir cannot be an ancestor of BuildDir'
    }

    if (-not $ToolchainBin) { $ToolchainBin = 'C:\msys64\ucrt64\bin' }
    $ToolchainBin = Full-Path $ToolchainBin $SourceRoot
    if (-not $CMake) { $CMake = Join-Path $ToolchainBin 'cmake.exe' }
    $CMake = Full-Path $CMake $SourceRoot
    $gcc = Join-Path $ToolchainBin 'gcc.exe'
    $gxx = Join-Path $ToolchainBin 'g++.exe'
    $make = Join-Path $ToolchainBin 'mingw32-make.exe'
    $runtimes = @('libgcc_s_seh-1.dll', 'libstdc++-6.dll', 'libwinpthread-1.dll', 'libgomp-1.dll')
    foreach ($tool in @($CMake, $gcc, $gxx, $make)) { Require-File $tool }
    foreach ($name in $runtimes) { Require-File (Join-Path $ToolchainBin $name) }
    # CMake/make/compiler subprocesses must use this runtime group, even if PATH
    # contains another MinGW installation. Empty flags below override CFLAGS etc.
    $env:PATH = $ToolchainBin + ';' + (Split-Path -Parent $CMake) + ';' + $originalPath
    foreach ($compiler in @($gcc, $gxx)) {
        $target = @(Invoke-Checked $compiler @('-dumpmachine')) -join ''
        if ($target.Trim() -cne 'x86_64-w64-mingw32') { throw "Expected x64 MinGW compiler, got $target" }
    }
    $tools = [ordered]@{}
    foreach ($entry in @(@('cmake', $CMake), @('gcc', $gcc), @('g++', $gxx), @('make', $make))) {
        $version = @(Invoke-Checked $entry[1] @('--version'))
        if ($version.Count -eq 0) { throw "Tool returned no version: $($entry[1])" }
        Write-Host "[tool] $($version[0])"
        $tools[$entry[0]] = [ordered]@{ path = $entry[1]; version = $version[0]; sha256 = (Hash-File $entry[1]) }
    }

    $definitions = [ordered]@{
        CMAKE_BUILD_TYPE = 'Release'; BUILD_SHARED_LIBS = 'ON'
        CMAKE_C_COMPILER = $gcc; CMAKE_CXX_COMPILER = $gxx; CMAKE_MAKE_PROGRAM = $make
        CMAKE_TOOLCHAIN_FILE = ''; CMAKE_PROJECT_TOP_LEVEL_INCLUDES = ''
        CMAKE_C_FLAGS = ''; CMAKE_CXX_FLAGS = ''
        CMAKE_C_FLAGS_RELEASE = '-O3 -DNDEBUG'; CMAKE_CXX_FLAGS_RELEASE = '-O3 -DNDEBUG'
        CMAKE_EXE_LINKER_FLAGS = ''; CMAKE_SHARED_LINKER_FLAGS = ''; CMAKE_MODULE_LINKER_FLAGS = ''
        LLAMA_USE_SYSTEM_GGML = 'OFF'; LLAMA_BUILD_COMMON = 'OFF'; LLAMA_BUILD_TOOLS = 'OFF'
        LLAMA_BUILD_TESTS = 'OFF'; LLAMA_BUILD_EXAMPLES = 'OFF'; LLAMA_BUILD_SERVER = 'OFF'
        LLAMA_BUILD_APP = 'OFF'; LLAMA_BUILD_UI = 'OFF'; LLAMA_USE_PREBUILT_UI = 'OFF'
        LLAMA_BUILD_MTMD = 'OFF'; LLAMA_OPENSSL = 'OFF'; LLAMA_CURL = 'OFF'; LLAMA_LLGUIDANCE = 'OFF'
        LLAMA_TOOLS_INSTALL = 'OFF'; LLAMA_TESTS_INSTALL = 'OFF'
        FETCHCONTENT_FULLY_DISCONNECTED = 'ON'; FETCHCONTENT_UPDATES_DISCONNECTED = 'ON'
        GGML_CPU = 'ON'; GGML_NATIVE = 'OFF'; GGML_AVX = 'ON'; GGML_AVX2 = 'ON'
        GGML_SSE42 = 'ON'; GGML_BMI2 = 'ON'; GGML_FMA = 'ON'; GGML_F16C = 'ON'
        GGML_AVX_VNNI = 'OFF'; GGML_AVX512 = 'OFF'; GGML_AVX512_VBMI = 'OFF'
        GGML_AVX512_VNNI = 'OFF'; GGML_AVX512_BF16 = 'OFF'
        GGML_AMX_TILE = 'OFF'; GGML_AMX_INT8 = 'OFF'; GGML_AMX_BF16 = 'OFF'
        GGML_CPU_ALL_VARIANTS = 'OFF'; GGML_BACKEND_DL = 'OFF'
        GGML_CPU_HBM = 'OFF'; GGML_CPU_KLEIDIAI = 'OFF'; GGML_CCACHE = 'OFF'
        GGML_OPENMP = 'ON'; GGML_LLAMAFILE = 'ON'
        GGML_BUILD_TESTS = 'OFF'; GGML_BUILD_EXAMPLES = 'OFF'
        GGML_ACCELERATE = 'OFF'; GGML_BLAS = 'OFF'; GGML_CUDA = 'OFF'; GGML_HIP = 'OFF'
        GGML_MUSA = 'OFF'; GGML_VULKAN = 'OFF'; GGML_SYCL = 'OFF'; GGML_RPC = 'OFF'
        GGML_METAL = 'OFF'; GGML_WEBGPU = 'OFF'; GGML_OPENCL = 'OFF'; GGML_OPENVINO = 'OFF'
        GGML_HEXAGON = 'OFF'; GGML_ZDNN = 'OFF'; GGML_ZENDNN = 'OFF'
        GGML_VIRTGPU = 'OFF'; GGML_VIRTGPU_BACKEND = 'OFF'
    }
    $configure = @('-S', $source, '-B', $BuildDir, '-G', 'MinGW Makefiles')
    foreach ($key in $definitions.Keys) { $configure += "-D$key=$($definitions[$key])" }
    New-Item -ItemType Directory -Path $BuildDir -Force | Out-Null
    Invoke-Checked $CMake $configure | Out-Host
    # Check the actual cache before compiling, including flags that could select
    # a host-specific ISA or a different ABI/toolchain.
    $cachePath = Join-Path $BuildDir 'CMakeCache.txt'
    Require-File $cachePath
    $cache = @{}
    foreach ($line in (Get-Content -LiteralPath $cachePath)) {
        if ($line -match '^([^#/:][^:]*):[^=]+=(.*)$') { $cache[$matches[1]] = $matches[2] }
    }
    foreach ($key in $definitions.Keys) {
        $expected = $definitions[$key].Replace('\', '/')
        if (-not $cache.ContainsKey($key) -or $cache[$key].Replace('\', '/') -cne $expected) {
            throw "Unexpected CMake cache value for $key; choose a fresh build directory"
        }
    }
    if ($cache['CMAKE_GENERATOR'] -cne 'MinGW Makefiles' -or
        $cache['CMAKE_HOME_DIRECTORY'].Replace('\', '/') -ine $source.Replace('\', '/')) {
        throw 'Unexpected CMake generator/source directory'
    }
    $compile = @('--build', $BuildDir, '--config', 'Release', '--target', 'llama', '--parallel', "$Jobs")
    Invoke-Checked $CMake $compile | Out-Host
    $nativeBin = Join-Path $BuildDir 'bin'
    $libraries = @('libllama.dll', 'ggml.dll', 'ggml-base.dll', 'ggml-cpu.dll')
    foreach ($name in $libraries) { Require-File (Join-Path $nativeBin $name) }
    Require-File (Join-Path $BuildDir 'src/libllama.dll.a')
    New-Item -ItemType Directory -Path $OutDir -Force | Out-Null
    if ($OutDir -ine $nativeBin) {
        foreach ($name in $libraries) {
            Copy-Item -LiteralPath (Join-Path $nativeBin $name) -Destination (Join-Path $OutDir $name)
        }
    }
    $shellName = if ($PSVersionTable.PSEdition -eq 'Desktop') { 'powershell.exe' } else { 'pwsh.exe' }
    $shell = Join-Path $PSHOME $shellName
    Require-File $shell
    $shimArgs = @('-NoProfile', '-ExecutionPolicy', 'Bypass', '-File', $shimScript,
        '-LlamaCppDir', $source, '-BuildDir', $BuildDir,
        '-OutPath', (Join-Path $OutDir 'qlh_keep_head.dll'), '-Gcc', $gcc)
    Invoke-Checked $shell $shimArgs | Out-Host
    $artifacts = @(
        foreach ($name in (@('qlh_keep_head.dll') + $libraries + $runtimes)) {
            $path = Join-Path $OutDir $name
            Require-File $path
            $size = (Get-Item -LiteralPath $path).Length
            if ($size -eq 0) { throw "Empty native output: $path" }
            $hash = Hash-File $path
            if ($name -in $runtimes -and $hash -cne (Hash-File (Join-Path $ToolchainBin $name))) {
                throw "Shim did not copy the selected toolchain runtime: $name"
            }
            Write-Host "[dll] $name sha256=$hash bytes=$size"
            [ordered]@{ name = $name; path = $path; size_bytes = $size; sha256 = $hash }
        }
    )
    Assert-CleanSource
    $receipt = [ordered]@{
        schema_version = 1; exit_code = 0; completed_utc = [DateTime]::UtcNow.ToString('o')
        source_root = $SourceRoot; source = $source; source_revision = $revision; source_tree = $tree
        upstream_revision = $lock.upstream.revision; source_git_clean = $true; patches = $patches
        lock_sha256 = (Hash-File $lockPath); builder_sha256 = (Hash-File $PSCommandPath)
        header_sha256 = (Hash-File (Join-Path $source 'include/llama.h'))
        shim_script_sha256 = (Hash-File $shimScript); shim_source_sha256 = (Hash-File $shimSource)
        build_dir = $BuildDir; out_dir = $OutDir; tools = $tools
        configure_arguments = $configure; build_arguments = $compile; shim_arguments = $shimArgs
        cache_sha256 = (Hash-File $cachePath); artifacts = $artifacts
    }
    $receiptPath = Join-Path $BuildDir 'keep-head-cpu-build.json'
    $receipt | ConvertTo-Json -Depth 8 | Set-Content -LiteralPath $receiptPath -Encoding UTF8
    Write-Host "[done] Nine fresh CPU DLLs: $OutDir"
    Write-Host "[receipt] $receiptPath"
} catch {
    Write-Host "[FAIL] $($_.Exception.Message)"
    exit 1
} finally {
    $env:PATH = $originalPath
    $env:GIT_OPTIONAL_LOCKS = $originalOptionalLocks
}
exit 0
