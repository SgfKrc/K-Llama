# Compile the prototype keep-head shim against the CUDA llama.dll build.
#
# Upstream shim builds use MinGW + a CPU-only libllama. The CUDA build here is
# MSVC, so the shim must be MSVC too: mixing ABIs across the llama.h boundary
# would break at the first struct-by-value call.
param(
    [string]$Source = 'H:\slimarc-proto\shim\qlh_keep_head_gpu.c',
    [string]$BuildDir = 'H:\slimarc-proto\build\llama-cuda-shim',
    [string]$OutDir = 'H:\slimarc-proto\shim\bin'
)

$ErrorActionPreference = 'Stop'
Set-StrictMode -Version Latest

$vswhere = "${env:ProgramFiles(x86)}\Microsoft Visual Studio\Installer\vswhere.exe"
$vsPath = & $vswhere -latest -products * -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath
Import-Module (Join-Path $vsPath 'Common7\Tools\Microsoft.VisualStudio.DevShell.dll')
Enter-VsDevShell -VsInstallPath $vsPath -SkipAutomaticLocation -DevCmdArguments '-arch=x64' | Out-Null

# Headers come from a llama-cpp-python wheel rather than a llama.cpp checkout: the
# upstream tree is gone from this machine, and only the two QLH-local layer_inp
# symbols are missing from the published llama.h (declared by hand in the shim).
$llamaSrc = 'H:\slimarc-proto\build-headers'

New-Item -ItemType Directory -Path $OutDir -Force | Out-Null

Write-Host '=== compiling shim (MSVC) ==='
# MinGW exports every symbol by default; MSVC exports none. The .def file keeps
# the qlh_kh_* surface ctypes expects.
$defFile = 'H:\slimarc-proto\shim\qlh_keep_head.def'
$args = @(
    '/nologo', '/LD', '/O2', '/W3',
    # The bundled llama.h comes from a newer experimental branch than the DLL. The
    # guard drops the three fields that branch added before n_threads, which is what
    # makes the struct layout match the DLL again.
    '/DQLH_LLAMA_ABI_NO_EXPERIMENTAL_TAIL',
    $Source,
    "/Fe:$OutDir\qlh_keep_head.dll",
    "/Fo:$OutDir\qlh_keep_head.obj",
    "/I$llamaSrc",
    "/link",
    "/DEF:$defFile",
    "/LIBPATH:$BuildDir\src",
    "/LIBPATH:$BuildDir\ggml\src",
    '/DEFAULTLIB:llama.lib',
    '/DEFAULTLIB:ggml.lib',
    '/DEFAULTLIB:ggml-base.lib'
)
& cl @args
if ($LASTEXITCODE -ne 0) { throw "compile failed: $LASTEXITCODE" }

Write-Host "`n=== copying runtime DLLs next to the shim ==="
foreach ($dll in @('llama.dll', 'ggml.dll', 'ggml-base.dll', 'ggml-cpu.dll', 'ggml-cuda.dll')) {
    $src = Join-Path $BuildDir "bin\$dll"
    if (Test-Path $src) {
        Copy-Item $src $OutDir -Force
        Write-Host "  $dll"
    } else {
        Write-Host "  MISSING: $dll"
    }
}

Write-Host "`n=== result ==="
Get-ChildItem $OutDir -Filter '*.dll' | ForEach-Object {
    "  $($_.Name)  $([math]::Round($_.Length / 1MB, 2)) MB"
}
Write-Host 'DONE'
