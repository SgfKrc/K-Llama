# Build llama.cpp with CUDA as shared libraries, so the keep-head shim can link
# against a libllama that has a CUDA backend. Existing shim builds are CPU-only
# (no ggml-cuda.dll in their bin dir), which caps the GPU stage at CPU speed.
param(
    [string]$LlamaCppDir = 'G:\C\PYT\qlh\android\app\src\main\cpp\llama.cpp',
    [string]$BuildDir = 'H:\slimarc-proto\build\llama-cuda-shim',
    [string]$CudaArch = '89',
    [int]$Jobs = 8
)

$ErrorActionPreference = 'Stop'
Set-StrictMode -Version Latest

Write-Host "=== locating toolchain ==="
$vswhere = "${env:ProgramFiles(x86)}\Microsoft Visual Studio\Installer\vswhere.exe"
if (-not (Test-Path $vswhere)) { throw 'vswhere not found' }
$vsPath = & $vswhere -latest -products * -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath
if (-not $vsPath) { throw 'MSVC not found via vswhere' }
Write-Host "  VS: $vsPath"

Import-Module (Join-Path $vsPath 'Common7\Tools\Microsoft.VisualStudio.DevShell.dll')
Enter-VsDevShell -VsInstallPath $vsPath -SkipAutomaticLocation -DevCmdArguments '-arch=x64' | Out-Null

$cmake = Get-Command cmake -ErrorAction SilentlyContinue
if (-not $cmake) { throw 'cmake not on PATH' }
Write-Host "  cmake: $($cmake.Source)"
$nvcc = Get-Command nvcc -ErrorAction SilentlyContinue
if (-not $nvcc) { throw 'nvcc not on PATH; install CUDA toolkit' }
Write-Host "  nvcc: $($nvcc.Source)"

Write-Host "`n=== configure ==="
$cfgArgs = @(
    '-S', $LlamaCppDir,
    '-B', $BuildDir,
    '-G', 'Ninja',
    '-DCMAKE_BUILD_TYPE=Release',
    '-DBUILD_SHARED_LIBS=ON',
    '-DGGML_CUDA=ON',
    "-DCMAKE_CUDA_ARCHITECTURES=$CudaArch",
    '-DGGML_NATIVE=OFF',
    # Runtime-dispatched per-ISA CPU backends. Without this the CPU stage runs the
    # baseline AVX2 kernel; Raptor Lake gets alderlake/icelake-class kernels with it.
    # GGML_CPU_ALL_VARIANTS requires dynamic backend loading (see ggml/src/CMakeLists).
    '-DGGML_CPU_ALL_VARIANTS=ON',
    '-DGGML_BACKEND_DL=ON',
    '-DGGML_OPENMP=OFF',
    '-DLLAMA_BUILD_EXAMPLES=OFF',
    '-DLLAMA_BUILD_TESTS=OFF',
    '-DLLAMA_BUILD_SERVER=OFF',
    '-DLLAMA_CURL=OFF'
)
& cmake @cfgArgs
if ($LASTEXITCODE -ne 0) { throw "configure failed: $LASTEXITCODE" }

Write-Host "`n=== build (jobs=$Jobs) ==="
& cmake --build $BuildDir --config Release --target llama -j $Jobs
if ($LASTEXITCODE -ne 0) { throw "build failed: $LASTEXITCODE" }

Write-Host "`n=== artifacts ==="
Get-ChildItem (Join-Path $BuildDir 'bin') -ErrorAction SilentlyContinue |
    ForEach-Object { "  $($_.Name)  $([math]::Round($_.Length / 1MB, 1)) MB" }
Write-Host 'DONE'
