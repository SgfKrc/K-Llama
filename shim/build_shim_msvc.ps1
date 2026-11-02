# 用 MSVC 编译 keep-head shim（Windows）。
#
# 头文件必须来自与被链接 DLL **同一棵树**：`llama_context_params` 是按值传递的大结构体，
# 用别处的 llama.h 编译会把字段写错位（典型症状：`llama_init_from_model: Unsupported ctx type`）。
#
# 两个开关都不能省：
#   /utf-8                     源码带 UTF-8 中文注释，不加 MSVC 会按代码页 936 解析
#   /DQLH_ABI_UPSTREAM_HEADER  shim 里设置 cparams.embeddings / embeddings_pre_norm 的
#                              整段包在这个条件里；不定义就整段被预处理掉，head 段
#                              会因为 embd.data == NULL 返回 rc=-3
#
# 千万别加 /DQLH_LLAMA_ABI_NO_EXPERIMENTAL_TAIL -- 那是配旧 DLL 的，会把
# embeddings_pre_norm 从结构体里摘掉，与当前 llama.dll 不再匹配。
#
# 用例：
#   powershell -File shim/build_shim_msvc.ps1 -LlamaRoot H:\llamacpp-c -OutDir .\.qlh_tmp\shimout

param(
    [Parameter(Mandatory = $true)][string]$LlamaRoot,
    [string]$OutDir = '',
    [string]$Source = '',
    [string]$Def    = ''
)

# $PSScriptRoot 在 param 默认值求值时还没赋值，默认路径必须挪到这里算。
if (-not $OutDir) { $OutDir = Join-Path (Get-Location) '.qlh_tmp\shimout' }
if (-not $Source) { $Source = Join-Path $PSScriptRoot 'qlh_keep_head_gpu.c' }
if (-not $Def)    { $Def    = Join-Path $PSScriptRoot 'qlh_keep_head.def' }

$ErrorActionPreference = 'Stop'
Set-StrictMode -Version Latest

$build = Join-Path $LlamaRoot 'build-vulkan'   # 按需改成实际的构建目录
$inc   = Join-Path $LlamaRoot 'include'
$gginc = Join-Path $LlamaRoot 'ggml\include'
$lib1  = Join-Path $build 'src'
$lib2  = Join-Path $build 'ggml\src'

foreach ($p in @($Source, $Def, (Join-Path $inc 'llama.h'), (Join-Path $lib1 'llama.lib'))) {
    if (-not (Test-Path $p)) { throw "缺少：$p" }
}
New-Item -ItemType Directory -Path $OutDir -Force | Out-Null

$vswhere = "${env:ProgramFiles(x86)}\Microsoft Visual Studio\Installer\vswhere.exe"
$vsPath = & $vswhere -latest -products * -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath
if (-not $vsPath) { throw 'vswhere 没找到 VC 工具集' }
Import-Module (Join-Path $vsPath 'Common7\Tools\Microsoft.VisualStudio.DevShell.dll')
Enter-VsDevShell -VsInstallPath $vsPath -SkipAutomaticLocation -DevCmdArguments '-arch=x64' | Out-Null
Write-Host "VS: $vsPath"

Write-Host '=== 编译 shim（MSVC）==='
& cl /nologo /LD /O2 /W3 /utf-8 `
    /DQLH_ABI_UPSTREAM_HEADER `
    $Source `
    "/Fe:$OutDir\qlh_keep_head.dll" `
    "/Fo:$OutDir\qlh_keep_head.obj" `
    "/I$inc" "/I$gginc" `
    /link `
    "/DEF:$Def" `
    "/LIBPATH:$lib1" "/LIBPATH:$lib2" `
    /DEFAULTLIB:llama.lib /DEFAULTLIB:ggml.lib /DEFAULTLIB:ggml-base.lib
if ($LASTEXITCODE -ne 0) { throw "编译失败：$LASTEXITCODE" }

$dll = Join-Path $OutDir 'qlh_keep_head.dll'
$size = (Get-Item $dll).Length
Write-Host "`n=== 产物：$dll（$size 字节）==="

# 自检：产物里没有这个字符串就说明宏没生效，head 段必然报 rc=-3。
$bytes  = [System.IO.File]::ReadAllBytes($dll)
$needle = [System.Text.Encoding]::ASCII.GetBytes('SA_NEEDS_LOGITS')
$found  = $false
for ($i = 0; $i -le ($bytes.Length - $needle.Length); $i++) {
    if ($bytes[$i] -ne $needle[0]) { continue }
    $ok = $true
    for ($j = 1; $j -lt $needle.Length; $j++) {
        if ($bytes[$i + $j] -ne $needle[$j]) { $ok = $false; break }
    }
    if ($ok) { $found = $true; break }
}
if ($found) {
    Write-Host '自检通过：SA_NEEDS_LOGITS 在产物里'
} else {
    Write-Warning '产物里没有 SA_NEEDS_LOGITS —— 宏没生效，用这个 dll 跑 head 会返回 rc=-3'
}
