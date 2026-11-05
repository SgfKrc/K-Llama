# 用 MSVC 编译 qlh_chat.dll（仅 Windows；纯 C++，零 llama 运行时依赖）。
#
# 与 shim/build_shim_msvc.ps1 的区别：那个编的是 keep-head shim（要链 llama.lib，
# 头文件必须与被链接的 llama.dll 同树）；这个编的是 llama.cpp common/chat 那套模板
# 渲染 + PEG 解析器，**只吃头文件**，产物导入表里不该出现 llama.dll / ggml.dll。
# 头文件仍然必须来自同一棵上游树（chat.h / 模板行为随上游变）。
#
# 三个不能省的编译开关：
#   /utf-8      源码与 vendored 头里有非 ASCII 注释，不加 MSVC 会按代码页 936 解析
#   /std:c++17  upstream 用 if-constexpr / string_view / variant
#   /MD         与 python 侧 CRT 对齐（跨边界只有 malloc/free + 我们自己的 str_free）
#
# 千万别加 -I shim\chat\jinja 或 -I <llama>\common\jinja：<string.h> 会被解析到
# jinja/string.h，报一屏 memchr/strlen is not a member of the global namespace。
# jinja/*.cpp 里那些 `#include "lexer.h"` 是靠"包含该文件所在目录优先"找到的。
#
# 用例：
#   powershell -File shim/build_chat_msvc.ps1 -LlamaRoot H:\llamacpp-c
#   powershell -File shim/build_chat_msvc.ps1 -LlamaRoot H:\llamacpp-c -OutDir D:\x

param(
    [string]$LlamaRoot = 'H:\llamacpp-c',
    [string]$OutDir = ''
)

# $PSScriptRoot 在 param 默认值求值时还没赋值，默认路径必须挪到这里算。
if (-not $OutDir) { $OutDir = Join-Path (Get-Location) '.qlh_tmp\chatout' }

$ErrorActionPreference = 'Stop'
Set-StrictMode -Version Latest

$SrcDir = Join-Path $PSScriptRoot 'chat'
$AbiSrc = Join-Path $PSScriptRoot 'qlh_chat.cpp'

$incCommon = Join-Path $LlamaRoot 'common'
$incLlama  = Join-Path $LlamaRoot 'include'
$incGgml   = Join-Path $LlamaRoot 'ggml\include'

foreach ($p in @($SrcDir, $AbiSrc, (Join-Path $incCommon 'chat.h'), (Join-Path $incLlama 'llama.h'))) {
    if (-not (Test-Path $p)) { throw "缺少：$p" }
}

$objDir = Join-Path $OutDir 'obj'
New-Item -ItemType Directory -Path $objDir -Force | Out-Null

$vswhere = "${env:ProgramFiles(x86)}\Microsoft Visual Studio\Installer\vswhere.exe"
$vsPath = & $vswhere -latest -products * -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath
if (-not $vsPath) { throw 'vswhere 没找到 VC 工具集' }
Import-Module (Join-Path $vsPath 'Common7\Tools\Microsoft.VisualStudio.DevShell.dll')
Enter-VsDevShell -VsInstallPath $vsPath -SkipAutomaticLocation -DevCmdArguments '-arch=x64' | Out-Null
Write-Host "VS: $vsPath"

# 注意 include 搜索顺序：vendored 树在前，上游只提供 common.h / llama*.h / ggml*.h。
# 没有 vendor 目录就找不到 <nlohmann/json.hpp>（json.cpp 用的是尖括号形式）。
$incArgs = @(
    "/I$SrcDir",
    "/I$SrcDir\vendor",
    "/I$incCommon",
    "/I$incLlama",
    "/I$incGgml"
)

$srcs = @(Get-ChildItem -Path $SrcDir -Recurse -Filter '*.cpp' | Sort-Object FullName | ForEach-Object FullName)
$srcs += $AbiSrc
Write-Host "`n=== 编译 $($srcs.Count) 个 .cpp（MSVC /std:c++17 /O2）==="

$t0 = Get-Date
foreach ($f in $srcs) {
    if ($f.StartsWith($SrcDir)) {
        $rel = $f.Substring($SrcDir.Length).TrimStart('\')
    } else {
        $rel = [System.IO.Path]::GetFileName($f)
    }
    $obj = Join-Path $objDir (($rel -replace '[\\/]', '_') -replace '\.cpp$', '.obj')
    $log = & cl /nologo /c /std:c++17 /O2 /W3 /utf-8 /EHsc /MD $incArgs "/Fo$obj" $f 2>&1
    if ($LASTEXITCODE -ne 0) {
        $log | Where-Object { $_ -match 'error|fatal' } | Select-Object -First 10 | ForEach-Object { Write-Host "   $_" }
        throw "编译失败：$rel"
    }
    Write-Host ("ok   {0}" -f $rel)
}
Write-Host ("编译用时 {0:N1}s" -f ((Get-Date) - $t0).TotalSeconds)

Write-Host "`n=== 链接（/LD /OPT:REF /OPT:ICF）==="
$objs = @(Get-ChildItem -Path $objDir -Filter '*.obj' | ForEach-Object FullName)
$dll  = Join-Path $OutDir 'qlh_chat.dll'
$log  = & cl /nologo /LD /EHsc "/Fe:$dll" $objs /link /OPT:REF /OPT:ICF 2>&1
if ($LASTEXITCODE -ne 0) {
    $log | Where-Object { $_ -match 'error|LNK' } | Select-Object -First 20 | ForEach-Object { Write-Host "   $_" }
    throw "链接失败：$LASTEXITCODE"
}

$size = (Get-Item $dll).Length
Write-Host ("`n=== 产物：{0}（{1:N0} 字节 / {2:N0} KB）===" -f $dll, $size, ($size / 1KB))

# 硬自检：导入表里不能有 llama / ggml。这是"独立 DLL"的判据，不靠体积猜。
Write-Host "`n=== dumpbin /dependents ==="
$deps = & dumpbin /nologo /dependents $dll 2>&1
$deps | Where-Object { $_ -match '\.dll' } | ForEach-Object { Write-Host "  $_" }
$bad = @($deps | Where-Object { $_ -match '(?i)llama|ggml' })
if ($bad.Count -gt 0) {
    throw "产物依赖了 llama/ggml：`n$($bad -join "`n")"
}
Write-Host '自检通过：导入表里没有 llama.dll / ggml.dll'
