<#
  ours_procs.ps1 —— 列出/强杀「我们自己的」llama.cpp 进程，供 verify.sh 的 cleanup_ours 用。

  为什么要有这个文件：旧的清理只看 netstat 里的 93xx LISTENING，而泄漏的恰恰是看不到的那些：
    * loading 中的 split-stage 还没 bind     -> netstat 里没有
    * split-smoke / split-rows / dflash-taps 从不监听端口 -> netstat 里永远没有
    * llama-server 启动失败（OOM / 参数错）    -> 没有 LISTENING
  于是它们活着、占着 1.3~4 GB 内存和 9331 端口，下一组就跑出「identity mismatch」「health=000」
  这种看起来像代码 bug 的假失败。

  识别条件只有一个：可执行文件路径在 -BinDir 下。不按端口、不按名字前缀模糊匹配：
  这台机器上还有别人的 ollama.exe / "ollama app.exe"（名字不在名单里、路径也不在 -BinDir 下，
  两道都挡住才动手），绝不能碰。

  为什么用 CIM 而不是 wmic：wmic 在这台机器上会静默返回空（正在被淘汰），静默漏杀正是上次误判的起因；
  Win32_Process 的 ExecutablePath 才拿得到「这个 llama-server.exe 是我这个 bin 里的还是别人的」。

  用法：
    powershell -NoProfile -File ours_procs.ps1 -BinDir "<QLH_BUILD_DIR>/bin" -List
    powershell -NoProfile -File ours_procs.ps1 -BinDir "<QLH_BUILD_DIR>/bin" -Kill
  输出：每行 "PID<TAB>NAME"（-Kill 时只输出真正杀掉的）。
#>
[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)][string] $BinDir,
    [switch] $Kill,
    [switch] $List,
    [string[]] $Images = @(
        'split-smoke.exe',
        'split-stage.exe',
        'split-rows.exe',
        'dflash-taps.exe',
        'llama-server.exe'
    )
)

$ErrorActionPreference = 'Continue'

# 统一成小写正斜杠再比前缀，避免 H:\ vs H:/ 、大小写、结尾斜杠的三种意外不相等。
$prefix = (($BinDir -replace '\\', '/').TrimEnd('/') + '/').ToLowerInvariant()

# 防御：$prefix 一旦算成空串，StartsWith('') 恒真，整台机器上所有同名进程都会被算成
# 「我们的」并被强杀。2026-11-10 实测到过这种形态：本文件无 UTF-8 BOM，Windows
# PowerShell 5.1 按 ANSI 解码中文注释时乱码把上面这条赋值吞掉，$prefix 变空 —— 表现是
# 「清理正常」，实际杀的是别人的 llama-server。宁可不动手，也不按空前缀匹配。
if ([string]::IsNullOrEmpty($prefix) -or $prefix -eq '/') {
    [Console]::Error.WriteLine("ours_procs.ps1: -BinDir 解析后为空（[$BinDir]），拒绝按空前缀匹配（会误杀整机同名进程）")
    exit 3
}

$filter = (@($Images) | ForEach-Object { "Name='$_'" }) -join ' or '

$found = @(Get-CimInstance Win32_Process -Filter $filter -ErrorAction SilentlyContinue | Where-Object {
    $exe = $_.ExecutablePath
    if ([string]::IsNullOrWhiteSpace($exe)) { return $false }
    return (($exe -replace '\\', '/').ToLowerInvariant().StartsWith($prefix))
})

if ($Kill) {
    foreach ($p in $found) {
        try {
            Stop-Process -Id $p.ProcessId -Force -ErrorAction Stop
            Write-Output ("{0}`t{1}" -f $p.ProcessId, $p.Name)
        } catch {
            [Console]::Error.WriteLine("kill {0} ({1}) failed: {2}" -f $p.ProcessId, $p.Name, $_.Exception.Message)
        }
    }
} else {
    foreach ($p in $found) {
        Write-Output ("{0}`t{1}" -f $p.ProcessId, $p.Name)
    }
}
