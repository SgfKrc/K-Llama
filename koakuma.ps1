# K-Llama 一行启动（PowerShell 直接调，不必经过 .cmd 的引号规则）。
# 参数原样透传： koakuma.ps1 --topology topology-2b.toml
$ErrorActionPreference = "Stop"
$here = Split-Path -Parent $MyInvocation.MyCommand.Path
$py = if ($env:KLLAMA_PYTHON) { $env:KLLAMA_PYTHON }
      else { "G:\C\PYT\qlh\.venv-test\Scripts\python.exe" }
if (-not (Test-Path $py)) {
    Write-Error "K-Llama: python 解释器不存在: $py`n用 KLLAMA_PYTHON 环境变量指定一个。"
    exit 1
}
& $py (Join-Path $here "koakuma.py") @args
exit $LASTEXITCODE
