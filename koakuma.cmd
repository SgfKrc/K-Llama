@echo off
rem K-Llama 一行启动。用 call 里的 python 解释器，避免 PATH 上没有 python 时静默失败。
rem 所有参数原样透传给 koakuma.py（--topology / --shim / --ctx / --prompt）。
setlocal
set "HERE=%~dp0"
if defined KLLAMA_PYTHON (
    set "PY=%KLLAMA_PYTHON%"
) else (
    set "PY=G:\C\PYT\qlh\.venv-test\Scripts\python.exe"
)
if not exist "%PY%" (
    echo K-Llama: python 解释器不存在: %PY%
    echo 用 KLLAMA_PYTHON 环境变量指定一个，或装好 .venv-test。
    exit /b 1
)
"%PY%" "%HERE%koakuma.py" %*
exit /b %ERRORLEVEL%
