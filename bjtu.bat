@echo off
chcp 65001 >nul
set "PYTHONUTF8=1"
set "PYTHONIOENCODING=utf-8"
cd /d "%~dp0"
set "QLH_APP_MODE="
set "PYTHON_CMD=python"
if exist ".venv\Scripts\python.exe" set "PYTHON_CMD=%CD%\.venv\Scripts\python.exe"
if not defined QLH_CORE_ROOT set "QLH_CORE_ROOT=%CD%"
if not defined QLH_RELEASE_ROOT set "QLH_RELEASE_ROOT=%CD%\packaging"
if not defined QLH_SHELL_ROOT set "QLH_SHELL_ROOT=%CD%\frontend_cybergothic"
if /i "%~1"=="check" goto :maintenance
if /i "%~1"=="download" goto :maintenance
if /i "%~1"=="install" goto :maintenance
if /i "%~1"=="version-status" goto :maintenance
if /i "%~1"=="version-stage" goto :maintenance
if /i "%~1"=="version-activate" goto :maintenance
if /i "%~1"=="version-rollback" goto :maintenance
if /i "%~1"=="version-recover" goto :maintenance
if /i "%~1"=="diagnostics" goto :maintenance
if /i "%~1"=="verify" goto :maintenance
if /i "%~1"=="diagnose" goto :maintenance
if /i "%~1"=="repair" goto :maintenance
if /i "%~1"=="data-status" goto :maintenance
if /i "%~1"=="retain-data" goto :maintenance
if /i "%~1"=="reassociate-data" goto :maintenance
if /i "%~1"=="reinstall" goto :maintenance
if /i "%~1"=="maintenance" goto :explicit_maintenance
if /i "%~1"=="update" goto :update
if /i "%~1"=="version" goto :version
if /i "%~1"=="ui" set "QLH_APP_MODE=--ui"
if /i "%~1"=="web" set "QLH_APP_MODE=--web"
if /i "%~1"=="headless" set "QLH_APP_MODE=--headless"
if defined QLH_APP_MODE goto :app_mode
if not exist "src\api_server.py" goto :packaged
"%PYTHON_CMD%" qlh.py %*
exit /b %errorlevel%
:app_mode
if exist "QLH-Edge-Inference.exe" (
    "QLH-Edge-Inference.exe" %QLH_APP_MODE% %2 %3 %4 %5 %6 %7 %8 %9
) else (
    "%PYTHON_CMD%" src\app_bootstrap.py %QLH_APP_MODE% %2 %3 %4 %5 %6 %7 %8 %9
)
exit /b %errorlevel%
:packaged
if /i "%~1"=="chat" (
    if not exist "%~dp0QLH-TUI-Chat\QLH-TUI-Chat.exe" exit /b 2
    "%~dp0QLH-TUI-Chat\QLH-TUI-Chat.exe" %2 %3 %4 %5 %6 %7 %8 %9
    exit /b %errorlevel%
)
"QLH-Edge-Inference.exe" --tui %*
exit /b %errorlevel%
:maintenance
if exist "QLH-Edge-Inference.exe" (
    "QLH-Edge-Inference.exe" --maintenance %*
) else (
    "%PYTHON_CMD%" qlh.py maintenance %*
)
exit /b %errorlevel%
:explicit_maintenance
if exist "QLH-Edge-Inference.exe" (
    "QLH-Edge-Inference.exe" --maintenance %2 %3 %4 %5 %6 %7 %8 %9
) else (
    "%PYTHON_CMD%" qlh.py %*
)
exit /b %errorlevel%
:update
if exist "QLH-Edge-Inference.exe" (
    "QLH-Edge-Inference.exe" --maintenance check %2 %3 %4 %5 %6 %7 %8 %9
) else (
    "%PYTHON_CMD%" qlh.py maintenance check %2 %3 %4 %5 %6 %7 %8 %9
)
exit /b %errorlevel%
:version
if exist "packaging\packaging\version.txt" type packaging\packaging\version.txt
if exist "version.txt" type version.txt
exit /b 0
