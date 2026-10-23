@echo off
chcp 65001 >nul
set "PYTHONUTF8=1"
set "PYTHONIOENCODING=utf-8"
rem Koakuma is the qlh alias entry point.
rem Keep this launcher ASCII-only for cmd.exe code-page safety.
cd /d "%~dp0"
rem Promote QLH_MODEL_API_TRUSTED_CIDRS from .env into this process environment.
rem src/model_api_access.py reads os.environ only; the SettingsStore .env snapshot
rem never reaches it, so that key silently had no effect (known issue #112).
if not defined QLH_MODEL_API_TRUSTED_CIDRS if exist "%~dp0.env" (
  for /f "usebackq tokens=1,* delims==" %%A in ("%~dp0.env") do (
    if /i "%%A"=="QLH_MODEL_API_TRUSTED_CIDRS" set "QLH_MODEL_API_TRUSTED_CIDRS=%%B"
  )
)
set "PYTHON_CMD=python"
where python >nul 2>nul
if not %errorlevel%==0 set "PYTHON_CMD=py -3"
%PYTHON_CMD% qlh.py %*
exit /b %errorlevel%
