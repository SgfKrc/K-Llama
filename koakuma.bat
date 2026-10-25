@echo off
chcp 65001 >nul
set "PYTHONUTF8=1"
set "PYTHONIOENCODING=utf-8"
rem Koakuma is the qlh alias entry point.
rem Keep this launcher ASCII-only for cmd.exe code-page safety.
cd /d "%~dp0"
rem Promote every QLH_* key from .env into this process environment
rem (2026-10-24, #117).
rem
rem src/config.py setting helpers and src/model_api_access.py read os.environ
rem only; the SettingsStore .env snapshot never reaches them, so a key set there
rem silently had no effect (known issues #112 and #117).  Previously only two keys
rem were copied by hand, which turned every other QLH_* setting in .env into a
rem no-op -- e.g. QLH_PIPELINE_TIMEOUT or QLH_CAPACITY_PLANNING_MODE looked
rem applied but never reached the process.
rem
rem Copy them all.  A value already present in the environment wins, so an
rem explicit override on the command line still takes precedence.
if exist "%~dp0.env" (
  for /f "usebackq eol=# tokens=1,* delims==" %%A in ("%~dp0.env") do (
    call :promote_qlh "%%A" "%%B"
  )
)
set "PYTHON_CMD=python"
where python >nul 2>nul
if not %errorlevel%==0 set "PYTHON_CMD=py -3"
%PYTHON_CMD% qlh.py %*
exit /b %errorlevel%

:promote_qlh
rem Only QLH_* keys; never touch PATH/PYTHON*/other toolchain variables.
set "_qlh_key=%~1"
if /i not "%_qlh_key:~0,4%"=="QLH_" exit /b 0
if defined %_qlh_key% exit /b 0
set "%_qlh_key%=%~2"
exit /b 0
