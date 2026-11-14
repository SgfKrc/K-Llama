@echo off
REM Build entry point for verify/: one cmake --build covering every target the
REM acceptance run needs.
REM
REM Usage: build.bat [--clean-first]
REM   verify.sh points QLH_BUILD_BAT here when the upstream tree has no
REM   .qlh_tmp/build.bat of its own.
REM
REM Paths come from the environment, exported by verify/_qlh_env.sh:
REM   QLH_SRC_DIR    upstream llama.cpp source root (default H:/llamacpp-c)
REM   QLH_BUILD_DIR  build directory with bin/ under it (default %QLH_SRC_DIR%/build-split)
REM   QLH_VCVARS     path to vcvars64.bat (default is the line below; override if
REM                  installed elsewhere)
REM
REM KEEP THIS FILE ASCII-ONLY. cmd parses a batch file in the console code page,
REM which here is GBK. UTF-8 Chinese bytes are not valid GBK pairs, so they shift
REM and can yield '|' -- which splits a REM line into "REM ... | command" and makes
REM cmd try to run the remainder. That is exactly how this file broke on
REM 2026-11-14: the REM block above turned into a handful of "'...' is not
REM recognized as an internal or external command" lines and the build never ran.
REM Chinese belongs in the .sh and .md files, not here.
REM
REM KEEP CRLF LINE ENDINGS TOO. With LF, cmd's parse of the call below leaves
REM vcvars' INCLUDE unset and every cl invocation fails with C1083 (cannot open
REM <stdbool.h>); measured with both -j 4 and -j 1.
setlocal
if "%QLH_SRC_DIR%"=="" set "QLH_SRC_DIR=H:/llamacpp-c"
if "%QLH_BUILD_DIR%"=="" set "QLH_BUILD_DIR=%QLH_SRC_DIR%/build-split"
if "%QLH_VCVARS%"=="" set "QLH_VCVARS=C:\Program Files (x86)\Microsoft Visual Studio\18\BuildTools\VC\Auxiliary\Build\vcvars64.bat"
if not exist "%QLH_VCVARS%" (
  echo [build] error: vcvars64.bat not found: %QLH_VCVARS%
  echo [build] set QLH_VCVARS=^<path to vcvars64.bat^> and retry
  exit /b 2
)
echo [build] src=%QLH_SRC_DIR%  build=%QLH_BUILD_DIR%
call "%QLH_VCVARS%" >nul 2>&1
cd /d "%QLH_SRC_DIR%"
if errorlevel 1 (
  echo [build] error: cannot cd to %QLH_SRC_DIR%
  exit /b 2
)
cmake --build "%QLH_BUILD_DIR%" --target llama split-stage llama-server split-smoke split-rows dflash-taps %* -- -j 4
exit /b %errorlevel%
