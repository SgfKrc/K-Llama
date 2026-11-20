@echo off
REM Build entry point for verify/: one cmake --build covering every target the
REM acceptance run needs.
REM
REM Usage: build.bat [--clean-first]
REM   verify.sh points QLH_BUILD_BAT here when the upstream tree has no
REM   .qlh_tmp/build.bat of its own.
REM
REM Paths come from the environment, exported by verify/_qlh_env.sh:
REM   QLH_SRC_DIR    upstream llama.cpp source root, default H:/llamacpp-c
REM   QLH_BUILD_DIR  build directory with bin/ under it, default QLH_SRC_DIR/build-split
REM   QLH_VCVARS     path to vcvars64.bat, default is the line below; override it
REM                  if Visual Studio lives elsewhere
REM
REM THREE HARD RULES FOR THIS FILE. All three were learned by breaking it.
REM
REM 1. ASCII ONLY, NO SHELL METACHARACTERS IN COMMENTS. cmd parses a batch file in
REM    the console code page, which here is GBK; UTF-8 Chinese bytes are not valid
REM    GBK pairs, so they shift and can produce a pipe, and a pipe in a REM line
REM    splits it into a REM line followed by a command, which then runs. The same
REM    goes for redirection characters. Keep Chinese in the .sh and .md files.
REM
REM 2. NO PARENTHESISED IF BLOCKS AROUND A VARIABLE THAT CAN CONTAIN A PAREN.
REM    QLH_VCVARS points under "Program Files (x86)", so expanding it inside an
REM    if-block body puts a closing paren where cmd expects the block to end. cmd
REM    parses the whole block before running any of it, so even a branch that
REM    never executes breaks the parse: the symptom is cmd complaining that
REM    \Microsoft is unexpected at this time. Use goto labels instead, as below.
REM
REM 3. CRLF LINE ENDINGS. With LF, cmd parses the call below in a way that leaves
REM    vcvars' INCLUDE unset, and every cl invocation fails with C1083, cannot
REM    open stdbool.h. Measured with both -j 4 and -j 1.
setlocal
if "%QLH_SRC_DIR%"=="" set "QLH_SRC_DIR=H:/llamacpp-c"
if "%QLH_BUILD_DIR%"=="" set "QLH_BUILD_DIR=%QLH_SRC_DIR%/build-split"
if "%QLH_VCVARS%"=="" set "QLH_VCVARS=C:\Program Files (x86)\Microsoft Visual Studio\18\BuildTools\VC\Auxiliary\Build\vcvars64.bat"
if exist "%QLH_VCVARS%" goto vcvars_ok
echo [build] error: vcvars64.bat not found: %QLH_VCVARS%
echo [build] set QLH_VCVARS=^<path to vcvars64.bat^> and retry
exit /b 2
:vcvars_ok
echo [build] src=%QLH_SRC_DIR%  build=%QLH_BUILD_DIR%
call "%QLH_VCVARS%" >nul 2>&1
cd /d "%QLH_SRC_DIR%"
if not errorlevel 1 goto cd_ok
echo [build] error: cannot cd to %QLH_SRC_DIR%
exit /b 2
:cd_ok
cmake --build "%QLH_BUILD_DIR%" --target llama split-stage llama-server split-smoke split-rows dflash-taps %* -- -j 4
exit /b %errorlevel%
