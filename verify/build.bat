@echo off
REM verify/ 的构建入口：一次 cmake --build 覆盖验收要用的全部目标。
REM
REM 用法: build.bat [--clean-first]
REM   verify.sh 用 QLH_BUILD_BAT 指到这里（上游树没有自己的 .qlh_tmp/build.bat 时）。
REM
REM 路径来自环境变量，verify/_qlh_env.sh 会 export：
REM   QLH_SRC_DIR    上游 llama.cpp 源码根（缺省 H:/llamacpp-c）
REM   QLH_BUILD_DIR  构建目录，其下有 bin/（缺省 %QLH_SRC_DIR%/build-split）
REM   QLH_VCVARS     vcvars64.bat 的路径（缺省用下面这一行，装到别处就自己设）
REM
REM 注意：这个文件必须保持 CRLF 行尾。写成 LF 时 cmd 解析下面那行 call 会让 vcvars 的
REM INCLUDE 不生效，任何 cl 调用都报 C1083（打不开 <stdbool.h>）——实测 -j 4 和 -j 1 都报。
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
