@echo off
REM usage: build_seq_rm_exact.bat ^<src_root^> ^<bin_dir^> ^<obj_tmp^> ^<build_dir^>
REM   src_root   upstream llama.cpp source root (has include\ and src\)
REM   bin_dir    output dir (= the build dir's bin\)
REM   obj_tmp    intermediate dir (= .qlh_tmp\verify)
REM   build_dir  build dir (llama.lib is under its src\)
REM
REM This probe only links llama.lib (+ ggml.lib when it calls the ggml API);
REM it is not part of the main build.
REM
REM PITFALL 1 - keep this file ASCII-only, CRLF line endings.
REM cmd parses .bat files with the OEM code page (936/GBK here). UTF-8 Chinese
REM comments can contain byte sequences that are invalid in GBK; cmd then
REM truncates or merges lines, the leftover text is executed as commands and the
REM bat dies with "xxx is not recognized as an internal or external command".
REM
REM PITFALL 2 - the vcvars existence check uses goto, never an if-block.
REM %%QLH_VCVARS%% expands to "C:\Program Files (x86)\..."; inside
REM "if ... ( ... )" that ")" closes the block early, the rest of the block and
REM the following lines run as commands, and cmd reports
REM "\Microsoft was unexpected at this time." with exit code 255.
setlocal
if "%~4"=="" (
  echo usage: build_seq_rm_exact.bat ^<src_root^> ^<bin_dir^> ^<obj_tmp^> ^<build_dir^>
  exit /b 2
)
if defined QLH_VCVARS goto :qlh_have_vcvars
set "QLH_VCVARS=%ProgramFiles(x86)%\Microsoft Visual Studio\18\BuildTools\VC\Auxiliary\Build\vcvars64.bat"
:qlh_have_vcvars
if exist "%QLH_VCVARS%" goto :qlh_vcvars_ok
echo [probe-build] error: vcvars64.bat not found: %QLH_VCVARS%
exit /b 2
:qlh_vcvars_ok
call "%QLH_VCVARS%" >nul 2>&1
cd /d "%~1"
if errorlevel 1 exit /b 2
if not exist "%~2" mkdir "%~2"
if not exist "%~3" mkdir "%~3"
REM seq_rm_exact.cpp uses only the llama API, so llama.lib is enough.
cl /nologo /utf-8 /std:c++17 /EHsc /O2 /MD /DNDEBUG /I include /I ggml/include /I src "%~dp0seq_rm_exact.cpp" /Fe:"%~2\seq_rm_exact.exe" /Fo:"%~3\seq_rm_exact.obj" /link /LIBPATH:"%~4\src" llama.lib
exit /b %errorlevel%
