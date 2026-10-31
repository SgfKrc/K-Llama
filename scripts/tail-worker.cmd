@echo off
rem ---------------------------------------------------------------- tail worker launcher
rem
rem Start one remote stage (the one whose `start_cmd` points here). Templated on purpose:
rem the topology file decides which model and which layer range this machine runs, so the
rem same script serves any machine that happens to host a stage.
rem
rem Args (all required):
rem   %1  python executable
rem   %2  stage_worker.py path
rem   %3  model path
rem   %4  shim dll path
rem   %5  port
rem   %6  layers
rem   %7  layer range, e.g. 32:40
rem   %8  threads
rem   %9  ctx
rem
rem Example generated invocation:
rem   tail-worker.cmd python C:\repo\stage_worker.py D:\m.gguf D:\shim.dll 9103 8 32:40 10 1024

setlocal

if "%~1"=="" (
  echo usage: tail-worker.cmd PYTHON WORKER MODEL SHIM PORT LAYERS RANGE THREADS CTX
  exit /b 2
)

set PY=%~1
set WORKER=%~2
set MODEL=%~3
set SHIM=%~4
set PORT=%~5
set LAYERS=%~6
set RANGE=%~7
set THREADS=%~8
set CTX=%~9

rem The tail is the only stage that emits tokens, so it needs the output projection.
set SA_NEEDS_LOGITS=1
set SA_MTP=1
rem Recurrent snapshot depth. A rejected round retracts up to (drafts per round) positions;
rem seq_rm refuses anything deeper, and a refused rollback leaves the state ahead of where
rem the chain thinks it is.
if "%SA_N_RS_SEQ%"=="" set SA_N_RS_SEQ=3
if "%SA_KH_RS_SEQ%"=="" set SA_KH_RS_SEQ=16
set LLAMA_LAYER_RANGE=%RANGE%

cd /d "%~dp0"
"%PY%" "%WORKER%" ^
  --role tail ^
  --model "%MODEL%" ^
  --shim "%SHIM%" ^
  --host 0.0.0.0 ^
  --port %PORT% ^
  --ctx %CTX% ^
  --threads %THREADS% ^
  --n-gpu-layers 0 ^
  --n-cpu-moe 0 ^
  --layers %LAYERS%

endlocal
