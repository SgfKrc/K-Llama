#!/data/data/com.termux/files/usr/bin/bash
# 把一台 Termux 设备准备成推理链的一段，然后后台常驻。
#
#   bash termux-install.sh --fork ~/qlh-fork --model ~/qwen36-35b-a3b-q6k.gguf \
#                          --range 24:40 --port 9106
#
# 脚本是幂等的：依赖已装、fork 已编、shim 已编都会跳过，重复跑只做缺的部分。
#
# 三个必须知道的点：
#   1. 编译用 -j 1。并发编译会被 Android 的 lmkd 直接 SIGKILL（实测 -j 6 在第 26% 被杀）。
#   2. 服务必须 setsid + termux-wake-lock，否则 adb/ssh 会话一断就被带走，或者 CPU 休眠时被冻结。
#   3. 模型必须放在 Termux 私有目录。run-as 启动的进程读不了 /sdcard（SELinux 拒绝），
#      而 mmap 在 FUSE 上极慢。

set -euo pipefail

FORK=""
MODEL=""
RANGE=""
PORT=9106
THREADS=5
CTX=1024
ROLE=tail
HOST=0.0.0.0
LOG="$HOME/qlh-stage.log"

say() { printf '\n\033[1m== %s ==\033[0m\n' "$*"; }
die() { printf '\033[31m错误：%s\033[0m\n' "$*" >&2; exit 1; }

while [ $# -gt 0 ]; do
    case "$1" in
        --fork)    FORK="$2"; shift 2 ;;
        --model)   MODEL="$2"; shift 2 ;;
        --range)   RANGE="$2"; shift 2 ;;
        --port)    PORT="$2"; shift 2 ;;
        --threads) THREADS="$2"; shift 2 ;;
        --ctx)     CTX="$2"; shift 2 ;;
        --role)    ROLE="$2"; shift 2 ;;
        --host)    HOST="$2"; shift 2 ;;
        --log)     LOG="$2"; shift 2 ;;
        -h|--help) sed -n '2,20p' "$0"; exit 0 ;;
        *) die "未知参数 $1" ;;
    esac
done

[ -n "$FORK" ]  || die "必须给 --fork（llama.cpp fork 的源码目录）"
[ -n "$MODEL" ] || die "必须给 --model（完整整模的 .gguf 路径）"
[ -n "$RANGE" ] || die "必须给 --range（如 0:24 或 24:40）"

case "$RANGE" in
    *:*) ;;
    *) die "--range 要写成 lo:hi" ;;
esac
LO=${RANGE%%:*}; HI=${RANGE##*:}
[ $((HI - LO)) -gt 0 ] || die "--range 的 hi 必须大于 lo"
# qwen35moe 是 hybrid（每 4 层一个 full attention），切点和段内层数都必须是 4 的倍数
[ $((LO % 4)) -eq 0 ] || die "切点 $LO 不是 4 的倍数，hybrid 架构会层类型错位"
[ $(((HI - LO) % 4)) -eq 0 ] || die "段内层数 $((HI - LO)) 不是 4 的倍数"

[ -f "$MODEL" ] || die "模型不存在：$MODEL"
say "模型 $MODEL（$(du -h "$MODEL" | cut -f1)）"
say "角色 $ROLE  层范围 $RANGE（$((HI - LO)) 层）"

# ---------- 1. 依赖 ----------
say "检查依赖"
MISSING=()
for pkg in clang cmake make; do
    command -v "$pkg" >/dev/null 2>&1 || MISSING+=("$pkg")
done
python -c "import numpy" 2>/dev/null || MISSING+=("python-numpy")
if [ ${#MISSING[@]} -gt 0 ]; then
    echo "缺：${MISSING[*]}"
    echo "装（网络慢，可能要十几分钟）："
    echo "  pkg install -y clang cmake make python-numpy"
    die "先装依赖再重跑"
fi
python -c "import numpy; print('numpy', numpy.__version__)"

# ---------- 2. fork 源码与构建 ----------
[ -d "$FORK/src" ] || die "$FORK 不像 llama.cpp 源码目录（没有 src/）"
BUILD="$FORK/build-android"
BIN="$BUILD/bin"

if [ ! -f "$BUILD/CMakeCache.txt" ]; then
    say "配置 fork（首次）"
    cd "$FORK"
    cmake -B build-android -DCMAKE_BUILD_TYPE=Release -DGGML_NATIVE=ON \
          -DLLAMA_CURL=OFF -DLLAMA_BUILD_TESTS=OFF -DLLAMA_BUILD_SERVER=OFF
fi

if [ ! -f "$BIN/libllama.so" ]; then
    say "编译 libllama（-j 1，首次约 20-40 分钟，中途被 SIGKILL 就重跑）"
    cd "$FORK"
    cmake --build build-android -j 1 --target llama
else
    say "libllama 已存在，跳过编译"
fi
[ -f "$BIN/libllama.so" ] || die "编译没产出 libllama.so"

# ---------- 3. shim ----------
SHIM="$BIN/libqlh_keep_head.so"
# 移植后的源码可能叫这几个名字之一（build_shim_posix.py 生成的是第一个）
SHIM_SRC=""
for cand in qlh_keep_head_posix.c qlh_keep_head_linux.embd-channel.c qlh_keep_head_linux.c; do
    if [ -f "$FORK/$cand" ]; then SHIM_SRC="$FORK/$cand"; break; fi
done
[ -n "$SHIM_SRC" ] || die "在 $FORK 里找不到移植后的 shim 源码 —— 用 shim/build_shim_posix.py 生成后传过来"
say "编译 shim（源：$(basename "$SHIM_SRC")）"
cd "$FORK"
clang -shared -O2 -fPIC -DQLH_ABI_UPSTREAM_HEADER -Wl,-rpath,'$ORIGIN' \
      -o "$SHIM" "$SHIM_SRC" \
      -Iinclude -Iggml/include -L"$BIN" \
      -lllama -lggml -lggml-base -lm
ls -la "$SHIM"

# ---------- 4. 停掉旧实例 ----------
say "清理旧实例"
pkill -f "[s]tage_worker.py" 2>/dev/null || true
pkill -f "[k]eep_head_worker.py" 2>/dev/null || true
sleep 2

# ---------- 5. 起服务 ----------
say "启动（后台，日志 $LOG）"
export HOME
export PREFIX="${PREFIX:-/data/data/com.termux/files/usr}"
export PATH="$PREFIX/bin:$PATH"
export LD_LIBRARY_PATH="$BIN:$PREFIX/lib"
export LLAMA_LAYER_RANGE="$RANGE"

command -v termux-wake-lock >/dev/null 2>&1 && termux-wake-lock || true

WORKER="$HOME/qlh-worker"
[ -f "$WORKER/stage_worker.py" ] || die "缺 $WORKER/stage_worker.py"

setsid python "$WORKER/stage_worker.py" \
    --role "$ROLE" \
    --model "$MODEL" \
    --shim "$SHIM" \
    --host "$HOST" \
    --port "$PORT" \
    --threads "$THREADS" \
    --layers "$((HI - LO))" \
    --ctx "$CTX" \
    >"$LOG" 2>&1 < /dev/null &

sleep 8
if pgrep -f "[s]tage_worker.py" >/dev/null; then
    say "已启动，PID $(pgrep -f '[s]tage_worker.py' | head -1)"
    printf '监听 %s:%s\n' "$HOST" "$PORT"
    printf '日志 %s\n' "$LOG"
    printf '\n它是脱离会话的，adb/ssh 断开不会带走它。\n'
    printf '停：pkill -f "[s]tage_worker.py"\n'
    printf '看状态：pgrep -f "[s]tage_worker.py" && tail -5 %s\n' "$LOG"
else
    say "启动失败，日志尾部："
    tail -30 "$LOG"
    exit 1
fi
