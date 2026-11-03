#!/data/data/com.termux/files/usr/bin/bash
# 把一台 Termux 设备准备成推理链的一段，然后后台常驻。
#
# 单段（调试用）：
#   bash termux-install.sh --fork ~/qlh-fork --model ~/qwen36-35b-a3b-q6k.gguf \
#                          --range 24:40 --port 9106
#
# 常驻（推荐）：两段链，本机 + 对端，拓扑由参数生成
#   bash termux-install.sh --fork ~/qlh-fork --model ~/qwen36-35b-a3b-q6k.gguf \
#                          --role supervisor --range 24:40 --port 9106 \
#                          --peer-host 100.100.52.106 --peer-port 9103 --peer-layers 24
#   生成 $HOME/qlh-worker/topology.toml 并打印出来供核对；本机段的 host 固定写 0.0.0.0。
#
# 也可以继续用自己写的拓扑（此时 --peer-* 不参与、--range 不参与，不覆盖任何文件）：
#   bash termux-install.sh --fork ~/qlh-fork --model ~/qwen36-35b-a3b-q6k.gguf \
#                          --role supervisor --topology ~/qlh-worker/topology.toml
#
# 脚本是幂等的：依赖已装、fork 已编、shim 已编都会跳过，重复跑只做缺的部分。
#
# 四个必须知道的点：
#   1. 编译用 -j 1。并发编译会被 Android 的 lmkd 直接 SIGKILL（实测 -j 6 在第 26% 被杀）。
#   2. 服务必须 setsid + termux-wake-lock，否则 adb/ssh 会话一断就被带走，或者 CPU 休眠时被冻结。
#   3. 模型必须放在 Termux 私有目录。run-as 启动的进程读不了 /sdcard（SELinux 拒绝），
#      而 mmap 在 FUSE 上极慢。
#   4. 本机段的 host 必须写 0.0.0.0，不能是 127.0.0.1：worker 拿它当 bind 地址，绑回环
#      的话对端连不进来（日志是 listening on 127.0.0.1:9106，对端 connection refused）。
#      脚本生成拓扑时固定这样写，所以这件事不用再记。

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
TOPOLOGY=""
# 对端那一段，只在"从参数生成拓扑"时用；给了 --topology 就完全不看它们。
PEER_HOST=""
PEER_PORT=""
PEER_LAYERS=""
PEER_MODEL=""
PEER_THREADS=""
# 部署目录：stage_worker.py / stage_supervisor.py / 生成的 topology.toml 都住这里。
WORKER="$HOME/qlh-worker"

say() { printf '\n\033[1m== %s ==\033[0m\n' "$*"; }
die() { printf '\033[31m错误：%s\033[0m\n' "$*" >&2; exit 1; }

while [ $# -gt 0 ]; do
    case "$1" in
        --fork)         FORK="$2"; shift 2 ;;
        --model)        MODEL="$2"; shift 2 ;;
        --range)        RANGE="$2"; shift 2 ;;
        --port)         PORT="$2"; shift 2 ;;
        --threads)      THREADS="$2"; shift 2 ;;
        --ctx)          CTX="$2"; shift 2 ;;
        --role)         ROLE="$2"; shift 2 ;;
        --host)         HOST="$2"; shift 2 ;;
        --topology)     TOPOLOGY="$2"; shift 2 ;;
        --peer-host)    PEER_HOST="$2"; shift 2 ;;
        --peer-port)    PEER_PORT="$2"; shift 2 ;;
        --peer-layers)  PEER_LAYERS="$2"; shift 2 ;;
        --peer-model)   PEER_MODEL="$2"; shift 2 ;;
        --peer-threads) PEER_THREADS="$2"; shift 2 ;;
        --log)          LOG="$2"; shift 2 ;;
        -h|--help) sed -n '2,27p' "$0"; exit 0 ;;
        *) die "未知参数 $1" ;;
    esac
done

[ -n "$FORK" ]  || die "必须给 --fork（llama.cpp fork 的源码目录）"
[ -f "$MODEL" ] || die "模型不存在：$MODEL"
# 端口要和 --peer-port 比较大小，先生成校验，别让 shell 在比较时才报 integer expected。
case "$PORT" in ''|*[!0-9]*) die "--port 要是数字，收到 '$PORT'" ;; esac
[ "$PORT" -ge 1024 ] && [ "$PORT" -le 65535 ] || die "--port 要在 1024-65535，收到 $PORT"
# 拓扑里的 model 必须是绝对路径：supervisor 的工作目录不一定是当前目录，
# 相对路径在它那里会解析到别处。命令行允许相对路径，这里补成绝对。
case "$MODEL" in
    /*|[A-Za-z]:/*) ;;
    *) MODEL="$(cd "$(dirname "$MODEL")" && pwd)/$(basename "$MODEL")" ;;
esac

# 层范围校验只写一份：单段形态和"从参数生成拓扑"形态都要跑它。
check_range() {
    case "$RANGE" in
        *:*) ;;
        *) die "--range 要写成 lo:hi（如 0:24 或 24:40）" ;;
    esac
    LO=${RANGE%%:*}; HI=${RANGE##*:}
    [ -n "$LO" ] && [ -n "$HI" ] || die "--range 要写成 lo:hi（如 0:24 或 24:40）"
    case "$LO$HI" in
        ''|*[!0-9]*) die "--range 的两端要是数字，收到 '$RANGE'" ;;
    esac
    [ $((HI - LO)) -gt 0 ] || die "--range 的 hi 必须大于 lo"
    # qwen35moe 是 hybrid（每 4 层一个 full attention），切点和段内层数都必须是 4 的倍数
    [ $((LO % 4)) -eq 0 ] || die "切点 $LO 不是 4 的倍数，hybrid 架构会层类型错位"
    [ $(((HI - LO) % 4)) -eq 0 ] || die "段内层数 $((HI - LO)) 不是 4 的倍数"
}

# 一个 [[stage]] 块。model 用 TOML 字面量字符串（单引号），
# 这样路径里的反斜杠不会被当成转义（Windows 上手工验证拓扑时会用到）。
emit_stage() {  # $1=model $2=host $3=port $4=layers $5=lo $6=hi $7=kind $8=threads
    cat <<EOF

[[stage]]
model       = '$1'
host        = "$2"
port        = $3
layers      = $4
layer_range = "$5:$6"
kind        = "$7"
threads     = $8
EOF
}

# 从 --range / --peer-* 生成两段拓扑。
#
# 为什么不让人手写：本机段的 host 必须写 0.0.0.0，而这一点是 worker 的 bind 语义决定的，
# 写在别处又看不见——实测写 127.0.0.1 时日志正常、远端 connection refused，排查很久。
# 生成器保证这一条，其余每一项都从命令行推出来。
write_generated_topology() {
    local missing=""
    [ -n "$PEER_HOST" ]   || missing="$missing --peer-host"
    [ -n "$PEER_PORT" ]   || missing="$missing --peer-port"
    [ -n "$PEER_LAYERS" ] || missing="$missing --peer-layers"
    if [ -n "$missing" ]; then
        die "生成拓扑还需要$missing（描述对端那一段：地址 / 端口 / 层数）"
    fi

    case "$PEER_PORT" in ''|*[!0-9]*) die "--peer-port 要是数字，收到 '$PEER_PORT'" ;; esac
    [ "$PEER_PORT" -ge 1024 ] && [ "$PEER_PORT" -le 65535 ] \
        || die "--peer-port 要在 1024-65535，收到 $PEER_PORT"
    case "$PEER_LAYERS" in ''|*[!0-9]*) die "--peer-layers 要是数字，收到 '$PEER_LAYERS'" ;; esac
    [ "$PEER_LAYERS" -gt 0 ] || die "--peer-layers 必须大于 0"
    [ $((PEER_LAYERS % 4)) -eq 0 ] \
        || die "对端层数 $PEER_LAYERS 不是 4 的倍数，hybrid 架构会层类型错位"
    # topology 要求每段端口唯一。两台机器各绑 9106 是物理上可行的，但同一份拓扑里
    # 两个 [[stage]] 写同一个 port 会被 load() 拒绝，这里先说清楚。
    [ "$PEER_PORT" -ne "$PORT" ] \
        || die "--peer-port 不能等于 --port（$PORT）：同一份拓扑里每段端口必须唯一"

    # 对端在那台机器上。写成回环或全零会被 topology.is_loopback 认成"本机段"，
    # supervisor 会在这台机器上把它起起来，然后撞端口、按退避反复重启。
    case "$PEER_HOST" in
        127.0.0.1|localhost|0.0.0.0|::1|::)
            die "--peer-host 要写对端那台机器的地址，不能写 $PEER_HOST（那会被当成本机段）" ;;
    esac

    local me_layers=$((HI - LO)) peer_lo peer_hi me_kind peer_kind
    local peer_model="${PEER_MODEL:-$MODEL}" peer_threads="${PEER_THREADS:-$THREADS}"
    # kind 由层位置决定，不另要参数：topology.py 要求 kind 顺序是 gpu 在前、tail 收尾，
    # 所以低层那段只能是 gpu、末段只能是 tail。--role supervisor 不区分 gpu/tail，
    # 位置就是唯一依据；两者永远一致，不会再造出它拒绝的拓扑。
    if [ "$LO" -eq 0 ]; then
        # 本机段从第 0 层开始，对端接在它后面。
        peer_lo=$HI
        peer_hi=$((HI + PEER_LAYERS))
        me_kind=gpu; peer_kind=tail
    else
        peer_lo=$((LO - PEER_LAYERS))
        peer_hi=$LO
        [ "$peer_lo" -ge 0 ] \
            || die "本机段从第 $LO 层开始，前面放不下对端的 $PEER_LAYERS 层（对端会从第 $peer_lo 层开始）"
        me_kind=tail; peer_kind=gpu
    fi

    local source_layers=$((me_layers + PEER_LAYERS))
    mkdir -p "$WORKER"
    local tmp="$TOPOLOGY.new.$$"
    say "生成的拓扑（写 $TOPOLOGY）"
    {
        cat <<EOF
# 由 scripts/termux-install.sh 从命令行生成，不要手改——重跑脚本会覆盖它。
# 本机段 $RANGE（$me_layers 层，$me_kind），对端 $PEER_HOST:$PEER_PORT $peer_lo:$peer_hi（$PEER_LAYERS 层，$peer_kind）。
#
# 本机段的 host 写 0.0.0.0 不是笔误：worker 拿它当 bind 地址，写 127.0.0.1 就只绑回环，
# 对端会 connection refused。topology.is_loopback 也认 0.0.0.0，所以 supervisor 仍把它
# 当本机段自己起。对端段的 model 指那台机器上的路径，本机不会打开它。

source_layers = $source_layers
EOF
        if [ "$LO" -eq 0 ]; then
            emit_stage "$MODEL" 0.0.0.0 "$PORT" "$me_layers" "$LO" "$HI" "$me_kind" "$THREADS"
            emit_stage "$peer_model" "$PEER_HOST" "$PEER_PORT" "$PEER_LAYERS" \
                       "$peer_lo" "$peer_hi" "$peer_kind" "$peer_threads"
        else
            emit_stage "$peer_model" "$PEER_HOST" "$PEER_PORT" "$PEER_LAYERS" \
                       "$peer_lo" "$peer_hi" "$peer_kind" "$peer_threads"
            emit_stage "$MODEL" 0.0.0.0 "$PORT" "$me_layers" "$LO" "$HI" "$me_kind" "$THREADS"
        fi
    } >"$tmp"
    # 打印出来让人核对：生成的 host 是不是 0.0.0.0、层区间有没有重叠或空洞。
    cat "$tmp"

    # 用仓库自己的解析器兜一遍，而不是在 shell 里重写它的一半规则（层数总和、4 对齐、
    # 端口唯一、kind 顺序都在 topology.py 里）。topology.py 随 worker 一起部署；不在就
    # 只提示——部署包不全时不该卡在生成拓扑这一步。
    if [ -f "$WORKER/topology.py" ]; then
        ( cd "$WORKER" && python -c 'import sys, topology; topology.load(sys.argv[1])' "$tmp" ) \
            || { rm -f "$tmp"; die "生成的拓扑没通过 topology.load() 校验（报错见上）；临时文件已删，$TOPOLOGY 没动"; }
        printf 'topology.load() 校验通过\n'
    else
        printf '提示：%s 不在，跳过 topology.load() 校验\n' "$WORKER/topology.py"
    fi
    mv "$tmp" "$TOPOLOGY"
}

if [ "$ROLE" = "supervisor" ]; then
    if [ -n "$TOPOLOGY" ]; then
        # 手写拓扑：保持旧行为，--range / --peer-* 都不参与，脚本不碰用户的文件。
        [ -f "$TOPOLOGY" ] || die "拓扑不存在：$TOPOLOGY"
        if [ -n "$PEER_HOST$PEER_PORT$PEER_LAYERS" ]; then
            printf '提示：已给 --topology，沿用这份文件，--peer-* 未使用\n'
        fi
    else
        [ -n "$RANGE" ] \
            || die "role=supervisor 且没给 --topology 时，要用 --range 描述本机段（如 24:40）"
        check_range
        TOPOLOGY="$WORKER/topology.toml"
        write_generated_topology
    fi
else
    # --peer-* 只在生成拓扑时用；单段形态下给了也不生效，说一句免得以为已经生成。
    if [ -n "$PEER_HOST$PEER_PORT$PEER_LAYERS" ]; then
        printf '提示：--peer-* 只在 --role supervisor 且没给 --topology 时用，单段形态不生成拓扑\n'
    fi
    [ -n "$RANGE" ] || die "必须给 --range（如 0:24 或 24:40）"
    check_range
fi

say "模型 $MODEL（$(du -h "$MODEL" | cut -f1)）"
if [ "$ROLE" = "supervisor" ]; then
    say "角色 supervisor  拓扑 $TOPOLOGY"
else
    say "角色 $ROLE  层范围 $RANGE（$((HI - LO)) 层）"
fi

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

[ -f "$WORKER/stage_worker.py" ] || die "缺 $WORKER/stage_worker.py"

if [ "$ROLE" = "supervisor" ]; then
    # supervisor 自己读拓扑决定起几段、绑哪个地址，所以不传 --range/--host/--port；
    # 它还会在段崩溃时按退避重启（见 stage_supervisor.py）。
    [ -f "$WORKER/stage_supervisor.py" ] || die "缺 $WORKER/stage_supervisor.py"
    export SA_TOPOLOGY="$TOPOLOGY"
    export SA_SHIM="$SHIM"
    unset LLAMA_LAYER_RANGE
    setsid python "$WORKER/stage_supervisor.py" >"$LOG" 2>&1 < /dev/null &
    PATTERN="[s]tage_supervisor.py"
else
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
    PATTERN="[s]tage_worker.py"
fi

sleep 8
if pgrep -f "$PATTERN" >/dev/null; then
    say "已启动，PID $(pgrep -f "$PATTERN" | head -1)"
    if [ "$ROLE" != "supervisor" ]; then
        printf '监听 %s:%s\n' "$HOST" "$PORT"
    fi
    printf '日志 %s\n' "$LOG"
    printf '\n它是脱离会话的，adb/ssh 断开不会带走它。\n'
    printf '停：pkill -f "%s"\n' "$PATTERN"
    printf '看状态：pgrep -f "%s" && tail -5 %s\n' "$PATTERN" "$LOG"
else
    say "启动失败，日志尾部："
    tail -30 "$LOG"
    exit 1
fi
