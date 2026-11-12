#!/bin/bash
# 分段链四组验收，一条命令跑完。
#
# 为什么要这个：这四组验证之前只有"上一次某个人手工跑过"的口头结论，没人能重跑。
# 每次改完 llama.cpp 侧的分段代码，跑这一遍就是验收；任何一组退化都说明改坏了。
#
# 用法：
#   bash verify/verify.sh              # 增量编译 + 全部四组
#   bash verify/verify.sh --clean      # 全量重编 + 全部四组（改过头文件必须用这个）
#   bash verify/verify.sh --skip-build # 只跑验证
#
# 注意：上游构建目录的 ninja 依赖记录不可靠（多数对象 deps=0），改头文件不会触发重编，
# 增量构建可能产出 ABI 混用的 dll。改过 .h 之后一律 --clean。
#
# 「远端 stage 消失」那一类（head 必须活着、请求级失败、/health?deep=1）不在这个脚本里：
# 它要真的 NtSuspendProcess 挂起一个 stage 再等 120 s 读超时，跑一趟约 6 分钟。
# 单独跑：python verify/verify_chain_death.py
#
# 巡检脚本请用浅 /health，不要轮询 /health?deep=1：deep 探针是一次真请求
# （cache_prompt=false，强制真的 decode 1 个 token），它会清掉该 slot 的 KV，远端链（单 seq）
# 上下一个真实请求因此要重新算 prompt；一次两跳链上实测约 80 ms，代价在缓存不在时间。
# deep=1 用于「已经在怀疑链路」或改完配置后各问一次。
#
# 退出码 0 = 四组全过。
#
# 项数：44 -> 52（2026-11-09 加组 11）-> 53（2026-11-11 加组 12）-> 57（2026-11-12 加组 13）。
# 组 13 的 4 项覆盖 **KVMem 的 KV 虚拟化**（本地双段 + MTP + 驱逐，见下）。组 11 的 8 项覆盖
# **DFlash 投机的 device-buffer 注入路径**：组 9/10 用的 tests/dflash-taps.cpp 只 link `llama`，
# 从不调 llama_set_embeddings_layer_inp_dev（ctx_tgt 没有 device buffer），batch 用
# llama_batch_init 而不是 llama_batch_ext_add_embd_row（ubatch.embd_row 恒空，graph 走 host
# embd 分支），所以 common/speculative.cpp 的那条路径**此前零覆盖**。v9 的默认路径在远程链上
# 第一次请求就断言（dflash.cpp 的 embd-row 断言）——产品级崩溃、而组 9/10 全绿，缺口就在这里。
# 组 11 直接跑产品路径：llama-server + 链 + -md dflash。
#
# 组 12 的 1 项覆盖**段级 offload**：组 1-8 都跑在 build-split 上（GGML_CUDA=OFF，`-ngl` 被静默
# 忽略），所以「一段里的前 N 层放 GPU、其余留 CPU」从没被验收过。组 12 用 CUDA 构建跑
# 「整模 vs --split-layer-range 0:24，都 -ngl 4」，判据是逐字节相同。CUDA 构建不存在时 SKIP。
#
# 环境变量（都有默认值，一般不用管）：
#   QLH_VERIFY_FAST=1           只跑 2B 的组（1-8 与 12），组 9/10/11 明确 SKIP（SKIP 不算通过）
#   QLH_VERIFY_MIN_FREE_GB=20   组 9/10 的内存闸门（mmap 的模型不需要全占物理内存，见下文注释）
#   QLH_VERIFY_SMOKE_TIMEOUT=900 单次 split-smoke / split-rows 的超时（秒）
#   QLH_VERIFY_PY_TIMEOUT=1800  组 5 python 的超时（秒）
#   QLH_VERIFY_BIG_TIMEOUT=3600 组 9/10 dflash-taps 的超时（秒）
#   QLH_VERIFY_NO_LOCK=1        跳过「同一时间只允许一个 verify」的闸门（不推荐）
#   QLH_CUDA_BUILD_DIR         组 12 的 CUDA 构建目录（默认 H:/llamacpp-c/build-cuda）
#   QLH_VERIFY_NGL_PARTIAL     组 12 的 -ngl 层数（默认 4）
#   QLH_KVMEM_BUILD_DIR        组 13 的 KVMem 构建目录（默认 H:/llamacpp-c/build-kvmem2）
#   LLAMA_KVMEM_ROOT           组 13 的 KVMem 源码根（默认 .qlh_tmp/kvmem-probe）
# 退出码：0 = 被测到的组全过（SKIP 不算失败，但会显著打印「没测到」）；1 = 有 FAIL；
#         2 = 另一个 verify.sh 正在跑（并发跑会互抢 93xx、互相强杀，两边都会报假失败）。
#
# 进程清理：每次跑完 / 被 Ctrl-C / 被 kill 都会把自己起的 split-smoke / split-stage /
# split-rows / dflash-taps / llama-server 杀干净——按「镜像名 + $BIN 路径」识别
# （见 ours_procs.ps1），不看端口，因为 loading 中的 stage 和不监听端口的 smoke 在 netstat 里
# 根本不存在，那正是过去三次「进程没清干净 → 被报成代码 bug」的起因。机器上别人的
# ollama.exe / "ollama app.exe" 名字和路径都不在名单里，绝不碰。
# 组 12 的 CUDA 构建不在 $BIN 下，由 cleanup_cuda 用同一个 ours_procs.ps1、换 -BinDir 清。

set -u

# 路径定位（QLH_BUILD_DIR / QLH_SRC_DIR / QLH_TMP）见 _qlh_env.sh；不要在下面再写死机器路径。
source "$(dirname "${BASH_SOURCE[0]}")/_qlh_env.sh"
# 构建入口：默认沿用上游树的 .qlh_tmp/build.bat；那份不在时退到本目录的 build.bat（同样只是
# 一次 cmake --build 的包装）。两者都没有就明确报错，不静默跳过编译。
QLH_BUILD_BAT=${QLH_BUILD_BAT:-$QLH_SRC_DIR/.qlh_tmp/build.bat}
[ -f "$QLH_BUILD_BAT" ] || QLH_BUILD_BAT=$QLH_VERIFY_DIR/build.bat

MODE=${1:-}

# ---------------------------------------------------------------- 环境闸门（先于一切）
# 缺二进制就点名退出（rc=3）。不这么做的话，指错目录会跑出几十条 FAIL，看起来像代码坏了；
# 静默 skip 更糟——那会让「52 项全过」变成一句谎话。
qlh_show_env
qlh_require_bins split-smoke.exe split-stage.exe llama-server.exe
for opt in split-rows.exe dflash-taps.exe; do
    if [ ! -x "$BIN/$opt" ]; then
        echo "  [env] 警告：缺 $BIN/$opt —— 依赖它的组会记 FAIL（不会静默跳过）：$([ "$opt" = split-rows.exe ] && echo '组 8（2 项）' || echo '组 9/10（22 项）')"
    fi
done
if [ ! -f "$QLH_BUILD_BAT" ]; then
    if [ "$MODE" = "--skip-build" ]; then
        echo "  [env] 提示：找不到构建脚本（$QLH_SRC_DIR/.qlh_tmp/build.bat 与 $QLH_VERIFY_DIR/build.bat 都不在）；--skip-build 用不到它，继续"
    else
        echo "错误：缺构建脚本：$QLH_SRC_DIR/.qlh_tmp/build.bat 与 $QLH_VERIFY_DIR/build.bat 都不在。" >&2
        echo "  用 --skip-build 跳过编译，或用 QLH_BUILD_BAT=<构建脚本> 指定一个。" >&2
        exit 3
    fi
fi

MODEL=${QLH_SPLIT_MODEL:-H:/qlh_models/qwen3-5-2b-gguf/qwen35-2b-Q4_K_M.gguf}
# 组 9 的 DFlash tap 测试要 40 层的 35B 模型，和上面 2B 不是同一个文件；它以前靠测试里
# 内置的缺省路径，现在缺省去掉了，所以这里显式传 argv[1]。
MODEL_DFLASH=${QLH_DFLASH_MODEL:-H:/qlh_models/qwen3-6-35b-a3b-opus47-distill-apex-mtp/Qwen3.6-35B-A3B-Claude-4.7-Opus-Reasoning-Distilled-APEX-MTP-I-Quality.gguf}
# 三个端到端测试（split-smoke / split-rows / dflash-taps）不再内置机器路径，模型路径只从
# argv[1] 或 QLH_SPLIT_MODEL 取；这里把脚本用的路径导出，它们才能照旧跑起来。
export QLH_SPLIT_MODEL="$MODEL"
SMOKE=$BIN/split-smoke.exe
# 端口可以用环境变量改：验收时要「人为制造一次失败」——把 PORT_N2 指到一个已经被别的服务占着的端口，
# 组 3/7 就会失败，正好用来证明失败路径也把进程清干净了。（默认值不变。）
PORT_BASE=${QLH_VERIFY_PORT_BASE:-9341}
PORT_N1=${QLH_VERIFY_PORT_N1:-9342}
PORT_N2=${QLH_VERIFY_PORT_N2:-9343}
# 组 4 的 tail 段（PORT_STAGE）与组 6/7 的 relay 段（PORT_RELAY）默认是同一个端口 9331：
# 每组收尾都 cleanup，串行跑不会撞。要分开放就单独设 QLH_VERIFY_PORT_RELAY。
PORT_STAGE=${QLH_VERIFY_PORT_STAGE:-9331}
PORT_RELAY=${QLH_VERIFY_PORT_RELAY:-$PORT_STAGE}
PORT_TAIL=${QLH_VERIFY_PORT_TAIL:-9332}
# 组 12（段级 offload）用 CUDA 构建的二进制：build-split 是 GGML_CUDA=OFF，-ngl 在那里被静默忽略，
# 所以「一部分层放到 GPU 上」这条路径只能拿 CUDA 构建验收。构建不存在时组 12 记 SKIP（不是 PASS）。
CUDA_BUILD=${QLH_CUDA_BUILD_DIR:-H:/llamacpp-c/build-cuda}
CUDA_BIN=$CUDA_BUILD/bin
PORT_CUDA_BASE=${QLH_VERIFY_PORT_CUDA_BASE:-9357}
PORT_CUDA_N1=${QLH_VERIFY_PORT_CUDA_N1:-9358}
NGL_PARTIAL=${QLH_VERIFY_NGL_PARTIAL:-4}
# [C4] 唯一事实源是 _qlh_env.sh 的 QLH_RS_SEQ（llama-server 的 --split-rs-seq 与各段
# split-stage 的 --rs-seq 必须是同一个数）。这里只是本脚本的别名，改值请改那边。
RS_SEQ=$QLH_RS_SEQ

PASS=0
FAIL=0
SKIP=0
# 组间共享的对照输出。**在这里显式初始化**：组 3 要拿组 2 取到的整模基线 `a` 做比较，
# 组 2 一旦没取到（服务没起来），组 3 在 set -u 下读未定义变量会直接终止整个脚本——
# 后面 8 组一条都不跑，看起来像代码 bug。
# [C5] 组 2 的 `n1` 不在这里初始化：它只在组 2 内部「先赋值、后比较」，没有跨组的读者。
a=""
n2=""

# 组 9 / 组 10 要独占内存（整模 23.5 GB 起），机器上只要还有别的 35B 进程在跑，跑出来的结果不是失败
# 就是没意义。QLH_VERIFY_FAST=1 只跑 2B 的组（1-8 与 12，组 12 用 CUDA 构建），用来在别人占着机器时先确认没有回归；
# 正式验收必须不带这个变量跑完整的一遍。
SKIP_BIG=${QLH_VERIFY_FAST:-0}

# 组 9/10 之前要求的最小可用内存（GB）。不够就 SKIP，绝不硬跑：
#
# 为什么是 20 而不是「整模 23.5 GB」：模型是 mmap 的，**不需要整个装进物理内存**——
# 真正占物理内存的是 KV cache + 临时缓冲（几百 MB 到 2 GB），剩下的部分由页文件/文件缓存兜着。
# 拿「模型大小」当闸门是把 mmap 的语义搞反了：那样一台 31.6 GB 的机器会永远 SKIP，
# 而它实测跑得过（44/44 就是在可用 20 GB 时跑出来的）。20 GB 是有实测支撑的下限。
# 需要更严/更松时用 QLH_VERIFY_MIN_FREE_GB 覆盖。
MIN_FREE_GB=${QLH_VERIFY_MIN_FREE_GB:-20}

# 每个我们自己的进程最多等多久（秒）。split-smoke 每次都会静默约 2 分钟，见 run_limited 的注释。
SMOKE_TIMEOUT=${QLH_VERIFY_SMOKE_TIMEOUT:-900}
PY_TIMEOUT=${QLH_VERIFY_PY_TIMEOUT:-1800}
BIG_TIMEOUT=${QLH_VERIFY_BIG_TIMEOUT:-3600}

ok()   { echo "  [PASS] $*"; PASS=$((PASS+1)); }
bad()  { echo "  [FAIL] $*"; FAIL=$((FAIL+1)); }
skip() { echo "  [SKIP] $*"; SKIP=$((SKIP+1)); }
hdr()  { echo; echo "=== $* ==="; }

# ---------------------------------------------------------------- 清理（第一件事，先于一切）
# 旧的 kill_ours 只杀 93xx 上 LISTENING 的 PID，漏掉的恰好是最常见的三种残留：
#   * loading 中的 split-stage —— 还没 bind，netstat 里根本不存在（组 4/6 的 identity mismatch 就是它）
#   * split-smoke / split-rows / dflash-taps —— 从不监听端口
#   * 启动失败或卡住的 llama-server —— 没有 LISTENING
# 所以改成按「镜像名 + 可执行文件路径在 $BIN 下」识别（见 ours_procs.ps1）：
# 五个镜像名都在名单里，且路径必须以 QLH_BUILD_DIR\bin 开头，两道都满足才动手。
# 机器上别人的 ollama.exe / "ollama app.exe" 名字不在名单、路径也不在 $BIN 下，绝不碰。
# 用 PowerShell CIM 而不是 wmic：wmic 在这台机器上会静默返回空，静默漏杀正是上次误判的起因。
win_path() { printf '%s' "$1" | sed -E 's#^/([A-Za-z])/#\1:/#'; }
OURS_PS=$(win_path "$QLH_VERIFY_DIR/ours_procs.ps1")
BIN_WIN=$(win_path "$BIN")

list_ours() {      # 只列，不杀：每行 "PID<TAB>NAME"
    powershell -NoProfile -ExecutionPolicy Bypass -File "$OURS_PS" -BinDir "$BIN_WIN" 2>/dev/null | tr -d '\r'
}

kill_ours_tree() { # 强杀，只输出真正杀掉的 "PID<TAB>NAME"
    powershell -NoProfile -ExecutionPolicy Bypass -File "$OURS_PS" -BinDir "$BIN_WIN" -Kill 2>/dev/null | tr -d '\r'
}

listen_93xx() {    # 93xx 上正在 LISTENING 的 PID（兜底用：不是从 $BIN 起的老构建也要清）
    netstat -ano 2>/dev/null | grep -E 'LISTENING' | grep -E ':93[0-9][0-9][[:space:]]' \
        | awk '{print $NF}' | tr -d '\r' | sort -u
}

# 杀 + 等。只 sleep 1 是组 4/6/7 误判的直接原因：上一组的 stage 还没退，下一组已经在 bind 同一个端口了，
# 于是客户端和幽灵 stage 握手，报出 identity mismatch / health=000 这种看起来像代码 bug 的失败。
cleanup_ours() {
    local tag=${1:-组间} pid name n=0 i
    while IFS=$'\t' read -r pid name; do
        [ -n "${pid:-}" ] || continue
        n=$((n+1)); echo "  [cleanup:$tag] 杀掉 $name (pid $pid)"
    done <<< "$(kill_ours_tree)"
    for pid in $(listen_93xx); do
        if taskkill //F //PID "$pid" >/dev/null 2>&1; then
            n=$((n+1)); echo "  [cleanup:$tag] 杀掉 93xx 监听 pid $pid"
        fi
    done
    for i in $(seq 1 10); do    # 最多等 10 s，直到进程真的没了、端口不再 LISTENING
        if [ -z "$(list_ours)" ] && [ -z "$(listen_93xx)" ]; then
            [ "$n" -gt 0 ] && echo "  [cleanup:$tag] 已清 $n 个进程，端口已释放（等待 ${i}s）"
            return 0
        fi
        sleep 1
    done
    echo "  [cleanup:$tag] 警告：10 s 后仍有残留，继续跑可能对着幽灵端口失败："
    list_ours | sed 's/^/      /'
    return 0    # 不因为清理不掉就中断：让后面那一组自己报失败，比无声挂住好
}

# CUDA 构建的二进制不在 $BIN 下，cleanup_ours 的「路径前缀 = $BIN」认不到它（组 12 用）。
# 单独一条，且只在 CUDA_BIN_WIN 非空时动作：其余组不付这份 powershell 开销。
CUDA_BIN_WIN=""
cleanup_cuda() {
    local tag=${1:-cuda} pid name n=0
    [ -n "$CUDA_BIN_WIN" ] || return 0
    while IFS=$'\t' read -r pid name; do
        [ -n "${pid:-}" ] || continue
        n=$((n+1)); echo "  [cleanup:$tag] 杀掉 $name (pid $pid)"
    done <<< "$(powershell -NoProfile -ExecutionPolicy Bypass -File "$OURS_PS" -BinDir "$CUDA_BIN_WIN" -Kill 2>/dev/null | tr -d '\r')"
    return 0
}

# 任何退出路径都清一遍：正常结束、exit 1、Ctrl-C、TERM。上次「人一中断脚本进程就变孤儿」就是缺这个。
# HAVE_LOCK 只在自己真的拿到锁之后置 1：并发拒绝（exit 2）那条路径上，$BIN 下跑的正是
# 另一个 verify 的进程，无条件 cleanup 会把它杀掉、再把它的锁删掉——那恰好制造出这把锁要防的假失败。
HAVE_LOCK=0
on_exit() {
    local rc=$?
    trap - EXIT INT TERM
    if [ "$HAVE_LOCK" = "1" ]; then
        cleanup_ours 退出
        cleanup_cuda 退出
        rm -rf "$QLH_TMP/.verify.lock" 2>/dev/null
    fi
    exit "$rc"
}
trap on_exit EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

# ---------------------------------------------------------------- 串行闸门
# 同一台机器同一时间只允许一个 verify：两个并发跑会互抢 93xx、互把对方的进程当残留杀掉，
# 然后双方各报一堆假失败（今天那三次误判里，至少有一次就是并发跑出来的）。
if [ "${QLH_VERIFY_NO_LOCK:-0}" != "1" ]; then
    LOCK=$QLH_TMP/.verify.lock
    if ! mkdir "$LOCK" 2>/dev/null; then
        other=$(cat "$LOCK/pid" 2>/dev/null)
        if [ -n "$other" ] && ! kill -0 "$other" 2>/dev/null; then
            echo "  [cleanup] 发现上次留下的死锁文件（pid $other 已不在），接管" >&2
            rm -rf "$LOCK"; mkdir "$LOCK" 2>/dev/null
        else
            echo "另一个 verify.sh 正在跑（$LOCK，pid ${other:-?}）。同一台机器同一时间只允许一个：" >&2
            echo "  并发跑会互抢 93xx 端口并互相强杀，两边都会报假失败。" >&2
            echo "  确实要并发（不推荐）：QLH_VERIFY_NO_LOCK=1 bash verify/verify.sh ..." >&2
            exit 2
        fi
    fi
    echo $$ > "$LOCK/pid"
    HAVE_LOCK=1
fi

# ---------------------------------------------------------------- 带超时跑我们自己的 exe
# 为什么必须加：split-smoke 每次跑都会在模型/context 建图之后**静默约 2 分钟**再继续——
# 实测 119 s 里 CPU 不涨、4 个线程全是 Wait/UserRequest、读写 I/O 和页错误都是 0，stderr 停在
# 完全相同的一行（第 13073 行，一个 context 的 sched_reserve 之后），然后 5 秒内跑完。也就是说
# 它看起来像挂死，但（至少在这台机器上）会自己恢复；机器还被残留进程挤着时会更久。
# 没有超时的后果：人以为它卡住，Ctrl-C 掉脚本 → bash 死了它还活着，1.37 GB 内存 + 9331 端口一直占着，
# 下一轮就跑出「identity mismatch」「-np 4 超时」「health=000」。
# 注：这台机器的 `timeout` 是 Windows 自带的 timeout.exe（不是 coreutils），不能直接用。
# 注：$! 是 MSYS pid，taskkill 要 Windows pid，所以从 /proc/<pid>/winpid 取。
run_limited() {
    local secs=$1 outf=$2; shift 2
    "$@" >"$outf" 2>&1 &
    local pid=$! waited=0 wp
    while [ "$waited" -lt "$secs" ]; do
        if ! kill -0 "$pid" 2>/dev/null; then
            wait "$pid"
            return $?
        fi
        sleep 1
        waited=$((waited+1))
    done
    wp=$(cat "/proc/$pid/winpid" 2>/dev/null)
    [ -n "$wp" ] && taskkill //F //PID "$wp" >/dev/null 2>&1
    kill -9 "$pid" 2>/dev/null
    wait "$pid" 2>/dev/null
    echo "  [timeout] 超过 ${secs}s 未退出，已强杀：$*" | tee -a "$outf"
    return 124
}

wait_health() {
    local port=$1
    for _ in $(seq 1 40); do
        if [ "$(curl -s -o /dev/null -w '%{http_code}' "http://127.0.0.1:$port/health" 2>/dev/null)" = "200" ]; then
            return 0
        fi
        sleep 2
    done
    return 1
}

ask() {
    curl -s "http://127.0.0.1:$1/v1/chat/completions" -H 'Content-Type: application/json' \
        -d '{"messages":[{"role":"user","content":"What is the capital of France? Answer with one word."}],"max_tokens":512,"temperature":0}' \
        | python -c "import json,sys; d=json.load(sys.stdin); m=d['choices'][0]['message']; print((m.get('reasoning_content') or '') + '|' + (m.get('content') or ''))" 2>/dev/null
}

start_server() {
    local port=$1; shift
    # 直接后台，不用 ( ... & ) 双层 fork：双层 fork 出来的进程脱离 job control，
    # 脚本被 Ctrl-C / kill 时它们不会跟着走，只能靠按端口或名字再杀一遍。
    "$BIN/llama-server.exe" -m "$MODEL" --host 127.0.0.1 --port "$port" -c 4096 -ngl 0 --no-warmup \
          "$@" > "$QLH_TMP/srv_$port.log" 2>&1 &
    # [C5] 不记 $!：清理按「名字 + $BIN 路径」（见 list_ours/kill_ours_tree），这个 pid 没有读者。
}

# MSYS 的 PATH 项按 POSIX 形式更稳；cygpath 不在时退回原样（EXE 一律按绝对路径调用）。
export PATH="$(cygpath -u "$BIN" 2>/dev/null || echo "$BIN"):$PATH"

if [ "$MODE" != "--skip-build" ]; then
    hdr "编译 llama / split-stage / llama-server"
    cleanup_ours 编译前
    CLEAN=""
    [ "$MODE" = "--clean" ] && CLEAN="--clean-first"
    if ! cmd //c "$QLH_BUILD_BAT" $CLEAN > "$QLH_TMP/build.log" 2>&1; then
        bad "编译失败，见 $QLH_TMP/build.log"
        tail -25 "$QLH_TMP/build.log"
        exit 1
    fi
    if grep -qE "error (C|LNK)[0-9]+" "$QLH_TMP/build.log"; then
        bad "编译有 error"
        grep -E "error (C|LNK)[0-9]+" "$QLH_TMP/build.log" | head
        exit 1
    fi
    ok "三目标构建完成$([ -n "$CLEAN" ] && echo '（全量）')"
fi

# [QLH-SPLIT] 无条件清一次：--skip-build 时上面那段不会跑，上一次手工留下的 stage 会一直占着
# 9331（loading 中的 stage 还没 bind，旧版只看 LISTENING 就看不见它），组 4/6/7 就会连到错的段上去
# （实测过：组 10 的 stage bind 失败，客户端却和残留的 relay 握手，报成 identity mismatch）。
# 现在按「镜像名 + $BIN 路径」杀，不监听端口的进程也能清掉。
cleanup_ours 开始

# ---------------------------------------------------------------- 组 1：本地双段 + 三段链
hdr "组 1  本地链逐位（0:20+20:24 与 0:8+8:16+16:24）"
if [ ! -x "$SMOKE" ]; then
    bad "缺 $SMOKE"
else
    # [QLH-SPLIT] 落文件而不是 $(...) 捕获：这个二进制要加载 24 层模型，偶尔会因为上一次构建刚写完
    # dll、或者机器正被别的 35B 进程挤着，起来就失败。用文件至少能在失败时看到它到底说了什么，
    # 而空 stdout 曾经被误判成「逐位不一致」。
    SMOKE_OUT=$QLH_TMP/smoke_group1.log
    if run_limited "$SMOKE_TIMEOUT" "$SMOKE_OUT" "$SMOKE"; then
        out=$(cat "$SMOKE_OUT")
        echo "$out" | grep -q "=== ARGMAX 一致 ===" && ok "双段 argmax 一致" || bad "双段 argmax 不一致"
        echo "$out" | grep -q "\[A\] vs \[B\] hidden 2048 维逐位相同" && ok "head hidden 与整模层输入逐位相同" || bad "head hidden 与基线不同"
        echo "$out" | grep -q "广播生效" && ok "seq_rm 广播到两段" || bad "seq_rm 未广播"
        echo "$out" | grep -q "\[E\] 三段链与整模 logits 逐位相同" && ok "三段链逐位相同" || bad "三段链不一致"
        echo "$out" | grep -q "\[F\] n_stages after rejected attaches = 0" && ok "不衔接的链被拒（-1）" || bad "衔接校验没生效"
        if [ -s "$SMOKE_OUT" ] && ! grep -q "=== ARGMAX 一致 ===" "$SMOKE_OUT"; then
            echo "  （smoke 输出尾部，见 .qlh_tmp/verify/smoke_group1.log）"
            tail -8 "$SMOKE_OUT" | sed 's/^/    /'
        fi
    else
        # 空输出不能算成 5 条「逐位不一致」：那是假的代码 bug，正是今天误判三次的那种报法。
        bad "split-smoke 没跑完（rc=$?），见 .qlh_tmp/verify/smoke_group1.log —— 不把空输出算成「逐位不一致」"
        tail -8 "$SMOKE_OUT" | sed 's/^/    /'
    fi
fi
cleanup_ours 组1

# ---------------------------------------------------------------- 组 2：N=1 直通
hdr "组 2  N=1 直通（整模 vs --split-layer-range 0:24）"
start_server $PORT_BASE
start_server $PORT_N1 --split-layer-range 0:24
if wait_health $PORT_BASE && wait_health $PORT_N1; then
    a=$(ask $PORT_BASE)
    n1=$(ask $PORT_N1)
    if [ -z "$a" ]; then
        bad "整模基线没取到（$PORT_BASE 的请求失败），组 2 无法对照；组 3 会沿用这个空基线并单独报错"
    elif [ "$a" = "$n1" ]; then
        ok "0:24 单段输出与整模逐字节相同（${#a} 字符）"
    else
        bad "0:24 与整模不一致（整模 ${#a} / 单段 ${#n1} 字符）"
    fi
else
    bad "服务没起来，见 .qlh_tmp/verify/srv_*.log"
fi
cleanup_ours 组2

# ---------------------------------------------------------------- 组 3：N=2 经 server
hdr "组 3  N=2 经 llama-server（0:20 + 20:24）"
start_server $PORT_N2 --split-layer-range 0:20 --split-tail-range 20:24 --split-rs-seq $RS_SEQ
if wait_health $PORT_N2; then
    n2=$(ask $PORT_N2)
    if [ -z "$a" ]; then
        # 组 2 没取到整模基线：这不能算「双段链不对」，也不能让 $a 未定义把脚本带走。
        bad "双段链没有对照：组 2 没取到整模基线输出（$PORT_BASE）"
    elif [ "$a" = "$n2" ]; then
        ok "双段链输出与整模逐字节相同"
    else
        bad "双段链与整模不一致（整模 ${#a} / 双段 ${#n2} 字符）"
    fi
else
    bad "服务没起来，见 .qlh_tmp/verify/srv_$PORT_N2.log"
fi
cleanup_ours 组3

# ---------------------------------------------------------------- 组 4：跨进程
hdr "组 4  跨进程（本地 head + 远程 split-stage tail）"
# 不用 ( ... & ) 双层 fork：那样脚本被中断时 stage 会脱离 job control 留下来。
# 直接后台 + 记 pid，等待循环里能判断它是不是已经自己退了（加载失败要报「stage 没起来」，
# 而不是等满 30 s 再让 smoke 连一个空端口、最后报成「链不对」）。
"$BIN/split-stage.exe" -m "$MODEL" --layer-range 20:24 --role tail --rs-seq $RS_SEQ \
      --port $PORT_STAGE -c 512 -b 256 > "$QLH_TMP/stage.log" 2>&1 &
STAGE_PID=$!
stage_up=0
for _ in $(seq 1 30); do
    # [v10] the bind now happens before the model load, so "listening" only means the port is
    # ours. "ready:" is printed once the model, contexts and offload plan are settled -- that is
    # the signal this loop must wait for, otherwise it would race the load.
    grep -q "ready:" "$QLH_TMP/stage.log" 2>/dev/null && { stage_up=1; break; }
    kill -0 "$STAGE_PID" 2>/dev/null || break
    sleep 1
done
if [ "$stage_up" -ne 1 ]; then
    bad "stage 没起来（$PORT_STAGE 未 ready：端口没绑上或模型没加载完）：这是环境/启动问题，不是链本身，见 .qlh_tmp/verify/stage.log"
    tail -5 "$QLH_TMP/stage.log" | sed 's/^/    /'
elif run_limited "$SMOKE_TIMEOUT" "$QLH_TMP/smoke_group4.log" \
        env ONLY_SPLIT=1 QLH_SPLIT_REMOTE=127.0.0.1:$PORT_STAGE "$SMOKE"; then
    if grep -q "argmax=3709" "$QLH_TMP/smoke_group4.log"; then
        ok "远程链 argmax=3709（与整模一致）"
    else
        bad "远程链结果不对"
        grep -E "argmax|role=|failed|error" "$QLH_TMP/smoke_group4.log" "$QLH_TMP/stage.log" | head -5
    fi
else
    bad "远程链 smoke 没跑完（rc=$?），见 .qlh_tmp/verify/smoke_group4.log"
    tail -5 "$QLH_TMP/smoke_group4.log" | sed 's/^/    /'
fi
cleanup_ours 组4

# ---------------------------------------------------------------- 组 5：多 slot
hdr "组 5  多 slot（hybrid + 分段链 + slot 复用）"
# 单独一个脚本，因为它要发长度交替的多个请求才触发那条回滚；组 3 只发一个请求，
# 永远只用一个 slot，这类崩溃在它那里是隐形的。
for np in 2 4; do
    if run_limited "$PY_TIMEOUT" "$QLH_TMP/multislot_np$np.out" \
            python "$QLH_VERIFY_DIR/verify_multislot.py" --np $np --port $((PORT_BASE + 10 + np)) --ctx 8192; then
        ok "-np $np 十个不同长度请求全 200，进程存活，prefix cache 仍生效"
    else
        bad "-np $np 失败（rc=$?），见 .qlh_tmp/verify/multislot_np$np.out"
        tail -6 "$QLH_TMP/multislot_np$np.out"
    fi
    cleanup_ours "组5-np$np"
done

# ---------------------------------------------------------------- 组 6：三段跨进程
hdr "组 6  三段跨进程（本地 head 0:8 + 远程 relay 8:16 + 远程 tail 16:24）"
# 远程 relay 段在修段角色之前从来没工作过：它被判成 tail，返回 logits 而不是 hidden，
# 客户端拿到空 hidden 就保持 carrier 不变，静默算错。这一组是它的第一个端到端证据。
#
# 直接后台 + 记 pid，不用 `( ... & )` 双层 fork：双层 fork 出来的进程脱离 job control，
# 脚本被 Ctrl-C / kill 时它们不会跟着走（组 4 的注释记了这条）。记下 pid 还让等待循环能在
# 段自己退出时早退，而不是白等满 40 s 再让 smoke 连一个空端口、最后报成「链不对」。
"$BIN/split-stage.exe" -m "$MODEL" --layer-range 8:16  --role relay --rs-seq $RS_SEQ \
      --port "$PORT_RELAY" -c 512 -b 256 > "$QLH_TMP/stage_relay.log" 2>&1 &
STAGE_RELAY_PID=$!
"$BIN/split-stage.exe" -m "$MODEL" --layer-range 16:24 --role tail  --rs-seq $RS_SEQ \
      --port "$PORT_TAIL" -c 512 -b 256 > "$QLH_TMP/stage_tail.log" 2>&1 &
STAGE_TAIL_PID=$!
for _ in $(seq 1 40); do
    grep -q "ready:" "$QLH_TMP/stage_relay.log" 2>/dev/null && \
    grep -q "ready:" "$QLH_TMP/stage_tail.log"  2>/dev/null && break
    kill -0 "$STAGE_RELAY_PID" 2>/dev/null || { echo "  [warn] 组 6 的 relay 段已退出，不再等 ready"; break; }
    kill -0 "$STAGE_TAIL_PID"  2>/dev/null || { echo "  [warn] 组 6 的 tail 段已退出，不再等 ready"; break; }
    sleep 1
done
[ -s "$QLH_TMP/stage_relay.log" ] || echo "  [warn] 组 6 的 relay 段一个字都没输出（可能起不来），见 stage_relay.log"
SMOKE6=$QLH_TMP/smoke_group6.log
if run_limited "$SMOKE_TIMEOUT" "$SMOKE6" \
        env QLH_SPLIT_REMOTE2=127.0.0.1:$PORT_RELAY,127.0.0.1:$PORT_TAIL "$SMOKE"; then
    out6=$(cat "$SMOKE6")
    if echo "$out6" | grep -q "logits 全 .* 维: 逐位相同"; then
        ok "三段链全 vocab 逐位相同（$(echo "$out6" | grep -o 'logits 全 [0-9]* 维' | head -1)）"
    else
        bad "三段链不一致"
        echo "$out6" | grep -E "^\[G\]" | head -5
    fi
    if echo "$out6" | grep -qE "seq_rm\(0, 2, -1\) -> 1"; then
        ok "远端 relay/tail 的 KV 都被裁到 pos_max=1"
    else
        bad "远端 KV 没被裁"
        grep -h "OP_SEQ_RM" "$QLH_TMP/stage_relay.log" "$QLH_TMP/stage_tail.log" | tail -4
    fi
    grep -q "OP_SEQ_RM" "$QLH_TMP/stage_relay.log" && ok "relay 段收到 OP_SEQ_RM" || bad "relay 段没收到 OP_SEQ_RM"
    grep -q "OP_SEQ_RM" "$QLH_TMP/stage_tail.log"  && ok "tail 段收到 OP_SEQ_RM"  || bad "tail 段没收到 OP_SEQ_RM"

    # [G2]：checkpoint 往返（只在 QLH_SPLIT_REMOTE2 下跑）。这一组比 [E]/[G] 的 1~3 token
    # 更重要——它跑 20+ token，而 1 token 的逐位对比曾经掩盖过 M-RoPE 位置分叉（p>0 才暴露）。
    echo "$out6" | grep -qE "\[G2\] local: .*逐位相同" && ok "本地链 checkpoint save→restore→重算逐位相同" || bad "本地链 checkpoint 往返不一致"
    echo "$out6" | grep -qE "\[G2\] whole: .*逐位相同" && ok "整模 checkpoint 往返逐位相同（对照）" || bad "整模 checkpoint 往返不一致"
    echo "$out6" | grep -q "remote chain vs whole after checkpoint restore: 逐位相同" && ok "远程链 checkpoint 恢复后与整模逐位相同" || bad "远程链 checkpoint 恢复后与整模不同"
    echo "$out6" | grep -q "restore a .* -> 0 (expect 0)" && ok "清空后恢复被拒（漂移检测）" || bad "漂移检测没生效"
else
    # 超时/崩溃时输出是残缺的，逐个 grep 会刷出 8 条假 FAIL。报一条、贴尾巴。
    bad "三段链 smoke 没跑完（rc=$?），见 .qlh_tmp/verify/smoke_group6.log"
    tail -6 "$SMOKE6" | sed 's/^/    /'
    grep -h "listening\|error\|failed" "$QLH_TMP/stage_relay.log" "$QLH_TMP/stage_tail.log" | tail -4 | sed 's/^/    /'
fi
cleanup_ours 组6

# ---------------------------------------------------------------- 组 7：远程链经 server
hdr "组 7  远程链经 llama-server（--split-remote 两个端点）"
"$BIN/split-stage.exe" -m "$MODEL" --layer-range 8:16  --role relay --rs-seq $RS_SEQ \
      --port "$PORT_RELAY" -c 8192 -b 512 > "$QLH_TMP/stage_relay.log" 2>&1 &
STAGE_RELAY_PID=$!
"$BIN/split-stage.exe" -m "$MODEL" --layer-range 16:24 --role tail  --rs-seq $RS_SEQ \
      --port "$PORT_TAIL" -c 8192 -b 512 > "$QLH_TMP/stage_tail.log" 2>&1 &
STAGE_TAIL_PID=$!
for _ in $(seq 1 40); do
    grep -q "ready:" "$QLH_TMP/stage_relay.log" 2>/dev/null && \
    grep -q "ready:" "$QLH_TMP/stage_tail.log"  2>/dev/null && break
    kill -0 "$STAGE_RELAY_PID" 2>/dev/null || { echo "  [warn] 组 7 的 relay 段已退出，不再等 ready"; break; }
    kill -0 "$STAGE_TAIL_PID"  2>/dev/null || { echo "  [warn] 组 7 的 tail 段已退出，不再等 ready"; break; }
    sleep 1
done
"$BIN/llama-server.exe" -m "$MODEL" --host 127.0.0.1 --port $PORT_N2 -c 8192 -ngl 0 --no-warmup \
      --split-layer-range 0:8 --split-remote "127.0.0.1:$PORT_RELAY" --split-remote "127.0.0.1:$PORT_TAIL" \
      > "$QLH_TMP/srv_remote.log" 2>&1 &
if wait_health $PORT_N2; then
    rm_ok=0
    for u in 48 120 60 160 70; do
        txt=$(python - "$u" << 'PY'
import sys, json
u = int(sys.argv[1])
print(json.dumps([{"role": "user", "content": ("Name the capital of France. " * u)[:u*8]}]))
PY
)
        code=$(curl -s -o /dev/null -w '%{http_code}' "http://127.0.0.1:$PORT_N2/v1/chat/completions" \
            -H 'Content-Type: application/json' \
            -d "{\"messages\":$txt,\"max_tokens\":16,\"temperature\":0}" 2>/dev/null)
        [ "$code" = "200" ] && rm_ok=$((rm_ok+1))
    done
    health_now=$(curl -s -o /dev/null -w '%{http_code}' "http://127.0.0.1:$PORT_N2/health" 2>/dev/null)
    if [ "$rm_ok" -eq 5 ] && [ "$health_now" = "200" ]; then
        ok "5/5 请求 200，进程存活（远程链长度交替）"
    else
        bad "远程链经 server：${rm_ok}/5 通过，health=$health_now"
    fi
    grep -q "OP_SEQ_RM" "$QLH_TMP/stage_relay.log" && ok "远程 server 请求真的裁到了 relay 段" || bad "远程段 KV 没被 server 裁"
else
    bad "远程链 server 没起来，见 .qlh_tmp/verify/srv_remote.log"
    tail -6 "$QLH_TMP/srv_remote.log"
fi
cleanup_ours 组7

# ---------------------------------------------------------------- 组 8：MTP 投机
hdr "组 8  MTP 投机（链上投机 · 链上 MTP 与整模 MTP 逐位）"
# 为什么单独一组：投机路径不经过前面任何一组的代码路径（它要 tail 的逐行 logits、
# 独立的 MTP draft context、每轮回滚 k 个位置）。这条比加速比重要——接受/拒绝逻辑错了
# 不会崩，会静默产出错误 token。
if [ -x "$BIN/split-rows.exe" ]; then
    if run_limited "$SMOKE_TIMEOUT" "$QLH_TMP/smoke_group8.log" "$BIN/split-rows.exe"; then
        out8=$(cat "$QLH_TMP/smoke_group8.log")
        echo "$out8" | grep -q "链上每一行 logits 与整模逐位相同" && ok "链上逐行 logits 与整模逐位相同" || bad "链上逐行 logits 与整模不同"
        echo "$out8" | grep -q "链上 MTP draft 与整模 MTP 逐位相同" && ok "链上 MTP draft 与整模 MTP 逐位相同" || bad "链上 MTP draft 与整模 MTP 不同"
    else
        bad "split-rows 没跑完（rc=$?），见 .qlh_tmp/verify/smoke_group8.log"
        tail -6 "$QLH_TMP/smoke_group8.log" | sed 's/^/    /'
    fi
else
    bad "缺 $BIN/split-rows.exe（--target split-rows）"
fi
cleanup_ours 组8

# ---------------------------------------------------------------- 组 9/10：先过内存闸门
# 闸门的量法（`qlh_available_gb` / `qlh_available_gb_settled`）住在 _qlh_env.sh：
# 组 11 的子脚本 verify_dflash_dev_path.sh 独立跑时也要用同一份口径，不能各写一遍
# （它以前用 FreePhysicalMemory，比这里的 Available MBytes 低约 3 GB，注释却写「同 verify.sh」）。
# 闸门的理由见上面 MIN_FREE_GB 的注释（mmap：整模不需要全占物理内存）。

# ---------------------------------------------------------------- 组 9：DFlash
if [ "$SKIP_BIG" = "1" ]; then
    echo
    echo "=== 组 9 / 组 10 / 组 11 没测：QLH_VERIFY_FAST=1 只跑 2B 的组（1-8 与 12）==="
    echo "  [SKIP] 不计入 PASS、不算通过：这一轮没有测 35B 的 DFlash / DSpark 两组。"
    skip "组 9/10 没测（QLH_VERIFY_FAST=1），不是 PASS"
else
AVAIL_GB=$(qlh_available_gb_settled)
if [ "$AVAIL_GB" -lt "$MIN_FREE_GB" ]; then
    hdr "组 9 / 组 10 没测：可用内存不足"
    skip "组 9/10 没测：可用 ${AVAIL_GB} GB < 闸门 ${MIN_FREE_GB} GB —— 这不是 PASS，这一轮没有测到 35B（DFlash / DSpark）"
    echo "  想真测：关掉占内存的进程再跑；或按实际情况调 QLH_VERIFY_MIN_FREE_GB=${MIN_FREE_GB}。"
    echo "  只想确认 2B 的组没有回归：QLH_VERIFY_FAST=1（组 1-8 与 12 照样全跑）。"
else
echo "  [mem] 可用 ${AVAIL_GB} GB ≥ ${MIN_FREE_GB} GB，组 9/10 照跑"
hdr "组 9  DFlash 草稿模型（8 路 tap 跨两段 · 单 ubatch 逐位）"
# DFlash 的 8 个 target_layers 分布在 head（2/7/12/17/23/28）和 tail（33/38）两段，
# 这个测试直接证明「跨 context 拼出来的 8 路 tap」与整模逐位相同——拼错了会静默喂坏
# draft，而 draft 喂坏了只是接受率下降，不会报错。
if [ -x "$BIN/dflash-taps.exe" ]; then
    if run_limited "$BIG_TIMEOUT" "$QLH_TMP/taps_group9.log" \
            "$BIN/dflash-taps.exe" "$MODEL_DFLASH"; then
        out9=$(cat "$QLH_TMP/taps_group9.log")
        echo "$out9" | grep -q "ALL PASS" && ok "8 路 tap 与 logits 全部逐位相同" || bad "DFlash tap 不一致"
        echo "$out9" | grep -q "tail local 1 == whole 33" && ok "tail 的 tap 映射到正确的源层（33/38）" || bad "tail tap 映射错"
    else
        bad "dflash-taps 没跑完（rc=$?），见 .qlh_tmp/verify/taps_group9.log"
        tail -6 "$QLH_TMP/taps_group9.log" | sed 's/^/    /'
    fi
else
    bad "缺 $BIN/dflash-taps.exe（--target dflash-taps，需要 35B 模型）"
fi
cleanup_ours 组9

# ---------------------------------------------------------------- 组 10：DFlash / DSpark 远程链
hdr "组 10  DFlash / DSpark 远程链（tail 侧 tap 走 OP_LAYER_INP）"
# 为什么单独一组：v9 的 layer-input tap 通道是「远程链能用 DFlash/DSpark」的唯一障碍。组 9 证明的是
# 同一进程内跨 context 的 8 路拼接；这一组证明 tail 那几层的 layer_inp 是从**另一台机器的字节**里读回来的，
# 而且读回来之后与整模逐位相同。
#
# 分两趟跑的原因：整模 23.5 GB 与一个 stage 进程放不进这台机器（31.6 GB）。所以先整模落盘（此时没有
# stage），再起 stage 跑远程链比对。tap 列表从草稿模型的 dflash.target_layers 读，因此 DSpark（5 路）
# 用的是同一条命令。
DFLASH_DRAFT=${QLH_DFLASH_DRAFT:-H:/qlh_models/dflash/qwen36-35b-a3b-dflash-Q4_K_M.gguf}
DSPARK_DRAFT=${QLH_DSPARK_DRAFT:-H:/qlh_models/dspark/Qwen3.6-35B-A3B-DSpark-Q4_K_M.gguf}

remote_tap_group() {
    local name=$1 draft=$2 taps=$3 wire=$4
    local dump=$QLH_TMP/dflash_whole_${name}.bin
    local log_w=$QLH_TMP/taps_${name}_whole.log
    local log_c=$QLH_TMP/taps_${name}_chain.log
    local log_s=$QLH_TMP/stage_${name}.log
    if [ ! -f "$MODEL_DFLASH" ] || [ ! -f "$draft" ]; then
        bad "$name: 缺模型（目标 $MODEL_DFLASH / 草稿 $draft），这一组跳过"
        return
    fi
    # 不能用 `if ! run_limited ...`：`$?` 在 `!` 之后是取反值（恒 0），超时（124）会被
    # 报成 rc=0，日志里就看不出是超时还是崩溃。先存真实退出码再判。
    run_limited "$BIG_TIMEOUT" "$log_w" "$BIN/dflash-taps.exe" "$MODEL_DFLASH" \
        --draft "$draft" --whole-save "$dump"
    local rc=$?
    if [ "$rc" -ne 0 ]; then
        bad "$name: 整模一趟没跑完（rc=$rc），见 $(basename "$log_w")"
        tail -5 "$log_w"
        cleanup_ours "$name-整模"
        return
    fi
    ok "$name: 整模 $taps 路 tap 已落盘（$(basename "$dump")）"

    # 同组 6：直接后台 + 记 pid，不用 `( ... & )` 双层 fork（脱离 job control，中断后留孤儿）。
    "$BIN/split-stage.exe" -m "$MODEL_DFLASH" --layer-range 32:40 --role tail --rs-seq $RS_SEQ \
          --port $PORT_STAGE -c 512 -b 256 > "$log_s" 2>&1 &
    local stage_pid=$!
    local up=0
    for _ in $(seq 1 90); do
        if grep -q "ready:" "$log_s" 2>/dev/null; then up=1; break; fi
        kill -0 "$stage_pid" 2>/dev/null || break
        sleep 2
    done
    if [ "$up" -ne 1 ]; then
        bad "$name: stage 没起来，见 $(basename "$log_s")"
        tail -5 "$log_s"
        cleanup_ours "$name-stage"
        return
    fi
    run_limited "$BIG_TIMEOUT" "$log_c" "$BIN/dflash-taps.exe" "$MODEL_DFLASH" --draft "$draft" \
        --remote-compare "$dump" --endpoint "127.0.0.1:$PORT_STAGE"
    rc=$?
    grep -q "^\[PASS\] $taps 路 tap 全部与整模逐位相同" "$log_c" \
        && ok "$name: $taps 路 tap 全部与整模逐位相同（$(sed -n 's/.*(\([0-9]*\/[0-9]*\)；.*/\1/p' "$log_c" | head -1)）" \
        || bad "$name: tap 拼接与整模不同，见 $(basename "$log_c")"
    grep -q "^\[PASS\] wire 上的 $wire 路 tap 与整模逐位相同" "$log_c" \
        && ok "$name: $wire 路 tap 经 wire 取回后与整模逐位相同" \
        || bad "$name: wire 上的 tap 与整模不同"
    grep -q "^\[PASS\] 远程链 vs 整模 logits 逐位相同" "$log_c" \
        && ok "$name: 远程链 logits 与整模逐位相同（$(grep -o '([0-9]*/[0-9]* 个位置)' "$log_c" | head -1)）" \
        || bad "$name: 远程链 logits 与整模不同"
    grep -q "^\[PASS\] 远程链续写 token 与整模逐 token 相同" "$log_c" \
        && ok "$name: 远程链续写 token 与整模逐 token 相同（$(grep -o '([0-9]*/[0-9]*)' "$log_c" | tail -1)）" \
        || bad "$name: 远程链续写 token 与整模不同"
    grep -q "^\[PASS\] 续写每一步的" "$log_c" \
        && ok "$name: 1 行帧的 tap 也与整模逐位相同（$(sed -n 's/.*逐位相同 (\([0-9]*\) 步)/\1 步/p' "$log_c" | head -1)）" \
        || bad "$name: 1 行帧的 tap 与整模不同"
    grep -q "^\[PASS\] verify 形状帧（" "$log_c" \
        && ok "$name: k=2 verify 帧（3 行）逐行 argmax 与整模相同" \
        || bad "$name: verify 形状帧的逐行 argmax 与整模不同"
    grep -q "^\[PASS\] verify 形状帧的 3 行 x" "$log_c" \
        && ok "$name: k=2 verify 帧的 3 行 x $taps 路 tap 逐位相同" \
        || bad "$name: verify 形状帧的 tap 与整模不同"
    grep -q "^\[PASS\] tap 传输量" "$log_c" \
        && ok "$name: tap 传输字节数与预期一致（$(grep -o '[0-9]* B，实测 [0-9]* B' "$log_c" | head -1)）" \
        || bad "$name: tap 传输字节数不符"
    grep -q "layer-input request" "$log_s" \
        && ok "$name: stage 收到了 layer-input 请求（$(grep -o '[0-9]* ids, [0-9]* owned' "$log_s" | head -1)）" \
        || bad "$name: stage 没收到 layer-input 请求"
    if [ "$rc" -ne 0 ]; then
        tail -12 "$log_c"
    fi
    cleanup_ours "$name"
}

remote_tap_group dflash "$DFLASH_DRAFT" 8 2
remote_tap_group dspark "$DSPARK_DRAFT" 5 1

# ---------------------------------------------------------------- 组 11：DFlash 投机的默认路径
hdr "组 11  DFlash 投机的默认路径（整模 device buffer / 远程链回退 host copies）"
# 为什么必须有这一组：组 9/10 跑的是 tests/dflash-taps.cpp —— 它只 link `llama`，从不调
# llama_set_embeddings_layer_inp_dev（ctx_tgt 没有 device buffer），batch 用 llama_batch_init
# 而不是 llama_batch_ext_add_embd_row（ubatch.embd_row 恒空，graph 走 host embd 分支），
# 所以 common/speculative.cpp 那条 device-buffer 注入路径在验收里**零覆盖**。
# v9 的默认路径在远程链上第一次请求就断言，产品崩而两组全绿，就是这么漏掉的。
# 这一组直接跑产品路径：llama-server + 链 + -md <dflash 草稿>，不设 LLAMA_DFLASH_DEV。
# 判据见 verify_dflash_dev_path.sh 头部：整模必须仍走 device buffer（宽度谓词不能误杀整模）；
# 链必须回退 host copies、无断言，且输出与整模逐字节相同（特征错了 T=0 输出必分叉）。
# 端口用脚本内默认的 9336/9337/9338（都在 93xx，避开本脚本已用的 9331/9332/9341-9343）。
DFDEV=$QLH_VERIFY_DIR/verify_dflash_dev_path.sh
if [ -f "$DFDEV" ]; then
    LOG11=$QLH_TMP/dfdev_group11.log
    # 内存闸门刚过（组 9/10 与它同一道），所以让子脚本跳过自己的闸门；它自己的互斥锁也要让路，
    # 因为 .verify.lock 此刻正被本进程持有。
    # 显式把本脚本用的 35B target / DFlash 草稿传下去：不能靠子脚本的默认值，也绝不能让它
    # 捡到本脚本 export 的 QLH_SPLIT_MODEL（那是组 1-8 的 2B，套上 DFlash 的 extract layer
    # 会在 llama_set_embeddings_layer_inp 断言——第一次并组的实测就是这个失败）。
    # [C1] 计分走结构化文件：子脚本每条判据追加一行 "PASS<TAB>消息" / "FAIL<TAB>消息"，
    # 这里读它，不再从 $LOG11 的文本里 sed 刮 "[PASS]"/"[FAIL]" 前缀（那样计数就绑在
    # 输出措辞上，改一句话就少计一条）。$LOG11 仍保留，供人看细节。
    R11=$QLH_TMP/dfdev_group11.result
    rm -f "$R11"
    run_limited "$BIG_TIMEOUT" "$LOG11" \
        env QLH_DFDEV_UNDER_VERIFY=1 QLH_DFDEV_MIN_FREE_GB=0 QLH_GROUP_RESULT="$R11" \
            QLH_DFDEV_MODEL="$MODEL_DFLASH" QLH_DFDEV_DRAFT="$DFLASH_DRAFT" \
            bash "$DFDEV"
    rc11=$?
    n11=0
    if [ -s "$R11" ]; then
        while IFS=$'\t' read -r v11 msg11; do
            n11=$((n11+1))
            if [ "$v11" = PASS ]; then
                ok "组 11: $msg11"
            else
                bad "组 11: $msg11"
            fi
        done < "$R11"
    fi
    if [ "$n11" -eq 0 ]; then
        bad "组 11 没跑完（rc=$rc11，见 $(basename "$LOG11")）"
        tail -8 "$LOG11" | sed 's/^/    /'
    elif [ "$rc11" -ne 0 ] && [ "$rc11" -ne 1 ]; then
        # 1 = 有 FAIL（上面已逐条计过）；其它值说明脚本自己挂了，不能被当成「8 项里挂了某几条」
        bad "组 11 rc=$rc11（不是 0/1），见 $(basename "$LOG11")"
        tail -8 "$LOG11" | sed 's/^/    /'
    fi
else
    bad "缺 $DFDEV —— 组 9/10 只覆盖 host copies，device-buffer 路径就没有别的覆盖了"
fi
cleanup_ours 组11
fi   # 内存闸门
fi   # SKIP_BIG

# ---------------------------------------------------------------- 组 12：段级 offload（CUDA 构建）
hdr "组 12  段级 offload 正确性（CUDA 构建：整模 vs --split-layer-range 0:24，-ngl $NGL_PARTIAL）"
# 为什么单独一组：组 1-8 全部跑在 build-split 上，那份构建是 GGML_CUDA=OFF —— `-ngl` 被**静默忽略**，
# 所以「把一段里的前 N 层放到 GPU、其余留在 CPU」这条路径在验收里零覆盖。段级 offload 的语义是
# 「层在哪算不改变数值」，判据只有一条：0:24 单段（-ngl N）与整模（-ngl N）逐字节相同。
# 二进制必须来自 CUDA 构建；构建不存在、或构建在但设备不可用，都记 SKIP 并明说「这不是 PASS」。
CUDA_SERVER=$CUDA_BIN/llama-server.exe
if [ ! -x "$CUDA_SERVER" ]; then
    skip "组 12 没测：缺 $CUDA_SERVER（CUDA 构建不存在，QLH_CUDA_BUILD_DIR=$CUDA_BUILD）—— 这不是 PASS，段级 offload 没有覆盖"
else
    CUDA_BIN_WIN=$(win_path "$CUDA_BIN")
    CUDA_LOG_W=$QLH_TMP/cuda_whole.log
    CUDA_LOG_N1=$QLH_TMP/cuda_split024.log
    "$CUDA_SERVER" -m "$MODEL" --host 127.0.0.1 --port "$PORT_CUDA_BASE" -c 4096 \
        -ngl "$NGL_PARTIAL" --no-warmup > "$CUDA_LOG_W" 2>&1 &
    CUDA_PID_W=$!
    "$CUDA_SERVER" -m "$MODEL" --host 127.0.0.1 --port "$PORT_CUDA_N1" -c 4096 \
        -ngl "$NGL_PARTIAL" --no-warmup --split-layer-range 0:24 > "$CUDA_LOG_N1" 2>&1 &
    CUDA_PID_N1=$!
    if wait_health "$PORT_CUDA_BASE" && wait_health "$PORT_CUDA_N1"; then
        cw=$(ask "$PORT_CUDA_BASE")
        cn=$(ask "$PORT_CUDA_N1")
        # 防假绿：CUDA 后端没加载时 `-ngl` 会被静默忽略，两个 server 都在 CPU 上跑，
        # 「逐字节相同」照样成立 —— 但那就什么也没测到。日志里的 `(CUDA0; N/M blocks)`
        # 是「层真的上了卡」的唯一直接证据，N 必须 > 0。
        gpu_w=$(grep -oaE 'CUDA[0-9]+; [0-9]+/[0-9]+ blocks' "$CUDA_LOG_W" 2>/dev/null | head -1)
        gpu_n1=$(grep -oaE 'CUDA[0-9]+; [0-9]+/[0-9]+ blocks' "$CUDA_LOG_N1" 2>/dev/null | head -1)
        if [ -z "$gpu_w" ] || [ -z "$gpu_n1" ]; then
            bad "组 12：-ngl $NGL_PARTIAL 没有真的把层放到 GPU 上（日志里没有 CUDA0; N/M blocks）—— 逐字节相同是在 CPU 上跑出来的，不算段级 offload"
        elif [ -z "$cw" ]; then
            bad "组 12：CUDA 整模没取到输出（$PORT_CUDA_BASE 的请求失败），无法对照"
        elif [ "$cw" = "$cn" ]; then
            ok "段级 offload（-ngl $NGL_PARTIAL；整模 $gpu_w / 单段 $gpu_n1）：0:24 与整模逐字节相同（${#cw} 字符）"
        else
            bad "段级 offload（-ngl $NGL_PARTIAL）：0:24 与整模不一致（整模 ${#cw} / 单段 ${#cn} 字符）"
        fi
    elif grep -qiE 'no CUDA-capable device|CUDA error|failed to initialize CUDA|no usable GPU|out of memory' \
            "$CUDA_LOG_W" "$CUDA_LOG_N1" 2>/dev/null; then
        # 构建在、但驱动/显存不允许：这是环境，不是代码，按 SKIP 记（照组 9/10 的写法，SKIP 不算通过）。
        skip "组 12 没测：CUDA 构建在，但设备不可用（见 cuda_whole.log / cuda_split024.log）—— 这不是 PASS"
        grep -hiE 'no CUDA-capable device|CUDA error|failed to initialize CUDA|no usable GPU|out of memory' \
            "$CUDA_LOG_W" "$CUDA_LOG_N1" 2>/dev/null | head -2 | sed 's/^/    /'
    else
        bad "组 12：CUDA server 没起来，见 .qlh_tmp/verify/cuda_whole.log / cuda_split024.log"
        echo "    pids: whole=$CUDA_PID_W split024=$CUDA_PID_N1"
        tail -6 "$CUDA_LOG_W" | sed 's/^/    /'
        tail -6 "$CUDA_LOG_N1" | sed 's/^/    /'
    fi
    cleanup_cuda 组12
fi

# ---------------------------------------------------------------- 组 13：KVMem 的 KV 虚拟化
hdr "组 13  KVMem 的 KV 虚拟化（本地双段 + MTP + 驱逐）"
# 为什么单独一组：组 1-12 全部跑在不带 `--kvmem` 的构建上。KVMem 把 KV 虚拟化到 host RAM、GPU 只留
# 有界工作集，接在 `llama_model::create_memory`（与分段链是嵌套关系），所以「head 的 memory 换成
# kvmem」这条路径在别处零覆盖。四项判据的理由见 verify/verify_kvmem.py 的模块 docstring。
KVMEM_BUILD=${QLH_KVMEM_BUILD_DIR:-H:/llamacpp-c/build-kvmem2}
KVMEM_ROOT=${LLAMA_KVMEM_ROOT:-$QLH_VERIFY_DIR/../.qlh_tmp/kvmem-probe}
if [ ! -x "$KVMEM_BUILD/bin/llama-server.exe" ]; then
    skip "组 13 没测：缺 $KVMEM_BUILD/bin/llama-server.exe（带 LLAMA_KVMEM=ON 的构建不存在）—— 这不是 PASS，KVMem 没有覆盖"
elif [ ! -d "$KVMEM_ROOT" ]; then
    skip "组 13 没测：LLAMA_KVMEM_ROOT 不存在（$KVMEM_ROOT）—— 这不是 PASS，KVMem 没有覆盖"
else
    KVMEM_LOG=$QLH_TMP/kvmem_group13.log
    QLH_KVMEM_BUILD_DIR="$KVMEM_BUILD" LLAMA_KVMEM_ROOT="$KVMEM_ROOT" \
    QLH_MODEL_2B="$MODEL" QLH_TMP="$QLH_TMP" \
    QLH_VERIFY_PORT_KVMEM=${QLH_VERIFY_PORT_KVMEM:-9361} \
    # run_limited 的签名是 <秒> <输出文件> <命令...>：输出文件必须在第二参数位，
    # 否则它会把命令名当成输出文件（曾经写成 `run_limited 3600 python ...`，
    # 结果创建了一个 0 字节的 `python` 文件、并直接执行 .py 得到 rc=49）。
    run_limited 3600 "$KVMEM_LOG" python "$QLH_VERIFY_DIR/verify_kvmem.py" \
        --build-dir "$KVMEM_BUILD" --model "$MODEL" --kvmem-root "$KVMEM_ROOT" \
        --tmp "$QLH_TMP" --port ${QLH_VERIFY_PORT_KVMEM:-9361}
    KVMEM_RC=$?
    # 逐条转记账：脚本自己打 [PASS]/[FAIL]，这里按行转成 ok/bad，项数才对得上 57
    while IFS= read -r line; do
        case "$line" in
            *"[PASS]"*) ok "$(echo "$line" | sed 's/^ *\[PASS\] *//')" ;;
            *"[FAIL]"*) bad "$(echo "$line" | sed 's/^ *\[FAIL\] *//')" ;;
        esac
    done < "$KVMEM_LOG"
    # 防假绿：脚本必须真的打出 4 条 PASS/FAIL 记账，否则说明它中途崩了
    KVMEM_LINES=$(grep -cE '\[(PASS|FAIL)\]' "$KVMEM_LOG" 2>/dev/null || echo 0)
    if [ "$KVMEM_LINES" -eq 0 ]; then
        bad "组 13：脚本没打出任何 [PASS]/[FAIL]（rc=$KVMEM_RC），见 .qlh_tmp/verify/kvmem_group13.log"
        tail -8 "$KVMEM_LOG" | sed 's/^/    /'
    elif [ "$KVMEM_RC" -ne 0 ] && [ "$(grep -c '\[FAIL\]' "$KVMEM_LOG")" -eq 0 ]; then
        bad "组 13：脚本退出码 $KVMEM_RC 但没打出 [FAIL]（中途崩了）"
        tail -8 "$KVMEM_LOG" | sed 's/^/    /'
    fi
    cleanup_cuda 组13
fi

hdr "结果"
# 项数断言：「57 项」这个数字被 docs/deploy/BUILD.md、docs/deploy/PATCHES.md、docs/REPORT.md 引用，
# 但以前没有任何地方守它——某条判据因日志措辞改动而静默不匹配时，总数会变少，而
# 「57 项全过」照样成立。期望值：组 1-8 = 22、组 9/10 = 2+20、组 11 = 8、组 12 = 1、组 13 = 4；
# 不带 --skip-build 时另有「三目标构建完成」1 项（58）。
# 只在**没有 SKIP** 的完整跑上断言：SKIP 时少跑的是整组（组 9/10/11/12），
# 与 skip() 记的 1 条不是一比一，硬算会误报。
EXPECT_PASS=57
[ "$MODE" = "--skip-build" ] || EXPECT_PASS=$((EXPECT_PASS + 1))
# 只在「全绿且无 SKIP」时断言：环境缺失（已 FAIL）时不再叠一条噪声；
# 而「某组静默少跑」的特征恰好是 FAIL=0 且 PASS 变少，这条正是抓它的。
if [ "$SKIP" -eq 0 ] && [ "$FAIL" -eq 0 ] && [ "$PASS" -ne "$EXPECT_PASS" ]; then
    bad "项数变了：PASS=$PASS，期望 $EXPECT_PASS（有判据漏跑，或被测日志的措辞改了）"
fi
echo "  PASS=$PASS  FAIL=$FAIL  SKIP=$SKIP"
if [ "$SKIP" -gt 0 ]; then
    echo "  !! SKIP=$SKIP 表示有组**根本没被测到**：它们既不算通过也不算失败。"
    echo "     FAIL=0 只说明被测过的那些组没有退化；这一轮不能当成完整验收。"
fi
# 收尾清理放在退出之前（trap 里还有一次兜底）：脚本返回的那一刻 tasklist 里就不该再有我们的进程。
cleanup_ours 收尾
[ "$FAIL" -eq 0 ] || exit 1