#!/bin/bash
# [QLH-SPLIT] v8 远程 MTP draft 的验收：远程 tail 段与「整模 + MTP」逐 token 对比。
#
# 为什么单独一个脚本：这一组必须有一个**和 split-rows 的 head 分界一致**的 tail 段在跑
# （split-rows 的 head 是 0:20，所以 tail 段要 20:24 --role tail）。verify.sh 的组 6/7 用的是
# 8:16 / 16:24，形状对不上，混在一起会让「远程 MTP」这件事说不清是谁在验证谁。
#
# 判据（打印 [F] 那几行）：
#   1. wire 上回传的 h_nextn（最后一行的）与整模 h_nextn 逐位相同；
#   2. 远程 tail 对同一 (h, token, pos) 跑 3 步 MTP，得到的 3 个 token 与整模 + MTP 逐个相同。
#
# 用法：
#   bash verify/verify_mtp.sh            # 用 2B 模型（默认）
#   QLH_SPLIT_MODEL=... bash verify/verify_mtp.sh
# 退出码 0 = 全过。只杀 93xx 上的监听，9103/9104/9106 绝不碰。

set -u

source "$(dirname "${BASH_SOURCE[0]}")/_qlh_env.sh"
ROOT=$QLH_SRC_DIR
MODEL=${QLH_SPLIT_MODEL:-H:/qlh_models/qwen3-5-2b-gguf/qwen35-2b-Q4_K_M.gguf}
PORT_STAGE=9331
PORT_OTHER=9399
STAGE_LOG=$QLH_TMP/stage_mtp.log

PASS=0
FAIL=0
ok()  { echo "  [PASS] $*"; PASS=$((PASS+1)); }
bad() { echo "  [FAIL] $*"; FAIL=$((FAIL+1)); }

kill_ours() {
    for pid in $(netstat -ano 2>/dev/null | grep -E 'LISTENING' | grep -E ':93[0-9][0-9][[:space:]]' \
                 | awk '{print $NF}' | tr -d '\r' | sort -u); do
        taskkill //F //PID "$pid" >/dev/null 2>&1
    done
    sleep 2
}

# 等到日志里出现某一行；超时返回 1。
#
# [v11] 这里以前等的是端口 LISTENING。v10 把 split-stage 的 bind 提到了模型加载之前，端口一绑上
# 就 LISTENING，所以那个信号只说明「端口是我们的」，一行权重都还没读。用它当就绪信号，后面的
# grep 会和加载赛跑：同一次运行里 has_nextn=1 明明写了 3 处，脚本却判 FAIL。
# 现在等的是 split-stage 自己打的 "[split-stage] ready:"（模型、ctx 与 offload 计划都已落定）。
# 等 stage 就绪的超时。可用 QLH_STAGE_READY_TIMEOUT 覆盖（用来验证「加载没完时判 FAIL 而不是挂住」，
# 不是常规旋钮）。
STAGE_READY_TIMEOUT=${QLH_STAGE_READY_TIMEOUT:-240}

wait_log() { # <log> <ERE> <timeout_s>
    local log=$1 pat=$2 timeout=$3
    for _ in $(seq 1 "$timeout"); do
        grep -qE "$pat" "$log" 2>/dev/null && return 0
        sleep 1
    done
    # 边界：最后一次 sleep 期间可能刚好写完这一行（热缓存下 2B 的 ready 就在 1.0-1.4 s 之间）。
    # 再确认一次，免得「刚好就绪」被记成超时；这不放宽判定，读的还是同一行日志。
    grep -qE "$pat" "$log" 2>/dev/null && return 0
    return 1
}

if [ ! -x "$BIN/split-rows.exe" ] || [ ! -x "$BIN/split-stage.exe" ]; then
    echo "missing $BIN/split-rows.exe or split-stage.exe; build with --target split-rows split-stage"
    exit 1
fi

kill_ours

echo
echo "=== 远程 MTP draft：tail 20:24 在 $PORT_STAGE ==="
# [v11] 先截断日志（父 shell 做，立即生效）。不截断的话循环的第一次 grep 可能读到上一轮跑
# 留下的 ready: —— 那是假 PASS，比假 FAIL 更伤：它放行的是一个还没加载完的 stage。
: > "$STAGE_LOG"
( "$BIN/split-stage.exe" -m "$MODEL" --layer-range 20:24 --role tail --rs-seq "$QLH_RS_SEQ" \
      --port $PORT_STAGE -c 512 -b 256 > "$STAGE_LOG" 2>&1 & )

# 判据一：tail 段真的加载完了。ready: 是 split-stage 在模型/ctx/offload 都就绪、进 accept 循环
# 之前打的那一行；端口绑上不算（v10 把 bind 提到了加载之前）。
if wait_log "$STAGE_LOG" '^\[split-stage\] ready:' "$STAGE_READY_TIMEOUT"; then
    ok "tail 段 ready（$STAGE_LOG）"
else
    bad "tail 段 ${STAGE_READY_TIMEOUT}s 内没打出 ready:（没加载完或已经退出）"
    tail -12 "$STAGE_LOG"
    kill_ours
    exit 1
fi

# 判据二：阶段自报的能力。identity: 是无条件打印的（has_nextn 是 0 还是 1 都写），所以「等它
# 出现」等于「这一行已经写完」；超时才是真的没有。不能用「重试 N 次就算过」——那样真缺
# NextN 层也会被放过。
if wait_log "$STAGE_LOG" '^\[split-stage\] identity:' 30; then
    if grep -q "has_nextn=1" "$STAGE_LOG"; then
        ok "tail 段在 OP_HELLO 里报 has_nextn=1"
    else
        bad "tail 段的 identity 行报 has_nextn=0（模型没有 NextN 层，MTP draft 不可能工作）"
        grep -a "has_nextn\|NextN" "$STAGE_LOG" | head -3
    fi
else
    bad "ready: 之后 30 s 内没打出 identity: 行（stage 加载完了但日志没走完）"
    tail -12 "$STAGE_LOG"
fi

echo
out=$(QLH_SPLIT_REMOTE=127.0.0.1:$PORT_STAGE "$BIN/split-rows.exe" "$MODEL" 2>&1)
echo "$out" | grep -E "^(tokens|  wire|  draft\[|\[PASS\]|\[FAIL\]|\[SKIP\])" | sed 's/^/    /'

if echo "$out" | grep -q "wire h_nextn(last row): 与整模 h_nextn 逐位相同"; then
    ok "wire 上的 h_nextn 与整模逐位相同"
else
    bad "wire 上的 h_nextn 与整模不同"
fi
if echo "$out" | grep -q "\[PASS\] 远程 tail 的 3 个 MTP draft 与整模 MTP 逐个相同"; then
    ok "远程 MTP draft 逐个相同（3/3）"
else
    bad "远程 MTP draft 与整模 MTP 不同"
    echo "$out" | grep -E "draft\[|remote mtp draft" | sed 's/^/      /'
fi

# 反证：tail 段的日志里必须真的收到过 OP_MTP_DRAFT，否则上面比的是别的东西
if grep -q "first OP_MTP_DRAFT" "$STAGE_LOG"; then
    ok "tail 段日志确认跑过 OP_MTP_DRAFT（$(grep -a 'first OP_MTP_DRAFT' "$STAGE_LOG" | head -1 | sed 's/.*first OP_MTP_DRAFT: //')）"
else
    bad "tail 段日志里没有 OP_MTP_DRAFT"
fi

# 协议版本镜像：远端 stage 只在 PROTO_VER 相同的情况下接受第一帧
if grep -q "protocol version mismatch" "$STAGE_LOG"; then
    bad "协议版本不匹配（PROTO_VER 两侧不是同一份源码）"
else
    ok "协议版本一致"
fi

kill_ours

echo
echo "=== 结果 ==="
echo "  PASS=$PASS  FAIL=$FAIL"
[ "$FAIL" -eq 0 ] || exit 1
