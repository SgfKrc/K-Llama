#!/bin/bash
# [QLH-SPLIT] v8: 远程链 MTP 投机的端到端等价 + 三组速度对比（本机回环）。
#
# 目的（硬验收里第 3 条）：远程链 + MTP 投机（temp=0）与**不投机**逐字节相同。投机只改变
# 「谁来猜」，不改变「采出什么 token」—— 一旦投机路径把位置、h 链或接受逻辑弄错，输出就会
# 从某个 token 开始分叉，而 tok/s 看起来一切正常。
#
# 跑四组（同一 prompt / 同一 temp=0 / 同一模型）：
#   A 远程链，不投机
#   B 远程链，ngram 投机（v7 已有的弱 draft）
#   C 远程链，MTP 投机（v8）
#   D 整模（不分段），MTP 投机 —— 判据的参照
#
# 判据（v9 改写）：**C 与 D 逐字节相同**，即「分段不改变投机的结果」。这是这条链唯一该保证的
# 性质：同样的 k+1 行验证、同样的 batch 形状，head 的 h 与整模的前缀逐位相同（verify.sh 组 1），
# tail 的逐行 argmax 与整模的 logits 逐位相同（verify.sh 组 8）。
#
# 「投机 == 不投机」这条旧判据**做不到**，不是这条链的 bug：llama.cpp 本身一次 forward k+1 行
# 与逐行 forward 的第 0 行就不逐位相同（v9 实测：整模、n_ubatch=512、2 行 vs 1 行，h 的 2048 个
# 元素全部不同，max|Δ|=0.27，norm 40.84 vs 40.65；同一情形跑两遍 max|Δ|=0，所以是确定性的 kernel
# 形状差异，不是抖动）。argmax 在 top-1/top-2 差小于这个漂移的位置翻转，之后全文分叉 —— 所以
# A 与 C 是否相同只取决于 prompt 有没有踩到近似并列。把它当硬判据，测的就不是这条链，而是
# 「这个 prompt 的 logits 有没有近似并列」。
# 因此 A 与 C 的差异改为**信息行**（打印出来、不判 PASS/FAIL），并给出第一次分歧的 token 位置。
#
# 用法：bash verify/verify_mtp_e2e.sh [--skip-build]
#   QLH_E2E_MODEL   模型（默认 2B）
#   QLH_HEAD_RANGE  head 段（默认 0:16）
#   QLH_TAIL_RANGE  tail 段（默认 16:24）
# 只杀 93xx；9103/9104/9106 绝不碰。

set -u

source "$(dirname "${BASH_SOURCE[0]}")/_qlh_env.sh"
ROOT=$QLH_SRC_DIR
QLH_BUILD_BAT=${QLH_BUILD_BAT:-$QLH_SRC_DIR/.qlh_tmp/build.bat}
[ -f "$QLH_BUILD_BAT" ] || QLH_BUILD_BAT=$QLH_VERIFY_DIR/build.bat
MODEL=${QLH_E2E_MODEL:-H:/qlh_models/qwen3-5-2b-gguf/qwen35-2b-Q4_K_M.gguf}
HEAD_RANGE=${QLH_HEAD_RANGE:-0:16}
TAIL_RANGE=${QLH_TAIL_RANGE:-16:24}
HEAD_HI=${HEAD_RANGE#*:}
STAGE_PORT=9331
SRV_PORT=9344
PROMPT=${QLH_E2E_PROMPT:-"List the first ten prime numbers, separated by commas."}
MAX_TOK=${QLH_E2E_MAX_TOK:-48}
# 投机深度：两组必须一样，否则 C 与 D 比的是两件事。
SPEC_K=${QLH_E2E_SPEC_K:-2}

# 等 stage 就绪的超时（可用 QLH_STAGE_READY_TIMEOUT 覆盖，只为验证超时路径，不是常规旋钮）
STAGE_READY_TIMEOUT=${QLH_STAGE_READY_TIMEOUT:-240}

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

wait_health() {
    for _ in $(seq 1 60); do
        [ "$(curl -s -o /dev/null -w '%{http_code}' "http://127.0.0.1:$SRV_PORT/health" 2>/dev/null)" = "200" ] && return 0
        sleep 1
    done
    return 1
}

ask() {
    curl -s "http://127.0.0.1:$SRV_PORT/v1/chat/completions" -H 'Content-Type: application/json' \
        -d "{\"messages\":[{\"role\":\"user\",\"content\":\"$PROMPT\"}],\"max_tokens\":$MAX_TOK,\"temperature\":0}" \
        | python -c "import json,sys
d=json.load(sys.stdin)
m=d['choices'][0]['message']
print((m.get('reasoning_content') or '') + '|' + (m.get('content') or ''))
print(d['usage']['completion_tokens'])" 2>/dev/null
}

# start_whole <spec-type|none> <logfile>：同一个模型不分段（判据 D 的参照）
start_whole() {
    local spec=$1
    local log=$2
    local extra=()
    if [ "$spec" != "none" ]; then
        extra=(--spec-type "$spec" --spec-draft-n-max "$SPEC_K")
    fi
    ( "$BIN/llama-server.exe" -m "$MODEL" --host 127.0.0.1 --port $SRV_PORT -c 1024 -ngl 0 --no-warmup \
          -b 256 -ub 256 "${extra[@]}" > "$log" 2>&1 & )
    wait_health
}

# start_chain <spec-type|none> <logfile>
start_chain() {
    local spec=$1
    local log=$2
    local extra=()
    if [ "$spec" != "none" ]; then
        extra=(--spec-type "$spec" --spec-draft-n-max "$SPEC_K")
    fi
    # [v11] 等 stage 的 ready:，不再用固定 sleep 8 赌加载时间（2B 够、35B 必假 FAIL）。
    # 端口 bind 在加载之前，所以「端口可连」不是就绪。
    # [v11] 先截断：否则第一次 grep 可能读到上一轮的 ready:（假 PASS）。
    : > "${log}.stage"
    "$BIN/split-stage.exe" -m "$MODEL" --layer-range "$TAIL_RANGE" --role tail --rs-seq "$QLH_RS_SEQ" \
          --port $STAGE_PORT -c 1024 -b 256 > "${log}.stage" 2>&1 &
    local stage_pid=$!
    local t=0
    while [ "$t" -lt "$STAGE_READY_TIMEOUT" ]; do
        grep -q '^\[split-stage\] ready:' "${log}.stage" 2>/dev/null && break
        kill -0 "$stage_pid" 2>/dev/null || break
        sleep 1
        t=$((t + 1))
    done
    # 边界：最后一次 sleep 期间可能刚好写完 ready:（热缓存下 2B 约 1.0-1.4 s）。
    if ! grep -q '^\[split-stage\] ready:' "${log}.stage" 2>/dev/null; then
        echo "      [stage] ${STAGE_READY_TIMEOUT}s 内没有 ready:（pid=$stage_pid），日志尾：" >&2
        tail -8 "${log}.stage" 2>/dev/null | sed 's/^/      /' >&2
        return 1
    fi
    ( "$BIN/llama-server.exe" -m "$MODEL" --host 127.0.0.1 --port $SRV_PORT -c 1024 -ngl 0 --no-warmup \
          --split-layer-range "$HEAD_RANGE" --split-remote 127.0.0.1:$STAGE_PORT "${extra[@]}" \
          > "$log" 2>&1 & )
    wait_health
}

run_case() {
    local name=$1
    local spec=$2
    local mode=${3:-chain}
    local log=$QLH_TMP/e2e_${name}.log
    kill_ours
    local started=0
    if [ "$mode" = "whole" ]; then
        start_whole "$spec" "$log" && started=1
    else
        start_chain "$spec" "$log" && started=1
    fi
    if [ "$started" != "1" ]; then
        bad "$name: server 没起来（$log）"
        tail -5 "$log" | sed 's/^/      /'
        return 1
    fi
    local t0=$(date +%s%N)
    local out
    out=$(ask)
    local t1=$(date +%s%N)
    local ms=$(( (t1 - t0) / 1000000 ))
    local text
    text=$(echo "$out" | head -n -1)
    local ntok
    ntok=$(echo "$out" | tail -n 1)
    echo "  $name: ${ntok:-?} tokens, ${ms} ms$([ "${ntok:-0}" -gt 0 ] 2>/dev/null && echo ", $(python -c "print(f'{${ntok}/(${ms}/1000):.2f}')" 2>/dev/null) tok/s")"
    echo "$text" > "$QLH_TMP/e2e_${name}.txt"
    kill_ours
}

echo
echo "=== 远程链 MTP 端到端等价：head $HEAD_RANGE + 远程 tail $TAIL_RANGE of $(basename "$MODEL") ==="

run_case base none
run_case ngram ngram-simple
run_case mtp draft-mtp
run_case whole_mtp draft-mtp whole

echo
echo "=== 判据：分段不改变投机的结果（C 与 D 逐字节相同） ==="
if cmp -s "$QLH_TMP/e2e_mtp.txt" "$QLH_TMP/e2e_whole_mtp.txt"; then
    ok "链上 MTP 与整模 MTP 逐字节相同（k=$SPEC_K，同样 k+1 行验证）"
else
    bad "链上 MTP 与整模 MTP 不同（分段改变了答案，这才是真 bug）"
    diff <(head -c 400 "$QLH_TMP/e2e_mtp.txt") <(head -c 400 "$QLH_TMP/e2e_whole_mtp.txt") \
        | head -6 | sed 's/^/      /'
fi

# 参照组自己必须真的在投机：整模的 --spec-type 若被静默忽略，D 就是 A，上一条断言会变成
# 「链上 MTP == 不投机」，通过得毫无意义。
if grep -q "draft acceptance" "$QLH_TMP/e2e_whole_mtp.log"; then
    ok "整模 MTP 参照组真的在投机（$(grep -a 'draft acceptance' "$QLH_TMP/e2e_whole_mtp.log" | head -1 | sed 's/.*draft/draft/')）"
else
    bad "整模 MTP 参照组没在投机（D 等于 A，上一条断言失去意义）"
fi

echo
echo "=== 信息行：投机 vs 不投机（不做断言，原因见文件头） ==="
for name in ngram mtp; do
    if cmp -s "$QLH_TMP/e2e_base.txt" "$QLH_TMP/e2e_${name}.txt"; then
        echo "  [INFO] $name 与不投机逐字节相同"
    else
        echo "  [INFO] $name 与不投机不同 —— 批量验证（k+1 行）与逐行 decode 的 kernel 形状不同，"
        echo "         logits 有 1e-2..1e0 的确定性漂移，argmax 在近似并列处翻转；整模的"
        echo "         draft-mtp 也有同样的漂移（这就是 D 必须当参照的原因）"
    fi
done

echo
echo "=== 投机是否真的在跑（stage / server 日志） ==="
if grep -q "first OP_MTP_DRAFT" "$QLH_TMP/e2e_mtp.log.stage"; then
    ok "MTP case: stage 真的收到 OP_MTP_DRAFT（$(grep -a 'first OP_MTP_DRAFT' "$QLH_TMP/e2e_mtp.log.stage" | head -1 | sed 's/.*token=/token=/')）"
else
    bad "MTP case: stage 没收到 OP_MTP_DRAFT（投机没跑起来，等价性证明无意义）"
fi
if grep -q "first OP_DECODE verify frame" "$QLH_TMP/e2e_mtp.log.stage"; then
    ok "MTP case: 走的是逐行验证通道（每行一个 argmax，不是整行 logits）"
else
    bad "MTP case: 没走逐行验证通道"
fi
if grep -q "MTP drafts will run in the stage" "$QLH_TMP/e2e_mtp.log"; then
    ok "MTP case: head 确认远端 tail 有 MTP draft（ctx_tail_remote 生效）"
else
    bad "MTP case: head 没识别到远端 MTP"
fi
if grep -q "speculative decoding is off for this request" "$QLH_TMP/e2e_mtp.log"; then
    bad "MTP case: 请求被判定为不可投机（逐行通道闸门）"
else
    ok "MTP case: 请求没有被闸门挡掉"
fi

echo
echo "=== 结果 ==="
echo "  PASS=$PASS  FAIL=$FAIL"
[ "$FAIL" -eq 0 ] || exit 1
