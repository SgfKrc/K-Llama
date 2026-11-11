#!/bin/bash
# 用例：DFlash 投机在「远程链 + 默认路径」下不崩（**不设** LLAMA_DFLASH_DEV）。
#
# 为什么要单独一条：verify.sh 的组 9/10 跑的是 tests/dflash-taps.cpp，它只走 host copies ——
# 从不调 llama_set_embeddings_layer_inp_dev（ctx_tgt 没有 device buffer）、用 llama_batch_init
# 的普通 batch（不加 embd_row，graph 走 build_inp_target_features 的 inp.embd 分支）、也不经过
# common/speculative.cpp 的 DFlash impl。v9 的 device-buffer 注入路径只存在于
# 「llama-server + 分段链 + -md <dflash 草稿>」这条线上，所以只能在这条线上验收。
#
# 判据（两半都要）：
#   1. 整模 + DFlash：日志出现 "via device buffer"。
#      修链上的回退很容易顺手把 device 路径整个关掉，那样整模会静默退化回 host copies；
#      这一半把「整模仍然走 device buffer」钉住（宽度谓词的另一个方向）。
#   2. 远程链 + DFlash：同 prompt 的输出与整模**逐字节相同**，日志出现 "via host copies"、
#      没有 GGML_ASSERT、进程存活。
#      「逐字节相同」同时证明回退后喂进 draft KV 的特征是对的：特征错了采样分布就变，
#      输出不可能仍逐字节相同。
#
# 依赖：35B target + DFlash 草稿（同 verify.sh 组 9/10），内存闸门也同 verify.sh。
#       target 由 QLH_DFDEV_MODEL 给、草稿由 QLH_DFDEV_DRAFT 给；verify.sh 调用时显式传
#       它自己的 MODEL_DFLASH / DFLASH_DRAFT，不依赖两边默认值恰好一致。
# 端口：QLH_DFDEV_PORT_STAGE=9336 / QLH_DFDEV_PORT_WHOLE=9337 / QLH_DFDEV_PORT_CHAIN=9338。
#       都在 93xx 里，且避开 verify.sh 的 9331/9332/9341-9343；绝不碰 9103/9104/9106。
# 两种用法：
#   1. 独立跑：bash verify/verify_dflash_dev_path.sh
#   2. 被 verify.sh 当**组 11** 调用：父进程正持有 .verify.lock，所以由它传
#      QLH_DFDEV_UNDER_VERIFY=1（跳过 .verify.lock 检查，只查自己的锁）和
#      QLH_DFDEV_MIN_FREE_GB=0（内存闸门由 verify.sh 在组 9/10 前已经过）。
#      输出格式不变："  [PASS] ..." / "  [FAIL] ..."；另外**如果**父进程给了
#      QLH_GROUP_RESULT，每条判据还会追加一行 "PASS<TAB>消息" / "FAIL<TAB>消息" 到那个文件，
#      verify.sh 读它计分（[C1] 不再 sed 解析本脚本的文本输出）。独立跑时不传就没有这个文件。
# 退出码：0 = 两半都过；1 = 有 FAIL；2 = 另一个 verify.sh / 本脚本正在跑。

set -u

source "$(dirname "${BASH_SOURCE[0]}")/_qlh_env.sh"
ROOT=$QLH_SRC_DIR
QLH_BUILD_BAT=${QLH_BUILD_BAT:-$QLH_SRC_DIR/.qlh_tmp/build.bat}
[ -f "$QLH_BUILD_BAT" ] || QLH_BUILD_BAT=$QLH_VERIFY_DIR/build.bat
# 默认值取 verify.sh 的 MODEL_DFLASH（组 9/10 用的那一个），这样独立跑和并进 verify.sh
# 跑的是同一个目标模型；要换就显式给 QLH_DFDEV_MODEL。
# 注意：**不要**用 QLH_SPLIT_MODEL —— verify.sh 会把它 export 成组 1-8 的 2B 模型。
# DFlash 草稿的 extract layer 是 40 层模型里的 2..38，套到 2B 上会在
# llama_set_embeddings_layer_inp 直接断言（实测 llama-context.cpp:1481）。所以这里只认
# QLH_DFDEV_MODEL，由 verify.sh 在调用时显式传它自己的 MODEL_DFLASH。
MODEL=${QLH_DFDEV_MODEL:-H:/qlh_models/qwen3-6-35b-a3b-opus47-distill-apex-mtp/Qwen3.6-35B-A3B-Claude-4.7-Opus-Reasoning-Distilled-APEX-MTP-I-Quality.gguf}
DRAFT=${QLH_DFDEV_DRAFT:-H:/qlh_models/dflash/qwen36-35b-a3b-dflash-Q4_K_M.gguf}
PORT_STAGE=${QLH_DFDEV_PORT_STAGE:-9336}
PORT_WHOLE=${QLH_DFDEV_PORT_WHOLE:-9337}
PORT_CHAIN=${QLH_DFDEV_PORT_CHAIN:-9338}
MIN_FREE_GB=${QLH_DFDEV_MIN_FREE_GB:-20}
# 一次请求就够：错的特征第一轮抽样就偏，24 个 token（T=0）足以让输出分叉。
N_PREDICT=24
PROMPT="Explain in a few paragraphs how a B-tree maintains balance when keys are inserted, and why a large branching factor matters for disk I/O."

LOG_WHOLE=$QLH_TMP/dfdev_whole.log
LOG_CHAIN=$QLH_TMP/dfdev_chain.log
LOG_STAGE=$QLH_TMP/dfdev_stage.log

PASS=0
FAIL=0
# [C1] 机器可读的判据结果，供父脚本（verify.sh 组 11）计分：每行 "verdict<TAB>message"。
# 之前父脚本用 sed 从本脚本的 "[PASS]"/"[FAIL]" 文本里刮消息，父子两套计数就绑在了
# 输出措辞上——改一句话就少计一条。现在计数走这个文件，文本输出只给人看。
GROUP_RESULT=${QLH_GROUP_RESULT:-}
_group_log() { [ -z "$GROUP_RESULT" ] || printf '%s\t%s\n' "$1" "$2" >> "$GROUP_RESULT"; }
ok()  { echo "  [PASS] $*"; PASS=$((PASS+1)); _group_log PASS "$*"; }
bad() { echo "  [FAIL] $*"; FAIL=$((FAIL+1)); _group_log FAIL "$*"; }
hdr() { echo; echo "=== $* ==="; }

win_path() { printf '%s' "$1" | sed -E 's#^/([A-Za-z])/#\1:/#'; }
OURS_PS=$(win_path "$QLH_VERIFY_DIR/ours_procs.ps1")
BIN_WIN=$(win_path "$BIN")

kill_ours_tree() { # 只杀「可执行文件在 $BIN 下」的进程，见 ours_procs.ps1
    powershell -NoProfile -ExecutionPolicy Bypass -File "$OURS_PS" -BinDir "$BIN_WIN" -Kill 2>/dev/null | tr -d '\r'
}

cleanup_ours() {
    local tag=${1:-收尾} pid name n=0 i
    while IFS=$'\t' read -r pid name; do
        [ -n "${pid:-}" ] || continue
        n=$((n+1)); echo "  [cleanup:$tag] 杀掉 $name (pid $pid)"
    done <<< "$(kill_ours_tree)"
    for i in $(seq 1 15); do
        if [ -z "$(powershell -NoProfile -ExecutionPolicy Bypass -File "$OURS_PS" -BinDir "$BIN_WIN" -List 2>/dev/null | tr -d '\r')" ]; then
            [ "$n" -gt 0 ] && echo "  [cleanup:$tag] 已清 $n 个进程（等待 ${i}s）"
            return 0
        fi
        sleep 1
    done
    echo "  [cleanup:$tag] 警告：15 s 后仍有残留进程"
    return 0
}

unlock() {
    trap - EXIT INT TERM
    cleanup_ours 退出
    rm -rf "$QLH_TMP/.verify_dflash_dev.lock" 2>/dev/null
    local rc=${1:-0}
    exit "$rc"
}

trap 'unlock $?' EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

# 与 verify.sh 互斥：两边都用 93xx，并发跑会互相当成残留进程杀掉，各报一堆假失败。
# 但被 verify.sh 当组 11 调用时，.verify.lock 正是父进程持有的那把，必须跳过。
LOCKS="$QLH_TMP/.verify_dflash_dev.lock"
if [ "${QLH_DFDEV_UNDER_VERIFY:-0}" != "1" ]; then
    LOCKS="$QLH_TMP/.verify.lock $LOCKS"
fi
for other_lock in $LOCKS; do
    if [ -d "$other_lock" ]; then
        other=$(cat "$other_lock/pid" 2>/dev/null)
        if [ -n "$other" ] && kill -0 "$other" 2>/dev/null; then
            echo "另一个验收正在跑（$other_lock，pid $other）。" >&2
            exit 2
        fi
        rm -rf "$other_lock"
    fi
done
mkdir "$QLH_TMP/.verify_dflash_dev.lock" 2>/dev/null && echo $$ > "$QLH_TMP/.verify_dflash_dev.lock/pid"

cleanup_ours 开始

wait_health() { # $2 = 可选 pid：进程没了就别再等满 150*2 s
    local port=$1 pid=${2:-}
    for _ in $(seq 1 150); do
        if [ "$(curl -s -o /dev/null -w '%{http_code}' --max-time 3 "http://127.0.0.1:$port/health" 2>/dev/null)" = "200" ]; then
            return 0
        fi
        if [ -n "$pid" ] && ! kill -0 "$pid" 2>/dev/null; then
            return 1
        fi
        sleep 2
    done
    return 1
}

ask() { # 与 verify.sh 的 ask() 同一形状：T=0，只取文本，便于逐字节比
    curl -s --max-time 900 "http://127.0.0.1:$1/v1/chat/completions" -H 'Content-Type: application/json' \
        -d "{\"messages\":[{\"role\":\"user\",\"content\":\"$PROMPT\"}],\"max_tokens\":$N_PREDICT,\"temperature\":0,\"top_k\":1,\"cache_prompt\":false}" \
        | python -c "import json,sys; d=json.load(sys.stdin); m=d['choices'][0]['message']; print((m.get('reasoning_content') or '') + (m.get('content') or ''))" 2>/dev/null
}

hdr "前置：模型 / 内存"
if [ ! -f "$MODEL" ] || [ ! -f "$DRAFT" ]; then
    bad "缺模型：target=$MODEL（或 -md $DRAFT）"
    exit 1
fi
# 内存闸门用 _qlh_env.sh 的 qlh_available_gb_settled：与 verify.sh 组 9/10 同一口径
# （Available MBytes = 空闲 + 可回收 standby）。旧版这里自己用 FreePhysicalMemory，
# 比 verify.sh 低约 3 GB，独立跑时闸门比并进组 11 时严得多。
# 被 verify.sh 当组 11 调用时它传 QLH_DFDEV_MIN_FREE_GB=0（父进程刚过同一道闸门），
# 此时跳过测量，省掉 4 次 powershell + 8 s 等待。
AVAIL_GB=""
if [ "$MIN_FREE_GB" -gt 0 ]; then
    AVAIL_GB=$(qlh_available_gb_settled)
    if [ "$AVAIL_GB" -lt "$MIN_FREE_GB" ]; then
        bad "可用内存 ${AVAIL_GB} GB < 闸门 ${MIN_FREE_GB} GB，这一轮没测到（不是 PASS）"
        exit 1
    fi
    echo "  [mem] 可用 ${AVAIL_GB} GB ≥ ${MIN_FREE_GB} GB"
else
    echo "  [mem] 闸门 $MIN_FREE_GB：跳过内存测量（父脚本已过同一道闸门）"
fi

# ---------------------------------------------------------------- 第 1 半：整模仍走 device buffer
hdr "第 1 半  整模 + DFlash（device buffer 必须仍然被选中）"
"$BIN/llama-server.exe" -m "$MODEL" --host 127.0.0.1 --port "$PORT_WHOLE" -c 4096 -ngl 0 --no-warmup \
      -md "$DRAFT" --spec-type draft-dflash --spec-draft-n-max 2 > "$LOG_WHOLE" 2>&1 &
srv_whole=$!
if wait_health "$PORT_WHOLE" "$srv_whole"; then
    grep -q "DFlash target features via device buffer" "$LOG_WHOLE" \
        && ok "整模：device-buffer 路径仍然启用（宽度谓词没有误杀整模）" \
        || bad "整模退回了 host copies：device 路径被误杀（见 $(basename "$LOG_WHOLE")）"
    whole=$(ask "$PORT_WHOLE")
    if [ -n "$whole" ]; then
        ok "整模：请求 200，${#whole} 字符"
    else
        bad "整模请求失败"
        tail -6 "$LOG_WHOLE" | sed 's/^/    /'
    fi
else
    bad "整模 server 没起来，见 $(basename "$LOG_WHOLE")"
    tail -8 "$LOG_WHOLE" | sed 's/^/    /'
fi
cleanup_ours 整模

# ---------------------------------------------------------------- 第 2 半：远程链默认路径不崩
hdr "第 2 半  远程链 + DFlash（不设 LLAMA_DFLASH_DEV，必须回退且结果与整模逐字节相同）"
"$BIN/split-stage.exe" -m "$MODEL" --layer-range 32:40 --role tail --rs-seq "$QLH_RS_SEQ" \
      --port "$PORT_STAGE" -c 8192 -b 512 > "$LOG_STAGE" 2>&1 &
srv_stage=$!
# [v11] 等的是 split-stage 的 ready:，不是 listening：v10 之后 bind 在加载之前，端口绑上只说明
# 端口是它的。用 listening 当信号，后面 llama-server 会在 stage 还没加载完时去连它，最后报成
# 「链不对」而不是「stage 没起来」。
for _ in $(seq 1 180); do
    grep -q "ready:" "$LOG_STAGE" 2>/dev/null && break
    kill -0 "$srv_stage" 2>/dev/null || break
    sleep 2
done
if ! grep -q "ready:" "$LOG_STAGE" 2>/dev/null; then
    bad "tail stage 没起来（180 s 内没有 ready:），见 $(basename "$LOG_STAGE")"
    tail -8 "$LOG_STAGE" | sed 's/^/    /'
    exit 1
fi

"$BIN/llama-server.exe" -m "$MODEL" --host 127.0.0.1 --port "$PORT_CHAIN" -c 4096 -ngl 0 --no-warmup \
      --split-layer-range 0:32 --split-remote "127.0.0.1:$PORT_STAGE" \
      -md "$DRAFT" --spec-type draft-dflash --spec-draft-n-max 2 > "$LOG_CHAIN" 2>&1 &
srv_chain=$!
if wait_health "$PORT_CHAIN" "$srv_chain"; then
    grep -q "DFlash target features via host copies" "$LOG_CHAIN" \
        && ok "链：按预期回退到 host copies（device buffer 装不下跨段的那几路 tap）" \
        || bad "链：没有回退（日志里没有 via host copies）—— 请求可能仍会走进 dflash.cpp 的断言"
    grep -q "device buffer unusable on this chain" "$LOG_CHAIN" \
        && ok "链：回退原因写明（head 只装了 8 路里的 6 路）" \
        || bad "链：回退原因没写（日志缺少 device buffer unusable 行）"
    chain=$(ask "$PORT_CHAIN")
    if [ -z "$chain" ]; then
        bad "链请求失败"
    elif [ "$chain" = "$whole" ]; then
        ok "链：输出与整模逐字节相同（${#chain} 字符）—— 回退后的特征是对的"
    else
        bad "链：输出与整模不同（整模 ${#whole} 字符 / 链 ${#chain} 字符）"
    fi
    if grep -q "GGML_ASSERT" "$LOG_CHAIN"; then
        bad "链：日志出现 GGML_ASSERT（崩溃）"
        grep -n "GGML_ASSERT\|embd-row batch" "$LOG_CHAIN" | head -3 | sed 's/^/    /'
    else
        ok "链：没有 GGML_ASSERT"
    fi
    [ "$(curl -s -o /dev/null -w '%{http_code}' --max-time 10 "http://127.0.0.1:$PORT_CHAIN/health" 2>/dev/null)" = "200" ] \
        && ok "链：请求后进程仍存活" \
        || bad "链：请求后 /health 不再是 200（进程可能已崩）"
    grep -q "draft acceptance" "$LOG_CHAIN" && ok "链：这一轮真的投机了（日志里有 draft acceptance）" \
        || bad "链：没有 draft acceptance 行，投机可能没被触发"
else
    bad "链 server 没起来，见 $(basename "$LOG_CHAIN")"
    tail -8 "$LOG_CHAIN" | sed 's/^/    /'
fi
cleanup_ours 链

hdr "结果"
echo "  PASS=$PASS  FAIL=$FAIL"
unlock $([ "$FAIL" -eq 0 ] && echo 0 || echo 1)
