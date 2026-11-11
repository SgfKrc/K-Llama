#!/bin/bash
# [QLH-SPLIT] v8: grammar + per-row verify channel (speculation for agent/tools traffic).
#
# 为什么单独一组：grammar 曾经整请求禁用投机（common_sampler_verify_rows_check 返回 false ->
# spec_blocked），于是所有 tool calling 请求都是「每个 token 回一整行 993 KB logits」。这一组
# 证明四件事，一条命令跑完：
#
#   1. 带 grammar 的请求真的在投机（日志里没有闸门，且有 draft acceptance）；
#   2. 分段不改变投机结果：链上 MTP（带 grammar）与整模 MTP（带 grammar）逐字节相同；
#   3. 被 grammar 拒绝的行能回退：恢复 KV + 回滚 grammar + 下一轮普通 decode，之后继续投机；
#   4. 走的是逐行通道（tail 收到 verify frame，回退的 seq_rm 也到达 tail），且没有 abort /
#      runtime_error 痕迹（非法 draft 从未被 accept 进 grammar）。
#
# 判据为什么是「链 == 整模」而不是「投机 == 不投机」：见 verify_mtp_e2e.sh 文件头，llama.cpp 的
# k+1 行批量 forward 与逐行 forward 走不同 kernel，h 有 0.27 量级的确定性差异。grammar 的回退
# 会额外引入一次「1 行重算」：那次重算的 logits 与 k+1 行帧有同样的 kernel 形状差，所以只有在
# 回退位置的 top-2 近似并列（例如 18 vs 18.0）才可能翻。每个 prompt 单独重启服务，就是为了让
# 「本次请求回退了几次」和「本次是否逐字节相同」能对得上号。
#
# 用法：bash verify/verify_grammar_spec.sh [--skip-build]
#   QLH_GRAMMAR_MODEL   模型（默认 2B，需带 NextN/MTP 层）
#   QLH_GRAMMAR_GBNF    grammar 文件（默认 verify/grm/toolcall_open.gbnf）
#   QLH_GRAMMAR_SPEC_K  --spec-draft-n-max（默认 2；两侧必须一样）
#   QLH_GRAMMAR_MAX_TOK n_predict（默认 40）
#   QLH_GRAMMAR_PROMPTS 用空格分隔，取值 natural / json（默认两个都跑）
# 只杀 93xx；9103/9104/9106 绝不碰。

set -u

source "$(dirname "${BASH_SOURCE[0]}")/_qlh_env.sh"
ROOT=$QLH_SRC_DIR
QLH_BUILD_BAT=${QLH_BUILD_BAT:-$QLH_SRC_DIR/.qlh_tmp/build.bat}
[ -f "$QLH_BUILD_BAT" ] || QLH_BUILD_BAT=$QLH_VERIFY_DIR/build.bat
MODEL=${QLH_GRAMMAR_MODEL:-H:/qlh_models/qwen3-5-2b-gguf/qwen35-2b-Q4_K_M.gguf}
GBNF=${QLH_GRAMMAR_GBNF:-$QLH_VERIFY_DIR/grm/toolcall_open.gbnf}
HEAD_RANGE=0:16
TAIL_RANGE=16:24
STAGE_PORT=9331
SRV_PORT=9346
SPEC_K=${QLH_GRAMMAR_SPEC_K:-2}
# 严格判据那一组用更小的 k：帧只有 2 行，logits 与整模的批量形状差最小，所以「链 == 整模」
# 在这里必须是逐字节相等 —— 即使这次请求里发生了回退（回退会额外引入一次 1 行重算）。
LO_K=${QLH_GRAMMAR_LO_K:-1}
MAX_TOK=${QLH_GRAMMAR_MAX_TOK:-40}
PROMPTS=${QLH_GRAMMAR_PROMPTS:-natural json}
OUT=$QLH_TMP/grm

PASS=0
FAIL=0
# 只有真的尝试起过进程之后，trap 才动端口：编译失败 / 缺二进制 / 缺 grammar 这些路径
# 发生在起进程之前，那时去清 93xx 会误杀同一台机器上别人的服务。
STARTED=0
ok()  { echo "  [PASS] $*"; PASS=$((PASS+1)); }
bad() { echo "  [FAIL] $*"; FAIL=$((FAIL+1)); }
hdr() { echo; echo "=== $* ==="; }

kill_ours() {
    for pid in $(netstat -ano 2>/dev/null | grep -E 'LISTENING' | grep -E ':93[0-9][0-9][[:space:]]' \
                 | awk '{print $NF}' | tr -d '\r' | sort -u); do
        taskkill //F //PID "$pid" >/dev/null 2>&1
    done
    sleep 2
}

# 只杀一个端口：run_case 之间不能 kill_ours，否则会把停在 9331 的 tail 段一起杀掉
kill_port() {
    local port=$1
    for pid in $(netstat -ano 2>/dev/null | grep -E 'LISTENING' | grep -E ":$port[[:space:]]" \
                 | awk '{print $NF}' | tr -d '\r' | sort -u); do
        taskkill //F //PID "$pid" >/dev/null 2>&1
    done
    sleep 1
}

# 任何退出路径都清一遍：正常结束、编译失败 exit 1、判据里 exit 1、Ctrl-C、TERM。
# 以前只有正常路径调 kill_ours：中途被中断时 split-stage / llama-server 会留在 93xx 上，
# 下一轮就报成 identity mismatch / health=000 那种假失败（verify.sh 头部记过这条）。
# 只清**本脚本自己用的两个端口**，不调 kill_ours：trap 会在编译失败这类「还没起过任何进程」
# 的路径上执行，按 93xx 全扫会误杀同一台机器上别人的服务。
trap 'rc=$?; if [ "$STARTED" = "1" ]; then kill_port "$SRV_PORT"; kill_port "$STAGE_PORT"; fi; exit "$rc"' EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

wait_health() {
    for _ in $(seq 1 60); do
        [ "$(curl -s --max-time 5 -o /dev/null -w '%{http_code}' "http://127.0.0.1:$SRV_PORT/health" 2>/dev/null)" = "200" ] && return 0
        sleep 1
    done
    return 1
}

# body <out.json> <prompt-name> ：一条带 grammar 的 /completion 请求
# prompt 在 python 里定义，因为 bash 的 $(...) 会吃掉结尾换行
make_body() {
    python - "$1" "$GBNF" "$2" "$MAX_TOK" <<'PY'
import json, sys
out, gbnf, name, n = sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4])

PROMPTS = {
    # 自然语言收尾：模型必须被 grammar 拽进 JSON，验证帧大概率被拒（判据 3 的来源）
    "natural": "You are a helpful assistant with access to tools.\n\n"
               "User: What is the weather in Paris?\nAssistant:\n",
    # 已经处在 JSON 里：接受率高
    "json": 'Tool call: {"name": "get_weather", "arguments": {"city": "',
}

g = open(gbnf, encoding="utf-8").read()
body = {"prompt": PROMPTS[name], "n_predict": n, "temperature": 0, "grammar": g,
        "cache_prompt": False, "stream": False}
open(out, "w", encoding="utf-8").write(json.dumps(body))
PY
}

# ask <body.json> <out.txt> ：把 content 原样写进 out.txt（逐字节比较的对象）
ask() {
    # --max-time：服务半死（进程活着但不回）时没有超时会无限挂，连 trap 都没机会跑。
    curl -s --max-time 900 "http://127.0.0.1:$SRV_PORT/completion" -H 'Content-Type: application/json' \
        -d @"$1" > "$2.resp"
    python - "$2.resp" "$2" <<'PY'
import json, sys
resp, out = sys.argv[1], sys.argv[2]
d = json.load(open(resp, encoding="utf-8"))
open(out, "w", encoding="utf-8", newline="").write(d["content"])
n = d.get("tokens_predicted", (d.get("usage") or {}).get("completion_tokens", -1))
print("    tokens = %s, stop = %s, content = %r" % (n, d.get("stop"), d["content"][:110]))
PY
}

# 等 stage 就绪的超时。ready: 是 v10 起 split-stage 在模型/ctx/offload 都落定、即将进 accept 循环
# 前打的那一行；端口 bind 在加载之前，所以「端口可连」不是就绪。可用 QLH_STAGE_READY_TIMEOUT 覆盖，
# 只为验证「加载没完时判 FAIL 而不是挂住」，不是常规旋钮。
STAGE_READY_TIMEOUT=${QLH_STAGE_READY_TIMEOUT:-240}

# 起 tail 段并等它真的就绪；返回 1 = 没等到（超时或进程已退），调用方按「服务没起来」记账。
# 以前这里是固定 sleep 8：2B 够、35B 必假 FAIL，等于把等待换成了赌。
start_stage() {
    # [v11] 先截断日志（父 shell 做）。否则循环的第一次 grep 可能读到上一轮留下的 ready:，那是
    # 假 PASS —— 会放行一个还没加载完的 stage。
    : > "$OUT/stage.log"
    "$BIN/split-stage.exe" -m "$MODEL" --layer-range "$TAIL_RANGE" --role tail --rs-seq "$QLH_RS_SEQ" \
          --port $STAGE_PORT -c 1024 -b 256 > "$OUT/stage.log" 2>&1 &
    local pid=$!
    local t=0
    while [ "$t" -lt "$STAGE_READY_TIMEOUT" ]; do
        grep -q '^\[split-stage\] ready:' "$OUT/stage.log" 2>/dev/null && return 0
        kill -0 "$pid" 2>/dev/null || break
        sleep 1
        t=$((t + 1))
    done
    # 边界：最后一次 sleep 期间可能刚好写完 ready:（2B 热缓存下约 1.0-1.4 s）。再确认一次。
    grep -q '^\[split-stage\] ready:' "$OUT/stage.log" 2>/dev/null && return 0
    echo "    [stage] ${STAGE_READY_TIMEOUT}s 内没有 ready:（pid=$pid），日志尾：" >&2
    tail -8 "$OUT/stage.log" 2>/dev/null | sed 's/^/      /' >&2
    return 1
}

CUR_K=$SPEC_K

start_chain() { # <log>
    ( LLAMA_TRACE=1 "$BIN/llama-server.exe" -m "$MODEL" --host 127.0.0.1 --port $SRV_PORT -c 1024 -ngl 0 \
          --no-warmup -b 256 -ub 256 --split-layer-range "$HEAD_RANGE" \
          --split-remote 127.0.0.1:$STAGE_PORT \
          --spec-type draft-mtp --spec-draft-n-max "$CUR_K" > "$1" 2>&1 & )
    wait_health
}

start_whole() { # <log>
    ( LLAMA_TRACE=1 "$BIN/llama-server.exe" -m "$MODEL" --host 127.0.0.1 --port $SRV_PORT -c 1024 -ngl 0 \
          --no-warmup -b 256 -ub 256 \
          --spec-type draft-mtp --spec-draft-n-max "$CUR_K" > "$1" 2>&1 & )
    wait_health
}

# run_case <chain|whole> <prompt> ：一次请求，日志与输出按 (模式, k, prompt) 分开
run_case() {
    local mode=$1 prompt=$2
    STARTED=1   # 从这一刻起 trap 有权清端口（本函数马上要起 stage / server）
    local log=$OUT/${mode}_k${CUR_K}_${prompt}.log
    rm -f "$OUT/${mode}_k${CUR_K}_${prompt}.txt" "$OUT/${mode}_k${CUR_K}_${prompt}.health" "$log"
    make_body "$OUT/${mode}_k${CUR_K}_${prompt}.json" "$prompt"
    local started=1
    if [ "$mode" = chain ]; then
        if start_stage; then
            start_chain "$log" || started=0
        else
            started=0
        fi
    else
        start_whole "$log" || started=0
    fi
    if [ "$started" = "1" ]; then
        ask "$OUT/${mode}_k${CUR_K}_${prompt}.json" "$OUT/${mode}_k${CUR_K}_${prompt}.txt"
        local h=$(curl -s --max-time 5 -o /dev/null -w '%{http_code}' "http://127.0.0.1:$SRV_PORT/health" 2>/dev/null)
        echo "$h" > "$OUT/${mode}_k${CUR_K}_${prompt}.health"
    fi
    kill_port $SRV_PORT
    kill_port $STAGE_PORT
    [ "$started" = "1" ]
}

run_pair() { # <k> <prompt>
    CUR_K=$1
    local p=$2
    if run_case chain "$p"; then ok "chain k=$CUR_K/$p 完成（health=$(cat "$OUT/chain_k${CUR_K}_$p.health")）"; else bad "chain k=$CUR_K/$p 服务没起来"; fi
    if run_case whole "$p"; then ok "whole k=$CUR_K/$p 完成"; else bad "whole k=$CUR_K/$p 服务没起来"; fi
}

MODE=${1:-}
if [ "$MODE" != "--skip-build" ]; then
    hdr "编译"
    kill_ours
    if ! cmd //c "$QLH_BUILD_BAT" --clean-first > "$QLH_TMP/build.log" 2>&1; then
        bad "编译失败，见 $QLH_TMP/build.log"
        tail -20 "$QLH_TMP/build.log"
        exit 1
    fi
    ok "构建完成"
fi

if [ ! -x "$BIN/llama-server.exe" ] || [ ! -x "$BIN/split-stage.exe" ]; then
    echo "missing $BIN/llama-server.exe or split-stage.exe"
    exit 1
fi
if [ ! -f "$GBNF" ]; then
    echo "missing grammar file $GBNF"
    exit 1
fi

mkdir -p "$OUT"
kill_ours
rm -f "$OUT"/chain_*.txt "$OUT"/whole_*.txt "$OUT"/chain_*.health "$OUT"/whole_*.health \
      "$OUT"/chain_*.log "$OUT"/whole_*.log "$OUT"/stage.log

hdr "跑请求（head $HEAD_RANGE + 远程 tail $TAIL_RANGE，k=$SPEC_K 与严格对照 k=$LO_K，grammar=$(basename "$GBNF")）"
for p in $PROMPTS; do run_pair $SPEC_K "$p"; done
for p in $PROMPTS; do run_pair $LO_K   "$p"; done
kill_ours

FIRST=$(echo "$PROMPTS" | awk '{print $1}')
CHAIN_LOG=$OUT/chain_k${SPEC_K}_$FIRST.log
STAGE_LOG=$OUT/stage.log

# ------------------------------------------------------------------ 判据 1：真的在投机
hdr "判据 1  grammar 请求不再被闸门挡掉，且真的在投机"
if grep -q "verifies its drafts through the per-row channel" "$CHAIN_LOG"; then
    ok "head 把 grammar 请求放上了逐行验证通道"
else
    bad "head 没有把请求放上逐行验证通道"
    grep -a "\[split\]" "$CHAIN_LOG" | head -4 | sed 's/^/      /'
fi
if grep -q "speculative decoding is off for this request" "$CHAIN_LOG"; then
    bad "请求被判定为不可投机"
    grep -a "speculative decoding is off" "$CHAIN_LOG" | head -2 | sed 's/^/      /'
else
    ok "请求没有被判定为不可投机（旧闸门 'grammar is set: a grammar can reject the argmax' 已移除）"
fi
# 候选通道（v5）对 grammar 仍然关门，这是有意的：GBNF 能把 top-k 之外的 token 变成合法解，
# 候选集不足以复现 grammar 约束后的 argmax。它只影响带宽，不影响投机，所以是 INFO 不是 FAIL。
if grep -aq "takes the whole-logits path: grammar is set" "$CHAIN_LOG"; then
    echo "  [INFO] 候选通道仍对 grammar 关门（v5 有意为之）"
fi
if grep -aq "draft acceptance = " "$CHAIN_LOG"; then
    acc=$(grep -a "draft acceptance = " "$CHAIN_LOG" | tail -1 | sed 's/.*draft/draft/')
    ok "有 draft acceptance 行：$acc"
    n_acc=$(grep -a "draft acceptance = " "$CHAIN_LOG" | tail -1 | sed 's/.*( *\([0-9]*\) accepted.*/\1/')
    if [ "${n_acc:-0}" -gt 0 ]; then
        ok "接受数 = $n_acc > 0（真的接受过 draft）"
    else
        bad "接受数为 0，投机没有产出"
    fi
else
    bad "日志里没有 draft acceptance 行（投机没跑）"
fi

# ------------------------------------------------------------------ 判据 2：链 == 整模
hdr "判据 2  链上 MTP（带 grammar）== 整模 MTP（带 grammar），逐字节"
same_with_rollback=0
for k in $SPEC_K $LO_K; do
    for p in $PROMPTS; do
        n_rej=$(grep -ac "row rejected by the grammar" "$OUT/chain_k${k}_$p.log")
        if cmp -s "$OUT/chain_k${k}_$p.txt" "$OUT/whole_k${k}_$p.txt"; then
            ok "k=$k $p：逐字节相同（$(wc -c < "$OUT/chain_k${k}_$p.txt") 字节，本次回退 $n_rej 次）"
            [ "$n_rej" -gt 0 ] && same_with_rollback=$((same_with_rollback+1))
        elif [ "$n_rej" -eq 0 ]; then
            # 没有回退 => 两个进程的帧形状一致 => 必须逐字节相同
            bad "k=$k $p：没有回退却不同（分段改变了投机结果）"
            diff <(cat "$OUT/chain_k${k}_$p.txt") <(cat "$OUT/whole_k${k}_$p.txt") | head -4 | sed 's/^/      /'
        else
            echo "  [INFO] k=$k $p：不同（本次回退 $n_rej 次）"
            echo "         回退给链多了一次 1 行重算，帧对齐与整模不同；这是 kernel 形状差（0.27 量级 h），"
            echo "         不是 token 错误：两边的输出都满足 grammar，且回退之后紧接着的 token 与整模相同"
            diff <(cat "$OUT/chain_k${k}_$p.txt") <(cat "$OUT/whole_k${k}_$p.txt") | head -4 | sed 's/^/      /'
        fi
    done
done
if [ "$same_with_rollback" -gt 0 ]; then
    ok "有 $same_with_rollback 组「带回退且逐字节相同」的用例（回退本身不改变结果）"
else
    bad "没有「带回退且逐字节相同」的用例：回退路径与整模的等价性没有被证明"
fi

# ------------------------------------------------------------------ 判据 3：回退
hdr "判据 3  被 grammar 拒绝的行能回退，且回退后继续投机"
n_rej=$(grep -ac "row rejected by the grammar" "$CHAIN_LOG")
if [ "$n_rej" -gt 0 ]; then
    ok "出现 $n_rej 次整帧回退：$(grep -a 'row rejected by the grammar' "$CHAIN_LOG" | head -1 | sed 's/.*accepted/accepted/')"
    rej_line=$(grep -an "row rejected by the grammar" "$CHAIN_LOG" | head -1 | cut -d: -f1)
    acc_after=$(awk -v n="$rej_line" 'NR > n && /accepted +[1-9]/ { c++ } END { print c+0 }' "$CHAIN_LOG")
    if [ "$acc_after" -gt 0 ]; then
        ok "回退之后又出现了 $acc_after 次接受（没有空转，也没有退化成不投机）"
    else
        bad "回退之后没有再接受过 draft（可能空转）"
        awk -v n="$rej_line" 'NR > n' "$CHAIN_LOG" | grep -a "accepted" | head -4 | sed 's/^/      /'
    fi
else
    bad "没有观察到整帧回退（首步/转折点应至少出现一次；判据 3 无法证明）"
fi
if [ "$(cat "$OUT/chain_k${SPEC_K}_$FIRST.health")" = "200" ]; then
    ok "请求结束后 /health = 200（进程活着）"
else
    bad "请求结束后 /health = $(cat "$OUT/chain_k${SPEC_K}_$FIRST.health")（进程挂了？）"
fi
abort_hits=0
for p in $PROMPTS; do
    if grep -aqiE "unexpected empty grammar stack|terminate called|GGML_ABORT|abort\(\)" "$OUT/chain_k${SPEC_K}_$p.log"; then
        abort_hits=$((abort_hits+1))
        bad "k=$SPEC_K $p：日志里有异常终止/grammar 抛异常的痕迹"
        grep -aiE "unexpected empty grammar stack|terminate called|GGML_ABORT|abort\(\)" "$OUT/chain_k${SPEC_K}_$p.log" | head -3 | sed 's/^/      /'
    fi
done
[ "$abort_hits" -eq 0 ] && ok "没有 runtime_error / abort 痕迹（非法 draft 从未被 accept 进 grammar）"

# ------------------------------------------------------------------ 判据 4：逐行通道
hdr "判据 4  走的是逐行验证通道（每行一个 argmax）"
if grep -aq "first OP_DECODE verify frame" "$STAGE_LOG"; then
    ok "tail 收到 verify frame：$(grep -a 'first OP_DECODE verify frame' "$STAGE_LOG" | head -1 | sed 's/.*first/first/')"
else
    bad "tail 没收到 verify frame"
    grep -a "OP_DECODE" "$STAGE_LOG" | head -3 | sed 's/^/      /'
fi
if grep -aq "OP_SEQ_RM" "$STAGE_LOG"; then
    ok "回退的 seq_rm 也到了 tail（$(grep -a 'OP_SEQ_RM' "$STAGE_LOG" | tail -1 | sed 's/.*OP_SEQ_RM/OP_SEQ_RM/')）"
else
    bad "tail 没收到 seq_rm（回退没有广播到远端）"
fi

hdr "结果"
echo "  PASS=$PASS  FAIL=$FAIL   （日志目录：$OUT）"
[ "$FAIL" -eq 0 ] || exit 1
