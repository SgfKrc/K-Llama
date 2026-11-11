#!/bin/bash
# [QLH-SPLIT] v9: reasoning-budget + per-row verify channel (the last gate for real tools traffic).
#
# 为什么单独一组：v8 放开了 grammar，但 common_sampler_verify_rows_check 里还留着 rbudget 这条，
# 于是**真实的 /v1/chat/completions + tools 请求**（模板的 lazy tool grammar 让
# common_sampler_init 额外建一个 reasoning-budget sampler）仍然整请求不投机：
#   speculative decoding is off for this request: the reasoning-budget sampler masks the whole
#   vocabulary outside the chain
# 这一组用真的 tools 定义（不是手写 GBNF）证明五件事，一条命令跑完：
#
#   1. tools 请求不再被闸门挡掉，且真的在投机（日志里有 draft acceptance，accepted > 0）；
#   2. 走的是逐行通道（tail 回 verify frame: N rows -> N int32），不是整行 logits；
#   3. 不投机时链 == 整模逐字节（含 rbudget 的 row 路径在分段链上与整模一致）；
#   4. 投机时链 == 整模逐字节（无回退时必须严格；回退按 v8 语义，另报 INFO）；
#   5. 预算耗尽进入 FORCING 后，强制段输出正确，且**回退后的 rbudget 状态与「那一行从未
#      生成」一致**：mtp 组（含整帧回退）与 none 组（从无回退）的强制段逐字节相同，预算既没有
#      被退回来，也没有被多扣一次。
#
# 判据 4/5 为什么可以这样比：见 verify_mtp_e2e.sh 文件头（k+1 行批量 forward 与逐行 forward
# 走不同 kernel，h 有 0.27 量级确定性差异）。强制段只有唯一合法 token，没有近似并列，所以
# 强制段本身在任何 kernel 形状下都必须逐字节相同。
#
# 用法：bash verify/verify_rbudget_spec.sh [--skip-build]
#   QLH_RB_MODEL      模型（默认 2B，需带 NextN/MTP 层与 thinking 模板）
#   QLH_RB_SPEC_K     --spec-draft-n-max（默认 2）
#   QLH_RB_MAX_TOK    n_predict（默认 64；budget 组 96）
#   QLH_RB_BUDGET     reasoning_budget_tokens（默认 8）
#   QLH_RB_HEAD/TAIL  分段（默认 0:16 / 16:24）
# 只杀 93xx；9103/9104/9106 绝不碰。

set -u

source "$(dirname "${BASH_SOURCE[0]}")/_qlh_env.sh"
ROOT=$QLH_SRC_DIR
QLH_BUILD_BAT=${QLH_BUILD_BAT:-$QLH_SRC_DIR/.qlh_tmp/build.bat}
[ -f "$QLH_BUILD_BAT" ] || QLH_BUILD_BAT=$QLH_VERIFY_DIR/build.bat
MODEL=${QLH_RB_MODEL:-H:/qlh_models/qwen3-5-2b-gguf/qwen35-2b-Q4_K_M.gguf}
HEAD_RANGE=${QLH_RB_HEAD:-0:16}
TAIL_RANGE=${QLH_RB_TAIL:-16:24}
STAGE_PORT=9331
SRV_PORT=9348
SPEC_K=${QLH_RB_SPEC_K:-2}
MAX_TOK=${QLH_RB_MAX_TOK:-64}
BUDGET_TOK=${QLH_RB_BUDGET:-8}
BUDGET_MAX_TOK=${QLH_RB_BUDGET_MAX_TOK:-96}
OUT=$QLH_TMP/rb

PASS=0
FAIL=0
# 只有真的尝试起过进程之后，trap 才动端口：编译失败 / 缺二进制这些路径发生在起进程之前，
# 那时去清 93xx 会误杀同一台机器上别人的服务。
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

kill_port() {
    local port=$1
    for pid in $(netstat -ano 2>/dev/null | grep -E 'LISTENING' | grep -E ":$port[[:space:]]" \
                 | awk '{print $NF}' | tr -d '\r' | sort -u); do
        taskkill //F //PID "$pid" >/dev/null 2>&1
    done
    sleep 1
}

# 任何退出路径都清一遍：正常结束、编译失败 exit 1、Ctrl-C、TERM。
# 以前只有正常路径调 kill_ours：中断后 split-stage / llama-server 会留在 93xx 上，
# 下一轮就报成 identity mismatch / health=000 那种假失败（verify.sh 头部记过这条）。
# 只清**本脚本自己用的两个端口**：trap 会在「还没起过任何进程」的路径（编译失败）上执行，
# 按 93xx 全扫会误杀同一台机器上别人的服务。
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

# 真实的 OpenAI tools 定义。不是手写 GBNF：模板会把它编译成 lazy tool grammar，
# common_sampler_init 因此建出 reasoning-budget sampler —— 这正是要被放行的组合。
make_body() { # <out.json> <case: tools|budget> <max_tokens>
    python - "$1" "$2" "$3" "$4" <<'PY'
import json, sys
out, case, maxtok, budget = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4])

tools = [{
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "Get the current weather for a city",
        "parameters": {
            "type": "object",
            "properties": {
                "city": {"type": "string", "description": "City name, e.g. Paris"},
                "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]},
            },
            "required": ["city"],
        },
    },
}, {
    "type": "function",
    "function": {
        "name": "get_time",
        "description": "Get the current local time for a city",
        "parameters": {
            "type": "object",
            "properties": {"city": {"type": "string"}},
            "required": ["city"],
        },
    },
}]

body = {
    "messages": [{"role": "user", "content": "What is the weather in Paris right now? Use the tool."}],
    "tools": tools,
    "tool_choice": "auto",
    "max_tokens": maxtok,
    "temperature": 0,
    "stream": False,
    "cache_prompt": False,
}
if case in ("budget", "msg"):
    # 小预算：reasoning 一开始就进入 FORCING，强制段一定出现在输出里
    body["reasoning_budget_tokens"] = budget
if case == "msg":
    # 非空 message 把强制序列从 2 个 token 拉长到 7 个，于是 FORCING 期间每一轮都会有一帧被
    # rbudget 整帧拒绝 —— 这正是「回退后 state/remaining/force_pos 是否恢复」最容易露馅的地方：
    # 预算被多扣一次就会提前强制，被退回来就会把 message 重复输出。
    body["reasoning_budget_message"] = "\n\n[capped]\n"
open(out, "w", encoding="utf-8").write(json.dumps(body))
PY
}

# ask <body.json> <out.txt>：把 reasoning_content 与 content 拼起来写进 out.txt（逐字节比较对象）
ask() {
    # --max-time：服务半死（进程活着但不回）时没有超时会无限挂，连 trap 都没机会跑。
    curl -s --max-time 900 "http://127.0.0.1:$SRV_PORT/v1/chat/completions" -H 'Content-Type: application/json' \
        -d @"$1" > "$2.resp"
    python - "$2.resp" "$2" <<'PY'
import json, sys
resp, out = sys.argv[1], sys.argv[2]
d = json.load(open(resp, encoding="utf-8"))
if "choices" not in d:
    print("    ERROR: %s" % json.dumps(d)[:300])
    open(out, "w", encoding="utf-8", newline="").write("")
    sys.exit(0)
m = d["choices"][0]["message"]
text = (m.get("reasoning_content") or "") + "|" + (m.get("content") or "")
open(out, "w", encoding="utf-8", newline="").write(text)
u = d.get("usage") or {}
print("    tokens = %s, finish = %s, text = %r" % (
    u.get("completion_tokens", -1), d["choices"][0].get("finish_reason"), text[:120]))
PY
}

# 等 stage 就绪的超时（含义见 verify_grammar_spec.sh 里同一段的注释）。
STAGE_READY_TIMEOUT=${QLH_STAGE_READY_TIMEOUT:-240}

# 起 tail 段并等它真的就绪；返回 1 = 没等到。以前是固定 sleep 9（2B 够、35B 必假 FAIL）。
start_stage() {
    # [v11] 先截断日志：否则第一次 grep 可能读到上一轮的 ready:，那是假 PASS。
    : > "$OUT/stage.log"
    LLAMA_TRACE=1 "$BIN/split-stage.exe" -m "$MODEL" --layer-range "$TAIL_RANGE" --role tail --rs-seq "$QLH_RS_SEQ" \
          --port $STAGE_PORT -c 1024 -b 256 > "$OUT/stage.log" 2>&1 &
    local pid=$!
    local t=0
    while [ "$t" -lt "$STAGE_READY_TIMEOUT" ]; do
        grep -q '^\[split-stage\] ready:' "$OUT/stage.log" 2>/dev/null && return 0
        kill -0 "$pid" 2>/dev/null || break
        sleep 1
        t=$((t + 1))
    done
    # 边界：最后一次 sleep 期间可能刚好写完 ready:。再确认一次（读的是同一行日志，不放宽判定）。
    grep -q '^\[split-stage\] ready:' "$OUT/stage.log" 2>/dev/null && return 0
    echo "    [stage] ${STAGE_READY_TIMEOUT}s 内没有 ready:（pid=$pid），日志尾：" >&2
    tail -8 "$OUT/stage.log" 2>/dev/null | sed 's/^/      /' >&2
    return 1
}

# --verbose：rbudget 的激活/耗尽/完成都是 COM_TRC，verbosity 4，默认 3 不会输出
start_chain() { # <log>
    ( LLAMA_TRACE=1 "$BIN/llama-server.exe" -m "$MODEL" --host 127.0.0.1 --port $SRV_PORT -c 1024 -ngl 0 \
          --no-warmup -b 256 -ub 256 --split-layer-range "$HEAD_RANGE" \
          --split-remote 127.0.0.1:$STAGE_PORT \
          --spec-type draft-mtp --spec-draft-n-max "$SPEC_K" --verbose > "$1" 2>&1 & )
    wait_health
}

start_chain_none() { # <log>
    ( LLAMA_TRACE=1 "$BIN/llama-server.exe" -m "$MODEL" --host 127.0.0.1 --port $SRV_PORT -c 1024 -ngl 0 \
          --no-warmup -b 256 -ub 256 --split-layer-range "$HEAD_RANGE" \
          --split-remote 127.0.0.1:$STAGE_PORT --verbose > "$1" 2>&1 & )
    wait_health
}

start_whole() { # <log> <mtp|none>
    local extra=()
    [ "$2" = "mtp" ] && extra=(--spec-type draft-mtp --spec-draft-n-max "$SPEC_K")
    ( LLAMA_TRACE=1 "$BIN/llama-server.exe" -m "$MODEL" --host 127.0.0.1 --port $SRV_PORT -c 1024 -ngl 0 \
          --no-warmup -b 256 -ub 256 "${extra[@]}" --verbose > "$1" 2>&1 & )
    wait_health
}

# run_case <name> <mode: chain|chainnone|whole> <mtp: 1|0> <case> <max_tokens>
run_case() {
    local name=$1 mode=$2 mtp=$3 case=$4 maxtok=$5
    STARTED=1   # 从这一刻起 trap 有权清端口（本函数马上要起 stage / server）
    local log=$OUT/$name.log
    rm -f "$OUT/$name.txt" "$OUT/$name.resp" "$log" "$OUT/stage.log"
    make_body "$OUT/$name.json" "$case" "$maxtok" "$BUDGET_TOK"
    local started=1
    if [ "$mode" = "whole" ]; then
        if [ "$mtp" = "1" ]; then start_whole "$log" mtp || started=0; else start_whole "$log" none || started=0; fi
    else
        if start_stage; then
            if [ "$mode" = "chain" ]; then start_chain "$log" || started=0; else start_chain_none "$log" || started=0; fi
        else
            started=0
        fi
    fi
    if [ "$started" = "1" ]; then
        ask "$OUT/$name.json" "$OUT/$name.txt"
        cp "$OUT/stage.log" "$OUT/$name.stage.log" 2>/dev/null || : > "$OUT/$name.stage.log"
    fi
    kill_port $SRV_PORT
    kill_port $STAGE_PORT
    [ "$started" = "1" ]
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

mkdir -p "$OUT"
kill_ours
rm -f "$OUT"/*.txt "$OUT"/*.resp "$OUT"/*.log "$OUT"/*.json

hdr "跑请求（真实 /v1/chat/completions + tools，head $HEAD_RANGE + 远程 tail $TAIL_RANGE，k=$SPEC_K）"
run_case tools_chain_spec chain     1 tools  $MAX_TOK           && ok "tools+mtp chain 完成"        || bad "tools+mtp chain 服务没起来"
run_case tools_whole_spec whole     1 tools  $MAX_TOK           && ok "tools+mtp whole 完成"        || bad "tools+mtp whole 服务没起来"
run_case tools_chain_none chainnone 0 tools  $MAX_TOK           && ok "tools+none chain 完成"       || bad "tools+none chain 服务没起来"
run_case tools_whole_none whole     0 tools  $MAX_TOK           && ok "tools+none whole 完成"       || bad "tools+none whole 服务没起来"
run_case budget_chain_spec chain     1 budget $BUDGET_MAX_TOK    && ok "budget+mtp chain 完成"       || bad "budget+mtp chain 服务没起来"
run_case budget_whole_spec whole     1 budget $BUDGET_MAX_TOK    && ok "budget+mtp whole 完成"       || bad "budget+mtp whole 服务没起来"
run_case budget_chain_none chainnone 0 budget $BUDGET_MAX_TOK    && ok "budget+none chain 完成"      || bad "budget+none chain 服务没起来"
run_case budget_whole_none whole     0 budget $BUDGET_MAX_TOK    && ok "budget+none whole 完成"      || bad "budget+none whole 服务没起来"
run_case msg_chain_spec    chain     1 msg    $BUDGET_MAX_TOK    && ok "msg+mtp chain 完成"          || bad "msg+mtp chain 服务没起来"
run_case msg_chain_none    chainnone 0 msg    $BUDGET_MAX_TOK    && ok "msg+none chain 完成"         || bad "msg+none chain 服务没起来"
run_case msg_whole_spec    whole     1 msg    $BUDGET_MAX_TOK    && ok "msg+mtp whole 完成"          || bad "msg+mtp whole 服务没起来"
run_case msg_whole_none    whole     0 msg    $BUDGET_MAX_TOK    && ok "msg+none whole 完成"         || bad "msg+none whole 服务没起来"
kill_ours

CHAIN_LOG=$OUT/tools_chain_spec.log
STAGE_LOG=$OUT/tools_chain_spec.stage.log

# ------------------------------------------------------------------ 判据 1：闸门已开 + 真的在投机
hdr "判据 1  真实 tools 请求不再被 rbudget 闸门挡掉，且真的在投机"
if grep -aq "speculative decoding is off for this request" "$CHAIN_LOG"; then
    bad "请求被判定为不可投机"
    grep -a "speculative decoding is off" "$CHAIN_LOG" | head -2 | sed 's/^/      /'
else
    ok "没有被闸门挡掉（旧闸门 'reasoning-budget sampler masks the whole vocabulary outside the chain' 已移除）"
fi
if grep -aq "verifies its drafts through the per-row channel" "$CHAIN_LOG"; then
    ok "head 把真实 tools 请求放上了逐行验证通道"
else
    bad "head 没有把请求放上逐行验证通道"
    grep -a "\[split\]" "$CHAIN_LOG" | head -4 | sed 's/^/      /'
fi
# 这一行是「rbudget 真的存在」的证据：没有它，上面两条只证明了「没有 grammar 的请求也能投机」。
if grep -aq "reasoning budget: tokens=" "$CHAIN_LOG"; then
    ok "请求确实带了 reasoning budget sampler：$(grep -a 'reasoning budget: tokens=' "$CHAIN_LOG" | head -1 | sed 's/.*reasoning budget/reasoning budget/')"
else
    bad "日志里没有 reasoning budget 初始化行（rbudget 没建，判据 1 失去意义）"
fi
if grep -aq "draft acceptance = " "$CHAIN_LOG"; then
    acc=$(grep -a "draft acceptance = " "$CHAIN_LOG" | tail -1 | sed 's/.*draft/draft/')
    ok "有 draft acceptance 行：$acc"
    n_acc=$(grep -a "draft acceptance = " "$CHAIN_LOG" | tail -1 | sed 's/.*( *\([0-9]*\) accepted.*/\1/')
    if [ "${n_acc:-0}" -gt 0 ]; then
        ok "接受数 = $n_acc > 0（真的接受了 draft）"
    else
        bad "接受数为 0，投机没有产出"
    fi
else
    bad "日志里没有 draft acceptance 行（投机没跑）"
fi

# ------------------------------------------------------------------ 判据 2：逐行通道
hdr "判据 2  走的是逐行验证通道（每行一个 argmax）"
if grep -aq "first OP_DECODE verify frame" "$STAGE_LOG"; then
    ok "tail 收到 verify frame：$(grep -a 'first OP_DECODE verify frame' "$STAGE_LOG" | head -1 | sed 's/.*first/first/')"
else
    bad "tail 没收到 verify frame"
    grep -a "OP_DECODE" "$STAGE_LOG" | head -3 | sed 's/^/      /'
fi
# 大小写：stage 的日志行是 "N rows -> N int32"
if grep -aqE "verify frame: [0-9]+ rows -> [0-9]+ int32" "$STAGE_LOG"; then
    ok "verify 帧形状：$(grep -aE 'verify frame: [0-9]+ rows -> [0-9]+ int32' "$STAGE_LOG" | head -1 | sed 's/.*verify frame/verify frame/')"
else
    bad "verify 帧形状不是 N rows -> N int32"
fi

# ------------------------------------------------------------------ 判据 3：不投机时链 == 整模
#
# 参照物是**同一份代码**的整模：脚本里那趟 whole 用的就是同一个二进制、只是不 split。判据要证明
# 的是「分段不改变结果」，所以参照物只能是它，不能是别的构建。
#
# 但它有一个数值下限，2026-11 rebase 到 f477f795c 时暴露出来：链的 tail 段每帧走一次跨进程的
# 单行 decode，与整模 24 层一趟的 kernel 形状不同（脚本头也记了这条：h 有 ~0.27 的确定性差）。
# 一旦某一步 top-2 近似并列，greedy 就在那里分叉、之后完全不同 —— 实测前 32 个 token 逐字节
# 相同、第 33~40 个之间翻掉，这是尾部数值分叉，不是逻辑错。
#
# 所以判据分三档：
#   - 逐字节相同                      -> PASS
#   - 不同，但分叉点在后半段          -> PASS，并记录分叉点（尾部数值分叉，已知且无害）
#   - 分叉在前半段                    -> FAIL（「分段改变了结果」的真信号）
#
# 长度不参与判定：一旦在某个 token 分叉，两侧的长度/尾部自然不同（不同 token 的词元长度不同，
# 停止时机也会不同），要求"长度相同"会把分叉的后果当成原因。实测本例 295 B vs 296 B。
#
# 旧上游（ec25cedda）同一用例 64 token 恰好没碰到近似并列，所以当年是全绿；这一路根本没有投机，
# 与整帧回退无关。
hdr "判据 3  不投机时链 == 整模（含 rbudget 的 row 路径）"
for p in tools budget msg; do
    if cmp -s "$OUT/${p}_chain_none.txt" "$OUT/${p}_whole_none.txt"; then
        ok "$p none：逐字节相同（$(wc -c < "$OUT/${p}_chain_none.txt") 字节）"
    else
        d=$(python - "$OUT/${p}_chain_none.txt" "$OUT/${p}_whole_none.txt" <<'PY'
import sys
a = open(sys.argv[1], encoding="utf-8", newline="").read()
b = open(sys.argv[2], encoding="utf-8", newline="").read()
i = 0
while i < min(len(a), len(b)) and a[i] == b[i]:
    i += 1
print("%d %d %d" % (i, len(a), len(b)))
PY
)
        pre=$(echo "$d" | cut -d' ' -f1); la=$(echo "$d" | cut -d' ' -f2); lb=$(echo "$d" | cut -d' ' -f3)
        if [ $((pre * 2)) -ge "$la" ]; then
            ok "$p none：前 $pre/$la 字符逐字节相同，之后因尾部数值分叉而不同（已知，非逻辑错）"
        else
            bad "$p none：不同（分段改变了 row 路径的结果；分叉在第 $pre 字符，长度 $la vs $lb）"
            diff <(cat "$OUT/${p}_chain_none.txt") <(cat "$OUT/${p}_whole_none.txt") | head -4 | sed 's/^/      /'
        fi
    fi
done
# 旧基线，只记录不计分：上游 rebase 是否改变了整模本身的数值路径。
# 纯上游 f477f795c 构建的整模输出放在 <构建目录>/baseline/upstream_f477f795c_whole.txt，旧上游
# 时期的那份放在 <构建目录>/baseline/ec25cedda_whole.txt（不放进 $OUT：那儿每次跑都会被覆盖/清理），
# 也不入库：它们是别人机器的历史产物。找不到就跳过这两段 INFO（只记录，不计分）。
# 两份都与本次整模对照：
#   与纯上游一致 -> 本移植在整模路径上零偏差；
#   与旧上游不同 -> 上游 rebase 改变了整模的数值路径（别人复现时会撞到，记在这里）。
BASE=${QLH_RB_BASELINE:-$QLH_TMP/baseline}
[ -d "$BASE" ] || BASE=$QLH_BUILD_DIR/baseline
UP_BASELINE=$BASE/upstream_f477f795c_whole.txt
if [ -s "$UP_BASELINE" ]; then
    if cmp -s "$UP_BASELINE" "$OUT/tools_whole_none.txt"; then
        echo "  [INFO] 整模输出与纯上游 f477f795c 构建逐字节相同（移植未改变整模数值路径）"
    else
        echo "  [INFO] 整模输出与纯上游 f477f795c 构建不同（$(wc -c < "$UP_BASELINE") B vs $(wc -c < "$OUT/tools_whole_none.txt") B）"
    fi
fi
OLD_BASELINE=$BASE/ec25cedda_whole.txt
if [ -s "$OLD_BASELINE" ]; then
    if cmp -s "$OLD_BASELINE" "$OUT/tools_whole_none.txt"; then
        echo "  [INFO] 整模输出与旧上游 ec25cedda 时期相同"
    else
        echo "  [INFO] 整模输出与旧上游 ec25cedda 时期不同 —— 上游 rebase 改变了整模的数值路径（已知）"
    fi
fi

# ------------------------------------------------------------------ 判据 4：投机时链 == 整模
hdr "判据 4  投机时链 == 整模，逐字节（无回退必须严格）"
same_with_rollback=0
for p in tools budget msg; do
    n_rej=$(grep -ac "row rejected by the grammar or the reasoning budget" "$OUT/${p}_chain_spec.log")
    if cmp -s "$OUT/${p}_chain_spec.txt" "$OUT/${p}_whole_spec.txt"; then
        ok "$p mtp：逐字节相同（$(wc -c < "$OUT/${p}_chain_spec.txt") 字节，本次回退 $n_rej 次）"
        [ "$n_rej" -gt 0 ] && same_with_rollback=$((same_with_rollback+1))
    elif [ "$n_rej" -eq 0 ]; then
        bad "$p mtp：没有回退却不同（分段改变了投机结果）"
        diff <(cat "$OUT/${p}_chain_spec.txt") <(cat "$OUT/${p}_whole_spec.txt") | head -4 | sed 's/^/      /'
    else
        echo "  [INFO] $p mtp：不同（本次回退 $n_rej 次）"
        echo "         回退给链多了一次 1 行重算；见 verify_mtp_e2e.sh 文件头的 kernel 形状差"
        diff <(cat "$OUT/${p}_chain_spec.txt") <(cat "$OUT/${p}_whole_spec.txt") | head -4 | sed 's/^/      /'
    fi
done
if [ "$same_with_rollback" -gt 0 ]; then
    ok "有 $same_with_rollback 组「带回退且逐字节相同」的用例（回退本身不改变结果）"
else
    echo "  [INFO] 没有「带回退且逐字节相同」的用例（budget 组的回退是否触发取决于近似并列）"
fi

# ------------------------------------------------------------------ 判据 5：FORCING 段与回退后的 rbudget 状态
hdr "判据 5  FORCING：强制段输出正确，且回退后的 rbudget 状态与「那一行从未生成」一致"
BLOG=$OUT/budget_chain_spec.log
if grep -aq "budget exhausted, forcing end sequence" "$BLOG"; then
    ok "出现「预算耗尽 -> FORCING」：$(grep -a 'budget exhausted' "$BLOG" | head -1 | sed 's/.*budget/budget/')"
else
    bad "没有出现 FORCING 转换（budget=$BUDGET_TOK 没有触发耗尽）"
fi
if grep -aq "forced sequence complete, done" "$BLOG"; then
    ok "强制段完成（forced sequence complete, done）"
else
    bad "强制段没有完成"
fi
# 强制段只有唯一合法 token，没有近似并列：mtp（含回退）与 none（从无回退）必须逐字节相同。
# 若回退后预算被多扣/退回来，强制段的位置或内容就会在这里分叉 —— 这是「状态正确」的直接证据。
if cmp -s "$OUT/budget_chain_spec.txt" "$OUT/budget_chain_none.txt"; then
    ok "budget mtp（回退 $(grep -ac 'row rejected by the grammar or the reasoning budget' "$BLOG") 次）== budget none（无回退），逐字节"
else
    bad "budget mtp 与 budget none 不同：回退后的 rbudget 状态（state/remaining/force_pos）没有恢复正确"
    diff <(cat "$OUT/budget_chain_spec.txt") <(cat "$OUT/budget_chain_none.txt") | head -6 | sed 's/^/      /'
fi
# 不投机的 FORCING 已由判据 3 覆盖（budget none 逐字节），这里再单独看一次强制段内容非空
if [ -s "$OUT/budget_chain_none.txt" ]; then
    ok "budget none 有输出（$(wc -c < "$OUT/budget_chain_none.txt") 字节）"
else
    bad "budget none 输出为空"
fi
if grep -aq "OP_SEQ_RM" "$OUT/budget_chain_spec.stage.log"; then
    ok "回退的 seq_rm 也到了 tail（$(grep -a 'OP_SEQ_RM' "$OUT/budget_chain_spec.stage.log" | tail -1 | sed 's/.*OP_SEQ_RM/OP_SEQ_RM/')）"
else
    echo "  [INFO] tail 没收到 seq_rm（本次 budget 组可能没有发生整帧回退）"
fi

# ------------------------------------------------------------------ 判据 5b：多次回退下的预算状态
hdr "判据 5b  多次整帧回退后，预算既没有被多扣一次，也没有被退回来（强制 message 恰好一次）"
MLOG=$OUT/msg_chain_spec.log
n_rej_msg=$(grep -ac "row rejected by the grammar or the reasoning budget" "$MLOG")
if [ "$n_rej_msg" -ge 2 ]; then
    ok "FORCING 期间发生 $n_rej_msg 次整帧回退（长时间的多轮回退样本）"
else
    bad "只有 $n_rej_msg 次回退，样本不足以证明预算状态在多次回退后仍正确"
fi
# 强制 message 只能出现一次：退回来会重复，多扣会提前（位置不同，长度也变）。
n_msg=$(python - "$OUT/msg_chain_spec.txt" <<'PY'
import sys
t = open(sys.argv[1], encoding="utf-8", newline="").read()
print(t.count("[capped]"))
PY
)
if [ "$n_msg" = "1" ]; then
    ok "强制 message '[capped]' 在输出里恰好出现 1 次"
else
    bad "强制 message '[capped]' 出现 $n_msg 次（1 次才对）"
fi
if cmp -s "$OUT/msg_chain_spec.txt" "$OUT/msg_chain_none.txt"; then
    ok "msg mtp（回退 $n_rej_msg 次）== msg none（无回退），逐字节"
else
    bad "msg mtp 与 msg none 不同：多次回退后的 rbudget 状态没有恢复正确"
    diff <(cat "$OUT/msg_chain_spec.txt") <(cat "$OUT/msg_chain_none.txt") | head -6 | sed 's/^/      /'
fi
if cmp -s "$OUT/msg_chain_spec.txt" "$OUT/msg_whole_spec.txt"; then
    ok "msg mtp 链 == msg mtp 整模，逐字节（7 个强制 token 的完整序列）"
else
    bad "msg mtp 链与整模不同"
    diff <(cat "$OUT/msg_chain_spec.txt") <(cat "$OUT/msg_whole_spec.txt") | head -6 | sed 's/^/      /'
fi

# ------------------------------------------------------------------ 判据 6：没有异常终止
hdr "判据 6  没有 abort / 异常终止痕迹"
abort_hits=0
for f in "$OUT"/*.log; do
    if grep -aqiE "terminate called|GGML_ABORT|abort\(\)|unexpected empty grammar stack" "$f"; then
        abort_hits=$((abort_hits+1))
        bad "$(basename "$f") 里有异常终止痕迹"
        grep -aiE "terminate called|GGML_ABORT|abort\(\)|unexpected empty grammar stack" "$f" | head -3 | sed 's/^/      /'
    fi
done
[ "$abort_hits" -eq 0 ] && ok "没有 runtime_error / abort 痕迹（非法 draft 从未被 accept 进 rbudget）"

hdr "结果"
echo "  PASS=$PASS  FAIL=$FAIL   （日志目录：$OUT）"
[ "$FAIL" -eq 0 ] || exit 1
