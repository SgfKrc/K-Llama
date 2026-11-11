#!/bin/bash
# 只读重测：2B 链上 MTP 的「每负载加速比」——docs/SPLIT-MEASUREMENTS.md 需要出处。
# off 与 k=2 各 3 个 prompt × 3 次；贪心、-np 1、n_predict 192、三次取中位数。
# 只用 93xx；跑完杀掉自己起的进程。路径约定见 _qlh_env.sh。
set -u
source "$(dirname "${BASH_SOURCE[0]}")/_qlh_env.sh"
ROOT=$QLH_SRC_DIR
OUT=$QLH_TMP
MODEL=${QLH_SPLIT_MODEL:-H:/qlh_models/qwen3-5-2b-gguf/qwen35-2b-Q4_K_M.gguf}
PORT=9379
TS=$(date +%H%M%S)

kill_ours() {
    for pid in $(netstat -ano 2>/dev/null | grep LISTENING | grep -E ":$PORT[[:space:]]" \
                 | awk '{print $NF}' | tr -d '\r' | sort -u); do
        taskkill //F //PID "$pid" >/dev/null 2>&1
    done
    sleep 2
}

wait_health() {
    for _ in $(seq 1 120); do
        [ "$(curl -s -o /dev/null -w '%{http_code}' "http://127.0.0.1:$PORT/health" 2>/dev/null)" = "200" ] && return 0
        sleep 2
    done
    return 1
}

ask() { # $1=tag
    python - "$1" "$PORT" <<'PY'
import json, sys, urllib.request
tag, port = sys.argv[1], sys.argv[2]
prompts = [
    ("code",   "def quicksort(arr):\n    if len(arr) <= 1:\n        return arr\n    pivot = arr[len(arr) // 2]\n    left = [x for x in arr if x < pivot]\n    mid = [x for x in arr if x == pivot]\n    right = [x for x in arr if x > pivot]\n    return quicksort(left) + mid + quicksort(right)\n\n# Usage:\nresult = quicksort([3, 6, 8, 10, 1, 2, 1])\nprint(result)\n# The function works by"),
    ("mixed",  "请用 Python 写一个函数，返回斐波那契数列的第 n 项（用记忆化）。\n\n```python\ndef fib(n, memo={}):\n    if n in memo:\n        return memo[n]\n    if n <= 1:\n        return n\n    memo[n] = fib(n - 1, memo) + fib(n - 2, memo)\n    return memo[n]\n```\n\n这个实现的时间复杂度是"),
    ("zh",     "中国的首都是北京。北京是中国的政治、文化中心，也是世界上人口最多的城市之一。北京有着悠久的历史，"),
]
for i, (name, p) in enumerate(prompts):
    body = {"prompt": p, "n_predict": 192, "temperature": 0, "top_k": 1, "cache_prompt": False}
    req = urllib.request.Request(f"http://127.0.0.1:{port}/completion",
                                 data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    d = json.loads(urllib.request.urlopen(req, timeout=600).read().decode('utf-8', 'surrogateescape'))
    t = d.get("timings", {})
    print(f"{tag} {name}: prompt_n={t.get('prompt_n')} predicted_n={t.get('predicted_n')} "
          f"ms/tok={t.get('predicted_per_token_ms'):.3f} tok/s={t.get('predicted_per_second'):.2f} "
          f"draft_n={t.get('draft_n')} acc={t.get('draft_n_accepted')}", flush=True)
PY
}

kill_ours
for cfg in off k2; do
    case $cfg in
        off) SPEC="";;
        k2)  SPEC="--spec-type draft-mtp --spec-draft-n-max 2";;
    esac
    echo "=== chain 0:20+20:24 cfg=$cfg ==="
    ( cd "$BIN" && ./llama-server.exe -m "$MODEL" --host 127.0.0.1 --port "$PORT" \
        -c 4096 -ngl 0 --no-warmup -lv 3 -np 1 \
        --split-layer-range 0:20 --split-tail-range 20:24 $SPEC \
        > "$OUT/mtp_re_${cfg}_${TS}.srv.log" 2>&1 & )
    wait_health || { echo "$cfg FAILED"; tail -20 "$OUT/mtp_re_${cfg}_${TS}.srv.log"; kill_ours; exit 1; }
    # 预热一轮（不进统计）
    ask "${cfg}_warm" > /dev/null
    for r in 1 2 3; do ask "${cfg}_r$r"; done
    grep -E 'MTP draft|n_rs_seq|draft acceptance' "$OUT/mtp_re_${cfg}_${TS}.srv.log" | head -5
    kill_ours
done
echo DONE
