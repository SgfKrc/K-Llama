# shellcheck shell=bash
# verify/ 的共享路径约定。被各 .sh `source`，不单独执行。
#
# 三个变量，都有缺省值，都能用环境变量覆盖：
#   QLH_BUILD_DIR  上游 llama.cpp 的**构建目录**（它下面有 bin/）。缺省 H:/llamacpp-c/build-split。
#   QLH_SRC_DIR    上游源码根（stage / server 的 cwd，构建脚本的 cwd）。缺省 = QLH_BUILD_DIR 的父目录。
#   QLH_TMP        日志与临时工件的落点。缺省 <本仓库>/.qlh_tmp/verify（本机 /tmp 不可靠）。
#
# 路径形式：H:/x、H:\x、/h/x 三种写法都收，内部统一成 Windows 形式（MSYS 的 bash 认它，
# exe 和 python 也认它）。导出给子进程的也是这一种，避免两层脚本各解释一遍。
#
# 这里只做定位，不猜实现：缺二进制就报错退出，绝不静默跳过某一项。

# [C3] 路径规范化的**唯一事实源**。verify/_qlh_env.py 的 _norm() 是同一张规则表的 Python
# 版：跨语言共用要付一次子进程开销，所以保留两份实现，规则必须一字不差——
#   1. 反斜杠 -> 正斜杠
#   2. ^[A-Za-z]:/  -> 盘符大写
#   3. ^/[A-Za-z]/  -> 盘符大写 + ':'（MSYS 的 /h/x 写法）
#   4. 去掉尾部斜杠
# 改这里必须同步改那边；两处注释互相指名，就是为了让人一眼看到「还有另一份」。
_qlh_norm() {   # H:\x / H:/x / /h/x -> H:/x
    local p=${1//\\//}
    case "$p" in
        [A-Za-z]:/*) p="$(printf '%s' "${p:0:1}" | tr 'a-z' 'A-Z'):${p:2}" ;;
        /[A-Za-z]/*) p="$(printf '%s' "${p:1:1}" | tr 'a-z' 'A-Z'):${p:2}" ;;
    esac
    p=${p%/}
    printf '%s\n' "$p"
}

QLH_VERIFY_DIR=$(_qlh_norm "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)")
QLH_REPO_DIR=$(_qlh_norm "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)")

if [ -n "${QLH_BUILD_DIR+x}" ] && [ -n "${QLH_BUILD_DIR}" ]; then
    QLH_BUILD_SOURCE=env
else
    QLH_BUILD_DIR=H:/llamacpp-c/build-split
    QLH_BUILD_SOURCE=default
fi
QLH_BUILD_DIR=$(_qlh_norm "$QLH_BUILD_DIR")
BIN=$QLH_BUILD_DIR/bin

if [ -n "${QLH_SRC_DIR:-}" ]; then
    QLH_SRC_DIR=$(_qlh_norm "$QLH_SRC_DIR")
else
    QLH_SRC_DIR=$(_qlh_norm "$(dirname "$QLH_BUILD_DIR")")
fi

QLH_TMP=$(_qlh_norm "${QLH_TMP:-$QLH_REPO_DIR/.qlh_tmp/verify}")
mkdir -p "$QLH_TMP" 2>/dev/null

# [C4] `--rs-seq` 的取值：`llama-server --split-rs-seq` 与每一段的 `split-stage --rs-seq`
# 必须是同一个数，否则链能起来但 KV 的 rs 分区对不上。以前五个子脚本各自手写 8，改一处漏
# 一处就是一次排查；这里放唯一事实源，要换就设 QLH_RS_SEQ。
QLH_RS_SEQ=${QLH_RS_SEQ:-8}

# 推导结果如实打出来：指错目录时要能一眼看出它到底读了哪儿。
qlh_show_env() {
    echo "  [env] QLH_BUILD_DIR=$QLH_BUILD_DIR（$QLH_BUILD_SOURCE）"
    echo "  [env] QLH_SRC_DIR=$QLH_SRC_DIR"
    echo "  [env] QLH_TMP=$QLH_TMP"
}

# 缺哪个二进制就点名，并给出该设什么；退出码 3 专用于「环境没准备好」，与判据 FAIL(1) 分开。
qlh_require_bins() {
    local missing=() n
    for n in "$@"; do
        [ -x "$BIN/$n" ] || missing+=("$n")
    done
    [ ${#missing[@]} -eq 0 ] && return 0
    echo "错误：$BIN 下缺少可执行文件：${missing[*]}" >&2
    if [ -d "$BIN" ]; then
        echo "  该目录存在，但没有上面这些文件；它是不是别的构建目录？" >&2
    else
        echo "  该目录不存在。" >&2
    fi
    echo "  QLH_BUILD_DIR 现在 = $QLH_BUILD_DIR（来源：$QLH_BUILD_SOURCE）" >&2
    echo "  请把它指向 llama.cpp 的构建目录（其下有 bin/），例如：" >&2
    echo "    QLH_BUILD_DIR=H:/llamacpp-c/build-split bash verify/verify.sh --skip-build" >&2
    echo "  构建方式见 verify/README.md。" >&2
    exit 3
}

# 组 9/10/11 的内存闸门：**一份口径，别在每个脚本里重写一遍**。
#
# 「可用」= 任务管理器那个数（空闲 + 可回收 standby）。只看 FreePhysicalMemory 会低估一台
# 其实跑得动的机器（mmap 的模型不需要整个装进物理内存），实测两者差约 3 GB。
qlh_available_gb() {
    local mb
    mb=$(powershell -NoProfile -Command "\$p = Get-Counter '\Memory\Available MBytes' -ErrorAction SilentlyContinue; if (\$p) { [int]\$p.CounterSamples[0].CookedValue } else { [int]((Get-CimInstance Win32_OperatingSystem).FreePhysicalMemory/1KB) }" 2>/dev/null | tr -d '\r[:space:]')
    case "$mb" in ''|*[!0-9]*) echo 0 ;; *) echo $((mb / 1024)) ;; esac
}

# 刚杀完进程时读会读低：几十秒内内核才把 WORKING SET 真正还回去。
# 实测同一时刻：清理后立刻读 19 GB（于是误判不足），稳定后 22 GB。所以取 4 次的最大值再判断，
# 否则闸门会在边界上随机抖，变成另一种「环境造成的假失败」。
qlh_available_gb_settled() {
    local i cur best=0
    for i in 1 2 3 4; do
        cur=$(qlh_available_gb)
        [ "$cur" -gt "$best" ] && best=$cur
        sleep 2
    done
    echo "$best"
}

export QLH_BUILD_DIR QLH_SRC_DIR QLH_TMP QLH_VERIFY_DIR QLH_REPO_DIR QLH_RS_SEQ
