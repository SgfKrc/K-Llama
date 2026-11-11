#!/usr/bin/env bash
# export_patches.sh —— 从 fork 的 HEAD 重新导出 docs/llamacpp-*.patch，并证明
# 两个补丁叠起来正好等于 HEAD（overlay 判据）。
#
# 用法：
#   bash scripts/export_patches.sh [--fork H:/llamacpp-c] [--base <commit>] [--write]
#
#   --fork <path>   上游 fork 工作区（默认 H:/llamacpp-c）
#   --base <rev>    主补丁的基底（默认 fork HEAD 的父提交）
#   --write         校验通过后覆盖 docs/llamacpp-split.patch 与 docs/llamacpp-slimarc.patch
#                   不带 --write 时只打印统计，不写任何文件
#
# 流程：
#   1. 在 fork 里建临时 worktree 到 <base>（不会碰 fork 的 index / 工作区）。
#   2. git apply 主补丁 → 树 A；再 git apply 增量补丁 → 树 B。
#   3. 断言 树 B == fork HEAD 的树。不通过则 exit 1，一个字节都不写。
#   4. 用 git diff <base> <A> / <A> <B> 重新生成两个补丁，并回放一遍确认能重建 A。
#   5. 无论成败都移除临时 worktree（trap EXIT）。

set -euo pipefail

SELF_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd -- "$SELF_DIR/.." && pwd)
DOCS="$REPO_ROOT/docs"

FORK="H:/llamacpp-c"
BASE_REF=""
WRITE=0

usage() {
    cat <<'EOF'
usage: bash scripts/export_patches.sh [--fork H:/llamacpp-c] [--base <commit>] [--write]
  --fork <path>   上游 fork 工作区（默认 H:/llamacpp-c）
  --base <rev>    主补丁基底（默认 fork HEAD 的父提交）
  --write         校验通过后覆盖 docs/llamacpp-split.patch 与 docs/llamacpp-slimarc.patch
EOF
}

while [ $# -gt 0 ]; do
    case "$1" in
        --fork)   FORK="${2:?--fork needs a path}"; shift 2 ;;
        --fork=*) FORK="${1#*=}"; shift ;;
        --base)   BASE_REF="${2:?--base needs a revision}"; shift 2 ;;
        --base=*) BASE_REF="${1#*=}"; shift ;;
        --write)  WRITE=1; shift ;;
        -h|--help) usage; exit 0 ;;
        *) echo "error: unknown argument: $1" >&2; usage >&2; exit 2 ;;
    esac
done

# ---------- 前置检查 ----------
if ! git -C "$FORK" rev-parse --git-dir >/dev/null 2>&1; then
    echo "error: not a git work tree: $FORK" >&2
    exit 2
fi
FORK_DIR=$(cd -- "$FORK" && { pwd -W 2>/dev/null || pwd; })

MAIN_PATCH="$DOCS/llamacpp-split.patch"
INCR_PATCH="$DOCS/llamacpp-slimarc.patch"
for p in "$MAIN_PATCH" "$INCR_PATCH"; do
    if [ ! -f "$p" ]; then
        echo "error: missing patch: $p" >&2
        exit 2
    fi
done

if [ -z "$BASE_REF" ]; then
    BASE_REF="HEAD^"
fi
BASE_SHA=$(git -C "$FORK" rev-parse --verify "$BASE_REF^{commit}")
HEAD_SHA=$(git -C "$FORK" rev-parse --verify 'HEAD^{commit}')
HEAD_TREE=$(git -C "$FORK" rev-parse "$HEAD_SHA^{tree}")

# ---------- 临时目录（与 fork 同盘，避免跨卷 worktree） ----------
PARENT=$(dirname -- "$FORK_DIR")
WT="$PARENT/.qlh-export-$$"
TMPD="$PARENT/.qlh-export-tmp-$$"
mkdir -p -- "$TMPD"

cleanup() {
    set +e
    if [ -n "${WT:-}" ] && [ -e "$WT" ]; then
        git -C "$FORK" worktree remove --force -- "$WT" >/dev/null 2>&1 \
            || rm -rf -- "$WT"
    fi
    git -C "$FORK" worktree prune >/dev/null 2>&1
    [ -n "${TMPD:-}" ] && rm -rf -- "$TMPD"
}
trap cleanup EXIT INT TERM

echo "fork:      $FORK_DIR"
echo "base:      $BASE_SHA ($BASE_REF)"
echo "HEAD:      $HEAD_SHA"
echo "HEAD tree: $HEAD_TREE"
echo

# ---------- 1) worktree 到 base ----------
git -C "$FORK" -c core.autocrlf=false worktree add --detach -- "$WT" "$BASE_SHA" >/dev/null
git -C "$WT" config core.autocrlf false

# ---------- 2) 主补丁 → 树 A ----------
git -C "$WT" apply --whitespace=nowarn -- "$MAIN_PATCH"
git -C "$WT" add -A -f -- .
TREE_A=$(git -C "$WT" write-tree)

# ---------- 3) 增量补丁 → 树 B ----------
git -C "$WT" apply --whitespace=nowarn -- "$INCR_PATCH"
git -C "$WT" add -A -f -- .
TREE_B=$(git -C "$WT" write-tree)

echo "tree A:    $TREE_A  (base + main)"
echo "tree B:    $TREE_B  (base + main + incr)"
if [ "$TREE_B" != "$HEAD_TREE" ]; then
    echo "FAIL: overlay mismatch — tree B != fork HEAD tree" >&2
    echo "      expected $HEAD_TREE" >&2
    echo "      got      $TREE_B" >&2
    echo "      nothing written" >&2
    exit 1
fi
echo "overlay:   OK (tree B == HEAD tree)"
echo

# ---------- 4) 重新生成两个补丁并回放 ----------
git -C "$WT" diff --no-color --no-ext-diff "$BASE_SHA" "$TREE_A" > "$TMPD/main.patch"
git -C "$WT" diff --no-color --no-ext-diff "$TREE_A" "$TREE_B" > "$TMPD/incr.patch"

git -C "$WT" read-tree --reset -u -- "$BASE_SHA"
git -C "$WT" apply --check -- "$TMPD/main.patch"
git -C "$WT" apply --whitespace=nowarn -- "$TMPD/main.patch"
git -C "$WT" add -A -f -- .
REPLAY_A=$(git -C "$WT" write-tree)
if [ "$REPLAY_A" != "$TREE_A" ]; then
    echo "FAIL: regenerated main patch does not replay to tree A" >&2
    exit 1
fi

# ---------- 5) 统计 ----------
cr_count() {
    awk '{ n = gsub(/\r/, "&"); t += n } END { print t + 0 }' "$1"
}

print_stats() {
    local label=$1 file=$2
    local files adds dels lines bytes kib cr
    read -r files adds dels < <(git -C "$WT" apply --numstat -- "$file" \
        | awk '{ a += $1; d += $2 } END { printf "%d %d %d\n", NR, a, d }')
    lines=$(wc -l < "$file" | tr -d ' ')
    bytes=$(wc -c < "$file" | tr -d ' ')
    kib=$(awk -v b="$bytes" 'BEGIN { printf "%.1f", b / 1024 }')
    cr=$(cr_count "$file")
    printf '%-10s %2d files  +%-6d -%-5d %6d lines  %8s KiB  CR=%d\n' \
        "$label" "$files" "$adds" "$dels" "$lines" "$kib" "$cr"
    if [ "$cr" -ne 0 ]; then
        echo "FAIL: $label patch contains CR bytes (must be LF)" >&2
        exit 1
    fi
}

echo "patch stats (regenerated):"
print_stats "main" "$TMPD/main.patch"
print_stats "incr" "$TMPD/incr.patch"
echo

# ---------- 6) 写盘（只有 --write） ----------
if [ "$WRITE" -eq 1 ]; then
    cp -- "$TMPD/main.patch" "$MAIN_PATCH"
    cp -- "$TMPD/incr.patch" "$INCR_PATCH"
    echo "wrote: docs/llamacpp-split.patch"
    echo "wrote: docs/llamacpp-slimarc.patch"
else
    echo "dry run: docs/llamacpp-*.patch unchanged (pass --write to replace them)"
fi
