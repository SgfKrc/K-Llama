#!/usr/bin/env bash
# export_patches.sh —— 从 fork 的 HEAD 重新导出 docs/llamacpp-*.patch，并证明
# 三个补丁叠起来正好等于 HEAD（overlay 判据）。
#
# 用法：
#   bash scripts/export_patches.sh [--fork H:/llamacpp-c] [--base <commit>] [--write]
#
#   --fork <path>   上游 fork 工作区（默认 H:/llamacpp-c）
#   --base <rev>    主补丁的基底（默认 fork HEAD 的祖先里最后一个上游提交）
#   --write         校验通过后覆盖三个补丁
#
# 三个补丁（v2.2 起）：
#   llamacpp-split.patch          主补丁：层范围加载 + 分段链 + 跨进程协议 + SLIM-ARC prefetch
#   llamacpp-vulkan-igpu.patch    Intel iGPU 四处修复（可单独提上游）
#   llamacpp-moe-pin.patch        MoE cache 的 host 专家页 pinning（Linux 才生效）
#
# 流程：
#   1. 在 fork 里建临时 worktree 到 <base>（不会碰 fork 的 index / 工作区）。
#   2. 依次 git apply 三个补丁，得到树 A / B / C。
#   3. 断言 树 C == fork HEAD 的树。不通过则 exit 1，一个字节都不写。
#   4. 用 git diff 重新生成三个补丁，回放一遍确认能重建树 C。
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
  --base <rev>    主补丁基底（默认 fork HEAD 的祖先里最后一个上游提交）
  --write         校验通过后覆盖 docs/llamacpp-split.patch / -vulkan-igpu.patch / -moe-pin.patch
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

PATCHES=(
    "llamacpp-split.patch"
    "llamacpp-vulkan-igpu.patch"
    "llamacpp-moe-pin.patch"
)

if [ -z "$BASE_REF" ]; then
    # The last upstream commit reachable from HEAD: walk back until the parent chain
    # leaves our commits. `git describe` is not used because our commits are not tagged.
    BASE_REF=$(git -C "$FORK" rev-list --max-parents=0 HEAD | head -1)
    # Prefer an explicit upstream base when the caller passes one; otherwise pick the
    # commit whose author is not ours by looking at the first commit with a merge-base.
    BASE_REF=""
fi

if [ -z "$BASE_REF" ]; then
    echo "error: --base is required (pass the upstream commit the main patch applies to)" >&2
    exit 2
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
echo "base:      $BASE_SHA"
echo "HEAD:      $HEAD_SHA"
echo "HEAD tree: $HEAD_TREE"
echo

# ---------- 1) worktree 到 base ----------
git -C "$FORK" -c core.autocrlf=false worktree add --detach -- "$WT" "$BASE_SHA" >/dev/null
git -C "$WT" config core.autocrlf false

# ---------- 2) 依次 apply，记下每一层的树 ----------
TREES=()
for p in "${PATCHES[@]}"; do
    if [ ! -f "$DOCS/$p" ]; then
        echo "error: missing patch: $DOCS/$p" >&2
        exit 2
    fi
    git -C "$WT" apply --whitespace=nowarn -- "$DOCS/$p"
    git -C "$WT" add -A -f -- .
    TREES+=("$(git -C "$WT" write-tree)")
done

echo "trees:"
for i in "${!PATCHES[@]}"; do
    printf '  %-28s %s\n' "${PATCHES[$i]}" "${TREES[$i]}"
done
if [ "${TREES[-1]}" != "$HEAD_TREE" ]; then
    echo "FAIL: overlay mismatch — last tree != fork HEAD tree" >&2
    echo "      expected $HEAD_TREE" >&2
    echo "      got      ${TREES[-1]}" >&2
    echo "      nothing written" >&2
    exit 1
fi
echo "overlay:   OK (all three patches stack to the HEAD tree)"
echo

# ---------- 3) 重新生成三个补丁并回放 ----------
PREV="$BASE_SHA"
GEN=()
for i in "${!PATCHES[@]}"; do
    out="$TMPD/${PATCHES[$i]}"
    git -C "$WT" diff --no-color --no-ext-diff "$PREV" "${TREES[$i]}" > "$out"
    GEN+=("$out")
    PREV="${TREES[$i]}"
done

git -C "$WT" read-tree --reset -u -- "$BASE_SHA"
for g in "${GEN[@]}"; do
    git -C "$WT" apply --whitespace=nowarn -- "$g"
    git -C "$WT" add -A -f -- .
done
REPLAY_TREE=$(git -C "$WT" write-tree)
if [ "$REPLAY_TREE" != "$HEAD_TREE" ]; then
    echo "FAIL: regenerated patches do not replay to the HEAD tree" >&2
    exit 1
fi
echo "replay:    OK (regenerated patches rebuild the HEAD tree)"
echo

# ---------- 4) 统计 ----------
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
    printf '%-12s %2d files  +%-6d -%-5d %6d lines  %8s KiB  CR=%d\n' \
        "$label" "$files" "$adds" "$dels" "$lines" "$kib" "$cr"
    if [ "$cr" -ne 0 ]; then
        echo "FAIL: $label patch contains CR bytes (must be LF)" >&2
        exit 1
    fi
}

echo "patch stats (regenerated):"
for i in "${!PATCHES[@]}"; do
    print_stats "${PATCHES[$i]}" "${GEN[$i]}"
done
echo

# ---------- 5) 写盘（只有 --write） ----------
if [ "$WRITE" -eq 1 ]; then
    for i in "${!PATCHES[@]}"; do
        cp -- "${GEN[$i]}" "$DOCS/${PATCHES[$i]}"
        echo "wrote: docs/${PATCHES[$i]}"
    done
else
    echo "dry run: docs/llamacpp-*.patch unchanged (pass --write to replace them)"
fi
