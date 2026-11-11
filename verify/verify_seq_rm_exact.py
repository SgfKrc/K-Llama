"""seq_rm 的"删尾部再重算同一行"必须逐位不变 —— 回归测试（不依赖投机、不依赖分段）。

判据（为什么这必须成立）
------------------------------------------------------------
`seq_rm(p, -1)` 删掉 p 之后的 KV，不改变"第 p 行"的输入：它的注意力只覆盖 0..p（含自己），
第 p 个 token 也没变。所以重新计算第 p 行必须得到**逐位相同**的 logits。任何一条"回滚后重算"
的路径都建立在这个前提上：投机验证帧被拒后的回退、prefix cache 复用、context checkpoint 恢复。

用例（全部只加载 2B 整模，不 split、不起服务、不占端口）
------------------------------------------------------------
  A. n_rs_seq = 0（不投机 server 的默认 -> FULL 语义）：
     `llama_memory_hybrid::seq_rm` 先删 recurrent，recurrent 因为 n_rs_seq=0 拒绝部分回退，
     于是**整个 seq_rm 返回 false，连 attention 的 KV 一行都没删**。
     当前结果：seq_rm=false、pos_max 不变、重放失败（或写到错误位置）-> 用例**红**。
     这是上游 hybrid memory 的缺陷，整模即可复现，与本仓库的 split/投机改造无关。
  B. n_rs_seq = 3（投机一帧所需的快照数 -> RS 语义）：
     2026-11 rebase 到 f477f795c 之后**当前结果：seq_rm=false、一行都没删 -> 用例红**。
     原因不是本地缺陷，是上游 ab3768cdc 改了语义："允许回退多少"从 `rollback <= n_rs_seq`
     变成 `rollback <= cell.rs_max`，而 `rs_max = 本 ubatch 写入的快照数 - 1 = n_seq_tokens - 1`。
     本 probe 是**逐 token decode**（每帧一个 token），所以 rs_max = 0，任何回退都必须被拒。
     产品形状下的回退（投机验证帧 = n_draft + 1 行放在一个 ubatch，回退 n_draft 行）由
     verify_grammar_spec.sh（4 组「带回退且逐字节相同」）与 verify_rbudget_spec.sh
     （整帧回退 seq_rm 到 tail 成功、回退 6 次仍逐字节相同）覆盖，那两组是全绿的。
     要让 B 组重新变绿，得让 probe 把尾部 token 放进一个多 token 的 ubatch —— 但那样
     "重放 logits 逐位相同"就不再成立（批量 1 行与批量 k+1 行走不同 kernel），判据本身
     会变成不可靠，所以这里选择更新基线而不是改 probe。
  C. n_rs_seq = 3 但回退 k = 4：
     越界，seq_rm 必须**明确拒绝**（返回 false），而不是删一半。
     当前结果：false -> 用例**绿**（拒绝是对的；调用方必须检查返回值，这是另一条）。

用法：
  python verify/verify_seq_rm_exact.py                      # 严格：有红即失败（当前预期非 0）
  python verify/verify_seq_rm_exact.py --expect-known-fail
      # 已知缺陷基线：A（FULL 不删）+ B（rs_max=0，逐 token 形状下恒拒）红 + C 绿 -> 0
"""
import os
import re
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _qlh_env import BIN, ROOT, TMP, build_dir   # noqa: E402
_HERE = os.path.dirname(os.path.abspath(__file__))
EXE = os.path.join(BIN, "seq_rm_exact.exe")
# 探针源码入库在 verify/probes/：build-split/ 是构建产物目录，它不在那儿。
SRC = os.path.join(_HERE, "probes", "seq_rm_exact.cpp")
BUILD_BAT = os.path.join(_HERE, "probes", "build_seq_rm_exact.bat")
MODEL = os.environ.get("QLH_SEQ_RM_MODEL", r"H:\qlh_models\qwen3-5-2b-gguf\qwen35-2b-Q4_K_M.gguf")
P = 29

# name, n_rs_seq, k, k_replay, expect
#   expect="exact"     : seq_rm 必须 true、pos_max 必须降到 p-1、重放 logits 必须逐位相同
#   expect="refuse"    : seq_rm 必须 false（快照窗口不够时宁可拒绝）
CASES = [
    ("A1 FULL n_rs_seq=0, k=1", 0, 1, 1, "exact"),
    ("A2 FULL n_rs_seq=0, k=3", 0, 3, 1, "exact"),
    ("B1 RS  n_rs_seq=3, k=1", 3, 1, 1, "exact"),
    ("B2 RS  n_rs_seq=3, k=3, 重放 1 行（投机回退的真实形状）", 3, 3, 1, "exact"),
    ("B3 RS  n_rs_seq=3, k=3, 重放 3 行", 3, 3, 3, "exact"),
    ("C1 RS  n_rs_seq=3, k=4 > 窗口", 3, 4, 1, "refuse"),
]

FETCH = re.compile(r"\[seq_rm_exact\]\s+(.*)")


def _exe_is_stale():
    r"""exe 比探针源码或构建脚本旧 -> 必须重编。

    旧版只判 `os.path.isfile(EXE)`：源码改了、bat 坏了都检测不到，一直拿旧 exe 跑，
    报出来的 9 项是历史构建的结果。2026-11-11 就撞上过：`build_seq_rm_exact.bat` 的
    「vcvars 不存在」分支写成 `if ... ( echo %QLH_VCVARS% )`，而 `%QLH_VCVARS%` 展开成
    `C:\Program Files (x86)\...`，`)` 提前闭合块，整个 bat 返回 255 —— 但 exe 一直在，
    这一层把它完全掩盖了（打印的「9 项」全是旧二进制跑出来的）。
    """
    if not os.path.isfile(EXE):
        return True
    exe_m = os.path.getmtime(EXE)
    for src in (SRC, BUILD_BAT):
        if os.path.isfile(src) and os.path.getmtime(src) > exe_m:
            return True
    return False


def ensure_exe():
    if not _exe_is_stale():
        return True
    if not os.path.isfile(BUILD_BAT):
        print("missing %s and cannot build (%s absent)" % (EXE, BUILD_BAT))
        print("  QLH_BUILD_DIR = %s（该目录下应有 bin/seq_rm_exact.exe）" % build_dir())
        return False
    print("building %s ..." % EXE)
    # 参数依次是：上游源码根 / 输出 bin / obj 临时目录 / 构建目录（llama.lib 所在的那个）
    # errors="replace"：cl / cmd 的输出按 locale 编码（这台机器是 GBK），不要因为一个坏字节
    # 抛 UnicodeDecodeError —— 那样反而看不到真正的编译错误。
    rc = subprocess.run(["cmd", "/c", BUILD_BAT, ROOT, BIN, TMP, build_dir()],
                        capture_output=True, text=True, errors="replace")
    if _exe_is_stale():
        print("build failed:\n" + (rc.stdout or "")[-2000:] + (rc.stderr or "")[-2000:])
        print("  QLH_BUILD_DIR = %s  QLH_SRC_DIR = %s" % (build_dir(), ROOT))
        print("  构建方式见 verify/README.md（探针需要上游树已构建过 llama.lib，以及 MSVC 的 cl）。")
        return False
    return True


def parse(out):
    d = {}
    for line in out.splitlines():
        m = FETCH.search(line)
        if not m:
            continue
        body = m.group(1).strip()
        if body.startswith("RESULT "):
            d["result"] = body.split()[1]
        elif body.startswith("seq_rm="):
            mm = re.match(r"seq_rm=(true|false) pos_max (-?\d+) -> (-?\d+)", body)
            if mm:
                d["rm"] = mm.group(1) == "true"
                d["pos_before"] = int(mm.group(2))
                d["pos_after"] = int(mm.group(3))
        elif body.startswith("replay rc="):
            mm = re.match(r"replay rc=(-?\d+) pos_max=(-?\d+)", body)
            if mm:
                d["rc"] = int(mm.group(1))
                d["pos_replay"] = int(mm.group(2))
        elif body.startswith("logits diff="):
            mm = re.match(r"logits diff=(\d+)/(\d+) first_diff=(-?\d+) max_abs=(\S+)", body)
            if mm:
                d["diff"] = int(mm.group(1))
                d["n_vocab"] = int(mm.group(2))
                d["first_diff"] = int(mm.group(3))
                d["max_abs"] = mm.group(4)
        elif body.startswith("argmax "):
            mm = re.match(r"argmax first=(-?\d+) second=(-?\d+)", body)
            if mm:
                d["am1"] = int(mm.group(1))
                d["am2"] = int(mm.group(2))
    return d


def judge(name, rs, k, krep, expect, d):
    """返回 (ok, 说明)。ok=False 就是"红"。"""
    if "rm" not in d:
        return False, "探针没有输出可解析的行（崩溃？）"
    if expect == "refuse":
        if d["rm"]:
            return False, "快照窗口不够却返回了 true（删一半比拒绝更糟）"
        return True, "明确拒绝（false），符合预期"
    # expect == "exact"
    if not d["rm"]:
        return False, ("seq_rm 返回 false，pos_max %s -> %s：尾部 KV 一行都没删"
                       % (d.get("pos_before"), d.get("pos_after")))
    if d.get("pos_after") != P - 1:
        return False, "seq_rm 返回 true 但 pos_max=%s，应为 %d" % (d.get("pos_after"), P - 1)
    if d.get("rc") != 0:
        return False, "seq_rm 之后重放 decode 失败 rc=%s" % d.get("rc")
    if d.get("diff", -1) != 0:
        return False, ("重放 logits 不逐位相同：diff=%s/%s, first_diff=%s, max_abs=%s, argmax %s->%s"
                       % (d.get("diff"), d.get("n_vocab"), d.get("first_diff"), d.get("max_abs"),
                          d.get("am1"), d.get("am2")))
    return True, "seq_rm 成功、pos_max=%d、重放 logits 逐位相同" % (P - 1)


def main():
    expect_known_fail = "--expect-known-fail" in sys.argv[1:]
    if not os.path.isfile(MODEL):
        print("missing model: %s" % MODEL)
        return 2
    if not ensure_exe():
        return 2

    print("model: %s" % MODEL)
    print("probe: %s  (p=%d, 整模，CPU，n_threads=4)" % (EXE, P))

    red, green = [], []
    for name, rs, k, krep, expect in CASES:
        proc = subprocess.run([EXE, MODEL, str(rs), str(P), str(k), str(krep)],
                              capture_output=True, text=True, cwd=os.path.dirname(EXE))
        d = parse(proc.stdout or "")
        ok, why = judge(name, rs, k, krep, expect, d)
        (green if ok else red).append((name, why))
        print("  [%s] %-52s %s" % ("PASS" if ok else "FAIL", name, why))

    print()
    print("PASS=%d FAIL=%d" % (len(green), len(red)))

    # 当前已知形态（2026-11，rebase 到 f477f795c 之后）：
    #   A 组：FULL（n_rs_seq=0）的部分回退本来就失败 —— 旧上游缺陷，未变；
    #   B 组：rs_max 语义下，逐 token decode 的帧 rs_max = 0，任何回退都必须被拒 ——
    #         上游 ab3768cdc 的设计决定，见文件头 B 段的说明；
    #   C 组：越界仍然正确拒绝。
    # --expect-known-fail 只承认这一种红（A1 A2 B1 B2 B3，C 绿）。
    known = {name for name, _ in red}
    print()
    if not red:
        print("全部通过：seq_rm 的删尾部语义在两种形态下都精确。"
              "（如果这是修复之后的第一次运行，请把本文件的 --expect-known-fail 默认关掉并更新注释）")
        return 0
    print("红：")
    for name, why in red:
        print("  - %s：%s" % (name, why))
    if expect_known_fail:
        expected = {"A1 FULL n_rs_seq=0, k=1", "A2 FULL n_rs_seq=0, k=3",
                    "B1 RS  n_rs_seq=3, k=1", "B2 RS  n_rs_seq=3, k=3, 重放 1 行（投机回退的真实形状）",
                    "B3 RS  n_rs_seq=3, k=3, 重放 3 行"}
        if known == expected:
            print("\nKNOWN-FAIL：A 组是 n_rs_seq=0（FULL）的旧缺陷，B 组是上游 rs_max 语义下"
                  "「逐 token 帧不允许回退」的必然结果（ab3768cdc），C 组正确地拒绝越界。"
                  "产品形状下的回退由 grammar/rbudget 两组覆盖。")
            return 0
        print("\nKNOWN-FAIL 不匹配：红的集合变了（%s），行为变了，请重新评估。"
              % ", ".join(sorted(known)))
        return 1
    print("\n（当前是预期中的红：n_rs_seq=0 时 hybrid 的 seq_rm 部分回退失败。"
          "加 --expect-known-fail 可把它当作已知缺陷基线跑。）")
    return 1


if __name__ == "__main__":
    sys.exit(main())
