"""verify/ 的共享路径约定（Python 侧）。与 _qlh_env.sh 同一套规则。

    QLH_BUILD_DIR  上游 llama.cpp 的构建目录（其下有 bin/）。缺省 H:/llamacpp-c/build-split。
    QLH_SRC_DIR    上游源码根（stage / server 的 cwd）。缺省 = QLH_BUILD_DIR 的父目录。
    QLH_TMP        日志与临时工件的落点。缺省 <本仓库>/.qlh_tmp/verify。

只做定位，不猜实现：缺二进制就点名退出（rc=3），绝不静默跳过某一项。
"""
from __future__ import annotations

import os
import re
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(_HERE)


def _norm(p: str) -> str:
    """H:\\x / H:/x / /h/x -> H:/x（Windows 原生形式，exe 与 python 都认）。

    [C3] 与 verify/_qlh_env.sh 的 _qlh_norm() 是同一张规则表的两份实现（跨语言共用要付一次
    子进程开销，所以不合并成一份）。规则，两边必须一致：
      1. 反斜杠 -> 正斜杠
      2. ^[A-Za-z]:/  -> 盘符大写
      3. ^/[A-Za-z]/  -> 盘符大写 + ':'（MSYS 的 /h/x 写法）
      4. 去掉尾部斜杠
    改这里必须同步改那边；两处注释互相指名，就是为了让人一眼看到「还有另一份」。
    """
    p = str(p).replace("\\", "/")
    if re.match(r"^[A-Za-z]:/", p):
        p = p[0].upper() + p[1:]
    elif re.match(r"^/[A-Za-z]/", p):
        p = p[1].upper() + ":" + p[2:]
    return p.rstrip("/") or p


def build_dir() -> str:
    return _norm(os.environ.get("QLH_BUILD_DIR") or "H:/llamacpp-c/build-split")


def bin_dir() -> str:
    return build_dir() + "/bin"


def src_root() -> str:
    v = os.environ.get("QLH_SRC_DIR")
    return _norm(v) if v else os.path.dirname(build_dir())


def tmp_dir() -> str:
    d = _norm(os.environ.get("QLH_TMP") or os.path.join(_REPO, ".qlh_tmp", "verify"))
    os.makedirs(d, exist_ok=True)
    return d


def require(*names: str, group: str = "") -> None:
    """缺哪一个就点名退出（rc=3），并说清楚该设什么。"""
    b = bin_dir()
    missing = [n for n in names if not os.access(os.path.join(b, n), os.R_OK)]
    if missing:
        where = "该目录不存在。" if not os.path.isdir(b) else "该目录存在，但没有上面这些文件。"
        sys.stderr.write(
            "错误：%s 下缺少可执行文件：%s\n  %s\n"
            "  QLH_BUILD_DIR 现在 = %s\n"
            "  请把它指向 llama.cpp 的构建目录（其下有 bin/），例如：\n"
            "    QLH_BUILD_DIR=H:/llamacpp-c/build-split python verify/<script>.py\n"
            "  构建方式见 verify/README.md。\n"
            % (b, " ".join(missing), where, build_dir())
        )
        raise SystemExit(3)
    sys.stderr.write(
        "[env] QLH_BUILD_DIR=%s  QLH_SRC_DIR=%s  QLH_TMP=%s\n"
        % (build_dir(), src_root(), tmp_dir())
    )


# 常用别名：脚本里直接用 BIN / ROOT / TMP，与原来的变量名对齐，判据代码不用动。
BIN = bin_dir()
ROOT = src_root()
TMP = tmp_dir()
