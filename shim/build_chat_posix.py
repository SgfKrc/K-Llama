"""把 qlh_chat 编成 POSIX 共享库（Linux x86-64 / aarch64、Android Termux）。

与 ``build_shim_posix.py`` 是两件不同的事，不要混：

  * ``build_shim_posix.py`` 做的是**文本移植**——把 keep-head shim 里的
    ``<windows.h>`` / ``GetProcAddress`` / ``_putenv_s`` 换成 dlopen / dladdr，然后
    生成 ``qlh_keep_head_posix.c`` 再编。
  * 本脚本编的是 ``shim/chat/`` 那批 vendored 的纯 C++（llama.cpp 的 chat 模板 +
    PEG 解析器）与 ``shim/qlh_chat.cpp`` 这个 C ABI 外壳。**没有任何 Win32 面**，
    所以不需要经过文本转换，直接 clang++ 编。

同一个 include 纪律：**不要**加 ``-I shim/chat/jinja`` 或
``-I <llama>/common/jinja``，``<string.h>`` 会解析到 ``jinja/string.h`` 并炸掉
（MSVC 报 memchr/strlen 不是 global namespace 成员，clang 报类似的一屏错误）。
``jinja/*.cpp`` 内部的 ``#include "lexer.h"`` 靠"先查包含它的目录"找到。

产物落在 ``--out-dir``（默认 ``.qlh_tmp/chatout``），不进 ``shim/bin``——
那个目录是 shim 推理用的，混在一起会误导。

用法：

    python shim/build_chat_posix.py --llama-src /path/to/llamacpp-c
    python shim/build_chat_posix.py --llama-src /path/to/llamacpp-c --dry-run
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
CHAT_DIR = HERE / "chat"
ABI_SRC = HERE / "qlh_chat.cpp"


def collect_sources() -> list[Path]:
    srcs = sorted(CHAT_DIR.rglob("*.cpp"))
    srcs.append(ABI_SRC)
    return srcs


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--llama-src", required=True,
                    help="llama.cpp 源码根（含 common/、include/、ggml/include/）")
    ap.add_argument("--out-dir", default=str(Path.cwd() / ".qlh_tmp" / "chatout"),
                    help="产物目录（默认 ./.qlh_tmp/chatout）")
    ap.add_argument("--cxx", default="clang++")
    ap.add_argument("--so", default="libqlh_chat.so")
    ap.add_argument("--dry-run", action="store_true", help="只打印命令")
    args = ap.parse_args()

    llama = Path(args.llama_src).resolve()
    for p in (CHAT_DIR, ABI_SRC, llama / "common" / "chat.h", llama / "include" / "llama.h"):
        if not p.exists():
            raise SystemExit(f"缺少：{p}")

    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    srcs = collect_sources()
    cmd = [
        args.cxx, "-std=c++17", "-O2", "-fPIC", "-shared",
        "-fvisibility=hidden",          # 只为让 QLH_CHAT_API 的 visibility attribute 有意义
        "-Wl,-soname," + args.so,
        "-o", str(out_dir / args.so),
        *[str(s) for s in srcs],
        f"-I{CHAT_DIR}",
        f"-I{CHAT_DIR / 'vendor'}",
        f"-I{llama / 'common'}",
        f"-I{llama / 'include'}",
        f"-I{llama / 'ggml' / 'include'}",
    ]

    print(f"{len(srcs)} 个 .cpp -> {out_dir / args.so}")
    if args.dry_run:
        print(" ".join(cmd))
        return 0

    if shutil.which(args.cxx) is None:
        raise SystemExit(f"找不到编译器：{args.cxx}")

    rc = subprocess.call(cmd)
    if rc != 0:
        print(f"编译失败：rc={rc}", file=sys.stderr)
        return rc

    so = out_dir / args.so
    print(f"产物：{so}（{so.stat().st_size:,} 字节）")

    # 依赖自检：只在 Linux 上有 ldd；Android/Termux 上 ldd 常常不存在，跳过即可。
    if shutil.which("ldd") is not None:
        deps = subprocess.run(["ldd", str(so)], capture_output=True, text=True)
        lines = [l for l in deps.stdout.splitlines() if "=>" in l]
        print("\nldd:")
        for l in lines:
            print("  " + l.strip())
        bad = [l for l in lines if "llama" in l or "ggml" in l]
        if bad:
            print("WARNING：产物依赖了 llama/ggml", file=sys.stderr)
            return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
