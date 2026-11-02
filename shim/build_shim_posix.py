"""把 shim 源码移植到 POSIX（aarch64 / x86-64 Linux、Android Termux）。

shim 的平台相关代码只有三处（2026-11-03 在 Y700 上实测）：

  1. ``#include <windows.h>``
  2. ``qlh_resolve_layer_inp``：``GetModuleHandleA`` + ``GetProcAddress``（MSVC mangled 名）
  3. ``qlh_ensure_backends_loaded``：``GetModuleHandleExA`` + ``GetModuleFileNameA`` 求自身目录

外加两处 MSVC 专有运行时（``_putenv_s``）和一处源码里把 NUL 写成 ``'\\0'``（两个字符，
clang 当多字符常量拒收）。

``llama_set/get_embeddings_layer_inp`` 在 ``src/llama-ext.h`` 里没有 ``extern "C"``，所以
Linux 上同样是 Itanium mangled 名——``dlsym(RTLD_DEFAULT, ...)`` 是 ``GetProcAddress``
的等价物，不是"删掉兼容层"。

用法：

    python shim/build_shim_posix.py                       # 生成 shim/qlh_keep_head_posix.c
    python shim/build_shim_posix.py --compile             # 顺手调 clang 编成 .so

编译时必须带 ``-DQLH_ABI_UPSTREAM_HEADER``：shim 里设置 ``cparams.embeddings`` /
``embeddings_pre_norm`` 的那一整段包在这个条件里，不定义就整段被预处理掉，
head 段会因为 ``embd.data == NULL`` 返回 ``rc=-3``。

链接时 ``-Wl,-rpath,'$ORIGIN'`` 不能省，否则运行时找不到同目录的 ``libllama.so.0``。
"""

from __future__ import annotations

import argparse
import re
import shutil
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent

WIN_HEADERS = "#include <windows.h>"

POSIX_HEADERS = r"""/* ---- POSIX 移植：Win32 面只有模块查找、自身路径和 _putenv_s ---- */
#include <dlfcn.h>
#include <limits.h>
#ifndef MAX_PATH
#define MAX_PATH PATH_MAX
#endif
typedef void * HMODULE;
typedef unsigned long DWORD;
typedef const char * LPCSTR;"""

# 原文件里两次 GetProcAddress 各占两行。
RESOLVE_OLD = re.compile(
    r'    HMODULE m = GetModuleHandleA\("llama\.dll"\);\n'
    r'    if \(m == NULL\) \{\n'
    r'        return;\n'
    r'    \}\n'
    r'    qlh_p_set_layer_inp = \(qlh_set_layer_inp_fn\) \(void \*\) GetProcAddress\(\n'
    r'        m, "\?llama_set_embeddings_layer_inp@@YAXPEAUllama_context@@I_N@Z"\);\n'
    r'    qlh_p_get_layer_inp = \(qlh_get_layer_inp_fn\) \(void \*\) GetProcAddress\(\n'
    r'        m, "\?llama_get_embeddings_layer_inp@@YAPEAMPEAUllama_context@@I@Z"\);\n'
)

RESOLVE_NEW = r"""    /* POSIX：这两个入口在 libllama.so 里是 Itanium mangled 名，且位于全局符号表，
     * 所以 dlsym(RTLD_DEFAULT, ...) 就是 GetProcAddress 的直接等价物。
     * 名字可用 `nm -D libllama.so | grep layer_inp` 核对。 */
    qlh_p_set_layer_inp = (qlh_set_layer_inp_fn) (void *) dlsym(
        RTLD_DEFAULT, "_Z30llama_set_embeddings_layer_inpP13llama_contextjb");
    qlh_p_get_layer_inp = (qlh_get_layer_inp_fn) (void *) dlsym(
        RTLD_DEFAULT, "_Z30llama_get_embeddings_layer_inpP13llama_contextj");
"""

SELF_OLD = re.compile(
    r'    char self\[MAX_PATH\];\n'
    r'    HMODULE self_module = NULL;\n'
    r'.*?'
    r'    ggml_backend_load_all\(\);',
    re.S,
)

SELF_NEW = r"""    /* POSIX：对自己的函数地址做 dladdr 就能拿到本 .so 的路径。 */
    {
        Dl_info info;
        if (dladdr((void *) &qlh_ensure_backends_loaded, &info) && info.dli_fname != NULL) {
            char self[MAX_PATH];
            snprintf(self, sizeof(self), "%s", info.dli_fname);
            char * slash = strrchr(self, '/');
            if (slash != NULL) {
                *slash = 0;
                fprintf(stderr, "[qlh_keep_head] loading backends from self dir=%s\n", self);
                ggml_backend_load_all_from_path(self);
            }
        }
    }
    ggml_backend_load_all();"""


def port(src_text: str) -> str:
    if WIN_HEADERS not in src_text:
        raise SystemExit("失败：源码里找不到 #include <windows.h>")
    src = src_text.replace(WIN_HEADERS, POSIX_HEADERS, 1)

    # 原源码把 NUL 写成 '\\0'（反斜杠反斜杠零），clang 会按多字符常量拒收。
    src = src.replace("'\\\\0'", "0")

    src = src.replace('_putenv_s("LLAMA_LAYER_RANGE", "")', 'unsetenv("LLAMA_LAYER_RANGE")')
    src = src.replace('_putenv_s("LLAMA_LAYER_RANGE", saved_range)',
                      'setenv("LLAMA_LAYER_RANGE", saved_range, 1)')

    # 用 lambda 作替换：re.sub 的普通替换串会解释反斜杠转义，把新增 fprintf 里的
    # 字面 "\n" 变成真换行，产生语法错误的字符串字面量。
    if RESOLVE_OLD.search(src) is None:
        raise SystemExit("失败：qlh_resolve_layer_inp 的模式没匹配上")
    src = RESOLVE_OLD.sub(lambda m: RESOLVE_NEW, src, count=1)

    if SELF_OLD.search(src) is None:
        raise SystemExit("失败：qlh_ensure_backends_loaded 的自身路径段没匹配上")
    src = SELF_OLD.sub(lambda m: SELF_NEW, src, count=1)

    def is_comment(line: str) -> bool:
        s = line.strip()
        return s.startswith("*") or s.startswith("/*") or s.startswith("//")

    win_calls = ("GetProcAddress(", "GetModuleHandleA(", "GetModuleHandleExA(",
                 "GetModuleFileNameA(", "_putenv_s(")
    left = [l for l in src.splitlines() if not is_comment(l) and any(c in l for c in win_calls)]
    if left:
        raise SystemExit("失败：仍有 Win32 调用残留：\n" + "\n".join(left))
    return src


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", default=str(HERE / "qlh_keep_head_gpu.c"))
    ap.add_argument("--out", default=str(HERE / "qlh_keep_head_posix.c"))
    ap.add_argument("--compile", action="store_true", help="生成后调 clang 编成 .so")
    ap.add_argument("--llama-src", default="", help="llama.cpp 源码根（含 include/ 与 ggml/include/）")
    ap.add_argument("--llama-lib-dir", default="", help="libllama.so 所在目录")
    ap.add_argument("--so", default="libqlh_keep_head.so")
    args = ap.parse_args()

    src_path, out_path = Path(args.src), Path(args.out)
    out_path.write_text(port(src_path.read_text(encoding="utf-8")), encoding="utf-8")
    print(f"已生成 {out_path}（{len(out_path.read_text(encoding='utf-8').splitlines())} 行）")

    if not args.compile:
        print("\n编译（注意 -DQLH_ABI_UPSTREAM_HEADER 与 $ORIGIN 都不能省）：")
        print(f"  clang -shared -O2 -fPIC -DQLH_ABI_UPSTREAM_HEADER -Wl,-rpath,'$ORIGIN' \\")
        print(f"      -o {args.so} {out_path.name} \\")
        print(f"      -I<llama>/include -I<llama>/ggml/include -L<llama>/build/bin \\")
        print(f"      -lllama -lggml -lggml-base -lm")
        return 0

    if not args.llama_src or not args.llama_lib_dir:
        raise SystemExit("--compile 需要同时给 --llama-src 与 --llama-lib-dir")
    if shutil.which("clang") is None:
        raise SystemExit("找不到 clang")

    cmd = [
        "clang", "-shared", "-O2", "-fPIC",
        "-DQLH_ABI_UPSTREAM_HEADER",
        "-Wl,-rpath,$ORIGIN",
        "-o", args.so, str(out_path),
        f"-I{args.llama_src}/include", f"-I{args.llama_src}/ggml/include",
        f"-L{args.llama_lib_dir}",
        "-lllama", "-lggml", "-lggml-base", "-lm",
    ]
    print("\n" + " ".join(cmd))
    return subprocess.call(cmd)


if __name__ == "__main__":
    raise SystemExit(main())
