"""打一个 Windows 绿色目录：拷到没装过 Python 的机器上，改模型路径就能跑。

    python scripts/pack-portable.py \
        --python  C:/Python312 \
        --llama-bin H:/llamacpp-c/build-vulkan/bin \
        --shim    .qlh_tmp/shimout/qlh_keep_head.dll

产出跑完打印目录和体积，然后自己 tar 一下就能传：

    cd .qlh_tmp/dist && tar czf qlh-portable.tgz qlh-portable

三处外部输入（都是本机路径，所以必须显式给）：
  --python    一个装好 numpy 的 CPython 3.12+ 目录（脚本会精简它）
  --llama-bin 编好的 llama.dll / ggml*.dll 所在目录
  --shim      带 QLH_ABI_UPSTREAM_HEADER 编出来的 qlh_keep_head.dll

包里的 Vulkan 后端是**故意不带**的：实测它只要被加载，即使 -ngl 0 也会把 CPU 侧权重
塞进 Vulkan host buffer，iGPU 机器上 head 段因此从 245 ms 掉到 112 ms。
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent

# stdlib 里对本程序没用，删掉能省 32 MB。
SKIP_STDLIB = {"test", "idlelib", "tkinter", "turtledemo", "lib2to3", "ensurepip",
               "__pycache__", "site-packages", "pydoc_data"}

# DLLs 下与本程序无关的扩展模块。
SKIP_DLLS = {"_tkinter.pyd", "tcl86t.dll", "tk86t.dll", "tkinter_common.zip"}

WORKER_FILES = ["stage_worker.py", "engine.py", "keep_head.py", "keep_head_worker.py",
                "config.py", "protocol.py", "wire.py", "chat.py", "server.py", "stream.py",
                # 常驻形态：supervisor 起本机的段并负责重启，它需要这两个。
                "stage_supervisor.py", "chain_client.py", "topology.py"]

LLAMA_DLLS = ["llama.dll", "ggml.dll", "ggml-base.dll", "mtmd.dll"]
GGML_CPU_DLLS = ["ggml-cpu-alderlake.dll", "ggml-cpu-cannonlake.dll", "ggml-cpu-cascadelake.dll",
                 "ggml-cpu-haswell.dll", "ggml-cpu-icelake.dll", "ggml-cpu-sandybridge.dll",
                 "ggml-cpu-skylakex.dll", "ggml-cpu-sse42.dll", "ggml-cpu-x64.dll"]
MSVC_RUNTIME = ["MSVCP140.dll", "MSVCP140_1.dll", "VCRUNTIME140.dll", "VCRUNTIME140_1.dll"]


def fresh(p: Path) -> None:
    """清空输出目录。删不掉就改名让开——Windows 上 dll 常被扫描器或残留进程短暂占用。"""
    if p.exists():
        try:
            shutil.rmtree(p)
        except PermissionError as exc:
            stale = p.with_name(p.name + ".stale")
            shutil.rmtree(stale, ignore_errors=True)
            p.rename(stale)
            print(f"注意：有文件被占用（{exc.filename}），已改名为 {stale.name}")
    p.mkdir(parents=True, exist_ok=True)


def copytree_filtered(src: Path, dst: Path, skip: set[str]) -> int:
    n = 0
    dst.mkdir(parents=True, exist_ok=True)
    for item in src.iterdir():
        if item.name in skip:
            continue
        if item.is_dir():
            n += copytree_filtered(item, dst / item.name, skip)
        else:
            shutil.copy2(item, dst / item.name)
            n += 1
    return n


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--python", required=True, help="装好 numpy 的 CPython 目录")
    ap.add_argument("--llama-bin", required=True, help="llama.dll / ggml*.dll 所在目录")
    ap.add_argument("--shim", required=True, help="qlh_keep_head.dll 路径")
    ap.add_argument("--out", default=str(REPO / ".qlh_tmp" / "dist" / "qlh-portable"))
    ap.add_argument("--msvc-from", default="", help="MSVC 运行时 dll 的来源目录（默认找 llamba-bin 和 shim 同级）")
    args = ap.parse_args()

    py, lbin, shim = Path(args.python), Path(args.llama_bin), Path(args.shim)
    out = Path(args.out)
    for p in (py, lbin, shim):
        if not p.exists():
            print(f"缺少：{p}", file=sys.stderr)
            return 1

    fresh(out)
    for d in ("runtime", "bin", "worker"):
        (out / d).mkdir()

    # ---- runtime：解释器 + 精简 stdlib + numpy ----
    rt = out / "runtime"
    for f in ("python.exe", "python312.dll", "vcruntime140.dll", "vcruntime140_1.dll"):
        src = py / f
        if src.exists():
            shutil.copy2(src, rt / f)
    n_std = copytree_filtered(py / "Lib", rt / "Lib", SKIP_STDLIB)
    n_dlls = copytree_filtered(py / "DLLs", rt / "DLLs", SKIP_DLLS)
    sp = rt / "Lib" / "site-packages"
    sp.mkdir(parents=True, exist_ok=True)
    found_numpy = False
    for name in ("numpy", "numpy.libs"):
        s = py / "Lib" / "site-packages" / name
        if s.exists():
            shutil.copytree(s, sp / name)
            found_numpy = True
    if not found_numpy:
        print("错误：源 python 里没有 numpy，worker 跑不起来", file=sys.stderr)
        return 2
    print(f"runtime: stdlib {n_std} 文件, DLLs {n_dlls} 文件")

    # ---- bin ----
    missing = []
    for f in LLAMA_DLLS + GGML_CPU_DLLS:
        s = lbin / f
        if s.exists():
            shutil.copy2(s, out / "bin" / f)
        else:
            missing.append(f)
    shutil.copy2(shim, out / "bin" / "qlh_keep_head.dll")

    msvc_dirs = [Path(args.msvc_from)] if args.msvc_from else [lbin, shim.parent, Path("C:/Windows/System32")]
    for f in MSVC_RUNTIME:
        for d in msvc_dirs:
            s = d / f
            if s.exists():
                shutil.copy2(s, out / "bin" / f)
                break
        else:
            missing.append(f)
    if missing:
        print(f"警告：这些 dll 没找到，目标机可能报缺依赖：{missing}")
    print(f"bin: {len(list((out / 'bin').iterdir()))} 个文件")

    # ---- worker ----
    for f in WORKER_FILES:
        s = REPO / f
        if s.exists():
            shutil.copy2(s, out / "worker" / f)
        else:
            print(f"警告：worker 文件缺失 {f}")
    print(f"worker: {len(list((out / 'worker').iterdir()))} 个文件")

    # ---- 配置 / 启动器 / 自检客户端 ----
    (out / "config.toml").write_text(CONFIG, encoding="utf-8")
    (out / "launcher.py").write_text(LAUNCHER, encoding="utf-8")
    (out / "run.bat").write_text(RUN_BAT, encoding="ascii")
    (out / "README.txt").write_text(README, encoding="utf-8")
    probe = HERE / "probe.py"
    if probe.exists():
        shutil.copy2(probe, out / "probe.py")
    else:
        print(f"警告：{probe} 不存在，包里没有自检客户端")

    total = sum(f.stat().st_size for f in out.rglob("*") if f.is_file())
    print(f"\n包目录：{out}\n总大小：{total / 1048576:.1f} MB")
    return 0


CONFIG = r"""# qlh-portable 配置。改这三个就够了：model.path、stage.role、stage.layer_range。

[model]
# 必须是完整整模（LLAMA_LAYER_RANGE 只 touch 自己那几层，不建裁层工件）
path = "C:/models/qwen36-35b-a3b-q6k.gguf"

[stage]
# gpu = 第一段（收 token，出 hidden）；tail = 最后一段（收 hidden，出 token）
role = "gpu"
# hybrid 架构要求切点是 full_attention_interval=4 的倍数
layer_range = "0:24"
threads = 6
ctx = 2048

[net]
host = "127.0.0.1"
port = 9103
"""

LAUNCHER = r'''"""读 config.toml，校验模型，起 stage_worker。"""
import os
import sys
import tomllib
from pathlib import Path

HERE = Path(__file__).resolve().parent


def main() -> int:
    cfg = tomllib.loads((HERE / "config.toml").read_text(encoding="utf-8"))

    st = cfg["stage"]
    role = st["role"]

    # role = "supervisor" 时不看 layer_range：它读 topology.toml，把本机的每一段都起起来，
    # 崩了负责重启。这是常驻形态；单跑一段用 gpu / tail。模型路径也由 topology 各自声明，
    # 所以这里的 model.path 不参与校验。
    if role == "supervisor":
        env = dict(os.environ)
        env["PATH"] = str(HERE / "bin") + os.pathsep + env.get("PATH", "")
        env["PYTHONPATH"] = str(HERE / "worker")
        # supervisor 自己不认 --shim，它靠 config.find_shim() 从 cwd 往上找 checkout 的
        # shim/bin/；包里的布局是 bin/，所以必须显式告诉它。
        env["SA_SHIM"] = str(HERE / "bin" / "qlh_keep_head.dll")
        topo = HERE / "topology.toml"
        if not topo.is_file():
            print(f"role=supervisor 需要 {topo}：它描述整条链，本机是哪几段由它决定。")
            return 2
        env["SA_TOPOLOGY"] = str(topo)
        print(f"拓扑：{topo}")
        import subprocess
        return subprocess.call([sys.executable, str(HERE / "worker" / "stage_supervisor.py")],
                               env=env, cwd=str(HERE / "worker"))

    model = Path(cfg["model"]["path"])
    if not model.is_file():
        print(f"模型不存在：{model}")
        print("改 config.toml 里的 model.path，指向完整整模的 .gguf。")
        return 2

    rng = st["layer_range"]
    lo, hi = (int(x) for x in rng.split(":"))

    print(f"模型：{model}（{model.stat().st_size / 1073741824:.2f} GiB）")
    print(f"角色：{role}  层范围：{rng}（{hi - lo} 层）")

    env = dict(os.environ)
    env["LLAMA_LAYER_RANGE"] = rng
    env["SA_BACKEND_DIR"] = str(HERE / "bin")
    # 需要 logits 的只有 tail；head 只出 hidden
    env["SA_NEEDS_LOGITS"] = "0" if role == "gpu" else "1"
    env["PATH"] = str(HERE / "bin") + os.pathsep + env.get("PATH", "")

    argv = [
        str(HERE / "worker" / "stage_worker.py"),
        "--role", role,
        "--model", str(model),
        "--shim", str(HERE / "bin" / "qlh_keep_head.dll"),
        "--host", cfg["net"]["host"],
        "--port", str(cfg["net"]["port"]),
        "--threads", str(st["threads"]),
        "--layers", str(hi - lo),
        "--ctx", str(st["ctx"]),
    ]
    print("启动：", " ".join(argv))
    # 不用 os.execve：它替换进程后，调用方的 ssh/shell 会认为本进程已结束并关闭连接，
    # 新进程随即被杀（Windows 上实测 exit 0 且没有 worker 存活）。
    import subprocess
    return subprocess.call([sys.executable] + argv, env=env, cwd=str(HERE / "worker"))


if __name__ == "__main__":
    raise SystemExit(main())
'''

RUN_BAT = r"""@echo off
rem qlh-portable entry point. All logic lives in launcher.py.
setlocal
cd /d "%~dp0"
set PATH=%~dp0bin;%PATH%
"%~dp0runtime\python.exe" "%~dp0launcher.py" %*
if errorlevel 1 pause
endlocal
"""

README = """qlh-portable
============

把 35B-A3B 模型切成两段、分两台机器跑。这是其中一台的运行时。

怎么用
------
1. 准备一份完整整模（.gguf）。
2. 打开 config.toml 改三项：
   - model.path        模型路径（正斜杠）
   - stage.role        gpu = 第一段；tail = 最后一段
   - stage.layer_range 自己负责哪几层，如 "0:24" / "24:40"
     （切点必须是 4 的倍数，qwen35moe 的 hybrid 结构要求）
3. 双击 run.bat。

两台机器的层范围要首尾相接：
    机器 A：0:24   机器 B：24:40

自检
----
两端都起好之后，在任意一台能同时访问两者的机器上：
    runtime\\python.exe probe.py --head 127.0.0.1:9103 --tail <另一台>:9106

不做的事
--------
- 不打包模型（23.5 GiB），config.toml 指过去。
- 不带 Vulkan 后端（iGPU 上反而拖慢 CPU 侧权重）。
- 不含服务发现，两台机器的地址自己填、网络自己保证（Tailscale 之类）。
"""


if __name__ == "__main__":
    raise SystemExit(main())
