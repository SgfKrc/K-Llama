# 打成 Windows 绿色目录

> 状态：**已在一台清空的机器上验证**（2026-11-03）
> 适用：把当前这套 worker 交给别人跑，或换一台机器复现。

目标很窄：**拷走一个目录，改一行模型路径就能跑**。不做安装器、不做服务发现。

---

## 1. 打包

```bash
python scripts/pack-portable.py \
    --python  C:/Users/<you>/AppData/Local/Programs/Python/Python312 \
    --llama-bin H:/llamacpp-c/build-vulkan/bin \
    --shim    .qlh_tmp/shimout/qlh_keep_head.dll

cd .qlh_tmp/dist && tar czf qlh-portable.tgz qlh-portable
```

三个外部输入都是本机路径，所以必须显式给：

| 参数 | 要求 |
| --- | --- |
| `--python` | 一个装好 numpy 的 CPython 3.12+ 目录，脚本会精简掉 test/tkinter/site-packages 等 |
| `--llama-bin` | 编好的 `llama.dll` 和 `ggml*.dll` |
| `--shim` | 带 `QLH_ABI_UPSTREAM_HEADER` 编出来的 `qlh_keep_head.dll`（见 `shim/build_shim_msvc.ps1`） |
| `--msvc-from` | MSVC 运行时 dll 的来源目录，缺省依次找 llama-bin、shim 同级、`C:/Windows/System32` |

产物体积 **96 MB**（tar 后 32 MB）。不带 Vulkan 后端——见 §4。

## 2. 包结构

```
qlh-portable/
    runtime/     精简 Python 3.12（stdlib 529 文件 + numpy 2.5.1）
    bin/         llama.dll、ggml*.dll×10、qlh_keep_head.dll、MSVC 运行时×4
    worker/      stage_worker.py 那套（10 个文件）
    config.toml  模型路径、角色、层范围、线程数、端口
    launcher.py  读配置、校验模型、设环境变量、起 worker
    probe.py     链路自检客户端
    run.bat      双击入口
    README.txt   给最终用户看的
```

## 3. 用

两台机器各解压一份，各改 `config.toml`：

```toml
[model]
path = "D:/models/qwen36-35b-a3b-q6k.gguf"

[stage]
role = "gpu"          # 第一段收 token 出 hidden
layer_range = "0:24"  # 切点必须是 4 的倍数（hybrid 结构）

[net]
host = "127.0.0.1"
port = 9103
```

另一台 `role = "tail"`、`layer_range = "24:40"`、`host = "0.0.0.0"`。然后双方各双击 `run.bat`。

自检（任意一台能同时访问两者的机器）：

```bash
runtime\python.exe probe.py --head 127.0.0.1:9103 --tail 100.x.x.x:9106
```

## 4. 几个刻意的决定

- **不带 Vulkan 后端**。实测它只要被加载，即使 `-ngl 0` 也会把 CPU 侧权重塞进 Vulkan host
  buffer；Surface 上 head 段因此从 245 ms 掉到 112 ms。
- **`launcher.py` 不用 `os.execve`**。它替换进程后，调用方的 ssh/shell 会认为本进程已结束、
  关闭连接，新进程随即被杀（Windows 上实测 exit 0 且没有 worker 存活）。用 `subprocess`
  保持前台阻塞。
- **模型不进包**。23.5 GiB，只能由用户自己放，`config.toml` 指过去。
- **两份整模**。`LLAMA_LAYER_RANGE` 要求每个节点持有完整文件，只 touch 自己那几层。

## 5. 已验证的边界

在一台清空过的 Surface 上跑通：删掉原来 223 MB 的工作区之后，包仍然独立工作，
端到端 **98.7 ms / 10.13 t/s**（head 46.7 ms / 24 层，tail 52.0 ms，含 Tailscale 往返）。

**没验证的**：换一台全新的 Windows（没装过任何东西）。这次验证用的机器虽然清空了工作区，
但 VC++ 运行库、显卡驱动这些系统级依赖还在。`bin/` 里带的那 4 个 MSVC dll 就是为此准备的，
但没有在真正干净的机器上确认过。

## 6. Termux 侧（Y700）

安卓上没有 APK——不走 NDK 那条路，直接用 Termux + 一个脚本：

```bash
bash scripts/termux-install.sh \
    --fork ~/qlh-fork \
    --model ~/qwen36-35b-a3b-q6k.gguf \
    --range 24:40 --port 9106
```

脚本做五件事：查依赖 → 编 libllama（**缺才编**，`-j 1`）→ 编 shim → 校验层范围 →
`setsid` + `termux-wake-lock` 后台起服务。幂等，重复跑只补缺的部分。

**为什么不是 APK**：NDK 要重编（Termux 的 `libc++_shared.so` 和 bionic 不是一回事），
python 栈要么塞解释器要么重写，23.5 GiB 模型还得让用户 SAF 选一次再复制。而一键脚本
只要装个 Termux，成本差一个数量级，且是我们自己调研用的最佳形态。

**三个写进脚本校验的约束**：

| 约束 | 原因 |
| --- | --- |
| 切点与段内层数都是 4 的倍数 | qwen35moe 是 hybrid，每 4 层一个 full attention，错位会层类型不匹配 |
| 编译 `-j 1` | 并发编译会被 Android 的 lmkd 直接 SIGKILL（实测 `-j 6` 在第 26% 被杀） |
| `setsid` + `termux-wake-lock` | 否则 adb/ssh 一断就被带走（实测 `exit 255`），或 CPU 休眠时被冻结 |

实测：启动后 PID 17412，`adb kill-server` + `start-server` 之后进程仍存活。

## 7. 还没做的骨架

这些不是打包问题，是产品化缺口，按影响排序：

1. **地址配置**：现在 Tailscale IP 要手填，且两端都不知道对方在不在。
2. **断线重连**：worker 崩了只能手动重启（这轮调试里重启了十几次）。
3. **启动编排**：两个 `run.bat` 谁先谁后、对方没起来时怎么重试，现在靠人。
4. **模型校验**：只检查文件存在，没校验 sha256 / 是不是同一份整模。
5. **多轮会话**：`probe.py` 每轮都从 `n_past=0` 开始，真实对话要维护 KV 和位置。
