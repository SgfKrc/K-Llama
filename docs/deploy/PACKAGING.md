# 打成 Windows 绿色目录

> 状态：**已验证**（文中引用的 v2 工具链——`chain_client` / `stage_worker` / `shim/` / `koakuma`——已退场，归档在 `slimarc/prototype`，命令与路径只作历史口径）
> 更新日期：2026-11-03
> 适用范围：把当前这套 worker 打成 Windows 绿色目录交给别人跑，或换一台机器复现；
> 不含安装器、服务发现与模型分发。

目标很窄：**拷走一个目录，改一行模型路径就能跑**。不做安装器、不做服务发现。

## 1. 打包

```bash
# 脚本已随 v2 退场，位置 slimarc/prototype:scripts/pack-portable.py，先取出到当前目录：
#   git show slimarc/prototype:scripts/pack-portable.py > pack-portable.py
python pack-portable.py \
    --python  C:/Users/<you>/AppData/Local/Programs/Python/Python312 \
    --llama-bin H:/llamacpp-c/build-vulkan/bin \
    --shim    .qlh_tmp/shimout/qlh_keep_head.dll

cd .qlh_tmp/dist && tar czf qlh-portable.tgz qlh-portable
```

**注意 `--out`**：默认输出到 `.qlh_tmp/dist/qlh-portable`，且会**先删掉同名目录**。要保留
上一份包就显式给 `--out <别的路径>`，否则照上面原样跑会覆盖已有的产物。

三个外部输入都是本机路径，所以必须显式给：

| 参数 | 要求 |
| --- | --- |
| `--python` | 一个装好 numpy 的 CPython 3.12+ 目录，脚本会精简掉 test/tkinter/site-packages 等 |
| `--llama-bin` | 编好的 `llama.dll` 和 `ggml*.dll` |
| `--shim` | 带 `QLH_ABI_UPSTREAM_HEADER` 编出来的 `qlh_keep_head.dll`（见 `slimarc/prototype:shim/build_shim_msvc.ps1`） |
| `--msvc-from` | MSVC 运行时 dll 的来源目录，缺省依次找 llama-bin、shim 同级、`C:/Windows/System32` |

产物体积 **96 MB**（tar 后 32 MB）。不带 Vulkan 后端——见 §4。

## 2. 包结构

```
qlh-portable/
    runtime/     精简 Python 3.12（stdlib 529 文件 + numpy 2.5.1）
    bin/         llama.dll、ggml*.dll×10、qlh_keep_head.dll、MSVC 运行时×4
    worker/      stage_worker.py 那套（13 个文件）
    config.toml  模型路径、角色、层范围、线程数、端口
    launcher.py  读配置、校验模型、设环境变量、起 worker
    probe.py     链路自检客户端
    run.bat      双击入口
    README.txt   给最终用户看的
```

**`chain_client.py` 在仓库里有两份**：根目录一份、`worker/` 一份（打包时只带 `worker/`
那份）。开发机的 supervisor 工作目录是 `worker/`，`import chain_client` 命中的是**后者**。
所以改部署逻辑（比如给 worker 传环境变量）只更新根那份会静默失效——2026-11-04 的心跳机制
就这样在 Surface 上白接了一次。两份都要同步。

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

**两种形态别混用**：

- **单段**（上面这套）：`config.toml` 写 `[stage] role`，一台机器一段
- **多段常驻**（推荐）：另放一份 `topology.toml` 在包根，把 `config.toml` 的
  `[stage] role` 改成 `supervisor`。此时 `config.toml` 的 `layer_range` 与 `ctx` 都不生效
  ——层段由 topology 决定，`ctx` 只能由 `SA_CTX` 给

**包里能不能直接出 token**：能（2026-11-04 第三次从零演练实测，包内解释器 5.1 / 5.2 tok/s）。
`runtime/` 只带 numpy，`chain_client.py` 需要 `psutil` 时靠 `_available_bytes()` 自己读系统
信息（Windows `GlobalMemoryStatusEx`、POSIX `sysconf`），量到的就是真实可用内存，不是常量
——所以「本机装得下就让你跑」这条判断在包里同样成立。要在包内跑推理：

```bash
runtime\python.exe worker\chain_client.py --topology topology.toml \
    --shim bin\qlh_keep_head.dll --greedy --prompt-text "你好" --gen 32
```

`run.bat` 失败会 `pause` 等按键，无人值守（ssh / 计划任务）下会挂住，改用
`runtime\python.exe launcher.py` 直接起。

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
- **模型不进包**。21.87 GiB（23.48 GB），只能由用户自己放，`config.toml` 指过去。
- **两份整模**。`LLAMA_LAYER_RANGE` 要求每个节点持有完整文件，只 touch 自己那几层。

## 5. 已验证的边界

在一台清空过的 Surface 上跑通：删掉原来 223 MB 的工作区之后，包仍然独立工作，
端到端 **98.7 ms / 10.13 t/s**（head 46.7 ms / 24 层，tail 52.0 ms，含 Tailscale 往返）。

**没验证的**：换一台全新的 Windows（没装过任何东西）。这次验证用的机器虽然清空了工作区，
但 VC++ 运行库、显卡驱动这些系统级依赖还在。`bin/` 里带的那 4 个 MSVC dll 就是为此准备的，
但没有在真正干净的机器上确认过。

## 6. Termux 侧（Y700）

安卓上没有 APK——不走 NDK 那条路，直接用 Termux + 一个脚本：

> **`termux-install.sh` 已退场（面向 v2）**：它装的是 `stage_worker.py` / `stage_supervisor.py` /
> `shim/` / 参数生成的 `topology.toml`，这些已随 v2 归档在 `slimarc/prototype`。v2.1 的安卓部署见
> [`CROSS-MACHINE.md`](CROSS-MACHINE.md) §6 与 [`DEPLOYMENT.md`](DEPLOYMENT.md)——段进程是
> `split-stage`，拓扑由 `control/qlhctl.py` 读 TOML 驱动。下面的命令只作历史口径。

```bash
# 单段（调试，v2 口径）
bash scripts/termux-install.sh --fork ~/qlh-fork \
    --model ~/qwen36-35b-a3b-q6k.gguf --range 24:40 --port 9106

# 常驻（推荐）：交给 supervisor 管本机段，崩了自动重启
bash scripts/termux-install.sh --fork ~/qlh-fork \
    --model ~/qwen36-35b-a3b-q6k.gguf \
    --role supervisor --topology ~/qlh-worker/topology.toml
```

supervisor 形态不需要 `--range` / `--port`：本机是哪几段、切点在哪，全部来自拓扑。

脚本做五件事：查依赖 → 编 libllama（**缺才编**，`-j 1`）→ 编 shim → 校验层范围 →
`setsid` + `termux-wake-lock` 后台起服务。幂等，重复跑只补缺的部分。

**为什么不是 APK**：NDK 要重编（Termux 的 `libc++_shared.so` 和 bionic 不是一回事），
python 栈要么塞解释器要么重写，21.87 GiB 模型还得让用户 SAF 选一次再复制。而一键脚本
只要装个 Termux，成本差一个数量级，且是我们自己调研用的最佳形态。

**三个写进脚本校验的约束**：

| 约束 | 原因 |
| --- | --- |
| 切点与段内层数都是 4 的倍数 | qwen35moe 是 hybrid，每 4 层一个 full attention，错位会层类型不匹配 |
| 编译 `-j 1` | 并发编译会被 Android 的 lmkd 直接 SIGKILL（实测 `-j 6` 在第 26% 被杀） |
| `setsid` + `termux-wake-lock` | 否则 adb/ssh 一断就被带走（实测 `exit 255`），或 CPU 休眠时被冻结 |

实测：启动后 PID 17412，`adb kill-server` + `start-server` 之后进程仍存活。

## 7. 实现现状与缺口

| 缺口 | 状态 |
| --- | --- |
| 启动编排 | ✅ `chain_client.bring_up()` + `stage_supervisor.py` 常驻形态 |
| 断线重连 | ✅ `stage_supervisor.py`：死的只重启那一个，退避 2→4→8→16→30 封顶，稳定存活 60 s 才归零 |
| 模型校验 | ✅ `topology.verify_model_hashes()`：拓扑里声明 `sha256` 才校验，没声明就跳过 |
| 地址配置 | ⚠️ `topology.toml` 的 host/port，**没有服务发现**，IP 仍要手填 |
| 多轮会话 | ❌ `probe.py` 每轮从 `n_past=0` 开始，真实对话要维护 KV 与位置 |

**常驻形态**：`config.toml` 里 `role = "supervisor"`，同目录放一份 `topology.toml`（两台机器内容相同）。
supervisor 自己判断哪几段在 `127.0.0.1` 上、只起那些；远端段由那台机器自己的 supervisor 负责。
只把本机段交给 `bring_up` 是刻意的——传整份拓扑会让 A 机按 `start_cmd` 去 ssh 拉起 B 机的段，
而 B 机已占着端口，新 worker bind 失败后会被当成反复崩溃。

实测（Surface head 24 层 + Y700 tail 16 层）：kill 掉 head 的 `stage_worker`（PID 11028）后，
6 秒内 9103 由新 PID 4816 重新监听，90 秒后 `probe.py` 仍能出 token。
Y700 侧同样验过：`kill -9` 掉 tail 的 worker（PID 24931）→ `restart in 2s` → 新 PID 25800，
supervisor 自己的 PID 不变。

**拓扑里的 host 有两个含义，容易踩**：它既是客户端的连接地址，也是 worker 的 bind 地址。
本机段如果写 `127.0.0.1`，worker 就只绑回环，远端连不进来（日志是 `listening on 127.0.0.1:9106`，
对端是 connection refused）。写 `0.0.0.0` 才既绑全接口、又被 `topology.is_loopback` 认成本机段。
远端段照旧写真实地址（`100.x` / 主机名）。

**端口被占不会无限重启**：重启前用 `chain_client._port_open` 探端口（本机段的 bind 地址
`0.0.0.0` 换成 `127.0.0.1` 探），被占则等 5 秒再看，12 次（约 60 秒）仍被占就放弃该段、
其他段照常跑；端口空出来会自己恢复。Windows 上占住 `0.0.0.0` 时 worker 绑 `127.0.0.1`
仍会成功（`SO_REUSEADDR`），两个 listener 并存、连接落到谁身上不确定——所以这道检查
同时省掉了白加载和这种更难查的抢端口。

**段内部失效会自愈**：`keep_head` 子进程才真正持有权重，它没了之后 `stage_worker`
会自己以非零码退出（先关监听 socket，好让接手的进程立刻绑回同一端口），让 supervisor
的 `poll()` 看见。这样「进程存活」重新等价于「段可用」，不需要第二套健康探测——
就绪判据仍然只有 HELLO。

## 8. Windows 侧常驻：三个坑

Surface Pro 7 是纯 Windows，没有 termux 的 `setsid`，让 supervisor 真正常驻踩了三个：

| 坑 | 现象 | 解法 |
| --- | --- | --- |
| ssh 一断子进程被收走 | `start /b` 起的 python 在会话结束时悄悄消失，日志停在中间 | `schtasks /ru SYSTEM /rl HIGHEST` 建任务，进程挂到 service 会话下 |
| `.cmd` 里的中文注释 | cmd.exe 按 GBK 解析 UTF-8，注释被当成命令：`'...' 不是内部或外部命令` | 启动脚本只写 ASCII 注释 |
| 防火墙拦入站 | 本机连 9104 是 connection refused，对端 `netstat` 却显示 LISTENING | `netsh advfirewall firewall add rule name=qlh9104 dir=in action=allow protocol=TCP localport=9104` |

`SA_CTX` 必须在启动脚本里显式导出（`set SA_CTX=1024`），否则 `config.DEFAULT_CTX=4096`
会让这一段的 seq 边界和另外两段错开——不报错，只是算得不对。
