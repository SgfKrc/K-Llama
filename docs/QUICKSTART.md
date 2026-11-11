# 快速开始：从拿到仓库到第一次跑通一条分段链

> 状态：**现行**
> 更新日期：2026-11-11
> 适用范围：第一次接触本仓库的人，照着做到「一条分段链出了 token」。只讲操作与每步该看到什么；为什么这么设计见 [`ARCHITECTURE.md`](design/ARCHITECTURE.md)，边界代价见 [`SEGMENTED-LIMITS.md`](limits/SEGMENTED-LIMITS.md)。

## 1. 这是什么、能干什么
K-Llama 让 llama.cpp 按层分段运行：几台机器各持同一份 GGUF，每台只加载自己那几层，段间只传一个隐藏态；对客户端它就是一个普通的 `llama-server`。

| 能干 | 边界（一句话，详版见括号） |
| --- | --- |
| 一台装不下的模型拆到多台跑（32 GiB 内存跑 35B 就是这个形态） | 远程链单序列，`-np > 1` 被拒（`SEGMENTED-LIMITS.md` §2.4） |
| 纯 CPU 可跑；`-ngl` 按段决定哪些层上卡 | 链上 MTP 投机比整模低 13%（`MODEL-SWAP.md` §11.4） |
| 远程链上 DFlash / DSpark 与整模逐 prompt 相同 | 切点必须是 4 的整数倍，没有代码校验（`DEPLOYMENT.md` §2） |
| `control/qlhctl.py` 一份拓扑起停整条链 | 链上没有认证，`bind` 缺省只绑回环（`DEPLOYMENT.md` §5） |

## 2. 你需要什么
| 模型 | 内存 | 磁盘 |
| --- | --- | --- |
| 2B（24 层，`qwen35-2b-Q4_K_M.gguf`） | 两段实测占 1.44 GB；开工前 `available > 12 GB`（`SAMPLING.md` §8） | 1,312,164,672 B |
| 9B（33 block / 32 层，`qwen3-5-9b-local-Q4_K_M.gguf`） | 无独立内存实测；按同一纪律 `available > 12 GB`（`MEMORY-LIMITS.md` §7） | 5.37 GiB（`ANDROID-SURFACE-段接入评估.md` §2.2） |
| 35B（40 层，`Qwen3.6-35B-A3B-UD-Q4_K_M.gguf`） | 单机两段要 `available ≥ 20 GB`（`verify/verify.sh` 的 `QLH_VERIFY_MIN_FREE_GB`）；权重驻留 head `0:32` 15.690 + tail `32:40` 4.90 GiB（`CROSS-MACHINE.md` §6） | 22,663,387,424 B |

模型是 mmap 的，**不需要整个装进物理内存**——真正占物理内存的是 KV cache 与临时缓冲，20 GB 是有实测支撑的下限（`verify/verify.sh` 组 9/10 的注释）。

| 项 | 要求 |
| --- | --- |
| OS 与编译器 | Windows 10/11 x64 + MSVC 14.5x（实测 14.51）+ CMake ≥ 3.20（实测 4.4.2）+ Ninja；Linux / Android 用 clang（实测 21.1.8），安卓是 15 arm64-v8a / Termux（`BUILD.md` §2.1、`ANDROID-SURFACE-段接入评估.md` §1.4） |
| Python 与磁盘 | Python 3.11+（`control/qlhctl.py` 只用标准库 `tomllib`）；模型之外还要源码树与构建目录，`build-split/bin` 实测 111 个文件（2026-11-11） |

## 3. 拿到模型
模型是**整模 GGUF**，不是裁切件。35B 官方版是 `Qwen3.6-35B-A3B-UD-Q4_K_M.gguf`，**22,663,387,424 B**，来自 `unsloth/Qwen3.6-35B-A3B-MTP-GGUF`（`CROSS-MACHINE.md` §6）。

**每台机器都要放同一份完整模型**：MTP 头在 `blk.40`、每段都要带 `token_embd`/`output`、换分段零传输，三条理由见 `DEPLOYMENT.md` §1。

放哪：任意本地磁盘目录，路径写进拓扑的 `model` 字段（例 `H:/qlh_models/qwen3-5-2b-gguf/qwen35-2b-Q4_K_M.gguf`）；**每台机器的路径可以不同**，写那台机器上的。

**下载地址仓库里没有**，从项目组提供的网盘/直链获取；体积见上，sha256 也由项目组给。拿到后核对：

```bash
stat -c '%s %n' <模型路径>/Qwen3.6-35B-A3B-UD-Q4_K_M.gguf   # 期望 22663387424 <路径>
sha256sum <模型路径>/Qwen3.6-35B-A3B-UD-Q4_K_M.gguf        # 与项目组给的 sha256 比对
```

`<模型路径>` 换成你放模型的目录。之后 `up` 自动比各段 `model` 的 size 与 mtime：size 不同拒绝启动，mtime 不同只警告（scp 会改）；`up --verify-weights` 另做完整 sha256（35B 几分钟）。这是启发式，不是密码学证明（`DEPLOYMENT.md` §4）。

## 4. 编译
源码按 `BUILD.md` §2.2 取：上游 `ggml-org/llama.cpp` 的 `f477f795c` + `docs/llamacpp-split.patch` + `docs/llamacpp-slimarc.patch`，顺序不能反；叠完 `git write-tree` 应得 `95291539ae833860cc8f49040bad3c187d42daa9`（只叠主补丁是 `5520f52cb`）。

**4.1 Windows（MSVC + Ninja）**：Ninja 必须在 `vcvars64` 环境里跑，Git Bash 里直接 `ninja` 会因缺 `INCLUDE`/`LIB` 报 `C1083: 无法打开包括文件: "stdbool.h"`。存成源码根的 `build-qlh.bat`：

```bat
call "C:\Program Files (x86)\Microsoft Visual Studio\18\BuildTools\VC\Auxiliary\Build\vcvars64.bat" >nul 2>&1
cmake -S . -B build-split -G Ninja -DCMAKE_BUILD_TYPE=Release -DGGML_NATIVE=ON -DGGML_CPU_ALL_VARIANTS=OFF -DGGML_CUDA=OFF -DGGML_VULKAN=OFF -DGGML_RPC=OFF -DLLAMA_CURL=OFF -DLLAMA_BUILD_TESTS=ON
cmake --build build-split --target split-stage llama-server llama-cli llama-bench split-smoke split-rows dflash-taps -- -j 8
```

执行 `cmd //c build-qlh.bat`；VS 版本号按自己机器改，ninja 不在 `PATH` 上时补 `-DCMAKE_MAKE_PROGRAM=<ninja.exe>`。实测 configure 16 秒，从零编 `split-stage` + `llama-server` 1 分 52 秒（187 步），其余 5 个 target 10 秒，产物在 `build-split/bin/`（`BUILD.md` §2.3）。

**4.2 Linux / Android（clang）**：同一组开关矩阵，去掉 `vcvars64`；Android 必须 `-j 1`（`-j 6` 会被 Android LMK SIGKILL，`ANDROID-SURFACE-段接入评估.md` §1.3）。**下面这条按 Windows 的开关矩阵写**——Android 上编过（clang 21.1.8），Linux 未逐字跑过：

```bash
cmake -S . -B build-split -G Ninja -DCMAKE_BUILD_TYPE=Release -DGGML_NATIVE=ON -DGGML_CPU_ALL_VARIANTS=OFF -DGGML_CUDA=OFF -DGGML_VULKAN=OFF -DGGML_RPC=OFF -DLLAMA_CURL=OFF -DLLAMA_BUILD_TESTS=ON
cmake --build build-split --target split-stage llama-server llama-cli llama-bench -- -j 1
```

**4.3 产物与冒烟**：`bin/` 里要有 `split-stage`、`llama-server` + `llama-server-impl.dll`、`llama-cli` / `llama-bench`、`split-smoke` / `split-rows` / `dflash-taps`（`BUILD.md` §6）。

```bash
cd build-split/bin && ./llama-cli.exe --version   # 期望 version: 0.9.0-dev (build 195, commit 1c4bda400)
./split-stage.exe                                 # 期望打印 usage，退出码 0
```

两条都过 = 动态库齐全、链接自洽。**只拷 exe 不算编好**：`llama-server.exe` 只有 10 KB，实体在 `llama-server-impl.dll`，必须连 `llama.dll`、`llama-common.dll`、`ggml*.dll`、`mtmd.dll` 一起拷（`BUILD.md` §2.4）。

**改过 `.h` 之后必须 `--clean-first`**：否则新旧 obj 混链进同一个 dll，ABI 不一致，症状是随机崩溃或数值错误而不是编译报错（`build-vulkan` 838 个对象里 528 个 `#deps 0`，`BUILD.md` §5.1）：

```bash
ninja -C build-split -t deps | awk '/#deps /{if($3+0==0)c++} END{print c}'   # 非 0 → 不可信
cmake --build build-split --clean-first -- -j 8                             # 全量重编，约 2 分钟
```

## 5. 第一次跑通：本地两段
tail 在同一进程里，不需要第二台机器。切点必须是 4 的整数倍（每 4 层才有一个 full attention）。

```bash
cd build-split/bin
./llama-server -m <模型路径>/qwen35-2b-Q4_K_M.gguf \
  --split-layer-range 0:20 --split-tail-range 20:24 -c 8192 -np 1 -t 4 --port 8080
./llama-server -m <模型路径>/Qwen3.6-35B-A3B-UD-Q4_K_M.gguf \
  --split-layer-range 0:32 --split-tail-range 32:40 -c 4096 -np 1 -t 8 -ngl 0 --port 8080
```

上面两条依次是 2B（24 层，切点 20，`README.md`「快速开始」口径）与 35B（40 层，切点 32，纯 CPU，`MODEL-SWAP.md` §11.2 口径）。`<模型路径>` 换成 §3 的目录；`--split-layer-range` 是本进程加载的层（从 0 起），`--split-tail-range` 是同机尾段，两者与 `--split-remote` 互斥。

**应该看到什么**：加载完之后一行 `listening on http://127.0.0.1:8080`（`DEPLOYMENT.md` §3）。另开一个终端验证：

```bash
curl -s http://127.0.0.1:8080/health          # 期望 {"status":"ok"}
curl -s http://127.0.0.1:8080/v1/chat/completions -H 'Content-Type: application/json' \
  -d '{"messages":[{"role":"user","content":"用一句话说明什么是 MoE"}],"max_tokens":64,"temperature":0,
       "chat_template_kwargs":{"enable_thinking":false}}'
```

期望一个 JSON：`content` 非空、`finish_reason` 是 `stop`、`timings.prompt_n` / `prompt_ms` / `predicted_n` / `predicted_ms` 有数（`CROSS-MACHINE.md` §6.2）。

**2B 与 35B 都是 thinking 模型**：不加 `enable_thinking: false` 时 token 全落在 `reasoning_content`、`content` 是空字符串（实测 2B 在 32 token 下就是空的）——那不是链坏了，是预算不够（§6.4）。

## 6. 用控制面起链
手敲超过两段就不划算。`control/qlhctl.py` 读一份拓扑起停与探活，存成 `topology-2b.toml`（可直接抄，改三处：`bin_dir`、两个 `model`、端口别和别人撞）：

```toml
[chain]
name = "2b-local"                            # run 目录与 state 的名字
bin_dir = "H:/llamacpp-c/build-split/bin"    # llama-server(.exe) 与 split-stage(.exe) 在这
rs_seq = 8                                   # 链首 --split-rs-seq 与各段 --rs-seq，两端必须一致
ready_timeout = 240                          # 单段 spawn 到 ready 的上限（秒）
[[stage]]                                    # 链首由 llama-server 承担
name = "head"
role = "head"
host = "127.0.0.1"                           # 该段在哪台机器；链首用它拼 --split-remote
port = 9461                                  # head 是 HTTP 端口
model = "<模型路径>/qwen35-2b-Q4_K_M.gguf"
layers = "0:12"                              # 原样给 --split-layer-range；控制面不校验
ngl = 0                                      # -ngl；上卡写 N 或 "all"
[[stage]]
name = "tail"
role = "tail"
host = "127.0.0.1"
port = 9462
model = "<模型路径>/qwen35-2b-Q4_K_M.gguf"
layers = "12:24"
```

**注意这份起的是回环 remote 链**（`--split-remote 127.0.0.1:9462` + 独立进程），不是 §5 那种同进程本地链：单序列、无链上 catch-up、每次 decode 多一次 socket。要本地链就只写一个 head stage、把 `--split-tail-range` 放进 `extra`。

字段全表见 `DEPLOYMENT.md` §2，能真跑的完整版是 `control/topology.example.toml`。

```bash
python control/qlhctl.py up     -f topology-2b.toml   # 尾段先起、链首最后
python control/qlhctl.py status -f topology-2b.toml
python control/qlhctl.py down   -f topology-2b.toml
```

**`up` 应该看到什么**：每段一行 `ready in N.Ns, pid <pid>: <ready 行>`，最后是 `链就绪：head 在 http://127.0.0.1:9461`；中途失败会停掉已起的段。

`--dry-run` 只打印命令行，`--kill-stale` 停掉先前留下的旧进程。

**`status` 应该看到什么**（照 `DEPLOYMENT.md` §3 的真实输出，最后一行是汇总）：

```
stage   role   layers   addr                  pid      proc  port   ready  verdict
head    head   0:12     127.0.0.1:9461        9876     live  open   yes    ok
tail    tail   12:24    127.0.0.1:9462        17932    live  open   yes    ok

2/2 段 ok
```

| verdict | 含义 |
| --- | --- |
| `ok` | pid 活 + 见过 ready + 端口在监听 |
| `DEAD` | pid 不在了（附日志最后一行） |
| `starting` | pid 活、还没见过 ready（正在加载） |
| `LISTEN-GONE` | pid 活、ready 见过，内核里却没有监听 |
| `unknown` | state 里没有这一段的记录 |

`status --deep` 另问链首 `/health?deep=1`（真发 1-token 请求、清该 slot 的 KV，只在怀疑链是不是活的时候用）；`--json` 给脚本用。

## 7. 跨机
| 机器 | 要做什么 |
| --- | --- |
| 两端 | 各放**同一份**整模（§3）；各自编出自己的二进制（§4） |
| 远端段 | 段级 `bin_dir` / `model` 写**那台机器上的**路径（段级 `bin_dir` 覆盖 `[chain]` 的） |
| 链首 | `host` 写远端地址；`--split-remote` 由控制面按链序拼，不用手写 |

`ssh` 是**本机与远端的唯一判据**：有它就走 ssh，没有就在本机 spawn——少了它，控制面会在本机起一个只存在于别的机器上的路径（`DEPLOYMENT.md` §2）。先确认免密：

```bash
ssh -o BatchMode=yes -o ConnectTimeout=10 surface@<tailscale-ip-surface> true   # 期望无输出、退出码 0
```

`bind`：链上没有认证，缺省只绑回环；`host` 不是回环时控制面会自己张到 `0.0.0.0`，要跨机连的那段必须让它监听非回环地址。

防火墙：跨机实测本机直连 `9462` 0.06 s 通、入站没挡，换机器换网络要自己确认（`DEPLOYMENT.md` §5）。

最小跨机改动只动 tail 段（`name` / `role` 照 §6；`workdir` 决定远端日志落点，缺省是远端 home）：

```toml
[[stage]]
name = "tail"
role = "tail"
host = "<tailscale-ip-surface>"                      # Tailscale / 局域网地址
ssh = "surface@<tailscale-ip-surface>"               # 有它才走 ssh
port = 9462
model = "C:/Users/<user>/Documents/llamacpp-split/Qwen3.6-35B-A3B-UD-Q4_K_M.gguf"
layers = "32:40"
bin_dir = "C:/Users/<user>/Documents/llamacpp-split/bin"
```

验收流程（`CROSS-MACHINE.md` §6.3 实测全过）：

| 步骤 | 命令 | 应该看到 |
| --- | --- | --- |
| 1 起链 | `python control/qlhctl.py up -f topology.toml` | `链就绪：head 在 http://…` |
| 2 探活 | `python control/qlhctl.py status -f topology.toml` | `2/2 段 ok` |
| 3 生成 | `curl` 打 `/v1/chat/completions` | 正常 JSON（35B 注意 thinking，§5） |
| 4 杀尾段 | 在远端 `kill <pid>`，再跑 `status` | `DEAD` 且退出码 1 |
| 5 判死 | 浅 `/health`；`/health?deep=1` | 浅探针仍 `200 {"status":"ok"}`；`?deep=1` `503` 并点名哪段 |
| 6 收尾 | `python control/qlhctl.py down -f topology.toml` | 两端干净，端口 connection refused |

段挂了只重起那一段，不要重起整链（`DEPLOYMENT.md` §4）。

## 8. 安卓（Termux）
尾段跑在安卓上：原生 ARM64 ELF，在 Termux 里编，不走 NDK（`CROSS-MACHINE.md` §6）。五条实测的坑：

1. **模型必须放 Termux 私有目录**，不能放 `/sdcard`：FUSE 上 mmap 极慢甚至失败，`run-as` 起的进程连 `/sdcard` 都读不了（SELinux 拒绝）；传模型走 `adb forward tcp:18022 tcp:8022` + `scp`，实测 40 MB/s（`ANDROID-SURFACE-段接入评估.md` §1.3 第 3、4 条）。
2. **编译 `-j 1`**：`-j 6` 会被 Android LMK SIGKILL；推理时 20 层（11 GiB）会 OOM、16 层（8.8 GiB）稳定（同 §1.3 第 5 条）。那个水位是 v2 的 Q6_K worker 口径，不是 v2.1 的上限：v2.1 的 UD-Q4_K_M 尾段 `32:40` 是 8 层约 4.90 GiB。
3. **`LD_LIBRARY_PATH` 没有槽位**：二进制要它才能找到同目录的 `libllama.so` / `libggml*.so`，而拓扑没有环境变量字段。写 wrapper 脚本、把 `bin_dir` 指向 wrapper 所在目录（§1.4）：

```sh
#!/data/data/com.termux/files/usr/bin/sh
BIN=$HOME/qlh/build/bin
export LD_LIBRARY_PATH="$BIN:$LD_LIBRARY_PATH"
exec "$BIN/split-stage" "$@"
```

   wrapper 命名成 `split-stage` 放在独立目录（如 `~/qlh/binwrap/`），真二进制留在 `~/qlh/build/bin/` **不改名**——`exec` 之后命令行是真二进制，所以 `pkill -f split-stage` 杀得到、`pkill -f "binwrap/split-stage"` 杀不到。
4. **`ps` / `pgrep` 看不见别的进程**：`ps -ef` 只列自己那条 ssh 会话，`pgrep -af split-stage` 在尾段活着时返回空。判断「那段还在吗」只能 `kill -0 <pid>`（控制面正是这么探的）或问端口 / 协议层（§1.4、`CROSS-MACHINE.md` §6.5）。
5. **长命进程要脱离 ssh 会话**：`setsid nohup ./split-stage ... > run.log 2>&1 < /dev/null &`，否则 ssh 一结束进程就死（`CROSS-MACHINE.md` §6.5）。就绪行形如 `[split-stage] ready: layers [32, 40) of 40, role=tail, n_embd=2048, n_rs_seq=8`（同 §6.2）。

## 9. 故障排查
| 症状 | 最可能原因 | 处置 |
| --- | --- | --- |
| `status` 报 `DEAD`，日志停在 `OP_STATE_SAVE`、没有退出原因 | 段被外部结束，不是加载失败 | 只重起那一段，不要重起整链（`DEPLOYMENT.md` §4） |
| `up` 当场停下，报「端口 X 被 pid Y 占着」 | 先前起的段没清干净 | `up --kill-stale`；不是自己起的就换端口（`DEPLOYMENT.md` §4） |
| `up` 报各段 size 不同、拒绝启动 | 两台的模型不是同一份 | 重传；`--verify-weights` 做 sha256（`DEPLOYMENT.md` §4） |
| 启动报 `C1083: 无法打开包括文件: "stdbool.h"` | Git Bash 里直接跑 ninja，没有 vcvars64 的 INCLUDE/LIB | 用 `build-qlh.bat`（内部 `call vcvars64.bat`）（`BUILD.md` §2.3） |
| 编完随机崩溃或数值不对，但没有编译错 | 改过 `.h` 后没 `--clean-first`，新旧 obj 混链进同一个 dll | `cmake --build build-split --clean-first -- -j 8`（`BUILD.md` §5.1） |
| 启动报「找不到指定的模块」 | 只拷了 exe，没拷 `llama.dll` / `llama-common.dll` / `ggml*.dll` / `mtmd.dll` | 整个 `bin/` 一起拷（`BUILD.md` §2.4） |
| 起链被拒，说 `n_seq_max > 1` | 远程链单序列，`-np > 1` 不被支持 | `-np 1`；要多并发就在 head 起多个 server（`SEGMENTED-LIMITS.md` §2.4） |
| 加载后、建上下文前报错（范围不从 0 起，或中间结束没给 tail/remote） | `--split-layer-range` 单独起 server 的两道检查 | 从 0 起；中间结束补 `--split-tail-range` 或 `--split-remote`（§2.6） |
| 浅 `/health` 一直 `200`，生成请求却 `500` | 链死了而进程活着（`/health` 是存活语义） | 问 `status` 或 `/health?deep=1`（`SEGMENTED-LIMITS.md` §2.7） |
| 输出通顺但内容是错的 | 段读错了层，不报错 | 跑 `verify/verify.sh` 逐位验收（`SEGMENTED-LIMITS.md` §2.6 残留边界） |
| 回退场景下同一位置算出不同 `h` | `seq_rm` 失败被静默忽略（上游 hybrid 的已知基线） | 本仓已改 `bool` 并降级；`verify_seq_rm_exact.py --expect-known-fail` 对照（§2.9） |
| 安卓上的编译 OOM / 模型读不了 / `pgrep` 说进程没了 | Android LMK、`/sdcard` 的 FUSE+SELinux、Termux 的 `ps` 看不见别的进程 | 见 §8 的五条坑：`-j 1`、模型挪私有目录、`kill -0 <pid>` |

## 下一步读什么
| 想知道 | 读 |
| --- | --- |
| 机制与硬约束 / 每台放什么、拓扑与故障处置 | [`ARCHITECTURE.md`](design/ARCHITECTURE.md)、[`DEPLOYMENT.md`](deploy/DEPLOYMENT.md) |
| 每条边界的根因与代价 / 判据与验收通道 | [`SEGMENTED-LIMITS.md`](limits/SEGMENTED-LIMITS.md)、[`TESTING.md`](design/TESTING.md) |
| 构建开关矩阵、CPU 变体、MSVC 坑 | [`BUILD.md`](deploy/BUILD.md) |
