# 部署与分段

> 状态：**现行**
> 更新日期：2026-11-11
> 适用范围：把一条 v2.1 分段链在自己的机器上跑起来——每台机器放什么、拓扑怎么写、怎么起停与
> 观察、段挂了怎么办、跨机要配什么。读者是拿到 patch 与仓库的人。二进制怎么编见
> [`BUILD.md`](BUILD.md)，能力边界见 [`SEGMENTED-LIMITS.md`](SEGMENTED-LIMITS.md)。

v2.1 的一条链 = **一个 `llama-server` 链首 + N 个 `split-stage` 段**。链首加载源模型的
`[0, N)` 层，用 `--split-remote` 按链序连下游；每段只加载自己的层范围，段间只传 `n_embd` 个
float 的隐藏态。控制面 `control/qlhctl.py` 只做三件事：读拓扑、起停进程、探活。它**不**实现
协议、不碰 token、不理解层、不采样——协议在 `split-stage` 与 `llama_context` 里
（[`PROTOCOL-V9.md`](PROTOCOL-V9.md)）。

## 1. 每台机器放什么

```
<整模>.gguf           35B-A3B 约 21.87 GiB；每台机器都要有同一份
build-split/bin/      llama-server(.exe) + split-stage(.exe)，由 BUILD.md 编出
topology-<站点>.toml  拓扑，人写，不入库
```

**没有"自己那段"这种东西。** 每台机器持有同一份完整 checkpoint，加载时由
`--split-layer-range` / `--layer-range` 决定只创建、只读取哪几层。

存整模的三个原因：

1. **MTP 头**。裁切件没有 `blk.40`，而投机解码要它。
2. **空间上反而更省**。每段都要重复带 `token_embd` / `output.weight`（各约 0.39 GiB），
   四段合计比整模还大。
3. **换分段零传输**。段是派生物，改拓扑里的数字就是换分段；裁段方案要重切、重 scp。整模传一次即可——ROG 实测 22 MB/s（跨 Tailscale），21.87 GiB 约 17 分钟。

同一台机器可以同时跑多段，但**两种写法各自成链**：`--split-tail-range` 是同进程本地链（省一次 socket、
`-np > 1`、链上 catch-up 可用）；拓扑里两个都不带 `ssh` 的 `[[stage]]` 起的是**回环 remote 链**（单序列、无
catch-up、每次 decode 多一次 socket），控制面只会拼后者。两种形态与写法见 `control/topology.example.toml`。

## 2. 拓扑怎么写

`[[stage]]` 按链序写：第一个是 head，最后一个是 tail。完整可跑的例子是
`control/topology.example.toml`。

| 字段 | 说明 |
| --- | --- |
| `[chain] name` | run 目录与状态文件的名字；同机跑多条链靠它区分 |
| `[chain] bin_dir` | 两个二进制所在目录（本地或该段机器上的路径） |
| `[chain] rs_seq` | recurrent 快照数，同一个值写进链首 `--split-rs-seq` 与各段 `--rs-seq` |
| `[chain] ready_timeout` | 单段 spawn 到 ready 的上限（秒） |
| `[chain] ctx` / `batch` | 各段 `-c` / `-b` 的缺省值 |
| `[chain] run_dir` | 可选；state 与日志目录，缺省是当前目录下的 `.qlh_tmp/qlhctl/<name>`（换目录跑就看不到上次的记录） |
| `name` | 日志与 state 里的标识 |
| `role` | `head`（llama-server）/ `relay` / `tail` |
| `host` | 该段在哪台机器；链首用它拼 `--split-remote` |
| `ssh` | **本机与远端的唯一判据**：有它就走 ssh，没有就在本机 spawn |
| `port` | head 是 HTTP 端口，其余是 split-stage 的监听端口 |
| `model` | 该段加载的整模路径（远端写远端的路径） |
| `layers` | `"lo:hi"`，原样转成 `--split-layer-range` 或 `--layer-range` |
| `ctx` / `batch` / `threads` / `ngl` | 段级覆盖 `-c` / `-b` / `-t` / `-ngl`（`ngl` 可为 `"all"`） |
| `bind` | 监听地址；缺省 host 是回环→`127.0.0.1`，否则 `0.0.0.0` |
| `workdir` | 远端：ssh 起进程前 `cd` 到此，日志落 `<workdir>/<name>.log` |
| `extra` | **字符串数组**，按序追加：`["-md", "dflash.gguf", "--spec-type", "draft-dflash"]`。写成单个字符串会被**逐字符**拆成参数（`[str(a) for a in extra]` 对 str 会迭代字符），不报错，只是命令行变成一堆单字母 |

三条容易写错的：

- **`ssh` 决定"谁来起"。** 没有它，控制面会在本机 spawn 一个只存在于别的机器上的路径。
- **`rs_seq` 必须两端一致。** 给不一致时 C++ 不报错，症状是回退静默失效而不是启动失败。
- **`layers` 控制面不校验。** 它只做字符串转发：`role=tail` 要求 `layer_hi == 源模型层数`、
  `role=relay` 要求 `layer_hi < 源模型层数`（`split-stage` 的 `validate_options`）；整条链
  首尾相接、末端等于源模型层数，由链首的 `split_attach_remote()` 在 attach 时判。

链首另有三条硬约束：层范围从 0 开始；`--split-remote` 是单序列的（`-np 1`）；每段的 `-c` / `-b`
不得小于链首——identity 握手**不校验容量**，给小了要跑到长 prompt 才报 `ST_NO_SPACE`。

切点取 4 的整数倍（`qwen35` / `qwen35moe` 每 4 层才有一个 full attention）。**v2.1 没有代码校验
这一条**：控制面不查，`split-stage` 也不查，判据只能是 [`TESTING.md`](TESTING.md) 的逐位验收。

## 3. 起停与观察

```bash
python control/qlhctl.py up     -f control/topology.example.toml
python control/qlhctl.py status -f control/topology.example.toml
python control/qlhctl.py down   -f control/topology.example.toml
```

`up` 按链序反着起（尾段先、链首最后），每段等到它自己打印 ready 才起下一段：`split-stage`
打 `[split-stage] ready: layers [12, 24) of 24, role=tail, …`，`llama-server` 打
`listening on http://…`。中途失败会把自己已起的段全部停掉，不留半条链。

`status` 的一段真实输出（2B 两段，head `0:12` + tail `12:24`）：

```
chain 2b-local    run_dir <工作目录>\.qlh_tmp\qlhctl\2b-local

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

`status --deep` 另问一次链首的 `/health?deep=1`：它会真发一个 1-token 请求，代价是清掉该 slot
的 KV，只在这条链是不是活的存疑时用。`--json` 给脚本用。`up --dry-run` 只打印命令行。

`down` 只停 state 里记过的 pid：停前校验映像名、停后确认不在，最后复核端口；不碰别人的进程。

## 4. 故障处置

**段挂了**（`status` 报 `DEAD`，实测 0.1–0.2 s、exit 1）：重起那一段，不要重起整链——实测 tail
原样重起后，链首的下一个请求 0.63 s → 200。

**杀远端段要杀对进程。** Windows 远端段是 `cmd.exe` 壳 + `split-stage.exe` 子进程、`status` 报的是壳的 pid；只杀壳会得到「DEAD 但链还在答」——杀 `split-stage.exe` 本身或 `taskkill /F /T` 杀树（`CROSS-MACHINE.md` §7）。

**不要用 `/health` 判断链的死活。** 浅探针在链死时仍 `200 {"status":"ok"}`（存活语义）。链的
真假只有两处能问：`/health?deep=1`（链死 503 并点名哪段）或控制面的 `status`。

**生成请求快速失败。** 杀掉的段让 socket 立刻断，第一个请求 1.5 s 量级拿到 500
（`the split chain is no longer trustworthy: remote stage 1 …`），之后每个几十毫秒失败。120 s
那个数字属于**挂起**（`NtSuspendProcess`），不属于杀死。

**端口被占。** `up` 起进程前先查监听表，占着就当场停下，报 pid、映像名，以及"这是不是我上次
留下的"（比对映像名 + 命令行里的 `--port` 与层范围）。是的话 `up --kill-stale`；不是只报告。

**权重可能不是同一份。** `up` 默认比各段 `model` 的 size 与 mtime：size 不同拒绝启动，mtime 不同
只警告（scp 会改）。`--verify-weights` 另做完整 sha256（35B 几分钟）。**这是启发式，不是密码学
证明**——`OP_HELLO` 只报范围 / 深度 / 宽度，同 size 同 mtime 的两份不同权重比不出来。

## 5. 跨机

把某段挪走：`host` 写那台机器，加 `ssh = "user@host"`（免密），`bin_dir` 与 `model` 写**那台机器上的**路径，`workdir` 决定日志落点。先确认免密能进：

```bash
ssh -o BatchMode=yes -o ConnectTimeout=10 surface@<tailscale-ip-surface> true
```

**远端 shell 不用配。** 控制面探一次 `echo $0`：POSIX shell（Linux / Termux / MSYS2）脚本走
参数；Windows 的 `cmd.exe` 脚本走 stdin 交给 Git Bash，并用 WMI 起进程。Windows 上 `nohup`
不管用——OpenSSH 会话结束会杀掉整棵进程树（实测 `nohup sleep 60 &` 在 ssh 退出 2 秒后就没了），
绕到 WMI 之后 pid 变成一层 `cmd.exe`，停止用 `taskkill /T` 杀树。Git Bash 默认按
`C:/Program Files/Git/bin/bash.exe` 找，装在别处用 `remote_shell` 指路。

**跨机链已实测**（本机 head `0:32` + Surface tail `32:40`，Tailscale，35B 官方版）：tail 10 s、
head 15 s 就绪；`/v1/chat/completions` 11 s 出 48 个 token；杀掉 Surface 上的段之后 `status`
一次调用报 `DEAD` 并指出 `<tailscale-ip-surface>:9462`，浅 `/health` 仍 200、`?deep=1` 503 并点名远端，
生成请求 0.01 s 失败——与本机两段完全一致。**入站防火墙未挡**：本机直连 `9462` 0.06 s 通。

链上没有认证，所以 `bind` 缺省只绑回环，要跨机连才写 `0.0.0.0`。35B 跨机（本机 head `0:32` + 安卓 Y700 tail `32:40`）也走完了同一套五步验收，数字在 `CROSS-MACHINE.md`。

**安卓/Termux 作尾段**：二进制要 `LD_LIBRARY_PATH` 才能找到同目录的 `libllama.so` / `libggml*.so`，而拓扑没有环境变量槽位；用 wrapper 脚本（设 `LD_LIBRARY_PATH` 后
`exec` 真二进制）并把 `bin_dir` 指向 wrapper 所在目录。细节与实测见 `ANDROID-SURFACE-段接入评估.md` §1.4 与 `CROSS-MACHINE.md` 的安卓实测。

**未验证**：网络抖动与 MTU（Tailscale 退回 DERP 时的行为）、两端时钟同步、超过两段的链。

## 6. 调参实测：`-ngl`、`threads` 与按段量化

这一节回答"这一段该给多少资源"。数字在 35B 两段上测得（v2 的驱动链，量的是段内耗时与整链
tg），结论对 v2.1 成立；节号 `§6.5`–`§6.8` 沿用 v2 的编号，别处按它引用。

### 6.5 `-ngl` 怎么定：扫一遍，别猜

`-ngl` 决定这段有多少层上卡。它有明确拐点，拐点之后性能回落，而且回落时**单段耗时反而更短**。
本机两段（head 32 层）实测（2026-11-01，DFlash k=5，`--gen 128`，代码 prompt）：

| `-ngl` | tg | stage 0 中位 | 判 |
| ---: | ---: | ---: | --- |
| 4 | 7.2 tok/s | 394.6 ms | — |
| 5 | 7.9 | 363.1 | +10% |
| **6** | **8.7** | 354.0 | **+21%，拐点** |
| 7 | 8.1 | **346.1** | 回落 |

判据是 tg，不是单段耗时：7 层那行 stage 更快（346.1）、tg 更慢（8.1），因为溢出的权重每 token
过一次系统内存。6 层本身也在溢出（84% 驻留，低于 `VRAM_LANDED_SLACK=0.85` 会告警），但它仍是
四档里最快的——溢出与更慢不是一回事。

### 6.6 `-ngl` 与 `cpu_moe`：专家留 CPU，注意力上卡

对 35B-A3B 这种 MoE，把**专家**留在 CPU、把**注意力**搬上卡，比整层 offload 好得多。本机
（RTX 4060 Laptop 8 GiB）+ Surface 两段，k=2、直连、2026-11-02：

| `-ngl` / `cpu_moe` | tg | stage 0 | 预期驻留 |
| --- | ---: | ---: | ---: |
| 6 / 0（整层 offload） | 3.8 tok/s | 179.5 ms | 7.55 GiB |
| 14 / 32 | 4.2 | 168.3 | 10.61 |
| 22 / 32 | 4.8 | 166.2 | 13.67 |
| **26 / 32** | **5.1** | 147.4 | 15.20 |
| 32 / 32 | 5.0 | **135.4** | 17.50 |

判据是整链 tg，不是 stage 耗时。**32/32 是陷阱**：stage 最快（135.4）、整链最慢（5.0），因为
预期驻留 17.50 GiB 逼近内存预算开始换页（显存只多占 1.6 GiB）；内存预检随机器空闲程度变
（`-ngl 32` 忙时 15.90 GiB 拒绝、空闲 22.72 放行）。前提是权重构成：专家占 91%（19.86 / 21.87 GiB）
但每 token 只命中 8 个，注意力只占 9% 却每 token 全部要算。机制与完整实测见
[`SEGMENT-OFFLOAD.md`](SEGMENT-OFFLOAD.md) §5。

### 6.7 弱机器段的 `threads`：6 比 4 快，8 会崩

Surface（i5-1035G4，4 物理核 8 逻辑核）跑 8 层尾段，k=2、直连、head 固定 26/32：

| `-t` | tg | tail 段耗时 |
| ---: | ---: | ---: |
| 3 | 4.0 tok/s | 151.4 ms |
| 4（物理核数） | 3.6 | 138.5 |
| **6** | **4.3 / 5.2**（两次测量） | **130.4 / 113.2** |
| 8（全部逻辑核） | **0.2** | **6994** |

判据仍是整链 tg：**6 比 4 快**（tail 段有大量等内存的停顿，超线程那一半能填进去——"取物理核
数"这条经验在内存受限的模型上不成立），**8 会崩**（慢 50 倍）。同一配置两次测量差 21%，跨时间
比较前先确认机器空闲。

### 6.8 量化类型按段分布：IQ4_XS 全在 head

直读 GGUF 头（753 个张量，整模 21.86 GiB；2026-11-11 逐张量复算，按 `split-stage` 的 `tok_embd=skip / output=load` 归段）：

| 段 | 驻留 | 构成 |
| --- | ---: | --- |
| head（blk.0–31 + `token_embd`） | 15.88 GiB | **IQ4_XS 8.37** + Q6_K 3.86 + Q5_K 3.09 + Q8_0 0.10 + `token_embd` 0.39 |
| tail（blk.32–39 + NextN + `output`） | 5.99 GiB | Q6_K 2.66 + Q5_K 2.06 + Q8_0 0.03（NextN 0.84 也是 Q8_0、`output` 0.39 是 Q6_K），**零 IQ4_XS** |

两件事因此成立：

- **repack 换量化换不来**：IQ4_XS 占权重面积最大，而它在所有后端都只有 MMV、没有 MMQ。
- **核显只可能救 tail**（对 35B）：head 的 8.37 GiB IQ4_XS 塞进核显共享堆会连系统一起挤死，
  tail 全是 Q5_K / Q6_K / Q8_0；9B / 2B 这类小模型不受此限（[`IGPU-SLOWDOWN.md`](IGPU-SLOWDOWN.md)）。

## 7. 控制面不做什么

- **不做自动切分求解。** 拓扑是人写的；prima.cpp 的 806 行整数规划前提是"设备未知、数量不定"
  （条件见 [`AUTO-ASSIGN-PLAN.md`](AUTO-ASSIGN-PLAN.md)）。
- **不做工件传输。** 没有工件，整模一次性 scp，之后不再传模型文件。
- **不做动态重平衡。** 推理中不重新分配层；prima 的调度也是纯静态。
- **不实现协议、不碰 token、不理解层、不采样。** 那些在 `split-stage` 与 `llama_context` 里。

全部实现是 `control/qlhctl.py`（944 行）加 `control/topology.example.toml`（56 行），合计 1000
行——控制面一千行以内的承诺。删掉它链照样能手工起，它省掉的只是"记住五条命令、盯五个日志、
猜哪一段死了"。

## 8. v2 形态已退场

v2 用 `shim` + `chain_client`，已归档在 `slimarc/prototype`；`_archive/topology-v2/` 的 15 份拓扑只作字段对照；v2 实测数据仍可用（口径见 [`MEASUREMENTS.md`](MEASUREMENTS.md)），但当年的部署步骤不是现行。
