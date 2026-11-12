# 跨机实测：ROG、Surface、Y700 安卓、回环与 Tailscale

> 状态：**现行**
> 更新日期：2026-11-12
> 适用范围：远程链投机在三种真实部署形态上的收益（LAN 强机、弱机、回环）、Tailscale →Surface 的首次真跨机零损失实测，以及安卓尾段（
> Termux / arm64，Lenovo Y700）上 v2.1 跨机的吞吐、就绪与破坏性验收。逐 prompt 与 k 扫描归
> [`SPECULATION-DATA.md`](../measure/SPECULATION-DATA.md)，MTP 链上折扣的根因归 [`MODEL-SWAP.md`](../measure/MODEL-SWAP.md) §11.4，协议与传输
> 量归[`PROTOCOL-V9.md`](../design/PROTOCOL-V9.md)。

口径：35B（`qwen35moe`，40 层 trunk + 1 个 NextN 层），head `0:32` + tail `32:40`，贪心，32-token 生成。数据来自 ROG（LAN）、
Surface（4C8T 弱机器）与本机回环；`build-split/` 里没有与之对应的日志文件，按「可信到实测记录这一层、独立复算没做过」对待。**§1–§2 测在旧蒸馏版上**（官方模型到位之前），官
方版接受率高 30–76%，收益预计更高但未重测（见 [`MODEL-SWAP.md`](../measure/MODEL-SWAP.md) §11.7）。

## 1. ROG（LAN）：五类 prompt × 三配置

| prompt（token） | 不投机 | ngram | MTP k=2 | MTP 变化 |
|---|---:|---:|---:|---:|
| 散文 882 | 8.18 | 7.33 | 8.47 | +3.6% |
| 代码 1205 | 8.74 | 6.92 | 9.47 | **+8.4%** |
| JSON 2379 | 9.47 | 7.09 | 9.36 | **−1.2%** |
| agent 2508 | 7.51 | 5.09 | 7.62 | +1.5%（当时无 draft） |
| 人为重复 `A+B+A+B+A` 89 | 9.73 | 7.75 | 10.75 | +10.5% |

**中位约 +6%，区间 +8.4% ~ −1.2%。这就是跨机投机的真实量级。** 历史那个 +15%（10.20 vs 8.88 tok/s）来自一个 27-token 的高确定任务：同一任务上 k=1
是 10.11、k=2 是 10.20、k=3 是 9.76——k≥3 的验证帧要 k+1 行，而 **head 有 32 层、tail 只有 8 层，宽帧的代价几乎全在 head**，所以「k=2 是甜点」只
在那个切点上成立。

**agent 那行 +1.5% 是在「当时无 draft」下测的**（tail 没给出 NextN 草稿），它证明的是「没有 draft 时投机不会变慢太多」，不是 MTP 在 agent 负载上的上限。
**人为重复那行 +10.5% 与 ngram 那列的 7.75 一样**，只说明「可复制的 prompt 上投机好用」。ngram 五类全部低于不投机，原因见
[`SPECULATION-DATA.md`](../measure/SPECULATION-DATA.md) §4。

## 2. 弱机器与回环：收益全部转负

| 配置 | 不投机 | 投机 k=2 | 相对不投机 |
|---|---:|---:|---:|
| Surface（4C8T，纯 CPU）散文 | 8.65 | 5.21 | **−40%** |
| 本机回环，带 grammar（`toolcall.gbnf`） | 20.82 | 13.95 | **0.67×** |

**四个真实原因**：

① **每帧两次 hop**——`OP_MTP_DRAFT` 出草稿、`OP_DECODE` 再验证，一轮走两趟链。

② **MTP draft 不看 grammar / rbudget**——同一个 tail 上，接受率从无 grammar 时的 **2.76 token/帧**掉到 **1.46–1.73**。

③ **回退率 9–19%**——`toolcall.gbnf` 这种严格语法实测 **11/16 帧被拒**，而每次回退 ≈ 一整个普通 round（撤 KV + 重放 prompt + sampler 回滚
）。

④ **回环上 993 KB 只要约 1 ms**——验证通道省掉的四个数量级带宽在回环上本来就不值钱，于是这笔账只剩成本。另有：**FORCING 期间 tail 回的未掩码 argmax 基本每轮都被拒
**，有约 **2 倍白跑**。

所以「真实多机上划算」只能写成条件句：**LAN + 强 head + 接受率高的负载**。同一副链换切点、换机器或换到回环都会翻号。这与「本机双进程测不出投机收益」互为两面：本机的负数不能外推成「多机一定负
」，多机的中位 +6% 也不能外推成「任何多机都为正」。

## 3. Tailscale → Surface：第一次真跨机零损失（2026-11-09）

口径：head 本机加载 `0:32`，tail 在 Surface 上 `32:40`，用
`--split-layer-range 0:32 --split-remote <surface>:<port>`（`--split-remote` 与 `--split-tail-range` 互斥，所以
只能这么写）；target 是官方 `UD-Q4_K_M`（`22,663,387,424 B`，Surface 上这份是这一轮刚传的，与本地大小一致）；k=2、贪心；prompt 1102 token，三个
prompt 各跑一遍，三配置 A / B / C 共用同一份 target 与同一组 prompt。

### 3.1 接受与输出：零损失

| 配置 | 接受 | 与整模逐 prompt 比 |
| --- | --- | --- |
| A DFlash k=2 | **117/141 = 0.82979** | **相同** |
| B DSpark k=2 | **117/140 = 0.83571** | **相同** |
| C MTP k=2（对照） | 108/157 = 0.68790 | 对照（旧路径；nextn 通道的 0.688 上限，见 [`MODEL-SWAP.md`](../measure/MODEL-SWAP.md) §11.4） |

三个 prompt 的输出 **3/3 逐字节 IDENTICAL**（250 / 244 / 249 字符）。这是本项目第一次在**真跨机**链上拿到「与整模逐 prompt 相同」的接受率——此前只在
loopback 上验过 tap 开销可忽略，跨机验的是 wire 上 tap 逐位与回退后的输出。

### 3.2 prefill 的 tap 代价（1102 token）

| 配置 | prefill 耗时 |
| --- | --- |
| A DFlash | 32784 / 31506 ms → **~32.1 s** |
| B DSpark | 32840 ms → **~32.8 s** |
| C MTP | 27499 / 28212 / 29698 / 30783 ms → **~29.05 s**（四次均值 29048 ms） |

**A − C ≈ +3.05 s**，与「25 MB ÷ 9.5 MB/s ≈ 2.6 s」同量级（[`PROTOCOL-V9.md`](../design/PROTOCOL-V9.md) §3.3：prefill 每个 512
行 ubatch 是 8 MB，1102 token 约 3 个 ubatch 即 24 MB）。**没有 10 s 级的低效**——如果 tap 是逐 token 往返，1102 个 token 会报出数量
级更大的差；实测差值与带宽估算同量级，说明 tap 是**成批**走的。与 loopback 对照：1192 token 差 **0.2%**；跨机 1102 token 差 **~3.05 s（9–10%）
**。

### 3.3 结论与两个边界

- **accept 零损失**；长 prompt 多花 2–3 s 可接受（数千 token 的 prompt 会到 10 s 级）；收益排序**DSpark ≥ DFlash > MTP**；长
  prompt 用 DFlash 划算，短 prompt 上 tap 的固定开销占比变大。
- **边界一**：`--split-remote` 与 `--split-tail-range` 互斥，远程链只能用
  `--split-layer-range 0:32 --split-remote host:port`，不能同时指定同机尾段。
- **边界二**：Surface 原本没有这份 21 GB（`22,663,387,424 B`）的 target——只有旧切分片 `6.85G` / `2.66G` 与 23.5G 的蒸馏版。体积对不上时
  跨机输出会直接分叉，所以必须先核对大小。

## 4. 本机双进程为什么测不出投机收益

`SPECULATION-DATA.md` 与 `MODEL-SWAP.md` 里本机测的投机数字都是「head 与 tail 跑在同一台机器上」，这个配置对投机是退化的。

**它拿不到投机唯一该拿的好处**——投机减少的是每轮往返次数（一轮前进 3.2 token 而不是 1），而本机双进程里 head↔tail 走内存与管道，没有 25–30 ms 那个跨机往返，付出
draft 成本却省下一个不存在的量。

**它放大了投机的代价**——两个进程抢同一套 CPU 核心、L3 与内存带宽，而 draft（MTP 的 nextn 层 / DFlash 的 6 层 dense）是纯额外算力，全压在已经很挤的 tail
那台「机器」上。

所以本机测出的「亏」是对一个不会部署的配置的测量；它不能反推出「多机一定赚」，跨机的正确答案是「小幅正收益，随负载翻号」。反过来，checkpoint 复用、并发、内存这些不受这条影响——它们的收益不来自跨机
往返。

## 5. `split-stage` 的线程与 ubatch（Surface 实测）

`GGML_DEFAULT_N_THREADS = 4`，**而且 `n_threads_batch` 也是 4、不会自动跟随**——Surface（4C8T）上原来只用了 4 个线程。现在
`split-stage` 有 `-t/--threads`（同时设两者）、`-tb/--threads-batch`（单独覆盖 batch 那一半）、`-ub/--ubatch-size`（让验证帧在帧内逐
行跑，是测量旋钮不是调优旋钮）。

| 配置（Surface） | 原来（4 线程） | `-t 8` | 变化 |
|---|---:|---:|---:|
| 不投机 | 7.41 | 7.67 | +3.5% |
| MTP k=2 | 5.12 | 5.48 | +7% |

**只设 `n_threads` 拿不到这笔收益**，batch 那一半必须一起设；head 的 `-t` 不会过网。

## 6. 安卓尾段（Y700 / Termux / arm64）

口径：v2.1 分段链，本机（Windows + RTX 4060 Laptop）跑链首 `llama-server`，Lenovo Legion Y700（型号 TB321FU，Android 15，
arm64-v8a，骁龙 8 Gen 3，15.6 GB RAM）在 Termux 里跑尾段 `split-stage`，两端经 Tailscale 连通（Y700 的 tailscale IP 是
`<tailscale-ip-y700>`，sshd 在 8022），两端各自持有同一份整模。尾段二进制是原生 ARM64 ELF，在 Termux 里编，不走 NDK。拓扑字段与 ssh 配法见
[`DEPLOYMENT.md`](DEPLOYMENT.md) §5，本节只记安卓形态的数字与结论。

| 模型 | 整模 | 切点（head / tail） | 尾段配置 |
| --- | --- | --- | --- |
| 2B | `qwen35-2b-Q4_K_M.gguf`（1,312,164,672 B） | `0:12` / `12:24` | — |
| 35B | `Qwen3.6-35B-A3B-UD-Q4_K_M.gguf`（22,663,387,424 B） | `0:32` / `32:40` | 8 层，驻留约 4.90 GiB，`-t 8`、`ngl 0` |

尾段权重读 GGUF 头逐张量累加（`gguf` python 包，753 个张量）：整模 **21.10 GiB** = head `0:32` 15.690 + tail `32:40` **4.022
** + NextN（blk.40）0.493 + `output.weight` 0.389 + `token_embd.weight` 0.503。尾段是末段（`has_nextn=1`），NextN 层也
在它这儿，所以实际驻留 **约 4.90 GiB**（= 4.022 + 0.493 + 0.389，`output_norm` 可忽略）。这与
[`DEPLOYMENT.md`](DEPLOYMENT.md) §6.8 里 Q6_K 蒸馏版尾段的**5.99 GiB** 不是同一份权重（那一版 blk.32–39 是 4.76、NextN 是
0.836 的 Q8_0、`output` 0.389），两处数字不能互相印证。

`-t 8` 是这组实测采用的配置值，**不是调优结论**；v2 时代[`ANDROID-SURFACE-段接入评估.md`](ANDROID-SURFACE-段接入评估.md) §6 记的「`-t 5` 最
优、`-t 8` 退化 40%」是 16 层 Q6_K worker 的口径，两者不可直接比较。

### 6.1 2B：15.76 tok/s，杀尾段后链可判死

| 检查 | 实测 |
| --- | --- |
| `status` | `2/2 段 ok` |
| 64 token 生成 | **15.76 tok/s** |
| 杀掉 Y700 上的尾段后 `status` | `DEAD`，退出码 1 |
| 浅 `/health` | 仍 `200` |
| `/health?deep=1` | `503`，点名 `remote stage 1 (<tailscale-ip-y700>:9482)` |
| 生成请求 | 3 ms / 12 ms 拿到 `500` |
| `down` 之后 | 两端干净 |

**2B 的尾段 `12:24`（12 层）压在这块平板上是 15.76 tok/s**；**杀段后链的死活只有 `status` 与 `?deep=1` 能判**——浅探针照样 `200`，生成请求几毫秒返回
`500` 而不挂住。

### 6.2 35B：就绪 4.8 s，吞吐 6.34–8.50 tok/s

就绪时间（同一份拓扑，两次启动）：

| 段 | 实测 |
| --- | ---: |
| 尾段（Y700，`32:40`） | **4.8 s**（另一次 7.0 s） |
| 链首（本机，`0:32`） | **11.7 s**（另一次 13.6 s） |

尾段的就绪行形如 `[split-stage] ready: layers [32, 40) of 40, role=tail, n_embd=2048, n_rs_seq=8`。

生成吞吐（贪心，`temperature 0`，同一份 35B 整模）：

| 生成 | 吞吐 | 计时口径 |
| --- | ---: | --- |
| 48 token | **6.34 tok/s** | `prompt_ms` 3023.88，`predicted_ms` 7418.48 |
| 96 token | **6.79 tok/s** | `prompt_n` 27 / `prompt_ms` 3605.68；`predicted_n` 96 / `predicted_ms` 13990.18 |
| 26 token | **8.50 tok/s** | `prompt_n` 24；这一发另带了 `enable_thinking: false`（§6.4） |

表里的 tok/s 取 llama.cpp 报的 `predicted_per_second`，与 `predicted_n ÷ predicted_ms` 差约 2%（分母口径不同，前者含最后一步的采样与收尾），只影响末位。

**35B 的尾段 `32:40`（8 层）压在这块平板上量到 6.3–8.5 tok/s，三次不是单调关系**：96 token（6.79）高于 48 token（6.34），26 token（8.50）最
高。差值能拆开：26 token 那一发是**同一进程的第二个请求**（`graphs reused = 118`，图已建），prompt eval 是 1905.98 ms / 24 token，而第一个请
求是 3605.68 ms / 27 token——**冷启动的图构建摊在第一个请求里**，拿它当基线会低估。所以这三次**不能当「生成越短越快」的曲线读**；要一条吞吐曲线，得在同一时段交替跑、并且丢掉每个
进程的第一个请求。

### 6.3 破坏性验收：五步全过

`status` 作基线，其后五步都打在这一条链上：

| 步骤 | 实测 |
| --- | --- |
| `status`（基线） | `2/2 段 ok` |
| 杀掉 Y700 上的尾段 | `status` 报 `DEAD`、退出码 1；日志最后一行是 `[split-stage] OP_STATE_SAVE …`，没有退出原因 |
| 浅 `/health` | `200 {"status":"ok"}`，13.7 ms |
| `/health?deep=1` | `503`，1.52 s，点名 `remote stage 1 (<tailscale-ip-y700>:9482)`，报文 `failed to remove seq 0 positions [0, -1): recv failed` |
| 生成请求 | `500`，**2.6 ms** 返回，不挂住 |
| `down` 之后 | 两端干净，9481 端口 connection refused |

**尾段是被外部结束，不是加载失败**：日志停在 `OP_STATE_SAVE` 而没有退出原因。处置与[`DEPLOYMENT.md`](DEPLOYMENT.md) §4 同一口径——重起那一段，不要重起
整链。

### 6.4 吞吐验收的坑：thinking 模型吃掉 token 预算

Qwen3.6-35B-A3B 是 thinking 模型。`max_tokens` 给到 96 时，96 个 token 全落在 `reasoning_content` 里，`content` 是空字符串且
`finish_reason=length`。**吞吐验收要么给足 token 预算，要么发 `chat_template_kwargs: {"enable_thinking": false}`**——否则量到
的是一段没有正文的生成，失效的是验收方法，与链无关。

### 6.5 Termux 侧的坑：`LD_LIBRARY_PATH`、看不见的进程、长命进程

三条都不是链的问题，是安卓 / Termux 的部署形状：

① **`LD_LIBRARY_PATH` 没有槽位**——二进制要它才能找到同目录的 `libllama.so` / `libggml*.so`，而拓扑没有环境变量字段，解法是 wrapper 脚本 +
`bin_dir` 指向 wrapper 目录。

② **`ps` / `pgrep` 看不见别的进程**——`ps -ef` 只列自己那条 ssh 会话的进程，`pgrep -af split-stage` 在尾段明明活着时返回空，所以「那段还在吗」只能问
`kill -0 <pid>`（控制面正是这么探的）或从端口、协议层问，**别用 `pgrep` 做验收**——它给的是假的「进程没了」。

③ **长命进程要 `setsid nohup … > log 2>&1 < /dev/null &`**，否则 ssh 会话一结束进程就死。

①②的完整版（含 `pkill -f "binwrap/split-stage"` 杀不到这个副作用）见
[`ANDROID-SURFACE-段接入评估.md`](ANDROID-SURFACE-段接入评估.md) §1.4，「模型必须放 Termux 私有目录」那条见同文档 §1.3。

### 6.6 DSH 崩溃重启：远端尾段活着，链首死了

本机调度进程（DSH）崩溃重启后，**本机链首跟着死了、Y700 上的尾段还活着**，`status` 如实报 `1/2 段 ok` + head `DEAD`。远端段不受本机父进程生命周期影响，两面都成立：好
处是链首重启不用重传、不用重启尾段；代价是留下一个占着约 4.90 GiB 内存的孤儿尾段，必须用 `down` 或 `kill -0` 核对后清掉。

## 7. 两段与三段跨机：投机收益与两跳代价

### 7.1 两段（本机 head `0:32` + Y700 tail `32:40`）

口径：官方 35B、k=2、T=0、**1522 token** prompt、64 token 生成，两轮：

| 配置 | 第 1 轮 | 第 2 轮 | accept（两轮逐位相同） | mean len |
| --- | ---: | ---: | ---: | ---: |
| 不投机 | 7.42 | 8.71 | — | — |
| DFlash k=2 | 7.41 | 8.29 | 0.69231（36/52） | 2.38 |
| **DSpark k=2** | **8.70** | **9.79** | 0.74000（37/50） | 2.48 |

DSpark **+12.4% ~ +17.2%**、DFlash 持平到 −4.8%。差别有机制解释：DFlash 的 8 个 extract layer 里**2 个在 head**（
`2 of 8 DFlash taps cross the wire`），DSpark 的 5 个里只有 **1 个**（`1 of 5`），所以 DSpark 的 tap 过网更少。**短 prompt 上
不划算**：draft ctx 的固定建仓约 2.1 s，摊在 64 token 上就是 −25%（39–46 token 那批量到 DFlash 5.62 tok/s，而不投机是 8.63）。

**口径标注（两批，别混用）**：本批是 **1522-token** 口径（DSpark accept **0.740**，37/50）；**941-token** 口径见
[`MODEL-SWAP.md`](../measure/MODEL-SWAP.md) §11.8（accept **0.932**，+51%，**历史值**）。两批不可互相换算——prompt 长度
变了，接受率跟着变。按 [`INDEX.md`](../measure/INDEX.md) 第二节，现行值取本批的 **+12~17%**。

### 7.2 三段（本机 head + Surface relay + Y700 tail）

拓扑：本机 `0:20` + Surface `20:28`（`role=relay`）+ Y700 `28:40`（`role=tail`），三段同一份整模。ready 时间（两次）：tail **10.4
/ 10.5 s**、relay **9.9 / 15.9 s**、head **4.8 / 6.0 s**。

**两跳的代价**：1522 token prompt + 64 token 生成，三段两次 **6.53 / 6.94 tok/s**（prefill 49.8 / 52.9 s、decode 9.64 /
9.07 s，日志 `.qlh_tmp/3seg_throughput.log`）；同 prompt 的两段链不投机是 7.42 / 8.71 tok/s（§7.1），**慢 12–20%**（两轮分别 −12.0% / −20.3%）。结构原因在
`llama_context::decode_split()`：它对 `split_remotes` **逐个串行**调用、head 是中心，所以三段链的 hidden 过网 **4 次**（两段 2 次）、两
个来回串行，checkpoint 每段一次往返（v6 起只传 8 字节 handle，`SEGMENTED-LIMITS.md` §2.5）。**三段只在单段装不下时才有意义。**

**三套切分不是同一拓扑**：本批是 `0:20` / `20:28` / `28:40`（1522 token）；[`MODEL-SWAP.md`](../measure/MODEL-SWAP.md) §11.8 的
**9.09 t/s** 是 `0:32` / `32:36` / `36:40`（941 token、`-t 14`）；2026-11-12 又测了第三套 `0:24` / `24:32` / `32:40`（`-t 8`），
得 **5.85 t/s**，那是 [`INDEX.md`](../measure/INDEX.md) 第一节的现行值。三者不可互相换算。

**两种死法都点名到段**（`?deep=1` 的 `chain` 字段按链序列出两段，`status` 两次都是 `2/3 段 ok`、退出码 1）：

| 杀哪段 | 浅 `/health` | `?deep=1` | 生成 |
| --- | --- | --- | --- |
| 中间段（Surface relay） | `200` | `503` 点名 `remote stage 1 (<tailscale-ip-surface>:9482)` | `500` / 4.2 ms |
| 尾段（Y700 tail） | `200` | `503` 点名 `remote stage 2 (<tailscale-ip-y700>:9483)` | `500` / 5.5 ms |

**一个验收方法上的坑**：Windows 远端段是 `cmd.exe` 外壳 + `split-stage.exe` 子进程，`status` 报的是**壳的 pid**。只
`taskkill /F /PID <壳>` 会得到「status 报 DEAD、链却还在答」——要么杀 `split-stage.exe` 本身，要么 `/T` 杀树。
