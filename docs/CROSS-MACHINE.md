# 跨机实测：ROG、Surface、回环与 Tailscale

> 状态：**现行**
> 更新日期：2026-11-09
> 适用范围：远程链投机在三种真实部署形态上的收益（LAN 强机、弱机、回环），以及 Tailscale →
> Surface 的首次真跨机零损失实测。逐 prompt 与 k 扫描归 [`SPECULATION-DATA.md`](SPECULATION-DATA.md)，
> MTP 链上折扣的根因归 [`MODEL-SWAP.md`](MODEL-SWAP.md) §11.4，协议与传输量归
> [`PROTOCOL-V9.md`](PROTOCOL-V9.md)。

口径：35B（`qwen35moe`，40 层 trunk + 1 个 NextN 层），head `0:32` + tail `32:40`，贪心， 32-token 生成。数据来自
ROG（LAN）、Surface（4C8T 弱机器）与本机回环；`build-split/` 里没有与之
对应的日志文件，按「可信到实测记录这一层、独立复算没做过」对待。**这一整节测在旧蒸馏版上**
（官方模型到位之前），官方版接受率高 30–76%，收益预计更高但未重测（见 [`MODEL-SWAP.md`](MODEL-SWAP.md) §11.7）。

## 1. ROG（LAN）：五类 prompt × 三配置

| prompt（token） | 不投机 | ngram | MTP k=2 | MTP 变化 |
|---|---:|---:|---:|---:|
| 散文 882 | 8.18 | 7.33 | 8.47 | +3.6% |
| 代码 1205 | 8.74 | 6.92 | 9.47 | **+8.4%** |
| JSON 2379 | 9.47 | 7.09 | 9.36 | **−1.2%** |
| agent 2508 | 7.51 | 5.09 | 7.62 | +1.5%（当时无 draft） |
| 人为重复 `A+B+A+B+A` 89 | 9.73 | 7.75 | 10.75 | +10.5% |

**中位约 +6%，区间 +8.4% ~ −1.2%。这就是跨机投机的真实量级，不是 +15%。** 历史那个 +15% （10.20 vs 8.88 tok/s）来自一个
27-token 的高确定任务：同一任务上 k=1 是 10.11、k=2 是 10.20、 k=3 是 9.76——k≥3 的验证帧要 k+1 行，而 **head 有 32
层、tail 只有 8 层，宽帧的代价几乎全在 head**，所以「k=2 是甜点」只在那个切点上成立。**agent 那行 +1.5% 是在「当时无
draft」下测的** （tail 没给出 NextN 草稿），它证明的是「没有 draft 时投机不会变慢太多」，不是 MTP 在 agent 负载上
的上限。**人为重复那行 +10.5% 是送分题**，与 ngram 那列的 7.75 一样，只说明「可复制的 prompt 上投机好用」。ngram
五类全部低于不投机，原因见 [`SPECULATION-DATA.md`](SPECULATION-DATA.md) §4。

## 2. 弱机器与回环：收益全部转负

| 配置 | 不投机 | 投机 k=2 | 相对不投机 |
|---|---:|---:|---:|
| Surface（4C8T，纯 CPU）散文 | 8.65 | 5.21 | **−40%** |
| 本机回环，带 grammar（`toolcall.gbnf`） | 20.82 | 13.95 | **0.67×** |

**四个真实原因**：① **每帧两次 hop**——`OP_MTP_DRAFT` 出草稿、`OP_DECODE` 再验证，一轮走两趟链。 ② **MTP draft 不看
grammar / rbudget**——同一个 tail 上，接受率从无 grammar 时的 **2.76 token/帧** 掉到 **1.46–1.73**。③ **回退率
9–19%**——`toolcall.gbnf` 这种严格语法实测 **11/16 帧被拒**，而每次回退 ≈ 一整个普通 round（撤 KV + 重放 prompt +
sampler 回滚）。④ **回环上 993 KB 只要约 1
ms**——验证通道省掉的四个数量级带宽在回环上本来就不值钱，于是这笔账只剩成本。另有：**FORCING 期间 tail 回的未掩码 argmax
基本每轮都被拒**，有约 **2 倍白跑**。

所以「真实多机上划算」只能写成条件句：**LAN + 强 head + 接受率高的负载**。同一副链换切点、换机器
或换到回环都会翻号。这与「本机双进程测不出投机收益」互为两面：本机的负数不能外推成「多机一定负」，多机的中位 +6%
也不能外推成「任何多机都为正」。

## 3. Tailscale → Surface：第一次真跨机零损失（2026-11-09）

口径：head 本机加载 `0:32`，tail 在 Surface 上 `32:40`，用 `--split-layer-range 0:32 --split-remote
<surface>:<port>`（`--split-remote` 与 `--split-tail-range` 互斥，所以只能这么写）；target 是官方 `UD-Q4_K_M`
（`22,663,387,424 B`，Surface 上这份是这一轮刚传的，与本地大小一致）；k=2、贪心；prompt 1102 token，三个 prompt
各跑一遍，三配置 A / B / C 共用同一份 target 与同一组 prompt。

### 3.1 接受与输出：零损失

| 配置 | 接受 | 与整模逐 prompt 比 |
| --- | --- | --- |
| A DFlash k=2 | **117/141 = 0.82979** | **相同** |
| B DSpark k=2 | **117/140 = 0.83571** | **相同** |
| C MTP k=2（对照） | 108/157 = 0.68790 | 对照（nextn 通道的 0.688 上限，见 [`MODEL-SWAP.md`](MODEL-SWAP.md) §11.4） |

三个 prompt 的输出 **3/3 逐字节 IDENTICAL**（250 / 244 / 249 字符）。这是本项目第一次在**真跨机** 链上拿到「与整模逐
prompt 相同」的接受率——此前只在 loopback 上验过 tap 开销可忽略，跨机验的是 wire 上 tap 逐位与回退后的输出。

### 3.2 prefill 的 tap 代价（1102 token）

| 配置 | prefill 耗时 |
| --- | --- |
| A DFlash | 32784 / 31506 ms → **~32.1 s** |
| B DSpark | 32840 ms → **~32.8 s** |
| C MTP | 27499 / 28212 / 29698 / 30783 ms → **~29.7 s** |

**A − C ≈ +2.4 s**，与「25 MB ÷ 9.5 MB/s ≈ 2.6 s」吻合（[`PROTOCOL-V9.md`](PROTOCOL-V9.md) §3.3： prefill 每个 512 行
ubatch 是 8 MB，1102 token 约 3 个 ubatch 即 24 MB）。**没有 10 s 级的低效**——如果 tap 是逐 token 往返，1102 个 token
会报出数量级更大的差；实测差值与带宽估算同量级，说明 tap 是**成批**走的。与 loopback 对照：1192 token 差 **0.2%**；跨机
1102 token 差 **~2.4 s（7–8%）**。

### 3.3 结论与两个边界

- **accept 零损失**；长 prompt 多花 2–3 s 可接受（数千 token 的 prompt 会到 10 s 级）；收益排序
  **DSpark ≥ DFlash > MTP**；长 prompt 用 DFlash 划算，短 prompt 上 tap 的固定开销占比变大。
- **边界一**：`--split-remote` 与 `--split-tail-range` 互斥，远程链只能用
  `--split-layer-range 0:32 --split-remote host:port`，不能同时指定同机尾段。
- **边界二**：Surface 原本没有这份 21 GB（`22,663,387,424 B`）的 target——只有旧切分片 `6.85G` /
  `2.66G` 与 23.5G 的蒸馏版。体积对不上时跨机输出会直接分叉，所以必须先核对大小。

## 4. 本机双进程为什么测不出投机收益

`SPECULATION-DATA.md` 与 `MODEL-SWAP.md` 里本机测的投机数字都是「head 与 tail 跑在同一台机器
上」，这个配置对投机是退化的：**它拿不到投机唯一该拿的好处**——投机减少的是每轮往返次数（一轮前进 3.2 token 而不是
1），而本机双进程里 head↔tail 走内存与管道，没有 25–30 ms 那个跨机往返，付出 draft
成本却省下一个不存在的量；**它放大了投机的代价**——两个进程抢同一套 CPU 核心、L3 与内存带宽，而 draft（MTP 的 nextn 层 /
DFlash 的 6 层 dense）是纯额外算力，全压在已经很挤的 tail
那台「机器」上。所以本机测出的「亏」是对一个不会部署的配置的测量；它不能反推出「多机一定赚」，
跨机的正确答案是「小幅正收益，随负载翻号」。反过来，checkpoint 复用、并发、内存这些不受这条
影响——它们的收益不来自跨机往返。

## 5. `split-stage` 的线程与 ubatch（Surface 实测）

`GGML_DEFAULT_N_THREADS = 4`，**而且 `n_threads_batch` 也是 4、不会自动跟随**——Surface（4C8T）上原来只用了 4
个线程。现在 `split-stage` 有 `-t/--threads`（同时设两者）、`-tb/--threads-batch` （单独覆盖 batch
那一半）、`-ub/--ubatch-size`（让验证帧在帧内逐行跑，是测量旋钮不是调优旋钮）。

| 配置（Surface） | 原来（4 线程） | `-t 8` | 变化 |
|---|---:|---:|---:|
| 不投机 | 7.41 | 7.67 | +3.5% |
| MTP k=2 | 5.12 | 5.48 | +7% |

**只设 `n_threads` 拿不到这笔收益**，batch 那一半必须一起设；head 的 `-t` 不会过网。
