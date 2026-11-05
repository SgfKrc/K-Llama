# 分段链实测数据

> 状态：**现行**
> 更新日期：2026-11-06
> 适用范围：`H:\llamacpp-c`（HEAD `f563a3c`）的 `build-split`，2B 与 35B 两条分段链上这一周的
> 全部数字。读者是调参、写文档、以及下一次要判断「这个数还成不成立」的人。
> 相关：[`SEGMENTED-LIMITS.md`](SEGMENTED-LIMITS.md)（这些边界）、[`SPECULATION.md`](SPECULATION.md)
> （投机判定）、[`PATCHES.md`](PATCHES.md)（第四代实现）。

这份文档存在的理由：本周的性能数字一度只存在于某次对话里，`SEGMENTED-LIMITS.md` 引用了它们
却没有出处。下面每一条都注明日志文件与时间戳；**日志里找不到的，一律进 §7 标成未复现，不在这里
编一个。**

**结论先行**：

- checkpoint 覆盖全链之后，3103 token 的历史第二轮只算 38 个新 token：**714 ms**，而整段重算是
  **19847 ms**。
- tail 的 logits 缓冲从 **1710.76 MiB** 降到 **3.79 MiB**，同一 ubatch 的 prefill 顺带从
  172.6 涨到 219.4 t/s。
- MTP 的收益完全由接受率决定：本仓 2B 上只有退化 prompt 能到 1.3×，真实代码/中文 prompt 上
  **k=2 反而慢 13%–37%**（§2）。
- DFlash k=3 在 35B 链上是唯一稳定的正收益：**1.12×–1.24×**（§3）；k=15 亏（0.46×）。

---

## 0. 口径

| 项 | 口径 |
| --- | --- |
| 硬件 | 本机 i9-13900H + RTX 4060 Laptop；`build-split` 是**纯 CPU** 构建，启动即打印 `warning: no usable GPU found`，全部数字 `-ngl 0` |
| 2B 链 | `qwen35-2b-Q4_K_M`，head `0:20` + tail `20:24`（24 层），本地双段（进程内 tail） |
| 35B 链 | `Qwen3.6-35B-A3B-Claude-4.7-Opus-Reasoning-Distilled-APEX-MTP-I-Quality`（qwen35moe，40 层 trunk + 1 个 nextn 层），head `0:32` + tail `32:40` |
| 采样 | 贪心（`temperature 0`、`top_k 1`） |
| `-np` | 除注明外都是 1 |
| 逐位对比 | 只在**单 ubatch 内**成立；跨 ubatch 的差异是浮点归约序，见 [`SEGMENTED-LIMITS.md`](SEGMENTED-LIMITS.md) §3.1 |
| 日志根 | `H:\llamacpp-c\build-split\` 与 `H:\llamacpp-c\.qlh_tmp\`（下表只写相对路径） |

### 0.1 最要紧的一条口径：**本机双进程测不出投机收益**

§2/§3 所有投机数字都是「head 与 tail 跑在同一台机器上」。这个配置对投机是**退化**的：

- **它拿不到投机唯一该拿的好处**。投机减少的是**每轮往返次数**（一轮前进 3.2 token
  而不是 1），而在本机双进程里 head↔tail 走内存与管道，**没有 25–30 ms 那个跨机往返**。
  付出 draft 成本、却省下一个不存在的量。
- **它放大了投机的代价**。两个进程抢同一套 CPU 核心、L3 与内存带宽，而 draft（MTP 的
  nextn 层 / DFlash 的 6 层 dense）是纯额外算力，全压在已经很挤的 tail 那台「机器」上。

所以本机测出的「亏」是**对一个不会部署的配置**的测量。多机场景的账要反过来算：省下的是
`(N − N/3.2) × 25–30 ms` 的往返，而多花的是 tail 那台机器上一次 draft 前向——后者与前者
差一个量级。**在真实多机上投机大概率是划算的，这一点没有测过，不能用本文的数字反驳。**

反过来，checkpoint 复用（§1）、并发（§4）、内存（§5）这些不受这条影响：它们的收益不来自
跨机往返。

---

## 1. checkpoint 复用（远程链）

远程链 = 本机 head + 两个 `split-stage` 进程（`:9331` relay、`:9332` tail），`-c 8192`、`-np 1`。

| 项 | 值 | 出处 |
| --- | --- | --- |
| 3103 token 历史的**整段** prompt eval | **19847.34 ms**（19.85 s，156.3 t/s） | `build-split/reuselong_9412.log` 0.21.859 |
| 同一 fit 的**第二轮**（历史 3119 + 新 38） | prompt eval **714.02 ms**，total 1689.66 ms | 同上 0.23.563 |
| 另一轮同配置 | 整段 15890.42 ms（195.3 t/s）→ 第二轮 **603.73 ms** | `build-split/reuselong_9404.log` 0.17.781 / 0.19.228 |
| 链路失效时的样子（修复前） | `seq_rm: remote stage 1 failed … send failed`；`state_seq_set_data: … failed the state load` | `build-split/kill_ckpt_9415.log` 0.03.615、`kill_ckpt_9416.log` 0.11.175 |
| 本地链上下文 checkpoint 大小 | **19.266 MiB** @1806 token（`checkpoint 1 of 32`） | `.qlh_tmp/issue1_after_final.log`、`issue1_before_ub2048.log` |
| 远程段 state blob（2B 的 8 层段、`--rs-seq 8`） | **6,733,996 B = 6.42 MiB**，与历史长度无关（`pos_max` 17 与 59 同值） | `build-split/stage{816,1624}.log` 的 `OP_STATE_SAVE` |

**19.85 s 与 0.71 s 是同一份日志里的两件事，不是一次 A/B**：前者是「3103 token 整段重算」的成本，
后者是「只算 38 个新 token」的成本。修复前的失败形态在 `kill_ckpt_9415/9416` 里是**报错**，不是
「慢 19.85 s」——链首恢复了、尾段停在旧位置，下一次截断式 `seq_rm` 在尾段失败。日志里没有
「修复前第二轮 19.85 s」这一行（§7）。

---

## 2. MTP（2B 链，`-np 1`，`n_predict 192`，三次取中位）

### 2.1 退化 prompt（5 token，输出是复读）

| 配置 | ms/token（3 次） | 中位 | 相对 off | 接受率 |
| --- | --- | ---: | ---: | ---: |
| off | 36.84 / 37.36 / 37.32 | **37.32** | 1.00× | — |
| k=1 | 29.54 / 30.96 / 31.24 | **30.96** | 1.21× | 98.96%（95/96） |
| k=2 | 26.26 / 28.32 / 29.44 | **28.32** | **1.32×** | 97.67%（126/129） |
| k=3 | 23.18 / 25.09 / 25.50 | **25.09** | **1.49×** | 96.60%（142/147） |

出处：`build-split/mtp_time/np1_chain_{off,k1,k2,k3}_*.json`（01:50–01:51）、
`mtp_time/srv2_chain_{k1,k2,k3}.log` 的 `draft acceptance`。prompt 是 `The capital of France is`，
模型只会复读 `The capital of France is Paris.`，**接受率 97% 是这句话的性质，不是 MTP 的性质**。

### 2.2 三个真实负载（`-np 1`，三个 prompt 各三次，今日 06:44 重测）

| 负载 | prompt token | off | k=2 | 比值 | 接受率 |
| --- | ---: | ---: | ---: | ---: | ---: |
| 代码续写（`def quicksort` 上下文） | 131 | 35.76 | **41.30** | **0.87×** | 46.9%（92/196） |
| 中英混合（Python + 中文提问） | 95 | 35.74 | **51.69** | **0.69×** | 27.0%（67/248） |
| 纯中文（续写北京介绍） | 23 | 35.78 | **56.86** | **0.63×** | 19.4%（53/273） |

出处：`.qlh_tmp/mtp_remeasure_out.log`（脚本 `.qlh_tmp/mtp_remeasure.sh`）、服务日志
`.qlh_tmp/mtp_re_{off,k2}_064306.srv.log`。复现命令：`bash .qlh_tmp/mtp_remeasure.sh`。

**这张表是本节最重要的一条。** 把 2.1 与 2.2 两组按「每轮成本」折算，同一副链上 k=2 的一轮
实际花 **2.21–2.24 个 plain step**（用 `mean len × ms/token ÷ off 的 ms/token` 算：83.3/37.3、
80.1/35.8、79.6/35.7、79.0/35.8），这与 [`SPECULATION.md`](SPECULATION.md) §0 的成本模型
（k=2 是 4.36 ÷ 2 = 2.18 个单位）吻合。于是回本线是 `mean len > 2.22`，即**每轮接受 1.22 个草稿
（接受率 61%）以上**；三个真实负载的 `mean len` 是 1.94 / 1.54 / 1.39，全在亏的那一侧，
而 2.1 的复读 prompt 是 2.94。**1.3× 不是 MTP 的性质，是接受率的性质**——2B 上别按 1.3× 规划。

### 2.3 `-np > 1` 与 MTP

| 配置 | np=1 | np=4 |
| --- | ---: | ---: |
| 链 + MTP k=2 | **28.33** | **34.55**（1.22× 慢） |
| 整模 + MTP k=2 | **22.56** | **28.37**（1.26× 慢） |
| 整模不投机（对照） | 32.09 | — |

出处：`build-split/mtp_time/srv_np1_spec.log`、`srv_np4_spec.log`（链）；`build-split/srv_w1_m.log`、
`srv_w4_m.log`（整模，`n_slots` 1→4）；`mtp_time/whole_off_*.json`（对照行，**`predicted_n=128`**，
而 MTP 各行是 192 token，两次长度不同，只能当量级参考）。**`-np 4` 的三次请求仍是逐个发的**
（日志里没有重叠的 batch），所以变慢不是排队，而是 `n_seq_max` 带大的缓冲——与
[`SEGMENTED-LIMITS.md`](SEGMENTED-LIMITS.md) §3.2 的猜测一致，但机理仍未定论。

---

## 3. DFlash（35B 链，CPU）

四个 prompt（两段代码、一段解释、一句中文），`n_predict 160`，`-np 1`，`-c 4096`，
每个配置各跑一遍（`.qlh_tmp/casc_probe.sh`、`casc_base.sh`）：

| 配置 | ms/token（4 prompt） | 均值 | 相对不投机 | 整块接受率 |
| --- | --- | ---: | ---: | ---: |
| 不投机 | 74.79 / 68.35 / 71.43 / 74.57 | 72.28 | 1.00× | — |
| MTP k=2 | 74.97 / 74.19 / 61.50 / 74.59 | 71.31 | 1.01× | 56.4%–80.3% |
| DFlash k=3 | 59.43 / 67.38 / 52.25 / 78.12 | **64.30** | **1.12×** | 49.7%–76.4% |
| DFlash k=15 | 150.90 / 197.41 / 118.36 / 155.40 | 155.52 | 0.46× | 12.6%–26.2% |

出处：`.qlh_tmp/casc/{run,run_base}.log`（逐请求 tok/s 与 `draft_n/accepted`），服务日志
`.qlh_tmp/casc/srv_{base,mtp2,k3,k15}.log` 的 `eval time`。

单 prompt 对照（`n_max=3`，两条请求的 `predicted_n` 不同——402 vs 512，投机那条提前停了，
所以比的是 ms/token 而不是总时长）：链上 **59.96 / 61.66** vs 不投机 **76.72 / 74.18**
→ **1.24×**；整模 + DFlash k=3 是 63.96 / 61.93（`build-split/dflash/speed_{chain,chain_nospec,whole}.log`，
03:19–03:20）。`draft acceptance = 0.71916 (274/381)`（`dflash/speed_chain.log`），链与整模逐 token
相同 3/3（`dflash/accept.log`）。

8 路 tap 的逐位结论（`dflash/taps_final.log`，单 ubatch，15 token）：8/8 与整模相同；
tail 的 local 1 = 整模 33、local 6 = 整模 38；logits 15/15 行逐位相同。

---

## 4. 并发与多 slot

| 项 | np=2 | np=4 | 出处 |
| --- | ---: | ---: | --- |
| 10 个不同长度请求 | 10/10 200，存活 | 10/10 200，存活 | `build-split/multislot_{np2,np4}.out` |
| prefix cache 复用（round1/round0） | 0.115（408.1→47.0 ms） | 0.145（406.6→59.0 ms） | 同上（15:12 重跑覆盖；早先两轮是 0.100 / 0.117） |
| 串行/并发输出逐字节一致 | PASS | — | `.qlh_tmp/accept_np.py --np 2` |
| tail 的 logits 缓冲按「实际要读的行」算 | 1 行 → 0.95 MiB（`n_outputs_max=1`，`build-split/stage_tail.log`） | 4 行 → 3.79 MiB（`n_outputs_max=4`，`.qlh_tmp/issue1_after_final.log`） | 缓冲 = `4 B × n_vocab × 行数`，行数 = `n_seq_max × (1+n_max)` |

链上 `-np` 的代价见 §2.3。`verify_multislot.py` 的两项（np 2 / np 4）在 `verify.sh` 组 5。

---

## 5. 内存

同一个 1806 token 的 ubatch（`-c 8192 -np 4`，2B 链，`-b 2048`）：

| 项 | 修前 | 修后 | 出处 |
| --- | ---: | ---: | --- |
| tail 的 logits 缓冲 | **1710.76 MiB**（`n_outputs_max=1806`） | **3.79 MiB**（`n_outputs_max=4`） | `issue1_before_ub2048.log:533` / `issue1_after_final.log:459` |
| head 的 hidden 缓冲（`logits=no, embd=yes`） | 14.11 MiB | 14.11 MiB | 两 log 的 532 行 |
| 1806 token 的那次 prefill（1810 含采样行） | 172.61 t/s（10485.85 ms） | **219.39 t/s**（8250.22 ms） | 同上 |
| 上下文 checkpoint（1806 token） | — | 19.266 MiB | `create_check` |
| 远程段 state blob（§1 的三段远程链，2B 的 8 层段、`--rs-seq 8`） | — | 6.42 MiB | §1 |

**读法**：修前 tail 按 `n_batch` 行分配 logits，修后只为「实际要被读的行」分配
（`n_seq_max × (1+n_max)`）。省下来的不只是内存——同一 ubatch 的 prefill 顺带快了 27%，
因为不再申请/清零那 1.7 GiB。

远程 stage 那边同一笔账更大：`split-stage` 只做 tail、`-b 2048`、248320 词表时，`sched_reserve`
的 pp 图缓冲（`graph (pp bs=2048, tg bs=1)`）从 **2052.10 MiB** 降到 **240.09 MiB**。口径：
`--role tail`、2B 的 20:24 段、`-c 4096 -b 2048`、`n_seq_max=1`；两次运行其余参数相同，只差
logits 行数 `n_outputs_max`（2048 → 1）。出处 `build-split/i2_tail_before.log:553`（2048 行）与
`i2_tail_after.log:553`（1 行）。

---

## 6. 验收覆盖

| 入口 | 覆盖 | 本次结果（2026-11-06，增量构建后） |
| --- | --- | --- |
| `bash build-split/verify.sh` | 九组 **25 项**（含一次三目标构建检查；只用 `--skip-build` 跑时是 24 项）：本地链逐位、N=1 直通、server 双段、跨进程、多 slot、三段链、远程 server、链上 MTP、DFlash 8 tap | **PASS=25 FAIL=0** |
| `python build-split/verify_multislot.py --np 2` | 组 5 的一半 | 10/10 200，prefix cache 生效 |
| `python build-split/verify_multislot.py --np 4` | 组 5 的另一半 | 10/10 200，prefix cache 生效 |
| `.qlh_tmp/accept_np.py --np 2` | 串行 vs 并发四个输出的逐字节一致 | **ALL PASS**（并发墙钟 1.6 s） |
| `bash .qlh_tmp/mtp_remeasure.sh` | §2.2 | 三个负载 ×（off / k=2）× 3 次，见 §2.2 |

`verify.sh` 的判据全部是**逐位/逐字节**（`split-smoke`、`split-rows`、`dflash-taps` 三个目标），
不看生成文字；组 9 需要 35B + DFlash 权重。上表第一行的 `PASS=25 FAIL=0` 有两份完整记录：
`.qlh_tmp/verify_after_comments.log`（06:52）与 `build-split/verify_run1.log`（10:37）；
`--skip-build` 那 24 项的同一条结论见 `build-split/verify_final_hello.log`（15:15）。

---

## 7. 未落盘 / 未复现

| 数字 | 谁在引用 | 现状 |
| --- | --- | --- |
| MTP k=2 **1.33× / 1.06× / 1.00×**（代码 / 混合 / 中文），口径见 §2 | [`SEGMENTED-LIMITS.md`](SEGMENTED-LIMITS.md) §3.2、`ARCHITECTURE.md` | 日志里没有这三个数。本日重测是 **0.87× / 0.69× / 0.63×**；1.3× 只出现在第 2.1 节那种复读 prompt 上（1.32×）。**需在真实负载上重测，不能继续当结论用** |
| DFlash k=3 **+23.7%** | [`SEGMENTED-LIMITS.md`](SEGMENTED-LIMITS.md) §3.2 | 日志复算是 **1.24×（+24.1%）**，差 0.4pp 在复现噪声内 |
| 级联整块全接受 **42.5% / 2.8%**（k=3 / k=15） | [`SEGMENTED-LIMITS.md`](SEGMENTED-LIMITS.md) §2.3、[`SPECULATION-MATRIX.md`](SPECULATION-MATRIX.md) §4 | 42.5% / 2.8% 就是从 `.qlh_tmp/casc/srv_{k3,k15}.log` 数出来的：`3/3` 96/226 轮、`15/15` 5/177 轮。把 `1/1`、`2/2`、`4/4` 这类块尾也算成全接受是 **43.4%（98/226）/ 4.5%（8/177；复算到 9 轮，多出的一轮是 `4/4`）**。两个口径不完全可比（那边数的是 Python 驱动的级联路径），且 k=15 只有 5–9 轮全接受，比例本身噪声很大 |
| checkpoint **19.85 s → 0.71 s**，口径见 §1 | [`PATCHES.md`](PATCHES.md) 第四代、[`SEGMENTED-LIMITS.md`](SEGMENTED-LIMITS.md) §1 | 两个数各自成立，但「修前第二轮 19.85 s」没有日志：19.85 s 是整段重算的成本，修前实测到的是**失败**（`kill_ckpt_9415/9416`） |
| **3137 token** 历史 | 同上 | 日志是 **3103**（+16 生成 = 3119）；3137 在任何日志里都找不到 |
| 2B 上 k=3 与 k=2 的 **25.09 vs 25.87** | [`SEGMENTED-LIMITS.md`](SEGMENTED-LIMITS.md) §3.2 | k=3 中位 25.09 对得上（第 2.1 节）；k=2 中位是 **28.32**，25.87 未复现 |
| 35B MTP k=2 **1.13–1.25×**，本周对照见 §3 | [`SPECULATION.md`](SPECULATION.md) §1 | 那份是 10-31 的草稿成本模型 + 单 prompt；本周链上端到端（4 prompt）是 **1.01×**。两个口径都要保留，不能互相替代 |
| v5 候选通道的等价性 / 带宽 / 延迟（§8） | [`SEGMENTED-LIMITS.md`](SEGMENTED-LIMITS.md) §2.2、`README.md` 边界③ | 只在 fork 的提交说明里，**`build-split/` 日志中没有这些行**；按「未落盘到日志」对待 |

另有**两项能力做了但没有端到端证据**。它们不是数字，所以不进上表：

- **多模态全维位置**：`split_expand_pos()` 与 `req_header::n_pos` 让 4 个 M-RoPE section 跨段
  传下去，验收组 6 的三段跨进程在 `pos = [p,0,0,0]` 上逐位通过。但**没有图像输入**的端到端验证——本机
  `H:/qlh_models/mmproj` 有投影模型，分段链 + mtmd 的组合从未跑过。所以「图像位置能传」目前
  是代码级结论，不是实测结论。
- **本地纯 KV 链的 ctx_shift**：`can_shift=false` 对 qwen35/qwen35moe 是忠实的（整模同样
  不支持），但对非 MROPE 的本地链属于过度保守，从未验证过打开它是否安全。`seq_add` 是
  void、无返回值，要做到「逐段都真的移了同样量」只能靠 shift 后断言各段 `seq_pos_max`。

---

## 8. 远程 tail 的候选通道（协议 v5）

**这一节的数字只出现在 fork 的提交说明里（`f563a3c` 第十二节），不在 `build-split/` 的日志里。**
所以它们按「未落盘到日志」对待：机制与数据可信到「提交说明」这一层，独立复算没做过。

口径：248320 词表的分段链、单序列；采样参数逐项注明在下面，fuzz 的 10 组随机参数是
「把整条链随机化」而不是只换 `top_p`。

- **为什么等价**：`common_sampler` 的链顺序是 `penalties → dry → top_n_sigma → top_k →
  typ_p → top_p → min_p → xtc → temp`，**`top_k` 之后的所有步骤只在 k 个候选内作用**。
  所以 tail 本地跑完 `logit_bias → penalties → top_n_sigma → top_k`，回传 40 个 `(id, logit)`，
  server 从 `cand_first`（`top_k` 在链里的下一个节点）接着跑，与回传整行**数学等价**。
  回退是白名单预判（`common_sampler_candidates_check()`，每请求一次），不是运行时兜底；
  条件见 [`SEGMENTED-LIMITS.md`](SEGMENTED-LIMITS.md) §2.2。
- **惩罚历史按帧重放**：server 每帧带最近 `min(penalty_last_n, n_prev)` 个 token，tail 从零
  `reset` + 重放。代价 256 B/帧；换来的是重连、`seq_rm`、`state_restore`、slot 复用之后
  两端自动一致（增量式一旦两端对「哪些 token 被 accept」有分歧就会静默错）。
- **帧大小**：`rsp_header` 32 B + 40 × `token_cand` 8 B = **352 B/token**（两个 `static_assert`
  锁了 32 与 8 的尺寸），对整行 **993 KB** 是 2821 倍。
- **等价性实测**：4 prompt × 64 token、penalties 三参、`top_n_sigma`、xtc、temp=0、`top_k=1`、
  `logit_bias`、ignore_eos、post-sampling logprobs、cache 复用两轮、400 token 长生成、
  10 组随机参数 fuzz —— **全部逐 token 相同**。
- **带宽**（TCP 代理计数）：同一 64-token 请求，整行 67,932,452 B vs 候选 4,383,012 B →
  每 token 省 992,960 B。
- **loopback 延迟**：57.92 → 52.92 ms/token（+8.6%）。loopback 等效 ~200 MB/s，所以只值 5 ms；
  按跨机 LAN 的 24 ms/0.95 MiB 折算，跨机 decode 从 4.67 / 5.38 / 6.05 预计到
  **6.6 / 8.1 / 9.7 tok/s**（整模的 66–87%，原 45–54%）——**这是折算，不是实测**。
- **没做**：prefill 阶段的 tail 仍整行回传（只有「将被采样的那一行」才请求候选），prompt
  处理的带宽未优化。
