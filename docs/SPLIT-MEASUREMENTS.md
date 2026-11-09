# 分段链实测数据

> 状态：**现行**
> 更新日期：2026-11-09
> 适用范围：v2.1 分段链（`build-split`，2B 与 35B 两条链）的核心测量摘要与出处索引：checkpoint
> 复用、MTP、DFlash、并发、内存、验收覆盖、协议 v9、选型矩阵与跨机零损失。逐项明细归四份专题：
> [`SPECULATION-DATA.md`](SPECULATION-DATA.md)（逐 prompt、k 扫描、per-position、结构成本）、
> [`MODEL-SWAP.md`](MODEL-SWAP.md)（换模型的接受率基线）、[`PROTOCOL-V9.md`](PROTOCOL-V9.md)
> （协议 v5/v7/v8/v9 与 `seq_rm` 探针）、[`CROSS-MACHINE.md`](CROSS-MACHINE.md)（ROG / Surface /
> 回环 / Tailscale）。

这份文档存在的理由：本周的性能数字一度只存在于某次对话里，`SEGMENTED-LIMITS.md` 引用了它们却没有
出处。下面每节注明日志文件与时间戳；**日志里找不到的，一律进 §7 标成未复现，不编一个。**

**结论先行**：checkpoint 覆盖全链之后，3103 token 的历史第二轮只算 38 个新 token：**714 ms**，整段重算是 **19847
ms**；tail 的 logits 缓冲从 **1710.76 MiB** 降到 **3.79 MiB**，同一 ubatch 的 prefill 从 172.6 涨到 219.4 t/s；MTP
的收益完全由接受率决定，2B 上只有退化 prompt 能到 1.3×，真实代码 / 中文 prompt 上 **k=2 反而慢 13%–37%**；DFlash k=3 在
35B 链上是唯一稳定的正收益（**1.12×–1.24×**），k=15 亏（0.46×）；远程链投机终态是 ROG 五类 prompt 上 **MTP k=2 中位
+6%** （+8.4% ~ −1.2%）、Surface **−40%**、单帧回复从 **993,312 B 降到 48 B**；`seq_rm` 的删尾部在上游 hybrid 默认
`n_rs_seq = 0` 下永远失败（上游 open PR #28007 / #31656），本仓改成失败可见；2026-11-08 换官方 `UD-Q4_K_M`
后形状全部对得上、零改代码，整模 MTP k=2 接受率从 0.699 回到 **0.910**，链上只有 **0.688**（低
24%，设计选择）；2026-11-09 协议 v9 补上远程链的 `layer_inp` tap，DFlash / DSpark
在远程链上**零损失**，真跨机也零损失、prefill 只多 2.4 s；markov 头的收益是「可预测 / 可抄写」，不是「长距离依赖」。

## 0. 口径

| 项 | 口径 |
| --- | --- |
| 硬件 | 本机 i9-13900H + RTX 4060 Laptop；`build-split` 是**纯 CPU** 构建（启动即 `warning: no usable GPU found`），全部数字 `-ngl 0` |
| 2B 链 | `qwen35-2b-Q4_K_M`，head `0:20` + tail `20:24`（24 层），本地双段（进程内 tail） |
| 35B 链 | head `0:32` + tail `32:40`（qwen35moe，40 层 trunk + 1 个 NextN 层）。**2026-11-08 起是官方 `Qwen3.6-35B-A3B-UD-Q4_K_M`**（21.11 GiB）；§3 与 §9 的 35B 数字测在旧蒸馏版上，两份权重不能混用（口径见 [`MODEL-SWAP.md`](MODEL-SWAP.md)） |
| 采样 | 贪心（`temperature 0`、`top_k 1`）；`-np` 除注明外都是 1 |
| 逐位对比 | 只在**单 ubatch 内**成立；跨 ubatch 的差异是浮点归约序（[`SEGMENTED-LIMITS.md`](SEGMENTED-LIMITS.md) §3.1） |
| 日志根 | `H:\llamacpp-c\build-split\` 与 `H:\llamacpp-c\.qlh_tmp\`（下文只写相对路径） |
| 代码版本 | 数字在 HEAD `0c5c3bdd`（树 `982ccecd`）上测；现行 HEAD `48dfb3ea8`（树 `e76601df`）只加段级 offload、MoE 缓存默认与显存预算，不改逐位判据（吞吐的部署含义见 [`SEGMENT-OFFLOAD.md`](SEGMENT-OFFLOAD.md)） |

**本机双进程测不出投机收益**：§2/§3 的投机数字都是「head 与 tail 跑在同一台机器上」——拿不到投机唯一该拿的好处（没有
25–30 ms 跨机往返可省），却放大了代价（两进程抢同一套 CPU、L3 与带宽）。它只
说明本机的负数不能外推；跨机的正确答案是「小幅正收益，随负载翻号」（[`CROSS-MACHINE.md`](CROSS-MACHINE.md) §4）。

## 1. checkpoint 复用（远程链）

远程链 = 本机 head + 两个 `split-stage` 进程（`:9331` relay、`:9332` tail），`-c 8192`、`-np 1`。

| 项 | 值 | 出处 |
| --- | --- | --- |
| 3103 token 历史的整段 prompt eval | **19847.34 ms**（19.85 s，156.3 t/s）；另一轮同配置 15890.42 ms（195.3 t/s） | `build-split/reuselong_9412.log` 0.21.859；`reuselong_9404.log` 0.17.781 |
| 同一 fit 的第二轮（历史 3119 + 新 38） | prompt eval **714.02 ms**，total 1689.66 ms；另一轮 **603.73 ms** | `reuselong_9412.log` 0.23.563；`reuselong_9404.log` 0.19.228 |
| 链路失效时的样子（修复前） | `seq_rm: remote stage 1 failed … send failed`；`state_seq_set_data: … failed the state load` | `kill_ckpt_9415.log` 0.03.615、`kill_ckpt_9416.log` 0.11.175 |
| 本地链上下文 checkpoint 大小 | **19.266 MiB** @1806 token（`checkpoint 1 of 32`） | `.qlh_tmp/issue1_after_final.log`、`issue1_before_ub2048.log` |
| 远程段 state blob（2B 的 8 层段、`--rs-seq 8`） | **6,733,996 B = 6.42 MiB**，与历史长度无关（`pos_max` 17 与 59 同值） | `stage{816,1624}.log` 的 `OP_STATE_SAVE` |

**19.85 s 与 0.71 s 是同一份日志里的两件事，不是一次 A/B**：前者是 3103 token 整段重算的成本，后者是只算 38 个新
token。修复前的失败形态在 `kill_ckpt_9415/9416` 里是**报错**，不是「慢 19.85
s」——链首恢复了、尾段停在旧位置，下一次截断式 `seq_rm` 在尾段失败。35B 段的 12.56 MiB 与盈亏平衡见
[`SEGMENTED-LIMITS.md`](SEGMENTED-LIMITS.md) §2.5。

## 2. MTP（2B 链）

退化 prompt（5 token 复读，`The capital of France is`）：off **37.32** / k=1 **30.96**（1.21×， 98.96%）/ k=2
**28.32**（**1.32×**，97.67%）/ k=3 **25.09**（**1.49×**，96.60%），ms/token 中位。 **97% 是这句话的性质，不是 MTP
的性质。** 三个真实负载：代码续写 131 token 35.76 → **41.30** （**0.87×**，46.9%）；中英混合 95 token 35.74 →
**51.69**（**0.69×**，27.0%）；纯中文 23 token 35.78 → **56.86**（**0.63×**，19.4%）。回本线是 `mean len >
2.22`（每轮接受 1.22 个草稿、接受率 61%），三个负载的 `mean len` 是 1.94 / 1.54 / 1.39，全在亏的一侧；复读 prompt 是
2.94。**1.3× 不是 MTP 的性质，是接受率的性质。** `-np > 1`：链 + MTP k=2 28.33 → **34.55**（np=4，1.22× 慢），整模 +
MTP 22.56 → **28.37**（1.26× 慢），整模不投机对照 32.09；`-np 4` 的三次请求仍是逐个发的，所以变慢不是排队，而是
`n_seq_max` 带大的缓冲（机理未定论）。全表、成本折算与出处见 [`SPECULATION-DATA.md`](SPECULATION-DATA.md) §1。

## 3. DFlash（35B 链，CPU）

四个 prompt（两段代码、一段解释、一句中文），`n_predict 160`，`-np 1`，`-c 4096`：不投机均值 72.28 ms/token；MTP k=2
71.31（1.01×）；**DFlash k=3 64.30（1.12×）**；DFlash k=15 155.52 （0.46×）。单 prompt 对照：链上 **59.96 / 61.66** vs
不投机 **76.72 / 74.18** → **1.24×**； `draft acceptance = 0.71916 (274/381)`，链与整模逐 token 相同 3/3；8 路 tap 逐位
8/8、logits 15/15 行逐位相同。全表、出处与逐位证据见 [`SPECULATION-DATA.md`](SPECULATION-DATA.md) §2。

## 4. 并发与多 slot

| 项 | np=2 | np=4 | 出处 |
| --- | ---: | ---: | --- |
| 10 个不同长度请求 | 10/10 200，存活 | 10/10 200，存活 | `build-split/multislot_{np2,np4}.out` |
| prefix cache 复用（round1/round0） | 0.115（408.1→47.0 ms） | 0.145（406.6→59.0 ms） | 同上（早先两轮是 0.100 / 0.117） |
| 串行 / 并发输出逐字节一致 | PASS | — | `.qlh_tmp/accept_np.py --np 2` |
| tail 的 logits 缓冲 | 1 行 → 0.95 MiB | 4 行 → 3.79 MiB | 缓冲 = `4 B × n_vocab × 行数`，行数 = `n_seq_max × (1+n_max)` |

链上 `-np` 的代价见 §2。`verify_multislot.py` 的两项（np 2 / np 4）在 `verify.sh` 组 5。

## 5. 内存

同一个 1806 token 的 ubatch（`-c 8192 -np 4`，2B 链，`-b 2048`）：tail 的 logits 缓冲从 **1710.76
MiB**（`n_outputs_max=1806`）降到 **3.79 MiB**（`n_outputs_max=4`）；head 的 hidden 缓冲（`logits=no, embd=yes`）14.11
MiB 不变；1806 token 的 prefill（1810 含采样行）从 172.61 t/s （10485.85 ms）到 **219.39 t/s**（8250.22 ms）；上下文
checkpoint 19.266 MiB。修前 tail 按 `n_batch` 行分配 logits，修后只为「实际要被读的行」分配（`n_seq_max ×
(1+n_max)`）——省的不只是内存，同一 ubatch 的 prefill 顺带快 27%，因为不再申请 / 清零那 1.7 GiB。远程 stage
同一笔账更大： `--role tail`、2B 的 20:24 段、`-c 4096 -b 2048`、`n_seq_max=1` 时，`sched_reserve` 的 pp 图缓冲从
**2052.10 MiB** 降到 **240.09 MiB**（只差 logits 行数 2048 → 1）。出处：
`issue1_{before_ub2048,after_final}.log`、`i2_tail_{before,after}.log`。

## 6. 验收覆盖

| 入口 | 覆盖 | 最近一次结果（2026-11-09，纯分段树 `2949267e`） |
| --- | --- | --- |
| `bash build-split/verify.sh` | 十一组：组 1–8 共 **22 项**、组 9 本地 DFlash tap **2 项**、组 10 远程 DFlash/DSpark **20 项**、组 11 DFlash device-buffer **8 项** | 十一组全跑 **52 / 0 / 0**（可用内存 ≥ 闸门） |
| `verify_multislot.py --np 2` / `--np 4`；`accept_np.py --np 2` | 组 5 的两半；串行 vs 并发逐字节 | 各 10/10 200；**ALL PASS**（并发墙钟 1.6 s） |
| `mtp_remeasure.sh`；`verify_seq_rm_exact.py --expect-known-fail`；`verify_seq_rm_shared.py` | §2 的三负载；`seq_rm` 删尾部精度（A/B/C 三形态，无 split 无 server）；shared-prefix 误伤 | 三负载 ×（off / k=2）× 3 次；**1 pass / 5 known-fail / 0 unexpected**，退出 0；shared-prefix 两条 completion 逐字节相同、`[INFO] injection never fired` |

判据全部是逐位 / 逐字节（`split-smoke`、`split-rows`、`dflash-taps` 三个目标，组 11 直接跑产品路径），不看生成文字；组
9/10/11 需要 35B + DFlash 权重。**项数以脚本逐项计数为准**：组 1–8 = 22 项，全十一组 = 52 项（44 + 组 11 的 8
项）；历史上记过的 `PASS=25`、23 项、45 项都是旧口径。 `verify.sh` 默认先跑一次增量构建，`PASS`
里会多出「三目标构建完成」这一项（`PASS=53`）； `--skip-build` 跑出来就是纯验收的 **52**，两种跑法的失败数都是
0，别把那一行算进 52 项。

**纯分段树上的独立验收**：`f477f795c` + 主补丁 = 树 `2949267e`（44 文件，不含 SLIM-ARC）， `--clean-first` 全量构建（385
个编译单元）后 `verify.sh` **52 / 0 / 0**、`verify_chain_death.py` **36 / 0**；组 6 的三段链 checkpoint 往返、组 8
的链上 MTP 逐位、组 10 的 dflash 8/8（wire 2 路、 245760 B）与 dspark 5/5（wire 1 路、122880 B）tap
全部与整模逐位相同、组 11 的整模 device buffer
仍启用。日志：`.qlh_tmp/pure-split-acceptance/`。这证明主补丁不依赖增量补丁。

### 6.1 红的是环境还是代码：先看 93xx 上还有没有上一次的 stage

`verify.sh` 的组 4（`20:24` tail）与组 6（`8:16` relay + `16:24` tail）**都用 9331**；上一次 run 的 stage 没清干净时，组
6 的 client 会连到组 4 那台还在跑的 tail，报 `stage identity mismatch: ... role tail vs
relay`，随后五项一起红。**这是环境，不是代码。根因**： `kill_ours` 只杀「已经绑定 93xx」的进程，而 `netstat`
看不见一个还在 loading 模型、尚未 listen 的 stage；上一轮被打断后紧接着再跑，孤儿进程会在组 6
起来之后才绑上端口。**跑验收前先做两步检查**：

```
netstat -ano | grep LISTENING | grep -E ':93[0-9][0-9]'     # 必须空
tasklist | grep -i 'split-stage\|llama-server'              # 必须空
```

两条都空再跑。**9103/9104/9106 是常驻服务，绝不放进 `kill_ours` 的范围**——脚本只用 `:93xx`
匹配，就是为了这个。**另一个坑：这些 Python 验收脚本不能用 `python -X utf8` 跑**—— `verify_chain_death.py` 的
`kill_93xx()` 用 `subprocess.run(["netstat", "-ano"], text=True)` 读中文 Windows 的 GBK 输出，UTF-8 模式会让 `stdout`
变 `None`，随后在 `.splitlines()` 上抛 `AttributeError`（2026-11-08 实测）。用默认 locale 跑，日志是 GBK，要留档就
`decode("gbk")`。

## 7. 未落盘 / 未复现

完整清单（谁在引用、现状、该不该继续用）见 [`SPECULATION-DATA.md`](SPECULATION-DATA.md) §5。这里
只记两条口径：**「修前第二轮 19.85 s」没有日志**——19.85 s 是整段重算的成本，修前实测到的是失败； **3137 token
在任何日志里都找不到**，日志是 3103（+16 生成 = 3119）。另有两项能力做了但没有端到端
证据：多模态全维位置（代码级结论，分段链 + mtmd 从未跑过）与本地纯 KV 链的 ctx_shift（从未验证打开是否安全）。

## 8. 远程 tail 的候选通道（协议 v5 引入，v8 仍在用）

tail 本地跑 `logit_bias → penalties → top_n_sigma → top_k` 后回传 40 个 `(id, logit)`，server 从 `cand_first`
接着跑，与回传整行数学等价；帧大小 352 B/token（对整行 993 KB 是 2821 倍）；loopback 延迟 57.92 → 52.92
ms/token（+8.6%）。**这些数字只出现在 fork 的提交说明里，不在 `build-split/`
日志里**，按「未落盘到日志」对待。机制、等价性实测与带宽见 [`PROTOCOL-V9.md`](PROTOCOL-V9.md) §1。

## 9. 远程链的投机：五类 prompt 的真实收益

口径：35B `qwen35moe`，head `0:32` + tail `32:40`，贪心，32-token 生成；数据来自 ROG（LAN）、
Surface（4C8T）与本机回环三处，**`build-split/` 里没有对应日志**，可信到「实测记录」这一层。 ROG 五类：散文 882：8.18 /
ngram 7.33 / MTP 8.47（+3.6%）；代码 1205：8.74 / 6.92 / 9.47 （**+8.4%**）；JSON 2379：9.47 / 7.09 /
9.36（**−1.2%**）；agent 2508：7.51 / 5.09 / 7.62 （+1.5%，当时无 draft）；人为重复 89：9.73 / 7.75 /
10.75（+10.5%）。**中位约 +6%，区间 +8.4% ~ −1.2%**；历史那个 +15%（10.20 vs 8.88）来自一个 27-token 的高确定任务，k≥3
的宽帧代价几乎全在 head（32 层 vs tail 8 层），所以「k=2 是甜点」只在该切点成立。**弱机器与回环全负**：Surface 散文
**8.65 → 5.21（−40%）**；本机回环带 grammar（`toolcall.gbnf`）**20.82 → 13.95（0.67×）**，原因是每帧两次 hop、draft
不看 grammar / rbudget（接受率 2.76 → 1.46–1.73）、回退率 9–19%、回环上 993 KB 只要约 1 ms。v7 验证通道单帧回复
**993,312 B → 48 B（20,694×）**；v8 的 `OP_MTP_DRAFT` k 步递归在 stage 内完成，`stage_id.has_nextn`
不参与身份比较。ngram 在 chat 模板下全程 0 draft，只有 raw completion + 人为重复才有收益（27.85 vs 9.59 tok/s =
2.9×、接受率 1.0）。判据不是「与不投机逐字节相同」：llama.cpp 不承诺 batch invariance，CPU + 量化落在
`docs/DETERMINISM.md` 的空白里，三个 ggml 分界与实测见 [`PROTOCOL-V9.md`](PROTOCOL-V9.md) §4。全表与四个原因见
[`CROSS-MACHINE.md`](CROSS-MACHINE.md) §1、§2，逐 prompt 与 k 扫描见 [`SPECULATION-DATA.md`](SPECULATION-DATA.md) §1。

**`split-stage` 的线程与 ubatch**：`GGML_DEFAULT_N_THREADS = 4` 且 `n_threads_batch` 也是 4、不会自动跟随；Surface 上
`-t 8` 让不投机 7.41 → 7.67（+3.5%）、MTP k=2 5.12 → 5.48（+7%），只设 `n_threads` 拿不到收益。见
[`CROSS-MACHINE.md`](CROSS-MACHINE.md) §5。

### 9.7 正确性验收

`verify_mtp.sh`（6 项）：同一行 h 与整模 `h_nextn` 逐位相同、远程 tail 递归 3 步与「整模 + MTP」逐个相同（11751 / 13 /
271）；`verify_mtp_e2e.sh`（6 项）：远程链三组输出逐字节相同。全套 `verify.sh` 11 组 52 项、`verify_chain_death.py`
36、`verify_seq_id.py` 9、`verify_grammar_spec.sh` 23、`verify_rbudget_spec.sh` 36 在 v9 之后全过。

## 10. `seq_rm` 的精确性

判据是「删掉 p 之后的 KV 后重放第 p 行，logits 必须逐位相同」。2B 整模、纯 CPU、p=29、vocab 248320 的探针结果：**1 pass
/ 5 known-fail / 0 unexpected**，退出 0；A1/A2（FULL）与 B1/B2/B3（RS）是已知红（B 组是 rebase 后翻的，上游 `ab3768cdc`
把上界改成 `cell.rs_max`，而 probe 是逐 token decode、`rs_max = 0`），C1 绿（超窗明确拒绝）。产品形状下的回退仍由
grammar / rbudget 两组逐位覆盖。形态表、原始输出与「为什么不改 probe」见 [`PROTOCOL-V9.md`](PROTOCOL-V9.md) §5。

## 11. 换模型：从蒸馏版到官方版（2026-11-08）

两份权重：官方 `UD-Q4_K_M` 21.11 GiB（5.10 BPW）vs 蒸馏版 21.87 GiB（5.29 BPW）；官方 `block_count` 41、`nextn` 1、trunk
40、`ssm.inner_size` 4096、`expert_count` 256 / 8，代码零改动； DFlash / DSpark / NextN
头与官方文件逐字节相同（人工比对、未落盘）。官方整模 k=1..8 的甜点仍是 2： k=2 接受率 **0.9098**、mean len **2.81**、gen
10.54 tok/s；k=3 起接受率掉到 0.859、mean len 4.05 只换来 0.79。官方 vs 蒸馏在 k=2 / 4 / 8 上提升 **+30% / +46% /
+76%**；pos0 两边接近、pos1 起分叉——差的是 hidden 分布一致性，所以蒸馏版的接受率不能拿来做容量规划。**链上折扣**见 [`MODEL-SWAP.md`](MODEL-SWAP.md) §11.4。代码回归用 2B（`verify.sh` 24/24、`verify_mtp.sh` 6/6），新模型验收走「本地双段链 vs 整模逐字节」。
**口径提醒：35B 的「整模 0.910 / 链上 0.688」是官方版上的数，本文 §3 与 §9 的 35B 数字是旧蒸馏版上的数，两份权重不能混用。**
全部细节见 [`MODEL-SWAP.md`](MODEL-SWAP.md)。

### 11.4 链上接受率低 24%

整模 **0.9098 / 2.81** vs 双段链 **0.6879 / 2.37**（低 24%，数值逐字节相同）；根因是本仓加的两处：
`speculative::process()` 的 catch-up 条件是 `!is_mem_shared && !is_chain && !is_remote`（上游只有
`!is_mem_shared`），`draft()` 在 chain 下每轮清草稿 KV。两者构成自洽循环（不 catch-up → 草稿没有 prompt 的 KV →
必须每轮清空 → 即使 catch-up 也读不到），当前策略下跳过 catch-up 是对的，代价是接受率封在
0.688。要做必须同时改两处，收益是估算、应做成 opt-in。见 [`MODEL-SWAP.md`](MODEL-SWAP.md) §11.4。

## 12. 远程链的 `layer_inp`：协议 v9（2026-11-09）

MTP 走 nextn 通道，链上被 `!is_chain` 压到 0.6879；DFlash（0.8298）与 DSpark-5L（0.7986）走 `layer_inp`
tap、本地链与整模逐位相同，但远程链原本没有 tap 通道，等于三种源只能用损失最大的 MTP。v9 补上：`PROTO_VER =
9`；请求末尾追加 `layer_inp_req{n_taps,n_embd}` + n_taps 个 int32 源层号，回复最后追加 n_taps 个 id + `n_rows × n_taps
× n_embd` 个 f32；`_pad` 一字节改存 `n_taps` （尺寸仍 32，前四段解析与 v8 逐字相同）；`MAX_LAYER_INP_TAPS =
16`。传输量的大头是 prefill 而不是 draft：1 行 decode 16 / 8 KB、k=2 verify 48 / 24 KB、prefill 一个 512 行 ubatch **8 /
4 MB**；回环 1192 token 差 **0.2%**，真跨机 1102 token 多 **2.4 s**。tap 缺失只让本请求停投机，不 `send_error`、
不算链死。验收组 10 两个草稿源各 10 项：dflash 8/8 路、dspark 5/5 路 tap 逐位，远程链 logits 9/9、续写 token 8/8、tap
字节数 245760 B。两个接线缺口（`ctx_tail_remote` 只对 MTP 置位；plain 帧 nextn 优先把 tap 吞掉、接受率掉到
0.111）已修，远程链 DFlash / DSpark 零损失。全部细节见 [`PROTOCOL-V9.md`](PROTOCOL-V9.md) §3。

## 13. 投机选型的实测矩阵：markov 头的收益是「可抄写性」

k=2、`n_predict 64`：MTP 0.9098 / 2.820、DSpark-6L 0.8357 / 2.671、DFlash 0.8298 / 2.660、 DSpark-5L 0.7986 / 2.597；6L
DSpark 的 base 是 NVFP4 而非官方 BF16，比较只在「这份文件 vs 那份文件」这一层成立。k=8、`n_predict 256`：short
+2.9、mid（1146）**+15.3**、long（3154）**+14.5**、 hard（1157，中文改写）**+4.1** 点；mid 与 hard
长度几乎相同而差值差近 4 倍——被放大的是可抄写性，不是长度；per-position 里 mid 档 pos6 **+20.2**、pos7 **+21.6**，hard
档 pos0 **−0.3**、pos1 **−3.2**，k=2 上 −0.5 ~ −0.9 点（markov 的价值从 pos2 才开始）。文档问答档的 accept 从 0.71 涨到
0.92，两种头都受益（DFlash 0.33 → 0.62）——抄写红利，不能记在 markov 头上。矩阵与两个口径坑见
[`SPECULATION-DATA.md`](SPECULATION-DATA.md) §3。

### 13.4 两个口径坑

`n_predict` 不可跨比（同一对草稿源 `n=64` 差 +0.59 点、`n=256` 差 −0.94 点）；基座不是同一个 BF16。报 accept 必须带
`n_predict` 与 k，且只能说「这份文件比那份文件高」。

## 14. 跨机实测：Tailscale → Surface，第一次拿到零损失（2026-11-09）

head 本机 `0:32`、tail Surface `32:40`（`--split-remote` 与 `--split-tail-range` 互斥）；官方
`UD-Q4_K_M`（`22,663,387,424 B`）、k=2、贪心、prompt 1102 token。DFlash **117/141 = 0.82979**、 DSpark **117/140 =
0.83571**，与整模逐 prompt **相同**；MTP 108/157 = 0.68790（对照）；输出 3/3 逐字节相同（250 / 244 / 249
字符）。prefill：A ~32.1 s / B ~32.8 s / C ~29.7 s，A−C ≈ +2.4 s。全表、两个边界（互斥参数、target 体积必须一致）见
[`CROSS-MACHINE.md`](CROSS-MACHINE.md) §3。
