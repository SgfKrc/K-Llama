# 投机实测数据：k 扫描、DFlash、选型矩阵与未落盘数字

> 状态：**现行**
> 更新日期：2026-11-11
> 适用范围：MTP / DFlash / DSpark 的逐项实测（2B 与 35B）、草稿源选型矩阵、投机的结构成本，以及「日志里找不到」的数字清单。跨机收
> 益归 [`CROSS-MACHINE.md`](CROSS-MACHINE.md)，换模型的接受率基线归 [`MODEL-SWAP.md`](MODEL-SWAP.md)，协议实现归
> [`PROTOCOL-V9.md`](PROTOCOL-V9.md)。

口径：除注明外 2B 为 `qwen35-2b-Q4_K_M`（head `0:20` + tail `20:24`，本地双段，纯 CPU `build-split`，`-np 1`，
`n_predict 192`，三次取中位）；35B 为 `qwen35moe`（head `0:32` + tail `32:40`，贪心）。

**本机双进程测不出投机收益**（原因见 [`CROSS-MACHINE.md`](CROSS-MACHINE.md) §4）：下面本机的数字只说明「k 与源要按负载和拓扑选」，不能外推「多机一定赚」。跨机实
测见[`CROSS-MACHINE.md`](CROSS-MACHINE.md) §1、§2。

## 1. MTP（2B 链）

### 1.1 退化 prompt（5 token，输出是复读）

| 配置 | ms/token（3 次） | 中位 | 相对 off | 接受率 |
| --- | --- | ---: | ---: | ---: |
| off | 36.84 / 37.36 / 37.32 | **37.32** | 1.00× | — |
| k=1 | 29.54 / 30.96 / 31.24 | **30.96** | 1.21× | 98.96%（95/96） |
| k=2 | 26.26 / 28.32 / 29.44 | **28.32** | **1.32×** | 97.67%（126/129） |
| k=3 | 23.18 / 25.09 / 25.50 | **25.09** | **1.49×** | 96.60%（142/147） |

出处：`build-split/mtp_time/np1_chain_{off,k1,k2,k3}_*.json`、`mtp_time/srv2_chain_{k1,k2,k3}.log` 的
`draft acceptance`。prompt 是 `The capital of France is`，模型只会复读 `The capital of France is Paris.`，**接受率 97
% 是这句话的性质，不是 MTP 的性质**。

### 1.2 三个真实负载（`-np 1`，各三次）

| 负载 | prompt token | off | k=2 | 比值 | 接受率 |
| --- | ---: | ---: | ---: | ---: | ---: |
| 代码续写（`def quicksort` 上下文） | 131 | 35.76 | **41.30** | **0.87×** | 46.9%（92/196） |
| 中英混合（Python + 中文提问） | 95 | 35.74 | **51.69** | **0.69×** | 27.0%（67/248） |
| 纯中文（续写北京介绍） | 23 | 35.78 | **56.86** | **0.63×** | 19.4%（53/273） |

出处：`.qlh_tmp/mtp_remeasure_out.log`（脚本 `.qlh_tmp/mtp_remeasure.sh`）、服务日志
`.qlh_tmp/mtp_re_{off,k2}_064306.srv.log`。把 1.1 与 1.2 按「每轮成本」折算，同一副链上 k=2 的一轮实际花 **2.21–2.24 个 plain
step**（`mean len × ms/token ÷ off`：83.3/37.3、80.1/35.8、79.6/35.7、79.0/35.8），与
[`SPECULATION.md`](SPECULATION.md) §0 的成本模型（k=2 是 4.36 ÷ 2 = 2.18 个单位）吻合。

于是回本线是 `mean len > 2.18`（4.36 ÷ 2，1 个 token 值 2 个单位），即**每轮接受 1.18 个草稿（接受率 59%）以上**；三个真实负载的 `mean len` 是 1.94 / 1.54 / 1.39，全在亏的一侧，而复
读 prompt 是 2.94。**1.3× 不是 MTP 的性质，是接受率的性质**。

### 1.3 `-np > 1` 与 MTP

| 配置 | np=1 | np=4 |
| --- | ---: | ---: |
| 链 + MTP k=2 | **28.33** | **34.55**（1.22× 慢） |
| 整模 + MTP k=2 | **22.56** | **28.37**（1.26× 慢） |
| 整模不投机（对照） | 32.09 | — |

出处：`build-split/mtp_time/srv_np1_spec.log`、`srv_np4_spec.log`（链）；`build-split/srv_w1_m.log`、
`srv_w4_m.log`（整模，`n_slots` 1 → 4）；`mtp_time/whole_off_*.json`（对照行，`predicted_n=128`，而 MTP 各行是 192 token，只能
当量级参考）。`-np 4` 的三次请求仍是逐个发的（日志里没有重叠的 batch），所以变慢不是排队，而是 `n_seq_max` 带大的缓冲——机理仍未定论。

## 2. DFlash（35B 链，CPU）

四个 prompt（两段代码、一段解释、一句中文），`n_predict 160`，`-np 1`，`-c 4096`，每配置一遍（`.qlh_tmp/casc_probe.sh`、
`casc_base.sh`）：

| 配置 | ms/token（4 prompt） | 均值 | 相对不投机 | 整块接受率 |
| --- | --- | ---: | ---: | ---: |
| 不投机 | 74.79 / 68.35 / 71.43 / 74.57 | 72.28 | 1.00× | — |
| MTP k=2 | 74.97 / 74.19 / 61.50 / 74.59 | 71.31 | 1.01× | 56.4%–80.3% |
| DFlash k=3 | 59.43 / 67.38 / 52.25 / 78.12 | **64.30** | **1.12×** | 49.7%–76.4% |
| DFlash k=15 | 150.90 / 197.41 / 118.36 / 155.40 | 155.52 | 0.46× | 12.6%–26.2% |

出处：`.qlh_tmp/casc/{run,run_base}.log`（逐请求 tok/s 与 `draft_n/accepted`），服务日志
`.qlh_tmp/casc/srv_{base,mtp2,k3,k15}.log` 的 `eval time`。

单 prompt 对照（`n_max=3`，两条请求的 `predicted_n` 不同——402 vs 512，投机那条提前停了，所以比 ms/token 而不是总时长）：链上 **59.96 / 61.66
** vs 不投机 **76.72 / 74.18**→ **1.24×**；整模 + DFlash k=3 是 63.96 / 61.93（
`build-split/dflash/speed_{chain,chain_nospec,whole}.log`）。`draft acceptance = 0.71916 (274/381)`，链与整模逐
token 相同 3/3。8 路 tap 逐位（`dflash/taps_final.log`，单 ubatch，15 token）：8/8 与整模相同；tail 的 local 1 = 整模 33、
local 6 = 整模 38；logits 15/15 行逐位相同。

## 3. 选型矩阵：markov 头的收益是「可抄写性」

口径：35B `UD-Q4_K_M` 作 target，整模（不分段，纯 CPU），贪心，`-c 4096 -b 512 -t 8 -ngl 0`；指标取 server 日志的
`draft acceptance = a/g` 与 `mean len`。**每一格的 `n_predict` 与 k 都写在节标题**——这两个口径不能跨比。B 树 / 代码 / 中文三个 prompt
原文在 `verify/acc_test.py` 的 `PROMPTS`。

### 3.1 短上下文（`n_predict 64`，k=2）

| 配置 | accept_rate | mean len |
| --- | ---: | ---: |
| 整模 MTP k=2 | **0.9098** | 2.820 |
| 整模 DSpark-6L k=2 | 0.8357 | 2.671 |
| 整模 DFlash（6 层）k=2 | 0.8298 | 2.660 |
| 整模 DSpark-5L k=2 | 0.7986 | 2.597 |

**6 层 DSpark 的来源与混杂**：`Danny-Dasilva/Qwen3.6-35B-A3B-Aggressive-Q4_K_M-DSPARK-GGUF`，6 层 / block 8 / 8 个
`target_layers`（`[2,7,12,17,23,28,33,38]`，与 DFlash 相同）。它的 base 是 `unsloth/Qwen3.6-35B-A3B-NVFP4`，不是官方
BF16——所以「6L 比 DFlash 高」这一格不能排除混杂。GGUF 里缺 `sample_from_anchor` / `has_confidence_head` 字段，但代码默认
`sample_from_anchor=true`（日志已确认），张量布局正确、能跑。

### 3.2 k=8 长上下文矩阵（`n_predict 256`）

| 档 | prompt_n | DSpark-6L accept | DFlash accept | 6L − DF（百分点） |
| --- | ---: | ---: | ---: | ---: |
| short | ~40 | 0.3619 | 0.3327 | **+2.9** |
| mid | 1146 | 0.7711 | 0.6181 | **+15.3** |
| long | 3154 | 0.7491 | 0.6046 | **+14.5** |
| **hard** | 1157 | 0.3760 | 0.3345 | **+4.1** |

**mid 与 hard 的长度几乎相同（1146 vs 1157），差值却是 +15.3 vs +4.1。所以被放大的不是「长度」，是「可抄写性」**：hard 档是同一份文档 + 要求中文翻译改写（不能逐
字抄），short / long 是原文续写。

per-position 印证：mid 档 pos6 **+20.2**、pos7 **+21.6**（抄写红利最大处）；hard 档 pos0 **−0.3**、pos1 **−3.2**（近处 6L 反而
略差）；**k=2 完全没有优势**（−0.5 ~ −0.9 点）——k=2 只覆盖 pos0/pos1，而 markov 的价值从 pos2 才开始。

结论：markov head 的收益是「预测会重复 / 结构化的内容」，不是「长距离语义依赖」，这恰好是 agent 后端（JSON 键名、代码签名、工具调用）而不是长文档问答的特征。**文档问答档的高接受率
是抄写红利**：三个问答 prompt 的 accept 从 short 的 0.71 涨到 mid/long 的 0.92，两种头都受益（DFlash 从 0.33 到 0.62），不能记在 markov 头
上。

### 3.3 两个口径坑

① **`n_predict` 不可跨比**：6L vs DFlash 的差值在同「短档」`n=64` 时是 **+0.59 点**，`n=256` 时是**−0.94 点**——报 accept 率必须带
`n_predict`，否则同一个结论可以有两个符号。② **基座不是同一个 BF16**（6L DSpark 的 base 是 NVFP4 量化版）：同一份 target 上换草稿源，只能说「这份文件比那份文
件高」，不能说「6 层 markov 结构比 6 层 DFlash 结构高」。用法：选型先看「负载是否可抄写」，再看 `n_predict` 与 k 是否落在这个头擅长的那一段（k≥6 才有 pos2 之后的收
益），最后才比较具体权重文件。

## 4. 结构成本：哪些投机形态在成本上就不成立

- **采样投机不做**。rejection 要比 p 与 q 两个分布、每行需要完整 logits：k=5 一轮 5–7 行 ≈ 5–7 MB，在实测 5–16 MB/s 链路上是 250–800 ms/帧
  。v2 的采样路径正是这么做的，实测比不投机慢 1.9×。远程链的投机是**贪心专用**，上面所有表都是贪心口径。

- **贪心投机在远程链上也不走候选通道，所以每轮照样回传整行 logits。** `plan_remote_candidates()` 把 `spec != nullptr` 直接判成「不能用 top-k」
  （理由：投机每步要采样多行，候选通道只描述单行），于是 temp = 0 的投机在远程链上仍然走 whole-logits（248320 × 4 B ≈**993 KB/轮**），而不投机是 **48
  B/轮**。v7 的 per-row 验证通道（48 B/行，`MAX_VERIFY_ROWS`）只解决「怎么验证草稿」，解决不了「怎么采样」——采样那一步仍要整行。

**但整行 logits 的代价能被接受率赚回来，前提是 prompt 够长、草稿头的 tap 够少**：官方 35B 的**1522 token** prompt 上 DSpark 比不投机快 **+
12.4% ~ +17.2%**、DFlash 持平到 −4.8%；短 prompt 上不划算（draft ctx 的固定建仓约 2.1 s）。表、口径、tap 过网层数与短 prompt 的实测见
[`CROSS-MACHINE.md`](CROSS-MACHINE.md) §7。

- **MTP 只能跟末段**。MTP 不是「草稿头小部件」，是完整的一层 transformer（`blk.40` 共**856.36 MiB**：MoE 专家 820.18 MiB 占 95.8%、
  attention 27.63 MiB、NextN 胶水 8.52 MiB），全文件只有 1 个 `output.weight` 与 1 个 `token_embd.weight`——它不自带
  lm_head、也不自带 embedding。输入是 `t_h_nextn = RMSNorm(最后一层输出, output_norm)`（中间层的 hidden 是未训练的），
  `needs_output() = hi >= n_layer_src`，loader 写死 `nextn_keep = (hi >= n_layer_src) ? nextn_all : 0`。硬挪中段只
  能反着数据流走（tail 算完 → 下发 hidden → mid 跑 MTP → draft 上行 → tail 验证），k=2 实测每轮省 75.5 ms、而两次 RTT 已 50 ms，**净负**；
  复制一份到每段是 856 MiB 起步，且 96% 是 MoE 专家。

- **异构弱网上张量并行是纯负收益，不是「慢一点」**：TP 每 token 要 960 KiB、80 次同步，本项目这种流水线只要 8–16 KiB、1–2 次——差 60–120 倍。TP 通信 ∝ 层
  ×token，PP 通信 ∝ 段数×token，这个差别在 5 MB/s 上是 196.6 ms/token vs 1.6 / 3.3 ms。

- **ngram 的真相**：chat 模板下 ngram **全程 0 draft**——机制是「复制上下文里最后 12 个 token 的后续」，而 chat 模式下模型对重复文本的反应是**评论**（
  "It looks like you've pasted the same paragraph…"）而不是续写，所以永不命中；代码 / JSON 生成的是新内容，同理。**只有 raw
  completion + 人为重复文本才有收益**：同一 prompt 上 ngram **27.85 vs none 9.59 tok/s = 2.9×**、接受率**1.0**——这就是历史那个「
  4.3×」的全部来源，而那次用的 prompt 是 `A + B + A + B + A`。测 ngram 必须用真实负载（散文、代码、agent 多轮）。

## 5. 未落盘 / 未复现

| 数字 | 谁在引用 | 现状 |
| --- | --- | --- |
| MTP k=2 **1.33× / 1.06× / 1.00×**（代码 / 混合 / 中文） | `ARCHITECTURE.md` 与 [`SEGMENTED-LIMITS.md`](SEGMENTED-LIMITS.md) §3.2 | 日志里没有这三个数。重测是 **0.87× / 0.69× / 0.63×**；1.3× 只出现在 §1.1 那种复读 prompt 上（1.32×）。**需在真实负载上重测，不能当结论用** |
| DFlash k=3 **+23.7%** | [`SEGMENTED-LIMITS.md`](SEGMENTED-LIMITS.md) §3.2 | 日志复算是 **1.24×（+24.1%）**，差 0.4pp 在复现噪声内 |
| 级联整块全接受 **42.5% / 2.8%**（k=3 / k=15） | [`SEGMENTED-LIMITS.md`](SEGMENTED-LIMITS.md) §2.3、[`SPECULATION-MATRIX.md`](SPECULATION-MATRIX.md) §4 | 从 `.qlh_tmp/casc/srv_{k3,k15}.log` 数出：`3/3` 96/226 轮、`15/15` 5/177 轮。把 `1/1`、`2/2`、`4/4` 这类块尾也算上是 **43.4%（98/226）/ 4.5%（8/177；复算到 9 轮，多出的一轮是 `4/4`）**。两个口径不完全可比（那边数的是 Python 驱动的级联路径），k=15 只有 5–9 轮全接受，噪声很大 |
| checkpoint **19.85 s → 0.71 s** | [`PATCHES.md`](PATCHES.md) 第四代、[`SEGMENTED-LIMITS.md`](SEGMENTED-LIMITS.md) §1 | 两个数各自成立，但「修前第二轮 19.85 s」没有日志：19.85 s 是整段重算的成本，修前实测到的是**失败**（`kill_ckpt_9415/9416`） |
| **3137 token** 历史 | 同上 | 日志是 **3103**（+16 生成 = 3119）；3137 在任何日志里都找不到 |
| 2B 上 k=3 与 k=2 的 **25.09 vs 25.87** | 同上 §3.2 | k=3 中位 25.09 对得上；k=2 中位是 **28.32**，25.87 未复现 |
| 35B MTP k=2 **1.13–1.25×** | [`SPECULATION.md`](SPECULATION.md) §1 | 那份是 10-31 的草稿成本模型 + 单 prompt；链上端到端（4 prompt）是 **1.01×**。两个口径都要保留，不能互相替代 |
| v5 候选通道的等价性 / 带宽 / 延迟 | [`SEGMENTED-LIMITS.md`](SEGMENTED-LIMITS.md) §1（数据见 [`PROTOCOL-V9.md`](PROTOCOL-V9.md) §1） | 只在 fork 的提交说明里，`build-split/` 日志中没有这些行；按「未落盘到日志」对待 |
| DFlash 头 / DSpark 头 / NextN 层「逐字节官方」（sha256 / LFS oid） | [`MODEL-SWAP.md`](MODEL-SWAP.md) §11.1 | 当日的人工比对，`build-split/` 下没有日志；独立复算没做过。它支撑的是「接受率退化不可能来自头是兼容版」这个排除项 |
| catch-up 让链上接受率回到整模（0.688 → ~0.91） | [`MODEL-SWAP.md`](MODEL-SWAP.md) §11.4 | **已实现并实测**（2026-11-11，自适应默认）：mode 2 得 0.9098 = 整模逐位相同（121/133），mode 1 得 0.7355（512 档）。成本实测 1.19 ms/行 + 2.1 s 固定，原来「每请求一次 N 行 draft 前向」的估法**偏高 10–40 倍** |

另有两项能力做了但没有端到端证据（不是数字）：**多模态全维位置**——`split_expand_pos()` 与 `req_header::n_pos` 让 4 个 M-RoPE section 跨段传下去
，验收组 6 的三段跨进程在 `pos = [p,0,0,0]` 上逐位通过，但没有图像输入的端到端验证（`H:/qlh_models/mmproj` 有投影模型，分段链 + mtmd 的组合从未跑过），所以「图
像位置能传」是代码级结论。**本地纯 KV 链的 ctx_shift**——`can_shift=false` 对 qwen35/qwen35moe 忠实，但对非 MROPE 的本地链过度保守，从未验证打开是否
安全；`seq_add` 是 void、无返回值，要做到「逐段都真移了同样量」只能靠 shift 后断言各段 `seq_pos_max`。

## 6. 草稿模型：文件清单与来源状态

DFlash / DSpark 的草稿是**独立 GGUF**，用 `-md` 加载，与 target 整模分开。实测用的文件与它们的元数据（`general.name` /
`general.architecture`，`gguf` python 包直读）：

| 文件（`H:/qlh_models/`） | 大小 | `general.name` | arch |
| --- | ---: | --- | --- |
| `dflash/dflash-merged-Q4_K_M.gguf` | 1.00 GiB | `Qwen3.6 35B A3B DFlash` | dflash |
| `dflash/qwen36-35b-a3b-dflash-Q4_K_M.gguf` | 0.22 GiB | 同上 | dflash |
| `dspark/Qwen3.6-35B-A3B-DSpark-Q4_K_M.gguf` | 0.53 GiB | `Qwen3.6 35B A3B Speculator.Dspark` | dflash |
| `dspark/dspark_conf_high.gguf`、`dspark_markov_zero.gguf` | 0.53 GiB 各 | 同上 | dflash |
| `dspark6/Qwen3.6-35B-A3B-DSPARK-6L.gguf` | 0.97 GiB | `Qwen3.6 35B A3B NVFP4 DSPARK` | dflash |

`arch` 一律是 `dflash`——DSpark 在这个 fork 里走的是同一条 `layer_inp` tap 通道，不是另一套加载器。`dspark_conf_high` /
`dspark_markov_zero` 是同尺寸的配置消融（confidence / markov 头开关）；`DSPARK-6L` 是 6 层版，name 里带 **NVFP4**（base 是 NVFP4
量化版而非官方 BF16，见 `MODEL-SWAP.md` §11.3 的口径坑）。

**来源状态：未落盘。** 元数据只能证到「它自称是 Qwen3.6 35B A3B 的 DFlash / DSpark 头」；`MODEL-SWAP.md` §11.1 记的「与官方文件逐字节比对（
sha256 / HF LFS oid）」是当日人工比对，`build-split/` 下没有日志、独立复算没做过（§5 表里同一条）。**这几个文件怎么得到的、从哪个仓的哪个 revision 取的，仓库里没
有任何记录**——它支撑的只是「接受率退化不可能来自头是兼容版」这一个排除项，不能当既成事实引用。
