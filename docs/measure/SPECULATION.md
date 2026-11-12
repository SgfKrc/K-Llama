# 投机解码：四个方向的判定

> 状态：**已验证**（文中引用的 v2 工具链——`chain_client` / `stage_worker` / `shim/` / `koakuma`——已退场，归档在
> `slimarc/prototype`，命令与路径只作历史口径）
> 更新日期：2026-11-12
> 适用范围：Qwen3.6-35B-A3B（Q4_K_XL，40 层）、两段跨机（本机 0:32、ROG 32:
> 40）上四种投机方向的判定与实测；读者为调参与后续实施者。未做部分：PLD 影子统计。相关：`SPECULATION-MATRIX.md`（草稿源 × 段结构）、
> `MOE-SPECULATION-ECONOMICS.md`（MoE 成本结构）

> **口径（2026-11-08 换模型后补）**：本文所有 35B 数字都是 v2 时期、旧权重上的数——蒸馏版（
> `Qwen3.6-35B-A3B-Claude-4.7-Opus-Reasoning-Distilled-APEX-MTP-I-Quality`，见 §4）或更早的 `UD-Q4_K_XL`，**不要当官方
> 模型的基线**。主线自 2026-11-08 起换成 `unsloth/Qwen3.6-35B-A3B-MTP-GGUF` 的 `UD-Q4_K_M`。所以下文的「MTP 1.13–1.25×」「
> DFlash 1.5×」「接受 42–95%」都是旧模型 + 旧驱动链上的数。官方模型的 k 扫描基线见[`MODEL-SWAP.md`](MODEL-SWAP.md) §11.2，链上折扣见 §11.4
> 。

本文记录四种投机方案在本项目上的可行性判定，每条都带实测依据；没有实测的标注为估计。

---

## 0. 成本模型：k 决定一切

一轮投机的位置操作账（两段链，每段各算一次）：

```
一轮 = tail 单步(1) + head 批量(k) + tail 批量(k) + head 预写(1)
一个 k 行 batch 的实测成本 = 1 + 0.178 × (k-1) 个单行
  （0:32 段 72 / 81.8 / 105 / 161.3 / 269.3 ms 于 k=1/2/4/8/16）
plain 生成 2 个 token = 4.00 个单位
推算：成本(k) = 2 + 2 × [1 + 0.178(k-1)]；"盈亏线"与"接受率 85% 时"两列同为推算
```

| k | 成本 | 每草稿需接受的盈亏线 | 接受率 85% 时 |
| ---: | ---: | ---: | ---: |
| 1 | 4.00 | **0（r > 0 即回本）** | 1.40 单位/token，快 1.43×（推算；实测 90.9% 时 1.18×） |
| 2 | 4.36 | 0.09 | 1.61，快 1.24× |
| 4 | 5.07 | 0.13 | 1.15，快 1.74× |
| 8 | 6.49 | 0.16 | 0.83，快 2.40× |
| 16 | 9.34 | 0.17 | 0.64，快 3.13× |

**k=1 不亏，只是不如 k=2**：验证块 2 行的成本与 plain 生成 2 个 token 相当，但前进的不是 `1 + r` 而是 **`2 + r`**——`speculate()` 的契约里，验
证块最后一个位置的那个 token 是"免费"的（块已经算出来了，调用方不许重算）。所以 k=1 在 90.9% 接受率下每轮前进 2.91 个，实测 1.18×。

**k ≥ 2 才有意义**，门槛很低（每草稿接受 9-17% 即回本）。这是 MTP 与 DFlash 的分水岭。

## 1. 判定汇总

| 方向 | 判定 | 依据 |
| --- | --- | --- |
| MTP（模型自带） | **代码场景实测 1.13–1.25×** | k=1 是 1.18×（含免费尾 token），k=2 是 1.25×；递归上限 2 |
| DFlash（外挂 0.24 GB） | **第一个可落地的外挂草稿，实测 1.5×** | 代码场景接受 42-95%，远超 24.5% 盈亏线（§5） |
| PLD / n-gram | **真实负载 replay 已测不达标，停手**；影子统计未做 | 中文对话 m=1.05（无效），代码续写 m=2.56–3.37；判据见 `SPECULATION-MATRIX.md` §6 |
| 浅层投机（前 k 层） | 不做 | 实测 α=0.00–0.08，E[accept]≈1.0 |
| dSpark（markov 头） | **有 Qwen3.6 版本了（2026-11-09），长上下文优于 DFlash 但只在「可抄写」负载上** | 矩阵与两个口径坑见 [`SPECULATION-DATA.md`](SPECULATION-DATA.md) §3 |

**这张表的 MTP 与 DFlash 两行都测在旧权重上**（2026-10-31～11-02，蒸馏版或 `UD-Q4_K_XL`）。官方 `UD-Q4_K_M` 的 MTP 接受率基线是 **k=2 的
0.910**，不是这里的 70%；DFlash **没有在新模型上重测**。两套数字不能并列比较，见 [`MODEL-SWAP.md`](MODEL-SWAP.md) §11.2。**dSpark 那一行是
新权重（社区量化，基座是 `unsloth` 的 NVFP4 版本，不是官方 BF16）**，口径见[`SPECULATION-DATA.md`](SPECULATION-DATA.md) §3。

**接受率必须带 `n_predict` 与 k**：换 `n_predict` 能让同一个草稿源的相对优劣翻号，换 k 能让方向性结论失效（markov 的红利从 pos2 才开始，k=2 上完全没有优势）
。两个口径坑与具体差值见[`SPECULATION-DATA.md`](SPECULATION-DATA.md) §3.3。

## 2. 前置条件：recurrent state 回滚

hybrid 架构（`LLL F` 交替）的 linear 层有 recurrent state，而它**没有位置概念**。验证 k 个候选后只接受 j<k 个时，必须把位置 j 之后的 KV **和
**state 一起退回去。不开快照时上下文日志是 `n_rs_seq = 0`（shim 的注释是 "no recurrent-state rollback"）：

```
llama_context: n_rs_seq = 0
llama_memory_recurrent: size = 50.25 MiB (1 cells, 32 layers, 1 seqs 0 rs_seq)
```

内存代价已实测（`SA_N_RS_SEQ=4`，35B）：head 0:32 → **251.25 MiB**（R 11.25 + S 240）；tail 32:40 → **62.81 MiB**（R
2.81 + S 60）。S 随 `n_rs_seq` 线性增长；tail 那 63 MiB 塞进 ROG 的 0.82 GiB 可行，但吃掉七分之一余量。

**回滚已可用**：不开快照时 `qlh_kh_seq_rm` 返回 **-6**，开到 4 后返回 0（与 llama.cpp #31368 一致，只放行最近一批快照内的回退）。shim 侧
`SA_N_RS_SEQ ≥ k` 即可，无需改 llama.cpp。

---

## 3. 验证成本：k 个候选只需一次绕链

`qlh_kh_forward_embd_tokens` 接受 k 行 hidden、返回 k 个位置的 argmax，所以验证 k 个候选只走**一次**本机→ROG→回的往返。但摊薄不均匀：整段耗时随 k
增长（35B tail 段实测 23.5 / 27.9 / 34.7 / 58.3 / 90.3 ms 于 k=1/2/4/8/16），层计算能被 batch 摊薄（MoE 专家共享），而 lm_head 的
增长**不是线性的**——按 16 列一档分块，档内加行几乎免费（M=8 与 M=16 同为 53 ms，M=17 跳到 81 ms）。

档位见 `MOE-SPECULATION-ECONOMICS.md` §5，成本不可按"每行固定毫秒"线性推算。**这是本项目所有投机方案的成本上限**（`SPECULATION.md` §9 另证批量与逐位
置逐位一致）。

## 4. MTP：首选

手上有**一份带 MTP 的蒸馏变体**（`H:\qlh_models\qwen3-6-35b-a3b-opus47-distill-apex-mtp\`）：`block_count = 41`（40 主层
+ `blk.40`）、`nextn_predict_layers = 1`、`blk.40.nextn.{eh_proj,enorm,hnorm,shared_head_norm}`。

`common/speculative.cpp` 的注释写明档位：`chain_heads`（step35）每步一个训练好的头，`qwen35` / `qwen35moe` 只有一个训练好的 MTP head
。

**qwen35moe 是单 head**。k=1 能到 1.18×（§10），但**递归才让它真正值钱**：同一个 hidden 反复喂自己的输出，`(h_p, d1) -> d2`，k=2 到 1.25×
。第 3 层掉到 36%，所以递归上限是 2。

递归的接受率按层衰减，实测（50 token，k=2）：

| 场景 | 第 1 层 | 第 2 层 | 平均接受 | 4.36 ÷ (1+接受) | 结果 |
| --- | ---: | ---: | ---: | ---: | --- |
| **代码**（`def quicksort`） | 85.1% | 87.5% | **1.60** | 1.68 | **快 1.19×** |
| 文档（Markdown 表头） | 62.2% | 65.2% | 1.03 | 2.15 | 慢 7% |
| 对话（`中国的首都是`） | 60.0% | 73.3% | 1.24（k=4） | >2.0 | 慢 |

k=2 盈亏线是 **1.18 个草稿**，只有代码场景达标。第 3 层掉到 36%、第 4 层 17%——**递归上限就是 2**。MTP 的定位是「代码续写 1.19×」，不是通用加速，必须按负载开关（§
10 的 `--spec-k -1`）。

**接受率已实测**（head 0:32 + tail 32:40 + MTP，贪心，三个 prompt）：`中国的首都是` 12/19 = 63.2%、`用一句话解释什么是层段推理` 13/15 =
86.7%、`def quicksort(arr):` 14/15 = 93.3%，合计 **39/49 = 79.6%**。代码类最高、对话居中偏上，与 Qwen 官方的 60-80% 一致。

**这一段是蒸馏版上的数**（见文档头的口径块）；官方模型的同口径数字与链上折扣见[`MODEL-SWAP.md`](MODEL-SWAP.md) §11.2、§11.4。

**部件已接通**：`qlh_kh_load` 在 `SA_MTP=1` 且模型带 NextN 时创建第二个 context（`ctx_type = LLAMA_CONTEXT_TYPE_MTP`），配套
`qlh_kh_last_hidden` 取主图 `t_h_nextn`、`qlh_kh_mtp_draft` 用 `llama_batch_ext` 提交 `(token, hidden)` 并
argmax；loader 侧的两个障碍也已修（`nextn` 保留、`n_layer_all` 减去 MTP 层）。

## 5. DFlash：第一个可落地的外挂草稿

权重可用：[z-lab/Qwen3.6-35B-A3B-DFlash](https://huggingface.co/z-lab/Qwen3.6-35B-A3B-DFlash)（Q4_K_M 仅 0.24
GB，本机 `H:\qlh_models\dflash\`），llama.cpp 已支持。GGUF 关键参数：`block_count=6`、**`block_size=16`（一次出 16 个位置）**、
`target_layers=[2,7,12,17,23,28,33,38]`——后两个落在 ROG 段，两段各交一部分 hidden 即可（顺序天然是「前 6 本地 + 后 2 远程」）。

**实测接受率**（2026-10-31，`llama-speculative-simple` + 完整 40 层 target + 本模型，CPU 版；`n_max` 是草稿深度，实际 1 anchor +
n_max 草稿）：

| n_max | 场景 | 草稿/接受 | α | 前进/轮 | 预估加速 |
|---:|---|---:|---:|---:|---:|
| 3 | 代码 | 40/38 | **95.0%** | 3.85 | **1.52×** |
| 3 | 对话 | 54/33 | 61.1% | 2.83 | 1.12× |
| 15 | 代码 | 131/55 | **42.0%** | 7.30 | **1.56×** |
| 15 | 对话 | 192/39 | 20.3% | 4.05 | **0.87×（亏）** |

**盈亏线 24.5%**（成本 9.34 单位/轮、1+15α 前进）。**代码场景远超，对话在深 draft 下亏**——所以 DFlash 也要按负载选 `n_max`：代码 15、对话 3。**对比
MTP k=2 跨机实测 1.13×，DFlash 代码场景 1.5×**——这才是本项目的大增量。

**这张表的模型口径**：2026-10-31 用 `llama-speculative-simple` 加完整 40 层 target 测的，target **不是** 2026-11-08 的官方
`UD-Q4_K_M`（那时还没有这份权重）。换官方模型之后 DFlash **没有重测**，所以表里的 1.5× 与上面的 MTP 1.13× 都不该跟官方模型的接受率并列。官方基线见
[`MODEL-SWAP.md`](MODEL-SWAP.md) §11.2。

三条已确认的事实：**① llama.cpp 零改动**（`embeddings_layer_inp` 本来就是 per-layer vector，可同时开 8 个槽）；**② 真正的障碍是词表**（
DFlash GGUF 无 `tok_embd`/`output.weight`，靠 `ctx_other` 借 target 的，跨段后没有任何一段同时持有两者，跑 draft 的机器要多背 1.03 GiB
）；**③ k=16 是 1 anchor + 15 草稿**（`speculative.cpp` 把 `n_max` clamp 到 `block_size-1`）。

> **2026-11-09 跨机实测（协议 v9 的 `layer_inp` tap 通道）**：Tailscale → Surface，1102 token 的 prompt，DFlash k=2 接受 *
> *117/141 = 0.82979**，与整模**逐 prompt 相同**；DSpark k=2 **117/140 = 0.83571**，同样与整模相同；对照 MTP k=2 旧路径只有
**> 108/157 = 0.68790**。prefill 比 MTP 多约**3.05 s**（32.1 s vs 29.05 s）。也就是说：**DFlash / DSpark 在真跨机链上不再有「跨机损失」
> 这一项**——此前缺的正是把 8 路 tap 送到远端的通道；上文的 1.5× 是理想模型，它不是加速比的预测值，只是「DFlash 值得接」的判据。数据与口径见
> [`CROSS-MACHINE.md`](../deploy/CROSS-MACHINE.md) §3。

## 6. PLD / n-gram：按负载开关

llama.cpp 内置 5 种 n-gram 草稿策略，核心是 `common/ngram-map.cpp` 的 `common_ngram_simple_draft`——纯函数，不依赖 llama 上下文
，Python 侧 20 行可移植。

2B 真实 gold 序列上的模拟：中文对话 1.05–1.08、Markdown 表格 1.36、中文代码 1.23、**代码续写补全 2.56–3.37**、**原样重复/抄写 2.67–4.92**。
**m 由文本重复性决定，与模型规模几乎无关**，盈亏线 m≥2.5。落地取**影子模式**：每步只统计命中与可接受长度（零开销），确认某类请求 m>2.5 后再启用。真实负载 replay 已否掉它（
`SPECULATION-MATRIX.md` §6）。

## 7. 两个被否掉的方向

**浅层投机（取前 k 层当草稿）**：2B 切出 head4/8/12，`token_embd` 单行反量化后经 embd 注入，逐位置 teacher forcing 对比完整链：α 分别 0.083 /
0.042 / 0.062，代码生成全为 0.000，E[accept]=1.00；对照组完整 24 层是 0.969 / 0.906（注入路径本身没问题）。未训练的截断层接上 `output_norm` +
`lm_head` 后分布错位——LayerSkip 那类效果来自训练时的 early-exit loss。

**dSpark**：外挂草稿 + Markov 头 + 置信度调度。**2026-11-09 起有 Qwen3.6 版本**（社区量化 `Danny-Dasilva/…-DSPARK-GGUF`，6 层 /
block 8 / 8 个 `target_layers`，基座是 `unsloth` 的 NVFP4 而不是官方 BF16）。收益的性质是「预测会重复 / 结构化的内容」，不是长距离语义；矩阵与口径坑见
[`SPECULATION-DATA.md`](SPECULATION-DATA.md) §3。

## 8. 落地顺序

1. **MTP 递归 k=2**：已接入，跨机实测 **1.13×**（70% 接受，§10），按负载开关用。
2. **n-gram / PLD**：真实负载 replay 已测不达标（`SPECULATION-MATRIX.md` §6），停手；影子统计仍未做。
3. **DFlash**：k=5 已在盈利区间，是**第一个**值得投入完整工程的外挂草稿（§5、§11）；外挂草稿现在多了 DSpark（markov 头）这个竞争者，选型按负载的「可抄写性」而不是
   prompt 长度（[`SPECULATION-DATA.md`](SPECULATION-DATA.md) §3）。

## 9. 数值验证结论

位置账在 `slimarc/prototype:checks/probe_spec_wip.py`。三个问题都已定论：

1. **head 的多 token 批量 ≠ 逐位置**：hidden 差 2.5e-2~3.8e-2 但不翻转 argmax——用「完美草稿」跑 30 个 token，2B 与 35B 都逐位一致，块的第
   1 行（验货）与第 2 行（免费 token）都对（`slimarc/prototype:checks/probe_perfect_draft.py`）。
2. **`qlh_kh_seq_rm` 的回滚**已修，根因是 `n_rs_seq` 快照只有单 ubatch 内语义、跨 ubatch 不滚动，修法是在 `build_rs` 里插移位节点；同形状回滚实测
   **0.000e+00**，跨形状仍有 ~1e-2，是 GDN 两套 kernel（CH/AR）的算术差异，不影响 argmax。官方从不跨 ubatch 回滚（verify batch 一次写
   `n_draft+1` 个 token），所以这个缺陷只有「分段 + 单 token draft」才暴露（
   [`ISSUE-recurrent-snapshot-rollback.md`](../issues/ISSUE-recurrent-snapshot-rollback.md)）。
3. **tail 段 8 层里 6 层是 linear**（32/33/34/36/37/38，只有 35/39 是 full attention），recurrent state 没有位置索引——这是上面
   那条缺陷只在本项目暴露的根本原因。

## 10. 接入与实测

已接进 `chain_client.py`（`--spec-k N`），`forward_batch` 一次绕链验证整个候选块。**正确性**：token 序列与 `--spec-k 0` 逐位相同（本机、跨机
都验过）。

**批次 A：跨机 k=2 的 A/B 重跑**（`topology-cross-mtp.toml`：本机 0:32 + ROG 32:40，U 盘传模型后 SHA256 一致；每配置多次重跑，取中位）：

| 配置 | 中位 | 平均 | 范围 |
|---|---:|---:|---|
| baseline | **205.8 ms/token** | 200.2 | 174-226 |
| spec k=2 | **182.2 ms/token** | 180.1 | 165-187 |

**1.11–1.13×**（中位 205.8 → 182.2 是 **1.13×**，平均 200.2 → 180.1 是 1.11×），接受率 70.0%，每轮前进 3.20 token。比 §4 预测的 1.19× 略低，因为那个数是按 85%接受率算的。**波动也变小**：批量验证把每步一次的不稳定开销摊到
3.2 个 token 上。

**批次 B：k 扫描**（同一 prompt、同机、gen=32 热态；每个 k 单次运行，基线取 k=0 那一格 **234.8**）：

| k | 接受率 | 每轮 token | ms/token | vs baseline |
|---:|---:|---:|---:|---:|
| 0 | — | 1 | 234.8 | — |
| 1 | 90.9% | 2.91 | 198.7 | 1.18× |
| **2** | **70.0%** | **3.20** | **187.9** | **1.25×** |
| 3 | 46.7% | 3.20 | 226.7 | 0.96× |
| 4 | 35.0% | 3.20 | 301.9 | 0.78× |

**k=3/4 的每轮 token 与 k=2 完全相同（3.20）**：第 3、4 层草稿一次都没被接受，只是多付验证成本，独立复核了 §4 的「递归上限 2」。

**两批的 baseline 不同，加速比不能互相替换**：批次 A 的基线是 **205.8**（多次重跑中位），批次 B 是 **234.8**（同一次 k 扫描里的 k=0），两者差 29 ms（相对 205.8 高 14%），来自批次与口径不同（A 是 A/B 重跑取中位，B 是单次热态扫描）。**跨机实测取批次 A 的 1.13×**；1.25× 是批次 B 内部的 k=0 → k=2 比值，只在「同一进程、同一时刻的 k 扫描」这一口径下成立。两批都测在旧权重上（见文档头的口径块）。

**默认改为按负载自动选择**（`--spec-k -1`）：prompt 像代码就 k=2，否则 k=0；跑满 6 步后若接受率低于 35% 自动关闭。`--spec-k N` 仍可强制。

本机两段（`topology-local-32-8.toml`）**实测有效**（2026-11-01）：不投机 4.9 tok/s / 204.1 ms/token，DFlash k=5 是 **8.5
tok/s / 117.8 ms/token（1.73×）**，接受 97/150 = 64.7%；注意这条收益会被配置不一致或进程泄漏污染显存吃掉（`3b37f299` 已修）。

**已知限制**：只对 temp=0 无损（只暴露 argmax，做不了 rejection sampling）；自动切换的阈值（35% / 6 步）是按 k=2 的成本模型定的，换 k 要重算；
**DFlash 仍缺同口径的 A/B**——跨机 k=5 的端到端是 12.9 tok/s（`check_p4_cross.py`，含 prefill），而 12.2 那个基线是 128-token 的
decode-only 口径（§11），两个数不能直接相减。

## 11. k 扫描：拐点是 k=5

2026-10-31，`slimarc/prototype:scripts/check_p4_cross.py`，32:40 跨机链路，prompt `def quicksort(arr):`，只计解码阶段（
不含 prefill 与协议开销），同一 prompt 同一 worker；每个 k 单次运行，故不报分散度。

**口径**：这一节是旧权重 + v2 驱动链，不是官方 `UD-Q4_K_M` + v2.1 原生投机；官方模型上重扫的 k=1..8、甜点仍是 k=2，见
[`MODEL-SWAP.md`](MODEL-SWAP.md) §11.2。

| k | 64 token：接受率 | token/轮 | ms/轮 | tok/s | 128 token：接受率 | token/轮 | ms/轮 | tok/s |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 3 | 98.0% | 3.94 | 232 | 17.0 | 74.2% | 3.23 | 263 | 12.2 |
| **5** | **96.4%** | **5.82** | **278** | **20.9** | **70.3%** | **4.52** | **338** | **13.4** |
| 8 | — | — | — | — | 31.1% | 3.49 | 364 | 9.6 |
| 15 | — | — | — | — | 11.1% | 2.67 | 473 | 5.6 |

两件事同时成立：

- **最优 k 是 5，不是更大。** 每多验证一行，tail 段就多算一行层计算，`lm_head` 另按 16 列一档跳（§3）；k=8 之后多出的草稿几乎全被拒，成本照付。
- **接受率强烈依赖生成区段。** 同一 prompt 前 64 token 是 96–98%，写到 128 token 时掉到 70–74%——代码的开头可预测、后段不可预测。所以每个"接受率"数字都必须
  带生成长度口径，否则 98% 与 74% 会被读成矛盾。
