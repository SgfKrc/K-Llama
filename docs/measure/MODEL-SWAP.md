# 换模型：从蒸馏版到官方版（2026-11-08）

> 状态：**现行**
> 更新日期：2026-11-12
> 适用范围：35B target 从蒸馏版换到官方 `UD-Q4_K_M` 的全部结论——两份权重的 metadata、形状核对、接受率基线、官方 vs 蒸馏
> 的差距、链上 13% 折扣的根因，以及「代码回归」与「新模型验收」的分界。`build-split/accept_baseline.md` 是原始记录，两份不一致时以日志为准（下文标出两处小出入）。投机矩
> 阵与 k 扫描归 [`SPECULATION-DATA.md`](SPECULATION-DATA.md)，协议归[`PROTOCOL-V9.md`](../design/PROTOCOL-V9.md)。

## 11.1 两份权重、为什么换

| | 旧（**已弃用**） | 新（**现行**） |
| --- | --- | --- |
| 文件 | `Qwen3.6-35B-A3B-Claude-4.7-Opus-Reasoning-Distilled-APEX-MTP-I-Quality.gguf` | `Qwen3.6-35B-A3B-UD-Q4_K_M.gguf` |
| 来源 | lordx64 蒸馏 + **mudler 的 "apex" 工作流**做的混合量化（**模型卡说法**：Q6_K 主 + Q8_0 nextn；与逐张量清点不符，见下） | `unsloth/Qwen3.6-35B-A3B-MTP-GGUF` 的 `UD-Q4_K_M`（官方 MTP 仓） |
| 体积 | 23.48 GB（21.87 GiB） | **21.11 GiB** |
| 本机路径 | `H:/qlh_models/qwen3-6-35b-a3b-opus47-distill-apex-mtp/` | `H:/qlh_models/qwen3-6-35b-a3b-mtp-official/` |
| `general.name` | **`Safetensors`**（转换时没传 `--model-name`） | `Qwen3.6-35B-A3B` |
| 其他遗留 metadata | `general.dataset.0` 指向 `Lordx64 / Reasoning Distill Opus 4 7 Max Sft`；`quantize.imatrix.file` 是构建机的 `/home/ubuntu/work/apex/...`，`quantize.imatrix.dataset` 是 `/home/mudler/autoresearch-quant/...` | `quantize.imatrix.file` 是 `Qwen3.6-35B-A3B-GGUF/imatrix_unsloth...`，`quantize.imatrix.dataset` 是 `unsloth_calibration_...` |

出处：旧模型 metadata 在 `build-split/acc/dist_pos_k8.log` 的 `llama_model_loader` 段，新模型在
`acc/official_pos_k8.log` 的同一位置。体积以两份日志的 `print_info: file size` 为准：官方 **21.10 GiB（5.10 BPW）**、蒸馏
**21.86 GiB（5.29 BPW）**；上表的 21.11 / 21.87 与 `accept_baseline.md` 一致，是同一份文件的四舍五入差（0.01 GiB）。

「Q6_K 主 + Q8_0 nextn」与 mudler 的 apex 工作流来自该 GGUF 的模型卡，不是日志——日志只能证到 `general.dataset.0` 指向 lordx64 的蒸馏仓、以及
5.29 BPW 比官方的 5.10 BPW 更高。**但「Q6_K 主」这个说法与逐张量清点不符**：按 `scripts/inspect_gguf_quant.py` 清点
（[`THREE-SEGMENT.md`](../deploy/THREE-SEGMENT.md) §2），蒸馏版最大的一块是 **IQ4_XS 8.367 GiB（38.3%）**，Q6_K 只有
**7.294 GiB（33.4%）**、Q5_K **5.156 GiB（23.6%）**、Q8_0 0.960（4.4%）——是 **IQ4_XS / Q6_K / Q5_K 三档混合**，不是 Q6_K 主。
所以「Q6_K」只能当模型卡标签用，不能拿它算每层重量或估算显存（**以逐张量清点为准**）。

> **与 `accept_baseline.md` 的一处出入**：那份记录写 `quantize.imatrix.file = /home/mudder/autoresearch-quant/...`。日
> 志里这是两条不同的 key——`imatrix.file` 指 `/home/ubuntu/work/apex/...`，`/home/mudler/autoresearch-quant/...` 是
> `imatrix.dataset`（作者名是 `mudler`，不是 `mudder`）。本节按日志写。

**新模型的形状与代码假设全部吻合**（`llama_model_loader` / `print_info` 段）：

| key | 值 | key | 值 |
| --- | --- | --- | --- |
| `general.architecture` | `qwen35moe` | `qwen35moe.ssm.state_size` | **128** |
| `general.name` | `Qwen3.6-35B-A3B` | `qwen35moe.ssm.group_count` | **16** |
| `qwen35moe.block_count` | **41** | `qwen35moe.ssm.conv_kernel` | **4** |
| `qwen35moe.nextn_predict_layers` | **1** | `qwen35moe.full_attention_interval` | **4** |
| `qwen35moe.ssm.inner_size` | **4096** | `qwen35moe.expert_count` / `expert_used_count` | **256** / 8 |

trunk depth 读作 **40**（41 block − 1 个 NextN），与 `n_layer_nextn == 1` 的断言一致；所以在 tail 段起 `32:40` 时
`--role tail` 的 `has_nextn=1` 成立、MTP draft context 建得起来。**换模型零改代码**——形状这一层没有任何要适配的东西。

**头的来源也核对过**：DFlash 头、DSpark 头、NextN 层都与官方文件做过**逐字节**比对（sha256 / Hugging Face LFS oid），三者都是官方的，不是「自己拼的兼容
版」；这把「接受率退化」的候选原因砍掉了一类——但它是当日人工比对、`build-split/` 下没有日志，记为「未落盘」（
[`SPECULATION-DATA.md`](SPECULATION-DATA.md) §5）。

## 11.2 接受率基线（官方模型，整模，MTP）

口径：3 个真实 prompt（英文 prose / Python 代码 / 中文技术），`temperature 0`、`n_predict 64`、`-c 4096 -b 512 -t 8 -ngl 0`
、纯 CPU。prompt 原文在 `verify/acc_test.py` 的 `PROMPTS`：B-tree 插入怎么保持平衡（prose，39 token）、`merge_intervals` 的实现
（code，38 token）、MoE 为什么比同等总量的稠密模型省算力（zh，46 token）；请求走 `/v1/chat/completions`、`cache_prompt: false`（每次全新
prefill，避免前缀缓存污染 tok/s）：

```
llama-server -m Qwen3.6-35B-A3B-UD-Q4_K_M.gguf -c 4096 -b 512 -t 8 -ngl 0 --no-warmup \
             --spec-type draft-mtp --spec-draft-n-max <k>
```

指标取 server 日志的 `draft acceptance = x (a accepted / g generated), mean len = y`（
`mean len = 1 + accepted / verify_steps`），只统计 3 个正式请求（跳过 task 0 的 warmup）：

| k | accept_rate（a/g） | mean len | per-prompt acc（prose / code / zh） | per-prompt mean len | gen tok/s |
| ---: | ---: | ---: | --- | --- | ---: |
| 1 | 0.9479（91/96） | 1.95 | 1.000 / 0.938 / 0.909 | 2.00 / 1.94 / 1.91 | 8.95 |
| **2** | **0.9098**（121/133） | **2.81** | 0.932 / 0.909 / 0.889 | 2.86 / 2.82 / 2.74 | **10.54** |
| 3 | 0.8590（134/156） | 3.54 | 0.882 / 0.882 / 0.815 | 3.65 / 3.65 / 3.32 | 9.80 |
| 4 | 0.7933（142/179） | 4.05 | 0.907 / 0.758 / 0.730 | 4.50 / 3.94 / 3.71 | 9.68 |
| 6 | 0.6773（149/220） | 4.84 | 0.750 / 0.628 / 0.662 | 5.25 / 4.77 / 4.50 | 8.54 |
| 8 | 0.5667（153/270） | 5.48 | 0.761 / 0.537 / 0.462 | 7.00 / 5.25 / 4.20 | 8.84 |

出处：`build-split/acc/official_clean_k{1,2,3,4,6}.log` 与 `official_k8.log`（13:31–13:36），每行 3 条
`draft acceptance`（task 编号逐 k 不同；同一 k 的三条就是 prose / code / zh）。`gen tok/s` 是三个 prompt 的 `gen_tps` 均值（
`official_clean_k2.json` 里 8.39 / 11.24 / 11.99 → 10.54）。

**这一列受同机负载影响**，只在同一时段内相对可比：空闲时整模 k=2 单 prompt 到 13.99–16.86 tok/s（这一对只在当天的测量记录里，`acc/*.json` 复算不到，按「未落盘」
看），表里的 10.54 是有背景负载时的数。

**k 的甜点仍是 2**：k=2 接受率还有 0.910（mean len 2.81，上限 3），k=3 起每多一个 draft 就多一份 verify 成本，而接受率掉到 0.859、k=4 的 mean
len 4.05 只换来 0.79。**这个结论与模型无关，换模型不必重扫 k**：官方与蒸馏两版都落在 k=2（蒸馏版 k=2 是 0.699 / 2.39，k=4 掉到 0.545，k=8 掉到
0.321，衰减更快，更不可能往大 k 走）。甜点由「每轮 draft 的边际成本 / 每轮 verify 的边际收益」的比值决定（CPU 上 verify 是 k+1 行的批前向，draft 是 k 次单行前
向），这个比值对模型不敏感；只有当接受率水平整体移动（例如将来换更好的 MTP 头）时才需要重扫。

## 11.3 官方 vs 蒸馏：差的是 hidden 分布的一致性

同 prompt、同代码、同机：

| k | 官方 acc / mean len | 蒸馏 acc / mean len | accept_rate 提升 | gen tok/s（官方 / 蒸馏） |
| ---: | --- | --- | ---: | --- |
| 2 | 0.910 / 2.81（121/133） | 0.699 / 2.39（109/156） | **+30%** | 10.54 / 9.17 |
| 4 | 0.793 / 4.05（142/179） | 0.545 / 3.18（128/235） | **+46%** | 9.68 / 9.29 |
| 8 | 0.567 / 5.48（153/270） | 0.321 / 3.40（132/411） | **+76%** | 8.84 / 6.11 |

出处：官方见 §11.2 的日志；蒸馏是 `build-split/acc/dist_k{2,4,8}.log`（13:21–13:22）。蒸馏的 per-prompt acc：k=2 是 0.596 /
0.692 / 0.830，k=4 是 0.435 / 0.538 / 0.708，k=8 是 0.278 / 0.303 / 0.398。tok/s 两列都在 13:12–13:36 交错跑出来，**相对可
比、绝对值仍受同机负载影响**：k=2 是 10.54 vs 9.17、k=8 是 8.84 vs 6.11，而 k=4 只差 4%（9.68 vs 9.29）——那一档差距不成比例，**不要拿单档
tok/s 反推接受率**（k=4 官方 9.68 甚至比 k=3 的 9.80 低）。

per-position 接受率（k=8，`-lv 4` 日志里的 `acc per pos`，pos0..pos7）：

| prompt | 官方 | 蒸馏 |
| --- | --- | --- |
| prose | 0.889 0.889 0.889 0.889 0.778 0.667 0.667 0.333 | 0.762 0.476 0.333 0.190 0.095 0.048 0.048 0.048 |
| code | 0.917 0.750 0.667 0.583 0.417 0.333 0.333 0.250 | 0.789 0.632 0.421 0.158 0.105 0.053 0.053 0.053 |
| zh | 0.800 0.667 0.400 0.333 0.333 0.333 0.267 0.067 | 0.875 0.688 0.438 0.312 0.188 0.188 0.188 0.062 |

出处：`build-split/acc/official_pos_k8.log`、`dist_pos_k8.log`（两份都是 3 条正式请求的 `acc per pos` 行；task 0 的 warmup
行是 `1.000, 1.000, 0.000...`，别数进去）。

**读法：pos0 两边接近，pos1 起分叉，之后蒸馏掉得很快。** 逐位看：「pos2 之后 0.16–0.44」这个概括对应 pos2 的 0.33–0.44 与 pos3 的 0.16–0.31；
pos4 之后再降到 0.05–0.19。pos0 接近说明蒸馏版不是「整体变笨」——它靠局部特征还能猜中第一个草稿；差的是 **hidden state 分布的一致性**，所以 draft 走得越远（k 越
大）误差累积越多，官方与蒸馏的差距也越大：k=2 +30%、k=8 +76%。**推论：蒸馏版的任何接受率都不能拿来做容量规划**，它是另一个分布，不是「同一模型慢一点」。

## 11.4 链上接受率低 13%（旧路径 24%）：是我们自己加的代码

本地双段链（`--split-layer-range 0:32 --split-tail-range 32:40`，官方模型，
`--spec-type draft-mtp --spec-draft-n-max 2`）与整模同 prompt、同机（T=0、3 prompt、`verify/acc_test.py`、
`-c 4096 -ngl 0`）：

| 配置 | 64 token 生成 | 512 token 生成 |
| --- | ---: | ---: |
| 旧路径（每轮清空草稿 KV） | 0.6879（108/157） | 0.5823（824/1415） |
| keep-KV（2026-11-10 起默认） | 0.7877（115/146） | 0.7236（906/1252） |
| keep-KV + catch-up mode 1（仅 prefill） | 0.8551（118/138） | 0.7395（914/1236） |
| **keep-KV + catch-up mode 2（每批）** | **0.9098（121/133）** | **0.7557（922/1220）** |
| 整模对照（同机、同 prompt） | 0.9098（121/133） | 0.7557（922/1220） |

三次独立重跑逐位相同（每格三次同一个分数）；mode 2 与整模连每条 prompt 的 accepted/generated 都一样。**512 档的 0.7557 是模型在长生成上的真实上限，不是链的
**——整模自己也只有 0.7557。出处：`build-split/acc/chain_off_k2c.json`（旧路径，64 档）与
`H:/llamacpp-c/.qlh_tmp/catchup/acc_{keepkv,catchup1,catchup2,whole}_r{1,2,3}_n{64,512}.json`（2026-11-11
）。

**根因在 `common/speculative.cpp` 的两处，都是本仓加的**（测量时 HEAD `19776b74e`；`c5bd787b` 加的）：① `process()` 的 catch-up
decode 条件 `if (!is_mem_shared && !is_chain && !is_remote)`（上游只有 `!is_mem_shared`）；② `draft()` 在 chain 下每轮
清空草稿 KV。两处自洽成一个循环：不 catch-up → 草稿没有 prompt 的 KV → 必须每轮清空 → 即使 catch-up 也读不到。

qwen35moe 的 NextN 块带 attention（`cache_k_l8` / `FLASH_ATTN_EXT`，见 `build-split/shape_tail.log`），所以「草稿看不到
prompt 的 KV」是真实差异，不是注释说的「2B 上 70–93% 就够了」——那正是它被小模型掩盖的证据。

**keep-KV（2026-11-10 起默认，`QLH_CHAIN_DRAFT_KEEP_KV=0` 关）留的是生成历史**：`begin()` 每请求清一次；`draft()` 每轮按本轮 `pos0`
做 per-seq `seq_rm` 代替每轮 `clear`；server 在 draft 之后的那次 `seq_rm(ckpt.pos_max+1)` 跳过。兜底：`accept()` 记下下一轮应有的
`pos0`（`pos0 + n_accepted + 1`），draft 入口对不上就清空。零额外前向、零传输。位置稀疏**实测过**（`.qlh_tmp/pos_sparse_probe.cpp`，2B
与 35B）：稀疏位置、`seq_rm` 后重写、分批续写与一次喂逐字节相同（maxdiff 0）；`pos 0..N-1` 空着不影响正确性，KV 按 cell 数计（`n_ctx=1024` 时能用到
pos=1033）。

**catch-up（`QLH_CHAIN_DRAFT_CATCHUP=1` 仅 prefill，`=2` 每批；默认关）补的是 prompt**：本地链的 tail 是同进程 ctx（`ctx_tgt`），非
masked `embeddings_nextn` 一行一个 server batch 行，batch 仍带 token id——上游非 chain 分支的构造照抄即可（位置 i 的 token + 位置 i−
1 的 h、每 seq 第一行用 `pending_h`、每 MTP layer 一次 `llama_process`），位置用 tail 的真实 pos，不重排。mode 1 只填 prompt，生成阶段仍是
草稿自己的 h；mode 2 每批都填，被接受的行拿到 tail 的真实 h。

**`begin()` 的时机**：它原来在 `SLOT_STATE_DONE_PROMPT`（prompt 处理完之后）调用，而 prompt 的 `process()` 批在那之前——catch-up 写了
行、`begin()` 随后把它清掉，mode 1 等于没开（实测：开与不开接受率完全相同）。现在移到 `batch_building` 的 `SLOT_STATE_STARTED` 分支（prefill 之
前），语义不变（清该 seq 的草稿 KV、reset 草稿 sampler），窗口才打开。

**`keep_kv_expected` 的初值**由 catch-up 在 prefill 写完后设成「最后写入位置 + 1」，第一轮 draft 用它判断续写；生成阶段由 `accept()` 独占维护（
mode 2 的批末尾会越过被拒的草稿行，catch-up 不覆盖它——写错会让 mode 2 掉到 0.70，实测过）。失败只降级：`llama_process` 返回非 0 时清该 seq 的草稿 KV
并 WARN，请求继续——草稿是加分项，不是必需品。

**成本（安静机器，A/B 交替两轮；`B−A` = 整模+MTP 减整模不投机的 `prompt_ms`）**：N=100/500/2000 = **+2215 / +2100 / +4467 ms**，两
轮同号（上一轮的 +1013/+936/−1679 作废：19% 机器漂移，A 的 2000 从 55 s 掉到 46 s，压过了真实值）。消掉 draft ctx 的固定建仓（~2.1 s，
`graphs reused=0`，keep-KV 路径本来也要付）后每行 **1.19 ms**，2000 行 ≈ 2.4 s；生成侧 mode 2 每轮多一次 1 行 decode（~5–10 ms 调度开
销）。收益：64 档 `gen_ms` −4.2%（mode 1）/ −8.8%（mode 2）；512 档 −0.3% / +5.5%（mode 2 的每轮开销与轮数减少相抵，落在噪声内——512 档真正值
钱的是 mode 1 的 0.7395 与 mode 2 的 0.7557，两者生成耗时都等于 keep-KV）。

**自适应默认（已实现）**：成本 `N_prompt × 1.19 ms`，收益是轮数减少 × 每轮时间（64 档每输出 token 约 3 ms，512 档约 0.2 ms）。规则：
**`N_prompt > 3 × n_predict` → 不做；否则 `n_predict ≤ 128` →mode 2；否则 → mode 1**；`n_predict` 为 -1/-2（无限）时不套 3×
公式，直接走后两条。**这是唯一的行为变化：以前「未设置」等价于 0，现在等价于逐请求自适应；显式 `QLH_CHAIN_DRAFT_CATCHUP=0/1/2` 一旦设置就完全接管**，`=0` 与未设置靠
`catchup_mode_env()` 的 -1 区分。

判定在 server 侧 `batch_building` 的 `SLOT_STATE_STARTED` 分支（两个量都在手边），结果写进 `QLH_CHAIN_DRAFT_CATCHUP_AUTO`；spec
侧 `begin()` 按「显式变量 > AUTO > 关」取 per-seq mode，接口一个字节没动（沿用环境变量这条已有通道，`common_set_env`/`common_get_env` 是仓库
自带的跨平台封装）。命中时 server 无条件打
`[QLH-SPLIT] chain draft catch-up: adaptive mode N (N_prompt = …, n_predict =…)`。**只对本地链有效**：remote 链（
`--split-remote`）的草稿在 stage 里跑，catch-up 整条关闭，实测该拓扑上曾谎报 mode 2 而接受率仍是 0.65–0.72 档，现已加
`ctx_tail != nullptr` 条件。

实测（本机 `0:32+32:40`、T=0、k=2、3 prompt、跳过 warmup；日志 `.qlh_tmp/acc/logs/`）：

| 场景 | N_prompt | n_predict | mode | accept_rate | mean_len | gen tok/s |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| 短 prompt + 短生成 | 39/38/46 | 64 | **2** | **0.9098**（121/133） | 2.86/2.82/2.74 | 9.09/11.33/11.07 |
| 短 prompt + 长生成 | 39/38/46 | 512 | **1** | **0.7355**（912/1240） | 2.40/2.67/2.37 | 11.32/11.92/11.20 |
| 长 prompt + 短生成 | 2019/2010/2044 | 64 | **0** | **0.7467**（112/150） | 2.30/2.74/2.48 | 11.67/12.99/11.37 |
| 显式 `=0`（覆盖） | 39/38/46 | 64 | 0（强制） | **0.7877**（115/146） | 2.52/2.58/2.62 | 7.42/11.26/11.73 |

mode 2 与整模逐位相同；mode 1 对得上旧批次 0.7395（同档 2026-11-11 新批次 0.7355，分母差 4 token）；显式 `=0` 与 keep-KV 档逐位相同（**覆盖优先**）。长 prompt 那行低于 0.7877 是内容差异（同 prompt 强制
mode 2 是 0.882），两次重跑逐位相同；`prompt_ms` mode 0 是 49–60 s、强制 mode 2 是 55–62 s（方向对，2.4 s 被漂移淹掉）。

## 11.5 「代码回归」与「新模型验收」是两件事

| 入口 | 测的是 | 结果 | 日志 |
| --- | --- | --- | --- |
| `bash verify/verify.sh --skip-build` | **代码没退化**（2B，24 层） | **53/53 PASS，exit 0** | `build-split/verify_catchup_default.log` |
| `bash verify/verify_mtp.sh` | 链上 MTP 机制（2B，tail `20:24`） | **6/6 PASS，exit 0** | `build-split/verify_mtp_catchup.log` |
| 本地双段链 vs 整模（35B 官方） | **形状对得上 + 数值正确** | **逐字节相同**（4 配置 × 3 重跑，64/512 各 3 prompt） | §11.4 的 `acc_*.json` |
| 形状复位验（split-stage tail `32:40`、整模 `--spec-type draft-mtp`） | 新模型的形状 | 通过 | `build-split/shape_tail.log`、`acc/official_k2.log` |

形状复位验的具体输出：tail 段报 `layers [32, 40) of 40, role=tail, n_embd=2048, n_rs_seq=8`、
`MTP draft context: 1 NextN block(s), n_ctx=4096, own KV, n_outputs_max=1; OP_MTP_DRAFT enabled`、
`identity: ... has_nextn=1`；整模 `--spec-type draft-mtp` 加载 **17.63 s**、日志有
`creating MTP draft context against the target model`。

`verify/verify_mtp.sh`：tail 报 `has_nextn=1`、链上 5/5 行 logits 逐位、wire 上 `h_nextn` 逐位、3 个 draft 逐个相同（11751 /
13 / 271）、tail 日志确认跑过 `OP_MTP_DRAFT`、协议版本一致；`verify/verify.sh` 的逐组结果见上表与 `verify/README.md`；2026-11-11 的
52/52 与 catch-up 开（`QLH_CHAIN_DRAFT_CATCHUP=1`）的组 1-8 22/0（+组 9/10 SKIP）、`verify_mtp.sh` 6/6、
`verify_grammar_spec.sh` 23/0、`verify_rbudget_spec.sh` 36/0、`verify_multislot.py --np 2` 10/10 都在
`build-split/verify_*catchup*.log`。

**为什么这一步必须用 2B**：`verify/verify.sh` 组 1 的断言按 24 层模型写死（`0:20+20:24`、`0:8+8:16+16:24`、hidden 2048）；
`verify/verify_mtp.sh` 要求 tail `20:24` 恰好等于模型末层，而 `--role tail` 的硬性校验是 `layer_hi == trunk depth`。对 40 层的
35B 新模型，`20:24` 只是 relay 段，这两条结构上都不成立。所以 2B 跑的是「代码没退化」，不是「新模型能用」；新模型的分段 / MTP 验收只能走「本地双段链 `0:32+32:40`
vs 整模、逐字节一致」这条路。

**闭环结论**：换模型不影响分段链与 MTP 的数值正确性（2B 24/24 + 6/6 回归，当时套件 24 项；35B 链 vs 整模逐字节一致）；影响只在接受率——整模上是 trunk 漂移（0.910 vs 蒸馏
0.699），链上原本是草稿 KV 的差异（0.910 → 0.788，旧路径 0.688），**catch-up 每批（mode 2）把这个差异也消掉了：0.910，与整模逐位相同**（mode 1 仅
prefill 是 0.855，默认关，§11.4）。

## 11.6 复现脚本与日志

- `verify/acc_test.py --port <p> --out <json>`：发 3 个 prompt（prose / code / zh），存 timings +全文；
  `build-split/ksweep.sh <tag> <model> <k1,k2,...> <port>`：串行重启 server、逐 k 测、抓 `draft acceptance`；
  per-position 要额外给 `LV="-lv 4"`。
- `verify/parse_pos.py <log>…`：从 `-lv 4` 的日志抓 per-position 接受率（默认扫 `build-split/acc/*.log`）。日志全在
  `build-split/acc/`：官方 k 扫描 `official_clean_k*.log`、蒸馏 `dist_k*.log`、per-position
  `{official,dist}_pos_k8.log`、链上 `chain_off_k2{,c}.log`。
- 验收 / 回归日志：`build-split/verify_catchup_default.log`（52/52）、`verify_catchup_fast.log`（组 1-8，catch-up=1）。
- catch-up 的三个工具（都在 `H:/llamacpp-c/.qlh_tmp/`）：`acc_catchup.py`（4 配置 × 3 重跑 ×64/512，抓 `draft acceptance`
  ）、`bytecmp.py`（跨配置/重跑逐字节比 `acc_*.json` 的 `text`，即 reasoning+content）、`catchup_ab.py`（成本 A/B 交替两轮，读
  `timings.prompt_ms`）。

## 11.7 旧数字归谁

| 旧数字 | 在哪 | 属于谁 |
| --- | --- | --- |
| MTP 1.13–1.25×、接受率 70%（v2 链） | [`SPECULATION.md`](SPECULATION.md) §4 / §10 / §11 | 蒸馏版 + v2 的 Python 驱动链，**不是**本节的 v2.1 链 |
| DFlash 1.5×、接受 42–95% | [`SPECULATION.md`](SPECULATION.md) §5 | 旧模型（且是 `llama-speculative-simple` 的 40 层 target 口径） |
| DFlash k=3 的 1.12×、MTP k=2 的 1.01× | [`SPECULATION-DATA.md`](SPECULATION-DATA.md) §2 | 蒸馏版（`.qlh_tmp/casc_probe.sh` 的 `TGT` 写的就是蒸馏版），换模型未重测 |
| 远程链 MTP k=2 中位 +6% | [`CROSS-MACHINE.md`](../deploy/CROSS-MACHINE.md) §1 | 蒸馏版（ROG 那组测量在官方模型到位之前）。官方版接受率高 30–76%，收益预计更高，**但未重测** |
| 2B 的 MTP / `seq_rm` 数字 | [`SPECULATION-DATA.md`](SPECULATION-DATA.md) §1、[`PROTOCOL-V9.md`](../design/PROTOCOL-V9.md) §5 | 2B，与换模型无关 |

**不要在两套数字之间做换算**：能做的只有「同一模型、同一口径的 A/B」；跨模型只能比接受率本身，且要说清哪个模型。

## 11.8 跨机链的投机吞吐：三种拓扑，结论按「两端谁更弱」翻号

上面几节都是**整模**或**两段本地链**的口径。跨机链要另算一笔账：draft 在本机 head 上跑，但每轮
verify 的 logits 要走 `993 KB/轮`（整行）而不是候选通道的 `48 B/轮`，所以「投机能不能赚」取决于
省下的前向能不能盖过这条带宽。

实测（941 token prompt、`max_tokens=64`、T=0、`--spec-draft-n-max 2`、单个 prompt，不是三 prompt
均值；口径与 §11.2 的整模数据不可直接比）：

| 拓扑 | 不投机 | DFlash | DSpark |
| --- | ---: | ---: | ---: |
| 本机 → Surface（`0:32` + `32:40`） | 10.77 | 6.67（**−38%**） | **12.36（+15%）** |
| 本机 → Y700（`0:32` + `32:40`） | 7.51 | 8.93（**+19%**） | **11.35（+51%）** |
| 三段（本机 `0:32` + Surface relay `32:36` + Y700 tail `36:40`） | 9.09 | 6.25（**−31%**） | 9.05（−0.4%） |

**口径标注（两批，别混用）**：本表是 **941-token** 口径（DSpark accept **0.932**），Y700 那格给出 **+51%**（11.35 vs 7.51）；
**1522-token** 口径见 [`CROSS-MACHINE.md`](../deploy/CROSS-MACHINE.md) §7.1（accept **0.740**，**+12.4% ~ +17.2%**）。
两批的差异来自 prompt 长度导致的接受率差，不是代码差异。按 [`INDEX.md`](INDEX.md) 的现行值取 **1522-token 批的 +12~17%**，
本表 941-token 批的 **+51% 是历史值**。三段那行的 9.09 是 `-t 14` 的旧测；2026-11-12 的现行值是 **5.85 t/s**
（`0:24` / `24:32` / `32:40`、`-t 8`，见 [`INDEX.md`](INDEX.md) 第一节），与本节不是同一拓扑。**另一批测量**（1522 token、
`0:20` / `20:28` / `28:40`）的三段链是 **6.53 / 6.94 t/s**，见 [`CROSS-MACHINE.md`](../deploy/CROSS-MACHINE.md) §7.2；
它和本节的 9.09、5.85 都不可互相换算——prompt 长度与切点都不同。

同一批测量里三种链形态给出的 `chain_kind` 一致：DFlash **0.848**（39/46，mean len 2.70）、DSpark **0.932**（41/44，mean len 2.86）。
这一点本身是判据——它说明收敛后的 `chain_kind` 在三种链形态上给出了同一个答案。**接受率随 prompt 内容与长度变**
（0.932 @941 token vs 0.740 @1522 token），不随拓扑变。DSpark 那 6 次重跑的接受率逐位相同（`0.93182 / 41 accepted / 44 generated`），
所以下面的速度差不是「有时投机没生效」。

**Surface 那格的 12.36 是 6 次重跑的中位数**（稳态 5 次：12.04 / 12.25 / 12.36 / 13.07 / 13.45，
首跑 10.75 是 warmup）。单次测量会给出误导性结果：同一配置在不同时刻量到过 2.27 与 7.54——两次的
接受率同样是 0.93182，所以那是 Surface 侧的缓存与网络状态，不是代码。**跨机数字必须多跑取中位**，
一次不够。

**三个反直觉的点**：

1. **DFlash 在 Surface 上 −38%、在 Y700 上 +19%**。同样接受率，差别在 host copies 的相对开销：
   `device buffer unusable on this chain: it holds the head's 6 of 8 extract layers` 让它每轮都
   拷 hidden，而 Surface 的 CPU 比 Y700 快得多，这份固定开销相对就压过了 draft 的收益。
2. **DSpark 在 Y700 上 +51%，是这三组里最大收益**。Y700 单 token 慢（不投机 7.51），每省一次
   前向的相对收益就更大。**越弱的机器越吃投机**——这条以前只有整模数据，现在跨机链上也有了。
3. **三段链的投机几乎全废**（DSpark −0.4%）：两跳让 hidden 过网两次，draft 省下的那点前向被
   多出来的一跳吃掉。

**所以 §11.4 那条「长 prompt + DSpark 才转正」要补一个前提**：它成立的前提是**两段**。tail 换成
Surface 这种强机，同一套 DSpark 只剩 **+15%**（比 Y700 的 +51% 少，但仍是正的）；链变三段，
连这点也没了。

**没测成的一种**：Surface 当 head + Y700 当 tail。Surface 可用 7.8 GB、Y700 可用 8.3 GB，而
head `0:32` 要 15.69 GiB、tail `32:40` 要 4.90 GiB；把 head 缩到 `0:16`（7.8 GiB）又让 tail 变成
`16:40`（12 GiB），它也只有 8.3 GB。任何切法都至少有一台装不下，所以这个拓扑在本轮硬件上不存在。

**测量装置**：`llama-server` 的 `/v1/chat/completions` + 响应里的 `timings.predicted_per_second`；
stage 侧 `split-stage --host 0.0.0.0 --port ... -c 4096 -b 2048 -ub 2048 --rs-seq 8`（**`-b` 必须
≥ head 的 `n_batch`，否则长 prompt 报 `stage 1 failed the hop (rc=1)`**——这个坑本轮踩过一次）。
