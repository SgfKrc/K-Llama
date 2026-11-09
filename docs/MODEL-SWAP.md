# 换模型：从蒸馏版到官方版（2026-11-08）

> 状态：**现行**
> 更新日期：2026-11-09
> 适用范围：35B target 从蒸馏版换到官方 `UD-Q4_K_M` 的全部结论——两份权重的 metadata、形状
> 核对、接受率基线、官方 vs 蒸馏的差距、链上 24% 折扣的根因，以及「代码回归」与「新模型验收」的
> 分界。数字原样搬进仓库，出处逐条注明；`build-split/accept_baseline.md` 仍是原始记录，两份不
> 一致时以日志为准（下文标出两处小出入）。投机矩阵与 k 扫描归
> [`SPECULATION-DATA.md`](SPECULATION-DATA.md)，协议归 [`PROTOCOL-V9.md`](PROTOCOL-V9.md)。

## 11.1 两份权重、为什么换

| | 旧（**已弃用**） | 新（**现行**） |
| --- | --- | --- |
| 文件 | `Qwen3.6-35B-A3B-Claude-4.7-Opus-Reasoning-Distilled-APEX-MTP-I-Quality.gguf` | `Qwen3.6-35B-A3B-UD-Q4_K_M.gguf` |
| 来源 | lordx64 蒸馏 + **mudler 的 "apex" 工作流**做的混合量化（Q6_K 主 + Q8_0 nextn） | `unsloth/Qwen3.6-35B-A3B-MTP-GGUF` 的 `UD-Q4_K_M`（官方 MTP 仓） |
| 体积 | 23.48 GB（21.87 GiB） | **21.11 GiB** |
| 本机路径 | `H:/qlh_models/qwen3-6-35b-a3b-opus47-distill-apex-mtp/` | `H:/qlh_models/qwen3-6-35b-a3b-mtp-official/` |
| `general.name` | **`Safetensors`**（转换时没传 `--model-name`） | `Qwen3.6-35B-A3B` |
| 其他遗留 metadata | `general.dataset.0` 指向 `Lordx64 / Reasoning Distill Opus 4 7 Max Sft`；`quantize.imatrix.file` 是构建机的 `/home/ubuntu/work/apex/...`，`quantize.imatrix.dataset` 是 `/home/mudler/autoresearch-quant/...` | `quantize.imatrix.file` 是 `Qwen3.6-35B-A3B-GGUF/imatrix_unsloth...`，`quantize.imatrix.dataset` 是 `unsloth_calibration_...` |

出处：旧模型 metadata 在 `build-split/acc/dist_pos_k8.log` 的 `llama_model_loader` 段，新模型在
`acc/official_pos_k8.log` 的同一位置。体积以两份日志的 `print_info: file size` 为准：官方 **21.10 GiB（5.10
BPW）**、蒸馏 **21.86 GiB（5.29 BPW）**；上表的 21.11 / 21.87 与 `accept_baseline.md`
一致，是同一份文件的四舍五入差（0.01 GiB）。「Q6_K 主 + Q8_0 nextn」与 mudler 的 apex 工作流来自该 GGUF
的模型卡，不是日志——日志只能证到 `general.dataset.0` 指向 lordx64 的蒸馏仓、以及 5.29 BPW 比官方的 5.10 BPW
更高（与「主权重放大到 Q6_K」一致）。

> **与 `accept_baseline.md` 的一处出入**：那份记录写 `quantize.imatrix.file =
> /home/mudder/autoresearch-quant/...`。日志里这是两条不同的 key——`imatrix.file` 指
> `/home/ubuntu/work/apex/...`，`/home/mudler/autoresearch-quant/...` 是 `imatrix.dataset`（作者名
> 是 `mudler`，不是 `mudder`）。本节按日志写。

**新模型的形状与代码假设全部吻合**（`llama_model_loader` / `print_info` 段）：

| key | 值 | key | 值 |
| --- | --- | --- | --- |
| `general.architecture` | `qwen35moe` | `qwen35moe.ssm.state_size` | **128** |
| `general.name` | `Qwen3.6-35B-A3B` | `qwen35moe.ssm.group_count` | **16** |
| `qwen35moe.block_count` | **41** | `qwen35moe.ssm.conv_kernel` | **4** |
| `qwen35moe.nextn_predict_layers` | **1** | `qwen35moe.full_attention_interval` | **4** |
| `qwen35moe.ssm.inner_size` | **4096** | `qwen35moe.expert_count` / `expert_used_count` | **256** / 8 |

trunk depth 读作 **40**（41 block − 1 个 NextN），与 `n_layer_nextn == 1` 的断言一致；所以在 tail 段起 `32:40` 时
`--role tail` 的 `has_nextn=1` 成立、MTP draft context 建得起来。**换模型零改
代码**——形状这一层没有任何要适配的东西。**头的来源也核对过**：DFlash 头、DSpark 头、NextN 层都
与官方文件做过**逐字节**比对（sha256 / Hugging Face LFS oid），三者都是官方的，不是「自己拼的
兼容版」；这把「接受率退化」的候选原因砍掉了一类——但它是当日人工比对、`build-split/` 下没有
日志，记为「未落盘」（[`SPECULATION-DATA.md`](SPECULATION-DATA.md) §5）。

## 11.2 接受率基线（官方模型，整模，MTP）

口径：3 个真实 prompt（英文 prose / Python 代码 / 中文技术），`temperature 0`、`n_predict 64`、 `-c 4096 -b 512 -t 8
-ngl 0`、纯 CPU。prompt 原文在 `build-split/acc_test.py` 的 `PROMPTS`：B-tree 插入怎么保持平衡（prose，39
token）、`merge_intervals` 的实现（code，38 token）、MoE 为什么比同等总量的稠密模型省算力（zh，46 token）；请求走
`/v1/chat/completions`、`cache_prompt: false` （每次全新 prefill，避免前缀缓存污染 tok/s）：

```
llama-server -m Qwen3.6-35B-A3B-UD-Q4_K_M.gguf -c 4096 -b 512 -t 8 -ngl 0 --no-warmup \
             --spec-type draft-mtp --spec-draft-n-max <k>
```

指标取 server 日志的 `draft acceptance = x (a accepted / g generated), mean len = y` （`mean len = 1 + accepted /
verify_steps`），只统计 3 个正式请求（跳过 task 0 的 warmup）：

| k | accept_rate（a/g） | mean len | per-prompt acc（prose / code / zh） | per-prompt mean len | gen tok/s |
| ---: | ---: | ---: | --- | --- | ---: |
| 1 | 0.9479（91/96） | 1.95 | 1.000 / 0.938 / 0.909 | 2.00 / 1.94 / 1.91 | 8.95 |
| **2** | **0.9098**（121/133） | **2.81** | 0.932 / 0.909 / 0.889 | 2.86 / 2.82 / 2.74 | **10.54** |
| 3 | 0.8590（134/156） | 3.54 | 0.882 / 0.882 / 0.815 | 3.65 / 3.65 / 3.32 | 9.80 |
| 4 | 0.7933（142/179） | 4.05 | 0.907 / 0.758 / 0.730 | 4.50 / 3.94 / 3.71 | 9.68 |
| 6 | 0.6773（149/220） | 4.84 | 0.750 / 0.628 / 0.662 | 5.25 / 4.77 / 4.50 | 8.54 |
| 8 | 0.5667（153/270） | 5.48 | 0.761 / 0.537 / 0.462 | 7.00 / 5.25 / 4.20 | 8.84 |

出处：`build-split/acc/official_clean_k{1,2,3,4,6}.log` 与 `official_k8.log`（13:31–13:36），每行 3 条 `draft
acceptance`（task 编号逐 k 不同；同一 k 的三条就是 prose / code / zh）。`gen tok/s` 是三个 prompt 的 `gen_tps`
均值（`official_clean_k2.json` 里 8.39 / 11.24 / 11.99 → 10.54）。**这一列受
同机负载影响**，只在同一时段内相对可比：空闲时整模 k=2 单 prompt 到 13.99–16.86 tok/s（这一对只
在当天的测量记录里，`acc/*.json` 复算不到，按「未落盘」看），表里的 10.54 是有背景负载时的数。

**k 的甜点仍是 2**：k=2 接受率还有 0.910（mean len 2.81，上限 3），k=3 起每多一个 draft 就多一份 verify
成本，而接受率掉到 0.859、k=4 的 mean len 4.05 只换来 0.79。**这个结论与模型无关，换模型不必重扫
k**：官方与蒸馏两版都落在 k=2（蒸馏版 k=2 是 0.699 / 2.39，k=4 掉到 0.545，k=8 掉到 0.321，衰减更快，更不可能往大 k
走）。甜点由「每轮 draft 的边际成本 / 每轮 verify 的边际收益」的比值决定（CPU 上 verify 是 k+1 行的批前向，draft 是 k
次单行前向），这个比值对模型不敏感；只有当接受率水平整体移动（例如将来换更好的 MTP 头）时才需要重扫。

## 11.3 官方 vs 蒸馏：差的是 hidden 分布的一致性

同 prompt、同代码、同机：

| k | 官方 acc / mean len | 蒸馏 acc / mean len | accept_rate 提升 | gen tok/s（官方 / 蒸馏） |
| ---: | --- | --- | ---: | --- |
| 2 | 0.910 / 2.81（121/133） | 0.699 / 2.39（109/156） | **+30%** | 10.54 / 9.17 |
| 4 | 0.793 / 4.05（142/179） | 0.545 / 3.18（128/235） | **+46%** | 9.68 / 9.29 |
| 8 | 0.567 / 5.48（153/270） | 0.321 / 3.40（132/411） | **+76%** | 8.84 / 6.11 |

出处：官方见 §11.2 的日志；蒸馏是 `build-split/acc/dist_k{2,4,8}.log`（13:21–13:22）。蒸馏的 per-prompt acc：k=2 是
0.596 / 0.692 / 0.830，k=4 是 0.435 / 0.538 / 0.708，k=8 是 0.278 / 0.303 / 0.398。tok/s 两列都在 13:12–13:36
交错跑出来，**相对可比、绝对值仍受同机负载影响**：k=2 是 10.54 vs 9.17、k=8 是 8.84 vs 6.11，而 k=4 只差 4%（9.68 vs
9.29）——那一档差距不成比例，**不要拿单档 tok/s 反推接受率**（k=4 官方 9.68 甚至比 k=3 的 9.80 低）。

per-position 接受率（k=8，`-lv 4` 日志里的 `acc per pos`，pos0..pos7）：

| prompt | 官方 | 蒸馏 |
| --- | --- | --- |
| prose | 0.889 0.889 0.889 0.889 0.778 0.667 0.667 0.333 | 0.762 0.476 0.333 0.190 0.095 0.048 0.048 0.048 |
| code | 0.917 0.750 0.667 0.583 0.417 0.333 0.333 0.250 | 0.789 0.632 0.421 0.158 0.105 0.053 0.053 0.053 |
| zh | 0.800 0.667 0.400 0.333 0.333 0.333 0.267 0.067 | 0.875 0.688 0.438 0.312 0.188 0.188 0.188 0.062 |

出处：`build-split/acc/official_pos_k8.log`、`dist_pos_k8.log`（两份都是 3 条正式请求的 `acc per pos` 行；task 0 的
warmup 行是 `1.000, 1.000, 0.000...`，别数进去）。**读法：pos0 两边接近，pos1 起分叉，之后蒸馏掉得很快。**
逐位看：「pos2 之后 0.16–0.44」这个概括对应 pos2 的 0.33–0.44 与 pos3 的 0.16–0.31；pos4 之后再降到 0.05–0.19。pos0
接近说明蒸馏版不是「整体变笨」——它靠局部特征还能猜中第一个草稿；差的是 **hidden state 分布的一致性**，所以 draft
走得越远（k 越大）误差累积越多，官方与蒸馏的差距也越大：k=2 +30%、k=8 +76%。**推论：蒸馏版的任何接受率都不能拿来做
容量规划**，它是另一个分布，不是「同一模型慢一点」。

## 11.4 链上接受率低 24%：是我们自己加的代码

本地双段链（`--split-layer-range 0:32 --split-tail-range 32:40`，官方模型，`--spec-type draft-mtp --spec-draft-n-max
2`）与整模同 prompt、同机：

| 配置 | accept_rate（a/g） | mean len | per-prompt acc | gen tok/s |
| --- | ---: | ---: | --- | ---: |
| 整模 + MTP k=2 | 0.9098（121/133） | 2.81 | 0.932 / 0.909 / 0.889 | 15.54（空闲）/ 8.4–12.0（有负载） |
| 双段链 + MTP k=2 | **0.6879**（108/157） | 2.37 | 0.648 / 0.725 / 0.692 | 8.76 |

出处：整模见 §11.2；链上 `build-split/acc/chain_off_k2c.log`（13:29，tag `chain_official_k2`， `chain_off_k2c.json` 的
`gen_tps` 6.75 / 9.40 / 10.12 → 8.76）。链上比整模低 **(0.9098 − 0.6879) / 0.9098 ≈ 24%**。这是本地双段链（进程内
tail）的数；远端链走同一个 `!is_chain && !is_remote` 分支，同样没有 prompt
历史，但远程链的接受率没有单独测（[`CROSS-MACHINE.md`](CROSS-MACHINE.md) §1 的 +6%
是蒸馏版上的端到端收益，不是接受率）。整模那格 tok/s 的「15.54（空闲）」是当天一次空闲单轮测量的数，`acc/`
里没有可复算的 json（同批 `acc_off_k2.json` 的均值是 10.38），按「未落盘」看；有负载那档 8.4–12.0 与 §11.2 的 10.54
同量级。

**数值正确性没有退化**：3/3 prompt、64 token 与整模**逐字节相同**（含 `reasoning_content`， 250 / 244 / 249
字符）。可比的是 `acc/official_clean_k2.json` 与 `acc/chain_off_k2c.json` 的 `text`
字段——三行两两相等、长度也相等。所以不是「链上算错了」，只是「链上的草稿更差」。 **根因在 `common/speculative.cpp`
的两处，都是本仓加的**（HEAD `48dfb3ea8`；这段代码与测量时的 `0c5c3bdd` 逐字节相同）：① `speculative::process()` 的
catch-up decode 条件是 `if (!is_mem_shared && !is_chain && !is_remote)`（`L2025`），上游 `f477f795c` 只有 `if
(!is_mem_shared)`——`&& !is_chain && !is_remote` 是 `c5bd787b` 新加的；② `speculative::draft()` 在 chain 下每轮清空草稿
KV：`if (is_chain && !is_mem_shared) llama_memory_clear( llama_get_memory(ctx_dft), true);`（`L2228`，注释在
`L2218–2227`）。

`draft()` 里那段注释就是理由原文（`L2152–L2159`）：

> the draft never saw the prompt (process() skips the catch-up), so a round must not inherit rows
> from whatever the previous round / previous request left at the same positions -- that is another
> token's state, and it would be read silently. One clear per round costs nothing (the draft KV is
> k+1 rows per sequence) and is far cheaper than replaying the prompt into the draft. Measured
> acceptance on the 2B model stays in the 70-93% range v1 reported.

**这是一个自洽的循环依赖，不是漏判**：

```
不 catch-up（链上补历史要整段 prompt 额外过一遍 draft）
  ↓ 所以 draft 没有 prompt 的 KV
  ↓ 所以 必须每轮清空（否则继承到「另一个 token 的状态」，静默读脏）
  ↓ 所以 即使 catch-up 了也读不到
  ↓ 所以 不做 catch-up 是"对的"
```

在当前这套草稿 KV 策略下，catch-up 建出来的 prompt 历史没有任何读者，跳过它正确且必要；代价是链上 MTP 接受率封在
0.688（k=2），而不是整模的 0.910。**这是架构边界，不是 bug。** 注释最后那句「Measured acceptance on the 2B model stays
in the 70-93%」正是它被小模型掩盖的证据：2B 上可接受（70–93%），到 35B 掉到 0.688 才暴露。qwen35moe 的 NextN 块带
attention（MTP 图里有 `cache_k_l8` / `FLASH_ATTN_EXT`，见 `build-split/shape_tail.log` 的 `mtp_tok_embd-8` /
`mtp_h_input` / `mtp_shared_head_norm` 几行），所以「草稿看不到 prompt 的 KV」是真实差异，不是理论洁癖。

**如果要做 catch-up，代价与收益（全部是估算，没有实测；代码未改）**：

| 场景 | 净收益 |
| --- | --- |
| 短 prompt + 短输出（40 prompt token / 64 生成） | **≈ 0**（catch-up 0.5–1.5 s，省 1.2–1.8 s） |
| 长生成（512 token） | **省 10–14 s** |
| 长 prompt（2000 token） | **净负**（catch-up 本身 30–60 s） |

代价是每请求一次 N 行 batch 前向（N = prompt 长度，output head 248k vocab 是大头，∝ N）；收益是 k=2 的接受率 0.69 → 可能
~0.91（若与整模持平）。按 64-token 生成从 ~28 步降到 ~23 步算，省 1.2–1.8 s，所以短 prompt
短输出这一档基本白做。**改动要点（将来要做，不是现在）**：必须**同时** 改两处——catch-up 的 `is_chain` 判断，以及
`draft()` 的 chain 分支不再清空草稿 KV（改成按 target 的 `seq_rm` 严格同步）；只改一处会让「历史建了没人读」或「读到别的
token 的状态」之一重新出现。上游非 chain 路径的 batch 构造（token × h shift-right-1）在链上可以直接复用（`ctx_tgt` 已是
tail ctx，tail 的 `embeddings_nextn` 非 masked 通道正是逐行 h）；远端链还要额外把 prompt 的 token id 送过去（N × 4
B，head 侧有、tail 侧没有）。判断：**值得做，但应做成 opt-in 开关**（按 prompt 长度 / 生成长度启用），风险在 KV
位置对齐与 rs-seq checkpoint 回滚；**不建议现在默认打开**。

## 11.5 「代码回归」与「新模型验收」是两件事

| 入口 | 测的是 | 结果 | 日志 |
| --- | --- | --- | --- |
| `bash build-split/verify.sh --skip-build` | **代码没退化**（2B，24 层） | **24/24 PASS，exit 0** | `build-split/verify_newmodel.log` |
| `bash build-split/verify_mtp.sh` | 链上 MTP 机制（2B，tail `20:24`） | **6/6 PASS，exit 0** | `build-split/verify_mtp_newmodel.log` |
| 本地双段链 vs 整模（35B 官方） | **形状对得上 + 数值正确** | **3/3 逐字节相同** | §11.4 的两份 `.json` |
| 形状复位验（split-stage tail `32:40`、整模 `--spec-type draft-mtp`） | 新模型的形状 | 通过 | `build-split/shape_tail.log`、`acc/official_k2.log` |

形状复位验的具体输出：tail 段报 `layers [32, 40) of 40, role=tail, n_embd=2048, n_rs_seq=8`、 `MTP draft context: 1
NextN block(s), n_ctx=4096, own KV, n_outputs_max=1; OP_MTP_DRAFT enabled`、 `identity: ... has_nextn=1`；整模
`--spec-type draft-mtp` 加载 **17.63 s**、日志有 `creating MTP draft context against the target model`。**与
`accept_baseline.md` 的第二处出入**：那份记 17.9 s，是同一件事的两次计时口径差，这里取日志。`verify.sh` 逐组：组 1
本地链逐位 5/5、组 2 N=1 直通 1/1、组 3 N=2 经 server 1/1、组 4 跨进程 1/1、组 5 多 slot 2/2、组 6 三段跨进程 8/8、组 7
远程链经 server 2/2、组 8 链上 MTP 逐位 2/2、组 9 DFlash 8 路 tap 2/2。`verify_mtp.sh`：tail 报 `has_nextn=1`、链上 5/5
行 logits 逐位、wire 上 `h_nextn` 逐位、3 个 draft 逐个相同（11751 / 13 / 271）、tail 日志确认跑过
`OP_MTP_DRAFT`、协议版本一致。

**为什么这一步必须用 2B**：`verify.sh` 组 1 的断言按 24 层模型写死（`0:20+20:24`、 `0:8+8:16+16:24`、hidden
2048）；`verify_mtp.sh` 要求 tail `20:24` 恰好等于模型末层，而 `--role tail` 的硬性校验是 `layer_hi == trunk depth`。对
40 层的 35B 新模型，`20:24` 只是 relay 段，这两条结构上都不成立。所以 2B
跑的是「代码没退化」，不是「新模型能用」；新模型的分段 / MTP 验收只能走「本地双段链 `0:32+32:40` vs
整模、逐字节一致」这条路。**闭环结论**：换模型不影响分段链与 MTP 的数值正确性（2B 24/24 + 6/6 回归；35B 链 vs
整模逐字节一致）；影响只在接受率——整模上是 trunk 漂移（0.910 vs 蒸馏 0.699），链上是草稿 KV 的差异（0.910 →
0.688），后者是设计选择（§11.4）。

## 11.6 复现脚本与日志

- `build-split/acc_test.py --port <p> --out <json>`：发 3 个 prompt（prose / code / zh），存
  timings + 全文；`build-split/ksweep.sh <tag> <model> <k1,k2,...> <port>`：串行重启 server、逐 k
  测、抓 `draft acceptance`；per-position 要额外给 `LV="-lv 4"`。
- `build-split/parse_pos.py <log>…`：从 `-lv 4` 的日志抓 per-position 接受率（默认扫
  `build-split/acc/*.log`）。日志全在 `build-split/acc/`：官方 k 扫描 `official_clean_k*.log`、
  蒸馏 `dist_k*.log`、per-position `{official,dist}_pos_k8.log`、链上 `chain_off_k2{,c}.log`。
- 验收 / 回归日志：`build-split/verify_newmodel.log`（24/24）、`verify_mtp_newmodel.log`（6/6）。

## 11.7 旧数字归谁

| 旧数字 | 在哪 | 属于谁 |
| --- | --- | --- |
| MTP 1.13–1.25×、接受率 70%（v2 链） | [`SPECULATION.md`](SPECULATION.md) §4 / §10 / §11 | 蒸馏版 + v2 的 Python 驱动链，**不是**本节的 v2.1 链 |
| DFlash 1.5×、接受 42–95% | [`SPECULATION.md`](SPECULATION.md) §5 | 旧模型（且是 `llama-speculative-simple` 的 40 层 target 口径） |
| DFlash k=3 的 1.12×、MTP k=2 的 1.01× | [`SPECULATION-DATA.md`](SPECULATION-DATA.md) §2 | 蒸馏版（`.qlh_tmp/casc_probe.sh` 的 `TGT` 写的就是蒸馏版），换模型未重测 |
| 远程链 MTP k=2 中位 +6% | [`CROSS-MACHINE.md`](CROSS-MACHINE.md) §1 | 蒸馏版（ROG 那轮在官方模型到位之前）。官方版接受率高 30–76%，收益预计更高，**但未重测** |
| 2B 的 MTP / `seq_rm` 数字 | [`SPECULATION-DATA.md`](SPECULATION-DATA.md) §1、[`PROTOCOL-V9.md`](PROTOCOL-V9.md) §5 | 2B，与换模型无关 |

**不要在两套数字之间做换算**：能做的只有「同一模型、同一口径的 A/B」；跨模型只能比接受率本身，且要说清哪个模型。
