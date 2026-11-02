# 草稿源 × 段结构：能配什么

> 状态：**A 二选一 / B 级联已实现并实测**，其余为调研结论
> 更新日期：2026-11-01

回答一个问题：**能不能让每个节点各自投机，按它加载了哪些层选方法。**
答案是可以，但形状不是"每个段都起草 token"。

---

## 1. 三条硬约束

**① 验证器只有一个。** 只有 tail 建 `output`（`config.LOGITS_ROLES` 是 `frozenset({"tail"})`），
非 tail 段在 `stage_worker` 关掉 logits（`skipping logits (hidden-only stage)`），argmax 只在
tail 出。所以**起草可以分散，验证必然收敛到 tail**。

**② 出不了 token 的段等于没在投机。** 出 token 只需要词表投影。loader 的按需 lazy 判据是
写死的：首段建 `token_embd`、末段建 `output`、中段两个都不建（`needs_tok_embd()` /
`needs_output()`，`llama-model-loader.cpp`），**没有任何配置项能改**。想让非 tail 段独立
起草，只有两条路：带一份自带词表的 draft（DFlash 的路子，+1.03 GiB），或者改 loader 给它
加装 `output`（515 MiB，无开关）。

**③ 一个段只能提供"它自己那几层的 hidden"。** `add_layer_inp` / `get_layer_inp` 按**本地
lid** 开槽（`qlh_kh_add_layer_inp` / `qlh_kh_get_layer_inp`，`shim/qlh_keep_head_gpu.c`），
`chain_client.dflash_split_layers` 做的就是层号映射。把中段当"target 第 K 层"用会错——K 必须
落在它自己的 `[lo,hi)` 内，否则直接 `RuntimeError`，不是降级。

---

## 2. 方法按"接触目标模型多深"分类

| 方法 | 需要 target 的什么 | 需要额外训练 |
| --- | --- | --- |
| n-gram / prompt-lookup（arXiv:2309.08168） | **零层**，只要 token 序列 | 否 |
| REST 检索式（arXiv:2311.08252） | **零层**，要一个 token 序列 datastore | 否 |
| MTP / NextN（arXiv:2412.19437） | 末层 hidden + 模型自带的 nextn 头 | 模型自带 |
| DFlash | 若干中间层的**输入** hidden + 独立 draft（含词表） | 是 |
| EAGLE-1/2（arXiv:2401.15077 / 2406.16858） | 倒数第一层 hidden + embedding + lm_head | 是 |
| EAGLE-3（arXiv:2503.01840） | 低/中/高多层 hidden 融合 | 是 |
| Medusa（arXiv:2401.10774） | 末层 hidden + lm_head，挂多个头 | 是 |
| LayerSkip / SpS（arXiv:2404.16710 / 2309.01841） | 前若干层 logits，要 lm_head | LayerSkip 是，SpS 否 |

---

## 3. 段结构 → 能支撑什么

| 段 | 手上有什么 | 能起草的 | 不能（为什么） |
| --- | --- | --- | --- |
| head `0:32` | 0..31 层 hidden、`token_embd`、**无 output** | n-gram / REST；DFlash（采样层 2/7/12/17/23/28 都在本段） | MTP（nextn 只在末段）；Medusa（无 lm_head）；LayerSkip（无 lm_head，且未训练截断层实测 α≤0.08 已否决） |
| tail `32:40` | 32..39 层 hidden、`output`、`blk.40` nextn、**无 token_embd** | MTP（现成）；Medusa（有 last hidden + lm_head） | EAGLE（draft 要 token embedding，本段没建）；DFlash 只能供 33/38 两路，**不能独立起草** |
| mid（如 `8:24`） | 中间层 hidden，**两个词表张量都没有** | n-gram / REST；作为 DFlash 的 hidden 供给方 | **任何要出 token 的方法** |
| 任意段 | — | **n-gram / REST 是唯一"哪段都能跑"的一类** | — |

---

## 4. 会打架吗

**机制上能共存，但收益不相加。** 两个草稿源用的是两套独立的 context：tail 的 MTP 是
`ctx_type = LLAMA_CONTEXT_TYPE_MTP` 的第二个 context（`shim/qlh_keep_head_gpu.c` 里
`mp.ctx_type` 那一段），head 的 DFlash 是另一份 GGUF 的 `draft_ctx`（同一文件），
`draft_*` op 是 role-agnostic 的。所以"同时启用"在协议层就成立——挡住它的是
`chain_client.main()` 里的一行 `SystemExit`，不是能力。

但**两段各自投机不等于两倍加速**：加速来源只有一处——每轮前进多少 token，而两段在同一轮里
各被调用一次，k 行摊薄的是各自的层计算，那是同一笔账的两半。

### 三种形态

| 形态 | 内容 | 状态 |
| --- | --- | --- |
| **A 二选一** | 按负载启用一个：代码走 DFlash k=5，其他走 MTP k=2 或关 | **已实现**：`--dflash` 与纯 MTP `--spec-k>0` 仍互斥（`chain_client.main()` 的同一处校验） |
| **B 级联** | DFlash 起草 → 验证块一次绕链 → 取块内末行 nextn hidden → MTP 起草 ≤N → 追加验证 | **已实现**：`--cascade-mtp N`（默认 0 = 关，`chain_client.main()`） |
| **C 并行双源各自验证** | head 自己起草自己验证 | **不可能**：head 没有 `output` |

形态 B 的实测（本机两段 `topology-local-32-8.toml`，代码 prompt `def quicksort(arr):`，
`--gen 128 --repeat 3` 取后两轮，2026-11-01）：

| 组 | tg | ms/token | 前进/轮 | 轮中位 | DFlash 接受 | MTP 接受 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 不投机 | 4.2 t/s | 237 | 1.00 | 237–240 ms | — | — |
| DFlash k=5 | 8.4 t/s | 119 | 4.27 | 487–499 ms | 64.7% | — |
| **级联 MTP=1** | **8.9 t/s** | 112 | 5.82 | 655–727 ms | 76.2% | **100%** |
| 级联 MTP=2 | 8.8 t/s | 114 | 6.10 | 695–774 ms | 72.9% | 62.5% |

四组生成文本逐字相同（greedy，验证路径不变，所以这只是正确性证据，不是覆盖率证据）。
**级联不是 1.15×，是 +6%。** 每轮前进 +36~43%，轮时也涨 +34~42%，两者几乎抵消。根因是
**head 前向对行数不敏感**：单行 194 ms、6 行 320 ms（`stage_ms[0]` 中位，同一口径），所以
"多验 3 行"接近"少前进 3 个 token"的代价。第二个 MTP 草稿尤其亏：接受 15/24 里第 2 位只中
3 次，所以 MTP=1 反而比 MTP=2 快。

### 落地时踩到的两处 shim 层问题

| # | 问题 | 处理 |
| --- | --- | --- |
| 1 | `last_hidden` 只导出一行，而 nextn 导出本来就每 token 一行（`llama-context.cpp` 的 `embd_nextn.size = n_embd_out * n_batch`） | 给 `qlh_kh_last_hidden` 加 `row` 参数，只差一个 memcpy 偏移 |
| 2 | `draft_rollback` 的 no-op 分支把计数器设成 `n_keep`；级联轮前进得比块写入多，于是下一轮 `draft_inject` 从空位开始 | 在 `qlh_kh_draft_rollback` 里改设真实长度 `kv_max+1`；纯块轮两值相同，行为不变 |

回滚深度：级联轮最多写 `block + cascade_mtp` 行，所以 tail 的 `SA_N_RS_SEQ` 提到这个深度
（`chain_client.tail_rs_depth`）。**只提 tail**：同一个值加在 GPU head 上会连它的 recurrent 快照
一起放大，实测 `gpu_layers=4` 要 2.19 GiB 却只驻留 1.85 GiB（静默溢出到系统内存）。

**共享成本这一点不变：**

- 验证器只认 argmax，任何来源的草稿都只是"待验证的行"，合并改变不了正确性
  （`forward_batch` 一次绕链验完整块）——所以级联只需多验几行，不需要第二个验证器。
- 成本是硬的：每多验一行，head 与 tail 各多算一行，而 head 是按"一次前向"收费的（见上表）。
- driver 仍是**全局互斥 + prompt 猜测**：纯 MTP 的 `--spec-k>0` 与 `--dflash` 不能并存
  （`chain_client.main()` 的同一处校验），`--spec-k -1` 按 `looks_like_code()` 选 k、接受率
  低于 35% 自动关。要换成"按段注册草稿源"，得用源列表 + "从 anchor 往后能连成的最长草稿"合并。

---

## 5. 说明书：怎么手动配

| 段 | 方法 | 要配什么 | 要什么文件 | 代码现状 |
| --- | --- | --- | --- | --- |
| tail | MTP | `kind="tail"`、`layer_range="32:40"`、`layers=8`、`source_layers=40`；`SA_SPEC=1`（默认）、`SA_SPEC_K=k`；`--spec-k 2`（`-1` 表示按 prompt 自动） | 带 `blk.40.nextn.*` 的 GGUF，**所有段必须同一个文件** | **已实现** |
| head | DFlash | `--dflash <draft.gguf> --dflash-block 6 --greedy`；head 段必须 `kind="gpu"`；`SA_KH_RS_SEQ ≥ block` | draft GGUF **必须合并 target 词表**（否则 `dflash requires ctx_other`），+1.03 GiB 落在 head | **部分实现**：两段可用，**三段不支持**（`dflash_split_layers` 只吃 head/tail） |
| mid | 给 DFlash 供 hidden | 无独立配置项 | 无 | **需新写**：`dflash_split_layers` / `_rows` / `DFlashDriver.__init__` 三处支持 N 段 |
| 任意 | n-gram / REST | **目前没有任何开关** | REST 需要一个 token 序列 datastore | **需新写**，量最小：上游 `ngram-map.cpp` 是纯函数，Python 侧约 20 行，不碰 shim 与拓扑 |
| tail | Medusa | 需要新 CLI | Medusa 头 GGUF | **需新写**（shim 只有 MTP 与 DFlash 两套入口） |
| head/tail | EAGLE 1/2/3 | 需新 CLI，且要让该段**同时持有 embedding 与 lm_head** | 训练好的 EAGLE 头 | **需新写**，卡在 loader 的按需判据上 |
| head | LayerSkip / SpS | — | 重训过的浅层出口 | **不建议**（未训练截断层 α≤0.08） |

两条容易踩的配置顺序：

- `LLAMA_LAYER_RANGE` 由 harness 从 `topology.toml` 的 `layer_range` 注入
  （`chain_client.spawn_worker`），手改环境变量会被覆盖。
- 切点必须是 4 的倍数且精确覆盖（`topology._validate`），`32:38` 这类在解析
  阶段就被拒——这限制了哪些段的边界能挂 draft 头。

---

## 5.5 为什么验证必须在尾段——这是定义，不是选择

验证在数学上是 `argmax over softmax(W_out · norm(h_39))`。`h_39` 只在第 39 层算完之后才存在，
head 手上只有 `h_31`，中间隔着 tail 的全部 8 层。所以"哪一段能验证"不是工程取舍——
**只有持有完整层序列的实体才能验证**，而那个实体是整条链，链的读出端是 tail（`output` 只在
末段建立）。tail 并不"具有验证能力"，它只是读出端；`_walk` 推进到 tail 就停也是同一结构事实。

一个容易忽略的点：**每台机器物理上都持有整模的字节**（mmap 整个文件），所以 head 机器*其实
能*算 32:40 层——验证能力在物理上是分布式的，被故意关掉了。打开它就等于让一段持有整模，
分段随即没有意义（本机 32 层已用掉 22 GiB，再加 8 层跑 CPU 是 3–8 tok/s，不如直接单机）。

三种"让两段各自验证"的做法都走不通：

| 做法 | 实际是什么 | 为什么不行 |
| --- | --- | --- |
| 给 head 加装 `output` | 换裁判：裁判变成"0:31 层的模型" | 未训练截断层实测 α≤0.08；而且裁判与草稿同源，自证必然一致 |
| 双 tail 冗余验证 | 两台机器各跑 32:40 再比对 | 正确但零收益（两份 lm_head 都付），且拓扑校验现在硬性只允许一个 tail |
| 分片验证 | 每段自洽检查（NaN/范数/逐位） | 那是健康检查，属于测试工具，不在解码路径上 |

**唯一有价值的变体：head 有否决权，tail 有判决权。** 在 head 上训一个**接受率预测器**
（不是 token 预测器），砍掉验证块的低价值行，只把有希望的送进 tail。终审仍在 tail，正确性
不变，但直接砍最贵的部分。**注意别按"每行几毫秒"来估这笔收益**：lm_head 按 16 列一档分块，
k=5 的 6 行整个落在第一档内，砍掉 2 行基本不省（见 `MOE-SPECULATION-ECONOMICS.md` §5）。
这个方向真正值钱的地方是**减少 tail 端的层计算与往返**，不是减少 lm_head 的行数。
代价是要训练，卡住的是**用 35B target 采 hidden 的数据从哪来**（`LILICORR-轻量头.md`）。

## 5.6 接受率预测器：数据能过，但收益被 k 扫描吃掉了

"在 head 上训一个预测器，判断草稿第 i 个位置会不会被接受"——**数据关能过**（自采 10 万 token
约 2 小时、只存标签 0.5 MB），但**收益关过不去**。账（两行的接受率见 `SPECULATION.md` §11）：

| 口径 | 被拒行/轮 | 可省 | 占轮时 | 判 |
| --- | ---: | ---: | ---: | --- |
| k=5 / 64 token | 0.18 | 2.3–3.2 ms | **0.8–1.2%** | 不值得 |
| k=5 / 128 token | 1.48 | 19–27 ms | 5.6–8.0% | 唯一有空间 |

关键在**验证块必须连续**：所以"砍行"只能砍尾部，预测器的动作空间其实是"这一轮起几行"
也就是**选 k**——而 k 已经用离线扫描选到 argmax（§11），剩余价值只来自轮间异质性。
而预测器本身的开销在另一侧倒挂：读 hidden 跑 MLP（2048→512→1，6 行约 12.6 MFLOP）在
driver 侧 numpy 单线程要 **6–12 ms/轮**，与"可省 2–3 ms"直接亏。

**但有一个特征已经付过钱了**：`qlh_kh_draft_block` 已经为每一行扫满 `n_vocab=248320` 的
logits 做 argmax，再累加一个熵项是**同一次循环里的加法**——零额外前向、零额外内存。前人的
现成答案也在这里：**SVIP / "Draft Model Knows When to Stop"（EMNLP 2025）** 用 draft 模型
自己的预测熵近似 draft-target 差异，**training-free**。对照 SpecDec++（2405.19715）训了接受
预测头、报告 +7~11%，但它的基线是固定 K 的粗糙启发式——那部分本项目已经用离线扫描拿到了。

**收窄到中文文字工作更救不了。** 2026-11-01 实测（本机两段 + DFlash `--dflash-block 6`）：

| prompt | 草稿/接受 | α | 前进/轮 | 自动切换 |
| --- | ---: | ---: | ---: | --- |
| 中文散文 + `/no_think` | 190/25 | **13.2%** | 1.68 | 第 6 步关闭 |
| 中文（默认，模型先输出英文思考） | 105/45 | 42.9% | 3.05 | 第 6 步关闭 |
| 代码（同机同参数） | 70/49 | **70.0%** | 4.57 | 保持 k=5 |

中文散文 13.2% 远低于盈亏线，同机同参数的代码是 70%。注意第二行：先输出**英文**思考时
42.9%，换成纯中文就掉到 13.2%——**草稿模型对英文的把握明显好于中文**（符合对它训练分布的
猜测，未单独验证）。这组数同时说明自动关闭是对的：两个中文场景都在第 6 步触发，代码没有。

**逐轮接受分布**（`--spec-k 0` 关掉自动切换，k=5，`chain_client.py` 的 `per_round` 记账）：

| 场景 | 0 | 1 | 2 | 3 | 4 | 5 | 全拒 | 满额 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 中文散文（57 轮） | 28 | 23 | 3 | 3 | 0 | 0 | **49%** | **0%** |
| 代码（22 轮） | 3 | 2 | 3 | 2 | 1 | 11 | 14% | **50%** |

**两种形状相反**：代码是双峰（一半的轮次整块全中），中文是低端单峰（一半的轮次一个都不中，
**从来没有满额过**）。

这否掉了"中文场景值得训接受率预测头"的猜想：预测头只能**选择哪些轮次值得试**，改变不了
**草稿本身的质量**——即使完美识别出那 49% 全拒的轮次，剩下 51% 的前进量也只有 1–2。

⚠️ **但这组数据只能说明"DFlash 草稿头在中文上不行"，不能说明"中文不可投机"。** DFlash 的
draft 本来就是面向代码与数学训练的（`SPECULATION.md` §6 的 n-gram 模拟也只统计了代码），
用它评判中文是**不公平的对照**。公平的做法是换一个在中文上有把握的草稿源——最现成的是
**模型自带的 MTP 头**（`blk.40`，它是同一个 checkpoint 的一部分，训练语料与 target 一致）。
MTP 在代码场景跨机实测 1.13×（`SPECULATION.md`），**但中文场景从未测过**。

所以正确的结论是：**中文投机要换草稿头，而不是放弃投机**。在换之前，现有自动关闭机制
（接受率低于 35% 连续 6 步就关）继续生效，中文场景退化到普通解码，不损失什么。

**收窄到代码/数学场景救不了这条路**：代码的高接受率恰恰说明不需要预测器——它的可预测性主要
来自字面重复（`SPECULATION.md` §6 的 n-gram 模拟 m=2.56–4.92），那部分 n-gram 已覆盖；
真正需要模型的"非重复模式延续"正是 DFlash 在做的。数学场景更糟：DFlash 在那里根本没测过。

**所以**：不训预测器。先做影子统计——`qlh_kh_draft_block` 多回一个 `top1_prob`，记
`(熵, 位置, 是否接受)`，**熵对接受率的 AUC > 0.7 才继续**；达标上 SVIP 式阈值规则，不达标转
PLD 影子统计（零训练，代码场景 2.56–3.37×）。

## 6. 如果要做，顺序

现状与顺序：

1. **n-gram：已测，不达标，停手**（2026-11-01，`checks/probe_ngram_replay.py`）。6 个真实场景里
   只有"重述已有代码 128 token"过盈亏线（m_eff 3.05），而它需要 m=8 → 9 行验证，plain 等效成本
   6.10 对 k=8 的 6.49，**仍然亏 6%**；其余 5 个 m_eff 只有 1.15–2.10。两条口径教训：
   **`m < n` 是恒空配置**（`ngram-map.cpp` 的 `copy_max < n_draft_min` 直接返回空），所以
   m=3/n=4 这种"按 k≤5 猜"的参数无效；**"过盈亏线"与"能赚"不是同一个判据**——它们在不同 k 下
   成立。最优参数是 `n=2`；收益由**"输出有多少在复述历史"**决定，不由"是不是代码"决定。128 token
   会系统性低估（同一 prompt 64→128→512 是 1.83→3.05→4.65），有长文件重写负载时按 ≥512 重测。
2. **级联 MTP：已实现并实测，+6%**（不是推断的 1.15×，见 §4）。开关 `--cascade-mtp N` 默认关；
   只在 DFlash 整块被接受时才继续，部分接受时不多付草稿成本。
3. **DFlash 支持 N 段**（让中段供 hidden）：改动集中在三个函数，不动协议；但三段拓扑在 35B
   上本来就更慢，只在"装不下"时有意义。
4. 其余（Medusa / EAGLE）都要改 loader 的按段 lazy 判据或新增 shim 入口，等有明确的收益证据再说。
