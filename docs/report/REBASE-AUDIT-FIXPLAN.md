# Rebase 审计修复腹稿

> 面向 2026-11-20 rebase 后的审计发现 · 基于三份审计 + 原审计 · 产出修复顺序与分工
> 状态：**腹稿**（等 `audit-single-source` 与 `r9-vulkan-remeasure` 回来后定稿）
> 更新日期：2026-11-20

---

## 1. 已经拿到的输入

| 来源 | 内容 | 状态 |
|---|---|---|
| `merge-correctness` | 12 个 rebase 残余风险（R1–R12） | 已治理 7 条（R1/R2/R7/R8/R11 已修，R10/R12 判断为不修） |
| `audit-code-quality` | 22 个屎山/数据流/历史包袱问题 + 3 个「只改 3 处」建议 | 已到 |
| `audit-single-source` | 单一事实源 + 重复造轮子 | **等** |
| `r9-vulkan-remeasure` | Vulkan 两项调优在新 pipeline 下的收益 | **等** |

---

## 2. 修复优先级

### P0（正确性隐患，必须先修）

| # | 位置 | 问题 | 修复 |
|---|---|---|---|
| 1 | `tools/split-stage/main.cpp:2454-2461` | **正确性靠「内存越界刚好读到无害数据」**：手工 `free(batch.pos)` + `malloc` 更大的数组绕过 `llama_batch_init` 的所有权契约；compat 转换按 `n_pos_per_embd`（M-RoPE 为 4）读 section-major pos，`llama_batch_init` 只按 `n_tok` 分配 → 越界读 3/4 个位置 | stage 改用 `llama_batch_ext`（head 侧已在用），`add_embd`/`set_pos` 天然给出正确的 section-major pos，`free/malloc` 整段删除 |
| 15 | 同 #1 | **一条 hop 两条构造路径**：head→local 用 `llama_batch_ext`（`split_build_hop`），head→stage 用旧 `llama_batch`（`decode_run`） | 与 #1 一起做，收敛到一处 |
| 11 | `src/llama-context.cpp:2910 vs 2933` | `split_add_hop_row` 里两个索引用法不一致：2910 算出 `added`，2918/2925 用 `added`，2933 却用 `i`。正确性依赖「hop 是空 batch、每次恰好追加一行」的隐含前提 | 2933 改用 `added`（零成本） |

### P1（高维护成本 / 数据流断裂）

| # | 位置 | 问题 | 修复 |
|---|---|---|---|
| 3 | `src/llama-split-remote.cpp:629-1192` | `decode()` 564 行、19 形参（12 个 `&` 输出）、5 个 `return 0` 出口，每个都要记得 `read_tap()` | ①12 个 out 收进 `split_reply` struct；②body 按 section 拆 `read_rows/read_verify/read_nextn/read_output`；③5 个 early-return 收敛到单一尾部 |
| 5 | `src/llama-memory-split.h:126` + 8 写入点 | 字段叫 `remote_failure` 但本地链失败也写；**8 个写入点跨 6 个方法 2 个文件，20+ 读** | 改名 `chain_failure`/`diverged`，8 个写入点收进 `mark_diverged(reason, stage_no)` |
| 4 | `src/llama-context.cpp:3228-3621` | `split_drive_remote` 394 行；注释说「Every failure exit below has the same shape」**不成立**，5 处 `return -3` 绕过 `fail()` 不记录 divergence | 拆 plan → send → validate；统一走 `fail()`，或给「tail 已答完、KV 仍自洽」加显式参数 |
| 2 | `common/sampling.cpp:212-243` | 四个谓词**永久镜像** `src/llama-sampler.cpp` 的 early-out，靠注释里的位置描述同步；上游一改就静默出错 | **fork 内无法闭合**：正解是让 sampler 暴露 `llama_sampler_*_is_noop()`（动上游文件）。fork 内的临时方案：在谓词旁加一个启动期自检（对已知 sampler 列表验证），或至少把「必须与哪几行同步」写成可 grep 的锚点 |

### P2（结构清理，收益随 P0/P1 之后）

| # | 位置 | 问题 |
|---|---|---|
| 6 | `src/llama-context.cpp:2739-2746` | 8 行注释归属错位（`split_drive_*` 的文档飘在 `split_expand_pos` 上方） |
| 7 | `src/llama-context.h:386-389` | 注释说拆函数是为了「on one screen」，而 `split_drive_remote` 是 394 行 |
| 8 | `tools/split-stage/main.cpp:2401-2403` | 调试过程残留（描述一个已修 bug 的排查） |
| 9 | `src/llama-memory-split.cpp` | 重复样板 11 次 `head->xxx(); for (auto * m : tails) m->xxx();` + 8 次 remote 循环；`seq_cp/keep/add/div` 四函数逐字相同 |
| 10 | `src/llama-split-proto.h` | 912 行里约 700 行注释；全仓库「used to」32 处 /「the old」11 处 /「before this」10 处 |
| 12 | `src/llama-context.h:481-560` | `split_frame_answer` 12 个平行字段 = 5 组三元组 |
| 13 | `common/speculative.cpp:1042-1072` | 8 个平行数组描述同一个 tap，靠下标对齐 |
| 14 | `src/llama-context.cpp:3651-3684` | 同一条 hidden 有 5 个名字（carrier/hidden/embd/layer_inp/t_layer_inp） |
| 16 | `src/llama-split-proto.h:~380` | 注释残缺句（"identical to" 两边是同一个东西） |
| 17 | `common/common.cpp:1957-1958` | `(void) params.split_tail_lo/hi`（字段存在但在该函数不用） |
| 18 | `tools/server/server-context.cpp:1350` | 又手推一次角色，而 `common.h:386` 已有 `chain_kind` 枚举 |
| 19 | 全仓库 | 角色散在 4 套表达（协议 `role{ANY,RELAY,TAIL}` 无 HEAD；`chain_kind` 无 HEAD/RELAY；`layer_lo==0 && layer_hi<n` 推 head；`--role`）；**head 从未被命名成一个角色** |
| 20 | `src/llama-split-remote.cpp:1039` | `nextn` 与 `tap` 在 wire 上互斥（`rsp_header` 只有一个 `n_embd`），只在运行时拒绝 |
| 21 | `src/qlh-vram-budget.h:108-110` | 三个经验系数无推导（`VRAM_PER_BYTE=1.13` / `GPU_VRAM_OVERHEAD=640MiB` / `VRAM_LANDED_SLACK=0.85`） |
| 22 | 提交结构 | **范围蔓延**：13 提交里 4-5 个与 split 无关（vulkan shader 205、MoE cache 191、wbuf 153、vram-budget 660 + slim-arc 788 ≈ 1600 行） |

---

## 3. 分工原则

**复用原 agent**（它们对自己发现的问题最了解）：

| Agent | 负责 | 理由 |
|---|---|---|
| `audit-code-quality` | P0 #1/#15/#11、P1 #3/#4/#5、P2 中它列出的注释/结构问题 | 它逐行读过 `decode`/`split_drive_remote`/`memory-split`，行号和语义最准 |
| `fix-r1-array` | P2 #22 的 per-layer 数组部分的提交拆分建议（它刚改过 78 个文件） | 它对「哪些改动属于 split、哪些不属于」有完整清单 |
| `merge-correctness` | 复核修复后的语义等价性 | 它已经建好三方对比的方法 |

**我自己做**：
- 汇总各 agent 的 diff、复核
- 跑全套 verify
- 重导补丁 + 更新 `PATCHES.md`

---

## 4. 验证要求（每个修复都要过）

1. **编译**：clean rebuild（删 `build-cuda/{src,common,tools}` 的 `*.obj` + touch 12 个 `Unity/unity_*_cxx.cxx`，因为这个 build 目录在中文 locale 下 depfile 解析失败）
2. **smoke**：`QLH_SPLIT_MODEL=<model> build-cuda/bin/split-smoke.exe` 五项判据全过
3. **逐位一致**：修复不改变数值（`[A]` vs `[B]` hidden bit-identical、三段链 logits bit-identical）
4. **不引入新路径**：修复后的代码路径必须仍是「`steer == nullptr` 时拓扑不变」这一类设计

---

## 5. 第二份审计（audit-single-source）并入

23 条「单一事实源 / 重复造轮子」。**最高优先级是它找到的 #1**，比屎山审计的 #1 更严重：

### 新增 P0（静默错误路径，各 ~10 行改动）

| # | 位置 | 问题 | 修复 |
|---|---|---|---|
| **S1** | `src/llama-model.cpp:1839-1866` + `src/qlh-vram-budget.h:399-434` | **上游 `get_layer_buft_list()` 被插进 `layer_range_active()` 三元双轨**，而 vram-budget 的注释直接写「Replicates llama_model::load_tensors()' placement rules」，把 `i_gpu_start`/`act_gpu_layers`/`offload_output` 各抄一遍。**上游一改 offload 规则，VRAM 预算就静默算错——而这份预算的唯一目的就是抓静默 spill** | 抽 `llama_hparams::placement(n_gpu_layers, has_devices)` 纯函数，`llama-model.cpp` 与 `qlh-vram-budget.h` 同时调用；估算器只留「按 tensor 累加字节」 |
| **S2** | `common/reasoning-budget.cpp:292-313` vs `:167-192` | `common_reasoning_budget_rejects()` 把 `apply()` 两条守卫原样抄了一遍，注释自认 "apply()'s guards, in the same order"。这是 per-row verify 通道**判断「argmax 能否采信」的唯一 guard**（`llama-ext.h:348-351` 明说 "the check is the only guard"） | 在 `apply()` 提出 `forced_token(ctx) -> optional<llama_token>`，两处只问它一次 |
| **S3** | `src/llama-batch.cpp:191-203` vs `src/llama-context.cpp:2762-2768` | M-RoPE 位置展开规则两份（`expand ? (j<3 ? pos[0] : 0) : pos[j]`），注释自认 "reproduces this expression verbatim. The two must change together"。分歧静默：pos 0 两边一致，单 token 测试必过 | 抽 `static inline llama_batch_expand_pos()` 到 `llama-batch.h`，两处各调一次 |

### 新增 P1

| # | 位置 | 问题 |
|---|---|---|
| **S4** | `llama-context.cpp:2693-2702`、`2359-2375`、`common/common.cpp:1359-1396`、`qlh-vram-budget.h:421,474`、`split-stage/main.cpp:1042-1058` | 「这段范围覆盖 output / 谁是 relay / 谁是 tail」推导 **6 次**；且 `llama-context.cpp:2677-2679` 注释说 "Same predicate here"，**实际不是**（`layer_hi >= 0` vs `layer_range_active()` 的 `> 0`，`layer_hi==0` 时结论相反） |
| **S5** | `common/common.cpp:2034-2038` vs `server-context.cpp:1518-1527` | **head 线程数两处推导且互相矛盾**：common 把 head 折半，server 却拿**未折半的**值算 tail 并打印。20 核 `-t 14`：head 实际 7、tail 3，**日志说 14 + 3**。这段日志存在的理由恰恰是「修了重复但没修数字是假的」 |
| **S6** | `common/common.cpp:1316-1317`、`server-context.cpp:1913,2397` | per-slot draft 行上限三份实现，且 common 那份用**未解析**的 `split_verify_rows` |
| **S7** | `common/common.h:605`、`common/arg.cpp:2954`、`src/llama-split-proto.h:62` | `split_verify_rows` 默认值 17 硬编码三处 |
| **S8** | `llama-model-loader.cpp:962-975` vs `common/arg.cpp:1289-1300,2857,2874` | `LLAMA_LAYER_RANGE` 解析两次、`lo:hi` 格式解析三份，且校验态度不同（CLI 非法抛异常、env 非法静默忽略） |
| **S9** | `common/common.cpp:1308` vs `server-context.cpp:1916,1920` | **同一条启动警告打两遍**，措辞不同，`-np 4 --spec-draft-n-max 16` 会让操作员读到两条不一致的「怎么修」建议 |

### P2（重复造轮子，理由多可接受）

| # | 位置 | 问题 |
|---|---|---|
| S10 | `slim-arc-prefetch.cpp:113-134` vs `llama-mmap.cpp:1115-1134` | `system_advice()` 重写上游 `llama_prefetch()`（Windows `PrefetchVirtualMemory` / POSIX `madvise`）。理由可接受（要 per-layer 增量 + 后台线程 + 预算节流） |
| S11 | `slim-arc-page-range.cpp:40-116` vs `llama-mmap.cpp:1093-1114` | `covering_page_range()` / `coalesce_page_ranges()` 复制上游 |
| S12 | `split-stage/main.cpp:1042-1045` vs `llama-hparams::n_pos_per_embd()` | 第三份 M-RoPE 推导 |
| S13 | `speculative.cpp:2112,2119,2380` vs `server-context.cpp:4734` | 环境变量名散落字面量 |
| S14 | `llama-context.cpp:2306-2308` | `split_arch_exports_pre_norm()` 硬编码 arch 白名单，与 `qwen35.cpp` 建图代码无机制关联 |
| S15 | `llama-context.cpp:2402` vs `2544` | 本地/远程互斥检查两份 |
| S16 | `common/common.cpp:1445,2081,2097`、`server-context.cpp:1350` | 「这个进程是不是链的一环」重复 4 次 |
| S17 | `split-stage/main.cpp:95-148` | 自己实现 argmax（`llama_sampler_init_greedy` 是公开 API） |
| S18 | `qlh-vram-budget.h:274-371` vs `common/fit.h:77-110` | 自己解析 GGUF 估 VRAM，而上游 `common_fit_measure` + `llama_memory_breakdown()` 能给同一数据 |
| S19 | `common/sampling.cpp:291-340` + `split-stage:387-434` | 候选通道正向/反向两份（作者已用 `static_assert` 钉住 enum，但 penalties 窗口语义没约束） |
| S20 | `llama-context.cpp:1169-1230`、`1487-1517` | `get_logits_ith` 的异常语义分叉、`set_embeddings(false)` 被静默忽略 |
| S22 | `server-context.cpp:590` | 上游 `server_slot::can_split()`（batch 切分）与本 fork 的 split-chain 同名不同义 |
| S23 | 多处 | 注释历史叙事可压缩（`llama-split-proto.h` 注释/代码 = 622/228） |

### 核实后「不是问题」的（避免重复投入）

- **S21**：`tools/rpc`（ggml-rpc）不复用的决定**干净，无半复用痕迹**。`llama-split-net.h:8-12` 已写三条理由，全仓无 `ggml_backend_rpc`/`rpc_server` 引用
- `llama_memory_split` 没有重复 `llama_memory_hybrid` 的组合逻辑，而是正确实现 `llama_memory_i` 并把 `state_write/read` 转给 head
- `split_add_hop_row()` 正确复用了 `llama_batch_ext`
- `decode()` 早分叉成 `decode_plain()`/`decode_split()` 是正确形状
- `static_assert` 钉住 `CAND_STEP_*` 是亮点

---

## 6. 合并后的最终执行顺序

| 轮次 | 内容 | 负责 |
|---|---|---|
| **1** | S2 + S3（各 ~10 行，消灭两处静默错误路径）+ S5（日志撒谎，直接可验证） | `audit-single-source` |
| **2** | S1（架构债：抽 placement 纯函数）+ S7 + S6（默认值/公式收口） | `audit-single-source` |
| **3** | S4（6 处谓词收口）+ S9 + S8 + S13（启动期 UX 与解析） | `audit-single-source` |
| **4** | C1/C15/C11（stage 换 `llama_batch_ext`，正确性隐患） | `audit-code-quality` |
| **5** | C3（拆 `decode` 564 行）+ C5（`remote_failure` → `chain_failure`）+ C4（`return -3` 收口） | `audit-code-quality` |
| **6** | P2 结构清理（注释归属、调试残留、重复样板、注释瘦身） | `audit-code-quality` |
| ~~7~~ | ~~#22 提交拆分~~ **取消**：用户 2026-11-20 决定不提上游 PR，「和上游关系维持现状」——不需要为一个不存在的 PR 拆提交。13 个提交的结构保持不变。 | — |

**每轮后必须**：clean rebuild + split-smoke 五项 + `[A]`/`[B]` bit-identical。

**跨份审计的重复**：两份审计都提到「注释历史叙事」和「角色推导散落」，以 `audit-single-source` 的 S4/S16 为主（它给了更精确的 6 个位置）。

---

## 7. R9 结果（已到）：两项调优都保留

| 项 | 旧 pipeline | 新 pipeline | 结论 |
|---|---|---|---|
| XE1 MMVQ 放行（35B-A3B Q2_K, tg128, Iris Xe） | 10.66 → 16.64（**+56%**） | 8.80 → 14.33（**+63%**） | **保留**。内核带宽 33.2 GB/s ≈ 旧 33.9（噪声内），是「修复前基线自己快了 23%」 |
| `op_offload` 64（2B Q4_K_M, pp32） | 45.7 → 89.9（+96.9%） | 63.9 → 94.3（**+47.6%**） | **保留 64，不要回 32**（回 32 要 −48%）。阈值 64 正好落在**新的交叉点**（从旧 pp>96 左移到 pp≈56–60） |

**附带发现（超出 R9，归 P2 决策）**：标量阈值表达不了「小 pp 完全不 offload」——非 MUL_MAT 走 `ggml_nrows`，flash-attn/rope/3D unary 在 pp32 就远大于任何阈值，照旧搬上核显，使 pp32/48 比纯 CPU 低 **14% / 13%**（旧 pipeline 更差：20%）。要拿回这段得改判据，不是改数字。

**装置**：独立 worktree `H:\llamacpp-r9` + `build-r9`（不碰共享树），无源码改动（用现成 env `GGML_VK_DISABLE_MMVQ` / `GGML_OP_OFFLOAD_MIN_BATCH`），另做删分支真 baseline 验证 env 等价，24 次构建 / 60 次 bench，清理干净。报告 `G:\C\PYT\qlh\docs\measure\R9-VULKAN-REBASE.md`。

**结论：R9 无需修复动作；两项调优保留，只需把注释里的旧数字（6.5→33.0 / 10.7→16）更新成新口径（8.0→33.2 GB/s / +63%）——归入 P2 注释项。**
