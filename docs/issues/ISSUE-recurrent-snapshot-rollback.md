# ISSUE：recurrent 快照不跨 ubatch 滚动，跨 ubatch 回滚读到陈旧槽

> 状态：**已退场**（2026-11-09）
> 更新日期：2026-11-09
>
> **这一份的修复代码已经删掉了**：rebase 到上游 `f477f795c` 之后，上游 `ab3768cdc` 改成
> **拒绝**超出「最后一个 ubatch」的快照回退（`cell.rs_max = n_seq_tokens − 1`），并由
> `split_equal()` 的 `n_keep_tail` 保证整帧回退落在同一个 ubatch 内——和我们原来的解法
> （`build_rs_shift_snapshots` 让快照按 ubatch 往前挪）是同一个问题的两种解法，而且槽位语义
> 互斥。两套并存会让后来人以为 shift 在解决上游没解决的问题，所以 shift 整块删除，
> `llama-graph.h` / `llama-memory-recurrent.{cpp,h}` / `delta-net-base.cpp` 因此变成与上游
> 逐字节相同。本文保留的是**现象与判据**（怎么观测、怎么证明逐位相同），不是现行实现；
> 整帧回退在 `rs_max` 下反而永远不合法，那个回归与修法见
> [`SEGMENTED-LIMITS.md`](../limits/SEGMENTED-LIMITS.md) §0。
> 适用范围：hybrid 模型（Qwen3.5/3.6 的 GDN + full attention 交替结构）上 recurrent 快照的跨 ubatch 回滚；读者为内核维护者。修复曾是 `H:\llamacpp-c` 的本地 patch，未提交上游。
> 相关：`ISSUE-dflash-kv-rollback.md`（同一 hybrid memory 的 `seq_rm` 路径）

`llama_memory_seq_rm()` 把位置退掉之后，用相同输入重写同一位置，得到的 hidden 与从未写
过那次不同，而 `seq_rm` 本身返回 true 不报错。根因是 `n_rs_seq` 的逐 token 快照只在一个
ubatch 内有语义；修复后单 token 与 2-token 同形状回滚的 max|d| 均降到 0。跨形状残留的
2.5e-2 来自两套 GDN kernel 的浮点累加顺序，与回滚无关（〈修复后实测〉）。

---

## 现象

hybrid 模型（Qwen3.5/3.6 的 `LLL F` 交替结构）上，`llama_memory_seq_rm()` 把位置退掉之后，
用相同输入重写同一位置，得到的 hidden **不等于**从没写过那次的值：

```
写位置 P（单 token）→ seq_rm(P) → 同输入重写 P
  max|hidden diff| = 0.36 ~ 0.83     ← 相对误差 2~10%
```

`seq_rm` 本身返回 true，不报错。表现成 token 偶发翻转（8 个位置里 3 次）。

## 根因

`n_rs_seq` 的逐 token 快照**只有「单个 ubatch 内」的语义，跨 ubatch 不做滚动**。

1. `llama_memory_recurrent::seq_rm` 的 partial 分支（`src/llama-memory-recurrent.cpp:197-216`）
   只做两件事：`cell.pos = p0-1` 和 `rs_idx = rollback`。**它不拷贝任何数据**。
2. 真正的状态恢复在下次 graph 构建时，由 `s_copy()`（`:1361`）按
   `idx * mem->size + src0` 取「第 rollback 个快照槽」。
3. 快照槽的语义由 GDN 的 `ggml_gated_delta_net(..., K)` 决定，源码注释写得很明确
   （`ggml/src/ggml-cuda/gated_delta_net.cu:158-169`、`ggml/src/ggml-cpu/ops.cpp:11256-11263`）：
   *slot 0 = 最新；n_tokens < K 时只写 slot 0..n_tokens-1，其余 caller-owned*。
4. 上游唯一的保护是 `split_equal(n_ubatch, ..., n_rs_seq+1)`——保证一个序列的**最后 K 个 token
   不被拆到两个 ubatch**。这只保护 ubatch 内部，**跨 ubatch 没有任何搬运逻辑**。

于是单 token decode（每次 ubatch 只写 slot 0）之后，slot 1..K-1 是若干轮之前的陈旧状态。
`seq_rm(P)` 后重写 P 时读 slot 1，拿到的是某个远古 ubatch 的「倒数第二个 token」。

**第二个缺陷**：`delta-net-base.cpp` 的 conv 槽写入用 `max(0, n_tokens-K+t)` 截断，
`n_tokens < K` 时把多个槽写成同一段旧窗口，主动写坏本应 caller-owned 的历史
（这是 2-token 批量回滚后仍差 2e-2 的原因）。

## 上游为何未覆盖这条路径

**llama.cpp 官方投机从不跨 ubatch 回滚**：

- **target 端**：verify batch 一次提交 `n_draft + 1` 个 token，槽 `0..n_draft` 全被这次 ubatch
  正确写入；接受 j 个后回滚量 `= n_draft + 1 - accepted ≤ n_draft`，**永远落在刚写的那批槽里**
  （`tools/server/server-context.cpp:5034`）。
- **draft 端**：`cparams_dft.n_rs_seq = 0`（`common/common.cpp:1295`），**不支持部分回滚**，
  verify 后整段删掉重跑（`server-context.cpp:3926-3936`）。

只有「**分段 target + 单 token draft**」这个组合才需要跨 ubatch 回滚。本项目正是这个组合，
所以踩到了上游从未覆盖的路径。

## 修复

5 个文件、+139/−5，改在 `H:\llamacpp-c`：

- **`llama-graph.cpp` / `.h`**：新增 `build_rs_shift_snapshots()`，在 `build_rs()` 读出
  pre-batch state 之后、调用方写新槽之前插入移位节点

  ```
  new_slot[s] = old_slot[s + r - n_tokens]     s ∈ [n_tokens, K-1]
  ```

  （`r` = 该 cell 的回滚距离，越界清零；`r < n` 时从高到低走、`r > n` 时从低到高走，
  避免源被先覆盖）。语义从「每 ubatch 一份快照」变成「每 ubatch 之后 slot s 恒等于
  最后位置−s」，跨 ubatch 连续。`llm_graph_input_rs` 记录 `shifts` 并纳入 `can_reuse`
  （回滚距离变了必须重建图，因为移位已固化进节点）。
- **`llama-memory-recurrent.h/.cpp`**：新增只读 `get_rs_shift(i)`（`s_copy()` 会消费并清零
  `rs_idx`，graph 构建期需要单独读）。
- **`delta-net-base.cpp`**：conv 只写本 ubatch 覆盖的槽 `0..min(n_tokens,K)-1`，
  去掉 `max(0,…)` 截断。

## 修复后实测

`SA_N_RS_SEQ=4`，head `0:32`：

| 场景 | 修复前 max\|d\| | 修复后 |
|---|---|---:|
| 单 token 写 P → rm1 → 重写（P=10/11/12） | 0.67 / 0.51 / 0.36 | **0 / 0 / 0** |
| 单 token rm(P−1) → 重写 | 0.83 / 0.34 | **0 / 0** |
| 回滚后继续 3 步 | 0.61 / 1.34 / 0.50 | **0 / 0 / 0** |
| 2-token 写 → rm2 → **同形状**重写 | 0 | **0** |
| 2-token 写 → rm1 → 单 token 重写 | 2.5e-2 | 2.5e-2（**不变**） |

最后一行不是回滚缺陷：`n_tokens > 1` 走 `GDN_CH`（chunked）、单 token 走 `GDN_AR`，
两套 kernel 的浮点累加顺序不同，同一位置的 state 差 ~1e-2。**同形状回滚是精确的**，
跨形状的差异与回滚无关。

**「不影响 argmax」只在短上下文成立。** 35B 多轮对话实测：`spec_k=0`
时续用增量 KV 与从头重放逐位相同（23/23 token）；`spec_k=1/2` 时出现措辞分叉
（`first_diff` 9~11，上下文 100+ token），答案仍对，但已经不是同一条计算。这 ~2e-2
会随上下文累积，长上下文里足以翻一个 argmax。判据与数据见
`slimarc/prototype:checks/verify_multiturn.py` 的 R4 与 `checks/_multiturn-35b-spec{1,2}.json`。

## 代价与限制

- **性能**：每步每层每 tensor 增加 `K-1` 次 state 拷贝（head 约 +192 个图节点/步，
  734 → 926）。本机 GPU 频率漂移太大无法可靠量化，推算 +2~8 ms/step（<10%）；
  `n_rs_seq=1` 时约 1/4。可将 `K-1` 次单槽 cpy 合并成 scratch 往返（节点减半），未做。
- **其它 arch**：mamba / plamo2 / glm5-next / kimi-k3 / lfm / bailingmoe3 共用同一个
  `build_rs()`，自动获得移位；但它们的 conv/scan 写入若有同类 clamp（如
  `bailingmoe3.cpp:194-269`）需单独改。本次只改 `delta-net-base`（qwen35/qwen35moe 用）。

## 证据

`checks/` 下：`_rb_probe.py`（修复前后矩阵）、`_rb_e2e.py`（head+tail 端到端）、
`_rb_fix.diff`（完整 diff）。
