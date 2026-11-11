# ISSUE：多序列共享的可行性

> 状态：**方案**（文中引用的 v2 工具链——`chain_client` / `stage_worker` / `shim/` / `koakuma`——已退场，归档在 `slimarc/prototype`，命令与路径只作历史口径）
> 更新日期：2026-11-03
> 适用范围：一条链路内多序列（`n_seq_max > 1`）的可行性评估与最小改动清单；读者为链路与内核维护者。代码未实现。
> 相关：`SPECULATION.md`（`SA_N_RS_SEQ`）、`MEMORY-LIMITS.md`（每段固定开销）

README「未做」第 3 条把 `kv_unified` 与 recurrent state 隔离当成前提，两个前提都不成立
（见下文）。可行路径是「一条连接内 2 个 seq 轮流 decode」，**有条件做**，3~4 人天，收益是
会话共存与前缀复用。**待定项**：三条推断未经实验——交替 decode 与各自单独 decode 逐位
等价、`build_rs_shift_snapshots` 在多 seq 下逐 cell 正确、两段 `n_seq_max` 同值即可同步；
实验设计与靶子见〈证据边界〉。

---

## 结论

`kv_unified` **不是**多序列的前提，hybrid 的 recurrent state **本来就按序列隔离**。缺的是本仓库三处硬编码：装载时 `n_seq_max = 1`、token 通道的 `seq_id = 0`、协议里 `n_past` 没有 seq 维度。

多序列买到的是**会话共存 + 前缀复用**（`server.py:159` 现在每个请求都 `client.reset()` 重放整个 prompt），不是吞吐：一个 worker 一个 context，decode 仍串行。

## README「未做」第 3 条的两个前提不成立

> 多序列共享。需 `kv_unified` 与每段多序列位置；先解决 hybrid 模型那 30 层 linear 的 recurrent state 隔离。

**① `kv_unified` 不需要。** 它解决两件事：一个 token 同时属于多条序列（共享前缀），以及 KV 池共享而非预切分。前者才是它的硬门槛——`src/llama-batch.cpp:620-624` 在 `sequential && has_cpl` 时要求 `-kvu`。两条独立对话交替 decode，每 token 只属于一个 seq，`kv_unified=false` 走得通：`src/llama-kv-cache.cpp:84` 用 `n_stream = n_seq_max` 给每个 seq 一份独立 KV，`:149-154` 建 `seq_to_stream`，`:965-986` 只要求「每 token 一个 seq、各 seq token 数相等」。本仓库 `slimarc/prototype:checks/verify_multiseq.py` 已测过这条：`kv_unified=0` 时被拒的是 `per_token > 1`，不是多序列本身。

**② recurrent state 不用「先解决隔离」。** 它一直是每序列一个 cell：`src/llama-memory-recurrent.h:71` 的 `size` 注释是 *shared across all sequences*（指 cell 池），`:95` 的 `mem_cell::seq_id` 是 `std::set<llama_seq_id>`；`src/llama-model.cpp:3226-3227` 把 `mem_size` 直接设成 `n_seq_max`；`:77` 的 `rs_idx` 是 per-seq 数组；快照张量按 `mem_size * (1+n_rs_seq)` 行（`llama-memory-recurrent.cpp:102`）以 `idx*size + src0` 分块（`:1378`），每 cell 一份。`find_slot` 的注释直白（`:525`）：*each cache cell can store the state for a whole sequence*。

所以我们那条「硬骨头」不在内核，在下面三处。

## 现状

### 内核（`H:\llamacpp-c`）：支持，默认关

| 位置 | 事实 |
| --- | --- |
| `include/llama.h:416` | `kv_unified` =「use a unified buffer across the input sequences」 |
| `src/llama-context.cpp:4071,4102` | 默认 `n_seq_max = 1`、`kv_unified = false` |
| `src/llama-context.cpp:391-405` | unified 时 `n_ctx_seq = n_ctx`；否则 `n_ctx / n_seq_max` 并对齐 256，`n_ctx` 反被重算 |
| `src/llama-context.cpp:1893` | 采样路径的 `n_seq_max` 在 unified 时取 `LLAMA_MAX_SEQ` |
| `common/arg.cpp:1750-1758` | `-kvu/--kv-unified`，help 写「default: enabled if number of slots is auto」——那是 server 的行为：`tools/server/server.cpp:127-130` 在 `n_parallel=auto` 时置 4 并开 `kv_unified` |
| `common/arg.cpp:1675-1683` | 新增 `--kv-unified-per-slot`，限制每 slot 上下文（本项目用不到） |
| `src/llama-memory-recurrent.cpp:529,545` | `GGML_ASSERT(ubatch.equal_seqs())`：一个 ubatch 里各 seq 的 token 数必须相等；`seq_id >= n_seq_max` 报「Try using a bigger --parallel」 |
| `src/llama-memory-recurrent.cpp:193-216` | 部分回滚的条件是 `rollback <= n_rs_seq` 且 `cell.seq_id.size() == 1`——我们那条跨 ubatch 快照缺陷的同一个函数 |
| `src/llama-memory-recurrent.cpp:781-861` | `state_write(io, seq_id)` 支持 per-seq 序列化；API 在 `include/llama.h:900-915` |
| `src/llama-memory-hybrid.cpp:144-151` | `seq_rm` 先 recurrent 后 attn，前者失败直接 return false |

**跨 ubatch 快照与跨 seq 不是同一个根。** 前者是 `n_rs_seq` 的时间维度（快照不滚动，已修）；后者是 cell 索引的空间维度（`s_copy` 按 cell 的 `seq_id` 取 `rs_idx`，`:1361-1378`）。我们的补丁 `build_rs_shift_snapshots`（`src/llama-graph.cpp:4028-4090`）以 `ubatch.n_seq_tokens` 为单位、逐 cell 独立移位，与 seq 数无关，两者不冲突。

### shim（本项目）：ABI 有一半，装载与分发全锁死

| 位置 | 事实 |
| --- | --- |
| `keep_head_worker.py:240` | `qlh_kh_load(..., 1, ...)`——`n_seq_max` 硬编码 1。这是装载侧唯一的根：context 里只有 1 个 cell、1 个 KV stream |
| `slimarc/prototype:shim/qlh_keep_head_gpu.c` | `handle->kv_unified = 0`，注释写明「llama.cpp 默认 false」 |
| `slimarc/prototype:shim/qlh_keep_head_gpu.c` | `qlh_fill_explicit`：`per_token > 1 && !kv_unified` 返回 -5（拒绝一次 token 挂多 seq，不是拒绝多序列） |
| `slimarc/prototype:shim/qlh_keep_head_gpu.c` | `qlh_kh_forward_embd_seq` 已支持显式 `n_seq_id/seq_ids/positions` |
| `slimarc/prototype:shim/qlh_keep_head_gpu.c` | **token 通道**（gpu 段用）`seq_id` 硬编码 0；同文件的 `forward_embd_token/tokens` 限「全部 token 同一个 seq」 |
| `slimarc/prototype:shim/qlh_keep_head_gpu.c` | `qlh_kh_seq_rm(handle, seq_id, p0)` **有** seq_id 参数 |
| `keep_head_worker.py:456` | 调用时传 `0`；`keep_head.py:342` 与 `engine.py:202` 一路上都没有 seq 维度 |

### 链路：位置是一维的

- `protocol.py:31-36` 六种消息；meta 是自由 dict，加字段不改格式。
- `stage_worker.py:136-138` 只读 `n_tokens/n_past`；`:259-261` `op=seq_rm` → `engine.seq_rm(n_past)`；`:128-131` `RESET` → 全局 `engine.reset()`。
- `stage_worker.py:61-89` 串行 accept：一个 worker 同时只服务一个连接。**但这条不是阻塞项**——多序列可以走一条连接（meta 带 seq_id），客户端本来就是串行的。
- `chain_client.py:1000-1046` `_walk(clients, tokens, n_past)`：全链路共享一个位置；`:1352` 回滚、`:1684` reset 都遍历所有段，两段同步是现成的。
- `server.py:92-97,150,159`：`ThreadingHTTPServer` 收并发，但一把锁串行化，且每个请求 `reset()`。

**显存增量实测**（`.qlh_tmp/clean-igpu-ncmoe40-v.log:2141,2186`，整模 40 层、ctx 512、`n_rs_seq=0`）：

```
llama_kv_cache:        10.00 MiB (512 cells, 10 layers, 1/1 seqs)
llama_memory_recurrent: 62.81 MiB (1 cells, 40 layers, 1 seqs 0 rs_seq)
```

KV 总量不随 seq 数增长（非 unified 时 ctx 被切分，unified 时池共享）；**RS 每 seq 一份，约 62.81 MiB/seq（ctx 无关）**。开投机时按 `n_rs_seq+1` 再乘：`docs/SPECULATION.md:61` 的 `SA_N_RS_SEQ=4` 下 head 251.25 + tail 62.81 MiB，2 seq 推算 `(251.25 + 62.81) × 2 = 628.12` MiB。

## 三条路径

| 路径 | 做法 | 改动面 | 代价 | 人天（推算） |
| --- | --- | --- | --- | --- |
| **A 单连接多 seq** | 装载放开 `n_seq_max=2`；shim 加 token 通道多 seq 前向；协议 meta 加 `seq_id`；客户端按 seq 记位置 | `keep_head_worker.py` 2 行、shim +40 行 C、`keep_head/engine/stage_worker` 约 30 行、`chain_client` 约 60 行、`server.py` 会话→seq 映射约 80 行 | 显存 +63 MiB/seq（不投机）；decode 仍串行，收益只在前缀复用 | **3~4** |
| **B 每 seq 一个 context** | 每段起 N 个 worker，一条连接绑一个 context | 内核零改动，但拓扑、启动、预算、端口全要扩；权重靠 OS 页缓存共享 | 每 seq 多一份 compute buffer + RS + KV：`docs/MEMORY-LIMITS.md` §2 的每段固定开销 640 MiB 会乘 N | **4~6** |
| **C state 存盘/重放** | 导出 `llama_state_seq_get_data/set_data`（`include/llama.h:900-915`），切换时整体换 state | shim +60 行（4 个导出）；Python 侧只用 seq 0 | 每次切换 memcpy 约 140 MiB（4096 ctx，全模），几十 ms；对每步 48 ms 的链路是可感知的抖动 | **2~3** |

真并发混批（多 seq 的 token 进同一个 ubatch 换吞吐）不建议：要同时满足 `equal_seqs`、batch 内 seq 递增（`src/llama-batch.cpp:648`）、kq_mask 按 `n_stream` 分块（`src/llama-graph.cpp:37`）、per-seq 采样；而且 DFlash 的 draft context 固定 `n_seq_max=1`（`slimarc/prototype:shim/qlh_keep_head_gpu.c`），混批后它按行取 `layer_inp` 会对不上 seq。**8~12 人天（推算）**，且收益是吞吐不是延迟。

## 只做「2 个 seq 轮流用」的最小改动

按依赖顺序，除 `server.py` 外共 5 处：

| # | 位置 | 改什么 | 量（推算） |
| --- | --- | --- | --- |
| 1 | `keep_head_worker.py:240` | 从 `SA_N_SEQ_MAX` 读，替掉硬编码的 `1` | 2 行 |
| 2 | `slimarc/prototype:shim/qlh_keep_head_gpu.c` | 新增 token 通道多 seq 前向：照抄 `qlh_kh_forward`，改用 `qlh_fill_explicit` 填 `n_seq_id/seq_ids/positions` | +35 行 C |
| 3 | `keep_head_worker.py:456` → `keep_head.py:342` → `engine.py:202` | `seq_rm` 一路加 `seq_id`（shim 的 ABI 已有，只是被传 0） | 约 12 行 |
| 4 | `stage_worker.py:136-138` 与 `:259-261` | forward / `op=seq_rm` 从 meta 读 `seq_id`，缺省 0 即完全向后兼容；`RESET` 保持全局 | 约 10 行 |
| 5 | `chain_client.py:470-497,1000-1046` | `StageClient` 与 `_walk` 透传 `seq_id`；调用方把单个 `n_past` 换成每 seq 一份位置 | 约 50 行 |

表中行数按现有调用链估算，非实测。

`protocol.py` 一行不动（meta 是自由 dict）。真正的大头在 `server.py:150-159` 的「会话 → seq」映射与去掉每请求 `reset()`（约 80 行），但那是产品层，不参与「内核能不能做」的判断。

## 建议：有条件做，走 A

条件按重要性：

1. **收益场景必须确认为 `server.py` 的多轮前缀复用**。若目标是吞吐，别做——瓶颈在跨机往返与串行 decode，多序列一点不改善。
2. **先花 0.5 人天做等价性判据**：2B 整模、`n_seq_max=2`、`SA_N_RS_SEQ=0`，同一段 prompt 交替 decode 两条 seq（覆盖反序首次 decode、清空一条后重建），与各自单独跑逐位对比 token 与 hidden（判据沿用 `slimarc/prototype:checks/verify_equiv.py`）。代码读下来结构是自洽的，但有两处只靠约定：`seq_rm` 的部分回滚用 `cells[seq_id].tail` 取本 seq 的 cell（`:193`），而 `cells` 同时兼作「每 seq 一条元数据」的表——这个双重身份只在 `size == n_seq_max`（`src/llama-model.cpp:3226`）且 `seq_id < size`（`:187` 与 `:545` 两处检查）时成立；`rs_idx` 是消费式的（`s_copy` 读一次清零，`:1370-1375`），多 cell 交错 decode 时是否每次都取对，只有跑一遍才算数。
3. **两段的 `n_seq_max` 与 `n_ctx` 必须同值**。`n_ctx_seq = n_ctx / n_seq_max` 在两段各算一次，不一致会让两个 seq 的位置边界错开——这是静默的。
4. **DFlash 先不开**。draft context 是独立 context 且 `n_seq_max=1`（`slimarc/prototype:shim/qlh_keep_head_gpu.c`），多 seq 与 `n_rs_seq` 相乘的显存账要先算清。
5. `RESET` 保持全局，`seq_rm` 加 `seq_id`——回滚本来就是 per-seq 的（`rs_idx`），别做成全局删。

## 证据边界

**代码确认**：上面所有行号；`verify_multiseq.py` 的三条实测（多 seq 请求被清拒、超容量被拒、单序列正常）；`.qlh_tmp/` 下的 `llama_kv_cache` / `llama_memory_recurrent` 显存日志。

**推断未实测**：①两条 seq 交替 decode 与各自单独 decode 逐位等价（路径依据齐全，但没有一次实验）；②`build_rs_shift_snapshots` 在多 seq 下逐 cell 正确（逻辑上 `n_seq_tokens` 与 cell 索引正交，未跑）；③两段 `n_seq_max` 同值即可同步（未在两段拓扑上验）。这三条就是条件 2 要打的靶。
