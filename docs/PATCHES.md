# 四代补丁，以及与原生 RPC 的分界

> 状态：**现行**
> 更新日期：2026-11-07（v8 落地后重导）
> 适用范围：本项目对 `H:\llamacpp-c` 四代补丁的构成，以及与 llama.cpp 原生 RPC 的分界；读者为内核与部署维护者。
> 相关：`SLIM-ARC-接入计划.md`（第三代第二级未接的理由）、`CUT-TOOL.md`（裁段工具退场）、`SEGMENTED-LIMITS.md`（第四代的已知边界）

本项目是 K-Llama 的第二代。两代的差别不在协议层，而在**模型怎么装进显存和内存**：
第一代靠"裁好的层段工件"，第二代靠"整模 + 层范围"。llama.cpp 侧的补丁因此完全不同。

**代码现状**：相对基提交 `e0f142d` 共 53 文件 / **+15200 −199**（`git diff --stat`，fork HEAD
`77adc23`，树 `e790f7e6`），覆盖本文第一代到第四代的全部改动。这 53 个文件拆成**两个可独立
应用的补丁**：

| 补丁 | 包含 | 统计 |
| --- | --- | --- |
| [`docs/llamacpp-split.patch`](llamacpp-split.patch)（主补丁） | 第一、二、四代：层范围加载、分段链、远程段、聚合 memory、checkpoint、server/MTP/DFlash 接线、协议 v7 验证通道与 v8 远程 MTP draft。**不含任何 SLIM-ARC** | 43 文件 / +11656 −198 / 13321 行 / 674.1 KiB |
| [`docs/llamacpp-slimarc.patch`](llamacpp-slimarc.patch)（可选增量） | 第三代 SLIM-ARC prefetch：10 个源文件 + 3 个文件里的 6 处钩子（清单见第三代小节） | 13 文件 / +3544 −1 / 3662 行 / 155.1 KiB |

两者是**叠加**关系，主补丁可以单独用：

```bash
git worktree add /tmp/v1 e0f142d && cd /tmp/v1
git apply docs/llamacpp-split.patch        # 纯分段，无 SLIM-ARC
git add -A && git write-tree               # 必须 = 34cfbd1d（纯分段树）
git apply docs/llamacpp-slimarc.patch      # 再叠增量，回到完整功能
git add -A && git write-tree               # 必须 = e790f7e6（完整树）
```

- 只应用主补丁得到的树是 `34cfbd1d`：它能独立 `--clean-first` 编译，并独立通过 `verify.sh`
  25 项与 `verify_chain_death.py` 36 项——没有 SLIM-ARC 不影响分段链的任何一条验收。
- 叠加两个补丁后 `git write-tree` = `e790f7e6`（fork HEAD 的树），这是"拆分没有改动任何
  内容"的唯一判据，比"能 apply"更强。
- SLIM-ARC 与分段链零耦合：那 10 个文件只 include 彼此和上游 `ggml-backend.h`，不碰本 fork
  新增的任何头；6 处钩子摘掉后，`llama-context.cpp` / `llama-model-loader.cpp` 的对应片段与
  `e0f142d` 逐字节相同。

**历史全文**：`docs/llamacpp-qlh-slimarc.patch`（27 文件 / 4014 行），基提交更早，
只到第三代；保留它是为了对照第一/二代的实现，不能再正向应用。

fork 侧的历史是**两个提交**：上游 `e0f142d` 加一个 `[QLH-SPLIT]`（`77adc23`）。开发期间的
wip 提交已经 squash 掉；当时记录的 squash 前后 tree 值是 `d628e2a8`，与当前 HEAD 的
`e790f7e6` 不同（差异来自 squash 之后又合入的修复）。这项核对只能当历史记录，不能重放。

---

## 第一代（K-Llama / QLH）：层输出通道

改动集中在一处：让上下文多出一个"第 N 层输出"的槽位，供段间接力取数。

- `src/llama-context.cpp`：`layer_inp` 的槽位多分配一个（`n_layer + 1`），
  索引 `lid == n_layer` 即"第 n_layer 层的输入" = **末层输出（`output_norm` 之前）**；
  `set_embeddings_layer_inp` 的断言同步放宽。
- 在 `output_norm` **之前**登记该槽位，覆盖 9 个架构：
  `qwen2` / `qwen35` / `gemma4` / `llama` / `openai-moe` / `qwen3` / `qwen3moe` /
  `qwen35moe` / `qwen3next`。
- 对外是 4 个导出符号，shim 里用到的是
  `llama_set_embeddings_layer_inp()` / `llama_get_embeddings_layer_inp()`。

**为什么不能复用 `t_h_nextn`**：各架构的挂点不一样。`qwen2` 的 `t_h_nextn` 在
`output_norm` 之前，而 `qwen35` 在**之后**（那里的消费方是 MTP head）。拿 `nextn` 当层段
上游，在 Qwen3.5 上会多过一次 RMSNorm，接力的数值首步就分叉。

代价是这套改动要跟着每个架构走，9 个架构就有 9 处挂点。

## 第二代（K-Llama v2）：层范围加载

不再需要"层段工件"，改成直接打开整模、只把属于自己那段读进来。8 个文件，+227/−13：

| 文件 | 改动 |
| --- | --- |
| `include/llama.h` | 新增 `llama_context_params::embeddings_pre_norm` |
| `src/llama-cparams.h` | 内部结构同步该字段 |
| `src/llama-context.cpp` | 默认 `false`，从参数拷贝，打日志 |
| `src/llama-model-loader.h` | `layer_lo` / `layer_hi`、`n_layer_all`、`needs_tok_embd()` / `needs_output()` |
| `src/llama-model-loader.cpp` | `LLAMA_LAYER_RANGE` 解析、`get_weight` 重映射、全局张量的 lazy 标记 |
| `src/llama-model.cpp` | `done_getting_tensors` 传范围、`init_mappings` 按范围、host-only 预读、`size_data` 重算 |
| `src/models/qwen35.cpp` | 在 `build_norm` 之前 tap 出 `t_embd` |
| `src/models/qwen35moe.cpp` | 同上；另有 relay 末层不裁剪（见下文第 4 条机制） |

四个关键机制：

1. **`LLAMA_LAYER_RANGE=lo:hi`**：覆盖 `block_count` 与 `nextn`，让架构以为自己只有 `hi-lo`
   层；`get_weight` 把本地 `blk.i` 映射回源文件的 `blk.lo+i`。
2. **`embeddings_pre_norm`**：接力需要的是 `output_norm` **之前**的末层输出。这个开关把
   tap 点前移，替代第一代的 `layer_inp` 槽位。只改了 qwen35 和 qwen35moe 两个架构。
3. **lazy 全局张量**：head 段不需要 `output`、tail 段不需要 `token_embd`、middle 段两者都不需要。
   不创建就不读，middle 段因此省 1030.62 MiB。
4. **relay 末层不裁剪**（2026-11-03 实测）：`qwen35moe.cpp` 的 `crop_before_nextn`
   在 `n_outputs == 0`（只要 hidden、不要 logits）时会把末层裁成 `[n_embd, 0]`，而
   `t_layer_inp[n_layer]` 登记的正是这条张量，于是 `extract_layer_inputs` 的
   `row_floats == model.hparams.n_embd` 断言失败。修法是新增
   `qlh_keep_full_rows = embeddings_pre_norm || embeddings_layer_inp[n_layer]`，为真时跳过
   这次裁剪，并在 LM head 之前补回 `ggml_get_rows`。其余路径图完全不变。

   这个坑在 Windows 上不会出现——那边靠编译宏 `QLH_ABI_UPSTREAM_HEADER` 把 `batch.logits`
   钉成 1（`n_outputs == n_tokens`），绕开了裁剪。两条路实测 hidden **逐位相同**，但只有
   补丁这条路不依赖编译期定义。

**为什么第二代反而更小**：第一代要 9 个架构各自登记挂点，第二代把"取哪一层的输出"变成
"只加载哪几层"——后者是加载器的事，与架构无关。架构侧只剩 2 处 tap。

---

## 第三代：SLIM-ARC prefetch（2026-10-31，现为可选增量）

上游 SLIM-ARC 是给 Linux 写的，本机是 Windows，所以是**带回移植的接入**，只接它两级里的
第一级（`prefetch`）。第二级（on-demand 加载器）仍未接，理由见
[`SLIM-ARC-接入计划.md`](SLIM-ARC-接入计划.md)。

新增 5 对 / 10 个文件：`slim-arc-prefetch.{h,cpp}`（2085 + 495 行）加它依赖的
`page-range` / `expert-reclaim` / `expert-residency` / `expert-transition`（各一对 `.h/.cpp`），
合计 3479 行。这 10 个文件自包含：只 include 彼此和上游 `ggml-backend.h`，不碰本 fork 新增
的任何头，所以能整块摘下而不牵连分段链。

**这一代不在主补丁里**，只存在于 `docs/llamacpp-slimarc.patch`（13 文件 / +3544 −1），叠在
`docs/llamacpp-split.patch` 之上才回到完整树。6 处钩子（下表按位置归并成 4 行）：

| 文件:行（完整树 `e790f7e6`） | 位置 | 作用 |
| --- | --- | --- |
| `src/CMakeLists.txt:50-60` | `LLAMA_SLIMARC_SOURCES` | 注册 5 个 `.cpp`，并用 `SKIP_UNITY_BUILD_INCLUSION` 挡在 unity 批次外——这批文件带 `WIN32_LEAN_AND_MEAN` 和 POSIX 名字，混进 unity 会污染同批的翻译单元 |
| `src/llama-model-loader.cpp:1587-1615` | `init_mappings()` 末尾 | 只有设了 `SLIM_ARC_PREFETCH` 才构造全局调度器，把每个权重的 `(mmap 基址 + 偏移, 大小, 层号)` 注册进去 |
| `src/llama-context.cpp:1613-1624` | `process_ubatch()`：`ggml_backend_sched_reset()` 之后 | 调度器在装且 `model.n_gpu_layers() == 0` 时，把 `prefetch_scheduler::eval_cb` 装到 sched 上，用户 `cparams.cb_eval` 串在它里面；否则原样用用户回调（即 `e0f142d` 那一行） |
| `src/llama-context.cpp:1638-1644` | `process_ubatch()`：`build_graph()` 之后 | `prepare_graph(gf)` 扫一遍新图，记下每层的第一个节点，运行期由 `eval_cb` 在层边界上发预取 |

表里归并后的 4 行覆盖 6 处：`src/CMakeLists.txt` 的 `set(LLAMA_SLIMARC_SOURCES ...)` 与
`add_library()` 里的 `${LLAMA_SLIMARC_SOURCES}` 各一处，`src/llama-context.cpp:17` 与
`src/llama-model-loader.cpp:8` 的 `#include "slim-arc-prefetch.h"` 各一处，再加两个 `.cpp`
里的代码块。摘掉后 `src/llama-context.cpp` 的 `process_ubatch()` 恢复成 `e0f142d` 的
单行 `ggml_backend_sched_set_eval_callback(sched.get(), cparams.cb_eval, cparams.cb_eval_user_data);`，
`src/llama-model-loader.cpp` 的 `init_mappings()` 末尾恢复成空行加 `}`——两处都与 `e0f142d` 逐字节相同。

**层号怎么来**：上游从图节点名里读 `blk.N`，在本树里取不到——GGML 的 scheduler 在计算前把
每个节点重命名成 `node_%d`，真正的张量名留在**操作数**上。本树改成读 `graph_get_cb()` 给每个
tensor 打的 `-<层号>` 后缀（`ggml_format_name`，上游行为，不是本 fork 新增），`prepare_graph`
再按"层号只会前进"把名字尾数恰好是数字的节点滤掉。按上游写法接时 `calls` 恒为 0；改成读
后缀后 `calls=5`（3 个 token 各一次通知加上 prefill），链路确认通了。

win32 移植面（3 个函数 + 1 个锁包装）：

| POSIX | Windows | 说明 |
| --- | --- | --- |
| `madvise(WILLNEED)` | `PrefetchVirtualMemory` | 语义相同：把这些页读成驻留 |
| `madvise(DONTNEED)` | `DiscardVirtualMemory` | 内容作废、页丢弃，最接近的等价物 |
| `madvise(NORMAL/RANDOM/SEQUENTIAL)` | 空操作 | 只影响内核预读启发式，Windows 没有这个接口 |
| `mincore` | 不支持，返回"未知" | 唯一消费方是专家热锁 LRU，未知即跳过该区间 |
| `sysconf(_SC_PAGESIZE)` | `GetSystemInfo().dwPageSize` | |
| `mlock` / `munlock` | `VirtualLock` / `VirtualUnlock` | |

另有两处 MSVC 兼容：`NOMINMAX`（否则 `windows.h` 的 `max` 宏会把 `std::max` 和
`numeric_limits<T>::max()` 一起打掉），以及用 `_umul128` 拼出 64×64→128 的乘积替代
`unsigned __int128`（MSVC 没有该类型，用在专家 LRU 的打分比较上）。

**当前生效范围**：默认关闭，`SLIM_ARC_PREFETCH=1` 才装调度器。整模在页缓存里时
`issued` 为 0——它只在"段装不下、真的在读盘"的降级形态里才有事可做，与
`SLIM-ARC-接入计划.md` 的触发条件一致。这一代现在只存在于 `docs/llamacpp-slimarc.patch`：
主补丁不含这 10 个文件，所以不叠增量的构建里连相关符号都不存在。

---

## 第四代（现行）：分段链与跨进程

前三代解决"每台机器只装自己那几层"，第四代解决"这么多段怎么跑成一次推理"。一个
`llama_context` 成为链的驱动者：本地段输出 pre-norm hidden，后续段（本地或远程）消费
hidden 继续算，最后一段产 logits。llama-server 的 slot、prompt cache、`seq_rm` 通过聚合
memory 看到整条链，不需要知道段在哪台机器上。

这一代后来还长出了**远程链的投机**：协议 v7 加了逐行 argmax 的验证通道，v8 把 MTP draft 下沉
到远程 tail（`OP_MTP_DRAFT`）。两处都是**增量改协议**，没有动链的驱动结构——所以它们在这张
表里各占一行，而不是另开一代。

**实测数据见 [`SPLIT-MEASUREMENTS.md`](SPLIT-MEASUREMENTS.md)**（checkpoint 复用、MTP、DFlash、
并发、内存、验收覆盖各一节，每条带日志出处与口径）。

| 机制 | 位置 | 要点 |
| --- | --- | --- |
| 层范围进参数 | `include/llama.h` 的 `llama_model_params.split_layer_lo/hi` | 同一进程才能加载多个层范围；`LLAMA_LAYER_RANGE` 降级为兜底 |
| hidden 传递 | `llama_context::decode_split` | 走标准 embedding 路径（`embeddings_pre_norm` + `get_embeddings_ith`），不是 `nextn` 通道 |
| relay 段省 GET_ROWS | `src/llama-graph.cpp` 的 `build_inp_embd` | 没有 `tok_embd` 时不建；那一行被 `ggml_build_forward_order` 强制入图，只看分支选择不够 |
| 聚合 memory | `src/llama-memory-split.{h,cpp}` | 本地段走 `llama_memory_t`、远程段走协议；`seq_pos_*` 问链首，`can_shift` 恒 false |
| 远程段 | `src/llama-split-{proto,net,remote}.*`、`tools/split-stage` | 定长协议 **v8**（v3 定长头 + v4 身份握手 `OP_HELLO` + v5 候选通道 + v7 验证行 + v8 远程 MTP draft）；socket 层两端共用一份；`--host` 绑地址，默认 loopback |
| 远程段进聚合 | `llama_context::split_attach_remote` | 装 `llama_memory_split`，远程指针借给视图；`seq_rm` / `clear` 广播到协议 |
| server 入口 | `common/arg.cpp` 的 `--split-tail-range` / `--split-remote` / `--split-rs-seq` | `--split-remote` 可重复、按链序；与 `--split-tail-range` 互斥 |
| 位置下发 | `decode_split` 的 `split_expand_pos` | 每个 token 的 4 个 M-RoPE section 全量下发，`req.n_pos` 让两端对齐 stride |
| 角色显式声明 | `split-stage --role relay\|tail` | 用源层数校验：relay 要求 `hi < n_layer_src`，tail 要求 `hi == n_layer_src` |
| checkpoint 覆盖全链 | `state_seq_*` 走 `get_memory()`；`llama_memory_split::state_read()` | 远程段先恢复、链首最后；每段带 `pos_max`，落后于快照的一律拒绝 |
| 候选回传（v5） | `common/sampling.cpp` 的 `common_sampler_candidates_check()`、`server-context.cpp` 的 `plan_remote_candidates()` | 普通 decode 让 tail 跑 `logit_bias → penalties → top_n_sigma → top_k` 后只回候选、不回整行 logits；不等价时按白名单回退整行。帧尺寸与等价性实测见 [`SPLIT-MEASUREMENTS.md`](SPLIT-MEASUREMENTS.md) §8 |
| 验证通道（v7） | `DECODE_FLAG_VERIFY_ROWS`（`src/llama-split-proto.h`）、`request_remote_verify_rows()` / `plan_remote_verify_rows()`（`tools/server/server-context.cpp`） | 验证帧的**每行**回一个 int32 argmax，一帧 k+1 行是 4(k+1) B（单帧 993,312 → 48 B）；`MAX_VERIFY_ROWS` 是两端共享常量；要求 `off == 0`（帧内行号 = batch 绝对下标）；不满足就 `spec_blocked` 整个请求关投机，帧内无法改路 |
| 远程 MTP draft（v8） | `OP_MTP_DRAFT`、`DECODE_FLAG_NEXTN_ROWS` / `RSP_FLAG_NEXTN_ROWS`、`common/speculative.cpp` 的 `is_remote` | tail 内部跑 k 步递归（每步输入是上一步的输出），回 k 个 id；`stage_id.has_nextn` 占 `_pad` 且**不参与身份比较**；tail 没有 NextN 时 server 摘掉 draft-mtp 并 WARN，不拦启动 |
| 链死自愈 | `llama_memory_split::try_rebuild()` | 位置 0 的请求 → 逐个 `reconnect()`（含重握手）→ 整链 `clear()`；失败保持 dead + 5 s 节流 |
| MTP 投机在本地链上 | `tools/server/server-context.cpp` 的 `ctx_tail` | draft context 建在 tail 的模型上——head 段 nextn 为 0，建在 head 上会启动失败（远程链走 v8，没有 `ctx_dft`） |
| DFlash 8 路 tap 跨两段 | `common/speculative.cpp` 的 `tap_ctx` / `tap_lid` | 源层号按 head/tail 拆成（context, 局部层号）；tail 的 tap 是进程内读，不加往返 |
| tail 的 logits 缓冲按角色 | `common/common.cpp` 的 `n_outputs_max` | head 每 token 一行 hidden；tail 只为实际输出行（1 + k）分配 |
| socket 超时 | `src/llama-split-net.h` 的 `set_timeout` | 每 send/recv 120 s，只兜"整体停摆"；stage 侧不设，靠对端关闭 |
| 单独层范围的拒绝 | `common_init_from_params()`（`common/common.cpp`） | 范围结束早于末层、又没有后续段时，加载后立即报错；范围不从 0 开始且 `split_range_is_stage` 为假时同样拒绝 |

四处语义值得单独记，都改过：

- **`--split-rs-seq` 不再是正确性必需**。早先 `n_rs_seq = 0` 的段会让截断式 `seq_rm` 失败，
  而 `common_context_seq_rm` 把 false 当致命错误，于是必崩（当时远程链还没有 `remote_failure`
  这条「不 abort」的出口）；现在 checkpoint 覆盖全链，默认 0 是可服务的配置。这个参数只在
  投机时要抬高下限（一轮回滚 `n_max` 个位置），
  而 `split-stage` 仍强制 `--rs-seq N`：server 的 `common_context_can_seq_rm()` 会回退到探测
  `seq_rm`，0 快照的段会让这次探测失败（远端拒绝，链随即被标死），理由写在 `split-stage`
  的 `--rs-seq` 必填检查里。
- **失败不区分"对端拒绝"与"没连上"**。两者对链是同一件事——位置要分叉了。`decode` 的
  传输失败降级成 `-3` 让请求失败；`seq_rm` / `clear` 失败都写 `remote_failure` 把链标死，
  `common_context_seq_rm()` 对已死的远程链只记日志并 return（本地链仍 abort），此后
  `decode_split()` 在算任何东西之前就查这个状态并返 `-3`——请求失败，进程活着。
- **链死能自愈**（`llama_memory_split::try_rebuild()`）。下一个从位置 0 开始的 batch 逐个
  `reconnect()` → 整链 `clear()` → 恢复数据通路超时；失败保持 dead 并做 5 s 节流。闸门是
  结构性的：重建后唯一一致的状态是「所有段都空」，空链只能从 0 写。重连必须重新握手
  （`OP_HELLO`），接错的 stage 让重建失败，而不是被当成链好了。
- **checkpoint 覆盖全链之后**，3103 token 历史的第二轮从 19.85 s 降到 0.71 s（实测数据见
  [`SPLIT-MEASUREMENTS.md`](SPLIT-MEASUREMENTS.md) §1；那一节同时也记了两个数字各自的口径）。
- **v8 的 per-frame 槽位顺序**：`split_remote_reset()` 清掉所有 per-frame 槽位，包括新加的
  h_nextn 行。所以 `store_nextn` 必须在三个回答分支（候选 / 验证行 / 整行）**之后**执行——
  原来先存行、后存 logits，行被 `store_logits()` 的 reset 抹掉，结果 stage 回 2048 floats、
  head 收到 0 行，**draft 静默消失**（不报错、不变慢，只是不投机）。

验证（Qwen3.5-2B，`0:20` + `20:24`，独立 `build-split`）：

- N=1 直通：默认 / `--split-layer-range 0:24` / `0:20` 三种配置输出与改动前逐字节相同
- N=2 本地双段：head 输出与整模层 20 输入逐位相同；tail 单独跑与 baseline 相同；链路
  logits 逐位相同；`seq_rm` 广播后两段的 `pos_max` 同步下降
- N=2 经 llama-server：两个端口的 `reasoning_content` 一致
- 跨进程：本地 head + 远程 `split-stage`，logits 逐位相同；服务器的 `--split-remote` 走同一条路径
- 投机：2026-11-06 实测，本地双段链上 MTP 与 DFlash 的输出与不投机逐位相同（贪心）；
  远程链的三组（不投机 / ngram / MTP）输出逐字节相同（`verify_mtp_e2e.sh`），远程 tail
  内部递归 3 步与整模 + MTP 逐个相同（`verify_mtp.sh`）
- 远程链的死亡与自愈：`build-split/verify_chain_death.py` 36 项——挂起 stage 后请求 500 而 head
  存活、`/health?deep=1` 给 503、只重启 stage 就自愈（`prompt_n` / `cache_n` / 输出三项同时成立）、
  接错 stage 被 `OP_HELLO` 拒绝；候选通道的等价性见
  [`SPLIT-MEASUREMENTS.md`](SPLIT-MEASUREMENTS.md) §8

第四代仍成立的边界（`embeddings_pre_norm` 只覆盖两个架构、远程链的投机只做贪心且 k=2 是甜点、
DFlash 在远程链上仍缺 `OP_LAYER_INP`、级联在 server 侧不做、远程链是单序列 append-only、
checkpoint 每段约 12.56 MiB 过网、浅 `/health` 不反映链路状态（要问 `?deep=1`）、单独层范围的
拒绝只覆盖一条加载路径）逐条记在 [`SEGMENTED-LIMITS.md`](SEGMENTED-LIMITS.md) §2，那里写的是
代价而不是 TODO。

---

## 附：工具层的参数校验（非分布式必需）

`common/arg.cpp`，14 行。逻辑上与三代补丁无关，是为 `--spec-type ngram-simple` 加一条交叉
校验；它出现在这份 patch 里，是因为 patch 按工作区快照整体生成。

`ngram-map.cpp:95-101` 的实现是：`copy_max = min(size_m, 剩余历史)`，只要
`copy_max < n_draft_min` 就返回空草稿，而 `n_draft_min` 就是 `size_ngram`。所以
**`size_m < size_n` 是恒空配置**——一个草稿都产不出来，且没有任何警告。上游默认值 12/48
（`m >> n`）永远走不到这个分支，但用户单独调这两个参数时会。

校验放在 `common_params_parse` 解析完成之后而不是各选项的回调里，因为两个参数在命令行上
可以任意顺序出现，放在回调里会让 `-size-n 60 -size-m 100` 误报。

验证（`llama-server` 参数解析，不需要真模型）：

    -size-n 4 -size-m 3  ->  "ngram size M (3) must be >= ngram size N (4): smaller values
                              produce no drafts at all (ngram-map.cpp:95-101)"
    -size-n 2 -size-m 7  ->  通过参数校验

这条与上游无关：本项目的有效组合是 `m > n`，校验只在本仓的启动参数上生效，
提不提上游都不改变本地行为。

---

## 与 llama.cpp 原生 RPC 的分界

原生 RPC（`-DGGML_RPC=ON`）也能跨机，但两者解决的不是同一个问题。RPC 的协议命令里，
`SET_TENSOR` / `SET_TENSOR_2D` 上传张量数据，`GRAPH_COMPUTE` 让远端执行子图，`GET_TENSOR`
取回结果。也就是说：

**RPC 让远端当"远程 GPU"，主机仍然要有完整的模型。** 权重从主机通过网络发到远端
（`SET_TENSOR`），主机维护全局计算图，远端只负责执行被派过去的那部分 op。

**本项目让每个节点当"一段模型"，没有中心。** 整模文件在每个节点本地，各自 mmap 后只
touch 自己那几层，段间只传 `n_embd` 个浮点。

| 维度 | 原生 RPC | 本项目 |
| --- | --- | --- |
| 权重从哪来 | 主机通过 `SET_TENSOR` 发给远端 | 各节点本地读整模的对应层 |
| 首次跨机流量 | 权重总量（GB 级） | 0，只要本地有整模 |
| 主机必须装下整模 | 是 | 否（只装自己那几层） |
| 每步流量 | 激活 + 图指令 | hidden 一个向量（FP32，2048 × 4 B = 8 KB） |
| 计算图 | 主机维护全局图 | 每段是独立进程，无全局图 |
| 异构 | 需要相同的 ggml backend | 自定义 TCP 协议，可跨不同 CPU/GPU |
| 层范围加载 | 不支持，要自己切模型 | `LLAMA_LAYER_RANGE` 原生支持 |

一句话：RPC 是**把设备虚拟化**，模型仍然是一份；本项目是**把模型分段**，每段各自持有。
前者适合"一台主机 + 若干算力节点"，后者适合"几台都装不下整模的机器"。

代价也要说清楚：本项目的段是黑盒，段内不能跨机器切分算子；每一段必须能单独装下。
而 RPC 可以把任意 op 派到远端，粒度更细。
