# 四代补丁，以及与原生 RPC 的分界

> 状态：**现行**
> 更新日期：2026-11-09（device-buffer 修复后重导：52 文件 / +17831 −230，tree `982ccecd`）
> 适用范围：本项目对 `H:\llamacpp-c` 四代补丁的构成，以及与 llama.cpp 原生 RPC 的分界；读者为内核与部署维护者。
> 相关：`SLIM-ARC-接入计划.md`（第三代第二级未接的理由）、`CUT-TOOL.md`（裁段工具退场）、`SEGMENTED-LIMITS.md`（第四代的已知边界）

本项目是 K-Llama 的第二代。两代的差别不在协议层，而在**模型怎么装进显存和内存**：
第一代靠"裁好的层段工件"，第二代靠"整模 + 层范围"。llama.cpp 侧的补丁因此完全不同。

**代码现状**：相对基提交 `f477f795c` 共 **52 文件 / +17831 −230**（`git diff --stat`，fork HEAD
`0c5c3bdd`，树 `982ccecd`），覆盖本文第一代到第四代的全部改动。这 52 个文件拆成**两个可独立
应用的补丁**：

| 补丁 | 包含 | 统计 |
| --- | --- | --- |
| [`docs/llamacpp-split.patch`](llamacpp-split.patch)（主补丁） | 第一、二、四代：层范围加载、分段链、远程段、聚合 memory、checkpoint、server/MTP/DFlash/DSpark 接线、协议 v7 验证通道 / v8 远程 MTP draft / v9 远程 layer_inp tap、grammar / rbudget 兼容路径、`seq_rm` 失败显式化。**不含任何 SLIM-ARC** | **42 文件** / +14287 −229 / 16166 行 / 840.6 KiB |
| [`docs/llamacpp-slimarc.patch`](llamacpp-slimarc.patch)（可选增量） | 第三代 SLIM-ARC prefetch：10 个源文件 + 3 个文件里的 6 处钩子（清单见第三代小节） | 13 文件 / +3544 −1 / 3662 行 / 155.2 KiB |

两个补丁的文件数、行数、增删行**逐项相加等于 52 / +17831 −230**：42 + 13 里有 3 个文件重叠
（`src/CMakeLists.txt`、`src/llama-context.cpp`、`src/llama-model-loader.cpp` 在两边各出一份
diff），所以并集仍是 52 个文件。这一项是拆分没漏东西的算术校验。**这一轮 `src/models/dflash.cpp`
进了变更集**（上一版它与上游逐字节相同，现在多了一段诊断输出），所以文件数 51 → 52、主补丁
41 → 42、增删行 +17775 −227 → +17831 −230。

**基提交为什么是 `f477f795c` 而不是更早的 `e0f142d`**：本轮把 fork 整体 rebase 到上游
193 个提交之后的 `f477f795c`（细节见「与上游 193 个提交的重合」一节）。旧基线 `ec25cedda`
（树 `e14b92c9`，68 文件 / +18086 −504）仍在 `master` 上保留，但它已经不是补丁的基底。

两者是**叠加**关系，主补丁可以单独用：

```bash
git worktree add /tmp/v1 f477f795c && cd /tmp/v1
git apply docs/llamacpp-split.patch        # 纯分段，无 SLIM-ARC
git add -A && git write-tree               # 必须 = d28e10df（纯分段树）
git apply docs/llamacpp-slimarc.patch      # 再叠增量，回到完整功能
git add -A && git write-tree               # 必须 = 982ccecd（完整树，= fork HEAD 的树）
```

**补丁文件必须按字节保存（仓根 `.gitattributes` 的 `*.patch -text`）**。`core.autocrlf=true`
下把补丁转成 CRLF 之后**照样 apply 成功**——只有一堆 trailing-whitespace 警告——但文件末尾
`\ No newline at end of file` 那一处会多出一个裸 CR，产出的树与文档里的 tree hash 不再相等。
静默错，而且是「验收命令跑了、也过了、结果不对」那一类。这条是 rebase 那一轮实测撞到的
（CRLF 版主补丁得到 `d679f09b`，正确是 `1dcbce39`）；这一轮的对应判据是「主补丁 → `d28e10df`，
叠加 → `982ccecd`」，导出后已用 `grep -c $'\r'` 确认为 0 个 CR。

- 只应用主补丁得到的树是 `d28e10df`：它能独立 `--clean-first` 全量编译（188 个目标，
  `.qlh_tmp/build.bat --clean-first` 退出 0），并独立通过 `verify.sh` **全十一组 52 项
  PASS=52 FAIL=0 SKIP=0** 与 `verify_chain_death.py` **PASS=36 FAIL=0**。没有 SLIM-ARC 不影响
  分段链的任何一条验收。2026-11-09 在只有主补丁的工作树（`H:\pq-new`，基底 `f477f795c`）上实测，
  日志见 `.qlh_tmp/pure-split-acceptance/verify_pure_2026-11-09_v52.log` 与
  `chain_death_pure_2026-11-09_v52.log`。
- 上一版（`1dcbce39`，44 项）的同一棵树上**第一次**跑出的是 `PASS=34 FAIL=10`，两处都是环境：
  ① 组 5 的 `-np 4` 在启动瞬间死掉，日志 0 字节（同一命令单跑两次都过，20 个请求一个都没发出，
  所以不是数值问题）；② 组 10 的 dflash 十项里九条红（只有「整模 tap 落盘」那条过），
  `taps_dflash_chain.log` 里只有 `dflash-taps.exe: Device or resource busy`——整模那趟刚退出、
  文件锁未释放。run2 用同一份二进制、同一份模型得到 44/0。判据永远是「日志里有没有数值不一致」，
  不是「这一趟绿不绿」。
- 叠加两个补丁后 `git write-tree` = `982ccecd`（fork HEAD 的树），这是"拆分没有改动任何
  内容"的唯一判据，比"能 apply"更强。
- SLIM-ARC 与分段链零耦合：那 10 个文件只 include 彼此和上游 `ggml-backend.h`，不碰本 fork
  新增的任何头；6 处钩子摘掉后，`llama-context.cpp` / `llama-model-loader.cpp` 的对应片段与
  `f477f795c` 逐字节相同（重导时用字符串精确匹配摘除，每处都断言"恰好出现一次"）。

**历史全文**：`docs/llamacpp-qlh-slimarc.patch`（27 文件 / 4014 行），基提交更早，
只到第三代；保留它是为了对照第一/二代的实现，不能再正向应用。

fork 侧的历史是**两个提交**：上游 `f477f795c` 加一个 `[QLH-SPLIT]`（`0c5c3bdd`，树 `982ccecd`；
这一版比 `ed6f86b99` 多了 device-buffer 的崩溃修复与组 11，见第四节后的「device-buffer」小节）。
旧的一对（`e0f142d` + `ec25cedda`，树 `e14b92c9`）保留在 `master` 分支上作为历史，不再用于生成
补丁。开发期间的 wip 提交已经 squash 掉；更早记录的 squash 前后 tree 值是 `d628e2a8`，与那一代
HEAD 的 `e14b92c9` 不同（差异来自 squash 之后又合入的修复与两次上游移植）。这项核对只能当历史
记录，不能重放。

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

| 文件:行（完整树 `982ccecd`） | 位置 | 作用 |
| --- | --- | --- |
| `src/CMakeLists.txt:50-60` | `LLAMA_SLIMARC_SOURCES` | 注册 5 个 `.cpp`，并用 `SKIP_UNITY_BUILD_INCLUSION` 挡在 unity 批次外——这批文件带 `WIN32_LEAN_AND_MEAN` 和 POSIX 名字，混进 unity 会污染同批的翻译单元 |
| `src/llama-model-loader.cpp:1587-1615` | `init_mappings()` 末尾 | 只有设了 `SLIM_ARC_PREFETCH` 才构造全局调度器，把每个权重的 `(mmap 基址 + 偏移, 大小, 层号)` 注册进去 |
| `src/llama-context.cpp:1778-1793` | `process_ubatch()`：`ggml_backend_sched_reset()` 之后 | 调度器在装且 `model.n_gpu_layers() == 0` 时，把 `prefetch_scheduler::eval_cb` 装到 sched 上，用户 `cparams.cb_eval` 串在它里面；否则原样用用户回调（即 `f477f795c` 那一行） |
| `src/llama-context.cpp:1803-1811` | `process_ubatch()`：`build_graph()` 之后 | `prepare_graph(gf)` 扫一遍新图，记下每层的第一个节点，运行期由 `eval_cb` 在层边界上发预取 |

表里归并后的 4 行覆盖 6 处：`src/CMakeLists.txt` 的 `set(LLAMA_SLIMARC_SOURCES ...)` 与
`add_library()` 里的 `${LLAMA_SLIMARC_SOURCES}`（行 65）各一处，`src/llama-context.cpp:17` 与
`src/llama-model-loader.cpp:8` 的 `#include "slim-arc-prefetch.h"` 各一处，再加两个 `.cpp`
里的代码块。摘掉后 `src/llama-context.cpp` 的 `process_ubatch()` 恢复成 `f477f795c` 的
单行 `ggml_backend_sched_set_eval_callback(sched.get(), cparams.cb_eval, cparams.cb_eval_user_data);`，
`src/llama-model-loader.cpp` 的 `init_mappings()` 末尾恢复成空行加 `}`——两处都与 `f477f795c` 逐字节相同。

行号是 `0c5c3bdd`（完整树）上的。**换基底之后必须重新定位**：上游那些重构动过
`process_ubatch()`，更早一轮这两块在 1613 / 1638，rebase 后到了 1762 / 1787，这一轮修
`supports_embd_row()` 又往下推了 16 行，现在是 1778 / 1803。摘除脚本
（`.qlh_tmp/strip_hooks.py`）用的是**整块字符串精确匹配 + 「恰好出现一次」断言**，所以行号漂了
不会静默摘错——对不上就直接失败，这是它唯一需要的健壮性。

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
到远程 tail（`OP_MTP_DRAFT`），v9 又把 DFlash / DSpark 的 `layer_inp` tap 挂到 decode 回复上，
随后还补了 grammar 与 reasoning budget 两条兼容路径。它们都是
**增量改协议/加判据**，没有动链的驱动结构——所以在这张表里各占一行，而不是另开一代。

**收益结论按终态写**：能做，但 ROG 五类真实 prompt 上 MTP k=2 中位只有 **+6%**（区间
+8.4% ~ −1.2%），弱机器（Surface）与回环上是负的；判据不是「与不投机逐字节相同」，而是
「无回退时与**整模投机**逐字节相同」。数字与归因见
[`SPLIT-MEASUREMENTS.md`](SPLIT-MEASUREMENTS.md) §9、[`SEGMENTED-LIMITS.md`](SEGMENTED-LIMITS.md) §2.2。

**实测数据见 [`SPLIT-MEASUREMENTS.md`](SPLIT-MEASUREMENTS.md)**（checkpoint 复用、MTP、DFlash、
并发、内存、验收覆盖各一节，每条带日志出处与口径）。

| 机制 | 位置 | 要点 |
| --- | --- | --- |
| 层范围进参数 | `include/llama.h` 的 `llama_model_params.split_layer_lo/hi` | 同一进程才能加载多个层范围；`LLAMA_LAYER_RANGE` 降级为兜底 |
| hidden 传递 | `llama_context::decode_split` | 走标准 embedding 路径（`embeddings_pre_norm` + `get_embeddings_ith`），不是 `nextn` 通道 |
| relay 段省 GET_ROWS | `src/llama-graph.cpp` 的 `build_inp_embd` | 没有 `tok_embd` 时不建；那一行被 `ggml_build_forward_order` 强制入图，只看分支选择不够 |
| 聚合 memory | `src/llama-memory-split.{h,cpp}` | 本地段走 `llama_memory_t`、远程段走协议；`seq_pos_*` 问链首，`can_shift` 恒 false |
| 远程段 | `src/llama-split-{proto,net,remote}.*`、`tools/split-stage` | 定长协议 **v9**（v3 定长头 + v4 身份握手 `OP_HELLO` + v5 候选通道 + v7 验证行 + v8 远程 MTP draft + v9 远程 layer_inp tap）；socket 层两端共用一份；`--host` 绑地址，默认 loopback |
| 远程段进聚合 | `llama_context::split_attach_remote` | 装 `llama_memory_split`，远程指针借给视图；`seq_rm` / `clear` 广播到协议 |
| server 入口 | `common/arg.cpp` 的 `--split-tail-range` / `--split-remote` / `--split-rs-seq` | `--split-remote` 可重复、按链序；与 `--split-tail-range` 互斥 |
| 位置下发 | `decode_split` 的 `split_expand_pos` | 每个 token 的 4 个 M-RoPE section 全量下发，`req.n_pos` 让两端对齐 stride |
| 角色显式声明 | `split-stage --role relay\|tail` | 用源层数校验：relay 要求 `hi < n_layer_src`，tail 要求 `hi == n_layer_src` |
| checkpoint 覆盖全链 | `state_seq_*` 走 `get_memory()`；`llama_memory_split::state_read()` | 远程段先恢复、链首最后；每段带 `pos_max`，落后于快照的一律拒绝 |
| 候选回传（v5） | `common/sampling.cpp` 的 `common_sampler_candidates_check()`、`server-context.cpp` 的 `plan_remote_candidates()` | 普通 decode 让 tail 跑 `logit_bias → penalties → top_n_sigma → top_k` 后只回候选、不回整行 logits；不等价时按白名单回退整行。帧尺寸与等价性实测见 [`SPLIT-MEASUREMENTS.md`](SPLIT-MEASUREMENTS.md) §8 |
| 验证通道（v7） | `DECODE_FLAG_VERIFY_ROWS`（`src/llama-split-proto.h`）、`request_remote_verify_rows()` / `plan_remote_verify_rows()`（`tools/server/server-context.cpp`） | 验证帧的**每行**回一个 int32 argmax，一帧 k+1 行是 4(k+1) B（单帧 993,312 → 48 B）；`MAX_VERIFY_ROWS` 是两端共享常量；要求 `off == 0`（帧内行号 = batch 绝对下标）；不满足就 `spec_blocked` 整个请求关投机，帧内无法改路 |
| 远程 MTP draft（v8） | `OP_MTP_DRAFT`、`DECODE_FLAG_NEXTN_ROWS` / `RSP_FLAG_NEXTN_ROWS`、`common/speculative.cpp` 的 `is_remote` | tail 内部跑 k 步递归（每步输入是上一步的输出），回 k 个 id；`stage_id.has_nextn` 占 `_pad` 且**不参与身份比较**；tail 没有 NextN 时 server 摘掉 draft-mtp 并 WARN，不拦启动 |
| grammar / rbudget 兼容 | `common/sampling.cpp` 的 `common_sampler_grammar_rejects()` / `common_sampler_rbudget_rejects()`、`common/reasoning-budget.cpp` 的 `common_reasoning_budget_rejects()`、`tools/server/server-context.cpp` 的 `skip_draft_once` | draft 不看这两个采样器，所以接受率会掉（无 grammar 2.76 → 1.46–1.73 token/帧）：head 侧对每个候选 token 做**单 token 自检**（只掩码不推进状态），被拒就**整帧回退**（撤 KV + prompt + sampler）；预算耗尽只收紧不放宽；`skip_draft_once` 防一条路径空转 |
| 远程 layer_inp tap（v9） | `DECODE_FLAG_LAYER_INP` / `RSP_FLAG_LAYER_INP`（`src/llama-split-proto.h`）、`request_remote_layer_inp()` / `plan_remote_layer_inp()`（`tools/server/server-context.cpp`）、tail 侧 `layer_tap_out`（`tools/split-stage/main.cpp`） | DFlash / DSpark 的 tap 是「这一帧 tail 隐藏行的附带品」，所以复用 decode 回复而不是新开 op：请求体末尾追加 `layer_inp_req{n_taps,n_embd}` + n_taps 个 **int32 源层号**（不是 stage 本地层号，于是同一份请求能发到链上每一段），回复体最后一个 n_taps 个 id + `n_rows × n_taps × n_embd` 个 f32；`rsp_header._pad` 的一个字节改成 `n_taps`（尺寸仍 32，前四段解析与 v8 逐字相同）；`MAX_LAYER_INP_TAPS = 16`；某帧 tap 缺失只让**本请求**停投机，不 send_error、不算链死 |
| device-buffer 可用性（v9 回归的修复） | `src/llama-context.cpp` 的 `supports_embd_row()`、`common/speculative.cpp` 的每帧判定 | 谓词同时回答「buffer 存在」与「宽度 == `n_embd_inp_enc()`」；`decode()` 准入、`graph_params()` 的 `layer_inp_src`、batch 构造、每帧投机判定问同一句，见上面「device-buffer」小节 |
| 两个接线缺口（v9 同期） | `tools/server/server-context.cpp` 的 `ctx_tail_remote` 置位条件、`tools/split-stage/main.cpp` 的三条回包分支 | ① `ctx_tail_remote` 原来只对 MTP 置位，DFlash/DSpark 走 `has_draft`，tap 被解析到 head 上下文 → `src/llama-context.cpp:1425` 的 `GGML_ASSERT(lid <= model.hparams.n_layer())`；② plain 帧原来是 nextn 优先、verify/candidate 才是 tap 优先，而一帧只能声明一个 `n_embd` 宽度，于是 prefill 帧回了 nextn、把 tap 吞掉（接受率掉到 0.111 而不是崩）。两处修完，远程链 DFlash/DSpark **零损失** |
| split-stage 的线程 / ubatch | `tools/split-stage` 的 `-t/--threads`、`-tb/--threads-batch`、`-ub/--ubatch-size` | `GGML_DEFAULT_N_THREADS = 4` **且 `n_threads_batch` 也是 4、不会自动跟随**；`-t` 同时设两者（head 的 `-t` 不过网），`-ub` 是测量旋钮（让验证帧在帧内逐行跑）。Surface 上不投机 +3.5%、MTP +7% |
| 链死自愈 | `llama_memory_split::try_rebuild()` | 位置 0 的请求 → 逐个 `reconnect()`（含重握手）→ 整链 `clear()`；失败保持 dead + 5 s 节流 |
| MTP 投机在本地链上 | `tools/server/server-context.cpp` 的 `ctx_tail` / `ctx_tail_remote` | draft context 建在 tail 的模型上——head 段 nextn 为 0，建在 head 上会启动失败（远程链走 v8 的 `OP_MTP_DRAFT`，没有 `ctx_dft`）；DFlash/DSpark 的 tap 也必须解析到 `ctx_tail_remote`，见上面 v9 那一行 |
| DFlash / DSpark 的 tap 跨两段 | `common/speculative.cpp` 的 `tap_ctx` / `tap_lid` | 源层号按 head/tail 拆成（context, 局部层号）；本地链的 tail tap 是进程内读、不加往返，远程链的 tail tap 走 v9 `OP_LAYER_INP`（附加在 decode 回复上） |
| tail 的 logits 缓冲按角色 | `common/common.cpp` 的 `n_outputs_max` | head 每 token 一行 hidden；tail 只为实际输出行（1 + k）分配 |
| socket 超时 | `src/llama-split-net.h` 的 `set_timeout` | 每 send/recv 120 s，只兜"整体停摆"；stage 侧不设，靠对端关闭 |
| 单独层范围的拒绝 | `common_init_from_params()`（`common/common.cpp`） | 范围结束早于末层、又没有后续段时，加载后立即报错；范围不从 0 开始且 `split_range_is_stage` 为假时同样拒绝 |

六处语义值得单独记，都改过：

- **`seq_rm` 失败不再静默**（2026-11-08）。`common_memory::seq_rm` / `common_context_seq_rm`
  从 `void` → `bool`，五个调用点按场景降级（全量重算 / 结束响应 / 请求失败），不再
  `GGML_ABORT`。触发它的是上游 hybrid 的短路（先删 recurrent，失败就 `return false`，
  attention KV 一行都不删；server 默认 `n_rs_seq=0` 下 partial 分支恒不成立），上游有
  open PR #28007 与 issue #31656，所以不另提 issue。两个坑——链死时**不能**结束请求
  （会跳过 `try_rebuild` 唯一触发的那次 pos 0 decode，自愈永久失效）、`p0 > pos_max`
  的空删除**是成功**——见 [`SEGMENTED-LIMITS.md`](SEGMENTED-LIMITS.md) §2.9。
- **`--split-rs-seq` 不再是正确性必需**。早先 `n_rs_seq = 0` 的段会让截断式 `seq_rm` 失败，
  而 `common_context_seq_rm` 把 false 当致命错误，于是必崩（当时远程链还没有 `remote_failure`
  这条「不 abort」的出口）；现在 checkpoint 覆盖全链，默认 0 是可服务的配置。这个参数只在
  投机时要抬高下限（一轮回滚 `n_max` 个位置），
  而 `split-stage` 仍强制 `--rs-seq N`：server 的 `common_context_can_seq_rm()` 会回退到探测
  `seq_rm`，0 快照的段会让这次探测失败（远端拒绝，链随即被标死），理由写在 `split-stage`
  的 `--rs-seq` 必填检查里。
- **失败不区分"对端拒绝"与"没连上"**。两者对链是同一件事——位置要分叉了。`decode` 的
  传输失败降级成 `-3` 让请求失败；`seq_rm` / `clear` 失败都写 `remote_failure` 把链标死，
  `common_context_seq_rm()` 对已死的远程链只记日志并 return **false**（`bool` 化之后本地链
  也走同一条，不再 `GGML_ABORT`：调用方降级成「清空重算 / 结束响应 / 请求失败」），此后
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
- **v9 的「一帧只带一种 payload」**：decode 回复只能声明一个 `n_embd` 宽度，nextn 行（8 KB）与
  tap 段（prefill 时 8 MB）不能同发。三条回包分支必须共用同一个优先级，而且 **tap 优先、nextn
  兜底**——原来只有 plain 分支是 nextn 优先，于是 DFlash 链的 prefill 帧把 tap 吞掉，接受率掉到
  0.111 而不是崩（见 [`SEGMENTED-LIMITS.md`](SEGMENTED-LIMITS.md) §4 的第四个形状）。

验证（Qwen3.5-2B，`0:20` + `20:24`，独立 `build-split`）：

> **这一组测的是 2B 上的「代码没退化」，不是「35B 新模型能用」**——`verify.sh` 组 1 的断言与
> `verify_mtp.sh` 的 `tail 20:24 == 末层` 都是按 24 层模型写的，对 40 层的 35B 结构上不成立。
> 35B（官方 `UD-Q4_K_M`）的验收是另一件事：形状复位验 + 本地双段链 vs 整模逐字节一致，见
> [`SPLIT-MEASUREMENTS.md`](SPLIT-MEASUREMENTS.md) §11。

- N=1 直通：默认 / `--split-layer-range 0:24` / `0:20` 三种配置输出与改动前逐字节相同
- N=2 本地双段：head 输出与整模层 20 输入逐位相同；tail 单独跑与 baseline 相同；链路
  logits 逐位相同；`seq_rm` 广播后两段的 `pos_max` 同步下降
- N=2 经 llama-server：两个端口的 `reasoning_content` 一致
- 跨进程：本地 head + 远程 `split-stage`，logits 逐位相同；服务器的 `--split-remote` 走同一条路径
- 投机：**判据分两段**（2026-11-08 更正）。严格判据是「无回退时链上投机与**整模投机**逐字节
  相同」；「与不投机逐字节相同」降级为 INFO——它测的是 batch invariance，而 llama.cpp 在
  CPU + 量化上不承诺它（分段链的分界见
  [`SPLIT-MEASUREMENTS.md`](SPLIT-MEASUREMENTS.md) §9.4）。远程链的三组（不投机 / ngram / MTP）
  输出逐字节相同（`verify_mtp_e2e.sh`），远程 tail 内部递归 3 步与整模 + MTP 逐个相同
  （`verify_mtp.sh`）
- 远程链的死亡与自愈：`build-split/verify_chain_death.py` 36 项——挂起 stage 后请求 500 而 head
  存活、`/health?deep=1` 给 503、只重启 stage 就自愈（`prompt_n` / `cache_n` / 输出三项同时成立）、
  接错 stage 被 `OP_HELLO` 拒绝；候选通道的等价性见
  [`SPLIT-MEASUREMENTS.md`](SPLIT-MEASUREMENTS.md) §8
- `seq_rm` 的删尾部：`verify_seq_rm_exact.py --expect-known-fail` 的 A/B/C 三形态（纯 llama API、
  无 split、无 server）——FULL 形态两红（`seq_rm=false`、`pos_max` 不变、重放 `-1`）、RS 形态三红
  （rebase 后上游 `ab3768cdc` 把上界改成 `cell.rs_max`，而 probe 是逐 token decode，`rs_max=0`）、
  超窗明确拒绝一绿；`verify_seq_rm_shared.py` 的 shared-prefix 注入不触发
  （`[INFO] injection never fired`）。见 [`SPLIT-MEASUREMENTS.md`](SPLIT-MEASUREMENTS.md) §10

第四代仍成立的边界（`embeddings_pre_norm` 只覆盖两个架构、上游 hybrid 的 partial `seq_rm` 在
`n_rs_seq = 0` 下永远失败而我们只做「失败可见 + 调用方降级」、远程链的投机能做但收益随负载从
+8.4% 到 −1.2% 且判据不是「与不投机一致」、**MTP 在链上的接受率被封在 0.688（nextn 通道的设计
选择，与 DFlash/DSpark 的 tap 通道无关——后者 v9 之后远程链零损失）**、markov 头擅长可预测/
结构化内容而不是长距离依赖、级联在 server 侧不做、远程链是单序列 append-only、checkpoint 每段
约 12.56 MiB 过网、浅 `/health` 不反映链路状态（要问 `?deep=1`）、单独层范围的拒绝只覆盖一条
加载路径）逐条记在 [`SEGMENTED-LIMITS.md`](SEGMENTED-LIMITS.md) §2，那里写的是代价而不是 TODO。

### device-buffer 在 split 链下的崩溃与修复（2026-11-09，52 文件那一版）

**现象**：`llama-server` + 分段链 + `-md <DFlash 草稿>`，**跨机和 loopback 都崩**，整模不崩：

```
src/models/dflash.cpp:456（修前 440）
GGML_ASSERT(g.layer_inp_src != nullptr && g.layer_inp_src->ne[0] == n_embd_inp) failed
```

这是 rebase 引入的回归：上游 `2cbca389f dflash: copy hidden features via device copies` 之后，
DFlash 的特征从 host 的 `inp.embd` 改成优先读 device 侧的 `layer_inp_src`。

**根因不是「buffer 是 null」，而是「buffer 窄了」**：

```
head (0:32) 只装了 8 路 extract layer 里的 6 路（层 2/7/12/17/23/28）
→ device buffer = 6 × 2048 = 12288 列
→ dflash 的 fc 期望 8 × 2048 = 16384 列
→ 断言挂在第二个合取项，不是 nullptr
```

实测插桩输出：

```
[dbg] dflash: embd-row batch, layer_inp_src=0000026FB4028060 ne0=12288 n_embd_inp=16384
```

**为什么这个比 null 难查**：`supports_embd_row()` 回答的是「buffer 存不存在、读得到吗」，
而**不是**「buffer 装的是不是我的 encoder 输入」。上游没有分段链，隐含了「列数 = target_layers
数量」这个前提，**从未被检查过**。所以两边当时都认为「支持」，图构建时才炸。

（反证：如果真是 null，`decode()` 在 `:3211` 就返回 -1 了，根本到不了图构建。）

**修法（单一事实源）**：

1. `src/llama-context.cpp:1528` — `supports_embd_row()` 加宽度检查
   `src->ne[0] == model.hparams.n_embd_inp_enc()`。链下 12288 ≠ 16384 → 假；整模仍真。**这是唯一
   判定入口**：`decode()` 准入、`graph_params()` 的 `layer_inp_src`、batch 构造三处问同一句。
2. `common/speculative.cpp:1439` — **每帧重问一次**
   （`use_dev_features && llama_supports_embd_row(ctx_dft)`），而构造期答案可能变陈旧（buffer 被
   重分配 / 分配失败），图的答案每次现算——现在两边每帧都现算同一个谓词，**不可能再漂移**。
3. `src/models/dflash.cpp:456` — **断言原样保留**，只在前面加自解释诊断（直写 stderr，因为
   common 的 logger 是**异步**的，紧贴 abort 的 `LLAMA_LOG_ERROR` 会被丢——实测过），并加一段
   说明「这个行号会随上面的块移动、它始终指这条断言」的注释。

**为什么不选「构造时判 `!is_split_chain`」**：

- 它把「链的形状」硬编码进 drafter，谓词本身仍然是错的——任何别的
  `llama_batch_ext_add_embd_row()` 使用者照样撞断言；
- 而且会**误杀一个合法情形**：链的 head 覆盖全部 extract layer（比如 head `0:39` + tail `39:40`）
  时，device 路径其实是正确可用的，宽度条件自动允许它。**这一条是代码推出来的，未实测**。

**宽度条件才是真不变量**：「这个 buffer 装的正好是我 fc 要的全部层，且顺序一致」。顺序由
`build_layer_inp_dev` 升序 + 构造期 ascending 检查保证；层数靠宽度。

### 新增组 11：44 → 52 项

`build-split/verify_dflash_dev_path.sh` 并进 `verify.sh`，8 项：

```
[PASS] 组 11: 整模：device-buffer 路径仍然启用（宽度谓词没有误杀整模）
[PASS] 组 11: 整模：请求 200，112 字符
[PASS] 组 11: 链：按预期回退到 host copies（device buffer 装不下跨段的那几路 tap）
[PASS] 组 11: 链：回退原因写明（head 只装了 8 路里的 6 路）
[PASS] 组 11: 链：输出与整模逐字节相同（112 字符）—— 回退后的特征是对的
[PASS] 组 11: 链：没有 GGML_ASSERT
[PASS] 组 11: 链：请求后进程仍存活
[PASS] 组 11: 链：这一轮真的投机了（日志里有 draft acceptance）
```

**为什么必须加**：`verify.sh` 组 9/10 跑的是 `tests/dflash-taps.cpp`，它**只 link `llama`、
不 link `llama-common`**：

- 从不调 `llama_set_embeddings_layer_inp_dev` → `layer_inp_dev` 始终 nullptr
- batch 是普通 `llama_batch_init`，从不 `llama_batch_ext_add_embd_row` → `ubatch.embd_row == nullptr`
- 不可能实例化 `common_speculative_impl_draft_dflash`

**所以组 9/10 验的是 tap 通道（host-copies 路径）——正好是这条 bug 没弄坏的那一半。**整模的
device 路径同样零覆盖。

**并组时抓到并修掉的问题**（正是「新起 server 引入竞态」那一类，但不是端口竞态）：`verify.sh`
会 `export QLH_SPLIT_MODEL`（组 1-8 的 2B 模型），而组 11 脚本最初用 `${QLH_SPLIT_MODEL:-...}`
当 target 默认值 → 拿 2B 配 DFlash（extract layer 是 40 层模型的 2..38）→
`GGML_ASSERT(lid <= n_layer())`。修法：组 11 只认 `QLH_DFDEV_MODEL` / `QLH_DFDEV_DRAFT`，
`verify.sh` 显式传自己的 `MODEL_DFLASH` / `DFLASH_DRAFT`，**两边不再依赖默认值恰好一致**。

两个加固：

- `wait_health` 带 pid，进程死了立刻返回（原来会干等满 150×2 s，一个崩溃拖 12 分钟才报）
- 脚本默认 target 改成和 `verify.sh` 的 `MODEL_DFLASH` 同一文件

---

## 与上游 193 个提交的重合（2026-11-09 rebase）

这一节记的是**基底换掉之后**的事：fork 从 `e0f142d` 整体 rebase 到上游 193 个提交之后的
`f477f795c`。旧基线 `ec25cedda` 留在 `master` 上，新基线在 `rebase-merge` 分支（`0c5c3bdd`，
父提交 `f477f795c`）。

### 规模

| | 数字 |
| --- | --- |
| 我们的改动（相对旧基线 `e0f142d`） | 68 文件 / +18086 −504 |
| 上游 `e0f142d..f477f795c` | **193 个提交 / 659 文件** |
| 两边都碰过的文件 | **35 个** |

把我们的改动 `git apply --3way` 打到 `f477f795c` 上：

| 结果 | 文件数 | 说明 |
| --- | --- | --- |
| 新文件，无冲突 | **22** | 分段链的核心文件全是新建（`llama-split-*`、`llama-memory-split.*`、`tools/split-stage`、`tests/split-*`） |
| 干净应用 | **26** | 上游没碰，或只改了别处 |
| 冲突块 | **12 个文件 / 29 个块** | 全部落在上游重构过的 `llama-context.cpp`、`llama-graph.cpp`、`server-context.cpp` 这一层 |

rebase 之后相对上游只剩 **52 文件 / +17831 −230**：68 − **17**，再加这一轮 `src/models/dflash.cpp`
因 device-buffer 诊断行重新进入变更集。这 17 个文件里 **16 个**现在与上游**逐字节
相同**，也就是 `git diff f477f795c 0c5c3bdd` 里根本不再出现它们。少掉的是两类：

| 类别 | 文件 | 为什么消失 |
| --- | --- | --- |
| 上一轮「两次对齐」手工适配过的 13 个 | `llama-kv-cache.{cpp,h}`、`llama-kv-cache-dsv4.{cpp,h}`、`deepseek32/4/41`、`dots3note`、`glm-dsa`、`minimax-m3`、`mistral-large4`、`qwen4exp`、`dflash.cpp` | 那些改动本来就是把**上游的新 API 抄回来**；现在上游自己带着这些提交，我们那份适配就没有存在理由了（`dflash.cpp` 这一轮因诊断行又回来，见前面「device-buffer」小节） |
| 只承载 `shifts` 机制的 4 个 | `llama-graph.h`、`llama-memory-recurrent.{cpp,h}`、`delta-net-base.cpp` | 删掉 `build_rs_shift_snapshots` 之后，这四处剩下的只有上游的代码（见下面「删掉 shift」） |

这 17 个里被逐个核对过、且现在仍逐字节相同的 8 个（`llama-kv-cache.cpp` / `llama-kv-cache.h` /
`llama-kv-cache-dsv4.h`、`deepseek4.cpp`、`delta-net-base.cpp`、`llama-graph.h`、
`llama-memory-recurrent.cpp` / `llama-memory-recurrent.h`）是这一轮的判据样本：它们与匹配的
上游文件**逐字节相同**，不是「差几行」。rebase 那一版的 17 个全部用 `git rev-parse <commit>:<path>`
的 blob 哈希逐个比过（不是抽样），结果 17/17 相同；这一轮 `dflash.cpp` 多出的只是诊断输出。

### `llama-graph.cpp` 只剩一段是我们自己的

`src/llama-graph.cpp` 在被上游改掉 `llm_attn_mask`（`ca8daad19` 去掉了 `cnv`）之后，我们这一侧
剩下的唯一改动是 `build_inp_embd` 的 **relay 段**（`tok_embd == nullptr` 时不建 `GET_ROWS`）。
那是分段链真正需要的：relay / tail 段拿不到 token embedding，图里就不能有那条取值。
其余一百多行随上游一起走掉了——不是删我们的东西，是上游把同一段代码重写了。

### 采用的上游提交（我们据此删掉了自己的实现）

| 上游 | 内容 | 在我们这里的表现（可核对） |
| --- | --- | --- |
| `aa588a346` | drop Windows argv reconstruction | 我们的 diff 里没有 argv 重建代码，不需要跟改 |
| `fb5997274` | per-layer rotation（`self_k_rot` → 按层数组，构造期建张量、直接上传） | 上一轮抄回来的等价适配整块消失：`llama-kv-cache*` / `deepseek*` 等 12 个文件与上游逐字节相同 |
| `ca8daad19` | `llm_attn_mask` 去掉 `cnv` | 同上；`dflash.cpp` 当时也逐字节相同，这一轮因 device-buffer 诊断行重新进入变更集 |
| `1b6935b22` | `flash_attn_ext_rows` | 我们的 diff 里已无这个符号——图侧完全走上游 |
| `6023692b1` | per-model `build_rope` / kv shift 修复 | 同上（diff 里无 `build_rope`） |
| `b74298c10` + `040ce199a` | conv state store / MTP conv 快照 | 我们不再自己存 conv 状态；`delta-net-base.cpp` 只剩上游代码 |
| `ab3768cdc` | 拒绝超出「最后一个 ubatch」的快照回退 → `cell.rs_max` | **换掉了我们的 `shifts`**（下一节）；我们的 `seq_rm` 判定改成按 `rs_max` 分层（见「整帧回退」那节） |
| `e0f96a1a1` | `model` 变成 `llm_graph_context` 的成员 | 我们的 graph 接线（`build_inp_embd` 的 relay 段、qwen35/qwen35moe 的 tap）按成员访问 |
| `2cbca389f` | dflash 设备侧拷贝 | DFlash tap 的取数跟上游走设备侧拷贝，我们只加「跨段拼接」 |
| `e95426f16` | `moe_cache_cpu_share` | 我们的 diff 里无这个符号——专家缓存的内存账完全走上游 |

### 删掉 `build_rs_shift_snapshots`

我们原来让「recurrent 快照按 ubatch 往前挪」（shift）：新 ubatch 到来时先把上一批的快照整体
上移 `n_tokens`，于是**跨 ubatch 的回退**也能读到对应位置的状态。上游 `ab3768cdc` 的做法相反：
**拒绝**超出本 ubatch 快照的回退（`cell.rs_max = n_seq_tokens − 1`），并且 `split_equal()` 的
`n_keep_tail` 保证「整帧回退」永远落在同一个 ubatch 内。

同一个问题的两种解法。留着双套会让后来人以为 shift 在解决上游没解决的问题，而且两套机制的
槽位语义**互相矛盾**（shift 要求旧槽位保持旧内容，上游的 slot 语义是按本 ubatch 的行号写）。
所以整块删掉：`shifts` 字段、`can_reuse` 比较、`get_rs_shift()`、全部调用点。删完之后
`llama-graph.h` / `llama-memory-recurrent.{cpp,h}` 变成与上游逐字节相同。

### Rebase 暴露的真回归：整帧回退在上游 `rs_max` 下永远不合法

删 shift 换用上游 `rs_max` 之后，一个**在旧上游不会出现**的 bug 露出来了：

```
上游 ab3768c：cell.rs_max = n_seq_tokens − 1
一个验证帧写 n_draft + 1 行 → rs_max = n_draft

部分接受   删 n_draft 行       → 合法 ✓
整帧放弃   删 n_draft + 1 行   → 永远必被拒 ✗
```

而 `frame_needs_ckpt = n_draft + 1 > n_rs_seq`（`n_rs_seq = n_max + 1`）在 `n_draft <= n_max`
内**恒假**，于是整帧回退永远落到 `seq_rm` → 被拒 → 请求 500。旧上游只有 `rollback <= n_rs_seq`
一条判据，所以从来不会走到这里。

这个 bug 只有 `verify_grammar_spec.sh` / `verify_rbudget_spec.sh` 能发现——`verify.sh` 的组
8/9/10 都不走 grammar / reasoning budget 这两条路径。修法是三个名字各司其职：

| 名字 | 回答的问题 | 判据 |
| --- | --- | --- |
| `frame_needs_target_ckpt(slot, n_draft)` | **要不要捕获**（宽） | `FULL` → true；`RS` 且帧可能被整帧放弃（`has_grammar \|\| has_reasoning_budget`）→ true；`RS` 且 `n_draft + 1 > n_rs_seq` → true；其余 → false |
| `partial_uses_ckpt` | **部分接受能不能不用 ckpt**（窄） | `FULL \|\| n_draft > n_rs_seq` |
| `frame_has_ckpt = !slot.spec_ckpt.empty()` | **这一帧到底有没有捕获到**（运行时事实） | 整帧回退用它 |

关键在第三个：整帧回退**不能重新调用判定函数**。两次求值一旦漂移（比如判定依赖的 `n_rs_seq`
在这中间变了），就会走错路径，而走错 = `seq_rm` 被拒 = 该请求 500（实测出现过）。配套：新帧
开始时 `spec_ckpt.clear()`，于是 `empty()` 精确等于「本帧没捕获过」。

**权衡**：没有 grammar / reasoning budget 的请求保持**零捕获**的便宜路径——为了一个不存在的
整帧回退去付每次捕获的代价没有意义。

### 两处「上游基线漂移」——不是我们的问题，但必须记

**第一处：`verify_rbudget_spec.sh` 判据 3**。它要求「不投机时链 == 整模逐字节」。三方对照：

| | 整模输出 |
| --- | --- |
| 纯上游 `f477f795c` | `...with Paris as the city name...`（296 B） |
| rebase 后本仓 | **与纯上游逐字节相同** |
| 旧上游 `ec25cedda` 时期 | `...with the city parameter set to "Paris"...`（295 B） |
| rebase 后本仓链 | **与旧上游链逐字节相同** |

差异全部来自**上游改了整模的数值路径**，链侧没变。判据因此改成三档：逐字节相同 → PASS；
不同但分叉点在后半段 → PASS + 记录分叉点；分叉在前半段 → FAIL。长度不参与判定（一旦分叉，
两侧长度自然不同）。实测：不投机 8/16/32 token 链与整模**全部逐字节相同**，33~40 token 之间
才翻——是长序列的尾部敏感，不是链的问题。

**第二处：`verify_seq_rm_exact.py` 的 B 组（RS 形态）从绿变红**。原因是 `ab3768c` 之后
`rs_max = n_seq_tokens − 1`，而这个 probe 是**逐 token decode**（`llama_batch_get_one(&t, 1)`），
所以 `rs_max = 0`，任何回退都必须被拒。这是上游「只保证最后一个 ubatch 的快照」的设计后果。

没有改 probe：让它用多 token ubatch 确实能让 B 组变绿，但「重放 logits 逐位相同」这条判据会因
1 行 vs k+1 行的 kernel 差异（0.27 量级）失效——等于把可靠判据换成不可靠的。所以更新
known-fail 集合（A1 A2 B1 B2 B3 已知红、C1 绿）并把原因写进文件头。

### 验收（主仓，`--clean-first` 全量构建之后）

| 套件 | 结果 |
| --- | --- |
| `verify.sh` | **52 / 0 / 0** |
| `verify_chain_death.py` | **36 / 0** |
| `verify_grammar_spec.sh` | **23 / 0** |
| `verify_rbudget_spec.sh` | **36 / 0** |
| `verify_seq_id.py` | **9 / 0** |
| `verify_mtp.sh` | **6 / 0** |
| `verify_mtp_e2e.sh` | **6 / 0** |
| `verify_seq_rm_exact.py --expect-known-fail` | **1 pass / 5 known-fail / 0 unexpected** |

### 一个该记的疏漏：静默指向错的东西 + 绿

做 worktree 副本验收时，`verify.sh` 的 `ROOT` 改了，但 `verify_multislot.py` 里的
`ROOT` / `BIN` 是**硬编码**的没改，于是组 5 实际跑的是**主仓的二进制**——而且表现是「通过」。

**静默指向错的东西 + 绿，是最坏的一类**：它不会失败，只会让一次验收变成一次误会。教训是
机械的：做 worktree 副本验收时，`grep -n 'ROOT\|BIN\|build-split\|H:'` **每一个**副本脚本，
别只改主入口。本轮重导的纯分段验收就是按这条做完了全量替换再检查残渣为空的。

---

## 与上游版本的两次对齐（2026-11-09，上一轮；本轮 rebase 已整体吸收）

逐行对比 `src/models/dflash.cpp` 与上游 `f477f795c` 时发现：**我们和上游只差 118 行，而且全部是
「上游改了 API，我们没跟上」**。两处，都是上游提交：

| 上游 | 内容 | 在我们这里的形态 |
| --- | --- | --- |
| `fb5997274`（#31073） | **per-layer rotation**：rotary 常量从「每次 `set_input` 从 host memcpy」改成「kv-cache 构造时按层建张量、放进独立的 `attn_rot_bufs`、直接上传」，于是 `self_k_rot` → `self_k_rot[il]`。按层的原因是每层 cache 的 buffer / 放置可以不同（混合 SWA、mixed placement） | `llm_graph_input_*` 的 `self_k_rot` / `self_v_rot` 全部改成按层数组，`build_input_k_rot(ctx0)` 换成 `get_k_rot()`，graph 层那批 `set_input_k_rot` / `set_input_v_rot` 的 host 拷贝整块删掉；kv-cache 侧新增 `attn_rot_bufs` 的建立与上传 |
| `ca8daad19`（#31558） | **`block_uniform`**：给 `llm_attn_mask` 加 `block_uniform`，`iq = block_uniform ? seq_first[seq_id] : i`——同 seq 的后续 token 复制首行 | `llm_graph_input_attn_k_iswa` 加 `self_kq_mask_swa_block_uniform`（默认 false）；`set_input_kq_mask` 加同名参数；`args_set_input_kq_mask` 加 `seq_first` + `iq`；`dflash.cpp` 删掉自造的 `build_dspark_block_mask()`（15 行 reshape/repeat），换成一行 `self_kq_mask_swa_block_uniform = true` |

**移植方式是等价适配，不是合并上游**：不引入 `llm_attn_mask` 结构、不改 graph 构造签名
（上游那两个提交顺带重构了 `graph(const llm_graph_params&)`，那属于完整 rebase——上一轮没有做，本轮做了）。

**数值等价由 tap 逐位证明**：组 9/10 的 8 路 / 5 路 tap 与整模逐位相同——mask 从 graph 层搬到
`set_input` 填值，没有改变一个比特。

**改动面（`git diff c5bd787b ec25cedda`）**：13 个被上游 API 变更牵连的文件（`llama-kv-cache-dsv4.*`、
`llama-kv-cache.*`、`deepseek32/4/41`、`dots3note`、`glm-dsa`、`minimax-m3`、`mistral-large4`、
`qwen4exp`、`dflash.cpp`）合计净 **+3 行**；其中 `src/llama-graph.cpp` +32 −103、`src/models/dflash.cpp`
+16 −29。`graph.cpp` 删掉的那一百多行**主要是上游 `fb5997274` 一并删掉的 host 拷贝调用**（跟着上游
走，不是删我们自己的东西）；我们自己的胶水只有 `build_dspark_block_mask()` 那 15 行，整块消失，
换成一个布尔标志。

**本轮 rebase 之后这一节已经退场**：上面这 13 个文件现在与上游 `f477f795c` **逐字节相同**（上游自己带着 `fb5997274` / `ca8daad19`），`docs/llamacpp-split.patch` 里不再出现它们。留下来是为了记录「先等价适配、后整体 rebase」这条路径当时怎么判断的。

**方法论**：「不重复造轮子」不等于「所有重复都要自己消掉」。判断标准是**改动能否独立成立**：
这两处都不依赖上游的新 API 签名，所以能移；如果依赖（比如 `graph(const llm_graph_params&)`），
就不要硬移，记下来放进完整 rebase——「为了消掉重复而引入一个半对的实现」违反另一条原则。

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
