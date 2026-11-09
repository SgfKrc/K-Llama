# 四代补丁，以及与原生 RPC 的分界

> 状态：**现行**
> 更新日期：2026-11-09（段级 offload 与显存预算之后重导：54 文件 / +18733 −240，树 `e76601df`）
> 适用范围：本项目对 `H:\llamacpp-c` 四代补丁的构成、代际差异，以及与 llama.cpp 原生 RPC 的分界；
> 读者为内核与部署维护者。逐行机制表、device-buffer 修复细节、与上游 193 个提交的重合明细见
> [`PATCHES-DETAIL.md`](PATCHES-DETAIL.md)。相关：[`SLIM-ARC-接入计划.md`](SLIM-ARC-接入计划.md)、
> [`CUT-TOOL.md`](CUT-TOOL.md)、[`SEGMENTED-LIMITS.md`](SEGMENTED-LIMITS.md)、
> [`SEGMENT-OFFLOAD.md`](SEGMENT-OFFLOAD.md)。

本项目是 K-Llama 的第二代。两代的差别不在协议层，而在模型怎么装进显存和内存：第一代靠「裁好的层段工件」，第二代靠「整模
+ 层范围」。llama.cpp 侧的补丁因此完全不同。

**代码现状**：相对基提交 `f477f795c` 共 **54 文件 / +18733 −240**（`git diff --stat`，fork HEAD `48dfb3ea8`，树
`e76601df`），覆盖第一代到第四代的全部改动，拆成两个可独立应用的补丁：

| 补丁 | 包含 | 统计 |
| --- | --- | --- |
| [`docs/llamacpp-split.patch`](llamacpp-split.patch)（主补丁） | 第一、二、四代：层范围加载、分段链、远程段、聚合 memory、checkpoint、server/MTP/DFlash/DSpark 接线、协议 v7/v8/v9、grammar / rbudget 兼容、`seq_rm` 失败显式化、device-buffer 宽度判定、段级 offload、显存预算。**不含 SLIM-ARC** | **44 文件** / +15189 −239 / 17149 行 / 891.0 KiB |
| [`docs/llamacpp-slimarc.patch`](llamacpp-slimarc.patch)（可选增量） | 第三代 SLIM-ARC prefetch：10 个源文件 + 3 个文件里的 6 处钩子 | 13 文件 / +3544 −1 / 3662 行 / 155.2 KiB |

两个补丁逐项相加等于 54 / +18733 −240：44 + 13 里有 3 个文件重叠（`src/CMakeLists.txt`、
`src/llama-context.cpp`、`src/llama-model-loader.cpp` 各出一份 diff），并集仍是 54（57 − 3）。算术校验：15189 + 3544 =
18733、239 + 1 = 240。**相对上一版多了 2 个文件、+902 −10**： `src/qlh-vram-budget.h` 与 `src/qlh-split-defaults.h`
是新增的 header-only 文件，另外 5 个文件
（`common/arg.cpp`、`common/common.h`、`src/llama-model.cpp`、`tools/server/server-context.cpp`、
`tools/split-stage/main.cpp`）被段级 offload、MoE 缓存默认值与显存预算各动了几十行。

**基提交是 `f477f795c`**：fork 整体 rebase 到上游 193 个提交之后（见「与上游的重合」）。旧基线 `ec25cedda`（树
`e14b92c9`，68 文件 / +18086 −504）留在 `master` 上，已不是补丁基底。

两者是叠加关系，主补丁可单独用：

```bash
git worktree add /tmp/v1 f477f795c && cd /tmp/v1
git apply docs/llamacpp-split.patch        # 纯分段，无 SLIM-ARC
git add -A && git write-tree               # 必须 = 2949267e（纯分段树）
git apply docs/llamacpp-slimarc.patch      # 再叠增量，回到完整功能
git add -A && git write-tree               # 必须 = e76601df（= fork HEAD 的树）
```

补丁文件必须按字节保存（仓根 `.gitattributes` 的 `*.patch -text`）。`core.autocrlf=true` 下把补丁转成 CRLF 之后照样
apply 成功——只有 trailing-whitespace 警告——但文件末尾 `\ No newline at end of file` 会多出一个裸 CR，产出的树与文档里的
tree hash 不再相等。静默错，而且属于「验收命令跑了、也过了、结果不对」那一类。这是 rebase 那轮实测撞到的（CRLF
版主补丁得到 `d679f09b`，正确是 `1dcbce39`）；这一轮的判据是「主补丁 → `2949267e`，叠加 → `e76601df`」，导出后已用
`grep -c $'\r'` 确认为 0 个 CR。

- 只应用主补丁得到的树是 `2949267e`（44 文件）：能独立 `--clean-first` 全量编译（385 个编译单元，
  `cmd //c '.qlh_tmp\vrambuild.bat build-split all --clean-first'` 退出 0，`dflash-taps.exe` /
  `split-smoke.exe` / `split-rows.exe` 等目标齐全），并独立通过 `verify.sh` **十一组 52 项**与
  `verify_chain_death.py` **36 项**。没有 SLIM-ARC 不影响分段链的任何一条验收。2026-11-09 在
  `H:\llamacpp-c` 上把工作区切到纯分段树实测，跑完切回完整树、`git write-tree` 仍等于
  `e76601df`，日志见 `.qlh_tmp/pure-split-acceptance/`。
- 上一版（`1dcbce39`，44 项）第一次跑出 `PASS=34 FAIL=10`，两处都是环境：组 5 的 `-np 4` 启动
  瞬间死掉（日志 0 字节，20 个请求一个都没发出，单跑两次都过）；组 10 的 dflash 十项里九条红，
  `taps_dflash_chain.log` 只有 `dflash-taps.exe: Device or resource busy`（整模那趟刚退出、文件锁
  未释放）。run2 同一份二进制、同一份模型得到 44/0。判据永远是「日志里有没有数值不一致」，不是
  「这一趟绿不绿」。
- 叠加两个补丁后 `git write-tree` = `e76601df`（fork HEAD 的树），这是「拆分没有改动任何内容」的
  唯一判据，比「能 apply」更强。
- SLIM-ARC 与分段链零耦合：那 10 个文件只 include 彼此和上游 `ggml-backend.h`，不碰本 fork 新增
  的任何头；6 处钩子摘掉后，`llama-context.cpp` / `llama-model-loader.cpp` 的对应片段与
  `f477f795c` 逐字节相同（重导时用字符串精确匹配摘除，每处断言「恰好出现一次」）。

**历史全文**：`docs/llamacpp-qlh-slimarc.patch`（27 文件 / 4014 行），基提交更早，只到第三代；保留它为了对照第一 /
二代的实现，不能正向应用。

fork 侧历史是两个提交：上游 `f477f795c` 加一个 `[QLH-SPLIT]`（`48dfb3ea8`，树 `e76601df`；比上一版 `0c5c3bdd` / 树
`982ccecd` 多了段级 offload、MoE 缓存默认值与显存预算）。旧的一对（`e0f142d` + `ec25cedda`，树 `e14b92c9`）留在
`master` 作为历史，不再用于生成补丁。开发期的 wip 提交已 squash；更早记录的 squash 前后 tree 值是 `d628e2a8`，与那一代
HEAD 的 `e14b92c9` 不同（差异来自 squash 之后又合入的修复与两次上游移植）。这项核对只能当历史记录，不能重放。

## 第一代（K-Llama / QLH）：层输出通道

改动集中在一处：让上下文多出一个「第 N 层输出」的槽位，供段间接力取数。`src/llama-context.cpp` 的 `layer_inp`
多分配一个槽位（`n_layer + 1`），索引 `lid == n_layer` 即「第 n_layer 层的输入」= 末层输出（`output_norm`
之前）；`set_embeddings_layer_inp` 的断言同步放宽。在 `output_norm` **之前** 登记该槽位，覆盖 9 个架构：`qwen2` /
`qwen35` / `gemma4` / `llama` / `openai-moe` / `qwen3` / `qwen3moe` / `qwen35moe` / `qwen3next`。对外是 4
个导出符号，shim 用到 `llama_set_embeddings_layer_inp()` / `llama_get_embeddings_layer_inp()`。

不能复用 `t_h_nextn`：各架构挂点不同——`qwen2` 的 `t_h_nextn` 在 `output_norm` 之前，`qwen35` 在 **之后**（消费方是 MTP
head）；拿 `nextn` 当层段上游，在 Qwen3.5 上会多过一次 RMSNorm，接力首步就分叉。代价是这套改动要跟着每个架构走，9
个架构 9 处挂点。

## 第二代（K-Llama v2）：层范围加载

打开整模、只把属于自己那段读进来。8 个文件，+227/−13：

| 文件 | 改动 |
| --- | --- |
| `include/llama.h` | 新增 `llama_context_params::embeddings_pre_norm` |
| `src/llama-cparams.h` | 内部结构同步该字段 |
| `src/llama-context.cpp` | 默认 `false`，从参数拷贝，打日志 |
| `src/llama-model-loader.h` | `layer_lo` / `layer_hi`、`n_layer_all`、`needs_tok_embd()` / `needs_output()` |
| `src/llama-model-loader.cpp` | `LLAMA_LAYER_RANGE` 解析、`get_weight` 重映射、全局张量的 lazy 标记 |
| `src/llama-model.cpp` | `done_getting_tensors` 传范围、`init_mappings` 按范围、host-only 预读、`size_data` 重算 |
| `src/models/qwen35.cpp` | 在 `build_norm` 之前 tap 出 `t_embd` |
| `src/models/qwen35moe.cpp` | 同上；另有 relay 末层不裁剪 |

四个关键机制：**`LLAMA_LAYER_RANGE=lo:hi`** 覆盖 `block_count` 与 `nextn`，让架构以为自己只有 `hi-lo` 层，`get_weight`
把本地 `blk.i` 映射回 `blk.lo+i`；**`embeddings_pre_norm`** 把 tap 点前移到 `output_norm` 之前，替代第一代的
`layer_inp` 槽位，只改 qwen35 / qwen35moe；**lazy 全局张量** （head 不需要 `output`、tail 不需要 `token_embd`、middle
都不需要，middle 段因此省 1030.62 MiB）；**relay 末层不裁剪**——`qwen35moe.cpp` 的 `crop_before_nextn` 在 `n_outputs ==
0` 时会把末层裁成 `[n_embd, 0]`，而 `t_layer_inp[n_layer]` 登记的正是这条张量，于是 `extract_layer_inputs` 的
`row_floats == n_embd` 断言失败；修法是 `qlh_keep_full_rows = embeddings_pre_norm ||
embeddings_layer_inp[n_layer]`，为真时跳过裁剪并在 LM head 之前补回 `ggml_get_rows`。Windows 上不出现这个坑：编译宏
`QLH_ABI_UPSTREAM_HEADER` 把 `batch.logits` 钉成 1，绕开了裁剪；两条路实测 hidden
逐位相同，但只有补丁这条路不依赖编译期定义。

第二代反而更小：第一代要 9 个架构各自登记挂点，第二代把「取哪一层的输出」变成「只加载哪几层」
——后者是加载器的事，与架构无关，架构侧只剩 2 处 tap。

## 第三代：SLIM-ARC prefetch（2026-10-31，可选增量）

上游 SLIM-ARC 是给 Linux 写的，本机是 Windows，所以是带回移植的接入，只接两级里的第一级
（`prefetch`）。第二级（on-demand 加载器）未接，理由见 [`SLIM-ARC-接入计划.md`](SLIM-ARC-接入计划.md)。新增 5 对 / 10
个文件： `slim-arc-prefetch.{h,cpp}`（2085 + 495 行）加依赖的 `page-range` / `expert-reclaim` / `expert-residency` /
`expert-transition`（各一对 `.h/.cpp`），合计 3479 行。这 10 个文件自包含，只 include 彼此和上游
`ggml-backend.h`，能整块摘下而不牵连分段链。

这一代不在主补丁里，只存在于 `docs/llamacpp-slimarc.patch`（13 文件 / +3544 −1），叠在 `docs/llamacpp-split.patch`
之上才回到完整树。6 处钩子（归并成 4 行）：

| 文件:行（完整树 `48dfb3ea8`） | 位置 | 作用 |
| --- | --- | --- |
| `src/CMakeLists.txt:50-60` | `LLAMA_SLIMARC_SOURCES` | 注册 5 个 `.cpp`，用 `SKIP_UNITY_BUILD_INCLUSION` 挡在 unity 批次外——这批文件带 `WIN32_LEAN_AND_MEAN` 和 POSIX 名字，混进 unity 会污染同批翻译单元 |
| `src/llama-model-loader.cpp:1587-1615` | `init_mappings()` 末尾 | 只有设了 `SLIM_ARC_PREFETCH` 才构造全局调度器，把每个权重的 `(mmap 基址 + 偏移, 大小, 层号)` 注册进去 |
| `src/llama-context.cpp:1778-1793` | `process_ubatch()`：`ggml_backend_sched_reset()` 之后 | 调度器在装且 `n_gpu_layers() == 0` 时把 `eval_cb` 装到 sched 上，用户 `cparams.cb_eval` 串在它里面；否则原样用用户回调 |
| `src/llama-context.cpp:1803-1811` | `process_ubatch()`：`build_graph()` 之后 | `prepare_graph(gf)` 扫新图、记下每层第一个节点，运行期由 `eval_cb` 在层边界发预取 |

表里 4 行覆盖 6 处：`set(LLAMA_SLIMARC_SOURCES ...)` 与 `add_library()` 里的 `${LLAMA_SLIMARC_SOURCES}`
各一处，`src/llama-context.cpp:17` 与 `src/llama-model-loader.cpp:8` 的 `#include` 各一处，再加两个 `.cpp`
里的代码块。摘掉后 `process_ubatch()` 恢复成 `f477f795c` 的单行
`ggml_backend_sched_set_eval_callback(...)`，`init_mappings()` 末尾恢复成空行加 `}`，两处与 `f477f795c`
逐字节相同。**换基底之后行号必须重新定位**：更早一轮这两块在 1613 / 1638，rebase 后到 1762 / 1787，修
`supports_embd_row()` 又往下推 16 行，现在是 1778 / 1803（本轮重导时逐个核对过，`strip_hooks.py`
一字未改就精确命中）。摘除脚本用整块字符串精确匹配 +「恰好出现一次」断言，对不上直接失败。

**层号怎么来**：上游从图节点名读 `blk.N`，本树取不到——GGML scheduler 在计算前把每个节点重命名成
`node_%d`，真正的张量名留在操作数上。本树改成读 `graph_get_cb()` 给每个 tensor 打的 `-<层号>`
后缀（`ggml_format_name`，上游行为），`prepare_graph` 再按「层号只会前进」把名字尾数恰好是数字的
节点滤掉。按上游写法接时 `calls` 恒为 0；改成读后缀后 `calls=5`（3 个 token 各一次通知加 prefill）。win32
移植面：`madvise(WILLNEED)` → `PrefetchVirtualMemory`、`DONTNEED` → `DiscardVirtualMemory`、`NORMAL/RANDOM/SEQUENTIAL` →
空操作；`mincore` 不支持、返回「未知」（消费方是专家热锁 LRU，未知即跳过）；页大小用
`GetSystemInfo().dwPageSize`；`mlock` / `munlock` → `VirtualLock` / `VirtualUnlock`。另有两处 MSVC
兼容：`NOMINMAX`（否则 `windows.h` 的 `max` 宏会打掉 `std::max` 与 `numeric_limits<T>::max()`），以及用 `_umul128` 拼
64×64→128 乘积替代 `unsigned __int128`。

**当前生效范围**：默认关闭，`SLIM_ARC_PREFETCH=1` 才装调度器；整模在页缓存里时 `issued` 为 0，
它只在「段装不下、真的在读盘」的降级形态里有事可做。

## 第四代（现行）：分段链与跨进程

前三代解决「每台机器只装自己那几层」，第四代解决「这么多段怎么跑成一次推理」。一个 `llama_context`
成为链的驱动者：本地段输出 pre-norm hidden，后续段（本地或远程）消费 hidden 继续算，最后一段产 logits。`llama-server`
的 slot、prompt cache、`seq_rm` 通过聚合 memory 看到整条链，不需要知道段在哪台机器上。

远程链的投机是这一代长出来的：协议 v7 加逐行 argmax 的验证通道，v8 把 MTP draft 下沉到远程 tail （`OP_MTP_DRAFT`），v9
又把 DFlash / DSpark 的 `layer_inp` tap 挂到 decode 回复上，随后补了 grammar 与 reasoning budget
两条兼容路径。它们都是增量改协议 / 加判据，没有动链的驱动结构，所以
各占一行而不是另开一代。**收益结论按终态写**：能做，但 ROG 五类真实 prompt 上 MTP k=2 中位只有 **+6%**（区间 +8.4% ~
−1.2%），弱机器与回环上是负的；判据是「无回退时与**整模投机**逐字节相同」。数字与归因见
[`SPLIT-MEASUREMENTS.md`](SPLIT-MEASUREMENTS.md) §9、 [`SEGMENTED-LIMITS.md`](SEGMENTED-LIMITS.md) §2.2。

机制行有 26 条，完整表、六处语义（`seq_rm` 失败显式化、`--split-rs-seq` 不再必需、失败不区分
对端拒绝与没连上、链死自愈、checkpoint 覆盖全链、v8/v9 的槽位顺序与通道优先级）与验收表在
[`PATCHES-DETAIL.md`](PATCHES-DETAIL.md) §1、§3。这里只留骨架：层范围进参数（`llama_model_params.split_layer_lo/hi`）→
`decode_split` 走标准 embedding 路径传 hidden → `llama-memory-split` 聚合本地 / 远程 memory →
`llama-split-{proto,net,remote}` 与 `tools/split-stage` 承担定长协议 v9（v3 定长头 + v4 身份握手 + v5 候选通道 + v7
验证行 + v8 远程 MTP draft + v9 tap）→ server 入口 `--split-tail-range` / `--split-remote` / `--split-rs-seq`。段级
offload、MoE 缓存与显存预算三行同样在第四代里，语义见 [`SEGMENT-OFFLOAD.md`](SEGMENT-OFFLOAD.md)。

第四代仍成立的边界逐条记在 [`SEGMENTED-LIMITS.md`](SEGMENTED-LIMITS.md) §2，那里写的是代价而不是
TODO：`embeddings_pre_norm` 只覆盖两个架构、上游 hybrid 的 partial `seq_rm` 在 `n_rs_seq = 0` 下
永远失败、远程链投机收益随负载翻号、MTP 在链上的接受率封在 0.688、markov 头擅长可抄写内容、级联在 server
侧不做、远程链单序列 append-only、checkpoint 每段约 12.56 MiB 过网、浅 `/health`
不反映链路状态、单独层范围的拒绝只覆盖一条加载路径。

## 与上游 193 个提交的重合（2026-11-09 rebase）

fork 从 `e0f142d` 整体 rebase 到上游 193 个提交之后的 `f477f795c`（193 个提交 / 659 文件）。把我们的改动 `git apply
--3way` 打到新基底：新文件无冲突 22 个、干净应用 26 个、冲突 12 个文件 / 29 个块（全部落在上游重构过的
`llama-context.cpp`、`llama-graph.cpp`、`server-context.cpp`）。 rebase 之后相对上游只剩 54 文件 / +18733 −240：68 −
17，再加 `src/models/dflash.cpp` 因 device-buffer 诊断行重新进入变更集。17 个消失的文件里 16
个与上游逐字节相同；被逐个核对过的 8 个用 `git rev-parse <commit>:<path>` 的 blob 哈希比过（不是抽样），17/17
相同。采用的上游提交（`aa588a346`、`fb5997274`、`ca8daad19`、`1b6935b22`、`6023692b1`、`b74298c10`、`040ce199a`、
`ab3768cdc`、`e0f96a1a1`、`2cbca389f`、`e95426f16`）逐条对应关系、两处「上游基线漂移」、以及 rebase
暴露的整帧回退回归，见 [`PATCHES-DETAIL.md`](PATCHES-DETAIL.md) §2。验收结果（全量构建之后）：`verify.sh`
52/0/0、`verify_chain_death.py` 36/0、`verify_grammar_spec.sh` 23/0、 `verify_rbudget_spec.sh` 36/0、`verify_seq_id.py`
9/0、`verify_mtp.sh` 6/0、`verify_mtp_e2e.sh` 6/0、`verify_seq_rm_exact.py --expect-known-fail` 1 pass / 5 known-fail /
0 unexpected。

## 与 llama.cpp 原生 RPC 的分界

原生 RPC（`-DGGML_RPC=ON`）也能跨机，但两者解决的不是同一个问题。RPC 的协议命令里 `SET_TENSOR` / `SET_TENSOR_2D`
上传张量数据、`GRAPH_COMPUTE` 让远端执行子图、`GET_TENSOR` 取回结果：**RPC 让远端当「远程
GPU」，主机仍然要有完整的模型**——权重从主机通过网络发到远端，主机维护全局计算图，远端只负责执行派过去的那部分
op。**本项目让每个节点当「一段模型」，没有中心**：整模文件在每个节点本地，各自 mmap 后只 touch 自己那几层，段间只传
`n_embd` 个浮点。

| 维度 | 原生 RPC | 本项目 |
| --- | --- | --- |
| 权重从哪来 | 主机通过 `SET_TENSOR` 发给远端 | 各节点本地读整模的对应层 |
| 首次跨机流量 | 权重总量（GB 级） | 0，只要本地有整模 |
| 主机必须装下整模 | 是 | 否（只装自己那几层） |
| 每步流量 | 激活 + 图指令 | hidden 一个向量（FP32，2048 × 4 B = 8 KB） |
| 计算图 | 主机维护全局图 | 每段是独立进程，无全局图 |
| 异构 | 需要相同的 ggml backend | 自定义 TCP 协议，可跨不同 CPU/GPU |
| 层范围加载 | 不支持，要自己切模型 | `LLAMA_LAYER_RANGE` 原生支持 |

一句话：RPC 是把设备虚拟化，模型仍是一份；本项目是把模型分段，每段各自持有。前者适合「一台主机 +
若干算力节点」，后者适合「几台都装不下整模的机器」。代价：本项目的段是黑盒，段内不能跨机器切分
算子，每一段必须能单独装下；RPC 可以把任意 op 派到远端，粒度更细。
