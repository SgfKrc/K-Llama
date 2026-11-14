# 四代补丁，以及与原生 RPC 的分界

> 状态：**现行**
> 更新日期：2026-11-13
> 适用范围：本项目对 `H:\llamacpp-c` 四代补丁的构成、代际差异，以及与 llama.cpp 原生 RPC 的分界；
> 读者为内核与部署维护者。逐行机制表、device-buffer 修复细节、与上游两次 rebase 的重合明细见
> [`PATCHES-DETAIL.md`](PATCHES-DETAIL.md)。相关：[`SLIM-ARC-接入计划.md`](../design/SLIM-ARC-接入计划.md)、
> [`CUT-TOOL.md`](../legacy/CUT-TOOL.md)、[`SEGMENTED-LIMITS.md`](../limits/SEGMENTED-LIMITS.md)、
> [`SEGMENT-OFFLOAD.md`](../measure/SEGMENT-OFFLOAD.md)。

本项目是 K-Llama 的第二代。两代的差别不在协议层，而在模型怎么装进显存和内存：第一代靠「裁好的层段工件」，第二代靠「整模
+ 层范围」。llama.cpp 侧的补丁因此完全不同。

**代码现状（v2.2）**：主补丁 + SLIM-ARC 相对基提交 `5676dfbe0` 共 **55 文件 / +21788 −238**
（`5676dfbe0` → `slimarc/v2.2`，树 `61781c712`），覆盖第一代到第五代的全部改动，拆成三个可独立应用的补丁。
**含 KVMem 的那一档是 v2.1 口径**：`09338b85e` → `kvmem/integration` 共 **91 文件 / +21611 −376**
（树 `192b2a3d0`）——KVMem 补丁**未随 v2.2 重导**（`kvmem/integration` 尚未 rebase），仍绑 v2.1 基底，
与上两档不共基底，别混用：

| 补丁 | 包含 | 统计（基底） |
| --- | --- | --- |
| [`docs/llamacpp-split.patch`](../llamacpp-split.patch)（主补丁） | 第一、二、四代：层范围加载、分段链、远程段、聚合 memory、checkpoint、server/MTP/DFlash/DSpark 接线、协议 v7/v8/v9/v10、grammar / rbudget 兼容、`seq_rm` 失败显式化、device-buffer 宽度判定、段级 offload、显存预算。**不含 SLIM-ARC** | **45 文件** / +18123 −237 / 20238 行 / 1077.2 KiB（`5676dfbe0`） |
| [`docs/llamacpp-slimarc.patch`](../llamacpp-slimarc.patch)（可选增量） | 第三代 SLIM-ARC prefetch：10 个源文件 + 3 个文件里的 6 处钩子 | 13 文件 / +3665 −1 / 3790 行 / 164.2 KiB（`5676dfbe0`，内容未变） |
| [`docs/llamacpp-kvmem.patch`](../llamacpp-kvmem.patch)（可选增量） | 第五代 KVMem 虚拟化：memory factory hook、attn Q/K capture 钩子、KV cache 跳过空洞清理、CUDA stage-in、`--kvmem*` CLI、`llama-kvmem-bridge`。**adapter 主体在外部**（`LLAMA_KVMEM_ROOT` 指向 `SgfKrc/kvmem`），本补丁只含 fork 侧接线 | **52 文件** / +1266 −135 / 2941 行 / 130.6 KiB（**未随 v2.2 重导，仍绑 v2.1 基底 `09338b85e`**） |

前两个补丁逐项相加等于 55 / +21788 −238：45 + 13 里有 3 个文件重叠（`src/CMakeLists.txt`、
`src/llama-context.cpp`、`src/llama-model-loader.cpp` 各出一份 diff），并集仍是 55（58 − 3）。算术校验：
18123 + 3665 = 21788、237 + 1 = 238。KVMem 那档（91 文件）是 v2.1 时 55 + 52 − 16 个重叠的结果，数字保持不变。

**重导沿革**（每次在前一版统计上累加）：

相对前一版多了 2 个文件、+902 −10：`src/qlh-vram-budget.h` 与 `src/qlh-split-defaults.h` 是新增的 header-only
文件，另外 5 个文件（`common/arg.cpp`、`common/common.h`、`src/llama-model.cpp`、`tools/server/server-context.cpp`、
`tools/split-stage/main.cpp`）被段级 offload、MoE 缓存默认值与显存预算各动了几十行。之后又重导过五次：

① 2026-11-09 把 `split-stage` 的 bind 提前到模型加载之前（端口被占时 214 ms 退出，而不是先读完几 GB
权重才发现）：`tools/split-stage/main.cpp` +49 −9，主补丁 +15189 → +15229、总 +18733 → +18773。

② 同日把源码注释里 6 处 `build-split/verify_*` 改成 `verify/verify_*`（见 [`../verify/README.md`](../../verify/README.md)）；
只改注释，统计仍是 +15229 −239，树 `0c307d82a` / `181edd82`。

③ 2026-11-10 **链上 MTP 的草稿 KV 续写（keep-KV）**，同日从 opt-in 翻成**默认开**（`QLH_CHAIN_DRAFT_KEEP_KV=0`
回旧路径，逐字节等于旧行为）：`speculative.cpp` +171 −7、`speculative.h` +6、`server-context.cpp` +19 −8，主补丁
+15229 → +15409、总 +18773 → +18953；`speculative.h` 首次进改动集（55 = 54 + 1），树 `0be1ec9c8` / `c06e0ed6b`。

④ 2026-11-11 **链上 MTP 的 catch-up，两笔**。第一笔让草稿真的看到 prompt（`speculative.cpp` +239 −3、
`server-context.cpp` +22 −1，`begin()` 从 `DONE_PROMPT` 移到 prefill 开始），主补丁 → +15669、总 → +19213（树
`b3cd73531` / `21bf46be9`）；第二笔把开关改成**逐请求自适应**（`speculative.cpp` +166 −34、`server-context.cpp` +60，
经 `QLH_CHAIN_DRAFT_CATCHUP_AUTO` 交给 spec 层、接口零改动），主补丁 → +16680、总 → +20345，树 `586a7e7a1` / `548c3dfb0`。

⑤ 2026-11-13 **v2.2 重导**：上游 31 个提交（`09338b85e` → `5676dfbe0`，tag `b12342`）把 18 个 NextN / layer-input /
embd-row / state-row API 从 staging header `src/llama-ext.h` 转正进 `include/llama.h`，主补丁补 staging 声明的那段
（`src/llama-ext.h:126`）被上游删掉，**v2.1 主补丁因此 apply 不到新基底**。重导走反推：从 v2.2 完整树反向 apply 旧
SLIM-ARC 补丁得 main-only 树 A，再正向生成两个 `git diff`。结果主补丁 +1443 −7 → **+18123 −237**、总 +20345 →
**+21788**；SLIM-ARC 仍是 13 文件 / +3665 −1，内容一字未变，只有 index 行与 `@@` 行号随基底偏移。树：`5676dfbe0` +
主补丁 = `2541c9d29`，再叠 SLIM-ARC = `61781c712`，与 fork `slimarc/v2.2`（HEAD `6d0fa16c9`）的树逐字节相同。

**基提交是 `5676dfbe0`**（v2.2 口径；v2.1 结题基底 `09338b85e`，KVMem 补丁仍绑它）。旧基线 `ec25cedda`（树
`e14b92c9`，68 文件 / +18086 −504）留在 `master` 上，已不是补丁基底。

**三个补丁的叠加点不同，别叠错**：

| 补丁 | 叠在什么之上 | 叠完的树 |
| --- | --- | --- |
| `llamacpp-split.patch`（主） | 直接叠在基提交 `5676dfbe0` 上 | `2541c9d29` |
| `llamacpp-slimarc.patch` | 叠在**主补丁之上**（不直接叠基提交） | `61781c712` |
| `llamacpp-kvmem.patch` | 叠在 **v2.1 的 `548c3dfb0` 之上**（主 + SLIM-ARC 之后；**未随 v2.2 重导**，对 v2.2 树 apply 会冲突） | `192b2a3d0` |

**顺序是硬性的**：主 → SLIM-ARC → KVMem。KVMem 的 `src/llama-context.cpp` 改动相对**叠了 SLIM-ARC 之后**的版本写，
直接叠在主补丁上会在 `src/llama-context.cpp:14` 冲突（已实测 `git apply --check` 失败）。「主 + KVMem」**不是**合法组合。

两者是叠加关系，主补丁可单独用：

```bash
git worktree add /tmp/v1 5676dfbe0 && cd /tmp/v1
git apply docs/llamacpp-split.patch        # 纯分段，无 SLIM-ARC
git add -A && git write-tree               # 必须 = 2541c9d29（纯分段树）
git apply docs/llamacpp-slimarc.patch      # 再叠增量，回到完整功能
git add -A && git write-tree               # 必须 = 61781c712（= fork HEAD 的树）
```

上面这套叠加校验已经写成 `scripts/export_patches.sh`：它建临时 worktree 复算两棵树、断言 overlay，无参数时只打印统计，
`--write` 才覆盖补丁文件。构建命令与开关矩阵见 [`BUILD.md`](BUILD.md)。

补丁文件必须按字节保存（仓根 `.gitattributes` 的 `*.patch -text`）。`core.autocrlf=true` 下把补丁转成 CRLF 之后照样
apply 成功——只有 trailing-whitespace 警告——但文件末尾 `\ No newline at end of file` 会多出一个裸 CR，产出的树与文档里的
tree hash 不再相等。静默错，属于「验收命令跑了、也过了、结果不对」那一类（CRLF 版主补丁得到 `d679f09b`，正确是
`1dcbce39`）。现行判据是「主补丁 → `2541c9d29`，叠加 → `61781c712`」，导出后 `grep -c $'\r'` = 0。

**第二处缺陷（2026-11-10）**：fork 的 `core.autocrlf` 系统配置是 `true`、仓内 `.git/config` 显式设成 `false`，而工作区
历史上混行（上游文件 CRLF、后期手写 LF）。`git add` 会把 CRLF 原样存成 blob，patch 里出现成片 CR；而 patch 应用到
干净基底后是 LF，两边一比就是两个不同的树——`overlay mismatch`，树号差 2777 行。两条判据都要过：导出后
`grep -c $'\r' <patch>` 必须是 0，且每个文件 `git cat-file blob HEAD:<path>` 也不能含 CRLF；重导时都套上
`git -c core.autocrlf=true`（`apply -R` 与 `diff` 各一次）。

- 只应用主补丁得到的纯分段树能独立 `--clean-first` 全量编译（385 个编译单元，`dflash-taps.exe` /
  `split-smoke.exe` / `split-rows.exe` 等目标齐全），并独立通过 `verify/verify.sh` 与
  `verify/verify_chain_death.py`；没有 SLIM-ARC 不影响分段链的任何一条验收。v2.1 实测过两版：
  `0c307d82a`（44 文件，2026-11-09，`verify.sh` 十二组 53 项——组 13 于 2026-11-12 加入、组 14 于 2026-11-13 加入（20 项），
  现行完整验收 **95 项**，见 `INDEX.md` §5——+ chain_death 36 项，日志 `.qlh_tmp/pure-split-acceptance/`）与 `586a7e7a1`（45 文件，2026-11-11，
  `verify.sh` **52/0/0**、chain_death **36/0**，日志 `.qlh_tmp/pure_split_recheck.log`；两版差 4 文件 / +605 −12）。
  v2.2 的 `2541c9d29` 于 2026-11-14 独立验过（`H:/wt-mainonly` + `build-v22-mainonly`，`git write-tree` 逐字节等于目标树；`--clean-first` 191 步全过，`verify.sh` **72 PASS / 0 FAIL / 2 SKIP**——72 = 77 − 组 12 的 1 − 组 13 的 4，这两组依赖别的构建与补丁，不在「主补丁能独立编译」这句话的范围），日志 `.qlh_tmp/mainonly-verify.log`。
- `1dcbce39`（44 项）有一趟 `PASS=34 FAIL=10`，两处都是环境：组 5 的 `-np 4` 启动瞬间死掉（日志 0 字节，
  单跑两次都过）；组 10 的 dflash 十项里九条红，`taps_dflash_chain.log` 只有 `dflash-taps.exe: Device or
  resource busy`（整模那趟刚退出、文件锁未释放）。run2 同一份二进制得到 44/0。判据是「日志里有没有数值
  不一致」，不是「这一趟绿不绿」。
- 叠加两个补丁后 `git write-tree` = `61781c712`（fork HEAD 的树），这是「拆分没有改动任何内容」的唯一判据，
  比「能 apply」更强。
- SLIM-ARC 与分段链零耦合：那 10 个文件只 include 彼此和上游 `ggml-backend.h`，不碰本 fork 新增的任何头；
  6 处钩子摘掉后，`llama-context.cpp` / `llama-model-loader.cpp` 的对应片段与上游逐字节相同（重导时用字符串
  精确匹配摘除，每处断言「恰好出现一次」）。

**历史全文**：`docs/llamacpp-qlh-slimarc.patch`（27 文件 / 4014 行），基提交更早，只到第三代；保留作第一 / 二代对照，不能正向应用。

fork 侧历史：v2.1 是 `09338b85e` + 单提交 `4acf03c8b`（树 `548c3dfb0`）；v2.2 是 `5676dfbe0` 之后 7 个提交
（`36cd4e75a` … `6d0fa16c9`，树 `61781c712`）。旧的一对（`e0f142d` + `ec25cedda`，树 `e14b92c9`）留在
`master` 作历史。开发期 wip 已 squash；更早记录的 squash 前后 tree 值 `d628e2a8` 与那代 HEAD 的 `e14b92c9`
不同（squash 后又合入修复与两次上游移植），只作历史记录，不能重放。

## 第一代（K-Llama / QLH）：层输出通道

让上下文多出一个「第 N 层输出」的槽位，供段间接力取数：`src/llama-context.cpp` 的 `layer_inp` 多分配一个槽位
（`n_layer + 1`），索引 `lid == n_layer` 即末层输出（`output_norm` 之前）；`set_embeddings_layer_inp` 的断言同步
放宽。在 `output_norm` **之前** 登记该槽位，覆盖 9 个架构：`qwen2` / `qwen35` / `gemma4` / `llama` /
`openai-moe` / `qwen3` / `qwen3moe` / `qwen35moe` / `qwen3next`。对外是 4 个导出符号，shim 用到
`llama_set_embeddings_layer_inp()` / `llama_get_embeddings_layer_inp()`。

`t_h_nextn` 的挂点因架构而异，当层段上游复用会在 Qwen3.5 上多过一次 RMSNorm：`qwen2` 的 `t_h_nextn` 在 `output_norm` 之前，`qwen35` 在 **之后**（消费方是 MTP
head），接力首步就分叉。代价是这套改动要跟着每个架构走，9
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
`layer_inp` 槽位，只改 qwen35 / qwen35moe；**lazy 全局张量**（head 不需要 `output`、tail 不需要 `token_embd`、middle
都不需要，middle 段因此省 ~913 MiB）；**relay 末层不裁剪**——`qwen35moe.cpp` 的 `crop_before_nextn` 在 `n_outputs == 0`
时会把末层裁成 `[n_embd, 0]`，而 `t_layer_inp[n_layer]` 登记的正是这条张量，于是 `extract_layer_inputs` 的
`row_floats == n_embd` 断言失败；修法是 `qlh_keep_full_rows = embeddings_pre_norm || embeddings_layer_inp[n_layer]`，
为真时跳过裁剪并在 LM head 之前补回 `ggml_get_rows`。Windows 上不触发这条：编译宏 `QLH_ABI_UPSTREAM_HEADER` 把
`batch.logits` 钉成 1，绕开了裁剪；两条路实测 hidden 逐位相同，但只有补丁这条路不依赖编译期定义。

第二代反而更小：第一代要 9 个架构各自登记挂点，第二代把「取哪一层的输出」变成「只加载哪几层」——后者是加载器的事，
与架构无关，架构侧只剩 2 处 tap。

## 第三代：SLIM-ARC prefetch（2026-10-31，可选增量）

上游 SLIM-ARC 是给 Linux 写的，本机是 Windows，所以是带回移植的接入，只接两级里的第一级（`prefetch`）；第二级
（on-demand 加载器）未接，理由见 [`SLIM-ARC-接入计划.md`](../design/SLIM-ARC-接入计划.md)。新增 5 对 / 10 个文件
（`slim-arc-prefetch.{h,cpp}` 2085 + 495 行，加 `page-range` / `expert-reclaim` / `expert-residency` /
`expert-transition`），合计 3479 行；自包含，只 include 彼此和上游 `ggml-backend.h`，能整块摘下。

这一代不在主补丁里，只存在于 `docs/llamacpp-slimarc.patch`（13 文件 / +3665 −1），叠在 `docs/llamacpp-split.patch`
之上才回到完整树。6 处钩子（归并成 4 行）：

| 文件:行（完整树 `61781c712`） | 位置 | 作用 |
| --- | --- | --- |
| `src/CMakeLists.txt:53-60` | `LLAMA_SLIMARC_SOURCES` | 注册 5 个 `.cpp`，用 `SKIP_UNITY_BUILD_INCLUSION` 挡在 unity 批次外——这批文件带 `WIN32_LEAN_AND_MEAN` 和 POSIX 名字，混进 unity 会污染同批翻译单元 |
| `src/llama-model-loader.cpp:1665-1700` | `init_mappings()` 末尾 | 只有设了 `SLIM_ARC_PREFETCH` 才构造全局调度器，把每个权重的 `(mmap 基址 + 偏移, 大小, 层号)` 注册进去 |
| `src/llama-context.cpp:1959-1974` | `process_ubatch()`：`ggml_backend_sched_reset()` 之后 | 调度器在装且 `n_gpu_layers() == 0` 时把 `eval_cb` 装到 sched 上，用户 `cparams.cb_eval` 串在它里面；否则原样用用户回调 |
| `src/llama-context.cpp:1982-1994` | `process_ubatch()`：`build_graph()` 之后 | `prepare_graph(gf)` 扫新图、记下每层第一个节点，运行期由 `eval_cb` 在层边界发预取 |

表里 4 行覆盖 6 处：`set(LLAMA_SLIMARC_SOURCES ...)` 与 `add_library()` 里的 `${LLAMA_SLIMARC_SOURCES}` 各一处，
`src/llama-context.cpp:17` 与 `src/llama-model-loader.cpp:8` 的 `#include` 各一处，再加两个 `.cpp` 里的代码块。
摘掉后 `process_ubatch()` 恢复成上游的单行 `ggml_backend_sched_set_eval_callback(...)`，`init_mappings()` 末尾恢复成
空行加 `}`，两处与上游逐字节相同。**换基底之后行号必须重新定位**：这两块在 `e0f142d` 基线是 1613 / 1638，v2.1 的
`09338b85e` 上是 1778 / 1803，v2.2 的 `5676dfbe0` 上是 **1959 / 1982**（每次重导都逐个核对过，`strip_hooks.py` 一字未改
就精确命中）。摘除脚本用整块字符串精确匹配 +「恰好出现一次」断言，对不上直接失败。

**层号怎么来**：上游从图节点名读 `blk.N`，本树取不到——GGML scheduler 在计算前把每个节点重命名成 `node_%d`，真正的
张量名留在操作数上。本树改成读 `graph_get_cb()` 给每个 tensor 打的 `-<层号>` 后缀（`ggml_format_name`，上游行为），
`prepare_graph` 再按「层号只会前进」把名字尾数恰好是数字的节点滤掉。按上游写法接时 `calls` 恒为 0；改成读后缀后
`calls=5`（3 个 token 各一次通知加 prefill）。win32 移植面：`madvise(WILLNEED)` → `PrefetchVirtualMemory`、`DONTNEED` →
`DiscardVirtualMemory`、`NORMAL/RANDOM/SEQUENTIAL` → 空操作；`mincore` 不支持、返回「未知」（消费方是专家热锁 LRU，
未知即跳过）；页大小用 `GetSystemInfo().dwPageSize`；`mlock` / `munlock` → `VirtualLock` / `VirtualUnlock`。另有两处
MSVC 兼容：`NOMINMAX`（否则 `windows.h` 的 `max` 宏会打掉 `std::max` 与 `numeric_limits<T>::max()`），以及用 `_umul128`
拼 64×64→128 乘积替代 `unsigned __int128`。

**当前生效范围**：默认关闭，`SLIM_ARC_PREFETCH=1` 才装调度器；整模在页缓存里时 `issued` 为 0，只在「段装不下、真的在读盘」的降级形态里有事可做。

## 第四代（现行）：分段链与跨进程

前三代解决「每台机器只装自己那几层」，第四代解决「这么多段怎么跑成一次推理」。一个 `llama_context` 成为链的驱动者：
本地段输出 pre-norm hidden，后续段（本地或远程）消费 hidden 继续算，最后一段产 logits。`llama-server` 的 slot、prompt
cache、`seq_rm` 通过聚合 memory 看到整条链，不需要知道段在哪台机器上。

远程链的投机是这一代引入的：协议 v7 加逐行 argmax 的验证通道，v8 把 MTP draft 下沉到远程 tail（`OP_MTP_DRAFT`），v9
又把 DFlash / DSpark 的 `layer_inp` tap 挂到 decode 回复上，随后补了 grammar 与 reasoning budget 两条兼容路径。
v2.2 升到 **v10**：帧尾 `rows` 段逐行带 `seq_id`、回复可带 N 行 logits、`stage_id` 加四个身份字段
（`n_seq_max` / `n_ctx_seq` / `n_rs_seq` / `n_verify_rows`）、四个投机通道改 per-seq。语义见
[`V22-MULTISEQ.md`](../design/V22-MULTISEQ.md) §3 与 [`V22-PERSEQ-SPEC.md`](../design/V22-PERSEQ-SPEC.md)。
这些都是增量改协议 / 加判据，没有动链的驱动结构，所以各占一行而不是另开一代。**收益结论按终态写**：能做，但 ROG
五类真实 prompt 上 MTP k=2 中位只有 **+6%**（区间 +8.4% ~ −1.2%），弱机器与回环上是负的；判据是「无回退时与
**整模投机**逐字节相同」。数字与归因见 [`SPLIT-MEASUREMENTS.md`](../measure/SPLIT-MEASUREMENTS.md) §9、
[`SEGMENTED-LIMITS.md`](../limits/SEGMENTED-LIMITS.md) §2.2。

机制行有 26 条，完整表、六处语义（`seq_rm` 失败显式化、`--split-rs-seq` 不再必需、失败不区分对端拒绝与没连上、
链死自愈、checkpoint 覆盖全链、v8/v9/v10 的槽位顺序与通道优先级）与验收表在 [`PATCHES-DETAIL.md`](PATCHES-DETAIL.md) §1、§3。
这里只留骨架：层范围进参数（`llama_model_params.split_layer_lo/hi`）→ `decode_split` 走标准 embedding 路径
传 hidden → `llama-memory-split` 聚合本地 / 远程 memory → `llama-split-{proto,net,remote}` 与 `tools/split-stage`
承担定长协议 v10（v3 定长头 + v4 身份握手 + v5 候选通道 + v7 验证行 + v8 远程 MTP draft + v9 tap + v10 逐行 seq_id）
→ server 入口 `--split-tail-range` / `--split-remote` / `--split-rs-seq`。段级 offload、MoE 缓存与显存预算三行同样在
第四代里，语义见 [`SEGMENT-OFFLOAD.md`](../measure/SEGMENT-OFFLOAD.md)。

第四代仍成立的边界（架构覆盖、`seq_rm`、链上投机收益、checkpoint 过网量、单序列、`/health` 语义、单独层范围的检查
覆盖）逐条记在 [`SEGMENTED-LIMITS.md`](../limits/SEGMENTED-LIMITS.md) §2，那里写的是代价而不是 TODO。

## 与上游 193 个提交的重合（2026-11-09 rebase；2026-11-13 又对齐 31 个提交）

v2.1：fork 从 `e0f142d` 整体 rebase 到上游 193 个提交之后的 `09338b85e`（193 个提交 / 659 文件）。把改动 `git apply --3way`
打到新基底后剩 54 文件 / +18773 −240；17 个文件因上游已含对应提交而退出变更集（16 个与上游逐字节相同，`src/models/dflash.cpp`
因 device-buffer 诊断行重新进入）。冲突分布、上游提交对应关系、两处「上游基线漂移」与验收见 [`PATCHES-DETAIL.md`](PATCHES-DETAIL.md) §4。

v2.2：上游再进 31 个提交（`09338b85e` → `5676dfbe0`，tag `b12342`），把 staging API 转正进 `include/llama.h`；
主补丁因此重导（见「重导沿革」⑤），改动面 +20345 → +21788。

## 与 llama.cpp 原生 RPC 的分界

原生 RPC（`-DGGML_RPC=ON`）也能跨机，但两者解决的不是同一个问题。RPC 的 `SET_TENSOR` / `SET_TENSOR_2D` 上传张量、
`GRAPH_COMPUTE` 让远端执行子图、`GET_TENSOR` 取回结果：**RPC 让远端当「远程 GPU」，主机仍然要有完整的模型**
——权重从主机发到远端，主机维护全局计算图，远端只执行派过去的 op。**本项目让每个节点当「一段模型」，没有中心**：
整模文件在各节点本地，各自 mmap 后只 touch 自己那几层，段间只传 `n_embd` 个浮点。

| 维度 | 原生 RPC | 本项目 |
| --- | --- | --- |
| 权重从哪来 | 主机通过 `SET_TENSOR` 发给远端 | 各节点本地读整模的对应层 |
| 首次跨机流量 | 权重总量（GB 级） | 0，只要本地有整模 |
| 主机必须装下整模 | 是 | 否（只装自己那几层） |
| 每步流量 | 激活 + 图指令 | hidden 一个向量（FP32，2048 × 4 B = 8 KB） |
| 计算图 | 主机维护全局图 | 每段是独立进程，无全局图 |
| 异构 | 需要相同的 ggml backend | 自定义 TCP 协议，可跨不同 CPU/GPU |
| 层范围加载 | 不支持，要自己切模型 | `LLAMA_LAYER_RANGE` 原生支持 |

一句话：RPC 把设备虚拟化，模型仍是一份；本项目把模型分段，每段各自持有。前者适合「一台主机 + 若干算力节点」，
后者适合「几台都装不下整模的机器」。代价：本项目的段是黑盒，段内不能跨机器切分算子；RPC 可以把任意 op 派到远端。

## 第五代：KVMem KV 虚拟化（2026-11-12，可选增量）

### 它解决什么

分段链解决了「整模装不下」，但 **KV cache 仍然 ∝ `n_ctx`**：35B 的 head 段 15.69 GiB 权重之后，`-c` 只能开到
4096–8192 量级，而 agent 场景要的是 100K+ token 的工作区。KVMem 把 KV 虚拟化到 host RAM / SSD，GPU 只留有界工作集
（`budget + gen_reserve`），通过 `llama_memory_i` 接入——**与分段链是嵌套关系而非竞争**：`llama_memory_split` 的
head 换成 `llama_memory_kvmem`（hook 在这里生效）、tails 仍是各 stage 的 memory，而 split 对 head 只走接口，
所以不需要改 split 的接口。

### 本补丁包含什么

**52 文件 / +1266 −135**，分三类（adapter 主体不在本补丁里——那是 `SgfKrc/kvmem` 的源码，约 9.4 K 行，本补丁只含 fork 侧接线）：

| 类 | 内容 |
| --- | --- |
| **hook 点** | `llama_model::create_memory` 的 factory hook（`src/llama-kvmem-factory.h`）、`attn` 的 Q/K capture、KV cache 跳过空洞清理、CUDA stage-in、fused-node 注册 |
| **fork 侧接线** | `--kvmem` / `--kvmem-budget` / `--kvmem-gen-reserve`（`common/arg.cpp`）、`src/llama-kvmem-bridge.cpp`、`common_init_result` 里的调用点 |
| **上游漂移适配** | `ggml-rpc.h` 的 `RPC_PROTO_PATCH_VERSION`（MAJOR 8→9 后取 1）、`gf_res_prev` / `gf_res_decode` 从值 vector 改成 unique_ptr array 后的 `res->reset()` 修正 |

### 依赖

**必须配合 `SgfKrc/kvmem` 的 `fix/seq-pos-min-max` 分支**（HEAD `461b05c`），三条修复是硬性依赖；构建时
`-DLLAMA_KVMEM=ON -DLLAMA_KVMEM_ROOT=<kvmem 仓库根>`：

| commit | 修什么 |
| --- | --- |
| `f04053a` | `seq_pos_min/max` 报 store 的逻辑历史（否则 prompt cache 全废、checkpoint 从不创建、有个 `GGML_ABORT` 悬着） |
| `723d424` | 每个 stage 在序列起点 rebind `memory`（否则双 context 下 head 的 capture 全落到 tail，一驱逐就 500） |
| `461b05c` | MTP follower 也 rebind（否则 split + MTP 同进程时 capture 错位） |

### 验收（2B qwen35，本地双段 `0:12+12:24`，`-np 1`，`-ngl 0`）

| 项 | 结果 |
| --- | --- |
| MTP + KVMem（`budget 8192`） | 200，`draft acceptance = 0.86207 (50/58)` |
| 驱逐（`budget 1024` + 3638 token，pool = 3072） | 200，`need_offload=1` ×3、`KVMEM_PRESSURE` ×4 |
| **secret-code 保留** | 200，`content: 'XK-7734-QP'`——驱逐后仍答对 |
| 确定性 | 两次输出 byte-identical，sha256 相同 |
| 对照 | 不带 `--kvmem` vs `--no-kvmem` 逐字节相同；**无驱逐时 KVMem 输出也是同一 sha256**（对输出透明） |

### 两个坑

1. **`-lv 3` 下 KVMem 核心日志不可见**（这个 fork 的 server 在 `-lv 3` 下过滤 llama.cpp 核心 INFO），验收必须
   `-lv 10` 或 `KVMEM_TRACE=1`。
2. **adapter 是 UNITY_BUILD 编进 llama 的**，ninja 的 depfile 不跟踪外部源——改完 adapter 直接 `cmake --build`
   会说 nothing to do，必须先 `touch build-kvmem2/src/CMakeFiles/llama.dir/Unity/unity_11_cxx.cxx` 再编。

### 明确不做

多序列（实测 `-np 2` 是 0.81×）、异步流水线（单序列可重叠比例为 0）、本地链 `-np > 1` + KVMem（进程级单例会再错位）。
详见 [`KVMEM-评估.md`](../design/KVMEM-评估.md) 与 [`ARCHITECTURE.md`](../design/ARCHITECTURE.md) §11。
