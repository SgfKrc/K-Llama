# SLIM-ARC 接入计划

> 状态：**方案**（§6.1 / §6.2 的结论已被实现推翻，见文首「现状」）
> 更新日期：2026-11-11
> 适用范围：SLIM-ARC 在什么条件下接入、接入后按什么顺序启用；读者为部署与内核侧维护者。

本文只回答两件事：SLIM-ARC 补丁在什么条件下接进来，以及接进来之后按什么顺序启用。
它不描述 SLIM-ARC 本身能做什么——那部分见 `docs/deploy/PATCHES.md` 与上游仓库。

现状（2026-11-11 复核）：`prefetch` 一级已编译进 `llama.dll`（默认关闭，`SLIM_ARC_PREFETCH=1` 开启），
并且**已经能真的发出预取请求**——§6.1 的「`memory_budget_` 恒为 0」与 §6.2 的「层号对不上」两条已修：
`default_prefetch_budget_bytes()` 从可用物理内存推导预算（可用 /8，夹在 64–1024 MiB，`SLIM_ARC_BUDGET_MB`
可覆盖）并按单层权重封顶；层号改由图节点名后缀 + `ggml_backend_sched_eval_callback` 提供，日志形如
`layer signalling armed: 40 boundaries over 3790 graph nodes, layers 0..39`。`on-demand` 加载器尚未实现。
§6.1 / §6.2 的结论已被实现推翻，保留作历史；§6.3（iGPU 上装不上）**仍成立**。

接入形态：`slim-arc-prefetch.{h,cpp}` 与它依赖的 4 个模块编译进 `llama.dll`
（`src/CMakeLists.txt` 里 `SKIP_UNITY_BUILD_INCLUSION`，不参与 unity 分批）。两个挂点在
`llama-model-loader.cpp::init_mappings`（注册每个权重的 mmap 区间）与
`llama-context.cpp::graph_compute`（每步通知当前层）。win32 侧把
`madvise(WILLNEED)` / `madvise(DONTNEED)` / `mincore` / `sysconf` / `mlock` 换成
`PrefetchVirtualMemory` / `DiscardVirtualMemory` / `GetSystemInfo` / `VirtualLock`。
实测通路（§6.1 修之前的旧口径）：`llama-cli -ngl 0` 跑 3 个 token 后打印
`[SLIM-ARC-PREFETCH] calls=5 issued=0.0MB window=3` —— 当时只证明通知链路工作，预取一次没发出去；
修通后见 §6.5 的 `layer signalling armed` 与 `issued > 0`。

---

## 1. 常规形态与降级形态

系统有两档工作形态，默认停在第一档。

**常规档**：整模 + `LLAMA_LAYER_RANGE`。每台设备一段，每段的权重总量能驻留在该设备的
内存里。此时 mmap 一次装入，解码过程不产生新的磁盘读取。这是 K-Llama v2 的默认形态，
也是当前两台机器跑 35B 时用的形态（本机 0:32、ROG 32:40）。

**降级档**：某台设备的一段装不进它的内存。此时 mmap 仍然能跑，但每一步都要为未驻留的页
产生缺页，从磁盘取回。系统不会拒绝启动，只是变慢——慢多少取决于磁盘随机读性能与缺页的
空间局部性。

SLIM-ARC 只服务于降级档。它把「随机缺页」变成「按层边界顺序预读」，减少每步的随机
I/O 次数。在常规档里它没有作用，因为那时根本没有缺页可优化。

## 2. 触发条件

**判据是"是否真的在读盘"，不是"是否装得下"。** 这两件事在页缓存足够大时会分开。

四个条件同时成立才考虑接入：

1. **段权重超过设备可用内存**。按 `topology.py` 的 `stage_bytes()` 与 psutil 的
   `virtual_memory().available` 比对。
2. **解码期间 `Pages Input/sec` 持续大于 0**。这是唯一的实测判据，用
   `Get-Counter "\Memory\Pages Input/sec"` 在跑链路时采样。
3. **加机器不可行**。给这段再分一台机器就能装下时，优先加机器——比在缺页上做优化彻底。
4. **调边界不可行**。层数只能按 4 层一档走（见 §2 末尾），降一档往往让邻段超容量。

条件 1 成立而条件 2 不成立时，SLIM-ARC 无事可做：权重已经在页缓存里，缺页只是页表映射，
没有磁盘 I/O 可供优化。**这时优先加内存或减少该段层数。**

### 为什么层数只能按 4 层一档走

`qwen35moe` 的 `full_attention_interval` 是 4（GGUF 里 `qwen35moe.full_attention_interval=4`），
类型序列是 `LLL F` 重复，full attention 落在 3 / 7 / 11 / … / 39。`src/models/qwen35moe.cpp:24`
用**本地层号**推算类型：

```cpp
hparams.is_recr_impl[i] = (i < hparams.n_layer()) && ((i + 1) % full_attn_interval != 0);
```

本地层号在 `get_weight` 里被映射回源层号 `lo + i`，所以只要 `lo` 是 4 的倍数，两者模 4
同余，类型对齐。而 tail 段的 `hi` 必须等于模型总层数 40，否则模型缺层。两个约束合起来：

- `lo` ∈ {36, 32, 28, …}，即 40 减去 4 的倍数
- 段层数 ∈ {4, 8, 12, …}

`32:38` 这类 6 层的划分有三重违规（`lo`、`hi`、`hi-lo` 中 `hi` 与差都不是 4 的倍数），
`topology.py` 在解析阶段就拒绝：`stage[1].layers=6 is not a multiple of 4`。

条件 1 单独成立不算触发。当前 ROG 的 8 层段就满足条件 1：4.16 GiB 权重压在 3.89 GiB 可用
内存上，实测 worker 模式 53.55 ms 对独占内存时的 30.85 ms，差 22.7 ms。

但它不满足条件 3，因为层数调整虽然只能按 4 层一档走，仍然走得通——只是不划算：

| 划分 | ROG 内存 | ROG 段 | 本机段 |
| --- | --- | --- | --- |
| 32:40（当前） | 4.16 GiB，可用压到 0.82 | +22.7 ms 内存压力 | 58–72 ms |
| 36:40 | 2.08 GiB，宽裕 | 无压力 | 96.9 ms（背 36 层） |

降到 4 层意味着本机要背 36 层，代价大于 ROG 省下的
22.7 ms。**已实测**：36:40 为 17.7 t/s、32:40 为 17.0 t/s，只快 4%，而负载从桌面机转到
笔记本（`ANDROID-SURFACE-段接入评估.md` §5）。所以当前 32:40 就是这台机器上的最优划分，
22.7 ms 属于该配置的固有代价。

## 3. 接入前必须先量的两件事

SLIM-ARC 的收益依赖"顺序读明显快于随机读"这个前提。接入之前先量它，否则可能白做。

**第一件：缺页的实际代价。** 在目标设备上跑装不下的段，同时采 `\Memory\Pages Input/sec`。
如果该值在解码期间接近 0，说明权重实际都在页缓存里，没有真正的磁盘 I/O，那 SLIM-ARC
无事可做。ROG 现在就是这个状态：段装不下，但 `Pages Input/sec` 是 0，因为整模 20.8 GiB
在页缓存里躺着。

**第二件：SSD 上顺序读与随机读的差距。** 用同一份权重文件，分别按 4 KB 随机偏移读与
按层边界顺序读，比较吞吐。NVMe 上这个差距可能只有 2–3 倍，而 SLIM-ARC 的预读只能拿到
其中一部分。

两项都量完，收益低于 10% 就不接。

## 4. 启用顺序

接入分两级，按代价从低到高，前一级无效再接下一级。

**第一级：`prefetch`。** 现有的 SLIM-ARC 补丁里只有这一项是接得上的，它做的是对已有映射
调 `madvise(WILLNEED)`。用来在每步开始前把下一层预热进页缓存。改动局限在加载器，不碰
计算图。

**第二级：`on-demand` 加载器。** 这一项在 SLIM-ARC 里存在但从未接进任何补丁，需要自己写：
在每层计算前按需读取，计算后释放。它与 `prefetch` 的区别是后者只是建议内核预读，前者
控制实际的驻留集合。

两级都接完之后，用 `docs/measure/MEASUREMENT-PITFALLS.md` 的量法验收：固定 `n_past`、跳过前
10 步、报谷底与后段。数值判据是逐位一致——降级档解决的只是读写顺序，不该改变计算结果。

## 5. 不做什么

- 不引入切好的层段工件。整模 + `layer_range` 已经是常规档的形态，降级档沿用同一份整模。
- 不为常规档接入 SLIM-ARC。常规档没有缺页，接了只是多一层间接。
- 不把降级档做成默认。它比常规档慢，只有在装不下时才用。

---

## 6. prefetch 曾从未真正触发（已修，保留作根因记录）

`issued=0` 不是「整模已在页缓存里」，而是两处移植缺失叠加的结果——预取一次都没发出去过。
**2026-11-11 复核：§6.1 / §6.2 两条已修，预取现在能真的发出（`issued > 0`）；§6.3 仍成立。**

### 6.1 `memory_budget_` 恒为 0

`slim-arc-prefetch.h` 里是 `std::atomic<size_t> memory_budget_{0}`，唯一的写入口是
`set_memory_budget()`，而它**在整个 `src/` 里没有任何调用点**。注释写明这个值「供
`unified_io_scheduler` 分配权重预取额度」——而 `unified_io_scheduler` **没有移植进来**
（全仓库只在三条注释里出现）。

后果在 `slim-arc-prefetch.cpp:773`：`admitted = available - available % page_size`，当
`memory_budget` 为 0 时恒为 0，`selected` 为空，`issued` 永远是 0。**表现和「页缓存命中」
一模一样，但原因完全不同。**

**已修（2026-11-11）**：构造函数改为 `memory_budget_.store(default_prefetch_budget_bytes())`——预算由
可用物理内存 /8 推导、夹在 64–1024 MiB（`SLIM_ARC_BUDGET_MB` 可覆盖），`plan_request()` 另按单层权重
封顶。`set_memory_budget()` 仍然没有调用点，但已不是唯一入口。

### 6.2 补上预算之后 `ranges` 仍然为空

把默认值临时改成 `256 MiB` 并重编后，`issued` 仍是 `0.0MB`。说明还有第二层：
`plan_request` 产出的 `request.ranges` 是空的。`init_mappings` 报告注册了 753 个张量，
所以问题在「按 `current_layer + 1 .. +window` 从 `tensors_by_layer_` 取」这一步，不在注册。

`docs/deploy/PATCHES.md` 记过同类历史：层号最早从图节点名读 `blk.N`，在本树取不到（GGML
scheduler 把节点重命名成 `node_%d`），改成读 `src[0..]` 之后 `calls` 才从 0 变成 5。
**现在 `calls=11` 但层号可能仍未对上，这是首要怀疑对象。**

**已修（2026-11-11）**：层号不再从 `src[]` 反推，改用图节点名后缀 + `ggml_backend_sched_eval_callback`
（做法见 §6.5）。日志确认 `layer signalling armed: 40 boundaries over 3790 graph nodes, layers 0..39`，
`issued` 不再恒为 0。

### 6.3 iGPU 上根本无法安装

`llama-model-loader.cpp:1583` 的安装条件是
`use_mmap && !mappings.empty() && SLIM_ARC_PREFETCH`。而 Vulkan 核显的
`.mmap_support = !ctx->is_integrated_gpu`，`load_tensors()` 据此把 `use_mmap` 清成
false（日志里 `load_mode = none`），**这个调度器在 iGPU 上永远装不上**。

实测确认：把 Surface 的 tail 段改成 32 层（17.0 GiB 进 16 GiB 机器，`size_data
17408.94 MiB`），加载成功但日志里 `SLIM-ARC` 零条。所以「用 Surface 造装不下场景来验证
prefetch」这条路走不通，与配置无关。

### 6.4 三条事实对计划的影响

- **`issued=0` 不是「无事可做」的证据**，它只是「预取没发出去」（修之前的口径；现在能发出）。
- **验证顺序**：先补 `memory_budget` 的来源与层号匹配，确认 `issued > 0`，再谈收益。
  在此之前量到的 tg 没有意义。
- **弱设备不是唯一出路**：本机 32 GiB 内存虽然装得下整模，但只要修通 §6.1/§6.2，就能在
  本机确认调度器真的在工作；收益仍要在真正缺页的机器上量（Y700 或人为压内存）。

### 6.5 层号对齐与剂量实验：抖动区里预取是负收益

`min_layer` 恒为 0 的根因在 `llama-context.cpp:2791`：循环条件写着 `&& min_layer != 0`，
而图里第一个节点的 `src[0]` 就是 `blk.0.*`，于是这个循环必然在第一轮退出。每个 token
重复预取 layer 1..3。

**做法**：不从 `src[]` 反推，改用图节点名后缀。`graph_get_cb()` 命名节点时就用
`"%s-%d"` 带上层号（形如 `ffn_out-12`），所以建图完成后扫一遍 `ggml_graph_node()`，把
层号变化处的节点指针存成有序表，再装一个 `ggml_backend_sched_eval_callback`：回调
`ask==false` 表示该节点刚算完，此时通知 `notify_layer_compute(L-1)`。这样拿到的是真实
的层序时序，而且不碰建图、不切图。配套两个调试开关：`SLIM_ARC_NOOP=1` 保留回调与切图
但不 advise；`SLIM_ARC_MAX_CALLS=N` 限制一个进程能发的请求总数。

**验证**（Surface，纯 CPU，`-ngl 0 -t 6 -p 0 -n 16 -r 1`，22 GB 模型 / 16 GB 内存）

日志确认层号全通：`layer signalling armed: 40 boundaries over 3790 graph nodes, layers 0..39`。

| 配置 | tg16（t/s） | 相对 OFF |
| --- | --- | --- |
| OFF | 2.77 / 2.85 | — |
| NOOP（装回调 + 切图，不 advise） | 2.86 / 2.89 | +1% |
| `MAX_CALLS=3` | 2.91 | +2% |
| `MAX_CALLS=40` | 1.99 | −29% |
| `MAX_CALLS=167`（实际无上限） | 1.03 | −64% |

两条结论：

1. **回调和切图不花钱**。NOOP 与 OFF 齐平，方案本身没有结构性代价；「40 次分割 submit」
   不成立。
2. **慢的是「持续把页读进来」，且与量严格反比**。3 次请求还略快于 OFF，167 次掉到 1.03。
   `PrefetchVirtualMemory` 单次调用并不贵（否则 cap=3 就该塌），贵的是它引发的换页。

**为什么**：本机 22 GB 模型 / 16 GB 内存，**工作集略大于物理内存**。这个比值下 OS 的
LRU 排序仍然是准的（依据真实访问历史），而按层号预取是个粗猜测——多路 MoE 每 token 只
碰 top-k 专家。每读进一页就要挤掉一页，被挤掉的往往是刚用过、马上还要用的。所以预取
在这里不是掩盖延迟，是拿一个更差的预测替换访问历史。

**适用区间**：§1 说 SLIM-ARC 服务「装不进内存」的降级档。本机实测表明，该档位里
还要再分两段——**比值接近 1（抖动区）时预取有害**，正确区间是比值显著大于 1（流式区，
绝大多数页永远不在内存、每次访问都是真缺页）或工作集装得下（预取不挤占热页）。
上游的实测在 80B 级模型上做，属于流式区；**本机没有这种体量的模型，未实测，因此
不否定上游结论**。要验证需要一个「模型 ≫ 内存」的环境，或人为压缩可用内存。

### 6.6 `-ngl 0` 不等于纯 CPU

`llama-model.cpp:1613` 遍历**所有**已注册 device，只要有一个 `mmap_support == false` 就
把 `ml.use_mmap` 清掉。iGPU 的 Vulkan device 正是 `mmap_support = !is_integrated_gpu`
= false，所以**只要 `ggml-vulkan.dll` 在场，`-ngl 0` 也会走 `load_mode = none`**，预取
因此根本装不上。

实测：带 Vulkan DLL 跑 `-ngl 0` 时日志里没有任何 `SLIM-ARC` 输出；改名后立刻出现
`armed: 40 boundaries`。这与 issue 里记的 host buffer bug 是同一行代码的两个后果。

