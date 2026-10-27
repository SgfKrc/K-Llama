# prima.cpp 对 QLH 三机异构推理的适用性评估

> 状态：现行（调研结论；判决为「只借鉴」，见 §2）
> 更新日期：2026-10-27

适用范围：评估 `OpenCPIL/prima.cpp`（ICLR 2026 论文项目，llama.cpp 的分布式实现）对 QLH 三机四段串行流水线的可迁移性。判据围绕切分调度、通信、I/O 模型、模型/平台兼容四类机制。本文不评估其他并行框架，不给出改造排期。

调研基线：仓库 `OpenCPIL/prima.cpp`，MIT，C++，122 stars / 21 forks / 创建 2026-06-30 / 最后 push 2026-09-06；仓库 `main` 分支 `src/llama.cpp` 为 1,049,326 字节、24,559 行（本次逐个读取源码文本所得，未 clone）。QLH 侧口径为 Qwen3.5-9B（`qwen35`，32 层，GGUF Q4_K_M，5.4 GiB），四段 `[0,8)` / `[8,24)` / `[24,28)` / `[28,32)`，端到端 598 ms/token 稳态、27.9 s 冷启动 prefill 固定开销 `[实测数据]`。

---

## 1. 一句话结论

prima.cpp 解决的是「单机装不下、且磁盘可以当第二级存储」的容量问题；QLH 的问题是「模型装得下，但每 token 的边界往返太贵、每请求的权重加载太贵」。两者共用 pipeline parallel 这一个范式，但在**权重驻留方式**（流式换页 vs 常驻）与**收益来源**（省内存 vs 省时延）上方向相反，因此整体不采用，机制单点借鉴。

---

## 2. 判决

**只借鉴。**

支撑理由，按决定性排序：

1. **模型架构不覆盖，这是硬阻塞。** `src/llama.cpp:17003` 在图构建入口断言 `model.arch == LLM_ARCH_LLAMA || model.arch == LLM_ARCH_QWEN2`；同文件的 `llm_arch` 枚举里只有 `LLM_ARCH_QWEN` / `LLM_ARCH_QWEN2` / `LLM_ARCH_QWEN2MOE`，没有 `QWEN3` 与 `QWEN35` 的任何枚举值。Qwen3.5-9B 的 GGUF 头为 `general.architecture=qwen35`，落到该枚举只能是 `LLM_ARCH_UNKNOWN`，随后在图构建处命中断言。README 的「Supported Models」也只列 Llama、Qwen 2.5/QwQ、DeepSeek R1 三族 `[README]`。QLH 现有 `Qwen3.5-9B-Q4_K_M.gguf` 无法被该引擎加载，裁层段同样不能。
2. **收益区与 QLH 不重叠。** prima.cpp 的内存压力来自「权重不全驻留、每 cycle 从磁盘换页」，而它的时间收益全部建立在「本来要读盘、现在把读盘藏进计算与通信」之上。QLH 的 `[8,24)` 段实测 2894 MB / 104.08 ms = 27.8 GB/s `[实测数据]`，这一速率对应内存纯读，段工件已经常驻页缓存；磁盘一次读取的上限是 Table 1 里 0.72–3.17 GB/s 量级 `[README]`，即使全速也只够把该段拖慢一个数量级。prima.cpp 的预取机制在数据已在 RAM 时没有可重叠的对象。
3. **它不消除 QLH 的链路开销。** prima.cpp 的 decode 循环为「recv 上一段 hidden → 计算本段 → 发送」严格交替（`src/llama.cpp:18503-18576`），同一 token 内不并发、不做 micro-batch，也不做 tensor parallel。它的 ring 是「一个 token 内对同一批设备跑 k 个 cycle」（`src/llama.cpp:3854` 的 `map_layer_to_local_id` 与 `L = k * W`，`common/common.cpp:1257`），用于让每台设备只驻留 `w_m` 层，属于空间换时间，不是流水线并行度。因此 QLH 的 598 ms 端到端中约 330 ms 的链路差额，prima.cpp 没有对应机制可迁移。
4. **平台缺 Windows，而 master 是 Windows。** README 的 FAQ 第 5 条写明 Windows「Not yet」`[README]`；`common/common.cpp:984`、`:1081`、`:1219` 三处在设备分类前断言 `GGML_ASSERT(!is_windows && "Windows is not tested yet")`；`src/llama.cpp:18153` 与 `:18207` 直接调用 `sysconf(_SC_PAGESIZE)` 与 `posix_madvise`，该函数没有任何 `_WIN32` 保护分支。MSVC 下这两行无法通过编译，运行期还有断言拦截。

**最关键的两条证据**：`src/llama.cpp:17003` 的架构断言（吃不了 qwen35）与 `common/common.cpp:984` 的 Windows 断言（master 装不上）。任一条单独成立即足以否掉整体移植；两条同时成立时，源码级移植的工作量等价于给 prima.cpp 重写一个 `qwen35` 图构建器加一套 Windows I/O 后端，超出「引入一个调度层」的范围。

**它的收益区在哪。** 同一张 Table 2 内的对照说明该收益区的边界：Llama 3-8B 上 llama.cpp 与 prima.cpp 都是 15 ms `[README]`，Llama 1-30B 是 202 → 72 ms，Llama 3-70B 是 10120 → 674 ms。加速比随「单机装不下」的程度单调上升，在单机装得下的区间为 1.0×。Qwen3.5-9B 全模 5.4 GiB，QLH 的 master 有 31.6 GB RAM，属于加速比 1.0× 的那一端。README 自己也记录了这一退化：旧版「每设备至少一层」导致 Llama 3-8B 按 1:1:29:1 切分而慢于 llama.cpp，现行版本在弱设备剔除后变成 32:0:0:0「prima.cpp becomes llama.cpp when serving small models」`[README]`。若把 QLH 的四段交给它，按同一规则大概率会被判为「弱设备全剔除、master 单机跑」，即 QLH 存在的意义被它的自动设备选择直接消解。

---

## 3. 表 1：机制维度对照

| 机制维度 | prima.cpp 做法 | QLH 现状 | 差距与可迁移性 |
|---|---|---|---|
| 并行范式 | pipeline parallel，层序切分；同一 token 内按 `L = k * W` 跑 k 个 cycle，形成环形重复（`common/common.cpp:1257`、`src/llama.cpp:3854`） | 4 段串行 pipeline，静态区间 `[0,8)`/`[8,24)`/`[24,28)`/`[28,32)` | 同范式。差异在 k：prima 用 k>1 压缩每设备驻留层数，QLH 用 k=1。可迁移性低——QLH 四段本来就装得下，k>1 只会增加往返次数 |
| 层分配依据 | HiGHS 整数规划（`common/common.cpp:1260-1424`），目标 `min k·Σ(α_m·w_m + β_m·n_m) + κ + k·Σc_m`；α_m 由 12 种量化类型逐项折算的 CPU FLOPS + KV 拷贝延时构成（`:906-923`），β_m 为 GPU 相对 CPU 的增量（`:966`），κ 含主设备输入/输出层经磁盘的时延 | 按容量预分配，静态 | 它是「时延最小化」而非「容量最大化」的目标函数，与 QLH 的切点求解（`src/relay_cut_objective.py` 的固定开销 + 每层成本两参数拟合）同构，但多了磁盘折价项与显存约束项。模型可迁移，代码不必 |
| 重切时机 | 启动期一次求解；设备掉线或被剔除后重建环拓扑，把弱设备降级为 forwarder 防 socket 阻塞（`src/llama.cpp:20684-20829`） | 静态，运行中不重切 | 两者都是「一次分配 + 拓扑修复」。prima 多一个自动剔除弱设备的判据（`w_m == 1 && n_m == 0`，`common/common.cpp:1451`），QLH 缺这一条 |
| 权重驻留 | mmap 惰性加载，每 cycle 只驻留 `w_m` 层，OS 按需回收页缓存；宣称内存压力 <10%（README）/<6%（论文摘要） | 段工件常驻内存，`[8,24)` 段 2046.9 MiB、Surface 段 1881.0 MiB（含 lm_head 795.7 MiB）`[实测数据]` | 方向相反。它的低内存压力以每 token 重复读盘为代价，QLH 已用常驻换掉读盘 |
| I/O 预取 | `manage_graph_tensors`（`src/llama.cpp:18152-18218`）：取图叶子张量中名字含 `weight`、非 CUDA buffer 的张量，按页对齐、排序、归并成段，逐段 `posix_madvise(MADV_WILLNEED)`；`--force` 时按每 4 页触摸一字节强制落页。调用点在 `send` 之后、下一子图 compute 之前（`:18567-18575`） | 无段内预取；27.9 s 冷启动在请求路径上同步发生 `[实测数据]` | **唯一值得迁移的机制**，见 §7 |
| 通信传输层 | ZeroMQ push/pull，`tcp://*:(9000+rank)`，signal 端口 `(10000+rank)`（`src/llama.cpp:20467`、`:20492-20527`） | 自建帧协议（`src/relay_transport.py` 的 `_HEADER` 结构 + 定长 header） | 功能等价。ZMTP 提供多段 message 与内置重连，QLH 的帧头已有版本/序号/长度字段 |
| 通信编码 | 原始 `float` 二进制，`sub_gf_out` 的 `ne` 数组加 `ne[0]*ne[1]*sizeof(float)` 字节，`pos` 另发一条 message（`src/llama.cpp:18031-18052`）；meta 走独立 multipart（`:17870-17918`） | `f32 → base64 → JSON → sha256`；已实现 `binary_hidden` 二进制体通道（`src/task_worker_protocol.py:1659`，票 21a） | prima 的格式 = QLH 已实现的二进制体通道。无新增可迁移项 |
| 通信与计算重叠 | 无。每子图 recv → compute → `ggml_backend_tensor_get_async` → send 严格串行；仅 GPU→CPU 拷贝用 async + `ggml_backend_sched_synchronize`（`src/llama.cpp:18555-18556`） | 同样同步往返 | 双方同一水平。prima 不提供 QLH 缺少的流水并行度 |
| 异构引擎 | 单一引擎（llama.cpp 内核），异构性体现在「同引擎内的 CPU/GPU 分工 + 设备间层分配」 | 混合：`[0,8)` 为 PyTorch BF16/CUDA，其余三段为 llama.cpp CPU | prima 无法承载 PyTorch 段。README FAQ 第 5/6 条明确 GPU 后端只有 CUDA（Vulkan/AMD 在 roadmap）`[README]`，且后端选择在 ggml 层，与 PyTorch 无交界 |
| 模型架构覆盖 | `llama_build_graph` 断言仅 LLAMA / QWEN2（`src/llama.cpp:17003`）；枚举无 QWEN3/QWEN35 | qwen35（hybrid：Gated DeltaNet + 每 4 层 1 层 full attention） | 决定性阻塞 |
| 平台 | Linux、macOS、Android/HarmonyOS（经 Termux）`[README][docs/android.md]`；Windows 未支持 | master Windows + y700（Termux Linux，原生 Android 另一路径）+ Surface | Windows 侧硬阻塞；y700 侧可满足 |

---

## 4. 切分与调度：HiGHS 目标函数的可读细节

`assign_layers_to_device`（`common/common.cpp:860`）把每台设备归入 M1–M4 四类，判据是「内存是否装得下本设备的层窗口 + KV + compute buffer」与「磁盘是否慢于 100 MB/s」：

- M1：macOS 无 Metal 且内存不足；M2：macOS 有 Metal 且显存不足；M3：Linux/Android 且内存不足；M4：内存充足或磁盘过慢（`common/common.cpp:1090-1102`）。
- M1/M3 的每层时延按 `α_m + b'/(disk_speed[m]·1e9)·1000` 计（`:1193`、`:1198`），M4 只按 `α_m` 计（`:1201`）。这就是「装得下的设备不计 I/O 摊销、装不下的设备计入」的双口径。
- `disk_speed` 取 `read_seq_bw`（Linux）或 `read_rnd_bw`（其他）（`:1035-1039`），两者由 `fio` 实测得出（`common/profiler.h` 的 `disk_props`、`common/profiler.cpp:927` 的 `external_fio_impl`，测试总量 500 MiB、顺序块 100 MiB、随机块 4096 B）。
- 约束为 `Σw_m = W`、`n_m ≤ w_m`、每设备 RAM 上限、每设备显存上限，全部变量声明为整数（`:1380`）。

对 QLH 的直接含义：这套模型把「设备是否装得下」写成显式分支，而 QLH 的切点求解目前是连续两参数拟合，没有「装得下 / 装不下」的跳变项。M4 分支恰好是 QLH 四段所处的类别，`α_m` 只剩 CPU/GPU 的 FLOPS 折算项与 KV 拷贝项。

值得记录的一处旁证：`common/common.cpp:921` 与 `:947`、`:964` 三行把 `t_read_ram_*` 整行注释掉，注释原文为 `// t_read_ram_cpu = b_prime / (dev.memory.cpu_read_ram_bw * 1e9) * 1000;`。这说明在 prima.cpp 的模型里**内存读带宽被当作可忽略项**——在有磁盘流式的场景下它确实被读盘掩盖。QLH 的 master 段瓶颈正是这一项（27.8 GB/s 接近纯读上限），因此 primacpp 的成本模型在 QLH 的 M4 分支上会低估主瓶颈。这也是它没有针对单机内带宽做任何优化的原因：该文件内没有对 CPU 读带宽做压缩、重排或分块的设计。

---

## 5. 通信机制：数字与结构

- 单跳结构：ring 覆盖层，物理上全互联（README 明确「This ring communication is a communication overlay, not the physical topology」）`[README]`。
- 帧内容：每跨一个子图边界发一次 `sub_gf_out`，字节数 = `ne[0] * ne[1] * sizeof(float)`（`src/llama.cpp:18038`）。QLH 的 hidden 为 16 KiB/step `[实测数据]`，即 4096 个 float，两者量级一致。
- 分块 / 双缓冲：**未找到**。`llama_send_tensors` 一次 `zmq::send_multipart` 发完整 hidden，无分片、无 ring buffer、无跨 token 预投递。
- 单跳延迟公开数字：**未找到**。README 只给端到端 TPOT（Table 2/3），论文摘要给 70B 的 674 ms/token 与 32B 投机解码 26 tok/s `[论文]`，均未拆分通信与计算占比。
- 与 QLH 的 330 ms 差额对照：以 16 KiB/step 计，千兆链路传输时间在 1 ms 以内，因此 330 ms / 3 个边界 ≈ 110 ms/边界不可能由数据量解释，只能由「同步往返 + 序列化 + 摘要校验 + 调度」解释。prima.cpp 的 wire 格式能砍掉后半段的一部分，但它的 recv→compute→send 交替结构与 QLH 相同，往返次数不变。**结论：prima.cpp 不消除 QLH 的链路开销。**

---

## 6. I/O 模型：mmap 与预取的实际形态

三个层次：

1. **映射**。`llama_mmap` 在 POSIX 下用 `mmap(PROT_READ, MAP_SHARED)`，并对整个文件 `posix_fadvise(POSIX_FADV_SEQUENTIAL)`；`prefetch` 为真时加 `MAP_POPULATE` 并对首段 `posix_madvise(POSIX_MADV_WILLNEED)`（`src/llama.cpp:1950-1985`）。Windows 分支用 `CreateFileMappingA` + `MapViewOfFile` 并在 `prefetch > 0` 时动态加载 `PrefetchVirtualMemory`（`:2063-2105`）——映射层本身跨平台，问题出在上层的 advise 调用。
2. **释放**。`unmap_fragment` 支持按页区间 `munmap`，维护已映射片段表（`:2011-2060`）。这是「prefetch-release 冲突」的解法载体：释放与预取都按页对齐区间进行，避免刚 advise 的页被同一轮 munmap 掉。
3. **发射点**。`manage_graph_tensors` 在每段 `send` 完成之后，对**下一个子图**的所有权重叶子张量做归并 advise（`:18567-18575`）。这个位置选择让 advise 落在「本段计算已结束、下一段数据尚未到达」的窗口里，与通信等待重叠。

对 QLH 的可迁移判断：第 1、2 层（mmap + 按页释放）在段已常驻时无收益；第 3 层（发射点）有收益，但受益对象是冷启动 prefill 的 27.9 s，不是 decode 的 330 ms。见 §7。

---

## 7. 表 2：QLH 四个痛点的逐条判定

| 痛点 | prima.cpp 是否解决 | 依据 |
|---|---|---|
| ① 链路开销约 330 ms/token（各段实测和 266 ms，端到端 598 ms） | **否** | 其 decode 循环为 recv→compute→send 严格交替（`src/llama.cpp:18503-18576`），无 micro-batch、无跨边界并发、无 tensor parallel；ring 的 k 个 cycle 是同一 token 内的串行重复（`L = k * W`，`common/common.cpp:1257`）。wire 格式（原始 float + multipart，`:18031-18052`）优于 QLH 的 `f32→base64→JSON→sha256`，但 QLH 已有等价的 `binary_hidden` 二进制体通道（`src/task_worker_protocol.py:1659`） |
| ② 27.9 s prefill 固定开销（每请求重复加载权重，`loaded=False`） | **部分解决，机制可独立复刻** | prima.cpp 的整个生命周期模型是「进程常驻、权重 mmap 常驻、只换页」，请求路径不含加载；`manage_graph_tensors` 的 advise 发射点（`src/llama.cpp:18567`）把预热放进计算/通信窗口。这是四个痛点里唯一有对症机制的一项 |
| ③ decode 纯读带宽受限、加线程无效（4/8/12/16 = 109.75/118.39/128.32/146.42 ms） | **否** | 成本模型把内存读带宽整项注释掉（`common/common.cpp:921`、`:947`、`:964`），说明该系统的瓶颈假设是磁盘而非 RAM；源码中无针对单机 RAM 带宽的压缩、重排或分块设计。prima.cpp 的优化面只覆盖「跨机」与「磁盘」两处 |
| ④ 层段静态切分、按容量预分配 | **是，但只做一次** | HiGHS 整数规划在启动期求解一次（`common/common.cpp:1260-1424`），运行期只有掉线后的环拓扑重建（`src/llama.cpp:20684`），无按负载重切。模型可借鉴，运行时不重切这一点与 QLH 现状相同 |

---

## 8. 表 3：移植评估

| 评估项 | 结论 | 阻塞项 |
|---|---|---|
| 只取调度层、保留现有推理内核 | **不可行** | 调度与图构建耦合在同一文件：层归属判定（`this_layer_is_mine`、`map_layer_to_local_id`，`src/llama.cpp:3838`、`:3854`）被 KV 分配、图构建、权重加载三处直接调用；抽出调度层等价于重写 `src/llama.cpp` 的图切分 |
| 直接吃现有 `Qwen3.5-9B-Q4_K_M.gguf` 与裁层段 | **不可行** | `llm_arch` 枚举无 `QWEN35`；`src/llama.cpp:17003` 断言仅 LLAMA/QWEN2；README 支持列表无 Qwen3 系列 |
| 异构引擎混合（PyTorch 段 + llama.cpp 段） | **不可行** | primacpp 的异构性是同一 ggml 图内的 CPU/GPU 分工与设备间层分配，后端只有 CUDA（`[README]` FAQ 第 6 条），与 PyTorch 运行时不共享图、张量布局或 KV 页 |
| Android（y700，Termux Linux） | **可行** | README 与 `docs/README` 均说明 Termux 路径可用 `[README]`；`common/profiler.cpp` 的 `device_os()` 把 Android 分支注释掉、统一返回 `"Linux"`，因此 Android 节点按 Linux 分支处理。原生 Android 支持：**未找到**相关构建目标 |
| Windows（master） | **不可行** | FAQ 第 5 条「Not yet」；`common/common.cpp:984`、`:1081`、`:1219` 断言 `!is_windows`；`src/llama.cpp:18153`、`:18207` 的 `sysconf`/`posix_madvise` 无 `_WIN32` 保护 |
| 依赖与构建 | **偏重** | `gcc ≥ 9.4`、`cmake ≥ 3.16`、`fio ≥ 3.16`（测盘）、`zmq ≥ 4.3.2`（跨设备通信）、`HiGHS ≥ 1.9.0`（rank 0 必需，`make USE_HIGHS=1`）、CUDA 可选 `[README]`。HiGHS 与 fio 是 QLH 现有工具链没有的 |
| 代码规模 | **中** | `src/llama.cpp` 24,559 行 / 1,049,326 字节；`common/common.cpp` 125,205 字节；`common/profiler.cpp` 100,454 字节。整体沿用 llama.cpp 的 ggml 子模块与 examples 树 |
| issue / PR 活跃度 | **未核实** | 本次调研的 GitHub API 请求被速率限制（`API rate limit exceeded`），issues 页面的 HTML 抓取被截断。「2 个 open issue」只作为既有已知信息，不单独作为活跃度判据 |
| ICLR 2026 核心贡献 | **已核实** | 摘要一句话：提出 pipelined-ring parallelism（PRP）把磁盘 I/O 与计算、通信重叠，并解决 mmap offloading 中的 prefetch-release 冲突；配套 Halda 异构感知调度器在 RAM/VRAM 约束下协同优化每设备 CPU/GPU 负载与设备选择。四设备上 70B 达 674 ms/token TPOT、内存压力 <6%，32B 加投机解码达 26 tok/s，对比 llama.cpp / exo / dllama 的 TPOT 低 5–17× `[论文]` |

---

## 9. 若只借鉴一件事

**把段权重的页预热发射点，从「请求入口」前移到「上一段 compute 完成、下一段 hidden 尚未到达」的空窗，并以实际要执行的张量地址区间为预热单位。**

prima.cpp 的做法（`src/llama.cpp:18152-18218`）可直接照搬思路而不引入任何代码：

1. 遍历**下一子图**的叶子张量，按名字筛出权重张量（`strstr(name, "weight")`），跳过已驻留在显存/常驻缓冲的类型。
2. 取每个张量的 `data` 地址与 `ggml_nbytes`，按页对齐后排序、归并成连续区间（`merged_segments`）。
3. 对每个区间调一次 `posix_madvise(…, POSIX_MADV_WILLNEED)`；需要更强保证时按每 4 页触摸一字节强制落页（`prefetch_dense = 4`）。
4. 发射点：在发送完本段 hidden 之后、阻塞等待下一段之前，对下一段做上述归并 advise。

对 QLH 的映射：`[8,24)` 与 Surface 段各约 2 GiB，冷启动时 27.9 s 全部落在请求关键路径上 `[实测数据]`。把预热改为「段就绪后不阻塞、请求到达时按下一 token 要用的层区间增量落页」，可把这段固定开销从串行等待变成与 decode 重叠的后台行为。该机制不需要改 QLH 的帧协议、不需要改引擎、不需要引入 C++ 依赖，实现面是段执行器进程内的一个预取线程加一组页区间表。

需要同时交代的边界：这条改动治的是痛点 ②，对痛点 ①（330 ms 链路）与痛点 ③（单机读带宽）没有作用。痛点的量级排序上 ② 是一次性 27.9 s、① 是每 token 330 ms，二者不可互相替代，需分别立项。

**不建议借鉴的一项**：`L = k * W` 的 cycle 分片。它成立的唯一前提是「每设备驻留层数小于承担层数」，即内存装不下。QLH 四段工件分别为 2046.9 MiB / 1881.0 MiB `[实测数据]`，各自装得下，引入 k>1 只会把每 token 的边界往返乘以 k。

---

## 10. 未找到与未核实

- prima.cpp 的**单跳通信延迟公开数字**：README、论文摘要、项目主页均只给端到端 TPOT，未拆分通信/计算占比。**未找到**。
- prima.cpp 的 **issue / PR 数量与回复时延**：GitHub API 限流、issues 页面抓取截断，本次未取得。**未核实**。
- prima.cpp 的**内存压力 <10% 的分母定义**：README 与论文摘要分别给 <10% 与 <6%，两者口径差异未在可见文本中说明（可能是「占物理内存比例」与「占模型体积比例」的差别）。**未找到**。
- **Qwen3.5/qwen35 在 prima.cpp 上运行的可能性**：源码枚举与断言、README 支持列表、Table 2/3 的模型清单三处一致指向不支持。若上游未来补齐 `QWEN35` 图构建器，此项结论需重估。**当前为不支持**。
- **Windows 支持的时间点**：FAQ 只有「on the roadmap」，无版本或日期。**未找到**。
- prima.cpp 是否提供**跨 token 的通信预投递**（把下一个 token 的输入提前送到下游）：源码 `llama_decode_internal` 的循环体内未见，`src/llama.cpp:18503-18576` 为严格交替。**按源码判断为无**，未在其文档中找到相反陈述。
- prima.cpp 的**原生 Android 构建目标**：`docs/android.md` 是 llama.cpp 上游文档（Termux 与 NDK 两条路径），仓库内未见 Android 应用工程或 Gradle 目标。**未找到**。
