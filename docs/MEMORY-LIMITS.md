# 内存预算与硬限制

> 状态：**现行**
> 更新日期：2026-11-03
> 适用范围：启动前的内存/显存预算怎么算（`chain_client` 的校准常数与实测依据），以及不遵守时会出什么。硬约束，启动前的两项检查超预算即 `SystemExit`。逐段吞吐见 [`MEASUREMENTS.md`](MEASUREMENTS.md)。

起因是 2026-10-29 凌晨多个进程同时加载整模、整机黑屏两次。下文两条账（内存、显存）与
测试纪律都是那次事故之后定的约束。

---

## 1. 本机内存账

| 项 | 数值 |
| --- | ---: |
| 物理内存 | 31.6 GiB |
| 典型可用（含 GUI、IDE、Harness） | 20–26 GiB |
| swap | 27.5 GiB（**严重使用时会拖死桌面**） |

**关键**：Windows 不会因超内存而拒绝分配，而是分页到 swap。一旦 swap 打满，
整机失去响应——不是报告 OOM，是黑屏。这是本项目的**最高优先级约束**。

---

## 2. 预算模型：一个公式 + 常数

现行判据是 `chain_client.estimate_ram_bytes(stages)`：
只算本地段，每段按"它实际会碰到的字节 × 驻留比例"计。

```
stage_bytes(stage) = 文件大小                              # 预切段
                   = 文件大小 × layers / source_total       # 整模 + layer_range 的段
ram_ratio(stage)   = frac_gpu × GPU_RAM_RATIO + (1 - frac_gpu) × CPU_RESIDENT_RATIO
预算               = psutil.virtual_memory().available × MEM_SAFETY
```

`stage_bytes` 为什么按层数折算：现行形态下每段读的是**同一份整模**，把一份 20.82 GiB 的
文件记到只加载 8 层的段上，会让 35B 看起来单机装不下（历史坑见 §4）。

| 常数 | 值 | 为什么是这个数 |
| --- | ---: | --- |
| `MEM_SAFETY` | 0.75 | 可用内存乘这个系数。留出的 25% 是操作系统拿不回来的页缓存；`SA_MEM_SAFETY` 可覆盖（见 §4 单机两段） |
| `CPU_RESIDENT_RATIO` | 0.30 | 纯 CPU 段的驻留比例，实测 **0.215**；取 0.30 给比测量时更长的上下文留余量 |
| `GPU_RAM_RATIO` | 1.00 | 上卡的层要整份过一遍页缓存才上传，这部分页在被回收前一直算着，所以按文件大小全额计入 |
| `VRAM_PER_BYTE` | 1.13 | 显存增量 / 文件大小。两份装得下的工件都是实测 1.13×（见 §5） |
| `GPU_VRAM_OVERHEAD` | 640 MiB | 每段 context + compute buffer 的固定开销，实测 597 MiB，向上取整 |
| `VRAM_LANDED_SLACK` | 0.85 | 加载后核对允许 15% 偏差：驱动自身的保留量会上下浮动几个百分点，真溢出会差一个量级 |

预算对比的对象是**整模权重按层折算后的字节**，不是各段工件权重之和。后者的口径绑在
裁段工件上（工件与生成器均已退场，[`CUT-TOOL.md`](CUT-TOOL.md)），在整模形态下会把
同一份整模记到每个段上，故不成立。

---

## 3. 检查：内存预算、显存预算、加载后核对

前两条在 `chain_client.main()` 的装配段，超了直接 `SystemExit`，不提供绕过开关。

```
local stages: {n}  weights {x} GiB  expected resident {y} GiB  budget {z} GiB
```

`x` 是本地段文件大小之和，`y` 是 §2 的估计值，`z` 是可用内存 × `MEM_SAFETY`：

```
refusing to start: local stages need an estimated {y} GiB resident but only {z} GiB
is safe (available x 75%). Move a stage to another host or lower the count.
```

显存是另一本账（WDDM 静默溢出，见 §5），先打印预算再按需拒绝：

```
VRAM: need ~{want} GiB  free {free} GiB  of {total} GiB
```

加载之后还有一次核对（`chain_client.vram_after_load`），用实测的显存增量与"真正上卡的权重"
比：`landed < want × VRAM_LANDED_SLACK` 判为溢出。溢出这一项不退出进程，只报名字——
它已经发生了，能做的只是让人知道为什么每一步都慢。

---

## 4. 单机两段：预算比实测紧

同一个 0:32 + 32:40 的切分在本机跑两段时，估计值完全取决上卡多少：`ram_ratio` 对上卡的部分按
1.00、对 CPU 的部分按 0.30，再用 `layers / source_layers` 把整模折算成"这段会碰到的字节"。
`topology-cross-mtp.toml`（head 32 层全上卡）推算 **18.81 GiB**：

    head  21.87 × 32/40 × 1.00 = 17.50
    tail  21.87 ×  8/40 × 0.30 =  1.31
    合计                         18.81

按 20.82 GiB 的 Q4_K_XL 用同一算式推算 **17.91 GiB**（`MEM_SAFETY` 的注释里记作 17.96）
——都超过 0.75 下的 17.71 GiB 预算，所以要多写一个环境变量：

```powershell
$env:SA_MEM_SAFETY = "0.85"; python chain_client.py --topology topology-cross-mtp.toml
```

`topology-local-32-8.toml` 只把 head 的 6 层放上卡，同样的切分推算 **8.86 GiB**：

    head  21.87 × 32/40 × (6/32 + 26/32 × 0.30) = 7.55
    tail  1.31
    合计                                          8.86

0.75 下直接放行。这个差不是估错，是"混合段"的固有性质：14/40 层上卡的段重算下来是 0.545，
而实测只用 0.44×。估计始终在保守一侧——保守的代价是多写一个环境变量，乐观的代价是黑屏。

内存放行不等于显存放行：`gpu_layers=6` 这一档实测仍在溢出（`2.74 GiB resident for 3.28 GiB`，
84% < `VRAM_LANDED_SLACK`），告警会照常打出来。它是四档里最快的一档，判据是吞吐不是告警，
推导见 [`DEPLOYMENT.md`](DEPLOYMENT.md) §6.5。

---

## 5. 显存：另一本账，且不报错

Windows 的 WDDM 允许分配超过专用显存，用系统内存兜住，llama.cpp 照旧打印 `offloaded N/N`，
每一步都在通过 PCIe 拷权重。实测（8188 MiB 的 RTX 4060 Laptop）：

| 工件 | 文件 | 显存增量 | 比值 | 每步 |
| --- | ---: | ---: | ---: | ---: |
| `stage 0-8` | 4640 MiB | 5237 MiB | 1.13× | 5.6 ms |
| `stage 24-32` | 4542 MiB | 5139 MiB | 1.13× | 6.9 ms |
| `stage 8-24`（溢出） | 8559 MiB | 7660 MiB | **0.89×** | 42.1 ms |

前两份是裁段时代的工件（生成器已退场，[`CUT-TOOL.md`](CUT-TOOL.md)），比值仍适用于现行的
整模 + `layer_range` 形态：上卡的层数与字节数没变，只是字节来自同一份文件。

装得下的两份都恰好比文件大 **597 MiB**，所以模型是"文件大小 × 固定系数 + 每段固定开销"。
装不下的那份增量反而**小于**文件大小，这是溢出的特征：权重有一部分留在了系统内存。

`-ngl` / `-ncmoe` 的显存账由 `chain_client.gpu_weight_bytes()` 一份实现出：
`-ncmoe N` 让前 N 层的专家留 CPU，所以只有第 N 层往上才贡献专家权重；非专家张量跟着层走。
预算检查与加载后核对共用这一份——`cpu_moe` 只在此处计入，两处各算一遍会让同一份配置
先被拒、放行之后又被报溢出。

---

## 6. 绑核：给少了会崩 24 倍

**实测数据**（旧形态下单段跑 32 层的对照，2026-10-28）：

| 绑核 | median | 说明 |
| --- | ---: | --- |
| 全部 20 逻辑核 | 106.7 ms | |
| **3 核（`2,3,4`）** | **1692.1 ms** | **24 倍惩罚** |
| 6 核（`2,3,4,5,6,7`） | **70.3 ms** | 最优 |

24 倍惩罚 = 1692.1 / 70.3（推算，相对最优档）。

**根因**：每个 worker 有 5+ 活动线程（`stage_worker` 主线程、`keep_head` 主线程、
llama 的 3 个计算线程），3 个逻辑核装不下。而且 i9-13900H 是 6P+8E 异构，
相邻逻辑核可能是同一物理 P-core 的超线程对（0/1、2/3……），给 `2,3,4`
实际只有 1.5 个物理核。

**规则：每段至少 6 个逻辑核**，不绑核。线程数写在 topology 的 `[[stage]]` 里。

---

## 7. 测试纪律

- 任何测试前先看 `available`，低于 12 GB 就停
- 测试完检查残留进程：
  ```powershell
  Get-CimInstance Win32_Process -Filter "Name='python.exe'" |
    Select-Object ProcessId, @{n='MB';e={[math]::Round($_.WorkingSetSize/1MB,0)}}
  ```
- 残留的 `stage_worker.py` / `keep_head_worker.py` 必须杀掉——它们各自持有整个模型的映射
- **诊断脚本本身也可能泄漏**：写死的 `while` 循环、无界 buffer 解析会吃满内存。
  2026-10-29 凌晨一个拓扑探测脚本死循环吃掉 10.4 GB，差点重演黑屏。
  诊断脚本必须带超时和迭代上限。

---

## 8. 崩溃时怎么恢复

1. **先杀进程，别等系统响应**
   ```powershell
   Get-CimInstance Win32_Process -Filter "Name='python.exe'" |
     Where-Object { $_.CommandLine -match 'slimarc|chain_client' } |
     ForEach-Object { Stop-Process -Id $_.ProcessId -Force }
   ```
   注意：PowerShell 里 `$pid` 是只读变量，循环变量不能叫 `$pid`。

2. 内存通常几秒内回升（实测 16.9 GB → 26.2 GB）

3. swap 不会立即释放，系统会慢慢回收；不必强求

---

## 9. 不用真实加载的检查

`--dry-run` 只读 TOML、文件大小与可用内存，不加载模型：

```powershell
python chain_client.py --dry-run
```

它验证：拓扑合法（层数恰好 40、切点 4 的倍数、端口不重复）、工件存在、§2 的预算与显存预算够用。
**除真实加载之外的检查都不需要模型，优先用静态检查。**

---

## 10. 为什么不能靠"按需释放"绕开内存墙

权重是 mmap 的 file-backed 干净页，内存一紧内核就会回收，下次访问再重读——**这本身就是
"按需释放"的自动实现，而它正是 thrashing 的来源**：工作集一旦大于物理内存，回收与重读就
进入循环，每步都在罚 page fault。实测 Y700 往 15.6 GB 机器上塞 8 层 MoE 尾段，RSS 涨到
8.4–10.3 GB、`MemFree` 只剩 111–135 MB，**但进程活着、HELLO 正常**——余量小不等于会崩，
代价是每一步都慢。

主动 `madvise(DONTNEED)` 只会更糟：释放后第一步必然重读，而 MoE 每步要 touch 约 110 MiB
专家权重（8 专家 × 8 层 × 1.72 MiB），UFS 上就是 55–110 ms/步，比计算本身还慢。

要真正解决只有三条：**工作集小于物理内存**（减层、换更小量化）、换内存更大的机器、换更小
的模型。`SLIM-ARC` 只做预取、不做丢弃，原因也是这个——丢得起的前提是能算准下次要什么。
