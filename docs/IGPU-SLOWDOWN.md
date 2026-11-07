# 核显能不能用：两轮实测的结论与修正

> 状态：**现行**
> 更新日期：2026-11-06

## 结论先行（第二轮）

**第一轮结论「核显在任何配置下都不如纯 CPU」是错的**，错在测量口径。第二轮把 Iris Xe 单卡隔离
（`GGML_VK_VISIBLE_DEVICES=0`）后重测，本机 9B/2B 上核显 **全量 offload 是赢的**：

| 指标（9B Q4_K_M，`-t 8`） | 纯 CPU | Iris `-ngl 99` | 比 |
|---|---:|---:|---:|
| decode（tg64） | 6.72 | **7.08** | **+5.4%** |
| prefill（pp256） | 29.20 | **90.80** | **3.1×** |

第二轮的完整数据：2B decode **+9.2%**、prefill **2.5×**；0.5B decode 持平、prefill **7.0×**。
prefill 的交叉点在 **pp≈32–64**（pp32 是唯一输的点，0.93×），再长就 2.6–3.2×。

**第一轮为什么测反了**：① 默认双设备枚举（Iris + 4060）走 `multi_device` 慢路径；
② `llama-cli` 的短 prompt（~50 token）正好落在交叉点左侧；③ `build-cpu` 与 `build-vulkan`
的 CPU 后端不同级（`GGML_CPU_ALL_VARIANTS`）。

**中间值仍然一律更差**，这条不变：`-ngl 0`（加载了后端却一层不上）比纯 CPU 慢 41%，
各模型的中间层数也都低于全量。原因是 host buffer 抢占——但注意 **`571a9d0` 已经包含
`#31538` 的修复**（`src/llama-model-loader.cpp` 的 `(use_mmap || device_type == IGPU)`
分支），所以「未 offload 层不进 host buffer」在上游层面已处理；试图在 `make_cpu_buft_list`
再改一次是**负优化**（实测 2B `-ngl 0` tg 30.53→22.18、4060 `-ngl 12` pp 1226→263，已回滚）。

**35B 无意义**：20.82 GiB > Iris 共享堆 ~15.4 GiB，装不进，部分 offload 会落进 OOM 路径。

**方法学提醒**：`llama-bench` 的 `-ngl` 不要一次传多个值（多值模式与单值能差 3 倍）；
测量时避免高负载，同一配置重复测能差 25%。

---

三个问题的答案（①–③ 各展开一节）：

1. **v2 解释过，而且结论是全的**——分散在 5 份文档里，没有归成「核显为什么慢」这一条，所以看起来像没查清。
2. **中间值与 `-ngl 0` 慢的主因既不是算力也不是「带宽征用」**：是**只要 GPU 后端被加载，未 offload 的层就进后端 host buffer，GPU 仍会接管大算子**，于是那些层失去了 CPU 本地的 L3 + AVX2 路径，改走一次往返拷贝。`-ngl` 越小越严重，`-ngl 0` 最严重。全量 offload 没有「未 offload 的层」，所以不受这条拖累。
3. 本机实测（9B dense，gen 中位）：纯 CPU 构建 **6.9 t/s** → `-ngl 0`（只加载 Vulkan）**4.1**（−41%）→ `-ngl 16` **5.8** → `-ngl 99` **6.7**。这是**第一轮口径**（默认双设备枚举，见 §③ 的注）：`-ngl 0` 与中间值的「输」在第二轮仍然成立；`-ngl 99` 那档第二轮单卡隔离后是 **7.08（+5.4%）**、prefill **3.1×**。第一轮的「任何 `-ngl` 下都不如纯 CPU」已作废。

---

## ① v2 到底有没有解释过：**有**

### 1.1 最直接的一条（Vulkan 加载即污染 CPU 层）

`slimarc/prototype:docs/ANDROID-SURFACE-段接入评估.md:90-99`

```
| Surface 单机 8 层 | 8 | 112 ms | 14 ms | **Vulkan 未禁用时** |
最后一行是关键对照：**Vulkan 后端只要被加载，即使 `-ngl 0`，CPU 侧权重也会进 Vulkan host
buffer**（`make_cpu_buft_list` 把 host buft 插在 CPU buft 之前）。禁掉 `ggml-vulkan.dll`
后 Surface 从 245 ms 降到 112 ms，再算上页缓存热起来，24 层只要 90 ms。
```

同一结论在 [`PROGRESS.md`](PROGRESS.md) §1.1 复述：「这两台的核显都不参与：Surface 的 Vulkan 后端一加载就把 CPU 侧权重塞进 host buffer（head 245 → 112 ms）」。

> ⚠️ 这里的表格行有一处笔误：`:95` 的 112 ms 标成「Vulkan **未**禁用时」，但 `:97-99` 正文说禁用后才从 245 ms 降到 112 ms。**能对上的读法是：未禁用 245 ms、禁用 112 ms。**

### 1.2 弱设备侧的「核显不值得用」并列结论（不算本机 Iris Xe）

`ANDROID-SURFACE-段接入评估.md` §4

```
| 核显 | Iris Plus，**比 CPU 慢 30%** | Adreno 750，**比 CPU 慢 5–25%** | 都不可用 |

**核显都不值得用**。Surface 关掉 Vulkan 后 head 从 245 ms 降到 112 ms；Y700 上
`-ngl 99` 是 29.8 t/s，纯 CPU 37.6 t/s。两者共享同一块 LPDDR，把层搬到另一组执行单元
只增加一次拷贝，不增加带宽。
```

**这句「共享同一块 LPDDR，只增加一次拷贝，不增加带宽」就是用户怀疑的答案**——不是带宽被征用，是根本没有增量带宽。

> ⚠️ 这段引文是**弱设备**（Surface 的 Iris Plus、Y700 的 Adreno 750）的结论，不适用于本机 Iris Xe：
> 第二轮单卡隔离后本机核显全量 offload 是赢的，实测见「结论先行」。两类设备只能各自实测。

### 1.3 核显的算子级实测（唯一一条直接测核显算力的）

`docs/DEPLOYMENT.md:207-221`

```
**① 为什么 `CPU_REPACK` 全部回退**：IQ4_XS 占了权重的大头，而它在**所有后端都只有
MMV（向量乘），没有 MMQ（矩阵乘）**——Vulkan 和 SYCL 一样，OpenCL 与 OpenVINO 干脆不支持。

**② 为什么核显只可能救 tail**：head 的 15.86 GiB 里 8.37 是 IQ4_XS，塞进核显那 15.82 GiB
共享堆会连系统一起挤死；而 tail 全是 Q5_K/Q6_K/Q8_0，四种后端全覆盖，4.88 GiB 能整段常驻。

**关于核显的一个实测**（本机 Iris Xe，ctypes 直调 OpenCL）：
流式读 780 MiB     13.1 ms = 58.2 GB/s   （本机 CPU 6 线程 23.7 GB/s）
lm_head 形状 GEMV  20.5 ms = 23.6 GB/s   （朴素/协作/向量化三版都在 22.6-24.2）

两者差 2.5 倍，说明**低算术强度的 GEMV 只能吃到约四成 DRAM 带宽**，别指望这条路上有数量级。
```

> 注意 `:211` 的 **15.82 GiB 共享堆**正是本机 Iris Xe（`--list-devices` 报 16194 MiB），说明这段就是在这台机器上测的。

### 1.4 关于 offload 粒度的实测表（有，而且是完整的）

| 位置 | 内容 |
|---|---|
| `docs/MEASUREMENTS.md:68-76` | 专家上卡 11 层 **30.45** vs 整层上卡 11 层 **10.11**（3.0×），原因：专家占权重 88% 但每 token 只用 8/256 |
| `docs/MEASUREMENTS.md:58-66` | `-ncmoe` 扫描：0→25.94、8→28.99、10→32.01、12→**17.31**、14→**11.23**、16→**9.75** |
| `docs/DESIGN.md:32-45` | 11 层 30.45 最优，12 层 **15.29 断崖**；「Windows 特有风险：超显存不报错，WDDM 静默把系统内存当显存用，只表现为掉速」 |
| `docs/DEPLOYMENT.md:115-140` | `gpu_layers` 4/5/6/7 → 7.2 / 7.9 / **8.7** / 8.1；「7 层 stage 0 更快（346.1）但 tg 变慢（8.1）」 |
| `docs/DEPLOYMENT.md:142-166` | 32/32 陷阱：stage 0 最快（135.4）但 tg 更慢，原因是**内存**换页，不是显存 |
| `docs/ARCHITECTURE.md:78-90` | `-ngl` 与 `-ncmoe` 两条正交路径的定义 |
| `README.md:150-151` | 「显存超额是静默的。WDDM 允许分配超过专用显存，用系统内存兜住，llama.cpp 照旧打印 `offloaded N/N`」 |

**所以：v2 有「开 N 层后 tok/s 下降」的实测表，而且不止一张。** 缺的只是把它们归因成一条机制说明。

---

## ② 机制分析：哪几条站得住

### ✅ 站得住（有代码或实测直接支撑）

**A. host buffer 抢占——这是「开核显反而慢」的第一因，与算力无关**

代码：`H:\llamacpp-c\src\llama-model.cpp:1155-1215`

```cpp
// CPU: ACCEL -> GPU host -> CPU extra -> CPU
static buft_list_t make_cpu_buft_list(...) {
    ...
    // add a host buffer type
    if (!no_host) {
        for (const auto & dev : devices) {
            ggml_backend_buffer_type_t buft = ggml_backend_dev_host_buffer_type(dev.dev);
            if (buft) { buft_list.emplace_back(dev.dev, buft); break; }   // ← 排在 CPU buft 之前
        }
    }
    ...
    // add the CPU buffer type   ← 最后才轮到
```

`--no-host` 是这条路径的开关：`common/arg.cpp:2500-2503`、`common/common.h:636`。

后果：**只要 `ggml-vulkan.dll` 被加载，`-ngl 0` 的「纯 CPU」就不存在了**，所有层都进 host buffer。

**B. GPU 接管大算子 → 往返拷贝（上游维护者确认，且被实测印证）**

llama.cpp issue `#31538` 里 Vulkan 后端维护者 `0cc4m` 的原话：

> It is intentional to use GPU host buffer there, because **the GPU will still take over for large operations**, and **copying from host buffers is much faster**.

我的实测反过来验证了这句话：关掉 host buffer 并不会变快。
- 9B `-ngl 16`：5.8 → `--no-host` **3.2**（更差，因为 GPU 要从 CPU buffer 跨总线拷）
- 9B `-ngl 0`：4.1 → `--no-host` 5.3（好一点，但到不了纯 CPU 的 6.9）

**两个方向都不如纯 CPU**（第一轮口径；第二轮没有重测 `--no-host`）——因为 host buffer 是「GPU 可直接读」，代价是 CPU 层失去本地性；关掉它，GPU 就得跨总线拷。**没有第三种选择。**

**C. 显存溢出静默（独显场景，v2 已实测）**

`DESIGN.md` §1.2 + `MEASUREMENTS.md` §3.2 + `README.md`（显存超额静默段）：WDDM 不拒绝超额分配，只掉速。12 层专家 33→17，14 层→11。

**但这条在核显上不成立**：核显的「显存」就是系统内存，没有独立显存可以撑爆。核显的问题是 issue `#31538` 描述的单 heap `budgeted heap` 分配——报的是 `ErrorOutOfDeviceMemory`，且错误指向 kv cache。**用户修掉了 OOM，所以这条已经不是他看到的「慢」的原因。** 这与用户的自述一致。

**D. 量化类型缺 MMQ（prefill 慢的一个来源）**

`DEPLOYMENT.md` §6.8：IQ4_XS 在所有后端只有 MMV 没有 MMQ。v2 的 35B 权重 8.37 GiB 是 IQ4_XS（同节）。这解释了 prefill 上不去，但不解释 decode 的损失。

**E. 共享内存总线（decode 的上限）**

本机 DDR5-4800 双通道，理论 76.8 GB/s。9B Q4_K_M（5.68 GB）decode 每 token 要过一遍全部权重：

| 配置 | gen | 推算有效带宽 |
|---|---:|---:|
| 纯 CPU | 6.9 t/s | 41.5 GB/s（54% 理论） |
| 核显全量 | 6.7 | 38.1 |
| 核显部分 | 5.8 | 32.9 |
| 核显 `-ngl 0` | 4.1 | 23.3 |

**核显和 CPU 吃同一根总线，核显能拿到的上限就是 CPU 的上限**，它不可能靠「搬到另一个执行单元」拿到增量带宽。v2 的 `ANDROID-SURFACE:168-169` 说的就是这个。

### ❌ 站不住的（要纠正的说法）

| 说法 | 判定 | 依据 |
|---|---|---|
| **「老核显自己算力太菜」** | **纸面上不成立** | Iris Xe 96EU 理论 ~2.3 TFLOPS fp32（96 EU × 8 lane × 2 × 1.5 GHz）；i9-13900H `-t 8` 约 1.0–1.1 TFLOPS，14 核全开约 1.6。**核显纸面算力更高**，实测却更慢 → 瓶颈不在算力，在带宽与拷贝。*（TFLOPS 是推算，未实测）* |
| **「带宽被核显征用」** | **不成立** | 不是互相抢，是本来就一条总线。核显 offload 不会让 CPU 少掉带宽，只会让被 offload 的层多一次往返。 |
| **「撑爆共享显存导致换页」** | **在核显上不成立** | 核显没有独立显存可撑爆；撑爆的是单 heap 的 budgeted heap（issue `#31538`），且用户已修。修完仍慢。 |

### ⚠️ 推测（未验证）

1. **为什么 2B 上核显 gen 是 +9.2%、9B 是 +5.4%**：2B 权重 1.31 GB，更容易命中 L3/页缓存，decode 没被带宽掐住，核显的算力能露出多一点；9B 5.68 GB 更接近带宽受限。*未验证，未做 L3 命中率测量。*
2. **Vulkan 在 Intel Windows 驱动上的 prefill 效率曲线**：第二轮 pp32 是唯一输的点（0.93×）、pp256 是 3.1×，曲线为什么在 32–64 之间翻过来没有单独做过 perf 对比。*未做 perf 对比，未验证。*
3. **issue 的修复是否只改了分配位置**：用户说修完仍慢。我这边没有复现 OOM（32 GiB 内存 + 16 GiB 共享堆装 5.68 GB 模型不触发），所以无法验证修复本身。*未验证。*

---

## ③ 第一轮实测：有，而且用的就是用户当年那份 Vulkan 构建（口径有误，见顶部修正）

### 为什么能测

`H:\llamacpp-c` 下现成就有四份构建，**没有编译任何东西**：

| 构建 | 后端 | 时间戳 | 用途 |
|---|---|---|---|
| `build-vulkan` | `GGML_VULKAN=ON`，`ggml-vulkan.dll` | **11-02 21:09**，commit `e0f142d` | 就是用户提 issue 那天编译的那份 |
| `build-cpu` | 全 OFF | 同 commit | 真正的纯 CPU 基线 |
| `build-cuda` | `GGML_CUDA=ON` | 10-30 | 4060 对照 |
| `build-split` | **全 OFF**（`GGML_CUDA/VULKAN/SYCL/HIP/OPENCL/OPENVINO/BLAS` 全关） | — | 这条是 v2.1 segments 时代的纯 CPU 构建，**它确实跑不了 GPU** |

**v2 时代的 GPU 怎么跑的**：走另一份资产——`H:\slimarc-proto\shim\bin\ggml-cuda.dll`（54 MB，10-28）+ `cublas64_13.dll` / `cublasLt64_13.dll`，配 `build_cuda_shim.ps1` 编的 shim。`DEPLOYMENT.md` §1 也记了「shim CPU 版约 8.1 MB，CUDA 版约 553 MB」。**所以 v2 的 GPU 实验全是 CUDA（RTX 4060），Vulkan/核显是另一条独立的对照线**，与 `build-split` 无关。

### 数据（`llama-cli`，`-t 8`，`-n 48`，每组 3 次取中位）

> **这一节是第一轮口径，标签不可按字面读。** 当时没有设 `GGML_VK_VISIBLE_DEVICES`，Vulkan 默认
> 枚举 Iris Xe + RTX 4060 两台设备，走 `multi_device` 慢路径；表里「Iris Xe 全量」等标签是当时
> 的写法，实际参与计算的设备不止 Iris。第二轮单卡隔离（`GGML_VK_VISIBLE_DEVICES=0`）的数据见
> 「结论先行」与 §④：`-ngl 99` 由「−3%」变成 **+5.4%**、prefill 由「短板」变成 **3.1×**。
> 下表里仍然有效的是 **`-ngl 0` 与中间值都比纯 CPU 差**这一半（第二轮复现了 −41%），
> 以及「部分 offload 在独显上也要付一半代价」。

**Qwen3.5-9B Q4_K_M（32 层，5.68 GB，dense）**

| 配置 | Prompt t/s | Gen t/s | Gen 相对纯 CPU |
|---|---:|---:|---:|
| **build-cpu，`-ngl 0`（不加载任何 GPU 后端）** | 21.4 | **6.9** | 1.00 |
| build-vulkan，`-ngl 99`（Iris Xe 全量） | 12.6 | 6.7 | −3% |
| build-vulkan，`-ngl 16`（部分） | 15.1 | 5.8 | −16% |
| build-vulkan，`-ngl 0`（**只加载 Vulkan**） | 7.9 | **4.1** | **−41%** |
| build-vulkan，`-ngl 0 --no-host` | 14.5 | 5.3 | −23% |
| build-vulkan，`-ngl 16 --no-host` | 8.5 | 3.2 | −54% |
| build-cuda，RTX 4060 `-ngl 99` | 117.8 | 29.3 | **4.2×** |

**Qwen3.5-2B Q4_K_M（25 层，1.31 GB）**

| 配置 | Prompt t/s | Gen t/s |
|---|---:|---:|
| build-cpu，`-ngl 0` | 105.3 | 24.6 |
| build-vulkan，`-ngl 0` | 74.7 | 26.2 |
| build-vulkan，`-ngl 0 --no-host` | 92.8 | 24.5 |
| build-vulkan，核显 `-ngl 12` | 59.0 | 23.5 |
| build-vulkan，核显 `-ngl 12 --no-host` | 60.9 | 16.0 |
| build-vulkan，核显 `-ngl 99` | 55.2 | 26.5 |
| build-vulkan，4060 `-ngl 12` | 87.7 | 39.3 |
| build-vulkan，4060 `-ngl 12 --no-host` | 120.6 | 39.6 |
| build-vulkan，4060 `-ngl 99` | 199.1 | 74.3 |
| build-cuda，4060 `-ngl 99` | 226.1 | 74.8 |

### 读数要点

1. **9B 上 `-ngl 0` 与中间值都输给纯 CPU**，最差的是 `-ngl 0`（−41%）——这推翻了「中间态最差」的简化说法：**`-ngl` 越接近 0 越差**，因为 host buffer 里的层越多。`-ngl 99` 那一档在第一轮标为 −3%，第二轮单卡隔离后是 **+5.4%**，差别来自双设备枚举（见上面的注）。
2. **部分 offload 在独显上也要付一半代价**：2B 上 4060 `-ngl 12` 39.3 vs `-ngl 99` 74.3（−47%）。「不要用中间值」对独显同样成立，只是绝对值仍是正收益。
3. **prefill 的结论两轮相反**：第一轮 9B 核显 12.6 / 15.1 vs CPU 21.4，看起来是绝对短板；第二轮单卡隔离后 pp256 是 **90.80 vs 29.20（3.1×）**，pp32 才是唯一输的点（0.93×），交叉点在 **pp≈32–64**。两轮的分歧已归因到双设备枚举与「短 prompt 落在交叉点左侧」。
4. **2B 上核显 gen 第一轮 +7%、第二轮 +9.2%**：方向一致。9B 第二轮也是 +5.4%，所以「优势随模型变大立刻消失」这句要收回——**装不进共享堆才是硬边界**（35B 20.82 GiB vs ~15.4 GiB）。
5. **测量陷阱**：Vulkan 首次运行要编译 pipeline，第一次采样能差一个量级（我第一轮 `-ngl 99` 4060 只有 9.5 t/s，热身两轮后 168.4）。与 [`MEASUREMENT-PITFALLS.md`](MEASUREMENT-PITFALLS.md) 记录的「同一个段量出来能差 19 倍」是同类问题。

### 为什么没测 35B

35B-A3B（20.82 GiB）在 CPU 上约 1 t/s，48 token 要 40 秒以上，且核显那 15.82 GiB 共享堆装不下 20.82 GiB 权重——会正好落进 issue `#31538` 的 OOM 路径，而不是用户想问的「慢」。2B/9B 已经把机制分离干净了。

---

## ④ 给部署文档的一句话建议

> **不要用部分 offload，也不要在纯 CPU 机器上留着 GPU 后端。** 只要 `ggml-vulkan` / `ggml-cuda` 被加载，未 offload 的层（含 `-ngl 0`）就会进后端 host buffer，GPU 仍会接管大算子——等于给这些层加一次往返拷贝，实测 9B dense 生成从 6.9 t/s 掉到 4.1（`-ngl 0`）/ 5.8（部分）。**要纯 CPU 就别链接 GPU 后端；要 GPU 就全量 offload，别取中间值**（全量是正收益，数字见「结论先行」）。另有单 heap iGPU 上 host buffer 从显存池分配的问题（llama.cpp `#31538`，上游称「没有简单的解」），`571a9d0` 已含修复。

补充三点：

- **「全 CPU」必须写成「不加载 GPU 后端」**：`-ngl 0` 与前者不是同一件事，实测差 41%。
- **核显的「全 offload」在装得进共享堆的模型上是值得的**（9B/2B 的实测见「结论先行」）。第一轮「9B 上全量也低于纯 CPU」是双设备枚举下的数，已作废。真正的硬边界是**体积**：35B 装不进共享堆。
- **中间态在自己身上也是负的**：`-ngl 0` 与部分 offload 在第二轮仍然输（−41% / −16%），所以「全量或不用」这条规则不变。

**这个项目选纯 CPU 有两条独立理由**：35B（20.82 GiB）装不进 ~15.4 GiB 共享堆；中间态（`-ngl 0` / 部分 offload）的代价结构不对——核显与 CPU 共享同一条内存总线，搬上去不增加带宽，却必然增加一次往返。全量 offload 在装得下的模型上是正收益（见「结论先行」）。
