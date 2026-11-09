# 段级 offload：`-ngl` 在分段下的语义、三种显存花法与显存预算

> 状态：**现行**
> 更新日期：2026-11-09
> 代码：`src/llama-model.cpp` · `tools/split-stage/main.cpp` · `tools/server/server-context.cpp` · `src/qlh-{vram-budget,split-defaults}.h`
> 硬件：i9-13900H + RTX 4060 Laptop（8188 MiB，实测空闲 7.9 GiB）+ 31.6 GB 内存
> 模型：`Qwen3.6-35B-A3B-UD-Q4_K_M`（21.11 GiB，40 trunk + 1 NextN，MoE 256 / 激活 8）

## 结论先行

1. **`-ngl N` 现在是「本段最后 N 块」**，整模路径零改动（`§2`）。
2. **tail 段全量上卡是碾压式正收益**：段内 pp `64.4 → 1018.5`、tg `35.7 → 1754.2 t/s`
   （4.2 GiB 全进显存，`§3`）。
3. **head 段在独显上的负收益来自 `op_offload` 默认值，不是设备**：关掉后 `-ngl 8` 从
   12.47 升到 21.00 t/s（+68%），`split-stage` 与 `llama-server` 现在都默认关（`§4`）。
4. **MoE 上值钱的是 `-ncmoe` + 专家缓存，不是层 offload**：`-ngl 32 --n-cpu-moe 32
   --moe-cache-mib 1024` 用 ~2.2 GiB 覆盖全部 32 层，tg **+95%**、pp **+205%**（`§5`）。
5. **判据要用部署形态**（head 独占本机、tail 远程）：`-ngl 8` +19.6%、`-ncmoe`+缓存 +40.9%。
   同机构造（`§7.1`）测不出 head 的收益，方向甚至会反。
6. **显存预算超了不报错**：`§10` 的预检/后检把 WDDM 静默溢出变成一行日志，部署形态零误报。

## 1. 改动清单

| 文件:行 | 改动 |
|---|---|
| `src/llama-model.cpp` `load_tensors` | range load 的 `i_gpu_start` 基准从 `n_layer_all + 1` 改为 `n_layer_all`；`offload_output` 单独判定；device 分配比例钳到 `act-1`（修 `-ngl == 本段块数` 时 `devices.at()` 越界）；`offloaded` 日志打印实际块区间 |
| `tools/split-stage/main.cpp` | `-ngl N\|all`（**默认 0**，不是 llama.cpp 的 `-1`）；`--no-op-offload`（默认开）；`--moe-cache-mib` / `--moe-cache-cpu-share`；`-nr`；启动时打印 offload 计划 |
| `src/qlh-vram-budget.h` · `src/qlh-split-defaults.h` | 新增，header-only；显存预检/后检（`§10`）、`op_offload` 的段默认（`§4`） |
| `tools/server/server-context.cpp` | 显存两处接入 + split 角色下 `op_offload` 默认关 + `no_op_offload_explicit` |

测量装置：`.qlh_tmp/seg_probe.cpp`（段级；正常退出让 `llama_moe_cache` 析构打印命中率）、
`.qlh_tmp/seg_bench.sh`、`.qlh_tmp/vram_accept.sh`。

## 2. 语义：`-ngl` 现在数的是「本段」

上游的基准含输出槽：

```cpp
const int i_gpu_start = std::max(n_layer_all + 1 - n_gpu_layers, 0);  // 上游
```

`+1` 是 logits 层——对整模合理（全部 block 之后才可达），但对**段**不合理：head/relay 没有
输出张量，白占一个槽会让 `-ngl 8` 只搬 7 块。改为按 `ml.layer_range_active()` 分叉：

```
整模      : i_gpu_start = n_layer_all + 1 - N    （逐位不变）
range 载入: i_gpu_start = n_layer_all     - N    （N 是本段块数）
```

输出槽单独判：段内只在 `N >= n_layer_all` 时带上，`-ngl 3` 不会因为 logits 是最后一个槽就把它
拖上卡，`-ngl all` 仍意味着「整段含输出层」。**tail 的本地块数含 NextN**（`32:40` 是 9 块），
`-ngl 8` 只搬 33..40，整段上卡要用 `-ngl 9` 或 `all`。

## 3. tail 段（32:40）实测

段级 probe，`-c 512`，4.2 GiB 权重：

| 配置 | pp | tg |
|---|---:|---:|
| 纯 CPU | 64.4 | 35.7 |
| `-ngl all` | **1018.5**（15.8×） | **1754.2**（49×） |

端到端（head 在 CPU、tail 上卡）只有 +15%：head 的 32 层在 CPU 上是主导项（`§7`）。

## 4. head 段（0:32）实测：负收益是怎么变成正收益的

### 4.1 先遇到负收益

| 构建 / 配置 | `-ngl 0` | `-ngl 8` |
|---|---:|---:|
| 纯 CPU 构建 | 16.37 | — |
| CUDA，`op_offload` 开（上游默认） | 11.25（−31%） | 12.47（−24%） |
| CUDA，`--no-op-offload` | 15.69 | **21.00（+28%）** |

### 4.2 根因：`op_offload`

CUDA 构建默认 `op_offload = true`，让 GPU 接管 `CUDA_Host` 里 host 权重的算子——decode 每
token 跨设备往返，把 CPU 层拖慢 31%；`--no-host` 是错误方向（跨 PCIe 读 pageable 内存）。

**这是「独显上段 offload 负收益」的完整归因：一个默认值，不是设备限制。**
`--no-op-offload` 后 CPU 层回到自己的 kernel（权重仍在 pinned buffer），GPU 只算它自己那 8 块。

**`llama-server` 直到 v2.4 才跟上**：`grep -rn "op_offload" tools/server/` 原本为空，所以每条链
的 head 都开着它——同一个 head 在 probe 上 +36%，经 server 只剩 −10%。修法见 `§1` 的两个新头
文件（`QLH_SPLIT_DEFAULT_NO_OP_OFFLOAD` + `no_op_offload_explicit`，后者防止段默认静默覆盖显式
的 `--op-offload`）。

### 4.3 排除项

`build-cuda` 原配置选中 `ggml-cpu-haswell.dll`（i9-13900H 支持 AVX-VNNI）；重配为
`GGML_NATIVE=ON + GGML_BACKEND_DL=OFF` 后 tg 几乎不变（11.67 vs 11.25），CPU 变体不是主因。

## 5. 三种显存花法

MoE 层的构成决定了这个差别：**专家占 95.8%，非专家（注意力、norm、路由）只占 4.2%**
（[`MEASUREMENTS.md`](MEASUREMENTS.md)：MoE 层 856.36 MiB 中专家 820.18 MiB）。

| 花法 | 上卡的张量 | 覆盖层数 | 显存 | tg |
|---|---|---|---:|---:|
| 整层 `-ngl N` | 层内全部张量（含专家） | N 层 | 0.53 GiB/层 | `-ngl 8` ≈ 4.2 GiB，+28% |
| `-ncmoe` + `-ngl` | 层的非专家部分 | **全部 32 层** | ~1.2 GiB | 18.05 |
| 再加 `--moe-cache-mib 1024` | 上面 + 热专家 | 全部 32 层 | +1.0 GiB | **31.99（+95%）** |

**`-ngl` 与 `-ncmoe` 不是互斥选项，是分工**：`-ngl N` 决定哪些层分配到 GPU（缓存的条件 2），
`-ncmoe M` 决定哪些层的专家留在 host（条件 1），`moe-cache` 在同时满足两者的层里按热度缓存。

`src/llama-moe-cache.cpp` 的启用循环要求两条同时成立：该层**全部**专家在 host
（`llama_moe_cache_is_host_weight`），**且**该层被分配到某个 GPU（`model.dev_layer(il)` 在
`devices` 里）；缺一条即打印 `no layer has all of its experts in host memory, MoE cache is
disabled`。所以缺一个参数就退化：缺 `-ncmoe` → 专家跟着层上卡、装不下；缺 `-ngl` → 层全在
CPU、条件 2 不成立；缺 `--moe-cache-mib` → 专家每次按需上传，那 61% 省不掉。

**MoE 上「整层 offload」是低效花法**：95.8% 的显存花在专家上；同样显存在 `-ncmoe` 形态下覆盖
4 倍层数，专家交给缓存（82% 命中）+ 按需上传。

### 5.1 命中率扫描

head `-ngl 32 --n-cpu-moe 32 --no-op-offload`，`-c 1024 -b 256 --pp 128 -n 24`：

| `--moe-cache-mib` | pp | tg | 命中率 | 上传 |
|---:|---:|---:|---:|---:|
| 0 | 75.68 | 18.05 | — | — |
| 512 | 77.69 | 22.49 | 66.6% | 3723 MiB |
| **1024** | 78.05 | **31.99** | **82.2%** | 1988 MiB |
| 2048 | 78.43 | 32.57 | 84.6% | 1718 MiB |
| 2048 + `--moe-cache-cpu-share 0.5` | 77.54 | 17.94 | 84.6% | 1718 MiB |

- **1024 MiB 就够**：只比 2048 少 2%，省 1 GiB。相对纯 CPU（16.37）tg **+99%**、pp **+205%**。
- 官方 `S = 1/(1−0.613h)`，h=0.846 → 2.08×；实测 32.57/18.05 = **1.80×**（差额来自非专家路径）。
- `--moe-cache-cpu-share 0.5` **负收益**（17.94），保持 0。
- 命中率只在进程**正常退出**时打印（`~impl()` 的 `log_stats`），`taskkill /F` 拿不到。

## 6. repack 与 offload 的交互

| 形态 | repack on | repack off | 结论 |
|---|---:|---:|---|
| `-ngl 8 --no-op-offload` | 22.25 | 21.66 | **+2.7%** |
| `-ngl 32 -ncmoe 32 --no-op`（无缓存） | 18.05 | 18.27 | −1.2% |
| `-ngl 32 -ncmoe 32 --no-op` + cache 1024 | 31.99 | 32.98 | **−3.0%** |

v2 的「offload 混合时 repack 负收益」只在**专家在 CPU + 缓存上传**这一形态复现；简单层
offload 下是 +2.7%（与 v2 相反，大概是 Q4_K_M 与当年 IQ4_XS 的差别）。幅度都在 3% 以内，
**按形态实测，不做全局开关**。纯 CPU 段（部署里 tail 那一侧）默认开：`GGML_CPU_REPACK=ON`、
`common.h` 的 `no_extra_bufts = false`；但 `llama.h` 的 `use_extra_bufts` 默认是 `false`，
绕过 `common` 的入口要自己设。

## 7. 端到端：同机构造不能当判据

### 7.1 同机（head + tail 同机）

`-c 1024 -b 512 -np 1`，prompt 约 55 token，`n_predict 20`，温度 0：

| | 形态 | pp | tg |
|---|---|---:|---:|
| A | 全 CPU | 42.3 | 9.34 |
| B | tail 全上卡（head 在 CPU） | 40.3 | 10.76（+15%） |
| E | head `-ngl 8` + tail CPU | 32.5 | 8.44 |

**逐帧分解**（`QLH_SPLIT_TRACE`，stderr，np 64）：

| 段 | E（`-ngl 8`） | 全 CPU | 占 E |
|---|---:|---:|---:|
| plain（head 图） | **80.30 ms** | 76.01 ms | **67.3%** |
| gather（hidden 交接） | 0.02 | 0.02 | 0.02% |
| remote（tail + TCP） | 39.78 | 44.20 | 33.3% |

remote 再拆（tail 全 GPU 作对照）：TCP/协议 ≈ 8.40 ms/帧（7.0%），tail CPU 计算 ≈ 31.4 ms。

**head 在 server 里没有变快**：`-ngl 8` 的 plain 反而慢 5.6%，抖动也大（52~160 vs 63~100 ms）。
差别指向调度——probe 是单 range 上的连续 decode，GPU kernel launch/sync 能与 CPU 线程池跨步
流水；server 每帧一次 decode + TCP 往返，两端争同一批 CPU 线程。**段级 probe 与服务路径的收益
不可折算，方向可能相反**（probe `-ngl 8` 47.6 vs 63.7 ms，快 25%）。

### 7.2 部署形态（head 独占本机，tail 在另一台机器）

probe，pp 55 / n 64：

| 档 | 配置 | tg |
|---|---|---:|
| a | 纯 CPU | 14.905 |
| b | `-ngl 8` | **17.834（+19.6%）**（另一轮 20.996 / +34%） |
| c | `-ngl 32 --n-cpu-moe 32 --moe-cache-mib 1024` | **20.998（+40.9%）**，命中率 68.9% |

**这一档才是 head offload 的判据。** 7.1 之所以反了：tail 的 31.4 ms 与 TCP 的 8.4 ms 是固定
成本，而 head 的非专家部分要等 GPU 同步。

## 8. 坑

1. **`-ngl == 本段块数` 会让 `upper_bound` 返回 `end()`**，`devices.at()` 抛 invalid vector
   subscript。已把分配比例钳到 `act-1`。
2. **tail 本地块数含 NextN**：`32:40` 是 9 块。
3. **`build-cuda` 增量必 LNK2001**，必须 `--clean-first`；`GGML_NATIVE` 与 `GGML_BACKEND_DL`
   不兼容（CMake FATAL）；旧的 `ggml-cpu-*.dll` 残留会被优先加载，要删。**注意
   `--clean-first` 会清掉共享的 `build-split/bin`，重建后要确认 `dflash-taps.exe` 等目标在。**
4. **`--ncmoe` 的长选项是 `--n-cpu-moe`**；`-ncmoe` + mmap 有性能警告，建议 `--load-mode none`（未测）。
5. **段级 probe 与 server head 不是同一张图**，速度不能互相折算。
6. **`--split-remote` 与 `--split-tail-range` 互斥**，只能用 `--split-layer-range 0:32 --split-remote host:port`。

## 9. 复现

```bash
.qlh_tmp/seg_probe.exe -m <target> 0:32 --ngl 8 --no-op-offload -c 1024 --pp 55 -n 64
bash .qlh_tmp/seg_bench.sh      # 端到端；vram_accept.sh 跑显存五组（组间清场）
```

## 10. 显存预算预检 + 溢出后检

详细理由（含实测对照）在 `src/qlh-vram-budget.h` 头部注释里。

### 10.1 两层机制

预检（加载前）打印 `VMEM: need ~X GiB  free Y GiB  of Z GiB`，
`want = 上卡张量字节 × 1.13 + 640 MiB`、`free = total - used`；后检（ctx 建好后）用
nvidia-smi `used` 的前后差得 `landed`，`spill = landed < want × 0.85`。`want` 复刻
`load_tensors()` 的上卡判定（`-ncmoe` / `-ot` 按 loader 原样匹配，`token_embd` 恒不上卡），
`--moe-cache-mib` **必须计入**——它是真实显存。

### 10.2 必须用 nvidia-smi

WDDM 允许 over-commit，所有「还剩多少显存」的 API 报的都是「驱动还愿意给多少」：tail 已驻留
5.2 GiB 时 `nvidia-smi` 报 `free 2517 MiB`，`cudaMemGetInfo`（= `ggml_backend_dev_memory`）
报 **7106 MiB**。第一版预算检查建立在这个数字上，把真会溢出的配置判成安全。查不到时打印
`cannot read committed VRAM; skipping budget check` 并跳过——宁可跳过并说明，不假装检查过。

### 10.3 告警 vs 拒绝

- **后检永不退出**：溢出已发生，命名它才有用；实测存在「溢出档仍是最快档」的情形。
- **预检默认也只告警**，`QLH_VRAM_STRICT=1` 才拒绝，`QLH_VRAM_ALLOW=1` 反向强制——硬拦会让
  「这个溢出档到底快不快」不可测，而那正是告警存在的意义。CPU 构建整体跳过；
  `QLH_VRAM_DISABLE=1` 总开关（对照 0.442 vs 0.440，零影响）。

### 10.4 验收

部署形态（head 单段在本机、tail 在另一台机器的 CPU）：两档都放行、后检判 verified
（`2.61 / 3.61 GiB resident for 1.19 GiB`），整链 tg 10.86 / 14.61（cache 1024 / 2048）。

两段同机（**压力场景，不是部署形态**）：D 组预检报 `short by 2.37 GiB`、后检
`2.44 for 3.92 GiB`、**请求照常跑完 tg 0.297**（事故实测 0.343）；C' 组（加 cache）报溢出、
tg 0.44。C/C' 保留的唯一理由：它们能把「`want` 漏算 moe cache」照出来。C' 的误报要靠
「overhead 按 context 长度缩放」消除——不改，固定 640 MiB 是 v2 的实测拟合。
