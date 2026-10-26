# llama.cpp MoE expert cache：本机 8 GB 实测与归因（2026-10-26）

> 状态：**现行**（本机测量结论；推理默认配置与 `--cpu-moe` 默认值未改动）
> 更新日期：2026-10-26
> 适用范围：本机（RTX 4060 Laptop 8 GB）上 `qwen35moe` 整模 GGUF 的 decode 吞吐与
> `--moe-cache-mib`（上游 PR #29887 引入的 host-expert GPU LRU 缓存）评估。
> 不覆盖：层段/接力路径、RPC 路径、长上下文、其他量化档位。

## 1. 结论

| 判据 | 结论 | 证据 |
| --- | --- | --- |
| 是否启用 `--moe-cache-mib` | **不启用**：同口径下与无缓存持平 | §5 表 5-2 |
| 可用缓存容量上限 | 5120 MiB（显存边界）；5632 MiB 触发 `--fit` 中止并降级 | §5 表 5-1 |
| `--fit` 与 `-cmoe` 的关系 | **互斥**：同时给出时 `--fit` 直接 abort | §5 表 5-3 |
| `b10964` → `b11889` 的纯 decode 收益 | **0**：23.90 vs 23.90 tok/s | §5 表 5-2 |
| decode 的瓶颈性质 | **每层串行延迟**（DRAM 聚合读 49.8~54.0 GB/s 只用到约 30%） | §6 |

## 2. 环境与口径

| 项 | 值 |
| --- | --- |
| CPU | i9-13900H，6P+8E / 14 物理核 / 20 逻辑核 |
| 内存 | 32 GB DDR5-4800，2 × 16 GB 双通道（`CT16G48C40S5.M8A1`） |
| GPU | RTX 4060 Laptop，8188 MiB，PCIe 4.0 **x8** |
| 模型 | `models/qwen3-6-35b-a3b-gguf/Qwen3.6-35B-A3B-UD-Q4_K_XL.gguf`（22,360,456,160 B = 20.82 GiB，34.66 B 参数，5.161 bit/参数） |
| llama.cpp | `runtime/llama-cpp/b10964`（2026-09-14）、`build/moe-cache-20261026/b11889`（2026-10-25，含 #29887 与 #30112） |
| 生成口径 | `-c 2048`、`-t 14`、`temperature=0`、`top_k=1`、`n_predict=128`、单序列 |
| 采样方式 | 每轮换一个 prompt（`--distinct`），排除 `llama-server` 按 LCP 复用 slot 造成的 prefill 免除 |

跨报告、跨脚本的 tok/s 不构成对照；本表数字只在上述口径内互相可比。

## 3. 模型侧账本（GGUF 元数据实测）

| 键 | 值 |
| --- | --- |
| `qwen35moe.block_count` | 40 |
| `qwen35moe.expert_count` | 256 |
| `qwen35moe.expert_used_count` | 8 |
| `qwen35moe.expert_feed_forward_length` | 512 |
| `qwen35moe.embedding_length` | 2048 |
| `qwen35moe.attention.head_count` / `head_count_kv` | 16 / 2 |

由此得到两个决定性能的量：

- **全部专家权重**：40 × 256 × 3 × 2048 × 512 = 32.21 B 参数 = **19.35 GiB**（占整模 93%）
- **每 token 激活**：40 × 8 × 3 × 2048 × 512 = 1.007 B 参数 = **619 MiB**

`--cpu-moe` 把 19.35 GiB 专家留在 host RAM，因此**每个 token 都必须把 619 MiB 重新读一遍**——
decode 是 batch=1，同一份专家没有第二个 token 可以摊薄。

## 4. 基线：线程数与 pp/tg 比

### 表 4-1：线程扫描（`llama-bench`，b10964，`-ngl 99 -ncmoe 40`，tg128，`-r 3`）

| `-t` | 4 | 6 | 8 | **14** | 20 |
| --- | --- | --- | --- | --- | --- |
| tok/s | 16.40 ± 1.29 | 19.76 ± 1.91 | 21.69 ± 0.60 | **23.87 ± 0.44** | 20.50 ± 0.22 |

峰值在 14，等于物理核数；超线程（20）反而低 14%。证据 `build/moe-cache-20261026/threads-b10964.log`。

### 表 4-2：prefill 与 decode（b11889，`-ngl 99 -ncmoe 40 -t 14`，`-r 3`）

| 测试 | tok/s | 每 token 耗时 | 每层耗时 |
| --- | --- | --- | --- |
| pp512 | 213.12 ± 18.41 | 4.69 ms | 0.117 ms |
| tg128 | 25.50 ± 3.23 | 39.2 ms | 0.98 ms |

**8.4× 差距**是本文归因的支点：prefill 时同一层的专家被 512 个 token 共用（每层最多读 256 个专家块），
decode 时 8 个专家块只服务 1 个 token。差距来自**摊薄**，不来自带宽。证据 `build/moe-cache-20261026/pp-tg-b11889.log`。

## 5. moe cache 扫描

### 表 5-1：缓存容量（b11889，`-ngl 99 -cmoe -t 14 -c 2048`，同 prompt × 2）

| `--moe-cache-mib` | 0 | 2048 | 4096 | 5120 | 5632 |
| --- | --- | --- | --- | --- | --- |
| tok/s | 25.16 | 15.71 | 22.20 | 18.45 | 14.68 |
| 相对无缓存 | 1.00× | 0.62× | 0.88× | 0.73× | 0.58× |

5632 超出显存，`common_fit_params` 报
`failed to fit params to free device memory: n_gpu_layers already set by user to 99` 后降级
（`r-b11889-cache5632.log:5`）。同一 prompt × 2 的方差极大（5120 两轮为 8.22 / 28.67），
该表只用于确定"容量越大越好"的单调性与显存上限，绝对值见下方表 5-2。

### 表 5-2：同口径对照（每轮换 prompt，5~10 轮）

| 配置 | tok/s（均值） | 逐轮 |
| --- | --- | --- |
| b10964 无缓存 | 23.90 | 24.85 / 23.74 / 23.91 / 23.39 / 23.60 |
| b11889 无缓存 | 23.90 | 24.87 / 24.95 / 23.49 / 22.91 / 23.30 |
| b11889 + cache 5120 | 23.64（后 3 轮 24.14） | 10 轮见 `r-cache5120-10r.json` |
| b11889 + cache 5120 + `ngram-mod` | 20.03 | 19.81 / 24.01 / 19.83 / 17.78 / 20.52 / 18.22 |
| b11889 `--fit` 接管 + cache 4096 | 13.58 | 12.15 / 13.79 / 11.88 / 11.77 / 14.30 |

结论：**5120 MiB 缓存与无缓存持平（23.64 vs 23.90，差距在测量噪声内）；叠 `ngram-mod`
或改由 `--fit` 接管都更慢。** b10964 与 b11889 的纯 decode 完全一致，版本升级本身不带吞吐收益。

### 表 5-3：`--load-mode none` 与 `--fit` 的边界

| 配置 | 结果 | 证据 |
| --- | --- | --- |
| `--load-mode none`（无缓存） | 25.24 | `r-b11889-loadmode-none.json` |
| `--load-mode none` + cache 3072 | 26.20 | `r-cache3072-loadnone.json` |
| `--load-mode none` + cache 5120 | CUDA OOM（`cudaGraphInstantiate`） | `r-cache5120-loadnone.log` |
| `-cmoe` + `--fit` | `--fit` abort：`tensor_buft_overrides already set by user` | `r-fit-cache4096.log:6` |
| 不给 `-ngl/-cmoe`，仅 `--fit` + cache 4096 | 13.58 | `r-fit-auto-cache4096.json` |

`--load-mode none` 在无缓存时与 mmap 持平（加载时间从 4~6 s 涨到 16~17 s），
缓存容量上限下调到 3072 才不 OOM，且没有复现上游同规格报告的收益。

## 6. 命中率与归因

### 6.1 命中率（`llama-cli -lv 4`，进程自然退出后打印）

```
llama_moe_cache: ubatch <= 8: hits = 34631, misses = 7990, hit rate = 81.25%, uploaded = 14796.81 MiB
llama_moe_cache: ubatch  > 8: hits = 527,  misses = 1436, hit rate = 26.85%, uploaded = 2614.44 MiB
```

配置为 `-ngl 99 -cmoe -t 14 -c 2048 --moe-cache-mib 5120 -n 128`，证据
`build/moe-cache-20261026/cli-hit.log:439-440`。两个可直接读出的量：

- **decode 桶（`ubatch<=8`）命中 81.25%**，`uploaded` 折算 **115.6 MiB/token**，
  与"619 MiB × 18.75% miss"自洽
- **prefill 桶（`ubatch>8`）命中仅 26.85%**，大 batch 基本吃不到缓存

该统计是 info 级日志，默认 `verbosity = 3` 不打印；`-lv 4` 且进程正常退出（非强杀）才输出，
格式串见 `b11889/llama.dll` 的 `llama_moe_cache: %s: hits = %zu, misses = %zu, hit rate = %.2f%%, uploaded = %.2f MiB`。

### 6.2 有效带宽与内存带宽对比

| 量 | 值（两次独立运行） | 来源 |
| --- | --- | --- |
| decode 有效带宽（619 MiB / 41.8 ms） | **15.5 GB/s** | 由 §5 表 5-2 的 23.90 tok/s 折算 |
| 单线程顺序读（8 GiB 缓冲） | 13.5 ~ 13.9 GB/s | `membw2.log` |
| 单线程 1728 KiB 随机块读（top-8 形态） | 11.8 ~ 12.6 GB/s | `membw.log` |
| 14 线程不相交分区聚合读 | **49.8 ~ 54.0 GB/s** | `membw2.log` |
| 20 线程不相交分区聚合读 | 51.7 ~ 54.5 GB/s | `membw2.log` |

decode 只用到 DRAM 聚合带宽的 **约 29~31%**，却与单线程小块随机读（11.8~12.6 GB/s）同量级。
即：每层 13.5 MiB（8 × 1728 KiB）的可并行度太低，读不满通道，**每层 0.98~1.05 ms 是延迟受限**。

### 6.3 缓存为何不转化为吞吐

按 81.25% 命中，host 侧读取从 619 MiB/token 降到 115.6 MiB/token（降 81%），
但实测 uploaded 同样是 115.6 MiB/token，即这部分要经 PCIe 上传；本机链路为 4.0 x8。
两条路径的实测均值相同（23.64 vs 23.90），说明省下的 host 读取被上传与每层额外同步抵掉，
而 §4 表 4-2 的 8.4× 已表明 decode 的临界路径是"逐层串行推进"，缓存没有缩短这条链。

## 7. 与上游作者数据的差异

| 项 | 上游 #29887 作者实测 | 本机 |
| --- | --- | --- |
| GPU / 链路 | 4090 24 GB、5090 32 GB，PCIe 4.0/5.0 **x16** | 4060 Laptop 8 GB，PCIe 4.0 **x8** |
| 专家总量 | 65.4 GiB | 19.35 GiB |
| 缓存容量 | 11000（17%）/ 19000（30%） | 5120（26%） |
| decode 命中率 | 77% / 89% | 81.25% |
| 收益 | 1.62× / 2.20× | 1.00× |

容量占比与命中率接近，差异集中在**链路代宽与显存容量**：上游同规格的 x8 报告（3090 @ PCIe 4.0 x8）
也只有 +1~5%。社区给出的盈亏平衡点约 85% 命中，本机 81.25% 落在其下。

## 8. 边界

- 长上下文（`c` > 2048）未测；KV 增长会进一步压缩可用缓存与 decode 速度。
- 只覆盖 `Q4_K_XL` 一档量化；更低比特位会按比例减少 619 MiB 这个量，方向未实测。
- 层段/接力与 RPC 的推理入口在 `src/inference_service/`，不经 `llama-server` 整模推理，
  本文结论限于 `llama_server` 路径。
- `src/inference_path_selector.py` 与 `src/llama_server_engine.py` 的注释引用
  `-ngl 99 --cpu-moe` = 15.94 tok/s（2026-10-13，配 `--spec-type ngram-mod`）。
  2026-10-26 用相同参数复测得到 25.48 tok/s（同 prompt × 2 轮，`r-b10964-t14-ngram.json`）；
  两组数字的采样口径不同，取吞吐数字时以表 5-2 为准。

## 9. 复现

```powershell
# 单次测量（每轮换 prompt，避免 slot 复用）
python build/moe-cache-20261026/measure_server.py `
  --exe build/moe-cache-20261026/b11889/llama-server.exe `
  --model models/qwen3-6-35b-a3b-gguf/Qwen3.6-35B-A3B-UD-Q4_K_XL.gguf `
  --threads 14 --runs 5 --distinct `
  --extra "--moe-cache-mib 5120" `
  --out <out.json> --log <out.log>

# 命中率（需自然退出，不能强杀）
llama-cli -m <model> -ngl 99 -cmoe -t 14 -c 2048 --moe-cache-mib 5120 -n 128 -st -lv 4 -p "..." < nul
```

原始产物：`build/moe-cache-20261026/`（`build/` 被 `.gitignore` 忽略，不入库）。
