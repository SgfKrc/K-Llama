# 进展：当前能跑到什么程度

> 状态：**现行**
> 更新日期：2026-10-31
>
> 适用范围：本仓库当前的能力快照——能跑什么、做到过什么、还差什么。
> 数字口径与出处不在本文重复，指向 `docs/MEASUREMENTS.md` 与 `docs/SPECULATION.md`。

---

## 1. 现在能跑什么

两台机器、一个 35B-A3B MoE 模型、跨机层段接力 + DFlash 投机解码：

```
本机（i9-13900H + RTX 4060 Laptop 8 GiB）  head 0:32   -ngl 32 -ncmoe 24
        ↕  TCP，每轮 8 KB hidden
ROG（Ryzen 5 9600X，15.2 GiB）             tail 32:40  10 线程、纯 CPU
```

- **20.9 tok/s**（64-token 生成口径；写满 128 token 时 13.4 —— 代码后段不可预测，
  接受率从 96% 掉到 70%），接受率 96.4%，每轮前进 5.82 token
  （`topology-cross-mtp.toml`，2026-10-31）
- 输出与不做投机的贪心解码**逐位一致**
- 对外是 OpenAI 兼容的 SSE 流式接口；TUI 与控制面在 Python 侧

作为对照：同一个模型在本机单卡整模 `-ngl 6` 只有 **10.5 tok/s**，
纯 CPU 是 7.1。跨机两段是这台机器上唯一超过单机的形态。

> 上面两个吞吐都是**解码阶段**口径（`scripts/check_p4_cross.py` 的 `t_gen`，
> 不含 prefill 与协议开销），prompt 是 `def quicksort(arr):`。
> 跨 prompt 比较要重测——接受率由文本可预测性决定，不由模型规模决定。

---

## 2. 里程碑

| 阶段 | 内容 | 结果 |
| --- | --- | --- |
| P0 | 单进程基线，`-ncmoe` 语义 | 28.02 ± 0.15 tok/s（旧 Q4_K_XL 20.81 GiB 模型） |
| P1 | 多槽 `layer_inp` 打通 worker/engine/StageClient | 与单槽逐位相同 |
| P2 | 两进程 + TCP 传输 | 段间 hidden 走 int16 定点，5 份拷贝降到 1 份 |
| P3 | DFlash 草稿三件套（open/inject/block）+ rollback | 8 路注入、15 草稿、16 验证跑通 |
| P4 | 六条定位修正后验收 | 98.0% 接受率（原生 `llama-speculative-simple` 94.1%），3.94 token/轮 |

`docs/PATCHES.md` 记的是两代补丁的分界，`docs/ISSUE-*.md` 记的是 P3/P4 期间定位到的
三个缺陷：warmup 越界、hybrid `seq_rm` 语义、recurrent 快照跨 ubatch 不滚动。

---

## 3. 已做的优化（按收益排）

| 优化 | 手段 | 收益 |
| --- | --- | --- |
| 专家级 offload | `-ncmoe` 把专家放 GPU，而非整层 | 整层 11 层 10.11 → 专家 8 层 28.47 tok/s |
| head 段混合切分 | `-ngl 32 -ncmoe 24`：注意力与 dense 上卡，前 24 层专家留 CPU、后 8 层专家上卡 | head 段单独 13.6 → 20.5 tok/s |
| 草稿上 GPU | DFlash Q4_K_M 0.24 GB 放 4060 | 跨机 5.7 → 10.0 tok/s |
| 释放远端内存 | 关掉 ROG 上的常驻浏览器 | `tail_fwd` 226 → 76 ms |
| tail 线程数 | ROG 侧 10 线程 | 并入上面那档 |
| 切点实验 | 32:40 → 36:40 | 17.0 → 17.7（+4%），**未采纳**，负载会压到笔记本 |
| DFlash 草稿深度 | k=3 → **k=5** | 17.0 → 20.9 tok/s（64 token）；12.2 → 13.4（128 token）。k=8 起崩塌 |
| 投机按负载开关 | 代码类 k=2，其余 k=0；6 步后接受率 <35% 自动关 | 见 `docs/SPECULATION.md` §10 |
| n-gram 草稿 | 上游原生，优先级在 DFlash 之前 | 代码 prompt 10.4 vs 无投机 7.4 tok/s；对话自动落空 |

已被实测排除的方向（各条口径见 `docs/SPECULATION.md` 与 `local_docs/`）：
`-fa on`、`-ub 64`、KV 量化 q8_0、CPU 绑核、`--load-mode mlock`、`--moe-cache-mib`、
浅层投机（未训练的截断层接 `lm_head` 分布错位，α < 0.09）、k ≥ 5 的深草稿。

---

## 4. 已接的组件

**llama.cpp fork（`H:\llamacpp-c`，相对上游 15 文件 +447/−26）**

| 标记 | 内容 |
| --- | --- |
| `[C-scheme]` | `LLAMA_LAYER_RANGE=lo:hi` 层范围加载、`embeddings_pre_norm` 接力向量、lazy 全局张量、跳过 lm_head |
| `[C-scheme]` | `qwen35` / `qwen35moe` 的 pre-norm tap；`extern "C"` 导出供 C 侧按名解析 |
| `[TAG_RECURRENT_ROLLBACK_SPLITS]` | recurrent 快照按 ubatch 滚动，跨批回滚不再读到过期槽位 |
| `[DFLASH-01]` | 末层输出挂 `t_layer_inp[n_layer]`、`llama_model_n_embd_inp_enc`、warmup 空跑越界修复 |
| `[SLIM-ARC]` | prefetch 一级：`init_mappings` 注册 mmap 区间、`graph_compute` 通知当前层；win32 侧用 `PrefetchVirtualMemory` 等替代 madvise，默认关闭（`SLIM_ARC_PREFETCH=1`） |
| `[DFLASH-01]` | 末层输出挂 `t_layer_inp[n_layer]`、`llama_model_n_embd_inp_enc`、warmup 空跑越界修复 |

**shim（`shim/qlh_keep_head_gpu.c`，约 1650 行 C）**

把 llama 上下文包成 C ABI：多槽 `layer_inp`、DFlash `draft_open/inject/block/rollback/reset`、
MTP 草稿、采样器、`seq_rm`。MSVC 导出的 C++ mangled 名用 `GetProcAddress` 运行时解析。

**Python 侧（扁平布局，约 30 个模块）**

| 文件 | 职责 |
| --- | --- |
| `topology.py` | TOML 拓扑解析，切分点按 `full_attention_interval` fail-closed |
| `chain_client.py` | 绕链驱动、两个草稿源（MTP 递归 / DFlash 并行块）、按负载选 k、内存预检 |
| `engine.py` / `keep_head.py` | shim 的 ctypes 绑定 |
| `stage_worker.py` / `protocol.py` / `wire.py` | 段进程与帧协议 |
| `server.py` / `stream.py` / `proto_responses.py` | OpenAI 兼容接口与 SSE |
| `chat_tui.py` / `koakuma.py` | TUI 与控制面 |

**通用跨机驱动（`scripts/`，设备无关，可入库）**

`check_p4_cross.py` 从拓扑读一切（含 `ssh_user` / `start_cmd`），target layer id 按切点动态算；
`tail-worker.cmd` 是远端启动模板；`topology-example.toml` 是占位模板。
站点配置（IP、账号、路径）全部在 `.gitignore` 的 `topology-*.toml` 里，不进版本库。

---

## 5. 未接 / 未做

| 方向 | 状态 | 依据 |
| --- | --- | --- |
| SLIM-ARC | **未接**，有触发条件 | `docs/SLIM-ARC-接入计划.md` |
| 跨框架接力（llama.cpp ↔ torch） | 未做 | `README.md` 未做节 |
| LiLiCorr 轻量重排头 | 方案已写，未实施 | `docs/LILICORR-轻量头.md` |
| 弱设备（Y700 / Surface）接入 | 评估已写，当前为负收益 | `docs/ANDROID-SURFACE-段接入评估.md` |
| 多序列 token 共享 | 未做，对这类请求明确报错 | `README.md` |
| n-gram 影子模式 | 未做（上游原生策略已可用） | `docs/SPECULATION.md` §6 |

---

## 6. 复现

```powershell
# 拓扑里写清两段的 model / host / port / layer_range / gpu_layers / cpu_moe
# --spec-k 默认 -1，按负载自动选（代码类 k=2，其余 k=0）；要强制就写具体值
python chain_client.py --topology topology-cross-mtp.toml

# 逐位正确性：与 --spec-k 0 的输出比对
# 段区间与显存：
python checks/verify_ranges.py ; python checks/verify_vram.py
```
