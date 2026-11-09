# K-Llama v2.1

**中文** | [English](docs/README.en.md)

> 状态：**现行**
> 更新日期：2026-11-09
> 适用范围：分段链的对外入口——定位、能力、快速开始、关键约束、实测口径与文档索引。机制归
> [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md)，边界代价归
> [`docs/SEGMENTED-LIMITS.md`](docs/SEGMENTED-LIMITS.md)，补丁构成归
> [`docs/PATCHES.md`](docs/PATCHES.md)。

K-Llama 让 llama.cpp 按层分段运行：每台机器映射同一份 GGUF，只创建并读取自己那几层的张量，
段间只传 `n_embd` 个 float 的隐藏态（`src/llama-context.cpp` 的 `ctx_tail`）。改造下沉在
`llama_context` 里，`llama-server` 自己就是链的驱动者：`/completion`、`/chat/completions`、
slot、prompt cache 全部走原生实现，客户端不需要知道段在哪台机器上。

分工：v2 是 Python 驱动链（`chain_client.py`、`stage_worker.py`、`stream.py`，TUI 入口
`koakuma`），解决「模型怎么分到多台机器」；v2.1 把这条链下沉进 `llama_context`，解决「分段
之后能力与整模对齐」。v2 链仍在，吞吐数据归 [MEASUREMENTS]。

## 能力

| 能力 | 结论与判据 | 详细版 |
| --- | --- | --- |
| 本地分段 | `--split-layer-range` + `--split-tail-range` 同进程出 token；`-np 4` 四并发输出与串行逐字节相同 | `verify.sh` 组 5；[SPLIT-MEASUREMENTS] §4 |
| 前缀复用 | 3103 token 历史整段重算 19847 ms，第二轮只解 38 个新 token 用 714 ms（2B 链，`-c 8192 -np 1`） | [SPLIT-MEASUREMENTS] §1 |
| 远程链 | `llama_split_attach_remote()` 建链；KV 广播、checkpoint 覆盖全链、120 s 读超时、链死而进程活 | `verify_chain_death.py`；[SEGMENTED-LIMITS] §2.7 |
| 远程链投机 | DFlash / DSpark 的 `layer_inp` tap 随 decode 回复跨进程（协议 v9），接受率与整模逐 prompt 相同 | 下表；[SPLIT-MEASUREMENTS] §14 |
| MTP 投机 | tail 的 NextN 头即 draft，k 步递归在 stage 内跑 | `verify.sh` 组 8；[SPECULATION] §4 |
| grammar / reasoning budget | 候选逐 token 自检、被拒整帧回退；预算耗尽只收紧不放宽 | `common_sampler_grammar_rejects()`；[PATCHES] 第四代 |
| 段的内存 | logits 与 pp 图缓冲按实际输出行数分配：tail 1710.76 → 3.79 MiB（1806-token ubatch，2B 链，`-c 8192 -np 4 -b 2048`），远程 20:24 段 2052.10 → 240.09 MiB（2B，`-c 4096 -b 2048`，词表 248320） | [SPLIT-MEASUREMENTS] §5 |
| `seq_rm` | 失败返回 `bool`，五个调用点各自降级，链路停摆不杀进程 | [SEGMENTED-LIMITS] §2.9 |

表中 2B 链指 `qwen35-2b-Q4_K_M`（head `0:20` + tail `20:24`，`build-split` 纯 CPU 构建），口径
全文见 [SPLIT-MEASUREMENTS] §0。

## 快速开始

```bash
# 本地两段：tail 在同一进程里
llama-server -m model.gguf --split-layer-range 0:32 --split-tail-range 32:40 -c 8192 -np 4
```

```bash
# 远程链：每台起一个 split-stage，head 用 --split-remote 按链序列出
split-stage -m model.gguf --layer-range 24:32 --role relay --rs-seq 8 -c 8192 --host 0.0.0.0 --port 9331
split-stage -m model.gguf --layer-range 32:40 --role tail  --rs-seq 8 -c 8192 --host 0.0.0.0 --port 9332
llama-server -m model.gguf --split-layer-range 0:24 -c 8192 \
  --split-remote 10.0.0.2:9331 --split-remote 10.0.0.3:9332
```

```bash
# 远程链 + DFlash k=2：产品路径就是 -md 加 --spec-type
llama-server -m model.gguf --split-layer-range 0:32 -c 4096 \
  --split-remote 10.0.0.3:9332 -md dflash.gguf --spec-type draft-dflash --spec-draft-n-max 2
```

- `--split-layer-range` 是本进程加载的层（链首必然是它）；`--split-tail-range` 是同机尾段，与
  `--split-remote` 互斥；`--split-remote` 可重复、按链序。
- `split-stage` 的 `--role` 与 `--rs-seq` 必填：角色过去由加载器推断，推错时得到的是通顺而错误的
  输出。
- 开投机时抬高 `--split-rs-seq`（一轮要回滚 `n_max` 个位置）；tail 必须在 `OP_HELLO` 里报
  `has_nextn`，报不出时 server 打 WARN 并把 draft-MTP 摘掉。弱机器上显式给 `-t 8`：它同时设
  `n_threads` 与 `n_threads_batch`，两者默认都是 4。
- k 的现行拼写是 `--spec-draft-n-max`。上游已移除 `--draft-max` / `--draft-n`，旧名会让进程直接
  报错退出。

## 关键约束

1. **架构**：链首必须导出 `output_norm` 之前的隐藏态，`split_arch_exports_pre_norm()`
   （`src/llama-context.cpp`）只认 `qwen35` / `qwen35moe`，其他架构在建上下文前报错退出。tail 段
   只出 logits，不受此限。多模态的 4 个 M-RoPE section 能跨段传（`src/llama-context.cpp` 的
   `split_expand_pos()`），端到端未跑，见 [SPLIT-MEASUREMENTS] §7。
2. **远程链单序列**：`llama_split_attach_remote()` 拒绝 `n_seq_max > 1`。并发在 head 那台起多个
   server、各走自己的链；单机内用本地链。见 [SEGMENTED-LIMITS] §2.4。
3. **链上投机源按通道挑**：DFlash / DSpark 走 `layer_inp` tap（协议 v9），远程链零损失；MTP 走
   nextn 通道，链上接受率低于整模，成因见「实测」一节。见 [SEGMENTED-LIMITS] §2.2。
4. **`--split-layer-range` 的合法性**：范围从 0 起；在模型中间结束时必须给 `--split-tail-range`
   或 `--split-remote`。两道检查都在 `common_init_from_params()`（`common/common.cpp`），读的是
   模型里读回的范围（`llama_model_layer_range()`），`LLAMA_LAYER_RANGE` 入口一并覆盖。残留边界
   见 [SEGMENTED-LIMITS] §2.6。
5. **核显按整段 offload**：`-ngl 99` 各段全上；中间档一律更差，交叉点随 prompt 长度移动，35B
   装不进共享堆，主线走纯 CPU。见 [IGPU-SLOWDOWN]。
6. **hybrid 的 partial `seq_rm`**：上游在默认 `n_rs_seq = 0` 下永远失败、attention KV 一行不删
   （上游 #28007 / #31656）；本仓返回 `bool`、五个调用点各自降级。见 [SEGMENTED-LIMITS] §2.9。

## 实测：投机选型与跨机

口径：35B `Qwen3.6-35B-A3B-UD-Q4_K_M`（`unsloth/Qwen3.6-35B-A3B-MTP-GGUF`，21.11 GiB），贪心，
k=2，2026-11-09。整模三行是纯 CPU（`-c 4096 -b 512 -t 8 -ngl 0`）、`n_predict 64`、B 树 / 代码 /
中文三个 prompt；链三行是本机 head `0:32` + Surface tail `32:40`（i5-1035G4 4C8T / 16 GB，
Tailscale 9.5 MB/s、RTT 10–12 ms）、prompt 1102 token。

| 形态 | accept（a/g） | mean_len | 相对整模 |
| --- | ---: | ---: | --- |
| 整模 + MTP k=2 | 0.9098 | 2.820 | — |
| 整模 + DSpark-6L k=2 | 0.8357 | 2.671 | — |
| 整模 + DFlash k=2 | 0.8298 | 2.660 | — |
| 链 + DFlash k=2 | 117/141 = 0.82979 | — | 逐 prompt 相同 |
| 链 + DSpark k=2 | 117/140 = 0.83571 | — | 逐 prompt 相同 |
| 链 + MTP k=2 | 108/157 = 0.68790 | — | −24% |

短上下文的最优解是整模 + MTP。链上 DFlash 与 DSpark 实现零损失，本地链与远程链一致：输出 3/3
逐字节相同（250 / 244 / 249 字符）。MTP 的落差来自 [SPLIT-MEASUREMENTS] §11.4 记的两处链上改动
——catch-up decode 只对 `!is_chain && !is_remote` 开，`draft()` 在 `is_chain` 下每轮清草稿 KV；
当前草稿 KV 策略下这是自洽的架构边界，改成 opt-in catch-up 才能同时动这两处。

prefill 的 tap 代价：A 32.1 s / B 32.8 s / C 29.7 s，A−C ≈ +2.4 s，与 25 MB ÷ 9.5 MB/s ≈ 2.6 s
吻合；tap 按 512 行 ubatch 成批过网，逐 token 往返会报出数量级更大的差。

markov 头（DSpark-6L）放大的是「可抄写性」：k=8、`n_predict 256` 下，mid 档（prompt 1146 token）
比 DFlash 高 15.3 点，hard 档（1157 token，中文改写）只高 4.1 点，short 档高 2.9 点——mid 与 hard
长度几乎相同，两个档的差值相差近 4 倍。per-position 里 markov 的价值从 pos2 才开始，k=2 覆盖不到。选型先看
负载的可抄写比例（JSON 键名、代码签名、工具调用），再看 k 与 `n_predict` 是否落在头擅长的区间。

口径坑：

- accept 率跨 `n_predict` 比较会得到相反符号：同一对草稿源在 `n=64` 时差 +0.59 点，`n=256` 时
  差 −0.94 点。报 accept 必须带 `n_predict` 与 k。
- `mean_len = 1 + accept × k`，不同 k 的 mean_len 直接比没有意义。
- DSpark-6L 的 base 是 NVFP4 量化版而非官方 BF16：比较只在「这份文件 vs 那份文件」这一层成立。

## 代码与验收

- fork：`H:\llamacpp-c`，分支 `rebase-merge`，单提交 `0c5c3bdd4`，父 `f477f795c`（上游最新，
  193 个提交之后），树 `982ccecd`；相对上游 52 文件 / +17831 −230。
- [llamacpp-split.patch]：42 文件 / +14287 −229 / 16166 行 / 840.6 KiB。纯分段链：层范围加载、
  本地 / 远程段、聚合 memory、checkpoint、协议 v7 验证通道 / v8 远程 MTP draft / v9 远程
  `layer_inp` tap、grammar / rbudget 兼容、`seq_rm` 失败显式化、device-buffer 宽度判定；不含
  SLIM-ARC，可单独编译、单独通过全部验收。
- [llamacpp-slimarc.patch]：13 文件 / +3544 −1 / 3662 行 / 155.2 KiB。SLIM-ARC prefetch，默认
  关闭，`SLIM_ARC_PREFETCH=1` 才装。借用的只有这一块：专家驻留、KV 驱逐、I/O 调度器未接，见
  [SLIM-ARC-接入计划]。
- 叠加判据：`f477f795c` + 主补丁 → 树 `d28e10df`；再叠增量 → 树 `982ccecd`，与 HEAD 的树相同
  （`git write-tree` 校验）。
- 与上游：16 个文件与 `f477f795c` 逐字节相同（rebase 前 0 个）。这次 rebase 移除的 17 个文件改动
  本属落后于上游——上游已包含那些提交，本仓那份适配已无必要；`src/models/dflash.cpp` 因
  device-buffer 诊断行重新进入变更集。清单见 [PATCHES]「与上游 193 个提交的重合」。

验收判据是逐位 / 逐字节的 logits，不比生成的文字；下表是 2026-11-09 的结果：

| 套件 | 结果 | 判据 |
| --- | ---: | --- |
| `bash build-split/verify.sh` | 52（十一组：1–8 组 22 + 组 9 2 + 组 10 20 + 组 11 8） | 逐位 / 逐字节；组 11 直接跑 `llama-server + 链 + -md` 产品路径 |
| `verify_chain_death.py` | 36 | 120 s 读超时、链死而进程活、失败可见 |
| `verify_seq_id.py` | 9 | 跨段 `seq_id` 与位置 |
| `verify_mtp.sh` / `verify_mtp_e2e.sh` | 6 / 6 | 链上 draft 与整模逐位、端到端逐字节 |
| `verify_grammar_spec.sh` / `verify_rbudget_spec.sh` | 23 / 36 | grammar 整帧回退、预算只收紧 |
| `verify_seq_rm_exact.py --expect-known-fail` | 1 pass / 5 known-fail | 5 条是上游 hybrid 的已知基线，作为对照保留 |
| 纯分段版（只叠主补丁） | 52 + 36 | 树 `d28e10df` 上独立构建、独立跑过 |

v2 驱动链的验收走 `python checks/verify_*.py` 与 `python -m unittest test_topology test_contract`；
各脚本的项数与最近一次结果见 [SPLIT-MEASUREMENTS] §6 与 §9.7。

## 文档地图

| 文件 | 管什么 |
| --- | --- |
| [ARCHITECTURE] | 层范围加载的三个机制、接力点、硬约束、什么算证据、与原生 RPC 的分界 |
| [PATCHES] | 四代补丁的构成与代际差异、rebase 到上游的记录、补丁全文 |
| [SEGMENTED-LIMITS] | 每条边界的根因、影响范围、修的时候要付的代价 |
| [SPLIT-MEASUREMENTS] | v2.1 实测：checkpoint、MTP、DFlash、协议 v9、选型矩阵、跨机零损失 |
| [MEASUREMENTS] | v2 Python 驱动链的全部实测数据 |
| [SPECULATION] | 投机判定、成本模型、k 扫描、接入实测 |
| [DEPLOYMENT] | 每台放什么、怎么换分段、远端 worker 怎么起 |
| [TESTING] | 判据与测试通道：测什么、怎么跑、什么算过 |
| [MEASUREMENT-PITFALLS] | 段内计时为什么骗人，怎么量才有效 |
| [MEMORY-LIMITS] | 内存与显存的账、校准常数、绑核纪律 |
| [IGPU-SLOWDOWN] | 核显为什么拖慢、`-ngl` 扫描与交叉点 |
| [SEGMENT-OFFLOAD] | 分段下 `-ngl` 的语义、段级与端到端实测、MoE 缓存、显存预算 |
| [PROGRESS] | 当前能力快照与已做的优化 |
| [SAMPLING] · [TUI] · [RESPONSES-API] | 采样映射 · TUI 页面与命令 · `/v1/responses` 翻译层 |
| [SLIM-ARC-接入计划] | prefetch 之外的缺页优化在什么条件下才启用 |

其余专题（`LILICORR-轻量头.md`、`ANDROID-SURFACE-段接入评估.md`、`ISSUE-*.md` 等）见 `docs/`
目录，归档在 `docs/archive/`。同一套 relay 在扩散模型上的实现是
[K-sd.cpp](https://github.com/SgfKrc/K-sd.cpp)：段间载荷是 MB 级张量，同一条序列内部两边都等所有
段，吞吐空间在多请求。

## 许可

MIT，见 [`LICENSE`](LICENSE)。上游 llama.cpp 同为 MIT，本仓库的补丁按同一许可分发；模型权重等
第三方产物随其各自来源。

[ARCHITECTURE]: docs/ARCHITECTURE.md
[PATCHES]: docs/PATCHES.md
[SEGMENTED-LIMITS]: docs/SEGMENTED-LIMITS.md
[SPLIT-MEASUREMENTS]: docs/SPLIT-MEASUREMENTS.md
[MEASUREMENTS]: docs/MEASUREMENTS.md
[SPECULATION]: docs/SPECULATION.md
[DEPLOYMENT]: docs/DEPLOYMENT.md
[TESTING]: docs/TESTING.md
[MEASUREMENT-PITFALLS]: docs/MEASUREMENT-PITFALLS.md
[MEMORY-LIMITS]: docs/MEMORY-LIMITS.md
[IGPU-SLOWDOWN]: docs/IGPU-SLOWDOWN.md
[SEGMENT-OFFLOAD]: docs/SEGMENT-OFFLOAD.md
[PROGRESS]: docs/PROGRESS.md
[SAMPLING]: docs/SAMPLING.md
[TUI]: docs/TUI.md
[RESPONSES-API]: docs/RESPONSES-API.md
[SLIM-ARC-接入计划]: docs/SLIM-ARC-接入计划.md
[llamacpp-split.patch]: docs/llamacpp-split.patch
[llamacpp-slimarc.patch]: docs/llamacpp-slimarc.patch
