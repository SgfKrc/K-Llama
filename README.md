<p align="center">
  <img src="docs/assets/banner.svg" alt="K-Llama — Llama.cpp for Kircheis" width="100%">
</p>

# K-Llama v2.2

**面向非整模加载 · 基于定制化 llama.cpp 引擎 · 轻量化大模型分布式边缘推理优化系统**

**中文** | [English](docs/README.en.md)

> 状态：**现行开发版 v2.2**；结题版本 v2.1（已冻结）
> 更新日期：2026-11-13
> 适用范围：分段链的对外入口——定位、能力、快速开始、关键约束、实测口径与文档索引。机制归
> [`docs/design/ARCHITECTURE.md`](docs/design/ARCHITECTURE.md)，边界代价归
> [`docs/limits/SEGMENTED-LIMITS.md`](docs/limits/SEGMENTED-LIMITS.md)，补丁构成归
> [`docs/deploy/PATCHES.md`](docs/deploy/PATCHES.md)。

K-Llama 让 llama.cpp 按层分段运行：每台机器映射同一份 GGUF，只创建并读取自己那几层的张量，
段间只传 `n_embd` 个 float 的隐藏态（`src/llama-context.cpp` 的 `ctx_tail`）。改造下沉在
`llama_context` 里，`llama-server` 自己就是链的驱动者：`/completion`、`/chat/completions`、
slot、prompt cache 全部走原生实现，客户端不需要知道段在哪台机器上。

分工：v2 是 Python 驱动链（`chain_client.py`、`stage_worker.py`、`stream.py`，TUI 入口
`koakuma`），解决「模型怎么分到多台机器」；v2.1 把这条链下沉进 `llama_context`，解决「分段
之后能力与整模对齐」。

**v2 已退场**（归档在 `slimarc/prototype`）；v2.1 是结题版本、已冻结（本仓 tag `v2.1`），v2.2 是现行
开发版。v2 的吞吐数据归 [MEASUREMENTS]，本代（v2.1 / v2.2 共用）的判据归 [SPLIT-MEASUREMENTS]，不可混用。

## 上游与版本

2026-11-12，上游从 `09338b85e` 起更新 31 个提交（到 `5676dfbe0`，tag `b12342`）。核心是 PR #30565
"llama : prune"（为 v1.0.0 发布准备）：18 个 NextN / layer-input / embd-row / state-row API 从
`src/llama-ext.h`（staging header）转正进 `include/llama.h`，`llama_set_embd_state_src` 改名
`llama_set_ctx_state_src`。这是**地基转正**，不是能力变更：分段透传正建立在这三条通道上（head 出
pre-norm hidden，relay / tail 经 embd row 消费），补丁不必再往 staging header 补 `extern "C"`
（`include/llama.h:52` 自带）；`split_*` 协议层上游没有，一行不动。上游这套 API 面向进程内的 draft 头组合，
跨机走 RPC backend + RDMA，与本项目特征透传不撞车；同批 `3bce8bd50`、`3ba1ce0e2` 也直接受益。

| 版本 | 基底 | 状态与口径 |
| --- | --- | --- |
| v2.1 | `09338b85e` | **结题版本，已冻结**：代码侧 tag `v2.1`（`4acf03c8b`，树 `548c3dfb0`）、`v2.1-kvmem`（`3f930466d`，树 `192b2a3d0`）；本仓 tag `v2.1`（`074dea49`，树 `83f31877`） |
| v2.2 | `5676dfbe0` | **现行开发版**（跟随上游的下一版）：能力与判据沿用 v2.1；实测未重测，验收已重跑（73/0/1，见下）；是否纳入结题待定 |

v2.2 的目标是把 **prefill / decode 分离**做完整，含三件：推翻远程链的单序列边界（v2.1 的理由是
「远程链不是多租户」）；引入批处理（`OP_DECODE` 每行带 `seq_id`，prefill chunk 与 decode token 混在
同一 batch）；prefill / decode 分离调度。v2.1 多序列实测 `-np 2` 端到端 0.81×、并发 prefill 0.70×。
留到 v3：多请求批处理（N 个请求的 hidden 打包进一帧，往返除以 N）、链式拓扑（现为星形，一次 decode
4 次跨网单程，段间直连省一次单程 12.5–15 ms、约 6–7%，但要沿链重做 `seq_rm` / checkpoint / 错误传播，约 15–20 人天）。

## 能力

| 能力 | 结论与判据 | 详细版 |
| --- | --- | --- |
| 本地分段 | `--split-layer-range` + `--split-tail-range` 同进程出 token；`-np 4` 四并发输出与串行逐字节相同 | `verify/verify.sh` 组 5；[SPLIT-MEASUREMENTS] §4 |
| 前缀复用 | 3103 token 历史整段重算 19847 ms，第二轮只解 38 个新 token 用 714 ms（2B 链，`-c 8192 -np 1`） | [SPLIT-MEASUREMENTS] §1 |
| 远程链 | `llama_split_attach_remote()` 建链；KV 广播、checkpoint 覆盖全链、120 s 读超时、链死而进程活 | `verify/verify_chain_death.py`；[SEGMENTED-LIMITS] §2.7 |
| 远程链投机 | DFlash / DSpark 的 `layer_inp` tap 随 decode 回复跨进程（协议 v9/v10），接受率与整模逐 prompt 相同 | 下表；[SPLIT-MEASUREMENTS] §14 |
| MTP 投机 | tail 的 NextN 头即 draft，k 步递归在 stage 内跑 | `verify/verify.sh` 组 8；[SPECULATION] §4 |
| grammar / reasoning budget | 候选逐 token 自检、被拒整帧回退；预算耗尽只收紧不放宽 | `common_sampler_grammar_rejects()`；[PATCHES] 第四代 |
| 段的内存 | logits 与 pp 图缓冲按实际输出行数分配：tail 1710.76 → 3.79 MiB（1806-token ubatch，2B 链，`-c 8192 -np 4 -b 2048`），远程 20:24 段 2052.10 → 240.09 MiB（2B，`-c 4096 -b 2048`，词表 248320） | [SPLIT-MEASUREMENTS] §5 |
| 段级 offload | `-ngl N` 数的是「本段最后 N 块」，不是整模层号（`src/llama-model.cpp` 的 `i_gpu_start` 按 `layer_range_active()` 分叉）；整模路径逐位不变，tail `32:40` 全量上卡段内 tg 35.7 → 1754.2 | [SEGMENT-OFFLOAD] §2、§3 |
| MoE 缓存 | `--moe-cache-mib 1024` 命中率 82.2%，tg 18.05 → 31.99（纯 CPU 16.37） | [SEGMENT-OFFLOAD] §5 |
| 显存预算 | 加载前预检 `VMEM: need ~X free Y`，ctx 建好后按 nvidia-smi 前后差后检；WDDM 的静默溢出变成一行日志 | [SEGMENT-OFFLOAD] §10 |
| `seq_rm` | 失败返回 `bool`，五个调用点各自降级，链路停摆不杀进程 | [SEGMENTED-LIMITS] §2.9 |

表中 2B 链指 `qwen35-2b-Q4_K_M`（head `0:20` + tail `20:24`，`build-split` 纯 CPU 构建），口径
全文见 [SPLIT-MEASUREMENTS] §0。

## 做了什么、换来了什么

**一句话**：把模型按层拆到多台机器上，换到的是**容量**，不是速度。速度要靠投机与缓存去补，而它们在
某些形态上补得回来、在另一些形态上补不回来。

**容量**：35B 整模 21.11 GiB，16 GiB 的笔记本装不下；拆成 head `0:32` 15.69 GiB + tail `32:40` 4.90 GiB，
每台都装得下，其中 tail 那 4.90 GiB 能压进一台 15.6 GB 的安卓平板（[CROSS-MACHINE] §6）。三段的极端形态
（本机 + Surface + 安卓）也跑通了（§7）。

**速度**：拆开本身是负的——2B 本机双进程 22.56 → 28.33 ms/token（**慢 26%**，[SPECULATION-DATA] §1.3），
因为没有跨机往返可省，两个进程还抢同一套 CPU 与带宽。跨机才有意义，但收益随负载翻号。

| 你想要的 | 代价 | 实测 |
| --- | --- | --- |
| 单台装不下的模型能跑 | 拆开不变快 | 2B 本机双进程 **−26%** |
| 借另一台机器的算力 | 每轮一次往返 | 35B 跨机 6.34–8.71 tok/s；整模纯 CPU 10.54（**有背景负载**，空闲时单 prompt 到 13.99–16.86，这一对只有当天记录、未落盘），而 16 GiB 的机器装不下整模 |
| 三段：再多装一段 | hidden 过网 2 次 → **4 次** | 6.53–6.94 vs 两段 7.42–8.71（**−12% ~ −20%**） |
| 跨机开投机 | 整行 logits **993 KB/轮**（不投机 48 B） | 长 prompt + DSpark **+12% ~ +17%**；短 prompt **−35%** |
| 段上卡 | 显存装不下就溢出 | `-ngl 6` 是拐点，`32/32` 是陷阱（段更快、整链更慢） |
| 边缘弱机入链 | 它只能扛几层 | 安卓平板扛 8 层 4.90 GiB，35B 跨机 6.3–8.7 tok/s |

**换来这些的技术**：链下沉进 `llama_context`（slot / prompt cache / grammar / 投机全走原生路径，客户端
无感）；段间只传 `n_embd` 个 float 的 pre-norm 隐藏态（8 KB，不传 logits）；协议 v9 的候选通道（48 B/轮）
与 `layer_inp` tap；投机（MTP catch-up 追平整模 0.9098，DFlash / DSpark 跨机零损失）；logits 缓冲按实际
输出行数分配（1710.76 → 3.79 MiB）；段级 offload 与 MoE 专家缓存（命中 82.2%，tg +77%）；前缀复用
（19847 → 714 ms）。

**判据**：以上全部在「与整模逐位 / 逐字节相同」的前提下量到——分段不改变数值，只改变它在哪台机器上算
（[SEGMENTED-LIMITS]、[report/04-验证](docs/report/04-验证.md)）。

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
- `split-stage` 的 `--role` 与 `--rs-seq` 必填：角色不由加载器推断，推错时得到的是通顺而错误的
  输出。
- 开投机时抬高 `--split-rs-seq`（一轮要回滚 `n_max` 个位置）；tail 必须在 `OP_HELLO` 里报
  `has_nextn`，报不出时 server 打 WARN 并把 draft-MTP 摘掉。
- 弱机器上显式给 `-t 8`：它同时设 `n_threads` 与 `n_threads_batch`，两者默认都是 4。
- k 的现行拼写是 `--spec-draft-n-max`。上游已移除 `--draft-max` / `--draft-n`，旧名会让进程直接
  报错退出。
- 远程链的命令不必手敲：`control/qlhctl.py` 读一份拓扑（`control/topology.example.toml`）负责起停
  与探活，部署细节见 [DEPLOYMENT]。

## 关键约束

1. **架构**：链首必须导出 `output_norm` 之前的隐藏态。`split_arch_exports_pre_norm()`
   （`src/llama-context.cpp`）只认 `qwen35` / `qwen35moe`，其他架构在建上下文前报错退出。tail 段
   只出 logits，不受此限。
2. **多模态位置**：4 个 M-RoPE section 能跨段传（`src/llama-context.cpp` 的
   `split_expand_pos()`），端到端未跑，见 [SPLIT-MEASUREMENTS] §7。
3. **远程链单序列**：`llama_split_attach_remote()` 拒绝 `n_seq_max > 1`。并发在 head 那台起多个
   server、各走自己的链；单机内用本地链（v2.2 目标推翻此限）。见 [SEGMENTED-LIMITS] §2.4。
4. **链上投机源按通道挑**：DFlash / DSpark 走 `layer_inp` tap（协议 v9/v10），远程链零损失；MTP 走
   nextn 通道，链上接受率低于整模，成因见「实测」一节。见 [SEGMENTED-LIMITS] §2.2。
5. **`--split-layer-range` 的合法性**：范围从 0 起；在模型中间结束时必须给 `--split-tail-range`
   或 `--split-remote`。两道检查都在 `common_init_from_params()`（`common/common.cpp`），读的是
   模型里读回的范围（`llama_model_layer_range()`），`LLAMA_LAYER_RANGE` 入口一并覆盖。残留边界
   见 [SEGMENTED-LIMITS] §2.6。
6. **核显按整段 offload**：`-ngl 99` 各段全上；中间档一律更差，交叉点随 prompt 长度移动，35B
   装不进共享堆，主线走纯 CPU。见 [IGPU-SLOWDOWN]。
7. **hybrid 的 partial `seq_rm`**：上游在默认 `n_rs_seq = 0` 下永远失败、attention KV 一行不删
   （上游 #28007 / #31656）；本仓返回 `bool`、五个调用点各自降级。见 [SEGMENTED-LIMITS] §2.9。

## 实测：开投机 vs 不开投机

口径：35B `Qwen3.6-35B-A3B-UD-Q4_K_M`（21.11 GiB），贪心，k=2，T=0。tok/s 取
`predicted_per_second`（decode 阶段），除标注 `ms/tok` 的行外都是 tok/s（越高越好）；`相对` 列
正数 = 更快。

**只能行内比**：每行 prompt 长度与设备都不同。

| 形态 | 不投机 | 开投机 | 相对（正=更快） | accept |
| --- | ---: | ---: | ---: | ---: |
| 跨机 head 本机 `0:32` + tail Y700，1522 token，64 token（两轮） | 7.42 / 8.71 | DFlash **7.41 / 8.29** | −0.1% ~ −4.8% | 0.692 |
| 同上 | 7.42 / 8.71 | **DSpark 8.70 / 9.79** | **+12.4% ~ +17.2%** | 0.740 |
| 跨机，39–46 token 短 prompt | 8.00–8.63 | DFlash 5.62 | **−35%** | 0.39 |
| 本机两段 2B，`-np 1`，192 token | 22.56 ms/tok | MTP 28.33 ms/tok | **−26%** | — |
| 跨机 ROG LAN，旧蒸馏版，五类 prompt | 8.18 | MTP 8.47 | +3.6%（中位 +6%） | — |

**跨机投机的收益由「接受率能不能赚回带宽」决定**：远程链开投机就走 whole-logits（248320 × 4 B
≈ 993 KB/轮），不投机是 top-20 候选（48 B/轮）。

短 prompt 上是负的（draft ctx 固定建仓 2.1 s，摊在 64 token 上 −25%）；长 prompt + DSpark 才转正
（tap 更少：5 个里 1 个在 head，DFlash 是 8 个里 2 个）。

**本机拆开是纯负收益**（2B 慢 26%）：没有跨机往返可省，两个进程还抢同一套 CPU 与带宽。机制见
[SPECULATION-DATA] §4。

链上接受率（本地两段，`n_predict 64`，3 prompt，[MODEL-SWAP] §11.4）：keep-KV **0.78767**
（115/146）、catch-up mode 2 **0.9098**（121/133，与整模逐位相同）；DFlash / DSpark 在跨机上与整模
逐 prompt 相同（117/141、117/140）。

**同一个 `--spec-type draft-mtp` 在整模与链上是两套路径。**

口径坑：accept 跨 `n_predict` 比较会翻号（同一对草稿源 n=64 差 +0.59 点、n=256 差 −0.94 点）；
`mean_len = 1 + accept × k`，跨 k 不可比；DSpark-6L 的 base 是 NVFP4 而非官方 BF16，比较只在
「这份文件 vs 那份文件」这一层成立。

## 代码与验收

- fork（v2.2 现行版）：`H:\llamacpp-c`，分支 `slimarc/v2.2`，HEAD `6d0fa16c9`（基底 `5676dfbe0` 之后 7 个提交，
  tag `b12342`），树 `61781c712`；相对基底 55 文件 / +21788 −238。
- [llamacpp-split.patch]：45 文件 / +18123 −237 / 20238 行 / 1077.2 KiB。纯分段链的完整补丁（协议
  v7 / v8 / v9 / v10 都在其中），机制清单见 [PATCHES]；不含 SLIM-ARC，可单独编译、单独通过全部验收。
- [llamacpp-slimarc.patch]：13 文件 / +3665 −1 / 3790 行 / 164.2 KiB。SLIM-ARC prefetch，默认
  关闭，`SLIM_ARC_PREFETCH=1` 才装。借用的只有这一块：专家驻留、KV 驱逐、I/O 调度器未接，见
  [SLIM-ARC-接入计划]。
- 叠加判据：`5676dfbe0` + 主补丁 → 树 `2541c9d29`；再叠增量 → 树 `61781c712`，与 HEAD 的树相同
  （`git write-tree` 校验，见 `scripts/export_patches.sh`）。
- 构建：[BUILD] 给出从 `5676dfbe0` + 补丁到二进制的实测命令、8 个 `build-*` 的开关矩阵与 CPU
  变体基线；改 `.h` 必须 `--clean-first`。
- 与上游（v2.1 rebase 口径）：16 个文件与 `09338b85e` 逐字节相同（rebase 前 0 个）。移除的 17 个文件改动属落后于
  上游——上游已含那些提交；`src/models/dflash.cpp` 因 device-buffer 诊断行重新进入变更集。清单见
  [PATCHES]「与上游 193 个提交的重合」。

验收判据是逐位 / 逐字节的 logits，不比生成的文字；下表是 v2.1 结题的结果（2026-11-12）。**v2.2 已重跑**（2026-11-13）为 `PASS=73 FAIL=0 SKIP=1`——SKIP 的是组 13 的 4 项（KVMem 构建未随 v2.2 rebase），组 13 齐备时 77/0/0：

| 套件 | 结果 | 判据 |
| --- | ---: | --- |
| `bash verify/verify.sh` | 95（十五组：1–8 组 22 + 组 9 2 + 组 10 20 + 组 11 8 + 组 12 1 + 组 13 4 + 组 14 20 + 组 15 18） | 逐位 / 逐字节；组 11 直接跑 `llama-server + 链 + -md` 产品路径，组 12 用 CUDA 构建验段级 offload，组 13 用 KVMem 构建（`build-kvmem2`，`LLAMA_KVMEM=ON`）验 KV 虚拟化，组 14 验远程链 `-np 2` 的 per-seq 投机四通道（判据 7–11），组 15 验远程链多序列的基础语义（判据 2–6） |
| `verify/verify_chain_death.py` | 36 | 120 s 读超时、链死而进程活、失败可见 |
| `verify/verify_seq_id.py` | 9 | 跨段 `seq_id` 与位置 |
| `verify/verify_mtp.sh` / `verify/verify_mtp_e2e.sh` | 6 / 6 | 链上 draft 与整模逐位、端到端逐字节 |
| `verify/verify_grammar_spec.sh` / `verify/verify_rbudget_spec.sh` | 23 / 36 | grammar 整帧回退、预算只收紧 |
| `verify/verify_seq_rm_exact.py --expect-known-fail` | 1 pass / 5 known-fail | 5 条是上游 hybrid 的已知基线，作为对照保留 |
| 纯分段版（只叠主补丁） | 52 + 36 | 44 文件版（树 `0c307d82a`）独立构建并跑过；最新版见下 |

**纯分段版的最新状态**：纯分段树 `586a7e7a1`（45 文件）在独立 worktree @ `09338b85e` + 主补丁
（`git write-tree` = `586a7e7a1`）上 `--clean-first` 全量构建，`verify.sh` **52/0/0**、
`verify_chain_death.py` **36/0**（2026-11-11）。上一版 `0c307d82a`（44 文件，2026-11-09）为同一套
验收。

v2 驱动链的验收（`python checks/verify_*.py` 与 `python -m unittest test_topology test_contract`）已随 v2
退场到 `slimarc/prototype`，本分支不再持有。`checks/` 下只剩三个 v2 时代的工具（`inspect_disk.py`、
`probe_bw_ceiling.c`、`verify_cli.py`），不是验收套件；v2.1 的验收在 `verify/`，v2.2 沿用。条目与结果见
[SPLIT-MEASUREMENTS] §6。

## 文档地图

`docs/` 按主题分目录，根下只留入口文档：

| 目录 | 装什么 |
| --- | --- |
| `docs/` | 入口：`QUICKSTART.md`（从零跑通一条链）、`REPORT.md`（结题总纲）、`OPEN-ISSUES.md`（遗留问题）、`README.en.md`（英文镜像） |
| `docs/report/` | 结题六章：`01-背景与目标.md` … `06-局限与结论.md` |
| `docs/design/` | 机制、协议与规范：ARCHITECTURE / PROTOCOL-V9 / STYLE / TESTING / DESIGN / DECISIONS / SLIM-ARC 接入（计划与障碍）/ LILICORR-轻量头 / EXLLAMAV3-GPU段候选 |
| `docs/deploy/` | 构建、部署与交付：BUILD / DEPLOYMENT / CROSS-MACHINE / THREE-SEGMENT / PACKAGING / DELIVERY / PATCHES / PATCHES-DETAIL / ANDROID-SURFACE-段接入评估 |
| `docs/measure/` | 实测：SPLIT-MEASUREMENTS / MEASUREMENTS / MODEL-SWAP / SPECULATION / SPECULATION-DATA / SPECULATION-MATRIX / SEGMENT-OFFLOAD / MEASUREMENT-PITFALLS / IGPU-SLOWDOWN |
| `docs/limits/` | 边界与代价：SEGMENTED-LIMITS / MEMORY-LIMITS / KV-CACHE / MOE-SPECULATION-ECONOMICS |
| `docs/issues/` | 缺陷：`ISSUE-*.md` 五篇（定位、根因、修复判据）与 `FIXES.md`（已修复登记） |
| `docs/legacy/` | v2 时代已退场或已归档：ANTI-ROT / INVENTORY / PRIMA-ANALYSIS / PRIMA-LESSONS / TUI / CHAT-ENGINE / AUTO-ASSIGN-PLAN / BASELINE / PROGRESS / CUT-TOOL / RESPONSES-API / SAMPLING |

最常查的几篇：

| 文件 | 管什么 |
| --- | --- |
| [ARCHITECTURE] | 层范围加载的三个机制、接力点、硬约束、什么算证据、与原生 RPC 的分界 |
| [PATCHES] | 四代补丁的构成与代际差异、rebase 到上游的记录、补丁全文 |
| [BUILD] | 从补丁到可执行文件：实测命令、开关矩阵、CPU 变体与公平基线、MSVC 坑、产物清单 |
| [SEGMENTED-LIMITS] | 每条边界的根因、影响范围、修的时候要付的代价 |
| [SPLIT-MEASUREMENTS] | v2.1 实测（v2.2 沿用）：checkpoint、MTP、DFlash、协议 v9、选型矩阵、跨机零损失 |
| [SPECULATION-DATA] | 投机实测明细：逐 prompt、k 扫描、DFlash、选型矩阵、未落盘数字 |
| [MODEL-SWAP] | 35B 换模型：两份权重、接受率基线、链上 13% 折扣 |
| [PROTOCOL-V9] | 协议 v5/v7/v8/v9：候选通道、验证行、远程 draft、tap、`seq_rm` 探针 |
| [CROSS-MACHINE] | 跨机实测：ROG、Surface、回环、Tailscale → Surface |
| [PATCHES-DETAIL] | 补丁细节：第四代机制表、device-buffer 修复、与上游重合明细 |
| [MEASUREMENTS] | v2 驱动链的实测数据（量测工具链已退场，归档在 `slimarc/prototype`；引用须注明口径） |
| [SPECULATION] | 投机判定、成本模型、k 扫描、接入实测 |
| [DEPLOYMENT] | 每台放什么、拓扑怎么写、起停与观察、段挂了怎么处置、跨机要配什么 |
| [TESTING] | 判据与测试通道：测什么、怎么跑、什么算过 |
| [MEASUREMENT-PITFALLS] | 段内计时为什么骗人，怎么量才有效 |
| [MEMORY-LIMITS] | 内存与显存的账、校准常数、绑核纪律 |
| [IGPU-SLOWDOWN] | 核显为什么拖慢、`-ngl` 扫描与交叉点 |
| [SEGMENT-OFFLOAD] | 分段下 `-ngl` 的语义、段级与端到端实测、MoE 缓存、显存预算 |
| [PROGRESS] | 当前能力快照与已做的优化 |
| [SAMPLING] · [TUI] · [RESPONSES-API] | 采样映射 · TUI 页面与命令 · `/v1/responses` 翻译层 |
| [SLIM-ARC-接入计划] | prefetch 之外的缺页优化在什么条件下才启用 |

其余专题（`LILICORR-轻量头.md`、`ANDROID-SURFACE-段接入评估.md`、`ISSUE-*.md` 等）按上表落在对应
子目录，已退场的 v2 文档在 `docs/legacy/`。

## 许可

MIT，见 [`LICENSE`](LICENSE)。上游 llama.cpp 同为 MIT，本仓库的补丁按同一许可分发；模型权重等
第三方产物随其各自来源。

[ARCHITECTURE]: docs/design/ARCHITECTURE.md
[PATCHES]: docs/deploy/PATCHES.md
[BUILD]: docs/deploy/BUILD.md
[SEGMENTED-LIMITS]: docs/limits/SEGMENTED-LIMITS.md
[SPLIT-MEASUREMENTS]: docs/measure/SPLIT-MEASUREMENTS.md
[SPECULATION-DATA]: docs/measure/SPECULATION-DATA.md
[MODEL-SWAP]: docs/measure/MODEL-SWAP.md
[PROTOCOL-V9]: docs/design/PROTOCOL-V9.md
[CROSS-MACHINE]: docs/deploy/CROSS-MACHINE.md
[PATCHES-DETAIL]: docs/deploy/PATCHES-DETAIL.md
[MEASUREMENTS]: docs/measure/MEASUREMENTS.md
[SPECULATION]: docs/measure/SPECULATION.md
[DEPLOYMENT]: docs/deploy/DEPLOYMENT.md
[TESTING]: docs/design/TESTING.md
[MEASUREMENT-PITFALLS]: docs/measure/MEASUREMENT-PITFALLS.md
[MEMORY-LIMITS]: docs/limits/MEMORY-LIMITS.md
[IGPU-SLOWDOWN]: docs/measure/IGPU-SLOWDOWN.md
[SEGMENT-OFFLOAD]: docs/measure/SEGMENT-OFFLOAD.md
[PROGRESS]: docs/legacy/PROGRESS.md
[SAMPLING]: docs/legacy/SAMPLING.md
[TUI]: docs/legacy/TUI.md
[RESPONSES-API]: docs/legacy/RESPONSES-API.md
[SLIM-ARC-接入计划]: docs/design/SLIM-ARC-接入计划.md
[llamacpp-split.patch]: docs/llamacpp-split.patch
[llamacpp-slimarc.patch]: docs/llamacpp-slimarc.patch
