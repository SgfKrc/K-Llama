# K-Llama v2.1

**中文** | [English](docs/README.en.md)

> 状态：**现行** · 更新日期：2026-11-09
> 能力：**可用**。`build-split/verify.sh` 十一组 52 项全过（组 1–8 共 22 项、组 9 本地 8 路 tap
> 2 项、组 10 远程 DFlash/DSpark 20 项、组 11 DFlash device-buffer 8 项），判据是逐位 / 逐字节的
> logits，不比生成的文字。限定 qwen3.5 系（`qwen35` / `qwen35moe`）与单序列远程链；多模态图像
> 输入未验（边界④）。
> 跨机投机：五类真实 prompt 上中位 **+6%**（+8.4% ~ −1.2%），弱机器与回环上是负的，判据也不是
> 「与不投机一致」。**但投机源要按通道挑**：MTP 在链上被 nextn 通道的 `!is_chain` 压到 0.688，
> 而 DFlash / DSpark 走 `layer_inp` tap，协议 v9 之后**远程链零损失**（边界⑤）。

K-Llama 让 llama.cpp 按层分段加载：每台机器只读自己那段，段间只传 `n_embd` 个 float。v2.1 把
这条链做进 `llama_context`——`llama-server` 自己就是驱动者，不需要知道段在哪台机器上，
`/completion`、`/chat/completions`、slot、prompt cache 照常工作。

一句话分工：**v2 解决「模型怎么摊到多台机器上」，v2.1 解决「摊开之后能力还得跟整模一样」**。

---

## 两代的分野

| | v2（浅层改造） | v2.1（下沉） |
| --- | --- | --- |
| 改造层次 | Python 驱动 + 多进程协议，llama.cpp 只当哑计算 | 下沉到 `llama_context`；`llama-server` 原生支持 |
| 能用什么接口 | 自定义 chat 引擎（`chain_client.py`） | `llama-server` 的 `/completion`、`/chat/completions`、slot、prompt cache 全套 |
| 能力范围 | 只有 chat | 多 slot 并发、前缀复用 / checkpoint、MTP 与 DFlash 投机、远程段、KV 管理 |

### v2 的成果（数字都属于 v2）

v2 的 Python 驱动链还在（`chain_client.py` / `stage_worker.py` / `stream.py`，入口 `koakuma` TUI）；
下面的数属于它，不属于 v2.1：

| 配置 | 吞吐 | 口径 |
| --- | ---: | --- |
| 本机单卡整模 `-ngl 6` | 10.5 tok/s | 端到端 |
| 跨机两段 + DFlash k=5 | **20.9 tok/s** | 解码阶段、64-token 生成 |
| 两台弱设备两段（Surface + Y700，纯 CPU） | 6.50 tok/s | 解码阶段、24-token 生成 |
| 三段跨公网（Tailscale） | 5.00 tok/s | 解码阶段、24-token 生成 |

跨机比单机快出来的部分是投机挣的，**拆分本身是负收益**（同一块显存摊给两段比单进程慢）；三段
形态连跑 60 轮 `gen 256` 零异常。口径见 [`docs/MEASUREMENTS.md`](docs/MEASUREMENTS.md)、
[`docs/SPECULATION.md`](docs/SPECULATION.md) §11。

### v2.1 的数字

`build-split` 是纯 CPU 构建（启动即 `no usable GPU found`，全部 `-ngl 0`），双进程又测不出投机
收益（边界⑤），所以 v2.1 只报四个相对数：

- **前缀复用**：3103 token 的历史整段重算 **19847 ms**，第二轮只解 38 个新 token 是 **714 ms**；
- **候选回传**：跨机 decode 每 token **352 B**（原整行 993 KB）；
- **跨机投机**：ROG 五类 prompt 上 MTP k=2 中位 **+6%**（+8.4% ~ −1.2%）；
- **MTP 接受率**（2026-11-08 换官方 `UD-Q4_K_M`）：整模 k=2 **0.910**，本地双段链 **0.688**，
  换模型零改代码。
- **投机选型**（2026-11-09，整模、k=8、`n_predict 256`）：**整模 MTP k=2 是短上下文的最优**
  （0.910 / 2.82）；长上下文里 **DSpark-6L 的 markov 头比 DFlash 高 15.3 / 14.5 点**（mid / long），
  但那是「可抄写性」的红利——翻译改写档只剩 +4.1 点。**链上走 tap 的是 DSpark-6L / DFlash**，
  不是 MTP（MTP 在链上被封在 0.688）。

口径：[`docs/SPLIT-MEASUREMENTS.md`](docs/SPLIT-MEASUREMENTS.md) §1 / §8 / §9 / §11 / §12 / §13。

---

## v2.1 的能力

七项，每条都在验收里实测过：

| 能力 | 一句 | 怎么验的 |
| --- | --- | --- |
| 多 slot 并发 | `-np 4` 四并发的输出与串行**逐字节相同** | `verify.sh` 组 5、`accept_np.py --np 2` |
| 前缀复用 | chat 重渲染下，3103 token 历史第一轮 19847 ms，第二轮只解 38 个新 token、**714 ms** | `verify.sh` 链上 checkpoint、`reuselong` 日志 |
| MTP 投机 | 链上 draft 与整模 draft 的 logits **248320 维逐位相同** | `verify.sh` 组 8（链上 MTP） |
| DFlash | 8 个 `target_layers` 跨两段（head 6 + tail 2），单 ubatch 下 8 路 tap 与整模**逐位相同**；远程链上 tail 那几路 tap 走协议 v9，**零损失** | `verify.sh` 组 9（`dflash-taps`）、组 10（远程 DFlash/DSpark） |
| 远程链 | `--split-remote host:port`；KV 广播、checkpoint 覆盖全链、120 s 超时、身份握手、decode 回 40 个候选、链死自愈 | `verify.sh` 组 6 / 7；`verify_chain_death.py` |
| 远程链的投机 | v7 验证通道（单帧 993,312 B → 48 B）+ v8 把 MTP draft 下沉到 tail + **v9 让 DFlash/DSpark 的 layer_inp tap 随 decode 回复跨进程**；能做，收益见边界⑤ | `verify_mtp.sh` / `verify_mtp_e2e.sh`；`verify.sh` 组 10 |
| 段的内存 | tail 的 logits 缓冲 1710.76 MiB → **3.79 MiB**；远程 stage 的 CPU compute buffer 2052.10 → **240.09 MiB** | 同一 ubatch 的 prefill 172.6 → 219.4 t/s |

前两项的收益不来自跨机往返，单机上也是真的；MTP / DFlash 两行说明**正确性**，加速比只能按
部署形态报（边界⑤）。

---

## 代码在哪

- llama.cpp fork：单提交 `0c5c3bdd`，父提交 `f477f795c`（上游 193 个提交之后的基底），树
  `982ccecd`，**52 文件 / +17831 −230**。旧基线 `ec25cedda`（父 `e0f142d`，树 `e14b92c9`，
  68 文件）留在 `master` 上作历史。
- 主补丁 [`docs/llamacpp-split.patch`](docs/llamacpp-split.patch)：**42 文件、+14287 −229、16166 行、
  840.6 KiB**——纯分段链（层范围加载、本地/远程段、聚合 memory、checkpoint、协议 v7 验证通道 /
  v8 远程 MTP draft / v9 远程 layer_inp tap、grammar / rbudget 兼容、`seq_rm` 失败显式化、
  device-buffer 宽度判定），**不含 SLIM-ARC**；可单独编译、单独通过全部验收。
- 可选增量 [`docs/llamacpp-slimarc.patch`](docs/llamacpp-slimarc.patch)：13 文件、+3544 −1、3662 行、
  155.2 KiB——SLIM-ARC prefetch，默认关闭（`SLIM_ARC_PREFETCH=1` 才装）。
- 可重放：`f477f795c` + 主补丁 → 树 `d28e10df`；再叠增量 → 树 `982ccecd`，与 fork HEAD 逐字节
  相同（`git write-tree` 校验）。

机制（层范围进 `llama_model_params`、段间传 `n_embd` 个 float、隐藏态在 `output_norm` 之前
交接）见 [`docs/PATCHES.md`](docs/PATCHES.md) 第四代与 [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md)。
---

## 怎么跑

本地两段（tail 在同一进程里，最省事）：

```powershell
llama-server -m model.gguf --split-layer-range 0:32 --split-tail-range 32:40 -c 8192 -np 4
```

远程链：head 那台起 `llama-server`，其余每台起一个 `split-stage`，每个进程只加载自己那段。

```powershell
# 远端 relay：源层 24:32，只出 hidden
split-stage -m model.gguf --layer-range 24:32 --role relay --rs-seq 8 -c 8192 --host 0.0.0.0 --port 9331

# 远端 tail：源层 32:40（必须到末层），出 logits
split-stage -m model.gguf --layer-range 32:40 --role tail --rs-seq 8 -c 8192 --host 0.0.0.0 --port 9332

# 链首：本机加载 0:24，远程段按链序列出
llama-server -m model.gguf --split-layer-range 0:24 -c 8192 `
  --split-remote 10.0.0.2:9331 --split-remote 10.0.0.3:9332
```

`--split-layer-range` 是本进程加载的层（链首必然是它）；`--split-tail-range` 是同机尾段；
`--split-remote` 可重复、按链序，与 `--split-tail-range` 互斥。`split-stage` 的 `--role` 与
`--rs-seq` 都必填——角色从前是猜的，猜错不报错、只出错。`--split-rs-seq` 默认 0 也能服务，只在
开投机时要抬高下限（一轮要回滚 `n_max` 个位置）。

远程链开投机（贪心，`k` 用 `--draft-max`；**tail 段必须在 `OP_HELLO` 里报 `has_nextn`**）：

```powershell
# tail 照常起：NextN 头就是 draft，k 步递归在 stage 内部跑。
# 弱机器（4C8T 之类）要显式抬高线程：-t 同时设 n_threads 与 n_threads_batch，两者默认都是 4。
split-stage -m model.gguf --layer-range 32:40 --role tail --rs-seq 8 -c 8192 -t 8 --port 9332

# 链首：收益中位 +6%（+8.4% ~ −1.2%），JSON 负载与弱机器上是负的，别按「快 15%」规划。
llama-server -m model.gguf --split-layer-range 0:32 -c 8192 `
  --split-remote 10.0.0.3:9332 --spec-type draft-mtp --draft-max 2
```

tail 报不出 `has_nextn` 时 server 打 WARN 并把 draft-mtp 摘掉——链照常服务，只是不投机。**别用
ngram 测跨机收益**：命中取决于 prompt 重复性，真实负载上常常提不出 draft（`SEGMENTED-LIMITS.md` §2.2）。

单独跑 `--split-layer-range` 的两种误配都在建上下文前被拒：中间结束又没给后续段（`0:20`）、
不从 0 开始（`20:24` 返回 200 但前 20 层没算过——纯乱码）。残留边界见
[`docs/SEGMENTED-LIMITS.md`](docs/SEGMENTED-LIMITS.md) §2.6。

---

## 边界（六条）

1. **架构限定。** `embeddings_pre_norm` 只覆盖 `qwen35` / `qwen35moe`（`src/models/` 下 171 个
   模型实现文件里的 2 个）。别的架构**建链即拒**，不是静默降级——静默拿到 post-norm 向量会让
   下一段多 norm 一次，输出通顺但数值偏移。
2. **远程链是单序列。** 设计决定，不是缺口：要并发就在 head 那台起多个 server、各走自己的链，
   或改用本地链（`-np 4` 已验证逐字节相同）。
3. **远程链的投机能做。** v7 让 tail 对验证帧每行回一个 argmax（单帧回复 993,312 B → 48 B），
   v8 把 MTP draft 下沉到 tail，v9 让 DFlash/DSpark 的 `layer_inp` tap 随 decode 回复跨进程；
   grammar 被拒**整帧回退**，reasoning budget 耗尽只收紧。
   **采样式投机不做**。判据是「无回退时与**整模投机**逐字节相同」——CPU + 量化上 llama.cpp 不
   承诺 batch invariance。数字见⑤，细节见 [`docs/SEGMENTED-LIMITS.md`](docs/SEGMENTED-LIMITS.md) §2.2。
4. **多模态没有端到端验证。** 4 个 M-RoPE section 跨段传递的代码在（`split_expand_pos`），图像
   输入的端到端从未跑过——「图像位置能传」是代码级结论。见
   [`docs/SPLIT-MEASUREMENTS.md`](docs/SPLIT-MEASUREMENTS.md) §7 末尾。
5. **投机加速比只能按部署形态看。** 本机双进程的 0.6–0.9× 是退化配置，别反推「多机一定赚」。
   ROG 五类 prompt 中位 **+6%**（+3.6 / +8.4 / −1.2 / +1.5 / +10.5%），Surface **−40%**，回环
   带 grammar 也是负的。k 甜点随切点走（27-token 任务上 k≥3 的代价几乎全在 head），换链要重测。
   **投机源还要按通道挑**：MTP 在链上被 nextn 通道的 `!is_chain` 分支封在 0.688，DFlash / DSpark
   走 `layer_inp` tap，v9 之后远程链零损失；markov 头（DSpark-6L）的优势是「可抄写性」而不是
   长距离依赖（k=8 时 mid/long 高 15.3 / 14.5 点，翻译改写档只剩 4.1 点）。
   见 [`docs/SPLIT-MEASUREMENTS.md`](docs/SPLIT-MEASUREMENTS.md) §9 / §12 / §13。
6. **核显要用就全量 offload。** 单卡隔离实测（`-ngl 99`）：prefill 长 prompt **2.5–3.2×**、
   decode **+5~9%**；`-ngl 0` 反而比纯 CPU 慢 41%，中间值一律更差，**交叉点在 pp≈32–64**。
   35B（`UD-Q4_K_XL` 20.8 GiB；MTP 基线用官方 `UD-Q4_K_M` 21.11 GiB，§11）装不进共享堆
   （~15.4 GiB），35B 主线仍用纯 CPU。见 [`docs/IGPU-SLOWDOWN.md`](docs/IGPU-SLOWDOWN.md)。

完整版在 [`docs/SEGMENTED-LIMITS.md`](docs/SEGMENTED-LIMITS.md)，写的是「为什么不修」与代价，
不是 TODO。六条之外还有一条 2026-11-08 补的：上游 hybrid 的 partial `seq_rm` 在默认
`n_rs_seq = 0` 下永远失败且 attention KV 一行不删（上游 #28007 / #31656），本仓只做「失败可见 +
五个调用点各自降级」，见该文 §2.9。共同形状：**都不报错，只算错或悄悄退化**——所以验收比
logits，不比生成的文字。

---

## 与 SLIM-ARC 的关系

本项目与 [SLIM-ARC](https://github.com/Nexa-Language/SLIM-ARC) 不是同一层的东西。SLIM-ARC 是
中山大学参加 2026 年全国大学生计算机系统能力大赛 Proj59 的作品：内存受限端侧的 MoE 运行时
（mmap 按需加载、专家预测与回收、统一 I/O 预算 + KV 管理），代表结果是把 Qwen3-Next-80B-A3B
塞进 8 GiB cgroup（decode 0.08 → 0.43 tok/s）。

| | 本项目 | SLIM-ARC |
| --- | --- | --- |
| 做法 | **横向扩**：把层分给多台机器 | **纵向压**：单机按页换入换出 |
| 前提 | 每个节点装得下自己那段 | 单机装不下整模 |
| 交叉代价 | 8 KB hidden 过网络 | 缺页读盘 |
| 形态 | 多进程 + TCP | 单进程 + 页级调度 |

借用的只有 **prefetch 模块**（按层边界预读，把随机缺页变成顺序读），以可选增量补丁存在
（[`docs/llamacpp-slimarc.patch`](docs/llamacpp-slimarc.patch)，`SLIM_ARC_PREFETCH=1` 才装），
主补丁不含它；已做 win32 移植。专家驻留、KV 驱逐、I/O 调度器、on-demand 加载器都没接，见
[`docs/SLIM-ARC-接入计划.md`](docs/SLIM-ARC-接入计划.md)。

---

## 兄弟仓库

同一套思路在扩散模型上的实现是 [K-sd.cpp](https://github.com/SgfKrc/K-sd.cpp)：那边上游本来就
一个 block 一段地执行，补丁只补「让某一段在别处执行」的入口；这边要从无到有造段机制（改写
`block_count`、注入隐藏态、接管解码循环），工作量差别在这里。

代价方向相反：扩散的段间载荷是 MB 级张量，LLM 是 `n_embd` 个 float；但**同一条序列内部两边都
不能流水**（扩散每步等所有段，LLM 相邻 token 纯因果依赖）。吞吐空间在**多请求**：v2.1 的本地链
已落地（`-np 4` 逐字节相同），远程链仍是单序列（边界②）。

---

## 文档

| 文件 | 讲什么 |
| --- | --- |
| [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) | **核心机制**：凭什么非整模加载、硬约束、判据、与 RPC 的分界 |
| [`docs/PATCHES.md`](docs/PATCHES.md) | 四代补丁的差异（第四代现行：分段链与跨进程）、补丁全文 |
| [`docs/SEGMENTED-LIMITS.md`](docs/SEGMENTED-LIMITS.md) | **v2.1 的边界**：每条为什么不修、修要付什么代价 |
| [`docs/SPLIT-MEASUREMENTS.md`](docs/SPLIT-MEASUREMENTS.md) | **v2.1 的实测**：checkpoint、MTP、DFlash、并发、内存、验收覆盖，带日志出处 |
| [`docs/MEASUREMENTS.md`](docs/MEASUREMENTS.md) | v2 驱动链的全部实测数据 |
| [`docs/SPECULATION.md`](docs/SPECULATION.md) | 投机判定、成本模型、k 扫描（§11）与接入实测 |
| [`docs/DEPLOYMENT.md`](docs/DEPLOYMENT.md) | 每台放什么、换分段怎么做 |
| [`docs/MEASUREMENT-PITFALLS.md`](docs/MEASUREMENT-PITFALLS.md) | 段内计时为什么骗人 |
| [`docs/MEMORY-LIMITS.md`](docs/MEMORY-LIMITS.md) | 内存与显存的账怎么算 |
| [`docs/SLIM-ARC-接入计划.md`](docs/SLIM-ARC-接入计划.md) | 缺页优化在什么条件下才启用 |
| [`docs/LILICORR-轻量头.md`](docs/LILICORR-轻量头.md) | DFlash 的块内一致性重排，以及训练成本 |
| [`docs/ANDROID-SURFACE-段接入评估.md`](docs/ANDROID-SURFACE-段接入评估.md) | 弱设备接入的条件与安卓侧硬约束 |
| [`docs/IGPU-SLOWDOWN.md`](docs/IGPU-SLOWDOWN.md) | **为什么核显拖慢**：host buffer 抢占、往返拷贝、共享总线；含 `-ngl` 扫描实测 |
| [`docs/SAMPLING.md`](docs/SAMPLING.md) · [`docs/TUI.md`](docs/TUI.md) | 采样映射 · TUI 设计 |
| [`docs/ISSUE-recurrent-snapshot-rollback.md`](docs/ISSUE-recurrent-snapshot-rollback.md) | hybrid 模型跨 ubatch 回滚缺陷与修复 |
| [`docs/PROGRESS.md`](docs/PROGRESS.md) | 当前能力快照与已做的优化 |

验收：`bash build-split/verify.sh`（十一组，逐位 / 逐字节；组 9/10/11 需要 35B + DFlash 权重），
逐位对照的目标是 `split-smoke`、`split-rows`、`dflash-taps`，组 11 直接跑「llama-server + 链 +
`-md`」这条产品路径；各脚本的项数与最近结果见
[`docs/SPLIT-MEASUREMENTS.md`](docs/SPLIT-MEASUREMENTS.md) §6 与 §9.7。`seq_rm` 专项两件：
`verify_seq_rm_exact.py --expect-known-fail`、`verify_seq_rm_shared.py`。

v2 驱动链另有 `python checks/verify_*.py`、`python -m unittest test_topology test_contract`、
`checks/verify_equiv.py`、`checks/verify_cli.py`。

---

## 许可

MIT，见 [`LICENSE`](LICENSE)。

上游 llama.cpp 同为 MIT，本仓库的补丁（`docs/PATCHES.md`）按同一许可分发；PDF、模型权重等
第三方产物的许可见其各自来源。
