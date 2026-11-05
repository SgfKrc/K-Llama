# K-Llama v2.1

**中文** | [English](docs/README.en.md)

> 状态：**现行**
> 更新日期：2026-11-08
> 能力状态：**可用**。`build-split/verify.sh` 九组 25 项全过，判据全是逐位 / 逐字节的 logits
> 对比，不比生成的文字。架构限定 qwen3.5 系（`qwen35` / `qwen35moe`）；远程链单序列。没验过的
> 那一处（多模态图像输入）写在边界里，不打包成结论。跨机投机自 2026-11-08 起是五类真实 prompt
> 的实测：中位约 **+6%**（+8.4% ~ −1.2%），弱机器与回环上是负的，判据也不是「与不投机一致」。

v2.1 是一次**下沉式深度改造**。上一代 v2 是浅层改造：Python 驱动 + 多进程协议，llama.cpp
只当哑计算。这一代把分段链做进 `llama_context`——`llama-server` 自己就是链的驱动者，它不
需要知道段在哪台机器上，`/completion`、`/chat/completions`、slot、prompt cache 照常工作。
目标是：在限定的 qwen3.5 系架构上（`qwen35` / `qwen35moe`，主力是 MoE 那一支），
**llama.cpp 该有的都有**。

一句话的分工：**v2 解决「模型怎么摊到多台机器上」，v2.1 解决「摊开之后能力还得跟整模一样」**。

---

## 两代的分野

| | v2（浅层改造） | v2.1（下沉） |
| --- | --- | --- |
| 改造层次 | Python 驱动 + 多进程协议，llama.cpp 只当哑计算 | 下沉到 `llama_context`；`llama-server` 原生支持 |
| 能用什么接口 | 自定义 chat 引擎（`chain_client.py`） | `llama-server` 的 `/completion`、`/chat/completions`、slot、prompt cache 全套 |
| 能力范围 | 只有 chat | 多 slot 并发、前缀复用 / checkpoint、MTP 与 DFlash 投机、远程段、KV 管理 |

### v2 的成果（数字都属于 v2）

v2 那条 Python 驱动链还在仓库里（`chain_client.py` / `stage_worker.py` / `stream.py`，
`koakuma` TUI 是它的入口），它解决的是「分布式 chat 引擎」。下面的数是**它的**，不是 v2.1 的：

| 配置 | 吞吐 | 口径 |
| --- | ---: | --- |
| 本机单卡整模 `-ngl 6` | 10.5 tok/s | 端到端 |
| 跨机两段 + DFlash k=5 | **20.9 tok/s** | 解码阶段、64-token 生成 |
| 两台弱设备两段（Surface + Y700，纯 CPU） | 6.50 tok/s | 解码阶段、24-token 生成 |
| 三段跨公网（Tailscale） | 5.00 tok/s | 解码阶段、24-token 生成 |

三段形态连跑 60 轮 `gen 256` 零异常。跨机比单机快出来的部分是投机挣的，**拆分本身是负
收益**（同一块显存摊给两段比单进程慢）。口径见 [`docs/MEASUREMENTS.md`](docs/MEASUREMENTS.md)、
[`docs/SPECULATION.md`](docs/SPECULATION.md) §11。

v2.1 不给吞吐对标：本机的 `build-split` 是纯 CPU 构建（启动即打印 `no usable GPU found`，
全部 `-ngl 0`），而双进程又测不出投机收益（见边界⑤）。这一代可比的数字有三个——前缀复用
（3103 token 的历史整段重算 19847 ms，第二轮只解 38 个新 token 是 **714 ms**）、跨机 decode 的
候选回传（每 token 352 B）、跨机投机（ROG 五类 prompt 上 MTP k=2 中位 **+6%**，区间
**+8.4% ~ −1.2%**）。口径：
[`docs/SPLIT-MEASUREMENTS.md`](docs/SPLIT-MEASUREMENTS.md) §1 / §8 / §9。

---

## v2.1 的能力

七项，每条都在验收里实测过：

| 能力 | 一句 | 怎么验的 |
| --- | --- | --- |
| 多 slot 并发 | `-np 4` 四并发的输出与串行**逐字节相同** | `verify.sh` 组 5、`accept_np.py --np 2` |
| 前缀复用 | chat 重渲染场景下，3103 token 历史第一轮 19847 ms，第二轮只解 38 个新 token、**714 ms** | `verify.sh` 链上 checkpoint、`reuselong` 日志 |
| MTP 投机 | 链上 draft 与整模 draft 的 logits **248320 维逐位相同** | `verify.sh` 组 8（链上 MTP） |
| DFlash | 8 个 `target_layers` 跨两段（head 6 + tail 2），单 ubatch 下 8 路 tap 与整模**逐位相同** | `verify.sh` 组 9（`dflash-taps` 目标） |
| 远程链 | `--split-remote host:port`；KV 广播、checkpoint 覆盖全链、120 s 超时；身份握手；decode 每 token 回 40 个候选而不再回整行 logits；链死后可自愈 | `verify.sh` 组 6 / 7；链路死亡与自愈走 `verify_chain_death.py` |
| 远程链的投机 | 协议 v7 让 tail 对验证帧每行回一个 argmax（单帧 993,312 B → 48 B），协议 v8 把 MTP draft 下沉到 tail；**能做**，但收益随负载从 +8.4% 到 −1.2%，弱机器与回环上是负的 | `verify_mtp.sh` / `verify_mtp_e2e.sh`；数据见 [`docs/SPLIT-MEASUREMENTS.md`](docs/SPLIT-MEASUREMENTS.md) §9 |
| 段的内存 | tail 的 logits 缓冲 1710.76 MiB → **3.79 MiB**；远程 stage 的 CPU compute buffer 2052.10 → **240.09 MiB** | 同一 ubatch 的 prefill 172.6 → 219.4 t/s |

前两项的收益不来自跨机往返，所以在单机上测也是真的。MTP / DFlash 两行说明**正确性**；
加速比只能按部署形态报：本机双进程测不出来（边界⑤），ROG 跨机五类 prompt 中位 +6%、区间
+8.4% ~ −1.2%，Surface 上是 −40%（§9）。

---

## 代码在哪

- llama.cpp fork：单提交 `c5bd787b`，父提交 `e0f142d`，树 `f455ed79`，**55 文件 / +16216 −229**。
- 主补丁 [`docs/llamacpp-split.patch`](docs/llamacpp-split.patch)：**45 个文件、+12672 −228、14557 行、
  748.9 KiB**——纯分段链（层范围加载 + 本地/远程段 + 聚合 memory + checkpoint + 协议 v7/v8 的
  投机通道 + grammar / rbudget 的兼容路径 + `seq_rm` 失败显式化），**不含 SLIM-ARC**。它能单独
  编译、单独通过全部验收。
- 可选增量 [`docs/llamacpp-slimarc.patch`](docs/llamacpp-slimarc.patch)：13 个文件、+3544 −1、
  3662 行、155.2 KiB——SLIM-ARC prefetch，默认关闭（`SLIM_ARC_PREFETCH=1` 才装）。不需要预取
  就不必应用它。
- 可重放：`e0f142d` + 主补丁 → 树 `b1646d6d`；再叠增量 → 树 `f455ed79`，与 fork HEAD 逐字节
  相同（`git write-tree` 校验）。叠加顺序就是先主补丁、后增量。
- 主仓库分支 `slimarc/segmented`，已推送。

机制怎么成立的（层范围进 `llama_model_params`、段间传 `n_embd` 个 float、隐藏态在
`output_norm` 之前交接）见 [`docs/PATCHES.md`](docs/PATCHES.md) 第四代与
[`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md)。

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

三个参数的分工：`--split-layer-range` 是**本进程**加载的层，链首必然是它；`--split-tail-range`
是同机尾段；`--split-remote` 可重复、按链序，与 `--split-tail-range` 互斥。`split-stage` 的
`--role` 与 `--rs-seq` 都必填——角色原来是从本地层数猜的，猜错不报错，只是答案错。链首的
`--split-rs-seq` 默认 0 也能服务；只在开投机时要抬高下限（一轮要回滚 `n_max` 个位置）。

远程链开投机（贪心，`k` 用 `--draft-max`；**tail 段必须在 `OP_HELLO` 里报 `has_nextn`**）：

```powershell
# tail 端照常起，不要加额外参数；它的 NextN 头就是 draft，k 步递归在 stage 内部跑。
# 弱机器（4C8T 之类）要把线程显式抬高：-t 同时设 n_threads 与 n_threads_batch，
# 两者默认都是 4、不会自动跟随；只设 n_threads 拿不到收益。
split-stage -m model.gguf --layer-range 32:40 --role tail --rs-seq 8 -c 8192 -t 8 --port 9332

# 链首：--spec-type draft-mtp + --draft-max 2。收益中位约 +6%（+8.4% ~ −1.2%），
# JSON 负载与弱机器上是负的，别按「快 15%」规划。
llama-server -m model.gguf --split-layer-range 0:32 -c 8192 `
  --split-remote 10.0.0.3:9332 --spec-type draft-mtp --draft-max 2
```

tail 报不出 `has_nextn`（模型没有 NextN 块、范围不到末层、或 MTP context 建不起来）时，server
打一行 WARN 并把 draft-mtp 从类型表里摘掉——链照常服务，只是不投机；同一次请求里其它投机类型
（如 `ngram-simple`）不受影响。**别用 ngram 测跨机收益**：它的接受率取决于 prompt 的重复性，
真实负载上常常一个 draft 都提不出来（`SEGMENTED-LIMITS.md` §2.2 有反例）。

单独跑 `--split-layer-range` 的两道误配都在加载后、建上下文前被拒：范围在模型中间结束、又没给
后续段（`0:20` 这种缺尾巴）；以及范围不从 0 开始（`20:24` 加载了输出头、会返回 200，但前 20
层从没算过——纯乱码）。第二道只能靠一个内部标记区分「用户手打」与「server 内部加载尾段」：
`common_params::split_range_is_stage`，判据读的是从模型读回的层范围，所以 `LLAMA_LAYER_RANGE`
那条入口也一起覆盖。残留边界是「标记由谁置真」而不是「范围是否合法」，见
[`docs/SEGMENTED-LIMITS.md`](docs/SEGMENTED-LIMITS.md) §2.6。

---

## 边界（六条）

1. **架构限定。** `embeddings_pre_norm` 只覆盖 `qwen35` / `qwen35moe` 两个架构（`src/models/`
   下 171 个模型实现文件里的 2 个）。别的架构建链时**直接拒绝**，不是静默降级——静默拿到
   post-norm 向量会让下一段多 norm 一次，输出通顺但数值偏移。
2. **远程链是单序列。** 这是设计决定，不是缺口：要并发就在 head 那台起多个 server、各走自己
   的链，或者改用本地链（本地链 `-np 4` 已验证逐字节相同）。
3. **远程链的投机能做，但收益随负载从 +8.4% 到 −1.2%。** 协议 v7 让 tail 对验证帧的每一行回
   一个 argmax（一帧 k+1 行是 4(k+1) 字节，单帧回复从 993,312 B 降到 48 B），协议 v8 又把 MTP
   draft 下沉到 tail（`OP_MTP_DRAFT`）；grammar 与 reasoning budget 各有一条兼容路径（前者被拒
   就**整帧回退**，后者预算耗尽只收紧不放宽）。ROG 五类 prompt 上 MTP k=2 是 **+3.6% / +8.4% /
   −1.2% / +1.5% / +10.5%**，中位约 +6%；Surface 上 **−40%**，本机回环带 grammar 也是负的。
   **采样式投机仍不做**——rejection 要比两边完整分布。**判据也不是「与不投机逐字节相同」**：
   llama.cpp 在 CPU + 量化上不承诺 batch invariance（上游把确定性限定在 RMSNorm 与 CUDA 的
   MatMul/Attention），所以判据是「无回退时与**整模投机**逐字节相同」。详见
   [`docs/SEGMENTED-LIMITS.md`](docs/SEGMENTED-LIMITS.md) §2.2。
4. **多模态没有端到端验证。** 4 个 M-RoPE section 跨段传递的代码在（`split_expand_pos`），但
   图像输入的端到端从未跑过——「图像位置能传」目前是代码级结论。出处与未验范围见
   [`docs/SPLIT-MEASUREMENTS.md`](docs/SPLIT-MEASUREMENTS.md) §7 末尾。
5. **投机加速比只能按部署形态看。** 本机双进程 head↔tail 走内存、没有跨机往返可省，测出来的
   0.6–0.9× 是退化配置的数——**不要把它当结论**，但也别反推「多机一定赚」。跨机五类 prompt
   实测（ROG，LAN，协议 v8）：中位 **+6%**，区间 +8.4% ~ −1.2%；弱机器（Surface）**−40%**，
   回环带 grammar 也是负的。在 27-token 的高确定任务上 k=1/2/3 分别是 10.11/10.20/9.76 tok/s
   ——k≥3 变宽后代价几乎全在 head（32 层），所以甜点随切点走、换链就要重测。见
   [`docs/SPLIT-MEASUREMENTS.md`](docs/SPLIT-MEASUREMENTS.md) §9。
6. **核显要用就全量 offload。** 单卡隔离后实测（Iris Xe、`GGML_VK_VISIBLE_DEVICES=0`、
   `-ngl 99`）：prefill 长 prompt **2.5–3.2×**、decode **+5~9%**；而 `-ngl 0`（加载了后端
   却一层不上）反而比纯 CPU 慢 41%，中间值也一律更差。**交叉点在 pp≈32–64**，短 prompt 的
   prefill 是核显唯一输的地方。35B（20.8 GiB）装不进共享堆（~15.4 GiB），所以本项目 35B
   主线仍只用纯 CPU；机制见 [`docs/IGPU-SLOWDOWN.md`](docs/IGPU-SLOWDOWN.md)。

完整版在 [`docs/SEGMENTED-LIMITS.md`](docs/SEGMENTED-LIMITS.md)，那里面写的是「为什么不修」和
「修的时候要付什么代价」，不是 TODO。六条之外还有一条 2026-11-08 补的：上游 hybrid 的 partial
`seq_rm` 在 server 默认配置（`n_rs_seq = 0`）下**永远失败**且 attention KV 一行都不删（上游
open PR #28007 / issue #31656），本仓只做「失败可见 + 五个调用点各自降级」，见该文 §2.9。
六条有个共同形状：**都不报错，只算错或悄悄退化**。所以验收比 logits 不比生成的文字（argmax
一致经常是巧合），跨 ubatch 的逐位对比还会假阳性。

---

## 与 SLIM-ARC 的关系

本项目的层段接力是自研的，与 [SLIM-ARC](https://github.com/Nexa-Language/SLIM-ARC)
不是同一层的东西。

SLIM-ARC 是中山大学参加 2026 年全国大学生计算机系统能力大赛操作系统设计赛 Proj59 的作品，
定位是**内存受限端侧系统上的 MoE 推理运行时**：mmap 按需加载、阶段感知页建议、
MoE 专家预测与回收、压力感知驻留、统一 I/O 预算与 KV cache 管理。它解决的是
"模型体积远大于物理内存"——代表结果是把 Qwen3-Next-80B-A3B 塞进 8 GiB cgroup，
decode 0.08 → 0.43 tok/s。

同一个问题的两条路径：

| | 本项目 | SLIM-ARC |
| --- | --- | --- |
| 做法 | **横向扩**：把层分给多台机器 | **纵向压**：单机按页换入换出 |
| 前提 | 每个节点装得下自己那段 | 单机装不下整模 |
| 交叉代价 | 8 KB hidden 过网络 | 缺页读盘 |
| 形态 | 多进程 + TCP | 单进程 + 页级调度 |

借用的只有它的 **prefetch 模块**——按层边界预读，把随机缺页变成顺序读。它以**可选增量补丁**
的形式存在（[`docs/llamacpp-slimarc.patch`](docs/llamacpp-slimarc.patch)，`SLIM_ARC_PREFETCH=1`
才装调度器），主补丁 [`docs/llamacpp-split.patch`](docs/llamacpp-split.patch) 不含它；已做
win32 移植。它的专家驻留、KV 驱逐、统一 I/O 调度器、on-demand 加载器都没有接。触发条件见
[`docs/SLIM-ARC-接入计划.md`](docs/SLIM-ARC-接入计划.md)，接入形态见
[`docs/PATCHES.md`](docs/PATCHES.md)。

---

## 兄弟仓库

同一套层段接力的思路在扩散模型上的实现是 [K-sd.cpp](https://github.com/SgfKrc/K-sd.cpp)：
那边上游（`stable-diffusion.cpp`）本来就是一个 transformer block 一个段地执行，补丁只补
「让某一段在别处执行」的入口；这边要从无到有造出段机制（改写 `block_count`、注入 hidden、
接管解码循环），这是两者的工作量差别所在。

代价方向相反：扩散的段间载荷是 MB 级张量，LLM 这边是 `n_embd` 个 float。但**同一条序列
内部，两边都不能流水**——扩散每步要等所有段算完，LLM 的相邻 token 之间是纯因果依赖
（`t_{p+1}` 由尾段产出，头段才可能算下一步）。真正的吞吐空间在**多请求**：不同序列互不
依赖。这条在 v2.1 的本地链上落了地（`-np 4`，逐字节相同）；远程链仍是单序列，见边界②。

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

验收：`bash build-split/verify.sh`（九组，逐位 / 逐字节）。逐位对照的三个目标是
`split-smoke`、`split-rows`、`dflash-taps`，组 9 需要 35B + DFlash 权重。

七个脚本 + 两个 seq_rm 专项，全部独立可重跑：`verify.sh` **25 项**、
`verify_chain_death.py` **36 项**（链路死亡与自愈，单独跑，约 12 分钟）、`verify_seq_id.py` **7 项**、
`verify_mtp.sh` **6 项**、`verify_mtp_e2e.sh` **6 项**、`verify_grammar_spec.sh` **22 项**、
`verify_rbudget_spec.sh` **36 项**；`seq_rm` 专项是 `verify_seq_rm_exact.py --expect-known-fail`
（A/B/C 三形态，当前 A 两红、B 三绿、C 一绿，退出 0）与 `verify_seq_rm_shared.py`（shared-prefix
形态，注入不触发时打 `[INFO] injection never fired`）。

v2 驱动链另有：`python checks/verify_*.py`，`python -m unittest test_topology test_contract`，
`checks/verify_equiv.py`（token 序列 / 同通道 hidden / 层号 slot）、`checks/verify_cli.py`
（llama-server 对照）。

---

## 许可

MIT，见 [`LICENSE`](LICENSE)。

上游 llama.cpp 同为 MIT，本仓库的补丁（`docs/PATCHES.md`）按同一许可分发；PDF、模型权重等
第三方产物的许可见其各自来源。
