# 机制：非整模加载凭什么成立

> 状态：**现行**
> 更新日期：2026-11-12
>
> 适用范围：不切模型怎么分段加载、段之间传什么、哪些约束是硬的；读者为维护或移植该机制的开发者。
> 相关：补丁改动清单 [`PATCHES.md`](../deploy/PATCHES.md)、实测数字 [`MEASUREMENTS.md`](../measure/MEASUREMENTS.md)、
> 投机判定 [`SPECULATION.md`](../measure/SPECULATION.md)。

---

## 0. 一句话

**每个节点都持有同一份整模文件，加载时只创建、只读取属于自己那几层的张量。** 切分只体现为加载范围，工件这一层被省掉了。

---

## 1. 为什么整模可以不整读

GGUF 的内部布局是：一段元数据（键值对）+ 若干个 tensor，每个 tensor 的数据在文件里是 **连续区间**，起始偏移写在元数据里。

llama.cpp 用 mmap 把整个文件映射进地址空间。**映射不等于读入**——操作系统只在页被访问时才从磁盘取。所以"持有整模"和"把整模读进内存"本来就是两件事。

本项目把第二步也省掉：让 llama.cpp 在**加载阶段**就认为这个模型只有 `hi-lo` 层，于是它只为这几层创建 tensor，其余层的字节从头到尾没有被 touch。

---

## 2. 三个机制

### 2.1 层数覆盖：`LLAMA_LAYER_RANGE=lo:hi`

`llama-model-loader.cpp` 在元数据解析之后、`load_hparams()` 之前改写 `block_count`（以及 `nextn_predict_layers`）。架构读到的 `n_layer` 就是 `hi-lo`，它所有的循环都只在这几层里转。

元数据里同时保留原始层数，因为"这段是不是末段"要靠它判断（末段才保留 `output` 与 MTP 头）。

### 2.2 名字重映射：本地 `blk.i` → 源文件 `blk.lo+i`

`get_weight` 在校验名表**之前**先做重映射。这一步不能省：`lo > 0` 的段如果不重映射，会读到源文件的 `blk.0`——输出仍然通顺，只是错的。

### 2.3 全局张量按需 lazy

`token_embd`（515 MiB，Q8_0）只有首段要，`output`（398 MiB，Q6_K）只有末段要，中段两个都不要。不创建就不读：中段因此省 **~913 MiB**。

判断依据是 `tn.tensor` 而不是名字——tied 模型会把 `output` 建成 `token_embd` 的副本，名字说是 embedding，实际承担投影。

三者合起来：任何一个节点都没有"整模加载"这一步，只有"我这几层"。

---

## 3. 接力向量：必须在 `output_norm` 之前

上游的 `t_embd` 是 `build_norm` **之后**的值（pooling 要归一化后的结果）。接力要的是原始输出：多过一次 RMSNorm，第一个 token 就分叉。

`embeddings_pre_norm` 开关把 tap 点前移，顺带不建 lm_head——中段因此再省 398 MiB。

---

## 4. 段间传什么

一个 `n_embd` 宽的 float32 向量：2048 × 4 B = **8 KB**。

**为什么不用 fp16 / int16**：2B 两段实测（5 个贪心 prompt × 48 token），fp16 有 1 个在第 22 步换了 token，int16 定点同样 1/5（第 16 步）——量化步长细 20 倍也照样翻转，因为分叉点都落在输出接近平局的位置。

判据是**逐位一致**；"误差足够小"不算判据。条件见 [`MEASUREMENTS.md`](../measure/MEASUREMENTS.md) §8.1。

协议是定长头 + 原始字节（`src/llama-split-proto.h`）：`req_header` 带 `n_tokens` / `n_embd` / `n_pos` / `n_bytes`，随后是 `n_tokens × n_pos` 个 int32 位置、再是 `n_tokens × n_embd` 个 float 的 hidden。

位置必须全量下发：M-RoPE 每 token 4 个 section，只传标量会让后续段算出不同的旋转编码。

协议 v4 起，每条连接的第一帧是 `OP_HELLO` 身份握手（`stage_id`：层范围、源层数、宽度、vocab、role）；v5 起，tail 的回复可以只是 top-k 候选而不是整行 logits（普通 decode 默认走这条）。

---

## 5. 段内怎么 offload

两条互不干扰的路径：

| 参数 | 语义 | 什么时候用 |
| --- | --- | --- |
| `-ngl N` | 前 N 层的**全部**张量上卡 | 层少、显存够 |
| `-ncmoe N` | 前 N 层的**专家**留 CPU，其余专家上卡 | MoE |

MoE 的权重分布是专家占 88%，而每个 token 只用到 8/256 个专家——**体积最大、访问最稀疏的部分进最快的设备**收益最大。实测同一台机器：整层 11 层 10.11 tok/s，专家 8 层 28.47 tok/s（口径见 [`MEASUREMENTS.md`](../measure/MEASUREMENTS.md) §3）。

段内怎么 offload 与段间怎么切是正交的，这是本设计唯一的抽象。

---

## 6. 切分的硬约束

- 所有切点必须是 `full_attention_interval`（qwen35moe 是 4）的倍数
- 末段的 `hi` 必须等于模型总层数
- v2 的 `topology.py` 在解析阶段就拒绝，不自动纠正（该文件已归档，v2.1 的检查在下沉后的 C 侧）

原因：`src/models/qwen35moe.cpp:24` 用**本地层号**推算层类型，而本地层号被映射回源层号。只有 `lo` 是 4 的倍数时，两者模 4 同余，层类型序列才对得上。 `32:38` 这种 6 层划分会在解析阶段报 `layers=6 is not a multiple of 4`。

严格说 v2 的 `topology.py` 查的是**段宽**（`s.layers % 4`），不是切点偏移；但在「每段都声明 `layer_range`、宽度必须等于 `layers`、各段之和必须等于总层数」三条同时成立时，两者等价，所以没有漏网。

llama.cpp 自己不做这个检查——`LLAMA_LAYER_RANGE` 只校验 `lo >= 0 && hi > lo`。

**错位的失败方式是加载期直接报错**：本地层 `i` 被当成与源层 `lo+i` 同类型，于是它去要 `blk.<lo+i>.ssm_conv1d.weight`（或反过来要 `attn_q_norm`），而那个张量在源层不存在、且没有 `TENSOR_NOT_REQUIRED` 标记，`llama-model-loader.cpp` 走到 `throw "missing tensor"`。

2026-09-18 的裁层实验里 `N=2` 就是这样报的 `missing tensor 'blk.1.ssm_conv1d.weight'`。这是这条约束唯一的优点：从不静默。

**要解除的话**（评估过，暂不做）：把 `is_recr_impl[i]` 的取模从本地号换成源层号—— `ml.layer_lo` 在 `load_hparams()` 时已可见（`llama_model_loader` 是 struct，字段全 public），改 `qwen35moe.cpp` 与 `qwen35.cpp` 各两行即可；下游的 KV／recurrent 分配、graph 分派全走 `is_recr()` 这一真源，自动跟着对。

不做的理由是收益太小：可选切点从 9 个变成 39 个、最小台阶约 0.43 GiB，但 full 层在源模型里均匀固定（`attn_q.weight` 只出现在 `3,7,…,39`），任何切分都改不了「每 4 层必有 1 个 full」。那点内存收益会被多一个段边界的跨网往返吃掉。真遇到「某台设备必须 ≤3 层」时，先试 `-ncmoe`。

---

## 7. 投机怎么接进来

两个草稿源，都插在验证器前面：

| 源 | 是什么 | 深度 |
| --- | --- | --- |
| MTP | 模型自带的 NextN 头，递归喂自己的输出 | 随模型：35B 上 k=2 是经验上限（第 3 层掉到 36%），2B 上 k=3 优于 k=2 |
| DFlash | 独立的小模型，一次前向出 16 个位置 | k=5 在盈利区间（k=8 起接受率崩塌） |

深度是**实测值，不是结构上限**：2B 上 k=3 的接受率只从 97.7% 掉到 96.6%，所以「k=2 是上限」只对 35B 成立。链上默认不开投机，开之前按负载选源和 k。

**本机双进程测出的投机收益不能外推到跨机**：head↔tail 走内存与管道，没有 25–30 ms 跨机往返可省，两进程还抢同一套 CPU 与内存带宽，付出成本拿不到好处。跨机收益随负载变号，数字见 [`SPLIT-MEASUREMENTS.md`](../measure/SPLIT-MEASUREMENTS.md) §9。

**两条链的接法不同，结论不能互搬**：

- **llama-server 原生路径**：draft context 建在 **tail 段**的模型上。head 段的加载范围
  `nextn = 0`，NextN 头只在 tail 里；把 draft 建在 head 上会在启动阶段失败。DFlash 反过来
  要 head——它的词表与 embedding 在前段——8 路 tap 按 head/tail 拆成（context, 局部层号）
  解析，tail 那两路是进程内读，不增加往返。

  **远程链也能投机，收益随负载走，判据换了一条**：协议 v7 让 tail 对验证帧的每行回一个
  argmax（一帧 k+1 行是 4(k+1) B，单帧从 993,312 B 降到 48 B），协议 v8 又把 MTP draft 下沉到
  tail（`OP_MTP_DRAFT`，k 步递归在 stage 内跑，一帧只回 k 个 id）；grammar 与 reasoning budget
  各有一条兼容路径（被拒就整帧回退 / 预算耗尽只收紧）。ROG 五类 prompt 上 MTP k=2 中位约
  **+6%**（区间 +8.4% ~ −1.2%），Surface 上 **−40%**，逐 prompt 数字见
  [`SPLIT-MEASUREMENTS.md`](../measure/SPLIT-MEASUREMENTS.md) §9。严格判据是「无回退时与**整模投机**
  逐字节相同」——llama.cpp 在 CPU + 量化上不承诺 batch invariance
  （[`SEGMENTED-LIMITS.md`](../limits/SEGMENTED-LIMITS.md) §2.2）。
  **DFlash / DSpark 在远程链上零损失（协议 v9）**：tails 那两路 `layer_inp` tap 作为 decode 回复
  的附带段过网（请求带源层号，回复在最后追加一个 `n_rows × n_taps × n_embd` 的 f32 段），于是
  链上不再只买得到被 `!is_chain` 压住的 MTP；代价是 prefill 时的传输量（512 行 ubatch = 8 MB /
  4 MB），见 [`SPLIT-MEASUREMENTS.md`](../measure/SPLIT-MEASUREMENTS.md) §12。
- **qlh 的 Python 驱动链（v2，已退场）**：`chain_client.py` 的 `--spec-k` / `--dflash` /
  `--cascade-mtp`，验证由 `forward_batch` 一次绕链完成，只支持贪心；文件在 `slimarc/prototype`。

共同要求：**每一步必须能回滚**。验证块写 k+1 行，只保留接受的前缀，剩下的要退回去。 hybrid 模型（30 层 linear + 10 层 full attention）的 recurrent state 没有位置索引，退回需要按 ubatch 滚动的快照，这是本项目代价最高的一处实现，见 [`ISSUE-recurrent-snapshot-rollback.md`](../issues/ISSUE-recurrent-snapshot-rollback.md)。

采样：llama-server 的原生实现的验证器已经支持 rejection sampling（草稿侧按目标温度重采样，把 q 交给验证器），qlh 链这条路径还只支持贪心。shim 已能按行回传完整 logits （`qlh_kh_logits`，v2 的 `shim/qlh_keep_head.def`，已退场），缺的是一次判据实验——同一位置上整模与链路的 top-k 候选是否重合，v2 的探针 `slimarc/prototype:checks/probe_logits_delta.py` 也已归档。

---

## 8. 与原生 RPC 的分界

| 维度 | 原生 RPC | 本项目 |
| --- | --- | --- |
| 权重从哪来 | 主机通过 `SET_TENSOR` 发到远端 | 各节点本地读整模的对应层 |
| 首次跨机流量 | 权重总量（GB 级） | 0 |
| 主机必须装下整模 | 是 | 否 |
| 每步流量 | 激活 + 图指令 | 8 KB hidden |
| 计算图 | 主机维护全局图 | 每段独立进程，无全局图 |
| 异构 | 需要相同 ggml backend | 自定义 TCP，可跨不同 CPU/GPU |

一句话：RPC 是**把设备虚拟化**，模型仍然一份；本项目是**把模型分段**，每段各自持有。代价是段内不能跨机器拆算子，每段必须能单独装下。

---

## 9. 什么算证据

| 级别 | 例子 |
| --- | --- |
| **硬判据** | 贪心 token 与整模逐位一致；段间 hidden `max\|diff\| = 0` |
| 参考 | 吞吐、接受率（要带口径：模型、量化、生成长度、设备） |
| 不算 | "输出看起来通顺"、单次运行、跨配置比较 |

跨机同样验过逐位一致。判据脚本按这个标准写：v2.1 的在 `verify/`（77 项十四组），v2 时代的 24 个在
`slimarc/prototype`（已退场）。

**投机要单独记一条**：它的硬判据是「链上投机与**整模投机**逐字节相同」，不是「与不投机逐字节相同」——后者测的是 batch invariance，而 llama.cpp 只对 RMSNorm（全 backend）与 CUDA 的 MatMul/Attention 承诺它，CPU + 量化不在覆盖范围内。带回退时只能保证到「漂移翻掉的第一个近似并列处」（[`SEGMENTED-LIMITS.md`](../limits/SEGMENTED-LIMITS.md) §2.2）。

---

## 10. SLIM-ARC 在哪一层

它不改 §2–§6 的任何一条机制，只改"某段装不下、真的在读盘时"的**读盘顺序**：把随机缺页变成按层边界的顺序预读。

装得下的时候它是空操作——整模在页缓存里，没有可补的页，实测 `issued = 0`。接入形态与触发条件见 [`PATCHES.md`](../deploy/PATCHES.md) 与 [`SLIM-ARC-接入计划.md`](SLIM-ARC-接入计划.md)。

## 11. 为什么没有 token 间流水

本节说的是**相邻 token 之间**的流水：`t_{p+1}` 还没生成时，没有任何办法开始算 p+1。跨段投机把一个 batch 里的 k+1 个位置一次算完，正是这里唯一能重叠的形态（§7）；多请求在最后一行，有条件可行。

链是严格串行的：v2 的 `chain_client._walk`（`chain_client.py:1141`，已退场）就是 `for i, client in enumerate(clients)` 逐段同步调用（`:1158`），每段都阻塞在 `_forward` 的 `send`/`read` 上（`:519`）；v2.1 的转发同样逐段串行。一次 decode 的墙钟 ≈ 各段耗时**之和**。

这条串行是单序列 decode 的因果关系，不是排程偷懒：

```
head(p)   ← t_p                 → h_p
mid(p)    ← h_p                 → h'_p
tail(p)   ← h'_p                → t_{p+1}
head(p+1) ← t_{p+1}   ←── 只可能由 tail(p) 产出
```

要把 `tail(p)` 与 `head(p+1)` 重叠，得先知道 `t_{p+1}`，而它正是 `tail(p)` 的输出。依赖图是一条 3N 个节点的单链，没有第二条独立路径可以填：token 间流水在单序列上没有并行空间。

（该段结论与扩散模型的段间依赖同构，但那是另一个项目的事，不在此展开。）

**能重叠的只有相互独立的工作单位**：

| 方向 | 结论 | 机制与代价 |
| --- | --- | --- |
| 段内批位置（已做） | 收益已被吃掉 | 投机一轮把 k+1 个位置打进同一个 batch，每段 33–35 ms 的固定开销只付一次（**v2 时代的实测**；v2.1 反推是 10–20 ms，见 [`THREE-SEGMENT.md`](../deploy/THREE-SEGMENT.md) §5），每多一行边际成本 0.178 个单行（[`SPECULATION.md`](../measure/SPECULATION.md) §0）。这是单序列上唯一能「让 token 并行」的形态，代价是接受率，代码场景 1.13–1.25×（**旧权重 + v2 链**；官方 `UD-Q4_K_M` 的链上端到端是 **1.01×**，见 [`SPECULATION-DATA.md`](../measure/SPECULATION-DATA.md) §5） |
| 段内 offload | 另一个维度 | `-ncmoe`、专家预取只改每段耗时，不改依赖图与往返次数 |
| 计算 ∥ 传输 | 不可行 | 8 KB 传输是亚毫秒级，每段真正付的是往返里**等待**的那一半（尾段 25–30 ms，[`THREE-SEGMENT.md`](../deploy/THREE-SEGMENT.md) §4）；等待期没有别的工作可派，下游要的正是这一跳的载荷 |
| 多请求 / 多序列 | ~~有条件可行，唯一的大块~~ **2026-11-12 决定不做** | 理论上不同序列互不依赖，三条同时在飞能把合计吞吐从 `1/Σ` 推到 `1/max`（按 73.4/64.6/55.7 ms，上限 **13.6 t/s（2.6×）**，但每 token 延迟不降）。**不做的理由**：llama.cpp 的 slot 调度与 KV 预分配本来就是为单序列延迟优化的，多序列要改的三件前置（`n_seq_max`、协议的 per-row `seq_id`、驱动侧并发）都碰它的核心假设；而 v2.1 的目标是「单用户、低延迟、长上下文」，不是吞吐。真要做吞吐应该换后端，不是改这条链。**实测佐证**（2026-11-12，2B 本地双段 `0:12+12:24`，944-token prompt ×2，`temp 0`）：`-np 2` 的端到端聚合是单序列的 **0.99× / 0.81× / 0.81×**（三次），纯 decode 重叠有 1.12×，但并发 prefill 只有 **0.70×**（CPU 14 线程已满，合批让每 chunk 更慢）；而真实多序列请求的 prefill 会同时到达，正好落在 0.70× 那段。探查结论保留在此表里，作为「为什么不做」的依据 |
| prefill ∥ decode | 多序列之后才谈得上 | prefill 现在是一次走完整链的大 batch，`server.py` 一把锁串行；要让第二个请求在别人 prefill 时 decode，得先有 seq 维度、再把 prefill 切块插入。买的是公平性，不是吞吐 |

单序列的延迟只有三个旋钮：投机接受率、段数（每段 33–35 ms 串行叠加，**v2 口径**）、往返（星形拓扑，hidden 每段都回本机再发下一段）。吞吐那个旋钮叫多序列。见 [`THREE-SEGMENT.md`](../deploy/THREE-SEGMENT.md) §4。
