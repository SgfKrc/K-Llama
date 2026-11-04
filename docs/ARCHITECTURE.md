# 机制：非整模加载凭什么成立

> 状态：**现行**
> 更新日期：2026-11-03
>
> 适用范围：不切模型怎么分段加载、段之间传什么、哪些约束是硬的；读者为维护或移植该机制的开发者。
> 相关：补丁改动清单 [`PATCHES.md`](PATCHES.md)、实测数字 [`MEASUREMENTS.md`](MEASUREMENTS.md)、
> 投机判定 [`SPECULATION.md`](SPECULATION.md)。

---

## 0. 一句话

**每个节点都持有同一份整模文件，加载时只创建、只读取属于自己那几层的张量。**
没有"切模型"这个动作，也没有"层段工件"这类中间产物。

---

## 1. 为什么整模可以不整读

GGUF 的内部布局是：一段元数据（键值对）+ 若干个 tensor，每个 tensor 的数据在文件里是
**连续区间**，起始偏移写在元数据里。

llama.cpp 用 mmap 把整个文件映射进地址空间。**映射不等于读入**——操作系统只在页被访问时
才从磁盘取。所以"持有整模"和"把整模读进内存"本来就是两件事。

本项目把第二步也省掉：让 llama.cpp 在**加载阶段**就认为这个模型只有 `hi-lo` 层，
于是它只为这几层创建 tensor，其余层的字节从头到尾没有被 touch。

---

## 2. 三个机制

### 2.1 层数覆盖：`LLAMA_LAYER_RANGE=lo:hi`

`llama-model-loader.cpp` 在元数据解析之后、`load_hparams()` 之前改写 `block_count`
（以及 `nextn_predict_layers`）。架构读到的 `n_layer` 就是 `hi-lo`，它所有的循环都只在这几层里转。
元数据里同时保留原始层数，因为"这段是不是末段"要靠它判断（末段才保留 `output` 与 MTP 头）。

### 2.2 名字重映射：本地 `blk.i` → 源文件 `blk.lo+i`

`get_weight` 在校验名表**之前**先做重映射。这一步不能省：`lo > 0` 的段如果不重映射，
会读到源文件的 `blk.0`——输出仍然通顺，只是错的。

### 2.3 全局张量按需 lazy

`token_embd`（515 MiB）只有首段要，`output`（515 MiB）只有末段要，中段两个都不要。
不创建就不读：中段因此省 **1030.62 MiB**。

判断依据是 `tn.tensor` 而不是名字——tied 模型会把 `output` 建成 `token_embd` 的副本，
名字说是 embedding，实际承担投影。

三者合起来：任何一个节点都没有"整模加载"这一步，只有"我这几层"。

---

## 3. 接力向量：必须在 `output_norm` 之前

上游的 `t_embd` 是 `build_norm` **之后**的值（pooling 要归一化后的结果）。接力要的是原始输出：
多过一次 RMSNorm，第一个 token 就分叉。

`embeddings_pre_norm` 开关把 tap 点前移，顺带不建 lm_head——中段因此再省 515 MiB。

---

## 4. 段间传什么

一个 `n_embd` 宽的 float32 向量：2048 × 4 B = **8 KB**。

**为什么不用 fp16 / int16**：2B 两段实测（5 个贪心 prompt × 48 token），fp16 有 1 个在第 22 步换了
token，int16 定点同样 1/5（第 16 步）——量化步长细 20 倍也照样翻转，因为分叉点都落在输出接近
平局的位置。判据是**逐位一致**，不是"误差足够小"。条件见 [`MEASUREMENTS.md`](MEASUREMENTS.md) §8.1。

协议：4 字节大端长度 + 1 字节类型 + payload，hidden 走原始字节，不套 JSON。

---

## 5. 段内怎么 offload

两条互不干扰的路径：

| 参数 | 语义 | 什么时候用 |
| --- | --- | --- |
| `-ngl N` | 前 N 层的**全部**张量上卡 | 层少、显存够 |
| `-ncmoe N` | 前 N 层的**专家**留 CPU，其余专家上卡 | MoE |

MoE 的权重分布是专家占 88%，而每个 token 只用到 8/256 个专家——**体积最大、访问最稀疏的
部分进最快的设备**收益最大。实测同一台机器：整层 11 层 10.11 tok/s，专家 8 层 28.47 tok/s（口径见 [`MEASUREMENTS.md`](MEASUREMENTS.md) §3）。

段内怎么 offload 与段间怎么切是正交的，这是本设计唯一的抽象。

---

## 6. 切分的硬约束

- 所有切点必须是 `full_attention_interval`（qwen35moe 是 4）的倍数
- 末段的 `hi` 必须等于模型总层数
- `topology.py` 在解析阶段就拒绝，不自动纠正

原因：`src/models/qwen35moe.cpp:24` 用**本地层号**推算层类型，而本地层号被映射回源层号。
只有 `lo` 是 4 的倍数时，两者模 4 同余，层类型序列才对得上。
`32:38` 这种 6 层划分会在解析阶段报 `layers=6 is not a multiple of 4`。

严格说 `topology.py` 查的是**段宽**（`s.layers % 4`），不是切点偏移；但在「每段都声明
`layer_range`、宽度必须等于 `layers`、各段之和必须等于总层数」三条同时成立时，两者等价，
所以没有漏网。llama.cpp 自己不做这个检查——`LLAMA_LAYER_RANGE` 只校验 `lo >= 0 && hi > lo`。

**错位不是静默算错，是加载期直接失败**：本地层 `i` 被当成与源层 `lo+i` 同类型，于是它去要
`blk.<lo+i>.ssm_conv1d.weight`（或反过来要 `attn_q_norm`），而那个张量在源层不存在、且没有
`TENSOR_NOT_REQUIRED` 标记，`llama-model-loader.cpp` 走到 `throw "missing tensor"`。
2026-09-18 的裁层实验里 `N=2` 就是这样报的 `missing tensor 'blk.1.ssm_conv1d.weight'`。
这是这条约束唯一的优点：从不静默。

**要解除的话**（评估过，暂不做）：把 `is_recr_impl[i]` 的取模从本地号换成源层号——
`ml.layer_lo` 在 `load_hparams()` 时已可见（`llama_model_loader` 是 struct，字段全 public），
改 `qwen35moe.cpp` 与 `qwen35.cpp` 各两行即可；下游的 KV／recurrent 分配、graph 分派全走
`is_recr()` 这一真源，自动跟着对。不做的理由是收益太小：可选切点从 9 个变成 39 个、最小
台阶约 0.43 GiB，但「把 full attention 集中到有 GPU 的段」做不到——full 层在源模型里均匀
固定（`attn_q.weight` 只出现在 `3,7,…,39`），任何切分都改不了「每 4 层必有 1 个 full」。
那点内存收益会被多一个段边界的跨网往返吃掉。真遇到「某台设备必须 ≤3 层」时，先试 `-ncmoe`。

---

## 7. 投机怎么接进来

两个草稿源，都插在验证器前面：

| 源 | 是什么 | 深度 |
| --- | --- | --- |
| MTP | 模型自带的 NextN 头，递归喂自己的输出 | k=2 是上限（第 3 层掉到 36%） |
| DFlash | 独立的小模型，一次前向出 16 个位置 | k=5 是拐点（k=8 起接受率崩塌） |

共同要求：**每一步必须能回滚**。验证块写 k+1 行，只保留接受的前缀，剩下的要退回去。
hybrid 模型（30 层 linear + 10 层 full attention）的 recurrent state 没有位置索引，
退回需要按 ubatch 滚动的快照——这是本项目最贵的一个坑，见
[`ISSUE-recurrent-snapshot-rollback.md`](ISSUE-recurrent-snapshot-rollback.md)。

运行时只支持贪心：验证器给的是 argmax。采样要 rejection sampling，比的是两个分布的比值，
shim 已能按行回传完整 logits（`qlh_kh_logits`，`shim/qlh_keep_head.def`），缺的是一次判据实验——
同一位置上整模与链路的 top-k 候选是否重合，探针是 `checks/probe_logits_delta.py`，
缺口在哪见 `README.md` 的「未做」第 1 条。推理路径本身还没接采样。

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

一句话：RPC 是**把设备虚拟化**，模型仍然一份；本项目是**把模型分段**，每段各自持有。
代价是段内不能跨机器拆算子，每段必须能单独装下。

---

## 9. 什么算证据

| 级别 | 例子 |
| --- | --- |
| **硬判据** | 贪心 token 与整模逐位一致；段间 hidden `max\|diff\| = 0` |
| 参考 | 吞吐、接受率（要带口径：模型、量化、生成长度、设备） |
| 不算 | "输出看起来通顺"、单次运行、跨配置比较 |

跨机同样验过逐位一致。`checks/` 里 20 个脚本按这个标准写。

---

## 10. SLIM-ARC 在哪一层

它不改 §2–§6 的任何一条机制，只改"某段装不下、真的在读盘时"的**读盘顺序**：把随机缺页变成
按层边界的顺序预读。

装得下的时候它是空操作——整模在页缓存里，没有可补的页，实测 `issued = 0`。
接入形态与触发条件见 [`PATCHES.md`](PATCHES.md) 与 [`SLIM-ARC-接入计划.md`](SLIM-ARC-接入计划.md)。
