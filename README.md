# K-Llama v2

> 状态：两段跨机链路可用；TUI、OpenAI 兼容接口、采样、显存校验均已接入
> 更新日期：2026-10-30

面向**非整模加载**的定制 llama.cpp 引擎。整模文件留在原地，每个节点只把属于自己那几层
读进内存，段间只传 hidden state（`n_embd` 个 float32，8 KB）。

Qwen3.6-35B-A3B（MoE，40 层，20.8 GiB）在两台机器上分成 32 + 8 层，跑到 5.3 t/s。
一台 8 GiB 显存的笔记本加一台 15 GiB 内存的机器就装下了它。

与第一代 K-Llama 的差别见 `docs/PATCHES.md`：那代用"裁好的层段工件"，这代用"整模 + 层范围"。

---

## 这套东西的价值在哪

拆分的用处在单机装不下的时候。MoE 的权重分布是专家占 88%（18.3 GiB），每 token 只用
8/256，所以单机上"专家上卡、其余留 CPU"比整层上卡快得多：

| 方案 | tg128 |
| --- | ---: |
| 纯 CPU | 8.23 |
| 单进程 `-ncmoe 32` | 28.02 |
| 两进程拆分 | 约 17 |

**两进程比单进程慢**——同一块 8 GiB 显存要分摊给两段。拆分的价值出现在**单机装不下**时：
那时资源是相加而非共享。35B 就是这种情况（20.8 GiB 权重 vs 15.2 GiB 可用内存），
80B 以上更是如此。单机上拆分的吞吐低于整模。

---

## 适用范围

范围刻意收窄到一组特定组合，理由如下：

- **架构**：Qwen 3.5 及以上（`qwen35` / `qwen35moe`）。3.5 到 4 之间没有重大结构变化，
  所以这套按层段拆分的做法不用跟着改。别的架构要支持，得先补 tensor 命名、RoPE 变体、
  MTP 位置这几处差异——那是另一个项目的工作量。
- **量化**：只针对 8 bit 以下的量化（Q4_K 这类）。这不只是限制，还是**段间传输能用 FP16
  的前提**：Q4_K 权重的相对误差约 1e-2，比 FP16 传输的 5e-4 大 20 倍，传输误差落在
  量化噪声底下。换成 Q8_0 或 F16 权重，这个余量就没了，传输精度要重新论证。
- **为什么不做通用**：通用引擎得为每个架构准备分支，而这个项目的全部价值在于把一条
  具体路径做到能用。硬编码 ChatML、只认两个 arch，换来的是改动一屏能看完。

### 传输 dtype 保持 float32

试过压到 4 KB，两个方案都不行：

| 传输 dtype | 单跳 max\|diff\| | 贪心 token 一致 |
| --- | ---: | ---: |
| float32 | 0 | 5/5（基线） |
| float16 | 3.9e-3 | **4/5**，第 22 步分叉 |
| int16 定点 | 2.2e-4 | **4/5**，第 16 步分叉 |

int16 的量化步长比 float16 细 20 倍，照样翻转——因为分叉点都在输出接近平局的位置，
1e-4 级的扰动就够。两次独立运行逐位一致，这是确定性的量化影响。

顺带记一个我判断错的地方：曾以为"传输误差比新旧 llama.cpp 的 hidden 差异（0.042）小
180 倍，应该安全"。实测推翻了这个推理：0.042 那次输出没变属于运气。判据取**全部
prompt 的贪心 token 逐位一致**，"大多数一致"不够。

BF16 更不在考虑范围：它的尾数只有 7 位，精度比 float16 还差 8 倍，而动态范围优势
（1e-38~3e38）用不上——实测 hidden 的绝对值最大只有 6.2。

---

## 快速开始

```powershell
koakuma                                    # 一条命令进 TUI
koakuma --topology topology-cross.toml     # 跨机两段（本机 + ROG）
```

启动画面逐项显示真实进度（拓扑、shim、内存、显存、各段张量加载百分比），
每行对应一件真事，卡在哪看得出来。

### 依赖

```bash
pip install numpy psutil gguf textual
```

`gguf` 只有裁切和溯源校验要用，`textual` 只有 TUI 要用。

### 手动三步

```powershell
# 1. shim：本机有 CUDA 就编译，纯 CPU 机器直接拷已编好的（约 8 MB）
shim\build_shim_msvc.ps1

# 2. 准备一份整模，每台机器都放同一份（不切段）

# 3. 跑：拓扑里的 layer_range 决定各段拿哪几层
python chain_client.py --topology topology-cross.toml --prompt-text "中国的首都是" --chat --gen 32
```

远端机器先起 worker（`python stage_worker.py --role tail ...`），把拓扑里该段的
`host` 改成它的 IP。完整部署流程见 `docs/DEPLOYMENT.md`。

---

## 结构

```
protocol.py           6 种消息，长度前缀帧
wire.py               段间 hidden 的传输 dtype（默认 float32）
topology.py           拓扑解析、工件溯源与覆盖校验
keep_head.py          同进程的 shim 封装
keep_head_worker.py   子进程封装（n_gpu_layers=0 必须新进程）
engine.py             段的加载与前向
stage_worker.py       单段进程（TCP 服务端）
chain_client.py       拉起 worker、驱动解码、内存与显存预算
stream.py             流式生成与 UTF-8 重组
server.py             /v1/chat/completions、/v1/models
proto_responses.py    /v1/responses 的纯翻译层
koakuma.py            TUI 与启动画面
config.py             4 个环境变量
cut.py                离线裁段（现在只用于比对，主流程用整模 + layer_range）
chat.py               ChatML 渲染与 parse_special 开关
checks/               八份验收脚本（含 wire 的数值判据与链路基准）
```

不写行数：这类数字每次重构都会过期，而职责不会。规模看 `git ls-files | xargs wc -l`。

---

## 五条必须知道的约束

**① 段必须精确拼满源模型。** 缺层不会报错，只会输出通顺但错的中文。`cut.py` 把
源模型名与覆盖区间写进工件（`qlh.*` KV + `.qlh.json`），`topology.load()` 每次校验
区间连续、无重叠、宽度与声明一致。见 `docs/DEPLOYMENT.md`。

**② 显存超额是静默的。** WDDM 允许分配超过专用显存，用系统内存兜住，llama.cpp 照旧
打印 `offloaded N/N`。实测同一工件：GPU 版 41.4 ms/token，比纯 CPU 的 30.4 还慢。
现在启动前预算、加载后实测核对，见 `chain_client.py` 的 `VRAM_PER_BYTE`。

**③ 不要绑核。** 每个 worker 有 5+ 活动线程，绑到相邻逻辑核（同一物理核的超线程对）
会互相抢占：实测把 step 从 50 ms 拖到秒级。所有拓扑现在都是 `cpus = ""`。

**④ 边界必须是 `full_attention_interval` 的整数倍。** qwen35moe 是 4，类型序列是 `LLL F`
重复，full attention 落在 3 / 7 / 11 / … / 39。`src/models/qwen35moe.cpp:24` 用**本地层号**
推算类型，而本地层号被映射回源层号 `lo + i`，所以 `lo` 是 4 的倍数时两者模 4 同余、类型对齐。

配合「tail 段的 `hi` 必须等于模型总层数 40」，末段层数只能取 4 的倍数。`32:38` 这种 6 层的
划分会在解析阶段被 `topology.py` 拒绝（`layers=6 is not a multiple of 4`）。`cut.py` 切工件时
判据更严：`lo`、`hi`、`hi-lo` 三者都要合格。

**⑤ 不要用 float16 传 hidden。** 4 KB 的负载能把跨网每步少 3 个包（约 15 ms），
但 2B 上 5 个贪心 prompt 有 1 个在第 22 步换了 token；更精细的 int16 定点同样 1/5
（第 16 步）。默认 float32，`SA_WIRE_DTYPE` 只是实验开关。判据见
`docs/MEASUREMENTS.md` §9 与 `checks/verify_wire.py`。

---

## 实测数字

测量条件：Qwen3.6-35B-A3B（Q4_K_XL，40 层），单序列，prompt 22–27 token，gen 16–32，
2026-10-30，两根桩之间走 Tailscale 直连。区间来自背景负载差异，不是多次运行的置信区间。

| 配置 | median | 吞吐 |
| --- | ---: | ---: |
| 两段跨机（本机 0-32 段 12 层上卡 + ROG 32-40 段） | 140-188 ms | **5.3-7.1 t/s** |
| 两段全本机（同样划分） | 100-135 ms | 7.4-12.2 t/s |
| 单段 0-32（12 层上卡 + 20 层 CPU） | 谷底 52-58 ms | — |
| 单段 32-40（纯 CPU） | 32.1 ms | — |

**段数应等于节点数。** 早先把本机拆成三段是设计错误：每多一个进程就多一份全局张量读取
（`token_embd` / `output` 各 515 MB）、一块 compute buffer、一轮页竞争。两台机器就是两段。

跨机比全本机慢的那部分，拆开看是三笔：

- **ROG 段的内存压力 22.7 ms**。同一段在独占内存时 30.85 ms，worker 常驻后 53.55 ms；
  后者把 ROG 的可用内存压到 0.82 GiB（8 层权重 4.16 GiB，总内存 15.2 GiB）。
- **跨网 8 KB 传输 24 ms**。Tailscale 的 MTU 是 1280，一个 hidden 要分 6 个包，每包一个
  往返（实测上行字节数与往返时间线性：5600 B 是 9.41 ms）。
- **worker 自身的协议开销在 1 ms 以内**。2B 上进程内直调 17.45 ms、worker 往返 17.76 ms。

早先记在这里的"worker 协议开销 11 ms"来自并行测两个进程的伪影——两者各需 4.16 GiB，
而 ROG 只有约 4.5 GiB 可用，互相踩内存。串行独占重测后该结论作废。

三段时期的 3.4 t/s 与四段对比同样作废：那时段数和节点数不对应，前提就错了。

段内耗时随 `n_past` 呈 U 型，两端高中间低，量法见 `docs/MEASUREMENT-PITFALLS.md`。

---

## 文档

| 文件 | 讲什么 |
| --- | --- |
| `docs/DEPLOYMENT.md` | 每台放什么、换分段怎么做、控制面**不做**什么 |
| `docs/ARCHITECTURE.md` | 链路与协议 |
| `docs/MEASUREMENTS.md` | 全部实测数据 |
| `docs/MEASUREMENT-PITFALLS.md` | 段内计时为什么骗人 |
| `docs/MEMORY-LIMITS.md` | 内存与显存的账怎么算 |
| `docs/CUT-TOOL.md` | 裁切规则与 fail-closed 条件 |
| `docs/RESPONSES-API.md` | Responses 协议，含与计划不符的实测 |
| `docs/SAMPLING.md` | 采样参数如何映射到 llama.cpp |
| `docs/TUI.md` | TUI 布局与启动画面 |
| `docs/DECISIONS.md` | 为什么是这样 |

验收：`python checks/verify_*.py`（区间、显存、多序列、TUI、协议），
`python -m unittest test_topology`。

---

## 未做

- 跨框架接力（llama.cpp ↔ torch）。SLIM-ARC 已评估：它的 `on-demand` 加载器是唯一
  能救"装不下"的机制，但没接进任何补丁；`prefetch` 只是预热已有映射。
- 投机解码。词表已验证一致，但词表匹配的最小草稿模型就是 2B（33 ms/token），
  相对本机三段只快 3 倍，加验证成本后是亏的。
- 裁段工具的退场**已完成**。整模 + `layer_range` 与裁段工件逐位一致（`max|diff|=0`），
  跨机也验证过了。`cut.py` 留着只用于离线比对。
- 多序列 token 共享（需要 `kv_unified`；当前对这类请求明确报错）。
