# 层段分布式推理

> 状态：四段跨机链路可用；TUI、OpenAI 兼容接口、采样、显存与溯源校验均已接入
> 更新日期：2026-10-30

把一个大模型按层切成段，每段是一个独立的 llama.cpp 实例，跨设备串行接力：
一个 token 依次经过各段，段间只传 hidden state（`n_embd` 个 float32，8 KB）。

Qwen3.6-35B-A3B（MoE，40 层，20.8 GiB）切四段跨两台机器跑到 3.4 t/s。

---

## 这套东西的价值在哪

先说清楚它**不**用来提速。MoE 的权重分布是专家占 88%（18.3 GiB），每 token 只用
8/256，所以单机上"专家上卡、其余留 CPU"比整层上卡快得多：

| 方案 | tg128 |
| --- | ---: |
| 纯 CPU | 8.23 |
| 单进程 `-ncmoe 32` | 28.02 |
| 两进程拆分 | 约 17 |

**两进程比单进程慢**——同一块 8 GiB 显存要分摊给两段。拆分的价值只在**单机装不下**时：
那时资源是相加而非共享。35B 就是这种情况（20.8 GiB 权重 vs 15.2 GiB 可用内存），
80B 以上更是如此。

记住这一点，否则会误以为拆分能提速。

---

## 快速开始

```powershell
koakuma                                    # 一条命令进 TUI
koakuma --topology topology-rog4.toml      # 指定拓扑
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

# 2. 切段（每段 11-16 秒，412 MB/s）
python cut.py --src <整模> --dst stage-0-8.gguf  --keep-head 8
python cut.py --src <整模> --dst stage-8-24.gguf --k 8 --end 24
python cut.py --src <整模> --dst stage-24-40.gguf --k 24

# 3. 跑
python chain_client.py --topology topology-rog4.toml --gen 32
```

远端机器先起 worker（`python stage_worker.py --role tail ...`），把拓扑里该段的
`host` 改成它的 IP。完整部署流程见 `docs/DEPLOYMENT.md`。

---

## 结构

```
protocol.py           6 种消息，长度前缀帧
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
cut.py                离线裁段（流式，fail-closed）
checks/               七份验收脚本
```

不写行数：这类数字每次重构都会过期，而职责不会。规模看 `git ls-files | xargs wc -l`。

---

## 四条必须知道的约束

**① 段必须精确拼满源模型。** 缺层不会报错，只会输出通顺但错的中文。`cut.py` 把
源模型名与覆盖区间写进工件（`qlh.*` KV + `.qlh.json`），`topology.load()` 每次校验
区间连续、无重叠、宽度与声明一致。见 `docs/DEPLOYMENT.md`。

**② 显存超额是静默的。** WDDM 允许分配超过专用显存，用系统内存兜住，llama.cpp 照旧
打印 `offloaded N/N`。实测同一工件：GPU 版 41.4 ms/token，比纯 CPU 的 30.4 还慢。
现在启动前预算、加载后实测核对，见 `chain_client.py` 的 `VRAM_PER_BYTE`。

**③ 不要绑核。** 每个 worker 有 5+ 活动线程，绑到相邻逻辑核（同一物理核的超线程对）
会互相抢占：实测把 step 从 50 ms 拖到秒级。所有拓扑现在都是 `cpus = ""`。

**④ 切点必须是 `full_attention_interval` 的整数倍。** qwen35moe 是 4，否则层类型
错位、加载报 `missing tensor`。`cut.py` 会 fail-closed 拒绝。

---

## 实测数字

| 配置 | p50 | 吞吐 |
| --- | ---: | ---: |
| 四段跨机（0-8 卡 / 8-24 / 24-32 / ROG 32-40） | 297.5 ms | **3.4 t/s** |
| 三段全本机（0-8 / 8-24 / 24-32，缺 8 层） | 98.7 ms | 10.1 t/s |
| 单段 0-8 全上卡 | 3.2 ms | — |
| 单段 8-24 纯 CPU | 32.7 ms | — |

四段比三段（ROG 从 16 层降到 8 层）快 36%：2.50 → 3.40 t/s。收益来自 ROG 的内存
驻留改善，不是算术——减半层数只省 16% 的每层成本。

链路比各段独立测之和慢 2.4 倍，但协议开销是 -7 ms：差在每段启动时 CPU cache 都是冷
的。17.33 GiB 权重轮换访问，24 MB 的 L3 留不住。

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
- 裁段工具的退场。整模 + `layer_range` 已经能替代它（见 `docs/DEPLOYMENT.md` §6），
  与裁段工件逐位一致、链路性能相当；还差 ROG 段的跨机验证才切换默认流程。
- 多序列 token 共享（需要 `kv_unified`，当前明确拒绝而不是崩）。
