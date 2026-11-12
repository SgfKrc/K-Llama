# 轻量化大模型分布式边缘推理优化系统

> 结题技术报告 · 主体 **v2.1**（llama.cpp 分段 fork + 约一千行控制面）
> 状态：**现行**
> 更新日期：2026-11-11
> 适用范围：本报告的总纲——摘要、六章导读、关键结论、对照立项书的完成情况、代码与验收摘要。
> 各章各自带状态块与口径；数字的出处写在每章表内，本文只做索引，不重复展开。

## 摘要

本报告是北京交通大学大学生创新创业训练计划项目「轻量化大模型分布式边缘推理优化系统」的结题技术报告。
报告主体是仓库当前的形态 **v2.1**：一个 llama.cpp 的魔改 fork（把分段推理下沉进 `llama_context`）加一个约
一千行的控制面；v2 的 Python 驱动链已退场，只在交代沿革时出现。

要解决的问题有两个互相独立的维度：**装不下**与**算得慢**。以目标模型 `Qwen3.6-35B-A3B-UD-Q4_K_M`
（21.11 GiB）为例，一台 16 GiB 的笔记本连装都装不下；而单序列 decode 的依赖图是一条链，拆开也不会自动
变快——拆开买到的是**容量**，代价是往返与段数。

做法是把模型按层拆到多台机器上，同时要求拆完之后**能力与整模对齐**：每台机器持同一份整模，只创建、只
读取自己那几层，没有裁层工件；段间只传一个 `n_embd` 宽的 float32 隐藏态（8 KB），不传 logits。判据是
**逐位 / 逐字节的 logits 与 token**，不是生成的文字。链下沉进 `llama_context`，`llama-server` 自己就是链
的驱动者，`/completion`、`/chat/completions`、slot、prompt cache 全部走原生实现，客户端不需要知道段在哪
台机器上。

改造落在两个补丁上：主补丁把分段链、远程段、协议 v9、投机接线与段级 offload 一起带进来，可单独编译、
单独验收；SLIM-ARC 预取是可选增量，默认关闭。控制面只做三件事——读拓扑、起停进程、探活。

结果：`verify/verify.sh` **53 项**与 `verify/verify_chain_death.py` **36 项**全过；DFlash / DSpark 在真跨机
上**零损失**（117/141 = 0.82979，与整模逐 prompt 相同）；MTP 的 catch-up mode 2 把链上接受率追平整模
**0.9098**（121/133）；3103 token 历史整段重算 19847 ms，第二轮只解 38 个新 token 用 **714 ms**。安卓平板
（Y700 / Termux / arm64）作为尾段入链：35B 的 8 层段驻留约 **4.90 GiB**，量到 **6.34–8.50 tok/s**。

局限一句：远程链的 MTP 接受率停在 **0.788**，且是结构性的（远程 head 没有 `ctx_dft`）；远程链单序列；
架构白名单只认 `qwen35` / `qwen35moe`。

## 六章

| 章 | 一句话导读 |
| --- | --- |
| [01-背景与目标](report/01-背景与目标.md) | 问题（装不下 / 算得慢）、既有做法的代价、判据立场（逐位而不是「看着对」）、v2 → v2.1 的沿革 |
| [02-总体设计](report/02-总体设计.md) | 七个决策，每个写「选了什么 / 为什么 / 代价是什么」，以及这套形状主动放弃了什么 |
| [03-实现](report/03-实现.md) | 具体做了什么：补丁构成、关键符号与调用路径、协议版本、控制面 |
| [04-验证](report/04-验证.md) | 53 项十二组逐组「测什么 / 判据 / 为什么存在」，七个专项脚本，纯分段版独立验收，跨机与安卓的五步破坏性验收 |
| [05-结果](report/05-结果.md) | 分段 vs 整模的吞吐、投机接受率、跨机三形态、内存、段级 offload 与 MoE 缓存、tap 代价、三种「这样比是错的」 |
| [06-局限与结论](report/06-局限与结论.md) | 十条边界（现象 / 根因 / 影响 / 代价）、主动不做的分页 KV、已知未做、结论与后续方向 |

## 关键结论

每行一句结论加判据类别，数字口径与实测明细在「详见」那一篇，本文不重复。

| 结论 | 判据 | 详见 |
| --- | --- | --- |
| 分段不改变数值：链 == 整模 | 逐位 / 逐字节的 logits 与 token | [04](report/04-验证.md) §2、§4 |
| DFlash / DSpark 跨机零损失 | 逐 prompt 与整模相同，输出逐字节相同 | [05](report/05-结果.md) §2.2 |
| MTP 的链上折扣是结构性的，能消但默认关 | 整模、链默认、catch-up mode 2 三档接受率 | [05](report/05-结果.md) §2.1、[06](report/06-局限与结论.md) §1.3 |
| 装不下变成装得下 | 35B 的 8 层尾段驻留进安卓平板；2B 链吞吐 | [05](report/05-结果.md) §1.2、§3 |
| 前缀复用把整段重算变成增量 | 同一历史第二轮只解新增 token | [05](report/05-结果.md) §4 |
| 内存按实际输出行数分配 | 同一 ubatch 的 logits 与 pp 图缓冲 | [05](report/05-结果.md) §4 |
| 段级 offload 与 MoE 缓存 | 段内 tg 与专家缓存命中率 | [05](report/05-结果.md) §5 |
| 边界：远程链单序列、架构白名单、MTP 折扣 | 十条边界的现象 / 根因 / 影响 / 代价 | [06](report/06-局限与结论.md) §1 |

## 对照立项书：完成情况

`local_docs/plans/立项.md`（2026 年 6 月申报）的技术承诺是六项，成果形式勾的是 **☑软件程序**。逐项对照如下。

### 表一：六项承诺

| 立项承诺 | v2.1 现状 | 判定 |
| --- | --- | --- |
| 4/8bit 模型量化 | **本项目零代码**，用的是上游能力：GGUF 的动态混合量化（`UD-Q4_K_M`，5.10 BPW；段内可见 IQ4_XS / Q5_K / Q6_K 等类型，`DEPLOYMENT.md` §6.8），不是 bitsandbytes 的 INT4 / INT8。`src/llama-quant.cpp` 不在 55 文件变更集里 | 已实现，但**借用上游**（本项目只做选型与验收） |
| 算子融合 | **本项目零代码**，用的是上游能力：ggml 计算图与算子融合（llama.cpp 自带），不是 `torch.compile`。`src/llama-graph.cpp` 在本仓只有 relay 段 `GET_ROWS` 一处改动 | 已实现，但**借用上游**（本项目只做选型与验收） |
| K 缓存优化 / 分页 KV | **没有 PagedAttention**。等价物是三件：prompt cache 前缀复用、checkpoint 序列状态导出 / 导入、`--moe-cache-mib` 专家 LRU（数字见 [05](report/05-结果.md) §4、§5） | **部分完成：换了机制**。差在「分页 / 页表 / copy-on-write」这一层不存在——`llama.cpp` 的 KV 是启动时按 `n_ctx` 预分配的定长 slot；量化论据见 [06](report/06-局限与结论.md) §2 |
| 多终端协同 | 兑现且更强：跨机 ROG（LAN）/ Surface（4C8T 弱机）+ 安卓 Y700（Termux / arm64）入链，2B 与 35B 都走完五步破坏性验收（[04](report/04-验证.md) §6） | 超额完成 |
| 模型分层拆分 | 核心就是它：层范围加载 + 段间只传 8 KB hidden；链 == 整模逐位（53 项） | 完全兑现 |
| 可演示可部署 | `control/qlhctl.py`（944 行）+ 拓扑（56 行）+ 一键 `up` / `status` / `down` + 53 项验收 | 完全兑现 |

### 表二：原方案的技术组件 → v2.1 的对应物或退场原因

行数为 `git show main:<path> | wc -l`（2026-11-11）；原方案是 `main` 分支立项期的 Python 原型。

| 原方案组件 | v2.1 的对应物 / 退场原因 |
| --- | --- |
| `src/paged_kv_cache.py`（574 行，自研分页 KV） | **退场**。换成 `llama.cpp` 定长 slot + prompt cache 前缀复用 + checkpoint + 专家 LRU；「为什么不做分页」的量化论据见 [06](report/06-局限与结论.md) §2，原实现随 Python 推理栈退场，功能域取舍见 `INVENTORY.md` |
| `src/scheduler.py`（257 行，主从调度） | **退场**。v2.1 没有中心调度：链由 `llama_context` 驱动，控制面只做读拓扑、起停、探活三件事（`control/qlhctl.py`） |
| `src/tcp_comm.py`（377 行，长度头 + 张量序列化） | **换实现**：定长协议 v9（`src/llama-split-proto.h`，响应头 32 B），载荷是 8 KB f32 hidden；没有 JSON 序列化、没有独立心跳帧（链死由 120 s 读超时与 `?deep=1` 判定，`SEGMENTED-LIMITS.md` §2.7） |
| `src/model_module.py`（373 行，PyTorch 加载 + `torch.compile` + 分层） | **换实现**：GGUF + 层范围加载（`--split-layer-range` / `LLAMA_LAYER_RANGE`），每台机器持同一份整模、只读自己那几层 |
| `src/api_server.py`（1073 行，FastAPI） | **换实现**：`llama-server` 原生 HTTP（`/completion`、`/chat/completions`），slot 与 prompt cache 走原生路径 |
| `src/web_ui.py`（253 行，Streamlit）+ `frontend/`（14 个文件，React + Vite） | **退场**。服务面就是 `llama-server` 的 HTTP；控制面是 CLI（`qlhctl.py up/status/down`），可视化平台对「逐位一致」这条判据没有贡献 |
| `src/device_profiler.py`（1073 行） | **换实现**：显存预算预检 / 后检（`src/qlh-vram-budget.h`），`VMEM: need ~X free Y` 把 WDDM 的静默溢出变成一行日志 |
| `scripts/quantize_model.py` 等 | **换实现**：llama.cpp 自带量化（IQ4_XS / Q5_K / Q6_K）+ `verify/` 的 53 项验收装置 |

**相对原方案，v2.1 在四个维度上抬高了**：

1. 模型从 1.8B 到 35B-A3B（21.11 GiB，单台 16 GiB 装不下——这正是立项要解决的「装不下」）；
2. 判据从「看输出」升级为逐位 / 逐字节（52 + 36 项）；
3. 跨机从「同一台机器两进程」升级为真跨机零损失；
4. 设备从三台校园笔记本扩展到安卓平板入链。

原方案是 Python + PyTorch + Qwen-1.8B + token 级 TCP；v2.1 是 llama.cpp 下沉 + 35B-A3B + 边缘弱设备 +
pre-norm 隐藏态接力。

## 代码与验收摘要

- **fork**：`H:\llamacpp-c`，分支 `rebase-merge`，单提交 `4acf03c8b`，父 `09338b85e`（上游最新，193 个提交
  之后），树 `548c3dfb0`；相对上游 **55 文件 / +20345 −245**（`README.md` 代码与验收）。
- **两个补丁**：
  - 主补丁 [`docs/llamacpp-split.patch`](llamacpp-split.patch) **45 文件 / +16680 −244 / 18749 行 /
    982.8 KiB**（纯分段链，可单独编译、单独验收）；
  - 可选增量 [`docs/llamacpp-slimarc.patch`](llamacpp-slimarc.patch) **13 文件 / +3665 −1 / 3790 行 /
    164.2 KiB**（SLIM-ARC prefetch，默认关闭）；
  - 叠加判据是 `git write-tree`：主补丁 → 纯分段树，再叠增量 → 与 fork HEAD 的树相同（`PATCHES.md`、
    `scripts/export_patches.sh`）。
- **验收**：`verify/verify.sh` **53 项**（十二组：1–8 组 22 + 组 9 的 2 + 组 10 的 20 + 组 11 的 8 + 组 12 的 1）；
  `verify/verify_chain_death.py` 36、`verify/verify_seq_id.py` 9、`verify/verify_mtp.sh` / `verify_mtp_e2e.sh`
  6 / 6、`verify/verify_grammar_spec.sh` / `verify_rbudget_spec.sh` 23 / 36、
  `verify/verify_seq_rm_exact.py --expect-known-fail` 1 pass / 5 known-fail（5 条是上游 hybrid 的已知基线）。
  判据全部是逐位 / 逐字节（`README.md` 代码与验收、[04](report/04-验证.md) §2–§3）。
- **控制面**：`control/qlhctl.py` 944 行 + `control/topology.example.toml` 56 行，合计 1000 行以内
  （[02](report/02-总体设计.md) §6）。
- **最近一次回归**：2026-11-11，改过 `src/llama-split-net.h`（补 `#include <netdb.h>`）之后
  `bash verify/verify.sh --skip-build` = **52/52 PASS、0 FAIL、0 SKIP**（[04](report/04-验证.md) §5；
  那次是组 12 加入前的十一组口径，现在完整跑是 53 项）。
