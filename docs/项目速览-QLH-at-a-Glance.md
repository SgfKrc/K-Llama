# K-Llama 速览 / 新人 · 评审快速入口

> 状态：**现行**
>
> 更新日期：2026-10-19
>
> **Language**: [English](项目速览-QLH-at-a-Glance.en.md) · [简体中文](项目速览-QLH-at-a-Glance.md)
>
> 本页是 2 分钟导览。项目对外名称是 K-Llama，仓库内代号 QLH（代码符号用 `QLH_*`）；详细能力、边界与证据见 [README](../README.md) 全文与专项文档。

---

## 这是什么？

K-Llama 是面向异构边缘设备的**轻量化分布式大模型推理系统**（北京交通大学 2026 大创项目）。它把"谁持有哪些层、用什么引擎、在哪里、容量多大"统一成一条抽象 `(layer_range, engine, location)`，据此在**装不下整模**的机器群里把模型分层接力跑起来。

**两条引擎档位**，按设备画像选（不是"有 GPU 就用 torch"）：

| 档位 | 引擎 | 适用 |
|---|---|---|
| **L 档** | llama.cpp / GGUF（含裁层 GGUF + `embd` 注入） | 轻量、边缘、无 torch 的节点 |
| **D 档** | PyTorch（层拆分、层间流水线、多节点层段） | 有 CUDA 的 PC；也是层切点搜索与实验的平台 |

L档的显式本地整模路径由`src/inference_path_selector.py`的`select_inference_path`选择binding或llama-server；LL层段使用keep-head／native前向，DL由PyTorch首段接llama.cpp后续段。当前路径及能力边界见[整体架构](整体架构.md#2-当前执行路径)。

分布式模式由master协调并转发各连续层段，worker保持星状连接。用户显式退出计算参与并完成层段退休后，可加载本地完整模型；分布式容量不足返回具名等待或拒绝。版本化客户端和迁移合同见[R5客户端切换](重构客户端切换-R5-2026-10-19.md)。

核心设计原则：**数据不出集群、断网可自治、可复现验收**。

## 已验证的能力

| 能力 | 判据 / 日期 |
|---|---|
| 双机真机分层推理（QW1.8B 0-21 / 21-24，该模型已从内置候选列表退役） | 3 次 `distributed_required` 全过，RTT 6-12 ms（2026-08-20） |
| 历史任务链重启 / 杀进程恢复 + Tailnet IPv6（该机制已于票 14 R5 裁撤，仅存历史证据） | `wf_330d0aa1…`，重派 0（2026-08-21） |
| D→L 跨框架接力正确性 | 主仓双引擎矩阵 **27/27 逐 token 一致**（2026-09-21） |
| D→L 容量收益 | qwen2.5-0.5B **1.568×** / qwen3-2b **1.457×**（严格口径，`C` 只计不属于任何 `*.layers.<i>.*` 的张量；含视觉塔时为 1.613×）；同一受控 3.0 GB CUDA 预算下整模被拒而 12 层上游通过（2026-09-21） |
| L→L keep-head 通道 | `--path l2l_keep_head` / `d2l2l_keep_head` **32/32**，含「1 torch 上游 + 2 llama 下游」三段（2026-09-21） |
| 纯 L→L 多跳（三段全设备） | y700 head8 → y700 mid8-16 → y700 cut-k16 tail，本机只发命令 ⇒ **32/32**（2026-09-22） |
| 纯 L→L 跨设备（三段） | y700 head8 → **Surface** mid8-16 → y700 tail ⇒ **32/32**，两台真机协作、本机不参与计算（2026-09-22） |
| Android 层段数值验证 | 真 ARM64（Snapdragon 8 Gen 3 / Android 15）与 x86_64 逐 token 完全一致；dotprod/i8mm kernel 已证实（2026-09-21/22） |
| 切点求解闭环 | `scripts/relay_cut_plan.py` 从实测拟合段画像再求解；2 段闭环在 qwen2.5（r² 0.96/0.99）与 qwen3.5（0.79/0.96）通过 |
| Windows 原生编译路径 | `triton-windows==3.8.0.post28` 实测可用；`PYTHONUTF8=1` 是前置条件 |
| 判题口径修复 | 多模型 0/4 → loose+512 可区分；DS3-0324-7B 替代 R1（v2 口径 2/4×3、格式率 8/11×3） |
| 子项目：小模型 harness 工作台（S1-S8） | 上下文预算 / STATE 记忆 / RAG / MCP，本机门；相关实现已外置到 `harness_workbench`（Koakumix） |
| 联网搜索 / 轻量 Fetch（WEB-TOOL G1-G6） | 本机开发门；该能力与 `production_network_enabled` 开关已随迁外置到 `harness_workbench`，主仓 `src/` 不再包含 |

完整矩阵、口径与报告路径见 [跨框架接力-当前有效基线与后续优化计划](跨框架接力-当前有效基线与后续优化计划-2026-09-21.md)。

## 未完成的部分

- R6对新架构另采实际发行物首启、正式数据迁移和多设备9B LL／DL；历史设备数字按表中日期保留，完成门见[重构设计](单一事实源重构设计-2026-10-18.md#12-验收与退出标准)。
- 跨主多数复制、物理故障恢复和可信签名发行保持独立外部验收，当前状态以[排票计划](彻查后修复与重构排票计划-2026-10-09.md)为准。
- 分布式投机仍需局部draft、多位置输出与可验证状态回滚；独立PoC已裁撤，本地原生n-gram保留，详见[能力审计](当前功能与事实归属审计-2026-10-18.md#5-推理优化与图依赖边界)。
- 网络延迟直接影响逐步接力；部署应依据实测RTT与容量，比较使用相同模型、量化、构建和采样口径。

## 首次启动

依赖：

| 依赖 | 版本 | 用途 |
|---|---|---|
| Python | ≥ 3.10（推荐 3.12；Edge 档需要 3.12 及以上，因 `requirements-edge.txt` 锁定 `numpy==2.5.3`） | 主运行时与工具脚本 |
| Node.js + npm | Node ≥ 18 | 仅产品壳支线（`qlh-shell`） |
| JDK 17 + Android SDK（`compileSdk 36` / `minSdk 26`） | — | 仅构建 Android 时需要 |
| Tailscale | 最新 | 分布式模式；节点在同一局域网时最稳 |
| Git | — | clone（含 submodule） |
| NVIDIA 驱动 + CUDA（可选） | — | L 档 `llama_server` 路径（有 CUDA 且本机有 `llama-server` 时自动选）与 D 档 |

```bash
git clone --recurse-submodules https://github.com/SgfKrc/K-Llama
cd K-Llama
python scripts/setup_envs.py --all            # 主线 Python 环境（不含产品壳 Node，后者由 qlh-shell 自维护）
python scripts/setup_envs.py --only test    # 只配指定环境（可选名见下）
python scripts/setup_envs.py --check          # 只校验不安装（无副作用）
```

`setup_envs.py` 覆盖 `main` / `test`（`.venv-test`）/ `edge` / `gemma4-native` / `gemma4-pipeline` / `qwen3-sidecar` 六个环境；torch 等平台相关大件不自动安装，脚本会过滤并打印各环境的安装命令（如 `--torch-index-url https://download.pytorch.org/whl/cu126`），装完再跑一次 `--check`。其余环境与启动方式见 [README 快速开始](../README.md#快速开始)。

默认模型按画像选（`src/model_config.py` 的 `DEFAULT_MODEL_BY_TIER`，消费点 `src/config.py` 的 `get_active_model_paths()`）：

| 设备档 | 默认模型 |
|---|---|
| 移动端 | `<1B` 档：`qwen2.5-0.5b` |
| 超极本 / 边缘设备（共享显存 ≤ 2 GB） | `<1B` 档：`qwen3-0.6b` |
| PC（独显） | `~2B` 档：`qwen3-5-2b` |

2B 档实测 VRAM 峰值 2.24 GB，共享显存 ≤ 2 GB 的超极本放不下。模型文件不进 git（`models/` 已 gitignore），获取方式见 [README 模型与分布式](../README.md#模型与分布式)。

## 文档怎么读

- **总入口**：[README](../README.md)（中）· [README.en](README.en.md)（英）
- **接力（最活跃的方向）**：[跨框架接力-当前有效基线与后续优化计划](跨框架接力-当前有效基线与后续优化计划-2026-09-21.md) 为索引，配套 [项目报告](跨框架层接力-项目报告.md) 与 [容量收益实测](跨框架层接力-容量收益实测-2026-09-21.md)
- **架构与接口**：[整体架构](整体架构.md) · [模块接口说明](模块接口说明.md) · [核心技术原理](核心技术原理.md)
- **计划**：[主线开发计划](主线开发计划-分布式推理与边缘优化-2026-09-14.md) · [P4.5 立项](主节点动态选举与分布式管理-P4.5立项-2026-09-21.md)
- **TUI**：[使用指南](TUI使用指南.md) · [指令集](TUI指令集.md)
- **测试与判据**：[测试与评判标准](测试与评判标准.md) · [测试通道运行说明](测试通道运行说明.md)
- **Android**：[Android 验证替代路径](../android/Android验证替代路径-2026-09-18.md)
- **文档自身**：[文档状态与清理清单](文档状态与清理清单.md) · [归档索引](archive/README.md)

## 工程文化

- **证据链优先**：每个"已完成"附测试/日志/真机证据，判据与日期缺一不可。
- **判据纪律**：一致性只认 per-token argmax；分叉如实标 FAIL；区分 `mainrepo_end_to_end` / `raw_binding_probe` / `capacity_only` 三类记录，不混用。
- **机制化而非绿化**：主仓测试通道为 `unit`（并行）/ `external`（串行）/ `smoke`（真实权重，显式 opt-in）；票 14 重构另有 R0–R5 固定 stage 合同集（`scripts/run_refactor_contracts.py --stage`）与静态边界门（`scripts/check_refactor_boundaries.py`）。修复必配负例。
- **先测再解释**：性能结论先量出瓶颈在哪一环（分段耗时 / RTT / 离散度）。
- **用户主权**：模型工件、密钥、知识库归用户；离线可恢复。

## 许可证

MIT（Copyright (c) 2026 SgfKrc），详见仓库根 [LICENSE](../LICENSE)。
