# Patchouli 知识库/文档库管理 TUI 专项计划（2026-09-14）

> 状态：**已实施（PATCH-01～06 全部完成，2026-09-14；PATCH-03 的 RAG 对接部分待知识库服务化后接）**；PATCH-07 **已实施**（父文档映射 + rewrite/rerank 参数对照，真实对照 MRR 0.33→0.44）；PATCH-08 已完成（启动动画〔并行加载不延迟〕+ `.env.docagent` 检测引导与 TUI 内编辑）；本计划把 docagent（只读扫描引擎）与主项目 `docs/` 文档库接成可交互的"图书馆管理员"工作台。
>
> **支线收尾（2026-09-14，用户裁决）**：本支线**到此为止**。PATCH-01～09 全部落地（引擎/书架/检索/编目/统计/流转/审计路由/启动动画），**MCP server（8 个只读工具，stdio，零第三方）已交付并注册到 Reasonix 与 Codex CLI**；`patchouli` 全局命令与 docagent skill 常态可用。RAG/知识库服务化（`SIDE-PATCHOULI-01`）**不随本支线推进**，待知识库服务独立立项时再评估。
>
> 创建日期：2026-09-14
> 适用范围：主项目 `docs/`（当前 **92 份** Markdown）与其元数据（状态行 / 票号 / 日期 / 互链 / 变更记录）+ docagent 审计能力的交互前端。**不覆盖**：文档内容创作、知识库训练数据、运行时产品功能。
> 关联：[文档维护Agent工具设计](../tools/docagent/docs/文档维护Agent工具设计.md)（扫描引擎设计）、[TUI适配实施计划](TUI适配实施计划.md)（Textual 栈先例）、`tools/docagent`（引擎实现，v0.2.0）、[开发票计划-审计收口与答辩演示](../misc/开发票计划-审计收口与答辩演示-2026-09-09.md)（票体系）。

## 1. 背景与动机

### 1.1 docagent 现状（2026-09-14 查证）

| 项 | 现状 |
| --- | --- |
| 定位 | 独立、标准库优先的**只读**文档维护扫描器（`qlh-docagent` v0.2.0） |
| 已具备 | M1 机械化扫描（`rules` / `scan` / `audit` / `init` / `config`）；定点条目审查 `audit-entry`（表述 vs 实际差异、`TEST_RESULT_UNBOUND`）；CI 门产物 `gate verify` / `gate rescan`；规则演进状态机 `rules evolve`（`proposed → preflight → approved → released`）；基线增量 `--baseline --dry-run`；profile 体系（`qlh` / `minimal`）；主项目兼容入口 `scripts/doc_maintenance_audit.py` |
| 边界（设计纪律） | 不自动改写文档、不替代人工审核、不做运行时接入 |

### 1.2 docagent 待完善项（本次查证结论）

1. **M2 LLM 适配器未迁入新包**：`DOCAGENT_*` provider 配置已校验，但"独立扫描器不会发起 LLM 请求"——语义判定（状态行措辞 vs 现状）仍走旧实现分支；迁移后 docagent 才能独立给出分级判定。
2. **M3 扩展参数未迁**：`events.py` 事件骨架已有（`audit --events *.sqlite` 可用），但"哪些文档受某次改动影响"的检索与旧实现迁移未完成。
3. **包内测试覆盖薄**：`tools/docagent/tests/` 仅 `test_entry_audit.py`（288 行）+ `test_environment.py`（290 行）；`scan` / `audit` / `gate` / `evolution` 的**包内**回归待补强。
4. **无交互界面**：全部通过 CLI + JSON/Markdown 报告——"发现遗漏"对人是批处理式的，缺少**浏览、检索、抽查**的连续工作流。

### 1.3 动机

- 文档库已 92 份且互链/票号引用密集（专项计划、票表、报告、设计说明混排）；当前交互方式 = 人工翻找 + `grep` + 逐个打开。
- 需要一个"**图书馆管理员**"式工作台：按主题/状态/票号快速定位、看元数据卡、跑审计抽查、追踪单文档流通历史——把 docagent 的批处理输出变成可连续操作的对象。
- 命名：**Patchouli**（不动的大图书馆）——交互式知识库/文档库管理 TUI。

## 2. 定位与边界

- **定位**：交互式**只读**工作台（TUI）。前端组织 docagent 的扫描/审计结果与文档库目录结构；**不改动 docagent 引擎**（引擎事实源与 CI 门不变）。
- **技术栈**：**Textual**（主项目已有 `textual==8.2.8`，`harness_workbench/tui.py` 为先例）——复用测试依赖栈，作为开发期工具与 docagent 同级，不进入安装包。
- **边界**：
  1. **不自动改写文档**（延续 docagent 纪律；修复建议以 diff 呈现，落笔走人工/主 agent + git 评审）；
  2. **不替代 docagent/CI**：TUI 只读其输出，不重造扫描逻辑；
  3. **核心 TUI 不依赖 GPU/模型**（开发期工具）；知识库检索的**模型侧增强为可选接线**（模型就绪时启用、缺失时确定性降级，见 §3.6）；
  4. 不做内容创作与知识库训练；**embedding 模型微调明确不做**（沿用项目既有决策）。

## 3. 功能设计（图书馆视图）

### 3.1 书架（浏览）
- 布局：**书架列表 → 文档卡 → 预览** 三栏；列表按分类（专项计划 / 票计划 / 报告 / 设计 / 其他）与日期/状态排序。
- 元数据解析（只读）：状态行（`> 状态：` / `> 更新日期：`）、票号（`[A-Z]+-[A-Z0-9-]+` 形态）、日期、互链、变更记录段。
- 文档卡：状态、更新日期、关联票、入链/出链数、最近改动提交。

### 3.2 检索（图书检索台）
- **对接既有 RAG 检索全链路**（不重新实现检索）：主项目 `src/rag_store.py` 的 `hybrid_search` 与 harness `/v1/rag/search`；请求级参数可视化（`metadata_filters`、`rewrite_limit`、每路候选数、FTS/vector 权重、`rerank_weight`/`rerank_mode`）。
- 全文/标题/票号/日期检索（文档库自身元数据层，纯文本层）；「哪些文档提及票 X」一键视图。
- 结果直接跳转文档卡与定位行；展示检索路由信息（改写计划、RRF/加权融合、rerank 分数链路）。

### 3.3 编目（审计与核对）
- 触发 `docagent scan` / `audit`（子进程，只读）并展示矛盾清单（状态行 vs git、未提交登记、链接断裂、`TEST_RESULT_UNBOUND`）。
- `audit-entry` 定点审查：选中文档段落/条目行 → 结构化"表述 vs 实际"差异。
- 历史报告浏览：读 `build/doc-audit/*.json`（不重跑）。

### 3.4 流通记录（历史）
- 单文档 `git log` 视图；文档头部"变更记录"段解析；M3 `events.sqlite` 查询（表可用时）。

### 3.5 馆藏统计
- 文档总数与状态分布（已完成/规划/等待）、最近更新榜、孤儿文档（无入链）、票号覆盖热度。

### 3.6 知识库检索（RAG）能力：现状对照与联动（2026-09-14 查证）

> 依据：`docs/小模型轻量推理harness工作台调研与方案.md` §18~23 实施记录 + 双端代码查证。RAG 优化方向已由票系列大部分落地，Patchouli 检索台**直接复用**，不重复实现。

| 方向 | 实现状态 | 证据 |
| --- | --- | --- |
| **结构化元数据过滤** | ✅ **已完成** | `RAG-META-01`：双端 `metadata_filters` 透传、`metadata_index_version` 回填、revision 冲突显式；`/api/rag/search` 与 harness 同口径 |
| **Embedding 模型微调** | ⛔ **明确不做**（项目决策） | 方案 §"不做训练和微调框架"；各 RAG 票均标"不训练或微调 embedding"；未来若有收益须独立实验适配器 + 登记 |
| **重排序（Rerank）** | ✅ 规则层已完成；模型化 rerank **待小模型** | `RAG-RERANK-01`：lexical 重排、`rerank_candidate_k`/`rerank_weight`/`rerank_mode`、fusion/rerank 分数链路保留；"规则层是后续模型重排的稳定候选接口" |
| **分块策略（动态/父文档/重叠）** | ✅ 策略集 + 重叠已落地；**父文档上下文未实现** | `RAG-CHUNK-01`：`CHUNK_STRATEGIES = {fixed, paragraph, sentence, section, adaptive, semantic}` + 重叠窗口上限 + `granularity`/`start_offset/end_offset`；无 parent-document 机制 |
| **索引策略（多粒度/关键词/混合/图谱）** | ✅ **全部落地** | `RAG-IDX-01`：多粒度 + 倒排关键词 + 轻量图谱（`_index_entities`/`_index_relations`，含 cooccurs 兜底）+ FTS/向量 RRF 混合检索 |
| **查询改写与多路召回（HyDE/多查询/Step-back/拆解）** | ✅ 确定性部分完整；**HyDE / Step-back / LLM 改写未做** | `RAG-QRW-01`：归一 + 同义词/术语词典 + and/or 拆解 + 1~8 变体 + RRF（hit@5=1.000000 / MRR=1.000000）；明确"不执行外部 LLM 改写"（确定性边界） |

**缺口与联动**：
- **纯软件可做**：父文档上下文检索（chunk → parent 映射）；Patchouli 价值场景（"这份文档所属的计划全貌"）直接受益。
- **等小模型就绪**（`HW-DSV4-9B-EVAL-01` 正在跑、`HW-DSV4-2B-01` 已下载）：模型化 rerank、HyDE、Step-back、LLM 查询拆解——`RAG-RERANK-01` 收口明确"下一张可执行票为 9B 评估"，即模型侧 RAG 增强以此为前置。
- **Patchouli 的定位**：把这些能力的**参数与结果链路可视化/可操作**（检索台 = RAG 能力的"管理员视图"）；"父文档上下文"等增强作为 PATCH 票落地；模型侧增强仍归 RAG 票系（不混入 PATCH 引擎边界）。

## 4. 票分解（PATCH-01～07 草案）

| 票号 | 目标 | 验收门 |
| --- | --- | --- |
| `PATCH-01` | 只读数据层：文档树扫描 + 状态行/票/链接/日期解析（复用 docagent profile 布局约定） | 对 `docs/` 92 份全量产出稳定 JSON；未知格式 fail-soft 归"其他"不崩溃；快照测试 |
| `PATCH-02` | 书架 TUI：三栏浏览 + 分类/排序/过滤 | Textual 冒烟测试；真实 92 份文档流畅滚动 |
| `PATCH-03` | 检索台：文档库元数据检索（全文/票号/日期）+ **对接 RAG 检索全链路**（复用 `hybrid_search`/harness `/v1/rag/search`，参数与路由链可视化） | 查询正确性单测；空结果态；RAG 参数透传（`metadata_filters`/`rewrite_limit`/`rerank_*`）与 API 行为一致 |
| `PATCH-04` | 编目视图：委托 `docagent scan/audit` 子进程 + 矛盾清单浏览 + 报告回看 | 子进程只读（无写路径断言）；展示数字与 CLI 输出逐字段一致；失败态可见 |
| `PATCH-05` | 流通记录：git log / 变更记录 / 事件库视图 | 只读；大历史分页；无 git 环境降级 |
| `PATCH-06` | 馆藏统计 + 收口（README/接线说明/演示流程） | 全量回归 + 演示脚本（录屏级流程走查） |
| `PATCH-07` | 知识库检索增强落地：**父文档上下文检索**（chunk→parent 映射，"看一份 chunk 知全貌"）；RAG 参数影响的可视化对照（改写/rerank 前后） | 父文档映射回归；检索对照报告与既有 `hit@5/MRR` 口径一致；纯软件、不加载模型 |

> 执行建议序：`PATCH-01 → 02 → 03 → 04 → 05 → 06`（07 可与 03 并行，属检索增强；08 启动动画可随时插入）；01/02 完成即可日常试用。
>
> **执行状态（2026-09-14）**：`PATCH-01`～`06` **全部完成**——实现在 `tools/docagent/patchouli/`（docagent 子项目内）（34 测试全绿；editable 安装 + `run.py` fallback）；键位：`/` 检索台 · `c` 编目诊断 · `h` 流转记录 · `s` 馆藏统计 · `1-7` 分类 · `a` 归档。
>
> | `PATCH-08` | 启动动画（Claude Code/opencode 风格首页）——**硬约束：动画不延迟启动**（动画与数据加载并行，启动时长 = max(动画, 加载)；任意键跳过；非 TTY/管道自动跳过；低帧率开销、零额外依赖） | 启动耗时对比基准（有/无动画增量 ≤ ~50ms）；跳过路径测试；非 TTY 自动跳过测试 |
> 与 RAG 票系联动：模型化 rerank / HyDE / Step-back / LLM 拆解属 RAG 票系列（前置为 9B 评估），Patchouli 只做可视化与父文档等纯软件增强（见 §3.6）。

## 5. 验收门与度量

- **只读纪律**：任何操作不写 `docs/`（测试断言无写路径）；不修改 git 状态。
- **性能**：92 份冷启动 < 2s、缓存后 < 500ms（增量扫描）。
- **可测性**：数据层纯函数化（单测）；TUI 层冒烟（Textual pilot）。
- **一致性**：审计展示与 docagent CLI 输出逐字段一致（同一次运行）。
- **可演示**：一条命令启动，5 分钟内完成"找一份计划 → 看状态 → 跑定点审查 → 看历史"的完整走查。

## 6. 风险与对策

| 风险 | 对策 |
| --- | --- |
| 范围蔓延成"文档编辑器/知识库创作" | 边界第 1/4 条；写操作一律出界 |
| 文档格式漂移导致解析器脆弱 | fail-soft 归"其他"；解析器快照测试；格式有变化时先补测试 |
| Textual 进入产品依赖面 | 维持开发期工具定位（与 docagent 同级，test 依赖） |
| 与 docagent 能力重复实现 | 扫描/审计逻辑一律子进程复用；TUI 只做展示与编排 |
| 大库交互卡顿 | 增量解析 + 缓存 + 虚拟列表 |

## 7. 变更记录

| 日期 | 变更 |
| --- | --- |
| 2026-09-14 | 首版：登记 docagent 现状查证（v0.2.0 / M1 已迁 / M2-M3 未迁 / 测试薄 / 无界面）与 `PATCH-01`～`PATCH-06` 票草案；定位只读 TUI + Textual 栈复用 |
| 2026-09-14 | 完善：新增 §3.6 RAG 能力现状对照（6 类方向：元数据过滤 ✅ / embedding 微调 ⛔ 不做 / rerank ✅ 规则层 / 分块 ✅ 缺父文档 / 索引 ✅ 全落地 / 改写多路 ✅ 缺 HyDE·Step-back）；检索台（PATCH-03）改为对接既有 RAG 全链路；新增 `PATCH-07`（父文档上下文 + RAG 参数对照）；边界修订（模型侧增强为可选接线） |
| 2026-09-14 | **实施完成**：PATCH-01～06 全部落地（`tools/docagent/patchouli/`（docagent 子项目内），34 测试全绿；editable 安装 + `run.py` fallback）；登记 `PATCH-08`（启动动画，不延迟启动硬约束） |
| 2026-09-14 | **归属合订**（用户指示）：Patchouli 与文档维护工具**合二为一**——全部代码迁入子仓 `tools/docagent/patchouli/`（上游 `SgfKrc/qlh-docagent` `befb507`，v0.3.0，console scripts `patchouli`/`patchouli-shelf`，47 测试随迁全绿）；主仓 `tools/patchouli/` 移除（仅保留 gitlink 指针）；editable 重装为 `pip install -e tools/docagent` |
| 2026-09-14 | **通用化 v2**（用户指示）：文档目录自动探测（`docs/`>`documentation/`>`doc/`>裸 md≥3）、多库注册（`patchouli lib add|list|remove` + `--lib`）、`--docs` 显式目录；docagent 定位改"已安装优先"（`python -m docagent` → `run.py` 回退）；端到端验证（非标准布局库注册后任意 cwd 按名调用）；67 测试全绿 |
| 2026-09-14 | **观感与易用性修订**：启动屏配色改半透黑底 + 淡紫填充（40K 色孽调）；`install.ps1` 一键安装（npm 式：editable/全局命令/PATH/默认根配置/自检，幂等）；`roots.resolve_root` 修复任意目录调用（`--root` > env > cwd 上溯 > `~/.patchouli/config.json`）；`patchouli setup` 子命令；59 测试全绿 |
| 2026-09-14 | **入口与观感修订**：统一 CLI 分派（`patchouli` 无参=TUI〔非 TTY 退回摘要〕/`summary`/`json`/直通 catalog）；启动动画新增**最小展示时长**（默认 1.0s 可感知，`--splash-time` 可调；加载更慢不额外等待）；注册全局命令（`%USERPROFILE%in\patchouli.cmd` + User PATH）；53 测试全绿 |
| 2026-09-14 | **PATCH-08 实施完成**：`splash.py`（与数据加载并行、加载完即关、任意键跳过、非 TTY 自动跳过、`--no-splash`/env 关闭）；`config_check.py` + `config_editor.py`（`.env.docagent` 检测/脱敏键名/模板引导/TextArea 编辑屏 + Ctrl+S 保存 + `.bak` 备份 + 路径逃逸防护 + 保存后自动 `docagent config` 校验）；写权限边界限定 `.env.docagent`；47 测试全绿 |
| 2026-09-14 | **PATCH-07 实施完成**：`chunks.py`（标题分节 chunk→parent 映射 + 章节定位）+ `compare.py`（复用 `src/rag_store.rewrite_query` 规则层 + lexical rerank；MRR 口径对齐 `rag_quality`）；`v` 键对照视图；真实对照「缓存」MRR@5 0.3333→0.4444；40 测试全绿 |
