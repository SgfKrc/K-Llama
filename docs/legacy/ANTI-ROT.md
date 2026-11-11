# 防腐化体检（对照旧仓库）

> 状态：**已退场**（2026-11-10；体检对象是 v2 代码，已归档在 `slimarc/prototype`）
> 更新日期：2026-11-03
> 适用范围：本仓库产品代码的腐化度体检与季度自查，建议每季度或每加一个大功能后重跑；读者为维护者。基线是 `INVENTORY.md`（旧仓库 232 文件 / 129,337 行）的 §4「过度工程化清单」与 §6「与预期不符的五点」；只统计仓库根的 `.py` 与 `checks/`、`scripts/`，不含 `build/`、`video-remotion/`、第三方克隆。
>
> **本文全部数字（行数、命中清单、§2 的四条）统计的都是 v2 驱动链**，那批文件已随本次清理
> 归档在 `slimarc/prototype`，本分支不再持有。结论与判据仍可复用，但**重跑体检要以 v2.1 的树
> 为新基线**，不能拿本文的绝对值当现在的腐化度。

---

## 0. 一句话

旧仓库的墓志铭是 INVENTORY §6 的那句：**「不是功能多，是把一条 62 行的循环包在 29,752 行控制面里。」**
我们现在是**一条 64 行的循环 + 5,633 行产品代码**（2026-11-01 用 `wc -l` 重跑），比例 1:88——
**同一量级的问题、三分之一的腐化度**。没有病：`cut.py` 已随裁层方案退场（§2.1），
`main()` 这条样板膨胀已评估且暂不动（§2.2），段工件形态已清零（§2.3），只剩拓扑快照
需要分层（§2.4）。

---

## 1. 量化对比

| 指标 | 旧仓库 | 现在 | 判读 |
| --- | ---: | ---: | --- |
| 产品代码 | 129,337 行 | **5,633 行** | 1/23 |
| 核心解码循环 | 62 行 | **64 行**（v2 的 `chain_client._walk` + `forward_batch`，2026-11-01 统计；文件已退场） | 几乎相同 |
| **循环 : 产品代码** | **1 : 480** | **1 : 88** | ✅ 关键指标，好 5.5 倍 |
| legacy/兼容/backward 标记 | 439 处 | **5 处** | ✅ 好 88 倍 |
| 同功能多份实现 | scheduler 8 文件 / 21,016 行 | 0（协议 1 份、拓扑解析 1 份、GGUF 读取 1 处） | ✅ |
| 抽象套娃 | provider 3 份 / worker 12 份 / contract 10 份 | 0（无 registry / adapter / provider） | ✅ |
| 持久化 | SQLite 4 处 + 6 个仓储 | **0** | ✅ |
| 测试 : 产品代码 | 未统计 | 2,120 : 5,633 = **0.38**（`verify_*` 1,544 + `test_*` 576） | ✅ 健康 |
| 死代码 / 退场未删 | `relay_a1_legacy` 109 行、`pipeline_capacity` 21 行 | **0**（`cut.py` 已移 `_archive/tools/`；`topology.layers_from_gguf` 与 `StageEngine.cut_layer` 是裁段方案仅剩的两处代码遗留，已删；三个死脚本移 `_archive/scripts/`） | ✅ |

表中比值与倍数由本表数字相除得出（推算；旧仓库的 1:480 按 62 / 29,752 计算）。

**结论：七宗罪里六项是零，"退场未删"这一项也已清账；剩下的是配置类遗留，不是代码腐化。**

---

## 2. 遗留项

### 2.1 `cut.py` 已退场

`cut.py`（335 行）已移到 `_archive/tools/`（本机保留，不入库）。`import cut` 全仓**零命中**；
`git grep -n "cut\.py" -- '*.py'` 只剩注释与测试（`topology.provenance()` 与 `_validate()` 在读它、
`test_topology.ProvenanceTest` 用一份"按 `cut.py` 原样"的 sidecar 做格式断言），没有可执行引用。
README 的「裁段工具退场」与文件状态一致。

**保留的那一半契约**：`cut.py` 写出的 **provenance sidecar 格式**仍是活契约
（`topology.provenance()` 在读它），所以只退场生成器、不删格式。格式写在
`docs/legacy/CUT-TOOL.md`，由 `test_topology.ProvenanceTest` 钉住。

### 2.2 `main()`：已评估，暂不动

`chain_client.main()`（v2，文件已退场）是该文件最大的一个函数：369 行（该文件 1,841 行，占 **20%**；
此前的统计是 336 行 / 1,444 行，占 23%）。构成是三块直线流程：参数定义
（17 个 `add_argument`）、预检与装配（topology / shim / 内存 / VRAM / 起 worker / 加载核对）、
生成与结果报告，外加 20 余处 `print`（多数落在预检与结果报告里）。

**性质是样板膨胀，不是抽象套娃**——没有 registry / provider / adapter，就是一条直线流程。
它过了 §4 第 2 条红线（单函数 > 200 行），但**现在不动的理由**：

1. 它是 6 个入口的共同装配点（`koakuma.py` / `server.py` / `checks/*` 复用
   `spawn_worker` / `_terminate` / `mem_budget_bytes`），拆它要重跑全部验收：
   `--dry-run` 两条、两段生成、投机三档、`server` 与 TUI 各一次。
2. 拆出来的是**可读性**，不产生任何正确性收益；而当前这条链的验收成本约半小时一轮
   （要真起两段 worker）。
3. 真腐化的是"同一判定写了第七遍"，不是"一个函数长"。「不重复造轮子」那条纪律
   （`AGENTS.md` §7）针对的是前者。

**将来要动时的最小切法**（不改变副作用顺序）：

1. 参数按角色分五组抽成 `add_{chain,gen,sampling,spec,check}_args(ap)`，`main()` 只剩
   `ap = build_parser()`。17 个 `add_argument` 归组：链路 2（`--topology` `--shim`）、
   生成 5（`--gen` `--prompt-*` ×2 `--repeat` `--chat`）、采样 4（`--temperature` `--top-k`
   `--top-p` `--seed`）、投机 5（`--spec-k` `--greedy` `--dflash` `--dflash-block`
   `--cascade-mtp`）、检查 1（`--dry-run`）。
2. 三块 `print` 各抽一个 `_report_preflight()` / `_report_run()` / `_report_cleanup()`，
   `main()` 里只留调用。
3. 预检单独抽 `_preflight(stages, args) -> int | None`，返回 `None` 表示放行。
   这三步之后 `main()` 应落到 80 行以内；每一步都要能单独提交并且 `--dry-run` 仍逐字同输出。

### 2.3 段工件形态：本机已清零

`topology.toml` / `topology-rog.toml` / `topology-rog4.toml` **在本机已不存在**。
段工件方案已废弃（现行为整模 + `layer_range`），这批"段的时代的活化石"随文件删除
一并清账，不构成遗留。

`topology-2b.toml` 当时仍然是活的，但它也已改成整模 + `layer_range`（`"0:20"` / `"20:24"`）：
同一个 2B 模型（1.66 GB、24 层）做 TUI 与协议验证，两段各自从整模里选层。它的注释里
记着一个真实缺陷案例：早期的 head16 + cut-16-20 只覆盖 20 层、缺 `blk.20..23`。判据仍成立：
**看"目标模型 + 是否用 layer_range"，不是看层数。**

### 2.4 八个 topology 变体（本机）

本机有 8 个 `topology*.toml`：`topology-2b` / `topology-cross` / `topology-cross-36` /
`topology-cross-mtp` / `topology-cross-mtp-ext` / `topology-full` / `topology-local-32-8` /
`topology-example`。其中 3–4 个是活的（`cross-mtp`、`local-32-8`、`full`、`2b`），
其余是实验快照；`topology-*.toml` 全部在 `.gitignore` 里（含主机名/IP），只有
`topology-example.toml` 入库。

`[推断]` 这里要克制：拓扑快照就是实验记录，删掉会让"当时的数字怎么来的"无法复现。
**建议不是删，而是分层**：活动配置留仓库根，历史快照移到 `docs/topology-archive/`
并在 `MEASUREMENTS.md` 里引用路径。

> 2026-11-10 清理：v2 退场时这些文件一并处理——`topology-example.toml` 入库那份随 v2 删除，
> 其余 15 份是 `.gitignore` 的本机站点配置（从未入库），已移到本机 `_archive/topology-v2/`。
> §2.3 / §2.4 的清单与"哪个还活"按当时状态读。

---

## 3. 七种腐化模式的自查表

| # | 旧仓库的症状 | 现在的检查方法 | 当前结果 |
| --- | --- | --- | --- |
| 1 | 定位与实现冲突（relay 强制 loopback 却说跨机） | 找 `raise` 里带"不支持/non_loopback"的句子，问它是否与 README 定位矛盾 | 无 |
| 2 | 状态机/租约/事务/两阶段提交/代际 | `grep -n "lease\|epoch\|two_phase\|transaction"` | 0 |
| 3 | 重试/退避/心跳/故障恢复 | `grep -n "heartbeat\|backoff\|retry"` | 仅 spawn 的 connect 重试（合理） |
| 4 | 抽象套娃（provider/adapter/registry） | `grep -n "class .*Provider\|Adapter\|Registry"` | 0 |
| 5 | 同一功能多份实现 | 对"解析 GGUF / 发帧 / 校验拓扑"各数一次独立实现 | GGUF 1、发帧 1、校验 1 ✅ |
| 6 | 兼容历史分支 | `grep -c "legacy\|backward\|兼容"` | 5（其中 3 个是 topology 死配置） |
| 7 | 数据库/持久化 | `grep -n "sqlite\|journal\|persist"` | 0 ✅ |

---

## 4. 五条红线（触线即停，先拆再继续）

1. **循环 : 产品代码 > 1:150**（现在 1:88）。越过这条线说明加法都加在了循环外面，
   而外面的东西已经开始互相重复。
2. **单函数 > 200 行**（现在 `main()` 369，已评估，见 §2.2）。有理数的样板膨胀会掩盖真正的耦合。
3. **同一概念出现第二份实现**。例如第二个 GGUF 头解析、第二套拓扑校验、第二套预算估算。
   ——这是旧仓库从 1 万行走向 13 万行的主路径。
4. **新增一个"为兼容旧行为"的分支而没有删除计划**。旧仓库有 439 个，v2 时代有 5 个，
   其中已无死拓扑（`topology-2b.toml` 当时是活的，见 §2.3）。宁可写"已删除 + 迁移说明"。
5. **一个文件里同时有"设备控制"和"业务编排"**。旧仓库 8,691 行的 `scheduler_pipeline.py`
   就是 A3 供给编排吃掉了一条 62 行的接力循环。

---

## 5. 季度体检怎么跑（可复现）

```powershell
cd G:\C\PYT\qlh
# 1) 规模与比例
$py = Get-ChildItem . -Filter *.py -File
($py | ForEach-Object { (Get-Content $_ | Measure-Object -Line).Lines } | Measure-Object -Sum).Sum
# 2) 腐化标记（应各 < 10）
$py | Select-String -Pattern "legacy|backward|兼容|TODO|FIXME" | Measure-Object
# 3) 无 import 的模块 → 人工判读（入口脚本会产生误报）
$py | Where-Object { $n=$_.BaseName; -not (Select-String -Path *.py -Pattern "import $n|from $n" -Quiet) } | Select-Object Name
# 4) 无 layer_range 的拓扑 → 应为空（有命中就是段工件时代的残留，逐个人工判读）
Get-ChildItem topology*.toml | Where-Object { -not ((Get-Content $_ -Raw) -match 'layer_range') } | Select-Object Name
# 5) 抽象套娃（应为 0）
$py | Select-String -Pattern "class .*(Provider|Adapter|Registry|Factory)"
```

**其中第 3、4 条需要人工判读，不要自动断言**——这正是腐化检测的难点：

- 第 3 条当时命中 5 个：`server.py` / `stage_worker.py` / `keep_head_worker.py` /
  `test_contract.py` / `test_topology.py`——**这 5 个都已随 v2 删除**，本分支现在命中 0（`cut.py` 在
  `_archive/tools/`，见 §2.1）。判读标准仍是「有 `__main__` 且被 `.cmd` / `.ps1` 拉起的是入口」。
- 第 4 条当前命中 0（8 个拓扑全部声明 `layer_range`，见 §2.3）。

`[推断]` 这也是为什么"防腐化"不能只靠脚本：旧仓库那 439 处兼容标记，每一条单看都合理，
合起来才是屎山。脚本负责把候选找出来，判读必须是人做。

---

## 6. 一条原则

**每次加功能前先问：这是加在循环里，还是加在循环外面？**
如果在外面，再问一句：**它跟已有的外部代码，是不是同一件事？**

旧仓库每加一个功能都在外面（因为安全、可测、不用碰核心），于是外面长出了八套调度器、
十二套 worker、十套 contract，而核心循环 62 行没变。腐化不是写多了，是**同一个地方写了第七遍**。
