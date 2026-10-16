# 票 13 续验记录（2026-10-16）

状态：**三端「同 revision」已达成（开发级、可验证）；端到端三机严格 bundle 采集被两个代码级缺陷挡住；CI 发行链路的前置条件已查明（三项均未具备）。票 13 仍不能关闭。**

更新日期：2026-10-16

基准 revision `ed39e4c9e436cbf4a63f50d3e4fde6854e6fb275`（工作树干净、4 个子模块均干净）。本轮证据另存 `build/ci-release-followup-20261016/`（master 运行日志、ROG worker 运行日志、OpenAPI 快照原样保留）。上一轮见 [2026-10-15 续验](ci-release-followup-2026-10-15.md)。

## 1. 三端同 revision 已达成（可验证）

| 端 | 载体 | 证据 |
| --- | --- | --- |
| master（本机） | 主仓工作树 | `git log -1` = `ed39e4c9…`；`git status --short` 为空 |
| ROG（`rog-5060`） | `C:\Users\ROG\qlh-runtime-20261015\src` | 用 `git archive HEAD src` 覆盖后**逐文件比对：197/197 匹配、0 不一致、0 缺失**；多出的 120 项仅为 `__pycache__/*.pyc` 与 `import_time.log` |
| Y700（`android-d83bf4d0`） | `com.qlh.inference.debug` APK | 用当前 revision 重建（`assembleDebug` 55 s）：`source_revision=ed39e4c9…`、`source_tree_sha256=d4095f7dd1f0ca91dc4592ecbd9087aff998f20b725baf6c9a495e2c401c735a`、`source_dirty=true`（如实保留）；APK SHA-256 `9e41072e7d99331de3b19590a5e03aee143d2e67e10ba8fbca885253e4ad1476`；`adb install -r -d` 覆盖安装成功（保数据、未清 SAF 或身份） |

三端同时在线并被 master 认可：`/api/cluster/nodes` 返回 `count=3`，ROG 与 Y700 均完成 `hello_ack … accepted=true version=3`。

**边界**：debug 载体**不等于**最终签名发行物 ⇒「同来源」只是**开发级**具备；发行门仍要求可信签名包（见 §3）。

## 2. 端到端三机采集未完成：两个代码级阻塞

### 阻塞 A：ROG 连接反复被关闭且不自愈

现象：ROG worker 注册成功、`hello_ack accepted=true` 后 **10–45 秒**连接被关闭，master 记录
`客户端 rog-5060 已关闭连接: [WinError 10054] 远程主机强迫关闭了一个现有的连接` → 节点标记 `offline`。
ROG 侧 `uvicorn` 进程**仍然存活**（8001／8889 仍在监听），其日志在断开点静止且**无异常栈**。

已排除项：

- 不是进程崩溃退出（进程存活）；`-u` 无缓冲后仍无 traceback。
- 不是旧工件导致：把 ROG 的 `state` 改为备份目录后重启（`artifact_byte_layers=[]`，不再声明旧 2B 工件），**同样复现**。

结论：属「断线且不自愈」类缺陷，需代码级修复；本轮未改任何代码。
（该次介入已还原：`state.bak-20261016-104248` → `state`，worker 停止，ROG 回到介入前状态。）

### 阻塞 B：planner 的分段与裁层工件不对齐

- `llama_engine._find_layer_artifact()` **只认 manifest**（`source_layer_range` + `source_model_sha256`），搜索目录取 `QLH_LAYER_ARTIFACT_DIR`，未设时取模型文件所在目录。
- 但 **planner 完全不读工件目录**（`QLH_LAYER_ARTIFACT_DIR` 在 `src/` 内只有 `llama_engine` 引用）⇒ 它按容量自由分段。
- 实测（`qwen3-5-9b` + `llama_cpp` + `auto`）：规划出 `master [0,12)`，而工件库只有 `[0,4)`／`[4,24)`／`[24,32)`（`models/q35-9b-local-artifacts/`，manifest 齐全且 `source_model_sha256` 与节点上报一致）⇒ 加载失败：
  `pipeline_local_commit_failed … 没有覆盖层区间[0,12)的裁层工件（查了 QLH_LAYER_ARTIFACT_DIR 与模型目录）`
- 强制 `strategy=local_cut` 立即给出确切错误码：
  `SUPPLY_LOCAL_CUT_UNAVAILABLE: Worker android-d83bf4d0 lacks the verified source or cutter`
  —— `src/model_supply.py:950` 的 `can_cut` 要求**每个节点**都有 `inventories[node].local_cut`；
  `src/model_supply_worker.py:614` 从 `assets.local_cut_available()` 取值，而该名字在 `src/` 内**没有定义处**（`assets` 是运行时注入的模块）。

结论：要么 master 侧裁切能力被正确探测（`local_cut`），要么 planner 感知既有工件 —— **两者都不是配置项**。

## 3. CI 发行链路的前置条件（已查明，三项均未具备）

`release-acceptance.yml` 只能手动触发，其输入要求一个**已成功的 build run**（含 `qlh-final-*` 与 `qlh-product-relay-evidence` 两个 artifact），且三个平台 job 挂在 **self-hosted runner** 上。核实结果：

| 前置 | 现状 |
| --- | --- |
| 产出 `qlh-final-{windows,linux,android}` 与 `qlh-product-relay-evidence` 的 build run | **不存在**：全仓（`.github/workflows`、`scripts`）没有任何 workflow 上传这两个 artifact，只有 `release-acceptance.yml` 在消费 |
| self-hosted runner `["self-hosted","Windows","qlh-release"]`、`["self-hosted","Linux","qlh-release"]`、`["self-hosted","Linux","qlh-release-android"]` | **未注册**（现有 CI 全部跑在 `ubuntu-latest`） |
| 签名与真机 | `vars.ANDROID_RELEASE_CERT_SHA256`、`vars.QLH_RELEASE_ANDROID_SERIAL`、`QLH_SIGNING_KEY`；按既有裁决**不代持、不用临时密钥绕过** |

⇒ 「一次成功的最终 CI release run」在补齐三项之前**不可能发生**，不是"跑一次就能绿"。顺序天然是：**基础设施（self-hosted runner + 密钥/真机）→ build workflow → release-acceptance**。

## 4. 本轮净结论

- **达成**：三端同 revision（开发级）可验证；CI 发行链路缺口定性为三个明确前置。
- **未达成**：三机 DL／LL 严格 bundle 采集（被阻塞 A／B 挡住 —— 都是代码级，不是设备离线或配置问题）。
- **建议**：A、B 各立一票（A：worker 断线后不自愈；B：planner 分段与裁层工件对齐／`local_cut` 能力探测）。上一轮留下的「需同 revision 重跑严格 DL／LL bundle」，前置是先修 A／B。
