# 票 13 CI-RELEASE-01 实施与验收记录（2026-10-15）

状态：**本轮 CI／证据门改造及普通测试通道已通过回归；真实发行门未通过，三机验收未执行，整票保持开放。**

## 本轮交付

普通 CI 从固定子模块取真实 packaging 合同及 Android 文档；不递归拉模型／vendor 工件。原生 argmax 用例的头文件按现有 lock 获取、校验并应用已入库补丁。独立 docagent、退役 model-fleet 基准和外部离线资产工具的历史测试归回显式 workspace opt-in；核心质量及依赖断言继续收集，三个遗漏脚本补入白名单。四份已有 SHA 锁定 CRLF 的质量 fixture 指定一致的检出字节，没有重算质量锁。

测试通道记录预期 run 集合和生命周期，摘要同时核对退出码与真实 JUnit testcase。空 runs、缺文件、损坏 XML／JSON、失败但 exit 0、全 skip、重复使用 XML、越界路径与中断不会通过。物理授权、尝试、实际执行和通过分别记录；`--require-physical` 强制 external／smoke 每次重复都有实际物理用例通过。

[产品证据门](产品接力证据门-CI-RELEASE-01.md)要求 DL／LL 两档各有两个不同远端、连续有序完整层段、实际 completed handoff、required route、显式 fallback=false、同一模型操作／配置／精确代际、完整 OS 星状连接及每节点 proper native segment。master 发起前的 dispatch 日志不能作完成证据；新完成日志仅在远端结果校验后写入。EOS 最后一轮虽未进入可见 token steps，仍必须有完整远端完成记录。

真实 Textual 采集入口用 Pilot 创建专用会话、设置路由并发题，旁录真实 SSE，保留真实 widget 截图／正文和原生观测附件。采集器不加载整模、不指定各节点层段；已准备节点不足时在发题前失败。DL／LL 合并保持各自采集窗口，不能通过扩大联合时间窗拼接旧观测。

[发行物门](发行物门-RELEASE-ARTIFACTS.md)改为检查最终产物和真实执行。Windows 核对签名深校验、发行合同和构建 revision，运行自有隔离进程并核对就绪／实际角色；Linux 在禁网独立容器中安装最终 `.deb` 并检查已安装运行时；Android 检查 release APK 内版本／native ELF／证书签名，显式设备探针属于 opt-in。缺少产物、工具或实际执行面时保留 failed／unverified，退出非零。CUDA 包继续遵循票 15d 延后裁定，默认发行面为 CPU 与 Launcher，CUDA 为显式选项。

新增 `.github/workflows/release-acceptance.yml` 通过显式 dispatch 接收同一成功构建 run 的最终发行附件，核对构建 revision。三个平台及真实 DL／LL 证据都必需；总汇总再次核对每个必验面、版本合同和认证源码身份。普通 PR 流水线不会安装 APK 或发起物理推理。

## 验证证据

本轮工件目录：`build/ci-release-20261015/`。开始主仓 HEAD 为 `02de492f8d717b5dc6d698c7ebca723fa235af1d`；主仓起始仅 Android 子模块已有变动。未提交、未另开分支；Android 既有修改保留。

| 项目 | 当前证据 |
| --- | --- |
| 干净导出核心收集 | 最终 6,843 tests collected，退出 0；固定 gitlink 导出＋本轮源码覆盖，使用本机既有 `.venv-test`，不是远端全新 pip 安装 |
| 干净导出核心／文档／profile | 56 passed，0 skipped |
| 干净导出原生 argmax | 24 passed，0 skipped |
| 三组核心质量／依赖合同 | 18 passed，0 skipped |
| 通道与摘要专项 | 157 passed，0 skipped；其中包含实际 pytest marker 在串行／xdist 间传递的测试 |
| 产品／汇总／完成记录组合 | 初轮 250 passed；随后 EOS 与会话边界补充后的产品相关组合 112 passed（重叠，不累加） |
| 完整普通通道 | unit：6,759 passed／16 skipped；external：32 passed／0 skipped；退出均为 0，真实 JUnit 汇总 PASS |
| 物理／真实模型档 | 本轮均为 NOT_RUN，opt_in=false、attempted=false、ran=false；不计作通过 |
| 文档与 workflow | 63 份文档／314 个相对链接及双语结构通过；YAML 与内嵌 Python 本地语法通过，尚未在 GitHub 执行 |

完整普通通道使用固定顺序种子 `1973875963`、4 个 unit worker 和串行 external，证据位于 `channels/final-core/` 及 `final-core-summary.json`。合计 **6,791 passed、16 skipped**；skip 主要涉及 Windows 符号链接权限、单独子进程运行的 ABI 用例及缺少原生工件，始终单列。测试运行前后 827 份源码／配置摘要一致，生产节点身份两行摘要一致，见 `source-freeze-after.json` 与 `master-identity-after.json`。末次只读探活仍为 master＋ROG 在线，master `model_loaded=false`，没有发起推理。

定向数字仅证明对应门的实现与回归；其中的合成签名、临时 HTTP 就绪服务及产品输入不属于真实产品接力或发行物验收。干净导出收集保留独立 checkout 来源，但仍复用了本机依赖，不能称干净机全量执行。

## 保留的验收缺口

只读探活 `physical-preflight.json` 确认 master 与 `rog-5060` 两节点在线，当前意图 ready，master `model_loaded=false`。本轮未发起模型推理。Surface 离线、Y700 无电时，缺少第二个远端；两机不能替代三机容量／DL／LL 产品门。历史三机验收继续保留于 [模型意图产品验收](model-intent-ux-acceptance-2026-10-13.md)，不改写为当前 revision 的新完成证据。

本机新发行门已实际执行，退出 **1**，见 `local-release-artifacts.json` 与 `release-probe/` 原始输出：Windows CPU 包缺 `version.txt`／签名清单，Launcher 版本正确但当前信任库拒绝其 `release-20260809` 签名；Linux 没有最终 `.deb`，且 Docker CLI 虽存在，Linux daemon pipe 不存在；Android SDK 工具可找到，但没有最终 release APK。未启动未认证程序、未安装 APK。旧 Linux／Android PASS 结论已在发行文档撤销。

Android 还有可实施的代码缺口：当前 release APK 没有与本次进程绑定、可自动读取真实 worker 角色／native runtime 就绪的探针合同，因此这两面始终 unverified。补齐探针后，仍需要可信 release 签名包和独立验收设备；Activity 存活只能证明 UI smoke，不能证明 native 引擎及角色就绪。Linux 则需最终 `.deb`、可用 Linux Docker daemon 与预备镜像。

远端 Ubuntu 干净 CI、全新依赖安装、GitHub 工作流及三平台最终发行通过均尚未获得。票 13 不在这些缺口存在时关闭；票 14 的独立分支重构仍留待行为边界与证据稳定后决策。
