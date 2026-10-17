# 发行物门（RELEASE-ARTIFACTS）

> 状态：现行；三平台真实发行验收仍有缺口。
>
> 更新日期：2026-10-16

本门检查已构建的最终发行物。源码合同、脚本存在、`--help` 返回 0 和进程存活都不能替代实际启动就绪。任何声明平台出现 failed 或 unverified，CLI 都返回非零。

## 入口与范围

```powershell
.venv-test/Scripts/python.exe -B scripts/package_contract/release_artifacts_gate.py `
  --platforms windows,linux,android --dist dist `
  --require-revision <完整待发行Git-SHA> `
  --work-dir build/release-probes --json build/release-artifacts.json
```

默认 Windows 范围为 CPU 应用和独立 Launcher，Linux 为 CPU `.deb`。票 15d 的 CUDA 包继续延后，需要时显式传 `--windows-variants cpu,cuda,launcher` 或 `--linux-variants cpu,cuda`。单个平台诊断可只声明对应平台；**整票发行验收**由总门强制三个平台全部具备证据，不能用缩小声明范围通过。

`QLH_RELEASE_GATE_LAUNCH_PROBE=0` 会关闭实际启动，并把对应面记为 unverified，整体不通过。该开关用于诊断，不产生发行验收。每次探针使用唯一目录保存日志，不覆盖已有记录。

## Windows

- CPU／Launcher 的 `version.txt` 与主仓 `release-contract.json` 对应版本一致。
- 用当前 packaging 固定信任库中的公钥，对安装清单做 Ed25519 签名和全文件 SHA-256 深校验。清单自身携带的公钥不成为信任来源。校验器必须实际执行且退出 0。
- 包中的发行合同和 `build-info.json` 必须纳入签名文件集合；source revision 与 `--require-revision` 一致。仅在 CLI 指定 revision 不能补出构建来源证明。
- 只复制已认证程序文件到探针目录，不读取用户模型／状态。应用在独立状态目录、动态端口启动；核对 HTTP health／ready、必需启动组件及 API 观察到的实际角色。分别测试默认角色与已确认 worker。
- 独立 Launcher 执行自身签名树验证，并实际转交已验证 CPU 应用启动，核对其子进程的就绪和角色。

Windows 子进程隐藏启动，用私有 job 管理本次进程树。HTTP listener 必须属于该 job，不能把原有后台端口当成本次成功。进程超时或退出后仍没有本次 owned ready API，验收失败。

完整包的启动 smoke 覆盖实际 API 启动依赖；它不等价于每个可选推理后端都已执行。瘦身包若外部运行时未预备，会明确 unverified。最终 native 推理仍由产品门实测及对应 profile 的运行检查证明。

## Linux

门选择唯一最终 `.deb`，要求可用的 **Linux Docker daemon** 和事先准备好的镜像。镜像配置位于 `scripts/package_contract/Dockerfile.release-gate`，可在网络可用时预备：

```powershell
docker build -t qlh-release-gate:bookworm -f scripts/package_contract/Dockerfile.release-gate .
```

实际门启动独立、禁网容器，挂载 `.deb`、门脚本、信任公钥和 canonical 合同。**不挂载主仓源码或开发环境**。容器执行真实 dpkg 安装、状态／audit、已安装文件的签名与摘要校验、版本／源码身份及 profile 必需模块导入，然后分别启动默认角色和确认 worker。结果绑定本次 nonce、包 SHA、variant；缺项、旧报告和重复项均拒绝。最后只移除本次唯一容器。

Docker CLI 可执行不代表 daemon 已启动；源码 `linux_clean_gate.py` 的结果不能作为安装验收。

## Android

读取实际 release APK 的嵌入版本／package，拒绝 debuggable 包；核对 arm64 ELF64 JNI 和 C++ 共享库；调用 apksigner 验证签名，并与独立配置的发行证书 SHA-256 pin 比较。APK 中 `assets/build-info.json` 的源码身份只有在可信签名验证后才计入。

真机安装／启动要求显式 `--android-device-probe --android-serial <serial>` 和 `--android-cert-sha256 <pin>`。探针仅用于独立干净验收设备：设备已有 `com.qlh.inference` 时拒绝替换或停止，不触碰已有 SAF／模型数据。新安装后回读 APK 摘要，确认它就是本次接受的文件，再启动 Activity 并观察进程及 resumed 状态，最后移除本次安装。

Activity smoke 后必须执行现有非导出 `InferenceService` 的只读 dumpsys 探针：`--qlh-release-probe <32位小写hex nonce>`。主机在请求前后读取 PID，校验本次 nonce、进程版本、时间窗口、真实 worker 生命周期及同一模型锁内读取的 JNI 后端信息。空后端、stub、角色与连接矛盾、重启、超时和默认分布式加载整模均失败。新装发行包应呈现 distributed/client、worker STOPPED 且没有加载模型；已有开发设备的 READY worker 可单独诊断，不能代替新装发行验收。

探针不加载／卸载模型，不修改模式或凭据；JNI 查询使用有界单 worker，避免超时后持续创建线程。进程 socket 与模型映射是附加 OS 事实，拒绝或不完整时如实记录。它没有增加网络入口或 exported 组件。通过可信 APK 与运行进程的上述核对后，`android.roles`／`android.native-runtime` 才能通过；只有 Activity 或 pidof 不足以通过。

Gradle `preBuild` 接入真实构建来源生成任务，将 `assets/build-info.json` 纳入 APK。生成器从主仓与固定子模块 Git 图及实际源码字节计算身份，记录 `source_dirty` 与源码树摘要，不接受环境变量指定 revision。生成任务每次重算；所有 APK variant 在 package 动作前用 `--verify` 逐字段、逐类型核对构建前快照，原快照不刷新；缺仓库、缺子模块或并发源码变化均失败。发行门要求显式布尔 `source_dirty=false`；dirty、缺字段、null、0 和字符串均拒绝。debug 包可以用于开发真机验证，但不能充当同 revision 的可信发行包。

## CI 与总门

新增 `.github/workflows/release-build.yml` 作为最终附件的生产入口；它只允许 manual dispatch，输入完整 revision 必须等于该 run 的 `github.sha`。Windows／Linux／Android runner 标签与验收工作流一致，Windows 使用预置外部 packaging venv，Android 使用实验台预置发行 keystore 配置，不生成或替换签名密钥。

实验台需配置以下仓库变量；路径必须指向对应 self-hosted runner 已有资源，密钥字节不放入变量或仓库：

| 变量 | 用途 |
| --- | --- |
| `QLH_RELEASE_SIGNING_KEY_PATH` | 当前 packaging 信任库认可的既有 Ed25519 私钥文件路径 |
| `QLH_RELEASE_WINDOWS_VENV` | 外部已预备的 Windows packaging venv 绝对路径 |
| `QLH_RELEASE_ANDROID_KEYSTORE_PROPERTIES` | checkout 外既有 Android keystore.properties 路径；storeFile 为绝对路径 |
| `ANDROID_RELEASE_CERT_SHA256` | 独立配置的 Android 发行证书公开 pin |
| `QLH_RELEASE_PRODUCT_BUNDLE` | Linux 实验台上同 revision 的严格 DL／LL bundle.json 绝对路径及完整相对附件 |

工作流先用产品门校验已有 lab bundle，复制声明引用的完整附件，再校验复制结果；缺项或 dirty 时在构建矩阵之前失败。构建调用既有 Windows CPU／Launcher batch、Linux CPU build-deb、Android assembleRelease。`scripts/package_contract/release_build.py` 的 preflight 拒绝旧输出并保存 canonical 来源快照；stage 核对签名信任、版本、发行合同与原 build-info，只上传认证文件集合。Windows 真实构建输出来自 `packaging/dist/`。来源变化或错误签名不能通过事后重贴身份补救。

同一成功 run 上传 `qlh-final-windows`、`qlh-final-linux`、`qlh-final-android`、`qlh-product-relay-evidence`，与下游消费名称一一对应。build 阶段的静态认证不替代安装／启动；本轮仅完成接线与离线回归，尚未 dispatch、实际构建或上传。严格产品采集的运行端缺项见 [产品接力证据门](产品接力证据门-CI-RELEASE-01.md)。

`.github/workflows/release-acceptance.yml` 通过显式 dispatch 接收成功构建 run id、完整 revision 和物理 opt-in。它从同一 run 下载 `qlh-final-windows`、`qlh-final-linux`、`qlh-final-android`、`qlh-product-relay-evidence`；缺附件立即失败。实验台的 self-hosted runner 需准备平台工具、Linux 镜像，Android 需配置独立验收 serial 及可信证书 pin。

当前工作流使用 `qlh-release`／`qlh-release-android` labels；仓库未验证这些 runner 已配置。该工作流的源码接线已完成，不能把尚未配置或尚未执行的 job 记作成功。现有 Windows CPU／Launcher 及 Linux 构建入口已接入独立源码快照：构建前生成、签名前复核并原样纳入应用树。来源变化、已有签名树或已有 build-info 都会拒绝重新贴身份。Linux 无 Git 导出目录以 `SOURCE_CHECKOUT_REQUIRED` 失败，需真实主仓及固定子模块；不能用环境变量 revision 代替。旧包不能靠事后填 JSON 冒充已认证构建。

`scripts/package_contract/release_acceptance.py` 要求三个平台完整且真实执行的报告，再校验同 revision 的 [DL／LL 产品证据](产品接力证据门-CI-RELEASE-01.md)。每个必验项、发行合同摘要和各包源码身份均须一致。

## 本机复核与历史更正

2026-10-14 旧脚本曾把 Linux／Android 报成 PASS。复核发现它只确认了源码检查入口／构建入口存在，未安装最终 `.deb` 或检查实际 release APK，因此这两项 **不构成发行验收**。“三平台只有 Windows 缺一个版本文件”的旧结论撤销。

2026-10-15 用新门实际读取本机 `dist/`，整体退出 1，证据位于 `build/ci-release-20261015/local-release-artifacts.json`：

| 平台 | 实际阻断 |
| --- | --- |
| Windows CPU | 缺 version.txt、签名 install-manifest，未启动未认证程序 |
| Windows Launcher | 版本 0.1.8.2 正确；签名密钥 release-20260809 不在当前信任库，深校验拒绝；未启动 |
| Linux | 没有最终 CPU .deb；Docker CLI 存在，但 dockerDesktopLinuxEngine pipe 不存在，daemon 未运行 |
| Android | SDK 工具可找到；dist 中没有最终 release APK，未安装／未启动；角色／native 面仍有上述合同缺口 |

上述首次失败记录保留。后续复核见 `build/ci-release-followup-20261015/final-artifact-result.json`：Linux daemon 已能应答，但仍无最终 `.deb`；Windows 现有 CPU／Launcher 的失败条件未解除；Android 仍无最终 release APK，签名配置缺失。Android 角色／native 探针代码已补齐并在保数据更新的 Y700 debug 包上获得本次 nonce／PID 的真实 JNI 观测，其最终 release 验收仍待可信签名包。没有为通过门替换信任库或复用 debug 签名。

2026-10-15 后续复核已预备 Linux 验收镜像，并在禁网容器确认环境；最终 `.deb` 仍缺失。量化工具已完成真实 Windows GNU／Ninja Release 构建，补丁来源与运行依赖验证通过。最新三平台门仍退出 1，见 `build/ci-release-followup-20261015/final-artifact-retry-result.json`；具体构建修复与测试见 [票 13 续验](ci-release-followup-2026-10-15.md#构建来源与量化工具闭环)。


2026-10-16 的构建接线、Linux 禁网只读容器和当前发行物门实测见 [运行环境记录](ci-release-followup-2026-10-16.md#8-运行端回执与独立采集2026-10-16)；打包路径修复、观测合同及最终软件组合结果见 [工件身份记录](ci-release-followup-2026-10-16.md#9-工件摘要与-windows-映射身份2026-10-16)。当前 debug／dirty 设备实跑继续作为开发证据，最终签名三平台执行门保持开放。
