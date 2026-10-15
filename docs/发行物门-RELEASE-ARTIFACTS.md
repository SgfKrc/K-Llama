# 发行物门（RELEASE-ARTIFACTS）

> 状态：现行；三平台真实发行验收仍有缺口。
>
> 更新日期：2026-10-15

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

**Activity smoke 仅证明 UI 启动。** 当前 release 侧尚缺能自动读取实际 worker 角色／native runtime 就绪的合同，因此 `android.roles` 和 `android.native-runtime` 保留 unverified；不能以 pidof 或历史 logcat 抹掉缺口。后续需要 Android 端提供与接受的 APK／本次进程绑定的真实角色及 native 观测，才可关闭该面。

## CI 与总门

`.github/workflows/release-acceptance.yml` 通过显式 dispatch 接收成功构建 run id、完整 revision 和物理 opt-in。它从同一 run 下载 `qlh-final-windows`、`qlh-final-linux`、`qlh-final-android`、`qlh-product-relay-evidence`；缺附件立即失败。实验台的 self-hosted runner 需准备平台工具、Linux 镜像，Android 需配置独立验收 serial 及可信证书 pin。

当前工作流使用 `qlh-release`／`qlh-release-android` labels；仓库未验证这些 runner 已配置。该工作流的源码接线已完成，不能把尚未配置或尚未执行的 job 记作成功。新打包应把真实构建 revision 写入上述签名覆盖的 build-info；旧包不能靠事后填 JSON 冒充已认证构建。

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

这些失败记录保留；没有为通过门补写版本、替换信任库、复用 debug APK 或修改真实节点状态。
