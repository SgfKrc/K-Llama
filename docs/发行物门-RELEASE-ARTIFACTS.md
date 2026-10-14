# 发行物门（RELEASE-ARTIFACTS）

> 状态：**现行**
>
> 更新日期：2026-10-14
>
> 适用范围：发版前对**已构建**的 `dist/` 发行物做「版本 / 角色 / 依赖 / 启动 smoke」检查。不管发行物的构建过程（构建脚本在 `packaging/`），也不管测试通道（见[测试通道运行说明](测试通道运行说明.md)）。

## 为什么不是 CI 门

`dist/` 不入库，GitHub runner 上取不到发行物 ⇒ 云端跑这个门必然全部失败。因此它是**发版前门**：在打包机上、构建完成之后、上传发行物之前执行。

## 用法

```powershell
python scripts/package_contract/release_artifacts_gate.py --platforms windows,linux,android `
    --json build/release-artifacts-gate.json
```

- `--platforms` **必须显式声明**（`windows` / `linux` / `android`）。未声明的平台不计为未验，也不让门失败 —— 与 `QLH_RUN_PHYSICAL_TESTS` 相同的显式 opt-in 语义。
- **声明了的平台**若缺发行物、缺工具或任一判据不满足，门失败并**具名**给出原因。
- `QLH_RELEASE_GATE_LAUNCH_PROBE=0` 关闭启动探针（GUI 程序可能弹窗或抢占前台）；**关闭时该面记为未验**，不静默通过。

## 判据

| 面 | 判据 |
| --- | --- |
| 版本 | `dist/*/version.txt` 与 `release-contract.json` 一致。契约内的 mirrors（`packaging/packaging/version.txt`、`launcher-version.txt`、`release-versions.issinc`、`android/version.properties`）由 `scripts/version_contract.py --check` 覆盖 |
| 角色 | 契约 `profile.default_node_role` 合法（`master` / `worker`） |
| 依赖 | Launcher 的签名安装基线 `dist/QLH-Launcher/manifest/install-manifest.json` 存在且可解析（打包侧校验器为 `packaging/packaging/install_manifest.py` 的 `quick` / `full` / `deep`） |
| 启动 smoke | 三个可执行文件各做一次限时探针：`--help` 正常退出、或进程存活到超时（常驻/交互形态）都算启动成功；立即非零退出算失败 |
| Linux | 委托 `scripts/package_contract/linux_clean_gate.py`（版本 + 角色 + 干净启动决策）与 `scripts/run_linux_package_gate.ps1`（docker 容器内，`--network none --read-only`） |
| Android | `android/version.properties` 与契约一致；构建 smoke 走 `scripts/android_validation.py --assemble`，真机 `--install --launch` 属显式 opt-in |

## 实测结果（2026-10-14，本机）

```
[windows] FAIL
  [FAIL] 版本 QLH-Edge-Inference/version.txt: 发行物未声明版本（文件缺失）
  [ok  ] 版本 QLH-Edge-Inference-CUDA/version.txt: 0.1.8.3
  [ok  ] 版本 QLH-Launcher/version.txt: 0.1.8.2
  [ok  ] 角色 default_node_role: master
  [ok  ] 依赖 install-manifest.json: 200602 B，1014 条目
  [ok  ] 启动 smoke QLH-Edge-Inference/QLH-Edge-Inference.exe: 存活超过 20s（常驻/交互形态），已终止
  [ok  ] 启动 smoke QLH-Edge-Inference-CUDA/QLH-Edge-Inference.exe: `--help` 正常退出（exit 0）
  [ok  ] 启动 smoke QLH-Launcher/QLH-Launcher.exe: `--help` 正常退出（exit 0）

[linux] PASS
[android] PASS
```

三平台中只有一项缺口：非 CUDA 档 `dist/QLH-Edge-Inference/` 没有 `version.txt`（CUDA 与 Launcher 都有）。重建该发行物时补上即可；在补齐前 `windows` 平台保持 FAIL。

启动探针覆盖了两种形态：CUDA 包与 Launcher 支持 `--help` 且正常退出；非 CUDA 包是常驻/交互形态，存活到超时后由探针终止 —— 两者都判为启动成功。

Linux 平台由 `linux_clean_gate.py` 委托，其输出同时声明了拓扑不变式（`star` / `master-coordinated` / `D-to-L` / `L-to-L`）与自身局限（`not-a-final-deb-install` / `not-windows-setup` / `not-android-release-apk` / `not-real-network-or-model`）。产品 E2E 的星状连接证据要求以该不变式为准。
