# TUI 环境变量注册优化方案

> 状态：**Completed（开发门，2026-08-11）**
>
> 更新日期：2026-08-11
> 适用范围：打包安装时可选注册 QLH 环境变量（Windows 用户 PATH / Linux profile.d）的方案与验收口径；干净机安装/卸载/新会话门仍属最终发布验收
>
> **文档生命周期**：Active
>
> **实施日期**：2026-08-11
>
> **适用范围**：Windows Inno Setup 与 Linux `.deb` 的可选全局命令注册。只管理 `PATH` 中的 QLH 稳定命令入口，不写入端口、模型路径、SQLite 路径、代理、集群密钥或其他配置。
>
> **关联文档**：[TUI 使用指南](../../TUI使用指南.md)、[TUI 适配与聊天页实施计划](TUI适配实施计划.md)、[打包说明](../../qlh-release/README.md)

---

## 1. 目标与边界

打包安装后，用户可以选择让新终端直接找到 `bjtu`（包括 `bjtu chat`）和 `qlh-launcher`。环境注册是可选副作用，不得成为启动、认证、模型部署或数据保留的前置条件。

- 只追加一个精确的 QLH 安装目录 PATH 段，不重排或覆盖其他条目；
- 只移除本包确认由自身追加的条目；用户已有的相同路径视为用户所有，不会在卸载时删除；
- 不把密钥或运行时配置写入环境变量，避免明文泄漏和配置漂移；
- 源码模式保持原有手工启动方式，完全不受影响。

## 2. 已实施设计

### 2.1 Windows

`packaging/setup.iss` 和 `packaging/setup-cuda.iss` 共用 `packaging/env-registration.issinc`：

- 安装向导提供默认勾选的“注册全局 `bjtu` 命令”任务；用户可取消。
- 静默安装使用 `/ENVREG=1` 强制启用或 `/ENVREG=0` 强制关闭；缺值沿用向导任务，其他值在安装开始时拒绝。
- 当前用户 `PATH` 写入 `HKCU\Environment`，路径归属记录在 `HKCU\Software\QLH\EnvironmentRegistration`。记录精确路径和 `PathOwned`，因此重装改安装目录时会先移除旧的包拥有条目，卸载也只删除包拥有条目。
- Inno 的 `ChangesEnvironment=yes` 在安装结束时通知 Explorer 重新读取注册表；已打开终端仍需新开。

### 2.2 Linux

Linux 不引入 `debconf` 依赖，也不会在默认安装时修改环境。包内 `/usr/sbin/qlh-env-register` 是唯一的状态入口：

- `QLH_ENVREG=1` 执行 `enable`，将持久状态写到 `/etc/qlh-edge-inference/env-register`，并原子生成 `/etc/profile.d/qlh.sh`；
- `QLH_ENVREG=0` 执行 `disable`；未设置时 `postinst` 只执行 `apply`，重放既有状态，不为首次默认安装创建 profile；
- profile 仅在新登录 shell 中将 `/opt/qlh-edge-inference/bin` 前置到 `PATH`，重复 source 不会产生重复段；
- 常规卸载移除包拥有的 `/etc/profile.d/qlh.sh`，但保留启用/关闭选择，方便重装；`purge` 才删除该状态。用户数据根 `/var/lib/qlh-edge-inference/data` 不在本功能范围内。

Linux 包本来就维护 `/usr/local/bin/qlh-launcher` 和 `/usr/local/bin/bjtu` 符号链接；profile 注册只是用户明确选择的、可审计的 `/opt` PATH 入口，不是这些命令可用的唯一前提。

## 3. 自动化与验证

- `tests/test_env_registration.py`：启用、重复启用、禁用、状态、profile 的 PATH 去重，以及 Windows/Linux 打包接线，`3 passed`；完整 Python 回归为 `1829 passed / 32 skipped`；
- Windows CPU/CUDA 两份 Inno 脚本均通过 `ISCC /Q /O-` 编译；
- `packaging/linux/qlh-env-register`、`build-deb.sh`、`postinst`、`postrm` 均通过 `bash -n`。

尚未执行 Windows 干净机实际安装/卸载注册表检查，也未在 Linux 真机完成 `.deb` 安装、登录 shell 与 remove/purge 行为验证。这些仍属于跨平台发布门，不能以脚本和编译证据替代。

## 4. 操作约定

```powershell
# Windows 静默安装时明确关闭 PATH 注册
QLH-Edge-Inference-Setup-vX.Y.Z.exe /VERYSILENT /ENVREG=0
```

```bash
# Linux 安装时明确开启；安装后也可随时切换
sudo env QLH_ENVREG=1 dpkg -i qlh-edge-inference-cpu_X.Y.Z_amd64.deb
sudo qlh-env-register status
sudo qlh-env-register disable
```

下一票：`L4-TUI-CHAT T9.6-R2` Windows 干净机安装/卸载/真实后端与 Linux 原生 `.deb` 构建安装；T9.6-R 的 CPU/CUDA 专用 venv、签名 deep 和 Inno 构建已完成，环境变量的真实安装、卸载和新 shell 验收继续与包级发布门合并执行。
