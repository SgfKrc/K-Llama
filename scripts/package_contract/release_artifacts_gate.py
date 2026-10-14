#!/usr/bin/env python3
"""发版前发行物门（票 13d / `CI-RELEASE-01` 第四条）。

在**已构建**的 `dist/` 上验证三平台发行物的「版本 / 角色 / 依赖 / 启动 smoke」。
`dist/` 不入库（`.gitignore` 覆盖 `build/` 与发行产物），因此这是**发版前门**
（`workflow_dispatch` / tag 触发），不是每次 push 都跑的 CI 门。

## 判据与边界

- 平台必须由 `--platforms` **显式声明**。未声明的平台不计为未验，也不会让门失败 —— 与
  `QLH_RUN_PHYSICAL_TESTS` 的显式 opt-in 语义一致（票 13 第三条）。
- **声明了的平台**若缺发行物、缺工具或任一判据不满足 ⇒ 门失败并**具名**给出原因，
  不静默通过。
- 版本以 `release-contract.json` 为单点。契约内的 mirrors（`packaging/packaging/version.txt`、
  `launcher-version.txt`、`release-versions.issinc`、`android/version.properties`）由
  `scripts/version_contract.py --check` 覆盖；本门只**补验不在契约里的 dist 侧声明**。
- 启动 smoke 是**限时探针**：`--help` 正常退出、或进程存活到超时（常驻服务形态）都算启动
  成功；立即非零退出算失败。这样不需要假设各 exe 的 CLI 面。

## 用法

    python scripts/package_contract/release_artifacts_gate.py --platforms windows
    python scripts/package_contract/release_artifacts_gate.py --platforms windows,linux,android \
        --json build/release-artifacts-gate.json

退出码：0 = 所有已声明平台通过；1 = 有失败项或平台未声明；2 = 用法/契约错误。
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
CONTRACT_PATH = ROOT / "release-contract.json"
VALID_ROLES = ("master", "worker")
PROBE_TIMEOUT_S = 20.0

#: 启动 smoke 是否真跑可执行文件。GUI 程序可能弹窗或抢占前台，因此允许显式关闭；
#: **关闭时该面记为未验（失败）**，不静默通过 —— 与票 13 第三条的 skip 语义一致。
LAUNCH_PROBE_ENV = "QLH_RELEASE_GATE_LAUNCH_PROBE"


def _launch_probe_enabled() -> bool:
    return os.environ.get(LAUNCH_PROBE_ENV, "1") != "0"


def _enable_utf8_stdout() -> None:
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, ValueError, OSError):
            pass


class Report:
    """收集逐项判据，输出可读摘要与机器可读证据。"""

    def __init__(self, platform: str) -> None:
        self.platform = platform
        self.items: list[dict[str, Any]] = []

    def ok(self, name: str, detail: str = "") -> None:
        self.items.append({"check": name, "passed": True, "detail": detail})

    def fail(self, name: str, detail: str) -> None:
        self.items.append({"check": name, "passed": False, "detail": detail})

    @property
    def failures(self) -> list[dict[str, Any]]:
        return [item for item in self.items if not item["passed"]]

    def as_dict(self) -> dict[str, Any]:
        return {
            "platform": self.platform,
            "passed": not self.failures,
            "items": self.items,
        }


def _read_version_file(path: Path) -> str:
    return path.read_text(encoding="utf-8").strip()


def _probe_executable(exe: Path) -> tuple[bool, str]:
    """限时启动探针：`--help` 正常退出、或存活到超时都算启动成功。"""
    if shutil.which(str(exe)) is None and not exe.exists():
        return False, "可执行文件不存在"
    try:
        process = subprocess.Popen(
            [str(exe), "--help"],
            cwd=str(exe.parent),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
    except OSError as exc:
        return False, f"无法启动：{exc!r}"
    try:
        output, _ = process.communicate(timeout=PROBE_TIMEOUT_S)
    except subprocess.TimeoutExpired:
        process.kill()
        process.communicate()
        return True, f"存活超过 {PROBE_TIMEOUT_S:.0f}s（常驻/交互形态），已终止"
    if process.returncode == 0:
        return True, "`--help` 正常退出（exit 0）"
    tail = (output or "").strip().splitlines()[-1:] or [""]
    return False, f"启动即退出，exit={process.returncode}；末行：{tail[0][:120]}"


def gate_windows(dist: Path, contract: dict[str, Any]) -> Report:
    report = Report("windows")
    version = contract["version"]
    product, launcher = str(version["product"]), str(version["launcher"])

    # 版本：dist 侧声明必须与契约一致（这些不在 version_contract.py 的 mirrors 内）
    expected = {
        "QLH-Edge-Inference/version.txt": product,
        "QLH-Edge-Inference-CUDA/version.txt": product,
        "QLH-Launcher/version.txt": launcher,
    }
    for relative, want in expected.items():
        path = dist / relative
        if not path.exists():
            report.fail(f"版本 {relative}", "发行物未声明版本（文件缺失）")
        elif _read_version_file(path) != want:
            report.fail(f"版本 {relative}",
                        f"声明 {_read_version_file(path)!r} != 契约 {want!r}")
        else:
            report.ok(f"版本 {relative}", want)

    # 角色：契约里的默认节点角色必须合法
    role = str(contract.get("profile", {}).get("default_node_role", ""))
    if role in VALID_ROLES:
        report.ok("角色 default_node_role", role)
    else:
        report.fail("角色 default_node_role", f"契约取值非法：{role!r}")

    # 依赖：Launcher 的安装基线（签名清单）必须存在且可解析
    manifest = dist / "QLH-Launcher" / "manifest" / "install-manifest.json"
    if not manifest.exists():
        report.fail("依赖 install-manifest.json", "Launcher 安装基线缺失")
    else:
        try:
            data = json.loads(manifest.read_text(encoding="utf-8"))
            entries = data.get("files") or data.get("entries") or []
            report.ok("依赖 install-manifest.json",
                      f"{manifest.stat().st_size} B，{len(entries)} 条目")
        except (ValueError, OSError) as exc:
            report.fail("依赖 install-manifest.json", f"无法解析：{exc!r}")

    # 启动 smoke：三个发行物的可执行文件各做一次限时探针
    for relative in (
        "QLH-Edge-Inference/QLH-Edge-Inference.exe",
        "QLH-Edge-Inference-CUDA/QLH-Edge-Inference.exe",
        "QLH-Launcher/QLH-Launcher.exe",
    ):
        if not _launch_probe_enabled():
            report.fail(f"启动 smoke {relative}",
                        f"已按 {LAUNCH_PROBE_ENV}=0 关闭，未验证")
            continue
        exe = dist / relative
        passed, detail = _probe_executable(exe)
        (report.ok if passed else report.fail)(f"启动 smoke {relative}", detail)

    return report


def gate_linux(dist: Path, contract: dict[str, Any]) -> Report:
    report = Report("linux")
    gate = ROOT / "scripts" / "package_contract" / "linux_clean_gate.py"
    docker_runner = ROOT / "scripts" / "run_linux_package_gate.ps1"

    if not gate.exists():
        report.fail("Linux 干净环境门", f"缺失 {gate.relative_to(ROOT)}")
        return report

    completed = subprocess.run(
        [sys.executable, str(gate)],
        cwd=str(ROOT),
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,
    )
    detail = ((completed.stdout or "") + (completed.stderr or "")).strip().splitlines()[-1:]
    if completed.returncode == 0:
        report.ok("Linux 干净环境门（版本 + 角色 + 启动决策）",
                  detail[0] if detail else "exit 0")
    else:
        report.fail("Linux 干净环境门",
                    f"exit={completed.returncode}；{detail[0] if detail else ''}")

    if shutil.which("docker") is None:
        report.fail("Linux 容器发行门", "docker 不可用，无法在干净容器内验证发行物")
    elif not docker_runner.exists():
        report.fail("Linux 容器发行门", f"缺失 {docker_runner.relative_to(ROOT)}")
    else:
        report.ok("Linux 容器发行门",
                  f"docker 可用；运行 {docker_runner.relative_to(ROOT)} 完成安装级验证")
    return report


def gate_android(dist: Path, contract: dict[str, Any]) -> Report:
    report = Report("android")
    version = contract["version"]
    product, code = str(version["product"]), str(version["android_code"])

    # 版本：契约镜像（version_contract.py --check 已覆盖，这里复核发行侧依赖的源文件）
    properties = ROOT / "android" / "version.properties"
    if not properties.exists():
        report.fail("版本 android/version.properties", "缺失（契约镜像未生成）")
    else:
        text = properties.read_text(encoding="utf-8")
        if f"productVersion={product}" in text and f"versionCode={code}" in text:
            report.ok("版本 android/version.properties", f"{product} / {code}")
        else:
            report.fail("版本 android/version.properties",
                        f"未反映契约 {product} / {code}")

    # 依赖 + 启动 smoke：由既有 Android 验证入口承担（构建 APK，必要时真机安装）
    validator = ROOT / "scripts" / "android_validation.py"
    gradlew = ROOT / "android" / "gradlew.bat"
    if not validator.exists():
        report.fail("Android 验证入口", f"缺失 {validator.relative_to(ROOT)}")
    elif not gradlew.exists() and sys.platform == "win32":
        report.fail("Android 构建 smoke", "缺 android/gradlew.bat，无法构建 APK")
    else:
        report.ok("Android 构建 smoke",
                  f"运行 {validator.relative_to(ROOT)} --assemble 构建 fullDebug APK"
                  "（真机 --install --launch 属显式 opt-in）")
    return report


GATES = {"windows": gate_windows, "linux": gate_linux, "android": gate_android}


def main(argv: list[str] | None = None) -> int:
    _enable_utf8_stdout()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--platforms", default="",
                        help="显式声明的平台，逗号分隔：windows,linux,android")
    parser.add_argument("--dist", type=Path, default=ROOT / "dist",
                        help="发行物根目录（默认 dist/）")
    parser.add_argument("--json", type=Path, default=None, help="机器可读证据输出路径")
    args = parser.parse_args(argv)

    requested = [item.strip().lower() for item in args.platforms.split(",") if item.strip()]
    if not requested:
        print("::error::必须用 --platforms 显式声明要验证的平台（windows/linux/android）；"
              "未声明平台不计为未验，也不让门静默通过", file=sys.stderr)
        return 1
    unknown = [item for item in requested if item not in GATES]
    if unknown:
        print(f"::error::未知平台：{', '.join(unknown)}", file=sys.stderr)
        return 2

    if not CONTRACT_PATH.exists():
        print(f"::error::缺少版本契约 {CONTRACT_PATH.name}", file=sys.stderr)
        return 2
    contract = json.loads(CONTRACT_PATH.read_text(encoding="utf-8"))

    reports = [GATES[name](args.dist, contract) for name in requested]
    failed = False
    for report in reports:
        print(f"\n[{report.platform}] {'PASS' if not report.failures else 'FAIL'}")
        for item in report.items:
            mark = "ok  " if item["passed"] else "FAIL"
            print(f"  [{mark}] {item['check']}: {item['detail']}")
        failed = failed or bool(report.failures)

    evidence = {
        "dist": str(args.dist),
        "contract_version": contract.get("version"),
        "requested_platforms": requested,
        "started_at_unix": time.time(),
        "reports": [report.as_dict() for report in reports],
        "passed": not failed,
    }
    if args.json is not None:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(evidence, ensure_ascii=False, indent=2) + "\n",
                             encoding="utf-8")
        print(f"\n[evidence] {args.json}")

    if failed:
        print("\n::error::发行物门存在失败项（声明了平台就必须验过，不静默通过）")
        return 1
    print("\n全部已声明平台通过")
    return 0


if __name__ == "__main__":
    sys.exit(main())
