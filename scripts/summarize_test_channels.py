"""汇总 `scripts/run_test_channels.py` 的通道结果（CI 与本地共用）。

判据（票 13 `CI-RELEASE-01` 第三条）：**skipped 单独计数，不计作 passed**。
真机/重档属显式 opt-in：`smoke` 通道未开启时，`manifest.json` 里记 `smoke.ran=false`
与原因，因此"没跑"不会被读成"通过"。

用法：

    python scripts/summarize_test_channels.py                       # 默认 build/ci-test-artifacts
    python scripts/summarize_test_channels.py --artifacts-dir build/xxx
    python scripts/summarize_test_channels.py --artifacts-dir build/xxx --json build/summary.json

退出码：0 = 所有已跑通道 returncode 为 0；1 = 有通道失败或缺证据。
"""

from __future__ import annotations

import argparse
import json
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _enable_utf8_stdout() -> None:
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, ValueError, OSError):
            pass


def _counts(xml_path: Path) -> dict[str, int] | None:
    """从 JUnit XML 读出 passed / skipped / failed / errors（不存在时返回 None）。"""
    if not xml_path.exists():
        return None
    root = ET.parse(xml_path).getroot()
    suites = [root] if root.tag == "testsuite" else list(root)
    total = failures = errors = skipped = 0
    for suite in suites:
        total += int(suite.get("tests", 0))
        failures += int(suite.get("failures", 0))
        errors += int(suite.get("errors", 0))
        skipped += int(suite.get("skipped", 0))
    return {
        "tests": total,
        "passed": total - failures - errors - skipped,
        "skipped": skipped,
        "failed": failures,
        "errors": errors,
    }


def summarize(artifacts_dir: Path) -> tuple[int, dict[str, object]]:
    manifests = sorted(artifacts_dir.glob("*/manifest.json"))
    if not manifests:
        print("::error::未找到 manifest.json，测试通道未产生结果：%s" % artifacts_dir)
        return 1, {"artifacts_dir": str(artifacts_dir), "runs": [], "error": "no_manifest"}

    manifest_path = manifests[-1]
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    session = manifest_path.parent

    runs: list[dict[str, object]] = []
    failed_channels: list[str] = []
    for run in manifest.get("runs", []):
        counts = _counts(session / str(run.get("junitxml", "")))
        entry = {
            "channel": run.get("channel"),
            "run": run.get("run"),
            "returncode": run.get("returncode"),
            "duration_s": run.get("duration_s"),
            "junitxml": run.get("junitxml"),
            "counts": counts,
        }
        runs.append(entry)
        marker = "FAIL" if run.get("returncode") else "PASS"
        shown = counts or {}
        print("[%s] %-8s returncode=%s passed=%s skipped=%s failed=%s errors=%s duration=%ss (%s)"
              % (marker, run.get("channel"), run.get("returncode"),
                 shown.get("passed", "?"), shown.get("skipped", "?"),
                 shown.get("failed", "?"), shown.get("errors", "?"),
                 run.get("duration_s"), run.get("junitxml")))
        if run.get("returncode"):
            failed_channels.append(str(run.get("channel")))

    smoke = manifest.get("smoke") or {}
    print("[smoke] requested=%s ran=%s reason=%s"
          % (smoke.get("requested"), smoke.get("ran"), smoke.get("reason")))
    physical = manifest.get("physical") or {}
    print("[physical] opt_in_env=%s ran=%s reason=%s covers=%s"
          % (physical.get("opt_in_env"), physical.get("ran"), physical.get("reason"),
             physical.get("covers_markers")))
    if physical:
        print("[physical] external 段 -m = %s" % physical.get("external_marker_expression"))
    print("[note] skipped 单列且不计作 passed；物理设备档为显式 opt-in（physical.ran=false 不等于通过）")

    result: dict[str, object] = {
        "artifacts_dir": str(artifacts_dir),
        "session": session.name,
        "order_seed": manifest.get("order_seed"),
        "workers": manifest.get("workers"),
        "dist": manifest.get("dist"),
        "smoke": smoke,
        "physical": physical,
        "runs": runs,
        "failed_channels": failed_channels,
    }
    if failed_channels:
        print("::error::通道失败：%s" % ", ".join(failed_channels))
        return 1, result
    print("全部已跑通道通过")
    return 0, result


def main(argv: list[str] | None = None) -> int:
    _enable_utf8_stdout()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--artifacts-dir",
        type=Path,
        default=ROOT / "build" / "ci-test-artifacts",
        help="run_test_channels.py --artifacts-dir 指向的目录",
    )
    parser.add_argument("--json", type=Path, default=None, help="额外的机器可读输出路径")
    args = parser.parse_args(argv)

    code, result = summarize(args.artifacts_dir)
    if args.json is not None:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print("[summary] %s" % args.json)
    return code


if __name__ == "__main__":
    sys.exit(main())
