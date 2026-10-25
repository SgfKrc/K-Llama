"""Require the current release scope's real artifact reports plus the complete DL/LL product bundle.

2026-10-25 裁决：**发行范围只含 Android 与 Windows，Linux 暂不发行（延后）**。
依据 `docs/边缘最小发行包审计清单-2026-09-26.md` 的 2026-09-28 修订：
「只发行 Android 与 Windows。**Linux 发行变体不做**（`.deb` 变体 / 离线 bootstrap /
Ubuntu-Debian 原生机证据）」。
恢复 Linux 发行时取消 `PLATFORMS` 与 `MANDATORY` 中的 linux 注释，并同步
`.github/workflows/release-build.yml`／`release-acceptance.yml` 的 linux 矩阵项与
acceptance 的 `--report linux=...`。Linux 诊断仍可单独跑
`scripts/package_contract/release_artifacts_gate.py --platforms linux`。
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
PRODUCT_SPEC = importlib.util.spec_from_file_location("release_product_gate", ROOT / "scripts/product_relay_gate.py")
product_gate = importlib.util.module_from_spec(PRODUCT_SPEC)
PRODUCT_SPEC.loader.exec_module(product_gate)
PLATFORMS = (
    "windows",
    # linux 暂不发行（2026-10-25 裁决；见 docs/边缘最小发行包审计清单-2026-09-26.md
    # 的 2026-09-28 修订「只发行 Android 与 Windows，Linux 发行变体不做」）。
    # 恢复 Linux 发行时取消下一行注释，并一并恢复 MANDATORY 的 linux 项、
    # 两份 workflow 的 linux 矩阵项与 acceptance 的 --report linux=...。
    # "linux",
    "android",
)
MANDATORY = {
    "windows": {f"windows.{variant}.{check}" for variant in ("cpu", "launcher")
                for check in ("version", "integrity", "contract", "revision", "dependencies", "roles", "startup")},
    # linux 暂不发行（同上 2026-10-25 裁决）：判据定义原样保留，恢复发行时取消注释。
    # "linux": {f"linux.cpu.{check}" for check in (
    #     "package.version", "package.install", "package.audit", "bundle.integrity",
    #     "bundle.contract", "bundle.revision", "runtime.dependencies", "runtime.clean", "runtime.worker")},
    "android": {f"android.{check}" for check in (
        "version", "revision", "dependencies", "signature", "roles", "startup", "native-runtime")},
}


def summarize_release(reports: dict[str, Path], bundle: Path, revision: str,
                      contract_path: Path = ROOT / "release-contract.json") -> dict:
    result = {"gate": "release-acceptance", "schema_version": 1,
              "source_revision": revision, "passed": False, "errors": [], "platforms": {}}
    try:
        if set(reports) != set(PLATFORMS):
            raise ValueError(f"exactly {','.join(PLATFORMS)} reports are required")
        contract_bytes = contract_path.read_bytes()
        contract_hash = hashlib.sha256(contract_bytes).hexdigest()
        contract = json.loads(contract_bytes)
        for platform in PLATFORMS:
            try:
                path = reports[platform]
                data = json.loads(path.read_text(encoding="utf-8-sig"))
                if (not isinstance(data, dict) or type(data.get("schema_version")) is not int
                    or data["schema_version"] != 2 or data.get("gate") != "release-artifacts"
                    or data.get("passed") is not True or data.get("status") != "passed"):
                    raise ValueError("artifact gate failed/unverified or report schema invalid")
                if (data.get("source_revision") != revision or data.get("required_revision") != revision
                    or data.get("revision_verified") is not True):
                    raise ValueError("artifact source revision was not authenticated against required revision")
                if data.get("requested_platforms") != [platform]:
                    raise ValueError("artifact report does not match the required single platform")
                if data.get("contract_sha256") != contract_hash or data.get("contract_version") != contract["version"]:
                    raise ValueError("artifact report used another release contract")
                observations = data.get("reports")
                if not isinstance(observations, list) or len(observations) != 1:
                    raise ValueError("missing or ambiguous platform observation")
                observation = observations[0]
                if (observation.get("platform") != platform or observation.get("passed") is not True
                    or observation.get("status") != "passed"):
                    raise ValueError("platform was not actually verified")
                items = observation.get("items")
                if not isinstance(items, list) or not items:
                    raise ValueError("empty platform checks")
                names = []
                for item in items:
                    if not isinstance(item, dict) or item.get("passed") is not True or item.get("status") != "passed":
                        raise ValueError("a required platform check failed or was not executed")
                    name = item.get("check")
                    if not isinstance(name, str) or not name:
                        raise ValueError("invalid platform check name")
                    if name.endswith(".revision"):
                        evidence = item.get("evidence")
                        if (not isinstance(evidence, dict) or evidence.get("source_revision") != revision
                            or evidence.get("revision_verified") is not True):
                            raise ValueError("one artifact revision was not authenticated or differs")
                    names.append(name)
                if len(names) != len(set(names)) or not MANDATORY[platform].issubset(names):
                    raise ValueError("mandatory platform checks missing or duplicated")
                result["platforms"][platform] = {"passed": True, "checks": names,
                    "report_sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
            except (OSError, ValueError, KeyError, TypeError, AttributeError) as exc:
                result["platforms"][platform] = {"passed": False, "error": str(exc)}
                result["errors"].append(f"{platform}: {exc}")
        result["product"] = product_gate.validate_bundle(bundle, revision)
        if result["product"]["passed"] is not True:
            result["errors"].append("complete DL/LL product evidence did not pass")
        result["passed"] = not result["errors"]
    except (OSError, ValueError, KeyError, TypeError) as exc:
        result["errors"].append(str(exc))
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", action="append", default=[], metavar="PLATFORM=PATH")
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--require-revision", required=True)
    parser.add_argument("--json", type=Path, required=True)
    args = parser.parse_args(argv)
    reports = {}
    for entry in args.report:
        name, separator, path = entry.partition("=")
        if not separator or not path or name not in PLATFORMS or name in reports:
            parser.error(f"--report must uniquely name {','.join(PLATFORMS)} as PLATFORM=PATH")
        reports[name] = Path(path)
    result = summarize_release(reports, args.bundle, args.require_revision)
    rendered = json.dumps(result, ensure_ascii=False, indent=2) + "\n"
    args.json.parent.mkdir(parents=True, exist_ok=True)
    args.json.write_text(rendered, encoding="utf-8")
    print(rendered)
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
