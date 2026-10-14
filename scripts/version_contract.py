#!/usr/bin/env python3
"""Generate and verify release-version mirrors from one canonical contract."""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
CONTRACT_PATH = ROOT / "release-contract.json"
VERSION_RE = re.compile(r"^[0-9]+(?:\.[0-9]+){2,3}(?:[-+][0-9A-Za-z.-]+)?$")


def load_contract() -> dict[str, Any]:
    data = json.loads(CONTRACT_PATH.read_text(encoding="utf-8"))
    version = data.get("version") if isinstance(data, dict) else None
    profile = data.get("profile") if isinstance(data, dict) else None
    if data.get("schema_version") != 1 or not isinstance(version, dict):
        raise ValueError("unsupported release contract schema")
    for key in ("product", "launcher", "inference_service_contract"):
        value = str(version.get(key, ""))
        if not VERSION_RE.fullmatch(value):
            raise ValueError(f"invalid {key} version: {value!r}")
    if not isinstance(version.get("android_code"), int) or version["android_code"] <= 0:
        raise ValueError("android_code must be a positive integer")
    if not isinstance(profile, dict) or profile.get("default_node_role") not in {
        "master", "client", "auto",
    }:
        raise ValueError("invalid release profile")
    return data


def generated_files(contract: dict[str, Any]) -> dict[Path, str]:
    version = contract["version"]
    product = str(version["product"])
    launcher = str(version["launcher"])
    android_code = int(version["android_code"])
    return {
        ROOT / "packaging" / "packaging" / "version.txt": product + "\n",
        ROOT / "packaging" / "packaging" / "launcher-version.txt": launcher + "\n",
        ROOT / "packaging" / "packaging" / "release-versions.issinc": (
            "; Generated from release-contract.json; do not edit manually.\n"
            f'#define QlhProductVersion "{product}"\n'
            f'#define QlhLauncherVersion "{launcher}"\n'
        ),
        ROOT / "android" / "version.properties": (
            "# Generated from release-contract.json; do not edit manually.\n"
            f"productVersion={product}\n"
            f"versionCode={android_code}\n"
        ),
    }


def write_generated(files: dict[Path, str]) -> None:
    for path, content in files.items():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8", newline="\n")


def check_generated(files: dict[Path, str]) -> list[str]:
    errors: list[str] = []
    for path, expected in files.items():
        try:
            actual = path.read_text(encoding="utf-8")
        except OSError as exc:
            errors.append(f"missing generated file {path.relative_to(ROOT)}: {exc}")
            continue
        if actual != expected:
            errors.append(f"generated file is stale: {path.relative_to(ROOT)}")
    return errors


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--check", action="store_true")
    action.add_argument("--write", action="store_true")
    action.add_argument("--json", action="store_true", dest="as_json")
    args = parser.parse_args(argv)

    try:
        contract = load_contract()
        files = generated_files(contract)
    except (OSError, json.JSONDecodeError, ValueError) as exc:
        print(f"release contract error: {exc}", file=sys.stderr)
        return 2
    if args.write:
        write_generated(files)
        return 0
    if args.as_json:
        print(json.dumps(contract, ensure_ascii=False, sort_keys=True))
        return 0
    errors = check_generated(files)
    if errors:
        print("\n".join(errors), file=sys.stderr)
        return 1
    print("release contract mirrors are consistent")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
