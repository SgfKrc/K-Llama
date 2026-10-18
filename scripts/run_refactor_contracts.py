"""Run fixed refactor contracts and staged bootstrap/catalog regressions."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

try:
    from scripts.summarize_test_channels import evaluate_run
except ModuleNotFoundError as exc:
    if exc.name != "scripts":
        raise
    from summarize_test_channels import evaluate_run

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "tests" / "fixtures" / "refactor_suite_v1.json"


def suite_targets() -> list[str]:
    suite = json.loads(MANIFEST.read_text(encoding="utf-8"))
    if suite.get("schema_version") != 1 or not suite.get("groups"):
        raise ValueError("invalid R0 suite manifest")
    targets = [target for group in suite["groups"].values() for target in group]
    if len(targets) != len(set(targets)):
        raise ValueError("duplicate R0 test target")
    for target in targets:
        path = (ROOT / target.split("::", 1)[0]).resolve()
        if not path.is_relative_to(ROOT / "tests") or not path.is_file():
            raise ValueError(f"invalid R0 test target: {target}")
    return targets


def bootstrap_targets() -> list[str]:
    manifest = json.loads((MANIFEST.parent / "refactor_bootstrap_suite_v1.json").read_text(encoding="utf-8"))
    if manifest.get("schema_version") != 1 or manifest.get("base_suite") != MANIFEST.name:
        raise ValueError("invalid R1 bootstrap suite manifest")
    extra = manifest["additional_targets"]
    targets = suite_targets() + extra
    if len(targets) != len(set(targets)):
        raise ValueError("duplicate bootstrap test target")
    for target in extra:
        path = (ROOT / target.split("::", 1)[0]).resolve()
        if not path.is_relative_to(ROOT / "tests") or not path.is_file():
            raise ValueError(f"invalid bootstrap test target: {target}")
    return targets


def catalog_targets() -> list[str]:
    manifest = json.loads((MANIFEST.parent / "refactor_catalog_suite_v1.json").read_text(encoding="utf-8"))
    if manifest.get("schema_version") != 1 or manifest.get("base_suite") != "refactor_bootstrap_suite_v1.json":
        raise ValueError("invalid R2 catalog suite manifest")
    targets = bootstrap_targets() + manifest["additional_targets"]
    if len(targets) != len(set(targets)):
        raise ValueError("duplicate catalog test target")
    for target in targets:
        path = (ROOT / target.split("::", 1)[0]).resolve()
        if not path.is_relative_to(ROOT / "tests") or not path.is_file():
            raise ValueError(f"invalid catalog test target: {target}")
    return targets


def deployment_targets() -> list[str]:
    manifest = json.loads((MANIFEST.parent / "refactor_deployment_suite_v1.json").read_text(encoding="utf-8"))
    if manifest.get("schema_version") != 1 or manifest.get("base_suite") != "refactor_catalog_suite_v1.json":
        raise ValueError("invalid R3 deployment suite manifest")
    targets = catalog_targets() + manifest["additional_targets"]
    if len(targets) != len(set(targets)):
        raise ValueError("duplicate deployment test target")
    for target in targets:
        path = (ROOT / target.split("::", 1)[0]).resolve()
        if not path.is_relative_to(ROOT / "tests") or not path.is_file():
            raise ValueError(f"invalid deployment test target: {target}")
    return targets


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("contract", "bootstrap", "catalog", "deployment"), default="contract")
    parser.add_argument("--workers", type=int, choices=range(5), default=0)
    parser.add_argument("--seed", type=int, default=20261018)
    parser.add_argument("--junitxml", type=Path, default=ROOT / "test-results" / "refactor-contracts.xml")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    args.junitxml = (ROOT / args.junitxml).resolve()
    targets = {"contract": suite_targets, "bootstrap": bootstrap_targets, "catalog": catalog_targets, "deployment": deployment_targets}[args.stage]()
    command = [sys.executable, "-B", "-m", "pytest", *targets, "-n", str(args.workers),
               "--maxfail=0", "--tb=short", "-qq", "--qlh-order-seed", str(args.seed),
               "--junitxml", str(args.junitxml)]
    if args.workers:
        command += ["--dist", "loadscope"]
    if args.dry_run:
        print(json.dumps(command, ensure_ascii=False))
        return 0
    label = {"contract": "R0", "bootstrap": "R1 bootstrap", "catalog": "R2 catalog", "deployment": "R3 deployment"}[args.stage]
    print(f"{label} suite: {len(targets)} targets, workers={args.workers}, seed={args.seed}", flush=True)
    args.junitxml.parent.mkdir(parents=True, exist_ok=True)
    result = subprocess.run(command, cwd=ROOT, env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"}, check=False)
    evidence = evaluate_run({"channel": "refactor", "returncode": result.returncode,
                             "junitxml": args.junitxml.name}, args.junitxml.parent)
    print(json.dumps({"status": evidence["status"], "counts": evidence["counts"],
                      "errors": evidence["errors"]}, ensure_ascii=False))
    return result.returncode or (0 if evidence["status"] == "PASS" else 1)


if __name__ == "__main__":
    raise SystemExit(main())
