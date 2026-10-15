"""Read-only EX-N3 production-quality gate CLI."""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from pathlib import Path

from experiment_core.production_quality import audit_production_quality


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="experiment_quality_production_gate",
        description="Audit completed EX-N3 quality records without rerunning models.",
    )
    parser.add_argument("--plan", required=True, help="Experiment plan JSON")
    parser.add_argument("--records", required=True, help="Completed records.jsonl")
    parser.add_argument(
        "--output",
        help="Write the redacted decision receipt atomically (omitted for stdout only).",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="Read-only verification; cannot be combined with --output.",
    )
    return parser


def _write_receipt(path: Path, decision: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = (json.dumps(decision, ensure_ascii=False, sort_keys=True, indent=2) + "\n").encode("utf-8")
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent,
    )
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, path)
    finally:
        if os.path.exists(temporary_name):
            os.unlink(temporary_name)


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.check and args.output:
        print("--check cannot be combined with --output", file=sys.stderr)
        return 2
    decision = audit_production_quality(args.plan, args.records)
    if args.output:
        _write_receipt(Path(args.output).expanduser(), decision)
    print(json.dumps(decision, ensure_ascii=False, sort_keys=True, indent=2))
    return 0 if decision["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
